# ============================================================
# TD-JEPA style training script for FBFlowBC on AntMaze (D4RL)
# ============================================================

from __future__ import annotations

import copy
import time
import json
import random
import dataclasses
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
torch.set_float32_matmul_precision("high")

import gym
import d4rl
from gymnasium.spaces import Box

import tyro
from tqdm import tqdm
import wandb

# ------------------------------------------------------------
# Metamotivo imports
# ------------------------------------------------------------
from metamotivo.buffers.buffers import OfflineReplayBuffer
from metamotivo.fb_antmaze_flowbc.flow_bc.agent import FBFlowBCAgentConfig
from metamotivo.fb_antmaze_flowbc.nn_models import eval_mode


# ============================================================
# BASE CONFIG  (🔴 關鍵：td_jepa-style，一開始就填滿)
# ============================================================

BASE_CFG: Dict[str, Any] = {
    "agent": {
        "name": "FBFlowBCAgent",
        "compile": False,
        "cudagraphs": False,
        "model": {
            "name": "FBFlowBCModel",
            "device": "cuda",
            "seq_length": 1,
            "actor_encode_obs": False,
            "obs_normalizer": {
                "name": "IdentityNormalizerConfig",
            },
            "archi": {
                "L_dim": 50,
                "z_dim": 50,
                "norm_z": True,
                # Forward map F
                "f": {
                    "name": "ForwardArchi",
                    "hidden_dim": 512,
                    "hidden_layers": 2,
                },
                # Backward map B
                "b": {
                    "name": "BackwardArchi",
                    "hidden_dim": 512,
                    "hidden_layers": 4,
                    "norm": True,
                },
                # Left encoder
                "left_encoder": {
                    "name": "BackwardArchi",
                    "hidden_dim": 512,
                    "hidden_layers": 4,
                    "norm": True,
                },
                # Noise-conditioned actor
                "actor": {
                    "name": "noise_conditioned_actor",
                    "hidden_dim": 512,
                    "hidden_layers": 2,
                },
                # Vector field for Flow Matching
                "actor_vf": {
                    "hidden_dim": 512,
                    "hidden_layers": 4,
                },
            },
        },
        "train": {
            "batch_size": 1024,
            "discount": 0.99,
            "lr_f": 1e-4,
            "lr_b": 1e-4,
            "lr_actor": 1e-4,
            "lr_actor_vf": 3e-4,
            "bc_coeff": 0.3,
            "ortho_coef": 100.0,
            "train_goal_ratio": 0.5,
            "actor_pessimism_penalty": 0.5,
            "f_target_tau": 0.005,
            "b_target_tau": 0.005,
        },
    }
}


# ============================================================
# Dataset
# ============================================================

def load_antmaze_dataset(env_name: str) -> dict:
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)

    obs = dataset["observations"].astype(np.float32)
    next_obs = dataset["next_observations"].astype(np.float32)
    actions = dataset["actions"].astype(np.float32)
    rewards = dataset["rewards"].astype(np.float32).reshape(-1, 1)
    terminals = dataset["terminals"].astype(bool)

    max_steps = env._max_episode_steps
    timeout = np.zeros_like(terminals)
    step = 0
    for i in range(len(terminals)):
        step += 1
        if terminals[i] or step == max_steps:
            timeout[i] = not terminals[i]
            step = 0

    terminated = np.logical_or(terminals, timeout)

    return {
        "observation": obs,
        "action": actions,
        "reward": rewards,
        "next": {
            "observation": next_obs,
            "terminated": terminated,
        },
    }


# ============================================================
# Train config (CLI only, not agent config)
# ============================================================

@dataclasses.dataclass
class TrainConfig:
    antmaze_env: str

    seed: int = 0
    num_train_steps: int = 3_000_000
    log_every_steps: int = 10_000
    eval_every_steps: int = 100_000
    checkpoint_every_steps: int = 500_000

    num_eval_episodes: int = 15
    num_inference_samples: int = 50_000

    device: str = "cuda"
    compile: bool = False
    cudagraphs: bool = False

    use_wandb: bool = False
    wandb_project: str = "fb_flowbc_antmaze"
    wandb_name: Optional[str] = None

    ortho_coef: float = 100.0


# ============================================================
# Workspace (td_jepa-style)
# ============================================================

class Workspace:
    def __init__(self, cfg: TrainConfig, agent_cfg: FBFlowBCAgentConfig):
        self.cfg = cfg
        self.agent_cfg = agent_cfg

        self.work_dir = Path("tmp_fb_flowbc") / time.strftime("%Y%m%d_%H%M%S")
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # build env spec
        env = gym.make(cfg.antmaze_env)
        obs_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )
        action_dim = env.action_space.shape[0]

        # 🔴 td_jepa-style: agent built only from config
        self.agent = agent_cfg.build(
            obs_space=obs_space,
            action_dim=action_dim,
        )

        self.agent._model.train(True)
        set_seed(cfg.seed)

        if cfg.use_wandb:
            wandb.init(
                project=cfg.wandb_project,
                name=cfg.wandb_name,
                config=dataclasses.asdict(cfg),
            )

    def train(self):
        data = load_antmaze_dataset(self.cfg.antmaze_env)
        buffer = OfflineReplayBuffer(data, device=self.agent.device)
        self.replay = {"train": buffer}

        total_metrics = None
        t0 = time.time()

        for step in tqdm(range(self.cfg.num_train_steps)):
            if step % self.cfg.eval_every_steps == 0:
                self.eval(step)

            metrics = self.agent.update(self.replay, step)

            if total_metrics is None:
                total_metrics = {k: v.clone() for k, v in metrics.items()}
            else:
                for k in total_metrics:
                    total_metrics[k] += metrics[k]

            if step % self.cfg.log_every_steps == 0 and step > 0:
                log = {k: (v / self.cfg.log_every_steps).item()
                       for k, v in total_metrics.items()}
                log["fps"] = self.cfg.log_every_steps / (time.time() - t0)

                print(log)
                if self.cfg.use_wandb:
                    wandb.log(log, step=step)

                total_metrics = None
                t0 = time.time()

            if step % self.cfg.checkpoint_every_steps == 0 and step > 0:
                self.agent.save(str(self.work_dir / "checkpoint"))

        self.agent.save(str(self.work_dir / "checkpoint_final"))

    def eval(self, step: int):
        z = self.reward_inference().reshape(1, -1)
        env = gym.make(self.cfg.antmaze_env)

        returns = []
        scores = []

        for _ in range(self.cfg.num_eval_episodes):
            obs = env.reset()
            done = False
            ep_ret = 0.0
            steps = 0

            while not done and steps < env._max_episode_steps:
                with torch.no_grad(), eval_mode(self.agent._model):
                    obs_t = torch.tensor(obs, device=self.agent.device).float().unsqueeze(0)
                    action = self.agent.act(obs_t, z, mean=True).cpu().numpy()[0]

                obs, reward, done, *_ = env.step(action)
                ep_ret += reward
                steps += 1

            returns.append(ep_ret)
            scores.append(env.get_normalized_score(ep_ret) * 100.0)

        mean_return = float(np.mean(returns))
        std_return = float(np.std(returns))
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))

        print(
            f"[Eval {step}] "
            f"return={mean_return:.2f}±{std_return:.2f}, "
            f"score={mean_score:.2f}±{std_score:.2f}"
        )

        if self.cfg.use_wandb:
            wandb.log(
                {
                    f"{self.cfg.antmaze_env}/eval_return": mean_return,
                    f"{self.cfg.antmaze_env}/eval_return_std": std_return,
                    f"{self.cfg.antmaze_env}/eval_score": mean_score,
                    f"{self.cfg.antmaze_env}/eval_score_std": std_score,
                },
                step=step,
            )

    def reward_inference(self) -> torch.Tensor:
        batch = self.replay["train"].sample(self.cfg.num_inference_samples)
        rewards = batch["reward"]

        z = self.agent._model.reward_inference(
            next_obs=batch["next"]["observation"],
            reward=torch.tensor(rewards, device=self.agent.device),
        )
        return z


# ============================================================
# Utils
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Main
# ============================================================

def main():
    cfg = tyro.cli(TrainConfig)

    cfg_dict = copy.deepcopy(BASE_CFG)
    cfg_dict["agent"]["compile"] = cfg.compile
    cfg_dict["agent"]["cudagraphs"] = cfg.cudagraphs
    cfg_dict["agent"]["model"]["device"] = cfg.device
    cfg_dict["agent"]["train"]["ortho_coef"] = cfg.ortho_coef

    agent_cfg = FBFlowBCAgentConfig(**cfg_dict["agent"])

    ws = Workspace(cfg, agent_cfg)
    ws.train()


if __name__ == "__main__":
    main()
