# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations
import torch
torch.set_float32_matmul_precision("high")

import numpy as np
import dataclasses
from metamotivo.buffers.buffers import OfflineReplayBuffer
from metamotivo.fb_contrastive_finetune_antmaze import FBAgent, FBAgentConfig
from metamotivo.nn_models import eval_mode
from tqdm import tqdm
import time
import random
from pathlib import Path
import json
from typing import List, Optional
import argparse
import os
import pickle
import wandb

import gym
import d4rl


# =========================================================
# Utils
# =========================================================

def set_seed_everywhere(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# =========================================================
# AntMaze OFFLINE dataset (D4RL)
# =========================================================

def load_antmaze_offline_dataset(env_name: str) -> dict:
    """
    Returns storage compatible with metamotivo.buffers.buffers.OfflineReplayBuffer:
    {
      "observation": (N, obs_dim),
      "action": (N, act_dim),
      "reward": (N, 1),
      "next": {
         "observation": (N, obs_dim),
         "terminated": (N,)
      }
    }
    """
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)

    storage = {
        "observation": dataset["observations"].astype(np.float32),
        "action": dataset["actions"].astype(np.float32),
        "reward": dataset["rewards"].astype(np.float32).reshape(-1, 1),
        "next": {
            "observation": dataset["next_observations"].astype(np.float32),
            "terminated": dataset["terminals"].astype(bool),
        },
    }
    return storage


# =========================================================
# PT Human Preference Dataset (from ./human_label/<env_name>/...)
# =========================================================

def load_pt_human_preference_dataset(
    env_name: str,
    human_label_root: str,
    seq_len: int = 100,
    num_query: int = 1000,
):
    """
    Build preference dataset in the SAME format as your existing FB finetune expects:
    {
      "next_observations":   (N, L, obs_dim),
      "next_observations_2": (N, L, obs_dim),
      "labels":              (N, 2),
    }

    File convention (you specified):
      ./human_label/<env_name>/
        - indices_num1000
        - indices_2_num1000
        - label_human
    """

    env = gym.make(env_name)
    dset = d4rl.qlearning_dataset(env)

    observations = dset["observations"].astype(np.float32)
    terminals = dset["terminals"].astype(bool)

    def extract_segment(start_idx: int):
        """
        Safe extraction: stop at terminal, then pad to length seq_len by repeating last obs.
        (Prevents cross-episode leakage without needing trajectory boundaries.)
        """
        buf = []
        for t in range(seq_len):
            idx = start_idx + t
            if idx >= len(observations):
                break
            buf.append(observations[idx])
            if terminals[idx]:
                break
        buf = np.stack(buf, axis=0)
        if buf.shape[0] < seq_len:
            pad = np.repeat(buf[-1][None], seq_len - buf.shape[0], axis=0)
            buf = np.concatenate([buf, pad], axis=0)
        return buf

    base = Path(human_label_root) / env_name
    f1 = base / f"indices_num{num_query}"
    f2 = base / f"indices_2_num{num_query}"
    fl = base / "label_human"

    if not f1.exists():
        raise FileNotFoundError(f"Missing: {f1}")
    if not f2.exists():
        raise FileNotFoundError(f"Missing: {f2}")
    if not fl.exists():
        raise FileNotFoundError(f"Missing: {fl}")

    with open(f1, "rb") as f:
        idx1 = np.array(pickle.load(f), dtype=np.int64)
    with open(f2, "rb") as f:
        idx2 = np.array(pickle.load(f), dtype=np.int64)
    with open(fl, "rb") as f:
        raw_labels = np.array(pickle.load(f))

    if not (len(idx1) == len(idx2) == len(raw_labels)):
        raise RuntimeError(
            f"Length mismatch: len(idx1)={len(idx1)}, len(idx2)={len(idx2)}, len(labels)={len(raw_labels)}"
        )

    # labels -> (N, 2)
    # Convention (typical): 0 => prefer first, 1 => prefer second, else => tie/unknown
    labels = np.zeros((len(raw_labels), 2), dtype=np.float32)
    for i, l in enumerate(raw_labels):
        if l == 0:
            labels[i] = [1.0, 0.0]
        elif l == 1:
            labels[i] = [0.0, 1.0]
        else:
            labels[i] = [0.5, 0.5]

    seg1, seg2 = [], []
    for s1, s2 in zip(idx1, idx2):
        seg1.append(extract_segment(int(s1)))
        seg2.append(extract_segment(int(s2)))

    pref_dataset = {
        "next_observations": np.stack(seg1, axis=0),
        "next_observations_2": np.stack(seg2, axis=0),
        "labels": labels,
    }

    print("[PT] Human preference dataset loaded:")
    print("  next_observations   :", pref_dataset["next_observations"].shape)
    print("  next_observations_2 :", pref_dataset["next_observations_2"].shape)
    print("  labels              :", pref_dataset["labels"].shape)

    return pref_dataset


# =========================================================
# Agent config (AntMaze)
# =========================================================

def create_agent_for_antmaze(
    antmaze_env: str,
    device: str = "cuda",
    compile: bool = False,
    cudagraphs: bool = False,
    use_contrastive: bool = False,
    use_dynamic_contrastive_z: bool = False,
    contrastive_coef: float = 1.0,
    quad_loss_coef: float = 1.0,
    q_loss_coef: float = 0.0,
    ortho_coef: float = 1.0,
):
    """
    Keep your FB finetune agent hyperparams structure,
    but set obs_dim/action_dim from gym AntMaze.
    """
    env = gym.make(antmaze_env)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent_config = FBAgentConfig()
    agent_config.model.obs_dim = obs_dim
    agent_config.model.action_dim = act_dim
    agent_config.model.device = device
    agent_config.model.norm_obs = True  # antmaze typically benefits from obs norm
    agent_config.model.seq_length = 1

    agent_config.train.batch_size = 1024
    agent_config.model.archi.z_dim = 50  # reasonable default for antmaze

    agent_config.model.archi.b.norm = True
    agent_config.model.archi.norm_z = True
    agent_config.model.archi.b.hidden_dim = 512
    agent_config.model.archi.f.hidden_dim = 512
    agent_config.model.archi.actor.hidden_dim = 512
    agent_config.model.archi.f.hidden_layers = 2
    agent_config.model.archi.actor.hidden_layers = 2
    agent_config.model.archi.b.hidden_layers = 4

    agent_config.train.lr_f = 1e-4
    agent_config.train.lr_b = 1e-4
    agent_config.train.lr_actor = 1e-4
    agent_config.train.ortho_coef = ortho_coef
    agent_config.train.train_goal_ratio = 0.5
    agent_config.train.fb_pessimism_penalty = 0
    agent_config.train.actor_pessimism_penalty = 0.5
    agent_config.train.discount = 0.99

    # finetune extras
    agent_config.train.contrastive_coef = contrastive_coef
    agent_config.train.quad_loss_coef = quad_loss_coef
    agent_config.train.q_loss_coef = q_loss_coef

    agent_config.compile = compile
    agent_config.cudagraphs = cudagraphs
    agent_config.use_contrastive = use_contrastive
    agent_config.train.use_dynamic_contrastive_z = use_dynamic_contrastive_z

    return agent_config


# =========================================================
# Configs
# =========================================================

@dataclasses.dataclass
class TrainConfig:
    seed: int = 0

    # --- AntMaze env name (D4RL) ---
    antmaze_env: str = "antmaze-medium-play-v2"

    # --- dataset ---
    dataset_type: str = "PT"  # we will use PT human preference
    human_label_root: str = "./human_label"
    num_query: int = 1000
    query_len: int = 100

    # --- training ---
    num_train_steps: int = 1_000_000
    log_every_updates: int = 10_000
    checkpoint_every_steps: int = 100_000

    # --- eval ---
    num_eval_episodes: int = 10
    eval_every_steps: int = 100_000

    # --- inference ---
    num_inference_samples: int = 50_000

    # --- runtime ---
    work_dir: Optional[str] = None
    compile: bool = False
    cudagraphs: bool = False
    device: str = "cuda"

    # --- finetune options passed into agent ---
    use_contrastive: bool = False
    use_dynamic_contrastive_z: bool = False
    contrastive_coef: float = 10.0
    quad_loss_coef: float = 0.0
    q_loss_coef: float = 0.0

    # pretrained checkpoint (folder)
    load_dir: str | None = None

    use_wandb: bool = False
    wandb_ename: Optional[str] = None
    wandb_gname: Optional[str] = None
    wandb_pname: Optional[str] = "fb_train_dmc"
    wandb_name_prefix: Optional[str] = None

    ortho_coef: float = 1.0


# =========================================================
# Workspace
# =========================================================

class Workspace:
    def __init__(self, cfg: TrainConfig, agent_cfg: FBAgentConfig):
        self.cfg = cfg
        self.agent_cfg = agent_cfg

        if self.cfg.work_dir is None:
            import string
            tmp_name = "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
            self.work_dir = Path.cwd() / "tmp_fbcpr" / tmp_name
            self.cfg.work_dir = str(self.work_dir)
        else:
            self.work_dir = Path(self.cfg.work_dir)
        self.work_dir.mkdir(exist_ok=True, parents=True)
        print("working dir:", self.work_dir)

        if self.cfg.load_dir is None:
            raise RuntimeError("Finetune requires --checkpoint (pretrained folder)")
        print(f"Loading pretrained model from {self.cfg.load_dir}")

        # NOTE: keep algorithm unchanged: use your finetune agent loader
        self.agent = FBAgent.load(self.cfg.load_dir, device=self.cfg.device, override_cfg=self.cfg)

        set_seed_everywhere(self.cfg.seed)

        # -------------------------
        # 1) Load preference dataset (PT human)
        # -------------------------
        print(f"[INFO] Loading PT human preference dataset for {self.cfg.antmaze_env}")
        self.pref_dataset = load_pt_human_preference_dataset(
            env_name=self.cfg.antmaze_env,
            human_label_root=self.cfg.human_label_root,
            seq_len=self.cfg.query_len,
            num_query=self.cfg.num_query,
        )

        # -------------------------
        # 2) Load offline dataset (D4RL)
        # -------------------------
        print(f"[INFO] Loading AntMaze offline dataset from D4RL: {self.cfg.antmaze_env}")
        data = load_antmaze_offline_dataset(self.cfg.antmaze_env)
        self.replay_buffer = {"train": OfflineReplayBuffer(data, device=self.agent.device)}
        print(f"[ReplayBuffer] Loaded transitions = {len(self.replay_buffer['train'])}")

        # AntMaze: no recon env required (and dmc not available)
        # self.agent.set_recon_env(...)  # DO NOT call for antmaze
        if self.cfg.use_wandb:
            exp_name = "fb"
            wandb_name = exp_name
            if self.cfg.wandb_name_prefix:
                wandb_name = f"{self.cfg.wandb_name_prefix}_{exp_name}"
            wandb.init(
                entity=self.cfg.wandb_ename,
                project=self.cfg.wandb_pname,
                group=self.cfg.wandb_gname,
                name=wandb_name,
                config=dataclasses.asdict(self.cfg),
            )

        with open(self.work_dir / "config.json", "w") as f:
            json.dump(dataclasses.asdict(self.cfg), f, indent=4)

    def train(self):
        self.start_time = time.time()
        self.train_offline()

    def train_offline(self):
        total_metrics = None
        fps_start_time = time.time()
        best_eval_score = -float("inf")

        for t in tqdm(range(int(self.cfg.num_train_steps))):
            if t % self.cfg.eval_every_steps == 0:
                eval_score = self.eval(t)
                if eval_score > best_eval_score:
                    best_eval_score = eval_score
                    self.agent.save(str(self.work_dir / "best"))
                    print(f"Best model saved at step {t} with eval score {eval_score:.4f}")

            metrics = self.agent.update(self.replay_buffer, t, self.pref_dataset)

            if total_metrics is None:
                total_metrics = {k: metrics[k].clone() for k in metrics.keys()}
            else:
                total_metrics = {k: total_metrics[k] + metrics[k] for k in metrics.keys()}

            if t % self.cfg.log_every_updates == 0:
                m_dict = {}
                for k in sorted(list(total_metrics.keys())):
                    tmp = total_metrics[k] / (1 if t == 0 else self.cfg.log_every_updates)
                    m_dict[k] = np.round(tmp.mean().item(), 6)
                m_dict["duration"] = time.time() - self.start_time
                m_dict["FPS"] = (1 if t == 0 else self.cfg.log_every_updates) / (time.time() - fps_start_time)
                print(m_dict)
                if self.cfg.use_wandb:
                    wandb.log({f"train/{k}": v for k, v in m_dict.items()}, step=t)
                total_metrics = None
                fps_start_time = time.time()

            if t % self.cfg.checkpoint_every_steps == 0 and t > 0:
                self.agent.save(str(self.work_dir / "checkpoint"))

        self.agent.save(str(self.work_dir / "checkpoint"))

    def eval(self, t: int) -> float:
        """
        AntMaze eval: run gym env, report return + normalized score.
        """
        # pick z
        if self.cfg.use_dynamic_contrastive_z:
            z = self.agent.z.reshape(1, -1)
        else:
            z = self.reward_inference().reshape(1, -1)

        env = gym.make(self.cfg.antmaze_env)

        returns = []
        normalized_scores = []
        lengths = []

        for ep in range(self.cfg.num_eval_episodes):
            obs = env.reset()
            if isinstance(obs, tuple):  # gymnasium compat
                obs = obs[0]
            done = False
            ep_ret = 0.0
            steps = 0

            # guard max steps if present
            max_steps = getattr(env, "_max_episode_steps", 1000)

            while (not done) and (steps < max_steps):
                with torch.no_grad(), eval_mode(self.agent._model):
                    obs_t = torch.tensor(obs, device=self.agent.device, dtype=torch.float32).unsqueeze(0)
                    act = self.agent.act(obs=obs_t, z=z, mean=True).cpu().numpy()
                    # act may be (1, act_dim) or (act_dim,)
                    if act.ndim == 2:
                        act = act[0]

                step_out = env.step(act)
                if len(step_out) == 5:
                    obs, reward, terminated, truncated, info = step_out
                    done = terminated or truncated
                else:
                    obs, reward, done, info = step_out[:4]

                ep_ret += float(reward)
                steps += 1

            returns.append(ep_ret)
            lengths.append(steps)

            # normalized score (D4RL environments support this)
            if hasattr(env, "get_normalized_score"):
                normalized_scores.append(env.get_normalized_score(ep_ret) * 100.0)

        m_dict = {
            "return": float(np.mean(returns)),
            "return#std": float(np.std(returns)),
            "len": float(np.mean(lengths)),
            "len#std": float(np.std(lengths)),
        }
        if self.cfg.use_wandb:
            wandb.log({f"{self.cfg.antmaze_env}/{k}": v for k, v in m_dict.items()}, step=t)
        if len(normalized_scores) > 0:
            m_dict["normalized_score"] = float(np.mean(normalized_scores))
            m_dict["normalized_score#std"] = float(np.std(normalized_scores))

        print(f"[EVAL step={t}] {m_dict}")
        # use normalized score if available, else raw return
        return m_dict.get("normalized_score", m_dict["return"])

    def reward_inference(self) -> torch.Tensor:
        """
        Keep your existing reward_inference interface:
        sample transitions from offline buffer and run agent._model.reward_inference(...)
        """
        num_samples = self.cfg.num_inference_samples
        batch = self.replay_buffer["train"].sample(num_samples)

        rewards = batch["reward"]  # (N, 1)
        z = self.agent._model.reward_inference(
            next_obs=batch["next"]["observation"],
            reward=torch.tensor(rewards, dtype=torch.float32, device=self.agent.device),
        )
        return z


# =========================================================
# CLI
# =========================================================

def parse_args():
    p = argparse.ArgumentParser("Finetune FBAgent on AntMaze with PT human preference dataset")

    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_ename", type=str, default=None)
    p.add_argument("--wandb_gname", type=str, default=None)
    p.add_argument("--wandb_pname", type=str, default="fb_train_dmc")
    p.add_argument("--wandb_name_prefix", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--antmaze_env", type=str, required=True, help="e.g., antmaze-medium-play-v2")
    p.add_argument("--human_label_root", type=str, default="./human_label")
    p.add_argument("--num_query", type=int, default=1000)
    p.add_argument("--query_len", type=int, default=100)

    p.add_argument("--num_train_steps", type=int, default=1_000_000)
    p.add_argument("--log_every_updates", type=int, default=10_000)
    p.add_argument("--checkpoint_every_steps", type=int, default=100_000)

    p.add_argument("--num_eval_episodes", type=int, default=15)
    p.add_argument("--eval_every_steps", type=int, default=100_000)

    p.add_argument("--num_inference_samples", type=int, default=50_000)

    p.add_argument("--work_dir", type=str, default=None)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--cudagraphs", action="store_true")

    # finetune options (keep as-is)
    p.add_argument("--use_contrastive", action="store_true")
    p.add_argument("--use_dynamic_contrastive_z", action="store_true")
    p.add_argument("--contrastive_coef", type=float, default=10.0)
    p.add_argument("--quad_loss_coef", type=float, default=0.0)
    p.add_argument("--q_loss_coef", type=float, default=0.0)
    p.add_argument("--ortho_coef", type=float, default=1.0)

    p.add_argument("--checkpoint", required=True, help="Folder produced by FBAgent.save(...)")
    args = p.parse_args()

    cfg = TrainConfig(
        seed=args.seed,
        device=args.device,
        antmaze_env=args.antmaze_env,
        human_label_root=args.human_label_root,
        num_query=args.num_query,
        query_len=args.query_len,
        num_train_steps=args.num_train_steps,
        log_every_updates=args.log_every_updates,
        checkpoint_every_steps=args.checkpoint_every_steps,
        num_eval_episodes=args.num_eval_episodes,
        eval_every_steps=args.eval_every_steps,
        num_inference_samples=args.num_inference_samples,
        work_dir=args.work_dir,
        compile=args.compile,
        cudagraphs=args.cudagraphs,
        use_contrastive=args.use_contrastive,
        use_dynamic_contrastive_z=args.use_dynamic_contrastive_z,
        contrastive_coef=args.contrastive_coef,
        quad_loss_coef=args.quad_loss_coef,
        q_loss_coef=args.q_loss_coef,
        load_dir=args.checkpoint,
        dataset_type="PT",
        use_wandb=args.use_wandb,
        wandb_ename=args.wandb_ename,
        wandb_gname=args.wandb_gname,
        wandb_pname=args.wandb_pname,
        wandb_name_prefix=args.wandb_name_prefix,
        ortho_coef=args.ortho_coef,
    )
    return cfg


if __name__ == "__main__":
    cfg = parse_args()

    agent_cfg = create_agent_for_antmaze(
        antmaze_env=cfg.antmaze_env,
        device=cfg.device,
        compile=cfg.compile,
        cudagraphs=cfg.cudagraphs,
        use_contrastive=cfg.use_contrastive,
        use_dynamic_contrastive_z=cfg.use_dynamic_contrastive_z,
        contrastive_coef=cfg.contrastive_coef,
        quad_loss_coef=cfg.quad_loss_coef,
        q_loss_coef=cfg.q_loss_coef,
        ortho_coef=cfg.ortho_coef,
    )

    ws = Workspace(cfg, agent_cfg=agent_cfg)
    ws.train()
