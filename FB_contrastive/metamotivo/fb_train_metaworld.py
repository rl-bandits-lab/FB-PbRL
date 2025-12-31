# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import json
import time
import random
import warnings
import dataclasses
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

torch.set_float32_matmul_precision("high")

from tqdm import tqdm
import tyro
import wandb

# FB / Metamotivo
from metamotivo.buffers.buffers import OfflineReplayBuffer
from metamotivo.fb_dmc import FBAgent, FBAgentConfig
from metamotivo.nn_models import eval_mode

# ----------------------------
# DMC (optional; still supported)
# ----------------------------
try:
    from dmc_tasks import dmc
except Exception:
    dmc = None

# ----------------------------
# MetaWorld + wrappers
# ----------------------------
import metaworld
import metaworld.envs.mujoco.env_dict as _env_dict
from gym.wrappers.time_limit import TimeLimit
from rlkit.envs.wrappers import NormalizedBoxEnv
import pickle as pkl


ALL_TASKS = {
    "walker": ["walk", "run", "stand"],
    "cheetah": ["run", "run_backward", "walk", "walk_backward"],
    "quadruped": ["walk", "run", "stand", "jump"],
    # For metaworld, put the env-id directly as task_name, e.g. "button-press-topdown-v2"
    "metaworld": ["button-press-topdown-v2"],
}


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
# MetaWorld Env
# =========================================================
def make_metaworld_env(env_name: str, seed: int):
    """
    env_name examples:
      - "button-press-topdown-v2"
      - "metaworld_button-press-topdown-v2"  (we strip prefix)
    """
    env_name = env_name.replace("metaworld_", "")

    if env_name in _env_dict.ALL_V2_ENVIRONMENTS:
        env_cls = _env_dict.ALL_V2_ENVIRONMENTS[env_name]
    else:
        env_cls = _env_dict.ALL_V1_ENVIRONMENTS[env_name]

    env = env_cls()
    # make fully observable / randomized
    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True
    env.seed(seed)

    # RLKit wrappers (as in your reference code)
    env = TimeLimit(NormalizedBoxEnv(env), env.max_path_length)
    return env


# =========================================================
# MetaWorld Offline Dataset (your reference pipeline)
# =========================================================
def MetaWorld_dataset(config) -> dict:
    """
    Returns dict with keys:
      observations, actions, next_observations, rewards, terminals
    Dataset source:
      - non-human: ./dataset/MetaWorld/<env_name_without_prefix>/saved_replay_buffer_1000000_seed{0,1,2}.pkl
      - human:     ./human_feedback/<config.env>/dataset.pkl
    """
    if config.human is False:
        base_path = os.path.join(os.getcwd(), "dataset/MetaWorld/")
        env_name = config.env  # e.g. "metaworld_button-press-topdown-v2"
        base_path = os.path.join(base_path, str(env_name.replace("metaworld_", "")))

        dataset = dict()
        for seed in range(3):
            path = os.path.join(base_path, f"saved_replay_buffer_1000000_seed{seed}.pkl")
            with open(path, "rb") as f:
                load_dataset = pkl.load(f)

            # NOTE: this slicing logic matches your reference code (data_quality * 100_000)
            for key in list(load_dataset.keys()):
                load_dataset[key] = load_dataset[key][: int(config.data_quality * 100_000)]

            load_dataset["terminals"] = load_dataset["dones"][: int(config.data_quality * 100_000)]
            load_dataset.pop("dones", None)

            for key in load_dataset.keys():
                if key not in dataset:
                    dataset[key] = load_dataset[key]
                else:
                    dataset[key] = np.concatenate((dataset[key], load_dataset[key]), axis=0)

    else:
        base_path = os.path.join(os.getcwd(), "human_feedback", f"{config.task_name}", "dataset.pkl")
        with open(base_path, "rb") as f:
            dataset = pkl.load(f)

        dataset["observations"] = np.array(dataset["observations"])
        dataset["actions"] = np.array(dataset["actions"])
        dataset["next_observations"] = np.array(dataset["next_observations"])
        dataset["rewards"] = np.array(dataset["rewards"])
        dataset["terminals"] = np.array(dataset["dones"])

    N = dataset["rewards"].shape[0]

    dataset["rewards"] = dataset["rewards"].reshape(-1)
    dataset["terminals"] = dataset["terminals"].reshape(-1)

    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []

    for i in range(N):
        obs_.append(dataset["observations"][i].astype(np.float32))
        next_obs_.append(dataset["next_observations"][i].astype(np.float32))
        action_.append(dataset["actions"][i].astype(np.float32))
        reward_.append(np.float32(dataset["rewards"][i]))
        done_.append(bool(dataset["terminals"][i]))

    return {
        "observations": np.array(obs_, dtype=np.float32),
        "actions": np.array(action_, dtype=np.float32),
        "next_observations": np.array(next_obs_, dtype=np.float32),
        "rewards": np.array(reward_, dtype=np.float32),
        "terminals": np.array(done_, dtype=np.bool_),
    }


def load_metaworld_dataset_for_fb(cfg) -> dict:
    """
    Adapter: MetaWorld_dataset (standard offline RL keys)
    -> metamotivo OfflineReplayBuffer keys used by FBAgent update():
       observation, action, reward, next:{observation, terminated}
    """
    raw = MetaWorld_dataset(cfg)
    storage = {
        "observation": raw["observations"].astype(np.float32),
        "action": raw["actions"].astype(np.float32),
        "reward": raw["rewards"].astype(np.float32).reshape(-1, 1),
        "next": {
            "observation": raw["next_observations"].astype(np.float32),
            "terminated": raw["terminals"].astype(np.bool_),
        },
    }
    print(
        "Loaded MetaWorld dataset:",
        storage["observation"].shape,
        storage["action"].shape,
        storage["next"]["observation"].shape,
        storage["reward"].shape,
        storage["next"]["terminated"].shape,
    )
    return storage


# =========================================================
# Agent config
# =========================================================
def create_agent(
    domain_name="walker",
    task_name="walk",
    device="cpu",
    compile=False,
    cudagraphs=False,
    use_z_noise=False,
    z_noise_std=0.1,
    z_noise_clip=0.3,
) -> FBAgentConfig:
    if domain_name not in ["walker", "pointmass", "cheetah", "quadruped", "metaworld"]:
        raise RuntimeError(
            'FB configuration defined only for "walker", "pointmass", "cheetah", "quadruped", "metaworld"'
        )

    # Build a dummy env to read obs/action dims
    if domain_name == "metaworld":
        env = make_metaworld_env(task_name, seed=0)
        obs_dim = int(env.observation_space.shape[0])
        act_dim = int(env.action_space.shape[0])
    else:
        if dmc is None:
            raise RuntimeError("dmc_tasks.dmc import failed, but a DMC domain was requested.")
        env = dmc.make(f"{domain_name}_{task_name}")
        obs_dim = int(env.observation_spec().shape[0])
        act_dim = int(env.action_spec().shape[0])

    agent_config = FBAgentConfig()
    agent_config.model.obs_dim = obs_dim
    agent_config.model.action_dim = act_dim
    agent_config.model.device = device
    agent_config.model.norm_obs = False
    agent_config.model.seq_length = 1
    agent_config.train.batch_size = 1024

    # archi
    if domain_name in ["walker", "pointmass"]:
        agent_config.model.archi.z_dim = 100
    else:
        agent_config.model.archi.z_dim = 50

    agent_config.model.archi.b.norm = True
    agent_config.model.archi.norm_z = True
    agent_config.model.archi.b.hidden_dim = 256
    agent_config.model.archi.f.hidden_dim = 1024
    agent_config.model.archi.actor.hidden_dim = 1024
    agent_config.model.archi.f.hidden_layers = 1
    agent_config.model.archi.actor.hidden_layers = 1
    agent_config.model.archi.b.hidden_layers = 2

    # optim
    if domain_name == "pointmass":
        agent_config.train.lr_f = 1e-4
        agent_config.train.lr_b = 1e-6
        agent_config.train.lr_actor = 1e-6
    else:
        agent_config.train.lr_f = 1e-4
        agent_config.train.lr_b = 1e-4
        agent_config.train.lr_actor = 1e-4

    agent_config.train.ortho_coef = 1
    agent_config.train.train_goal_ratio = 0.5
    agent_config.train.fb_pessimism_penalty = 0
    agent_config.train.actor_pessimism_penalty = 0.5

    if domain_name == "pointmass":
        agent_config.train.discount = 0.99
    else:
        agent_config.train.discount = 0.98

    agent_config.compile = compile
    agent_config.cudagraphs = cudagraphs

    agent_config.train.use_z_noise = use_z_noise
    agent_config.train.z_noise_std = z_noise_std
    agent_config.train.z_noise_clip = z_noise_clip

    return agent_config


# =========================================================
# Train config
# =========================================================
@dataclasses.dataclass
class TrainConfig:
    # For metaworld, dataset_root is not used by MetaWorld_dataset() (it uses CWD-based paths).
    dataset_root: str = "./"

    seed: int = 0

    # Datasets:
    # - for metaworld we load from MetaWorld_dataset() using cfg.env / cfg.human / cfg.data_quality
    dataset_type: str = "rnd"  # kept for compatibility; ignored for metaworld

    # domain/task
    domain_name: str = "metaworld"
    task_name: Optional[str] = "button-press-topdown-v2"

    # MetaWorld dataset config
    env: str = "metaworld_button-press-topdown-v2"  # used by MetaWorld_dataset()
    human: bool = False
    data_quality: float = 1.0

    # training
    num_train_steps: int = 3_000_000
    log_every_updates: int = 10_000
    checkpoint_every_steps: int = 1_000_000
    work_dir: Optional[str] = None

    # eval
    num_eval_episodes: int = 10
    num_inference_samples: int = 50_000
    eval_every_steps: int = 100_000
    eval_tasks: Optional[List[str]] = None

    # misc
    compile: bool = False
    cudagraphs: bool = False
    device: str = "cuda"

    # WANDB
    use_wandb: bool = False
    wandb_ename: Optional[str] = None
    wandb_gname: Optional[str] = None
    wandb_pname: Optional[str] = "fb_train"
    wandb_name_prefix: Optional[str] = None

    use_z_noise: bool = False
    z_noise_std: float = 0.1
    z_noise_clip: float = 0.3

    def __post_init__(self):
        if self.eval_tasks is None:
            self.eval_tasks = ALL_TASKS[self.domain_name]


# =========================================================
# Workspace
# =========================================================
class Workspace:
    def __init__(self, cfg: TrainConfig, agent_cfg: FBAgentConfig) -> None:
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
        print(f"working dir: {self.work_dir}")

        set_seed_everywhere(self.cfg.seed)
        self.agent = FBAgent(**dataclasses.asdict(self.agent_cfg))

        if self.cfg.use_wandb:
            exp_name = f"fb_{self.cfg.domain_name}"
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

        with (self.work_dir / "config.json").open("w") as f:
            json.dump(dataclasses.asdict(self.cfg), f, indent=4)

    def train(self):
        self.start_time = time.time()
        self.train_offline()

    def train_offline(self) -> None:
        # -----------------------
        # Load offline dataset
        # -----------------------
        if self.cfg.domain_name == "metaworld":
            data = load_metaworld_dataset_for_fb(self.cfg)
        else:
            raise RuntimeError(
                "This script is currently wired to MetaWorld. "
                "If you also want DMC in the same file, we can add back your previous loaders."
            )

        self.replay_buffer = {"train": OfflineReplayBuffer(data, device=self.agent.device)}
        del data

        total_metrics = None
        fps_start_time = time.time()

        for t in tqdm(range(0, int(self.cfg.num_train_steps))):
            if t % self.cfg.eval_every_steps == 0:
                self.eval(t)

            metrics = self.agent.update(self.replay_buffer, t)

            if total_metrics is None:
                total_metrics = {k: metrics[k].clone() for k in metrics.keys()}
            else:
                total_metrics = {k: total_metrics[k] + metrics[k] for k in metrics.keys()}

            if t % self.cfg.log_every_updates == 0:
                m_dict = {}
                for k in sorted(list(total_metrics.keys())):
                    denom = 1 if t == 0 else self.cfg.log_every_updates
                    tmp = total_metrics[k] / denom
                    m_dict[k] = np.round(tmp.mean().item(), 6)
                m_dict["duration"] = time.time() - self.start_time
                m_dict["FPS"] = (1 if t == 0 else self.cfg.log_every_updates) / (time.time() - fps_start_time)

                if self.cfg.use_wandb:
                    wandb.log({f"train/{k}": v for k, v in m_dict.items()}, step=t)

                print(m_dict)
                total_metrics = None
                fps_start_time = time.time()

            if t % self.cfg.checkpoint_every_steps == 0 and t > 0:
                self.agent.save(str(self.work_dir / "checkpoint"))

        self.agent.save(str(self.work_dir / "checkpoint"))

    # -----------------------
    # Eval (MetaWorld)
    # -----------------------
    def eval(self, t: int):
        for task in self.cfg.eval_tasks:
            # reward_inference uses offline dataset reward (requested)
            z = self.reward_inference(task).reshape(1, -1)

            num_ep = self.cfg.num_eval_episodes
            total_reward = np.zeros((num_ep,), dtype=np.float64)

            for ep in range(num_ep):
                if self.cfg.domain_name == "metaworld":
                    eval_env = make_metaworld_env(task, seed=self.cfg.seed + ep)
                else:
                    raise RuntimeError("Eval is only implemented for metaworld in this script.")

                obs = eval_env.reset()
                done = False
                while not done:
                    with torch.no_grad(), eval_mode(self.agent._model):
                        obs_t = torch.tensor(
                            obs.reshape(1, -1),
                            device=self.agent.device,
                            dtype=torch.float32,
                        )
                        act = self.agent.act(obs=obs_t, z=z, mean=True).cpu().numpy()[0]

                    obs, r, done, _info = eval_env.step(act)
                    total_reward[ep] += float(r)

            m_dict = {"reward": float(np.mean(total_reward)), "reward#std": float(np.std(total_reward))}
            if self.cfg.use_wandb:
                wandb.log({f"{task}/{k}": v for k, v in m_dict.items()}, step=t)
            m_dict["task"] = task
            print(m_dict)

    # -----------------------
    # Reward inference: use OFFLINE DATASET reward directly
    # -----------------------
    def reward_inference(self, task) -> torch.Tensor:
        batch = self.replay_buffer["train"].sample(self.cfg.num_inference_samples)

        rewards = batch["reward"]
        # ensure shape (N, 1)
        if rewards.ndim == 1:
            rewards = rewards.unsqueeze(-1)

        z = self.agent._model.reward_inference(
            next_obs=batch["next"]["observation"],
            reward=rewards,
        )
        return z


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    cfg = tyro.cli(TrainConfig)

    warnings.warn(
        "MetaWorld version: reward_inference uses offline dataset rewards directly; no physics replay is performed."
    )

    # Default task_name/env if not specified
    if cfg.task_name is None:
        cfg.task_name = "button-press-topdown-v2"
    if cfg.env is None:
        cfg.env = f"metaworld_{cfg.task_name}"

    agent_cfg = create_agent(
        domain_name=cfg.domain_name,
        task_name=cfg.task_name,
        device=cfg.device,
        compile=cfg.compile,
        cudagraphs=cfg.cudagraphs,
        use_z_noise=cfg.use_z_noise,
        z_noise_std=cfg.z_noise_std,
        z_noise_clip=cfg.z_noise_clip,
    )

    ws = Workspace(cfg, agent_cfg=agent_cfg)
    ws.train()
