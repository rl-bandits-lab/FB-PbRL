# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.
#
# This file has been modified for the paper "From Reward-Free Representations
# to Preferences: Rethinking Offline Preference-Based Reinforcement Learning", 2026.

# =========================================================
# FB + Contrastive + Preference Training on MetaWorld
# =========================================================
from __future__ import annotations
import os
import json
import time
import pickle
import random
import argparse
import dataclasses
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
torch.set_float32_matmul_precision("high")

from tqdm import tqdm
import wandb

# ----------------------------
# Metamotivo
# ----------------------------
from metamotivo.fb_contrastive_finetune_metaworld import FBAgent, FBAgentConfig
from metamotivo.buffers.buffers import OfflineReplayBuffer
from metamotivo.nn_models import eval_mode

# ----------------------------
# MetaWorld
# ----------------------------
import metaworld
import metaworld.envs.mujoco.env_dict as _env_dict
from gym.wrappers.time_limit import TimeLimit
from rlkit.envs.wrappers import NormalizedBoxEnv
import pickle as pkl

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
    if env_name in _env_dict.ALL_V2_ENVIRONMENTS:
        env_cls = _env_dict.ALL_V2_ENVIRONMENTS[env_name]
    else:
        env_cls = _env_dict.ALL_V1_ENVIRONMENTS[env_name]

    env = env_cls()
    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True
    env.seed(seed)

    env = TimeLimit(NormalizedBoxEnv(env), env.max_path_length)
    return env


# =========================================================
# MetaWorld Offline Dataset
# =========================================================
def MetaWorld_dataset(config):
    """
    Returns:
        dict with keys:
            observations: (N, obs_dim)
            actions: (N, act_dim)
            next_observations: (N, obs_dim)
            rewards: (N,)
            terminals: (N,) bool
    """

    dataset = {}

    # =========================================================
    # Offline MetaWorld dataset (synthetic reward)
    # =========================================================
    if not config.human:
        base_path = os.path.join(os.getcwd(), "dataset/MetaWorld", config.env)

        for seed in range(3):
            path = os.path.join(
                base_path, f"saved_replay_buffer_1000000_seed{seed}.pkl"
            )
            with open(path, "rb") as f:
                buf = pkl.load(f)

            N = int(config.dataset_quality * 100_000)

            for k in buf:
                buf[k] = buf[k][:N]

            buf["terminals"] = buf["dones"]
            buf.pop("dones", None)

            for k in buf:
                if k not in dataset:
                    dataset[k] = buf[k]
                else:
                    dataset[k] = np.concatenate([dataset[k], buf[k]], axis=0)

    # =========================================================
    # Human-collected dataset
    # =========================================================
    else:
        path = os.path.join(
            os.getcwd(), "human_feedback", config.env, "dataset.pkl"
        )
        with open(path, "rb") as f:
            dataset = pkl.load(f)

        dataset["observations"] = np.asarray(dataset["observations"])
        dataset["actions"] = np.asarray(dataset["actions"])
        dataset["next_observations"] = np.asarray(dataset["next_observations"])
        dataset["rewards"] = np.asarray(dataset["rewards"]).reshape(-1)
        dataset["terminals"] = np.asarray(dataset["dones"]).reshape(-1).astype(bool)
        dataset.pop("dones", None)

    # =========================================================
    # Final type normalization (IMPORTANT for FB + obtain_labels)
    # =========================================================
    return {
        "observations": dataset["observations"].astype(np.float32),
        "actions": dataset["actions"].astype(np.float32),
        "next_observations": dataset["next_observations"].astype(np.float32),
        "rewards": dataset["rewards"].astype(np.float32),
        "terminals": dataset["terminals"].astype(np.bool_),
    }


def load_metaworld_dataset_for_fb(cfg):
    raw = MetaWorld_dataset(cfg)
    return {
        "observation": raw["observations"],
        "action": raw["actions"],
        "reward": raw["rewards"].reshape(-1, 1),
        "next": {
            "observation": raw["next_observations"],
            "terminated": raw["terminals"],
        },
    }

def build_synthetic_pref_dataset(
    dataset_raw,
    num_pairs: int,
    segment_size: int,
    threshold: float,
    noise: float,
):
    """
    dataset_raw keys:
      observations, actions, next_observations, rewards, terminals
    MetaWorld trajectory length = 500
    """

    rewards = dataset_raw["rewards"]
    next_obs = dataset_raw["next_observations"]
    N, D = next_obs.shape

    traj_len = 500
    traj_total = N // traj_len

    seg1_list, seg2_list, labels = [], [], []
    n_skip = 0

    while len(labels) < num_pairs:
        # sample two trajectories
        traj_idx1 = np.random.randint(traj_total)
        traj_idx2 = np.random.randint(traj_total)

        st1 = traj_idx1 * traj_len + np.random.randint(0, traj_len - segment_size)
        st2 = traj_idx2 * traj_len + np.random.randint(0, traj_len - segment_size)

        idx_1 = [[j for j in range(st1, st1 + segment_size)]]
        idx_2 = [[j for j in range(st2, st2 + segment_size)]]

        label = obtain_labels(
            dataset_raw,
            idx_1,
            idx_2,
            segment_size=segment_size,
            threshold=threshold,
            noise=noise,
        )[0]

        if np.all(label == [0.5, 0.5]):
            n_skip += 1

        seg1_list.append(next_obs[st1 : st1 + segment_size])
        seg2_list.append(next_obs[st2 : st2 + segment_size])
        labels.append(label)

    print(f"[Pref] Synthetic skip ratio = {n_skip / num_pairs:.2%}")

    return {
        "next_observations": np.stack(seg1_list, axis=0),   # (P, L, D)
        "next_observations_2": np.stack(seg2_list, axis=0), # (P, L, D)
        "labels": np.stack(labels, axis=0),                 # (P, 2)
    }

def obtain_labels(dataset, idx_1, idx_2, segment_size=25, threshold=0.5, noise=0.0):
    idx_1 = np.array(idx_1)
    idx_2 = np.array(idx_2)
    labels = []
    reward_1 = np.sum(dataset["rewards"][idx_1], axis=1)
    reward_2 = np.sum(dataset["rewards"][idx_2], axis=1)
    labels = np.where(reward_1 < reward_2, 1, 0)
    labels = np.array([[1, 0] if i == 0 else [0, 1] for i in labels]).astype(float)
    gap = segment_size * threshold

    equal_labels = np.where(
        np.abs(reward_1 - reward_2) <= segment_size * threshold, 1, 0
    )
    labels = np.array(
        [labels[i] if equal_labels[i] == 0 else [0.5, 0.5] for i in range(len(labels))]
    )
    if noise != 0.0:
        p = noise
        for i in range(len(labels)):
            if labels[i][0] == 1:
                if random.random() < p:
                    if random.random() < 0.5:
                        labels[i][0] = 0
                        labels[i][1] = 1
                    else:
                        labels[i][0] = 0.5
                        labels[i][1] = 0.5
            elif labels[i][1] == 1:
                if random.random() < p:
                    if random.random() < 0.5:
                        labels[i][0] = 1
                        labels[i][1] = 0
                    else:
                        labels[i][0] = 0.5
                        labels[i][1] = 0.5
            else:
                if random.random() < p:
                    if random.random() < 0.5:
                        labels[i][0] = 0
                        labels[i][1] = 1
                    else:
                        labels[i][0] = 1
                        labels[i][1] = 0
    return labels

def build_preference_dataset(cfg, offline_raw):
    if cfg.human:
        return load_human_pairwise_preferences(cfg, offline_raw)
    else:
        return build_synthetic_pref_dataset(
            dataset_raw=offline_raw,
            num_pairs=cfg.num_pref_pairs,
            segment_size=cfg.segment_size,
            threshold=cfg.threshold,
            noise=cfg.noise,
        )

def label_type(y):
    if np.allclose(y, [0.5, 0.5]):
        return "tie"
    elif y[0] > y[1]:
        return "right"
    else:
        return "left"
# =========================================================
# Preference Dataset (Human Pairwise)
# =========================================================
def load_human_pairwise_preferences(cfg, offline_dataset_raw):

    path = f"./human_feedback/{cfg.env}/_Independent.txt"
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    next_obs = offline_dataset_raw["next_observations"].astype(np.float32)  # (N, D)
    N, D = next_obs.shape
    L = cfg.segment_size

    total = 0
    disagree = 0
    tie_mismatch = 0
    
    pairs = []
    with open(path, "r") as f:
        for line in f:
            i, j, lab = map(int, line.strip().split())
            # guard: segment range must be valid
            if i < 0 or j < 0 or i + L > N or j + L > N:
                continue

            if lab == 1:
                y = np.array([1.0, 0.0], dtype=np.float32)
            elif lab == 3:
                y = np.array([0.0, 1.0], dtype=np.float32)
            else:
                y = np.array([0.5, 0.5], dtype=np.float32)

            y_gt = obtain_labels(
                offline_dataset_raw,
                idx_1=[[k for k in range(i, i + L)]],
                idx_2=[[k for k in range(j, j + L)]],
                segment_size=L,
                threshold=cfg.threshold,
                noise=0.0,
            )[0]

            # ----------------------
            # compare
            # ----------------------
            t_h = label_type(y)
            t_g = label_type(y_gt)

            total += 1
            if t_h != t_g:
                if "tie" in (t_h, t_g):
                    tie_mismatch += 1
                else:
                    disagree += 1

            seg1 = next_obs[i : i + L]  # (L, D)
            seg2 = next_obs[j : j + L]  # (L, D)
            pairs.append((seg1, seg2, y))

    random.shuffle(pairs)
    pairs = pairs[: int(cfg.num_pref_pairs)]
    if len(pairs) == 0:
        raise RuntimeError("No valid preference pairs loaded (check segment_size / indices range).")

    seg1 = np.stack([p[0] for p in pairs], axis=0)  # (P, L, D)
    seg2 = np.stack([p[1] for p in pairs], axis=0)  # (P, L, D)
    labels = np.stack([p[2] for p in pairs], axis=0)  # (P, 2)

    print(
        f"[Pref] Loaded human pairwise: seg1={seg1.shape} seg2={seg2.shape} labels={labels.shape} "
        f"(num_pref_pairs={len(pairs)}, segment_size={L})"
    )

    print(
        f"[Pref GT Check] total={total}, "
        f"disagree={disagree} ({disagree/total:.2%}), "
        f"tie_mismatch={tie_mismatch} ({tie_mismatch/total:.2%}), "
        f"overall_error={(disagree+tie_mismatch)/total:.2%}"
    )

    return {
        "next_observations": seg1,
        "next_observations_2": seg2,
        "labels": labels,
    }


# =========================================================
# Agent Config
# =========================================================
def create_agent(
    env_name: str,
    device: str,
    compile: bool,
    cudagraphs: bool,
    use_contrastive: bool,
    use_dynamic_contrastive_z: bool,
    contrastive_coef: float,
    quad_loss_coef: float,
    reg_coefficient: float,
    q_loss_coef: float,
    seed: int = 0,
):
    env = make_metaworld_env(env_name, seed=seed)

    agent_config = FBAgentConfig()
    agent_config.model.obs_dim = env.observation_space.shape[0]
    agent_config.model.action_dim = env.action_space.shape[0]
    agent_config.model.device = device
    agent_config.model.norm_obs = False
    agent_config.model.seq_length = 1

    # arch
    agent_config.model.archi.z_dim = 16
    agent_config.model.archi.b.norm = True
    agent_config.model.archi.norm_z = True
    agent_config.model.archi.b.hidden_dim = 256
    agent_config.model.archi.f.hidden_dim = 1024
    agent_config.model.archi.actor.hidden_dim = 1024
    agent_config.model.archi.f.hidden_layers = 1
    agent_config.model.archi.actor.hidden_layers = 1
    agent_config.model.archi.b.hidden_layers = 2

    # train
    agent_config.train.batch_size = 1024
    agent_config.train.lr_f = 1e-4
    agent_config.train.lr_b = 1e-4
    agent_config.train.lr_actor = 1e-4
    agent_config.train.ortho_coef = 1.0
    agent_config.train.train_goal_ratio = 0.5
    agent_config.train.fb_pessimism_penalty = 0.0
    agent_config.train.actor_pessimism_penalty = 0.5
    agent_config.train.discount = 0.98

    # contrastive knobs (KEEP)
    agent_config.use_contrastive = use_contrastive
    agent_config.train.use_dynamic_contrastive_z = use_dynamic_contrastive_z
    agent_config.train.contrastive_coef = contrastive_coef
    agent_config.train.quad_loss_coef = quad_loss_coef
    agent_config.train.q_loss_coef = q_loss_coef

    # keep reg_coefficient (even if your current update ignores it)
    # If your FBAgent uses it, it can be stored in cfg override.
    agent_config.train.reg_coefficient = reg_coefficient if hasattr(agent_config.train, "reg_coefficient") else reg_coefficient

    agent_config.compile = compile
    agent_config.cudagraphs = cudagraphs

    return agent_config


# =========================================================
# Config
# =========================================================
@dataclasses.dataclass
class TrainConfig:
    seed: int = 0
    env: str = "button-press-topdown-v2"

    # dataset
    dataset_path: str = "./"
    human: bool = False
    dataset_quality: float = 1.0

    # finetune
    num_train_steps: int = 1_000_000
    log_every_updates: int = 10_000
    eval_every_steps: int = 100_000
    checkpoint_every_steps: int = 1_000_000
    num_eval_episodes: int = 10
    num_inference_samples: int = 50_000

    # io
    work_dir: Optional[str] = None
    load_dir: Optional[str] = None  # checkpoint folder

    # system
    device: str = "cuda"
    compile: bool = False
    cudagraphs: bool = False

    # wandb
    use_wandb: bool = False
    wandb_ename: Optional[str] = None
    wandb_gname: Optional[str] = None
    wandb_pname: Optional[str] = "fb_finetune_metaworld"
    wandb_name_prefix: Optional[str] = None

    # preference
    num_pref_pairs: int = 200

    # contrastive knobs (KEEP)
    beta: float = 1.0
    use_contrastive: bool = False
    use_dynamic_contrastive_z: bool = False
    contrastive_coef: float = 10.0
    quad_loss_coef: float = 0.0
    reg_coefficient: float = 0.0
    q_loss_coef: float = 0.0

    segment_size: int = 25
    threshold: float = 0.5
    noise: float = 0.0


# =========================================================
# Workspace (Finetune)
# =========================================================
class Workspace:
    def __init__(self, cfg: TrainConfig, agent_cfg: FBAgentConfig):
        self.cfg = cfg
        self.agent_cfg = agent_cfg

        # work_dir
        if self.cfg.work_dir is None:
            import string
            tmp_name = "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
            self.work_dir = Path.cwd() / "tmp_fbcpr" / tmp_name
            self.cfg.work_dir = str(self.work_dir)
        else:
            self.work_dir = Path(self.cfg.work_dir)
        self.work_dir.mkdir(exist_ok=True, parents=True)
        print("working dir:", self.work_dir)

        # load pretrained checkpoint (IMPORTANT: finetune)
        if self.cfg.load_dir is None:
            raise RuntimeError("Finetune requires --checkpoint (pretrained folder)")
        print(f"Loading pretrained model from {self.cfg.load_dir}")
        self.agent = FBAgent.load(self.cfg.load_dir, device=self.cfg.device, override_cfg=self.cfg)

        set_seed_everywhere(self.cfg.seed)

        # offline dataset (raw + fb storage)
        offline_raw = MetaWorld_dataset(self.cfg)
        data = load_metaworld_dataset_for_fb(self.cfg)
        self.replay_buffer = {"train": OfflineReplayBuffer(data, device=self.agent.device)}
        print(f"[ReplayBuffer] Loaded transitions = {len(self.replay_buffer['train'])}")

        # preference dataset in your required schema
        self.pref_dataset = build_preference_dataset(self.cfg, offline_raw)

        # recon env (if your agent uses it)
        self.recon_env = make_metaworld_env(self.cfg.env, seed=self.cfg.seed)
        if hasattr(self.agent, "set_recon_env"):
            self.agent.set_recon_env(self.recon_env)

        # wandb
        if self.cfg.use_wandb:
            exp_name = "fb_finetune"
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

        for t in tqdm(range(0, int(self.cfg.num_train_steps))):
            # eval
            if t % self.cfg.eval_every_steps == 0:
                eval_score = self.eval(t)
                if eval_score > best_eval_score:
                    best_eval_score = eval_score
                    self.agent.save(str(self.work_dir / "best"))
                    print(f"Best model saved at step {t} with eval score {eval_score:.6f}")

            # update
            metrics = self.agent.update(self.replay_buffer, t, self.pref_dataset)

            # accumulate metrics
            if total_metrics is None:
                total_metrics = {k: metrics[k].clone() for k in metrics.keys()}
            else:
                total_metrics = {k: total_metrics[k] + metrics[k] for k in metrics.keys()}

            # log
            if t % self.cfg.log_every_updates == 0:
                m_dict = {}
                denom = 1 if t == 0 else self.cfg.log_every_updates
                for k in sorted(list(total_metrics.keys())):
                    tmp = total_metrics[k] / denom
                    m_dict[k] = float(np.round(tmp.mean().item(), 6))

                m_dict["duration"] = float(time.time() - self.start_time)
                m_dict["FPS"] = float(denom / (time.time() - fps_start_time))

                if self.cfg.use_wandb:
                    wandb.log({f"train/{k}": v for k, v in m_dict.items()}, step=t)
                print(m_dict)

                total_metrics = None
                fps_start_time = time.time()

            # checkpoint
            if t > 0 and (t % self.cfg.checkpoint_every_steps == 0):
                self.agent.save(str(self.work_dir / "checkpoint"))
                print(f"Checkpoint saved at step {t}")

        self.agent.save(str(self.work_dir / "checkpoint"))
        print("Final checkpoint saved.")

    def reward_inference(self):
        num_samples = self.cfg.num_inference_samples
        batch = self.replay_buffer["train"].sample(num_samples)
        rewards = batch["reward"]
        z = self.agent._model.reward_inference(
            next_obs=batch["next"]["observation"],
            reward=torch.tensor(rewards, dtype=torch.float32, device=self.agent.device),
        )
        return z.reshape(1, -1)

    def eval(self, t: int) -> float:
        # z selection
        if self.cfg.use_dynamic_contrastive_z and hasattr(self.agent, "z"):
            z = self.agent.z.reshape(1, -1)
        else:
            z = self.reward_inference()

        env = make_metaworld_env(self.cfg.env, seed=self.cfg.seed + 12345)

        num_ep = self.cfg.num_eval_episodes
        total_reward = np.zeros((num_ep,), dtype=np.float64)
        success = np.zeros((num_ep,), dtype=np.float64)
        ep_lengths = np.zeros((num_ep,), dtype=np.int32)

        for ep in range(num_ep):
            obs = env.reset()
            done = False
            steps = 0
            succ = False

            while not done:
                with torch.no_grad(), eval_mode(self.agent._model):
                    obs_tensor = torch.tensor(obs.reshape(1, -1), device=self.agent.device, dtype=torch.float32)
                    action = self.agent.act(obs=obs_tensor, z=z, mean=True).cpu().numpy()[0]

                obs, r, done, info = env.step(action)
                total_reward[ep] += float(r)
                steps += 1
                if isinstance(info, dict) and info.get("success", False):
                    succ = True

            ep_lengths[ep] = steps
            success[ep] = 1.0 if succ else 0.0

        m_dict = {
            "reward": float(np.mean(total_reward)),
            "reward#std": float(np.std(total_reward)),
            "success": float(np.mean(success)),
            "success#std": float(np.std(success)),
            "len": float(np.mean(ep_lengths)),
            "len#std": float(np.std(ep_lengths)),
        }

        if self.cfg.use_wandb:
            wandb.log({f"eval/{k}": v for k, v in m_dict.items()}, step=t)
        print(m_dict)

        # choose which metric is "eval score"
        # typically MetaWorld cares about success rate
        return m_dict["success"]


# =========================================================
# CLI
# =========================================================
def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser("FB finetune on MetaWorld with human preference pairs")

    # env/dataset
    parser.add_argument("--env", type=str, required=True, help="MetaWorld env name, e.g. button-press-topdown-v2")
    parser.add_argument("--dataset_path", type=str, default="./")
    parser.add_argument("--human", action="store_true", help="Use human_feedback/<env>/dataset.pkl as offline dataset")
    parser.add_argument("--dataset_quality", type=float, default=1.0)

    # finetune io
    parser.add_argument("--checkpoint", required=True, help="Folder produced by FBAgent.save(...)")
    parser.add_argument("--work_dir", type=str, default=None)

    # training
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_train_steps", type=int, default=1_000_000)
    parser.add_argument("--log_every_updates", type=int, default=10_000)
    parser.add_argument("--eval_every_steps", type=int, default=10_000)
    parser.add_argument("--checkpoint_every_steps", type=int, default=1_000_000)
    parser.add_argument("--num_eval_episodes", type=int, default=50)
    parser.add_argument("--num_inference_samples", type=int, default=50_000)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--cudagraphs", action="store_true")

    # preference
    parser.add_argument("--num_pref_pairs", type=int, default=200)

    # contrastive knobs (KEEP)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--use_contrastive", action="store_true")
    parser.add_argument("--use_dynamic_contrastive_z", action="store_true")
    parser.add_argument("--contrastive_coef", type=float, default=100.0)
    parser.add_argument("--quad_loss_coef", type=float, default=0.0)
    parser.add_argument("--reg_coefficient", type=float, default=0.0)
    parser.add_argument("--q_loss_coef", type=float, default=0.0)

    # wandb
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_ename", type=str, default=None)
    parser.add_argument("--wandb_gname", type=str, default=None)
    parser.add_argument("--wandb_pname", type=str, default="fb_finetune_metaworld")
    parser.add_argument("--wandb_name_prefix", type=str, default=None)

    parser.add_argument("--segment_size", type=int, default=200,
                        help="Preference segment length")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Tie threshold (per-step reward gap)")
    parser.add_argument("--noise", type=float, default=0.0,
                        help="Synthetic preference label noise")
    args = parser.parse_args()

    return TrainConfig(
        seed=args.seed,
        env=args.env,
        dataset_path=args.dataset_path,
        human=args.human,
        dataset_quality=args.dataset_quality,
        num_train_steps=args.num_train_steps,
        log_every_updates=args.log_every_updates,
        eval_every_steps=args.eval_every_steps,
        checkpoint_every_steps=args.checkpoint_every_steps,
        num_eval_episodes=args.num_eval_episodes,
        num_inference_samples=args.num_inference_samples,
        compile=args.compile,
        cudagraphs=args.cudagraphs,
        device=args.device,
        work_dir=args.work_dir,
        load_dir=args.checkpoint,
        use_wandb=args.use_wandb,
        wandb_ename=args.wandb_ename,
        wandb_gname=args.wandb_gname,
        wandb_pname=args.wandb_pname,
        wandb_name_prefix=args.wandb_name_prefix,
        beta=args.beta,
        use_contrastive=args.use_contrastive,
        use_dynamic_contrastive_z=args.use_dynamic_contrastive_z,
        contrastive_coef=args.contrastive_coef,
        quad_loss_coef=args.quad_loss_coef,
        reg_coefficient=args.reg_coefficient,
        q_loss_coef=args.q_loss_coef,
        num_pref_pairs=args.num_pref_pairs,
        segment_size=args.segment_size,
        threshold=args.threshold,
        noise=args.noise,
    )


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    cfg = parse_args()

    agent_config = create_agent(
        env_name=cfg.env,
        device=cfg.device,
        compile=cfg.compile,
        cudagraphs=cfg.cudagraphs,
        use_contrastive=cfg.use_contrastive,
        use_dynamic_contrastive_z=cfg.use_dynamic_contrastive_z,
        contrastive_coef=cfg.contrastive_coef,
        quad_loss_coef=cfg.quad_loss_coef,
        reg_coefficient=cfg.reg_coefficient,
        q_loss_coef=cfg.q_loss_coef,
        seed=cfg.seed,
    )

    ws = Workspace(cfg, agent_cfg=agent_config)
    ws.train()
