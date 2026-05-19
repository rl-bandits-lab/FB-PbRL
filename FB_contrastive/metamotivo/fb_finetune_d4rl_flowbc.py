# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.
#
# This file has been modified for the paper "From Reward-Free Representations
# to Preferences: Rethinking Offline Preference-Based Reinforcement Learning", 2026.

from __future__ import annotations
import torch
torch.set_float32_matmul_precision("high")

import numpy as np
import dataclasses
from metamotivo.buffers.buffers import OfflineReplayBuffer
#from metamotivo.fb_contrastive_finetune_antmaze import FBAgent, FBAgentConfig
#from metamotivo.fb_antmaze_flowbc.flow_bc.agent import FBFlowBCAgentConfig
#from metamotivo.fb_contrastive_finetune_antmaze_flowbc2.agent import FBAgent, FBAgentConfig
from metamotivo.fb_contrastive_finetune_adroit_flowbc.flow_bc.agent import FBFlowBCAgent, FBFlowBCAgentConfig
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

def load_d4rl_dataset(env_name: str) -> dict:
    import gym
    import d4rl
    import numpy as np

    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)

    obs = dataset["observations"].astype(np.float32)
    next_obs = dataset["next_observations"].astype(np.float32)
    actions = dataset["actions"].astype(np.float32)
    rewards = dataset["rewards"].astype(np.float32)
    terminals = dataset["terminals"].astype(bool)

    N = len(obs)

    # ------------------------------------------------------------
    # Recompute terminated using obs / next_obs consistency
    # ------------------------------------------------------------
    terminated = np.zeros((N, 1), dtype=bool)

    for i in range(N - 1):
        obs_break = not np.allclose(
            obs[i + 1],
            next_obs[i],
            atol=1e-5,
        )
        if obs_break or terminals[i]:
            terminated[i, 0] = True

    terminated[-1, 0] = True  # force last transition to terminate
    nonzero_reward_mask = rewards != 0.0
    num_nonzero = np.sum(nonzero_reward_mask)

    print(f"[AntMaze] Non-zero rewards count : {num_nonzero}")
    print(f"[AntMaze] Total transitions     : {len(rewards)}")
    print(f"[AntMaze] Non-zero ratio        : {num_nonzero / len(rewards):.6f}")
    env.close()

    return {
        "observation": obs,
        "action": actions,
        "reward": rewards.reshape(-1, 1),
        "next": {
            "observation": next_obs,
            "terminated": terminated,
        },
    }


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
        idx1 = np.array(pickle.load(f), dtype=np.int64)[:100]
    with open(f2, "rb") as f:
        idx2 = np.array(pickle.load(f), dtype=np.int64)[:100]
    with open(fl, "rb") as f:
        raw_labels = np.array(pickle.load(f))

    print(idx1.shape, idx2.shape, raw_labels.shape)
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
# Configs
# =========================================================

@dataclasses.dataclass
class TrainConfig:
    seed: int = 0

    # --- env name (D4RL) ---
    env_name: str = "antmaze-medium-play-v2"

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
    lr_b: float = 1e-5
    bs: int = 256

    # pretrained checkpoint (folder)
    load_dir: str | None = None

    use_wandb: bool = False
    wandb_ename: Optional[str] = None
    wandb_gname: Optional[str] = None
    wandb_pname: Optional[str] = "fb_train_d4rl"
    wandb_name_prefix: Optional[str] = None

    ortho_coef: float = 1.0


# =========================================================
# Workspace
# =========================================================

class Workspace:
    def __init__(self, cfg: TrainConfig, agent_cfg: FBFlowBCAgentConfig):
        self.cfg = cfg
        self.agent_cfg = agent_cfg

        if self.cfg.work_dir is None:
            import string
            tmp_name = time.strftime("%Y%m%d-%H%M%S") + '-fb-finetune-tdjepa-notie-' + self.cfg.env_name + f'-seed{cfg.seed}'
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
        #self.agent = FBAgent.load(self.cfg.load_dir, device=self.cfg.device, override_cfg=self.cfg)
        self.agent = FBFlowBCAgent.load(
            self.cfg.load_dir,
            device=self.cfg.device,
            override_cfg={
                "use_contrastive": self.cfg.use_contrastive,
                "use_dynamic_contrastive_z": self.cfg.use_dynamic_contrastive_z,
                "contrastive_coef": self.cfg.contrastive_coef,
                "quad_loss_coef": self.cfg.quad_loss_coef,
                "q_loss_coef": self.cfg.q_loss_coef,
                "ortho_coef": self.cfg.ortho_coef,
                "lr_b": self.cfg.lr_b,
                "bs": self.cfg.bs,
                "seq_length": self.cfg.query_len,
            },
        )

        set_seed_everywhere(self.cfg.seed)

        # -------------------------
        # 1) Load preference dataset (PT human)
        # -------------------------
        print(f"[INFO] Loading PT human preference dataset for {self.cfg.env_name}")
        self.pref_dataset = load_pt_human_preference_dataset(
            env_name=self.cfg.env_name,
            human_label_root=self.cfg.human_label_root,
            seq_len=self.cfg.query_len,
            num_query=self.cfg.num_query,
        )

        # -------------------------
        # 2) Load offline dataset (D4RL)
        # -------------------------
        print(f"[INFO] Loading AntMaze offline dataset from D4RL: {self.cfg.env_name}")
        data = load_d4rl_dataset(self.cfg.env_name)
        self.replay_buffer = {"train": OfflineReplayBuffer(data, device=self.agent.device)}
        print(f"[ReplayBuffer] Loaded transitions = {len(self.replay_buffer['train'])}")

        # AntMaze: no recon env required (and dmc not available)
        # self.agent.set_recon_env(...)  # DO NOT call for antmaze
        if self.cfg.use_wandb:
            exp_name = f"fb-finetune-tdjepa-notie-{cfg.env_name}-{cfg.seed}"
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
        Eval: run gym env, report return + normalized score.
        """
        # pick z
        if self.cfg.use_dynamic_contrastive_z:
            z = self.agent.z.reshape(1, -1)
        else:
            z = self.reward_inference().reshape(1, -1)

        env = gym.make(self.cfg.env_name)

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
        
        if len(normalized_scores) > 0:
            m_dict["normalized_score"] = float(np.mean(normalized_scores))
            m_dict["normalized_score#std"] = float(np.std(normalized_scores))
        if self.cfg.use_wandb:
            wandb.log({f"{self.cfg.env_name}/{k}": v for k, v in m_dict.items()}, step=t)

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
    p.add_argument("--wandb_pname", type=str, default="fb_train_d4rl")
    p.add_argument("--wandb_name_prefix", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--env_name", type=str, required=True, help="e.g., antmaze-medium-play-v2")
    p.add_argument("--human_label_root", type=str, default="./human_label")
    p.add_argument("--num_query", type=int, default=100)
    p.add_argument("--query_len", type=int, default=100)

    p.add_argument("--num_train_steps", type=int, default=1_000_000)
    p.add_argument("--log_every_updates", type=int, default=10_000)
    p.add_argument("--checkpoint_every_steps", type=int, default=100_000)

    p.add_argument("--num_eval_episodes", type=int, default=15)
    p.add_argument("--eval_every_steps", type=int, default=100_00)
    p.add_argument("--bs", type=int, default=256)

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
    p.add_argument("--lr_b", type=float, default=1e-5)

    p.add_argument("--checkpoint", required=True, help="Folder produced by FBAgent.save(...)")
    args = p.parse_args()

    cfg = TrainConfig(
        seed=args.seed,
        device=args.device,
        env_name=args.env_name,
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
        bs=args.bs,
        lr_b=args.lr_b,
    )
    return cfg


if __name__ == "__main__":
    cfg = parse_args()

    ws = Workspace(cfg, agent_cfg=None)
    ws.train()
