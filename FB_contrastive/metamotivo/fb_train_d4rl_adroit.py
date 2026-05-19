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
#from metamotivo.buffers.buffers import DictBuffer
from metamotivo.buffers.buffers import OfflineReplayBuffer
from metamotivo.fb import FBAgent, FBAgentConfig
from metamotivo.nn_models import eval_mode
from tqdm import tqdm
import time
import gym
import d4rl
import random
from pathlib import Path
import wandb
import json
from typing import List, Optional
import argparse

# Map domains to D4RL Adroit environments
ALL_ENVS = {
    "pen": ["pen-human-v1", "pen-cloned-v1", "pen-medium-v1", "pen-expert-v1"],
    "door": ["door-human-v1", "door-cloned-v1", "door-medium-v1", "door-expert-v1"],
    "hammer": ["hammer-human-v1", "hammer-cloned-v1", "hammer-medium-v1", "hammer-expert-v1"],
}

def create_agent(
    domain_name="pen",
    env_name="pen-human-v1",
    device="cpu",
    compile=False,
    cudagraphs=False,
):
    if domain_name not in ["pen", "door", "hammer"]:
        raise RuntimeError('FB configuration defined only for "pen", "door", "hammer"')
    
    env = gym.make(env_name)
    
    agent_config = FBAgentConfig()
    agent_config.model.obs_dim = env.observation_space.shape[0]
    agent_config.model.action_dim = env.action_space.shape[0]
    agent_config.model.device = device
    agent_config.model.norm_obs = False
    agent_config.model.seq_length = 1
    agent_config.train.batch_size = 1024

    # Adroit environments generally use z_dim = 50
    #agent_config.model.archi.z_dim = 50
    agent_config.model.archi.z_dim = 100

    agent_config.model.archi.b.norm = True
    agent_config.model.archi.norm_z = True
    agent_config.model.archi.b.hidden_dim = 256
    agent_config.model.archi.f.hidden_dim = 1024
    agent_config.model.archi.actor.hidden_dim = 1024
    agent_config.model.archi.f.hidden_layers = 1
    agent_config.model.archi.actor.hidden_layers = 1
    agent_config.model.archi.b.hidden_layers = 2

    agent_config.train.lr_f = 1e-4
    agent_config.train.lr_b = 1e-4
    agent_config.train.lr_actor = 1e-4
    #agent_config.train.ortho_coef = 1
    agent_config.train.ortho_coef = 100
    agent_config.train.train_goal_ratio = 0.5
    agent_config.train.fb_pessimism_penalty = 0
    agent_config.train.actor_pessimism_penalty = 0.5
    agent_config.train.discount = 0.98  # Adroit generally use 0.98
    agent_config.compile = compile
    agent_config.cudagraphs = cudagraphs

    return agent_config


def load_data(env_name, num_episodes=1):
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)
    
    storage = {
        "observation": dataset["observations"].astype(np.float32),
        "action": dataset["actions"].astype(np.float32),
        "reward": dataset["rewards"].astype(np.float32),
        "next": {
            "observation": dataset["next_observations"].astype(np.float32),
            "terminated": dataset["terminals"].astype(np.bool_),
        },
    }
    
    if num_episodes > 0:
        episode_lengths = np.where(dataset["terminals"])[0]
        if len(episode_lengths) >= num_episodes:
            end_idx = episode_lengths[num_episodes - 1] + 1
            for key in storage:
                if key == "next":
                    for subkey in storage[key]:
                        storage[key][subkey] = storage[key][subkey][:end_idx]
                else:
                    storage[key] = storage[key][:end_idx]
    
    return storage

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

@dataclasses.dataclass
class TrainConfig:
    seed: int = 0
    domain_name: str = "walker"
    env_name: Optional[str] = None
    num_train_steps: int = 3000000
    load_n_episodes: int = 5000
    log_every_updates: int = 10000
    work_dir: Optional[str] = None
    checkpoint_every_steps: int = 1000000
    num_eval_episodes: int = 10
    num_inference_samples: int = 50000
    eval_every_steps: int = 100000
    eval_envs: Optional[List[str]] = None
    compile: bool = False
    cudagraphs: bool = False
    device: str = "cuda"
    use_wandb: bool = False
    wandb_ename: Optional[str] = None
    wandb_gname: Optional[str] = None
    wandb_pname: Optional[str] = "fb_train_d4rl"
    wandb_name_prefix: Optional[str] = None

    def __post_init__(self):
        if self.eval_envs is None:
            self.eval_envs = ALL_ENVS[self.domain_name]

class Workspace:
    def __init__(self, cfg, agent_cfg):
        self.cfg = cfg
        self.agent_cfg = agent_cfg
        if self.cfg.work_dir is None:
            import string
            #tmp_name = "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
            tmp_name = time.strftime("%Y%m%d-%H%M%S") + '-fb-' + self.cfg.env_name
            self.work_dir = Path.cwd() / "tmp_fbcpr" / tmp_name
            self.cfg.work_dir = str(self.work_dir)
        else:
            self.work_dir = Path(self.cfg.work_dir)
        self.work_dir.mkdir(exist_ok=True, parents=True)
        print("working dir: {}".format(self.work_dir))

        self.agent = FBAgent(**dataclasses.asdict(self.agent_cfg))
        set_seed_everywhere(self.cfg.seed)

        if self.cfg.use_wandb:
            exp_name = "fb"
            wandb_name = exp_name
            if self.cfg.wandb_name_prefix:
                wandb_name = "{}_{}".format(self.cfg.wandb_name_prefix, exp_name)
            wandb_config = dataclasses.asdict(self.cfg)
            wandb.init(
                entity=self.cfg.wandb_ename,
                project=self.cfg.wandb_pname,
                group=self.cfg.wandb_gname,
                name=wandb_name,
                config=wandb_config,
            )

        with open(self.work_dir / "config.json", "w") as f:
            json.dump(dataclasses.asdict(self.cfg), f, indent=4)

    def train(self):
        self.start_time = time.time()
        self.train_offline()

    def train_offline(self):
        self.replay_buffer = {}
        data = load_data(
            self.cfg.env_name,
            self.cfg.load_n_episodes,
        )
        #self.replay_buffer = {"train": DictBuffer(capacity=data["observation"].shape[0], device=self.agent.device)}
        #self.replay_buffer["train"].extend(data)
        self.replay_buffer = {"train": OfflineReplayBuffer(data, device=self.agent.device)}
        print(self.replay_buffer["train"])
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
                    tmp = total_metrics[k] / (1 if t == 0 else self.cfg.log_every_updates)
                    m_dict[k] = np.round(tmp.mean().item(), 6)
                m_dict["duration"] = time.time() - self.start_time
                m_dict["FPS"] = (1 if t == 0 else self.cfg.log_every_updates) / (time.time() - fps_start_time)
                if self.cfg.use_wandb:
                    wandb.log(
                        {"train/{}".format(k): v for k, v in m_dict.items()},
                        step=t,
                    )
                print(m_dict)
                total_metrics = None
                fps_start_time = time.time()
            if t % self.cfg.checkpoint_every_steps == 0:
                self.agent.save(str(self.work_dir / "checkpoint"))
        self.agent.save(str(self.work_dir / "checkpoint"))

    def eval(self, t):
        for env_name in self.cfg.eval_envs:
            z = self.reward_inference(env_name).reshape(1, -1)
            eval_env = gym.make(env_name)
            num_ep = self.cfg.num_eval_episodes
            total_reward = np.zeros((num_ep,), dtype=np.float64)
            for ep in range(num_ep):
                obs = eval_env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
                done = False
                while not done:
                    with torch.no_grad(), eval_mode(self.agent._model):
                        obs_tensor = torch.tensor(
                            obs.reshape(1, -1),
                            device=self.agent.device,
                            dtype=torch.float32,
                        )
                        #action = self.agent.act(obs=obs_tensor, z=z, mean=True).cpu().numpy()
                        action = self.agent.act(obs=obs_tensor, z=z, mean=True).cpu().numpy().squeeze()
                    #next_obs, reward, terminated, truncated, _ = eval_env.step(action)
                    next_obs, reward, done, _ = eval_env.step(action)
                    total_reward[ep] += reward
                    obs = next_obs
                   # done = terminated or truncated
            m_dict = {
                "reward": np.mean(total_reward),
                "reward#std": np.std(total_reward),
            }
            if self.cfg.use_wandb:
                wandb.log(
                    {"{}/{}".format(env_name, k): v for k, v in m_dict.items()},
                    step=t,
                )
            m_dict["env"] = env_name
            print(m_dict)

    def reward_inference(self, env_name):
        num_samples = self.cfg.num_inference_samples
        batch = self.replay_buffer["train"].sample(num_samples)
        rewards = batch["reward"]
        z = self.agent._model.reward_inference(
            next_obs=batch["next"]["observation"],
            reward=torch.tensor(rewards, dtype=torch.float32, device=self.agent.device),
        )
        return z

def parse_args():
    parser = argparse.ArgumentParser(description="Train FBAgent on D4RL MuJoCo environment")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--domain_name", type=str, default="pen", help="Domain name (pen, door, hammer)")
    parser.add_argument("--env_name", type=str, default=None, help="D4RL Adroit environment name")
    parser.add_argument("--num_train_steps", type=int, default=3000000, help="Number of training steps")
    parser.add_argument("--load_n_episodes", type=int, default=5000, help="Number of episodes to load")
    parser.add_argument("--log_every_updates", type=int, default=10000, help="Log every N updates")
    parser.add_argument("--work_dir", type=str, default=None, help="Working directory")
    parser.add_argument("--checkpoint_every_steps", type=int, default=1000000, help="Checkpoint every N steps")
    parser.add_argument("--num_eval_episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--num_inference_samples", type=int, default=50000, help="Number of inference samples")
    parser.add_argument("--eval_every_steps", type=int, default=100000, help="Evaluate every N steps")
    parser.add_argument("--eval_envs", type=str, default=None, help="Comma-separated list of evaluation environments")
    parser.add_argument("--compile", action="store_true", help="Enable compilation")
    parser.add_argument("--cudagraphs", action="store_true", help="Enable CUDA graphs")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cpu, cuda)")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--wandb_ename", type=str, default=None, help="WandB entity name")
    parser.add_argument("--wandb_gname", type=str, default=None, help="WandB group name")
    parser.add_argument("--wandb_pname", type=str, default="fb_train_d4rl", help="WandB project name")
    parser.add_argument("--wandb_name_prefix", type=str, default=None, help="WandB name prefix")
    
    args = parser.parse_args()
    
    # Convert eval_envs to list if provided
    eval_envs = ALL_ENVS[args.domain_name] if args.eval_envs is None else args.eval_envs.split(",")
    
    return TrainConfig(
        seed=args.seed,
        domain_name=args.domain_name,
        env_name=args.env_name,
        num_train_steps=args.num_train_steps,
        load_n_episodes=args.load_n_episodes,
        log_every_updates=args.log_every_updates,
        work_dir=args.work_dir,
        checkpoint_every_steps=args.checkpoint_every_steps,
        num_eval_episodes=args.num_eval_episodes,
        num_inference_samples=args.num_inference_samples,
        eval_every_steps=args.eval_every_steps,
        eval_envs=eval_envs,
        compile=args.compile,
        cudagraphs=args.cudagraphs,
        device=args.device,
        use_wandb=args.use_wandb,
        wandb_ename=args.wandb_ename,
        wandb_gname=args.wandb_gname,
        wandb_pname=args.wandb_pname,
        wandb_name_prefix=args.wandb_name_prefix,
    )

if __name__ == "__main__":
    config = parse_args()

    if config.env_name is None:
        if config.domain_name == "pen":
            config.env_name = "pen-human-v1"
        elif config.domain_name == "door":
            config.env_name = "door-human-v1"
        elif config.domain_name == "hammer":
            config.env_name = "hammer-human-v1"
        else:
            raise RuntimeError("Unsupported domain, you need to specify env_name")

    
    agent_config = create_agent(
        domain_name=config.domain_name,
        env_name=config.env_name,
        device=config.device,
        compile=config.compile,
        cudagraphs=config.cudagraphs,
    )

    ws = Workspace(config, agent_cfg=agent_config)
    ws.train()
