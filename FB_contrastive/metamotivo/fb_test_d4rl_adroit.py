import torch
import numpy as np
import dataclasses
from metamotivo.fb import FBAgent, FBAgentConfig
from metamotivo.nn_models import eval_mode
from pathlib import Path
import pickle
import argparse
import os
import random
from metamotivo.buffers.buffers import OfflineReplayBuffer
from tqdm import tqdm
import mujoco
import gym
import d4rl
import imageio
import gymnasium
import gymnasium_robotics

gymnasium.register_envs(gymnasium_robotics)

# -----------------
# Dataset loaders
# -----------------
def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


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


# -----------------
# Config
# -----------------
@dataclasses.dataclass
class TestConfig:
    seed: int = 0
    env_name: str = "walk"
    device: str = "cuda"
    checkpoint: str = None
    num_eval_episodes: int = 10
    load_n_episodes: int = 5000

# -----------------
# Workspace
# -----------------
class Workspace:
    def __init__(self, cfg, agent_cfg):
        self.cfg = cfg
        self.agent_cfg = agent_cfg
        set_seed_everywhere(cfg.seed)
        print(f"Loading pretrained model from {cfg.checkpoint}")
        self.agent = FBAgent.load(cfg.checkpoint, device=cfg.device)

        data = load_data(
            self.cfg.env_name,
            self.cfg.load_n_episodes,
        )
        self.replay_buffer = {"train": OfflineReplayBuffer(data, device=self.agent.device)}

    def reward_inference(self, env_name):
        num_samples = 10000
        batch = self.replay_buffer["train"].sample(num_samples)
        rewards = batch["reward"]
        z = self.agent._model.reward_inference(
            next_obs=batch["next"]["observation"],
            reward=torch.tensor(rewards, dtype=torch.float32, device=self.agent.device),
        )
        return z

    def run_eval(self, z, return_traj=False, record_video=False, video_dir="videos"):

        eval_env = gymnasium.make('AdroitHandPen-v1', max_episode_steps=1000)
        num_ep = self.cfg.num_eval_episodes

        total_reward, ep_lengths = np.zeros(num_ep), np.zeros(num_ep, dtype=np.int32)
        trajectories = []

        for ep in range(num_ep):
            step = 0
            obs = eval_env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            tem = False
            while not tem:
                with torch.no_grad(), eval_mode(self.agent._model):
                    obs_tensor = torch.tensor(
                        obs.reshape(1, -1),
                        device=self.agent.device,
                        dtype=torch.float32,
                    )
                    #action = self.agent.act(obs=obs_tensor, z=z, mean=True).cpu().numpy()
                    action = self.agent.act(obs=obs_tensor, z=z, mean=True).cpu().numpy().squeeze()
                #next_obs, reward, terminated, truncated, _ = eval_env.step(action)
                next_obs, reward, done, trun, _ = eval_env.step(action)
                tem = done or trun
                total_reward[ep] += reward
                obs = next_obs
                step += 1
            ep_lengths[ep] = step

        res = {
            "reward": np.mean(total_reward),
            "reward#std": np.std(total_reward),
            "len": np.mean(ep_lengths),
            "len#std": np.std(ep_lengths),
        }
        if return_traj:
            return res, trajectories
        return res



    def test(self):
        all_trajs = []

        if hasattr(self.agent, "z") and self.agent.z is not None:
            print("[INFO] Loaded z from checkpoint, will use agent.z for evaluation")
            z = self.agent.z.reshape(1, -1)
        else:
            print("[INFO] No z stored in checkpoint, computing z using reward_inference()")
        z = self.reward_inference(self.cfg.env_name).reshape(1, -1)

        # Evaluate
        res, trajs = self.run_eval(z, return_traj=True, record_video=True, video_dir="videos/cheetah_walk_finetune")
        all_trajs.extend(trajs)
        print(f"[EVAL] mean_reward={res['reward']:.2f}, std_reward={res['reward#std']:.2f}")
        print(f"[EVAL] mean_length={res['len']:.2f}, std_length={res['len#std']:.2f}")



# -----------------
# Main
# -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="pen-human-v1")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_eval_episodes", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = TestConfig(
        env_name=args.env_name,
        device=args.device,
        num_eval_episodes=args.num_eval_episodes,
        checkpoint=args.checkpoint,
    )

    ws = Workspace(cfg, FBAgentConfig())
    ws.test()
