from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import gym
import os
import math
import pickle
from tqdm import tqdm
#from dm_control import suite
import mujoco
from metamotivo.fb_bt.agent import FBAgent
from metamotivo.nn_models import eval_mode
from bt_model import reward_model as rem
from url_benchmark import dmc
# ------------------------------------------------
# CLI
# ------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser("Offline→Online evaluation for FBAgent")
    p.add_argument("--checkpoint", required=True,
                   help="Folder produced by FBAgent.save(...)")
    p.add_argument("--dataset_path", type=str, default="datasets")
    p.add_argument("--expl_agent", type=str, default="rnd")
    p.add_argument("--num_episodes", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--domain_name", type=str, default="walker", help="Domain name (e.g., walker, cheetah, quadruped)")
    p.add_argument("--task_name", type=str, default=None, help="Task name (e.g., walk, run)")
    p.add_argument("--episodes", type=int, default=10,
                   help="Number of roll-out episodes")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--render", action="store_true",
                   help="Render environment while running")
    return p.parse_args()

def derive_z(agent: FBAgent, ds: dict, reward_model: rem) -> torch.Tensor:
    """
    隨機抽取 number of samples 筆 data → encode_expert → 取得 z
    """
    bs = agent.cfg.train.batch_size
    number_of_samples = 10000
    #print(f'[INFO] Using {number_of_samples} samples')
    idx = torch.randint(0, ds['observations'].shape[0], (number_of_samples,))
    ds["actions"] = ds["actions"].reshape(-1, ds["actions"].shape[-1])
    ds["next_observations"] = ds["next_observations"].reshape(-1, ds["next_observations"].shape[-1])
    ds["observations"] = ds["observations"].reshape(-1, ds["observations"].shape[-1])
    #print(ds["actions"].shape)

    batch_next_obs = ds['next_observations'][idx]
    batch_obs = ds['observations'][idx]
    batch_act = ds['actions'][idx]

    s_a = np.concatenate([
        batch_obs.detach().cpu().numpy() if isinstance(batch_obs, torch.Tensor) else batch_obs,
        batch_act.detach().cpu().numpy() if isinstance(batch_act, torch.Tensor) else batch_act
    ], axis=-1)

    rewards = []
    for member in range(3):
        reward = reward_model.r_hat_member(s_a, member)
        rewards.append(reward)
    rewards = torch.stack(rewards, dim=0).mean(dim=0).to('cuda')

    num_batches = int(np.ceil(number_of_samples / bs))
    z = 0

    with torch.no_grad():
        for i in range(num_batches):
            start_idx, end_idx = i * bs, (i + 1) * bs
            B = agent._model._backward_map(
                torch.tensor(batch_next_obs[start_idx:end_idx], device='cuda', dtype=torch.float32)
            ).detach()

            z += torch.matmul(rewards[start_idx:end_idx].T, B)

    z = math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)
    return z

def load_data(dataset_path, expl_agent, domain_name, num_episodes=1):
    path = Path(dataset_path) / f"{domain_name}/{expl_agent}/buffer"
    print(f"Data path: {path}")
    storage = {
        "observation": [],
        "action": [],
        "physics": [],
        "next": {"observation": [], "terminated": [], "physics": []},
    }
    files = list(path.glob("*.npz"))
    num_episodes = min(num_episodes, len(files))
    for i in tqdm(range(num_episodes)):
        f = files[i]
        data = np.load(str(f))
        storage["observation"].append(data["observation"][:-1].astype(np.float32))
        storage["action"].append(data["action"][1:].astype(np.float32))
        storage["next"]["observation"].append(data["observation"][1:].astype(np.float32))
        storage["next"]["terminated"].append(np.array(1 - data["discount"][1:], dtype=bool))
        storage["physics"].append(data["physics"][:-1])
        storage["next"]["physics"].append(data["physics"][1:])

    for k in storage:
        if k == "next":
            for k1 in storage[k]:
                #storage[k][k1] = np.concat(storage[k][k1])
                storage[k][k1] = np.concatenate(storage[k][k1])
        else:
            #storage[k] = np.concat(storage[k])
            storage[k] = np.concatenate(storage[k])
            
    return storage

# ------------------------------------------------
# Helper: 線上一回合
# ------------------------------------------------
@torch.no_grad()
def run_episode(env: gym.Env, agent: FBAgent, z: torch.Tensor,
                device: str, render: bool = False) -> float:
    obs = env.reset()
    if isinstance(obs, tuple):                      # gymnasium 相容
        obs = obs[0]
    ep_ret, done = 0.0, False
    while not done:
        if render:
            env.render()
        obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                device=device).unsqueeze(0)
        action = agent.act(obs=obs_t, z=z, mean=True).cpu().numpy().squeeze()
        obs, reward, terminated, truncated, _ = env.step(action)
        ep_ret += reward
        done = terminated or truncated
    return ep_ret

# ------------------------------------------------
# Main
# ------------------------------------------------
def main():
    args = parse_args()
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint '{ckpt}' 不存在")

    print(f"[INFO] Loading FBAgent from {ckpt}")
    agent: FBAgent = FBAgent.load(str(ckpt), device=args.device)
    agent.eval()

    # -- 1. 讀取離線 dataset（只要 next_observations）
    print(f"[INFO] Loading dataset '{args.domain_name}-{args.task_name}'")
    ds = load_data(
            dataset_path=args.dataset_path,
            expl_agent=args.expl_agent,
            domain_name=args.domain_name,
            num_episodes=args.num_episodes,
        )
    print(ds['action'].shape)

    # -- 2. 建立線上測試環境
    eval_env = dmc.make(f"{args.domain_name}_{args.task_name}")
    print(f"[INFO] Evaluating on '{args.domain_name}-{args.task_name}'  episodes={args.episodes}")

    #returns: list[float] = []
    reward_model = rem.RewardModel(eval_env, eval_env.env.observation_spec().shape[0], eval_env.action_spec().shape[0], ensemble_size=3, lr=3e-4,
                                    activation="tanh", device=args.device)
    reward_model.load_model(f'bt_model/reward_model_logs/{args.domain_name}-{args.task_name}/seed_{args.seed}/models/reward_model.pt')
    print(f"Loaded BT reward model from bt_model/reward_model_logs/{args.domain_name}-{args.task_name}/seed_{args.seed}/models/reward_model.pt")

    returns = []
    for ep in range(args.episodes):
        z = derive_z(agent, ds, reward_model)
        obs = eval_env.reset()
        total_reward = 0.0
        time_step = eval_env.reset()
        while not time_step.last():
            with torch.no_grad(), eval_mode(agent._model):
                obs = torch.tensor(
                    time_step.observation.reshape(1, -1),
                    device=agent.device,
                    dtype=torch.float32,
                )
                action = agent.act(obs=obs, z=z, mean=True).cpu().numpy()
            time_step = eval_env.step(action)
            total_reward += time_step.reward
        returns.append(total_reward)
        print(f"[Episode {ep+1:02d}] return = {total_reward:.2f}")

    returns = np.array(returns)

    print("\n===== Summary =====")
    print(f"Average return          : {np.mean(returns):.2f}")
    print(f"Std-dev return          : {np.std(returns):.2f}")
    print(f"{np.mean(returns):.0f}±{np.std(returns):.0f}")

    eval_env.close()
    print("[INFO] Done.")

if __name__ == "__main__":
    main()