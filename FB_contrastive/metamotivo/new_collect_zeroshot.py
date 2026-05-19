# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.
#
# This file has been modified for the paper "From Reward-Free Representations
# to Preferences: Rethinking Offline Preference-Based Reinforcement Learning", 2026.

import numpy as np
import pickle
import random
from pathlib import Path
from tqdm import tqdm
import mujoco
from url_benchmark import dmc

# ============================================================
#  Utils: Setup
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    print(f"🔒 Seed fixed to {seed}")

# ============================================================
#  Part 1. Load Data & Physics Calculation
# ============================================================
def compute_rewards_from_physics(domain_name: str, task: str, physics_batch, action_batch):
    env = dmc.make(
        f"{domain_name}_{task}",
        obs_type="states",
        frame_stack=1,
        action_repeat=1,
        seed=0,
    )
    physics_batch = np.array(physics_batch)
    action_batch = np.array(action_batch)

    rewards = []
    for i in range(len(action_batch)):
        with env._physics.reset_context():
            env._physics.set_state(physics_batch[i])
            env._physics.set_control(action_batch[i])
        mujoco.mj_forward(env._physics.model.ptr, env._physics.data.ptr)
        mujoco.mj_fwdPosition(env._physics.model.ptr, env._physics.data.ptr)
        mujoco.mj_sensorVel(env._physics.model.ptr, env._physics.data.ptr)
        mujoco.mj_subtreeVel(env._physics.model.ptr, env._physics.data.ptr)
        r = env._task.get_reward(env._physics)
        rewards.append(r)
    return np.array(rewards).reshape(-1, 1)


def load_rnd_dataset_as_trajectories(dataset_path: str, domain_name: str, expl_agent: str, task: str, num_episodes: int = 1):
    path = Path(dataset_path) / f"{domain_name}/{expl_agent}/buffer"
    print(f"Data path: {path}")
    files = sorted(list(path.glob("*.npz")))
    
    num_episodes = min(num_episodes, len(files))
    print(f"Loading {num_episodes} episodes...")

    trajectories = []
    for i in tqdm(range(num_episodes), desc="Loading episodes"):
        f = files[i]
        try:
            data = np.load(str(f))
            obs = data["observation"].astype(np.float32)
            act = data["action"].astype(np.float32)
            physics = data["physics"]
            discount = data["discount"]

            rewards = compute_rewards_from_physics(
                domain_name=domain_name,
                task=task,
                physics_batch=physics[1:],
                action_batch=act[1:],
            ).astype(np.float32)

            traj = {
                "observations": obs[:-1],
                "actions": act[1:],
                "rewards": rewards,
                "next_observations": obs[1:],
                "terminals": np.array(1 - discount[1:], dtype=bool),
            }
            trajectories.append(traj)
        except Exception as e:
            print(f"Error loading {f}: {e}")
            continue

    return trajectories

# ============================================================
#  Part 2. Sampling Segments (Pre-slicing)
# ============================================================
def sample_segment_pool(trajectories, pool_size=400, seg_len=25):

    segments = []
    num_trajs = len(trajectories)
    
    print(f"Sampling {pool_size} segments of length {seg_len}...")
    
    pbar = tqdm(total=pool_size)
    while len(segments) < pool_size:

        traj_idx = np.random.randint(num_trajs)
        traj = trajectories[traj_idx]
        
        traj_len = len(traj['observations'])
        if traj_len < seg_len:
            continue
            
        start_idx = np.random.randint(0, traj_len - seg_len + 1)
        end_idx = start_idx + seg_len
        
        seg = {
            'observations': traj['observations'][start_idx:end_idx],
            'next_observations': traj['next_observations'][start_idx:end_idx],
            'actions': traj['actions'][start_idx:end_idx],
            'rewards': traj['rewards'][start_idx:end_idx],
            'original_start_idx': start_idx
        }
        segments.append(seg)
        pbar.update(1)
        
    pbar.close()
    return segments

# ============================================================
#  Part 3. Build Preference Dataset from Pool
# ============================================================
def build_pref_dataset_from_pool(segment_pool, num_pairs=2000, teacher_eps_skip=0.5):

    all_seg_rewards = np.concatenate([seg["rewards"] for seg in segment_pool], axis=0)
    mean_reward = np.mean(all_seg_rewards)
    seg_len = len(segment_pool[0]['observations'])
    skip_margin = mean_reward * teacher_eps_skip * seg_len
    
    print(f"Building pairs... Margin: {skip_margin:.4f} (Mean R per step: {mean_reward:.4f})")

    pref_dataset = []
    num_segments = len(segment_pool)
    n_skip = 0
    
    max_attempts = num_pairs * 10
    attempts = 0

    pbar = tqdm(total=num_pairs, desc="Generating pairs")
    
    while len(pref_dataset) < num_pairs and attempts < max_attempts:
        attempts += 1
        
        idx1 = np.random.randint(num_segments)
        idx2 = np.random.randint(num_segments)
        
        if idx1 == idx2: continue
        
        seg1 = segment_pool[idx1]
        seg2 = segment_pool[idx2]

        R1, R2 = seg1['rewards'].sum(), seg2['rewards'].sum()

        if abs(R1 - R2) < skip_margin:
            n_skip += 1
            label = np.array([0.5, 0.5], dtype=np.float32)
        elif R1 > R2:
            label = np.array([1.0, 0.0], dtype=np.float32)
        else:
            label = np.array([0.0, 1.0], dtype=np.float32)

        pref_dataset.append({
            'seg1': seg1,
            'seg2': seg2,
            'label': label,
            'start1': seg1['original_start_idx'],
            'start2': seg2['original_start_idx']
        })
        pbar.update(1)
        
    pbar.close()
    print(f"Skip ratio (Neutral pairs): {n_skip}/{len(pref_dataset)} = {n_skip/len(pref_dataset):.2%}")
    return pref_dataset

# ============================================================
#  Part 4. Save
# ============================================================
def save_pref_dataset(pref_dataset, save_path):

    num_pairs = len(pref_dataset)
    if num_pairs == 0:
        print("❌ No pairs generated!")
        return

    seg_len = len(pref_dataset[0]['seg1']['observations'])

    human_labels = np.zeros((num_pairs, 2), dtype=np.float32)
    seg_obs_1, seg_act_1, seg_next_obs_1 = [], [], []
    seg_obs_2, seg_act_2, seg_next_obs_2 = [], [], []
    seq_timestep_1, seq_timestep_2 = [], []
    start_indices_1, start_indices_2 = [], []

    for i, pair in enumerate(pref_dataset):
        seg1, seg2, label = pair['seg1'], pair['seg2'], pair['label']

        human_labels[i] = label
        seg_obs_1.append(seg1['observations'])
        seg_act_1.append(seg1['actions'])
        seg_next_obs_1.append(seg1['next_observations'])

        seg_obs_2.append(seg2['observations'])
        seg_act_2.append(seg2['actions'])
        seg_next_obs_2.append(seg2['next_observations'])

        seq_timestep_1.append(np.arange(seg_len))
        seq_timestep_2.append(np.arange(seg_len))

        start_indices_1.append(pair['start1'])
        start_indices_2.append(pair['start2'])

    batch = dict(
        labels=human_labels,
        observations=np.array(seg_obs_1, dtype=np.float32),
        actions=np.array(seg_act_1, dtype=np.float32),
        next_observations=np.array(seg_next_obs_1, dtype=np.float32),
        observations_2=np.array(seg_obs_2, dtype=np.float32),
        actions_2=np.array(seg_act_2, dtype=np.float32),
        next_observations_2=np.array(seg_next_obs_2, dtype=np.float32),
        timestep_1=np.array(seq_timestep_1, dtype=np.int32),
        timestep_2=np.array(seq_timestep_2, dtype=np.int32),
        start_indices=np.array(start_indices_1, dtype=np.int32),
        start_indices_2=np.array(start_indices_2, dtype=np.int32),
    )

    # Ensure directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, "wb") as f:
        pickle.dump(batch, f)

    print(f"✅ Preference dataset saved at {save_path}, total pairs = {num_pairs}")
    return batch

# ============================================================
#  Main Execution
# ============================================================
if __name__ == "__main__":

    SEED = 0
    set_seed(SEED)
    
    dataset_path = "./datasets_dmc"
    domain_name = "walker"
    task = "walk"
    expl_agent = "rnd"
    print(domain_name, task)

    POOL_SIZE = 400
    NUM_PAIRS = 2000 
    SEG_LEN = 25
    TEACHER_SKIP = 0.05

    trajectories = load_rnd_dataset_as_trajectories(
        dataset_path, domain_name, expl_agent, task, num_episodes=5000
    )
    
    if not trajectories:
        print("❌ No trajectories loaded. Check path.")
        exit()

    print(f"\n📊 Total Trajectories Loaded: {len(trajectories)}")
    rewards = [np.sum(traj["rewards"]) for traj in trajectories]
    print(f"Global Avg Reward: {np.mean(rewards):.2f}")

    segment_pool = sample_segment_pool(
        trajectories, 
        pool_size=POOL_SIZE, 
        seg_len=SEG_LEN
    )

    pref_dataset = build_pref_dataset_from_pool(
        segment_pool, 
        num_pairs=NUM_PAIRS, 
        teacher_eps_skip=TEACHER_SKIP
    )

    save_path = f"zero_shot/{domain_name}-{task}_pref_dataset_Pool{POOL_SIZE}_K{SEG_LEN}.pkl"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    batch = save_pref_dataset(pref_dataset, save_path)
