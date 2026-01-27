import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import mujoco
from dm_control import suite
#from dmc_tasks import dmc
from url_benchmark import dmc

def compute_rewards_from_physics(domain_name: str, task: str, physics_batch, action_batch):

    #env = dmc.make(f"{domain_name}_{task}")
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
    print(f"Loading {num_episodes} episodes..." )

    trajectories = []
    for i in tqdm(range(num_episodes), desc="Loading episodes"):
        f = files[i]
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
            "physics": physics[:-1],
            "next_physics": physics[1:],
        }
        trajectories.append(traj)

    return trajectories

def build_pref_dataset(trajectories, num_pairs=2000, seg_len=200, teacher_eps_skip=0.5):

    all_rewards = np.concatenate([traj["rewards"] for traj in trajectories], axis=0)
    mean_reward = np.mean(all_rewards)
    skip_margin = mean_reward * teacher_eps_skip * seg_len

    pref_dataset = []
    num_trajs = len(trajectories)
    n_skip = 0
    while len(pref_dataset) < num_pairs:

        traj1 = trajectories[np.random.randint(num_trajs)]
        traj2 = trajectories[np.random.randint(num_trajs)]


        if len(traj1['observations']) < seg_len or len(traj2['observations']) < seg_len:
            continue


        start1 = np.random.randint(0, len(traj1['observations']) - seg_len + 1)
        start2 = np.random.randint(0, len(traj2['observations']) - seg_len + 1)

        seg1 = {
            'observations': traj1['observations'][start1:start1+seg_len],
            'next_observations': traj1['next_observations'][start1:start1+seg_len],
            'actions': traj1['actions'][start1:start1+seg_len],
            'rewards': traj1['rewards'][start1:start1+seg_len],
        }
        seg2 = {
            'observations': traj2['observations'][start2:start2+seg_len],
            'next_observations': traj2['next_observations'][start2:start2+seg_len],
            'actions': traj2['actions'][start2:start2+seg_len],
            'rewards': traj2['rewards'][start2:start2+seg_len],
        }


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
            'start1': start1,
            'start2': start2,
            'reward1': R1,
            'reward2': R2,
        })

    print(f"Skip ratio: {n_skip}/{len(pref_dataset)} = {n_skip/len(pref_dataset):.2%}")
    return pref_dataset


def save_pref_dataset(pref_dataset, save_path, seg_len=200):

    num_pairs = len(pref_dataset)

    human_labels = np.zeros((num_pairs, 2), dtype=np.float32)
    seg_obs_1, seg_act_1, seg_next_obs_1 = [], [], []
    seg_obs_2, seg_act_2, seg_next_obs_2 = [], [], []
    seq_timestep_1, seq_timestep_2 = [], []
    start_indices_1, start_indices_2 = [], []
    segment_return_1 = np.zeros((num_pairs,), dtype=np.float32)
    segment_return_2 = np.zeros((num_pairs,), dtype=np.float32)
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

        segment_return_1[i] = pair['reward1']
        segment_return_2[i] = pair['reward2']

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
        rewards=segment_return_1,
        rewards_2=segment_return_2,
    )

    with open(save_path, "wb") as f:
        pickle.dump(batch, f)

    print(f"✅ Preference dataset saved at {save_path}, total pairs = {num_pairs}")
    return batch


# ============================================================
#  Main Execution
# ============================================================
if __name__ == "__main__":
    dataset_path = "./datasets_dmc"
    domain_name = "walker"
    task = "walk"
    expl_agent = "rnd"

    print(task)
    trajectories = load_rnd_dataset_as_trajectories(dataset_path, domain_name, expl_agent, task, num_episodes=5000)

    rewards = [np.sum(traj["rewards"]) for traj in trajectories]
    lengths = [len(traj["rewards"]) for traj in trajectories]

    pref_dataset = build_pref_dataset(trajectories, num_pairs=2000, seg_len=200, teacher_eps_skip=0.05)

    save_path = f"datasets_dmc/rnd_dmc_preference_dataset/{domain_name}-{task}_pref_dataset.pkl"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    batch = save_pref_dataset(pref_dataset, save_path, seg_len=200)

    print(batch.keys())
    print("Labels shape:", batch['labels'].shape)
    print("Obs shape:", batch['observations'].shape)
    print("Next Obs shape:", batch['next_observations'].shape)
    print("Start idx (first 5):", batch['start_indices'][:5])
