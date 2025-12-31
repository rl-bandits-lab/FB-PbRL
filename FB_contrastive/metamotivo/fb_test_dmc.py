import torch
import numpy as np
import dataclasses
from metamotivo.fb_contrastive_finetune import FBAgent, FBAgentConfig
from metamotivo.nn_models import eval_mode
from dm_control import suite
from pathlib import Path
import pickle
import argparse
import os
import random
from metamotivo.buffers.buffers import OfflineReplayBuffer
from tqdm import tqdm
import mujoco
from dmc_tasks import dmc
import imageio

# -----------------
# Dataset loaders
# -----------------
def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_rnd_dataset(dataset_path: str, domain_name: str, expl_agent: str, num_episodes: int = 1) -> dict:
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
                storage[k][k1] = np.concatenate(storage[k][k1])
        else:
            storage[k] = np.concatenate(storage[k])
    return storage


def load_lire_dataset(base_path: str, env_name: str, data_quality: float = 1.0) -> dict:
    dataset = dict()
    for seed in range(3):
        path = os.path.join(base_path, f"{env_name}/saved_replay_buffer_1000000_seed{seed}.pkl")
        with open(path, "rb") as f:
            load_dataset = pickle.load(f)
        for key in load_dataset.keys():
            load_dataset[key] = load_dataset[key][0:int(data_quality * 100_000)]
        load_dataset["terminals"] = load_dataset["dones"][0:int(data_quality * 100_000)]
        load_dataset.pop("dones", None)
        for key in load_dataset.keys():
            if key not in dataset:
                dataset[key] = load_dataset[key]
            else:
                dataset[key] = np.concatenate((dataset[key], load_dataset[key]), axis=0)
    dataset["rewards"] = dataset["rewards"].reshape(-1)
    dataset["terminals"] = dataset["terminals"].reshape(-1)
    storage = {
        "observation": dataset["observations"].astype(np.float32),
        "action": dataset["actions"].astype(np.float32),
        "reward": dataset["rewards"].astype(np.float32),
        "next": {
            "observation": dataset["next_observations"].astype(np.float32),
            "terminated": dataset["terminals"].astype(np.bool_),
        },
    }
    return storage


# -----------------
# Config
# -----------------
@dataclasses.dataclass
class TestConfig:
    seed: int = 0
    domain_name: str = "walker"
    task_name: str = "walk"
    dataset_path: str = "./"
    expl_agent: str = "rnd"
    device: str = "cuda"
    checkpoint: str = None
    num_eval_episodes: int = 10
    load_n_episodes: int = 5000


# -----------------
# Preference dataset builder
# -----------------
def build_pref_dataset(trajectories, num_pairs=2000, seg_len=200, teacher_eps_skip=0.1):
    all_rewards = np.concatenate([traj["rewards"] for traj in trajectories], axis=0)
    mean_reward = np.mean(all_rewards)
    skip_margin = mean_reward * teacher_eps_skip * seg_len

    seg1_obs, seg1_act, seg1_next, seg1_rew = [], [], [], []
    seg2_obs, seg2_act, seg2_next, seg2_rew = [], [], [], []
    labels, starts1, starts2 = [], [], []
    n_skip, num_trajs = 0, len(trajectories)

    while len(labels) < num_pairs:
        traj1 = trajectories[np.random.randint(num_trajs)]
        traj2 = trajectories[np.random.randint(num_trajs)]

        if len(traj1["observations"]) < seg_len or len(traj2["observations"]) < seg_len:
            continue

        start1 = np.random.randint(0, len(traj1["observations"]) - seg_len + 1)
        start2 = np.random.randint(0, len(traj2["observations"]) - seg_len + 1)

        seg1_obs.append(traj1["observations"][start1:start1 + seg_len])
        seg1_act.append(traj1["actions"][start1:start1 + seg_len])
        seg1_next.append(traj1["next_observations"][start1:start1 + seg_len])
        seg1_rew.append(traj1["rewards"][start1:start1 + seg_len])

        seg2_obs.append(traj2["observations"][start2:start2 + seg_len])
        seg2_act.append(traj2["actions"][start2:start2 + seg_len])
        seg2_next.append(traj2["next_observations"][start2:start2 + seg_len])
        seg2_rew.append(traj2["rewards"][start2:start2 + seg_len])

        R1, R2 = np.sum(seg1_rew[-1]), np.sum(seg2_rew[-1])
        if abs(R1 - R2) < skip_margin:
            n_skip += 1
            label = np.array([0.5, 0.5], dtype=np.float32)
        elif R1 > R2:
            label = np.array([1.0, 0.0], dtype=np.float32)
        else:
            label = np.array([0.0, 1.0], dtype=np.float32)

        labels.append(label)
        starts1.append(start1)
        starts2.append(start2)

    pref_dataset = {
        "labels": np.stack(labels, axis=0).astype(np.float32),
        "observations": np.stack(seg1_obs, axis=0).astype(np.float32),
        "actions": np.stack(seg1_act, axis=0).astype(np.float32).squeeze(2),
        "next_observations": np.stack(seg1_next, axis=0).astype(np.float32),
        "rewards": np.stack(seg1_rew, axis=0).astype(np.float32),
        "observations_2": np.stack(seg2_obs, axis=0).astype(np.float32),
        "actions_2": np.stack(seg2_act, axis=0).astype(np.float32).squeeze(2),
        "next_observations_2": np.stack(seg2_next, axis=0).astype(np.float32),
        "rewards_2": np.stack(seg2_rew, axis=0).astype(np.float32),
        "timestep_1": np.tile(np.arange(seg_len, dtype=np.int32), (num_pairs, 1)),
        "timestep_2": np.tile(np.arange(seg_len, dtype=np.int32), (num_pairs, 1)),
        "start_indices": np.array(starts1, dtype=np.int32),
        "start_indices_2": np.array(starts2, dtype=np.int32),
    }

    print(f"Skip ratio: {n_skip}/{len(labels)} = {n_skip/len(labels):.2%}")
    print(f"[INFO] Built preference dataset: {len(labels)} pairs")
    for k, v in pref_dataset.items():
        print(f"  {k:20s}: shape={v.shape}, dtype={v.dtype}")

    return pref_dataset


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

        data = load_rnd_dataset(cfg.dataset_path, cfg.domain_name, cfg.expl_agent, cfg.load_n_episodes)
        self.replay_buffer = {"train": OfflineReplayBuffer(data, device=self.agent.device)}

    def reward_inference(self, task):
        #env = suite.load(domain_name=self.cfg.domain_name, task_name=task,
        #                 environment_kwargs={"flat_observation": True})
        env = dmc.make(f"{self.cfg.domain_name}_{task}")
        num_samples = 50000
        batch = self.replay_buffer["train"].sample(num_samples)
        rewards = []
        for i in range(num_samples):
            with env._physics.reset_context():
                env._physics.set_state(batch["next"]["physics"][i].cpu().numpy())
                env._physics.set_control(batch["action"][i].cpu().detach().numpy())
            mujoco.mj_forward(env._physics.model.ptr, env._physics.data.ptr)
            mujoco.mj_fwdPosition(env._physics.model.ptr, env._physics.data.ptr)
            mujoco.mj_sensorVel(env._physics.model.ptr, env._physics.data.ptr)
            mujoco.mj_subtreeVel(env._physics.model.ptr, env._physics.data.ptr)
            rewards.append(env._task.get_reward(env._physics))
        rewards = np.array(rewards).reshape(-1, 1)
        z = self.agent._model.reward_inference(
            next_obs=batch["next"]["observation"],
            reward=torch.tensor(rewards, dtype=torch.float32, device=self.agent.device),
        )
        return z.reshape(1, -1)

    def run_eval(self, z, return_traj=False, record_video=False, video_dir="videos"):

        eval_env = dmc.make(f"{self.cfg.domain_name}_{self.cfg.task_name}")
        num_ep = self.cfg.num_eval_episodes

        total_reward, ep_lengths = np.zeros(num_ep), np.zeros(num_ep, dtype=np.int32)
        trajectories = []

        if record_video:
            os.makedirs(video_dir, exist_ok=True)

        for ep in range(num_ep):
            ts = eval_env.reset()

            # buffer for video
            frames = []

            traj = {"observations": [], "next_observations": [], "actions": [],
                    "rewards": [], "physics": [], "next_physics": []}

            steps = 0

            # save first frame
            if record_video:
                frames.append(eval_env.physics.render(height=480, width=640, camera_id=0))

            while not ts.last():
                with torch.no_grad(), eval_mode(self.agent._model):
                    obs_tensor = torch.tensor(ts.observation.reshape(1, -1),
                                            device=self.agent.device,
                                            dtype=torch.float32)
                    action = self.agent.act(obs_tensor, z=z, mean=True).cpu().numpy()

                physics_before = eval_env._physics.get_state().copy()
                ts_next = eval_env.step(action)
                physics_after = eval_env._physics.get_state().copy()

                total_reward[ep] += ts_next.reward

                if return_traj:
                    traj["observations"].append(ts.observation)
                    traj["next_observations"].append(ts_next.observation)
                    traj["actions"].append(action)
                    traj["rewards"].append(ts_next.reward)
                    traj["physics"].append(physics_before)
                    traj["next_physics"].append(physics_after)

                # 🔥 add frame
                if record_video:
                    frame = eval_env.physics.render(height=480, width=640, camera_id=0)
                    frames.append(frame)

                ts = ts_next
                steps += 1

            ep_lengths[ep] = steps

            # 🔥 save mp4
            if record_video:
                video_path = os.path.join(video_dir, f"ep_{ep}.mp4")
                writer = imageio.get_writer(video_path, fps=30)
                for f in frames:
                    writer.append_data(f)
                writer.close()
                print(f"[VIDEO] Saved episode {ep} -> {video_path}")

            if return_traj:
                for k in traj:
                    traj[k] = np.array(traj[k])
                trajectories.append(traj)

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
        z = self.reward_inference(self.cfg.task_name).reshape(1, -1)

        # Evaluate
        res, trajs = self.run_eval(z, return_traj=True, record_video=True, video_dir="videos/cheetah_walk_finetune")
        all_trajs.extend(trajs)
        print(f"[EVAL] mean_reward={res['reward']:.2f}, std_reward={res['reward#std']:.2f}")



# -----------------
# Main
# -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain_name", type=str, default="walker")
    parser.add_argument("--task_name", type=str, default="walk")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--expl_agent", type=str, default="rnd")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_eval_episodes", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = TestConfig(
        domain_name=args.domain_name,
        task_name=args.task_name,
        dataset_path=args.dataset_path,
        checkpoint=args.checkpoint,
        device=args.device,
        num_eval_episodes=args.num_eval_episodes,
        expl_agent=args.expl_agent,
    )

    ws = Workspace(cfg, FBAgentConfig())
    ws.test()
