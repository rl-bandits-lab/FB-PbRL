# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations
import torch

torch.set_float32_matmul_precision("high")

import numpy as np
import dataclasses
#from metamotivo.buffers_dmc.buffers import DictBuffer
from metamotivo.buffers.buffers import OfflineReplayBuffer
from metamotivo.fb_antmaze import FBAgent, FBAgentConfig
from metamotivo.nn_models import eval_mode
from tqdm import tqdm
import time
from dm_control import suite
import random
from pathlib import Path
import wandb
import json
from typing import List
import mujoco
import warnings
import tyro
from dmc_tasks import dmc
import os
import pickle
import gym
import d4rl

ALL_TASKS = {
    "walker": [
        "walk",
        "run",
        "stand",
    ],
    #"cheetah": ["walk", "run"],
    "cheetah": ["run", "run_backward", "walk","walk_backward"],
    "quadruped": ["walk", "run", "stand", "jump"],
}

def load_antmaze_dataset(env_name: str) -> dict:
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)

    obs = dataset["observations"]
    next_obs = dataset["next_observations"]
    rewards = dataset["rewards"]
    terminals = dataset["terminals"].astype(bool)

    max_steps = env._max_episode_steps
    N = len(obs)

    timeout = np.zeros(N, dtype=bool)
    step = 0
    for i in range(N):
        step += 1
        if terminals[i] or step == max_steps:
            timeout[i] = (not terminals[i])
            step = 0

    terminated = np.expand_dims(np.logical_or(terminals, timeout), axis=1)
    return {
        "observation": obs.astype(np.float32),
        "action": dataset["actions"].astype(np.float32),
        "reward": rewards.astype(np.float32).reshape(-1, 1),
        "next": {
            "observation": next_obs.astype(np.float32),
            "terminated": terminated,
        },
    }

def create_agent(
    domain_name="walker",
    task_name="walk",
    device="cpu",
    compile=False,
    cudagraphs=False,
    use_z_noise=False,
    z_noise_std=0.1,
    z_noise_clip=0.3,
    antmaze_env: str | None = None,   # >>> ANTMAZE
    ortho_coef=100.0,
) -> FBAgent:
    if domain_name not in ["walker", "pointmass", "cheetah", "quadruped", "antmaze"]:
        raise RuntimeError('FB configuration defined only for "walker", "pointmass", "cheetah", "quadruped"')
    '''env = suite.load(
        domain_name=domain_name,
        task_name=task_name,
        environment_kwargs={"flat_observation": True},
    )'''
    if domain_name == "antmaze":
        env = gym.make(antmaze_env)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
    else:
        env = dmc.make(f"{domain_name}_{task_name}")
        obs_dim = env.observation_spec().shape[0]
        act_dim = env.action_spec().shape[0]

    agent_config = FBAgentConfig()
    #agent_config.model.obs_dim = env.observation_spec()["observations"].shape[0]
    #agent_config.model.obs_dim = env.observation_spec().shape[0]
    #agent_config.model.action_dim = env.action_spec().shape[0]
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
    if domain_name == "antmaze":
        agent_config.model.norm_obs = True
        agent_config.model.archi.z_dim = 50
    #agent_config.model.archi.z_dim = 16

    agent_config.model.archi.b.norm = True
    agent_config.model.archi.norm_z = True
    agent_config.model.archi.b.hidden_dim = 512
    agent_config.model.archi.f.hidden_dim = 512
    agent_config.model.archi.actor.hidden_dim = 512
    agent_config.model.archi.f.hidden_layers = 2
    agent_config.model.archi.actor.hidden_layers = 2
    agent_config.model.archi.b.hidden_layers = 4
    # optim
    if domain_name == "pointmass":
        agent_config.train.lr_f = 1e-4
        agent_config.train.lr_b = 1e-6
        agent_config.train.lr_actor = 1e-6
    else:
        agent_config.train.lr_f = 1e-4
        agent_config.train.lr_b = 1e-4
        agent_config.train.lr_actor = 1e-4
    agent_config.train.ortho_coef = ortho_coef
    agent_config.train.train_goal_ratio = 0.5
    agent_config.train.fb_pessimism_penalty = 0
    agent_config.train.actor_pessimism_penalty = 0.5

    if domain_name in ["pointmass", "antmaze"] :
        agent_config.train.discount = 0.99
    else:
        agent_config.train.discount = 0.98
    agent_config.compile = compile
    agent_config.cudagraphs = cudagraphs

    agent_config.train.use_z_noise = use_z_noise
    agent_config.train.z_noise_std = z_noise_std
    agent_config.train.z_noise_clip = z_noise_clip

    return agent_config


'''def load_data(dataset_path, expl_agent, domain_name, num_episodes=1):
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
'''

'''def load_rnd_dataset(dataset_path: str, domain_name: str, expl_agent: str, num_episodes: int = 1) -> dict:
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
    return storage'''

def load_rnd_dataset(dataset_path: str, domain_name: str, expl_agent: str, num_episodes: int = 1) -> dict:
    path = Path(dataset_path) / f"{domain_name}/{expl_agent.split('+')[0]}/buffer"
    print(f"Data path: {path}")

    storage = {
        "observation": [],
        "action": [],
        "physics": [],
        "next": {
            "observation": [],
            "terminated": [],
            "physics": [],
        },
    }

    # -----------------------------
    # Helper: load one episode file
    # -----------------------------
    def load_npz_episode(f):
        data = np.load(str(f))
        obs = data["observation"][:-1].astype(np.float32)
        act = data["action"][1:].astype(np.float32)
        phys = data["physics"][:-1]
        next_obs = data["observation"][1:].astype(np.float32)
        next_phys = data["physics"][1:]
        terminated = (1 - data["discount"][1:]).astype(bool)
        return obs, act, phys, next_obs, next_phys, terminated

    def load_pkl_episode(f):
        import pickle
        with open(f, "rb") as fp:
            data = pickle.load(fp)

        obs = data["observation"].astype(np.float32)
        act = data["action"].astype(np.float32)
        phys = data["physics"].astype(np.float32)

        next_obs = data["next"]["observation"].astype(np.float32)
        next_phys = data["next"]["physics"].astype(np.float32)
        terminated = data["next"]["terminated"].astype(bool)

        return obs, act, phys, next_obs, next_phys, terminated

    # -----------------------------
    # Case A: rnd-only or expert-only
    # -----------------------------
    if expl_agent != "rnd+expert":
        if expl_agent == "expert":
            files = list((Path(dataset_path) / f"{domain_name}/expert/buffer").glob("*.pkl"))
            loader = load_pkl_episode
        else:
            files = list((Path(dataset_path) / f"{domain_name}/{expl_agent}/buffer").glob("*.npz"))
            loader = load_npz_episode

        num = min(num_episodes, len(files))
        for f in tqdm(files[:num]):
            obs, act, phys, next_obs, next_phys, term = loader(f)

            storage["observation"].append(obs)
            storage["action"].append(act)
            storage["physics"].append(phys)
            storage["next"]["observation"].append(next_obs)
            storage["next"]["physics"].append(next_phys)
            storage["next"]["terminated"].append(term)

    # -----------------------------
    # Case B: rnd+expert (50% each)
    # -----------------------------
    else:
        rnd_path = Path(dataset_path) / f"{domain_name}/rnd/buffer"
        exp_path = Path(dataset_path) / f"{domain_name}/expert/buffer"

        rnd_files = list(rnd_path.glob("*.npz"))
        exp_files = list(exp_path.glob("*.pkl"))

        if len(rnd_files) == 0:
            raise FileNotFoundError("No RND npz found")
        if len(exp_files) == 0:
            raise FileNotFoundError("No expert pkl found")

        N_rnd = num_episodes // 2
        N_exp = num_episodes - N_rnd

        print(f"Loading {N_rnd} RND episodes + {N_exp} expert episodes")

        # Load RND episodes
        for f in tqdm(rnd_files[:N_rnd], desc="RND episodes"):
            obs, act, phys, next_obs, next_phys, term = load_npz_episode(f)
            storage["observation"].append(obs)
            storage["action"].append(act)
            storage["physics"].append(phys)
            storage["next"]["observation"].append(next_obs)
            storage["next"]["physics"].append(next_phys)
            storage["next"]["terminated"].append(term)

        # Load expert episodes
        for f in tqdm(exp_files[:N_exp], desc="Expert episodes"):
            obs, act, phys, next_obs, next_phys, term = load_pkl_episode(f)
            storage["observation"].append(obs)
            storage["action"].append(act)
            storage["physics"].append(phys)
            storage["next"]["observation"].append(next_obs)
            storage["next"]["physics"].append(next_phys)
            storage["next"]["terminated"].append(term)

    # -----------------------------
    # Concatenate
    # -----------------------------
    for k in storage:
        if k == "next":
            for k1 in storage[k]:
                storage[k][k1] = np.concatenate(storage[k][k1], axis=0)
        else:
            storage[k] = np.concatenate(storage[k], axis=0)

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
        },}

    return storage

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@dataclasses.dataclass
class TrainConfig:
    dataset_root: str
    seed: int = 0
    dataset_type: str = "rnd"  # "rnd" or "lire"
    domain_name: str = "walker"
    task_name: str | None = None
    dataset_expl_agent: str = "rnd"
    num_train_steps: int = 3_000_000
    load_n_episodes: int = 5_000
    log_every_updates: int = 10_000
    work_dir: str | None = None

    checkpoint_every_steps: int = 1_000_000

    # eval
    num_eval_episodes: int = 10
    num_inference_samples: int = 50_000
    eval_every_steps: int = 100_000
    eval_tasks: List[str] | None = None

    # misc
    compile: bool = False
    cudagraphs: bool = False
    device: str = "cuda"

    # WANDB
    use_wandb: bool = False
    wandb_ename: str | None = None
    wandb_gname: str | None = None
    wandb_pname: str | None = "fb_train_dmc"
    wandb_name_prefix: str | None = None

    use_z_noise: bool = False
    z_noise_std: float = 0.1
    z_noise_clip: float = 0.3

    dataset_quality: float = 1.0

    antmaze_env: str | None = None
    ortho_coef: float = 100.0

    def __post_init__(self):
        if self.eval_tasks is None:
            if self.domain_name == "antmaze":
                self.eval_tasks = [None]
            else:
                self.eval_tasks = ALL_TASKS[self.domain_name]


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
        self.work_dir = Path(self.work_dir)
        self.work_dir.mkdir(exist_ok=True, parents=True)
        print(f"working dir: {self.work_dir}")

        self.agent = FBAgent(**dataclasses.asdict(self.agent_cfg))
        set_seed_everywhere(self.cfg.seed)

        if self.cfg.use_wandb:
            exp_name = "fb"
            wandb_name = exp_name
            if self.cfg.wandb_name_prefix:
                wandb_name = f"{self.cfg.wandb_name_prefix}_{exp_name}"
            # fmt: off
            wandb_config = dataclasses.asdict(self.cfg)
            wandb.init(
                entity=self.cfg.wandb_ename,
                project=self.cfg.wandb_pname,
                group=self.cfg.wandb_gname,
                name=wandb_name,
                config=wandb_config,
            )
            # fmt: on

        with (self.work_dir / "config.json").open("w") as f:
            json.dump(dataclasses.asdict(self.cfg), f, indent=4)

    def train(self):
        self.start_time = time.time()
        self.train_offline()

    def train_offline(self) -> None:
        self.replay_buffer = {}
        # LOAD DATA FROM EXORL
        '''data = load_data(
            self.cfg.dataset_root,
            self.cfg.dataset_expl_agent,
            self.cfg.domain_name,
            self.cfg.load_n_episodes,
        )'''

        if self.cfg.dataset_type == "rnd":
            data = load_rnd_dataset(self.cfg.dataset_root, self.cfg.domain_name, self.cfg.dataset_expl_agent, self.cfg.load_n_episodes)
        elif self.cfg.dataset_type == "lire":
            base_path = os.path.join(self.cfg.dataset_root, "LIRE_dmc_preference_dataset")
            data = load_lire_dataset(base_path, f"{self.cfg.domain_name}-{self.cfg.task_name}", data_quality=self.cfg.dataset_quality)
        elif self.cfg.dataset_type == "antmaze":
            assert self.cfg.antmaze_env is not None
            data = load_antmaze_dataset(self.cfg.antmaze_env)
        #self.replay_buffer = {"train": DictBuffer(capacity=data["observation"].shape[0], device=self.agent.device)}
        self.replay_buffer = {"train": OfflineReplayBuffer(data, device=self.agent.device)}
        #self.replay_buffer["train"].extend(data)
        print(self.replay_buffer["train"])
        del data

        total_metrics = None
        fps_start_time = time.time()
        for t in tqdm(range(0, int(self.cfg.num_train_steps))):
            if t % self.cfg.eval_every_steps == 0:
                self.eval(t)

            # torch.compiler.cudagraph_mark_step_begin()
            metrics = self.agent.update(self.replay_buffer, t)

            # we need to copy tensors returned by a cudagraph module
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
                        {f"train/{k}": v for k, v in m_dict.items()},
                        step=t,
                    )
                print(m_dict)
                total_metrics = None
                fps_start_time = time.time()
            if t % self.cfg.checkpoint_every_steps == 0:
                self.agent.save(str(self.work_dir / "checkpoint"))
        self.agent.save(str(self.work_dir / "checkpoint"))
        return

    def eval(self, t):
        if self.cfg.dataset_type == "antmaze":
            z = self.reward_inference("antmaze").reshape(1, -1)
            env = gym.make(self.cfg.antmaze_env)

            returns = []
            normalized_scores = []
            
            for _ in range(self.cfg.num_eval_episodes):
                obs, done = env.reset(), False
                if isinstance(obs, tuple): obs = obs[0]
                    
                ep_ret = 0.0
                steps = 0
                max_steps = env._max_episode_steps
                
                while not done and steps < max_steps:
                    with torch.no_grad(), eval_mode(self.agent._model):
                        obs_t = torch.tensor(obs, device=self.agent.device).float().unsqueeze(0)
                        action = self.agent.act(obs=obs_t, z=z, mean=True).cpu().numpy()[0]
                    
                    obs, reward, done, info = env.step(action)[:4]
                    ep_ret += reward
                    steps += 1
                
                returns.append(ep_ret)
                normalized_scores.append(env.get_normalized_score(ep_ret) * 100.0)

            m_dict = {
                "return": np.mean(returns),
                "return#std": np.std(returns),
                "normalized_score": np.mean(normalized_scores),
                "normalized_score#std": np.std(normalized_scores),
            }
            if self.cfg.use_wandb:
                wandb.log({f"antmaze/{k}": v for k, v in m_dict.items()}, step=t)
            print(f"Step {t}: {m_dict}")
            return
        for task in self.cfg.eval_tasks:
            z = self.reward_inference(task).reshape(1, -1)
            '''eval_env = suite.load(
                domain_name=self.cfg.domain_name,
                task_name=task,
                environment_kwargs={"flat_observation": True},
            )'''
            eval_env = dmc.make(f"{self.cfg.domain_name}_{task}")
            
            num_ep = self.cfg.num_eval_episodes
            total_reward = np.zeros((num_ep,), dtype=np.float64)
            for ep in range(num_ep):
                time_step = eval_env.reset()
                while not time_step.last():
                    with torch.no_grad(), eval_mode(self.agent._model):
                        obs = torch.tensor(
                            #time_step.observation["observations"].reshape(1, -1),
                            time_step.observation.reshape(1, -1),
                            device=self.agent.device,
                            dtype=torch.float32,
                        )
                        action = self.agent.act(obs=obs, z=z, mean=True).cpu().numpy()
                    time_step = eval_env.step(action)
                    total_reward[ep] += time_step.reward
            m_dict = {
                "reward": np.mean(total_reward),
                "reward#std": np.std(total_reward),
            }
            if self.cfg.use_wandb:
                wandb.log(
                    {f"{task}/{k}": v for k, v in m_dict.items()},
                    step=t,
                )
            m_dict["task"] = task
            print(m_dict)

    def reward_inference(self, task) -> torch.Tensor:
        
        '''env = suite.load(
            domain_name=self.cfg.domain_name,
            task_name=task,
            environment_kwargs={"flat_observation": True},
        )'''
        #env = dmc.make(f"{self.cfg.domain_name}_{task}")
        num_samples = self.cfg.num_inference_samples
        batch = self.replay_buffer["train"].sample(num_samples)
        
        if self.cfg.dataset_type == "rnd":
            rewards = []
            for i in range(num_samples):
                with env._physics.reset_context():
                    env._physics.set_state(batch["next"]["physics"][i].cpu().numpy())
                    env._physics.set_control(batch["action"][i].cpu().detach().numpy())
                mujoco.mj_forward(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
                mujoco.mj_fwdPosition(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
                mujoco.mj_sensorVel(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
                mujoco.mj_subtreeVel(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
                rewards.append(env._task.get_reward(env._physics))
            rewards = np.array(rewards).reshape(-1, 1)
        else:
            rewards = batch["reward"]
        z = self.agent._model.reward_inference(
            next_obs=batch["next"]["observation"],
            reward=torch.tensor(rewards, dtype=torch.float32, device=self.agent.device),
        )
        return z


if __name__ == "__main__":
    config = tyro.cli(TrainConfig)

    warnings.warn(
        "Since the original creation of ExORL, mujoco has seen many updates. To rerun all the actions and collect a physics consistent data, you may optionally use the update_data.py utility from MTM (https://github.com/facebookresearch/mtm/tree/main/research/exorl)."
    )
    if config.task_name is None:
        if config.domain_name == "walker":
            config.task_name = "walk"
        elif config.domain_name == "cheetah":
            config.task_name = "run"
        elif config.domain_name == "pointmass":
            config.task_name = "reach_top_left"
        elif config.domain_name == "quadruped":
            config.task_name = "run"
        elif config.domain_name == "antmaze":
            config.task_name = "dummy"
        else:
            raise RuntimeError("Unsupported domain, you need to specify task_name")
    agent_config = create_agent(
        domain_name=config.domain_name,
        task_name=config.task_name,
        device=config.device,
        compile=config.compile,
        cudagraphs=config.cudagraphs,
        use_z_noise=config.use_z_noise,
        z_noise_std=config.z_noise_std,
        z_noise_clip=config.z_noise_clip,
        antmaze_env=config.antmaze_env,   # >>> ANTMAZE
        ortho_coef=config.ortho_coef,
    )

    ws = Workspace(config, agent_cfg=agent_config)
    ws.train()