

from __future__ import annotations
import torch


torch.set_float32_matmul_precision("high")


import numpy as np
import dataclasses
from metamotivo.buffers.buffers import OfflineReplayBuffer
from metamotivo.fb_contrastive_finetune import FBAgent, FBAgentConfig
from metamotivo.nn_models import eval_mode
from tqdm import tqdm
import time
from dm_control import suite
import random
from pathlib import Path
import wandb
import json
from typing import List, Optional
import argparse
import os
import pickle
#from dmc_tasks import dmc
from url_benchmark import dmc
import tyro

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
        },}

    return storage

def inject_preference_noise(labels: np.ndarray, noise: float):
    """
    labels: shape (N, 2), entries in {0, 1, 0.5}
    """
    if noise <= 0.0:
        return labels

    p = noise
    for i in range(len(labels)):
        if random.random() >= p:
            continue

        a, b = labels[i][0], labels[i][1]

        # case 1: (1, 0)
        if a == 1 and b == 0:
            if random.random() < 0.5:
                labels[i][0], labels[i][1] = 0, 1
            else:
                labels[i][0], labels[i][1] = 0.5, 0.5

        # case 2: (0, 1)
        elif a == 0 and b == 1:
            if random.random() < 0.5:
                labels[i][0], labels[i][1] = 1, 0
            else:
                labels[i][0], labels[i][1] = 0.5, 0.5

        # case 3: tie (0.5, 0.5)
        else:
            if random.random() < 0.5:
                labels[i][0], labels[i][1] = 1, 0
            else:
                labels[i][0], labels[i][1] = 0, 1

    return labels

def load_preference_dataset(
    domain_task: str,
    dataset_path: str,
    dataset_type: str,
    segment_length: int,
    num_pairs: int | None = None,
    noise: float = 0.0,
) -> dict:
    pref_dir = os.path.join(dataset_path, dataset_type)
    if segment_length == 200:
        fname = f"{domain_task}_pref_dataset.pkl"
    else:
        fname = f"{domain_task}_pref_dataset_K_{segment_length}.pkl"
    fpath = os.path.join(pref_dir, fname)

    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Preference dataset not found: {fpath}")

    with open(fpath, "rb") as f:
        batch = pickle.load(f)

    # -----------------
    # Slice pairs (forward order)
    # -----------------
    N, K, *_ = batch["next_observations"].shape
    if num_pairs is not None:
        num_pairs = min(num_pairs, N)
        batch = {
            "next_observations": batch["next_observations"][:num_pairs],
            "next_observations_2": batch["next_observations_2"][:num_pairs],
            "labels": batch["labels"][:num_pairs],
        }
    labels = batch["labels"]
    if noise > 0.0:
        labels = inject_preference_noise(labels, noise)
        batch["labels"] = labels
        print(f"[PreferenceDataset] Injected noise p={noise}")
        
    print(
        f"[PreferenceDataset] Loaded {len(batch['labels'])}/{N} pairs, "
        f"segment_length={K}"
    )

    return batch

def set_seed_everywhere(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# -----------------
# Agent config (DMC)
# -----------------

def create_agent(
    domain_name="walker",
    task_name="walk",
    device="cuda",
    compile=False,
    cudagraphs=False,
    use_contrastive=False,
    use_dynamic_contrastive_z=False,
    contrastive_coef=1.0,
    quad_loss_coef=1.0,
    reg_coefficient=0.0,
    q_loss_coef=0.0,
    batch_size_contrastive=256,
    seq_length=200
):
    #env = suite.load(domain_name=domain_name, task_name=task_name, environment_kwargs={"flat_observation": True})
    #env = dmc.make(f"{domain_name}_{task_name}")
    env = dmc.make(
    f"{domain_name}_{task_name}",
    obs_type="states",
    frame_stack=1,
    action_repeat=1,   # ← 跟 dataset 一致
    seed=0,
)
    agent_config = FBAgentConfig()
    #agent_config.model.obs_dim = env.observation_spec()["observations"].shape[0]
    agent_config.model.obs_dim = env.observation_spec().shape[0]
    agent_config.model.action_dim = env.action_spec().shape[0]
    agent_config.model.norm_obs = False
    agent_config.model.seq_length = 1
    agent_config.model.archi.b.norm = True
    agent_config.model.archi.norm_z = True
    agent_config.model.archi.b.hidden_dim = 256
    agent_config.model.archi.f.hidden_dim = 1024
    agent_config.model.archi.actor.hidden_dim = 1024
    agent_config.model.archi.f.hidden_layers = 1
    agent_config.model.archi.actor.hidden_layers = 1
    agent_config.model.archi.b.hidden_layers = 2
    agent_config.train.ortho_coef = 1
    agent_config.train.train_goal_ratio = 0.5
    agent_config.train.fb_pessimism_penalty = 0
    agent_config.train.actor_pessimism_penalty = 0.5
    agent_config.train.discount = 0.98

    return agent_config

def apply_train_cfg_to_agent(
    agent_cfg: FBAgentConfig,
    train_cfg: TrainConfig,
):
    # ----- model -----
    agent_cfg.model.archi.z_dim = train_cfg.z_dim

    # ----- training -----
    agent_cfg.train.batch_size = train_cfg.batch_size
    agent_cfg.train.batch_size_contrastive = train_cfg.batch_size_contrastive
    agent_cfg.train.seq_length = train_cfg.seq_length

    agent_cfg.train.lr_f = train_cfg.lr_f
    agent_cfg.train.lr_b = train_cfg.lr_b
    agent_cfg.train.lr_actor = train_cfg.lr_actor

    agent_cfg.train.contrastive_coef = train_cfg.contrastive_coef
    agent_cfg.train.quad_loss_coef = train_cfg.quad_loss_coef
    agent_cfg.train.q_loss_coef = train_cfg.q_loss_coef

    agent_cfg.use_contrastive = train_cfg.use_contrastive
    agent_cfg.train.use_dynamic_contrastive_z = train_cfg.use_dynamic_contrastive_z

    agent_cfg.compile = train_cfg.compile
    agent_cfg.cudagraphs = train_cfg.cudagraphs

@dataclasses.dataclass
class TrainConfig:
    # --- runtime / env ---
    seed: int = 0
    domain_name: str = "walker"
    task_name: str = "walk"
    device: str = "cuda"

    # --- dataset ---
    dataset_type: str = "rnd"
    dataset_path: str = "./"
    expl_agent: str = "rnd"
    load_n_episodes: int = 5000
    dataset_quality: float = 1.0
    num_pairs: int  = 2000

    # --- training loop ---
    num_train_steps: int = 1_000_000
    log_every_updates: int = 10_000
    eval_every_steps: int = 100_000
    checkpoint_every_steps: int = 100_000

    # --- eval ---
    num_eval_episodes: int = 10
    num_inference_samples: int = 50_000
    eval_tasks: Optional[List[str]] = None

    # --- algorithm knobs (CLI-adjustable) ---
    use_contrastive: bool = False
    use_dynamic_contrastive_z: bool = False
    batch_size: int = 1024
    batch_size_contrastive: int = 256
    seq_length: int = 200
    lr_f: float = 1e-4
    lr_b: float = 1e-4
    lr_actor: float = 1e-4
    contrastive_coef: float = 10.0
    quad_loss_coef: float = 0.0
    q_loss_coef: float = 0.0
    z_dim: int = 16
    work_dir: str | None = None

    # --- misc ---
    compile: bool = False
    cudagraphs: bool = False
    load_dir: str | None = None

    # --- wandb ---
    use_wandb: bool = False
    wandb_ename: Optional[str] = None
    wandb_gname: Optional[str] = None
    wandb_pname: Optional[str] = "fb_train_dmc"
    wandb_name_prefix: Optional[str] = None

    noise: float = 0.0

    def __post_init__(self):
        if self.eval_tasks is None:
            self.eval_tasks = [self.task_name]


# -----------------
# Workspace
# -----------------
class Workspace:
    def __init__(self, cfg, agent_cfg):
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
        print("working dir: {}".format(self.work_dir))

        if self.cfg.load_dir is None:
            raise RuntimeError("Finetune requires --checkpoint (pretrained folder)")
        print(f"Loading pretrained model from {self.cfg.load_dir}")
        self.agent = FBAgent.load(self.cfg.load_dir, device=self.cfg.device, override_cfg=self.cfg)
        
        set_seed_everywhere(self.cfg.seed)

        domain_task = f"{self.cfg.domain_name}-{self.cfg.task_name}"
        print(f"[INFO] Loading preference dataset for {domain_task}")
        pref_type = "LIRE_dmc_preference_dataset" if self.cfg.dataset_type == "LIRE" else "rnd_dmc_preference_dataset"
        self.pref_dataset = load_preference_dataset(domain_task, self.cfg.dataset_path, pref_type, self.cfg.seq_length, self.cfg.num_pairs, noise=self.cfg.noise)


        if self.cfg.dataset_type == "rnd":
            data = load_rnd_dataset(self.cfg.dataset_path, self.cfg.domain_name, self.cfg.expl_agent, self.cfg.load_n_episodes)
        else:
            base_path = os.path.join(self.cfg.dataset_path, "LIRE_dmc_preference_dataset")
            data = load_lire_dataset(base_path, f"{self.cfg.domain_name}-{self.cfg.task_name}", data_quality=self.cfg.dataset_quality,)
        self.replay_buffer = {"train": OfflineReplayBuffer(data, device=self.agent.device)}
        print(f"[ReplayBuffer] Loaded transitions = {len(self.replay_buffer['train'])}")
        #self.recon_env = dmc.make(f"{self.cfg.domain_name}_{self.cfg.task_name}")
        self.recon_env = dmc.make(
    f"{self.cfg.domain_name}_{self.cfg.task_name}",
    obs_type="states",
    frame_stack=1,
    action_repeat=1,   # ← 跟 dataset 一致
    seed=0,
)
        self.agent.set_recon_env(self.recon_env)

        if self.cfg.use_wandb:
            exp_name = "fb"
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
                if self.cfg.use_wandb:
                    wandb.log({f"train/{k}": v for k, v in m_dict.items()}, step=t)
                print(m_dict)
                total_metrics = None
                fps_start_time = time.time()
            if t % self.cfg.checkpoint_every_steps == 0:
                self.agent.save(str(self.work_dir / "checkpoint"))
        self.agent.save(str(self.work_dir / "checkpoint"))

    def eval(self, t):
        for task in self.cfg.eval_tasks:
            if self.cfg.use_dynamic_contrastive_z:
                z = self.agent.z.reshape(1, -1)
            else:
                z = self.reward_inference(env_name).reshape(1, -1)
            #eval_env = suite.load(domain_name=self.cfg.domain_name, task_name=task, environment_kwargs={"flat_observation": True})
            #eval_env  = dmc.make(f"{self.cfg.domain_name}_{task}")
            eval_env = dmc.make(
                f"{self.cfg.domain_name}_{task}",
                obs_type="states",
                frame_stack=1,
                action_repeat=1,
                seed=0,
            )
            num_ep = self.cfg.num_eval_episodes
            total_reward = np.zeros((num_ep,), dtype=np.float64)
            ep_lengths = np.zeros((num_ep,), dtype=np.int32)
            for ep in range(num_ep):
                ts = eval_env.reset()
                steps = 0
                while not ts.last():
                    with torch.no_grad(), eval_mode(self.agent._model):
                        #obs_tensor = torch.tensor(ts.observation["observations"].reshape(1, -1), device=self.agent.device, dtype=torch.float32)
                        obs_tensor = torch.tensor(ts.observation.reshape(1, -1), device=self.agent.device, dtype=torch.float32)
                        action = self.agent.act(obs=obs_tensor, z=z, mean=True).cpu().numpy()
                    ts = eval_env.step(action)
                    total_reward[ep] += ts.reward
                    steps += 1
                ep_lengths[ep] = steps
            m_dict = {
                "reward": np.mean(total_reward),
                "reward#std": np.std(total_reward),
                "len": np.mean(ep_lengths),
                "len#std": np.std(ep_lengths),
            }
            if self.cfg.use_wandb:
                wandb.log({f"{self.cfg.domain_name}-{task}/{k}": v for k, v in m_dict.items()}, step=t)
            print(m_dict)
            return m_dict["reward"]

    def reward_inference(self):
        num_samples = self.cfg.num_inference_samples
        batch = self.replay_buffer["train"].sample(num_samples)
        rewards = batch["reward"]
        z = self.agent._model.reward_inference(
            next_obs=batch["next"]["observation"],
            reward=torch.tensor(rewards, dtype=torch.float32, device=self.agent.device),
        )
        return z

if __name__ == "__main__":
    train_cfg = tyro.cli(TrainConfig)

    agent_cfg = create_agent(
        domain_name=train_cfg.domain_name,
        task_name=train_cfg.task_name,
        device=train_cfg.device,
        compile=train_cfg.compile,
        cudagraphs=train_cfg.cudagraphs,
    )

    apply_train_cfg_to_agent(agent_cfg, train_cfg)

    ws = Workspace(train_cfg, agent_cfg=agent_cfg)
    ws.train()
