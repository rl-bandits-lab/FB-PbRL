import torch
import torch.nn.functional as F
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
import csv
from dmc_tasks import dmc

# -----------------
# Dataset loaders
# -----------------

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_preference_dataset(domain_task: str, dataset_path: str, dataset_type: str):
    pref_dir = os.path.join(dataset_path, dataset_type)
    #fname = f"{domain_task}_pref_dataset.pkl"
    fname = f"{domain_task}_pref_20000.pkl"
    fpath = os.path.join(pref_dir, fname)
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Preference dataset not found: {fpath}")
    with open(fpath, "rb") as f:
        batch = pickle.load(f)
    return batch


# -----------------
# Config
# -----------------
@dataclasses.dataclass
class TestConfig:
    seed: int = 0
    domain_name: str = "walker"
    task_name: str = "walk"
    dataset_type: str = "rnd_dmc_preference_dataset"
    dataset_path: str = "./"
    expl_agent: str = "rnd"
    device: str = "cuda"
    checkpoint: str = None
    num_eval_episodes: int = 10
    load_n_episodes: int = 5000


# -----------------
# Inject modified methods into FBAgent
# -----------------
def encode_expert_contra(self, next_obs: torch.Tensor, batch_size=256):
    B_expert = self._model._backward_map(next_obs).detach()
    B_expert = B_expert.view(batch_size, 200, B_expert.shape[-1])  # N x L x d
    z_expert = B_expert.mean(dim=1)
    z_expert = self._model.project_z(z_expert)
    return z_expert


def compute_preference_loss(self, pref_dataset: dict):
    with eval_mode(self._model):
        batch_size = 256
        seq_length = 200
        device = self.device

        indices = np.random.choice(len(pref_dataset['labels']), size=(batch_size, 2), replace=False)
        indices1, indices2 = indices[:, 0], indices[:, 1]

        # pair1
        next_states1 = torch.FloatTensor(pref_dataset['next_observations'][indices1]).to(device)
        next_states2 = torch.FloatTensor(pref_dataset['next_observations_2'][indices1]).to(device)
        prefs1 = torch.FloatTensor(pref_dataset['labels'][indices1]).to(device)

        # pair2
        next_states1p = torch.FloatTensor(pref_dataset['next_observations'][indices2]).to(device)
        next_states2p = torch.FloatTensor(pref_dataset['next_observations_2'][indices2]).to(device)
        prefs2 = torch.FloatTensor(pref_dataset['labels'][indices2]).to(device)

        # reshape
        next_states1 = next_states1.view(batch_size * seq_length, -1)
        next_states2 = next_states2.view(batch_size * seq_length, -1)
        next_states1p = next_states1p.view(batch_size * seq_length, -1)
        next_states2p = next_states2p.view(batch_size * seq_length, -1)

        z_plus = self.encode_expert_contra(next_states1)
        z_minus = self.encode_expert_contra(next_states2)
        z_plus_p = self.encode_expert_contra(next_states1p)
        z_minus_p = self.encode_expert_contra(next_states2p)

        lb = prefs1[:, 0] == 1.0
        rb = prefs1[:, 1] == 1.0
        eb = prefs1[:, 0] == 0.5

        lbp = prefs2[:, 0] == 1.0
        rbp = prefs2[:, 1] == 1.0
        # Triplet loss
        positive = torch.cat((z_plus[lb], z_minus[rb]), dim=0)
        negative = torch.cat((z_minus[lb], z_plus[rb]), dim=0)
        anchor = self.z.expand(positive.shape[0], -1)
        triplet_fn = torch.nn.TripletMarginLoss(margin=1.0, p=2, reduction='sum')
        triplet_loss = torch.tensor(0.0, device=device)
        if positive.shape[0] > 0:
            triplet_loss += triplet_fn(anchor, positive, negative)
        triplet_loss /= max(positive.shape[0], 1)

        # Quadrilateral loss
        pos_pair = torch.cat((z_plus[lb], z_plus_p[lbp]), dim=0)
        neg_pair = torch.cat((z_minus[lb], z_minus_p[lbp]), dim=0)
        pos_neg1 = torch.cat((z_plus[lb], z_minus_p[lbp]), dim=0)
        pos_neg2 = torch.cat((z_plus_p[lbp], z_minus[lb]), dim=0)
        quad_loss = torch.tensor(0.0, device=device)
        if pos_pair.shape[0] > 1:
            min_len = min(pos_pair.shape[0], neg_pair.shape[0], pos_neg1.shape[0], pos_neg2.shape[0])
            if min_len % 2 != 0:
                min_len -= 1
            pos_pair = pos_pair[:min_len].view(-1, 2, pos_pair.shape[-1])
            neg_pair = neg_pair[:min_len].view(-1, 2, neg_pair.shape[-1])
            pos_neg1 = pos_neg1[:min_len].view(-1, 2, pos_neg1.shape[-1])
            pos_neg2 = pos_neg2[:min_len].view(-1, 2, pos_neg2.shape[-1])
            pos_pos_d = F.pairwise_distance(pos_pair[:, 0], pos_pair[:, 1])
            neg_neg_d = F.pairwise_distance(neg_pair[:, 0], neg_pair[:, 1])
            pos_neg_d1 = F.pairwise_distance(pos_neg1[:, 0], pos_neg1[:, 1])
            pos_neg_d2 = F.pairwise_distance(pos_neg2[:, 0], pos_neg2[:, 1])
            quad_loss = F.relu(pos_pos_d + neg_neg_d - pos_neg_d1 - pos_neg_d2).mean()

        pref_loss = triplet_loss
        return pref_loss, triplet_loss, quad_loss


# Register the functions to FBAgent
FBAgent.encode_expert_contra = encode_expert_contra
FBAgent.compute_preference_loss = compute_preference_loss


# -----------------
# Workspace
# -----------------
class Workspace:
    def __init__(self, cfg, agent_cfg):
        self.cfg = cfg
        self.agent_cfg = agent_cfg
        set_seed_everywhere(0)
        print(f"Loading pretrained model from {self.cfg.checkpoint}")
        self.agent = FBAgent.load(self.cfg.checkpoint, device=self.cfg.device)
        domain_task = f"{self.cfg.domain_name}-{self.cfg.task_name}"
        self.domain_task = domain_task
        self.pref_dataset = load_preference_dataset(domain_task, self.cfg.dataset_path, self.cfg.dataset_type)
    
    def optimize_z(self, num_iters=5000, lr=1e-4, save_interval=5000):
        
        self.agent.z = torch.nn.Parameter(self.agent._model.sample_z(1, device=self.agent.device))
        #self.agent.z = torch.nn.Parameter(self.z_init.clone().detach())
        optimizer = torch.optim.Adam([self.agent.z], lr=lr)

        save_dir = Path(f"z_checkpoints/{self.domain_task}_expert")
        save_dir.mkdir(parents=True, exist_ok=True)

        csv_path = save_dir / "eval_log.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["iter", "reward", "reward_std", "len", "len_std"])

        z_to_save = self.agent.z.detach().clone()
        z_path = save_dir / f"z_iter0000.pt"
        torch.save(z_to_save.cpu(), z_path)
        print(f"[Step 0] Saved z to {z_path}")
        eval_results = self.run_eval(z_to_save)
        print(f"[Eval @ Step 0] {eval_results}")
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([0, eval_results["reward"], eval_results["reward#std"], eval_results["len"], eval_results["len#std"]])

        for it in range(num_iters):
            optimizer.zero_grad()
            pref_loss, trip_loss, quad_loss = self.agent.compute_preference_loss(self.pref_dataset)
            trip_loss.backward()
            optimizer.step()

            # normalize z
            with torch.no_grad():
                self.agent.z.data = (
                    np.sqrt(self.agent.cfg.model.archi.z_dim)
                    * F.normalize(self.agent.z.data, dim=-1)
                )

            if (it + 1) % 100 == 0:
                print(f"[Iter {it+1}] pref={pref_loss.item():.4f}, trip={trip_loss.item():.4f}, quad={quad_loss.item():.4f}")

            if (it + 1) % save_interval == 0 or (it + 1) == num_iters:
                z_to_save = self.agent.z.detach().clone()
                z_path = save_dir / f"z_iter{it+1:05d}.pt"
                torch.save(z_to_save.cpu(), z_path)
                print(f"[Step {it+1}] Saved z to {z_path}")

                eval_results = self.run_eval(z_to_save)
                print(f"[Eval @ Step {it+1}] {eval_results}")

                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([it + 1, eval_results["reward"], eval_results["reward#std"], eval_results["len"], eval_results["len#std"]])

        return self.agent.z.detach()

    def run_eval(self, z):
        '''eval_env = suite.load(
            domain_name=self.cfg.domain_name,
            task_name=self.cfg.task_name,
            environment_kwargs={"flat_observation": True},
        )'''
        eval_env  = dmc.make(f"{self.cfg.domain_name}_{self.cfg.task_name}")
        num_ep = self.cfg.num_eval_episodes
        total_reward = np.zeros((num_ep,), dtype=np.float64)
        ep_lengths = np.zeros((num_ep,), dtype=np.int32)
        trajectories = []

        for ep in range(num_ep):
            ts = eval_env.reset()
            steps = 0
            while not ts.last():
                with torch.no_grad(), eval_mode(self.agent._model):
                    '''obs_tensor = torch.tensor(
                        ts.observation["observations"].reshape(1, -1),
                        device=self.agent.device,
                        dtype=torch.float32,
                    )'''
                    obs_tensor = torch.tensor(ts.observation.reshape(1, -1), device=self.agent.device, dtype=torch.float32)
                    action = self.agent.act(obs=obs_tensor, z=z, mean=True).cpu().numpy()

                ts_next = eval_env.step(action)

                total_reward[ep] += ts_next.reward
                steps += 1

                ts = ts_next

            ep_lengths[ep] = steps

        results = {
            "reward": np.mean(total_reward),
            "reward#std": np.std(total_reward),
            "len": np.mean(ep_lengths),
            "len#std": np.std(ep_lengths),
        }

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain_name", type=str, default="walker")
    parser.add_argument("--task_name", type=str, default="walk")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_type", type=str, default="rnd_dmc_preference_dataset")
    parser.add_argument("--expl_agent", type=str, default="rnd")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_eval_episodes", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = TestConfig(
        domain_name=args.domain_name,
        task_name=args.task_name,
        dataset_path=args.dataset_path,
        dataset_type=args.dataset_type,
        checkpoint=args.checkpoint,
        device=args.device,
        num_eval_episodes=args.num_eval_episodes,
        expl_agent=args.expl_agent,
    )

    agent_cfg = FBAgentConfig()
    ws = Workspace(cfg, agent_cfg)

    z_final = ws.optimize_z(num_iters=50000, lr=1e-4)
    print("\nFinal optimized z:", z_final)
    results = ws.run_eval(z_final)
    print("\n===== Evaluation Results =====")
    print(results)
    z_path = Path(cfg.checkpoint) / "z.pt"
    if z_path.exists():
        print(f"[INFO] Loading optimized z from {z_path}")
        z_loaded = torch.load(z_path, map_location=ws.agent.device)
    results = ws.run_eval(z_loaded)
    print("\n===== Evaluation Results with loaded z =====")
    print(results)
