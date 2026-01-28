

import dataclasses
import torch
import torch.nn.functional as F
from typing import Dict, Tuple

from .model import FBModel, config_from_dict
from .model import Config as FBModelConfig
from ..nn_models import weight_init, _soft_update_params, eval_mode
from ..misc.zbuffer import ZBuffer
from pathlib import Path
import json
import safetensors
from typing import Optional
import numpy as np

@dataclasses.dataclass
class TrainConfig:
    lr_f: float = 1e-4
    lr_b: float = 1e-4
    lr_actor: float = 1e-4
    weight_decay: float = 0.0
    clip_grad_norm: float = 0.0
    fb_target_tau: float = 0.01
    ortho_coef: float = 1.0
    train_goal_ratio: float = 0.5
    fb_pessimism_penalty: float = 0.0
    actor_pessimism_penalty: float = 0.5
    stddev_clip: float = 0.3
    q_loss_coef: float = 0.0
    batch_size: int = 1024
    discount: Optional[float] = None
    use_mix_rollout: bool = False
    update_z_every_step: int = 150
    z_buffer_size: int = 10000
    beta: float = 1.0
    contrastive_coef: float = 1.0
    quad_loss_coef: float = 1.0  # Coefficient for quadrilateral loss
    reg_coefficient: float = 0.0  # Coefficient for positive regularization
    use_dynamic_contrastive_z: bool = False
    batch_size_contrastive: int = 256
    seq_length: int = 200


    recon_coef: float = 0.0
    recon_num_trajs: int = 0
    recon_traj_len: int = 200
    recon_interval: int = 0

@dataclasses.dataclass
class Config:
    model: FBModelConfig = dataclasses.field(default_factory=FBModelConfig)
    train: TrainConfig = dataclasses.field(default_factory=TrainConfig)
    cudagraphs: bool = False
    compile: bool = False
    use_contrastive: bool = False

class FBAgent:
    def __init__(self, **kwargs):
        self.cfg = config_from_dict(kwargs, Config)
        self.cfg.train.fb_target_tau = float(min(max(self.cfg.train.fb_target_tau, 0), 1))
        self._model = FBModel(**dataclasses.asdict(self.cfg.model))
        self.setup_training()
        self.setup_compile()
        self._model.to(self.cfg.model.device)
        if self.cfg.use_contrastive or self.cfg.train.use_dynamic_contrastive_z:
            self.z = torch.nn.Parameter(self._model.sample_z(1, device=self.device))
            self.z_optimizer = torch.optim.Adam([self.z], lr=1e-5)

        self.z_recon_mean = None

    @property
    def device(self):
        return self._model.cfg.device

    def setup_training(self) -> None:
        self._model.train(True)
        self._model.requires_grad_(True)
        self._model.apply(weight_init)
        self._model._prepare_for_train()

        self.backward_optimizer = torch.optim.Adam(
            self._model._backward_map.parameters(),
            lr=self.cfg.train.lr_b,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )
        self.forward_optimizer = torch.optim.Adam(
            self._model._forward_map.parameters(),
            lr=self.cfg.train.lr_f,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )
        self.actor_optimizer = torch.optim.Adam(
            self._model._actor.parameters(),
            lr=self.cfg.train.lr_actor,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )

        self._forward_map_paramlist = tuple(x for x in self._model._forward_map.parameters())
        self._target_forward_map_paramlist = tuple(x for x in self._model._target_forward_map.parameters())
        self._backward_map_paramlist = tuple(x for x in self._model._backward_map.parameters())
        self._target_backward_map_paramlist = tuple(x for x in self._model._target_backward_map.parameters())

        self.off_diag = 1 - torch.eye(self.cfg.train.batch_size, self.cfg.train.batch_size, device=self.device)
        self.off_diag_sum = self.off_diag.sum()

        self.z_buffer = ZBuffer(self.cfg.train.z_buffer_size, self.cfg.model.archi.z_dim, self.cfg.model.device)

    def setup_compile(self):
        print(f"compile {self.cfg.compile}")
        if self.cfg.compile:
            mode = "reduce-overhead" if not self.cfg.cudagraphs else None
            print(f"compiling with mode '{mode}'")
            self.update_fb = torch.compile(self.update_fb, mode=mode)
            self.update_actor = torch.compile(self.update_actor, mode=mode)
            self.sample_mixed_z = torch.compile(self.sample_mixed_z, mode=mode, fullgraph=True)

        print(f"cudagraphs {self.cfg.cudagraphs}")
        if self.cfg.cudagraphs:
            from tensordict.nn import CudaGraphModule
            self.update_fb = CudaGraphModule(self.update_fb, warmup=5)
            self.update_actor = CudaGraphModule(self.update_actor, warmup=5)

    def act(self, obs: torch.Tensor, z: torch.Tensor, mean: bool = True) -> torch.Tensor:
        return self._model.act(obs, z, mean)

    def set_recon_env(self, env):
        self._recon_env = env

    @torch.no_grad()
    def generate_trajectories(self, env, z_anchor: torch.Tensor, num_trajs: int, traj_len: int):
        assert z_anchor.ndim == 2 and z_anchor.shape[0] == 1
        trajectories = []
        for _ in range(num_trajs):
            ts = env.reset()
            obs_seq = []
            steps = 0
            while steps < traj_len and not ts.last():
                obs_np = ts.observation.reshape(1, -1)
                obs_tensor = torch.tensor(obs_np, device=self.device, dtype=torch.float32)
                action = self.act(obs=obs_tensor, z=z_anchor, mean=True).cpu().numpy()
                ts = env.step(action)
                obs_seq.append(ts.observation)
                steps += 1
            if len(obs_seq) < traj_len:
                last = obs_seq[-1]
                obs_seq.extend([last] * (traj_len - len(obs_seq)))
            trajectories.append(np.stack(obs_seq, axis=0))
        arr = np.stack(trajectories, axis=0)
        return torch.from_numpy(arr).float().to(self.device)


    @torch.no_grad()
    def sample_mixed_z(self, train_goal: torch.Tensor, step: int = 0):
        if self.cfg.train.use_dynamic_contrastive_z:
            return self.z.expand(self.cfg.train.batch_size, -1).clone()
        z = self._model.sample_z(self.cfg.train.batch_size, device=self.device)
        if train_goal is not None:
            perm = torch.randperm(self.cfg.train.batch_size, device=self.device)
            goals = self._model._backward_map(train_goal[perm])
            goals = self._model.project_z(goals)
            mask = torch.rand((self.cfg.train.batch_size, 1), device=self.device) < self.cfg.train.train_goal_ratio
            z = torch.where(mask, goals, z)
        return z

    @torch.no_grad()
    def encode_expert(self, next_obs: torch.Tensor):
        B_expert = self._model._backward_map(next_obs).detach()
        B_expert = B_expert.view(
            self.cfg.train.batch_size // self.cfg.model.seq_length,
            self.cfg.model.seq_length,
            B_expert.shape[-1],
        )
        z_expert = B_expert.mean(dim=1)
        z_expert = self._model.project_z(z_expert)
        z_expert = torch.repeat_interleave(z_expert, self.cfg.model.seq_length, dim=0)
        return z_expert

    def encode_expert_test(self, next_obs: torch.Tensor):
        B_expert = self._model._backward_map(next_obs).detach()
        B_expert = B_expert.view(1, 1000, B_expert.shape[-1])
        z_expert = B_expert.mean(dim=1)
        z_expert = self._model.project_z(z_expert)
        return z_expert

    def encode_expert_contra(self, next_obs: torch.Tensor, batch_size=256, seq_len=200):
        B_expert = self._model._backward_map(next_obs)  # batch x d
        
        B_expert = B_expert.view(
            batch_size,
            seq_len,
            B_expert.shape[-1],
        )  # N x L x d
        z_expert = B_expert.mean(dim=1)  # N x d
        z_expert = self._model.project_z(z_expert)
        #z_expert = torch.repeat_interleave(z_expert, self.cfg.model.seq_length, dim=0)  # batch x d
        return z_expert

    def sample_preference_batch(self, pref_dataset):
        batch_size = self.cfg.train.batch_size_contrastive
        seq_len = self.cfg.train.seq_length
        device = self.device

        indices = np.random.choice(len(pref_dataset['labels']), size=(batch_size, 1), replace=False)
        indices1= indices[:, 0]

        next_states1 = torch.FloatTensor(pref_dataset['next_observations'][indices1]).to(device)
        next_states2 = torch.FloatTensor(pref_dataset['next_observations_2'][indices1]).to(device)
        prefs = torch.FloatTensor(pref_dataset['labels'][indices1]).to(device)

        next_states1 = next_states1.view(batch_size * seq_len, -1)
        next_states2 = next_states2.view(batch_size * seq_len, -1)

        return next_states1, next_states2, prefs, indices
    
    def compute_preference_loss(self, next_states1, next_states2, prefs1, indices, pref_dataset: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.device

        z_plus = self.encode_expert_contra(next_states1, batch_size=self.cfg.train.batch_size_contrastive, seq_len=self.cfg.train.seq_length)
        z_minus = self.encode_expert_contra(next_states2, batch_size=self.cfg.train.batch_size_contrastive, seq_len=self.cfg.train.seq_length)
        lb = prefs1[:, 0] == 1.0
        rb = prefs1[:, 1] == 1.0
        eb = prefs1[:, 0] == 0.5

        # Infonce loss components
        tau = max(0.05, 0.2 * np.exp(0 / 20000))
        z_anchor = self.z
        #pos = torch.cat((z_plus[lb], z_minus[rb]), dim=0)
        #neg = torch.cat((z_minus[lb], z_plus[rb]), dim=0)
        pos = torch.cat([
            z_plus[lb],
            z_minus[rb],
            z_plus[eb],
            z_minus[eb],
        ], dim=0)

        neg = torch.cat([
            z_minus[lb],
            z_plus[rb],
            z_minus[eb],
            z_plus[eb],
        ], dim=0)
        sim_pos = F.cosine_similarity(z_anchor.expand_as(pos), pos)
        sim_neg = F.cosine_similarity(z_anchor.expand_as(neg), neg)
        triplet_loss = -torch.log(
            torch.exp(sim_pos / tau) /
            (torch.exp(sim_pos / tau) + torch.exp(sim_neg / tau) + 1e-8)
        ).mean()

        # Quadrilateral loss components

        quad_loss = torch.tensor(0.0, device=device)
        sim = F.cosine_similarity(z_plus, z_minus, dim=1)   # (batch,)

        comp_mask = lb | rb
        unc_mask  = eb 
        pos = sim[unc_mask]
        neg = sim[comp_mask]

        pos_expanded = pos.unsqueeze(1)
        neg_expanded = neg.unsqueeze(0)
        numerator = torch.exp(pos_expanded / tau)

        # InfoNCE denominator: exp(pos/tau) + sum(exp(neg_j/tau))
        denominator = numerator + torch.sum(torch.exp(neg_expanded / tau), dim=1, keepdim=True)

        # Loss = -log(numerator / denominator)
        quad_loss = -torch.log(numerator / denominator).mean()
        if torch.isnan(quad_loss):
            quad_loss = torch.tensor(0.0, device=device)
        
        return triplet_loss, quad_loss


    def update(self, replay_buffer, step: int, pref_dataset: Optional[dict] = None) -> Dict[str, torch.Tensor]:
        batch = replay_buffer["train"].sample(self.cfg.train.batch_size)
        obs, action, next_obs, terminated = (
            batch["observation"],
            batch["action"],
            batch["next"]["observation"],
            batch["next"]["terminated"],
        )
        discount = self.cfg.train.discount * ~terminated

        self._model._obs_normalizer(obs)
        self._model._obs_normalizer(next_obs)
        with torch.no_grad(), eval_mode(self._model._obs_normalizer):
            obs, next_obs = self._model._obs_normalizer(obs), self._model._obs_normalizer(next_obs)

        z = self.sample_mixed_z(train_goal=next_obs, step=step).clone()
        self.z_buffer.add(z)

        q_loss_coef = self.cfg.train.q_loss_coef if self.cfg.train.q_loss_coef > 0 else None
        clip_grad_norm = self.cfg.train.clip_grad_norm if self.cfg.train.clip_grad_norm > 0 else None

        metrics = self.update_fb(
            obs=obs,
            action=action,
            discount=discount,
            next_obs=next_obs,
            goal=next_obs,
            z=z,
            q_loss_coef=q_loss_coef,
            clip_grad_norm=clip_grad_norm,
            pref_dataset=pref_dataset,
            step=step
        )
        metrics.update(
            self.update_actor(
                obs=obs,
                action=action,
                z=z,
                clip_grad_norm=clip_grad_norm,
            )
        )

        with torch.no_grad():
            _soft_update_params(self._forward_map_paramlist, self._target_forward_map_paramlist, self.cfg.train.fb_target_tau)
            _soft_update_params(self._backward_map_paramlist, self._target_backward_map_paramlist, self.cfg.train.fb_target_tau)

        return metrics

    def update_fb(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor,
        goal: torch.Tensor,
        z: torch.Tensor,
        q_loss_coef: Optional[float] = None,
        clip_grad_norm: Optional[float] = None,
        pref_dataset: Optional[dict] = None,
        step: int = 0
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            dist = self._model._actor(next_obs, z, self._model.cfg.actor_std)
            next_action = dist.sample(clip=self.cfg.train.stddev_clip)
            target_Fs = self._model._target_forward_map(next_obs, z, next_action)  # num_parallel x batch x z_dim
            target_B = self._model._target_backward_map(goal)  # batch x z_dim
            target_Ms = torch.matmul(target_Fs, target_B.T)  # num_parallel x batch x batch
            _, _, target_M = self.get_targets_uncertainty(target_Ms, self.cfg.train.fb_pessimism_penalty)  # batch x batch

        # compute FB loss
        Fs = self._model._forward_map(obs, z, action)  # num_parallel x batch x z_dim
        B = self._model._backward_map(goal)  # batch x z_dim
        Ms = torch.matmul(Fs, B.T)  # num_parallel x batch x batch

        diff = Ms - discount * target_M  # num_parallel x batch x batch
        fb_offdiag = 0.5 * (diff * self.off_diag).pow(2).sum() / self.off_diag_sum
        fb_diag = -torch.diagonal(diff, dim1=1, dim2=2).mean() * Ms.shape[0]
        fb_loss = fb_offdiag + fb_diag

        # compute orthonormality loss for backward embedding
        Cov = torch.matmul(B, B.T)
        orth_loss_diag = -Cov.diag().mean()
        orth_loss_offdiag = 0.5 * (Cov * self.off_diag).pow(2).sum() / self.off_diag_sum
        orth_loss = orth_loss_offdiag + orth_loss_diag
        fb_loss += self.cfg.train.ortho_coef * orth_loss

        q_loss = torch.zeros(1, device=z.device, dtype=z.dtype)
        if q_loss_coef is not None:
            with torch.no_grad():
                next_Qs = (target_Fs * z).sum(dim=-1)  # num_parallel x batch
                _, _, next_Q = self.get_targets_uncertainty(next_Qs, self.cfg.train.fb_pessimism_penalty)  # batch
                cov = torch.matmul(B.T, B) / B.shape[0]  # z_dim x z_dim
                inv_cov = torch.inverse(cov)  # z_dim x z_dim
                implicit_reward = (torch.matmul(B, inv_cov) * z).sum(dim=-1)  # batch
                target_Q = implicit_reward.detach() + discount.squeeze() * next_Q  # batch
                expanded_targets = target_Q.expand(Fs.shape[0], -1)
            Qs = (Fs * z).sum(dim=-1)  # num_parallel x batch
            q_loss = 0.5 * Fs.shape[0] * F.mse_loss(Qs, expanded_targets)
            fb_loss += q_loss_coef * q_loss

        pref_loss = torch.tensor(0.0, device=z.device)
        triplet_loss = torch.tensor(0.0, device=z.device)
        quad_loss = torch.tensor(0.0, device=z.device)
        recon_loss = torch.tensor(0.0, device=z.device)
        reg_pos = torch.tensor(0.0, device=z.device)
        if self.cfg.use_contrastive and pref_dataset is not None:
            next_states1, next_states2, prefs, indices = self.sample_preference_batch(pref_dataset)
            triplet_loss, quad_loss = self.compute_preference_loss(next_states1, next_states2, prefs, indices, pref_dataset)

            pref_loss = self.cfg.train.contrastive_coef * triplet_loss + self.cfg.train.quad_loss_coef * quad_loss
            
        
        #total_loss = fb_loss + pref_loss
        total_loss = fb_loss + pref_loss + self.cfg.train.recon_coef * recon_loss

        #self.check_recon_gradients(pref_loss)
        #self.check_recon_gradients(recon_loss)
        self.backward_optimizer.zero_grad(set_to_none=True)
        self.forward_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model._backward_map.parameters(), clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(self._model._forward_map.parameters(), clip_grad_norm)
        self.backward_optimizer.step()
        self.forward_optimizer.step()



        if self.cfg.use_contrastive and pref_dataset is not None:
            with torch.no_grad():
                z_plus = self.encode_expert_contra(next_states1, batch_size=self.cfg.train.batch_size_contrastive, seq_len=self.cfg.train.seq_length)
                z_minus = self.encode_expert_contra(next_states2, batch_size=self.cfg.train.batch_size_contrastive, seq_len=self.cfg.train.seq_length)

            lb = prefs[:, 0] == 1.0
            rb = prefs[:, 1] == 1.0
            eb = prefs[:, 0] == 0.5

            # Triplet loss components
            tau = max(0.05, 0.2 * np.exp(0 / 20000))
            triplet_losses = []
            z_anchor = self.z
            #pos = torch.cat((z_plus[lb], z_minus[rb]), dim=0)
            #neg = torch.cat((z_minus[lb], z_plus[rb]), dim=0)
            pos = torch.cat([
                z_plus[lb],
                z_minus[rb],
                z_plus[eb],
                z_minus[eb],
            ], dim=0)

            neg = torch.cat([
                z_minus[lb],
                z_plus[rb],
                z_minus[eb],
                z_plus[eb],
            ], dim=0)
            sim_pos = F.cosine_similarity(z_anchor.expand_as(pos), pos)
            sim_neg = F.cosine_similarity(z_anchor.expand_as(neg), neg)
            triplet_loss2 = -torch.log(
                torch.exp(sim_pos / tau) /
                (torch.exp(sim_pos / tau) + torch.exp(sim_neg / tau) + 1e-8)
            ).mean() * self.cfg.train.contrastive_coef
            self.z_optimizer.zero_grad(set_to_none=True)
            triplet_loss2.backward()
            self.z_optimizer.step()
            self.z.data = np.sqrt(self.cfg.model.archi.z_dim) * F.normalize(self.z, dim=-1)
            

        with torch.no_grad():
            output_metrics = {
                "target_M": target_M.mean(),
                "M": Ms.mean(),
                "F1": Fs[0].mean(),
                "B": B.mean(),
                "B_norm": torch.norm(B, dim=-1).mean(),
                "z_norm": torch.norm(z, dim=-1).mean(),
                "fb_loss": fb_loss,
                "fb_diag": fb_diag,
                "fb_offdiag": fb_offdiag,
                "orth_loss": orth_loss,
                "orth_loss_diag": orth_loss_diag,
                "orth_loss_offdiag": orth_loss_offdiag,
                "q_loss": q_loss,
                "pref_loss": pref_loss,
                "triplet_loss": triplet_loss,
                "triplet_loss2": triplet_loss2,
                "quad_loss": quad_loss,
                "reg_pos": reg_pos,
                "recon_loss": recon_loss,
            }
        return output_metrics

    def update_actor(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        z: torch.Tensor,
        clip_grad_norm: Optional[float] = None
        #clip_grad_norm: float | None,
    ) -> Dict[str, torch.Tensor]:
        return self.update_td3_actor(obs=obs, z=z, clip_grad_norm=clip_grad_norm)

    def update_td3_actor(self, obs: torch.Tensor, z: torch.Tensor, clip_grad_norm: float) -> Dict[str, torch.Tensor]:
        dist = self._model._actor(obs, z, self._model.cfg.actor_std)
        action = dist.sample(clip=self.cfg.train.stddev_clip)
        Fs = self._model._forward_map(obs, z, action)  # num_parallel x batch x z_dim
        Qs = (Fs * z).sum(-1)  # num_parallel x batch
        _, _, Q = self.get_targets_uncertainty(Qs, self.cfg.train.actor_pessimism_penalty)  # batch
        actor_loss = -Q.mean()

        # optimize actor
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model._actor.parameters(), clip_grad_norm)
        self.actor_optimizer.step()

        return {"actor_loss": actor_loss.detach(), "q": Q.mean().detach()}

    def get_targets_uncertainty(
        self, preds: torch.Tensor, pessimism_penalty: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dim = 0
        preds_mean = preds.mean(dim=dim)
        preds_uns = preds.unsqueeze(dim=dim)
        preds_uns2 = preds.unsqueeze(dim=dim + 1)
        preds_diffs = torch.abs(preds_uns - preds_uns2)
        num_parallel_scaling = preds.shape[dim] ** 2 - preds.shape[dim]
        preds_unc = (
            preds_diffs.sum(
                dim=(dim, dim + 1),
            )
            / num_parallel_scaling
        )
        return preds_mean, preds_unc, preds_mean - pessimism_penalty * preds_unc

    def maybe_update_rollout_context(self, z: torch.Tensor, step_count: torch.Tensor) -> torch.Tensor:
        # get mask for environmets where we need to change z
        if z is not None:
            mask_reset_z = step_count % self.cfg.train.update_z_every_step == 0
            if self.cfg.train.use_mix_rollout and not self.z_buffer.empty():
                new_z = self.z_buffer.sample(z.shape[0], device=self.cfg.model.device)
            else:
                new_z = self._model.sample_z(z.shape[0], device=self.cfg.model.device)
            z = torch.where(mask_reset_z, new_z, z.to(self.cfg.model.device))
        else:
            z = self._model.sample_z(step_count.shape[0], device=self.cfg.model.device)
        return z

    @classmethod
    def load(cls, path: str, device: str, override_cfg=None):
        path = Path(path)
        with (path / "config.json").open() as f:
            loaded_config = json.load(f)
        if device is not None:
            loaded_config["model"]["device"] = device
        if override_cfg is not None:
            loaded_config["use_contrastive"] = override_cfg.use_contrastive
            loaded_config["train"]["contrastive_coef"] = override_cfg.contrastive_coef
            loaded_config["train"]["quad_loss_coef"] = override_cfg.quad_loss_coef
            loaded_config["train"]["use_dynamic_contrastive_z"] = override_cfg.use_dynamic_contrastive_z
            loaded_config["train"]["recon_coef"] = getattr(override_cfg, "recon_coef", 0.0)
            loaded_config["train"]["recon_num_trajs"] = getattr(override_cfg, "recon_num_trajs", 0)
            loaded_config["train"]["recon_traj_len"] = getattr(override_cfg, "recon_traj_len", 200)
            loaded_config["train"]["recon_interval"] = getattr(override_cfg, "recon_interval", 0)
            loaded_config["train"]["q_loss_coef"] = override_cfg.q_loss_coef
            loaded_config["train"]["seq_length"] = override_cfg.seq_length
            loaded_config["train"]["batch_size_contrastive"] = override_cfg.batch_size_contrastive
            #loaded_config["train"]["reg_coefficient"] = override_cfg.reg_coefficient

        agent = cls(**loaded_config)
        optimizers = torch.load(str(path / "optimizers.pth"), weights_only=True)
        agent.actor_optimizer.load_state_dict(optimizers["actor_optimizer"])
        agent.backward_optimizer.load_state_dict(optimizers["backward_optimizer"])
        agent.forward_optimizer.load_state_dict(optimizers["forward_optimizer"])
        '''for pg in agent.forward_optimizer.param_groups:
            pg["lr"] = 1e-6'''
        if agent.cfg.use_contrastive or agent.cfg.train.use_dynamic_contrastive_z:
            if "z_optimizer" in optimizers and hasattr(agent, "z_optimizer"):
                agent.z_optimizer.load_state_dict(optimizers["z_optimizer"])

        safetensors.torch.load_model(agent._model, path / "model/model.safetensors", device=device)

        if not hasattr(agent, "z") or agent.z is None:
            agent.z = torch.nn.Parameter(agent._model.sample_z(1, device=agent.device))
            agent.z_optimizer = torch.optim.Adam([agent.z], lr=agent.cfg.train.lr_b)
        return agent

    def save(self, output_folder: str) -> None:
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True)
        with (output_folder / "config.json").open("w+") as f:
            json.dump(dataclasses.asdict(self.cfg), f, indent=4)
        optimizers = {
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "backward_optimizer": self.backward_optimizer.state_dict(),
            "forward_optimizer": self.forward_optimizer.state_dict(),
        }
        if self.cfg.use_contrastive or self.cfg.train.use_dynamic_contrastive_z:
            optimizers["z_optimizer"] = self.z_optimizer.state_dict()
        torch.save(optimizers, output_folder / "optimizers.pth")
        model_folder = output_folder / "model"
        model_folder.mkdir(exist_ok=True)
        self._model.save(output_folder=str(model_folder))
        if getattr(self, "z", None) is not None:
            torch.save(self.z.detach().to("cpu").clone(), output_folder / "z.pt")
