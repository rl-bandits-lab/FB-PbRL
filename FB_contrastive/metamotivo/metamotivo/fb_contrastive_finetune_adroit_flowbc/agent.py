# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import json
import pickle
from pathlib import Path
from typing import Dict, Literal, Tuple, Optional, Union

import safetensors
import torch
import torch.nn.functional as F
from torch.amp import autocast
import numpy as np

from .base import BaseConfig
from ..envs.utils.gym_spaces import json_to_space, space_to_json
from .nn_models import _soft_update_params, eval_mode, weight_init
from .model import FBModel, FBModelConfig


class FBAgentTrainConfig(BaseConfig):
    lr_f: float = 1e-4
    lr_b: float = 1e-4
    lr_actor: float = 1e-4
    weight_decay: float = 0.0
    clip_grad_norm: float = 0.0
    f_target_tau: float = 0.005
    b_target_tau: float = 0.005
    ortho_coef: float = 1.0
    train_goal_ratio: float = 0.5
    fb_pessimism_penalty: float = 0.0
    actor_pessimism_penalty: float = 0.0
    stddev_clip: float = 0.3
    q_loss_coef: float = 0.0
    batch_size: int = 1024
    discount: float = 0.99
    use_mix_rollout: bool = False
    update_z_every_step: int = 150
    z_buffer_size: int = 10000
    bc_coeff: float = 0.0

    beta: float = 1.0
    contrastive_coef: float = 1.0
    quad_loss_coef: float = 1.0  # Coefficient for quadrilateral loss
    reg_coefficient: float = 0.0  # Coefficient for positive regularization
    use_dynamic_contrastive_z: bool = False

    recon_coef: float = 0.0
    recon_num_trajs: int = 0
    recon_traj_len: int = 200
    recon_interval: int = 0
    bs: int = 256
    seq_length: int = 100


class FBAgentConfig(BaseConfig):
    name: Literal["FBAgent"] = "FBAgent"
    model: FBModelConfig
    train: FBAgentTrainConfig
    cudagraphs: bool = False
    compile: bool = False
    use_contrastive: bool = False

    def build(self, obs_space, action_dim):
        return self.object_class(obs_space, action_dim, self)

    @property
    def object_class(self):
        return FBAgent


class FBAgent:
    config_class = FBAgentConfig

    def __init__(self, obs_space, action_dim, cfg: FBAgentConfig):
        self.obs_space = obs_space
        self.action_dim = action_dim
        self.cfg = cfg
        self._model: FBModel = self.cfg.model.build(obs_space, action_dim)
        self.setup_training()
        self.setup_compile()
        self._model.to(self.device)
        if self.cfg.use_contrastive or self.cfg.train.use_dynamic_contrastive_z:
            self.z = torch.nn.Parameter(self._model.sample_z(1, device=self.device))
            #self.z_optimizer = torch.optim.Adam([self.z], lr=self.cfg.train.lr_b)
            self.z_optimizer = torch.optim.Adam([self.z], lr=1e-5)

        self.z_recon_mean = None

    @property
    def device(self):
        return self._model.device

    @property
    def optimizer_dict(self):
        return {
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "backward_optimizer": self.backward_optimizer.state_dict(),
            "forward_optimizer": self.forward_optimizer.state_dict(),
        }

    def setup_training(self) -> None:
        self._model.train(True)
        self._model.requires_grad_(True)
        self._model.apply(weight_init)
        self._model._prepare_for_train()  # ensure that target nets are initialized after applying the weights

        self.backward_optimizer = torch.optim.Adam(
            list(self._model._backward_map.parameters()) + list(self._model._bw_encoder.parameters()),
            lr=self.cfg.train.lr_b,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )
        self.forward_optimizer = torch.optim.Adam(
            list(self._model._forward_map.parameters())
            + list(self._model._left_encoder.parameters())
            + list(self._model._fw_encoder.parameters()),
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

        # prepare parameter list
        self._forward_map_paramlist = tuple(x for x in self._model._forward_map.parameters())
        self._target_forward_map_paramlist = tuple(x for x in self._model._target_forward_map.parameters())
        self._backward_map_paramlist = tuple(x for x in self._model._backward_map.parameters())
        self._target_backward_map_paramlist = tuple(x for x in self._model._target_backward_map.parameters())
        self._left_encoder_paramlist = tuple(x for x in self._model._left_encoder.parameters())
        self._target_left_encoder_paramlist = tuple(x for x in self._model._target_left_encoder.parameters())

        # precompute some useful variables
        self.off_diag = 1 - torch.eye(self.cfg.train.batch_size, self.cfg.train.batch_size, device=self.device)
        self.off_diag_sum = self.off_diag.sum()

    def setup_compile(self):
        print(f"compile {self.cfg.compile}")
        if self.cfg.compile:
            mode = "reduce-overhead" if not self.cfg.cudagraphs else None
            print(f"compiling with mode '{mode}'")
            self.update_fb = torch.compile(self.update_fb, mode=mode)  # use fullgraph=True to debug for graph breaks
            self.update_actor = torch.compile(self.update_actor, mode=mode)  # use fullgraph=True to debug for graph breaks
            # feel free to re-enable compilation if https://github.com/pytorch/pytorch/issues/166604 is resolved
            # self.sample_mixed_z = torch.compile(self.sample_mixed_z, mode=mode, fullgraph=True)
            self.aug = torch.compile(self.aug, mode=mode)
            self.enc = torch.compile(self.enc, mode=mode)

        print(f"cudagraphs {self.cfg.cudagraphs}")
        if self.cfg.cudagraphs:
            from tensordict.nn import CudaGraphModule

            self.update_fb = CudaGraphModule(self.update_fb, warmup=5)
            self.update_actor = CudaGraphModule(self.update_actor, warmup=5)

    def act(self, obs: torch.Tensor, z: torch.Tensor, mean: bool = True) -> torch.Tensor:
        return self._model.act(obs, z, mean)

    @torch.no_grad()
    def sample_mixed_z(self, train_goal: Optional[torch.Tensor] = None, *args, **kwargs):
        # samples a batch from the z distribution used to update the networks
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            if self.cfg.train.use_dynamic_contrastive_z:
                return self.z.expand(self.cfg.train.batch_size, -1).clone()
            z = self._model.sample_z(self.cfg.train.batch_size, device=self.device)
            if train_goal is not None:
                perm = torch.randperm(self.cfg.train.batch_size, device=self.device)
                train_goal = train_goal[perm]
                goals = self._model._backward_map(train_goal)
                goals = self._model.project_z(goals)
                mask = torch.rand((self.cfg.train.batch_size, 1), device=self.device) < self.cfg.train.train_goal_ratio
                z = torch.where(mask, goals, z)
        return z

    @torch.no_grad()
    def aug(self, obs, next_obs):
        """
        Augments observations when training from pixels, does nothing otherwise.
        """
        return self._model._augmentator(obs), self._model._augmentator(next_obs)

    def enc(self, obs, next_obs):
        """
        Encodes observations when training from pixels, does nothing otherwise.
        """
        obs = self._model._fw_encoder(obs)
        goal = self._model._bw_encoder(next_obs)
        with torch.no_grad():
            next_obs = self._model._fw_encoder(next_obs)
        return obs, next_obs, goal

    def encode_expert_contra(self, next_obs: torch.Tensor, batch_size=64):
        # encode expert trajectories through B
        #B_expert = self._model._backward_map(next_obs).detach()  # batch x d
        goal = self._model._bw_encoder(next_obs)
        B_expert = self._model._backward_map(goal)  # batch x d
        B_expert = B_expert.view(
            self.cfg.train.bs,
            #200,
            #100,
            self.cfg.train.seq_length,
            #1000,
            B_expert.shape[-1],
        )  # N x L x d
        z_expert = B_expert.mean(dim=1)  # N x d
        z_expert = self._model.project_z(z_expert)
        #z_expert = torch.repeat_interleave(z_expert, self.cfg.model.seq_length, dim=0)  # batch x d
        return z_expert
    
    def sample_preference_batch(self, pref_dataset):
        #batch_size = 256
        #batch_size = 64
        #seq_len = 200
        #seq_len = 100
        #seq_len = 50
        #seq_len = 1000
        batch_size = self.cfg.train.bs
        seq_len = self.cfg.train.seq_length
        device = self.device

        indices1 = np.random.choice(len(pref_dataset['labels']), size=(batch_size), replace=False)
        #indices1= indices[:, 0]

        next_states1 = torch.FloatTensor(pref_dataset['next_observations'][indices1]).to(device)
        next_states2 = torch.FloatTensor(pref_dataset['next_observations_2'][indices1]).to(device)
        prefs = torch.FloatTensor(pref_dataset['labels'][indices1]).to(device)

        next_states1 = next_states1.view(batch_size * seq_len, -1)
        next_states2 = next_states2.view(batch_size * seq_len, -1)

        return next_states1, next_states2, prefs, indices1
    
    def compute_preference_loss(self, next_states1, next_states2, prefs1, indices, pref_dataset: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = 256
        #seq_length = 200
        seq_length = 100
        #seq_length = 50
        #seq_length = 1000
        device = self.device
        #indices2 = indices[:, 1]

        # Second pair (σ′+, σ′−)
        #next_states1_prime = torch.FloatTensor(pref_dataset['next_observations'][indices2]).to(device)
        #next_states2_prime = torch.FloatTensor(pref_dataset['next_observations_2'][indices2]).to(device)
        #prefs2 = torch.FloatTensor(pref_dataset['labels'][indices2]).to(device)

        #next_states1_prime = next_states1_prime.view(batch_size * seq_length, -1)
        #next_states2_prime = next_states2_prime.view(batch_size * seq_length, -1)

        z_plus = self.encode_expert_contra(next_states1)
        z_minus = self.encode_expert_contra(next_states2)
        #z_plus_prime = self.encode_expert_contra(next_states1_prime)
        #z_minus_prime = self.encode_expert_contra(next_states2_prime)

        lb = prefs1[:, 0] == 1.0
        rb = prefs1[:, 1] == 1.0
        eb = prefs1[:, 0] == 0.5
        #lb_prime = prefs2[:, 0] == 1.0
        #rb_prime = prefs2[:, 1] == 1.0
        #eb_prime = prefs2[:, 0] == 0.5

        # Triplet loss components
        tau = max(0.05, 0.2 * np.exp(0 / 20000))
        z_anchor = self.z
        pos = torch.cat((z_plus[lb], z_minus[rb]), dim=0)
        neg = torch.cat((z_minus[lb], z_plus[rb]), dim=0)
        '''pos = torch.cat([
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
        ], dim=0)'''

        sim_pos = F.cosine_similarity(z_anchor.expand_as(pos), pos)
        sim_neg = F.cosine_similarity(z_anchor.expand_as(neg), neg)
        triplet_loss = -torch.log(
            torch.exp(sim_pos / tau) /
            (torch.exp(sim_pos / tau) + torch.exp(sim_neg / tau) + 1e-8)
        ).mean()

        # Quadrilateral loss components

        #pos2 = torch.cat((z_plus_prime[lb_prime],  z_minus_prime[rb_prime]), dim=0)
        #neg2 = torch.cat((z_minus_prime[lb_prime], z_plus_prime[rb_prime]),  dim=0)

        quad_loss = torch.tensor(0.0, device=device)
        sim = F.cosine_similarity(z_plus, z_minus, dim=1)   # (batch,)

        comp_mask = lb | rb     # comparable → want large distance
        unc_mask  = eb 
        # Positive = uncomparable (ambiguous)
        pos = sim[unc_mask]   # want HIGH similarity (close)

        # Negative = comparable
        neg = sim[comp_mask]  # want LOW similarity (far)

        # If either group empty → loss = 0
        # For each positive, compare to ALL negatives
        # pos: (P,)
        # neg: (N,)
        
        # expand to match shapes:
        # pos_expanded: (P, N)
        # neg_expanded: (P, N)
        pos_expanded = pos.unsqueeze(1)              # (P,1)
        neg_expanded = neg.unsqueeze(0)              # (1,N)
        # InfoNCE numerator: exp(pos/tau)
        numerator = torch.exp(pos_expanded / tau)    # (P,1)

        # InfoNCE denominator: exp(pos/tau) + sum(exp(neg_j/tau))
        denominator = numerator + torch.sum(torch.exp(neg_expanded / tau), dim=1, keepdim=True)

        # Loss = -log(numerator / denominator)
        quad_loss = -torch.log(numerator / denominator).mean()
        if torch.isnan(quad_loss):
            quad_loss = torch.tensor(0.0, device=device)
        '''if pos1.shape[0] > 0 and pos2.shape[0] > 0:
            n = min(pos1.shape[0], pos2.shape[0])

            pos1 = pos1[:n]
            neg1 = neg1[:n]
            pos2 = pos2[:n]
            neg2 = neg2[:n]

            pos_pos_dist  = F.pairwise_distance(pos1, pos2)
            neg_neg_dist  = F.pairwise_distance(neg1, neg2)
            pos_neg_dist1 = F.pairwise_distance(pos1, neg2)
            pos_neg_dist2 = F.pairwise_distance(pos2, neg1)

            # quadrilateral inequality:
            # d(pos,pos) + d(neg,neg) < d(pos,neg') + d(pos',neg)
            quad_term = (pos_pos_dist + neg_neg_dist
                               - pos_neg_dist1 - pos_neg_dist2)

            quad_loss = quad_term.mean()'''
        
        return triplet_loss, quad_loss

    def update(self, replay_buffer, step: int, pref_dataset: Optional[dict] = None) -> Dict[str, torch.Tensor]:
        batch = replay_buffer["train"].sample(self.cfg.train.batch_size)

        obs, action, next_obs, terminated = (
            batch["observation"].to(self.device),
            batch["action"].to(self.device),
            batch["next"]["observation"].to(self.device),
            batch["next"]["terminated"].to(self.device),
        )
        discount = self.cfg.train.discount * ~terminated

        self._model._obs_normalizer(obs)
        self._model._obs_normalizer(next_obs)
        with torch.no_grad(), eval_mode(self._model._obs_normalizer):
            obs, next_obs = self._model._obs_normalizer(obs), self._model._obs_normalizer(next_obs)

        #torch.compiler.cudagraph_mark_step_begin()

        obs, next_obs = self.aug(obs, next_obs)
        obs, next_obs, goal = self.enc(obs, next_obs)

        z = self.sample_mixed_z(train_goal=goal).clone()

        q_loss_coef = self.cfg.train.q_loss_coef if self.cfg.train.q_loss_coef > 0 else None
        clip_grad_norm = self.cfg.train.clip_grad_norm if self.cfg.train.clip_grad_norm > 0 else None

        metrics = self.update_fb(
            obs=obs,
            action=action,
            discount=discount,
            next_obs=next_obs,
            goal=goal,
            z=z,
            q_loss_coef=q_loss_coef,
            clip_grad_norm=clip_grad_norm,
            pref_dataset=pref_dataset,
        )
        metrics.update(
            self.update_actor(
                obs=obs.detach(),
                action=action,
                z=z,
                clip_grad_norm=clip_grad_norm,
            )
        )

        with torch.no_grad():
            _soft_update_params(self._forward_map_paramlist, self._target_forward_map_paramlist, self.cfg.train.f_target_tau)
            _soft_update_params(self._backward_map_paramlist, self._target_backward_map_paramlist, self.cfg.train.b_target_tau)
            if len(self._left_encoder_paramlist):
                _soft_update_params(self._left_encoder_paramlist, self._target_left_encoder_paramlist, self.cfg.train.f_target_tau)

        return metrics

    def sample_action_from_norm_obs(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            dist = self._model._actor(obs, z, self._model.cfg.actor_std)
            action = dist.sample(clip=self.cfg.train.stddev_clip)
        return action

    def update_fb(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor,
        goal: torch.Tensor,
        z: torch.Tensor,
        q_loss_coef: Optional[float],
        clip_grad_norm: Optional[float],
        pref_dataset: Optional[dict] = None,
    ) -> Dict[str, torch.Tensor]:
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            with torch.no_grad():
                next_left_enc = self._model._target_left_encoder(next_obs)  # batch x L_dim
                actor_in = next_left_enc if self.cfg.model.actor_encode_obs else next_obs
                next_action = self.sample_action_from_norm_obs(actor_in, z)
                target_Fs = self._model._target_forward_map(next_left_enc, z, next_action)  # num_parallel x batch x z_dim
                target_B = self._model._target_backward_map(goal)  # batch x z_dim
                target_Ms = torch.matmul(target_Fs, target_B.T)  # num_parallel x batch x batch
                _, _, target_M = self.get_targets_uncertainty(target_Ms, self.cfg.train.fb_pessimism_penalty)  # batch x batch

            # compute FB loss
            left_enc = self._model._left_encoder(obs)  # batch x L_dim
            Fs = self._model._forward_map(left_enc, z, action)  # num_parallel x batch x z_dim
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
                    # we disable autocast here to make sure B and cov have the same dtype (otherwise torch.linalg.solve fails)
                    with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=False):
                        cov = torch.matmul(B.T, B) / B.shape[0]  # z_dim x z_dim
                    B_inv_conv = torch.linalg.solve(cov, B, left=False)
                    implicit_reward = (B_inv_conv * z).sum(dim=-1)  # batch
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
            
            # ---------------------------------------------
            #  Reconstruction Consistency Loss (recon_loss)
            # ---------------------------------------------
            if (
                self.cfg.train.recon_coef > 0
                and hasattr(self, "_recon_env")
            ):
                '''if step % int(self.cfg.train.recon_interval) == 0:
                    
                    
                    num_recon = int(self.cfg.train.recon_num_trajs)
                    
                    z_batch = self._model.sample_z(num_recon, device=self.device)
                    
                    
                    trajs_list = []
                    with torch.no_grad():
                        for i in range(num_recon):
                            z_single = z_batch[i].unsqueeze(0) 
                            
                            single_traj = self.generate_trajectories(
                                env=self._recon_env,
                                z_anchor=z_single, 
                                num_trajs=1, 
                                traj_len=int(self.cfg.train.recon_traj_len),
                            ) # Shape: (1, L, D)
                            trajs_list.append(single_traj)
                    
                    trajs = torch.cat(trajs_list, dim=0)
                    
                    N, L, D = trajs.shape
                    flat_obs = trajs.reshape(N * L, D)
                    z_recon = self.encode_expert_contra(flat_obs, batch_size=N)
                    cos_sim = F.cosine_similarity(z_recon, z_batch, dim=-1)
                    
                    recon_loss = (1 - cos_sim).mean()'''
                if step % int(self.cfg.train.recon_interval) == 0:
                    with torch.no_grad():
                        trajs = self.generate_trajectories(
                            env=self._recon_env,
                            z_anchor=self.z,  # (1, z_dim)
                            num_trajs=int(self.cfg.train.recon_num_trajs),
                            traj_len=int(self.cfg.train.recon_traj_len),
                        )  # [N, L, obs_dim]

                    N, L, D = trajs.shape
                    flat_obs = trajs.reshape(N * L, D)  # (N*L, obs_dim)
                    z_recon = self.encode_expert_contra(flat_obs, batch_size=N)  # (N, z_dim)
                    self.z_recon_mean = z_recon.mean(dim=0, keepdim=True)      # (1, z_dim)
                    cos_sim = F.cosine_similarity(self.z_recon_mean, self.z.detach(), dim=-1)
                    recon_loss = (1 - cos_sim).mean()
                    #print(f"Step {step}: Recon Loss = {recon_loss.item():.6f}")
            else:
                recon_loss = torch.tensor(0.0, device=z.device)
        
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
                z_plus = self.encode_expert_contra(next_states1)
                z_minus = self.encode_expert_contra(next_states2)

            lb = prefs[:, 0] == 1.0
            rb = prefs[:, 1] == 1.0
            eb = prefs[:, 0] == 0.5

            # Triplet loss components
            tau = max(0.05, 0.2 * np.exp(0 / 20000))
            triplet_losses = []
            z_anchor = self.z
            pos = torch.cat((z_plus[lb], z_minus[rb]), dim=0)
            neg = torch.cat((z_minus[lb], z_plus[rb]), dim=0)
            '''pos = torch.cat([
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
            ], dim=0)'''
            sim_pos = F.cosine_similarity(z_anchor.expand_as(pos), pos)
            sim_neg = F.cosine_similarity(z_anchor.expand_as(neg), neg)
            triplet_loss2 = -torch.log(
                torch.exp(sim_pos / tau) /
                (torch.exp(sim_pos / tau) + torch.exp(sim_neg / tau) + 1e-8)
            ).mean() * self.cfg.train.contrastive_coef
            if self.cfg.train.reg_coefficient > 0:
                sim_z = F.cosine_similarity(self.z, pos.mean(dim=0, keepdim=True).detach(), dim=-1)
                reg_pos = (1.0 - sim_z).mean()
                triplet_loss2 += self.cfg.train.reg_coefficient * reg_pos
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
        clip_grad_norm: Optional[float],
    ) -> Dict[str, torch.Tensor]:

        return self.update_td3_actor(obs=obs, action=action, z=z, clip_grad_norm=clip_grad_norm)

    def update_td3_actor(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        z: torch.Tensor,
        clip_grad_norm: Optional[float],
    ) -> Dict[str, torch.Tensor]:
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            with torch.no_grad():
                left_enc = self._model._left_encoder(obs)
            actor_in = left_enc if self.cfg.model.actor_encode_obs else obs
            dist = self._model._actor(actor_in, z, self._model.cfg.actor_std)
            actor_action = dist.sample(clip=self.cfg.train.stddev_clip)
            Fs = self._model._forward_map(left_enc, z, actor_action)  # num_parallel x batch x z_dim
            Qs = (Fs * z).sum(-1)  # num_parallel x batch
            _, _, Q = self.get_targets_uncertainty(Qs, self.cfg.train.actor_pessimism_penalty)  # batch
            actor_loss = -Q.mean()

            # compute bc loss
            bc_error = torch.tensor([0.0], device=action.device)
            if self.cfg.train.bc_coeff > 0:
                bc_error = F.mse_loss(actor_action, action)
                bc_loss = self.cfg.train.bc_coeff * bc_error
                actor_loss = (actor_loss / Qs.abs().mean().detach()) + bc_loss

        # optimize actor
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model._actor.parameters(), clip_grad_norm)
        self.actor_optimizer.step()

        return {"actor_loss": actor_loss.detach(), "bc_error": bc_error.detach(), "q": Q.mean().detach()}

    def get_targets_uncertainty(
        self,
        preds: torch.Tensor,
        pessimism_penalty: Union[torch.Tensor, float],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dim = 0
        preds_mean = preds.mean(dim=dim)
        preds_uns = preds.unsqueeze(dim=dim)  # 1 x n_parallel x ...
        preds_uns2 = preds.unsqueeze(dim=dim + 1)  # n_parallel x 1 x ...
        preds_diffs = torch.abs(preds_uns - preds_uns2)  # n_parallel x n_parallel x ...
        num_parallel_scaling = preds.shape[dim] ** 2 - preds.shape[dim]
        preds_unc = (
            preds_diffs.sum(
                dim=(dim, dim + 1),
            )
            / num_parallel_scaling
        )
        return preds_mean, preds_unc, preds_mean - pessimism_penalty * preds_unc

    @classmethod
    def load(cls, path: str, device: str, override_cfg=None):
        path = Path(path)
        with (path / "config.json").open() as f:
            loaded_config = json.load(f)
        if device is not None:
            loaded_config["model"]["device"] = device

        if override_cfg is not None:
            # agent-level
            if "use_contrastive" in override_cfg:
                loaded_config["use_contrastive"] = override_cfg["use_contrastive"]

            # train-level
            train_cfg = loaded_config.setdefault("train", {})
            for k in [
                "contrastive_coef",
                "quad_loss_coef",
                "use_dynamic_contrastive_z",
                "q_loss_coef",
                "ortho_coef",
                "lr_b",
                "bs",
                "seq_length",
            ]:
                if k in override_cfg:
                    train_cfg[k] = override_cfg[k]
        if (path / "init_kwargs.pkl").exists():
            # Load arguments from a pickle file
            with (path / "init_kwargs.pkl").open("rb") as f:
                args = pickle.load(f)
            obs_space = args["obs_space"]
            action_dim = args["action_dim"]
        else:
            # load argeuments from a json file
            with (path / "init_kwargs.json").open("r") as f:
                args = json.load(f)
            obs_space = json_to_space(args["obs_space"])
            action_dim = args["action_dim"]

        config = cls.config_class(**loaded_config)
        agent = config.build(obs_space, action_dim)
        optimizers = torch.load(str(path / "optimizers.pth"), weights_only=True)
        for k, v in optimizers.items():
            getattr(agent, k).load_state_dict(v)
        safetensors.torch.load_model(agent._model, path / "model/model.safetensors", device=device)
        agent._model.train()
        agent._model.requires_grad_(True)
        z_path = path / "z.pth"
        if z_path.exists():
            z_state = torch.load(z_path, map_location=device)
            agent.z = torch.nn.Parameter(z_state["z"].to(agent.device))
            agent.z_optimizer = torch.optim.Adam([agent.z], lr=1e-5)
            agent.z_optimizer.load_state_dict(z_state["z_optimizer"])
        return agent

    def save(self, output_folder: str) -> None:
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True, parents=True)
        json_dump = self.cfg.model_dump()
        with (output_folder / "config.json").open("w+") as f:
            json.dump(json_dump, f, indent=4)
        # save optimizer
        torch.save(
            self.optimizer_dict,
            output_folder / "optimizers.pth",
        )
        # save model
        model_folder = output_folder / "model"
        model_folder.mkdir(exist_ok=True)
        self._model.save(output_folder=str(model_folder))

        # Save the arguments required to create this agent (in addition to the config)
        init_kwargs = {
            "obs_space": space_to_json(self.obs_space),
            "action_dim": self.action_dim,
        }
        with (output_folder / "init_kwargs.json").open("w") as f:
            json.dump(init_kwargs, f, indent=4)

        if hasattr(self, "z"):
            torch.save(
                {
                    "z": self.z.detach().cpu(),
                    "z_optimizer": self.z_optimizer.state_dict(),
                },
                output_folder / "z.pth",
            )