import time
import torch
import numpy as np
import torch.nn as nn

from decision_transformer.training.trainer import Trainer
from decision_transformer.training.seq_trainer import SequenceTrainer
from decision_transformer.models.decision_transformer import BidirectionalTransformer

class ContrativeTrainer(SequenceTrainer):
    
    def __init__(
            self,
            model: BidirectionalTransformer,
            optimizer,
            batch_size,
            get_batch,
            get_pref_batch,
            loss_fn,
            z_star, 
            z_star_optimizer,
            similarity_fn='l2',  # 'l2' or 'cosine'
            norm_loss_ratio=0.1,
            comp_loss_ratio=0.1,
            pref_loss_ratio=0.1,
            pref_loss_impl='pairwise',  # 'anchor' or 'pairwise'
            scheduler=None,
            eval_fns=None,
            eval_bdt_z_stars=None,
            embedding_net=None,
            add_rtg=False,
            device='cuda:0',
        ):
        super().__init__(
            model,
            optimizer,
            batch_size,
            get_batch,
            loss_fn,
            scheduler,
            eval_fns,
            embedding_net,
            add_rtg
        )
        self.z_dim = self.model.z_dim
        self.get_pref_batch = get_pref_batch
        self.z_star = z_star  # (z_dim)
        self.z_star_optimizer = z_star_optimizer
        self.eval_bdt_z_stars = [] if eval_bdt_z_stars is None else eval_bdt_z_stars
        if similarity_fn == 'l2':
            self.similarity_fn = lambda x, y: -torch.mean((x - y)**2, dim=1)
        elif similarity_fn == 'cosine':
            self.similarity_fn = nn.CosineSimilarity(dim=1)
        self.norm_loss_ratio = norm_loss_ratio
        self.comp_loss_ratio = comp_loss_ratio
        self.pref_loss_ratio = pref_loss_ratio
        self.pref_loss_impl = pref_loss_impl
        self.anchor_margin = 1.0 if self.pref_loss_impl == 'anchor' else 0.5
        self.device = device


    def train_iteration(self, num_steps, iter_num=0, print_logs=False):
        recon_losses, comp_losses, norm_losses, pref_losses, error_rates = [], [], [], [], []
        logs = dict()

        train_start = time.time()

        self.model.train()
        from tqdm import tqdm
        for _ in tqdm(range(num_steps)):
            recon_loss, comp_loss, norm_loss, pref_loss, error_rate = self.train_step()
            recon_losses.append(recon_loss)
            norm_losses.append(norm_loss)
            comp_losses.append(comp_loss)
            pref_losses.append(pref_loss)
            error_rates.append(error_rate)
            if self.scheduler is not None:
                self.scheduler.step()

        logs['time/training'] = time.time() - train_start

        eval_start = time.time()

        self.model.eval()
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.model, iter_num-1)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v
        for eval_fn in self.eval_bdt_z_stars:
            outputs = eval_fn(self.model, self.z_star, iter_num-1)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        logs['training/recon_loss_mean'] = np.mean(recon_losses)
        logs['training/recon_loss_std'] = np.std(recon_losses)
        logs['training/norm_loss_mean'] = np.mean(norm_losses)
        logs['training/norm_loss_std'] = np.std(norm_losses)
        logs['training/comp_loss_mean'] = np.mean(comp_losses)
        logs['training/comp_loss_std'] = np.std(comp_losses)
        logs['training/pref_loss_mean'] = np.mean(pref_losses)
        logs['training/pref_loss_std'] = np.std(pref_losses)
        logs['training/error_rate_mean'] = np.mean(error_rates)
        logs['training/error_rate_std'] = np.std(error_rates)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

    
    def train_step(self):
        # 1. reconstruction loss
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        action_target = torch.clone(actions)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        recon_loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        self.optimizer.zero_grad()
        recon_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        # 2. norm loss
        full_embedding = self.model.get_full_embedding(
            states, timesteps, attention_mask=attention_mask,
        )
        _, seq_len, _ = full_embedding.shape
        full_embedding = full_embedding.reshape(-1, self.z_dim)
        norm_loss = torch.mean((torch.norm(full_embedding, dim=1) - \
                                torch.ones(self.batch_size * seq_len).to(self.device))**2)

        # 3. comparability contrastive loss
        states_1, actions_1, rewards_1, dones_1, rtg_1, timesteps_1, attention_mask_1, \
        states_2, actions_2, rewards_2, dones_2, rtg_2, timesteps_2, attention_mask_2, \
            pref_label = self.get_pref_batch(self.batch_size)
        embedding_1 = self.model.get_embedding(
            states_1, timesteps_1, attention_mask=attention_mask_1,
        )  # (batch_size, z_dim)
        embedding_2 = self.model.get_embedding(
            states_2, timesteps_2, attention_mask=attention_mask_2,
        )
        embedding_sim = self.similarity_fn(embedding_1, embedding_2)  # (batch_size)

        uncomparable_index = pref_label[:, 0] == 0.5
        comparable_index = ~uncomparable_index
        error_rate = torch.mean(uncomparable_index.float()).item()
        uncomparable_sim = embedding_sim[uncomparable_index]
        comparable_sim = embedding_sim[comparable_index]
        comp_loss = (torch.tensor(0.).to(self.device) if uncomparable_sim.numel() == 0 \
                else torch.mean(torch.sigmoid(uncomparable_sim)) - \
            torch.tensor(0.).to(self.device) if comparable_sim.numel() == 0 \
                else torch.mean(torch.sigmoid(comparable_sim)) )  # uncomp down, comp up

        # 4. preference contrastive loss
        positive_index = pref_label[:, 0] == 1
        negative_index = pref_label[:, 0] == 0
        positive = torch.cat([embedding_1[positive_index], embedding_2[negative_index]], dim=0)
        negative = torch.cat([embedding_1[negative_index], embedding_2[positive_index]], dim=0)
        if positive.shape[0] == 0:
            pref_loss = torch.tensor(0.).to(self.device)
        elif self.pref_loss_impl == 'anchor':
            anchor = self.z_star.repeat(positive.shape[0], 1)  # (comparable_num, z_dim)
            positive_sim = self.similarity_fn(positive, anchor)
            negative_sim = self.similarity_fn(negative, anchor)
            pref_loss = torch.sigmoid(torch.max(torch.tensor(0.), negative_sim - positive_sim + self.anchor_margin))
            pref_loss = torch.mean(pref_loss)
        elif self.pref_loss_impl == 'pairwise':
            random_index = torch.randint(0, positive.shape[0], (50 * positive.shape[0],))
            aug_positive, aug_negative = positive[random_index], negative[random_index]
            positive, negative = positive.repeat(50, 1), negative.repeat(50, 1)
            positive_sim = self.similarity_fn(positive, aug_positive)
            negative_sim = self.similarity_fn(negative, aug_negative)
            pn_sim_1 = self.similarity_fn(positive, aug_negative)
            pn_sim_2 = self.similarity_fn(aug_positive, negative)
            pref_loss = torch.sigmoid(pn_sim_1 + pn_sim_2 - positive_sim - negative_sim)
            pref_loss = torch.mean(pref_loss)

        self.optimizer.zero_grad()
        self.z_star_optimizer.zero_grad()
        (
            self.norm_loss_ratio * norm_loss 
            + self.comp_loss_ratio * comp_loss
            + self.pref_loss_ratio * pref_loss
        ).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()
        self.z_star_optimizer.step()

        return recon_loss.detach().cpu().item(), comp_loss.detach().cpu().item(), \
            norm_loss.detach().cpu().item(), pref_loss.detach().cpu().item(), error_rate

