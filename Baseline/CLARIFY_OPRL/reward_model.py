from dataclasses import dataclass, field
from typing import List
import pickle
import gym
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import d4rl # Import required to register environments
import deepdish as dd


DEBUG = False
PROB_MARGIN = 0.1



def gen_net(in_size=1, out_size=1, H=128, n_layers=3, activation='tanh'):
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(in_size, H))
        net.append(nn.LeakyReLU())
        in_size = H
    net.append(nn.Linear(in_size, out_size))
    if activation == 'tanh':
        net.append(nn.Tanh())
    elif activation == 'sig':
        net.append(nn.Sigmoid())
    else:
        net.append(nn.ReLU())

    return net


def remove_outliers(distances, threshold=3.0):
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    return np.abs(distances - mean_distance) < threshold * std_distance


def compute_distribution(distances, num_discrete=32):
    hist, bin_edges = np.histogram(distances, bins=num_discrete, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist = hist / np.sum(hist)  # 归一化
    return bin_centers, hist


def compute_comparable_distribution_in_bins(new_distance, bin_centers, num_discrete=32):
    new_hist = np.zeros(num_discrete)
    
    for i, center in enumerate(bin_centers):
        lower_bound = bin_centers[i] - (bin_centers[1] - bin_centers[0]) / 2
        upper_bound = bin_centers[i] + (bin_centers[1] - bin_centers[0]) / 2
        new_hist[i] = np.sum((new_distance >= lower_bound) & (new_distance < upper_bound))
    
    new_hist /= (np.sum(new_hist) + 1e-6)
    return new_hist


def reject_sampling(distances, original_hist, target_hist, bin_centers, num_discrete=32):
    original_density = np.interp(distances, bin_centers, original_hist)
    target_density = np.interp(distances, bin_centers, target_hist)
    acceptance_prob = target_density / (original_density + 1e-6)
    acceptance_prob = np.clip(acceptance_prob, 0, 1)
    random_values = np.random.rand(len(distances))
    idx = random_values < acceptance_prob
    return np.where(idx)[0]



class RewardModel:
    def __init__(self, ds, da, device, trajectories, max_ep_len=500,
                 ensemble_size=3, lr=3e-4, mb_size=128, size_segment=1, activation='tanh',
                 capacity=5e5, large_batch=10,
                 teacher_beta=-1, teacher_gamma=1, 
                 teacher_eps_mistake=0, teacher_eps_skip=0, teacher_eps_equal=0,
                 use_skip_label=False,
                 ):
        # train data is trajectories, must process to sa and s..   
        self.ds = ds
        self.da = da
        self.max_ep_len = max_ep_len
        self.device = device
        self.de = ensemble_size
        self.lr = lr
        self.ensemble = []
        self.paramlst = []
        self.opt = None
        self.activation = activation
        self.size_segment = size_segment

        self.capacity = int(capacity)
        self.use_onehot_pref_label = True
        if self.use_onehot_pref_label:
            self.get_label = self.get_onehot_label
        self.buffer_seg1 = np.empty((self.capacity, size_segment, self.ds+self.da), dtype=np.float32)
        self.buffer_seg2 = np.empty((self.capacity, size_segment, self.ds+self.da), dtype=np.float32)
        self.buffer_reward1 = np.empty((self.capacity, size_segment, 1), dtype=np.float32)
        self.buffer_reward2 = np.empty((self.capacity, size_segment, 1), dtype=np.float32)
        if self.use_onehot_pref_label:
            self.buffer_label = np.empty((self.capacity, 2), dtype=np.float32)
        else:
            self.buffer_label = np.empty((self.capacity, 1), dtype=np.float32)
        self.buffer_index = 0
        self.buffer_full = False

        self.construct_ensemble()
        self.mb_size = mb_size
        self.origin_mb_size = mb_size
        self.large_batch = large_batch
        self.train_batch_size = 128
        if self.use_onehot_pref_label:
            self.CEloss = nn.CrossEntropyLoss()
            # self.CEloss = nn.BCEWithLogitsLoss()
        else:
            self.CEloss = nn.CrossEntropyLoss()
        self.process_offline_data(trajectories)

        self.num_discrete = 32

        # new teacher
        self.teacher_beta = teacher_beta
        self.teacher_gamma = teacher_gamma
        self.teacher_eps_mistake = teacher_eps_mistake
        self.teacher_eps_equal = teacher_eps_equal
        self.teacher_eps_skip = teacher_eps_skip
        self.teacher_thres_skip = 0
        self.teacher_thres_equal = 0
        self.use_skip_label = use_skip_label
        self.set_teacher_thres_skip(self.mean_reward)

    def process_offline_data(self, trajectories):
        # process the trajectories into inputs and targets
        self.inputs = []
        self.targets = []
        for traj in trajectories:
            states = traj['observations']
            actions = traj['actions']
            rewards = traj['rewards']
            sa_t = np.concatenate((states, actions), axis=-1)
            self.inputs.append(sa_t)
            self.targets.append(rewards)
        
        self.inputs = np.stack(self.inputs, axis=0).astype(np.float32)  # (1500, 500, 43)
        self.targets = np.stack(self.targets, axis=0).astype(np.float32)  # (1500, 500)
        self.len_traj = self.inputs.shape[1]  # 500
        self.sum_rewards = np.sum(self.targets, axis=1).astype(np.float32)  # (1500,)
        self.mean_reward = np.mean(self.targets)
        print(f'Processed offline data, inputs shape: {self.inputs.shape}, targets shape: {self.targets.shape}')

    def set_bdt_model(self, model):
        self.bdt_model = model

    def construct_ensemble(self):
        for i in range(self.de):
            model = nn.Sequential(*gen_net(in_size=self.ds+self.da, 
                                           out_size=1, H=256, n_layers=3, 
                                           activation=self.activation)).float().to(self.device)
            self.ensemble.append(model)
            self.paramlst.extend(model.parameters())
            
        self.opt = torch.optim.Adam(self.paramlst, lr = self.lr)

    def set_teacher_thres_skip(self, new_margin):
        self.teacher_thres_skip = new_margin * self.teacher_eps_skip * self.size_segment
        # self.teacher_thres_skip = new_margin * self.teacher_eps_skip * self.len_traj
        
    def p_hat_member(self, x_1, x_2, member=-1):
        # softmaxing to get the probabilities according to eqn 1
        with torch.no_grad():
            r_hat1 = self.r_hat_member(x_1, member=member)
            r_hat2 = self.r_hat_member(x_2, member=member)
            r_hat1 = r_hat1.sum(axis=1)
            r_hat2 = r_hat2.sum(axis=1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
        
        # taking 0 index for probability x_1 > x_2
        return F.softmax(r_hat, dim=-1)[:,0]
    
    def r_hat(self, x):
        # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the rewards are already normalized
        # but I don't understand how the normalization should be happening right now :(
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)
        return np.mean(r_hats)
    
    def r_hat_disagreement(self, x):
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member_ndarray(x, member=member).detach())
        r_hats = torch.cat(r_hats, axis=-1)

        return torch.mean(r_hats, axis=-1), torch.std(r_hats, axis=-1)
    
    def r_hat_member(self, x, member=-1):
        # the network parameterizes r hat in eqn 1 from the paper
        return self.ensemble[member](torch.from_numpy(x).float().to(self.device))

    def r_hat_member_ndarray(self, x, member=-1):
        # the network parameterizes r hat in eqn 1 from the paper
        return self.ensemble[member](x)

    def get_rank_probability(self, x_1, x_2):
        # get probability x_1 > x_2
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_member(x_1, x_2, member=member).cpu().numpy())
        probs = np.array(probs)
        
        return np.mean(probs, axis=0), np.std(probs, axis=0)

    def shuffle_dataset(self, max_len):
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))
        return total_batch_index

    def train_reward(self):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])

        max_len = self.capacity if self.buffer_full else self.buffer_index
        bf_seg1 = self.buffer_seg1[:max_len]
        bf_seg2 = self.buffer_seg2[:max_len]
        bf_labels = self.buffer_label[:max_len]
        if self.use_onehot_pref_label and (not self.use_skip_label):
            skip_index = (bf_labels[:, 0] == 0.5) * (bf_labels[:, 1] == 0.5)
            bf_seg1 = bf_seg1[~skip_index]
            bf_seg2 = bf_seg2[~skip_index]
            bf_labels = bf_labels[~skip_index]
            # print(f'Number of total queries: {max_len}, skipped queries: {sum(skip_index)}')
            max_len = bf_labels.shape[0]

        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))
        num_epochs = int(np.ceil(max_len / self.train_batch_size))
        total = 0

        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0
            last_index = min((epoch + 1) * self.train_batch_size, max_len)

            for member in range(self.de):

                # get random batch
                idxs = total_batch_index[member][epoch * self.train_batch_size:last_index]
                sa_t_1 = bf_seg1[idxs]
                sa_t_2 = bf_seg2[idxs]
                labels = bf_labels[idxs]
                if self.use_onehot_pref_label:
                    labels = torch.from_numpy(labels).float().to(self.device)
                else:
                    labels = torch.from_numpy(labels.flatten()).long().to(self.device)
                if member == 0:
                    total += labels.size(0)

                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                # compute loss
                curr_loss = self.CEloss(r_hat, labels)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())

                # compute acc
                if self.use_onehot_pref_label:  # don't count [0.5, 0.5] queries

                    r_hat_prob = F.softmax(r_hat, dim=-1)[:,0]
                    predicted_10 = (r_hat_prob > 0.5 + PROB_MARGIN).float()
                    predicted_01 = (r_hat_prob < 0.5 - PROB_MARGIN).float()
                    predicted_55 = (r_hat_prob > 0.5 - PROB_MARGIN).float() * (r_hat_prob < 0.5 + PROB_MARGIN).float() * 0.5
                    correct_10 = (predicted_10 * (predicted_10 == labels[:,0]).float()).sum().item()
                    correct_01 = (predicted_01 * (predicted_01 == labels[:,1]).float()).sum().item()
                    correct_55 = (predicted_55 * (predicted_55 == labels[:,1]).float()).sum().item() * 2
                    correct = correct_10 + correct_01 + correct_55
                    # _, predicted = torch.max(r_hat.data, 1)
                    # correct = (predicted == labels).sum().item()
                    if DEBUG:
                        print(f'r_hat_prob: {r_hat_prob}')
                        print(f'predicted_10: {predicted_10}')
                        print(f'predicted_01: {predicted_01}')
                        print(f'predicted_55: {predicted_55}')
                        print(f'correct_10: {correct_10}')
                        print(f'correct_01: {correct_01}')
                        print(f'correct_55: {correct_55}')
                        print(f'correct: {correct}')
                else:
                    _, predicted = torch.max(r_hat.data, 1)
                    correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct

            loss.backward()
            self.opt.step()

        ensemble_acc = ensemble_acc / total

        return ensemble_acc, np.mean(ensemble_losses)

    def save_reward_model(self, path):
        torch.save(self.ensemble, path)
        print(f'Model saved at {path}')
    
    def load_reward_model(self, path):
        self.ensemble = torch.load(path)
        self.paramlst = []
        for model in self.ensemble:
            model.to(self.device)
            self.paramlst.extend(model.parameters())
        self.opt = torch.optim.Adam(self.paramlst, lr = self.lr)

    def test_reward_model_accuracy(self):
        ''' test the reward model on the test set '''
        sa_t_1, sa_t_2, r_t_1, r_t_2, tr_1, tr_2 = self.get_queries(self.train_batch_size)
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels, _ = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2, tr_1, tr_2, use_skip=False)
        ensemble_acc = np.array([0 for _ in range(self.de)])
        total = r_t_1.shape[0]
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels).float().to(self.device)
        labels = torch.argmax(labels, dim=1)

        for member in range(self.de):
            # get logits
            r_hat1 = self.r_hat_member(sa_t_1, member=member)
            r_hat2 = self.r_hat_member(sa_t_2, member=member)
            r_hat1 = r_hat1.sum(axis=1)
            r_hat2 = r_hat2.sum(axis=1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

            # compute acc
            _, predicted = torch.max(r_hat.data, 1)
            correct = (predicted == labels).sum().item()
            ensemble_acc[member] += correct

        ensemble_acc = ensemble_acc / total

        return np.mean(ensemble_acc)

    def generate_traj_reward(self, states: np.ndarray, actions: np.ndarray):
        ''' generate rewards for a trajectory '''
        sa_t = np.concatenate((states, actions), axis=-1)
        rewards = []
        for member in range(self.de):
            rewards.append(self.r_hat_member(sa_t, member=member).detach().cpu().numpy())
        rewards = np.array(rewards)
        return np.mean(rewards, axis=0)

    def change_batch(self, new_frac):
        self.mb_size = int(self.origin_mb_size*new_frac)
    
    def set_batch(self, new_batch):
        self.mb_size = int(new_batch)

    def get_queries(self, mb_size=20):
        len_traj, max_len = len(self.inputs[0]), len(self.inputs)
        
        # get train traj
        train_inputs = np.array(self.inputs[:max_len])
        train_targets = np.array(self.targets[:max_len])
        
        batch_index_1 = np.random.choice(max_len, size=mb_size, replace=True)
        sa_t_1 = train_inputs[batch_index_1] # Batch x T x dim of s&a
        r_t_1 = train_targets[batch_index_1] # Batch x T x 1
   
        batch_index_2 = np.random.choice(max_len, size=mb_size, replace=True)
        sa_t_2 = train_inputs[batch_index_2] # Batch x T x dim of s&a
        r_t_2 = train_targets[batch_index_2] # Batch x T x 1
                
        sa_t_1 = sa_t_1.reshape(-1, sa_t_1.shape[-1]) # (Batch x T) x dim of s&a
        r_t_1 = r_t_1.reshape(-1, 1) # (Batch x T) x 1
        sa_t_2 = sa_t_2.reshape(-1, sa_t_2.shape[-1]) # (Batch x T) x dim of s&a
        r_t_2 = r_t_2.reshape(-1, 1) # (Batch x T) x 1

        # Generate time index 
        time_index = np.array([list(range(i*len_traj,
                                          i*len_traj+self.size_segment)) for i in range(mb_size)])
        time_index_2 = time_index + np.random.choice(len_traj-self.size_segment, size=mb_size, replace=True).reshape(-1,1)
        time_index_1 = time_index + np.random.choice(len_traj-self.size_segment, size=mb_size, replace=True).reshape(-1,1)
        
        sa_t_1 = np.take(sa_t_1, time_index_1, axis=0) # Batch x size_seg x dim of s&a
        r_t_1 = np.take(r_t_1, time_index_1, axis=0) # Batch x size_seg x 1
        sa_t_2 = np.take(sa_t_2, time_index_2, axis=0) # Batch x size_seg x dim of s&a
        r_t_2 = np.take(r_t_2, time_index_2, axis=0) # Batch x size_seg x 1
        tr_1 = self.sum_rewards[batch_index_1].reshape(-1, 1)  # Batch x 1
        tr_2 = self.sum_rewards[batch_index_2].reshape(-1, 1)
                
        return sa_t_1, sa_t_2, r_t_1, r_t_2, tr_1, tr_2

    def get_label(self, sa_t_1, sa_t_2, r_t_1, r_t_2, use_skip=True):
        raise NotImplementedError
        metrics = {}

        sum_r_t_1 = np.sum(r_t_1, axis=1)
        sum_r_t_2 = np.sum(r_t_2, axis=1)
        
        # skip the query
        if use_skip and self.teacher_thres_skip > 0:  # isn't the original skip teacher
            # we skip the query if return diff is too small
            sum_r_diff = np.mean(sum_r_t_1 - sum_r_t_2)
            max_index = (sum_r_diff > self.teacher_thres_skip).reshape(-1)
            metrics['skip_ratio'] = sum(max_index) / len(max_index)
            if sum(max_index) == 0:
                return None, None, None, None, [], metrics

            sa_t_1 = sa_t_1[max_index]
            sa_t_2 = sa_t_2[max_index]
            r_t_1 = r_t_1[max_index]
            r_t_2 = r_t_2[max_index]
            sum_r_t_1 = np.sum(r_t_1, axis=1)
            sum_r_t_2 = np.sum(r_t_2, axis=1)

        # equally preferable
        margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) < self.teacher_thres_equal).reshape(-1)
        
        # perfectly rational
        seg_size = r_t_1.shape[1]
        temp_r_t_1 = r_t_1.copy()
        temp_r_t_2 = r_t_2.copy()
        for index in range(seg_size-1):
            temp_r_t_1[:,:index+1] *= self.teacher_gamma
            temp_r_t_2[:,:index+1] *= self.teacher_gamma
        sum_r_t_1 = np.sum(temp_r_t_1, axis=1)
        sum_r_t_2 = np.sum(temp_r_t_2, axis=1)
            
        rational_labels = 1*(sum_r_t_1 < sum_r_t_2)
        if self.teacher_beta > 0: # Bradley-Terry rational model
            r_hat = torch.cat([torch.Tensor(sum_r_t_1), 
                               torch.Tensor(sum_r_t_2)], axis=-1)
            r_hat = r_hat*self.teacher_beta
            ent = F.softmax(r_hat, dim=-1)[:, 1]
            labels = torch.bernoulli(ent).int().numpy().reshape(-1, 1)
        else:
            labels = rational_labels
        
        # making a random mistake
        len_labels = labels.shape[0]
        rand_num = np.random.rand(len_labels)
        noise_index = rand_num <= self.teacher_eps_mistake
        labels[noise_index] = 1 - labels[noise_index]
 
        # equally preferable
        labels[margin_index] = -1 

        return sa_t_1, sa_t_2, r_t_1, r_t_2, labels, metrics

    def get_onehot_label(self, sa_t_1, sa_t_2, r_t_1, r_t_2, tr_1, tr_2, use_skip=True):
        ''' return [0, 1] [1, 0] [0.5, 0.5] '''
        metrics = {}
        sum_r_t_1 = np.sum(r_t_1, axis=1)
        sum_r_t_2 = np.sum(r_t_2, axis=1)

        # skip the query
        if use_skip and self.teacher_thres_skip > 0:  # isn't the original skip teacher
            # we skip the query if return diff is too small
            sum_r_diff = np.abs(sum_r_t_1 - sum_r_t_2)
            # traj_r_diff = np.abs(tr_1 - tr_2)
            max_index = (sum_r_diff > self.teacher_thres_skip).reshape(-1)
            # max_index = (traj_r_diff > self.teacher_thres_skip).reshape(-1)
            metrics['skip_ratio'] = sum(max_index) / len(max_index)

        # perfectly rational
        # rational_labels = 1 * (sum_r_t_1 < sum_r_t_2)
        rational_labels = np.zeros((len(sum_r_t_1), 2))
        rational_labels[sum_r_t_1[:, 0] < sum_r_t_2[:, 0]] = [0, 1]
        rational_labels[sum_r_t_1[:, 0] > sum_r_t_2[:, 0]] = [1, 0]
        rational_labels[sum_r_t_1[:, 0] == sum_r_t_2[:, 0]] = [0.5, 0.5]

        if self.teacher_beta > 0:  # Bradley-Terry rational model # our teacher_beta = -1
            r_hat = torch.cat([torch.Tensor(sum_r_t_1),
                               torch.Tensor(sum_r_t_2)], axis=-1)
            r_hat = r_hat * self.teacher_beta  # -r_hat
            ent = F.softmax(r_hat, dim=-1)[:, 1]
            labels = torch.bernoulli(ent).int().numpy().reshape(-1, 1)
            # labels = np.concatenate([1 - labels, labels], axis=1)  # verified
            labels = torch.concat([1 - labels, labels], dim=1)  # verified
        else:
            labels = rational_labels
        
        # skip the query
        if use_skip and self.teacher_thres_skip > 0:  # isn't the original skip teacher
            labels[~max_index] = [0.5, 0.5]

        return sa_t_1, sa_t_2, r_t_1, r_t_2, labels, metrics

    def put_queries(self, sa_t_1, sa_t_2, r_t_1, r_t_2, labels):
        total_sample = sa_t_1.shape[0]
        next_index = self.buffer_index + total_sample
        if next_index >= self.capacity:
            self.buffer_full = True
            maximum_index = self.capacity - self.buffer_index
            np.copyto(self.buffer_seg1[self.buffer_index:self.capacity], sa_t_1[:maximum_index])
            np.copyto(self.buffer_seg2[self.buffer_index:self.capacity], sa_t_2[:maximum_index])
            np.copyto(self.buffer_reward1[self.buffer_index:self.capacity], r_t_1[:maximum_index])
            np.copyto(self.buffer_reward2[self.buffer_index:self.capacity], r_t_2[:maximum_index])
            np.copyto(self.buffer_label[self.buffer_index:self.capacity], labels[:maximum_index])

            remain = total_sample - (maximum_index)
            if remain > 0:
                np.copyto(self.buffer_seg1[0:remain], sa_t_1[maximum_index:])
                np.copyto(self.buffer_seg2[0:remain], sa_t_2[maximum_index:])
                np.copyto(self.buffer_reward1[0:remain], r_t_1[maximum_index:])
                np.copyto(self.buffer_reward2[0:remain], r_t_2[maximum_index:])
                np.copyto(self.buffer_label[0:remain], labels[maximum_index:])

            self.buffer_index = remain
        else:
            np.copyto(self.buffer_seg1[self.buffer_index:next_index], sa_t_1)
            np.copyto(self.buffer_seg2[self.buffer_index:next_index], sa_t_2)
            np.copyto(self.buffer_reward1[self.buffer_index:next_index], r_t_1)
            np.copyto(self.buffer_reward2[self.buffer_index:next_index], r_t_2)
            np.copyto(self.buffer_label[self.buffer_index:next_index], labels)
            self.buffer_index = next_index

    def uniform_sampling(self):
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2, tr_1, tr_2 = self.get_queries(
            mb_size=self.mb_size)

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels, metrics = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2, tr_1, tr_2)
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, r_t_1, r_t_2, labels)
        
        return len(labels), metrics
    
    def disagreement_sampling(self):
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2, tr_1, tr_2 = self.get_queries(
            mb_size=self.mb_size*self.large_batch)
        
        # get final queries based on uncertainty
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:self.mb_size]
        r_t_1, sa_t_1, tr_1 = r_t_1[top_k_index], sa_t_1[top_k_index], tr_1[top_k_index]
        r_t_2, sa_t_2, tr_2 = r_t_2[top_k_index], sa_t_2[top_k_index], tr_2[top_k_index]
        
        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels, metrics = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2, tr_1, tr_2)        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, r_t_1, r_t_2, labels)
        
        return len(labels), metrics

    def sample_learned_queries(self, batch_size):
        max_pref_len = self.capacity if self.buffer_full else self.buffer_index
        bf_seg1 = self.buffer_seg1[:max_pref_len]
        bf_seg2 = self.buffer_seg2[:max_pref_len]
        bf_labels = self.buffer_label[:max_pref_len]

        sample_idxes = np.random.choice(max_pref_len, batch_size, replace=True)
        sa_t_1, sa_t_2, labels = bf_seg1[sample_idxes], bf_seg2[sample_idxes], bf_labels[sample_idxes]
        states_1, states_2 = sa_t_1[:, :, :self.ds], sa_t_2[:, :, :self.ds]
        states_1 = torch.from_numpy(states_1).to(dtype=torch.float32, device=self.device)
        states_2 = torch.from_numpy(states_2).to(dtype=torch.float32, device=self.device)
        
        random_time_start = np.random.randint(0, self.max_ep_len - self.size_segment - 1, size=(batch_size, 2))
        timesteps_1 = np.arange(self.size_segment).reshape(1, -1) + random_time_start[:, 0].reshape(-1, 1)
        timesteps_2 = np.arange(self.size_segment).reshape(1, -1) + random_time_start[:, 1].reshape(-1, 1)
        timesteps_1 = torch.from_numpy(timesteps_1).to(dtype=torch.long, device=self.device)
        timesteps_2 = torch.from_numpy(timesteps_2).to(dtype=torch.long, device=self.device)

        attention_mask_1 = torch.ones((batch_size, self.size_segment), dtype=torch.float32, device=self.device)
        attention_mask_2 = torch.ones((batch_size, self.size_segment), dtype=torch.float32, device=self.device)
        # return states_1, timesteps_1, attention_mask_1, states_2, timesteps_2, attention_mask_2, labels

        phi_1 = self.bdt_model.get_embedding(states_1, timesteps_1, attention_mask_1).detach().cpu().numpy()
        phi_2 = self.bdt_model.get_embedding(states_2, timesteps_2, attention_mask_2).detach().cpu().numpy()
        distances = np.linalg.norm(phi_1 - phi_2, axis=1)
        return distances, labels
    
    def convert_queries_to_embeddings(self, sa_t_1, sa_t_2):
        states_1, states_2 = sa_t_1[:, :, :self.ds], sa_t_2[:, :, :self.ds]
        states_1 = torch.from_numpy(states_1).to(dtype=torch.float32, device=self.device)
        states_2 = torch.from_numpy(states_2).to(dtype=torch.float32, device=self.device)

        random_time_start = np.random.randint(0, self.max_ep_len - self.size_segment - 1, size=(states_1.shape[0], 2))
        timesteps_1 = np.arange(self.size_segment).reshape(1, -1) + random_time_start[:, 0].reshape(-1, 1)
        timesteps_2 = np.arange(self.size_segment).reshape(1, -1) + random_time_start[:, 1].reshape(-1, 1)
        timesteps_1 = torch.from_numpy(timesteps_1).to(dtype=torch.long, device=self.device)
        timesteps_2 = torch.from_numpy(timesteps_2).to(dtype=torch.long, device=self.device)

        attention_mask_1 = torch.ones((states_1.shape[0], self.size_segment), dtype=torch.float32, device=self.device)
        attention_mask_2 = torch.ones((states_1.shape[0], self.size_segment), dtype=torch.float32, device=self.device)

        phi_1 = self.bdt_model.get_embedding(states_1, timesteps_1, attention_mask_1).detach().cpu().numpy()
        phi_2 = self.bdt_model.get_embedding(states_2, timesteps_2, attention_mask_2).detach().cpu().numpy()
        distances = np.linalg.norm(phi_1 - phi_2, axis=1)
        return distances

    def contrastive_sampling(self):
        distances, labels = self.sample_learned_queries(self.mb_size * self.large_batch)
        idx = remove_outliers(distances)  # remove outliers
        distances, labels = distances[idx], labels[idx]
        bin_centers, hist = compute_distribution(distances, num_discrete=self.num_discrete)

        comparable = (labels[:, 0] != 0.5)
        # comparable
        dist_comparable = distances[comparable]
        hist_comparable = compute_comparable_distribution_in_bins(
            dist_comparable, bin_centers, num_discrete=self.num_discrete)
        hist_comparable_density = hist_comparable / (hist + 1e-6)
        hist_comparable_density /= (np.sum(hist_comparable_density) + 1e-6)
        # non-comparable
        dist_non_comparable = distances[~comparable]
        hist_non_comparable = compute_comparable_distribution_in_bins(
            dist_non_comparable, bin_centers, num_discrete=self.num_discrete)
        hist_non_comparable_density = hist_non_comparable / (hist + 1e-6)
        hist_non_comparable_density /= (np.sum(hist_non_comparable_density) + 1e-6)

        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2, tr_1, tr_2 = self.get_queries(
            mb_size=self.mb_size * self.large_batch)
        new_distances = self.convert_queries_to_embeddings(sa_t_1, sa_t_2)
        idx = remove_outliers(new_distances)  # remove outliers
        sa_t_1, sa_t_2, r_t_1, r_t_2, tr_1, tr_2, new_distances = \
            sa_t_1[idx], sa_t_2[idx], r_t_1[idx], r_t_2[idx], tr_1[idx], tr_2[idx], new_distances[idx]
        
        new_hist = compute_comparable_distribution_in_bins(
            new_distances, bin_centers, num_discrete=self.num_discrete)
        target_hist_minus = np.clip((hist_comparable_density - hist_non_comparable_density), 0, None) * hist
        target_hist_minus /= (np.sum(target_hist_minus) + 1e-6)
        target_hist_div = np.clip((hist_comparable_density / (hist_non_comparable_density + 1e-6)), 0, None) * hist
        target_hist_div /= (np.sum(target_hist_div) + 1e-6)
        target_hist = (target_hist_minus + target_hist_div) * 0.5

        # reject sampling
        idx = reject_sampling(new_distances, new_hist, target_hist, bin_centers, 
                              num_discrete=self.num_discrete)
        sa_t_1, sa_t_2, r_t_1, r_t_2, tr_1, tr_2, new_distances = \
            sa_t_1[idx], sa_t_2[idx], r_t_1[idx], r_t_2[idx], tr_1[idx], tr_2[idx], new_distances[idx]

        # get final queries based on uncertainty
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:self.mb_size]
        r_t_1, sa_t_1, tr_1 = r_t_1[top_k_index], sa_t_1[top_k_index], tr_1[top_k_index]
        r_t_2, sa_t_2, tr_2 = r_t_2[top_k_index], sa_t_2[top_k_index], tr_2[top_k_index]
        
        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels, metrics = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2, tr_1, tr_2)
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, r_t_1, r_t_2, labels)
        
        return len(labels), metrics

    def max_ratio_sampling(self):
        distances, labels = self.sample_learned_queries(self.mb_size * 10 * self.large_batch)
        idx = remove_outliers(distances)  # remove outliers
        distances, labels = distances[idx], labels[idx]
        bin_centers, hist = compute_distribution(distances, num_discrete=self.num_discrete)

        comparable = (labels[:, 0] != 0.5)
        # comparable
        dist_comparable = distances[comparable]
        hist_comparable = compute_comparable_distribution_in_bins(
            dist_comparable, bin_centers, num_discrete=self.num_discrete)
        hist_comparable_density = hist_comparable / (hist + 1e-6)
        hist_comparable_density /= np.sum(hist_comparable_density)
        # non-comparable
        dist_non_comparable = distances[~comparable]
        hist_non_comparable = compute_comparable_distribution_in_bins(
            dist_non_comparable, bin_centers, num_discrete=self.num_discrete)
        hist_non_comparable_density = hist_non_comparable / (hist + 1e-6)
        hist_non_comparable_density /= np.sum(hist_non_comparable_density)

        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2, tr_1, tr_2 = self.get_queries(
            mb_size=self.mb_size * self.large_batch)
        new_distances = self.convert_queries_to_embeddings(sa_t_1, sa_t_2)
        idx = remove_outliers(new_distances)  # remove outliers
        sa_t_1, sa_t_2, r_t_1, r_t_2, tr_1, tr_2, new_distances = \
            sa_t_1[idx], sa_t_2[idx], r_t_1[idx], r_t_2[idx], tr_1[idx], tr_2[idx], new_distances[idx]
        
        new_hist = compute_comparable_distribution_in_bins(
            new_distances, bin_centers, num_discrete=self.num_discrete)
        # target_hist = 1 only when hist_comparable_density is the maximum
        comparable_ratio = hist_comparable_density / (hist_non_comparable_density + 1e-6)
        target_hist = (comparable_ratio == np.max(comparable_ratio)).astype(np.float32) + 1e-3

        # reject sampling
        idx = reject_sampling(new_distances, new_hist, target_hist, bin_centers, 
                              num_discrete=self.num_discrete)
        sa_t_1, sa_t_2, r_t_1, r_t_2, tr_1, tr_2, new_distances = \
            sa_t_1[idx], sa_t_2[idx], r_t_1[idx], r_t_2[idx], tr_1[idx], tr_2[idx], new_distances[idx]

        # get final queries based on uncertainty
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:self.mb_size]
        r_t_1, sa_t_1, tr_1 = r_t_1[top_k_index], sa_t_1[top_k_index], tr_1[top_k_index]
        r_t_2, sa_t_2, tr_2 = r_t_2[top_k_index], sa_t_2[top_k_index], tr_2[top_k_index]
        
        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels, metrics = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2, tr_1, tr_2)
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, r_t_1, r_t_2, labels)
        
        return len(labels), metrics

