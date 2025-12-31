import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
import math
import re
import numpy as np

def load_queries_with_indices(dataset, num_query, len_query, saved_indices, saved_labels=None, label_type=1,
                               equivalence_threshold=0, partition_idx=None):
    total_reward_seq_1, total_reward_seq_2 = np.zeros((num_query, len_query)), np.zeros((num_query, len_query))
    
    observation_dim = (dataset["observations"].shape[-1], )
    action_dim = dataset["actions"].shape[-1]

    total_obs_seq_1, total_obs_seq_2 = np.zeros((num_query, len_query) + observation_dim), np.zeros(
        (num_query, len_query) + observation_dim)
    total_act_seq_1, total_act_seq_2 = np.zeros((num_query, len_query, action_dim)), np.zeros(
        (num_query, len_query, action_dim))
    total_timestep_1, total_timestep_2 = np.zeros((num_query, len_query), dtype=np.int32), np.zeros(
        (num_query, len_query), dtype=np.int32)
    
    if saved_labels is None:
        query_range = np.arange(num_query)
    else:
        # do not query all label
        if partition_idx is None:
            query_range = np.arange(len(saved_labels) - num_query, len(saved_labels))
        else:
            # If dataset is large, you should load the dataset in slices.
            query_range = np.arange(partition_idx * num_query, (partition_idx + 1) * num_query)

    for query_count, i in enumerate(query_range):
        temp_count = 0
        while temp_count < 2:
            start_idx = saved_indices[temp_count][i]
            end_idx = start_idx + len_query
            
            reward_seq = dataset['rewards'][start_idx:end_idx]
            obs_seq = dataset['observations'][start_idx:end_idx]
            act_seq = dataset['actions'][start_idx:end_idx]
            timestep_seq = np.arange(1, len_query + 1)
            
            if temp_count == 0:
                total_reward_seq_1[query_count] = reward_seq
                total_obs_seq_1[query_count] = obs_seq
                total_act_seq_1[query_count] = act_seq
                total_timestep_1[query_count] = timestep_seq
            else:
                total_reward_seq_2[query_count] = reward_seq
                total_obs_seq_2[query_count] = obs_seq
                total_act_seq_2[query_count] = act_seq
                total_timestep_2[query_count] = timestep_seq
            
            temp_count += 1
    
    seg_reward_1 = total_reward_seq_1.copy()
    seg_reward_2 = total_reward_seq_2.copy()
    
    seg_obs_1 = total_obs_seq_1.copy()
    seg_obs_2 = total_obs_seq_2.copy()
    
    seq_act_1 = total_act_seq_1.copy()
    seq_act_2 = total_act_seq_2.copy()
    
    seq_timestep_1 = total_timestep_1.copy()
    seq_timestep_2 = total_timestep_2.copy()
    
    batch = {}
    # script_labels
    # label_type = 0 perfectly rational / label_type = 1 equivalence_threshold
    if label_type == 0:  # perfectly rational
        sum_r_t_1 = np.sum(seg_reward_1, axis=1)
        sum_r_t_2 = np.sum(seg_reward_2, axis=1)
        binary_label = 1 * (sum_r_t_1 < sum_r_t_2)
        rational_labels = np.zeros((len(binary_label), 2))
        rational_labels[np.arange(binary_label.size), binary_label] = 1.0
    elif label_type == 1:
        sum_r_t_1 = np.sum(seg_reward_1, axis=1)
        sum_r_t_2 = np.sum(seg_reward_2, axis=1)
        binary_label = 1 * (sum_r_t_1 < sum_r_t_2)
        rational_labels = np.zeros((len(binary_label), 2))
        rational_labels[np.arange(binary_label.size), binary_label] = 1.0
        margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) <= equivalence_threshold).reshape(-1)
        rational_labels[margin_index] = 0.5
    batch['script_labels'] = rational_labels

    # human label
    human_labels = np.zeros((len(saved_labels), 2))
    human_labels[np.array(saved_labels) == 0, 0] = 1.
    human_labels[np.array(saved_labels) == 1, 1] = 1.
    human_labels[np.array(saved_labels) == -1] = 0.5
    human_labels = human_labels[query_range]
    batch['labels'] = human_labels
    # print(batch['labels'])
    
    batch['observations'] = seg_obs_1
    batch['actions'] = seq_act_1
    batch['observations_2'] = seg_obs_2
    batch['actions_2'] = seq_act_2
    batch['timestep_1'] = seq_timestep_1
    batch['timestep_2'] = seq_timestep_2
    batch['start_indices'] = saved_indices[0]
    batch['start_indices_2'] = saved_indices[1]
    
    return batch

def get_d4rl_dataset(env):
    import d4rl
    dataset = d4rl.qlearning_dataset(env)
    return dict(
        observations=dataset['observations'],
        actions=dataset['actions'],
        next_observations=dataset['next_observations'],
        rewards=dataset['rewards'],
        dones=dataset['terminals'].astype(np.float32),
    )