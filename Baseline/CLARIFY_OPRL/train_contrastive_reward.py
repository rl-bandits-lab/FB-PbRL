import argparse
from dataclasses import dataclass, field
from typing import List
import json
import math
import random
import time
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import d4rl # Import required to register environments
import deepdish as dd
import os
import wandb
from dm_control import suite
import dmc2gym
import mujoco
from dmc_tasks import dmc

import sys
sys.path.append('./')
sys.path.append('./..')
sys.path.append('./../..')
torch.set_num_threads(4)

from reward_model import RewardModel
from decision_transformer.models.decision_transformer import BidirectionalTransformer
from decision_transformer.training.contrastive_trainer import ContrativeTrainer
from env_wrapper import create_env, create_dataset, set_seed_everywhere, discount_cumsum
from tqdm import tqdm
from pathlib import Path

def reward_inference(trajs, domain_name='walker', task_name='walk'):
    env = suite.load(
        domain_name=domain_name,
        task_name=task_name,
        environment_kwargs={"flat_observation": True},
    )
    rewards = []
    for t in tqdm(trajs):
        reward = []
        _ = env.reset()
        for i in range(len(t['observation'])):
            with env._physics.reset_context():
                env._physics.set_state(t["next"]["physics"][i])
                env._physics.set_control(t["action"][i])
            mujoco.mj_forward(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
            mujoco.mj_fwdPosition(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
            mujoco.mj_sensorVel(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
            mujoco.mj_subtreeVel(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
            reward.append(env._task.get_reward(env._physics))
        rewards.append(np.sum(reward))
    return np.mean(rewards), np.std(rewards)

def reward_inference2(trajs, domain_name='walker', task_name='walk'):
    env = dmc2gym.make(
        domain_name=domain_name,
        task_name=task_name,
    )
    rewards = []
    for t in tqdm(trajs):
        reward = []
        _ = env.reset()
        for i in range(len(t['observation'])):
            next_state, re, done, info, _ = env.step(t["action"][i])
            reward.append(re)
        rewards.append(np.sum(reward))
    return np.mean(rewards), np.std(rewards)

def get_rnd_dataset(dataset_path, domain_name, task_name, expl_agent, num_episodes=None):
    """
    Load RND dataset and split into trajectories by termination signal.
    Returns trajectories, states, traj_lens, returns, rewards.
    """
    path = Path(dataset_path) / f"{domain_name}/{expl_agent}/buffer"
    print(f"Data path: {path}")

    files = sorted(path.glob("*.npz"))
    if num_episodes is not None:
        files = files[:num_episodes]

    # === Split into trajectories ===
    trajectories, states, traj_lens, returns, rewards = [], [], [], [], []
    trajectory = {
        "observations": [],
        "next_observations": [],
        "actions": [],
        "rewards": [],
        "terminals": [],
        "next_physics": [],
    }

    for f in tqdm(files, desc="Loading RND data"):
        storage = np.load(str(f))
        trajectory["observations"].append(storage["observation"][:-1].astype(np.float32))
        trajectory["next_observations"].append(storage["observation"][1:].astype(np.float32))
        trajectory["actions"].append(storage["action"][1:].astype(np.float32))
        #trajectory["rewards"].append(storage["reward"][1:].astype(np.float32))
        trajectory["terminals"].append(np.array(1 - storage["discount"][1:], dtype=bool))
        trajectory["next_physics"].append(storage["physics"][1:])

        trajectory["observations"] = np.concatenate(trajectory["observations"], axis=0)
        trajectory["next_observations"] = np.concatenate(trajectory["next_observations"], axis=0)
        trajectory["actions"] = np.concatenate(trajectory["actions"], axis=0)
        #trajectory["rewards"] = np.concatenate(trajectory["rewards"], axis=0)
        trajectory["terminals"] = np.concatenate(trajectory["terminals"], axis=0)
        trajectory["next_physics"] = np.concatenate(trajectory["next_physics"], axis=0)

        '''env = suite.load(
            domain_name=domain_name,
            task_name='walk',
            environment_kwargs={"flat_observation": True},
        )'''
        env = dmc.make(f"{domain_name}_{task_name}")

        for i in range(len(trajectory['observations'])):
            with env._physics.reset_context():
                env._physics.set_state(trajectory["next_physics"][i])
                env._physics.set_control(trajectory["actions"][i])
            mujoco.mj_forward(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
            mujoco.mj_fwdPosition(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
            mujoco.mj_sensorVel(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
            mujoco.mj_subtreeVel(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
            trajectory['rewards'].append(env._task.get_reward(env._physics))
        
        '''env = dmc2gym.make(
            domain_name=domain_name,
            task_name=task_name,
        )
        _ = env.reset()

        for i in range(len(trajectory['observations'])):
            next_state, re, done, info, _ = env.step(trajectory["actions"][i])
            trajectory['rewards'].append(re)'''

        if len(trajectory["observations"]) > 0:
            traj_lens.append(len(trajectory["observations"]))
            rewards.extend(trajectory["rewards"])
            returns.append(sum(trajectory["rewards"]))
            for k in trajectory:
                trajectory[k] = np.array(trajectory[k])
            trajectories.append(trajectory)
            states.append(trajectory["observations"])
        # reset trajectory buffer
        trajectory = {
            "observations": [],
            "next_observations": [],
            "actions": [],
            "rewards": [],
            "terminals": [],
            "next_physics": [],
        }

    traj_lens = np.array(traj_lens)
    returns = np.array(returns)

    return trajectories, states, traj_lens, returns, rewards

def experiment(output_dir, eval_dir, variant):
    gpu = variant.get('gpu', 0)
    device = torch.device(
        f"cuda:{gpu}" if (torch.cuda.is_available() and gpu >= 0) else "cpu"
    )

    env_name, dataset = variant['env'], variant['dataset']
    seed = variant['seed']
    n_bins = variant['n_bins']
    use_contrastive = not variant['no_use_contrastive']
    print(f'use_contrastive: {use_contrastive}')
    gamma = variant['gamma']
    assert gamma == 1.
    z_dim = variant['z_dim']

    # suppose env_name is like 'cheetah-run'
    domain_name, task_name = env_name.split('-')[0], '-'.join(env_name.split('-')[1:])
    print(f'Domain: {domain_name}, Task: {task_name}')
    '''env = suite.load(
        domain_name=domain_name,
        task_name=task_name,
        environment_kwargs={"flat_observation": True},
    )'''
    env = dmc.make(f"{domain_name}_{task_name}")
    scale = 1.
    max_ep_len = 1000
    state_dim = env.observation_spec().shape[0]
    act_dim = env.action_spec().shape[0]
    trajectories, states, traj_lens, returns, rewards = get_rnd_dataset(
        dataset_path="../metamotivo/datasets",
        domain_name=domain_name,
        task_name=task_name,
        expl_agent="rnd",
        num_episodes=5000
    )

    '''env, eval_env, scale, max_ep_len = create_env(env_name, seed)
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]'''
    #trajectories, states, traj_lens, returns, rewards = create_dataset(env_name, dataset)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
    
    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print(f'z-dim: {z_dim}')
    print('=' * 50)

    K = variant['K']
    batch_size = variant['batch_size']

    # for evaluation with best/50% trajectories
    _idxes = np.argsort([np.sum(path['rewards']) for path in trajectories]) # rank 0 is the most bad demo.
    trajs_rank = np.empty_like(_idxes)
    trajs_rank[_idxes] = np.arange(len(_idxes))
    train_indices =  [i for i in range(len(trajs_rank))]


    def get_batch(batch_size=256, max_len=K):
        batch_inds = np.random.choice(
            np.array(train_indices),
            size=batch_size,
            replace=True,
        )
        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(batch_inds[i])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)

            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1)
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)  # (B, K, state_dim)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)  # (B, K, 1)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device) / scale  # (B, K, 1)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, d, rtg, timesteps, mask


    # reward learning config
    max_feedback = variant['max_feedback']
    total_feedback, labeled_feedback = 0, 0
    segment_length = variant['segment_length']
    reward_lr = variant['reward_lr']
    reward_batch = variant['reward_batch']
    reward_update = variant['reward_update']
    feed_type = variant['feed_type']
    teacher_eps_skip = variant['teacher_eps_skip']
    reward_model = RewardModel(ds=state_dim, da=act_dim, device=device, mb_size=reward_batch,
                               size_segment=segment_length, trajectories=trajectories,
                               max_ep_len=max_ep_len, teacher_eps_skip=teacher_eps_skip, lr=reward_lr)


    def get_pref_batch(batch_size=256, max_len=K, reward_model=reward_model):
        max_pref_len = reward_model.capacity if reward_model.buffer_full else reward_model.buffer_index
        batch_inds = np.random.choice(np.arange(max_pref_len), size=batch_size, replace=True)
        sa_t_1 = reward_model.buffer_seg1[batch_inds]
        sa_t_2 = reward_model.buffer_seg2[batch_inds]
        labels = reward_model.buffer_label[batch_inds]
        states_1, actions_1 = sa_t_1[:, :, :state_dim], sa_t_1[:, :, state_dim:]
        states_2, actions_2 = sa_t_2[:, :, :state_dim], sa_t_2[:, :, state_dim:]
        rewards_1 = reward_model.buffer_reward1[batch_inds]
        rewards_2 = reward_model.buffer_reward2[batch_inds]
        dones_1 = dones_2 = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        attention_mask_1 = torch.ones((batch_size, max_len), dtype=torch.float32, device=device)
        attention_mask_2 = torch.ones((batch_size, max_len), dtype=torch.float32, device=device)
        rtg_1 = rtg_2 = torch.ones((batch_size, max_len, 1), dtype=torch.float32, device=device)
        random_time_start = np.random.randint(0, max_ep_len - max_len - 1, size=(batch_size, 2))
        timesteps_1 = np.arange(max_len).reshape(1, -1) + random_time_start[:, 0].reshape(-1, 1)
        timesteps_2 = np.arange(max_len).reshape(1, -1) + random_time_start[:, 1].reshape(-1, 1)

        # to tensor
        states_1 = torch.from_numpy(states_1).to(dtype=torch.float32, device=device)
        actions_1 = torch.from_numpy(actions_1).to(dtype=torch.float32, device=device)
        rewards_1 = torch.from_numpy(rewards_1).to(dtype=torch.float32, device=device)
        timesteps_1 = torch.from_numpy(timesteps_1).to(dtype=torch.long, device=device)
        states_2 = torch.from_numpy(states_2).to(dtype=torch.float32, device=device)
        actions_2 = torch.from_numpy(actions_2).to(dtype=torch.float32, device=device)
        rewards_2 = torch.from_numpy(rewards_2).to(dtype=torch.float32, device=device)
        timesteps_2 = torch.from_numpy(timesteps_2).to(dtype=torch.long, device=device)
        labels = torch.from_numpy(labels).to(dtype=torch.float32, device=device)
        return states_1, actions_1, rewards_1, dones_1, rtg_1, timesteps_1, attention_mask_1, \
               states_2, actions_2, rewards_2, dones_2, rtg_2, timesteps_2, attention_mask_2, \
               labels


    # define the contrastive model
    model = BidirectionalTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        hidden_size=variant['embed_dim'],
        z_dim=z_dim,
        max_length=K,
        max_ep_len=max_ep_len,
        # transformer parameters
        n_layer=variant['n_layer'],
        n_head=variant['n_head'],
        n_inner=4*variant['embed_dim'],
        activation_function=variant['activation_function'],
        n_positions=1024,
        resid_pdrop=variant['dropout'],
        attn_pdrop=variant['dropout'],
    )
    model = model.to(device=device)
    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )
    z_star = torch.nn.parameter.Parameter(torch.empty(z_dim, requires_grad=True, device=device))
    torch.nn.init.normal_(z_star)
    z_star_optimizer = torch.optim.AdamW(
        [z_star],
        lr=variant["z_star_lr"],
        weight_decay=variant['weight_decay']
    )
    reward_model.set_bdt_model(model)

    trainer = ContrativeTrainer(
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,
        get_batch=get_batch,
        get_pref_batch=get_pref_batch,
        z_star=z_star,
        z_star_optimizer=z_star_optimizer,
        scheduler=scheduler,
        loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
        eval_fns=[],
        eval_bdt_z_stars=[],
        similarity_fn=variant['similarity_fn'], 
        norm_loss_ratio=variant['norm_loss_ratio'],
        comp_loss_ratio=variant['comp_loss_ratio'],
        pref_loss_ratio=variant['pref_loss_ratio'],
        pref_loss_impl=variant['pref_loss_impl'], 
        device=device,
    )

    if use_contrastive:
        name = f'conbdt_{env_name}-{dataset}'
        name += f'_norm_{variant["norm_loss_ratio"]}_comp_{variant["comp_loss_ratio"]}_pref_{variant["pref_loss_ratio"]}'
        name += f'_fb_{max_feedback}_q_{reward_batch}_skip_{teacher_eps_skip}_{feed_type}'
        name += f'_ctx_{K}_seed_{seed}_{time.strftime("%Y%m%d-%H%M%S")}'
    else:
        name = f'reward_{env_name}-{dataset}_fb_{max_feedback}_q_{reward_batch}_skip_{teacher_eps_skip}_{feed_type}'
        name += f'_ctx_{K}_seed_{seed}_{time.strftime("%Y%m%d-%H%M%S")}'
        assert feed_type != 'c'  # shouldn't use contrastive sampling
    wandb.init(project='clarify_reward', name=name, config=variant)

    def learn_reward(first_flag=0):
        # get feedbacks
        if first_flag == 1:
            # if it is first time to get feedback, need to use random sampling
            labeled_queries, metrics = reward_model.uniform_sampling()
        else:
            if feed_type == 0 or feed_type == 'u':
                labeled_queries, metrics = reward_model.uniform_sampling()
            elif feed_type == 1 or feed_type == 'd':
                labeled_queries, metrics = reward_model.disagreement_sampling()
            elif feed_type == 2 or feed_type == 'c':
                labeled_queries, metrics = reward_model.contrastive_sampling()
            elif feed_type == 20 or feed_type == 'cm':
                labeled_queries, metrics = reward_model.max_ratio_sampling()
            else:
                raise NotImplementedError

        nonlocal total_feedback, labeled_feedback
        total_feedback += reward_model.mb_size
        labeled_feedback += labeled_queries
        
        train_acc = 0
        if labeled_feedback > 0:
            # update reward
            for epoch in range(reward_update):
                train_acc, reward_loss = reward_model.train_reward()
                train_acc = np.mean(train_acc)
                if train_acc > 0.97:
                    break
        eval_acc = reward_model.test_reward_model_accuracy()
        print(f"Reward function is updated!! train ACC: {train_acc:.2f}, eval ACC: {eval_acc:.2f}, epoch: {epoch}")

        metrics['reward_train_acc'] = train_acc
        metrics['reward_eval_acc'] = eval_acc
        metrics['reward_loss'] = reward_loss
        return metrics


    # training loop
    itr = 0
    while total_feedback < max_feedback:
        reward_model.change_batch(1.0)
        if reward_model.mb_size + total_feedback > max_feedback:
            reward_model.set_batch(max_feedback - total_feedback)

        metrics = learn_reward(first_flag=1 if itr == 0 else 0)
        wandb.log(metrics, step=total_feedback)
        if use_contrastive:
            outputs = trainer.train_iteration(
                num_steps=variant['num_steps_init'] if itr == 0 else variant['num_steps_per_iter'], iter_num=itr+1, print_logs=True)
            wandb.log(outputs, step=total_feedback)
        itr += 1

        # save reward model, contrastive model
        if total_feedback % 1000 == 0:
            reward_model.save_reward_model(os.path.join(output_dir, f'reward_{total_feedback}.pth'))
            if use_contrastive:
                torch.save(model.state_dict(), os.path.join(output_dir, f'dt_{total_feedback}.pth'))
                # torch.save(z_star, os.path.join(output_dir, f'z_star_{total_feedback}.pth'))


    # test load
    # reward_model.load_reward_model(reward_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--env', type=str, default='hopper')
    parser.add_argument('--env', type=str, default='dial-turn')
    parser.add_argument('--dataset', type=str, default='rnd')
    parser.add_argument('--K', type=int, default=200)  # segment size & context size, original 20
    parser.add_argument('--segment_length', type=int, default=200)  # should be same as K
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    # contrastive model
    parser.add_argument('--no_use_contrastive', action='store_true', default=False)  # if True, only train reward model
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=256)  # original 128
    parser.add_argument('--n_layer', type=int, default=4)  # original 3
    parser.add_argument('--n_head', type=int, default=4)  # original 1
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--z_star_lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_steps_per_iter', type=int, default=500)  # original 2000
    parser.add_argument('--num_steps_init', type=int, default=20000)  # 10 iters
    # parser.add_argument('--num_steps_per_iter', type=int, default=2)  # debug
    # parser.add_argument('--num_steps_init', type=int, default=2)  # 10 iters  # debug
    parser.add_argument('--dist_dim', type=int, default=30)
    parser.add_argument('--n_bins', type=int, default=31)
    parser.add_argument('--gamma', type=float, default=1.00)
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--z_dim', type=int, default=16)  # original 1
    # Pbclarify
    parser.add_argument('--similarity_fn', type=str, default='l2')
    parser.add_argument('--norm_loss_ratio', type=float, default=0.1)
    parser.add_argument('--comp_loss_ratio', type=float, default=0.1)
    parser.add_argument('--pref_loss_ratio', type=float, default=1.0)
    parser.add_argument('--pref_loss_impl', type=str, default='pairwise')
    # for eval
    parser.add_argument('--num_eval_episodes', type=int, default=20)
    # parser.add_argument('--num_eval_episodes', type=int, default=2)  # debug
    parser.add_argument('--save_rollout', type=bool, default=False)
    parser.add_argument('--output_dir', type=str, default='./results')

    # reward learning
    parser.add_argument('--reward_batch', type=int, default=50)
    parser.add_argument('--max_feedback', type=int, default=2000)
    parser.add_argument('--feed_type', type=str, default='d')  # d, u, c (contrastive), cm (max ratio)
    parser.add_argument('--train_num_iter', type=int, default=50)
    parser.add_argument('--reward_lr', type=float, default=3e-4)
    parser.add_argument('--reward_update', type=int, default=50)
    # error teacher
    parser.add_argument('--teacher_eps_skip', type=float, default=0.5)
    parser.add_argument('--use_gt_reward', type=int, default=False)

    args = parser.parse_args()

    # random seed
    set_seed_everywhere(args.seed)

    # log dir
    use_contrastive = not args.no_use_contrastive
    if use_contrastive:
        save_dir = f'reward-{args.env}-{args.dataset}-conbdt'
        save_dir += f'-norm-{args.norm_loss_ratio}-comp-{args.comp_loss_ratio}-pref-{args.pref_loss_ratio}'
        save_dir += f'_fb_{args.max_feedback}_q_{args.reward_batch}_skip_{args.teacher_eps_skip}_{args.feed_type}'
        save_dir += f'-ctx_{args.K}-seed_{args.seed}-{time.strftime("%Y%m%d-%H%M%S")}'
    else:
        save_dir = f'reward-{args.env}-{args.dataset}-nocon'
        save_dir += f'_fb_{args.max_feedback}_q_{args.reward_batch}_skip_{args.teacher_eps_skip}_{args.feed_type}'
        save_dir += f'-ctx_{args.K}-seed_{args.seed}-{time.strftime("%Y%m%d-%H%M%S")}'
    output_dir = os.path.join(args.output_dir, save_dir)
    os.makedirs(output_dir, exist_ok=True)

    eval_dir = os.path.join(output_dir, f'eval')
    os.makedirs(eval_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'params.json'), mode="w") as f:
        json.dump(args.__dict__, f, indent=4)

    experiment(output_dir, eval_dir, variant=vars(args))


