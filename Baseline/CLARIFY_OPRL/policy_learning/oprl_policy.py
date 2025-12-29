import argparse
import json
import csv
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import gym
import d4rl
import wandb
import time
import re
from dm_control import suite
import mujoco
from dmc_tasks import dmc

import sys
sys.path.append('./')
sys.path.append('./..')
sys.path.append('./../..')

from td3bc import TD3_BC
from iql_LiRE import ImplicitQLearning
from policy_learning.policy_utils import ReplayBuffer, evaluate_agent
from env_wrapper import create_env, create_dataset, set_seed_everywhere, D4RL_ENVS, METAWORLD_ENVS
from reward_model import RewardModel
from pathlib import Path
from tqdm import tqdm

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
torch.set_num_threads(8)

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

def experiment(output_dir, variant):
    gpu = variant.get('gpu', 0)
    device = torch.device(
        f"cuda:{gpu}" if (torch.cuda.is_available() and gpu >= 0) else "cpu"
    )
    env_name, dataset = variant['env'], variant['dataset']
    env_type = None
    if env_name in D4RL_ENVS:
        env_type = 'd4rl'
    elif env_name in METAWORLD_ENVS:
        env_type = 'metaworld'
    seed = variant['seed']
    use_contrastive = not variant['no_use_contrastive']

    # suppose env_name is like 'cheetah-run'
    domain_name, task_name = env_name.split('-')[0], '-'.join(env_name.split('-')[1:])
    print(f'Loading {domain_name}-{task_name} dataset...')
    '''env = suite.load(
        domain_name=domain_name,
        task_name=task_name,
        environment_kwargs={"flat_observation": True},
    )
    eval_env = suite.load(
        domain_name=domain_name,
        task_name=task_name,
        environment_kwargs={"flat_observation": True},
    )'''
    env = dmc.make(f"{domain_name}_{task_name}")
    eval_env = dmc.make(f"{domain_name}_{task_name}")
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


    # reward learning config
    max_feedback = variant['max_feedback']
    segment_length = variant['segment_length']
    reward_lr = variant['reward_lr']
    reward_batch = variant['reward_batch']
    feed_type = variant['feed_type']
    teacher_eps_skip = variant['teacher_eps_skip']
    use_gt_reward = variant['use_gt_reward']
    reward_model = RewardModel(ds=state_dim, da=act_dim, device=device, mb_size=reward_batch,
                               size_segment=segment_length, trajectories=trajectories,
                               teacher_eps_skip=teacher_eps_skip, lr=reward_lr)
    # reward model
    if not use_gt_reward:
        reward_model_name = variant['reward_model_name']
        path = os.path.join('./results', reward_model_name)
        reward_model_path = os.path.join(path, f'reward_{max_feedback}.pth')
        reward_model.load_reward_model(reward_model_path)

    # replay buffer
    replay_buffer = ReplayBuffer(state_dim=state_dim, act_dim=act_dim, 
        max_size=np.sum(traj_lens)+10, device=device)
    for traj in trajectories:
        if len(traj['observations']) == 0:
            continue
        if use_gt_reward:
            preference_rewards = traj['rewards']
        else:
            preference_rewards = reward_model.generate_traj_reward(traj['observations'], traj['actions'])
        # add batch transition
        replay_buffer.add_batch_transition(
            states=traj['observations'], actions=traj['actions'], 
            next_states=traj['next_observations'], 
            rewards=preference_rewards,
            dones=traj['terminals'], 
        )
    # normalize offline data
    mean, std = 0, 1
    if variant['normalize']:
        mean, std = replay_buffer.normalize_states()
    if variant['normalize_reward']:
        replay_buffer.normalize_reward()
    print(f'[offline dataset] mean: {mean.shape}, std: {std.shape}')
    del reward_model


    # policy learning config
    offline_agent = variant['offline_agent']
    offline_rl_update_num = variant['offline_rl_update_num']
    offline_rl_batch_size = variant['offline_rl_batch_size']
    agent_lr = variant['agent_lr']
    policy_eval_freq = variant['policy_eval_freq']
    save_video = variant['save_video']

    # offline RL agent
    if offline_agent == 'iql':
        agent = ImplicitQLearning(
            env, env_name, device=device, 
            vf_lr=agent_lr, qf_lr=agent_lr, actor_lr=agent_lr,
            iql_tau=variant['iql_tau'], beta=variant['beta'], max_steps=offline_rl_update_num, 
        )
    elif offline_agent == 'td3bc':
        agent = TD3_BC(env, device=device, lr=agent_lr, alpha=variant['alpha'])
    print(f"{offline_agent} agent created")


    name = f'{offline_agent}-{env_name}-{dataset}'
    if use_gt_reward:
        name += f'-gt-seed_{args.seed}-{time.strftime("%Y%m%d-%H%M%S")}'
    else:
        if use_contrastive:
            name += f'-conbdt-norm-{args.norm_loss_ratio}-comp-{args.comp_loss_ratio}-pref-{args.pref_loss_ratio}'
        name += f'_fb_{args.max_feedback}_q_{args.reward_batch}_skip_{args.teacher_eps_skip}_{args.feed_type}'
        name += f'-ctx_{args.K}-seed_{args.seed}-{time.strftime("%Y%m%d-%H%M%S")}'
    wandb.init(project='CLARIFY_policy', name=name, config=variant)


    # train offline RL agent
    for offline_update_idx in tqdm(range(offline_rl_update_num)):
        train_data = replay_buffer.sample(batch_size=offline_rl_batch_size)
        log_dict = agent.train_model(train_data)

        # evaluate
        if offline_update_idx % 5000 == 0:
            wandb.log(log_dict, step=offline_update_idx)
        if (offline_update_idx + 1) % policy_eval_freq == 0:
            avg_return, std_return, eval_success = evaluate_agent(eval_env, agent, 
                episode_num=variant['num_eval_episodes'], mean=mean, std=std, 
                save_video=save_video, save_num=2, 
                video_path=os.path.join(output_dir, 'video'), 
                exp_name=str(offline_update_idx), 
            )
            normalized_score = env.get_normalized_score(avg_return) * 100 if env_type == 'd4rl' else avg_return
            print(f't: {offline_update_idx}, return: {avg_return}, std: {std_return}, normalized_score: {normalized_score}, success: {eval_success}')
            wandb.log({
                'avg_return': avg_return, 
                'std_return': std_return,
                'normalized_score': normalized_score,
                'eval_success': eval_success,
                'training_steps': offline_update_idx,
            }, step=offline_update_idx)
    agent.save_policy(os.path.join(output_dir, 'final.pth'))
        
    print('=' * 80)
    print(f'exp_name: {name}')
    print(f'avg_return: {avg_return}\n')
    print(f'std_return: {std_return}\n')
    print('=' * 80)

    os.makedirs("logs", exist_ok=True)
    csv_path = "logs/eval_results.csv"

    # 如果檔案不存在就寫標題
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["exp_name", "avg_return", "std_return", "seed"])

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, avg_return, std_return, seed])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--env', type=str, default='hopper')
    parser.add_argument('--env', type=str, default='dial-turn')
    parser.add_argument('--dataset', type=str, default='rnd')
    parser.add_argument('--K', type=int, default=200)  # segment size & context size, original 20
    parser.add_argument('--segment_length', type=int, default=200)  # should be same as K
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--comment', type=str, default='')

    # reward learning
    parser.add_argument('--reward_batch', type=int, default=50)
    parser.add_argument('--max_feedback', type=int, default=2000)
    parser.add_argument('--feed_type', type=str, default='d')  # dummy
    parser.add_argument('--reward_lr', type=float, default=1e-4)
    parser.add_argument('--reward_model_name', type=str, default='reward-drawer-open-medium-expert-nocon_fb_500_q_50_skip_0.0_d-ctx_50-seed_0-20241218-200421')
    parser.add_argument('--reward_model_name_mapping', type=str, default='undefined')
    # error teacher
    parser.add_argument('--teacher_eps_skip', type=float, default=0.3)
    parser.add_argument('--use_gt_reward', action='store_true', default=False)

    # policy learning
    parser.add_argument('--offline_agent', type=str, default='iql')
    parser.add_argument('--normalize', type=bool, default=True)
    parser.add_argument('--normalize_reward', type=bool, default=True)
    parser.add_argument('--offline_rl_update_num', type=int, default=int(2e5 + 1e4))
    parser.add_argument('--offline_rl_batch_size', type=int, default=256)
    parser.add_argument('--agent_lr', type=float, default=3e-4)
    parser.add_argument('--policy_eval_freq', type=int, default=5000)
    parser.add_argument('--num_eval_episodes', type=int, default=20)
    parser.add_argument('--save_video', action='store_true', default=False)
    # iql
    parser.add_argument('--iql_tau', type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=3.0)
    # td3+bc
    parser.add_argument('--alpha', type=float, default=2.5)

    args = parser.parse_args()

    # random seed
    set_seed_everywhere(args.seed)

    # extract reward model name from reward_model_name_mapping
    if (not args.use_gt_reward) and args.reward_model_name_mapping != 'undefined':
        rmapping_json = json.load(open(args.reward_model_name_mapping, 'r'))
        # KEY="${ENV}_${ERROR}_${FEED_TYPE}_${SEED}"
        rmapping_key = f"{args.env}_{args.teacher_eps_skip}_{args.feed_type}_{args.seed}"
        args.reward_model_name = rmapping_json[rmapping_key]

    # extract reward args from reward_model_name
    if (not args.use_gt_reward) and args.reward_model_name != 'undefined':
        reward_model_name = args.reward_model_name
        if 'norm' in reward_model_name and 'comp' in reward_model_name:
            args.no_use_contrastive = False
            args.norm_loss_ratio = float(re.findall(r'norm-(\d+\.\d+|\d+)', reward_model_name)[0])
            args.comp_loss_ratio = float(re.findall(r'comp-(\d+\.\d+|\d+)', reward_model_name)[0])
            args.pref_loss_ratio = float(re.findall(r'pref-(\d+\.\d+|\d+)', reward_model_name)[0])
        else:
            args.no_use_contrastive = True
        if not 'human' in args.comment:
            args.feed_type = re.findall(r'skip_(\d+\.\d+|\d+)_([^_]+)-ctx', reward_model_name)[0][1]
            args.teacher_eps_skip = float(re.findall(r'_skip_(\d+\.\d+|\d+)_', reward_model_name)[0])
        else:
            args.feed_type = re.findall(r'human_([^_]+)-ctx', reward_model_name)[0]
            args.teacher_eps_skip = 0.5
        args.reward_batch = int(re.findall(r'_q_(\d+)_', reward_model_name)[0])
        args.max_feedback = int(re.findall(r'_fb_(\d+)_', reward_model_name)[0])

    # log dir
    if args.use_gt_reward:
        save_dir = f'{args.offline_agent}-{args.env}-{args.dataset}-gt-seed_{args.seed}-{time.strftime("%Y%m%d-%H%M%S")}'
        args.no_use_contrastive = True
    else:
        save_dir = f'{args.offline_agent}-{args.env}-{args.dataset}'
        if not args.no_use_contrastive:
            save_dir += f'-conbdt-norm-{args.norm_loss_ratio}-comp-{args.comp_loss_ratio}-pref-{args.pref_loss_ratio}'
        if not 'human' in args.comment:
            save_dir += f'_fb_{args.max_feedback}_q_{args.reward_batch}_skip_{args.teacher_eps_skip}_{args.feed_type}'
        else:
            save_dir += f'_fb_{args.max_feedback}_q_{args.reward_batch}_human_{args.feed_type}'
        save_dir += f'-ctx_{args.K}-seed_{args.seed}-{time.strftime("%Y%m%d-%H%M%S")}'
    output_dir = os.path.join('./results', save_dir)
    os.makedirs(output_dir, exist_ok=True)

    eval_dir = os.path.join(output_dir, f'eval')
    os.makedirs(eval_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'params.json'), mode="w") as f:
        json.dump(args.__dict__, f, indent=4)

    experiment(output_dir, variant=vars(args))

