from dataclasses import dataclass
import numpy as np
import gym
import imageio
import torch
import torch.nn as nn
import os
import pickle


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0



class ReplayBuffer(object):
    def __init__(self, 
                 state_dim=None, act_dim=None, max_size=int(1e7), 
                 device='cuda:0'):
        self.device = device

        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.state = torch.zeros((max_size, state_dim), dtype=torch.float32, device=self.device)
        self.action = torch.zeros((max_size, act_dim), dtype=torch.float32, device=self.device)
        self.next_state = torch.zeros((max_size, state_dim), dtype=torch.float32, device=self.device)
        self.reward = torch.zeros((max_size, 1), dtype=torch.float32, device=self.device)
        self.done = torch.zeros((max_size, 1), dtype=torch.float32, device=self.device)


    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self.device)


    def add_single_transition(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def add_batch_transition(self, states, actions, next_states, rewards, dones):
        traj_len = len(next_states)
        if self.ptr + traj_len < self.max_size:
            self.state[self.ptr:self.ptr+traj_len] = self._to_tensor(states)
            self.action[self.ptr:self.ptr+traj_len] = self._to_tensor(actions)
            self.next_state[self.ptr:self.ptr+traj_len] = self._to_tensor(next_states)
            self.reward[self.ptr:self.ptr+traj_len] = self._to_tensor(rewards.reshape(-1, 1))
            self.done[self.ptr:self.ptr+traj_len] = self._to_tensor(dones.reshape(-1, 1))
        else:
            self.state[self.ptr:] = self._to_tensor(states[:self.max_size - self.ptr])
            self.action[self.ptr:] = self._to_tensor(actions[:self.max_size - self.ptr])
            self.next_state[self.ptr:] = self._to_tensor(next_states[:self.max_size - self.ptr])
            self.reward[self.ptr:] = self._to_tensor(rewards[:self.max_size - self.ptr].reshape(-1, 1))
            self.done[self.ptr:] = self._to_tensor(dones[:self.max_size - self.ptr].reshape(-1, 1))

            self.state[:traj_len - (self.max_size - self.ptr)] = self._to_tensor(states[self.max_size - self.ptr:])
            self.action[:traj_len - (self.max_size - self.ptr)] = self._to_tensor(actions[self.max_size - self.ptr:])
            self.next_state[:traj_len - (self.max_size - self.ptr)] = self._to_tensor(next_states[self.max_size - self.ptr:])
            self.reward[:traj_len - (self.max_size - self.ptr)] = self._to_tensor(rewards[self.max_size - self.ptr:].reshape(-1, 1))
            self.done[:traj_len - (self.max_size - self.ptr)] = self._to_tensor(dones[self.max_size - self.ptr:].reshape(-1, 1))

        self.ptr = (self.ptr + traj_len) % self.max_size
        self.size = min(self.size + traj_len, self.max_size)


    def sample(self, batch_size: int = 256):
        ''' state, action, next_state, reward, done '''
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            self.state[ind], self.action[ind], self.next_state[ind],
            self.reward[ind], self.done[ind]
        )


    def normalize_states(self, eps = 1e-3):
        mean = torch.mean(self.state, dim=0, keepdims=True)
        std = torch.std(self.state, dim=0, keepdims=True) + eps
        self.state = (self.state - mean) / std
        self.next_state = (self.next_state - mean) / std
        return mean.detach().cpu().numpy(), std.detach().cpu().numpy()
    

    def normalize_reward(self):
        min_reward = self.reward.min()
        max_reward = self.reward.max()
        self.reward = (self.reward - min_reward) / (max_reward - min_reward)
        return min_reward, max_reward


@torch.no_grad()
def evaluate_agent(env: gym.Env, agent: nn.Module, # state_mean: torch.Tensor, state_std: torch.Tensor,
                   mean=0, std=1, 
                   episode_num: int = 10, save_video=False, save_num=2,
                   video_path=None, exp_name='', device='cuda:3'):

    agent.actor.eval()
    episode_return_list = []
    episode_success_list = []
    for iter_num in range(episode_num):
        #state, _ = env.reset()
        #done = False
        ts = env.reset()

        episode_return, episode_success, timestep = 0, 0, 0
        frames = []
        traj_metadata = {'observations': [], 'next_observations': [], 'actions': [],
            'rewards': [], 'terminals': [], 'timesteps': [], 'success': []}

        '''while not done:
            #state = state[0]
            state = np.array(state).reshape(1, -1)
            if state.shape == (1, 2):
                state = state[0][0]
            # print(f'state: {state}, mean: {mean}, std: {std}')
            # print(f'state: {state.shape}, mean: {mean.shape}, std: {std.shape}')
            state = (np.array(state).reshape(1, -1) - mean) / std
            action = agent.select_action(state)
            next_state, reward, done, info, _ = env.step(action)
            done = done or info
            # print(f'info: {info}')
            episode_return += reward
            if 'success' in info:
                episode_success = max(episode_success, info["success"])
            if save_video and iter_num < save_num:
                frames.append(env.render(mode='rgb_array'))
                traj_metadata['observations'].append(state)
                traj_metadata['next_observations'].append(next_state)
                traj_metadata['actions'].append(action)
                traj_metadata['rewards'].append(reward)
                traj_metadata['terminals'].append(done)
                traj_metadata['timesteps'].append(timestep)
                traj_metadata['success'].append(episode_success)
            state = next_state
            timestep += 1'''

        while not ts.last():
            state = (np.array(ts.observation).reshape(1, -1) - mean) / std
            action = agent.select_action(state)
            ts = env.step(action)
            episode_return += ts.reward
            timestep += 1

        if iter_num % 10 == 0:
            print(f'evaluate, episode return: {episode_return}, success: {episode_success}')
        episode_return_list.append(episode_return)
        episode_success_list.append(episode_success)
        if save_video and iter_num < save_num:
            save_video_path = video_path if video_path else f'./video/{env.spec.id}/'  # need video_path
            if not os.path.exists(save_video_path):
                os.makedirs(save_video_path)
            imageio.mimsave(os.path.join(save_video_path, f'{exp_name}_{iter_num}.mp4'), frames, fps=30)
            pickle.dump(traj_metadata, open(os.path.join(save_video_path, f'{exp_name}_{iter_num}.pkl'), 'wb'))
    
    agent.actor.train()
    return np.mean(episode_return_list), np.std(episode_return_list), np.mean(episode_success_list) * 100 \
        if episode_success_list else 0


