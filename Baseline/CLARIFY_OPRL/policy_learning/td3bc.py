import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym


class Actor(nn.Module):
	def __init__(self, state_dim, act_dim, action_range):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, act_dim)
		
		self.action_range = action_range
		self.max_action = float((action_range[1] - action_range[0])/2)
		self.action_base = (action_range[1] + action_range[0])/2

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.action_base + self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, act_dim):
		super(Critic, self).__init__()
		# Q1 architecture
		self.l1 = nn.Linear(state_dim + act_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)
		# Q2 architecture
		self.l4 = nn.Linear(state_dim + act_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)

	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2

	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class TD3_BC():
	def __init__(
		self,
		env: gym.Env,
		lr=3e-4,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		alpha=2.5,
		device='cuda:0'
	):
		self.device = device
		
		self.env = env
		state_dim = env.observation_space.shape[0]
		act_dim = env.action_space.shape[0] # env.action_space: Box(-1.0, 1.0, (6,), float32)
		action_range = (env.action_space.low[0], env.action_space.high[0])
		print(f'env_name: {env.spec.id}, state_dim: {state_dim}, act_dim: {act_dim}, action_range: {action_range}')

		# Actor
		self.actor = Actor(state_dim, act_dim, action_range).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
		# Critic
		self.critic = Critic(state_dim, act_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

		self.action_range = action_range
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise * (action_range[1] - action_range[0])
		self.noise_clip = noise_clip * (action_range[1] - action_range[0])
		self.policy_freq = policy_freq
		self.alpha = alpha

		self.total_it = 0
		self.saved_actor_loss = 0


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train_model(self, train_data):
		'''
		train_data: state, action, next_state, reward, done
		'''
		self.total_it += 1
		state, action, next_state, reward, done = train_data
		log_dict = {}

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(next_state) + noise
			).clamp(self.action_range[0], self.action_range[1])

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward.unsqueeze(1) + (1. - done.float()) * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()
		log_dict['critic_loss'] = critic_loss.item()
		log_dict['Q_mean'] = current_Q1.mean().item()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor loss
			pi = self.actor(state)
			Q = self.critic.Q1(state, pi)
			lmbda = self.alpha / Q.abs().mean().detach()

			# 1. Maximize Q;   2. Penalize OOD action
			actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, action) 
			self.saved_actor_loss = actor_loss.item()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()
			log_dict['actor_loss'] = actor_loss.item()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		return log_dict


	def save_policy(self, filename: str = './td3_bc_policy.pth'):
		torch.save(self.actor.state_dict(), filename)


	def load_policy(self, filename: str = './td3_bc_policy.pth'):
		self.actor.load_state_dict(torch.load(filename))

