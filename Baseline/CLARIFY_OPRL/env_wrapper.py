import argparse
import os
import pickle
import gym
from gym.wrappers.time_limit import TimeLimit
from gym.spaces import Box
import numpy as np
import random
import torch
import metaworld
import metaworld.envs.mujoco.env_dict as _env_dict
import dmc2gym
import imageio
import pynvml
pynvml.nvmlInit()


D4RL_ENVS = ['hopper', 'halfcheetah', 'walker2d']
METAWORLD_ENVS = ['box-close', 'dial-turn', 'drawer-open', 'hammer', 'handle-pull-side', 'peg-insert-side', 'sweep-into']
DMCONTROL_ENVS = ['cheetah-run', 'hopper-hop', 'humanoid-walk', 'quadruped-walk', 'walker-walk']

metaworld_dataset_quality = {
    'box-close-v2': 9.0,
    'dial-turn-v2': 3.5,
    'drawer-open-v2': 1.0,
    'hammer-v2': 5.0,
    'handle-pull-side-v2': 2.5,
    'peg-insert-side-v2': 5.0,
    'sweep-into-v2': 1.5,
}

dmcontrol_dataset_quality = {
    'walker-walk': 1.0,
    'cheetah-run': 6.0,
}



def set_seed_everywhere(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(False)
    # torch.backends.cudnn.deterministic = False
    # torch.backends.cudnn.benchmark = False


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


def str2bool(value):
    if value.lower() in ['true', '1', 't', 'y', 'yes']:
        return True
    elif value.lower() in ['false', '0', 'f', 'n', 'no']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_free_gpu():
    device_count = pynvml.nvmlDeviceGetCount()  
    
    free_gpu = None
    min_memory_used = float('inf')
    
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        memory_used = memory_info.used / memory_info.total
        
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_util = utilization.gpu 
        

        if memory_used < min_memory_used and gpu_util < 50:
            min_memory_used = memory_used
            free_gpu = i
    
    if free_gpu is not None:
        device = torch.device(f'cuda:{free_gpu}')
    else:
        device = torch.device('cpu')
    
    return device


class ProxyEnv(gym.Env):
    def __init__(self, wrapped_env):
        self._wrapped_env = wrapped_env
        self.action_space = self._wrapped_env.action_space
        self.observation_space = self._wrapped_env.observation_space

    @property
    def wrapped_env(self):
        return self._wrapped_env

    def reset(self, **kwargs):
        # return self._wrapped_env.reset(**kwargs)
        return self._wrapped_env.reset(**kwargs)[0]

    def step(self, action):
        return self._wrapped_env.step(action)

    def render(self, mode):
        self._wrapped_env.render_mode = mode
        self._wrapped_env.camera_name = 'corner3'
        # self._wrapped_env.camera_name = 'gripperPOV'
        # self._wrapped_env.camera_name = 'behindGripper'
        # self._wrapped_env.camera_name = 'topview'
        return self._wrapped_env.render()

    @property
    def horizon(self):
        return self._wrapped_env.horizon

    def terminate(self):
        if hasattr(self.wrapped_env, "terminate"):
            self.wrapped_env.terminate()

    def __getattr__(self, attr):
        if attr == '_wrapped_env':
            raise AttributeError()
        return getattr(self._wrapped_env, attr)

    def __getstate__(self):
        """
        This is useful to override in case the wrapped env has some funky
        __getstate__ that doesn't play well with overriding __getattr__.
        The main problematic case is/was gym's EzPickle serialization scheme.
        :return:
        """
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __str__(self):
        return '{}({})'.format(type(self).__name__, self.wrapped_env)


class NormalizedBoxEnv(ProxyEnv):
    """
    Normalize action to in [-1, 1].
    Optionally normalize observations and scale reward.
    """

    def __init__(
            self,
            env,
            reward_scale=1.,
            obs_mean=None,
            obs_std=None,
    ):
        ProxyEnv.__init__(self, env)
        self._should_normalize = not (obs_mean is None and obs_std is None)
        if self._should_normalize:
            if obs_mean is None:
                obs_mean = np.zeros_like(env.observation_space.low)
            else:
                obs_mean = np.array(obs_mean)
            if obs_std is None:
                obs_std = np.ones_like(env.observation_space.low)
            else:
                obs_std = np.array(obs_std)
        self._reward_scale = reward_scale
        self._obs_mean = obs_mean
        self._obs_std = obs_std
        ub = np.ones(self._wrapped_env.action_space.shape)
        self.action_space = Box(-1 * ub, ub)

    def estimate_obs_stats(self, obs_batch, override_values=False):
        if self._obs_mean is not None and not override_values:
            raise Exception("Observation mean and std already set. To "
                            "override, set override_values to True.")
        self._obs_mean = np.mean(obs_batch, axis=0)
        self._obs_std = np.std(obs_batch, axis=0)

    def _apply_normalize_obs(self, obs):
        return (obs - self._obs_mean) / (self._obs_std + 1e-8)

    def step(self, action):
        lb = self._wrapped_env.action_space.low
        ub = self._wrapped_env.action_space.high
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)

        wrapped_step = self._wrapped_env.step(scaled_action)
        next_obs, reward, terminate, done, info = wrapped_step
        done = done or terminate
        # if done == True:
        #     done = 1.0
        # else:
        #     done = 0.0
        if self._should_normalize:
            next_obs = self._apply_normalize_obs(next_obs)
        return next_obs, reward * self._reward_scale, done, info

    def __str__(self):
        return "Normalized: %s" % self._wrapped_env


def make_metaworld_env(env_name, seed=0):
    if env_name in _env_dict.ALL_V2_ENVIRONMENTS:
        env_cls = _env_dict.ALL_V2_ENVIRONMENTS[env_name]
    else:
        env_cls = _env_dict.ALL_V1_ENVIRONMENTS[env_name]

    env = env_cls(render_mode='rgb_array', camera_name='corner2')
    # print("partially observe", env._partially_observable) Ture
    # print("env._freeze_rand_vec", env._freeze_rand_vec) True
    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True
    env.seed(seed)
    env.action_space.seed(seed)
    return TimeLimit(NormalizedBoxEnv(env), env.max_path_length)


def make_dmc_env(env_name, seed):
    env_name = env_name.replace("dmc_", "")
    domain_name, task_name = env_name.split("-")
    domain_name = domain_name.lower()
    task_name = task_name.lower()
    env = dmc2gym.make(
        domain_name=domain_name,
        task_name=task_name,
        seed=seed,
    )
    return env


def create_env(env_name: str, seed: int=0):
    # d4rl
    if env_name in D4RL_ENVS:
        env = gym.make(env_name.capitalize() + '-v3')
        eval_env = gym.make(env_name.capitalize() + '-v3')
        scale = 1000.
        max_ep_len = 1000
    # metaworld
    elif env_name in METAWORLD_ENVS:
        env = make_metaworld_env(env_name + '-v2', seed)
        eval_env = make_metaworld_env(env_name + '-v2', 2 ** 32 - 1 - seed)
        scale = 1000.
        max_ep_len = 500
    # dmcontrol
    elif env_name in DMCONTROL_ENVS:
        env = make_dmc_env(env_name, seed)
        eval_env = make_dmc_env(env_name, 2 ** 32 - 1 - seed)
        scale = 1.
        max_ep_len = 1000
    else:
        print(f'env_name: {env_name} not in D4RL_ENVS or METAWORLD_ENVS, seed: {seed}')
        raise NotImplementedError
    
    env.seed(seed)
    eval_env.seed(2 ** 32 - 1 - seed)
    return env, eval_env, scale, max_ep_len


def get_metaworld_dataset(env_name, data_quality=1, using_notebook=False):
    base_path = f"./data/metaworld_data/"
    if using_notebook:
        base_path = f"./../data/metaworld_data/"
    trajectories, states, traj_lens, returns, rewards = [], [], [], [], []

    # get dataset
    dataset = dict()
    for seed in range(3):
        path = base_path + f"{env_name}-v2/saved_replay_buffer_1000000_seed{seed}.pkl"
        with open(path, "rb") as f:
            load_dataset = pickle.load(f)

        print(f'[loading...] load from file: {path}')
        print(f'[loading...] keys: {load_dataset.keys()}')
        print(f'[loading...] length: {len(load_dataset["dones"])}, length we take: {int(data_quality * 100_000)}')
        for key in load_dataset.keys():
            load_dataset[key] = load_dataset[key][: int(data_quality * 100_000)]
        load_dataset["terminals"] = load_dataset["dones"][: int(data_quality * 100_000)]
        load_dataset.pop("dones", None)

        for key in load_dataset.keys():
            if key not in dataset:
                dataset[key] = load_dataset[key]
            else:
                dataset[key] = np.concatenate((dataset[key], load_dataset[key]), axis=0)

    N = dataset["rewards"].shape[0]  # number of transitions
    trajectory = {
        'observations': [], 
        'next_observations': [],
        'actions': [], 
        'rewards': [], 
        'terminals': [],
    }

    dataset["rewards"] = dataset["rewards"].reshape(-1)
    dataset["terminals"] = dataset["terminals"].reshape(-1)

    env = make_metaworld_env(env_name + '-v2', 0)
    for i in range(N):
        trajectory['observations'].append(dataset["observations"][i].astype(np.float32))
        trajectory['next_observations'].append(dataset["next_observations"][i].astype(np.float32))
        trajectory['actions'].append(dataset["actions"][i].astype(np.float32))
        trajectory['rewards'].append(dataset["rewards"][i].astype(np.float32))
        trajectory['terminals'].append(True if bool(dataset["terminals"][i]) else False)

        if bool(dataset["terminals"][i]) or len(trajectory['rewards']) == \
                env.max_path_length or i == N - 1:
            if len(trajectory['observations']) == 0:
                continue
            traj_lens.append(len(trajectory['observations']))
            rewards.extend(trajectory['rewards'])
            for key in trajectory.keys():  # to np array
                trajectory[key] = np.array(trajectory[key])
            trajectories.append(trajectory)
            states.append(trajectory['observations'])
            returns.append(trajectory['rewards'].sum())
            trajectory = {
                'observations': [], 
                'next_observations': [],
                'actions': [], 
                'rewards': [], 
                'terminals': [],
            }
    
    traj_lens, returns = np.array(traj_lens), np.array(returns)
    return trajectories, states, traj_lens, returns, rewards


def get_d4rl_dataset(env_name, dataset):
    dataset_path = f'data/d4rl_data/{env_name}-{dataset}-v2.pkl'
    trajectories, states, traj_lens, returns, rewards = [], [], [], [], []

    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)

    for path in trajectories:
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
        rewards.extend(path['rewards'])
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    return trajectories, states, traj_lens, returns, rewards


def get_dmc_dataset(env_name, data_quality=1, using_notebook=False):
    base_path = f"../metamotivo/datasets/"
    if using_notebook:
        base_path = f"./../data/dmcontrol_data/"
    trajectories, states, traj_lens, returns, rewards = [], [], [], [], []

    dataset = dict()
    for seed in range(3):
        path = base_path + f"{env_name}/saved_replay_buffer_1000000_seed{seed}.pkl"
        with open(path, "rb") as f:
            load_dataset = pickle.load(f)

        if "humanoid" in env_name:
            for key in load_dataset.keys():
                load_dataset[key] = load_dataset[key][
                    200000 : int(data_quality * 100_000)
                ]
            load_dataset["terminals"] = load_dataset["dones"][
                0 : int(data_quality * 100_000) - 200000
            ]
            load_dataset.pop("dones", None)
        else:
            for key in load_dataset.keys():
                load_dataset[key] = load_dataset[key][
                    0 : int(data_quality * 100_000)
                ]
            load_dataset["terminals"] = load_dataset["dones"][
                0 : int(data_quality * 100_000) - 0
            ]
            load_dataset.pop("dones", None)

        for key in load_dataset.keys():
            if key not in dataset:
                dataset[key] = load_dataset[key]
            else:
                dataset[key] = np.concatenate((dataset[key], load_dataset[key]), axis=0)
        # print("shape", load_dataset["rewards"].shape, "from seed ", seed, end=",  ")
    
    N = dataset["rewards"].shape[0]  # number of transitions
    trajectory = {
        'observations': [], 
        'next_observations': [],
        'actions': [], 
        'rewards': [], 
        'terminals': [],
    }

    dataset["rewards"] = dataset["rewards"].reshape(-1)
    dataset["terminals"] = dataset["terminals"].reshape(-1)

    for i in range(N):
        trajectory['observations'].append(dataset["observations"][i].astype(np.float32))
        trajectory['next_observations'].append(dataset["next_observations"][i].astype(np.float32))
        trajectory['actions'].append(dataset["actions"][i].astype(np.float32))
        trajectory['rewards'].append(dataset["rewards"][i].astype(np.float32))
        trajectory['terminals'].append(True if bool(dataset["terminals"][i]) else False)

        if bool(dataset["terminals"][i]) or i == N - 1:
            if len(trajectory['observations']) == 0:
                continue
            traj_lens.append(len(trajectory['observations']))
            rewards.extend(trajectory['rewards'])
            for key in trajectory.keys():  # to np array
                trajectory[key] = np.array(trajectory[key])
            trajectories.append(trajectory)
            states.append(trajectory['observations'])
            returns.append(trajectory['rewards'].sum())
            trajectory = {
                'observations': [], 
                'next_observations': [],
                'actions': [], 
                'rewards': [], 
                'terminals': [],
            }
    
    traj_lens, returns = np.array(traj_lens), np.array(returns)
    return trajectories, states, traj_lens, returns, rewards


def create_dataset(env_name, dataset, using_notebook=False):
    '''
    - trajectories: list of dict, each dict contains keys: 
        - observations, next_observations, actions, rewards, terminals
    - states: list of np.ndarray, each np.ndarray is a trajectory of states
    - traj_lens: list of int, each int is the length of a trajectory
    - returns: np.ndarray, each element is the return of a trajectory
    - rewards: list of float, each float is a reward (of a trasition) in the dataset
    '''
    print(f'current working directory: {os.getcwd()}')
    # if env_name in D4RL_ENVS:
    #     return get_d4rl_dataset(env_name, dataset)
    if env_name in METAWORLD_ENVS:
        return get_metaworld_dataset(env_name, 
            data_quality=metaworld_dataset_quality[env_name + '-v2'], 
            using_notebook=using_notebook)
    elif env_name in DMCONTROL_ENVS:
        return get_dmc_dataset(env_name, data_quality=dmcontrol_dataset_quality[env_name], 
                               using_notebook=using_notebook)
    else:
        print(f'env_name: {env_name} not in D4RL_ENVS or METAWORLD_ENVS, dataset: {dataset}')
        raise NotImplementedError


if __name__ == '__main__':
    # for env_name in ["box-close", "dial-turn", "drawer-open", "hammer", "handle-pull-side", "peg-insert-side", "sweep-into"]:
    for env_name in ["cheetah-run", "walker-walk"]:
        print(env_name)
        env, eval_env, scale, max_ep_len = create_env(env_name)
        env.reset()
        # frame = env.render('rgb_array', shape=(480, 480))
        frame = env.physics.render(height=480, width=480, camera_id=0)
        print(frame.shape)  # (480, 480, 3)
        # frame = np.rot90(frame, 2)
        imageio.imwrite(f"./{env_name}.png", frame)
    # print(env, eval_env, scale, max_ep_len)
    # print(env.observation_space, env.action_space)

    # print(env.reset())
    # print(env.reset())
    # print(env.reset())
