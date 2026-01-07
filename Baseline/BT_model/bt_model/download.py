import d4rl
import gym

env_names = [
    "halfcheetah-medium-v2", "halfcheetah-medium-replay-v2", "halfcheetah-medium-expert-v2",
    "hopper-medium-v2", "hopper-medium-replay-v2", "hopper-medium-expert-v2",
    "walker2d-medium-v2", "walker2d-medium-replay-v2", "walker2d-medium-expert-v2"
]

for name in env_names:
    env = gym.make(name)
    dataset = d4rl.qlearning_dataset(env)
    print(f"Downloaded: {name}")
