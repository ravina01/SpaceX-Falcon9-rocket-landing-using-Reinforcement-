

import gym
from sac3 import SAC

from gym.envs.registration import register
import time  # Import the time module

# Set up the environment
ENV_ID = "RocketLander-v0"  # Replace with your environment name

if ENV_ID not in gym.envs.registry.env_specs:
    register(
        id=ENV_ID,
        entry_point='env2:RocketLander',  # Replace with the correct path to your module
    )

env = gym.make(ENV_ID)
state_dim = env.observation_space.shape[0]

num_actions = env.action_space.shape[0]

sac_agent = SAC(state_dim, num_actions)

# Create gym environment


# Run SAC on the environment
sac_agent.run(env, max_steps=1200000, batch_size=20, checkpoint_interval= 100000)
