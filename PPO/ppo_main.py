# from env_og import RocketLander  # Assuming RocketLander is your environment class
from ppo_torch import Agent  # Assuming Agent is your PPO agent class
import gym
import numpy as np
from gym.envs.registration import register
import time  # Import the time module

# Set up the environment
ENV_ID = "RocketLander-v0"  # Replace with your environment name

if ENV_ID not in gym.envs.registry.env_specs:
    register(
        id=ENV_ID,
        entry_point='env_og:RocketLander',  # Replace with the correct path to your module
    )

env = gym.make(ENV_ID)

# Define hyperparameters
# env = RocketLander()  # Initialize your environment
input_dims = env.observation_space.shape
# n_actions = env.action_space.n  # Modify according to your action space
state_dim = env.observation_space.shape
num_actions = env.action_space.shape[0]
agent = Agent(n_actions=num_actions, input_dims=input_dims)  # Initialize PPO agent

n_episodes = 1000  # Number of episodes for training
update_every = 5  # Update frequency

for episode in range(n_episodes):
    observation = env.reset()
    print("observation = ", observation)
    done = False
    score = 0

    while not done:
        action, prob, val = agent.choose_action(observation)
        # print("action = ", action)
        
        observation_, reward, done, _ = env.step(action)
        agent.remember(observation, action, prob, val, reward, done)
        score += reward
        observation = observation_

    agent.learn()

    if episode % update_every == 0:
        agent.save_models()

env.close()  # Close the environment when done
