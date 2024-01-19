import gym
import numpy as np
import matplotlib.pyplot as plt
from sac2 import Agent  # Assuming your SAC agent class is in a file named sac2.py
from gym.envs.registration import register

# Set up the environment
ENV_ID = 'RocketLander-v0'

if ENV_ID not in gym.envs.registry.env_specs:
    register(
        id=ENV_ID,
        entry_point='env:RocketLander',  # Replace with the correct path to your module
    )

env = gym.make(ENV_ID)
state_dim = env.observation_space.shape
num_actions = env.action_space.shape[0]

# Set up the SAC agent
agent = Agent(input_dims=state_dim, env=env, n_actions=num_actions, alpha=1e-4, beta=1e-4)

# Load the saved models
agent.load_models()

# Run the agent in the environment for a large number of episodes
num_episodes = 5000  # or any large number
episode_rewards = []

for episode in range(num_episodes):
    observation = env.reset()
    total_reward = 0

    while True:
        action = agent.choose_action(observation)
        new_observation, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            break

    episode_rewards.append(total_reward)
    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

env.close()

# Visualize the last 50 episodes
num_last_episodes = 500
for episode in range(num_episodes - num_last_episodes, num_episodes):
    observation = env.reset()
    total_reward = 0

    while True:
        action = agent.choose_action(observation)
        new_observation, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()

        if done:
            break

print("Rendering completed. Closing the environment.")
env.close()

# Plot the rewards for all episodes
plt.plot(episode_rewards)
plt.title('Total Rewards for All Episodes')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()
