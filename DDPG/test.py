import gym
import numpy as np
import matplotlib.pyplot as plt
from ddpg_torch import Agent  # Assuming your DDPG agent class 
from gym.envs.registration import register
import torch

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
input_dims = env.observation_space.shape
n_actions = env.action_space.shape[0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up the SAC agent
agent = Agent(alpha=0.000025, beta=0.00025, input_dims=input_dims, tau=0.0005, env=env,
              batch_size=32, layer1_size=256, layer2_size=256, n_actions=n_actions, device=device)

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
window_size = 10
# Calculate rolling mean of episode rewards
rolling_mean_rewards = np.convolve(episode_rewards, np.ones(window_size), 'valid') / window_size

# Visualize the last 50 episodes
# num_last_episodes = 500
# for episode in range(num_episodes):
#     observation = env.reset()
#     total_reward = 0

#     while True:
#         action = agent.choose_action(observation)
#         new_observation, reward, done, _ = env.step(action)
#         total_reward += reward
#         # env.render()

#         if done:
#             break

# print("Rendering completed. Closing the environment.")
# env.close()

# Plot the rewards for all episodes
plt.plot(episode_rewards)
plt.title('Total Rewards for All Episodes')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()

# Plotting all episodes versus reward with rolling mean
plt.plot(range(1, len(episode_rewards) + 1), episode_rewards, label='Episode Reward')
plt.plot(range(window_size, len(episode_rewards) + 1), rolling_mean_rewards, label=f'Rolling Mean (Window={window_size})')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('All Episodes vs. Reward with Rolling Mean')
plt.legend()
plt.show()