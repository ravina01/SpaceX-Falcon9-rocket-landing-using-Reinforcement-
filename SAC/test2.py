import gym
import torch
from sac import SAC 
from gym.envs.registration import register
import time  # Import the time module
import matplotlib.pyplot as plt
import numpy as np
# def visualize_agent(agent, env, max_episodes=5000):
#     total_reward = 0.0
#     episode_rewards = []  # List to store the total rewards for each episode
#     episodes = []  # List to store the episode numbers
#     max_steps = 1000000
#     for episode in range(max_episodes):
#         state = env.reset()

#         for step in range(max_steps):
#             action = agent.actor(torch.tensor(state).float().to("cpu")).cpu().detach().numpy()

#             next_state, reward, done, _ = env.step(action)
#             total_reward += reward
#             state = next_state

#             if done:
#                 episode_rewards.append(total_reward)
#                 episodes.append(episode)
#                 print(f"Episode: {episode}, Total Reward: {total_reward:.2f}")
#                 total_reward = 0.0
#                 break

#     env.close()

#     # Plot the rewards vs episodes
#     plt.plot(episodes, episode_rewards)
#     plt.xlabel('Episodes')
#     plt.ylabel('Total Reward')
#     plt.title('Reward vs Episodes')
#     plt.show()

# def print_model_weights(model):
#     for name, param in model.named_parameters():
#         if param.requires_grad:
#             print(f"Layer: {name}, Size: {param.size()}, Values: {param.data}")


# def test_agent(agent, env, num_episodes=10):
#     for episode in range(num_episodes):
#         state = env.reset()
#         total_reward = 0.0

#         while True:
#             action = agent.actor(torch.tensor(state).float().to("cpu")).cpu().detach().numpy()
#             next_state, reward, done, _ = env.step(action)
#             total_reward += reward
#             state = next_state

#             env.render()

#             if done:
#                 print(f"Episode {episode + 1}, Total Reward: {total_reward:.2f}")
#                 break

#     env.close()


# if __name__ == "__main__":
#     # Instantiate the SAC agent and load the trained models


#     # Set up the environment
#     ENV_ID = "RocketLander-v0"  # Replace with your environment name

#     if ENV_ID not in gym.envs.registry.env_specs:
#         register(
#             id=ENV_ID,
#             entry_point='env2:RocketLander',  # Replace with the correct path to your module
#         )

#     env = gym.make(ENV_ID)
#     state_dim = env.observation_space.shape[0]

#     num_actions = env.action_space.shape[0]
#     sac_agent = SAC(state_dim, num_actions)
#     sac_agent.load_models('sac/actor.pth', 'sac/critic1.pth', 'sac/critic2.pth')



#     # Visualize the SAC agent in the environment
#     # visualize_agent(sac_agent, env)
#     test_agent(sac_agent, env, num_episodes=100)
#     # print_model_weights(sac_agent.actor)


def test_model(agent, env, num_episodes=10):
    total_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0

        while True:
            action = agent([state])[0]
            next_state, reward, done, _ = env.step(action)
            env.render()
            episode_reward += reward
            state = next_state

            if done:
                total_rewards.append(episode_reward)
                print(f"Episode {episode + 1}, Total Reward: {episode_reward:.2f}")
                break

    print(f"Mean Reward over {num_episodes} episodes: {np.mean(total_rewards):.2f}")

# Assuming you have already trained the SAC model and have an instance of the SAC class called 'sac_agent'
# Also, you need to have an instance of the environment, for example, 'test_env'


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
sac_agent.load_models('sac/actor_checkpoint_9.pth', 'sac/critic1_checkpoint_9.pth', 'sac/critic2_checkpoint_9.pth')
# Test the trained model on different episodes
test_model(sac_agent.agent, env, num_episodes=10)
