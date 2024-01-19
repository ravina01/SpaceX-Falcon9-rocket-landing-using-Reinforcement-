from ddpg_torch import Agent
import gym
import numpy as np
from utils import plotLearning
from gym.envs.registration import register
import torch
import time

# Set up the environment
ENV_ID = "RocketLander-v0"  # Replace with your environment name

if ENV_ID not in gym.envs.registry.env_specs:
    register(
        id=ENV_ID,
        entry_point='env:RocketLander',  # Replace with the correct path to your module
    )

env = gym.make(ENV_ID)
input_dims = env.observation_space.shape
n_actions = env.action_space.shape[0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device = ", device)
agent = Agent(alpha=0.000025, beta=0.00025, input_dims=input_dims, tau=0.0005, env=env,
              batch_size=32, layer1_size=256, layer2_size=256, n_actions=n_actions, device=device)

# agent.load_models()
np.random.seed(0)

# Training parameters
num_episodes = 5000  # Adjust as needed
max_steps_per_episode = 5000  # Adjust as needed
eval_episodes = 100  # Number of episodes for evaluation
eval_interval = 100  # Evaluate the agent every 'eval_interval' episodes
render_interval = 100  # Render the environment every 'render_interval' episodes
save_threshold = 1.8  # Save the model when average reward goes below this threshold

# Additional counter for rendering
render_counter = 0

# Training loop
for episode in range(num_episodes):
    observation = env.reset()
    episode_reward = 0

    # Measure the time at the beginning of the episode
    start_time = time.time()

    for step in range(max_steps_per_episode):
        # Check if rendering is needed
        if (episode + 1) % render_interval == 0 and render_counter == 0:
            env.render()

        # Choose action from the DDPG agent
        action = agent.choose_action(observation)
        
        # Take the chosen action
        new_observation, reward, done, _ = env.step(action)

        # Store the transition in the replay buffer
        agent.remember(observation, action, reward, new_observation, int(done))
        
        # Learn from the experiences in the replay buffer
        agent.learn()

        # Update the observation for the next step
        observation = new_observation
        episode_reward += reward

        if done:
            # Calculate the FPS at the end of the episode
            end_time = time.time()
            fps = int(step / (end_time - start_time))

            # Print episode statistics
            print(f"Episode: {episode + 1}, Steps: {step + 1}, Reward: {episode_reward}, FPS: {fps}")

            # Increment render counter
            render_counter += 1

            # Check the agent's performance every 'eval_interval' episodes
            if (episode + 1) % eval_interval == 0:
                total_eval_reward = 0

                for eval_episode in range(eval_episodes):
                    eval_observation = env.reset()
                    eval_done = False

                    while not eval_done:
                        eval_action = agent.choose_action(eval_observation)
                        eval_observation, eval_reward, eval_done, _ = env.step(eval_action)
                        total_eval_reward += eval_reward

                # Move this line outside the evaluation episode loop
                avg_eval_reward = total_eval_reward / eval_episodes
                print(f"Avg Eval Reward over {eval_episodes} episodes: {avg_eval_reward}")

                # Save the model if the average reward goes below the threshold
                if abs(avg_eval_reward) > save_threshold:
                    print(f"Saving model. Average Eval Reward below {save_threshold}.")
                    agent.save_models()

            break

    # Check if it's time to render and reset the counter
    if render_counter > 0:
        render_counter -= 1

# Save the trained models at the end of training
agent.save_models()

# Close the environment
env.close()

# score_history = []
# for i in range(1000):
#     obs = env.reset()
#     done = False
#     score = 0
#     while not done:
#         act = agent.choose_action(obs)
#         new_state, reward, done, info = env.step(act)
#         agent.remember(obs, act, reward, new_state, int(done))
#         agent.learn()
#         score += reward
#         obs = new_state
#         env.render()
#     score_history.append(score)

#     # if i % 25 == 0:
#     #    agent.save_models()

#     print('episode ', i, 'score %.2f' % score,
#           'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))

# filename = 'RocketLander-alpha000025-beta00025-400-300.png'
# plotLearning(score_history, filename, window=100)