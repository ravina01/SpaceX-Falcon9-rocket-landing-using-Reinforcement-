import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from gym.envs.registration import register
from torch.distributions import MultivariateNormal

# Define the actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        return mean, log_std

# Define the critic network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.value(x)
        return value

# Define the Soft Actor-Critic agent
class SACAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.replay_buffer = []

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        mean, log_std = self.actor(state)
        normal = MultivariateNormal(mean, torch.diag_embed(torch.exp(log_std)))
        action = normal.sample()
        return action.squeeze(0).numpy()  # Convert the action tensor to a NumPy array


    def render_episode(self, env):
        state = env.reset()
        while True:
            action = self.select_action(state)
            state, _, done, _ = env.step(action)
            env.render()
            if done:
                break

if __name__ == "__main__":
    # Define the environment and its rendering setup
    ENV_ID = "RocketLander-v0"
    if ENV_ID not in gym.envs.registry.env_specs:
        register(
            id=ENV_ID,
            entry_point='env:RocketLander',  # Replace with the correct path to your module
        )

    env = gym.make(ENV_ID)
    env.render()

    # Define the state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Create the SAC agent
    sac_agent = SACAgent(state_dim, action_dim)

    # Load the pre-trained model
    model_path = 'sac_models\sac_model_episode_1000.pt'
    checkpoint = torch.load(model_path)

    # Check the keys in the checkpoint dictionary
    print("Keys in the checkpoint:", checkpoint.keys())

    # Load the actor's state dict using the correct keys
    sac_agent.actor.load_state_dict({
        'fc1.weight': checkpoint['fc1.weight'],
        'fc1.bias': checkpoint['fc1.bias'],
        'fc2.weight': checkpoint['fc2.weight'],
        'fc2.bias': checkpoint['fc2.bias'],
        'mean.weight': checkpoint['mean.weight'],
        'mean.bias': checkpoint['mean.bias'],
        'log_std.weight': checkpoint['log_std.weight'],
        'log_std.bias': checkpoint['log_std.bias'],
    })

    # Render the agent's performance
    sac_agent.render_episode(env)

    # Close the environment rendering
    env.close()
