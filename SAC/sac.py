import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import ptan
from torch.distributions import Normal
import matplotlib.pyplot as plt


# Define the Actor and Critic networks
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class SACAgent(ptan.agent.BaseAgent):
    def __init__(self, net, device="cpu"):
        self.net = net
        self.device = device

    def __call__(self, states):
        states = torch.tensor(states, dtype=torch.float32).to('cpu')
        actions = torch.tanh(self.net(states))
        return actions.cpu().detach().numpy()



# Soft Actor-Critic algorithm
class SAC:
    def __init__(self, state_dim, action_dim, alpha=0.3, gamma=0.99, tau=5e-3, lr=3e-4):
        self.actor = Actor(state_dim, action_dim)
        self.critic1 = Critic(state_dim, action_dim)
        self.critic2 = Critic(state_dim, action_dim)
        self.target_critic1 = Critic(state_dim, action_dim)
        
        self.target_critic2 = Critic(state_dim, action_dim)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.target_actor = Actor(state_dim, action_dim)  # Add target actor
        self.target_actor.load_state_dict(self.actor.state_dict())


        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

        self.buffer = ptan.experience.ExperienceReplayBuffer(None, buffer_size=100000000)
        self.agent = SACAgent(self.actor, device="cpu")
        self.observation_mean = np.zeros(state_dim, dtype=np.float32)
        self.observation_var = np.ones(state_dim, dtype=np.float32)
        self.observation_count = 0

    def normalize_observation(self, observation):
        # Update mean and variance incrementally
        self.observation_count += 1
        delta = observation - self.observation_mean
        self.observation_mean += delta / self.observation_count
        delta2 = observation - self.observation_mean
        self.observation_var += delta * delta2

        # Normalize observation
        normalized_observation = (observation - self.observation_mean) / np.sqrt(self.observation_var / self.observation_count)

        return normalized_observation



    def update_critic(self, states, actions, rewards, next_states, dones):
        states = np.array([self.normalize_observation(s) for s in states])
        next_states = np.array([self.normalize_observation(s) for s in next_states])
        states = torch.tensor(states, dtype=torch.float32).to("cpu")
        actions = torch.tensor(actions, dtype=torch.float32).to("cpu")
        rewards = torch.tensor(rewards, dtype=torch.float32).to("cpu")
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to("cpu")
        dones = torch.tensor(dones, dtype=torch.float32).to("cpu")

        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            next_q1 = self.target_critic1(next_states, next_actions)
            next_q2 = self.target_critic2(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2)
            next_value = next_q - self.alpha * torch.log(self.actor(next_states).clamp(1e-6, 1.0))

        target_value = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_value

        critic1_loss = nn.MSELoss()(self.critic1(states, actions), target_value)
        critic2_loss = nn.MSELoss()(self.critic2(states, actions), target_value)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=1.0)
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1.0)
        self.critic2_optimizer.step()    

    # def update_critic(self, states, actions, rewards, next_states, dones):
    #     states = torch.tensor(np.array(states), dtype=torch.float32).to("cpu")
    #     actions = torch.tensor(np.array(actions), dtype=torch.float32).to("cpu")
    #     rewards = torch.tensor(rewards, dtype=torch.float32).to("cpu")
    #     next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to("cpu")
    #     dones = torch.tensor(dones, dtype=torch.float32).to("cpu")

    #     with torch.no_grad():
    #         next_actions = self.target_actor(next_states)
    #         next_q1 = self.target_critic1(next_states, next_actions)
    #         next_q2 = self.target_critic2(next_states, next_actions)
    #         next_q = torch.min(next_q1, next_q2)
    #         next_value = next_q - self.alpha * torch.log(self.actor(next_states).clamp(1e-6, 1.0))

    #     target_value = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_value
        
    #     # Ensure target_value has the same size as the output of the critic networks
    #     target_value = target_value.squeeze(-1)
      
        
    #     # target_value = target_value.view(-1, 1)
      

    #     critic1_loss = nn.MSELoss()(self.critic1(states, actions), target_value)
    #     critic2_loss = nn.MSELoss()(self.critic2(states, actions), target_value)

    #     self.critic1_optimizer.zero_grad()
    #     critic1_loss.backward()
    #     self.critic1_optimizer.step()

    #     self.critic2_optimizer.zero_grad()
    #     critic2_loss.backward()
    #     self.critic2_optimizer.step()



    def update_actor_and_alpha(self, states):
        states = np.array([self.normalize_observation(s) for s in states])
        states = torch.tensor(states, dtype=torch.float32).to("cpu")
        actions = self.actor(states)
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        q_min = torch.min(q1, q2)
        # q_min = torch.min(q1, q2).clamp(-1.0 / (1.0 - self.gamma), 0)
        actor_loss = (self.alpha * torch.log(actions.clamp(-1.0, 1.0)) - q_min).mean()


        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # alpha_loss = (-self.alpha * (torch.log(actions.clamp(1e-6, 1.0)) - q_min.detach())).mean()

        self.alpha = torch.clamp(torch.tensor(self.alpha - 1e-4), 0.05, 1.0).item()

   


    def update_targets(self):
        for target_param, param in zip(
            self.target_critic1.parameters(), self.critic1.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

        for target_param, param in zip(
            self.target_critic2.parameters(), self.critic2.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def train_step(self, batch_size):
        if len(self.buffer) < batch_size:
            return

        batch = self.buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        self.update_critic(states, actions, rewards, next_states, dones)
        self.update_actor_and_alpha(states)
        self.update_targets()

    def save_models(self, actor_path, critic1_path, critic2_path):
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic1.state_dict(), critic1_path)
        torch.save(self.critic2.state_dict(), critic2_path)
        print("Models saved successfully!")

    def load_models(self, actor_path, critic1_path, critic2_path):
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic1.load_state_dict(torch.load(critic1_path))
        self.critic2.load_state_dict(torch.load(critic2_path))
        print("Models loaded successfully!")


    def run(self, env, max_steps=10000, batch_size=256):
        state = env.reset()
        total_reward = 0.0
        rewards_history = []
        for step in range(max_steps):
            action = self.actor(torch.tensor(self.normalize_observation(state)).float().to("cpu")).cpu().detach().numpy()
            # env.render()
      
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            self.buffer._add((state, action, reward, next_state, done))
     
            state = next_state

            if len(self.buffer) > batch_size:
                self.train_step(batch_size)

            if done:
                state = env.reset()
                rewards_history.append(total_reward)
                print(f"Step: {step}, Total Reward: {total_reward:.2f}")
                total_reward = 0.0
                

        self.save_models('sac3/actor.pth', 'sac3/critic1.pth', 'sac3/critic2.pth')

        self.save_plot(rewards_history, 'sac3/episode_vs_rewards.png')

    def save_plot(self, rewards_history, filename):
        plt.figure(figsize=(10, 6))
        plt.plot(rewards_history, label='Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Episodes vs. Rewards')
        plt.legend()
        plt.savefig(filename)
        plt.show()