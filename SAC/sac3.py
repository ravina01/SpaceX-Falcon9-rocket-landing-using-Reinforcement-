import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import ptan
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
    def __init__(self, net, scale_factor=2.0, device="cpu"):
        self.net = net
        self.scale_factor = scale_factor
        self.device = device

    def __call__(self, states):
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)

        # Scale the output of the actor before applying tanh
        scaled_output = self.scale_factor * self.net(states)
        
        # Use tanh activation function to squash values in the range [-1, 1]
        actions = torch.tanh(scaled_output).cpu().detach().numpy()

        return actions


class SAC:
    def __init__(self, state_dim, action_dim, alpha=0.1, gamma=0.99, tau=5e-3, lr=1e-4, buffer_size=100000000, device = 'cpu'):
        self.actor = Actor(state_dim, action_dim)
        self.critic1 = Critic(state_dim, action_dim)
        self.critic2 = Critic(state_dim, action_dim)
        self.target_critic1 = Critic(state_dim, action_dim)
        self.target_critic2 = Critic(state_dim, action_dim)

        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.target_actor = Actor(state_dim, action_dim)
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

        self.buffer = ptan.experience.ExperienceReplayBuffer(None, buffer_size=buffer_size)
        self.agent = SACAgent(self.actor, device="cpu")
        self.device = device

    def update_critic(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            next_q1 = self.target_critic1(next_states, next_actions)
            next_q2 = self.target_critic2(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2)
            entropy_term_per_dim = -self.alpha * torch.log(self.actor(next_states).clamp(1e-6, 1.0))

            # Aggregate the entropy terms (e.g., take the mean)
            entropy_term = entropy_term_per_dim.sum(dim=1, keepdim=True)

            # Calculate next_value with entropy regularization
            next_value = next_q - entropy_term

        # target_value = rewards + (1 - dones) * self.gamma * next_value
        target_value = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_value
        target_value = target_value.squeeze(1)
    
      
        
        critic1_loss = nn.MSELoss()(self.critic1(states, actions), target_value.unsqueeze(1))
        critic2_loss = nn.MSELoss()(self.critic2(states, actions), target_value.unsqueeze(1))


        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

    def update_actor_and_alpha(self, states):
        actions = self.actor(states)
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        q_min = torch.min(q1, q2)
        # actor_loss = (self.alpha * torch.log(actions.clamp(-1.0, 1.0)) - q_min).mean()
        actor_loss = self.alpha * torch.log(actions - q_min).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # self.alpha = torch.clamp(torch.tensor(self.alpha - 1e-4), 0.05, 1.0).item()
        self.alpha = torch.clamp(torch.tensor(self.alpha - 1e-6), 1e-4, 1.0).item()

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

        for target_param, param in zip(
                self.target_actor.parameters(), self.actor.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def train_step(self, batch_size):
        if len(self.buffer) < batch_size:
            return

        batch = self.buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32).to('cpu')
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32)

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

    def run(self, env, max_steps=10000, batch_size=256, print_mean_reward=True, checkpoint_interval=1000):
        state = env.reset()
        total_reward = 0.0
        rewards_history = []

        last_100_rewards = []
        checkpoint_counter = 0  # Counter for checkpoint filenames

        for step in range(max_steps):
            action = self.agent([state])[0]

            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            # env.render()
            self.buffer._add((state, action, reward, next_state, done))

            state = next_state

            if len(self.buffer) > batch_size:
                self.train_step(batch_size)

            if done:
                state = env.reset()
                rewards_history.append(total_reward)
                last_100_rewards.append(total_reward)

                if len(last_100_rewards) > 100:
                    last_100_rewards.pop(0)

                mean_reward_last_100 = np.mean(last_100_rewards)

                print(f"Step: {step}, Total Reward: {total_reward:.2f}, Mean Reward (Last 100 episodes): {mean_reward_last_100:.2f}")

                total_reward = 0.0

            # Save model at checkpoints
            if step % checkpoint_interval == 0 and step > 0:
                checkpoint_counter += 1
                actor_filename = f'sac/actor_checkpoint_{checkpoint_counter}.pth'
                critic1_filename = f'sac/critic1_checkpoint_{checkpoint_counter}.pth'
                critic2_filename = f'sac/critic2_checkpoint_{checkpoint_counter}.pth'
                self.save_models(actor_filename, critic1_filename, critic2_filename)

        # Save the final models
        self.save_models('sac/actor_final.pth', 'sac/critic1_final.pth', 'sac/critic2_final.pth')
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