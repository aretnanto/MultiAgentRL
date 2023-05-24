import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import random
from pettingzoo.mpe import simple_spread_v3
from pettingzoo.utils import average_total_reward

class AgentQNetwork(nn.Module):
    def __init__(self, state_space, action_space, agent_id, hidden_size=24):
        super(AgentQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_space, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_space)
        self.agent_id = agent_id

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class VDNLearner():
    def __init__(
        self,
        learning_rate=1e-3,
        epsilon=0.1,
        gamma=0.99,
        episodes=1000,
        batchsize=64,
        update_target_freq=100
    ):
        self.epsilon = epsilon
        self.gamma = gamma
        self.criterion = nn.MSELoss()
        self.episodes = episodes
        self.batchsize = batchsize
        self.update_target_freq = update_target_freq
        self.env = simple_spread_v3.parallel_env()
        self.env.reset()

        self.agents = self.env.agents
        self.q_networks = {}
        self.target_q_networks = {}
        self.optimizers = {}
        self.buffer = {key: [] for key in self.agents}

        for agent_id in self.agents:
            obs_dim = self.env.observation_space(agent_id).shape[0]
            action_dim = self.env.action_space(agent_id).n
            q_network = AgentQNetwork(obs_dim, action_dim, agent_id)
            target_q_network = AgentQNetwork(obs_dim, action_dim, agent_id)
            target_q_network.load_state_dict(q_network.state_dict())
            self.q_networks[agent_id] = q_network
            self.target_q_networks[agent_id] = target_q_network
            self.optimizers[agent_id] = optim.Adam(q_network.parameters(), lr=learning_rate)

    def _update_target_networks(self):
        for agent_id in self.agents:
            self.target_q_networks[agent_id].load_state_dict(self.q_networks[agent_id].state_dict())

    def train(self):
        total_rewards = {agent_id: [] for agent_id in self.agents}

        for episode in range(self.episodes):
            obs, _ = self.env.reset()
            done = {agent_id: False for agent_id in self.agents}
            step_count = 0
            episode_rewards = {agent_id: 0 for agent_id in self.agents}

            while not all(done.values()):
                actions = self._get_actions(obs)

                next_obs, rewards, done, _, _ = self.env.step(actions)
            
                if done == {}:
                    break
                if next_obs == {}:
                    break

                # adding observation noise to see how robust it is
                # this would be the norm in real world applications

                for key in next_obs.keys():
                    next_obs[key] = next_obs[key]+ np.random.normal(0,2,18)


                self._store_transition(obs, actions, rewards, next_obs, done)

                for agent_id, reward in rewards.items():
                    episode_rewards[agent_id] += reward

                if step_count % self.update_target_freq == 0:
                    self._update_target_networks()

                self._update_q_networks()

                obs = next_obs
                step_count += 1

            for agent_id, reward in episode_rewards.items():
                total_rewards[agent_id].append(reward)

        # Calculate and print average total rewards for all agents
        avg_rewards = []

        for agent_id, rewards in total_rewards.items():
            avg_reward = sum(rewards) / len(rewards)
            avg_rewards.append(avg_reward)

        print("Average total reward for all agents:", sum(avg_rewards) / len(avg_rewards))

    def _get_actions(self, obs):
        actions = {}
        for agent_id in self.agents:
            if np.random.rand() < self.epsilon:
                action = self.env.action_space(agent_id).sample()
            else:
                q_values = self.q_networks[agent_id](torch.tensor(obs[agent_id]).float())
                action = torch.argmax(q_values).item()
            actions[agent_id] = action
        return actions

    def _store_transition(self, obs, actions, rewards, next_obs, done):
        for agent_id in self.agents:
            self.buffer[agent_id].append((obs[agent_id], actions[agent_id], rewards[agent_id], next_obs[agent_id], done[agent_id]))

    def _update_q_networks(self):
        for agent_id in self.agents:
            obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = self._get_batch_data(agent_id)

            if obs_batch is not None:
                q_values = self.q_networks[agent_id](obs_batch)
                q_values = torch.gather(q_values, 1, act_batch.unsqueeze(1))

                next_q_values = self.target_q_networks[agent_id](next_obs_batch)
                next_q_values = torch.max(next_q_values, dim=1, keepdim=True)[0]

                target_q_values = rew_batch + self.gamma * next_q_values * (1 - done_batch)
                loss = self.criterion(q_values, target_q_values.detach())

                self.optimizers[agent_id].zero_grad()
                loss.backward()
                self.optimizers[agent_id].step()

    def _get_batch_data(self, agent_id):
        if len(self.buffer[agent_id]) < self.batchsize:
            return None, None, None, None, None

        indices = np.random.choice(len(self.buffer[agent_id]), self.batchsize)
        batch = [self.buffer[agent_id][i] for i in indices]

        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = zip(*batch)

        obs_batch = torch.tensor(obs_batch, dtype=torch.float)
        act_batch = torch.tensor(act_batch, dtype=torch.long)
        rew_batch = torch.tensor(rew_batch, dtype=torch.float).unsqueeze(1)
        next_obs_batch = torch.tensor(next_obs_batch, dtype=torch.float)
        done_batch = torch.tensor(done_batch, dtype=torch.float).unsqueeze(1)

        return obs_batch, act_batch, rew_batch, next_obs_batch, done_batch

if __name__ == "__main__":
    vdn_learner = VDNLearner()
    vdn_learner.train()

