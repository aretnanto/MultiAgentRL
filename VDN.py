import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import random
from pettingzoo.mpe import simple_tag_v3
from pettingzoo.utils import average_total_reward

class AgentQNetwork(nn.Module):
    def __init__(self, state_space, action_space, hidden_size=24):
        super(AgentQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_space, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MixingNetwork(nn.Module):
    def __init__(self, num_agents, state_space, hidden_size=24):
        super(MixingNetwork, self).__init__()
        self.num_agents = num_agents
        self.state_space = state_space
        self.fc1 = nn.Linear(self.num_agents * state_space, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, agent_qs, states):
        agent_qs = agent_qs.view(-1, self.num_agents * self.state_space)
        states = states.view(-1, self.num_agents * self.state_space)
        x = torch.cat((agent_qs, states), 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ... (imports and AgentQNetwork class)

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
        self.env = simple_tag_v3.parallel_env()
        self.env.reset()

        self.adversaries = ['adversary_0', 'adversary_1', 'adversary_2']
        self.non_adversaries = ['agent_0']

        self.agent_types = {'adversary': self.adversaries, 'non_adversary': self.non_adversaries}

        self.q_networks = {}
        self.target_q_networks = {}
        self.optimizers = {}
        self.buffer = {key: [] for key in self.env.agents}

        for agent_type, agents in self.agent_types.items():
            obs_dim = self.env.observation_space(agents[0]).shape[0]
            action_dim = self.env.action_space(agents[0]).n
            q_network = AgentQNetwork(obs_dim, action_dim)
            target_q_network = AgentQNetwork(obs_dim, action_dim)
            target_q_network.load_state_dict(q_network.state_dict())
            self.q_networks[agent_type] = q_network
            self.target_q_networks[agent_type] = target_q_network
            self.optimizers[agent_type] = optim.Adam(q_network.parameters(), lr=learning_rate)

    def _update_target_networks(self):
        for agent_type in self.agent_types.keys():
            self.target_q_networks[agent_type].load_state_dict(self.q_networks[agent_type].state_dict())


    def train(self):
        total_rewards = {agent: [] for agent in self.env.agents}

        for episode in range(self.episodes):
            obs, _ = self.env.reset()
            done = {agent: False for agent in self.env.agents}
            step_count = 0
            episode_rewards = {agent: 0 for agent in self.env.agents}

            while not all(done.values()):
                actions = self._get_actions(obs)

                next_obs, rewards, done, _, _ = self.env.step(actions)

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

        # Calculate and print average total rewards for adversaries and non-adversaries
        avg_adversary_rewards = []
        avg_non_adversary_rewards = []

        for agent_id, rewards in total_rewards.items():
            avg_reward = sum(rewards) / len(rewards)
            if agent_id in self.agent_types['adversary']:
                avg_adversary_rewards.append(avg_reward)
            else:
                avg_non_adversary_rewards.append(avg_reward)

        print("Average total reward for adversaries:", sum(avg_adversary_rewards) / len(avg_adversary_rewards))
        print("Average total reward for non-adversaries:", sum(avg_non_adversary_rewards) / len(avg_non_adversary_rewards))


    def _get_actions(self, obs):
        actions = {}
        for agent_type, agents in self.agent_types.items():
            for agent_id in agents:
                if np.random.rand() < self.epsilon:
                    action = self.env.action_space(agent_id).sample()
                else:
                    q_values = self.q_networks[agent_type](torch.tensor(obs[agent_id]).float())
                    action = torch.argmax(q_values).item()
                actions[agent_id] = action
        return actions

    def _store_transition(self, obs, actions, rewards, next_obs, done):
        for agent_id in self.env.agents:
            self.buffer[agent_id].append((obs[agent_id], actions[agent_id], rewards[agent_id], next_obs[agent_id], done[agent_id]))

    def _update_q_networks(self):
        for agent_type, agents in self.agent_types.items():
            obs_batch_list, act_batch_list, rew_batch_list, next_obs_batch_list, done_batch_list = self._get_batch_data(agents)

            if obs_batch_list:
                joint_q_values, joint_target_q_values = self._calculate_joint_q_values(agents, agent_type, obs_batch_list, act_batch_list, next_obs_batch_list)

                target_joint_q_values = self._calculate_target_joint_q_values(rew_batch_list, act_batch_list, done_batch_list, joint_q_values, joint_target_q_values)

                loss = self.criterion(joint_q_values, target_joint_q_values.detach())

                self.optimizers[agent_type].zero_grad()
                loss.backward()
                self.optimizers[agent_type].step()

    def _get_batch_data(self, agents):
        obs_batch_list, act_batch_list, rew_batch_list, next_obs_batch_list, done_batch_list = [], [], [], [], []
        for agent_id in agents:
            if len(self.buffer[agent_id]) >= self.batchsize:
                minibatch = random.sample(self.buffer[agent_id], self.batchsize)
                obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = zip(*minibatch)
                obs_batch_list.append(torch.tensor(obs_batch).float())
                act_batch_list.append(torch.tensor(act_batch).long())
                rew_batch_list.append(torch.tensor(rew_batch).float())
                next_obs_batch_list.append(torch.tensor(next_obs_batch).float())
                done_batch_list.append(torch.tensor(done_batch).float())
        return obs_batch_list, act_batch_list, rew_batch_list, next_obs_batch_list, done_batch_list

    def _calculate_joint_q_values(self, agents, agent_type, obs_batch_list, act_batch_list, next_obs_batch_list):
        joint_q_values = torch.zeros(self.batchsize, self.env.action_space(agents[0]).n)
        joint_next_q_values = torch.zeros(self.batchsize, self.env.action_space(agents[0]).n)

        for i, agent_id in enumerate(agents):
            q_values = self.q_networks[agent_type](obs_batch_list[i])
            next_q_values = self.target_q_networks[agent_type](next_obs_batch_list[i])
            joint_q_values[np.arange(len(joint_q_values)), act_batch_list[i].squeeze()] += q_values[np.arange(len(q_values)), act_batch_list[i].squeeze()]
            joint_next_q_values += next_q_values

        return joint_q_values, joint_next_q_values

    def _calculate_target_joint_q_values(self, rew_batch_list, act_batch_list, done_batch_list, joint_q_values, joint_next_q_values):
        max_joint_next_q_values, _ = joint_next_q_values.max(dim=1)
        target_joint_q_values = joint_q_values.clone().detach()

        for i, (rew_batch, done_batch) in enumerate(zip(rew_batch_list, done_batch_list)):
            target_joint_q_values[np.arange(len(target_joint_q_values)), act_batch_list[i].squeeze()] = rew_batch + self.gamma * (1 - done_batch) * max_joint_next_q_values

        return target_joint_q_values

class QMIXLearner(VDNLearner):
    def __init__(self, *args, **kwargs):
        super(QMIXLearner, self).__init__(*args, **kwargs)
        num_agents = len(self.env.agents)
        state_space = self.env.observation_space(self.env.agents[0]).shape[0]
        self.mixing_networks = {}
        self.target_mixing_networks = {}
        self.mixing_optimizers = {}
        for agent_type in self.agent_types.keys():
            mixing_network = MixingNetwork(num_agents, state_space)
            target_mixing_network = MixingNetwork(num_agents, state_space)
            target_mixing_network.load_state_dict(mixing_network.state_dict())
            self.mixing_networks[agent_type] = mixing_network
            self.target_mixing_networks[agent_type] = target_mixing_network
            self.mixing_optimizers[agent_type] = optim.Adam(mixing_network.parameters(), lr=self.learning_rate)

    def _update_target_networks(self):
        super()._update_target_networks()
        for agent_type in self.agent_types.keys():
            self.target_mixing_networks[agent_type].load_state_dict(self.mixing_networks[agent_type].state_dict())

    def _calculate_joint_q_values(self, agents, agent_type, obs_batch_list, act_batch_list, next_obs_batch_list):
        joint_q_values = torch.zeros(self.batchsize, self.env.action_space(agents[0]).n)
        joint_next_q_values = torch.zeros(self.batchsize, self.env.action_space(agents[0]).n)

        for i, agent_id in enumerate(agents):
            q_values = self.q_networks[agent_type](obs_batch_list[i])
            next_q_values = self.target_q_networks[agent_type](next_obs_batch_list[i])
            joint_q_values[np.arange(len(joint_q_values)), act_batch_list[i].squeeze()] += q_values[np.arange(len(q_values)), act_batch_list[i].squeeze()]
            joint_next_q_values += next_q_values

        joint_q_values = self.mixing_networks[agent_type](joint_q_values, torch.cat(obs_batch_list, 1))
        joint_next_q_values = self.target_mixing_networks[agent_type](joint_next_q_values, torch.cat(next_obs_batch_list, 1))

        return joint_q_values, joint_next_q_values

    def _update_q_networks(self):
        for agent_type, agents in self.agent_types.items():
            obs_batch_list, act_batch_list, rew_batch_list, next_obs_batch_list, done_batch_list = self._get_batch_data(agents)

            if obs_batch_list:
                joint_q_values, joint_target_q_values = self._calculate_joint_q_values(agents, agent_type, obs_batch_list, act_batch_list, next_obs```
