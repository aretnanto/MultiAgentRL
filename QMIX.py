import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import random
from pettingzoo.mpe import simple_tag_v3
from pettingzoo.utils import average_total_reward



class MixingNetwork(nn.Module):
    def __init__(self, state_space, action_space, hidden_size=24):
        super(MixingNetwork, self).__init__()
        self.fc1 = nn.Linear(state_space, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.action_weights = nn.Linear(hidden_size, action_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_weights = self.action_weights(x)
        return action_weights

####################################

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
        self.mixers = {}
        self.buffer = {key: [] for key in self.env.agents}

        for agent_type, agents in self.agent_types.items():
            obs_dim = self.env.observation_space(agents[0]).shape[0]
            action_dim = self.env.action_space(agents[0]).n
            q_network = AgentQNetwork(obs_dim, action_dim)
            target_q_network = AgentQNetwork(obs_dim, action_dim)
            target_q_network.load_state_dict(q_network.state_dict())
            self.q_networks[agent_type] = q_network
            self.target_q_networks[agent_type] = target_q_network
            self.mixers[agent_type] = MixingNetwork(obs_dim, action_dim)
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
    
    
    
    def _get_q_values(self, obs):
        q_values = {}
        for agent_type, agents in self.agent_types.items():
            for agent_id in agents:
                q_values[agent_id] = self.q_networks[agent_type](torch.tensor(obs[agent_id]).float())
        return q_values
    

    def _update_q_networks(self):
        for agent_type, agents in self.agent_types.items():
            obs_batch_list, act_batch_list, rew_batch_list, next_obs_batch_list, done_batch_list = self._get_batch_data(agents)

            if obs_batch_list:
                joint_q_values, joint_target_q_values = self._calculate_joint_q_values(agents, agent_type, obs_batch_list, act_batch_list, next_obs_batch_list)

                target_joint_q_values = torch.tensor(rew_batch_list) + (1 - torch.tensor(done_batch_list)) * self.gamma * torch.max(joint_target_q_values, dim=-1)[0]

                for i, agent_id in enumerate(agents):
                    q_values_i = joint_q_values[:, i]
                    target_q_values_i = target_joint_q_values[:, i]
                    loss = self.criterion(q_values_i, target_q_values_i.detach())
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
        print(obs_batch_list[-1])
        q_values = self._get_q_values(obs_batch_list[-1])
        next_q_values = self._get_q_values(next_obs_batch_list[-1])
        next_actions = {}
        for agent_id in agents:
            if np.random.rand() < self.epsilon:
                next_actions[agent_id] = self.env.action_space(agent_id).sample()
            else:
                print(next_q_values[agent_id])
                next_actions[agent_id] = torch.argmax(next_q_values[agent_id]).item()

        joint_q_values = []
        joint_target_q_values = []
        for i, agent_id in enumerate(agents):
            q_values_i = q_values[agent_id]
            next_q_values_i = next_q_values[agent_id]
            next_action_i = next_actions[agent_id]
            mixer_input = torch.cat([q_values_i, next_q_values_i], dim=-1)
            action_weights_i = self.mixers[agent_type](mixer_input)
            q_values_i = torch.sum(action_weights_i * F.one_hot(torch.tensor(act_batch_list[i]), self.env.action_space(agent_id).n).float(), dim=-1)
            joint_q_values.append(q_values_i)
            target_q_values_i = self.target_q_networks[agent_type](torch.tensor(next_obs_batch_list[-1][agent_id]).float())
            target_q_values_i = torch.sum(target_q_values_i * F.one_hot(torch.tensor(next_action_i), self.env.action_space(agent_id).n).float(), dim=-1)
            joint_target_q_values.append(target_q_values_i)

        return torch.stack(joint_q_values, dim=-1), torch.stack(joint_target_q_values, dim=-1)

    def _calculate_target_joint_q_values(self, rew_batch_list, act_batch_list, done_batch_list, joint_q_values, joint_next_q_values):
        max_joint_next_q_values, _ = joint_next_q_values.max(dim=1)
        target_joint_q_values = joint_q_values.clone().detach()

        for i, (rew_batch, done_batch) in enumerate(zip(rew_batch_list, done_batch_list)):
            target_joint_q_values[np.arange(len(target_joint_q_values)), act_batch_list[i].squeeze()] = rew_batch + self.gamma * (1 - done_batch) * max_joint_next_q_values

        return target_joint_q_values

if __name__ == "__main__":
    learner = VDNLearner()
    learner.train()


