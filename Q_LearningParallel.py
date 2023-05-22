from pettingzoo.mpe import simple_tag_v3
from pettingzoo.utils import average_total_reward
from pettingzoo.utils.conversions import aec_to_parallel
import numpy as np
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import random


class QNetwork(nn.Module):
    def __init__(self, state_space, action_space, hidden_size=24):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_space, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_space)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class TagLearner(): 
    def __init__(self, learning_rate = 1e-3, epsilon = 0.1, gamma = 0.99, episodes = 1000, batchsize = 64): 
        self.models = {}
        self.optimizers = {}
        self.buffer = {}
        self.epsilon = epsilon
        self.gamma =0.99
        self.criterion = nn.MSELoss()
        self.episodes = episodes
        self.batchsize = batchsize
        self.env = simple_tag_v3.env(render_mode='rgb_array')
        self.env = aec_to_parallel(self.env)
        self.env.reset()
        self.buffer = {key: [] for key in self.env.agents}
        for agent in self.env.agents:
            output_actions = self.env.action_space(agent).n
            input_actions = self.env.observation_space(agent).shape[0]
            self.models[agent] = QNetwork(input_actions, output_actions)
            self.optimizers[agent] = optim.Adam(self.models[agent].parameters(), lr=learning_rate)
    
    def epsaction(self, agent, model, observation, epsilon):
        if np.random.uniform() < epsilon: 
            action = self.env.action_space(agent).sample()
        else: 
            obs_tensor = torch.from_numpy(observation).float()
            q_values = model(obs_tensor)
            action = torch.torch.argmax(q_values).item()
        return action
    
    
    def updateModel(self, model, optimizer, criterion,discount_factor, buffer, batchsize):
        model.eval() 
        minibatch = random.sample(buffer, batchsize)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.Tensor(states)
        next_states = torch.Tensor(next_states)
        rewards = torch.Tensor(rewards)
        actions = torch.Tensor(actions).long()
        dones = torch.Tensor(dones)
        cur_q = model(torch.Tensor(states))
        q_hold = cur_q.clone()
        next_q = model(torch.Tensor(next_states))
        max_val = torch.max(next_q, dim = 1)[0]
        td_target = rewards + discount_factor * max_val * (1 - dones)
        cur_q[torch.arange(len(actions)), actions] = td_target  #?
        optimizer.zero_grad()
        loss = criterion(q_hold, cur_q.detach())
        loss.backward()
        optimizer.step()

    def train(self):
        rewards_array = {key: [] for key in self.env.agents}
        for episode in tqdm(range(self.episodes)):
            observations, _ = self.env.reset()
            rewards = {key: 0 for key in self.env.agents}
            for cur_step in range(0, 25):
                actions = {}
                for agent in self.env.agents:
                    action = self.epsaction(agent, self.models[agent], observations[agent], self.epsilon)
                    actions[agent] = action
                next_observations, rewards, terminations, truncations, infos = self.env.step(actions)
                print(next_observations)

                '''
            for agent in self.env.agent_iter():
                observation, reward, termination, truncation, info = self.env.last()
                QNet = self.models[agent]
                cur_optim = self.optimizers[agent]

                # Q-Update - weird step
                if len(self.buffer[agent]) > 0: 
                    old_observation, action, old_reward, old_termination, old_truncation = self.buffer[agent].pop()
                    self.buffer[agent].append((old_observation, action, reward, observation, termination))

                if len(self.buffer[agent]) > self.batchsize:
                    self.updateModel(QNet, cur_optim, self.criterion, self.gamma, self.buffer[agent], self.batchsize)
                #Action Step 
                if termination or truncation:
                    action = None
                else: 
                    action = self.epsaction(agent, QNet, observation, self.epsilon)

                self.env.step(action)
                rewards[agent] = rewards[agent] + reward
                if action is not None: 
                    self.buffer[agent].append((observation, action, reward, termination, truncation))
            for agent in rewards_array.keys():
                rewards_array[agent].append(rewards[agent])
            '''
        return rewards_array

    def atr(self): 
        return average_total_reward(self.env, max_episodes=self.episodes, max_steps=self.episodes*25)

if __name__ == '__main__':
    tag = TagLearner()
    rewards = tag.train()
    print(rewards)
    random_policy = tag.atr()