from pettingzoo.mpe import simple_tag_v3
from pettingzoo.utils import average_total_reward
import numpy as np
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import random

class Actor(nn.Module):
    def __init__(self, state_space, action_space,  hidden_size = 24):
       super(Actor, self).__init__()
       self.actor1 = nn.Linear(state_space, hidden_size)
       self.actor2 = nn.Linear(hidden_size, action_space)

    def forward(self, x):
       actor = F.relu(self.actor1(x))
       actor = F.relu(self.actor2(actor))
       actor = F.softmax(actor)
       return actor
       
class Critic(nn.Module):
    def __init__(self, state_space,  hidden_size = 24):
       super(Critic, self).__init__()
       self.critic1 = nn.Linear(state_space, hidden_size)
       self.critic2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
       critic = F.relu(self.critic1(x))
       critic = F.relu(self.critic2(critic))
       return critic 
       
class TagLearner(): 
    def __init__(self, learning_rate_actor = 1e-3, learning_rate_critic = 1e-3, epsilon = 0.1, gamma = 0.99, episodes = 1000): 
        self.actors = {}
        self.critics = {}
        self.optimizers_actors = {}
        self.optimizers_critics = {}
        self.buffer = {}
        self.epsilon = epsilon
        self.gamma =0.99
        self.episodes = episodes
        self.env = simple_tag_v3.env(render_mode='rgb_array')
        self.env.reset()
        self.buffer = {key: [] for key in self.env.agents}
        for agent in self.env.agents:
            output_actions = self.env.action_space(agent).n
            input_actions = self.env.observation_space(agent).shape[0]
            self.actors[agent] = Actor(input_actions, output_actions)
            self.critics[agent] = Critic(input_actions, output_actions)
            self.optimizers_actors[agent] = optim.Adam(self.actors[agent].parameters(), lr=learning_rate_actor)
            self.optimizers_critics[agent] = optim.Adam(self.critics[agent].parameters(), lr=learning_rate_critic)

    def epsaction(self, agent, actor, observation, epsilon):
        if np.random.uniform() < epsilon: 
            action = self.env.action_space(agent).sample()
        else: 
            obs_tensor = torch.from_numpy(observation).float()
            actions = actor(obs_tensor)
            action = torch.multinomial(actions, num_samples = 1).item()
        return action
    
    def updateModel(self, agent,discount_factor, buffer):
        actor = self.actors[agent]
        critic = self.critics[agent]
        actor_optim = self.optimizers_actors[agent]
        critic_optim = self.optimizers_critics[agent]
        state, action, reward, next_state, done = buffer[0]
        state = torch.Tensor(state)
        reward = torch.Tensor([reward])
        action = torch.Tensor([action]).long()
        done = torch.Tensor([done])
        #Update Critic
        value = critic(torch.Tensor(state))
        value_next = critic(torch.Tensor(next_state))
        td_error = (reward + discount_factor * value_next - value)
        critic_optim.zero_grad()
        td_error.pow(2).backward()
        critic_optim.step()

        #Update Actor
        advantage = td_error.detach()
        action_probs = actor(state)
        actor_loss = -torch.log(action_probs[action]) * advantage
        actor_optim.zero_grad()
        actor_loss.backward()
        actor_optim.step()

    def train(self):
        rewards_array = {key: [] for key in self.env.agents}
        for episode in tqdm(range(self.episodes)):
            state = self.env.reset()
            rewards = {key: 0 for key in self.env.agents}
            for agent in self.env.agent_iter():
                observation, reward, termination, truncation, info = self.env.last()
                actor = self.actors[agent]
                cur_optim = self.optimizers_actors[agent]

                if len(self.buffer[agent]) > 0: 
                    old_observation, action, old_reward, old_termination, old_truncation = self.buffer[agent].pop()
                    self.buffer[agent].append((old_observation, action, reward, observation, termination))
                    self.updateModel(agent, self.gamma, self.buffer[agent])
                    self.buffer[agent].pop()

                #Action Step 
                if termination or truncation:
                    action = None
                else: 
                    action = self.epsaction(agent, actor, observation, self.epsilon)

                self.env.step(action)
                rewards[agent] = rewards[agent] + reward
                if action is not None: 
                    self.buffer[agent].append((observation, action, reward, termination, truncation))        
            
            for agent in rewards_array.keys():
                rewards_array[agent].append(rewards[agent])
        return rewards_array

    def atr(self): 
        return average_total_reward(self.env, max_episodes=self.episodes, max_steps=self.episodes*25)

if __name__ == '__main__':
    tag = TagLearner()
    rewards = tag.train()
    print(rewards)
    random_policy = tag.atr()
