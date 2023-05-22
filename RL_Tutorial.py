import gym 
import numpy as np
import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=24):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

env = gym.make('LunarLander-v2', render_mode="human") 
#make q tables of action and states
num_states = env.observation_space.shape[0]
num_actions = env.action_space.n

#init q
learning_rate = 0.1
discount_factor = 0.99
num_episodes = 1000
QNet = QNetwork(num_states, num_actions)
batch_size = 32

lr_nn = 1e-3

criterion = nn.MSELoss()
optimizer = optim.Adam(QNet.parameters(), lr=learning_rate)

pbar = tqdm(range(num_episodes))
replay_buffer = []
for episode in pbar:
    done = False
    state, _  = env.reset()
    total_reward = 0
    
    steps = 0 
    while steps <=1000 and not done: 
        QNet.train()
        env.render()  
        #epsilon greedy
        if np.random.uniform(0, 1) < 0.1:
            action = env.action_space.sample()
        else:
            q_values = QNet(torch.Tensor(state))
            action = torch.argmax(q_values).item()

        #bootstrap actions
        next_state, reward, done, info, _ = env.step(action)
        replay_buffer.append((state, action, reward, next_state, done))

        state = next_state
        total_reward += reward
        pbar.set_postfix(total_reward=total_reward)
        if len(replay_buffer) >=  batch_size:
            minibatch = random.sample(replay_buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*minibatch)
            states = torch.Tensor(states)
            next_states = torch.Tensor(next_states)
            rewards = torch.Tensor(rewards)
            actions = torch.Tensor(actions).long()
            dones = torch.Tensor(dones)
            cur_q = QNet(torch.Tensor(states))
            q_hold = cur_q.clone()
            next_q = QNet(torch.Tensor(next_states))
            max_val = torch.max(next_q, dim = 1)[0]
            td_target = rewards + discount_factor * max_val * (1 - dones) 
            cur_q[torch.arange(len(actions)), actions] = td_target  #?
            optimizer.zero_grad()
            loss = criterion(q_hold, cur_q.detach())
            loss.backward()
            optimizer.step()
            replay_buffer.pop(0)

        steps += 1
        '''
        
        Q[state, action] += learning_rate * (reward + discount_factor * np.max(Q[next_state]) - Q[state, action])
        state = next_state
        '''

env.close()

