from pettingzoo.mpe import simple_tag_v2
import numpy as np
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import random

good_agent = 1
bad_agent = 3
env = simple_tag_v2.env(render_mode='human')
num_episodes = 2

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


def epsaction(model, obsveration, epsilon):
    if np.random.uniform() < epsilon:
      action = env.action_space(agent).sample()
    else:
      obs_tensor = torch.from_numpy(observation).float()
      q_values = model(obs_tensor)
      action = torch.argmax(q_values).item()
    return action

def updateModel(model, optimizer, criterion,discount_factor, buffer, batchsize):
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


models = {}
optimizers = {}
buffer = {}
learning_rate = 1e-3
env.reset()
epsilon = 0.1
gamma = 0.99
criterion = nn.MSELoss()
episodes = 1000 
max_steps = 100 
batchsize = 64
buffer = {key: [] for key in env.agents}


##Init Network
for agent in env.agents:
  output_actions = env.action_space(agent).n
  input_actions = env.observation_space(agent).shape[0]
  models[agent] = QNetwork(input_actions, output_actions)
  optimizers[agent] = optim.Adam(models[agent].parameters(), lr=learning_rate)


for episode in tqdm(range(episodes)):
    state = env.reset()
    rewards = {key: 0 for key in env.agents}
    score = 0
    i = 0
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        QNet = models[agent]
        cur_optim = optimizers[agent]
        # Q-Update - weird step
        if len(buffer[agent]) > 0: 
          old_observation, action, old_reward, old_termination, old_truncation = buffer[agent].pop()
          buffer[agent].append((old_observation, action, reward, observation, termination))

        if len(buffer[agent]) > batchsize:
            updateModel(QNet, cur_optim, criterion, gamma, buffer[agent], batchsize)

        #Action Step 
        if termination or truncation:
            action = None
        else: 
            action = epsaction(QNet, observation, epsilon)

        env.step(action)
        rewards[agent] = rewards[agent] + reward
        if action is not None: 
          buffer[agent].append((observation, action, reward, termination, truncation))