from pettingzoo.mpe import simple_tag_v2
import numpy as np
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

env = simple_tag_v2.env(render_mode='rgb_array')
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


models = {}
optimizers = {}
learning_rate = 1e-3
env.reset()
epsilon = 0.1
gamma = 0.99
episodes = 2
criterion = nn.MSELoss()
max_steps = 2000


##Init Network
for agent in env.agents:
  output_actions = env.action_space(agent).n
  input_actions = env.observation_space(agent).shape[0]
  models[agent] = QNetwork(input_actions, output_actions)
  optimizers[agent] = optim.Adam(models[agent].parameters(), lr=learning_rate)


for episode in tqdm(range(episodes)):
    env.reset()
    score = 0
    for step in range(max_steps):
      for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()
        model = models[agent]
        optimizer = optimizers[agent]
        if np.random.uniform() < epsilon:
            action = env.action_space(agent).sample()
        else:
          obs_tensor = torch.from_numpy(obs).float()
          q_values = models[agent](obs_tensor)
          action = torch.argmax(q_values).item()