import numpy as np

import Settings as cfg

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class AC():
    def __init__(self, obs_len, actions_n, device):
        self.ac_net = ActorCriticNetwork(obs_len, actions_n).to(device)
        print(self.ac_net)
        self.device = device
        self.log_probs = None

        self.actor_loss = 0
        self.critic_loss = 0
        # self.critic_value_current = 0
        # self.critic_value_new = 0

    def action(self, obs):
        # policy
        policy, _ = self.ac_net(obs)
        policy = F.softmax(policy, dim=1)

        action_probs = torch.distributions.Categorical(policy)
        action = action_probs.sample()
        self.log_probs = action_probs.log_prob(action)
        action = action.item()
        return action

    def learn(self, current_state, action, reward, done, new_state):
        self.ac_net.optimizer.zero_grad()

        current_state = np.array([current_state], dtype=np.float32, copy=False)
        current_state = torch.tensor(current_state).to(self.device)

        new_state = np.array([new_state], dtype=np.float32, copy=False)
        new_state = torch.tensor(new_state).to(self.device)

        _, critic_value_current = self.ac_net.forward(current_state)
        _, critic_value_new = self.ac_net.forward(new_state)

        r = torch.tensor(reward, dtype=torch.float).to(self.device)
        delta = r + cfg.GAMMA * critic_value_new * (1-int(done)) - critic_value_current

        self.actor_loss = -self.log_probs * delta
        self.critic_loss = delta**2

        (self.actor_loss + self.critic_loss).backward()

        self.ac_net.optimizer.step()

    def update(self, ep, writer):
        if ep % cfg.TENSORBOARD_UPDATE:
            writer.add_scalar('training_actor_loss', self.actor_loss, ep)
            writer.add_scalar('training_critic_loss', self.critic_loss, ep)

class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dims, n_actions):
        super(ActorCriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = 128
        self.fc2_dims = 128
        self.n_actions = n_actions

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        
        self.pi = nn.Linear(self.fc2_dims, n_actions)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=cfg.LEARNING_RATE)

        # self.device = device
        # self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        pi = self.pi(x)
        v = self.v(x)

        return pi, v