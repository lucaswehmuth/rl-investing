import Settings as cfg
import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

class PPO():
    def __init__(self, obs_len, actions_n, device):
        self.ac = ActorCritic(obs_len, actions_n, 128).to(device)
        self.device = device
        self.optimizer = optim.Adam(self.ac.parameters(), lr=cfg.LEARNING_RATE)
        pass

    def save_checkpoint(self, name, i):
        pass

    def load_checkpoint(self):
        pass

    def compute_gae(next_value, rewards, masks, values):
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + cfg.PPO_GAMMA * values[step + 1] * masks[step] - values[step]
            gae = delta + cfg.PPO_GAMMA * cfg.PPO_GAE_LAMBDA * masks[step] * gae
            # prepend to get correct order back
            returns.insert(0, gae + values[step])
        return returns

    def ppo_iter(states, actions, log_probs, returns, advantage):
        batch_size = states.size(0)
        # generates random mini-batches until we have covered the full batch
        for _ in range(batch_size // MINI_BATCH_SIZE):
            rand_ids = np.random.randint(0, batch_size, MINI_BATCH_SIZE)
            yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]

    def action(self, obs):
        dist, value = self.ac(obs)
        action = dist.sample()

        # _, act_v = torch.max(q_vals_v, dim=1)
        # action = int(act_v.item())
        return action

        pass

    def learn(self, current_state, action, reward, done, new_state):
        pass

    def update(self, ep, writer):
        pass

    def ppo_update(frame_idx, states, actions, log_probs, returns, advantages, clip_param=cfg.PPO_EPSILON):
        # PPO EPOCHS is the number of times we will go through ALL the training data to make updates
        for _ in range(cfg.PPO_EPOCHS):
            # grabs random mini-batches several times until we have covered all data
            for state, action, old_log_probs, return_, advantage in ppo_iter(states, actions, log_probs, returns, advantages):
                dist, value = model(state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

                actor_loss  = - torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()

                loss = cfg.PPO_CRITIC_DISCOUNT * critic_loss + actor_loss - cfg.PPO_ENTROPY_BETA * entropy

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size):
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
        )

        # self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        
    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor(x)
        # std   = self.log_std.exp().expand_as(mu)
        # dist  = Normal(mu, std)
        # return dist, value

        return Categorical(mu), value


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, device):
        super(ActorNetwork, self).__init__()

        self.actor = nn.Sequential(
                nn.Linear(input_dims, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, n_actions),
                nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=cfg.LEARNING_RATE)
        self.device = device
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        return dist

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, device):
        super(CriticNetwork, self).__init__()

        self.critic = nn.Sequential(
                nn.Linear(input_dims, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=cfg.LEARNING_RATE)
        self.device = device
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value