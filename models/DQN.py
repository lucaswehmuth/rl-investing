import math
import numpy as np
import collections
import os

import Settings as cfg

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

def soft_update(local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(cfg.TAU*local_param.data + (1.0-cfg.TAU)*target_param.data)

def calc_loss_diff(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(states).float().to(device)
    next_states_v = torch.tensor(next_states).float().to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * cfg.GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)

def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch

    # print("states=", states)
    # print(type(states))
    # print(type(states[0][0]))

    states_v = torch.tensor(states).float().to(device)
    next_states_v = torch.tensor(next_states).float().to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    # print()
    # print("actions_v=", actions_v)
    # print(type(actions_v))
    # print("states_v=", states_v)
    # print(type(states_v))
    # print()

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    # state_action_values = net(states_v).gather(1, actions_v.unsqueeze(0)).squeeze(-1)
    # state_action_values = net(states_v).gather(1, actions_v)
    next_state_actions = net(next_states_v).max(1)[1]

    next_state_values = tgt_net(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
    next_state_values[done_mask] = 0.0

    expected_state_action_values = next_state_values.detach() * cfg.GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)

class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)

class DQN():
    def __init__(self, obs_len, actions_n, device):
        self.qnet_local = TestDQN(obs_len, actions_n).to(device)
        self.qnet_target = TestDQN(obs_len, actions_n).to(device)
        print(self.qnet_local)
        print(self.qnet_target)
        self.optimizer = optim.Adam(self.qnet_local.parameters(), lr=cfg.LEARNING_RATE)
        self.exp_buffer = ExperienceBuffer(cfg.EXPERIENCE_BUFFER_SIZE)
        self.device = device

    def save_checkpoint(self, name, i):
        checkpoint = {'qnet_local_dict': self.qnet_local.state_dict(),
                    'qnet_target_dict': self.qnet_target.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                        }

        path_str = "checkpoints/"+name+"/"
        os.makedirs(path_str, exist_ok=True)
        check_name = name+"_"+str(i)+".pth"
        torch.save(checkpoint, path_str+check_name)

    def load_checkpoint(self):
        checkpoint = torch.load(cfg.MODEL_TO_LOAD)
        self.qnet_local.load_state_dict(checkpoint['qnet_local_dict'])
        self.qnet_target.load_state_dict(checkpoint['qnet_target_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def action(self, obs):
        q_vals_v = self.qnet_local(obs)
        _, act_v = torch.max(q_vals_v, dim=1)
        action = int(act_v.item())
        return action

    def learn(self, current_state, action, reward, done, new_state):
        exp = Experience(current_state, action, reward, done, new_state)
        self.exp_buffer.append(exp)

    def update(self, ep, writer):
        if len(self.exp_buffer) >= cfg.EXPERIENCE_START_SIZE:
        # Update target network with current weights
            # Soft update
            if cfg.SOFT_UPDATE:
                soft_update(self.qnet_local, self.qnet_target)

                # Hard update
            elif ep % cfg.SYNC_TARGET == 0:
                self.qnet_target.load_state_dict(self.qnet_local.state_dict())

            # At this point we can proceed with training in random batches from the buffer

            # Zero-ing gradients before backpropagation
            self.optimizer.zero_grad()

            # Sample batch from buffer
            batch = self.exp_buffer.sample(cfg.BATCH_SIZE)

            # Calculating loss
            loss_t = calc_loss(batch, self.qnet_local, self.qnet_target, device=self.device)

            # Backpropagate the loss
            loss_t.backward()

            # Perform opt step
            self.optimizer.step()

            # if len(self.exp_buffer) >= cfg.EXPERIENCE_START_SIZE:
            # if ep % cfg.TENSORBOARD_UPDATE == 0 and cfg.TENSORBOARD_SAVE == True:
                # writer.add_scalar('training_loss', loss_t.item(), ep)

class SimpleFFDQN(nn.Module):
    def __init__(self, obs_len, actions_n):
        super(SimpleFFDQN, self).__init__()

        self.fc_val = nn.Sequential(
            nn.Linear(obs_len, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(obs_len, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, actions_n)
        )

    def forward(self, x):
        val = self.fc_val(x)
        adv = self.fc_adv(x)
        # print("val=", val)
        # print("adv=",adv)
        result = val + adv - adv.mean(dim=1, keepdim=True)
        # print("result=",result)
        return result
        # return val + adv - adv.mean(dim=1, keepdim=True)
        # return val + adv - adv.mean()

class TestDQN(nn.Module):
    def __init__(self, obs_len, actions_n):
        super(TestDQN, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(obs_len, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, actions_n)
        )

        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.softmax(x)
        return x
        