import Settings as cfg
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions import MultivariateNormal

class PPOMemory:
    def __init__(self):
        # Batch data
        self.batch_obs = []
        self.batch_acts = []
        self.batch_log_probs = []
        self.batch_rews = []
        self.batch_rtgs = []
        # batch_lens = []
        self.batch_lens = 0

        self.action_history = []

        self.ep_rewards = []
        # self.batch_size = batch_size

    def clearMemory(self):
        # print(self.action_history)
        # self.action_history = []

        self.batch_obs = []
        self.batch_acts = []
        self.batch_log_probs = []
        self.batch_rews = []
        self.batch_rtgs = []
        self.batch_lens = 0

        self.ep_rewards = []

class PPO():
    def __init__(self, obs_len, actions_n, device):
        self.memory = PPOMemory()
        self.actor = FeedForwardNN(obs_len, actions_n).to(device)
        self.critic = FeedForwardNN(obs_len, 1).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.PPO_LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.PPO_LEARNING_RATE)

        self.device = device
        
        self.cov_var = torch.full(size=(actions_n,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

    def save_checkpoint(self, name, i):
        pass

    def load_checkpoint(self):
        pass

    # def action(self, obs):
    #     dist, value = self.ac(obs)
    #     action = dist.sample()

    #     # _, act_v = torch.max(q_vals_v, dim=1)
    #     # action = int(act_v.item())
    #     return action
    #     pass

    def action(self, obs):
        # Query the actor network for a mean action
        mean = self.actor(obs)

        # Create a distribution with the mean action and std from the covariance matrix above.
        # For more information on how this distribution works, check out Andrew Ng's lecture on it:
        # https://www.youtube.com/watch?v=JjB58InuTqM
        dist = MultivariateNormal(mean, self.cov_mat)
        # dist = Categorical(mean)

        # Sample an action from the distribution
        action = dist.sample()

        # Calculate the log probability for that action
        log_prob = dist.log_prob(action)

        # Save to memory
        # Store action and log_prob temporarily
        self.last_action = action.detach().numpy()[0]
        self.last_log_probs = log_prob.detach()
        # self.memory.batch_log_probs.append(log_prob.detach())
        # self.memory.batch_acts.append(action.detach().numpy())

        # Return the sampled action and the log probability of that action in our distribution
        # return action.detach().numpy(), log_prob.detach()
        # print("action=",action.detach().numpy())

        # action = max(action.detach().numpy()[0])
        # print("action=",action)

        _, act_v = torch.max(action, dim=1)
        action = int(act_v.item())
        # self.memory.action_history.append(action)
        # print("action=",action)

        return action
        # return action.detach().numpy()

    def compute_rtgs(self):
        """
            Compute the Reward-To-Go of each timestep in a batch given the rewards.
            Parameters:
                batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)
            Return:
                batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
        """
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []

        # print("len(self.memory.batch_rews)=", len(self.memory.batch_rews))

        # Iterate through each episode
        for ep_rews in reversed(self.memory.batch_rews):

            discounted_reward = 0 # The discounted reward so far

            # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
            # discounted return (think about why it would be harder starting from the beginning)
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * cfg.PPO_GAMMA
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        # print("len(batch_rtgs)=", len(batch_rtgs))
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

    def evaluate(self, batch_obs, batch_acts):
        """
            Estimate the values of each observation, and the log probs of
            each action in the most recent batch with the most recent
            iteration of the actor network. Should be called from learn.
            Parameters:
                batch_obs - the observations from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of observation)
                batch_acts - the actions from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of action)
            Return:
                V - the predicted values of batch_obs
                log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
        """
        # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most recent actor network.
        # This segment of code is similar to that in get_action()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        # print("len(log_probs) =", log_probs)

        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return V, log_probs

    def learn(self, current_state, action, reward, done, new_state):
        # batch_obs = []
        # batch_acts = []
        # batch_log_probs = []
        # batch_rews = []
        # batch_rtgs = []
        # batch_lens = []

        self.memory.batch_obs.append(current_state)
        self.memory.batch_log_probs.append(self.last_log_probs)
        self.memory.batch_acts.append(self.last_action)
        # self.memory.batch_acts.append(action)
        # self.memory.batch_rews.append(reward)
        self.memory.batch_lens += 1

        if done == True:
            self.memory.ep_rewards.append(reward)
            self.memory.batch_rews.append(self.memory.ep_rewards)
            self.memory.ep_rewards = []
            # print(self.memory.caction_history)
        else:
            self.memory.ep_rewards.append(reward)

    def update(self, ep, writer):
        # print("self.memory.batch_lens =", self.memory.batch_lens)
        if self.memory.batch_lens >= cfg.PPO_STEPS_PER_BATCH:
            # print("updated")
            batch_rtgs = self.compute_rtgs()
            batch_obs = torch.tensor(self.memory.batch_obs, dtype=torch.float)
            batch_acts = torch.tensor(self.memory.batch_acts, dtype=torch.float)

            # print("self.memory.batch_lens =", self.memory.batch_lens)            
            # print("len(self.memory.batch_log_probs)=", len(self.memory.batch_log_probs))
            batch_log_probs = torch.tensor(self.memory.batch_log_probs, dtype=torch.float)

            # Calculate advantage at k-th iteration
            V, _ = self.evaluate(batch_obs, batch_acts)
            A_k = batch_rtgs - V.detach()

            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # This is the loop where we update our network for some n epochs
            for _ in range(cfg.PPO_N_UPDATES_PER_ITERATION):                                                       # ALG STEP 6 & 7
                # Calculate V_phi and pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
                # NOTE: we just subtract the logs, which is the same as
                # dividing the values and then canceling the log with e^log.
                # For why we use log probabilities instead of actual probabilities,
                # here's a great explanation: 
                # https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
                # TL;DR makes gradient ascent easier behind the scenes.
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses.
                surr1 = ratios * A_k
                # print("surr1 =", surr1)
                surr2 = torch.clamp(ratios, 1 - cfg.PPO_CLIP, 1 + cfg.PPO_CLIP) * A_k
                # print("surr2 =", surr2)

                # Calculate actor and critic losses.
                # NOTE: we take the negative min of the surrogate losses because we're trying to maximize
                # the performance function, but Adam minimizes the loss. So minimizing the negative
                # performance function maximizes it.
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                # Calculate gradients and perform backward propagation for actor network
                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                # actor_loss.backward()
                self.actor_optimizer.step()

                # Calculate gradients and perform backward propagation for critic network
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

            self.memory.clearMemory()

    # def ppo_update(frame_idx, states, actions, log_probs, returns, advantages, clip_param=cfg.PPO_EPSILON):
    #     pass

class FeedForwardNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeedForwardNN, self).__init__()

        # self.layer1 = nn.Linear(in_dim, 256)
        # self.layer2 = nn.Linear(256, 256)
        # self.layer3 = nn.Linear(256, out_dim)

        self.NN = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, out_dim),
        )

    def forward(self, obs):
        """
            Runs a forward pass on the neural network.
            Parameters:
                obs - observation to pass as input
            Return:
                output - the output of our forward pass
        """
        # Convert observation to tensor if it's a numpy array
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        # activation1 = F.relu(self.layer1(obs))
        # activation2 = F.relu(self.layer2(activation1))
        # output = self.layer3(activation2)

        # output = Categorical(output)

        output = self.NN(obs)

        return output
