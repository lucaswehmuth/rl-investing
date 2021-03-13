import Settings as cfg
import numpy as np
import torch
torch.manual_seed(999)

class Agent():
	def __init__(self, env, algo, name):
		self.active = 0
		self.buyPrice = 0
		self.env = env
		self.algo = algo
		self.algo_name = name
		# self.done = False

	def choose_action(self, epsilon, device):
		# prob = 0
		if self.algo_name == cfg.DQN:
			if np.random.random() < epsilon:
				action = self.env.random_action()
			else:
				# state_a = np.array([self.env.obs.flatten()], dtype=np.float32, copy=False)
				if cfg.ATTENTION_LAYER:
					state_a = np.array([self.env.attention_obs], dtype=np.float32, copy=False)
				else:
					state_a = np.array([self.env.obs.flatten()], dtype=np.float32, copy=False)
				state = torch.tensor(state_a).to(device)
				action = self.algo.action(state)

		# elif self.algo_name == cfg.PPO:
		# 	state_a = np.array([self.env.obs.flatten()], dtype=np.float32, copy=False)
		# 	state = torch.tensor(state_a).to(device)
		# 	# action, prob = self.algo.action(state)
		# 	action = self.algo.action(state)

		else:
			if cfg.ATTENTION_LAYER:
				state_a = np.array([self.env.attention_obs], dtype=np.float32, copy=False)
			else:
				state_a = np.array([self.env.obs.flatten()], dtype=np.float32, copy=False)

			state = torch.tensor(state_a).to(device)
			action = self.algo.action(state)

		return action

	def learn(self, current_state, action, reward, done, new_state):
		if done == True:
			self.buyPrice = 0
			self.active = 0
		else:
			self.buyPrice = new_state[-1]
			self.active = new_state[-2]
		# self.done = done
		self.algo.learn(current_state, action, reward, done, new_state)

	def val_update(self, new_state, done):
		if done == True:
			self.buyPrice = 0
			self.active = 0
		else:
			self.buyPrice = new_state[-1]
			self.active = new_state[-2]

	def update(self, ep, writer):
		self.algo.update(ep, writer)

	def clear_memory(self):
		torch.cuda.empty_cache()