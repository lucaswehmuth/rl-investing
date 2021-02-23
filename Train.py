# Flags for the current algo being used
_DQN_ = 0
_AC_ = 0
_PPO_ = 0

from datetime import datetime
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Local files
import Settings as cfg
from Agent import Agent
from Environment import Environment

import pandas as pd
import torch
from tensorboardX import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device.type)

# Tensorboard
# writer = SummaryWriter('runs/TestDQN_Checkpoint_test_1')
# writer = SummaryWriter('runs/TestAC_9')
# 'model_x_a={}_lr={}'.format(a, lr)

# Loading datasets
train_data = pd.read_csv(cfg.TRAIN_DATA)
val_data = pd.read_csv(cfg.VAL_DATA)

# Creating environment
env = Environment(train_data, val_data)
env.reset()

######################## Setting up the algorithm and assigning to the agent ########################
## DQN
# _DQN_ = 1
# from models.DQN import DQN
# dqn = DQN(env.obs.shape[0], len(env.actions()), device)
# if (cfg.LOAD_MODEL):
# 	dqn.load_checkpoint()
# agent = Agent(env, dqn, cfg.DQN)
run_name = '{}_{}d_EAS={}_RAPC={}'.format(cfg.RUN_NAME, cfg.EPISODE_LENGTH, cfg.END_AFTER_SELL, cfg.REWARD_AFTER_PRICE_CHANGE)
# date_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
# if cfg.TENSORBOARD_SAVE:
# 	writer = SummaryWriter('runs/' + run_name + date_time)

## AC
# _AC_ = 1
# from models.AC import AC
# ac = AC(env.obs.shape[0], len(env.actions()), device)
# agent = Agent(env, ac, cfg.AC)

# PPO
_PPO_ = 1
from models.PPO2 import PPO
ppo = PPO(env.obs.shape[0], len(env.actions()), device)
if (cfg.LOAD_MODEL):
	ppo.load_checkpoint()
agent = Agent(env, ppo, cfg.PPO)
run_name = '{}_{}d_EAS={}_RAPC={}'.format(cfg.RUN_NAME, cfg.EPISODE_LENGTH, cfg.END_AFTER_SELL, cfg.REWARD_AFTER_PRICE_CHANGE)
date_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
# if cfg.TENSORBOARD_SAVE:
writer = SummaryWriter('runs/' + run_name + date_time)
#####################################################################################################


# Reward and profit tracker
eps_mean_rewards = []
eps_profit_or_loss = []

# Evaluation runs counter
eval_i = 0

for i in range(cfg.MAX_EPISODES):
	# print("i=", i)
	done = False
	agent.env.evalRun = False
	agent.env.reset()
	rewards = []
	profits = []

	while not done:
		if _DQN_:
			epsilon = max(cfg.DQN_EPSILON_FINAL, cfg.DQN_EPSILON_START - i / cfg.DQN_EPSILON_DECAY_LAST_FRAME)
			action = agent.choose_action(epsilon, device=device)
		else:
			action = agent.choose_action(None, device=device)

		current_state = agent.env.obs.flatten()
		new_state, reward, done, action_performed, profit = agent.env.step(action,agent.buyPrice,agent.active)

		agent.learn(current_state, action_performed, reward, done, new_state)
		# agent.update(i, writer)
		
		rewards.append(reward)
		profits.append(profit)
	# Episode done

	agent.update(i, writer)

	# Store sum of profit/loss for the past episode
	eps_profit_or_loss.append(sum(profits))
	eps_mean_rewards.append(round(sum(rewards)/len(rewards), 4))

	# Update Tensorboard
	if i % cfg.TENSORBOARD_UPDATE == 0:
		# mean_reward = round(sum(acc_rewards)/len(acc_rewards),4)
		mean_reward = round(sum(eps_mean_rewards)/len(eps_mean_rewards), 4)
		mean_profit = round(sum(eps_profit_or_loss)/len(eps_profit_or_loss),4)
		total_profit_or_loss = round(sum(eps_profit_or_loss), 4)

		if _DQN_:
			if cfg.TENSORBOARD_SAVE:
				writer.add_scalar('epsilon', epsilon, i)
			print("Episode", i, "ended. Mean reward=", mean_reward, "| Mean profit=", mean_profit, "| Total profit/loss =", total_profit_or_loss, "| epsilon =", epsilon)
		else:
			print("Episode", i, "ended. Mean reward=", mean_reward, "| Mean profit=", mean_profit, "| Total profit/loss =", total_profit_or_loss,)

		if cfg.TENSORBOARD_SAVE:
			writer.add_scalar('mean reward', mean_reward, i)
			writer.add_scalar('mean profit', mean_profit, i)
			writer.add_scalar('total profit or loss', total_profit_or_loss, i)

		acc_reward = []
		eps_profit_or_loss = []
		eps_mean_rewards = []

	# Save checkpoint (every 10% of max episodes)
	if i % cfg.CHECKPOINT_STEP == 0:
		if cfg.SAVE_CHECKPOINTS:
			agent.algo.save_checkpoint(run_name + date_time, i)

	# Evaluation run (1 episode)
	if i % cfg.EVALUATE_EVERY_N_EPISODES == 0:
		agent.env.evalRun = True
		agent.env.reset()
		done = False
		eval_rewards = []
		eval_profits = []
		actions = []

		while not done:
			if _DQN_:
				action = agent.choose_action(0, device=device)
			else:
				action = agent.choose_action(None, device=device)

			new_state, reward, done, action_performed, profit = agent.env.step(action,agent.buyPrice,agent.active)
			agent.val_update(new_state, done)
			actions.append(action_performed)
			eval_rewards.append(reward)
			eval_profits.append(profit)

		eval_mean_reward = round(sum(eval_rewards)/len(eval_rewards),4)
		eval_total_profit = round(sum(eval_profits), 4)
		if cfg.TENSORBOARD_SAVE:
			writer.add_scalar('eval mean reward', eval_mean_reward, eval_i)
			writer.add_scalar('eval profit', eval_total_profit, eval_i)
		print("Validation episode", eval_i, "ended. Mean reward =", eval_mean_reward, "| Total profit =", eval_total_profit)
		print("Actions:", actions)
		print()
		eval_i += 1
	# End eval

