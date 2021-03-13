from datetime import datetime
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Local files
import Settings as cfg
from Utils import Logger
from Agent import Agent
from Environment import Environment

import pandas as pd
import torch
from tensorboardX import SummaryWriter

print()

# Torch device (CUDA if available)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("DEVICE =", device.type)
print()

run_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
cfg.RUN_FOLDER = "runs/runs_all_{}/".format(run_time)

for df in cfg.ALL_DATASETS:
	cfg.DATASET_NAME = df
	cfg.TRAIN_DATA = "data/{}_2000-2019.csv".format(df)
	cfg.VAL_DATA = "data/{}_2020.csv".format(df)

	# Loading datasets
	train_data = pd.read_csv(cfg.TRAIN_DATA)
	val_data = pd.read_csv(cfg.VAL_DATA)

	# Creating environment
	env = Environment(train_data, val_data)
	env.reset()

	######################## Setting up the algorithm and assigning to the agent ########################
	## DQN
	if cfg._DQN_ == 1:
		from models.DQN import DQN
		dqn = DQN(env.obs.shape[0], len(env.actions()), device)
		if (cfg.LOAD_MODEL):
			dqn.load_checkpoint()
		agent = Agent(env, dqn, cfg.DQN)
		
	## AC
	if cfg._AC_ == 1:
		from models.AC import AC
		ac = AC(env.obs.shape[0], len(env.actions()), device)
		if (cfg.LOAD_MODEL):
			ac.load_checkpoint()
		agent = Agent(env, ac, cfg.AC)

	# PPO
	if cfg._PPO_ == 1:
		from models.PPO import PPO
		ppo = PPO(env.obs.shape[0], len(env.actions()), device)
		if (cfg.LOAD_MODEL):
			ppo.load_checkpoint()
		agent = Agent(env, ppo, cfg.PPO)
	#####################################################################################################
	if cfg.ATTENTION_LAYER:
		cfg.RUN_NAME = "attention_{}_{}".format(cfg.ALGO_NAME, df)
	else:
		cfg.RUN_NAME = "{}_{}".format(cfg.ALGO_NAME, df)

	# Tensorboard
	run_name = '{}_LR={}_RDTrain={}_RDVal={}_EP{}d_EAS={}_RAPC={}_'.format(cfg.RUN_NAME, cfg.LEARNING_RATE, cfg.RANDOM_START_TRAINING_DATE, cfg.RANDOM_START_VAL_DATE , cfg.EPISODE_LENGTH, cfg.END_AFTER_SELL, cfg.REWARD_AFTER_PRICE_CHANGE)
	date_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
	# filename_path = cfg.RUN_FOLDER + run_name + date_time
	writer = SummaryWriter(cfg.RUN_FOLDER + run_name + date_time)
	# writer = SummaryWriter(filename_path)

	# Logger
	logger = Logger(cfg.RUN_FOLDER + run_name + date_time)
	logger.print_run_info()
	logger.print_algo_info(agent.algo)

	# Reward and profit tracker
	eps_mean_rewards = []
	eps_profit_or_loss = []

	# Evaluation runs counter
	eval_i = 0
	eval_acc_profit = 0

	# Main training loop
	for i in range(cfg.MAX_EPISODES):
		done = False
		agent.env.evalRun = False
		agent.env.reset()
		rewards = []
		profits = []

		# Episode loop
		while not done:
			if cfg._DQN_:
				epsilon = max(cfg.DQN_EPSILON_FINAL, cfg.DQN_EPSILON_START - i / cfg.DQN_EPSILON_DECAY_LAST_FRAME)
				action = agent.choose_action(epsilon, device=device)
			else:
				action = agent.choose_action(None, device=device)

			current_state = agent.env.obs.flatten()
			new_state, reward, done, action_performed, profit = agent.env.step(action,agent.buyPrice,agent.active)

			agent.learn(current_state, action_performed, reward, done, new_state)
			if cfg._DQN_:
				agent.update(i, writer)
			
			rewards.append(reward)
			profits.append(profit)
		# End of episode

		if cfg._PPO_ or cfg._AC_:
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

			if cfg._DQN_:
				if cfg.TENSORBOARD_SAVE:
					writer.add_scalar('epsilon', epsilon, i)
				# print("Episode", i, "ended. Mean reward=", mean_reward, "| Mean profit=", mean_profit, "| Total profit/loss =", total_profit_or_loss, "| epsilon =", epsilon)
				# print("Episode {} ended. Mean reward = {} | Mean profit = {} | Total profit/loss = {} | epsilon = {}".format(i, mean_reward, mean_profit, total_profit_or_loss, epsilon))
				logger.print_out("Episode {} ended. Mean reward = {} | Mean profit = {} | Total profit/loss = {} | epsilon = {}".format(i, mean_reward, mean_profit, total_profit_or_loss, round(epsilon, 3)))

			else:
				# print("Episode", i, "ended. Mean reward=", mean_reward, "| Mean profit=", mean_profit, "| Total profit/loss =", total_profit_or_loss)
				# print("Episode {} ended. Mean reward = {} | Mean profit = {} | Total profit/loss = {} | epsilon = {}".format(i, mean_reward, mean_profit, total_profit_or_loss))
				logger.print_out("Episode {} ended. Mean reward = {} | Mean profit = {} | Total profit/loss = {}".format(i, mean_reward, mean_profit, total_profit_or_loss))

			if cfg.TENSORBOARD_SAVE:
				# print("here")
				writer.add_scalar('mean_reward', mean_reward, i)
				writer.add_scalar('mean_profit', mean_profit, i)
				writer.add_scalar('total_profit_or_loss', total_profit_or_loss, i)

			acc_reward = []
			eps_profit_or_loss = []
			eps_mean_rewards = []

		# Save checkpoint (every 10% of max episodes)
		if i % cfg.CHECKPOINT_STEP == 0:
			if cfg.SAVE_CHECKPOINTS:
				# agent.algo.save_checkpoint(run_name + date_time, i)
				agent.algo.save_checkpoint(cfg.RUN_FOLDER + run_name + date_time, i)

		# Evaluation run (1 episode)
		if i % cfg.EVALUATE_EVERY_N_EPISODES == 0:
			agent.env.evalRun = True
			agent.env.reset()
			done = False
			eval_rewards = []
			eval_profits = []
			actions = []
			start_date = agent.env.currentValDate

			while not done:
				if cfg._DQN_:
					action = agent.choose_action(0, device=device)
				else:
					action = agent.choose_action(None, device=device)

				new_state, reward, done, action_performed, profit = agent.env.step(action,agent.buyPrice,agent.active)
				agent.val_update(new_state, done)
				actions.append(action_performed)
				eval_rewards.append(reward)
				eval_profits.append(profit)
				eval_acc_profit = round(eval_acc_profit + profit, 2)

			eval_mean_reward = round(sum(eval_rewards)/len(eval_rewards),4)
			eval_total_profit = round(sum(eval_profits), 4)

			if cfg.TENSORBOARD_SAVE:
				writer.add_scalar('eval_mean_reward', eval_mean_reward, eval_i)
				writer.add_scalar('eval_profit', eval_total_profit, eval_i)
				writer.add_scalar('eval_acc_profit', eval_acc_profit, eval_i)
			# print("Validation episode", eval_i, "ended. Mean reward =", eval_mean_reward, "| Total profit =", eval_total_profit, "(Start date =", start_date, ")")
			# print("Validation episode {} ended. Mean reward = {} | Total profit = {}".format(eval_i, eval_mean_reward, eval_total_profit))
			logger.print_out("Validation episode {} ended. Mean reward = {} | Total profit = {} | Acc profit = {}".format(eval_i, eval_mean_reward, eval_total_profit, eval_acc_profit))

			# print("Actions: {} (Start date = {})".format(actions, start_date))
			logger.print_out("Actions: {} (Start date = {})".format(actions, start_date))

			# rounded_obs = [round(x,2) for x in agent.env.obs.flatten()]
			# logger.print_out("obs: {}".format(rounded_obs))
			# logger.print_out("attention_probs: {}".format(agent.env.attention_probs))
			# logger.print_out("attention_obs: {}".format(agent.env.attention_obs))

			logger.print_out("")
			# print()

			eval_i += 1
		# End eval
	
	logger.close()
	os.system('clear')
	agent.clear_memory()
