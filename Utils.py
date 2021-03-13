import logging
import Settings as cfg
from Environment import Environment
import os
import inspect

class Logger():
	def __init__(self, filename_path):
		filename = filename_path.split('/')[1]

		print("Storing logs at: ")
		print(".../{}/{}.log".format(filename_path, filename))
		print()

		logging.basicConfig(filename='{}/{}.log'.format(filename_path, filename), level=logging.INFO, format='%(message)s')

		self.logger = logging.getLogger()  # Logger
		logger_handler = logging.StreamHandler()  # Handler for the logger
		logger_handler.setFormatter(logging.Formatter('%(message)s'))
		self.logger.addHandler(logger_handler)

	def print_run_info(self):
		self.print_out("************* STEP FUNCTION *************")
		self.print_out(inspect.getsource(Environment.step))
		self.print_out("************* RUN INFO *************")
		self.print_out("DATASET_NAME = {}".format(cfg.DATASET_NAME))
		self.print_out("MAX_EPISODES = {}".format(cfg.MAX_EPISODES))
		self.print_out("LEARNING_RATE = {}".format(cfg.LEARNING_RATE))
		self.print_out("TENSORBOARD_SAVE = {}".format(cfg.TENSORBOARD_SAVE))
		self.print_out("TENSORBOARD_UPDATE = {}".format(cfg.TENSORBOARD_UPDATE))
		self.print_out("EVALUATE_EVERY_N_EPISODES = {}".format(cfg.EVALUATE_EVERY_N_EPISODES))
		self.print_out("SAVE_CHECKPOINTS = {}".format(cfg.SAVE_CHECKPOINTS))
		self.print_out("CHECKPOINT_STEP = {}".format(cfg.CHECKPOINT_STEP))
		self.print_out("LOAD_MODEL = {}".format(cfg.LOAD_MODEL))
		self.print_out("STATE_N_DAYS = {}".format(cfg.STATE_N_DAYS))
		self.print_out("RANDOM_START_TRAINING_DATE = {}".format(cfg.RANDOM_START_TRAINING_DATE))
		self.print_out("RANDOM_START_VAL_DATE = {}".format(cfg.RANDOM_START_VAL_DATE))
		self.print_out("END_AFTER_SELL = {}".format(cfg.END_AFTER_SELL))
		self.print_out("EPISODE_LENGTH = {}".format(cfg.EPISODE_LENGTH))
		
		self.print_out("REWARD_AFTER_PRICE_CHANGE = {}".format(cfg.REWARD_AFTER_PRICE_CHANGE))
		self.print_out("REWARD_BUY = {}".format(cfg.REWARD_BUY))
		self.print_out("REWARD_SELL_MULTIPLIER = {}".format(cfg.REWARD_SELL_MULTIPLIER))
		self.print_out("REWARD_HOLD_ACTIVE_MULTIPLIER = {}".format(cfg.REWARD_HOLD_ACTIVE_MULTIPLIER))
		# self.print_out("REWARD_HOLD_ACTIVE_POSITIVE_CHANGE = {}".format(cfg.REWARD_HOLD_ACTIVE_POSITIVE_CHANGE))
		# self.print_out("REWARD_HOLD_ACTIVE_NEGATIVE_CHANGE = {}".format(cfg.REWARD_HOLD_ACTIVE_NEGATIVE_CHANGE))
		self.print_out("REWARD_HOLD_INACTIVE_PRICE_UP = {}".format(cfg.REWARD_HOLD_INACTIVE_PRICE_UP))
		self.print_out("REWARD_HOLD_INACTIVE_PRICE_DOWN = {}".format(cfg.REWARD_HOLD_INACTIVE_PRICE_DOWN))
		self.print_out("REWARD_INVALID = {}".format(cfg.REWARD_INVALID))
		self.print_out("")
		self.print_out("ALGO_NAME = {}".format(cfg.ALGO_NAME))
		self.print_out("ATTENTION_LAYER = {}".format(cfg.ATTENTION_LAYER))
		
		if cfg.ALGO_NAME == cfg.DQN:
			self.print_out("")
			self.print_out("DQN Settings:")
			self.print_out("\tDQN_GAMMA = {}".format(cfg.DQN_GAMMA))
			self.print_out("\tDQN_TAU = {}".format(cfg.DQN_TAU))
			self.print_out("\tDQN_BATCH_SIZE = {}".format(cfg.DQN_BATCH_SIZE))
			self.print_out("\tDQN_EXPERIENCE_BUFFER_SIZE = {}".format(cfg.DQN_EXPERIENCE_BUFFER_SIZE))
			self.print_out("\tDQN_EXPERIENCE_START_SIZE = {}".format(cfg.DQN_EXPERIENCE_START_SIZE))
			self.print_out("\tDQN_LEARNING_RATE = {}".format(cfg.DQN_LEARNING_RATE))
			self.print_out("\tDQN_SOFT_UPDATE = {}".format(cfg.DQN_SOFT_UPDATE))
			self.print_out("\tDQN_SYNC_TARGET = {}".format(cfg.DQN_SYNC_TARGET))
			self.print_out("\tDQN_EPSILON_DECAY_LAST_FRAME = {}".format(cfg.DQN_EPSILON_DECAY_LAST_FRAME))
			self.print_out("\tDQN_EPSILON_START = {}".format(cfg.DQN_EPSILON_START))
			self.print_out("\tDQN_EPSILON_FINAL = {}".format(cfg.DQN_EPSILON_FINAL))

		elif cfg.ALGO_NAME == cfg.AC:
			pass

		elif cfg.ALGO_NAME == cfg.PPO:
			self.print_out("PPO Settings:")
			self.print_out("\tPPO_GAMMA = {}".format(cfg.PPO_GAMMA))
			self.print_out("\tPPO_CLIP = {}".format(cfg.PPO_CLIP))
			self.print_out("\tPPO_STEPS_PER_BATCH = {}".format(cfg.PPO_STEPS_PER_BATCH))
			self.print_out("\tPPO_N_UPDATES_PER_ITERATION = {}".format(cfg.PPO_N_UPDATES_PER_ITERATION))

		self.print_out("************************************")
		
	def print_out(self, string):
		if cfg.LOG_OUTPUTS:
			self.logger.info(string)
		else:
			print(string)

	def print_algo_info(self, algo):
		info = algo.get_nn_info()

		print("Neural network:")
		print()
		for x in info:
			self.print_out(x)

		print()
		self.print_out("************************************")

	def close(self):
		# self.logger.handlers.clear()
		handlers = self.logger.handlers[:]
		for handler in handlers:
			handler.close()
			self.logger.removeHandler(handler)
		os.system('clear')