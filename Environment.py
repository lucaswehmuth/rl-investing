import Settings as cfg
import random

class Environment():
	def __init__(self, data, val_data):
		# self.train_data = train_data
		self.val_data = val_data
		self.data = data
		# self.currentDate = random.randint(cfg.STATE_N_DAYS+1, len(self.data)-2)
		# self.currentValDate = random.randint(cfg.STATE_N_DAYS+1, len(self.val_data)-2)
		self.evalRun = False

		# PPO
		# self.currentDate = random.randint(cfg.STATE_N_DAYS+1, len(self.data)-32)
		# self.currentValDate = random.randint(cfg.STATE_N_DAYS+1, len(self.val_data)-32)
		if cfg.RANDOM_START_TRAINING_DATE:
			self.currentDate = random.randint(cfg.STATE_N_DAYS+1, len(self.data) - cfg.EPISODE_LENGTH - 2)
		else:
			self.currentDate = int(cfg.STATE_N_DAYS+1)

		if cfg.RANDOM_START_VAL_DATE:
			self.currentValDate = random.randint(cfg.STATE_N_DAYS+1, len(self.val_data) - cfg.EPISODE_LENGTH - 2)
		else:
			self.currentValDate = int(cfg.EPISODE_LENGTH/2)

		self.episode_steps = 0

	def reset(self):
		# First N+1 days will be taken out to feed as historical data
		# Last day will also be taken out as it will be the final state
		# self.currentDate = random.randint(cfg.STATE_N_DAYS+1, len(self.data)-2)
		# self.currentValDate = random.randint(cfg.STATE_N_DAYS+1, len(self.val_data)-2)
		# self.currentDate = random.randint(cfg.STATE_N_DAYS+1, len(self.data)-32)
		# self.currentValDate = random.randint(cfg.STATE_N_DAYS+1, len(self.val_data)-32)
		if cfg.RANDOM_START_TRAINING_DATE:
			self.currentDate = random.randint(cfg.STATE_N_DAYS+1, len(self.data) - cfg.EPISODE_LENGTH - 2)
		else:
			self.currentDate = int(cfg.STATE_N_DAYS+1)

		if cfg.RANDOM_START_VAL_DATE:
			self.currentValDate = random.randint(cfg.STATE_N_DAYS+1, len(self.val_data) - cfg.EPISODE_LENGTH - 2)
		else:
			self.currentValDate = int(cfg.EPISODE_LENGTH/2)

		self.episode_steps = 0

		if self.evalRun == False:
			self.obs = State(self.data, self.currentDate, 0, 0)
		else:
			self.obs = State(self.val_data, self.currentValDate, 0, 0)

	def actions(self):
		return [cfg.ACTION_HOLD, cfg.ACTION_BUY, cfg.ACTION_SELL]

	def random_action(self):
		actions = self.actions()
		return actions[random.randint(0,2)]

	def isEnd(self):
		# print(self.currentDate)
		if self.episode_steps == cfg.EPISODE_LENGTH:
			return True
		else:
			return False

		# if self.evalRun == False:
		# 	if self.currentDate == len(self.data)-1 or self.episode_steps == cfg.EPISODE_LENGTH:
		# 		return True
		# 	else:
		# 		return False
		# else:
		# 	if self.currentValDate == len(self.val_data)-1 or self.episode_steps == cfg.EPISODE_LENGTH:
		# 		return True
		# 	else:
		# 		return False

	def step(self,action,buyPrice,active):
		if self.evalRun == False:
			self.currentDate += 1
			self.episode_steps += 1
			done = self.isEnd()
			action_performed = action
			profit = 0

			if action == cfg.ACTION_BUY and active == 0:
				reward = cfg.REWARD_BUY
				price = self.obs.currentClosePrice

				new_state = State(self.data, self.currentDate, price, 1)

			elif action == cfg.ACTION_SELL and active == 1:
				reward = cfg.REWARD_SELL_MULTIPLIER * (self.obs.currentClosePrice - buyPrice) / buyPrice
				# if reward > 0:
				# 	reward = 1
				# else:
				# 	reward = -1
				profit = self.obs.currentClosePrice - buyPrice

				if cfg.END_AFTER_SELL:
					done = True
				
				new_state = State(self.data, self.currentDate, 0, 0)

			elif action == cfg.ACTION_HOLD and active == 1:
				reward = 0
				if cfg.REWARD_AFTER_PRICE_CHANGE:
					# reward = cfg.REWARD_HOLD_ACTIVE_MULTIPLIER * (self.obs.currentClosePrice - self.obs.previousDayClosePrice) / self.obs.previousDayClosePrice
					reward = cfg.REWARD_HOLD_ACTIVE_MULTIPLIER * (self.obs.currentClosePrice - buyPrice) / buyPrice
					# reward = (self.obs.currentClosePrice - self.obs.previousDayClosePrice) / self.obs.previousDayClosePrice

					# if reward < 0:
						# reward *= 10

					# reward = (self.obs.currentClosePrice - self.obs.previousDayClosePrice) / self.obs.previousDayClosePrice

				new_state = State(self.data, self.currentDate, buyPrice, active)

			elif action == cfg.ACTION_HOLD and active == 0:
				# Price went up and we did not buy => Small penalty
				if (self.obs.currentClosePrice - self.obs.previousDayClosePrice) > 0:
					reward = cfg.REWARD_HOLD_INACTIVE_PRICE_UP
				# Price went down and we did not buy => No reward
				else:
					reward = cfg.REWARD_HOLD_INACTIVE_PRICE_DOWN

				new_state = State(self.data, self.currentDate, buyPrice, active)

			# Invalid action
			else:
				reward = cfg.REWARD_INVALID
				# reward = -0.05 * self.episode_steps
				# reward = -100.0
				new_state = State(self.data, self.currentDate, buyPrice, active)
				# action_performed = cfg.ACTION_HOLD

			self.obs = new_state
			
			return new_state.flatten(), reward, done, action_performed, profit

		# Validation step			
		else:
			self.currentValDate += 1
			self.episode_steps += 1
			done = self.isEnd()
			action_performed = action
			profit = 0

			if action == cfg.ACTION_BUY and active == 0:
				reward = cfg.REWARD_BUY
				price = self.obs.currentClosePrice

				# new_state = State(self.data, self.currentDate, buyPrice, 1)
				new_state = State(self.val_data, self.currentValDate, price, 1)

			elif action == cfg.ACTION_SELL and active == 1:
				reward = cfg.REWARD_SELL_MULTIPLIER * (self.obs.currentClosePrice - buyPrice) / buyPrice
				# if reward > 0:
				# 	reward = 1
				# else:
				# 	reward = -1
				profit = self.obs.currentClosePrice - buyPrice

				if cfg.END_AFTER_SELL:
					done = True
				
				# new_state = State(self.data, self.currentDate, 0, 0)
				new_state = State(self.val_data, self.currentValDate, 0, 0)

			elif action == cfg.ACTION_HOLD and active == 1:
				reward = 0
				if cfg.REWARD_AFTER_PRICE_CHANGE:
					# reward = 10.0 * (self.obs.currentClosePrice - self.obs.previousDayClosePrice) / self.obs.previousDayClosePrice
					reward = cfg.REWARD_HOLD_ACTIVE_MULTIPLIER * (self.obs.currentClosePrice - buyPrice) / buyPrice
					# reward = 100.0 * (self.obs.currentClosePrice - self.obs.previousDayClosePrice) / self.obs.previousDayClosePrice
					# reward = (self.obs.currentClosePrice - self.obs.previousDayClosePrice) / self.obs.previousDayClosePrice

					# if reward < 0:
						# reward *= 10

				# new_state = State(self.data, self.currentDate, buyPrice, active)
				new_state = State(self.val_data, self.currentValDate, buyPrice, active)

			elif action == cfg.ACTION_HOLD and active == 0:
				# Price went up and we did not buy => Small penalty
				if (self.obs.currentClosePrice - self.obs.previousDayClosePrice) > 0:
					reward = cfg.REWARD_HOLD_INACTIVE_PRICE_UP

				# Price went down and we did not buy => Small reward
				else:
					reward = cfg.REWARD_HOLD_INACTIVE_PRICE_DOWN
				# reward = -0.05
				# reward = -0.01 * self.episode_steps
				# new_state = State(self.data, self.currentDate, buyPrice, active)
				new_state = State(self.val_data, self.currentValDate, buyPrice, active)

			# Invalid action
			else:
				reward = cfg.REWARD_INVALID
				# reward = -0.05 * self.episode_steps
				# reward = -100.0
				# new_state = State(self.data, self.currentDate, buyPrice, active)
				new_state = State(self.val_data, self.currentValDate, buyPrice, active)
				# action_performed = cfg.ACTION_HOLD

			self.obs = new_state
			
			return new_state.flatten(), reward, done, action_performed, profit


class State():
	def __init__(self, df, currentDate, buyPrice, hasStock):
		self.hasStock = hasStock
		
		if self.hasStock == 1:
			self.buyPrice = buyPrice
		else:
			self.buyPrice = 0
		
		# Fetching relative close, low and high prices for the past N-1 days and also the respective volumes
		# self.closePrices, self.lowPrices, self.highPrices, self.volumes = self.fetchData(df, currentDate)
		self.closePrices, self.lowPrices, self.highPrices = self.fetchData(df, currentDate)

		# Actual current price
		self.currentClosePrice = round(df.close[currentDate],2)

		# Previous day price
		self.previousDayClosePrice = round(df.close[currentDate-1],2)

	@property
	def shape(self):
		# [h, l, c] * bars + position_flag + rel_profit (since open)
		# if self.volumes:
			# return (4 * self.bars_count + 1 + 1, )
		# else:
		
		# [rel close], [rel low], [rel high] + current close, has stock, buy price
		return (3 * cfg.STATE_N_DAYS + 1 + 1 + 1, )

	def fetchData(self, df, currentDate):
		start = currentDate - cfg.STATE_N_DAYS
		openPrices = df.open[start:currentDate].to_list()
		closePrices = df.close[start:currentDate].to_list()
		lowPrices = df.low[start:currentDate].to_list()
		highPrices = df.high[start:currentDate].to_list()
		# volumes = df.volume[start:currentDate].to_list()

		rel_close, rel_low, rel_high = [], [], []
		for i in range(cfg.STATE_N_DAYS):
			rel_close.append(round((closePrices[i] - openPrices[i]) / openPrices[i], 4))
			rel_low.append(round((lowPrices[i] - openPrices[i]) / openPrices[i], 4))
			rel_high.append(round((highPrices[i] - openPrices[i]) / openPrices[i], 4))

		# return rel_close, rel_low, rel_high, volumes
		return rel_close, rel_low, rel_high

	def flatten(self):
		return self.closePrices + self.lowPrices + self.highPrices + [self.currentClosePrice, self.hasStock, self.buyPrice]

