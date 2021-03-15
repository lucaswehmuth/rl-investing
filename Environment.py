import Settings as cfg
from models.SimpleAttention import SimpleAttention
import random
random.seed(999)

class Environment():
	def __init__(self, data, val_data):
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

		self.attention = SimpleAttention(self.obs.shape[0])
		self.attention_probs = [1] * self.obs.shape[0]
		self.attention_obs = self.obs.flatten()

	def actions(self):
		return [cfg.ACTION_HOLD, cfg.ACTION_BUY, cfg.ACTION_SELL]

	def random_action(self):
		actions = self.actions()
		return actions[random.randint(0,2)]

	def isEnd(self):
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
		else:
			self.currentValDate += 1

		self.episode_steps += 1
		done = self.isEnd()
		action_performed = action
		profit = 0

		if action == cfg.ACTION_BUY and active == 0:
			reward = cfg.REWARD_BUY
			price = self.obs.currentClosePrice

			if self.evalRun == False:
				new_state = State(self.data, self.currentDate, price, 1)
			else:
				new_state = State(self.val_data, self.currentValDate, price, 1)

		elif action == cfg.ACTION_SELL and active == 1:
			profit = self.obs.currentClosePrice - buyPrice

			reward = cfg.REWARD_SELL_MULTIPLIER * (self.obs.currentClosePrice - buyPrice) / buyPrice
			
			if cfg.END_AFTER_SELL:
				done = True
			
			if self.evalRun == False:
				new_state = State(self.data, self.currentDate, 0, 0)
			else:
				new_state = State(self.val_data, self.currentValDate, 0, 0)

		elif action == cfg.ACTION_HOLD and active == 1:
			reward = 0

			if cfg.REWARD_AFTER_PRICE_CHANGE:
				reward = cfg.REWARD_HOLD_ACTIVE_MULTIPLIER * (self.obs.currentClosePrice - self.obs.previousDayClosePrice) / self.obs.previousDayClosePrice

			if self.evalRun == False:
				new_state = State(self.data, self.currentDate, buyPrice, active)
			else:
				new_state = State(self.val_data, self.currentValDate, buyPrice, active)

		elif action == cfg.ACTION_HOLD and active == 0:
			# Price went up and we did not buy => Small penalty
			if (self.obs.currentClosePrice - self.obs.previousDayClosePrice) > 0:
				reward = cfg.REWARD_HOLD_INACTIVE_PRICE_UP
			# Price went down and we did not buy
			else:
				reward = cfg.REWARD_HOLD_INACTIVE_PRICE_DOWN

			if self.evalRun == False:
				new_state = State(self.data, self.currentDate, buyPrice, active)
			else:
				new_state = State(self.val_data, self.currentValDate, buyPrice, active)

		# Invalid action
		else:
			reward = cfg.REWARD_INVALID

			if self.evalRun == False:
				new_state = State(self.data, self.currentDate, buyPrice, active)
			else:
				new_state = State(self.val_data, self.currentValDate, buyPrice, active)
			# action_performed = cfg.ACTION_HOLD

		if cfg.ATTENTION_LAYER:
			new_current_price = new_state.currentClosePrice
			attention_probs = self.attention.fit(self.obs.flatten(), new_current_price)
			attention_probs = attention_probs.detach().numpy()[0]
			attention_probs[-1] = 1
			attention_probs[-2] = 1
			self.attention_probs = [round(x, 4) for x in attention_probs]
			self.attention_obs = [round(a*b, 4) for a,b in zip(new_state.flatten(), attention_probs)]
			self.obs = new_state
			return self.attention_obs, reward, done, action_performed, profit
		else:
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
		# if self.volumes:
			# return (4 * cfg.STATE_N_DAYS + 1 + 1, )
		# else:
		return (3 * cfg.STATE_N_DAYS + 1 + 1 + 1, )

	def fetchData(self, df, currentDate):
		start = currentDate - cfg.STATE_N_DAYS
		openPrices = df.open[start:currentDate].to_list()
		closePrices = df.close[start:currentDate].to_list()
		lowPrices = df.low[start:currentDate].to_list()
		highPrices = df.high[start:currentDate].to_list()
		# volumes = df.volume[start:currentDate].to_list()

		if cfg.NORMALIZED_STATE_PRICES == False:
			closePrices = [round(x, 2) for x in closePrices]
			lowPrices = [round(x, 2) for x in lowPrices]
			highPrices = [round(x, 2) for x in highPrices]
			return closePrices, lowPrices, highPrices
		else:
			rel_close, rel_low, rel_high = [], [], []
			for i in range(cfg.STATE_N_DAYS):
				rel_close.append(round((closePrices[i] - openPrices[i]) / openPrices[i], 4))
				rel_low.append(round((lowPrices[i] - openPrices[i]) / openPrices[i], 4))
				rel_high.append(round((highPrices[i] - openPrices[i]) / openPrices[i], 4))
			# return rel_close, rel_low, rel_high, volumes
			return rel_close, rel_low, rel_high

	def flatten(self):
		return self.closePrices + self.lowPrices + self.highPrices + [self.currentClosePrice, self.hasStock, self.buyPrice]