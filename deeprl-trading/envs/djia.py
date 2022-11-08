from envs.base import Environment
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class DJIA(Environment):
    def __init__(self, args=None):
        self.args = args

        # check dates
        start_train = datetime.strptime(args.start_train, "%Y-%M-%d")
        start_val = datetime.strptime(args.start_val, "%Y-%M-%d")
        start_test = datetime.strptime(args.start_test, "%Y-%M-%d")
        assert start_train < start_val, "the start of training must be earlier than the validation"
        assert start_val < start_test, "the start of validation must be earlier than the test"

        # predefined constants
        # modified from https://github.com/AI4Finance-Foundation/FinRL
        self._balance_scale = 1e-4
        self._price_scale = 1e-2
        self._reward_scale = 1e-4
        self._max_stocks = 100
        self._min_action = int(0.1 * self._max_stocks)

        # load stock data
        prices, tickers = [], []
        root = os.path.join(args.data_dir, 'dow30')
        for filename in os.listdir(root):
            if filename.endswith('.csv'):
                tickers.append(filename[:-4])
                path = os.path.join(root, filename)
                df = pd.read_csv(path, index_col='Date')
                prices.append(df.Close)
        prices = pd.concat(prices, axis=1)
        prices.index = pd.to_datetime(prices.index)
        prices.columns = tickers
        prices.sort_index(axis=0, inplace=True)
        self.all_prices = prices

        # default to training
        self.train()

        # initialize environment
        _ = self.reset()

    def train(self):
        start = datetime.strptime(self.args.start_train, "%Y-%m-%d")
        end = datetime.strptime(self.args.start_val, "%Y-%m-%d")
        self.prices = self.all_prices[start:end - timedelta(days=1)]

    def eval(self):
        start = datetime.strptime(self.args.start_val, "%Y-%m-%d")
        end = datetime.strptime(self.args.start_test, "%Y-%m-%d")
        self.prices = self.all_prices[start:end - timedelta(days=1)]

    def test(self):
        start = datetime.strptime(self.args.start_test, "%Y-%m-%d")
        self.prices = self.all_prices[start:]

    @property
    def observation_space(self):
        return (61,)

    @property
    def action_space(self):
        # actions are assumed to be constrained to [-1.0, 1.0]
        return (30,)

    def reset(self):
        self.head = 0
        self.balance = self.args.initial_balance
        self.holdings = np.zeros(30)
        self.total_asset = self.balance
        self.total_reward = 0.0

        p = self.prices.iloc[self.head].values * self._price_scale
        h = self.holdings * self._price_scale
        b = max(self.balance, 1e4)  # cutoff value defined in FinRL
        b *= np.ones(1) * self._balance_scale
        return np.concatenate([p, h, b], axis=0)

    def step(self, action):
        # rescale actions
        action = (action * self._max_stocks).astype(int)

        # update prices and holdings
        self.head += 1
        if self.head >= len(self.prices):
            raise KeyError("environment must be reset")

        prices = self.prices.iloc[self.head].values
        tc = self.args.transaction_cost
        # sells
        for idx in np.where(action < -self._min_action)[0]:
            shares = min(-action[idx], self.holdings[idx])
            self.holdings[idx] -= shares
            self.balance += prices[idx] * shares * (1 - tc)
        # buys
        for idx in np.where(action > self._min_action)[0]:
            shares = min(action[idx], self.balance // prices[idx])
            self.holdings[idx] += shares
            self.balance -= prices[idx] * shares * (1 + tc)

        # calculate asset gains
        total_asset = self.balance + (prices * self.holdings).sum()
        reward = (total_asset - self.total_asset) * self._reward_scale
        self.total_asset = total_asset
        self.total_reward = self.args.gamma * self.total_reward + reward

        # check if at terminal state
        if self.head == len(self.prices) - 1:
            reward = self.total_reward
            profit = self.total_asset / self.args.initial_balance - 1.0
            state = self.reset()
            return state, reward, True, {'profit': profit}

        # create state vector
        p = prices * self._price_scale
        h = self.holdings * self._price_scale
        b = max(self.balance, 1e4)  # cutoff value defined in FinRL
        b *= np.ones(1) * self._balance_scale
        state = np.concatenate([p, h, b], axis=0)
        return state, reward, False, {}


class DJIANew(DJIA):
    def __init__(self, args=None):
        super().__init__(args)
        # predefined constants
        self._action_scale = 10.0
        self._reward_scale = 100.0

    @property
    def observation_space(self):
        return (61, 25)

    @property
    def action_space(self):
        return (31,)

    def reset(self):
        self.head = 26
        self.balance = self.args.initial_balance
        self.holdings = np.zeros(30)
        self.weights = np.zeros((25, 31))
        self.total_asset = self.balance

        pct_change = self.prices.iloc[self.head - 26:self.head].pct_change()
        state = np.concatenate([
            pct_change.dropna().values.T,
            self.weights.T,
        ])
        return state

    def step(self, action):
        # assume that action is clipped to [-1.0, 1.0]
        action *= self._action_scale

        # update prices and holdings
        self.head += 1
        if self.head >= len(self.prices):
            raise KeyError("environment must be reset")

        prev_prices = self.prices.iloc[self.head - 1].values
        exp = np.exp(action - action.max())
        weights = (exp / exp.sum())[1:]
        holdings = self.total_asset * weights // prev_prices

        prices = self.prices.iloc[self.head].values
        tc = self.args.transaction_cost
        # sells
        for idx in np.where(holdings < self.holdings)[0]:
            shares = self.holdings[idx] - holdings[idx]
            self.holdings[idx] -= shares
            self.balance += prices[idx] * shares * (1 - tc)
        # buys
        for idx in np.where(holdings > self.holdings)[0]:
            shares = holdings[idx] - self.holdings[idx]
            shares = min(shares, self.balance // (prices[idx] * (1 + tc)))
            self.holdings[idx] += shares
            self.balance -= prices[idx] * shares * (1 + tc)

        # calculate asset gains
        total_asset = self.balance + (prices * self.holdings).sum()
        reward = (total_asset - self.total_asset) / (self.total_asset + 1e-8)
        reward *= self._reward_scale
        self.total_asset = total_asset
        profit = self.total_asset / self.args.initial_balance - 1.0

        # check if at terminal state
        if self.head == len(self.prices) - 1:
            state = self.reset()
            return state, reward, True, {'profit': profit, "return": reward / self._reward_scale}

        # create state vector
        weights = np.concatenate([
            np.array([self.balance]),
            prices * self.holdings
        ]) / self.total_asset
        self.weights = np.concatenate([
            self.weights[1:],
            weights.reshape(1, -1)
        ])
        pct_change = self.prices.iloc[self.head - 26:self.head].pct_change()
        state = np.concatenate([
            pct_change.dropna().values.T,
            self.weights.T,
        ])
        return state, reward, False, {'profit': profit, "return": reward / self._reward_scale}
