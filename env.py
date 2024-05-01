import pandas as pd
import numpy as np
feats = ['AAPL', 'MSFT', 'AMZN']
class Env:
    def __init__(self, df):
        self.df = df
        self.n = len(df)
        self.current_idx = 0
        self.action_space = [0, 1, 2]  # Actions: BUY, SELL, HOLD
        self.is_invested = 0
        self.states = self.df[feats].to_numpy()
        self.rewards = self.df['SPY'].to_numpy()

    def reset(self):
        self.current_idx = 0
        return self.states[self.current_idx]

    def step(self, action):
        if self.current_idx >= self.n - 1:
            raise IndexError("No more data available. Reset the environment.")

        self._process_action(action)
        self.current_idx += 1

        reward = self.rewards[self.current_idx] if self.is_invested else 0
        next_state = self.states[self.current_idx]
        done = self.current_idx == self.n - 1

        return next_state, reward, done

    def _process_action(self, action):
        if action == 0:  # BUY
            self.is_invested = 1
        elif action == 1:  # SELL
            self.is_invested = 0