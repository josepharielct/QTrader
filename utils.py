import yfinance as yf
import pandas as pd
import numpy as np

def get_sp500_daily_returns(start_date = '2010-01-01', end_date = '2021-12-31'):
  #Referenced from this gist: https://gist.github.com/quantra-go-algo/ac5180bf164a7894f70969fa563627b2
  a = np.zeros(1)
  tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist() + ['SPY']
  df_price = yf.download(tickers, start_date, end_date, auto_adjust=True)['Close'] \
      .dropna(axis=0, how='all') \
      .dropna(axis=1, how='any')
  returns = {}
  for name in df_price.columns:
    returns[name] = np.log(df_price[name]).diff()
  df_returns = pd.DataFrame(data=returns)
  return df_price, df_returns

def tt_split(df, n_test = 1000):
  train_df = df.iloc[:-n_test]
  test_df = df.iloc[-n_test:]
  return train_df, test_df

def play_one_episode(agent, env, is_train):
  state = env.reset()
  done = False
  total_reward = 0

  while not done:
    action = agent.act(state)
    next_state, reward, done = env.step(action)
    total_reward += reward
    if is_train:
      agent.train(state, action, reward, next_state, done)
    state = next_state

  return total_reward