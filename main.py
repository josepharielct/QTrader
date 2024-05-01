from env import Env
from utils import get_sp500_daily_returns, tt_split, play_one_episode
from state_mapper import StateMapper
from q_learning import Agent
import numpy as np
import matplotlib.pyplot as plt

df_price,df_returns = get_sp500_daily_returns()
train_data, test_data = tt_split(df_returns)

num_episodes = 500
train_env = Env(train_data)
test_env = Env(test_data)
action_size = len(train_env.action_space)
state_mapper = StateMapper(train_env)
agent = Agent(action_size, state_mapper)
train_rewards = np.empty(num_episodes)
test_rewards = np.empty(num_episodes)
for e in range(num_episodes):
  r = play_one_episode(agent, train_env, is_train=True)
  train_rewards[e] = r

  # test on the test set
  tmp_epsilon = agent.epsilon
  agent.epsilon = 0.
  tr = play_one_episode(agent, test_env, is_train=False)
  agent.epsilon = tmp_epsilon
  test_rewards[e] = tr

  print(f"eps: {e + 1}/{num_episodes}, train: {r:.5f}, test: {tr:.5f}")

plt.figure(figsize=(10, 5))
plt.plot(train_rewards, label='Train Rewards')
plt.plot(test_rewards, label='Test Rewards', linestyle='--')
plt.title('Training vs. Testing Rewards Over Episodes')
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)
plt.show()