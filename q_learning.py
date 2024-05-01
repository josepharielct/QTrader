import numpy as np

class Agent:
    def __init__(self, action_size, state_mapper):
        self.action_size = action_size
        self.gamma = 0.8  
        self.epsilon = 0.1  
        self.learning_rate = 0.1
        self.state_mapper = state_mapper
        self.Q = self._initialize_q_table()

    def _initialize_q_table(self):
        Q = {}
        for state in self.state_mapper.all_possible_states():
            for action in range(self.action_size):
                Q[(tuple(state), action)] = np.random.randn()
        return Q

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)

        state = tuple(self.state_mapper.transform(state))
        action_values = [self.Q[(state, a)] for a in range(self.action_size)]
        return np.argmax(action_values)  # Choose the action with the highest Q-value

    def train(self, state, action, reward, next_state, done):
        state = tuple(self.state_mapper.transform(state))
        next_state = tuple(self.state_mapper.transform(next_state))

        if done:
            target = reward  
        else:
            future_rewards = [self.Q[(next_state, a)] for a in range(self.action_size)]
            target = reward + self.gamma * np.max(future_rewards)  

        self.Q[(state, action)] += self.learning_rate * (target - self.Q[(state, action)])