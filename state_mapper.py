import numpy as np
import itertools

class StateMapper:
    def __init__(self, env, n_bins=6, n_samples=10000):
        self.n_bins = n_bins
        self.n_samples = n_samples
        self.D = None
        self.bins = []

        self._sample_states_and_create_bins(env)

    def _sample_states_and_create_bins(self, env):
        states = self._collect_state_samples(env)
        self.D = len(states[0])  
        states = np.array(states)  

        for d in range(self.D):
            self.bins.append(self._create_bins_for_dimension(states[:, d]))

    def _collect_state_samples(self, env):
        states = []
        s = env.reset()
        states.append(s)
        while len(states) < self.n_samples:
            action = np.random.choice(env.action_space)
            s, _, done = env.step(action)
            states.append(s)
            if done:
                s = env.reset()
                states.append(s)
        return states

    def _create_bins_for_dimension(self, data):
        sorted_data = np.sort(data)
        return [
            sorted_data[int(self.n_samples / self.n_bins * (k + 0.5))]
            for k in range(self.n_bins)
        ]

    def transform(self, state):
        x = np.zeros(self.D, dtype=int)
        for d in range(self.D):
            x[d] = np.digitize(state[d], self.bins[d])
        return tuple(x)

    def all_possible_states(self):
        list_of_bins = [list(range(len(bin) + 1)) for bin in self.bins]
        return itertools.product(*list_of_bins)