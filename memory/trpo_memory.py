import numpy as np


class TRPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs_ = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def store_memory(self, state, action, probs_, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs_.append(probs_)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs_ = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def sample_batch(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]
        return np.array(self.states), np.array(self.actions), np.array(self.probs_), np.array(self.probs), \
            np.array(self.vals), np.array(self.rewards), np.array(self.dones), batches
