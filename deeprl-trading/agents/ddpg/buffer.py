import numpy as np


class ReplayBuffer:
    def __init__(self, state_shape, action_shape, size=50000):
        # initialize buffer entries
        self.state = np.empty((size, *state_shape)).astype(np.float32)
        self.action = np.empty((size, *action_shape)).astype(np.float32)
        self.reward = np.empty((size,)).astype(np.float32)
        self.done = np.empty((size,)).astype(np.bool)
        self.next_state = np.empty((size, *state_shape)).astype(np.float32)

        # initialize head
        self.max_size = size
        self.size = 0
        self.head = 0

    def push(self, *transition):
        state, action, reward, done, next_state = transition

        # store current transition at head
        self.state[self.head] = state
        self.action[self.head] = action
        self.reward[self.head] = reward
        self.done[self.head] = done
        self.next_state[self.head] = next_state

        # increment size and head
        self.size = min(self.size + 1, self.max_size)
        self.head = (self.head + 1) % self.max_size

    def sample(self, batch_size):
        idx = np.random.randint(self.size, size=batch_size)
        return (
            self.state[idx],
            self.action[idx],
            self.reward[idx],
            self.done[idx],
            self.next_state[idx],
        )
