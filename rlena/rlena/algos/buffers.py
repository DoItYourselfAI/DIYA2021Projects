import numpy as np
import pprint
import random

from collections import deque
from rl2.buffers.base import ReplayBuffer


class EpisodicBuffer(ReplayBuffer):
    def __init__(self, size=1, n_env=1, elements=None, max_episodes=30):
        if elements is None:
            elements = [
                'loc',
                'add',
                'hidden',
                'actions',
                'rewards',
                'dones',
                'values',
                'g_loc',
                'g_add',
            ]
        self.episodes = deque(maxlen=max_episodes)
        self.curr_episode = 0
        self.max_episodes = max_episodes
        super().__init__(size, elements)

    def __call__(self, idx=None):
        if idx is None:
            return list(self.episodes) + self.to_dict()
        else:
            return self.episodes[idx]

    def __repr__(self) -> str:
        if self.curr_episode == 1:
            out = ''
            for key, value in self.to_dict().items():
                if isinstance(value, np.ndarray):
                    value = value.shape
                out += key + '\n' + pprint.pformat(value) + '\n'
            return out
        else:
            return pprint.pformat(self.episodes)

    def reset(self):
        self.curr_episode += 1
        if self.curr_episode > 1:
            self.episodes.append(self.to_dict())
        super().reset()

    def push(self, *args):
        kwargs = dict(zip(self.keys, args))
        super().push(**kwargs)

    def sample(self, *args):
        out = []
        if len(args) == 0:
            args = self.keys
        for key in args:
            values = [d[key] for d in self.episodes]
            out.append(np.vstack(values))

        return out



# Priority Experience Memory
class PriorityBuffer:  # stored as ( s, a, r, s_ , done mask) in SumTree

    def __init__(self, args):
        self.capacity = int(args['capacity'])
        self.tree = SumTree(self.capacity)
        self.beta = args['per_beta']
        self.beta_increment_per_sampling = args['per_beta_inc']
        self.e = args['per_epsilon']
        self.a = args['per_alpha']
        self.n_step = args['n_step']
        

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        
        batch_state, batch_gru, batch_global_s, batch_action, batch_reward, batch_done_mask, batch_state_p = [],[],[],[],[],[],[]
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            state, gru, global_s, action, reward, done_mask, state_p = data
            priorities.append(p)
            batch_state.append(state)
            batch_gru.append(gru)
            batch_global_s.append(global_s)
            batch_action.append(action)
            batch_reward.append(reward)
            batch_done_mask.append(done_mask)
            batch_state_p.append(state_p)
            idxs.append(idx)
        batch = [batch_state, batch_gru, batch_global_s, batch_action, batch_reward, batch_done_mask, batch_state_p]
        batch = list(map(np.stack, batch))
        
        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch

    def update(self, idxs, errors):
        for idx, error in zip(idxs, errors):
            p = self._get_priority(error)
            self.tree.update(idx, p)

    def size(self):
        return self.tree.n_entries


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])
