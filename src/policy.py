import random
import numpy as np


class BasePolicy:

    def __init__(self):
        self.history = None

    def give_a(self, x):
        raise NotImplementedError

    def add_info(self, x, a, r):
        raise NotImplementedError

    def give_probability(self, x, a):
        raise NotImplementedError

    def mean_reward(self):
        raise NotImplementedError


class RandomPolicy(BasePolicy):

    def __init__(self, actions):
        super().__init__()
        self.actions = actions
        self.number_of_action = len(actions)
        self.history = []

    def give_a(self, x):
        i = random.randint(0, self.number_of_action - 1)
        return self.actions[i]

    def add_info(self, x, a, r):
        self.history.append((x, a, r))

    def give_probability(self, x, a):
        return 1 / self.number_of_action

    def mean_reward(self):
        return sum(map(lambda x: x[2], self.history)) / len(self.history)


class DeterministicPolicy(BasePolicy):

    def __init__(self, actions):
        super().__init__()
        self.actions = actions
        self.number_of_action = len(actions)
        self.history = []

    def give_a(self, x):
        i = random.randint(0, self.number_of_action - 1)
        return self.actions[i]

    def add_info(self, x, a, r):
        self.history.append((x, a, r))

    def give_probability(self, x, a):
        for (x_, a_, r_) in self.history:
            if np.array_equal(x_, x):
                return int(a_ == a)
        return 0

    def mean_reward(self):
        return sum(map(lambda x: x[2], self.history)) / len(self.history)
