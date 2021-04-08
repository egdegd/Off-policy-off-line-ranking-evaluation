import random


class BasePolicy:
    def give_a(self, x):
        raise NotImplementedError

    def add_info(self, x, a, r):
        raise NotImplementedError


class RandomPolicy(BasePolicy):

    def __init__(self, actions):
        self.actions = actions
        self.number_of_action = len(actions)
        self.history = []

    def give_a(self, x):
        i = random.randint(0, self.number_of_action)
        return self.actions[i]

    def add_info(self, x, a, r):
        self.history.append((x, a, r))
