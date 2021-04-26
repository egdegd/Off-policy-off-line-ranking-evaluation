import random
import numpy as np
from vowpalwabbit import pyvw


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

    def train(self, data):
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

    def train(self, data):
        pass


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

    def train(self, data):
        pass


class CBVowpalWabbit(BasePolicy):
    def __init__(self, actions):
        super().__init__()
        self.actions = actions
        self.number_of_action = len(actions)
        self.vw = pyvw.vw(f"--quiet --cb_explore {self.number_of_action}")
        self.history = []

    def train(self, train_data):
        for (x, a, r) in train_data:
            feature_string = self.create_string(x)
            learn_example = str(a + 1) + ':' + str(r) + ':' + str(
                round(1 / self.number_of_action, 3)) + ' | ' + feature_string
            self.vw.learn(learn_example)

    def create_string(self, x):
        feature = list(map(lambda t: chr(t + 97), list(x)))
        return ' '.join(feature)

    def add_info(self, x, a, r):
        self.history.append((x, a, r))

    def give_a(self, x):
        feature_string = self.create_string(x)
        probabilities = np.array(self.vw.predict('| ' + feature_string))
        return probabilities.argmax()

    def give_probability(self, x, a):
        feature_string = self.create_string(x)
        probabilities = self.vw.predict('| ' + feature_string)
        return probabilities[a]

    def mean_reward(self):
        return sum(map(lambda x: x[2], self.history)) / len(self.history)