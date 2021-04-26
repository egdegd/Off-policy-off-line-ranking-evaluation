import numpy as np
from sklearn.linear_model import LogisticRegression

from src.policy import BasePolicy


class IPSEvaluator:
    def __init__(self, log_policy: BasePolicy, eval_policy: BasePolicy):
        self.log_policy = log_policy
        self.eval_policy = eval_policy

    def evaluate_one_reward(self, x, a, r):
        return r * self.eval_policy.give_probability(x, a) / self.log_policy.give_probability(x, a)

    def evaluate_policy(self):
        expected_rewards = []
        for (x, a, r) in self.log_policy.history:
            expected_rewards.append(self.evaluate_one_reward(x, a, r))
        return np.array(expected_rewards).mean()

    def train(self, data):
        pass


class DoublyRobustEstimator:

    def __init__(self, log_policy: BasePolicy, eval_policy: BasePolicy):
        self.log_policy = log_policy
        self.eval_policy = eval_policy
        self.model_based_estimator = None

    def train(self, data):
        X = np.array(list(map(lambda x: np.append(x[0], [x[1]]), data)))
        y = np.array(list(map(lambda x: x[2], data)))
        self.model_based_estimator = LogisticRegression()
        self.model_based_estimator.fit(X, y)

    def evaluate_one_reward(self, x, a, r):
        mb = self.model_based_estimator.predict([np.append(x, [self.eval_policy.give_a(x)])])[0]  # it can be round
        return mb + (r - mb) * self.eval_policy.give_probability(x, a) / self.log_policy.give_probability(x, a)

    def evaluate_policy(self):
        expected_rewards = []
        for (x, a, r) in self.log_policy.history:
            expected_rewards.append(self.evaluate_one_reward(x, a, r))
        return np.array(expected_rewards).mean()


class ModelBasedEstimator:

    def __init__(self, log_policy: BasePolicy, eval_policy: BasePolicy):
        self.log_policy = log_policy
        self.eval_policy = eval_policy
        self.model_based_estimator = None

    def train(self, data):
        X = np.array(list(map(lambda x: np.append(x[0], [x[1]]), data)))
        y = np.array(list(map(lambda x: x[2], data)))
        self.model_based_estimator = LogisticRegression()
        self.model_based_estimator.fit(X, y)

    def evaluate_one_reward(self, x, a, r):
        mb = self.model_based_estimator.predict([np.append(x, [self.eval_policy.give_a(x)])])[0]  # it can be round
        return mb

    def evaluate_policy(self):
        expected_rewards = []
        for (x, a, r) in self.log_policy.history:
            # mb = self.evaluate_one_reward(x, a, r)
            # print(self.eval_policy.history[i][2], mb)
            # expected_rewards.append(mb)
            expected_rewards.append(self.evaluate_one_reward(x, a, r))
        return np.array(expected_rewards).mean()
