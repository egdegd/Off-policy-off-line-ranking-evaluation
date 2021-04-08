import numpy as np

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


class DoublyRobustEstimator:

    def __init__(self, policy, history, model_based_estimator, mu, pi):
        self.policy = policy
        self.history = history
        self.model_based_estimator = model_based_estimator(history)
        self.mu = mu
        self.pi = pi

    def get_action_and_reward(self, x_i):
        for (x, a, r) in self.history:
            if x == x_i:
                return a, r

    def evaluate_one_reward(self, x, a, real_a, real_r):
        mb = self.model_based_estimator(x, a)
        return mb + (real_r - mb) * self.pi(a, x) / self.mu(real_a, x)

    def evaluate_policy(self):
        rewards = []
        for (x, a, r) in self.history:
            rewards.append(self.evaluate_one_reward(x, self.policy(x), a, r))
        return np.array(rewards).mean()
