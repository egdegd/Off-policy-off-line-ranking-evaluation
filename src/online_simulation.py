import random
import numpy as np
from scipy.stats import bernoulli


class TwoStageSimulator:
    def __init__(self, contexts, actions, log_policy, q=0.5):
        self.contexts = contexts
        self.actions = actions
        self.weights = None
        self.q = q
        self.log_policy = log_policy
        self.number_of_action = len(actions)
        self.number_of_context, self.dim_state = contexts.shape
        self.init_weights()
        self.random_rewards = np.random.rand(1, len(self.actions))[0]

    def compute_reward(self, a, x):
        # return np.random.choice([0, 1], p=[self.random_rewards[a], 1 - self.random_rewards[a]])
        return 1 / (1 + np.exp(- self.weights[a].T @ x))

    def create_cov_matrix(self):
        cov = np.zeros((self.dim_state, self.dim_state))
        for i in range(self.dim_state):
            cov[i][i] = 1
        return cov

    def init_weights(self):
        cov = self.create_cov_matrix()
        self.weights = np.random.multivariate_normal(mean=np.zeros(self.dim_state), cov=cov, size=self.number_of_action)

    def update_reward(self):
        q_vec = bernoulli.rvs(self.q, size=self.number_of_action)
        cov = self.create_cov_matrix()
        addition = np.random.multivariate_normal(mean=np.zeros(self.dim_state), cov=cov, size=self.number_of_action)
        self.weights += np.multiply(q_vec.reshape((self.number_of_action, 1)), addition)

    def simulate_train(self, T):
        history_train = []
        for t in range(T):
            i = random.randint(0, self.number_of_context - 1)
            x = self.contexts[i]
            a = self.log_policy.give_a(x)
            r = self.compute_reward(a, x)
            history_train.append((x, a, round(r)))
            self.update_reward()
        return history_train

    def simulate(self, T, eval_policy):
        for t in range(T):
            i = random.randint(0, self.number_of_context - 1)
            x = self.contexts[i]
            a_log = self.log_policy.give_a(x)
            r_log = self.compute_reward(a_log, x)
            a_eval = eval_policy.give_a(x)
            r_eval = self.compute_reward(a_eval, x)
            self.log_policy.add_info(x, a_log, round(r_log))
            eval_policy.add_info(x, a_eval, round(r_eval))
            self.update_reward()
