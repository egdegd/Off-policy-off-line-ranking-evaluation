import random
import numpy as np
from scipy.stats import bernoulli


class Simulator:
    def __init__(self, contexts, actions, pi, mu, q=0.5):
        self.contexts = contexts
        self.actions = actions
        self.weights = None
        self.q = q
        self.pi = pi(actions)
        self.mu = mu(actions)
        self.number_of_action = len(actions)
        self.dim_state = len(contexts)  # number of contexts
        self.init_weights()

    def compute_reward(self, a, x):
        return 1 / (1 + np.exp(- self.weights[a].T @ x))

    def init_weights(self):
        cov = np.zeros((self.dim_state, self.dim_state))
        for i in range(self.dim_state):
            cov[i][i] = 1
        self.weights = np.random.multivariate_normal(mean=np.zeros(self.dim_state), cov=cov, size=self.number_of_action)

    def update_reward(self):
        q_vec = bernoulli.rvs(self.q, size=self.number_of_action)
        cov = np.zeros((self.dim_state, self.dim_state))  # todo: create method for cov
        for i in range(self.dim_state):
            cov[i][i] = 1
        addition = np.random.multivariate_normal(mean=np.zeros(self.dim_state), cov=cov, size=self.number_of_action)
        self.weights += np.multiply(q_vec.reshape((self.number_of_action, 1)), addition)

    def simulate(self, T):
        for t in range(T):
            i = random.randint(0, self.dim_state)
            x = self.contexts[i]
            a_log = self.pi.give_a(x)
            r_log = self.compute_reward(a_log, x)
            a_eval = self.mu.give_a(x)
            r_eval = self.compute_reward(a_eval, x)
            self.pi.add_info(x, a_log, r_log)
            self.mu.add_info(a_eval, r_eval)
            self.update_reward()
