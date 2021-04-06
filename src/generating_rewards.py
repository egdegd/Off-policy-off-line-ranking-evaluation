import numpy as np
from scipy.stats import bernoulli


def update_rewards(weights, q):
    number_of_action, dim_state = weights.shape
    q_vec = bernoulli.rvs(q, size=number_of_action)
    cov = np.zeros((dim_state, dim_state))
    for i in range(dim_state):
        cov[i][i] = 1
    addition = np.random.multivariate_normal(mean=np.zeros(dim_state), cov=cov, size=number_of_action)
    new_weights = weights + np.multiply(q_vec.reshape((number_of_action, 1)), addition)
    return new_weights


def init_weights(number_of_action, dim_state):
    cov = np.zeros((dim_state, dim_state))
    for i in range(dim_state):
        cov[i][i] = 1
    weights = np.random.multivariate_normal(mean=np.zeros(dim_state), cov=cov, size=number_of_action)
    return weights

