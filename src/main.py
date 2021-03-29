import os

from src.data_synthesis import create_data
from src.evaluator import IPSEvaluator

data = create_data('data/track2/training.txt')


def pi(a, x):
    for (x_i, a_i, r_i) in data:
        if x_i == x:
            return int(a == a_i)
    return 0


def mu(a, x):
    pass


def policy(x):
    pass


ips_eval = IPSEvaluator(policy, data, mu, pi)
