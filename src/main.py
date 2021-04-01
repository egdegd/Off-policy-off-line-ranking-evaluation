import os
import random
from src.data_synthesis import create_triples_from_context_vectors, create_context_vector, do_binary_vectors
from src.evaluator import IPSEvaluator

# path = create_context_vector()
context = do_binary_vectors('data/track2/simple_context', 20)
data = create_triples_from_context_vectors(context)


def pi(a, x):
    for (x_i, a_i, r_i) in data:
        if x_i == x:
            return int(a == a_i)
    return 0


def mu(a, x):
    pass


def policy(x):
    return random.uniform(0, 1)


ips_eval = IPSEvaluator(policy, data, mu, pi)
