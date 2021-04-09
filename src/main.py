import os
import random
from abc import ABC

from src.data_synthesis import create_context_vector, do_binary_vectors, create_data_from_training_file
from src.evaluator import IPSEvaluator, DoublyRobustEstimator
import numpy as np
from scipy.stats import bernoulli
from src.online_simulation import Simulator
from src.policy import RandomPolicy, DeterministicPolicy
from sklearn.linear_model import LinearRegression
# path = create_context_vector()
# context = do_binary_vectors('data/track2/simple_context', 20)
# data = create_triples_from_context_vectors(context)


actions = np.arange(100)
contexts = np.random.randint(2, size=(50, 10))
log_policy = RandomPolicy(actions)
eval_policy = DeterministicPolicy(actions)
simulator = Simulator(contexts, actions, log_policy, eval_policy)
simulator.simulate(10000)


def ips():
    ips_eval = IPSEvaluator(log_policy, eval_policy)
    print(ips_eval.evaluate_policy())
    print(eval_policy.mean_reward())


def dr():
    X = np.array(list(map(lambda x: np.append(x[0], [x[1]]), log_policy.history)))
    y = np.array(list(map(lambda x: x[2], log_policy.history)))
    reg = LinearRegression()
    reg.fit(X, y)
    dr_eval = DoublyRobustEstimator(log_policy, eval_policy, reg)
    print(dr_eval.evaluate_policy())
    print(eval_policy.mean_reward())