import os
import random
from abc import ABC

from src.data_synthesis import create_context_vector, do_binary_vectors, create_data_from_training_file
from src.evaluator import IPSEvaluator
import numpy as np
from scipy.stats import bernoulli
from src.online_simulation import Simulator
from src.policy import RandomPolicy, DeterministicPolicy

# path = create_context_vector()
# context = do_binary_vectors('data/track2/simple_context', 20)
# data = create_triples_from_context_vectors(context)


actions = np.arange(1000)
contexts = np.random.randint(2, size=(500, 100))
log_policy = RandomPolicy(actions)
eval_policy = RandomPolicy(actions)
simulator = Simulator(contexts, actions, log_policy, eval_policy)
simulator.simulate(1000)


ips_eval = IPSEvaluator(log_policy, eval_policy)
print(ips_eval.evaluate_policy())
print(eval_policy.mean_reward())
