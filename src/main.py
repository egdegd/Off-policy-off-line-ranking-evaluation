from src.evaluator import IPSEvaluator, DoublyRobustEstimator, ModelBasedEstimator
import numpy as np
from src.online_simulation import ParallelSimulator, Simulator
from src.policy import RandomPolicy, DeterministicPolicy, CBVowpalWabbit
from sklearn.linear_model import LinearRegression, LogisticRegression
import matplotlib.pyplot as plt


# path = create_context_vector()
# context = do_binary_vectors('data/track2/simple_context', 20)
# data = create_triples_from_context_vectors(context)


def parallel_stimulation():
    actions = np.arange(10)
    contexts = np.random.randint(2, size=(20, 5))
    log_policy = RandomPolicy(actions)
    eval_policy = DeterministicPolicy(actions)
    simulator = ParallelSimulator(contexts, actions, log_policy, eval_policy)
    simulator.simulate(100)
    return log_policy, eval_policy


def simulation(PolicyClass, *args):
    actions = np.arange(10)
    contexts = np.random.randint(2, size=(20, 5))
    policy = PolicyClass(actions, *args)
    simulator = Simulator(contexts, actions, policy)
    simulator.simulate(100)
    return policy


def ips(log_policy, eval_policy):
    ips_eval = IPSEvaluator(log_policy, eval_policy)
    print(ips_eval.evaluate_policy(), eval_policy.mean_reward())
    return ips_eval.evaluate_policy(), eval_policy.mean_reward()


def dr(log_policy, eval_policy):
    X = np.array(list(map(lambda x: np.append(x[0], [x[1]]), log_policy.history)))
    y = np.array(list(map(lambda x: x[2], log_policy.history)))
    reg = LogisticRegression()
    reg.fit(X, y)
    dr_eval = DoublyRobustEstimator(log_policy, eval_policy, reg)
    print(dr_eval.evaluate_policy(), eval_policy.mean_reward())
    return eval_policy.mean_reward(), dr_eval.evaluate_policy()


def mb(log_policy, eval_policy):
    X = np.array(list(map(lambda x: np.append(x[0], [x[1]]), log_policy.history)))
    y = np.array(list(map(lambda x: x[2], log_policy.history)))
    reg = LogisticRegression()
    reg.fit(X, y)
    dr_eval = ModelBasedEstimator(log_policy, eval_policy, reg)
    print(dr_eval.evaluate_policy(), eval_policy.mean_reward())
    return eval_policy.mean_reward(), dr_eval.evaluate_policy()


log_policy_ = simulation(RandomPolicy)
for _ in range(30):
    eval_policy_ = simulation(CBVowpalWabbit, log_policy_.history)
    dr(log_policy_, eval_policy_)

# log_policy_, eval_policy_ = parallel_stimulation()
# ips(log_policy_, eval_policy_)
