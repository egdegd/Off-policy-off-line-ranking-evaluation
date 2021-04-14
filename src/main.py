from src.evaluator import IPSEvaluator, DoublyRobustEstimator, ModelBasedEstimator
import numpy as np
from src.online_simulation import ParallelSimulator, Simulator
from src.policy import RandomPolicy, DeterministicPolicy, CBVowpalWabbit
from sklearn.linear_model import LinearRegression, LogisticRegression


# path = create_context_vector()
# context = do_binary_vectors('data/track2/simple_context', 20)
# data = create_triples_from_context_vectors(context)


def parallel_stimulation(LogPolicyClass, EvalPolicyClass):
    actions = np.arange(10)
    contexts = np.random.randint(2, size=(20, 5))
    log_policy = LogPolicyClass(actions)
    eval_policy = EvalPolicyClass(actions)
    simulator = ParallelSimulator(contexts, actions, log_policy, eval_policy)
    simulator.simulate(1000)
    return log_policy, eval_policy


def simulation(PolicyClass, *args):
    actions = np.arange(10)
    contexts = np.random.randint(2, size=(20, 5))
    policy = PolicyClass(actions, *args)
    simulator = Simulator(contexts, actions, policy)
    simulator.simulate(1000)
    return policy


def ips(log_policy, eval_policy):
    ips_eval = IPSEvaluator(log_policy, eval_policy)
    print(eval_policy.mean_reward(), ips_eval.evaluate_policy())
    return eval_policy.mean_reward(), ips_eval.evaluate_policy()


def dr(log_policy, eval_policy):
    X = np.array(list(map(lambda x: np.append(x[0], [x[1]]), log_policy.history)))
    y = np.array(list(map(lambda x: x[2], log_policy.history)))
    reg = LogisticRegression()
    reg.fit(X, y)
    dr_eval = DoublyRobustEstimator(log_policy, eval_policy, reg)
    print(eval_policy.mean_reward(), dr_eval.evaluate_policy(), )
    return eval_policy.mean_reward(), dr_eval.evaluate_policy()


def mb(log_policy, eval_policy):
    X = np.array(list(map(lambda x: np.append(x[0], [x[1]]), log_policy.history)))
    y = np.array(list(map(lambda x: x[2], log_policy.history)))
    reg = LogisticRegression()
    reg.fit(X, y)
    mb_eval = ModelBasedEstimator(log_policy, eval_policy, reg)
    print(eval_policy.mean_reward(), mb_eval.evaluate_policy())
    return eval_policy.mean_reward(), mb_eval.evaluate_policy()

