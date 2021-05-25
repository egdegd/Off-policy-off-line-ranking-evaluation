from matplotlib import pyplot as plt

from src.evaluator import IPSEvaluator, DoublyRobustEstimator, ModelBasedEstimator
import numpy as np
from src.online_simulation import TwoStageSimulator
from src.policy import RandomPolicy, DeterministicPolicy, CBVowpalWabbit
from sklearn.linear_model import LinearRegression, LogisticRegression


def simulation(EstimatorClass, EvalPolicyClass):
    # actions = np.arange(5)
    # contexts = np.random.randint(2, size=(1, 1))
    actions = np.arange(50)
    contexts = np.random.randint(2, size=(20, 20))

    log_policy = RandomPolicy(actions)
    simulator = TwoStageSimulator(contexts, actions, log_policy)

    train_history = simulator.simulate_train(100)

    eval_policy = EvalPolicyClass(actions)
    eval_policy.train(train_history)

    simulator.simulate(100, eval_policy)

    estimator = EstimatorClass(log_policy, eval_policy)
    estimator.train(train_history)

    # print(eval_policy.mean_reward(), estimator.evaluate_policy())
    return eval_policy.mean_reward(), estimator.evaluate_policy()
