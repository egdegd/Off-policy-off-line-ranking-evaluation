from matplotlib import pyplot as plt

from src.evaluator import IPSEvaluator, DoublyRobustEstimator, ModelBasedEstimator
import numpy as np
import random
from src.policy import RandomPolicy, DeterministicPolicy, CBVowpalWabbit
from sklearn.linear_model import LinearRegression, LogisticRegression
from scipy.stats import bernoulli


# path = create_context_vector()
# context = do_binary_vectors('data/track2/simple_context', 20)
# data = create_triples_from_context_vectors(context)


def parallel_stimulation(LogPolicyClass, EvalPolicyClass):
    actions = np.arange(20)
    contexts = np.random.randint(2, size=(20, 100))
    log_policy = LogPolicyClass(actions)
    eval_policy = EvalPolicyClass(actions)
    simulator = ParallelSimulator(contexts, actions, log_policy, eval_policy)
    simulator.simulate(1000)
    return log_policy, eval_policy


def simulation(PolicyClass, *args):
    actions = np.arange(20)
    contexts = np.random.randint(2, size=(20, 100))
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



class Simulator:
    def __init__(self, contexts, actions, policy, q=0.5):
        self.contexts = contexts
        self.actions = actions
        self.weights = None
        self.q = q
        self.policy = policy
        self.number_of_action = len(actions)
        self.number_of_context, self.dim_state = contexts.shape
        self.init_weights()

    def compute_reward(self, a, x):
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

    def simulate(self, T):
        for t in range(T):
            i = random.randint(0, self.number_of_context - 1)
            x = self.contexts[i]
            a = self.policy.give_a(x)
            r = self.compute_reward(a, x)
            self.policy.add_info(x, a, round(r))
            self.update_reward()


class ParallelSimulator(Simulator):
    def __init__(self, contexts, actions, log_policy, eval_policy, q=0.5):
        self.contexts = contexts
        self.actions = actions
        self.weights = None
        self.q = q
        self.log_policy = log_policy
        self.eval_policy = eval_policy
        self.number_of_action = len(actions)
        self.number_of_context, self.dim_state = contexts.shape
        self.init_weights()

    def simulate(self, T):
        for t in range(T):
            i = random.randint(0, self.number_of_context - 1)
            x = self.contexts[i]
            a_log = self.log_policy.give_a(x)
            r_log = self.compute_reward(a_log, x)
            a_eval = self.eval_policy.give_a(x)
            r_eval = self.compute_reward(a_eval, x)
            self.log_policy.add_info(x, a_log, round(r_log))
            self.eval_policy.add_info(x, a_eval, round(r_eval))
            self.update_reward()
            
            
def create_plot(PolicyClass, policy_name):
    real_means_ips = []
    eval_means_ips = []
    real_means_dr = []
    eval_means_dr = []
    real_means_mb = []
    eval_means_mb = []
    for _ in range(100):
        log_policy_, eval_policy_ = parallel_stimulation(RandomPolicy, PolicyClass)
        real_mean_ips, eval_mean_ips = ips(log_policy_, eval_policy_)
        real_mean_dr, eval_mean_dr = dr(log_policy_, eval_policy_)
        real_mean_mb, eval_mean_mb = mb(log_policy_, eval_policy_)
        real_means_ips.append(real_mean_ips)
        eval_means_ips.append(eval_mean_ips)
        real_means_dr.append(real_mean_dr)
        eval_means_dr.append(eval_mean_dr)
        real_means_mb.append(real_mean_mb)
        eval_means_mb.append(eval_mean_mb)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(policy_name)

    ax1.set_title("Inverse Propensity Scoring")
    ax1.plot(real_means_ips, label="real average rewards")
    ax1.plot(eval_means_ips, label="estimated average rewards")
    ax1.legend()

    ax2.set_title("Doubly Robust")
    ax2.plot(real_means_dr, label="real average rewards")
    ax2.plot(eval_means_dr, label="estimated average rewards")
    ax2.legend()

    ax3.set_title("Model Based")
    ax3.plot(real_means_mb, label="real average rewards")
    ax3.plot(eval_means_mb, label="estimated average rewards")
    ax3.legend()
    plt.savefig("../results/" + '_'.join(policy_name.split()))
    plt.show()


# todo: combine into one function or delete copy paste

def create_plot_CBVowpalWabbit():
    real_means_ips = []
    eval_means_ips = []
    real_means_dr = []
    eval_means_dr = []
    real_means_mb = []
    eval_means_mb = []
    # log_policy_ = simulation(RandomPolicy) # I'm not sure if I need to generate log_policy_ many times
    for _ in range(100):
        log_policy_ = simulation(RandomPolicy)
        eval_policy_ = simulation(CBVowpalWabbit, log_policy_.history)
        real_mean_ips, eval_mean_ips = ips(log_policy_, eval_policy_)
        real_mean_dr, eval_mean_dr = dr(log_policy_, eval_policy_)
        real_mean_mb, eval_mean_mb = mb(log_policy_, eval_policy_)
        real_means_ips.append(real_mean_ips)
        eval_means_ips.append(eval_mean_ips)
        real_means_dr.append(real_mean_dr)
        eval_means_dr.append(eval_mean_dr)
        real_means_mb.append(real_mean_mb)
        eval_means_mb.append(eval_mean_mb)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Contextual Bandits with Vowpal Wabbit")

    ax1.set_title("Inverse Propensity Scoring")
    ax1.plot(real_means_ips, label="real average rewards")
    ax1.plot(eval_means_ips, label="estimated average rewards")
    ax1.legend()

    ax2.set_title("Doubly Robust")
    ax2.plot(real_means_dr, label="real average rewards")
    ax2.plot(eval_means_dr, label="estimated average rewards")
    ax2.legend()

    ax3.set_title("Model Based")
    ax3.plot(real_means_mb, label="real average rewards")
    ax3.plot(eval_means_mb, label="estimated average rewards")
    ax3.legend()
    plt.savefig("../results/CB_Vowpal_Wabbit")
    plt.show()