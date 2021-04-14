import matplotlib.pyplot as plt

from src.main import parallel_stimulation, ips, dr, mb, simulation
from src.policy import RandomPolicy, DeterministicPolicy, CBVowpalWabbit


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

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(policy_name)

    ax1.set_title("Inverse Propensity Scoring")
    ax1.plot(real_means_ips, label="real average rewards")
    ax1.plot(eval_means_ips, label="estimated average awards")
    ax1.legend()

    ax2.set_title("Doubly Robust")
    ax2.plot(real_means_dr, label="real average rewards")
    ax2.plot(eval_means_dr, label="estimated average awards")
    ax2.legend()

    ax3.set_title("Model Based")
    ax3.plot(real_means_mb, label="real average rewards")
    ax3.plot(eval_means_mb, label="estimated average awards")
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
    log_policy_ = simulation(RandomPolicy)
    for _ in range(100):
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

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle("Contextual Bandits with Vowpal Wabbit")

    ax1.set_title("Inverse Propensity Scoring")
    ax1.plot(real_means_ips, label="real average rewards")
    ax1.plot(eval_means_ips, label="estimated average awards")
    ax1.legend()

    ax2.set_title("Doubly Robust")
    ax2.plot(real_means_dr, label="real average rewards")
    ax2.plot(eval_means_dr, label="estimated average awards")
    ax2.legend()

    ax3.set_title("Model Based")
    ax3.plot(real_means_mb, label="real average rewards")
    ax3.plot(eval_means_mb, label="estimated average awards")
    ax3.legend()
    plt.savefig("../results/CB_Vowpal_Wabbit")
    plt.show()


create_plot(DeterministicPolicy, "Deterministic Policy")
create_plot(RandomPolicy, "Random Policy")
create_plot_CBVowpalWabbit()
