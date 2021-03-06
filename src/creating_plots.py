import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from src.evaluator import ModelBasedEstimator, DoublyRobustEstimator, IPSEvaluator
from src.main import simulation
from src.policy import RandomPolicy, DeterministicPolicy, CBVowpalWabbit


def create_plot(PolicyClass, policy_name):
    real_means_ips = []
    eval_means_ips = []
    real_means_dr = []
    eval_means_dr = []
    real_means_mb = []
    eval_means_mb = []

    all_res = []

    for i in tqdm(range(100)):
        real_means_ips_once = []
        eval_means_ips_once = []
        real_means_dr_once = []
        eval_means_dr_once = []
        real_means_mb_once = []
        eval_means_mb_once = []
        for _ in range(100):
            r_mean_ips, e_mean_ips = simulation(IPSEvaluator, PolicyClass)
            real_means_ips_once.append(r_mean_ips)
            eval_means_ips_once.append(e_mean_ips)
            r_mean_dr, e_mean_dr = simulation(DoublyRobustEstimator, PolicyClass)
            real_means_dr_once.append(r_mean_dr)
            eval_means_dr_once.append(e_mean_dr)
            r_mean_mb, e_mean_mb = simulation(ModelBasedEstimator, PolicyClass)
            real_means_mb_once.append(r_mean_mb)
            eval_means_mb_once.append(e_mean_mb)
        all_res += real_means_ips_once + eval_means_ips_once + real_means_dr_once + eval_means_dr_once + real_means_mb_once + eval_means_mb_once
        real_means_ips.append(np.array(real_means_ips_once).mean())
        eval_means_ips.append(np.array(eval_means_ips_once).mean())
        real_means_dr.append(np.array(real_means_dr_once).mean())
        eval_means_dr.append(np.array(eval_means_dr_once).mean())
        real_means_mb.append(np.array(real_means_mb_once).mean())
        eval_means_mb.append(np.array(eval_means_mb_once).mean())
    np.save('../results/25.05.2021/simulate_10_2.npy', all_res)

    # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(35, 20))
    # fig.suptitle(policy_name, fontsize=35)
    #
    # ax1.set_title("Inverse Propensity Scoring")
    # ax1.plot(real_means_ips, label="real average rewards")
    # ax1.plot(eval_means_ips, label="estimated average rewards")
    # ax1.legend()
    #
    # ax2.set_title("Doubly Robust")
    # ax2.plot(real_means_dr, label="real average rewards")
    # ax2.plot(eval_means_dr, label="estimated average rewards")
    # ax2.legend()
    #
    # ax3.set_title("Model Based")
    # ax3.plot(real_means_mb, label="real average rewards")
    # ax3.plot(eval_means_mb, label="estimated average rewards")
    # ax3.legend()
    # plt.savefig("../results/" + '_'.join(policy_name.split()) + '_1000')
    # plt.show()


    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 20))
    fig.suptitle(policy_name, fontsize=25)
    ax1.hist([real_means_ips, eval_means_ips], edgecolor="black", bins='rice',
             label=["real average rewards", "estimated average rewards"])
    ax1.set_title("Inverse Propensity Scoring")
    ax1.legend()

    ax2.hist([real_means_dr, eval_means_dr], edgecolor="black", bins='rice',
             label=["real average rewards", "estimated average rewards"])
    ax2.set_title("Doubly Robust")
    ax2.legend()

    ax3.hist([real_means_mb, eval_means_mb], edgecolor="black", bins='rice',
             label=["real average rewards", "estimated average rewards"])
    ax3.set_title("Model Based")
    ax3.legend()

    plt.savefig("../results/" + '_'.join(policy_name.split()) + '_hist_100_10_lot_of_context')

    plt.show()


create_plot(CBVowpalWabbit, "CBVowpalWabbit")
