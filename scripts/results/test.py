"""This script runs the simulation"""
import numpy as np
from tqdm import tqdm

from environment import Environment
from parameters import Parameters
from functions import run_episode, compute_returns, zero_pad, compute_baselines, compute_advantages, create_jobs, plot_rew, plot_iter_2, plot_test_bars, early_stopping, plot_memory_usage
from policy_network import PolicyGradient

# Create an object of parameters
param = Parameters()

# Create the environment with the desired parameters
env = Environment(param)

# Create the policy network
pg_network = PolicyGradient()

# Build placeholders and operations
pg_network.build(env, param)

jobsets = [create_jobs(param.jobs_types, param) for jobset in range(param.jobsets)]

print("\nLB scheduler:")
lb_list = []
for i, jobset in enumerate(jobsets):
    print("\nJobset " + str(i) + ":")
    states, actions, rewards, train_LB, train_memory_LB = run_episode(env, jobset, pg_network, scheduler='LB', info=True)
    lb_list.append(train_LB)
lb_duration = np.mean(lb_list)