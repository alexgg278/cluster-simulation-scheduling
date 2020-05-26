"""This script runs the simulation"""
import numpy as np

from environment import Environment
from parameters import Parameters
from functions import run_episode, compute_returns, zero_pad, compute_baselines, compute_advantages, create_jobs, plot_iter, plot_rew, plot_iter_2, plot_test_bars, plot_memory_usage
from policy_network import PolicyGradient

np.random.seed(49)

# Create an object of parameters
param = Parameters()

# Create the environment with the desired parameters
env = Environment(param)

# Create the policy network
pg_network = PolicyGradient()

# Build placeholders and operations
pg_network.build(env, param)

# Visualization
avg_episode_duration = []
avg_job_duration = []
avg_reward = []

jobsets = [create_jobs(param.jobs_types, param.number_jobs, param) for jobset in range(param.jobsets)]

print("\nLB scheduler:")
lb_list = []
for i, jobset in enumerate(jobsets):
    print("\nJobset " + str(i) + ":")
    states, actions, rewards, x, _ = run_episode(env, jobset, pg_network, scheduler='LB', info=True)
    lb_list.append(x)

print(actions)
lb_duration = np.mean(lb_list)
print(lb_duration)