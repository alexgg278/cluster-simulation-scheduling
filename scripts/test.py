"""This script runs the simulation"""
import numpy as np
from tqdm import tqdm
import random

from environment import Environment
from parameters import Parameters
from functions import run_episode, compute_returns, zero_pad, compute_baselines, compute_advantages, create_jobs, plot_iter, plot_rew, plot_iter_2, plot_test_bars, early_stopping, plot_memory_usage
from policy_network import PolicyGradient

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

# How does the process look like step by step for a test-jobset
print("\nTest-jobset:")
test_jobset = create_jobs(param.jobs_types, param)
states, actions_y, rewards, test_LB, test_memory_LB = run_episode(env, test_jobset, pg_network, info=True, scheduler='LB')

