"""This script runs the simulation"""
import numpy as np
from tqdm import tqdm

from environment import Environment
from parameters import Parameters
from functions import run_episode, compute_returns, zero_pad, compute_baselines, compute_advantages, create_jobs, plot_iter
from policy_network import PolicyGradient

# Create an object of parameters
param = Parameters()

# Create the environment with the desired parameters
env = Environment(param.nodes_types, param.jobs_types, param.number_jobs)

# Create the policy network
pg_network = PolicyGradient()

# Build placeholders and operations
pg_network.build(env, param)

# Performance
avg_episode_duration = []

for iteration in tqdm(range(param.iterations)):
    states_episodes = []
    actions_episodes = []
    rewards_episodes = []

    jobset = create_jobs(param.jobs_types, param.number_jobs)
    # For each episode record the states, actions and rewards per time-step and store them in corresponding lists
    for episode in range(param.episodes):
        states, actions, rewards = run_episode(env, jobset, pg_network)

        states_episodes.append(states)
        actions_episodes.append(actions)
        rewards_episodes.append(rewards)

    # Compute returns
    returns = [compute_returns(rewards, param.gamma) for rewards in rewards_episodes]

    # Zero pad returns to have equal length
    zero_padded_returns = zero_pad(returns)

    # Compute baselines
    baselines = compute_baselines(zero_padded_returns)

    # Compute advantages
    advantages = compute_advantages(returns, baselines)

    # Update weights
    for idx in range(param.episodes):
        pg_network.optimize_pg(states_episodes[idx], actions_episodes[idx], advantages[idx], param.lr)

    avg_episode_duration.append(np.mean([i.shape[0] for i in states_episodes]))

print(avg_episode_duration)
plot_iter(avg_episode_duration)


