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
env = Environment(param)

# Create the policy network
pg_network = PolicyGradient()

# Build placeholders and operations
pg_network.build(env, param)

# Performance
avg_episode_duration = []
avg_job_duration = []

jobsets = [create_jobs(param.jobs_types, param.number_jobs) for jobset in range(param.jobsets)]

for iteration in tqdm(range(param.iterations)):

    states_jobsets = []
    actions_jobsets = []
    rewards_jobsets = []
    advantages_jobsets = []

    # Performance
    avg_episode_duration_jobset = []
    avg_job_duration_jobset = []

    for jobset in jobsets:

        states_episodes = []
        actions_episodes = []
        rewards_episodes = []

        avg_job_duration_ep_list = []

        # For each episode record the states, actions and rewards per time-step and store them in corresponding lists
        for episode in range(param.episodes):
            states, actions, rewards, avg_job_duration_ep = run_episode(env, jobset, pg_network)

            states_episodes.append(states)
            actions_episodes.append(actions)
            rewards_episodes.append(rewards)

            avg_job_duration_ep_list.append(avg_job_duration_ep)

        # Compute returns
        returns = [compute_returns(rewards, param.gamma) for rewards in rewards_episodes]

        # Zero pad returns to have equal length
        zero_padded_returns = zero_pad(returns)

        # Compute baselines
        baselines = compute_baselines(zero_padded_returns)

        # Compute advantages
        advantages = compute_advantages(returns, baselines)

        states_jobsets.append(states_episodes)
        actions_jobsets.append(actions_episodes)
        rewards_jobsets.append(rewards_episodes)
        advantages_jobsets.append(advantages)

        # Store in a list the avg duration of the jobs of all the episodes of the iteration
        avg_job_duration_jobset.append(sum(avg_job_duration_ep_list) / param.episodes)
        # Store average episode duration
        avg_episode_duration_jobset.append(np.mean([i.shape[0] for i in states_episodes]))

    # Update weights
    for j in range(param.jobsets):
        for i in range(param.episodes):
            pg_network.optimize_pg(states_jobsets[j][i], actions_jobsets[j][i], advantages_jobsets[j][i], param.lr)

    # Store in a list the avg duration of the jobs of all the episodes of the iteration
    avg_job_duration.append(sum(avg_job_duration_jobset) / param.jobsets)
    # Store average episode duration
    avg_episode_duration.append(sum(avg_episode_duration_jobset) / param.jobsets)

jobset = create_jobs(param.jobs_types, param.number_jobs)
states, actions, rewards, x = run_episode(env, jobset, pg_network, info=True)

print(avg_episode_duration)
print(avg_job_duration)
print(actions)
plot_iter(avg_episode_duration)
plot_iter(avg_job_duration)


