"""This script runs the simulation"""
import numpy as np
from tqdm import tqdm

from environment import Environment
from parameters import Parameters
from functions import run_episode, compute_returns, zero_pad, compute_baselines, compute_advantages, create_jobs, plot_iter, plot_rew
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

jobsets = [create_jobs(param.jobs_types, param.number_jobs) for jobset in range(param.jobsets)]

for iteration in tqdm(range(param.iterations)):

    states_jobsets = []
    actions_jobsets = []
    rewards_jobsets = []
    advantages_jobsets = []

    # Visualization
    avg_episode_duration_jobset = []
    avg_job_duration_jobset = []
    avg_reward_jobset = []

    for jobset in jobsets:

        states_episodes = []
        actions_episodes = []
        rewards_episodes = []

        # Visualization
        avg_job_duration_ep_list = []
        total_reward_episodes = []

        # For each episode record the states, actions and rewards per time-step and store them in corresponding lists
        for episode in range(param.episodes):
            states, actions, rewards, avg_job_duration_ep = run_episode(env, jobset, pg_network)

            states_episodes.append(states)
            actions_episodes.append(actions)
            rewards_episodes.append(rewards)

            # Visualization
            avg_job_duration_ep_list.append(avg_job_duration_ep)
            total_reward_episodes.append(sum(rewards))

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

        # Visualization
        # Store in a list the avg duration of the jobs of all the episodes of the iteration
        avg_job_duration_jobset.append(sum(avg_job_duration_ep_list) / param.episodes)
        # Store average episode duration
        avg_episode_duration_jobset.append(np.mean([i.shape[0] for i in states_episodes]))
        # Store average episode reward
        avg_reward_jobset.append(sum(total_reward_episodes) / param.episodes)

    # Update weights
    for j in range(param.jobsets):
        for i in range(param.episodes):
            pg_network.optimize_pg(states_jobsets[j][i], actions_jobsets[j][i], advantages_jobsets[j][i], param.lr)

    # Visualization
    # Store in a list the avg duration of the jobs of all the episodes of the iteration
    avg_job_duration.append(sum(avg_job_duration_jobset) / param.jobsets)
    # Store average episode duration
    avg_episode_duration.append(sum(avg_episode_duration_jobset) / param.jobsets)
    # Store average iteration episode reward
    avg_reward.append(sum(avg_reward_jobset) / param.jobsets)

# How does the process look like step by step for the training jobsets
print("Training-jobsets:")
for i, jobset in enumerate(jobsets):
    print("\nJobset "+ str(i) + ":")
    states, actions, rewards, x = run_episode(env, jobset, pg_network, info=True)

# How does the process look like step by step for a test-jobset
print("\nTest-jobset:")
jobset = create_jobs(param.jobs_types, param.number_jobs)
states, actions, rewards, x = run_episode(env, jobset, pg_network, info=True)

print('\nAverage episode durations training jobsets: ' + str(avg_episode_duration))
print('Average job durations training jobsets: ' + str(avg_job_duration))

print('Test jobset actions:' + str(actions))
print('Test jobset avg. job duration:' + str(x))

# plot_iter(avg_episode_duration, 'Avg. episode duration')
plot_iter(avg_job_duration, 'Avg. job duration')
plot_rew(avg_reward, 'Avg. total reward')