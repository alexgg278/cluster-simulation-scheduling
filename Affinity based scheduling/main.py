"""This script runs the simulation"""
import numpy as np
from tqdm import tqdm

from environment import Environment
from parameters import Parameters
from functions import run_episode, compute_returns, zero_pad, compute_baselines, compute_advantages, create_jobs, plot_iter, plot_rew, plot_iter_2, plot_test_bars, early_stopping, plot_memory_usage, plot_diff_memory_usage
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

jobsets = [create_jobs(param.jobs_types, param.number_jobs, param) for jobset in range(param.jobsets)]

for iteration in tqdm(range(param.iterations)):
    if not early_stopping(avg_reward, param.patience):
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
                states, actions, rewards, avg_job_duration_ep, _ = run_episode(env, jobset, pg_network)

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
    else:
        break

# How does the process look like step by step for the training jobsets
print("Training-jobsets:")
print("\nRL scheduler:")
for i, jobset in enumerate(jobsets):
    print("\nJobset " + str(i) + ":")
    states, actions, rewards, train_RL, train_memory_RL = run_episode(env, jobset, pg_network, info=True)

print("\nLB scheduler:")
lb_list = []
for i, jobset in enumerate(jobsets):
    print("\nJobset " + str(i) + ":")
    states, actions, rewards, train_LB, train_memory_LB = run_episode(env, jobset, pg_network, scheduler='LB', info=True)
    lb_list.append(train_LB)
lb_duration = np.mean(lb_list)

# How does the process look like step by step for a test-jobset
print("\nTest-jobset:")
test_jobset = create_jobs(param.jobs_types, 20, param)
print("\nRL scheduler:")
states, actions_x, rewards, test_RL, test_memory_RL = run_episode(env, test_jobset, pg_network, info=True)
print("\nLB scheduler:")
states, actions_y, rewards, test_LB, test_memory_LB = run_episode(env, test_jobset, pg_network, info=True, scheduler='LB')

print('\nAverage episode durations training jobsets: ' + str(avg_episode_duration))
print('Average job durations training jobsets: ' + str(avg_job_duration))

print('\nTest jobset actions RL:' + str(actions_x))
print('Test jobset avg. job duration RL:' + str(test_RL))

print('\nTest jobset actions LB:' + str(actions_y))
print('Test jobset avg. job duration LB:' + str(test_LB))

# plot_iter(avg_episode_duration, 'Avg. episode duration')
folder = 'Test18'
plot_iter_2(avg_job_duration, lb_duration, 'Avg. job duration', folder)
plot_rew(avg_reward, 'Avg. total reward', folder)
plot_test_bars(train_RL, train_LB, 'Training set', 'final_duration_training.png', folder)
plot_test_bars(test_RL, test_LB, 'Test set', 'duration_test.png', folder)
plot_memory_usage(train_memory_RL, train_memory_LB, 'memory_usage_training.png', folder)
plot_memory_usage(test_memory_RL, test_memory_LB, 'memory_usage_test.png', folder)
plot_diff_memory_usage(train_memory_RL, train_memory_LB, 'diff_memory_usage_training.png', folder)
plot_diff_memory_usage(test_memory_RL, test_memory_LB, 'diff_memory_usage_test.png', folder)
