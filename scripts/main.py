"""This script runs the simulation"""

from environment import Environment
from parameters import Parameters
from functions import run_episode, compute_returns, zero_pad, compute_baselines, compute_advantages,create_jobs

# Create an object of parameters
param = Parameters()

# Create the environment with the desired parameters
env = Environment(param.nodes_types, param.jobs_types, param.number_jobs)

for iteration in range(param.iterations):
    states_episodes = []
    actions_episodes = []
    rewards_episodes = []

    jobset = create_jobs(param.jobs_types, param.number_jobs)
    # For each episode record the states, actions and rewards per time-step and store them in corresponding lists
    for episode in range(param.episodes):
        states, actions, rewards = run_episode(env, jobset)

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

    # Update grad

    # Update weights

print("Total accumulated rewards: " + str(sum(rewards)))
env.visualize()


