"""This script runs the simulation"""

from environment import Environment
from parameters import Parameters
from functions import run_episode, create_jobs

# Create an object of parameters
param = Parameters()

# Create the environment with the desired parameters
env = Environment(param)

jobsets = [create_jobs(param.jobs_types, param.number_jobs) for jobset in range(param.jobsets)]

for jobset in jobsets:
    states, actions, rewards, avg_job_duration_ep = run_episode(env, jobset, scheduler='lb')

print(avg_job_duration_ep)

