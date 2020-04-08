"""This script runs the simulation"""
import functions_cluster as fc
import random

random.seed(1)

# Number of jobs
n_jobs = 100

job_1 = {
    'probability': 0.5,
    'cpu': 2,
    'memory': 500,
    'file_size': 2
}

job_2 = {
    'probability': 0.5,
    'cpu': 2,
    'memory': 500,
    'file_size': 4
}

jobs_types = [job_1, job_2]

# Type node 1 characteristics and number
node_1 = {
    'number': 1,
    'cpu': 8,
    'memory': 2000,
    'bw': 1
}

# Type node 2 characteristics and number
node_2 = {
    'number': 1,
    'cpu': 8,
    'memory': 2000,
    'bw': 2
}

nodes_types = [node_1, node_2]
fc.main_simulation(nodes_types, n_jobs, jobs_types, 0)
