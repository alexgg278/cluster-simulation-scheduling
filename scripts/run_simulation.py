"""This script runs the simulation"""
import functions_cluster as fc
import random

random.seed(1)

# Number of jobs
n_jobs = 20

job_1 = {
    'probability': 0.5,
    'cpu': 1,
    'memory': 800,
    'file_size': 4
}

job_2 = {
    'probability': 0.5,
    'cpu': 3,
    'memory': 100,
    'file_size': 2
}

jobs_types = [job_1, job_2]

# Type node 1 characteristics and number
node_1 = {
    'number': 1,
    'cpu': 8,
    'memory': 3000,
    'bw': 1
}

# Type node 2 characteristics and number
node_2 = {
    'number': 1,
    'cpu': 4,
    'memory': 5000,
    'bw': 2
}

nodes_types = [node_1, node_2]
fc.main_simulation(nodes_types, n_jobs, jobs_types, 0)
