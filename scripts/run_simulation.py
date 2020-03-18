import random

from job import Job
import functions_cluster as fc

# Number of jobs
n_jobs = 20

# Characteristics job 1
cpu_job_1 = 1
memory_job_1 = 200
file_size_job_1 = 2

# Characteristics job 2
cpu_job_2 = 0.5
memory_job_2 = 500
file_size_job_2 = 4

# Append in lists the requirements of the jobs
cpu_jobs_reqs = [cpu_job_1, cpu_job_2]
memory_jobs_reqs = [memory_job_1, memory_job_2]
file_size_jobs_reqs = [file_size_job_1, file_size_job_2]

# Create a list to store the jobs
jobs_list = []

# We store 1 job in the list in each iteration randomly choosing between the two types of jobs
for i in range(n_jobs):
    job_type = random.randint(0, 1)
    cpu_job = cpu_jobs_reqs[job_type]
    memory_job = memory_jobs_reqs[job_type]
    file_size_job = file_size_jobs_reqs[job_type]
    jobs_list.append(Job(i, cpu_job, memory_job, file_size_job))
    
# Create the two nodes

# Characteristics node 1
cpu_node_1 = 20
memory_node_1 = 10000
bw_node_1 = 1

# Characteristics node 2
cpu_node_2 = 20
memory_node_2 = 10000
bw_node_2 = 2

nodes_cpu = [cpu_node_1, cpu_node_2]
nodes_memory = [memory_node_1, memory_node_2]
nodes_bw = [bw_node_1, bw_node_2]

fc.main_simulation(2, nodes_cpu, nodes_memory, nodes_bw, jobs_list, 0)