"""This document contains the functions used for the simulation. The first one is the main simulation loop"""

import random
import matplotlib.pyplot as plt

from node import Node
from job import Job


def main_simulation(nodes_types, n_jobs, jobs_types, display='0'):
    """
    This is the function that executes the main simulation
    Input:
    nodes_types: Is a list of dictionaries with the following structure:
    {'number': number of nodes of this type, 'cpu': cpu of this type of nodes,
    'memory': memory of this type of nodes, 'bw': bw of this type opf nodes}
    jobs_list: list containing the jobs objects to be allocated
    display: 0: display only plots, 1: display plots and node's status, 2: display node's status

    The simulation selects creates the cluster nodes from the list nodes_types.
    Runs while there is jobs to allocate or while there are jobs running in any node
    At each time step a random number between 2 limits of jobs is selected to be allocated
    The jobs are allocated randomly to one of the nodes of the cluster
    In every time step the remaining time of exec of the nodes is reduced by one,
    when this time is 0, the job is removed and resourced are released
    Finally performance data is plot
    """

    # Create a list with the nodes specified in nodes_types
    nodes = create_nodes(nodes_types)

    # Create jobs and return them to a list
    jobs_list = select_jobs(n_jobs, jobs_types)

    # Create buffer
    buffer = []

    # Create lists for later performance visualization
    n_new_jobs = []
    n_jobs_nodes = [[] for i in range(len(nodes))]
    cpu_nodes_list = [[] for i in range(len(nodes))]
    memory_nodes_list = [[] for i in range(len(nodes))]
    total_job_duration_list = []
    buffer_list = []
    iteration = 0

    # Run while jobs in list of pending jobs or while any node has jobs running
    while jobs_list or jobs_running(nodes):
        print('\nIteration: ' + str(iteration))

        # Decrease exec time by 1 of all the jobs allocated in the nodes
        for node in nodes:
            terminated_jobs_time = node.decrease_job_time()
            # If job is terminated append its total duration to list
            if terminated_jobs_time:
                total_job_duration_list += terminated_jobs_time

        for node in nodes:
            if node.jobs:
                for job in node.jobs:
                    print('Job: ' + str(job['job'].job_id) + '. Time: ' + str(job['time']))

        # Try to allocate jobs from the buffer first
        buffer_copy = buffer[:]
        for job in buffer_copy:
            # Scheduler: Selects node
            node = random.choice(nodes)
            # Increase by one the time of the job since being in the buffer counts as being deployed for job duration
            job.time += 1
            # Attempt to append job to node
            job_duration = node.append_job(job)
            if job_duration:
                print('Job ' + str(job.get_job_id()) + ' allocated' + '. Time: ' + str(job_duration))
                buffer.remove(job)
        
        # Generate a number of jobs to allocate
        n_jobs_iteration = number_of_jobs(jobs_list, 2, 4)
        if jobs_list:
            # Allocate each job in a random node in the cluster
            for i in range(n_jobs_iteration):
                # Scheduler: Selects node
                node = random.choice(nodes)

                # Select and drops last element from jobs_list
                job = jobs_list.pop()

                # Attempt to append job to node
                job_duration = node.append_job(job)

                # If job was successfully allocated, print message and store expected time
                if job_duration:
                    print('Job ' + str(job.get_job_id()) + ' allocated' + '. Time: ' + str(job_duration))

                # Otherwise store job in buffer
                else:
                    buffer.append(job)

        # Store buffer size for visualization
        if buffer:
            buffer_list.append(len(buffer))
        else:
            buffer_list.append(0)
        
        # Store data for visualization
        n_new_jobs.append(n_jobs_iteration)
        for i, node in enumerate(nodes):
            n_jobs_nodes[i].append(len(node.get_jobs())) 
            cpu_nodes_list[i].append(node.get_cpu_used())
            memory_nodes_list[i].append(node.get_memory_used())
            
        if display == 1 or display == 2:
            # Display real time status of nodes
            display_node_status(nodes, iteration)
            
        iteration += 1
    
    if display == 0 or display == 1:
        # Visualize data from the simulation
        display_avg_job_duration(total_job_duration_list)
        print(total_job_duration_list)
        display_buffer(buffer_list)
        display_njobs_ts(n_new_jobs, n_jobs_nodes)
        display_cpu_usage(nodes, cpu_nodes_list)
        display_memory_usage(nodes, memory_nodes_list)


def create_nodes(nodes_types):
    """
    In this function the nodes are created and appended to a list based
    on the parameters stored in nodes_types
    """
    nodes = []
    i = 0
    for node_type in nodes_types:
        for node in range(node_type['number']):
            nodes.append(Node(i, node_type['cpu'], node_type['memory'], node_type['bw']))
            i += 1

    return nodes


def number_of_jobs(jobs_list, n_min, n_max):
    """
    This function generates a number between n_min and n_max
    for the number of jobs to be allocated at each time-step.
    When the number of remaining jobs in the list is less than the generated number,
    the number of jobs is the remaining jobs to be allocated
    """
    n_jobs_iteration = random.randint(n_min, n_max)
    if n_jobs_iteration > len(jobs_list):
        n_jobs_iteration = len(jobs_list)
    elif not jobs_list:
        n_jobs_iteration = 0
    return n_jobs_iteration


def select_jobs(n_jobs, jobs_types):
    """
    This function takes as input the total number of jobs and the a list of dicts with the jobs types probabilities
    and characteristics Returns a list with the jobs
    """
    job_list = []

    # Store in a list the probabilities of each job type
    prob_seq = [job_type['probability'] for job_type in jobs_types]

    for i in range(n_jobs):
        job_type = random.choices(jobs_types, prob_seq)[0]
        job_list.append(Job(i, job_type['cpu'], job_type['memory'], job_type['file_size']))

    return job_list


def display_node_status(nodes, iteration):
    """
    This function displays the status of each node at each time step.
    CPU usage, memory usage and allocated jobs.
    Requires as an input a list of nodes
    """
    status = '\n Iteration ' + str(iteration)
    for node in nodes:
        status += '\n \t Status of Node ' + str(node.get_node_id()) + ':'
        status += '\n \t \t Bw: ' + str(node.get_bw())
        status += '\n \t \t CPU: ' + str(node.get_cpu_used()) + '/' + str(node.get_cpu_capacity()) + ' c'
        status += '\n \t \t Memory: ' + str(node.get_memory_used()) + '/' + str(node.get_memory_capacity()) + ' MB'
        status += '\n \t \t Jobs allocated:'
        for job in node.get_jobs():
            status += '\n \n \t \t \t Job: ' + str(job['job'].get_job_id())
            status += '\n \t \t \t CPU req: ' + str(job['job'].get_cpu_request()) + ' c'
            status += '\n \t \t \t Memory req: ' + str(job['job'].get_memory_request()) + ' MB'
            status += '\n \t \t \t Time remaining: ' + str(job['time']) + 's'
        print(status)
        status = '' 
        
        
def display_njobs_ts(n_new_jobs, n_jobs_nodes):
    """
    This function plots the new number of jobs allocated at each time step and
    the number of jobs in each node in each time-step as a time series
    """
    max_value = 0
    max_value_new_jobs = max(n_new_jobs)
    max_value_jobs_nodes = max([max(node) for node in n_jobs_nodes if max_value < max(node)])
    plt.figure(figsize=(9, 12))
    plt.subplot(len(n_jobs_nodes) + 1, 1, 1)
    plt.xticks(range(len(n_new_jobs)))
    plt.ylim(top=max_value_new_jobs+1)
    plt.title('Number of new jobs per iteration')
    plt.plot(n_new_jobs)
    
    for i, node in enumerate(n_jobs_nodes):
        plt.subplot(len(n_jobs_nodes)+1, 1, i+2)
        plt.xticks(range(len(n_new_jobs)))
        plt.ylim(top=max_value_jobs_nodes+1)
        plt.title('Number of jobs in Node ' + str(i+1))
        plt.plot(node)

    plt.show()
    
    
def display_cpu_usage(nodes, cpu_nodes_list):
    """This function plot the time-series of the CPU usage of the cluster nodes"""
    plt.figure(figsize=(9, 12))
    for i, node in enumerate(cpu_nodes_list):
        plt.subplot(len(cpu_nodes_list), 1, i+1)
        plt.xticks(range(len(node)))
        plt.ylim(top=nodes[i].get_cpu_capacity())
        plt.title('Node ' + str(i+1) + ' CPU usage')
        plt.plot(node)

    plt.show()


def display_memory_usage(nodes, memory_nodes_list):
    """This function plot the time-series of the memory usage of the cluster nodes"""
    plt.figure(figsize=(9, 12))
    for i, node in enumerate(memory_nodes_list):
        plt.subplot(len(memory_nodes_list), 1, i+1)
        plt.xticks(range(len(node)))
        plt.ylim(top=nodes[i].get_memory_capacity())
        plt.title('Node ' + str(i+1) + ' memory usage')
        plt.plot(node)

    plt.show()


def display_buffer(buffer_list):
    """This function plots the contents of buffer_list, that is the size of ethe buffer at each time step"""
    plt.figure(figsize=(9, 12))
    plt.xticks(range(len(buffer_list)))
    plt.ylim(top=max(buffer_list)+1)
    plt.title('Buffer size')
    plt.plot(buffer_list)


def display_avg_job_duration(job_duration_list):
    """
    This function displays the average duration of all the jobs
    allocated in the cluster after all the execution finishes
    """
    mean_job_duration = sum(job_duration_list) / len(job_duration_list)
    print('The average job duration in the nodes is: ' + str(mean_job_duration))
    
    
def jobs_running(nodes):
    """
    Returns True if any node in nodes has jobs running
    Returns False otherwise
    """
    flag = False
    for node in nodes:
        if node.check_jobs():
            flag = True
    return flag
