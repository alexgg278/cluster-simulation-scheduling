"""This document contains the functions used for the simulation. The first one is the main simulation loop"""

import random
import matplotlib.pyplot as plt
import numpy as np
import os

from job import Job
from node import Node

def load_balancer_scheduler(env):
    """
    This function implements an scheduler that takes scheduling decisions with the goal of balancing the load of the
    cluster nodes
    """
    action = None
    available_nodes = []

    # Create list with nodes with available memory
    for node in env.nodes.values():
        if node.memory_available != 0:
            available_nodes.append(node)

    # If there is just one node with available memory allocate the first job there
    if len(available_nodes) == 1:
        if available_nodes[0].node_id == 0:
            action = 1
        if available_nodes[0].node_id == 1:
            action = 4
    elif not available_nodes:
        action = 0
    else:
        actions = [3, 6, 7, 8]
        action = random.choice(actions)
    return action


def run_episode(env, jobset, pg_network=None, info=False, scheduler='RL'):
    """
    This function runs a full trajectory or episode using the current policy.
    It returns the state, actions and rewards at each time-step
    """

    # Lists to record data in each time-step
    rewards = []
    states = []
    actions = []

    # Resets the environment. Creates list of new jobs
    ob = env.reset(jobset)

    done = False

    while not done:

        # Pick action with RL agent
        if scheduler == 'RL':
            action = pg_network.get_action(ob.reshape((1, ob.shape[0])))
        elif scheduler == 'lb':
            action = load_balancer_scheduler(env)

        # Step forward the environment given the action
        new_ob, r, done = env.step(action, info)

        # Store state, action and reward
        states.append(ob)
        actions.append(action)
        rewards.append(r)

        # Update observation
        ob = new_ob
    x = env.jobs_total_time
    avg_job_duration = sum(env.jobs_total_time) / env.number_jobs

    return np.array(states), np.array(actions), np.array(rewards), avg_job_duration


def compute_returns(rewards, gamma):
    """
    This function compute the returns at every time-step of one or more episodes
    Input: List of rewards from one episode
    Output: Array of returns at each time-step of the episode
    """
    returns = []
    for t in range(len(rewards)):
        v = 0
        for s, reward in enumerate(rewards[t:]):
            v += (gamma ** (s-t)) * reward
        returns.append(v)

    return np.array(returns)


def zero_pad(elements):
    """
    This function pads with 0 the inputs list so that their length is equal to the longest list
    Input: Lists or arrays to be padded
    Output: Zero-padded lists or arrays
    """
    # Calculate length of the longest element
    max_len = max([len(e) for e in elements])

    zero_padded_elements = []
    for e in elements:
        diff = max_len - len(e)
        zero_padded_elements.append(np.concatenate((e, np.zeros(diff))))

    return zero_padded_elements


def compute_baselines(returns_episodes):
    """
    This function computes the baselines. One baseline is computed per time-step for all episodes
    Input: List of elements from which we calculate baselines
    Returns: Array array of baselines, one baseline per time-step
    """
    # L = max([len(l) for l in returns_episodes])
    # baselines = []
    # for t in range(L):

    baseline = np.mean(returns_episodes, axis=0)

    return baseline


def compute_advantages(returns, baselines):
    """
    This function computes the difference between the return and the baseline at each time step
    """
    advantages = [r - baselines[:len(r)] for r in returns]

    return advantages


def create_nodes(nodes_types):
    """
    In this function the nodes are created and stored in a dict based
    on the parameters stored in nodes_types
    """
    nodes = {}
    i = 0
    for node_type in nodes_types:
        for node in range(node_type['number']):
            nodes['node_' + str(i + 1)] = Node(i, node_type['cpu'], node_type['memory'], node_type['bw'])
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


def create_jobs(jobs_types, number_jobs):
    """
    This function takes as input the total number of jobs and the a list of dicts with the jobs types probabilities
    and characteristics
    Returns a list with the jobs
    """
    job_list = []

    # Store in a list the probabilities of each job type
    prob_seq = [job_type['probability'] for job_type in jobs_types]

    for i in range(number_jobs):
        job_type = random.choices(jobs_types, prob_seq)[0]
        job_list.append(Job(number_jobs-i, job_type['cpu'], job_type['memory'], job_type['file_size'], job_type['transmit']))

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


def display_jobs_nodes(n_jobs_nodes):
    """
    This function plots the new number of jobs allocated at each time step and
    the number of jobs in each node in each time-step as a time series
    """
    max_value = 0
    max_value_jobs_nodes = max([max(node) for node in n_jobs_nodes if max_value < max(node)])
    plt.figure(figsize=(9, 12))
    for i, node in enumerate(n_jobs_nodes):
        plt.subplot(len(n_jobs_nodes), 1, i + 1)
        plt.xticks(range(len(node)))
        plt.ylim(top=max_value_jobs_nodes + 1)
        plt.title('Number of jobs in Node ' + str(i + 1))
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


def plot_iter(iter_list, title):
    """
    Plots the evolution of parameters across iterations
    """
    plt.figure(figsize=(12, 12))
    plt.xticks(range(0, len(iter_list), 100))
    plt.ylim(top=max(iter_list)+1)
    plt.xlabel('Iterations')
    plt.ylabel(title)
    plt.plot(iter_list)

    # Save figure
    my_path = os.getcwd()
    plt.savefig(my_path + "/results/BW/Test3/job.png")

    plt.show()


def plot_rew(iter_list, title):
    """
    Plots the evolution of parameters across iterations
    """
    plt.figure(figsize=(12, 12))
    plt.xticks(range(0, len(iter_list), 100))
    plt.xlabel('Iterations')
    plt.ylabel(title)
    plt.plot(iter_list)

    # Save figure
    my_path = os.getcwd()
    plt.savefig(my_path + "/results/BW/Test3/reward.png")

    plt.show()
