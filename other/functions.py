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
    if len(env.buffer) == 2:
        sum_jobs = env.buffer[0].memory_request + env.buffer[1].memory_request
        if (env.nodes['node_1'].memory_available - sum_jobs) >= env.nodes['node_1'].memory_available and env.nodes["node_1"].check_resources(env.buffer):
            action = 3
        elif (env.nodes['node_2'].memory_available - sum_jobs) >= env.nodes['node_2'].memory_available and env.nodes["node_2"].check_resources(env.buffer):
            action = 6
        else:
            if env.nodes['node_1'].check_resources([env.buffer[0]]) and env.nodes['node_2'].check_resources([env.buffer[1]]) and env.nodes['node_2'].check_resources([env.buffer[0]]) and env.nodes['node_2'].check_resources([env.buffer[1]]):
                if env.nodes['node_1'].memory_available > env.nodes['node_2'].memory_available:
                    if env.buffer[0].memory_request > env.buffer[1].memory_request:
                        action = 7
                    elif env.buffer[0].memory_request < env.buffer[1].memory_request:
                        action = 8
                    else:
                        action = np.random.choice([7, 8])
                elif env.nodes['node_1'].memory_available < env.nodes['node_2'].memory_available:
                    if env.buffer[0].memory_request > env.buffer[1].memory_request:
                        action = 8
                    elif env.buffer[0].memory_request < env.buffer[1].memory_request:
                        action = 7
                    else:
                        action = np.random.choice([7, 8])
                else:
                    action = np.random.choice([7, 8])
            elif env.nodes['node_1'].check_resources([env.buffer[0]]) and env.nodes['node_2'].check_resources([env.buffer[1]]):
                action = 7
            elif env.nodes['node_2'].check_resources([env.buffer[0]]) and env.nodes['node_1'].check_resources([env.buffer[1]]):
                action = 8
            else:
                if not env.nodes['node_2'].check_resources([env.buffer[0]]) and not env.nodes['node_2'].check_resources([env.buffer[1]]):
                    if env.nodes['node_1'].check_resources([env.buffer[0]]) and not env.nodes['node_1'].check_resources([env.buffer[1]]):
                        action = 1
                    elif not env.nodes['node_1'].check_resources([env.buffer[0]]) and env.nodes['node_1'].check_resources([env.buffer[1]]):
                        action = 2
                    elif env.nodes['node_1'].check_resources(env.buffer):
                        action = 3
                    else:
                        action = 0
                elif not env.nodes['node_1'].check_resources([env.buffer[0]]) and not env.nodes['node_1'].check_resources([env.buffer[1]]):
                    if env.nodes['node_2'].check_resources([env.buffer[0]]) and not env.nodes['node_2'].check_resources([env.buffer[1]]):
                        action = 4
                    elif not env.nodes['node_2'].check_resources([env.buffer[0]]) and env.nodes['node_2'].check_resources([env.buffer[1]]):
                        action = 5
                    elif env.nodes['node_1'].check_resources(env.buffer):
                        action = 6
                    else:
                        action = 0
                else:
                    action = 0
    elif len(env.buffer) == 1:
        if env.nodes['node_1'].check_resources([env.buffer[0]]) and env.nodes['node_2'].check_resources([env.buffer[0]]):
            if env.nodes['node_1'].memory_available > env.nodes['node_2'].memory_available:
                action = 1
            elif env.nodes['node_1'].memory_available < env.nodes['node_2'].memory_available:
                action = 4
            else:
                action = np.random.choice([1, 4])
        elif env.nodes['node_1'].check_resources([env.buffer[0]]) and not env.nodes['node_2'].check_resources([env.buffer[0]]):
            action = 1
        elif not env.nodes['node_1'].check_resources([env.buffer[0]]) and env.nodes['node_2'].check_resources([env.buffer[0]]):
            action = 4
        else:
            action = 0
    else:
        action = 0
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

    nodes_memory = [[] for _ in env.nodes.items()]

    # Resets the environment. Creates list of new jobs
    ob = env.reset(jobset)

    done = False
    flag = False
    i = 0

    while not done and not flag:
        if i >= 300:
            flag = True

        # Pick action with RL agent
        if scheduler == 'RL':
            action = pg_network.get_action(ob.reshape((1, ob.shape[0])))
        elif scheduler == 'LB':
            action = load_balancer_scheduler(env)

        for idx, node in enumerate(env.nodes.values()):
            nodes_memory[idx].append(node.memory_used)

        # Step forward the environment given the action
        new_ob, r, done = env.step(action, info)

        # Store state, action and reward
        states.append(ob)
        actions.append(action)
        rewards.append(r)

        # Update observation
        ob = new_ob

        i += 1

    avg_job_duration = sum(env.jobs_total_time) / len(jobset)

    return np.array(states), np.array(actions), np.array(rewards), avg_job_duration, nodes_memory

def early_stopping(rewards, patience=25):
    if len(rewards) < patience:
        return False
    else:
        if rewards[-1] <= rewards[-patience]:
            return True
        elif rewards[-1] - rewards[-2] >= 20:
            return True
        else:
            return False

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


def create_jobs(jobs_types, number_jobs, param):
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
        if job_type['distr'] == 'n':
            x = np.random.normal(loc=param.mean, scale=param.std)
            x = np.round(x)
            while (x%2 != 0 or x<2):
                    x = np.random.normal(loc=param.mean, scale=param.std)
                    x = np.round(x)
            job_type['file_size'] = x
        if job_type['distr_mem']:
            job_type['memory'] = np.random.choice(job_type['distr_mem'])
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


def plot_iter(iter_list, title, folder):
    """
    Plots the evolution of parameters across iterations
    """
    plt.figure(figsize=(12, 9))
    plt.xticks(range(0, len(iter_list), 100))
    plt.ylim(top=max(iter_list)+1)
    plt.xlabel('Iterations')
    plt.ylabel(title)
    plt.plot(iter_list)

    # Save figure
    my_path = os.getcwd()
    plt.savefig(my_path + "/results/Affinity/" + folder + "/job.png")

    plt.show()


def plot_iter_2(iter_list_1, n, title, folder):
    """
    Plots the evolution of parameters across iterations
    """
    iter_list_2 = [n for _ in iter_list_1]
    plt.figure(figsize=(12, 9))
    plt.xticks(range(0, len(iter_list_1), 10))
    plt.yticks(np.arange(int(min(iter_list_1)-3), int(max(iter_list_1)+3), 1))
    plt.ylim(top=max(iter_list_1)+2, bottom=min(iter_list_1)-2)
    plt.xlabel('Iterations')
    plt.ylabel(title)
    plt.plot(iter_list_1, 'b-', label='DRL scheduler')
    plt.plot(iter_list_2, 'g--', label='LB scheduler')
    plt.legend()

    # Save figure
    my_path = os.getcwd()
    plt.savefig(my_path + "/results/Affinity/" + folder + "/duration_training.png")

    plt.show()


def plot_test_bars(x, y, title, file, folder):
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 9)
    ax.bar(x=(0, 1), height=(x, y), width=0.2, color=[(0.2, 0.4, 0.6, 0.6), 'green'])
    plt.xticks(np.arange(2), ('DRL scheduler', 'LB scheduler'))
    plt.ylabel('Avg. job duration')
    """
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    """
    # Save figure
    my_path = os.getcwd()
    plt.savefig(my_path + "/results/Affinity/" + folder + "/" + file)

    plt.show()


def plot_rew(iter_list, title, folder):
    """
    Plots the evolution of parameters across iterations
    """
    plt.figure(figsize=(12, 9))
    plt.xticks(range(0, len(iter_list), 10))
    plt.yticks(np.arange(int(min(iter_list)-20), int(max(iter_list)+20), 50))
    plt.xlabel('Iterations')
    plt.ylabel(title)
    plt.plot(iter_list)

    # Save figure
    my_path = os.getcwd()
    plt.savefig(my_path + "/results/Affinity/" + folder + "/reward_training.png")

    plt.show()


def plot_memory_usage(memory_nodes_RL, memory_nodes_LB, file, folder):
    figure, axes = plt.subplots(nrows=2, ncols=2)
    figure.set_size_inches(13, 10)
    figures = [[memory_nodes_RL[0], memory_nodes_LB[0]], [memory_nodes_RL[1], memory_nodes_LB[1]]]
    titles = [['DRL scheduler\n Node 1', 'LB scheduler\n Node 1'], ['Node 2', 'Node 2']]
    ylabel = [['Memory usage', ''], ['Memory usage', '']]

    for i, row in enumerate(axes):
        for j, col in enumerate(row):
            col.plot(figures[i][j])
            col.set_title(titles[i][j])
            col.set_ylabel(ylabel[i][j])
    figure.tight_layout(pad=3.0)
    # Save figure
    my_path = os.getcwd()
    plt.savefig(my_path + "/results/Affinity/" + folder + "/" + file)

    plt.show()


def plot_diff_memory_usage(memory_nodes_RL, memory_nodes_LB, file, folder):
    figure, axes = plt.subplots(nrows=1, ncols=2)
    figure.set_size_inches(13, 6)

    diff_RL = abs(np.array(memory_nodes_RL[0]) - np.array(memory_nodes_RL[1]))
    diff_LB = abs(np.array(memory_nodes_LB[0]) - np.array(memory_nodes_LB[1]))

    # Defining custom 'xlim' and 'ylim' values.
    maximum = np.max(np.concatenate((diff_RL, diff_LB)))
    custom_ylim = (0, maximum)

    figures = [diff_RL, diff_LB]
    titles = ['DRL scheduler', 'LB scheduler']
    ylabel = ['Diff. memory usage', 'Diff. memory usage']
    xlabel = ['Time-steps', 'Time-steps']

    plt.setp(axes, ylim=custom_ylim)

    for j, col in enumerate(axes):
        col.plot(figures[j])
        col.set_title(titles[j])
        col.set_xlabel(xlabel[j])
        col.set_ylabel(ylabel[j])

    figure.tight_layout(pad=3.0)

    # Save figure
    my_path = os.getcwd()
    plt.savefig(my_path + "/results/Affinity/" + folder + "/" + file)

    plt.show()
