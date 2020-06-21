"""
This is the environment class that models the cluster simulation
"""
import numpy as np

import functions as fc

from keras.utils import to_categorical

class Environment():

    def __init__(self, param):

        # Dictionary containing the nodes
        self.nodes = fc.create_nodes(param.nodes_types)

        # List containing the jobs to be allocated
        self.jobs_types = param.jobs_types
        self.number_jobs = param.number_jobs
        self.jobs = None

        # Buffer or job slots. Max size == 2
        self.bff_size = 1
        self.buffer = []

        self.jobs_total_time = []

        self.time = 0

        self.action_space = param.action_space
        self.state_space = param.state_space

        # Visualization lists
        self.n_jobs_nodes = [[] for i in range(len(self.nodes.values()))]
        self.cpu_nodes_list = [[] for i in range(len(self.nodes.values()))]
        self.memory_nodes_list = [[] for i in range(len(self.nodes.values()))]
        self.buffer_list = []

        self.param = param

        self.acc = False

    def reset(self, jobset):
        """Resets environment by resetting nodes, job list and filling the buffer with the first two jobs"""

        # Reset time
        self.time = 0

        self.acc_rew = 0

        self.jobs_total_time = []

        # Reset nodes
        for node in self.nodes.values():
            node.reset()

        # Create list of jobs
        for job in jobset:
            job.reset()

        self.jobs = jobset[:]
        self.buffer = []

        # Fill the buffer with the first jobs
        for i in range(self.bff_size):
            self.buffer.append(self.jobs.pop())

        return self.observation(0)

    def observation(self, action):
        """Returns the state of the system"""

        state = []
        if self.param.jobs_types[0]['distr_mem']:
            nodes_space = []
            for node in self.nodes.values():
                if node.memory_available == 0:
                    nodes_space.append(0)
                elif node.memory_available == 250:
                    nodes_space.append(1)
                elif node.memory_available == 500:
                    nodes_space.append(2)
                elif node.memory_available == 750:
                    nodes_space.append(3)
                else:
                    print('Error')
            for node in self.nodes.values():
                if node.jobs:
                    for job in node.jobs:
                        state.append(job['job'].app)

                diff = 1 - len(node.jobs)
                for i in range(diff):
                    state.append(0)

            nodes_space = np.array(nodes_space)
            nodes_space = to_categorical(nodes_space, num_classes=4)
            nodes_space = np.concatenate(nodes_space)

            jobs_size = []
            for job in self.buffer:
                if job.memory_request == 250:
                    jobs_size.append(1)
                elif job.memory_request == 500:
                    jobs_size.append(2)
                elif job.memory_request == 750:
                    jobs_size.append(3)
                else:
                    print('Error')

                state.append(job.app)

            diff = 1 - len(self.buffer)
            if diff > 0:
                for _ in range(diff):
                    jobs_size.append(0)
                    state.append(0)

            jobs_size = np.array(jobs_size)
            jobs_size = to_categorical(jobs_size, num_classes=4)
            jobs_size = np.concatenate(jobs_size)

            jobs_queue_size = []
            for job in self.jobs[-2:]:
                if job.memory_request == 250:
                    jobs_queue_size.append(1)
                elif job.memory_request == 500:
                    jobs_queue_size.append(2)
                elif job.memory_request == 750:
                    jobs_queue_size.append(3)
                else:
                    print('Error')

                state.append(job.app)

            diff = 2 - len(self.jobs)
            for i in range(diff):
                jobs_queue_size.append(0)
                state.append(0)

            jobs_queue_size = np.array(jobs_queue_size)
            jobs_queue_size = to_categorical(jobs_queue_size, num_classes=4)
            jobs_queue_size = np.concatenate(jobs_queue_size)

            state = np.array(state)

            state = to_categorical(state, num_classes=3)
            state = np.concatenate(state)
            state = np.concatenate((state, nodes_space, jobs_size, jobs_queue_size))

        else:
            for node in self.nodes.values():
                if node.jobs:
                    for job in node.jobs:
                        state.append(job['job'].app)

                diff = 1 - len(node.jobs)

                for i in range(diff):
                    state.append(0)

            for job in self.buffer:
                state.append(job.app)

            diff = 1 - len(self.buffer)
            if diff > 0:
                for _ in range(diff):
                    state.append(0)

            for job in self.jobs[-3:]:
                state.append(job.app)

            diff = 3 - len(self.jobs)
            for i in range(diff):
                state.append(0)

            state = np.array(state)

            state = to_categorical(state, num_classes=4)
            state = np.concatenate(state)

            action = to_categorical(action, num_classes=7)

            state = np.concatenate((state, action))

        return state


    def update_running_jobs(self):
        """
        Decreases running time of jobs and if their time is over terminates them.
        """
        # Decrease exec time by 1 of all the jobs allocated in the nodes
        # If a Job time is over we remove it from the node
        for node in self.nodes.values():
            terminated_jobs_time = node.decrease_job_time()
            # If job is terminated append its total duration to list
            if terminated_jobs_time:
                self.jobs_total_time += terminated_jobs_time

    def allocate(self, action):
        """Given the action allocates the waiting jobs accordingly"""
        try:
            if action == 0:
                pass
            elif action == 1:
                if self.nodes["node_1"].append_job(self.buffer[0], self.nodes):
                    del self.buffer[0]
                else:
                    self.acc = True
            elif action == 2:
                if self.nodes["node_2"].append_job(self.buffer[0], self.nodes):
                    del self.buffer[0]
                else:
                    self.acc = True
            elif action == 3:
                if self.nodes["node_3"].append_job(self.buffer[0], self.nodes):
                    del self.buffer[0]
                else:
                    self.acc = True
            elif action == 4:
                if self.nodes["node_4"].append_job(self.buffer[0], self.nodes):
                    del self.buffer[0]
                else:
                    self.acc = True
            elif action == 5:
                if self.nodes["node_5"].append_job(self.buffer[0], self.nodes):
                    del self.buffer[0]
                else:
                    self.acc = True
            elif action == 6:
                if self.nodes["node_6"].append_job(self.buffer[0], self.nodes):
                    del self.buffer[0]
                else:
                    self.acc = True
            else:
                print("Action not in action-space")
        except IndexError:
            pass

    def inc_job_bff_time(self):
        """Increases the time of jobs running in the cluster"""
        for job in self.buffer:
            job.time += 1

    def reward(self):
        """
        Generate the corresponding reward in each time step, given the state an action
        Optimization goal: Average job execution time
        Reward: -1 for each job in the system (buffer and nodes)
        """
        reward = 0
        for node in self.nodes.values():
            reward -= len(node.jobs)

        for _ in self.buffer:
            reward -= 1

        for _ in self.jobs:
            reward -= 1

        if self.acc:
            reward -= 5

        self.acc = False

        return reward

    def fill_buffer(self):
        """Stores in the buffer new jobs until the size of the buffer is the desired"""
        while (len(self.buffer) < self.bff_size) and self.jobs:
            self.buffer.append(self.jobs.pop())

    def done(self):
        """returns done == True if simulation finished, False otherwise"""
        done = False
        if not (self.jobs or fc.jobs_running(self.nodes.values()) or self.buffer):
            done = True

        return done

    def node_info(self):
        """Prints in screen the allocated jobs in each node and their remaining time"""
        print("\nSimulation time " + str(self.time))
        for node in self.nodes.keys():
            print(str(node) + ":")
            for job in self.nodes[node].jobs:
                print('Job: ' + str(job['job'].job_id) + '. Time: ' + str(job['time']))

    def visualization_info(self):
        """This function appends simulation data into lists for visualization"""
        self.buffer_list.append(len(self.buffer))
        for i, node in enumerate(self.nodes.values()):
            self.n_jobs_nodes[i].append(len(node.get_jobs()))
            self.cpu_nodes_list[i].append(node.get_cpu_used())
            self.memory_nodes_list[i].append(node.get_memory_used())

    def visualize(self):
        """Visualize the data stored in the lists"""
        print("Average job duration:" + str(sum(self.jobs_total_time)/self.number_jobs))
        fc.display_buffer(self.buffer_list)
        fc.display_jobs_nodes(self.n_jobs_nodes)
        fc.display_cpu_usage(list(self.nodes.values()), self.cpu_nodes_list)
        fc.display_memory_usage(list(self.nodes.values()), self.memory_nodes_list)

    def step(self, action, info):
        """
        Takes an action as input, changes the state given the action.
        Returns next state, reward and Done if simulation finished
        """
        # Print nodes info
        if info:
            self.node_info()

        # Decrease allocated jobs times and terminate jobs if over
        self.update_running_jobs()

        # Shift to next state given the chosen action
        # self.allocate_info(action)
        self.allocate(action)

        # Append data for visualization
        self.visualization_info()

        # Increase by one the running time of jobs staying in the buffer
        self.inc_job_bff_time()

        # Obtain the reward for the previous state-action
        reward = self.reward()

        # Fill the buffer
        self.fill_buffer()

        # Obtain next state
        ob = self.observation(action)

        # Returns done = True if simulation ended
        done = self.done()

        # Increase simulation time by one
        self.time += 1

        return ob, reward, done
