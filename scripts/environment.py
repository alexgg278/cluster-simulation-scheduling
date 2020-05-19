"""
This is the environment class that models the cluster simulation
"""
import numpy as np

import functions as fc

class Environment():

    def __init__(self, param):

        # Dictionary containing the nodes
        self.nodes = fc.create_nodes(param.nodes_types)

        # List containing the jobs to be allocated
        self.jobs_types = param.jobs_types
        self.number_jobs = param.number_jobs
        self.jobs = None

        # Buffer or job slots. Max size == 2
        self.bff_size = 2
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

    def reset(self, jobset):
        """Resets environment by resetting nodes, job list and filling the buffer with the first two jobs"""

        # Reset time
        self.time = 0

        self.jobs_total_time = []

        # Reset nodes
        for node in self.nodes.values():
            node.reset()

        # Create list of jobs
        for job in jobset:
            job.reset()

        self.jobs = jobset[:]

        # Fill the buffer with the first jobs
        for i in range(self.bff_size):
            self.buffer.append(self.jobs.pop())

        return self.observation()

    def observation(self):
        """Returns the state of the system"""

        state = []

        # get maximum values to normalize observation parameters
        max_cpu_capacity = max([node.cpu_capacity for node in self.nodes.values()])
        max_memory_capacity = max([node.memory_capacity for node in self.nodes.values()])
        max_bw = max([node.bw for node in self.nodes.values()])

        nodes_space = []
        for node in self.nodes.values():
            # state.append(node.cpu_available / max_cpu_capacity)
            if node.memory_available / max_memory_capacity == 0:
                nodes_space.append(0)
            else:
                nodes_space.append(1)
        nodes_space =

            state.append(node.memory_available / max_memory_capacity)
           # state.append(node.bw / max_bw)

        max_cpu_req = max([job['cpu'] for job in self.jobs_types])
        max_memory_req = max([job['memory'] for job in self.jobs_types])
        max_file_size = max([job['file_size'] for job in self.jobs_types])
        max_job_transmit = max([job['transmit'] for job in self.jobs_types])

        for job in self.buffer:
            # state.append(job.cpu_request / max_cpu_req)
            # state.append(job.memory_request / max_memory_req)
            # state.append(job.file_size / max_file_size)
            state.append(job.transmit / max_job_transmit)

        diff = self.bff_size - len(self.buffer)
        if diff > 0:
            for _ in range(diff):
                # state.append(0)
                # state.append(0)
                # state.append(0)
                state.append(0)

        # Add to the observation the next two jobs to come

        for job in self.jobs[-2:]:
            # state.append(job.file_size / max_file_size)
            state.append(job.transmit / max_job_transmit)

        diff = self.bff_size - len(self.jobs)
        for i in range(diff):
            # state.append(0)
            state.append(0)

        # state.append(len(self.jobs) / self.number_jobs)

        return np.array(state)

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
                if self.nodes["node_1"].append_job(self.buffer[0]):
                    del self.buffer[0]
            elif action == 1:
                if self.nodes["node_1"].append_job(self.buffer[1]):
                    del self.buffer[1]
            elif action == 2:
                if self.nodes["node_1"].check_resources(self.buffer):
                    self.nodes["node_1"].append_job(self.buffer[1])
                    self.nodes["node_1"].append_job(self.buffer[0])
                    del self.buffer[1]
                    del self.buffer[0]
            elif action == 3:
                if self.nodes["node_2"].append_job(self.buffer[0]):
                    del self.buffer[0]
            elif action == 4:
                if self.nodes["node_2"].append_job(self.buffer[1]):
                    del self.buffer[1]
            elif action == 5:
                if self.nodes["node_2"].check_resources(self.buffer):
                    self.nodes["node_2"].append_job(self.buffer[1])
                    self.nodes["node_2"].append_job(self.buffer[0])
                    del self.buffer[1]
                    del self.buffer[0]
            elif action == 6:
                if self.nodes["node_2"].check_resources([self.buffer[1]]) and self.nodes["node_1"].check_resources([self.buffer[0]]):
                    self.nodes["node_2"].append_job(self.buffer[1])
                    self.nodes["node_1"].append_job(self.buffer[0])
                    del self.buffer[1]
                    del self.buffer[0]
            elif action == 7:
                if self.nodes["node_1"].check_resources([self.buffer[1]]) and self.nodes["node_2"].check_resources([self.buffer[0]]):
                    self.nodes["node_1"].append_job(self.buffer[1])
                    self.nodes["node_2"].append_job(self.buffer[0])
                    del self.buffer[1]
                    del self.buffer[0]
            else:
                print("Action not in action-space")
        except IndexError:
            pass

    def allocate_info(self, action):
        """Given the action allocates the waiting jobs accordingly and prints info"""
        try:
            if action == 0:
                print("Scheduler: Action " + str(action))
                print("Jobs are not allocated, remain in buffer")
            elif action == 1:
                job = self.buffer.pop(0)
                self.nodes["node_1"].append_job(job)
                print("Scheduler: Action " + str(action))
                print("Job_" + str(job.job_id) + " allocated to node_1")
            elif action == 2:
                job = self.buffer.pop(1)
                self.nodes["node_1"].append_job(job)
                print("Scheduler: Action " + str(action))
                print("Job_" + str(job.job_id) + " allocated to node_1")
            elif action == 3:
                job2 = self.buffer.pop(1)
                job1 = self.buffer.pop(0)
                self.nodes["node_1"].append_job(job2)
                self.nodes["node_1"].append_job(job1)
                print("Scheduler: Action " + str(action))
                print("Job_" + str(job1.job_id) + " allocated to node_1")
                print("Job_" + str(job2.job_id) + " allocated to node_1")
            elif action == 4:
                job = self.buffer.pop(0)
                self.nodes["node_2"].append_job(job)
                print("Scheduler: Action " + str(action))
                print("Job_" + str(job.job_id) + " allocated to node_2")
            elif action == 5:
                job = self.buffer.pop(1)
                self.nodes["node_2"].append_job(job)
                print("Scheduler: Action " + str(action))
                print("Job_" + str(job.job_id) + " allocated to node_2")
            elif action == 6:
                job2 = self.buffer.pop(1)
                job1 = self.buffer.pop(0)
                self.nodes["node_2"].append_job(job1)
                self.nodes["node_2"].append_job(job2)
                print("Scheduler: Action " + str(action))
                print("Job_" + str(job1.job_id) + " allocated to node_2")
                print("Job_" + str(job2.job_id) + " allocated to node_2")
            elif action == 7:
                job2 = self.buffer.pop(1)
                job1 = self.buffer.pop(0)
                self.nodes["node_2"].append_job(job2)
                self.nodes["node_1"].append_job(job1)
                print("Scheduler: Action " + str(action))
                print("Job_" + str(job1.job_id) + " allocated to node_1")
                print("Job_" + str(job2.job_id) + " allocated to node_2")
            elif action == 8:
                job2 = self.buffer.pop(1)
                job1 = self.buffer.pop(0)
                self.nodes["node_1"].append_job(job2)
                self.nodes["node_2"].append_job(job1)
                print("Scheduler: Action " + str(action))
                print("Job_" + str(job1.job_id) + " allocated to node_2")
                print("Job_" + str(job2.job_id) + " allocated to node_1")
            else:
                print("Not valid action")
        except IndexError:
            print("Invalid action: " + str(action))

    def inc_job_bff_time(self):
        """Increases the time of jobs running in the cluster"""
        for job in self.buffer:
            job.time += 1

    def reward(self):
        """
        Generate the corresponding reward in each time step, given the state an action
        Optimization goal: Average jopb execution time
        Reward: -1 for each job in the system (buffer and nodes)
        """
        reward = 0
        for node in self.nodes.values():
            reward -= len(node.jobs)

        for _ in self.buffer:
            reward -= 1

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
        ob = self.observation()

        # Returns done = True if simulation ended
        done = self.done()

        # Increase simulation time by one
        self.time += 1

        return ob, reward, done
