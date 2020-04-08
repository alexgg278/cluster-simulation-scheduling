"""
This is the environment class that models the cluster simulation
"""
import functions_cluster as fc

class Environment():

    def __init__(self, nodes_types, jobs_types, number_jobs):
        # Dictionary containing the nodes
        self.nodes = fc.create_nodes(nodes_types)
        # List containing the jobs to be allocated
        self.jobs_types = jobs_types
        self.number_jobs = number_jobs
        self.jobs = None
        # Buffer or job slots. Max size == 2
        self.bff_size = 2
        self.buffer = []

    def reset(self):
        """Resets environment by resetting nodes, job list and filling the buffer with the first two jobs"""

        # Reset nodes
        for node in self.nodes.values():
            node.reset()

        # Create list of jobs
        self.jobs = fc.create_jobs(self.jobs_types, self.number_jobs)

        # Fill the buffer with the first jobs
        for i in range(self.bff_size):
            self.buffer.append(self.jobs.pop())

    def observation(self):
        """Returns the state of the system"""

        state = []

        for node in self.nodes.values():
            state.append(node.cpu_available)
            state.append(node.memory_available)
            state.append(node.bw)

        for job in range(self.bff_size):
            if self.buffer[job]:
                state.append(job.cpu_request)
                state.append(job.memory_request)
                state.append(job.file_size)
            else:
                state.append(0)
                state.append(0)
                state.append(0)

        state.append(len(self.jobs))

        return state

    def next_state(self, action):
        """Given an action transitions to the next state"""
        try:
            if (action == 0):
                pass
            elif (action == 1):
                self.nodes["node_1"].append_job(self.buffer.pop(0))
            elif (action == 2):
                self.nodes["node_1"].append_job(self.buffer.pop(1))
            elif (action == 3):
                self.nodes["node_1"].append_job(self.buffer.pop(1))
                self.nodes["node_1"].append_job(self.buffer.pop(0))
            elif (action == 4):
                self.nodes["node_2"].append_job(self.buffer.pop(0))
            elif (action == 5):
                self.nodes["node_2"].append_job(self.buffer.pop(1))
            elif (action == 6):
                self.nodes["node_2"].append_job(self.buffer.pop(1))
                self.nodes["node_2"].append_job(self.buffer.pop(0))
            elif (action == 7):
                self.nodes["node_2"].append_job(self.buffer.pop(1))
                self.nodes["node_1"].append_job(self.buffer.pop(0))
            elif (action == 8):
                self.nodes["node_1"].append_job(self.buffer.pop(1))
                self.nodes["node_2"].append_job(self.buffer.pop(0))
            else:
                print("Not valid action")
        except IndexError:
            print("Invalid action: " + str(action))

    def reward(self):
        """
        Generate the corresponding reward in each time step, given the state an action
        Optimization goal: Average jopb execution time
        Reward: -1 for each job in the system (buffer and nodes)
        """

        reward = 0
        for node in self.nodes.values():
            reward -= len(node.jobs)

        for job in self.buffer():
            reward -= 1

        return reward

    def fill_buffer(self):
        """Stores in the buffer new jobs until the size of the buffer is the desired"""
        while (len(self.buffer) < self.bff_size) and self.jobs:
            self.buffer.append(self.jobs.pop())


    def done(self):
        """returns done == True if simulation finished, False otherwise"""
        done = False
        if not (self.jobs or fc.jobs_running(self.nodes.values())):
            done = True

        return done

    def step(self, action):
        """
        Takes an action as input, changes the state given the action.
        Returns next state, reward and Done if simulation finished
        """

        # action = NN(state)

        # Shift to next state given the chosen action
        self.next_state(action)

        # Fill the buffer
        self.fill_buffer()

        # Obtain the reward for the previous state-action
        reward = self.reward()

        # Obtain next state
        ob  = self.observation()

        # Returns done = True if simulation ended
        done = self.done()
