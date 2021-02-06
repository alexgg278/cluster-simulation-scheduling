"""This is the Node class which models nodes in the cluster"""
import functions as fn

class Node():
    """
    This class aims to represente computing nodes of a cluster
    The basic properties of the nodes are CPU, memory and bandwidth (BW)
    BW is considered the capacity of data transmision between the node and an hypothetic node where data is sent
    """
    
    def __init__(self, node_id, cpu_capacity, memory_capacity, region, bw=0):
        self.node_id = node_id
        self.cpu_capacity = cpu_capacity
        self.memory_capacity = memory_capacity
        self.bw = bw
        self.cpu_available = cpu_capacity
        self.memory_available = memory_capacity
        self.cpu_used = 0
        self.memory_used = 0
        self.jobs = []
        self.region = region

    def reset(self):
        self.cpu_available = self.cpu_capacity
        self.memory_available = self.memory_capacity
        self.cpu_used = 0
        self.memory_used = 0
        self.jobs = []

    def set_cpu_capacity(self, cpu_capacity):
        """
        This method sets the total CPU capacity of the node
        The CPU units are in milicores: 1000m = 1core
        Whenever this method is call the available CPU gets to 100%. The jobs are released
        """
        if cpu_capacity - self.cpu_used < 0:
            print('CPU resources cannot be reduced')
        else:
            self.cpu_capacity = cpu_capacity
            self.update_available_resources()
    
    def set_memory_capacity(self, memory_capacity):
        """
        This method sets the total memory capacity of the node
        The memory units are MB: 1GB = 1000MB = 1000000KB
        Whenever this method is call the available CPU gets to 100%. The jobs are released
        """
        if memory_capacity - self.memory_used < 0:
            print('Memory resources cannot be reduced')
        else:
            self.memory_capacity = memory_capacity
            self.update_available_resources()
        
    def set_bw(self, bw):
        """
        This method sets the total bw capacity of the link b
        For now the bandwidth is represented as the number of file_size units
        the node is capable to consume in one time step
        """
        self.bw = bw
        
    def get_node_id(self):
        """Returns node ID"""
        return self.node_id
        
    def get_cpu_capacity(self):
        """This method returns the cpu capacity of the node"""
        return self.cpu_capacity
    
    def get_memory_capacity(self):
        """This method returns the memory capcity of the node"""
        return self.memory_capacity
    
    def get_bw(self):
        """This method returns the bw"""
        return self.bw
    
    def get_cpu_used(self):
        """This method returns the used CPU"""
        return self.cpu_used

    def get_cpu_ratio(self):
        """This method returns the used CPU"""
        return self.cpu_used / self.cpu_capacity
        
    def get_memory_used(self):
        """This method returns the used memory"""
        return self.memory_used

    def get_memory_ratio(self):
        """This method returns the used CPU"""
        return self.memory_used / self.memory_capacity
    
    def get_cpu_available(self):
        """This method returns the available CPU"""
        return self.cpu_available
    
    def get_memory_available(self):
        """This method returns the available memory"""
        return self.memory_available
    
    def get_jobs(self):
        """
        This method returns a list of dicts containing the jobs allocated in the nodes
        and their remaining time of execution in the node
        """
        return self.jobs
    
    def update_available_resources(self):
        """This method updates the available resources"""
        self.cpu_available = self.cpu_capacity - self.cpu_used
        self.memory_available = self.memory_capacity - self.memory_used
        
    def consume_resources(self, job):
        """
        This method decreases the available cpu and memory
        based on the appended job CPU and memory requests.
        """
        self.cpu_used += job.get_cpu_request()
        self.memory_used += job.get_memory_request()
        self.update_available_resources()
        
    def release_resources(self, job):
        """
        This method calculates and returns the file transfer time
        given the file size and the bw of the node
        """
        self.cpu_used -= job.get_cpu_request()
        self.memory_used -= job.get_memory_request()
        self.update_available_resources()
        
    def transfer_duration(self, job):
        """
        This method calculates and returns the file transfer time
        given the file size and the bw of the node

        """
        if job.transmit == self.bw:
            total_transfer_duration = job.get_file_size() / 2
        else:
            total_transfer_duration = job.get_file_size()
        return total_transfer_duration

        """
        if job.transmit == True:
            total_transfer_duration = job.get_file_size() / self.bw
        else:
            total_transfer_duration = job.get_file_size()
        return total_transfer_duration
        """

    def check_resources(self, jobs):
        """
        This method returns a False if the node has not enough available resources to allocate the job.
        Returns True otherwise
        """
        total_cpu_request = 0
        total_memory_request = 0
        x = jobs
        for job in jobs:
            total_cpu_request += job.get_cpu_request()
            total_memory_request += job.get_memory_request()
        cpu_resources = total_cpu_request > self.get_cpu_available()
        memory_resources = total_memory_request > self.get_memory_available()
        if cpu_resources or memory_resources:
            return False
        else:
            return True
    
    def check_jobs(self):
        """
        Returns True if the node contain running jobs
        Returns False if the node does not contain running jobs
        """
        flag = True
        if not self.jobs:
            flag = False
            
        return flag  
        
    def decrease_job_time(self):
        """
        This method decreases the duration of all the jobs allocated in that node by one
        If the remaining time is 0, the job is terminated from the node
        """
        jobs_copy = self.jobs[:]
        job_terminated_time = []
        for job in jobs_copy:
            if job['job'].exec == True:
                if job['time'] <= 1:
                    self.terminate_job(job)
                    # Increase job duration by 1 and return its total time since job is terminated
                    job['job'].exec = False
                    job['job'].time += 1
                    job_terminated_time.append(job['job'].time)

        for job in self.jobs:
            job['job'].time += 1
            if job['job'].exec == True:
                job['time'] -= 1

        return job_terminated_time

    def append_job(self, job, nodes):
        """
        This method appends the job passed as an argument to the node.
        The available resources in the node are updated.
        The transfer time of the file attached to job is calculated and returned
        """
        if self.check_resources([job]):
            total_transfer_duration = fn.find_jobs_2(nodes, job, self)
            self.jobs.append({
                'job': job,
                'time': total_transfer_duration
            })
            self.consume_resources(job)   
            return True
        else:
            alert = 'The job ' + str(job.get_job_id()) + ' cannot be allocated. '
            alert += 'There is not enough resources in the node'
            # print(alert)
            return False

    def terminate_job(self, job):
        """
        This method releases the specified job running in the node
        The available resources in the node are updated
        """
        self.jobs.remove(job)
        self.release_resources(job['job'])
