"""This is the job class that model tasks or applications to be distributed in the nodes"""


class Job():
    """
    This class aims to represent jobs that are to be assigned to nodes
    The basic characteristics of nodes jobs are CPU request,
    memory request and file size to send to the master. These numbers represent the max,
    resources the job will use from the node
    """
    
    def __init__(self, job_id, cpu_request, memory_request, file_size, app, transmit=True, exec=False):
        self.job_id = job_id
        self.cpu_request = cpu_request
        self.memory_request = memory_request
        self.file_size = file_size
        self.time = 0
        self.app = app
        self.transmit = transmit
        self.exec = exec

    def reset(self):
        self.time = 0
    
    def set_cpu_request(self, cpu_request):
        """ This method sets the CPU request of the job"""
        self.cpu_request = cpu_request
        
    def set_memory_request(self, memory_request):
        """ This method sets the memory request of the job"""
        self.memory_request = memory_request
        
    def set_file_size(self, file_size):
        """
        This method sets the file size of the job
        For now the file_size is represented as standard integer units.
        More info in bw method of class Node()
        """
        self.file_size = file_size
    
    def get_job_id(self):
        """This method returns the id of the job"""
        return self.job_id
        
    def get_cpu_request(self):
        """ This method returns CPU request of the job"""
        return self.cpu_request
    
    def get_memory_request(self):
        """ This method returns the memory request"""
        return self.memory_request
    
    def get_file_size(self):
        """ This method returns the file size"""
        return self.file_size
