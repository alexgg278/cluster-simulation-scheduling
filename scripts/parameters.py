class Parameters:

    def __init__(self):
        self.jobs_types = [{'probability': 0.5,
                            'cpu': 2,
                            'memory': 500,
                            'file_size': 2},
                           {'probability': 0.5,
                            'cpu': 2,
                            'memory': 500,
                            'file_size': 4}]

        self.nodes_types = [{'number': 1,
                             'cpu': 8,
                             'memory': 2000,
                             'bw': 1},
                            {'number': 1,
                             'cpu': 8,
                             'memory': 2000,
                             'bw': 2}]

        self.number_jobs = 20

        self.iterations = 1

        self.episodes = 5

        self.jobsets = 20

        # Discount factor
        self.gamma = 1
