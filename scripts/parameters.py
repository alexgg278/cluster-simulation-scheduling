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

        self.number_jobs = 40

        self.iterations = 100

        self.episodes = 20

        self.jobsets = 1

        # Discount factor
        self.gamma = 1

        # Learning rate
        self.lr = 0.001

        # Layer shapes
        self.layer_shapes = [32, 32]
