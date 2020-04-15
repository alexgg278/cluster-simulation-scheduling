class Parameters:

    def __init__(self):
        self.jobs_types = [{'probability': 0.2,
                            'cpu': 2,
                            'memory': 500,
                            'file_size': 2,
                            'transmit': False},
                           {'probability': 0.8,
                            'cpu': 2,
                            'memory': 500,
                            'file_size': 8,
                            'transmit': True}]

        self.nodes_types = [{'number': 1,
                             'cpu': 20,
                             'memory': 1500,
                             'bw': 1},
                            {'number': 1,
                             'cpu': 20,
                             'memory': 1000,
                             'bw': 2}]

        self.number_jobs = 15

        self.iterations = 30

        self.episodes = 10

        self.jobsets = 5

        # Discount factor
        self.gamma = 0.9

        # Learning rate
        self.lr = 0.001

        # Layer shapes
        self.layer_shapes = [32, 32]

        self.action_space = 8
        self.state_space = 10

