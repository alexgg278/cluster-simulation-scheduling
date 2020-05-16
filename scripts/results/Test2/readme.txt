class Parameters:

    def __init__(self):
        self.jobs_types = [{'number': 4,
                            'cpu': 2,
                            'memory': 500,
                            'file_size': 16,
                            'app': 1},
                           {'number': 4,
                            'cpu': 2,
                            'memory': 500,
                            'file_size': 16,
                            'app': 2}]

        self.nodes_types = [{'number': 2,
                             'cpu': 20,
                             'memory': [500, 500],
                             'region': 1},
                            {'number': 2,
                             'cpu': 20,
                             'memory': [500, 500],
                             'region': 2}]

        self.number_jobs = self.jobs_types[0]['number'] + self.jobs_types[1]['number']

        self.iterations = 500

        self.episodes = 20

        self.jobsets = 4

        # Discount factor
        self.gamma = 1

        # Learning rate
        self.lr = 0.0005
        # Layer shapes
        self.layer_shapes = [32, 16]

        self.action_space = 5
        self.state_space = 11
