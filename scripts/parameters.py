class Parameters:

    def __init__(self):
        # Normal distribution
        self.mean = 16
        self.std = 6

        self.jobs_types = [{'number': 10,
                            'cpu': 2,
                            'memory': 750,
                            'file_size': 20,
                            'app': 1,
                            'distr': None,
                            'distr_mem': []},
                           {'number': 10,
                            'cpu': 2,
                            'memory': 750,
                            'file_size': 20,
                            'app': 2,
                            'distr': None,
                            'distr_mem': []}]

        self.nodes_types = [{'number': 2,
                             'cpu': 20,
                             'memory': [500, 500],
                             'region': 1},
                            {'number': 2,
                             'cpu': 20,
                             'memory': [500, 500],
                             'region': 2}]

        self.number_jobs = self.jobs_types[0]['number'] + self.jobs_types[1]['number']

        self.iterations = 350

        self.episodes = 20

        self.jobsets = 5

        # Discount factor
        self.gamma = 1

        # Learning rate
        self.lr = 0.001
        # Layer shapes
        self.layer_shapes = [32, 16]

        self.action_space = 5
        self.state_space = 21

        # Early stoping patience
        self.patience = 100

