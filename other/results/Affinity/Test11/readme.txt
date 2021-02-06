class Parameters:

    def __init__(self):
        # Normal distribution
        self.mean = 16
        self.std = 6

        self.jobs_types = [{'probability': 0.5,
                            'cpu': 2,
                            'memory': 500,
                            'file_size': 16,
                            'transmit': 1,
                            'distr': None,
                            'distr_mem': [250, 500, 750]},
                           {'probability': 0.5,
                            'cpu': 2,
                            'memory': 500,
                            'file_size': 16,
                            'transmit': 2,
                            'distr': None,
                            'distr_mem': [250, 500, 750]}]

        self.nodes_types = [{'number': 1,
                             'cpu': 20,
                             'memory': 1500,
                             'bw': 1},
                            {'number': 1,
                             'cpu': 20,
                             'memory': 1500,
                             'bw': 2}]

        self.number_jobs = 40

        self.iterations = 200

        self.episodes = 20

        self.jobsets = 2

        # Discount factor
        self.gamma = 1

        # Learning rate
        self.lr = 0.0005

        # Layer shapes
        self.layer_shapes = [128, 64, 32]

        self.action_space = 9
        self.state_space = 42

