Test 1:

- 2 types of nodes with same memory
- 2 types of jobs with same memory
- Different jobs types execute faster in one type of node than in the other and viceversa

        self.jobs_types = [{'probability': 0.5,
                            'cpu': 2,
                            'memory': 500,
                            'file_size': 8,
                            'transmit': 1},
                           {'probability': 0.5,
                            'cpu': 2,
                            'memory': 500,
                            'file_size': 8,
                            'transmit': 2}]

        self.nodes_types = [{'number': 1,
                             'cpu': 20,
                             'memory': 1500,
                             'bw': 1},
                            {'number': 1,
                             'cpu': 20,
                             'memory': 1500,
                             'bw': 2}]
                             
        self.number_jobs = 20

        self.iterations = 250

        self.episodes = 30

        self.jobsets = 1

        # Discount factor
        self.gamma = 1

        # Learning rate
        self.lr = 0.001

        # Layer shapes
        self.layer_shapes = [32, 32]

        self.action_space = 8
        self.state_space = 6