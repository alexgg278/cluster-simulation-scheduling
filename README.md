# cluster-simulatuion-scheduling

## Directory and files

### Under the ***/scripts*** you can find the python files:

1. node.py: Contains the class Node()
2. job.py: Contains the class Job()
3. functions_cluster.py: Contains the different functions used in the cluster
4. run_simulation.py: Defines parameters of the simulation and runs simulation

To run the simulation just run run_simulation.py

The output are a set of timeseries plot of the performance of the cluster and the average time of execution of all the jobs during the simulation

### Under the ***/notebooks*** you can find the cluster_simulation.ipynb:

This notebook contains all the code of all the previous scripts in a single notebook.

To run it, just run all the cells.

# READ

Currently the scheduling optimization is located in the branch RL_optimization of the repo.

The files are the following:

1. node.py: Contains the class Node()
2. job.py: Contains the class Job()
3. functions_cluster.py: Contains the different functions used in the cluster and the simulation
4. run_simulation.py: Defines parameters of the simulation and runs simulation
5. parameters.py: A class that initialize all the general parameters of the simulation and of the optimization.
6. policy_network.py: A class incluiding the RL policy gradient algorithm and the methods to be optimized and to execute actions.
7. environment.py: A class containing modelling the cluster environment for the simulation (rewards, observation, step, etc..)
