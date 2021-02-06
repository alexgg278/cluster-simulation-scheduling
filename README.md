# Policy gradient Scheduler

This repo contains the code and the results of two scheduling experiments performed in a simulated computing cluster. The files and the results corresponding to each experiment are contained in the following folders:

* ***/Affinity based scheduling***
  * *node.py*: Contains the node class of the simulated cluster.
  * *job.py*: Contains the job class of the simulated cluster.
  * *environment.py*: Contains the environment class (whole cluster).
  * *functions.py*: Contains all the functions used in the simulation.
  * *parameters.py*: Contains the parameters class, where the different parameters of the simulation are set.
  * *default_scheduler.py*: Contains a standard scheduler to perform as a baseline comparison with respec to our scheduler.
  * *policy_network.py*: Contains the policy gradient scheduler.
  * *main.py*: It is the file that runs the experiment.
  
* ***/Distance based scheduling***
  * *node.py*: Contains the node class of the simulated cluster.
  * *job.py*: Contains the job class of the simulated cluster.
  * *environment.py*: Contains the environment class (whole cluster).
  * *functions.py*: Contains all the functions used in the simulation.
  * *parameters.py*: Contains the parameters class, where the different parameters of the simulation are set.
  * *policy_network.py*: Contains the policy gradient scheduler.
  * *main.py*: It is the file that runs the experiment.

Additionally there are two other folders:

* ***/slides***: Contains the slides explaining the experiment. In order to understund the logic behind the experiment read them.

* ***/other***: Contains additional files.


## Description

The goal of this project was to prove that RL learning algorithms such as, policy gradient are useful to perform the task of scheduling in computing clusters, where a lot of telemetry data is generated.

For more information regarding the experiments and the followed methodology consult the slides.

### Affinity based scheduling

* Experiment schema

![alt text](/other/affinity.PNG "Affinity experiment schema")

* Results

![alt text](/other/affinity_results.PNG "Affinity experiment results")

### Distance based scheduling

* Experiment schema

![alt text](/other/distance.PNG "Distance experiment schema")

* Results

![alt text](/other/distance_results.PNG "Distance experiment results")
