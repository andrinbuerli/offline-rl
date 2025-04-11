# Offline RL from random trajectories

This repository contains the code to train optimal policies from random trajectories using offline reinforcement learning.


## Datasets

The data is collected from the [PointMaze](https://robotics.farama.org/envs/maze/point_maze/) environment, which contains an open arena with only perimeter walls. The agent uses a uniform random sampling or a PD controller (fetched from [here](https://minari.farama.org/datasets/D4RL/pointmaze/open-v2/)) to follow a path of waypoints generated with QIteration until it reaches the goal. The task is continuing which means that when the agent reaches the goal the environment generates a new random goal without resetting the location of the agent. The reward function is sparse, only returning a value of 1 if the goal is reached, otherwise 0. To add variance to the collected paths random noise is added to the actions taken by the agent.

#### Renderings
| Env | 100k | 1M | PD Controller |
| --- | ----------- | --------- | ----- |
| Open | <img src="assets/pointmaze_open_random-v2_trajectories.png" alt="Demo" width="250"/> | <img src="assets/pointmaze_open_random-v3_trajectories.png" alt="Demo" width="250"/> | <img src="assets/open-v2_trajectories.png" alt="Demo" width="250"/> |
| Medium | <img src="assets/pointmaze_medium_random-v1_trajectories.png" alt="Demo" width="250"/> | <img src="assets/pointmaze_medium_random-v2_trajectories.png" alt="Demo" width="250"/> | <img src="assets/medium-v2_trajectories.png" alt="Demo" width="250"/> |
| Large | <img src="assets/pointmaze_large_random-v1_trajectories.png" alt="Demo" width="250"/> | <img src="assets/pointmaze_large_random-v2_trajectories.png" alt="Demo" width="250"/> | <img src="assets/large-v2_trajectories.png" alt="Demo" width="250"/> |

## Results

### PointMaze_Open-v3 (episode=500)

| Algorithm | Dataset Size | Dataset Sampling | Eval Reward |
| --------- | ----- | ----- | ----- |
| Uniform Random | - | - |  0.1 |
| IQL | 100k | Uniform Random | 4.7 |
| IQL | 1M | Uniform Random | 8.0 |
| IQL | 10M | Uniform Random | 8.1 |
| IQL | 1M | PD Controller | 8.6 |

We can see that the random uniform baseline achieves a cummulative reward of 0.1, while the IQL algorithm achieves an max reward of 8.1 with 10M, 8.0 with 1M and 4.7 with 100k step which where sampled using the uniform random baseline. Thus the algorithm learns a approximate optimal policy from the from (very) suboptimal trajectories. We can also see that the PD controller achieves a max reward of 8.6 with 1M steps, which is better than the uniform random sampling.

#### Renderings
| Uniform Random | IQL 100k | IQL 1M |
| ----------- | --------- | ----- |
| <img src="assets/PointMaze_Open_random_uniform.gif" alt="Demo" width="200"/> | <img src="assets/PointMaze_Open_IQL_100k.gif" alt="Demo" width="200"/> | <img src="assets/PointMaze_Open_IQL_1M.gif" alt="Demo" width="200"/> |

### PointMaze_Medium-v3 (episode=500)

| Algorithm | Dataset Size | Dataset Sampling | Eval Reward |
| --------- | ----- | ----- | ----- |
| Uniform Random | - | - |  0.1 |
| IQL | 100k | Uniform Random | 0.1 |
| IQL | 1M | Uniform Random | 2.3|
| IQL | 10M | Uniform Random | 2.4 |
| IQL | 1M | PD Controller | 2.7 |


#### Renderings
| Uniform Random | IQL 100k | IQL 1M |
| ----------- | --------- | ----- |
| <img src="assets/PointMaze_Medium_random_uniform.gif" alt="Demo" width="200"/> | <img src="assets/PointMaze_Medium_IQL_100k.gif" alt="Demo" width="200"/> | <img src="assets/PointMaze_Medium_IQL_1M.gif" alt="Demo" width="200"/> |


### PointMaze_Large-v3 (episode=500)

| Algorithm | Dataset Size | Dataset Sampling | Eval Reward |
| --------- | ----- | ----- | ----- |
| Uniform Random | - | - |  0.0 |
| IQL | 100k | Uniform Random | 0.2 |
| IQL | 1M | Uniform Random |  0.4 |
| IQL | 10M | Uniform Random | 0.4 |
| IQL | 1M | PD Controller | 0.1 |


#### Renderings
| Uniform Random | IQL 100k | IQL 1M |
| ----------- | --------- | ----- |
| <img src="assets/PointMaze_Large_random_uniform.gif" alt="Demo" width="200"/> | <img src="assets/PointMaze_Large_IQL_100k.gif" alt="Demo" width="200"/> | <img src="assets/PointMaze_Large_IQL_1M.gif" alt="Demo" width="200"/> |

## Setup
Install GNU make: https://www.gnu.org/software/make/.

Make sure that the default python interpreter is python >=3.10.

Setup env with
```
make setup
```

Further inspect CLI using 
```
make help
```