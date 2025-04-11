# Offline RL from random trajectories

This repository contains the code to train optimal policies from random trajectories using offline reinforcement learning.


## Datasets

Datasets where either generated using uniform random sampling or using a PD controller (fetched from [here](https://minari.farama.org/datasets/D4RL/pointmaze/open-v2/)). 

#### Renderings
| 100k | 1M | PD Controller |
| ----------- | --------- | ----- |
| <img src="assets/pointmaze_open_random-v2_trajectories.png" alt="Demo" width="250"/> | <img src="assets/pointmaze_open_random-v3_trajectories.png" alt="Demo" width="250"/> | <img src="assets/open-v2_trajectories.png" alt="Demo" width="250"/> |

## Results

| Environment | Algorithm | Dataset Size | Dataset Sampling | Eval Reward |
| ----------- | --------- | ----- | ----- | ----- |
| PointMaze_Open-v3 (episode=500) | Uniform Random | - | - |  0.1 |
| PointMaze_Open-v3 (episode=500) | IQL | 100k | Uniform Random | 4.7 |
| PointMaze_Open-v3 (episode=500) | IQL | 1M | Uniform Random | 8.0 |
| PointMaze_Open-v3 (episode=500) | IQL | 10M | Uniform Random | 8.1 |
| PointMaze_Open-v3 (episode=500) | IQL | 1M | PD Controller | 8.6 |

We can see that the random uniform baseline achieves a cummulative reward of 0.1, while the IQL algorithm achieves an max reward of ? with 10M, 8.0 with 1M and 4.7 with 100k step which where sampled using the uniform random baseline. Thus the algorithm learns a approximate optimal policy from the from (very) suboptimal trajectories. We can also see that the PD controller achieves a max reward of 8.6 with 1M steps, which is better than the uniform random sampling.

#### Renderings
| Uniform Random | IQL 100k | IQL 1M |
| ----------- | --------- | ----- |
| <img src="assets/PointMaze_Open_random_uniform.gif" alt="Demo" width="200"/> | <img src="assets/PointMaze_Open_IQL_100k.gif" alt="Demo" width="200"/> | <img src="assets/PointMaze_Open_IQL_1M.gif" alt="Demo" width="200"/> |
 

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