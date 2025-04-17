import argparse
from pathlib import Path

import gymnasium_robotics  # pylint: disable=unused-import
import matplotlib.pyplot as plt
import minari
import numpy as np

import offline_rl
from offline_rl.env.point_maze_wall_wrapper import get_wall_info


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize trajectories and goals from a Minari dataset.")
    parser.add_argument("--dataset_id", type=str, required=True, help="ID of the Minari dataset (e.g., PointMaze_Open-v3).")
    parser.add_argument("--max_trajectories", type=int, default=None, help="Maximum number of trajectories to plot (default: all).")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load dataset
    dataset = minari.load_dataset(args.dataset_id)
    print(f"Loaded dataset '{args.dataset_id}' with {len(dataset)} episodes.")

    fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharex=True, sharey=True)
    max_trajectories = args.max_trajectories or len(dataset)
    
    
    #num_episodes = len(dataset)
    #avg_steps = np.mean([len(episode.rewards) for episode in dataset])
    #avg_reward = np.mean([episode.rewards.sum() for episode in dataset])
    #
    #print(f"Number of episodes: {num_episodes}")
    #print(f"Average number of steps per episode: {avg_steps:.2f}")
    #print(f"Average reward per episode: {avg_reward:.2f}")
    
    env = dataset.recover_environment()
    pointmaze_env = env.unwrapped # PointMazeEnv
    maze_map = np.array(pointmaze_env.maze._maze_map) # shape: [height, width]
    height, width = maze_map.shape
    extent = (0, width, height, 0)
    
    print(f"Map: \n{maze_map}")
    
    x_map_center = pointmaze_env.maze._x_map_center
    y_map_center = pointmaze_env.maze._y_map_center
    
    if max_trajectories < len(dataset):
        sampled_indices = np.random.choice(len(dataset), max_trajectories, replace=False)
        dataset = [dataset[i] for i in sampled_indices]
    
    axes[0].imshow(maze_map, extent=extent, origin="lower", cmap='gray', alpha=0.3, aspect='auto')
    axes[1].imshow(maze_map, extent=extent, origin="lower",  cmap='gray', alpha=0.3, aspect='auto')
    axes[2].imshow(maze_map, extent=extent,  origin="lower", cmap='gray', alpha=0.3, aspect='auto')
    for i, episode in enumerate(dataset):
        obs = episode.observations
        infos = episode.infos

        # Extract 2D positions
        pos = np.array([o[:2] for o in obs["observation"]])
        
        pos[:, 0] += x_map_center
        pos[:, 1] += y_map_center
        
        # Plot trajectory in the first subplot
        color = plt.cm.tab20c(i % 20)  # Use a colormap to ensure consistent colors
        axes[0].plot(pos[:, 0], pos[:, 1], alpha=0.6, label=f'Traj {i+1}', color=color)

        # Plot desired goal in the second subplot
        goal = obs["desired_goal"]
        goal[:, 0] += x_map_center
        goal[:, 1] += y_map_center
        goal[:, 1] = goal[:, 1]  # Flip y-axis to match image coordinates
        axes[1].scatter(goal[:, 0], goal[:, 1], color=color, marker="*", s=100, label=f'Goal {i+1}')
        
        
        #if "wall_info" in obs:
        #    wall_info = obs["wall_info"]
        #    
        #    rand_idx = np.random.randint(0, len(wall_info))
        #    
        #    wall_info = wall_info[rand_idx]
        #    
        #    pos = obs["observation"][rand_idx][:2]
        #    pos[0] += x_map_center
        #    pos[1] += y_map_center
        #    corrected = get_wall_info(pos[0], pos[1], maze_map, width, height, pointmaze_env.maze._maze_size_scaling)
        #    axes[2].scatter(pos[0], pos[1], color=color, marker="o", s=100, label=f'({pos[0]:0.2f},{pos[1]:0.2f}), Wall {wall_info} {corrected}')
           
        print(f"Episode {i}: {len(pos)} steps, goal: {obs['desired_goal'][-1]}")
        if i + 1 >= max_trajectories:
            break

    # Configure subplot for trajectories
    axes[0].set_title("Trajectories")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].grid(True)

    # Configure subplot for goals
    axes[1].set_title("Goals")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].grid(True)

    # Add legends if the number of trajectories is small
    if max_trajectories <= 10:
        axes[0].legend(loc="upper right", fontsize="small")
        axes[1].legend(loc="upper right", fontsize="small")

    plt.tight_layout()

    plt.title(f"Trajectories and Goals from {args.dataset_id}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.tight_layout()

    if max_trajectories <= 10:
        plt.legend(loc="upper right", fontsize="small")

    path = f"plots/{args.dataset_id}_trajectories.png"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)

if __name__ == "__main__":
    main()
