import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import minari
import numpy as np


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

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    max_trajectories = args.max_trajectories or len(dataset)

    for i, episode in enumerate(dataset):
        obs = episode.observations
        infos = episode.infos

        # Extract 2D positions
        pos = np.array([o[:2] for o in obs["observation"]])

        # Plot trajectory in the first subplot
        color = plt.cm.tab20c(i % 10)  # Use a colormap to ensure consistent colors
        axes[0].plot(pos[:, 0], pos[:, 1], alpha=0.6, label=f'Traj {i+1}', color=color)

        # Plot desired goal in the second subplot
        goal = obs["desired_goal"]
        axes[1].scatter(goal[:, 0], goal[:, 1], color=color, marker="*", s=100, label=f'Goal {i+1}')
            
        print(f"Episode {i}: {len(pos)} steps, goal: {obs['desired_goal'][-1]}")
        if i + 1 >= max_trajectories:
            break

    # Configure subplot for trajectories
    axes[0].set_title("Trajectories")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].axis("equal")
    axes[0].grid(True)

    # Configure subplot for goals
    axes[1].set_title("Goals")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].axis("equal")
    axes[1].grid(True)

    # Add legends if the number of trajectories is small
    if max_trajectories <= 10:
        axes[0].legend(loc="upper right", fontsize="small")
        axes[1].legend(loc="upper right", fontsize="small")

    plt.tight_layout()

    plt.title(f"Trajectories and Goals from {args.dataset_id}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()

    if max_trajectories <= 10:
        plt.legend(loc="upper right", fontsize="small")

    path = f"plots/{args.dataset_id}_trajectories.png"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)

if __name__ == "__main__":
    main()
