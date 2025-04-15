import os

from offline_rl.env.point_maze_wall_wrapper import get_wall_info

os.environ["MUJOCO_GL"] = "egl"
from pathlib import Path

import gymnasium_robotics  # pylint: disable=unused-import
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm
from omegaconf import DictConfig
from scipy.ndimage import zoom
from tensordict import TensorDict
from torchrl.envs import DoubleToFloat, TransformedEnv
from torchrl.envs.libs.gym import GymEnv

from offline_rl.nn.iql import IQLNetwork


@hydra.main(config_path=str((Path(__file__).parent.parent.parent / "configs").resolve()), config_name="eval", version_base=None)
def main(cfg: DictConfig):
    device = torch.device(cfg.model.device)

    # Load environment
    env = GymEnv(cfg.env.id, continuing_task=True, reset_target=True,
                 max_episode_steps=500, device=device, render_mode="rgb_array")
    env = TransformedEnv(env, device=device, transform=DoubleToFloat())

    # Get the PointMazeEnv
    pointmaze_env = env.unwrapped # PointMazeEnv

    # Extract maze map (bool array: True = wall, False = free space)
    maze_map = np.array(pointmaze_env.maze._maze_map) # shape: [height, width]

    # Compute extent from maze bounds
    _x_map_center = pointmaze_env.maze._x_map_center
    _y_map_center = pointmaze_env.maze._y_map_center
    _map_width = pointmaze_env.maze._map_width
    _map_height = pointmaze_env.maze._map_length
    
    x_min, x_max = (_x_map_center - _map_width / 2, _x_map_center + _map_width / 2)  # typically (-1.0, 1.0)
    y_min, y_max = (_y_map_center - _map_height / 2, _y_map_center + _map_height / 2)  # typically (-1.0, 1.0)
    extent = [x_min, x_max, y_min, y_max]

    # Load model
    model = IQLNetwork(env, cfg.model, cfg.model.device)
    ckpt = torch.load(cfg.eval.ckpt_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    print("Evaluating value function on grid...")

    tensordict = env.reset(seed=cfg.seed)
    pos_x = torch.range(x_min, x_max, 0.1).to(device)
    pos_x_rel = pos_x - _x_map_center
    pos_y = torch.range(y_min, y_max, 0.1).to(device)
    pos_y_rel = pos_y - _y_map_center
    positions = torch.cartesian_prod(pos_x_rel, pos_y_rel)
    
    velocities = torch.tensor([
        [0.0, 0.0],
        [0.1, 0.1],
        [-0.1, -0.1],
        [0.1, -0.1],
        [-0.1, 0.1]
    ], dtype=torch.float32, device=device)
    
    num_positions = positions.shape[0]
    num_velocities = velocities.shape[0]
    
    # Resize maze overlay to match grid size
    maze_overlay = maze_map.astype(np.float32)
    # Compute zoom factors to match grid_values shape
    zoom_factors = (
        len(pos_y) / maze_overlay.shape[0],
        len(pos_x) / maze_overlay.shape[1],
    )

    # Upsample using bilinear interpolation (order=1)
    maze_overlay_upsampled = zoom(maze_overlay, zoom_factors, order=0)
    # round to int
    maze_overlay_upsampled = np.round(maze_overlay_upsampled).astype(np.int32)
    #maze_overlay_upsampled = np.clip(maze_overlay_upsampled, 0, 1)  # Ensure values are in [0, 1]

    positions_expanded = positions.repeat_interleave(num_velocities, dim=0)  # [2500*5, 2]
    velocities_tiled = velocities.repeat(num_positions, 1)  # [2500*5, 2]

    states = torch.cat([positions_expanded, velocities_tiled], dim=1)  # [N, 4]

    wall_infos = torch.tensor(np.stack(
        [
            get_wall_info(_map_width - (x[0] + _x_map_center), _map_height - (x[1] + _y_map_center), maze_map, _map_width,  _map_height, 1) for x in states[:, :2]
        ]
    ))
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    clb = axes[0,0].imshow(wall_infos[...,0].reshape(-1, num_velocities).to(float).mean(-1).reshape(len(pos_x), len(pos_y)).cpu().numpy(), origin='lower', extent=extent, aspect='auto', alpha=1.0)
    fig.colorbar(clb, ax=axes[0,0], label="Wall Info")
    clb = axes[0,1].imshow(wall_infos[...,1].reshape(-1, num_velocities).to(float).mean(-1).reshape(len(pos_x), len(pos_y)).cpu().numpy(), origin='lower', extent=extent, aspect='auto', alpha=1.0)
    fig.colorbar(clb, ax=axes[0,1], label="Wall Info")
    clb = axes[0,2].imshow(wall_infos[...,2].reshape(-1, num_velocities).to(float).mean(-1).reshape(len(pos_x), len(pos_y)).cpu().numpy(), origin='lower', extent=extent, aspect='auto', alpha=1.0)
    fig.colorbar(clb, ax=axes[0,2], label="Wall Info")
    clb = axes[1,0].imshow(wall_infos[...,3].reshape(-1, num_velocities).to(float).mean(-1).reshape(len(pos_x), len(pos_y)).cpu().numpy(), origin='lower', extent=extent, aspect='auto', alpha=1.0)
    fig.colorbar(clb, ax=axes[1,0], label="Wall Info")
    clb = axes[1,2].imshow(maze_overlay_upsampled, origin='lower', extent=extent, cmap='gray', aspect='auto', alpha=0.1)
    clb = axes[0,0].imshow(maze_overlay_upsampled, origin='lower', extent=extent, cmap='gray', aspect='auto', alpha=0.1)
    clb = axes[0,1].imshow(maze_overlay_upsampled, origin='lower', extent=extent, cmap='gray', aspect='auto', alpha=0.1)
    clb = axes[0,2].imshow(maze_overlay_upsampled, origin='lower', extent=extent, cmap='gray', aspect='auto', alpha=0.1)
    clb = axes[1,0].imshow(maze_overlay_upsampled, origin='lower', extent=extent, cmap='gray', aspect='auto', alpha=0.1)
    fig.colorbar(clb, ax=axes[1,2], label="Maze Overlay")
    fig.savefig("wall_info.png")

    desired_goals = tensordict["desired_goal"].unsqueeze(0).repeat(states.shape[0], 1)
    inpt = TensorDict({
        "observation": states,
        "desired_goal": desired_goals,
        "wall_info": wall_infos
    }, batch_size=[states.shape[0]], device=device)
    
    state_values = model.value_net(inpt)["state_value"]
    mean_value_per_position = state_values.view(num_positions, num_velocities).mean(-1)
    grid_values = mean_value_per_position.view(len(pos_x), len(pos_y)).detach().cpu().numpy().T    
    grid_values_masked = grid_values * (1 - maze_overlay_upsampled)  # Mask out the walls in the grid values
    
    # compute actions
    with torch.no_grad():
        action_preds = model.actor(inpt)["action"]

    # Create vector field components
    dx_full = action_preds[:, 0].view(num_positions, num_velocities).mean(-1).view(len(pos_x), len(pos_y)).cpu().numpy().T
    dy_full = action_preds[:, 1].view(num_positions, num_velocities).mean(-1).view(len(pos_x), len(pos_y)).cpu().numpy().T
    masked_dx = dx_full * (1 - maze_overlay_upsampled)
    masked_dy = dy_full * (1 - maze_overlay_upsampled)
    # Create meshgrid for quiver plot
    X_full, Y_full = np.meshgrid(pos_x.cpu().numpy(), pos_y.cpu().numpy())

    # Downsample by a factor of 10 in both x and y (total 100)
    step = 3
    step_scale = 3
    X = X_full[::step, ::step]
    Y = Y_full[::step, ::step]
    dx = dx_full[::step, ::step]
    masked_dx = masked_dx[::step, ::step]
    dy = dy_full[::step, ::step]
    masked_dy = masked_dy[::step, ::step]

    goal = tensordict["desired_goal"].cpu().numpy()
    goal[0] = (goal[0] + _x_map_center)
    goal[1] = (goal[1] + _y_map_center)
    
    # Plotting
    # Create subplots
    fig, axs = plt.subplots(2, 3, figsize=(16, 8))

    # 1. Just the maze
    axs[0, 0].imshow(maze_overlay_upsampled, origin='lower', extent=extent, cmap='gray', aspect='auto')
    axs[0, 0].set_title("Maze Layout")
    axs[0, 0].set_xlabel("X Position")
    axs[0, 0].set_ylabel("Y Position")

    # 2. Just the unmasked value function
    im2 = axs[0, 1].imshow(grid_values, origin='lower', extent=extent, cmap='viridis', aspect='auto')
    axs[0, 1].set_title("State Value Heatmap")
    axs[0, 1].set_xlabel("X Position")
    axs[0, 1].set_ylabel("Y Position")
    fig.colorbar(im2, ax=axs[0, 1], label="Value")

    # 3. Combined maze and value function
    axs[1, 0].imshow(grid_values_masked, origin='lower', extent=extent, cmap='viridis', aspect='auto')
    axs[1, 0].imshow(maze_overlay_upsampled, origin='lower', extent=extent, cmap='gray', alpha=0.3, aspect='auto')
    axs[1, 0].scatter(goal[0], goal[1], color='red', marker="*", s=100,
                    label=f"Desired Goal {goal[0]:.2f}, {goal[1]:.2f}")
    axs[1, 0].legend()
    axs[1, 0].set_title("Heatmap with Maze Overlay")
    axs[1, 0].set_xlabel("X Position")
    axs[1, 0].set_ylabel("Y Position")

    # 4. Combined with log-scaled value function
    im4 = axs[1, 1].imshow(grid_values_masked, origin='lower', extent=extent, cmap='viridis', norm=LogNorm(), aspect='auto')
    axs[1, 1].imshow(maze_overlay_upsampled, origin='lower', extent=extent, cmap='gray', alpha=0.3, aspect='auto')
    axs[1, 1].scatter(goal[0], goal[1], color='red', marker="*", s=100,
                    label=f"Desired Goal {goal[0]:.2f}, {goal[1]:.2f}")
    axs[1, 1].legend()
    axs[1, 1].set_title("Log-Scaled Heatmap with Maze Overlay")
    axs[1, 1].set_xlabel("X Position")
    axs[1, 1].set_ylabel("Y Position")
    fig.colorbar(im4, ax=axs[1, 1], label="Log Value")
    
    # Vector field plot
    axs[0, 2].imshow(maze_overlay_upsampled, origin='lower', extent=extent, cmap='gray', alpha=0.3, aspect='auto')
    axs[0, 2].quiver(X, Y, dx, dy, scale=step_scale, scale_units='xy', width=0.0025, color='blue')
    #axs[0, 2].scatter(goal[0], goal[1], color='red', marker="*", s=100,
    #                 label=f"Desired Goal {goal[0]:.2f}, {goal[1]:.2f}")
    axs[0, 2].set_title("Policy Vector Field")
    axs[0, 2].set_xlabel("X Position")
    axs[0, 2].set_ylabel("Y Position")

    # Masked vector field plot
    axs[1, 2].imshow(maze_overlay_upsampled, origin='lower', extent=extent, cmap='gray', alpha=0.3, aspect='auto')
    axs[1, 2].quiver(X, Y, masked_dx, masked_dy, scale=step_scale, scale_units='xy', width=0.0025, color='blue')
    axs[1, 2].scatter(goal[0], goal[1], color='red', marker="*", s=100,
                     label=f"Desired Goal {goal[0]:.2f}, {goal[1]:.2f}")
    axs[1, 2].set_title("Masked Policy Vector Field")
    axs[1, 2].set_xlabel("X Position")
    axs[1, 2].set_ylabel("Y Position")
    axs[1, 2].legend()


    fig.tight_layout()
    
    # yaxis should start from top
    #plt.gca().invert_yaxis()
    
    output_path = Path(cfg.eval.output_file).with_name("value_heatmap_overlay.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    print(f"✅ Saved value heatmap with overlay to: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
