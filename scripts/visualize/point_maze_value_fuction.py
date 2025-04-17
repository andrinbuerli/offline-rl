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
    
    extent = [0, _map_width, 0, _map_height]

    # Load model
    model = IQLNetwork(env, cfg.model, cfg.model.device)
    ckpt = torch.load(cfg.eval.ckpt_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    print("Evaluating value function on grid...")

    tensordict = env.reset(seed=cfg.seed)
    step = 0.1
    pos_x = torch.range(0, _map_width - step, step).to(device)
    pos_x_rel = pos_x - _x_map_center
    pos_y = torch.range(0, _map_height - step, step).to(device)
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
        int(len(pos_y) / maze_overlay.shape[0]),
        int(len(pos_x) / maze_overlay.shape[1]),
    )
    
    #import pdb; pdb.set_trace()

    #interleave maze overlay
    maze_overlay_upsampled_mask = torch.tensor(maze_overlay).repeat_interleave(zoom_factors[0], dim=0).repeat_interleave(zoom_factors[1], dim=1).numpy()
    # round to int
    maze_overlay_upsampled_mask = np.round(maze_overlay_upsampled_mask).astype(np.int32) #[::-1]

    positions_expanded = positions.repeat_interleave(num_velocities, dim=0)  # [2500*5, 2]
    velocities_tiled = velocities.repeat(num_positions, 1)  # [2500*5, 2]

    states = torch.cat([positions_expanded, velocities_tiled], dim=1)  # [N, 4]

    flip = not env.spec.id.__contains__("Medium")
    wall_infos = torch.tensor(np.stack(
        [
            get_wall_info((x[0] + _x_map_center), (x[1] + _y_map_center), maze_map, _map_width,  _map_height, 1, flip_y=flip) for x in states[:, :2]
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
    clb = axes[1,2].imshow(maze_overlay_upsampled_mask, origin='lower', extent=extent, cmap='gray', aspect='auto', alpha=0.1)
    clb = axes[0,0].imshow(maze_overlay_upsampled_mask, origin='lower', extent=extent, cmap='gray', aspect='auto', alpha=0.1)
    clb = axes[0,1].imshow(maze_overlay_upsampled_mask, origin='lower', extent=extent, cmap='gray', aspect='auto', alpha=0.1)
    clb = axes[0,2].imshow(maze_overlay_upsampled_mask, origin='lower', extent=extent, cmap='gray', aspect='auto', alpha=0.1)
    clb = axes[1,0].imshow(maze_overlay_upsampled_mask, origin='lower', extent=extent, cmap='gray', aspect='auto', alpha=0.1)
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
    grid_values_masked = grid_values * (1 - maze_overlay_upsampled_mask)  # Mask out the walls in the grid values
    
    # compute actions
    with torch.no_grad():
        action_preds = model.actor(inpt)["action"]

    # Create vector field components
    dx_full = action_preds[:, 0].view(num_positions, num_velocities).mean(-1).view(len(pos_x), len(pos_y)).cpu().numpy().T
    dy_full = action_preds[:, 1].view(num_positions, num_velocities).mean(-1).view(len(pos_x), len(pos_y)).cpu().numpy().T
    masked_dx = dx_full * (1 - maze_overlay_upsampled_mask)
    masked_dy = dy_full * (1 - maze_overlay_upsampled_mask)
    # Create meshgrid for quiver plot
    X_full, Y_full = np.meshgrid(pos_x.cpu().numpy(), pos_y.cpu().numpy())

    # Downsample by a factor of 10 in both x and y (total 100)
    skip = 2 
    step_scale = 4
    print(f"Downsampling by a factor of {skip} in both x and y (total {step_scale})")
    X = X_full[::skip, ::skip]
    Y = Y_full[::skip, ::skip]
    dx = dx_full[::skip, ::skip]
    masked_dx = masked_dx[::skip, ::skip]
    dy = dy_full[::skip, ::skip]
    masked_dy = masked_dy[::skip, ::skip]

    goal = tensordict["desired_goal"].cpu().numpy()
    goal[0] = goal[0] + _x_map_center
    goal[1] = goal[1] + _y_map_center
    
    # Plotting
    # Create subplots
    fig, axs = plt.subplots(2, 3, figsize=(16, 8))

    # 1. Just the maze
    axs[0, 0].imshow(maze_map, origin='lower', extent=extent, cmap='gray', aspect='auto')
    
    rand_idx = np.random.randint(0, len(wall_infos))
    wall_info = inpt["wall_info"].cpu().numpy()[rand_idx]
    pos = inpt["observation"].cpu().numpy()[rand_idx][:2]
    pos[0] = pos[0] + _x_map_center
    pos[1] = pos[1] + _y_map_center
    axs[0,0].scatter(pos[0], pos[1], color='blue', marker="o", s=100,
                    label=f"Rand Position {pos[0]:.2f}, {pos[1]:.2f}, wall_info: {wall_info}")
    axs[0,0].legend()
    axs[0, 0].set_title("Maze Layout")
    axs[0, 0].set_xlabel("X Position")
    axs[0, 0].set_ylabel("Y Position")

    # 2. Just the unmasked value function
    im2 = axs[0, 1].imshow(grid_values, extent=extent, origin="lower", cmap='viridis', aspect='auto')
    axs[0, 1].set_title("State Value Heatmap")
    axs[0, 1].set_xlabel("X Position")
    axs[0, 1].set_ylabel("Y Position")
    fig.colorbar(im2, ax=axs[0, 1], label="Value")

    # 3. Combined maze and value function
    axs[1, 0].imshow(grid_values_masked, extent=extent, origin="lower", cmap='viridis', aspect='auto')
    axs[1, 0].imshow(maze_map, origin='lower', extent=extent, cmap='gray', alpha=0.3, aspect='auto')
    axs[1, 0].scatter(goal[0], goal[1], color='red', marker="*", s=100,
                    label=f"Desired Goal {goal[0]:.2f}, {goal[1]:.2f}")
    axs[1, 0].legend()
    axs[1, 0].set_title("Heatmap with Maze Overlay")
    axs[1, 0].set_xlabel("X Position")
    axs[1, 0].set_ylabel("Y Position")

    # 4. Combined with log-scaled value function
    im4 = axs[1, 1].imshow(grid_values_masked, extent=extent, origin="lower", cmap='viridis', norm=LogNorm(), aspect='auto')
    axs[1, 1].imshow(maze_map, origin='lower', extent=extent, cmap='gray', alpha=0.3, aspect='auto')
    axs[1, 1].scatter(goal[0], goal[1], color='red', marker="*", s=100,
                    label=f"Desired Goal {goal[0]:.2f}, {goal[1]:.2f}")
    axs[1, 1].legend()
    axs[1, 1].set_title("Log-Scaled Heatmap with Maze Overlay")
    axs[1, 1].set_xlabel("X Position")
    axs[1, 1].set_ylabel("Y Position")
    fig.colorbar(im4, ax=axs[1, 1], label="Log Value")
    
    # Vector field plot
    axs[0, 2].imshow(maze_map, origin='lower', extent=extent, cmap='gray', alpha=0.3, aspect='auto')
    axs[0, 2].quiver(X, Y, dx, dy, scale=step_scale, scale_units='xy', width=0.0025, color='blue')
    #axs[0, 2].scatter(goal[0], goal[1], color='red', marker="*", s=100,
    #                 label=f"Desired Goal {goal[0]:.2f}, {goal[1]:.2f}")
    axs[0, 2].set_title("Policy Vector Field")
    axs[0, 2].set_xlabel("X Position")
    axs[0, 2].set_ylabel("Y Position")

    # Masked vector field plot
    axs[1, 2].imshow(maze_map, origin='lower', extent=extent, cmap='gray', alpha=0.3, aspect='auto')
    axs[1, 2].quiver(X, Y, masked_dx, masked_dy, scale=step_scale, scale_units='xy', width=0.0025, color='blue')
    axs[1, 2].scatter(goal[0], goal[1], color='red', marker="*", s=100,
                     label=f"Desired Goal {goal[0]:.2f}, {goal[1]:.2f}")
    #axs[1, 2].imshow(maze_overlay_upsampled_mask.T, origin='lower', extent=extent, cmap='gray', alpha=0.5, aspect='auto')
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
