import argparse
import os

os.environ["MUJOCO_GL"] = "egl"
from pathlib import Path

import gymnasium_robotics  # pylint: disable=unused-import
import hydra
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from torchrl.envs import DoubleToFloat, TransformedEnv
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type

from offline_rl.nn.iql import IQLNetwork


class ConcatenateObsGoal(torch.nn.Module):
    def forward(self, obs, goal):
        return torch.cat([obs, goal], dim=-1)


def load_policy(cfg: DictConfig):
    # Build environment
    env = GymEnv(cfg.env.id, continuing_task=True, reset_target=True, max_episode_steps=cfg.eval.max_steps_per_episode, device=cfg.model.device, render_mode="rgb_array")
    env = TransformedEnv(env, device=cfg.model.device, transform=DoubleToFloat())

    algo =  cfg.model.get("algorithm", "iql")

    if algo == "iql":
        model = IQLNetwork(env, cfg.model, cfg.model.device)
        
        # Load weights
        print(f"Total parameter sum in actor: {sum(p.sum() for p in model.parameters())}")
        ckpt = torch.load(cfg.eval.ckpt_path, map_location=cfg.model.device)

        model.load_state_dict(ckpt)
        print(f"Total parameter sum after loading: {sum(p.sum() for p in model.parameters())}")
        model.eval()
        
        actor = model.actor
        #env.reset(seed=42)
        #episode_rewards = []
        #for _ in range(10):
        #    eval_td = env.rollout(max_steps=500, policy=model.actor, auto_cast_to_device=True)
        #    episode_rewards.append(eval_td["next", "reward"].sum().item())
        #print(f"Average episode reward: {np.mean(episode_rewards)}")
    elif algo == "random":
        def random_actor(args):
            args["action"] = env.action_spec.sample()[None]
            return args
        actor = random_actor
        
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
    
    return env, actor

@hydra.main(config_path=str((Path(__file__).parent.parent / "configs").resolve()), config_name="eval", version_base=None)
def main(cfg: DictConfig):
    device = torch.device(cfg.model.device)
    
    env, actor = load_policy(cfg)

    # Run rollout
    frames = []
    tensordict = env.reset(seed=123)
    done = tensordict["done"].item()
    
    pointmaze_env = env.env.unwrapped
    _x_map_center = pointmaze_env.maze._x_map_center
    _y_map_center = pointmaze_env.maze._y_map_center
    maze_map =  np.array(pointmaze_env.maze._maze_map)
    maze_width = maze_map.shape[1]
    maze_height = maze_map.shape[0]
    step = 0

    num_episodes = cfg.eval.num_episodes
    
    all_observations = []
    all_desired_goals = []
    all_rewards = []
    episode = 0
    rng = np.random.default_rng(cfg.seed)
    env.reset(seed=int(rng.uniform(0, int(1e10))))
    print(f"Running {num_episodes} episodes")
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        while episode < num_episodes:
            observations = []
            desired_goals = []
            rewards = []
            while not done:
                frame = env.render()
                frames.append(frame)

                tensordict = tensordict.to(device).float()
                action_dict = actor(tensordict.unsqueeze(0)).squeeze(0)
                
                step_input = TensorDict({"action": action_dict["action"]}, batch_size=[])
                tensordict = env.step(step_input)
                tensordict = tensordict["next"]
                done = tensordict["done"].item()
                terminated = tensordict["terminated"].item()
                truncated = tensordict["truncated"].item()
                
                observations.append(tensordict["observation"].cpu().numpy())
                desired_goals.append(tensordict["desired_goal"].cpu().numpy())
                rewards.append(tensordict["reward"].cpu().numpy())
                
                step += 1
                #print(f"Step {step}: Action: {step_input['action']}, Obs: {tensordict['observation']}, Done: {done}, Terminated: {terminated}, Truncated: {truncated}", end="\r")
            episode += 1
            print(f"Episode {episode}/{num_episodes} finished after {step} steps with reward {np.sum(rewards)}")
            env.reset(seed=int(rng.uniform(0, int(1e10))))
            done = False
            
            all_observations.append(observations)
            all_desired_goals.append(desired_goals)
            all_rewards.append(rewards)

    mean_cummulative_reward = np.mean([np.sum(np.array(rewards)) for rewards in all_rewards])
    print(f"Cumulative reward: {mean_cummulative_reward}")

    # Animate and save
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(frames[0])
    ax.axis("off")
    plt.tight_layout()

    def update(frame):
        im.set_array(frame)
        return [im]
    # Set the interval to 1 ms for a smoother animation
    ani = animation.FuncAnimation(fig, update, frames=frames[::2], interval=1)

    output_path = Path(cfg.eval.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # plot obs trajectory
    
    for i, (observations, desired_goals, rewards) in enumerate(zip(all_observations, all_desired_goals, all_rewards)):
        plt.figure(figsize=(8, 8))
        # Convert to numpy arrays
        obs = np.array(observations)
        
        obs[:, 0] += _x_map_center
        obs[:, 1] += _y_map_center
        
        plt.plot(obs[:, 0], obs[:, 1], label="Trajectory")
        plt.scatter(obs[0, 0], obs[0, 1], color="green", label="Start")
        plt.scatter(obs[-1, 0], obs[-1, 1], color="red", label="End")
        plt.imshow(maze_map, cmap="gray", alpha=0.5, extent=(0, maze_width, 0, maze_height))   
        # Plot all unique desired goals
        unique_goals = np.unique(np.array(desired_goals), axis=0)
        unique_goals[:, 0] += _x_map_center
        unique_goals[:, 1] += _y_map_center
        plt.scatter(unique_goals[:, 0], unique_goals[:, 1], color="blue", marker="*", s=100, label="Desired Goals")
        
        # Plot a circle of radius 0.5 around each desired goal
        for goal in unique_goals:
            circle = plt.Circle(goal, 0.5, color="blue", fill=False, linestyle="--", alpha=0.7)
            plt.gca().add_artist(circle)
        
        plt.title("Agent Trajectory and Desired Goals, Mean Cum. Reward: {:.2f}".format(mean_cummulative_reward))
        
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.grid()
        plt.axis("equal")
        plt.savefig(output_path.with_stem(f"episode_{i}").with_suffix(f".png"))
        plt.close()
    
    print(f"✅ Saved trajectory plot as PNG: {output_path.with_suffix('.png')}")
    
    if cfg.eval.get("render_video", False):
        print("Saving rollout as video...")
        output_path = output_path.with_suffix(f".{cfg.eval.get('video_format', 'mp4')}")
        ani.save(str(output_path), writer="ffmpeg", fps=30)
        
        print(f"✅ Saved rollout as MP4: {output_path}")



if __name__ == "__main__":
    main()
