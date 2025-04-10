import argparse
import os

os.environ["MUJOCO_GL"] = "egl"
from pathlib import Path

import gymnasium_robotics  # pylint: disable=unused-import
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
    env = GymEnv(cfg.env_id, continuing_task=True, reset_target=True, max_episode_steps=500, device=cfg.device, render_mode="rgb_array")
    env = TransformedEnv(env, device=cfg.device, transform=DoubleToFloat())

    model = IQLNetwork(env, cfg.model, cfg.device)
    
    # Load weights
    print(f"Total parameter sum in actor: {sum(p.sum() for p in model.parameters())}")
    ckpt = torch.load(cfg.ckpt_path, map_location=cfg.device)

    model.load_state_dict(ckpt)
    print(f"Total parameter sum after loading: {sum(p.sum() for p in model.parameters())}")
    model.eval()
    
    env.reset(seed=42)
    episode_rewards = []
    for _ in range(10):
        eval_td = env.rollout(max_steps=500, policy=model.actor, auto_cast_to_device=True)
        episode_rewards.append(eval_td["next", "reward"].sum().item())
    print(f"Average episode reward: {np.mean(episode_rewards)}")

    return env, model.actor

@hydra.main(config_path=str((Path(__file__).parent.parent / "configs" / "eval").resolve()), version_base=None)
def main(cfg: DictConfig):
    device = torch.device(args.device)
    env, actor = load_policy(
        env_id=args.env_id,
        actor_ckpt_path=args.checkpoint,
        device=device,
        action_dim=args.action_dim,
        encoder_layers=args.encoder_layers
    )

    # Run rollout
    frames = []
    tensordict = env.reset(seed=123)
    done = tensordict["done"].item()

    step = 0
    observations = []
    desired_goals = []
    rewards = []
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
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
            print(f"Step {step}: Action: {step_input['action']}, Obs: {tensordict['observation']}, Done: {done}, Terminated: {terminated}, Truncated: {truncated}", end="\r")

    cummulative_reward = np.sum(rewards)
    print(f"\nCumulative reward: {cummulative_reward}")

    # Animate and save
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(frames[0])
    ax.axis("off")
    plt.title("Policy Rollout")

    def update(frame):
        im.set_array(frame)
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=100)

    output_path = Path(args.output_gif).with_suffix(".mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(str(output_path), writer="ffmpeg", fps=30)
    
    # plot obs trajectory
    obs = np.array(observations)
    plt.figure(figsize=(8, 8))
    plt.plot(obs[:, 0], obs[:, 1], label="Trajectory")
    plt.scatter(obs[0, 0], obs[0, 1], color="green", label="Start")
    plt.scatter(obs[-1, 0], obs[-1, 1], color="red", label="End")
    
    # Plot all unique desired goals
    unique_goals = np.unique(np.array(desired_goals), axis=0)
    plt.scatter(unique_goals[:, 0], unique_goals[:, 1], color="blue", marker="*", s=100, label="Desired Goals")
    
    # Plot a circle of radius 0.5 around each desired goal
    for goal in unique_goals:
        circle = plt.Circle(goal, 0.5, color="blue", fill=False, linestyle="--", alpha=0.7)
        plt.gca().add_artist(circle)
    
    plt.title("Agent Trajectory and Desired Goals, Cum. Reward: {:.2f}".format(cummulative_reward))
    
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid()
    plt.axis("equal")
    plt.savefig(output_path.with_suffix(".png"))
    plt.close()
    
    print(f"✅ Saved rollout as MP4: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
