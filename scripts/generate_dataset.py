# generate_pointmaze_dataset.py

from pathlib import Path

import gymnasium as gym
import gymnasium_robotics  # pylint: disable=unused-import
import hydra
import numpy as np
from minari import DataCollector
from omegaconf import DictConfig
from tqdm import tqdm

import offline_rl  # pylint: disable=unused-import


@hydra.main(
    config_path=str((Path(__file__).parent.parent / "configs").resolve()),
    config_name="generate_dataset",
    version_base=None,
)
def main(cfg: DictConfig):
    # Create environment
    env = gym.make(cfg.env.id, continuing_task=cfg.env.continuing_task, reset_target=cfg.env.reset_target, max_episode_steps=cfg.dataset.max_steps_per_episode)

    total_steps = 0

    print(f"Observation space: {env.observation_space}")

    env = DataCollector(
        env,
        observation_space=env.observation_space,
        action_space=env.action_space,
    )

    rng = np.random.default_rng(cfg.seed)
    pbar = tqdm(total=cfg.dataset.total_steps, desc="Generating dataset", unit="step")
    while total_steps < cfg.dataset.total_steps:
        obs = env.reset(seed=int(rng.uniform(0, int(1e10))))
        done = False
        episode_steps = 0

        while not done and episode_steps < cfg.dataset.max_steps_per_episode:
            if cfg.dataset.algorithm_name == "random":
                action = env.action_space.sample()
            else:
                raise ValueError(f"Unknown policy: {cfg.dataset.algorithm_name}")

            obs, reward, terminated, truncated, info = env.step(action)

            episode_steps += 1
            total_steps += 1

            if total_steps >= cfg.dataset.total_steps:
                break
            pbar.update(1)

    dataset = env.create_dataset(
        eval_env=env,
        dataset_id=cfg.dataset.id,
        algorithm_name=cfg.dataset.algorithm_name,
    )

    obs_keys = dataset[0].observations.keys()

    print(f"Dataset keys: {obs_keys}")
    print(f"Dataset size: {len(dataset)}")
    print("Done !")


if __name__ == "__main__":
    main()
