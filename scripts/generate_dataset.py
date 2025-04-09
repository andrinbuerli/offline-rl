# generate_pointmaze_dataset.py

from pathlib import Path

import gymnasium as gym
import gymnasium_robotics  # pylint: disable=unused-import
import hydra
import numpy as np
from minari import DataCollector
from omegaconf import DictConfig


@hydra.main(
    config_path=str((Path(__file__).parent.parent / "configs" / "dataset").resolve()),
    version_base=None,
)
def main(cfg: DictConfig):
    # Create environment
    env = gym.make(cfg.env_id)

    total_steps = 0

    print(f"Observation space: {env.observation_space}")

    env = DataCollector(
        env,
        observation_space=env.observation_space,
        action_space=env.action_space,
    )

    rng = np.random.default_rng(cfg.seed)
    while total_steps < cfg.total_steps:
        obs = env.reset(seed=int(rng.uniform(0, int(1e10))))
        done = False
        episode_steps = 0

        while not done and episode_steps < cfg.max_steps_per_episode:
            if cfg.algorithm_name == "random":
                action = env.action_space.sample()
            else:
                raise ValueError(f"Unknown policy: {cfg.algorithm_name}")

            obs, reward, terminated, truncated, info = env.step(action)

            episode_steps += 1
            total_steps += 1

            if total_steps >= cfg.total_steps:
                break

    dataset = env.create_dataset(
        eval_env=env,
        dataset_id=cfg.dataset_id,
        algorithm_name=cfg.algorithm_name,
    )

    obs_keys = dataset[0].observations.keys()

    print(f"Dataset keys: {obs_keys}")
    print(f"Dataset size: {len(dataset)}")
    print("Done !")


if __name__ == "__main__":
    main()
