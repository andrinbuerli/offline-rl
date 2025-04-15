from pathlib import Path

import gymnasium as gym
import minari
import numpy as np
from minari.data_collector import EpisodeBuffer
from minari.dataset.minari_storage import MinariStorage
from tqdm import tqdm

from offline_rl.env.point_maze_wall_wrapper import MazeWallObservationWrapper

# Load dataset
dataset_id = "D4RL/pointmaze/medium-v2"
dataset = minari.load_dataset(dataset_id)

# Create dummy env to extract maze/map info
dummy_env = MazeWallObservationWrapper(dataset.recover_environment())

def compute_wall_info(obs_dict, wrapper):
    x = obs_dict["observation"][0] + wrapper.x_map_center
    y = obs_dict["observation"][1] + wrapper.y_map_center
    return wrapper.get_wall_info(x, y)

new_episodes = []

print(f"Processing {len(dataset)} episodes...")

# Loop over existing episodes and add wall_info to observations
for ep in tqdm(dataset):
    obs = ep.observations  # dict with 'observation', 'achieved_goal', etc.
    new_obs = {k: np.copy(v) for k, v in obs.items()}  # shallow copy

    wall_info_list = []

    for i in range(len(obs["observation"])):
        obs_dict = {k: v[i] for k, v in obs.items()}  # simulate timestep obs dict
        wall_info = compute_wall_info(obs_dict, dummy_env)
        wall_info_list.append(wall_info)

    wall_info_array = np.stack(wall_info_list, axis=0)
    new_obs["wall_info"] = wall_info_array

    new_episode = EpisodeBuffer(
        id=ep.id,
        seed=123,
        observations=new_obs,
        actions=ep.actions,
        rewards=ep.rewards,
        terminations=ep.terminations,
        truncations=ep.truncations,
        infos=ep.infos,
    )

    new_episodes.append(new_episode)

# Save new dataset
new_dataset_id = f"{dataset_id.replace('/', '-')}-with-wall"
print(f"Saving enriched dataset as: {new_dataset_id}")
dataset_path = Path.home() /".minari" / "datasets" / new_dataset_id
dataset_path.mkdir(parents=True, exist_ok=True)
storage = MinariStorage.new(
    data_path=dataset_path / "data",
    observation_space=dummy_env.observation_space,
    action_space=dummy_env.action_space,
    env_spec=dummy_env.spec,
    data_format="hdf5",
)
storage.update_episodes(new_episodes)
storage.update_metadata({
    "dataset_id": new_dataset_id,
    "minari_version": minari.__version__,
})


new_dataset = minari.load_dataset(new_dataset_id)

print(f"✅ Enriched dataset saved as: {new_dataset_id}")
