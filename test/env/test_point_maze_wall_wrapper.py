import gymnasium as gym
import gymnasium_robotics  # pylint: disable=unused-import
import numpy as np
import pytest

from offline_rl.env.point_maze_wall_wrapper import MazeWallObservationWrapper


@pytest.fixture
def wrapped_env():
    env = gym.make("PointMaze_Open-v3")
    return MazeWallObservationWrapper(env)


def test_observation_shape(wrapped_env):
    obs, _  = wrapped_env.reset()
    assert "wall_info" in obs


def test_wall_info_correct(wrapped_env):
    obs, _ = wrapped_env.reset(seed=42)
    wall_info = obs["wall_info"]
    expected = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # walls on left & down
    np.testing.assert_array_equal(wall_info, expected)
