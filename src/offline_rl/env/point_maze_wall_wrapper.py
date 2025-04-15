import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


def get_wall_info(x, y, maze, grid_width, grid_height, maze_size_scaling):
    """Check for wall presence in four directions relative to agent position."""
    grid_x = int(x / maze_size_scaling)
    grid_y = int(y / maze_size_scaling)

    def is_wall(dx, dy):
        gx, gy = grid_x + dx, grid_y + dy
        if 0 <= gx < grid_width and 0 <= gy < grid_height:
            return maze[gy, gx] > 0
        return True  # Out of bounds is considered a wall

    left  = is_wall(-1,  0)
    right = is_wall( 1,  0)
    up    = is_wall( 0,  1)
    down  = is_wall( 0, -1)

    return np.array([left, right, up, down], dtype=np.float32)


class MazeWallObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        """
        Adds wall info [left, right, up, down] to the observation.
        
        Parameters:
        - resolution: size of each maze cell in world units
        """
        super().__init__(env)
        self.maze = np.array(env.unwrapped.maze.maze_map)
        self.x_map_center = self.env.unwrapped.maze.x_map_center
        self.y_map_center = self.env.unwrapped.maze.y_map_center
        self.maze_size_scaling = self.env.unwrapped.maze.maze_size_scaling

        self.grid_height, self.grid_width = self.maze.shape

        # Extend the observation space
        old_space = self.observation_space
        if isinstance(old_space, gym.spaces.Dict):
            # If the observation space is a dictionary, extend the relevant part
            new_space = gym.spaces.Dict({
                **old_space.spaces,
                'wall_info': Box(low=0, high=1, shape=(4,), dtype=np.float32)
            })
            self.observation_space = new_space
        else:
            raise TypeError("Only supports Box observation spaces.")

    def observation(self, obs):
        agent_x, agent_y = obs["observation"][0], obs["observation"][1]
        #agent_x = self.grid_width - (agent_x + self.x_map_center)
        agent_x = agent_x + self.x_map_center
        #agent_y = self.grid_height - (agent_y + self.y_map_center)
        agent_y = agent_y + self.y_map_center
        wall_info = self.get_wall_info(agent_x, agent_y)
        obs_dict = obs.copy()
        obs_dict['wall_info'] = wall_info
        return obs_dict
    
    def get_wall_info(self, x, y):
        """Get wall info for a given position."""
        return get_wall_info(x, y, self.maze, self.grid_width, self.grid_height, self.maze_size_scaling)