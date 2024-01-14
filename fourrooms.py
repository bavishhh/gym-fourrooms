import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces


class FourRoomsEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    _colors = {"BACKGROUND": (74, 102, 117), 
               "WALL": (31, 46, 54),
               "GRID": (65, 87, 99),
               "AGENT": (219, 209, 66),
               "TARGET": (72, 163, 54)}

    def __init__(self, render_mode=None):
        self.room_size = 5
        self.size = 2 * self.room_size + 1
        self.window_size = 512  # The size of the PyGame window

        self.walls = self._generate_walls()

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}
    

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        agent_init_room = self.np_random.integers(0, 2, size=2, dtype=int)
        agent_init_pos = self.np_random.integers(0, self.room_size, size=2, dtype=int)
        self._agent_location = agent_init_room * (self.room_size + 1) + agent_init_pos

        # We will sample the target's location randomly until it does not coincide with the agent's location
        target_init_room = agent_init_room
        while np.array_equal(target_init_room, agent_init_room):
            target_init_room = self.np_random.integers(0, 2, size=2, dtype=int)
        target_init_pos = self.np_random.integers(0, self.room_size, size=2, dtype=int)
        self._target_location = target_init_room * (self.room_size + 1) + target_init_pos

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        new_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # make sure there is no wall present in the new location
        if self._is_safe(new_location):
            self._agent_location = new_location
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info
    
    def _generate_walls(self):
        doors = [[5, 1], [5, 8], [2, 5], [9, 5]]
        walls = []
        for i in range(0, 11):
            if [5, i] in doors:
                continue
            walls.append([5, i])
        
        for i in range(0, 11):
            if [i, 5] in doors:
                continue
            walls.append([i, 5])

        return walls
    
    def _is_safe(self, location):
        x = np.array(location)
        v = np.array(self.walls)

        return not np.any(np.all(x == v, axis=1))

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill(self._colors["BACKGROUND"])
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            self._colors["TARGET"],
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            self._colors["AGENT"],
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Draw the walls
        for wall in self.walls:
            pygame.draw.rect(
                canvas, 
                self._colors["WALL"],
                pygame.Rect(
                    (pix_square_size) * np.array(wall),
                    (pix_square_size, pix_square_size),
                ),
            )
        
        # Add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                self._colors["GRID"],
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                self._colors["GRID"],
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()