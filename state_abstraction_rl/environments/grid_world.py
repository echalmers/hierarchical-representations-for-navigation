import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import time
import matplotlib.pyplot as plt
import itertools
import random
import os
import imageio.v2 as iio
import distinctipy


GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100000}

    def __init__(self, img, start_location=None, target_location=None, render_mode=None, randomize_environment_states=False, randomize_environment=False, block_size=10, max_reward=10):
        if isinstance(img, str):
            if os.path.isfile(img):
                self.img = iio.imread(img)
            else:
                self.img = self.get_named_layout(img)
        else:
            self.img = img

        self.size_x = self.img.shape[1]
        self.size_y = self.img.shape[0]
        self.block_size = block_size
        self.window_width = self.size_x * self.block_size
        self.window_height = self.size_y * self.block_size
        self.randomize_environment_states = randomize_environment_states
        self.randomize_environment = randomize_environment

        self.observation_space = spaces.Box(np.array([0, 0]), np.array([self.size_x, self.size_y]), dtype=int)
        # 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)
        self.max_reward = max_reward

        # locate start
        if start_location is None and not randomize_environment_states:
            locations = np.where(np.all(self.img == [0, 0, 255], axis=-1))
            self.start_location = np.array([locations[1][0], locations[0][0]])
        elif start_location is None and randomize_environment_states:
            start_loc = self._generate_loc()
            self.start_location = np.array([start_loc[0], start_loc[1]])
        else:
            self.start_location = np.array(start_location)

        # locate goal
        if target_location is None and not randomize_environment_states:
            locations = np.where(np.all(self.img == [0, 255, 0], axis=-1))
            self.target_location = np.array([locations[1][0], locations[0][0]])
        elif target_location is None and randomize_environment_states:
            target_loc = self._generate_loc()
            while target_loc == start_loc:
                target_loc = self._generate_loc()
            self.target_location = np.array([target_loc[0], target_loc[1]])
        else:
            self.target_location = np.array(target_location)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken. i.e., 0 corresponds to "right", 1 to "up" etc.
        Note the above predefined "constants" above for 0, 1, 2, 3
        """
        self.action_to_direction = {
            0:  np.array([1, 0]),  # right
            1:  np.array([0, 1]),  # up
            2:  np.array([-1, 0]),  # left
            3:  np.array([0, -1]),  # down
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

        self.q_ax = None
        self.heatmap = np.full((self.size_y, self.size_x), fill_value=0, dtype=int)

    def _get_obs(self):
        return self._agent_location

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def _generate_loc(self):
        locations = np.where(np.all(self.img == [255, 255, 255], axis=-1))
        x = np.random.randint(0, len(locations[0]))
        y = np.random.randint(0, len(locations[1]))

        return locations[0][x], locations[1][y]

    def _generate_new_env(self):
        imgs = ["4_rooms.bmp", "4_rooms_2.bmp", "4_rooms_3.bmp"]
        img = random.choice(imgs)
        self.img = self.get_named_layout(img)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.randomize_environment:
            self._generate_new_env()

        if self.randomize_environment_states:
            agent_loc = self._generate_loc()
            if (options["terminated"] and options["same_scenario_count"] >= 5) or options["start"]:
                # generate a new start/target location if the agent has reached the target 5 times in a row
                target_loc = self._generate_loc()
                while target_loc == agent_loc:
                    target_loc = self._generate_loc()
                self._agent_location = np.array([agent_loc[0], agent_loc[1]])
                self._target_location = np.array([target_loc[0], target_loc[1]])
            else:
                # only reset the starting location if the same_scenario_count is less than 5 or its not the start
                while agent_loc[0] == self._target_location[0] and agent_loc[1] == self._target_location[1]:
                    agent_loc = self._generate_loc()
                self._agent_location = np.array([agent_loc[0], agent_loc[1]])
        else:
            # Set agent and target location on reset
            self._agent_location = self.start_location
            self._target_location = self.target_location

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self.action_to_direction[action]
        next_cell = self._agent_location + direction

        # Only move agent if next cell is not a wall
        if not np.array_equal(self.img[next_cell[1]][next_cell[0]], [0, 0, 0]):
            self._agent_location = self._agent_location + direction

        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)

        observation = self._get_obs()
        info = self._get_info()

        if (self.img[next_cell[1]][next_cell[0]] == [0, 0, 0]).all():  # if agent hits a
            reward = -0.1
        elif terminated:
            reward = self.max_reward
        else:
            reward = -0.01

        if self.render_mode == "human":
            self._render_frame()

        self.heatmap[observation[1], observation[0]] += 1

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self, path=None):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((255, 255, 255))

        pix_square_size = (
            self.block_size
        )  # The size of a single grid square in pixels

        # Draw target
        pygame.draw.rect(
            canvas,
            GREEN,
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )

        # Draw agent
        pygame.draw.circle(
            canvas,
            BLACK,
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        for x in range(0, self.window_width, self.block_size):
            for y in range(0, self.window_height, self.block_size):
                rect = pygame.Rect(x, y, self.block_size, self.block_size)

                _x = int(x / self.block_size)
                _y = int(y / self.block_size)

                # Draw walls
                if (self.img[_y][_x] == [0, 0, 0]).all():
                    pygame.draw.rect(canvas, BLUE, rect)

                pygame.draw.rect(canvas, BLACK, rect, 1)

        # Draw best path if its passed in
        if path:
            for x, y in path:
                rect = pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size)
                pygame.draw.rect(canvas, (255, 0, 0), rect)
                pygame.draw.rect(canvas, BLACK, rect, 1)

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

    def draw_best_path(self, curr_path, prev_path):
        if curr_path != prev_path:
            self._render_frame(curr_path)
            time.sleep(1)

    def show_plots(self, best_values_dict: dict):
        if self.q_ax is None:
            _, self.q_ax = plt.subplots(1, 2)
            plt.ion()

        self.q_ax[0].cla()
        self.q_ax[0].set_title('perceived state values (max Q values)')
        self.q_ax[0].imshow(self.get_value_map(best_values_dict), cmap='copper')

        self.q_ax[1].cla()
        self.q_ax[1].set_title('agent heat map')
        self.q_ax[1].imshow(self.heatmap, cmap='hot')

        plt.pause(0.00001)

    def get_value_map(self, best_values_dict):
        value_map = np.zeros((self.size_y, self.size_x))
        for coords in np.ndindex(value_map.shape):
            s = (coords[1], coords[0])
            value_map[coords[0], coords[1]] = best_values_dict.get(s, 0)
        return value_map

    def color_grid(self, _x, _y, colors):
        rgb_colors = []
        for c in colors:
            rgb_colors.append(distinctipy.get_rgb256(c))
        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((255, 255, 255))

        i = 0
        for x, y in zip(_x, _y):
                rect = pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size)

                # Draw walls
                if not (self.img[y][x] == [0, 0, 0]).all():
                    pygame.draw.rect(canvas, rgb_colors[i], rect)
                    i+=1

                # pygame.draw.rect(canvas, BLACK, rect, 1)

        return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    @staticmethod
    def get_named_layout(name):
        layout_folder = os.path.join(os.path.dirname(__file__), 'grid_world_layouts')
        return iio.imread(os.path.join(layout_folder, name))

    @classmethod
    def get_random_gridworld(cls, size, n_walls, n_doors=1, return_inversed=False, **kwargs):
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

        # claim start and goal spaces
        occupied = [(1,1), (size-2, size-2)]

        # claim outer wall spaces
        occupied += list(itertools.chain.from_iterable([[(0, i), (i, 0), (size-1, i), (i, size-1)] for i in range(size)]))

        avoid = [tuple(np.array(state) + direction) for direction in directions for state in [(0, 0), (0, size-1), (size-1, 0), (size-1, size-1)]]
        doors = []

        n_walls_added = 0
        while n_walls_added < n_walls:
            # draw a random wall
            direction = random.choice(directions)
            wall = [random.choice(list(set(occupied[2:]) - set(avoid)))]
            for step in range(size):
                next = tuple(np.array(wall[-1]) + direction)

                if max(next) >= size or min(next) < 0:
                    break

                if next in doors:
                    wall.clear()
                    break

                wall.append(next)
                if next in occupied:
                    break

            if len(wall) > 2 + n_doors:
                for _ in range(n_doors):
                    this_door = random.choice(wall[1:-2])
                    wall.remove(this_door)
                    doors.append(this_door)
                occupied += wall[1:-1]
                n_walls_added += 1

                avoid.extend([tuple(np.array(wall[0]) + direction) for direction in directions])
                avoid.extend([tuple(np.array(wall[-1]) + direction) for direction in directions])

        # construct image
        im = np.ones((size, size, 3)) * 255
        for state in occupied[2:]:
            im[state[0], state[1], :] = [0, 0, 0]
        im[occupied[0][0], occupied[0][1], :] = [0, 0, 255]
        im[occupied[1][0], occupied[1][1], :] = [0, 255, 0]

        if return_inversed:
            # generate an inverse of the random layout
            im_inversed = im.copy()
            im_inversed[occupied[0][0], occupied[0][1], :] = [255, 255, 255]
            im_inversed[occupied[1][0], occupied[1][1], :] = [255, 255, 255]
            im_inversed[size-2, 1, :] = [0, 0, 255]
            im_inversed[1, size-2, :] = [0, 255, 0]

            return GridWorldEnv(img=im.astype(int), **kwargs), GridWorldEnv(img=im_inversed.astype(int), **kwargs)

        return GridWorldEnv(img=im.astype(int), **kwargs)


if __name__ == '__main__':
    import imageio.v2 as iio
    env, env_inverse = GridWorldEnv.get_random_gridworld(size=15, n_walls=5, n_doors=1, return_inversed=True, render_mode='rgb_array')
    # env = GridWorldEnv(iio.imread('4_rooms.bmp', render_mode='rgb_array', )
    env.reset()
    env_inverse.reset()

    plt.subplot(1,2,1)
    plt.imshow(env.render())

    plt.subplot(1,2,2)
    plt.imshow(env_inverse.render())

    plt.show()
