"""
Snake environment wrapper for Gymnasium - CNN version
Compatible with gymnasium>=0.29.0 and stable-baselines3>=2.2.0
"""
import math
import gymnasium as gym
import numpy as np

from snake_game import SnakeGame

class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(self, seed=0, board_size=12, silent_mode=True, limit_step=True, render_mode=None):
        super().__init__()
        self.game = SnakeGame(seed=seed, board_size=board_size, silent_mode=silent_mode)
        self.game.reset()

        self.silent_mode = silent_mode
        self.render_mode = render_mode

        self.action_space = gym.spaces.Discrete(4)  # 0: UP, 1: LEFT, 2: RIGHT, 3: DOWN
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(84, 84, 3),
            dtype=np.uint8
        )

        self.board_size = board_size
        self.grid_size = board_size ** 2
        self.init_snake_size = len(self.game.snake)
        self.max_growth = self.grid_size - self.init_snake_size

        if limit_step:
            self.step_limit = self.grid_size * 4
        else:
            self.step_limit = 1e9
        self.reward_step_counter = 0
        
        self._seed = seed

    def reset(self, seed=None, options=None):
        # Handle new gymnasium reset API
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
            self.game = SnakeGame(seed=seed, board_size=self.board_size, silent_mode=self.silent_mode)
        
        self.game.reset()
        self.reward_step_counter = 0

        obs = self._generate_observation()
        info = {}
        return obs, info
    
    def step(self, action):
        game_over, info = self.game.step(action)
        obs = self._generate_observation()

        reward = 0.0
        self.reward_step_counter += 1

        terminated = False
        truncated = False

        if info["snake_size"] == self.grid_size:
            reward = self.max_growth * 0.1
            terminated = True
            if not self.silent_mode:
                self.game.sound_victory.play()
            return obs, reward, terminated, truncated, info
        
        if self.reward_step_counter > self.step_limit:
            self.reward_step_counter = 0
            truncated = True
        
        if game_over:
            reward = - math.pow(self.max_growth, (self.grid_size - info["snake_size"]) / self.max_growth)
            reward = reward * 0.1
            terminated = True
            return obs, reward, terminated, truncated, info
          
        elif info["food_obtained"]:
            reward = info["snake_size"] / self.grid_size
            self.reward_step_counter = 0
        
        else:
            if np.linalg.norm(info["snake_head_pos"] - info["food_pos"]) < np.linalg.norm(info["prev_snake_head_pos"] - info["food_pos"]):
                reward = 1 / info["snake_size"]
            else:
                reward = - 1 / info["snake_size"]
            reward = reward * 0.1

        return obs, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == "human":
            self.game.render()
        return self.game.render()

    def get_action_mask(self):
        return np.array([[self._check_action_validity(a) for a in range(self.action_space.n)]])
    
    def _check_action_validity(self, action):
        current_direction = self.game.direction
        snake_list = self.game.snake
        row, col = snake_list[0]
        
        if action == 0:  # UP
            if current_direction == "DOWN":
                return False
            else:
                row -= 1
        elif action == 1:  # LEFT
            if current_direction == "RIGHT":
                return False
            else:
                col -= 1
        elif action == 2:  # RIGHT 
            if current_direction == "LEFT":
                return False
            else:
                col += 1     
        elif action == 3:  # DOWN 
            if current_direction == "UP":
                return False
            else:
                row += 1

        if (row, col) == self.game.food:
            game_over = (
                (row, col) in snake_list
                or row < 0
                or row >= self.board_size
                or col < 0
                or col >= self.board_size
            )
        else:
            game_over = (
                (row, col) in snake_list[:-1]
                or row < 0
                or row >= self.board_size
                or col < 0
                or col >= self.board_size
            )

        return not game_over

    def _generate_observation(self):
        obs = np.zeros((self.game.board_size, self.game.board_size), dtype=np.uint8)
        obs[tuple(np.transpose(self.game.snake))] = np.linspace(200, 50, len(self.game.snake), dtype=np.uint8)
        obs = np.stack((obs, obs, obs), axis=-1)
        obs[tuple(self.game.snake[0])] = [0, 255, 0]
        obs[tuple(self.game.snake[-1])] = [255, 0, 0]
        obs[self.game.food] = [0, 0, 255]
        obs = np.repeat(np.repeat(obs, 7, axis=0), 7, axis=1)
        return obs
