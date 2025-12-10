"""
Snake environment wrapper for Gymnasium - CNN version V3
Anti-looping reward function to prevent circular behavior
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

        self.action_space = gym.spaces.Discrete(4)
        
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
            self.step_limit = self.grid_size * 2  # Reduced step limit
        else:
            self.step_limit = 1e9
        self.reward_step_counter = 0
        self.prev_distance = None
        self.recent_positions = []
        self.steps_without_food = 0
        
        self._seed = seed

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
            self.game = SnakeGame(seed=seed, board_size=self.board_size, silent_mode=self.silent_mode)
        
        self.game.reset()
        self.reward_step_counter = 0
        self.prev_distance = self._get_distance_to_food()
        self.recent_positions = []
        self.steps_without_food = 0

        obs = self._generate_observation()
        info = {}
        return obs, info
    
    def _get_distance_to_food(self):
        """Calculate Manhattan distance to food"""
        head = self.game.snake[0]
        food = self.game.food
        return abs(head[0] - food[0]) + abs(head[1] - food[1])
    
    def step(self, action):
        game_over, info = self.game.step(action)
        obs = self._generate_observation()

        reward = 0.0
        self.reward_step_counter += 1
        self.steps_without_food += 1

        terminated = False
        truncated = False

        # Victory condition - snake fills entire board
        if info["snake_size"] == self.grid_size:
            reward = 100.0
            terminated = True
            if not self.silent_mode:
                self.game.sound_victory.play()
            return obs, reward, terminated, truncated, info
        
        # Step limit exceeded
        if self.reward_step_counter > self.step_limit:
            self.reward_step_counter = 0
            truncated = True
        
        # Game over - collision or out of bounds
        if game_over:
            # Heavy penalty for dying
            reward = -50.0
            terminated = True
            return obs, reward, terminated, truncated, info
          
        # Food obtained - BIG REWARD!
        elif info["food_obtained"]:
            # Large reward that scales with snake size
            base_reward = 50.0
            size_bonus = (info["snake_size"] - self.init_snake_size) * 5.0
            efficiency_bonus = max(0, 50 - self.steps_without_food) * 0.5  # Bonus for quick food collection
            
            reward = base_reward + size_bonus + efficiency_bonus
            self.reward_step_counter = 0
            self.steps_without_food = 0
            self.prev_distance = self._get_distance_to_food()
            self.recent_positions = []  # Clear position history after eating
        
        # Moving towards or away from food
        else:
            current_distance = self._get_distance_to_food()
            head_pos = info["snake_head_pos"]
            
            # Base step penalty - encourages efficiency
            reward = -0.5
            
            # Distance-based reward (much more aggressive)
            if current_distance < self.prev_distance:
                reward += 2.0  # Good reward for getting closer
            else:
                reward -= 5.0  # Heavy penalty for moving away
            
            # Anti-looping mechanisms
            # 1. Penalty for revisiting recent positions
            if len(self.recent_positions) > 0:
                min_distance_to_recent = min(
                    np.linalg.norm(head_pos - pos) for pos in self.recent_positions
                )
                if min_distance_to_recent < 3.0:
                    reward -= 10.0  # Heavy penalty for looping
            
            # 2. Penalty for staying too long without food
            if self.steps_without_food > self.grid_size:
                reward -= (self.steps_without_food - self.grid_size) * 0.1
            
            # 3. Penalty for being too far from food for too long
            if current_distance > self.board_size and self.steps_without_food > 20:
                reward -= 2.0
            
            # Update tracking
            self.prev_distance = current_distance
            self.recent_positions.append(head_pos.copy())
            if len(self.recent_positions) > 15:  # Keep last 15 positions
                self.recent_positions.pop(0)

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