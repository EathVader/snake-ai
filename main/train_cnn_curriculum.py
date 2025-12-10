"""
Curriculum learning training script - progressively increase difficulty
Start with smaller board, gradually increase size as agent improves
"""
import os
import sys
import random

import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from snake_game_custom_wrapper_cnn_v2 import SnakeEnv

if torch.cuda.is_available():
    NUM_ENV = 64
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    NUM_ENV = 64
    DEVICE = "mps"
else:
    NUM_ENV = 32
    DEVICE = "cpu"

LOG_DIR = "logs/PPO_CNN_CURRICULUM"
os.makedirs(LOG_DIR, exist_ok=True)

class CurriculumCallback(BaseCallback):
    """
    Callback to implement curriculum learning
    Increases board size when agent reaches performance threshold
    """
    def __init__(self, check_freq, performance_threshold=0.7, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.performance_threshold = performance_threshold
        self.current_stage = 0
        self.stages = [6, 8, 10, 12]  # Board sizes
        self.best_mean_reward = -float('inf')
        
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            # Check recent performance
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = sum([ep_info['r'] for ep_info in self.model.ep_info_buffer]) / len(self.model.ep_info_buffer)
                
                if self.verbose > 0:
                    print(f"Stage {self.current_stage} (board={self.stages[self.current_stage]}): Mean reward = {mean_reward:.2f}")
                
                # Check if ready to advance to next stage
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                
                # Advance curriculum if performance is good and not at final stage
                if (mean_reward > self.performance_threshold and 
                    self.current_stage < len(self.stages) - 1):
                    self.current_stage += 1
                    self.best_mean_reward = -float('inf')
                    
                    if self.verbose > 0:
                        print(f"Advancing to stage {self.current_stage} with board size {self.stages[self.current_stage]}")
                    
                    # Note: In practice, you'd need to recreate environments here
                    # This is a simplified version
        
        return True

def linear_schedule(initial_value, final_value=0.0):
    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert (initial_value > 0.0)

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler

def make_env(rank, seed=0, board_size=6):
    def _init():
        env = SnakeEnv(seed=seed + rank, board_size=board_size)
        env = ActionMasker(env, SnakeEnv.get_action_mask)
        env = Monitor(env)
        return env
    return _init

def train_stage(stage, board_size, model=None, total_timesteps=10_000_000):
    """Train one curriculum stage"""
    print(f"\n{'='*60}")
    print(f"STAGE {stage}: Training with board size {board_size}x{board_size}")
    print(f"{'='*60}\n")
    
    set_random_seed(42 + stage)
    
    # Generate seeds
    seed_set = set()
    while len(seed_set) < NUM_ENV:
        seed_set.add(random.randint(0, int(1e9)))
    seed_list = list(seed_set)

    # Create environments with current board size
    env = SubprocVecEnv([make_env(i, seed=s, board_size=board_size) 
                         for i, s in enumerate(seed_list)])

    # Create or load model
    if model is None:
        lr_schedule = linear_schedule(3e-4, 1e-6)
        clip_range_schedule = linear_schedule(0.2, 0.05)
        
        model = MaskablePPO(
            "CnnPolicy",
            env,
            device=DEVICE,
            verbose=1,
            n_steps=2048,
            batch_size=1024 if DEVICE != "cpu" else 512,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            learning_rate=lr_schedule,
            clip_range=clip_range_schedule,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=LOG_DIR,
            policy_kwargs=dict(normalize_images=False)
        )
    else:
        # Update environment for existing model
        model.set_env(env)

    # Save directory for this stage
    save_dir = f"trained_models_cnn_curriculum/stage_{stage}_size_{board_size}"
    os.makedirs(save_dir, exist_ok=True)

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        reset_num_timesteps=False  # Continue counting timesteps
    )
    
    # Save stage model
    model.save(os.path.join(save_dir, f"ppo_snake_stage_{stage}.zip"))
    
    env.close()
    
    return model

def main():
    """Main curriculum learning pipeline"""
    # Curriculum stages: board sizes
    stages = [
        (0, 6, 5_000_000),    # Stage 0: 6x6 board, 5M steps
        (1, 8, 10_000_000),   # Stage 1: 8x8 board, 10M steps
        (2, 10, 20_000_000),  # Stage 2: 10x10 board, 20M steps
        (3, 12, 50_000_000),  # Stage 3: 12x12 board, 50M steps
    ]
    
    model = None
    
    for stage, board_size, timesteps in stages:
        model = train_stage(stage, board_size, model, timesteps)
    
    # Save final curriculum-trained model
    final_dir = "trained_models_cnn_curriculum"
    os.makedirs(final_dir, exist_ok=True)
    final_path = os.path.join(final_dir, "ppo_snake_curriculum_final.zip")
    model.save(final_path)
    
    print(f"\n{'='*60}")
    print(f"Curriculum training complete!")
    print(f"Final model saved to: {final_path}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
