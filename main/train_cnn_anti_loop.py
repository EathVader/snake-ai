"""
Anti-looping training script for CNN model
Designed to prevent circular behavior and encourage food-seeking
"""
import os
import sys
import random

import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from snake_game_custom_wrapper_cnn_v3 import SnakeEnv

# Conservative settings for stability
if torch.backends.mps.is_available():
    NUM_ENV = 16  # Reduced for more stable training
    DEVICE = "mps"
elif torch.cuda.is_available():
    NUM_ENV = 16
    DEVICE = "cuda"
else:
    NUM_ENV = 8
    DEVICE = "cpu"

LOG_DIR = "logs/PPO_CNN_ANTI_LOOP"
os.makedirs(LOG_DIR, exist_ok=True)

def linear_schedule(initial_value, final_value=0.0):
    """Learning rate schedule"""
    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert (initial_value > 0.0)

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler

def make_env(rank, seed=0):
    """Create environment with unique seed"""
    def _init():
        env = SnakeEnv(seed=seed + rank)
        env = ActionMasker(env, SnakeEnv.get_action_mask)
        env = Monitor(env)
        return env
    return _init

def main():
    print("="*60)
    print("Anti-Looping Training Configuration")
    print("反转圈训练配置")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Parallel Environments: {NUM_ENV}")
    print(f"Key Changes:")
    print(f"- Heavy penalty for looping behavior")
    print(f"- Large rewards for eating food")
    print(f"- Aggressive distance-based rewards")
    print(f"- Position tracking to prevent circles")
    print("="*60)
    print()
    
    response = input("Start anti-looping training? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled.")
        return
    
    # Set random seed
    set_random_seed(42)
    
    # Generate random seeds
    seed_set = set()
    while len(seed_set) < NUM_ENV:
        seed_set.add(random.randint(0, 1e9))
    seed_list = list(seed_set)

    # Create vectorized environment
    print(f"\nCreating {NUM_ENV} parallel environments...")
    env = SubprocVecEnv([make_env(i, seed=s) for i, s in enumerate(seed_list)])

    # Conservative learning rate for stability
    lr_schedule = linear_schedule(5e-5, 1e-6)  # Even lower learning rate
    clip_range_schedule = linear_schedule(0.15, 0.05)
    
    # Create model with conservative hyperparameters
    print(f"Initializing PPO model on {DEVICE}...")
    model = MaskablePPO(
        "CnnPolicy",
        env,
        device=DEVICE,
        verbose=1,
        n_steps=2048,
        batch_size=256,  # Smaller batch for stability
        n_epochs=3,  # Fewer epochs
        gamma=0.995,  # Higher discount for long-term planning
        gae_lambda=0.95,
        learning_rate=lr_schedule,
        clip_range=clip_range_schedule,
        ent_coef=0.02,  # Higher entropy for more exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=LOG_DIR,
        policy_kwargs=dict(normalize_images=False)
    )

    # Setup directories
    save_dir = f"trained_models_cnn_anti_loop_{DEVICE}"
    os.makedirs(save_dir, exist_ok=True)

    # Checkpoint callback - save more frequently
    checkpoint_interval = 15625  # Every ~1M steps with 16 envs
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_interval,
        save_path=save_dir,
        name_prefix="ppo_snake_anti_loop"
    )

    # Evaluation callback
    eval_env = SubprocVecEnv([make_env(0, seed=999999)])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=save_dir,
        eval_freq=7812,  # Every ~500k steps
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    # Training
    print(f"\nStarting anti-looping training...")
    print(f"Logs: {LOG_DIR}")
    print(f"Models: {save_dir}")
    print(f"\nExpected behavior:")
    print(f"- Snake should actively seek food")
    print(f"- No more circular patterns")
    print(f"- Lower but more meaningful rewards")
    print(f"- Better actual game performance\n")
    
    # Write logs to file
    original_stdout = sys.stdout
    log_file_path = os.path.join(save_dir, "training_log.txt")
    
    with open(log_file_path, 'w') as log_file:
        sys.stdout = log_file
        
        model.learn(
            total_timesteps=int(50_000_000),  # 50M steps
            callback=[checkpoint_callback, eval_callback]
        )
        
        env.close()
        eval_env.close()

    sys.stdout = original_stdout
    
    # Save final model
    final_model_path = os.path.join(save_dir, "ppo_snake_anti_loop_final.zip")
    model.save(final_model_path)
    
    print(f"\n{'='*60}")
    print(f"Anti-looping training complete!")
    print(f"Final model: {final_model_path}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()