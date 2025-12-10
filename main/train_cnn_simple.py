"""
Simplified training script using train_config.py
使用train_config.py的简化训练脚本

Adjust settings in train_config.py to control:
在train_config.py中调整设置以控制：
- Number of parallel environments / 并行环境数量
- Memory usage / 内存使用
- Training speed / 训练速度
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

from snake_game_custom_wrapper_cnn_v2 import SnakeEnv
import train_config as config

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
        env = SnakeEnv(
            seed=seed + rank,
            board_size=config.BOARD_SIZE
        )
        env = ActionMasker(env, SnakeEnv.get_action_mask)
        env = Monitor(env)
        return env
    return _init

def main():
    # Print configuration
    print(config.get_config_summary())
    config.print_performance_tips()
    
    # Ask for confirmation
    response = input("\nStart training with these settings? (y/n) / 使用这些设置开始训练？(y/n): ")
    if response.lower() != 'y':
        print("Training cancelled. Edit train_config.py to adjust settings.")
        print("训练已取消。编辑train_config.py调整设置。")
        return
    
    # Set random seed
    set_random_seed(42)
    
    # Generate random seeds for each environment
    seed_set = set()
    while len(seed_set) < config.NUM_ENV:
        seed_set.add(random.randint(0, 1e9))
    seed_list = list(seed_set)

    # Create vectorized environment
    print(f"\nCreating {config.NUM_ENV} parallel environments...")
    print(f"创建{config.NUM_ENV}个并行环境...")
    env = SubprocVecEnv([make_env(i, seed=s) for i, s in enumerate(seed_list)])

    # Learning rate schedules
    lr_schedule = linear_schedule(
        config.LEARNING_RATE_START,
        config.LEARNING_RATE_END
    )
    clip_range_schedule = linear_schedule(
        config.CLIP_RANGE_START,
        config.CLIP_RANGE_END
    )
    
    # Create model
    print(f"\nInitializing PPO model on {config.DEVICE}...")
    print(f"在{config.DEVICE}上初始化PPO模型...")
    model = MaskablePPO(
        "CnnPolicy",
        env,
        device=config.DEVICE,
        verbose=1,
        n_steps=config.N_STEPS,
        batch_size=config.BATCH_SIZE,
        n_epochs=config.N_EPOCHS,
        gamma=config.GAMMA,
        gae_lambda=config.GAE_LAMBDA,
        learning_rate=lr_schedule,
        clip_range=clip_range_schedule,
        ent_coef=config.ENT_COEF,
        vf_coef=config.VF_COEF,
        max_grad_norm=config.MAX_GRAD_NORM,
        tensorboard_log=config.LOG_DIR,
        policy_kwargs=dict(normalize_images=False)
    )

    # Setup directories
    save_dir = f"{config.SAVE_DIR_PREFIX}_{config.DEVICE}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=config.CHECKPOINT_INTERVAL,
        save_path=save_dir,
        name_prefix="ppo_snake_v2"
    )

    # Evaluation callback
    eval_env = SubprocVecEnv([make_env(0, seed=999999)])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=save_dir,
        eval_freq=config.EVAL_FREQ,
        n_eval_episodes=config.N_EVAL_EPISODES,
        deterministic=True,
        render=False
    )

    # Training
    print(f"\nStarting training...")
    print(f"开始训练...")
    print(f"Logs: {config.LOG_DIR}")
    print(f"Models: {save_dir}")
    print(f"\nYou will see {config.NUM_ENV} Python processes - this is normal!")
    print(f"你会看到{config.NUM_ENV}个Python进程 - 这是正常的！\n")
    
    # Write logs to file
    original_stdout = sys.stdout
    log_file_path = os.path.join(save_dir, "training_log.txt")
    
    with open(log_file_path, 'w') as log_file:
        sys.stdout = log_file
        
        model.learn(
            total_timesteps=config.TOTAL_TIMESTEPS,
            callback=[checkpoint_callback, eval_callback]
        )
        
        env.close()
        eval_env.close()

    sys.stdout = original_stdout
    
    # Save final model
    final_model_path = os.path.join(save_dir, "ppo_snake_final_v2.zip")
    model.save(final_model_path)
    
    print(f"\n{'='*50}")
    print(f"Training complete! / 训练完成！")
    print(f"Final model: {final_model_path}")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()
