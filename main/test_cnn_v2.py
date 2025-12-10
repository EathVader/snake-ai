"""
Test script for enhanced CNN models
Visualize trained agent performance
"""
import os
import time
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from snake_game_custom_wrapper_cnn_v2 import SnakeEnv

def test_model(model_path, num_episodes=10, render=True, delay=0.1):
    """
    Test a trained model
    
    Args:
        model_path: Path to saved model (.zip file)
        num_episodes: Number of episodes to run
        render: Whether to render the game
        delay: Delay between steps (seconds)
    """
    print(f"Loading model from: {model_path}")
    
    # Load model
    model = MaskablePPO.load(model_path)
    
    # Create environment
    env = SnakeEnv(silent_mode=False, render_mode="human" if render else None)
    env = ActionMasker(env, SnakeEnv.get_action_mask)
    
    # Statistics
    episode_rewards = []
    episode_lengths = []
    max_snake_sizes = []
    
    print(f"\nTesting for {num_episodes} episodes...\n")
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        max_snake_size = 3
        
        while not done:
            # Get action mask
            action_masks = env.action_masks()
            
            # Predict action
            action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            max_snake_size = max(max_snake_size, info.get("snake_size", 3))
            
            if render:
                env.render()
                time.sleep(delay)
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        max_snake_sizes.append(max_snake_size)
        
        print(f"Episode {episode + 1}/{num_episodes}:")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Length: {episode_length}")
        print(f"  Max Snake Size: {max_snake_size}")
        print()
    
    # Print summary statistics
    print("="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Average Max Snake Size: {np.mean(max_snake_sizes):.1f} ± {np.std(max_snake_sizes):.1f}")
    print(f"Best Snake Size: {np.max(max_snake_sizes)}")
    print(f"Win Rate: {sum(1 for s in max_snake_sizes if s == 144) / num_episodes * 100:.1f}%")
    print("="*60)
    
    env.close()

def compare_models(model_paths, num_episodes=10):
    """
    Compare multiple models
    
    Args:
        model_paths: List of (name, path) tuples
        num_episodes: Number of episodes per model
    """
    results = {}
    
    for name, path in model_paths:
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print(f"{'='*60}\n")
        
        model = MaskablePPO.load(path)
        env = SnakeEnv(silent_mode=True)
        env = ActionMasker(env, SnakeEnv.get_action_mask)
        
        rewards = []
        snake_sizes = []
        
        for _ in range(num_episodes):
            obs, info = env.reset()
            done = False
            episode_reward = 0
            max_size = 3
            
            while not done:
                action_masks = env.action_masks()
                action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                max_size = max(max_size, info.get("snake_size", 3))
            
            rewards.append(episode_reward)
            snake_sizes.append(max_size)
        
        results[name] = {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_size': np.mean(snake_sizes),
            'max_size': np.max(snake_sizes),
            'win_rate': sum(1 for s in snake_sizes if s == 144) / num_episodes * 100
        }
        
        env.close()
    
    # Print comparison table
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}\n")
    print(f"{'Model':<20} {'Avg Reward':<15} {'Avg Size':<12} {'Max Size':<12} {'Win Rate':<10}")
    print("-"*60)
    
    for name, stats in results.items():
        print(f"{name:<20} {stats['mean_reward']:>6.2f} ± {stats['std_reward']:<5.2f} "
              f"{stats['mean_size']:>6.1f}      {stats['max_size']:>6}      "
              f"{stats['win_rate']:>5.1f}%")
    
    print("="*60)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Test single model:")
        print("    python test_cnn_v2.py <model_path> [num_episodes] [render] [delay]")
        print("  Compare models:")
        print("    python test_cnn_v2.py --compare <model1_path> <model2_path> ... [num_episodes]")
        print("\nExamples:")
        print("  python test_cnn_v2.py trained_models_cnn_v2_cuda/ppo_snake_final_v2.zip")
        print("  python test_cnn_v2.py trained_models_cnn_v2_cuda/ppo_snake_final_v2.zip 20 True 0.05")
        print("  python test_cnn_v2.py --compare model1.zip model2.zip 50")
        sys.exit(1)
    
    if sys.argv[1] == "--compare":
        # Compare mode
        model_paths = [(os.path.basename(p), p) for p in sys.argv[2:] if p.endswith('.zip')]
        num_episodes = int(sys.argv[-1]) if not sys.argv[-1].endswith('.zip') else 10
        compare_models(model_paths, num_episodes)
    else:
        # Single model test
        model_path = sys.argv[1]
        num_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        render = sys.argv[3].lower() == 'true' if len(sys.argv) > 3 else True
        delay = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1
        
        test_model(model_path, num_episodes, render, delay)
