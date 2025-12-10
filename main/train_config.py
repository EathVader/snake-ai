"""
Training configuration / 训练配置
Adjust these settings based on your hardware
根据你的硬件调整这些设置
"""

import torch

# ============================================
# Environment Settings / 环境设置
# ============================================

# Number of parallel environments / 并行环境数量
# More environments = faster training but more CPU/RAM usage
# 更多环境 = 更快训练但占用更多CPU/内存
#
# Recommended values / 推荐值:
# - High-end Mac (M1 Max/Ultra, M2 Max/Ultra): 64
# - Mid-range Mac (M1/M2 Pro): 32
# - Entry-level Mac (M1/M2): 16
# - Low memory (<16GB RAM): 8

if torch.backends.mps.is_available():
    NUM_ENV = 32  # Default for MPS / MPS默认值
    DEVICE = "mps"
elif torch.cuda.is_available():
    NUM_ENV = 64  # Default for CUDA / CUDA默认值
    DEVICE = "cuda"
else:
    NUM_ENV = 16  # Default for CPU / CPU默认值
    DEVICE = "cpu"

# ============================================
# Training Hyperparameters / 训练超参数
# ============================================

# Total training steps / 总训练步数
# Adjust based on your needs:
# - 10M steps: ~1-2 hours (quick test)
# - 50M steps: ~4-6 hours (medium training)
# - 100M steps: ~8-12 hours (full training) ⭐
# - 200M steps: ~16-24 hours (deep training)
TOTAL_TIMESTEPS = 100_000_000  # 100M steps (default)

# Steps per environment before policy update / 每个环境在策略更新前的步数
N_STEPS = 2048

# Batch size for training / 训练批次大小
# Larger = more stable but slower / 更大 = 更稳定但更慢
BATCH_SIZE = 1024 if DEVICE != "cpu" else 512

# Number of epochs per update / 每次更新的轮数
# Reduced to prevent overfitting / 减少以防止过拟合
N_EPOCHS = 4  # Reduced from 10

# Discount factor / 折扣因子
# Higher = more long-term planning / 更高 = 更注重长期规划
GAMMA = 0.99

# GAE lambda / GAE参数
GAE_LAMBDA = 0.95

# Learning rate schedule / 学习率调度
# Reduced to prevent instability / 降低以防止不稳定
LEARNING_RATE_START = 1e-4  # Reduced from 3e-4
LEARNING_RATE_END = 1e-6

# Clip range schedule / 裁剪范围调度
CLIP_RANGE_START = 0.2
CLIP_RANGE_END = 0.05

# Entropy coefficient / 熵系数
# Higher = more exploration / 更高 = 更多探索
# Current models stabilize at ~-0.15 entropy
# Increase to 0.02-0.05 for more exploration
ENT_COEF = 0.01  # Try 0.02 or 0.05 for more exploration

# Value function coefficient / 价值函数系数
VF_COEF = 0.5

# Gradient clipping / 梯度裁剪
MAX_GRAD_NORM = 0.5

# ============================================
# Checkpoint Settings / 检查点设置
# ============================================

# Save checkpoint every N steps (per environment) / 每N步保存检查点
# checkpoint_interval * NUM_ENV = total steps per checkpoint
# Example: 15625 * 64 = 1M steps
CHECKPOINT_INTERVAL = 15625

# Evaluation frequency / 评估频率
# eval_freq * NUM_ENV = total steps per evaluation
EVAL_FREQ = 7812  # ~500k steps with 64 envs

# Number of evaluation episodes / 评估回合数
N_EVAL_EPISODES = 10

# ============================================
# Environment Settings / 环境设置
# ============================================

# Board size / 棋盘大小
BOARD_SIZE = 12

# Step limit multiplier / 步数限制倍数
# step_limit = BOARD_SIZE^2 * STEP_LIMIT_MULTIPLIER
STEP_LIMIT_MULTIPLIER = 4

# ============================================
# Logging / 日志
# ============================================

LOG_DIR = "logs/PPO_CNN_V2"
SAVE_DIR_PREFIX = "trained_models_cnn_v2"

# ============================================
# Helper Functions / 辅助函数
# ============================================

def get_config_summary():
    """Get configuration summary / 获取配置摘要"""
    return f"""
Training Configuration / 训练配置:
{'='*50}
Device / 设备: {DEVICE}
Parallel Environments / 并行环境: {NUM_ENV}
Total Timesteps / 总步数: {TOTAL_TIMESTEPS:,}
Batch Size / 批次大小: {BATCH_SIZE}
Learning Rate / 学习率: {LEARNING_RATE_START} → {LEARNING_RATE_END}
Board Size / 棋盘大小: {BOARD_SIZE}x{BOARD_SIZE}
{'='*50}

Expected Memory Usage / 预期内存占用:
- RAM: ~{NUM_ENV * 100}MB - {NUM_ENV * 200}MB
- VRAM: ~2-4GB

Checkpoint Frequency / 检查点频率:
- Every {CHECKPOINT_INTERVAL * NUM_ENV:,} steps
- Evaluation every {EVAL_FREQ * NUM_ENV:,} steps
{'='*50}
"""

def print_performance_tips():
    """Print performance optimization tips / 打印性能优化建议"""
    print("""
Performance Tips / 性能建议:
{'='*50}
If training is too slow / 如果训练太慢:
1. Reduce NUM_ENV (e.g., 32 → 16)
2. Reduce BATCH_SIZE (e.g., 1024 → 512)
3. Reduce N_EPOCHS (e.g., 10 → 4)

If running out of memory / 如果内存不足:
1. Reduce NUM_ENV significantly (e.g., 64 → 16)
2. Close other applications
3. Reduce BOARD_SIZE (e.g., 12 → 10)

If you see many Python processes / 如果看到很多Python进程:
- This is normal! Each environment runs in its own process
- 这是正常的！每个环境在独立进程中运行
- Number of processes = NUM_ENV + 1 (main process)
- 进程数 = NUM_ENV + 1（主进程）
{'='*50}
""")
