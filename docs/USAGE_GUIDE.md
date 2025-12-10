# Snake AI Training Scripts / 贪吃蛇AI训练脚本

## 项目结构 / Project Structure

```
main/
├── 🎮 Core / 核心
│   ├── snake_game.py                           # Game engine / 游戏引擎
│   ├── snake_game_custom_wrapper_cnn_v2.py     # CNN environment wrapper / CNN环境包装器
│   └── snake_game_custom_wrapper_mlp.py        # MLP environment wrapper / MLP环境包装器
│
├── ⚙️ Configuration / 配置
│   └── train_config.py                         # Centralized training config / 集中训练配置
│
├── 🚂 Training / 训练
│   ├── train_cnn_simple.py                     # ⭐ Recommended: Config-based training / 推荐：基于配置的训练
│   ├── train_cnn.py                            # Baseline CNN training / 基准CNN训练
│   ├── train_mlp.py                            # MLP training / MLP训练
│   └── train_cnn_curriculum.py                 # Curriculum learning / 课程学习
│
├── 🧪 Testing / 测试
│   ├── test_cnn_v2.py                          # CNN model testing / CNN模型测试
│   └── test_mlp.py                             # MLP model testing / MLP模型测试
│
├── 🔄 Baseline / 基准
│   └── hamiltonian_agent.py                    # Hamiltonian cycle baseline / 哈密尔顿回路基准
│
└── 📁 Output / 输出
    ├── trained_models_*/                       # Saved models / 保存的模型
    └── logs/                                   # TensorBoard logs / TensorBoard日志
```

---

## 快速开始 / Quick Start

### 1. 推荐训练方式 / Recommended Training ⭐

```bash
# Activate environment / 激活环境
conda activate SnakeAI-new

# Navigate to main directory / 进入main目录
cd main

# Start training with config file / 使用配置文件开始训练
python train_cnn_simple.py
```

**特点 / Features:**
- ✅ 使用 `train_config.py` 集中管理参数
- ✅ 训练前显示配置摘要
- ✅ 需要确认后才开始训练
- ✅ 便于调整参数实验

---

### 2. 调整训练参数 / Adjust Training Parameters

编辑 `train_config.py`:

```python
# 修改这些参数 / Modify these parameters
NUM_ENV = 32                    # 并行环境数 / Parallel environments
TOTAL_TIMESTEPS = 100_000_000   # 总训练步数 / Total training steps
LEARNING_RATE_START = 1e-4      # 初始学习率 / Initial learning rate
N_EPOCHS = 4                    # 训练轮数 / Training epochs
BATCH_SIZE = 1024               # 批次大小 / Batch size
```

---

### 3. 监控训练 / Monitor Training

```bash
# 在新终端中启动TensorBoard / Start TensorBoard in new terminal
tensorboard --logdir main/logs
```

然后在浏览器打开 `http://localhost:6006`

**关键指标 / Key Metrics:**
- `rollout/ep_rew_mean` - 平均回合奖励（最重要）
- `rollout/ep_len_mean` - 平均回合长度
- `train/learning_rate` - 当前学习率
- `train/entropy_loss` - 探索程度

---

### 4. 测试训练模型 / Test Trained Model

```bash
# 测试单个模型 / Test single model
python test_cnn_v2.py trained_models_cnn_v2_mps/ppo_snake_final_v2.zip 10

# 对比多个模型 / Compare multiple models
python test_cnn_v2.py --compare model1.zip model2.zip model3.zip 50
```

---

## 训练脚本说明 / Training Scripts Explained

### `train_cnn_simple.py` ⭐ (推荐)

**用途 / Purpose:** 日常训练和参数实验

**特点 / Features:**
- 从 `train_config.py` 读取所有参数
- 显示配置摘要和性能建议
- 训练前需要确认
- 便于快速调整参数

**何时使用 / When to Use:**
- 需要频繁调整参数
- 进行对比实验
- 日常训练任务

---

### `train_cnn.py` (基准)

**用途 / Purpose:** 原始基准训练

**特点 / Features:**
- 保留原始训练配置
- 参数硬编码
- 作为性能基准参考

**何时使用 / When to Use:**
- 需要与原始版本对比
- 验证改进效果
- 作为基准测试

---

### `train_mlp.py`

**用途 / Purpose:** MLP网络训练

**特点 / Features:**
- 使用13维特征向量
- 训练速度最快
- 观察空间更小

**何时使用 / When to Use:**
- 快速原型验证
- 资源受限环境
- 对比CNN vs MLP

---

### `train_cnn_curriculum.py`

**用途 / Purpose:** 课程学习训练

**特点 / Features:**
- 渐进式训练（6×6 → 12×12）
- 更高的样本效率
- 更稳定的学习曲线

**何时使用 / When to Use:**
- 追求更稳定的训练
- 样本效率优先
- 从简单到复杂的学习

---

## 配置文件说明 / Configuration File

### `train_config.py`

集中管理所有训练超参数 / Centralized training hyperparameters

**主要配置项 / Main Configurations:**

```python
# 环境设置 / Environment Settings
NUM_ENV = 32                    # 并行环境数量
DEVICE = "mps"/"cuda"/"cpu"     # 自动检测

# 训练超参数 / Training Hyperparameters
TOTAL_TIMESTEPS = 100_000_000   # 总训练步数
N_STEPS = 2048                  # 每次更新前的步数
BATCH_SIZE = 1024               # 批次大小
N_EPOCHS = 4                    # 每次更新的轮数
GAMMA = 0.99                    # 折扣因子
LEARNING_RATE_START = 1e-4      # 初始学习率
LEARNING_RATE_END = 1e-6        # 最终学习率

# 探索与稳定性 / Exploration & Stability
ENT_COEF = 0.01                 # 熵系数（探索）
VF_COEF = 0.5                   # 价值函数系数
MAX_GRAD_NORM = 0.5             # 梯度裁剪

# 检查点 / Checkpoints
CHECKPOINT_INTERVAL = 15625     # 保存频率
EVAL_FREQ = 7812                # 评估频率
```

---

## 性能对比 / Performance Comparison

| 模型 | 训练速度 | 最终奖励 | 稳定性 | 推荐场景 |
|------|----------|----------|--------|----------|
| CNN (Simple) | ⚡⚡ | ~15-17 | ⭐⭐⭐ | **日常训练** ⭐ |
| CNN (Baseline) | ⚡ | ~13 | ⭐⭐⭐ | 基准对比 |
| MLP | ⚡⚡⚡ | ~17 | ⭐⭐ | 快速原型 |
| Curriculum | ⚡⚡ | ~14-16 | ⭐⭐⭐ | 稳定训练 |

---

## 常见问题 / FAQ

### Q: 为什么有这么多Python进程？
A: 每个并行环境运行在独立进程中。32个环境 = 32个子进程 + 1个主进程。

### Q: 训练需要多长时间？
A: 100M步约需8-12小时（取决于硬件）。可在 `train_config.py` 中调整 `TOTAL_TIMESTEPS`。

### Q: 如何加速训练？
A: 
1. 增加 `NUM_ENV`（如果内存足够）
2. 减少 `TOTAL_TIMESTEPS`
3. 使用GPU/MPS而非CPU

### Q: 训练崩溃了怎么办？
A: 
1. 降低学习率（`LEARNING_RATE_START`）
2. 减少训练轮数（`N_EPOCHS`）
3. 减少并行环境数（`NUM_ENV`）

### Q: 如何提高性能？
A:
1. 增加训练时间（`TOTAL_TIMESTEPS`）
2. 调整奖励函数（在 wrapper 中）
3. 使用课程学习（`train_cnn_curriculum.py`）

---

## 训练时长参考 / Training Time Reference

| 总步数 | 预计时间 | 适用场景 |
|--------|----------|----------|
| 10M | 1-2小时 | 快速测试 |
| 50M | 4-6小时 | 中等训练 |
| 100M | 8-12小时 | 完整训练 ⭐ |
| 200M | 16-24小时 | 深度训练 |

---

## 输出目录说明 / Output Directories

### 模型保存 / Model Saves
```
trained_models_cnn_v2_{device}/
├── ppo_snake_v2_1000000_steps.zip    # 检查点
├── ppo_snake_v2_2000000_steps.zip
├── ...
├── ppo_snake_final_v2.zip            # 最终模型
├── best_model.zip                    # 最佳模型（评估）
└── training_log.txt                  # 训练日志
```

### TensorBoard日志 / TensorBoard Logs
```
logs/
├── PPO_CNN_V2/                       # train_cnn_simple.py
├── PPO_CNN/                          # train_cnn.py
├── PPO_MLP/                          # train_mlp.py
└── PPO_CNN_CURRICULUM/               # train_cnn_curriculum.py
```

---

## 最佳实践 / Best Practices

### 1. 开始新训练前
- ✅ 检查 `train_config.py` 配置
- ✅ 确保有足够的磁盘空间（~5GB）
- ✅ 关闭其他占用GPU的程序

### 2. 训练过程中
- ✅ 定期查看TensorBoard监控进度
- ✅ 注意 `ep_rew_mean` 是否持续上升
- ✅ 如果性能下降，及时停止

### 3. 训练完成后
- ✅ 使用 `test_cnn_v2.py` 测试性能
- ✅ 对比多个检查点找最佳模型
- ✅ 保存训练日志和配置

---

## 故障排除 / Troubleshooting

### 训练不稳定 / Training Unstable
```python
# 在 train_config.py 中调整
LEARNING_RATE_START = 5e-5  # 降低学习率
N_EPOCHS = 3                # 减少训练轮数
```

### 内存不足 / Out of Memory
```python
# 在 train_config.py 中调整
NUM_ENV = 16                # 减少并行环境
BATCH_SIZE = 512            # 减小批次大小
```

### 探索不足 / Insufficient Exploration
```python
# 在 train_config.py 中调整
ENT_COEF = 0.02             # 增加熵系数
```

---

## 更多信息 / More Information

详细的项目架构说明请查看：
- `PROJECT_ARCHITECTURE.md` - 完整的项目架构文档
- `../TRAINING_GUIDE.md` - 训练指南和高级技巧

---

**最后更新 / Last Updated:** 2024-12-09
