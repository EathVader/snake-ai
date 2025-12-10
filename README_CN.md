# SnakeAI - 深度强化学习贪吃蛇游戏

简体中文 | [English](README.md)

使用深度强化学习PPO（Proximal Policy Optimization）算法训练的贪吃蛇游戏AI。项目包含基于CNN和MLP的智能体，其中CNN版本性能更优。

## 🎮 特性

- **经典贪吃蛇游戏** - 使用Pygame实现的可玩游戏
- **CNN智能体** - 基于卷积神经网络的视觉输入智能体
- **MLP智能体** - 基于多层感知机的特征输入智能体
- **课程学习** - 从简单到复杂的渐进式训练
- **动作掩码** - 防止非法移动，提高训练效率
- **并行训练** - 多进程环境加速学习

## 📊 性能对比

| 模型 | 训练速度 | 平均奖励 | 稳定性 | 推荐场景 |
|------|----------|----------|--------|----------|
| CNN (改进版) | ⚡⚡ | ~15-17 | ⭐⭐⭐ | **生产环境** ⭐ |
| MLP | ⚡⚡⚡ | ~17 | ⭐⭐ | 快速原型 |
| 课程学习 | ⚡⚡ | ~14-16 | ⭐⭐⭐ | 稳定训练 |

## 🚀 快速开始

### 环境要求

- Python 3.11+
- Conda（推荐）
- 支持CUDA的GPU（可选，用于加速训练）

### 安装

#### 方式1：自动安装（推荐）

```bash
# CUDA GPU系统
conda env create -f environment.yml

# 仅CPU系统
conda env create -f environment-cpu.yml

# 激活环境
conda activate SnakeAI-new

# 验证安装
python utils/check_cuda_status.py
```

#### 方式2：手动安装

```bash
# 创建conda环境
conda create -n SnakeAI-new python=3.11
conda activate SnakeAI-new

# 安装PyTorch（选择一个）
# CUDA 12.1版本
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 仅CPU版本
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# 安装其他依赖
conda install pygame numpy tensorboard -c conda-forge
pip install gymnasium stable-baselines3 sb3-contrib

# 验证安装
python utils/check_cuda_status.py
```

**当前环境版本：**
- Python: 3.11.14
- PyTorch: 2.5.1
- Stable-Baselines3: 2.7.1
- Gymnasium: 1.2.2
- Pygame: 2.6.1

### 玩游戏

```bash
cd main
python snake_game.py
```

### 训练你自己的智能体

```bash
cd main

# 推荐：基于配置文件的训练
python train_cnn_simple.py

# 或使用其他训练脚本
python train_cnn.py          # 基准CNN
python train_mlp.py          # MLP版本
python train_cnn_curriculum.py  # 课程学习
```

### 测试训练好的模型

```bash
cd main

# 测试单个模型
python test_cnn_v2.py trained_models_cnn_v2_mps/ppo_snake_final_v2.zip 10

# 对比多个模型
python test_cnn_v2.py --compare model1.zip model2.zip model3.zip 50
```

### 监控训练

```bash
# 启动TensorBoard
tensorboard --logdir main/logs

# 在浏览器打开 http://localhost:6006
```

## 📁 项目结构

```
snake-ai/
├── main/                           # 主代码目录
│   ├── snake_game.py              # 游戏引擎
│   ├── snake_game_custom_wrapper_cnn_v2.py  # CNN环境包装器
│   ├── snake_game_custom_wrapper_mlp.py     # MLP环境包装器
│   ├── train_config.py            # 集中训练配置
│   ├── train_cnn_simple.py        # ⭐ 推荐训练脚本
│   ├── train_cnn.py               # 基准CNN训练
│   ├── train_mlp.py               # MLP训练
│   ├── train_cnn_curriculum.py    # 课程学习
│   ├── test_cnn_v2.py             # CNN模型测试
│   ├── test_mlp.py                # MLP模型测试
│   ├── hamiltonian_agent.py       # 基准算法
│   ├── trained_models_*/          # 保存的模型
│   ├── logs/                      # TensorBoard日志
│   ├── sound/                     # 音效文件
│   ├── README.md                  # 详细使用指南
│   └── PROJECT_ARCHITECTURE.md    # 架构文档
├── utils/                         # 工具脚本
│   ├── check_gpu_status.py        # GPU检测
│   └── compress_code.py           # 代码压缩工具
├── README.md                      # 英文README
├── README_CN.md                   # 本文件
├── TRAINING_GUIDE.md              # 训练指南
├── requirements.txt               # Python依赖
├── train_with_conda.sh            # 训练启动脚本
└── monitor_training.sh            # 训练监控脚本
```

## 🎯 训练配置

编辑 `main/train_config.py` 调整训练参数：

```python
# 关键参数
NUM_ENV = 32                    # 并行环境数量
TOTAL_TIMESTEPS = 100_000_000   # 总训练步数（约8-12小时）
LEARNING_RATE_START = 1e-4      # 初始学习率
N_EPOCHS = 4                    # 每次更新的训练轮数
BATCH_SIZE = 1024               # 批次大小
GAMMA = 0.99                    # 折扣因子
```

## 📈 训练技巧

### 稳定训练
- 使用 `train_cnn_simple.py` 和默认配置
- 在TensorBoard中监控 `rollout/ep_rew_mean`
- 每1M步保存检查点

### 加速训练
- 增加 `NUM_ENV`（如果内存足够）
- 使用GPU/MPS加速
- 减少 `TOTAL_TIMESTEPS` 进行快速测试

### 训练崩溃
- 降低 `LEARNING_RATE_START`（如5e-5）
- 减少 `N_EPOCHS`（如3）
- 降低 `NUM_ENV`（如16）

## 🔬 高级功能

### 课程学习

在逐渐增大的棋盘上渐进式训练：

```bash
python train_cnn_curriculum.py
```

阶段：6×6 → 8×8 → 10×10 → 12×12

### 哈密尔顿基准

测试理论性能上限：

```bash
python hamiltonian_agent.py
```

### 模型对比

对比多个训练好的模型：

```bash
python test_cnn_v2.py --compare \
  trained_models_cnn/ppo_snake_final.zip \
  trained_models_cnn_v2_mps/ppo_snake_final_v2.zip \
  50
```

## 📚 文档

- **[docs/](docs/)** - 完整文档中心
- **[docs/PROGRESS_REPORT.md](docs/PROGRESS_REPORT.md)** - 🆕 最新训练进度和成果
- **[docs/USAGE_GUIDE.md](docs/USAGE_GUIDE.md)** - 训练脚本详细使用指南
- **[docs/PROJECT_ARCHITECTURE.md](docs/PROJECT_ARCHITECTURE.md)** - 完整架构文档
- **[docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)** - 高级训练策略和故障排除

## 🛠️ 故障排除

### 为什么有这么多Python进程？
正常现象！每个并行环境在独立进程中运行。32个环境 = 32个子进程 + 1个主进程。

### 训练太慢？
- 增加 `NUM_ENV`（更多并行环境）
- 使用GPU/MPS而非CPU
- 减少 `TOTAL_TIMESTEPS` 进行测试

### 内存不足？
- 减少 `NUM_ENV`（如16）
- 降低 `BATCH_SIZE`（如512）
- 关闭其他应用程序

### 性能下降？
- 在 `train_config.py` 中降低学习率
- 减少 `N_EPOCHS`
- 在TensorBoard中检查不稳定迹象

## 🤝 贡献

欢迎贡献！请随时提交问题和拉取请求。

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - 强化学习算法
- [Gymnasium](https://gymnasium.farama.org/) - 强化学习环境接口
- [Pygame](https://www.pygame.org/) - 游戏引擎
- [PPO论文](https://arxiv.org/abs/1707.06347) - 算法参考

## 📞 联系方式

如有问题和讨论，请在GitHub上提交issue。

---

**最后更新：** 2024-12-09
