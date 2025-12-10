# Snake AI Training Guide / 贪吃蛇AI训练指南

> **Note:** For detailed project structure and file descriptions, see [main/PROJECT_ARCHITECTURE.md](main/PROJECT_ARCHITECTURE.md)
> 
> **注意：** 详细的项目结构和文件说明请查看 [main/PROJECT_ARCHITECTURE.md](main/PROJECT_ARCHITECTURE.md)

## Overview / 概述

This guide provides strategies to train better-performing Snake AI agents using reinforcement learning.

本指南提供了使用强化学习训练更高性能贪吃蛇AI的策略。

---

## Key Improvements / 主要改进

### 1. Enhanced Reward Shaping / 改进的奖励塑形

**Original Issues / 原始问题:**
- Rewards too small and sparse / 奖励过小且稀疏
- Death penalty not well-scaled / 死亡惩罚缩放不当
- Insufficient guidance for food-seeking / 寻找食物的引导不足

**New Reward Structure / 新奖励结构:**

```python
# Food obtained / 吃到食物
reward = 10.0 + (snake_size - init_size) * 0.5  # 10-15 range

# Moving towards food / 靠近食物
reward = +0.1

# Moving away from food / 远离食物  
reward = -0.15

# Death penalty / 死亡惩罚
reward = -10.0 * (1.0 - progress)  # Scaled by progress

# Victory / 胜利
reward = 100.0
```

### 2. Improved Hyperparameters / 改进的超参数

**Key Changes / 关键变化:**

| Parameter | Old Value | New Value | Reason |
|-----------|-----------|-----------|--------|
| `gamma` | 0.94 | 0.99 | Better long-term planning / 更好的长期规划 |
| `n_epochs` | 4 | 10 | More thorough policy updates / 更彻底的策略更新 |
| `batch_size` | 512 | 1024 | More stable gradients / 更稳定的梯度 |
| `ent_coef` | default | 0.01 | Encourage exploration / 鼓励探索 |
| `learning_rate` | 2.5e-4 | 3e-4 → 1e-6 | Better initial learning / 更好的初始学习 |

### 3. Curriculum Learning / 课程学习

Train progressively on increasing board sizes:
逐步在增大的棋盘上训练：

1. **Stage 0**: 6x6 board (5M steps) - Learn basic movement / 学习基本移动
2. **Stage 1**: 8x8 board (10M steps) - Develop strategies / 发展策略
3. **Stage 2**: 10x10 board (20M steps) - Handle complexity / 处理复杂性
4. **Stage 3**: 12x12 board (50M steps) - Master full game / 掌握完整游戏

---

## Training Scripts / 训练脚本

### Option 1: Enhanced Training (Recommended) / 增强训练（推荐）

```bash
cd main
python train_cnn_v2.py
```

**Features / 特性:**
- Improved reward shaping / 改进的奖励塑形
- Better hyperparameters / 更好的超参数
- Evaluation callback / 评估回调
- Saves best model / 保存最佳模型

**Expected Results / 预期结果:**
- Better food-seeking behavior / 更好的寻找食物行为
- Fewer early deaths / 更少的早期死亡
- Higher average snake length / 更高的平均蛇长

### Option 2: Curriculum Learning / 课程学习

```bash
cd main
python train_cnn_curriculum.py
```

**Features / 特性:**
- Progressive difficulty / 渐进式难度
- Transfer learning between stages / 阶段间迁移学习
- More sample-efficient / 更高的样本效率

**Best For / 最适合:**
- Limited compute resources / 有限的计算资源
- Faster convergence / 更快的收敛
- More stable training / 更稳定的训练

### Option 3: Original Training / 原始训练

```bash
cd main
python train_cnn.py  # or train_mlp.py
```

---

## Monitoring Training / 监控训练

### TensorBoard

```bash
tensorboard --logdir main/logs
```

**Key Metrics to Watch / 关键指标:**
- `rollout/ep_rew_mean`: Average episode reward / 平均回合奖励
- `rollout/ep_len_mean`: Average episode length / 平均回合长度
- `train/learning_rate`: Current learning rate / 当前学习率
- `train/entropy_loss`: Exploration level / 探索水平

### Training Logs / 训练日志

Check `trained_models_*/training_log.txt` for detailed output.
查看 `trained_models_*/training_log.txt` 获取详细输出。

---

## Advanced Techniques / 高级技术

### 1. Data Augmentation / 数据增强

Add random rotations/flips to observations:
对观察添加随机旋转/翻转：

```python
def augment_observation(obs):
    # Random rotation (0, 90, 180, 270 degrees)
    k = np.random.randint(0, 4)
    obs = np.rot90(obs, k)
    
    # Random flip
    if np.random.rand() > 0.5:
        obs = np.fliplr(obs)
    
    return obs
```

### 2. Prioritized Experience Replay / 优先经验回放

Use PPO with prioritized sampling for important transitions.
使用PPO和优先采样处理重要转换。

### 3. Self-Play / 自我对弈

Train multiple agents and select best performers.
训练多个智能体并选择最佳表现者。

### 4. Ensemble Methods / 集成方法

Combine multiple trained models for more robust decisions.
组合多个训练模型以获得更稳健的决策。

---

## Troubleshooting / 故障排除

### Problem: Agent gets stuck in loops / 智能体陷入循环

**Solution:**
- Increase entropy coefficient / 增加熵系数
- Add penalty for revisiting positions / 添加重访位置的惩罚
- Use longer step limits / 使用更长的步数限制

### Problem: Training is unstable / 训练不稳定

**Solution:**
- Reduce learning rate / 降低学习率
- Increase batch size / 增加批次大小
- Use gradient clipping / 使用梯度裁剪

### Problem: Agent doesn't explore / 智能体不探索

**Solution:**
- Increase entropy coefficient / 增加熵系数
- Use curriculum learning / 使用课程学习
- Add exploration bonus / 添加探索奖励

---

## Performance Benchmarks / 性能基准

### Expected Performance After Training / 训练后的预期性能

| Metric | Original | Enhanced | Curriculum |
|--------|----------|----------|------------|
| Avg Length | 8-12 | 15-25 | 20-35 |
| Win Rate (12x12) | <1% | 2-5% | 5-15% |
| Training Time | 24h | 24h | 36h |

---

## Hardware Recommendations / 硬件建议

### Minimum / 最低配置
- GPU: GTX 1060 or equivalent / GTX 1060或同等显卡
- RAM: 16GB
- Training time: ~48 hours / 训练时间：约48小时

### Recommended / 推荐配置
- GPU: RTX 3070 or better / RTX 3070或更好
- RAM: 32GB
- Training time: ~24 hours / 训练时间：约24小时

### Optimal / 最优配置
- GPU: RTX 4090 or A100 / RTX 4090或A100
- RAM: 64GB
- Training time: ~12 hours / 训练时间：约12小时

---

## Next Steps / 下一步

1. **Start with Enhanced Training** / 从增强训练开始
   ```bash
   python main/train_cnn_v2.py
   ```

2. **Monitor Progress** / 监控进度
   ```bash
   tensorboard --logdir main/logs
   ```

3. **Test Trained Model** / 测试训练模型
   ```bash
   python main/test_cnn.py
   ```

4. **Iterate and Improve** / 迭代和改进
   - Adjust hyperparameters / 调整超参数
   - Try curriculum learning / 尝试课程学习
   - Experiment with rewards / 实验奖励函数

---

## References / 参考资料

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [Reward Shaping](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf)
- [Curriculum Learning](https://arxiv.org/abs/2003.04960)
