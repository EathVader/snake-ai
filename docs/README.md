# Snake AI Documentation / 贪吃蛇AI文档

Complete documentation for the Snake AI project.

Snake AI项目的完整文档。

---

## 📚 Documentation Overview / 文档概览

### 🏠 Getting Started / 入门指南

#### [../README.md](../README.md) | [../README_CN.md](../README_CN.md)
**Quick start guide and project overview**

**快速开始指南和项目概览**

- Installation instructions / 安装说明
- Basic usage / 基本使用
- Quick commands / 快速命令
- Troubleshooting / 故障排除

**Start here if you're new to the project!**

**如果你是新用户，从这里开始！**

---

### 📖 User Guides / 用户指南

#### [USAGE_GUIDE.md](USAGE_GUIDE.md)
**Detailed guide for training scripts and testing**

**训练脚本和测试的详细指南**

- Training script explanations / 训练脚本说明
- Configuration file usage / 配置文件使用
- Testing procedures / 测试流程
- Output directory structure / 输出目录结构
- Best practices / 最佳实践
- FAQ / 常见问题

**Read this to understand how to use the training scripts.**

**阅读此文档了解如何使用训练脚本。**

---

#### [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
**Advanced training strategies and optimization**

**高级训练策略和优化**

- Reward shaping improvements / 奖励塑形改进
- Hyperparameter tuning / 超参数调优
- Curriculum learning / 课程学习
- Advanced techniques / 高级技术
- Performance benchmarks / 性能基准
- Hardware recommendations / 硬件建议

**Read this to improve your training performance.**

**阅读此文档提高训练性能。**

---

### 🔧 Technical Documentation / 技术文档

#### [PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md)
**Complete technical architecture and implementation details**

**完整的技术架构和实现细节**

- System architecture diagrams / 系统架构图
- File-by-file functionality / 逐文件功能说明
- Data flow diagrams / 数据流图
- Training pipeline / 训练流程
- Dependency graphs / 依赖关系图
- Version history / 版本历史

**Read this to understand the codebase structure.**

**阅读此文档了解代码库结构。**

---

## 🎯 Documentation by Use Case / 按用途查找文档

### I want to... / 我想要...

#### Get Started Quickly / 快速开始
```
1. Read: ../README.md or ../README_CN.md
2. Install dependencies
3. Run: python main/train_cnn_simple.py
```

#### Understand Training Scripts / 了解训练脚本
```
1. Read: USAGE_GUIDE.md
2. Check: main/train_config.py
3. Experiment with different scripts
```

#### Improve Training Performance / 提高训练性能
```
1. Read: TRAINING_GUIDE.md
2. Adjust: main/train_config.py
3. Try: Curriculum learning or different hyperparameters
```

#### Understand the Code / 理解代码
```
1. Read: PROJECT_ARCHITECTURE.md
2. Review: Source code with architecture understanding
3. Contribute: Make improvements
```

#### Monitor Training / 监控训练
```
1. Use: ./monitor_training.sh (CLI)
2. Use: tensorboard --logdir main/logs (Visual)
3. Check: TRAINING_GUIDE.md for key metrics
```

---

## 📖 Recommended Reading Order / 推荐阅读顺序

### For Beginners / 新手
```
1. ../README.md
   Project overview and installation
   项目概览和安装
   ↓
2. USAGE_GUIDE.md
   How to use training scripts
   如何使用训练脚本
   ↓
3. Start training!
   开始训练！
```

### For Advanced Users / 高级用户
```
1. ../README.md
   Quick overview
   快速概览
   ↓
2. TRAINING_GUIDE.md
   Optimization strategies
   优化策略
   ↓
3. Experiment with configurations
   实验不同配置
```

### For Developers / 开发者
```
1. ../README.md
   Project overview
   项目概览
   ↓
2. PROJECT_ARCHITECTURE.md
   Technical architecture
   技术架构
   ↓
3. Review and contribute code
   审查和贡献代码
```

---

## 🔍 Quick Reference / 快速参考

### File Locations / 文件位置

| Document / 文档 | Purpose / 用途 |
|-----------------|----------------|
| [../README.md](../README.md) | Project overview / 项目概览 |
| [USAGE_GUIDE.md](USAGE_GUIDE.md) | Training scripts guide / 训练脚本指南 |
| [TRAINING_GUIDE.md](TRAINING_GUIDE.md) | Training optimization / 训练优化 |
| [PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md) | Technical details / 技术细节 |

### Common Commands / 常用命令

| Task / 任务 | Command / 命令 |
|-------------|----------------|
| Start training / 开始训练 | `cd main && python train_cnn_simple.py` |
| Monitor training / 监控训练 | `./monitor_training.sh` |
| View TensorBoard | `tensorboard --logdir main/logs` |
| Test model / 测试模型 | `cd main && python test_cnn_v2.py <model.zip>` |
| Adjust config / 调整配置 | Edit `main/train_config.py` |

---

## 📝 Documentation Principles / 文档原则

1. **Bilingual / 双语** - All docs in Chinese and English / 所有文档使用中英文
2. **Clear / 清晰** - Simple language and examples / 简单的语言和示例
3. **Updated / 更新** - Keep in sync with code / 与代码保持同步
4. **Linked / 链接** - Cross-reference related docs / 交叉引用相关文档
5. **Practical / 实用** - Include runnable examples / 包含可运行示例

---

## 🤝 Contributing / 贡献

When contributing documentation / 贡献文档时：

1. Follow bilingual format / 遵循双语格式
2. Use clear headings / 使用清晰的标题
3. Include code examples / 包含代码示例
4. Update this index / 更新此索引
5. Keep technical accuracy / 保持技术准确性

---

## 📞 Need Help? / 需要帮助？

- Check the FAQ in [USAGE_GUIDE.md](USAGE_GUIDE.md)
- Review troubleshooting in [../README.md](../README.md)
- Open an issue on GitHub

---

**Last Updated / 最后更新:** 2024-12-09
