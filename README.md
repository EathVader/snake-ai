# SnakeAI - Deep Reinforcement Learning Snake Game

[简体中文](README_CN.md) | English

A Snake game AI trained with Deep Reinforcement Learning using PPO (Proximal Policy Optimization) algorithm. The project includes CNN and MLP-based agents, with the CNN version achieving superior performance.

## 🎮 Features

- **Classic Snake Game** - Playable game implementation using Pygame
- **CNN Agent** - Convolutional Neural Network based agent with visual input
- **MLP Agent** - Multi-Layer Perceptron agent with feature-based input
- **Curriculum Learning** - Progressive training from simple to complex
- **Action Masking** - Prevents invalid moves for efficient training
- **Parallel Training** - Multi-process environment for faster learning

## 📊 Performance

| Model | Training Speed | Avg Reward | Stability | Use Case |
|-------|---------------|------------|-----------|----------|
| CNN (Improved) | ⚡⚡ | ~15-17 | ⭐⭐⭐ | **Production** ⭐ |
| MLP | ⚡⚡⚡ | ~17 | ⭐⭐ | Fast Prototyping |
| Curriculum | ⚡⚡ | ~14-16 | ⭐⭐⭐ | Stable Training |

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Conda (recommended)
- CUDA-capable GPU (optional, for faster training)

### Installation

```bash
# Create conda environment
conda create -n SnakeAI-new python=3.11
conda activate SnakeAI-new

# Install dependencies
pip install -r requirements.txt

# [Recommended] For GPU training on NVIDIA
# Check your CUDA version first: nvidia-smi
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# [Optional] For Apple Silicon (M1/M2/M3)
# MPS (Metal Performance Shaders) is automatically detected
# PyTorch 2.5+ has native MPS support

# Verify installation
python utils/check_cuda_status.py
```

**Current Environment Versions:**
- Python: 3.11.14
- PyTorch: 2.5.1
- Stable-Baselines3: 2.7.1
- Gymnasium: 1.2.2
- Pygame: 2.6.1

### Play the Game

```bash
cd main
python snake_game.py
```

### Train Your Own Agent

```bash
cd main

# Recommended: Config-based training
python train_cnn_simple.py

# Or use other training scripts
python train_cnn.py          # Baseline CNN
python train_mlp.py          # MLP version
python train_cnn_curriculum.py  # Curriculum learning
```

### Test Trained Model

```bash
cd main

# Test single model
python test_cnn_v2.py trained_models_cnn_v2_mps/ppo_snake_final_v2.zip 10

# Compare multiple models
python test_cnn_v2.py --compare model1.zip model2.zip model3.zip 50
```

### Monitor Training

```bash
# Start TensorBoard
tensorboard --logdir main/logs

# Open browser to http://localhost:6006
```

## 📁 Project Structure

```
snake-ai/
├── main/                           # Main code directory
│   ├── snake_game.py              # Game engine
│   ├── snake_game_custom_wrapper_cnn_v2.py  # CNN environment wrapper
│   ├── snake_game_custom_wrapper_mlp.py     # MLP environment wrapper
│   ├── train_config.py            # Centralized training config
│   ├── train_cnn_simple.py        # ⭐ Recommended training script
│   ├── train_cnn.py               # Baseline CNN training
│   ├── train_mlp.py               # MLP training
│   ├── train_cnn_curriculum.py    # Curriculum learning
│   ├── test_cnn_v2.py             # CNN model testing
│   ├── test_mlp.py                # MLP model testing
│   ├── hamiltonian_agent.py       # Baseline algorithm
│   ├── trained_models_*/          # Saved models
│   ├── logs/                      # TensorBoard logs
│   └── sound/                     # Sound effects
├── docs/                          # Documentation
│   ├── README.md                  # Documentation hub
│   ├── USAGE_GUIDE.md             # Training scripts guide
│   ├── TRAINING_GUIDE.md          # Training optimization
│   └── PROJECT_ARCHITECTURE.md    # Technical architecture
├── utils/                         # Utility scripts
│   ├── check_gpu_status.py        # GPU detection
│   └── compress_code.py           # Code compression tool
├── README.md                      # This file
├── README_CN.md                   # Chinese README
├── requirements.txt               # Python dependencies
├── train_with_conda.sh            # Training launcher script
└── monitor_training.sh            # Training monitor script
```

## 🎯 Training Configuration

Edit `main/train_config.py` to adjust training parameters:

```python
# Key parameters
NUM_ENV = 32                    # Parallel environments
TOTAL_TIMESTEPS = 100_000_000   # Total training steps (~8-12 hours)
LEARNING_RATE_START = 1e-4      # Initial learning rate
N_EPOCHS = 4                    # Training epochs per update
BATCH_SIZE = 1024               # Batch size
GAMMA = 0.99                    # Discount factor
```

## 📈 Training Tips

### For Stable Training
- Use `train_cnn_simple.py` with default config
- Monitor `rollout/ep_rew_mean` in TensorBoard
- Save checkpoints every 1M steps

### For Faster Training
- Increase `NUM_ENV` (if memory allows)
- Use GPU/MPS acceleration
- Reduce `TOTAL_TIMESTEPS` for quick tests

### If Training Crashes
- Reduce `LEARNING_RATE_START` (e.g., 5e-5)
- Decrease `N_EPOCHS` (e.g., 3)
- Lower `NUM_ENV` (e.g., 16)

## 🔬 Advanced Features

### Curriculum Learning

Train progressively on increasing board sizes:

```bash
python train_cnn_curriculum.py
```

Stages: 6×6 → 8×8 → 10×10 → 12×12

### Hamiltonian Baseline

Test the theoretical upper bound:

```bash
python hamiltonian_agent.py
```

### Model Comparison

Compare multiple trained models:

```bash
python test_cnn_v2.py --compare \
  trained_models_cnn/ppo_snake_final.zip \
  trained_models_cnn_v2_mps/ppo_snake_final_v2.zip \
  50
```

## 📚 Documentation

- **[docs/](docs/)** - Complete documentation hub
- **[docs/PROGRESS_REPORT.md](docs/PROGRESS_REPORT.md)** - 🆕 Latest training progress and achievements
- **[docs/USAGE_GUIDE.md](docs/USAGE_GUIDE.md)** - Detailed usage guide for training scripts
- **[docs/PROJECT_ARCHITECTURE.md](docs/PROJECT_ARCHITECTURE.md)** - Complete architecture documentation
- **[docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)** - Advanced training strategies and troubleshooting

## 🛠️ Troubleshooting

### Many Python Processes?
Normal! Each parallel environment runs in its own process. 32 environments = 32 child processes + 1 main process.

### Training Too Slow?
- Increase `NUM_ENV` (more parallel environments)
- Use GPU/MPS instead of CPU
- Reduce `TOTAL_TIMESTEPS` for testing

### Out of Memory?
- Reduce `NUM_ENV` (e.g., 16)
- Decrease `BATCH_SIZE` (e.g., 512)
- Close other applications

### Performance Degradation?
- Lower learning rate in `train_config.py`
- Reduce `N_EPOCHS`
- Check TensorBoard for instability signs

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - RL algorithms
- [Gymnasium](https://gymnasium.farama.org/) - RL environment interface
- [Pygame](https://www.pygame.org/) - Game engine
- [PPO Paper](https://arxiv.org/abs/1707.06347) - Algorithm reference

## 📞 Contact

For questions and discussions, please open an issue on GitHub.

---

**Last Updated:** 2024-12-09
