#!/bin/bash
# Training script with conda environment activation
# 带conda环境激活的训练脚本

set -e

CONDA_ENV="SnakeAI-new"

echo "=========================================="
echo "Snake AI Training with Conda"
echo "使用Conda环境训练贪吃蛇AI"
echo "=========================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found"
    echo "错误：未找到conda"
    exit 1
fi

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Check if environment exists
if ! conda env list | grep -q "^${CONDA_ENV} "; then
    echo "Error: Conda environment '${CONDA_ENV}' not found"
    echo "错误：未找到Conda环境 '${CONDA_ENV}'"
    echo ""
    echo "Available environments:"
    conda env list
    echo ""
    echo "To create the environment / 创建环境:"
    echo "  conda create -n ${CONDA_ENV} python=3.11"
    echo "  conda activate ${CONDA_ENV}"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Check if required packages are installed
echo "Checking dependencies / 检查依赖..."
conda activate ${CONDA_ENV}

if ! python -c "import torch, stable_baselines3, gymnasium, pygame" 2>/dev/null; then
    echo "Warning: Some dependencies missing / 警告：缺少某些依赖"
    echo "Installing missing packages / 安装缺失的包..."
    pip install -r requirements.txt
    
    # Check for CUDA if available
    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU detected, installing CUDA PyTorch..."
        echo "检测到NVIDIA GPU，安装CUDA版PyTorch..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    fi
fi

echo "Activating conda environment: ${CONDA_ENV}"
echo "激活conda环境: ${CONDA_ENV}"
conda activate ${CONDA_ENV}

# Show current directory and git status
echo ""
echo "Current directory / 当前目录: $(pwd)"
if [ -d ".git" ]; then
    echo "Git branch / Git分支: $(git branch --show-current 2>/dev/null || echo 'unknown')"
    echo "Git status / Git状态: $(git status --porcelain | wc -l | tr -d ' ') files changed"
fi

# Check hardware and show recommendations
echo ""
echo "Checking hardware / 检查硬件..."
python utils/check_cuda_status.py

echo ""
echo "Select training mode / 选择训练模式:"
echo "1) Anti-Looping Training (Recommended) / 反转圈训练（推荐）⭐"
echo "2) Config-Based Training / 配置化训练"
echo "3) Curriculum Learning / 课程学习"
echo "4) Original Training / 原始训练"
echo ""
read -p "Enter choice (1-4) / 输入选择 (1-4): " choice

case $choice in
    1)
        echo ""
        echo "Starting Anti-Looping Training..."
        echo "开始反转圈训练..."
        echo "This will fix circular behavior and create a truly game-playing AI"
        echo "这将修复转圈行为并创建真正会玩游戏的AI"
        echo ""
        echo "Expected training time / 预期训练时间:"
        echo "  - CUDA GPU: 4-6 hours (50M steps)"
        echo "  - CPU: 24-48 hours"
        echo ""
        read -p "Continue? (y/n) / 继续？(y/n): " confirm
        if [ "$confirm" != "y" ]; then
            echo "Training cancelled / 训练已取消"
            exit 0
        fi
        cd main
        python train_cnn_anti_loop.py
        ;;
    2)
        echo ""
        echo "Starting Config-Based Training..."
        echo "开始配置化训练..."
        echo "Uses centralized configuration from train_config.py"
        echo "使用train_config.py的集中配置"
        echo ""
        echo "Expected training time / 预期训练时间:"
        echo "  - CUDA GPU: 8-12 hours (100M steps)"
        echo "  - MPS: 12-16 hours"
        echo "  - CPU: 48-72 hours"
        echo ""
        read -p "Continue? (y/n) / 继续？(y/n): " confirm
        if [ "$confirm" != "y" ]; then
            echo "Training cancelled / 训练已取消"
            exit 0
        fi
        cd main
        python train_cnn_simple.py
        ;;
    3)
        echo ""
        echo "Starting Curriculum Learning..."
        echo "开始课程学习..."
        echo "Progressive training from 6x6 to 12x12 board"
        echo "从6x6到12x12棋盘的渐进式训练"
        echo ""
        echo "Expected training time / 预期训练时间:"
        echo "  - CUDA GPU: 12-18 hours (85M steps total)"
        echo "  - MPS: 18-24 hours"
        echo "  - CPU: 72-96 hours"
        echo ""
        read -p "Continue? (y/n) / 继续？(y/n): " confirm
        if [ "$confirm" != "y" ]; then
            echo "Training cancelled / 训练已取消"
            exit 0
        fi
        cd main
        python train_cnn_curriculum.py
        ;;
    4)
        echo ""
        echo "Starting Original Training..."
        echo "开始原始训练..."
        echo "Baseline training for comparison"
        echo "用于对比的基准训练"
        echo ""
        echo "Expected training time / 预期训练时间:"
        echo "  - CUDA GPU: 8-12 hours (100M steps)"
        echo "  - CPU: 48-72 hours"
        echo ""
        read -p "Continue? (y/n) / 继续？(y/n): " confirm
        if [ "$confirm" != "y" ]; then
            echo "Training cancelled / 训练已取消"
            exit 0
        fi
        cd main
        python train_cnn.py
        ;;
    *)
        echo "Invalid choice / 无效选择"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Training Complete! / 训练完成！"
echo "=========================================="
echo ""
echo "Next Steps / 下一步:"
echo ""
echo "1. Test your model / 测试模型:"
echo "   conda activate ${CONDA_ENV}"
echo "   cd main"
echo "   python test_cnn_v2.py <model_path> 10"
echo ""
echo "2. View training logs / 查看训练日志:"
echo "   tensorboard --logdir main/logs"
echo ""
echo "3. Compare models / 对比模型:"
echo "   python test_cnn_v2.py --compare model1.zip model2.zip 50"
echo ""
echo "4. Check training progress / 检查训练进度:"
echo "   ./monitor_training.sh"
echo ""
echo "Model locations / 模型位置:"
case $choice in
    1)
        echo "   main/trained_models_cnn_anti_loop_*/ppo_snake_anti_loop_final.zip"
        ;;
    2)
        echo "   main/trained_models_cnn_v2_*/ppo_snake_final_v2.zip"
        ;;
    3)
        echo "   main/trained_models_cnn_curriculum/ppo_snake_curriculum_final.zip"
        ;;
    4)
        echo "   main/trained_models_cnn/ppo_snake_final.zip"
        ;;
esac
echo ""
