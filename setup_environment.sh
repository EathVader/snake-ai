#!/bin/bash
# Quick environment setup script for Snake AI
# Snake AI 快速环境设置脚本

set -e

echo "=========================================="
echo "Snake AI Environment Setup"
echo "贪吃蛇AI环境设置"
echo "=========================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda not found"
    echo "❌ 错误：未找到conda"
    echo ""
    echo "Please install Anaconda or Miniconda first:"
    echo "请先安装Anaconda或Miniconda："
    echo "https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Initialize conda for bash
eval "$(conda shell.bash hook)"

echo "✅ Conda found / 找到Conda"
echo ""

# Detect hardware
echo "Detecting hardware / 检测硬件..."
if command -v nvidia-smi &> /dev/null; then
    echo "🚀 NVIDIA GPU detected / 检测到NVIDIA GPU"
    ENV_FILE="environment.yml"
    ENV_TYPE="CUDA"
else
    echo "💻 CPU-only system detected / 检测到仅CPU系统"
    ENV_FILE="environment-cpu.yml"
    ENV_TYPE="CPU"
fi

echo ""
echo "Environment type / 环境类型: $ENV_TYPE"
echo "Using file / 使用文件: $ENV_FILE"
echo ""

# Check if environment already exists
ENV_NAME="SnakeAI-new"
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "⚠️  Environment '${ENV_NAME}' already exists"
    echo "⚠️  环境 '${ENV_NAME}' 已存在"
    echo ""
    read -p "Remove and recreate? (y/n) / 删除并重新创建？(y/n): " confirm
    if [ "$confirm" = "y" ]; then
        echo "Removing existing environment / 删除现有环境..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Setup cancelled / 设置已取消"
        exit 0
    fi
fi

# Create environment
echo "Creating environment / 创建环境..."
conda env create -f $ENV_FILE

echo ""
echo "✅ Environment created successfully / 环境创建成功"
echo ""

# Activate and test
echo "Testing installation / 测试安装..."
conda activate ${ENV_NAME}

if python -c "import torch, stable_baselines3, gymnasium, pygame" 2>/dev/null; then
    echo "✅ All packages installed successfully / 所有包安装成功"
else
    echo "❌ Some packages failed to install / 某些包安装失败"
    exit 1
fi

# Run hardware check
echo ""
echo "Running hardware check / 运行硬件检查..."
python utils/check_cuda_status.py

echo ""
echo "=========================================="
echo "Setup Complete! / 设置完成！"
echo "=========================================="
echo ""
echo "To activate the environment / 激活环境:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To start training / 开始训练:"
echo "  ./train_with_conda.sh"
echo ""
echo "To test the game / 测试游戏:"
echo "  cd main && python snake_game.py"
echo ""