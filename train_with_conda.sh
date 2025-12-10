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
    exit 1
fi

echo "Activating conda environment: ${CONDA_ENV}"
echo "激活conda环境: ${CONDA_ENV}"
conda activate ${CONDA_ENV}

echo ""
echo "Select training mode / 选择训练模式:"
echo "1) Enhanced Training (Recommended) / 增强训练（推荐）"
echo "2) Curriculum Learning / 课程学习"
echo "3) Original Training / 原始训练"
echo ""
read -p "Enter choice (1-3) / 输入选择 (1-3): " choice

case $choice in
    1)
        echo ""
        echo "Starting Enhanced Training..."
        echo "开始增强训练..."
        echo ""
        cd main
        python train_cnn_v2.py
        ;;
    2)
        echo ""
        echo "Starting Curriculum Learning..."
        echo "开始课程学习..."
        echo ""
        cd main
        python train_cnn_curriculum.py
        ;;
    3)
        echo ""
        echo "Starting Original Training..."
        echo "开始原始训练..."
        echo ""
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
echo "To test your model / 测试模型:"
echo "  conda activate ${CONDA_ENV}"
echo "  cd main"
echo "  python test_cnn_v2.py <model_path>"
echo ""
echo "To view training logs / 查看训练日志:"
echo "  tensorboard --logdir main/logs"
echo ""
