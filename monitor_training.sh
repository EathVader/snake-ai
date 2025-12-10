#!/bin/bash
# Monitor training progress and resource usage
# 监控训练进度和资源使用

echo "=========================================="
echo "Snake AI Training Monitor"
echo "贪吃蛇AI训练监控"
echo "=========================================="
echo ""

# Check if training is running
if ! pgrep -f "train_cnn" > /dev/null; then
    echo "No training process found"
    echo "未找到训练进程"
    exit 1
fi

echo "Training is running / 训练正在运行"
echo ""

# Count Python processes
PYTHON_PROCS=$(pgrep -f "python.*train_cnn" | wc -l | tr -d ' ')
echo "Python processes / Python进程数: $PYTHON_PROCS"
echo ""

# Show resource usage
echo "Resource Usage / 资源使用:"
echo "----------------------------------------"

# CPU and Memory
echo "Top Python processes by CPU:"
ps aux | grep python | grep -v grep | head -5 | awk '{printf "  PID: %-8s CPU: %-6s MEM: %-6s CMD: %s\n", $2, $3"%", $4"%", $11}'

echo ""
echo "Total Memory Usage / 总内存使用:"
ps aux | grep python | grep -v grep | awk '{sum+=$4} END {printf "  %.1f%% of RAM\n", sum}'

echo ""
echo "----------------------------------------"
echo ""

# Check for log files
echo "Recent Training Logs / 最近的训练日志:"
echo "----------------------------------------"

# Find most recent log directory
if [ -d "main/trained_models_cnn_v2_mps" ]; then
    LOG_DIR="main/trained_models_cnn_v2_mps"
elif [ -d "main/trained_models_cnn_v2_cuda" ]; then
    LOG_DIR="main/trained_models_cnn_v2_cuda"
else
    LOG_DIR=""
fi

if [ -n "$LOG_DIR" ] && [ -f "$LOG_DIR/training_log.txt" ]; then
    echo "Last 10 lines from training log:"
    tail -10 "$LOG_DIR/training_log.txt"
else
    echo "No training log found yet"
    echo "训练日志尚未生成"
fi

echo ""
echo "----------------------------------------"
echo ""
echo "Tips / 提示:"
echo "- Training alternates between data collection (low GPU) and"
echo "  neural network updates (high GPU)"
echo "- 训练在数据收集（低GPU）和神经网络更新（高GPU）之间交替"
echo ""
echo "- One main process uses GPU heavily during training phase"
echo "- 一个主进程在训练阶段大量使用GPU"
echo ""
echo "- Multiple child processes collect game data using CPU"
echo "- 多个子进程使用CPU收集游戏数据"
echo ""
echo "To view TensorBoard / 查看TensorBoard:"
echo "  tensorboard --logdir main/logs"
echo ""
echo "To stop training / 停止训练:"
echo "  pkill -f train_cnn"
echo ""
