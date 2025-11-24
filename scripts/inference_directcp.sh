#!/bin/bash
# Directed-CP Inference Script
# Usage: bash scripts/inference_directcp.sh <model_dir>

cd /data2/user2/yihang/Direct-CP-Lidar/Directed-CP

# check if the model directory is provided
if [ -z "$1" ]; then
    echo "Error: Please provide model directory"
    echo "Usage: bash scripts/inference_directcp.sh <model_dir>"
    exit 1
fi

MODEL_DIR=$1
FUSION_METHOD="intermediate"
SAVE_VIS_INTERVAL=10
NOTE="directcp_inference"

# print the configuration
echo "=========================================="
echo "Starting Directed-CP Inference"
echo "=========================================="
echo "Model Dir: $MODEL_DIR"
echo "Fusion Method: $FUSION_METHOD"
echo "Visualization Interval: $SAVE_VIS_INTERVAL"
echo "Note: $NOTE"
echo "=========================================="

# run the inference
python opencood/tools/inference.py \
    --model_dir $MODEL_DIR \
    --fusion_method $FUSION_METHOD \
    --save_vis_interval $SAVE_VIS_INTERVAL \
    --save_npy \
    --note $NOTE

echo "=========================================="
echo "Inference completed!"
echo "Results saved in: $MODEL_DIR"
echo "=========================================="

