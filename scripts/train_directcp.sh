#!/bin/bash
# Directed-CP Training Script
# Usage: bash scripts/train_directcp.sh

# set the working directory
cd /data2/user2/yihang/Direct-CP-Lidar/Directed-CP

# training parameters
CONFIG="opencood/hypes_yaml/v2xsim2/v2xsim2_directcp_attn_multiscale_resnet.yaml"
MODEL_DIR="" # leave empty to train from scratch, fill in path to continue training

# print the configuration
echo "=========================================="
echo "Starting Directed-CP Training"
echo "=========================================="
echo "Config: $CONFIG"
echo "Model Dir: ${MODEL_DIR:-'Training from scratch'}"
echo "=========================================="

# run the training
if [ -z "$MODEL_DIR" ]; then
    # train from scratch
    python opencood/tools/train.py \
        --hypes_yaml $CONFIG \
        --fusion_method intermediate
else
    # continue training
    python opencood/tools/train.py \
        --hypes_yaml $CONFIG \
        --model_dir $MODEL_DIR \
        --fusion_method intermediate
fi

echo "=========================================="
echo "Training completed!"
echo "=========================================="

