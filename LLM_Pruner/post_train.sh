#!/bin/bash

# 设置 CUDA 设备和 WANDB 模式
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0

DATA_PATH="data/alpaca-cleaned"
LORA_R=8
NUM_EPOCHS=2
LEARNING_RATE=1e-4
BATCH_SIZE=64

# 定义模型和输出目录
declare -A MODELS
# MODELS["llama2-7b-prune0.24"]="prune_log/llama2-7b-prune0.24/pytorch_model.bin"


OUTPUT_DIR="tune_log"

# 循环执行微调任务
for MODEL_NAME in "${!MODELS[@]}"; do
    MODEL_PATH="${MODELS[$MODEL_NAME]}"
    OUTPUT_PATH="$OUTPUT_DIR/$MODEL_NAME"

    echo "开始微调模型: $MODEL_NAME"

    python post_training.py \
        --prune_model "$MODEL_PATH" \
        --data_path "$DATA_PATH" \
        --lora_r "$LORA_R" \
        --num_epochs "$NUM_EPOCHS" \
        --learning_rate "$LEARNING_RATE" \
        --batch_size "$BATCH_SIZE" \
        --output_dir "$OUTPUT_PATH"

    echo "$MODEL_NAME 微调完成，保存到 $OUTPUT_PATH"
done
