#!/bin/bash
export PYTHONPATH='.'

# 定义函数来执行微调和评估
run_evaluation() {
    local base_model=$1
    local prune_ckpt=$2
    local tune_ckpt_name=$3
    local save_name=$4
    local epochs=("${!5}") # 使用间接引用获取epochs数组

    for epoch in "${epochs[@]}"; do
        checkpoint_dir="${tune_ckpt_name}/checkpoint-$epoch"
        mkdir -p "$checkpoint_dir" # 确保检查点目录存在

        # 复制配置文件
        cp "$tune_ckpt_name/adapter_config.json" "$checkpoint_dir/" 2>/dev/null || { echo "Error copying config file for epoch $epoch"; continue; }

        # 重命名模型文件（如果存在）
        if [ -f "$checkpoint_dir/pytorch_model.bin" ]; then
            mv "$checkpoint_dir/pytorch_model.bin" "$checkpoint_dir/adapter_model.bin" 2>/dev/null || { echo "Error renaming model file for epoch $epoch"; continue; }
        fi

        # 执行微调和评估
        python lm-evaluation-harness/main.py --model hf-causal-experimental \
            --model_args "checkpoint=$prune_ckpt/pytorch_model.bin,peft=$tune_ckpt_name/checkpoint-$epoch,config_pretrained=$base_model" \
            --tasks "openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq" \
            --device cuda:0 \
            --output_path "results/${save_name}_$epoch.json" \
            --no_cache
    done
}

# 设置模型参数
models=(
#     "Llama-2-13b-hf 0.24 prune_log/llama2-13b-prune0.24 tune_log/llama2-13b-prune0.24 llama2-13b-prune0.24"
)
epochs=(1400) # 定义轮次数组

# 遍历模型参数并执行评估
for model_params in "${models[@]}"; do
    # 解析模型参数（这里假设参数之间用空格分隔，并且最后一个参数是save_name）
    IFS=' ' read -r -a params <<< "$model_params"
    base_model_name="${params[0]}"
    prune_ratio="${params[1]}"
    prune_ckpt_path="${params[2]}"
    tune_ckpt_path="${params[3]}"
    save_name="${params[-1]}"

    # 设置base_model变量（这里假设所有模型都位于同一个基路径下）
    base_model="model/${base_model_name}"

    # 调用函数执行评估
    run_evaluation "$base_model" "$prune_ckpt_path" "$tune_ckpt_path" "$save_name" epochs
done