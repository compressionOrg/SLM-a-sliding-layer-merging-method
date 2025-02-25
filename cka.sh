#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

run_command () {
    python src/eval_cka.py \
        --model hf-causal-experimental --no_cache \
        --model_args pretrained=$1 \
        --batch_size 16 \
        --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,mnli \
        --device cuda \
        --output_base_path results_cka/$2 \
        --samples_num 32
}

run_command "vicuna-7b-v1.3" "vicuna-7b-v1.3" "--use_bfloat"