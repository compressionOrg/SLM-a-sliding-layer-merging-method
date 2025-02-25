#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

run_command () {
    python src/eval_ppl.py --base_model $1 --output_dir results/$2/ppl $3

    python src/eval_zeroshot_acc.py \
        --model hf-causal-experimental --no_cache \
        --model_args pretrained=$1 \
        --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
        --device cuda --output_json ablation2/$2/zeroshot_acc.json | tee results/$2/zeroshot_acc.txt
}


run_command "llama2-7b-slm0.95" "llama2-7b-slm0.95"



