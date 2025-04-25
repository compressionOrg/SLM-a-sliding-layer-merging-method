#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

pruned_models="output/Llama-2-7b-hf-SLM0.72"
save_dir=""Llama-2-7b-hf-SLM0.72""

run_ppl () {
    python slm/eval.py --model_name $1 --output_dir results/$2/ppl 
}

run_zeroshot (){
    python slm/eval_zeroshot_acc.py \
        --model hf-causal-experimental --no_cache \
        --model_args pretrained=$1 \
        --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
        --device cuda --output_json ablation2/$2/zeroshot_acc.json | tee results/$2/zeroshot_acc.txt
}


run_ppl $pruned_models $save_dir
# run_zeroshot $pruned_models $save_dir


