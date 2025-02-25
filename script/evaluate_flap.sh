#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

run_command () {
    python src/eval_ppl.py --base_model $1 --flap_model $2 --model_type "flap" --output_dir results/$3/ppl $4

    python src/eval_flap.py \
        --base_model $1 \
        --pruned_model $2 \
        --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
        --output_path results/$3
}

run_command "Meta-Llama-3-8B" "FLAP/llm_weights/flap_llama3_p0.2/pruned_model.pt" "flap_llama3_p0.2"
