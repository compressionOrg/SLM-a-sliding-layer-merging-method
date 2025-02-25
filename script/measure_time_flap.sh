#!/bin/bash

# Set the CUDA device to use
export CUDA_VISIBLE_DEVICES=0

# Function to run the command with different batch sizes and sequence lengths
run_command() {
    # Define batch sizes and sequence lengths
#     batch_sizes=(1 16 8)
#     max_seq_lens=(128 512)
    batch_sizes=(1)
    max_seq_lens=(128)

    # Loop through each batch size and sequence length combination
    for batch_size in "${batch_sizes[@]}"; do
        for max_seq_len in "${max_seq_lens[@]}"; do
            for script in "gen_batch_eval_time.py" "gen_batch_eval_time_gpuutil.py" "gen_batch_eval_vram.py"; do
                python src/$script --base_model $1 \
                    --flap_model $2 \
                    --output_dir results_efficiency/$3/batch_gen_out${max_seq_len}_bs${batch_size} \
                    --model_type "flap" \
                    --batch_size $batch_size --max_seq_len $max_seq_len $4 
            done
        done
    done
}


run_command "/Llama-2-7b-hf" "FLAP/llm_weights/flap_p0.2_WIFV_ALAM_llama2_7b/pruned_model.pt" "flap_p0.2_WIFV_ALAM_llama2_7b" "--use_bfloat"