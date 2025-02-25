#!/bin/bash

# Set the CUDA device to use
export CUDA_VISIBLE_DEVICES=0


base_model="/nfs/home/9303_xiechuanlong/dx/zhuyao/model/vicuna-13b-v1.3" 
prune_ckpt="/nfs/home/9303_xiechuanlong/dx/zhuyao/LLM-Pruner/prune_log/vicuna-13b-prune0.42/pytorch_model.bin" 
tune_ckpt_name="/nfs/home/9303_xiechuanlong/dx/zhuyao/LLM-Pruner/tune_log/vicuna-13b-prune0.42" 
save_name="vicuna-13b-prune0.42"
epoch=1400

cp $tune_ckpt_name/adapter_config.json $tune_ckpt_name/checkpoint-$epoch/
mv $tune_ckpt_name/checkpoint-$epoch/pytorch_model.bin $tune_ckpt_name/checkpoint-$epoch/adapter_model.bin

#     batch_sizes=(1 16 8)
#     max_seq_lens=(128 512)
batch_sizes=(1)
max_seq_lens=(128)

# Loop through each batch size and sequence length combination
for batch_size in "${batch_sizes[@]}"; do
    for max_seq_len in "${max_seq_lens[@]}"; do
        for script in "gen_batch_eval_time.py" "gen_batch_eval_time_gpuutil.py" "gen_batch_eval_vram.py"; do
        
            python src/$script --base_model $base_model \
                --ckpt $prune_ckpt \
                --output_dir results_efficiency/$save_name/batch_gen_out${max_seq_len}_bs${batch_size} \
                --model_type "tune_pruneLLM" \
                --lora_ckpt $tune_ckpt_name/checkpoint-$epoch \
                --batch_size $batch_size --max_seq_len $max_seq_len $3 
        done
    done
done



# run_command "/nfs/home/9303_xiechuanlong/dx/zhuyao/FLAP/llm_weights/flap_p0.35_WIFV_ALAM_llama2-13b/pruned_model.pt" "flap_p0.35_WIFV_ALAM_llama2-13b"

# run_command "/nfs/home/9303_xiechuanlong/dx/zhuyao/FLAP/llm_weights/flap_p0.2_WIFV_ALAM_vicuna-7b/pruned_model.pt" "flap_p0.2_WIFV_ALAM_vicuna-7b"
# run_command "/nfs/home/9303_xiechuanlong/dx/zhuyao/FLAP/llm_weights/flap_p0.35_WIFV_ALAM_vicuna-7b/pruned_model.pt" "flap_p0.35_WIFV_ALAM_vicuna-7b"
# run_command "/nfs/home/9303_xiechuanlong/dx/zhuyao/FLAP/llm_weights/flap_p0.2_WIFV_ALAM_vicuna-13b/pruned_model.pt" "flap_p0.2_WIFV_ALAM_vicuna-13b"
