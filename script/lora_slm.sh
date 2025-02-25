#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export BASE_MODEL="/llama2-7b-slm0.68"
export MODEL_NAME=llama2-7b-slm0.68
export NUM_CALIB_DATA=10
export NUM_PRUNED_BLOCKS=11
export OUTPUT_SENSITIVITY=output_block_sensitivity/$MODEL_NAME/ppl_n${NUM_CALIB_DATA}
export OUTPUT_PRUNE=output_prune/$MODEL_NAME/ppl_n${NUM_CALIB_DATA}/rm_${NUM_PRUNED_BLOCKS}_blocks
export OUTPUT_TUNE=output_tune/$MODEL_NAME/ppl_n${NUM_CALIB_DATA}/rm_${NUM_PRUNED_BLOCKS}_blocks


# Perform LoRA-based retraining
python src/lora_retrain.py \
    --base_model $BASE_MODEL \
    --data_path "/nfs/home/9303_xiechuanlong/dx/zhuyao/shortened-llm/data/alpaca-cleaned" \
    --output_dir $OUTPUT_TUNE \
    --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64 --micro_batch_size 4 \
    --save_lora_merge --use_bfloat
    
# Compute Zero-shot PPL on WikiText2 and PTB 
python src/eval_ppl.py \
    --base_model $BASE_MODEL \
    --output_dir ${OUTPUT_TUNE}_score_nolora --fix_decapoda_config

# Compute Zero-shot accuracy on seven commonsense reasoning tasks 
python src/eval_zeroshot_acc.py \
    --model hf-causal-experimental --no_cache \
    --model_args pretrained=$BASE_MODEL \
    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
    --device cuda \
    --output_json ${OUTPUT_TUNE}_score_nolora/zeroshot_acc.json \
    | tee ${OUTPUT_TUNE}_score_nolora/zeroshot_acc.txt
    