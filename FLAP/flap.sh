python main.py \
    --model "/nfs/home/9303_xiechuanlong/dx/zhuyao/modelMeta-Llama-3-8B" \
    --prune_method wanda_sp \
    --pruning_ratio 0.2 \
    --remove_heads -1 \
    --metrics WIFV \
    --structure AL-AM \
    --nsamples 10 \
    --save_model "llm_weights/wanda_sp_p0.2_WIFV_ALAM_llama3/" \
    --eval \
