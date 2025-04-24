export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

model="meta-llama/Llama-2-7b-hf"

python main.py \
    --model $model \
    --prune_method flap \
    --pruning_ratio 0.2 \
    --remove_heads -1 \
    --metrics WIFV \
    --structure AL-AM \
    --nsamples 10 \
    --save_model "out/flap_0.2_WIFV_ALAM_Llama-2-7b-hf/" \
    --eval \
