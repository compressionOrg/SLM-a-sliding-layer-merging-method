python hf_prune.py --pruning_ratio 0.09 \
      --base_model "Llama-2-7b-hf" \
      --block_wise \
      --block_mlp_layer_start 4 --block_mlp_layer_end 28 \
      --block_attention_layer_start 4 --block_attention_layer_end 28 \
      --pruner_type taylor \
      --test_after_train \
      --device cpu  --eval_device cuda \
      --save_ckpt_log_name vicuna-13b-prune-merge-6-1