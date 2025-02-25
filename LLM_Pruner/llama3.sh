python llama3.py --pruning_ratio 0.43 \
                 --device cuda --eval_device cuda \
                 --base_model "Meta-Llama-3-8B" \
                 --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 30 \
                 --block_attention_layer_start 4 --block_attention_layer_end 30 \
                 --save_ckpt_log_name llama3-8b-prune0.43 \
                 --pruner_type taylor --taylor param_first \
                 --test_after_train --save_model 
                