python LLM_Pruner/llama3.py --pruning_ratio 0.2 \
                 --device cuda --eval_device cuda \
                 --base_model "llama-3-laco0.72" \ # slm model to be pruned continually
                 --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 26 \
                 --block_attention_layer_start 4 --block_attention_layer_end 26 \
                 --save_ckpt_log_name llama3_prune \
                 --pruner_type taylor --taylor param_first \
                 --test_after_train --save_model 