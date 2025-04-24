#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/home/zhangyingying/cyl/prune/SLM-a-sliding-layer-merging-method
export CUDA_VISIBLE_DEVICES=0

model="meta-llama/Llama-2-7b-hf"
model_name=$(echo "$model" | awk -F'/' '{print $2}')
  python SLM.py \
  --model_name  ${model} 
  
# > logs/${model_name}_slm.log 
