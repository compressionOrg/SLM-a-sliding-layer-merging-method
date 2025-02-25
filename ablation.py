from tqdm.notebook import tqdm
from copy import deepcopy
import copy

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)
from transformers import default_data_collator, Trainer, TrainingArguments

from short_hf import ShortHFModel
import pandas as pd
import numpy as np
import argparse
import random

from datasets import load_dataset
#--------------------------------------------------------------------------------------------------------------------------------------------------

def get_examples(
    dataset,
    tokenizer,
    n_samples,
    seq_len=128,
    field_name="text",
    add_bos_to_every=False,
    return_raw_dataset=False,
):
    if dataset == "c4":
        traindata = load_dataset(
            "allenai/c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )
    elif dataset == "bookcorpus":
        dataset = load_dataset("bookcorpus/bookcorpus.py", "plain_text")
        traindata = dataset["train"]
    else:
        raise NotImplementedError

    if return_raw_dataset:
        return traindata

    tokenized_samples, history = [], []

    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(
                traindata[i][field_name],
                return_tensors="pt",
                add_special_tokens=not add_bos_to_every,
            )
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        j = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tmp_ids = tokenized_sample.input_ids[:, j : j + seq_len]
        if add_bos_to_every:  # add bos token to every segment (especially for gemma)
            tmp_ids = torch.cat(
                (torch.LongTensor([[tokenizer.bos_token_id]]), tmp_ids[:, :-1]), dim=1
            )
        tokenized_samples.append(tmp_ids)

    return torch.cat(tokenized_samples, dim=0)

#--------------------------------------------------------------------------------------------------------------------------------------------------
    
def merge_layers_return_model(model, low_lay, high_lay, weight_factor):
    
    if low_lay < 0 or high_lay >= len(model.model.layers):
        raise ValueError("层的索引超出了模型的范围")
        
#     print(f"合并的最低层：{low_lay}, 合并的最高层：{high_lay}, 合并的总层数：{high_lay - low_lay + 1}")

    model_copy = deepcopy(model)
    
    for current_layer_idx in range(low_lay, high_lay + 1): 
#         print('正在处理的层索引：', current_layer_idx)

        for projection in ['gate_proj', 'down_proj', 'up_proj']:
            model_copy.model.layers[low_lay].mlp.__getattr__(projection).weight.data.add_(
                (model.model.layers[current_layer_idx].mlp.__getattr__(projection).weight.data - model_copy.model.layers[low_lay].mlp.__getattr__(projection).weight.data) * weight_factor
            )

        for projection in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            model_copy.model.layers[low_lay].self_attn.__getattr__(projection).weight.data.add_(
                (model.model.layers[current_layer_idx].self_attn.__getattr__(projection).weight.data - model_copy.model.layers[low_lay].self_attn.__getattr__(projection).weight.data) * weight_factor
            )            

    for current_layer_idx in range(high_lay, low_lay, -1):
        del(model_copy.model.layers[current_layer_idx])

    for layer_idx, module in enumerate(model_copy.model.layers):
        module.self_attn.layer_idx = layer_idx

    return model_copy


def cal_sim(model1, model2, example_prompts, low_lay, high_lay):
    
    sim_ls, biases, sim_ls2 = [], [], []   
    for i in range(0, example_prompts.size(0), args.batch_size):
        example_prompts_tmp = example_prompts[i : i + args.batch_size]
        
        with torch.no_grad():
            outputs1 = model1.model(example_prompts_tmp, labels=example_prompts_tmp, output_hidden_states=True)
            outputs2 = model2(example_prompts_tmp, labels=example_prompts_tmp, output_hidden_states=True)
            
            hidden_states1 = outputs1.hidden_states[-1]  # (1, seq_len, hidden)
            hidden_states2 = outputs2.hidden_states[-1]
            
        sim_ls.append(torch.cosine_similarity(hidden_states1.squeeze(0).flatten().unsqueeze(0), hidden_states2.squeeze(0).flatten().unsqueeze(0)))
        
    sim_ls = [i.item() for i in sim_ls]
    print('sim_ls:', np.mean(sim_ls))

    return np.mean(sim_ls)    


def save_merged_model(model_copy, save_path):
    print(f"保存合并后的模型到 {save_path}")
    model_copy.save_pretrained(save_path)
    
def average_merge(model, low_lay, high_lay, weight_factor):
    print(f"合并的最低层：{low_lay}, 合并的最高层：{high_lay}")
    merge_layer_num = high_lay - low_lay + 1
    print(f"合并的总层数：{merge_layer_num}")
    
    model_copy = deepcopy(model)
    
    # 初始化权重累加器
    gate_proj_avg = torch.zeros_like(model_copy.model.layers[low_lay].mlp.gate_proj.weight.data)
    down_proj_avg = torch.zeros_like(model_copy.model.layers[low_lay].mlp.down_proj.weight.data)
    up_proj_avg = torch.zeros_like(model_copy.model.layers[low_lay].mlp.up_proj.weight.data)
    q_proj_avg = torch.zeros_like(model_copy.model.layers[low_lay].self_attn.q_proj.weight.data)
    k_proj_avg = torch.zeros_like(model_copy.model.layers[low_lay].self_attn.k_proj.weight.data)
    v_proj_avg = torch.zeros_like(model_copy.model.layers[low_lay].self_attn.v_proj.weight.data)
    o_proj_avg = torch.zeros_like(model_copy.model.layers[low_lay].self_attn.o_proj.weight.data)
    
    
    for diff_lay in range(low_lay, high_lay + 1):
        print(f'正在处理的层索引：{diff_lay}')
        
        weight_factor = 1
        
        gate_proj_avg.add_(model.model.layers[diff_lay].mlp.gate_proj.weight.data * weight_factor)
        down_proj_avg.add_(model.model.layers[diff_lay].mlp.down_proj.weight.data * weight_factor)
        up_proj_avg.add_(model.model.layers[diff_lay].mlp.up_proj.weight.data * weight_factor)
        q_proj_avg.add_(model.model.layers[diff_lay].self_attn.q_proj.weight.data * weight_factor)
        k_proj_avg.add_(model.model.layers[diff_lay].self_attn.k_proj.weight.data * weight_factor)
        v_proj_avg.add_(model.model.layers[diff_lay].self_attn.v_proj.weight.data * weight_factor)
        o_proj_avg.add_(model.model.layers[diff_lay].self_attn.o_proj.weight.data * weight_factor)
        
    model_copy.model.layers[low_lay].mlp.gate_proj.weight.data = gate_proj_avg
    model_copy.model.layers[low_lay].mlp.down_proj.weight.data = down_proj_avg
    model_copy.model.layers[low_lay].mlp.up_proj.weight.data = up_proj_avg
    model_copy.model.layers[low_lay].self_attn.q_proj.weight.data = q_proj_avg
    model_copy.model.layers[low_lay].self_attn.k_proj.weight.data = k_proj_avg
    model_copy.model.layers[low_lay].self_attn.v_proj.weight.data = v_proj_avg
    model_copy.model.layers[low_lay].self_attn.o_proj.weight.data = o_proj_avg
    
    for diff_lay in range(high_lay, low_lay, -1):
        del(model_copy.model.layers[diff_lay])

    for layer_idx, module in enumerate(model_copy.model.layers):
        module.self_attn.layer_idx = layer_idx
    return model_copy

#--------------------------------------------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="")
parser.add_argument("--model_path", type=str, default='Meta-Llama-3-8B')
parser.add_argument("--tokenizer_path", type=str, default='Meta-Llama-3-8B')
parser.add_argument("--model_name", type=str, default='Meta-Llama-3-8B')
parser.add_argument("--dataset", type=str, default="bookcorpus")
parser.add_argument("--threshold", type=float, default=0.51)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--target_count", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--if_bias", type=str, default="False")
parser.add_argument("--type", type=str, default="slm", help="slm/delete/average")


args = parser.parse_args()

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if args.if_bias == "True":
    bias = True
else:
    bias = False
#--------------------------------------------------------------------------------------------------------------------------------------------------

model_name = args.model_path
short_model = ShortHFModel(model_name=model_name, layers_path="model.layers")
tokenizer = short_model.tokenizer
model_copy_to_compress = copy.deepcopy(short_model.model) 
    
example_prompts = get_examples(
            dataset=args.dataset,
            tokenizer=tokenizer,
            n_samples=args.target_count,
            seq_len=128,
            field_name="text",
#             add_bos_to_every=args.add_bos_to_every,
        ).to("cuda")

    
#-----------------------------------------------------------------------------------------------------------------------------------------------------
    
high_lay = len(model_copy_to_compress.model.layers) - 1 - 1
low_lay = high_lay - 1
THRESHOLD = args.threshold

count = 0

if args.type == 'delete':
    weight_factor = 0
else:
    weight_factor = 1


records = []
while low_lay >= 0:
    if args.type == 'average':
        tmp_merged_model = average_merge(model_copy_to_compress, low_lay, high_lay, weight_factor)
    else:
        tmp_merged_model = merge_layers_return_model(model_copy_to_compress, low_lay, high_lay, weight_factor)
        
    sim_value = cal_sim(short_model, tmp_merged_model, example_prompts, low_lay, high_lay)

    if sim_value > THRESHOLD:
        print("相似度合格，继续往下合并")
        low_lay -= 1  
    else:
        print("相似度过低，保存最佳模型")

        if low_lay + 1 != high_lay:
            count += 1
            
            if args.type == 'average':
                model_copy_to_compress = average_merge(model_copy_to_compress, low_lay + 1, high_lay, weight_factor)
                
            else:
                model_copy_to_compress = merge_layers_return_model(model_copy_to_compress, low_lay + 1, high_lay, weight_factor)
            
            record = list(range(low_lay + 1, high_lay + 1))
            print(f'{low_lay+1}至{high_lay}层合并为一层, {record}')
            records.append(record)

#             high_lay = low_lay + 1
            high_lay = low_lay
        else:
            high_lay -= 1
            low_lay -= 1
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------

model_copy_to_compress.config.num_hidden_layers = len(model_copy_to_compress.model.layers)

save_merged_model(model_copy_to_compress, f'output/{args.model_name}-{args.type}{THRESHOLD}')
print(f'Total count: {count}, new model layer length:{len(model_copy_to_compress.model.layers)}, records:{records}')
