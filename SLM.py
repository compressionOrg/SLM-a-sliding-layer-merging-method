from tqdm.notebook import tqdm
from copy import deepcopy
import copy
from pdb import set_trace as st
from datasets import load_dataset, load_from_disk
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
        # traindata = load_dataset(
        #     "allenai/c4",
        #     data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        #     split="train",
        # )
        traindata = load_from_disk("datasets/c4/train")
    elif dataset == "bookcorpus":
        # dataset = load_dataset("bookcorpus/bookcorpus.py", "plain_text")
        # traindata = dataset["train"]
        # traindata = load_dataset('bookcorpus', split='train')
        traindata = load_from_disk("datasets/bookcorpus/train")
        
        # st()
        
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


def cal_sim(hidden_states, model2, example_prompts):
    sim_ls = []   
    count = 0
    for i in range(0, example_prompts.size(0), args.batch_size):
        example_prompts_tmp = example_prompts[i : i + args.batch_size].to("cuda")
        hidden_states1 = hidden_states[count]
        with torch.no_grad():
            outputs2 = model2(example_prompts_tmp, labels=example_prompts_tmp, output_hidden_states=True)       
            hidden_states2 = outputs2.hidden_states[-1]
            hidden_states2 = hidden_states2.to("cpu")
            
        sim_ls.append(torch.cosine_similarity(hidden_states1.squeeze(0).flatten().unsqueeze(0), hidden_states2.squeeze(0).flatten().unsqueeze(0)))
        count += 1
        
    sim_ls = [i.item() for i in sim_ls]
    print('sim_ls:', np.mean(sim_ls))

    return np.mean(sim_ls)  

def save_merged_model(model_copy, save_path, tokenizer):
    print(f"保存合并后的模型到 {save_path}")
    model_copy.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    

#--------------------------------------------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="")
# parser.add_argument("--model_path", type=str, default='Llama-2-13b-hf')
# parser.add_argument("--tokenizer_path", type=str, default='Llama-2-13b-hf')
parser.add_argument("--model_name", type=str, default='meta-llama/Llama-2-7b-hf')
parser.add_argument("--dataset", type=str, default="bookcorpus")
parser.add_argument("--threshold", type=float, default=0.72)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--target_count", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=1)

args = parser.parse_args()

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


#--------------------------------------------------------------------------------------------------------------------------------------------------

model_name = args.model_name
short_model = ShortHFModel(model_name=model_name, layers_path="model.layers")
tokenizer = short_model.tokenizer
    
example_prompts = get_examples(
            dataset=args.dataset,
            tokenizer=tokenizer,
            n_samples=args.target_count,
            seq_len=128,
            field_name="text",
        ).to("cuda")

short_model = short_model.model.to("cuda")
hidden_states1 = []
for i in range(0, example_prompts.size(0), args.batch_size):
    example_prompts_tmp = example_prompts[i : i + args.batch_size].to("cuda")

    with torch.no_grad():
        outputs1 = short_model(example_prompts_tmp, labels=example_prompts_tmp, output_hidden_states=True)
        hidden_states = outputs1.hidden_states[-1]  # (1, seq_len, hidden)
        hidden_states = hidden_states.to("cpu")  
        hidden_states1.append(hidden_states)
        
#-----------------------------------------------------------------------------------------------------------------------------------------------------
    
high_lay = len(short_model.model.layers) - 1 - 1
low_lay = high_lay - 1
THRESHOLD = args.threshold

count = 0

records = []
while low_lay >= 0:
#     print('当前模型层数:', len(model_copy_to_compress.model.layers))
    tmp_merged_model = merge_layers_return_model(short_model, low_lay, high_lay, weight_factor=1)
#     print('合并后的模型层数：', len(tmp_merged_model.model.layers))

    sim_value = cal_sim(hidden_states1, tmp_merged_model, example_prompts)
#     print(f"计算的相似度：{sim_value}, high_lay:{high_lay}, low_lay:{low_lay}")


    if sim_value > THRESHOLD:
        print("相似度合格，继续往下合并")
        low_lay -= 1  
    else:
        print("相似度过低，保存最佳模型")

        if low_lay + 1 != high_lay:
            count += 1
            short_model = merge_layers_return_model(short_model, low_lay + 1, high_lay, weight_factor=1)
            
            record = list(range(low_lay + 1, high_lay + 1))
            print(f'{low_lay+1}至{high_lay}层合并为一层, {record}')
            records.append(record)

            high_lay = low_lay
        else:
            high_lay -= 1
            low_lay -= 1
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------

short_model.config.num_hidden_layers = len(short_model.model.layers)
save_merged_model(short_model, f'output/{args.model_name.split("/")[-1]}-SLM{THRESHOLD}', tokenizer)
print(f'SLM finish! Total count: {count}, new model layer length:{len(short_model.model.layers)}, records:{records}')
