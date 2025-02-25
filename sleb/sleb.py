import fire
import copy
import time
from tqdm import tqdm
import os

import torch
import torch.nn as nn

from utils.model_utils import get_llm
from utils.onoff_utils.onoff import block_replace, turn_off, turn_on
from utils.data_utils import *
from utils.block_remove import block_remove
from utils.eval_utils import load_and_eval_ppl, eval_zero_shot

from transformers import AutoTokenizer, AutoModelForCausalLM

@torch.no_grad()
def get_loss(model, testenc, bs=1, device=None, nsamples=10):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen
    print(testenc.numel(), model.seqlen)
    # List to store negative log likelihoods
    losses = []
    #print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0,nsamples,bs):

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        loss = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        losses.append(loss)
    
    # Compute sum of negative log_likelihood
    loss_sum = torch.stack(losses).sum()

    return loss_sum.item()


def sleb(
        model_name: str = "vicuna-13b-v1.3",
        num_blocks: int = 40,
        num_remove_blocks: int = 8,
        early_barrier: int = 1,
        latter_barrier: int = 1,
        seed: int = 1234,
        nsamples: int = 128,
        result_folder: str = 'sleb_results',
        result_file: str = 'sleb_results.txt',
        dataset: str = 'bookcorpus',
        eval_ppl: bool = True,
        eval_zeroshot: bool = False
):
    alive_list = [i for i in range(num_blocks)]
    removal_list = []

    model = get_llm(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    use_cache = model.config.use_cache
    model.config.use_cache = False
    print(f"Loaded Model: {model.name}")
    
    # replace
    model = block_replace(model)
    model.eval()

    dataloader = get_trainloaders(dataset,
                                  nsamples=nsamples,
                                 seed=seed,
                                 model=model_name,
                                 )
    print(f"Dataloader({dataset}) loaded.")

    # check start time
    start_point = time.time()
    for i in range(num_remove_blocks):

        phase_start_point = time.time()
        print(f"Phase {i+1} of {num_remove_blocks}")

        min_loss = 1e99
        min_loss_idx = -1

        search_bound = num_blocks - i

        for j in range(early_barrier, search_bound-latter_barrier):

            # kill j-th alive block
            turn_off(model, alive_list[j])

            loss = get_loss(model, dataloader, bs=1, device=torch.device("cuda:0"), nsamples=nsamples)
            torch.cuda.empty_cache()
            
            if loss < min_loss:
                min_loss = loss
                min_loss_idx = j

            print(
                f"[Block {j} (Original block {alive_list[j]}) removed] Loss={loss:.3f}, Current Min Loss={min_loss:.3f} / Layer {alive_list[min_loss_idx]}"
            )
            # unkill j-th alive block
            turn_on(model, alive_list[j])

        
        phase_finish_point = time.time()
        phase_time_elapsed = phase_finish_point -  phase_start_point

        # remove block causing the least snlls increase
        print(f"Phase_time_elapsed (s): {phase_time_elapsed}")
        print(f"[SELECTED block {min_loss_idx} (Originally block {alive_list[min_loss_idx]})] Loss={min_loss:.3f}")      

        turn_off(model, alive_list[min_loss_idx])
        removal_list.append(alive_list[min_loss_idx])
        print(f"Current Block Removal List: {removal_list}")
        del alive_list[min_loss_idx]
        
    pruned_model = block_remove(model, removal_list)
    pruned_model.config.use_cache = use_cache
    save_model = f'output_model/vicuna_13b_remove{num_remove_blocks}'        
    if not os.path.exists(save_model):
        os.makedirs(save_model)

    torch.save(pruned_model, f'{save_model}/pruned_model.pt')    
    torch.save(pruned_model, f'{save_model}/pruned_model.bin')
    pruned_model.save_pretrained(save_model, safe_serialization=True)
    tokenizer.save_pretrained(save_model)    



if __name__ == "__main__":
    fire.Fire(sleb)