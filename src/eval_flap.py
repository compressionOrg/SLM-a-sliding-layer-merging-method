from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Union
from argparse import ArgumentParser
from pathlib import Path
import torch
import gc

from convert_to_hf import load_and_replace_weights
from utils import convert_json2csv_zeroshot
from lm_eval import evaluator, tasks, utils

import argparse
import json
import logging
import os
import numpy as np

from lm_eval import simple_evaluate
from lm_eval.utils_orig import make_table
from lm_eval.models.huggingface_flap import HFLM

    
def free(object):
    del object
    gc.collect()
    torch.cuda.empty_cache()


def get_args():
    parser = ArgumentParser()
    # add arguments for basemodel and pruned model path
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--pruned_model", type=str, default=None)
    parser.add_argument(
        "--tasks", default=None, choices=utils.MultiChoice(tasks.ALL_TASKS)
    )
    # cache
    parser.add_argument(
        "--cache_dir", type=str, help="Path to cache directory", default=None
    )
    parser.add_argument("--provide_description", action="store_true")    
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=str, default=None)
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        help="Maximal batch size to try with --batch_size auto",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Limit the number of examples per task. "
        "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument("--data_sampling", type=float, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--write_out", action="store_true", default=False)
    parser.add_argument("--output_base_path", type=str, default=None)    
    args = parser.parse_args()
    return args


def _evaluate_lm_eval(
    model_name: str,
    model: Union[AutoModelForCausalLM, torch.nn.Module],
    tokenizer: AutoTokenizer,
    task_names: tuple,
    num_fewshot: int
):

    # evaluate the model
    eval_lm = HFLM(
        model_name=model_name, pretrained=model, tokenizer=tokenizer, batch_size=4
    )
    print(
        "========================================================================================================================\n"
        f"Running eval: model=({model_name}) task=({task_names}) nshot=({num_fewshot})"
        "\n========================================================================================================================"
    )
    eval_results = simple_evaluate(
        eval_lm, tasks=task_names, num_fewshot=num_fewshot, batch_size=4
    )
    free(eval_lm)

    return eval_results


def evaluate(
    model_name: str,
    model: Union[AutoModelForCausalLM, torch.nn.Module],
    tokenizer: AutoTokenizer,
    task_names: list,
    num_fewshot: int
):
    # evaluate the model
    print('tasks:',tasks)
    num_fewshot = args.num_fewshot

    print(
        "WARNING: We suggest to use lm_eval's cli for evaluating tasks other than gptq_wikitext."
    )
    results = _evaluate_lm_eval(
        model_name, model, tokenizer, task_names, num_fewshot
    )
    
    return results


if __name__ == "__main__":
    args = get_args()
    if args.tasks is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = utils.pattern_match(args.tasks.split(","), tasks.ALL_TASKS)
    print(f"Selected Tasks: {task_names}")

    # evaluate pruned model
    if args.pruned_model:
#         pruned_model = load_and_replace_weights(
#             args.base_model,
#             args.pruned_model,
#             cache_dir=args.cache_dir,
#             overwrite_config=False,
#         )
        pruned_model = torch.load(args.pruned_model)
        pruned_tokenizer = AutoTokenizer.from_pretrained(
            args.base_model, cache_dir=args.cache_dir
        )
        if not hasattr(pruned_model, "device"):
            setattr(pruned_model, "device", pruned_model.model.device)
#         evaluate("pruned", pruned_model, pruned_tokenizer, Path(args.results), tasks)
        results = evaluate("pruned", pruned_model, pruned_tokenizer, task_names, args.num_fewshot)
        free(pruned_model)
        free(pruned_tokenizer)


    tabular = make_table(results)
    print(tabular)
    # save
    results_path = Path(args.output_path)
    results_path.mkdir(parents=True, exist_ok=True)
    with open(results_path / f"results_{args.num_fewshot}.txt", "w") as f:
        f.write(tabular)
