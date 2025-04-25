"""
Code modified from
https://github.com/IST-DASLab/gptq/blob/main/datautils.py
https://github.com/horseee/LLM-Pruner/blob/main/LLMPruner/datasets/example_samples.py
https://github.com/horseee/LLM-Pruner/blob/main/LLMPruner/datasets/ppl_dataset.py
"""

import random

import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data.dataset import Dataset
from datasets import Dataset

class IndexDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, index):
        return self.tensors[index]

    def __len__(self):
        return len(self.tensors)


def get_wikitext2():
#     traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
#     testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    # traindata = load_dataset("parquet", data_files="/nfs/home/9303_xiechuanlong/dx/zhuyao/shortened-llm/data/wikitext/train-00000-of-00001.parquet", split="train")
    # testdata = load_dataset("parquet", data_files="/nfs/home/9303_xiechuanlong/dx/zhuyao/shortened-llm/data/wikitext/test-00000-of-00001.parquet", split="train")
    traindata = load_from_disk("/home/zhangyingying/.cache/datasets/wikitext/train")
    testdata = load_from_disk("/home/zhangyingying/.cache/datasets/wikitext/test")
    print(traindata)
    return traindata, testdata


def get_ptb():
#     traindata = load_dataset("ptb_text_only", "penn_treebank", split="train")
#     valdata = load_dataset("ptb_text_only", "penn_treebank", split="validation")
#     traindata = load_dataset("parquet", data_files="/nfs/home/9303_xiechuanlong/dx/zhuyao/shortened-llm/data/ptb/train-00000-of-00001.parquet", split="train")
#     valdata = load_dataset("parquet", data_files="/nfs/home/9303_xiechuanlong/dx/zhuyao/shortened-llm/data/ptb/valid-00000-of-00001.parquet", split="train")


    train_path = "/nfs/home/9303_xiechuanlong/dx/zhuyao/shortened-llm/data/ptb/ptb.train.txt",
    with open(train_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    def process_line(line):
        words_tags = line.strip().split()  
        words = [wt.split('/')[0] for wt in words_tags]  
        tags = [wt.split('/')[1] for wt in words_tags]  
        return {"words": words, "tags": tags}

    train_dict = {"sentence": [process_line(line) for line in lines]}
    traindata = Dataset.from_dict(train_dict)

    valid_path = "/nfs/home/9303_xiechuanlong/dx/zhuyao/shortened-llm/data/ptb/ptb.valid.txt",
    with open(valid_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    def process_line(line):
        words_tags = line.strip().split()  
        words = [wt.split('/')[0] for wt in words_tags]  
        tags = [wt.split('/')[1] for wt in words_tags]  
        return {"words": words, "tags": tags}

    valid_dict = {"sentence": [process_line(line) for line in lines]}
    valdata = Dataset.from_dict(valid_dict)
    
    print(traindata)
    print(valdata)
    return traindata, valdata


def process_data(samples, tokenizer, seq_len, field_name, add_bos_to_every=False):
    test_ids = tokenizer(
        "\n\n".join(samples[field_name]),
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[0]
    if not add_bos_to_every:  # add bos token to only the first segment
        test_ids = torch.cat(
            (torch.LongTensor([tokenizer.bos_token_id]), test_ids), dim=0
        )

    test_ids_batch = []
    nsamples = test_ids.numel() // seq_len

    for i in range(nsamples):
        batch = test_ids[(i * seq_len) : ((i + 1) * seq_len)]
        if add_bos_to_every:  # add bos token to every segment (especially for gemma)
            batch = torch.cat(
                (torch.LongTensor([tokenizer.bos_token_id]), batch), dim=0
            )
        test_ids_batch.append(batch)
    test_ids_batch = torch.stack(test_ids_batch)

    return IndexDataset(tensors=test_ids_batch)


def get_loaders(name, tokenizer, seq_len=2048, batch_size=1, add_bos_to_every=False):
    if "wikitext2" in name:
        train_data, test_data = get_wikitext2()
        test_dataset = process_data(
            test_data, tokenizer, seq_len, "text", add_bos_to_every
        )
    if "ptb" in name:
        train_data, test_data = get_ptb()
        test_dataset = process_data(
            test_data, tokenizer, seq_len, "sentence", add_bos_to_every
        )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,  drop_last=True
    )
    print(test_loader)
    return train_data, test_loader


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
        traindata = load_from_disk("/home/zhangyingying/.cache/datasets/c4/train")
    elif dataset == "bookcorpus":
        traindata = load_dataset("bookcorpus", split="train")
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
