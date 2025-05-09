# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
from datasets import load_dataset, load_from_disk

# Set random seed for reproducibility
def set_seed(seed):
    """
    Set the random seed for NumPy and PyTorch for reproducibility.
    
    Args:
        seed (int): The random seed.
    """
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper class for tokenized input IDs
class TokenizerWrapper:
    """
    Wrapper class for tokenized input IDs.

    Args:
        input_ids (tensor): The tokenized input IDs from the tokenizer.
    """
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process PTB (Penn Treebank) dataset
def get_ptb(nsamples, seed, seqlen, tokenizer):
    """
    Load and process PTB (Penn Treebank) dataset.
    
    Args:
        nsamples (int): Number of samples to extract.
        seed (int): Random seed for reproducibility.
        seqlen (int): Sequence length for each sample.
        tokenizer (Tokenizer): Tokenizer to use for text encoding.

    Returns:
        tuple: A tuple containing trainloader (list of input and target pairs) and encoded test set.
    """
    # Load train and test datasets
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')
    
    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set using random seed and specified sequence length
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    """
    Load and process the Wikitext-2 dataset.

    Args:
        nsamples (int): Number of samples to generate from the training set.
        seed (int): Random seed for reproducibility.
        seqlen (int): Sequence length for generated samples.
        tokenizer (Tokenizer): Tokenizer instance for encoding texts.

    Returns:
        tuple: A tuple containing trainloader (list of input and target pairs) and encoded test dataset.
    """
    # Load train and test datasets
    # traindata = load_dataset('/nfs/home/9303_xiechuanlong/dx/zhuyao/shortened-llm/data/wikitext', split='train')
    # testdata = load_dataset('/nfs/home/9303_xiechuanlong/dx/zhuyao/shortened-llm/data/wikitext', split='test')
    # traindata = load_dataset('text', data_files='datasets/wikitext/wiki.train.raw', split="train")
    # testdata = load_dataset('text', data_files='datasets/wikitext/wiki.test.raw', split="train")
    traindata = load_from_disk("/home/zhangyingying/.cache/datasets/wikitext/train")
    testdata = load_from_disk("/home/zhangyingying/.cache/datasets/wikitext/test")

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set using random seed and specified sequence length
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

# Load and process C4 (Common Crawl) dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    """
    Load and process the C4 (Common Crawl) dataset.

    Args:
        nsamples (int): Number of samples to generate from the training set.
        seed (int): Random seed for reproducibility.
        seqlen (int): Sequence length for generated samples.
        tokenizer (Tokenizer): Tokenizer instance for encoding texts.

    Returns:
        tuple: A tuple containing trainloader (list of input and target pairs) and encoded validation dataset.
    """
    # Load train and validation datasets
    # traindata = load_dataset('allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    # valdata = load_dataset('allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
    # traindata = load_dataset('json', data_files={'train': 'datasets/c4/c4-train.00000-of-01024.json.gz'}, split='train')
    # valdata = load_dataset('json', data_files={'validation': 'datasets/c4/c4-validation.00000-of-00008.json.gz'}, split='validation')
    traindata = load_from_disk("/home/zhangyingying/.cache/datasets/c4/train")
    valdata = load_from_disk("/home/zhangyingying/.cache/datasets/c4/validation")
    # Generate samples from training set using random seed and specified sequence length
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc




def get_bookcorpus(nsamples, seed, seqlen, tokenizer):
    """
    Load and process the C4 (Common Crawl) dataset.

    Args:
        nsamples (int): Number of samples to generate from the training set.
        seed (int): Random seed for reproducibility.
        seqlen (int): Sequence length for generated samples.
        tokenizer (Tokenizer): Tokenizer instance for encoding texts.

    Returns:
        tuple: A tuple containing trainloader (list of input and target pairs) and encoded validation dataset.
    """
    # Load train and validation datasets
    # dataset = load_dataset("/nfs/home/9303_xiechuanlong/dx/zhuyao/shortened-llm/data/bookcorpus/bookcorpus.py", "plain_text")
    # traindata = dataset["train"]
    traindata = load_from_disk("/home/zhangyingying/.cache/datasets/bookcorpus/train")
    # traindata = load_dataset('json', data_files={'train': 'datasets/c4/c4-train.00000-of-01024.json.gz'}, split='train')
    # valdata = load_dataset('json', data_files={'validation': 'datasets/c4/c4-validation.00000-of-00008.json.gz'}, split='validation')
    
    # Generate samples from training set using random seed and specified sequence length
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    valenc = ''
    return trainloader, valenc



# Function to select the appropriate loader based on dataset name
def get_loaders(name='wikitext2', nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    """
    Select the appropriate loader based on dataset name.

    Args:
        name (str): The name of the dataset ('wikitext2', 'c4', or 'ptb').
        nsamples (int): Number of samples to generate from the training set.
        seed (int): Random seed for reproducibility.
        seqlen (int): Sequence length for generated samples.
        tokenizer (Tokenizer): Tokenizer instance for encoding texts.

    Returns:
        tuple: A tuple containing trainloader (list of input and target pairs) and encoded validation/test set.
    """
    # Determine which dataset to use based on 'name' parameter and return corresponding loader
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    elif "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
    elif "ptb" in name:
        return get_ptb(nsamples, seed, seqlen, tokenizer)
    elif 'bookcorpus' in name:
        return get_bookcorpus(nsamples, seed, seqlen, tokenizer)



    
if __name__ == "__main__": 
    get_loaders('bookcorpus', seed=0, seqlen=2048, tokenizer=None)

# Note:
# This script is designed to load and process various text datasets for training language models.
# It includes functions to load PTB (Penn Treebank), Wikitext-2, and C4 (Common Crawl) datasets.
# Each loading function returns a trainloader (list of input and target pairs) and encoded validation/test set.
