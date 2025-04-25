import torch
import torch.nn as nn
import argparse
import os

# Import get_loaders function from data module within the same directory
from data import get_loaders 
from transformers import AutoModelForCausalLM, AutoTokenizer

# Function to evaluate perplexity (ppl) on a specified model and tokenizer
def eval_ppl(model, tokenizer, device=torch.device("cuda:0")):
    """
    Evaluate perplexity (ppl) on a specified model and tokenizer.

    Args:
        model (torch.nn.Module): The language model to be evaluated.
        tokenizer (Tokenizer): Tokenizer instance for encoding texts.
        device (torch.device): Device to move data onto (e.g., 'cuda:0' or 'cpu').

    Returns:
        float: The perplexity of the language model on the test dataset.
    """
    # Set dataset
    dataset = "wikitext2"   # Dataset consisting of extracted sentences from Wikipedia articles

    # Print status
    print(f"evaluating on {dataset}")

    # Get the test loader
    _, testloader = get_loaders(
        dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer 
    )

    # Evaluate perplexity in no grad context to avoid updating the model
    with torch.no_grad():
        # Perplexity measures how well the probability distribution predicted by the model aligns with the actual distribution of the words. Lower perplexity is better.
        ppl = eval_ppl_wikitext(model, testloader, 1, device)
    return ppl 

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext(model, testenc, bs=1, device=None):
    """
    Evaluate perplexity (ppl) specifically on the wikitext dataset.

    Args:
        model (torch.nn.Module): The language model to be evaluated.
        testenc (TokenizerWrapper): Encoded input IDs from test set.
        bs (int): Batch size for evaluation.
        device (torch.device): Device to move data onto (e.g., 'cuda:0' or 'cpu').

    Returns:
        float: The perplexity of the language model on the wikitext test dataset.
    """
    # Get input IDs from the TokenizerWrapper instance
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0, nsamples, bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:, (i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)
        
        # Forward pass through the model
        lm_logits = model(inputs).logits    

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()    # Example: [cat, sat, on, ???] -> [cat, sat, on]
        shift_labels = inputs[:, 1:]    # Example: [The, cat, sat, on] -> [cat, sat, on]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)    # nll = loss * sequence_length * batch_size

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))    # ppl = exp(∑(nlls) / (num_samples * sequence_length))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()

# Note: 
# 1. Perplexity (ppl) is a measure of how well a probability model predicts a sample. 
# 2. Lower perplexity indicates better performance of the model.
# 3. In this script, the perplexity of a language model is evaluated using a specific dataset ('wikitext2').

def parse_args():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(description="评估语言模型的困惑度(PPL)")
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="facebook/opt-125m",
        help="预训练模型名称或路径"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="运行设备，例如 'cuda:0' 或 'cpu'"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="results",
        help="结果输出目录"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1,
        help="评估时的批处理大小"
    )
    return parser.parse_args()

def main():
    """
    主函数：用于评估语言模型的困惑度(PPL)
    """
    # 解析命令行参数
    args = parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 导入模型和分词器
    try:
        print(f"正在加载模型: {args.model_name}")
        # 加载预训练模型和分词器
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        
        # 确保模型有 seqlen 属性
        if not hasattr(model, "seqlen"):
            # 对于大多数模型，可以使用最大位置嵌入长度或配置中的上下文长度
            if hasattr(model.config, "max_position_embeddings"):
                model.seqlen = model.config.max_position_embeddings
            else:
                # 默认值，可以根据模型类型调整
                model.seqlen = 2048
                print(f"警告: 模型没有明确的序列长度，使用默认值: {model.seqlen}")
        
        # 将模型移至指定设备
        model = model.to(device)
        
        # 设置为评估模式
        model.eval()
        
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 计算PPL
        print("开始评估困惑度...")
        ppl = eval_ppl(model, tokenizer, device)
        print(f"模型 {args.model_name} 在测试集上的困惑度(PPL)为: {ppl:.2f}")
        
        # 将结果保存到文件
        result_file = os.path.join(args.output_dir, "ppl_results.txt")
        with open(result_file, "w") as f:
            f.write(f"模型: {args.model_name}\n")
            f.write(f"困惑度(PPL): {ppl:.4f}\n")
        
        print(f"结果已保存到: {result_file}")
        
    except Exception as e:
        print(f"评估过程中出现错误: {str(e)}")

if __name__ == "__main__":
    main()


