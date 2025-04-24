# SLM: a sliding layer merging method

* we proposes a sliding layer merging method that dynamically selects and fuses consecutive layers from top to bottom according to a pre-defined similarity threshold, thereby simplifying the model structure while maintaining its performance.
* we introduces an innovative fusion of width-wise (LLM_Pruner) and depth-wise pruning  (SLM) technique



## Installation
  ```bash
  conda create -n SLM python=3.9 -y
  conda activate SLM
  pip install -r requirement.txt
  conda install pytorch==2.5.1  cudatoolkit=11.3 -c pytorch -c conda-forge
  ```

<details>
<summary>
Note on package versions:
</summary>


- Part of the below repositories is included for evaluation:
  - `src/LLMPruner`: horseee/LLM-Pruner version [213ffa4](https://github.com/horseee/LLM-Pruner/tree/213ffa4d02f92f16d29219a97fd01a8622db1550)
  - `src/lm_eval`: EleutherAI/lm-evaluation-harness version [3326c54](https://github.com/EleutherAI/lm-evaluation-harness/tree/3326c547a733d598b4377e54be96e194861b964c)

</details>


</details>


## Models we used in article:
  | Source<br>Model | Pruning<br>Ratio | 🤗Hugging Face<br>Link 
  |:---:|:---:|:---:|
  | Llama-2-7B | 20%, 35% | [Llama-2-7B](https://huggingface.co/meta-llama/Llama-2-7b-hf) 
  | Llama-2-13B | 20%, 35% | [Llama-2-13B](https://huggingface.co/meta-llama/Llama-2-13b-hf) 
  | Llama-3-8B | 20%, 35% | [Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) 
  | Vicuna-v1.3-7B | 20%, 35% | [Vicuna-v1.3-7B ](https://huggingface.co/lmsys/vicuna-7b-v1.3) 
  | Vicuna-v1.3-13B | 20%, 35% |[Vicuna-v1.3-7B ](https://huggingface.co/lmsys/vicuna-13b-v1.3) 

<details>




## Scripts of pruning method
- To try our pruning method, use:
  ```bash
  python SLM.py
  ```

- To try baseline pruning method, use:
  ```bash
  bash FLAP/flap.sh
  bash LLM-Pruner/pruner.sh
  python sleb/sleb.py
  bash script/shortened-llm.sh
  ```

## Scripts of lora retrain

- To retrain our pruning method, use:
  ```bash
  bash script/lora_slm.sh
  ```
- To retrain Shortened-llm, use:
  ```bash
  bash script/shortened-llm.sh
  ```
- To retrain LLM-Pruner, use:
  ```bash
  bash LLM_Pruner/post_train.sh
  ```

## Scripts of evaluation on seven commonsense reasoning tasks

- To measure accuracy of SLM, SLEB and shortened-llm pruning method on seven commonsense reasoning tasks, use: (EleutherAI/lm-evaluation-harness version [3326c54](https://github.com/EleutherAI/lm-evaluation-harness/tree/3326c547a733d598b4377e54be96e194861b964c))
  ```bash
  bash script/evaluate_slm.sh
  ```

- To measure accuracy of LLM_Pruner, use: 
  ```bash
  bash LLM_Pruner/eval.sh
  ```

- To measure accuracy of FLAP, use: 
  ```bash
  bash script/evaluate_flap.sh
  ```

## Scripts of measuring latency & throughput

- SLM, SLEB and shortened-llm:
  ```bash
  bash script/measure_time.sh
  ```

- LLM_Pruner: 
  ```bash
  bash script/measure_time_pruner.sh
  ```

- FLAP: 
  ```bash
  bash script/measure_time_flap.sh
  ```

## Integration of width-wise pruning(LLM_Pruner) and depth-wise pruning(SLM)

  ```bash
  bash merge_prune.sh
  ```


## License
- The intended use is strictly limited to research and non-commercial projects.

## Acknowledgments
- [LLM-Pruner](https://github.com/horseee/LLM-Pruner) and [Shortened-llm](https://github.com/Nota-NetsPresso/shortened-llm), which utilize [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness), [PEFT](https://github.com/huggingface/peft), and [Alpaca-LoRA](https://github.com/tloen/alpaca-lora). Thanks for the pioneering work on structured pruning of LLMs! 
- [LLaMA](https://github.com/facebookresearch/llama), [Vicuna](https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md). Thanks for the open-source LLMs and data!

