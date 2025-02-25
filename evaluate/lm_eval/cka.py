import collections
import itertools
import random

import lm_eval.metrics
import lm_eval.models
import lm_eval.tasks
import lm_eval.base
from lm_eval.utils import positional_deprecated, run_task_tests
from lm_eval.models.gpt2 import HFLM

import numpy as np
import transformers
from pathlib import Path
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch 
from einops import rearrange, repeat
from loguru import logger
from torchmetrics import Metric, BootStrapper
import matplotlib.pyplot as plt
import timm
from collections import defaultdict
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap


def gram(X):
    # ensure correct input shape
    X = rearrange(X, 'b ... -> b (...)')
    return X @ X.T

def centering_mat(n, device):
    v_i = torch.ones(n,1, device=device)
    H = torch.eye(n, device=device) - (v_i @ v_i.T) / n
    return H

def centered_gram(X, device):
    K = gram(X)
    m = K.shape[0]
    H = centering_mat(m, device)
    #logger.info(H.shape)
    #logger.info(K.shape)
    return H @ K @ H

def unbiased_hsic_xy(X,Y,device):
    n = X.shape[0]
#     print('shape of X:', X.shape)
    assert n > 3 
    v_i = torch.ones(n,1, device=device)
    K = centered_gram(X, device)
    L = centered_gram(Y, device)
    KL = K @ L
    iK = v_i.T @ K
    Li = L @ v_i
    iKi = iK @ v_i
    iLi = v_i.T @ Li

    a = torch.trace(KL)
    b = iKi * iLi / ((n-1)*(n-2))
    c = iK @ Li * 2 / (n-2)

    outv = (a + b - c) / (n*(n-3))
    return outv.long().item()

class MinibatchCKA(Metric):
    def __init__(self, dist_sync_on_step=False):
        """
        Introduced in: https://arxiv.org/pdf/2010.15327.pdf
        Implemented to reproduce the results in: https://arxiv.org/pdf/2108.08810v1.pdf
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("_xx", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("_xy", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("_yy", default=torch.tensor(0), dist_reduce_fx="sum")
    def update(self, X: torch.Tensor, Y: torch.Tensor, device):
        # NB: torchmetrics Bootstrap resampling janks up batch shape by varying number of samples per batch
        self._xx += unbiased_hsic_xy(X,X,device)
        self._xy += unbiased_hsic_xy(X,Y,device)
        self._yy += unbiased_hsic_xy(Y,Y,device)
    def compute(self):
        xx, xy, yy = self._xx, self._xy, self._yy
        
        return xy / (torch.sqrt(xx) * torch.sqrt(yy))


class HookedCache:
    def __init__(self, model, target):
        self.model = model
        self.target = target
        
        self.clear()
        self._extract_target()
        self._register_hook()

    @property
    def value(self):
        return self._cache
    def clear(self):
        self._cache = None
    def _extract_target(self):
        for name, module in self.model.model.named_modules():
#             if name == self.target:
            if self.target in name:
#                 print('target:', self.target, 'module:', module)
                self._target = module
                return
    def _register_hook(self):
        def _hook(module, in_val, out_val):
             self._cache = out_val
        self._target.register_forward_hook(_hook)


def get_simmat_from_metrics(metrics):
    vals = []
    for i, ckas in enumerate(metrics):
        for j, cka in enumerate(ckas):
            z = cka.compute().item()
            vals.append((i,j,z))

    sim_mat = torch.zeros(i+1,j+1)
    for i,j,z in vals:
        sim_mat[i,j] = z
    
    return sim_mat

def make_pairwise_metrics(mod1_hooks, mod2_hooks, device):
    metrics = []
    for i_ in mod1_hooks:
        metrics.append([])
        for j_ in mod2_hooks:
            metrics[-1].append(MinibatchCKA().to(device))
    return metrics

def update_metrics(mod1_hooks, mod2_hooks, metrics, metric_name, do_log, device):
    for i, hook1 in enumerate(mod1_hooks):
        for j, hook2 in enumerate(mod2_hooks):
            cka = metrics[i][j]
            X,Y = hook1.value, hook2.value
            
            if isinstance(X, tuple):
                X = X[0].float() 
            if isinstance(Y, tuple):
                Y = Y[0].float() 
            
            cka.update(X,Y,device)
            if do_log and 0 in (i,j):
                _metric_name = f"{metric_name}_{i}-{j}"
                v = cka.compute()
#                 writer.add_scalar(_metric_name, v, it)
    if do_log:
        sim_mat = get_simmat_from_metrics(metrics)
        sim_mat = sim_mat.unsqueeze(0) * 255
#         writer.add_image(metric_name, sim_mat, it)



@positional_deprecated
def get_cka(
    model,
    model_args=None,
    tasks=[],
    num_fewshot=0,
    batch_size=None,
    max_batch_size=None,
    device=None,
    no_cache=False,
    limit=None,
    bootstrap_iters=100000,
    description_dict=None,
    check_integrity=False,
    decontamination_ngrams_path=None,
    write_out=False,
    output_base_path=None,
    samples_num=None,
):
    """Instantiate and evaluate a model on a list of tasks.

    :param model: Union[str, LM]
        Name of model, transformers.PreTrainedModel object, or LM object, see lm_eval.models.get_model
    :param model_args: Optional[str]
        String arguments for each model class, see LM.create_from_arg_string.
        Ignored if `model` argument is a LM object.
    :param tasks: list[Union[str, Task]]
        List of task names or Task objects. Task objects will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param num_fewshot: int
        Number of examples in few-shot context
    :param batch_size: int or str, optional
        Batch size for model
    :param max_batch_size: int, optional
        Maximal batch size to try with automatic batch size detection
    :param device: str, optional
        PyTorch device (e.g. "cpu" or "cuda:0") for running models
    :param no_cache: bool
        Whether or not to cache
    :param limit: int or float, optional
        Limit the number of examples per task (only use this for testing), If <1, limit is a percentage of the total number of examples.
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param description_dict: dict[str, str]
        Dictionary of custom task descriptions of the form: `task_name: description`
    :param check_integrity: bool
        Whether to run the relevant part of the test suite for the tasks
    :param write_out: bool
        If True, write details about prompts and logits to json for all tasks
    :param output_base_path: str, optional
        Directory to which detailed eval info will be written. Defaults to present working dir.
    :return
        Dictionary of results
    """
    random.seed(1234)
    np.random.seed(1234)

    assert tasks != [], "No tasks specified"

    if isinstance(model, str):
        if model_args is None:
            model_args = ""
        lm = lm_eval.models.get_model(model).create_from_arg_string(
            model_args, {"batch_size": batch_size, "max_batch_size": max_batch_size, "device": device}
        )
    elif isinstance(model, transformers.PreTrainedModel):
        lm = lm_eval.models.get_model("hf-causal")(
                pretrained=model,
                batch_size=batch_size,
                max_batch_size=max_batch_size,
                )
        no_cache = True
    else:
        assert isinstance(model, lm_eval.base.LM)
        lm = model

    if not no_cache:
        lm = lm_eval.base.CachingLM(
            lm,
            "lm_cache/"
            + (model if isinstance(model, str) else model.model.config._name_or_path)
            + "_"
            + model_args.replace("=", "-").replace(",", "_").replace("/", "-")
            + ".db",
        )
    
    task_dict = lm_eval.tasks.get_task_dict(tasks)

    if check_integrity:
        run_task_tests(task_list=tasks)

    results = get_cka_image(
        lm=lm,
        task_dict=task_dict,
        num_fewshot=num_fewshot,
        limit=limit,
        bootstrap_iters=bootstrap_iters,
        description_dict=description_dict,
        decontamination_ngrams_path=decontamination_ngrams_path,
        write_out=write_out,
        output_base_path=output_base_path,
        batch_size=batch_size,
        device=device,
        samples_num=samples_num,
    )
    
    return results


decontaminate_suffix = "_decontaminate"


@positional_deprecated
def get_cka_image(
    lm,
    task_dict,
    provide_description=None,
    num_fewshot=0,
    limit=None,
    bootstrap_iters=100000,
    description_dict=None,
    decontamination_ngrams_path=None,
    write_out=False,
    output_base_path=None,
    batch_size=None,
    device=None,
    samples_num=None,
):
    """Instantiate and evaluate a model on a list of tasks.

    :param lm: obj
        Language Model
    :param task_dict: dict[str, Task]
        Dictionary of tasks. Tasks will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param provide_description: bool
        Not implemented, and this option is deprecated and will be removed in a future version in favor of a different description providing method
    :param num_fewshot: int
        Number of examples in few-shot context
    :param limit: int, optional
        Limit the number of examples per task (only use this for testing)
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param description_dict: dict[str, str]
        Dictionary of custom task descriptions of the form: `task_name: description`
    :param write_out: bool
        If True, write all prompts, logits and metrics to json for offline analysis
    :param output_base_path: str, optional
        Directory to which detailed eval info will be written. Defaults to present working dir
    :return
        Dictionary of results
    """
    # TODO: completely refactor this entire function to not be a huge mess, ideally breaking it down into smaller pieces

    # TODO: todo: implement proper description-providing system
    assert not provide_description  # not implemented.
    if provide_description is not None:
        # nudge people to not specify it at all
        print(
            "WARNING: provide_description is deprecated and will be removed in a future version in favor of description_dict"
        )

    decontaminate = decontamination_ngrams_path is not None

    task_dict_items = [
        (name, task)
        for name, task in task_dict.items()
        if (task.has_validation_docs() or task.has_test_docs())
    ]
    print('task+dict_items:', task_dict_items)
    
    results = collections.defaultdict(dict)
    versions = collections.defaultdict(dict)

    requests = collections.defaultdict(list)
    requests_origin = collections.defaultdict(list)

    overlaps = collections.defaultdict(list)  # {task_name: contaminated_docs}

    # If we ever run into issues where the eval tasks don't fit in memory and we can't afford a machine with bigger
    # memory, we can always modify this plumbing to support that, but I didn't want to include it just yet because
    # over-engineering is bad (or we could make it write the requests to disk and then read them back out again
    #  - probably using an sqlite db because of all the moving parts we have

    # TODO: we need unit tests & sanity checks or something to ensure that the return of `validation_docs` is stable
    docs = {}
    write_out_info = {}

    docs_for_decontamination = collections.defaultdict(list)
    
    sim_mat_dict = {}

    # get lists of each type of request
    for task_name, task in task_dict_items:
        versions[task_name] = task.VERSION
        # default to test doc, fall back to val doc if validation unavailable
        # TODO: the test-fallback-to-val system isn't final, we should revisit it at some point
        if task.has_test_docs():
            task_doc_func = task.test_docs
            task_set = "test"  # Required for caching in the decontamination
        elif task.has_validation_docs():
            task_set = "val"  # Required for caching in the decontamination
            task_doc_func = task.validation_docs
        else:
            raise RuntimeError("Task has neither test_docs nor validation_docs")

        # deterministically shuffle docs and chop off the first `limit` because sometimes docs are in some kind of order
        task_docs = list(task_doc_func())
        rnd = random.Random()
        rnd.seed(42)
        rnd.shuffle(task_docs)
        print(f"Task: {task_name}; number of docs: {len(task_docs)}")

        if write_out:
            prompt_details = []
        

        description = (
            description_dict[task_name]
            if description_dict and task_name in description_dict
            else ""
        )
        if limit is not None:
            limit = int(len(task_docs) * limit) if limit < 1.0 else int(limit)

        for doc_id, doc in enumerate(itertools.islice(task_docs, 0, limit)):
            if decontaminate and task.should_decontaminate():
                docs_for_decontamination[(task_name, task_set)].append(
                    task.doc_to_decontamination_query(doc)
                )

            docs[(task_name, doc_id)] = doc
            ctx = task.fewshot_context(
                doc=doc, num_fewshot=num_fewshot, rnd=rnd, description=description
            )
            reqs = task.construct_requests(doc, ctx)

            if write_out:
                prompt_details.append({"doc_id": doc_id})

            # print the prompt for the first few documents
            if doc_id < 1:
                print(
                    f"Task: {task_name}; document {doc_id}; context prompt (starting on next line):\n{ctx}\n(end of prompt on previous line)"
                )
                print("Requests:", reqs)

            if not isinstance(reqs, (list, tuple)):
                reqs = [reqs]
            for i, req in enumerate(reqs):
                requests[req.request_type].append(req)
                # i: index in requests for a single task instance
                # doc_id: unique id that we can get back to a doc using `docs`
                requests_origin[req.request_type].append((i, task_name, doc, doc_id))

                if write_out:
                    prompt_details[-1][f"prompt_{i}"] = "".join(
                        (map(lambda x: "".join(x), req.args))
                    )

        if write_out:
            write_out_info[task_name] = prompt_details
        
        
    # Compare all tasks/sets at once to ensure a single training set scan
    if decontaminate:
        from lm_eval.decontamination.decontaminate import get_train_overlap

        print("Finding train/test overlap, please wait...")
        overlaps = get_train_overlap(
            docs_for_decontamination, decontamination_ngrams_path, limit
        )

    # all responses for each (task, doc)
    process_res_queue = collections.defaultdict(list)

    # execute each type of request
    for reqtype, reqs in requests.items():
        # TODO: right now, this code runs multiple separate LM requests for multiple Requests differing
        #       only in index. We could implement some kind of caching, but that would be more of a band-aid
        #       solution. we could also implement some kind of auto-grouping here;
        #       they should end up next to each other.

        print("Running", reqtype, "requests")
        print('len of reqs:', len(reqs))

        task_names = set(task_name for _, task_name, _, _ in requests_origin[reqtype])
        print(task_names)
        
        grouped_reqs = defaultdict(list)
        resps = []
        for req, (i, task_name, doc, doc_id) in zip(reqs, requests_origin[reqtype]):
            grouped_reqs[task_name].append(req)


        # 对每个 task_name 对应的请求进行循环
        for task_name, task_reqs in grouped_reqs.items():
            print(f"Processing task: {task_name}")
            
#             resp = getattr(lm, reqtype)([req.args for req in task_reqs])

            # hook
            modc_hooks = []
            print('*' * 100)
#             print(lm.model)
            for i, stage in enumerate(lm.model.model.layers):
                tgt = f'model.layers.{i}'
                hook = HookedCache(lm, tgt)
                modc_hooks.append(hook)

            metrics_cc = make_pairwise_metrics(modc_hooks, modc_hooks, device)

            def chunk_task_reqs(task_reqs, batch_size):
                # 按照给定的 batch_size 将 task_reqs 拆分
                for i in range(0, len(task_reqs), batch_size):
                    yield task_reqs[i:i + batch_size]


            for batch in chunk_task_reqs(task_reqs[:samples_num], int(batch_size)):
                batch_args = [req.args for req in batch]
                # 调用对应方法
                resp = getattr(lm, reqtype)(batch_args)
                resps.extend(resp)
                update_metrics(modc_hooks, modc_hooks, metrics_cc, "cka/cc", do_log = False, device=device)   
                     
                for hook0 in modc_hooks:
                    hook0.clear() 
                    
            sim_mat_dict[task_name] = get_simmat_from_metrics(metrics_cc)
            sim_mat = get_simmat_from_metrics(metrics_cc)
            plt.imshow(sim_mat)

            # 显示相似度矩阵
            os.makedirs(f'{output_base_path}/samples_{samples_num}_batch_{batch_size}', exist_ok=True)
#             plt.colorbar()
            plt.title('c-c')
            plt.savefig(f'{output_base_path}/samples_{samples_num}_batch_{batch_size}/{task_name}.png')   
            plt.close()
            
    return sim_mat_dict

