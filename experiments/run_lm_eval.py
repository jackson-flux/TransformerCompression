# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import json
import logging
import os

import lm_eval
import torch
import wandb
from lm_eval import tasks
from lm_eval import utils as lm_eval_utils
from lm_eval.api.registry import ALL_TASKS
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import initialize_tasks
from transformers import AutoModelForCausalLM, AutoTokenizer

from quarot.model_phi35 import Phi3ForCausalLM
from slicegpt import gpu_utils, hf_utils, utils
from slicegpt.config import config

TASK_METRIC_MAP = {
    "mmlu_abstract_algebra": "acc,none",
    "mmlu_business_ethics": "acc,none",
    "mmlu_college_computer_science": "acc,none",
    "mmlu_college_mathematics": "acc,none",
    "mmlu_conceptual_physics": "acc,none",
    "mmlu_formal_logic": "acc,none",
    "mmlu_machine_learning": "acc,none",
    "mmlu_miscellaneous": "acc,none",
    "mmlu_philosophy": "acc,none",
    "mmlu_global_facts": "acc,none",
    "arc_challenge": "acc_norm,none",
    "arc_easy": "acc_norm,none",
    "hellaswag": "acc_norm,none",
    "piqa": "acc_norm,none",
    "winogrande": "acc,none",
}


def eval_arg_parser(interactive: bool = True) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-125m",
        help="Model to load",
    )
    path_group = parser.add_mutually_exclusive_group()
    path_group.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to load the model and tokenizer from (required for local models, not required for HF models)",
    )
    path_group.add_argument(
        "--sliced-model-path",
        type=str,
        help="Path to load the model to fine-tune (sliced) and tokenizer from",
        default=None,
    )
    parser.add_argument(
        "--sparsity", type=float, default=0.0, help="A measure of how much slicing is applied (in the range [0, 1))"
    )
    parser.add_argument(
        "--round-interval",
        type=int,
        default=8,
        help="Interval for rounding the weights (the best value may depend on your hardware)",
    )
    parser.add_argument('--hf-token', type=str, default=os.getenv('HF_TOKEN', None))
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for evaluating with lm eval harness.")
    parser.add_argument(
        "--distribute-model",
        action="store_true",
        help="Use accelerate to put the model on multiple GPUs for evaluation. It is recommended to use it for models with 30B parameters and above.",
    )
    parser.add_argument('--wandb-project', type=str, default="slicegpt-lm-eval", help="wandb project name.")
    parser.add_argument('--no-wandb', action="store_true", help="Disable wandb.")
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande"],
    )
    parser.add_argument('--num-fewshot', type=int, default=0, help="Number of fewshots for all tasks.")
    parser.add_argument("--save-dir", type=str, default=".", help="Path to save the lm eval results")
    return parser.parse_args() if interactive else parser.parse_args('')


def process_eval_args(args: argparse.Namespace):
    logging.info(f'Parsed arguments:')
    for arg, argv in vars(args).items():
        logging.info(f'{arg} = {argv}')


def calculate_avg_accuracy(task_names: str, results: dict) -> float:
    n_tasks = len(task_names)
    acc_cumul = sum(result.get(TASK_METRIC_MAP[task]) for task, result in results.items() if 'mmlu' not in task)

    questions_per_mmlu_task = {
        task_name: lm_eval.tasks.get_task_dict([task_name])[task_name].dataset["test"].num_rows
        for task_name in task_names
        if 'mmlu' in task_name
    }

    if not questions_per_mmlu_task:
        return acc_cumul / n_tasks

    # Calculate average accuracy for mmlu tasks, weighted by number of questions in each task
    acc_mmlu = sum(
        result.get(TASK_METRIC_MAP[task]) * questions_per_mmlu_task[task]
        for task, result in results.items()
        if 'mmlu' in task
    )
    acc_mmlu_avg = acc_mmlu / sum(questions_per_mmlu_task.values())
    wandb.log({'acc_mmlu_avg': acc_mmlu_avg})

    return (acc_cumul + acc_mmlu_avg) / (n_tasks - len(questions_per_mmlu_task) + 1)

def run_lm_eval(
    hflm: HFLM, task_list: list, fewshot: int, batch_size: int, fraction: float, output_file: str, log_msg: str
):
    results = lm_eval.simple_evaluate(
        hflm, tasks=task_list, num_fewshot=fewshot, batch_size=batch_size, limit=fraction
    )['results']
    metrics = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in results.items()}
    metrics['acc_avg'] = round(sum(metrics.values()) / len(metrics.values()), 4)
    metrics['num_fewshot'] = fewshot
    metrics['limit'] = fraction
    logging.info(f"{log_msg} {metrics}")
    with open(output_file, "w") as f:
        json.dump(metrics, f)


def eval_main(args: argparse.Namespace) -> None:
    logging.info("Running SliceGPT LM eval experiment.")

    logging.info(f"PyTorch device: {config.device}")
    logging.info(f"Number of available cuda devices: {torch.cuda.device_count()}")

    try:
        wandb.init(project=args.wandb_project, config=args, mode='disabled' if args.no_wandb else None)
    except wandb.UsageError as e:
        # wandb.init will throw an error if the user is not logged in and the process is running in a non-shell
        # environment, e.g. notebook, IDE, no-shell process, etc. In this case, we want to continue without wandb.
        logging.info(f'Failed to initialize wandb: {e}, continuing without wandb')
        wandb.init(project=args.wandb_project, mode='disabled')

    model = Phi3ForCausalLM.from_pretrained(args.model_path, torch_dtype="auto",
                                                 trust_remote_code=True, local_files_only=True,
                                                 attn_implementation="flash_attention_2")  # or eager for onnx
    model.to(config.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)

    # the lm eval harness ties the weights, but this should not be done for sliced models unless the lm_head was sliced
    model.tie_weights = lambda: None

    ### LM Eval Harness ###
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)

    initialize_tasks()
    task_names = lm_eval_utils.pattern_match(args.tasks, ALL_TASKS)

    mmlu_tasks = [
        "mmlu_abstract_algebra",
        "mmlu_business_ethics",
        "mmlu_college_computer_science",
        "mmlu_college_mathematics",
        "mmlu_conceptual_physics",
        "mmlu_formal_logic",
        "mmlu_machine_learning",
        "mmlu_miscellaneous",
        "mmlu_philosophy",
        "mmlu_global_facts",
    ]

    run_lm_eval(
        hflm,
        task_list=task_names,
        batch_size=args.batch_size,
        fewshot=args.num_fewshot,
        fraction=1,
        output_file=f"{args.save_dir}/lm_eval.json",
        log_msg="LM Eval results (limit=1): ",
    )


if __name__ == "__main__":
    # Use the logger from lm_eval, adding a file handler to write the log to file
    logging = lm_eval_utils.eval_logger
    logging.addHandler(utils.create_file_handler(log_dir="log"))

    os.environ["WANDB__SERVICE_WAIT"] = "300"

    eval_args = eval_arg_parser()
    process_eval_args(eval_args)
    eval_main(eval_args)
