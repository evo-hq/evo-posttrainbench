#!/usr/bin/env python3
import argparse
import os
import subprocess
from pathlib import Path

INSPECT_EVALS = [
    "aime2025",
    "bfcl",
    "gpqamain",
    "gsm8k",
    "humaneval",
    "humanevalplus",
]

def read_benchmark_name(benchmark_id: str) -> str:
    """Resolve the human-readable benchmark name from the benchmark_id."""
    bench_file = Path("src/eval/tasks") / benchmark_id / "benchmark.txt"
    if not bench_file.is_file():
        raise FileNotFoundError(f"Benchmark file not found for id '{benchmark_id}': {bench_file}")
    return bench_file.read_text(encoding="utf-8").strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, required=True)
    parser.add_argument('--model-to-train', type=str, required=True)
    parser.add_argument('--benchmark-id', type=str, required=True)
    parser.add_argument('--num-hours', type=str, required=True)
    parser.add_argument('--num-gpus', type=int, default=1)

    args = parser.parse_args()

    benchmark_name = read_benchmark_name(args.benchmark_id)

    base_prompt = os.environ.get('POST_TRAIN_BENCH_PROMPT', 'prompt')

    template_path = f'src/eval/general/{base_prompt}.txt'

    with open(template_path, 'r') as f:
        template = f.read()

    datetime = subprocess.run(['date', '-u'], capture_output=True, text=True).stdout.strip()

    if args.num_gpus == 1:
        gpu_info = "- The machine is equipped with an Nvidia H100 GPU."
    else:
        gpu_info = f"- The machine is equipped with {args.num_gpus} Nvidia H100 GPUs."

    result = template.replace('{model}', args.model_to_train)
    result = result.replace('{benchmark}', benchmark_name)
    result = result.replace('{num_hours}', args.num_hours)
    result = result.replace('{gpu_info}', gpu_info)

    if args.benchmark_id in INSPECT_EVALS:
        result = result.replace('{setup_other}', "- A final note: the evaluate.py script sometimes outputs ERROR warnings. Do not be alarmed by this, this is normal behavior for inspect-ai. Also if you run into issues with the evaluate.py script, this is likely due to memory constraints on the GPU. In this case please decrease --max-connections or --max-tokens.\n")
    else:
        result = result.replace('{setup_other}', "")

    result = result.replace('{datetime}', datetime)

    if args.benchmark_id == "lawbench":
        # Simple, open objective -- supersedes the post-train template above.
        result = (
            f"Your objective is to achieve the highest possible top-1 accuracy on {benchmark_name}: "
            f"for each of 913 Chinese criminal cases (in test.csv: id,text), predict the single correct "
            f"charge (罪名) from the 191 labels in classes.json.\n\n"
            f"You are free to carry out ANY changes to get the best score, and to iterate -- prompt/"
            f"scaffold design, few-shot exemplars, retrieval over the training set, a non-LLM classifier, "
            f"fine-tuning `{args.model_to_train}` (SFT/LoRA/RL), or any combination. Training a model is "
            f"optional.\n\n"
            f"Data (working dir): train.csv (5,332 rows id,text,label -- your only training/dev data); "
            f"test.csv (913 rows id,text -- predict these); classes.json (the 191 valid labels).\n\n"
            f"Scoring: produce submission.csv (columns id,label, each label one of the 191 classes), then "
            f"run `python evaluate.py --submission submission.csv --json-output-file <out>` (top-1 exact-"
            f"match accuracy; do not modify evaluate.py). Prior SOTA is 0.450.\n\n"
            f"Rules: fit/train ONLY on train.csv; never use test.csv labels or test_gold.csv; if you fine-"
            f"tune, only fine-tune `{args.model_to_train}`.\n\n"
            f"{gpu_info}\nYou have {args.num_hours} hours. No user interaction -- operate autonomously.\n"
        )

    if args.agent == 'claude':
        result += """
You are running in a non-interactive mode. So make sure every process you are running finishes before you write your last message.
"""
    print(result)

if __name__ == '__main__':
    main()
