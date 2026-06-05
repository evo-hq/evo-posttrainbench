#!/usr/bin/env python3
"""LawBench — Criminal Charge Prediction (191-class). Top-1 accuracy.

This is the WHOLE-SYSTEM task, SIA-style: the agent may improve accuracy by ANY
combination of levers — the harness/scaffold/prompts/retrieval/approach AND/OR
the model weights (fine-tuning the base). The model is optional: a pure-harness
solution (e.g. TF-IDF + linear classifier, retrieval, prompt engineering of the
base model) is valid, as is a fine-tuned model, as is both.

Scoring is fixed and is the only thing the agent must NOT touch: given the
held-out gold labels, top-1 exact-match accuracy over the 913 test cases
(prediction must be one of the 191 classes; off-list / missing = wrong).

Two ways to be scored:
  * `evaluate.py --submission submission.csv`  — score predictions your pipeline
    produced (CSV with columns id,label). This is the general path: your harness
    can be anything that emits predictions.
  * `evaluate.py --model-path <dir>`           — convenience path: serve a model
    with a default prompt, predict, and score (weights-only iteration).

Data (in the job from task_context/): classes.json (191 labels), test.csv
(id,text — the 913 to predict), test_gold.csv (id,label — gold). Train on
train.csv; do NOT use test.csv labels / test_gold.csv for training.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LawBench charge prediction — top-1 accuracy.")
    p.add_argument("--submission", type=str, default=None,
                   help="CSV (id,label) your pipeline produced. Takes precedence over --model-path.")
    p.add_argument("--model-path", type=str, default=None,
                   help="HF model dir/id to serve with a default prompt (weights-only convenience path).")
    p.add_argument("--limit", type=int, default=None,
                   help="Evaluate only the first N test cases (fast smoke).")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--json-output-file", type=str, default=None)
    p.add_argument("--max-connections", type=int, default=6, help="Unused; harness parity.")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    p.add_argument("--templates-dir", type=str, default="templates/")
    p.add_argument("--data-dir", type=str, default=".",
                   help="Where classes.json / test.csv / test_gold.csv live.")
    return p.parse_args()


# --- model-type -> chat template (mirrors aime2025/evaluate.py) ---------------
def model_type(model_path: str) -> str:
    ml = model_path.lower()
    for k in ("qwen", "llama", "gemma", "smollm"):
        if k in ml:
            return k
    cfg = os.path.join(model_path, "config.json")
    if os.path.isfile(cfg):
        arch = json.load(open(cfg))["architectures"][0].lower()
        for k in ("gemma", "llama", "qwen", "smollm"):
            if k in arch:
                return k
    return "qwen"


def load_template(templates_dir: str, model_path: str) -> str | None:
    name = {"qwen": "qwen3.jinja", "llama": "llama3.jinja",
            "gemma": "gemma3.jinja", "smollm": "smollm.jinja"}[model_type(model_path)]
    path = os.path.join(templates_dir, name)
    return open(path).read() if os.path.isfile(path) else None


def build_prompt(text: str) -> str:
    return (
        "你是中国刑事审判法官。请根据下面的案件事实，判断被告人所犯的罪名。"
        "只输出一个罪名的名称本身，不要输出案由分析、刑期或任何其他内容。\n\n"
        f"案件事实：{text}\n\n罪名："
    )


def parse_pred(output: str, classes: list[str], class_set: set[str]) -> str:
    s = re.sub(r"^罪名[:：]\s*", "", (output or "").strip()).strip()
    if s in class_set:
        return s
    first = re.sub(r"^罪名[:：]\s*", "", s.splitlines()[0].strip()).strip() if s else ""
    if first in class_set:
        return first
    cand = [c for c in classes if c and c in s]
    return max(cand, key=len) if cand else "__unknown__"


def predict_with_model(args, tests, classes, class_set) -> dict[str, str]:
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model_path, gpu_memory_utilization=args.gpu_memory_utilization,
              trust_remote_code=True)
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    tmpl = load_template(args.templates_dir, args.model_path)
    convs = [[{"role": "user", "content": build_prompt(r["text"])}] for r in tests]
    try:
        outs = llm.chat(convs, sp, chat_template=tmpl) if tmpl else llm.chat(convs, sp)
    except Exception:
        outs = llm.generate([build_prompt(r["text"]) for r in tests], sp)
    return {r["id"]: parse_pred(o.outputs[0].text, classes, class_set) for r, o in zip(tests, outs)}


def main() -> None:
    args = parse_args()
    dd = args.data_dir
    classes = json.load(open(os.path.join(dd, "classes.json")))
    class_set = set(classes)
    tests = list(csv.DictReader(open(os.path.join(dd, "test.csv"))))
    gold = {r["id"]: r["label"] for r in csv.DictReader(open(os.path.join(dd, "test_gold.csv")))}
    if args.limit is not None and args.limit > 0:
        tests = tests[: args.limit]
    ids = [r["id"] for r in tests]

    if args.submission:
        pred = {r["id"]: r.get("label", "") for r in csv.DictReader(open(args.submission))}
    elif args.model_path:
        pred = predict_with_model(args, tests, classes, class_set)
    else:
        raise SystemExit("provide --submission <csv> or --model-path <dir>")

    correct = sum(1 for i in ids if pred.get(i, "__missing__") == gold.get(i, "__gold_missing__"))
    n = len(ids)
    acc = round(correct / n, 4) if n else 0.0
    metrics = {"accuracy": acc, "n_correct": correct, "n_total": n}
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.json_output_file:
        with open(args.json_output_file, "w") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
