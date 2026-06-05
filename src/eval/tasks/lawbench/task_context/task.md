# LawBench — Criminal Charge Prediction (191-class)

Maximize **top-1 accuracy** on 913 held-out Chinese criminal cases: given a case
fact summary (事实), predict the single correct charge (罪名) from the **fixed list
of 191 charges** in `classes.json`.

## This is a whole-system task — improve anything

You are NOT restricted to fine-tuning. Improve accuracy with **any combination**
of levers, and iterate on them:

- **Harness / scaffold / prompts** — how you elicit a prediction: prompt design,
  few-shot exemplars from the train set, retrieval over training cases,
  answer-extraction/normalization, even a non-LLM classifier (e.g. TF-IDF +
  linear model) over the training data. A strong harness-only solution is valid.
- **Model weights** — fine-tune the base model (SFT / LoRA / RL) on the training
  data if it helps. Optional, not required.
- **Both** — a fine-tuned model inside a tuned harness.

## Data (in your working dir)

- `train.csv` — 5,332 rows `id,text,label`. Your only training/development data.
- `test.csv` — 913 rows `id,text` (no labels). The cases you must predict.
- `classes.json` — the 191 valid charge labels. Predict ONLY from this list;
  anything off-list scores as wrong.
- `test_gold.csv` — held-out gold labels, used ONLY by the scorer.

## Rules

- **Do NOT train on, fit to, or read `test.csv` labels / `test_gold.csv`.** All
  fitting uses `train.csv` only. (The validity verifier audits for leakage.)
- Predict exactly one of the 191 classes per case.

## Scoring (fixed — do not modify `evaluate.py`)

Top-1 exact-match accuracy over the 913. Two ways to be scored:

- General path — produce `submission.csv` (columns `id,label`) with your pipeline,
  then: `python evaluate.py --submission submission.csv --json-output-file out.json`
- Weights-only convenience — `python evaluate.py --model-path <dir> ...` serves the
  model with a default prompt and scores it.

Reference bars: prior SOTA 0.450; baseline ~0.135.
