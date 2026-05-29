# evo-posttrainbench

Replication harness for running **[evo](https://github.com/evo-hq/evo)** on **PostTrainBench** — measuring whether evo's structured optimize loop plus its `finetuning` skill helps a CLI agent post-train a model, versus the published baselines.

Fork of **[aisa-group/PostTrainBench](https://github.com/aisa-group/PostTrainBench)** (arXiv:2603.08640, MIT). We keep their eval harness (the task `evaluate.py` + templates + prompt) and add an evo-driven agent plus an apptainer-free runner for a single rented H100. Upstream's other agents, container images, judge, and tooling are removed — pull them from `upstream` if you need them.

## What it runs

The agent is **Claude Code + the evo plugin + its `finetuning` skill**, on a Max subscription (OAuth), Opus 4.6 at effort=max. It post-trains a base model on **AIME 2025** under PostTrainBench's rules and is scored by the task's `evaluate.py`.

Rules (unchanged): 10h on 1 H100; no test data in training; don't modify `evaluate.py` or `templates/`; only fine-tune the provided base model; `final_model` must run in the starting environment. These map onto evo gates.

## Layout

- `agents/claude_evo_max/` — the OAuth agent (evo engaged).
- `scripts/run_evo_jarvislabs.sh` — apptainer-free `bootstrap` + `run`.
- `src/eval/` — upstream eval harness (the AIME task, jinja templates, prompt generator).
- `containers/requirements-direct.txt` — the pinned starting environment.

## Replicate (single rented H100; keep everything under `/home`, which persists)

1. `git clone https://github.com/evo-hq/evo-posttrainbench.git && cd evo-posttrainbench`
2. `WORK=/home/$(whoami)/ptb bash scripts/run_evo_jarvislabs.sh bootstrap`
   — installs the pinned env + vLLM + flash-attn + `inspect_evals` + Claude Code, and evo from its `feat/model-update` branch (which carries the `finetuning` skill).
3. Once: `claude setup-token` (browser OAuth) → save the token to `$WORK/oauth_token`; put `HF_TOKEN` (Gemma-3-4B is gated) and `WANDB_API_KEY` in `$WORK/.env`.
4. Smoke-test one short cell, then run:
   `WORK=/home/$(whoami)/ptb bash scripts/run_evo_jarvislabs.sh run aime2025 Qwen/Qwen3-4B 10`

Results land in `$WORK/runs/<...>/`: `prompt.txt`, `solve_out.txt` + `solve_parsed.txt` (the agent transcript), `final_model/`, `metrics.json`, `final_eval.txt`.

## Models

AIME 2025 × {Qwen3-4B, Gemma-3-4B, SmolLM3-3B, Qwen3-1.7B}. Lean on the 4B models (1.7B tends to floor on AIME). Compare to the published Claude-Code baselines on the [leaderboard](https://posttrainbench.com) rather than running your own baseline.

## Monitor

evo dashboard (tree/scores/frontier; served on `:8080`, reach via `ssh -L 8080:localhost:8080 <host>`), W&B for training curves (survives pause, no tunnel), `nvtop` for the GPU — all inside `tmux` so it survives disconnect.

## Caveats — untested; smoke-test first

The runner runs the agent and `evaluate.py` **directly on the host** (no apptainer). It does **not** replicate the contamination judge or the fuse-overlayfs HF isolation, and it does a single eval pass (upstream adds max-token retries). Diff against upstream's harness before trusting numbers. Things to validate: evo's optimize loop inside a single headless `claude --print`; single-GPU serialization (one training job at a time, rollouts/eval batch within one vLLM server); apptainer-free eval parity.

## Credit

PostTrainBench by aisa-group (arXiv:2603.08640). This fork retains their MIT `LICENSE`.
