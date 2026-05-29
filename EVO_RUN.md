# Running the evo variant on a single rented H100

This fork adds an **evo** variant of PostTrainBench: the agent is Claude Code + the
[evo](https://github.com/evo-hq/evo) plugin + its `finetuning` skill, on a Max
subscription (OAuth). It runs the same task under the same rules; we're measuring
whether evo's structured loop + finetuning judgment helps an agent post-train a
model, vs the published Claude-Code baselines.

## What's added

- `agents/claude_evo_max/` — Claude Code on OAuth (Opus 4.6, effort=max), evo engaged.
- `scripts/run_evo_jarvislabs.sh` — apptainer-free `bootstrap` + `run` for a bare H100.

The upstream `src/run_task.sh` wraps everything in apptainer `.sif` images, which is
painful inside an already-containerized cloud GPU (JarvisLabs/RunPod). Our runner runs
the agent and `evaluate.py` **directly on the host**. It does **not** replicate the
contamination judge or the fuse-overlayfs HF isolation, and it does a single eval pass
(upstream adds max-token retries). So treat its numbers as indicative until you've
diffed against the official harness.

## Constraints (unchanged — they map onto evo gates)

- 10h, single H100. No test data in training. Don't touch `evaluate.py`/`templates/`.
  Only fine-tune the provided base model. `final_model` must run in the starting env.

## Setup (JarvisLabs: keep everything under `/home`, which is persistent)

```bash
git clone -b feat/model-update https://github.com/evo-hq/evo.git   # (the runner also does this)
cd <this-fork>
WORK=/home/$(whoami)/ptb bash scripts/run_evo_jarvislabs.sh bootstrap
```

Then, once:
- `claude setup-token` (locally, browser OAuth) → save the token to `$WORK/oauth_token`.
- `$WORK/.env`: `HF_TOKEN` (Gemma-3-4B is gated — accept its licence first), `WANDB_API_KEY`, optional `OPENAI_API_KEY` (judge).

## Run one cell

```bash
WORK=/home/$(whoami)/ptb bash scripts/run_evo_jarvislabs.sh run aime2025 Qwen/Qwen3-4B 10
```

Results land in `$WORK/runs/<...>/`: `prompt.txt`, `solve_out.txt` + `solve_parsed.txt`
(the agent transcript), `final_model/`, `metrics.json`, `final_eval.txt`.

**Smoke-test first:** run one short cell (e.g. Qwen3-1.7B, 1h) and confirm bootstrap →
agent → `evaluate.py` → results all work before spending a 10h block or the matrix.

## Matrix

AIME 2025 × {Qwen3-4B, Gemma-3-4B, SmolLM3-3B, Qwen3-1.7B}. Lean on the 4B models
(1.7B tends to floor on AIME). Compare to the published Claude-Code baselines on the
[leaderboard](https://posttrainbench.com) rather than running your own baseline arm.

## Monitor

- **evo dashboard** (tree, scores, frontier, traces): served on `:8080` → SSH tunnel
  `ssh -L 8080:localhost:8080 <host>`.
- **W&B** for training curves (loss/reward/KL) — cloud, survives pause, no tunnel.
- `nvtop` for the GPU; `tail -f $WORK/runs/<...>/solve_out.txt` for the agent.
- Run inside `tmux` so it survives SSH disconnect.

## Open items to validate

- evo's optimize loop running inside a single headless `claude --print` invocation.
- Single-GPU serialization: only one GPU training job at a time (`subagents=1` for the
  GPU step); rollouts/eval batch within one vLLM server.
- Apptainer-free eval parity with the official harness before reporting numbers.
