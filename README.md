# evo-posttrainbench

Replication harness for running **[evo](https://github.com/evo-hq/evo)** on **PostTrainBench** — measuring whether evo's structured optimize loop helps a CLI agent post-train a model, versus the published baselines.

Fork of **[aisa-group/PostTrainBench](https://github.com/aisa-group/PostTrainBench)** (arXiv:2603.08640, MIT). We keep their eval harness (the task `evaluate.py` + templates + prompt) and add an evo-driven agent plus an apptainer-free runner for a single rented H100. Upstream's other agents, container images, judge, and tooling are removed — pull them from `upstream` if you need them.

## What it runs

The agent is **Claude Code + the evo plugin**, running on a **Claude Max subscription via OAuth** (consistent with PostTrainBench's `claude_non_api_max` agent — Opus 4.6, effort=max). It post-trains a base model on **AIME 2025** under PostTrainBench's rules and is scored by the task's `evaluate.py`.

Rules (unchanged): 10h on 1 H100; no test data in training; don't modify `evaluate.py` or `templates/`; only fine-tune the provided base model; `final_model` must run in the starting environment. These map onto evo gates.

## Layout

- `agents/claude_evo_max/` — the OAuth agent (evo engaged).
- `scripts/run.sh` — apptainer-free `bootstrap` + `run` (used by both paths below).
- `scripts/setup.sh` — interactive setup for a rented H100 (SSH).
- `scripts/modal_app.py` — Modal wrapper (containerized H100, persistent Volume, public dashboard URL).
- `src/eval/` — upstream eval harness (the AIME task, jinja templates, prompt generator).
- `containers/requirements-direct.txt` — the pinned starting environment.

## How to replicate

You need a single rented **H100** (JarvisLabs/RunPod). Keep everything under `/home` — it persists across pause; `/root` and system installs get wiped.

1. **Provision + SSH** into the H100. On JarvisLabs, also open port **8080** in the instance's exposed ports (for the dashboard).
2. **Set it up** — one interactive script (prompts for workspace + auth + secrets, installs everything, sanity-checks, and **exits early if there's no GPU**):
   ```
   git clone https://github.com/evo-hq/evo-posttrainbench.git && cd evo-posttrainbench && bash scripts/setup.sh
   ```
   Claude auth uses your **Max-subscription OAuth token** — run `claude setup-token` on your laptop, paste the token when the script asks. `google/gemma-3-4b-pt` is gated, so have an `HF_TOKEN` ready.
3. **Run** — inside `tmux` so it survives disconnect:
   ```
   WORK=/home/ptb bash scripts/run.sh run aime2025 Qwen/Qwen3-4B-Base 1     # 1h smoke first
   WORK=/home/ptb bash scripts/run.sh run aime2025 Qwen/Qwen3-4B-Base 10
   WORK=/home/ptb bash scripts/run.sh run aime2025 google/gemma-3-4b-pt 10
   ```
   Results land in `$WORK/runs/<...>/`: `prompt.txt`, `solve_parsed.txt` (agent transcript), `final_model/`, `metrics.json`, `final_eval.txt`.
4. **Monitor** — `run` exports `EVO_DASHBOARD_HOST=0.0.0.0` so evo's auto-started dashboard is reachable on the instance's port 8080 directly; plus `nvtop` and W&B (see [Monitor](#monitor)).

After a pause, re-run step 2 (`setup.sh`, or just `… bootstrap`) — `/home` survives but the installs don't. Prefer manual? `setup.sh` only chains `run.sh bootstrap` → write `$WORK/.env` (+ `$WORK/oauth_token`) → `run`.

## Replicate (Modal — serverless H100)

Same flow, in a container Modal grants per function call. No instance to babysit, persistent `Volume` across runs, public HTTPS dashboard with scale-to-zero.

1. Install + auth locally: `pip install modal && modal token new`.
2. Generate a Claude Code OAuth token on your laptop: `claude setup-token`. Then create the Modal secrets once:
   ```
   modal secret create anthropic CLAUDE_CODE_OAUTH_TOKEN=...   # Max subscription
   modal secret create hf        HF_TOKEN=...                  # gemma-3-4b-pt is gated
   modal secret create wandb     WANDB_API_KEY=...             # optional
   ```
3. **Dry run** — cheap pipeline check (~few min): GPU, deps, Volume, secrets, CLIs.
   ```
   modal run scripts/modal_app.py::dry_run
   ```
4. **Train** — real 10h run on an H100 (~$40 per cell):
   ```
   modal run scripts/modal_app.py::train --model Qwen/Qwen3-4B-Base
   modal run scripts/modal_app.py::train --model google/gemma-3-4b-pt
   ```
   `train()` runs the same `scripts/run.sh run aime2025 ...` inside the container, with `$WORK=/workspace` on a persistent Volume (HF cache + `.evo/` + checkpoints survive between runs). The function is server-side — close your terminal; Modal keeps running.
5. **Dashboard** — public HTTPS URL, scale-to-zero:
   ```
   modal deploy scripts/modal_app.py
   ```
   Modal prints a URL like `https://<workspace>--evo-posttrainbench-dashboard.modal.run`.

First image build ~15 min; cached afterwards. `gpu="H100!"` pins H100 (without `!` Modal silently upgrades to H200, changing kernels + price). `modal shell <container-id>` attaches a debug shell to a live container.

## Models

AIME 2025 × two base models: **`Qwen/Qwen3-4B-Base`** and **`google/gemma-3-4b-pt`**. Compare to the published Claude-Code baselines on the [leaderboard](https://posttrainbench.com) rather than running your own baseline.

## Monitor

- **evo dashboard** (tree/scores/frontier/traces): served on `:8080`. `run.sh run` already exports `EVO_DASHBOARD_HOST=0.0.0.0` so evo's auto-started dashboard binds publicly — **open port 8080 on the instance** (JarvisLabs UI exposed ports) and browse the proxy URL. For a standalone view of the latest run: `bash scripts/run.sh dashboard`. SSH tunnel `ssh -L 8080:localhost:8080 <host>` works as a fallback.
- **W&B** for training curves (survives pause, no tunnel); **`nvtop`** for the GPU.
- Run inside `tmux` so it survives disconnect.

## Caveats — untested; smoke-test first

The runner runs the agent and `evaluate.py` **directly on the host** (no apptainer). It does **not** replicate the contamination judge or the fuse-overlayfs HF isolation, and it does a single eval pass (upstream adds max-token retries). Diff against upstream's harness before trusting numbers. Things to validate: evo's optimize loop inside a single headless `claude --print`; single-GPU serialization (one training job at a time, rollouts/eval batch within one vLLM server); apptainer-free eval parity.

## Credit

PostTrainBench by aisa-group (arXiv:2603.08640). This fork retains their MIT `LICENSE`.
