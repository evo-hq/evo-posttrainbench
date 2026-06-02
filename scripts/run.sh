#!/bin/bash
# Apptainer-free PostTrainBench runner for a single H100 host -- works on a rented
# instance (JarvisLabs/RunPod) or inside a Modal container. Runs the agent + the
# evaluate.py directly on the host (their src/run_task.sh wraps everything in
# apptainer .sif images, which is painful inside an already-containerized cloud GPU).
# Deliberately skips the contamination judge and the fuse-overlayfs HF isolation,
# and does a single eval pass -- SMOKE-TEST one cell before trusting any number.
#
# Layout: everything lives under $WORK (default /home/<user>/ptb on JarvisLabs;
# /workspace inside Modal). Persisted there across pauses / function calls.
set -euo pipefail

CMD="${1:-help}"
WORK="${WORK:-/home/$(whoami)/ptb}"
# Derive REPO from this script's own location instead of $(pwd) -- otherwise
# `ssh host 'bash /abs/path/run.sh ...'` resolves REPO to the SSH login CWD
# (typically $HOME), which breaks every $REPO-relative path the script uses.
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(dirname "$_SCRIPT_DIR")}"
EVO_BRANCH="${EVO_BRANCH:-feat/model-update}"
export HF_HOME="${HF_HOME:-$WORK/hf}"
export CLAUDE_CONFIG_DIR="${CLAUDE_CONFIG_DIR:-$WORK/.claude}"
# uv (and the per-user-installed evo CLI it brings in) lives in ~/.local/bin.
# Interactive shells get this via .profile; non-interactive ssh invocations
# and child processes spawned by the agent's claude session do not. Export
# at top level so every subcommand (run, dashboard) and every shell the
# agent spawns can find `evo`.
export PATH="$HOME/.local/bin:$PATH"

require_gpu() {
  if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L 2>/dev/null | grep -q 'GPU'; then
    echo "ERROR: no NVIDIA GPU found (need a GPU host)." >&2; exit 1
  fi
}

bootstrap() {
  require_gpu
  # System-package and global-Python installs need root; per-user uv tool
  # install + Claude Code plugin install do not. SUDO is empty when already
  # root (e.g. JL container template), `sudo` otherwise (e.g. JL vm template
  # whose default user is `ubuntu`). Passwordless sudo is assumed on
  # non-root hosts -- without it, apt-get/npm/uv-system installs hang.
  local SUDO=""
  [ "$(id -u)" -ne 0 ] && SUDO="sudo"
  mkdir -p "$WORK" "$HF_HOME" "$CLAUDE_CONFIG_DIR" "$WORK/runs"
  # python-is-python3: Ubuntu 22.04 ships only `python3`; many scripts (this
  # one, evaluate.py invocations, the agent's training code) call bare `python`.
  # The symlink package is one line and removes a class of "command not found"
  # surprises mid-run.
  $SUDO apt-get install -y -q python-is-python3
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  command -v node >/dev/null || { curl -fsSL https://deb.nodesource.com/setup_22.x | $SUDO bash - && $SUDO apt-get install -y nodejs; }
  $SUDO npm install -g @anthropic-ai/claude-code@2.1.76          # match the version they ran

  # PostTrainBench starting environment (pinned) + vLLM + flash-attn
  # Pin to cu128: --torch-backend=auto fails during builds with no GPU attached
  # (e.g. Modal image builds); cu128 wheels are compatible with our cuda:12.9.1 base.
  $SUDO env "PATH=$PATH" uv pip install --system --no-cache vllm==0.11.0 --torch-backend=cu128
  $SUDO env "PATH=$PATH" uv pip install --system --no-cache -r "$REPO/containers/requirements-direct.txt"
  # trackio: wandb-API-compatible OSS tracker; logs to a HF Space.
  # Pin <0.10 -- 0.10+ requires gradio 6 + huggingface-hub>=1.0, conflicts with
  # PostTrainBench's pinned transformers 4.57.3 (needs hf-hub<1.0).
  $SUDO env "PATH=$PATH" uv pip install --system --no-cache 'trackio<0.10'
  # wheel needed for flash-attn's --no-build-isolation (it doesn't declare wheel as a build dep)
  $SUDO env "PATH=$PATH" uv pip install --system --no-cache wheel setuptools
  $SUDO env "PATH=$PATH" uv pip install --system --no-cache flash-attn==2.8.3 --no-build-isolation

  # eval deps: inspect_evals registers the task (e.g. inspect_evals/aime2025); the
  # vLLM-stdout inspect_ai fork is what their evaluate.py uses. Pinned to match upstream.
  local INS; INS=$(mktemp -d)
  git clone https://github.com/UKGovernmentBEIS/inspect_evals.git "$INS/inspect_evals" \
    && ( cd "$INS/inspect_evals" && git checkout 06001a83e6d7c709c2ede0570dce7f1031a0bad8 \
         && $SUDO env "PATH=$PATH" uv pip install --system --no-cache . )
  git clone https://github.com/rank-and-file/inspect_ai_vllm_stdout.git "$INS/inspect_ai_vllm_stdout" \
    && ( cd "$INS/inspect_ai_vllm_stdout" && $SUDO env "PATH=$PATH" uv pip install --system --no-cache . )

  # evo from our branch + register the plugin (incl. the finetuning skill) into Claude Code
  # On re-bootstrap (after a JL pause, etc.) $WORK/evo persists -- fetch + hard-reset to
  # branch tip so we don't sit on a stale commit. uv tool install --editable below picks
  # up the refreshed tree; --force re-creates the entry point shim if the tool was already
  # installed from an older sha.
  if [ -d "$WORK/evo/.git" ]; then
    ( cd "$WORK/evo" && git fetch origin "$EVO_BRANCH" \
        && git checkout "$EVO_BRANCH" \
        && git reset --hard "origin/$EVO_BRANCH" )
  else
    git clone -b "$EVO_BRANCH" https://github.com/evo-hq/evo.git "$WORK/evo"
  fi
  uv tool install --force --editable "$WORK/evo/plugins/evo"
  # Install the plugin from the LOCAL evo clone (feat/model-update tip)
  # rather than the public marketplace -- the marketplace points at the
  # stable release tag (currently 0.4.4) which lags behind feat/model-update.
  # --from-path uses the same source the CLI was built from, so skills + CLI
  # versions stay in sync. Critical for picking up the rewritten finetuning
  # skill, the new prompt, etc.
  evo install claude-code --from-path "$WORK/evo"

  echo "Bootstrap done."
  echo "  1) generate an OAuth token locally:  claude setup-token   -> save it to $WORK/oauth_token"
  echo "  2) put keys in $WORK/.env:           HF_TOKEN (Gemma is gated), WANDB_API_KEY, OPENAI_API_KEY (judge, optional)"
}

run() {
  require_gpu
  local TASK="${1:-aime2025}"
  local MODEL="${2:-Qwen/Qwen3-4B-Base}"
  local HOURS="${3:-10}"
  local AGENT="claude_evo_max"
  local AGENT_CONFIG="${AGENT_CONFIG:-claude-opus-4-6}"

  [ -f "$WORK/.env" ] && { set -a; source "$WORK/.env"; set +a; }
  export OAUTH_TOKEN_FILE="$WORK/oauth_token"
  if [ ! -f "$OAUTH_TOKEN_FILE" ] && [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -z "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]; then
    echo "no Claude auth: set CLAUDE_CODE_OAUTH_TOKEN env (e.g. via Modal secret), create $OAUTH_TOKEN_FILE (claude setup-token), or set ANTHROPIC_API_KEY in $WORK/.env"; exit 1
  fi

  # Bind evo's auto-started dashboard to 0.0.0.0 so it's reachable on the cloud
  # instance / Modal web URL (requires evo >= the EVO_DASHBOARD_HOST commit).
  export EVO_DASHBOARD_HOST="${EVO_DASHBOARD_HOST:-0.0.0.0}"
  export EVO_DASHBOARD_PORT="${EVO_DASHBOARD_PORT:-8080}"
  # Sandbox-mode escape hatch: cloud containers run as root by default, and
  # claude --dangerously-skip-permissions otherwise refuses root. The whole
  # container IS the sandbox, so this is the intended use.
  export IS_SANDBOX="${IS_SANDBOX:-1}"
  # trackio: free, OSS, wandb-API-compatible -- logs to a HF Space for live curves
  export TRACKIO_SPACE_ID="${TRACKIO_SPACE_ID:-alok97/posttrain-runs}"

  local RUN JOB
  RUN="$WORK/runs/${AGENT}_${TASK}_$(echo "$MODEL" | tr '/:' '__')_$(date +%s)"
  JOB="$RUN/task"; mkdir -p "$JOB"

  # task setup (apptainer-free copy of run_task.sh's prep)
  cp "src/eval/tasks/$TASK/evaluate.py" "$JOB/"
  [ -d "src/eval/tasks/$TASK/evaluation_code" ] && cp -r "src/eval/tasks/$TASK/evaluation_code" "$JOB/"
  cp -r src/eval/templates "$JOB/"
  [ -d "src/eval/tasks/$TASK/task_context" ] && cp -r "src/eval/tasks/$TASK/task_context/"* "$JOB/" || true
  bash src/utils/create_timer.sh "$HOURS" "$JOB/timer.sh"

  # Stop hook: don't let the agent declare itself "done" before the budget is
  # spent or a final_model exists. Scoped per-run via .claude/settings.json in
  # the task dir (= CLAUDE_PROJECT_DIR at agent launch).
  mkdir -p "$JOB/.claude"
  cp "$REPO/scripts/keep_going.sh" "$JOB/keep_going.sh"
  chmod +x "$JOB/keep_going.sh"
  cat > "$JOB/.claude/settings.json" <<'EOF'
{
  "hooks": {
    "Stop": [
      { "matcher": "", "hooks": [ { "type": "command", "command": "bash ${CLAUDE_PROJECT_DIR}/keep_going.sh" } ] }
    ]
  }
}
EOF

  # prompt = imperative evo-engagement preamble + PostTrainBench's standard task prompt.
  # NB: the heredoc is written to a file *outside* a $(...) subshell -- bash 3.2
  # (macOS default; matters for `bash -n` linting locally) miscounts heredocs in
  # command substitution. The Modal container has bash 5 and doesn't care, but
  # we want clean local lints too.
  local BASE BENCH_NAME PROMPT
  BASE=$(python3 src/eval/general/get_prompt.py --model-to-train "$MODEL" --benchmark-id "$TASK" --num-hours "$HOURS" --num-gpus 1 --agent "$AGENT")
  BENCH_NAME=$(tr -d '\n' < "src/eval/tasks/$TASK/benchmark.txt")
  cat > "$RUN/prompt.txt" <<'EOF'
We want to improve __MODEL__'s performance on __BENCH__ via evo's
optimization process.

This is a fully autonomous research run -- no user is present, operate
end-to-end on your own to the time budget.

The evo plugin is available with these skills (invoke via the Skill tool
when relevant; load on demand, not upfront):

  - evo:discover    initialize evo for the project: explore, propose
                    optimization dimensions, construct the benchmark,
                    run the first experiment
  - evo:optimize    run the optimization loop. Spawns parallel subagents
                    that each carry one experiment to completion (the
                    subagents auto-load the evo:subagent protocol).
                    Pass args "autonomous" for continuous mode.
  - evo:finetuning  pick or diagnose a training move (SFT/LoRA/DPO/KTO/
                    ORPO/RFT/GRPO/PPO/RLOO); reward-shape decision tree,
                    smoke-run gate, failure diagnostics

Start with evo:discover. When discover is done, invoke evo:optimize with
args "autonomous". Pull evo:finetuning when picking or diagnosing a
training technique.

Available infra in env (use when relevant, ignore otherwise):
  - TRACKIO_SPACE_ID -- wandb-API-compatible OSS tracker. Wire into
    training scripts (TRL: report_to="trackio"; custom loop: see
    evo:finetuning references/observability.md) for live loss curves
    in a public HF Space. Reduces the observability-blind window during
    long training runs.
  - HF_TOKEN -- HuggingFace auth. Use for gated datasets/models
    (Gemma, Llama-Instruct, etc.) and private Hub uploads if useful.

EOF
  sed -i.bak -e "s|__MODEL__|$MODEL|g" -e "s|__BENCH__|$BENCH_NAME|g" "$RUN/prompt.txt"
  rm -f "$RUN/prompt.txt.bak"
  printf '%s' "$BASE" >> "$RUN/prompt.txt"
  PROMPT=$(cat "$RUN/prompt.txt")

  # run the agent directly (no apptainer), bounded by the hour budget
  export PROMPT AGENT_CONFIG
  # tee so the agent's stream-json shows in `modal app logs` AND persists to disk
  ( cd "$JOB" && timeout --signal=TERM --kill-after=60s "$((HOURS * 60 + 5))m" \
      bash "$REPO/agents/$AGENT/solve.sh" ) 2>&1 | tee "$RUN/solve_out.txt" || true
  python3 "agents/$AGENT/human_readable_trace.py" "$RUN/solve_out.txt" -o "$RUN/solve_parsed.txt" || true

  # evaluate final_model (single pass; their harness adds judge + max-token retries)
  if [ -d "$JOB/final_model" ]; then
    ( cd "src/eval/tasks/$TASK" && python3 evaluate.py \
        --model-path "$JOB/final_model" --templates-dir ../../templates \
        --limit -1 --json-output-file "$RUN/metrics.json" ) | tee "$RUN/final_eval.txt"
  else
    echo "no final_model produced -- baseline score stands" | tee "$RUN/final_eval.txt"
  fi
  echo "results: $RUN"
}

dashboard() {
  # Standalone dashboard against the latest run dir (during a run, evo auto-starts
  # one inside the agent's session bound to 0.0.0.0:8080 via EVO_DASHBOARD_HOST).
  local LATEST
  LATEST=$(ls -1dt "$WORK"/runs/*/task 2>/dev/null | head -1)
  [ -n "$LATEST" ] || { echo "no runs under $WORK/runs yet"; exit 1; }
  echo "evo dashboard for $LATEST on 0.0.0.0:8080 (open port 8080 on the instance)"
  ( cd "$LATEST" && EVO_DASHBOARD_HOST=0.0.0.0 EVO_DASHBOARD_PORT=8080 exec evo dashboard )
}

watch() {
  # Live human-readable view of the latest run's stream-json transcript.
  # No-arg: auto-find the latest $WORK/runs/*/solve_out.txt.
  # First arg as existing path: tail that path instead.
  # All remaining args (e.g. -v, -t, --no-color) are forwarded to watch_run.py.
  local LATEST=""
  if [ "$#" -gt 0 ] && [ -e "$1" ]; then
    LATEST="$1"; shift
  fi
  if [ -z "$LATEST" ]; then
    LATEST=$(ls -1dt "$WORK"/runs/*/solve_out.txt 2>/dev/null | head -1)
    [ -n "$LATEST" ] || { echo "no solve_out.txt under $WORK/runs yet"; exit 1; }
  fi
  echo "watching: $LATEST   (Ctrl-C to stop; agent run continues)" >&2
  exec python3 "$REPO/scripts/watch_run.py" "$LATEST" "$@"
}

case "$CMD" in
  bootstrap) bootstrap ;;
  run) shift; run "$@" ;;
  dashboard) dashboard ;;
  watch) shift; watch "$@" ;;
  *) echo "usage: $0 bootstrap | run [task=aime2025] [model=Qwen/Qwen3-4B-Base] [hours=10] | dashboard | watch [<log_path>]" ;;
esac
