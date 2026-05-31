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
REPO="${REPO:-$(pwd)}"                 # this PostTrainBench-evo checkout
EVO_BRANCH="${EVO_BRANCH:-feat/model-update}"
export HF_HOME="${HF_HOME:-$WORK/hf}"
export CLAUDE_CONFIG_DIR="${CLAUDE_CONFIG_DIR:-$WORK/.claude}"

require_gpu() {
  if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L 2>/dev/null | grep -q 'GPU'; then
    echo "ERROR: no NVIDIA GPU found (need a GPU host)." >&2; exit 1
  fi
}

bootstrap() {
  require_gpu
  mkdir -p "$WORK" "$HF_HOME" "$CLAUDE_CONFIG_DIR" "$WORK/runs"
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  command -v node >/dev/null || { curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && apt-get install -y nodejs; }
  npm install -g @anthropic-ai/claude-code@2.1.76          # match the version they ran

  # PostTrainBench starting environment (pinned) + vLLM + flash-attn
  # Pin to cu128: --torch-backend=auto fails during builds with no GPU attached
  # (e.g. Modal image builds); cu128 wheels are compatible with our cuda:12.9.1 base.
  uv pip install --system --no-cache vllm==0.11.0 --torch-backend=cu128
  uv pip install --system --no-cache -r "$REPO/containers/requirements-direct.txt"
  # trackio: wandb-API-compatible OSS tracker; logs to a HF Space.
  # Pin <0.10 -- 0.10+ requires gradio 6 + huggingface-hub>=1.0, conflicts with
  # PostTrainBench's pinned transformers 4.57.3 (needs hf-hub<1.0).
  uv pip install --system --no-cache 'trackio<0.10'
  # wheel needed for flash-attn's --no-build-isolation (it doesn't declare wheel as a build dep)
  uv pip install --system --no-cache wheel setuptools
  uv pip install --system --no-cache flash-attn==2.8.3 --no-build-isolation

  # eval deps: inspect_evals registers the task (e.g. inspect_evals/aime2025); the
  # vLLM-stdout inspect_ai fork is what their evaluate.py uses. Pinned to match upstream.
  local INS; INS=$(mktemp -d)
  git clone https://github.com/UKGovernmentBEIS/inspect_evals.git "$INS/inspect_evals" \
    && ( cd "$INS/inspect_evals" && git checkout 06001a83e6d7c709c2ede0570dce7f1031a0bad8 \
         && uv pip install --system --no-cache . )
  git clone https://github.com/rank-and-file/inspect_ai_vllm_stdout.git "$INS/inspect_ai_vllm_stdout" \
    && ( cd "$INS/inspect_ai_vllm_stdout" && uv pip install --system --no-cache . )

  # evo from our branch + register the plugin (incl. the finetuning skill) into Claude Code
  [ -d "$WORK/evo" ] || git clone -b "$EVO_BRANCH" https://github.com/evo-hq/evo.git "$WORK/evo"
  uv tool install --editable "$WORK/evo/plugins/evo"
  evo install claude-code

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
  if [ ! -f "$OAUTH_TOKEN_FILE" ] && [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "no Claude auth: create $OAUTH_TOKEN_FILE (claude setup-token) or set ANTHROPIC_API_KEY in $WORK/.env"; exit 1
  fi

  # Bind evo's auto-started dashboard to 0.0.0.0 so it's reachable on the cloud
  # instance / Modal web URL (requires evo >= the EVO_DASHBOARD_HOST commit).
  export EVO_DASHBOARD_HOST="${EVO_DASHBOARD_HOST:-0.0.0.0}"
  export EVO_DASHBOARD_PORT="${EVO_DASHBOARD_PORT:-8080}"
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

  # prompt = PostTrainBench's standard task prompt + an evo-engagement preamble
  local BASE EVO_PRE PROMPT
  BASE=$(python src/eval/general/get_prompt.py --model-to-train "$MODEL" --benchmark-id "$TASK" --num-hours "$HOURS" --num-gpus 1 --agent "$AGENT")
  EVO_PRE="Use evo to structure this work: initialise evo here, treat evaluate.py as the benchmark/gate, and run the optimize loop -- propose post-training experiments, score each on a held-out split you carve from training data (NEVER the test set), and keep what improves. Load the 'finetuning' skill for method and diagnostics judgment; take the LOCAL training path (this box's TRL/PEFT + vLLM serving) since no managed service is available. For training metrics use trackio (installed; wandb-API-compatible -- 'import trackio as wandb; wandb.init(project=\"ptb\", space_id=os.environ[\"TRACKIO_SPACE_ID\"])'), which logs to a free HF Space; do not use real W&B. final_model is evo's best gate-passing checkpoint. Obey every rule below.

"
  PROMPT="${EVO_PRE}${BASE}"
  printf '%s' "$PROMPT" > "$RUN/prompt.txt"

  # run the agent directly (no apptainer), bounded by the hour budget
  export PROMPT AGENT_CONFIG
  ( cd "$JOB" && timeout --signal=TERM --kill-after=60s "$((HOURS * 60 + 5))m" \
      bash "$REPO/agents/$AGENT/solve.sh" ) > "$RUN/solve_out.txt" 2>&1 || true
  python "agents/$AGENT/human_readable_trace.py" "$RUN/solve_out.txt" -o "$RUN/solve_parsed.txt" || true

  # evaluate final_model (single pass; their harness adds judge + max-token retries)
  if [ -d "$JOB/final_model" ]; then
    ( cd "src/eval/tasks/$TASK" && python evaluate.py \
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

case "$CMD" in
  bootstrap) bootstrap ;;
  run) shift; run "$@" ;;
  dashboard) dashboard ;;
  *) echo "usage: $0 bootstrap | run [task=aime2025] [model=Qwen/Qwen3-4B-Base] [hours=10] | dashboard" ;;
esac
