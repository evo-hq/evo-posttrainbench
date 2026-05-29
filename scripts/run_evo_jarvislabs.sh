#!/bin/bash
# Apptainer-free PostTrainBench runner for a single rented H100 (e.g. JarvisLabs),
# running the evo variant: Claude Code + evo plugin + finetuning skill, on OAuth.
#
# Their src/run_task.sh wraps everything in apptainer .sif images, which is painful
# inside an already-containerized cloud GPU. This runs the same core logic directly
# on the host. It deliberately does NOT replicate the contamination judge or the
# fuse-overlayfs HF isolation -- for our own experiment we run the agent and
# evaluate.py directly. SMOKE-TEST one cell before trusting any number.
#
# Layout: everything lives under $WORK (default /home/<user>/ptb) so it survives a
# JarvisLabs pause (/home is persistent; /root is wiped).
set -euo pipefail

CMD="${1:-help}"
WORK="${WORK:-/home/$(whoami)/ptb}"
REPO="${REPO:-$(pwd)}"                 # this PostTrainBench-evo checkout
EVO_BRANCH="${EVO_BRANCH:-feat/model-update}"
export HF_HOME="${HF_HOME:-$WORK/hf}"
export CLAUDE_CONFIG_DIR="${CLAUDE_CONFIG_DIR:-$WORK/.claude}"

bootstrap() {
  mkdir -p "$WORK" "$HF_HOME" "$CLAUDE_CONFIG_DIR" "$WORK/runs"
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  command -v node >/dev/null || { curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && apt-get install -y nodejs; }
  command -v socat >/dev/null || apt-get install -y socat   # to expose the dashboard port
  npm install -g @anthropic-ai/claude-code@2.1.76          # match the version they ran

  # PostTrainBench starting environment (pinned) + vLLM + flash-attn
  uv pip install --system --no-cache vllm==0.11.0 --torch-backend=auto
  uv pip install --system --no-cache -r "$REPO/containers/requirements-direct.txt"
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
  local TASK="${1:-aime2025}"
  local MODEL="${2:-Qwen/Qwen3-4B-Base}"
  local HOURS="${3:-10}"
  local AGENT="claude_evo_max"
  local AGENT_CONFIG="${AGENT_CONFIG:-claude-opus-4-6}"

  [ -f "$WORK/.env" ] && { set -a; source "$WORK/.env"; set +a; }
  export OAUTH_TOKEN_FILE="$WORK/oauth_token"
  [ -f "$OAUTH_TOKEN_FILE" ] || { echo "missing $OAUTH_TOKEN_FILE (run: claude setup-token)"; exit 1; }

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
  EVO_PRE="Use evo to structure this work: initialise evo here, treat evaluate.py as the benchmark/gate, and run the optimize loop -- propose post-training experiments, score each on a held-out split you carve from training data (NEVER the test set), and keep what improves. Load the 'finetuning' skill for method and diagnostics judgment; take the LOCAL training path (this box's TRL/PEFT + vLLM serving) since no managed service is available. final_model is evo's best gate-passing checkpoint. Obey every rule below.

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
  # evo's dashboard binds 127.0.0.1 only; bridge it to 0.0.0.0 so the port can be
  # opened on the cloud instance. Open PUBLIC_PORT on JarvisLabs to reach it.
  local PUBLIC="${PUBLIC_PORT:-8090}" INTERNAL="${EVO_DASH_PORT:-8080}"
  command -v socat >/dev/null || { echo "socat missing -- run bootstrap"; exit 1; }
  echo "exposing evo dashboard 127.0.0.1:$INTERNAL -> 0.0.0.0:$PUBLIC (open port $PUBLIC on the instance)"
  exec socat "TCP-LISTEN:${PUBLIC},fork,reuseaddr" "TCP:127.0.0.1:${INTERNAL}"
}

case "$CMD" in
  bootstrap) bootstrap ;;
  run) shift; run "$@" ;;
  dashboard) dashboard ;;
  *) echo "usage: $0 bootstrap | run [task=aime2025] [model=Qwen/Qwen3-4B-Base] [hours=10] | dashboard" ;;
esac
