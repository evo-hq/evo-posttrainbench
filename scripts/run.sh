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
  BASE=$(python src/eval/general/get_prompt.py --model-to-train "$MODEL" --benchmark-id "$TASK" --num-hours "$HOURS" --num-gpus 1 --agent "$AGENT")
  BENCH_NAME=$(tr -d '\n' < "src/eval/tasks/$TASK/benchmark.txt")
  cat > "$RUN/prompt.txt" <<'EOF'
Goal: take a baseline measurement of __MODEL__ on __BENCH__, then improve that score via post-training. Concretely, you will use post-training techniques (SFT, DPO/KTO/ORPO, RFT, GRPO/PPO/RLOO -- pick by reward shape) applied to __MODEL__'s weights to produce a final_model/ that beats the base model on __BENCH__.

STEP 0 -- internalize every evo skill before doing anything else.

The evo plugin exposes the following skills. Invoke each ONE BY ONE via the Skill tool with no args, read the full body (not just the description), and let the content inform every subsequent decision. Do this BEFORE evo init, before any bash beyond reading files, before any planning, before writing any code.

- evo:discover    -- baseline + gates setup; how to construct an experiment
- evo:optimize    -- the improvement loop after baseline
- evo:finetuning  -- which post-training technique fits which reward shape, what never counts as progress, and diagnostics for when an approach is exhausted. LOAD THIS BEFORE WRITING ANY TRAINING CODE -- the reward-shape decision tree decides whether you should be doing SFT or RL on this benchmark, and the answer is not always SFT.
- evo:ideator     -- proposing what to try next
- evo:subagent    -- spawning parallel experiments
- evo:verifier    -- catching false-progress (held-out leakage, format mismatch, etc.)
- evo:report      -- summarizing runs
- evo:infra-setup -- backend choices

Only after you have invoked and read all eight skills do you proceed below.

STEP 1 -- invoke evo:discover, seeded with: "improve __MODEL__ on __BENCH__ via post-training; the benchmark is ./evaluate.py (already provided -- do not modify per rule 4); curate training data from public sources only, NEVER __BENCH__ test data (per rule 3); only fine-tune __MODEL__ (per rule 7); final_model must be the best gate-passing checkpoint." Discover constructs the baseline + gates and commits the first experiment (the baseline-untrained score) before any post-training begins. Do not skip this commit -- it is your comparison point for every subsequent experiment.

STEP 2 -- invoke evo:optimize. The optimize loop drives all post-training experiments after baseline. Per-experiment decisions on technique + hyperparameters come from evo:finetuning (which you just read), not from your own priors.

WORKFLOW: train first, benchmark second. The two are separate steps.

  1. You make changes -- data curation, hyperparameter selection, technique choice, training code edits.
  2. You run training/finetuning to produce a checkpoint at <worktree>/final_model/ (or the path evo:finetuning's references/glue.md specifies for your technique).
  3. ONLY THEN do you run the benchmark to score the trained model.

The evo benchmark command must be eval-only -- it loads the checkpoint at final_model/ (or the relevant artifact path) and scores it. Do NOT wrap train + eval into a single benchmark command. If you do, every gate re-validation and every re-score retrains from scratch, and you burn the 10h budget on duplicated training instead of new experiments. The benchmark you register in evo init should call evaluate.py (or a thin wrapper that just evals), NOT train.py.

Training observability: trackio is installed and ships logs to a HuggingFace Space. Trackio is wandb-API-compatible but does NOT register itself as the `wandb` module, so `report_to="wandb"` in TrainingArguments does nothing. Use the bundled callback instead:

    from scripts.trl_trackio_callback import TrackioCallback
    trainer = SFTTrainer(..., args=SFTConfig(..., report_to="none"))
    trainer.add_callback(TrackioCallback(project="ptb", run_name="<exp_id>"))
    trainer.train()

Leave `report_to="none"` in the config. The callback handles trackio.init(), log forwarding on every Trainer log event, and finish() on train_end. TRACKIO_SPACE_ID and HF_TOKEN env vars are already set; the callback validates them and fails loudly if either is missing.

final_model/ at the end must be evo's best gate-passing checkpoint. Obey every PostTrainBench rule below.

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
