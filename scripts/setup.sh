#!/usr/bin/env bash
# Interactive one-shot setup for the evo PostTrainBench run on a fresh rented H100
# (JarvisLabs/RunPod). Idempotent -- safe to re-run after a pause (which wipes
# /root + system installs but keeps /home). Reads input from the terminal, so it
# works under `curl ... | bash` too.
set -euo pipefail
TTY=/dev/tty
info() { printf '\n\033[1m%s\033[0m\n' "$*"; }

info "evo-posttrainbench setup"

# 0. preflight: this needs an NVIDIA GPU host -- bail early, before installing a GPU stack
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L 2>/dev/null | grep -q 'GPU'; then
  echo "ERROR: no NVIDIA GPU found (nvidia-smi missing or no device). Run this on a GPU host." >&2
  exit 1
fi

# 1. persistent workspace
read -rp "Persistent workspace dir [/home/ptb]: " WORK <"$TTY"; WORK="${WORK:-/home/ptb}"
mkdir -p "$WORK"; export WORK

# 2. repo (use current checkout if we're in it, else clone into the workspace)
if [ -f scripts/run_evo_jarvislabs.sh ]; then
  REPO="$(pwd)"
else
  REPO="$WORK/repo"
  [ -d "$REPO/.git" ] || git clone https://github.com/evo-hq/evo-posttrainbench.git "$REPO"
fi
cd "$REPO"
info "repo: $REPO   workspace: $WORK"

# 3. install (deps, vLLM, inspect_evals, evo from feat/model-update, claude-code, socat)
read -rp "Run install/bootstrap now? [Y/n]: " yn <"$TTY"
[ "${yn:-Y}" = "n" ] || WORK="$WORK" bash scripts/run_evo_jarvislabs.sh bootstrap

# 4. Claude auth (idempotent)
touch "$WORK/.env"; chmod 600 "$WORK/.env"
if [ -f "$WORK/oauth_token" ] || grep -q '^ANTHROPIC_API_KEY=' "$WORK/.env"; then
  info "Claude auth already configured -- skipping"
else
  info "Claude auth: [1] OAuth / Max subscription   [2] API key"
  read -rp "choose [1]: " a <"$TTY"
  if [ "${a:-1}" = "2" ]; then
    read -rsp "paste ANTHROPIC_API_KEY: " k <"$TTY"; echo
    echo "ANTHROPIC_API_KEY=$k" >> "$WORK/.env"
  else
    echo "  (run 'claude setup-token' on your laptop first)"
    read -rsp "paste the OAuth token: " t <"$TTY"; echo
    printf '%s' "$t" > "$WORK/oauth_token"; chmod 600 "$WORK/oauth_token"
  fi
fi

# 5. other secrets (idempotent)
grep -q '^HF_TOKEN=' "$WORK/.env" || { read -rsp "HF_TOKEN (needed for gemma-3-4b-pt; Enter to skip): " h <"$TTY"; echo; [ -n "${h:-}" ] && echo "HF_TOKEN=$h" >> "$WORK/.env"; }
grep -q '^WANDB_API_KEY=' "$WORK/.env" || { read -rsp "WANDB_API_KEY (optional; Enter to skip): " w <"$TTY"; echo; [ -n "${w:-}" ] && echo "WANDB_API_KEY=$w" >> "$WORK/.env"; }

# 6. sanity
info "Sanity check"
nvidia-smi -L || true
python -c "import torch, vllm, trl, inspect_evals.aime2025; print('deps ok:', torch.cuda.get_device_name(0))" \
  || echo "WARN: deps not importable yet -- re-run with install."

# 7. next steps
info "Setup done. Run it (inside tmux):"
cat <<EOF
  tmux new -s ptb
  WORK=$WORK bash scripts/run_evo_jarvislabs.sh run aime2025 Qwen/Qwen3-4B-Base 1     # 1h smoke
  WORK=$WORK bash scripts/run_evo_jarvislabs.sh run aime2025 Qwen/Qwen3-4B-Base 10
  WORK=$WORK bash scripts/run_evo_jarvislabs.sh run aime2025 google/gemma-3-4b-pt 10
  WORK=$WORK bash scripts/run_evo_jarvislabs.sh dashboard    # then open port 8090 on the instance
EOF
