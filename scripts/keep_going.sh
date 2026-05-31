#!/bin/bash
# Claude Code Stop hook: don't let the agent declare itself "done" before the
# time budget is meaningfully spent OR a final_model exists. PostTrainBench
# gives a 10h H100 budget; the agent's tendency is to wrap up after one or two
# experiments. This enforces using the budget for iteration.
#
# Conventions:
#   * runs with cwd = CLAUDE_PROJECT_DIR (the task dir).
#   * relies on PostTrainBench's `timer.sh` (HH:MM:SS remaining).
#   * exit 2 + stderr  = block stop, stderr becomes the model-visible reason.
#   * exit 0           = allow stop.
set -uo pipefail

TIMER="./timer.sh"
FINAL_MODEL="./final_model"
MIN_SECONDS_LEFT="${KEEP_GOING_MIN_SECONDS:-300}"   # allow stop only when <5 min remain

remaining_seconds() {
  [ -x "$TIMER" ] || { echo 0; return; }
  local hms H M S
  hms=$(bash "$TIMER" 2>/dev/null | head -1 | tr -dc '0-9:')
  IFS=: read -r H M S <<<"$hms"
  echo $(( 10#${H:-0} * 3600 + 10#${M:-0} * 60 + 10#${S:-0} ))
}

LEFT=$(remaining_seconds)
HAS_MODEL=0
[ -d "$FINAL_MODEL" ] && [ -n "$(ls -A "$FINAL_MODEL" 2>/dev/null)" ] && HAS_MODEL=1

if [ "$LEFT" -gt "$MIN_SECONDS_LEFT" ]; then
  printf 'You still have %02d:%02d:%02d of your time budget. Do not stop -- propose another experiment via evo (a new method, different data slice, hyperparameter change), train, score on your held-out split, keep what improves. Only stop when timer.sh shows under %ds OR you have a non-empty final_model/ AND further attempts have plateaued on the held-out score.\n' \
    "$((LEFT/3600))" "$(((LEFT%3600)/60))" "$((LEFT%60))" "$MIN_SECONDS_LEFT" >&2
  exit 2
fi

if [ "$HAS_MODEL" -eq 0 ] && [ "$LEFT" -gt 60 ]; then
  echo "No final_model/ produced yet -- per rule 7 you must submit a fine-tune of the provided base as final_model. Keep iterating until you have one." >&2
  exit 2
fi

exit 0
