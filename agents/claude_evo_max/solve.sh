#!/bin/bash
# Claude Code + evo plugin (finetuning skill), Max subscription (OAuth, non-API),
# Opus 4.6 at effort=max. Mirrors agents/claude_non_api_max; evo is engaged via the
# installed plugin plus the evo-engagement preamble the runner prepends to $PROMPT.
unset GEMINI_API_KEY
unset CODEX_API_KEY

# Auth: prefer OAuth / Max subscription if a token is present, else fall back to
# an API key. (OAuth token via env, or a file -- default path matches the
# apptainer harness convention.)
TOKEN_FILE="${OAUTH_TOKEN_FILE:-/home/ben/oauth_token}"
if [ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ] || [ -f "$TOKEN_FILE" ]; then
    [ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ] || export CLAUDE_CODE_OAUTH_TOKEN="$(cat "$TOKEN_FILE")"
    export ANTHROPIC_API_KEY=""          # use the subscription path
elif [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    :                                    # use the API key as-is
else
    echo "ERROR: no Claude auth -- set an OAuth token ($TOKEN_FILE or CLAUDE_CODE_OAUTH_TOKEN) or ANTHROPIC_API_KEY" >&2
    exit 1
fi

export BASH_MAX_TIMEOUT_MS="36000000"
export CLAUDE_CODE_EFFORT_LEVEL="max"   # Opus 4.6 only

# Run claude with NO controlling tty. Background:
# When solve.sh is invoked from a tmux pane (e.g. JarvisLabs, manual local
# tmux), the pane has a controlling pty. `claude --print --verbose` plus the
# tee pipeline in run.sh creates a multi-stage process group where claude's
# foreground status flips during pipe setup. Once claude is in the background
# of that pty, any stdout write triggers SIGTTOU and the process gets STOPPED
# (state `T` in /proc/<pid>/status) -- silently, with no error -- and the run
# hangs forever.
# On Modal this never reproduced because subprocess.run gives the container
# no tty at all. The fix makes solve.sh portable across both:
#   - `setsid -w` creates a new session with NO controlling terminal; claude
#     can never bump into a tty regardless of how the parent set things up.
#   - `< /dev/null` closes stdin defensively; prevents any read-from-tty path.
# Output still flows through stdout to the caller's tee pipeline as normal.
exec setsid -w claude --print --verbose --model "$AGENT_CONFIG" \
    --output-format stream-json --dangerously-skip-permissions "$PROMPT" \
    < /dev/null
