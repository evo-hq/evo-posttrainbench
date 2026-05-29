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

claude --print --verbose --model "$AGENT_CONFIG" --output-format stream-json \
    --dangerously-skip-permissions "$PROMPT"
