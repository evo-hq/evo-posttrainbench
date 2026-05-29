#!/bin/bash
# Claude Code + evo plugin (finetuning skill), Max subscription (OAuth, non-API),
# Opus 4.6 at effort=max. Mirrors agents/claude_non_api_max; evo is engaged via the
# installed plugin plus the evo-engagement preamble the runner prepends to $PROMPT.
unset GEMINI_API_KEY
unset CODEX_API_KEY
export ANTHROPIC_API_KEY=""   # force the OAuth/subscription path

# OAuth token: use it if the runner already exported it, else read the file
# (default path matches the apptainer harness convention).
if [ -z "${CLAUDE_CODE_OAUTH_TOKEN}" ]; then
    TOKEN_FILE="${OAUTH_TOKEN_FILE:-/home/ben/oauth_token}"
    if [ -f "$TOKEN_FILE" ]; then
        export CLAUDE_CODE_OAUTH_TOKEN="$(cat "$TOKEN_FILE")"
    else
        echo "ERROR: no OAuth token (set CLAUDE_CODE_OAUTH_TOKEN or place a file at $TOKEN_FILE)" >&2
        exit 1
    fi
fi

export BASH_MAX_TIMEOUT_MS="36000000"
export CLAUDE_CODE_EFFORT_LEVEL="max"   # Opus 4.6 only

claude --print --verbose --model "$AGENT_CONFIG" --output-format stream-json \
    --dangerously-skip-permissions "$PROMPT"
