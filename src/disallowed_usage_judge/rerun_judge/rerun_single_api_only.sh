#!/bin/bash
#
# Rerun only the GPT-5.4 third-party API usage judge on a single result
# directory. Thin wrapper around run_judge.sh --api-only (which writes
# judgement_api_rerun.json + judge_output_api_rerun.{json,txt} and leaves
# the contamination judges' outputs untouched).
#
# Usage: rerun_single_api_only.sh <result_dir>

set -e

RESULT_DIR="$1"

if [ -z "$RESULT_DIR" ]; then
    echo "Usage: $0 <result_dir>" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "$SCRIPT_DIR/../run_judge.sh" --api-only "$RESULT_DIR"
