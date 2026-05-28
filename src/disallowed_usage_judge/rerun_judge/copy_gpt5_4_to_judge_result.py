#!/usr/bin/env python3
"""
Copy judgement_gpt5_4_rerun.json over judge_result.json for every sub-run
directory found under the results root.

WARNING: the two files have different schemas:
    judgement_gpt5_4_rerun.json -> contamination, disallowed_model,
                                   justification_contamination,
                                   justification_disallowed_model
    judge_result.json (current) -> the above PLUS disallowed_api_usage and
                                   justification_disallowed_api_usage
                                   (older files use the legacy
                                   contamination_detected/disallowed_model_detected
                                   keys)

Overwriting drops the API-usage keys and any consumer that reads them will
break. Defaults to --dry-run; pass --apply to actually overwrite.

Reads POST_TRAIN_BENCH_RESULTS_DIR from the repo's .env file. The .env value
wins over any value already in the shell environment, and a missing .env is a
hard error. Only --root on the command line beats .env.

Usage:
    python copy_gpt5_4_to_judge_result.py                  # dry run, root from .env
    python copy_gpt5_4_to_judge_result.py --apply          # actually overwrite
    python copy_gpt5_4_to_judge_result.py --root <dir>     # override results root
    python copy_gpt5_4_to_judge_result.py --env-file <p>   # override .env path
"""

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

SUB_RUN_RE = re.compile(r"^(?P<bench>.+?)_(?P<model>.+)_(?P<cid>\d+)$")
SRC_NAME = "judgement_gpt5_4_rerun.json"
DST_NAME = "judge_result.json"
REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = REPO_ROOT / ".env"


def load_env_file(path: Path) -> None:
    """Load KEY=VALUE lines from `path` into os.environ, overriding any
    existing values. A missing .env file is a hard error."""
    if not path.is_file():
        sys.exit(f"error: .env file not found at {path}\nCopy example.env to .env and fill in your values.")
    with path.open() as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            if key:
                os.environ[key] = value


def iter_sub_runs(root):
    for exp in sorted(os.listdir(root)):
        exp_path = os.path.join(root, exp)
        if not os.path.isdir(exp_path):
            continue
        for sub in sorted(os.listdir(exp_path)):
            sub_path = os.path.join(exp_path, sub)
            if not os.path.isdir(sub_path):
                continue
            if SUB_RUN_RE.match(sub):
                yield exp, sub, sub_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=None, help="Results root (default: POST_TRAIN_BENCH_RESULTS_DIR from .env)")
    parser.add_argument("--env-file", default=str(ENV_FILE), help=f"Path to .env file (default: {ENV_FILE})")
    parser.add_argument("--apply", action="store_true", help="Actually overwrite files (otherwise dry-run)")
    args = parser.parse_args()

    load_env_file(Path(args.env_file))

    root = args.root or os.environ.get("POST_TRAIN_BENCH_RESULTS_DIR")
    if not root:
        sys.exit(f"error: --root not given and POST_TRAIN_BENCH_RESULTS_DIR not set in {args.env_file}")
    if not os.path.isdir(root):
        sys.exit(f"error: root is not a directory: {root}")
    args.root = root

    will_copy = []
    will_create = []
    skipped_no_src = []

    for exp, sub, sub_path in iter_sub_runs(args.root):
        src = os.path.join(sub_path, SRC_NAME)
        dst = os.path.join(sub_path, DST_NAME)
        if not os.path.exists(src):
            skipped_no_src.append((exp, sub))
            continue
        if os.path.exists(dst):
            will_copy.append((exp, sub, src, dst))
        else:
            will_create.append((exp, sub, src, dst))

    print(f"Root:                       {args.root}")
    print(f"Sub-runs with {SRC_NAME}: {len(will_copy) + len(will_create)}")
    print(f"  -> would overwrite {DST_NAME}: {len(will_copy)}")
    print(f"  -> would create    {DST_NAME}: {len(will_create)}")
    print(f"Sub-runs missing {SRC_NAME}: {len(skipped_no_src)}")

    if not args.apply:
        print("\nDry-run only. Re-run with --apply to perform the copies.")
        return

    print("\nApplying copies...")
    n_done = 0
    n_failed = 0
    for exp, sub, src, dst in will_copy + will_create:
        try:
            shutil.copyfile(src, dst)
            n_done += 1
        except Exception as e:
            n_failed += 1
            print(f"  FAILED: {exp}/{sub}: {e}", file=sys.stderr)
    print(f"Done. Copied: {n_done}, failed: {n_failed}")


if __name__ == "__main__":
    main()
