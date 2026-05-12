#!/usr/bin/env python3
"""Aggregate per-judge judgement.json files into a single judge_result.json.

Each per-judge file must contain:
  - contamination (bool)
  - disallowed_model (bool)
  - justification_contamination (string)
  - justification_disallowed_model (string)

The aggregated output uses the same schema:
  - contamination: True if ANY judge flagged contamination
  - disallowed_model: True if ANY judge flagged disallowed model use
  - justification_*: per-judge justifications concatenated with a `[judge_name]` tag

If a per-judge file is missing or cannot be parsed it is recorded with a
placeholder justification and contributes False to the boolean aggregation.

Usage:
    aggregate_judgement.py --output judge_result.json \
        --judge gpt5_4=judgement_gpt5_4.json \
        --judge sonnet4_6=judgement_sonnet4_6.json
"""

import argparse
import json
from pathlib import Path


REQUIRED_FIELDS = (
    "contamination",
    "disallowed_model",
    "justification_contamination",
    "justification_disallowed_model",
)


def read_judgement(path: Path) -> tuple[dict | None, str | None]:
    """Return (judgement_dict, error_message). Exactly one is non-None."""
    if not path.exists():
        return None, f"file not found: {path.name}"
    raw = path.read_text()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON in {path.name}: {exc}"
    if not isinstance(data, dict):
        return None, f"{path.name} is not a JSON object"
    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        return None, f"{path.name} missing fields: {', '.join(missing)}"
    return data, None


def aggregate(judgements: dict[str, tuple[dict | None, str | None]]) -> dict:
    contamination = any(
        bool(j["contamination"]) for j, _ in judgements.values() if j is not None
    )
    disallowed_model = any(
        bool(j["disallowed_model"]) for j, _ in judgements.values() if j is not None
    )

    def collect(field: str) -> str:
        parts = []
        for name, (data, err) in judgements.items():
            if data is None:
                parts.append(f"[{name}] ERROR: {err}")
            else:
                parts.append(f"[{name}] {data[field]}")
        return "\n\n".join(parts)

    return {
        "contamination": contamination,
        "disallowed_model": disallowed_model,
        "justification_contamination": collect("justification_contamination"),
        "justification_disallowed_model": collect("justification_disallowed_model"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--judge",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Per-judge JSON file, e.g. gpt5_4=judgement_gpt5_4.json. May be repeated.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the aggregated judge_result.json",
    )
    args = parser.parse_args()

    judgements: dict[str, tuple[dict | None, str | None]] = {}
    for spec in args.judge:
        if "=" not in spec:
            raise SystemExit(f"--judge value must be NAME=PATH, got: {spec!r}")
        name, path_str = spec.split("=", 1)
        judgements[name] = read_judgement(Path(path_str))

    result = aggregate(judgements)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"Aggregated {len(judgements)} judge(s) -> {args.output} "
        f"(contamination={result['contamination']}, "
        f"disallowed_model={result['disallowed_model']})"
    )


if __name__ == "__main__":
    main()
