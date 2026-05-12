#!/usr/bin/env python3
import csv
import json
import os

OUTPUT_PREFIX = "contamination_"        # e.g. "contamination_method.csv"


def load_judge_result(run_dir: str) -> tuple:
    """Return (contamination, disallowed_model) read from judge_result.json.

    Each value is True/False on success, "ERR" if the file is missing, or
    "IMPORTANT ERR" if the file exists but is malformed.
    """
    path = os.path.join(run_dir, "judge_result.json")
    if not os.path.exists(path):
        return "ERR", "ERR"
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return "IMPORTANT ERR", "IMPORTANT ERR"

    def coerce(field: str):
        if field not in data or not isinstance(data[field], bool):
            return "IMPORTANT ERR"
        return data[field]

    return coerce("contamination"), coerce("disallowed_model")


def combine_results(contamination, disallowed_model):
    """
    Combine contamination and disallowed model results into a single cell value.

    Returns:
        - "" if no illegal use detected (and no contamination)
        - "M" if disallowed model detected (but no contamination)
        - "MC" if disallowed model detected & contamination
        - "C" if only contamination detected (and no disallowed model)
        - Error string if either result is an error
    """
    # Handle error cases first
    if contamination in ("ERR", "IMPORTANT ERR") or disallowed_model in ("ERR", "IMPORTANT ERR"):
        errors = []
        if contamination in ("ERR", "IMPORTANT ERR"):
            errors.append(f"C:{contamination}")
        if disallowed_model in ("ERR", "IMPORTANT ERR"):
            errors.append(f"M:{disallowed_model}")
        return " ".join(errors)

    # Both are boolean now
    if disallowed_model and contamination:
        return "MC"
    elif disallowed_model and not contamination:
        return "M"
    elif not disallowed_model and contamination:
        return "C"
    else:  # not disallowed_model and not contamination
        return ""


def process_method(method_path: str, method_name: str):
    """
    For a single method dir (results/method_name), collect the newest run per
    (benchmark, model), then write a CSV.
    """
    # key: (benchmark, model) -> value: {"run_id": int, "path": str}
    latest_runs = {}

    for entry in os.listdir(method_path):
        entry_path = os.path.join(method_path, entry)
        if not os.path.isdir(entry_path):
            continue

        # Format: {benchmark}_{provider}_{model}_{cluster_id}
        # Split from right to get cluster_id, then from left to get benchmark
        parts = entry.rsplit("_", 1)
        if len(parts) < 2:
            raise ValueError(f"Invalid result directory name: {entry}, {method_path}")
        run_id = parts[1]
        rest = parts[0]

        benchmark_end = rest.find("_")
        if benchmark_end == -1:
            raise ValueError(f"Invalid result directory name: {entry}, {method_path}")
        benchmark = rest[:benchmark_end]
        model = rest[benchmark_end + 1:]  # provider_model (e.g. meta-llama_Llama-3.2-1B)
        key = (benchmark, model)

        # keep only highest run_id per (benchmark, model)
        if key not in latest_runs or run_id > latest_runs[key]["run_id"]:
            latest_runs[key] = {
                "run_id": run_id,
                "path": entry_path,
            }

    if not latest_runs:
        # nothing to do for this method
        return

    # Collect distinct benchmarks and models
    benchmarks = sorted({b for (b, m) in latest_runs.keys()})
    models = sorted({m for (b, m) in latest_runs.keys()})

    # Prepare CSV path (next to results/ or inside results/)
    csv_path = os.path.join(get_results_dir(), f"{OUTPUT_PREFIX}{method_name}.csv")

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # header
        writer.writerow(["model"] + benchmarks)

        # rows
        for model in models:
            row = [model]
            for bench in benchmarks:
                cell = ""
                key = (bench, model)
                if key in latest_runs:
                    run_dir = latest_runs[key]["path"]
                    contamination, disallowed_model = load_judge_result(run_dir)
                    cell = combine_results(contamination, disallowed_model)
                row.append(cell)
            writer.writerow(row)

    print(f"Written: {csv_path}")


def get_results_dir():
    return os.environ.get("POST_TRAIN_BENCH_RESULTS_DIR", 'results')


def main():
    results_dir = get_results_dir()
    for method_name in os.listdir(results_dir):
        method_path = os.path.join(results_dir, method_name)
        if not os.path.isdir(method_path):
            continue
        # treat every subdirectory of results/ as a "method"
        process_method(method_path, method_name)


if __name__ == "__main__":
    main()
