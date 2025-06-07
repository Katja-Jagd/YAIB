import argparse
import json
import re
from pathlib import Path
import numpy as np
from collections import defaultdict

def summarize_across_folds(repetition_path: Path):
    """Summarize test metrics from fold_0 to fold_4 and save to mean_result.json."""
    metrics_list = []

    for i in range(5):
        fold_path = repetition_path / f"fold_{i}" / "test_metrics.json"
        if not fold_path.exists():
            print(f"⚠️  Missing: {fold_path}")
            continue
        with open(fold_path, "r") as f:
            metrics = json.load(f)
            metrics_list.append(metrics)

    if len(metrics_list) == 0:
        print("❌ No metrics found across folds.")
        return

    losses = np.array([m["loss"] for m in metrics_list])
    aucs = np.array([m["AUC"] for m in metrics_list])
    prs = np.array([m["PR"] for m in metrics_list])

    summary = {
        "loss": {"mean": float(np.mean(losses)), "std": float(np.std(losses, ddof=1))},
        "AUC": {"mean": float(np.mean(aucs)), "std": float(np.std(aucs, ddof=1))},
        "PR": {"mean": float(np.mean(prs)), "std": float(np.std(prs, ddof=1))}
    }

    output_path = repetition_path / "mean_result.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"✅ Saved cross-fold mean results to: {output_path}")

def extract_log_dataset_pairs(err_path: Path):
    """Extract (log_path, dataset) pairs from the SLURM .err file."""
    log_pattern = re.compile(r"Logging to (\/[\w\/\.-]+)")
    data_pattern = re.compile(r"Loading cached data from .*?/mortality24/(\w+)/cache/")

    log_dataset_pairs = []
    current_log_path = None

    with err_path.open("r") as f:
        lines = f.readlines()

    for line in lines:
        log_match = log_pattern.search(line)
        if log_match:
            current_log_path = log_match.group(1)

        data_match = data_pattern.search(line)
        if data_match and current_log_path:
            dataset = data_match.group(1)
            log_dataset_pairs.append((current_log_path, dataset))
            current_log_path = None

    return log_dataset_pairs

def summarize_by_dataset(log_dataset_pairs, output_dir: Path):
    """Group test metrics by dataset (from fold_0 only) and save per-dataset summaries."""
    dataset_metrics = defaultdict(list)

    for log_path, dataset in log_dataset_pairs:
        metrics_path = Path(log_path) / "repetition_0" / "fold_0" / "test_metrics.json"
        if not metrics_path.exists():
            print(f"⚠️  Missing: {metrics_path}")
            continue
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
                dataset_metrics[dataset].append(metrics)
        except Exception as e:
            print(f"❌ Failed to read {metrics_path}: {e}")

    for dataset, metrics_list in dataset_metrics.items():
        if len(metrics_list) == 0:
            print(f"⚠️  No metrics found for dataset {dataset}")
            continue

        losses = np.array([m["loss"] for m in metrics_list])
        aucs = np.array([m["AUC"] for m in metrics_list])
        prs = np.array([m["PR"] for m in metrics_list])

        summary = {
            "loss": {"mean": float(np.mean(losses)), "std": float(np.std(losses, ddof=1))},
            "AUC": {"mean": float(np.mean(aucs)), "std": float(np.std(aucs, ddof=1))},
            "PR": {"mean": float(np.mean(prs)), "std": float(np.std(prs, ddof=1))}
        }

        output_file = output_dir / f"{dataset}_mean_results.json"
        try:
            with open(output_file, "w") as f:
                json.dump(summary, f, indent=4)
            print(f"✅ Saved {dataset} summary to: {output_file}")
        except Exception as e:
            print(f"❌ Failed to save {dataset} summary: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize test metrics from folds and error logs.")
    parser.add_argument("err_file", type=str, help="Path to .err file (for dataset-level summaries)")
    parser.add_argument("weights_path", type=str, help="Path to repetition_0 directory (for cross-fold summary)")

    args = parser.parse_args()

    err_path = Path(args.err_file)
    weights_path = Path(args.weights_path)

    if not err_path.exists():
        raise FileNotFoundError(f"❌ Error file not found: {err_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"❌ Weights path not found: {weights_path}")

    # Part 1: summarize across fold_0 to fold_4 in weights_path
    summarize_across_folds(weights_path)

    # Part 2: summarize all fold_0 test metrics from the err file, grouped by dataset
    log_dataset_pairs = extract_log_dataset_pairs(err_path)
    summarize_by_dataset(log_dataset_pairs, weights_path)

