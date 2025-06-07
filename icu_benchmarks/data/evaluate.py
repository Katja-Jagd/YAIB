import json
from pathlib import Path
import numpy as np
import argparse

def summarize_metrics(repetition_path: Path):
    # Collect metrics from all folds
    metrics_list = []

    for i in range(5):  # folds 0 to 4
        fold_path = repetition_path / f"fold_{i}" / "test_metrics.json"
        if not fold_path.exists():
            print(f"Warning: Missing file {fold_path}")
            continue
        
        with open(fold_path, "r") as f:
            metrics = json.load(f)
            metrics_list.append(metrics)

    # Check if we got enough data
    if len(metrics_list) < 1:
        raise ValueError("No test_metrics.json files found in any folds.")

    # Convert to arrays
    losses = np.array([m["loss"] for m in metrics_list])
    aucs = np.array([m["AUC"] for m in metrics_list])
    prs = np.array([m["PR"] for m in metrics_list])

    # Compute summary
    summary = {
        "loss": {"mean": float(np.mean(losses)), "std": float(np.std(losses, ddof=1))},
        "AUC": {"mean": float(np.mean(aucs)), "std": float(np.std(aucs, ddof=1))},
        "PR": {"mean": float(np.mean(prs)), "std": float(np.std(prs, ddof=1))}
    }

    # Save summary to mean_result.json
    output_path = repetition_path / "mean_result.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"✅ Saved mean results to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize test metrics across folds and save to JSON.")
    parser.add_argument("repetition_path", type=str, help="Path to the repetition_0/ directory containing fold subfolders")

    args = parser.parse_args()
    repetition_dir = Path(args.repetition_path)

    if not repetition_dir.exists():
        raise FileNotFoundError(f"Provided path does not exist: {repetition_dir}")

    summarize_metrics(repetition_dir)

