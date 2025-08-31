#!/usr/bin/env python3
import json
import sys
import statistics
import collections
import argparse
import csv

METRICS = ["test/AUC", "test/PR", "test/loss"]

def summarize(json_path, csv_out=None):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Group records by num_samples
    groups = collections.defaultdict(list)
    for rec in data:
        ns = rec.get("num_samples")
        if ns is not None:
            groups[ns].append(rec)

    rows = []
    for ns, recs in sorted(groups.items()):
        n_group = len(recs)
        if n_group != 5:
            print(f"Warning: num_samples={ns} has {n_group} entries (expected 5).", file=sys.stderr)

        for metric in METRICS:
            vals = [float(r[metric]) for r in recs if metric in r]
            if not vals:
                mean = None
                sd = None
                n = 0
                print(f"Warning: num_samples={ns} missing values for '{metric}'.", file=sys.stderr)
            else:
                mean = statistics.mean(vals)
                sd = statistics.stdev(vals) if len(vals) > 1 else 0.0  # sample SD
                n = len(vals)

            rows.append({
                "num_samples": ns,
                "metric": metric,
                "n": n,
                "mean": mean,
                "sd": sd,
            })

    # Print a simple CSV to stdout
    print("num_samples,metric,n,mean,sd")
    for r in rows:
        mean_str = "" if r["mean"] is None else f"{r['mean']:.6f}"
        sd_str = "" if r["sd"] is None else f"{r['sd']:.6f}"
        print(f"{r['num_samples']},{r['metric']},{r['n']},{mean_str},{sd_str}")

    # Optional: also write to a CSV file if requested
    if csv_out:
        with open(csv_out, "w", newline="", encoding="utf-8") as out:
            w = csv.DictWriter(out, fieldnames=["num_samples", "metric", "n", "mean", "sd"])
            w.writeheader()
            for r in rows:
                w.writerow(r)

def main():
    parser = argparse.ArgumentParser(description="Summarize metrics by num_samples.")
    parser.add_argument("json_path", help="Path to JSON file (list of records).")
    parser.add_argument("--csv", help="Optional: write summary to this CSV file.")
    args = parser.parse_args()
    summarize(args.json_path, args.csv)

if __name__ == "__main__":
    main()

