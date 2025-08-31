import json, statistics, sys
from collections import defaultdict, Counter

META_FIELDS = ["dataset", "batch_size", "lr", "fine_tune_head"]
IGNORE_FIELDS = {"dataset","size","seed","batch_size","lr","num_epochs","fine_tune_head","model_path"}

def fmt(values, ndigits=6):
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{round(mean, ndigits)}±{round(sd, ndigits)}"

def main(inp, out):
    rows = [json.loads(l) for l in open(inp) if l.strip()]

    meta = {}
    for k in META_FIELDS:
        vals = [r.get(k) for r in rows if k in r]
        if vals:
            meta[k] = Counter(vals).most_common(1)[0][0]

    metric_keys = {k for r in rows for k,v in r.items() if k not in IGNORE_FIELDS and isinstance(v,(int,float))}

    by_size = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for m in metric_keys:
            if m in r:
                by_size[r["size"]][m].append(r[m])

    results = {}
    for size in sorted(by_size, key=int):
        results[str(size)] = {m: fmt(by_size[size][m]) for m in metric_keys}

    payload = {
        "dataset": meta.get("dataset"),
        "bz": meta.get("batch_size"),
        "lr": meta.get("lr"),
        "fine_tune_head": meta.get("fine_tune_head"),
        "results": results
    }

    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python summarize_runs.py input.jsonl output.json")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
