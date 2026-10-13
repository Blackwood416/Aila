#!/usr/bin/env python3
"""Build an A/B receipt from run logs (Strata C8 methodology package).

Reads the JSON sample lines emitted by the *runner scripts (one JSON object
per line, e.g. {"label": "context-14000", "first_content_seconds": ...}) or
plain JSON result files, extracts one metric per label, and writes a receipt
with per-arm medians, p10/p90, pair counts ("which arm won each pair") and the
A/A noise floor when both arms are the same configuration.

Usage:
  python tools/perf/ab_receipt.py --metric first_content_seconds \
      --arm a=tmp/run-a-run.log --arm b=tmp/run-b-run.log \
      --label context-14000 --out tmp/receipt.json [--higher-is-better]
"""

import argparse
import json
from pathlib import Path
import statistics
import subprocess


def load_metric(path, label, metric):
    values = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("label") != label or metric not in obj:
            continue
        values.append(float(obj[metric]))
    return values


def git_state():
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"],
                                    capture_output=True, text=True).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--arm", action="append", required=True, help="name=log_path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--higher-is-better", action="store_true")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    arms = {}
    for spec in args.arm:
        name, path = spec.split("=", 1)
        arms[name] = load_metric(path, args.label, args.metric)
    for name, values in arms.items():
        assert values, f"arm {name}: no {args.metric} samples for label {args.label}"

    sign = -1.0 if args.higher_is_better else 1.0
    names = list(arms)
    summary = {}
    for name, values in arms.items():
        values_sorted = sorted(values)
        summary[name] = {
            "n": len(values),
            "median": statistics.median(values),
            "min": values_sorted[0],
            "max": values_sorted[-1],
            "p10": values_sorted[max(0, int(0.1 * (len(values_sorted) - 1)))],
            "p90": values_sorted[int(0.9 * (len(values_sorted) - 1))],
            "samples": values,
        }

    pairs = []
    if len(names) == 2:
        a, b = names
        for x, y in zip(arms[a], arms[b]):
            winner = a if sign * x < sign * y else b
            pairs.append({"a": x, "b": y, "winner": winner})
    faster = {name: sum(1 for p in pairs if p["winner"] == name) for name in names}
    ratio = (summary[names[0]]["median"] / summary[names[1]]["median"]
             if len(names) == 2 else None)

    receipt = {
        "schema": "aila-bench-receipt/1",
        "metric": args.metric,
        "label": args.label,
        "higher_is_better": bool(args.higher_is_better),
        "git": git_state(),
        "arms": summary,
        "paired": pairs,
        "pairs_won": faster,
        "median_ratio_a_over_b": ratio,
        "note": args.note,
    }
    Path(args.out).write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"receipt written: {args.out}")
    for name in names:
        s = summary[name]
        print(f"  {name}: n={s['n']} median={s['median']:.4f} "
              f"p10={s['p10']:.4f} p90={s['p90']:.4f}")
    if ratio is not None:
        print(f"  median ratio {names[0]}/{names[1]} = {ratio:.4f} "
              f"pairs won {faster}")


if __name__ == "__main__":
    main()
