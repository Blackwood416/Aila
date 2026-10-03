"""Causal prompt-only coverage; future decode routes are evaluation labels only.

No latency prediction: rebuilding a tier transfers weights and may displace a
better learned decode set. Keep each request separate, including switches.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def trace_rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def audit(path, profile, capacity=3551):
    lines = Path(profile).read_text().splitlines()
    layers, experts, _ = map(int, lines[2].split())
    ranking = [int(line.split()[0]) for line in lines[3:]]
    baseline = set(ranking[:capacity])
    quotas = Counter(k // experts for k in baseline)
    rank = {k: i for i, k in enumerate(ranking)}
    requests, current = [], None
    for r in trace_rows(path):
        if r["type"] == "forward_begin" and r["position"] == 0:
            current = dict(prefill=[], decode=[])
            requests.append(current)
        if r["type"] == "route" and current is not None:
            current["prefill" if r["rows"] > 1 else "decode"].append(r)
    result = []
    for idx, req in enumerate(requests):
        if not req["prefill"] or not req["decode"]:
            continue
        labels = [[r["layer"] * experts + e for e in r["ids"]] for r in req["decode"]]
        def coverage(keys):
            return {str(n): sum(k in keys for row in labels[:n*layers] for k in row) /
                    sum(len(row) for row in labels[:n*layers]) for n in (8, 48)}
        row = dict(request=idx, tokens=sum(r["rows"] for r in req["prefill"] if r["layer"] == 0),
                   decode_forwards=len(labels)//layers, profile=coverage(baseline), candidates={})
        if row['tokens'] < 4:  # runtime warmup is not an input prompt
            continue
        for half in (32, 128, 0):
            scores = [0.] * (layers * experts)
            for r in req["prefill"]:
                offset = r["layer"] * experts
                decay = math.exp2(-1 / half) if half else 1.
                for k in range(offset, offset+experts):
                    scores[k] *= decay ** r["rows"]
                for i, e in enumerate(r["ids"]):
                    scores[offset+e] += decay ** (r["rows"] - 1 - i//r["top_k"])
            chosen = []
            for l in range(layers):
                chosen.extend(sorted(range(l*experts, (l+1)*experts),
                    key=lambda k: (-scores[k], rank[k]))[:quotas[l]])
            row["candidates"][str(half)] = dict(coverage=coverage(set(chosen)),
                                                   changed_from_profile=len(set(chosen)-baseline))
        result.append(row)
    return dict(path=path, sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(), requests=result)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--profile", default="perf/qwen4exp/profiles/qwen38-nf4-h4-v1.profile")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    result = [audit(p, args.profile) for p in args.traces]
    Path(args.output).write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
