"""Attribute Q4E oneDNN GEMM using adjacent GPU consumer names.

For the measured in-order Q4E path only. Copy/barrier tasks are omitted before
matching. This is sequence attribution, not an independently instrumented op
range. Unknown consumers remain explicit; no submission-gap savings inferred.
Requires a validated high-resolution timeline summary and its source TSV.
"""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean


def attribute(prefix, first, last):
    timeline = json.loads(Path(prefix + ".timeline.json").read_text())
    path = Path(prefix + ".gpu.tsv")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == timeline["tsv_sha256"]
    with path.open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert rows and all(int(r["duplicateCount"]) == 0 for r in rows)
    events = sorted((float(r["startTime"]), float(r["endTime"]), r["name"]) for r in rows
                    if not r["name"].startswith("zeCommandList"))
    starts = [a for a, _, _ in events]
    forwards = []
    mapping = [("apply_silu_div4(", "hc_down"), ("apply_sigmoid(", "hc_up"),
               ("apply_two_sigmoid_div4(", "hc_inject"),
               ("router_softmax_topk_device(", "router"),
               ("router_topk_ids_device(", "prediction_router")]
    for f in timeline["forwards"]:
        if not first <= f["forward"] <= last:
            continue
        seq = events[bisect_left(starts, f["start_s"]):bisect_right(starts, f["end_s"])]
        seq = [(a,b,n) for a,b,n in seq if b <= f["end_s"]]
        counts, times, unknown = Counter(), defaultdict(float), Counter()
        for index, (a,b,name) in enumerate(seq):
            if "apply_silu_div4(" in name:
                kind = "hc_silu"
            elif "apply_sigmoid(" in name:
                kind = "hc_sigmoid"
            elif "apply_two_sigmoid_div4(" in name:
                kind = "hc_inject_sigmoid"
            elif "ops::hc_mix_reduce(" in name:
                kind = "hc_mix_reduce"
            elif "hc_decode_down_inject_impl<(bool)1>" in name:
                kind = "hc_fused_down_inject"
            elif "hc_decode_up_mix_impl<(bool)1>" in name:
                kind = "hc_fused_up_mix"
            elif name == "gemm_kernel":
                following = seq[index+1][2] if index+1 < len(seq) else "<end>"
                kind = next((kind for marker,kind in mapping if marker in following), None)
                if kind is None and "packed_nf4_decode_rows_bf16(" in following:
                    kind = "dense_before_nf4"
                if kind is None and following in ("Q4ETraceForwardEnd", "<end>"):
                    kind = "lm_head_by_sequence"
                if kind is None:
                    kind = "unattributed_gemm"
                    unknown[following] += 1
            else:
                continue
            counts[kind] += 1
            times[kind] += (b-a)*1000
        forwards.append(dict(forward=f["forward"], tokens=f["tokens"], counts=dict(counts),
                             kernel_sum_ms=dict(times), unknown_consumers=dict(unknown)))
    assert forwards
    kinds = sorted(set().union(*(f["counts"] for f in forwards)))
    return dict(prefix=prefix, first=first, last=last, tsv_sha256=timeline["tsv_sha256"],
                caveats=__doc__, complete_forwards=len(forwards),
                mean_kernel_ms={k:mean(f["kernel_sum_ms"].get(k,0) for f in forwards) for k in kinds},
                mean_count={k:mean(f["counts"].get(k,0) for f in forwards) for k in kinds}, forwards=forwards)


if __name__ == "__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("prefix"); ap.add_argument("--first",type=int,required=True)
    ap.add_argument("--last",type=int,required=True); ap.add_argument("--output",required=True)
    a=ap.parse_args(); result=attribute(a.prefix,a.first,a.last)
    Path(a.output).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k != "forwards"},indent=2))
