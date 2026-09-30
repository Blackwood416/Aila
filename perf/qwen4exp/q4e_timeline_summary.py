"""Summarize VTune GPUComputeTask interval exports between Q4E GPU markers.

Export with -report timeline -report-knob column-by=GPUComputeTask
-report-knob query-type=interval -report-knob object-names=true
-report-knob bin-count=10000000 -report-knob time-precision=9 -format csv.
VTune emits TSV even with -format csv. Coalesced/missing markers are rejected.
Uncovered time means no recorded application kernel/copy, not hardware idle.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

from q4e_cache_replay import read_trace, require, validate


def union(intervals):
    result = []
    for begin, end in sorted(intervals):
        if result and begin <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([begin, end])
    return result


def length(intervals):
    return sum(b-a for a, b in union(intervals))


def summarize(tsv, trace):
    rows = list(csv.DictReader(Path(tsv).open(encoding="utf-8-sig"), delimiter="	"))
    require(rows and all(int(r["duplicateCount"]) == 0 for r in rows), "Coalesced or empty GPU interval export; increase bin-count")
    events = sorted([(float(r["startTime"]), float(r["endTime"]), r["name"]) for r in rows])
    starts = [a for a, b, n in events if n == "Q4ETraceForwardBegin"]
    ends = [b for a, b, n in events if n == "Q4ETraceForwardEnd"]
    observed = validate(read_trace(trace))
    # VTune can lose the last GPU event during process teardown. Never infer its
    # end from host time. Only a single trailing unmatched begin may be excluded.
    require(len(starts) == len(observed) and len(ends) in (len(starts), len(starts)-1), "Forward/marker count mismatch")
    excluded = [r["forward"] for r in observed[len(ends):]]
    results = []
    for index, (begin, end, forward) in enumerate(zip(starts, ends, observed)):
        require(begin < end and (index == 0 or begin >= ends[index-1]), "Overlapping forward markers")
        kernels, copies = [], []
        names = defaultdict(float)
        for a, b, name in events:
            a, b = max(a, begin), min(b, end)
            if a >= b or name.startswith("Q4ETraceForward"):
                continue
            # All copy directions, not just demand H2D.
            if "MemoryCopy" in name or "MemoryFill" in name:
                copies.append((a, b))
            else:
                kernels.append((a, b))
                names[name] += b-a
        kernel_time, copy_time = length(kernels), length(copies)
        covered = length(kernels + copies)
        overlap = kernel_time + copy_time - covered
        gaps = []
        cursor = begin
        for a, b in union(kernels + copies):
            if a > cursor:
                gaps.append(a-cursor)
            cursor = max(cursor, b)
        if cursor < end:
            gaps.append(end-cursor)
        results.append(dict(
            forward=forward["forward"], tokens=forward["tokens"], position=forward["position"], logits=forward.get("logits"),
            start_s=begin, end_s=end, span_ms=(end-begin)*1000,
            kernel_union_ms=kernel_time*1000, all_copy_union_ms=copy_time*1000,
            overlap_ms=overlap*1000, uncovered_ms=(end-begin-covered)*1000,
            kernel_count=len(kernels), copy_count=len(copies),
            max_uncovered_gap_ms=max(gaps, default=0)*1000,
            gaps_over_100us=sum(g > .0001 for g in gaps),
            host_acquire_ms=forward["acquire_ns"]*1e-6,
            demand_bytes=forward["bytes"],
            top_kernels=[dict(name=n, sum_ms=v*1000) for n, v in sorted(names.items(), key=lambda kv: -kv[1])[:8]],
        ))
    return dict(
        tsv_sha256=hashlib.sha256(Path(tsv).read_bytes()).hexdigest(),
        trace_sha256=hashlib.sha256(Path(trace).read_bytes()).hexdigest(),
        caveats=["Diagnostic profiling and marker overhead included",
                 "Uncovered is absence of recorded application tasks, not hardware utilization",
                 "Copy includes all directions; kernel union excludes two marker kernels",
                 "GPU markers bracket the backend profile region, excluding initial embedding/PLE submission",
                 "Host acquire durations use a separate clock; do not subtract them from GPU spans"],
        excluded_incomplete_forwards=excluded, forwards=results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tsv")
    ap.add_argument("trace")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    result = summarize(args.tsv, args.trace)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    for r in result["forwards"]:
        print({k:v for k,v in r.items() if k not in ("top_kernels",)})
