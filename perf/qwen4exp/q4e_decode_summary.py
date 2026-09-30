"""Summarize fixed-continuation target work; never label it generated throughput.

The first rows=1 run for each input is the numerical reference. NumPy is used
only for offline logits scoring, in small row batches to bound host memory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_run(prefix):
    prefix = str(prefix)
    meta = json.loads(Path(prefix + ".meta.json").read_text(encoding="utf-8-sig"))
    require(meta.get("exit_code") == 0 and not meta["watchdog_stop"], "Failed or incomplete probe: " + prefix)
    require(sha256(prefix + ".log").upper() == meta["log_sha256"].upper(), "Changed log: " + prefix)
    lines = Path(prefix + ".log").read_text(encoding="utf-8").splitlines()
    identity = next(s for s in lines if s.startswith("[DECODE INPUT] "))
    passes = [json.loads(s.split("[DECODE PASS] ", 1)[1]) for s in lines if "[DECODE PASS] {" in s]
    require(len(passes) == meta["input"]["repeats"], "Missing passes: " + prefix)
    for i, sample in enumerate(passes):
        require(sample["pass"] == i and sample["steps"] == 128, "Wrong pass/step count")
        require(sample["forwards"] * sample["rows"] == 128, "Wrong forward count")
        require(len(sample["forward_ms"]) == sample["forwards"], "Missing forward measurements")
        require(len(sample["step_hashes"]) == 128 and len(sample["top1"]) == 128, "Missing logits checks")
        require(sample["step_hashes"] == passes[0]["step_hashes"], "Nondeterministic replay")
        # Earlier executable called this new_usm_allocations. It actually only
        # compared live ledger counts; do not claim detection of allocate/free.
        sample["live_usm_count_delta"] = sample.pop("new_usm_allocations", sample.get("live_usm_count_delta", 0))
    warm = [p["core_ms_per_input"] for p in passes[1:]]
    middle = statistics.median(warm)
    rows = passes[0]["rows"]
    late = [p["core_ms_per_input"] for p in passes[2:]]
    return dict(prefix=prefix, metadata=meta, identity=identity, passes=passes,
                logits_sha256=sha256(prefix + ".logits.bf16"),
                warm=dict(core_ms_per_input_median=middle,
                          loop_ms_per_input_median=statistics.median(p["loop_ms_per_input"] for p in passes[1:]),
                          core_pass_cv=statistics.pstdev(warm) / statistics.mean(warm),
                          core_ms_per_block=middle * rows,
                          minimum_committed_tokens_for_20tps_zero_draft=middle * rows / 50,
                          max_draft_and_rollback_ms_for_20tps_all_accepted=(50-middle) * rows),
                after_two_passes=dict(sample_count=len(late),
                                      core_ms_per_input_median=statistics.median(late) if late else None,
                                      core_ms_per_input_samples=late))


def compare(reference, candidate):
    import numpy as np
    paths = [run["prefix"] + ".logits.bf16" for run in (reference, candidate)]
    size = Path(paths[0]).stat().st_size
    require(Path(paths[1]).stat().st_size == size and size % (128 * 2) == 0, "Incompatible logits shape")
    vocab = size // (128 * 2)
    a, b = [np.memmap(p, dtype=np.uint16, mode="r", shape=(128, vocab)) for p in paths]
    bitwise_rows = 0
    different_top1_steps, kls, nrmses = [], [], []
    for start in range(0, 128, 4):
        aa, bb = a[start:start+4], b[start:start+4]
        bitwise_rows += int(np.sum(np.all(aa == bb, axis=1)))
        x, y = [(v.astype(np.uint32) << 16).view(np.float32).astype(np.float64) for v in (aa, bb)]
        require(np.isfinite(x).all() and np.isfinite(y).all(), "Nonfinite logits artifact")
        different_top1_steps.extend((np.flatnonzero(x.argmax(1) != y.argmax(1)) + start).tolist())
        nrmses.extend(np.sqrt(np.sum((x-y)**2, axis=1) / np.maximum(np.sum(x*x, axis=1), 1e-12)).tolist())
        lx, ly = x-x.max(1, keepdims=True), y-y.max(1, keepdims=True)
        lx -= np.log(np.exp(lx).sum(1, keepdims=True))
        ly -= np.log(np.exp(ly).sum(1, keepdims=True))
        kls.extend(np.sum(np.exp(lx)*(lx-ly), axis=1).tolist())
    return dict(reference=reference["prefix"], candidate=candidate["prefix"],
                rows=128, vocab=vocab, bitwise_rows=bitwise_rows,
                top1_matches=128-len(different_top1_steps), different_top1_steps=different_top1_steps,
                mean_kl=statistics.mean(kls), max_kl=max(kls), max_nrmse=max(nrmses),
                exact_greedy_parity=len(different_top1_steps) == 0,
                note="Teacher-forced decisions, not MTP draft acceptance or free-generation quality")


def add_teacher_nll(result, tokenizer_path, probe_source):
    """Score only the 127 known next targets; verify token identity before use."""
    import ast
    import re
    import struct
    import numpy as np
    from tokenizers import Tokenizer
    source = Path(probe_source).read_text(encoding="utf-8")
    encoded = source.split("auto continuation = tokenizer.encode(", 1)[1].split(");", 1)[0]
    text = "".join(ast.literal_eval(s) for s in re.findall(r'"(?:[^"\\]|\\.)*"', encoded))
    ids = Tokenizer.from_file(str(tokenizer_path)).encode(text, add_special_tokens=False).ids[:128]
    require(len(ids) == 128, "Missing teacher tokens")
    fnv = 14695981039346656037
    for byte in struct.pack("<128i", *ids):
        fnv = ((fnv ^ byte) * 1099511628211) & ((1 << 64)-1)
    values = {}
    for run in result["runs"]:
        require("continuation_fnv64="+str(fnv) in run["identity"], "Teacher tokenizer identity mismatch")
        path = run["prefix"] + ".logits.bf16"
        vocab = Path(path).stat().st_size // (128*2)
        data = np.memmap(path, mode="r", dtype=np.uint16, shape=(128, vocab))
        scores = []
        for i in range(127):
            row = (data[i].astype(np.uint32) << 16).view(np.float32).astype(np.float64)
            require(np.isfinite(row).all(), "Nonfinite teacher logits")
            maximum = row.max()
            scores.append(float(maximum + np.log(np.exp(row-maximum).sum()) - row[ids[i+1]]))
        values[run["prefix"]] = statistics.mean(scores)
    for comparison in result["numerical_comparisons"]:
        comparison["teacher_targets"] = 127
        comparison["mean_nll_delta"] = values[comparison["candidate"]] - values[comparison["reference"]]
    result["teacher_target_identity"] = dict(first_128_ids_fnv64=str(fnv),
        tokenizer_sha256=sha256(tokenizer_path), probe_source_sha256=sha256(probe_source),
        mean_nll_by_run=values, note="Score logits0..126 against consumed ids1..127; unknown final next target omitted. Small-text observation, not a quality guarantee.")


def summarize(prefixes):
    runs = [read_run(p) for p in prefixes]
    references, comparisons = {}, []
    for run in runs:
        # Rows differ by definition; token identities must not.
        key = run["identity"].split(" prompt_fnv64=", 1)[1]
        if key not in references and run["passes"][0]["rows"] == 1:
            references[key] = run
        if key in references:
            comparisons.append(compare(references[key], run))
    return dict(caveats=[
        "Fixed 128 consumed tokens, all logits rows; no sampling, draft, rollback or prefix reuse",
        "Core sums forward-to-synchronize wall; loop additionally includes D2H, finite checks and CPU scoring",
        "Pass zero reported separately; subsequent passes share one process with capacity-limited caches",
        "OS file cache was not flushed and host/GPU cache states are not forcibly identical",
        "Live USM counts are not cumulative allocation counts; transient allocate/free is not detected",
        "Driver/library allocations and DXGI resident VRAM are not measured by the application ledger"
    ], runs=runs, numerical_comparisons=comparisons)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefixes", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher-tokenizer", help="Optional tokenizer.json for verified fixed-trajectory NLL")
    parser.add_argument("--probe-source", default="tests/models/Qwen4ExpDecodeProbe.hpp")
    args = parser.parse_args()
    result = summarize(args.prefixes)
    if args.teacher_tokenizer:
        add_teacher_nll(result, args.teacher_tokenizer, args.probe_source)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for run in result["runs"]:
        print(run["prefix"], run["warm"])
    for comparison in result["numerical_comparisons"]:
        print(comparison)
