"""Offline, guard-bound analysis of repeated HTTP service measurements.

Input files are acceptance records containing directory/guard/result SHA256.
The analyzer launches no process or GPU work. It keeps request compositions
separate and reports a heuristic screening band from independent run medians;
three repetitions do not establish a statistical tail-latency guarantee.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse_json(value):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON member: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Non-finite JSON number: {value}")

    return json.loads(value, object_pairs_hook=pairs, parse_constant=invalid)


def read_json(path):
    return parse_json(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def distribution(values):
    values = sorted(values)
    require(values and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values),
            "Expected nonempty, finite, nonnegative measurements")

    def percentile(p):
        rank = (len(values) - 1) * p
        lower, upper = math.floor(rank), math.ceil(rank)
        return values[lower] + (values[upper] - values[lower]) * (rank - lower)

    median, mean = statistics.median(values), statistics.fmean(values)
    return {"count": len(values), "min": values[0], "p10": percentile(.1), "p50": median,
            "p90": percentile(.9), "max": values[-1], "mean": mean,
            "mad": statistics.median(abs(v - median) for v in values),
            "cv": statistics.pstdev(values) / mean if mean else 0.0}


def screening(values, direction):
    summary = distribution(values)
    center = summary["p50"]
    span = summary["max"] - summary["min"]
    require(direction in ("higher", "lower"), "Unknown performance direction")
    return {"run_medians": list(values), "distribution": summary,
            "observed_span_fraction": span / center if center else None,
            "heuristic_screening_bound": max(0.0, center - 2 * span) if direction == "higher" else center + 2 * span,
            "better": direction}


def semantic(row):
    return {key: row[key] for key in ("content_sha256", "reasoning_sha256", "usage", "finish_reason")}


def analyze_rows(requests, batches):
    """Recompute metrics from every retained request; never discard failures."""
    require(requests and batches, "Missing raw measurements")
    require(all(not row.get("error") and row.get("outcome") in ("completed", "disconnected") for row in requests),
            "A failed request cannot be removed from performance evidence")
    by_label = {row["label"]: row for row in requests}
    require(len(by_label) == len(requests), "Duplicate request labels")
    require(len({row["id"] for row in requests}) == len(requests), "Duplicate response identities")
    references = {}
    for row in requests:
        if row["phase"] == "reference":
            require(row["outcome"] == "completed" and row["width"] == 1, "Reference must complete independently")
            require(row["payload_sha256"] not in references, "Duplicate reference payload")
            references[row["payload_sha256"]] = semantic(row)
    require(references, "No independent references")
    for row in requests:
        require(row["end_ns"] > row["start_ns"], "Invalid request clock interval")
        require((row["action"], row["outcome"]) in (("complete", "completed"), ("disconnect", "disconnected")),
                "Request action/outcome mismatch")
        if row["outcome"] == "completed":
            require(row["payload_sha256"] in references and semantic(row) == references[row["payload_sha256"]],
                    "Completed request differs from its independent reference")
            require(math.isclose(row["request_latency_seconds"], (row["end_ns"] - row["start_ns"]) / 1e9,
                                 rel_tol=1e-9, abs_tol=1e-9), "Request latency differs from raw timestamps")
            observations = [event["observed_ns"] for event in row["content_events"]]
            require(observations and all(type(value) is int and row["start_ns"] <= value <= row["end_ns"]
                    for value in observations) and observations == sorted(observations),
                    "Content observations must be ordered within the request")
            require(math.isclose(row["ttft_seconds"], (observations[0] - row["start_ns"]) / 1e9,
                                 rel_tol=1e-9, abs_tol=1e-9), "TTFT differs from first content observation")
            gaps = [(right - left) / 1e9 for left, right in zip(observations, observations[1:])]
            recorded = row["content_event_intervals_seconds"]
            require(len(gaps) == len(recorded) and all(math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)
                    for left, right in zip(gaps, recorded)), "Content intervals differ from raw observations")
    grouped = defaultdict(lambda: {"rates": [], "ttft": defaultdict(list), "latency": defaultdict(list),
                                  "content_intervals": defaultdict(list),
                                  "prompt_tokens": defaultdict(set), "output_tokens": defaultdict(list),
                                  "short_before_long": [], "short_progress_before_long": [], "tokens": 0})
    seen = []
    for batch in batches:
        require(batch.get("passed"), "Missing/failed batch cannot contribute measurements")
        require(all(label in by_label for label in batch["labels"]), "Batch refers to missing request")
        rows = [by_label[label] for label in batch["labels"]]
        seen.extend(batch["labels"])
        require(len(rows) == batch["width"] and all(row["phase"] == batch["phase"] and
                row["width"] == batch["width"] for row in rows), "Batch width/phase mismatch")
        span = (max(row["end_ns"] for row in rows) - min(row["start_ns"] for row in rows)) / 1e9
        require(math.isclose(span, batch["span_seconds"], rel_tol=1e-9, abs_tol=1e-9), "Batch span differs from raw clocks")
        if len(rows) > 1:
            require(max(row["start_ns"] for row in rows) < min(row["end_ns"] for row in rows), "Requests did not overlap")
        if batch["phase"] != "measured":
            continue
        require(all(row["outcome"] == "completed" for row in rows), "Cancelled measured request")
        composition = tuple(sorted(Counter(row["prompt_kind"] for row in rows).items()))
        group = grouped[(batch["profile"], batch["width"], composition)]
        tokens = sum(row["usage"]["completion_tokens"] for row in rows)
        require(tokens == batch["completion_tokens"] and math.isclose(tokens / span,
                batch["completion_tokens_per_second"], rel_tol=1e-9), "Batch throughput/count mismatch")
        group["rates"].append(tokens / span)
        group["tokens"] += tokens
        short, long = [], []
        for row in rows:
            kind = row["prompt_kind"]
            group["ttft"][kind].append(row["ttft_seconds"])
            group["latency"][kind].append(row["request_latency_seconds"])
            group["content_intervals"][kind].extend(row["content_event_intervals_seconds"])
            group["prompt_tokens"][kind].add(row["usage"]["prompt_tokens"])
            group["output_tokens"][kind].append(row["usage"]["completion_tokens"])
            if kind == "short": short.append(row)
            elif kind == "long": long.append(row)
        if short and long:
            group["short_before_long"].extend(row["end_ns"] < min(peer["end_ns"] for peer in long) for row in short)
            group["short_progress_before_long"].extend(row["content_events"][0]["observed_ns"] <
                min(peer["end_ns"] for peer in long) for row in short)
    require(len(seen) == len(set(seen)) == len(requests), "Missing/repeated batch request coverage")
    output = []
    for (profile, width, composition), values in sorted(grouped.items()):
        short_first = values["short_before_long"]
        short_progress = values["short_progress_before_long"]
        output.append({"profile": profile, "concurrency": width, "composition": dict(composition),
            "batch_throughput": distribution(values["rates"]), "completion_tokens": values["tokens"],
            "ttft_seconds": {kind: distribution(items) for kind, items in values["ttft"].items()},
            "request_latency_seconds": {kind: distribution(items) for kind, items in values["latency"].items()},
            "content_event_intervals_seconds": {kind: distribution(items) if items else None
                                                 for kind, items in values["content_intervals"].items()},
            "actual_prompt_tokens": {kind: sorted(items) for kind, items in values["prompt_tokens"].items()},
            "actual_completion_tokens": {kind: distribution(items) for kind, items in values["output_tokens"].items()},
            "short_completed_before_every_long": {"observations": len(short_first), "count": sum(short_first)},
            "short_first_content_before_every_long_done": {"observations": len(short_progress), "count": sum(short_progress)}})
    return {"groups": output, "phase_counts": dict(Counter(row["phase"] for row in requests)),
            "independent_references": references}


def load_run(acceptance_path):
    acceptance = read_json(acceptance_path)
    require(acceptance.get("passed"), "Acceptance record did not pass")
    directory = Path(acceptance["directory"])
    result_path, guard_path = directory / "result.json", Path(acceptance["guard"])
    require(sha256(result_path) == acceptance["result_sha256"] and sha256(guard_path) == acceptance["guard_sha256"],
            "Acceptance result/guard hashes changed")
    result, guard = read_json(result_path), read_json(guard_path)
    require(result.get("passed") and guard.get("status") == "finished" and not guard.get("evidence") and
            result["gpu_guard_run_id"] == guard["run_id"] and guard["details"]["result_sha256"] == sha256(result_path) and
            Path(guard["details"]["output"]).resolve() == directory.resolve(),
            "Missing successful guard/result binding")
    require(not result.get("error") and not result.get("log_errors") and not result.get("post_commit_errors"),
            "Successful flag cannot hide recorded run errors")
    require(result["cleanup"]["job_empty"] and not result["cleanup"]["forced_termination"] and
            not result["cleanup"]["errors"] and result["cleanup"]["exit_code"] == 0, "Run did not close normally")
    required_artifacts = {directory / name for name in ("arguments.json", "metadata.json", "measurements.jsonl",
                                                       "batches.jsonl", "final-report.json")}
    artifact_paths = [Path(artifact["path"]).resolve() for artifact in acceptance["artifacts"]]
    require(len(set(artifact_paths)) == len(artifact_paths) and
            {path.resolve() for path in required_artifacts}.issubset(artifact_paths),
            "Acceptance must bind every required raw artifact exactly once")
    for artifact in acceptance["artifacts"]:
        require(sha256(artifact["path"]) == artifact["sha256"], f"Changed evidence: {artifact['path']}")
    arguments, metadata = read_json(directory / "arguments.json"), read_json(directory / "metadata.json")
    require(result["backend"] == acceptance["backend"] == arguments["backend"] and
            result["gpu_guard_run_id"] == acceptance["guard_run_id"], "Acceptance identity differs from run")
    require(arguments["mode"] == "benchmark" and not arguments["execution_trace"], "Use non-trace benchmark runs")
    read_lines = lambda name: [parse_json(line) for line in (directory / name).read_text(encoding="utf-8").splitlines()]
    requests, batches = read_lines("measurements.jsonl"), read_lines("batches.jsonl")
    require(len(requests) == acceptance["request_count"] == result["request_count"] and
            len(batches) == result["batch_count"], "Recorded request/batch counts changed")
    require(all(row["outcome"] == "completed" for row in requests), "Benchmark must complete every request")
    analysis = analyze_rows(requests, batches)
    require(analysis["phase_counts"] == result["phase_request_counts"], "Recorded phase counts changed")
    expected = {(profile, width): arguments["rounds"] for profile in arguments["profiles"]
                for width in arguments["concurrency"]}
    require(dict(Counter((batch["profile"], batch["width"]) for batch in batches if batch["phase"] == "measured"))
            == expected, "Measured rounds differ from declared workload")
    keys = ("backend", "model", "context", "sessions", "concurrency", "profiles", "rounds", "warmup", "new_tokens",
            "long_prompt_repeat", "prompt", "seed", "temperature", "enable_thinking", "sample_interval", "output_bytes",
            "max_device_bytes", "max_pinned_host_bytes", "max_session_host_bytes")
    identity = {key: arguments[key] for key in keys}
    identity["files"] = metadata["file_identity"]
    identity["child_environment"] = metadata["child_environment_overrides"]
    identity["inherited_selected_environment"] = metadata["inherited_selected_environment"]
    identity["checkpoint_files"] = metadata["checkpoint_files"]
    identity["checkpoint_identity_scope"] = metadata["checkpoint_identity_scope"]
    identity["measurement_runtime"] = {key: metadata[key] for key in ("python", "platform", "machine", "clock", "definitions")}
    identity["boot_id"] = guard["snapshot"]["boot_id"]
    return {"acceptance": str(Path(acceptance_path).resolve()), "directory": str(directory.resolve()),
            "backend": result["backend"], "guard_run_id": guard["run_id"], "identity": identity, **analysis}


def combine(runs, minimum_repetitions=3):
    require(runs and type(minimum_repetitions) is int and minimum_repetitions >= 3,
            "Repeated baseline requires at least three independent runs")
    require(len({run["directory"] for run in runs}) == len(runs) and
            len({run["guard_run_id"] for run in runs}) == len(runs), "A run cannot be counted twice")
    by_backend = defaultdict(list)
    for run in runs:
        by_backend[run["backend"]].append(run)
    output = {}
    for backend, replicas in by_backend.items():
        require(len(replicas) >= minimum_repetitions, "Not enough independent process runs")
        first = replicas[0]
        require(all(run["identity"] == first["identity"] for run in replicas), "Software/workload/environment identity changed")
        require(all(run["independent_references"] == first["independent_references"] for run in replicas),
                "Independent process references changed")
        keyed = lambda run: {(group["profile"], group["concurrency"], tuple(sorted(group["composition"].items()))): group
                             for group in run["groups"]}
        groups = [keyed(run) for run in replicas]
        require(groups[0] and all(group.keys() == groups[0].keys() for group in groups), "Measured case coverage changed")
        cases = []
        for key in sorted(groups[0]):
            values = [group[key] for group in groups]
            require(all(value["actual_prompt_tokens"] == values[0]["actual_prompt_tokens"] for value in values),
                    "Prompt lengths changed")
            cases.append({"profile": key[0], "concurrency": key[1], "composition": dict(key[2]),
                "actual_prompt_tokens": values[0]["actual_prompt_tokens"], "runs": values,
                "throughput": screening([value["batch_throughput"]["p50"] for value in values], "higher"),
                "ttft": {kind: screening([value["ttft_seconds"][kind]["p50"] for value in values], "lower")
                         for kind in values[0]["ttft_seconds"]},
                "request_latency": {kind: screening([value["request_latency_seconds"][kind]["p50"] for value in values], "lower")
                                    for kind in values[0]["request_latency_seconds"]}})
        output[backend] = {"process_runs": len(replicas), "identity": first["identity"], "cases": cases,
                           "acceptance_records": [run["acceptance"] for run in replicas]}
    return {"passed": True, "backends": output,
        "screening_rule": "HEURISTIC: median of independent run medians +/- twice their observed max-min span; zero-clamped throughput floor. A review trigger, not a statistical confidence interval or universal regression limit.",
        "scope": "Only measured phases. Mixed width-one short and long waves are separate compositions. Client-observed SSE timing includes host/IO observation costs; it is not kernel or per-token latency.",
        "long_term_stability_claimed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = combine([load_run(path) for path in args.acceptance])
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    print(json.dumps({"passed": True, "backends": list(result["backends"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
