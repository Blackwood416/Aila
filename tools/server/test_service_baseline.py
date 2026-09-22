"""Offline evidence validation; no Server, Worker, or GPU is started."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import service_baseline as baseline


def fixture(scale=1.0):
    requests, batches = [], []
    cursor = 1_000_000_000

    def batch(phase, profile, kinds):
        nonlocal cursor
        rows = []
        for lane, kind in enumerate(kinds):
            duration = scale * (1 if kind == "short" else 3)
            if phase == "warmup": duration *= 50
            label = f"row-{len(requests)}"
            prompt = 27 if kind == "short" else 1825
            row = {"label": label, "id": "id-" + label, "phase": phase, "profile": profile, "width": len(kinds),
                "prompt_kind": kind, "payload_sha256": "payload-" + kind, "outcome": "completed", "action": "complete",
                "start_ns": cursor, "end_ns": cursor + int(duration * 1e9), "request_latency_seconds": duration,
                "ttft_seconds": duration / 10, "content_sha256": "text-" + kind, "reasoning_sha256": "empty",
                "finish_reason": "length", "usage": {"prompt_tokens": prompt, "completion_tokens": 10, "total_tokens": prompt + 10}}
            row["content_events"] = [{"observed_ns": cursor + int(duration * fraction * 1e9)} for fraction in (.1, .5)]
            row["content_event_intervals_seconds"] = [duration * .4]
            requests.append(row); rows.append(row)
        span = (max(row["end_ns"] for row in rows) - cursor) / 1e9
        batches.append({"phase": phase, "profile": profile, "width": len(kinds), "labels": [row["label"] for row in rows],
                        "passed": True, "span_seconds": span, "completion_tokens": 10 * len(kinds),
                        "completion_tokens_per_second": 10 * len(kinds) / span})
        cursor += int((span + 1) * 1e9)

    batch("reference", "short", ["short"])
    batch("reference", "long", ["long"])
    batch("warmup", "short", ["short"])
    batch("measured", "short", ["short"])
    batch("measured", "mixed", ["short"])
    batch("measured", "mixed", ["long"])
    batch("measured", "mixed", ["short", "long"])
    return requests, batches


def run(index, scale=1.0):
    return {"directory": f"run-{index}", "guard_run_id": f"guard-{index}", "backend": "fake",
            "acceptance": f"acceptance-{index}", "identity": {"software": "unchanged", "context": 2048},
            **baseline.analyze_rows(*fixture(scale))}


class ServiceBaselineTests(unittest.TestCase):
    def test_distribution_and_screening_are_explicit(self):
        value = baseline.distribution([1, 2, 3])
        self.assertEqual(value["p50"], 2)
        self.assertAlmostEqual(value["p10"], 1.2)
        self.assertEqual(value["mad"], 1)
        value = baseline.screening([98, 100, 102], "higher")
        self.assertEqual(value["heuristic_screening_bound"], 92)
        self.assertEqual(baseline.screening([98, 100, 102], "lower")["heuristic_screening_bound"], 108)
        for samples in ([], [float("nan")], [float("inf")], [-1], [True]):
            with self.assertRaises(ValueError): baseline.distribution(samples)

    def test_warmup_is_excluded_and_mixed_compositions_stay_separate(self):
        value = baseline.analyze_rows(*fixture())
        self.assertEqual(len(value["groups"]), 4)
        groups = {(row["profile"], row["concurrency"], tuple(row["composition"])): row for row in value["groups"]}
        self.assertEqual(groups[("short", 1, ("short",))]["batch_throughput"]["p50"], 10)
        self.assertEqual(groups[("mixed", 1, ("short",))]["batch_throughput"]["p50"], 10)
        self.assertAlmostEqual(groups[("mixed", 1, ("long",))]["batch_throughput"]["p50"], 10 / 3)
        mixed = next(row for row in value["groups"] if row["concurrency"] == 2)
        self.assertEqual(mixed["short_completed_before_every_long"], {"observations": 1, "count": 1})

    def test_failed_warmup_cannot_be_hidden(self):
        requests, batches = fixture()
        requests[2]["error"] = "lost output"
        with self.assertRaisesRegex(ValueError, "failed request"):
            baseline.analyze_rows(requests, batches)

    def test_progress_is_distinct_from_finish_order(self):
        requests, batches = fixture()
        # The long prompt can finish fewer output tokens sooner. That does
        # not prevent the short prompt from making progress while it runs.
        last = requests[-1]
        last["end_ns"] = last["start_ns"] + 500_000_000
        last["request_latency_seconds"] = .5
        last["ttft_seconds"] = .05
        last["content_events"] = [{"observed_ns": last["start_ns"] + value} for value in (50_000_000, 250_000_000)]
        last["content_event_intervals_seconds"] = [.2]
        batches[-1].update(span_seconds=1.0, completion_tokens_per_second=20.0)
        value = baseline.analyze_rows(requests, batches)
        mixed = next(row for row in value["groups"] if row["concurrency"] == 2)
        self.assertEqual(mixed["short_completed_before_every_long"], {"observations": 1, "count": 0})
        self.assertEqual(mixed["short_first_content_before_every_long_done"], {"observations": 1, "count": 1})

    def test_mismatch_cannot_be_its_own_oracle(self):
        requests, batches = fixture()
        requests[-1]["content_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "independent reference"):
            baseline.analyze_rows(requests, batches)

    def test_complete_request_and_batch_coverage(self):
        requests, batches = fixture()
        with self.assertRaisesRegex(ValueError, "coverage"):
            baseline.analyze_rows(requests, batches[:-1])
        with self.assertRaisesRegex(ValueError, "coverage"):
            baseline.analyze_rows(requests, batches + [batches[-1]])
        requests[-1]["id"] = requests[0]["id"]
        with self.assertRaisesRegex(ValueError, "identities"):
            baseline.analyze_rows(requests, batches)

    def test_raw_clock_and_throughput_must_match(self):
        requests, batches = fixture()
        batches[-1]["span_seconds"] += .1
        with self.assertRaisesRegex(ValueError, "raw clocks"):
            baseline.analyze_rows(requests, batches)
        requests, batches = fixture()
        batches[-1]["completion_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "throughput/count"):
            baseline.analyze_rows(requests, batches)

    def test_content_timing_is_bound_to_raw_observations(self):
        requests, batches = fixture()
        requests[-1]["ttft_seconds"] += .1
        with self.assertRaisesRegex(ValueError, "TTFT differs"):
            baseline.analyze_rows(requests, batches)
        requests, batches = fixture()
        requests[-1]["content_events"].reverse()
        with self.assertRaisesRegex(ValueError, "ordered within"):
            baseline.analyze_rows(requests, batches)
        requests, batches = fixture()
        requests[-1]["content_event_intervals_seconds"] = [0]
        with self.assertRaisesRegex(ValueError, "intervals differ"):
            baseline.analyze_rows(requests, batches)

    def test_zero_gaps_are_preserved_as_network_observations(self):
        requests, batches = fixture()
        for row in requests:
            row["content_events"][1]["observed_ns"] = row["content_events"][0]["observed_ns"]
            row["content_event_intervals_seconds"] = [0]
        value = baseline.analyze_rows(requests, batches)
        self.assertEqual(value["groups"][0]["content_event_intervals_seconds"]["long"]["p50"], 0)

    def test_three_independent_repetitions_and_identity(self):
        runs = [run(0), run(1, 1.02), run(2, .98)]
        value = baseline.combine(runs)
        self.assertEqual(value["backends"]["fake"]["process_runs"], 3)
        self.assertIn("HEURISTIC", value["screening_rule"])
        with self.assertRaisesRegex(ValueError, "independent process"):
            baseline.combine(runs[:2])
        with self.assertRaisesRegex(ValueError, "counted twice"):
            baseline.combine([runs[0], runs[1], runs[0]])
        changed = copy.deepcopy(runs)
        changed[2]["identity"]["software"] = "changed"
        with self.assertRaisesRegex(ValueError, "identity changed"):
            baseline.combine(changed)
        changed = copy.deepcopy(runs)
        changed[1]["groups"].pop()
        with self.assertRaisesRegex(ValueError, "coverage changed"):
            baseline.combine(changed)
        with self.assertRaisesRegex(ValueError, "at least three"):
            baseline.combine([])
        with self.assertRaisesRegex(ValueError, "at least three"):
            baseline.combine(runs, minimum_repetitions=1)

    def test_strict_json_evidence(self):
        for text in ('{"passed": false, "passed": true}', '{"time": NaN}', '{"time": Infinity}'):
            with self.assertRaises(ValueError): baseline.parse_json(text)

    def test_acceptance_requires_complete_unique_raw_artifact_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = {"passed": True, "gpu_guard_run_id": "guard-1", "cleanup": {
                "job_empty": True, "forced_termination": False, "errors": [], "exit_code": 0}}
            (root / "result.json").write_text(json.dumps(result), encoding="utf-8")
            guard = {"status": "finished", "run_id": "guard-1", "details": {
                "output": str(root), "result_sha256": baseline.sha256(root / "result.json")}}
            (root / "guard.json").write_text(json.dumps(guard), encoding="utf-8")
            artifacts = []
            for name in ("arguments.json", "metadata.json", "measurements.jsonl", "batches.jsonl", "final-report.json"):
                path = root / name
                path.write_text("{}", encoding="utf-8")
                artifacts.append({"path": str(path), "sha256": baseline.sha256(path)})
            acceptance = {"passed": True, "directory": str(root), "guard": str(root / "guard.json"),
                          "result_sha256": baseline.sha256(root / "result.json"),
                          "guard_sha256": baseline.sha256(root / "guard.json")}
            path = root / "acceptance.json"
            for invalid in (artifacts[:-1], artifacts + [artifacts[0]]):
                acceptance["artifacts"] = invalid
                path.write_text(json.dumps(acceptance), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "every required raw artifact"):
                    baseline.load_run(path)

    def test_acceptance_cannot_bind_different_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "result.json").write_text("{}", encoding="utf-8")
            (root / "guard.json").write_text("{}", encoding="utf-8")
            acceptance = {"passed": True, "directory": str(root), "guard": str(root / "guard.json"),
                          "result_sha256": "0" * 64, "guard_sha256": baseline.sha256(root / "guard.json")}
            path = root / "acceptance.json"
            path.write_text(json.dumps(acceptance), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hashes changed"):
                baseline.load_run(path)


if __name__ == "__main__":
    unittest.main()
