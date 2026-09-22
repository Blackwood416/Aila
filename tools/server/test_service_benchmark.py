"""CPU checks only; the Windows smoke uses isolated Python helper processes."""

import contextlib
import copy
import hashlib
import io
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import service_benchmark as bench
import gpu_run_guard as guard_module


def envelope(delta=None, reason=None, usage=None, empty=False):
    return {"id": "request-test", "created": 100, "model": "qwen3", "object": "chat.completion.chunk",
            "choices": [] if empty else [{"index": 0, "delta": delta or {}, "finish_reason": reason}], "usage": usage}


def packet(value, newline=b"\n"):
    raw = value if isinstance(value, bytes) else bench.encoded_json(value)
    return b"data: " + raw + newline + newline


def write_execution_trace(directory, pid=123, complete=True):
    trace_id = f"execution-{pid}-1-1"
    trace_directory = directory / "execution-trace"
    trace_directory.mkdir(exist_ok=True)
    record = {"schema_version": 1, "trace_id": trace_id, "seq": 1, "pid": pid, "tid": 1,
              "utc_ns": 1, "steady_ns": 0, "event": "trace_open", "scope_id": 0,
              "parent_scope_id": 0, "operation": "trace", "status": "ok", "model_id": 0,
              "session_id": 0, "phase": "none", "token_count": -1, "position": -1,
              "target_position": -1, "bytes": 0, "memory": "none"}
    data = bench.encoded_json(record) + b"\n"
    if complete:
        data += bench.encoded_json({**record, "seq": 2, "event": "trace_close"}) + b"\n"
    else:
        data += b'{"event":"begin"'
    path = trace_directory / (trace_id + ".jsonl")
    path.write_bytes(data)
    return path


def frames():
    return [envelope({"role": "assistant"}), envelope({"content": "é中"}),
            envelope({"reasoning_content": "🙂"}), envelope(reason="length"),
            envelope(usage={"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}, empty=True), b"[DONE]"]


class ParserTests(unittest.TestCase):
    def parser(self, maximum=2, context=64, output=65536):
        return bench.SseParser("qwen3", maximum, context, output)

    def test_ttft_excludes_role_and_intervals_include_reasoning(self):
        parser = self.parser()
        items = frames()
        parser.feed(packet(items[0]), 1_000_000_000)
        parser.feed(packet(items[1]), 2_000_000_000)
        parser.feed(packet(items[2]), 3_000_000_000)
        parser.feed(b"".join(packet(item) for item in items[3:]), 4_000_000_000)
        parser.finish()
        value = parser.measurement(500_000_000)
        self.assertEqual(value["ttft_seconds"], 1.5)
        self.assertEqual(value["request_latency_seconds"], 3.5)
        self.assertEqual(value["content_event_intervals_seconds"], [1.0])
        self.assertEqual(value["content_events"][1]["channels"], ["reasoning_content"])
        self.assertEqual(value["usage"]["completion_tokens"], 2)

    def test_coalesced_events_keep_zero_intervals(self):
        parser = self.parser()
        parser.feed(b"".join(packet(item) for item in frames()), 2_000_000_000)
        parser.finish()
        value = parser.measurement(1_000_000_000)
        self.assertEqual(value["content_event_intervals_seconds"], [0.0])
        self.assertEqual(value["ttft_seconds"], 1.0)

    def test_every_byte_fragment_and_crlf(self):
        parser = self.parser()
        wire = b"".join(packet(item, b"\r\n") for item in frames())
        for index, byte in enumerate(wire):
            parser.feed(bytes([byte]), index + 1)
        parser.finish()
        self.assertEqual("".join(parser.content), "é中")
        self.assertEqual("".join(parser.reasoning), "🙂")

    def test_error_event_fails_instead_of_counting_a_completion(self):
        parser = self.parser()
        parser.feed(packet(envelope({"role": "assistant"})), 1)
        with self.assertRaisesRegex(bench.BenchmarkError, "Unexpected SSE error"):
            parser.feed(packet({"error": {"message": "worker failed"}}), 2)

    def test_usage_rejects_wrong_count_types_sum_and_context(self):
        values = [
            {"prompt_tokens": True, "completion_tokens": 2, "total_tokens": 3},
            {"prompt_tokens": 4, "completion_tokens": 1.5, "total_tokens": 5.5},
            {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 7},
            {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
            {"prompt_tokens": 63, "completion_tokens": 2, "total_tokens": 65},
        ]
        for usage in values:
            with self.subTest(usage=usage):
                parser = self.parser()
                parser.feed(b"".join(packet(item) for item in frames()[:4]), 1)
                with self.assertRaises(bench.BenchmarkError):
                    parser.feed(packet(envelope(usage=usage, empty=True)), 2)

    def test_missing_terminal_and_duplicate_done_fail(self):
        parser = self.parser()
        parser.feed(b"".join(packet(item) for item in frames()[:-1]), 1)
        with self.assertRaises(bench.BenchmarkError):
            parser.finish()
        parser.feed(packet(b"[DONE]"), 2)
        parser.finish()
        with self.assertRaises(bench.BenchmarkError):
            parser.feed(packet(b"[DONE]"), 3)

    def test_id_changes_and_boolean_indices_fail(self):
        for field, value in (("id", "different"), ("created", True), ("index", False)):
            with self.subTest(field=field):
                parser = self.parser()
                parser.feed(packet(envelope({"role": "assistant"})), 1)
                item = envelope({"content": "x"})
                if field == "index":
                    item["choices"][0][field] = value
                else:
                    item[field] = value
                with self.assertRaises(bench.BenchmarkError):
                    parser.feed(packet(item), 2)

    def test_invalid_or_incomplete_utf8_and_bounds(self):
        with self.assertRaises(UnicodeDecodeError):
            self.parser().feed(b"data: \xff", 1)
        parser = self.parser()
        parser.feed(b"data: \xc3", 1)
        with self.assertRaises(UnicodeDecodeError):
            parser.finish()
        with self.assertRaises(bench.BenchmarkError):
            self.parser().feed(b"data: " + b"x" * bench.FRAME_BYTES, 1)
        with self.assertRaises(bench.BenchmarkError):
            self.parser(output=8).feed(b"data: more", 1)

    def test_strict_json_rejects_duplicates_and_nan(self):
        for value in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}'):
            with self.subTest(value=value), self.assertRaises(bench.BenchmarkError):
                bench.strict_json(value)


class StatisticsTests(unittest.TestCase):
    def test_interpolated_percentiles_and_empty(self):
        result = bench.distribution([3, 1])
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["p50"], 2)
        self.assertAlmostEqual(result["p90"], 2.8)
        self.assertIsNone(bench.distribution([])["mean"])
        with self.assertRaises(bench.BenchmarkError):
            bench.distribution([float("nan")])

    @staticmethod
    def row(phase, tokens, latency, ttft):
        return {"phase": phase, "profile": "mixed", "width": 2, "prompt_kind": "short", "outcome": "completed",
                "usage": {"prompt_tokens": 4, "completion_tokens": tokens}, "ttft_seconds": ttft,
                "request_latency_seconds": latency, "completion_tokens_per_second": tokens / latency,
                "content_event_intervals_seconds": [0.0, 0.1]}

    def test_warmup_excluded_and_batch_throughput_is_not_sum_of_request_rates(self):
        rows = [self.row("reference", 200, 200, 199), self.row("warmup", 100, 100, 99),
                self.row("measured", 4, 2, 1), self.row("measured", 8, 4, 3)]
        batches = [{"phase": "reference", "profile": "mixed", "width": 2, "passed": True, "span_seconds": 200,
                    "completion_tokens_per_second": 1},
                   {"phase": "warmup", "profile": "mixed", "width": 2, "passed": True, "span_seconds": 100,
                    "completion_tokens_per_second": 1},
                   {"phase": "measured", "profile": "mixed", "width": 2, "passed": True, "span_seconds": 4,
                    "completion_tokens_per_second": 3}]
        result = bench.summarize_measurements(rows, batches)[0]
        self.assertEqual(result["requests"], 2)
        self.assertEqual(result["ttft_seconds"]["maximum"], 3)
        self.assertEqual(result["total_completion_tokens"], 12)
        self.assertEqual(result["completion_tokens_per_second_over_batch_spans"], 3)
        self.assertEqual(result["request_completion_tokens_per_second"]["mean"], 2)
        rows[2]["outcome"] = "failed"
        with self.assertRaises(bench.BenchmarkError):
            bench.summarize_measurements(rows, batches)

    def test_memory_growth_requires_same_process_instance(self):
        before = {"processes": [{"role": "worker", "pid": 12, "creation_filetime": 500,
                                 "rss_bytes": 100, "private_bytes": 200, "handles": 15}]}
        after = copy.deepcopy(before)
        after["processes"][0].update(rss_bytes=90, private_bytes=250, handles=17)
        self.assertEqual(bench.memory_growth(before, after)["worker"],
                         {"rss_bytes_delta": -10, "private_bytes_delta": 50, "handles_delta": 2})
        after["processes"][0]["creation_filetime"] = 501
        with self.assertRaises(bench.BenchmarkError):
            bench.memory_growth(before, after)


class MemoryLog:
    def __init__(self):
        self.rows = []

    def append(self, row):
        self.rows.append(copy.deepcopy(row))


class SyntheticHealth:
    def __init__(self):
        self.lock = threading.Lock()
        self.counters = dict.fromkeys(("admitted", "completed", "cancelled", "disconnect_cancellations"), 0)
        self.recent = []

    def retire(self, row):
        with self.lock:
            self.counters["admitted"] += 1
            self.counters["completed" if row["action"] == "complete" else "cancelled"] += 1
            self.counters["disconnect_cancellations"] += row["action"] == "disconnect"
            self.recent.append({"id": row["id"], "status": "completed" if row["action"] == "complete" else "disconnect"})

    def idle(self, label):
        with self.lock:
            return {"counters": dict(self.counters), "recent_requests": copy.deepcopy(self.recent[-64:])}


class SyntheticSampler:
    def __init__(self):
        self.phase = "ready"

    def check(self):
        pass

    def snapshot(self, label):
        return {"phase": self.phase, "label": label, "processes": [
            {"role": role, "pid": index + 1, "creation_filetime": 100,
             "rss_bytes": 1000, "private_bytes": 2000, "handles": 10}
            for index, role in enumerate(("server", "worker"))]}


class SyntheticRunner(bench.Runner):
    """Use real batch/reference validation without sockets, processes or a GPU."""

    def __init__(self, directory):
        self.directory = directory
        self.args = SimpleNamespace(sessions=4, seed=42, model_id="qwen3", new_tokens=2, cancel_new_tokens=4,
                                    enable_thinking=False, temperature=0, profiles=["mixed"], concurrency=[2, 4],
                                    warmup=2, rounds=1, lifecycle_width=4, lifecycle_requests=512,
                                    max_private_growth_mib=None, max_handle_growth=None, probe_cancel=False)
        self.prompts = {"short": "Short prompt", "long": "A distinct longer prompt"}
        self.health, self.sampler = SyntheticHealth(), SyntheticSampler()
        self.registry = bench.SocketRegistry()
        self.batch_log, self.idle_log = MemoryLog(), MemoryLog()
        self.requests, self.batches = [], []
        self.references, self.ids = {}, set()
        self.next_id = 0
        self.lock = threading.Lock()
        self.corrupt = set()

    def request(self, spec, barrier):
        payload = self.payload(spec["prompt_kind"], spec["seed_lane"], spec["action"])
        key = hashlib.sha256(bench.encoded_json(payload)).hexdigest()
        barrier.wait(timeout=5)
        started = time.perf_counter_ns()
        barrier.wait(timeout=5)  # All offered starts precede any synthetic end.
        ended = time.perf_counter_ns()
        completed = spec["action"] == "complete"
        usage = {"prompt_tokens": 4 if spec["prompt_kind"] == "short" else 20,
                 "completion_tokens": payload["max_tokens"]} if completed else None
        if usage:
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        latency = (ended - started) / 1e9
        row = {**spec, "id": "response-" + spec["label"], "payload_sha256": key,
               "start_ns": started, "end_ns": ended, "outcome": "completed" if completed else "disconnected",
               "usage": usage, "finish_reason": "length" if completed else None,
               "content_sha256": "corrupted" if (spec["phase"], spec["seed_lane"]) in self.corrupt else key,
               "reasoning_sha256": "empty", "ttft_seconds": latency / 2,
               "request_latency_seconds": latency if completed else None, "content_event_intervals_seconds": []}
        if completed:
            row["completion_tokens_per_second"] = usage["completion_tokens"] / latency
        with self.lock:
            self.requests.append(row)
        self.health.retire(row)
        return row


class IndependentReferenceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def test_every_prompt_seed_runs_alone_before_concurrency_without_width_one_selected(self):
        runner = SyntheticRunner(self.directory)
        with contextlib.redirect_stdout(io.StringIO()):
            runner.benchmark()
        reference_rows = [row for row in runner.requests if row["phase"] == "reference"]
        self.assertEqual(len(reference_rows), 8)
        self.assertEqual({(row["prompt_kind"], row["seed_lane"]) for row in reference_rows},
                         {(kind, lane) for kind in ("short", "long") for lane in range(4)})
        self.assertTrue(all(row["width"] == 1 and row["lane"] == 0 for row in reference_rows))
        self.assertTrue(all(row["phase"] == "reference" for row in runner.requests[:8]))
        reference_labels = {row["label"] for row in reference_rows}
        self.assertTrue(all(set(batch["reference_labels"].values()) <= reference_labels for batch in runner.batches))
        measured = bench.summarize_measurements(runner.requests, runner.batches)
        self.assertEqual(sum(row["requests"] for row in measured), 6)
        self.assertEqual({row["concurrency"] for row in measured}, {2, 4})

    def test_concurrent_warmup_cannot_install_its_own_reference(self):
        runner = SyntheticRunner(self.directory)
        with self.assertRaisesRegex(bench.BenchmarkError, "no independent single-request reference"):
            runner.batch("warmup", "mixed", 2, 0)
        self.assertEqual(runner.references, {})

    def test_repeatably_wrong_concurrent_seed_is_rejected(self):
        runner = SyntheticRunner(self.directory)
        with contextlib.redirect_stdout(io.StringIO()):
            runner.ensure_references(("mixed",), 4)
        runner.corrupt.add(("warmup", 3))
        with self.assertRaisesRegex(bench.BenchmarkError, "changed from its independent single-request reference"):
            runner.batch("warmup", "mixed", 4, 0)

    def test_lifecycle_output_after_a_cancelled_wave_uses_independent_reference(self):
        runner = SyntheticRunner(self.directory)
        with contextlib.redirect_stdout(io.StringIO()):
            runner.ensure_references(("mixed",), 4)
            first = runner.batch("lifecycle", "mixed", 4, 0, lifecycle=True, request_offset=0)
        self.assertEqual(first["disconnected"], 2)
        runner.corrupt.add(("lifecycle", 0))
        with self.assertRaisesRegex(bench.BenchmarkError, "changed from its independent single-request reference"):
            runner.batch("lifecycle", "mixed", 4, 1, lifecycle=True, request_offset=4)

    def test_lifecycle_warmup_does_not_skip_semantic_validation(self):
        runner = SyntheticRunner(self.directory)
        with contextlib.redirect_stdout(io.StringIO()):
            runner.ensure_references(("mixed",), 2)
        runner.corrupt.add(("lifecycle_warmup", 0))
        with self.assertRaisesRegex(bench.BenchmarkError, "changed from its independent single-request reference"):
            runner.batch("lifecycle_warmup", "mixed", 2, 0, lifecycle=True)


class BoundedProbeTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.runner = SyntheticRunner(Path(directory.name))
        self.runner.args.sessions = 1

    def run_probe(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.runner.probe()

    def test_default_probe_remains_two_complete_short_requests(self):
        self.run_probe()
        rows = self.runner.requests
        self.assertEqual([r["phase"] for r in rows], ["reference", "probe"])
        self.assertTrue(all(r["action"] == "complete" and r["prompt_kind"] == "short" and r["width"] == 1
                            for r in rows))

    def test_cancel_probe_has_one_disconnect_then_reuses_independent_reference(self):
        self.runner.args.probe_cancel = True
        self.run_probe()
        rows = self.runner.requests
        self.assertEqual([r["phase"] for r in rows], ["reference", "probe_cancel", "probe_after_cancel"])
        self.assertEqual([r["action"] for r in rows], ["complete", "disconnect", "complete"])
        self.assertTrue(all(r["width"] == 1 and r["prompt_kind"] == "short" for r in rows))
        self.assertEqual(self.runner.semantic(rows[0]), self.runner.semantic(rows[2]))
        self.assertEqual(self.runner.batches[-1]["reference_labels"], {rows[2]["label"]: rows[0]["label"]})
        self.assertEqual(self.runner.health.counters,
                         {"admitted": 3, "completed": 2, "cancelled": 1, "disconnect_cancellations": 1})

    def test_corrupt_output_after_cancel_is_rejected(self):
        self.runner.args.probe_cancel = True
        self.runner.corrupt.add(("probe_after_cancel", 0))
        with self.assertRaisesRegex(bench.BenchmarkError, "changed from its independent single-request reference"):
            self.run_probe()

    def test_completed_cancellation_target_stops_before_reuse(self):
        self.runner.args.probe_cancel = True
        original_idle = self.runner.health.idle

        def target_completed(label):
            value = original_idle(label)
            if label == "after:probe_cancel":
                value["counters"]["completed"] += 1
                value["counters"]["cancelled"] -= 1
                value["counters"]["disconnect_cancellations"] -= 1
            return value

        self.runner.health.idle = target_completed
        with self.assertRaisesRegex(bench.BenchmarkError, "count differs from its planned outcomes"):
            self.run_probe()
        self.assertEqual(len(self.runner.requests), 2)

    def test_timeout_retirement_cannot_pass_as_intentional_disconnect(self):
        self.runner.args.probe_cancel = True
        original_idle = self.runner.health.idle

        def target_timed_out(label):
            value = original_idle(label)
            if label == "after:probe_cancel":
                value["recent_requests"][-1]["status"] = "request_timeout"
            return value

        self.runner.health.idle = target_timed_out
        with self.assertRaisesRegex(bench.BenchmarkError, "Request retirement was not disconnect"):
            self.run_probe()
        self.assertEqual(len(self.runner.requests), 2)


class StabilityContainmentTests(unittest.TestCase):
    def test_guard_rejection_happens_before_any_process_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            server, worker = root / "server.exe", root / "worker.exe"
            server.write_bytes(b"")
            worker.write_bytes(b"")
            guard = mock.MagicMock()
            guard.__enter__.side_effect = RuntimeError("GPU crash quarantine")
            with mock.patch.object(bench, "GpuRunGuard", return_value=guard), \
                 mock.patch.object(bench, "WindowsJobProcess") as process, \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = bench.main(["--server", str(server), "--worker", str(worker),
                                   "--model", str(root), "--output", str(root / "out")])
            self.assertEqual(code, 1)
            process.assert_not_called()
            guard.finish.assert_not_called()
            saved = list((root / "out").glob("run-*/result.json"))
            self.assertEqual(len(saved), 1)
            self.assertFalse(bench.strict_json(saved[0].read_bytes())["passed"])

    def test_watchdog_signals_shutdown_and_only_terminates_owned_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            terminated = []
            process = SimpleNamespace(poll=lambda: None, owned_pids=lambda: [123],
                                      terminate_owned=lambda: terminated.append(123) or [123])
            watchdog = bench.RunWatchdog(process, root / "shutdown.flag", root, 0.01, grace=0.01)
            watchdog.start()
            watchdog.thread.join(timeout=3)
            watchdog.stop()
            self.assertTrue((root / "shutdown.flag").exists())
            self.assertEqual(terminated, [123])
            with self.assertRaisesRegex(bench.BenchmarkError, "deadline"):
                watchdog.check()

    def test_watchdog_still_contains_when_diagnostics_fail(self):
        terminated = []
        process = SimpleNamespace(terminate_owned=lambda: terminated.append(True))
        watchdog = bench.RunWatchdog(process, Path("unused.flag"), Path("unused"), 0.01, grace=0.01)
        with mock.patch.object(bench, "write_json", side_effect=OSError("disk failed")):
            watchdog.start()
            watchdog.thread.join(timeout=3)
            watchdog.stop()
        self.assertEqual(terminated, [True])
        self.assertIn("disk failed", watchdog.error)

    def test_atomic_report_failure_preserves_last_complete_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            bench.write_json(path, {"state": "started"})
            with mock.patch.object(bench.os, "replace", side_effect=OSError("rename failed")), \
                 self.assertRaisesRegex(OSError, "rename failed"):
                bench.write_json(path, {"state": "finished"})
            self.assertEqual(bench.strict_json(path.read_bytes()), {"state": "started"})


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeProcess:
    def __init__(self, clock, finish=None, orphan=False, interrupt=False):
        self.clock, self.finish, self.orphan, self.interrupt = clock, finish, orphan, interrupt
        self.forced = False
        self.closed = 0
        self.actions = []

    def poll(self):
        if self.forced:
            return 1
        return 0 if self.finish is not None and self.clock() >= self.finish else None

    def owned_pids(self):
        return [] if self.forced else [10, 11] if self.poll() is None else [11] if self.orphan else []

    def wait(self, seconds):
        self.actions.append(("wait", self.clock(), seconds))
        self.clock.sleep(seconds)
        if self.interrupt:
            self.interrupt = False
            raise KeyboardInterrupt()
        if self.poll() is None:
            raise TimeoutError()
        return self.poll()

    def terminate_owned(self):
        values = self.owned_pids()
        self.actions.append(("terminate_owned", self.clock(), values))
        self.forced = True
        return values

    def close(self):
        self.closed += 1


class FakeFlag:
    def __init__(self, process, fail=False):
        self.process, self.fail = process, fail

    def write_text(self, text, encoding):
        self.process.actions.append(("shutdown_file", self.process.clock()))
        if self.fail:
            raise OSError("Cannot write the shutdown flag")


class CleanupTests(unittest.TestCase):
    def stop(self, process, timeout=1, fail_flag=False):
        return bench.stop_owned_process(process, FakeFlag(process, fail_flag), timeout,
                                        clock=process.clock, sleeper=process.clock.sleep)

    def test_graceful_shutdown_first_and_close_exactly_once(self):
        process = FakeProcess(FakeClock(), finish=0.2)
        result = self.stop(process)
        self.assertFalse(result["forced_termination"])
        self.assertTrue(result["job_empty"])
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(process.actions[0][0], "shutdown_file")
        self.assertEqual(process.closed, 1)

    def test_deadline_terminates_only_owned_tree_after_grace(self):
        process = FakeProcess(FakeClock())
        result = self.stop(process, timeout=0.5)
        self.assertTrue(result["forced_termination"])
        self.assertEqual(result["forced_pids"], [10, 11])
        self.assertGreaterEqual(next(item[1] for item in process.actions if item[0] == "terminate_owned"), 0.5)
        self.assertEqual(process.closed, 1)

    def test_exited_server_does_not_leave_owned_worker(self):
        process = FakeProcess(FakeClock(), finish=0, orphan=True)
        result = self.stop(process, timeout=0.2)
        self.assertFalse(result["shutdown_file_written"])
        self.assertEqual(result["forced_pids"], [11])
        self.assertTrue(result["job_empty"])

    def test_interrupt_preserves_grace_period(self):
        process = FakeProcess(FakeClock(), finish=0.3, interrupt=True)
        result = self.stop(process)
        self.assertTrue(result["interrupted_during_shutdown"])
        self.assertFalse(result["forced_termination"])
        self.assertEqual(result["exit_code"], 0)

    def test_shutdown_flag_error_still_contains_owned_processes(self):
        process = FakeProcess(FakeClock())
        result = self.stop(process, fail_flag=True)
        self.assertTrue(result["forced_termination"])
        self.assertTrue(result["job_empty"])
        self.assertEqual(len(result["errors"]), 1)
        self.assertEqual(process.closed, 1)


class FinalizationTests(unittest.TestCase):
    """Real temporary guard files, injected snapshots, and no GPU processes."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "run"
        self.directory.mkdir()
        self.snapshot = {"schema_version": 1, "captured_at": "2026-09-08T00:00:00+00:00",
                         "host": "cpu-test-host", "boot_id": "boot-one", "events": [], "dumps": [],
                         "event_log_heads": {"System": 1000, "Application": 2000}, "read_errors": []}
        self.guard = self.make_guard()
        self.guard.__enter__()
        self.addCleanup(self.guard.lock.close)
        self.result = {"passed": True}
        self.process = FakeProcess(FakeClock(), finish=0)
        self.args = SimpleNamespace(shutdown_timeout=0.01, model_id="qwen3", backend="qwen3_nf4",
                                    decode_batch_size=1)
        self.report_file = self.directory / "final-report.json"
        bench.write_json(self.report_file, {
            "model": "qwen3", "backend": "qwen3_nf4", "status": "closed", "active_requests": 0,
            "max_decode_batch_size": 1,
            "shutdown": {"graceful": True, "idle": True},
            "ipc": {"status": "closed", "outstanding_requests": 0, "pending_commands": 0},
            "connections": {"active": 0, "accepted": 0, "closed": 0},
            "counters": {"admitted": 0, "completed": 0, "cancelled": 0, "failed": 0, "rejected": 0,
                         "slow_client_cancellations": 0}})

    def make_guard(self):
        return bench.GpuRunGuard("CPU finalization test", state_dir=self.root / "machine-guard",
                                 snapshot_reader=lambda: copy.deepcopy(self.snapshot))

    def finish(self, **kwargs):
        bench.finalize_run(self.args, self.directory, self.result,
                           shutdown_file=self.directory / "shutdown.flag", report_file=self.report_file,
                           process=self.process, guard=self.guard, **kwargs)

    def assert_blocked(self):
        self.assertFalse(self.result["passed"])
        self.assertIsNone(self.guard.lock.file)
        self.assertEqual(self.process.closed, 1)
        with self.assertRaises(guard_module.GuardBlocked):
            with self.make_guard():
                self.fail("Failed finalization must prevent the next GPU run")

    def test_success_persists_result_before_final_guard_commit(self):
        write_json = bench.write_json

        def saved_before_commit(path, value):
            self.assertEqual(self.process.closed, 1)
            self.assertEqual(self.guard.record["status"], "started")
            self.assertIsNotNone(self.guard.lock.file)
            write_json(path, value)

        with mock.patch.object(bench, "write_json", side_effect=saved_before_commit) as write:
            self.finish()
        write.assert_called_once()
        self.assertTrue(self.result["passed"])
        self.assertIsNone(self.guard.lock.file)
        self.assertEqual(self.guard.record["status"], "finished")
        self.assertEqual(self.guard.record["details"]["result_sha256"],
                         bench.file_hash(self.directory / "result.json"))
        self.assertEqual(bench.strict_json((self.directory / "result.json").read_bytes())["gpu_guard_run_id"],
                         self.guard.run_id)

    def test_result_write_failure_closes_job_and_quarantines(self):
        with mock.patch.object(bench, "write_json", side_effect=OSError("result disk failed")):
            self.finish()
        self.assertEqual(self.guard.record["status"], "failed")
        self.assert_blocked()

    def test_result_and_guard_write_failures_preserve_unfinished_run(self):
        with mock.patch.object(bench, "write_json", side_effect=OSError("result disk failed")), \
             mock.patch.object(self.guard.store, "commit", side_effect=OSError("guard disk failed")):
            self.finish()
        persisted = bench.strict_json(self.guard.state_path.read_bytes())["payload"]
        self.assertEqual(persisted["status"], "started")
        self.assert_blocked()

    def test_cleanup_and_diagnostic_failures_do_not_skip_owned_job_or_other_closes(self):
        runner = SimpleNamespace(registry=mock.Mock(), requests=[], batches=[])
        runner.registry.close_all.side_effect = KeyboardInterrupt()
        sampler = SimpleNamespace(stop=mock.Mock(side_effect=OSError("sampler failed")), rows=[{}])
        watchdog = mock.Mock()
        outputs, writers = [mock.Mock(), mock.Mock()], [mock.Mock(), mock.Mock()]
        outputs[0].close.side_effect = OSError("output flush failed")
        writers[0].close.side_effect = OSError("journal flush failed")
        with mock.patch.object(bench, "log_errors", side_effect=PermissionError("logs unreadable")):
            self.finish(runner=runner, sampler=sampler, watchdog=watchdog, outputs=outputs, writers=writers)
        watchdog.stop.assert_called_once()
        watchdog.check.assert_called_once()
        for output in outputs + writers:
            output.close.assert_called_once()
        stages = {item["stage"] for item in self.result["finalization_errors"]}
        self.assertTrue({"request_shutdown", "sampler_stop", "memory_summary", "log_check",
                         "output_close_0", "writer_close_0"}.issubset(stages))
        self.assertEqual(self.result["final_report"]["status"], "closed")
        self.assert_blocked()

    def test_unexpected_cleanup_failure_still_closes_owned_job(self):
        with mock.patch.object(bench, "stop_owned_process", side_effect=RuntimeError("cleanup failed")):
            self.finish()
        self.assert_blocked()

    def test_postflight_fault_overrides_success_and_blocks_next_run(self):
        self.snapshot["events"] = [{"source": "CPU injection", "bugcheck": "0x00000116"}]
        self.finish()
        saved = bench.strict_json((self.directory / "result.json").read_bytes())
        self.assertFalse(saved["passed"])
        self.assertEqual(self.guard.record["status"], "failed")
        self.assert_blocked()

    def test_guard_commit_failure_invalidates_saved_result(self):
        with mock.patch.object(self.guard.store, "commit", side_effect=OSError("guard disk failed")):
            self.finish()
        saved = bench.strict_json((self.directory / "result.json").read_bytes())
        self.assertFalse(saved["passed"])
        self.assertEqual(bench.strict_json(self.guard.state_path.read_bytes())["payload"]["status"], "started")
        self.assert_blocked()

    def test_unexpected_finalization_exception_is_delivered_to_guard(self):
        with mock.patch.object(bench, "utc_now", side_effect=RuntimeError("outside attempt")), \
             self.assertRaisesRegex(RuntimeError, "outside attempt"):
            self.finish()
        self.assertEqual(self.guard.record["details"]["error"], "RuntimeError: outside attempt")
        self.assert_blocked()

    def test_lock_release_error_after_success_does_not_rewrite_committed_report(self):
        close = self.guard.lock.close

        def close_then_raise():
            close()
            raise OSError("post-commit release error")

        with mock.patch.object(self.guard.lock, "close", side_effect=close_then_raise), \
             mock.patch.object(bench, "write_json", wraps=bench.write_json) as write:
            self.finish()
        write.assert_called_once()
        self.assertTrue(self.result["passed"])
        self.assertIsNone(self.guard.lock.file)
        self.assertEqual(self.guard.record["status"], "finished")
        self.assertEqual(self.result["post_commit_errors"][0]["stage"], "gpu_guard_close")
        self.assertEqual(self.guard.record["details"]["result_sha256"],
                         bench.file_hash(self.directory / "result.json"))

    def test_trace_checked_after_owned_job_cleanup_before_guard_success(self):
        self.args.execution_trace = True
        self.process.worker_pid = 123
        write_execution_trace(self.directory)
        analyze = bench.analyze_execution_trace

        def after_cleanup(path):
            self.assertEqual(self.process.closed, 1)
            self.assertEqual(self.guard.record["status"], "started")
            return analyze(path)

        with mock.patch.object(bench, "analyze_execution_trace", side_effect=after_cleanup):
            self.finish()
        self.assertTrue(self.result["passed"])
        self.assertTrue(self.result["execution_trace"]["complete"])
        self.assertEqual(self.guard.record["status"], "finished")

    def test_partial_trace_blocks_success_without_skipping_cleanup(self):
        self.args.execution_trace = True
        self.process.worker_pid = 123
        write_execution_trace(self.directory, complete=False)
        self.finish()
        self.assertFalse(self.result["execution_trace"]["complete"])
        self.assert_blocked()

    def test_complete_trace_does_not_override_worker_shutdown_failure(self):
        self.args.execution_trace = True
        self.process.worker_pid = 123
        write_execution_trace(self.directory)
        report = bench.strict_json(self.report_file.read_bytes())
        # A final close line may be visible even though its flush failed. The
        # Worker's nonzero exit reaches the host as a failed graceful shutdown.
        report["shutdown"]["graceful"] = False
        bench.write_json(self.report_file, report)
        self.finish()
        self.assertTrue(self.result["execution_trace"]["complete"])
        self.assertEqual(self.result["finalization_errors"][0]["stage"], "final_report")
        self.assert_blocked()

    def test_trace_reader_exception_cannot_skip_cleanup_or_commit_success(self):
        self.args.execution_trace = True
        self.process.worker_pid = 123
        write_execution_trace(self.directory)
        with mock.patch.object(bench, "analyze_execution_trace", side_effect=PermissionError("trace unavailable")):
            self.finish()
        self.assertEqual(self.result["finalization_errors"][0]["stage"], "execution_trace")
        self.assert_blocked()


class ExecutionTraceRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def test_disabled_trace_scrubs_inherited_path_without_creating_directory(self):
        inherited = {"AILA_EXECUTION_TRACE_DIR": "old", "aila_execution_trace_dir": "other", "KEEP": "value"}
        with mock.patch.object(bench.os, "environ", inherited):
            result = bench.child_environment(SimpleNamespace(execution_trace=False), self.directory)
        self.assertNotIn("AILA_EXECUTION_TRACE_DIR", result)
        self.assertNotIn("aila_execution_trace_dir", result)
        self.assertEqual(inherited["AILA_EXECUTION_TRACE_DIR"], "old")
        self.assertEqual(result["KEEP"], "value")
        self.assertFalse((self.directory / "execution-trace").exists())

    def test_enabled_trace_uses_fresh_run_directory(self):
        args = SimpleNamespace(execution_trace=True)
        environment = bench.child_environment(args, self.directory)
        self.assertEqual(environment["AILA_EXECUTION_TRACE_DIR"], str((self.directory / "execution-trace").resolve()))
        with self.assertRaises(FileExistsError):
            bench.child_environment(args, self.directory)

    def test_missing_trace_and_worker_identity_are_failures(self):
        self.assertFalse(bench.execution_trace_summary(self.directory, 123)["complete"])
        write_execution_trace(self.directory)
        self.assertFalse(bench.execution_trace_summary(self.directory, None)["complete"])
        self.assertFalse(bench.execution_trace_summary(self.directory, 124)["complete"])
        self.assertTrue(bench.execution_trace_summary(self.directory, 123)["complete"])

    def test_foreign_file_and_duplicate_trace_identity_rejected(self):
        path = write_execution_trace(self.directory)
        other = path.with_name("execution-123-1-2.jsonl")
        other.write_bytes(path.read_bytes())
        self.assertFalse(bench.execution_trace_summary(self.directory, 123)["complete"])
        other.unlink()
        path.with_name("unexpected.txt").write_text("not evidence", encoding="utf-8")
        self.assertFalse(bench.execution_trace_summary(self.directory, 123)["complete"])


class CliAndFinalReportTests(unittest.TestCase):
    def arguments(self, *extra):
        return bench.parse_arguments(["--server", "server.exe", "--worker", "worker.exe", "--model", "model",
                                      "--output", "output", *extra])

    def test_defaults_and_context_only_benchmark(self):
        default = self.arguments()
        self.assertEqual(default.mode, "probe")
        self.assertFalse(default.execution_trace)
        self.assertTrue(self.arguments("--execution-trace").execution_trace)
        self.assertEqual(default.concurrency, [1])
        self.assertEqual(default.context, 512)
        self.assertEqual(default.sessions, 1)
        self.assertEqual(default.new_tokens, 8)
        self.assertEqual(default.max_run_seconds, 60)
        self.assertEqual(default.lifecycle_requests, 512)
        self.assertEqual(self.arguments("--backend", "qwen35_nf4").model_id, "qwen3.5")
        self.assertEqual(self.arguments("--backend", "qwen35_bf16").model_id, "qwen3.5")
        self.assertEqual(self.arguments("--mode", "benchmark", "--allow-stress", "--context", "64", "--new-tokens", "8").context, 64)

    def test_execution_trace_cannot_silently_instrument_stress(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.arguments("--mode", "benchmark", "--allow-stress", "--execution-trace")

    def test_decode_batch_bound_is_opt_in_bounded_and_matches_the_service(self):
        self.assertEqual(self.arguments().decode_batch_size, 1)
        self.assertEqual(self.arguments("--mode", "benchmark", "--allow-stress", "--sessions", "4",
                                        "--decode-batch-size", "4").decode_batch_size, 4)
        for extra in (("--decode-batch-size", "0"), ("--decode-batch-size", "5"),
                      ("--mode", "benchmark", "--allow-stress", "--sessions", "2", "--decode-batch-size", "4")):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.arguments(*extra)

    def test_service_reports_a_different_decode_batch_bound_is_rejected(self):
        report = {"model": "qwen3", "backend": "qwen3_nf4", "status": "closed", "active_requests": 0,
                  "max_decode_batch_size": 1,
                  "shutdown": {"graceful": True, "idle": True},
                  "ipc": {"status": "closed", "outstanding_requests": 0, "pending_commands": 0},
                  "connections": {"active": 0, "accepted": 8, "closed": 8},
                  "counters": {"admitted": 8, "completed": 8, "cancelled": 0, "failed": 0, "rejected": 0,
                               "slow_client_cancellations": 0}}
        bench.validate_final_report(report, "qwen3", "qwen3_nf4", 8, 1)
        with self.assertRaises(bench.BenchmarkError):
            bench.validate_final_report(report, "qwen3", "qwen3_nf4", 8, 2)

    def test_invalid_unwarmed_or_small_lifecycle_runs_are_rejected(self):
        invalid = [("--warmup", "1"), ("--mode", "both"), ("--sample-interval", "nan"),
                   ("--concurrency", "1", "1"), ("--seed", str((1 << 64) - 1)), ("--model-id", "")]
        for extra in invalid:
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                self.arguments(*extra)
            self.assertEqual(error.exception.code, 2)

    def test_lifecycle_measured_count_must_cross_the_handle_limit_without_warmup_credit(self):
        self.assertEqual(self.arguments("--mode", "lifecycle", "--allow-stress", "--lifecycle-requests", "257").lifecycle_requests, 257)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.arguments("--mode", "lifecycle", "--allow-stress", "--warmup", "100", "--lifecycle-requests", "256")
        self.assertEqual(self.arguments("--mode", "benchmark", "--allow-stress", "--lifecycle-requests", "1").lifecycle_requests, 1)

    def test_probe_and_deadline_cannot_silently_expand(self):
        for extra in (("--sessions", "4"), ("--context", "2048"), ("--new-tokens", "32"),
                      ("--rounds", "2"), ("--profiles", "mixed"), ("--max-run-seconds", "61"),
                      ("--max-run-seconds", "nan"), ("--max-device-bytes", "0")):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.arguments(*extra)

    def test_cancellation_probe_is_explicit_small_and_traced(self):
        default = self.arguments()
        self.assertFalse(default.probe_cancel)
        self.assertEqual(default.cancel_new_tokens, 128)
        args = self.arguments("--probe-cancel", "--execution-trace")
        self.assertTrue(args.probe_cancel)
        self.assertEqual((args.mode, args.sessions, args.new_tokens, args.cancel_new_tokens), ("probe", 1, 8, 16))
        self.assertEqual(args.max_run_seconds, 60)
        self.assertEqual(self.arguments("--mode", "lifecycle", "--allow-stress").cancel_new_tokens, 128)

    def test_cancellation_probe_rejects_untraced_or_expanded_work(self):
        invalid = [(), ("--execution-trace", "--cancel-new-tokens", "17"),
                   ("--execution-trace", "--cancel-new-tokens", "8"),
                   ("--execution-trace", "--new-tokens", "16"),
                   ("--execution-trace", "--max-run-seconds", "61"),
                   ("--execution-trace", "--sessions", "2"),
                   ("--execution-trace", "--mode", "benchmark", "--allow-stress")]
        for extra in invalid:
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.arguments("--probe-cancel", *extra)

    def test_final_resources_and_request_accounting(self):
        report = {"model": "qwen3", "backend": "qwen3_nf4", "status": "closed", "active_requests": 0,
                  "max_decode_batch_size": 1,
                  "shutdown": {"graceful": True, "idle": True},
                  "ipc": {"status": "closed", "outstanding_requests": 0, "pending_commands": 0},
                  "connections": {"active": 0, "accepted": 20, "closed": 20},
                  "counters": {"admitted": 8, "completed": 4, "cancelled": 4, "failed": 0, "rejected": 0,
                               "slow_client_cancellations": 0}}
        bench.validate_final_report(report, "qwen3", "qwen3_nf4", 8)
        for group, key, value in (("ipc", "outstanding_requests", 1), ("connections", "active", 1),
                                   ("connections", "closed", 19), ("counters", "admitted", 9)):
            broken = copy.deepcopy(report)
            broken[group][key] = value
            with self.subTest(group=group, key=key), self.assertRaises(bench.BenchmarkError):
                bench.validate_final_report(broken, "qwen3", "qwen3_nf4", 8)


@unittest.skipUnless(os.name == "nt", "Windows Job and process counters are platform-specific")
class NativeWindowsOwnershipTests(unittest.TestCase):
    def test_python_only_job_drains_after_shutdown_file_without_force(self):
        parent = Path(__file__).resolve().parents[2] / "tmp" / "refactor" / "service-benchmark-cpu"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="graceful-python-", dir=parent) as temporary:
            directory = Path(temporary).resolve()
            self.assertTrue(directory.is_relative_to(parent.resolve()))
            process = None
            with (directory / "stdout.log").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr:
                try:
                    executable = Path(sys.executable).resolve()
                    child = "import pathlib,sys,time\nf=pathlib.Path(sys.argv[1])\nwhile not f.exists(): time.sleep(0.02)"
                    script = ("import pathlib,subprocess,sys,time\n"
                              f"p=subprocess.Popen([sys.executable,'-I','-S','-c',{child!r},sys.argv[1]],creationflags=0x08000000)\n"
                              "f=pathlib.Path(sys.argv[1])\nwhile not f.exists(): time.sleep(0.02)\np.wait(timeout=3)\n")
                    flag = directory / "shutdown.flag"
                    process = bench.WindowsJobProcess([str(executable), "-I", "-S", "-c", script, str(flag)],
                                                       directory, stdout, stderr, dict(os.environ))
                    deadline = time.monotonic() + 5
                    while process.find_worker(executable) is None and time.monotonic() < deadline:
                        time.sleep(0.02)
                    self.assertIsNotNone(process.worker_pid)
                    result = bench.stop_owned_process(process, flag, 5)
                    self.assertFalse(result["forced_termination"])
                    self.assertTrue(result["shutdown_file_written"])
                    self.assertTrue(result["job_empty"])
                    self.assertEqual(result["exit_code"], 0)
                    self.assertFalse(result["errors"])
                finally:
                    if process is not None and not process.closed:
                        bench.stop_owned_process(process, directory / "shutdown.flag", 0.1)

    def test_python_only_owned_parent_child_sampling_and_cleanup(self):
        # No Aila binary, runtime, model or GPU is loaded. The child belongs to
        # the Job before it can run, and cleanup retains that Job throughout.
        parent = Path(__file__).resolve().parents[2] / "tmp" / "refactor" / "service-benchmark-cpu"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="owned-python-", dir=parent) as temporary:
            directory = Path(temporary).resolve()
            self.assertTrue(directory.is_relative_to(parent.resolve()))
            process = None
            with (directory / "stdout.log").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr:
                try:
                    script = ("import subprocess,sys,time; "
                              "p=subprocess.Popen([sys.executable,'-I','-S','-c','import time; time.sleep(30)'],"
                              "creationflags=0x08000000); print(p.pid,flush=True); time.sleep(30)")
                    executable = Path(sys.executable).resolve()
                    process = bench.WindowsJobProcess([str(executable), "-I", "-S", "-c", script],
                                                       directory, stdout, stderr, dict(os.environ))
                    deadline = time.monotonic() + 5
                    while process.find_worker(executable) is None and time.monotonic() < deadline:
                        time.sleep(0.02)
                    self.assertIsNotNone(process.worker_pid)
                    self.assertNotEqual(process.pid, process.worker_pid)
                    # Windows may also attach its hidden console helper to the
                    # owned Job. Membership, not a global two-process guess,
                    # defines the cleanup boundary.
                    self.assertTrue({process.pid, process.worker_pid}.issubset(set(process.owned_pids())))
                    samples = process.sample()
                    self.assertEqual({item["role"] for item in samples}, {"server", "worker"})
                    for item in samples:
                        self.assertGreater(item["rss_bytes"], 0)
                        self.assertGreater(item["private_bytes"], 0)
                        self.assertGreater(item["handles"], 0)
                        self.assertGreater(item["creation_filetime"], 0)
                    # The Python helpers intentionally do not implement the
                    # Server shutdown-file protocol, so force is expected here.
                    result = bench.stop_owned_process(process, directory / "shutdown.flag", 0.1)
                    self.assertTrue(result["forced_termination"])
                    self.assertTrue(result["job_empty"])
                    self.assertTrue({item["pid"] for item in samples}.issubset(set(result["forced_pids"])))
                    self.assertFalse(result["errors"])
                finally:
                    if process is not None and not process.closed:
                        bench.stop_owned_process(process, directory / "cleanup.flag", 0.1)

    def test_concurrency_and_lifecycle_width_three_accepted(self):
        args = bench.parse_arguments([
            "--server", "stub-server.exe", "--worker", "stub-worker.exe",
            "--output", "tmp/stub-out", "--mode", "benchmark", "--backend", "qwen3_nf4", "--model", "stub-model",
            "--allow-stress", "--sessions", "3", "--concurrency", "1", "3",
            "--lifecycle-width", "3"
        ])
        self.assertEqual(args.concurrency, [1, 3])
        self.assertEqual(args.lifecycle_width, 3)

    def test_query_dxgi_video_memory_returns_expected_structure(self):
        vram = bench.query_dxgi_video_memory()
        if vram is not None:
            self.assertIn("local_vram_budget_bytes", vram)
            self.assertIn("local_vram_usage_bytes", vram)
            self.assertGreater(vram["local_vram_budget_bytes"], 0)
            self.assertGreaterEqual(vram["local_vram_usage_bytes"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
