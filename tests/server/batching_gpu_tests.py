"""Bounded HTTP functionality checks for native decode batching.

The scalar process supplies an exact singleton JSON/SSE reference. A second
process enables max_decode_batch_size=4 and checks the same singleton plus
finite 1/2/4-way streams, disconnect isolation and active graceful shutdown.
Concurrent output is validated independently; it is not compared token-for-token
with the scalar sample. Native batch execution, logits/RNG correctness and OS
stability are separate gates owned by the outer guarded test run.

Only the standard library is required. No inference request is retried. The
output directory must not exist, and every process retains its own launch,
stdout/stderr, request bodies, raw JSON/SSE bodies, health samples and report.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
from pathlib import Path
import threading
import time
import traceback

from socket_tests import (
    MODEL_IDS, Runner, ServerProcess, check, integer, json_bytes, utc_now, write_json,
)


MAX_DEVICE_BYTES = 8 * 1024**3
MAX_PINNED_HOST_BYTES = 16 * 1024**2
MAX_SESSION_HOST_BYTES = 256 * 1024**2
SINGLETON_PROMPT = "Explain in several sentences why leaves are green."
ROWS = (
    ("Explain several practical ways to save water at home.", 32, 701),
    ("Describe in detail how a rainbow forms after rain.", 24, 809),
    ("用几句话介绍阅读纸质书的好处。", 20, 907),
    ("Explain several useful properties of the number seven.", 16, 1009),
)
SCOPE = (
    "Finite HTTP configuration, singleton B=1 equivalence, concurrent stream "
    "routing, terminal/usage integrity, disconnect isolation and cleanup. "
    "No cross-width token-equality, numerical, throughput or long-term stability claim. "
    "The outer guard must separately prove native decode_batch execution and OS health."
)


class EvidenceRunner(Runner):
    """Keep the original protocol assertions and add per-exchange evidence."""

    def __init__(self, args, process, directory, result):
        super().__init__(args, process, directory, result)
        self.evidence_lock = threading.Lock()
        self.exchange_number = 0
        self.id_lock = threading.Lock()

    def json_call(self, label, path, method="GET", payload=None, expected=200,
                  timeout=None, save=True):
        with self.evidence_lock:
            self.exchange_number += 1
            artifact_label = f"{self.exchange_number:06d}-{label}"
        write_json(self.responses / f"{artifact_label}.request.json", {
            "at": utc_now(), "method": method, "path": path, "payload": payload,
        })
        if payload is not None:
            (self.responses / f"{artifact_label}.request.body.json").write_bytes(json_bytes(payload))
        try:
            # In particular, retain each full health body, not just its latency
            # or the last snapshot. No health result overwrites an earlier one.
            value, timing = super().json_call(
                artifact_label, path, method, payload, expected, timeout, save=True,
            )
            write_json(self.responses / f"{artifact_label}.exchange.json", timing)
            return value, timing
        except BaseException as error:
            write_json(self.responses / f"{artifact_label}.exchange.json", {
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            })
            raise

    def health(self, allow_fault=False):
        value = super().health(allow_fault=allow_fault)
        if value["status"] == "ok":
            self.validate_budgets(value)
        return value

    def validate_budgets(self, value):
        check(value.get("max_decode_batch_size") == self.args.decode_batch_size,
              "The server did not retain its requested batch bound")
        memory = value.get("memory")
        check(isinstance(memory, dict), "Health/report omitted model memory limits")
        for name, requested in (
            ("max_device_bytes", MAX_DEVICE_BYTES),
            ("max_pinned_host_bytes", MAX_PINNED_HOST_BYTES),
            ("max_session_host_bytes", MAX_SESSION_HOST_BYTES),
        ):
            check(integer(memory.get(name), name) == requested,
                  f"Applied {name} differs from the fixture's fixed limit")

    def validate_cleanup(self, cleanup):
        report = super().validate_cleanup(cleanup)
        self.validate_budgets(report)
        return report

    def remember_id(self, identity):
        with self.id_lock:
            super().remember_id(identity)

    def open_stream(self, label, payload):
        # These bytes are serialized exactly as SseClient sends them.
        (self.responses / f"{label}.request.body.json").write_bytes(json_bytes(payload))
        try:
            client = self.stream(label, payload)
            write_json(self.responses / f"{label}.headers.json", {
                "status": client.response.status,
                "version": client.response.version,
                "headers": client.response.getheaders(),
                "header_request_id": client.header_id,
            })
            client.first_role()
            self.remember_id(client.id)
            return client
        except BaseException as error:
            write_json(self.responses / f"{label}.error.json", {
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            })
            raise

    def close_stream(self, client, disposition):
        client.close()  # Flush the raw SSE body even when deliberately interrupted.
        write_json(self.responses / f"{client.label}.stream.json", {
            "disposition": disposition,
            "summary": client.summary(),
            "semantic": client.semantic(),
            "role_count": client.role_count,
            "finish_count": client.finish_count,
            "usage_count": client.usage_count,
            "done_count": client.done_count,
        })

    def complete_stream(self, label, payload):
        client = self.open_stream(label, payload)
        disposition = "failed"
        try:
            summary = client.complete()
            semantic = client.semantic()
            validate_length(semantic, payload["max_tokens"])
            check(summary["content_frames"] > 0, "Successful SSE had no text or reasoning delta")
            check(summary["first_content_at"] < summary["done_at"],
                  "The SSE terminal preceded its content")
            disposition = "completed"
            return {"summary": summary, "semantic": semantic}
        finally:
            self.close_stream(client, disposition)


def validate_length(semantic, maximum):
    # A normal model EOS is allowed. A length finish must account for every
    # requested token; neither a silent short stream nor duplicate usage passes.
    if semantic["finish_reason"] == "length":
        check(semantic["usage"]["completion_tokens"] == maximum,
              "Length completion lost generated tokens")


def expect_counters(before, after, *, admitted, completed, cancelled=0, disconnect=0):
    wanted = {
        "admitted": admitted, "completed": completed, "cancelled": cancelled,
        "disconnect_cancellations": disconnect, "failed": 0, "rejected": 0,
        "slow_client_cancellations": 0,
    }
    for key, expected in wanted.items():
        delta = integer(after["counters"].get(key), key) - integer(before["counters"].get(key), key)
        check(delta == expected, f"Unexpected {key} delta: {delta}, expected {expected}")


def completed_records(runner, health, summaries):
    records = []
    for summary in summaries:
        matches = [item for item in health["recent_requests"] if item["id"] == summary["id"]]
        check(len(matches) == 1, f"Missing or duplicated retained request {summary['id']}")
        record = matches[0]
        check(record["status"] == "completed", f"Successful request retired as {record['status']}")
        check(record["generated_tokens"] == summary["usage"]["completion_tokens"],
              "Retained generation count differs from delivered usage")
        records.append(record)
    return records


def singleton(runner, reference):
    before = runner.idle()
    payload = runner.chat(SINGLETON_PROMPT, 24, seed=417, temperature=0.0)
    semantic, stats = runner.completion("singleton_json", payload)
    validate_length(semantic, payload["max_tokens"])
    runner.idle()
    streamed = runner.complete_stream(
        "singleton_sse", runner.chat(SINGLETON_PROMPT, 24, stream=True, seed=417, temperature=0.0),
    )
    check(streamed["semantic"] == semantic, "Singleton JSON and SSE differ in full content/reasoning/usage/finish")
    if reference is not None:
        check(semantic == reference, "Enabling batching changed the exact singleton B=1 result")
    after = runner.idle()
    records = completed_records(runner, after, [stats, streamed["summary"]])
    expect_counters(before, after, admitted=2, completed=2)
    return {
        "semantic": semantic, "json": stats, "stream": streamed["summary"],
        "retained": records, "idle": after,
        "equivalence": "Full singleton content, reasoning, finish and usage; transport IDs are intentionally unique.",
    }


def concurrent_group(runner, width):
    before = runner.idle()
    barrier = threading.Barrier(width)
    peak_active = 0
    progress_samples = []

    def run_one(index):
        prompt, maximum, seed = ROWS[index]
        barrier.wait(timeout=10.0)
        result = runner.complete_stream(
            f"concurrent_{width}_{index}",
            runner.chat(prompt, maximum, stream=True, seed=seed, temperature=0.6),
        )
        return {"row": index, "seed": seed, "max_tokens": maximum, **result}

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=width)
    futures = []
    try:
        futures = [pool.submit(run_one, index) for index in range(width)]
        pending = set(futures)
        deadline = time.perf_counter() + runner.read_timeout + 5.0
        while pending:
            check(runner.process.process.poll() is None, "Server exited during concurrent streams")
            health = runner.health()
            peak_active = max(peak_active, health["active_requests"])
            progress_samples.append({
                "at_seconds": time.perf_counter() - runner.started,
                "active_requests": health["active_requests"],
                "unfinished_readers": sum(not future.done() for future in futures),
                "ipc": health["ipc"],
            })
            check(time.perf_counter() < deadline, "Concurrent HTTP group exceeded its deadline")
            done, pending = concurrent.futures.wait(
                pending, timeout=0.025, return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                future.result()  # Fail promptly; never retry a rejected or broken stream.
        rows = [future.result() for future in futures]
    except BaseException:
        runner.close_clients()
        # Closing the owned server also wakes a thread still waiting for HTTP
        # headers, before ThreadPoolExecutor waits for those threads to join.
        runner.process.stop()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    summaries = [row["summary"] for row in rows]
    check(peak_active >= width, f"Never observed {width} live admitted requests (peak={peak_active})")
    if width > 1:
        check(max(item["started_at"] for item in summaries) < min(item["done_at"] for item in summaries),
              "Concurrent SSE lifetimes did not overlap")
    after = runner.idle()
    records = completed_records(runner, after, summaries)
    expect_counters(before, after, admitted=width, completed=width)
    return {
        "width": width, "peak_active": peak_active, "requests": rows,
        "progress_samples": progress_samples, "retained": records, "idle": after,
        "comparison": "Independent protocol/usage/lifecycle checks; no cross-width scalar sample oracle.",
    }


def disconnect_and_reuse(runner):
    before = runner.idle()
    victim = peer = None
    try:
        victim = runner.open_stream(
            "disconnect_victim",
            runner.chat("Explain in detail at least twenty steps in planning a scientific experiment.",
                        128, stream=True, seed=1201),
        )
        peer = runner.open_stream(
            "disconnect_peer",
            runner.chat("Explain in detail at least ten practical benefits of clear written instructions.",
                        24, stream=True, seed=1301),
        )
        victim.first_content()
        during = runner.health()
        check(during["active_requests"] >= 2, "Victim and peer were not both active before disconnect")
        victim_id = victim.id
        closed_at = time.perf_counter()
        runner.close_stream(victim, "intentional_client_disconnect")
        observed = runner.wait_health(
            lambda item: item["counters"]["disconnect_cancellations"] >
            before["counters"]["disconnect_cancellations"],
            timeout=8.0, description="the victim's actual disconnect cancellation",
        )
        cancel_seconds = time.perf_counter() - closed_at
        peer_summary = peer.complete()
        peer_semantic = peer.semantic()
        check(peer_semantic["finish_reason"] == "length" and
              peer_summary["usage"]["completion_tokens"] == 24,
              "The peer did not deliver its complete 24-token SSE response")
        runner.close_stream(peer, "completed_after_peer_disconnect")
        after_disconnect = runner.idle()
        victim_records = [item for item in after_disconnect["recent_requests"] if item["id"] == victim_id]
        check(len(victim_records) == 1 and victim_records[0]["status"] == "disconnect",
              "The disconnected request has no unique disconnect outcome")
        peer_records = completed_records(runner, after_disconnect, [peer_summary])
        expect_counters(before, after_disconnect, admitted=2, completed=1, cancelled=1, disconnect=1)
        replacement = runner.complete_stream(
            "disconnect_slot_reuse",
            runner.chat("Describe several simple ways to keep a desk organized.", 16,
                        stream=True, seed=1409),
        )
        after_reuse = runner.idle()
        reused_records = completed_records(runner, after_reuse, [replacement["summary"]])
        expect_counters(after_disconnect, after_reuse, admitted=1, completed=1)
        return {
            "active_before_disconnect": during["active_requests"], "cancel_seconds": cancel_seconds,
            "cancellation_observed": observed, "victim": victim_records[0],
            "peer": {"summary": peer_summary, "semantic": peer_semantic, "retained": peer_records},
            "idle_after_disconnect": after_disconnect, "slot_reuse": replacement,
            "reused_retained": reused_records, "idle_after_reuse": after_reuse,
        }
    finally:
        for client in (victim, peer):
            if client is not None and not client.closed:
                runner.close_stream(client, "failed_case_cleanup")


def active_shutdown(runner):
    before = runner.idle()
    clients = []
    try:
        for index, prompt in enumerate((
            "Explain in detail at least twenty stages of building a house.",
            "Explain in detail at least twenty stages of organizing a large library.",
        )):
            clients.append(runner.open_stream(
                f"shutdown_{index}", runner.chat(prompt, 128, stream=True, seed=1601 + index),
            ))
        clients[0].first_content()
        during = runner.health()
        check(during["active_requests"] >= 1, "Graceful shutdown had no active request")
        cleanup = runner.process.stop()  # Keep client sockets open until server-owned shutdown finishes.
        report = runner.validate_cleanup(cleanup)
        ids = {client.id for client in clients}
        retained = [item for item in report["recent_requests"] if item["id"] in ids]
        for identity in ids:
            check(sum(item["id"] == identity for item in retained) == 1,
                  f"Shutdown lost or duplicated the outcome of request {identity}")
        check(any(item["status"] == "shutdown" for item in retained),
              "No active request retired due to server shutdown")
        check(all(item["status"] in ("shutdown", "completed") for item in retained),
              "Shutdown hid an unrelated request failure")
        cancelled = sum(item["status"] == "shutdown" for item in retained)
        expect_counters(before, report, admitted=len(clients),
                        completed=len(clients) - cancelled, cancelled=cancelled)
        runner.last_health = report
        return {"active_at_shutdown": during["active_requests"], "retained": retained, "cleanup": cleanup}
    finally:
        for client in clients:
            runner.close_stream(client, "server_shutdown_interrupts_stream")


def run_process(args, name, width, reference):
    directory = args.output / name
    directory.mkdir()
    settings = copy.copy(args)
    settings.decode_batch_size = width
    result = {
        "case": name, "max_decode_batch_size": width, "started_at": utc_now(),
        "passed": False, "tests": [], "scope": SCOPE,
    }
    write_json(directory / "result.json", result)
    process = runner = None
    semantic = None
    try:
        process = ServerProcess(settings, directory, args.model)
        runner = EvidenceRunner(settings, process, directory, result)
        result["initial_health"] = runner.wait_ready()
        runner.case("singleton_json_sse_b1_equivalence", lambda: singleton(runner, reference))
        semantic = result["tests"][-1]["details"]["semantic"]
        if width > 1:
            for count in (1, 2, 3, 4):
                runner.case(f"concurrent_{count}_independent_sse", lambda count=count: concurrent_group(runner, count))
            runner.case("disconnect_peer_and_slot_reuse", lambda: disconnect_and_reuse(runner))
            runner.case("graceful_shutdown_with_active_requests", lambda: active_shutdown(runner))
        result["passed"] = True
    except BaseException as error:
        result.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    finally:
        if runner is not None:
            runner.close_clients()
            result["health_latency_samples"] = runner.health_samples
            result["last_health"] = runner.last_health
        if process is not None:
            try:
                result["cleanup"] = process.stop()
                check(runner is not None, "Runner initialization failed")
                runner.validate_cleanup(result["cleanup"])
            except BaseException as error:
                result.update(passed=False, cleanup_error=f"{type(error).__name__}: {error}")
        result["finished_at"] = utc_now()
        write_json(directory / "result.json", result)
    return result, semantic


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backend", choices=tuple(MODEL_IDS), required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory for this run; never overwritten")
    args = parser.parse_args()
    for field in ("server", "worker", "model", "output"):
        setattr(args, field, getattr(args, field).resolve())
    if not args.server.is_file() or not args.worker.is_file() or not args.model.is_dir():
        parser.error("--server/--worker must be files and --model must be an existing directory")
    if args.worker.name.lower() in ("ailafakeinferenceworker.exe", "ailafakeinferenceworker"):
        parser.error("This fixture requires the native GPU Worker, not the CPU fake Worker")
    args.mode = "gpu"
    args.model_id = MODEL_IDS[args.backend]
    args.context = 512
    args.startup_timeout, args.shutdown_timeout = 300.0, 45.0
    args.max_device_bytes = MAX_DEVICE_BYTES
    args.max_pinned_host_bytes = MAX_PINNED_HOST_BYTES
    args.max_session_host_bytes = MAX_SESSION_HOST_BYTES
    return args


def main():
    args = parse_arguments()
    try:
        args.output.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        print(f"Cannot create an independent output directory: {error}", flush=True)
        return 1
    manifest = {
        "started_at": utc_now(), "passed": False, "scope": SCOPE, "cases": [],
        "backend": args.backend, "model": str(args.model), "context": args.context,
        "limits": {
            "max_device_bytes": MAX_DEVICE_BYTES,
            "max_pinned_host_bytes": MAX_PINNED_HOST_BYTES,
            "max_session_host_bytes": MAX_SESSION_HOST_BYTES,
        },
        "artifacts": [
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in (Path(__file__).resolve(),)
        ],
        "inference_retries": 0,
    }
    write_json(args.output / "result.json", manifest)
    reference = None
    try:
        for name, width in (("scalar_reference", 1), ("batch_four_functional", 4)):
            result, semantic = run_process(args, name, width, reference)
            manifest["cases"].append(result)
            write_json(args.output / "result.json", manifest)
            print(json.dumps({"case": name, "passed": result["passed"], "error": result.get("error")}), flush=True)
            if not result["passed"]:
                break
            if width == 1:
                reference = semantic
        manifest["passed"] = len(manifest["cases"]) == 2 and all(item["passed"] for item in manifest["cases"])
    except BaseException as error:
        manifest.update(passed=False, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    finally:
        manifest["finished_at"] = utc_now()
        write_json(args.output / "result.json", manifest)
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
