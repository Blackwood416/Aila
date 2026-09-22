"""CPU-only HTTP acceptance for optional runtime memory observation."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import time
import traceback

from socket_tests import Runner, ServerProcess, check, write_fake_tokenizer, write_json, utc_now


CASES = {
    "disabled": ("fake-runtime-memory-model", 0),
    "uninitialized": ("fake-pipe-model", 100),
    "ready_busy": ("fake-runtime-memory-model", 100),
    "query_error": ("fake-runtime-memory-error-model", 100),
    "query_timeout": ("fake-runtime-memory-delay-model", 100),
}


def run_case(args, name, reference):
    fixture, interval = CASES[name]
    directory = args.output / name
    directory.mkdir()
    model = directory / fixture
    write_fake_tokenizer(model, args.backend)
    settings = copy.copy(args)
    settings.memory_sample_ms = interval
    record = {"case": name, "passed": False, "tests": [], "started_at": utc_now()}
    process = runner = None
    semantic = None
    observations = []
    try:
        process = ServerProcess(settings, directory, model)
        runner = Runner(settings, process, directory, record)
        runner.wait_ready()
        if interval:
            target = "uninitialized" if name == "uninitialized" else "ready" if name == "ready_busy" else "error"
            deadline = time.monotonic() + 5
            while True:
                health = runner.health()
                sample = health["runtime_memory"]
                observations.append(sample)
                check(sample["epoch"] == health["ipc"]["epoch"], "Memory sample lost its Worker identity")
                if sample["state"] in ("busy", "uninitialized", "pending", "error"):
                    check(sample["memory"] is None, "Unavailable state published stale or zero memory")
                if sample["state"] == target and (name != "ready_busy" or sample["samples"] >= 3):
                    break
                check(time.monotonic() < deadline, "Expected sampler state did not arrive")
                time.sleep(.02)
            check(sample["age_ms"] >= 0, "Memory age is missing")
            if name == "ready_busy":
                check(sample["busy_samples"] >= 1 and sample["failures"] == 0, "Busy sampling was lost")
                check(sample["memory"]["device_bytes"] == 1024 and sample["memory"]["pinned_host_bytes"] == 16,
                      "Factory sentinels did not reach HTTP")
                check(sample["memory"]["device_limit_bytes"] == sample["memory"]["pinned_host_limit_bytes"] == 0,
                      "HTTP observation differs from applied limits")
            if target == "error":
                check(sample["sampling_stopped"] and sample["failures"] == 1, "Failed sampling did not stop")
                check(sample["error"] == ("timeout" if name == "query_timeout" else "query_failed"), "Wrong query failure")
                time.sleep(.35)
                later = runner.health()["runtime_memory"]
                check(later["samples"] == sample["samples"] and later["age_ms"] > sample["age_ms"],
                      "Failed sampler retried or refreshed stale data")
            check(max(row["seconds"] for row in runner.health_samples) < .75,
                  "HTTP health waited for the memory RPC")
        else:
            check("runtime_memory" not in runner.health(), "Disabled observation changed the old report")
        payload = runner.chat("hello", 8)
        semantic, stats = runner.completion("after_observation", payload)
        if reference is not None:
            check(semantic == reference, "Memory observation changed the independent disabled reference")
        client = runner.stream("after_observation_sse", {**payload, "stream": True, "stream_options": {"include_usage": True}})
        try:
            client.complete()
            check(client.semantic() == semantic, "Memory observation changed SSE semantics")
        finally:
            client.close()
        runner.idle()
        record.update(passed=True, completion=stats, reference_semantic=semantic)
    except BaseException as error:
        record.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    finally:
        if runner:
            runner.close_clients()
            record["health_latency_samples"] = runner.health_samples
        if process:
            try:
                cleanup = process.stop()
                record["cleanup"] = cleanup
                report = runner.validate_cleanup(cleanup) if runner else None
                if interval:
                    check(report["runtime_memory"]["sampling_stopped"], "Memory thread survived shutdown")
            except BaseException as error:
                record.update(passed=False, cleanup_error=str(error))
        record.update(finished_at=utc_now(), memory_observations=observations)
        write_json(directory / "result.json", record)
    return record, semantic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.server, args.worker, args.output = args.server.resolve(), args.worker.resolve(), args.output.resolve()
    check(args.worker.name.lower() == "ailafakeinferenceworker.exe", "This fixture must never start a GPU Worker")
    args.mode, args.backend, args.model_id, args.model = "fake", "qwen3_nf4", "qwen3", None
    args.startup_timeout, args.shutdown_timeout, args.context = 20, 10, 512
    args.output.mkdir(parents=True, exist_ok=False)
    records, reference = [], None
    for name in CASES:
        record, semantic = run_case(args, name, reference)
        records.append(record)
        print(json.dumps({"case": name, "passed": record["passed"], "error": record.get("error")}))
        if not record["passed"]:
            break
        if name == "disabled": reference = semantic
    invalid = []
    if len(records) == len(CASES) and all(item["passed"] for item in records):
        for value in ("1", "99", "60001", "-1", "1.5"):
            # The missing tokenizer prevents any Worker launch even if this
            # validation regresses; the specific diagnostic is mandatory.
            result = subprocess.run([str(args.server), "--worker", str(args.worker), "--model",
                str(args.output / "missing-tokenizer"), "--memory-sample-ms", value], capture_output=True, timeout=5)
            passed = result.returncode == 1 and b"--memory-sample-ms" in result.stderr
            invalid.append({"value": value, "passed": passed, "stderr": result.stderr.decode("utf-8", errors="replace")})
    passed = len(records) == len(CASES) and all(item["passed"] for item in records) and len(invalid) == 5 and all(item["passed"] for item in invalid)
    write_json(args.output / "result.json", {"passed": passed, "cases": records, "invalid_options": invalid,
        "scope": "CPU fake HTTP, state/counter routing, failure latch and health liveness; no GPU or performance claim."})
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
