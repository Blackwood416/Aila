"""CPU HTTP checks for decode batching configuration and serial fallback.

The CPU fake Worker does not advertise a native batch executor. These checks
cover the opt-in wire path, unchanged independent-request semantics, health and
argument rejection. They make no GPU correctness or throughput claim.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import traceback

from socket_tests import Runner, ServerProcess, check, utc_now, write_fake_tokenizer, write_json


def run_case(args, name, size, reference):
    directory = args.output / name
    directory.mkdir()
    model = directory / "fake-pipe-model"
    write_fake_tokenizer(model, args.backend)
    settings = copy.copy(args)
    settings.decode_batch_size = size
    record = {"case": name, "max_decode_batch_size": size or 1, "passed": False,
              "tests": [], "started_at": utc_now()}
    process = runner = None
    semantic = None
    try:
        process = ServerProcess(settings, directory, model)
        runner = Runner(settings, process, directory, record)
        health = runner.wait_ready()
        check(health.get("max_decode_batch_size") == (size or 1), "Applied decode batch bound was not reported")
        payload = runner.chat("hello", 8)
        semantic, stats = runner.completion("batching_reference", payload)
        if reference is not None:
            check(semantic == reference, "Batching configuration changed the independent default reference")
        runner.idle()
        runner.case("independent_requests_1_2_4", runner.concurrency)
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
                check(report and report.get("max_decode_batch_size") == (size or 1),
                      "Shutdown report lost its decode batch bound")
            except BaseException as error:
                record.update(passed=False, cleanup_error=str(error))
        record["finished_at"] = utc_now()
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
    cases = (("default", None), ("explicit_one", 1), ("batch_two", 2), ("batch_three", 3), ("batch_four", 4))
    for name, size in cases:
        record, semantic = run_case(args, name, size, reference)
        records.append(record)
        print(json.dumps({"case": name, "passed": record["passed"], "error": record.get("error")}), flush=True)
        if not record["passed"]:
            break
        if name == "default":
            reference = semantic
    invalid = []
    invalid_cases = (("0", "4"), ("5", "4"), ("-1", "4"), ("1.5", "4"),
                     ("true", "4"), ("4294967296", "4"), ("2", "1"))
    if len(records) == len(cases) and all(item["passed"] for item in records):
        for value, sessions in invalid_cases:
            # A missing tokenizer is a second barrier against Worker launch if
            # the argument check regresses. Its unrelated diagnostic must fail.
            result = subprocess.run(
                [str(args.server), "--worker", str(args.worker), "--model",
                 str(args.output / "missing-tokenizer"), "--sessions", sessions, "--decode-batch-size", value],
                capture_output=True, timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            passed = result.returncode == 1 and b"--decode-batch-size" in result.stderr
            invalid.append({"value": value, "sessions": sessions, "passed": passed,
                            "stderr": result.stderr.decode("utf-8", errors="replace")})
    passed = (len(records) == len(cases) and all(item["passed"] for item in records)
              and len(invalid) == len(invalid_cases) and all(item["passed"] for item in invalid))
    write_json(args.output / "result.json", {
        "passed": passed, "cases": records, "invalid_options": invalid,
        "scope": "CPU configure/capability routing and unsupported-backend serial fallback; no GPU or performance claim.",
    })
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
