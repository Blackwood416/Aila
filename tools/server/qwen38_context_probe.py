"""Small real-HTTP long-context/continuation probe; owns and shuts down its server."""
import argparse
import hashlib
import json
import os
import sys
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/server"))
import socket_tests as st


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--prompt-tokens", type=int, nargs="*", default=[6000, 14000],
                        help="context targets; pass without values for cancellation-only validation")
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--generate", type=int, default=64)
    parser.add_argument("--reuse", type=int, choices=[0, 1], default=1)
    parser.add_argument("--pi-smoke", action="store_true")
    parser.add_argument("--pi-context-file", type=Path)
    parser.add_argument("--pi-only", action="store_true")
    parser.add_argument("--cancel-recover", action="store_true", help="cancel one stream, then verify a fresh single-stream request")
    parser.add_argument("--cancel-prefill", action="store_true", help="after timing, cancel a long prefill before output and verify recovery")
    parser.add_argument("--fixture", type=Path, help="frozen natural inputs and fixed PI workspace")
    parser.add_argument("--production-timing", action="store_true", help="clear diagnostics before starting Worker")
    cli = parser.parse_args()
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(cli.model / "tokenizer.json"))
    fixture = None
    if cli.fixture:
        from freeze_qwen38_agent_inputs import load_fixture
        cli.fixture = cli.fixture.resolve()
        fixture = load_fixture(cli.fixture, tokenizer)
    cleared = []
    if cli.production_timing:
        for key in list(os.environ):
            # HOTSET_PROFILE is a placement input, not diagnostic profiling.
            if key.startswith(("AILA_DEBUG_", "AILA_PROFILE_")) or (key.startswith("AILA_Q4E_") and
                    any(part in key for part in ("DIAGNOSTIC", "TRACE", "WORK_SUMMARY", "PROFILE_COMPONENTS", "HISTOGRAM"))) or key in (
                    "AILA_SERVER_WORKER_LOG", "DNNL_VERBOSE", "ONEDNN_VERBOSE", "SYCL_PI_TRACE", "UR_LOG_LEVEL"):
                cleared.append(key)
                os.environ.pop(key)
    args = SimpleNamespace(server=ROOT / "build-refactor-server/AilaServer.exe",
        worker=ROOT / "build-refactor/AilaWorker.exe", backend="qwen4exp_nf4", model_id="qwen4exp",
        model=cli.model, mode="gpu", context=cli.context, sessions=1,
        max_body_bytes=4 * 1024 * 1024, request_timeout_ms=600000,
        prefill_chunk_size=cli.chunk, reuse_sessions=cli.reuse,
        startup_timeout=300, shutdown_timeout=45)
    out = cli.output.resolve()
    startup_started = time.perf_counter()
    process = st.ServerProcess(args, out, args.model)
    result = {"passed": False, "context": cli.context, "chunk": cli.chunk, "reuse": cli.reuse, "requests": []}
    result.update(fixture=fixture, production_timing=cli.production_timing, cleared_diagnostics=cleared,
                  worker_environment={k:v for k,v in os.environ.items() if k.startswith(("AILA_", "SYCL_", "UR_L0_"))})
    runner = st.Runner(args, process, out, result)

    def request(label, messages, maximum):
        payload = runner.chat("", maximum, stream=True)
        payload["messages"] = messages
        st.write_json(out / f"{label}-request.json", payload)
        client = runner.stream(label, payload)
        try:
            stats = client.complete()
            runner.remember_id(stats["id"])
            semantic = client.semantic()
        finally:
            client.close()
        runner.idle()
        record = runner.wait_record(stats["id"])
        first, duration = stats["first_content_seconds"], stats["duration_seconds"]
        rate = (stats["usage"]["completion_tokens"] - 1) / (duration - first) if first and duration > first else None
        item = {"label": label, "stream": stats, "semantic": semantic,
                "cached_prompt_tokens": record.get("cached_prompt_tokens", 0), "after_first_content_tokens_s": rate,
                "messages_sha256": hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()}
        result["requests"].append(item)
        st.write_json(out / "result.json", result)
        print(json.dumps({"label": label, "prompt_tokens": stats["usage"]["prompt_tokens"],
            "first_content_seconds": first, "tokens_s": rate, "cached_prompt_tokens": item["cached_prompt_tokens"]}), flush=True)
        return semantic

    try:
        runner.wait_ready()
        result["worker_startup_seconds"] = time.perf_counter() - startup_started
        result["health_at_load"] = runner.health()
        if cli.pi_smoke or cli.pi_context_file or cli.pi_only:
            command = [sys.executable, str(ROOT / "tools/server/pi_agent_smoke.py"),
                "--endpoint", f"http://127.0.0.1:{process.port}/v1", "--model-id", args.model_id,
                "--context", str(cli.context), "--max-tokens", "256", "--timeout", "600",
                "--tmp-root", str(out / "pi"), "--keep"]
            if cli.pi_context_file:
                command += ["--context-file", str(cli.pi_context_file.resolve())]
            if fixture:
                command += ["--work-dir", str(cli.fixture / "pi-work"), "--marker-token", fixture["marker"]]
            pi = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=630)
            (out / "pi-stdout.log").write_text(pi.stdout, encoding="utf-8")
            (out / "pi-stderr.log").write_text(pi.stderr, encoding="utf-8")
            result["pi"] = {"exit_code": pi.returncode, "command": command}
            summaries = list((out / "pi").glob("aila-pi-smoke-*/result.json"))
            st.check(len(summaries) <= 1, "Output directory contains multiple PI runs; choose a fresh directory")
            if summaries:
                result["pi"]["summary"] = json.loads(summaries[0].read_text(encoding="utf-8"))
                result["pi"]["phase"] = "first_request_after_worker_ready_no_warmup"
                result["startup_plus_pi_seconds"] = result["worker_startup_seconds"] + result["pi"]["summary"]["seconds"]
            st.write_json(out / "result.json", result)
            st.check(pi.returncode == 0, "PI agent smoke failed; see pi-stderr.log")
            runner.idle()
            print("PI agent tool round trip passed", flush=True)
        if cli.cancel_recover:
            before = runner.health()["counters"]["disconnect_cancellations"]
            victim = runner.stream("cancel-victim", runner.chat(
                "Explain the steps of implementing a compiler in detail.", 128, stream=True))
            try:
                victim.first_content()
                victim_id = victim.id
            finally:
                victim.close()
            cancelled = runner.wait_health(
                lambda health: health["counters"]["disconnect_cancellations"] > before,
                timeout=30, description="single-stream disconnect cancellation")
            runner.idle(timeout=30)
            record = runner.wait_record(victim_id)
            st.check(record["status"] == "disconnect", "Stream was not cancelled by disconnect")
            answer = request("after-cancel", [{"role": "user", "content": "Reply with exactly READY."}], 16)
            st.check("READY" in answer["content"], "Fresh request did not recover after cancellation")
            result["cancel_recover"] = {"victim": record, "passed": True,
                "disconnect_cancellations": cancelled["counters"]["disconnect_cancellations"]}
        if cli.pi_only:
            result["health_final"] = runner.health()
            result["passed"] = True
            return
        if not fixture:
            request("warmup", [{"role": "user", "content": "Write a short sentence about programming."}], 16)
        for target in cli.prompt_tokens:
            # Fixed synthetic repository notes keep input reproducible. Actual
            # rendered token counts are reported by the Server, never assumed.
            if fixture:
                entry = next(item for item in fixture["inputs"] if item["target"] == target)
                content = (cli.fixture / entry["file"]).read_text(encoding="utf-8")
            else:
                lines = "".join(f"module_{i:05d}: reads a file, validates input, returns a result; errors are logged.\n" for i in range(target // 10 + 100))
                ids = tokenizer.encode(lines).ids[:target - 100]
                content = tokenizer.decode(ids) + "\nSummarize these repository notes in a detailed paragraph."
            messages = [{"role": "user", "content": content}]
            answer = request(f"context-{target}", messages, cli.generate)
            messages += [{"role": "assistant", "content": answer["content"]},
                         {"role": "user", "content": "Continue the summary with error handling and validation details."}]
            request(f"context-{target}-continue", messages, cli.generate)
        if cli.cancel_prefill:
            st.check(fixture is not None, "Prefill cancellation requires a frozen fixture")
            entry = next(item for item in fixture["inputs"] if item["target"] == 14000)
            payload = runner.chat("", 32, stream=True)
            payload["messages"] = [
                {"role": "system", "content": "Summarize the repository carefully. This is the fixed prefill cancellation case."},
                {"role": "user", "content": (cli.fixture / entry["file"]).read_text(encoding="utf-8")}
            ]
            st.write_json(out / "cancel-prefill-request.json", payload)
            victim = runner.stream("cancel-prefill", payload)
            try:
                runner.wait_health(lambda health: health["active_requests"] == 1)
                # The request is admitted; allow the first chunk to begin, then
                # disconnect before any generated content is consumed.
                time.sleep(0.5)
                victim_id = victim.header_id
            finally:
                cancelled_at = time.perf_counter()
                victim.close()
            runner.idle(timeout=90)
            record = runner.wait_record(victim_id)
            result["prefill_cancel_recover"] = {"passed": False, "victim": record,
                "drain_seconds": time.perf_counter() - cancelled_at}
            # A disconnect before token delivery has no final Worker usage
            # snapshot, so generated_tokens is legitimately null. The retained
            # stream peak confirms that no content was queued before closing.
            st.check(record["status"] == "disconnect" and record["generated_tokens"] in (None, 0)
                     and record["peak_pending_bytes"] == 0,
                     "Expected cancellation during prefill before output")
            answer = request("after-prefill-cancel", [{"role": "user", "content": "Reply with exactly READY."}], 16)
            st.check("READY" in answer["content"], "Fresh request failed after prefill cancellation")
            result["prefill_cancel_recover"]["passed"] = True
        result["health_final"] = runner.health()
        result["passed"] = True
    finally:
        runner.close_clients()
        result["cleanup"] = process.stop()
        st.write_json(out / "result.json", result)
        runner.validate_cleanup(result["cleanup"])


if __name__ == "__main__":
    main()
