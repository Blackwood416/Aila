#!/usr/bin/env python3
"""Isolated PI coding-agent smoke test for the Aila OpenAI-compatible server.

This helper drives the *real* installed PI coding agent CLI against a local
OpenAI-compatible endpoint.  It never reads or writes the user's global PI
configuration and never reads ``auth.json`` or provider credentials; it points
PI at a dummy key and its own agent directory.  It calls no hosted provider,
though PI still speaks ordinary HTTP to the local endpoint.  ``PI_OFFLINE=1``
only suppresses Pi's catalog and version-check requests.  Every file it needs
lives in a fresh directory under the system temp folder, or in ``--tmp-root``.

Two modes:

``--stub``
    Serve a canned OpenAI ``chat/completions`` SSE reply from an in-process
    HTTP server on ``127.0.0.1``.  No GPU and no AilaServer are required.  This
    proves the PI wiring (isolated config, model discovery, request shape) and,
    with ``--capture``, records the exact request body and headers PI sends so
    a server-side field gap can be confirmed instead of guessed.

default (no ``--stub``)
    Point PI at a running ``AilaServer`` (``--endpoint``) and require the agent
    to answer with the contents of a marker file using only the ``read`` tool.

The agent is given only the ``read`` tool and an isolated working directory.
This is not a filesystem sandbox. The saved JSON events verify that the single
tool call read the marker and that the final assistant answer contains it.

Exit codes: 0 success, 1 smoke assertion failed, 2 setup/environment error.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import string
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MARKER_FILE = "smoke-marker.txt"
PROVIDER = "aila"
DEFAULT_MODEL_ID = "qwen4exp"

# Environment variables that carry provider credentials.  They are stripped
# from the child environment so the smoke cannot accidentally authenticate
# against a real provider if the local endpoint is misconfigured.
CREDENTIAL_ENV_PREFIXES = (
    "PI_",
    "OPENAI_",
    "ANTHROPIC_",
    "OPENROUTER_",
    "GOOGLE_",
    "GEMINI_",
    "MISTRAL_",
    "GROQ_",
    "XAI_",
    "DEEPSEEK_",
    "AWS_",
    "AZURE_",
)


class SetupError(RuntimeError):
    """Raised for a missing launcher, port conflict, or other setup problem."""


def token(length: int = 12) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(random.choice(alphabet) for _ in range(length))


def child_environment(agent_dir: Path, session_dir: Path) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith(CREDENTIAL_ENV_PREFIXES)
    }
    env.update(
        {
            "PI_CODING_AGENT_DIR": str(agent_dir),
            "PI_CODING_AGENT_SESSION_DIR": str(session_dir),
            "PI_OFFLINE": "1",
            "PI_SKIP_VERSION_CHECK": "1",
            "PI_TELEMETRY": "0",
        }
    )
    return env


def resolve_pi_launcher(explicit: str | None) -> list[str]:
    """Return the argv prefix that runs the real PI CLI.

    ``explicit`` may be a bare launcher (``pi``) or a ``cli.js`` bundle path.
    Otherwise the globally installed npm package is located through
    ``npm root -g``.  A ``.js`` target is always executed with ``node``.
    """

    def run_cli(target: Path) -> list[str]:
        node = shutil.which("node")
        if node is None:
            raise SetupError("node is not on PATH; install Node.js >= 22.19 or pass --pi")
        return [node, str(target)]

    if explicit:
        candidate = Path(explicit)
        if candidate.suffix == ".js" or (candidate.is_file() and candidate.suffix != ".ps1" and candidate.suffix != ".cmd"):
            if not candidate.is_file():
                raise SetupError(f"--pi target does not exist: {candidate}")
            return run_cli(candidate)
        resolved = shutil.which(explicit)
        if resolved is None and not candidate.is_file():
            raise SetupError(f"--pi launcher not found: {explicit}")
        launcher = Path(resolved) if resolved else candidate
        if launcher.suffix == ".js":
            return run_cli(launcher)
        return [str(launcher)]

    npm = shutil.which("npm") or shutil.which("npm.cmd")
    roots: list[Path] = []
    if npm is not None:
        try:
            output = subprocess.run(
                [npm, "root", "-g"],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            if output.returncode == 0:
                roots.append(Path(output.stdout.strip()))
        except (OSError, subprocess.SubprocessError):
            pass
    roots.append(Path(os.environ.get("APPDATA", "")) / "npm" / "node_modules")

    for root in roots:
        bundle = root / "@earendil-works" / "pi-coding-agent" / "dist" / "bundle" / "cli.js"
        if bundle.is_file():
            return run_cli(bundle)
    raise SetupError(
        "could not locate the PI coding agent bundle; install it with "
        "`npm i -g @earendil-works/pi-coding-agent` or pass --pi <path-to-cli.js>"
    )


def write_isolated_config(agent_dir: Path, model_id: str, endpoint: str,
                          context_window: int, max_tokens: int) -> None:
    agent_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "defaultProvider": PROVIDER,
        "defaultModel": model_id,
        "defaultTools": ["read"],
        "compaction": {"reserveTokens": 4096, "keepRecentTokens": 8192},
    }
    models = {
        "providers": {
            PROVIDER: {
                "baseUrl": endpoint.rstrip("/"),
                "api": "openai-completions",
                # A dummy key makes the local model selectable without any
                # credential lookup; the local server does not authenticate.
                "apiKey": "aila-smoke-local",
                "models": [
                    {
                        "id": model_id,
                        "name": f"Aila {model_id}",
                        "input": ["text"],
                        "output": ["text"],
                        # Advertise only what the running server was configured
                        # for, so Pi does not plan a prompt past its capacity.
                        "contextWindow": context_window,
                        "maxTokens": max_tokens,
                        "reasoning": True,
                        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                    }
                ],
            }
        }
    }
    (agent_dir / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    (agent_dir / "models.json").write_text(json.dumps(models, indent=2), encoding="utf-8")


class RequestRecorder:
    """Records the first request a stub endpoint receives."""

    def __init__(self) -> None:
        self.record: dict | None = None
        self.event = threading.Event()

    def capture(self, path: str, headers: dict[str, str], body: str) -> None:
        if self.record is not None:
            return
        try:
            parsed: object = json.loads(body)
        except ValueError:
            parsed = body
        self.record = {"path": path, "headers": headers, "body": parsed}
        self.event.set()


def make_stub_handler(recorder: RequestRecorder, model_id: str):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # silence the default access log
            pass

        def _wire(self, payload: dict, status: int = 200) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # model discovery is not required, but stay friendly
            self._wire({"object": "list", "data": []})

        def do_POST(self):
            length = int(self.headers.get("content-length", 0))
            body = self.rfile.read(length).decode("utf-8", "replace")
            recorder.capture(self.path, {k: v for k, v in self.headers.items()}, body)
            envelope = {
                "id": "chatcmpl-smoke",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_id,
            }
            frames = [
                dict(envelope, choices=[{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]),
                dict(envelope, choices=[{"index": 0, "delta": {"content": "stub-ok"}, "finish_reason": None}]),
                dict(envelope, choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}],
                     usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}),
            ]
            payload = "".join("data: " + json.dumps(frame) + "\n\n" for frame in frames) + "data: [DONE]\n\n"
            data = payload.encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("cache-control", "no-cache")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def start_stub(recorder: RequestRecorder, model_id: str, port: int) -> ThreadingHTTPServer:
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), make_stub_handler(recorder, model_id))
    except OSError as error:
        raise SetupError(f"could not bind the stub endpoint on 127.0.0.1:{port}: {error}") from error
    thread = threading.Thread(target=server.serve_forever, name="pi-smoke-stub", daemon=True)
    thread.start()
    return server


def build_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080/v1",
                        help="OpenAI-compatible base URL (default: %(default)s)")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="served model id (default: %(default)s)")
    parser.add_argument("--context", type=int, default=32768,
                        help="advertised context window, must match the server (default: %(default)s)")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="advertised max output tokens, must match the server (default: %(default)s)")
    parser.add_argument("--context-file", type=Path, help="attach background text to the real PI prompt")
    parser.add_argument("--stub", action="store_true",
                        help="serve a canned SSE reply in-process instead of contacting --endpoint")
    parser.add_argument("--stub-port", type=int, default=8137, help="stub endpoint port (default: %(default)s)")
    parser.add_argument("--capture", metavar="PATH",
                        help="write the captured request (headers + body) as JSON; requires --stub")
    parser.add_argument("--print-request", action="store_true", help="also print the captured request to stdout")
    parser.add_argument("--pi", help="PI launcher or cli.js path (default: globally installed package)")
    parser.add_argument("--tmp-root", help="parent directory for the isolated workspace (default: system temp)")
    parser.add_argument("--timeout", type=float, default=180.0, help="wall-clock limit in seconds (default: %(default)s)")
    parser.add_argument("--keep", action="store_true", help="keep the isolated workspace after the run")
    parser.add_argument("--marker-token", help="marker contents (default: a random token)")
    parser.add_argument("--work-dir", type=Path,
                        help="fixed external workspace with a pre-existing marker; never modified or cleaned up")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = build_arguments(argv)
    if args.capture and not args.stub:
        print("error: --capture requires --stub", file=sys.stderr)
        return 2

    mark = args.marker_token or token()
    root = Path(args.tmp_root).resolve() if args.tmp_root else Path(tempfile.mkdtemp(prefix="aila-pi-smoke-"))
    if args.tmp_root:
        root = root / f"aila-pi-smoke-{token(6).lower()}"
    agent_dir = root / "agent"
    session_dir = root / "sessions"
    work_dir = args.work_dir.resolve(strict=True) if args.work_dir else root / "work"
    recorder = RequestRecorder()
    stub: ThreadingHTTPServer | None = None

    try:
        launcher = resolve_pi_launcher(args.pi)
    except SetupError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    try:
        endpoint = args.endpoint
        if args.stub:
            stub = start_stub(recorder, args.model_id, args.stub_port)
            endpoint = f"http://127.0.0.1:{args.stub_port}/v1"
        work_dir.mkdir(parents=True, exist_ok=True)
        session_dir.mkdir(parents=True, exist_ok=True)
        write_isolated_config(agent_dir, args.model_id, endpoint, args.context, args.max_tokens)
        marker = work_dir / MARKER_FILE
        if args.work_dir:
            if marker.read_text(encoding="utf-8") != mark + "\n":
                raise SetupError("fixed workspace marker must match --marker-token")
        else:
            marker.write_text(mark + "\n", encoding="utf-8")

        prompt = (
            f"Read the file {MARKER_FILE} in the current directory with the read tool once, "
            "then reply with only its exact contents and no other text."
        )
        if args.context_file:
            prompt = ("The attached background text is already included in this message; "
                      "it requires no file reads. Your only task and only tool call is the following: " + prompt)
        # Isolate hard: no discovered/configured extensions, skills, prompt
        # templates or AGENTS.md; pin the exact provider and turn thinking off;
        # expose only the read tool.
        command = launcher + [
            "--mode", "json", "--no-session", "--no-extensions", "--no-skills",
            "--no-prompt-templates", "--no-context-files",
            "--provider", PROVIDER, "--model", args.model_id,
            "--thinking", "off", "--tools", "read",
        ]
        if args.context_file:
            command.append("@" + str(args.context_file.resolve(strict=True)))
        command.append(prompt)
        print("pi-smoke: " + " ".join(command), file=sys.stderr)
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=str(work_dir),
            env=child_environment(agent_dir, session_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=args.timeout,
            check=False,
        )
        stdout, stderr, code = completed.stdout, completed.stderr, completed.returncode
        elapsed = time.perf_counter() - started
        (root / "events.jsonl").write_text(stdout, encoding="utf-8")
        (root / "stderr.log").write_text(stderr, encoding="utf-8")
    except subprocess.TimeoutExpired:
        print(f"FAIL: PI did not finish within {args.timeout:.0f}s", file=sys.stderr)
        return 1
    except SetupError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    finally:
        if stub is not None:
            stub.shutdown()
            stub.server_close()

    if args.capture:
        capture_path = Path(args.capture)
        capture_path.parent.mkdir(parents=True, exist_ok=True)
        capture_path.write_text(json.dumps(recorder.record, indent=2), encoding="utf-8")
        print(f"pi-smoke: captured request -> {capture_path}", file=sys.stderr)
    if args.print_request and recorder.record is not None:
        print(json.dumps(recorder.record, indent=2))

    if code != 0:
        print(f"FAIL: PI exited with {code}", file=sys.stderr)
        print(stdout, file=sys.stderr)
        print(stderr, file=sys.stderr)
        return 1
    if args.stub:
        print("OK: PI completed a stub-endpoint turn")
        return 0
    events = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    calls = [e for e in events if e.get("type") == "tool_execution_start"]
    ends = [e for e in events if e.get("type") == "tool_execution_end"]
    messages = [e["message"] for e in events if e.get("type") == "message_end"
                and e.get("message", {}).get("role") == "assistant"]
    answer = "".join(block.get("text", "") for block in messages[-1].get("content", [])
                     if block.get("type") == "text") if messages else ""
    passed = (len(calls) == 1 and calls[0].get("toolName") == "read"
              and (work_dir / calls[0].get("args", {}).get("path", "")).resolve() == marker.resolve()
              and len(ends) == 1 and not ends[0].get("isError", True)
              and ends[0].get("toolCallId") == calls[0].get("toolCallId")
              and mark in answer)
    summary = {"passed": passed, "seconds": elapsed, "calls": calls, "answer": answer,
               "work_dir": str(work_dir), "marker": mark, "prompt": prompt,
               "usage": [message.get("usage") for message in messages]}
    (root / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"pi-smoke: artifacts -> {root}", file=sys.stderr)
    if not passed:
        print("FAIL: expected exactly one successful marker read and the correct final answer", file=sys.stderr)
        return 1
    if not args.keep:
        expected_parent = Path(args.tmp_root).resolve() if args.tmp_root else Path(tempfile.gettempdir()).resolve()
        if root.parent != expected_parent or not root.name.startswith("aila-pi-smoke-"):
            raise SetupError("refusing cleanup outside the owned temporary workspace")
        shutil.rmtree(root)
    print(f"OK: PI read {MARKER_FILE} and reported the marker via {endpoint}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
