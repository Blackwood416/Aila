"""Real HTTP/1.1 socket acceptance for AilaServer; Python standard library only.

This runner owns each Popen handle and a unique shutdown file. It first requests
server shutdown, waits a bounded interval, and only then kills that exact child.
Fake mode writes a byte-BPE tokenizer under the output directory. It exercises
network backpressure with roughly 4 KiB per generated token; GPU mode uses the
provided checkpoint and does not claim to exercise that artificial byte limit.
Both modes accept --backend qwen3_nf4|qwen35_nf4|qwen35_bf16. A --case selection keeps every
assertion of the selected cases and the final resource/transport cleanup gate;
the report explicitly lists cases excluded from a partial run.
"""

from __future__ import annotations

import argparse
import codecs
import concurrent.futures
import datetime
import hashlib
import http.client
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

MODEL_IDS = {"qwen3_nf4": "qwen3", "qwen35_nf4": "qwen3.5", "qwen35_bf16": "qwen3.5"}
COMMON_CASES = (
    "health_models_parameters", "incremental_utf8_json_sse", "concurrent_1_2_4_fixed_seed",
    "multiturn_text_thinking_json_sse", "connection_close_expect_continue", "same_socket_reuse_pipeline",
    "thinking_budget_json_sse_and_concurrency",
    "disconnect_isolates_other_request", "graceful_shutdown_with_active_requests",
)
FAKE_CASES = (
    "slow_reader_resume_and_peer", "slow_reader_stall_cleanup", "admission_429_and_slot_reuse",
    "worker_fault_error_terminal",
)
OUTPUT_LIMIT = 1048576
STREAM_LIMIT = 65536
FRAME_LIMIT = 16384
SEND_BUFFER = 16384
SLOW_RECEIVE_BUFFER = 4096
LONG_FAKE_TOKENS = 160
STALL_SECONDS = 5.0
MAX_WIRE_BYTES = OUTPUT_LIMIT + 65536


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def integer(value, label, minimum=0):
    check(type(value) is int and value >= minimum, f"{label}: expected integer >= {minimum}, got {value!r}")
    return value


def strict_json(value):
    def no_constant(text):
        raise ValueError(f"Non-finite JSON number: {text}")

    def no_duplicates(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON member: {key}")
            result[key] = item
        return result

    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="strict")
    return json.loads(value, parse_constant=no_constant, object_pairs_hook=no_duplicates)


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def framework_errors(directory):
    # Trantor's LOG_SYSERR is rendered as FATAL on this pinned version even
    # when the process survives. A successful response cannot hide that error.
    pattern = re.compile(r"^\d{8} \d{2}:\d{2}:\d{2}\.\d+\s+\S+\s+\S+\s+(?:ERROR|FATAL)\b")
    errors = []
    for name in ("stdout.log", "stderr.log"):
        path = directory / name
        if not path.is_file():
            continue
        with path.open(encoding="utf-8", errors="replace") as log:
            for number, line in enumerate(log, 1):
                if pattern.match(line):
                    errors.append({"file": name, "line": number, "message": line.rstrip()[:2048]})
                    if len(errors) == 32:
                        return errors
    return errors


def fake_text(token_id):
    return ("é中🙂" * 450) + f":{token_id:03d};\n"


def write_fake_tokenizer(directory, backend, tools=False):
    directory.mkdir(parents=True, exist_ok=False)
    # Match Tokenizer::byte_to_unicode, including the nonprinting-byte range.
    byte_values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    codepoints = byte_values.copy()
    next_codepoint = 256
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            codepoints.append(next_codepoint)
            next_codepoint += 1
    byte_map = dict(zip(byte_values, map(chr, codepoints)))
    check(len(byte_map) == 256, "Incomplete fake byte-to-Unicode table")

    vocab = {
        "".join(byte_map[byte] for byte in fake_text(token_id).encode("utf-8")): token_id
        for token_id in range(1000)
    }
    for byte in range(256):
        token_id = 1000 + byte + (1 if byte >= 23 else 0)
        check(token_id != 1023, "Ordinary byte accidentally assigned the Worker crash token")
        vocab[byte_map[byte]] = token_id
    special = {
        "<|crash_worker|>": 1023,
        "<|im_start|>": 2000,
        "<|im_end|>": 2001,
        "<think>": 2002,
        "</think>": 2003,
        "<|endoftext|>": 2047,
    }
    if tools:
        special.update({f"<|tool_{name}|>": 2010 + index for index, name in enumerate(
            ("json", "xml", "parallel", "malformed", "plain", "long", "unknown", "unfinished"))})
        special.update({"<tool_call>": 2018, "</tool_call>": 2019})
        fragments = ('<tool_call>{"name":"weather","arguments":{"city":"', "北京🙂" * 100, '"}}</tool_call>')
        for index, text in enumerate(fragments):
            vocab["".join(byte_map[byte] for byte in text.encode("utf-8"))] = 1400 + index
    vocab.update(special)
    check(len(set(vocab.values())) == len(vocab), "Duplicate fake vocabulary ID")
    check(max(vocab.values()) == 2047, "Fake vocabulary must report exactly 2048 IDs")
    template = (
        "{%- for message in messages -%}"
        "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' -}}"
        "{%- endfor -%}"
        "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' -}}"
        "{%- if enable_thinking -%}{{- '<think>\\n' -}}{%- endif -%}{%- endif -%}"
    )
    if tools:
        template = ""  # Exercise the service's real dense/Hybrid tool fallbacks.
    (directory / "vocab.json").write_bytes(json_bytes(vocab))
    (directory / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    write_json(directory / "tokenizer_config.json", {
        "tokenizer_class": "Qwen2Tokenizer",
        "bos_token": None,
        "eos_token": "<|im_end|>",
        "chat_template": template,
        "added_tokens_decoder": {
            str(token_id): {"content": text, "special": True}
            for text, token_id in special.items()
        },
    })
    write_json(directory / "config.json", {
        "model_type": "qwen3" if backend == "qwen3_nf4" else "qwen3_5", "vocab_size": 2048,
    })
    manifest = {
        "kind": "AilaFakeInferenceWorker HTTP tool fixture" if tools else "AilaFakeInferenceWorker byte-BPE HTTP fixture",
        "backend": backend,
        "vocab_size": 2048,
        "output_token_ids": [0, 999],
        "output_token_utf8_bytes": len(fake_text(0).encode("utf-8")),
        "ordinary_byte_ids": [1000, 1256],
        "excluded_ordinary_id": 1023,
        "crash_special_token": "<|crash_worker|>",
        "merges": 0,
        "files": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.iterdir()) if path.is_file()
        },
    }
    write_json(directory / "fixture.json", manifest)
    return manifest


def new_connection(port, timeout, receive_buffer=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    if receive_buffer is None:
        connection.connect()
    else:
        # Set the receive window before the handshake, not after a large initial
        # window could already have admitted the whole synthetic response.
        raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            raw.settimeout(timeout)
            raw.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, receive_buffer)
            raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            raw.connect(("127.0.0.1", port))
            connection.sock = raw
        except BaseException:
            raw.close()
            raise
    return connection


def read_interim_header(raw):
    # HTTPResponse.begin intentionally skips 100; read precisely this interim
    # header first so the body is sent only after a real network 100 Continue.
    data = bytearray()
    while not data.endswith(b"\r\n\r\n"):
        byte = raw.recv(1)
        check(byte, "EOF before Expect: 100-continue response")
        data.extend(byte)
        check(len(data) <= FRAME_LIMIT, "Oversized interim HTTP header")
    check(data.split(b"\r\n", 1)[0].startswith(b"HTTP/1.1 100 "), f"Expected 100 Continue, got {bytes(data)!r}")
    return bytes(data)


def raw_request(method, path, port, body=b"", extra_headers=None):
    headers = {
        "Host": f"127.0.0.1:{port}",
        "Connection": "keep-alive",
        "Content-Length": str(len(body)),
    }
    if body:
        headers["Content-Type"] = "application/json"
    headers.update(extra_headers or {})
    start = f"{method} {path} HTTP/1.1\r\n"
    return (start + "".join(f"{key}: {value}\r\n" for key, value in headers.items()) + "\r\n").encode("ascii") + body


def read_raw_to_close(raw, timeout=8.0, reader=None):
    deadline = time.perf_counter() + timeout
    data = bytearray()
    reset = False
    while True:
        remaining = deadline - time.perf_counter()
        check(remaining > 0, "Connection did not close before the socket deadline")
        raw.settimeout(remaining)
        try:
            part = reader.read1(4096) if reader is not None else raw.recv(4096)
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            reset = True
            break
        except socket.timeout as error:
            raise AssertionError("Connection was not force-closed before the socket deadline") from error
        if not part:
            break
        data.extend(part)
        check(len(data) <= MAX_WIRE_BYTES, "Unexpected unbounded response while waiting for close")
    return bytes(data), reset


def validate_usage(value, maximum):
    check(isinstance(value, dict), "Missing usage object")
    prompt = integer(value.get("prompt_tokens"), "usage.prompt_tokens", 1)
    completion = integer(value.get("completion_tokens"), "usage.completion_tokens", 1)
    check(completion <= maximum, f"Generated {completion} tokens, requested at most {maximum}")
    check(integer(value.get("total_tokens"), "usage.total_tokens") == prompt + completion, "Usage counts do not add up")
    return value


def validate_tool_calls(calls):
    check(isinstance(calls, list) and 0 < len(calls) <= 128, "Invalid tool call array")
    ids = set()
    for call in calls:
        check(isinstance(call, dict) and set(call) == {"id", "type", "function"}, "Invalid tool call fields")
        identity = call.get("id")
        check(isinstance(identity, str) and identity and identity not in ids, "Missing or duplicate tool call ID")
        ids.add(identity)
        check(call.get("type") == "function", "Unsupported output tool type")
        function = call.get("function")
        check(isinstance(function, dict) and set(function) == {"name", "arguments"}, "Invalid output function fields")
        check(isinstance(function["name"], str) and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", function["name"]), "Invalid output function name")
        check(isinstance(function["arguments"], str) and isinstance(strict_json(function["arguments"]), dict),
              "Tool arguments are not a string containing a strict JSON object")
    return calls


def validate_completion(value, maximum, model_id, allow_tools=False):
    check(value.get("object") == "chat.completion", "Wrong completion object")
    check(isinstance(value.get("id"), str) and value["id"], "Missing completion ID")
    check(value.get("model") == model_id, "Wrong served model")
    integer(value.get("created"), "completion.created", 1)
    choices = value.get("choices")
    check(isinstance(choices, list) and len(choices) == 1, "Expected one completion choice")
    choice = choices[0]
    reasons = ("length", "stop", "tool_calls") if allow_tools else ("length", "stop")
    check(choice.get("index") == 0 and choice.get("finish_reason") in reasons, "Invalid completion choice")
    message = choice.get("message")
    check(isinstance(message, dict) and message.get("role") == "assistant", "Missing assistant message")
    calls = message.get("tool_calls")
    if calls is not None:
        check(allow_tools, "Text-only response exposed a tool call")
        validate_tool_calls(calls)
    check(isinstance(message.get("content"), str) or (calls and message.get("content") is None), "Invalid completion content")
    check(choice["finish_reason"] != "tool_calls" or calls, "Tool terminal has no calls")
    check(isinstance(message.get("reasoning_content", ""), str), "Non-text reasoning")
    usage = validate_usage(value.get("usage"), maximum)
    semantic = {
        "content": message["content"] or "", "reasoning_content": message.get("reasoning_content", ""),
        "finish_reason": choice["finish_reason"], "usage": usage,
    }
    if calls is not None:
        semantic["tool_calls"] = calls
    return semantic


class SseClient:
    def __init__(self, runner, label, payload, receive_buffer=None, expect_continue=False):
        self.runner = runner
        self.label = label
        self.payload = payload
        self.started = time.perf_counter()
        self.deadline = self.started + runner.read_timeout
        self.connection = new_connection(runner.process.port, runner.read_timeout, receive_buffer)
        self.socket = self.connection.sock
        self.receive_buffer = self.socket.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        self.response = None
        self.closed = False
        self.wire = None
        self.buffer = bytearray()
        self.utf8 = codecs.getincrementaldecoder("utf-8")("strict")
        self.reads = self.bytes_received = self.frame_count = self.content_frames = 0
        self.done_count = self.finish_count = self.role_count = self.usage_count = 0
        self.id = self.header_id = None
        self.created = None
        self.content = []
        self.reasoning = []
        self.tool_calls = []
        self.allow_tools = bool(payload.get("tools")) and payload.get("tool_choice") != "none"
        self.errors = []
        self.usage = None
        self.finish_reason = None
        self.first_content_at = self.finished_at = None
        self.interim_header = None
        self.artifact = runner.responses / f"{label}.sse"
        try:
            body = json_bytes(payload)
            self.connection.putrequest("POST", "/v1/chat/completions")
            self.connection.putheader("Content-Type", "application/json")
            self.connection.putheader("Accept", "text/event-stream")
            self.connection.putheader("Connection", "close")
            self.connection.putheader("Content-Length", str(len(body)))
            if expect_continue:
                self.connection.putheader("Expect", "100-continue")
            self.connection.endheaders()
            if expect_continue:
                self.interim_header = read_interim_header(self.socket).decode("ascii")
            self.connection.send(body)
            self.response = self.connection.getresponse()
            self.headers_at = time.perf_counter()
            self.header_id = self.response.getheader("X-Request-Id")
            if self.response.status != 200:
                raise AssertionError(f"{label}: HTTP {self.response.status}: {self.response.read(4096)!r}")
            check((self.response.getheader("Content-Type") or "").lower().startswith("text/event-stream"), "Missing SSE Content-Type")
            check((self.response.getheader("Transfer-Encoding") or "").lower() == "chunked", "SSE response is not chunked")
            check((self.response.getheader("Connection") or "").lower() == "close", "SSE does not advertise its final close")
            check(self.response.getheader("Content-Encoding") is None, "SSE compression unexpectedly enabled")
            self.wire = self.artifact.open("wb")
            with runner.clients_lock:
                runner.clients.append(self)
        except BaseException:
            self.close()
            raise

    def next_event(self):
        while True:
            while b"\n\n" not in self.buffer:
                remaining = self.deadline - time.perf_counter()
                check(remaining > 0, f"{self.label}: SSE request deadline exceeded")
                self.socket.settimeout(remaining)
                part = self.response.read1(257)
                if not part:
                    self.utf8.decode(b"", final=True)
                    check(not self.buffer, f"{self.label}: EOF split an SSE frame")
                    return False
                self.utf8.decode(part, final=False)
                self.reads += 1
                self.bytes_received += len(part)
                check(self.bytes_received <= OUTPUT_LIMIT, "SSE output exceeded max-output-bytes")
                self.wire.write(part)
                self.buffer.extend(part)
                check(len(self.buffer) <= FRAME_LIMIT + 257, "Single SSE frame exceeded the handoff limit")
            raw, _, tail = self.buffer.partition(b"\n\n")
            self.buffer = bytearray(tail)
            frame = raw.decode("utf-8", errors="strict")
            fields = []
            for line in frame.splitlines():
                if line.startswith("data:"):
                    fields.append(line[5:].removeprefix(" "))
                elif line and not line.startswith(":"):
                    check(line.startswith(("event:", "id:", "retry:")), f"Malformed SSE field: {line[:100]!r}")
            if not fields:
                continue
            self.frame_count += 1
            data = "\n".join(fields)
            check(self.done_count == 0, "SSE data followed [DONE]")
            if data == "[DONE]":
                self.done_count += 1
                self.finished_at = time.perf_counter()
                return True
            event = strict_json(data)
            check(isinstance(event, dict), "SSE data is not a JSON object")
            if "error" in event:
                check(isinstance(event["error"], dict), "Malformed SSE error")
                check(self.finish_count == 0 and not self.errors, "Duplicated or contradictory SSE failure terminal")
                self.errors.append(event["error"])
                return True
            check(not self.errors, "Normal SSE data followed an error")
            check(event.get("object") == "chat.completion.chunk" and event.get("model") == self.runner.model_id,
                  "Invalid SSE envelope")
            identity = event.get("id")
            check(isinstance(identity, str) and identity, "Missing SSE ID")
            if self.id is None:
                self.id = identity
                self.created = integer(event.get("created"), "SSE.created", 1)
                check(not self.header_id or self.header_id == identity, "HTTP request ID differs from SSE ID")
            check(identity == self.id and event.get("created") == self.created, "Crossed request IDs or timestamps in a stream")
            choices = event.get("choices")
            check(isinstance(choices, list), "Missing SSE choices")
            if not choices:
                check(self.payload.get("stream_options", {}).get("include_usage") is True, "Unrequested usage frame")
                check(self.finish_count == 1 and self.usage_count == 0, "Usage arrived before finish or more than once")
                self.usage = validate_usage(event.get("usage"), self.payload["max_tokens"])
                self.usage_count += 1
                return True
            check(len(choices) == 1 and self.finish_count == 0, "Invalid choice count or delta after finish")
            choice = choices[0]
            check(choice.get("index") == 0, "Nonzero SSE choice index")
            delta = choice.get("delta")
            check(isinstance(delta, dict), "Missing SSE delta")
            fields = {"role", "content", "reasoning_content"} | ({"tool_calls"} if self.allow_tools else set())
            check(set(delta) <= fields, "Unexpected SSE delta field")
            include_usage = self.payload.get("stream_options", {}).get("include_usage") is True
            check(("usage" in event and event["usage"] is None) if include_usage else "usage" not in event, "Incorrect optional usage field")
            if "role" in delta:
                check(self.role_count == 0 and self.frame_count == 1 and delta["role"] == "assistant", "Invalid or repeated initial role")
                self.role_count += 1
            check(self.role_count == 1, "Content preceded the assistant role")
            emitted = False
            if "tool_calls" in delta:
                check(isinstance(delta["tool_calls"], list) and delta["tool_calls"], "Empty tool delta")
                for call in delta["tool_calls"]:
                    index = integer(call.get("index"), "tool delta index")
                    check(index <= len(self.tool_calls), "Tool delta skipped an index")
                    function = call.get("function")
                    check(isinstance(function, dict) and isinstance(function.get("arguments"), str), "Invalid tool delta function")
                    if index == len(self.tool_calls):
                        check(set(call) == {"index", "id", "type", "function"} and
                              set(function) == {"name", "arguments"}, "Initial tool delta is missing metadata")
                        self.tool_calls.append({"id": call["id"], "type": call["type"],
                                                "function": {"name": function["name"], "arguments": ""}})
                    else:
                        check(set(call) == {"index", "function"} and set(function) == {"arguments"},
                              "Tool continuation repeated metadata")
                    self.tool_calls[index]["function"]["arguments"] += function["arguments"]
                emitted = True
            for key, output in (("content", self.content), ("reasoning_content", self.reasoning)):
                if key in delta:
                    check(isinstance(delta[key], str), f"Non-string {key}")
                    output.append(delta[key])
                    emitted |= bool(delta[key])
            if emitted:
                self.content_frames += 1
                if self.first_content_at is None:
                    self.first_content_at = time.perf_counter()
            reason = choice.get("finish_reason")
            if reason is not None:
                check(reason in (("length", "stop", "tool_calls") if self.allow_tools else ("length", "stop")) and not delta,
                      "Invalid successful SSE terminal")
                check(reason != "tool_calls" or self.tool_calls, "Tool terminal has no calls")
                self.finish_reason = reason
                self.finish_count += 1
            return True

    def first_content(self):
        while self.first_content_at is None:
            check(self.next_event(), "EOF before the first content delta")
            check(not self.done_count and not self.errors, "Terminal arrived before content")
        return self.id

    def first_role(self):
        while self.role_count == 0:
            check(self.next_event(), "EOF before the initial role")
            check(not self.done_count and not self.errors, "Terminal arrived before the initial role")
        return self.id

    def complete(self, allow_error=False, expected_generated=None):
        while self.next_event():
            pass
        check(self.done_count == 1, "Missing unique [DONE] terminal")
        if self.errors:
            check(allow_error and len(self.errors) == 1 and self.finish_count == 0, "Unexpected failed SSE completion")
        else:
            check(self.role_count == 1 and self.finish_count == 1, "Missing role or successful finish")
            if self.tool_calls:
                validate_tool_calls(self.tool_calls)
            expected_usage = self.payload.get("stream_options", {}).get("include_usage") is True
            check(self.usage_count == int(expected_usage), "Wrong final usage count")
            if self.runner.mode == "fake" and self.usage:
                expected = self.payload["max_tokens"] if expected_generated is None else expected_generated
                check(self.usage["completion_tokens"] == expected, "Fake request lost generated tokens")
        self.wire.flush()
        return self.summary()

    def semantic(self):
        result = {"content": "".join(self.content), "reasoning_content": "".join(self.reasoning),
                  "finish_reason": self.finish_reason, "usage": self.usage}
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        return result

    def summary(self):
        return {
            "id": self.id or self.header_id, "headers_seconds": self.headers_at - self.started,
            "first_content_seconds": None if self.first_content_at is None else self.first_content_at - self.started,
            "duration_seconds": None if self.finished_at is None else self.finished_at - self.started,
            "started_at": self.started, "first_content_at": self.first_content_at, "done_at": self.finished_at,
            "frames": self.frame_count, "content_frames": self.content_frames, "read_fragments": self.reads,
            "wire_payload_bytes": self.bytes_received, "receive_buffer": self.receive_buffer,
            "finish_reason": self.finish_reason, "usage": self.usage, "errors": self.errors,
            "content_sha256": sha256("".join(self.content)), "reasoning_sha256": sha256("".join(self.reasoning)),
            "tool_call_count": len(self.tool_calls), "tool_calls_sha256": sha256(json.dumps(self.tool_calls, ensure_ascii=False)),
            "artifact": str(self.artifact),
        }

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        if self.response is not None:
            self.response.close()
        self.connection.close()
        if self.wire is not None:
            self.wire.close()


class ServerProcess:
    def __init__(self, args, directory, model):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.shutdown_file = directory / f"shutdown-{uuid.uuid4().hex}.flag"
        self.report_file = directory / "report.json"
        self.shutdown_timeout = args.shutdown_timeout
        self.stopped = False
        self.cleanup = {}
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
            reservation.bind(("127.0.0.1", 0))
            self.port = reservation.getsockname()[1]
        self.command = [
            str(args.server), "--worker", str(args.worker), "--model", str(model),
            "--backend", args.backend, "--model-id", args.model_id, "--port", str(self.port), "--context", str(getattr(args, "context", 512)),
            "--sessions", "4", "--max-new-tokens", "256", "--max-output-bytes", str(OUTPUT_LIMIT),
            "--stream-buffer-bytes", str(STREAM_LIMIT), "--stream-stall-ms", "5000",
            "--request-timeout-ms", "120000", "--shutdown-file", str(self.shutdown_file),
            "--report", str(self.report_file),
        ]
        if getattr(args, "memory_sample_ms", 0):
            self.command += ["--memory-sample-ms", str(args.memory_sample_ms)]
        if getattr(args, "decode_batch_size", None) is not None:
            self.command += ["--decode-batch-size", str(args.decode_batch_size)]
        for attribute, flag in (("max_device_bytes", "--max-device-bytes"),
                                ("max_pinned_host_bytes", "--max-pinned-host-bytes"),
                                ("max_session_host_bytes", "--max-session-host-bytes")):
            if getattr(args, attribute, None) is not None:
                self.command += [flag, str(getattr(args, attribute))]
        self.stdout = (directory / "stdout.log").open("wb")
        self.stderr = (directory / "stderr.log").open("wb")
        launch = {"created_at": utc_now(), "command": self.command, "cwd": str(directory),
                  "mode": args.mode, "backend": args.backend, "model_id": args.model_id,
                  "requested_model": None if args.model is None else str(args.model),
                  "effective_model": str(model), "shutdown_file": str(self.shutdown_file),
                  "report_file": str(self.report_file), "creationflags": "CREATE_NO_WINDOW" if os.name == "nt" else 0}
        launch["artifacts"] = [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                               for path in (args.server, args.worker, Path(__file__).resolve())]
        write_json(directory / "launch.json", launch)
        try:
            self.process = subprocess.Popen(
                self.command, cwd=directory, stdin=subprocess.DEVNULL,
                stdout=self.stdout, stderr=self.stderr, close_fds=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except BaseException:
            self.stdout.close()
            self.stderr.close()
            raise
        launch["pid"] = self.process.pid
        write_json(directory / "launch.json", launch)

    def stop(self):
        if self.stopped:
            return self.cleanup
        started = time.perf_counter()
        forced = False
        shutdown_written = False
        interrupted = False
        try:
            if self.process.poll() is None:
                self.shutdown_file.write_text("Owned socket acceptance runner requested shutdown.\n", encoding="utf-8")
                shutdown_written = True
                deadline = time.perf_counter() + self.shutdown_timeout
                while self.process.poll() is None:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        forced = True
                        # Popen retains this exact child's process handle on
                        # Windows; no unrelated PID or task name is involved.
                        self.process.kill()
                        self.process.wait(timeout=10)
                        break
                    try:
                        self.process.wait(timeout=remaining)
                    except subprocess.TimeoutExpired:
                        continue
                    except KeyboardInterrupt:
                        # An interrupt must not mark a still-running child as
                        # stopped or skip its already-requested grace period.
                        interrupted = True
            self.stopped = self.process.poll() is not None
        finally:
            self.stdout.close()
            self.stderr.close()
        self.cleanup = {
            "pid": self.process.pid, "shutdown_file_written": shutdown_written, "forced_kill": forced,
            "interrupted_during_shutdown": interrupted,
            "exit_code": self.process.returncode, "seconds": time.perf_counter() - started,
            "report_path": str(self.report_file),
            "framework_errors": framework_errors(self.directory),
        }
        if self.report_file.is_file():
            self.cleanup["report"] = strict_json(self.report_file.read_bytes())
        return self.cleanup


class Runner:
    def __init__(self, args, process, directory, result):
        self.args = args
        self.mode = args.mode
        self.backend = args.backend
        self.model_id = args.model_id
        self.process = process
        self.directory = directory
        self.responses = directory / "responses"
        self.responses.mkdir(parents=True, exist_ok=True)
        self.result = result
        self.read_timeout = 15.0 if self.mode == "fake" else 125.0
        self.clients = []
        self.clients_lock = threading.Lock()
        self.ids = set()
        self.last_health = None
        self.health_samples = []
        self.started = time.perf_counter()

    def chat(self, prompt, maximum, stream=False, seed=42, temperature=0.0, include_usage=True):
        payload = {"model": self.model_id, "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": maximum, "seed": seed, "temperature": temperature,
                   "stream": stream, "enable_thinking": False}
        if stream:
            payload["stream_options"] = {"include_usage": include_usage}
        return payload

    def json_call(self, label, path, method="GET", payload=None, expected=200, timeout=None, save=True):
        started = time.perf_counter()
        connection = new_connection(self.process.port, timeout or self.read_timeout)
        try:
            body = payload if isinstance(payload, bytes) else None if payload is None else json_bytes(payload)
            headers = {"Connection": "close", "Accept": "application/json"}
            if body is not None:
                headers["Content-Type"] = "application/json"
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            raw = response.read(OUTPUT_LIMIT + 1)
            if save:
                (self.responses / f"{label}.json").write_bytes(raw)
            wanted = (expected,) if isinstance(expected, int) else expected
            check(response.status in wanted, f"{label}: expected HTTP {wanted}, got {response.status}: {raw[:300]!r}")
            check(len(raw) <= OUTPUT_LIMIT, f"{label}: JSON output exceeded limit")
            check((response.getheader("Content-Type") or "").lower().startswith("application/json"), f"{label}: non-JSON Content-Type")
            value = strict_json(raw)
            check(isinstance(value, dict), f"{label}: expected JSON object")
            return value, {"status": response.status, "seconds": time.perf_counter() - started, "bytes": len(raw)}
        finally:
            connection.close()

    def health(self, allow_fault=False):
        value, timing = self.json_call("health", "/health", expected=(200, 503) if allow_fault else 200,
                                       timeout=2.0, save=False)
        self.last_health = value
        sample = {"at_seconds": time.perf_counter() - self.started, "seconds": timing["seconds"],
                  "status": value.get("status"), "active_requests": value.get("active_requests")}
        self.health_samples.append(sample)
        check(timing["seconds"] < 2.0, f"Health endpoint blocked for {timing['seconds']:.3f} seconds")
        self.validate_health(value)
        check(value.get("model") == self.model_id and value.get("backend") == self.backend,
              "Health reports a different served model or backend")
        if getattr(self.args, "decode_batch_size", None) is not None:
            check(value.get("max_decode_batch_size") == self.args.decode_batch_size,
                  "Health reports a different decode batch bound")
        return value

    @staticmethod
    def validate_health(value):
        check(value.get("status") in ("ok", "starting", "stopping", "faulted", "closed"), "Unknown health status")
        integer(value.get("active_requests"), "active_requests")
        for key in ("admitted", "completed", "failed", "cancelled", "disconnect_cancellations", "slow_client_cancellations"):
            integer(value["counters"].get(key), f"counters.{key}")
        transport = value["transport"]
        check(transport.get("stream_limit_bytes") == STREAM_LIMIT, "Unexpected stream budget")
        check(transport.get("frame_limit_bytes") == FRAME_LIMIT, "Unexpected frame budget")
        # Parser-generated writes on deliberate malformed/reused connections
        # are outside our producer gate. Check the stream bound on each normal
        # request record, not on this lifetime high-water value for all traffic.
        integer(transport.get("peak_pending_bytes"), "peak_pending_bytes")
        check(integer(transport.get("peak_handoff_bytes"), "peak_handoff_bytes") <= FRAME_LIMIT, "Single-request handoff exceeded frame budget")
        integer(transport.get("pause_ticks"), "pause_ticks")
        ipc = value["ipc"]
        check(isinstance(ipc.get("epoch"), str) and ipc["epoch"], "Missing Worker epoch")
        check(ipc.get("status") in ("starting", "ready", "stopping", "faulted", "closed"), "Unknown IPC state")
        integer(ipc.get("outstanding_requests"), "ipc.outstanding_requests")
        integer(ipc.get("pending_commands"), "ipc.pending_commands")
        sockets = value["connections"]
        for key in ("accepted", "closed", "socket_option_callbacks", "reused_rejected"):
            integer(sockets.get(key), f"connections.{key}")
        check(sockets["closed"] <= sockets["accepted"], "Connection-close callbacks exceed accepted sockets")
        check(sockets["socket_option_callbacks"] > 0, "Windows after-accept callback was not reached")
        check(sockets.get("socket_option_failures", 0) == 0, "SO_SNDBUF setup failed")
        check(0 < sockets["min_send_buffer"] <= sockets["max_send_buffer"] <= SEND_BUFFER * 2,
              f"Unexpected send-buffer readback: {sockets}")
        recent = value.get("recent_requests")
        check(isinstance(recent, list) and len(recent) <= 64, "Unbounded or absent recent request summaries")
        for record in recent:
            check(isinstance(record.get("id"), str) and record["id"], "Missing retained request ID")
            pending = integer(record.get("peak_pending_bytes"), "request.peak_pending_bytes")
            if record.get("status") not in ("connection_reused", "transport_limit"):
                check(pending <= STREAM_LIMIT, "Ordinary request exceeded pending-byte budget")
            integer(record.get("pause_ticks"), "request.pause_ticks")
            if record.get("generated_tokens") is not None:
                integer(record["generated_tokens"], "request.generated_tokens")

    def wait_health(self, predicate, timeout=6.0, allow_fault=False, description="health condition"):
        deadline = time.perf_counter() + timeout
        while True:
            value = self.health(allow_fault=allow_fault)
            if predicate(value):
                return value
            check(time.perf_counter() < deadline, f"Timed out waiting for {description}; last={value}")
            time.sleep(0.025)

    def idle(self, timeout=8.0, allow_fault=False):
        return self.wait_health(lambda item: item["active_requests"] == 0 and
                                item["ipc"]["outstanding_requests"] == 0 and item["ipc"]["pending_commands"] == 0,
                                timeout, allow_fault, "request/session release and IPC drain")

    @staticmethod
    def record(value, identity):
        return next((item for item in value["recent_requests"] if item["id"] == identity), None)

    def wait_record(self, identity, timeout=8.0):
        value = self.wait_health(lambda item: self.record(item, identity) is not None,
                                 timeout, description=f"retired request {identity}")
        return self.record(value, identity)

    def remember_id(self, identity):
        check(identity and identity not in self.ids, f"Completion ID was reused: {identity}")
        self.ids.add(identity)

    def completion(self, label, payload):
        value, timing = self.json_call(label, "/v1/chat/completions", "POST", payload)
        semantic = validate_completion(value, payload["max_tokens"], self.model_id)
        self.remember_id(value["id"])
        if self.mode == "fake":
            check(semantic["usage"]["completion_tokens"] == payload["max_tokens"], "Fake JSON response lost tokens")
        return semantic, {"id": value["id"], **timing, "content_sha256": sha256(semantic["content"]),
                          "reasoning_sha256": sha256(semantic["reasoning_content"]), "usage": semantic["usage"]}

    def stream(self, label, payload, **kwargs):
        return SseClient(self, label, payload, **kwargs)

    def close_clients(self):
        with self.clients_lock:
            clients = self.clients[:]
        for client in clients:
            client.close()

    def case(self, name, function):
        entry = {"name": name, "passed": False, "started_at": utc_now()}
        self.result["tests"].append(entry)
        started = time.perf_counter()
        print(f"RUN {name}", flush=True)
        try:
            entry["details"] = function()
            entry["passed"] = True
            print(f"PASS {name}", flush=True)
        except BaseException as error:
            entry["error"] = f"{type(error).__name__}: {error}"
            entry["traceback"] = traceback.format_exc()
            print(f"FAIL {name}: {error}", flush=True)
            raise
        finally:
            entry["seconds"] = time.perf_counter() - started
            self.close_clients()
            self.result["health_latency_samples"] = self.health_samples
            self.result["last_health"] = self.last_health
            write_json(self.directory / "result.json", self.result)

    def wait_ready(self):
        deadline = time.perf_counter() + self.args.startup_timeout
        last_error = None
        while time.perf_counter() < deadline:
            check(self.process.process.poll() is None,
                  f"Server exited before readiness (exit={self.process.process.returncode}); see stdout.log/stderr.log")
            try:
                value = self.health(allow_fault=True)
                if value["status"] == "ok":
                    return value
                last_error = f"status={value['status']}"
            except (OSError, http.client.HTTPException) as error:
                last_error = str(error)
            time.sleep(0.1)
        raise AssertionError(f"Server readiness deadline exceeded: {last_error}")

    def basics(self):
        healthy = self.health()
        models, timing = self.json_call("models", "/v1/models")
        check(models.get("object") == "list" and isinstance(models.get("data"), list), "Invalid model list")
        check([item.get("id") for item in models["data"]] == [self.model_id], "Model list does not identify the loaded checkpoint")
        valid = self.chat("hello", 8)
        malformed = [
            ("invalid_json", b'{"model":'),
            ("unknown_parameter", {**valid, "unknown_option": 1}),
            ("negative_seed", {**valid, "seed": -1}),
            ("too_many_tokens", {**valid, "max_tokens": 257}),
            ("multiple_choices", {**valid, "n": 2}),
            ("usage_without_stream", {**valid, "stream_options": {"include_usage": True}}),
            ("image_content", {**valid, "messages": [{"role": "user", "content":
                [{"type": "image_url", "image_url": {"url": "https://invalid.example/image.png"}}]}]}),
        ]
        invalid_results = []
        for label, payload in malformed:
            error, result = self.json_call(label, "/v1/chat/completions", "POST", payload, expected=400)
            check(isinstance(error.get("error"), dict) and isinstance(error["error"].get("message"), str), "Invalid API error envelope")
            invalid_results.append({"case": label, **result, "error": error["error"]})
            self.idle()
        return {"initial_health": healthy, "models": models, "model_query": timing, "invalid_parameters": invalid_results}

    def incremental(self):
        maximum = 64 if self.mode == "fake" else 24
        prompt = "Explain briefly why leaves are green."
        reference, reference_stats = self.completion("incremental_reference", self.chat(prompt, maximum))
        self.idle()
        client = self.stream("incremental", self.chat(prompt, maximum, stream=True))
        client.first_content()
        during = self.health()
        check(during["active_requests"] >= 1, "First content was only visible after request reclamation")
        summary = client.complete()
        self.remember_id(summary["id"])
        check(client.semantic() == reference, "JSON and incremental SSE output or usage differ")
        check(summary["content_frames"] >= 2, "Response was delivered as one accumulated completion")
        check(summary["first_content_at"] < summary["done_at"], "First content did not precede the terminal")
        check(summary["read_fragments"] > summary["content_frames"], "Socket reads did not segment SSE frames")
        client.close()
        retained = self.wait_record(summary["id"])
        check(retained["status"] == "completed", "Successful stream was not retired as completed")
        self.idle()
        no_usage = self.stream("usage_disabled", self.chat("Say hello.", 8, stream=True, include_usage=False))
        no_usage_stats = no_usage.complete()
        self.remember_id(no_usage_stats["id"])
        no_usage.close()
        self.idle()
        return {"reference": reference_stats, "stream": summary, "active_at_first_content": during["active_requests"],
                "retained": retained, "usage_disabled": no_usage_stats}

    def concurrency(self):
        maximum = 40 if self.mode == "fake" else 16
        prompts = [f"Describe a useful property of number {index + 2}." for index in range(4)]
        reference = []
        for index, prompt in enumerate(prompts):
            semantic, timing = self.completion(f"seed_reference_{index}", self.chat(prompt, maximum, seed=700 + index, temperature=0.6))
            reference.append(semantic)
            self.idle()
        groups = []
        for width in (1, 2, 3, 4):
            barrier = threading.Barrier(width)
            peak_active = 0

            def run_one(index):
                barrier.wait(timeout=10)
                client = self.stream(f"concurrent_{width}_{index}",
                                     self.chat(prompts[index], maximum, stream=True, seed=700 + index, temperature=0.6))
                try:
                    stats = client.complete()
                    check(client.semantic() == reference[index], f"Request {index} changed under {width}-way concurrency")
                    return stats
                finally:
                    client.close()

            with concurrent.futures.ThreadPoolExecutor(max_workers=width) as pool:
                futures = [pool.submit(run_one, index) for index in range(width)]
                deadline = time.perf_counter() + self.read_timeout + 5
                while not all(future.done() for future in futures):
                    peak_active = max(peak_active, self.health()["active_requests"])
                    check(time.perf_counter() < deadline, "Concurrent requests exceeded their deadline")
                    time.sleep(0.015)
                stats = [future.result() for future in futures]
            for item in stats:
                self.remember_id(item["id"])
            if width > 1:
                check(max(item["started_at"] for item in stats) < min(item["done_at"] for item in stats), "Clients did not overlap")
            check(peak_active >= width, f"Never observed {width} admitted requests together (peak={peak_active})")
            groups.append({"width": width, "peak_active": peak_active, "requests": stats})
            self.idle()
        return {"groups": groups, "seed_policy": "Each concurrent request matches its own independent fixed-seed reference."}

    def multiturn_thinking(self):
        maximum = 8 if self.mode == "fake" else 24
        messages = [
            {"role": "system", "content": "Answer briefly, using the details in the conversation."},
            {"role": "user", "content": "Remember the color violet and the number seven. 记住这两个信息。"},
            {"role": "assistant", "content": "I will remember violet and seven.",
             "reasoning_content": "The user supplied a color and a number."},
            {"role": "user", "content": [
                {"type": "text", "text": "Use the remembered color in a sentence. "},
                {"type": "text", "text": "Then multiply the remembered number by three and add two."},
            ]},
        ]
        references = {}
        variants = []
        for thinking in (False, True):
            label = f"multiturn_thinking_{str(thinking).lower()}"
            payload = self.chat("", maximum, seed=126)
            payload["messages"] = messages
            payload["enable_thinking"] = thinking
            reference, reference_stats = self.completion(label + "_reference", payload)
            self.idle()
            client = self.stream(label, {**payload, "stream": True, "stream_options": {"include_usage": True}})
            try:
                summary = client.complete()
                check(client.semantic() == reference, "Multi-turn text/thinking JSON and SSE differ")
                check(reference["content"] or reference["reasoning_content"], "Text/thinking request returned no text")
                for text in (reference["content"], reference["reasoning_content"]):
                    check("<think>" not in text and "</think>" not in text, "Thinking delimiters leaked into response text")
            finally:
                client.close()
            self.remember_id(summary["id"])
            retained = self.wait_record(summary["id"])
            check(retained["status"] == "completed", "Multi-turn stream did not retire as completed")
            self.idle()
            references[thinking] = reference
            variants.append({"enable_thinking": thinking, "reference": reference_stats, "stream": summary,
                             "content_characters": len(reference["content"]),
                             "reasoning_characters": len(reference["reasoning_content"]), "retained": retained})
        check(references[False]["usage"]["prompt_tokens"] != references[True]["usage"]["prompt_tokens"],
              "enable_thinking did not alter the trusted chat-template prefix")

        # Convert the final two text parts into their equivalent string after a
        # thinking request. This also catches parser/Session state surviving a
        # previous request with the opposite thinking mode.
        repeated = self.chat("", maximum, seed=126)
        repeated["messages"] = [*messages[:-1], {
            "role": "user", "content": "".join(part["text"] for part in messages[-1]["content"]),
        }]
        repeated_semantic, repeated_stats = self.completion("multiturn_text_string_after_thinking", repeated)
        check(repeated_semantic == references[False], "Text-part flattening or prior thinking state changed a fresh request")
        self.idle()
        return {"variants": variants, "string_after_thinking": repeated_stats,
                "comparison": "Full content, reasoning_content, finish_reason and token usage; no answer-quality assertion."}

    def thinking_budget(self):
        maximum = 16
        prompt = "What is 7 times 9? State the number, then explain in one sentence."
        variants = []
        payloads = []
        references = []
        aliases = ("max_thinking_tokens", "thinking_budget", "thinking_budget_tokens")
        for index, budget in enumerate((0, 2, 5, -1)):
            payload = self.chat(prompt, maximum, seed=420 + index, temperature=0.6)
            alias = aliases[index % len(aliases)]
            payload[alias] = budget
            if budget > 0:
                payload.pop("enable_thinking")  # A positive explicit budget enables the trusted thinking prefix.
            elif budget == 0:
                payload["enable_thinking"] = True  # Zero budget takes precedence over this switch.
            label = f"thinking_budget_{budget}"
            value, timing = self.json_call(label + "_reference", "/v1/chat/completions", "POST", payload)
            reference = validate_completion(value, maximum, self.model_id)
            self.remember_id(value["id"])
            check(reference["content"].strip(), f"{label}: no answer after thinking budget")
            check("<think>" not in reference["content"] and "</think>" not in reference["content"] and
                  "<think>" not in reference["reasoning_content"] and "</think>" not in reference["reasoning_content"],
                  f"{label}: thinking marker leaked into response text")
            if budget == 0:
                check(not reference["reasoning_content"], "Zero budget did not suppress the thinking prefix")
            if self.mode == "fake":
                check(reference["usage"]["completion_tokens"] == maximum, "Budget dropped fake completion tokens")
                expected_thinking = max(0, budget)
                check(reference["reasoning_content"].count(";\n") == expected_thinking,
                      "Fake reasoning tokens did not match the independent budget")
                # The byte-BPE fixture encodes the closing marker and two newline bytes as 3 tokens.
                expected_content = maximum - expected_thinking - (3 if budget > 0 else 0)
                check(reference["content"].count(";\n") == expected_content,
                      "Forced closing tokens were omitted from usage or sampled as content")
            self.idle()
            payloads.append(payload)
            references.append(reference)
            variants.append({"budget": budget, "alias": alias, "reference": timing, "usage": reference["usage"],
                             "content_characters": len(reference["content"]),
                             "reasoning_characters": len(reference["reasoning_content"])})

        barrier = threading.Barrier(len(payloads))

        def run_one(index):
            barrier.wait(timeout=10)
            payload = {**payloads[index], "stream": True, "stream_options": {"include_usage": True}}
            client = self.stream(f"thinking_budget_stream_{index}", payload)
            try:
                stats = client.complete()
                check(client.semantic() == references[index], "Concurrent thinking budget differs from independent JSON")
                return stats
            finally:
                client.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(run_one, index) for index in range(4)]
            streams = [future.result(timeout=self.read_timeout + 10) for future in futures]
        check(max(item["started_at"] for item in streams) < min(item["done_at"] for item in streams),
              "Budgeted SSE clients did not overlap")
        for stats in streams:
            self.remember_id(stats["id"])
            check(self.wait_record(stats["id"])["status"] == "completed", "Budgeted SSE request was not reclaimed")
        self.idle()

        # A two-token budget cannot fit its complete closing sequence in the last slot.
        boundary = self.chat(prompt, 3, seed=432, temperature=0.0)
        boundary.pop("enable_thinking")
        boundary["max_thinking_tokens"] = 2
        value, boundary_timing = self.json_call("thinking_budget_short_reference", "/v1/chat/completions", "POST", boundary)
        expected = validate_completion(value, 3, self.model_id)
        self.remember_id(value["id"])
        if self.mode == "fake":
            check(expected["usage"]["completion_tokens"] == 2 and expected["finish_reason"] == "length" and
                  not expected["content"] and expected["reasoning_content"].count(";\n") == 2,
                  "Insufficient closing capacity did not produce the exact early Length completion")
        self.idle()
        client = self.stream("thinking_budget_short_stream",
                             {**boundary, "stream": True, "stream_options": {"include_usage": True}})
        try:
            short_stream = client.complete(expected_generated=2)
            check(client.semantic() == expected, "Early Length JSON/SSE counts or termination differ")
        finally:
            client.close()
        self.remember_id(short_stream["id"])
        self.idle()

        invalid = []
        for alias in aliases:
            for value in (-2, True, 1.5, None):
                payload = {**self.chat(prompt, 8), alias: value}
                error, timing = self.json_call(f"budget_invalid_{alias}_{len(invalid)}", "/v1/chat/completions", "POST",
                                               payload, expected=400)
                check(error["error"].get("param") == alias, "Invalid budget did not identify its own field")
                invalid.append({"alias": alias, "value": value, **timing})
        error, _ = self.json_call("budget_conflicting_aliases", "/v1/chat/completions", "POST",
                                  {**self.chat(prompt, 8), "thinking_budget": 2, "max_thinking_tokens": 2}, expected=400)
        check(error["error"].get("code") == "invalid_parameter", "Conflicting budget aliases were accepted")
        self.idle()
        return {"variants": variants, "concurrent_streams": streams, "invalid_parameters": invalid,
                "short_completion": {"reference": boundary_timing, "usage": expected["usage"],
                                     "finish_reason": expected["finish_reason"], "stream": short_stream},
                "comparison": "Independent JSON versus four concurrent SSE budgets, exact content/reasoning/usage; real Worker forced closing."}

    def close_and_expect(self):
        close_client = self.stream("connection_close", self.chat("Describe the sky.", 12, stream=True))
        close_stats = close_client.complete()
        self.remember_id(close_stats["id"])
        close_client.close()
        self.idle()
        expect_client = self.stream("expect_continue", self.chat("Describe the sea.", 12, stream=True), expect_continue=True)
        expect_stats = expect_client.complete()
        self.remember_id(expect_stats["id"])
        interim = expect_client.interim_header
        expect_client.close()
        self.idle()
        return {"connection_close": close_stats, "expect_continue": expect_stats, "interim_header": interim}

    def ownership(self):
        results = []
        before = self.health()["connections"]["reused_rejected"]
        connection = new_connection(self.process.port, 8.0)
        raw = connection.sock
        try:
            first = raw_request("POST", "/v1/chat/completions", self.process.port,
                                json_bytes(self.chat("First pipeline request.", 16, stream=True)))
            second = raw_request("GET", "/v1/models", self.process.port)
            raw.sendall(first + second)  # One write, without waiting for response one.
            data, reset = read_raw_to_close(raw)
            (self.responses / "pipeline_same_write.http").write_bytes(data)
            finals = [int(code) for code in re.findall(rb"(?:^|\r\n)HTTP/1\.[01] ([0-9]{3}) ", data) if int(code) >= 200]
            check(len(finals) <= 1, f"Second application response escaped the pipelining guard: {finals}")
            check(not re.search(rb'"object"\s*:\s*"list"', data), "Second /v1/models response was delivered")
            results.append({"case": "two_requests_one_write", "bytes": len(data), "statuses": finals, "reset": reset})
        finally:
            connection.close()
        self.wait_health(lambda item: item["connections"]["reused_rejected"] > before, description="same-write reuse rejection")
        self.idle()

        for expect in (False, True):
            before = self.health()["connections"]["reused_rejected"]
            label = "second_expect_request" if expect else "second_delayed_request"
            maximum = LONG_FAKE_TOKENS if self.mode == "fake" else 128
            client = self.stream(label, self.chat("Explain a long sequence of simple facts.", maximum, stream=True),
                                 receive_buffer=SLOW_RECEIVE_BUFFER)
            client.first_content()
            if expect:
                body = json_bytes(self.chat("This second request must not run.", 8))
                request = raw_request("POST", "/v1/chat/completions", self.process.port, body,
                                      {"Expect": "100-continue"})
                header, _, suffix = request.partition(b"\r\n\r\n")
                client.socket.sendall(header + b"\r\n\r\n")
                time.sleep(0.025)  # Give the parser's interim-write path a real turn.
                client.socket.sendall(suffix)
            else:
                client.socket.sendall(raw_request("GET", "/v1/models", self.process.port))
            data, reset = read_raw_to_close(client.socket, reader=client.response.fp)
            (self.responses / f"{label}.http-tail").write_bytes(bytes(client.buffer) + data)
            finals = [int(code) for code in re.findall(rb"(?:^|\r\n)HTTP/1\.[01] ([0-9]{3}) ", data) if int(code) >= 200]
            check(not finals, f"A second final HTTP response followed an active SSE: {finals}")
            check(not re.search(rb'"object"\s*:\s*"list"', data), "Second application response was delivered")
            client.close()
            value = self.wait_health(lambda item: item["connections"]["reused_rejected"] > before,
                                     description="delayed request reuse rejection")
            results.append({"case": label, "bytes_after_second_request": len(data), "reset": reset,
                            "reused_rejected": value["connections"]["reused_rejected"]})
            self.idle()

        # Do not let HTTPConnection transparently reconnect: HTTPResponse parses
        # a response on the exact raw socket retained for the second write.
        connection = new_connection(self.process.port, 8.0)
        raw = connection.sock
        response = None
        try:
            raw.sendall(raw_request("GET", "/health", self.process.port))
            response = http.client.HTTPResponse(raw)
            response.begin()
            check(response.status == 200, "Initial reused-connection health request failed")
            strict_json(response.read(OUTPUT_LIMIT))
            send_error = None
            try:
                raw.sendall(raw_request("GET", "/v1/models", self.process.port))
            except OSError as error:
                send_error = str(error)
            data, reset = read_raw_to_close(raw)
            check(not data, "Completed HTTP connection was reused for another response")
            results.append({"case": "reuse_after_health", "reset": reset, "send_error": send_error})
        finally:
            if response is not None:
                response.close()
            connection.close()
        self.idle()
        return {"scenarios": results}

    def disconnect(self):
        maximum = 40 if self.mode == "fake" else 24
        prompt = "Explain one benefit of clear written instructions."
        reference, reference_stats = self.completion("disconnect_reference", self.chat(prompt, maximum, seed=82))
        self.idle()
        before = self.health()["counters"]["disconnect_cancellations"]
        victim = self.stream("disconnect_victim",
                             self.chat("Explain many steps in a careful calculation.",
                                       LONG_FAKE_TOKENS if self.mode == "fake" else 128, stream=True),
                             receive_buffer=SLOW_RECEIVE_BUFFER)
        survivor = self.stream("disconnect_survivor", self.chat(prompt, maximum, stream=True, seed=82))
        try:
            # Receiving B's headers and initial role proves its admission
            # before A disconnects; thread scheduling cannot reverse them.
            survivor.first_role()
            victim.first_content()
            victim_id = victim.id
            during = self.health()
            check(during["active_requests"] >= 2, "Both requests must be active before disconnecting the victim")
            closed_at = time.perf_counter()
            victim.close()
            observed = self.wait_health(lambda item: item["counters"]["disconnect_cancellations"] > before,
                                        timeout=6.0, description="actual disconnect cancellation callback")
            delay = time.perf_counter() - closed_at
            survivor_stats = survivor.complete()
            check(survivor.semantic() == reference, "Disconnect changed the independent survivor output")
        finally:
            survivor.close()
        self.remember_id(survivor_stats["id"])
        retained = self.wait_record(victim_id)
        check(retained["status"] == "disconnect", f"Victim was not cancelled by its socket: {retained}")
        self.idle()
        return {"cancel_seconds": delay, "active_before_disconnect": during["active_requests"],
                "victim": retained, "reference": reference_stats,
                "survivor": survivor_stats, "disconnect_cancellations": observed["counters"]["disconnect_cancellations"]}


    def slow_resume(self):
        peer_prompt = "Describe a small blue object."
        reference, reference_stats = self.completion("slow_peer_reference", self.chat(peer_prompt, 8, seed=92))
        self.idle()
        before = self.health()
        slow = self.stream("slow_resume", self.chat("Emit a long deterministic sequence.", LONG_FAKE_TOKENS, stream=True),
                           receive_buffer=SLOW_RECEIVE_BUFFER)
        slow.first_role()
        paused = self.wait_health(
            lambda item: item["transport"]["pause_ticks"] >= before["transport"]["pause_ticks"] + 2,
            timeout=3.0, description="actual pending-byte backpressure while the socket is unread")
        check(paused["active_requests"] == 1, "The unread stream disappeared before the pause observation")
        check(paused["transport"]["peak_pending_bytes"] > 0, "No Trantor backlog was observed")
        samples = []
        for _ in range(3):
            value = self.health()
            samples.append({"pause_ticks": value["transport"]["pause_ticks"],
                            "peak_pending_bytes": value["transport"]["peak_pending_bytes"],
                            "active_requests": value["active_requests"]})
            time.sleep(0.025)
        peer, peer_stats = self.completion("slow_peer", self.chat(peer_prompt, 8, seed=92))
        check(peer == reference, "An unread connection changed the independent request")
        check(self.health()["active_requests"] >= 1, "The slow request was reclaimed before the client resumed")
        resumed_at = time.perf_counter()
        summary = slow.complete()
        summary["resume_to_done_seconds"] = time.perf_counter() - resumed_at
        self.remember_id(summary["id"])
        slow.close()
        retained = self.wait_record(summary["id"])
        check(retained["status"] == "completed" and retained["pause_ticks"] > 0, "Paused stream did not resume and complete")
        check(0 < retained["peak_pending_bytes"] <= STREAM_LIMIT, "Paused stream exceeded its pending-byte budget")
        end = self.idle()
        check(end["counters"]["slow_client_cancellations"] == before["counters"]["slow_client_cancellations"],
              "Brief read pause was treated as a stalled client")
        return {"stream": summary, "pause_samples": samples, "retained": retained,
                "independent_reference": reference_stats, "independent_peer": peer_stats,
                "scope": "Trantor queued plaintext payload and one handoff frame, excluding kernel buffers and total RSS."}

    def slow_stall(self):
        self.idle()
        before = self.health()
        slow = self.stream("slow_stall", self.chat("Emit another long deterministic sequence.", LONG_FAKE_TOKENS, stream=True),
                           receive_buffer=SLOW_RECEIVE_BUFFER)
        identity = slow.first_role()
        paused = self.wait_health(lambda item: item["transport"]["pause_ticks"] > before["transport"]["pause_ticks"],
                                  timeout=3.0, description="stalled reader reaching transport pressure")
        started = time.perf_counter()
        cancelled = self.wait_health(
            lambda item: item["counters"]["slow_client_cancellations"] > before["counters"]["slow_client_cancellations"],
            timeout=STALL_SECONDS + 6.0, description="slow-client deadline and cancellation")
        data, reset = read_raw_to_close(slow.socket, reader=slow.response.fp)
        (self.responses / "slow_stall.http-tail").write_bytes(bytes(slow.buffer) + data)
        slow.close()
        retained = self.wait_record(identity)
        check(retained["status"] == "slow_client" and retained["pause_ticks"] > 0, "Stall was not attributed to the unread client")
        self.idle()
        _, reuse = self.completion("after_stall", self.chat("Slot reuse after stall.", 8))
        self.idle()
        return {"seconds_after_pause": time.perf_counter() - started, "record": retained,
                "pause_ticks_at_start": paused["transport"]["pause_ticks"], "reset": reset,
                "remaining_socket_bytes": len(data), "slow_client_cancellations": cancelled["counters"]["slow_client_cancellations"],
                "next_request": reuse}

    def admission(self):
        self.idle()
        holders = []
        try:
            for index in range(4):
                client = self.stream(f"capacity_holder_{index}",
                                     self.chat(f"Long capacity request {index}.", LONG_FAKE_TOKENS, stream=True),
                                     receive_buffer=SLOW_RECEIVE_BUFFER)
                client.first_role()
                holders.append(client)
            full = self.wait_health(lambda item: item["active_requests"] == 4, timeout=3.0,
                                    description="four occupied request slots")
            error, rejected = self.json_call("capacity_rejection", "/v1/chat/completions", "POST",
                                             self.chat("Fifth request.", 8), expected=429, timeout=2.0)
            check(isinstance(error.get("error"), dict), "Capacity rejection is not a JSON API error")
            old_id = holders[0].id
            holders[0].close()
            self.wait_health(lambda item: item["active_requests"] == 3, timeout=3.0, description="disconnected capacity slot release")
            replacement = self.stream("capacity_replacement", self.chat("Replacement request.", 8, stream=True))
            replacement_stats = replacement.complete()
            self.remember_id(replacement_stats["id"])
            check(replacement_stats["id"] != old_id, "Replacement reused the disconnected request ID")
            replacement.close()
            return {"peak_active": full["active_requests"], "rejection": rejected, "error": error["error"],
                    "replacement": replacement_stats, "released_request_id": old_id}
        finally:
            for client in holders:
                client.close()
            self.idle()

    def shutdown(self):
        before = self.idle()
        maximum = LONG_FAKE_TOKENS if self.mode == "fake" else 128
        first = self.stream("shutdown_slow", self.chat("Explain several stages of a long process.", maximum, stream=True),
                            receive_buffer=SLOW_RECEIVE_BUFFER)
        first.first_role()
        second = self.stream("shutdown_active", self.chat("Explain several different stages of another process.",
                                                        maximum, stream=True))
        second.first_role()
        during = self.health()
        check(during["active_requests"] >= 1, "Shutdown scenario had no active request")
        # Keep both peer sockets open until the process exits. A successful
        # cleanup must originate at the server, including its blocked producer.
        cleanup = self.process.stop()
        report = self.validate_cleanup(cleanup)
        check(report["counters"]["cancelled"] > before["counters"]["cancelled"], "Active requests were not cancelled by shutdown")
        self.last_health = report
        first.close()
        second.close()
        return {"active_at_shutdown": during["active_requests"], "cleanup": cleanup}

    def validate_cleanup(self, cleanup):
        # Also mandatory for --case runs that omit the active-shutdown case.
        check(not cleanup["forced_kill"], "Server needed the runner's forced process kill")
        check(cleanup["exit_code"] == 0, f"Graceful server exit failed: {cleanup}")
        check(not cleanup["framework_errors"], f"Server transport logged errors: {cleanup['framework_errors']}")
        report = cleanup.get("report")
        check(isinstance(report, dict), "Server did not write its final report")
        self.validate_health(report)
        check(report.get("model") == self.model_id and report.get("backend") == self.backend,
              "Final report lost the served model or backend")
        check(report["shutdown"]["graceful"] is True and report["shutdown"]["idle"] is True, "Shutdown did not finish its close barriers")
        check(report["active_requests"] == 0 and report["ipc"]["outstanding_requests"] == 0 and
              report["ipc"]["pending_commands"] == 0, "Shutdown stranded a request or command")
        check(report["connections"]["accepted"] == report["connections"]["closed"], "Shutdown stranded accepted sockets")
        counts = report["counters"]
        check(counts["admitted"] == counts["completed"] + counts["failed"] + counts["cancelled"],
              "An admitted request has a missing or duplicated final outcome")
        return report


def worker_fault_case(args, directory, model):
    directory.mkdir(parents=True, exist_ok=True)
    process = None
    runner = None
    result = {"mode": "fake", "backend": args.backend, "model_id": args.model_id,
              "tests": [], "purpose": "isolated Worker crash terminal", "passed": False}
    try:
        process = ServerProcess(args, directory, model)
        runner = Runner(args, process, directory, result)
        runner.wait_ready()
        client = runner.stream("worker_fault", runner.chat("<|crash_worker|>", 8, stream=True))
        summary = client.complete(allow_error=True)
        check(len(summary["errors"]) == 1 and summary["finish_reason"] is None, "Worker failure looked like successful text")
        client.close()
        faulted = runner.wait_health(lambda item: item["status"] in ("faulted", "closed"), timeout=6.0,
                                     allow_fault=True, description="Worker crash reaching health")
        check(faulted["status"] != "ok", "Crashed Worker remained healthy")
        error, refusal = runner.json_call("after_worker_fault", "/v1/chat/completions", "POST",
                                          runner.chat("Must be rejected.", 8), expected=503)
        check(isinstance(error.get("error"), dict), "Faulted Worker rejection lost its API error")
        runner.idle(allow_fault=True)
        result.update({"passed": True, "stream": summary, "faulted_health": faulted, "refusal": refusal})
        return result
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        try:
            if runner is not None:
                runner.close_clients()
                result["health_latency_samples"] = runner.health_samples
            if process is not None:
                cleanup = process.stop()
                result["cleanup"] = cleanup
                check(not cleanup["forced_kill"], "Faulted server required a forced process kill")
                # A deliberately dead Worker may make the server report
                # unsuccessful Worker shutdown (exit 2), independently of the
                # HTTP socket and request close barriers.
                check(cleanup["exit_code"] in (0, 2), f"Unexpected faulted-server exit: {cleanup}")
                check(not cleanup["framework_errors"], f"Faulted server transport logged errors: {cleanup['framework_errors']}")
                check(isinstance(cleanup.get("report"), dict), "Faulted server did not write its final report")
                check(cleanup["report"]["shutdown"]["idle"] is True, "Faulted server stranded requests or sockets")
        except BaseException as error:
            result["passed"] = False
            result["cleanup_error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            write_json(directory / "result.json", result)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True, help="Ordinary MSVC AilaServer executable")
    parser.add_argument("--worker", type=Path, required=True, help="AilaWorker or AilaFakeInferenceWorker executable")
    parser.add_argument("--model", type=Path, help="GPU checkpoint directory; fake mode always generates its own fixture")
    parser.add_argument("--mode", choices=("fake", "gpu"), required=True)
    parser.add_argument("--backend", choices=tuple(MODEL_IDS), default="qwen3_nf4")
    parser.add_argument("--model-id", help="Explicit HTTP model name; defaults to qwen3 or qwen3.5 for the selected backend")
    parser.add_argument("--decode-batch-size", type=int, choices=(1, 2, 3, 4),
                        help="Opt in to the configured compatible decode batch bound.")
    parser.add_argument("--case", action="append", choices=COMMON_CASES + FAKE_CASES,
                        help="Run only selected cases; repeat to select more. Omit for the complete suite.")
    parser.add_argument("--output", type=Path, required=True, help="Artifact directory; existing runs are preserved in a new child directory")
    parser.add_argument("--startup-timeout", type=float, default=300.0)
    parser.add_argument("--shutdown-timeout", type=float, default=45.0)
    args = parser.parse_args()
    if args.mode == "gpu" and args.model is None:
        parser.error("--model is required in GPU mode")
    if args.startup_timeout <= 0 or args.shutdown_timeout <= 0:
        parser.error("Timeouts must be positive")
    if args.model_id is None:
        args.model_id = MODEL_IDS[args.backend]
    if not re.fullmatch(r"[A-Za-z0-9._/-]{1,128}", args.model_id):
        parser.error("--model-id must contain 1..128 ASCII model-name characters")
    if args.case:
        if len(args.case) != len(set(args.case)):
            parser.error("--case names must not repeat")
        if args.mode == "gpu" and any(name in FAKE_CASES for name in args.case):
            parser.error("Fake-only cases cannot be selected in GPU mode")
    args.server = args.server.resolve()
    args.worker = args.worker.resolve()
    if args.model is not None:
        args.model = args.model.resolve()
    args.output = args.output.resolve()
    return args


def main():
    args = parse_arguments()
    directory = args.output
    directory.mkdir(parents=True, exist_ok=True)
    if any((directory / name).exists() for name in ("launch.json", "result.json", "report.json", "fake-pipe-model")):
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
        directory = directory / f"run-{stamp}-{uuid.uuid4().hex[:8]}"
        directory.mkdir()
    result = {
        "schema_version": 1, "started_at": utc_now(), "mode": args.mode,
        "backend": args.backend, "model_id": args.model_id, "passed": False,
        "requested_cases": args.case, "full_suite": args.case is None,
        "output": str(directory), "tests": [], "scope": {
            "transport": "Plaintext HTTP/1.1; current Trantor memory/async node bytes and one per-request handoff frame.",
            "excluded": "Total RSS, kernel memory, TCP acknowledgements, HTTP/2, TLS, and proxy behavior.",
            "slow_reader_gate": "Required in a full fake run; not claimed by short-token GPU or excluded cases.",
        },
    }
    process = None
    runner = None
    planned = []
    failed = False
    try:
        check(args.server.is_file(), f"Server executable does not exist: {args.server}")
        check(args.worker.is_file(), f"Worker executable does not exist: {args.worker}")
        if args.mode == "fake":
            model = directory / "fake-pipe-model"
            result["fixture"] = write_fake_tokenizer(model, args.backend)
        else:
            model = args.model
            check(model.is_dir(), f"Model directory does not exist: {model}")
        process = ServerProcess(args, directory, model)
        runner = Runner(args, process, directory, result)
        result["initial_health"] = runner.wait_ready()
        planned = [
            ("health_models_parameters", runner.basics),
            ("incremental_utf8_json_sse", runner.incremental),
            ("concurrent_1_2_4_fixed_seed", runner.concurrency),
            ("multiturn_text_thinking_json_sse", runner.multiturn_thinking),
            ("thinking_budget_json_sse_and_concurrency", runner.thinking_budget),
            ("connection_close_expect_continue", runner.close_and_expect),
            ("same_socket_reuse_pipeline", runner.ownership),
            ("disconnect_isolates_other_request", runner.disconnect),
        ]
        if args.mode == "fake":
            planned.extend([
                ("slow_reader_resume_and_peer", runner.slow_resume),
                ("slow_reader_stall_cleanup", runner.slow_stall),
                ("admission_429_and_slot_reuse", runner.admission),
            ])
        else:
            result["not_applicable"] = [
                {"name": "synthetic_slow_reader_and_capacity_holders",
                 "reason": "GPU token strings may never fill 64 KiB; fake mode provides the deterministic byte-pressure acceptance."}
            ]
        planned.append(("graceful_shutdown_with_active_requests", runner.shutdown))
        if args.mode == "fake":
            planned.append(("worker_fault_error_terminal", lambda: worker_fault_case(args, directory / "worker-fault", model)))
        available = COMMON_CASES + (FAKE_CASES if args.mode == "fake" else ())
        check({name for name, _ in planned} == set(available), "Case registry and implementations differ")
        selected = set(args.case) if args.case else set(available)
        result["excluded_cases"] = [name for name, _ in planned if name not in selected]
        planned = [(name, function) for name, function in planned if name in selected]
        result["planned_cases"] = [name for name, _ in planned]
        for name, function in planned:
            runner.case(name, function)
    except BaseException as error:
        failed = True
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
        print(result["error"], file=sys.stderr, flush=True)
    finally:
        if runner is not None:
            runner.close_clients()
            result["health_latency_samples"] = runner.health_samples
            result["last_health"] = runner.last_health
        if process is not None:
            try:
                result["cleanup"] = process.stop()
                if (result["cleanup"].get("forced_kill") or result["cleanup"].get("exit_code") != 0 or
                        result["cleanup"].get("framework_errors")):
                    failed = True
                if runner is not None:
                    runner.validate_cleanup(result["cleanup"])
                    result["last_health"] = result["cleanup"]["report"]
            except BaseException as error:
                failed = True
                result["cleanup_error"] = f"{type(error).__name__}: {error}"
        completed = {item["name"] for item in result["tests"]}
        result["not_run"] = [name for name, _ in planned if name not in completed]
        result["passed"] = bool(planned) and not failed and all(item["passed"] for item in result["tests"]) and not result["not_run"]
        result["finished_at"] = utc_now()
        if runner is not None and runner.health_samples:
            seconds = sorted(item["seconds"] for item in runner.health_samples)
            result["health_latency"] = {"samples": len(seconds), "maximum_seconds": seconds[-1],
                                        "median_seconds": seconds[len(seconds) // 2]}
        write_json(directory / "result.json", result)
    print(json.dumps({"passed": result["passed"], "tests": len(result["tests"]), "output": str(directory)}, ensure_ascii=False), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
