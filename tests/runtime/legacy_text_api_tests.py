"""Public legacy C ABI -> text adapter -> real pipe/Scheduler -> CPU backend.

The parent installs a hash-checked AilaFakeInferenceWorker.exe under the required
AilaWorker.exe name in an isolated runtime. No GPU runtime is copied or loaded.
The byte tokenizer and token recurrence below are independent output oracles.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import ctypes as c
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from inference_c_api_tests import assert_no_gpu_modules, process_handle_count, until

I32, U32, U64, PTR = c.c_int32, c.c_uint32, c.c_uint64, c.c_void_p


class Config(c.Structure):
    _fields_ = [("max_new_tokens", I32), ("temperature", c.c_float), ("top_k", I32),
                ("top_p", c.c_float), ("repetition_penalty", c.c_float),
                ("presence_penalty", c.c_float), ("frequency_penalty", c.c_float),
                ("do_sample", I32), ("decode_chunk_size", I32), ("stream_chunk_size", I32)]


class ConfigV2(c.Structure):
    _fields_ = [("struct_size", U32), *Config._fields_, ("thinking_budget_tokens", I32),
                ("sampling_seed", U64), ("use_fixed_seed", I32), ("reserved", I32 * 8)]


class ChatEvent(c.Structure):
    _fields_ = [("struct_size", U32), ("type", I32)] + [(name, c.c_char_p) for name in (
        "text", "tool_call_id", "tool_name", "arguments_delta", "finish_reason", "warnings_json", "tool_calls_json")]


TokenCallback = c.CFUNCTYPE(I32, c.c_char_p, PTR)
ChatCallback = c.CFUNCTYPE(I32, c.POINTER(ChatEvent), PTR)
LogCallback = c.CFUNCTYPE(None, I32, c.c_char_p, PTR)


class Api:
    def __init__(self, path):
        self.dll = c.CDLL(str(Path(path).resolve()))
        signatures = {
            "engine_create": ([], PTR), "engine_init": ([PTR, c.c_char_p, I32], I32),
            "engine_destroy": ([PTR], None), "engine_reset_context": ([PTR], None),
            "engine_context_length": ([PTR], I32), "last_error_code": ([PTR], I32),
            "last_error_message": ([PTR], c.c_char_p), "free_string": ([PTR], None),
            "default_gen_config": ([], Config), "default_gen_config_v2": ([], ConfigV2),
            "generate": ([PTR, c.c_char_p, c.POINTER(Config)], PTR),
            "generate_messages": ([PTR, c.c_char_p, c.POINTER(Config)], PTR),
            "generate_chat_json": ([PTR, c.c_char_p, c.POINTER(Config)], PTR),
            "generate_chat_json_ex": ([PTR, c.c_char_p, c.POINTER(ConfigV2)], PTR),
            "generate_stream": ([PTR, c.c_char_p, c.POINTER(Config), TokenCallback, PTR], I32),
            "generate_messages_stream": ([PTR, c.c_char_p, c.POINTER(Config), TokenCallback, PTR], I32),
            "generate_chat_json_stream_ex": ([PTR, c.c_char_p, c.POINTER(ConfigV2), ChatCallback, PTR], I32),
            "set_log_level": ([I32], None),
            "set_log_callback": ([PTR, PTR], None),
            "detect_encoded": ([PTR, PTR, c.c_size_t, PTR, c.POINTER(PTR), c.POINTER(I32)], I32),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.dll, "aila_" + name)
            function.argtypes, function.restype = arguments, result
            setattr(self, name, function)
        self.set_log_level(3)

    def config(self, count=16, v2=False):
        value = self.default_gen_config_v2() if v2 else self.default_gen_config()
        value.max_new_tokens, value.do_sample = count, 0
        if v2:
            value.use_fixed_seed, value.sampling_seed = 1, 921
        return value

    def message(self, engine):
        return self.last_error_message(engine).decode("utf-8", errors="replace")

    def text(self, method, engine, prompt, config):
        pointer = getattr(self, method)(engine, prompt.encode("utf-8"), c.byref(config))
        if not pointer:
            raise AssertionError(f"{method}: {self.last_error_code(engine)} {self.message(engine)}")
        try:
            return c.string_at(pointer).decode("utf-8")
        finally:
            self.free_string(pointer)


SPECIAL = {"<|im_start|>": 256, "<|im_end|>": 257, "<|endoftext|>": 258, "<think>": 259, "</think>": 260}
TEMPLATE = ("{%- for message in messages -%}"
            "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' -}}"
            "{%- endfor -%}{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' -}}{%- endif -%}")


def encode(text):
    result, index = [], 0
    while index < len(text):
        match = next((key for key in SPECIAL if text.startswith(key, index)), None)
        if match:
            result.append(SPECIAL[match])
            index += len(match)
        else:
            result.extend(text[index].encode("utf-8"))
            index += 1
    return result


def render(messages):
    return "".join(f"<|im_start|>{role}\n{text}<|im_end|>\n" for role, text in messages) + "<|im_start|>assistant\n"


def oracle(prompt, count):
    tokens = encode(prompt)
    length, previous, output = len(tokens), tokens[-1], []
    for _ in range(count):
        token = ord("A") + (previous + length + 1) % 26
        output.append(chr(token))
        previous, length = token, length + 1
    return "".join(output)


def request(text="ignored", template=None, **options):
    value = {"messages": [{"role": "user", "content": text}], **options}
    if template is not None:
        value["chat_template_kwargs"] = {"chat_template": template}
    return json.dumps(value, ensure_ascii=False)


def write_model(path):
    path.mkdir()
    extra, vocab = 256, {}
    for byte in range(256):
        visible = 33 <= byte <= 126 or 161 <= byte <= 172 or byte >= 174
        vocab[chr(byte if visible else extra)] = byte
        extra += not visible
    files = {
        "vocab.json": vocab,
        "tokenizer.json": {"added_tokens": [{"content": text, "id": token, "special": True} for text, token in SPECIAL.items()]},
        "tokenizer_config.json": {"chat_template": TEMPLATE, "eos_token": "<|im_end|>", "added_tokens_decoder": {}},
        "config.json": {"model_type": "qwen3", "vocab_size": 261, "eos_token_id": 257,
                        "quantization_config": {"quant_method": "bitsandbytes", "load_in_4bit": True,
                                                "bnb_4bit_quant_type": "nf4"}},
    }
    for name, value in files.items():
        (path / name).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    (path / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")


class ProcessEntry(c.Structure):
    _fields_ = [("size", U32), ("usage", U32), ("pid", U32), ("heap", c.c_size_t),
                ("module", U32), ("threads", U32), ("parent", U32), ("priority", I32),
                ("flags", U32), ("exe", c.c_wchar * 260)]


kernel = c.WinDLL("kernel32", use_last_error=True)
kernel.CreateToolhelp32Snapshot.argtypes, kernel.CreateToolhelp32Snapshot.restype = [U32, U32], PTR
kernel.Process32FirstW.argtypes = kernel.Process32NextW.argtypes = [PTR, c.POINTER(ProcessEntry)]
kernel.CloseHandle.argtypes = [PTR]
kernel.OpenProcess.argtypes, kernel.OpenProcess.restype = [U32, I32, U32], PTR
kernel.GetModuleHandleW.argtypes, kernel.GetModuleHandleW.restype = [c.c_wchar_p], PTR
kernel.K32EnumProcessModules.argtypes = [PTR, PTR, U32, c.POINTER(U32)]
kernel.K32GetModuleFileNameExW.argtypes = [PTR, PTR, c.c_wchar_p, U32]


def workers():
    snapshot = kernel.CreateToolhelp32Snapshot(2, 0)
    assert snapshot and snapshot != PTR(-1).value
    result, entry = [], ProcessEntry()
    entry.size = c.sizeof(entry)
    try:
        valid = kernel.Process32FirstW(snapshot, c.byref(entry))
        while valid:
            if entry.parent == os.getpid() and entry.exe.lower() == "ailaworker.exe":
                result.append(entry.pid)
            valid = kernel.Process32NextW(snapshot, c.byref(entry))
    finally:
        kernel.CloseHandle(snapshot)
    return result


def assert_cpu_worker(pid, runtime):
    handle = kernel.OpenProcess(0x410, 0, pid)
    assert handle
    try:
        modules, count = (PTR * 2048)(), U32()
        assert kernel.K32EnumProcessModules(handle, modules, c.sizeof(modules), c.byref(count))
        assert count.value <= c.sizeof(modules)
        paths = []
        for module in modules[:count.value // c.sizeof(PTR)]:
            path = c.create_unicode_buffer(32768)
            assert kernel.K32GetModuleFileNameExW(handle, module, path, len(path))
            paths.append(Path(path.value))
        assert (runtime / "AilaWorker.exe").resolve() == paths[0].resolve()
        for path in paths:
            assert not any(part in path.name.lower() for part in ("sycl", "dnnl", "ze_loader", "ur_adapter")), path.name
    finally:
        kernel.CloseHandle(handle)


@contextmanager
def environment(**changes):
    old = {name: os.environ.get(name) for name in changes}
    try:
        for name, value in changes.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class LegacyTests(unittest.TestCase):
    api = api_directory = runtime = model = None

    def setUp(self):
        self.engines = []
        self.initial_env, self.initial_cwd = dict(os.environ), os.getcwd()
        assert_no_gpu_modules()
        self.assertFalse(workers())

    def tearDown(self):
        for engine in self.engines:
            self.api.engine_destroy(engine)
        until(lambda: not workers(), "Legacy adapter left a Worker process", 8)
        self.assertEqual(dict(os.environ), self.initial_env)
        self.assertEqual(os.getcwd(), self.initial_cwd)
        assert_no_gpu_modules()

    def engine(self, capacity=2048, model=None):
        engine = self.api.engine_create()
        self.assertTrue(engine)
        self.engines.append(engine)
        self.assertEqual(self.api.engine_init(engine, str(model or self.model).encode("utf-8"), capacity), 0,
                         self.api.message(engine))
        return engine

    def destroy(self, engine):
        self.engines.remove(engine)
        self.api.engine_destroy(engine)

    def test_01_layout_history_and_reset(self):
        self.assertEqual((c.sizeof(Config), c.sizeof(ConfigV2), c.sizeof(ChatEvent)), (40, 96, 64))
        engine, config = self.engine(), self.api.config(12)
        history = [("system", "You are a helpful assistant."), ("user", "first-中文")]
        first = self.api.text("generate", engine, history[-1][1], config)
        self.assertEqual(first, oracle(render(history), 12))
        self.assertEqual(self.api.engine_context_length(engine), len(encode(render(history))) + 12)
        history.extend([("assistant", first), ("user", "second")])
        second = self.api.text("generate", engine, "second", config)
        self.assertEqual(second, oracle(render(history), 12))
        self.assertEqual(self.api.engine_context_length(engine), len(encode(render(history))) + 12)
        self.api.engine_reset_context(engine)
        self.assertEqual(self.api.engine_context_length(engine), 0)
        self.assertEqual(self.api.text("generate", engine, "first-中文", config), first)

    def test_02_json_and_structured_events(self):
        engine, config = self.engine(), self.api.config(16, v2=True)
        body = request(template="known-prefix")
        expected = oracle("known-prefix", 16)
        result = json.loads(self.api.text("generate_chat_json_ex", engine, body, config))
        self.assertEqual((result["content"], result["raw_text"], result["finish_reason"]), (expected, expected, "length"))
        self.assertEqual(result["metadata"]["template_name"], "request-override")
        events, callback_errors = [], []
        thread_id = threading.get_ident()

        @ChatCallback
        def callback(pointer, _):
            try:
                event = pointer.contents
                assert event.struct_size == c.sizeof(ChatEvent) and threading.get_ident() == thread_id
                events.append({name: (getattr(event, name).decode("utf-8") if getattr(event, name) else None)
                               for name, _ in ChatEvent._fields_[2:]} | {"type": event.type})
                return 0
            except BaseException as error:
                callback_errors.append(str(error))
                return 1

        self.assertEqual(self.api.generate_chat_json_stream_ex(engine, body.encode(), c.byref(config), callback, None), 0)
        self.assertFalse(callback_errors)
        self.assertEqual("".join(event["text"] or "" for event in events if event["type"] == 1), expected)
        self.assertEqual(sum(event["type"] == 4 for event in events), 1)
        self.assertEqual(events[-1]["finish_reason"], "length")

    def test_03_four_engines_share_one_worker_and_reuse_slot(self):
        with ThreadPoolExecutor(4) as pool:
            engines = list(pool.map(lambda _: self.engine(), range(4)))
        self.assertEqual(len(workers()), 1)
        assert_cpu_worker(workers()[0], self.runtime)
        config = self.api.config(24)
        with ThreadPoolExecutor(4) as pool:
            results = list(pool.map(lambda i: self.api.text("generate_messages", engines[i],
                request(template=f"peer-{i}"), config), range(4)))
        self.assertEqual(results, [oracle(f"peer-{i}", 24) for i in range(4)])
        pid = workers()[0]
        self.destroy(engines[0])
        replacement = self.engine()
        self.assertEqual(workers(), [pid])
        self.assertEqual(self.api.text("generate_messages", replacement, request(template="fresh"), config), oracle("fresh", 24))
        self.assertEqual(self.api.text("generate_messages", engines[1], request(template="still-live"), config),
                         oracle("still-live", 24))

    def test_04_raw_callback_abort_reentrancy_and_peer(self):
        engine, peer, config = self.engine(), self.engine(), self.api.config(96)
        pieces, failures = [], []
        thread_id = threading.get_ident()

        @TokenCallback
        def callback(piece, _):
            try:
                assert threading.get_ident() == thread_id
                pieces.append(piece.decode("utf-8"))
                if len(pieces) == 1:
                    rejected = self.api.generate(engine, b"reenter", c.byref(config))
                    assert not rejected and self.api.last_error_code(engine) == 1
                    assert self.api.text("generate_messages", peer, request(template="peer"), self.api.config(4)) == oracle("peer", 4)
                return len(pieces) == 3
            except BaseException as error:
                failures.append(str(error))
                return 1

        body = request(template="cancel-prefix")
        self.assertEqual(self.api.generate_messages_stream(engine, body.encode(), c.byref(config), callback, None), 1)
        self.assertFalse(failures)
        self.assertEqual("".join(pieces), oracle("cancel-prefix", 3))
        self.assertEqual(self.api.engine_context_length(engine), len("cancel-prefix") + 3)
        self.assertEqual(self.api.text("generate_messages", engine, request(template="continued"), self.api.config(4)),
                         oracle("continued", 4))

    def test_05_destroy_from_raw_and_structured_callback(self):
        peer, config = self.engine(), self.api.config(32)
        for structured in (False, True):
            engine, failures, called = self.engine(), [], []

            def destroy():
                self.destroy(engine)
                called.append(1)
                return 1

            @TokenCallback
            def raw_callback(piece, _):
                try:
                    return destroy()
                except BaseException as error:
                    failures.append(str(error))
                    return 1

            @ChatCallback
            def structured_callback(event, _):
                try:
                    return destroy()
                except BaseException as error:
                    failures.append(str(error))
                    return 1

            if structured:
                v2 = self.api.config(32, v2=True)
                result = self.api.generate_chat_json_stream_ex(engine, request(template="destroy").encode(),
                    c.byref(v2), structured_callback, None)
            else:
                result = self.api.generate_stream(engine, b"destroy", c.byref(config), raw_callback, None)
            self.assertEqual(result, 1)
            self.assertEqual(called, [1])
            self.assertFalse(failures)
            self.assertEqual(len(workers()), 1)
            self.assertEqual(self.api.text("generate_messages", peer, request(template="survivor"), config), oracle("survivor", 32))

    def test_06_local_errors_do_not_break_text_session(self):
        engine, config = self.engine(), self.api.config(8)
        self.api.text("generate_messages", engine, request(template="base"), config)
        before = self.api.engine_context_length(engine)
        pointer = self.api.generate_chat_json(engine, b"{", c.byref(config))
        self.assertFalse(pointer)
        self.assertEqual(self.api.last_error_code(engine), 3)
        media = json.dumps({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x.png"}}]}]})
        self.assertFalse(self.api.generate_chat_json(engine, media.encode(), c.byref(config)))
        self.assertEqual(self.api.last_error_code(engine), 4)
        output, count, data = PTR(), I32(), c.create_string_buffer(b"invalid-image")
        self.assertEqual(self.api.detect_encoded(engine, data, len(data), None, c.byref(output), c.byref(count)), 7)
        self.assertFalse(output.value)
        self.assertEqual(count.value, 0)
        self.assertEqual(self.api.engine_context_length(engine), before)
        self.assertEqual(self.api.text("generate_messages", engine, request(template="recovered"), config), oracle("recovered", 8))

    def test_07_zero_generation_and_long_prefix(self):
        engine = self.engine(7000)
        prefix = "p" * 4201
        result = json.loads(self.api.text("generate_chat_json", engine, request(template=prefix), self.api.config(0)))
        self.assertEqual((result["content"], result["finish_reason"]), ("", "length"))
        self.assertEqual(self.api.engine_context_length(engine), len(prefix))
        expected = oracle(prefix, 16)
        self.assertEqual(self.api.text("generate_messages", engine, request(template=prefix), self.api.config(16)), expected)
        extended = self.api.config(1200)
        self.assertEqual(self.api.text("generate_messages", engine, request(template="extended"), extended),
                         oracle("extended", 1200))
        self.assertEqual(self.api.engine_context_length(engine), 1208)

    def test_08_template_changes_and_host_isolation(self):
        engine, config = self.engine(), self.api.config(4, v2=True)
        with environment(AILA_CHAT_TEMPLATE="ENV-A", AILA_CHAT_TEMPLATE_PATH=None):
            result = json.loads(self.api.text("generate_chat_json_ex", engine, request(), config))
            self.assertEqual(result["content"], oracle("ENV-A", 4))
            self.assertEqual(result["metadata"]["template_name"], "AILA_CHAT_TEMPLATE")
            self.assertEqual(json.loads(self.api.text("generate_chat_json_ex", engine, request(template="OVERRIDE"), config))["content"],
                             oracle("OVERRIDE", 4))
        with environment(AILA_CHAT_TEMPLATE=None, AILA_CHAT_TEMPLATE_PATH=None):
            result = json.loads(self.api.text("generate_chat_json_ex", engine, request("after-env"), config))
            self.assertEqual(result["content"], oracle(render([("user", "after-env")]), 4))
        original_cwd = os.getcwd()
        try:
            os.chdir(self.model)
            self.assertEqual(json.loads(self.api.text("generate_chat_json_ex", engine, request(template="cwd"), config))["content"],
                             oracle("cwd", 4))
            self.assertEqual(os.getcwd(), str(self.model))
        finally:
            os.chdir(original_cwd)

    def test_09_runtime_unload_after_shutdown_and_init_failure(self):
        path = str(self.api_directory / "AilaInference.dll")
        self.assertFalse(kernel.GetModuleHandleW(path))
        bad_model = self.model.parent / "unknown-cpu-model"
        bad_model.mkdir(exist_ok=True)
        for file in self.model.iterdir():
            if file.is_file():
                shutil.copy2(file, bad_model / file.name)

        def cycle(failed):
            if failed:
                engine = self.api.engine_create()
                self.engines.append(engine)
                self.assertNotEqual(self.api.engine_init(engine, str(bad_model).encode(), 2048), 0)
            else:
                engine = self.engine()
                self.assertTrue(kernel.GetModuleHandleW(path))
                self.api.text("generate_messages", engine, request(template="lifecycle"), self.api.config(2))
            self.destroy(engine)
            until(lambda: not workers(), "Worker did not exit before DLL release", 8)
            self.assertFalse(kernel.GetModuleHandleW(path), "Inference DLL retained an owner after cleanup")
            return process_handle_count()

        # Both first use and the first exception initialize process-level
        # facilities. Record the cold values, then bound repeated mixed cycles.
        cold = [process_handle_count(), cycle(False), cycle(True)]
        baseline, counts = process_handle_count(), []
        for _ in range(5):
            counts.extend((cycle(False), cycle(True)))
            until(lambda: process_handle_count() <= baseline + 2, "Legacy bridge grew handles after warmup")
        print(json.dumps({"legacy_lifecycle_cold_handles": cold, "baseline": baseline,
                          "legacy_lifecycle_handle_counts": counts}), flush=True)
        self.assertLessEqual(max(counts), baseline + 2)

    def test_11_worker_logs_once_for_shared_model(self):
        collected, lock = [], threading.Lock()

        @LogCallback
        def callback(level, message, _):
            text = message.decode("utf-8")
            if text.startswith("[FakeLog]"):
                with lock:
                    collected.append((level, text, threading.get_ident()))

        def count(message):
            with lock:
                return sum(item[1] == message for item in collected)

        self.api.set_log_callback(c.cast(callback, PTR), None)
        try:
            with environment(AILA_FAKE_WORKER_LOGS="1"):
                self.api.set_log_level(0)
                engines = [self.engine() for _ in range(4)]
                pid = workers()[0]
                self.assertEqual(len(workers()), 1)
                until(lambda: count("[FakeLog] model-load") == 1, "Shared model load log was not forwarded")
                self.api.set_log_level(2)
                self.api.text("generate_messages", engines[1], request(template="log-one"), self.api.config(4))
                until(lambda: count("[FakeLog] error") >= 1, "Worker error log missing")
                self.destroy(engines[0])
                self.assertEqual(workers(), [pid])
                self.api.text("generate_messages", engines[2], request(template="log-two"), self.api.config(4))
                until(lambda: count("[FakeLog] error") >= 2, "Logs did not transfer to a live Engine source")
                self.assertEqual(count("[FakeLog] debug"), 0)
                self.assertEqual(count("[FakeLog] 信息 α"), 0)
                self.api.set_log_level(0)
                self.api.text("generate_messages", engines[3], request(template="log-three"), self.api.config(4))
                until(lambda: count("[FakeLog] error") >= 3, "Re-enabled logs did not arrive")
                with lock:
                    self.assertEqual(sum(item[1] == "[FakeLog] model-load" for item in collected), 1)
                    self.assertEqual(sum(item[1] == "[FakeLog] warning" for item in collected), 3)
                    self.assertEqual(sum(item[1] == "[FakeLog] error" for item in collected), 3)
                    self.assertEqual(sum(item[1] == "[FakeLog] debug" for item in collected), 1)
                    self.assertEqual(sum(item[1] == "[FakeLog] 信息 α" for item in collected), 1)
                    self.assertTrue(all(item[2] != threading.get_ident() for item in collected))
        finally:
            self.api.set_log_callback(None, None)
            self.api.set_log_level(3)

    def test_12_slow_log_callback_does_not_block_inference(self):
        entered, release = threading.Event(), threading.Event()

        @LogCallback
        def callback(level, message, _):
            if message == b"[FakeLog] error" and not entered.is_set():
                entered.set()
                release.wait(8)

        self.api.set_log_callback(c.cast(callback, PTR), None)
        try:
            with environment(AILA_FAKE_WORKER_LOGS="1"):
                engine, peer = self.engine(), self.engine()
                config = self.api.config(8)
                with ThreadPoolExecutor(2) as pool:
                    first = pool.submit(self.api.text, "generate_messages", engine, request(template="slow-log"), config)
                    self.assertTrue(entered.wait(3))
                    self.assertEqual(first.result(timeout=2), oracle("slow-log", 8))
                    destroy = pool.submit(self.destroy, engine)
                    second = pool.submit(self.api.text, "generate_messages", peer, request(template="live-peer"), config)
                    try:
                        self.assertEqual(second.result(timeout=2), oracle("live-peer", 8))
                        self.assertFalse(destroy.done(), "Engine destruction ignored an active log callback lease")
                    finally:
                        release.set()
                    destroy.result(timeout=3)
                self.assertEqual(len(workers()), 1)
        finally:
            release.set()
            self.api.set_log_callback(None, None)
            self.api.set_log_level(3)

    def test_13_destroy_last_engine_from_worker_log_callback(self):
        finished, failures, target = threading.Event(), [], [None]

        @LogCallback
        def callback(level, message, _):
            if message != b"[FakeLog] error" or finished.is_set():
                return
            try:
                self.destroy(target[0])
            except BaseException as error:
                failures.append(error)
            finally:
                finished.set()

        self.api.set_log_callback(c.cast(callback, PTR), None)
        try:
            with environment(AILA_FAKE_WORKER_LOGS="1"):
                target[0] = self.engine()
                self.assertEqual(self.api.text("generate_messages", target[0], request(template="callback-close"),
                                               self.api.config(4)), oracle("callback-close", 4))
                self.assertTrue(finished.wait(5), "Callback-reentrant Worker close deadlocked")
                self.assertFalse(failures)
                until(lambda: not workers(), "Callback destroy left a Worker")
        finally:
            self.api.set_log_callback(None, None)
            self.api.set_log_level(3)

    def test_14_generation_reservation_boundary_keeps_context(self):
        # A one-token prompt still needs credit when max_new_tokens reaches
        # the negotiated 32768 ceiling. These requests must fail before the
        # adapter truncates an already useful context. No long generation runs.
        engine = self.engine(70000)
        self.assertEqual(self.api.text("generate_messages", engine, request(template="baseline"),
                                      self.api.config(6)), oracle("baseline", 6))
        before = self.api.engine_context_length(engine)
        for maximum in (32769, (1 << 31) - 1, 32768):
            config = self.api.config(maximum)
            self.assertFalse(self.api.generate_messages(engine, request(template="p").encode(), c.byref(config)))
            self.assertEqual(self.api.last_error_code(engine), 6)
            self.assertEqual(self.api.engine_context_length(engine), before,
                             f"Impossible reservation {maximum} changed the existing context")
        pieces, errors = [], []

        @TokenCallback
        def cancel_after_one(piece, _):
            try:
                pieces.append(piece.decode("utf-8"))
            except BaseException as error:
                errors.append(str(error))
            return 1

        limit = self.api.config(32767)
        self.assertEqual(self.api.generate_messages_stream(engine, request(template="p").encode(),
            c.byref(limit), cancel_after_one, None), 1)
        self.assertFalse(errors)
        self.assertEqual(pieces, [oracle("p", 1)])
        self.assertEqual(self.api.engine_context_length(engine), 2)
        self.assertEqual(self.api.text("generate_messages", engine, request(template="recovered"),
                                      self.api.config(6)), oracle("recovered", 6))

    def test_15_session_count_cache_admission_and_reuse(self):
        for limit, settings in ((4, {}), (2, {"AILA_LEGACY_TEXT_MAX_SESSIONS": "2"}),
                                (2, {"AILA_LEGACY_TEXT_MAX_SESSION_CACHE_BYTES": str(2048 * 16 * 2)})):
            with self.subTest(settings=settings), environment(**settings):
                live = [self.engine() for _ in range(limit)]
                pid = workers()[0]
                self.assertEqual(len(workers()), 1)
                rejected = self.api.engine_create()
                self.engines.append(rejected)
                self.assertNotEqual(self.api.engine_init(rejected, str(self.model).encode(), 2048), 0)
                self.assertEqual(self.api.last_error_code(rejected), 6)
                self.assertIn("Session" if not settings or "AILA_LEGACY_TEXT_MAX_SESSIONS" in settings else "reservation",
                              self.api.message(rejected))
                self.assertEqual(workers(), [pid], "Quota refusal started another Worker")
                self.destroy(rejected)
                peer = live[-1]
                self.assertEqual(self.api.text("generate_messages", peer, request(template="survivor"),
                                              self.api.config(4)), oracle("survivor", 4))
                self.destroy(live.pop(0))
                replacement = self.engine()
                live.append(replacement)
                self.assertEqual(workers(), [pid])
                self.assertEqual(self.api.text("generate_messages", replacement, request(template="reused"),
                                              self.api.config(4)), oracle("reused", 4))
                for engine in live:
                    self.destroy(engine)
                until(lambda: not workers(), "Admission scenario retained a Worker", 8)

    def test_16_model_allocation_limits_reach_worker(self):
        model = self.model.parent / "fake-legacy-text-limits-model"
        shutil.copytree(self.model, model)
        with environment(AILA_LEGACY_TEXT_MAX_SESSIONS="2", AILA_LEGACY_TEXT_MAX_SESSION_CACHE_BYTES="65536",
                         AILA_LEGACY_TEXT_MAX_DEVICE_BYTES="111111", AILA_LEGACY_TEXT_MAX_PINNED_HOST_BYTES="222222",
                         AILA_LEGACY_TEXT_MAX_SESSION_HOST_BYTES="333333"):
            engine = self.engine(model=model)
            self.assertEqual(self.api.text("generate_messages", engine, request(template="forwarded-limits"),
                                          self.api.config(4)), oracle("forwarded-limits", 4))

    def test_17_invalid_admission_settings_fail_before_worker_start(self):
        invalid = [("AILA_LEGACY_TEXT_MAX_SESSIONS", value) for value in ("0", "5", "4294967296")]
        for name in ("AILA_LEGACY_TEXT_MAX_SESSIONS", "AILA_LEGACY_TEXT_MAX_SESSION_CACHE_BYTES",
                     "AILA_LEGACY_TEXT_MAX_DEVICE_BYTES", "AILA_LEGACY_TEXT_MAX_PINNED_HOST_BYTES",
                     "AILA_LEGACY_TEXT_MAX_SESSION_HOST_BYTES"):
            invalid.extend((name, value) for value in ("-1", "+1", " 16", "1KiB", "18446744073709551616"))
        for name, value in invalid:
            with self.subTest(name=name, value=value), environment(**{name: value}):
                engine = self.api.engine_create()
                self.engines.append(engine)
                self.assertNotEqual(self.api.engine_init(engine, str(self.model).encode(), 2048), 0)
                self.assertEqual(self.api.last_error_code(engine), 1, self.api.message(engine))
                self.assertIn(name, self.api.message(engine))
                self.assertFalse(workers())
                self.destroy(engine)
        for value in ("0", "18446744073709551615"):
            with self.subTest(valid_bytes=value), environment(AILA_LEGACY_TEXT_MAX_DEVICE_BYTES=value,
                AILA_LEGACY_TEXT_MAX_PINNED_HOST_BYTES=value, AILA_LEGACY_TEXT_MAX_SESSION_HOST_BYTES=value,
                AILA_LEGACY_TEXT_MAX_SESSION_CACHE_BYTES=value, AILA_LEGACY_TEXT_MAX_SESSIONS="1"):
                engine = self.engine()
                self.assertEqual(self.api.text("generate_messages", engine, request(template="byte-limits"),
                                              self.api.config(2)), oracle("byte-limits", 2))
                self.destroy(engine)
                until(lambda: not workers(), "Limit parsing case retained a Worker", 8)

    def test_10_environment_separates_model_owners(self):
        first = self.engine()
        first_pid = workers()[0]
        with environment(AILA_LEGACY_TEST_DOMAIN="separate"):
            second = self.engine()
            self.assertEqual(len(workers()), 2)
            self.assertIn(first_pid, workers())
            self.assertEqual(self.api.text("generate_messages", second, request(template="domain-two"), self.api.config(4)),
                             oracle("domain-two", 4))
            self.destroy(second)
        until(lambda: workers() == [first_pid], "Separate environment Worker did not stop")
        self.assertEqual(self.api.text("generate_messages", first, request(template="domain-one"), self.api.config(4)),
                         oracle("domain-one", 4))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dll", required=True)
    parser.add_argument("--worker")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--runtime")
    args, remaining = parser.parse_known_args()
    if args.child:
        LegacyTests.api_directory = Path(args.dll).resolve().parent
        LegacyTests.runtime = Path(args.runtime).resolve()
        LegacyTests.model, LegacyTests.api = Path(args.model).resolve(), Api(args.dll)
        unittest.main(argv=[sys.argv[0], *remaining], verbosity=2)
        return
    dll, worker = Path(args.dll).resolve(), Path(args.worker).resolve()
    if worker.name.lower() != "ailafakeinferenceworker.exe":
        parser.error("This CPU suite requires AilaFakeInferenceWorker.exe")
    with tempfile.TemporaryDirectory(prefix="aila-legacy-api-") as temporary:
        root = Path(temporary).resolve()
        api_directory, model = root / "host", root / "fake-legacy-text-model"
        runtime = api_directory / "aila_runtime"
        runtime.mkdir(parents=True)
        write_model(model)
        host_patterns = ("msvcp140*.dll", "vcruntime140*.dll", "concrt140.dll", "libmmd.dll", "svml_dispmd.dll")
        for name in ("AilaShared.dll", "AilaInference.dll"):
            shutil.copy2(dll.parent / name, api_directory / name)
        for pattern in host_patterns:
            for file in dll.parent.glob(pattern):
                shutil.copy2(file, runtime / file.name)
                shutil.copy2(file, api_directory / file.name)
        renamed = runtime / "AilaWorker.exe"
        shutil.copy2(worker, renamed)
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest(worker) == digest(renamed)
        print(json.dumps({"cpu_worker_sha256": digest(worker), "proxy_sha256": digest(api_directory / "AilaShared.dll"),
                          "inference_sha256": digest(api_directory / "AilaInference.dll"),
                          "separate_host_and_worker_directories": True}), flush=True)
        env = dict(os.environ)
        for name in ("AILA_CHAT_TEMPLATE", "AILA_CHAT_TEMPLATE_PATH", "AILA_VLM_KV_QUANT", "AILA_KV_QUANT",
                     "AILA_LEGACY_TEXT_MAX_SESSIONS", "AILA_LEGACY_TEXT_MAX_SESSION_CACHE_BYTES",
                     "AILA_LEGACY_TEXT_MAX_DEVICE_BYTES", "AILA_LEGACY_TEXT_MAX_PINNED_HOST_BYTES",
                     "AILA_LEGACY_TEXT_MAX_SESSION_HOST_BYTES"):
            env.pop(name, None)
        env.update(AILA_LEGACY_TEXT_V2="1", AILA_RUNTIME_DLL_DIR="aila_runtime")
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child",
            "--dll", str(api_directory / "AilaShared.dll"), "--model", str(model),
            "--runtime", str(runtime), *remaining], env=env, timeout=90)
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
