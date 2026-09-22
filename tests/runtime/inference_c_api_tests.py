"""ctypes consumer of the public DLL, with an independent CPU token oracle.

The fake Worker uses the real transport, controller and Scheduler. This suite
does not load any model or GPU runtime. GPU ABI comparisons have a separate
guarded runner; never substitute a GPU Worker in this CPU suite.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes as c
import math
import os
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

U32, I32, U64 = c.c_uint32, c.c_int32, c.c_uint64
OK, TIMEOUT, END, ARG, HANDLE, BUSY, LIMIT, OVERFLOW, UNSUPPORTED, CLOSED = range(10)
PROTOCOL, FAULT, FAILED = 10, 11, 12


class Versioned(c.Structure):
    _pack_ = 8
    _fields_ = [("struct_size", U32), ("abi_version", U32)]


class Error(Versioned):
    _fields_ = [("code", I32), ("remote_error", U32), ("message", c.c_char * 1025)]


class ClientOptions(Versioned):
    _fields_ = [(name, U32) for name in (
        "handshake_timeout_ms", "command_timeout_ms", "shutdown_timeout_ms", "max_active_requests",
        "max_outstanding_requests", "max_prompt_tokens", "max_generated_tokens", "max_sessions",
        "max_retained_request_handles", "max_buffered_tokens_per_request", "token_window", "max_prefill_chunk_size")]
    _fields_ += [("max_outstanding_tokens", U64), ("max_session_cache_bytes", U64), ("reserved", U32 * 8)]


class ModelOptions(Versioned):
    _fields_ = [("max_sequence_length", U32), ("max_sessions", U32), ("max_session_cache_bytes", U64),
                ("allocation_limits_enabled", U32), ("reserved0", U32), ("max_device_bytes", U64),
                ("max_pinned_host_bytes", U64), ("max_session_host_bytes", U64), ("reserved", U32 * 8)]


class RequestOptions(Versioned):
    _fields_ = [("max_new_tokens", U32), ("prefill_chunk_size", U32), ("do_sample", U32),
                ("use_fixed_seed", U32), ("sampling_seed", U64), ("temperature", c.c_double),
                ("top_p", c.c_double), ("repetition_penalty", c.c_double), ("presence_penalty", c.c_double),
                ("frequency_penalty", c.c_double), ("top_k", U32), ("stop_token_count", U32),
                ("stop_token_ids", c.POINTER(I32)), ("thinking_budget_tokens", I32), ("prompt_opens_think", U32),
                ("thinking_open_token_id", I32), ("thinking_close_token_id", I32),
                ("thinking_close_token_ids", c.POINTER(I32)), ("thinking_close_token_count", U32), ("reserved", U32 * 8)]


class RequestPolicy(Versioned):
    _fields_ = [("thinking_policy", U32), ("sampling_policy", U32),
                ("prefill_logits_policy", U32), ("reserved", U32 * 3)]


class Progress(c.Structure):
    _pack_ = 8
    _fields_ = [("status", U32), ("end_reason", U32), ("remote_error", U32), ("reserved0", U32),
                ("prompt_tokens_processed", U64), ("generated_tokens", U64), ("event_count", U64),
                ("message", c.c_char * 1025)]


class Event(Versioned):
    _fields_ = [("kind", U32), ("local_failure", U32), ("sequence", U64), ("token_id", I32),
                ("delivery_abandoned", U32), ("delivered_tokens", U64), ("discarded_published_tokens", U64),
                ("progress", Progress)]


class ClientInfo(Versioned):
    _fields_ = [("domain_status", U32), ("reserved0", U32)]
    _fields_ += [(name, U64) for name in ("outstanding_requests", "retained_request_handles", "pending_commands",
                 "queued_commands", "reserved_tokens", "ingress_bytes", "transport_bytes", "peak_ingress_bytes",
                 "peak_transport_bytes", "dropped_logs")]
    _fields_ += [("epoch", c.c_char * 33), ("message", c.c_char * 1025)]


class RuntimeMemoryInfo(Versioned):
    _fields_ = [("state", U32), ("reserved0", U32)]
    _fields_ += [(name, U64) for name in ("device_bytes", "peak_device_bytes", "pinned_host_bytes",
                 "peak_pinned_host_bytes", "device_allocations", "pinned_host_allocations",
                 "device_limit_bytes", "pinned_host_limit_bytes")]
    _fields_ += [("epoch", c.c_char * 33)]


class LogEvent(Versioned):
    _fields_ = [("event_ordinal", U64), ("level", U32), ("message_bytes", U32), ("dropped_logs", U64),
                ("epoch", c.c_char * 33), ("message", c.c_char * 1025)]


class ModelInfo(Versioned):
    _fields_ = [("model_id", U64), ("vocab_size", U32), ("max_sequence_length", U32),
                ("backend", U32), ("has_memory_info", U32)]
    _fields_ += [(name, U64) for name in ("session_reservation_bytes", "resident_device_bytes",
                 "session_host_reservation_bytes", "max_device_bytes", "max_pinned_host_bytes", "max_session_host_bytes")]
    _fields_ += [("epoch", c.c_char * 33)]


class SessionInfo(Versioned):
    _fields_ = [("session_id", U64), ("model_id", U64), ("capacity", U32), ("busy", U32),
                ("cache_bytes", U64), ("has_context_length", U32), ("context_length", U32), ("epoch", c.c_char * 33)]


class RequestInfo(Versioned):
    _fields_ = [("inference_id", U64), ("buffered_tokens", U64), ("last_sent_sequence", U64),
                ("last_acknowledged_sequence", U64), ("completion_ready", U32), ("reserved0", U32),
                ("epoch", c.c_char * 33), ("progress", Progress)]


def output(kind):
    value = kind()
    value.struct_size, value.abi_version = c.sizeof(kind), 1
    return value


class Api:
    def __init__(self, path):
        self.dll = c.CDLL(str(Path(path).resolve()))
        signatures = {
            "client_start": [c.c_char_p, c.c_char_p, c.POINTER(ClientOptions), c.POINTER(U64)],
            "client_start_with_batching": [c.c_char_p, c.c_char_p, c.POINTER(ClientOptions), U32, c.POINTER(U64)],
            "client_snapshot": [U64, c.POINTER(ClientInfo)], "client_shutdown": [U64, U32],
            "client_query_memory": [U64, U32, c.POINTER(RuntimeMemoryInfo)],
            "client_read_log": [U64, U32, c.POINTER(LogEvent)], "client_set_log_level": [U64, U32, U32],
            "model_load": [U64, U32, c.c_char_p, c.POINTER(ModelOptions), U32, c.POINTER(U64)],
            "model_info": [U64, c.POINTER(ModelInfo)], "session_create": [U64, U32, c.POINTER(U64)],
            "session_query": [U64, U32, c.POINTER(SessionInfo)],
            "session_reset": [U64, U32, c.POINTER(SessionInfo)],
            "session_truncate": [U64, U32, U32, c.POINTER(SessionInfo)],
            "request_submit": [U64, c.POINTER(I32), U32, c.POINTER(RequestOptions), U32, c.POINTER(U64)],
            "request_submit_with_policy": [U64, c.POINTER(I32), U32, c.POINTER(RequestOptions),
                                           c.POINTER(RequestPolicy), U32, c.POINTER(U64)],
            "request_prefill": [U64, c.POINTER(I32), U32, U32, U32, c.POINTER(U64)],
            "request_read": [U64, U32, c.POINTER(Event)], "request_cancel": [U64, U32, c.POINTER(U32)],
            "request_query": [U64, U32, c.POINTER(RequestInfo)],
        }
        signatures.update({kind + "_release": [U64] for kind in ("client", "model", "session", "request")})
        for name, arguments in signatures.items():
            function = getattr(self.dll, "aila_inference_" + name)
            function.argtypes, function.restype = arguments + [c.POINTER(Error)], I32
        for name, kind in (("client", ClientOptions), ("model", ModelOptions), ("request", RequestOptions)):
            function = getattr(self.dll, "aila_inference_default_" + name + "_options")
            function.argtypes, function.restype = [c.POINTER(kind), U32], I32

    def call(self, name, *args, expect=OK):
        error = output(Error)
        code = getattr(self.dll, "aila_inference_" + name)(*args, c.byref(error))
        if code != expect or error.code != code:
            raise AssertionError(f"{name}: result={code}, error={error.code}, expected={expect}, "
                                 f"remote={error.remote_error}, message={error.message.decode('utf-8', errors='replace')}")
        return error

    def defaults(self, name, kind):
        value = kind()
        assert getattr(self.dll, "aila_inference_default_" + name + "_options")(c.byref(value), c.sizeof(value)) == OK
        return value

    def request_options(self, count):
        value = self.defaults("request", RequestOptions)
        value.max_new_tokens, value.prefill_chunk_size = count, 16
        value.do_sample, value.use_fixed_seed, value.sampling_seed = 0, 1, 921
        return value

    def read(self, handle, timeout=50):
        event, error = output(Event), output(Error)
        code = self.dll.aila_inference_request_read(handle, timeout, c.byref(event), c.byref(error))
        assert code == error.code
        return code, event

    def read_log(self, handle, timeout=50):
        event, error = output(LogEvent), output(Error)
        code = self.dll.aila_inference_client_read_log(handle, timeout, c.byref(event), c.byref(error))
        assert code == error.code
        return code, event

    def drain(self, handle, timeout=5):
        tokens, deadline = [], time.monotonic() + timeout
        while time.monotonic() < deadline:
            code, event = self.read(handle)
            if code == TIMEOUT:
                assert event.kind == 0 and event.token_id == -1
                continue
            assert code == OK, f"read failed: {code}"
            if event.kind == 1:
                tokens.append(event.token_id)
                assert event.sequence == len(tokens) and event.progress.generated_tokens == len(tokens)
                continue
            assert event.kind == 2
            assert event.sequence == (0 if event.local_failure else len(tokens) + 1)
            assert event.delivered_tokens == len(tokens)
            assert self.read(handle, 0)[0] == END, "terminal must be delivered exactly once"
            return tokens, event
        raise AssertionError("Request did not complete before the CPU test deadline")


def until(predicate, message, timeout=4):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.002)
    raise AssertionError(message)


class Run:
    def __init__(self, api, worker, backend=1, retained=8, allocation=False, max_generated=256, fixture=None,
                 batch_size=None):
        self.api, self.handles = api, []
        self.client = self.model = 0
        options = api.defaults("client", ClientOptions)
        options.handshake_timeout_ms, options.command_timeout_ms, options.shutdown_timeout_ms = 3000, 3000, 5000
        options.max_outstanding_requests, options.max_retained_request_handles = min(retained, 8), retained
        options.max_prompt_tokens, options.max_prefill_chunk_size, options.max_generated_tokens = 64, 32, max_generated
        options.max_outstanding_tokens, options.token_window = 2048, 8
        try:
            handle = U64()
            worker = Path(worker).resolve()
            if batch_size is None:
                api.call("client_start", str(worker.parent).encode("utf-8"), str(worker).encode("utf-8"),
                         c.byref(options), c.byref(handle))
            else:
                api.call("client_start_with_batching", str(worker.parent).encode("utf-8"), str(worker).encode("utf-8"),
                         c.byref(options), batch_size, c.byref(handle))
            self.client = self.own("client", handle.value)
            model = api.defaults("model", ModelOptions)
            model.max_sequence_length, model.max_sessions = 2048, 4
            if allocation:
                model.allocation_limits_enabled = 1
                model.max_device_bytes, model.max_pinned_host_bytes, model.max_session_host_bytes = 111111, 222222, 333333
            directory = "E:/ABI-中文路径/" + (fixture or ("fake-pipe-model" if backend == 1 else "fake-pipe-hybrid-model"))
            api.call("model_load", self.client, backend, directory.encode("utf-8"), c.byref(model), 0, c.byref(handle))
            self.model = self.own("model", handle.value)
        except BaseException:
            self.close()
            raise

    def own(self, kind, handle):
        self.handles.append((kind, handle))
        return handle

    def release(self, kind, handle):
        self.api.call(kind + "_release", handle)
        self.handles.remove((kind, handle))
        if kind == "client":
            self.client = 0
        if kind == "model":
            self.model = 0

    def session(self):
        handle = U64()
        self.api.call("session_create", self.model, 0, c.byref(handle))
        return self.own("session", handle.value)

    def submit(self, session, prompt, count=16, options=None):
        options = options if options is not None else self.api.request_options(count)
        ids, handle = (I32 * len(prompt))(*prompt), U64()
        self.api.call("request_submit", session, ids, len(prompt), c.byref(options), 0, c.byref(handle))
        return self.own("request", handle.value)

    def prefill(self, session, prompt, chunk=2):
        ids, handle = (I32 * len(prompt))(*prompt), U64()
        self.api.call("request_prefill", session, ids, len(prompt), chunk, 0, c.byref(handle))
        return self.own("request", handle.value)

    def snapshot(self):
        result = output(ClientInfo)
        self.api.call("client_snapshot", self.client, c.byref(result))
        return result

    def session_info(self, session):
        result = output(SessionInfo)
        self.api.call("session_query", session, 0, c.byref(result))
        return result

    def quiet(self):
        def empty():
            info = self.snapshot()
            return not (info.outstanding_requests or info.retained_request_handles or
                        info.reserved_tokens or info.transport_bytes)
        until(empty, "Released requests retained host reservations")

    def close(self):
        if self.client:
            error = output(Error)
            self.api.dll.aila_inference_client_shutdown(self.client, 5000, c.byref(error))
        for kind, handle in reversed(self.handles):
            self.api.call(kind + "_release", handle)
        self.handles.clear()
        self.client = self.model = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def oracle(prompt, count, initial_length=0, budget=-1):
    length, previous, result = initial_length + len(prompt), prompt[-1], []
    for index in range(count):
        if budget >= 0 and index == budget and count - index < 2:
            break
        token = 1002 + index - budget if budget >= 0 and budget <= index < budget + 2 else (previous + length + 1) % 1000
        result.append(token)
        length, previous = length + 1, token
    return result


def process_handle_count():
    count = U32()
    kernel = c.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = c.c_void_p
    kernel.GetProcessHandleCount.argtypes = [c.c_void_p, c.POINTER(U32)]
    assert kernel.GetProcessHandleCount(kernel.GetCurrentProcess(), c.byref(count))
    return count.value


def assert_no_gpu_modules():
    kernel = c.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = c.c_void_p
    kernel.K32EnumProcessModules.argtypes = [c.c_void_p, c.c_void_p, U32, c.POINTER(U32)]
    kernel.GetModuleFileNameW.argtypes = [c.c_void_p, c.c_wchar_p, U32]
    modules, count = (c.c_void_p * 2048)(), U32()
    assert kernel.K32EnumProcessModules(kernel.GetCurrentProcess(), modules, c.sizeof(modules), c.byref(count))
    assert count.value <= c.sizeof(modules)
    for module in modules[:count.value // c.sizeof(c.c_void_p)]:
        path = c.create_unicode_buffer(32768)
        assert kernel.GetModuleFileNameW(module, path, len(path))
        name = Path(path.value).name.lower()
        assert not any(part in name for part in ("sycl", "dnnl", "onednn", "ze_loader", "ur_adapter")), name


class CApiTests(unittest.TestCase):
    api = None
    worker = None

    def setUp(self):
        self.environment = dict(os.environ)
        self.cwd = os.getcwd()
        assert_no_gpu_modules()

    def tearDown(self):
        self.assertEqual(os.getcwd(), self.cwd)
        self.assertEqual(dict(os.environ), self.environment)
        assert_no_gpu_modules()

    def complete(self, captured, expected):
        tokens, terminal = captured
        self.assertEqual(tokens, expected)
        self.assertEqual((terminal.progress.status, terminal.progress.end_reason, terminal.progress.remote_error), (4, 1, 0))
        self.assertEqual(terminal.progress.generated_tokens, len(expected))
        self.assertEqual(terminal.local_failure, 0)

    def test_01_layout_and_argument_validation(self):
        sizes = {Error: 1044, ClientOptions: 104, ModelOptions: 88, RequestOptions: 152, Progress: 1072,
                 Event: 1120, ClientInfo: 1160, LogEvent: 1096, ModelInfo: 120, SessionInfo: 88, RequestInfo: 1160}
        for kind, expected in sizes.items():
            self.assertEqual(c.sizeof(kind), expected, kind.__name__)
        options = self.api.defaults("client", ClientOptions)
        worker = Path(self.worker).resolve()
        handle = U64(123)
        self.api.call("client_start", b"\xffinvalid", None, c.byref(options), c.byref(handle), expect=ARG)
        self.assertEqual(handle.value, 0)
        options.reserved[3] = 1
        self.api.call("client_start", str(worker.parent).encode(), str(worker).encode(), c.byref(options), c.byref(handle), expect=ARG)
        options.reserved[3], options.abi_version = 0, 99
        self.api.call("client_start", str(worker.parent).encode(), str(worker).encode(), c.byref(options), c.byref(handle), expect=ARG)
        self.api.call("request_release", 0)
        self.api.call("request_release", 12345, expect=HANDLE)

    def test_20_decode_batching_configuration_and_cpu_fallback(self):
        self.assertEqual(c.sizeof(ClientOptions), 104)
        options = self.api.defaults("client", ClientOptions)
        worker = Path(self.worker).resolve()
        runtime, executable = str(worker.parent).encode("utf-8"), str(worker).encode("utf-8")
        for size in (0, 5, 0xffffffff):
            handle = U64(123)
            self.api.call("client_start_with_batching", runtime, executable, c.byref(options),
                          size, c.byref(handle), expect=ARG)
            self.assertEqual(handle.value, 0)
        options.max_active_requests = 1
        handle = U64(123)
        self.api.call("client_start_with_batching", runtime, executable, c.byref(options),
                      2, c.byref(handle), expect=ARG)
        self.assertEqual(handle.value, 0)
        options.max_active_requests, options.reserved[0] = 4, 1
        self.api.call("client_start_with_batching", runtime, executable, c.byref(options),
                      2, c.byref(handle), expect=ARG)
        options.reserved[0], options.abi_version = 0, 99
        self.api.call("client_start_with_batching", runtime, executable, c.byref(options),
                      2, c.byref(handle), expect=ARG)
        for backend in (1, 2, 3):
            for size in (1, 2, 4):
                with self.subTest(backend=backend, batch_size=size):
                    with Run(self.api, self.worker, backend=backend, batch_size=size) as run:
                        sessions = [run.session() for _ in range(4)]
                        prompts = [[3 + index, 31 + index] for index in range(4)]
                        requests = [run.submit(session, prompt, count=8)
                                    for session, prompt in zip(sessions, prompts)]
                        for request, prompt in zip(requests, prompts):
                            self.complete(self.api.drain(request), oracle(prompt, 8))
                            run.release("request", request)
                        run.quiet()
                        for session in sessions:
                            run.release("session", session)

    def test_19_runtime_memory_states_layout_and_lifetime(self):
        self.assertEqual(c.sizeof(RuntimeMemoryInfo), 120)
        self.assertEqual(RuntimeMemoryInfo.device_bytes.offset, 16)
        self.assertEqual(RuntimeMemoryInfo.epoch.offset, 80)
        with Run(self.api, self.worker) as run:
            info = output(RuntimeMemoryInfo)
            self.api.call("client_query_memory", run.client, 0, c.byref(info))
            self.assertEqual(info.state, 0)  # The default CPU factory owns no runtime.
            self.assertEqual(info.epoch, run.snapshot().epoch)
        with Run(self.api, self.worker, allocation=True, fixture="fake-runtime-memory-model") as run:
            bad = output(RuntimeMemoryInfo)
            bad.struct_size -= 1
            before_bad = bytes(bad)
            self.api.call("client_query_memory", run.client, 0, c.byref(bad), expect=ARG)
            self.assertEqual(bytes(bad), before_bad)
            first = output(RuntimeMemoryInfo)

            def ready(target):
                def available():
                    self.api.call("client_query_memory", run.client, 0, c.byref(target))
                    return target.state == 2
                until(available, "CPU runtime observation did not become available")

            ready(first)
            self.assertEqual((first.state, first.device_bytes, first.pinned_host_bytes), (2, 1024, 16))
            self.assertEqual((first.device_limit_bytes, first.pinned_host_limit_bytes), (111111, 222222))
            second = output(RuntimeMemoryInfo)
            second.device_bytes = 999
            self.api.call("client_query_memory", run.client, 0, c.byref(second))
            self.assertEqual((second.state, second.device_bytes, second.device_allocations), (1, 0, 0))
            self.assertEqual(second.epoch, first.epoch)

            class Extended(c.Structure):
                _fields_ = [("info", RuntimeMemoryInfo), ("tail", U64)]

            extended = Extended(output(RuntimeMemoryInfo), 0xA5A5A5A5A5A5A5A5)
            extended.info.struct_size = c.sizeof(extended)
            ready(extended.info)
            self.assertEqual(extended.tail, 0xA5A5A5A5A5A5A5A5)
            self.assertEqual(extended.info.state, 2)
            session = run.session()
            request = run.submit(session, [5, 7], 12)
            tokens, terminal = self.api.drain(request)
            self.assertEqual(tokens, oracle([5, 7], 12))
            self.assertEqual(terminal.progress.status, 4)
            run.release("request", request)
            run.release("model", run.model)
            ready(first)
            self.assertEqual(first.device_bytes, 1024)  # Session still holds the model.
            run.release("session", session)

            def freed():
                self.api.call("client_query_memory", run.client, 0, c.byref(first))
                return first.state == 2 and first.device_bytes == first.pinned_host_bytes == 0

            until(freed, "Runtime memory query retained a released model")
            self.assertEqual((first.device_allocations, first.pinned_host_allocations), (0, 0))
            self.assertEqual((first.peak_device_bytes, first.peak_pinned_host_bytes), (4096, 128))

    def test_02_three_backends_concurrent_continuation(self):
        for backend in (1, 2, 3):
            with self.subTest(backend=backend), Run(self.api, self.worker, backend, allocation=True) as run:
                info = output(ModelInfo)
                self.api.call("model_info", run.model, c.byref(info))
                self.assertEqual((info.backend, info.vocab_size, info.has_memory_info), (backend, 2048, 1))
                self.assertEqual((info.max_device_bytes, info.max_pinned_host_bytes, info.max_session_host_bytes), (111111, 222222, 333333))
                self.assertEqual(info.epoch, run.snapshot().epoch)
                sessions = [run.session() for _ in range(4)]
                prompts = [[11 + i, 20 + i, 31 + i] for i in range(4)]
                requests = [run.submit(session, prompt, 24 + i) for i, (session, prompt) in enumerate(zip(sessions, prompts))]
                with ThreadPoolExecutor(4) as pool:
                    captures = list(pool.map(self.api.drain, requests))
                for i, (session, request, prompt, captured) in enumerate(zip(sessions, requests, prompts, captures)):
                    self.complete(captured, oracle(prompt, 24 + i))
                    state = run.session_info(session)
                    self.assertEqual((state.has_context_length, state.context_length, state.busy), (1, len(prompt) + 24 + i, 0))
                    run.release("request", request)
                for i, session in enumerate(sessions):
                    request = run.submit(session, [71, 72], 4)
                    self.complete(self.api.drain(request), oracle([71, 72], 4, 27 + i))
                    run.release("request", request)
                run.quiet()

    def test_03_bad_output_does_not_consume(self):
        with Run(self.api, self.worker) as run:
            session = run.session()
            request = run.submit(session, [10, 12], 8)
            event, error = output(Event), output(Error)
            event.struct_size -= 1
            before = bytes(event)
            self.api.call("request_read", request, 0, c.byref(event), expect=ARG)
            self.assertEqual(bytes(event), before)
            event.struct_size, error.abi_version = c.sizeof(Event), 99
            self.assertEqual(self.api.dll.aila_inference_request_read(request, 0, c.byref(event), c.byref(error)), ARG)
            self.api.call("request_read", session, 0, c.byref(event), expect=HANDLE)
            self.complete(self.api.drain(request), oracle([10, 12], 8))

    def test_04_request_validation_and_busy(self):
        with Run(self.api, self.worker) as run:
            session, handle = run.session(), U64()
            prompt, settings = (I32 * 2)(10, 11), self.api.request_options(8)
            for field, value in (("temperature", math.nan), ("top_p", math.inf), ("do_sample", 2),
                                 ("thinking_budget_tokens", -2), ("stop_token_count", 1000)):
                invalid = RequestOptions.from_buffer_copy(settings)
                setattr(invalid, field, value)
                self.api.call("request_submit", session, prompt, 2, c.byref(invalid), 0, c.byref(handle), expect=ARG)
                self.assertEqual(handle.value, 0)
            self.api.call("request_submit", session, None, 65, c.byref(settings), 0, c.byref(handle), expect=ARG)
            bad = (I32 * 1)(2048)
            self.api.call("request_submit", session, bad, 1, c.byref(settings), 0, c.byref(handle), expect=ARG)
            self.assertEqual(run.session_info(session).context_length, 0)
            active = run.submit(session, [1022], 8)
            failure = self.api.call("request_submit", session, prompt, 2, c.byref(settings), 0, c.byref(handle), expect=BUSY)
            self.assertEqual(failure.remote_error, 2)
            self.complete(self.api.drain(active), oracle([1022], 8))

    def test_05_thinking_budgets_and_short_capacity(self):
        with Run(self.api, self.worker) as run:
            sessions, requests, closes = [], [], (I32 * 2)(1002, 1003)
            for budget in (-1, 0, 2, 5):
                session = run.session()
                sessions.append(session)
                settings = self.api.request_options(8)
                if budget >= 0:
                    settings.thinking_budget_tokens, settings.prompt_opens_think = budget, 1
                    settings.thinking_open_token_id, settings.thinking_close_token_id = 1001, 1002
                    settings.thinking_close_token_ids, settings.thinking_close_token_count = closes, 2
                requests.append(run.submit(session, [41], options=settings))
            with ThreadPoolExecutor(4) as pool:
                captures = list(pool.map(self.api.drain, requests))
            for budget, capture in zip((-1, 0, 2, 5), captures):
                self.complete(capture, oracle([41], 8, budget=budget))
            for request in requests:
                run.release("request", request)
            settings = self.api.request_options(1)
            settings.thinking_budget_tokens, settings.prompt_opens_think = 0, 1
            settings.thinking_open_token_id, settings.thinking_close_token_id = 1001, 1002
            settings.thinking_close_token_ids, settings.thinking_close_token_count = closes, 2
            request = run.submit(sessions[0], [71, 72], options=settings)
            captured = self.api.drain(request)
            self.complete(captured, [])
            self.assertEqual(captured[1].progress.prompt_tokens_processed, 2)
            self.assertEqual(run.session_info(sessions[0]).context_length, 11)

    def test_06_read_release_and_single_consumer(self):
        with Run(self.api, self.worker) as run:
            session = run.session()
            request = run.submit(session, [1022], 16)
            with ThreadPoolExecutor(1) as pool:
                reader = pool.submit(self.api.read, request, 3000)
                until(lambda: self.api.read(request, 0)[0] == BUSY, "Overlapping Read did not return BUSY", 0.3)
                started = time.monotonic()
                run.release("request", request)
                self.assertLess(time.monotonic() - started, 0.2)
                self.assertEqual(reader.result(timeout=1)[0], HANDLE)
            self.assertEqual(self.api.read(request, 0)[0], HANDLE)
            until(lambda: not run.session_info(session).busy, "Released request did not cancel")
            run.quiet()
            length = run.session_info(session).context_length
            self.assertIn(length, (0, 1))
            replacement = run.submit(session, [12], 2)
            self.complete(self.api.drain(replacement), oracle([12], 2, length))
            self.assertNotEqual(replacement, request)

    def test_07_slow_consumer_cancel_and_peer(self):
        with Run(self.api, self.worker) as run:
            a, b = run.session(), run.session()
            slow, peer = run.submit(a, [13, 14], 256), run.submit(b, [71, 72], 16)
            self.complete(self.api.drain(peer), oracle([71, 72], 16))
            cancelled = U32()
            self.api.call("request_cancel", slow, 0, c.byref(cancelled))
            self.api.call("request_cancel", slow, 0, c.byref(cancelled))
            captured = self.api.drain(slow)
            self.assertIn(captured[1].progress.end_reason, (3, 4))
            run.release("request", slow)
            run.release("request", peer)
            run.quiet()

    def test_08_parent_release_preserves_children(self):
        with Run(self.api, self.worker) as run:
            model, client, session = run.model, run.client, run.session()
            run.release("model", model)
            run.release("client", client)
            request = run.submit(session, [61, 62], 16)
            run.release("session", session)
            self.complete(self.api.drain(request), oracle([61, 62], 16))
            self.api.call("model_info", model, c.byref(output(ModelInfo)), expect=HANDLE)
            self.api.call("client_snapshot", client, c.byref(output(ClientInfo)), expect=HANDLE)
            run.release("request", request)

    def test_09_retained_handle_limit_and_slot_reuse(self):
        with Run(self.api, self.worker, retained=4) as run:
            session, requests = run.session(), []
            for i in range(4):
                request = run.submit(session, [11], 2)
                self.complete(self.api.drain(request), oracle([11], 2, 3 * i))
                requests.append(request)
            settings, handle, ids = self.api.request_options(2), U64(99), (I32 * 1)(11)
            self.api.call("request_submit", session, ids, 1, c.byref(settings), 0, c.byref(handle), expect=LIMIT)
            self.assertEqual(handle.value, 0)
            self.assertEqual(run.session_info(session).context_length, 12)
            for request in requests:
                run.release("request", request)
            run.quiet()
            replacement = run.submit(session, [11], 2)
            self.complete(self.api.drain(replacement), oracle([11], 2, 12))
            self.assertNotIn(replacement, requests)
            self.api.call("request_query", requests[0], 0, c.byref(output(RequestInfo)), expect=HANDLE)

    def test_10_crash_and_old_epoch(self):
        with Run(self.api, self.worker) as old:
            previous_epoch, session = old.snapshot().epoch, old.session()
            peer = old.submit(session, [71, 72], 256)
            crashed = old.submit(old.session(), [1023], 16)
            for request in (peer, crashed):
                _, terminal = self.api.drain(request)
                self.assertEqual((terminal.local_failure, terminal.sequence, terminal.progress.status), (1, 0, 6))
                self.assertIn(terminal.progress.remote_error, (6, 14))
            with Run(self.api, self.worker) as new:
                self.assertNotEqual(new.snapshot().epoch, previous_epoch)
                handle = U64()
                self.api.call("session_create", old.model, 0, c.byref(handle), expect=CLOSED)
                request = new.submit(new.session(), [11], 2)
                self.complete(self.api.drain(request), oracle([11], 2))

    def test_11_repeated_lifecycle_os_handles(self):
        baseline = process_handle_count()
        for _ in range(3):
            with Run(self.api, self.worker) as run:
                for _ in range(4):
                    session = run.session()
                    request = run.submit(session, [31, 32], 2)
                    self.complete(self.api.drain(request), oracle([31, 32], 2))
                    run.release("request", request)
                    run.quiet()
                    run.release("session", session)
        until(lambda: process_handle_count() <= baseline + 2, "C ABI lifecycles leaked OS handles")

    def test_12_context_replay_and_output_validation(self):
        for backend in (1, 2, 3):
            with self.subTest(backend=backend), Run(self.api, self.worker, backend) as run:
                session, peer, state = run.session(), run.session(), output(SessionInfo)
                prompt = [21, 22, 23]
                initial = run.submit(session, prompt, 6)
                captured = self.api.drain(initial)
                self.complete(captured, oracle(prompt, 6))
                run.release("request", initial)
                history = prompt + captured[0]
                reference = run.submit(peer, history, 4)
                expected = self.api.drain(reference)
                run.release("request", reference)
                self.api.call("session_truncate", session, 4, 0, c.byref(state))
                self.assertEqual((state.busy, state.has_context_length, state.context_length), (0, 1, 4))
                replay = run.submit(session, history[state.context_length:], 4)
                self.complete(self.api.drain(replay), expected[0])
                run.release("request", replay)
                before = run.session_info(session)
                state.struct_size -= 1
                invalid_output = bytes(state)
                self.api.call("session_reset", session, 0, c.byref(state), expect=ARG)
                self.api.call("session_truncate", session, 0, 0, c.byref(state), expect=ARG)
                self.assertEqual(bytes(state), invalid_output)
                state.struct_size = c.sizeof(SessionInfo)
                error = output(Error)
                error.abi_version = 99
                self.assertEqual(self.api.dll.aila_inference_session_reset(session, 0, c.byref(state), c.byref(error)), ARG)
                self.assertEqual(self.api.dll.aila_inference_session_truncate(session, 0, 0, c.byref(state), c.byref(error)), ARG)
                for length in (before.context_length + 1, 2 ** 32 - 1):
                    self.api.call("session_truncate", session, length, 0, c.byref(state), expect=ARG)
                self.assertEqual(run.session_info(session).context_length, before.context_length)
                self.assertEqual(run.session_info(peer).context_length, before.context_length)
                self.api.call("session_reset", session, 0, c.byref(state))
                self.assertEqual((state.context_length, state.busy, state.epoch), (0, 0, before.epoch))
                repeated = run.submit(session, prompt, 6)
                self.complete(self.api.drain(repeated), captured[0])
                run.release("request", repeated)
                run.quiet()

    def test_17_worker_logs_and_level_control(self):
        with patch.dict(os.environ, {"AILA_FAKE_WORKER_LOGS": "1"}), Run(self.api, self.worker) as run:
            invalid = output(LogEvent)
            invalid.struct_size -= 1
            self.api.call("client_read_log", run.client, 0, c.byref(invalid), expect=ARG)
            status, initial = self.api.read_log(run.client, 3000)
            self.assertEqual(status, OK)
            self.assertEqual(initial.message, b"[FakeLog] model-load")
            self.assertEqual(initial.message_bytes, len(initial.message))
            self.assertEqual(initial.epoch, run.snapshot().epoch)
            self.api.call("client_set_log_level", run.client, 4, 0, expect=ARG)
            self.api.call("client_set_log_level", run.client, 0, 0)
            session, previous = run.session(), initial.event_ordinal
            request = run.submit(session, [11, 12], 4)
            self.complete(self.api.drain(request), oracle([11, 12], 4))
            run.release("request", request)
            captured = []
            for _ in range(4):
                status, event = self.api.read_log(run.client, 3000)
                self.assertEqual(status, OK)
                self.assertGreater(event.event_ordinal, previous)
                self.assertEqual(event.epoch, initial.epoch)
                self.assertEqual(event.message_bytes, len(event.message))
                self.assertEqual(event.dropped_logs, 0)
                captured.append((event.level, event.message.decode("utf-8")))
                previous = event.event_ordinal
            self.assertEqual(captured, [(0, "[FakeLog] debug"), (1, "[FakeLog] 信息 α"),
                                        (2, "[FakeLog] warning"), (3, "[FakeLog] error")])
            self.assertEqual(self.api.read_log(run.client, 0)[0], TIMEOUT)
            self.api.call("client_set_log_level", run.client, 2, 0)
            request = run.submit(session, [13, 14], 4)
            self.complete(self.api.drain(request), oracle([13, 14], 4, 6))
            run.release("request", request)
            run.quiet()
            self.api.call("client_shutdown", run.client, 5000)
            final_logs = []
            while True:
                status, event = self.api.read_log(run.client, 0)
                if status == END:
                    break
                self.assertEqual(status, OK)
                final_logs.append((event.level, event.message.decode("utf-8")))
            self.assertEqual(final_logs, [(2, "[FakeLog] warning"), (3, "[FakeLog] error")])

    def test_18_log_reader_release_preserves_children(self):
        with Run(self.api, self.worker) as run:
            session, client = run.session(), run.client
            with ThreadPoolExecutor(1) as pool:
                reader = pool.submit(self.api.read_log, client, 3000)
                until(lambda: self.api.read_log(client, 0)[0] == BUSY, "Concurrent log read did not return BUSY", 0.5)
                started = time.monotonic()
                run.release("client", client)
                self.assertLess(time.monotonic() - started, 0.2)
                self.assertEqual(reader.result(timeout=1)[0], END)
            self.assertEqual(self.api.read_log(client, 0)[0], HANDLE)
            request = run.submit(session, [11, 12], 4)
            self.complete(self.api.drain(request), oracle([11, 12], 4))

    def test_14_prefill_without_generation(self):
        api = self.api
        for backend in (1, 2, 3):
            with Run(api, self.worker, backend) as run:
                session = run.session()
                ids, handle = (I32 * 3)(1, 2, 3), U64(123)
                bad_error = output(Error)
                bad_error.struct_size -= 1
                self.assertEqual(api.dll.aila_inference_request_prefill(
                    session, ids, 3, 2, 0, c.byref(handle), c.byref(bad_error)), ARG)
                self.assertEqual(run.session_info(session).context_length, 0)
                for count, chunk in ((0, 2), (3, 0), (3, 33)):
                    api.call("request_prefill", session, ids, count, chunk, 0, c.byref(handle), expect=ARG)
                    self.assertEqual(handle.value, 0)
                request = run.prefill(session, [1, 2, 3, 4, 5])
                tokens, terminal = api.drain(request)
                self.assertEqual(tokens, [])
                self.assertEqual((terminal.sequence, terminal.progress.status, terminal.progress.end_reason,
                                  terminal.progress.prompt_tokens_processed, terminal.progress.generated_tokens),
                                 (1, 4, 1, 5, 0))
                self.assertEqual(run.session_info(session).context_length, 5)
                run.release("request", request)
                request = run.submit(session, [6], 4)
                self.assertEqual(api.drain(request)[0], oracle([6], 4, 5))
                run.release("request", request)
                info = output(SessionInfo)
                api.call("session_truncate", session, 3, 0, c.byref(info))
                accepted = [1, 2, 3, 51, 52]
                request = run.prefill(session, accepted[info.context_length:])
                self.assertEqual(api.drain(request)[0], [])
                self.assertEqual(run.session_info(session).context_length, len(accepted))
                run.release("request", request)
                run.quiet()

    def test_16_extended_generation_and_reuse(self):
        self.assertEqual(self.api.defaults("client", ClientOptions).max_generated_tokens, 1024)
        for backend in (1, 2, 3):
            with self.subTest(backend=backend), Run(self.api, self.worker, backend, max_generated=1500) as run:
                session = run.session()
                request = run.submit(session, [11, 12], 1100)
                # The fake backend sleeps once per step. On Windows each 1 ms
                # sleep may take a timer tick; 1100 steps outgrow the short
                # tests' 5 s deadline without any protocol stall.
                started = time.monotonic()
                self.complete(self.api.drain(request, timeout=30), oracle([11, 12], 1100))
                print(f"[long_cpu] backend={backend} tokens=1100 seconds={time.monotonic() - started:.3f}", flush=True)
                self.assertEqual(run.session_info(session).context_length, 1102)
                run.release("request", request)
                request = run.submit(session, [13], 16)
                self.complete(self.api.drain(request), oracle([13], 16, 1102))
                run.release("request", request)
                # Upper bounds are still checked before a remote side effect.
                prior = run.session_info(session).context_length
                ids, handle = (I32 * 1)(14), U64()
                excessive = self.api.request_options(1501)
                self.api.call("request_submit", session, ids, 1, c.byref(excessive), 0, c.byref(handle), expect=LIMIT)
                self.assertEqual(handle.value, 0)
                self.assertEqual(run.session_info(session).context_length, prior)
                run.quiet()
                state = output(SessionInfo)
                self.api.call("session_reset", session, 0, c.byref(state))
                peer = run.session()
                slow = run.submit(session, [1022], 1500)
                # The 1501-token reservation remains held until terminal ACK,
                # even if execution has finished or the consumer is paused.
                self.assertEqual(run.snapshot().reserved_tokens, 1501)
                competing = self.api.request_options(600)
                self.api.call("request_submit", peer, ids, 1, c.byref(competing), 0, c.byref(handle), expect=LIMIT)
                self.assertEqual(run.session_info(peer).context_length, 0)
                cancel = U32()
                self.api.call("request_cancel", slow, 0, c.byref(cancel))
                tokens, terminal = self.api.drain(slow)
                self.assertEqual(terminal.progress.status, 5)
                self.assertEqual(terminal.progress.event_count, len(tokens) + 1)
                run.release("request", slow)
                run.quiet()
                continued = run.submit(peer, [14], 32)
                self.complete(self.api.drain(continued), oracle([14], 32))
                run.release("request", continued)
                run.quiet()

    def test_15_request_policy_validation_and_dispatch(self):
        self.assertEqual(c.sizeof(RequestPolicy), 32)
        for backend in (1, 2, 3):
            with self.subTest(backend=backend), Run(self.api, self.worker, backend) as run:
                session = run.session()
                ids, handle = (I32 * 2)(11, 12), U64()
                settings, policy = self.api.request_options(4), output(RequestPolicy)
                for field, value in (("thinking_policy", 2), ("sampling_policy", 2),
                                     ("prefill_logits_policy", 2),
                                     ("struct_size", 31), ("abi_version", 2)):
                    invalid = RequestPolicy.from_buffer_copy(policy)
                    setattr(invalid, field, value)
                    self.api.call("request_submit_with_policy", session, ids, 2, c.byref(settings),
                                  c.byref(invalid), 0, c.byref(handle), expect=ARG)
                    self.assertEqual(handle.value, 0)
                policy.reserved[2] = 1
                self.api.call("request_submit_with_policy", session, ids, 2, c.byref(settings),
                              c.byref(policy), 0, c.byref(handle), expect=ARG)
                self.assertEqual(run.session_info(session).context_length, 0)
                policy.reserved[2] = 0
                for selected in (None, c.byref(policy)):
                    state = output(SessionInfo)
                    self.api.call("session_reset", session, 0, c.byref(state))
                    self.api.call("request_submit_with_policy", session, ids, 2, c.byref(settings),
                                  selected, 0, c.byref(handle))
                    request = run.own("request", handle.value)
                    self.complete(self.api.drain(request), oracle([11, 12], 4))
                    run.release("request", request)
                state = output(SessionInfo)
                self.api.call("session_reset", session, 0, c.byref(state))
                # The CPU backend follows its independent token recurrence; the
                # wire must accept optional policies without changing it.
                policy.thinking_policy = policy.sampling_policy = policy.prefill_logits_policy = 1
                settings.thinking_budget_tokens, settings.prompt_opens_think = 1, 1
                settings.thinking_open_token_id, settings.thinking_close_token_id = 1000, 1001
                closing = (I32 * 1)(1001)
                settings.thinking_close_token_ids, settings.thinking_close_token_count = closing, 1
                self.api.call("request_submit_with_policy", session, ids, 2, c.byref(settings),
                              c.byref(policy), 0, c.byref(handle))
                request = run.own("request", handle.value)
                tokens, terminal = self.api.drain(request)
                first = oracle([11, 12], 1)[0]
                expected = [first, 1001] + oracle([1001], 2, 3)
                self.complete((tokens, terminal), expected)
                run.release("request", request)
                run.quiet()

    def test_13_context_busy_parent_release_and_stale_handle(self):
        with Run(self.api, self.worker) as run:
            session, peer, state = run.session(), run.session(), output(SessionInfo)
            active = run.submit(session, [1022, 2], 2)
            self.api.call("session_reset", session, 0, c.byref(state), expect=BUSY)
            self.api.call("session_truncate", session, 0, 0, c.byref(state), expect=BUSY)
            request = run.submit(peer, [41, 42], 2)
            self.complete(self.api.drain(active), oracle([1022, 2], 2))
            self.complete(self.api.drain(request), oracle([41, 42], 2))
            run.release("request", active)
            run.release("request", request)
            run.quiet()
            run.release("session", session)
            replacement = run.session()
            self.assertNotEqual(session, replacement)
            self.api.call("session_reset", session, 0, c.byref(state), expect=HANDLE)
            self.api.call("session_truncate", session, 0, 0, c.byref(state), expect=HANDLE)
            self.api.call("session_reset", run.model, 0, c.byref(state), expect=HANDLE)
            run.release("model", run.model)
            self.api.call("session_reset", peer, 0, c.byref(state))
            self.assertEqual((state.context_length, state.busy), (0, 0))
            self.api.call("session_truncate", replacement, 0, 0, c.byref(state))
            self.assertEqual(state.context_length, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dll", required=True)
    parser.add_argument("--worker", required=True)
    args, remaining = parser.parse_known_args()
    if Path(args.worker).name.lower() != "ailafakeinferenceworker.exe":
        parser.error("This CPU suite requires AilaFakeInferenceWorker.exe")
    CApiTests.api, CApiTests.worker = Api(args.dll), args.worker
    unittest.main(argv=[sys.argv[0], *remaining], verbosity=2)
