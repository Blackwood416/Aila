"""Warmed Aila HTTP/SSE measurements and repeated Session lifecycle acceptance.

Windows x64 and Python standard library only. The runner starts its own Server
inside a Job Object before its primary thread can run; it never scans global
processes or attaches to an existing service. Independent single-request
references and warmup requests are retained but excluded from measured
distributions. SSE content-event intervals are client
observations, not individual-token latency; one read can contain several events.

Examples (only after crash review, with no other GPU work):
  python tools/server/service_benchmark.py --server build-refactor-server/AilaServer.exe \
      --worker build-refactor/AilaWorker.exe --model models/qwen3-0.6B-bnb-nf4-offline \
      --output tmp/refactor/service --mode probe --execution-trace
  python tools/server/service_benchmark.py ... --backend qwen35_nf4 --mode lifecycle --allow-stress \
      --lifecycle-requests 512 --lifecycle-width 4

The configured prompt size is text/repetitions, not a tokenizer estimate.
Actual template-inclusive prompt token counts come from validated final usage.
The OS samples cover Server/Worker RSS, private bytes and handles, not total GPU
memory. Optional growth gates compare the idle samples surrounding lifecycle
requests; an ungated flat-looking sample is not proof that every allocator frees.
"""

from __future__ import annotations

import argparse
import codecs
import concurrent.futures
import ctypes
import datetime
import hashlib
import http.client
import json
import math
import os
import platform
import re
import socket
import statistics
import subprocess
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

from gpu_run_guard import GpuRunGuard

# The standalone runner and offline analyzer share no service/GPU imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "diagnostics"))
from execution_trace import analyze_file as analyze_execution_trace


MODEL_IDS = {"qwen3_nf4": "qwen3", "qwen35_nf4": "qwen3.5", "qwen35_bf16": "qwen3.5"}
FRAME_BYTES = 16384
STREAM_BYTES = 65536
BODY_BYTES = 65536
RETAINED_REQUEST_HANDLES = 256  # Current IPC hard limit; lifecycle must exceed it.
DEFAULT_PROMPT = "Explain briefly why independent request state matters in a shared language model service."


class BenchmarkError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise BenchmarkError(message)


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, f"{name} must be an integer >= {minimum}: {value!r}")
    return value


def strict_json(value):
    def constant(text):
        raise BenchmarkError(f"Non-finite JSON constant: {text}")

    def pairs(items):
        result = {}
        for key, item in items:
            require(key not in result, f"Duplicate JSON member: {key}")
            result[key] = item
        return result

    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="strict")
    return json.loads(value, parse_constant=constant, object_pairs_hook=pairs)


def encoded_json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def write_json(path, value):
    # A bugcheck can discard buffered diagnostics even after a Python close.
    # Replace complete JSON atomically, after flushing its data to the OS disk.
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, ensure_ascii=False, allow_nan=False, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def query_dxgi_video_memory():
    """Query Windows DXGI Video Memory statistics (local and shared segment budget/usage)."""
    try:
        from ctypes import wintypes
        dxgi = ctypes.WinDLL("dxgi", use_last_error=True)
        HRESULT = ctypes.c_long
        class GUID(ctypes.Structure):
            _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD),
                        ("Data3", wintypes.WORD), ("Data4", ctypes.c_byte * 8)]
        iid_factory4 = GUID(0x1bc6ea02, 0xef36, 0x464f, (ctypes.c_byte * 8)(0xbf, 0x0c, 0x21, 0xca, 0x39, 0xe5, 0x16, 0x8a))
        iid_adapter3 = GUID(0x645967A4, 0x1392, 0x4310, (ctypes.c_byte * 8)(0xA7, 0x98, 0x80, 0x53, 0xCE, 0x3E, 0x93, 0xFD))
        factory = ctypes.c_void_p()
        if hasattr(dxgi, "CreateDXGIFactory2"):
            hr = dxgi.CreateDXGIFactory2(0, ctypes.byref(iid_factory4), ctypes.byref(factory))
        else:
            hr = dxgi.CreateDXGIFactory1(ctypes.byref(iid_factory4), ctypes.byref(factory))
        if hr != 0 or not factory.value:
            return None
        vtbl = ctypes.cast(ctypes.cast(factory, ctypes.POINTER(ctypes.c_void_p)).contents, ctypes.POINTER(ctypes.c_void_p))
        EnumAdapters1 = ctypes.WINFUNCTYPE(HRESULT, ctypes.c_void_p, wintypes.UINT, ctypes.POINTER(ctypes.c_void_p))(vtbl[12])
        ReleaseF = ctypes.WINFUNCTYPE(wintypes.ULONG, ctypes.c_void_p)(vtbl[2])
        adapter = ctypes.c_void_p()
        res = None
        if EnumAdapters1(factory, 0, ctypes.byref(adapter)) == 0 and adapter.value:
            adapter_vtbl = ctypes.cast(ctypes.cast(adapter, ctypes.POINTER(ctypes.c_void_p)).contents, ctypes.POINTER(ctypes.c_void_p))
            QueryInterface = ctypes.WINFUNCTYPE(HRESULT, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))(adapter_vtbl[0])
            ReleaseA = ctypes.WINFUNCTYPE(wintypes.ULONG, ctypes.c_void_p)(adapter_vtbl[2])
            adapter3 = ctypes.c_void_p()
            if QueryInterface(adapter, ctypes.byref(iid_adapter3), ctypes.byref(adapter3)) == 0 and adapter3.value:
                adapter3_vtbl = ctypes.cast(ctypes.cast(adapter3, ctypes.POINTER(ctypes.c_void_p)).contents, ctypes.POINTER(ctypes.c_void_p))
                class DXGI_QUERY_VIDEO_MEMORY_INFO(ctypes.Structure):
                    _fields_ = [("Budget", ctypes.c_uint64), ("CurrentUsage", ctypes.c_uint64),
                                ("AvailableForReservation", ctypes.c_uint64), ("CurrentReservation", ctypes.c_uint64)]
                QueryVideoMemoryInfo = ctypes.WINFUNCTYPE(HRESULT, ctypes.c_void_p, wintypes.UINT, wintypes.UINT, ctypes.POINTER(DXGI_QUERY_VIDEO_MEMORY_INFO))(adapter3_vtbl[14])
                ReleaseA3 = ctypes.WINFUNCTYPE(wintypes.ULONG, ctypes.c_void_p)(adapter3_vtbl[2])
                local_mem = DXGI_QUERY_VIDEO_MEMORY_INFO()
                shared_mem = DXGI_QUERY_VIDEO_MEMORY_INFO()
                if QueryVideoMemoryInfo(adapter3, 0, 0, ctypes.byref(local_mem)) == 0:
                    res = {
                        "local_vram_budget_bytes": local_mem.Budget,
                        "local_vram_usage_bytes": local_mem.CurrentUsage,
                    }
                    if QueryVideoMemoryInfo(adapter3, 0, 1, ctypes.byref(shared_mem)) == 0:
                        res["shared_sysmem_budget_bytes"] = shared_mem.Budget
                        res["shared_sysmem_usage_bytes"] = shared_mem.CurrentUsage
                ReleaseA3(adapter3)
            ReleaseA(adapter)
        ReleaseF(factory)
        return res
    except Exception:
        return None


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distribution(values):
    values = sorted(float(value) for value in values)
    require(all(math.isfinite(value) for value in values), "Non-finite measurement")
    if not values:
        return {"count": 0, "minimum": None, "maximum": None, "mean": None,
                "p50": None, "p90": None, "p95": None, "p99": None}

    def percentile(fraction):
        rank = (len(values) - 1) * fraction
        lower = math.floor(rank)
        upper = math.ceil(rank)
        return values[lower] + (values[upper] - values[lower]) * (rank - lower)

    return {"count": len(values), "minimum": values[0], "maximum": values[-1],
            "mean": statistics.fmean(values), **{f"p{p}": percentile(p / 100) for p in (50, 90, 95, 99)}}


def summarize_measurements(requests, batches):
    """Only the explicitly measured phase contributes to performance results."""
    measured = [item for item in requests if item["phase"] == "measured"]
    require(all(item.get("outcome") == "completed" and not item.get("error") for item in measured),
            "Failed measured requests cannot be removed to improve a distribution")
    result = []
    for profile, width in sorted({(item["profile"], item["width"]) for item in measured}):
        rows = [item for item in measured if (item["profile"], item["width"]) == (profile, width)]
        groups = [item for item in batches if item["phase"] == "measured" and
                  (item["profile"], item["width"]) == (profile, width)]
        require(groups and all(item["passed"] for item in groups), "Missing or failed measured batch")
        intervals = [value for item in rows for value in item["content_event_intervals_seconds"]]
        tokens = sum(item["usage"]["completion_tokens"] for item in rows)
        seconds = sum(item["span_seconds"] for item in groups)
        result.append({
            "profile": profile, "concurrency": width, "requests": len(rows), "rounds": len(groups),
            "ttft_seconds": distribution(item["ttft_seconds"] for item in rows),
            "request_latency_seconds": distribution(item["request_latency_seconds"] for item in rows),
            "content_event_interval_seconds": distribution(intervals),
            "request_completion_tokens_per_second": distribution(item["completion_tokens_per_second"] for item in rows),
            "batch_completion_tokens_per_second": distribution(item["completion_tokens_per_second"] for item in groups),
            "total_completion_tokens": tokens, "sum_batch_spans_seconds": seconds,
            "completion_tokens_per_second_over_batch_spans": tokens / seconds,
            "actual_prompt_tokens": distribution(item["usage"]["prompt_tokens"] for item in rows),
            "by_prompt_kind": {kind: {
                "requests": sum(item["prompt_kind"] == kind for item in rows),
                "ttft_seconds": distribution(item["ttft_seconds"] for item in rows if item["prompt_kind"] == kind),
                "request_latency_seconds": distribution(item["request_latency_seconds"] for item in rows if item["prompt_kind"] == kind),
                "actual_prompt_tokens": distribution(item["usage"]["prompt_tokens"] for item in rows if item["prompt_kind"] == kind),
            } for kind in sorted({item["prompt_kind"] for item in rows})},
        })
    return result


class JsonLines:
    def __init__(self, path):
        self.file = path.open("x", encoding="utf-8", newline="\n")
        self.lock = threading.Lock()

    def append(self, value):
        line = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        with self.lock:
            self.file.write(line + "\n")
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.file.close()


class SseParser:
    """Incremental parser whose event timestamps are the completing read time."""

    def __init__(self, model_id, maximum, context, output_limit):
        self.model_id = model_id
        self.maximum = maximum
        self.context = context
        self.output_limit = output_limit
        self.utf8 = codecs.getincrementaldecoder("utf-8")("strict")
        self.buffer = bytearray()
        self.data = []
        self.frame_bytes = 0
        self.total_bytes = 0
        self.identity = None
        self.created = None
        self.role_seen = False
        self.finish_reason = None
        self.usage = None
        self.done_ns = None
        self.content = []
        self.reasoning = []
        self.content_events = []
        self.events = []

    def feed(self, value, observed_ns):
        require(isinstance(value, bytes), "SSE input must be bytes")
        self.total_bytes += len(value)
        require(self.total_bytes <= self.output_limit, "Response exceeded configured output budget")
        self.utf8.decode(value, final=False)
        self.buffer.extend(value)
        while b"\n" in self.buffer:
            line, _, remaining = self.buffer.partition(b"\n")
            self.buffer = bytearray(remaining)
            self.frame_bytes += len(line) + 1
            require(self.frame_bytes <= FRAME_BYTES, "SSE frame exceeded the server handoff bound")
            if line.endswith(b"\r"):
                line = line[:-1]
            if not line:
                if self.data:
                    self._event(b"\n".join(self.data), observed_ns)
                self.data = []
                self.frame_bytes = 0
            elif line.startswith(b"data:"):
                self.data.append(bytes(line[5:]).removeprefix(b" "))
            elif line.startswith(b":"):
                continue
            else:
                raise BenchmarkError(f"Unexpected SSE field: {bytes(line[:80])!r}")
        require(self.frame_bytes + len(self.buffer) <= FRAME_BYTES, "Unbounded incomplete SSE frame")

    def _event(self, raw, observed_ns):
        require(self.done_ns is None, "Data followed the unique [DONE] event")
        if raw == b"[DONE]":
            require(self.finish_reason is not None and self.usage is not None, "[DONE] preceded finish or final usage")
            self.done_ns = observed_ns
            self.events.append({"kind": "done", "observed_ns": observed_ns})
            return
        value = strict_json(raw)
        require(isinstance(value, dict), "SSE JSON is not an object")
        require("error" not in value, f"Unexpected SSE error: {value.get('error')}")
        require(value.get("object") == "chat.completion.chunk" and value.get("model") == self.model_id,
                "Unexpected SSE object or served model")
        identity = value.get("id")
        require(isinstance(identity, str) and identity, "Missing SSE request ID")
        if self.identity is None:
            self.identity = identity
            self.created = integer(value.get("created"), "created", 1)
        require(identity == self.identity and integer(value.get("created"), "created", 1) == self.created,
                "Crossed SSE request identity")
        choices = value.get("choices")
        require(isinstance(choices, list), "Missing SSE choices")
        if not choices:
            require(self.finish_reason is not None and self.usage is None, "Duplicated or premature usage event")
            self.usage = self._usage(value.get("usage"))
            self.events.append({"kind": "usage", "observed_ns": observed_ns})
            return
        require(len(choices) == 1 and self.finish_reason is None, "Choices followed finish or had an unexpected count")
        require("usage" in value and value["usage"] is None, "Ordinary include_usage chunk must carry usage:null")
        choice = choices[0]
        require(isinstance(choice, dict) and integer(choice.get("index"), "choice.index") == 0, "Unexpected SSE choice index")
        delta = choice.get("delta")
        require(isinstance(delta, dict) and set(delta) <= {"role", "content", "reasoning_content"}, "Non-text SSE delta")
        if "role" in delta:
            require(not self.role_seen and not self.events and delta["role"] == "assistant", "Invalid initial assistant role")
            self.role_seen = True
        require(self.role_seen, "Generated content preceded the assistant role")
        channels = []
        characters = 0
        for key, output in (("content", self.content), ("reasoning_content", self.reasoning)):
            if key in delta:
                require(isinstance(delta[key], str), f"Non-string {key}")
                output.append(delta[key])
                if delta[key]:
                    channels.append(key)
                    characters += len(delta[key])
        if channels:
            self.content_events.append({"observed_ns": observed_ns, "channels": channels, "characters": characters})
        reason = choice.get("finish_reason")
        if reason is not None:
            require(reason in ("length", "stop") and not delta, "Invalid successful SSE finish")
            self.finish_reason = reason
        self.events.append({"kind": "finish" if reason is not None else "content" if channels else "role",
                            "observed_ns": observed_ns})

    def _usage(self, value):
        require(isinstance(value, dict), "Missing final usage object")
        prompt = integer(value.get("prompt_tokens"), "usage.prompt_tokens", 1)
        completion = integer(value.get("completion_tokens"), "usage.completion_tokens", 1)
        require(completion <= self.maximum, "Usage exceeds requested generation limit")
        require(integer(value.get("total_tokens"), "usage.total_tokens") == prompt + completion, "Inconsistent usage sum")
        require(prompt + self.maximum <= self.context, "Usage contradicts configured context admission")
        require(self.finish_reason != "length" or completion == self.maximum, "Length finish lost completion tokens")
        return value

    def finish(self):
        self.utf8.decode(b"", final=True)
        require(not self.buffer and not self.data and self.frame_bytes == 0, "EOF split an SSE frame")
        require(self.done_ns is not None and self.usage is not None and self.finish_reason is not None,
                "EOF without a successful finish, usage and [DONE]")
        require(self.content_events, "No generated content was available for TTFT; choose another prompt")

    def measurement(self, start_ns):
        times = [item["observed_ns"] for item in self.content_events]
        require(all(left <= right for left, right in zip(times, times[1:])), "Non-monotonic event clock")
        content = "".join(self.content)
        reasoning = "".join(self.reasoning)
        return {
            "id": self.identity, "usage": self.usage, "finish_reason": self.finish_reason,
            "ttft_seconds": (times[0] - start_ns) / 1e9 if times else None,
            "request_latency_seconds": (self.done_ns - start_ns) / 1e9 if self.done_ns is not None else None,
            "content_event_intervals_seconds": [(right - left) / 1e9 for left, right in zip(times, times[1:])],
            "content_events": self.content_events, "events": self.events,
            "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "reasoning_sha256": hashlib.sha256(reasoning.encode("utf-8")).hexdigest(),
            "content_characters": len(content), "reasoning_characters": len(reasoning),
            "wire_payload_bytes": self.total_bytes,
        }


class WindowsJobProcess:
    """Own the suspended Server and all descendants before allowing startup.

    Only JobObjectBasicProcessIdList is enumerated, never system-wide processes.
    A Worker query handle is retained after checking job membership and its
    executable path. PID reuse outside this job cannot redirect a later sample.
    """

    def __init__(self, command, directory, stdout, stderr, environment):
        require(os.name == "nt" and ctypes.sizeof(ctypes.c_void_p) == 8, "Service execution requires Windows x64")
        import msvcrt

        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.psapi = ctypes.WinDLL("psapi", use_last_error=True)
        self.job = None
        self.handle = None
        self.pid = None
        self.worker_handle = None
        self.worker_pid = None
        self.worker_image = None
        self.returncode = None
        self.closed = False
        DWORD = ctypes.c_uint32
        HANDLE = ctypes.c_void_p
        SIZE_T = ctypes.c_size_t
        BOOL = ctypes.c_int32

        class StartupInfo(ctypes.Structure):
            _fields_ = [("cb", DWORD), ("lpReserved", ctypes.c_wchar_p), ("lpDesktop", ctypes.c_wchar_p),
                        ("lpTitle", ctypes.c_wchar_p), ("dwX", DWORD), ("dwY", DWORD),
                        ("dwXSize", DWORD), ("dwYSize", DWORD), ("dwXCountChars", DWORD),
                        ("dwYCountChars", DWORD), ("dwFillAttribute", DWORD), ("dwFlags", DWORD),
                        ("wShowWindow", ctypes.c_uint16), ("cbReserved2", ctypes.c_uint16),
                        ("lpReserved2", ctypes.c_void_p), ("hStdInput", HANDLE),
                        ("hStdOutput", HANDLE), ("hStdError", HANDLE)]

        class StartupInfoEx(ctypes.Structure):
            _fields_ = [("StartupInfo", StartupInfo), ("lpAttributeList", ctypes.c_void_p)]

        class ProcessInfo(ctypes.Structure):
            _fields_ = [("hProcess", HANDLE), ("hThread", HANDLE), ("dwProcessId", DWORD), ("dwThreadId", DWORD)]

        class BasicLimits(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", DWORD), ("MinimumWorkingSetSize", SIZE_T), ("MaximumWorkingSetSize", SIZE_T),
                        ("ActiveProcessLimit", DWORD), ("Affinity", SIZE_T), ("PriorityClass", DWORD), ("SchedulingClass", DWORD)]

        class IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount", "ReadTransferCount",
                "WriteTransferCount", "OtherTransferCount")]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BasicLimits), ("IoInfo", IoCounters),
                        ("ProcessMemoryLimit", SIZE_T), ("JobMemoryLimit", SIZE_T),
                        ("PeakProcessMemoryUsed", SIZE_T), ("PeakJobMemoryUsed", SIZE_T)]

        class MemoryCounters(ctypes.Structure):
            _fields_ = [("cb", DWORD), ("PageFaultCount", DWORD)] + [(name, SIZE_T) for name in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]

        class FileTime(ctypes.Structure):
            _fields_ = [("low", DWORD), ("high", DWORD)]

        self.MemoryCounters = MemoryCounters
        self.FileTime = FileTime
        signatures = {
            "CreateJobObjectW": (HANDLE, [ctypes.c_void_p, ctypes.c_wchar_p]),
            "SetInformationJobObject": (BOOL, [HANDLE, ctypes.c_int, ctypes.c_void_p, DWORD]),
            "AssignProcessToJobObject": (BOOL, [HANDLE, HANDLE]),
            "QueryInformationJobObject": (BOOL, [HANDLE, ctypes.c_int, ctypes.c_void_p, DWORD, ctypes.POINTER(DWORD)]),
            "TerminateJobObject": (BOOL, [HANDLE, DWORD]),
            "IsProcessInJob": (BOOL, [HANDLE, HANDLE, ctypes.POINTER(BOOL)]),
            "OpenProcess": (HANDLE, [DWORD, BOOL, DWORD]),
            "CloseHandle": (BOOL, [HANDLE]),
            "SetHandleInformation": (BOOL, [HANDLE, DWORD, DWORD]),
            "InitializeProcThreadAttributeList": (BOOL, [ctypes.c_void_p, DWORD, DWORD, ctypes.POINTER(SIZE_T)]),
            "UpdateProcThreadAttribute": (BOOL, [ctypes.c_void_p, DWORD, SIZE_T, ctypes.c_void_p, SIZE_T,
                                                 ctypes.c_void_p, ctypes.c_void_p]),
            "DeleteProcThreadAttributeList": (None, [ctypes.c_void_p]),
            "CreateProcessW": (BOOL, [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_void_p, ctypes.c_void_p,
                                      BOOL, DWORD, ctypes.c_void_p, ctypes.c_wchar_p,
                                      ctypes.POINTER(StartupInfoEx), ctypes.POINTER(ProcessInfo)]),
            "ResumeThread": (DWORD, [HANDLE]),
            "TerminateProcess": (BOOL, [HANDLE, DWORD]),
            "WaitForSingleObject": (DWORD, [HANDLE, DWORD]),
            "GetExitCodeProcess": (BOOL, [HANDLE, ctypes.POINTER(DWORD)]),
            "QueryFullProcessImageNameW": (BOOL, [HANDLE, DWORD, ctypes.c_wchar_p, ctypes.POINTER(DWORD)]),
            "GetProcessHandleCount": (BOOL, [HANDLE, ctypes.POINTER(DWORD)]),
            "GetProcessTimes": (BOOL, [HANDLE, ctypes.POINTER(FileTime), ctypes.POINTER(FileTime),
                                        ctypes.POINTER(FileTime), ctypes.POINTER(FileTime)]),
        }
        for name, (restype, argtypes) in signatures.items():
            function = getattr(self.kernel, name)
            function.restype, function.argtypes = restype, argtypes
        self.psapi.GetProcessMemoryInfo.restype = BOOL
        self.psapi.GetProcessMemoryInfo.argtypes = [HANDLE, ctypes.POINTER(MemoryCounters), DWORD]

        thread_handle = None
        attribute_list = None
        attribute_initialized = False
        inherited = []
        assigned = False
        try:
            self.job = self.kernel.CreateJobObjectW(None, None)
            self._check(self.job, "CreateJobObjectW")
            limits = ExtendedLimits()
            limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            self._check(self.kernel.SetInformationJobObject(self.job, 9, ctypes.byref(limits), ctypes.sizeof(limits)),
                        "SetInformationJobObject")
            with open(os.devnull, "rb") as stdin:
                inherited = [msvcrt.get_osfhandle(item.fileno()) for item in (stdin, stdout, stderr)]
                for handle in inherited:
                    self._check(self.kernel.SetHandleInformation(handle, 1, 1), "SetHandleInformation")
                size = SIZE_T()
                self.kernel.InitializeProcThreadAttributeList(None, 1, 0, ctypes.byref(size))
                require(size.value > 0, "No attribute-list storage size was returned")
                storage = ctypes.create_string_buffer(size.value)
                attribute_list = ctypes.cast(storage, ctypes.c_void_p)
                self._check(self.kernel.InitializeProcThreadAttributeList(attribute_list, 1, 0, ctypes.byref(size)),
                            "InitializeProcThreadAttributeList")
                attribute_initialized = True
                handle_list = (HANDLE * len(inherited))(*inherited)
                self._check(self.kernel.UpdateProcThreadAttribute(attribute_list, 0, 0x20002,
                            ctypes.cast(handle_list, ctypes.c_void_p), ctypes.sizeof(handle_list), None, None),
                            "UpdateProcThreadAttribute(handle list)")
                startup = StartupInfoEx()
                startup.StartupInfo.cb = ctypes.sizeof(startup)
                startup.StartupInfo.dwFlags = 0x100  # STARTF_USESTDHANDLES
                startup.StartupInfo.hStdInput, startup.StartupInfo.hStdOutput, startup.StartupInfo.hStdError = inherited
                startup.lpAttributeList = attribute_list
                info = ProcessInfo()
                command_line = ctypes.create_unicode_buffer(subprocess.list2cmdline(command))
                environment_block = ctypes.create_unicode_buffer("\0".join(
                    f"{key}={value}" for key, value in sorted(environment.items(), key=lambda item: item[0].casefold())) + "\0\0")
                flags = 0x08000000 | 0x00000004 | 0x00080000 | 0x00000400
                self._check(self.kernel.CreateProcessW(command[0], command_line, None, None, True, flags,
                            ctypes.cast(environment_block, ctypes.c_void_p), str(directory),
                            ctypes.byref(startup), ctypes.byref(info)), "CreateProcessW")
                self.handle, thread_handle, self.pid = info.hProcess, info.hThread, info.dwProcessId
                self._check(self.kernel.AssignProcessToJobObject(self.job, self.handle), "AssignProcessToJobObject")
                assigned = True
                require(self.kernel.ResumeThread(thread_handle) != 0xFFFFFFFF, "ResumeThread failed")
                # Clear inheritance while stdin's owned descriptor is still open.
                for handle in inherited:
                    self._check(self.kernel.SetHandleInformation(handle, 1, 0), "Clear inherited handle flag")
                inherited = []
        except BaseException:
            if self.handle:
                if assigned:
                    self.kernel.TerminateJobObject(self.job, 1)
                else:
                    self.kernel.TerminateProcess(self.handle, 1)  # Still suspended, never launched a Worker.
                self.kernel.WaitForSingleObject(self.handle, 5000)
            self.close()
            raise
        finally:
            for handle in inherited:
                self.kernel.SetHandleInformation(handle, 1, 0)
            if attribute_initialized:
                self.kernel.DeleteProcThreadAttributeList(attribute_list)
            if thread_handle:
                self.kernel.CloseHandle(thread_handle)

    @staticmethod
    def _check(value, operation):
        if not value:
            raise OSError(ctypes.get_last_error(), f"{operation} failed")

    def poll(self):
        result = self.kernel.WaitForSingleObject(self.handle, 0)
        if result == 258:
            return None
        require(result == 0, f"WaitForSingleObject failed: {result}")
        value = ctypes.c_uint32()
        self._check(self.kernel.GetExitCodeProcess(self.handle, ctypes.byref(value)), "GetExitCodeProcess")
        self.returncode = value.value
        return self.returncode

    def wait(self, seconds):
        result = self.kernel.WaitForSingleObject(self.handle, max(0, min(0xFFFFFFFE, math.ceil(seconds * 1000))))
        if result == 258:
            raise TimeoutError("Owned Server is still running")
        require(result == 0, f"WaitForSingleObject failed: {result}")
        return self.poll()

    def owned_pids(self):
        # The first two DWORDs precede an aligned array of ULONG_PTR on Win64.
        for capacity in (16, 64, 256, 1024):
            storage = ctypes.create_string_buffer(8 + capacity * ctypes.sizeof(ctypes.c_size_t))
            if self.kernel.QueryInformationJobObject(self.job, 3, storage, len(storage), None):
                count = ctypes.c_uint32.from_buffer(storage, 4).value
                require(count <= capacity, "Job returned more process IDs than its buffer")
                values = (ctypes.c_size_t * count).from_buffer(storage, 8)
                return [int(value) for value in values]
            require(ctypes.get_last_error() == 234, "QueryInformationJobObject(process IDs) failed")
        raise BenchmarkError("Owned process tree exceeded 1024 members")

    def image_path(self, handle):
        size = ctypes.c_uint32(32768)
        buffer = ctypes.create_unicode_buffer(size.value)
        self._check(self.kernel.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)),
                    "QueryFullProcessImageNameW")
        text = buffer.value
        if text.startswith("\\\\?\\UNC\\"):
            text = "\\\\" + text[8:]
        elif text.startswith("\\\\?\\"):
            text = text[4:]
        return Path(text).resolve()

    def find_worker(self, executable):
        if self.worker_handle:
            return self.worker_pid
        matches = []
        try:
            for pid in self.owned_pids():
                if pid == self.pid:
                    continue
                handle = self.kernel.OpenProcess(0x0400 | 0x0010 | 0x00100000, False, pid)
                if not handle:
                    continue  # A short-lived owned helper may have already exited.
                keep = False
                try:
                    member = ctypes.c_int32()
                    self._check(self.kernel.IsProcessInJob(handle, self.job, ctypes.byref(member)), "IsProcessInJob")
                    if member.value and self.kernel.WaitForSingleObject(handle, 0) == 258:
                        image = self.image_path(handle)
                        if os.path.normcase(str(image)) == os.path.normcase(str(executable)):
                            matches.append((pid, handle, image))
                            keep = True
                finally:
                    if not keep:
                        self.kernel.CloseHandle(handle)
            require(len(matches) <= 1, "Several matching Workers exist inside the owned process tree")
            if matches:
                self.worker_pid, self.worker_handle, self.worker_image = matches.pop()
            return self.worker_pid
        finally:
            for _, handle, _ in matches:
                self.kernel.CloseHandle(handle)

    def memory(self, role, handle, pid):
        require(self.kernel.WaitForSingleObject(handle, 0) == 258, f"{role} exited during measurement")
        counters = self.MemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        self._check(self.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb), "GetProcessMemoryInfo")
        handles = ctypes.c_uint32()
        self._check(self.kernel.GetProcessHandleCount(handle, ctypes.byref(handles)), "GetProcessHandleCount")
        creation, exit_time, kernel_time, user_time = [self.FileTime() for _ in range(4)]
        self._check(self.kernel.GetProcessTimes(handle, ctypes.byref(creation), ctypes.byref(exit_time),
                    ctypes.byref(kernel_time), ctypes.byref(user_time)), "GetProcessTimes")
        return {"role": role, "pid": pid, "creation_filetime": (creation.high << 32) | creation.low,
                "rss_bytes": counters.WorkingSetSize, "private_bytes": counters.PrivateUsage,
                "handles": handles.value}

    def sample(self):
        require(self.worker_handle is not None, "Worker query handle has not been established")
        samples = [self.memory("server", self.handle, self.pid), self.memory("worker", self.worker_handle, self.worker_pid)]
        vram = query_dxgi_video_memory()
        if vram:
            for s in samples:
                s.update(vram)
        return samples

    def terminate_owned(self):
        identities = self.owned_pids()
        self._check(self.kernel.TerminateJobObject(self.job, 1), "TerminateJobObject")
        return identities

    def close(self):
        if self.closed:
            return
        self.closed = True
        for handle in (self.worker_handle, self.handle, self.job):
            if handle:
                self.kernel.CloseHandle(handle)
        self.worker_handle = self.handle = self.job = None


def stop_owned_process(process, shutdown_file, timeout, clock=time.monotonic, sleeper=time.sleep):
    """A tested cleanup policy; only the supplied owned process object can die."""
    started = clock()
    result = {"forced_termination": False, "forced_pids": [], "shutdown_file_written": False,
              "interrupted_during_shutdown": False, "job_empty": False, "errors": []}
    try:
        if process.poll() is None:
            shutdown_file.write_text("Owned service benchmark requested shutdown.\n", encoding="utf-8")
            result["shutdown_file_written"] = True
        deadline = started + timeout
        while process.poll() is None or process.owned_pids():
            remaining = deadline - clock()
            if remaining <= 0:
                result["forced_termination"] = True
                result["forced_pids"] = process.terminate_owned()
                break
            try:
                if process.poll() is None:
                    process.wait(min(0.25, remaining))
                else:
                    sleeper(min(0.05, remaining))
            except TimeoutError:
                pass
            except KeyboardInterrupt:
                result["interrupted_during_shutdown"] = True
        if result["forced_termination"]:
            deadline = clock() + 10
            while (process.poll() is None or process.owned_pids()) and clock() < deadline:
                sleeper(0.05)
        result["exit_code"] = process.poll()
        result["remaining_owned_pids"] = process.owned_pids()
        result["job_empty"] = not result["remaining_owned_pids"] and result["exit_code"] is not None
    except BaseException as error:
        result["errors"].append(f"{type(error).__name__}: {error}")
        result["forced_termination"] = True
        try:
            result["forced_pids"] = process.terminate_owned()
            process.wait(10)
            result["exit_code"] = process.poll()
            result["remaining_owned_pids"] = process.owned_pids()
            result["job_empty"] = not result["remaining_owned_pids"] and result["exit_code"] is not None
        except BaseException as final_error:
            result["errors"].append(f"{type(final_error).__name__}: {final_error}")
    finally:
        # The Job's kill-on-close flag also contains exceptional cleanup paths.
        process.close()
        result["seconds"] = clock() - started
    return result


def log_errors(directory):
    pattern = re.compile(r"^\d{8} \d{2}:\d{2}:\d{2}\.\d+\s+\S+\s+\S+\s+(?:ERROR|FATAL)\b")
    service_pattern = re.compile(r"^(?:AilaServer:|Worker shutdown:|\[?(?:ERROR|FATAL)\]?(?:\s|:))", re.IGNORECASE)
    errors = []
    for name in ("stdout.log", "stderr.log"):
        path = directory / name
        if path.is_file():
            with path.open(encoding="utf-8", errors="replace") as source:
                for number, line in enumerate(source, 1):
                    if pattern.match(line) or service_pattern.match(line):
                        errors.append({"file": name, "line": number, "message": line.rstrip()[:2048]})
                        if len(errors) == 64:
                            return errors
    return errors


def validate_health(value, model_id, backend, allow_starting=False, expected_decode_batch_size=None):
    require(isinstance(value, dict), "Health is not an object")
    allowed = ("ok", "starting") if allow_starting else ("ok",)
    require(value.get("status") in allowed, f"Unexpected service status: {value.get('status')}")
    require(value.get("model") == model_id and value.get("backend") == backend, "Health model/backend mismatch")
    # The reported bound is the configuration ceiling, never a measured batch
    # width. It must still match the requested run, so a mistyped or dropped
    # option cannot silently benchmark the default single-request path.
    integer(value.get("max_decode_batch_size"), "max_decode_batch_size")
    if expected_decode_batch_size is not None:
        require(value["max_decode_batch_size"] == expected_decode_batch_size,
                "Health decode batch bound differs from the requested configuration")
    integer(value.get("active_requests"), "active_requests")
    for name in ("admitted", "rejected", "completed", "failed", "cancelled", "disconnect_cancellations", "slow_client_cancellations"):
        integer(value["counters"].get(name), f"counters.{name}")
    require(value["counters"]["failed"] == 0 and value["counters"]["rejected"] == 0,
            "A request failed or was rejected during the run")
    require(value["counters"]["slow_client_cancellations"] == 0, "The benchmark itself became a stalled reader")
    ipc = value["ipc"]
    require(isinstance(ipc.get("epoch"), str) and ipc["epoch"], "Missing Worker epoch")
    require(ipc.get("status") in (("starting", "ready") if allow_starting else ("ready",)), "Worker is not ready")
    for name in ("outstanding_requests", "pending_commands"):
        integer(ipc.get(name), f"ipc.{name}")
    connections = value["connections"]
    for name in ("active", "accepted", "closed", "reused_rejected", "socket_option_callbacks", "socket_option_failures"):
        integer(connections.get(name), f"connections.{name}")
    require(connections["closed"] <= connections["accepted"] and
            connections["active"] == connections["accepted"] - connections["closed"], "Connection counters disagree")
    require(connections["socket_option_failures"] == 0 and connections["reused_rejected"] == 0,
            "Socket setup or ownership failed during ordinary independent requests")
    transport = value["transport"]
    require(transport.get("stream_limit_bytes") == STREAM_BYTES and transport.get("frame_limit_bytes") == FRAME_BYTES,
            "Unexpected server transport budgets")
    require(integer(transport.get("peak_handoff_bytes"), "peak_handoff_bytes") <= FRAME_BYTES, "Handoff exceeded its frame budget")
    recent = value.get("recent_requests")
    require(isinstance(recent, list) and len(recent) <= 64, "Invalid bounded recent-request records")
    for item in recent:
        require(item.get("status") in ("completed", "disconnect"), f"Unexpected retired request: {item}")
        require(integer(item.get("peak_pending_bytes"), "request.peak_pending_bytes") <= STREAM_BYTES,
                "Request exceeded its transport budget")


def validate_final_report(report, model_id, backend, expected_requests, expected_decode_batch_size=None):
    require(isinstance(report, dict), "Server did not write its final report")
    require(report.get("model") == model_id and report.get("backend") == backend, "Final model/backend mismatch")
    integer(report.get("max_decode_batch_size"), "max_decode_batch_size")
    if expected_decode_batch_size is not None:
        require(report["max_decode_batch_size"] == expected_decode_batch_size,
                "Final report decode batch bound differs from the requested configuration")
    require(report.get("status") == "closed" and report["ipc"].get("status") == "closed", "Worker domain did not close")
    require(report["shutdown"]["graceful"] is True and report["shutdown"]["idle"] is True, "Shutdown barriers failed")
    require(report["active_requests"] == 0 and report["ipc"]["outstanding_requests"] == 0 and
            report["ipc"]["pending_commands"] == 0, "Final request or IPC reservations did not reach zero")
    sockets = report["connections"]
    require(sockets["active"] == 0 and sockets["accepted"] == sockets["closed"], "Final connections did not reach zero")
    counts = report["counters"]
    require(counts["admitted"] == expected_requests and counts["admitted"] == counts["completed"] + counts["cancelled"],
            "Final admitted/completed/cancelled counts disagree with recorded requests")
    require(counts["failed"] == 0 and counts["rejected"] == 0 and counts["slow_client_cancellations"] == 0,
            "Unexpected failed/rejected/stalled requests")


class HealthClient:
    def __init__(self, args, port, log):
        self.args, self.port, self.log = args, port, log
        self.epoch = None
        self.lock = threading.Lock()

    def get(self, label, allow_starting=False):
        # Serializing probes avoids the monitor's connection confusing the
        # main thread's idle observation. This is not an inference/RPC lock.
        with self.lock:
            started = time.perf_counter_ns()
            connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=self.args.health_timeout)
            try:
                connection.request("GET", "/health", headers={"Connection": "close", "Accept": "application/json"})
                response = connection.getresponse()
                body = response.read(BODY_BYTES + 1)
                elapsed = (time.perf_counter_ns() - started) / 1e9
                require(response.status in ((200, 503) if allow_starting else (200,)), f"Health HTTP {response.status}: {body[:500]!r}")
                require(len(body) <= BODY_BYTES, "Health response exceeded its expected bound")
                require((response.getheader("Content-Type") or "").lower().startswith("application/json"), "Health is not JSON")
                value = strict_json(body)
                self.log.append({"observed_ns": time.perf_counter_ns(), "label": label, "latency_seconds": elapsed, "health": value})
                validate_health(value, self.args.model_id, self.args.backend, allow_starting,
                                self.args.decode_batch_size)
                if self.epoch is None:
                    self.epoch = value["ipc"]["epoch"]
                require(value["ipc"]["epoch"] == self.epoch, "Worker epoch changed during the run")
                return value
            finally:
                connection.close()

    def idle(self, label, timeout=20):
        deadline = time.monotonic() + timeout
        while True:
            value = self.get(label)
            # /health is itself the one currently accepted connection. Only
            # the final report, written after the HTTP loops drain, can be zero.
            if (value["active_requests"] == 0 and value["ipc"]["outstanding_requests"] == 0 and
                    value["ipc"]["pending_commands"] == 0 and value["connections"]["active"] <= 1):
                return value
            require(time.monotonic() < deadline, f"Request/connection/IPC cleanup timed out: {value}")
            time.sleep(0.025)


class Sampler:
    def __init__(self, process, health, log, interval):
        self.process, self.health, self.log, self.interval = process, health, log, interval
        self.phase = "ready"
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="aila-service-sampler", daemon=True)
        self.error = None
        self.rows = []
        self.lock = threading.Lock()

    def snapshot(self, label):
        value = {"observed_ns": time.perf_counter_ns(), "phase": self.phase, "label": label,
                 "processes": self.process.sample()}
        self.log.append(value)
        with self.lock:
            self.rows.append(value)
        return value

    def _run(self):
        try:
            while not self.stop_event.wait(self.interval):
                self.health.get("periodic:" + self.phase)
                self.snapshot("periodic")
        except BaseException as error:
            self.error = f"{type(error).__name__}: {error}"

    def start(self):
        self.thread.start()

    def check(self):
        require(self.error is None, f"Periodic sampling failed: {self.error}")

    def stop(self):
        self.stop_event.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=self.health.args.health_timeout + 5)
            require(not self.thread.is_alive(), "Periodic sampler did not stop")
        self.check()


class RunWatchdog:
    """Bound an owned run, including startup and blocked request readers.

    A user-mode watchdog cannot prevent a kernel-driver bugcheck. This limits
    further work/owned processes when Windows is still responding.
    """

    def __init__(self, process, shutdown_file, directory, timeout, grace=5.0):
        self.process, self.shutdown_file, self.directory = process, shutdown_file, directory
        self.timeout, self.grace = timeout, grace
        self.stopped = threading.Event()
        self.expired = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, name="aila-run-watchdog", daemon=True)

    def _run(self):
        if self.stopped.wait(self.timeout):
            return
        self.expired.set()
        try:
            write_json(self.directory / "watchdog.json", {"expired_at": utc_now(), "timeout_seconds": self.timeout})
            self.shutdown_file.write_text("Run deadline reached.\n", encoding="utf-8")
            if not self.stopped.wait(self.grace) and (self.process.poll() is None or self.process.owned_pids()):
                pids = self.process.terminate_owned()
                write_json(self.directory / "watchdog-termination.json", {"at": utc_now(), "owned_pids": pids})
        except BaseException as error:
            self.error = f"{type(error).__name__}: {error}"
            # Diagnostic I/O failure must not disable containment.
            try:
                self.process.terminate_owned()
            except BaseException as stop_error:
                self.error += f"; termination: {stop_error}"

    def start(self):
        self.thread.start()

    def stop(self):
        self.stopped.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=self.grace + 10)
            require(not self.thread.is_alive(), "Run watchdog did not stop")

    def check(self):
        require(not self.expired.is_set(), "Whole-run deadline reached; no further batches may execute")
        require(self.error is None, f"Run watchdog failed: {self.error}")


class SocketRegistry:
    def __init__(self):
        self.lock = threading.Lock()
        self.values = set()

    def add(self, value):
        with self.lock:
            self.values.add(value)

    def remove(self, value):
        with self.lock:
            self.values.discard(value)

    def close_all(self):
        with self.lock:
            values = list(self.values)
        for value in values:
            try:
                value.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


class Runner:
    def __init__(self, args, directory, port, health, sampler, request_log, batch_log, idle_log):
        self.args, self.directory, self.port = args, directory, port
        self.health, self.sampler = health, sampler
        self.request_log, self.batch_log, self.idle_log = request_log, batch_log, idle_log
        self.responses = directory / "requests"
        self.responses.mkdir()
        self.registry = SocketRegistry()
        self.requests = []
        self.batches = []
        self.references = {}
        self.ids = set()
        self.next_id = 0
        self.lock = threading.Lock()
        self.prompts = {
            "short": args.prompt,
            "long": "Read these notes, then answer the final instruction.\n" + "\n".join(
                f"Note {index + 1}: {args.prompt}" for index in range(args.long_prompt_repeat)) +
                "\nFinal instruction: Give one concise conclusion from these notes.",
        }
        write_json(directory / "prompts.json", {key: {"text": value, "utf8_bytes": len(value.encode('utf-8'))}
                                               for key, value in self.prompts.items()})

    def payload(self, kind, lane, action):
        return {"model": self.args.model_id, "messages": [{"role": "user", "content": self.prompts[kind]}],
                "max_tokens": self.args.cancel_new_tokens if action == "disconnect" else self.args.new_tokens,
                "stream": True, "stream_options": {"include_usage": True}, "enable_thinking": self.args.enable_thinking,
                "temperature": self.args.temperature, "seed": self.args.seed + lane}

    def request(self, spec, barrier):
        path = self.responses / spec["label"]
        payload = self.payload(spec["prompt_kind"], spec["seed_lane"], spec["action"])
        body = encoded_json(payload)
        row = {**spec, "payload_sha256": hashlib.sha256(body).hexdigest(), "started_at": utc_now(), "outcome": "failed"}
        parser = SseParser(self.args.model_id, payload["max_tokens"], self.args.context, self.args.output_bytes)
        connection = raw = response = None
        start_ns = None
        try:
            write_json(path.with_suffix(".request.json"), payload)
            require(len(body) <= BODY_BYTES, "Configured prompt exceeds the HTTP body limit")
            barrier.wait(timeout=15)
            start_ns = time.perf_counter_ns()
            row["start_ns"] = start_ns
            deadline = time.monotonic() + self.args.request_timeout + 5
            connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=self.args.request_timeout + 5)
            connection.connect()
            raw = connection.sock
            self.registry.add(raw)
            connection.request("POST", "/v1/chat/completions", body=body, headers={
                "Content-Type": "application/json", "Accept": "text/event-stream", "Connection": "close"})
            response = connection.getresponse()
            row["headers_seconds"] = (time.perf_counter_ns() - start_ns) / 1e9
            row["http_status"] = response.status
            row["headers"] = response.getheaders()
            with path.with_suffix(".sse").open("xb") as artifact:
                if response.status != 200:
                    error = response.read(self.args.output_bytes + 1)
                    artifact.write(error)
                    raise BenchmarkError(f"Unexpected HTTP {response.status}: {error[:1000]!r}")
                require((response.getheader("Content-Type") or "").lower().startswith("text/event-stream"), "Missing SSE Content-Type")
                require((response.getheader("Transfer-Encoding") or "").lower() == "chunked", "SSE response is not chunked")
                require((response.getheader("Connection") or "").lower() == "close", "Server did not advertise connection ownership")
                require(response.getheader("Content-Encoding") is None, "Compressed SSE changes the expected measurement path")
                while True:
                    remaining = deadline - time.monotonic()
                    require(remaining > 0, "Request exceeded its wall-clock deadline")
                    raw.settimeout(remaining)
                    block = response.read1(8192)
                    observed_ns = time.perf_counter_ns()
                    if not block:
                        parser.finish()
                        row["eof_ns"] = observed_ns
                        row["end_ns"] = parser.done_ns
                        row["outcome"] = "completed"
                        break
                    artifact.write(block)
                    parser.feed(block, observed_ns)
                    if spec["action"] == "disconnect" and parser.content_events:
                        require(parser.finish_reason is None and parser.done_ns is None,
                                "Cancellation target finished before the intentional disconnect; increase --cancel-new-tokens")
                        raw.shutdown(socket.SHUT_RDWR)
                        row["end_ns"] = time.perf_counter_ns()
                        row["outcome"] = "disconnected"
                        row["disconnect_seconds"] = (row["end_ns"] - start_ns) / 1e9
                        break
            require(spec["action"] != "disconnect" or row["outcome"] == "disconnected", "Disconnect request completed without cancellation")
            header_id = response.getheader("X-Request-Id")
            require(not header_id or header_id == parser.identity, "HTTP and SSE request IDs differ")
        except BaseException as error:
            row["error"] = f"{type(error).__name__}: {error}"
            row["traceback"] = traceback.format_exc()
            raise
        finally:
            if raw is not None:
                self.registry.remove(raw)
            if response is not None:
                response.close()
            if connection is not None:
                connection.close()
            if start_ns is not None:
                row.update(parser.measurement(start_ns))
            if row["outcome"] == "completed":
                require(row["request_latency_seconds"] > 0, "Non-positive completion latency")
                row["completion_tokens_per_second"] = row["usage"]["completion_tokens"] / row["request_latency_seconds"]
            write_json(path.with_suffix(".measurement.json"), row)
            self.request_log.append(row)
            with self.lock:
                self.requests.append(row)
        return row

    @staticmethod
    def semantic(row):
        return {key: row[key] for key in ("content_sha256", "reasoning_sha256", "usage", "finish_reason")}

    def ensure_references(self, profiles, width):
        # Each exact prompt/seed payload runs alone before it is admitted to a
        # concurrent wave. A repeatable concurrency bug must never become its
        # own oracle, including when the caller did not select width one.
        kinds = ("short", "long") if "mixed" in profiles else ("short",)
        for seed_lane in range(width):
            for kind in kinds:
                key = hashlib.sha256(encoded_json(self.payload(kind, seed_lane, "complete"))).hexdigest()
                if key not in self.references:
                    self.batch("reference", kind, 1, 0, reference_seed_lane=seed_lane)

    def batch(self, phase, profile, width, round_index, lifecycle=False, request_offset=0,
              reference_seed_lane=None):
        if phase == "reference":
            require(width == 1 and not lifecycle and reference_seed_lane is not None and
                    0 <= reference_seed_lane < self.args.sessions and profile in ("short", "long"),
                    "Independent references must be complete single-request runs")
        else:
            require(reference_seed_lane is None, "Only independent reference runs override the seed lane")
        self.sampler.phase = phase
        self.sampler.check()
        before = self.health.idle("before:" + phase)
        spec = []
        for lane in range(width):
            kind = ("long" if (lane + round_index) % 2 else "short") if profile == "mixed" else profile
            action = "disconnect" if lifecycle and (request_offset + lane) % 2 else "complete"
            seed_lane = lane if reference_seed_lane is None else reference_seed_lane
            self.next_id += 1
            spec.append({"label": f"{self.next_id:06d}-{phase}-{profile}-w{width}-{action}", "phase": phase,
                          "profile": profile, "width": width, "round": round_index, "lane": lane, "seed_lane": seed_lane,
                         "prompt_kind": kind, "action": action})
        barrier = threading.Barrier(width)
        entry = {"phase": phase, "profile": profile, "width": width, "round": round_index, "passed": False,
                 "labels": [item["label"] for item in spec], "started_at": utc_now()}
        write_json(self.directory / "active-batch.json", {**entry, "state": "started"})
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=width)
        rows = []
        try:
            futures = [pool.submit(self.request, item, barrier) for item in spec]
            for future in concurrent.futures.as_completed(futures):
                rows.append(future.result())
                self.sampler.check()
            rows.sort(key=lambda item: item["lane"])
            require(width == 1 or max(item["start_ns"] for item in rows) < min(item["end_ns"] for item in rows),
                    "Offered concurrent requests did not overlap")
            after = self.health.idle("after:" + phase)
            completed = sum(item["action"] == "complete" for item in rows)
            cancelled = width - completed
            for name, expected in (("admitted", width), ("completed", completed), ("cancelled", cancelled),
                                   ("disconnect_cancellations", cancelled)):
                require(after["counters"][name] - before["counters"][name] == expected,
                        f"Batch {name} count differs from its planned outcomes")
            recent = {item["id"]: item for item in after["recent_requests"]}
            reference_labels = {}
            for row in rows:
                identity = row["id"]
                require(identity and identity not in self.ids, "Request ID was missing or reused")
                self.ids.add(identity)
                expected = "completed" if row["action"] == "complete" else "disconnect"
                require(identity in recent and recent[identity]["status"] == expected,
                        f"Request retirement was not {expected}: {identity}")
                if row["action"] == "complete":
                    key = row["payload_sha256"]
                    semantic = self.semantic(row)
                    if phase == "reference":
                        require(key not in self.references, "Independent reference payload was duplicated")
                        self.references[key] = {"label": row["label"], "semantic": semantic}
                    else:
                        require(key in self.references, "Completed request has no independent single-request reference")
                        require(semantic == self.references[key]["semantic"],
                                "Completed request output changed from its independent single-request reference")
                    reference_labels[row["label"]] = self.references[key]["label"]
            span = (max(item["end_ns"] for item in rows) - min(item["start_ns"] for item in rows)) / 1e9
            tokens = sum(item["usage"]["completion_tokens"] for item in rows if item["outcome"] == "completed")
            entry.update({"passed": True, "span_seconds": span, "completion_tokens": tokens,
                          "completion_tokens_per_second": tokens / span, "completed": completed,
                           "disconnected": cancelled, "partial_usage_due_to_disconnect": bool(cancelled),
                           "independent_reference": phase == "reference", "reference_labels": reference_labels,
                          "retired": [recent[item["id"]] for item in rows]})
            sample = self.sampler.snapshot("idle-after-batch")
            self.idle_log.append({"phase": phase, "profile": profile, "round": round_index,
                                  "recorded_requests": len(self.requests), "health": after, "memory": sample})
        except BaseException as error:
            entry["error"] = f"{type(error).__name__}: {error}"
            self.registry.close_all()
            barrier.abort()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
            self.batch_log.append(entry)
            self.batches.append(entry)
            write_json(self.directory / "active-batch.json", {**entry, "state": "finished", "finished_at": utc_now()})
        print(json.dumps({"phase": phase, "profile": profile, "concurrency": width,
                          "round": round_index, "completed": entry["completed"], "disconnected": entry["disconnected"]}), flush=True)
        return entry

    def benchmark(self):
        self.ensure_references(self.args.profiles, max(self.args.concurrency))
        for profile in self.args.profiles:
            for width in self.args.concurrency:
                for index in range(self.args.warmup):
                    self.batch("warmup", profile, width, index)
                for index in range(self.args.rounds):
                    self.batch("measured", profile, width, index)

    def probe(self):
        # Two short serial requests, optionally separated by one disconnect.
        # Reuse the batch's retirement/counter/reference checks without entering
        # the long lifecycle loop. The cancelled response is never an oracle.
        self.ensure_references(("short",), 1)
        if self.args.probe_cancel:
            self.batch("probe_cancel", "short", 1, 0, lifecycle=True, request_offset=1)
        self.batch("probe_after_cancel" if self.args.probe_cancel else "probe", "short", 1, 0)

    def lifecycle(self):
        width = self.args.lifecycle_width
        self.ensure_references(("mixed",), width)
        for index in range(self.args.warmup):
            self.batch("lifecycle_warmup", "mixed", width, index, True, index * width)
        self.health.idle("lifecycle-baseline")
        before = self.sampler.snapshot("lifecycle-baseline")
        offset = 0
        rounds = 0
        while offset < self.args.lifecycle_requests:
            count = min(width, self.args.lifecycle_requests - offset)
            self.batch("lifecycle", "mixed", count, rounds, True, offset)
            offset += count
            rounds += 1
        self.health.idle("lifecycle-final")
        after = self.sampler.snapshot("lifecycle-final")
        growth = memory_growth(before, after)
        for item in growth.values():
            if self.args.max_private_growth_mib is not None:
                require(item["private_bytes_delta"] <= self.args.max_private_growth_mib * 1024 * 1024,
                        "Lifecycle private-byte growth exceeded the explicitly configured gate")
            if self.args.max_handle_growth is not None:
                require(item["handles_delta"] <= self.args.max_handle_growth,
                        "Lifecycle handle growth exceeded the explicitly configured gate")
        return {"requests": offset, "rounds": rounds, "width": width,
                "completed": (offset + 1) // 2, "disconnected": offset // 2,
                "baseline": before, "final": after, "growth": growth,
                "max_private_growth_mib": self.args.max_private_growth_mib, "max_handle_growth": self.args.max_handle_growth}


def memory_growth(before, after):
    previous = {item["role"]: item for item in before["processes"]}
    result = {}
    for item in after["processes"]:
        old = previous[item["role"]]
        require((item["pid"], item["creation_filetime"]) == (old["pid"], old["creation_filetime"]),
                "Process identity changed between memory samples")
        result[item["role"]] = {key + "_delta": item[key] - old[key] for key in ("rss_bytes", "private_bytes", "handles")}
    return result


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backend", choices=tuple(MODEL_IDS), default="qwen3_nf4")
    parser.add_argument("--model-id")
    parser.add_argument("--output", type=Path, required=True, help="Parent directory; every run creates a unique child")
    parser.add_argument("--mode", choices=("probe", "benchmark", "lifecycle", "both"), default="probe")
    parser.add_argument("--execution-trace", action="store_true",
                        help="Probe only: persist C++ execution/wait/free boundaries; changes wall time, not a performance run")
    parser.add_argument("--probe-cancel", action="store_true",
                        help="Trace probe only: reference, one disconnect after first content, then reference-checked reuse")
    parser.add_argument("--allow-stress", action="store_true", help="Explicitly select non-probe work; does not clear a crash quarantine")
    parser.add_argument("--context", type=int)
    parser.add_argument("--sessions", type=int)
    parser.add_argument("--decode-batch-size", type=int,
                        help="Opt-in continuous decode batch bound 1..4, never above --sessions; reported as a "
                             "configuration ceiling, not a measured batch width")
    parser.add_argument("--concurrency", type=int, nargs="+", choices=(1, 2, 3, 4))
    parser.add_argument("--profiles", nargs="+", choices=("short", "mixed"))
    parser.add_argument("--rounds", type=int, help="Measured rounds per profile/concurrency")
    parser.add_argument("--warmup", type=int, default=2, help="Excluded warmup rounds per configuration; at least two")
    prompts = parser.add_mutually_exclusive_group()
    prompts.add_argument("--prompt", help="Short prompt and repeated long-prompt note")
    prompts.add_argument("--prompt-file", type=Path, help="UTF-8 text replacing --prompt")
    parser.add_argument("--long-prompt-repeat", type=int, default=32, help="Text repetitions, not an estimated token count")
    parser.add_argument("--new-tokens", type=int)
    parser.add_argument("--cancel-new-tokens", type=int,
                        help="Disconnect target limit; default 16 for --probe-cancel, otherwise 128")
    parser.add_argument("--lifecycle-requests", type=int, default=512,
                        help="Measured lifecycle requests; must exceed 256, excluding references/warmup")
    parser.add_argument("--lifecycle-width", type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--health-timeout", type=float, default=2.0)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--startup-timeout", type=float, default=300.0)
    parser.add_argument("--shutdown-timeout", type=float, default=45.0)
    parser.add_argument("--output-bytes", type=int, default=1048576)
    parser.add_argument("--max-private-growth-mib", type=float, help="Optional per-process lifecycle idle growth gate")
    parser.add_argument("--max-handle-growth", type=int, help="Optional per-process lifecycle idle growth gate")
    parser.add_argument("--max-run-seconds", type=float, help="Whole-run watchdog including startup; default 60 for probe, 300 for stress")
    parser.add_argument("--max-device-bytes", type=int, help="Explicit USM ceiling; default 4 GiB for probe, 8 GiB for stress")
    parser.add_argument("--max-pinned-host-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--max-session-host-bytes", type=int, default=256 * 1024 * 1024)
    args = parser.parse_args(argv)

    def option(condition, message):
        if not condition:
            parser.error(message)

    probe = args.mode == "probe"
    option(not args.execution_trace or probe, "--execution-trace is limited to the short probe mode")
    option(not args.probe_cancel or (probe and args.execution_trace),
           "--probe-cancel requires --mode probe --execution-trace")
    option(probe or args.allow_stress, "Non-probe work requires --allow-stress and a healthy GPU run guard")
    defaults = {"context": 512 if probe else 2048, "sessions": 1 if probe else 4, "decode_batch_size": 1,
                "concurrency": [1] if probe else [1, 2, 4], "profiles": ["short"] if probe else ["short", "mixed"],
                "rounds": 1 if probe else 5, "new_tokens": 8 if probe else 24,
                "cancel_new_tokens": 16 if args.probe_cancel else 128,
                "max_run_seconds": 60.0 if probe else 300.0,
                "max_device_bytes": (4 if probe else 8) * 1024 ** 3}
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if probe:
        option(args.context <= 512 and args.sessions == 1 and args.concurrency == [1] and
               args.profiles == ["short"] and args.new_tokens <= 16 and args.rounds == 1,
               "Probe requires one short stream, context <= 512, <= 16 generated tokens and one round")
        option(args.max_run_seconds <= 60, "Probe whole-run deadline cannot exceed 60 seconds")
    if args.probe_cancel:
        option(args.new_tokens < args.cancel_new_tokens <= 16,
               "Cancellation probe requires new-tokens < cancel-new-tokens <= 16")
    option(math.isfinite(args.max_run_seconds) and 1 <= args.max_run_seconds <= 3600,
           "--max-run-seconds must be finite in 1..3600")
    for name in ("max_device_bytes", "max_pinned_host_bytes", "max_session_host_bytes"):
        option(1 <= getattr(args, name) <= (1 << 64) - 1, f"--{name.replace('_', '-')} must be a positive uint64 ceiling")
    option(32 <= args.context <= 4096, "--context must be 32..4096")
    option(1 <= args.sessions <= 4, "--sessions must be 1..4")
    option(1 <= args.decode_batch_size <= 4, "--decode-batch-size must be 1..4")
    option(args.decode_batch_size <= args.sessions, "--decode-batch-size cannot exceed --sessions")
    option(len(args.concurrency) == len(set(args.concurrency)) and max(args.concurrency) <= args.sessions,
           "--concurrency must be unique and fit --sessions")
    option(len(args.profiles) == len(set(args.profiles)), "--profiles must not repeat")
    option(args.rounds >= 1 and args.warmup >= 2, "--rounds must be positive and --warmup at least two")
    option(args.long_prompt_repeat >= 2, "--long-prompt-repeat must be at least two")
    option(1 <= args.new_tokens <= 1024 and args.new_tokens < args.context, "--new-tokens must fit 1..1024 and --context")
    option(1 <= args.cancel_new_tokens <= 1024 and (args.mode in ("probe", "benchmark") or args.cancel_new_tokens < args.context),
           "--cancel-new-tokens must fit 1..1024 and --context")
    option(args.lifecycle_requests >= 1, "--lifecycle-requests must be positive")
    option(args.mode in ("probe", "benchmark") or args.lifecycle_requests > RETAINED_REQUEST_HANDLES,
           f"--lifecycle-requests must exceed {RETAINED_REQUEST_HANDLES} retained request handles; references/warmup are additional")
    if args.lifecycle_width is None:
        args.lifecycle_width = max(width for width in (1, 2, 3, 4) if width <= args.sessions)
    option(args.lifecycle_width <= args.sessions, "--lifecycle-width exceeds --sessions")
    option(0 <= args.seed <= (1 << 64) - 4, "--seed must leave room for four independent uint64 request seeds")
    option(math.isfinite(args.temperature) and 0 <= args.temperature <= 100, "--temperature must be finite and 0..100")
    option(math.isfinite(args.sample_interval) and args.sample_interval >= 0.05, "--sample-interval must be finite and at least 0.05 seconds")
    for name in ("health_timeout", "request_timeout", "startup_timeout", "shutdown_timeout"):
        option(math.isfinite(getattr(args, name)) and getattr(args, name) > 0, f"--{name.replace('_', '-')} must be finite and positive")
    option(1 <= args.request_timeout <= 600, "--request-timeout must be 1..600 seconds")
    option(4096 <= args.output_bytes <= 4194304, "--output-bytes must be 4096..4194304")
    option(args.max_private_growth_mib is None or (math.isfinite(args.max_private_growth_mib) and args.max_private_growth_mib >= 0),
           "--max-private-growth-mib must be finite and nonnegative")
    option(args.max_handle_growth is None or args.max_handle_growth >= 0, "--max-handle-growth must be nonnegative")
    if args.model_id is None:
        args.model_id = MODEL_IDS[args.backend]
    option(re.fullmatch(r"[A-Za-z0-9._/-]{1,128}", args.model_id), "Invalid --model-id")
    for name in ("server", "worker", "model", "output", "prompt_file"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if args.prompt_file:
        args.prompt = args.prompt_file.read_text(encoding="utf-8", errors="strict")
    elif args.prompt is None:
        args.prompt = DEFAULT_PROMPT
    option(bool(args.prompt.strip()), "The prompt must not be empty")
    return args


def child_environment(args, directory):
    # Never let an inherited diagnostic path mix this run with old evidence or
    # silently add disk flushes to performance measurements.
    environment = {key: value for key, value in os.environ.items()
                   if key.casefold() != "aila_execution_trace_dir"}
    environment.update(AILA_KV_QUANT="0", AILA_VLM_KV_QUANT="0")
    if args.execution_trace:
        trace_directory = directory / "execution-trace"
        trace_directory.mkdir()  # New run, refuse to reuse an existing trace directory.
        environment["AILA_EXECUTION_TRACE_DIR"] = str(trace_directory.resolve())
    return environment


def execution_trace_summary(directory, worker_pid):
    summary = {"complete": False, "expected_worker_pid": worker_pid, "files": [], "errors": [],
               "scope": "Diagnostic host boundaries; backend_forward end is submission return, not GPU completion"}
    trace_directory = directory / "execution-trace"
    if type(worker_pid) is not int or worker_pid <= 0:
        summary["errors"].append("No verified Worker PID for execution trace attribution")
    if not trace_directory.is_dir():
        summary["errors"].append("Execution trace directory is missing")
        return summary
    paths = sorted(trace_directory.iterdir())
    if not paths or len(paths) > 16:
        summary["errors"].append("Expected 1..16 execution trace files")
        return summary
    trace_ids = set()
    for path in paths:
        if not path.is_file() or path.is_symlink() or not re.fullmatch(r"execution-[0-9-]+\.jsonl", path.name):
            summary["errors"].append(f"Unexpected execution trace entry: {path.name}")
            continue
        report = analyze_execution_trace(path)
        summary["files"].append(report)
        if not report["complete"]:
            summary["errors"].append(f"Incomplete or invalid execution trace: {path.name}")
        if report["pid"] != worker_pid:
            summary["errors"].append(f"Execution trace Worker PID mismatch: {path.name}")
        trace_id = report["trace_id"]
        if trace_id in trace_ids or trace_id != path.stem:
            summary["errors"].append(f"Execution trace identity mismatch/duplicate: {path.name}")
        trace_ids.add(trace_id)
    summary["complete"] = bool(summary["files"]) and not summary["errors"]
    return summary


def metadata(args, command, environment_overrides=None):
    files = {}
    candidates = [Path(__file__).resolve(), Path(__file__).resolve().parent.parent / "diagnostics" / "execution_trace.py",
                  args.server, args.worker, args.worker.parent / "build_info.json",
                  args.server.parent / "server-build-info.json", args.model / "config.json",
                  args.model / "tokenizer_config.json", args.model / "model.safetensors.index.json"]
    for path in candidates:
        if path.is_file():
            files[str(path)] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
    checkpoint = [{"name": path.name, "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
                  for path in sorted(args.model.iterdir()) if path.is_file()]
    keys = ("AILA_KV_QUANT", "AILA_VLM_KV_QUANT", "ONEAPI_DEVICE_SELECTOR", "SYCL_DEVICE_FILTER",
            "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS", "SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE",
            "DNNL_VERBOSE", "ONEDNN_VERBOSE", "AILA_EXECUTION_TRACE_DIR")
    return {
        "created_at": utc_now(), "python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
        "command": command, "file_identity": files, "checkpoint_files": checkpoint,
        "checkpoint_identity_scope": "Weights listed by path/size/mtime; weight-file contents are not hashed by this runner.",
        "inherited_selected_environment": {key: os.environ[key] for key in keys if key in os.environ},
        "child_environment_overrides": environment_overrides or {"AILA_KV_QUANT": "0", "AILA_VLM_KV_QUANT": "0",
                                                                "AILA_EXECUTION_TRACE_DIR": None},
        "execution_trace": {"enabled": args.execution_trace, "diagnostic_only": True,
                            "null_environment_override_means_removed": True},
        "clock": {"name": "perf_counter_ns", "resolution_seconds": time.get_clock_info("perf_counter").resolution},
        "definitions": {
            "ttft_seconds": "From the request's post-barrier start, including connect/write, to its first nonempty content or reasoning_content SSE event; role is excluded.",
            "content_event_intervals_seconds": "Between client read observations completing consecutive nonempty content/reasoning SSE events. Not token latency; events in one read share a timestamp and can have zero gaps.",
            "request_latency_seconds": "From request start to the successful [DONE] event; EOF time is retained separately.",
            "batch_completion_tokens_per_second": "Sum of validated completion_tokens divided by first request start to last [DONE] in that measured batch; no startup, warmup or between-batch idle probes.",
            "references": "Each exact prompt/seed payload first runs in an isolated width-one batch. All subsequent completed requests, including lifecycle warmup and post-cancellation runs, must match that independent reference. Reference requests are excluded from performance statistics.",
            "warmup": "At least two explicit rounds of every selected width/profile precede measurement, warming both mixed prompt variants. References and warmup are excluded from performance statistics; finite warmup does not prove that every future runtime path is free from JIT.",
            "lifecycle_request_floor": f"Measured lifecycle requests alone must exceed the current IPC retained-request-handle hard limit of {RETAINED_REQUEST_HANDLES}; final zero outstanding requests does not expose retained terminal handles.",
            "percentiles": "Linear interpolation at (count - 1) * p; counts are supplied, and small runs do not establish tail-latency guarantees.",
            "memory": "Windows working set (RSS), PROCESS_MEMORY_COUNTERS_EX.PrivateUsage and GetProcessHandleCount for retained Server/Worker handles; no GPU, kernel TCP or total allocator accounting.",
            "sampling": "Periodic HTTP health and Windows process probes remain enabled during measurement; their small observer cost is included.",
            "client_instrumentation": "Network observations include client scheduling/parsing and any delay from saving raw SSE bodies between reads. Raw .sse artifacts are decoded HTTP entity bodies, with HTTP headers saved in measurements.",
            "logs": "stdout/stderr framework and recognizable service error prefixes fail the run. IPC diagnostic log frames discarded by InferenceClient are not exposed; request terminals and health faults are checked independently.",
        },
    }


def read_ready(directory):
    path = directory / "stdout.log"
    with path.open(encoding="utf-8", errors="replace") as source:
        for line in source:
            if line.startswith("{"):
                try:
                    item = strict_json(line)
                except (ValueError, BenchmarkError):
                    continue
                if isinstance(item, dict) and item.get("event") == "ready":
                    return item
    raise BenchmarkError("No ready identity record was found in Server stdout")


def finalize_run(args, directory, result, *, shutdown_file, report_file, process=None,
                 runner=None, sampler=None, watchdog=None, outputs=(), writers=(), guard=None):
    """Contain owned work before diagnostics; commit the guard only after saving them."""
    failure = (None, None, None)

    def record_failure(stage, error):
        nonlocal failure
        if failure[0] is None:
            failure = (type(error), error, error.__traceback__)
        result["passed"] = False
        result.setdefault("finalization_errors", []).append(
            {"stage": stage, "error": f"{type(error).__name__}: {error}"})

    def attempt(stage, action):
        try:
            action()
            return True
        except BaseException as error:
            record_failure(stage, error)
            return False

    def stop_process():
        cleanup = stop_owned_process(process, shutdown_file, args.shutdown_timeout)
        result["cleanup"] = cleanup
        if cleanup["forced_termination"] or not cleanup["job_empty"] or cleanup.get("exit_code") != 0 or cleanup["errors"]:
            result["passed"] = False

    def request_summary():
        result["request_count"] = len(runner.requests)
        result["batch_count"] = len(runner.batches)
        result["phase_request_counts"] = {phase: sum(item["phase"] == phase for item in runner.requests)
                                          for phase in sorted({item["phase"] for item in runner.requests})}

    def memory_summary():
        result["memory_distributions"] = {role: {
            key: distribution(process_row[key] for sample in sampler.rows for process_row in sample["processes"]
                              if process_row["role"] == role)
            for key in ("rss_bytes", "private_bytes", "handles")
        } for role in ("server", "worker")}

    def check_logs():
        result["log_errors"] = log_errors(directory)
        if result["log_errors"]:
            result["passed"] = False

    def check_report():
        report = strict_json(report_file.read_bytes())
        result["final_report"] = report
        validate_final_report(report, args.model_id, args.backend, len(runner.requests) if runner else 0,
                              args.decode_batch_size)

    def check_execution_trace():
        summary = execution_trace_summary(directory, getattr(process, "worker_pid", None))
        result["execution_trace"] = summary
        require(summary["complete"], "C++ execution trace did not complete with valid Worker identity")

    try:
        try:
            if runner is not None:
                attempt("request_shutdown", runner.registry.close_all)
            if sampler is not None:
                attempt("sampler_stop", sampler.stop)
            if watchdog is not None:
                attempt("watchdog_stop", watchdog.stop)
                attempt("watchdog_check", watchdog.check)
        finally:
            if process is not None and not attempt("process_shutdown", stop_process):
                # Keep Job kill-on-close reachable even if the cleanup helper fails.
                attempt("job_close", process.close)
        for index, output in enumerate(outputs):
            if output is not None:
                attempt(f"output_close_{index}", output.close)
        for index, writer in enumerate(writers):
            attempt(f"writer_close_{index}", writer.close)
        if runner is not None:
            attempt("request_summary", request_summary)
        if sampler is not None:
            attempt("memory_summary", memory_summary)
        attempt("log_check", check_logs)
        if process is not None:
            attempt("final_report", check_report)
            if getattr(args, "execution_trace", False):
                attempt("execution_trace", check_execution_trace)
        result["finished_at"] = utc_now()
        if guard is not None:
            # This report is accepted only with a matching finished guard record
            # and a zero runner exit. A crash between the two writes stays blocked.
            result["gpu_guard_run_id"] = guard.run_id
        saved = attempt("result_write", lambda: write_json(directory / "result.json", result))
        if guard is not None:
            attempt("gpu_guard_finish", lambda: guard.finish(success=result["passed"], details={
                "output": str(directory), "cleanup": result.get("cleanup"),
                "result_sha256": file_hash(directory / "result.json") if saved else None}))
        # Never require another write after committing a successful guard record:
        # simultaneous result/journal I/O failures could otherwise leave it healthy.
        if failure[0] is not None:
            attempt("failed_result_write", lambda: write_json(directory / "result.json", result))
    finally:
        escaping = sys.exc_info()
        if escaping[0] is not None:
            record_failure("finalization", escaping[1])
        if guard is not None:
            try:
                guard.__exit__(*failure)
            except BaseException as error:
                if not failure[0] and guard.finished and guard.record["status"] == "finished":
                    # Successful commit is the acceptance boundary. A later lock
                    # release error cannot undo its durable report/clean snapshot.
                    # Report it on stdout; do not require a post-commit rollback.
                    result.setdefault("post_commit_errors", []).append(
                        {"stage": "gpu_guard_close", "error": f"{type(error).__name__}: {error}"})
                else:
                    record_failure("gpu_guard_close", error)
                    attempt("failed_result_write", lambda: write_json(directory / "result.json", result))


def main(argv=None):
    args = parse_arguments(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
    directory = args.output / f"run-{stamp}-{uuid.uuid4().hex[:12]}"
    directory.mkdir()
    arguments = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    write_json(directory / "arguments.json", arguments)
    result = {"schema_version": 1, "started_at": utc_now(), "passed": False, "output": str(directory),
              "backend": args.backend, "model": str(args.model), "mode": args.mode, "measured": [], "lifecycle": None}
    process = sampler = runner = stdout = stderr = watchdog = guard = None
    guard_entered = False
    writers = []
    shutdown_file = directory / f"shutdown-{uuid.uuid4().hex}.flag"
    report_file = directory / "final-report.json"
    try:
        require(args.server.is_file() and args.worker.is_file() and args.model.is_dir(), "Server/Worker/model paths must exist")
        guard = GpuRunGuard("aila-http-service-" + args.mode,
                            metadata={"output": str(directory), "arguments": arguments})
        guard.__enter__()
        guard_entered = True
        result["gpu_guard_state"] = str(guard.state_path)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        maximum = (max(args.new_tokens, args.cancel_new_tokens)
                   if args.probe_cancel or args.mode in ("lifecycle", "both") else args.new_tokens)
        command = [str(args.server), "--worker", str(args.worker), "--model", str(args.model), "--backend", args.backend,
                   "--model-id", args.model_id, "--port", str(port), "--context", str(args.context), "--sessions", str(args.sessions),
                   "--decode-batch-size", str(args.decode_batch_size),
                   "--max-new-tokens", str(maximum), "--max-output-bytes", str(args.output_bytes),
                   "--stream-buffer-bytes", str(STREAM_BYTES), "--stream-stall-ms", "5000",
                   "--request-timeout-ms", str(round(args.request_timeout * 1000)),
                   "--max-device-bytes", str(args.max_device_bytes),
                   "--max-pinned-host-bytes", str(args.max_pinned_host_bytes),
                   "--max-session-host-bytes", str(args.max_session_host_bytes),
                   "--shutdown-file", str(shutdown_file), "--report", str(report_file)]
        environment = child_environment(args, directory)
        info = metadata(args, command, {key: environment.get(key) for key in
                        ("AILA_KV_QUANT", "AILA_VLM_KV_QUANT", "AILA_EXECUTION_TRACE_DIR")})
        write_json(directory / "metadata.json", info)
        stdout, stderr = (directory / "stdout.log").open("xb"), (directory / "stderr.log").open("xb")
        startup_ns = time.perf_counter_ns()
        process = WindowsJobProcess(command, directory, stdout, stderr, environment)
        watchdog = RunWatchdog(process, shutdown_file, directory, args.max_run_seconds)
        watchdog.start()
        for name in ("health", "memory", "measurements", "batches", "idle"):
            writers.append(JsonLines(directory / (name + ".jsonl")))
        health_log, memory_log, request_log, batch_log, idle_log = writers
        health = HealthClient(args, port, health_log)
        deadline = time.monotonic() + args.startup_timeout
        while True:
            watchdog.check()
            require(process.poll() is None, f"Server exited before readiness: {process.returncode}")
            try:
                ready_health = health.get("startup", allow_starting=True)
                if ready_health["status"] == "ok" and process.find_worker(args.worker) is not None:
                    break
            except (ConnectionError, OSError, http.client.HTTPException):
                pass
            require(time.monotonic() < deadline, "Server readiness/owned Worker discovery timed out")
            time.sleep(0.1)
        ready = read_ready(directory)
        require(ready.get("epoch") == health.epoch and ready.get("backend") == args.backend and ready.get("model") == args.model_id,
                "Ready log disagrees with health identity")
        require(ready.get("max_decode_batch_size") == args.decode_batch_size,
                "Ready log decode batch bound differs from the requested configuration")
        result["startup"] = {"seconds": (time.perf_counter_ns() - startup_ns) / 1e9, "excluded_from_performance": True,
                             "server_pid": process.pid, "worker_pid": process.worker_pid, "ready": ready}
        info["process_identity"] = {"server_pid": process.pid, "worker_pid": process.worker_pid,
                                    "worker_image": str(process.worker_image), "initial_samples": process.sample(),
                                    "owned_job_pids_at_readiness": process.owned_pids()}
        info["ready"] = ready
        write_json(directory / "metadata.json", info)
        print(json.dumps({"event": "ready", "output": str(directory), **result["startup"]}), flush=True)
        sampler = Sampler(process, health, memory_log, args.sample_interval)
        sampler.snapshot("ready")
        sampler.start()
        runner = Runner(args, directory, port, health, sampler, request_log, batch_log, idle_log)
        if args.mode == "probe":
            runner.probe()
            result["probe"] = {"requests": 3 if args.probe_cancel else 2, "concurrency": 1,
                               "cancellation_and_reuse_covered": args.probe_cancel,
                               "long_lifecycle_covered": False, "performance_measurement": False}
        if args.mode in ("benchmark", "both"):
            runner.benchmark()
            result["measured"] = summarize_measurements(runner.requests, runner.batches)
        if args.mode in ("lifecycle", "both"):
            result["lifecycle"] = runner.lifecycle()
        result["pre_shutdown_health"] = health.idle("pre-shutdown")
        sampler.snapshot("pre-shutdown")
        sampler.check()
        watchdog.check()
        result["passed"] = True
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
        print(result["error"], file=sys.stderr, flush=True)
    finally:
        finalize_run(args, directory, result, shutdown_file=shutdown_file, report_file=report_file,
                     process=process, runner=runner, sampler=sampler, watchdog=watchdog,
                     outputs=(stdout, stderr), writers=writers, guard=guard if guard_entered else None)
    print(json.dumps({"passed": result["passed"], "output": str(directory),
                      "requests": result.get("request_count", 0), "mode": args.mode,
                      "finalization_errors": result.get("finalization_errors", []),
                      "post_commit_errors": result.get("post_commit_errors", [])}), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
