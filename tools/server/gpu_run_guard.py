"""Fail-closed admission and durable evidence for cooperating Windows GPU runners.

This is a before/after check, not a watchdog or a guarantee against an OS crash.
The caller must stop its owned GPU processes before finish() and retain the guard
until that cleanup is complete. All runners must use the same default directory;
state_dir and snapshot_reader are injection points for CPU-only tests.

No GPU work, driver changes, TDR registry changes, or recovery runs occur here.
There is deliberately no command-line reset. acknowledge() requires the exact
pending evidence ID and an explicit review reason, and cannot acknowledge an
unreadable current snapshot. Acknowledgement is not permission to launch a test.
"""

from __future__ import annotations

import ctypes
import datetime
import hashlib
import json
import ntpath
import os
import platform
import re
import subprocess
import uuid
from pathlib import Path


SCHEMA_VERSION = 1


class GuardError(RuntimeError):
    pass


class GuardBusy(GuardError):
    """Another cooperating process still owns the machine's GPU-run lock."""


class GuardPersistenceError(GuardError):
    """Evidence cannot be durably read or written; no run may be admitted."""


class GuardBlocked(GuardError):
    def __init__(self, evidence):
        self.evidence = evidence
        self.evidence_id = evidence["id"]
        kinds = ", ".join(item["kind"] for item in evidence["reasons"])
        super().__init__(
            f"GPU run blocked: {kinds}; evidence_id={self.evidence_id}. "
            "Review the evidence before any explicit acknowledgement; no automatic retry."
        )


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def default_state_dir():
    """Machine scope, independent of checkout, current directory, or output."""
    if os.name != "nt" or not os.environ.get("PROGRAMDATA"):
        raise GuardPersistenceError("Windows PROGRAMDATA is required for the GPU-run guard")
    directory = Path(os.environ["PROGRAMDATA"])
    if not directory.is_absolute():
        raise GuardPersistenceError("PROGRAMDATA must be an absolute machine-level path")
    return directory / "Aila" / "gpu-run-guard"


class ProcessFileLock:
    """OS-owned byte/file lock; PIDs and leftover lock files never grant admission."""

    def __init__(self, path):
        self.path = Path(path)
        self.file = None

    def acquire(self):
        if self.file is not None:
            raise GuardError("This lock is already held")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            stream = self.path.open("a+b", buffering=0)
            if stream.seek(0, os.SEEK_END) == 0:
                stream.write(b"\0")
                stream.flush()
                os.fsync(stream.fileno())
            stream.seek(0)
        except OSError as error:
            if "stream" in locals():
                stream.close()
            raise GuardPersistenceError(f"Cannot open GPU-run lock {self.path}: {error}") from error
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:  # Allows real interprocess-lock CPU tests on non-Windows hosts.
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            stream.close()
            raise GuardBusy(f"GPU-run lock is held or unavailable: {self.path}: {error}") from error
        self.file = stream
        return self

    def close(self):
        stream, self.file = self.file, None
        if stream is not None:
            # Closing the handle releases the kernel lock even after process death.
            stream.close()

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *_):
        self.close()


def _replace_write_through(source, destination):
    if os.name == "nt":
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        move = kernel.MoveFileExW
        move.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD]
        move.restype = wintypes.BOOL
        # MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH.
        if not move(str(source), str(destination), 0x1 | 0x8):
            raise ctypes.WinError(ctypes.get_last_error())
    else:
        os.replace(source, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def _atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(_json_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        _replace_write_through(temporary, path)
    finally:
        # A failed replacement leaves the previous state and durable journal intact.
        if temporary.exists():
            temporary.unlink()


def _append_jsonl(path, value):
    with path.open("ab") as stream:
        stream.write(_json_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _strict_json(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError(f"Duplicate JSON key: {key}")
            value[key] = item
        return value

    def constant(value):
        raise ValueError(f"Invalid JSON constant: {value}")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


class _Store:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.state_path = self.directory / "state.json"
        self.journal_path = self.directory / "journal.jsonl"
        self.record = None
        self.write_failed = False

    def load(self):
        try:
            has_state, has_journal = self.state_path.exists(), self.journal_path.exists()
            if not has_state and not has_journal:
                return None
            if has_state != has_journal:
                raise ValueError("State/journal pair is incomplete")
            raw = self.journal_path.read_bytes()
            if not raw or not raw.endswith(b"\n"):
                raise ValueError("Empty or incomplete journal")
            previous = None
            for number, line in enumerate(raw.splitlines(), 1):
                record = _strict_json(line)
                payload = record["payload"]
                if (record["sequence"] != number or
                        record["previous"] != (previous["digest"] if previous else None) or
                        record["digest"] != _digest({key: record[key] for key in
                                                     ("sequence", "previous", "payload")}) or
                        payload["schema_version"] != SCHEMA_VERSION or
                        payload["status"] not in ("started", "finished", "failed", "blocked", "acknowledged")):
                    raise ValueError(f"Invalid journal record {number}")
                previous = record
            if _strict_json(self.state_path.read_bytes()) != previous:
                raise ValueError("Atomic state and durable journal disagree")
            self.record = previous
            return previous["payload"]
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise GuardPersistenceError(
                f"GPU guard evidence is unreadable or incomplete in {self.directory}: {error}. "
                "Preserve these files for review; no automatic repair or launch."
            ) from error

    def commit(self, payload):
        if self.write_failed:
            raise GuardPersistenceError("A previous durable guard write failed; no further writes are allowed")
        payload = {**payload, "schema_version": SCHEMA_VERSION, "recorded_at": _utc_now()}
        record = {"sequence": self.record["sequence"] + 1 if self.record else 1,
                  "previous": self.record["digest"] if self.record else None,
                  "payload": payload}
        record["digest"] = _digest(record)
        try:
            # No work is admitted until both writes succeed. An interruption between
            # them is deliberately a hard stop rather than an old healthy baseline.
            _append_jsonl(self.journal_path, record)
            _atomic_json(self.state_path, record)
        except (OSError, ValueError, TypeError) as error:
            self.write_failed = True
            raise GuardPersistenceError(
                f"Cannot durably record GPU guard state in {self.directory}: {error}"
            ) from error
        self.record = record
        return payload


# These are read-only Windows queries. Event data, not localized rendered messages,
# supplies bugcheck/LiveKernelEvent codes. Empty query results and access failures
# are distinct. The finite event window is recorded rather than claiming full history.
_WINDOWS_QUERY = r"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$failures = [System.Collections.Generic.List[object]]::new()
$events = [System.Collections.Generic.List[object]]::new()
$heads = @{}
$boot = $null
try {
    $boot = (Get-CimInstance Win32_OperatingSystem -ErrorAction Stop).LastBootUpTime.ToUniversalTime().ToString('o')
} catch { $failures.Add(@{source='boot'; error=$_.Exception.Message}) }
foreach ($log in @('System', 'Application')) {
    try {
        $last = @(Get-WinEvent -LogName $log -MaxEvents 1 -ErrorAction Stop)
        $heads[$log] = if ($last.Count) { [long]$last[0].RecordId } else { 0 }
    } catch {
        if ($_.FullyQualifiedErrorId -like 'NoMatchingEventsFound*') { $heads[$log] = 0 }
        else { $failures.Add(@{source="event_head/$log"; error=$_.Exception.Message}) }
    }
    $query = if ($log -eq 'System') {
        "*[System[(EventID=1001) or (EventID=4101 and Provider[@Name='Display']) or (EventID=41 and Provider[@Name='Microsoft-Windows-Kernel-Power']) or (EventID=6008 and Provider[@Name='EventLog'])]]"
    } else {
        "*[System[EventID=1001] and EventData[Data[@Name='EventName']='LiveKernelEvent' or Data[@Name='EventName']='BlueScreen']]"
    }
    try {
        $records = @(Get-WinEvent -LogName $log -FilterXPath $query -MaxEvents 256 -ErrorAction Stop)
        foreach ($entry in $records) {
            $xml = [xml]$entry.ToXml()
            $data = @{}
            $index = 0
            foreach ($item in $xml.Event.EventData.Data) {
                $key = if ($item.Name) { [string]$item.Name } else { "param$index" }
                $data[$key] = [string]$item.'#text'
                $index++
            }
            $events.Add(@{log=$log; record_id=[long]$entry.RecordId;
                event_id=[int]$entry.Id; provider=[string]$entry.ProviderName;
                time_created=$entry.TimeCreated.ToUniversalTime().ToString('o'); data=$data})
        }
    } catch {
        if ($_.FullyQualifiedErrorId -notlike 'NoMatchingEventsFound*') {
            $failures.Add(@{source="events/$log"; error=$_.Exception.Message})
        }
    }
}
@{boot_id=$boot; events=@($events.ToArray()); event_log_heads=$heads;
  read_errors=@($failures.ToArray()); event_window_per_log=256} | ConvertTo-Json -Depth 12 -Compress
"""


def _classify_event(event):
    event = dict(event)
    data = event.get("data", {})
    event_id = event.get("event_id")
    if event_id == 4101:
        event["signal"] = "display_driver_recovered"
    elif event_id == 41:
        event["signal"] = "unexpected_power_loss"
    elif event_id == 6008:
        event["signal"] = "unexpected_shutdown"
    elif data.get("EventName") == "LiveKernelEvent":
        event["signal"] = "live_kernel_event"
        event["live_kernel_code"] = data.get("P1")
    else:
        event["signal"] = "bugcheck_or_system_error"
    for value in data.values():
        match = re.search(r"\b0x([0-9a-fA-F]{3,8})\b", str(value))
        if match:
            event["bugcheck_code"] = hex(int(match.group(1), 16))
            break
    if event_id == 41 and str(data.get("BugcheckCode", "")).isdigit():
        event["bugcheck_code"] = hex(int(data["BugcheckCode"]))
    if data.get("EventName") == "BlueScreen" and data.get("P1"):
        try:
            event["bugcheck_code"] = hex(int(str(data["P1"]), 16))
        except ValueError:
            pass
    return event


def _dump_metadata(path, records, errors):
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_symlink():
                    errors.append({"source": str(entry.path), "error": "Reparse/symlink metadata not followed"})
                elif entry.is_dir(follow_symlinks=False):
                    _dump_metadata(Path(entry.path), records, errors)
                elif entry.name.lower().endswith(".dmp"):
                    info = entry.stat(follow_symlinks=False)
                    records.append({"path": str(Path(entry.path).absolute()),
                                    "size": info.st_size, "mtime_ns": info.st_mtime_ns})
    except FileNotFoundError:
        # No dump directory is observable and is different from access denied.
        return
    except OSError as error:
        errors.append({"source": str(path), "error": f"{type(error).__name__}: {error}"})


def collect_windows_crash_snapshot():
    """Read boot/event/dump metadata only; failures remain visible and block use."""
    snapshot = {"schema_version": SCHEMA_VERSION, "captured_at": _utc_now(),
                "host": platform.node(), "boot_id": None, "events": [], "dumps": [],
                "event_log_heads": {}, "read_errors": []}
    if os.name != "nt":
        snapshot["read_errors"].append({"source": "platform", "error": "Windows evidence collection required"})
        return snapshot
    windows = Path(os.environ.get("SYSTEMROOT", r"C:\Windows"))
    snapshot["dump_roots"] = [str(windows / name) for name in ("LiveKernelReports", "Minidump")]
    snapshot["full_dump_path"] = str(windows / "MEMORY.DMP")
    executable = windows / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    try:
        completed = subprocess.run(
            [str(executable), "-NoProfile", "-NonInteractive", "-Command", _WINDOWS_QUERY],
            capture_output=True, encoding="utf-8-sig", errors="strict", timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW, check=False)
        if completed.returncode:
            raise RuntimeError(f"PowerShell exit {completed.returncode}: {completed.stderr.strip()}")
        payload = _strict_json(completed.stdout)
        for name in ("boot_id", "events", "event_log_heads", "read_errors"):
            snapshot[name] = payload[name]
        snapshot["event_window_per_log"] = payload["event_window_per_log"]
        snapshot["events"] = [_classify_event(item) for item in snapshot["events"]]
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.TimeoutExpired) as error:
        snapshot["read_errors"].append({"source": "windows_events", "error": f"{type(error).__name__}: {error}"})
    for name in ("LiveKernelReports", "Minidump"):
        _dump_metadata(windows / name, snapshot["dumps"], snapshot["read_errors"])
    try:
        full_dump = windows / "MEMORY.DMP"
        info = full_dump.stat()
        snapshot["dumps"].append({"path": str(full_dump.absolute()), "size": info.st_size,
                                  "mtime_ns": info.st_mtime_ns})
    except FileNotFoundError:
        pass
    except OSError as error:
        snapshot["read_errors"].append({"source": str(full_dump), "error": f"{type(error).__name__}: {error}"})
    return snapshot


def _read_snapshot(reader):
    try:
        value = _strict_json(_json_bytes(reader()))
        if (not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION or
                not isinstance(value.get("host"), str) or not value["host"] or
                not isinstance(value.get("events"), list) or
                not isinstance(value.get("dumps"), list) or
                not isinstance(value.get("read_errors"), list) or
                not isinstance(value.get("event_log_heads"), dict) or
                any(not isinstance(item, dict) for name in ("events", "dumps", "read_errors")
                    for item in value[name]) or
                any(type(item) is not int or item < 0 for item in value["event_log_heads"].values())):
            raise ValueError("Malformed crash snapshot")
        if not isinstance(value.get("boot_id"), str) or not value["boot_id"]:
            value["read_errors"].append({"source": "boot_id", "error": "Boot identity is unavailable"})
        for name in ("System", "Application"):
            if name not in value["event_log_heads"]:
                value["read_errors"].append({"source": f"event_head/{name}", "error": "Log head is unavailable"})
        if ("dump_roots" in value or "full_dump_path" in value) and _dump_scan(value) is None:
            raise ValueError("Malformed dump scan coverage")
        return value
    except Exception as error:
        # Injected/provider errors cannot silently become an empty healthy baseline.
        return {"schema_version": SCHEMA_VERSION, "captured_at": _utc_now(),
                "host": platform.node(), "boot_id": None, "events": [], "dumps": [],
                "event_log_heads": {}, "read_errors": [
                    {"source": "snapshot_reader", "error": f"{type(error).__name__}: {error}"}]}


def _signals(snapshot):
    return {f"{kind}:{_digest(item)}": {"kind": kind, "value": item}
            for kind, name in (("event", "events"), ("dump", "dumps"))
            for item in snapshot[name]}


def _dump_path(value):
    if not isinstance(value, str) or not value or "\0" in value:
        return None
    if value.startswith("\\\\?\\"):
        value = value[4:]
    return ntpath.normcase(ntpath.normpath(value)) if ntpath.isabs(value) else None


def _wer_report(event):
    """Only dump-backed, uniquely identified Windows kernel reports qualify."""
    if (event.get("log") != "Application" or event.get("provider") != "Windows Error Reporting" or
            event.get("event_id") != 1001):
        return None
    data = event.get("data")
    if not isinstance(data, dict) or data.get("EventName") not in ("BlueScreen", "LiveKernelEvent"):
        return None
    report_id, attached = data.get("ReportId"), data.get("AttachedFiles")
    if (not isinstance(report_id, str) or not re.fullmatch(
            r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", report_id) or
            not int(report_id.replace("-", ""), 16) or not isinstance(attached, str)):
        return None
    dumps = []
    for line in attached.splitlines():
        value = line.strip().strip('"')
        if ntpath.splitext(value)[1].lower() == ".dmp":
            path = _dump_path(value)
            if path is None: return None
            dumps.append(path)
    if not dumps: return None
    # Preserve every EventData field, including report status and attachments.
    return _digest({key: event[key] for key in ("log", "provider", "event_id", "data")}), sorted(set(dumps))


def _dump_scan(snapshot):
    roots = snapshot.get("dump_roots")
    full = _dump_path(snapshot.get("full_dump_path"))
    if not isinstance(roots, list) or not roots or full is None:
        return None
    normalized = [_dump_path(path) for path in roots]
    if any(path is None for path in normalized): return None
    return tuple(sorted(set(normalized))), full


def _dump_scan_covers(scan, path):
    return bool(scan and (path == scan[1] or any(path.startswith(root.rstrip("\\") + "\\") for root in scan[0])))


def historical_wer_replays(before, current):
    """Identify retransmission of the same retained report, never a new crash.

    Raw events remain in snapshots. A new ReportId, changed payload/dump,
    incomplete scan coverage, system event or unreadable evidence is not a replay.
    This does not clear a pending failure or acknowledge a boot change.
    """
    if (before is None or before.get("read_errors") or current.get("read_errors") or
            before.get("host") != current.get("host")):
        return {}
    old_reports = {}
    for event in before["events"]:
        report = _wer_report(event)
        if report:
            old_reports.setdefault(report[0], []).append(event.get("record_id"))
    old_dumps = {_dump_path(row.get("path")): row for row in before["dumps"]}
    new_dumps = {_dump_path(row.get("path")): row for row in current["dumps"]}
    old_signals = _signals(before)
    old_scan, new_scan = _dump_scan(before), _dump_scan(current)
    unchanged_inventory = old_dumps == new_dumps
    replays = {}
    for event in current["events"]:
        key = "event:" + _digest(event)
        if key in old_signals: continue
        report = _wer_report(event)
        if not report or report[0] not in old_reports: continue
        present, absent = [], []
        for path in report[1]:
            if (path in old_dumps and path in new_dumps and old_dumps[path] == new_dumps[path] and
                    type(old_dumps[path].get("size")) is int and old_dumps[path]["size"] > 0 and
                    type(old_dumps[path].get("mtime_ns")) is int and old_dumps[path]["mtime_ns"] > 0):
                present.append(path)
            elif (path not in old_dumps and path not in new_dumps and unchanged_inventory and
                    old_scan is not None and old_scan == new_scan and _dump_scan_covers(old_scan, path)):
                # Complete scans can prove unchanged absence, not just an
                # unchanged file. Missing legacy coverage remains conservative.
                absent.append(path)
            else: break
        if len(present) + len(absent) != len(report[1]): continue
        replays[key] = {"record_id": event.get("record_id"), "time_created": event.get("time_created"),
            "report_id": event["data"]["ReportId"], "full_payload_sha256": report[0],
            "matched_records": old_reports[report[0]], "unchanged_dump_paths": present,
            "unchanged_absent_dump_paths": absent}
    return replays


def _snapshot_reasons(before, current):
    reasons = []
    if current["read_errors"]:
        reasons.append({"kind": "evidence_unreadable", "errors": current["read_errors"]})
    current_signals = _signals(current)
    if before is None:
        if current_signals:
            reasons.append({"kind": "initial_fault_evidence", "signals": current_signals})
        return reasons
    for name in ("host", "boot_id"):
        if current.get(name) != before.get(name):
            reasons.append({"kind": f"{name}_changed", "before": before.get(name), "after": current.get(name)})
    if _dump_scan(before) is not None and _dump_scan(before) != _dump_scan(current):
        reasons.append({"kind": "dump_scan_changed", "before": _dump_scan(before), "after": _dump_scan(current)})
    old_signals = _signals(before)
    replays = historical_wer_replays(before, current)
    added = {key: value for key, value in current_signals.items() if key not in old_signals and key not in replays}
    if added:
        reasons.append({"kind": "new_fault_evidence", "signals": added})
    for name, old_head in before["event_log_heads"].items():
        if name in current["event_log_heads"] and current["event_log_heads"][name] < old_head:
            reasons.append({"kind": "event_log_rewound", "log": name, "before": old_head,
                            "after": current["event_log_heads"][name]})
    return reasons


def _evidence(reasons):
    unique = {_digest(item): item for item in reasons}
    reasons = [unique[key] for key in sorted(unique)]
    return {"id": _digest(reasons), "reasons": reasons}


def _pending_reasons(previous, current):
    reasons = _snapshot_reasons(previous["snapshot"] if previous else None, current)
    if previous:
        if previous["status"] in ("blocked", "failed") and previous.get("evidence"):
            reasons += previous["evidence"]["reasons"]
        elif previous["status"] in ("started", "failed"):
            reasons.append({"kind": "unfinished_run" if previous["status"] == "started" else "previous_failed",
                            "run_id": previous["run_id"], "recorded_at": previous["recorded_at"]})
    return reasons


class GpuRunGuard:
    def __init__(self, label, *, metadata=None, state_dir=None, snapshot_reader=None):
        if not isinstance(label, str) or not label.strip():
            raise ValueError("A nonempty GPU-run label is required")
        self.directory = Path(state_dir) if state_dir is not None else default_state_dir()
        self.label = label
        self.metadata = _strict_json(_json_bytes(metadata or {}))
        self.reader = snapshot_reader or collect_windows_crash_snapshot
        self.lock = ProcessFileLock(self.directory / "gpu.lock")
        self.store = _Store(self.directory)
        self.run_id = uuid.uuid4().hex
        self.record = None
        self.finished = False
        self.entered = False

    @property
    def state_path(self):
        return self.store.state_path

    def __enter__(self):
        if self.entered:
            raise GuardError("A GpuRunGuard instance may be entered only once")
        self.entered = True
        self.lock.acquire()
        try:
            previous = self.store.load()
            snapshot = _read_snapshot(self.reader)
            reasons = _pending_reasons(previous, snapshot)
            base = {"run_id": self.run_id, "label": self.label, "metadata": self.metadata,
                    "pid": os.getpid(), "snapshot": snapshot}
            base["historical_wer_replays"] = [dict(row, phase="before_start") for row in
                historical_wer_replays(previous["snapshot"] if previous else None, snapshot).values()]
            if reasons:
                self.record = self.store.commit({**base, "status": "blocked", "evidence": _evidence(reasons)})
                raise GuardBlocked(self.record["evidence"])
            self.record = self.store.commit({**base, "status": "started"})
            return self
        except BaseException:
            self.lock.close()
            raise

    def finish(self, *, success=True, details=None):
        """Call only after all owned GPU work has stopped. Failed runs stay blocked."""
        if self.lock.file is None or self.record is None or self.finished:
            raise GuardError("No active unfinished GPU run")
        if type(success) is not bool:
            raise TypeError("success must be bool")
        snapshot = _read_snapshot(self.reader)
        reasons = _snapshot_reasons(self.record["snapshot"], snapshot)
        if not success:
            reasons.append({"kind": "run_failed", "run_id": self.run_id, "details": details})
        self.record = self.store.commit({
            **self.record, "status": "failed" if reasons else "finished",
            "before_snapshot": self.record.get("before_snapshot", self.record["snapshot"]), "snapshot": snapshot,
            "details": details,
            "historical_wer_replays": self.record.get("historical_wer_replays", []) +
                [dict(row, phase="during_run") for row in historical_wer_replays(self.record["snapshot"], snapshot).values()],
            **({"evidence": _evidence(reasons)} if reasons else {})})
        self.finished = True
        if success and reasons:
            raise GuardBlocked(self.record["evidence"])
        return self.record

    def __exit__(self, exc_type, exc, _traceback):
        try:
            if exc_type and self.finished and self.record["status"] == "finished":
                # An exception after finish() is still a failed guarded operation.
                self.finished = False
            if not self.finished and not self.store.write_failed:
                self.finish(success=False, details={
                    "error": f"{exc_type.__name__}: {exc}" if exc_type else "Explicit finish() was not called"})
                if exc_type is None:
                    raise GuardBlocked(self.record["evidence"])
        finally:
            self.lock.close()
        return False


def acknowledge(evidence_id, reason, *, state_dir=None, snapshot_reader=None):
    """Explicit reviewed-evidence API, never called automatically by a runner."""
    if not isinstance(evidence_id, str) or not re.fullmatch(r"[0-9a-f]{64}", evidence_id):
        raise ValueError("The exact pending evidence ID is required")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("An explicit review reason is required")
    directory = Path(state_dir) if state_dir is not None else default_state_dir()
    reader = snapshot_reader or collect_windows_crash_snapshot
    with ProcessFileLock(directory / "gpu.lock"):
        store = _Store(directory)
        previous = store.load()
        if (not previous or previous["status"] not in ("blocked", "failed") or
                not previous.get("evidence") or previous["evidence"]["id"] != evidence_id):
            raise GuardError("Acknowledgement does not match the pending evidence")
        snapshot = _read_snapshot(reader)
        current = _evidence(_pending_reasons(previous, snapshot))
        if current["id"] != evidence_id or snapshot["read_errors"]:
            store.commit({**previous, "status": "blocked", "snapshot": snapshot, "evidence": current})
            raise GuardBlocked(current)
        return store.commit({
            **previous, "status": "acknowledged", "snapshot": snapshot,
            "acknowledgement": {"evidence_id": evidence_id, "reason": reason.strip(), "at": _utc_now()}})
