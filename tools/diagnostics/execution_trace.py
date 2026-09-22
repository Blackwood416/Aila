"""Offline reader for Aila execution trace schema 1; never starts a workload.

Every physical record, including its LF (or CRLF), is limited to 1536 bytes.
Files are read incrementally, with a 64 MiB limit. The original evidence is
never changed: valid_prefix_bytes/sha256 identify every accepted record before
the first invalid or unterminated row without copying the whole trace into the
report. An end with status=error is a valid event, but prevents a successful
result even when cleanup and trace_close subsequently complete.

"complete" describes the observed, stable, logically closed trace. The caller
must independently establish that its owned process has exited. An unfinished
scope identifies missing end evidence, not the cause of a crash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys


MAX_RECORD_BYTES = 1536
MAX_FILE_BYTES = 64 * 1024 * 1024
READ_CHUNK_BYTES = 64 * 1024
MAX_ERROR_DETAILS = 64
# Directory mode has explicit aggregate limits as well as the per-file limit.
MAX_DIRECTORY_FILES = 256
MAX_DIRECTORY_BYTES = 256 * 1024 * 1024

OPERATIONS = frozenset((
    "context_initialize", "context_release", "device_shutdown", "model_load",
    "model_destroy", "session_create", "session_destroy", "session_reset",
    "session_truncate", "forward", "backend_forward", "sampling", "logits_read",
    "queue_wait", "copy_h2d_wait", "copy_d2h_wait", "workspace_release", "usm_free", "decode_batch",
))
INTEGER_MINIMUMS = {
    "schema_version": 1, "seq": 1, "pid": 1, "tid": 1, "utc_ns": 1,
    "steady_ns": 0, "scope_id": 0, "parent_scope_id": 0, "model_id": 0,
    "session_id": 0, "token_count": -1, "position": -1,
    "target_position": -1, "bytes": 0,
}
# Match the signed clocks/counts and unsigned identifiers in the C++ encoder.
# Python integers stay exact above 2**53, but values outside the writer's
# representation are corrupt evidence rather than larger valid identifiers.
INTEGER_MAXIMUMS = {
    key: (2**63 - 1 if key in ("utc_ns", "steady_ns", "token_count", "position", "target_position")
          else 2**64 - 1)
    for key in INTEGER_MINIMUMS
}
STRING_FIELDS = frozenset(("trace_id", "event", "operation", "status", "phase", "memory"))
FIELDS = frozenset(INTEGER_MINIMUMS) | STRING_FIELDS
END_MUTABLE_FIELDS = frozenset(("seq", "utc_ns", "steady_ns", "event", "status"))
ROOT_FIELDS = {
    "scope_id": 0, "parent_scope_id": 0, "operation": "trace", "status": "ok",
    "model_id": 0, "session_id": 0, "phase": "none", "token_count": -1,
    "position": -1, "target_position": -1, "bytes": 0, "memory": "none",
}


class _InvalidRecord(ValueError):
    def __init__(self, code, message, **details):
        super().__init__(message)
        self.code = code
        self.details = details


def _reject(code, message, **details):
    raise _InvalidRecord(code, message, **details)


def _unique_members(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _reject("duplicate_key", "Duplicate JSON object member.", field=key)
        result[key] = value
    return result


def _constant(value):
    _reject("invalid_json_constant", "Non-finite JSON constants are not permitted.", value=value)


def _decode_record(raw):
    if b"\0" in raw:
        _reject("nul_byte", "A record contains a NUL byte.")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        _reject("invalid_utf8", "A complete record is not valid UTF-8.")
    try:
        row = json.loads(text, object_pairs_hook=_unique_members, parse_constant=_constant)
    except _InvalidRecord:
        raise
    except (ValueError, RecursionError):
        _reject("invalid_json", "A complete record is not valid JSON.")
    if type(row) is not dict:
        _reject("invalid_record_type", "Each record must be a JSON object.")
    missing, unknown = FIELDS - row.keys(), row.keys() - FIELDS
    if missing or unknown:
        _reject("invalid_fields", "Record fields differ from schema 1.",
                missing=sorted(missing), unknown=sorted(unknown))
    for key, minimum in INTEGER_MINIMUMS.items():
        # bool is an int subclass; floats also lose exact identifiers.
        maximum = INTEGER_MAXIMUMS[key]
        if type(row[key]) is not int or not minimum <= row[key] <= maximum:
            _reject("invalid_integer", "An integer field has the wrong type or is outside the encoder's range.",
                    field=key, minimum=minimum, maximum=maximum)
    if row["schema_version"] != 1:
        _reject("unsupported_schema", "Only schema_version=1 is supported.")
    for key in STRING_FIELDS:
        if type(row[key]) is not str:
            _reject("invalid_string", "A string field has the wrong type.", field=key)
        if "\0" in row[key]:
            _reject("nul_character", "A string contains an escaped NUL character.", field=key)
        try:
            row[key].encode("utf-8", errors="strict")
        except UnicodeEncodeError:
            _reject("invalid_unicode", "A string contains an unpaired Unicode surrogate.", field=key)
    if not row["trace_id"]:
        _reject("empty_trace_id", "trace_id must not be empty.")
    if row["event"] not in ("trace_open", "begin", "end", "trace_close"):
        _reject("invalid_event", "Unknown event name.")
    if row["phase"] not in ("none", "prefill", "decode"):
        _reject("invalid_phase", "Unknown execution phase.")
    if row["memory"] not in ("none", "device", "pinned_host"):
        _reject("invalid_memory", "Unknown memory kind.")
    if row["event"] in ("trace_open", "trace_close"):
        changed = [key for key, value in ROOT_FIELDS.items() if row[key] != value]
        if changed:
            _reject("invalid_trace_boundary", "Trace boundaries must use neutral scope fields.", fields=changed)
    else:
        if row["scope_id"] == 0:
            _reject("invalid_scope_id", "begin/end require a positive scope_id.")
        if row["operation"] not in OPERATIONS:
            _reject("invalid_operation", "Unknown execution operation.")
        statuses = ("ok", "error") if row["event"] == "end" else ("ok",)
        if row["status"] not in statuses:
            _reject("invalid_status", "Status is not permitted for this event.")
    return row


class _Analysis:
    def __init__(self, path):
        self.result = {
            "path": str(path), "complete": False, "state": "incomplete",
            "pid": None, "trace_id": None, "sha256": None, "bytes": 0,
            "hash_complete": False, "bytes_read": 0, "stable_snapshot": False,
            "record_count": 0, "last_record": None, "open_scopes": [],
            "trace_open": False, "trace_close": False, "scope_error_count": 0,
            "valid_prefix_bytes": 0, "valid_prefix_sha256": hashlib.sha256(b"").hexdigest(),
            "tail": None, "errors": [], "error_count": 0, "errors_truncated": False,
            "interpretation": "Unclosed scopes locate missing end evidence; the last begin does not establish crash causality or process liveness.",
        }
        self.prefix_hash = hashlib.sha256()
        self.scopes = {}
        self.seen_scopes = set()
        self.stacks = {}
        self.invalid = False

    def error(self, code, message, *, invalid=False, **details):
        self.invalid |= invalid
        self.result["error_count"] += 1
        if len(self.result["errors"]) < MAX_ERROR_DETAILS:
            self.result["errors"].append({"code": code, "message": message, **details})
        else:
            self.result["errors_truncated"] = True

    def accept(self, row, raw, line, offset):
        result = self.result
        expected = result["record_count"] + 1
        if row["seq"] != expected:
            _reject("sequence_gap" if row["seq"] > expected else "sequence_reuse_or_reorder",
                    "seq must start at 1 and remain contiguous.", expected=expected, actual=row["seq"])
        previous = result["last_record"]
        if previous and row["steady_ns"] < previous["steady_ns"]:
            _reject("steady_clock_regressed", "steady_ns decreased in file order.")
        if result["trace_open"]:
            if row["trace_id"] != result["trace_id"] or row["pid"] != result["pid"]:
                _reject("trace_identity_changed", "trace_id and pid must remain constant.")
        if result["trace_close"]:
            _reject("record_after_trace_close", "trace_close must be the last record.")
        event, scope, tid = row["event"], row["scope_id"], row["tid"]
        stack = self.stacks.get(tid, [])
        if not result["trace_open"]:
            if event != "trace_open":
                _reject("missing_initial_trace_open", "The first record must be trace_open.")
        elif event == "trace_open":
            _reject("duplicate_trace_open", "trace_open may appear only once.")
        elif event == "begin":
            if scope in self.seen_scopes:
                _reject("scope_id_reused", "A scope ID was reused within the trace.", scope_id=scope)
            expected_parent = stack[-1] if stack else 0
            if row["parent_scope_id"] != expected_parent:
                _reject("parent_mismatch", "begin parent must be the last unended scope on the same thread.",
                        expected=expected_parent, actual=row["parent_scope_id"])
        elif event == "end":
            begin = self.scopes.get(scope)
            if begin is None:
                _reject("unknown_end_scope", "end has no matching live begin.", scope_id=scope)
            if tid != begin["tid"]:
                _reject("end_thread_mismatch", "end must use the begin thread.", scope_id=scope)
            if not stack or stack[-1] != scope:
                _reject("end_order_mismatch", "Scopes must end in LIFO order on each thread.", scope_id=scope)
            changed = sorted(key for key in FIELDS - END_MUTABLE_FIELDS if row[key] != begin[key])
            if changed:
                _reject("end_fields_mismatch", "end identity and fields must match begin.", fields=changed)
        elif event == "trace_close" and self.scopes:
            _reject("trace_close_with_open_scopes", "trace_close appeared while scopes remain open.", count=len(self.scopes))

        # Commit only after all row checks succeed, preserving the valid prefix
        # and live stacks if a malformed end follows a valid begin.
        if event == "trace_open":
            result.update(trace_open=True, pid=row["pid"], trace_id=row["trace_id"])
        elif event == "begin":
            self.seen_scopes.add(scope)
            self.scopes[scope] = {**row, "begin_line": line, "begin_byte_offset": offset}
            self.stacks.setdefault(tid, []).append(scope)
        elif event == "end":
            del self.scopes[scope]
            self.stacks[tid].pop()
            if not self.stacks[tid]:
                del self.stacks[tid]
            if row["status"] == "error":
                result["scope_error_count"] += 1
                self.error("scope_error", "A scope ended with status=error.", line=line,
                           byte_offset=offset, scope_id=scope, operation=row["operation"], tid=tid)
        else:
            result["trace_close"] = True
        result["record_count"] += 1
        result["last_record"] = row
        result["valid_prefix_bytes"] = offset + len(raw)
        self.prefix_hash.update(raw)

    def finish(self, inspected=True):
        result = self.result
        result["valid_prefix_sha256"] = self.prefix_hash.hexdigest()
        result["open_scopes"] = sorted(self.scopes.values(), key=lambda row: row["seq"])
        result["open_scope_count"] = len(self.scopes)
        if inspected:
            if not result["trace_open"]:
                self.error("missing_trace_open", "No valid trace_open was recorded.")
            if not result["trace_close"]:
                self.error("missing_trace_close", "No valid trace_close was recorded; evidence is incomplete.")
            if self.scopes:
                self.error("unclosed_scopes", "Some recorded begins have no matching valid end.", count=len(self.scopes))
        result["complete"] = (not self.invalid and result["stable_snapshot"] and result["hash_complete"] and
                              result["trace_open"] and result["trace_close"] and not self.scopes and
                              result["tail"] is None and result["scope_error_count"] == 0)
        if result["complete"]:
            result["state"] = "complete"
        elif self.invalid:
            result["state"] = "invalid"
        elif result["tail"] is not None or not result["trace_close"] or self.scopes:
            result["state"] = "incomplete"
        else:
            result["state"] = "error"
        return result


def _identity(info):
    return info.st_dev, info.st_ino


def _snapshot(info):
    return *_identity(info), info.st_size, info.st_mtime_ns


def analyze_file(path) -> dict:
    """Analyze a stable snapshot without modifying the file or starting work.

Oversized/unreadable/changing files have sha256=None and hash_complete=False.
For any stable file within the limit, sha256 covers *all* raw bytes, including
invalid records or a partial tail. Validation stops at the first bad physical
record; its preceding prefix remains identified by byte count and SHA-256.
"""
    path = Path(path).absolute()
    analysis = _Analysis(path)
    result = analysis.result
    digest = hashlib.sha256()
    reached_eof = False
    opened = ended = None
    prefix_active = True
    line = 1
    try:
        initial = path.stat()
        result["bytes"] = initial.st_size
        if not stat.S_ISREG(initial.st_mode):
            analysis.error("not_regular_file", "Input must be a regular file.", invalid=True)
            return analysis.finish(inspected=False)
        if initial.st_size > MAX_FILE_BYTES:
            analysis.error("file_too_large", "Input exceeds the 64 MiB file limit; full hash was not attempted.",
                           invalid=True, limit=MAX_FILE_BYTES)
            return analysis.finish(inspected=False)
        with path.open("rb") as source:
            opened = os.fstat(source.fileno())
            if not stat.S_ISREG(opened.st_mode) or _identity(initial) != _identity(opened):
                analysis.error("file_changed", "Input identity changed before reading.", invalid=True)
                return analysis.finish(inspected=False)
            if opened.st_size > MAX_FILE_BYTES:
                result["bytes"] = opened.st_size
                analysis.error("file_too_large", "Input grew beyond the file limit before reading.",
                               invalid=True, limit=MAX_FILE_BYTES)
                return analysis.finish(inspected=False)
            while True:
                offset = result["bytes_read"]
                remaining = MAX_FILE_BYTES + 1 - offset
                # readline is bounded even when a hostile file has no newline.
                block = (source.readline(min(MAX_RECORD_BYTES + 1, remaining)) if prefix_active else
                         source.read(min(READ_CHUNK_BYTES, remaining)))
                if not block:
                    reached_eof = True
                    break
                digest.update(block)
                result["bytes_read"] += len(block)
                if result["bytes_read"] > MAX_FILE_BYTES:
                    analysis.error("file_too_large", "Input grew beyond the file limit while reading.",
                                   invalid=True, limit=MAX_FILE_BYTES)
                    break
                if not prefix_active:
                    continue
                if len(block) > MAX_RECORD_BYTES:
                    analysis.error("record_too_large", f"A physical record exceeds {MAX_RECORD_BYTES} bytes including its newline.",
                                   invalid=True, line=line, byte_offset=offset, limit=MAX_RECORD_BYTES)
                    prefix_active = False
                elif not block.endswith(b"\n"):
                    result["tail"] = {"byte_offset": offset, "bytes": len(block),
                                      "sha256": hashlib.sha256(block).hexdigest(), "contains_nul": b"\0" in block}
                    analysis.error("incomplete_tail", "Final bytes lack a record newline and were not accepted.",
                                   line=line, byte_offset=offset)
                    prefix_active = False
                else:
                    try:
                        analysis.accept(_decode_record(block), block, line, offset)
                    except _InvalidRecord as error:
                        analysis.error(error.code, str(error), invalid=True, line=line,
                                       byte_offset=offset, **error.details)
                        prefix_active = False
                    line += 1
            ended = os.fstat(source.fileno())
        current = path.stat()
        result["bytes"] = ended.st_size
        result["stable_snapshot"] = (_snapshot(opened) == _snapshot(ended) == _snapshot(current) and
                                     result["bytes_read"] == ended.st_size)
        if not result["stable_snapshot"]:
            analysis.error("file_changed", "File identity, size, or modification time changed while reading.", invalid=True)
    except (OSError, ValueError) as error:
        analysis.error("io_error", "Could not read a stable file snapshot.", invalid=True,
                       exception=type(error).__name__)
    result["observed_sha256"] = digest.hexdigest()
    result["hash_complete"] = reached_eof and result["stable_snapshot"]
    if result["hash_complete"]:
        result["sha256"] = digest.hexdigest()
    return analysis.finish(inspected=opened is not None)


def analyze_directory(path) -> dict:
    """Analyze immediate execution-*.jsonl children, with bounded aggregation."""
    directory = Path(path).absolute()
    result = {"path": str(directory), "complete": False, "files": [], "errors": [], "file_count": 0}
    candidates, total = [], 0
    try:
        if not directory.is_dir():
            result["errors"].append({"code": "not_directory", "message": "Input is not a directory."})
            return result
        for candidate in directory.glob("execution-*.jsonl"):
            if len(candidates) == MAX_DIRECTORY_FILES:
                result["errors"].append({"code": "directory_file_limit", "message": "Directory exceeds the trace file count limit.", "limit": MAX_DIRECTORY_FILES})
                break
            size = candidate.stat().st_size
            if total + size > MAX_DIRECTORY_BYTES:
                result["errors"].append({"code": "directory_byte_limit", "message": "Directory exceeds the aggregate input byte limit.", "limit": MAX_DIRECTORY_BYTES})
                break
            total += size
            candidates.append(candidate)
    except OSError as error:
        result["errors"].append({"code": "directory_io_error", "message": type(error).__name__})
    if not candidates and not result["errors"]:
        result["errors"].append({"code": "no_traces", "message": "No execution-*.jsonl traces were found."})
    identities = set()
    for candidate in sorted(candidates):
        report = analyze_file(candidate)
        result["files"].append(report)
        identity = report["trace_id"]
        if identity is not None and identity in identities:
            result["errors"].append({"code": "duplicate_trace_id", "message": "Multiple files use the same trace_id.", "trace_id": identity})
        identities.add(identity)
    result["file_count"] = len(result["files"])
    result["complete"] = bool(result["files"]) and not result["errors"] and all(item["complete"] for item in result["files"])
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="One trace file or a directory of execution-*.jsonl files")
    parser.add_argument("--output", type=Path, help="Create a new JSON report; existing files are never overwritten")
    args = parser.parse_args(argv)
    report = analyze_directory(args.path) if args.path.is_dir() else analyze_file(args.path)
    output_error = None
    if args.output is not None:
        try:
            # Exclusive creation also refuses existing hard links and symlinks.
            with args.output.open("x", encoding="utf-8", newline="\n") as destination:
                json.dump(report, destination, ensure_ascii=True, allow_nan=False, indent=2)
                destination.write("\n")
        except OSError as error:
            output_error = {"code": "output_not_created", "message": type(error).__name__, "path": str(args.output)}
    if output_error is not None:
        report = {**report, "output_error": output_error}
    print(json.dumps(report, ensure_ascii=True, allow_nan=False, indent=2))
    return 2 if output_error is not None else 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
