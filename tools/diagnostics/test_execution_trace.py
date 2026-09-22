"""Pure CPU tests; these fixtures never create a Context or start a worker."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

if __package__:
    from . import execution_trace as trace
else:
    import execution_trace as trace


class Records:
    def __init__(self, pid=123, trace_id=None):
        self.pid = pid
        self.trace_id = trace_id or f"execution-{pid}-1788955200000000000-1"
        self.rows = []

    def add(self, event, **changes):
        sequence = len(self.rows) + 1
        row = {
            "schema_version": 1, "trace_id": self.trace_id, "seq": sequence,
            "pid": self.pid, "tid": 11, "utc_ns": 1788955200000000000 + sequence,
            "steady_ns": sequence * 100, "event": event, "scope_id": 0,
            "parent_scope_id": 0, "operation": "trace", "status": "ok",
            "model_id": 0, "session_id": 0, "phase": "none", "token_count": -1,
            "position": -1, "target_position": -1, "bytes": 0, "memory": "none",
        }
        row.update(changes)
        self.rows.append(row)
        return row

    def begin(self, scope, operation="forward", **changes):
        return self.add("begin", scope_id=scope, operation=operation, **changes)

    def end(self, begin, **changes):
        immutable = {key: value for key, value in begin.items() if key not in trace.END_MUTABLE_FIELDS}
        return self.add("end", **{**immutable, **changes})

    def data(self):
        return b"".join(encode(row) for row in self.rows)


def encode(row):
    return (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


class ExecutionTraceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def analyze(self, data, name="execution-test.jsonl"):
        path = self.directory / name
        path.write_bytes(data)
        report = trace.analyze_file(path)
        self.assertEqual(path.read_bytes(), data, "analyzer must preserve the evidence")
        return report

    def code(self, report, code):
        self.assertIn(code, [item["code"] for item in report["errors"]])

    def minimal(self, **changes):
        records = Records(**changes)
        records.add("trace_open")
        records.add("trace_close")
        return records

    def nested(self):
        records = Records()
        records.add("trace_open")
        parent = records.begin(1, model_id=10, session_id=20, phase="prefill", token_count=32, position=0, target_position=32)
        child = records.begin(2, "backend_forward", parent_scope_id=1, model_id=10, session_id=20,
                              phase="prefill", token_count=32, position=0, target_position=32)
        records.end(child)
        records.end(parent)
        records.add("trace_close")
        return records

    def test_minimal_closed_trace_and_full_raw_hash(self):
        records = self.minimal()
        data = records.data()
        report = self.analyze(data)
        self.assertTrue(report["complete"])
        self.assertTrue(report["trace_open"])
        self.assertTrue(report["trace_close"])
        self.assertTrue(report["stable_snapshot"])
        self.assertTrue(report["hash_complete"])
        self.assertEqual(report["sha256"], hashlib.sha256(data).hexdigest())
        self.assertEqual(report["valid_prefix_sha256"], report["sha256"])
        self.assertEqual(report["valid_prefix_bytes"], len(data))
        self.assertEqual(report["record_count"], 2)
        self.assertEqual(report["last_record"], records.rows[-1])
        self.assertEqual(report["open_scopes"], [])
        self.assertEqual(report["errors"], [])

    def test_boundary_clock_minimums_are_valid(self):
        records = self.minimal()
        for row in records.rows:
            row.update(utc_ns=1, steady_ns=0)
        report = self.analyze(records.data())
        self.assertTrue(report["complete"], report["errors"])
        self.assertEqual(report["last_record"]["utc_ns"], 1)
        self.assertEqual(report["last_record"]["steady_ns"], 0)

    def test_decode_batch_keeps_independent_row_sampling_identity(self):
        records = Records()
        records.add("trace_open")
        batch = records.begin(1, "decode_batch", model_id=10, phase="decode", token_count=3)
        backend = records.begin(2, "backend_forward", parent_scope_id=1,
                                model_id=10, phase="decode", token_count=3)
        records.end(backend)
        for index, position in enumerate((2, 17, 63)):
            sample = records.begin(3 + index, "sampling", parent_scope_id=1,
                                   model_id=10, session_id=20 + index, phase="decode", token_count=1,
                                   position=position)
            records.end(sample)
        records.end(batch)
        records.add("trace_close")
        report = self.analyze(records.data())
        self.assertTrue(report["complete"], report["errors"])
        self.assertEqual(report["errors"], [])

    def test_interleaved_threads_have_independent_lifo_stacks(self):
        records = Records()
        records.add("trace_open")
        a = records.begin(100, model_id=7)
        b = records.begin(200, "model_load", tid=22, model_id=8)
        child = records.begin(101, "queue_wait", parent_scope_id=100)
        records.end(b)
        b2 = records.begin(201, "usm_free", tid=22, bytes=4096, memory="device")
        records.end(child)
        records.end(a)
        records.end(b2)
        records.add("trace_close", tid=33)
        report = self.analyze(records.data())
        self.assertTrue(report["complete"], report["errors"])
        self.assertEqual(report["record_count"], 10)

    def test_complete_removes_logical_parent_before_object_destruction(self):
        records = Records()
        records.add("trace_open")
        parent = records.begin(10, "model_load")
        child = records.begin(11, "queue_wait", parent_scope_id=10)
        records.end(child)
        next_child = records.begin(12, "copy_h2d_wait", parent_scope_id=10)
        records.end(next_child)
        records.end(parent)
        independent = records.begin(13, "session_create", parent_scope_id=0)
        records.end(independent)
        records.add("trace_close")
        self.assertTrue(self.analyze(records.data())["complete"])

    def test_interrupted_process_retains_all_live_scopes_and_prefix(self):
        records = Records()
        records.add("trace_open")
        records.begin(5, model_id=9, session_id=10)
        records.begin(6, "queue_wait", parent_scope_id=5)
        records.begin(7, "model_load", tid=22)
        report = self.analyze(records.data())
        self.assertFalse(report["complete"])
        self.assertEqual(report["state"], "incomplete")
        self.assertEqual([row["scope_id"] for row in report["open_scopes"]], [5, 6, 7])
        self.assertEqual(report["open_scopes"][1]["begin_line"], 3)
        self.assertEqual(report["valid_prefix_bytes"], len(records.data()))
        self.assertTrue(report["hash_complete"])
        self.code(report, "missing_trace_close")
        self.code(report, "unclosed_scopes")
        self.assertIn("does not establish crash causality", report["interpretation"])

    def test_unterminated_tail_is_not_accepted_even_if_json_is_complete(self):
        for tail in (b'{"schema_version":1,"seq":', b"not-json", b"\xff", b"\0", encode(self.minimal().rows[-1]).rstrip(b"\n")):
            with self.subTest(tail=tail[:20]):
                prefix = encode(self.minimal().rows[0])
                data = prefix + tail
                report = self.analyze(data)
                self.assertFalse(report["complete"])
                self.assertEqual(report["state"], "incomplete")
                self.assertEqual(report["record_count"], 1)
                self.assertFalse(report["trace_close"])
                self.assertEqual(report["tail"]["bytes"], len(tail))
                self.assertEqual(report["valid_prefix_bytes"], len(prefix))
                self.assertEqual(report["sha256"], hashlib.sha256(data).hexdigest())
                self.code(report, "incomplete_tail")

    def test_invalid_complete_row_stops_prefix_but_hashes_remainder(self):
        records = self.minimal()
        prefix = encode(records.rows[0])
        data = prefix + b"not-json\n" + encode(records.rows[1])
        report = self.analyze(data)
        self.assertEqual(report["state"], "invalid")
        self.assertEqual(report["record_count"], 1)
        self.assertEqual(report["valid_prefix_bytes"], len(prefix))
        self.assertEqual(report["valid_prefix_sha256"], hashlib.sha256(prefix).hexdigest())
        self.assertEqual(report["sha256"], hashlib.sha256(data).hexdigest())
        self.assertFalse(report["trace_close"])
        self.code(report, "invalid_json")

    def test_error_scope_then_successful_cleanup_and_trace_close(self):
        records = Records()
        records.add("trace_open")
        forward = records.begin(1)
        records.end(forward, status="error")
        cleanup = records.begin(2, "workspace_release")
        records.end(cleanup)
        records.add("trace_close")
        report = self.analyze(records.data())
        self.assertFalse(report["complete"])
        self.assertEqual(report["state"], "error")
        self.assertTrue(report["trace_close"])
        self.assertEqual(report["scope_error_count"], 1)
        self.assertEqual(report["open_scopes"], [])
        self.assertEqual(report["record_count"], len(records.rows))
        self.code(report, "scope_error")

    def test_end_matches_every_immutable_scope_field(self):
        changes = {"operation": "sampling", "model_id": 11, "session_id": 21, "phase": "decode",
                   "token_count": 31, "position": 1, "target_position": 33, "bytes": 2,
                   "memory": "pinned_host", "parent_scope_id": 99}
        for key, value in changes.items():
            with self.subTest(field=key):
                records = self.nested()
                records.rows[3][key] = value
                report = self.analyze(records.data())
                self.assertFalse(report["complete"])
                self.assertEqual(report["record_count"], 3)
                self.assertEqual([row["scope_id"] for row in report["open_scopes"]], [1, 2])
                self.code(report, "end_fields_mismatch")

    def test_parent_cannot_reference_another_thread(self):
        records = Records()
        records.add("trace_open")
        records.begin(1)
        records.begin(2, tid=22, parent_scope_id=1)
        report = self.analyze(records.data())
        self.code(report, "parent_mismatch")
        self.assertEqual(report["open_scope_count"], 1)

    def test_nested_begin_must_name_current_live_parent(self):
        records = self.nested()
        records.rows[2]["parent_scope_id"] = 0
        self.code(self.analyze(records.data()), "parent_mismatch")

    def test_end_thread_and_lifo_mismatches(self):
        records = self.nested()
        records.rows[3]["tid"] = 22
        self.code(self.analyze(records.data()), "end_thread_mismatch")
        records = self.nested()
        wrong = {**records.rows[1], "event": "end", "seq": 4, "steady_ns": 400}
        records.rows[3] = wrong
        report = self.analyze(records.data())
        self.code(report, "end_order_mismatch")
        self.assertEqual(report["open_scope_count"], 2)

    def test_unknown_end_and_scope_id_reuse(self):
        records = self.nested()
        records.rows[3]["scope_id"] = 99
        self.code(self.analyze(records.data()), "unknown_end_scope")
        records = Records()
        records.add("trace_open")
        begin = records.begin(1)
        records.end(begin)
        records.begin(1)
        self.code(self.analyze(records.data()), "scope_id_reused")

    def test_sequence_missing_duplicate_and_reorder(self):
        for seq, code in ((4, "sequence_gap"), (1, "sequence_reuse_or_reorder")):
            records = self.minimal()
            records.rows[1]["seq"] = seq
            self.code(self.analyze(records.data()), code)
        records = self.minimal()
        records.rows[0]["seq"] = 2
        self.code(self.analyze(records.data()), "sequence_gap")

    def test_trace_identity_is_constant(self):
        for key, value in (("trace_id", "execution-123-2-1"), ("pid", 456)):
            records = self.nested()
            records.rows[2][key] = value
            self.code(self.analyze(records.data()), "trace_identity_changed")

    def test_steady_clock_is_nondecreasing_but_wall_clock_can_regress(self):
        records = self.minimal()
        records.rows[1]["utc_ns"] = 1
        records.rows[1]["steady_ns"] = records.rows[0]["steady_ns"]
        self.assertTrue(self.analyze(records.data())["complete"])
        records.rows[1]["steady_ns"] -= 1
        self.code(self.analyze(records.data()), "steady_clock_regressed")

    def test_exact_integer_ids_and_clocks_beyond_javascript_precision(self):
        huge = 2**64 - 1
        records = Records(pid=2**53 + 1)
        records.add("trace_open", tid=huge)
        begin = records.begin(huge, tid=huge, model_id=2**53 + 1, session_id=huge,
                              token_count=2**63 - 1, position=2**53 + 3, target_position=2**53 + 5)
        records.end(begin)
        records.add("trace_close", tid=huge)
        report = self.analyze(records.data())
        self.assertTrue(report["complete"], report["errors"])
        self.assertEqual(report["pid"], 2**53 + 1)
        records.rows = records.rows[:2]
        report = self.analyze(records.data())
        self.assertEqual(report["open_scopes"][0]["scope_id"], huge)
        self.assertEqual(report["open_scopes"][0]["model_id"], 2**53 + 1)

    def test_maximum_encoder_values_remain_exact(self):
        unsigned_max, signed_max = 2**64 - 1, 2**63 - 1
        records = Records(pid=unsigned_max)
        records.add("trace_open")
        fields = dict(model_id=unsigned_max, session_id=unsigned_max, bytes=unsigned_max,
                      token_count=signed_max, position=signed_max, target_position=signed_max)
        parent = records.begin(unsigned_max - 1, **fields)
        child = records.begin(unsigned_max, "queue_wait", parent_scope_id=unsigned_max - 1, **fields)
        records.end(child)
        records.end(parent)
        records.add("trace_close")
        for row in records.rows:
            row.update(tid=unsigned_max, utc_ns=signed_max, steady_ns=signed_max)
        report = self.analyze(records.data())
        self.assertTrue(report["complete"], report["errors"])
        self.assertEqual(report["pid"], unsigned_max)
        self.assertEqual(report["last_record"]["steady_ns"], signed_max)
        records.rows = records.rows[:3]
        report = self.analyze(records.data())
        self.assertEqual(report["open_scopes"][1]["scope_id"], unsigned_max)
        self.assertEqual(report["open_scopes"][1]["parent_scope_id"], unsigned_max - 1)
        self.assertEqual(report["open_scopes"][1]["bytes"], unsigned_max)
        self.assertEqual(report["open_scopes"][1]["target_position"], signed_max)

    def test_integers_beyond_encoder_range_are_invalid(self):
        groups = (
            (("utc_ns", "steady_ns", "token_count", "position", "target_position"), 2**63),
            (("schema_version", "seq", "pid", "tid", "scope_id", "parent_scope_id",
              "model_id", "session_id", "bytes"), 2**64),
        )
        for fields, overflow in groups:
            for field in fields:
                with self.subTest(field=field):
                    row = self.minimal().rows[0].copy()
                    row[field] = overflow
                    report = self.analyze(encode(row))
                    self.code(report, "invalid_integer")
                    self.assertEqual(report["record_count"], 0)

    def test_boundary_records_and_close_with_live_scope(self):
        records = self.nested()
        records.rows = records.rows[:2]
        records.add("trace_close")
        report = self.analyze(records.data())
        self.code(report, "trace_close_with_open_scopes")
        self.assertFalse(report["trace_close"])
        self.assertEqual(report["open_scope_count"], 1)
        for key, value in (("scope_id", 1), ("parent_scope_id", 1), ("status", "error"), ("model_id", 1), ("operation", "forward")):
            records = self.minimal()
            records.rows[0][key] = value
            self.code(self.analyze(records.data()), "invalid_trace_boundary")

    def test_repeated_open_and_any_record_after_close_are_invalid(self):
        records = self.minimal()
        records.rows[1]["event"] = "trace_open"
        self.code(self.analyze(records.data()), "duplicate_trace_open")
        records = self.minimal()
        records.add("trace_close")
        report = self.analyze(records.data())
        self.code(report, "record_after_trace_close")
        self.assertTrue(report["trace_close"])
        self.assertFalse(report["complete"])
        self.assertEqual(report["record_count"], 2)
        records = Records()
        records.begin(1)
        self.code(self.analyze(records.data()), "missing_initial_trace_open")

    def test_partial_bytes_after_close_do_not_become_success(self):
        report = self.analyze(self.minimal().data() + b"{")
        self.assertTrue(report["trace_close"])
        self.assertFalse(report["complete"])
        self.code(report, "incomplete_tail")

    def test_duplicate_members_including_escaped_key_names(self):
        first = encode(self.minimal().rows[0])
        for replacement in (b'"seq":1,"seq":1', b'"seq":1,"se\\u0071":1'):
            data = first.replace(b'"seq":1', replacement)
            self.code(self.analyze(data), "duplicate_key")

    def test_numeric_types_and_minimums(self):
        for key in trace.INTEGER_MINIMUMS:
            for bad in (True, False, 1.0, "1", None):
                with self.subTest(field=key, value=bad):
                    row = self.minimal().rows[0].copy()
                    row[key] = bad
                    self.code(self.analyze(encode(row)), "invalid_integer")
            row = self.minimal().rows[0].copy()
            row[key] = trace.INTEGER_MINIMUMS[key] - 1
            self.code(self.analyze(encode(row)), "invalid_integer")

    def test_missing_and_unknown_fields(self):
        for key in trace.FIELDS:
            row = self.minimal().rows[0].copy()
            del row[key]
            self.code(self.analyze(encode(row)), "invalid_fields")
        row = self.minimal().rows[0].copy()
        row["extra"] = 1
        self.code(self.analyze(encode(row)), "invalid_fields")

    def test_version_enums_and_strings(self):
        for key, value, code in (
            ("schema_version", 2, "unsupported_schema"), ("event", "tick", "invalid_event"),
            ("phase", "batch", "invalid_phase"), ("memory", "host", "invalid_memory"),
            ("trace_id", "", "empty_trace_id"), ("trace_id", None, "invalid_string"),
        ):
            row = self.minimal().rows[0].copy()
            row[key] = value
            self.code(self.analyze(encode(row)), code)
        for key, value, code in (("operation", "unknown", "invalid_operation"),
                                 ("scope_id", 0, "invalid_scope_id"), ("status", "error", "invalid_status")):
            records = self.nested()
            records.rows[1][key] = value
            self.code(self.analyze(records.data()), code)

    def test_all_known_operations_are_accepted(self):
        records = Records()
        records.add("trace_open")
        for scope, operation in enumerate(trace.OPERATIONS, 1):
            begin = records.begin(scope, operation)
            records.end(begin)
        records.add("trace_close")
        self.assertTrue(self.analyze(records.data())["complete"])

    def test_nul_invalid_utf8_surrogates_bom_and_json_shapes(self):
        good = encode(self.minimal().rows[0])
        for raw, code in ((b"\0\n", "nul_byte"), (b"\xff\n", "invalid_utf8"),
                          (b"\xef\xbb\xbf" + good, "invalid_json"),
                          (b"[]\n", "invalid_record_type"), (b"null\n", "invalid_record_type"),
                          (b'{"seq":NaN}\n', "invalid_json_constant"),
                          (b'{"seq":Infinity}\n', "invalid_json_constant"),
                          (good.replace(b'execution-', b'\\u0000execution-', 1), "nul_character"),
                          (good.replace(b'execution-', b'\\ud800execution-', 1), "invalid_unicode")):
            self.code(self.analyze(raw), code)

    def test_exact_record_limit_includes_newline(self):
        records = self.minimal()
        first = encode(records.rows[0]).rstrip(b"\n")
        self.assertEqual(trace.MAX_RECORD_BYTES, 1536)
        for newline in (b"\n", b"\r\n"):
            with self.subTest(newline=newline):
                padded = first + b" " * (1536 - len(first) - len(newline)) + newline
                self.assertEqual(len(padded), 1536)
                self.assertTrue(self.analyze(padded + encode(records.rows[1]))["complete"])
                report = self.analyze(padded[:-len(newline)] + b" " + newline + encode(records.rows[1]))
                self.code(report, "record_too_large")
                self.assertTrue(report["hash_complete"])

    def test_long_non_newline_input_is_bounded_and_hashed(self):
        data = b"x" * (trace.READ_CHUNK_BYTES * 3 + 17)
        report = self.analyze(data)
        self.code(report, "record_too_large")
        self.assertEqual(report["record_count"], 0)
        self.assertEqual(report["sha256"], hashlib.sha256(data).hexdigest())

    def test_file_size_limit_and_exact_limit(self):
        self.assertEqual(trace.MAX_FILE_BYTES, 64 * 1024 * 1024)
        oversized = self.directory / "execution-large.jsonl"
        with oversized.open("wb") as output:
            output.truncate(trace.MAX_FILE_BYTES + 1)
        with mock.patch.object(Path, "open", side_effect=AssertionError("oversized input must not be opened")):
            report = trace.analyze_file(oversized)
        self.code(report, "file_too_large")
        self.assertIsNone(report["sha256"])
        self.assertFalse(report["hash_complete"])
        self.assertEqual(report["bytes_read"], 0)
        data = self.minimal().data()
        with mock.patch.object(trace, "MAX_FILE_BYTES", len(data)):
            self.assertTrue(self.analyze(data)["complete"])
            self.code(self.analyze(data + b"\n"), "file_too_large")

    def test_empty_missing_directory_and_permission_error_inputs(self):
        report = self.analyze(b"")
        self.assertEqual(report["state"], "incomplete")
        self.assertEqual(report["sha256"], hashlib.sha256(b"").hexdigest())
        self.code(report, "missing_trace_open")
        self.code(trace.analyze_file(self.directory / "absent.jsonl"), "io_error")
        self.code(trace.analyze_file(self.directory), "not_regular_file")
        path = self.directory / "execution-permission.jsonl"
        path.write_bytes(self.minimal().data())
        with mock.patch.object(Path, "open", side_effect=PermissionError("denied")):
            self.code(trace.analyze_file(path), "io_error")

    def test_file_growth_after_eof_invalidates_success_and_hash_claim(self):
        path = self.directory / "execution-changing.jsonl"
        path.write_bytes(self.minimal().data())
        original_stat = Path.stat
        calls = 0

        def changing_stat(target, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                with path.open("ab") as output:
                    output.write(b"{")
            return original_stat(target, *args, **kwargs)

        with mock.patch.object(Path, "stat", changing_stat):
            report = trace.analyze_file(path)
        self.assertFalse(report["complete"])
        self.assertFalse(report["hash_complete"])
        self.assertIsNone(report["sha256"])
        self.code(report, "file_changed")

    def test_error_details_are_bounded_without_losing_error_count(self):
        records = Records()
        records.add("trace_open")
        for scope in range(1, 71):
            begin = records.begin(scope)
            records.end(begin, status="error")
        records.add("trace_close")
        report = self.analyze(records.data())
        self.assertTrue(report["trace_close"])
        self.assertFalse(report["complete"])
        self.assertEqual(report["scope_error_count"], 70)
        self.assertEqual(report["error_count"], 70)
        self.assertEqual(len(report["errors"]), trace.MAX_ERROR_DETAILS)
        self.assertTrue(report["errors_truncated"])

    def test_directory_requires_traces_and_all_files_to_complete(self):
        self.code(trace.analyze_directory(self.directory), "no_traces")
        (self.directory / "ignored.jsonl").write_bytes(b"broken\n")
        (self.directory / "execution-b.jsonl").write_bytes(self.minimal(pid=456).data())
        (self.directory / "execution-a.jsonl").write_bytes(self.minimal().data())
        report = trace.analyze_directory(self.directory)
        self.assertTrue(report["complete"])
        self.assertEqual(report["file_count"], 2)
        self.assertEqual([Path(item["path"]).name for item in report["files"]], ["execution-a.jsonl", "execution-b.jsonl"])
        (self.directory / "execution-b.jsonl").write_bytes(encode(self.minimal(pid=456).rows[0]))
        self.assertFalse(trace.analyze_directory(self.directory)["complete"])

    def test_directory_duplicate_ids_and_aggregate_limits(self):
        data = self.minimal().data()
        for name in ("execution-a.jsonl", "execution-b.jsonl"):
            (self.directory / name).write_bytes(data)
        self.code(trace.analyze_directory(self.directory), "duplicate_trace_id")
        with mock.patch.object(trace, "MAX_DIRECTORY_FILES", 1):
            self.code(trace.analyze_directory(self.directory), "directory_file_limit")
        with mock.patch.object(trace, "MAX_DIRECTORY_BYTES", len(data)):
            self.code(trace.analyze_directory(self.directory), "directory_byte_limit")

    def test_cli_json_exit_codes_new_output_and_evidence_overwrite_refusal(self):
        source = self.directory / "execution-cli.jsonl"
        data = self.minimal().data()
        source.write_bytes(data)
        output = self.directory / "analysis.json"
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            code = trace.main([str(source), "--output", str(output)])
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(captured.getvalue())["complete"])
        self.assertEqual(json.loads(output.read_text()), json.loads(captured.getvalue()))
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            code = trace.main([str(source), "--output", str(source)])
        self.assertEqual(code, 2)
        self.assertEqual(source.read_bytes(), data)
        self.assertEqual(json.loads(captured.getvalue())["output_error"]["code"], "output_not_created")
        source.write_bytes(encode(self.minimal().rows[0]))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(trace.main([str(source)]), 1)
            self.assertEqual(trace.main([str(self.directory)]), 1)


if __name__ == "__main__":
    unittest.main()
