"""CPU-only tests. Every guard snapshot is injected; no GPU or Windows probe runs."""

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gpu_run_guard as guard


def healthy():
    return {"schema_version": 1, "captured_at": "2026-09-08T00:00:00+00:00",
            "host": "cpu-test-host", "boot_id": "boot-one", "events": [], "dumps": [],
            "event_log_heads": {"System": 1000, "Application": 2000}, "read_errors": []}


def fault(code="0x00000116", record=1001):
    return guard._classify_event({
        "log": "System", "record_id": record, "event_id": 1001,
        "provider": "Microsoft-Windows-WER-SystemErrorReporting",
        "time_created": "2026-09-08T00:01:00+00:00",
        "data": {"param1": code + " (0x0, 0x0, 0x0, 0x0)"}})


def wer_snapshot():
    value = healthy()
    value["dump_roots"] = [r"C:\Windows\LiveKernelReports", r"C:\Windows\Minidump"]
    value["full_dump_path"] = r"C:\Windows\MEMORY.DMP"
    value["dumps"] = [{"path": r"C:\Windows\LiveKernelReports\WATCHDOG\old.dmp", "size": 8192, "mtime_ns": 42}]
    value["events"] = [guard._classify_event({"log": "Application", "event_id": 1001, "record_id": 2001,
        "provider": "Windows Error Reporting", "time_created": "2026-09-08T00:01:00Z",
        "data": {"EventName": "LiveKernelEvent", "P1": "141", "P2": "abc", "ReportStatus": "2051",
                 "ReportId": "12345678-1234-4321-9876-123456789abc",
                 "AttachedFiles": "\n" + r"\\?\C:\WINDOWS\LiveKernelReports\WATCHDOG\old.dmp" + "\n",
                 "StorePath": "old-report-queue"}})]
    return value


def add_wer_replay(value, record=2002):
    event = copy.deepcopy(value["events"][0])
    event.update(record_id=record, time_created="2026-09-08T00:20:00Z")
    value["events"].append(event)
    value["event_log_heads"]["Application"] = record


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "machine-guard"
        self.snapshot = healthy()

    def make_guard(self, **kwargs):
        return guard.GpuRunGuard(
            "CPU-injected test", state_dir=self.directory,
            snapshot_reader=lambda: copy.deepcopy(self.snapshot), **kwargs)

    def state(self):
        return json.loads((self.directory / "state.json").read_bytes())["payload"]

    def expect_blocked(self):
        with self.assertRaises(guard.GuardBlocked) as caught:
            with self.make_guard():
                self.fail("A hazardous run was admitted")
        return caught.exception

    def finish_cleanly(self):
        with self.make_guard() as run:
            self.assertEqual(self.state()["status"], "started")
            self.assertIsNotNone(run.lock.file)
            run.finish()
            self.assertIsNotNone(run.lock.file)
        self.assertEqual(self.state()["status"], "finished")

    def acknowledge(self, evidence_id, reason="Reviewed the injected CPU test evidence"):
        return guard.acknowledge(
            evidence_id, reason, state_dir=self.directory,
            snapshot_reader=lambda: copy.deepcopy(self.snapshot))

    def test_clean_start_finish_and_next_run(self):
        self.finish_cleanly()
        self.finish_cleanly()
        lines = (self.directory / "journal.jsonl").read_bytes().splitlines()
        self.assertEqual([json.loads(line)["payload"]["status"] for line in lines],
                         ["started", "finished", "started", "finished"])

    def test_first_start_never_accepts_existing_bugchecks(self):
        for code in ("0x00000116", "0x00000119", "0x00000139"):
            with self.subTest(code=code):
                self.snapshot["events"] = [fault(code)]
                evidence = self.expect_blocked()
                self.assertTrue(any(item["kind"] == "initial_fault_evidence"
                                    for item in evidence.evidence["reasons"]))
                self.assertEqual(self.state()["status"], "blocked")

    def test_first_start_never_accepts_existing_dump(self):
        self.snapshot["dumps"] = [{"path": r"C:\Windows\Minidump\one.dmp", "size": 4096, "mtime_ns": 42}]
        self.expect_blocked()

    def test_new_fault_after_clean_finish_blocks(self):
        self.finish_cleanly()
        self.snapshot["events"] = [fault()]
        self.snapshot["event_log_heads"]["System"] += 1
        evidence = self.expect_blocked()
        self.assertIn("new_fault_evidence", [item["kind"] for item in evidence.evidence["reasons"]])

    def test_replayed_report_never_clears_unreviewed_failure(self):
        self.snapshot = wer_snapshot()
        pending = self.expect_blocked()
        add_wer_replay(self.snapshot)
        repeated = self.expect_blocked()
        self.assertEqual(repeated.evidence_id, pending.evidence_id)
        self.assertEqual(self.state()["status"], "blocked")
        self.acknowledge(pending.evidence_id)
        with self.make_guard() as run:
            add_wer_replay(self.snapshot, 2003)
            run.finish()
        record = self.state()
        self.assertEqual(record["status"], "finished")
        self.assertEqual(len(record["snapshot"]["events"]), 3)
        self.assertEqual(record["historical_wer_replays"][0]["phase"], "during_run")
        self.assertEqual(record["historical_wer_replays"][0]["record_id"], 2003)

    def test_replayed_report_does_not_hide_failed_workload(self):
        self.snapshot = wer_snapshot()
        pending = self.expect_blocked()
        self.acknowledge(pending.evidence_id)
        with self.make_guard() as run:
            add_wer_replay(self.snapshot)
            run.finish(success=False, details={"exit_code": 9})
        self.assertEqual(self.state()["status"], "failed")
        self.assertIn("run_failed", [row["kind"] for row in self.expect_blocked().evidence["reasons"]])

    def test_replay_before_start_is_retained_in_journal(self):
        self.snapshot = wer_snapshot()
        pending = self.expect_blocked()
        self.acknowledge(pending.evidence_id)
        self.finish_cleanly()
        add_wer_replay(self.snapshot)
        self.finish_cleanly()
        rows = self.state()["historical_wer_replays"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["phase"], "before_start")

    def test_new_or_changed_dump_after_finish_blocks(self):
        self.finish_cleanly()
        self.snapshot["dumps"] = [{"path": r"C:\Windows\LiveKernelReports\WATCHDOG\one.dmp",
                                  "size": 4096, "mtime_ns": 42}]
        first = self.expect_blocked()
        self.acknowledge(first.evidence_id)
        self.finish_cleanly()
        self.snapshot["dumps"][0]["mtime_ns"] = 43
        changed = self.expect_blocked()
        self.assertNotEqual(first.evidence_id, changed.evidence_id)

    def test_boot_change_requires_review_even_without_bugcheck(self):
        self.finish_cleanly()
        self.snapshot["boot_id"] = "boot-two"
        evidence = self.expect_blocked()
        self.assertIn("boot_id_changed", [item["kind"] for item in evidence.evidence["reasons"]])

    def test_event_log_rewind_cannot_hide_fault_history(self):
        self.finish_cleanly()
        self.snapshot["event_log_heads"]["System"] = 1
        evidence = self.expect_blocked()
        self.assertIn("event_log_rewound", [item["kind"] for item in evidence.evidence["reasons"]])

    def test_failed_run_blocks_next_admission(self):
        with self.make_guard() as run:
            run.finish(success=False, details={"exit_code": 1})
        self.assertEqual(self.state()["status"], "failed")
        evidence = self.expect_blocked()
        self.assertIn("run_failed", [item["kind"] for item in evidence.evidence["reasons"]])

    def test_body_exception_writes_failed_and_preserves_exception(self):
        with self.assertRaisesRegex(RuntimeError, "injected failure"):
            with self.make_guard():
                raise RuntimeError("injected failure")
        self.assertEqual(self.state()["status"], "failed")
        self.assertIn("injected failure", self.state()["details"]["error"])
        self.expect_blocked()

    def test_exception_after_finish_does_not_leave_a_success_record(self):
        with self.assertRaisesRegex(RuntimeError, "late failure"):
            with self.make_guard() as run:
                run.finish()
                raise RuntimeError("late failure")
        self.assertEqual(self.state()["status"], "failed")
        self.expect_blocked()

    def test_a_guard_instance_cannot_be_reused(self):
        run = self.make_guard()
        with run:
            run.finish()
        with self.assertRaises(guard.GuardError):
            with run:
                self.fail("A reused guard kept its old finish marker")

    def test_missing_explicit_finish_is_a_failure(self):
        with self.assertRaises(guard.GuardBlocked):
            with self.make_guard():
                pass
        self.assertEqual(self.state()["status"], "failed")
        self.expect_blocked()

    def test_post_run_fault_turns_requested_success_into_failure(self):
        with self.assertRaises(guard.GuardBlocked):
            with self.make_guard() as run:
                self.snapshot["events"] = [fault()]
                run.finish()
        self.assertEqual(self.state()["status"], "failed")
        self.expect_blocked()

    def test_permission_failure_is_retained_not_healthy(self):
        self.snapshot["read_errors"] = [{"source": "Minidump", "error": "PermissionError: access denied"}]
        error = self.expect_blocked()
        self.assertIn("access denied", json.dumps(self.state()))
        with self.assertRaises(guard.GuardBlocked):
            self.acknowledge(error.evidence_id)
        self.assertEqual(self.state()["status"], "blocked")

    def test_reader_exception_is_retained_not_healthy(self):
        def denied():
            raise PermissionError("injected OS access denied")

        with self.assertRaises(guard.GuardBlocked):
            with guard.GpuRunGuard("reader failure", state_dir=self.directory, snapshot_reader=denied):
                self.fail("Unreadable evidence was admitted")
        self.assertIn("injected OS access denied", json.dumps(self.state()))

    def test_missing_boot_or_log_head_is_not_healthy(self):
        for field in ("boot_id", "event_log_heads"):
            with self.subTest(field=field):
                self.snapshot = healthy()
                del self.snapshot[field]
                self.expect_blocked()

    def test_snapshot_order_and_capture_time_do_not_change_pending_id(self):
        self.snapshot["events"] = [fault(record=1001), fault("0x00000139", 1002)]
        first = self.expect_blocked()
        self.snapshot["events"].reverse()
        self.snapshot["captured_at"] = "2026-09-08T00:20:00+00:00"
        second = self.expect_blocked()
        self.assertEqual(first.evidence_id, second.evidence_id)

    def test_evidence_disappearance_does_not_clear_pending_failure(self):
        self.snapshot["events"] = [fault()]
        first = self.expect_blocked()
        self.snapshot["events"] = []
        second = self.expect_blocked()
        self.assertEqual(first.evidence_id, second.evidence_id)

    def test_acknowledgement_requires_exact_id_and_nonempty_reason(self):
        self.snapshot["events"] = [fault()]
        pending = self.expect_blocked()
        with self.assertRaises(guard.GuardError):
            self.acknowledge("0" * 64)
        with self.assertRaises(ValueError):
            self.acknowledge(pending.evidence_id, "  ")
        self.assertEqual(self.state()["status"], "blocked")
        self.acknowledge(pending.evidence_id)
        self.assertEqual(self.state()["status"], "acknowledged")
        self.assertEqual(self.state()["acknowledgement"]["evidence_id"], pending.evidence_id)
        self.finish_cleanly()

    def test_new_evidence_invalidates_old_acknowledgement(self):
        self.snapshot["events"] = [fault()]
        pending = self.expect_blocked()
        self.snapshot["events"].append(fault("0x00000139", 1002))
        with self.assertRaises(guard.GuardBlocked) as caught:
            self.acknowledge(pending.evidence_id)
        self.assertNotEqual(pending.evidence_id, caught.exception.evidence_id)
        self.assertEqual(self.state()["status"], "blocked")

    def test_atomic_replace_failure_never_restores_healthy_admission(self):
        self.finish_cleanly()
        previous_bytes = (self.directory / "state.json").read_bytes()
        with mock.patch.object(guard, "_replace_write_through", side_effect=OSError("injected rename failure")):
            with self.assertRaises(guard.GuardPersistenceError):
                with self.make_guard():
                    self.fail("A non-durable started record was admitted")
        self.assertEqual((self.directory / "state.json").read_bytes(), previous_bytes)
        with self.assertRaisesRegex(guard.GuardPersistenceError, "disagree"):
            with self.make_guard():
                self.fail("Mismatched journal and state were admitted")
        with guard.ProcessFileLock(self.directory / "gpu.lock"):
            pass  # Failure released the OS lock but did not clear evidence.

    def test_failed_finish_write_is_not_retried_during_context_cleanup(self):
        with self.assertRaises(guard.GuardPersistenceError):
            with self.make_guard() as run:
                with mock.patch.object(guard, "_replace_write_through", side_effect=OSError("injected finish failure")):
                    run.finish()
        rows = (self.directory / "journal.jsonl").read_bytes().splitlines()
        self.assertEqual(len(rows), 2)
        self.assertEqual([json.loads(row)["sequence"] for row in rows], [1, 2])
        with self.assertRaises(guard.GuardPersistenceError):
            with self.make_guard():
                self.fail("An incompletely committed finish was admitted")

    def test_each_commit_flushes_both_journal_and_atomic_state(self):
        with mock.patch.object(guard.os, "fsync", wraps=os.fsync) as synced:
            self.finish_cleanly()
        # Lock-file creation plus two records with journal + state fsync.
        self.assertGreaterEqual(synced.call_count, 5)

    def test_empty_nul_or_truncated_records_fail_closed(self):
        self.finish_cleanly()
        original = (self.directory / "state.json").read_bytes()
        for raw in (b"", b"\0" * 128, b'{"payload":'):
            with self.subTest(raw=raw):
                (self.directory / "state.json").write_bytes(raw)
                with self.assertRaises(guard.GuardPersistenceError):
                    with self.make_guard():
                        self.fail("Corrupt persistent state was admitted")
        (self.directory / "state.json").write_bytes(original)
        with (self.directory / "journal.jsonl").open("ab") as stream:
            stream.write(b'{"partial":')
        with self.assertRaises(guard.GuardPersistenceError):
            with self.make_guard():
                self.fail("A partial journal was admitted")

    def test_missing_state_with_existing_journal_fails_closed(self):
        self.finish_cleanly()
        (self.directory / "state.json").unlink()
        with self.assertRaises(guard.GuardPersistenceError):
            with self.make_guard():
                self.fail("A missing half of the evidence pair was admitted")

    def test_unwritable_state_location_does_not_fallback(self):
        self.directory.write_text("A file occupies the required directory", encoding="utf-8")
        with self.assertRaises(guard.GuardPersistenceError):
            with self.make_guard():
                self.fail("An unavailable shared directory was bypassed")
        self.assertTrue(self.directory.is_file())

    def child_guard(self):
        script = """
import json, sys
from gpu_run_guard import GpuRunGuard
snapshot = json.loads(sys.argv[2])
with GpuRunGuard('CPU child', state_dir=sys.argv[1], snapshot_reader=lambda: snapshot) as run:
    print('locked', flush=True)
    if sys.stdin.readline().strip() == 'finish':
        run.finish()
"""
        child = subprocess.Popen(
            [sys.executable, "-u", "-c", script, str(self.directory), json.dumps(healthy())],
            cwd=Path(__file__).resolve().parent, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)

        def cleanup():
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=10)
        self.addCleanup(cleanup)
        self.assertEqual(child.stdout.readline().strip(), "locked")
        return child

    def test_interprocess_lock_ignores_distinct_output_metadata(self):
        child = self.child_guard()
        with self.assertRaises(guard.GuardBusy):
            with self.make_guard(metadata={"output": "an-entirely-different-output"}):
                self.fail("Two processes simultaneously owned the machine guard")
        stdout, stderr = child.communicate("finish\n", timeout=10)
        self.assertEqual(child.returncode, 0, (stdout, stderr))
        self.finish_cleanly()

    def test_process_death_releases_os_lock_but_unfinished_run_still_blocks(self):
        child = self.child_guard()
        child.kill()
        child.communicate(timeout=10)
        evidence = self.expect_blocked()
        self.assertIn("unfinished_run", [item["kind"] for item in evidence.evidence["reasons"]])

    def test_bugcheck_and_live_kernel_classification_is_not_locale_dependent(self):
        for code in (0x116, 0x119, 0x139):
            self.assertEqual(fault(f"0x{code:08x}")["bugcheck_code"], hex(code))
        event = guard._classify_event({"event_id": 1001, "data": {"EventName": "LiveKernelEvent", "P1": "141"}})
        self.assertEqual(event["signal"], "live_kernel_event")
        self.assertEqual(event["live_kernel_code"], "141")
        event = guard._classify_event({"event_id": 41, "data": {"BugcheckCode": "278"}})
        self.assertEqual(event["signal"], "unexpected_power_loss")
        self.assertEqual(event["bugcheck_code"], "0x116")

    def test_dump_enumeration_reads_only_metadata(self):
        root = Path(self.temporary.name) / "dumps"
        nested = root / "WATCHDOG"
        nested.mkdir(parents=True)
        (nested / "fault.dmp").write_bytes(b"CPU fixture")
        (root / "unrelated.txt").write_text("ignore", encoding="utf-8")
        rows, errors = [], []
        guard._dump_metadata(root, rows, errors)
        self.assertEqual(errors, [])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["size"], len(b"CPU fixture"))
        self.assertNotIn("contents", rows[0])


class WerReplayTests(unittest.TestCase):
    def test_unchanged_absence_requires_complete_unchanged_scan_coverage(self):
        before = wer_snapshot()
        before["dumps"].clear()
        after = copy.deepcopy(before)
        add_wer_replay(after)
        replayed = guard.historical_wer_replays(before, after)
        self.assertEqual(len(replayed), 1)
        self.assertEqual(len(next(iter(replayed.values()))["unchanged_absent_dump_paths"]), 1)
        for change in ("missing_coverage", "outside_root", "changed_root", "read_error", "new_dump"):
            old, new = copy.deepcopy(before), copy.deepcopy(after)
            if change == "missing_coverage": old.pop("dump_roots")
            elif change == "outside_root":
                old["dump_roots"] = new["dump_roots"] = [r"C:\Windows\Minidump"]
            elif change == "changed_root": new["dump_roots"] = [r"D:\Windows\LiveKernelReports"]
            elif change == "read_error": new["read_errors"].append({"source": "dump", "error": "denied"})
            else: new["dumps"] = [{"path": r"C:\Windows\Minidump\new.dmp", "size": 1234, "mtime_ns": 84}]
            with self.subTest(change=change):
                self.assertFalse(guard.historical_wer_replays(old, new))
                self.assertTrue(guard._snapshot_reasons(old, new))

    def test_bad_scan_metadata_is_unreadable(self):
        value = healthy()
        value["dump_roots"] = ["relative/path"]
        value["full_dump_path"] = r"C:\Windows\MEMORY.DMP"
        parsed = guard._read_snapshot(lambda: value)
        self.assertTrue(parsed["read_errors"])

    def test_complete_report_and_dump_identity_are_required(self):
        before = wer_snapshot()
        after = copy.deepcopy(before)
        add_wer_replay(after)
        self.assertEqual(guard._snapshot_reasons(before, after), [])
        replays = guard.historical_wer_replays(before, after)
        self.assertEqual(len(replays), 1)
        row = next(iter(replays.values()))
        self.assertEqual(row["matched_records"], [2001])
        self.assertEqual(row["unchanged_dump_paths"], [r"c:\windows\livekernelreports\watchdog\old.dmp"])
        self.assertEqual(len(after["events"]), 2)  # Never discard the raw record.

    def test_changed_report_fields_still_block(self):
        before = wer_snapshot()
        for key, changed in (("ReportId", "87654321-1234-4321-9876-123456789abc"), ("P1", "119"),
                             ("P2", "def"), ("ReportStatus", "0"), ("StorePath", "new-report"),
                             ("AttachedFiles", r"C:\Windows\Minidump\new.dmp")):
            with self.subTest(key=key):
                after = copy.deepcopy(before)
                add_wer_replay(after)
                after["events"][-1]["data"][key] = changed
                self.assertFalse(guard.historical_wer_replays(before, after))
                self.assertTrue(guard._snapshot_reasons(before, after))

    def test_changed_or_missing_dump_still_blocks(self):
        for change in ("size", "mtime_ns", "missing_after", "missing_before"):
            with self.subTest(change=change):
                before = wer_snapshot()
                after = copy.deepcopy(before)
                add_wer_replay(after)
                if change == "missing_after": after["dumps"].clear()
                elif change == "missing_before": before["dumps"].clear()
                else: after["dumps"][0][change] += 1
                self.assertFalse(guard.historical_wer_replays(before, after))
                self.assertTrue(guard._snapshot_reasons(before, after))

    def test_missing_identity_or_attachments_are_not_replays(self):
        for key, invalid in (("ReportId", ""), ("ReportId", "0" * 36),
                             ("ReportId", "00000000-0000-0000-0000-000000000000"),
                             ("AttachedFiles", ""), ("AttachedFiles", "relative.dmp")):
            with self.subTest(key=key, invalid=invalid):
                before = wer_snapshot()
                before["events"][0]["data"][key] = invalid
                after = copy.deepcopy(before)
                add_wer_replay(after)
                self.assertFalse(guard.historical_wer_replays(before, after))
                self.assertTrue(guard._snapshot_reasons(before, after))

    def test_system_signals_boot_rewind_and_read_errors_are_not_suppressed(self):
        for change in ("system_event", "unrelated_dump", "boot", "rewind", "read_error"):
            with self.subTest(change=change):
                before = wer_snapshot()
                after = copy.deepcopy(before)
                add_wer_replay(after)
                if change == "system_event": after["events"].append(fault())
                elif change == "unrelated_dump": after["dumps"].append({"path": r"C:\Windows\Minidump\new.dmp", "size": 8192, "mtime_ns": 84})
                elif change == "boot": after["boot_id"] = "new-boot"
                elif change == "rewind": after["event_log_heads"]["System"] = 1
                else: after["read_errors"].append({"source": "dump", "error": "denied"})
                self.assertTrue(guard._snapshot_reasons(before, after))

    def test_other_provider_and_initial_history_still_block(self):
        before = wer_snapshot()
        self.assertTrue(guard._snapshot_reasons(None, before))
        for key, invalid in (("provider", "Microsoft-Windows-WER-SystemErrorReporting"),
                             ("log", "System"), ("event_id", 41)):
            with self.subTest(key=key):
                sample = copy.deepcopy(before)
                sample["events"][0][key] = invalid
                after = copy.deepcopy(sample)
                add_wer_replay(after)
                self.assertFalse(guard.historical_wer_replays(sample, after))
                self.assertTrue(guard._snapshot_reasons(sample, after))


if __name__ == "__main__":
    unittest.main()
