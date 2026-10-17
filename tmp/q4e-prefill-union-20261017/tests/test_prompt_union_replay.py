import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prompt_union_replay import (  # noqa: E402
    aggregate_route_profile,
    profile_only_budget,
    read_hotset_profile,
    read_route_trace,
    simulate_budget,
)


IDENTITY = "synthetic-test-identity"


def write_profile(path: Path, decode_counts, prefill_counts):
    lines = [
        "AILA_Q4E_HOTSET 1",
        IDENTITY,
        "2 4 8",
    ]
    for key in range(8):
        lines.append(
            f"{key} {decode_counts[key]} {prefill_counts[key]} {prefill_counts[key]}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_trace(path: Path):
    rows = [
        {
            "type": "config",
            "schema": 1,
            "layers": 2,
            "experts": 4,
            "top_k": 1,
            "slot_bytes": 100,
        },
        {
            "type": "request_begin",
            "request": "synthetic",
            "prompt_tokens": 8,
        },
    ]
    for block, position in enumerate((0, 4)):
        forward = f"f{block}"
        rows.append({
            "type": "forward_begin",
            "forward": forward,
            "request": "synthetic",
            "position": position,
            "tokens": 4,
            "phase": "prefill",
        })
        rows.append({
            "type": "route",
            "forward": forward,
            "layer": 0,
            "rows": 4,
            "top_k": 1,
            "ids": [0, 1, 2, 2],
        })
        rows.append({
            "type": "route",
            "forward": forward,
            "layer": 1,
            "rows": 4,
            "top_k": 1,
            "ids": [0, 1, 2, 2],
        })
    rows.append({"type": "request_end", "request": "synthetic"})
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


class PromptUnionReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.route_profile_path = self.dir / "route.profile"
        self.decode_profile_path = self.dir / "decode.profile"
        write_profile(
            self.route_profile_path,
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 2, 2, 1, 1, 3, 1],
        )
        write_profile(
            self.decode_profile_path,
            [8, 7, 6, 5, 4, 3, 2, 1],
            [0] * 8,
        )
        self.trace_path = self.dir / "trace.jsonl"
        write_trace(self.trace_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_profiles_and_aggregate_union(self):
        route = read_hotset_profile(self.route_profile_path)
        decode = read_hotset_profile(self.decode_profile_path)
        aggregate = aggregate_route_profile(route, 100)
        self.assertEqual(route["layers"], 2)
        self.assertEqual(decode["experts"], 4)
        self.assertEqual(aggregate["prefill_nonzero_keys"], 8)
        self.assertEqual(aggregate["union_bytes"], 800)
        self.assertEqual(aggregate["per_layer_union"], [4, 4])

    def test_exact_trace_promotes_later_blocks(self):
        route = read_hotset_profile(self.route_profile_path)
        decode = read_hotset_profile(self.decode_profile_path)
        trace = read_route_trace(self.trace_path)
        request = trace["requests"]["synthetic"]
        result = simulate_budget(
            trace, request, route, decode,
            budget=4,
            promotion_slots=2,
            min_occurrences=2,
            slot_bytes=100,
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["blocks"]), 2)
        self.assertEqual(result["blocks"][0]["baseline_staged_slots"], 3)
        self.assertEqual(result["blocks"][0]["promoted_staged_slots"], 3)
        self.assertEqual(result["blocks"][1]["promoted_staged_slots"], 2)
        self.assertGreaterEqual(result["summary"]["saved_slots"], 1)
        self.assertEqual(result["decode_hotset"]["post_rollback_missing_slots_vs_reference"], 4)
        self.assertEqual(result["layer_distribution"][0]["layer"], 0)

    def test_profile_only_is_explicitly_blocked(self):
        route = read_hotset_profile(self.route_profile_path)
        decode = read_hotset_profile(self.decode_profile_path)
        result = profile_only_budget(route, decode, 4, 2, 100)
        self.assertEqual(result["status"], "blocked_missing_per_block_route_trace")
        self.assertEqual(result["blocks"], [])
        self.assertIsNone(result["summary"])
        self.assertEqual(result["decode_hotset"]["worst_case_missing_slots_vs_reference"], 6)


if __name__ == "__main__":
    unittest.main()
