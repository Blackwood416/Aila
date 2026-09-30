import copy
import unittest

from q4e_cache_replay import offline_reference, simulate, validate


def fixture():
    # The second batch has a hit *after* a miss in input order. The hit must
    # already be reserved when choosing the first victim. Then demote reverses
    # the next victim choice. Expected slots were worked out by hand.
    return [
        dict(type="config", schema=1, slots=2, experts=8, layers=1, host_slots=0, cost_aware=0, slot_bytes=64),
        dict(type="forward_begin", forward=1, position=0, tokens=2, ns=0),
        dict(type="acquire_begin", forward=1, acquire=1, layer=0, ids=[0, 1], initial_slots=[-1, -1], host_states=[0, 0], pinned=0),
        dict(type="victim", acquire=1, slot=1, key=0, old_key=-1, candidates=[], backed=[]),
        dict(type="victim", acquire=1, slot=0, key=1, old_key=-1, candidates=[], backed=[]),
        end(1, 0, 2),
        dict(type="acquire_begin", forward=1, acquire=2, layer=0, ids=[2, 0], initial_slots=[-1, 1], host_states=[0, 0], pinned=0),
        dict(type="victim", acquire=2, slot=0, key=2, old_key=1, candidates=[], backed=[]),
        end(2, 1, 1),
        dict(type="demote", forward=1, layer=0, ids=[2]),
        dict(type="acquire_begin", forward=1, acquire=3, layer=0, ids=[3], initial_slots=[-1], host_states=[0], pinned=0),
        dict(type="victim", acquire=3, slot=0, key=3, old_key=2, candidates=[], backed=[]),
        end(3, 0, 1),
        dict(type="forward_end", forward=1, position=0, tokens=2, ns=100),
    ]


def end(acquire, hits, misses):
    return dict(type="acquire_end", acquire=acquire, hits=hits, misses=misses, bytes=64*misses,
                host_hits=0, file_fills=misses, fill_ns=0, prefetch_wait_ns=0, ns=acquire*20, begin_ns=acquire*20-10)


class ReplayTests(unittest.TestCase):
    def test_batch_reservation_and_demote(self):
        observed = validate(fixture())[0]
        predicted = simulate(fixture())[0]
        self.assertEqual((observed["hits"], observed["misses"], observed["bytes"]), (1, 4, 256))
        self.assertEqual(predicted["bytes"], observed["bytes"])
        self.assertEqual(offline_reference(fixture())[0]["bytes"], observed["bytes"])

    def test_cost_aware_window(self):
        rows = fixture()
        rows[0].update(host_slots=4, cost_aware=1)
        for r in rows:
            if r["type"] == "victim":
                r.update(candidates=[r["slot"]], backed=[int(r["old_key"] >= 0)])
        validate(rows)
        rows[3]["candidates"] = [0]  # Skipped the actual LRU-tail slot.
        with self.assertRaisesRegex(ValueError, "candidate window"):
            validate(rows)

    def test_corruption_and_truncation_rejected(self):
        for index, key, value in ((3, "old_key", 7), (5, "bytes", 0), (6, "initial_slots", [-1, -1])):
            rows = copy.deepcopy(fixture())
            rows[index][key] = value
            with self.assertRaises(ValueError):
                validate(rows)
        with self.assertRaisesRegex(ValueError, "Truncated"):
            validate(fixture()[:-1])

    def test_empty_slots_follow_runtime_demote(self):
        # Runtime can recycle a demoted slot before unused slots, so modeling
        # only resident keys (a usual textbook LRU) would predict a false hit.
        rows = [dict(type="config", slots=3, experts=8, slot_bytes=64),
                dict(type="forward_begin", forward=1, position=0, tokens=3),
                dict(type="acquire_begin", layer=0, ids=[0]),
                dict(type="demote", layer=0, ids=[0]),
                dict(type="acquire_begin", layer=0, ids=[1]),
                dict(type="acquire_begin", layer=0, ids=[0]),
                dict(type="forward_end")]
        self.assertEqual(simulate(rows)[0]["hits"], 0)


if __name__ == "__main__":
    unittest.main()
