"""Validate schema-1 cache traces, then compare GPU-only admission hypotheses.

The observed replay checks the *actual* eight-candidate host-ready decisions,
batch reservations and demotes. Counterfactual policies deliberately make no
host-cache/I/O/time predictions: different GPU misses change prefetch and host
replacement. Their byte counts are hypotheses for a subsequent real canary.
Legacy binary traces cannot supply this evidence; use q4e_execution_probe.ps1.
"""
import argparse
from collections import OrderedDict, defaultdict
from collections import deque
import heapq
import hashlib
import json
from pathlib import Path


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_trace(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    require(rows and rows[0].get("schema") == 1, "Missing schema-1 config")
    require(sum(r["type"] == "config" for r in rows) == 1, "Multiple cache instances")
    return rows


def validate(rows):
    cfg = rows[0]
    cap, experts = cfg["slots"], cfg["experts"]
    # Oldest first. Runtime starts with slots [0..cap-1], MRU first.
    lru = OrderedDict((s, None) for s in reversed(range(cap)))
    resident, slots = {}, [-1] * cap
    forward, acquire, reserved = None, None, set()
    totals = []
    routed, acquired = {}, {}
    direct, leased = None, set()
    frozen = set()
    epoch_loading = False
    pending = {}
    cold_ring, cold_cursor = [], 0
    for r in rows[1:]:
        kind = r["type"]
        if kind == 'epoch_load_begin':
            require(forward is None and acquire is None and direct is None and not frozen and not pending, 'Unsafe profile refill boundary')
            epoch_loading = True
            forward = dict(r, hits=0, misses=0, bytes=0, host_hits=0, file_fills=0,
                acquire_ns=0, fill_ns=0, prefetch_wait_ns=0, layers={}, acquire_calls=0)
        elif kind == 'epoch_protect':
            require(epoch_loading and acquire is None and all(slots[s]>=0 and s not in frozen for s in r['slots']), 'Invalid profile protection')
            frozen.update(r['slots'])
        elif kind == 'epoch_load_end':
            require(epoch_loading and acquire is None and direct is None, 'Unfinished profile refill')
            epoch_loading, forward = False, None
            cold_ring, cold_cursor = [s for s in range(cap) if s not in frozen], 0
        elif kind == 'epoch_retire':
            s = r['slot']
            require(forward is None and s in frozen and s not in pending and slots[s]==r['old_key'], 'Invalid epoch retirement')
            require(r['key']//experts==r['old_key']//experts, 'Cross-layer swap')
            del resident[slots[s]]
            slots[s] = -1
            pending[s] = r['key']
        elif kind == 'epoch_borrow':
            require(forward is None and not pending and set(r['slots']) <= frozen, 'Unsafe prefill borrowing')
            frozen.difference_update(r['slots'])
            cold_ring, cold_cursor = [s for s in range(cap) if s not in frozen], 0
        elif kind == 'epoch_publish':
            s = r['slot']
            require(forward is None and s in frozen and pending.get(s)==r['key'] and r['key'] not in resident, 'Invalid epoch publication')
            slots[s] = r['key']
            resident[r['key']] = s
            del pending[s]
        elif kind == "freeze":
            require(acquire is None and direct is None and not frozen, "Invalid freeze boundary")
            require(len(set(r['slots'])) == len(r['slots']) and all(0 <= s < cap and slots[s] >= 0 for s in r['slots']), "Invalid frozen slots")
            frozen = set(r['slots'])
        elif kind == "thaw":
            require(acquire is None and direct is None, "Thaw with active readers")
            frozen.clear()
            cold_ring, cold_cursor = [], 0
        elif kind == "discard":
            s = r['slot']
            require(acquire is None and 0 <= s < cap and s not in frozen and s not in leased and slots[s] >= 0, "Invalid transient discard")
            del resident[slots[s]]
            slots[s] = -1
        elif kind == "cold_cpu":
            require(direct and acquire is None and direct['layer'] == r['layer'], "Orphan CPU cold batch")
            require(len(r['ids']) == len(r['ranks']) and len(set(r['ranks'])) == len(r['ranks']), "Invalid CPU cold ranks")
            for e,rank in zip(r['ids'],r['ranks']):
                require(0 <= rank < len(direct['ids']) and direct['ids'][rank] == e and direct['direct_slots'][rank] < 0 and e in direct['cold_ids'], "CPU stole a hot/duplicate contribution")
                direct['cold_ids'].remove(e)
            acquired.setdefault(r['layer'], set()).update(r['ids'])
            forward['cpu_tasks'] = forward.get('cpu_tasks',0) + len(r['ids'])
        elif kind == "forward_begin":
            require(forward is None and acquire is None and direct is None, "Unclosed forward")
            forward = dict(r, hits=0, misses=0, bytes=0, host_hits=0, file_fills=0,
                           acquire_ns=0, fill_ns=0, prefetch_wait_ns=0, layers={},
                           direct_entries=0, resident_route_entries=0, cold_request_entries=0, acquire_calls=0)
            routed, acquired = {}, {}
        elif kind == "route":
            require(forward is not None and acquire is None and r["forward"] == forward["forward"], "Orphan route")
            require(r["rows"] == forward["tokens"] and 0 < r["top_k"] <= experts, "Invalid route shape")
            require(len(r["ids"]) == r["rows"]*r["top_k"] and all(0 <= e < experts for e in r["ids"]), "Invalid route IDs")
            require(0 <= r["layer"] < cfg["layers"] and r["layer"] not in routed, "Duplicate/invalid route layer")
            require(all(len(set(r["ids"][i:i+r["top_k"]])) == r["top_k"] for i in range(0,len(r["ids"]),r["top_k"])), "Duplicate expert within a token")
            routed[r["layer"]] = set(r["ids"])
        elif kind == "direct_bind":
            require(forward is not None and acquire is None and direct is None, "Overlapping direct lease")
            require(r["forward"] == forward["forward"] and 0 <= r["layer"] < cfg["layers"], "Invalid direct identity")
            keys = [r["layer"] * experts + e for e in r["ids"]]
            require(all(0 <= e < experts for e in r["ids"]), "Invalid direct expert")
            require(len(keys) == len(r["initial_slots"]) == len(r["direct_slots"]), "Invalid direct arrays")
            require([resident.get(k, -1) for k in keys] == r["initial_slots"], "Direct resident map mismatch")
            require(all(d == -1 or d == s >= 0 for d,s in zip(r["direct_slots"],r["initial_slots"])), "Invalid direct binding")
            cold = [e for e,d in zip(r["ids"],r["direct_slots"]) if d < 0]
            require(cold == r["cold_ids"], "Direct/cold partition mismatch")
            direct = dict(r, cold_ids=list(r['cold_ids']), cold_acquired=False)
            leased = {s for s in r["direct_slots"] if s >= 0}
            for s in dict.fromkeys(r["initial_slots"]):
                if s >= 0: lru.move_to_end(s)
            ready = [e for e,d in zip(r["ids"],r["direct_slots"]) if d >= 0]
            acquired.setdefault(r["layer"], set()).update(ready)
            layer = forward["layers"].setdefault(r["layer"], defaultdict(int))
            for name,value in (("hits",len(ready)), ("direct_entries",len(ready)),
                               ("resident_route_entries",sum(s >= 0 for s in r["initial_slots"])),
                               ("cold_request_entries",len(cold))):
                forward[name] += value; layer[name] += value
        elif kind == "direct_release":
            require(direct and acquire is None and r["forward"] == forward["forward"] and r["layer"] == direct["layer"], "Orphan direct release")
            require(not direct["cold_ids"] or direct["cold_acquired"], "Missing cold transaction")
            direct, leased = None, set()
        elif kind == "acquire_begin":
            require(forward is not None and acquire is None, "Acquire outside forward")
            require(not r["pinned"], "Pinned traces require a pin event schema")
            require(r["forward"] == forward["forward"], "Wrong forward id")
            acquire = r
            if not r.get("touch_hits",1) and not epoch_loading:
                require(direct and not direct["cold_acquired"] and r["layer"] == direct["layer"] and r["ids"] == direct["cold_ids"], "Wrong cold transaction")
                direct["cold_acquired"] = True
            forward["acquire_calls"] += 1
            acquired.setdefault(r["layer"], set()).update(r["ids"])
            keys = [r["layer"] * experts + e for e in r["ids"]]
            require(len(keys) == len(r["initial_slots"]) == len(r["host_states"]), "Invalid arrays")
            require(all(0 <= e < experts for e in r["ids"]), "Invalid expert")
            require([resident.get(k, -1) for k in keys] == r["initial_slots"], "Resident map mismatch")
            acquire["expected_hits"] = sum(s >= 0 for s in r["initial_slots"])
            acquire["expected_misses"] = len({k for k in keys if k not in resident})
            acquire["victims"] = 0
            reserved = set(leased)
            # All hits are touched/reserved before any miss gets a slot.
            for s in r["initial_slots"]:
                if s >= 0 and s not in reserved:
                    reserved.add(s)
                    if r.get("touch_hits",1): lru.move_to_end(s)
        elif kind == "victim":
            require(acquire and r["acquire"] == acquire["acquire"], "Orphan victim")
            eligible = [s for s in lru if s not in reserved and s not in frozen]
            c, backed = r["candidates"], r["backed"]
            require(len(c)==len(backed) and len(c)<=8, 'Invalid candidate observations')
            if r.get('transient_ring'):
                require(cold_ring and not c and not backed, 'Invalid transient ring')
                chosen = None
                for _ in cold_ring:
                    candidate = cold_ring[cold_cursor]
                    cold_cursor = (cold_cursor+1)%len(cold_ring)
                    if candidate not in reserved:
                        chosen = candidate
                        break
                require(chosen is not None and chosen not in frozen, 'Transient ring exhausted')
            elif cfg["cost_aware"] and cfg["host_slots"]:
                require(c == eligible[:len(c)] and len(c) <= 8 and len(c) == len(backed), "LRU candidate window mismatch")
                require(bool(c), "Missing cost-aware observations")
                chosen = c[0]
                for i, s in enumerate(c):
                    if slots[s] < 0 or backed[i]:
                        chosen = s
                        require(i == len(c) - 1, "Observed search continued after match")
                        break
                else:
                    require(len(c) == min(8, len(eligible)), "Short candidate window")
            else:
                require(not c, "Unexpected cost-aware candidates")
                chosen = eligible[0]
            require(r["slot"] == chosen and r["old_key"] == slots[chosen], "Victim mismatch")
            require(r["key"] in {acquire["layer"] * experts + e for e in acquire["ids"]}, "Unrequested binding")
            require(r["key"] not in resident, "Duplicate binding")
            if slots[chosen] >= 0:
                del resident[slots[chosen]]
            slots[chosen] = r["key"]
            resident[r["key"]] = chosen
            reserved.add(chosen)
            lru.move_to_end(chosen)
            acquire["victims"] += 1
        elif kind == "acquire_end":
            require(acquire and r["acquire"] == acquire["acquire"], "Orphan end")
            require(r["hits"] == acquire["expected_hits"], "Hit mismatch")
            require(r["misses"] == acquire["expected_misses"] == acquire["victims"], "Miss mismatch")
            require(r["bytes"] == r["misses"] * cfg["slot_bytes"], "Payload mismatch")
            layer = forward["layers"].setdefault(acquire["layer"], defaultdict(int))
            for field in ("hits", "misses", "bytes", "host_hits", "file_fills", "fill_ns", "prefetch_wait_ns"):
                forward[field] += r[field]
                layer[field] += r[field]
            forward["acquire_ns"] += r["ns"] - r["begin_ns"]
            layer["acquire_ns"] += r["ns"] - r["begin_ns"]
            acquire, reserved = None, set()
        elif kind == "demote":
            require(acquire is None, "Demote inside acquire")
            for e in r["ids"]:
                s = resident.get(r["layer"] * experts + e)
                if s is not None:
                    lru.move_to_end(s, last=False)
        elif kind == "forward_end":
            require(forward and r["forward"] == forward["forward"] and acquire is None and direct is None, "Wrong forward end")
            if routed:
                require(set(routed) == set(range(cfg["layers"])) and routed == acquired, "Route/demand coverage mismatch")
            forward["host_span_ns"] = r["ns"] - forward["ns"]
            totals.append(forward)
            forward = None
        elif kind == "host":
            require(forward and r["forward"] == forward["forward"], "Orphan host interval")
        elif kind == "device":
            require(forward and r['forward']==forward['forward'] and r['clock']=='sycl_event' and r['end']>=r['start'], "Invalid device interval")
        else:
            raise ValueError(f"Unknown event {kind}")
    require(forward is None and acquire is None and direct is None and not pending and not epoch_loading and totals, "Truncated trace")
    return totals


def admission_rows(rows):
    """Counterfactuals consume full routes, never only the observed cold subset."""
    direct = False
    for row in rows[1:]:
        if row["type"] == "direct_bind":
            direct = True
            yield dict(row, type="acquire_begin")
        elif row["type"] == "direct_release":
            direct = False
        elif row["type"] == "acquire_begin" and direct:
            continue
        else:
            yield row


def simulate(rows, protected_fraction=0):
    """SLRU: resident re-use promotes, demote cannot eject a protected key.

    Capacity stays fixed, no static per-layer allotment, all hits are reserved
    before misses. A large batch may overflow the probationary area and safely
    evict the oldest unreserved protected entry. No host cost preference.
    """
    cap, experts = rows[0]["slots"], rows[0]["experts"]
    protected_cap = int(cap * protected_fraction)
    lru = OrderedDict((s, None) for s in reversed(range(cap)))
    protected, resident = OrderedDict(), {}
    slots = [-1] * cap
    output = []
    current = None
    for r in admission_rows(rows):
        if r["type"] == "forward_begin":
            current = dict(forward=r["forward"], position=r["position"], tokens=r["tokens"], hits=0, misses=0)
        elif r["type"] == "acquire_begin":
            keys = [r["layer"] * experts + e for e in r["ids"]]
            batch = set(keys)
            require(len(batch) <= cap, "Batch exceeds capacity")
            # Match runtime hit accounting for duplicates; miss accounting is unique.
            current["hits"] += sum(k in resident for k in keys)
            reserved = {resident[k] for k in keys if k in resident}
            for k in dict.fromkeys(keys):
                if k in resident:
                    s = resident[k]
                    lru.move_to_end(s)
                    if protected_cap:
                        protected[s] = None
                        protected.move_to_end(s)
                        if len(protected) > protected_cap:
                            protected.popitem(last=False)
            for k in dict.fromkeys(keys):
                if k in resident:
                    continue
                victim = next((v for v in lru if v not in reserved and v not in protected), None)
                if victim is None:
                    victim = next(v for v in lru if v not in reserved)
                if slots[victim] >= 0:
                    del resident[slots[victim]]
                protected.pop(victim, None)
                slots[victim] = k
                resident[k] = victim
                reserved.add(victim)
                lru.move_to_end(victim)
                current["misses"] += 1
        elif r["type"] == "demote":
            for e in r["ids"]:
                k = r["layer"] * experts + e
                s = resident.get(k)
                if s is not None and s not in protected:
                    lru.move_to_end(s, last=False)
        elif r["type"] == "forward_end":
            current["bytes"] = current["misses"] * rows[0]["slot_bytes"]
            output.append(current)
    return output


def report(path):
    rows = read_trace(path)
    observed = validate(rows)
    return {
        "schema": 1, "trace": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "config": rows[0], "validation": "all observed hits, victims, demotes and bytes matched",
        "observed": observed,
        "counterfactual_scope": "GPU byte counts only; no host cache, prefetch, DMA fence or timing prediction",
        "lru_demote_no_cost": simulate(rows),
        "slru75_demote_no_cost": simulate(rows, .75),
        "offline_farthest_next_use": offline_reference(rows),
    }


def offline_reference(rows):
    """Uniform-slot farthest-next-use reference with whole-batch reservations.

    Uses future route knowledge. This is a GPU transfer lower-bound reference,
    not an online policy or a prediction of feasible host I/O latency.
    """
    cap, experts = rows[0]["slots"], rows[0]["experts"]
    future = defaultdict(deque)
    batches = []
    for r in admission_rows(rows):
        if r["type"] == "acquire_begin":
            batch = [r["layer"] * experts + e for e in r["ids"]]
            for key in set(batch):
                future[key].append(len(batches))
            batches.append(batch)
    resident, heap, output = {}, [], []
    pos = 0
    current = None
    for r in admission_rows(rows):
        if r["type"] == "forward_begin":
            current = dict(forward=r["forward"], position=r["position"], tokens=r["tokens"], hits=0, misses=0)
        elif r["type"] == "acquire_begin":
            batch = batches[pos]
            reserved = set(batch)
            require(len(reserved) <= cap, "Batch exceeds capacity")
            current["hits"] += sum(k in resident for k in batch)
            for k in dict.fromkeys(batch):
                require(future[k].popleft() == pos, "Future-use indexing error")
                if k in resident:
                    resident[k] = future[k][0] if future[k] else len(batches)
                    heapq.heappush(heap, (-resident[k], k))
            for k in dict.fromkeys(batch):
                if k in resident:
                    continue
                if len(resident) == cap:
                    skipped = []
                    while heap:
                        nxt, victim = heapq.heappop(heap)
                        if resident.get(victim) != -nxt:
                            continue
                        if victim in reserved:
                            skipped.append((nxt, victim))
                            continue
                        del resident[victim]
                        break
                    else:
                        raise ValueError("No offline reference victim")
                    for item in skipped:
                        heapq.heappush(heap, item)
                resident[k] = future[k][0] if future[k] else len(batches)
                heapq.heappush(heap, (-resident[k], k))
                current["misses"] += 1
            pos += 1
        elif r["type"] == "forward_end":
            current["bytes"] = current["misses"] * rows[0]["slot_bytes"]
            output.append(current)
    return output


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    result = report(args.trace)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    for label in ("observed", "lru_demote_no_cost", "slru75_demote_no_cost", "offline_farthest_next_use"):
        groups = defaultdict(lambda: [0, 0, 0])
        for row in result[label]:
            key = f"prefill{row['tokens']}" if row["tokens"] > 1 else "decode"
            for i, field in enumerate(("hits", "misses", "bytes")):
                groups[key][i] += row[field]
        print(label, dict(groups))
