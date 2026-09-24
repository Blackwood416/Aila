"""Replays Qwen3.8-Flash-Next expert-acquire traces under several eviction policies.

Record a trace with AILA_DEBUG_Q4E_ROUTE_TRACE=<file.bin> (int32 records
{layer, count, ids[count]}, one per Qwen4ExpExpertCache::acquire call), then:

    python perf/qwen4exp/q4e_cache_sim.py tmp/route_essay.bin tmp/route_code.bin --cap 3463

Every call's experts must be resident at once (they may not evict each other), as in
acquire(). Hit rates are reported over decode calls (count == top_k) only. LRU matches the
production cache (validated: 72.7% simulated vs 72.9% measured on a 400-token essay).
"""
import argparse
import heapq
from collections import OrderedDict, defaultdict

import numpy as np


def load(path):
    raw = np.fromfile(path, dtype=np.int32)
    calls, i = [], 0
    while i < len(raw):
        layer, count = int(raw[i]), int(raw[i + 1])
        calls.append((layer, [int(x) for x in raw[i + 2:i + 2 + count]]))
        i += 2 + count
    return calls


def prepare(calls, top_k):
    batches = [[(layer << 16) | e for e in ids] for layer, ids in calls]
    decode = [len(ids) == top_k for _, ids in calls]
    return batches, decode


def _evict_lru(od, batch):
    for victim in od:
        if victim not in batch:
            del od[victim]
            return
    raise RuntimeError('cache smaller than one batch')


def sim_lru(batches, decode, cap):
    lru = OrderedDict()
    hits = total = 0
    for b, d in zip(batches, decode):
        for k in b:
            if k in lru:
                lru.move_to_end(k)
                hits += d
            else:
                if len(lru) >= cap:
                    _evict_lru(lru, b)
                lru[k] = True
            total += d
    return hits / total


def sim_lru_per_layer(batches, decode, cap, layers=48):
    per = cap // layers
    lrus = defaultdict(OrderedDict)
    hits = total = 0
    for b, d in zip(batches, decode):
        for k in b:
            lru = lrus[k >> 16]
            if k in lru:
                lru.move_to_end(k)
                hits += d
            else:
                if len(lru) >= per:
                    _evict_lru(lru, b)
                lru[k] = True
            total += d
    return hits / total


def sim_slru(batches, decode, cap, protected_frac=0.8):
    """Segmented LRU: first touch -> probation; re-touch -> protected."""
    prot_cap = int(cap * protected_frac)
    prob, prot = OrderedDict(), OrderedDict()
    hits = total = 0
    for b, d in zip(batches, decode):
        bset = set(b)
        for k in b:
            if k in prot:
                prot.move_to_end(k)
                hits += d
            elif k in prob:
                del prob[k]
                prot[k] = True
                hits += d
                while len(prot) > prot_cap:
                    old, _ = prot.popitem(last=False)
                    prob[old] = True
            else:
                if len(prob) + len(prot) >= cap:
                    if any(v not in bset for v in prob):
                        _evict_lru(prob, bset)
                    else:
                        _evict_lru(prot, bset)
                prob[k] = True
            total += d
    return hits / total


def _heap_victim(heap, is_current, bset):
    skipped = []
    while True:
        item = heapq.heappop(heap)
        if not is_current(item):
            continue  # stale entry
        if item[-1] in bset:
            skipped.append(item)
            continue
        for s in skipped:
            heapq.heappush(heap, s)
        return item[-1]


def sim_lfu(batches, decode, cap, decay=1.0):
    """Evict the lowest exponentially decayed use count (decay applied per acquire call)."""
    score = defaultdict(float)
    last = {}
    resident = set()
    heap = []
    hits = total = t = 0
    scale = 1.0
    for b, d in zip(batches, decode):
        t += 1
        scale *= decay
        bset = set(b)
        for k in b:
            score[k] += 1.0 / scale
            last[k] = t
            if k in resident:
                hits += d
            else:
                if len(resident) >= cap:
                    v = _heap_victim(heap, lambda it: it[2] in resident and it[0] == score[it[2]] and it[1] == last[it[2]], bset)
                    resident.discard(v)
                resident.add(k)
            heapq.heappush(heap, (score[k], last[k], k))
            total += d
    return hits / total


def sim_belady(batches, decode, cap):
    """Offline optimum: evict the resident expert whose next use is farthest away."""
    flat = [k for b in batches for k in b]
    nxt = [0] * len(flat)
    seen = {}
    for i in range(len(flat) - 1, -1, -1):
        nxt[i] = seen.get(flat[i], 1 << 60)
        seen[flat[i]] = i
    resident = {}
    heap = []
    hits = total = pos = 0
    for b, d in zip(batches, decode):
        bset = set(b)
        for k in b:
            nu = nxt[pos]
            pos += 1
            if k in resident:
                hits += d
            elif len(resident) >= cap:
                v = _heap_victim(heap, lambda it: it[1] in resident and resident[it[1]] == -it[0], bset)
                del resident[v]
            resident[k] = nu
            heapq.heappush(heap, (-nu, k))
            total += d
    return hits / total


POLICIES = {
    'lru': sim_lru,
    'lru_per_layer': sim_lru_per_layer,
    'slru': sim_slru,
    'lfu_0.999': lambda b, d, c: sim_lfu(b, d, c, 0.999),
    'lfu_0.998': lambda b, d, c: sim_lfu(b, d, c, 0.998),
    'belady': sim_belady,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('traces', nargs='+')
    ap.add_argument('--cap', type=int, nargs='+', default=[3463], help='slot counts to simulate')
    ap.add_argument('--top-k', type=int, default=10)
    ap.add_argument('--policies', default=','.join(POLICIES))
    args = ap.parse_args()
    for path in args.traces:
        batches, decode = prepare(load(path), args.top_k)
        print(f'{path}: {len(batches)} calls, ~{sum(decode) // 48} decode tokens', flush=True)
        for cap in args.cap:
            row = [f'  cap={cap:5d}']
            for name in args.policies.split(','):
                row.append(f'{name}={POLICIES[name](batches, decode, cap):.4f}')
            print('  '.join(row), flush=True)


if __name__ == '__main__':
    main()
