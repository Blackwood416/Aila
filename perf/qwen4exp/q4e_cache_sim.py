"""Legacy binary-trace exploration; not a replay of the current runtime.

For explicit forward phases, demote, eight-candidate cost-aware decisions and
observed hit/byte validation, use q4e_cache_replay.py with AILA_Q4E_CACHE_TRACE.
The legacy format cannot model current asynchronous host readiness or fences.

Record a trace with AILA_DEBUG_Q4E_ROUTE_TRACE=<file.bin> (int32 records
{layer, count, ids[count]}, one per Qwen4ExpExpertCache::acquire call), then:

    python perf/qwen4exp/q4e_cache_sim.py tmp/route_essay.bin tmp/route_code.bin --cap 3463

Every call's experts must be resident at once (they may not evict each other), as in
acquire(). Here count == top_k is only a legacy decode heuristic. An old 400-token
essay matched LRU at 72.7% vs 72.9%; that is not current-policy validation.
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


def sim_twotier(batches, decode, gpu_cap, host_cap, cost_aware=True, slot_bytes=1173248, protected_frac=0.8):
    """Simulates a two-tier GPU + Host RAM cache.
    GPU cache: capacity `gpu_cap`, LRU eviction.
    Host RAM cache: capacity `host_cap`, SLRU eviction (probationary + protected).
    Cost-aware eviction (B4): when evicting from GPU, preferentially select victims
    already resident in Host RAM (so future reloads are H2D copy rather than disk mmap).
    """
    gpu_lru = OrderedDict()
    prot_cap = int(host_cap * protected_frac)
    host_prob = OrderedDict()
    host_prot = OrderedDict()

    def in_host(k):
        return k in host_prob or k in host_prot

    def touch_host(k):
        nonlocal host_prob, host_prot
        if k in host_prot:
            host_prot.move_to_end(k)
        elif k in host_prob:
            del host_prob[k]
            host_prot[k] = True
            while len(host_prot) > prot_cap:
                old, _ = host_prot.popitem(last=False)
                host_prob[old] = True
        else:
            if len(host_prob) + len(host_prot) >= host_cap:
                if host_prob:
                    host_prob.popitem(last=False)
                elif host_prot:
                    host_prot.popitem(last=False)
            host_prob[k] = True

    gpu_hits = gpu_misses = 0
    host_hits = host_misses = 0
    gpu_evictions = 0
    cost_aware_hits = 0

    for b, d in zip(batches, decode):
        bset = set(b)
        to_insert = []
        for k in b:
            if k in gpu_lru:
                gpu_lru.move_to_end(k)
                if d: gpu_hits += 1
            else:
                if d: gpu_misses += 1
                if in_host(k):
                    if d: host_hits += 1
                    touch_host(k)
                else:
                    if d: host_misses += 1
                    touch_host(k)
                to_insert.append(k)

        for k in to_insert:
            if k in gpu_lru:
                gpu_lru.move_to_end(k)
                continue
            if len(gpu_lru) >= gpu_cap:
                victim = None
                if cost_aware:
                    for cand in gpu_lru:
                        if cand not in bset and in_host(cand):
                            victim = cand
                            cost_aware_hits += 1
                            break
                if victim is None:
                    for cand in gpu_lru:
                        if cand not in bset:
                            victim = cand
                            break
                if victim is None:
                    raise RuntimeError("GPU cache capacity smaller than single batch")
                del gpu_lru[victim]
                if d: gpu_evictions += 1
            gpu_lru[k] = True

    total = gpu_hits + gpu_misses
    gpu_hit_rate = gpu_hits / total if total > 0 else 0.0
    host_total = host_hits + host_misses
    host_hit_rate = host_hits / host_total if host_total > 0 else 0.0
    disk_reads = host_misses
    h2d_transfers = gpu_misses

    return {
        'gpu_hits': gpu_hits,
        'gpu_misses': gpu_misses,
        'gpu_hit_rate': gpu_hit_rate,
        'host_hits': host_hits,
        'host_misses': host_misses,
        'host_hit_rate': host_hit_rate,
        'combined_hit_rate': (gpu_hits + host_hits) / total if total > 0 else 0.0,
        'gpu_evictions': gpu_evictions,
        'cost_aware_evictions': cost_aware_hits,
        'disk_reads': disk_reads,
        'h2d_transfers': h2d_transfers,
        'disk_read_bytes': disk_reads * slot_bytes,
        'h2d_bytes': h2d_transfers * slot_bytes,
    }


POLICIES = {
    'lru': sim_lru,
    'lru_per_layer': sim_lru_per_layer,
    'slru': sim_slru,
    'lfu_0.999': lambda b, d, c: sim_lfu(b, d, c, 0.999),
    'lfu_0.998': lambda b, d, c: sim_lfu(b, d, c, 0.998),
    'belady': sim_belady,
}


def run_self_test():
    print("Running q4e_cache_sim self-test...")
    import random
    random.seed(42)
    # Generate 50 synthetic decode calls, 48 layers, top_k = 10 from 512 experts
    batches = []
    decode = []
    for _ in range(50):
        for layer in range(48):
            # Zipf-like distribution over experts
            pool = list(range(64)) + list(range(512))
            chosen = random.sample(pool, 10)
            batches.append([(layer << 16) | e for e in chosen])
            decode.append(True)

    gpu_cap = 500
    host_cap = 2000

    res_cost = sim_twotier(batches, decode, gpu_cap, host_cap, cost_aware=True)
    res_nocost = sim_twotier(batches, decode, gpu_cap, host_cap, cost_aware=False)

    print(f"Cost-aware: GPU hit={res_cost['gpu_hit_rate']:.4f}, Host hit={res_cost['host_hit_rate']:.4f}, "
          f"Combined={res_cost['combined_hit_rate']:.4f}, Disk={res_cost['disk_read_bytes']/(1024*1024):.1f}MB, "
          f"Cost-aware evictions={res_cost['cost_aware_evictions']}")
    print(f"No-cost:    GPU hit={res_nocost['gpu_hit_rate']:.4f}, Host hit={res_nocost['host_hit_rate']:.4f}, "
          f"Combined={res_nocost['combined_hit_rate']:.4f}, Disk={res_nocost['disk_read_bytes']/(1024*1024):.1f}MB, "
          f"Cost-aware evictions={res_nocost['cost_aware_evictions']}")

    assert res_cost['gpu_hits'] + res_cost['gpu_misses'] == len(batches) * 10
    assert res_cost['cost_aware_evictions'] > 0, "Cost aware evictions should be triggered"
    assert res_cost['disk_read_bytes'] <= res_nocost['disk_read_bytes'] * 1.05, "Cost aware should not substantially increase disk reads"
    print("Self-test passed!")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('traces', nargs='*')
    ap.add_argument('--cap', type=int, nargs='+', default=[3463], help='slot counts to simulate')
    ap.add_argument('--gpu-cap', type=int, default=None, help='GPU slot count for twotier simulation')
    ap.add_argument('--host-cap', type=int, default=14336, help='Host RAM slot count for twotier simulation')
    ap.add_argument('--cost-aware', action='store_true', default=True, help='Enable cost-aware GPU eviction')
    ap.add_argument('--no-cost-aware', dest='cost_aware', action='store_false', help='Disable cost-aware GPU eviction')
    ap.add_argument('--twotier', action='store_true', help='Run two-tier simulation')
    ap.add_argument('--slot-bytes', type=int, default=1173248, help='Bytes per expert slot')
    ap.add_argument('--top-k', type=int, default=10)
    ap.add_argument('--policies', default=','.join(POLICIES))
    ap.add_argument('--self-test', action='store_true', help='Run internal self-tests')
    args = ap.parse_args()

    if args.self_test:
        run_self_test()
        return

    if not args.traces:
        print("No traces provided. Run with --self-test to verify.")
        return

    for path in args.traces:
        batches, decode = prepare(load(path), args.top_k)
        print(f'{path}: {len(batches)} calls, ~{sum(decode) // 48} decode tokens', flush=True)
        for cap in args.cap:
            row = [f'  cap={cap:5d}']
            for name in args.policies.split(','):
                row.append(f'{name}={POLICIES[name](batches, decode, cap):.4f}')
            print('  '.join(row), flush=True)

            if args.twotier or args.gpu_cap is not None:
                gcap = args.gpu_cap if args.gpu_cap is not None else cap
                res = sim_twotier(batches, decode, gcap, args.host_cap,
                                 cost_aware=args.cost_aware, slot_bytes=args.slot_bytes)
                print(f'    twotier (gpu={gcap}, host={args.host_cap}, cost_aware={args.cost_aware}): '
                      f'GPU hit={res["gpu_hit_rate"]:.4f}, Host hit={res["host_hit_rate"]:.4f}, '
                      f'Combined={res["combined_hit_rate"]:.4f}, Disk={res["disk_read_bytes"]/(1024*1024):.1f}MB, '
                      f'H2D={res["h2d_bytes"]/(1024*1024):.1f}MB, CostEvictions={res["cost_aware_evictions"]}',
                      flush=True)


if __name__ == '__main__':
    main()
