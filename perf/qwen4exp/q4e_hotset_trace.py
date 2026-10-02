"""Check hot/cold transaction coverage and host submission order in H1/H2 traces.

Host timestamps establish enqueue order, not GPU execution overlap or device latency.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics as st

from q4e_cache_replay import read_trace, validate


def summarize(path, last_forwards=16, min_position=41):
    rows = read_trace(path)
    observed = validate(copy.deepcopy(rows))
    eligible = [r for r in observed if r['tokens'] == 1 and r['position'] >= min_position]
    selected = eligible[-last_forwards:]
    assert len(selected) == last_forwards
    ids = {r['forward'] for r in selected}
    layers = {}
    for row in rows:
        if row.get('forward') not in ids or 'layer' not in row:
            continue
        entry = layers.setdefault((row['forward'], row['layer']), {})
        if row['type'] in ('direct_bind', 'direct_release', 'acquire_begin'):
            assert row['type'] not in entry
            entry[row['type']] = row
        elif row['type'] == 'host':
            entry[row['name']] = row
    assert len(layers) == 48 * last_forwards
    gaps, ahead = [], []
    ready = cold = all_hot = mixed = all_cold = 0
    for entry in layers.values():
        bind = entry['direct_bind']
        hot = sum(s >= 0 for s in bind['direct_slots'])
        missing = len(bind['cold_ids'])
        assert hot + missing == 10 and 'direct_release' in entry
        ready += hot
        cold += missing
        all_hot += missing == 0
        all_cold += hot == 0
        mixed += hot > 0 and missing > 0
        submit = entry.get('resident_gpu_submit') or entry.get('resident_submit')
        if 'resident_gpu_submit' in entry:
            lead = entry['router_wait']['begin_ns'] - submit['end_ns']
            assert lead >= 0, 'H2 hot submission occurred after host route wait began'
            ahead.append(lead / 1000)
        if missing:
            acquire = entry['acquire_begin']
            assert acquire['ids'] == bind['cold_ids'] and acquire['touch_hits'] == 0
            if submit and hot:
                gap = acquire['ns'] - submit['end_ns']
                assert gap >= 0, 'Cold service began before hot enqueue completed'
                gaps.append(gap / 1000)
        else:
            assert 'acquire_begin' not in entry, 'All-hot created a cold transaction'
    def spread(values):
        values = sorted(values)
        return dict(samples=len(values), median=st.median(values), p10=values[len(values)//10],
                    p90=values[len(values)*9//10]) if values else None
    return dict(trace=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                validation='all observed routes, resident bindings, leases, cold transactions, victims and bytes matched',
                forward_ids=sorted(ids), layers=len(layers), direct_entries=ready, cold_entries=cold,
                all_hot_layers=all_hot, mixed_layers=mixed, all_cold_layers=all_cold,
                cold_acquire_calls=mixed+all_cold,
                hot_enqueued_before_cold_service_us=spread(gaps),
                hot_enqueued_before_route_host_wait_us=spread(ahead),
                gpu_dispatch_gap_measured=False, gpu_overlap_measured=False,
                caveat='Host enqueue timing only. H2 router_wait also includes the subsequent host lease binding; no GPU wait time is inferred.',
                observed=selected)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('trace')
    ap.add_argument('--last-forwards', type=int, default=16)
    ap.add_argument('--min-position', type=int, default=41)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    result = summarize(args.trace, args.last_forwards, args.min_position)
    Path(args.output).write_text(json.dumps(result, indent=2) + chr(10), encoding='utf-8')
    print({key: value for key, value in result.items() if key != 'observed'})
