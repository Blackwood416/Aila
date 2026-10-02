"""Bound the traffic of Strata-style stream-all from a target-only cache trace."""
import argparse
import hashlib
import json
from pathlib import Path


def analyze(path):
    forwards, acquisitions, config = {}, {}, None
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for line in f:
            digest.update(line)
            r = json.loads(line)
            if r['type'] == 'config':
                assert config is None, 'Mixed caches are not supported'
                config = r
            elif r['type'] == 'forward_begin':
                forwards[r['forward']] = dict(begin=r, ids=set(), demand_bytes=0, file_fills=0)
            elif r['type'] == 'acquire_begin':
                acquisitions[r['acquire']] = r['forward']
                out = forwards[r['forward']]
                out['ids'].update(r['layer']*config['experts']+e for e in r['ids'])
            elif r['type'] == 'acquire_end':
                out = forwards[acquisitions.pop(r['acquire'])]
                out['demand_bytes'] += r['bytes']
                out['file_fills'] += r['file_fills']
            elif r['type'] == 'forward_end':
                forwards[r['forward']]['end'] = r
    assert config and config['layers'] == 48 and not acquisitions
    total = config['layers']*config['experts']
    lower = max(0, total-config['slots'])*config['slot_bytes']
    output = []
    for out in forwards.values():
        assert 'end' in out
        if out['begin']['tokens'] <= 4: continue
        n = len(out['ids'])
        output.append(dict(position=out['begin']['position'], tokens=out['begin']['tokens'],
            unique_layer_experts=n, coverage=n/total, demand_bytes=out['demand_bytes'],
            demand_file_fills=out['file_fills'], minimum_stream_all_bytes=lower,
            minimum_stream_all_vs_observed_demand=lower/out['demand_bytes'] if out['demand_bytes'] else None,
            full_catalog_bytes=total*config['slot_bytes']))
    return dict(path=str(path), sha256=digest.hexdigest(), config=config, forwards=output,
        caveat='Traffic lower bound assumes every initial GPU slot avoids one transfer and no extra evictions. Not a stream-all runtime simulation. Existing demand bytes include actual expert H2D, not speculative host reads. Coverage is per real chunk; do not extrapolate its threshold to other prompts.')


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('traces', nargs='+');ap.add_argument('--output', required=True)
    a=ap.parse_args();result=[analyze(p) for p in a.traces]
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    for r in result:print(r['path'],r['forwards'])
