"""Validate/merge Aila identity-bound profiles and measure held-out route coverage."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    lines = Path(path).read_text(encoding='utf-8-sig').splitlines()
    if lines[0] != 'AILA_Q4E_HOTSET 1':
        raise ValueError('profile version')
    layers, experts, count = map(int, lines[2].split())
    rows = [tuple(map(int, line.split())) for line in lines[3:]]
    if count != layers*experts or len(rows) != count:
        raise ValueError('profile dimensions')
    if any(len(r) != 4 or min(r) < 0 for r in rows) or {r[0] for r in rows} != set(range(count)):
        raise ValueError('profile keys/counts')
    return dict(identity=lines[1], layers=layers, experts=experts, rows=rows)


def merge(paths, output):
    profiles = [read(p) for p in paths]
    base = profiles[0]
    sums = [[0, 0, 0] for _ in base['rows']]
    for p in profiles:
        if (p['identity'], p['layers'], p['experts']) != (base['identity'], base['layers'], base['experts']):
            raise ValueError('profile model mismatch')
        for key, *counts in p['rows']:
            sums[key] = [a+b for a,b in zip(sums[key], counts)]
    order = sorted(range(len(sums)), key=lambda k: (-sums[k][0], -sums[k][1], k))
    body = ['AILA_Q4E_HOTSET 1', base['identity'], f"{base['layers']} {base['experts']} {len(order)}"]
    body += [' '.join(map(str, [k]+sums[k])) for k in order]
    Path(output).write_text('\n'.join(body)+'\n', encoding='utf-8')


def coverage(profile, traces, slots):
    hot = {r[0] for r in profile['rows'][:slots]}
    result = []
    for path in traces:
        phases = {p:dict(entries=0, hits=0, layers_all_hot=0, layers=0) for p in ('prefill','decode')}
        active = None
        real_request = False
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            r = json.loads(line)
            if r['type'] == 'forward_begin':
                active = r
                if r['position']==0 and r['tokens']>2:
                    real_request = True
            if r['type'] != 'route' or active is None or not real_request:
                continue
            phase = 'decode' if r['rows']==1 else 'prefill'
            hits = sum(r['layer']*profile['experts']+e in hot for e in r['ids'])
            d = phases[phase]
            d['entries'] += len(r['ids']); d['hits'] += hits; d['layers'] += 1
            d['layers_all_hot'] += hits == len(r['ids'])
        for d in phases.values():
            d['coverage'] = d['hits']/d['entries'] if d['entries'] else None
        result.append(dict(path=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(), phases=phases))
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('profiles', nargs='+'); ap.add_argument('--merge-output')
    ap.add_argument('--traces', nargs='*', default=[]); ap.add_argument('--slots', type=int, default=3055)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    if args.merge_output:
        merge(args.profiles, args.merge_output)
    profile = read(args.merge_output or args.profiles[0])
    result = dict(identity=profile['identity'], slots=args.slots,
        sources=[dict(path=p,sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest()) for p in args.profiles],
        per_layer_slots=[sum(k//profile['experts']==l for k,*_ in profile['rows'][:args.slots]) for l in range(profile['layers'])],
        coverage=coverage(profile,args.traces,args.slots))
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
