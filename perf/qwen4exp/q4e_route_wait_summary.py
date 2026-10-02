"""Identify real small-row expert batches that mix resident and missing slots."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics as st
from q4e_cache_replay import read_trace, validate


def summarize(path, prompt=122, tokens=24, sweep=None):
    rows=read_trace(path);observed=validate(rows)
    selected={r['forward'] for r in observed if r['tokens']==4 and
              prompt <= r['position'] < prompt+tokens and (r['position']-prompt)%4==0}
    if sweep is not None:
        frames=tokens//4
        assert frames>0 and tokens%4==0 and len(selected)>=(sweep+1)*frames
        selected=set(sorted(selected)[sweep*frames:(sweep+1)*frames])
    routes={};acquires=[];mixed=[];hist=Counter();union=[]
    for r in rows:
        if r.get('forward') not in selected:continue
        if r['type']=='route':
            freq=Counter(r['ids']);routes[r['forward'],r['layer']]=freq
            hist.update(freq.values());union.append(len(freq))
        elif r['type']=='acquire_begin':
            hit=[e for e,s in zip(r['ids'],r['initial_slots']) if s>=0]
            miss=[e for e,s in zip(r['ids'],r['initial_slots']) if s<0]
            a=dict(forward=r['forward'],layer=r['layer'],resident=len(hit),missing=len(miss),
                resident_acts=sum(routes[r['forward'],r['layer']][e] for e in hit))
            acquires.append(a)
            if hit and miss:mixed.append(a)
    assert routes and len(routes)==48*len(selected)
    return dict(trace=str(path),sweep=sweep,forward_ids=sorted(selected),layers=len(routes),batches=len(acquires),
        mixed_batches=len(mixed),resident_experts_in_mixed=sum(a['resident'] for a in mixed),
        resident_acts_in_mixed=sum(a['resident_acts'] for a in mixed),
        layer_union_mean=st.mean(union),expert_token_multiplicity=dict(hist),acquires=acquires,
        caveat='Resident is slot presence, not DMA completion. A mixed batch guards all its kernels on the missing payload event. This is scheduling exposure, not a measured time saving.')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('trace');ap.add_argument('--prompt',type=int,default=122)
    ap.add_argument('--tokens',type=int,default=24);ap.add_argument('--output',required=True)
    ap.add_argument('--sweep',type=int)
    a=ap.parse_args();r=summarize(a.trace,a.prompt,a.tokens,a.sweep)
    Path(a.output).write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
    print({k:v for k,v in r.items() if k!='acquires'})
