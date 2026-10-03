"""Validate epoch ownership and aggregate host intervals; no device-time claims."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from q4e_cache_replay import read_trace, validate


def inspect(path, cold_queue=0):
    raw=read_trace(path);forwards=validate(raw,cold_queue=cold_queue)
    by_id={r['forward']:r for r in forwards}
    timings=defaultdict(lambda:defaultdict(int))
    requests={};request=-1
    for r in raw:
        if r['type']=='forward_begin':
            if r['position']==0:
                request+=1
            requests[r['forward']]=request
        if r['type']=='host':
            timings[r['forward']][r['name']]+=r['end_ns']-r['begin_ns']
    groups=defaultdict(list)
    for fid,f in by_id.items():
        if f['tokens']==1:
            groups[requests[fid]].append(fid)
    result=[]
    for req,ids in groups.items():
        spans={k:sum(timings[fid][k] for fid in ids)/len(ids)/1e6
               for k in {k for fid in ids for k in timings[fid]}}
        result.append(dict(request=req,decode_forwards=len(ids),
            mean_forward_ms=sum(by_id[fid]['host_span_ns'] for fid in ids)/len(ids)/1e6,
            mean_acquire_ms=sum(by_id[fid]['acquire_ns'] for fid in ids)/len(ids)/1e6,
            mean_fill_accounted_ms=sum(by_id[fid]['fill_ns'] for fid in ids)/len(ids)/1e6,
            mean_prefetch_wait_ms=sum(by_id[fid]['prefetch_wait_ns'] for fid in ids)/len(ids)/1e6,
            mean_interval_ms=spans,
            demand_bytes=sum(by_id[fid]['bytes'] for fid in ids),
            direct=sum(by_id[fid].get('direct_entries',0) for fid in ids),
            cold=sum(by_id[fid].get('cold_request_entries',0) for fid in ids)))
    return dict(path=path,sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        forwards=len(forwards),epoch_pairs=sum(r['type']=='epoch_retire' for r in raw),
        prompt_handoffs=sum(r['type']=='prompt_handoff' for r in raw),
        prefill_handoffs=sum(r['type']=='prefill_handoff' for r in raw),
        ready_adoptions=sum(r['type']=='ready_adopt' for r in raw),cold_queue=cold_queue,
        decode_requests=result,
        limitations=['Host intervals include waits and submission, not pure GPU kernel execution.',
            'CPU compute/join are nested or concurrent with expert_acquire_submit: do not add them together.',
            'Epoch rebuild is outside forward spans and is counted separately in full request measurements.',
            'Trace overhead is present; these are diagnostic timings, not benchmark throughput.'])


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('traces',nargs='+');ap.add_argument('--output',required=True)
    ap.add_argument('--cold-queue',type=int,choices=(0,1,2),default=0,
                    help='Use the queue policy recorded in the probe metadata')
    args=ap.parse_args()
    result=[inspect(p,args.cold_queue) for p in args.traces]
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
