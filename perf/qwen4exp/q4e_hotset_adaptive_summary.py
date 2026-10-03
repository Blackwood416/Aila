"""H4/C2 full-request measurements, including profile/refill traffic and cold work."""
import argparse
import hashlib
import json
import re
import statistics as st
from pathlib import Path
from q4e_architecture_summary import read as read_cli
from q4e_cold_summary import read as read_canary
from q4e_cache_replay import read_trace, validate


def identity(path):
    p=Path(path)
    return dict(path=p.as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def fields(line):
    return {k:int(v) for k,v in re.findall(r'([a-z_]+)=(\d+)',line)}


def read(prefix):
    run = read_cli(prefix)
    err = Path(prefix+'.err.log').read_text(encoding='utf-8')
    active = None
    samples = []
    previous = dict(swaps=0,refill_bytes=0,refill_ns=0,publishes=0)
    # The first epoch line may include startup profile preload. Report it in
    # startup, not as first-request demand; subsequent deltas include handoff.
    last = None
    for line in err.splitlines():
        if '[Q4E HotsetEpoch]' in line:
            last = fields(line)
            if active is None:
                previous = last
        if '[GenerateMessages] Full prefill:' in line:
            active = dict(cpu_tasks=0,compute_ns=0,join_ns=0,direct=0,cold=0)
        if active is not None and '[Q4E Cold]' in line:
            row = fields(line)
            if row['count']==1:
                for k in active: active[k] += row[k]
        if '[GenerateMessages] Prompt=' in line:
            if last:
                active['epoch'] = {k:last[k]-previous[k] for k in ('swaps','refill_bytes','refill_ns','publishes')}
                for k in ('ready_adoptions','ready_adoption_ns'):
                    if k in last:
                        active['epoch'][k] = last[k]-previous.get(k,0)
                previous = last
            samples.append(active); active = None
    assert len(samples)==len(run['requests'])
    for request, sample in zip(run['requests'], samples):
        request['hotset'] = sample
    return run


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('prefixes',nargs='+');ap.add_argument('--output',required=True)
    ap.add_argument('--canaries',nargs='*',default=[])
    ap.add_argument('--traces',nargs='*',default=[])
    ap.add_argument('--logs',nargs='*',default=[])
    args=ap.parse_args()
    runs=[read(p) for p in args.prefixes]
    groups={}
    for run in runs:
        env=run['metadata']['env_overrides']
        key=str((run['requests'][0]['prompt_tokens'],env.get('AILA_Q4E_HOTSET_PROFILE',''),
            env.get('AILA_Q4E_HOTSET_BORROW','0'),env.get('AILA_Q4E_COLD_MODE','0'),
            env.get('AILA_Q4E_COLD_POLICY','0'),env.get('AILA_Q4E_CPU_THREADS','0'),
            env.get('AILA_Q4E_COLD_BUDGET_US','0'),env.get('AILA_Q4E_COLD_PCIE_NUM','141'),
            env.get('AILA_Q4E_HOTSET_RING','legacy'),env.get('AILA_Q4E_HOTSET_COLD_QUEUE','legacy'),
            env.get('AILA_Q4E_HOTSET_ADOPT_READY','0'),env.get('AILA_Q4E_HOTSET_ADOPT_LIMIT','96'),
            run['metadata']['exe_sha256']))
        groups.setdefault(key,[]).extend(run['requests'][1:])
    summary={key:dict(samples=len(rows),decode=[r['decode_tok_s'] for r in rows],
        median_decode=st.median(r['decode_tok_s'] for r in rows),
        median_ttft_ms=st.median(r['ttft_ms'] for r in rows),
        median_request_ms=st.median(r['request_ms'] for r in rows),
        median_decode_demand_bytes=st.median(r['phases']['decode']['demand_bytes'] for r in rows))
        for key,rows in groups.items()}
    traces=[]
    for p in args.traces:
        raw=read_trace(p);rows=validate(raw)
        traces.append(dict(**identity(p),forwards=len(rows),cpu_tasks=sum(r.get('cpu_tasks',0) for r in rows),
            swaps=sum(r['type']=='epoch_retire' for r in raw),publishes=sum(r['type']=='epoch_publish' for r in raw)))
    sources=['src/models/Qwen4ExpHotsetPolicy.hpp','src/models/Qwen4ExpExpertCache.cpp',
        'src/models/Qwen4ExpExpertCache.hpp','src/models/Qwen4ExpExpertExecutor.cpp',
        'src/models/Qwen4ExpExpertExecutor.hpp','src/models/Qwen4ExpBnb4Backend.cpp',
        'src/engine/EngineLoad.cpp','tests/models/Qwen4ExpColdAcceptance.hpp',
        'tests/models/Qwen4ExpExpertReadinessTests.cpp','perf/qwen4exp/q4e_hotset_profile.py',
        'perf/qwen4exp/q4e_hotset_adaptive_summary.py','perf/qwen4exp/q4e_execution_probe.ps1',
        'perf/qwen4exp/q4e_cache_replay.py','perf/qwen4exp/profiles/qwen38-nf4-h4-v1.profile',
        'perf/qwen4exp/profiles/qwen38-nf4-h4-v1.json','perf/qwen4exp/q4e_cold_summary.py',
        'perf/qwen4exp/test_q4e_hotset_profile.py','perf/qwen4exp/test_q4e_cache_replay.py']
    result=dict(runs=runs,groups=summary,canaries=[read_canary(p) for p in args.canaries],traces=traces,
        validation_logs=[dict(**identity(p),tail=Path(p).read_text(encoding='utf-8-sig').splitlines()[-20:]) for p in args.logs],
        source_artifacts=[identity(p) for p in sources],limitations=[
            'Every request, including adaptation/cold-start and outliers, is retained. Request 2 is not necessarily a converged epoch.',
            'CLI request_ms ends before final alignment; throughput uses the same CLI timing convention for every arm.',
            'Profile preload is in model load. Borrowed-slot restore at first single-row decode is included in decode and request time.',
            'Per-run EXE hashes distinguish earlier FIFO/full-LRU variants from final cold-only LRU; do not equate their implementations.',
            'Ordinary trace validates host ordering/ownership, not GPU device overlap. Full-model profiling was not enabled.',
            '1833 input tokens do not cover the later QSA context regime. No multi-session HTTP/MTP acceptance in this tranche.'
        ])
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))
