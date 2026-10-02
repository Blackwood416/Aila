"""C1/H3 generated-request evidence, with frozen-set and phase-coverage checks."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics as st
from q4e_architecture_summary import read as read_cli
from q4e_cache_replay import read_trace, validate


def identity(path):
    p=Path(path)
    return dict(path=p.as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def fields(line):
    return {k:float(v) if any(c in v for c in '.eE') else int(v)
            for k,v in re.findall(r'([a-z_][a-z_0-9]*)=([-+0-9.eE]+)',line)}


def read(prefix):
    meta=json.loads(Path(prefix+'.meta.json').read_text(encoding='utf-8-sig'))
    assert meta['exit_code']==0 and not meta['watchdog_stop'] and not meta.get('memory_pressure'),prefix
    for suffix,key in (('.log','log_sha256'),('.err.log','err_log_sha256')):
        assert identity(prefix+suffix)['sha256']==meta[key].lower()
    log=Path(prefix+'.log').read_text(encoding='utf-8')
    err=Path(prefix+'.err.log').read_text(encoding='utf-8')
    result=dict(prefix=prefix,metadata=meta)
    configs=[line for line in err.splitlines() if '[Q4E ExecutionConfig]' in line]
    assert len(configs)==1
    result['effective_config']=dict(re.findall(r'([a-z_]+)=([^ ]+)',configs[0]))
    if meta['workload']=='cold_quality':
        assert '[COLD MODEL PASS]' in log
        result['observations']=[line for line in log.splitlines() if line.startswith(('[COLD ','[X2 QUALITY]'))]
        return result
    assert meta['workload']=='cold_bench' and '[COLD BENCH PASS]' in log
    requests=[fields(line) for line in log.splitlines() if line.startswith('[COLD BENCH]')]
    assert all(not r.get('adaptive',0) for r in requests), 'Adaptive epochs require the CLI hotset summary, not frozen-set accounting'
    calibration=[fields(line) for line in log.splitlines() if line.startswith('[COLD CALIBRATION]')]
    texts=log.split('[COLD TEXT] ')[1:]
    texts=[t.split('\n[/COLD TEXT]')[0] for t in texts]
    assert len(requests)==len(texts)==4 and len(calibration)==1
    assert all(r['identity']==calibration[0]['identity'] for r in requests)
    completed=[];active=None
    for line in err.splitlines():
        if '[GenerateMessages] Full prefill:' in line:
            assert active is None
            active=[]
        if active is not None and '[Q4E Work]' in line:active.append(fields(line))
        match=re.search(r'\[GenerateMessages\] Prompt=(\d+) Generated=(\d+) in ([\d.]+) ms',line)
        if match:
            prompt,generated=int(match[1]),int(match[2]);assert active is not None
            position=0
            for row in active:
                assert row['pos']==position
                position+=row['count']
            assert position==prompt+generated
            phases={phase:{key:sum(r[key] for r in active if (r['pos']<prompt)==(phase=='prefill'))
                for key in ('count','demand_bytes','host_hits','host_misses')} for phase in ('prefill','decode')}
            completed.append(dict(prompt=prompt,generated=generated,engine_request_ms=float(match[3]),phases=phases));active=None
    assert len(completed)==4 and active is None
    for r,t,c in zip(requests,texts,completed):
        assert r['prompt']==c['prompt'] and r['callbacks']==c['generated']==48
        assert r['demand_bytes']==sum(p['demand_bytes'] for p in c['phases'].values())
        r.update(c,text=t,text_sha256=hashlib.sha256(t.encode()).hexdigest(),warm=r['index']>=2)
        r['callback_decode_tok_s']=(r['generated']-1)*1000/(r['last_callback_ms']-r['ttft_ms'])
        # CLI uses the return of generate(), not the final text callback.
        # The harness additionally drains the final GPU work before recording return.
        r['decode_tok_s']=(r['generated']-1)*1000/(r['request_ms']-r['ttft_ms'])
        r['prompt_to_first_tok_s']=r['prompt']*1000/r['ttft_ms']
    result.update(calibration=calibration[0],requests=requests)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('prefixes',nargs='*');ap.add_argument('--cli',nargs='*',default=[])
    ap.add_argument('--canaries',nargs='*',default=[]);ap.add_argument('--traces',nargs='*',default=[])
    ap.add_argument('--excluded',nargs='*',default=[]);ap.add_argument('--diagnostics',nargs='*',default=[])
    ap.add_argument('--regression-log');ap.add_argument('--python-log')
    ap.add_argument('--output',required=True);args=ap.parse_args()
    runs=[read(p) for p in args.prefixes]
    if runs:
        for key in ('exe_sha256','model_config_sha256'):
            assert len({r['metadata'][key] for r in runs})==1,key
        for key in ('slots','host_ram_mb','chunk'):
            assert len({r['effective_config'][key] for r in runs})==1,key
        for key in ('AILA_Q4E_CPU_THREADS','AILA_Q4E_CPU_MAX_EXPERTS','AILA_Q4E_COLD_BUDGET_US','AILA_Q4E_COLD_LOW_RANK'):
            assert len({r['metadata']['env_overrides'].get(key) for r in runs})==1,key
    samples=[row for run in runs if not run['metadata']['diagnostic_only'] for row in run['requests'] if row['warm']]
    groups=[]
    for prompt,h3,mode in sorted({(r['prompt'],r['h3'],r['mode']) for r in samples}):
        selected=[r for r in samples if (r['prompt'],r['h3'],r['mode'])==(prompt,h3,mode)]
        groups.append(dict(prompt=prompt,h3=h3,mode=mode,warm_samples=len(selected),
            decode_samples=[r['decode_tok_s'] for r in selected],
            ttft_samples_ms=[r['ttft_ms'] for r in selected],
            request_samples_ms=[r['request_ms'] for r in selected],
            **{key:st.median(r[key] for r in selected) for key in ('ttft_ms','request_ms','engine_request_ms','decode_tok_s','cpu_tasks','join_ns','compute_ns','demand_bytes','direct','acquire_calls')}))
    traces=[]
    for p in args.traces:
        meta=json.loads(Path(p+'.meta.json').read_text(encoding='utf-8-sig'))
        assert meta['exit_code']==0 and meta['diagnostic_only'] and not meta['watchdog_stop'] and not meta.get('memory_pressure')
        assert identity(p+'.cache.jsonl')['sha256']==meta['trace_sha256'].lower()
        rows=read_trace(p+'.cache.jsonl');observed=validate(rows);device=[r for r in rows if r['type']=='device']
        assert sum(r.get('cpu_tasks',0) for r in observed)>0
        if meta['env_overrides'].get('AILA_PROFILE_Q4E_COLD_DEVICE')=='1':
            assert {'resident_hot_down','cold_gpu_down','cold_input_copy','rank_merge'} <= {r['label'] for r in device}
        traces.append(dict(prefix=p,identity=identity(p+'.cache.jsonl'),forwards=len(observed),
            cpu_tasks=sum(r.get('cpu_tasks',0) for r in observed),device_intervals=len(device),
            device_median_ns={label:st.median(r['end']-r['start'] for r in device if r['label']==label) for label in {r['label'] for r in device}}))
    sources=['src/engine/EngineLoad.cpp','src/models/Qwen4ExpBnb4Backend.cpp','src/models/Qwen4ExpBnb4Backend.hpp',
        'src/models/Qwen4ExpExpertCache.cpp','src/models/Qwen4ExpExpertCache.hpp','src/models/Qwen4ExpExpertExecutor.cpp',
        'src/models/Qwen4ExpExpertExecutor.hpp','tests/models/Qwen4ExpColdAcceptance.hpp',
        'tests/models/Qwen4ExpExpertReadinessTests.cpp','tests/models/Qwen4ExpToleranceTests.cpp',
        'perf/qwen4exp/q4e_execution_probe.ps1','perf/qwen4exp/q4e_cache_replay.py','perf/qwen4exp/q4e_cold_summary.py']
    result=dict(runs=runs,groups=groups,cli=[read_cli(p) for p in args.cli],canaries=[read(p) for p in args.canaries],
        traces=traces,source_artifacts=[identity(p) for p in sources],limitations=[
        'CPU/GPU clocks are separate; host enqueue or join times are not GPU idle intervals.',
        'Each cold benchmark shares one frozen set; its prompt-only calibration is reported separately.',
        'Quality uses fixed teacher tokens; generated-text agreement alone is not an accuracy gate.',
        'First request of each mode is separate; finite samples do not define production dispatch.',
        'Demand bytes are logical scheduled payload, not measured PCIe traffic. USM is not DXGI residency.'])
    result['cli_groups']=[]
    for prompt,h3 in sorted({(q['prompt_tokens'],r['metadata']['env_overrides'].get('AILA_Q4E_RESIDENT_PREFILL')=='1') for r in result['cli'] for q in r['requests']}):
        selected=[q for r in result['cli'] for q in r['requests'][1:] if q['prompt_tokens']==prompt and (r['metadata']['env_overrides'].get('AILA_Q4E_RESIDENT_PREFILL')=='1')==h3]
        result['cli_groups'].append(dict(prompt=prompt,h3=h3,warm_samples=len(selected),
            ttft_samples_ms=[q['ttft_ms'] for q in selected],decode_samples=[q['decode_tok_s'] for q in selected],
            **{key:st.median(q[key] for q in selected) for key in ('ttft_ms','decode_tok_s','request_ms')}))
    for group,paths in (('excluded_runs',args.excluded),('diagnostics',args.diagnostics)):
        result[group]=[]
        for prefix in paths:
            meta=json.loads(Path(prefix+'.meta.json').read_text(encoding='utf-8-sig'))
            artifacts=[identity(prefix+suffix) for suffix in ('.meta.json','.log','.err.log','.memory.json') if Path(prefix+suffix).exists()]
            result[group].append(dict(prefix=prefix,metadata=meta,artifacts=artifacts,performance_comparable=False))
    if args.regression_log:
        assert '100% tests passed, 0 tests failed out of 6' in Path(args.regression_log).read_text(encoding='utf-8-sig')
        result['regression']=dict(passed=6,failed=0,artifact=identity(args.regression_log))
    if args.python_log:
        data=Path(args.python_log).read_text(encoding='utf-8-sig')
        assert 'Ran 7 tests' in data and 'OK' in data
        result['python_regression']=dict(passed=7,artifact=identity(args.python_log))
    Path(args.output).write_text(json.dumps(result,indent=2,ensure_ascii=False)+chr(10),encoding='utf-8')
    print(json.dumps(groups,indent=2))
