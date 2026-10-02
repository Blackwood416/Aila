"""Compare bounded host RAM pools with fixed GPU capacity and identical inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics as st

from q4e_scratch_summary import read as read_memory


def distribution(values):
    ordered = sorted(values)
    return dict(values=values, median=st.median(values),
                minimum=ordered[0], maximum=ordered[-1],
                mad=st.median(abs(v-st.median(values)) for v in values))


def read(prefix):
    meta = json.loads(Path(prefix+'.meta.json').read_text(encoding='utf-8-sig'))
    assert meta['exit_code'] == 0 and not meta['watchdog_stop']
    assert not meta.get('memory_pressure', False)
    for suffix, key in [('.log', 'log_sha256'), ('.err.log', 'err_log_sha256')]:
        assert hashlib.sha256(Path(prefix+suffix).read_bytes()).hexdigest() == meta[key].lower()
    log = Path(prefix+'.log').read_text(encoding='utf-8')
    err = Path(prefix+'.err.log').read_text(encoding='utf-8')
    slots = re.findall(r'ExpertCache\] slots=(\d+) slot_bytes=(\d+)', err)
    assert len(slots) == 1, 'Expected target-only cache'
    out = dict(prefix=prefix, metadata=meta, gpu_slots=int(slots[0][0]),
               slot_bytes=int(slots[0][1]),
               host_ram_mb=int(meta['env_overrides']['AILA_Q4E_HOT_RAM_MB']))
    if meta['workload'] == 'memory':
        memory = read_memory(prefix)
        out.update(prefill=memory['prefill'], tail=memory['tail'], memory=memory['memory'])
    elif meta['workload'] == 'generation':
        records = re.findall(r'\[TTFT: ([\d.]+)ms, (\d+) tokens, ([\d.]+) tok/s\]', log)
        generated = re.findall(r'\[GenerateMessages\] Prompt=(\d+) Generated=(\d+) in ([\d.]+) ms', err)
        texts = re.findall(r'Aila: (.*?)\n\[TTFT:', log, re.S)
        assert len(records) == len(generated) == len(texts) == meta['input']['repeats']
        out['generation'] = []
        for (ttft, callbacks, tps), (prompt, tokens, elapsed), text in zip(records, generated, texts):
            assert int(callbacks) == int(tokens) == meta['input']['max_tokens'], 'Callback count differs from generated tokens'
            out['generation'].append(dict(ttft_ms_rounded=float(ttft), tokens=int(tokens),
                prompt_tokens=int(prompt), request_ms=float(elapsed), decode_tok_s_rounded=float(tps),
                text_sha256=hashlib.sha256(text.encode()).hexdigest(), text=text))
        assert len({g['text_sha256'] for g in out['generation']}) == 1
    else:
        raise ValueError(meta['workload'])
    return out


def trace_summary(prefix):
    path = Path(prefix+'.cache.jsonl')
    meta = read(prefix)['metadata']
    assert meta['diagnostic_only']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == meta['trace_sha256'].lower()
    forwards, active, config = [], {}, None
    route_hash = hashlib.sha256()
    route_count = 0
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['type'] == 'config':
                assert config is None
                config = row
            elif row['type'] == 'forward_begin':
                forwards.append(dict(begin=row, hits=0, misses=0, bytes=0, host_hits=0, file_fills=0,
                                     fill_ns=0, prefetch_wait_ns=0, acquire_ns=0))
                active[row['forward']] = forwards[-1]
            elif row['type'] == 'route':
                assert len(row['ids']) == row['rows']*row['top_k']
                assert row['rows'] == active[row['forward']]['begin']['tokens']
                route_hash.update(json.dumps([row[k] for k in ('forward', 'layer', 'rows', 'top_k', 'ids')], separators=(',', ':')).encode())
                route_count += 1
            elif row['type'] == 'acquire_begin':
                active['acquire', row['acquire']] = (row['forward'], row['ns'])
            elif row['type'] == 'acquire_end':
                forward, start = active.pop(('acquire', row['acquire']))
                out = active[forward]
                for key in ('hits', 'misses', 'bytes', 'host_hits', 'file_fills', 'fill_ns', 'prefetch_wait_ns'):
                    out[key] += row[key]
                out['acquire_ns'] += row['ns']-start
            elif row['type'] == 'forward_end':
                active.pop(row['forward'])['end'] = row
    assert not active and config
    passes = []
    keys = ('hits', 'misses', 'bytes', 'host_hits', 'file_fills', 'fill_ns', 'prefetch_wait_ns', 'acquire_ns')
    for forward in forwards:
        begin = forward['begin']
        if begin['position'] == 0 and begin['tokens'] > 4:
            passes.append(dict(prefill={key: 0 for key in keys}, decode={key: 0 for key in keys},
                               prefill_tokens=0, decode_tokens=0))
        if not passes:
            continue  # Initial model warmup is not a request.
        phase = 'prefill' if begin['tokens'] > 1 else 'decode'
        for key in keys:
            passes[-1][phase][key] += forward[key]
        passes[-1][phase+'_tokens'] += begin['tokens']
    assert len(passes) == meta['input']['repeats']
    assert route_count == len(forwards)*config['layers']
    return dict(prefix=prefix, metadata=meta, config=config, forwards=forwards, passes=passes,
                route_sha256=route_hash.hexdigest(), route_count=route_count)


def summarize(runs):
    groups = []
    for workload in sorted({r['metadata']['workload'] for r in runs}):
        selected = [r for r in runs if r['metadata']['workload'] == workload]
        ref = selected[0]
        reference_env = {k: v for k, v in ref['metadata']['env_overrides'].items() if k != 'AILA_Q4E_HOT_RAM_MB'}
        for run in selected:
            assert run['metadata']['exe_sha256'] == ref['metadata']['exe_sha256']
            assert run['metadata']['model_config_sha256'] == ref['metadata']['model_config_sha256']
            assert run['metadata']['input'] == ref['metadata']['input']
            assert run['gpu_slots'] == ref['gpu_slots']
            assert {k: v for k, v in run['metadata']['env_overrides'].items() if k != 'AILA_Q4E_HOT_RAM_MB'} == reference_env
            assert not run['metadata']['diagnostic_only']
            if workload == 'memory':
                assert run['prefill'][0]['hash'] == ref['prefill'][0]['hash']
                assert run['tail'][0]['hashes'] == ref['tail'][0]['hashes']
                assert run['memory'][-1]['device_bytes'] == ref['memory'][-1]['device_bytes']
            else:
                assert run['generation'][0]['text_sha256'] == ref['generation'][0]['text_sha256']
        for ram in sorted({r['host_ram_mb'] for r in selected}):
            sample = [r for r in selected if r['host_ram_mb'] == ram]
            row = dict(workload=workload, host_ram_mb=ram, process_count=len(sample), gpu_slots=ref['gpu_slots'],
                peak_working_set_gib=max(r['metadata']['peak_working_set_bytes'] for r in sample)/2**30,
                minimum_available_ram_gib=min(r['metadata']['minimum_system_free_mb'] for r in sample)/1024)
            if workload == 'memory':
                row['first_prefill_ms'] = [r['prefill'][0]['ms'] for r in sample]
                row['warm_prefill_ms'] = distribution([p['ms'] for r in sample for p in r['prefill'][1:]])
                row['warm_prefill_tok_s'] = ref['prefill'][0]['tokens']*1000/row['warm_prefill_ms']['median']
                row['warm_tail_ms_per_step'] = distribution([st.mean(t['ms']) for r in sample for t in r['tail'][1:]])
            else:
                row['first_request_ms'] = [r['generation'][0]['request_ms'] for r in sample]
                row['warm_request_ms'] = distribution([g['request_ms'] for r in sample for g in r['generation'][1:]])
                row['warm_decode_tok_s'] = distribution([g['decode_tok_s_rounded'] for r in sample for g in r['generation'][1:]])
                row['warm_ttft_ms'] = distribution([g['ttft_ms_rounded'] for r in sample for g in r['generation'][1:]])
            groups.append(row)
    return groups


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefixes', nargs='+')
    parser.add_argument('--traces', nargs='*', default=[])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    runs = [read(prefix) for prefix in args.prefixes]
    traces = [trace_summary(p) for p in args.traces]
    trace_comparisons = []
    for workload in sorted({t['metadata']['workload'] for t in traces}):
        pair = sorted([t for t in traces if t['metadata']['workload'] == workload],
                      key=lambda t: int(t['metadata']['env_overrides']['AILA_Q4E_HOT_RAM_MB']))
        assert len(pair) == 2
        base, candidate = pair
        assert base['metadata']['input'] == candidate['metadata']['input']
        assert base['metadata']['exe_sha256'] == candidate['metadata']['exe_sha256']
        assert base['metadata']['model_config_sha256'] == candidate['metadata']['model_config_sha256']
        omit = {'AILA_Q4E_HOT_RAM_MB', 'AILA_Q4E_CACHE_TRACE'}
        assert {k:v for k,v in base['metadata']['env_overrides'].items() if k not in omit} == {k:v for k,v in candidate['metadata']['env_overrides'].items() if k not in omit}
        assert base['config']['slots'] == candidate['config']['slots']
        assert base['route_sha256'] == candidate['route_sha256']
        trace_comparisons.append(dict(workload=workload, same_routes=True,
            baseline=base['prefix'], candidate=candidate['prefix'],
            base_last_pass=base['passes'][-1], candidate_last_pass=candidate['passes'][-1]))
    result = dict(runs=runs, summary=summarize(runs), traces=traces, trace_comparisons=trace_comparisons, limitations=[
        'First request/pass retained separately; OS file cache not cleared. Repeated lighthouse and short story are canaries.',
        'CLI decode TPS is rounded to 0.1 and TTFT to 1 ms. Stream chunk=decode chunk=1; callback counts validated against engine generated count.',
        'Fixed eight-token tail is teacher forcing, not generated throughput. Trace timings are diagnostic only.',
        'File fills include buffered ReadFile and packing; they are not physical SSD read counts. USM ledger is not WDDM residency.'])
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result['summary'], indent=2))
