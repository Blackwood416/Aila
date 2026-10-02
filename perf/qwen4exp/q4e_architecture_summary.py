"""R0/R1 CLI evidence with non-synchronizing byte counters and explicit effective config."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics as st


def read(prefix):
    meta = json.loads(Path(prefix + '.meta.json').read_text(encoding='utf-8-sig'))
    assert meta['exit_code'] == 0 and not meta['watchdog_stop'] and not meta.get('memory_pressure', False), prefix
    for suffix, key in [('.log', 'log_sha256'), ('.err.log', 'err_log_sha256')]:
        assert hashlib.sha256(Path(prefix + suffix).read_bytes()).hexdigest() == meta[key].lower()
    log = Path(prefix + '.log').read_text(encoding='utf-8')
    err = Path(prefix + '.err.log').read_text(encoding='utf-8')
    assert not meta['diagnostic_only'] and meta['workload'] == 'generation'
    configs = [line for line in err.splitlines() if '[Q4E ExecutionConfig]' in line]
    assert len(configs) == 1
    config = dict(re.findall(r'([a-z_]+)=([^ ]+)', configs[0]))
    records = re.findall(r'\[TTFT: ([\d.]+)ms, (\d+) tokens, ([\d.]+) tok/s\]', log)
    texts = re.findall(r'Aila: (.*?)\n\[TTFT:', log, re.S)
    requests, active = [], None
    for line in err.splitlines():
        if '[GenerateMessages] Full prefill:' in line:
            assert active is None
            active = []
        if '[Q4E Work]' in line and active is not None:
            active.append({k: int(v) for k, v in re.findall(r'([a-z_]+)=(\d+)', line)})
        match = re.search(r'\[GenerateMessages\] Prompt=(\d+) Generated=(\d+) in ([\d.]+) ms', line)
        if match:
            prompt, generated = int(match[1]), int(match[2])
            assert active is not None and generated == meta['input']['max_tokens']
            next_pos = 0
            for row in active:
                assert row['pos'] == next_pos
                next_pos += row['count']
            assert next_pos == prompt + generated
            phases = {}
            for phase in ('prefill', 'decode'):
                selected = [r for r in active if (r['pos'] < prompt) == (phase == 'prefill')]
                phases[phase] = {k: sum(r[k] for r in selected) for k in ('count', 'demand_bytes', 'host_misses', 'host_hits')}
                phases[phase]['shapes'] = [r['count'] for r in selected]
            requests.append(dict(prompt_tokens=prompt, generated=generated, request_ms=float(match[3]),
                device_bytes=max(r['device_bytes'] for r in active), phases=phases))
            active = None
    assert active is None and len(requests) == len(records) == len(texts) == meta['input']['repeats']
    for request, (ttft, callbacks, tps), text in zip(requests, records, texts):
        assert int(callbacks) == request['generated']
        request.update(ttft_ms=float(ttft), decode_tok_s=float(tps), text=text,
            text_sha256=hashlib.sha256(text.encode()).hexdigest(), input_tok_s_to_first_token=request['prompt_tokens']*1000/float(ttft))
    warm = requests[1:]
    summary = {k: st.median(r[k] for r in warm) for k in ('ttft_ms', 'decode_tok_s', 'request_ms', 'input_tok_s_to_first_token')}
    summary['prefill_demand_bytes'] = st.median(r['phases']['prefill']['demand_bytes'] for r in warm)
    summary['decode_demand_bytes'] = st.median(r['phases']['decode']['demand_bytes'] for r in warm)
    summary['peak_device_gib'] = max(r['device_bytes'] for r in requests) / 2**30
    summary['peak_working_set_gib'] = meta['peak_working_set_bytes'] / 2**30
    summary['minimum_available_ram_gib'] = meta['minimum_system_free_mb'] / 1024
    return dict(prefix=prefix, metadata=meta, effective_config=config, requests=requests, warm_summary=summary)


def read_canary(prefix):
    meta = json.loads(Path(prefix + '.meta.json').read_text(encoding='utf-8-sig'))
    assert meta['exit_code'] == 0 and not meta['watchdog_stop'] and not meta.get('memory_pressure', False)
    assert meta['diagnostic_only']
    for suffix, key in [('.log', 'log_sha256'), ('.err.log', 'err_log_sha256')]:
        assert hashlib.sha256(Path(prefix + suffix).read_bytes()).hexdigest() == meta[key].lower()
    log = Path(prefix + '.log').read_text(encoding='utf-8')
    markers = {'preset': '[R0 PASS]', 'window': '[R1 QUALITY]', 'mature': '[R2 PROFILE PASS]' if meta['env_overrides'].get('AILA_Q4E_MATURE_PROFILE') == '1' else '[R2 PASS]'}
    assert markers[meta['workload']] in log
    lines = [line for line in log.splitlines() if line.startswith(('[R0 ', '[R1 ', '[R2 ', '[X2 QUALITY]'))]
    result = dict(prefix=prefix, metadata=meta, observations=lines)
    if meta['workload'] == 'mature' and '[R2 TIME]' in log:
        rows = [dict((k, float(v)) for k, v in re.findall(r'([a-z_]+)=([0-9.]+)', line)) for line in lines if line.startswith('[R2 TIME]')]
        assert len(rows) == 9
        result['timings'] = rows
        result['median_us_by_mode'] = {str(mode): st.median(r['median_us'] for r in rows if r['mode'] == mode) for mode in range(3)}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefixes', nargs='+')
    parser.add_argument('--output', required=True)
    parser.add_argument('--canary', nargs='*', default=[])
    parser.add_argument('--vtune-report')
    args = parser.parse_args()
    runs = [read(prefix) for prefix in args.prefixes]
    groups = []
    for prompt in sorted({r['requests'][0]['prompt_tokens'] for r in runs}):
        same = [r for r in runs if r['requests'][0]['prompt_tokens'] == prompt]
        ref = same[0]
        for run in same:
            assert run['metadata']['exe_sha256'] == ref['metadata']['exe_sha256']
            assert run['metadata']['model_config_sha256'] == ref['metadata']['model_config_sha256']
            assert run['metadata']['input'] == ref['metadata']['input']
            assert run['effective_config']['host_ram_mb'] == ref['effective_config']['host_ram_mb'] == '32768'
            assert run['effective_config']['slots'] == ref['effective_config']['slots']
        for preset, chunk in sorted({(r['effective_config']['preset'], r['effective_config']['chunk']) for r in same}):
            selected = [r for r in same if (r['effective_config']['preset'], r['effective_config']['chunk']) == (preset, chunk)]
            warm = [request for run in selected for request in run['requests'][1:]]
            groups.append(dict(prompt_tokens=prompt, preset=preset, chunk=int(chunk), processes=len(selected),
                warm_ttft_ms=[r['ttft_ms'] for r in warm], median_ttft_ms=st.median(r['ttft_ms'] for r in warm),
                warm_decode_tok_s=[r['decode_tok_s'] for r in warm], median_decode_tok_s=st.median(r['decode_tok_s'] for r in warm),
                median_request_ms=st.median(r['request_ms'] for r in warm),
                median_prefill_demand_gb=st.median(r['phases']['prefill']['demand_bytes'] for r in warm)/1e9,
                median_decode_demand_gb=st.median(r['phases']['decode']['demand_bytes'] for r in warm)/1e9,
                input_tok_s_to_first_token=prompt*1000/st.median(r['ttft_ms'] for r in warm)))
    vtune = None
    if args.vtune_report:
        with open(args.vtune_report, encoding='utf-8-sig', newline='') as stream:
            rows = [row for row in csv.DictReader(stream, delimiter=chr(9)) if row.get('GPU Instructions Executed')]
        vtune = dict(path=args.vtune_report, sha256=hashlib.sha256(Path(args.vtune_report).read_bytes()).hexdigest(), instructions=rows)
    result = dict(runs=runs, groups=groups, canaries=[read_canary(p) for p in args.canary], vtune=vtune, limitations=[
        'Input tokens / TTFT is effective prompt-to-first-token throughput, not an isolated GPU prefill timer.',
        'Work summaries use existing demand counters without new queue waits; byte counts are logical scheduled demand, not measured bus bandwidth.',
        'Different prefill arithmetic can change routing and generated text. Quality uses separate fixed-token canaries.',
        'USM ledger bytes are not DXGI residency or measured WDDM paging. CPU working set and available RAM are sampled.',
        'R0 combines several existing features; its generated TPS difference cannot be attributed only to checkpointing.'
    ])
    Path(args.output).write_text(json.dumps(result, indent=2) + chr(10), encoding='utf-8')
    print(json.dumps(groups, indent=2))
