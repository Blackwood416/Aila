"""Summarize H1/H2 CLI comparisons without treating bypass as missing demand."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics as st

from q4e_architecture_summary import read


def fields(line):
    result = {}
    for word in line.split(']', 1)[1].split():
        if '=' not in word:
            continue
        key, value = word.split('=', 1)
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
        result[key] = value
    return result


def identity(path):
    p = Path(path)
    return dict(path=p.as_posix(), bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def read_run(prefix):
    run = read(prefix)
    run['mode'] = int(run['metadata']['env_overrides'].get('AILA_Q4E_RESIDENT_DECODE', '0'))
    groups, current = [], None
    for line in Path(prefix + '.err.log').read_text(encoding='utf-8').splitlines():
        if '[GenerateMessages] Full prefill:' in line:
            assert current is None
            current = []
        if '[Q4E Resident]' in line and current is not None:
            current.append(fields(line))
        if '[GenerateMessages] Prompt=' in line:
            groups.append(current)
            current = None
    assert current is None and len(groups) == len(run['requests'])
    for request, rows in zip(run['requests'], groups):
        if run['mode']:
            assert len(rows) == request['phases']['decode']['count']
            assert [row['pos'] for row in rows] == list(range(request['prompt_tokens'], request['prompt_tokens'] + len(rows)))
            assert all(row['mode'] == run['mode'] and row['direct'] + row['cold'] == row['route'] == 480 for row in rows)
            assert sum(row['demand_bytes'] for row in rows) == request['phases']['decode']['demand_bytes']
            request['resident'] = {key: sum(row[key] for row in rows)
                                   for key in ('route', 'resident', 'direct', 'cold', 'acquire_calls', 'demand_bytes', 'metadata_bytes')}
            request['resident_rows'] = rows
        else:
            assert not rows
    return run


def read_canary(prefix):
    meta = json.loads(Path(prefix + '.meta.json').read_text(encoding='utf-8-sig'))
    assert meta['exit_code'] == 0 and not meta['watchdog_stop'] and not meta.get('memory_pressure')
    assert meta['diagnostic_only'] and meta['workload'] == 'resident'
    for suffix, key in (('.log', 'log_sha256'), ('.err.log', 'err_log_sha256')):
        assert identity(prefix + suffix)['sha256'] == meta[key].lower()
    lines = Path(prefix + '.log').read_text(encoding='utf-8').splitlines()
    assert '[HOTSET MODEL PASS] quality=1 continuation=1 checkpoint_replay=1 modes=0,1,2' in lines
    models = [fields(line) for line in lines if line.startswith('[HOTSET MODEL]')]
    assert len(models) == 3 and {row['mode'] for row in models} == {0, 1, 2}
    assert all(row['replay_exact'] == row['snapshot_exact'] == 1 and row['live_usm_delta'] == 0 for row in models)
    assert all(row['exact_rows'] == row['top1_match'] == 48 for row in models if row['mode'])
    return dict(prefix=prefix, metadata=meta, models=models,
                quality=[line for line in lines if line.startswith('[X2 QUALITY]')])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('prefixes', nargs='+')
    ap.add_argument('--canaries', nargs='*', default=[])
    ap.add_argument('--traces', nargs='*', default=[])
    ap.add_argument('--regression-log')
    ap.add_argument('--regression-detail')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    runs = [read_run(prefix) for prefix in args.prefixes]
    groups = []
    for prompt in sorted({run['requests'][0]['prompt_tokens'] for run in runs}):
        same = [run for run in runs if run['requests'][0]['prompt_tokens'] == prompt]
        ref = same[0]
        for run in same:
            assert run['metadata']['exe_sha256'] == ref['metadata']['exe_sha256']
            assert run['metadata']['input'] == ref['metadata']['input']
            assert run['effective_config']['slots'] == ref['effective_config']['slots']
            assert run['effective_config']['host_ram_mb'] == '32768'
            assert run['effective_config']['chunk'] == ref['effective_config']['chunk']
            assert all(row['text_sha256'] == ref['requests'][0]['text_sha256'] for row in run['requests'])
        for mode in sorted({run['mode'] for run in same}):
            selected = [run for run in same if run['mode'] == mode]
            warm = [row for run in selected for row in run['requests'][1:]]
            group = dict(prompt_tokens=prompt, mode=mode, processes=len(selected), warm_requests=len(warm),
                         warm_ttft_ms=[row['ttft_ms'] for row in warm],
                         warm_decode_tok_s=[row['decode_tok_s'] for row in warm],
                         warm_request_ms=[row['request_ms'] for row in warm],
                         median_ttft_ms=st.median(row['ttft_ms'] for row in warm),
                         median_decode_tok_s=st.median(row['decode_tok_s'] for row in warm),
                         median_request_ms=st.median(row['request_ms'] for row in warm),
                         median_prefill_demand_bytes=st.median(row['phases']['prefill']['demand_bytes'] for row in warm),
                         median_decode_demand_bytes=st.median(row['phases']['decode']['demand_bytes'] for row in warm))
            if mode:
                group['median_direct_entries'] = st.median(row['resident']['direct'] for row in warm)
                group['median_acquire_calls'] = st.median(row['resident']['acquire_calls'] for row in warm)
                group['median_metadata_bytes'] = st.median(row['resident']['metadata_bytes'] for row in warm)
            groups.append(group)
    output = dict(date='2026-10-02', checkpoint_commit='5ea0171', production_default_mode=0,
                  runs=runs, groups=groups, canaries=[read_canary(p) for p in args.canaries],
                  source_artifacts=[identity(p) for p in (
                      'src/models/Qwen4ExpExpertCache.hpp', 'src/models/Qwen4ExpExpertCache.cpp',
                      'src/models/Qwen4ExpBnb4Backend.hpp', 'src/models/Qwen4ExpBnb4Backend.cpp',
                      'src/ops/MoeExpertOps.hpp', 'src/ops/MoeExpertOps.cpp',
                      'tests/models/Qwen4ExpExpertReadinessTests.cpp', 'tests/models/Qwen4ExpResidentAcceptance.hpp',
                      'tests/models/Qwen4ExpToleranceTests.cpp',
                      'perf/qwen4exp/q4e_execution_probe.ps1', 'perf/qwen4exp/q4e_cache_replay.py',
                      'perf/qwen4exp/q4e_hotset_summary.py', 'perf/qwen4exp/q4e_hotset_trace.py')],
                  limitations=['One short and one natural long prompt; no broad workload claim.',
                               'Finite warm request samples; CPU cold placement and adaptive fixed hotsets are not tested.',
                               'Existing dynamic LRU/admission and fixed expert pool budget; timing-dependent prefetch readiness can still change victims.',
                               'Resident metadata traffic is reported separately from NF4 demand bytes.',
                               'No old-driver A/B; negative variants are retained, not promoted.'])
    output['trace_checks'] = []
    for prefix in args.traces:
        meta = json.loads(Path(prefix + '.meta.json').read_text(encoding='utf-8-sig'))
        assert meta['exit_code'] == 0 and not meta['watchdog_stop'] and not meta.get('memory_pressure')
        assert meta['diagnostic_only'] and identity(prefix + '.cache.jsonl')['sha256'] == meta['trace_sha256'].lower()
        result = json.loads(Path(prefix + '.summary.json').read_text(encoding='utf-8'))
        assert result['sha256'] == meta['trace_sha256'].lower()
        output['trace_checks'].append(dict(metadata=meta, result=result,
            artifacts=[identity(prefix + suffix) for suffix in ('.cache.jsonl', '.summary.json', '.log', '.err.log', '.memory.json')]))
    if args.regression_log:
        assert '100% tests passed, 0 tests failed out of 6' in Path(args.regression_log).read_text(encoding='utf-8-sig')
        output['regression'] = dict(passed=6, failed=0, log=identity(args.regression_log),
            readiness_exe=identity('build-refactor/AilaQwen4ExpExpertReadinessTests.exe'))
    if args.regression_detail:
        assert b'cancellation_drain=1' in Path(args.regression_detail).read_bytes()
        output['regression']['detail'] = identity(args.regression_detail)
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2) + chr(10), encoding='utf-8')
    for group in groups:
        print(json.dumps(group))
