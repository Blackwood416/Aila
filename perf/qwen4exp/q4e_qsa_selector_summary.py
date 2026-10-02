"""Validate selector artifacts; separate wall costs from actual generated TPS."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics as st
from q4e_ram_summary import read as read_generation


def read(prefix):
    meta = json.loads(Path(prefix + '.meta.json').read_text(encoding='utf-8-sig'))
    assert meta['exit_code'] == 0 and not meta['watchdog_stop'] and not meta.get('memory_pressure', False), prefix
    for suffix, key in [('.log', 'log_sha256'), ('.err.log', 'err_log_sha256')]:
        assert hashlib.sha256(Path(prefix + suffix).read_bytes()).hexdigest() == meta[key].lower(), prefix
    if meta['workload'] == 'generation':
        assert not meta['diagnostic_only']
        return read_generation(prefix)
    assert meta['diagnostic_only']
    log = Path(prefix + '.log').read_text(encoding='utf-8')
    result = dict(prefix=prefix, metadata=meta)
    if meta['workload'] == 'qsa':
        marker, keys = '[QSA COST]', ('length', 'stage')
        assert log.count(marker) == 135 and log.count('[QSA CHECK]') == 18 and '[QSA COST END]' in log
    elif meta['workload'] == 'qsa_batches':
        marker, keys = '[QSA BATCH COST]', ('start', 'rows')
        match = re.search(re.escape('[QSA BATCH END]') + r' cases=([0-9]+) exact_ids=1 profile_only=0', log)
        assert match and log.count(marker) == 3 * int(match[1])
    elif meta['workload'] == 'qsa_tests':
        match = re.search(re.escape('[QSA SELECTOR PASS]') + r' cases=([0-9]+) exact_ids=1 sentinel_guards=1 hot_allocations=0', log)
        assert match and 'QSA tests PASSED; teardown USM zero' in log
        result['exact_cases'] = int(match[1])
        return result
    elif meta['workload'] == 'qsa_profile':
        assert '[QSA BATCH END] cases=1 exact_ids=1 profile_only=1' in log
        match = re.search(r'^ops::.*? +([0-9,]+) *$', log, re.M)
        assert match
        result['gpu_instructions_executed'] = int(match[1].replace(',', ''))
        return result
    else:
        raise ValueError(meta['workload'])
    rows = [json.loads(line.split('] ', 1)[1]) for line in log.splitlines() if line.startswith(marker)]
    result['costs'] = []
    for identity in sorted({tuple(row[k] for k in keys) for row in rows}):
        subset = [row for row in rows if tuple(row[k] for k in keys) == identity]
        assert len(subset) == 3 and {row['round'] for row in subset} == {0, 1, 2}
        result['costs'].append(dict(zip(keys, identity),
            median_wall_us=st.median(row['median_us'] for row in subset),
            median_queued_wall_us=st.median(row['queued_wall_us'] for row in subset), rounds=subset))
    return result


def generation_groups(runs):
    selected = [r for r in runs if r['metadata']['workload'] == 'generation']
    if not selected:
        return []
    ref = selected[0]
    def env(run):
        return {k: v for k, v in run['metadata']['env_overrides'].items() if k != 'AILA_QSA_SELECTOR'}
    for run in selected:
        assert run['metadata']['exe_sha256'] == ref['metadata']['exe_sha256']
        assert run['metadata']['model_config_sha256'] == ref['metadata']['model_config_sha256']
        assert run['metadata']['input'] == ref['metadata']['input'] and env(run) == env(ref)
        assert run['gpu_slots'] == ref['gpu_slots']
        assert run['generation'][0]['text_sha256'] == ref['generation'][0]['text_sha256']
    groups = []
    for mode in sorted({int(r['metadata']['env_overrides']['AILA_QSA_SELECTOR']) for r in selected}):
        group = [r for r in selected if int(r['metadata']['env_overrides']['AILA_QSA_SELECTOR']) == mode]
        warm = [request for r in group for request in r['generation'][1:]]
        groups.append(dict(mode=mode, processes=len(group),
            warm_decode_tok_s=[r['decode_tok_s_rounded'] for r in warm],
            median_warm_decode_tok_s=st.median(r['decode_tok_s_rounded'] for r in warm),
            warm_request_ms=[r['request_ms'] for r in warm],
            median_warm_request_ms=st.median(r['request_ms'] for r in warm),
            median_warm_ttft_ms=st.median(r['ttft_ms_rounded'] for r in warm),
            peak_working_set_gib=max(r['metadata']['peak_working_set_bytes'] for r in group) / 2**30,
            minimum_available_ram_gib=min(r['metadata']['minimum_system_free_mb'] for r in group) / 1024))
    return groups


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefixes', nargs='+')
    parser.add_argument('--vtune-csv', nargs='*', default=[])
    parser.add_argument('--submission-trace', nargs='*', default=[])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    runs = [read(prefix) for prefix in args.prefixes]
    profiles = []
    for path in args.vtune_csv:
        with open(path, encoding='utf-8-sig', newline='') as stream:
            rows = [r for r in csv.DictReader(stream, delimiter=chr(9)) if r['Computing Task'].startswith('ops::')]
        assert len(rows) == 1
        profiles.append(dict(path=path, sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(), rows=rows))
    submission = []
    for prefix in args.submission_trace:
        meta = json.loads(Path(prefix + '.meta.json').read_text(encoding='utf-8-sig'))
        assert meta['exit_code'] == 0 and not meta['watchdog_stop'] and not meta.get('memory_pressure', False)
        for suffix, key in [('.log', 'log_sha256'), ('.err.log', 'err_log_sha256')]:
            assert hashlib.sha256(Path(prefix + suffix).read_bytes()).hexdigest() == meta[key].lower()
        log = Path(prefix + '.err.log').read_text(encoding='utf-8', errors='replace')
        submission.append(dict(prefix=prefix, metadata=meta,
            immediate_list_creates=log.count('zeCommandListCreateImmediate('),
            regular_list_creates=log.count('zeCommandListCreate('),
            queue_executes=log.count('zeCommandQueueExecuteCommandLists(')))
    result = dict(runs=runs, generation=generation_groups(runs), vtune=profiles, submission=submission, limitations=[
        'Operator costs are synchronized or queued wall times, not hardware event timings or generated TPS.',
        'VTune instruction-count collection perturbs timings; compare instruction categories and zero spill only.',
        'Generation ABBA uses two processes per mode and one warm request per process; tail latency remains variable.',
        'Immediate command lists were tested on isolated QSA, not validated as a whole-engine default.',
        'Out-of-order compute queues were audited only; implicit dependencies and shared scratch remain unchanged.'
    ])
    Path(args.output).write_text(json.dumps(result, indent=2) + chr(10), encoding='utf-8')
    print(json.dumps(result['generation'], indent=2))
