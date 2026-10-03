"""Position-preserving screening evidence for expert supply and phase prefetch."""
import argparse
import json
import statistics as st
from pathlib import Path

from q4e_prompt_hotset_summary import summarize
from q4e_hotset_adaptive_summary import identity
from q4e_cold_summary import read as read_canary
from q4e_prompt_hotset_trace import inspect
from q4e_refill_cost_trace import inspect as inspect_refill


IGNORED_ENV = {
    'AILA_Q4E_PREFETCH_DECODE', 'AILA_Q4E_PREFETCH_PREFILL',
    'AILA_Q4E_HOTSET_SWAPS',
}
MODES = ('all', 'prefill', 'off', 'off-static')


def flag(env, name, default=True):
    value = str(env.get(name, str(int(default)))).lower()
    if value in ('1', 'true', 'yes', 'on'):
        return True
    if value in ('0', 'false', 'no', 'off'):
        return False
    raise ValueError(f'Invalid boolean {name}={value!r}')


def supply_mode(env):
    enabled = flag(env, 'AILA_Q4E_PREFETCH')
    decode = enabled and flag(env, 'AILA_Q4E_PREFETCH_DECODE')
    prefill = enabled and flag(env, 'AILA_Q4E_PREFETCH_PREFILL')
    if decode and not prefill:
        raise ValueError('Decode-only prefetch is outside this screening campaign')
    phase = 'all' if decode else 'prefill' if prefill else 'off'
    swaps = int(env.get('AILA_Q4E_HOTSET_SWAPS', '96'))
    if swaps == 0:
        if phase != 'off':
            raise ValueError('Static-hotset arm must also disable both prefetch phases')
        return 'off-static'
    return phase


def summarize_supply(prefixes, allow_window_change=False):
    if not prefixes:
        raise ValueError('At least one run prefix is required')
    if len({Path(prefix).resolve() for prefix in prefixes}) != len(prefixes):
        raise ValueError('Duplicate run prefixes do not represent independent processes')
    runs = [summarize(p) for p in prefixes]
    reference = runs[0]
    reference_env = reference['metadata']['env_overrides']
    ignored_env = IGNORED_ENV | ({'AILA_Q35_PREFILL_CHUNK'} if allow_window_change else set())
    ignored_config = {'chunk'} if allow_window_change else set()
    shared_env = {k: v for k, v in reference_env.items() if k not in ignored_env}
    shared_config = {k: v for k, v in reference['effective_config'].items() if k not in ignored_config}
    arm_env = {}
    shared_online_budget = None
    for run in runs:
        meta = run['metadata']
        for field in ('exe_sha256', 'model_config_sha256', 'input', 'arguments',
                      'hotset_profile', 'stack', 'gpu'):
            if meta[field] != reference['metadata'][field]:
                raise ValueError(f'{run["prefix"]}: mismatched {field}')
        if {k: v for k, v in run['effective_config'].items() if k not in ignored_config} != shared_config:
            raise ValueError(f'{run["prefix"]}: mismatched effective_config')
        env = meta['env_overrides']
        if {k: v for k, v in env.items() if k not in ignored_env} != shared_env:
            raise ValueError(f'{run["prefix"]}: mismatched non-campaign environment')
        mode = supply_mode(env)
        swaps = int(env.get('AILA_Q4E_HOTSET_SWAPS', '96'))
        if swaps < 0:
            raise ValueError(f'{run["prefix"]}: negative online refill budget')
        if mode != 'off-static':
            if shared_online_budget is not None and swaps != shared_online_budget:
                raise ValueError(f'{run["prefix"]}: mismatched online refill budget across prefetch arms')
            shared_online_budget = swaps
        run['supply_mode'] = mode
        chunk = int(run['effective_config']['chunk'])
        if chunk <= 0:
            raise ValueError(f'{run["prefix"]}: invalid effective chunk')
        run['supply_chunk'] = chunk
        # Flags are the grouping identity, not a filename convention. Do not mix
        # different refill budgets or phase overrides within an arm.
        normalized = (flag(env, 'AILA_Q4E_PREFETCH_DECODE'),
                      flag(env, 'AILA_Q4E_PREFETCH_PREFILL'),
                      int(env.get('AILA_Q4E_HOTSET_SWAPS', '96')))
        arm = (mode, chunk)
        if arm in arm_env and arm_env[arm] != normalized:
            raise ValueError(f'{run["prefix"]}: different settings within {mode}/{chunk}')
        arm_env[arm] = normalized
        if len(run['requests']) != len(reference['requests']):
            raise ValueError(f'{run["prefix"]}: mismatched request count')

    chunks = sorted({run['supply_chunk'] for run in runs})
    case_names = {(mode, chunk): f'{mode}/{chunk}' if len(chunks) > 1 else mode
                  for mode in MODES for chunk in chunks
                  if (mode, chunk) in arm_env}
    cases = []
    for (mode, chunk), name in case_names.items():
        selected = [run for run in runs if (run['supply_mode'], run['supply_chunk']) == (mode, chunk)]
        for run in selected:
            run['supply_case'] = name
        totals = [sum(q['request_ms'] for q in run['requests']) for run in selected]
        cases.append(dict(name=name, mode=mode, chunk=chunk, processes=len(selected),
                          prefixes=[run['prefix'] for run in selected],
                          corpus_request_ms=dict(samples=totals, median=st.median(totals),
                              cv=st.pstdev(totals) / st.mean(totals) if st.mean(totals) else None),
                          effective_config=selected[0]['effective_config']))
    fields = {
        'request_ms': lambda q: q['request_ms'],
        'ttft_ms': lambda q: q['ttft_ms'],
        'decode_tok_s': lambda q: q['decode_tok_s'],
        'prefill_demand_bytes': lambda q: q['phases']['prefill']['demand_bytes'],
        'decode_demand_bytes': lambda q: q['phases']['decode']['demand_bytes'],
        'refill_bytes': lambda q: q['hotset']['epoch']['refill_bytes'],
    }
    groups = []
    for position, request in enumerate(reference['requests']):
        group = dict(position=position + 1, prompt_tokens=request['prompt_tokens'],
                     generated=request['generated'])
        corpus = reference['metadata']['input'].get('corpus', [])
        if corpus:
            group['corpus'] = corpus[position % len(corpus)]
        for case in cases:
            name = case['name']
            selected = [run for run in runs if run['supply_case'] == name]
            samples = {field: [] for field in fields}
            for run in selected:
                sample = run['requests'][position]
                if (sample['prompt_tokens'], sample['generated']) != (
                        group['prompt_tokens'], group['generated']):
                    raise ValueError(f'{run["prefix"]}: mismatched request at {position + 1}')
                for field, get in fields.items():
                    samples[field].append(get(sample))
            cv = {field: st.pstdev(values) / st.mean(values)
                  if st.mean(values) else None for field, values in samples.items()}
            group[name] = dict(mode=case['mode'], chunk=case['chunk'],
                               effective_config=case['effective_config'],
                               processes=len(selected),
                               prefixes=[run['prefix'] for run in selected],
                               samples=samples,
                               median={field: st.median(values) for field, values in samples.items()},
                               cv=cv, request_cv=cv['request_ms'],
                               variable=cv['request_ms'] is not None and cv['request_ms'] > 0.15,
                               replicated=len(selected) > 1)
        for case in cases:
            baseline = case_names.get(('all', case['chunk']))
            if baseline and case['mode'] != 'all':
                group[case['name']]['request_ratio_vs_all'] = (
                    group[baseline]['median']['request_ms'] / group[case['name']]['median']['request_ms'])
        groups.append(group)
    return dict(runs=runs, groups=groups, cases=cases,
                evidence_kind='screening',
                performance_resources=dict(
                    peak_working_set_bytes=max(run['metadata']['peak_working_set_bytes'] for run in runs),
                    minimum_system_free_mb=min(run['metadata']['minimum_system_free_mb'] for run in runs),
                    peak_device_ledger_bytes=max(q['device_bytes'] for run in runs for q in run['requests'])),
                allow_window_change=allow_window_change,
                processes_by_mode={mode: sum(run['supply_mode'] == mode for run in runs)
                                   for mode in MODES},
                processes_by_case={case['name']: case['processes'] for case in cases},
                comparison_identity=dict(exe_sha256=reference['metadata']['exe_sha256'],
                                         model_config_sha256=reference['metadata']['model_config_sha256'],
                                         input=reference['metadata']['input'],
                                         effective_config=reference['effective_config'],
                                         shared_effective_config=shared_config,
                                         shared_env_overrides=shared_env,
                                         ignored_env=sorted(ignored_env),
                                         ignored_effective_config=sorted(ignored_config)),
                limitations=[
                    'These CLI processes are screening evidence; sample counts may differ by arm.',
                    'A single sample has CV zero by definition and provides no repeatability evidence.',
                    'Every startup, adaptation and return-domain request stays at its corpus position.',
                    'Disabling prefetch can change host residency, CPU/GPU placement and generated routing.',
                    'Off-static disables online hotset swaps; Ready adoption and prompt handoff remain governed by shared settings.',
                    'CLI timing conventions are unchanged; model/profile load is outside request time.',
                    'Demand/refill bytes count scheduled payloads, not measured PCIe bandwidth.',
                    'Ordinary traces validate host ordering and ownership, not GPU device overlap.',
                ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefixes', nargs='+')
    parser.add_argument('--canaries', nargs='*', default=[])
    parser.add_argument('--traces', nargs='*', default=[])
    parser.add_argument('--logs', nargs='*', default=[])
    parser.add_argument('--refill-traces', nargs='*', default=[])
    parser.add_argument('--allow-window-change', action='store_true',
                        help='Allow only prefill chunk to differ; keep slots and all other effective settings identical')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = summarize_supply(args.prefixes, args.allow_window_change)
    result['canaries'] = [read_canary(p) for p in args.canaries]
    result['traces'] = [inspect(p, cold_queue=2) for p in args.traces]
    result['refill_history'] = []
    for path in args.refill_traces:
        history = inspect_refill(path)
        compact = {key: value for key, value in history.items() if key != 'batches'}
        compact['batches'] = [{**{key: value for key, value in batch.items() if key != 'entries'},
            'incoming_by_category': {category: [entry['key'] for entry in batch['entries']
                if entry['category'] == category] for category in history['category_counts']}}
            for batch in history['batches']]
        result['refill_history'].append(compact)
    result['validation_logs'] = [dict(**identity(p), tail=Path(p).read_text(
        encoding='utf-8-sig').splitlines()[-20:]) for p in args.logs]
    sources = ['src/models/Qwen4ExpExpertCache.cpp', 'src/models/Qwen4ExpExpertCache.hpp',
               'src/models/Qwen4ExpExpertExecutor.cpp', 'src/models/Qwen4ExpBnb4Backend.cpp',
               'perf/qwen4exp/q4e_execution_probe.ps1',
               'tests/models/Qwen4ExpColdAcceptance.hpp',
               'perf/qwen4exp/q4e_supply_campaign.ps1',
               'perf/qwen4exp/q4e_supply_chat.ps1',
               'perf/qwen4exp/q4e_refill_cost_trace.py',
               'perf/qwen4exp/q4e_supply_summary.py',
               'perf/qwen4exp/q4e_prompt_hotset_summary.py',
               'perf/qwen4exp/q4e_hotset_adaptive_summary.py',
               'perf/qwen4exp/q4e_architecture_summary.py']
    result['source_artifacts'] = [identity(p) for p in sources]
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    for group in result['groups']:
        print(group['position'], group['prompt_tokens'],
              {case['name']: group[case['name']]['median'] for case in result['cases']})


if __name__ == '__main__':
    main()
