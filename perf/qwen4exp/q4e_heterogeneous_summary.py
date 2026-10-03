"""Position-preserving H8 CPU arithmetic and complementary supply comparisons."""
import argparse
import json
import re
import statistics as st
from pathlib import Path

from q4e_prompt_hotset_summary import summarize
from q4e_hotset_adaptive_summary import identity
from q4e_cold_summary import read as read_canary
from q4e_prompt_hotset_trace import inspect

VARIABLES = {
    'AILA_Q4E_CPU_NF4_Q8': '0', 'AILA_Q4E_HOST_COMPLEMENT': '0',
    'AILA_Q4E_CPU_MMAP_COLD': '0',
    'AILA_Q4E_CPU_MMAP_PREFETCH': '0',
    'AILA_Q4E_CPU_MMAP_TOUCH': '0',
    'AILA_Q4E_HOT_RAM_MB': '32768',
    'AILA_Q4E_FILL_SHORT_PARTS': '0',
    'AILA_Q4E_HOST_COMPLEMENT_SCAN': '512', 'AILA_Q4E_FILL_PARTS': '8',
    'AILA_Q4E_FILL_THREADS': '8', 'AILA_Q4E_COLD_POLICY': '1',
    'AILA_Q4E_COLD_PCIE_NUM': '141', 'AILA_Q4E_CPU_THREADS': '12',
}


def read_extra(run):
    active, observations = None, []
    for line in Path(run['prefix'] + '.err.log').read_text(encoding='utf-8').splitlines():
        if '[GenerateMessages] Full prefill:' in line:
            active = dict(complement_evictions=0, complement_scans=0, complement_searches=0,
                mapped_tasks=0, short_part_fills=0, source_prefetch_ns=0, source_prefetch_calls=0,
                source_prefetch_ranges=0, source_prefetch_bytes=0, source_prefetch_failures=0,
                source_touch_ns=0, source_touch_pages=0)
        if active is not None and '[Q4E Cold]' in line:
            values = dict(re.findall(r'([a-z_]+)=(\d+)', line))
            for key in ('mapped_tasks', 'source_prefetch_ns', 'source_prefetch_calls',
                        'source_prefetch_ranges', 'source_prefetch_bytes', 'source_prefetch_failures',
                        'source_touch_ns', 'source_touch_pages'):
                active[key] += int(values.get(key, 0))
        if active is not None and '[Q4E Work]' in line:
            values = dict(re.findall(r'([a-z_]+)=(\d+)', line))
            for key in ('complement_evictions', 'complement_scans', 'complement_searches', 'short_part_fills'):
                active[key] += int(values.get(key, 0))
        if active is not None and '[GenerateMessages] Prompt=' in line:
            observations.append(active)
            active = None
    if len(observations) != len(run['requests']):
        raise ValueError('Complement observations disagree with request count')
    for request, extra in zip(run['requests'], observations):
        request['supply'] = extra


def case_name(settings):
    q8, hc = settings['AILA_Q4E_CPU_NF4_Q8'], settings['AILA_Q4E_HOST_COMPLEMENT']
    name = 'q8-complement' if q8 == '1' and hc == '1' else 'q8' if q8 == '1' else 'complement' if hc == '1' else 'baseline'
    if settings['AILA_Q4E_CPU_MMAP_COLD'] == '1':
        name += '-mapped'
    if settings['AILA_Q4E_HOT_RAM_MB'] != '32768':
        name += '-ram' + settings['AILA_Q4E_HOT_RAM_MB']
    if settings['AILA_Q4E_FILL_SHORT_PARTS'] != '0':
        name += '-short' + settings['AILA_Q4E_FILL_SHORT_PARTS']
    if settings['AILA_Q4E_CPU_MMAP_PREFETCH'] == '1':
        name += '-prefetch'
    if settings['AILA_Q4E_CPU_MMAP_TOUCH'] == '1':
        name += '-touch'
    for key, label in [('AILA_Q4E_FILL_PARTS', 'parts'), ('AILA_Q4E_FILL_THREADS', 'fill'),
                       ('AILA_Q4E_CPU_THREADS', 'cpu'), ('AILA_Q4E_HOST_COMPLEMENT_SCAN', 'scan')]:
        if settings[key] != VARIABLES[key]:
            name += '-' + label + settings[key]
    if settings['AILA_Q4E_COLD_POLICY'] != '1' or settings['AILA_Q4E_COLD_PCIE_NUM'] != '141':
        name += '-policy' + settings['AILA_Q4E_COLD_POLICY'] + '-pcie' + settings['AILA_Q4E_COLD_PCIE_NUM']
    return name


def summarize_runs(prefixes):
    if not prefixes or len({Path(p).resolve() for p in prefixes}) != len(prefixes):
        raise ValueError('Run prefixes must be nonempty independent processes')
    runs = [summarize(p) for p in prefixes]
    reference = runs[0]
    common_env = {k: v for k, v in reference['metadata']['env_overrides'].items() if k not in VARIABLES}
    configs, order = {}, []
    for run in runs:
        for field in ('exe_sha256', 'model_config_sha256', 'input', 'arguments', 'hotset_profile', 'stack', 'gpu'):
            if run['metadata'][field] != reference['metadata'][field]:
                raise ValueError(f'{run["prefix"]}: different {field}')
        env = run['metadata']['env_overrides']
        if {k: v for k, v in env.items() if k not in VARIABLES} != common_env:
            raise ValueError(f'{run["prefix"]}: non-campaign environment differs')
        settings = {k: str(env.get(k, default)) for k, default in VARIABLES.items()}
        config = dict(run['effective_config'])
        expected_config = dict(reference['effective_config'])
        if config.pop('host_ram_mb') != settings['AILA_Q4E_HOT_RAM_MB']:
            raise ValueError(f'{run["prefix"]}: host RAM override differs from effective configuration')
        expected_config.pop('host_ram_mb')
        if config != expected_config:
            raise ValueError(f'{run["prefix"]}: different effective_config')
        name = case_name(settings)
        if name in configs and configs[name] != settings:
            raise ValueError(f'{run["prefix"]}: ambiguous case label')
        if name not in configs:
            order.append(name)
        configs[name] = settings
        run['case'] = name
        read_extra(run)
        if len(run['requests']) != len(reference['requests']):
            raise ValueError('Different request histories')
    fields = {
        'request_ms': lambda q: q['request_ms'], 'ttft_ms': lambda q: q['ttft_ms'],
        'decode_tok_s': lambda q: q['decode_tok_s'],
        'prefill_demand_bytes': lambda q: q['phases']['prefill']['demand_bytes'],
        'decode_demand_bytes': lambda q: q['phases']['decode']['demand_bytes'],
        'host_pool_misses': lambda q: sum(p['host_misses'] for p in q['phases'].values()),
        'cpu_tasks': lambda q: q['hotset']['cpu_tasks'],
        'cpu_compute_ns': lambda q: q['hotset']['compute_ns'],
        'cpu_join_ns': lambda q: q['hotset']['join_ns'],
        'complement_evictions': lambda q: q['supply']['complement_evictions'],
        'complement_scans': lambda q: q['supply']['complement_scans'],
        'complement_searches': lambda q: q['supply']['complement_searches'],
        'mapped_tasks': lambda q: q['supply']['mapped_tasks'],
        'short_part_fills': lambda q: q['supply']['short_part_fills'],
        'source_prefetch_ns': lambda q: q['supply']['source_prefetch_ns'],
        'source_prefetch_calls': lambda q: q['supply']['source_prefetch_calls'],
        'source_prefetch_ranges': lambda q: q['supply']['source_prefetch_ranges'],
        'source_prefetch_bytes': lambda q: q['supply']['source_prefetch_bytes'],
        'source_prefetch_failures': lambda q: q['supply']['source_prefetch_failures'],
        'source_touch_ns': lambda q: q['supply']['source_touch_ns'],
        'source_touch_pages': lambda q: q['supply']['source_touch_pages'],
    }
    groups = []
    for position, reference_request in enumerate(reference['requests']):
        group = dict(position=position + 1, prompt_tokens=reference_request['prompt_tokens'],
                     generated=reference_request['generated'], cases={})
        for name in order:
            selected = [r for r in runs if r['case'] == name]
            requests = [r['requests'][position] for r in selected]
            if any((q['prompt_tokens'], q['generated']) != (group['prompt_tokens'], group['generated']) for q in requests):
                raise ValueError('Different prompt/output token counts')
            samples = {field: [get(q) for q in requests] for field, get in fields.items()}
            cv = {field: st.pstdev(values) / st.mean(values) if st.mean(values) else None
                  for field, values in samples.items()}
            group['cases'][name] = dict(samples=samples,
                median={field: st.median(values) for field, values in samples.items()}, cv=cv,
                replicated=len(selected) > 1, variable=cv['request_ms'] is not None and cv['request_ms'] > 0.15)
        groups.append(group)
    cases = []
    for name in order:
        selected = [r for r in runs if r['case'] == name]
        totals = [sum(q['request_ms'] for q in r['requests']) for r in selected]
        cases.append(dict(name=name, settings=configs[name], processes=len(selected),
            prefixes=[r['prefix'] for r in selected],
            corpus_request_ms=dict(samples=totals, median=st.median(totals),
                cv=st.pstdev(totals) / st.mean(totals))))
    return dict(schema=1, runs=runs, groups=groups, cases=cases,
        comparison_identity=dict(exe_sha256=reference['metadata']['exe_sha256'],
            model_config_sha256=reference['metadata']['model_config_sha256'], input=reference['metadata']['input'],
            arguments=reference['metadata']['arguments'], hotset_profile=reference['metadata']['hotset_profile'],
            stack=reference['metadata']['stack'], gpu=reference['metadata']['gpu'],
            effective_config=reference['effective_config'], shared_environment=common_env),
        performance_resources=dict(peak_working_set_bytes=max(r['metadata']['peak_working_set_bytes'] for r in runs),
            minimum_system_free_mb=min(r['metadata']['minimum_system_free_mb'] for r in runs),
            peak_device_ledger_bytes=max(q['device_bytes'] for r in runs for q in r['requests'])),
        limitations=['Single-process screens do not establish repeatable gains.',
            'Request positions and every startup/return observation are preserved; model/profile load is outside request time.',
            'The system file cache is not reset between processes; reverse ordering reduces but does not eliminate order bias.',
            'CPU NF4-Q8 changes arithmetic; model quality acceptance is independent of real greedy continuation.',
            'Complementary host sources can change CPU/GPU placement and generated routing.',
            'CPU compute/join may overlap supply/GPU work; these durations must not be added as exclusive costs.',
            'Byte counters are submitted payloads, not measured PCIe throughput.',
            'Host-pool misses exclude direct staging reads when the persistent host pool is disabled.',
            'Source prefetch bytes are advisory requests; touch pages count range visits, not unique OS faults.',
            'Source prefetch/touch time is included in cold queue and may overlap GPU work; compute excludes it.',
            'Ordinary traces show host ordering/ownership, not pure GPU execution or overlap.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefixes', nargs='+')
    parser.add_argument('--canaries', nargs='*', default=[])
    parser.add_argument('--traces', nargs='*', default=[])
    parser.add_argument('--logs', nargs='*', default=[])
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-manifest', help='Saved source identity for the measured binary phase')
    args = parser.parse_args()
    result = summarize_runs(args.prefixes)
    result['canaries'] = [read_canary(p) for p in args.canaries]
    result['traces'] = [inspect(p, cold_queue=2) for p in args.traces]
    result['validation_logs'] = [dict(**identity(p), tail=Path(p).read_text(encoding='utf-8-sig').splitlines()[-20:]) for p in args.logs]
    sources = ['src/ops/Qwen4ExpCpuExpertOps.cpp', 'src/ops/Qwen4ExpCpuExpertOps.hpp',
        'src/models/Qwen4ExpExpertExecutor.cpp', 'src/models/Qwen4ExpExpertExecutor.hpp', 'src/models/Qwen4ExpExpertCache.cpp',
        'src/models/Qwen4ExpExpertCache.hpp', 'src/models/Qwen4ExpBnb4Backend.cpp',
        'tests/ops/Qwen4ExpCpuExpertTests.cpp', 'tests/models/Qwen4ExpExpertReadinessTests.cpp',
        'tests/models/Qwen4ExpColdAcceptance.hpp', 'perf/qwen4exp/q4e_execution_probe.ps1',
        'perf/qwen4exp/q4e_heterogeneous_campaign.ps1', 'perf/qwen4exp/q4e_heterogeneous_summary.py',
        'perf/qwen4exp/q4e_supply_chat.ps1']
    result['source_artifacts'] = [identity(p) for p in sources]
    if args.source_manifest:
        result['compiled_source_manifest'] = json.loads(Path(args.source_manifest).read_text(encoding='utf-8-sig'))
        manifest = result['compiled_source_manifest']
        if manifest['exe_sha256'].lower() != result['comparison_identity']['exe_sha256'].lower():
            raise ValueError('Compiled source manifest describes a different measured EXE')
        for artifact in manifest['files']:
            if identity(artifact['snapshot'])['sha256'] != artifact['sha256'].lower():
                raise ValueError('Compiled source snapshot hash differs')
        result['compiled_source_manifest_identity'] = identity(args.source_manifest)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    for group in result['groups']:
        print(group['position'], group['prompt_tokens'], {name: value['median'] for name, value in group['cases'].items()})


if __name__ == '__main__':
    main()
