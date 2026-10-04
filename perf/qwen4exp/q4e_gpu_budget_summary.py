"""GPU expert-budget comparisons with fixed H8 mapped NF4-Q8 CPU supply."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import re

import q4e_cpu_supply_summary as h9
import q4e_heterogeneous_summary as h8

VARIABLES = {'AILA_Q4E_EXPERT_CACHE_MB': '0'}
CPU_DEFAULTS = dict(h9.VARIABLES)
FIXED_ENV = {**h9.FIXED_ENV, **CPU_DEFAULTS}
SLOT_BYTES = 2534400
AUTO_SLOTS = 3567
PROFILE_GEOMETRY = 'nf4-original-expert-order-48x512-2560x640-b64-nested1-'


def case_name(settings):
    value = settings['AILA_Q4E_EXPERT_CACHE_MB']
    if value not in ('0', '8192', '7168', '6144'):
        raise ValueError('GPU budget only compares auto/8192/7168/6144 MiB')
    return 'baseline' if value == '0' else 'gpu' + value


def read_budget_observations(prefix):
    lines = Path(prefix + '.err.log').read_text(encoding='utf-8').splitlines()
    caches = [line for line in lines if '[Q4E ExpertCache]' in line]
    profiles = [line for line in lines if '[Q4E Hotset] profile=' in line]
    if len(caches) != 1 or len(profiles) != 1:
        raise ValueError(f'{prefix}: expected one expert cache and profile initialization')
    cache = {k: int(v) for k, v in re.findall(r'([a-z_]+)=(\d+)', caches[0])}
    match = re.search(r'identity=([^ ]+)', profiles[0])
    if not match or not match[1].startswith(PROFILE_GEOMETRY):
        raise ValueError(f'{prefix}: different model/profile geometry')
    startup, requests, active = [], [], None
    for line in lines:
        if '[GenerateMessages] Full prefill:' in line:
            if active is not None:
                raise ValueError(f'{prefix}: incomplete hotset observations')
            active = []
        kind = next((name for name in ('HotsetEpoch', 'PromptHotset', 'PrefillHotset', 'Frozen')
                     if '[Q4E ' + name + ']' in line), None)
        if kind:
            row = dict(kind=kind, **{k: float(v) if '.' in v else int(v)
                for k, v in re.findall(r'([a-z_]+)=([0-9.]+)', line)})
            (startup if active is None else active).append(row)
        if '[GenerateMessages] Prompt=' in line:
            if active is None:
                raise ValueError(f'{prefix}: missing hotset request observations')
            requests.append(active)
            active = None
    if active is not None:
        raise ValueError(f'{prefix}: incomplete hotset observations')
    return dict(cache=cache, profile_geometry=match[1], startup=startup, requests=requests)


def validate_run(run, observations):
    meta, env = run['metadata'], run['metadata']['env_overrides']
    for key, default in CPU_DEFAULTS.items():
        if str(env.get(key, default)) != default:
            raise ValueError(f'{run["prefix"]}: fixed CPU implementation differs at {key}')
    h9.validate_run(run)
    if meta.get('runtime_defaults', False):
        raise ValueError(f'{run["prefix"]}: runtime defaults are not the fixed campaign')
    settings = {k: str(env.get(k, default)) for k, default in VARIABLES.items()}
    case_name(settings)
    mb = int(settings['AILA_Q4E_EXPERT_CACHE_MB'])
    expected = AUTO_SLOTS if mb == 0 else mb * 1048576 // SLOT_BYTES
    cache = observations['cache']
    if not observations['profile_geometry'].startswith(PROFILE_GEOMETRY):
        raise ValueError(f'{run["prefix"]}: different model/profile geometry')
    if cache.get('slot_bytes') != SLOT_BYTES or cache.get('nested_codes') != 1:
        raise ValueError(f'{run["prefix"]}: different expert slot geometry')
    if cache.get('slots') != expected or run['effective_config'].get('slots') != str(expected):
        raise ValueError(f'{run["prefix"]}: requested GPU budget differs from actual slots')
    if len(observations['requests']) != len(run['requests']):
        raise ValueError(f'{run["prefix"]}: hotset observations disagree with request count')
    for request, rows in zip(run['requests'], observations['requests']):
        request['budget_hotset_observations'] = rows
    run['gpu_budget'] = dict(requested_mb=mb, actual_slots=expected,
        slot_bytes=SLOT_BYTES, pool_bytes=expected * SLOT_BYTES,
        profile_geometry=observations['profile_geometry'], startup=observations['startup'])


@contextmanager
def campaign_rules(actual_configs):
    # Validate the actual slot count first. H8 then compares every other config
    # field normally; only slots are normalized inside that comparison adapter.
    # Restore original configurations before returning the public result.
    old_variables, old_name, old_read = h8.VARIABLES, h8.case_name, h8.summarize
    def read(prefix):
        run = old_read(prefix)
        validate_run(run, read_budget_observations(prefix))
        actual_configs[prefix] = dict(run['effective_config'])
        run['effective_config'] = {**run['effective_config'], 'slots': str(AUTO_SLOTS)}
        return run
    # Legacy H8 variables are required by its RAM check; FIXED_ENV/H9 validation
    # locks them all. Only the expert budget can vary in this campaign.
    h8.VARIABLES, h8.case_name, h8.summarize = {**old_variables, **VARIABLES}, case_name, read
    try:
        yield
    finally:
        h8.VARIABLES, h8.case_name, h8.summarize = old_variables, old_name, old_read


def summarize_runs(prefixes):
    actual_configs = {}
    with campaign_rules(actual_configs):
        result = h8.summarize_runs(prefixes)
    for run in result['runs']:
        run['effective_config'] = actual_configs[run['prefix']]
    identity_config = result['comparison_identity']['effective_config']
    identity_config.pop('slots')
    result['comparison_identity']['allowed_effective_config_differences'] = ['slots']
    for case in result['cases']:
        case['settings'] = {k: case['settings'][k] for k in VARIABLES}
        selected = [run for run in result['runs'] if run['case'] == case['name']]
        case['actual_slots'] = selected[0]['gpu_budget']['actual_slots']
        case['effective_config'] = selected[0]['effective_config']
    result['campaign'] = dict(name='H9 GPU expert budget', baseline='H8 q8-mapped-ram0-short1',
        variable_environment=VARIABLES, fixed_environment=FIXED_ENV,
        geometry=dict(layers=48, experts=512, hidden=2560, intermediate=640,
            block_size=64, nested=True, slot_bytes=SLOT_BYTES), auto_slots=AUTO_SLOTS)
    result['limitations'].extend([
        'Auto baseline must have 3567 slots; this experiment is specific to the recorded model and A770 setup.',
        'Budget changes GPU hotset capacity and CPU/GPU placement; the same profile ranking is retained.',
        'All per-request hot/frozen/epoch observations and actual effective configurations are preserved.',
        'Reduced allocation does not itself prove reduced WDDM paging; independent diagnostics are required.'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefixes', nargs='+')
    parser.add_argument('--canaries', nargs='*', default=[])
    parser.add_argument('--traces', nargs='*', default=[])
    parser.add_argument('--logs', nargs='*', default=[])
    parser.add_argument('--source-manifest')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = summarize_runs(args.prefixes)
    result['canaries'] = [h8.read_canary(p) for p in args.canaries]
    result['traces'] = [h8.inspect(p, cold_queue=2) for p in args.traces]
    result['validation_logs'] = [dict(**h8.identity(p),
        tail=Path(p).read_text(encoding='utf-8-sig').splitlines()[-20:]) for p in args.logs]
    sources = ['src/ops/Qwen4ExpCpuExpertOps.cpp', 'src/ops/Qwen4ExpCpuExpertOps.hpp',
        'src/ops/MoeExpertOps.cpp', 'src/ops/MoeExpertOps.hpp',
        'src/models/Qwen4ExpExpertExecutor.cpp', 'src/models/Qwen4ExpExpertExecutor.hpp',
        'src/models/Qwen4ExpExpertCache.cpp', 'src/models/Qwen4ExpExpertCache.hpp',
        'src/models/Qwen4ExpBnb4Backend.cpp', 'tests/ops/Qwen4ExpCpuExpertTests.cpp',
        'tests/models/Qwen4ExpModuleTests.cpp', 'tests/models/Qwen4ExpExpertReadinessTests.cpp',
        'perf/qwen4exp/q4e_execution_probe.ps1', 'perf/qwen4exp/q4e_cpu_supply_summary.py',
        'perf/qwen4exp/q4e_heterogeneous_summary.py', 'perf/qwen4exp/q4e_gpu_budget_campaign.ps1',
        'perf/qwen4exp/q4e_gpu_budget_summary.py']
    result['source_artifacts'] = [h8.identity(p) for p in sources]
    if args.source_manifest:
        manifest = json.loads(Path(args.source_manifest).read_text(encoding='utf-8-sig'))
        if manifest['exe_sha256'].lower() != result['comparison_identity']['exe_sha256'].lower():
            raise ValueError('Compiled source manifest describes a different measured EXE')
        for artifact in manifest['files']:
            if h8.identity(artifact['snapshot'])['sha256'] != artifact['sha256'].lower():
                raise ValueError('Compiled source snapshot hash differs')
        result['compiled_source_manifest'] = manifest
        result['compiled_source_manifest_identity'] = h8.identity(args.source_manifest)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    for group in result['groups']:
        print(group['position'], group['prompt_tokens'],
            {name: value['median'] for name, value in group['cases'].items()})


if __name__ == '__main__':
    main()
