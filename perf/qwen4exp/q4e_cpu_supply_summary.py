"""H9 CPU supply and decode fusion comparisons on the fixed H8 mapped NF4-Q8 setup."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path

import q4e_heterogeneous_summary as h8

VARIABLES = {'AILA_Q4E_DECODE_PENDING_MLP_RESIDUAL': '0',
    'AILA_Q4E_HC_DECODE_DOWN_WG': '256', 'AILA_Q4E_HC_DECODE_UP_STREAMS': '0',
    'AILA_Q4E_CPU_NF4_Q8_ROW4': '0', 'AILA_Q4E_CPU_Q8_CHUNKS': '0',
    'AILA_Q4E_ROUTER_RESIDENT_FUSED': '0', 'AILA_Q4E_CPU_NF4_Q8_BLOCK64': '0'}
FIXED_ENV = {
    'AILA_Q4E_CPU_NF4_Q8': '1', 'AILA_Q4E_CPU_MMAP_COLD': '1',
    'AILA_Q4E_CPU_MMAP_PREFETCH': '0', 'AILA_Q4E_CPU_MMAP_TOUCH': '0',
    'AILA_Q4E_HOT_RAM_MB': '0', 'AILA_Q4E_FILL_SHORT_PARTS': '1',
    'AILA_Q4E_HOST_COMPLEMENT': '0', 'AILA_Q4E_HOST_COMPLEMENT_SCAN': '512',
    'AILA_Q4E_FILL_PARTS': '8', 'AILA_Q4E_FILL_THREADS': '8',
    'AILA_Q4E_COLD_POLICY': '1', 'AILA_Q4E_COLD_PCIE_NUM': '141',
    'AILA_Q4E_CPU_THREADS': '12', 'AILA_Q4E_CPU_MAX_EXPERTS': '10',
    'AILA_Q4E_COLD_MODE': '2', 'AILA_Q4E_COLD_BUDGET_US': '1000',
    'AILA_Q4E_EXPERT_READER': 'readfile', 'AILA_Q4E_PREFETCH_DECODE': '0',
    'AILA_Q4E_PREFETCH_PREFILL': '0', 'AILA_Q4E_PHASE_SCRATCH': '1',
    'AILA_Q4E_DEMAND_MICROBATCH': '32', 'AILA_Q4E_INLINE_CHECKPOINT': '1',
    'AILA_Q35_PREFILL_CHUNK': '2048', 'AILA_Q4E_HOTSET_BORROW': '1',
    'AILA_Q4E_HOTSET_COLD_QUEUE': '2', 'AILA_Q4E_RESIDENT_PREFILL': '1',
    'AILA_Q4E_HOTSET_PROMPT': '1', 'AILA_Q4E_HOTSET_PROMPT_PREFILL': '0',
    'AILA_Q4E_HOTSET_PROMPT_SWAPS': '0', 'AILA_Q4E_HOTSET_SWAPS': '0',
    'AILA_Q4E_HOTSET_INTERVAL': '4', 'AILA_Q4E_HOTSET_ADOPT_READY': '1',
    'AILA_Q4E_HOTSET_ADOPT_LIMIT': '4096', 'AILA_Q4E_WORK_SUMMARY': '1',
    'AILA_Q4E_CPU_EXPERT_DISPATCH': '0', 'AILA_Q4E_PREFILL_XMX': '1',
    'AILA_Q4E_STATIC_PINNING': '0', 'AILA_Q4E_GDN_CHUNK_MODE': '0',
    'AILA_Q4E_GROUPED_XMX_MIN_ROWS': '0', 'AILA_PROFILE_Q4E_DECODE': '0',
}


def case_name(settings):
    row4, chunks = settings['AILA_Q4E_CPU_NF4_Q8_ROW4'], settings['AILA_Q4E_CPU_Q8_CHUNKS']
    fused = settings['AILA_Q4E_ROUTER_RESIDENT_FUSED']
    block64 = settings['AILA_Q4E_CPU_NF4_Q8_BLOCK64']
    pending = settings['AILA_Q4E_DECODE_PENDING_MLP_RESIDUAL']
    down, up = settings['AILA_Q4E_HC_DECODE_DOWN_WG'], settings['AILA_Q4E_HC_DECODE_UP_STREAMS']
    if down not in ('64','128','256') or up not in ('0','1'):
        raise ValueError('H9 HC compares down WG64/128/256 and up streams=0/1')
    if row4 not in ('0', '1') or chunks not in ('0', '3') or fused not in ('0', '1') or block64 not in ('0', '1') or pending not in ('0', '1'):
        raise ValueError('H9 only compares row4/block64/fused/pending=0/1 and chunks=0/3')
    if row4 == '1' and block64 == '1':
        raise ValueError('H9 row4 and block64 cannot both be enabled')
    labels = []
    if row4 == '1':
        labels.append('row4')
    if block64 == '1':
        labels.append('block64')
    if chunks == '3':
        labels.append('chunks3')
    if fused == '1':
        labels.append('fused')
    if down != '256':
        labels.append('hc' + down)
    if up == '1':
        labels.append('hc-up')
    if pending == '1':
        labels.append('pending-mlp')
    return '-'.join(labels) or 'baseline'


def validate_run(run):
    meta, env = run['metadata'], run['metadata']['env_overrides']
    if meta.get('diagnostic_only') or meta.get('gpu_memory_sha256') or any(str(env.get(k, '0')) != '0' for k in ('AILA_Q4E_CPU_DIAGNOSTIC','AILA_Q4E_CPU_Q8_VERIFY','AILA_Q4E_CPU_A64_VERIFY','AILA_Q4E_COLD_TEST_HC_REFERENCE')):
        raise ValueError(f'{run["prefix"]}: diagnostic samples are not performance comparisons')
    if meta.get('exit_code') != 0 or meta.get('watchdog_stop') or meta.get('memory_pressure', False):
        raise ValueError(f'{run["prefix"]}: failed or memory-pressure sample')
    for key, value in FIXED_ENV.items():
        if str(env.get(key, 'missing')) != value:
            raise ValueError(f'{run["prefix"]}: H8 fixed environment differs at {key}')
    if run['effective_config']['host_ram_mb'] != '0':
        raise ValueError(f'{run["prefix"]}: host RAM override differs from effective configuration')
    inputs = meta['input']
    if (inputs['max_tokens'], inputs['requests']) not in ((24, 3), (48, 5)):
        raise ValueError(f'{run["prefix"]}: expected short screen 24/3 or formal corpus 48/5')
    if inputs['max_seq'] != 2560 or inputs.get('repeats') != 1 or meta.get('performance_preset') != 'qwen38-a770':
        raise ValueError(f'{run["prefix"]}: H8 fixed request setup differs')
    case_name({key: str(env.get(key, default)) for key, default in VARIABLES.items()})


@contextmanager
def campaign_rules():
    # Reuse H8's exact EXE/model/input/arguments/profile/stack/GPU/config/token
    # checks, parser, metrics and output format. H8's settings include its RAM
    # override; FIXED_ENV rejects any change to those original variables, so only
    # H9's implementation flags can actually vary. Missing flags use original-path defaults.
    old_variables, old_name, old_read = h8.VARIABLES, h8.case_name, h8.summarize
    def read(prefix):
        run = old_read(prefix)
        validate_run(run)
        return run
    h8.VARIABLES, h8.case_name, h8.summarize = {**old_variables, **VARIABLES}, case_name, read
    try:
        yield
    finally:
        h8.VARIABLES, h8.case_name, h8.summarize = old_variables, old_name, old_read


def summarize_runs(prefixes):
    with campaign_rules():
        return h8.summarize_runs(prefixes)


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
    result['validation_logs'] = [dict(**h8.identity(p), tail=Path(p).read_text(encoding='utf-8-sig').splitlines()[-20:]) for p in args.logs]
    sources = ['src/ops/Qwen4ExpCpuExpertOps.cpp', 'src/ops/Qwen4ExpCpuExpertOps.hpp',
        'src/ops/MoeExpertOps.cpp', 'src/ops/MoeExpertOps.hpp',
        'src/models/Qwen4ExpExpertExecutor.cpp', 'src/models/Qwen4ExpExpertExecutor.hpp',
        'src/models/Qwen4ExpExpertCache.cpp', 'src/models/Qwen4ExpExpertCache.hpp',
        'src/models/Qwen4ExpBnb4Backend.cpp', 'src/models/Qwen4ExpBnb4Backend.hpp',
        'src/ops/HyperConnectionOps.cpp', 'src/ops/HyperConnectionOps.hpp',
        'tests/models/Qwen4ExpColdAcceptance.hpp', 'tests/ops/Qwen4ExpCpuExpertTests.cpp',
        'tests/models/Qwen4ExpModuleTests.cpp',
        'tests/models/Qwen4ExpExpertReadinessTests.cpp', 'perf/qwen4exp/q4e_execution_probe.ps1',
        'perf/qwen4exp/q4e_heterogeneous_summary.py', 'perf/qwen4exp/q4e_cpu_supply_campaign.ps1',
        'perf/qwen4exp/q4e_cpu_supply_summary.py', 'perf/qwen4exp/q4e_cpu_diagnostic_summary.py']
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
    result['campaign'] = dict(name='H9 CPU supply', baseline='H8 q8-mapped-ram0-short1',
        variable_environment=VARIABLES, fixed_environment=FIXED_ENV)
    result['limitations'].extend(['H9 baseline retains H8 approximate NF4-Q8 CPU arithmetic.',
        'Row4, block64 and coarse chunks must pass bitwise CPU checks independently of model quality.',
        'Resident router fusion is a separate flag; its independent and combined cases remain distinct.',
        'Original-path H9 defaults permit historical record parsing; different EXE identities still cannot be pooled.',
        'CPU diagnostic/page-fault samples are excluded from performance comparisons.'])
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    for group in result['groups']:
        print(group['position'], group['prompt_tokens'], {name: value['median'] for name, value in group['cases'].items()})


if __name__ == '__main__':
    main()
