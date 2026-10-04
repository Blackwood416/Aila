"""Bounded original NF4 CPU copies on the fixed H9 pending-only setup."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import re
import q4e_cpu_supply_summary as h9
import q4e_heterogeneous_summary as h8

VARIABLES={'AILA_Q4E_CPU_PACKED_CACHE_MB':'0','AILA_Q4E_CPU_PACKED_CACHE_PROMPT':'0'}
FIXED={**h9.FIXED_ENV,**h9.VARIABLES,'AILA_Q4E_DECODE_PENDING_MLP_RESIDUAL':'1'}
PACKED_SLOT_BYTES=2457600

def case_name(settings):
    mb=settings['AILA_Q4E_CPU_PACKED_CACHE_MB']
    prompt=settings.get('AILA_Q4E_CPU_PACKED_CACHE_PROMPT','0')
    if prompt not in ('0','1') or (prompt=='1' and mb=='0'):
        raise ValueError('Prompt CPU cache requires a positive cache and prompt=0/1')
    if mb not in ('0','1024','2048','4096'):
        raise ValueError('CPU cache compares0/1024/2048/4096 MiB')
    return 'baseline' if mb=='0' else 'cpu-cache'+mb+('-prompt' if prompt=='1' else '')

def validate(run):
    h9.validate_run(run)
    env=run['metadata']['env_overrides']
    for key,default in FIXED.items():
        if str(env.get(key,default))!=default:
            raise ValueError(f'{run["prefix"]}: fixed H9 pending setup differs at{key}')
    mb=int(env.get('AILA_Q4E_CPU_PACKED_CACHE_MB','0'))
    prompt=int(env.get('AILA_Q4E_CPU_PACKED_CACHE_PROMPT','0'))
    case_name({'AILA_Q4E_CPU_PACKED_CACHE_MB':str(mb),'AILA_Q4E_CPU_PACKED_CACHE_PROMPT':str(prompt)})
    text=Path(run['prefix']+'.err.log').read_text(encoding='utf-8')
    run['cpu_cache_log_identity']=h8.identity(run['prefix']+'.err.log')
    for suffix,field in [('.err.log','err_log_sha256'),('.log','log_sha256')]:
        if h8.identity(run['prefix']+suffix)['sha256']!=run['metadata'][field].lower():
            raise ValueError('CPU cache evidence log hash differs')
    caches=[line for line in text.splitlines() if '[Q4E CpuPackedCache]' in line]
    if bool(caches)!=bool(mb) or len(caches)>1:
        raise ValueError('Expected exactly one CPU cache initialization only for enabled cache')
    if mb:
        if 'selection=offline-profile-exclude-initial-gpu ' not in caches[0]:
            raise ValueError('CPU cache selection differs')
        cache={k:float(v) if '.' in v else int(v) for k,v in re.findall(r'([a-z_]+)=([0-9.]+)',caches[0])}
        if (cache['requested_mb'],cache['packed_slot_bytes'],cache['slots'],cache['bytes'],cache['pageable'])!=(mb,PACKED_SLOT_BYTES,mb*1048576//PACKED_SLOT_BYTES,(mb*1048576//PACKED_SLOT_BYTES)*PACKED_SLOT_BYTES,1):
            raise ValueError('CPU cache geometry/budget differs')
        run['cpu_cache_init']=cache
    active=None;observations=[]
    for line in text.splitlines():
        if '[GenerateMessages] Full prefill:' in line:
            active=dict(cached_tasks=0,mapped_tasks=0,cpu_tasks=0,refreshes=[])
        if active is not None and '[Q4E CpuPackedPrompt]' in line:
            if not prompt:raise ValueError('Disabled prompt cache emitted a refresh')
            refresh={k:float(v) if '.' in v else int(v) for k,v in re.findall(r'([a-z_]+)=([0-9.]+)',line)}
            if refresh['capacity_bytes']!=run['cpu_cache_init']['bytes'] or refresh['copied_bytes']>refresh['capacity_bytes'] or refresh['copied_bytes']%PACKED_SLOT_BYTES or refresh['pageable']!=1 or 'selection=past-prefill-exclude-ready ' not in line:
                raise ValueError('CPU prompt cache refresh geometry/selection differs')
            active['refreshes'].append(refresh)
        if active is not None and '[Q4E Cold]' in line:
            row={k:int(v) for k,v in re.findall(r'([a-z_]+)=(\d+)',line)}
            if row['count']==1:
                if row['mapped_tasks']+row['cpu_cache_tasks']!=row['cpu_tasks']:
                    raise ValueError('CPU cache/mapped tasks do not cover zero-host-pool CPU work')
                for dest,src in [('cached_tasks','cpu_cache_tasks'),('mapped_tasks','mapped_tasks'),('cpu_tasks','cpu_tasks')]:active[dest]+=row[src]
        if '[GenerateMessages] Prompt=' in line:
            if active is None:raise ValueError('Incomplete CPU cache observations')
            observations.append(active);active=None
    if active is not None or len(observations)!=len(run['requests']):
        raise ValueError('CPU cache history length differs')
    if bool(sum(r['cached_tasks'] for r in observations))!=bool(mb):
        raise ValueError('Enabled cache must actually serve CPU work; disabled cache must not')
    for request,observation in zip(run['requests'],observations):
        if len(observation['refreshes'])!=prompt:
            raise ValueError('CPU prompt cache must refresh once per completed prefill only when enabled')
        if prompt and observation['refreshes'][0]['rows']!=request['prompt_tokens']:
            raise ValueError('CPU prompt cache route count differs from completed prefill')
        request['cpu_cache']=dict(**observation,hit_fraction=observation['cached_tasks']/observation['cpu_tasks'] if observation['cpu_tasks'] else None)

@contextmanager
def rules():
    old_vars,old_name,old_read=h8.VARIABLES,h8.case_name,h8.summarize
    def read(prefix):
        run=old_read(prefix);validate(run);return run
    h8.VARIABLES,h8.case_name,h8.summarize={**old_vars,**VARIABLES},case_name,read
    try:yield
    finally:h8.VARIABLES,h8.case_name,h8.summarize=old_vars,old_name,old_read

def summarize_runs(prefixes):
    with rules():return h8.summarize_runs(prefixes)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefixes',nargs='+');parser.add_argument('--output',required=True)
    parser.add_argument('--source-manifest');parser.add_argument('--canaries',nargs='*',default=[])
    args=parser.parse_args();result=summarize_runs(args.prefixes)
    result['campaign']=dict(name='H10 bounded packed CPU cache',variable_environment=VARIABLES,fixed_environment=FIXED)
    result['canaries']=[h8.read_canary(p) for p in args.canaries]
    if args.source_manifest:
        manifest=json.loads(Path(args.source_manifest).read_text(encoding='utf-8-sig'))
        if manifest['exe_sha256'].lower()!=result['comparison_identity']['exe_sha256'].lower():raise ValueError('Source manifest EXE differs')
        for row in manifest['files']:
            if h8.identity(row['snapshot'])['sha256']!=row['sha256'].lower():raise ValueError('Source snapshot hash differs')
        result['compiled_source_manifest']=manifest
        result['compiled_source_manifest_identity']=h8.identity(args.source_manifest)
    result['limitations'].extend(['CPU cache copies only original packed bytes; nested metadata still borrows original mapping.',
        'CPU cache is pageable anonymous memory, not locked or guaranteed physically resident.',
        'Independent offline profile excludes only initial GPU hot tier; later ready adoption may overlap it.',
        'Initialization copying is outside request timing and its duration/bytes remain recorded.',
        'Expanded host pool stays zero; added CPU cache has a separate explicit RAM budget.'])
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    for case in result['cases']:print(case['name'],case['corpus_request_ms'])

if __name__=='__main__':main()
