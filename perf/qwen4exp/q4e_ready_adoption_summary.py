"""H6 paired, position-preserving screening evidence for physical Ready adoption."""
import argparse
import json
import statistics as st
from pathlib import Path
from q4e_prompt_hotset_summary import summarize
from q4e_hotset_adaptive_summary import identity
from q4e_cold_summary import read as read_canary
from q4e_prompt_hotset_trace import inspect


def summarize_pairs(prefixes):
    runs = [summarize(p) for p in prefixes]
    assert len({r['metadata']['exe_sha256'] for r in runs}) == 1
    assert len({r['metadata']['model_config_sha256'] for r in runs}) == 1
    reference = runs[0]
    ignored = {'AILA_Q4E_HOTSET_ADOPT_READY'}
    for r in runs:
        assert r['metadata']['input'] == reference['metadata']['input']
        assert r['effective_config'] == reference['effective_config']
        env = r['metadata']['env_overrides']
        assert {k:v for k,v in env.items() if k not in ignored} == {
            k:v for k,v in reference['metadata']['env_overrides'].items() if k not in ignored}
    groups = []
    for position in range(len(reference['requests'])):
        group = dict(position=position+1, prompt_tokens=reference['requests'][position]['prompt_tokens'])
        for arm in ('control', 'adopt'):
            selected = [r['requests'][position] for r in runs if
                        r['metadata']['env_overrides']['AILA_Q4E_HOTSET_ADOPT_READY'] == str(int(arm=='adopt'))]
            assert selected and all(s['prompt_tokens']==group['prompt_tokens'] for s in selected)
            fields = {'request_ms': lambda s:s['request_ms'], 'ttft_ms': lambda s:s['ttft_ms'],
                      'decode_tok_s': lambda s:s['decode_tok_s'],
                      'prefill_demand_bytes': lambda s:s['phases']['prefill']['demand_bytes'],
                      'decode_demand_bytes': lambda s:s['phases']['decode']['demand_bytes'],
                      'refill_bytes': lambda s:s['hotset']['epoch']['refill_bytes'],
                      'ready_adoptions': lambda s:s['hotset']['epoch']['ready_adoptions'],
                      'ready_adoption_ns': lambda s:s['hotset']['epoch']['ready_adoption_ns']}
            samples = {k:[get(s) for s in selected] for k,get in fields.items()}
            cv = st.pstdev(samples['request_ms'])/st.mean(samples['request_ms'])
            group[arm] = dict(samples=samples, median={k:st.median(v) for k,v in samples.items()},
                              request_cv=cv, variable=cv>0.15)
        group['request_ratio'] = group['control']['median']['request_ms']/group['adopt']['median']['request_ms']
        groups.append(group)
    return dict(runs=runs, groups=groups,
                limitations=['Three processes per arm are screening evidence, not a production dispatch.',
                             'Keep every startup, adaptation and return-domain request at its corpus position.',
                             'CLI timing conventions are unchanged; model/profile load is outside request time.',
                             'Demand/refill bytes are scheduled payload counters, not measured PCIe bandwidth.'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('prefixes', nargs='+')
    ap.add_argument('--canaries', nargs='*', default=[])
    ap.add_argument('--traces', nargs='*', default=[])
    ap.add_argument('--logs', nargs='*', default=[])
    ap.add_argument('--excluded', nargs='*', default=[])
    ap.add_argument('--exploratory', nargs='*', default=[])
    ap.add_argument('--diagnostics', nargs='*', default=[])
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    result = summarize_pairs(args.prefixes)
    result['canaries'] = [read_canary(p) for p in args.canaries]
    result['exploratory'] = [summarize(p) for p in args.exploratory]
    result['traces'] = [inspect(p,cold_queue=2) for p in args.traces]
    result['validation_logs'] = [dict(**identity(p),tail=Path(p).read_text(encoding='utf-8-sig').splitlines()[-20:]) for p in args.logs]
    result['excluded'] = [dict(prefix=p,reason='Campaign omitted GenerationRepeats=1; manually stopped and excluded from performance comparisons.',
                               metadata=json.loads(Path(p+'.meta.json').read_text(encoding='utf-8-sig')),
                               logs=[identity(p+s) for s in ('.log','.err.log')]) for p in args.excluded]
    result['diagnostics'] = [dict(prefix=p,metadata=json.loads(Path(p+'.meta.json').read_text(encoding='utf-8-sig')),
                                 logs=[identity(p+s) for s in ('.log','.err.log')],
                                 limitations=['Instruction-count collection hit its watchdog before a complete request.',
                                              'VTune stop returned an internal error; no usable GPU timing claim.',
                                              'Original metadata is preserved; these hashes were taken after log handles closed.',
                                              'Profiler memory sampling includes immediate children, not the whole process tree.']) for p in args.diagnostics]
    sources = ['src/models/Qwen4ExpExpertCache.cpp','src/models/Qwen4ExpExpertCache.hpp',
               'src/models/Qwen4ExpBnb4Backend.cpp','tests/models/Qwen4ExpExpertReadinessTests.cpp',
               'perf/qwen4exp/q4e_execution_probe.ps1','perf/qwen4exp/q4e_ready_adoption_campaign.ps1',
               'perf/qwen4exp/q4e_ready_adoption_summary.py','perf/qwen4exp/q4e_cache_replay.py',
               'perf/qwen4exp/test_q4e_cache_replay.py','perf/qwen4exp/q4e_prompt_hotset_trace.py',
               'perf/qwen4exp/q4e_hotset_adaptive_summary.py']
    result['source_artifacts'] = [identity(p) for p in sources]
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    for group in result['groups']:
        print(group['position'],group['prompt_tokens'],
              {arm:group[arm]['median'] for arm in ('control','adopt')})
