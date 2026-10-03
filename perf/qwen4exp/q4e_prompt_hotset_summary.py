"""Request-by-request H5 evidence; never pool different prompt positions/domains."""
import argparse
import hashlib
import json
import re
from pathlib import Path
from q4e_hotset_adaptive_summary import read


def summarize(prefix):
    run = read(prefix)
    run.pop('warm_summary',None)  # A corpus sequence has no meaningful mixed-domain warm median.
    samples, active = [], None
    for line in Path(prefix+'.err.log').read_text(encoding='utf-8').splitlines():
        if '[GenerateMessages] Full prefill:' in line:
            active = dict(handoffs=[], prefill_handoffs=[], decode_windows=[])
        if active is None:
            continue
        if '[Q4E PromptHotset]' in line:
            active['handoffs'].append({k:float(v) if '.' in v else int(v)
                for k,v in re.findall(r'([a-z_]+)=([0-9.]+)',line)})
        if '[Q4E PrefillHotset]' in line:
            active['prefill_handoffs'].append({k:int(v) for k,v in re.findall(r'([a-z_]+)=(\d+)',line)})
        if '[Q4E Cold]' in line and 'count=1 ' in line:
            row = {k:int(v) for k,v in re.findall(r'([a-z_]+)=(\d+)',line)}
            active['decode_windows'].append(row)
        if '[GenerateMessages] Prompt=' in line:
            windows=active['decode_windows']
            active['decode_windows']=[dict(tokens=len(windows[i:i+8]),
                **{k:sum(r[k] for r in windows[i:i+8]) for k in
                   ('direct','cold','cpu_tasks','compute_ns','join_ns','acquire_calls')})
                for i in range(0,len(windows),8)]
            samples.append(active);active=None
    assert len(samples)==len(run['requests'])
    for r,s in zip(run['requests'],samples):
        r['prompt_adaptation']=s
    return run


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('prefixes',nargs='+');ap.add_argument('--output',required=True)
    args=ap.parse_args()
    runs=[summarize(p) for p in args.prefixes]
    sources=['src/models/Qwen4ExpHotsetPolicy.hpp','src/models/Qwen4ExpExpertCache.cpp',
             'src/models/Qwen4ExpExpertCache.hpp','tests/models/Qwen4ExpExpertReadinessTests.cpp',
             'src/models/Qwen4ExpBnb4Backend.cpp','src/models/IModelBackend.hpp',
             'src/engine/EngineGeneration.cpp','include/engine/Engine.hpp','tests/models/Qwen4ExpGenerationTests.cpp',
             'perf/qwen4exp/q4e_execution_probe.ps1','perf/qwen4exp/q4e_prompt_hotset_summary.py',
             'perf/qwen4exp/q4e_prompt_hotset_audit.py','perf/qwen4exp/q4e_prompt_hotset_trace.py',
             'perf/qwen4exp/q4e_cache_replay.py','perf/qwen4exp/q4e_architecture_summary.py',
             'perf/qwen4exp/test_q4e_cache_replay.py','perf/qwen4exp/test_q4e_prompt_hotset_audit.py']
    result=dict(runs=runs,source_artifacts=[dict(path=p,sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest()) for p in sources])
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    for r in runs:
        print(r['prefix'])
        for idx,q in enumerate(r['requests']):
            print(idx+1,q['prompt_tokens'],q['decode_tok_s'],q['ttft_ms'],q['request_ms'],q['prompt_adaptation']['handoffs'])
