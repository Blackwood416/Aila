"""Summarize conditional Q4E greedy MTP probes, not production generation TPS."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics as stats


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def distribution(v):
    v=sorted(v)
    return dict(count=len(v),mean=stats.mean(v),median=stats.median(v),p10=v[len(v)//10],p90=v[min(len(v)-1,len(v)*9//10)],cv=stats.pstdev(v)/stats.mean(v))


def read_run(prefix):
    meta=json.loads(Path(prefix+'.meta.json').read_text(encoding='utf-8-sig'))
    assert meta['exit_code']==0 and not meta['watchdog_stop']
    assert meta['log_sha256'].lower()==digest(prefix+'.log')
    lines=Path(prefix+'.log').read_text(encoding='utf-8').splitlines()
    def records(tag):return [json.loads(s.split('] ',1)[1]) for s in lines if s.startswith('['+tag+'] {')]
    target=records('MTP TARGET')[0];passes=records('MTP PASS');cycles=records('MTP CYCLES')[0]
    verify=records('MTP VERIFY');state=records('MTP STATE')
    loadline=next(s for s in lines if s.startswith('[MTP LOAD] '))
    load={k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',loadline)}
    p=target['prompt_tokens'];anchors=target['anchors'];tokens=target['tokens']
    assert len(passes)==int(meta['env_overrides']['AILA_Q4E_MTP_REPEATS'])
    assert len(verify)==int(meta['env_overrides']['AILA_Q4E_MTP_VERIFY_PASSES'])
    for run in passes:
        assert run['draft_misses']>=0 and run['live_usm_delta']==0
        if load.get('gpu_expert_slots',load.get('resident_experts'))==512:assert run['draft_misses']==0
        assert len(run['core_ms'])==len(run['loop_ms'])==len(run['accepted_prefix'])==anchors
        assert len(run['draft_tokens'])==anchors*3 and run['draft_tokens']==passes[0]['draft_tokens']
        for a in range(anchors):
            accepted=0
            for step in range(3):
                if run['draft_tokens'][a*3+step]!=tokens[p+a+1+step]:break
                accepted+=1
            assert accepted==run['accepted_prefix'][a]
    next_anchor=0
    for a,n in zip(cycles['anchor'],cycles['accepted_prefix']):
        assert a==next_anchor and n==passes[0]['accepted_prefix'][a]
        next_anchor+=1+n
    assert next_anchor>=anchors
    committed=sum(n+1 for n in cycles['accepted_prefix']);count=len(cycles['anchor'])
    result=dict(prefix=prefix,metadata=meta,load=load,target=target,passes=passes,cycles=cycles,verify=verify,state=state,
        sliding_prefix_histogram=dict(Counter(passes[0]['accepted_prefix'])),
        per_depth_prefix_acceptance=[sum(n>=d for n in passes[0]['accepted_prefix'])/anchors for d in [1,2,3]],
        steady_draft_core_ms=distribution([v for r in passes[1:] or passes for v in r['core_ms']]),
        selected_cycles=dict(count=count,committed=committed,mean_committed=committed/count,
            core_ms=distribution(cycles['core_ms']),loop_ms=distribution(cycles['loop_ms']),
            catchup_rows=sum(cycles['catchup_rows'])))
    if verify:
        # Use per-pass total work / block count, then median of passes after
        # the first. Keep all samples, including the first, in the report.
        per_pass=[stats.mean(r['block_ms']) for r in verify]
        cost=stats.median(per_pass[1:] or per_pass)
        optimistic=(count*cost+sum(cycles['core_ms']))/committed
        result['target_verification']=dict(per_pass_mean_block_ms=per_pass,post_first_median_block_ms=cost)
        result['optimistic_cost_model']=dict(ms_per_committed_token=optimistic,tokens_per_s=1000/optimistic,
            verify_only_zero_draft_tokens_per_s=1000*committed/(count*cost),
            all_four_accepted_zero_draft_tokens_per_s=4000/cost,
            note='Combines measured costs from separate phases on the same fixed trajectory. Excludes target rollback/replay, sampling/IPC and cache interleaving effects; not measured generation throughput.')
    if state:
        assert state[0]['replay_bitwise'] and state[0]['live_usm_delta']==0
        result['checkpoint_cost']=dict(bytes=state[0]['bytes'],capture_ms=distribution(state[0]['capture_ms']),restore_ms=distribution(state[0]['restore_ms']),
            note='One checkpoint before the block. Accepted-prefix replay cost is not included in these copy timings.')
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('prefixes',nargs='+');ap.add_argument('--output',required=True)
    a=ap.parse_args();runs=[read_run(p) for p in a.prefixes]
    result=dict(caveats=[__doc__,'Three recursive draft tokens. Sliding anchors and selected cycles are different statistics.',
        'Selected cycles catch up MTP KV with target hidden states; no target accepted-prefix state commit.',
        'MTP GPU cache size and charged target VRAM reserve are reported per run.',
        'Reference numeric checks are separate; no exact target distribution preservation claim.'],runs=runs,
        memory=dict(peak_working_set_gib=max(r['metadata']['peak_working_set_bytes'] for r in runs)/(1<<30),
            minimum_system_free_gib=min(r['metadata']['minimum_system_free_mb'] for r in runs)/1024))
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    for r in runs:print(r['prefix'],r['sliding_prefix_histogram'],r['selected_cycles'],r.get('optimistic_cost_model'))
    print(result['memory'])
