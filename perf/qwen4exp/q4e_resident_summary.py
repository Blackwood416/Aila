"""Paired, same-model resident/missing expert boundary trials for small rows."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics as st


def summarize(prefix):
    meta=json.loads(Path(prefix+'.meta.json').read_text(encoding='utf-8-sig'))
    assert meta['exit_code']==0 and not meta['watchdog_stop']
    assert meta['env_overrides']['AILA_Q4E_MTP_COMPARE_RESIDENT']=='1'
    assert hashlib.sha256(Path(prefix+'.log').read_bytes()).hexdigest()==meta['log_sha256'].lower()
    lines=Path(prefix+'.log').read_text(encoding='utf-8').splitlines()
    def records(name):return [json.loads(s.split('] ',1)[1]) for s in lines if s.startswith('['+name+'] {')]
    target=records('MTP TARGET')[0];verify=records('MTP VERIFY');cycles=records('MTP CYCLES')
    passes=int(meta['env_overrides']['AILA_Q4E_MTP_VERIFY_PASSES'])
    assert len(verify)==2*passes
    base=[v for v in verify if not v['resident_split']]
    split=[v for v in verify if v['resident_split']]
    assert len(base)==len(split)==passes and passes>0
    reference=base[0]['hashes']
    assert all(v['hashes']==reference and v['top1']==base[0]['top1'] for v in verify)
    assert all(v['width']==4 and len(v['block_ms'])*4==v['tokens'] for v in verify)
    paired=[]
    for a,b in zip(base,split):
        assert a['pass']==b['pass']
        x,y=st.mean(a['block_ms']),st.mean(b['block_ms'])
        paired.append(dict(pass_id=a['pass'],base_ms=x,split_ms=y,reduction=1-y/x,
            base_bytes=a['demand_bytes'],split_bytes=b['demand_bytes'],base_misses=a['misses'],split_misses=b['misses']))
    warm=paired[1:] or paired
    x,y=st.median(p['base_ms'] for p in warm),st.median(p['split_ms'] for p in warm)
    out=dict(prefix=prefix,metadata=meta,target=target,cycles=cycles,verify=verify,paired=paired,
        result=dict(base_ms=x,split_ms=y,median_paired_reduction=st.median(p['reduction'] for p in warm),
                    base_pass_cv=st.pstdev(p['base_ms'] for p in warm)/st.mean(p['base_ms'] for p in warm),
                    split_pass_cv=st.pstdev(p['split_ms'] for p in warm)/st.mean(p['split_ms'] for p in warm),
                    logits_bitwise=True,compared_rows=len(reference)))
    kv=[c for c in cycles if c['kv_only']]
    if kv:
        advance=sum(n+1 for n in kv[0]['accepted_prefix'])/len(kv[0]['anchor'])
        draft=st.median(st.mean(c['core_ms']) for c in kv)
        out['conditional_cost']=dict(advance=advance,draft_ms=draft,base_tok_s=1000*advance/(x+draft),
            split_tok_s=1000*advance/(y+draft),commit_budget_for20_ms=50*advance-y-draft)
    return out


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('prefixes',nargs='+');ap.add_argument('--output',required=True)
    a=ap.parse_args();runs=[summarize(p) for p in a.prefixes]
    result=dict(runs=runs,limitations=['Fixed target trajectory, not a production speculative loop.',
        'No target prefix commit/replay, rejected draft target path, IPC/sampling or full interleaving cost.',
        'First pair retained; post-first per-pass means drive the summary. Cache/prefetch state is not forcibly restored.',
        'Only GPU ALU expert-major small rows with existing demand microbatch enabled; default unchanged.'])
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    for r in runs:print(r['prefix'],r['paired'],r['result'],r.get('conditional_cost'))
