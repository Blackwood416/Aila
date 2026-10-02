"""Compare bounded Prefill scratch lifetimes with unchanged arithmetic."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics as st


def read(prefix):
    meta=json.loads(Path(prefix+'.meta.json').read_text(encoding='utf-8-sig'))
    assert meta['exit_code']==0 and not meta['watchdog_stop']
    assert hashlib.sha256(Path(prefix+'.log').read_bytes()).hexdigest()==meta['log_sha256'].lower()
    lines=Path(prefix+'.log').read_text(encoding='utf-8').splitlines()
    pf=[dict(tokens=int(n),ms=float(ms),hash=h) for n,ms,h in
        [re.search(r'tokens=(\d+) ms=([\d.]+) logits_fnv64=(\d+)',line).groups() for line in lines if line.startswith('[X2 MEMORY FORWARD]')]]
    tail=[json.loads(line.split('] ',1)[1]) for line in lines if line.startswith('[S4 TAIL]')]
    mem=[dict(re.findall(r'(\w+)=(\S+)',line)) for line in lines if line.startswith('[X2 MEMORY]')]
    phases=[dict((k,int(v)) for k,v in re.findall(r'(\w+)=(\d+)',line)) for line in
        Path(prefix+'.err.log').read_text(encoding='utf-8').splitlines() if '[Q4E PhaseScratch]' in line]
    assert len(pf)==len(tail)==meta['input']['repeats']
    assert all(p['hash']==pf[0]['hash'] for p in pf)
    assert all(t['hashes']==tail[0]['hashes'] and t['live_usm_delta']==0 for t in tail)
    return dict(prefix=prefix,metadata=meta,prefill=pf,tail=tail,memory=mem,phase_allocations=phases,
        warmed_prefill_tok_s=pf[0]['tokens']*1000/st.median(p['ms'] for p in pf[1:]),
        warmed_tail_ms=st.median(st.mean(t['ms']) for t in tail[1:]))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('prefixes',nargs='+');ap.add_argument('--output',required=True)
    a=ap.parse_args();runs=[read(p) for p in a.prefixes];comparisons=[]
    assert len(runs)%2==0
    for base,candidate in zip(runs[::2],runs[1::2]):
        assert base['metadata']['input']==candidate['metadata']['input']
        assert base['metadata']['exe_sha256']==candidate['metadata']['exe_sha256']
        assert base['prefill'][0]['hash']==candidate['prefill'][0]['hash']
        assert base['tail'][0]['hashes']==candidate['tail'][0]['hashes']
        b=int(base['memory'][-1]['device_bytes']);c=int(candidate['memory'][-1]['device_bytes'])
        expected=candidate['phase_allocations'][-1]['saved_bytes']
        assert b-c==expected,(b-c,expected)
        comparisons.append(dict(baseline=base['prefix'],candidate=candidate['prefix'],
            prefill_and_eight_tail_rows_bitwise=True,device_bytes_saved=b-c,
            base_prefill_tok_s=base['warmed_prefill_tok_s'],candidate_prefill_tok_s=candidate['warmed_prefill_tok_s'],
            base_tail_ms=base['warmed_tail_ms'],candidate_tail_ms=candidate['warmed_tail_ms']))
    result=dict(runs=runs,comparisons=comparisons,limitations=[
        'Same process warmed replay, three post-first samples; no OS cache reset, no speedup claim for small/noisy wall differences.',
        'Device bytes are the USM ledger, not measured WDDM dedicated residency. Host pool remains 16 GiB.',
        'Tail is eight fixed teacher-forced tokens, not generated throughput; serving cancellation/interleaving not tested here.'])
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    for c in comparisons:print(c)
