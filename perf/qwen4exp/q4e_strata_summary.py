"""S1/S2 conditional MTP costs. Does not measure a production generation loop."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics as st


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dist(values):
    v = sorted(values)
    return dict(n=len(v), mean=st.mean(v), median=st.median(v),
                p10=v[len(v)//10], p90=v[min(len(v)-1, len(v)*9//10)],
                cv=st.pstdev(v)/st.mean(v) if st.mean(v) else 0)


def read_run(prefix):
    meta = json.loads(Path(prefix+'.meta.json').read_text(encoding='utf-8-sig'))
    assert meta['exit_code'] == 0 and not meta['watchdog_stop'], prefix
    if meta['env_overrides'].get('AILA_Q4E_MTP_COMPARE_RESIDENT') == '1':
        raise ValueError('Use q4e_resident_summary.py for paired resident-split trials')
    assert digest(prefix+'.log') == meta['log_sha256'].lower()
    lines = Path(prefix+'.log').read_text(encoding='utf-8').splitlines()
    def rec(tag):
        return [json.loads(s.split('] ', 1)[1]) for s in lines if s.startswith('['+tag+'] {')]
    target = rec('MTP TARGET')[0]
    passes, cycles, verify = rec('MTP PASS'), rec('MTP CYCLES'), rec('MTP VERIFY')
    assert len(passes) == int(meta['env_overrides']['AILA_Q4E_MTP_REPEATS'])
    sweep = meta['env_overrides'].get('AILA_Q4E_MTP_WINDOW_SWEEP') == '1'
    assert len(verify) == int(meta['env_overrides']['AILA_Q4E_MTP_VERIFY_PASSES'])*(4 if sweep else 1)
    compare_kv = meta['env_overrides'].get('AILA_Q4E_MTP_COMPARE_KV') == '1'
    assert len(cycles) == (len(passes)*(4 if sweep else 2) if sweep or compare_kv else 1)
    p, anchors, tokens = target['prompt_tokens'], target['anchors'], target['tokens']
    for r in passes:
        assert r['draft_tokens'] == passes[0]['draft_tokens'] and r['live_usm_delta'] == 0
        for a, accepted in enumerate(r['accepted_prefix']):
            n = 0
            for d in range(3):
                if r['draft_tokens'][3*a+d] != tokens[p+a+1+d]: break
                n += 1
            assert n == accepted
    for c in cycles:
        a, width = 0, c.get('width', 4)
        for pos, n in zip(c['anchor'], c['accepted_prefix']):
            assert pos == a and n == min(width-1, passes[0]['accepted_prefix'][a])
            a += 1+n
        assert a >= anchors and c['continuation_bitwise'] and c['live_usm_delta'] == 0
    summaries = []
    for width, kv in sorted({(c.get('width', 4), c['kv_only']) for c in cycles}):
        cc = [c for c in cycles if c.get('width', 4) == width and c['kv_only'] == kv]
        vv = [v for v in verify if v.get('width', 4) == width]
        committed = sum(n+1 for n in cc[0]['accepted_prefix'])
        count = len(cc[0]['anchor'])
        draft_costs = [st.mean(c['core_ms']) for c in cc]
        draft_cost = st.median(draft_costs)
        s = dict(width=width, kv_only=kv, cycle_count=count, committed=committed,
                 mean_advance=committed/count, draft_pass_mean_ms=draft_costs,
                 draft_median_pass_mean_ms=draft_cost,
                 draft_cycles_ms=dist([t for c in cc for t in c['core_ms']]),
                 catchup_ms=dist([t for c in cc for t in c['catchup_ms']]),
                 demand_bytes=[c['draft_demand_bytes'] for c in cc],
                 misses=[c['draft_misses'] for c in cc])
        if vv:
            means = [st.mean(v['block_ms']) for v in vv]
            verify_cost = st.median(means[1:] or means)
            s.update(verify_pass_mean_ms=means, verify_ms=verify_cost,
                     conditional_tokens_s=1000*committed/count/(verify_cost+draft_cost),
                     free_draft_tokens_s=1000*committed/count/verify_cost,
                     commit_budget_for_20_ms=50*committed/count-verify_cost-draft_cost,
                     verify_block_ms=dist([t for v in vv[1:] or vv for t in v['block_ms']]),
                     verify_demand_bytes=[v['demand_bytes'] for v in vv])
            if width > 1 and vv[0].get('union_counts'):
                s['mean_union_per_layer'] = st.mean(vv[-1]['union_counts'])/48
                s['ready_fraction_at_router'] = sum(vv[-1]['ready_counts'])/sum(vv[-1]['union_counts'])
            if vv[0].get('top1'):
                actual = vv[0]['top1']
                reference = tokens[p+1:p+1+len(actual)]
                s['known_target_top1_rows'] = len(reference)
                s['known_target_top1_mismatches'] = [i for i,(a,b) in enumerate(zip(actual,reference)) if a != b]
                for v in vv: assert v['top1'] == actual
        summaries.append(s)
    return dict(prefix=prefix, metadata=meta, target=target, passes=passes, cycles=cycles,
                verify=verify, equivalence=rec('MTP KV EQUIVALENCE'), state=rec('MTP STATE'),
                accepted_histogram=dict(Counter(passes[0]['accepted_prefix'])), summaries=summaries)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('prefixes', nargs='+'); ap.add_argument('--output', required=True)
    args = ap.parse_args(); runs = [read_run(p) for p in args.prefixes]
    result = dict(runs=runs, limitations=[
        'Costs from separately measured draft cycles and target trajectory blocks; not interleaved generation throughput.',
        'Excludes target prefix commit/replay, sampling/IPC, reject-path target input changes and cache interference.',
        'Conditional acceptance uses single-row target decisions. Multi-row decision differences are reported separately.',
        'Prior whole-step CPU numeric reference failures remain unresolved; local KV equivalence does not replace them.',
        'Union counts are sums across 48 layers. Ready is cache residency at routing, not completed DMA.'],
        memory=dict(peak_ws_gib=max(r['metadata']['peak_working_set_bytes'] for r in runs)/2**30,
                    min_free_gib=min(r['metadata']['minimum_system_free_mb'] for r in runs)/1024))
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    for r in runs:
        print(r['prefix'], r['accepted_histogram'])
        for s in r['summaries']: print({k:v for k,v in s.items() if k in ['width','kv_only','mean_advance','draft_median_pass_mean_ms','verify_ms','conditional_tokens_s','commit_budget_for_20_ms','known_target_top1_mismatches','mean_union_per_layer','ready_fraction_at_router']})
    print(result['memory'])
