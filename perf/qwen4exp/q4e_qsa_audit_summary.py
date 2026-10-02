"""Summarize isolated QSA costs and diagnostic model stages; never infer generated TPS."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics as st


def artifact(prefix):
    meta = json.loads(Path(prefix+'.meta.json').read_text(encoding='utf-8-sig'))
    assert meta['exit_code'] == 0 and not meta['watchdog_stop'] and not meta.get('memory_pressure', False)
    assert meta['diagnostic_only']
    for suffix, key in [('.log', 'log_sha256'), ('.err.log', 'err_log_sha256')]:
        assert hashlib.sha256(Path(prefix+suffix).read_bytes()).hexdigest() == meta[key].lower()
    return meta


def curve(prefix):
    meta = artifact(prefix)
    lines = Path(prefix+'.log').read_text(encoding='utf-8').splitlines()
    rows = [json.loads(line.split('] ', 1)[1]) for line in lines if line.startswith('[QSA COST]')]
    checks = [line for line in lines if line.startswith('[QSA CHECK]')]
    assert len(rows) == 9*5*3 and len(checks) == 18
    assert any(line.startswith('[QSA COST END]') for line in lines)
    result = []
    for length in sorted({r['length'] for r in rows}):
        out = dict(length=length)
        for stage in sorted({r['stage'] for r in rows}):
            selected = [r for r in rows if r['length'] == length and r['stage'] == stage]
            assert len(selected) == 3
            out[stage] = dict(median_wall_us=st.median(r['median_us'] for r in selected),
                             median_queued_wall_us=st.median(r['queued_wall_us'] for r in selected),
                             rounds=selected)
        result.append(out)
    return dict(prefix=prefix, metadata=meta, curve=result, checks=checks)


def stages(prefix):
    meta = artifact(prefix)
    assert meta['input']['max_tokens'] == 16 and meta['input']['repeats'] == 2
    lines = Path(prefix+'.err.log').read_text(encoding='utf-8').splitlines()
    profile, rows, completions = None, [], []
    for line in lines:
        if '[Q4E Profile]' in line:
            profile = {k: float(v) for k, v in re.findall(r'(pos|count|total|fill|pf_wait|xfer)=([0-9.]+)', line)}
        if '[Q4E Stages]' in line and profile and profile['count'] == 1:
            rows.append(dict(profile, **{k: float(v) for k, v in re.findall(r'(ple|hc|linear_attn|full_attn|router|experts|acquire|shared|head)=([0-9.]+)', line)}))
        match = re.search(r'\[GenerateMessages\] Prompt=(\d+) Generated=(\d+)', line)
        if match:
            completions.append(tuple(map(int, match.groups())))
    assert len(completions) == 2 and all(c == completions[0] for c in completions)
    prompt, tokens = completions[-1]
    assert tokens == 16
    rows = rows[-tokens:]
    assert [r['pos'] for r in rows] == list(range(prompt, prompt+tokens))
    means = {k: st.mean(r[k] for r in rows) for k in rows[0] if k not in ('pos', 'count')}
    return dict(prefix=prefix, metadata=meta, prompt=prompt, last_request_rows=rows, means=means,
                interpretation='Stage synchronization changes scheduling. Means are diagnostic milliseconds; xfer is demand MiB. Includes final alignment. Different prompt/output trajectories, not a causal context-only comparison.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--curve', required=True)
    ap.add_argument('--short', required=True)
    ap.add_argument('--long', required=True)
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    short, long = stages(a.short), stages(a.long)
    assert short['metadata']['exe_sha256'] == long['metadata']['exe_sha256']
    assert short['metadata']['model_config_sha256'] == long['metadata']['model_config_sha256']
    assert short['metadata']['env_overrides'] == long['metadata']['env_overrides']
    assert short['metadata']['input']['max_seq'] == long['metadata']['input']['max_seq'] == 2560
    result = dict(qsa=curve(a.curve), model_stages=[short, long], limitations=[
        'QSA uses fixed synthetic prepared BF16 data and wall timings, not hardware event times or production throughput.',
        'Selected block IDs checked against host oracle; attention checked against CPU f64 for two heads, finite values for all 24. Indexer scoring numerics not newly certified here.',
        'Post-projection path excludes Q/K/V projections, indexer projection, norm/RoPE, compression, gating and output projection.',
        'Model stages serialize execution; different prompts and generated tokens can change expert demand. No direct WDDM residency measurement.'])
    Path(a.output).write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({s['prefix']: s['means'] for s in result['model_stages']}, indent=2))
