"""R1b attention tiling: keep driver, binary, routing window and tile separate."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

from q4e_architecture_summary import read


def driver(meta):
    return next(g['DriverVersion'] for g in meta['gpu'] if 'A770' in g['Name'])


def canary(prefix):
    meta = json.loads(Path(prefix + '.meta.json').read_text(encoding='utf-8-sig'))
    assert meta['exit_code'] == 0 and not meta['watchdog_stop'] and not meta.get('memory_pressure', False)
    for suffix, key in [('.log', 'log_sha256'), ('.err.log', 'err_log_sha256')]:
        assert hashlib.sha256(Path(prefix + suffix).read_bytes()).hexdigest() == meta[key].lower()
    log = Path(prefix + '.log').read_text(encoding='utf-8')
    marker = {'tile': '[R1b PASS]', 'window': '[R1b QUALITY]', 'compact': '[COMPACT PASS]', 'preset': '[R0 PASS]'}[meta['workload']]
    assert marker in log
    return dict(prefix=prefix, metadata=meta, observations=[s for s in log.splitlines()
                if s.startswith(('[R0 ', '[R1b ', '[COMPACT ', '[X2 QUALITY]'))])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('prefixes', nargs='+')
    ap.add_argument('--canary', nargs='*', default=[])
    ap.add_argument('--vtune-report')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    runs = [read(p) for p in args.prefixes]
    groups = {}
    for run in runs:
        meta, cfg = run['metadata'], run['effective_config']
        assert meta['model_config_sha256'] == runs[0]['metadata']['model_config_sha256']
        assert meta['input'] == runs[0]['metadata']['input']
        assert cfg['host_ram_mb'] == '32768' and cfg['slots'] == '3567'
        key = (driver(meta), meta['exe_sha256'], int(cfg['chunk']), int(cfg.get('attn_tile', 0)), int(cfg.get('compact_acts', 0)))
        groups.setdefault(key, []).append(run)
    summaries = []
    for (drv, exe, chunk, tile, compact), rs in groups.items():
        warm = [r for run in rs for r in run['requests'][1:]]
        median = lambda key: statistics.median(r[key] for r in warm)
        summaries.append(dict(driver=drv, exe_sha256=exe, chunk=chunk, attention_tile=tile, compact_acts=compact,
            prefixes=[r['prefix'] for r in rs], warm_ttft_ms=[r['ttft_ms'] for r in warm],
            warm_decode_tok_s=[r['decode_tok_s'] for r in warm], median_ttft_ms=median('ttft_ms'),
            median_decode_tok_s=median('decode_tok_s'), median_request_ms=median('request_ms'),
            input_tok_s=1833000/median('ttft_ms'),
            median_prefill_demand_gb=statistics.median(r['phases']['prefill']['demand_bytes'] for r in warm)/1e9,
            median_decode_host_misses=statistics.median(r['phases']['decode']['host_misses'] for r in warm),
            peak_device_gib=max(r['device_bytes'] for run in rs for r in run['requests'])/2**30,
            minimum_free_ram_gib=min(r['warm_summary']['minimum_available_ram_gib'] for r in rs)))
    vtune = None
    if args.vtune_report:
        report = Path(args.vtune_report)
        with report.open(encoding='utf-8-sig', newline='') as stream:
            rows = [row for row in csv.DictReader(stream, delimiter=chr(9)) if row.get('GPU Instructions Executed')]
        vtune = dict(path=str(report), sha256=hashlib.sha256(report.read_bytes()).hexdigest(), instructions=rows)
    result = dict(runs=runs, groups=summaries, canaries=[canary(p) for p in args.canary], vtune=vtune, limitations=[
        'Historical driver comparisons are not same-day driver A/B experiments; no automatic driver attribution.',
        'Only same-driver same-binary groups with other configuration fixed isolate the candidate.',
        'Input tokens/TTFT is warm effective throughput; demand counters are logical payload, not bus bytes.',
        'USM ledger is not WDDM residency; expert slots remain fixed in this campaign.',
    ])
    Path(args.output).write_text(json.dumps(result, indent=2) + chr(10), encoding='utf-8')
    print(json.dumps(summaries, indent=2))
