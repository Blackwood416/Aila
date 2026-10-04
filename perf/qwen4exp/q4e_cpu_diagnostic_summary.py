"""H9 process and CPU phase diagnostics, excluded from performance comparisons."""
import argparse
import json
import re
from pathlib import Path
from q4e_hotset_adaptive_summary import identity


def read(prefix):
    meta = json.loads(Path(prefix + '.meta.json').read_text(encoding='utf-8-sig'))
    if (not meta['diagnostic_only'] or meta['exit_code'] != 0 or meta['watchdog_stop']
            or meta.get('memory_pressure') or meta['env_overrides'].get('AILA_Q4E_CPU_DIAGNOSTIC') != '1'
            or meta['env_overrides'].get('AILA_Q4E_CPU_Q8_VERIFY', '0') != '0'):
        raise ValueError('Expected a successful CPU diagnostic run')
    for suffix, key in (('.log', 'log_sha256'), ('.err.log', 'err_log_sha256')):
        if identity(prefix + suffix)['sha256'] != meta[key].lower():
            raise ValueError('Diagnostic log hash differs from metadata')
    requests, active, pending = [], None, None
    rows = []
    fields = ('batches', 'failures', 'process_faults', 'process_user_ns', 'process_kernel_ns',
              'gate_up_ns', 'quantize_ns', 'down_ns', 'gate_chunks', 'down_chunks')
    for line in Path(prefix + '.err.log').read_text(encoding='utf-8').splitlines():
        if '[GenerateMessages] Full prefill:' in line:
            active = []
        if '[Q4E CpuDiagnostic]' in line:
            if pending is not None:
                raise ValueError('Unpaired CPU diagnostic row')
            pending = {k: int(v) for k, v in re.findall(r'([a-z_]+)=(\d+)', line)}
            if any(k not in pending for k in fields) or not pending['batches'] or pending['failures']:
                raise ValueError('Missing or failed process/phase counters')
        if '[Q4E Cold]' in line and pending is not None:
            cold = {k: int(v) for k, v in re.findall(r'([a-z_]+)=(\d+)', line)}
            if (cold['pos'], cold['count']) != (pending['pos'], pending['count']):
                raise ValueError('CPU phase and cold rows disagree')
            phase_ns = sum(pending[k] for k in ('gate_up_ns', 'quantize_ns', 'down_ns'))
            if phase_ns > cold['compute_ns'] or not phase_ns or not cold['cpu_tasks']:
                raise ValueError('CPU phases do not fit compute interval')
            pending.update(compute_ns=cold['compute_ns'], cpu_tasks=cold['cpu_tasks'],
                           compute_residual_ns=cold['compute_ns'] - phase_ns)
            rows.append(pending)
            if active is not None:
                active.append(pending)
            pending = None
        if '[GenerateMessages] Prompt=' in line:
            if active is None or not active:
                raise ValueError('Request lacks CPU diagnostics')
            decode = [r for r in active if r['count'] == 1]
            requests.append(dict(position=len(requests) + 1, rows=active,
                decode_forwards=len(decode), decode_totals={k: sum(r[k] for r in decode)
                    for k in (*fields, 'compute_ns', 'cpu_tasks', 'compute_residual_ns')}))
            active = None
    if pending is not None or active is not None or not rows or not requests:
        raise ValueError('Incomplete CPU diagnostic log')
    return dict(prefix=prefix, metadata=meta, rows=rows, requests=requests,
        artifacts=[identity(prefix + suffix) for suffix in ('.meta.json', '.err.log', '.log', '.memory.json')],
        performance_comparable=False, limitations=[
            'PageFaultCount aggregates soft and hard faults across the entire process.',
            'Process CPU times include all process threads and have coarse accounting resolution.',
            'Counters bracket CPU compute; concurrent process work can contribute.',
            'Phase timers and process sampling perturb wall time; this run is diagnostic only.',
            'Phase intervals are sequential CPU walls; compute residual includes arena, LUT and dispatch costs.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefixes', nargs='+')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = dict(runs=[read(p) for p in args.prefixes])
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    for run in result['runs']:
        for request in run['requests']:
            print(run['prefix'], request['position'], request['decode_totals'])
