"""Describe observed CPU/GPU supply before online hotset refills in cache traces.

This is placement history, not a counterfactual speed prediction. GPU evidence
means a routed demand acquire; it does not necessarily mean a new H2D copy.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from q4e_cache_replay import read_trace


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inspect(path):
    rows = read_trace(path)
    experts = rows[0]['experts']
    slot_bytes = rows[0]['slot_bytes']
    forwards, prompts, batches = {}, [], []
    active = None
    last_completed = None
    loading = False
    pending = {}
    acquire = None
    current_prompt = None
    publish_before_next = []
    ignored_acquires = Counter()
    retire_count = publish_count = 0

    for line, row in enumerate(rows, 1):
        kind = row['type']
        if kind == 'epoch_load_begin':
            require(not loading and active is None, 'Nested restoration or active forward')
            loading = True
        elif kind == 'epoch_load_end':
            require(loading and acquire is None, 'Unfinished restoration')
            loading = False
        elif kind == 'forward_begin':
            require(active is None and not loading, 'Overlapping forward/restoration')
            fid = row['forward']
            require(fid not in forwards, 'Duplicate forward identity')
            if row['position'] == 0:
                current_prompt = dict(prompt=len(prompts), first_forward=fid,
                                      handoffs=[], last_refill_forward=None)
                prompts.append(current_prompt)
            require(current_prompt is not None, 'Trace starts inside an unidentified prompt')
            active = dict(forward=fid, prompt=current_prompt['prompt'],
                          position=row['position'], tokens=row['tokens'],
                          begin_ns=row['ns'], end_ns=None, cpu=defaultdict(list),
                          gpu=defaultdict(list), handoff_seen=bool(current_prompt['handoffs']))
            forwards[fid] = active
            for entry in publish_before_next:
                entry['publish_before_forward'] = fid
            publish_before_next.clear()
        elif kind == 'forward_end':
            require(active is not None and active['forward'] == row['forward'], 'Unmatched forward end')
            require(acquire is None, 'Acquire overlaps forward end')
            active['end_ns'] = row['ns']
            last_completed, active = active, None
        elif kind == 'prompt_handoff':
            require(active is None and not loading, 'Handoff inside forward/restoration')
            require(current_prompt is not None, 'Handoff without prompt')
            current_prompt['handoffs'].append(dict(line=line, ns=row['ns'],
                after_forward=last_completed['forward'] if last_completed else None))
        elif kind == 'cold_cpu':
            require(active is not None and not loading, 'CPU route outside forward')
            require(len(row['ids']) == len(row['ranks']), 'CPU route rank mismatch')
            for expert, rank in zip(row['ids'], row['ranks']):
                key = row['layer'] * experts + expert
                active['cpu'][key].append(dict(forward=active['forward'], rank=rank,
                                              line=line, ns=row['ns']))
        elif kind == 'acquire_begin':
            require(acquire is None, 'Overlapping acquire')
            acquire = (row, line, active)
        elif kind == 'acquire_end':
            require(acquire is not None and acquire[0]['acquire'] == row['acquire'], 'Unmatched acquire end')
            begin, begin_line, owner = acquire
            if owner is None or loading:
                ignored_acquires['restoration' if loading else 'outside_forward'] += 1
            else:
                require(owner is active and begin['forward'] == active['forward'], 'Acquire forward mismatch')
                require(len(begin['ids']) == len(begin['initial_slots']), 'Acquire identity mismatch')
                for expert, initial_slot in zip(begin['ids'], begin['initial_slots']):
                    key = begin['layer'] * experts + expert
                    owner['gpu'][key].append(dict(forward=owner['forward'],
                        acquire=begin['acquire'], initial_slot=initial_slot,
                        line=begin_line, begin_ns=begin['ns'], end_ns=row['ns']))
            acquire = None
        elif kind == 'epoch_retire':
            require(active is None and not loading and last_completed is not None,
                    'Online retirement inside forward/restoration or without predecessor')
            require(last_completed['tokens'] == 1 and last_completed['position'] > 0,
                    'Online retirement attributed to prefill')
            fid = last_completed['forward']
            prompt = prompts[last_completed['prompt']]
            require(prompt is current_prompt, 'Retirement crosses prompt boundary')
            if not batches or batches[-1]['retire_after_forward'] != fid:
                previous = prompt['last_refill_forward']
                require(prompt['handoffs'],
                        'Cannot exclude one-row prefill tails without a prompt_handoff marker')
                candidates = [f for f in forwards.values() if f['prompt'] == prompt['prompt']
                              and f['tokens'] == 1 and f['position'] > 0
                              and f['forward'] <= fid
                              and (previous is None or f['forward'] > previous)
                              and f['handoff_seen']]
                # A recorded prompt handoff distinguishes one-row prefill tails
                # from decode. Do not silently classify phase from row count.
                require(candidates, 'Empty decode observation window')
                batch = dict(batch=len(batches), prompt=prompt['prompt'],
                    retire_after_forward=fid, position=last_completed['position'],
                    previous_refill_forward=previous,
                    window_forwards=[f['forward'] for f in candidates],
                    phase_evidence='prompt_handoff',
                    entries=[])
                batches.append(batch)
                prompt['last_refill_forward'] = fid
            batch = batches[-1]
            key, slot = row['key'], row['slot']
            require(slot not in pending, 'Slot retired again before publication')
            require(key // experts == row['old_key'] // experts, 'Cross-layer retirement')
            cpu, gpu = [], []
            for window_fid in batch['window_forwards']:
                cpu.extend(forwards[window_fid]['cpu'].get(key, []))
                gpu.extend(forwards[window_fid]['gpu'].get(key, []))
            category = ('cpu_gpu' if gpu else 'cpu_only') if cpu else ('gpu_only' if gpu else 'unobserved')
            entry = dict(key=key, layer=key // experts, expert=key % experts,
                outgoing_key=row['old_key'], slot=slot, category=category,
                refill_bytes=slot_bytes, cpu_observations=cpu, gpu_demand_acquires=gpu,
                retire_line=line, retire_ns=row['ns'], publish_line=None,
                publish_ns=None, publish_before_forward=None)
            batch['entries'].append(entry)
            pending[slot] = entry
            retire_count += 1
        elif kind == 'epoch_publish':
            require(active is None and not loading, 'Publication inside forward/restoration')
            entry = pending.pop(row['slot'], None)
            require(entry is not None and entry['key'] == row['key'], 'Publication does not match retirement')
            entry.update(publish_line=line, publish_ns=row['ns'])
            publish_before_next.append(entry)
            publish_count += 1

    require(active is None and acquire is None and not loading, 'Truncated active operation')
    categories = ('cpu_only', 'cpu_gpu', 'gpu_only', 'unobserved')
    totals = Counter({category: 0 for category in categories})
    for batch in batches:
        counts = Counter({category: 0 for category in categories})
        counts.update(entry['category'] for entry in batch['entries'])
        batch['category_counts'] = dict(counts)
        batch['refill_bytes'] = len(batch['entries']) * slot_bytes
        totals.update(counts)
    return dict(schema=1, path=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        slot_bytes=slot_bytes, prompts=prompts, forward_metadata=[
            {k: f[k] for k in ('forward', 'prompt', 'position', 'tokens', 'begin_ns', 'end_ns', 'handoff_seen')}
            for f in forwards.values()],
        online_retirements=retire_count, matched_publications=publish_count,
        unpublished_entries=len(pending), category_counts=dict(totals),
        previously_cpu_entries=totals['cpu_only'] + totals['cpu_gpu'],
        refill_bytes=retire_count * slot_bytes,
        ignored_acquire_batches=dict(ignored_acquires), batches=batches,
        semantics=[
            'Each window begins after the previous observed online retirement batch in the same prompt and includes the forward immediately preceding this retirement.',
            'The first window begins at the first decode forward after the recorded prompt handoff. Traces without this marker are rejected because row count alone cannot exclude one-row prefill tails.',
            'Zero-swap rebalance calls emit no retirement records; these windows describe actual nonempty refill batches, not an assumed fixed interval.',
            'Retirement records create refill entries; matching publications close those entries and are not counted as separate transfers.',
            'Startup/profile restoration acquires and all prefill forwards are excluded. Ready adoptions are not online H2D refills.',
            'cpu_only/cpu_gpu/gpu_only/unobserved describe the incoming key in the backward observation window. GPU evidence is a successful routed demand acquire, which may hit cache rather than copy.',
            'Unobserved does not mean globally unused: prompt-seeded/decayed earlier history can select keys absent from this window.',
            'CPU eligibility and future residency can change. Placement history alone does not prove wasted H2D, quantify supply cost, or predict net speedup.',
            'Publication without a following forward is retained with publish_before_forward=null, including end-of-trace draining. Unpublished retirement entries are retained explicitly.'
        ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = inspect(args.trace)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('online_retirements', 'matched_publications',
        'unpublished_entries', 'category_counts', 'previously_cpu_entries', 'refill_bytes')}, indent=2))
