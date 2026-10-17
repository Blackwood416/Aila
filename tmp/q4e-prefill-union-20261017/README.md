# Qwen4Exp prefill-union Probe B

Date: 2026-10-17. CPU-only replay tool for prompt-scoped cold-key promotion.
It does not import Aila, start a worker/server, or touch the GPU.

## Current input status

The repository currently has a real 14K-run aggregate hotset export:

- `tmp/q4e-sprint-r1-20261014/prefill/ttft14k-export.profile`
  - identity-bound `AILA_Q4E_HOTSET 1`
  - 48 layers x 512 experts
  - `decode`, `prefill`, and `unique` counts per key
  - produced by the 14K P-1 sizing arm

It is run-level data: the prefill count can include the fixture warm-up, so it
is not a four-block route order and must not be treated as one.

The decode baseline used for resident-slot loss is:

- `perf/qwen4exp/profiles/qwen38-nf4-agent6k-v1.profile`
  - the current agent6k decode ranking
  - the same NF4 expert-order identity

There is **no existing 14K trace with four prefill blocks and complete
per-layer route IDs** in the repository. Aggregate profile counts cannot
reconstruct block boundaries or route order. `prompt_union_replay.py` therefore
refuses to invent four-block results: profile-only mode returns
`blocked_missing_per_block_route_trace`, empty block/layer arrays, and an
explicit TODO. Its decode-loss numbers in that mode are labelled worst-case
bounds, not replay measurements.

The profile-only readiness receipt currently reports:

- 23,716 prefill-nonzero keys;
- per-layer union **466 / 495 / 512** (min / median / max);
- union bytes **60,105,830,400 B = 55.978 GiB**;
- 2,792 slots: 512 borrowed overlay slots, worst-case decode mass at risk
  **8.94694%**, post-rollback loss **0**;
- 2,400 slots: 512 borrowed + 392 outside-budget slots, worst-case decode mass
  at risk **16.913487%**, post-rollback loss **6.7243%**.

## Inputs

### Aggregate hotset profile

The parser accepts the existing text format:

```text
AILA_Q4E_HOTSET 1
<identity>
<layers> <experts> <count>
<key> <decode> <prefill> <unique>
...
```

It is used only for validation and aggregate per-layer union statistics.

### Exact route trace

The optional exact input is JSONL with one object per line and the schema in
[`schema/route-trace-v1.schema.json`](schema/route-trace-v1.schema.json).
Required record flow:

```text
config
request_begin
forward_begin             (prefill, position 0, tokens 4096)
route x 48                (one complete ids array per layer)
forward_begin             (prefill, position 4096, tokens 4096)
route x 48
forward_begin             (prefill, position 8192, tokens 4096)
route x 48
forward_begin             (prefill, position 12288, tokens 1656)
route x 48
request_end
```

For the current 14K fixture the prompt size is 13944 tokens, so the four
prefill windows are `4096 + 4096 + 4096 + 1656`. A short synthetic prompt is
not a substitute: QSA/expert behavior differs below its enablement threshold.

Each `route.ids` array must have exactly `rows * top_k` IDs in row-major
`[row][rank]` order. The trace must contain every layer exactly once per
forward. The tool rejects missing/duplicate layers, duplicate positions,
out-of-range IDs, wrong array lengths, and non-contiguous prefill positions.

## Simulation semantics

- Total resident budget is `--slots` (default `2792,2400`).
- The prompt overlay uses `--promotion-slots` (default 512, matching the
  `AILA_Q4E_HOTSET_PROMPT_SWAPS=512` setting in the xpb config).
- Permanent decode base = top `budget - promotion_slots` keys from the decode
  profile.
- The overlay starts with the next `promotion_slots` decode keys, so block 0
  is a no-promotion control.
- At the end of each block, cold keys are observed. Keys occurring at least
  `--min-occurrences` times are admitted in descending frequency order into
  the overlay. The overlay uses LRU order; borrowed decode entries are evicted
  before promoted entries when possible.
- The overlay is discarded after the request and the decode hotset is restored
  to the top `budget` keys. For the 2400 budget, the post-rollback comparison
  against the current 2792-slot hotset therefore still has 392 fewer keys.
- `staged_slots` is the unique cold-key count per layer/window, matching the
  production staging granularity. `staged_bytes = staged_slots * 2,534,400`.

Decode loss is a **mass proxy** from the decode profile's `decode` counts, not
a throughput or quality prediction.

## Commands

Profile-only readiness check (current state; no GPU):

```powershell
python tmp/q4e-prefill-union-20261017/prompt_union_replay.py `
  --route-profile tmp/q4e-sprint-r1-20261014/prefill/ttft14k-export.profile `
  --decode-profile perf/qwen4exp/profiles/qwen38-nf4-agent6k-v1.profile `
  --slots 2792,2400 `
  --promotion-slots 512 `
  --json-out tmp/q4e-prefill-union-20261017/out/current-profile-only.json `
  --csv-out tmp/q4e-prefill-union-20261017/out/current-profile-only.csv
```

Exact replay after a four-block route trace exists:

```powershell
python tmp/q4e-prefill-union-20261017/prompt_union_replay.py `
  --route-profile tmp/q4e-sprint-r1-20261014/prefill/ttft14k-export.profile `
  --decode-profile perf/qwen4exp/profiles/qwen38-nf4-agent6k-v1.profile `
  --trace tmp/q4e-prefill-union-20261017/out/ttft14k-routes-v1.jsonl `
  --slots 2792,2400 `
  --promotion-slots 512 `
  --json-out tmp/q4e-prefill-union-20261017/out/replay.json `
  --csv-out tmp/q4e-prefill-union-20261017/out/replay.csv
```

Unit tests use only synthetic small data:

```powershell
python -m unittest discover -s tmp/q4e-prefill-union-20261017/tests -v
```

The synthetic tests verify parser and promotion mechanics; they are not
production receipts.

## Output

The JSON report contains:

- source paths and SHA-256 for both profiles and the optional trace;
- aggregate per-layer union distribution and union bytes;
- one entry for each slot budget;
- four block summaries with staged slots/bytes and delta;
- per-layer baseline/promoted staged counts and saved percentages;
- decode hotset worst-case/interleaved mass and post-rollback loss;
- `limitations` and, when exact routes are absent, `todo`.

The CSV is a flat layer/block table for spreadsheet inspection. In
profile-only mode it contains a header and one explicit `profile_only` row per
budget, so it cannot be mistaken for a replay result.

## TODO for the exact trace

1. Capture a 13944-token prompt with the canonical 4096 prefill windows.
2. Emit the `route-trace-v1` records without changing the model input, route
   top-k, or staging order.
3. Preserve all 48 layers x four forwards per request; do not downsample
   layers or rows.
4. Record the source binary/commit and config identity in the trace config or
   a sidecar note.
5. Rerun the exact command above. Until then, block/layer promotion fields must
   remain `null`/empty rather than being inferred from aggregate counts.
