# R1 decode attention GQA K/V reuse (pair2) - implementation, 2026-10-15

Role K5 implementation window. Base: `E:\RiderProjects\Aila` HEAD `fe1bf9e`,
pair retained25. Zero GPU, zero build, zero commit, zero git write in this
window: every performance statement below is a quoted prior measurement or an
explicit `[estimate]`.

## 1. Deliverable

| Item | Value |
|---|---|
| Env switch | `AILA_ATTN_KVREUSE` |
| Accepted values | `pair2` = G=2 head-pair K/V sharing; unset / `0` / anything else = unchanged default path |
| Read policy | read once per process (`aila::env::read_string` inside a function-local static) |
| Scope of effect | `attention_decode` D256 exact partial-merge path only, bf16 KV cache only, single-row calls only |
| Variant kernel | `attention_decode_exact_head256_pair2_impl<kKvFmtBf16>` (`src/ops/AttentionOps.cpp`) |
| Dispatcher change | `attention_decode_exact_head256_partial_merge` (`src/ops/AttentionOps.cpp`) |
| New bench arm | `--decode-attn-reuse-bench [--json]` (`tests/ops/Qwen4ExpKernelTests.cpp`) |
| Default-path source change | none: both file diffs are pure insertions, so with the env unset the compiled default path is unchanged |
| D128 twin | not implemented this window (see section 6) |

Callers of the affected wrapper: exactly one (`attention_decode`, single-row
decode dispatch). Prefill exact-256, vision, and the multi-row decode kernels
are different functions and are untouched.

## 2. Design (sketch B, verified against the code)

Phase 1 of the D256 exact partial-merge kernel is re-issued with grid
`(num_heads/2) * num_tiles` work-groups of 128 threads. Work-group
`head_pair * num_tiles + tile_idx` owns token tile `tile_idx` for q-heads
`head_a = 2*head_pair` and `head_b = head_a + 1`, which share kv-head
`head_a / heads_per_kv`.

Per lane: one K vec8 load and one `float` conversion feed both heads
(`score_a`/`score_b`); one V element load feeds both heads
(`acc_a`/`acc_b`). Token-index mapping (`active_blocks` / `active_indices` /
sink+tail) is computed once per lane and is head-independent because both heads
resolve to the same kv-head.

Bit-exactness argument (the variant is required to be bitwise identical to the
default path):

1. Per-lane token mapping is unchanged: lane `lid` still owns token
   `tile_start + lid`, so at the moment a head's group reduction runs every
   lane holds exactly the value it holds for that head in the G=1 kernel.
2. Expression and accumulation order per head is unchanged and is written out
   verbatim: `score += q_cache[...] * static_cast<float>(k_vec[j])` over
   `d` ascending / `j` 0..7, then `score *= scale`; V is token-major
   `acc += logits_g[i] * v`. Sharing a load cannot change a value: the load is
   a pure function of immutable KV bytes, and both heads consume the identical
   converted `float`.
3. Both `reduce_over_group(item.get_group(), ...)` calls per head run over the
   same 128 work items with the same lane-to-value map (the SG16 tree shape
   depends only on lane ids), so `tile_max` and `tile_sum` are bitwise equal.
   All four calls per work-group are executed by every lane in the same order
   (compile-time G=2, no early exit), which is required for a group reduce.
4. The max/exp/`logits` SLM round trip is kept verbatim, including the
   "masked lanes store 0" branch, so `sycl::native::exp(logits[i] - stats[0])`
   sees identical inputs.
5. `partials` layout `[head][tile][stride=272, acc_offset=16]` and phase 2 (the
   merge, the bf16 rounding, the gate fusion) are copied verbatim, so the
   merged `bf16` output is byte-identical.

Disk/DRAM effect: per token the K row is read once instead of twice
(12 q-heads per kv-head -> 6 K reads), and each V element is read once instead
of twice. FMA count, SLM traffic per head, reduction count per head and the
partials traffic are unchanged; the number of work-groups halves, so the device
sees the same total instruction stream with about half the KV read traffic.

## 3. Corrections to the sketch (recorded as required)

The sketch's structure (grid, `acc[G]`, SLM sizes, "partials layout and merge
phase untouched") survived review. Five corrections:

1. Missing barriers. The sketch's per-head loop shows
   `logits[g][lid] = ...; reduce(max); exp; reduce(sum)` with no
   `item.barrier(local_space)` between the max and the exp. The G=1 kernel has
   two barriers per head (after `stats[0] = tile_max`, after `partial[0..1]`),
   and the exp reads `stats[0]` from SLM, so both must be kept per head. The
   implementation keeps exactly two barriers per head, four per work-group.
2. Uniform reduce participation. `reduce_over_group` with a *work-group* group
   requires every work item to reach the call; the per-head sections therefore
   cannot be inside a divergent branch or a data-dependent loop. G is a
   compile-time 2 here.
3. Pairing legality. `head0 = head_group * G` only keeps both heads on one
   kv-head when `heads_per_kv` is even; the sketch does not state this. The
   dispatcher guards on `num_heads % 2 == 0 && (num_heads / num_kv_heads) % 2 == 0`
   (24/2 -> 12, so all 12 pairs are legal) and falls back otherwise.
4. `acc[G]` register count. For D256 the V loop strides `d` by `tile_wg = 128`
   over `head_dim = 256`, so each lane holds two accumulators per head; G=2 is
   4 live floats (the sketch's "4 with the lo/hi columns pattern" note is
   correct, but the plain reading of `acc[G]` underestimates it by 2).
5. The K load form differs between the two D256 kernels and the sketch does not
   say which one is being shared: the single-row kernel uses a `sycl::vec<bf16,8>`
   reinterpret load, while the multi-row twin (`attention_decode_rows_exact_merge_impl`)
   uses scalar `KVCacheAccessor::load` calls. The pair2 variant shares the vec8
   load (the fast form); both forms convert elementwise through
   `static_cast<float>(bf16)`, so the shared value is identical either way.

## 4. Budgets

| Resource | Default G=1 | pair2 G=2 | Note |
|---|---|---|---|
| SLM `q_cache` | 256 f32 = 1024 B | 512 f32 = 2048 B | `q_cache[head_dim + d]` holds head b |
| SLM `logits` | 128 f32 = 512 B | 256 f32 = 1024 B | head b at offset `tile_t` |
| SLM `stats` | 2 f32 = 8 B | 2 f32 = 8 B | reused serially per head |
| SLM total | 1544 B | 3080 B | vs 128 KB per Xe-core: no occupancy class change |
| Live private floats | `score` xor `acc` (1) | `score_a/score_b` xor `acc_a/acc_b` (2) + live vec8 | +1..2 GRF words per lane |
| Group reductions per WG | 2 | 4 (2 per head, serialized) | total reductions per token unchanged |
| WG barriers | 3 (1 q staging + 2) | 5 (1 q staging + 4) | per head cost unchanged |
| Work-groups | `num_heads * num_tiles` | `num_heads/2 * num_tiles` | 4096: 768 -> 384, 8192: 1536 -> 768, 14336: 2688 -> 1344 |
| K bytes read per token | `num_heads/num_kv_heads` x row | half | the intended win |

Occupancy `[estimate]`: A770 has 4096 hardware threads and a 128-thread WG with
SG16 costs 8 hardware threads, so at most 512 such WGs are resident at once.
At 4096 tokens the pair2 grid is 384 WGs = 0.75 of one full device wave
(default: 768 = 1.5 waves); at 8192 it is 768 WGs = 1.5 waves. The 4096 shape
is therefore the shape where the variant has the least occupancy headroom, and
each WG also has twice the serialized per-head tail (2 reductions + 2 exp
phases before the V loop). A 4096 regression is a plausible outcome; that is
what the arm measures.

Prior baseline (window A, real `--decode-attn-bench` A1/A2 runs, D256 bf16
Hq24/Hkv2, median us/row): 4096 -> 352-380, 8192 -> 567-571, 14336 -> 838-846
(T1) and 869-893 (T2). `[estimate]` pair2 should remove roughly the KV DRAM
traffic share of a row, i.e. tens of percent at 4096/8192 if the phase-1 K and
V reads dominate; the floors to clear are only 3% (adoption) and 15%/25%
(e2e spend).

## 5. Risks and interactions

1. Occupancy / wave count at 4096 (section 4) - the main risk of a wash or a
   small regression on the smallest gate shape.
2. Serialized reductions: two heads' softmax sections run back-to-back in one
   WG. If the kernel is latency-bound rather than KV-bandwidth-bound, this
   cancels the load win; the arm's 8192-vs-4096 contrast separates the two.
3. Cancel path / USM safety: unchanged by construction. The variant issues the
   same two submissions (phase 1, phase 2) on the same in-order queue with the
   same buffers; no new USM allocation, no rebinding, no change to the
   drain-before-free contract.
4. rows >= 2 (T2 and the e2e batched shape): NOT covered. With `row_count > 1`
   the D256 dispatcher takes `attention_decode_rows_exact_merge_impl`, a
   different phase-1 kernel that re-reads K/V per head; pair2 does not touch it,
   so a B=2 e2e run should show ~0% change and must not be used to accept or
   reject the variant. Extending the same pairing to that kernel is the obvious
   follow-up (its grid is flattened over (sequence, head, tile), so the prefix
   scan has to be re-derived for pairs).
5. INT8KV (`s8`, with per-token scales) and FP8 (`f8_e4m3`): the env is ignored
   (once-logged message `kvreuse=pair2 ignored`), so the int8 DPAS and int8
   vec-read specializations are untouched. The pair2 kernel's accessor branch
   would be correct for them but deliberately does not carry the int8 vec-read
   specialization; re-verify before enabling.
6. QSA routing (`active_indices` / `active_blocks`) and sink/window variants go
   through the same pair2 body and keep the same index mapping; they are not
   exercised by this arm (the arm uses plain contiguous caches).
7. `AILA_ATTN_KVREUSE` values other than the exact string `pair2` silently keep
   the default path. Case and spelling matter; no aliases are accepted.
8. Grid pairing is only legal for even `heads_per_kv` (guarded). Shapes with
   `heads_per_kv = 1` (MQA) or odd head counts keep the default path.

## 6. Not done / TODO

- D128 twin (`attention_decode_exact_head128_partial_merge_impl`, the shorter
  dual of the same design): not implemented this window. It is the path Qwen3
  dense (head_dim 128, len >= 128, `AILA_ATTN_JM` irrelevant because the
  exact-merge branch is checked first) would use.
- The same pairing inside `attention_decode_rows_exact_merge_impl` (rows > 1).
- INT8/FP8 enablement of pair2 and the int8 vec-read specialization for it.
- `pair4` / `head12` variants named in the probe doc: not implemented; the
  dispatcher only recognizes `pair2`.

## 7. Coordinator commands and pass/fail rules

Build once (outside this window), then run the arm twice from the repo root.
The first run writes the bit-exactness ledger, the second compares against it.
Delete the ledger first so a stale file cannot be mistaken for a fresh one.

```powershell
pwsh -NoProfile -File build.ps1 -BuildDir build-refactor
Remove-Item -ErrorAction SilentlyContinue tmp\q4e-sprint-r1-20261014\decode-attn\reuse-ref.txt

$env:AILA_ATTN_KVREUSE="0"
$env:AILA_ATTN_KVREUSE_REF="tmp/q4e-sprint-r1-20261014/decode-attn/reuse-ref.txt"
build-refactor\AilaQwen4ExpKernelTests.exe --decode-attn-reuse-bench --json | Tee-Object tmp\q4e-sprint-r1-20261014\decode-attn\reuse-base.log

$env:AILA_ATTN_KVREUSE="pair2"
build-refactor\AilaQwen4ExpKernelTests.exe --decode-attn-reuse-bench --json | Tee-Object tmp\q4e-sprint-r1-20261014\decode-attn\reuse-pair2.log

Remove-Item Env:AILA_ATTN_KVREUSE
Remove-Item Env:AILA_ATTN_KVREUSE_REF
```

Parse: one `[DECODE ATTN REUSE] {json}` line per (length, rows); fields
`variant`, `reps`, `warm_ms`, `median_ms_total`, `median_us_per_row`,
`min_us_per_row`, `max_us_per_row`, `finite`, `rows_duplicate_eq`, `max_abs`,
`row_hash`, `bitwise_eq`, `ref_mode`, `pair2_scope`. Also read the one-time
`[ATTN] kvreuse=pair2 ...` log line to confirm the variant was active.

Pass (adopt pair2, then spend e2e time):

- `bitwise_eq = 1` for every rows=1 shape in the pair2 run (rows=2 lines are
  `pair2_scope = 0` controls; a rows=2 hash difference versus the ledger is a
  red flag about the environment, not a pair2 failure, because pair2 cannot
  reach that kernel).
- `max_abs <= 2e-4` and `rows_duplicate_eq = 1` everywhere (the arm already
  throws otherwise).
- rows=1 `median_us_per_row` improves >= 3% at BOTH 4096 and 8192.
- rows=1 `median_us_per_row` at 333 and 1834 regresses <= 2%. 333 is the
  no-change control (below the 1024 exact-merge threshold it stays on the
  baseline kernel, so it should be identical); 1834 is the first shape pair2
  can move.
- A/A sanity: rerun the env=0 command and keep the two runs' 4096/8192 medians
  within +-1% before trusting any A/B delta.

Reject / stop: any mismatch, any regression beyond 2% at 333 or 1834, or a
delta under 3% at either gate shape. Spend e2e (`R4`, `AILA_ATTN_KVREUSE=0` vs
`pair2`, plus an A/A pair) only if the kernel gate is >= 15% at 4096 and
>= 25% at 8192. The e2e arm must use the single-row (T1) decode shape: with
`row_count > 1` no code path changes (risk 4).

Rollback: unset `AILA_ATTN_KVREUSE` (or set it to `0`). No default-path code
changed, so no revert is needed to keep the old behavior.

## 8. Provenance

- Kernel and bench written from the G=1 D256 exact partial-merge kernel at
  `src/ops/AttentionOps.cpp:432-736` (pre-window line numbers) and the existing
  `--decode-attn-bench` arm at `tests/ops/Qwen4ExpKernelTests.cpp:5122`.
- Variant sketch: `tmp/q4e-sprint-r1-20261014/decode-attn/variant-sketch.md`
  (section B).
- Gate, rejection and e2e-spend rules: `tmp/q4e-sprint-r1-20261014/decode-attn/probe-requests.md` (R3/R4).
- Baseline timings: prior window A `--decode-attn-bench` runs quoted in the K5
  task brief; not remeasured here.
- Hardware occupancy arithmetic: A770-Kernel-Optimization skill
  `references/hardware/hardware.md` (4096 hardware threads, 8 hardware threads
  per vector engine, 128 KB SLM per Xe-core).

## 9. Coordinator verification (2026-10-15 ~19:48-20:02, window B)

Three bench runs on uild-refactor/AilaQwen4ExpKernelTests.exe --decode-attn-reuse-bench --json
(GPU exclusive; base=ref-write, pair2=ref-compare, base2=A/A rerun):

| len  | rows=1 base1/base2 (us) | A/A % | rows=1 pair2 (us) | pair2 delta % | bitwise_eq |
|------|--------------------------|-------|--------------------|----------------|------------|
| 333  | 366.1 / 368.2            | +0.57 | 365.2              | -0.25          | 1          |
| 1834 | 269.1 / 266.6            | -0.93 | 290.6              | **+7.99**      | 1          |
| 4096 | 377.6 / 378.8            | +0.32 | 317.1              | **-16.02**     | 1          |
| 8192 | 570.3 / 570.8            | +0.09 | 427.1              | **-25.11**     | 1          |
| 14336| 837.6 / 839.7            | +0.25 | 684.9              | **-18.23**     | 1          |

- All rows=1 shapes are bitwise against the ref file; rows=2 lines are pair2_scope=0
  controls (unchanged path, bitwise too).
- Gate call: bitwise PASS; >=3% at 4096+8192 PASS; e2e-spend thresholds (>=15%/>=25%)
  PASS; A/A within 1% PASS; **pre-registered reject fired: 1834 regression +7.99%**
  (reproduced against base2, so it is real, not noise).
- e2e projection stays <= ~3% (14K production is QSA-routed; attention share ~6.5%),
  below the 5% default gate.
- Verdict: NO-GO for blanket adoption tonight. A length-gated variant (min length
  >= 2048) would clear all rules, but the e2e gain would still be under gate.
  Keep the variant opt-in/uncommitted; no default change. Receipt:
  	mp/q4e-sprint-r1-20261014/receipt-k5-pair2.json.
