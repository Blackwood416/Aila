# E5: decode kernel roofline audit (device side, per token)

Date: 2026-10-15. Repo: `E:\RiderProjects\Aila`, HEAD `df1f45a`. Role: E5
(charter sec 4 role K, second pass; the first pass is
`docs/plans/qwen38-r1-kernel-2026-10-14.md`, the family pass is
`docs/plans/qwen38-r1-decode-misc-2026-10-15.md`).

Scope: CPU/source only. Zero GPU, zero build, zero `src/` edits, zero commit.
This document is the only file written. All numbers carry a source; every
derived number is marked `[est]` with its arithmetic.

---

## 0. Method, constants, and the one caveat that invalidates naive sums

Bandwidth references (A770 / DG2, oneAPI 2026.1):

| Constant | Value | Source |
|---|---:|---|
| DRAM measured contiguous read (256 MiB, 256 B msgs) | 291.11 GB/s | A770 skill `references/hardware/dram-contiguous-roofline.md:21-23` (quoted in `qwen38-r1-kernel-2026-10-14.md:20`) |
| DRAM spec (brief) / (skill) | 512 / 560 GB/s | E5 brief; skill `:36-39` |
| L2 reuse / L2 size | 854.6 GB/s / 16 MB | skill `:25`, `hardware.md:33` |
| SLM read | 4115.8 GB/s | skill `:26` |
| Xe cores | 512 | skill `hardware.md` |

Measured device spans come from ONE arm: `c2-round2-b2-worker.log`
(`[Q4E AdaptiveWork] count=1`, n=128, ctx ~6016, A/A pair b1/b2) parsed with
`tmp/q4e-sprint-r1-20261014/dec_instr_dev_parse.py`. That arm runs
`AILA_Q4E_DECODE_STAGE_EVENTS=1` (profiling queue + 530 barrier commands, and
~530 marker submissions per token). D1 measured the observer cost at
+11-13 ms/token (`qwen38-r1-decode-boundary-2026-10-15.md` sec 1.4:
74.49/72.90 instrumented vs 60.21/61.67 clean), so:

- `clean = span x 0.82` is an `[est]` mapping to the clean regime, not a
  measurement. Rank by mechanism, never by raw instrumented ms.
- The A/A spread of the two attribution arms is 73.499 vs 71.036 ms
  (-3.4%), i.e. these sensors resolve ~3% of token wall, nothing finer.
- The clean decode receipt at HEAD is `e10-bundle` ForwardWall
  54.772 / 55.461 ms (n=128 each; `e10-bundle{,-b2}-worker.log`), the best
  measured bundle (HOTSET a6k + INT8 KV + MOE_FRONT_FUSED).

The `inter_dev` span in that arm (8.794 ms) is NOT device time: D1 proved it
is the PLE worker gather window (`corr(inter_dev, ple_host)=0.998`,
`qwen38-r1-decode-boundary-2026-10-15.md` sec 0.1). It was fixed at HEAD by
`AILA_Q4E_PLE_PREFETCH_MIN=1` default (`b92bd31`+`681a06e`: decode ple_host
6.55 -> 0.26 ms, `qwen38-r1-decode-20-2026-10-15.md` sec 1.1/1.2). Treat
`inter_dev` as ~0 at HEAD and drop it from the device budget.

---

## 1. Decode device-side cost list (per token, T1 rows=1, ctx ~6016)

Raw spans are c2-round2-b2 medians (n=128). "clean" = x0.82 `[est]`.
"device busy" is the bench-measured kernel time where a bench exists.

| # | span | raw ms | clean ms [est] | what it covers | device-busy ms | source of the split |
|---:|---|---:|---:|---|---:|---|
| 1 | `experts_dev` | 26.855 | 22.02 | hot gu+dn kernels + `JoinResidentCold` + merge | 3.6-5.1 | `qwen38-r1-decode-moe-2026-10-15.md` sec 1.3 (K1 attainment scaled to E_g=3.9) |
| 2 | `inter_dev` | 8.794 | ~0.3 | PLE worker gather (host) | 0 | D1 doc sec 0.1-0.3; fixed by PLE prefetch default |
| 3 | `hc_dev` | 10.079 | 8.26 | 97 HC calls x (down + up [+ inject]) = 194 kernels | 6.00 | `decode-misc/misc-A1.log` `hc_token_prod span_ms=6.0023` (194 kernels, 213 GB/s) |
| 4 | `linear_dev` | 8.049 | 6.60 | 36 GDN layers: all_proj NF4 + delta rule + o_proj NF4 | 2.42 (GDN kernel only) | `misc-A1.log` `gdn_token36 span_ms=2.416`; projections not in that arm |
| 5 | `full_dev` | 5.440 | 4.46 | 12 full layers: qkv NF4 + indexer + QSA select + selected attn + o_proj NF4 | n/a (bench is dense-only) | `--decode-attn-reuse-bench` dense rows=1: 0.378 / 0.570 / 0.838 ms at 4096/8192/14336 |
| 6 | `shared_dev` | 3.311 | 2.71 | 48 layers x (fused gate/up/SiLU + down + add) = 144 kernels | n/a (bench arm FAILED) | `misc-A1.log` last line: "shared fused gate/up/swiglu is unsupported for the bench shape" |
| 7 | `head_dev` | 3.299 | 2.71 | final mixer HC + BF16 LM head (1.271 GB) | 2.71 | `qwen38-r1-decode-misc-2026-10-15.md` sec 5 (infers ~470 GB/s) |
| 8 | `router_dev` | 2.638 | 2.16 | 48 x (MoE-front fused gate GEMV + top-k) = 96 kernels | n/a | span arithmetic only (23 us/kernel) |
| 9 | `stagein_dev` | 1.934 | 1.59 | 48 x 5 KB resident-input D2H + previous-copy wait | n/a (bubble) | D1 doc sec 1.5 |
| 10 | `resid_dev` | 0.309 | 0.25 | shared-expert add (+ HC inject) | n/a | B6 doc sec 2.2 |
| 11 | `ple_dev` | 0.261 | 0.21 | PLE block marker window | n/a | B6 doc sec 2.2 |
| 12 | `prerouter_dev` | 0.066 | 0.05 | flag tail | 0 | B6 doc sec 2.2 |
| | `misc_dev` | 0.000 | 0.000 | conservation self-check (must be 0) | 0 | C2 doc sec 3 |
| | sum | 71.04 | ~51.4 | | ~30-32 | vs e10 ForwardWall 54.8-55.5 (fits `[est]`) |

Two readings fall out of the table:

1. **Device-busy time is only ~30-32 ms of the 54.8 ms token (~57%).** The
   rest is host submit latency (`misc-A1.log` `submit_cost
   host_us_per_kernel=13.97`, and 6.4-47 us/submit observed per family),
   dependency stalls and the two D2H windows. The kernel audit therefore
   cannot be the source of a large decode win; the schedule is.
2. **`experts_dev` is a host-side item wearing a device name**: 3.6-5.1 ms of
   kernels inside a 22 ms clean span; the other ~17 ms is the in-order queue
   idling while `JoinResidentCold` waits for the CPU arena batch
   (`src/models/Qwen4ExpBnb4Backend.cpp:1446,1517`; the C1 line
   `qwen38-r1-decode-2026-10-15.md` sec 4 owns the rebalance, upper bound
   10.2 ms/token; E3 is on the per-row serial join).

---

## 2. Roofline for the three largest spans (plus the honest extras)

Byte models (explicit so they can be checked):

- experts (hot only): `hits x 2.534 MB`. hits 243.5/token -> 617 MB;
  hits 186/token -> 471 MB (both from real runs: E2 doc sec 1.2 and
  `dec-instr-b2-worker.log` sampled tokens via decode-moe doc sec 1.2).
- hc: 97 x (6.554 MB down + 6.554 MB up) + 96 x 0.082 MB inject = 1.279 GB
  (geometry: lowrank 320, group_dim 2560, groups 4, BF16; E5 predecessor
  doc sec 2, kernel source `src/ops/HyperConnectionOps.cpp:365-433,447-509`).
- linear (GDN): 36 x (21.09+2.64 all_proj NF4 + 7.86+0.98 o_proj NF4 +
  6.29 state R+W + 0.25 conv) = 1.408 GB (E5 predecessor doc sec 2;
  `config.json` 16 key/48 value heads x 128).
- full attention (12 layers, QSA-routed at ctx 6016: `num_complete_blocks
  1486 > block_topk 512` -> sparse path, `Qwen4ExpPrefill.cpp:171-246`):
  projections 28.2 MB/layer (qkv 13312x2560 NF4 19.2 MB incl. absmax,
  o_proj 6144x2560 NF4 8.84 MB, indexer-k 0.16 MB) + selected attention
  4.19 MB/layer unique (2048 keys x 2 kv heads x 256 dim x 2 B x 2) x 12
  q-heads/kv-head re-read amplification = 50.3 MB/layer -> 942 MB/token.
  Without the amplification the same 12 layers touch 50.3 MB/token.
- shared: 48 x (1.64+0.20 gu + 0.82+0.10 down + 0.005 gate) = 133 MB/token.
- head: 1.271 GB BF16 [2560,248320] (default; `AILA_Q4E_LM_HEAD_Q8` is
  opt-in, `Qwen4ExpBnb4Backend.cpp:540-571`).
- router: 48 x (512x2560 NF4 0.655+0.082 MB + shared-gate BF16 5 KB) =
  35.5 MB/token.

Roofline table. "attained %" = floor/measured at 291.11 GB/s.

| item | measured clean ms [est] | bytes/token | floor @291.11 | floor @512 | attained % | distance |
|---|---:|---:|---:|---:|---:|---:|
| experts span | 22.02 | 471-617 MB | 1.62-2.12 | 0.92-1.21 | 7-10% | **10.4-13.6x** |
| experts kernels only | 3.6-5.1 | 471-617 MB | 1.62-2.12 | - | 32-62% | 1.6-2.4x (hidden under the join) |
| hc | 8.26 (bench 6.00) | 1.279 GB | 4.39 | 2.50 | 53% (bench 73%) | 1.88x (bench 1.37x) |
| linear (GDN) | 6.60 | 1.408 GB | 4.84 | 2.75 | 73% | 1.36x |
| full attention | 4.46 | 942 MB | 3.24 | 1.84 | 73% | 1.38x (unique-byte floor 0.173 -> 26x) |
| shared | 2.71 | 133 MB | 0.46 | 0.26 | 17% | 5.9x, but 18.8 us/kernel -> command-latency bound |
| head (+mixer HC) | 2.71 | 1.271 GB | 4.37 | 2.48 | 161% | 0 -- already above the pattern (470 GB/s) |
| router | 2.16 | 35.5 MB | 0.122 | 0.069 | 5.6% | 18x, but 23 us/kernel -> command-latency bound |

Ranking by distance from an attainable bound (the brief's question):

1. `router` 18x and `shared` 5.9x top the ratio table, but both are 1-row
   GEMV chains whose per-kernel floor is launch+dispatch (~19-23 us; measured
   directly for shared/HC in `misc-A1.log`). Their byte floors are not
   attainable by any kernel variant -- the lever is command count, not bytes.
2. `experts` is the largest span and the largest distance (10.4-13.6x), but
   that distance is idle queue time; its kernels are 1.6-2.4x and already
   hidden. No kernel lever here (decode-moe doc verdict: close the line).
3. The three big streaming families -- hc 1.37x, full attention 1.38x,
   linear 1.36x -- land in a 2% band around 73% of the measured DRAM pattern
   (213-215 GB/s effective). This uniformity is the real finding: there is
   no single "furthest" kernel; there is a family-wide ~1.37x streaming gap.
   Per item absolute headroom if it hit 291.11 GB/s: hc 1.6, linear 1.8,
   full 1.2 ms = ~4.6 ms total (8% of the token), each below the 3% kernel
   gate on its own.
4. `head` is a negative result worth keeping: 1.271 GB in 2.71 ms = 470 GB/s,
   i.e. above the 291.11 pattern. The 291.11 reference is a 256 B-message
   stream, not a device ceiling; large-row GEMVs already beat it. Do not
   spend kernel time on the head (only the opt-in Q8 head changes it).

---

## 3. Candidate for the top kernel-attributable item

The item to work is **decode full attention (12 layers, `full_dev` 4.46 ms
clean)**, because it is the only one of the three ~1.37x families with a
live, bit-exact, already-bench-instrumented mechanism: cross-head K/V tile
reuse. `experts` ranks higher by raw distance but has no kernel lever (idle),
`hc` had every knob swept flat or worse (`misc-A1.log`: pl2/pl3 11.6 ms vs
6.0 prod, streams 5.94 vs 6.00, wg 64/128/256 within 1.5%), and `linear` is
the closed NF4 GEMV family (charter sec 2 + P4).

Concrete change: extend the existing head-grouping from G=2 to G=4 in the
D256 decode attention phase 1.

- Code: the gate/helper is `src/ops/AttentionOps.cpp:742` (`attn_kv_reuse_pair2`)
  and `:1061-1074` (dispatch into the phase-1 variant); the phase-1 body to
  generalize is `:432-683` (K tile load `:521-530`, V pass `:675-677`). One
  WG keeps one (kv_head, tile) and runs G q-heads through the identical tile
  body with `score[G]` / `acc[G]` accumulators; per-head expression order and
  `reduce_over_group` are unchanged, so outputs stay bit-identical (proven
  for G=2 by `tmp/q4e-sprint-r1-20261014/receipt-k5-pair2.json`,
  `bitwise_eq_rows1_all_shapes=1`).
- Expected: G=2 measured -16.0% / -25.1% / -18.2% at 4096/8192/14336 rows=1.
  Re-read count is 12/G, so G=4 should land near -30..-38% vs base
  `[est = pair2 traffic share extrapolated]`. At the routed 14K production
  shape that is an extra 0.5-1.0 ms over pair2 -> e2e -1.9..-3.8% `[est]`.
  pair2's own e2e receipt is -2.7% wall / +4.1% rate (59.644 -> 58.088 ms,
  16.175 -> 16.845 tok/s). Ceiling of the whole line (attention -> 0) is
  4.46 ms = 8.1% of the token.
- Mandatory companion: a `cached_len >= 2048` length gate. pair2's blanket
  adoption was rejected by its pre-registered rule (`1834: +7.99%`), and the
  routed production shape is always above 2052, so the gate costs nothing.
- Probe: existing arm `AilaQwen4ExpKernelTests.exe --decode-attn-reuse-bench
  --json` with `AILA_ATTN_KVREUSE=<0,pair4>` rows=1 at 4096/8192 (gate:
  >=3% both, A/A within +-1%) plus the 1834 control (reject rule), then the
  standard 14K e2e A/B/B/A via `tmp/q4e-sprint-r1-20261014/run.py` with
  `receipt-pair2-e2e.json` as the template. No new probe code needed; only
  the G=4 width has to be added behind the same env.
- Why this is not charter sec 2: the two denied attention items are ESIMD
  replacing SG8 DPAS and INT8 attention DPAS -- both change the compute
  instruction/datatype. This changes neither; it removes redundant DRAM
  traffic (one tile load serves G heads) with the per-head arithmetic order
  verbatim, which is exactly why it is bit-exact. Also note this is K3's
  documented "B - head group" direction; this audit's addition is the
  ceiling arithmetic above, which says: realistic yield +2..+4%, i.e. a
  bundle component (like PLE/hotset/MOE_FRONT), not a standalone 5% winner.

Conditional second candidate (kernel, bit-exact, one command to unblock):
the decode MoE hot kernels are 30% (gu) / 65% (dn) slower in the
`arrayform` (production call form) than in the device-descriptor form
(`decode-moe-A1.log` BENCH lines) -- worth 1.0-1.4 ms/token `[est: at
E_g=3.9, (0.1518-0.1165)+(0.1088-0.0661) ms per E=10 layer x 0.39 = 30.4
us/layer]`. It is unbanked because that run failed its own sentinel guard.

---

## 4. What this audit cannot determine, and the one command that decides it

1. **The arrayform-vs-device MoE kernel delta** (candidate 2 above):
   `decode-moe-A1.log` ends `[DECODE MOE GUARD] sentinel_ok=0` then
   `BENCH FAIL`, so the 30%/65% spread is printed but not admissible. One
   command decides: `build-refactor\AilaQwen4ExpDecodeMoeBench.exe --experts
   10,20 --rows 1` (GPU mutex first; ~1 min; read `[DECODE MOE GUARD]
   sentinel_ok` and the `gu-ref-arrayform` vs `gu-ref-nested-div` medians).
   If it passes again, this is a bit-exact ~1.0-1.4 ms/token lever.
2. **The shared-expert device time**: no bench number exists (arm failed at
   the bench shape). The 2.71 ms span / 18.8 us-per-kernel split is
   arithmetic only. Deciding command = fix the bench arm's shape guard in
   `perf/qwen4exp/q4e_decode_misc_bench.cpp` and rerun
   `AilaQwen4ExpDecodeMiscBench.exe --arms shared`. Do not schedule shared
   work before that; the family is command-bound either way.
3. **Whether `router_dev` (2.16) and `stagein_dev` (1.59) contain any device
   work** vs pure queue bubbles: the markers bracket host-driven submissions
   (MoE-front gate + top-k; one 5 KB D2H), so no existing field can split
   them. Would need new markers; both are <=2% of the token, not worth it.
4. **The dense-attention number at ctx 14336 in production**: the K3 bench
   (0.838 ms/row) is the *dense* path; at >=2052 the production decode takes
   the QSA sparse route (selected 2048 keys), where the whole
   selector+attention block is ~0.37-0.45 ms/layer (`qsa-cost-curve-A1.log`
   routed 8192 = 0.3748 ms; c2 `full_dev`/12 = 0.453). Do not quote the
   9.5 ms/token dense figure for the routed workload.
5. Known sensor limits: (a) all spans are from the +18% instrumented regime;
   (b) the A/A gap of the two arms is 3.4%; (c) `hc_dev` clean 8.26 vs bench
   6.00 means ~2.3 ms of the production HC span is host-pacing bubble, i.e.
   per-family spans overstate device time by ~25%.

---

## 5. Sources

- Spans: `tmp/q4e-sprint-r1-20261014/c2-round2-b{1,2}-worker.log`;
  parser `dec_instr_dev_parse.py`.
- Benches: `decode-misc/misc-A1.log` (HC/GDN/submit cost), `kernel/decode-gu-A{1,2}.log`
  (K1 A/A), `decode-moe/decode-moe-A1.log` (guard FAIL), `kernel/qsa-cost-curve-A1.log`,
  `kernel/qsa-batches-A1.log`, `decode-attn/reuse-{base,base2,pair2}.log`.
- Receipts: `receipt-k5-pair2.json`, `receipt-pair2-e2e.json`,
  `e10-bundle{,-b2}-monitor.json`.
- Docs: `qwen38-r1-kernel-2026-10-14.md` (role K round 1),
  `qwen38-r1-decode-misc-2026-10-15.md` (family pass + HC bench),
  `qwen38-r1-decode-moe-2026-10-15.md` (MoE kernel line closure),
  `qwen38-r1-decode-boundary-2026-10-15.md` (observer effect, PLE inter),
  `qwen38-r1-decode-20-2026-10-15.md` (clean T1 table),
  `qwen38-performance-research-charter-2026-10-14.md` sec 2/5.
- Code: `src/ops/AttentionOps.cpp:432-683,742,1058-1074`;
  `src/ops/HyperConnectionOps.cpp:365-509`; `src/models/Qwen4ExpPrefill.cpp:14-256`;
  `src/models/Qwen4ExpBnb4Backend.cpp:540-571,1446,1517,1911-1918,2589-2620`.
