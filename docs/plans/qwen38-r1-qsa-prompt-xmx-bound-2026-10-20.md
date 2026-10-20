# Qwen4Exp QSA prompt attention: 12+4-row XMX bound (2026-10-20)

Scope: A770, production Qwen4Exp BF16 KV, sparse QSA prompt path. This note
started from existing receipts and source, then closed with one standalone A/B
probe.

## 0. Verdict
**NO-GO. The bounded probe measured the 12+4 XMX prototype as 7.1x slower at
2052 and 8.0x slower at 4096, and it also failed the first numeric bound.**

The candidate has a real structural distinction from Aila's current prompt
kernel: 12 query heads plus 4 zero rows can cover one KV head in one
work-group, while the current 4-head path reads the same K/V three times per
KV head ([`AttentionOps.cpp:5652`](../../src/ops/AttentionOps.cpp#L5652),
[`AttentionOps.cpp:5711`](../../src/ops/AttentionOps.cpp#L5711)).
The prior is nevertheless strongly negative. On Intel B70, Strata's earlier
XMX prompt attention was 3x slower than its FP32 fallback per chunk; v2 improved
to 13.4 vs 5.5 ms, still about 2x slower, and remained gather-bound with only
12 of 16 matrix rows useful ([`INTEL.md:299`](../../tmp/Strata/docs/INTEL.md#L299),
[`INTEL.md:362`](../../tmp/Strata/docs/INTEL.md#L362)). The K8V4 form was 0.84x
([`qsa_prompt_attn.cpp:1113`](../../tmp/XeStrata/src/kernels/xe/qsa_prompt_attn.cpp#L1113)).

The implementation was measured once with production-shaped 64-row tiles:

```text
case=2052 A=8.434167 ms B=59.719792 ms speedup=0.141229
  numeric bound1=0 exact=99.771881% one_ulp=99.982198%
case=4096 A=6.961250 ms B=55.576563 ms speedup=0.125255
  numeric bound1=0 exact=99.777985% one_ulp=99.980418%
summary numeric_gate=fail speed_gate_40pct_both=fail probe_go=no
```

The one-stream geometry did not overcome gather/tile overhead on A770. This
candidate is closed; do not integrate it into the production dispatcher.

## 1. Current shape and evidence
Existing model/kernel facts ([`qwen38-r1-kernel-2026-10-14.md:4`](qwen38-r1-kernel-2026-10-14.md#L4)):
48 layers = 36 GDN + 12 full attention; Hq=24, Hkv=2, D=256, hence 12 query
heads per KV head; indexer 4x128, compress ratio 4, block budget 2048, hence
top-512 blocks. QSA becomes sparse at 2052 visible tokens (513 blocks kept as 512).

Production execution:
The gate is `(start_pos + count) / 4 > 512`
([`Qwen4ExpPrefill.cpp:330`](../../src/models/Qwen4ExpPrefill.cpp#L330));
queries use 16..64-row tiles, commonly 64
([`Qwen4ExpBnb4Backend.cpp:1266`](../../src/models/Qwen4ExpBnb4Backend.cpp#L1266)).
The inner kernel defaults to four query heads per 64-thread work-group
([`AttentionOps.cpp:5652`](../../src/ops/AttentionOps.cpp#L5652)), so each row
launches three groups per KV head (`12 / 4`), each re-streaming K/V
([`AttentionOps.cpp:5661`](../../src/ops/AttentionOps.cpp#L5661),
[`AttentionOps.cpp:5719`](../../src/ops/AttentionOps.cpp#L5719)).

The only current cost receipt is stage-level:

| item | receipt | status |
|---|---:|---|
| full-attention stage per 4096 block | 2.85 s | measured envelope |
| per full layer | 237.5 ms | derived |
| share of 19.9 s block | 14.3% | derived |

Sources: [`charter:32`](qwen38-performance-research-charter-2026-10-14.md#L32),
[`r1-kernel:39`](qwen38-r1-kernel-2026-10-14.md#L39),
[`r1-prefill-supply:105`](qwen38-r1-prefill-supply-2026-10-14.md#L105).
Important gap: no Aila receipt isolates `attention_prefill_qsa_batched`. The
2.85 s includes projections/indexer/attention, so the inner-kernel share is
`[待测]`; assigning all 2.85 s to selected attention would overclaim.

## 2. Why 12+4 is plausible, and the absolute bounds
Production uses BF16 KV by default; INT8 KV is opt-in
([`charter:15`](qwen38-performance-research-charter-2026-10-14.md#L15)).
Shared inference explicitly requires BF16 KV
([`InferenceWorker.cpp:546`](../../src/worker/InferenceWorker.cpp#L546)).

At the sparse budget, use 2050 selected cells (2048 block cells plus up to three
tail tokens). One K/V pass is `2050 x 2 KV heads x (K+V) x 256 dims x 2 B =
4.1984 MB` per query row per full layer.

Aila's three 4-head groups therefore stream `3 x 4.1984 = 12.595 MB` per query
row/layer, or `51.59 GB/layer` at 4096 rows; at 291.11 GB/s that is
`177.2 ms/layer`, `2.13 s` over 12 layers.

That is the same order as 237.5 ms/layer, making the duplicated K/V stream a
credible major contributor. It is an estimate, not a counter measurement; L2
reuse can reduce the DRAM component.

The candidate's one-pass DRAM floor is `17.20 GB/layer @291.11 GB/s =
59.1 ms/layer`, or `0.709 s` over 12 layers for a 4096-row sparse block.

For 2052 uniform sparse-budget rows, the floors are 0.355 s for one pass and
1.066 s for the current three-pass stream. A 2052-token prompt from position 0
has shorter early causal prefixes, so 1.43 s below is a conservative ceiling.

### "If X were free" table

Rows are `[估算]` from the stage receipt unless marked measured. They are
non-additive upper bounds.

| X made free | 2052 prompt | 4096 prompt | per token | reading |
|---|---:|---:|---:|---|
| whole measured full-attention stage | <=1.43 s | 2.85 s measured | 0.696 ms | maximum stage saving |
| inner prompt-attention kernel | <=1.43 s | <=2.85 s | <=0.696 ms | true value unknown |
| XMX arithmetic at validated BF16 peak | 12.3 ms | 24.6 ms | 6.0 us | below 1% of stage |
| XMX arithmetic with 12/16 useful rows | 16.4 ms | 32.8 ms | 8.0 us | padding is not the lever |
| current 3 K/V streams reduced to 1 | 0.355 s floor | 0.709 s floor | 0.173 ms | structural target |

The XMX floor uses 206.4 GFLOP/layer at 4096 and 100.627 TFLOPS
([`r1-kernel:21`](qwen38-r1-kernel-2026-10-14.md#L21),
[`r1-kernel:88`](qwen38-r1-kernel-2026-10-14.md#L88)). Matrix arithmetic alone
is not the lever; only the one-stream 12+4 gather topology is worth probing.

## 3. Numerical order and tolerance
The 12+4 design is **not bitwise equal** to current FP32 accumulation:

- 12 real query rows plus 4 zero rows execute as an 8x16x16 XMX tile.
- QK is split into four 64-dimension groups and accumulated by matrix tiles.
- Online-softmax rescaling and chunked PV change accumulation order.
- PV needs probabilities as matrix inputs; one low-precision P adds input
  rounding, not merely summation order.

XeStrata avoids most extra loss by scaling and splitting Q and P into FP16
hi+lo parts, retaining about 22 bits
([`qsa_prompt_attn.hpp:12`](../../tmp/XeStrata/include/strata/kernels/qsa_prompt_attn.hpp#L12)).
If Aila uses a single BF16 P, that is additional quantization and should not be
admitted as a pure rounding-order relaxation. A BF16 `P_hi/P_lo` pair is the
minimum defensible mapping; FP16 hi/lo needs a finite-range check first.

Existing parity bounds from the 12+4 XMX reference
([`qsa_prompt_attn_parity.cpp:11`](../../tmp/XeStrata/src/kernels/qsa_prompt_attn_parity.cpp#L11),
[`:217`](../../tmp/XeStrata/src/kernels/qsa_prompt_attn_parity.cpp#L217)):
`max_abs(new-old) <= 1e-4 * max_abs(old)` and
`max_abs(new-FP64) <= max(4 * max_abs(old-FP64), 1e-6 * reference_scale)`.

Aila writes BF16 attention output, so also record exact/1-ULP fraction,
`max_abs`, NRMSE, and cosine. Bitwise equality is not required; these diagnostics
do not replace the two bounds above.

## 4. Minimum generalizable probe
Use one standalone probe, not a production dispatcher or env sweep.

**Arms**

- A: current `attention_prefill_qsa_batched`, `AILA_QSA_HEADS_PER_WG=4` forced.
- B: one 12+4-row XMX kernel, independently implemented; do not copy the
  mixed-license XeStrata source.

**Shapes**

- BF16 KV, Hq=24, Hkv=2, D=256, 512 blocks, compress ratio 4.
- Real causal lengths 2052 and 4096 from the same tokenizer prefix.
- Production 64-row tiles; 2052 covers first sparse/tail state, 4096 the
  production block.
- Precompute each row's active blocks and 0..3 tail IDs with the real selector
  outside timing. Avoid contiguous synthetic IDs that are unrealistically
  cache-friendly.

**Protocol**

- One warm-up per arm, then one timed A/B round under GPU-exclusive ownership.
- Sum all 64-row tile kernels with device events; also print per-tile
  median/p10/p90, bytes, and GiB/s.
- No fixed sleep; poll process/event with a bounded wait.

This generalizes to the production sparse prompt path. Any case below 2052 is
invalid for sparse QSA.

## 5. Go / no-go
**Hard NO-GO**

- NaN/Inf or failure of either numeric bound.
- B is slower at either 2052 or 4096.
- B needs per-query union gathering, dynamic worklists, or other new machinery
  merely to reach parity in the probe.

**Pass, but not default**

- B is 15..40% faster but still near the stage-share limit.
- Or the numeric difference is not explained by tile order plus the declared P
  representation.

**GO to integration**

- Numeric bounds pass.
- B is at least 40% faster at both 2052 and 4096 in the single A/B round.
- Rationale: `5% end-to-end / 14.3% stage share ~= 35%`; 40% leaves margin for
  the unmeasured non-attention part of the full-attention stage.

The measured probe hit the hard NO-GO branch. Receipt:
`tmp/q4e-qsa-prompt-xmx-20261020/probe-fixed.log`. No production integration
or further kernel work is justified.
