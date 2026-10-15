# R1 Role K4 - GDN (linear-attention) prefill: upper-bound model, measurement gap, probe spec

Date: 2026-10-15. Author: role K4 (kernel, GDN prefill only). Repo `E:\RiderProjects\Aila`,
HEAD `9abb858` (pair `retained23`). Charter: `docs/plans/qwen38-performance-research-charter-2026-10-14.md`.

Scope discipline for this round: zero GPU actions, zero builds, zero `src/tests/tools` edits, no git writes.
Only this document and `tmp/q4e-sprint-r1-20261014/gdn/` were written. All numbers below are either
MEASURED (with an artifact path) or tagged `[estimate]`. Predicted/derived values carry `[estimate]`
everywhere (the marker requested as the Chinese "estimate" tag is written `[estimate]` to keep this file
ASCII-only). Reference hardware constants are cited from the A770 skill package, the same source the K
audit used, so all ratios are comparable across R1 documents.

---

## 0. Conclusions

1. **The GDN prefill recurrent kernel is 74.82 ms/layer at the production geometry**, not 113.1 ms/layer.
   `tmp/q4e-gdn-k1-20261010/sweep-off.log` (HEAD `e808f63`, 2026-10-10, env off = default path) reports
   `vh=48 n=4096 median_ms=74.8172 device_span_ms=74.6627 per_head_token_ns=380.54` for
   `run_linear_delta_prefill_gpu_batched` = one work-group per value head, serial over 4096 tokens.
   That bench drives the exact production geometry (qh=16, vh=48, Dk=Dv=128, conv kernel=4, Sigmoid gate,
   bf16 input, fp32 state `[48,128,128]`), see section 1.1.
2. **The "GDN 4.07 s/block" segment is a stage, not a kernel.** The stage is `kStageLinearAttn`
   (`src/models/Qwen4ExpBnb4Backend.cpp:2095`), i.e. linear_all_proj NF4 GEMM + the recurrent kernel +
   linear_o_proj NF4 GEMM, serialized on the in-order queue. retained20 re-measurement:
   4126.9 / 4120.9 / 4040.4 ms per 4096-token chunk (`docs/plans/qwen38-r1-cost-model-2026-10-14.md`
   section 3, ledger L11) = 4096.1 ms/chunk = 113.78 ms/layer over 36 GDN layers (36 of 48 layers,
   `C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2/config.json` `layer_types` = 36 linear / 12 full).
   The kernel is 2693.4 ms of that (65.7%); the remaining 1402.7 ms/chunk = 38.96 ms/layer is the two
   NF4 projections plus submits `[estimate]` (a measurement gap, section 4).
3. **Current bandwidth account: the kernel runs at 516.7 GB/s of effective 2R+1W state traffic**
   (38.6547 GB/layer/chunk over 74.8172 ms `[estimate]` from the code-derived byte model), i.e. **1.78x the
   DRAM contiguous read rate (291.11 GB/s) and 1.52x the DRAM copy rate (340.03 GB/s), but only 0.60x of the
   L2 reuse rate (854.6 GB/s)**. The state (3.0 MiB/layer) is partly L2-served; the kernel is not at the DRAM
   wall. The earlier reading "measured = 0.85x of the 2R+1W DRAM model" (`qwen38-r1-kernel-2026-10-14.md`
   section 2.5) was an artifact of charging the whole 113.78 ms/layer stage to the state stream: the stage
   happens to sit at 339.8 GB/s `[estimate]`, i.e. almost exactly the DRAM copy constant 340.03 GB/s.
4. **The "12x SLM headroom" is really 8.0x for the kernel** (74.82 / 9.39 ms), and it is not free:
   an fp32 128x128 state is 65536 B = the entire 64 KB per-work-group SLM cap (A770 hardware reference),
   so an SLM-resident design needs a v-column split (32 KiB per WG) or a bf16 state (precision risk).
   The 12x number itself is still arithmetically reproducible from the *stage* figure
   (113.78 / 9.39 = 12.1x) but overstates what a kernel rewrite can win, because 39 ms/layer of the stage
   is projection work that the state-stream model never described.
5. **Three "if X were free" bounds (14K production request, TTFT 71.82 s; see section 3c):**
   state stream free -> -9.17 s TTFT (-12.7%); state fully in SLM -> -8.02 s TTFT (-11.2%);
   DRAM state writes only at chunk boundaries -> -0.003% TTFT by itself (the residual cost is on-chip
   bandwidth and occupancy, not DRAM traffic). All are far above the 5% e2e gate, so the question is
   delivery, not ceiling. The only structural alternative already in the tree (chunkwise,
   `AILA_Q4E_GDN_CHUNK_MODE=1/2`) measured -2.7% ttft and -4..7% decode
   (`docs/plans/qwen38-qwen4exp-pipeline-status-2026-10-13.md:172`), i.e. about a quarter of its own
   traffic bound - the kernel-level reason is unmeasured today (section 4).
6. **Charter section 3 question 5 verdict (wording in section 5):** by the x-locality criterion the GDN
   prefill kernel is **not** at its upper bound (8.0x from the SLM-traffic bound, 1.78x above the DRAM-read
   model), and by the ALU criterion the question is empty (architectural FP32 floor ~0.012 ms/layer
   `[estimate]`, ~6000x below measured). But 34% of the stage is non-kernel, no arm currently times any of
   it at prefill scale, and the one existing structural alternative delivered only -2.7% e2e. Therefore:
   **do not start a rewrite yet; build the K4 arm first, and apply the gate in section 4.4
   (kernel <3% at two shapes, or no e2e path >=5% -> NO-GO).**

---

## 1. Code facts: prefill entry, state storage, per-token traffic, reuse, switches

### 1.1 Entry path (production)

```
worker ForwardImpl (Qwen4ExpBnb4Backend.cpp:2083 RunAttentionWindow)
  -> RunAttentionWindow            Qwen4ExpPrefill.cpp:557      (tile split only if AILA_Q4E_PREFILL_ATTN_TILE)
  -> RunAttentionLayerTile         Qwen4ExpPrefill.cpp:12      (layer.is_linear branch at :44)
       project_prefill(layer.linear_all_proj)                  (NF4 2560 -> 16480 prefill GEMM)
       run_range(...)                                          Qwen4ExpPrefill.cpp:105-121
         if gdn_chunk_mode_ > 0 && grouped_xmx_supported_
              ops::try_linear_delta_prefill_chunked            GdnChunkOps.cpp:180   (mode 1/2)
         else ops::run_linear_delta_prefill_gpu_batched        GatedDeltaNetOps.cpp:772 (mode 0, default)
       project_prefill(layer.linear_o_proj)                    (NF4 6144 -> 2560 prefill GEMM)
```

The layer-major prototype calls the same tile through `RunAttentionWindowAt`
(`src/models/Qwen4ExpLayerMajor.cpp:135`, used at `:315` at HEAD `9abb858`; the current working tree
carries unrelated layer-major v4 edits that shift the second reference to `:323`); the GDN kernel is
unchanged on that path.

Geometry (all from `C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2/config.json` text_config, mirrored by
`Qwen4ExpBnb4Backend.cpp:419-426`): hidden 2560, q_heads 16, v_heads 48, Dk=Dv=128, conv kernel 4
(conv_rows 3), Sigmoid output gate (`Qwen4ExpBnb4Backend.cpp:594`), 48 layers of which 36 GDN. Derived:
q_dim 2048, kv_dim 6144, qkv_dim 10240, z_dim 6144, all_dim 16480.

### 1.2 Kernel shape and state storage

| Fact | Evidence |
|---|---|
| One work-group per value head, 128 threads, `[[sycl::reqd_sub_group_size(16)]]` -> 48 WGs, 6144 work-items = 384 of 4096 A770 hardware threads (9.4% occupancy `[estimate]`) | `GatedDeltaNetOps.cpp:490-491`; skill `references/hardware/hardware.md` |
| Serial token loop `for (t = 0; t < seq_len; ++t)`, no token-level parallelism | `GatedDeltaNetOps.cpp:518` |
| 5 WG barriers + 3 sub-group reductions per token | `GatedDeltaNetOps.cpp:566,588,598,732,743` and `:579,580,735` |
| S matrix = `state_ptr + hv*128*128`, **fp32 in global (device USM)**, `[v_heads,128,128]` row-major, 65536 B per head, 3.0 MiB per layer | `GatedDeltaNetOps.cpp:514`; `Qwen4ExpBnb4Backend.cpp:233` (`Tensor::allocate` -> `ctx.alloc_device`, `src/core/Tensor.hpp:20-30`) |
| conv history = `conv_state` **fp32 in global USM**, `[3,10240]` = 122880 B per layer | `Qwen4ExpBnb4Backend.cpp:237`; `GatedDeltaNetOps.cpp:545-564` |
| SLM holds only per-token temporaries: q_local/k_local/out_local 512 B each + head_params 8 B = 1544 B (2.4% of the 64 KB/WG cap); the state is never in SLM | `GatedDeltaNetOps.cpp:484-487` |
| Sub-group-cooperative GQA conv path (`grouped_heads` = v_heads > q_heads) reads history rows t-3..t-1 straight out of the bf16 input buffer and skips per-token conv_state writes | `GatedDeltaNetOps.cpp:531-548` (grouped branch), `:568-575` (write path, `!grouped_heads` only) |
| conv_state is committed once per window: 3 rows x 10240 ch x 4 B = 122880 B write per layer per chunk | `GatedDeltaNetOps.cpp:750-753`; `commit_linear_conv_history` `:30-43` |
| State is also snapshotted (D2H) at interior checkpoints: 3.146 MB + 122880 B per GDN layer per checkpoint | `Qwen4ExpPrefill.cpp:127-130` |

### 1.3 Per-token traffic (the accounting that drives the model)

Per (head, token), with `dv` = lane id in the 128-thread WG and the state column `S[k, dv]` at stride
512 B (row-major 128x128 fp32):

| Access | Bytes | Where in code |
|---|---:|---|
| pass 1 read: `sv = S[off]*decay; acc += sv*k_local[j]` (dot with k, for dval) | 65536 | `GatedDeltaNetOps.cpp:646-663` |
| pass 2 read: `sv = S[off]*decay + k_local[j]*dval` | 65536 | `GatedDeltaNetOps.cpp:711-731` |
| pass 2 write: `S[off] = sv` (in place) | 65536 | same |
| **S total** | **196608 (2R + 1W)** |  |
| conv history reads: 13 bf16 loads/lane = 3 prior rows x 3 channels + current row x 3 channels + z (mostly L2/L1 hits, shared across all 48 heads) | ~3328/head/token | `GatedDeltaNetOps.cpp:531-548` |
| conv weights: 4 bf16 per channel, L1-resident | small | `GatedDeltaNetOps.cpp:543` |
| output write: 128 bf16 | 256 | `GatedDeltaNetOps.cpp:741` |

Per layer per 4096-token chunk: `196608 x 48 x 4096 = 38,654,705,664 B` (38.65 GB / 36.0 GiB).
Per 36-layer block: 1.3916 TB. Input-row traffic for the same chunk is 135.0 MB
`[estimate]` (4096 x 16480 bf16), i.e. 0.35% of the state bytes, and is not the issue.
For scale: at the measured 516.7 GB/s effective rate the same kernel costs 0.66 ms/token at decode
(36 layers x 9.44 MB) `[estimate]`, i.e. ~1% of the 66 ms/token decode budget
(`qwen38-r1-cost-model-2026-10-14.md` section 4.1) - the state stream is a prefill-scale problem because
prefill multiplies it by 4096 tokens per chunk.

### 1.4 Reuse available inside a chunk (and what is provably not reusable)

- **The two reads of the same 64 KiB column are the obvious reuse.** They exist because a full column
  (128 fp32) cannot be held per lane: 128 registers/lane at 128 threads consumes the whole regular GRF
  budget. The measured register-pressure experiment agrees: a software-pipelined variant that stages only
  8 elements (8 extra registers) is +84% at `vh=48 n=256` and +92% at `vh=48 n=4096`
  (`tmp/q4e-gdn-k1-20261010/REPORT.md` section 3; env `AILA_Q4E_GDN_MLP_PREFETCH`, default off) - so the
  "free" fix does not exist at this WG geometry. A fix has to *reduce registers per lane or elements per
  lane*, i.e. change the mapping.
- conv history rows t-3..t-1 are shared by all 48 heads of the same layer and are read from the input
  buffer, so they are L2-served and already amortized.
- q/k normalized vectors are already SLM-resident and reused by both passes in the same token.
- Across the chunk boundary nothing is reusable without an on-chip home for the state, because the state
  changes on every token.
- Any tiling that re-enters the kernel (`AILA_Q4E_PREFILL_ATTN_TILE` = 128/256/512, non-default) does not
  change the bytes above (each token still costs 2R+1W), but it *does* forbid an SLM-resident state (the
  on-chip state would have to be re-staged per tile). Default is 0 = one 4096-row window
  (`Qwen4ExpBnb4Backend.cpp:399-401`).

### 1.5 AILA_Q4E switch status (as of HEAD `9abb858`)

| Switch | Default | Production env | Effect on GDN prefill | Measured status |
|---|---|---|---|---|
| `AILA_Q4E_GDN_CHUNK_MODE` 0/1/2 | 0 | 0 | 0 = recurrence (this doc); 1 = FP32 chunkwise control, 2 = BF16 XMX chunkwise (T=16); needs `grouped_xmx_supported_` = A770 name match | ttft -2.7%, decode -4..7% -> not defaulted (`pipeline-status:172`) |
| `AILA_Q4E_GDN_MLP_PREFETCH` | false | unset | software-pipelined state walk inside the same kernel | +84..+92% regression at production geometry (`q4e-gdn-k1-20261010`) |
| `AILA_Q4E_GDN_HISTORY_FUSED` | false | 1 | decode-only conv-history fusion (`Qwen4ExpPrefill.cpp:71`); prefill is unaffected | adopted for decode |
| `AILA_Q4E_VERIFY_GDN_ORDER` | false | 0 | test-only: routes <=4-token prefill through the decode recurrence | test-only |
| `AILA_Q4E_PREFILL_ATTN_TILE` | 0 | unset | splits the attention window into 128/256/512-row tiles (state bytes unchanged, but SLM-residency-hostile) | not measured |
| `AILA_Q4E_PREFILL_XMX` / `AILA_Q4E_FUSED_PREFILL` | false | 1 (the first) | affects the two **projections** only, not the recurrent kernel: `out_features > 8192 && seq_len >= 64` still prefers oneDNN (`Bnb4BitLinear.cpp:2444-2449`) | not measured per projection |
| `AILA_BNB4_CACHE_DEQUANT` | false | unset | if off, each all_proj call dequantizes an 84.4 MB bf16 weight into scratch `[estimate]` | not measured |

Chunk-mode eligibility is real in production (`grouped_xmx_supported_` = device name contains
"Arc(TM) A770", `Qwen4ExpBnb4Backend.cpp:403`), so the chunkwise form is a *config flip* away from being
measurable at kernel level - it just has no arm that reports mode 1/2 at n=4096 (section 4.2).

---

## 2. Measured anchors used by the model

| # | Quantity | Value | Source |
|---|---|---:|---|
| M1 | GDN prefill kernel, vh=48, n=4096, mode 0 (default) | median 74.8172 ms, device span 74.6627 ms, 380.54 ns/head/token, p10 74.7753, p90 74.8613 | `tmp/q4e-gdn-k1-20261010/sweep-off.log` (HEAD `e808f63`, 2026-10-10); 5 samples/cell round-robin + 1 warm-up |
| M2 | Same kernel, vh=48, n=256 | 5.0417 ms (= 410.295 ns/head/token) | same; cross-checked by `--baseline-bench` 5.0043/5.0043/4.9961 ms (`baseline-off.log`) |
| M3 | Same kernel, vh=96, n=4096 (head-count scaling) | 119.031 ms (302.71 ns/head/token) | same |
| M4 | GDN stage (all_proj + kernel + o_proj + submits), retained20, 3 x 4096-token chunks | 4126.9 / 4120.9 / 4040.4 ms | `docs/plans/qwen38-r1-cost-model-2026-10-14.md` section 3 (`AILA_PROFILE_Q4E_STAGES=1`, serialized) |
| M5 | C1-era device split (independent pair) | GDN 4.07 s + full attn 2.85 s + experts 12.58 s ~ 19.8 s/chunk | `docs/plans/qwen38-strata-c1-prefill-staging-screen-2026-10-13.md` section 2 |
| M6 | 14K request TTFT (primed protocol) | 71.82 s (block wall 19.75-20.43 s) | `qwen38-r1-cost-model-2026-10-14.md` sections 1.1/3 |
| M7 | chunkwise e2e (`GDN_CHUNK_MODE=1`) | ttft -2.7% (68.719/68.720 double run), decode -4..7% | `docs/plans/qwen38-qwen4exp-pipeline-status-2026-10-13.md:172` |
| M8 | MLP-prefetch variant e2e-relevant kernel cost | +84% (n=256) / +92% (n=4096) at vh=48 | `tmp/q4e-gdn-k1-20261010/REPORT.md` section 3 |
| C1 | DRAM contiguous read (256 MiB) | 291.11 GB/s | skill `references/hardware/dram-contiguous-roofline.md:21-23` |
| C2 | DRAM contiguous copy (r+w) | 340.03 GB/s | same |
| C3 | L2 reuse | 854.6 GB/s | same |
| C4 | SLM read | 4115.8 GB/s | same |
| C5 | DRAM spec / XMX bf16 validated | 560 GB/s / 100.627 TFLOPS | same `:36-38` |
| C6 | SLM per Xe-core / max SLM per WG / L1 per Xe-core / L2 / HW threads | 128 KB / 64 KB / 192 KB / 16 MB / 4096 | skill `references/hardware/hardware.md` |

Derived ([estimate], arithmetic on M1/M3/M4 and the geometry):

| D# | Quantity | Value |
|---|---|---:|
| D1 | State bytes per layer per 4096 chunk (2R+1W) | 38.6547 GB |
| D2 | Kernel time per 36-layer block | 2693.4 ms (2.69 s) |
| D3 | Stage minus kernel per chunk | 1402.7 ms (38.96 ms/layer) |
| D4 | Kernel effective bandwidth (D1 / M1) | 516.7 GB/s = 1.78x C1, 1.52x C2, 0.60x C3, 0.126x C4 |
| D5 | Stage as pure state stream (D1 / M4 mean) | 339.8 GB/s ~ C2 (the coincidence behind the old "0.85x" reading) |
| D6 | SLM-resident lower bound (D1 / C4) | 9.392 ms/layer (0.338 s/block) |
| D7 | Headroom ratios | kernel 8.0x to D6; stage 12.1x to D6 |
| D8 | Kernel instruction/issue floor at 9.4% occupancy `[estimate]` | ~1.8-2.0 ms/layer (1.42e9 SIMD16 instr/layer at 384 HW threads x 2.05 GHz) |
| D9 | Kernel FP32 ALU floor `[estimate]` | ~0.012 ms/layer (196.6 MFLOP at ~16.8 TFLOP/s) |
| D10 | 14K kernel total (3 x 4096 + 1 x 1656 chunk) | 9.169 s of the 71.82 s TTFT |

---

## 3. Modeling

### 3a. Current bandwidth account (kernel vs stage)

| Line | Bytes / per chunk | Time | Implied rate | Reference |
|---|---:|---:|---:|---|
| State stream (2R+1W), 36 layers | 1.3916 TB | 2693.4 ms (M1 x 36) | 516.7 GB/s | 1.78x C1, 1.52x C2, 0.60x C3 |
| GDN stage (kernel + 2 NF4 projections + submits) | 1.3916 TB attributed | 4096.1 ms | 339.8 GB/s | ~C2; not a physical conclusion, see below |
| Projection + submit residual | n/a | 1402.7 ms | n/a | `[estimate]`, difference of two measurements from two pairs |

Reading: the kernel is **above** the DRAM read/copy rates, so part of the state stream is L2-served
(state footprint 3.0 MiB/layer vs 16 MB L2). It is **below** the L2 reuse rate and far below the SLM rate,
and it runs at 9.4% thread occupancy with 5 barriers + 3 reductions per token, so the binding constraints
are on-chip path bandwidth and latency hiding, not DRAM. The stage-level number (339.8 GB/s) only looks
like a DRAM wall because 38.96 ms/layer of projection work is charged to the same interval.

### 3b. Lower bounds under SLM residency / chunkwise forms

| Variant | State traffic per layer per chunk | Time floor | Notes |
|---|---:|---:|---|
| (i) all state traffic at SLM rate (C4) | 38.6547 GB | 9.392 ms/layer = 0.338 s/block | D6; requires the state to be on-chip, else this is not an achievable rate |
| (ii) chunkwise T=16 (existing `GdnChunkOps.cpp`) | state 1.611 GB (1R+1W per 16 tokens) + scratch ~1.68 GB `[estimate]` = ~3.29 GB | 11.3 ms/layer at C1; 6.4 ms/layer at D4 | measured e2e (M7) is ~2.7% ttft, i.e. the existing implementation is ~10x off its own traffic bound `[estimate]` (512 extra launches/layer plus per-chunk serial solve) |
| (iii) state on-chip for a whole 4096 chunk, DRAM only at boundaries | 6.29 MB (1R+1W of 3.0 MiB) | 0.78 ms/chunk at C1 | needs 64 KiB/WG residency = exactly the per-WG SLM cap (C6); feasible only with a v-column split (32 KiB/WG) or bf16 state; incompatible with `PREFILL_ATTN_TILE` tiling |

Chunkwise scratch traffic `[estimate]` from the tensor shapes in `GdnChunkOps.cpp:166-179`
(per layer per 4096 chunk): prepared 303.6 MB W + 303.6 MB R, matrix 113.2 MB W + 113.2 MB R,
delta+base 201.4 MB W + 201.4 MB R, raw 100.7 MB W + 100.7 MB R, input ~185 MB R (135 prepare +
50.3 finish z slice), output 50.3 MB W; scratch/io subtotal 1.67 GB, plus state 1.61 GB.

### 3c. Three "if X were free" upper bounds (14K request, TTFT 71.82 s, 3 x 4096 + 1 x 1656 chunks)

| X removed | Kernel effect | Stage effect | TTFT | Delta |
|---|---|---|---:|---:|
| **state stream free** (all 2R+1W state memory ops cost 0) | 74.82 -> ~0 ms/layer | 113.78 -> 38.96 ms/layer = 1.403 s/chunk | 62.66 s | **-12.7%** (-9.17 s) |
| **state fully in SLM** (traffic served at C4; projections unchanged) | 74.82 -> 9.39 ms/layer | 113.78 -> 48.35 ms/layer = 1.742 s/chunk | 63.80 s | **-11.2%** (-8.02 s) |
| **DRAM state traffic only at chunk boundaries** | removes DRAM bytes only; per-token cost moves on-chip | 226.5 MB/chunk = 0.78 ms at C1 | 71.82 s | **-0.003%** by itself |

All three are idealizations (no real kernel reaches 0 ms; the all-SLM bound ignores the issue/barrier
floor D8 ~1.8-2.0 ms/layer and the extra SLM bank pressure). They exist to bound the investment: the *ceiling* is
11-13% TTFT, not 2-3%. The realized value must be measured;
(iii) states the structural point that once the state is on-chip for the whole chunk, DRAM traffic stops
being the cost - the residual cost is on-chip bandwidth and occupancy, so "state on chip" and
"chunk-boundary-only DRAM" are the same design target, not two.

### 3d. ALU / x-locality cross-check (charter section 3 question 5 wording)

| Criterion | Floor | Measured | Ratio |
|---|---:|---:|---:|
| FP32 ALU (D9) | 0.012 ms/layer `[estimate]` | 74.82 ms/layer | ~6000x |
| Issue at 9.4% occupancy (D8) | 1.8-2.0 ms/layer `[estimate]` | 74.82 ms/layer | ~40x |
| SLM-resident x-locality (D6) | 9.39 ms/layer `[estimate]` | 74.82 ms/layer | 8.0x |
| DRAM-read stream model (D1/C1) | 132.79 ms/layer | 74.82 ms/layer | 0.56x (kernel is already inside it) |

So the ALU criterion is empty for this kernel (it is 3-4 orders of magnitude away), and the DRAM-stream
model is already refuted by the measurement. x-locality is the only meaningful criterion and it shows
8.0x of theoretical headroom - but the headroom is *structural* (occupancy 9.4%, an on-chip state that does
not fit in one WG's SLM), so it cannot be harvested by block-size or prefetch tuning, as M8 shows.

---

## 4. Minimal verification: the K4 GDN prefill arm

### 4.1 What exists today (no code change needed)

`build-refactor/AilaGdnChunkTests.exe` (target `AilaGdnChunkTests`, `CMakeLists.txt:720-725`) already
contains the exact fixture the model needs: `Fixture(ctx, tokens, qh, vh)` with Sigmoid gate, bf16 input,
fp32 state `[vh,128,128]` + fp32 conv history `[3,qkv_dim]`, and these arms
(`tests/ops/GdnChunkTests.cpp`):

| Arm | Line | Coverage | Timing output |
|---|---:|---|---|
| `--mlp-sweep` | 186-218 | vh in {32,48,64,96} x n in {256..4096}, mode 0 only | per cell: median/p10/p90 wall_ms, device_span_ms, per_head_us, per_head_token_ns; 1 warm-up + 5 samples |
| `--baseline-bench` | 165-185 | n in {30,256}, mode 0 | 3 series x 20 samples, median/p10/p90 + device span |
| `--bench` | 250-275 | n in {30,256}, modes 0/1/2 interleaved round-robin (20 samples x 3 series) | median/p10/p90/device span + CV |
| `--profile-chunk` | 341-349 | n=256 modes 1/2 | none (externally profiled) |
| `--mlp-parity` | 219-249 | bit-exact A/B of the prefetch variant | pass/fail + first-diff index |

So the *mode 0* production kernel is already measurable at production scale with zero code (M1 came from
exactly this). The gaps are: (a) modes 1/2 are only benchable at n in {30,256}; (b) no JSON; (c) no
projection split; (d) the M1 anchor is from HEAD `e808f63`, not the current pair.

### 4.2 Step 1 - zero-code re-anchor (run this first, ~2-3 min GPU)

```powershell
# GPU exclusivity (charter 5.4); must print nothing
Get-Process AilaWorker,AilaServer -ErrorAction SilentlyContinue
# clear AILA_/SYCL_/UR_L0_ so the arm runs the default path
Get-ChildItem Env: | Where-Object { $_.Name -like 'AILA_*' -or $_.Name -like 'SYCL_*' -or $_.Name -like 'UR_L0_*' } | ForEach-Object { Remove-Item "Env:$($_.Name)" }
Get-FileHash build-refactor/AilaGdnChunkTests.exe -Algorithm SHA256
New-Item -ItemType Directory -Force tmp/q4e-sprint-r1-20261014/gdn | Out-Null
build-refactor/AilaGdnChunkTests.exe --mlp-sweep   *> tmp/q4e-sprint-r1-20261014/gdn/gdn-mlp-sweep-A1.log
build-refactor/AilaGdnChunkTests.exe --mlp-sweep   *> tmp/q4e-sprint-r1-20261014/gdn/gdn-mlp-sweep-A2.log
build-refactor/AilaGdnChunkTests.exe --baseline-bench *> tmp/q4e-sprint-r1-20261014/gdn/gdn-baseline-A1.log
build-refactor/AilaGdnChunkTests.exe --bench         *> tmp/q4e-sprint-r1-20261014/gdn/gdn-modes-A1.log
```

Fields to record from A1/A2: `vh=48 n=4096 median_ms`, `p10`, `p90`, `device_span_ms`,
`per_head_token_ns`, `[X4 DEVICE]`, exe SHA256. Judgement:

- A/A: `abs(median_A1 - median_A2) / median <= 3%` at (vh=48, n=4096); otherwise repeat once.
- Re-anchor: if the new median is within 3% of M1 (74.82 ms) the 2026-10-10 anchor transfers to the
  current pair and section 3 stands as written; otherwise substitute the new number and recompute D4/D5.
- `--bench` gives the mode 1/2 vs mode 0 scaling at n in {30,256} (a cheap falsification check on the
  "chunkwise is launch-bound" hypothesis in 3b) but cannot answer the production-scale question.

### 4.3 Step 2 - new arm spec `--gdn-prefill-bench [--json]` (to be implemented later; spec only)

Add to the existing `tests/ops/GdnChunkTests.cpp` (reuse `Fixture`, `reset()`, `ext_oneapi_submit_barrier`
timing, `quantile`), no new source file, no `src/` change:

```
--gdn-prefill-bench [--json] [--seq=N] [--qh=16] [--vh=48] [--mode=0|1|2]
                    [--warm=2] [--samples=45] [--series=3] [--cells=P1,P2]
```

Fixed for v1 (must be printed in the JSON so a receipt is self-describing):
`seq=4096`, `qh=16`, `vh=48`, `head_k_dim=128`, `head_v_dim=128`, `conv_kernel=4`, `gate_act=sigmoid`,
`input_dtype=bf16`, `state_dtype=fp32`, `output_dtype=bf16`, `state_layout=[vh,128,128] row-major
contiguous fp32`, `mode=0` (0 = `run_linear_delta_prefill_gpu_batched`, 1/2 = `try_linear_delta_prefill_chunked`
with `xmx=false/true`), `state_bytes_per_head=65536`, `state_bytes_per_head_token=196608`.
Cells: `P1 = (vh=48, n=4096)` production; `P2 = (vh=96, n=4096)` head-count scaling; optional
`C = (vh=48, n=2048)` shape sweep. Protocol: reset state before the timed barrier (reset excluded),
1 warm-up per cell, samples round-robin across cells (frequency-drift isolation, same convention as
`mlp_sweep`), timing = wall clock between `submit_barrier` calls + device span from event profiling.

JSON contract (`aila.gdn-prefill-bench.v1`); values below are the 2026-10-10 anchor for P1 shown only to
fix field names/order, not a claim by the new arm:

```json
{
  "schema": "aila.gdn-prefill-bench.v1",
  "arm": "gdn-prefill-bench",
  "git_head": "9abb858",
  "binary_sha256": "...",
  "device": "Intel(R) Arc(TM) A770 Graphics",
  "config": {"seq": 4096, "qh": 16, "vh": 48, "head_k_dim": 128, "head_v_dim": 128,
             "conv_kernel": 4, "gate_act": "sigmoid", "input_dtype": "bf16",
             "state_dtype": "fp32", "output_dtype": "bf16",
             "state_layout": "[vh,128,128] row-major contiguous fp32", "mode": 0},
  "protocol": {"warm": 2, "samples": 45, "series": 3, "reset_before_timing": true,
               "timing": "submit_barrier_wall+device_span"},
  "wall_ms": {"median": 74.82, "p10": 74.78, "p90": 74.86, "cv": 0.001},
  "device_span_ms": {"median": 74.66},
  "per_head_token_ns": 380.54,
  "derived": {"state_bytes_per_layer_chunk": 38654705664, "effective_gbps": 516.7,
              "x_dram_read": 1.78, "x_dram_copy": 1.52, "x_l2_reuse": 0.60,
              "x_slm_read": 0.126, "wall_over_device_span": 1.002},
  "guards": {"output_guard_untouched": true, "state_nrmse_vs_cpu_recurrent": 0.0,
             "mode0_vs_mode1_out_max_abs": null}
}
```

Guard requirements (all must be true before a JSON is promoted to a receipt): output sentinel tail not
overwritten, state NRMSE vs the CPU recurrence < 1.5e-2 (`check_gpu` canary), and for mode 1/2 the same
output max-abs vs mode 0 that `correctness` already asserts.

Commands after implementation (one process per cell x mode; two processes for A/A):

```powershell
build-refactor/AilaGdnChunkTests.exe --gdn-prefill-bench --json --seq=4096 --vh=48 --mode=0 *> tmp/q4e-sprint-r1-20261014/gdn/k4-p1-vh48-m0-A1.json
build-refactor/AilaGdnChunkTests.exe --gdn-prefill-bench --json --seq=4096 --vh=48 --mode=0 *> tmp/q4e-sprint-r1-20261014/gdn/k4-p1-vh48-m0-A2.json
build-refactor/AilaGdnChunkTests.exe --gdn-prefill-bench --json --seq=4096 --vh=96 --mode=0 *> tmp/q4e-sprint-r1-20261014/gdn/k4-p2-vh96-m0-A1.json
build-refactor/AilaGdnChunkTests.exe --gdn-prefill-bench --json --seq=4096 --vh=48 --mode=1 *> tmp/q4e-sprint-r1-20261014/gdn/k4-p1-vh48-m1-A1.json
build-refactor/AilaGdnChunkTests.exe --gdn-prefill-bench --json --seq=4096 --vh=48 --mode=2 *> tmp/q4e-sprint-r1-20261014/gdn/k4-p1-vh48-m2-A1.json
```

Optional, only if a projection-split question becomes load-bearing (needs a `src/` change, so it is a
separate request, not part of K4): add a `--split` mode that times `project_prefill` and the recurrent
kernel separately with the same protocol; today the only component split
(`[Q4E Components] ... gdn=`, `Qwen4ExpBnb4Backend.cpp:2634`) is gated on
`decode_verification && count == 2` (`:1790`), i.e. it never fires at prefill scale.

### 4.4 Decision rules (frozen; NO-GO thresholds)

| Rule | Condition | Action |
|---|---|---|
| R0 sanity | A/A spread <= 3% at P1; `wall_ms/device_span_ms <= 1.05` | else repeat once, then treat the arm as unstable and stop |
| R1 wall | mode 0 at P1 and P2: `median <= 1.10 x` L2-equivalent state time (38.6547 GB / 854.6 GB/s = 45.23 ms/layer), i.e. <= 49.8 ms/layer | kernel is already at the fastest memory-path reference short of an on-chip redesign -> close the line |
| R1' headroom | `median >= 1.25 x` L2-equivalent time, i.e. >= 56.6 ms/layer, at both P1 and P2 | headroom exists -> a variant is justified (today: 74.82 = 1.65x) |
| R1'' gray | between R1 and R1' | repeat once, then decide on the repeat |
| R2 chunkwise | mode 1 and mode 2 vs mode 0 at P1 and P2 | if `< 3%` on both shapes, drop chunkwise as a default candidate and record M7's e2e as structural |
| R3 candidate gate | a variant needs `kernel saving >= 3%` at **both** P1 and P2 **and** a projected e2e path `saving_per_layer x 36 x 3.4045 / 71.82 s >= 5%`, i.e. `saving_per_layer >= 29.3 ms` | else **NO-GO** for that variant (write the line into the ledger, do not spend GPU time) |
| R4 default | kernel >= 3% two shapes + e2e >= 5% measured on the 14K fixture (charter 5.3) | only then propose defaulting |

Note: the DRAM-read stream model (132.79 ms/layer) is already beaten by M1, so no rule is defined against
it - "the kernel is DRAM-bound" is a finding this arm cannot produce. R3 is the operative NO-GO: the ideal
bounds in 3c (65.4 and 74.8 ms/layer of headroom) *pass* it, while
the only realized structural alternative (M7, ~1.5-3 ms/layer effective `[estimate]`) fails it by ~10x.
So the arm's job is to convert the 8.0x theoretical headroom into a measured candidate delta, and the
answer "the kernel is already at its wall" is also acceptable (R1) - both outcomes close the question.

E2E arm (only after a kernel-level >= 3% two-shape delta exists), 14K fixture, primed protocol:

```powershell
python tmp/q4e-sprint-r1-20261014/run.py k4-gdnchunk-b1 AILA_Q4E_GDN_CHUNK_MODE=1
python tmp/q4e-sprint-r1-20261014/run.py k4-gdnchunk-b2 AILA_Q4E_GDN_CHUNK_MODE=1
```

(The runner clears AILA_/SYCL_/UR_L0_, applies `docs/plans/qwen38-server-mtp-2026-10-07.json`
worker_environment + MTP=0/reserve 3584/chunk 4096, and needs a fresh label directory per arm.)

---

## 5. Answer to charter section 3 question 5 (conclusion wording)

Charter text (translated, section 3 question 5): "Have prefill GDN (4.07 s/block) and full attention
(2.85 s/block) already reached the kernel upper bound for their shape; compare against the x-locality /
ALU criteria (P4 tooling reusable)".

Recommended wording for the ledger (ASCII):

> **Q3.5 GDN part - ANSWERED FOR x-LOCALITY, NOT YET FOR ATTAINABILITY.** The GDN prefill *stage*
> (4096.1 ms/chunk at retained20, 4.07 s at C1) is not a kernel number: the recurrent kernel measured in
> isolation at production geometry is 74.8172 ms/layer = 2693.4 ms/chunk (65.7%), and 38.96 ms/layer
> (~1.40 s/chunk) is the two NF4 projections + submits `[estimate]` with no prefill-scale arm. The kernel
> is **not** at its kernel upper bound by the x-locality criterion: it runs at 516.7 GB/s of effective
> 2R+1W state traffic, 1.78x the DRAM contiguous read rate and 1.65x the L2-reuse-bound time, with 8.0x
> between it and the SLM-resident traffic bound (9.39 ms/layer) `[estimate]`; the old "0.85x of the
> 2-read+1-write DRAM model" reading came from charging the whole stage to the state stream and should be
> retired. The ALU criterion is empty for this kernel (FP32 floor ~0.012 ms/layer, ~6000x below
> measured) `[estimate]`. The 12x headroom figure is reproducible only against the *stage*
> (113.78/9.39); against the kernel it is 8.0x, and it is not free because an fp32 128x128 state equals
> the 64 KB per-WG SLM cap. Verdict: **headroom exists, attainment unproven** - the existing structural
> alternative (chunkwise modes 1/2) delivered only -2.7% ttft e2e, so the next step is the K4 kernel arm
> (section 4) with the frozen gates; a rewrite is GO only if the kernel shows >= 3% at two shapes and an
> e2e path >= 5% TTFT, otherwise NO-GO and close the line.

---

## 6. Open items and non-claims

1. The 39 ms/layer projection residual (D3) is a difference of two measurements taken on different pairs
   (retained20 stage vs `e808f63` kernel) - it is `[estimate]` and it is the largest single unmeasured
   term inside the GDN stage. Do not quote it as measured until the `--split` arm exists.
2. No DRAM-vs-L2 traffic attribution exists for the kernel (would need VTune counters). All bandwidth
   reasoning here is byte-model + rate-reference arithmetic `[estimate]`.
3. The chunkwise scratch estimate (3b) is derived from tensor shapes, not from a profiler; the mode 1/2
   numbers from 4.2/4.3 are what make it falsifiable.
4. This document proposes no `src/` change and no default flip. Anything that changes the summation order
   (e.g. a column-split or register-resident redesign) is "output will change" territory: keep raw token
   ids and an independent validator per charter 5.9 before any adoption.
5. Design directions that the numbers above select (for the next round, not this one): (i) split the 128
   v-columns across 2 WGs per head - each column is mathematically independent in both passes, so this is
   order-preserving and doubles WGs to 96 while halving per-WG state to 32 KiB; (ii) SLM-resident state on
   top of (i) (32 KiB < 64 KB cap) with the chunk-boundary-only DRAM write; (iii) register-resident
   column blocks (e.g. 512-thread WG, 32 fp32/lane) which removes both global reads but changes reduction
   order. (i) is the only one compatible with the bit-exactness discipline and is therefore the first
   candidate to gate.

---

## 7. Sources

- Code (HEAD `9abb858`): `src/ops/GatedDeltaNetOps.cpp` (30-43, 437, 440, 484-487, 490-491, 514, 518,
  531-575, 579-580, 588, 598, 646-663, 711-731, 732-743, 741, 750-753, 772, 795);
  `src/ops/GdnChunkOps.cpp` (9, 45, 88, 106-118, 119-149, 151, 166-179, 180-199);
  `src/models/Qwen4ExpPrefill.cpp` (12, 44-121, 127-130, 557);
  `src/models/Qwen4ExpBnb4Backend.cpp` (233, 237, 399-405, 594, 1790, 2083, 2095, 2634);
  `src/models/Qwen4ExpLayerMajor.cpp` (135, 315); `src/ops/Bnb4BitLinear.cpp` (2027-2035, 2442-2449);
  `src/core/Tensor.hpp` (20-30); `src/utils/Qwen38A770Preset.hpp` (23);
  `tests/ops/GdnChunkTests.cpp` (165-185, 186-218, 219-249, 250-275, 341-349).
- Measurements: `tmp/q4e-gdn-k1-20261010/sweep-off.log`, `baseline-off.log`, `REPORT.md` (M1/M2/M3/M8);
  `docs/plans/qwen38-r1-cost-model-2026-10-14.md` sections 1.1/3/8 (M4/M6, ledger L11);
  `docs/plans/qwen38-strata-c1-prefill-staging-screen-2026-10-13.md` section 2 (M5);
  `docs/plans/qwen38-qwen4exp-pipeline-status-2026-10-13.md` 103/142/172 (M7, switch defaults, env);
  `docs/plans/qwen38-server-mtp-2026-10-07.json` worker_environment (production env).
- References: `docs/plans/qwen38-performance-research-charter-2026-10-14.md` (0, 1.1, 2, 3.5, 5);
  `docs/plans/qwen38-r1-kernel-2026-10-14.md` (0, 1.1, 2.5); A770 skill
  `references/hardware/dram-contiguous-roofline.md` (C1-C5) and `references/hardware/hardware.md` (C6).
- Model geometry: `C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2/config.json` text_config.
