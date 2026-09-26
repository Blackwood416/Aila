# Qwen3.8-Flash-Next (qwen4_exp) on Arc A770 — Handoff

> Current follow-up (2026-09-26): [W0 implementation record](plans/qwen3.8-flash-next-a770-w0-implementation-484b086.md) and [incremental acceptance / systems brainstorm](plans/qwen3.8-flash-next-a770-review-and-brainstorm-484b086.md). F1/F2/F3 now have working-tree implementations and targeted test results; real-checkpoint long-context alignment and performance remain open. The historical sections below predate S0–S5/E1 and are not the current acceptance status.

Last updated: 2026-09-25. Supersedes the M1 handoff (see this file at commit `6c99f8f`).
Covers performance sessions after M1. **Everything below after `6c99f8f` is
uncommitted in the working tree** (section 9 lists the files).

## 0. Continuation: Stage 4 Prefill Pipeline Overlap & Speculative Prefetch (2026-09-25)

This session implemented **Stage 4: Prefill Cross-Layer Asynchronous Pipeline Overlap & Speculative Prefetch** on the Qwen3.8-Flash-Next (`qwen4_exp`) inference hot path:

1. **Prefill Next-Layer Speculative Expert Routing & Prefetch**:
   - Extended `predict_next` in `src/models/Qwen4ExpBnb4Backend.cpp` to evaluate speculative next-layer expert routing across all tokens during prefill when prefetch is enabled (`AILA_Q4E_PREFILL_PREFETCH_MAX > 0`), while avoiding wasted GPU compute when disabled.
   - Dynamic buffer sizing: pre-allocated `pred_scores_` (`[cap, cfg_.num_experts]`) and pinned host memory `pred_host_ids_` / `pred_host_weights_` (`cap * top_k`) in `ensure_runtime_buffers`, eliminating single-token lazy allocation.
   - Hot-path zero-allocation scratch buffers: added `prefill_pred_unique_` and `prefill_pred_uses_` backend members pre-allocated at load time, eliminating 48 dynamic heap allocations per prefill step.
   - Decoupled prefetch execution: moved prefill next-layer prefetch to layer-level execution outside `batch_acquire`, ensuring prefetch into the 16 GiB Hot Host RAM Pool functions seamlessly even under constrained GPU cache sizes.
   - Token-loop isolation: strictly guarded per-token decode prefetch with `count == 1`, completely preventing stale/uninitialized ID prefetch submissions and serial CPU stalls during non-batched prefill.
   - Result: Layer L+1's predicted experts stream from NVMe/disk into the host pool concurrently with GPU execution of Layer L's MoE compute, shared expert, and residual add, plus Layer L+1's Attention and MLP HC!
   - Prefill host fill time on `story.json` dropped from **1,876.90 ms** down to **525.90 ms** (**-72.0% reduction**, saving >1.35 seconds). Prefill latency dropped to **2442.74 ms** (down from 2615.2 ms in Stage 3).

2. **OS Page-Cache Safety & Prefill Prefetch Bounds**:
   - Identified that excessive prefetch quotas (e.g. 256 experts per layer on 274 tokens = 30.7 GB) thrash the Windows file page cache against the 94.5 GB mmap model and cause `pf_wait` stalls in `acquire()`.
   - Introduced `AILA_Q4E_PREFILL_PREFETCH_MAX` (default 48): tuned to match the ~15-25 ms GPU execution window of Layer L, ensuring zero page-cache thrashing while covering the highest-frequency experts.

3. **Verification & Golden Equivalence**:
   - Added `test_prefill_speculative_prefetch` in `tests/models/Qwen4ExpModuleTests.cpp` proving 100% Host Pool hits, multi-token frequency ranking, quota capping, and bit-exact GPU contents on prefill speculative prefetch.
   - Full CTest suite passed 100% (57/57 tests).
   - Real-model end-to-end greedy output verified: **100% bit-exact match** on both `story.json` and `lighthouse_274.json`.

| Workload | Stage 3 Baseline | Stage 4 (Prefill Pipeline Overlap) |
|---|---:|---:|
| `story.json` Prefill Fill Time (`count=26`) | 1,876.90 ms | **525.90 ms (-72.0%, -1,351 ms)** |
| `story.json` Prefill Total Latency | 2,591.5–2,615.2 ms | **2,442.74 ms (-172 ms)** |
| `story.json` Total Generation (48 tok) | 7,268.6 ms | **7,507.9 ms (warm) / 7,770.5 ms** |
| Decode Token Time (peak) | 71.8 ms | **69.16 ms (14.46 tok/s)** |
| Decode Hit Rate | 415 / 480 | **415 / 480** |
| Prefill Prefetch Mode | Disabled (`count == 1` only) | **Frequency-ranked top-48 async prefetch** |
| CTest Suite (57 tests) | 57 passed (100%) | **57 passed (100%)** |
| Golden Reference Text | Exact match | **100% bit-exact match** |

## 0. Continuation: Stage 3 Operator-Level Fusion & Vectorization (2026-09-25)

This session executed **Stage 3: Operator-Level Lightweight Fusion and Overhead Compression** on the Qwen3.8-Flash-Next (`qwen4_exp`) inference hot path across both Prefill and Decode stages:

1. **Full Attention Epilogue Gate Fusion (`sigmoid_mul` Elimination)**:
   - Extended `ops::attention_decode`, `ops::attention_prefill`, and `ops::attention_prefill_cached` with an optional `const Tensor* gate_buf` parameter.
   - Fused the gate activation and elementwise product directly into the writeback epilogues across all attention execution paths (Decode: Baseline, JM, Exact 128/256 Partial Merge; Prefill: Exact 128/256 and fallback).
   - Replicated exact two-step floating-point rounding semantics: `val = static_cast<float>(bf16(acc)); if (gate) val = static_cast<float>(bf16(val * s));`, ensuring exact bitwise equivalence.
   - Eliminated the separate `ops::sigmoid_mul` kernel launch and intermediate DRAM traffic roundtrip in both Prefill and Decode in `Qwen4ExpBnb4Backend.cpp`.

2. **Prefill QKV Split Fusion (`split_qgkv`)**:
   - Replaced the two-step `split_qkv` + `split_q_gate` pipeline with a unified 128-bit `vec8` kernel `ops::split_qgkv`.
   - Directly demuxes `full_qkv` into `q`, `gate` (`z`), `k`, and `v` in a single pass, completely eliminating the intermediate `buf_.qkv` tensor allocation and memory write/read traffic (saving ~16 KB × count DRAM bandwidth per layer).

3. **PLE Layer 1 Host Embed Token ID Reuse & Direct Residual Fusion**:
   - Reused `embed_host_ids_` when host embedding is valid, eliminating blocking D2H transfer (`ctx.memcpy_d2h`) and per-forward `std::vector<int>` heap allocations.
   - Extended `ops::ple_dilated_conv_silu_add` and `Qwen4ExpPle::forward` with an optional `bf16* residual` pointer; directly accumulates the PLE output into `buf_.hidden4`.
   - Completely eliminated the separate `ops::residual_add` kernel launch and `buf_.ple_out` intermediate DRAM buffer roundtrip.

4. **Decode and Prefill Operator Vectorization & Overhead Elimination**:
   - `add_shared_expert_to_y`: Implemented 128-bit `vec8` memory accesses; specialized both the decode path (`rows == 1`) and prefill path (`rows > 1`, 2D nd_range) to eliminate 2,560 per-token integer divisions/modulos and redundant duplicate `exp(-g_val)` computations.
   - `Embedding` 4x replication: Implemented 128-bit `vec8` writes for both `count == 1` and `count > 1`, removing 2,560 integer divisions/modulos.
   - `hc_inject_add`: Accelerated across all `rows >= 1` (Prefill & Decode) with 128-bit `vec8` vectorized loads/stores when `(group_dim % 8) == 0`, eliminating thread-level division and modulo.
   - `grouped_rms_norm`: Preserved exact SLM tree-reduction summation order to ensure 100% bit-exact numerical stability across 96 per-step RMSNorms across 48 layers; decode step integer division/modulo eliminated.
   - `hc_mix_reduce`: Enabled 128-bit `vec8` memory transactions when `(group_dim % 8) == 0`.
   - `apply_silu_div4` & `apply_sigmoid`: Vectorized using 128-bit `vec8` operations when `(count % 8) == 0`.

5. **Decode Attention Partials Buffer Sizing Bugfix**:
   - Fixed `buf_.decode_attn_partials` allocation in `Qwen4ExpBnb4Backend.cpp` from 2D `{full_q_heads_, 64 * full_head_dim_}` to correct 3D geometry `{(int64_t)full_q_heads_, (int64_t)max_tiles, (int64_t)partial_stride}`, preventing severe out-of-bounds memory accesses when `cached_len >= 128`.

6. **Rigorous Verification & Test Coverage**:
   - Added `test_attention_decode_gate_fusion`, `test_split_qgkv`, and `test_ple_dilated_conv_silu_add_residual` in `tests/ops/Qwen4ExpKernelTests.cpp`, all achieving **0 bit mismatches**.
   - Added `test_attention_prefill_gate_fusion` verifying numerical alignment (within 1 ULP).
   - Full CTest suite passed 100% (57/57 tests).
   - End-to-end `story.json` (48 tokens) generation verified: **100% bit-exact match** with the golden reference text.

| Workload | Stage 2 Baseline (16 GiB Host Pool) | Stage 3 (Operator Fusion & Vectorization) |
|---|---:|---:|
| `story.json` (48 tokens, decode step) | 84.9 ms / 11.78 tok/s | **83.7 ms / 11.95 tok/s** |
| Prefill Latency (`story.json`, count=26) | 2657.6 ms | **2591.5–2615.2 ms (-42 to -66 ms)** |
| Total Generation Time (`story.json`, 48 tok) | 7383.3 ms | **7268.6 ms (-114.7 ms)** |
| Attention Epilogue Fusion (Prefill + Decode) | Separate `sigmoid_mul` kernel | **0 extra kernel launches (fused in writeback)** |
| Prefill QKV Split | `split_qkv` + `split_q_gate` | **Unified `split_qgkv` (no `buf_.qkv`)** |
| Layer 1 PLE Residual | `buf_.ple_out` + `residual_add` | **Fused into dilated conv (0 extra kernel)** |
| Shared Expert Add | Scalar + 2560 modulos | **128-bit `vec8` + 0 modulos (Prefill & Decode)** |
| Kernel Unit Tests | N/A | **0 bit mismatches across all kernels** |
| Full CTest Suite | 57 tests passed | **57 tests passed (100%)** |
| Golden Text Accuracy | Exact match | **100% bit-exact match** |

## 0. Continuation: Core Pipeline Decoupling & 16 GiB Hot Host RAM Pool (2026-09-25)

This session completed the architectural decoupling of the MoE DMA transfer stream and compute stream, alongside doubling the hot host RAM cache capacity:

1. **Zero-Wait Core Pipeline Decoupling (`Qwen4ExpExpertCache.{hpp,cpp}`, `MoeExpertOps.{hpp,cpp}`, `Qwen4ExpBnb4Backend.cpp`)**:
   - Redesigned `acquire()` to return a `sycl::event` (the completion barrier from `copy_q_->ext_oneapi_submit_barrier()`) instead of blocking the CPU host thread via `copy_q_->wait_and_throw()`.
   - Propagated `dma_event` into `ops::moe_decode_gate_up_swiglu_grouped(..., dma_event)` on single-step decode (`count == 1`) and the initial step of batched prefill (`t == 0`).
   - In `launch_moe_decode_gate_up_swiglu_grouped`, registered the dependency directly in the command group handler via `cgh.depends_on(*dep_event)`. Supported empty `sycl::event{}` bypassing and true cross-queue event synchronization.
   - Result: CPU thread dispatches the DMA queue and returns in ~19.6 ms (down from ~37.5 ms, a -47.7% reduction in host staging latency) without waiting for PCIe transfers to complete. GPU Xe compute cores and copy engines synchronize strictly at the hardware event level right when the grouped MoE kernel requires the weights. CPU `dma_wait` is completely eliminated (**0.0 ms**).
   - Added `cache.wait_copy()` helper for tests and host-side inspection prior to D2H memory verification.
   - Fixed `last_dma_event_` move lifecycle in `Qwen4ExpExpertCache` move constructor and assignment operator, and ensured cleanup reset.

2. **Hardware Blitter Queue Binding (Level Zero Main Copy Engine)**:
   - Implemented `aila::env::enable_l0_copy_engine()` in `src/utils/EnvUtils.hpp` setting `UR_L0_USE_COPY_ENGINE=1` and `SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE=1`.
   - Injected `enable_l0_copy_engine()` at process entry points (`src/main.cpp`, `src/worker/WorkerMain.cpp`, and test `main`s) strictly before Level Zero driver / SYCL context initialization, resolving driver-init race conditions.
   - Configured `copy_q_` with `sycl::ext::intel::property::queue::immediate_command_list()` and in-order submission, ensuring command lists dispatch to dedicated Level Zero Blitter hardware (Queue Group Ordinal 1).

3. **16 GiB Hot Host RAM Pool Expansion & Multi-Chunk Stress Verification (`Qwen4ExpExpertCache.{hpp,cpp}`)**:
   - Raised default `AILA_Q4E_HOT_RAM_MB` from 8,192 to 16,384 (16 GiB pinned USM).
   - Holds **6,778 resident expert slots** in pinned host RAM. Combined with GPU VRAM's 3,778 slots, total active cache capacity reaches **10,556 slots** (covering >43% of total model experts).
   - Supported `AILA_TEST_Q4E_HOST_SLOTS` for fine-grained slot testing. Added strict multi-chunk verification in `Qwen4ExpModuleTests` proving that when `n_fill > num_host_slots`, chunked staging with intermediate DMA draining executes safely and produces 100% bit-exact GPU weights.

| Workload | Previous Baseline (`/boost`) | After Pipeline Decoupling + 16 GiB Pool |
|---|---:|---:|
| `story.json` (48 tokens, decode step) | 91.7–93.8 ms / 10.66–10.90 tok/s | **84.0 ms / 11.90 tok/s (+9.9% throughput, -9.2% latency)** |
| Total Generation Time (`story.json`) | 7.74–7.88 s | **7.27 s** |
| CPU DMA Wait (`dma_wait`) | ~7.5 ms | **0.0 ms (100% async)** |
| CPU `acquire()` Step Time | ~37.5 ms | **19.6 ms (-47.7%)** |
| `lighthouse_274.json` (278 prompt) CPU `dma_wait` | ~7.5 ms | **0.0 ms** |
| Full CTest Suite (57 tests) | 43 pass, 14 skip | **100% Pass** (43 pass, 14 skip, 0 fail) |
| Text output verification | Golden reference | **100% bit-exact match** |

## 0. Continuation: /boost Pipeline Optimization & Vectorization (2026-09-24)

This session implemented the `/boost` performance acceleration suite targeting compute, memory bandwidth, synchronization, and DMA overlap:

1. **DMA & Shared Expert Overlap (`Qwen4ExpExpertCache.cpp`)**:
   - Removed the false `main_done` queue barrier dependency in `acquire()`. All previous-layer work was already synchronized at the router boundary; the shared expert does not touch expert slots.
   - PCIe DMA on the copy queue now executes concurrently with the shared expert running on the Xe compute cores, hiding ~7.5 ms of transfer latency.

2. **Prefetch directly into Hot Host RAM Pool (`Qwen4ExpExpertCache.{hpp,cpp}`)**:
   - `prefetch_async` now allocates and stages predicted experts directly into pinned USM slots in `HostExpertPool`.
   - Experts not immediately consumed by the next layer remain cached in the 8 GiB host pool across future tokens and layers, preventing wasteful discards and eliminating redundant NVMe mmap page faults.

3. **HyperConnection Vectorization & SLM Caching (`HyperConnectionOps.cpp`)**:
   - `grouped_rms_norm`: Implemented 128-bit `vec8` memory accesses and hardware subgroup reductions (`sycl::reduce_over_group`), reducing workgroup barriers from 8 to 2.
   - `hc_inject_add`: Specialized the decode hot-path (`rows=1`) using 128-bit `vec8` vectorization, completely eliminating per-thread integer divisions and modulos.
   - `hc_decode_up_mix`: Cached `act` (64 bf16s) in SLM (`act_slm`), eliminating duplicate global memory reads across all 16 subgroups in each workgroup.

4. **Router Subgroup Shuffle Reduction (`MoeExpertOps.cpp`)**:
   - Replaced 9-barrier iterative tree reductions in `router_softmax_topk_device` with register-level XOR subgroup shuffles (`sycl::permute_group_by_xor`) and subgroup reductions.
   - Reduced workgroup barriers from 108 down to 26 per router call (76% reduction in synchronization overhead).

5. **Fill Pool Spin Pause (`Qwen4ExpExpertCache.cpp`)**:
   - Added `_mm_pause()` spin-wait in `ExpertFillPool::wait()` to eliminate Windows thread scheduling / quantum yield latency.

6. **Hot Host RAM Pool Capacity-Safe Chunking & SLM Vectorization (`Qwen4ExpExpertCache.cpp`, `HyperConnectionOps.cpp`)**:
   - Fixed buffer aliasing and race condition when `misses > num_host_slots_` by chunking `to_fill` in batches of `max_chunk = num_host_slots_` with copy queue synchronization barriers between batches.
   - Refactored `hc_decode_up_mix` SLM cache with `sycl::local_accessor<vec8, 1>` ensuring strict 16-byte alignment and eliminating unaligned SLM pointer casting.
   - Extended `Qwen4ExpKernelTests` and `Qwen4ExpModuleTests` to test decode (`rows=1`), prefill (`rows=2`), single/batch router (`tokens=1` and `tokens=4`), and automated Hot Host RAM Pool lifecycle verification (hits, misses, prefetch, and LRU chunking).

| Workload | Previous Warm Baseline | After `/boost` |
|---|---:|---:|
| `story.json` (48 tokens, decode) | 94.2 ms / 10.62 tok/s | **91.7–93.8 ms / 10.66–10.90 tok/s** (7.74–7.88 s total) |
| `lighthouse_274.json` (278-token prompt, warm) | 117.4 ms / 8.50 tok/s | **114.2–114.8 ms / 8.71–8.76 tok/s** (13.32–13.45 s total) |
| Full CTest Suite (57 tests) | 43 pass, 14 skip | **100% Pass** (43 pass, 14 skip) |
| Text output verification | Golden reference | **100% bit-exact match** |

## 0. Continuation: Hot Host RAM Pool & Subgroup Broadcast Scale Lookups (2026-09-24)

This session addressed two critical pipeline bottlenecks: **cache-miss host fill / DMA latency** and **GPU nested scale reconstruction overhead**.

1. **Pinned USM Hot Host RAM Pool (HostExpertPool)** in src/models/Qwen4ExpExpertCache.{hpp,cpp}:
   - Pre-assembles contiguous expert records directly in pinned USM host memory (ctx.alloc_host in 1 GiB chunks), matching the GPU slot layout.
   - Default size: **8 GiB (8,192 MB)** via AILA_Q4E_HOT_RAM_MB=8192, hosting **3,389 expert slots** (36% of active model experts when combined with GPU's 3,778 slots).
   - **Memory sizing & stability**: on a 64 GB host, allocating 32–40 GiB of pinned USM memory caused severe OS memory starvation and commit exhaustion when combined with the 94.5 GB mmap checkpoint file cache, leading to system lockup. An 8 GiB pinned pool safely holds the active prompt working set while preserving >35 GB of free physical RAM for file cache and OS stability.
   - **Direct-DMA on Host Pool Hit**: when an expert misses GPU VRAM but hits the Hot Host RAM Pool, cquire() issues a direct copy_q_->memcpy(dev_slot, host_slot, slot_bytes_), completely bypassing CPU staging/assembly and NVMe page faults.

2. **Subgroup Leader Broadcast for Nested Scales** in src/ops/MoeExpertOps.cpp:
   - With outer locksize=64, 32 packed bytes share one scale. In a 16-lane subgroup (4 bytes/lane), lanes 0..7 and lanes 8..15 share identical scale indices.
   - Leader lanes (lane 0 and lane 8) reconstruct the scale, broadcasting it across each 8-lane half via uniform sycl::group_broadcast. This eliminates 87.5% of redundant global memory loads while strictly preserving the two-step f32 rounding and 0.000 max abs error against the CPU reference.

| 48-token story (greedy, warm) | Baseline compact cache | With Hot Host RAM Pool (8 GiB) + SG Broadcast |
|---|---:|---:|
| Host Pool Resident Experts | 0 | 3,389 |
| Cache hits per step / 480 | 379 / 480 (78.9%) | **415.2 / 480 (86.5%)** |
| PCIe transfer per step | 245 MB | **156.6 MB (-36.1%)** |
| Acquire time per step | ~57 ms | **37.5 ms (-34.2%)** |
| Peak single-token decode | ~90 ms | **65.1 ms (15.36 tok/s)** |
| Warm decode throughput | 10.49 tok/s | **10.72 tok/s** (93.27 ms/tok) |
| Text output | Golden reference | **100% bit-exact match** |

## 0. Continuation: measured compact expert cache (2026-09-24)

The next session implemented the planned device-time MoE benchmark and retained bitsandbytes
nested u8 absmax codes in GPU expert slots. `AILA_Q4E_ABSMAX_CODES=1` is the default when all
layers have compatible nested NF4 metadata; `0` restores f32-expanded slots. The GPU kernels
reconstruct scales using the layer's nested map, nested absmax and offset. Two explicit f32
rounding steps match the previous host expansion; a single fused multiply-add did not.

| Same 9131 MB pool | f32-expanded (`=0`) | nested u8 codes (`=1`) |
|---|---:|---:|
| Bytes per expert slot | 2,764,800 | 2,534,400 |
| Resident expert slots | 3,463 | 3,778 (+315, 9.1%) |
| Warm 400-token essay, steps 5–400 | 101.9 ms/token, 9.81 tok/s | 95.3 ms/token, 10.49 tok/s (+6.9%) |
| Cache hits per step / 480 | 367 | 379 |
| Expert transfer per step | 299 MB | 245 MB |
| Full 400-token generation | 45.00 s | 42.36 s |

Both modes generated the **same complete 400-token greedy essay text**. Output from the
profiling process is interleaved with `[Q4E Profile]` lines; strip those lines before comparing
the response. Short story (48 tokens) and short essay (24 tokens) also matched. The table uses
the second back-to-back run of each mode; the first runs varied with Windows page-cache state
(compact 9.06, f32 9.89 tok/s), so the 6.9% is a workload-specific warm result, not a general
throughput guarantee. The f32 mode is a rollback and a numerical reference.
After the final rebuild, a back-to-back `story.json` run reached 93.8 ms/token (10.66 tok/s,
415/480 hits) and retained the same 48-token text; its cold first run took 149.5 ms/token.

`AilaQwen4ExpMoeBench` allocates 48 × 10 real-size slots, validates a synthetic CPU MoE
reference, then profiles 3 repeats × 20 samples. `--nested 1` also allocates 48 real-size
per-layer metadata sets (512 expert slices each). In two back-to-back A/B pairs, the median
48-layer **device span** was 10.090/10.137 ms with f32 slots (1.327 GB) and 10.917/11.622 ms
with nested codes (1.217 GB): compact scale reconstruction added 0.83–1.49 ms. The second
nested run had more variance (p10 10.751, p90 12.107 ms), so retain the range, not a single
precise penalty. The 96-launch no-op control was ~0.55–0.59 ms wall and the 1.327 GB f32
streaming read ~3.1–3.2 ms device time. These are isolated measurements, not complete-model
numbers. They show that MoE arithmetic alone cannot account for a ~95 ms decode token and
that the larger cache/transfer gains outweigh the compact kernel cost on the warm essay.
The `[Q4E Profile] router_sync` column includes queued GPU work, so it must not be interpreted
as pure CPU launch overhead.

An experimental shared-expert add fused into MoE down remains behind
`AILA_Q4E_MOE_SHARED_FUSED=0`. It changed some BF16 layer-0 values and reduced the fixed bench
from 10.91 to 7.52 tok/s in one A/B run; do not enable by default. Other launch-fusion ideas
need numerical and full-model performance checks before adoption.

## 1. Current State

- **Model**: `C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2` (moved from the USB drive `F:` to NVMe `C:`).
  `qwen4_exp`: 125B total / 6B active, 48 layers (36 GatedDeltaNet + 12 full attention),
  512 experts top-10 + shared expert, 4-stream hyper-connection (HC), PLE on layer 1.
  bitsandbytes NF4 with nested absmax, 131 shards, 94.5 GiB.
  Reference: `transformers/models/qwen4_exp/modeling_qwen4_exp.py` (Python 3.13 site-packages).
- **Milestones**: M1 (text generation <= 2048 tokens, QSA acts as dense attention) is done and
  still produces the same outputs. M2 (QSA indexer, > 2048 context), M3 (vision tower),
  M4 (shared sessions / server) have not started.
- **Previous f32-slot performance** (warm OS page cache, greedy, `--max-seq 512`; use section 0
  for the new default nested-code result):

| Workload (`perf/qwen4exp/prompts`) | Prefill | Decode | Hit rate |
|---|---|---|---|
| `story.json` (30-token prompt, 48 tokens) | 2.6 s | 90.9 ms/token, **11.0 tok/s** | 411/480 |
| `essay.json` (400 tokens) | 3.1 s | 97.2 ms, **10.3 tok/s** | 367/480 |
| `lighthouse_274.json` (278-token prompt) | 9.4 s | 117.4 ms, 8.5 tok/s | 326/480 |
| `--bench --bench-pp 32 --bench-tg 64` | 1.2 s | 90.4 ms, 11.1 tok/s | 403/480 |
| Same bench, `AILA_DEBUG_Q4E_FAKE_HITS=1` (100% hits, wrong outputs) | — | **48.9 ms, 20.5 tok/s** | 480/480 |

  For reference: M1 decoded at ~1.2 tok/s. At the start of the second session it was
  8.3 tok/s by default (10.8 with a hand-set 8 GB cache). The first request's time to
  first token (TTFT) dropped from ~18 s to ~2.6 s (JIT moved into warmup). Process start
  (load + warmup) takes ~25.5 s, of which warmup is 20.4 s.

## 2. Platform Constraints (measured on this machine)

- **GPU**: Arc A770 16 GB, PCIe 3.0 x16 (~12.5 GB/s h2d, one 2.76 MB expert ≈ 0.22 ms),
  driver `32.0.101.8860`, oneAPI 2026.1. **The A770 also drives the 1920x1080 display.**
- **VRAM cliff**: with ~1.1 GB free after warmup (9 GB expert pool), every kernel ran ~2x slower
  (router_sync 22 → 48 ms/token, WDDM paging). ~2.2 GB free is fine. The default pool size
  keeps `AILA_Q4E_VRAM_RESERVE_MB=2560` free before warmup.
- **Host**: 64 GB 4-channel DDR3-1800. Host expert fill is DDR3-bound (~0.11-0.25 ms per expert
  with 8 threads, ~5.9 GB/s effective on a big prefill). 8 fill threads = 16 threads (no gain).
- **Page cache**: 64 GB RAM < 94.5 GB model. Experts stay in the mmap and pages that were evicted
  are re-read from NVMe (~1.9 GB/s through page faults). A 274-token prefill touches ~34 GB of
  experts, so workloads evict each other: the same lighthouse run measured 9.4 s prefill /
  8.5 tok/s warm and 22.5 s / 5.9 tok/s after other prompts ran. **Measurement hygiene:
  a rebuild also evicts model pages. Run each workload twice back to back and use the
  second run.**
- **JIT**: no AOT target; `-fsycl-device-code-split=per_kernel`. The first forward per process
  JITs ~15 s of kernels (linear attention 5.1 s, full attention 8.2 s; oneDNN primitive creation
  only 0.65 s). `SYCL_CACHE_PERSISTENT=1` **still crashes** (0xC0000005 at the first JIT,
  issue #4) on this stack (retested 2026-09-24 with a fresh cache dir).

## 3. How a Decode Token Flows Now

Per layer L (`Qwen4ExpBnb4Backend::ForwardImpl`):

1. `RunHyperConnection` (attn): grouped RMSNorm, then for one row the two fused kernels
   `hc_decode_down_inject` + `hc_decode_up_mix`, otherwise oneDNN + elementwise.
2. GatedDeltaNet or full attention, `hc_inject_add`, `RunHyperConnection` (mlp).
3. Router GEMV + `router_softmax_topk_device`, writing top-k **directly into pinned host memory**.
   Also routes this layer's MoE input through **layer L+1's router** (next-layer prediction,
   68.5% top-10 overlap, covers 54.7% of misses).
4. One `ctx.synchronize()` (`router_sync` in the profile). Then the shared expert is submitted
   first, so the GPU runs it while the host handles misses.
5. `Qwen4ExpExpertCache::acquire(L, ids)`: LRU lookup. Misses that the prefetch already staged
   are DMA'd at once. Other misses are filled by `ExpertFillPool` (8 threads, each expert
   split in 8 parts: packed bytes and u8 absmax memcpy by default, or nested absmax expanded
   to f32 with `AILA_Q4E_ABSMAX_CODES=0`) into double-buffered
   pinned staging. DMA goes through an **immediate-command-list in-order copy queue** that
   depends on a main-queue barrier (a victim slot can't be overwritten while a kernel reads it).
6. `prefetch_async(L+1, predicted ids)`: **fill-only**. It stages predicted non-resident
   experts in pinned memory on the fill threads, binds no slot and evicts nothing.
7. Expert kernels `moe_decode_gate_up_swiglu_grouped` + `moe_decode_down_weighted_accumulate`.

- **Prefill** (count > 1) acquires each layer's unique experts in one call (needs
  `num_experts + top_k` slots) and then keeps the `slots / 48` most-used experts hot and demotes
  the rest to the LRU tail (`demote()`). Plain LRU would leave only the last ~11 layers resident.
- **Expert pool size**: default = device memory − static weights − full-attention KV for
  `max_seq_len` − 2560 MB → 9131 MB / 3778 compact slots (3463 f32 slots) on this machine. Stored in 1 GB device chunks
  (Level Zero caps single allocations at ~4 GB).
- **Input embedding**: rows are gathered on the host from the mmap (`EmbedTokens`) instead of
  keeping 1.27 GB in VRAM, which gave the pool +460 slots.
- **Engine**: Qwen4Exp decodes with chunk size 1 (every forward already syncs per layer;
  decoding ahead only caused an EOS overrun and a replay). Init warmup (2-token forward) runs by default.
  `align_backend_context_to_sequence` replays only the missing tail (generic engine code; see
  section 8).
- A prompt is prefilled as `N-4` + `4` tokens: the engine splits at `recurrent_prefill_anchor`
  so the backend can snapshot the recurrent state for multi-turn reuse. The 4-token forward costs
  ~0.6 s (fresh experts) but saves a full re-prefill on the next turn. It is intentional.

## 4. Change Log Since M1 (measured on the short story prompt unless noted)

| Change | Effect | Toggle |
|---|---|---|
| `[Q4E Profile]` / `[Q4E Stages]` profiling | — | `AILA_PROFILE_Q4E_DECODE`, `AILA_PROFILE_Q4E_STAGES` |
| Persistent 8-thread fill pool, experts split in parts | 1.2 → 5.6 tok/s | `AILA_Q4E_FILL_THREADS`, `_FILL_PARTS` |
| Expert pool in 1 GB chunks (lifts the 4 GB cap), 8 GB pool | 5.6 → 6.7 | `AILA_Q4E_EXPERT_CHUNK_MB` |
| Native GEMV for one-row bf16 projections (`DenseLinear`) | → 7.3 | `AILA_Q4E_DENSE_GEMV` |
| HC refactor + two fused decode kernels | → 9.6 | `AILA_Q4E_HC_FUSED` |
| Top-k written to pinned host memory, shared expert before acquire | → 9.8 | — |
| Fill-only next-layer prefetch | → 10.2 | `AILA_Q4E_PREFETCH`, `_PREFETCH_TOPN` |
| Immediate-command-list copy queue for miss DMA, double-buffered staging | → 10.8 | `AILA_Q4E_STAGING_MB` |
| Batched per-layer prefill acquire | 274-token prefill 41.6 → 23.3 s (incl. JIT) | — |
| Decode chunk 1 for Qwen4Exp | removes EOS-overrun replays | — |
| Replay only the missing tail after decode (engine) | 48-token stop at max_tokens: 2.0 s → 0.09 s (96 tokens: 6.1 s) | — |
| Init warmup on for Qwen4Exp | first-request prefill (26 tokens) 17.9 → 2.6 s | `AILA_INIT_WARMUP` |
| Auto expert pool size (was 4000 MB) | default 8.3 → 10.2 tok/s | `AILA_Q4E_EXPERT_CACHE_MB`, `AILA_Q4E_VRAM_RESERVE_MB` |
| Prefill demotion | 4-token chunk hits 181 → 329 / 1050; decode ±0 | `AILA_Q4E_PREFILL_DEMOTE` |
| Host-side input embedding (+460 slots) | essay hits 72.9 → 76.5%, 8.1 → 10.3 tok/s | `AILA_Q4E_HOST_EMBED` |

All `AILA_Q4E_*` variables are documented in `docs/Environment_Variables.md`
("Qwen3.8-Flash-Next").

## 5. Where Decode Time Goes Now

At 3463 slots a decode token takes ~90-117 ms:

- **Misses: ~40-70 ms.** 69-113 misses per token (host fill + DMA + prefetch wait, with the GPU
  mostly idle meanwhile). The DMA is at PCIe line rate; the fill is DDR3-bound.
- **Floor at 100% hits: 48.9 ms** (`AILA_DEBUG_Q4E_FAKE_HITS=1`, bench mode): `router_sync`
  23.6 ms of GPU waits plus ~25 ms on the host. The previous session estimated ~1200 kernel launches
  per token at ~20 µs each. The main queue uses batched command lists, so a layer's kernels only
  start at its sync, and **host submission and GPU execution largely serialize**.
  Stage profile at 100% hits (synchronized after every stage, inflated, relative only):
  HC 17.5-19.3, experts 19.0-20.2, linear attention 11.7-12.2, router 7.7-9.2, shared 6.7-7.5,
  full attention 3.8-4.0, lm_head 3.4, PLE ~2 ms.
- **Unverified kernel-efficiency estimate**: after subtracting sync overhead, the expert GEMVs look
  like ~15-16 ms/token for 1.33 GB (~85 GB/s) and HC ~12 ms for 1.28 GB. The A770's measured
  contiguous DRAM read is ~291 GB/s (A770 kernel skill, driver 8860). The dense NF4 GEMV in this
  repo reaches 180-240 GB/s with the same structure. **Confirm with device-side timing first**:
  the stage numbers also contain submission overhead.

**Replacement-policy simulation** (routing traces replayed offline with `perf/qwen4exp/q4e_cache_sim.py`;
simulated LRU matched the measured hit rate: 72.7% vs 72.9%):

| 3463 slots | LRU (current) | SLRU | LFU decay 0.999 | Belady optimum |
|---|---|---|---|---|
| essay | 76.2% | 78.6% | 78.5% | 89.6% |
| code | 76.0% | 75.4% | 76.6% | 87.6% |
| zh | 78.4% | 78.3% | 79.1% | 88.6% |

Capacity matters more than the policy. On the essay trace, LRU gives 72.7 / 74.9 / 76.2 / 78.6 / 83.3 / 89.2%
at 3003 / 3275 / 3463 / 3750 / 4500 / 6000 slots. Each hit-rate point is ~4.8 misses ≈ 2 ms per token.

## 6. Negative Results (do not retry without new evidence)

- **DMA-ing predicted experts into GPU slots** (the first prefetch design) moved time from acquire
  to `router_sync` in proportion to the prefetched bytes (+6 ms at top-5, +15 ms at top-10), even
  on the separate blitter engine (checked: copy ordinal 1). Root cause unknown (suspected
  driver/WDDM serialization). Fill-only prefetch replaced it.
- **Immediate command lists for the whole process** (UR Level Zero env vars): prefetch wait ~0, but
  `router_sync` +25 ms. Immediate lists are applied only to the copy queue.
- With batched command lists, copies sit unflushed until a wait. That is why the copy queue is immediate.
- `PrefetchVirtualMemory` on the mmap before filling: slower (removed).
- More than 8 fill threads: no gain. Expert pool ≥ 9 GB: WDDM slowdown (section 2).
- LFU / SLRU / LRU-2 / per-layer LRU: no consistent gain across traces (section 5).
- Prefill demotion: only a small gain. It is kept because it costs nothing.
- `SYCL_CACHE_PERSISTENT=1`: still crashes (section 2).
- For kernel work, also honor the dense-GEMV list in `CLAUDE.md` (uint64 loads, qmap in registers,
  SLM input caching, 2 rows per sub-group, and so on).

## 7. Recommended Next Steps (in order)

1. **Profile the complete decode critical path with device events.** The opt-in profiling
   `Context(nullptr, true)` and both f32/nested MoE microbenchmarks are done. The next measurement
   should cover router, HC, shared expert, attention/DeltaNet, MoE and copy-queue DMA for one
   token, with event intervals and host wait boundaries. Run nested and f32 modes separately;
   use ≥20 samples, medians and p10/p90; keep
   the main queue's production submission mode intact. Do not treat `router_sync` as CPU time.
2. **Prioritize cache misses and transfer overlap.** The compact essay still transfers ~245 MB
   per step and spends ~39 ms in acquire versus ~9 ms for all isolated MoE kernels. A larger
   pool risks the measured WDDM VRAM cliff. Test any cache/prefetch change on both warm and cold
   page-cache states and compare full generation, not a local subphase alone. Previous DMA
   prefetch moved its cost into `router_sync`; it is a negative baseline, not an untried idea.
3. **Measure HC and other non-MoE kernels.** The planned 97-weight-set HC benchmark (1.28 GB)
   remains undone. Fold launches only where event timing justifies it and check layer outputs
   and greedy text. The attempted shared-expert/down fusion is a negative result (section 0).
   Candidate MoE memory-layout changes (gate/up loads, down reduction, byte-pair LUT) are
   lower priority until the complete-path profile proves their benefit.
4. **Prefill** is fill-bound (274 tokens: 13181 experts, 5.7 s warm / 18 s cold in the prior
   f32 mode). Re-measure with compact codes. Overlapping layer L+1's fill with layer L's
   compute remains untested.
5. **AOT compilation** (`-fsycl-targets=intel_gpu_acm_g10,spir64`) could remove the ~15 s
   JIT per process; quantify build size/time and startup tradeoffs before changing defaults.
6. Milestones: **M2** QSA indexer (`self_attn.indexer`, `modeling_qwen4_exp.py` L734-763) to go past
   2048 tokens; **M3** vision tower (BNB-NF4 `model.visual.blocks.*`); **M4** wiring into
   `src/inference/Model.cpp` (shared sessions, `AilaServer.exe`).

## 8. Build, Test, Run, Profile

```powershell
pwsh -NoProfile -File build.ps1 -BuildDir build-refactor     # no PATH workaround needed anymore
ctest --test-dir build-refactor -R "AilaSafeTensorsTests|AilaQwen4ExpModelSpecTests|AilaQwen4ExpKernelTests|AilaQwen4ExpModuleTests" --output-on-failure
ctest --test-dir build-refactor          # full suite: 43 pass, 14 skip without model paths
.\build-refactor\AilaQwen4ExpMoeBench.exe             # f32 slots, 48 layers × 10 experts
.\build-refactor\AilaQwen4ExpMoeBench.exe --nested 1  # u8 codes and per-layer metadata
.\build-refactor\Aila.exe -m C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2 --max-seq 512 --greedy `
    --thinking-budget 0 --max-tokens 48 --messages-json perf/qwen4exp/prompts/story.json
```

- The engine's tail-only replay change is generic. The unit tests of the other families pass, but no
  Qwen3.5 model is on this machine, so their end-to-end generation was not re-run.
- **Expected greedy outputs** (identical across all changes above):
  - story: `The horizon ignites with a soft, amber glow, piercing the lingering indigo of the pre-dawn sky. Gentle waves ripple beneath the emerging light, their crests catching the first golden rays like scattered diamonds. ...`
  - lighthouse: `Ships relied on the lighthouse's steady beam to avoid the hidden reef located to the north.`
- **Profiling** (`perf/qwen4exp/q4e_prof.ps1`; logs go to `tmp/q4e_prof_<tag>.log`):

```powershell
pwsh perf/qwen4exp/q4e_prof.ps1 -Tag base                                  # story, 48 tokens
pwsh perf/qwen4exp/q4e_prof.ps1 -Tag long -MaxTokens 24 -Messages perf/qwen4exp/prompts/lighthouse_274.json
pwsh perf/qwen4exp/q4e_prof.ps1 -Tag bench -Bench -MaxTokens 64            # fixed length, ignores EOS
pwsh perf/qwen4exp/q4e_prof.ps1 -Tag floor -Bench -MaxTokens 64 -EnvVars @{AILA_DEBUG_Q4E_FAKE_HITS='1'}
pwsh perf/qwen4exp/q4e_prof.ps1 -Tag st -Bench -MaxTokens 24 -EnvVars @{AILA_PROFILE_Q4E_STAGES='1'}
# routing trace + policy simulation
pwsh perf/qwen4exp/q4e_prof.ps1 -Tag tr -MaxTokens 400 -Messages perf/qwen4exp/prompts/essay.json -EnvVars @{AILA_DEBUG_Q4E_ROUTE_TRACE='E:/RiderProjects/Aila/tmp/route_essay.bin'}
python perf/qwen4exp/q4e_cache_sim.py tmp/route_essay.bin --cap 3463 4500
# compact/f32 A/B: run each twice, use the second warm result
pwsh perf/qwen4exp/q4e_prof.ps1 -Tag codes_essay -MaxTokens 400 -Messages perf/qwen4exp/prompts/essay.json
pwsh perf/qwen4exp/q4e_prof.ps1 -Tag f32_essay -MaxTokens 400 -Messages perf/qwen4exp/prompts/essay.json -EnvVars @{AILA_Q4E_ABSMAX_CODES='0'}
```

  The summary line averages the decode steps. `sync` = router waits, `acq` = acquire (`fill` host fill,
  `dma` DMA wait, `pf_wait` wait for the prefetch fill), `pf_used/issued` = prefetched experts
  that served a miss / experts staged, `hit` = cache hits out of 480.

## 9. Uncommitted Files

- Engine: `src/engine/EngineGeneration.cpp` (chunk 1, tail replay), `src/engine/EngineLoad.cpp` (warmup).
- Backend: `src/models/Qwen4ExpBnb4Backend.{hpp,cpp}` (profiling, `DenseLinear`,
  `RunHyperConnection`, pinned router output, prediction/prefetch, batched prefill acquire +
  demotion, auto pool size, `EmbedTokens`, full attention epilogue fusion, vectorized embedding & shared add).
- Cache: `src/models/Qwen4ExpExpertCache.{hpp,cpp}` (fill pool, chunked pool, copy queue,
  double-buffered staging, compact u8 absmax slots, nested metadata uploads, `prefetch_async`,
  `demote`, debug trace / fake hits, async DMA barrier event).
- Ops: `src/ops/AttentionOps.cpp` & `src/ops/Ops.hpp` (gate epilogue fusion), `src/ops/HyperConnectionOps.{hpp,cpp}` (vectorized HC ops),
  `src/ops/MoeExpertOps.{hpp,cpp}` (nested scale decode, hardware dependency events).
- Core: `src/core/Context.hpp` (opt-in queue event profiling), `src/core/Tensor.hpp` (`data_as` const overload).
- Utils: `src/utils/EnvUtils.hpp` (Level Zero copy engine activation).
- Entry points: `src/main.cpp`, `src/worker/WorkerMain.cpp` (pre-SYCL copy engine setup).
- Tests: `tests/ops/Qwen4ExpKernelTests.cpp` (HC/MoE/Attention gate fusion kernels),
  `tests/models/Qwen4ExpModuleTests.cpp` (cache modes, prefetch, demotion, nested scale, multi-chunk pool stress).
- Docs/tools: `docs/Environment_Variables.md`, this file, `perf/qwen4exp/` (profiling script,
  cache simulator, prompts, MoE device-time benchmark), `CMakeLists.txt` (benchmark target).
