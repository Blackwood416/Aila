# Qwen3.8-Flash-Next (`qwen4_exp`) on Intel Arc A770 — Architecture & Handoff

> **Status:** Post-Round-6 Architecture Reference.
> `docs/plans/` intermediate logs and the legacy `Qwen4ExpExpertCache` / `Qwen4ExpLayerStage` / `Qwen4ExpLayerMajor` / `L0HostImport` modules have been retired.
> Environment variable reference: [`docs/Environment_Variables.md`](Environment_Variables.md).

---

## 1. Hardware & Model Overview

| Component | Value |
|---|---|
| GPU | Intel Arc A770 16 GB (DG2 / Alchemist, ACM-G10, 15,933 MB visible VRAM) |
| Interconnect | PCIe 3.0 x16 (~15.75 GB/s theoretical, ~9.2–9.8 GB/s sustained DMA) |
| CPU | Intel Xeon E5-2696 v3 (18C / 18T, AVX2) |
| Host RAM | 64 GiB quad-channel DDR3-1866 (~45–52 GB/s sustained) |
| Toolchain | Intel oneAPI 2026.1 (`icx-cl`, oneDNN 3.11, oneTBB, SYCL / Level Zero 1.33.1), MSVC |
| Checkpoint | `Qwen3.8-Flash-Next-BNB-NF4-v2` (bitsandbytes NF4 with nested block-256 absmax) |

Model geometry (`qwen4_exp`):

| Property | Value |
|---|---|
| Layers | 48 (36 GDN Linear Attention + 12 QSA Sparse Full Attention at `(i+1) % 4 == 0`) |
| Hidden / Residual | `hidden_size = 2560`, 4-stream Hyper-Connections (`10240` total stream width, low-rank `320`) |
| GDN Linear Attention | 16 Q/K heads, 48 V heads, `head_dim = 128`, causal `conv1d` kernel=4, chunk size `T = 16` |
| QSA Full Attention | 24 Q heads, 2 KV heads (`GQA = 12`), `head_dim = 256`, indexer 4 Q / 1 KV head (`dim = 128`, ratio=4, `block_topk = 512`) |
| MoE | 512 routed experts per layer (`top_k = 8`, `moe_intermediate_size = 640`) + 1 shared expert per layer |
| Per-Layer Embedding | PLE injected at Layer 1 (16 n-gram heads, 128 shards, dilated conv1d) |

---

## 2. Memory & Expert Residency Architecture

Because 48 layers × 512 experts × 2.5344 MB = **62.28 GB** of expert weights cannot fit in 16 GB VRAM, `Qwen4ExpBnb4Backend` partitions all 24,576 experts across two complementary tiers with zero checkpoint mmap fallback at runtime:

### 2.1 `Qwen4ExpAdaptiveCache` (GPU Resident Tier)
- **3,360 GPU resident expert slots (~8.12 GB payload)** allocated in bounded device chunks, seeded at load from the 48×72 multi-domain hotset table (`Qwen4ExpHotExperts.hpp`).
- Sized automatically at model load from available VRAM after subtracting static weights (4,320 MB), session KV/recurrent state (445 MB at `max_seq = 12288`), and headroom reserve (`AILA_Q4E_VRAM_RESERVE_MB = 1400`, `AILA_Q4E_MAX_GPU_SLOTS = 3360`), maintaining ~1.3–1.4 GB free physical VRAM at peak 8K prefill.
- Each slot stores compact packed NF4 `gate_up` (1,638,400 B) + `gate_up` u8 absmax codes (51,200 B) + `down` packed (819,200 B) + `down` u8 absmax codes (25,600 B) = **2,534,400 bytes**.

### 2.2 `Qwen4ExpHostArena` (Large-Page Host Complement)
- **21,216 cold experts (50.11 GiB)** held in Windows large-page (`MEM_LARGE_PAGES`) host memory, storing the exact complement of the 3,360 GPU-resident experts plus per-layer nested scale metadata (`LayerNestedMetadata`).
- Together, `3,360 + 21,216 = 24,576` experts are 100% memory-resident after load.

### 2.3 Direct-Arena DMA Staging via `copy_q_`
- Cold experts routed during prefill are streamed directly from `Qwen4ExpHostArena` into a **2-bank × 16-slot** GPU staging ring over a dedicated in-order `copy_q_`.
- **Adjacent-slot copy coalescing**: Consecutive cold experts in the arena are merged into single contiguous `memcpy` transfers (~13–20 MB each).
- **Decoupled host submission & dequant release fence**: `StageColdExpertsAsync` does not stall the host submit thread on `staging_copy_` under `direct_arena`, and each staging bank is released as soon as `submit_dequant` completes (`AILA_Q4E_STAGE_DEQUANT_FENCE=1`) rather than waiting for the end of the chunk's GEMM/scatter pipeline.

---

## 3. Prefill Architecture & Round 6 Scoreboard

### 3.1 Measured Scoreboard (`tools/prefill-bench.ps1 -Chunk 8192`, `max-seq 12288`)

| Prompt Tokens | Round 4 | Round 5 | **Round 6 Wall** | **ms/token** | **Round 6 Tok/s** |
|---:|---:|---:|---:|---:|---:|
| **2,007** | 11,217 ms (178.9 tok/s) | 10,338.0 ms (194.1 tok/s) | **10,177.8 ms** | 5.071 | **197.2 tok/s** |
| **4,895** | 15,318 ms (319.6 tok/s) | 11,894.0 ms (411.6 tok/s) | **11,701.2 ms** | 2.390 | **418.3 tok/s** |
| **7,764** | 20,467 ms (379.3 tok/s) | 15,584.3 ms (498.2 tok/s) | **15,071.3 ms** | 1.941 | **515.1 tok/s** |

- **Linear fit**: `8,472 ms` fixed cost + **`0.85 ms/token` marginal cost (`1,176 tok/s` marginal throughput)**.
- **Decode throughput (story 30p+48g / 20g)**: E2E 48-token generation improved from `11,432 ms` to **`8,752 ms`** (-23.4%); pure single-stream decode latency improved from `~207.5 ms/tok` (`4.82 tok/s`) to **`~155.3 ms/tok` (`6.44 tok/s`, +33.6%)** with `fill_ms` reduced from `~90 ms/tok` to `0.02 ms/tok` via direct-arena DMA staging, fused MoE front (`bf16_moe_front_fused`), deferred MLP residual (`AILA_Q4E_DECODE_PENDING_MLP_RESIDUAL=1`), and fused GDN conv history commit (`AILA_Q4E_GDN_HISTORY_FUSED=1`).

### 3.2 Key Prefill Optimizations
1. **Dense NF4 Projections via Vectorized Dequant + oneDNN (`AILA_BNB4_ONEDNN_MIN_M = 128`)**: Routes all `seq_len >= 128` dense NF4 projections (including the 97 `N = 2560` calls: 36 `linear_o_proj`, 12 QSA `o_proj`, 48 `shared_down_proj`, 1 PLE `value_proj`) to word-vectorized `dequantize_weight` + oneDNN `ab` MatMul (16.0× faster than the custom wide joint-matrix kernel on `N = 2560`, saving ~2.95 s at 8K).
2. **4-Head Block-Batched QSA Prefill Attention (`AILA_QSA_HEADS_PER_WG = 25`)**: 64-thread work-groups (four 16-lane sub-groups) process 4 Q heads per work-group with `Q` cached in registers, contiguous `vec<bf16, 4>` K/V/gate loads, and double-buffered parity SLM partials that require only **1 work-group barrier per 4-token compressed block (0.25 barriers/token)** — 2.22× faster on sparse tiles. Paired with 8-block sub-group indexer scoring (`AILA_QSA_FAST_BLOCK_SCORES = 1`) and dense-tile score skipping (`AILA_QSA_SKIP_DENSE_SCORES = 1`).
3. **Fused GDN Chunk Scan (`AILA_Q4E_GDN_CHUNK_MODE = 1`, `ops::chunk_scan`)**: Single 48-work-group launch per GDN layer walking all `T = 16` chunks in order with `q/k` staged in SLM, `kPrefetch = 8` state columns, and `kUpdateBlock = 4`.
4. **Expert-XMX Batched GEMM & Single Pinned Plan Upload (`AILA_Q4E_PREFILL_EXPERT_XMX = 1`, `MIN_M = 192`)**:
   - `RunAdaptivePrefill` buckets `(token, rank)` by expert in one `O(count * top_k)` pass per layer (`AILA_Q4E_XMX_BUCKET_ROUTE = 1`).
   - `RunGroupedExpertsXmx` packs `bases + tokens + ranks + slots` into disjoint per-chunk slices of pinned host USM (`xmx_plan_pinned_[0]`) and uploads each chunk's plan in a single `memcpy` (`AILA_Q4E_XMX_SINGLE_PLAN_COPY = 1`).
   - `submit_dequant` uses 128-bit `vec<uint32_t, 4>` loads (32 NF4 codes per work-item, 1 nested scale lookup per 32 weights), and `submit_gather` / `submit_scatter` use division-free 3D grids.
5. **Fused HC Sigmoid & Fused Shared Expert Accumulation**: `AILA_Q4E_HC_FUSED_SIGMOID = 1` folds sigmoid into `ops::hc_mix_reduce` (eliminating 96 launches and ~30.5 GB DRAM traffic per prefill); `AILA_Q4E_MOE_SHARED_FUSED = 1` folds the shared expert addition into vectorized `ops::moe_ordered_accumulate_down_rows`.
6. **Long Interior Checkpointing (`AILA_Q4E_LONG_INLINE_CHECKPOINT = 1`)**: Captures the reusable prefix checkpoint inline at `checkpoint_after` inside a single 7,764-token forward and skips the redundant end-of-slice snapshot (`AILA_Q4E_SKIP_REDUNDANT_END_SNAPSHOT = 1`), eliminating the `7760 + 4` tail split (-1.37 s).
7. **3.14 GB Multi-Phase Scratch VRAM Reclamation & 3× GQA-Grouped GDN `prepare_inputs`**: `buf_.phase_scratch` (1,022 MB at 8K) aliases four mutually exclusive lifetimes on the in-order queue: Layer-1 PLE intermediates (835 MB), Hyper-Connection intermediates (`hc_N`, `hc_mix_down`, `hc_mix_up` = 323 MB), GDN `linear_all + raw + prepared` (1,022 MB), and MoE `moe_down_acc + router_scores + moe_h + shared_*` (952 MB), freeing >1.6 GB of VRAM to expand resident GPU experts from 2,800 to 3,360 slots while keeping ~1.3–1.4 GB free VRAM headroom. GDN `prepare_inputs` groups 3 V heads per Q/K head (3× fewer work-groups and barrier reductions).

---

## 4. Decode & Retained Research Subsystems

The following decode and experimental subsystems are actively maintained and covered by unit/module tests:

- **Adaptive Resident Decode (`RunResidentDecode`)**: Single-token and small-window decode over `Qwen4ExpAdaptiveCache` + `Qwen4ExpHostArena`, supporting fused router + shared gate (`bf16_moe_front_fused`), pending MLP residual deferral into the next layer's HC RMSNorm, and dynamic hotset swaps.
- **Whole-Token SYCL Command-Graph Replay (`Qwen4ExpWholeTokenGraph`, `Qwen4ExpTokenGraph*`)**: Captures and replays the decode token graph with device-side GDN/PLE ring heads and uncached-host doorbell service for cold experts (`AILA_Q4E_TOKEN_GRAPH`).
- **Multi-Token Prediction (`Qwen4ExpMtpDraft`, `Qwen4ExpMtpRuntime`)**: Speculative draft head (`mtp.layers.0`) with a self-contained 64-slot GPU expert LRU cache and multi-row verification alignment (`AILA_Q4E_MTP_DRAFT`).
- **CPU Heterogeneous Cold Expert Execution (`Qwen4ExpExpertExecutor`, `Qwen4ExpCpuExpertOps`)**: Bounded TBB/AVX2 worker executing cold experts directly from `Qwen4ExpHostArena` in parallel with GPU hotset execution (`StartArenaCold*`, `RunGraphColdPacket`, `StartArenaVerifyRows`).
- **Kernel Variants (`src/ops/*Variant*`, `MoeGroupedXmx`, `Qwen4ExpDecodeFusion`)**: Retained for decode/prefill A/B microbenchmarks (`perf/qwen4exp/`).

---

## 5. Active Preset (`qwen38-a770`) & Verification Commands

`aila::ApplyQwen38A770Preset()` (`src/utils/Qwen38A770Preset.hpp`) pins:
- `AILA_Q4E_PREFILL_XMX=1`, `AILA_Q4E_DIRECT_PREFILL_IO=1`, `AILA_Q4E_PHASE_SCRATCH=1`
- `AILA_Q35_PREFILL_CHUNK=8192`, `AILA_Q4E_INLINE_CHECKPOINT=1`, `AILA_Q4E_LONG_INLINE_CHECKPOINT=1`
- `AILA_Q4E_PREFILL_EXPERT_XMX_MIN_M=192`, `AILA_Q4E_GDN_CHUNK_MODE=1`
- `AILA_Q4E_VRAM_RESERVE_MB=1400`, `AILA_Q4E_MAX_GPU_SLOTS=3360`, `AILA_Q4E_HOT_RAM_MB=32768`, `AILA_Q4E_EXPERT_READER=readfile`, `AILA_Q4E_CPU_EXPERT_DISPATCH=0`

### Unified Verification & Benchmarking (`tools/verify-q4e.ps1`)
```powershell
# Tier 1: Fast operator/kernel inner loop (~2 min)
pwsh -NoProfile -File tools/verify-q4e.ps1 -Tier fast

# Tier 2: Pre-commit CTest gate (qwen4_exp 11/11 + cpu|abi 20/20, ~6 min)
pwsh -NoProfile -File tools/verify-q4e.ps1 -Tier ctest

# Tier 3: Real-model E2E tolerance, preset interior-checkpoint parity, and 8K prefill benchmark (~5 min)
pwsh -NoProfile -File tools/verify-q4e.ps1 -Tier real

# Full 3-point Prefill Scoreboard (2K / 5K / 8K)
pwsh -NoProfile -File tools/prefill-bench.ps1 -Tokens @(2026, 4952, 7878) -Chunk 8192
```
