# Qwen3.8-Flash-Next (`qwen4_exp`) on Intel Arc A770 — Architecture & Handoff

> **Status:** Post-Round-9 Architecture Reference.
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
| Checkpoint | `D:\QwenModels\Qwen3.8-Flash-Next-BNB-NF4-v2` (bitsandbytes NF4 with nested block-256 absmax) |

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

## 3. Prefill & Decode Architecture & Round 9 Scoreboard

### 3.1 Measured Prefill Scoreboard (`tools/prefill-bench.ps1 -Chunk 8192`, `max-seq 12288`)

| Prompt Tokens | Round 6 | Round 7 / 8 | **Round 9 Wall** | **ms/token** | **Round 9 Tok/s** |
|---:|---:|---:|---:|---:|---:|
| **2,007** | 10,177.8 ms (197.2 tok/s) | 8,904.9 ms (225.4 tok/s) | **8,277.9 ms** | 4.125 | **242.5 tok/s** |
| **4,895** | 11,701.2 ms (418.3 tok/s) | 10,165.4 ms (481.5 tok/s) | **9,847.4 ms** | 2.012 | **497.1 tok/s** |
| **7,764** | 15,071.3 ms (515.1 tok/s) | 14,878.8 ms (521.8 tok/s) | **12,282.8 ms** | 1.582 | **632.1 tok/s** |

- **Linear fit**: `6,882 ms` fixed cost + **`0.70 ms/token` marginal cost (`1,437 tok/s` marginal throughput across 2K–8K)**.
- **Decode throughput (single-stream, no MTP)**:
  - **Short Decode (`story.json` 30p + 20g)**: **`43.02 ~ 44.5 ms/token` (`22.4 ~ 23.24 tok/s`)**, E2E wall time **`2,487.0 ms`** (down from `5,563.1 ms` in Round 5, `-55.3%`).
  - **Long-Horizon Decode (`tools/decode-bench.ps1`, `essay.json` 40p + 256g / 512g)**:
    - **256 Generated Tokens**: **`22.4 tok/s`** (`mean = 44.60 ms/tok`, `p50 = 43.50 ms`, `p90 = 47.25 ms`; intervals `1-64: 22.4 tok/s`, `65-256: 22.4 tok/s`, cache hit `36.4%`).
    - **512 Generated Tokens**: **`22.3 tok/s`** (`mean = 44.81 ms/tok`, `p50 = 43.82 ms`, `p90 = 46.69 ms`; intervals `1-64: 22.3 tok/s`, `65-256: 22.3 tok/s`, `257-512: 22.3 tok/s`, cache hit `36.5%`).

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
8. **Adaptive Expert-XMX Band Sizing (`GATE_ROWS = 1024`, `MIN_M = 64` for `< 4K` and `192` for `>= 4K`), Process-Wide oneDNN Primitive Sharing & Prefill Warmup**: Lowering the expert-XMX gate to 1,024 rows with a 64-row minimum band below 4,096 tokens eliminates 3× padding waste at 2K tokens (`10,191 ms -> 8,904.9 ms`, **+16.1%**), while process-wide `dnnl::matmul` primitive caches in `Bnb4BitLinear` and `Linear` plus `WarmupPrefillKernels` remove ~570 redundant primitive creations per prefill (`7,764 tokens -> 14,878.8 ms / 521.8 tok/s`).
9. **Disjoint Physical-Core Pinned CPU/GPU Decode Pipeline & Fused `bf16_moe_front_fused` Host Stage-In**: Reserves Core 0 for the GPU host submission thread, Core 1 for the PLE worker, Core 2 for `CpuWorker`, and Cores 3..17 for 15 `CpuTilePool` workers with a 20 ms lock-free `sleepers_`-gated spin window. Fuses the 5 KB `x_host_` stage-in into the idle work-group of `bf16_moe_front_fused` and carries route weights by value into the `vec4`-vectorized `moe_ordered_accumulate_cpu_host_rows` merge kernel, eliminating all per-layer Level Zero waits on `CpuWorker` and all 48 per-layer 40 B weight DMA copies.
10. **4-Row AVX2 A64 Dot Products with Software Prefetch, Inline Per-Expert `h` Quantization, Batched Zero-Memcpy Adaptive Swaps & Long-Context Decode Attention Tuning (Round 8)**:
    - `nf4_q8_dot_activation64_rows4` shares quantized activation loads across 4 rows simultaneously with L1 software prefetching, while `CpuTilePool` claims 1 tile at a time (`AILA_Q4E_CPU_POOL_CLAIM=1`) and quantizes each expert's intermediate `h` inline as soon as its 40th gate tile finishes.
    - `GatedDeltaNetOps` caches the 128-element recurrent column `s_col[128]` in registers across the two decode passes (saving 108 MB/tok of global DRAM reads), `HyperConnectionOps` hoists `a_row` into registers in `hc_decode_up_mix`, unrolls `hc_decode_down_inject_wgrow`, and routes single-token `mlp_hc` through `try_hc_decode_inject_add_and_grouped_rms_norm` with in-register `v_reg`.
    - `AttentionOps` unrolls the baseline `V` accumulation 8-way and lowers the `head_dim=256` exact tiled decode threshold from 1,024 to 256 tokens (`AILA_Q4E_DECODE_EXACT256_MIN_LEN=256`), while `ExecuteSwapsAsync` batches swaps through `staging_dev_[0]` with zero CPU `std::memcpy` (`AILA_Q4E_ADAPTIVE_SWAP_FREQ=16`, `AILA_Q4E_MAX_SWAPS=4`, `AILA_Q4E_ADAPTIVE_SWAP_MIN_POS=64`).
11. **VTune-Guided Cross-Layer Deferred CPU Join, Level Zero Batch-64 Coalescing, 2.94 GB QSA Phase-Scratch Aliasing & Subgroup-Reduced Fused HC RMSNorm (Round 9)**:
    - **Cross-Layer Deferred CPU Join (`AILA_Q4E_DECODE_DEFER_CPU_JOIN=1`) + Level Zero Batch Size 64 (`UR_L0_BATCH_SIZE=64`)**: `RunResidentDecode(L)` pre-populates `PrepareDeferredHostReadRows` and submits `ops::moe_ordered_accumulate_cpu_host_rows(L)` immediately without blocking the host thread on `JoinResidentCold`, deferring the CPU completion wait to layer `L+1` right after `router_softmax_topk_device(L+1)` is submitted. Paired with `UR_L0_BATCH_SIZE=64`, the Level Zero command list remains open across the 10–12 intervening kernels so `JoinResidentCold(L)` completes before the batch flushes to the GPU, collapsing `cpu_join_ms` from `8–15 ms` to `0.3–2.7 ms` while preserving 100% deterministic token parity.
    - **Unconditional AVX2 `_mm256_permute4x64_epi64` Byte Shuffle & Coarse `steady_clock` Polling**: VTune hotspot profiling exposed two cross-lane `_mm256_permute2x128_si256` merges per weight vector on the default `Nested256` scale mode and `3.7%` total CPU time in `Query_perf_counter` inside spin loops. Enabling `_mm256_permute4x64_epi64` across all scale modes with precomputed `_mm256_abs_epi8(activation)`, widening spin-loop clock polling from `& 63u` to `& 1023u`, and boosting pinned decode threads to `THREAD_PRIORITY_HIGHEST` eliminate both bottlenecks.
    - **2-Barrier Subgroup-Reduced Fused HC Inject + RMSNorm (`try_hc_decode_inject_add_and_grouped_rms_norm`)**: Replaces the 8-barrier full-workgroup SLM tree reduction and scalar strided loop with `vec8` loads cached in registers (`vh_reg[2]`) plus a 2-barrier `sycl::reduce_over_group` subgroup reduction that is bitwise identical to `hc_inject_add + grouped_rms_norm`.
    - **QSA Attention Intermediate Aliasing into `buf_.phase_scratch`**: Aliases `buf_.q`, `buf_.z`, `buf_.k`, `buf_.v`, `buf_.indexer_qk`, and `buf_.indexer_q` into `buf_.phase_scratch` above `qsa_base_bytes`, increasing 8K prefill scratch savings from `2.31 GB` to **`2.94 GB`** without growing `phase_scratch` (`1,022 MB`) and lifting 8K prefill from `14,878.8 ms (521.8 tok/s)` to **`12,282.8 ms (632.1 tok/s)`**.

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
- `AILA_Q4E_VRAM_RESERVE_MB=1400`, `AILA_Q4E_MAX_GPU_SLOTS=3360`, `AILA_Q4E_HOT_RAM_MB=32768`, `AILA_Q4E_EXPERT_READER=readfile`
- `AILA_Q4E_CPU_OVERLAP=1`, `AILA_Q4E_MOE_COPY_VIA_COPYQ=1`, `AILA_Q4E_STAGEIN_HOIST=1`, `AILA_Q4E_CPU_ROUTE_EARLY=1`, `AILA_Q4E_JOIN_RELEASE=1`, `AILA_Q4E_CPU_RESULT_HOST_READ=1`
- `UR_L0_BATCH_SIZE=64`, `SYCL_PI_LEVEL_ZERO_BATCH_SIZE=64`

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
