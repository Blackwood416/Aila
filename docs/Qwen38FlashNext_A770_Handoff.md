# Qwen3.8-Flash-Next (`qwen4_exp`) on Intel Arc A770 — Architecture & Handoff

> **Status:** Post-Round-12 Architecture Reference.
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
| Toolchain | Intel oneAPI 2026.2 (`icx-cl 2026.2.0`, oneDNN 3.13.2, oneTBB 2023.2, SYCL / Level Zero 1.33.1, SYCL fat PCH), MSVC |
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
- **3,456 GPU resident expert slots (~8.35 GB payload, full 48×72 hotset table)** when MTP is disabled (`3,360` when `AILA_Q4E_LM_HEAD_Q8=0`), or **3,072 GPU resident expert slots (64 slots/layer × 48 layers)** when MTP is enabled (`AILA_Q4E_MTP_DRAFT=1` or `AILA_Q4E_SERVER_MTP=1`, reserving `705 MiB` for the draft model, 64 draft expert slots, and recurrent checkpoint).
- Sized automatically at model load from available VRAM after subtracting static weights (~3,672–3,752 MB with `AILA_Q4E_LM_HEAD_Q8=1`, saving 568.4 MiB vs. BF16 `lm_head`), session KV/recurrent state (445 MB at `max_seq = 12288`), MTP reserve (`705 MiB` when active), and headroom reserve (`AILA_Q4E_VRAM_RESERVE_MB = 1400`), maintaining **~1.15–1.6 GB** free physical VRAM at peak 8K prefill (preventing WDDM eviction stalls during 8K `PhaseScratch` expansion).
- Each slot stores compact packed NF4 `gate_up` (1,638,400 B) + `gate_up` u8 absmax codes (51,200 B) + `down` packed (819,200 B) + `down` u8 absmax codes (25,600 B) = **2,534,400 bytes**.

### 2.2 `Qwen4ExpHostArena` (Large-Page Host Complement)
- **21,120 cold experts (49.89 GiB)** in standard non-MTP mode (`3,456` GPU slots), or **21,504 cold experts (50.80 GiB)** under MTP (`3,072` GPU slots), held in Windows large-page (`MEM_LARGE_PAGES`) host memory alongside per-layer nested scale metadata (`LayerNestedMetadata`).
- Together, `3,456 + 21,120 = 3,072 + 21,504 = 24,576` experts are 100% memory-resident after load.

### 2.3 Direct-Arena DMA Staging via `copy_q_`
- Cold experts routed during prefill are streamed directly from `Qwen4ExpHostArena` into a **2-bank × 16-slot** GPU staging ring over a dedicated in-order `copy_q_`.
- **Adjacent-slot copy coalescing**: Consecutive cold experts in the arena are merged into single contiguous `memcpy` transfers (~13–20 MB each).
- **Decoupled host submission & dequant release fence**: `StageColdExpertsAsync` does not stall the host submit thread on `staging_copy_` under `direct_arena`, and each staging bank is released as soon as `submit_dequant` completes (`AILA_Q4E_STAGE_DEQUANT_FENCE=1`) rather than waiting for the end of the chunk's GEMM/scatter pipeline.

---

## 3. Prefill & Decode Architecture & Round 12 Scoreboard

### 3.1 Measured Prefill Scoreboard (`tools/prefill-bench.ps1 -Chunk 8192`, `max-seq 12288`)

| Prompt Tokens | Round 6 | Round 9 | Round 11 (`9169dc0`) | **Round 12 Wall** | **ms/token** | **Round 12 Tok/s** |
|---:|---:|---:|---:|---:|---:|---:|
| **2,007** | 10,177.8 ms (197.2 tok/s) | 8,277.9 ms (242.5 tok/s) | 7,966.9 ms (251.9 tok/s) | **7,421.5 ms** | 3.698 | **270.4 tok/s** |
| **4,895** | 11,701.2 ms (418.3 tok/s) | 9,847.4 ms (497.1 tok/s) | 9,751.9 ms (502.0 tok/s) | **9,526.6 ms** | 1.946 | **513.8 tok/s** |
| **7,764** | 15,071.3 ms (515.1 tok/s) | 12,282.8 ms (632.1 tok/s) | 12,317.0 ms (630.4 tok/s) | **12,008.2 ms** | 1.547 | **646.6 tok/s** |

- **Linear fit**: **`~5,888 ms` fixed cost** (down **`-782 ms`** from Round 11's `~6,670 ms`) + **`0.81 ms/token` marginal cost (`~1,235 tok/s` marginal throughput across 2K–8K)**.
- **Decode throughput (single-stream, no MTP)**:
  - **Long-Horizon Decode (`tools/decode-bench.ps1`, `essay.json` 40p + 256g / 512g)**:
    - **256 Generated Tokens**: **`23.5–23.7 tok/s`** (`mean = 42.22–42.62 ms/tok`, **`p50 = 41.87–41.98 ms`**, `p90 = 44.60–44.93 ms`; prefill **`1,907.2 ms`**).
    - **512 Generated Tokens**: **`23.4–23.5 tok/s`** (`mean = 42.62–42.73 ms/tok`, **`p50 = 42.32 ms`**, `p90 = 45.15–45.47 ms`; prefill **`1,913.0 ms`**, intervals `1-64: 23.7 tok/s`, `65-256: 23.4 tok/s`, `257-512: 23.3 tok/s`).
- **MTP Speculative Decode (`AILA_Q4E_MTP_DRAFT=1`, `T=2` preset `3,072` slots / `15` CPU threads, oneAPI 2026.2)**:
  - **256 Generated Tokens (`essay.json`)**: **`28.8–29.2 tok/s`** (`MtpWindow 28.85 tok/s`, `p50 = 30.07 ms`, `accepted = 108/146`).
  - **512 Generated Tokens (`essay.json`)**: **`27.9 tok/s`** (`MtpWindow 27.93 tok/s`, `257–512` interval `27.2 tok/s`, `accepted = 207/303`).
  - **3-Prompt Evaluation (256g)**: `essay` **`29.15 tok/s`**, `mtp-code` **`31.14 tok/s`**, `mtp-zh` **`25.31 tok/s`**.
  - **8K Prefill with MTP Active (`7,688` tokens, `max_seq=12288`)**: **`12,290.7 ms` (`625.5 tok/s`, `-3.2%` vs. non-MTP)**.
  - **Built-in Hotset & Threading**: `Qwen4ExpHotExperts.hpp` rebuilt from 38 multi-domain prompts + `agent6k-v1` + `h4-v1` (2.41M decode observations); `CpuTilePool` uses `DefaultTilePoolThreads() = clamp(total - 3, 1, 15)` (`15` workers on 18C Xeon) with `PinCurrentThreadToNonWorkerCores()` pinning Main/PLE off worker cores; adaptive swaps run asynchronously on `copy_q_` (`RunPendingAdaptiveSwap` overlapped with `ForwardLmHead` / `draft->Step`).
  - **T=3 / T=4 MTP Optimization & Dynamic Margin Gating**:
    - `Bnb4BitLinear::ForwardDecodeRows` (`packed_nf4_decode_rows_bf16<1..4>`) and `HyperConnectionOps` (`hc_decode_down_inject_k10240_rows_impl<2..4>`, `hc_decode_up_mix_k320_rows_impl<3..4>`) reuse weights across 3- and 4-row verification windows with exact B1 bitwise parity, while `Qwen4ExpMtpRuntime::ForwardWindow` extends the overlapped `greedy_bf16` + `kvOnly(proposal[1])` + `RunPendingAdaptiveSwap` fast path to `width = 2..4` and `EngineLoad` warms all widths `2..ConfiguredWidth()`.
    - Fixed `T=3` jumped from `24.14 / 28.48 / 20.28 tok/s` (`90–96 ms`) to **`27.46 / 32.34 / 22.86 tok/s`** (`79–85 ms`, **`+13.6%`**), surpassing `T=2` (`31.14`) on `mtp-code`.
    - Zero-cost top-2 logit margin tracking inside `ops::greedy_bf16` (`GreedyResult::reserved`) enables dynamic `T=2..4` gating via `AILA_Q4E_MTP_MARGIN_TH1`, `AILA_Q4E_MTP_MARGIN_TH2`, `AILA_Q4E_MTP_VERIFY_MARGIN_TH2`, and `AILA_Q4E_MTP_VERIFY_MARGIN_TH3`:
      - Dynamic `T=2/3` (`TH1=4.0, VERIFY_TH2=2.0`): `essay 29.16`, `mtp-code 31.87` (`32.56` at `TH1=4.5`), `mtp-zh 24.54 tok/s`.
      - Dynamic `T=2/3/4` (`TH1=4.5, TH2=6.0, VERIFY_TH2=2.0`): **`mtp-code 32.76 tok/s`** (`em/win=2.802`, `count=4` windows at `35.65 tok/s` / `3.921/4`), `essay 28.28`, `mtp-zh 24.72 tok/s`.
    - Remaining bottleneck for `T=3/T=4` on low-predictability prose/Chinese: CPU cold-expert compute (`~34–43 ms` across 15 AVX2 cores) + `draft->Step` LM-head bandwidth (`~4.2 ms/step`) + partial-window 36-layer GDN replay (`~3.2 ms`), so `T=2` remains the balanced default while `AILA_Q4E_MTP_WIDTH=3/4` with margin gating excels on code/structured workloads.

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
12. **Completion-Ordered Single-Pass CPU Cold-Expert Pipeline, Q8 LM Head + Full 3,456-Slot Hotset Residency, Decode-Stride Snapshot Suppression & Division-Free GEMV/HC Unrolling (Round 10)**:
    - **Completion-Ordered Single-Pass `CpuTilePool` Dispatch (`AILA_Q4E_CPU_COMPLETION_SINGLE_PASS=1`)**: Replaces the two-pass (`gate_up` barrier + `down` barrier) `ParallelFor` in `nf4_q8_batch_forward` with a single `ParallelFor` over `count * (gate_tile_cols + down_tile_cols)`. The thread finishing an expert's 40th gate tile quantizes `h` inline and publishes the expert ID into a completion-ordered queue (`gate_countdown[slot].ready_unit`), so `down` tiles immediately drain finished experts in completion order without a mid-layer thread-pool barrier or straggler stall.
    - **Q8 Quantized LM Head (`AILA_Q4E_LM_HEAD_Q8=1`) & Full 3,456-Slot GPU Hotset (`AILA_Q4E_MAX_GPU_SLOTS=3456`)**: Quantizes the `[248320, 2560]` LM head to block-64 INT8 under `qwen38-a770`, reclaiming **568.4 MiB of static VRAM**, expanding GPU resident experts from 3,360 to the full **3,456 slots (72 experts/layer × 48 layers)**, and halving `lm_head` decode bandwidth while deferring Layer-47 `complete_deferred_cpu_join()` behind host-side `mixer_ + lm_head_` recording (`head_host_ms = 0.00 ms` on non-swap tokens).
    - **Unforced Decode-Stride Snapshot Suppression (`AILA_Q4E_DECODE_STRIDE_SNAPSHOT=0`)**: Prevents unforced single-token decode steps from firing `CaptureSnapshot` every 64 tokens (`cached_len % 64 == 0`), eliminating a `35.6 ms` 112.2 MB D2H GDN state copy stall every 64 tokens and preserving the 2-slot reusable prompt prefix checkpoints across long generations.
    - **Division-Free NF4 GEMV Scales, Redundant Input-Prefetch Elimination & 4-Stream HC Up-Mix Unrolling**: Replaces signed integer division `byte_offset / packed_bytes_per_block` with `byte_offset >> 5` on `blocksize == 64`, disables redundant per-subgroup LSC input prefetches (`AILA_BNB4_GEMV_INPUT_PREFETCH_OFF=1`), unrolls the 4-stream `hc_decode_up_mix` loop on `lowrank == 320`, and gates chunked GDN prefill on `count > 64 || !cold_epoch_enabled_` so the 16-token load-time warmup compiles `ops::try_linear_delta_prefill_chunked` while short post-warmup prompts preserve exact interior-checkpoint bitwise parity.
13. **262K-Context QSA Sparse Attention Scaling, Unified Q8/MTP Speculative Runtime & Server Preset Propagation (Round 11)**:
    - **262K-Context QSA Radix Selector & 8-Block Sub-Group Decode Scoring**: Raises the `qsa_selector_mode` radix-selector threshold from `32,768` blocks (`131,072` tokens) to **`65,536` blocks (`262,144` tokens at `compress_ratio = 4`)**, replaces the single-lane serial scan in the legacy cutoff selector with a 256-lane `sycl::exclusive_scan_over_group` stable compaction, adds the 8-blocks-per-workgroup sub-group shuffle fastpath (`AILA_QSA_FAST_BLOCK_SCORES=1`) to single-token `qsa_compute_block_scores_compressed` past 514 blocks, and retains cross-layer deferred CPU join past `2,051` tokens during block-native single-token decode (`!s_qsa_block_native || count > 1`).
    - **Unified `ForwardLmHead` Entry Point, 2-Row Q8 Head Weight-Reuse Kernel & MTP Runtime Unblocking**: Unifies target and draft LM-head dispatch through `Qwen4ExpBnb4Backend::ForwardLmHead`, adds a single-launch 2-row weight-reuse fastpath to `ops::bf16_q8_head_forward` alongside default-on 2-row NF4 verification weight reuse (`AILA_Q4E_VERIFY_LINEAR_REUSE=3`, `AILA_Q4E_MTP_HIDDEN_REUSE=1`), and removes the legacy `!lmHeadQ8Enabled_` and `gdn_chunk_mode_ == 0` blocks from `Qwen4ExpMtpRuntime` while bounding MTP session allocations by `EffectiveCapacity(maxSequenceLength)`.
    - **Sparse QSA & Fused Operators in `Qwen4ExpMtpDraft`**: Upgrades the 1-layer MTP draft head to use `target_.RunHyperConnectionCombined`, `ops::bf16_moe_front_fused`, `target_.ForwardLmHead`, and sparse QSA block selection past 2,048 tokens using `mtp.layers.0.self_attn.indexer.*`.
    - **Adaptive Cache Warmup Isolation, Usage Decay & Worker Preset Inheritance**: Guards `adaptive_cache_.ObserveRouting` on `cold_epoch_enabled_` so synthetic load-time warmup tokens cannot pollute hotset frequencies, ages `usage_` by `0.85` after each swap interval (`AILA_Q4E_ADAPTIVE_SWAP_DECAY=1`, `AILA_Q4E_ADAPTIVE_SWAP_FREQ=16`), and propagates `AILA_Q4E_ACTIVE_PRESET=qwen38-a770` into `AilaWorker.exe` (`WorkerMain.cpp` + `tools/server/start-qwen38.ps1`).
14. **Pre-Warmed Canonical Expert-XMX Band Bucketing, Per-Token PLE Page Prefetch, Zero-Readback Host Token IDs & Multi-Row Verification Fusion (Round 12)**:
    - **Canonical Expert-XMX Band Splitting & `(B, M)` Primitive Pre-Warming (`AILA_Q4E_PREFILL_EXPERT_XMX_MAX_M=256`, `moe_expert_xmx_warmup_primitives`)**: Splits hot prefill experts (`count >= 1024`) into `<= 256`-row bands rounded to `{64, 128, 192, 256}`, buckets tail chunk expert counts to `{1, 2, 4, 8, 12, 16}` in `ensure_xmx_prim`, and pre-warms all 24 `(B, M)` `dnnl::matmul` primitives during `WarmupPrefillKernels`, eliminating mid-prefill oneDNN JIT compilation across 2K/5K/8K prefills (`2K: 7,966.9 -> 7,421.5 ms`, `5K: 9,751.9 -> 9,526.6 ms`, `8K: 12,317.0 -> 12,008.2 ms / 646.6 tok/s`).
    - **Dynamic Budget-Derived QSA Query Tile (`AILA_QSA_QUERY_TILE` unset -> `16..1024`)**: Sizes `qsa_tile_active_blocks` and `qsa_tile_valid_counts` for up to 1,024 query rows and derives `effective_query_tile` per prefill window from `AILA_QSA_LOGITS_BUDGET_BYTES / blocks` (`16..1024`), reducing QSA prefill launches up to 16× before sparse context grows huge.
    - **Per-Token PLE `PrefetchVirtualMemory` (`AILA_Q4E_PLE_PREFETCH_MIN=1`) & Zero-Readback `host_token_ids`**: Lowers the PLE Win32 `PrefetchVirtualMemory` floor from 64 tokens to 1 token (eliminating 16–32 demand page faults per decode step on the PLE worker, `ple_host: 6.55 -> 0.26 ms`, `ForwardWall: 64.17 -> 60.14 ms`) and passes `options.host_token_ids` + `next_prefill_tokens` directly from `EngineGeneration` so neither prompt slices nor decode steps block on a D2H token-ID readback.
    - **Fused Multi-Row Verification Prepare (`AILA_Q4E_VERIFY_DECODE_PREPARE=1`), Pinned Host Readback in `JoinArenaColdRowsRow` & 64-Slot `Qwen4ExpMtpDraft` Host Embedding Ring**: Replaces the 6-launch `split_qgkv + head_rms_norm` sequence on 2..4-row verification windows with two fused `ops::decode_prepare_qgkv_packed_partial` launches, enables zero-upload pinned host readback (`moe_ordered_accumulate_cpu_host_rows`) on batched verification rows, and replaces the single-slot blocking host embedding upload in `Qwen4ExpMtpDraft::Step` with a 64-slot non-blocking ring (`hostEmbedEvents_[64]`).

---

## 4. Decode & Retained Research Subsystems

The following decode and experimental subsystems are actively maintained and covered by unit/module tests:

- **QSA KV Streaming (`QsaKvStreamOps`, `AILA_Q4E_KV_STREAM=1`) & Dynamic DXGI Budget**: Keeps authoritative full-context QSA K/V (and INT8 scales) in lazily grown `16K`-token pinned host USM segments (`AILA_Q4E_KV_HOST_SEGMENT_TOKENS=16384`) and replaces the `768 MB` full-context host `indexer_k` with a `24 KB` (`{8, 128}`) per-layer device tail ring + shared prefill staging buffer, while holding `indexer_compressed_k` and a bounded KV VRAM slot pool with device-side CLOCK eviction (`ops::qsa_kv_stream_resolve`), and stages chunked prefill 1 layer at a time inside `buf_.phase_scratch` (`ops::qsa_kv_stream_stage_from_host`). At load and prefill boundaries, `DxgiMemory` (`AILA_Q4E_DXGI_BUDGET=1`) adapts `ExpertCache` ceiling to real available VRAM, auto-selects KV resident tiers (`8208/16384/32768/65536` when `AILA_Q4E_KV_RESIDENT_TOKENS` is unset), and trims `phase_scratch` + prefill scratch (`1,356.7 MB`) right after prefill (`AILA_Q4E_AUTO_DECODE_TRIM=1`, trigger `2048 MiB`). Combined with pageable shared `embed_tokens` (`-1.18 GiB` locked RAM), post-load unmapping of `96` unused SafeTensors shards, `512`-row `hiddenRows` (`-150 MB` MTP VRAM), QSA-bounded `decode_scores`/`decode_attn_partials` (`-73 MB` VRAM), batched MTP draft catch-up (`StepBatchKvOnly`), and multi-row sparse QSA decode attention (`attention_decode_qsa_rows`), `262K` INT8 Session VRAM stays `503.2 MB`, GPU slots rise `2,978 -> 3,070` (`7,499 MB`, `54.5%` decode hit rate), peak system RAM drops from `>98%` to `93.9%` peak / `93.3%` steady (`4.0-4.4 GiB` free on 64 GiB host), and 20K TTFT reaches `35.67s (~560 tok/s)` with decode `~24-25 tok/s`. Auto-enabled by `tools/server/start-qwen38.ps1` at `>=64K` BF16 or `>=128K` INT8.
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
- `AILA_Q4E_VRAM_RESERVE_MB=1400`, `AILA_Q4E_LM_HEAD_Q8=1`, `AILA_Q4E_MAX_GPU_SLOTS=3072` (MTP) or `3456` (non-MTP), `AILA_Q4E_HOT_RAM_MB=32768`, `AILA_Q4E_EXPERT_READER=readfile`
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
