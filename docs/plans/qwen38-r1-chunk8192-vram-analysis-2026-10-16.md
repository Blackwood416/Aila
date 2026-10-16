# Chunk-8192 VRAM scaling analysis (Qwen4Exp / Qwen3.8 Flash Next R1)

Date: 2026-10-16. Repo `E:\RiderProjects\Aila`, HEAD-ish working tree (`f798c18` + the
uncommitted expert-XMX work). Read-only: no source edit, no build, no GPU run, no commit.

Question: the sprint runner pins the prefill chunk to 4096 (`AILA_Q35_PREFILL_CHUNK=4096`);
the previous chunk-8192 attempt died inside the 3584 MB reserve (audit §4.4,
`docs/plans/qwen38-r1-bottleneck-audit-2026-10-16.md:169-170`). With the new expert-XMX
prefill consumer (`AILA_Q4E_PREFILL_EXPERT_XMX=1`,
`docs/plans/qwen38-r1-prefill-expert-xmx-plan-2026-10-16.md`) we want a retry, because one
8192-wide forward roughly halves the H2D bytes per token.

Labels: `[measured]` = number read out of an existing log or reproduced from the source
formula to the byte; `[est]` = derived here.

Model under test: `C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2` — 48 layers (12 full-attn,
36 DeltaNet), `hidden=2560`, `num_experts=512`, `top_k=10`, `moe_intermediate=640`,
`head_dim=256`, `num_attention_heads=24`, `num_key_value_heads=2`, `vocab=248320`,
`indexer_{n_heads=4,kv_heads=1,head_dim=128,budget=2048,compress_ratio=4}`.
Sprint worker env: `AILA_Q4E_PHASE_SCRATCH=1`, `AILA_Q4E_VRAM_RESERVE_MB=3584`,
`AILA_Q4E_MAX_GPU_SLOTS=2800`, `AILA_Q4E_MTP_HIDDEN_REUSE=1` + `AILA_Q4E_SERVER_MTP=0`,
`AILA_Q4E_ATTN_NF4_Q8=0`, `AILA_Q4E_GPU_NF4_Q8=0`, `AILA_Q4E_GDN_CHUNK_MODE=0`,
`AILA_Q4E_DECODE_TRIM_CAP=0` (`tmp/q4e-sprint-r1-20261014/run.py:31-33` +
`docs/plans/qwen38-server-mtp-2026-10-07.json:worker_environment`). `max_seq_len_=32768`
(`max_position_embeddings=262144`, probe/server `--context 32768`).

---

## 0. Verdict in one paragraph

`cap` is not the context window; it is the backend's *runtime capacity*, and it is grown by a
**doubling** rule, not to the exact request size: `cap = max(seq_len, 2 * previous_cap)`
(`src/models/Qwen4ExpBnb4Backend.cpp:1019-1021`). With chunk 8192 and the runner's 6K warmup
request first, the failed arm therefore landed on **cap=11888, not 8192** — a 2× capacity
for a 1.45× chunk. The phase scratch peak is exactly `cap * top_k * hidden * 4` bytes
(`moe_down_acc`, `:1047`,`:1174`), i.e. 1.217 GB at 11888 and 800 MB at 8192 `[measured]`.
Everything else that scales with `cap` adds another ~542 MB going 4096 -> 8192 (and ~542 MB
again going 8192 -> 11888), so the reserve (3584 MB) must absorb **~2.26 GiB** of runtime
buffers at cap 8192 with the expert-XMX path on — tight but survivable — and **~3.1 GiB** at
cap 11888, which is over budget and is what paged. Conclusion: chunk 8192 is worth retrying,
but the retry must (a) keep `cap` at 8192 (no 6K warmup in the same process, or a growth-rule
change) and (b) ideally shave ~0.5 GB more via `moe_h` + `down_acc`; the reserve alone should
not be raised to hide the cap inflation.

---

## 1. Where `cap` comes from and why it scales with the chunk

### 1.1 Chunk plumbing (how a "4096/8192 chunk" becomes a forward count)

| Step | Anchor | Behaviour |
|---|---|---|
| Server per-request prefill step | `src/server/ServerMain.cpp:82` (`--prefill-chunk-size`, also `--n`), `src/server/HttpServer.cpp:250` | server option `prefillChunkSize`, clamped to `maxPromptTokens`; range 1..8192 |
| Scheduler chunk loop | `src/inference/Request.cpp:278-288` | `chunkSize = min(remaining, options.prefillChunkSize)`; one `session->Execute(...)` per chunk |
| Engine chunk cap | `src/engine/EngineGeneration.cpp:72-78` | `chunk_size = min(AILA_Q35_PREFILL_CHUNK, len)` — the **env is an upper bound applied on top of the server's `n`** |
| Effective chunk | — | `effective = min(server n, AILA_Q35_PREFILL_CHUNK)`. Both must be >= 8192 for a single 8192-count forward |
| Backend entry | `src/models/Qwen4ExpBnb4Backend.cpp:2287` | `ensure_runtime_buffers(ctx, count)` inside `ForwardImpl`, before any allocation-sized work |

### 1.2 `ensure_runtime_buffers(Context&, int seq_len)` — the capacity rule

```cpp
// src/models/Qwen4ExpBnb4Backend.cpp:1017
void Qwen4ExpBnb4Backend::ensure_runtime_buffers(Context& ctx, int seq_len) {
    if (seq_len <= runtime_seq_capacity_) return;                       // :1018 short-circuit
    int cap = std::max(seq_len, runtime_seq_capacity_ == 0
            ? std::min(max_seq_len_, 64)                                // :1019 first touch -> 64
            : runtime_seq_capacity_ * 2);                               // :1019 DOUBLING
    cap = std::min(cap, max_seq_len_);                                  // :1020
    if (cap < seq_len) cap = seq_len;                                   // :1021
    ctx.synchronize();                                                  // :1022-1024 drain readers
```

Derived capacities (`:1026-1032`):

```cpp
constexpr int kDensePrefillLimit = 2052;                                 // :1026
attn_cap     = prefill_attention_tile_ ? min(cap, prefill_attention_tile_) : cap;  // :1028
dense_cap    = min(attn_cap, 2052);                                      // :1029
dense_kv_cap = min(max_seq_len_, 2052);                                  // :1030  CONSTANT
down_rows    = moe_compact_tile_ ? ... : cap * cfg_.num_experts_per_tok; // :1031 (compact path is dead: :416)
```

Consequences that matter for this question:

1. **`cap` scales with the largest single forward count, not with the session context.** KV,
   indexer and DeltaNet state are sized by `max_seq_len_` per session and never grow with the
   chunk (Section 3.2).
2. **Growth is geometric.** Equal-size chunks after the first never grow it
   (`seq_len <= runtime_seq_capacity_` at `:1018`), which is why the 4096 arm stops at 4096.
   Any *smaller* request followed by a larger one overshoots to `2x`:
   - chunk-4096 arm `[measured, tmp/q4e-sprint-r1-20261014/dpe-b1-worker.log:11,17]`:
     `cap=64` then `cap=4096` (4096 <= 4096, no second growth).
   - chunk-8192 arm `[measured, tmp/q4e-sprint-r1-20261014/g2-c8192-banks4-worker.log:11,17,60]`:
     `cap=64` -> first prompt (5944 tokens, one 8192-wide chunk) -> `cap=5944` -> second
     request's first chunk (8192) -> `cap = max(8192, 5944*2) = 11888`.
3. The capacity is reset to `trim_cap` by `TrimDecodeRuntimeBuffers` (`:1252-1270`,
   `AILA_Q4E_DECODE_TRIM_CAP`, default 0 = disabled), which re-enters
   `ensure_runtime_buffers(ctx, trim_cap)` with `runtime_seq_capacity_ = 0`. Useful as the
   "break the doubling" lever between two requests in one process.
4. Load path: `ensure_runtime_buffers(ctx, 1)` at `:925`, i.e. the very first capacity is the
   `min(max_seq_len_, 64)` = 64 floor — matches every sprint worker log line
   `[Q4E PhaseScratch] cap=64 ...`.

### 1.3 The `phase(...)` scratch allocator and the peak formula

```cpp
// src/models/Qwen4ExpBnb4Backend.cpp:1038-1056  (AILA_Q4E_PHASE_SCRATCH=1 in the sprint)
const size_t bytes[] = {attn_cap * full_fused_qkv_dim_ * sizeof(bf16),   // full_qkv     :1043
                        attn_cap * linear_all_dim_      * sizeof(bf16),   // linear_all   :1044
                        full_q_heads_ * dense_cap * dense_cap    * 4,     // scores       :1045
                        full_q_heads_ * dense_cap * dense_kv_cap * 4,     // incr_scores  :1046
                        down_rows * hidden_size_ * sizeof(float)};        // moe_down_acc :1047
size_t peak=0,total=0; for(size_t n:bytes){peak=max(peak,n);total+=n;}
buf_.phase_scratch = Tensor::allocate(ctx,{int64_t(peak)}, u8);            // :1049
AILA_LOG_INFO("[Q4E PhaseScratch] cap=%d bytes=%zu separate_bytes=%zu saved_bytes=%zu", ...)
auto phase = [&](shape, type) {                                            // :1053-1056
    return s_phase_scratch ? Tensor::view(ctx, buf_.phase_scratch.data(), shape, type)
                           : Tensor::allocate(ctx, shape, type); };
```

With this model's geometry the five members are
`full_qkv = cap*26624`, `linear_all = cap*32960`, `scores = 24*dense_cap^2*4`,
`incr_scores = 24*dense_cap*2052*4`, `moe_down_acc = cap*10*2560*4 = cap*102400`.

| cap | peak (`bytes`) | `separate_bytes` | dominant member | log anchor |
|---:|---:|---:|---|---|
| 64 | 12,607,488 | 23,367,680 | `incr_scores` 12.02 MiB | `dpe-b1-worker.log:11` |
| 4096 | 419,430,400 | 1,471,941,632 | `moe_down_acc` 400.0 MiB | `dpe-b1-worker.log:17` |
| 5944 | 608,665,600 | 1,771,288,064 | `moe_down_acc` 580.5 MiB | `g2-c8192-banks4-worker.log:17` |
| **11888** | **1,217,331,200** | **2,734,120,960** | `moe_down_acc` **1160.9 MiB** | `g2-c8192-banks4-worker.log:60` |
| 8192 (this retry) | **838,860,800** | 2,135,428,096 | `moe_down_acc` 800.0 MiB | `[est]` = formula |

Every logged number reproduces the formula to the byte `[measured]`. Two structural facts:

* From cap ~2052 upward the peak is **always `moe_down_acc`** = `cap * top_k * hidden * 4`
  (100 KiB per unit of cap for this model). `scores`/`incr_scores` are frozen at 385.5 MiB
  each by the D1 clamp `kDensePrefillLimit = 2052` (`:1026`,`:1083-1086`), so they stop
  contributing to the peak above cap 2052 — but they are also *not* used above it (count >
  2052 runs the sparse QSA path through `decode_scores`/`decode_attn_partials`).
* `full_qkv`/`linear_all` are *not* decode-only: `RunAttentionLayerTile` uses them for both
  branches (`src/models/Qwen4ExpPrefill.cpp:14-100`, `count == 1` path included), so they
  cannot simply be dropped after prefill unless `cap` itself is trimmed.

---

## 2. Every capacity-dependent runtime buffer

All sizes below are the allocations performed by `ensure_runtime_buffers` at `cap = 4096` and
`cap = 8192` (with `prefill_attention_tile_ = 0` -> `attn_cap = cap`), on the model geometry
above. "XMX?" = does `RunGroupedExpertsXmx` (`:1908-2027`, plan doc §2.2) touch it.
Paths: **P-ALU** = prefill expert path `RunAdaptivePrefill`+`RunGroupedExperts` (+`moe_h`);
**P-XMX** = `RunGroupedExpertsXmx` + `ops::moe_expert_xmx_prefill_chunk`;
**Dec** = `RunResidentDecode` (`:1280`,`:1587`) and the decode fusion at `:2800-2960`;
**MTP/ver** = decode-verification / MTP verify rows (`PrepareDecodeVerification` `:323-339`,
`verify_rows_alignment_`), which rides the same buffers with `count = 2..4`.

### 2.1 Device (USM) buffers — owned for the whole process

| # | Tensor / raw alloc | shape (texture) | @4096 bytes (MiB) | @8192 bytes (MiB) | Δ MiB | Paths | XMX? |
|---:|---|---|---:|---:|---:|---|---|
| 1 | `buf_.hidden4` `:1058` | `[cap,10240] bf16` | 83,886,080 (80.00) | 167,772,160 (160.00) | +80.00 | P, Dec, MTP/ver (HC residual stack, PLE) | n/a (attention/HC) |
| 2 | `buf_.hc_N` `:1059` | `[cap,10240] bf16` | 83,886,080 (80.00) | 167,772,160 (160.00) | +80.00 | P, Dec, MTP/ver | n/a |
| 3 | `buf_.hc_mix_down` `:1060` | `[cap,320] bf16` | 2,621,440 (2.50) | 5,242,880 (5.00) | +2.50 | P, Dec | n/a |
| 4 | `buf_.hc_mix_up` `:1061` | `[cap,10240] bf16` | 83,886,080 (80.00) | 167,772,160 (160.00) | +80.00 | P, Dec | n/a |
| 5 | `buf_.hc_inject_buf` `:1062` | `[cap,4] bf16` | 32,768 (0.03) | 65,536 (0.06) | +0.03 | P, Dec | n/a |
| 6 | `buf_.hc_inject_f32` `:1063` | `[cap,4] f32` | 65,536 (0.06) | 131,072 (0.13) | +0.06 | P, Dec | n/a |
| 7 | `buf_.x` `:1064` | `[cap,2560] bf16` | 20,971,520 (20.00) | 41,943,040 (40.00) | +20.00 | P, Dec, MTP/ver (embed/router/shared/GEMM input) | **yes** (GEMM input) |
| 8 | `buf_.y` `:1065` | `[cap,2560] bf16` | 20,971,520 (20.00) | 41,943,040 (40.00) | +20.00 | P, Dec, MTP/ver (merge dst / residual) | **yes** (merge dst) |
| 9 | `buf_.q` `:1069` | `[attn_cap,6144] bf16` | 50,331,648 (48.00) | 100,663,296 (96.00) | +48.00 | P, Dec(row0) full-attn | no |
| 10 | `buf_.z` `:1070` | `[attn_cap,6144] bf16` | 50,331,648 (48.00) | 100,663,296 (96.00) | +48.00 | P, Dec(row0) | no |
| 11 | `buf_.k` `:1071` | `[attn_cap,512] bf16` | 4,194,304 (4.00) | 8,388,608 (8.00) | +4.00 | P, Dec(row0) | no |
| 12 | `buf_.v` `:1072` | `[attn_cap,512] bf16` | 4,194,304 (4.00) | 8,388,608 (8.00) | +4.00 | P, Dec(row0) | no |
| 13 | `buf_.attn_out` `:1073` | `[attn_cap,6144] bf16` | 50,331,648 (48.00) | 100,663,296 (96.00) | +48.00 | P, Dec, verification dump `:2638` | no |
| 14 | `buf_.indexer_qk` `:1091` | `[attn_cap,640] bf16` | 5,242,880 (5.00) | 10,485,760 (10.00) | +5.00 | P (sparse QSA, count>2052) | no |
| 15 | `buf_.indexer_q` `:1092` | `[attn_cap,512] bf16` | 4,194,304 (4.00) | 8,388,608 (8.00) | +4.00 | P (sparse QSA) | no |
| 16 | `buf_.router_scores` `:1128` | `[cap,512] bf16` | 4,194,304 (4.00) | 8,388,608 (8.00) | +4.00 | P, Dec (`:2735-2768`, fused/router) | **yes** (feeds `router_host_ids_`) |
| 17 | `pred_scores_` `:1129` | `[cap,512] bf16` | 4,194,304 (4.00) | 8,388,608 (8.00) | +4.00 | Dec fusion / router lookahead `:2874-2883` only | no |
| 18 | `batch_weights_dev_` `:1146` | `cap*10 f32` | 163,840 (0.16) | 327,680 (0.31) | +0.16 | P-ALU, P-XMX `:1767`, Dec CPU-join | **yes** (merge) |
| 19 | `buf_.moe_h` `:1154` | `[cap,10,640] bf16` | 52,428,800 (50.00) | 104,857,600 (100.00) | +50.00 | P-ALU `:1782`, Dec `:1291`,`:1604`,`:2856/:2858`, MTP/ver | **NO — bypassed** |
| 20 | `buf_.shared_gate_buf` `:1168` | `[cap,1] bf16` | 8,192 (0.01) | 16,384 (0.02) | +0.01 | P, Dec shared expert | no |
| 21 | `buf_.shared_gate_up_buf` `:1169` | `[cap,1280] bf16` | 10,485,760 (10.00) | 20,971,520 (20.00) | +10.00 | P, Dec | no |
| 22 | `buf_.shared_inter_buf` `:1170` | `[cap,640] bf16` | 5,242,880 (5.00) | 10,485,760 (10.00) | +5.00 | P, Dec | no |
| 23 | `buf_.shared_out_buf` `:1171` | `[cap,2560] bf16` | 20,971,520 (20.00) | 41,943,040 (40.00) | +20.00 | P, Dec (+`add_shared_expert_to_y` in merge) | **yes** (`fuse_shared`) |
| 24 | `expert_tiles_dev_` `:1181` | `[cap,10,2] s32` | 327,680 (0.31) | 655,360 (0.63) | +0.31 | P-ALU only (`min_rows` tile dispatch `:1859`) | **NO** |
| 25 | `expert_map_token_dev_` `:1194` | `cap*10 s32` | 163,840 (0.16) | 327,680 (0.31) | +0.16 | P-ALU only `:1856-1884` | **NO** |
| 26 | `expert_map_rank_dev_` `:1196` | `cap*10 s32` | 163,840 (0.16) | 327,680 (0.31) | +0.16 | P-ALU only | **NO** |
| 27 | `expert_map_slot_dev_` `:1198` | `cap*10 * 96 B` | 3,932,160 (3.75) | 7,864,320 (7.50) | +3.75 | P-ALU only | **NO** |
| 28 | `xmx_plan_dev_[0..1]` `:1219-1220` | `xmx_plan_bytes_` x2 | 658,688 (0.63) | 1,314,048 (1.25) | +0.63 | P-XMX only `:2006-2020` | **only path** |
| 29 | `buf_.phase_scratch` `:1049` | `peak` u8 | 419,430,400 (400.00) | 838,860,800 (800.00) | **+400.00** | see 2.3 | mixed |
| | **Subtotal, phase scratch ON** | | **987,507,968 (941.76)** | **1,975,012,608 (1883.52)** | **+941.76** | | |

Fixed-size (capacity-independent) additions inside the same reserve, listed for completeness:
`decode_scores [24,32768] f32` 3.00 MiB, `decode_attn_partials [24,256,272] f32` 6.38 MiB,
`indexer_block_scores` 32 KiB, `indexer_active_tokens` 128 KiB, `qsa_tile_*` 2.13 MiB,
`logits [1,248320] bf16` 474 KiB, `resident_down_ [10,2560] f32` 100 KiB (`:939`),
`verify_down_/verify_weights_` 0.39 MiB total (`:335-338`), and — when the flag is on — the
expert-XMX scratch **393,248,836 B = 375.03 MiB** (`src/ops/MoeExpertXmx.cpp:342-417`:
2 x 100 MiB gate_up banks + 2 x 50 MiB down banks + 20/10/5/40 MiB activations + plan), which
is created once with `max_chunk_rows = 4096` (`Qwen4ExpBnb4Backend.cpp:1222`) and therefore
costs the same at chunk 4096 and 8192.

Not allocated in the sprint configuration: `moe_window_sum`/`moe_inverse_map`/`moe_compact_invalid`
(`moe_compact_tile_ = 0` is hard-coded at `:416`, so `compact_acts=0`), `attn_xq`/`attn_xs`
(`AILA_Q4E_ATTN_NF4_Q8=0`), `moe_x_q8`/`moe_h_q8*` (`AILA_Q4E_GPU_NF4_Q8=0`),
`gdn_chunk_scratch_` (`AILA_Q4E_GDN_CHUNK_MODE=0`, `:1124-1125`), `batch_slots_*`
(non-expert-major only).

### 2.2 Host (pinned) mirrors — RAM, not VRAM, but capacity-scaled

`router_host_ids_`/`router_host_weights_`/`pred_host_ids_`/`pred_host_weights_` (`:1134-1137`),
`embed_host_ids_` (`:1138`), `embed_host_rows_` (`:1140`, present when `embed_host_.valid()`,
40.00 MiB at 8192), `expert_tiles_host_` (`:1179`), `expert_map_{token,rank,slot}_host_`
(`:1188-1192`): 26.2 MiB at 4096 -> 50.0 MiB at 8192 (+23.8 MiB `[est]`). These do not count
against the device reserve; they matter only for host commit/`native_ram_mib`.

### 2.3 The five phase-scratch views (all folded into the single `peak` allocation)

| View | shape | @4096 | @8192 | Live paths |
|---|---|---:|---:|---|
| `buf_.full_qkv` `:1068` | `[attn_cap,13312] bf16` | 104.00 MiB | 208.00 MiB | P (full-attn layers 3,7,...,47) + Dec row 0 |
| `buf_.linear_all` `:1111` | `[attn_cap,16480] bf16` | 128.75 MiB | 257.50 MiB | P (DeltaNet layers) + Dec row 0 |
| `buf_.scores` `:1085` | `[24,dense_cap,dense_cap] f32` | 385.50 MiB | 385.50 MiB | P dense attention only (count <= 2052) |
| `buf_.incr_scores` `:1086` | `[24,dense_cap,2052] f32` | 385.50 MiB | 385.50 MiB | P incremental dense attention (start_pos>0, count<=2052) |
| `buf_.moe_down_acc` `:1174` | `[cap*10,2560] f32` | 400.00 MiB | **800.00 MiB** | P-ALU **and** P-XMX (`:1768`, `:1916`), then `moe_ordered_accumulate_down_rows` |

`moe_down_acc` is allocated whenever `s_expert_major` is true, which is hard-coded
(`const bool s_expert_major = true;` `:1025`), so it exists even in a pure-ALU run.
`Tensor::allocate` is device USM (`src/core/Tensor.hpp:20-29` -> `ctx.alloc_device`), i.e. all
rows above are ledgered on the device budget.

---

## 3. Total VRAM delta of chunk 4096 -> 8192, and the reserve's ability to absorb it

### 3.1 Delta

| Quantity | chunk 4096 (cap=4096) | chunk 8192 (cap=8192) | Δ |
|---|---:|---:|---:|
| Capacity-scaled device buffers, phase scratch **off** | 1,945.51 MiB | 3,120.02 MiB | +1,174.50 MiB `[est]` |
| Capacity-scaled device buffers, phase scratch **on** (sprint) | **941.76 MiB** | **1,883.52 MiB** | **+941.76 MiB** |
| of which `moe_down_acc` (inside the phase peak) | 400.00 MiB | 800.00 MiB | +400.00 MiB |
| of which all other cap-scaled buffers | 541.76 MiB | 1,083.52 MiB | +541.76 MiB |
| Expert-XMX scratch (fixed, flag-gated) | 375.03 MiB | 375.03 MiB | 0 |
| **Runtime buffers charged to the reserve** | **1,316.8 MiB** | **2,258.5 MiB** | **+941.8 MiB** |

For reference, the *failed* arm ran at cap=11888 with the phase scratch on:
`2,866,081,248 B = 2,733.35 MiB` of runtime buffers `[est from the same formula; the
`[Q4E PhaseScratch] cap=11888` line is `[measured]`]` — i.e. the capacity inflation alone
cost 849.8 MiB versus a clean cap=8192.

### 3.2 What does *not* change with the chunk (and why that matters)

The KV/session state is sized by `max_seq_len_`, not by `cap`:

```cpp
// src/memory/SessionMemory.hpp:107-127
bytes.kv = 2 * kvElementBytes * kvHeads * headDim * capacity * fullLayers; // capacity = max_seq_len_
```

At `max_seq_len_ = 32768`, bf16 KV: `2*2*2*256*32768*12 = 805,306,368 B = 768.0 MiB`
(+ indexer K 96 MiB + compressed K 24 MiB + recurrent 112.2 MiB -> session state 1000.2 MiB,
matching `g2-c8192-banks4-worker.log:7`). With `AILA_KV_QUANT_FORMAT=int8` it halves to
384.0 MiB / 622.2 MiB total (`rv-2048-3600-b1-worker.log:7`). **The chunk does not touch any
of this**; the only interaction is that the session state is pre-reserved at admission and
therefore reduces what is left for the expert cache.

### 3.3 The admission formula and the log line to quote

`src/models/Qwen4ExpBnb4Backend.cpp:717-772` (anchors: `:728-730` explicit override,
`:737-755` reserve, `:760-763` math, `:766-772` log):

```cpp
const bool int8_kv = aila::env::read_flag("AILA_KV_QUANT", false) && aila::env::kv_quant_format_is_int8();
const size_t kv_element_bytes = int8_kv ? 1 : sizeof(bf16);                  // :723-726
const size_t total_session_state = kv_bytes + kv_scale_bytes + indexer_k_bytes
                                 + indexer_compressed_k_bytes + recurrent_bytes;   // :735
const int reserve_default = s_layer_stage_reserve ? 2450 : 2560;             // :745
const size_t max_reserve = max(0, read_int("AILA_Q4E_VRAM_RESERVE_MB", reserve_default)) << 20; // :747
const size_t reserve = max_reserve + total_session_state;                    // :760
const size_t min_bytes = 1024 << 20;                                         // :761
const size_t used = ctx.current_allocated_bytes() + reserve;                 // :762
expert_cache_bytes = vram > used + min_bytes ? vram - used : min_bytes;      // :763
```

`ctx.current_allocated_bytes()` at that point is the **static weights only** — the runtime
buffers of Section 2 do not exist yet (`ensure_runtime_buffers(ctx,1)` runs later, at `:925`;
the growth to `cap` happens inside `ForwardImpl` at `:2287`). The runtime buffers are therefore
**not** part of the admission arithmetic: they must fit inside `max_reserve`, together with
oneDNN scratch and the desktop. Verbatim log format (`:766-772`):

```
[Qwen4Exp] Memory Admission Accounting: Total VRAM=%.0f MB, Static Weights=%.1f MB, Session State=%.1f MB
(KV=%.1f MB element_bytes=%zu kv_scales=%.1f MB, IndexerK=%.1f MB, IndexerCompressedK=%.1f MB,
Recurrent=%.1f MB), Reserve Headroom=%.0f MB, Expert Cache=%.0f MB
```

Measured instance (sprint default, `g2-c8192-banks4-worker.log:7` / `dpe-b1-worker.log:7`):

```
Total VRAM=15933 MB, Static Weights=4522.3 MB, Session State=1000.2 MB (KV=768.0 MB
element_bytes=2 kv_scales=0.0 MB, IndexerK=96.0 MB, IndexerCompressedK=24.0 MB,
Recurrent=112.2 MB), Reserve Headroom=3584 MB, Expert Cache=6826 MB
```

Slot arithmetic immediately below (`:801-809`):
`slot_bytes = 3*hidden*moe_intermediate*33/64 = 2,534,400 B = 2.41699 MiB`;
`num_gpu_slots = min(AILA_Q4E_MAX_GPU_SLOTS=2800, int(expert_cache_bytes/slot_bytes) - 2*16)`.
The sprint's 6826 MB therefore becomes exactly **2792 admitted slots + 32 staging slots**
(6825.6 MB) — the same 2792 printed in `[Q4E Adaptive Residency] ... admitted_slots=2792`.

### 3.4 Can 3584 MB absorb the chunk-8192 delta?

Free headroom inside the reserve = `3584 MB - runtime buffers - oneDNN scratch - desktop`:

| Scenario | Runtime buffers | Free inside 3584 MB | Measured/reference |
|---|---:|---:|---|
| chunk 4096, phase on, XMX off (today's baseline) | 941.8 MiB | 2,642 MiB | decode 17.0-17.6 tok/s (`e10-bundle-run.log`) |
| chunk 4096, phase on, XMX on | 1,316.8 MiB | 2,267 MiB | (steady state after R1 lands) |
| **chunk 8192, phase on, XMX on, cap=8192** | **2,258.5 MiB** | **1,325 MiB** | this analysis; `[est]` |
| chunk 8192, cap inflated to 11888, XMX off | 2,733.4 MiB | 851 MiB | paging, TTFT 134.8 s (audit §4.4, `g2-c8192-banks4-run.log`) |

The A770's WDDM cliff is documented at ~1 GB free ("~1 GB free after warmup slowed all kernels
~2x; ~2 GB free did not", `:718-721` and `docs/Environment_Variables.md:281`). So:

* **Yes, 3584 MB absorbs chunk 8192 — but only if `cap` stays at 8192.** 1.33 GB of headroom
  is above the cliff, yet well below the 2 GB comfort band, so it leaves nothing for a bigger
  oneDNN prefill scratch or a heavier desktop. It is a *marginal* pass.
* **No, it does not absorb the doubling.** At cap 11888 the runtime buffers alone are
  2.73 GiB; adding the 375 MiB XMX scratch would be 3.1 GiB of a 3.5 GiB reserve. The original
  failure is a capacity-inflation failure, not a chunk-8192 failure.

### 3.5 Expert-cache cost of raising the reserve instead

`+1 MB` of reserve removes `1 MB` from the expert cache (until the 1024 MB floor, `:761-763`),
i.e. `1/2.41699 = 0.4137` slots:

| Reserve | Expert cache | Slots (with 32 staging) | Δ slots |
|---:|---:|---:|---:|
| 3584 (sprint) | 6826 MB | 2792 | — |
| 3840 (+256) | 6570 MB | 2686 | -106 (-3.8 %) |
| **4096 (+512)** | 6314 MB | **2580** | **-212 (-7.6 %)** |
| 4608 (+1024) | 5802 MB | 2368 | -424 (-15.2 %) |

Fewer slots means more per-token CPU expert misses (the audit's 2792-slot baseline is
199-275 misses/token, §3.2), so decode pays for the headroom. The sprint's own counter-example
is `rv-2048-3600-b1`: `Reserve Headroom=2048 MB` + int8 KV + 3584 admitted slots put the
working set past the 15933 MB device and decode collapsed to 6.5-11.0 tok/s
(`rv-2048-3600-b1-run.log`, `-worker.log:7-8`; audit §4.4 quotes 7.4 tok/s). Do not buy the
8192 chunk with reserve money unless the items in Section 4 are exhausted.

---

## 4. Ranked, minimal changes to make chunk 8192 fit (no edits performed here)

Cumulative target: keep cap=8192 runtime buffers at or below ~1.4 GiB so that the reserve
keeps >= 2 GB free and the cliff is nowhere near.

| Rank | Change | MB saved at cap 8192 | Risk | Files that would need edits |
|---:|---|---:|---|---|
| **R1** | **Stop the capacity inflation**: cap should track the chunk, not double. Either (a) run the retry without the 6K prime in the same process (config-level, zero code), (b) call `TrimDecodeRuntimeBuffers` (`AILA_Q4E_DECODE_TRIM_CAP=64`) between the warmup and the big prompt, or (c) change the growth rule to `cap = max(seq_len, min(max_seq_len_, runtime_seq_capacity_ + runtime_seq_capacity_/2))` / `ceil_pow2(seq_len)` | **849.8** (relative to the observed 11888 run) | Low. No numerics. (c) adds a realloc on a 8192->4096->8192 alternating workload | `src/models/Qwen4ExpBnb4Backend.cpp:1019-1021` (option c only) |
| **R2** | **Make `moe_down_acc` narrower** (f32 -> bf16 or f16): halves the phase peak; the merge dst `buf_.y` is already bf16 | **400.0** (peak 800 -> 400; total 1883.5 -> 1483.5) | Medium: rank-sum precision (top_k=10 terms). Needs the §4 quality gate, not a hash diff | `Qwen4ExpBnb4Backend.cpp:1174` (alloc), `:1768-1771` (merge), `src/ops/MoeExpertXmx.cpp:313` (`prim.d2` dtype) + scatter in `run_prefill_chunk_typed`, `src/ops/MoeExpertOps.cpp` (`moe_expert_major_prefill_down`, `moe_ordered_accumulate_down_rows`) |
| **R3** | **Cap-shrink the HC stack** `hidden4`/`hc_N`/`hc_mix_up` ([cap,10240] bf16, 3 x 160 MiB at 8192) to Q8/int8 with an f32 (or bf16) dequant-on-read path, as the audit's lever 5 suggests | **240.0** (3 x 160 -> 3 x 80) | Medium-high: touches every layer's HC/PLE path and the PLE snapshots; long tail of consumers (`:2114-2235`, `:2499`, `:2675-2690`) | `Qwen4ExpBnb4Backend.cpp:1058-1061` + HC kernels in `src/ops` |
| **R4** | **`moe_h` lazily / decode-sized when expert-XMX prefill is on**: XMX never touches it (`RunGroupedExpertsXmx` `:1908-2027`), and every remaining consumer needs only `<= 2*top_k` rows (decode `:1291-1629`, XMX-decode `:2856/:2858`) | **100.0** (104,857,600 B) | Low-medium: keep a lazy re-alloc for the ALU fallback call shapes (`:1786-1789`) | `Qwen4ExpBnb4Backend.cpp:1154` (+ gate `:1224`) |
| **R5** | **Skip the ALU-only cap-sized descriptors when expert-XMX owns prefill**: `expert_map_token/rank/slot_dev_` + `expert_tiles_dev_` (`:1181`,`:1194-1198`) and their host mirrors | **8.75** device (+8.75 host) | Low; must remain allocated whenever `prefill_expert_xmx_ == false` or a non-standard call shape is used | `Qwen4ExpBnb4Backend.cpp:1179-1198` |
| **R6** | **Lazy `pred_scores_` + `pred_host_*`** (decode-fusion lookahead only, `:2874-2883`) | 4.0 device (+2.5 host) | Low | `Qwen4ExpBnb4Backend.cpp:1129`,`:1136-1137` |
| **R7** | **Raise the reserve** `AILA_Q4E_VRAM_RESERVE_MB=4096` (headroom +512, expert cache -212 slots / -7.6 %) as a *safety margin only* | 0 (it buys headroom, not bytes) | Low mechanically, but it taxes decode | env only: `tmp/q4e-sprint-r1-20261014/run.py:31`, `tools/server/start-qwen38.ps1:22` |
| **R8** | Orthogonal: `AILA_Q4E_DECODE_TRIM_CAP=64` after the last prefill chunk (`:1252-1270`) drops the decode-time runtime buffers to ~104 MiB | 1,779 (decode-time only) | Low, but it does **not** help the prefill peak and must not fire mid-prefill | env only |

Explicitly **not** worth it for this workload (checked, so the list stays minimal):

* "Split the phase scratch into prefill/decode sets" buys ~0 here. All five views are
  prefill-only lifetimes and the allocator already takes `max`, not `sum`; from cap 2052 up
  the peak is a single buffer (`moe_down_acc`). Splitting would only matter for
  `AILA_Q4E_PHASE_SCRATCH=0`, where chunk 8192 would be 3,120 MiB instead of 1,884 MiB.
* "Avoid allocating decode-only scratch during prefill" is worth ~10 MiB
  (`decode_scores` 3.0 + `decode_attn_partials` 6.4), and both are capacity-independent.
* `moe_window_sum`/`moe_inverse_map` (the compact-acts path) are dead code
  (`moe_compact_tile_ = 0`, `:416`); reviving it would cost +cap*hidden*8 = +167.8 MiB, so it
  is not a saving.

---

## 5. Recommended experiment once R1 (and ideally R2/R4) land

Run it through the sprint runner with bare `KEY=VALUE` overrides; note that the effective chunk
is `min(server n, AILA_Q35_PREFILL_CHUNK)`, so **both** must be 8192, and the 6K prime must be
absent so the capacity never doubles past the chunk. Keep the fixture (it suppresses the
probe's 16-token warmup request, `tools/server/qwen38_context_probe.py:138-140`) and use
`--diag` only when the worker log is wanted, because it turns on component profiling and
inflates TTFT:

```powershell
# memory gate first: cap must be 8192, not 11888
python tmp/q4e-sprint-r1-20261014/run.py c8192-xmx --diag --chunk=8192 `
    AILA_Q35_PREFILL_CHUNK=8192 AILA_Q4E_PREFILL_EXPERT_XMX=1
# then the clean timing arm (no --diag), same env, plus the decode headroom knob if desired
python tmp/q4e-sprint-r1-20261014/run.py c8192-xmx-prod --chunk=8192 `
    AILA_Q35_PREFILL_CHUNK=8192 AILA_Q4E_PREFILL_EXPERT_XMX=1
# if R1(c)/R7 are also under test, add: AILA_Q4E_DECODE_TRIM_CAP=64 (R8) or
# AILA_Q4E_VRAM_RESERVE_MB=4096 (R7) as bare overrides; the runner defaults to 3584
```

Acceptance signals to grep in `<label>-worker.log`:

```text
[Qwen4Exp] Memory Admission Accounting: ... Reserve Headroom=3584 MB, Expert Cache=6826 MB
[Q4E PrefillExpertXmx] enabled=1 max_chunk_rows=4096 ... plan_bytes=...
[Q4E PhaseScratch] cap=8192 bytes=838860800 separate_bytes=2135428096 saved_bytes=1296567296
[Q4E PrefillSlice] start=0 count=8192 ...           # one 8192-wide forward, not two 4096s
```

`cap=8192` with one `count=8192` slice means the runtime buffers are the 1,883.5 MiB column of
Section 3.1 and the reserve keeps ~1.33 GB free; if `--diag` shows `cap=11888` or two 4096
slices, the env/server chunk pair is not in effect (or a warmup request slipped in) and the
memory result is not comparable.

---

## Evidence index (read-only)

| Fact | Source |
|---|---|
| `cap` formula, `attn_cap`/`dense_cap`/`down_rows`, phase peak | `src/models/Qwen4ExpBnb4Backend.cpp:1017-1056` |
| Runtime buffer allocations | `src/models/Qwen4ExpBnb4Backend.cpp:1058-1245` |
| XMX early return; ALU-only descriptors | `src/models/Qwen4ExpBnb4Backend.cpp:1786-1789`, `:1856-1905` |
| XMX consumer touches only `x`, `down_acc`, `xmx_plan_dev_` | `src/models/Qwen4ExpBnb4Backend.cpp:1908-2027`; `src/ops/MoeExpertXmx.cpp:342-545` |
| Admission accounting + reserve + slot sizing | `src/models/Qwen4ExpBnb4Backend.cpp:717-809` |
| KV/session formula (capacity = max_seq_len) | `src/memory/SessionMemory.hpp:101-127` |
| Chunk plumbing (server n, engine env, per-request loop) | `src/server/ServerMain.cpp:82`; `src/engine/EngineGeneration.cpp:72-78`; `src/inference/Request.cpp:278-288` |
| Failed 8192 arm: cap 64 -> 5944 -> 11888, phase 12.6 MB -> 580.5 MB -> 1160.9 MB, TTFT 134.77 s | `tmp/q4e-sprint-r1-20261014/g2-c8192-banks4-worker.log:11,17,60`; `g2-c8192-banks4-run.log` |
| 4096 baseline: cap 64 -> 4096, phase 400.0 MB, 46.8 GB H2D per 4096 tokens | `tmp/q4e-sprint-r1-20261014/dpe-b1-worker.log:11,17,18` |
| 8192 chunk staged 49.6 GB for 8192 tokens (6.06 MB/token vs 11.4 MB/token at 4096) | same two logs, `h2d_mb=` fields |
| Reserve/slot sensitivity, paging cliff, audit §4.4 | `docs/plans/qwen38-r1-bottleneck-audit-2026-10-16.md:161-173`, `docs/Environment_Variables.md:280-281` |
| Sprint env defaults (reserve 3584, chunk 4096, phase scratch 1) | `tmp/q4e-sprint-r1-20261014/run.py:31-33`; `docs/plans/qwen38-server-mtp-2026-10-07.json` |
