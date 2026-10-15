# R1/A2: First-chunk PLE gather exposure (~20.5 s) - root cause and minimal fix

Author: A2 research subagent. Date: 2026-10-15. Repo HEAD: a3ee21e.
Scope: read-only source study + CPU-only probes. No src edit, no build, no GPU
process was started. All line numbers are from HEAD a3ee21e.

## 0. TL;DR

The 20.45 s / 20.70 s `ple_host_ms` on the first real 4096-token chunk is
**demand paging of the memory-mapped PLE n-gram tables**, not n-gram math,
NF4 dequant arithmetic, or H2D.

- One 4096-token chunk touches 4096 * 16 = 65,536 head rows; `dequant_to_staging`
  reads one 80 B packed row plus one 20 B absmax row per head row
  (Qwen4ExpPle.cpp:496-497), i.e. up to 131,072 random file pages.
- CPU-only replica of that access pattern over the real 33 safetensors files:
  **cold 27.585 s vs resident 0.048 s for the same 131,072 touches (575x)**;
  see tmp/q4e-sprint-r1-20261014/ple/probe-output.txt.
- Batched `PrefetchVirtualMemory` on fresh cold rows measured **12.18x**
  (186.9 us -> 15.3 us per access), so the paging queue depth is the lever.
- The load-time warmup cannot cover this: it uses a synthetic LCG token
  sequence (Model.cpp:857-860) whose pages are disjoint from real content.
- Recommended fix: F1 (prefetched gather, patch draft in section 3).
  Expected first-chunk exposure 20.5 s -> ~2-4 s [estimate], zero numeric impact.

## 1. Root cause

### 1.1 Where the 20.5 s is spent

Call chain for the first chunk (all in the worker build):

- `Qwen4ExpBnb4Backend::Forward` issues `ple_.prepare(...)` at the start of the
  forward (Qwen4ExpBnb4Backend.cpp:1802-1814) and consumes it at layer 1
  (:2022-2030). `ple_host_ms` is `stage_ms[kStagePle]` (:2493-2495), i.e. the
  host wall interval ending at `stage(kStagePle)` (:2033).
- `prepare` enqueues a worker task (:655-668). The worker runs
  `execute_prepare_slot` (Qwen4ExpPle.cpp:556-560), which computes n-gram ids
  (:532-534), gathers + dequants the rows into pinned staging
  (`dequant_to_staging`, :467-515), then enqueues the H2D copy on `copy_q_`
  (:538-542).
- `consume` blocks in `wait_task` and then on the H2D event
  (Qwen4ExpPle.cpp:704-711), so the gather wall time is charged to
  `ple_host_ms` whenever it exceeds the head start the worker got.
- The gather source pointers are `ModelWeights::host_view` results
  (Qwen4ExpPle.cpp:217-237), and for this model `host_view` returns pointers
  into memory-mapped safetensors files (SafeTensors.cpp:83-96 and
  897-932). Nothing reads those payloads at load time, so every new row is a
  demand page fault on first touch.

Table geometry (measured from the index + headers of
C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2):

- 128 `ngram_embedding.shard_s.weight` tensors, rows_per_shard = 2,500,012
  (Qwen4ExpPle.cpp:235-237).
- packed total 24,414.2 MiB + absmax total 6,103.5 MiB = 29.8 GiB, held in
  model-00005..model-00037-of-00131.safetensors.
- 4096 tokens * 16 heads = 65,536 packed row reads + 65,536 absmax row reads
  per chunk; rows are 80 B / 20 B, so nearly one distinct 4 KB page per read.

CPU-only replica (tmp/q4e-sprint-r1-20261014/ple/ple_gather_io_probe.py,
random rows over the real shard layout):

| pass | accesses | wall | per access |
|---|---:|---:|---:|
| cold (first touch) | 131,072 | 27.585 s | 210.5 us |
| resident (repeat) | 131,072 | 0.048 s | 0.4 us |

The same code path minus page faults is 575x faster, so the 13-20.5 s class of
measurements cannot be n-gram arithmetic (65,536 int64 mods), NF4 dequant
(10.5 M bf16 stores + 5.2 M LUT lookups [estimate << 100 ms]), or the 21 MB
H2D [estimate ~10 ms].

Measured `ple_host_ms` across runs (all 4096-row chunks unless noted):

| evidence | chunk | ple_host_ms | note |
|---|---|---:|---|
| w1-warmup-on-worker.log:19 | warmup pos=0 | 193.370 | pages hot at that moment |
| w1-warmup-on-worker.log:26 | first real pos=0 | 20699.374 | pages cold |
| w1-warmup-on-worker.log:28 | pos=4096 (1848 rows) | 0.938 | ple_reused=1, lookahead |
| w2-warmup-varied-worker.log:18 | warmup pos=0 | 36.830 | pages hot |
| w2-warmup-varied-worker.log:25 | first real pos=0 | 20450.427 | ids varied, still cold |
| w2-warmup-varied-worker.log:27 | pos=4096 (1848 rows) | 0.713 | ple_reused=1 |
| w3-committed-smoke-worker.log:19 | warmup pos=0 | 35.457 | pages hot |
| w3-committed-smoke-worker.log:26 | first real pos=0 (19 rows) | 0.317 | tiny row count |
| w3-committed-smoke-worker.log:58 | later session pos=0 | 1472.211 | 4096 rows, partly warm |
| diagnostic2-worker.log:17/19/21/23 (10-09, no lookahead build) | pos=0/4096/8192/12288 | 1667.160 / 13074.843 / 13712.281 / 6486.335 | every chunk cold; chunk-1 masked by JIT |

The 10-09 run is the cleanest proof that this is a per-chunk, per-new-rows
cost: with no lookahead every 4096-row chunk paid ~13 s, and the first chunk
only looked cheap (1.67 s) because the un-warmed JIT storm gave the worker a
multi-second head start before layer 1 consumed the ticket.

### 1.2 Why it is content-related and only appears on the first chunk

- The rows are addressed by the token n-gram ids, so the faulted page set is a
  function of the prompt content. No two texts share it.
- With the lookahead pipeline (Qwen4ExpBnb4Backend.cpp:1816-1827) chunk N+1's
  gather runs in the worker while chunk N still has 11-39 s of GPU work left to
  submit, so chunks 2+ of a request hide their fault storms
  (w1:28 / w2:27, ple_reused=1, <1 ms). Chunk 1 has no predecessor, and after
  `AILA_INIT_WARMUP=1` there is no JIT left to mask layer 0 either, so the
  whole gather is exposed at the layer-1 consume.
- "New session" is really "new page set": w3-committed-smoke-worker.log:58
  shows a later session still paying 1.47 s on a 4096-row chunk because that
  run's pages were only partly resident.

### 1.3 Why the 4096-row warmup does not cover it

- The warmup ids are synthetic: `(row * 4099 + 12345) % vocab`
  (Model.cpp:857-860) and `ForwardLogits` runs them on a temporary session
  (:852-870). Their n-gram ids map to a pseudo-random page set that is
  disjoint from any real prompt's.
- The warmup chunk itself measures 36.8-193.4 ms (w1:19, w2:18) while the very
  next 4096-row real chunk in the same process takes 20450-20699 ms
  (w1:26, w2:25): the pages were hot during warmup and had been evicted by the
  time real content arrived (the ~96 s warmup plus load-phase allocations is
  exactly the window in which standby pages get reclaimed).
- Consequence: warmup is a JIT/primitive warmer, not a PLE data warmer. Only a
  gather over the real token ids can warm the right pages, and that can only
  start when the prompt is known.

## 2. Fix candidates (ordered by engineering effort)

### F1. Batched page prefetch inside the PLE worker (recommended)

- Files/lines: `src/models/Qwen4ExpPle.hpp:205` (new private method),
  `src/models/Qwen4ExpPle.cpp` `execute_prepare_slot` (:532-536) plus a new
  helper next to `dequant_to_staging` (:467).
- Change: after `compute_ngram_ids`, build the (packed_row, 80 B) and
  (absmax_row, 20 B) address list exactly as `dequant_to_staging` does, then
  issue `PrefetchVirtualMemory` in batches of 64 before the scalar gather.
- Correctness: read-only prefetch of the same addresses the gather reads;
  no data, arithmetic, or ordering change -> bit-exact.
- Expected saving: the CPU replica measured 12.18x on cold rows; 20.5 s ->
  ~2-4 s [estimate] for the first chunk. Chunks 2+ already hidden, unchanged.
- Verify: first real chunk `[Q4E AdaptiveWork] pos=0 ... ple_host_ms` after
  `[Q4E Warmup]`, and `[Q4E ForwardWall]` total_ms; criteria below.

### F2. Parallelize the gather/dequant across 4 worker threads

- Files/lines: `src/models/Qwen4ExpPle.cpp:467-515` (split token range), and
  the worker task plumbing (:545-568) or a tiny persistent pool.
- Change: 4 sub-ranges, each thread computing its own n-gram ids slice,
  its own faults, and writing disjoint staging rows.
- Correctness: each row is independent and keeps its arithmetic order;
  staging writes are disjoint -> bit-exact. Not directly measured; the
  prefetch A/B shows page-in queue depth is the dominant lever, and threads
  provide the same queue depth.
- Expected saving: same class as F1, 20.5 s -> 3-6 s [estimate]. Useful as a
  fallback or addition if F1 lands below 2x on real hardware.

### F3. Policy: smaller first prefill slice for a cold session

- Files/lines: prefill slice scheduling (Qwen4ExpBnb4Backend.cpp:1816-1827
  lookahead assumptions; slice size produced by the scheduler/executor).
- Change: first forward of a session uses e.g. 1024 tokens, later slices keep
  4096 and stay lookahead-hidden.
- Correctness risk: changes chunk boundaries and kernel-path selection
  (QSA/attention thresholds); quantization math is unchanged but bit-exact
  layer dispatch cannot be assumed. QSA threshold discipline: the *prompt*
  length must stay >= 2052 for QSA paths; a smaller first slice must be
  validated against that rule, which is an open question.
- Expected saving: proportional to the slice, 20.5 s -> ~5 s [estimate] for
  1024-token first slice; final slice sizing washes out prefetch benefits.
- Only consider if the coordinator is willing to change prefill policy.

Recommendation: implement F1 now (S, measured 12x lever, zero numeric risk).
Keep F2 as the fallback if the GPU probe shows F1 < 2x. Defer F3.

## 3. Minimal diff draft (apply-ready, NOT applied)

```diff
--- a/src/models/Qwen4ExpPle.hpp
+++ b/src/models/Qwen4ExpPle.hpp
@@ -202,6 +202,7 @@
 private:
     void cleanup();
     void ensure_compute_capacity(Context& ctx, int rows);
+    void prefetch_gather_pages(const int64_t* ids, int num_tokens) const;
     void dequant_to_staging(const int64_t* ids, int num_tokens, int slot);
     void execute_prepare_slot(int slot, int num_tokens, const std::array<int, 2>& prev_tokens);
     void worker_loop();
--- a/src/models/Qwen4ExpPle.cpp
+++ b/src/models/Qwen4ExpPle.cpp
@@ -5,6 +5,10 @@
 #include <cstring>
 #include <iostream>
 #include <stdexcept>
+
+#ifdef _WIN32
+#include <windows.h>
+#endif
 
 static constexpr float kDefaultNf4QuantMap[16] = {
     -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
@@ -514,6 +518,43 @@
     }
 }
 
+void Qwen4ExpPle::prefetch_gather_pages(const int64_t* ids, int num_tokens) const {
+#ifdef _WIN32
+    // File-backed SafeTensors pages are demand-paged; one 4096-token chunk
+    // faults ~131k random pages serially. Prefetching them in batches lets the
+    // cache manager use real queue depth (measured 12.2x on the cold path).
+    static const bool s_enabled = aila::env::read_flag("AILA_Q4E_PLE_PREFETCH", true);
+    if (!s_enabled || num_tokens < 64 || shards_.empty() || rows_per_shard_ <= 0) return;
+
+    std::vector<WIN32_MEMORY_RANGE_ENTRY> ranges;
+    ranges.reserve(static_cast<size_t>(num_tokens) * NUM_HEADS * 2);
+    for (int t = 0; t < num_tokens; ++t) {
+        const int64_t* head_ids = ids + static_cast<size_t>(t) * NUM_HEADS;
+        for (int h = 0; h < NUM_HEADS; ++h) {
+            int shard_idx = static_cast<int>(head_ids[h] / rows_per_shard_);
+            const int64_t row_in_shard = head_ids[h] % rows_per_shard_;
+            if (shard_idx >= static_cast<int>(shards_.size()))
+                shard_idx = static_cast<int>(shards_.size()) - 1;
+            const auto& shard = shards_[shard_idx];
+            if (!shard.packed_weight) continue;
+            ranges.push_back({const_cast<uint8_t*>(shard.packed_weight) + row_in_shard * 80, 80});
+            if (shard.absmax)
+                ranges.push_back({const_cast<float*>(shard.absmax) + row_in_shard * 5,
+                                  5 * sizeof(float)});
+        }
+    }
+    const HANDLE process = GetCurrentProcess();
+    constexpr size_t kBatch = 64; // PrefetchVirtualMemory accepts at most 64 entries.
+    for (size_t i = 0; i < ranges.size(); i += kBatch) {
+        const size_t count = std::min(kBatch, ranges.size() - i);
+        PrefetchVirtualMemory(process, count, ranges.data() + i, 0);
+    }
+#else
+    (void)ids;
+    (void)num_tokens;
+#endif
+}
+
 void Qwen4ExpPle::execute_prepare_slot(int slot, int num_tokens, const std::array<int, 2>& prev_tokens) {
     if (num_tokens <= 0 || !ctx_ || slot < 0 || slot >= kNumStagingSlots) return;
 
@@ -533,6 +574,7 @@
                       layer_multipliers_.data(), ngram_heads_vocab_sizes_.data(),
                       ngram_heads_offsets_.data(), slot_ngram_ids_[slot].data(), EOS_TOKEN_ID);
 
+    prefetch_gather_pages(slot_ngram_ids_[slot].data(), num_tokens);
     dequant_to_staging(slot_ngram_ids_[slot].data(), num_tokens, slot);
 
     const size_t bytes = static_cast<size_t>(num_tokens) * EMBED_DIM * sizeof(bf16);
```

The patch above is verified: `git apply --check` passes on the current working tree. A generated copy lives at tmp/q4e-sprint-r1-20261014/ple/f1.patch; keep it LF (git apply rejects CRLF against these files).

Notes for the implementer:

- The address computation must mirror `dequant_to_staging`
  (Qwen4ExpPle.cpp:480-497), including the shard clamp at :485-487.
- 131,072 entries * 16 B = 2 MiB temporary vector per 4096-token chunk, on the
  worker thread only; decode (num_tokens=1) returns at the guard.
- Keep the env flag so the GPU probe can run A/B from one binary.

## 4. Probe requests (GPU; for the coordinator)

P1 (F1, required): A/B/B/A first-chunk exposure with cache control.

```powershell
# GPU mutex: no AilaWorker/AilaServer running.
python tmp/q4e-sprint-r1-20261014/run.py plepf-a1 --diag --prompts=6000 AILA_INIT_WARMUP=1 AILA_Q4E_PLE_PREFETCH=0
python tmp/q4e-sprint-r1-20261014/run.py plepf-b1 --diag --prompts=6000 AILA_INIT_WARMUP=1 AILA_Q4E_PLE_PREFETCH=1
python tmp/q4e-sprint-r1-20261014/run.py plepf-b2 --diag --prompts=6000 AILA_INIT_WARMUP=1 AILA_Q4E_PLE_PREFETCH=1
python tmp/q4e-sprint-r1-20261014/run.py plepf-a2 --diag --prompts=6000 AILA_INIT_WARMUP=1 AILA_Q4E_PLE_PREFETCH=0
```

- Read: in `*-worker.log`, the first `[Q4E AdaptiveWork] pos=0 count=4096` line
  after `[Q4E Warmup]` -> `ple_host_ms`, `submitted_ms`; matching
  `[Q4E ForwardWall]` `total_ms`; and TTFT from the run log.
- Criteria: B median `ple_host_ms` <= 4 s and <= 0.25x A (baseline
  ~20.5 s); TTFT improvement >= 15 s; chunk-2+ `ple_host_ms` unchanged;
  generated text for the fixed fixture identical to baseline (bit-exactness
  receipt).
- Cache caveat (mandatory): the effect exists only while the pages are cold.
  Either flush the standby list between every arm (RAMMap/EmptyStandbyList or
  reboot), or treat only the first run after a flush as the cold measurement
  and report the run order. Without cache control, A/B/B/A will read as noise
  because the pages stay resident (probe pass2: 0.4 us/access).

P2 (only if P1 shows F1 < 2x): repeat P1 with the F2 4-thread gather patch and
the same criteria. QSA threshold discipline does not apply to P1/P2 because
the prompt is 6000 tokens (> 2052) in both arms.

## 5. Limits of this study

- Absolute cold time depends on storage/cache state: the same 4096-row gather
  measured 13.07-13.71 s (10-09), 20.45-20.70 s (today), 1.47 s (partly warm)
  and 36.8-193 ms (hot). The doc separates the invariant (page-fault-bound
  gather) from the variable (storage state).
- No component timing inside `execute_prepare_slot` exists in the binary; the
  split between ids/dequant/H2D is arithmetic- and probe-derived, tagged
  [estimate] where not directly measured.
- Probe artifacts: tmp/q4e-sprint-r1-20261014/ple/ple_gather_io_probe.py,
  ple_prefetch_ab_probe.py, probe-output.txt.
