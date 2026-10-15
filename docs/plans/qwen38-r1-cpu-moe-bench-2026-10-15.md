# E9b: CPU-MoE prefill feasibility bench (2026-10-15)

Scope: prove (or kill) the coordinator insight "offload prefill MoE rows to the
CPU so the GPU staging H2D (46.8 GiB/block, ~5.0 s copy + 2.4 s fill, E1
six-bucket table) shrinks". This document is the design + reading guide; the
measurements are produced by the coordinator's build/run of
`AilaQwen4ExpCpuMoePrefillBench` (E9b is forbidden to build or run).

Discipline: charter sec 5 (evidence first, no fixed sleeps, GPU mutex is not
needed for a pure host bench but the machine must not be under load from
another agent's arms), sec 2 negative list untouched, ASCII only, estimates
marked `[estimate]`, unknowns marked TODO.

---

## 1. Design

### 1.1 What is measured

`perf/qwen4exp/q4e_cpu_moe_prefill_bench.cpp` drives the exact production
entry point `ops::cpu::cpu_expert_batch_forward` (the only caller of the Q8/A64
kernels; `src/models/Qwen4ExpExpertExecutor.cpp:488`) inside a
`tbb::task_arena`, mirroring `Qwen4ExpExpertExecutor::CpuWorker`
(`src/models/Qwen4ExpExpertExecutor.cpp:143,461-492`). Slots are synthetic host
memory in the production compact-nested layout (slot = 2,534,400 B; gu codes
51,200 B, dn codes 25,600 B; nested absmax gu 200 floats / dn 100 floats,
blocksize 256), identical to `q4e_decode_moe_bench.cpp:48-70`. Production env
defaults are passed explicitly as arguments (scale_shift=1, a64_specialize=1,
a64_row2=1; `docs/plans/qwen38-server-mtp-2026-10-07.json`), so no env capture
race exists.

Arms (one call = one `cpu_expert_batch_forward` invocation):

| arm | call shape | rows/call | window experts/call | note |
|---|---|---:|---:|---|
| decode16 | 16 tasks x 1 row, tokens=1 | 16 | 16 | decode job granularity anchor |
| flat40 | 40 tasks x 1 row, tokens=1 | 40 | 40 | max tasks, no row reuse |
| quad10x4 | 10 experts x 4 rows, tokens=4 | 40 | 10 | MTP-verify shape (`VERIFY_CPU_ROWS=4`) |
| oneex4 | 1 expert x 4 rows, tokens=4 | 4 | 1 | per-expert workhorse call |
| grp1x16 | oneex4 x 4 consecutive calls | 16 | 1 | weights hot across calls |
| grp1x32 | oneex4 x 8 consecutive calls | 32 | 1 | prefill-like rows/expert |

Each arm rotates its window by `window_experts` every iteration over a
512-expert pool (1.24 GiB) so consecutive timed iterations are DRAM-cold; the
`reps>1` arms intentionally keep one window hot to isolate weight reuse. Every
arm prints `us_per_row`, `rows_per_s`, `window_weight_gbps` (distinct slots per
call) and an `[CPU MOE EQ]` line with the equivalent per-layer / per-block time.

### 1.2 API caps that constrain the shape space (why the arms look like this)

- `cpu_expert_batch_forward` rejects `tokens>4` or `count>40`
  (`src/ops/Qwen4ExpCpuExpertOps.cpp:1581`); max rows per call is therefore 40.
- The per-expert AVX2/TBB entries fall back to the **scalar** kernel above 4
  tokens (`:1372, :1475, :1514`). There is no vectorized >4-row path today, so
  "8-32 rows/expert" can only be expressed as consecutive 4-row calls; the
  `grp1xR` arms measure exactly that (including call overhead).
- `nf4_q8_batch_forward` stack-quantizes at most 4 activations per call
  (`:910-931`); row grouping beyond 4 buys cache locality, not extra
  quantization amortization.

### 1.3 Guards (fail closed, exit 2/3)

- AVX2-batch vs scalar-forward canary on 1 expert x 1 row, tolerance
  2e-3 relative (backend-independent correctness of the synthetic slot build).
- Repeat determinism: the same batch executed twice must be bit-identical.
- Finiteness of the last output block; sentinel tails on `h` and `out` must be
  untouched.
- Q8/A64 arms are approximation paths by design (`CpuKernelBackend` comments,
  `Qwen4ExpCpuExpertOps.hpp:25-28`); only the AVX2 path is compared bitwise.
  A64 exactness vs the GPU is the existing `AILA_Q4E_CPU_Q8_VERIFY` /
  `AILA_Q4E_CPU_A64_VERIFY` machinery, out of scope here.

---

## 2. Three shapes, readings and pre-registered expectations

Status: pending coordinator run (TODO: fill the measured columns; keep the
logged JSON lines as receipts in `tmp/q4e-sprint-r1-20261014/cpu-moe/`).

Reading order per log: `[CPU MOE DEVICE]` -> `[CPU MOE CONFIG]` ->
`[CPU MOE GUARD]` -> `[CPU MOE BENCH]` per arm -> `[CPU MOE GUARD2]` ->
`[CPU MOE EQ]` per arm.

| quantity | decode anchor (evidence) | shape (a) decode16 | shape (b) flat40 | shape (c) quad10x4 / grp1xR |
|---|---|---|---|---|
| rows/call | ~230 activations/token across 48 layers (C1 report) | 16 | 40 | 40 / 4 |
| us per (expert,row) | **53.7** (briefing; 556 MB / 13.7 ms = 45 GB/s, C1/E2 logs) | = anchor check | = parallel-cap check | = reuse effect |

Pre-registered interpretations (charter sec 5.3-style thresholds):

1. Anchor: if `decode16` `us_per_row` is outside 45-70 us, the bench differs
   from production accounting; stop and explain before using any other arm.
2. Reuse: `quad10x4` vs `flat40` per-row difference isolates row grouping at
   equal rows/call. `[estimate]` compute-bound => within 10% of each other;
   a clear win (>15%) would mean weight streaming binds in flat shapes.
3. Depth: `grp1x16/32` vs `oneex4` isolates "weights hot across calls" for the
   real prefill pattern. Flat result => per-row cost is instruction-bound and
   no cache-depth lever exists below the >4-row API gap.
4. Threads: rerun `--threads 18` (production `AILA_Q4E_CPU_THREADS`) and
   `--threads 2` (library default clamp) to bound the coordinator's
   configuration choice.

Per-row FLOPs for interpreting GFLOPS (derived, not measured):
gate_up = 2*640*2560 = 3,276,800 MAC; down = 2560*640 = 1,638,400 MAC;
total 4,915,200 MAC = **9.83 MFLOP per (expert,row)**. At 53.7 us/row this
implies an aggregate ~183 GFLOPS [derived], i.e. the decode CPU path is
already near its integer/AVX2 ceiling, not DRAM-limited (45 GB/s matches
2,534,400 B / 53.7 us = 47 GB/s per slot).

---

## 3. Conclusions and the arithmetic to 400+ tok/s

### 3.1 Correct block arithmetic (the briefing's 2.2 s is per layer)

- Prefill routes: 4096 tokens x top_k 10 = **40,960 rows per layer**; x 48
  layers = **1,966,080 rows per 4096-token block**.
- The briefing's "40,960 activations x 53.7 us = 2.2 s/block" is arithmetically
  the **per-layer** time; at a constant per-row rate the full block is
  ~105 s (2.2 s x 48) [derived]. Any CPU-MoE prefill proposal must be priced
  against that, not 2.2 s.
- CPU share that fits an overlap budget B_s:
  `rows_cpu <= B_s / us_per_row`. At 53.7 us: 6.8 s (the device GDN+full-attn
  time per block, E1 section 1) buys ~126,600 rows = 3.1 layers = 6.4% of the
  block's rows [derived].

### 3.2 Comparison anchors

- GPU expert kernels: 0.4 s/block (50.26 GB @ 116.5 GB/s, E1 mode 0). The CPU
  would have to beat 0.2 us/row to be a straight replacement; at ~54 us/row it
  is ~250x slower [derived]. **CPU-MoE cannot replace the GPU MoE work; it can
  only, if ever, overlap a small slice.**
- Staging supply saved per row moved to CPU: 2,534,400 B of H2D + its host
  fill share (~46.8 GiB/block for ~19.3K cold slots/layer-block; E1). One row
  moved to CPU saves about 2.4 MB of PCIe copy (5.0 s/block basis) but costs
  ~54 us of CPU [derived].
- Break-even (ignoring overlap): CPU row at 54 us must save >= 54 us of
  critical-path staging. 54 us of the 9.9 GB/s copy is ~0.53 MB, so a CPU row
  only pays for itself if moving that expert's rows to the CPU removes more
  than ~0.5 MB from the critical path. With ~80-100 rows per cold expert, one
  expert's H2D (2.4 MB) is paid for by ~5 CPU rows [derived]; the binding
  constraint is the CPU's absolute time inside the block wall.
- Decision rule to pre-register in the receipt: propose a CPU-MoE slice only
  if `rows_cpu x us_per_row <= overlap_budget` AND the projected e2e wall gain
  is >= 5% (charter sec 5.3 default threshold); otherwise NO-GO / close.

### 3.3 Concurrency question (the coordinator's question 3)

Yes, a separate arm is required before wiring anything. The CPU streams slot
bytes from DRAM (~45 GB/s observed) while the staging pipeline also reads DRAM
for H2D and the GPU reads for GDN/full-attn. GPU mutex does not arbitrate the
host memory bus. Cheapest validation: run `AilaQwen4ExpPrefillBlockBench
--arms block` alone, then concurrently with one grp arm of this bench
(two windows, no shared state), and compare the block-bench `copy_ms` /
`bankwait_ms` inflation. If the inflation eats the CPU overlap [estimate],
NO-GO. This can be done without any source change.

---

## 4. Risks and TODOs

1. Synthetic slots do not reproduce the real arena's mmap/THP residency; a
   window rotation over 1.24 GiB is DRAM-cold by construction, but page-level
   behavior of the real files is not modeled. TODO: if the numbers land near
   the GO line, repeat the decisive arm against `AilaQwen4ExpHostArena` data
   (one call site, no new file needed).
2. The bench measures the compute core only. The production job adds
   activation/route waits, result packet H2D and join bookkeeping
   (`src/models/Qwen4ExpExpertExecutor.cpp:451-537`; decode cpu_join 9-18 ms
   per token). Prefill integration must keep jobs aggregated (40 rows) to
   amortize those fixed costs. TODO: price a per-job overhead term if a slice
   is ever wired.
3. `--backend h8|row4|block64` arms exist for completeness; only A64
   (production, `CPU_NF4_Q8 + Q8_ACTIVATION64`) should be used for decisions.
4. The missing >4-row vectorized entry point is a real API gap, not a bench
   limitation. Only if `grp1x32` beats `oneex4` by >= 15% does a >4-row kernel
   become a candidate work item (new K-lane), with the same 5% e2e gate.
5. Whole-MoE CPU takeover is dead on arrival independent of shape details
   (105 s/block at the observed rate vs 0.4 s of GPU expert kernels); do not
   spend arms trying to rescue it.

---

## 5. Commands (coordinator; CPU-only, no GPU process needed)

```powershell
Get-Process AilaWorker,AilaServer,Aila -ErrorAction SilentlyContinue   # expect empty (avoid CPU contention)
pwsh -NoProfile -File build.ps1 -BuildDir build-refactor
build-refactor\AilaQwen4ExpCpuMoePrefillBench.exe --arms all --threads 18 *> tmp/q4e-sprint-r1-20261014/cpu-moe/cpu-moe-A1.log
build-refactor\AilaQwen4ExpCpuMoePrefillBench.exe --arms all --threads 18 *> tmp/q4e-sprint-r1-20261014/cpu-moe/cpu-moe-A2.log
build-refactor\AilaQwen4ExpCpuMoePrefillBench.exe --arms decode16,flat40,quad10x4 --threads 2 *> tmp/q4e-sprint-r1-20261014/cpu-moe/cpu-moe-t2.log
build-refactor\AilaQwen4ExpCpuMoePrefillBench.exe --arms all --threads 18 --backend h8 *> tmp/q4e-sprint-r1-20261014/cpu-moe/cpu-moe-h8.log
```

A1/A2 form the A/A noise floor (expect cv_pct small, medians within a few
percent on an idle machine). To fill section 2's table, paste the
`[CPU MOE BENCH]` and `[CPU MOE EQ]` JSON lines per arm into the receipt
`tmp/q4e-sprint-r1-20261014/cpu-moe/receipt-cpu-moe.json`.

---

## 6. Sources

- `perf/qwen4exp/q4e_cpu_moe_prefill_bench.cpp` (this delivery).
- `src/ops/Qwen4ExpCpuExpertOps.cpp:41-53, 905-931, 1372, 1475, 1514, 1581`
  (loader, Q8 batch, caps).
- `src/models/Qwen4ExpExpertExecutor.cpp:143, 461-492, 717-718` (arena, batch
  dispatch, default threads clamp 1..18).
- `docs/plans/qwen38-server-mtp-2026-10-07.json` (production CPU env set).
- `tmp/q4e-sprint-r1-20261014/` d1/e2/c1 logs (decode CPU rate anchor);
  E1 six-bucket table (19.5 s block, H2D 46.8 GiB, expert kernel 0.4 s).
- `perf/qwen4exp/q4e_decode_moe_bench.cpp:48-70` (slot geometry mirror).
- Syntax check: `tmp/q4e-sprint-r1-20261014/cpu-moe/syntax-check.cmd`
  (icx-cl -fsyntax-only, exit 0; no build artifacts).
