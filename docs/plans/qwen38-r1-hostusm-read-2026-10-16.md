# Qwen4Exp host-USM streaming read probe (role P5, 2026-10-16)

Question: what is the effective kernel streaming-read rate **R** from host USM
(pinned `sycl::malloc_host`) on this A770 box? Candidate Z (point the MoE
consumer descriptors at host memory and delete the H2D pump) is the only
structural mechanism left with a >20% prefill claim, and its feasibility rests
entirely on a number nobody has measured. See
`docs/plans/qwen38-r1-prefill-overlap-design-2026-10-16.md` section 8 for the
Z1/Z2 derivation this probe feeds, and
`docs/plans/qwen38-r1-prefill-supply-breakdown-2026-10-16.md` for the supply
terms (fill 7.19-8.66 s, H2D 4.96 s @9.89 GB/s, fence ~0.5-1.3 s per 4096-token
block; device drain 8.58 s; block wall 19.4 s = 212 tok/s).

## 1. Existing tools: none of them measure this

Checked before writing anything:

- `tests/models/Qwen4ExpDmaTimingTests.cpp` - records submit/consumer
  timestamps of the staging pump (`DmaTimingStats`); not a bandwidth bench and
  no host-pointer kernel.
- `perf/qwen4exp/q4e_prefill_supply_bench.cpp` - fills host banks (memcpy) and
  streams them **into device staging**; every kernel reads device USM.
- `perf/qwen4exp/q4e_moe_bench.cpp --test-host-dma <GiB>` - allocates a pinned
  host pool and times the **H2D memcpy** of it. This is the 9.89 GB/s control
  family, still not a kernel read. (Owner can rerun it as an anchor:
  `build-refactor\AilaQwen4ExpMoeBench.exe --test-host-dma 1`.)
- `perf/qwen4exp/q4e_prefill_block_bench.cpp`,
  `perf/qwen4exp/q4e_layer_stage_bench.cpp` - supply vs kernel isolation and
  staging overlap, kernels read device USM only.
- Production has exactly one host-pointer kernel read today:
  `AILA_Q4E_CPU_RESULT_HOST_READ` (`src/models/Qwen4ExpExpertExecutor.cpp:655-661`,
  `merge_reader_fence`, host USM results read by GPU kernels). It moves
  `top_k * hidden * 4B` windows, is default off, and is not timed for
  bandwidth.

Conclusion: a new probe is required.

## 2. Probe

New files (no existing target touched):

- `perf/qwen4exp/q4e_hostread_bench.cpp` (probe)
- `CMakeLists.txt`: new `add_executable(AilaQwen4ExpHostReadBench ...)` only.

Binary: `build-refactor\AilaQwen4ExpHostReadBench.exe`.

Seven arms, each run at 512 MiB and 1 GiB, `--reps 5` + 1 warmup, median:

| arm | reads | purpose |
|---|---|---|
| `h2d` | `queue.memcpy` host->device | bandwidth control (expect ~9.89 GB/s) |
| `dev-flat` | device USM, uint32/lane, coalesced | VRAM roofline reference |
| `host-flat` | host USM, same pattern | host ceiling with best coalescing |
| `dev-flat16` / `host-flat16` | 16B `vec<uint32_t,4>` per lane | TLP-friendly variant |
| `dev-moe` / `host-moe` | production expert-major gate/up shape | the operative R |

The `moe` shape mirrors `launch_moe_expert_major_prefill_gate_up`
(`src/ops/MoeExpertOps.cpp:1584-1750`) at the real geometry: one 16-lane
sub-group per packed row of `hidden/2 = 1280B`, 20 steps of 64B, **two distant
streams per step** (gate row and up row), plus leader-only absmax gathers
(40 floats per row = 160B, blocksize 64) - exactly the load classes the
production kernel issues. The synthetic buffer is a sequence of 1440B cells
(1280B packed + 160B codes); one sub-group owns the pair `(c, c + cells/2)`, so
the two streams are half the buffer apart, like gate/up halves of a slot.

Guard (fail-closed): every arm is required to read **every** 32-bit word of the
effective prefix exactly once (packed lanes cover 0..1279 bytes, leaders cover
codes 0..159). Two order-independent accumulators (sum and sum-of-xor) are
atomically reduced into a device sink and compared bit-exactly against a CPU
reference. Coverage is structural (lane/k arithmetic), and the shared reference
means a missing or duplicated load shows up as `guard_ok=0` rather than an
inflated GB/s. The `h2d` arm has no kernel checksum, so its guard re-reads the
device buffer once through `dev-flat` and compares.

Other properties:

- Same kernel source for device and host arms; only the pointer differs, so
  `dev-moe / host-moe` is a direct ratio.
- `R` in the summary = logical bytes / median wall-clock (ms). Sector
  amplification is **not** folded in: the codes stream costs up to ~+9% extra
  PCIe sectors at blocksize 64 (160B of gathers per 1280B of packed), so a true
  effective-byte rate is at most ~9% below the printed `gbps`.
- Timed region is the kernel/copy event only; sink reset and checksum readback
  happen outside it. Event profiling is printed as `kernel_ms` when the
  runtime provides it; `wall` is the headline metric.
- Capabilities printed once: device name/vendor/driver, `usm_host_allocations`,
  `usm_device_allocations`, `usm_shared_allocations`, `usm_system_allocations`,
  `is_gpu`, local/global memory sizes.

## 3. Allocation and path

- Host side: `Context::alloc_host` -> `sycl::malloc_host`
  (`src/core/Context.hpp:247`), i.e. pinned host USM; requires the
  `usm_host_allocations` aspect (probe aborts with `[HR ERROR]` if missing).
- Device side: `Context::alloc_device` -> `sycl::malloc_device`.
- The production staging bank Z1 would read is already `sycl::malloc_host`
  (`src/models/Qwen4ExpAdaptiveCache.cpp:237`), so Z1 pays no new pinning cost.
- Z2 caveat (confirmed by source, not measured here): the large-page arena is
  **not** USM - `tests/models/Qwen4ExpHostArenaTests.cpp:26` asserts its pointer
  type is `unknown`. Z2 therefore requires pinning the ~1.15 GB active layer
  slice as host USM (or a USM-ified arena); a dual representation of the full
  51 GiB complement is impossible on a 64 GB box (overlap-design section 8,
  failure mode 2).
- `is_gpu=1` on the A770 plus the printed vendor/name means kernel reads of that pointer
  cross PCIe. SYCL exposes no finer peer-path query, so R itself is the proof
  of the path (a memcpy-class R proves a real link transfer; a KB/s-class R
  proves a pathological path and kills Z by itself).

## 4. How to run (owner; GPU exclusively held, >=15 s after the last arm)

    build-refactor\AilaQwen4ExpHostReadBench.exe
    build-refactor\AilaQwen4ExpHostReadBench.exe --smoke
    build-refactor\AilaQwen4ExpHostReadBench.exe --sizes 512 --reps 5 --arms h2d,host-moe,dev-moe

Read in this order: `[HR CAP]` (aspects), `[HR CONFIG]` (effective bytes),
`[HR REF]`, then seven `[HR BENCH]` lines per size, then one `[HR SUMMARY]` per
size. Costs ~15 s of GPU time (two sizes, 7 arms, 6 reps) plus ~1-2 s of CPU
for the 1 GiB reference scan.

Sanity gates (a run that fails these is void, not a result):

1. `guard_all_ok=1` in every `[HR SUMMARY]`; no `guard_ok=0` in any arm.
2. `dev-flat` >= 200 GB/s (VRAM streaming) and `h2d` in 9-11 GB/s. If `h2d`
   is far off, the box (not the kernel) is the problem.
3. `host-flat`/`host-moe` below `h2d` would signal a read path slower than
   DMA; report both sizes because 512 MiB may fit driver caches.

Decision table (from overlap-design section 9, applied to `R_moe`; `R_flat` is
the ceiling if only the packed stream were re-laid-out):

| R_moe | verdict for Z |
|---|---|
| >= 10 GB/s | Z1 and Z2 both live -> next step is a Z1 wiring probe (default off) |
| 8-10 | Z2 live, Z1 marginal (fill+fence terms dominate) |
| 6-8 | Z2 marginal (single digit win, L2 pollution unmodelled) |
| < 6 | close Z; write the negative result and stop spending GPU time |

## 5. What R buys (the arithmetic the probe decides)

From overlap-design section 8 (per 4096-token block, seconds):

    Z1(R) = 8.03 + 50.27/R + 2.47 (fill) + 1.32 (fence) + 0.6 (submits)
    Z2(R) = 8.03 + 50.27/R + 0.6
    tok/s = 4096 / wall

| R (GB/s) | Z1 wall | Z1 tok/s | Z2 wall | Z2 tok/s |
|---:|---:|---:|---:|---:|
| 6 | 20.8 | 197 | 17.0 | 241 |
| 8 | 18.7 | 219 | 14.9 | 275 |
| 10 | 17.5 | 235 | 13.7 | 300 |
| 12 | 16.6 | 247 | 12.8 | 319 |
| 15 | 15.8 | 260 | 12.0 | 342 |
| 31 | 14.0 | 292 | 10.2 | 400 |

Key arithmetic: 400 tok/s needs a 10.24 s block. Z2's floor at infinite R is
`8.03 + 0.6 = 8.63 s`, so Z2 reaches 400 only for R >= ~31 GB/s -
above the 9.89 GB/s the box actually delivers for H2D memcpy. In other words
**even a perfect host read lands Z2 at ~300 tok/s (R=10), a -29% block wall vs
19.4 s today, but not 400.** The probe can therefore cap or kill Z; it cannot
reach the sprint goal alone. If R lands at 8-12, the honest reading is "Z2 is
a real ~-20..-30% prefill lever, and 400 tok/s still needs the bytes cut
(chunk-8192 revival / routing rework) or a second mechanism on top".

Caveat carried from the design doc: these formulas assume the consumer kernels
already consume weights at 68-89 GB/s from VRAM, so at R >= 8 the read becomes
the limiter and the ALU rides along; below R ~= 6 the mechanism is negative
(the kernel would be slower than fill + H2D + device read is today).

## 6. Prediction and risks

Prediction: `R_flat` in 6-12 GB/s (PCIe gen4 x16 ~25 GB/s theoretical, WDDM +
uncached host reads cost roughly half; A770 kernel reads of host USM are
typically single-digit to low-teens GB/s), `R_moe` at or slightly below
`R_flat` (same 64B coalescing, plus the +~9% codes sector amplification is
*effective* bytes, which lowers the logical rate by the same order). Expected
outcome per the table: Z2 marginal-to-live, Z1 usually behind it. Treat
`R_moe >= 10` as a surprise worth a Z1/Z2 wiring probe, not as a 400 path.

Risks / failure modes:

1. Host USM allocation failure or OOM: 1 GiB pinned host + 1 GiB VRAM + 16B
   sink per size. Pinned memory is a limited WDDM resource; if `alloc_host`
   throws (`[HR ERROR] ... budget` or `malloc_host`), rerun with
   `--sizes 256,512`.
2. VRAM pressure: run only with the GPU held exclusively (no AilaWorker /
   AilaServer) and after the >=15 s drain, per sprint discipline.
3. `kernel_ms` may print -1 when profiling is unavailable for copy events;
   `median_ms`/`gbps` are wall-based and unaffected.
4. First-rep TLB/page-in effects are handled by the discarded warmup rep; if
   `--warmup 0` is used, expect low outliers - use the median anyway.
5. Signature risk of the conclusion: if `host-flat16` is much faster than
   `host-flat`, the dominant limit is request count/TLPs, which argues for a
   `vec`-widened consumer kernel rather than a redesign; if both sit at the
   same low rate, the link/latency is the wall and only byte reduction helps.

## 7. Verification already performed without the GPU

- `tmp/q4e-sprint-r1-20261014/hostread/syntax-check.cmd` (icx-cl
  `-fsyntax-only -fsycl`) exits 0 with no warnings (2026-10-16, off-slot).
- `tmp/q4e-sprint-r1-20261014/hostread/coverage_check.py` replays the moe
  lane/k/leader index arithmetic and proves the walk hits every word index
  exactly once (checked at pairs = 1, 2, 5, 17; the mapping is periodic), so
  the sum/sum-of-xor guard is exact rather than statistical.
- Not verified here: any GPU number. That is the owner's single run.
