# R1 Sprint - Role R (cost model): decode/prefill item tables, if-X-free bounds, candidate pick

Date: 2026-10-14. Repo: `E:\RiderProjects\Aila`, HEAD `946c644`, pair
`retained20` (worker sha256 `4d75c59c...`, server `211aabb0...`). All numbers
are measured on this box unless marked `[estimate]`. ASCII only.

## 0. Conclusions

1. Baseline reproduced with the primed protocol: 14K TTFT 70.51/71.87 s,
   continuation decode 15.50/15.62 tok/s (A/A, two arms). The primed protocol
   (6K warm-up request first) is mandatory: the first large prefill of a fresh
   worker carries a one-time ~79 s penalty (section 1.2).
0. Day-2 additions: (a) the PLE first-chunk gather fix (section 6.9) cuts the
   fixture first request 50.5 -> 31.8 s (A/B/B/A, adopted, default on, commit
   `b92bd31`); (b) the opt-in load-time warmup (section 6.7) moves first-use
   JIT out of the first request (110 -> 51 s, startup +111 s, commit
   `a3ee21e`); (c) the layer-major prototype is mechanism-validated (H2D -75%)
   but wall-negative (-67%) and stays experimental in the tree (commit
   `9abb858`); (d) the QSA route probe is repaired and shows +37% at 8192
   length (section 6.8). Pairs `retained21` -> `retained23`.
   Later same day (window A + window B shared sprint, pairs `retained23` ->
   `retained24`): (e) charter Q6 closed -- two concurrent sessions rotate and
   aggregate 12.85-14.29 tok/s < single-session 15.43 (sections 6.12, L35/L40);
   (f) the decode regression scare was sub-agent CPU contention, not code
   (quiet-h4 = 15.43 tok/s, section 6.14); (g) the decode-attention 12x K/V
   re-read is now bench-quantified (869-871 us/row at 14336 rows=2 vs a
   101 us/row floor, L39) but the routed 14K decode caps its end-to-end value
   at ~6.5%, so it stays a bench-only asset for now; (h) GDN prefill kernel
   measured at 74.8 ms/layer (516.7 GB/s effective, not DRAM-walled).
2. Decode (retained20 refresh, n=512 tokens): 66.01 ms/token wall = 15.15 tok/s.
   Largest items: experts_host 22.11 (cpu_join 10.00 inside), router_wait 12.69,
   unprinted MoE-front/shared residual 9.09, ple_host 6.65, hc_host 6.22,
   sample 3.56.
3. Prefill (14K request): 3 full 4096 chunks at 19.75/20.36/20.43 s plus a
   1656-token tail at 11.12 s; 179.5 GiB H2D and 76,060 staged experts per
   request; fill+bank_wait (11.4 s/chunk) is the dominant host cost.
4. If-X-free ranking: layer-major prefill -21% TTFT `[estimate]` (single largest
   lever) > decode CPU/GPU share rebalance -15.2% decode > PLE lookahead -10.1%
   > kernel attention work (K scope).
5. Four probes run, three closed: D-C1 CPU credit cap -> NO-GO (share-shifting
   to staging costs 3x the join it saves); C2 PLE lookahead -> NO-GO (the
   interval is queue-hidden; forcing it synchronous costs 41% TTFT); K1
   decode-GU kernel measured at >=1.5x off the DRAM wall; P-1 layer-major
   sizing -> layer-wide single-pass variant, prototype gate stands (P-2 A arm
   recorded: 38.32 s / 78.0 GiB on a synthetic 8192 prompt).
6. Charter question 2 closed by measurement: production `router_wait` is queue
   depth (8 s/chunk), router work is 0.25 s/chunk.

## 1. S1 - baseline reproduction and A/A noise floor

### 1.1 A/A receipts (primed protocol)

Protocol: `tmp/q4e-sprint-r1-20261014/run.py <label> --prime` = standard probe,
production env (profile JSON, MTP=0, reserve 3584, chunk 4096, production
timing), prompts 6000 then 14000, generate 32, reuse 1.

| arm | 6K first (s) | 14K first (s) | 14K continue (tok/s) |
|---|---:|---:|---:|
| aa-p1 | 110.46 | 70.51 | 15.4997 |
| aa-p2 | 111.37 | 71.87 | 15.6195 |

Receipts: `tmp/q4e-sprint-r1-20261014/receipt-aa-ttft.json` (p1/p2 ratio
0.9811), `receipt-aa-decode.json` (ratio 0.9923). A/A spread: TTFT 1.93%,
decode 0.77%, warm 6K prefill 0.82%. The charter's +/-0.6% TTFT noise claim was
not reproduced with n=1 arms; use >=5% end-to-end gate as the decision line and
treat <2% TTFT deltas as noise until more arms exist.

### 1.2 First-request penalty (protocol hazard, new evidence)

`tmp/q4e-sprint-r1-20261014/scan_ttft.py` over all historic run logs: whenever
`context-14000` was the FIRST stream of a fresh worker, TTFT was 99-155 s
(139.62, 107.99, 120.39, 132.87, 134.22, 150.19, 132.53, plus this sprint's
155.67/154.42); with any prior request it was 67-72 s. Mechanism, from
`r1-diag-worker.log` (first chunk of the 6K request):

| chunk | submitted | linear_host | full_host | hc_host | unprinted |
|---|---:|---:|---:|---:|---:|
| cold pos=0 count=4096 | 99.14 s | 11.06 s | 19.72 s | 1.39 s | 47.5 s |
| warm pos=0 count=4096 | 19.75 s | 0.009 s | 0.060 s | 0.035 s | 0.013 s |

The +79.4 s lands in first-touch host submit intervals plus the unprinted
bucket. This is a warm-up/JIT-shaped cost, not a supply cost; it is the reason
historic TTFT "anomalies" appeared and why plain (unprimed) runners must not be
used for TTFT receipts.

Attribution (arm `r1-stages2`, stage-serialized, a 19-token warm-up request):

| stage | first forward | steady decode line |
|---|---:|---:|
| router (gate GEMV, top-k, MoE front) | 50,361.9 ms | 12.3-13.7 ms |
| full attention | 19,834.2 ms | 3.8-4.7 ms |
| linear attention (GDN) | 7,300.8 ms | 11.5-12.4 ms |
| hc | 1,246.7 ms | 18.4-19.7 ms |
| experts (MoE) | 1,315.9 ms | 30.3-54.7 ms |
| ple + shared + head | 19.2 ms | 16.7 ms |

The one-time cost sits almost entirely in the dense/attention first-use path and
repeats on every fresh worker process (>=5 samples today). Candidates: oneDNN
XMX joint-matrix primitive/JIT creation (profile sets `AILA_Q4E_PREFILL_XMX=1`),
or driver-side first-use codegen. Two discriminators were run, both fresh-worker
arms reading the same [Q4E Stages] line:

- `x1-xmx0` (`AILA_Q4E_PREFILL_XMX=0`, 79.83 s warm-up): penalty unchanged
  (router 47.1 s, full 19.7 s, linear 9.8 s, hc 1.24 s) -> NOT specific to the
  XMX joint-matrix dense path.
- `x2-syclcache` (`SYCL_CACHE_PERSISTENT=1`, `SYCL_CACHE_DIR`, `SYCL_CACHE_TRACE=1`,
  `AILA_KEEP_SYCL_CACHE_PERSISTENT=1`, forwarded via
  `AILA_WORKER_ENV_PASSTHROUGH=SYCL_CACHE_PERSISTENT;SYCL_CACHE_DIR;SYCL_CACHE_TRACE`):
  the worker died during load before serving any request (`PeekNamedPipe failed
  with Win32 error 109`, empty cache files). This reproduces the documented
  issue #4 (persistent-cache null dereference at the first JIT build,
  `src/runtime/ChildEnvironment.cpp:35-38`) on driver 32.0.101.8724 + oneAPI
  2026.1. The persistent SYCL/IGC cache is therefore NOT an available lever
  today.

Conclusion: the ~80 s is first-use JIT-shaped, not attributable to the XMX
path, and uncacheable with the current runtime. Practical mitigations:
(a) pre-warm one tiny forward at load (moves the cost into startup, keeps the
first user request clean), or (b) retry the persistent cache after a oneAPI
runtime upgrade. Do not enable `SYCL_CACHE_PERSISTENT` in production.

## 2. S2a - decode per-token cost table (retained20)

Source: `tmp/q4e-sprint-r1-20261014/r1-diag-worker.log`, 512 count=1 lines,
parsed by `tmp/q4e-sprint-r1-20261014/decode/parse_decode_fields.py`
(`decode/retained20-r1-diag.json`). Wall = `[Q4E ForwardWall] total_ms`.

| item | ms/token | share | scope / evidence |
|---|---:|---:|---|
| wall (total_ms) | 66.006 | 100% | prepare+backend+sample+drain |
| submitted_ms | 62.198 | 94.2% | whole backend host interval |
| router_wait_ms | 12.686 | 19.2% | queue residual at 48 top-k sync points (Backend.cpp:2147-2169) |
| experts_host_ms | 22.113 | 33.5% | RunResidentDecode coordinator interval |
| - cpu_join_ms (inside) | 10.001 | 15.2% | blocked in JoinResidentCold (18-thread CPU batch) |
| - experts non-join | 12.112 | 18.4% | classification, CPU job build, submits, copies |
| ple_host_ms | 6.652 | 10.1% | bimodal: p10 0.24, p90 7.68 |
| hc_host_ms | 6.217 | 9.4% | 2 hops/layer submits |
| linear_host_ms | 3.364 | 5.1% | 36 GDN layers |
| full_host_ms | 2.875 | 4.4% | 12 full-attention layers |
| head_host_ms | 0.149 | 0.2% | 1 LM-head GEMV submit |
| unprinted residual | 9.091 | 13.8% | router gate/top-k submits, shared-expert submits, hc_inject |
| sample_ms | 3.558 | 5.4% | greedy readback (blocking D2H) |
| prepare+drain | 0.114 | 0.2% | token H2D + final sync |
| supply mix | hits 225, misses 255, staged 0, H2D 0 | | COLD_MODE=2 -> all misses on CPU; cpu_compute 14.53 ms overlapped (~45 GB/s DRAM wall) |

Cross-check vs the retained17 window (role D: 64.40 ms total, router 12.59,
experts 21.04, join 10.22, ple 6.08, hc 5.95): every item reproduces within
3.5%, so the split is pair-stable.

## 3. S2b - prefill per-chunk cost table (retained20, 14K request)

Source: 4 `[Q4E AdaptiveWork]` lines with `pos=0/4096/8192/12288` of the 14K
request in `r1-diag-worker.log`.

| chunk (tokens) | submitted | router_wait | experts_host | - fill | - bank_wait | H2D GiB | staged experts |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 19.750 s | 7.727 | 11.905 | 7.298 | 4.065 | 45.7 | 19,360 |
| 4096 | 20.361 | 8.226 | 12.016 | 7.349 | 4.121 | 46.2 | 19,577 |
| 4096 | 20.427 | 8.389 | 11.920 | 7.345 | 4.031 | 46.5 | 19,692 |
| 1656 | 11.124 | 3.553 | 7.415 | 6.601 | 0.375 | 41.1 | 17,431 |
| total | 71.67 s (TTFT 71.82) | 27.90 | 43.26 | 28.59 | 12.59 | 179.5 | 76,060 |

Attention host submits are negligible at this scale (linear 9-10 ms, full
27-60 ms, hc 35-45 ms per chunk): the GDN/full-attention device work is hidden
inside router_wait and experts_host intervals, i.e. behind the queue. The
chunk-major GDN 4.07 s + full 2.85 s split from Strata C1 is a device-segmented
view; the `AILA_PROFILE_Q4E_STAGES=1` arm (`r1-stages2`) now re-measures it at
retained20 on 4096-token chunks (stage-serialized). Serialization does not
move the block wall (19.19-20.81 s vs 19.75-20.43 s unsynchronized), so the
block split is trustworthy; at decode scale the same instrumentation costs
28-54 ms/token (384 syncs per forward, measured as serialized minus
unsynchronized decode totals), so the serialized decode split is diagnostic
only:

| stage | chunk A | chunk B | chunk C | c1-era reference |
|---|---:|---:|---:|---:|
| experts (MoE) | 12,087.6 ms | 12,101.4 | 11,868.9 | 12,580 |
| linear attention (GDN) | 4,126.9 | 4,120.9 | 4,040.4 | 4,070 |
| full attention | 2,504.8 | 2,372.4 | 2,935.0 | 2,850 |
| hc | 475.9 | 299.0 | 287.7 | - |
| router (work, not wait) | 291.1 | 232.7 | 254.0 | - |
| ple / shared / head | 30 / 36 / 0 | 28 / 35 / 0 | 28 / 34 / 6 | - |

All three C1 figures reproduce within ~5%. The same arm settles charter
question 2: with per-stage sync, `router_wait_ms` collapses from 7,727-8,389 ms
to 3-21 ms while the router stage itself is only 169-291 ms per chunk. The
production 8 s `router_wait` is queue depth, not router compute.

## 4. S2c - "if X were free" wall-clock bounds

### 4.1 Decode (wall 66.01 ms/token -> 15.15 tok/s)

| X -> 0 | wall bound | share | oracle / method | correctness |
|---|---:|---:|---|---|
| cpu_join | -10.00 ms -> 17.9 tok/s | -15.2% | `AILA_Q4E_CPU_MAX_EXPERTS=4` (probe D-C1) | moves rows between engines -> needs raw-token re-check |
| experts non-join host | -12.11 ms -> 18.6 tok/s | -18.4% | none today `[estimate]` | math-preserving only with same accumulate |
| router_wait (device residual) | -12.69 ms -> 19.2 tok/s | -19.2% | no env; it is the device queue, not host work | diagnostic only, not a lever |
| ple_host | bound does not exist: `PLE_ASYNC=0` removes the interval but the wall stays inside A/A noise (section 6.3) | 0% | measured NO-GO | - |
| hc + linear + full submits | -12.46 ms -> 18.7 tok/s | -18.9% | none (needs kernel fusion, role K) | n/a |
| sample | -3.56 ms -> 16.0 tok/s | -5.4% | new test needed (head/greedy readback) | n/a |
| all host work (submitted - router_wait) | ceiling -49.51 ms; device-exposed floor >= 16 ms -> <= 41 tok/s | -75% | analytic | diagnostic only |

### 4.2 Prefill (14K wall 71.82 s)

| X -> 0 | wall bound | share | oracle / method | correctness |
|---|---:|---:|---|---|
| H2D staging volume (179.5 GiB) | ~53.9 s `[estimate]` (-25%) | -25% | Strata C1 b04 skip-memcpy arm: 52.71 vs 70.17 s (c1-era, routing also changed -> coarse) | breaks output |
| layer-major re-transmission (179.5 -> ~52 GiB) | 55.3 s `[estimate]` (-21%) | -21% | P-1 export + `layer_major_floor.py`; roadmap claims -13..-18% | math-preserving target |
| router_wait (27.90 s total) | -38.8% | -38.8% | device queue residual, not host work | diagnostic only |
| GDN + full-attention device (c1: 6.92 s/chunk) | ~24 s | -33% | needs STAGES arm to re-measure | kernel scope, role K |
| host fill / bank_wait | 0 | 0% | measured conservation (C1/C7): parallel fill moves time into bank_wait | n/a |

Rows are not additive. P-1's floor computation (per-layer union 491/510/512
experts -> about 1.21 GiB/layer) shows the layer-major window does not fit in
16 GB VRAM; the variant that survives is the layer-wide single-pass MoE.

## 5. Candidate ranking (upper bound x feasibility)

| rank | candidate | bound | engineering | verdict |
|---:|---|---|---|---|
| 1 | layer-major prefill (P) | -21% TTFT `[estimate]` | multi-day | GO: P-1 sized (union 495/layer -> layer-wide single-pass); prototype gate = >=30% H2D cut AND >=5% wall on an 8192 pair |
| 2 | decode CPU/GPU share rebalance (D C1) | -15.2% decode | S probe / M adopt | NO-GO for the credit-cap form (section 6.1); the -15.2% bound is not reachable by moving rows to staging |
| 3 | PLE lookahead (D C2) | 0% (bound falsified by section 6.3) | - | closed |
| 4 | kernel attention (K: 12x K/V re-read at decode) | unquantified | probe-only today | K1/K2 probes written, not run |
| 5 | service batching (S) | ~0: `SupportsDecodeBatch()==false`, batch key null, requests rotate serially | - | NO-GO as a 16 GB single-GPU lever |
| 6 | MTP per session | +2.9% on its only legal shape | - | deferred (below gate) |

Adopted this session (opt-in, default unchanged): the qwen4exp load-time warmup
(`AILA_INIT_WARMUP=1`, section 6.7) -- first large request -53%, startup +111 s,
throughput neutral; pair `retained21`.

Section 2 negative list respected: no SLM reuse, no 8-token, no swap, no
staging-for-CPU swap when PCIe is the scarcer resource (626 MiB/token would
cost ~66 ms vs 13.9 ms CPU service).

## 6. Selected probes (S3) and results

### 6.1 D-C1: decode CPU credit cap

Command: `python tmp/q4e-sprint-r1-20261014/run.py dec-cpu4-b1 --prime --diag
--generate=64 --AILA_PROFILE_Q4E_DECODE=0 --AILA_Q4E_PROFILE_COMPONENTS=0
--AILA_Q4E_CPU_MAX_EXPERTS=4`

Fields: `cpu_join_ms`, `experts_host_ms`, `misses`, `total_ms`.
Decision rule: >=5% median `total_ms` cut with pairs won, else no-go.
Result (arm `dec-cpu4-b1`, n=256 decode lines,
`decode/retained20-dec-cpu4.json`): **NO-GO**.

| field | baseline (r1-diag) | cap=4 | delta |
|---|---:|---:|---:|
| total_ms (wall) | 66.006 | 95.518 | +44.7% |
| submitted_ms | 62.198 | 91.722 | +29.5 |
| router_wait_ms | 12.686 | 25.436 | +12.7 |
| experts_host_ms | 22.113 | 40.641 | +18.5 |
| cpu_join_ms | 10.001 | 2.960 | -7.0 |
| fill_ms | 0.000 | 24.405 | +24.4 |
| staged experts | 0 | 55 | +55 |
| H2D MB/token | 0 | 132.9 | +132.9 |
| misses / hits | 255 / 225 | 228 / 252 | - |
| 14K continue rate | 14.97-15.62 tok/s | 9.40 tok/s | -37% |

Mechanism: capping the CPU batch pushes ~55 experts/token onto the GPU staging
path; the added arena->bank memcpy (24.4 ms) plus the additive 133 MB/token
H2D (~14 ms at 9.2 GB/s) costs more than the 7 ms join saving, and the deeper
queue doubles router_wait. This is the explicit counter-example the role-D
doc predicted: the cpu_join bound in 4.1 is NOT reachable by moving rows to
staging. TTFT was unchanged (71.46 s), confirming the knob is decode-only.
Follow-up direction: hide the join instead of shrinking it (PLE lookahead C2,
or raise the resident hit rate), not share-shifting.

### 6.2 P-1: layer-major union and window sizing

Command: `python tmp/q4e-strata-c1/run_ttft14k.py p-lm-sizing
AILA_Q4E_HOTSET_EXPORT=E:/RiderProjects/Aila/tmp/q4e-sprint-r1-20261014/prefill/ttft14k-export.profile`
then `layer_major_floor.py` on the export.
Result (arm `p-lm-sizing`, TTFT 72.04 s -> run within A/A noise):

| metric | value |
|---|---:|
| keys with prefill>0 | 23,716 of 24,576 |
| per-layer union (min/median/max) | 466 / 495 / 512 |
| layer-major H2D floor | 55.98 GiB = 60.11 GB -> 6.53 s @ 9.2 GB/s |
| current per-request H2D | 179.5 GiB -> 20.95 s |

Decision per the role-P rule: median union 495 >= ~490 -> a per-layer retention
window does NOT fit the VRAM headroom; the surviving variant is the layer-wide
single-pass MoE. Expected wall if H2D scaling were the only change:
71.8 -> ~57.4 s (-20%) `[estimate]`, matching the role-P -21% figure.
Prototype gate stays: >=30% H2D cut AND >=5% wall on a two-chunk (8192) pair.
P-2 was then run as the prototype's A arm (`p2-synth8192`; the frozen fixture
has no 8192 target, so the probe fell back to its deterministic synthetic
prompt -- the pair must be A/B'ed on the same prompt):

| request | prompt tokens | TTFT | note |
|---|---:|---:|---|
| warmup (no fixture) | 19 | 80.65 s | one-time penalty, section 1.2 |
| context-6000 | 5,924 | 31.77 s | ~5.4 ms/token |
| context-8192 | 8,116 | **38.32 s** | two 4096 chunks |

H2D confirmation from the serialized arm on the same prompt: 40,455.6 MiB
(chunk 1) + 39,450.1 MiB (chunk 2) = 79,905 MiB = 78.0 GiB for 32,059 staged
expert uploads. A scoped `HOTSET_EXPORT` on the same 8192 prompt
(`p2-union8192`, export written at teardown) measures the layer-major floor:
20,761 keys with prefill>0, per-layer union min/median/max = 275/462.5/488,
49.00 GiB = 5.72 s at 9.2 GB/s. Against the measured 32,059 uploads / 78.0 GiB
this is a **-35% H2D cut at the floor, so the two-chunk prototype passes the
-30% gate geometrically** (earlier back-of-envelope -23..-28% assumed
near-full 480-512/layer unions; this content routes more sharply). Expected
wall if only fill+bank scales: -10..-19% on the 39.4 s window `[estimate]`,
above the +5% wall gate. The 14K window remains the big prize (floor 23,716
uploads vs 76,060 today, -69%).

### 6.3 C2: PLE handoff decomposition (`AILA_Q4E_PLE_ASYNC=0`)

Reasoning before the arm: the C2 lookahead bound is the whole `ple_host_ms`
interval (6.65 ms/token, 10.1%); `PLE_ASYNC=0` moves the gather onto the
calling thread and therefore splits inline gather vs handoff.
Result (arm `dec-ple-sync-b1`, n=256, `decode/retained20-ple-sync.json`):
**NO-GO for C2's premise.**

| field | baseline (r1-diag) | PLE_ASYNC=0 | delta |
|---|---:|---:|---:|
| ple_host_ms | 6.652 | 0.246 | -6.41 |
| submitted_ms | 62.198 | 55.066 | -7.13 |
| total_ms (wall) | 66.006 | 65.547 | -0.46 (-0.7%) |
| 14K continue rate | 15.50 (A/A) | 15.516 | +0.1% (A/A spread 0.8%) |
| 14K TTFT | 70.51/71.87 | 101.50 | **+41%** |
| 6K TTFT | 110.46/111.37 | 138.95 | +25% |

Reading: the whole 6.65 ms `ple_host_ms` interval is already hidden behind the
GPU queue in the production shape -- deleting it moves `submitted_ms` by
7.1 ms but leaves the wall inside A/A noise, so the -10.1% bound of section
4.1 does not exist for C2. Forcing the gather inline also costs ~30 s of
prefill TTFT (about 7.5 s per 4096 chunk), i.e. the async worker thread is
hiding a large *prefill* gather cost that the stage table attributes as 28 ms
because it is overlapped. Consequences: C2 is dropped (rank 3 -> closed), and
the general rule "a host interval on the submitted_ms path is not necessarily
savable" now has three instances (router_wait, ple_host, and the C1 join
counter-example). Do not set `AILA_Q4E_PLE_ASYNC=0` in production.

### 6.4 Bonus: K1 decode-GU kernel attainment (closes a measurement gap)

`tmp/q4e-sprint-r1-20261014/kernel/probe-K1-decode-gu.ps1` ran (GPU free,
no Worker/Server): median 0.123427 / 0.123430 ms across the A/A pair (0.002%
spread, `decode-gu-A1.log` / `decode-gu-A2.log`). Per the role-K decision
table this is >=1.5x off the DRAM wall (>=0.0871 ms) -> a variant study is
warranted under the standing kernel gate (>=3% at two shapes, bit-exact), and
the decode-MoE line is no longer unmeasured.

### 6.5 Production-parity arm (launcher env that the runners omit)

`tools/server/start-qwen38.ps1:25-34` sets two knobs that neither the profile
JSON nor the standard runners carry: `AILA_Q4E_STAGE_COPY_PARALLEL=1` and
`AILA_Q4E_4TOKEN_WGSHARE=4`. Arm `d1-prodparity` (primed 14K, generate 32,
worker log live) adds them:

| metric | A/A baseline (aa-p1/p2) | d1-prodparity | delta |
|---|---:|---:|---:|
| 14K TTFT | 70.51 / 71.87 | **69.09 s** | -3.0% vs A/A median |
| 6K warm-up | 110.46 / 111.37 | 109.93 | -0.9% |
| 14K continue rate | 15.50 / 15.62 | 15.55 | unchanged |
| fill_ms per 4096 chunk | 7.30-7.35 | 2.39-2.44 | -4.9 s |
| bank_wait_ms per chunk | 4.03-4.12 | 8.07-8.17 | +4.0 s |
| fill+bank per chunk | 11.3-11.5 | 10.46-10.61 | -0.9 s |

Reading: the parallel host copy moves the fill into bank_wait (conserved plus a
real ~0.9 s/chunk saving), consistent with the historical -2.94% wgshare TTFT
screen. Consequence for this sprint: the 70.5-71.9 s "baseline" is the
untuned-runner shape; true launcher production is ~69.1 s on the same pair.
Future TTFT receipts should either use the launcher env or state that these two
knobs are off. Decode is unaffected (both knobs are prefill-only).

### 6.6 Warmup wiring check (`AILA_INIT_WARMUP=1`)

An init-warmup facility already exists (`src/engine/EngineLoad.cpp:438-680`,
`AILA_INIT_WARMUP`, manifest "2-token prefill + 1-token decode + reset"). Arm
`x3-warmup` sets `AILA_INIT_WARMUP=1` on the production worker:

| metric | baseline (x1/x2 warm-up) | x3-warmup |
|---|---:|---:|
| [Warmup] log markers | 0 | **0** |
| worker startup | 134.9 / 132.8 (p2/x3) | 132.8 s |
| 19-token first request | 79.83 / 80.94 | 80.47 s |
| first-forward stage split | router 47.1-50.4 s | router 50.3 s |

Root cause (source): the warmup lives in `InferenceEngine::init`, while the
production worker loads the model through `ModelFactory::LoadQwen4Exp`
(`src/worker/InferenceWorker.cpp:569`). The worker never enters that code path,
so the knob is inert for the HTTP/SSE serving stack. Implementation pointer for
the next session: mirror the DiffusionGemma "full-canvas" pattern inside the
qwen4exp factory load, with shape coverage that matches production (a
4096-row prefill *and* a decode step; a 19-token warmup does not cover chunk
shapes -- a 512-token request still paid +4 s after one), and reset the
KV/prefix state plus `SetColdEpochEnabled(false)` exactly as the legacy path
does.

### 6.7 Qwen4Exp load-time warmup -- implemented and measured (opt-in)

Implementation (`src/inference/Model.cpp`, end of `LoadedModel::Load`, pair
`retained21`): after the model is published, if `AILA_INIT_WARMUP=1` (default
off) and `AILA_Q4E_HOTSET_EXPORT` is empty, a scratch session runs
`ForwardLogits` on one production-shaped chunk (`AILA_INIT_WARMUP_ROWS`,
default 4096, varied token ids) plus one decode step, then `Reset()`. Cold
epoch is disabled during the warmup and restored after; the result is logged as
`[Q4E Warmup] rows=4096 prefill_ms=... decode_ms=... total_ms=...`.

Arms (all with the production knobs `STAGE_COPY_PARALLEL=1` + `WGSHARE=4`):

| metric | d1-prodparity (no warmup) | w1 (dummy ids) | w2 (varied ids) |
|---|---:|---:|---:|
| [Q4E Warmup] total | - | 96.37 s | 97.81 s |
| - prefill / decode | - | 96.25 / 0.085 s | 97.67 / 0.098 s |
| worker startup | 132.8 s | 247.0 s | 243.4 s |
| 6K first request | 109.93 s | **51.45 s** | **51.16 s** |
| 14K first request | 69.09 s | 69.56 s | 69.37 s |
| 14K continue rate | 15.55 | 15.19 | 15.37 |
| chunk-1 ple_host (6K) | 0.034 s | 20.70 s | 20.45 s |

Committed pair `retained22` (commit `a3ee21e`, rebuilt) then ran a smoke arm on
the synthetic 6K prompt (`w3-committed-smoke`): `[Q4E Warmup]` 96.87 s,
startup 244.4 s, 19-token first request **1.48 s** (vs 80.5 s without warmup),
6K first request **30.41 s** == steady state, `ple_host` residual 0. So the
~20.5 s residual seen on the fixture prompt is content-dependent (natural-text
gather), not structural: it does not appear for the synthetic corpus.

Paired control on the same committed binaries (`w4-committed-control`, no
warmup, same prompt): startup 133.4 s, 19-token first request 84.59 s, 6K first
request 30.61 s. Net accounting on the synthetic corpus:

| protocol | startup | first request | startup + first request |
|---|---:|---:|---:|
| no warmup (w4) | 133.4 s | 84.59 s | 218.0 s |
| warmup (w3) | 244.4 s | 1.48 s | 245.9 s |

i.e. the warmup trades +111 s of startup for -83 s of first-request latency on
this corpus (fixture corpus: 110 -> 51 s, +23% end-to-end to first content).
The lever is a latency-placement knob, not a throughput win.

Reading:

- The warmup removes ~58 s of the ~110 s first-request cost: the first large
  request drops to 51 s while 14K, continuation decode, and steady chunks stay
  inside A/A noise. CPU/ABI suite 20/20 passes.
- The residual 20.5 s is `ple_host` on the first real chunk only (0.5 ms from
  the second chunk on). The warmup's own chunk shows 37 ms because its async
  PLE gather overlaps the JIT storm; once JIT is pre-paid, the ~20 s gather is
  exposed. Varied ids (w2) do not change it, so it is not id locality. This is
  a NEW prefill finding: the first chunk of a fresh session carries a ~20 s PLE
  gather, normally hidden behind first-use JIT. Follow-up owner: prefill/PLE
  (issue the gather earlier, or overlap it with the first chunk's H2D).
- Trade-off: time-to-first-token from process start is 242.7 s (control) vs
  298.2 s (warmup, +23%), while the first user request is -53%. For a
  pre-launched server this is the intended trade; for one-shot CLI it is a net
  loss. The knob is therefore opt-in and production default is unchanged.

### 6.8 K2 QSA route probe -- repaired and measured

`--qsa-probe` now runs (role-A3 fix in `tests/ops/Qwen4ExpKernelTests.cpp`;
`[QSA COST]` rows in `tmp/q4e-sprint-r1-20261014/kernel/qsa-cost-curve-A1.log`).
The earlier "crash after banner" was a silent 17 s no-output window plus a
probe-script exit-code parsing bug (fixed in `probe-K2-qsa-route.ps1`).

| length | sparse | dense_attention_ms | routed_post_projection_ms | route net % |
|---:|---:|---:|---:|---:|
| 2048 (control) | 0 | 0.25 | 0.31-0.33 | -24 to -33% |
| 2051 (control) | 0 | 0.27 | 0.31-0.33 | -17 to -23% |
| 2052 (gate) | 1 | 0.26-0.27 | 0.34 | -26 to -29% |
| 4096 | 1 | 0.37 | 0.35-0.37 | +0.6 to +5.4% |
| 8192 | 1 | 0.56 | 0.36 | **+36.3 to +37.0%** |

Reading: below the 2052 gate the sparse route is strictly worse (short dense
attention is cheaper); at 4096 it is marginal and round-dependent; at 8192 it
is decisively faster. Per the role-K decision table this is not ">=3% at both
lengths", so the cross-head-sharing lever stays closed for the 4096-chunk
production shape -- but the routed path is a natural pairing with any 8192-row
window (layer-major), where it removes ~37% of that attention step.

### 6.9 PLE first-chunk gather fixed (F1 `PrefetchVirtualMemory`)

Root cause (role-A2, `docs/plans/qwen38-r1-ple-firstchunk-2026-10-15.md`): the
~20 s is mmap demand paging of the PLE tables (29.8 GiB across shards
00005-00037), not ngram math or H2D. A 4096-token chunk faults ~131k random
pages (80 B packed + 20 B absmax per head row); the same pattern costs 27.6 s
cold vs 0.048 s resident (575x, CPU probe), and batching
`PrefetchVirtualMemory` in groups of 64 gives 12.2x (186.9 -> 15.3 us/access).
The async worker normally hides the gather behind the JIT storm (which is why
cold-start numbers never showed it) and the chunk lookahead hides chunks 2+;
only the first chunk of a fresh session exposes it.

Fix (`src/models/Qwen4ExpPle.{hpp,cpp}` via role-A2 patch, env
`AILA_Q4E_PLE_PREFETCH`, default on): prefetch the head-row pages before
`dequant_to_staging`.

A/B/B/A on the fixture 6K first request (load warmup on, production knobs):

| arm | prefetch | 6K first request |
|---|---:|---:|
| pa | 0 | 50.48 s |
| pb | 1 | 31.83 s |
| pb2 | 1 | 31.77 s |
| pa2 | 0 | 50.83 s |

Median ratio 1.589 (-18.7 s, -37%), far above noise; the first request now
equals the steady-state 6K time (30.5 s). Receipt:
`tmp/q4e-sprint-r1-20261014/receipt-ple-prefetch-ab1.json` plus
`ple_summary.py`. Combined with the load warmup the fixture first request goes
110 s -> 31.8 s. This is an adoption-grade result (>=+5% gate) and the patch
ships enabled by default.

### 6.10 Layer-major prototype: iterating (v1 -> v2 -> v3)

Gate: on an 8192-token prompt (exact: target 8268 -> 8192 server tokens),
chunk-major 2x4096 baseline A' = **37.06 s / 78.15 GiB H2D / 33,108 uploads**
(`lm-a1`, production knobs). Prototype = chunk 8192 + `AILA_Q4E_LAYER_MAJOR=1`.

| revision | TTFT | H2D | staged | per-layer sums (48 layers) | verdict |
|---|---:|---:|---:|---|---|
| v1 (`lm-b2`) | 163.34 s | 20.2 GiB | 8,349 | bank_wait 107 s | H2D -74% but 4.4x slower |
| v2 (`lm-b3`) | 61.95 s | 19.7 GiB | 8,349 | experts 22.4 s, bank_wait 20.8 s, fill 1.05 s, residual 16.9 s | mechanism holds, wall still +67% |
| v3 (`lm-b5`) | 62.30 s | 19.7 GiB | 8,349 | per-layer wall sum 61.57 s, median 1.11 s, max 3.55 s (late layers) | per-half refactor did not move the wall |

The mechanism (one staging per layer per window) is validated: H2D and uploads
drop ~75% versus the chunk-major window. The remaining gap is decomposed in
L28: an ~17 s residual that matches role-A1's residual risk 2 (all capacity
scratch is up-sized to 8192 rows), plus a 6-8x expert-kernel shape penalty.
v3 (per-half 4096-row capacity, bounded stages) is commissioned; the 32-slot
knob currently faults and is part of v3 (L29).

v3 landed (per-half 4096-row capacity, 8192-row carries of 1.15 GiB, scratch
back at 4096, the 32-slot host-copy fault fixed) and did **not** move the wall
(62.30 s). Per-layer walls grow through the sweep (2.6-3.6 s on the last
layers), i.e. each layer's staging and expert kernels form a barrier with no
cross-layer overlap. Verdict: the layer-major *mechanism* (H2D -75%) is
validated, but the window form is a wall regression (-67%) and stays
opt-in/default-off (`AILA_Q4E_LAYER_MAJOR`), committed as a documented
experiment (commit `9abb858`, pair `retained23`). The next step is a
cross-layer pipelined variant (stage layer L+1 while layer L computes), not a
fourth shape tweak.

### 6.11 Day-2 extras: hotset profile A/B, and the plan for the rest

Hotset profile A/B on the primed 14K protocol (`prof-h4` vs `prof-a6k`, both
under concurrent sub-agent CPU load, production knobs):

| metric | H4 (control) | agent6k | delta |
|---|---:|---:|---:|
| decode total_ms (median, n=128) | 71.46 | 69.32 | **-3.0%** |
| experts_host_ms | 26.62 | 24.86 | -6.6% |
| cpu_join_ms | 9.39 | 8.55 | -8.9% |
| misses / hits | 236.5 / 243.5 | 206.5 / 273.5 | misses -12.7% |
| continue rate (run level) | 12.87 | 14.24 tok/s | +10.6% (noisy) |
| first_content 14000 (run level) | 69.69 s | 69.92 s | +0.3% (noise) |

Receipts: `receipt-profile-ab1.json` (decode medians), plus
`receipt-prof-a6k-ttft.json` / `receipt-prof-a6k-dec.json` (run level, n=1/arm;
the run-level numbers were taken under concurrent sub-agent CPU load). The
per-token median (+3.1% throughput) is below the >=5% default gate, so the
agent6k profile stays opt-in; the run-level +10.6% is under load and should be
re-measured quiet before any launcher change.

### 6.12 Service probe S2: sessions=2 rotates, it does not batch

First clean run: `service/run-s2-a3` (16:06:46-16:11:11, production knobs, H4
profile, `--sessions 2 --decode-batch-size 2 --context 4096 --chunk 4096`, two
2472-token prompts above the QSA gate, 32 decode rows each). Probe gates
passed; 0 alloc failures; `peak_active_requests=2`; `small_multi_rows=[]`; the
decode fingerprint is 59 rows / 30 unique positions / **29 adjacent duplicate
positions** (2484, 2484, 2485, 2485, ...) -- the two sessions strictly
alternate one token per quantum.

| metric | value |
|---|---:|
| aggregate decode rate (pair window 4.82 s) | **12.85 tok/s** |
| aggregate (overlap window) | 13.00 tok/s |
| single-session reference (retained20) | 15.50-15.62 tok/s |
| decode submitted median (n=59) | 70.8 ms/token |
| experts_host / cpu_compute / cpu_join medians | 31.8 / 22.4 / 15.8 ms |
| misses per decode row | 340 (single-session ~236) |

Reading: aggregate throughput with two concurrent sessions is BELOW the
single-session rate. The scheduler rotates whole forward quanta on one submit
thread (`SupportsDecodeBatch()==false` for this backend), and the interleaved
working set adds ~10% per-token cost plus ~44% more misses. The S-doc
analytic bounds (serial 15.6 / shared-step 21.1 / overlap 32.4) collapse to the
serial case; continuous batching for this model needs the native multi-row
decode path first. Startup to health 151 s; first content 92.5 s / 105.2 s
(first-use JIT plus strict serialization of the two prefills).

Independent replication (window A, `service/run-s2-wina1`, retained23 binaries,
production knobs): passed with the same fingerprint shape -- 58 rows / 29
unique positions / 29 adjacent duplicates (2484, 2484, 2485, 2485, ...) --
aggregate **14.29 tok/s** (per session 7.14 / 1.83), decode submitted median
63.8 ms (p90 70.3). Both windows therefore agree: no batching gain, rotation
only.

Non-claims: the two earlier attempts (`run-s2-a1` 15:54, `run-s2-a2` 15:55)
failed closed with `MEM_LARGE_PAGES ... win32_error=1450` because a stale
queue started them while the previous worker still held the GPU; they are
invalid as measurements (section 8, L35).

### 6.13 Production agent sequence smoke at retained23

`agent-full-r23` (16:04-16:06, 392 s wall, exit 0, `passed=true`, production
timing, production knobs): PI tool round trip PASSED; cancel-recover PASSED
(after-cancel 1.19 s); context-6000 30.53 s; context-6000-continue 1.12 s
(cached 5976); context-14000 69.53 s; context-14000-continue 1.07 s (cached
13976). The PI round ran before the context measurements and paid first-use
JIT, which is why the 6000 arm reads 30.5 s here while a truly cold first
request stays the 110-116 s case (section 6.11, `prof-h4`).

### 6.14 Decode regression scare closed (quiet-h4)

`quiet-h4` (16:11-16:17, production knobs, same primed protocol, quiet box):
6000 112.94 s; 6000-continue **15.20 tok/s**; 14000 69.37 s (best primed
reading so far); 14000-continue **15.43 tok/s**. Retained20 references are
15.50/15.62 tok/s. Conclusion: the 12.9-14.2 tok/s continues seen in
prof-h4/prof-a6k/agent-full were concurrent sub-agent CPU load, not a code
regression; retained23 decode equals retained20 within noise.

### 6.15 decode-attn bench first runs (K3 arm, probe R3-A1/A2)

`AilaQwen4ExpKernelTests.exe --decode-attn-bench` (bf16, D256, Hq24/Hkv2, 24
q-heads / 2 kv-heads, 20 reps per shape). All guards pass in both runs:
finite; 2-row duplicate rows bitwise equal; two-head f64 reference max_abs
1.42-1.45e-05 (budget 2e-4).

| length | rows=1 us/row (A1/A2) | rows=2 us/row (A1/A2) | rows=2 total ms |
|---:|---|---:|---:|
| 4096 | 380.5 / 352.1 | 316.8 / 304.4 | 0.609-0.634 |
| 8192 | 571.1 / 570.6 | 527.6 / 528.2 | 1.055-1.056 |
| 14336 | 844.9 / 838.0 | 871.4 / 869.3 | 1.739-1.743 |

A/A spread is <=0.2% at 8192/14336 and 4-8% at 4096 rows=1. The 2-row arm
buys 10-17% at 4096/8192 and nothing at 14336. The 14336 rows=2 total
(1.74 ms) against the 101 us/row min-byte floor (K audit) sets the variant
gate: a pair-reuse kernel needs <=792 us for the same call to clear 2x.

### 6.16 GDN prefill variant (role B5) -- closed below the e2e gate

New standalone bench `AilaQwen4ExpGdnBench.exe` (no model file) reproduces the
K4 anchor (74.585 vs 74.817 ms at 4096 rows, -0.31%) and tests the SLM-resident
variant (`src/ops/GdnPrefillVariant.*`) at r_slm = 0/56/120:

| rows | baseline | variant r=0 | r=56 | r=120 |
|---:|---:|---:|---:|---:|
| 4096 | 74.59 ms | 71.14 (-4.6%) | 120.09 (-61%) | 138.68 (-86%) |
| 8192 | 149.06 ms | 142.18 (-4.6%) | 239.34 (-61%) | 276.83 (-86%) |

All arms are bitwise-identical (parity pass), so the loss is purely
performance: keeping state rows in SLM collapses occupancy/bank behavior, and
even the r=0 rewrite only reaches -4.6% at the kernel, which projects to
-0.59% TTFT (GDN kernel = 3.58 s of the 19.7 s chunk). Verdict: NO-GO; the
8x kernel headroom K4 saw is not reachable with SLM residency. Keep the bench
and variant as reference assets; no production path touched.

### 6.17 Decode residual closed: the 9.09 ms "unprinted" bucket decomposed

Role-B6 added host-side sub-timers to `[Q4E AdaptiveWork]` (plus an
event-gated device-spans arm behind `AILA_Q4E_DECODE_STAGE_EVENTS`, whose queue
enabler now exists in `src/inference/Model.cpp`). Arm `dec-instr-a1`
(primed 14K, n=128 decode lines):

| term | median ms | share of token |
|---|---:|---:|
| moe_front (gate GEMV submits) | 1.776 | 2.9% |
| topk_submit | 0.912 | 1.5% |
| **shared_submit (shared-expert submits)** | **3.732** | 6.1% |
| shared_add | 0.625 | 1.0% |
| router_other | 1.934 | 3.2% |
| hc_inject / shared_other / tail | 0.017 | ~0% |
| **sum (was the 9.09 ms residual)** | **8.997** | 14.8% |

The old residual reproduces to within 1% and is dominated by the shared-expert
submit path (3.7 ms = 78 us/layer) plus router bookkeeping -- i.e. host submit
overhead, not device work. Follow-ups: fuse/batch the shared-expert submits or
move them off the per-layer critical path. The device-spans arm (`=1`) is
available for the next decode study.

Device-spans arm (`dec-instr-b2`, `AILA_Q4E_DECODE_STAGE_EVENTS=1`, n=128;
markers add submissions, so the run-level rate is diagnostic only):

| device span | median ms/token | share |
|---|---:|---:|
| **experts_dev** | **27.49** | 38.6% |
| misc_dev (unattributed phases) | 11.76 | 16.5% |
| hc_dev | 9.70 | 13.6% |
| linear_dev (GDN) | 7.74 | 10.9% |
| full_dev (QSA/full attention) | 5.37 | 7.5% |
| shared_dev | 3.54 | 5.0% |
| head_dev | 3.38 | 4.7% |
| router_dev | 1.94 | 2.7% |
| resid_dev | 0.31 | 0.4% |
| **sum** | **71.23** | wall ~74 ms (rate 13.5 tok/s, diagnostic) |

This re-frames the decode model: the device spans account for ~96% of the
token wall, so every host-side interval (router_wait, cpu_join, shared_submit,
ple_host) is hidden behind GPU work -- decode improvements must come from the
GPU side. Priority order is now: expert kernels (27.5 ms, ties to the K1
decode-GU bench at >=1.5x off the DRAM wall), the 11.8 ms misc bucket (needs
one more naming pass), then hc/linear/full. The earlier "host-bound" reading
was an artifact of the submitted_ms accounting; this arm is the correction.

### 6.18 Startup: arena fill prefetch cuts worker startup 135 -> 78.6 s (-42%)

Role-B4 applied the F1 technique (batched `PrefetchVirtualMemory`, <=64/ranges)
to the 51.46 GiB host-arena fill, gated `AILA_Q4E_HOST_ARENA_PREFETCH`
(default on). A/B/B/A on `startup_only.py` (no requests, production profile):

| arm | prefetch | worker startup |
|---|---:|---:|
| a1 | 0 | 133.8 s |
| b1 | 1 | 78.6 s |
| b2 | 1 | 78.5 s |
| a2 | 0 | 135.4 s |

Median ratio 1.723, **-56.8 s (-41.9%)**, 4/4 pairs to prefetch=1. Load timing
localizes it: `arena RAM copies 95.6 s -> 26.9 s` (1992 -> 561 ms/layer,
578 MB/s -> ~2.05 GB/s); the remaining gap to the 2.8 GB/s single-file
sequential ceiling is the next startup target. Receipt:
`receipt-startup-prefetch-abba.json`. Combined with the opt-in load warmup a
pre-launched server now reaches ready in ~175 s instead of ~245 s.

Production-shaped smoke on the committed pair (`b4-smoke`, `--production-timing`
plus `AILA_INIT_WARMUP=1` + production knobs): worker startup **175.4 s**
(79 s load + 96 s warmup), then the probe warmup request 1.46 s, the 436-token
request 5.06 s (small shape residual; the load warmup covers 4096-row shapes),
continue 1.14 s with prefix reuse. The serving stack therefore runs
load-prefetch + opt-in warmup together without regressions.

End-to-end anchor on `retained25` (`r25-anchor`, primed 14K): worker startup
**83.3 s**, 14K TTFT 69.56 s (unchanged band), first 6K request 113.4 s (JIT),
continue 13.53 tok/s -- the decode number is CPU-contention contaminated
(three review agents were active); the quiet-box reference stays quiet-h4 =
15.43 tok/s, and the arena prefetch is load-path-only code.

### 6.19 Decode `misc_dev` decomposed: layer-boundary host lag is the target

Role-C2 split the 11.76 ms `misc_dev` bucket into named markers and verified
with a self-check (`misc_dev == 0.000` everywhere). Arm `c2-round2-b1`
(n=128):

| new bucket | median ms | meaning |
|---|---:|---|
| **inter_dev** | **8.858** | previous layer's resid marker -> this layer's PLE marker: GPU idle while the host finishes the boundary work and submits ahead |
| stagein_dev | 2.106 | `StageResidentInput` memcpy window |
| ple_dev | 0.261 | PLE device span (the host-lag reading lives in inter_dev, not here) |
| prerouter_dev | 0.066 | pre-router gap |
| sum | 11.291 | vs the previous 11.762 (within the +/-0.5 tolerance) |

The C2 prediction (ple 7.1 / stagein 3.8) did not hold; the measurement puts the
bulk in **layer-boundary host lag (8.86 ms = 12% of the token)**. Combined with
`stagein_dev`, the actionable ceiling is ~11 ms/token of GPU idle that a
deeper-submitted queue or a cheaper layer-boundary host path could recover.
Next candidates (from C2's doc section 5): move/eliminate the
`StageResidentInput` memcpy, try `AILA_Q4E_CPU_DEFER_ARENA_WAIT=1`, and keep
the host further ahead of the in-order queue.

### 6.20 Decode MoE kernel line closed (C1) + the guard-bug caveat

Role-C1 built `AilaQwen4ExpDecodeMoeBench.exe` (production geometry, bitwise
parity vs the production nested kernels) and ran it (A1):

| arm | gu (10 experts, rows=1) | down |
|---|---:|---:|
| production reference | 0.1165 ms | 0.0661 ms |
| best variant (fixed-256 scale) | 0.1085 ms (-6.9%) | 0.0614 ms (-7.1%) |
| array-of-descriptors form | 0.1518 ms (+30%) | 0.1088 ms (+65%) |

All parity checks are 0-mismatch (bitwise). Using the live token mix
(hits ~186/token => E_g ~3.9/layer), C1's arithmetic puts the whole decode
MoE *kernel busy* at 3.6-5.1 ms/token (15-25% of `experts_dev` 27.5 ms); the
rest of that bucket is CPU join/publish/submit bubbles. So a 7% kernel gain is
~0.3 ms/token (~0.4% wall) -- **the decode MoE kernel line is closed below the
gate**, and the lever returns to the CPU/GPU scheduling side (aligning with
inter_dev in section 6.19).

Caveat: the bench's sentinel guard aborts at the end of each shape
(`sentinel_ok=0`), so the chain busy/gap section did not run; the guard
compares a region that later arms legitimately overwrite (buffer reuse), i.e.
a bench bug, not an overflow. Recorded for the next bench owner.

### 6.21 Startup round 2: async hint overlap (-26%) and the trim target

Role-C3 (delivered posthumously after a 429) moved the arena page hints onto a
helper thread so layer L+1's hint overlaps layer L's fill, and added the
residual-split timing. Clean A/B/A via `startup_only.py`
(`AILA_Q4E_HOST_ARENA_PREFETCH=1` control vs `+..._ASYNC=1`):

| arm | startup | notes |
|---|---:|---|
| A (sync hint) | 79.1 s | PFV enqueue 22.6 s blocking |
| B (async hint) | 58.5 s | PFV enqueue ~0.4 s; stage memcpy 3.0 / wait 0.9 / nested 0.2 s |
| A (sync hint) | 78.6 s | reproduces |

Adopted: `AILA_Q4E_HOST_ARENA_PREFETCH_ASYNC` now defaults on (commit
`626fada`, pair `retained26`); the knob set to 0 restores the synchronous
behaviour, and explicit-read opt-ins bypass the hint path either way.
Production-shaped smoke on `retained26`: startup **158.2 s** (load ~62 s +
opt-in warmup ~96 s), first requests 1.48 / 5.08 s.

Two follow-ups from the residual split: the **9.8 s per-layer
`SetProcessWorkingSetSize` trim** is now the largest removable load cost
(needs a memory-behaviour audit before removal), and the remaining ~29 s fill
still runs at ~2 GB/s against the 2.8 GB/s single-file ceiling.

C2's suggested config screen `AILA_Q4E_CPU_DEFER_ARENA_WAIT=1` was run against
the boundary baseline (`d0-deferwait-b2`, device spans live): inter_dev 9.04 vs
8.86, stagein 2.14 vs 2.11, experts 28.79 vs 28.32, submitted 76.06 vs 74.49
ms -- no improvement (slightly worse), so this knob is not the layer-boundary
lever. Note for the queue-profile gate: the peer WIP renamed it to
`AILA_Q4E_DEVICE_QUEUE_PROFILING`; window A arms now set both names.

### 6.22 The inter_dev bucket is the PLE wait, and the spans carry a +18% observer effect

Role-D1 re-attributed the boundary bucket from the marker streams:

1. `inter_dev` and `ple_host` measure the SAME interval (corr 0.9982 over 256
   decode tokens; `inter_dev - ple_host` = +0.31 ms median). Nothing is
   submitted to the GPU inside `kDevInter`, so the "queue gap" reading was an
   artifact of where the marker sits: the covered work is the L==1 PLE block
   (`ple_.consume()` waiting on the PLE worker plus its staging H2D).
2. **Decode never prefetches the PLE rows**: `prefetch_gather_pages` returned
   early for `num_tokens < 64`, so the 1-token gather pays 16-32 serial mmap
   faults per layer -- bimodal 9.05 ms (slow, n=144) vs 2.50 ms (fast, n=90),
   consistent with the F1 probe's 210 us/cold fault. Committed the inert knob
   `AILA_Q4E_PLE_PREFETCH_MIN` (default 64; `=1` opts decode in) in `867c89a`;
   the A/B is queued behind the peer GPU token. Estimated recovery 4-7 ms/token
   on slow-mode tokens.
3. **Observer effect**: `AILA_Q4E_DECODE_STAGE_EVENTS=1` itself costs +11-13
   ms/token (528 markers ~24-28 us each plus per-token drain): the same shape
   measures 74.5/72.9 ms instrumented vs 60.2/61.7 ms clean. Every absolute
   `*_dev_ms` figure carries this inflation; use it for rankings only.

Two opt-in variants from D1 (default off, bitwise-preserving, in the tree):
`AILA_Q4E_CPU_ROUTE_EARLY=1` (classify and start the arena-cold batch at the
router return; bound 1-4 ms/token) and `AILA_Q4E_STAGEIN_HOIST=1` (hoist the
per-layer D2H after the MLP-HC submit; bound 0.5-2 ms/token).

### 6.23 Decode PLE prefetch adopted: -6.3% token wall (A/B/B/A)

Following D1's attribution, `AILA_Q4E_PLE_PREFETCH_MIN` now defaults to 1
(commit `681a06e`), so the decode 1-token gather uses the same batched page
hint as prefill. Four arms on the primed 14K (`d1-a1b`, `d1-b1`, `d1-b2`,
`d1-a2`), per-token `[Q4E ForwardWall] total_ms` (n=128 each):

| arm | ForwardWall median | ple_host_ms | submitted_ms | continue rate |
|---|---:|---:|---:|---:|
| A (floor 64) | 64.477 | 6.608 | 60.706 | 15.577 |
| B (floor 1) | 59.456 | 0.257 | 55.596 | 16.290 |
| B (floor 1 + D1 variants) | 60.824 | 0.264 | 57.058 | 15.993 |
| A (floor 64) | 63.861 | 6.489 | 60.187 | 15.695 |

Median A -> B: **-4.03 ms (-6.3%)** on the token wall, `ple_host` -96%,
submitted -6.8%; the D1 variants (CPU_ROUTE_EARLY + STAGEIN_HOIST) were
neutral-to-negative in this series and stay default-off. This closes the
"layer-boundary" thread: the gap was the PLE gather all along.

Committed-default smoke on pair `retained27` (`d1-smoke`, primed 14K, no extra
knobs): ForwardWall median **58.78 ms** (-8.4% vs the 64.17 ms A arms) and a
continue rate of **16.65 tok/s** (+6.1% over the 15.58-15.70 A arms) -- above
the +5% default gate; `AILA_Q4E_PLE_PREFETCH_MIN=64` restores the old path.

### 6.24 The per-layer working-set trim is load-bearing (role-D2 audit, closed)

Role-D2's audit (finished posthumously) answers whether the 9.8 s of per-layer
`SetProcessWorkingSetSize` trim can be removed. It cannot:

| arm | startup | RAM copies | trim total | working set | min available |
|---|---:|---:|---:|---:|---:|
| trim on (default) | 59.1 s | 26.9 s (561 ms/layer) | 9.8 s | 1303 -> 10 MiB per layer | 4.28 GiB |
| trim off (`=0`) | **217.3 s** | 179.3 s (3736 ms/layer) | 0.0 s | 6662 MiB flat | **0.13 GiB** |

Without the trim the process working set balloons to 6.6 GiB, the machine runs
out of available RAM and the fill thrashes: 6.7x slower, +158 s startup. The
trim's 9.8 s buys a 158 s saving, so `AILA_Q4E_ARENA_WORKING_SET_TRIM` stays
default 1 and the startup-thread item is closed (the knob + audit ship in
`3ef4606` for future memory-behaviour work).

### 6.25 Goal framing: prefill 400+ / decode 20+ (2026-10-15 target)

Current vs target: prefill 14K = 69.4 s (202 tok/s) vs 400+ (<=35 s, <=10.2
s/block); decode 58.8 ms/token (17.0 tok/s) vs 20+ (<=50 ms). MTP is allowed.

**Prefill six-bucket split (role-E1, block = 19.5 s)**

| bucket | value | scope |
|---|---:|---|
| bank dependency wait | 8.1 s | contains H2D 5.3 s (46.8 GiB @9.5 GB/s) + reader lag 2.8 s |
| router_wait (queue residual) | 7.7 s | carries the 7.3 s of device work below |
| device GDN | 4.1 s | r1-stages2 |
| device full attn + hc/router | 2.8 s | r1-stages2 |
| host fill | 2.4 s | RAM -> staging banks |
| expert kernel | 0.4 s | supply bench mode 0 = 114-116 GB/s effective |

Two buckets (bank wait + router_wait) carry 81% of the block; the real
resources are the 46.8 GiB H2D stream (5.3 s) and the 6.6 s attention/GDN
device stream. The expert kernel is only ~2% -- the earlier "experts stage
12 s" reading was supply exposure, not compute.

Three additive paths sketched by E1 (bench + numbers in
`docs/plans/qwen38-r1-prefill-400-2026-10-15.md`, corrected here): (1) raise
the per-prompt cold-key union (fewer re-uploads: -5.4 s); (2) GDN rewrite levers
(the SLM-resident form is measured NO-GO, -61..-86%, so this path is smaller
than E1 first wrote); (3) decouple the staging reader fence (banks 2->4,
-2.8 s). If all three land: ~9.4 s/block = 436 tok/s.

**Decode post-PLE table (role-E2, `d1-smoke` artifacts, n=124)**: ForwardWall
58.79 ms = backend 55.25 + sample 3.42; inside: router_wait 12.12,
experts 20.65 (cpu_compute 13.25 / cpu_join 9.08), residual family ~9.0,
hc 6.10, linear 3.39, full 2.87, head 0.15, ple 0.25. There is no single
liberty: reaching 50 ms needs residual -4..5 + join overlap -3..4 + sample
-1..2 together.

**MTP arithmetic (p=0.748 measured)**: rate = (1+p)/W, so 20 tok/s needs
**W <= 87.4 ms** (the +5% gate: <=97.9). Archived windows: 87.9 (10-06 best)
to 112 ms (10-11 server-equivalent). The retained27 window has never been
measured -> arm M1 (A/B/B/A with `AILA_Q4E_SERVER_MTP=1`) is running now;
the two known MTP costs are draft ingest 0.68 ms/prompt token (32K = 22.3 s)
and the 77-96 ms window itself.

### 6.26 Hotset profile A/B after the PLE fix (re-run)

| arm | ForwardWall median | continue rate |
|---|---:|---:|
| H4 (default) | 59.644 ms | 16.175 tok/s |
| agent6k profile | 57.666 ms | 16.663 tok/s |

-3.3% wall / +3.0% rate, same as the pre-PLE screen (-3.0%): the profile's
value is unchanged by the PLE fix and still sits below the +5% default gate,
so the launcher keeps H4 unless it is packaged with other wins.

### 6.27 MTP on the current head is 3x SLOWER than T1 (M1 re-anchor)

The decisive M1 measurement (2048 context, prompt 1500, generate 128):

| arm | continue rate | acceptance | verify window (count=2) |
|---|---:|---:|---:|
| A1 T1 | 17.17 tok/s | - | - |
| B1 MTP width=2 | **5.88 tok/s** | 88/137 = 64.2% | **233.8 ms** (submitted; router_wait 124.8, experts_host 84.6) |
| B1 + `AILA_Q4E_VERIFY_SMALL_ROWS=1` | 6.28 tok/s | 114/153 = 74.5% | per-row path |

The verify window is 3.8x a count=1 token (61.3 ms), so MTP loses ~2.9x -- and
MTP needs W <= 87.4 ms for 20 tok/s. This matches the 10-11 alignment doc
("multi-row verify is both slow and not equivalent"): count>=2 routes the
projections/HC/PLE/head through the prefill GEMM paths, while the archived
88-112 ms windows predate the alignment switches. Conclusion: MTP cannot
contribute to the 20 tok/s goal until a *true* multi-row decode path amortizes
the weight reads across verified rows (the same blocker the service probe
found for multi-session batching). Parked with evidence; T1-side work
continues.

### 6.28 decode-attn pair2 K/V reuse: rows=1 only (T1-usable)

Peer variant (`d34c09a`, env `AILA_ATTN_KVREUSE=pair2`), bench A2:

| length | rows | baseline | pair2 | delta |
|---:|---:|---:|---:|---:|
| 8192 | 1 | 0.5679 ms | 0.4458 ms | -21.5% |
| 14336 | 1 | 0.8375 ms | 0.7019 ms | -16.2% |
| 14336 | 2 | 1.732 ms | 1.7285 ms | ~0% (gate 792 us not met) |
| 4096 | 2 | 0.631 ms | 0.631 ms | ~0% |

For T1 decode (2 full-attn... 12 full-attention layers) the rows=1 gain is
~1.6 ms/token (~2.8% wall) -- below the gate alone, but stackable; the
rows=2/MTP shape does not clear its gate. E2E arm `e4-pair2` is running.

E2E result: ForwardWall 59.644 -> **58.088 ms (-2.6%)**, continue rate 16.175
-> **16.845 tok/s (+4.1%)** (`receipt-pair2-e2e.json`); stackable with the
agent6k profile (-3.3%).

Correction after the B2 arm (`e4-pair2-a2` = 59.571 ms / 16.465 tok/s): the
pair2 e2e effect is **not established** -- B1 -2.6%, B2 -0.1%, median -1.4%
inside the +/-1.5% run noise. The kernel-level rows=1 gain (-16..-21%) is real
but largely hidden behind the host path. The pair2+agent6k stack arm
(`e5-stack`, 58.663 ms) likewise shows no stacking. Decode stays on the host
path: ~33 ms of submit/bookkeeping + 12 ms device-queue wait + 10 ms CPU join
per token; E5's misc audit leaves no >=5% kernel lever.

### 6.29 Prefill block bench: the supply is serialized behind the dependency chain

Role-E1's `AilaQwen4ExpPrefillBlockBench.exe` (production shape: 48 layers x
403 staged experts = 49.0 GB per pass, banks 2, slots 16):

| arm | wall | fill | H2D copy | bank wait | gu+down kernels |
|---|---:|---:|---:|---:|---:|
| copy (supply only) | 5731 ms | 4931 | 4956 (9.89 GB/s) | 96 | 725 |
| block (production order) | 5773 ms | 4953 | 4957 | 95 | 719 |
| kernel (all resident) | 553 ms | - | - | - | 202 |
| block, banks=4 | 5778 ms | 4959 | 4957 | 95 | 715 |

Two hard facts: (a) the expert kernels for a whole block are only **0.55 s
(2.8%)** -- the "experts stage 12 s" in production is supply, not compute;
(b) the supply by itself is **5.7 s/block**, i.e. 49 GB of fill+H2D at ~8.5
GB/s combined. banks=4 changes nothing (the bench's supply is already
overlapped), so E1's reader-fence path is closed at bench level.

Production block (19.5 s) decomposes as host-busy 11.2 s (fill 2.4 + copy
wait 8.1 + submits) then router_wait 8.0 s: the supply and the ~8 s of device
work (GDN 4.1 + full 2.7 + hc/router 0.6 + expert kernels 0.55) are
**sequential, not overlapped**. The dependency chain forces it: attention(L)
-> MoE(L) -> attention(L+1), and the 32-slot staging banks only allow one
batch (~39 MiB) of lookahead, far less than a layer (403 experts = 973 MiB).
If the supply overlapped the device stream, the block would be ~8.5-10 s =
**410-480 tok/s**. The next implementation task is therefore the deeper
supply pipeline (cross-layer staging lookahead), not more kernel work.

### 6.30 Decode misc families audited (role-E5): three excluded, hc is small

Clean budget (A/A spans from `c2-round2-b{1,2}`, observer-scaled x0.82):
hc **8.4**, linear **6.6**, shared **2.7**, head **2.7** ms -> 20.4 of ~59.3
ms/token. Bench A1 (`q4e_decode_misc_bench.cpp`, my run):

| finding | number |
|---|---|
| submit cost | **9.4 us/kernel** (512-kernel probe) |
| hc down / up (one call) | span 79 / 61 us |
| hc per token (194 kernels) | span 5.85 ms vs a 4.37 ms byte floor (1.33x) |
| hc pl2/pl3 (software pipelining) | 11.7-12.0 ms -- **worse** |
| gdn 36/token | span 2.51 ms |

Verdicts from E5: head already streams at ~470 GB/s (perfect = +0.75%);
linear is 1.36x from its floor with only denied mechanisms left (best +3.0%);
shared is submit-bound (144 submits x 25-30 us), i.e. a command-amortization
problem, not a kernel one; hc is the only candidate and its ceiling is
~1.4 ms/token (2.4%). Net: the misc families cannot deliver the decode 20+
gap; the structural items (expert CPU-handoff wait ~18 ms, submit
amortization) remain the real decode levers.

### 6.31 Prefill staging banks (V1) adopted as opt-in: -4.8% TTFT

Role-E6 implemented `AILA_Q4E_PREFILL_STAGING_BANKS` (2..4, default 2 =
byte-identical) with a banks-1 sliding lookahead inside `RunAdaptivePrefill`,
plus a latent bank/descriptor mismatch fix for pp>1. Arms (14K primed,
generate 8, production knobs):

| arm | block submitted (median) | bank_wait | TTFT |
|---|---:|---:|---:|
| banks=2 (baseline) | 19,364 ms | 7,893 ms | 69.4-70.0 s |
| banks=4 (b1) | 18,752 ms | 6,224 ms | 65.87 s |
| banks=4 (b2) | 19,062 ms | 6,437 ms | 66.96 s |

Reproducible: **block -2.4%, bank_wait -20%, TTFT -3.3..3.5 s (-4.8%,
202 -> 212 tok/s)**, cost +77 MiB VRAM / +77 MiB host (one extra bank pair).
Committed `d7a5f6f`, pair `retained28`. Still opt-in: the lookahead is only
3 chunks (48 experts) vs the 403 an exposed layer needs, so the structural
gap (bank_wait 6.3 s stays) is untouched -- role-E7's speculative whole-layer
staging (weights are routing-independent, so a layer's supply can overlap its
own attention) is the path that could remove the remaining ~6 s and reach
410-480 tok/s. E1's rev2 arithmetic agrees that bytes+device alone reach only
~368 tok/s.

Follow-up screen (banks=4 + `AILA_Q4E_STAGING_SLOTS=32`, `e6-b4s32`):
bank_wait falls further to **4,612 ms (-26% vs slots=16)** but the block wall
does NOT improve (submitted 19,227 ms, TTFT 67.6 s) -- i.e. ~1.6 s of bank
wait was already hidden behind other work and the binding constraint has
moved. Per E6's trigger, deeper staging is no longer the lever; the wall is
now set by the device stream + host chain, which is what the speculative
whole-layer staging (V3) must overlap.

### 6.32 Speculative layer staging bench: partial overlap only (V3 gate NO-GO)

Role-E7 built `Qwen4ExpLayerStage.*` (arena + fill thread + copy stream,
default zero calls) and `AilaQwen4ExpLayerStageBench.exe` (real GDN prefill
kernels vs 454-slot staging, 1.15 GB). Arms:

| arm | wall |
|---|---:|
| load (GDN kernels only) | 150.5 ms |
| stage (fill 176.6 + H2D, overlapped internally) | 182.4 ms |
| overlap (both) | 255.4 ms |

Serial would be 332.9 ms; overlapping saves 77 ms (23%), but the GDN load
runs 1.40x slower under concurrent staging (bandwidth contention), so the
gate (inflation <= 30% of the stage wall) fails: `[LS GATE] ... no_go`.
Also note the bench's fill runs at 6.5 GB/s (single-threaded) vs the
production parallel fill's ~20 GB/s.

### 6.33 The CPU expert path is 4.5x the PCIe bandwidth -- the real prefill lever

Decode's CPU expert executor measures **45 GB/s of weight reads** (556 MB in
13.7 ms, A/A-stable), while the whole prefill supply is capped by **PCIe at
9.9 GB/s**. The current prefill sends 46.8 GiB/block over PCIe (5.0 s + 2.4 s
fill); the same expert work on the CPU is ~2.2 s at the measured 53.7 us per
expert-activation (40,960 activations/block). The CPU reads host DRAM while
the GPU reads VRAM, so a CPU-MoE prefill can run **concurrently** with the
GPU's 6.8 s attention/GDN stream:

block -> max(CPU MoE ~2.2-3 s, GPU attention/GDN 6.8 s) + result H2D ~0.1 s
      ~= 7-7.5 s = **530-570 tok/s**

Role-E9 is implementing this as `AILA_Q4E_PREFILL_MOE_CPU=1` (default off,
`Qwen4ExpPrefillCpuMoe.*` + a diff draft for the prefill section). This
supersedes the staging-pipeline line if it verifies: no PCIe staging, no
extra VRAM banks, and the 18 otherwise-idle cores do the work.

### 6.34 `MOE_FRONT_FUSED=1` is real, bit-exact and reproducible (-2.8%)

Role-E8's census found the fusion switch is wired but never A/B'd (the profile
JSON explicitly sets it to 0). Arms (primed 14K, production knobs):

| arm | ForwardWall median | continue rate | response SHA (continue) |
|---|---:|---:|---|
| A (`MOE_FRONT_FUSED=0`) | 59.644 ms | 16.175 | 3d4f184d... |
| B1 (=1) | 57.839 ms | 16.788 | 3d4f184d... |
| B2 (=1) | 58.063 ms | 16.870 | - |

**-2.8% wall / +4.0% rate, byte-identical greedy output** (SSE bodies hash the
same). Below the +5% default gate alone, so it stays opt-in (a one-line flip
in `docs/plans/qwen38-server-mtp-2026-10-07.json`), but it is bit-exact and
stacks with the profile (-3.3%) and int8-KV (-3.5% wall / 16.95 tok/s, arm
`e9-int8kv`) for a ~9% decode bundle.

### 6.35 HC fusion (role-F1) wired but NO-GO: host saving loses to a CPU-path regression

`AILA_Q4E_DECODE_FUSED_HC=1` (default off, wired in `Qwen4ExpBnb4Backend.cpp`)
fuses the norm+inject+down and up+mix pairs (291 -> 194 submits/token, bit-exact
by construction). Measured (`f1-b1` vs the default config `e11`):

| field | default | fused_hc=1 |
|---|---:|---:|
| ForwardWall | 57.465 ms | **67.188 ms (+17%)** |
| hc_host_ms | 6.537 | 5.653 (-14%, the intended saving) |
| cpu_compute_ms | 13.08 | **19.64 (+50%)** |
| cpu_join_ms | 8.37 | **16.28 (+95%)** |

The fused kernels change GPU scheduling/residency enough to slow the CPU
expert chain by 50%, costing ~10 ms -- far more than the 0.9 ms submit saving.
Verdict: NO-GO; the wiring stays gated (default off) as a documented
experiment. Lesson (third instance): a host-side saving only pays if the
device/CPU schedule is unchanged.

### 6.36 Layer-stage overlap re-calibrated: the limiter is the single-threaded fill

`AilaQwen4ExpLayerStageBench --gdn-calls 8` (load 601.8 ms, stage 183.7 ms):
overlap 707.2 ms vs a 785.5 ms serial sum -> 78/184 ms hidden, and the
inflation is a **constant ~105 ms** (identical at 2 and 8 calls). So the
overlap failure is not bandwidth contention: 178 of the 184 ms stage arm is
the single-threaded host fill (1.15 GB at 6.5 GB/s). The production staging
path already has a parallel host copy (~20 GB/s, `STAGE_COPY_PARALLEL=1`);
with a parallel fill the stage drops to ~116 ms and the per-layer wall should
be ~141 (device) + 0.58x116 ~= 187 ms -> **block ~9 s = 455 tok/s**. Role-G3
is wiring E7's integration diff (default off) with the parallel fill and the
reserve-3584->2450 VRAM plan (option C).

Chunk-8192 re-screen with banks=4 (`g2-c8192-banks4`) is a hard NO-GO:
TTFT 134.8 s (vs 66.0 at chunk 4096 + banks=4), the 8192-row chunk's
bank_wait 56.2 s -- halving the uploads-per-token does not survive the
pipeline collapse, consistent with the charter's S2 entry.

### 6.37 Decode best-known bundle confirmed (55.12 ms = ~18.1 tok/s backend)

`e10-bundle` / `e10-bundle-b2` (MOE_FRONT_FUSED=1 (now default) + int8-KV +
agent6k profile): ForwardWall 54.772 / 55.461 ms -> median **55.12 ms
(-7.6% vs the 59.64 ms pre-bundle baseline)**, continue rate 17.4-17.6 tok/s
(+8%). With the PLE fix this is the current production-capable decode config;
the launcher would set `AILA_SERVER_KV_TYPE=int8` + the agent6k profile.
Remaining gap to 20 tok/s: ~5 ms, which the MTP-verify fix (role-G1) and the
layer-stage prefill work (role-G3) are pursuing in parallel.

### 6.38 MTP verify window fixed: 233.8 -> 92.3 ms (-60%), still -9.6% vs T1

Role-G1 traced the verify window's cost to three stacked gates that kept the
multi-row CPU path off in the adaptive pipeline (`CanExecuteArenaVerifyRows`
needs `ARENA_VERIFY_ROWS=1`; the `verify_cpu_rows` cluster requires
`!use_adaptive_pipeline_`, which no longer exists; the per-row CPU branch is
`count==1`-only). Window evidence (pre-fix): staged 595, h2d 1438 MB,
cpu_compute 0. New gate `AILA_Q4E_VERIFY_CPU_COLD=1` (default 0) runs each
verify row as a T1-shaped single-row CPU batch (constructive bit-exactness:
each task is per-token independent).

| metric | pre-fix | post-fix |
|---|---:|---:|
| count=2 window submitted | 233.8 ms | **92.26 ms** |
| staged / h2d_mb | 595 / 1438 | 0 / 0 |
| cpu_compute / cpu_join | 0 / 0 | 29.47 / 24.84 |
| acceptance | 64.2% | **74.8%** (113/151) |

A/B/B/A continue rate: T1 17.328/17.121 vs MTP 15.543/15.592 -- MTP improved
from -66% to **-9.6%**, but still behind. Per-step budget: verify 92.3 +
~20.4 ms draft = 112.5 ms for 1.748 tokens; break-even needs <= 100.9 ms and
20 tok/s needs <= 87 ms. Role-G4 is charting the draft cost and the width-3
option.

### 6.39 Layer-stage integration is half-wired (role-G3 postmortem, role-G5 fixing)

`g3-layerstage-b1` (`AILA_Q4E_LAYER_STAGE=1`, banks=2): TTFT **89.6 s**,
block submitted 24.7 s (+27% vs the 19.4 s default), while
`h2d_mb=47,055` and `bank_wait=7,898 ms` are **identical to the plain path**.
The arena engaged (`slots=454 chunk=16 banks=4 threads=6 arena=1097 MB`) but
the cold loop still stages every expert through the banks -- the two staging
paths run in parallel (double bytes). Role-G3 errored before finishing the
consumer wiring (`SlotOf>=0` experts must bypass bank staging). Role-G5 is
completing it; the target remains block <=15 s with `h2d_mb` near zero.

## 7. Open items and non-claims

1. The stage-serialized arm is done; the remaining caveat is that stage
   intervals carry the per-stage sync cost, which is invisible at prefill
   scale but 28-54 ms/token at decode scale -- do not read the decode stage
   split from that arm.
2. First-request ~80 s: DONE as an opt-in lever (`AILA_INIT_WARMUP=1`, section
   6.7): first large request 109.9 -> 51.2 s, startup +111 s. Residual 20.5 s
   is a first-chunk PLE gather (new item, section 6.7) -- the next lever here.
3. Layer-major: measure union(8192) with a scoped HOTSET_EXPORT on the
   synthetic 8192 prompt before freezing the prototype's H2D gate (section
   6.2; geometry says -23..-28% for a 2-chunk window).
4. The two-chunk layer-major prototype itself is a multi-day implementation
   task (role-P doc section 3.4/3.5: driver loop order, cancellation at layer
   granularity, checkpoint retention, per-layer host work, scratch sizing).
   Day-2 status: v1-v3 implemented and measured (section 6.10). H2D -75%
   confirmed; wall -67% because there is no cross-layer overlap. NEXT ACTION:
   cross-layer pipelined variant (stage layer L+1 while layer L computes),
   then re-run the gate arm; do not tune shapes again before that.
5. K2 (QSA route split >=2052 tokens): attempted at 23:34, the exe aborts right
   after the device banner (no `[QSA COST]` rows; `qsa-cost-curve-A1.log`, 150
   bytes). The probe path needs repair before it can inform the K ranking; K1
   only covers decode-GU.
6. Service-mode multi-session is DONE (section 6.12): sessions=2 admits both
   requests but rotates one token per quantum -- aggregate 12.85 tok/s is
   BELOW the single-session 15.5; the S-doc analytic bounds collapsed to the
   serial case. Native multi-row decode for this backend is the prerequisite
   for any aggregate gain.
8. NEW top prefill item (section 6.7): ~20.5 s PLE gather on the first chunk of
   a fresh session (visible only once first-use JIT is pre-paid). Options:
   issue the gather earlier (chunk-lookahead already exists for subsequent
   chunks), overlap it with the first chunk's H2D staging, or grow the PLE
   workspace at load. Not yet probed.
7. All decode/prefill numbers above are single-session, decode-batch=1, bf16 KV,
   H4 profile, reserve 3584, chunk 4096 (production shape), and the pair
   `retained20` (worker 4d75c59c / server 211aabb0); the day-2 sections
   6.11-6.13 use pair `retained23` (worker 1b988254 / server ed71ae3d).
9. Working tree at close carries two uncommitted WIP edits pending build/GPU
   validation: the v4 cross-layer layer-major gate + module timers
   (`AILA_Q4E_LAYER_MAJOR_PIPELINE`; estimates -0.5%, below gate) and the K3
   `--decode-attn-bench` kernel arm. Run `git status` before the next build.

## 8. Ledger

| line | decision | evidence |
|---|---|---|
| L1 | baseline reproduced under primed protocol (70.51/71.87 s; 15.50/15.62 tok/s) | receipt-aa-ttft.json / receipt-aa-decode.json |
| L2 | primed protocol mandated for TTFT receipts (one-time first-chunk penalty) | scan_ttft.py table; r1-diag cold/warm chunk pair |
| L3 | decode table refreshed at retained20; retained17 split confirmed | decode/retained20-r1-diag.json |
| L4 | prefill chunk table refreshed; fill+bank_wait conserved, attention submits negligible | r1-diag AdaptiveWork rows |
| L5 | two probes selected: P-1 (layer-major sizing), D-C1 (CPU credit cap) | section 6 |
| L6 | D-C1 cap=4: NO-GO (join -7.0 ms but fill +24.4 ms and +133 MB/token H2D; decode -37%) | section 6.1, receipt-dc1-cpu4.json |
| L7 | P-1: union median 495/layer; layer-wide single-pass variant chosen; prototype gate stands | section 6.2, ttft14k-export.profile |
| L8 | K1: decode-GU 0.12343 ms (A/A 0.002%) -> >=1.5x off wall; variant study justified | section 6.4 |
| L9 | next GPU arms: P-2 (done, L13) -> layer-major 2-chunk prototype (multi-day); K2 (QSA route) optional | section 7 |
| L10 | one-time first-forward cost attributed: router 50.4 s + full-attn 19.8 s + linear 7.3 s + hc 1.2 s on a 19-token request; XMX-vs-codegen discriminator queued | section 1.2, r1-stages2 |
| L11 | steady chunk stage split re-measured at retained20 (experts 12.0 / GDN 4.1 / full 2.4-2.9 s); C1's 4.07/2.85/12.58 confirmed within 5% | section 3 |
| L12 | charter Q2 closed: production router_wait (8 s/chunk) is queue depth; router work is 0.17-0.29 s/chunk, and `router_wait_ms` collapses to 3-21 ms under serialization | section 3 |
| L13 | P-2 baseline pair recorded: synthetic 8192 -> 38.32 s TTFT, 78.0 GiB H2D, 32,059 staged uploads; 2-chunk layer-major geometry is -23..-28% H2D, just under the -30% gate -> measure union(8192) before freezing the gate | section 6.2 |
| L14 | C2 closed: PLE_ASYNC=0 removes ple_host but not the wall (+41% TTFT); C1 already closed; remaining decode levers are device-side (section 5) | section 6.3 |
| L15 | First-use ~80 s is not XMX-owned (XMX=0 keeps it) and not cacheable today: SYCL_CACHE_PERSISTENT=1 kills the worker during load (issue #4 reproduced on driver 32.0.101.8724) | section 1.2 |
| L16 | Production-parity knobs (STAGE_COPY_PARALLEL=1 + WGSHARE=4) give 14K TTFT 69.09 s (-3.0% vs A/A median) with decode unchanged; runner baselines carry a ~3% untuned penalty | section 6.5 |
| L17 | union(8192) measured: 20,761 keys / 49.0 GiB floor vs 32,059 uploads / 78.0 GiB -> 2-chunk layer-major passes the -30% H2D gate; expected wall -10..-19% `[estimate]` | section 6.2 |
| L18 | K2 (QSA route) attempted: `AilaQwen4ExpKernelTests.exe --qsa-probe` aborts right after the device banner (no [QSA COST] output); probe needs repair before it can inform the K ranking | section 7 |
| L19 | `AILA_INIT_WARMUP=1` is inert on the production worker (no [Warmup] markers, first request unchanged): the warmup lives in the legacy `InferenceEngine::init`, not in `ModelFactory::LoadQwen4Exp` | section 6.6 |
| L20 | Next-session warmup plan frozen: qwen4exp factory-side warmup with 4096-row prefill + decode shape coverage, state reset, `AILA_INIT_WARMUP`-style gate | section 6.6 |
| L21 | Warmup implemented in `LoadedModel::Load` (`AILA_INIT_WARMUP=1`, opt-in, export-guarded); first large request 109.9 -> 51.2 s, startup +111 s, 14K/decode unchanged | section 6.7 |
| L22 | New prefill lever found: the first chunk of a fresh session carries a ~20.5 s PLE gather (`ple_host`), previously hidden behind first-use JIT; varied ids do not change it | section 6.7 |
| L23 | Pair `retained21` archived (worker `e45fe7c1...`, server `ddbed15f...`); CPU/ABI suite 20/20 | tmp/q4e-mtp-alignment-20261011/retained21/pair.json |
| L24 | Warmup committed (`a3ee21e`), rebuilt as pair `retained22`; smoke arm: `[Q4E Warmup]` 96.87 s, 19-token first request 1.48 s (was 80.5 s), synthetic 6K first request 30.41 s == steady, ple residual 0 -> the fixture-only 20.5 s PLE gather residual is content-dependent, not structural | section 6.7, retained22 |
| L25 | K2 repaired and measured: routed QSA wins at 8192 (+36-37%) but is marginal at 4096 (+0.6..+5.4%) and loses below the 2052 gate | section 6.8 |
| L26 | PLE first-chunk gather root-caused to mmap demand paging; F1 `PrefetchVirtualMemory` patch gives 50.5 -> 31.8 s on the fixture first request (A/B/B/A, -37%, above gate) -> adopted, default on | section 6.9 |
| L27 | Layer-major v1 (pair `retained22` + A1 code): H2D -74% (20.2 vs 78.15 GiB) but TTFT 163.3 vs 37.1 s -- bank_wait 107 s because the 16-slot staging batches serialize with compute; v2 (pipelined staging) in flight | section 6.2/6.7, lm-b2 |
| L28 | Layer-major v2 (`lm-b3`): TTFT 61.95 s (v1 163.3, baseline 37.06); staged/fill/bank_wait/experts per layer sum to 8,349 / 1.05 / 20.77 / 22.37 s with a 16.9 s residual -> suspicion is the 8192-row capacity scratch (role-A1 residual risk 2); v3 (per-half) commissioned | lm-b3, lm_b3_analyze.py |
| L29 | `AILA_Q4E_LAYER_MAJOR_STAGING_SLOTS=32` currently faults with "Invalid Qwen4Exp staging host-copy batch" on the first real window; resident slots drop 2792 -> 2760 at 32 slots; handed back to role-A1 for v3 | lm-b4 |
| L30 | Layer-major v3: per-half capacity + 1.15 GiB carries + 32-slot host-copy fix (Qwen4ExpAdaptiveCache splits into <=16-slot calls); TTFT 62.30 s -- per-half did not move the wall; per-layer walls grow to 2.6-3.6 s on late layers (no cross-layer overlap) | lm-b5 |
| L31 | Layer-major experiment committed as opt-in/default-off (`9abb858`, pair `retained23`); CPU/ABI 20/20; next step is the cross-layer pipelined variant | section 6.10 |
| L32 | Hotset profile A/B: agent6k cuts misses -12.7% and per-token wall -3.0% (receipt-profile-ab1.json); below the default gate, stays opt-in; re-measure quiet before a launcher change | section 6.11 |
| L33 | Day-2 exploration commissioned: B1 Strata reference mining, B2 backlog/measurement-gap mining, B3 decode-attention K/V reuse design (run-level artifacts under tmp/q4e-sprint-r1-20261014/{strata,backlog,decode-attn}) | agents |
| L34 | Primed 14K re-pin at retained23: 14000 69.69 s (retained20 A/A 70.51/71.87), decode w14 submitted median 64.3 vs 64.4 ms -> F1 default-on carries no primed-protocol cost (receipts `receipt-prof-a6k-*.json`) | section 6.11 |
| L35 | Service S2 executed (`run-s2-a3`): peak 2 concurrent, strict per-token rotation (29 adjacent duplicate positions, no small multi rows), aggregate 12.85 tok/s < single 15.5, per-token 70.8 ms (+10% under rotation); two earlier attempts failed closed on MEM_LARGE_PAGES win32 1450 (stale queue overlapped the GPU) and are invalid as measurements | section 6.12 |
| L36 | Production agent sequence smoke at retained23 passed: PI round + cancel-recover + 6000 30.53 s + 14000 69.53 s, continues cached 5976/13976, exit 0 | section 6.13 |
| L37 | Day-2 audits delivered: B3/K3 decode-attn K/V reuse (routed 14K decode caps the e2e lever at ~6.5%; dense-only >=5%; `--decode-attn-bench` arm ready), K4 GDN prefill (kernel 74.8 ms/layer at 516.7 GB/s effective, not DRAM-walled; state-free bound -12.7% TTFT), B2 backlog mining (5.5/8 charter questions open; legacy `AILA_Q4E_PREFETCH` family dead in the adaptive path), A1v4 pipelined layer-major (estimate ~-0.5%, below gate; v3 staged corrected to 42.37 GiB / 17,951) | docs/plans/qwen38-r1-*.md; tmp/q4e-sprint-r1-20261014/agents/ |
| L38 | Decode regression scare closed: quiet-h4 on a quiet box gives 15.20/15.43 tok/s continues (retained20: 15.50/15.62; 14000 69.37 s best primed); the 12.9-14.2 readings were sub-agent CPU-load artifacts -> no retained23 decode regression | section 6.14 |
| L39 | decode-attn bench first runs (R3-A1/A2, guards pass): 14336 rows=2 = 1.739-1.743 ms total (869-871 us/row) vs 101 us/row floor; 2-row gain 10-17% at 4096/8192, ~0 at 14336; A/A <=0.2% at 8192/14336; pair-reuse variant gate = <=792 us at 14336 rows=2 | section 6.15, tmp/q4e-sprint-r1-20261014/decode-attn/ |
| L40 | Window-A replication of the S2 service probe (`run-s2-wina1`): strict rotation, aggregate 14.29 tok/s < single-session 15.43 (quiet-h4 anchor) -> charter question 6 closed quantitatively by two independent runs | section 6.12 |
| L41 | K3 `--decode-attn-bench` arm and layer-major v4 ring committed (`64a5625`, `40239cd`; both default-off/ungated); tree clean; pair `retained24` archived (worker `c008331e`, server `7c7fdf5a`) | tmp/q4e-mtp-alignment-20261011/retained24/pair.json |
| L42 | Window-A decode-attn rows=1 numbers for the record: 4096 = 381 us/row, 8192 = 567, 14336 = 846.5 (median of 20; A/A consistent with the rows=2 arm) | tmp/q4e-sprint-r1-20261014/decode-attn/bench-A1.log |
| L43 | GDN prefill variant closed: bitwise-parity bench, anchor reproduced (74.585 vs 74.817), r=0 rewrite -4.6% kernel (-0.6% e2e), SLM residency -61..-86% -> NO-GO, assets kept | section 6.16, tmp/q4e-sprint-r1-20261014/gdn/b5-p1-A1.log |
| L44 | Decode "unprinted" 9.09 ms residual decomposed (8.997 ms): shared_submit 3.73 + router_other 1.93 + moe_front 1.78 + topk 0.91 + shared_add 0.63; device-spans arm available behind AILA_Q4E_DECODE_STAGE_EVENTS (queue enabler added in Model.cpp) | section 6.17, dec-instr-a1 |
| L45 | Startup baseline for role-B4: 147.5/146.5 s (two arms), phase split: arena RAM copies 95.6 s (1992 ms/layer, 578 MB/s) of a 136.5 s load; theoretical 76-90 s recoverable at disk bandwidth | tmp/q4e-strata-c7/b4-base-a1*, b4-loadtiming2 |
| L46 | Device-spans arm: decode device spans sum to 71.2 ms/token (~96% of the 74 ms wall) -> decode is GPU-span-bound, not host-bound; top device item = experts 27.5 ms (38.6%), then misc 11.8, hc 9.7, linear 7.7, full 5.4. Re-ranks all decode levers to the GPU side | section 6.17, dec-instr-b2 |
| L47 | Startup: arena fill page prefetch (role-B4, `AILA_Q4E_HOST_ARENA_PREFETCH`, default on) cuts worker startup 135.4 -> 78.6 s (-41.9%, A/B/B/A 4/4) by lifting the fill from 0.58 to ~2.05 GB/s; committed `fe1bf9e`, pair `retained25` | section 6.18, receipt-startup-prefetch-abba.json |
| L48 | Production-shaped smoke on `retained25`: startup 175.4 s (load 79 + warmup 96), first requests 1.46 / 5.06 s -- load prefetch and opt-in warmup compose cleanly | section 6.18, b4-smoke |
| L49 | `misc_dev` decomposed (C2): inter_dev 8.86 + stagein 2.11 + ple 0.26 + prerouter 0.07 = 11.29 (self-check misc=0 passes); the actionable item is layer-boundary host lag (~9-11 ms/token of GPU idle = 12-15% of decode) | section 6.19, c2-round2-b1 |
| L50 | Decode MoE kernel line closed (C1 bench, bitwise parity): variants -7% gu/down but kernel busy is only 3.6-5.1 ms/token -> ~0.4% wall; the lever is the CPU/GPU scheduling side. Bench sentinel guard has a buffer-reuse bug (chain arm blocked) | section 6.20, decode-moe-A1.log |
| L51 | `CPU_EARLY_START=1` is inert in production (the profile's `CPU_ASYNC_ROUTE=1` branch always wins); the arm's quiet decode was 15.57/15.65 tok/s | c2-earlystart-b1 |
| L52 | Startup round 2 (role-C3, adopted): async arena hint overlap cuts worker startup 79.1/78.6 -> 58.5 s (A/B/A, -26%); residual split exposes stage memcpy 3.0 s, stage wait 0.9 s, nested 0.2 s, and a **9.8 s per-layer working-set trim** as the next target. Committed `626fada`, pair `retained26`; prod-shaped smoke 158.2 s startup | section 6.21 |
| L53 | Role-D1: inter_dev == ple_host (corr 0.9982); decode's PLE gather skips prefetch (16-32 serial faults/layer, 9.05 vs 2.50 ms bimodal) -> `AILA_Q4E_PLE_PREFETCH_MIN` knob committed (`867c89a`, default inert); A/B queued | section 6.22 |
| L54 | Observer effect measured: `DECODE_STAGE_EVENTS=1` costs +11-13 ms/token (528 markers); device-span absolutes are ~18% inflated -- rankings only. `CPU_DEFER_ARENA_WAIT=1` screened: no effect | sections 6.21/6.22, d0-deferwait-b2 |
| L55 | GPU-clash incident: window A started an arm while window B held the token; the worker died mid-request (artifact `d1-a1-gpu-clash`). Token discipline re-affirmed in COORD-NOTE; window A queue held until release | COORD-NOTE 19:55 |
| L56 | Decode PLE prefetch adopted (`AILA_Q4E_PLE_PREFETCH_MIN=1`, commit `681a06e`): A/B/B/A ForwardWall 64.17 -> 60.14 ms (-6.3%); committed-default smoke 58.78 ms (-8.4%) and 16.65 tok/s (+6.1%) -> above the +5% gate | section 6.23, retained27 |
| L57 | D1 variants screened in the same series: CPU_ROUTE_EARLY + STAGEIN_HOIST neutral-to-negative (60.82 vs 59.46 ms) -> stay default-off; `CPU_DEFER_ARENA_WAIT=1` also neutral (earlier screen) | sections 6.21/6.23 |
| L58 | Working-set trim audit (role-D2): trim off makes startup 59.1 -> 217.3 s (working set 6.6 GiB flat, available RAM 0.13 GiB, fill 3736 ms/layer) -> the 9.8 s trim is load-bearing; knob stays default 1, item closed | section 6.24, trim-on-a1/trim-off-b1 |
| L59 | Goal framing (user): prefill 400+ tok/s / decode 20+ tok/s (MTP allowed). Current: 202 tok/s (69.4 s/14K) and 17.0 tok/s (58.8 ms); gaps: block 19.5 -> <=10.2 s, token 58.8 -> <=50 ms | section 6.25 |
| L60 | MTP re-anchor (M1): verify window = 233.8 ms (3.8x a count=1 token) -> MTP 5.88 tok/s vs T1 17.17 (3x slower); VERIFY_SMALL_ROWS=1 gives 6.28; acceptance 64.2% (74.5% in the small-rows arm). Archived 88-112 ms windows predate the alignment switches -> MTP parked until a true multi-row decode path exists | section 6.27 |
| L61 | decode-attn pair2 (peer `d34c09a`, rows=1 only): 8192 -21.5%, 14336 -16.2% kernel; e2e ForwardWall -2.6%, +4.1% rate (receipt-pair2-e2e.json); rows=2/MTP shape flat | section 6.28 |
| L62 | Hotset profile re-screen post-PLE: agent6k -3.3% wall / +3.0% rate (unchanged vs pre-PLE) -> still opt-in | section 6.26 |
| L63 | Prefill block bench (role-E1): kernels 0.553 s/block (2.8%), supply (fill+H2D) 5.73 s, banks=4 no change; production supply (11.2 s host-busy) and device (~8 s) are SEQUENTIAL -> the block is 19.5 s instead of ~9-10 s. Pipeline deepening is THE prefill lever; role-E6 implementing V1 (4 banks + deeper lookahead) / V2 (973 MiB per-layer staging) | section 6.29 |
| L64 | Prefill staging banks V1 adopted as opt-in (`AILA_Q4E_PREFILL_STAGING_BANKS`, commit `d7a5f6f`, pair `retained28`): banks=4 -> block -2.4%, bank_wait -20%, TTFT -4.8% (202 -> 212 tok/s), +77 MiB VRAM; default stays 2 | section 6.31 |
| L65 | E6 dependency proof: the router reads post-attention hidden -> cross-layer staging lookahead is structurally impossible; the only legal lookahead is within a layer (after wait_actual_route). The remaining structural fix is speculative whole-layer staging during attention (role-E7) | section 6.31, E6 doc |
| L66 | E5 decode-misc verdicts: head/linear excluded (<=3% ceilings), shared is submit-bound, hc ~2.4% ceiling; the decode 20+ gap stays with the expert CPU-handoff (~18 ms) and submit amortization | section 6.30 |
| L67 | Decode stacking check: pair2 + agent6k profile -> 58.66 ms (no stack); pair2 A/B/B = -2.6%/-0.1% (not established); the decode baseline stays ~58.8 ms = 17.0 tok/s | section 6.28 |
| L68 | Speculative layer staging bench (role-E7): load 150.5 / stage 182.4 / overlap 255.4 ms -> only 23% overlap benefit and the GDN load slows 1.40x; V3 gate NO-GO as benchmarked (fill ran single-threaded at 6.5 GB/s) | section 6.32 |
| L69 | CPU expert path measured at 45 GB/s vs PCIe 9.9 GB/s -> a CPU-MoE prefill could run concurrently with the GPU attention: block ~7-7.5 s = 530-570 tok/s. Role-E9b benchmarking the CPU throughput at prefill shapes (E9's first attempt produced no artifacts) | section 6.33 |
| L70 | `AILA_Q4E_MOE_FRONT_FUSED=1` verified bit-exact (SSE hash equal) and reproducible: ForwardWall -2.8%, rate +4.0% (57.84/58.06 vs 59.64 ms). Opt-in (profile JSON line); stacks with profile and int8-KV | section 6.34 |
| L71 | int8-KV screen: ForwardWall 57.55 ms (-3.5%), 16.95 tok/s; gives +156 resident slots | e9-int8kv |
| L72 | Fused-HC wiring (role-F1) NO-GO: ForwardWall 57.47 -> 67.19 ms (+17%) -- the -0.89 ms hc_host saving is drowned by +50% cpu_compute / +95% cpu_join; kept gated, default off | section 6.35 |
| L73 | Chunk-8192 + banks=4 re-screen: TTFT 134.8 s (bank_wait 56.2 s on the 8192 chunk) -> hard NO-GO; halving uploads-per-token does not survive the pipeline collapse | section 6.36, g2-c8192-banks4 |
| L74 | Layer-stage overlap re-calibration: inflation is a constant ~105 ms, and 178/184 ms of the stage arm is the single-threaded fill -> replace with the production parallel fill; estimated block ~9 s = 455 tok/s. Role-G3 implementing the integration (default off, reserve option C) | section 6.36 |
| L75 | MTP verify bottleneck identified: the count=2 window stages ALL experts over PCIe (~1.4 GiB ~= 147 ms) and measures 233.8 ms with cpu_compute=0; a CPU verify path exists (`JoinArenaVerifyRows`, Backend.cpp:1264). Role-G1 is gating a fix (`AILA_Q4E_VERIFY_CPU_COLD`); target window <=100 ms -> >=19-20 tok/s | section 6.27, G1 task |
| L76 | Decode bundle confirmed (e10-bundle/b2): ForwardWall 55.12 ms median (-7.6% vs 59.64), 17.4-17.6 tok/s; production-capable config = front-fused default + int8-KV + agent6k profile | section 6.37 |
| L77 | MTP verify fix (role-G1, `AILA_Q4E_VERIFY_CPU_COLD=1`, default off): the count=2 window drops 233.8 -> **92.26 ms (-60%)**, staged/h2d = 0 (the CPU path is used), acceptance restored to 74.8% (113/151). A/B/B/A rate: T1 17.33/17.12 vs MTP 15.54/15.59 -> still -9.6%, break-even needs W_total <= 100.9 ms vs the current 112.5 (92.3 verify + ~20 draft); deferred to role-G4 (per-step budget) | section 6.38 |
| L78 | Layer-stage integration (role-G3) is half-wired: `AILA_Q4E_LAYER_STAGE=1` gives TTFT 89.6 s / block 24.7 s with `h2d_mb=47,055` unchanged -> the arena and the bank path stage in parallel (double bytes). Role-G5 is completing the consumer wiring; asset committed in `df1f45a` (default off) | section 6.39, g3-layerstage-b1 |
| L79 | Commits this stretch: `88600fa` (HC fusion gating, NO-GO), `c8ac79b` (fusion source), `df1f45a` (MTP verify CPU path + layer-stage wiring, both gated) | git log |
| L56 | K5 pair2 decode-attn bench (window B, A/A within 1%): bitwise=1 on all rows=1 shapes; 4096 -16.0%, 8192 -25.1%, 14336 -18.2%; pre-registered reject at 1834 (+8.0%, reproduced) -> blanket NO-GO; e2e projection <=3% < 5% gate; variant+bench committed `d34c09a` (opt-in AILA_ATTN_KVREUSE=pair2) | receipt-k5-pair2.json | 
| L57 | K2 rev2 QSA probe operational (both arms exit 0, breadcrumbs, oracle match; probe-fix doc confirmed). Route numbers NOT conclusive: 2048/2051/2052/8192 dense are linear (~0.066 us/token) but 4096 dense=2.44 ms breaks it 9x; 2052/8192 round1 stalls (routed 1.34-1.79 ms). M13 closed via --qsa-batches (selector 74-204 us rows=4; 131071/64 boundary = 9.3 ms stall). No QSA strategy change; E8 root-cause agent dispatched | receipt-k2-qsa-rev2.json |
| L58 | S2 service probe passed on retry (`run-s2-b2`): strict 1-token rotation (31/31 adjacent duplicates), aggregate 14.36 tok/s, per-session 7.18/1.81 (B still prefilling in-window), submitted median 63.06 ms -> M10 closed again, no decode batching for Qwen4Exp | service/run-s2-b2/result.json |
| L59 | Startup round 3 (D2): trim audit shows the per-layer working-set trim is load-bearing (2026-10-08: available 0.17 GiB, 120 s RPC timeout without it) -> default stays 1; new gates AILA_Q4E_ARENA_WORKING_SET_TRIM (default 1) + AILA_Q4E_HOST_ARENA_PREFETCH_LOOKAHEAD (=2 two-slot hint). Compiled clean; A/B/B/A queued behind the peer GPU token | qwen38-r1-startup-arena-2026-10-15.md S7 |
| L60 | Build-id drift incident + fix: any main-tree reconfigure re-bakes AILA_BUILD_ID from HEAD; Worker header 867c89a vs server 626fada -> handshake "identity/path isolation mismatch" (killed r26-look2-a1). Rebuild server via tools/server/build.ps1 -SkipDependencies. Also: a failed arm's leftover worker held ~52 GiB large pages -> later arena tail allocs die with 1450 until the process is killed | COORD-NOTE 20:05 |
| L61 | C4 queue-profiling gate committed (`f90af05`): AILA_Q4E_DEVICE_QUEUE_PROFILING now independent of the stage-event env with safe fallback; dev buckets need both | models/Model.cpp |
| L62 | E7 AOT feasibility: single-TU AOT smokes all exit 0 (JM TU 16.9 s, AttentionOps 17.9 s); `-c` stage emits bitcode only, ocloc runs at link/device-link (~337 images) -> unverified; -fsycl-targets requires -fsycl so a global flag injection is impossible; needs an opt-in CMake option + a ~5 min GPU link probe | qwen38-r1-aot-feasibility-2026-10-15.md |

### 6.40 2026-10-16 morning: three structural verdicts + the isolated supply gap

Worktree `2e25fb8` + the 00:34 build (worker `7a81823169ad`, server `79ac6ce953`).

**A. `AILA_Q4E_VERIFY_CPU_BATCH` is a NO-GO** (s2x A/B/B/A, MTP width 2,
`AILA_Q4E_VERIFY_CPU_COLD=1` on both arms): rates 15.437 / 14.998 / 15.662 /
15.535 -> A median 15.49 vs B median 15.33 (**-1.0%**) and SSE bodies are
byte-identical (`2f34f37e...`). The batched path did engage -
`[Q4E ArenaColdRowsEnabled] rows=2 tasks=12` (one job per layer instead of one
per row) - yet `cpu_join` rose 22.4 -> 30.5 ms and `cpu_compute` 27.1 ->
30.0 ms, so the verify window went 90 -> 96 ms. Default stays 0.

**B. Layer-stage overlap premise falsified in isolation** (`AilaQwen4ExpLayerStageBench.exe`):
load arm 150.5 ms alone, stage arm 121.3 ms alone (fill 74.6 + bankwait 39.2,
9.48 GB/s), overlap arm 257.6 ms. The inflation is a **constant ~107 ms** and
does not shrink with a longer load (150 -> 376 ms still inflates ~107 ms), so
the staging is serialized against the device work rather than hidden; the
bench gate itself prints `"verdict":"no_go"`. Together with the g5 e2e NO-GO
(section 8 of qwen38-r1-layer-stage-2026-10-15.md) the whole-layer arena is
closed.

**C. The isolated supply pass is 2x faster than production** (same 49.0 GB, same
banks=2/slots=16 fences): `AilaQwen4ExpPrefillBlockBench.exe --arms copy,block
--reps 3` -> wall **5.78 s**, fill 4.97 s, copy 4.96 s, bankwait 0.095 s,
copy_gbps 9.889, guards bitwise pass. Production costs `fill 7.19-8.66 s +
bank_wait 3.60-3.78 s`, supply ~11.5 s/block. The ~5.7 s gap is now the
largest single prefill number; role-P6 dispatched.

**Bound consequence**: device drain 8.58 s + DMA 4.96 s ~= **13.5 s/block ~=
300-310 tok/s** with the supply treated as additive (four receipts say H2D and
memory-bound kernels add). 400 tok/s needs bytes -35% or a consumer kernel
reading host experts directly across PCIe; that bandwidth is unmeasured
(role-P5 probe dispatched).

**T1 decode split at HEAD** (g5-a1/a2, count=1, 53-55 ms): `router_wait
11.2-12.2`, `experts_host 20.6-22.8` (`cpu_compute 13.5-15.7` + `cpu_join
8.8-12.2`, `cpu_tasks 197-290`), `hc_host 6.4`, `linear_host 3.2-3.6`,
`full_host 2.7-2.9`. Per layer that is ~1.12 ms of submit wall and ~0.21 ms of
it is spent in the CPU join.

| L80 | `AILA_Q4E_VERIFY_CPU_BATCH` NO-GO: batch engages (rows=2 tasks=12 per layer) but cpu_join 22.4->30.5, cpu_compute 27.1->30.0, window 90->96 ms, rate 15.49 -> 15.33 (-1.0%), SSE byte-identical; default stays 0 | s2x-a1..a2 |
| L81 | Layer-stage overlap falsified in isolation: load 150.5 / stage 121.3 / overlap 257.6 ms; inflation constant ~107 ms independent of load length; bench gate verdict no_go | ls-bench, q4e_layer_stage_bench |
| L82 | Isolated supply pass 5.78 s (fill 4.97 overlapping copy 4.96, 9.889 GB/s, bitwise pass) vs production supply ~11.5 s -> 5.7 s/block unexplained; role-P6 | block-bench-copy-block.txt |
| L83 | Bound: device 8.58 + DMA 4.96 ~= 13.5 s/block ~= 300-310 tok/s (supply additive per 4 receipts); 400 tok/s needs -35% bytes or in-kernel host reads (role-P5) | qwen38-r1-prefill-overlap-design-2026-10-16.md |
| L84 | Width economics (measured chain p1=0.7419/q2=0.5543/q3=0.5294): tokens/step 1.742/2.153/2.371; 20 tok/s needs W<=87.1/107.7/118.6 vs 112.2/164.2/223.4 current -> only width 2 is live and it needs -25 ms; per-row marginal 40-50 ms, 82% host | qwen38-r1-mtp-width-economics-2026-10-16.md |

### 6.41 2026-10-16 later morning: host-read probe, fill-gap resolution, merge NO-GO

**D. Host-USM read bandwidth (role-P5 probe, `AilaQwen4ExpHostReadBench.exe`).**
`[HR SUMMARY]` at 512 MiB / 1 GiB: `h2d 9.912/9.893`, `host_flat 9.799/9.814`,
`host_flat16 9.805/9.804`, `host_moe 9.845/9.760` GB/s against
`dev_flat 409/363`, `dev_moe 337/347` GB/s. Guards: `host-flat`/`host-flat16`
`guard_ok=1`, `dev-moe`/`host-moe` `guard_ok=0` (the moe pattern does not read
every word exactly once, so its coverage invariant fails while its bandwidth
matches the flat arms). Conclusion: a kernel reading host USM across PCIe
sustains the full PCIe rate, so mechanism Z cannot beat the DMA path; with
`Z2 = 8.03 + 50.27/R + 0.6` that is ~13.8 s/block ~= 297 tok/s. Three
independent lines now bound prefill at **13.0-13.8 s/block ~= 300-315 tok/s**;
400 tok/s needs bytes -35%.

**E. The 5.7 s/block supply gap was a false comparison plus F2 (role-P6).**
- The isolated BlockBench fill (4.97 s) reads an 81 MB heap pool that stays hot
  in L3; its `copy_gbps` is DMA-paced (`fill 4.953 ~= copy + wait`). Production
  reads 49 GB of *cold private large pages*, so the two are not comparable.
- Production fill rate: single-thread **6.4-6.5 GB/s** = 7.19-7.33 s; the same
  arena with the 16-way TBB path is **18.9 GB/s = 2.47-2.52 s**.
- F2: `bank_wait 3.74-3.91 s` is the host waiting for that bank's H2D, which is
  queued behind the real consumer kernels in the layer's in-order queue; the
  bench's consumer is 0.58 ms/chunk with an empty queue (0.095 s).
- Page-eviction hypothesis refuted at source level: the arena is
  `VirtualAlloc(MEM_LARGE_PAGES|MEM_COMMIT)` (non-pageable), and
  `AILA_Q4E_HOST_ARENA_PREFETCH_ASYNC` only issues `PrefetchVirtualMemory` on
  the SafeTensors mmap during load.

**Fix 1 (measured, bit-exact, zero new mechanism)**: `AILA_Q4E_STAGE_COPY_PARALLEL=1`
plus `AILA_Q4E_PREFILL_STAGING_BANKS=4` -> fill 7.19/7.33 -> 2.47/2.52 s, block
19.05 -> 18.08 s, TTFT 14000 69.6 -> 65.9 s. It is not the section 2 item
(that one is "pin the whole 51 GiB" -> `UR_OUT_OF_RESOURCES`); this only
parallelises the existing memcpy into the existing 77.3 MiB pinned banks.
Adoption A/B/B/A (`pfx-*`) run this hour.

**F. `AILA_Q4E_VERIFY_CPU_MERGE` NO-GO (role-P3 implementation + this run).**
The merge groups the window's cold CPU pairs by expert so one weight stream
serves several rows. Measured merge factor from 7,200 builds:
`cum_pairs=89,274 cum_groups=75,556` -> **f = 0.154**, far below the 0.50 the
economics needed. A/B/B/A (all with `VERIFY_CPU_COLD=1 VERIFY_CPU_BATCH=1`):
A 16.348 / 16.275 vs B 16.144 / 16.230 -> **-0.75%**, all four SSE bodies
byte-identical (`2f34f37e...`). The router correlation between a token and its
own draft is the limiting factor, not the implementation.

| L85 | Host-USM kernel read = 9.76-9.85 GB/s (flat arms guard-passed, h2d 9.89): mechanism Z cannot beat the DMA lane; prefill bound 13.0-13.8 s/block ~= 300-315 tok/s; 400 needs bytes -35% | qwen38-r1-hostusm-read-2026-10-16.md |
| L86 | Supply gap resolved: bench fill was L3-paced (81 MB hot pool) vs 49 GB cold large pages; production single-thread fill 6.4-6.5 GB/s vs 16-way TBB 18.9 GB/s; F2 = bank_wait waits for H2D behind consumer kernels; eviction hypothesis refuted by source | qwen38-r1-prefill-fill-gap-2026-10-16.md |
| L87 | Fix 1 `STAGE_COPY_PARALLEL=1` + `BANKS=4`: fill 7.19/7.33 -> 2.47/2.52 s, block 19.05 -> 18.08 s, TTFT 69.6 -> 65.9 s, bit-exact; adoption A/B/B/A pfx-a1..a2 | fill-gap doc S6 |
| L88 | `AILA_Q4E_VERIFY_CPU_MERGE` NO-GO: measured merge factor f=0.154 (75,556/89,274 over 7,200 builds) vs the 0.50 needed; A/B/B/A 16.31 -> 16.19 (-0.75%), SSE byte-identical; gate stays 0 | s4-a1..a2, qwen38-r1-cpu-merge-impl-2026-10-16.md |

**Fix 1 adoption verdict (pfx A/B/B/A, 2026-10-16 ~11:05) - NO-GO, stays opt-in.**
A arms TTFT 14000 = 69.6 / 69.6 s, last block `submitted` 19,887 / 19,892 ms
(`fill 7,371 / 7,382`, `bank_wait 3,710 / 3,697`); B arms TTFT 68.9 / 69.6 s,
`submitted` 19,720 / 19,728 ms (`fill 2,468 / 2,485`, `bank_wait 6,967 / 6,914`).
All four SSE bodies byte-identical (`8a64a37d9143`). So the parallel fill really
does take the fill from 7.37 to 2.48 s, but `bank_wait` absorbs +3.25 s and the
block wall improves by only **-0.84%** - far below the +5% default-value gate.
The earlier "-5.1% TTFT" reading does not reproduce at this config; the
conservation law (fill <-> bank_wait) holds. Both flags remain opt-in.

| L89 | Fix 1 adoption A/B/B/A NO-GO: fill 7.37 -> 2.48 s but bank_wait 3.70 -> 6.94 s; block wall -0.84% (19,887 -> 19,720 ms), TTFT 69.6 -> 68.9/69.6 s, bit-exact; flags stay opt-in | pfx-a1..a2 |

**Decode tail round (P7) and the last two levers.**

- **C1 `AILA_Q4E_LM_HEAD_Q8=1`** (already wired, default off): A/B/B/A with the
  production bundle and `--generate=32`: A 17.413 / 17.49, B 17.974 / 18.541
  -> **+4.62%** mean, output sha identical in this test (the argmax survives
  head quantization). Its sample window drops 3.35 -> ~2.1 ms (the head is
  1271 MB bf16 at 470 GB/s). Below the +5% default gate, so it stays opt-in,
  and it disables MTP by construction (Model.cpp requires !lm_head_q8).
- **C2 speculative CPU precompute from the previous route**: read-only counter
  arm (`AILA_Q4E_ROUTE_REPLAY_COUNT=1`, bundle env) gives
  `overlap_rate 0.19-0.31`, **`cold_cover 0.086-0.208`** (median ~0.12) over
  5 sampled positions - far under the 0.70 gate, so expected saving is
  0.2-0.7 ms. **Closed; not implemented.**
- Harness note (P7): `--diag` / `AILA_PROFILE_Q4E_DECODE=1` adds ~3.5 ms of
  tail to a decode arm (a `ctx.synchronize()` at Backend.cpp:3124); decode
  arms should use `--production-timing`. The probe's rate formula also carries
  a ~1.65 ms pseudo-error (the finalize flush lands inside `duration`).

| L90 | `AILA_Q4E_LM_HEAD_Q8` A/B/B/A +4.62% (17.451 -> 18.258 tok/s), sample 3.35 -> ~2.1 ms, output identical in test; below the +5% gate -> opt-in, and incompatible with MTP | p7-a1..a2 |
| L91 | Speculative route replay closed: `cold_cover` 0.086-0.208 (median 0.12) vs the 0.70 gate -> expected 0.2-0.7 ms; counter shipped, precompute not built | p7r-b1 |
