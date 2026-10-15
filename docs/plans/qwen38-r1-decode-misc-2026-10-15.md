# E5: decode device-side remaining items (hc / linear / shared / head)

Date: 2026-10-15. Repo: `E:\RiderProjects\Aila`, HEAD `3ef4606`, pair
`retained27`. Owner: subagent E5. Scope: audit + bench + one opt-in variant.
Zero GPU, zero commit, zero build by this agent (a syntax-only compile check
of the two new TUs was run with the production toolchain; no artifacts, no
build directory touched). ASCII only.

## 1. Per-token device budget (T1, rows=1)

Sources. Device spans (`dev_*_ms`): `tmp/q4e-sprint-r1-20261014/c2-round2-b{1,2}-worker.log`,
`[Q4E AdaptiveWork]` rows with `count=1`, n=128 each; b1/b2 are the same
configuration (an A/A pair). Parser: `tmp/q4e-sprint-r1-20261014/decode-misc/parse_devms.py`.
Observer cost of `AILA_Q4E_DECODE_STAGE_EVENTS=1` (D1,
`docs/plans/qwen38-r1-decode-boundary-2026-10-15.md` section 1): +11-13 ms of
token wall, so raw spans are not clean ms. Clean reference for the same
harness: `d1-a1b`/`d1-a2` submitted medians 60.71/60.19 ms (E2,
`docs/plans/qwen38-r1-decode-20-2026-10-15.md` section 1.1). Scale factor
0.82 is therefore applied to the A/A mean for "clean" columns. `[estimate]`

| Span | b1 ms | b2 ms | A/A mean | clean = mean x 0.82 | note |
|---|---:|---:|---:|---:|---|
| hc_dev | 10.426 | 10.079 | 10.25 | 8.4 | 97 HC calls x 2 kernels = 194 submits |
| linear_dev | 8.075 | 8.049 | 8.06 | 6.6 | 36 layers x (all_proj GEMV + GDN + o_proj GEMV) |
| full_dev | 5.541 | 5.440 | 5.49 | 4.5 | 12 layers (not this role; K3 owns it) |
| router_dev | 2.847 | 2.638 | 2.74 | 2.2 | MoE front + top-k |
| shared_dev | 3.370 | 3.311 | 3.34 | 2.7 | 48 layers x 3 kernels = 144 submits |
| experts_dev | 28.317 | 26.855 | 27.59 | 22.6 | C1 closed the kernel line; CPU-bound |
| resid_dev | 0.309 | 0.309 | 0.31 | 0.25 | fused residual add |
| inter_dev | 8.858 | 8.794 | 8.83 | 7.2 | PLE wait window (D1 root cause) |
| ple_dev | 0.261 | 0.261 | 0.26 | 0.21 | PLE-only marker window |
| stagein_dev | 2.106 | 1.934 | 2.02 | 1.7 | per-layer resident-input D2H |
| prerouter_dev | 0.066 | 0.066 | 0.07 | 0.05 | flag tail |
| sum | 73.50 | 71.04 | 72.27 | 59.3 | submitted medians 74.49 / 72.90 |

This role's four families: hc 8.4 + linear 6.6 + shared 2.7 + head (from the
same logs: `head_dev` 3.322/3.299 -> clean 2.7) = 20.4 ms of ~59.3 ms
`[estimate]`. The `head_dev` span covers the final mixer HC + LM head
(`Qwen4ExpBnb4Backend.cpp:2568-2611`), so ~0.1 ms of it is HC.

## 2. Layouts, kernel counts and byte floors

Production shapes (config.json + `Qwen4ExpBnb4Backend.cpp` load code):
H=2560, hc groups=4, group_dim=2560, lowrank=320, 512 experts top-10,
I=640, shared I=640, vocab=248320, 48 layers (36 GDN + 12 full).
Streaming bound 291.11 GB/s (K1,
`docs/plans/qwen38-r1-kernel-2026-10-14.md`, measured continuous read).

| Family | Kernels / token | Bytes / token | Floor @291.11 GB/s | Clean measured | Ratio | Verdict |
|---|---:|---:|---:|---:|---:|---|
| hc | 194 | 1279 MB | 4.39 ms | 8.4 ms | 1.9x | only family with a device-side lever |
| linear | ~108 | 1408 MB | 4.84 ms | 6.6 ms | 1.36x | NF4 GEMV family closed (section 2 / P4 / K1); leave |
| shared | 144 | 133 MB | 0.46 ms | 2.7 ms | 6.0x | not bandwidth-bound: 144 submits at 25-30 us is the cost |
| head | 1 (+2 HC) | 1271 MB | 4.37 ms | 2.7 ms | <1.0x | already faster than the 291 GB/s bound (approx 470 GB/s); exclude |

Byte arithmetic.

- hc: W_down 320x10240x2 = 6.554 MB + W_up 10240x320x2 = 6.554 MB +
  W_inject 4x10240x2 = 0.082 MB (96 of 97 calls) = 1279.2 MB.
- linear: all_proj NF4 [16480,2560] = 21.09 MB packed + 2.64 MB absmax;
  o_proj NF4 [2560,6144] = 7.86 + 0.98 MB; GDN state read+write
  2 x 48x128x128x4 = 6.29 MB; conv ring ~0.25 MB -> 32.6 MB/layer x 36 =
  1173 MB weights + 227 MB state = 1408 MB.
- shared: gate_up NF4 [1280,2560] = 1.64 + 0.20 MB absmax; down NF4
  [2560,640] = 0.82 + 0.10 MB; gate bf16 0.005 MB -> 2.77 MB/layer x 48.
- head: bf16 [2560,248320] x 2 B = 1271.4 MB. The Q8 head (opt-in,
  `AILA_Q4E_LM_HEAD_Q8`) halves the dominant term to ~675 MB but is gated by
  MTP enablement and is out of scope here.

Other evidence that hc is not at its floor (per-call units): floor 45.3 us per
HC call (13.19 MB); clean measured 86.6 us per call. Two clean submit-side
numbers land at ~25-32 us per kernel submit (`hc_host` 6.17 ms / 194 submits;
`shared_submit` 3.5-4.3 ms / 144 submits), i.e. for these small kernels host
submit time and device time are the same order of magnitude. Which one binds
is exactly what the bench in section 3 measures.

## 3. Bench: `perf/qwen4exp/q4e_decode_misc_bench.cpp`

Production code paths, synthetic weights only (no model file). Every arm
reports median `host_ms` (time issuing the batch, no drain), `span_ms`
(profiling events between two `ext_oneapi_submit_barrier` calls) and
`wall_ms`; `gbps_span` and `host_us_per_kernel` are derived. A raw
`submit_cost` arm submits `--submit-kernels` trivial kernels and reports
us/submit on the same in-process in-order queue, which calibrates the host
side of every other arm.

Arms:

- hc: single down (wg 256/128/64), single up (prod / streams / pl2 / pl3),
  and token arms = 96 inject pairs + 1 mixer pair (194 kernels) for prod,
  streams, pl2_wg256, pl2_wg128, pl3_wg256.
- gdn: one decode call, and 36 calls (36 linear layers) per token.
- shared: fused gate/up/SwiGLU, down, stack, and stack x 48.
- head: bf16 head, and optionally `head_q8`.
- submit: trivial-kernel submit cost.

Guards (fail closed): hc pl1/pl2/pl3/streams must be bitwise identical to the
production kernels (act bf16, inj f32, x bf16); the shared stack must be
deterministic bitwise and its down projection must match a CPU f64
dequantized oracle (NRMSE < 5e-2, sampled rows); GDN output must be finite.
`AILA_Q4E_*` env vars are not needed by the bench (it calls ops directly).

Coordinator command (GPU mutex precheck first; no fixed sleeps):

```powershell
Get-Process AilaWorker,AilaServer,Aila -ErrorAction SilentlyContinue   # must be empty
pwsh -NoProfile -File build.ps1 -BuildDir build-refactor
build-refactor/AilaQwen4ExpDecodeMiscBench.exe --arms all --samples 20 --warm 3 *> tmp/q4e-sprint-r1-20261014/decode-misc/misc-A1.log
build-refactor/AilaQwen4ExpDecodeMiscBench.exe --arms all --samples 20 --warm 3 *> tmp/q4e-sprint-r1-20261014/decode-misc/misc-A2.log   # A/A
```

Read order and decision rules:

1. `[MISC] submit_cost ... host_us_per_kernel` - if ~>=25 us, the decode loop
   is submit-throughput limited and hc/shared device work is partly masked.
2. `[MISC] hc_parity_*` - any `bitwise=0` fails the bench (stop).
3. `[MISC] hc_down_*` / `hc_up_*` - device cost per kernel vs the 22.6 us
   per-kernel floor.
4. `[MISC] hc_token_prod` vs `hc_token_streams` / `hc_token_pl2_*` -
   `span_ms` is the kernel lever; `host_ms` is the submit lever.
5. `[MISC] shared_token48`, `gdn_token36`, `head_bf16`, `head_q8`.
6. `[MISC GATE] hc_token_*` - kernel >=3% and projected e2e >=5% (charter
   section 5.3). Both PASS is required to promote a variant to default.

## 4. Variant: `src/ops/HcDecodeVariant.{hpp,cpp}` (opt-in, not wired)

`hc_decode_down_inject_pl` / `hc_decode_up_mix_pl`: same arithmetic as the
production kernels (identical per-lane element order for every FMA, same
`reduce_over_group`, same BF16 rounding points); only the vec8 weight and
activation loads are software pipelined with an explicit load-ahead depth
(1..4; 1 reproduces the production shape). Work-group is selectable
(64/128/256) for the down kernel. Rationale: for a rows=1 GEMV the bounding
resource is memory-level parallelism per lane; the production loop holds one
vec8 pair in flight (16 B/lane), while the pipelined loop holds `depth` of
them without changing the summation order.

Limits / TODOs:

- rows=2 (MTP verify rows) is not implemented; MTP work must first show a
  win at rows=1 and then this variant needs a rows variant.
- `hc_decode_up_mix_pl` pipelines within one g only; cross-g load hoisting
  (issue g+1 loads while reducing g) is the next step if pl2/pl3 is flat.
- The compiler may already hoist some loads; the bench decides, not theory.

## 5. Exclusions (spend no more time)

- head: measured clean 2.7 ms is below the 291.11 GB/s floor (4.37 ms),
  i.e. the BF16 head GEMV already streams at roughly 470 GB/s. Even a
  perfect 560 GB/s device bound would save only ~0.44 ms (0.75% e2e).
  EXCLUDE from kernel work; the only real head lever is the existing Q8
  opt-in, which is gated by MTP enablement.
- linear: 1.36x from the byte floor and 195 GB/s effective on the NF4 GEMVs,
  which is the same rate the K1/P4 audits measured for the closed NF4 GEMV
  family (section 2 denylist + P4). Best case if fully closed is ~1.8 ms
  (3.0% e2e) and the mechanisms are already denied -> EXCLUDE.
- shared: 6.0x from the byte floor, but the absolute gap (2.2 ms, 3.8% e2e)
  is almost exactly 144 kernel submits at 25-30 us. No kernel variant can
  recover it; it needs a command-count/fusion lever, which is the
  coordinator's submit-amortization line, not a device-side variant.
- hc is the only one of the four with both a device-side gap (1.9x) and a
  bit-exact lever; it is also the single largest clean span after experts
  (8.4 ms, 14%) -> the bench in section 3 is the gate for it.

## 6. Evidence index

- Spans and host splits: `tmp/q4e-sprint-r1-20261014/c2-round2-b{1,2}-worker.log`.
- Observer effect and PLE root cause: `docs/plans/qwen38-r1-decode-boundary-2026-10-15.md`.
- Clean T1 table (58.79 ms, hosts): `docs/plans/qwen38-r1-decode-20-2026-10-15.md`.
- Byte/roofline conventions and 291.11 GB/s: `docs/plans/qwen38-r1-kernel-2026-10-14.md`,
  `docs/plans/qwen38-performance-research-charter-2026-10-14.md` sections 2 and 5.
- NF4 fixture recipe: `tests/inference/DecodeRowProjectionTests.cpp:26,97-140,255-320`.
- Parser: `tmp/q4e-sprint-r1-20261014/decode-misc/parse_devms.py`.
