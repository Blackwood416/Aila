# Qwen3.8-Flash-Next CPU Expert Alignment

Date: 2026-09-28. Working tree over f78732d. Windows, A770 driver
32.0.101.8860, oneAPI 2026.1, Xeon E5-2696 v3 AVX2/FMA.
Model: `C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2`.

## Findings

The previous CPU route used a different accumulation order from the GPU.
It also permitted contraction of nested scale reconstruction into one FMA.
The production GPU deliberately uses two f32 roundings for that scale.

1. A synchronous, test-only executor observer samples real CPU-candidate
   experts while their host lease is held. The diagnostic uploads those same
   packed bytes/metadata and input to private test buffers, then compares
   GPU nested/expanded, CPU nested/expanded, h, down on identical h, and full
   expert output. It does not acquire/evict GPU cache entries.
2. Before the scale repair, CPU nested versus explicitly expanded down
   differed even on identical h. A volatile intermediate now enforces two
   f32 roundings. Strict scalar/AVX2/TBB tests at m=1/4 and sixteen real
   samples pass nested-versus-expanded bit comparisons after this repair.
3. Scale repair alone did NOT fix generation: story step 26 remained
   expected 23221 (` beneath`) versus actual 3808 (` across`).
4. The first diagnostic lane reference used `std::fma` without AVX2 flags.
   Its results matched explicitly nonfused multiply/add, not GPU FMA.
   The test target now enables AVX2/FMA and uses `_mm_fmadd_ss`. No global
   production compiler flags changed. Earlier lane/tree logs are retained
   but must not be used to infer the GPU arithmetic contract.
5. With explicit FMA, 16 real samples match every GPU lane partial, and
   the 8,4,2,1 tree matches every FP32 down output. Instrumented test GPU
   down also matches the production kernel on every sample. Other trees
   do not. This is A770-stack evidence, not a portable SYCL reduction rule.
6. One BF16 h value still differed when the host nonlinear expression was
   left to fast-math. The candidate uses explicit scalar division after
   exp and the GPU-ordered dot. Sixteen sampled candidate h and same-h down
   comparisons are then bit-exact. This combination was verified; the exact
   contribution of host exp versus reciprocal reassociation was not isolated.
   `std::exp` and GPU native exp are not guaranteed identical for all inputs.

The diagnostic is `AilaQwen4ExpGenerationTests --diagnose-cpu <model>`, with
CPU dispatch, one credit, ReadFile and ready-first enabled. It runs after
normal warmup, stops after sixteen tasks and is NOT an acceptance or timing
run. Its synchronization affects scheduling; samples are not asserted to be
the exact same schedule as uninstrumented generation. Diagnostics do not
prove a complete first-divergence trace at every layer/token.

## Candidate Implementation

`CpuKernelBackend::GpuOrderAvx2` is opt-in through
`AILA_Q4E_CPU_GPU_ORDER=1` in the executor. It changes neither Auto nor the
default CPU-dispatch setting. Fixed contract: H=2560, I=640, NF4 block=64,
BF16 activations/h, FP32 accumulation/down, m=1..4. Unsupported nonempty
shapes throw before reading buffers. Null/empty and zero-weight behavior
is covered by tests.

Two AVX2 vectors emulate sixteen GPU lanes. Each lane consumes eight
adjacent values per 128-column step using hardware FMA. Packed weights
remain row-major NF4; no repacking or expanded weight cache is introduced.
The final tree uses explicit rounded additions. TBB distributes output rows;
the existing scheduler and thread budget have not been redesigned.

The scale repair applies to all existing CPU backends. The ordered dot and
explicit nonlinear division are confined to the opt-in candidate. GPU
kernels are unchanged. No new default or universal exactness claim.

## Acceptance and Screening

Same pre-boundary-hardening Aila.exe SHA256:
`6BEB4ED48F1D049C3050F507AD99B7302961A81F7ECCC29D83559305F9EA47B9`.
Story 30 input tokens, reusable-prefix 26+4, greedy, thinking budget 0,
max sequence 512, 48 generated tokens. ReadFile and token/profiling logs
enabled. Load/warmup excluded from GenerateMessages. All GPU workloads serial.

| Configuration | GenerateMessages ms | Golden 48 IDs |
|---|---:|---|
| GPU-only reference | 5727.43 | Match |
| Ready-first + CPU1 + GPU-order, run 1 | 6078.45 | Match |
| Same candidate, run 2 | 6059.31 | Match |
| Same candidate, run 3 | 6010.05 | Match |

The original step-26 deviation is absent in all three candidate runs.
Run 1 actually executed 161 CPU tasks (cumulative including warmup),
116.48 ms CPU arithmetic. The full real-model acceptance also passed:
248320 BF16 logits for skip-logits/rollback, two-turn reuse, edited-history
rollback, snapshot-policy transitions. H6c anchor-2/16 diagnostic differences
remain unchanged and unresolved.

Candidate median 6059.31 ms, MAD 19.14 ms, range 6010.05-6078.45 ms.
Only one same-binary GPU baseline was run: this is negative screening, not
a paired speedup study. Candidate median is about 5.8% above that reference.
Keep CPU dispatch, ready-first and GPU-order OFF by default. Do not compare
against failed-token CPU variants to advertise a speedup.

Logs and metadata:

- `tmp/q4e-cpu-diagnostic-before.log`: original CPU scale mismatch. Its GPU
  expanded comparison mistakenly retained nested metadata, so that particular
  column is invalid; later diagnostics clear metadata before expanded dispatch.
- `tmp/q4e-cpu-diagnostic-scale-fixed.log`: repaired scale, remaining dot/h differences.
- `tmp/q4e-cpu-diagnostic-{lanes,trees,partials,arithmetic}.log`: preserved
  intermediate negative experiments; host lane FMA reference not yet valid.
- `tmp/q4e-cpu-diagnostic-fma.log`: explicit-FMA lane match and nonlinear residual.
- `tmp/q4e-cpu-diagnostic-candidate.log`: 16/16 h and same-h down exact.
- `tmp/q4e_prof_cpu_scale_{base,cpu1}.*`: scale-only golden gate failure.
- `tmp/q4e_prof_gpuorder_{base_1,cpu1_1,cpu1_2,cpu1_3}.*`: complete traces,
  hashes, environment and offline comparator results.
- `tmp/q4e-cpu-gpuorder-real.log`: explicit real-model acceptance.

### Final Boundary-Hardened Build

Final Aila.exe SHA256:
`964FE9384DCEF28F59A365498C48BF396517091F6678C9C104211EED62496EC8`.
Adds candidate null/empty/zero-weight handling and invalid-capacity tests;
diagnostic nonfinite checks use bit patterns so fast-math cannot remove them.
The full build passed (`tmp/q4e-cpu-final-build.log`). All eight Q4E CTests
passed serially in 62.22 s (`tmp/q4e-cpu-final-tests.log`). No full-repository
CTest run was performed.

One final same-binary scheduling screen, all golden IDs exact:

| Configuration | GenerateMessages ms | Relative to GPU-only |
|---|---:|---:|
| GPU-only | 5687.87 | reference |
| CPU1 GPU-order, original acquire order | 5963.22 | +4.84% |
| CPU1 GPU-order, ready-first | 6021.74 | +5.87% |

Logs: `tmp/q4e_prof_gpuorder_final_{base,cpu_acquire,cpu_ready}.*`.
This is one sample each, not a statistically established dispatch selection.
Ready-first is not required for correctness and did not improve this screen.
Five uninstrumented candidate story runs across the two builds passed the
golden gate; the real multi-turn acceptance preceded the final boundary-only
hardening. Full-model CPU/GPU logits equality on arbitrary prompts is not proven.
`git diff --check` passes. No commit and no default promotion in this delivery.

## Next Gate

Later implementation: [CPU/fill overlap results](qwen3.8-flash-next-cpu-overlap-2026-09-28.md).
Overlap is now implemented behind an OFF-by-default flag, with corrected
join-time throttling. Three paired story measurements still regress by
3.21% versus GPU-only; high-dispatch numerical equivalence remains unproven.
The following paragraph records the pre-overlap next gate.

With this bounded candidate, correctness is good enough to investigate
performance but not to claim universal bitwise identity. Profile actual
CPU/fill/compute overlap and extra ordered-merge/ready-first launches.
CPU still starts after synchronous acquire. Fix start timing and quantify
host-memory contention before adding credits or threads. Preserve the
golden-token and multi-turn gates on each structural change.
