# 2026-06-28 TTS Mimi backend optimization recon

Scope: reconnaissance for Qwen3-TTS Mimi decode latency/RTF work on
`src/models/Qwen3TTSBackend.cpp`, with no main-path behavior change. SYCL Graph
is intentionally out of scope because the current oneAPI 2025.3.3 + Arc A770
probe only exposes `ext_oneapi_limited_graph`, and graph replay was slower than
ordinary submit.

## Current hot path shape

Incremental streaming enters `decode_mimi_incremental`:

1. Host codec codes -> `codes_dev`, VQ lookup/add for first/rest codebooks, two
   Mimi projection linears, then copy/add into `latent_new`.
2. Append `latent_new` into `MimiStreamState::latent_buffer`.
3. Run pre-conv over the new frames plus 2 latent frames of causal overlap.
4. Run pre-transformer on new frames with persistent per-layer K/V cache, then
   append `pre_tfm_out_new` into `pre_tfm_out_buffer`.
5. Run `mimi_conv_stages` over a window from
   `AILA_TTS_MIMI_CONV_WINDOW_FRAMES` history plus new frames, then read back
   only `new_frames * kMimiSamplesPerFrame`.

The latest useful profile logs show the current split:

- First 8-frame chunk: `Mimi incremental total` around 119-128 ms, with
  `pre-transformer cached` around 21-25 ms and `conv+readback` around 96-100 ms.
- 16-frame window: `conv+readback` around 188-194 ms.
- 24-frame window: `conv+readback` around 277-281 ms, with occasional noisy
  upsample spikes.
- 32-frame window / 24+8 history path: `conv+readback` around 370-375 ms.
- VQ+projection is usually around 1-2 ms and pre-conv around 0.2-0.5 ms after
  warmup, so they are not first-order TTFA bottlenecks today.

Evidence logs inspected:

- `tmp/alia-real-smoke/tts_fused_residual_embed_short.log`
- `tmp/alia-real-smoke/tts_initial4_fused_embed_short.log`
- `tmp/alia-real-smoke/voice_matrix/*.log`

## Ranked candidates

### 1. Reuse Mimi conv-stage scratch buffers

Candidate: add a `MimiConvScratch` or extend `MimiStreamState` with reusable
buffers for the shapes used by `mimi_conv_stages`: upsample inputs/outputs,
ConvNeXt intermediate buffers, decoder block residual/activation/conv buffers,
and final `dec6_out`. The current path allocates many short-lived `Tensor`s per
chunk; `Tensor::allocate` is a raw `sycl::malloc_device` plus allocation-map
mutex, and `Tensor` destruction frees immediately.

Expected benefit:

- First 8-frame chunk: likely 5-20 ms, mostly lower launch-side and allocator
  overhead.
- Later 24+8 windows: likely 10-40 ms, with less variance from allocation/free.

Risk:

- Numerical/audio content should be unchanged if buffers are only reused and
  fully overwritten before read.
- Lifetime risk is real: current synchronizes are partly protecting queued
  kernels from storage being freed. Reused buffers must be double-buffered or
  scoped with explicit completion where a buffer is overwritten before dependent
  work is complete.

Validation needed:

- Build `AliaEngine`.
- Short smoke with `AILA_TTS_PROFILE=1` and default settings.
- Voice matrix, because allocator timing interacts with output length and
  foreground/TTS contention.
- Mimo-ASR on generated wav and compare against `foreground_assistant_text`.

Default:

- Default-off probe first, for example `AILA_TTS_MIMI_CONV_SCRATCH=1`.
- Consider default-on only after matrix + Mimo-ASR pass and no tail/window
  corruption.

### 2. Use existing `snake_causal_conv1d` for decoder residual `act1 + conv1`

Candidate: `src/ops/ConvOps.cpp` already has `ops::snake_causal_conv1d`, but
Mimi decoder residual blocks still run `snake_beta(xx_cur -> xx_act1)` followed
by `causal_conv1d(xx_act1 -> conv1_out)`. A default-off switch could route only
the `conv1` side of each residual block through the fused kernel and remove the
`xx_act1` write/kernel.

Expected benefit:

- First 8-frame chunk: likely 5-15 ms if the fused kernel is competitive.
- 24+8 steady window: likely 20-60 ms because there are 4 decoder stages x 3
  residual blocks.

Risk:

- Medium. The existing fused kernel appears to use the legacy
  `[out_ch, in_ch, kernel_size]` weight indexing and scalar channel loops. Mimi
  conv weights are transposed at load to `[out_ch, kernel_size, in_ch]` for vec8
  access. The fused kernel must either be updated for the transposed layout or
  guarded to avoid wrong audio.
- Floating-point order changes inside SnakeBeta + conv. Output should be close
  but not bit-identical.

Validation needed:

- Build.
- A/B short smoke with `AILA_TTS_PROFILE=1`.
- Matrix with `short_hello`, `persona_chat`, and `long_answer`.
- Mimo-ASR on at least `short_hello` and one longer matrix output.
- Optional waveform or sample-level comparison against baseline to catch
  obvious corruption before ASR.

Default:

- Default-off only until the fused kernel has layout-correct tests or real audio
  validation. Not suitable for immediate default-on.

### 3. Reduce pre-transformer cached-path synchronizations

Candidate: `decode_mimi_incremental` still synchronizes after nearly every
small pre-transformer operation: residual copies, RMS norms, q/k/v linears,
RoPE, cache copies, attention, projections, layer scale, MLP linears, SwiGLU,
and residual copies. Many of these are dependency-ordered on the same queue and
do not need host waits unless storage is about to be freed or host timing must
split stages.

Expected benefit:

- First 8-frame chunk: likely 5-15 ms, capped by current 21-25 ms
  pre-transformer cost.
- Later chunks: similar per Mimi callback; lower host overhead and less GPU
  bubble risk.

Risk:

- Low to medium. Math should not change, but removing waits can expose lifetime
  bugs because many temporaries are destroyed at loop iteration boundaries.
  This pairs best with reusable scratch or a conservative wait at each layer
  boundary.

Validation needed:

- Build.
- Short smoke with `AILA_TTS_PROFILE=1`.
- Matrix, because queue overlap can change contention with foreground decode.
- Mimo-ASR for output recognizability.

Default:

- Start as default-off, for example `AILA_TTS_MIMI_PTFM_SYNC_CLEANUP=1`.
- Could become default-on if the change is only wait removal with proven tensor
  lifetime safety.

### 4. Fuse scale + residual add + copy in pre-transformer

Candidate: replace patterns such as
`apply_layer_scale_gpu(proj_out); residual_add(residual, proj_out);
copy_tensor(residual, x)` and the MLP equivalent with a single kernel writing
directly to `x`. This is a small-kernel fusion and removes one full copy per
attention block and one per MLP block.

Expected benefit:

- First 8-frame chunk: likely 2-8 ms.
- Later chunks: similar per callback; more valuable if sync cleanup exposes
  kernel-launch overhead.

Risk:

- Low to medium. Numerics may differ only by bf16 write ordering if implemented
  as `x = residual + scaled`, but that should match the intended value.
- Needs care where the existing code relies on `residual`/`mlp_res` contents
  for subsequent operations.

Validation needed:

- Build.
- Short smoke with profile.
- Matrix and Mimo-ASR before default-on.

Default:

- Default-off probe first. Default-on is plausible after validation because the
  semantics are local.

### 5. Fuse VQ lookup/add and latent projection add

Candidate: VQ lookup currently runs one first-codebook lookup, one rest lookup,
14 rest accumulation kernels, two projection linears, then copy+residual add
into `latent_new`. VQ is already only around 1-2 ms, but it launches many small
kernels. Possible probes: combine all rest codebooks in one kernel, and write
`proj_first + proj_rest` directly into the stream latent buffer instead of
`latent_new` plus async copy.

Expected benefit:

- First chunk and later chunks: likely 1-4 ms total.

Risk:

- VQ rest accumulation order changes can alter bf16 rounding. Audio content
  should be nearly identical, but codec-to-audio sensitivity is unknown.
- Direct write into `latent_buffer` must preserve lifetime and capacity checks.

Validation needed:

- Build.
- Short smoke with profile.
- Matrix and Mimo-ASR, because small rounding differences can still be audible.

Default:

- Default-off probe only. Not the best first target because the measured cost is
  small.

### 6. oneDNN Linear per-call overhead audit

Candidate: Mimi linears already use persistent `Linear` wrappers with cached
oneDNN primitives, memory objects, args maps, and user scratchpads keyed by
`seq_len` and bias dtype. For `seq_len > 1`, the primitive is not rebuilt after
first use. Remaining overhead is mostly `set_data_handle`/arg updates and oneDNN
execute overhead on small batches.

Expected benefit:

- Unknown but probably low for first-audio Mimi relative to conv decoder:
  0-10 ms unless a specific per-call stall is measured.

Risk:

- Low for measurement-only probes.
- Medium if replacing oneDNN with custom small GEMM kernels; high chance of
  performance regression.

Validation needed:

- Add timing probe around `Linear::forward(_bias)` for Mimi only, default-off.
- Build and short profile.

Default:

- Measurement probe can be default-off.
- Do not default-on a custom replacement without strong matrix evidence.

### 7. More exact Mimi conv state carry / smaller tail decode window

Candidate: current incremental conv path replays a fixed input-frame history
window, default 24. A deeper stateful convolution implementation could carry
the exact needed residual/decoder state between chunks and decode only the new
tail. A smaller history window could also be probed.

Expected benefit:

- Potentially large for long answers: later chunks could drop from ~370 ms
  conv/readback toward first-window scale if exact state carry works.
- First-audio benefit is limited because the first chunk has no previous state.

Risk:

- High. Prior docs already note the conv-window change needed ASR validation
  because nonzero matrix audio was not enough. Incorrect receptive field/state
  produces recognizable but wrong or shifted audio.

Validation needed:

- Build.
- Short smoke.
- Full matrix.
- Mimo-ASR on outputs and comparison to `foreground_assistant_text`.
- Prefer an offline waveform equivalence harness against full-history decode
  before any product-path use.

Default:

- Default-off research branch only. Not default-on without strong audio
  equivalence evidence.

## Recommendation

The next practical probe should be small and default-off:

1. First try the existing `snake_causal_conv1d` route only after fixing/guarding
   its weight layout for Mimi transposed conv weights. This directly targets the
   decoder blocks that dominate conv time.
2. In parallel or next, add reusable conv scratch buffers. This is less likely
   to change audio but touches lifetimes and should remain default-off until the
   waits are re-audited.
3. Keep pre-transformer sync cleanup as a smaller follow-up. It cannot beat the
   conv decoder by itself but can shave the ~20-25 ms cached path.

No temporary code probe was run in this pass. This was a source/log
reconnaissance pass only.

## Follow-up probe

Added a default-off allocation timing probe for the Mimi conv stage:

```powershell
$env:AILA_TTS_MIMI_ALLOC_PROFILE = "1"
```

When enabled, `mimi_conv_stages` logs allocation/free counts, bytes, and
`sycl::malloc_device` / `sycl::free` wall time for the full conv-stage call.
The probe does not change tensor math or buffer lifetimes; it only enables
`Context` allocation accounting while the conv stage is active. Use it with
`AILA_TTS_PROFILE=1` so allocation cost can be compared against ConvNeXt,
decoder block, and total conv-stage timing.

Short real smoke result:

```text
ALIA_REAL_MODEL_SMOKE_PASS
first backend frames: 15
first backend codes: 316.599 ms
first backend audio: 442.729 ms
first audio: 933 ms

conv frames  alloc/free  bytes alloc/free  alloc ms  free ms  conv total
8            33 / 33     30.89 / 30.89 MB  5.228     0.249    99.3
15           33 / 33     57.92 / 57.92 MB  5.309     0.280    184.2
8            33 / 33     30.89 / 30.89 MB  2.212     0.322    96.9
16           33 / 33     61.78 / 61.78 MB  2.862     0.418    185.5
24           33 / 33     92.67 / 92.67 MB  5.580     0.419    400.2
32           33 / 33     123.55 / 123.55 MB 5.648    0.386    366.4
```

Interpretation: scratch reuse is still a valid cleanup, but allocation/free is
only a few milliseconds per Mimi conv call on this profile. It is not the
largest TTS RTF lever. Prioritize decoder kernel work, pre-transformer small
kernel fusion, or pipeline scheduling before a large scratch-buffer rewrite.

Added a default-off pre-transformer residual fusion probe:

```powershell
$env:AILA_TTS_MIMI_PTFM_FUSED_RESIDUAL = "1"
```

When enabled for `decode_mimi_incremental`, each pre-transformer layer replaces
the two local patterns:

```text
copy x -> residual
scale update
residual += update
copy residual -> x
```

with one kernel that writes `x = x + update * scale` directly. This removes two
temporary `[new_frames, 512]` tensors and several small kernels per layer. It is
kept default-off because it changes bf16 write ordering and should be validated
with real smoke/matrix plus output ASR before promotion.

Short smoke A/B:

```text
mode                  pre-transformer cached samples       first_backend_audio_ms  first_audio_ms
default               18.2-22.4 ms                         456.087                 969
fused residual probe  14.4-18.7 ms, one 19.7 ms tail call  442.223                 1150
```

Both runs passed `ALIA_REAL_MODEL_SMOKE_PASS`. The opt-in run generated a
longer first foreground/TTS chunk, so end-to-end first audio is not a fair A/B
for this probe. The backend first-audio value stayed in the same range while
the pre-transformer cached slice improved by roughly 3-5 ms per Mimi callback.

Output ASR checks:

```text
default:
Krash, I am Alia, a local companion. I can help with simple tasks.

fused residual probe:
Kurasho, Aliya is a local companion. I can't be a voice assistant, but I can try to answer as normal text.
```

The transcripts match their foreground text closely enough for this default-off
probe. The gain is real but small relative to the conv decoder; keep it as an
opt-in probe until matrix coverage confirms no audio regressions.

Added a default-off decoder residual block conv2 fusion probe:

```powershell
$env:AILA_TTS_MIMI_DECODER_FUSED_CONV2_RESIDUAL = "1"
```

This targets only Mimi decoder residual-block `conv2`, where the convolution is
kernel size 1 and the weight layout remains `[out_ch, in_ch, 1]`. The fused
kernel writes `residual += bf16(conv2(input))` directly, preserving the bf16
rounding point of the previous `conv2_out` intermediate while removing one
temporary tensor and one residual-add kernel per residual block. It deliberately
does not fuse `act1 + conv1`, because that would recompute SnakeBeta for every
output channel and is likely to regress.

Short opt-in smoke result:

```text
ALIA_REAL_MODEL_SMOKE_PASS
first backend frames: 15
first backend codes: 316.717 ms
first backend audio: 437.031 ms
first audio: 933 ms

conv frames  decoder blocks  conv total
8            89.7 ms         93.3 ms
15           166.2 ms        169.9 ms
8            90.3 ms         93.6 ms
16           172.0 ms        175.2 ms
24           260.2 ms        390.7 ms
32           345.4 ms        349.4 ms
32           349.5 ms        353.8 ms
29           314.3 ms        319.0 ms
8            87.4 ms         91.3 ms
16           174.9 ms        177.9 ms
24           264.2 ms        268.1 ms
25           276.2 ms        280.6 ms
```

External ASR check:

```text
Krash, I am Alia, a local companion. I can help with simple tasks.
```

Interpretation: the opt-in path keeps the generated content aligned with the
foreground text and trims a small amount from decoder-block time by removing
one pointwise conv output tensor plus one residual-add launch per decoder
residual block. The gain is modest and noisy, so keep this default-off until a
matrix run shows stable backend gains and no transcript regressions.

Added a default-off decoder block breakdown probe:

```powershell
$env:AILA_TTS_MIMI_DECODER_BLOCK_PROFILE = "1"
```

This inserts synchronization only while the probe is enabled and reports the
decoder substage split: initial `dec0`, per-stage SnakeBeta, transposed
convolution, residual copy, residual `act1`, `conv1`, `act2`, `conv2`, and
residual add. Use it to choose the next decoder optimization target; do not use
its end-to-end latency as an A/B result because the probe intentionally adds
extra waits.

Short default-path profile with the probe enabled:

```text
frames  decoder  transpose  conv1  conv2
8       105.3 ms 63.3 ms    17.7   12.1
15      185.5 ms 118.5 ms   33.6   19.7
16      194.0 ms 125.0 ms   35.3   21.2
24      286.6 ms 186.9 ms   53.2   30.8
32      376.6 ms 247.8 ms   70.6   40.4
29      343.4 ms 226.0 ms   63.8   36.2
```

Interpretation: transposed convolution is the dominant decoder block cost,
around 60-65% of decoder time. `conv1` is the next meaningful target, while
SnakeBeta, copies, and residual adds are small. The first transpose-conv cleanup
keeps the same `[in_ch, out_ch, kernel_size]` weight layout but iterates only
valid stride-compatible taps (`k = ot % stride; k += stride`) instead of scanning
the whole kernel and checking `rem % stride`. For Qwen3-TTS decoder stages this
reduces the per-output tap loop from `2 * stride` checks to at most two useful
taps.

Optimized default-path smoke result:

```text
ALIA_REAL_MODEL_SMOKE_PASS
first backend frames: 15
first backend codes: 323.036 ms
first backend audio: 410.409 ms
first audio: 900 ms
backend total: 5380.69 ms

frames  decoder  conv+readback  mimi incremental
8       57.5 ms  60.7 ms        84.2 ms
15      105.2 ms 108.5 ms       132.4 ms
16      109.5 ms 112.2 ms       134.2 ms
24      160.6 ms 162.8 ms       184.4 ms
32      213-215 ms 216-217 ms   238-239 ms
29      195.3 ms 198.1 ms       221.0 ms
25      168.4 ms 171.6 ms       191.9 ms
```

With block-profile enabled after the optimization, the transpose substage drops
from about `63/118/125/247 ms` for `8/15/16/32` frame windows to about
`27/47/51/96-99 ms`. External Mimo-ASR on both optimized smoke outputs reads:

```text
Krash, I am Alia, a local companion. I can help with simple tasks.
```

## Validation command template

Use the real model path and the worktree-local reference audio. API-only tests
are not enough for this area.

```powershell
. .\perf\PerfCommon.ps1
Initialize-AilaOneApiEnvironment
cmake --build build --target AliaEngine --config Release

$env:AILA_TTS_PROFILE = "1"
$env:AILA_TTS_MIMI_ALLOC_PROFILE = "1"
$env:AILA_TTS_MIMI_PTFM_FUSED_RESIDUAL = "1"
$env:AILA_TTS_MIMI_DECODER_FUSED_CONV2_RESIDUAL = "1"
$env:AILA_TTS_MIMI_DECODER_BLOCK_PROFILE = "1"
.\tools\alia\RunAliaTargetPipeline.ps1 `
  -SkipBuild -SkipToolProbe `
  -AudioPath 'tmp\alia-real-smoke\voice_matrix_stream_500ms_fg_suffix\short_hello_request.wav' `
  -OutputWav 'tmp\alia-real-smoke\<probe-name>.wav' `
  -LogPath 'tmp\alia-real-smoke\<probe-name>.log' `
  -RequestText 'Alia, please say hello in one short sentence.' `
  -MaxTokens 48 `
  -StreamAsrPrefill `
  -StreamChunkMs 500 `
  -StreamPrefillIntervalMs 500 `
  -TimeoutSec 1500

$env:MIMO_API_KEY = [Environment]::GetEnvironmentVariable("MIMO_API_KEY", "Machine")
powershell -ExecutionPolicy Bypass -File "E:\RiderProjects\Mimo-ASR\mimo-asr.ps1" `
  -AudioFile "tmp\alia-real-smoke\<probe-name>.wav"
```
