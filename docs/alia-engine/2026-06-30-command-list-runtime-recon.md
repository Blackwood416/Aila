# 2026-06-30 command-list runtime recon

Scope: exploratory runtime microbench for the Alia custom engine. This does not
change the speech pipeline, model path, or default runtime behavior. The goal is
to decide whether SYCL immediate command lists, direct Level Zero command lists,
or a manual graph-style replay path are worth exploring for fixed-shape decode
blocks.

## Setup

Worktree: `E:\RiderProjects\Aila\.worktrees\alia-custom-engine`

Base HEAD before this round:

```text
5c13c76 perf: enable playback-aware Alia TTS backend batches
```

Level Zero SDK:

```text
C:\level-zero-win-sdk-1.28.2
```

The optional benchmark target is `AilaCommandSubmissionBench`. It is only added
when both files exist:

```text
C:\level-zero-win-sdk-1.28.2\include\level_zero\ze_api.h
C:\level-zero-win-sdk-1.28.2\lib\ze_loader.lib
```

Repro runner:

```powershell
.\tools\perf\RunCommandSubmissionBench.ps1 -Warmup 200 -Iters 2000 -Bytes 4,4096,65536 -OpsPerGraph 16
```

The runner initializes the oneAPI environment, builds the optional benchmark,
then runs each payload size under:

```text
UR_L0_USE_IMMEDIATE_COMMANDLISTS=<unset|0|1|2>
```

Latest local output:

```text
tmp\command-submission-bench\20260630-183530\summary.csv
tmp\command-submission-bench\latest-summary.csv
tmp\command-submission-bench\20260630-183530\
```

Device in this run:

```text
Intel(R) Arc(TM) A770 Graphics
```

## Benchmark coverage

The benchmark measures host submit time and total submit-to-wait time for:

- SYCL `queue.memset`, one operation per wait.
- SYCL `queue.single_task`, one operation per wait.
- SYCL graph-like batch: append N `queue.memset` operations, wait once.
- Level Zero regular command list, re-recorded each iteration.
- Level Zero regular command list, recorded once and replayed.
- Level Zero regular command list, recorded once with N operations and replayed.
- Level Zero immediate command list, one operation per synchronize.
- Level Zero immediate command list, N appends per synchronize.

This is a microbench, not a real model validation. It is useful for runtime
direction only.

## Representative results

Single operation, host average / total average in microseconds:

```text
bytes  env    sycl_memset  sycl_single_task  l0_record_each  l0_replay  l0_immediate
4      unset  29.307/80.090 30.675/78.627   19.832/71.144   9.426/67.764 14.752/85.415
4      2      20.564/68.076 22.809/70.037   17.138/65.169   7.516/61.535 17.729/85.740
4096   unset  26.884/82.585 30.481/77.320   17.995/68.193   9.508/64.056 14.840/91.659
4096   2      22.090/72.633 23.179/69.976   17.039/68.893   9.002/69.875 17.361/82.842
65536  unset  26.124/81.626 28.980/76.167   14.458/63.238   7.329/60.877 12.277/89.303
65536  1      18.765/67.976 23.576/68.817   15.477/63.775   7.638/65.189 13.874/84.514
```

Sixteen operation graph-style batch, host average / total average in
microseconds:

```text
bytes  env    sycl_memset_graph  l0_replay_graph  l0_immediate_graph
4      unset  144.822/268.683    7.785/71.092     151.377/757.992
4      0      149.654/268.937    7.593/67.563     154.057/770.449
4      1      345.482/769.067    7.179/69.386     137.298/756.196
4      2      259.896/734.581    7.488/68.456     136.696/738.939
4096   unset  125.087/265.502    8.313/73.515     143.945/737.389
4096   0      138.060/271.137    7.656/70.180     142.167/746.728
4096   1      255.377/760.031    8.599/76.418     142.927/742.460
4096   2      254.305/751.632    17.623/78.179    150.965/765.587
65536  unset  121.767/257.358    7.471/68.177     133.154/732.648
65536  0      129.078/268.077    10.636/71.896    131.595/725.180
65536  1      234.653/778.732    7.550/70.815     138.335/740.318
65536  2      260.471/769.781    10.566/74.666    117.776/734.995
```

## Current interpretation

- `UR_L0_USE_IMMEDIATE_COMMANDLISTS=1` or `2` can reduce SYCL single-submit
  latency, especially for `queue.memset` at 4096-65536 bytes in this run.
- The same env var is not a blanket win. For a 16-operation SYCL submit batch,
  `1` and `2` were substantially worse than unset/0 on this stack.
- Direct Level Zero immediate command lists are not attractive for the tested
  repeated memory-fill shape. They add fast enough on the host for one op, but
  total time is worse, and N-op batches are very poor.
- Direct Level Zero regular command list replay is the clear signal. A fixed
  16-op command list replayed with stable pointers costs roughly 7-12 us host
  time and about 69-76 us total time across tested payload sizes.

## Implication for Alia

The useful direction is not "turn on immediate command lists everywhere". The
useful direction is a manual graph-style path for fixed-shape decode blocks:

1. Pre-record a Level Zero regular command list for a stable kernel sequence.
2. Keep pointers and tensor shapes stable across decode steps.
3. Put small dynamic values into shared/device runtime parameter buffers.
4. Use a tiny immediate or normal fallback path only when shape/pointer state
   changes.

This is most promising for custom SYCL kernels that already have stable buffers.
It is less immediately applicable to oneDNN-style primitives or code paths where
the underlying runtime owns opaque command submission.

## Next experiments

- Identify a fixed-shape TTS/Mimi or VLM decode sub-block made mostly of custom
  SYCL kernels and stable device buffers.
- Prototype regular-list replay around that sub-block before attempting a full
  model decode graph.
- Measure with the real speech pipeline after any runtime integration. This
  microbench only justifies the direction; it is not enough to claim TTFA wins.
- Explore FP8 KV cache separately by reading the current KV cache and attention
  paths first. It was not tested in this command-list pass.

## Follow-up: SYCL graph and copy_tensor probe

Added two more benchmark modes on top of the original command-submission bench:

- `queue_kernel_chain`: submit a 16-kernel `parallel_for` chain and wait once.
  The kernels read a small shared runtime parameter to approximate decode-step
  scalar parameter updates.
- `command_graph_kernel_chain`: record the same kernel chain with
  `sycl::ext::oneapi::experimental::command_graph` and replay it, if the device
  reports `sycl::aspect::ext_oneapi_graph`.

Latest focused command:

```powershell
.\tools\perf\RunCommandSubmissionBench.ps1 -SkipBuild -Warmup 100 -Iters 1000 -Bytes 4096 -OpsPerGraph 16
```

Output:

```text
tmp\command-submission-bench\20260630-190222\summary.csv
```

Key 4096-byte / 16-op results:

```text
env    queue_memset_graph  queue_kernel_chain  l0_replay_graph  l0_immediate_graph
unset  132.931/262.504     331.729/440.974     8.756/71.802    163.726/772.302
0      129.416/258.411     319.188/422.197     7.409/66.617    162.005/765.559
1      283.281/757.849     493.682/936.579     8.799/76.674    136.103/742.452
2      285.984/769.477     502.175/949.190     7.531/67.485    173.888/752.502
```

For every env mode in this run, `command_graph_kernel_chain` skipped:

```text
device does not report ext_oneapi_graph support
```

Interpretation:

- The installed oneAPI headers expose command graph APIs, but the current Arc
  A770 runtime/device path does not advertise `ext_oneapi_graph`; existing SYCL
  lambda kernels cannot currently be wrapped in SYCL graph replay on this stack.
- `UR_L0_USE_IMMEDIATE_COMMANDLISTS=1/2` is harmful for multi-kernel chain
  submission here. Keep `unset` or `0` for this workload.
- The direct Level Zero regular-list replay signal remains strong, but applying
  it to actual compute kernels would require either L0 module kernels or a
  runtime path that exposes existing kernels to L0. It is immediately practical
  only for memory copy/fill style nodes.

### copy_tensor memcpy probe

`ops::copy_tensor` is a pure contiguous tensor copy currently implemented as a
SYCL kernel. A guarded D2D `queue.memcpy` fast path was added behind:

```text
AILA_COPY_TENSOR_MEMCPY=1
```

It stays default-off because the full matrix did not show a clear win.

Short A/B smoke, no TTS profile:

```text
run                         vad_to_audio  first_audio  backend_audio  codes
default_noprofile_short     1119 ms       686 ms       247.234 ms     187.643 ms
memcpy_noprofile_short      1030 ms       670 ms       243.668 ms     186.069 ms
```

Generated audio was checked with the online Mimo ASR script:

```powershell
powershell -ExecutionPolicy Bypass -File "E:\RiderProjects\Mimo-ASR\mimo-asr.ps1" -AudioFile "E:\RiderProjects\Aila\.worktrees\alia-custom-engine\tmp\alia-real-smoke\copy_tensor_default_on\short.wav"
```

Recognized text:

```text
Kura叔，父亲大人教过我，要诚实回答，但我是AI，不是真的能说话，只当虚拟陪玩。
```

Default-on full matrix probe:

```text
tmp\alia-real-smoke\voice_matrix_copy_memcpy_default\summary.csv
```

It passed 5/5 scenarios with non-empty output ASR, but performance was not
better than the closest prior baseline:

```text
run                                 avg_vad_to_audio  avg_first_audio  avg_backend_audio  avg_codes
voice_matrix_backend8_split_default 1133.6 ms         700.4 ms         234.6 ms           177.4 ms
voice_matrix_copy_memcpy_default    1191.6 ms         758.6 ms         237.5 ms           181.0 ms
```

Conclusion: keep `AILA_COPY_TENSOR_MEMCPY` as a default-off probe and do not
promote D2D memcpy for `copy_tensor` yet.
