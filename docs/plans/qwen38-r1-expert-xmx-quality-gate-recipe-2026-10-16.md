# Qwen3.8-Flash-Next / R1: smallest credible quality gate for the expert-XMX prefill change

Date: 2026-10-16. Base: HEAD `bb07130`. Scope: prefill expert kernels moving from
bit-exact ALU NF4 (`moe_expert_major_prefill_*`) to dequantize-once-to-bf16 + oneDNN
XMX GEMM (`moe_grouped_xmx_*`), gated by `AILA_Q4E_PREFILL_EXPERT_XMX=1` (planned, not
yet in `src/`). Recipe only: no build, no GPU run, no src edit. Thresholds marked
`[chosen]` are new here; everything else is quoted from existing tooling/docs.

## 0. What the repo already has (search result, not assumption)
| Tool | What it gives | Where |
|---|---|---|
| Decode probe `--decode-model` | fixed teacher-forced forward, 128 consumed tokens; raw BF16 logits + per-step FNV hashes + top-1 | `tests/models/Qwen4ExpDecodeProbe.hpp`; entry `Qwen4ExpToleranceTests.cpp:711` |
| Logit comparator + KL | mean/max KL (nats), max NRMSE, top-1 row matches, bitwise rows | `perf/qwen4exp/q4e_decode_summary.py:63-95` |
| Existing tolerance contract | `nrmse <= 0.03`, `kl <= 0.01`, top1 diagnostic only | `perf/qwen4exp/q4e_mtp_reference.py:169,185` |
| Top-1 agreement gate | teacher-forced argmax agreement at width W, `--gate 0.95`; precedent 61/63 = 0.9683 PASS | `tmp/q4e-sprint-r1-20261014/run-top1-gate.ps1`, `.../mtp-top1/compare_top1.py`, `docs/plans/qwen38-r1-mtp-top1-2026-10-15.md` |
| Server runner + SSE content sha | HTTP/SSE, greedy, per-response `content_sha256`, `responses/*.sse` | `tmp/q4e-sprint-r1-20261014/run.py` -> `tools/server/qwen38_context_probe.py`; hash in `tests/server/socket_tests.py` |
| Kernel A/B vs oracle | grouped XMX gate/up+down vs double-precision NF4 reference | `tests/models/Qwen4ExpGroupedXmxAcceptance.hpp`, `AilaQwen4ExpGroupedXmxTests` (`CMakeLists.txt:854`) |
| Existing XMX switch + hooks | `AILA_Q4E_GROUPED_XMX_MIN_ROWS` (>0, Arc A770), `grouped_xmx_supported_`, grouped tiles/acts log | `src/models/Qwen4ExpBnb4Backend.cpp:414,421,1660,1837,3015` |

Missing (section 5): server-side logits/KL, prefill-logit dump, host per-token logits,
and `AILA_Q4E_PREFILL_EXPERT_XMX` itself.

## 1. Gate command: flag-off (ALU) vs flag-on (XMX), fixed prompts
Runner: **the in-process decode probe, not `run.py`**. The server SSE path sees only
tokens and hashes; the probe runs the real engine (prefill then decode) and its
`prompt_count=512` arm traverses the 512-token expert prefill, the code under test.
`run.py` is kept for the 2b confirmation. Sampling is off by construction: the probe
calls the argmax/verify path (`sampling=0` in its own `[DECODE PASS]` line), so
"temperature 0 / fixed seed" is not a variable. CPU-side except the two `& $exe` lines.

```powershell
Get-Process AilaWorker,AilaServer,AilaQwen4ExpToleranceTests -ErrorAction SilentlyContinue  # must print nothing
$exe='build-refactor\AilaQwen4ExpToleranceTests.exe'; $model='C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2'
$out='tmp/q4e-expert-xmx-gate-20261016'; New-Item -ItemType Directory -Force $out | Out-Null
# Production env (docs/plans/qwen38-server-mtp-2026-10-07.json) + fixed prompt/rows/repeats.
$env=@{
  AILA_Q4E_DECODE_PROBE_PROMPT='512'; AILA_Q4E_DECODE_PROBE_ROWS='1'; AILA_Q4E_DECODE_PROBE_REPEATS='2'
  AILA_Q4E_SERVER_MTP='0'; AILA_Q4E_COLD_MODE='2'; AILA_Q4E_CPU_MAX_EXPERTS='10'; AILA_Q4E_CPU_THREADS='18'
  AILA_Q4E_CPU_ASYNC_ROUTE='1'; AILA_Q4E_CPU_LIGHT_ROUTE='1'; AILA_Q4E_CPU_NF4_Q8='1'
  AILA_Q4E_CPU_Q8_ACTIVATION64='1'; AILA_Q4E_CPU_A64_SPECIALIZE='1'; AILA_Q4E_CPU_A64_ROW2='1'
  AILA_Q4E_MOE_FRONT_FUSED='1'; AILA_Q4E_GDN_HISTORY_FUSED='1'; AILA_Q4E_GDN_CHUNK_MODE='0'
}
foreach ($arm in @(@{tag='alu-off';flag='0'}, @{tag='xmx-on';flag='1'})) {
  Get-ChildItem Env:AILA_*,Env:SYCL_*,Env:UR_* -ErrorAction SilentlyContinue | Remove-Item
  $env.GetEnumerator() | ForEach-Object { Set-Item "Env:$($_.Key)" $_.Value }
  Set-Item Env:AILA_Q4E_PREFILL_EXPERT_XMX $arm.flag
  Set-Item Env:AILA_Q4E_DECODE_PROBE_LOGITS "$out/$($arm.tag).logits.bf16"
  "=== $($arm.tag) start $(Get-Date -Format o)"
  & $exe --decode-model $model *> "$out/$($arm.tag).log"
  "=== $($arm.tag) exit=$LASTEXITCODE $(Get-Date -Format o)"
  Start-Sleep -Seconds 15   # large-page pool drain between arms
}
python perf/qwen4exp/q4e_decode_summary.py "$out/alu-off" "$out/xmx-on" --output "$out/gate.json"
```

Repeat the block with `AILA_Q4E_DECODE_PROBE_PROMPT='30'` and a `-p30` suffix. The two
prompt lengths are **not** cross-comparable (`compare()` keys runs on `prompt_fnv64` and
requires equal artifact shape). Positive control: the `[DECODE PASS]` cache counters
(`hits`/`misses`/`demand_bytes`) must be nonzero, proving the expert path ran. Outputs:
`$out/*.log`, `$out/*.logits.bf16`, `$out/gate.json`; via
`perf/qwen4exp/q4e_execution_probe.ps1 -Workload decode` also `<prefix>.meta.json`.

## 2. Metrics and thresholds
### 2a. Primary: teacher-forced KL + top-1 (exists; no proxy needed)
Read `gate.json.numerical_comparisons[0]` (`rows=128`, full-vocab logits):
| Metric | Field | Threshold | Justification |
|---|---|---|---|
| mean KL | `mean_kl` | `<= 0.01` nats | the repo's own pre-existing contract (`q4e_mtp_reference.py:169,185`); precision swap, not a new algorithm |
| max KL | `max_kl` | `<= 0.05` nats `[chosen]` | 5x headroom; catches one catastrophic position the mean hides |
| max NRMSE | `max_nrmse` | `<= 0.03` | same contract |
| top-1 agreement | `top1_matches / 128` | `>= 122` (95.3%) | released bar `>= 0.95` (`qwen38-r1-mtp-top1-2026-10-15.md` sec 1.3); first integer clearing it |
| exact greedy parity | `exact_greedy_parity` | report, do not gate | *expected false*; `true` suggests the flag did not apply |

Report per prompt length and pooled. Reject the run if pass>0 replay changes logits bits
(the probe asserts it): that invalidates the measurement, not the model.

### 2b. End-to-end confirmation: greedy tokens + SSE content sha (exists)
Only after 2a passes. 6000-token context only; 14K is a ~30 min arm and adds no expert
coverage for this change.

```powershell
python tmp/q4e-sprint-r1-20261014/run.py gate-alu-6k --prompts 6000 AILA_Q4E_PREFILL_EXPERT_XMX=0
python tmp/q4e-sprint-r1-20261014/run.py gate-xmx-6k --prompts 6000 AILA_Q4E_PREFILL_EXPERT_XMX=1
python -c "import json,pathlib
a=json.loads(pathlib.Path('tmp/q4e-sprint-r1-20261014/gate-alu-6k/result.json').read_text(encoding='utf-8'))
b=json.loads(pathlib.Path('tmp/q4e-sprint-r1-20261014/gate-xmx-6k/result.json').read_text(encoding='utf-8'))
k=lambda r:(r['stream']['content_sha256'],r['stream']['usage']['completion_tokens'])
print([k(x) for x in a['requests']]==[k(x) for x in b['requests']])"
```

Gate: `True`. The probe is greedy (`tests/server/socket_tests.py:684`,
`temperature=0.0, seed=42`), so this is exact equality - a consistency check, not a
statistical measure.

### 2c. Statistical credibility without being expensive
The probe hard-caps `steps=128` (`Qwen4ExpDecodeProbe.hpp:5`) and the comparator gates on
`rows/128`, so one run = 128 independent teacher-forced positions; pooled over both
prompt lengths = 256. At N=128 the 95% Wilson half-width at p=0.95 is +/-3.9% `[computed]`,
and the pass line tolerates only 6 mismatches, so the sample is already discriminating.
This matches the accepted precedent (the MTP gate compared 63-64 positions per prompt).
If a tighter interval is ever required (e.g. +/-2%), it needs ~600 positions = 5 prompt
variants = 5 extra model loads; do that only if 2a lands inside the band.

## 3. Runtime cost per gate
| Gate | Work | Cost |
|---|---|---:|
| 2a, one arm | load + 512-token prefill + 128 x 1-row forwards | 6-7 min `[est]` |
| 2a, full primary gate | 2 arms x 2 prompt lengths (4 model loads) | ~25 min |
| 2a with `--teacher-tokenizer` | + NLL verification (`q4e_decode_summary.py:149`) | +seconds |
| 2b, server confirmation | 2 arms x 6000-token prefill, 32 tokens | ~20-26 min |
| Kernel pre-check (section 4) | `--grouped-kernel` | <1 min |

Assumes an otherwise idle A770; the probe owns the GPU for the durations above, with the
15 s pool drain between arms. The CPU comparator itself is seconds.

## 4. Smallest pre-build sanity check (CPU-only)
Replay the split rule offline, in the style of
`tmp/q4e-sprint-r1-20261014/actslm/coverage_check.py`: from the dispatcher's tile rule
(`src/models/Qwen4ExpBnb4Backend.cpp:1731-1810`), assert on the real routing histogram
that (i) every activation of every nonzero-usage expert is assigned to exactly one arm
(ALU or XMX) - no gaps, no double-consume; (ii) with the flag absent the ALU tile set is
bit-for-bit the old one; (iii) no tile straddles the 8-row boundary. Costs nothing, needs
no GPU. Then, as the first GPU check (still not a quality gate):
`build-refactor\AilaQwen4ExpToleranceTests.exe --grouped-kernel`, which compares the XMX
kernels to an independent double-precision NF4 oracle in-process - run it before the
25-minute gate so a kernel defect is not mistaken for a quality regression.

## 5. What the repo does NOT have, and the fallback chosen
- **No KL/logit tool at the HTTP layer.** No endpoint or artifact exposes per-token
  logits; `socket_tests.py`/`service_benchmark.py` see tokens, hashes, timings only.
  Fallback: in-process decode probe for 2a; server layer only for exact-equality 2b.
- **No prefill-logit dump.** Logits come only after a 128-step decode continuation, so
  expert quality is seen through its effect on decode. Fallback: the 512-token prompt
  arm plus the `[DECODE PASS]` cache counters as the positive control.
- **No whole-corpus teacher scoring.** `q4e_decode_summary.py` scores one fixed 128-token
  continuation; `tmp/q4e-agent-corpus-v1` is server prompt material only. Fallback:
  2 prompt lengths x 128 teacher-forced tokens (2c).
- **No `AILA_Q4E_PREFILL_EXPERT_XMX` yet.** The closest existing switch is
  `AILA_Q4E_GROUPED_XMX_MIN_ROWS` (>0 plus Arc A770, `...Backend.cpp:414`); it exercises
  the same `moe_grouped_xmx_*` kernels but is the resident-prefill path from
  `gdn_chunk_mode_` (`:1117`), not the flag under test - not a substitute verdict.
- **No bit-exactness requirement in this recipe.** The change is explicitly non-bit-exact:
  differing `step_hashes` and `exact_greedy_parity=false` are expected, not failures.
