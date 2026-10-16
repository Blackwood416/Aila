# R1 decode attention pair2: length gate + int8-KV extension (G=4 refuted)

Date: 2026-10-16. Repo: `E:\RiderProjects\Aila`, base HEAD `92d0bce`.
Scope: `src/ops/AttentionOps.cpp` only (plus this document and scratch under
`tmp/q4e-sprint-r1-20261014/p10/`). Discipline: no GPU run, no `build.ps1`, no
commit, no other source file touched. One `icx-cl -fsyntax-only` gate was run
(the actSLM/hostread pattern, `tmp/q4e-sprint-r1-20261014/p10/syntax-check.cmd`)
and returned `ATTENTIONOPS_SYNTAX_EXIT=0` with no diagnostics.

Target geometry: Qwen3.8-Flash-Next full attention, Hq=24 / Hkv=2 / D=256,
single-row decode. Production decode runs this shape at 0.5-0.6 ms/layer.

---

## 0. Conclusions

1. **G=4 (pair4) is refuted by the owner's bench and is not implemented.** Off
   566.4 / 842.5 us per row (8192 / 14336), pair2 443.7 / 704.2, pair4 565.0 /
   829.4 (rows=1, median, `--decode-attn-reuse-bench --json`). The E5
   extrapolation "G=2 -16..-25% -> G=4 -30..-38%" is withdrawn. Ledger caveat:
   the production selector `attn_kv_reuse_pair2()` matches only the exact string
   `pair2`, so a `pair4` process ran the default path; that row is a second
   off-control (its -0.2%/-1.6% spread is A/A noise). Either way the wave/latency
   argument stands: G=4 would issue `(num_heads/4)*num_tiles` work-groups with
   four serialized (reduce, exp, reduce, V-loop) sections each - at 4096 tokens
   that is 192 work-groups against 512 resident, i.e. the exact under-filled
   regime that already cost pair2 +8.0% at 1834.
2. **pair2 is now admissible: the length gate removes the K5 veto regime.**
   `AILA_ATTN_KVREUSE=pair2` applies only when `cached_len` and `effective_len`
   are both >= `AILA_ATTN_KVREUSE_MIN_LEN` (default 2048), so the 1834-token
   shape that failed `reject_1834_regression_le2pct` stays on the default path,
   while every shape where pair2 was measured to win (4096/8192/14336) keeps it.
3. **pair2 is extended to the int8 KV cache, because the best production decode
   bundle already uses int8 KV** (`AILA_KV_QUANT=1 AILA_KV_QUANT_FORMAT=int8`
   + `AILA_SERVER_KV_TYPE=int8`; e9-int8kv receipt: ForwardWall 57.55 vs 59.64
   = -3.5%). Before this change pair2 was inert on that bundle (`fmt == int8`
   failed the format clause), so an A/B/B/A with `AILA_ATTN_KVREUSE=pair2` on
   the production bundle would have measured 0% and produced a false NO-GO.
4. **The int8 extension is bit-exact by construction** (section 4): both new
   read branches reuse the G=1 int8 kernel's own expressions verbatim and only
   share the dequantized element between the two heads; the softmax, partial
   layout and merge kernels are untouched. The gpu-free evidence is the
   expression/order argument plus the pre-existing G=1 vec-vs-accessor
   equivalence note; the first GPU evidence is the `--kv-fp8-only` log diff
   (section 5.2), which re-prints identical `[KV ACC] arm=int8` numbers only if
   the bytes are identical.
5. **Honest expectation: +1.5..2.2% e2e [est], below the +3% adoption gate.**
   The production attention block is 6.06 ms/token of a 55.1 ms token (P7
   breakdown); a 15-20% int8 kernel cut is -0.9..-1.2 ms/token. The arm is
   still worth one slot because (a) it is the only way to price pair2 on the
   int8 path, (b) it re-verifies bit-exactness at 14336, and (c) it stacks with
   `AILA_Q4E_LM_HEAD_Q8` (measured +4.62% on the same bundle) toward 20 tok/s.

---

## 1. Production path and where the gate lives

| Item | Location |
|---|---|
| Variant switch (read once) | `attn_kv_reuse_pair2()` `src/ops/AttentionOps.cpp:744-747`, env `AILA_ATTN_KVREUSE`, value `pair2` |
| Length floor (read once) | `attn_kv_reuse_min_len()` `:757-760`, env `AILA_ATTN_KVREUSE_MIN_LEN`, default 2048 |
| Gate decision | `attention_decode` call site `:2213-2218`: `kv_reuse_len_ok = cached_len >= min && effective_len >= min` |
| Gate transport | new trailing parameter `bool kv_reuse_len_ok = false` on `attention_decode_exact_head256_partial_merge` `:1154` |
| Format/heads gate + dispatch | `:1165-1205` (`int8_dpas_path`, `fmt_ok`, one-time active/skip logs) |
| Variant kernel | `attention_decode_exact_head256_pair2_impl<Quantized>` `:790-1137` |
| Default kernel (unchanged) | `attention_decode_exact_head256_partial_merge_impl` `:432-...` |

Why the gate is at that call site: `attention_decode` is the frame that owns
`cached_len`, and it has just derived `effective_len` from the routing/window
state (`:2035-2072`). `effective_len <= cached_len` holds on every path, so
requiring `effective_len >= 2048` also implies `cached_len >= 2048`; the dense
shape has `effective_len == cached_len`, and the QSA-routed shape has
`effective_len = num_active_blocks*4 + cached_len % 4`. Testing both keeps the
intent literal (a dense 1834 cannot enter) and additionally keeps a *routed*
decode whose attended subset is short on the default path.

The gate changes only which phase-1 kernel is submitted. No arithmetic, no
buffer layout, no barrier count of either kernel is touched; with
`AILA_ATTN_KVREUSE` unset the extra parameter is false and the code path is the
pre-existing default one.

---

## 2. int8 interaction: why the extension was required, and its size

### 2.1 How the G=1 int8 kernel reads K/V and scales

`kv_i8_vec = kv_int8_vector_reads()` (`:306`, `AILA_KV_QUANT_VEC`, default true)
selects between two read forms; both are documented as bit-identical (`:464-467`).

| Path | K read | V read |
|---|---|---|
| vec (default) | `:535-560`: `sycl::vec<int8_t,8>` load, `k_row_scale = k_scale[kv_head*max_seq_len + cache_t]` hoisted per token, term `q_cache[d+j] * (float(codes[j]) * k_row_scale)` | `:604-630`: token-major, one `v_row_scale` load per token, `acc_lo/acc_hi += p * (float(v_row[lid{,+tile_wg}]) * v_row_scale)` |
| accessor | `:562-...`: `k_acc.load(...)` = `int8_kv_to_float(code, scale)` = `float(code) * scale` (`src/ops/Ops.hpp:76`) | `:637-659`: `acc += logits[i] * v_acc.load(...)` |

So the dequant is a single scalar multiply in every form; there is no shared
lambda, but the *expressions* are identical across forms.

### 2.2 What pair2 needed

The pair2 body was already format-generic: its K fallback branch used
`k_acc.load` for both heads and its V loop used `v_acc.load` for both heads.
Enabling int8 therefore needed (a) the dispatcher (format clause + instantiation
selection + a DPAS-shadowing guard) and (b) two read branches that mirror the
G=1 *vec* forms so the pair2 int8 path is not slower than the default int8 path
per element:

| Added block | Where | Size |
|---|---|---|
| K int8 vec branch (scale hoisted, 8 codes per load, shared by both heads) | `:897-948` | ~50 lines |
| V int8 token-major branch (`acc_{a,b}_{lo,hi}`, one scale load per token) | `:1023-1066` | ~44 lines |
| Dispatcher: `int8_dpas_path` hoisted, `fmt_ok`, int8 instantiation, log fields | `:1165-1175`, `:1191-1196`, `:1223` | ~15 lines |
| Comments | `:760-787`, `:1160-1172` | ~12 lines |

Total diff in the one allowed file: **+173 / -34 lines**. The user-visible
behaviour of the int8 path with `AILA_ATTN_KVREUSE` unset is unchanged.

### 2.3 Why int8 should gain at least as much as bf16 in relative terms

pair2 removes one of the two K and V read+dequant sequences per (kv_head, tile)
in phase 1: the second head consumes the value the first head already loaded and
dequantized. FMA count, reductions, SLM traffic and partials are unchanged.

* bf16 measured (K5 receipt, rows=1): -16.0% / -25.1% / -18.2% at 4096 / 8192 /
  14336, i.e. the duplicated K/V stream is a real cost.
* int8 is *not* purely bandwidth-bound: halving the bytes gave only -33%
  (14336, 2-row calls: 1.5833 ms bf16 -> 1.0532 ms int8, `qwen38-int8kv-vec-2026-10-12.md`).
  The residual is per-element work - loads, index math, the dequant multiply -
  and pair2 removes half of that work as well, not just half of the bytes
  (one `float(code)*scale` per element instead of two, and the vec form's
  8-code load replaces 16 single-code loads per pair).
* Therefore the relative win on int8 should be >= the bf16 relative win on the
  same shape. Absolute ms are smaller (the baseline is already faster), which is
  exactly why the e2e estimate in section 0.5 is only +1.5..2.2%.

---

## 3. The denominator the arm must beat

| Config | ForwardWall | rate |
|---|---:|---:|
| bf16 KV, no pair2 (pre-int8) | 57.55 ms | 17.38 tok/s |
| **int8 KV, no pair2 (best bundle)** | **55.1 ms** | **17.41-17.49 tok/s** |
| int8 KV + LM head Q8 (opt-in) | ~52.7 ms | 18.26 tok/s |
| e10 bundle arms (A/B/B/A) | 54.77 / 55.46 | 17.6 tok/s peak |

pair2 *alone* cannot beat int8 KV (both halve a stream; int8 halves all bytes,
pair2 only the duplicate reads), which is why the extension in section 2 exists:
the two effects are multiplicative on the *load and dequant* count, not
alternatives.

---

## 4. Bit-exactness argument (both formats, and int8 specifically)

1. Per-head arithmetic is the G=1 kernel's, written out verbatim: K is
   `d`-ascending (`j` 0..7 inside each 8-wide step), V is `i`-ascending with one
   accumulator pair per d-slice, then `score *= scale`. Sharing a load cannot
   change a value: the load is a pure function of immutable cache bytes.
2. For int8 the shared quantity is `float(code) * row_scale` - the same
   expression, in the same order, as the G=1 vec form (`float(codes[j]) *
   k_row_scale`, `float(v_row[d]) * v_row_scale`) and as the accessor form
   (`int8_kv_to_float` = `float(code) * scale`). The G=1 kernel documents its
   own vec form as bit-identical to its accessor form (`:464-467`), so pair2 is
   bit-identical to whichever form the default path uses.
3. The reductions stay `reduce_over_group` over the same 128 work items with the
   same lane-to-value map (two per head, four per work-group), the SLM
   exp/max round trip is copied verbatim, and the `[head][tile][stride]`
   partials layout and phase 2 merge kernel are untouched - so the merged bf16
   output is byte-identical.
4. Scope conditions: rows>=2 uses a different phase-1 kernel and is untouched
   (control, expect ~0%); fp8 stays on the default path; and the int8 DPAS
   partials kernel is *excluded from the pair2 branch* because DPAS changes the
   accumulation grouping (`int8_dpas_path`, `:1165`, mirrors the default
   classifier including its `AILA_KV_QUANT_DPAS` gate, default off).
5. The length gate cannot affect exactness either: it only selects between two
   kernels whose outputs are already equal, so gated and ungated runs must hash
   identically at every length >= 2048 - which is itself a cheap consistency
   check (K5's `AILA_ATTN_KVREUSE_REF` ledger).

---

## 5. Arms, commands, gates (owner runs these; none were run here)

### 5.1 bf16 kernel bench, gate evidence (0 build, ~3 min)

```powershell
# reuse-ref.txt already holds the rows=1/2 hashes of the K5 default-path run, so
# both arms run in "compare" mode and print bitwise_eq per (length, rows).
$env:AILA_ATTN_KVREUSE_REF = "tmp/q4e-sprint-r1-20261014/decode-attn/reuse-ref.txt"
build-refactor\AilaQwen4ExpKernelTests.exe --decode-attn-reuse-bench --json            # arm off
$env:AILA_ATTN_KVREUSE = "pair2"
build-refactor\AilaQwen4ExpKernelTests.exe --decode-attn-reuse-bench --json            # arm pair2
Remove-Item Env:AILA_ATTN_KVREUSE, Env:AILA_ATTN_KVREUSE_REF
```

Expected in the pair2 arm:

* `[ATTN] kvreuse=pair2 skipped len_gate head_dim=256 fmt=0 effective_len=1834 min_len=2048`
  exactly once (333 never reaches this wrapper: below `kDecodeExact256MinLen=1024`),
* then `[ATTN] kvreuse=pair2 head_dim=256 fmt=0 heads=24 kv_heads=2
  effective_len=4096 min_len=2048 groups=384` exactly once
  (`(num_heads/2) * num_tiles` = 12 * 32),
* `median_us_per_row` at 1834 within A/A (+-1.5%), at 4096/8192/14336
  approximately the K5 receipt (-16.0 / -25.1 / -18.2%),
* `bitwise_eq=1` at 4096/8192/14336 rows=1 against the off arm's ledger.

Note: the bench META still prints `pair2_len_floor:1024`
(`tests/ops/Qwen4ExpKernelTests.cpp:5404`) and its length list is hard-coded to
`{333,1834,4096,8192,14336}` (`:5357`). Both are now stale; the file is outside
this task's write scope, so the one-line fix is left to its owner - adding 2052
next to 1834 would price the exact production attended length (`shape_count`
5 -> 6, plus the META list string):

```cpp
const int lens[shape_count] = {333, 1834, 2052, 4096, 8192, 14336};
```

### 5.2 int8 kernel arm: bit-exactness + timing (0 build, ~1-2 min)

`--kv-fp8-only` already calls `attention_decode` once per row at
`len-1, len` for arms bf16/fp8/int8 at (cap,len) = (512,333), (2048,1834),
(16384,14336) and prints the accumulated output vs a bf16-true reference
(`[KV ACC]`) plus interleaved medians (`[KV WALL]`).

```powershell
build-refactor\AilaQwen4ExpKernelTests.exe --kv-fp8-only | Tee-Object tmp/q4e-sprint-r1-20261014/p10/kvfp8-off.log
$env:AILA_ATTN_KVREUSE = "pair2"
build-refactor\AilaQwen4ExpKernelTests.exe --kv-fp8-only | Tee-Object tmp/q4e-sprint-r1-20261014/p10/kvfp8-pair2.log
Remove-Item Env:AILA_ATTN_KVREUSE
```

Gates:

* **Bit-exactness (int8):** every `[KV ACC] len=14336 arm=int8...` line must be
  textually identical in the two logs (max_abs, nrmse, cos, bitwise_eq), and the
  test's own `diff(int8, int8-expanded) <= 1e-3` must pass in the pair2 arm.
  `len=1834 arm=int8` must also be identical (it is gated off).
* **Timing:** `[KV WALL] len=14336 ... arm=int8 median_ms` must drop >= 15%
  (bf16 arm is not gated at 333/1834 but is gated at 14336 too - its drop is the
  independent bf16 confirmation). No regression at 333 (baseline kernel) or
  1834 (default path).

### 5.3 e2e A/B/B/A on the production bundle (the adoption gate)

```powershell
$H = "C:\...\agent6k"   # absolute AILA_Q4E_HOTSET_PROFILE path used by the e10 arms
python tmp/q4e-sprint-r1-20261014/run.py pa2-a1 --prime --diag --generate=32 `
    AILA_Q4E_HOTSET_PROFILE=$H AILA_KV_QUANT=1 AILA_KV_QUANT_FORMAT=int8 AILA_SERVER_KV_TYPE=int8
# B arms: same env plus AILA_ATTN_KVREUSE=pair2   -> pa2-b1, pa2-b2
# A2: identical to a1                            -> pa2-a2
```

Gates (pre-registered):

1. **Adopt into the bundle** iff the B-arm median `rate` (14K continue) is
   >= +3.0% over the int8-only baseline (17.413 / 17.49 tok/s), with SSE content
   sha identical in all four arms and A/A spread <= 1.5%.
2. 0..+3%: keep opt-in, stack with `AILA_Q4E_LM_HEAD_Q8` (+4.62% measured) and
   re-price; negative: NO-GO and record the int8 kill.
3. Evidence line required in both B arms:
   `[ATTN] kvreuse=pair2 head_dim=256 fmt=2 heads=24 kv_heads=2 effective_len=<N> min_len=2048 groups=<G>`
   once per process, with `N` the production attended length. `fmt=2` is the
   int8 cache - `fmt=0` means the arm silently ran bf16 KV, and
   `skipped len_gate ... effective_len=<2047 or less>` means the routed subset
   fell below the floor (then and only then consider a bench at that exact
   length before lowering `AILA_ATTN_KVREUSE_MIN_LEN`).
4. Controls: 333 and 1834 in 5.1/5.2 must stay within A/A; rows=2 e2e is out of
   scope (different kernel).

Rollback: unset `AILA_ATTN_KVREUSE` (default) or set
`AILA_ATTN_KVREUSE_MIN_LEN=0` to reproduce the K5 blanket behaviour for
comparison.

### 5.4 e2e verdict (RUN 2026-10-16): below the gate - stays opt-in

A/B/B/A on the int8 bundle, `context-14000-continue`, rate =
`(n-1)/(done-first_content)` (n = 32), receipts
`tmp/q4e-sprint-r1-20261014/pa2-{a1,b1,b2,a2}/result.json` and `pa2-*-worker.log`:

| arm | rate tok/s |
|---|---:|
| a1 | 17.859 |
| b1 (pair2) | 17.746 |
| b2 (pair2) | 17.503 |
| a2 | 17.976 |

* A median 17.918 / B median 17.625 -> **B - A = -1.6%**: below the +3.0%
  adoption gate. A/A spread +0.65% (gate <= 1.5% met). This lands in the
  gate's "negative -> NO-GO" bin.
* Disposition: `AILA_ATTN_KVREUSE=pair2` was **not adopted** into the bundle and
  stays default-off; the int8 read branches + length gate remain in the tree as
  measurement infrastructure. Section 5.2's kernel-level numbers stand
  (14336: 1.0482 -> 0.9022 ms, -13.9%); this arm only kills the e2e claim on
  the current bundle, not the kernel path.
* Bit-exactness held e2e: the SSE content sha is identical in all four arms
  (`a9aa5132...`), as section 4 argued.
* The arm priced the gate's own boundary: both B arms logged
  `[ATTN] kvreuse=pair2 head_dim=256 fmt=2 heads=24 kv_heads=2
  effective_len=2049 min_len=2048 groups=204` - the production attended subset
  at 14K is **2049**, i.e. the flag engaged at the most fragile point of the
  2048 floor (risk 2), not at the 14336 bench shape. The 2048..4095 kernel
  regime is still unmeasured (the 2052 bench line in 5.1 remains the cheapest
  way to price it), so this result reads "pair2 at 2049 buys nothing e2e", not
  "pair2 is slow at 14336".
* Wider context for the disposition (2026-10-16 bottleneck audit,
  `tmp/q4e-bottleneck-audit-20261016/REPORT.md`): decode wall time on this
  machine is set by the per-layer submit/synchronise round trips, not by
  attention arithmetic, so a ~0.1-0.2 ms kernel delta at 2049 cannot clear a
  +3% e2e gate. The next decode lever is the control plane, not further
  attention micro-kernels.

---

## 6. Risks and open items

1. **Boundary:** the routed attended length is `num_active_blocks*4 +
   cached_len%4`; if production lands exactly at 2048 the gate is satisfied by
   one token and any routing change could disable the variant silently. The
   active/skip log lines are the mitigation.
2. **Unmeasured regime 2048..4095:** bf16 pair2 was +8.0% at 1834 and -16.0% at
   4096; the gate at 2048 bets that the crossover is below it. The 2052 bench
   (5.1 note) and the e2e arm are the only evidence that can close this; if the
   e2e and the int8 kernel arm disagree, keep opt-in.
3. **int8 bit-exactness is argued, not measured** until 5.2 runs. If the log
   diff mismatches, the cheap fallback is to delete the two new int8 read
   branches and instantiate `<kKvFmtInt8>` through the pre-existing accessor
   path (bit-exactness then follows from `int8_kv_to_float` directly, at the
   cost of the per-element scale loads).
4. **Register pressure:** the int8 V branch holds four accumulators plus a
   hoisted scale instead of two accumulators. Measured bf16 pair2 stayed
   bit-exact and faster, but int8 register allocation is unmeasured; the median
   in 5.2 is the check.
5. **DPAS:** `AILA_KV_QUANT_DPAS=1` on a dense int8 shape keeps the DPAS kernel
   and skips pair2 by design; that combination is unmeasured and not claimed.
6. **fsyntax-only is not a link gate** (see the actSLM LNK2019 precedent); the
   first `build.ps1` remains the link check.

---

## 7. Deliverables

* `src/ops/AttentionOps.cpp`: length gate (`attn_kv_reuse_min_len`, call-site
  decision), int8 pair2 support (K/V vec branches + dispatcher + logs),
  +173/-34 lines, syntax-only clean, not committed.
* This document.
* `tmp/q4e-sprint-r1-20261014/p10/syntax-check.cmd` (the `-fsyntax-only` gate).
