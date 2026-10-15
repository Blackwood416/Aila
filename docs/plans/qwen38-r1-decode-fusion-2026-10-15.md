# E8: decode host-path amortization (submission census + fusion variants)

Date: 2026-10-15. Repo: `E:\RiderProjects\Aila`, pair `retained28`.
Owner: subagent E8. Scope: census + designs + one standalone, not-wired
variant. Zero GPU, zero build (one `icx-cl -fsyntax-only` check of the new TU,
no artifacts), zero commit. ASCII only.

Target: decode 20+ tok/s (wall 58.79 ms/token measured, E2
`qwen38-r1-decode-20-2026-10-15.md` section 1). Everything below attacks the
host execution share (~34 ms) by removing *kernel submissions*, not bytes.

## 1. Per-token submission census (production decode, count = 1)

Submit-cost measurements used throughout:

- isolated submit rate: **9.4-14.0 us/kernel**
  (`tmp/q4e-sprint-r1-20261014/decode-misc/misc-A1.log`:
  `[MISC] submit_cost kernels=512 host_us_per_kernel=13.97`; E5 doc section 3).
- in-production rate: **21-32 us/submit**, derived from the instrumented
  spans themselves: `hc_host` 6.10 ms clean for 291 HC submissions
  (approx 21 us), `shared_submit` 3.5-4.3 ms for 144 submissions
  (approx 26-30 us), `hc_token_prod` back-to-back on a quiet queue is only
  6.4 us/kernel (`misc-A1.log`), so the production number is what binds.

| # | Family | Submits/token | Per layer/site | Source (file:line) | Host budget | Mergeable? |
|---|---|---:|---|---|---:|---|
| 1 | HC norm / inject-add | 97 | attn_hc + mlp_hc + mixer | `src/ops/HyperConnectionOps.cpp:5` (`grouped_rms_norm` submit :11), `:224` (`hc_inject_add_and_grouped_rms_norm` submit :242), `:301` (`try_hc_decode_...` submit :316, production decode path) | hc_host 6.10 ms total | **yes -> V1** |
| 2 | HC down+inject | 97 | 1 per call | `HyperConnectionOps.cpp:584` (impl :367) | same | no (see 2.2) |
| 3 | HC up+mix | 97 | 1 per call | `HyperConnectionOps.cpp:590` (impl :447) | same | no (see 2.2) |
| 4 | Shared expert | 192 | gate 1, gate_up 1, silu 1, down 1 x 48 | `Qwen4ExpBnb4Backend.cpp:2421-2431` (`dense_rows`, `ForwardDecodeRows`, `moe_silu_gate_up_mul_flat`, `ForwardDecodeRows`) | shared_submit 3.73 + shared_add 0.63 | **yes -> V2** (silu fold) |
| 5 | MoE front | 48-144 | router gate 1 (+shared gate if not fused), topk 1-2 | `:2285` (fused alternative) / `:2295` (`dense_rows`), topk `:2311/:2318` | moe_front 1.78 + topk_submit 0.91 + router_other 1.93 | **already fused, config-off -> V3** |
| 6 | Routed experts | approx 192 | resident-input D2H, gu, down, ordered-accumulate | `Qwen4ExpBnb4Backend.cpp:1193` (`RunResidentDecode`) + `ExpertExecutor` | experts_host 11.57 (non-join) | separate owner (R-line) |
| 7 | Attention | approx 300 `[estimate]` | 36 GDN layers + 12 full layers | `RunAttentionWindow` call site `:2202` | linear_host 3.39 + full_host 2.87 | not audited here |
| 8 | Head + sample + misc | approx 40 `[estimate]` | mixer HC 3, LM head 1, sampling | head_host 0.20, sample 3.42 (E2) | not audited here |

Exact counts: rows 1-5 (from the call sites above). Rows 7-8 are estimates.
Total is roughly 1000 +/- 150 submissions/token, which is why production
submit cost (21-32 us) is 3-5x the isolated rate: the queue is being refilled
constantly and every submission pays event/flush overhead.

Savings arithmetic (`saved submits x production rate`):

- V1 (rows 1+2 -> one kernel): **-97 submits = -2.0 to -3.1 ms/token**,
  i.e. 3.5-5.3% of the 58.79 ms wall `[estimate]`.
- V2 (row 4, fold silu into gate_up): **-48 submits = -1.0 to -1.5 ms**.
- V3 (row 5, enable the existing fused MoE front): **-48 submits = -1.0 to
  -1.5 ms** at zero code cost (`AILA_Q4E_MOE_FRONT_FUSED=1`, default 0 and
  explicitly 0 in `docs/plans/qwen38-server-mtp-2026-10-07.json:68`;
  `s_front_native_gemv` = `AILA_Q4E_DENSE_GEMV` defaults 1, so the gate is
  purely the profile flag; the `AILA_Q4E_GPU_NF4_Q8` variant saves 2 more
  launches/layer but changes input quantisation).

## 2. Variant designs

### 2.1 V1 (chosen, implemented): norm+inject-add fused into down

Production cost per HC call is 3 kernels: norm, down, up. Fusing the norm into
the down kernel removes 1 submission per call (97/token) with no new
submission elsewhere.

Contract (all bit-exact by construction; every expression, reduction order and
BF16 rounding point copied from the production ops):

```
H'[i] = bf16(H[i] + inj[g] * y[d])            (identity when y/inj null)
N[i]  = bf16(H'[i] * (inv_rms[g] * (1 + w[i])))
act[r] = bf16( (bf16(sum_i w_down[r][i] * N[i])) * inv_groups * sigmoid(...) )
inj_out[g] = 2 * sigmoid(...)
```

The fused kernel (one submit, 21 work-groups) materialises H' into SLM,
recomputes the grouped mean with the *identical* leaf pattern
(`lid, lid+256, ...`) and stride-128..1 tree, then runs the down GEMV with N
recomputed in-register from SLM in the same expression order as a global N
load. Only work-group 0 writes the caller's N.

The in-place `H += inj * y` update moves to the up kernel
(`run_hc_up_mix_with_hidden_update`): each element (g, d) is read and written
by exactly one sub-group, after the fused kernel has consumed the original H
snapshot. The fused kernel itself never writes H, which is what keeps it
race-free without a device-scope barrier.

Engineering: new TU `src/models/Qwen4ExpDecodeFusion.{hpp,cpp}` (not wired,
default zero calls), CMake registration only. Estimate S-M.

Bit-exactness risks: compiler contraction of `a + b * c` in the relocated
update must match the production kernel (same expression, same compile
options); `sycl::native::exp` and `sycl::fma` calls are copied verbatim.
Verification: dump `act`/`inj`/`N` from both paths and compare bitwise
(section 3.2).

### 2.2 Rejected: direct down+up merge (the obvious -97)

The up stage consumes all `lowrank` act values produced by the down stage
(320 values, produced by 21 work-groups that each own distinct rows), so a
single kernel needs a device-scope barrier. Two ways exist, both rejected for
this window:

- spin-flag/decoupled-lookback inside one kernel: 21-160 co-resident
  work-groups with unknown residency on DG2; a deadlock is unverifiable
  without GPU time and would hang the whole worker. Not acceptable as a
  default-off experiment either, because a deadlock is not a tolerance.
- redundant act recomputation per work-group (160 WGs x 3.3M FMA): approx
  105 us/call of extra device work versus a 21-32 us host saving. Net loss.

The act dependency is only 320 BF16 values, i.e. this is a *synchronisation*
problem, not a bandwidth problem. If a later round is willing to spend GPU
time on it, the honest experiment is a two-phase single kernel with a
residency assertion (bench first, no e2e arm).

### 2.3 V2 (design only): fold SwiGLU into the shared gate_up GEMV

`shared_gate_up_proj.ForwardDecodeRows` writes gu[1280]; the next submission
reads it and writes inter = silu(gu[:640]) * gu[640:]. A variant GEMV that
makes each sub-group compute the row pair (h, h+640) and stores
`bf16(silu_g * u_f)` directly removes 1 submission/layer (-48/token) and
halves the gu write.

Bit-exact iff the NF4 unpack/absmax/sum order of
`packed_nf4_gemv_bf16` and the silu expression
(`g / (1 + native::exp(-g))`, then `bf16(silu_g * u)`,
`src/ops/MoeExpertOps.cpp:5127`) are copied verbatim. Engineering M (must
mirror the GEMV internals). Gate: kernel-level >=3% on two shapes,
e2e >=5%.

### 2.4 V3 (config only): fused MoE front

`bf16_moe_front_fused` already exists (`Qwen4ExpBnb4Backend.cpp:2285`) and
fuses router gate + shared gate into 1 submission/layer (2 with the Q8 input
path). Production disables it, and no `[Q4E MoE Front Fused]` line exists in
any archived worker log, so it has never been A/B-tested on the current
profile. This is the cheapest probe in this document: flip the flag, compare
text bytes (native bf16 GEMV vs the oneDNN `DenseLinear` path may differ in
the last bits), then read wall.

## 3. Wiring draft (not applied)

### 3.1 Diff draft for V1

In `Qwen4ExpBnb4Backend::RunHyperConnectionCombined` replace the count == 1
branch (`Qwen4ExpBnb4Backend.cpp:1784-1803`, currently
`hc_decode_down_inject` + `hc_decode_up_mix`) with:

```cpp
static const bool s_hc_fusion = aila::env::read_flag("AILA_Q4E_HC_FUSION", false);
if (count == 1 && s_hc_fusion && s_fused_hc && hc.mix_down_w && hc.mix_up_w &&
    (!with_inject || hc.inject_w)) {
    bf16* hidden4 = buf_.hidden4.data_as<bf16>();
    const bf16* upd_y = nullptr;
    const float* upd_inj = nullptr;
    if (pending_mlp && pending_mlp->pending) {
        upd_y = pending_mlp->y;
        upd_inj = pending_mlp->inject;
    } else if (pending_y && pending_inj) {
        upd_y = pending_y;
        upd_inj = pending_inj;
    }
    // Returns false (no submission) for unsupported shapes; the previous
    // dispatch (pending_mlp->ConsumeNorm / grouped_rms_norm) must then run
    // unchanged, so keep it as the fallback arm of this if.
    if (q4e_fusion::run_hc_norm_down_fused(
            ctx, hidden4, upd_y, upd_inj,
            hc.hc_norm_weight->data_as<float>(), cfg_.rms_norm_eps,
            hc.mix_down_w->data_as<bf16>(),
            with_inject ? hc.inject_w->data_as<bf16>() : nullptr,
            buf_.hc_N.data_as<bf16>(), buf_.hc_mix_down.data_as<bf16>(),
            with_inject ? buf_.hc_inject_f32.data_as<float>() : nullptr,
            lowrank, hidden_size_, 4)) {
        if (pending_mlp && pending_mlp->pending) pending_mlp->MarkConsumed();
        q4e_fusion::run_hc_up_mix_with_hidden_update(
            ctx, buf_.hc_mix_down.data_as<bf16>(), hc.mix_up_w->data_as<bf16>(),
            buf_.hc_N.data_as<bf16>(), buf_.x.data_as<bf16>(),
            upd_y ? hidden4 : nullptr, upd_y, upd_inj, lowrank, hidden_size_, 4);
        return;
    }
}
```

Notes for the header owner:

- `MarkConsumed()` is a small addition to `DecodePendingMlpResidual` (clear
  `pending` without re-submitting the norm, mirroring `ConsumeNorm`'s
  post-submit clear); it must be called *after* the fused op accepted the
  shape, and before the up submission only if a throw there must not drop the
  residual - safest is to clear it right after the fused op returns true,
  matching the existing "clear immediately after submission" comment.
- The mixer call site `:2594` uses the same code with
  `with_inject == false` (no inject rows) and still needs the update
  relocation; the plain-norm path (`pending_mlp->pending == false`) passes
  `upd_y == nullptr` and needs no update.
- The fused op expects `hidden4` to still be the *original* residual stack;
  nothing between the previous layer's consumers and this call writes it.

### 3.2 Validation arms (GPU mutex first, no fixed sleeps)

1. Numerics (must be 0 differences): run one warm decode with
   `AILA_Q4E_HC_FUSION=0` and one with `=1` under
   `AILA_DEBUG_Q4E_DUMP_DIR`, then compare the dumped `hc_N`, `x` and
   `hc_inject_f32` buffers bitwise (same driver, same session shape).
2. A/B/B/A on `run.py` (`d1-smoke` harness): `--prime --diag`, read
   `submitted_ms` median (baseline 58.79 ms), `hc_host_ms` (expect 6.10 ->
   2.0-3.0 ms), plus exact-text equality between arms.
3. Gate: median `submitted_ms` improvement >= 5% with 2/2 pairs and A/A
   spread <= 1%; if `hc_host_ms` drops but wall does not move, the queue was
   already masked by `router_wait` and this variant should be closed (same
   lesson as the 8192 regression).
4. V3 probe (independent): `AILA_Q4E_MOE_FRONT_FUSED=1` single-arm screen,
   text equality first, then wall.

### 3.3 Status of the implemented code

- `src/models/Qwen4ExpDecodeFusion.hpp/.cpp`: implemented, compiles clean
  (`icx-cl -fsyntax-only -fsycl`, exit 0, command in
  `tmp/q4e-sprint-r1-20261014/decode-fusion/syntax-check.cmd`), not called
  anywhere, `AILA_Q4E_HC_FUSION` env gate not yet plumbed.
- CMake: one line added to `LIB_SOURCES`.

Superseded by section 3.4 (wired 2026-10-15 under the env name
`AILA_Q4E_DECODE_FUSED_HC`).

### 3.4 F1 wiring applied (2026-10-15, appended by subagent F1)

Status: wired, opt-in, `AILA_Q4E_DECODE_FUSED_HC=1` (default 0 = current
behaviour byte-identical). Zero GPU, zero build artifacts, zero commit; both
touched TUs pass `icx-cl -fsyntax-only -fsycl` exit 0
(`tmp/q4e-sprint-r1-20261014/decode-fusion/syntax-check-f1.cmd` covers the
backend TU; the fusion TU check is the pre-existing `syntax-check.cmd`).
First real link happens in the coordinator build.

Changed lines (post-patch numbers):

- `src/models/Qwen4ExpBnb4Backend.hpp:398-402`: `MarkConsumed()` declaration
  on `DecodePendingMlpResidual`.
- `src/models/Qwen4ExpBnb4Backend.cpp:6`: include of the fusion header.
- `:1741-1748`: `MarkConsumed()` implementation (clears `pending`, increments
  `consumedNorm` - the same accounting ConsumeNorm uses).
- `:1753-1755`: `hc_dim` / `lowrank` / `s_fused_hc` hoisted to the top of
  `RunHyperConnectionCombined` (declaration moves only, no semantic change).
- `:1757-1798`: gated fused block; `:1791` is the `MarkConsumed()` call, made
  immediately after `run_hc_norm_down_fused` returns true (same
  clear-after-submit discipline as `ConsumeNorm`; the up op cannot throw for
  shapes the fused op accepted, so no residual can be silently dropped by a
  host throw on a healthy queue).
- Call coverage: all three count==1 call sites funnel through
  `RunHyperConnectionCombined` - attn_hc pending `:2248`, mlp_hc attn-residual
  `:2278`, mixer `:2647` - so every one of the 97 HC calls per decode token
  takes the fused arm when the flag is on. count>1 (prefill, MTP verify rows
  path, layer-major) is excluded by the `count == 1` gate. The MTP draft
  count==1 calls (`Qwen4ExpMtpDraft.cpp:186/213/241`) also take it and are
  bit-exact by the same construction.

Fallback conditions (all lead to the unchanged production dispatch, with no
partial submissions - the fused op only submits at its very end):

1. `AILA_Q4E_DECODE_FUSED_HC` unset/0, or `count != 1`, or
   `AILA_Q4E_HC_FUSED=0`, or null `mix_down_w`/`mix_up_w`/`inject_w`.
2. Shape guard reject inside `run_hc_norm_down_fused` (returns false): null
   H / norm_w / w_down / n_out / act_out; groups outside 1..8; group_dim <= 0
   or not a multiple of 8; lowrank < 1; exactly one of y/inj null; w_inject
   set with null inj_out; k_dim not a multiple of 8; SLM footprint over
   48 KiB; H (and y/n_out in the update case) not 16-byte aligned.
3. While the flag is on, `AILA_Q4E_HC_DECODE_DOWN_WG` and
   `AILA_Q4E_HC_DECODE_UP_STREAMS` are ignored (the fused launches are fixed
   at the wg=256 geometry). The profile pins 256/0 anyway, and both knobs
   only change scheduling, not values, so this is not a numerics risk.
4. Rollback is a config change only: unset the env (or `=0`); no rebuild.
   If a B-arm log shows a device error (`UR_OUT_OF_RESOURCES` etc.) that the
   A arms do not, close the variant and keep the default off.

Coordinator arms (GPU mutex precheck first, no fixed sleeps):

```powershell
Get-Process AilaWorker,AilaServer -ErrorAction SilentlyContinue   # must be empty
pwsh -NoProfile -File build.ps1 -BuildDir build-refactor
python tmp/q4e-sprint-r1-20261014/run.py f1-a1 --prime --diag
python tmp/q4e-sprint-r1-20261014/run.py f1-b1 --prime --diag AILA_Q4E_DECODE_FUSED_HC=1
python tmp/q4e-sprint-r1-20261014/run.py f1-b2 --prime --diag AILA_Q4E_DECODE_FUSED_HC=1
python tmp/q4e-sprint-r1-20261014/run.py f1-a2 --prime --diag
```

Read-out (per-token `[Q4E AdaptiveWork] ... count=1` lines in
`f1-<arm>-worker.log`; the flag reaches the worker because run.py puts
`KEY=VALUE` overrides into the child environment):

- Gate: `submitted_ms` median -3% or better vs the A median (-2.0 to -3.1 ms
  expected `[estimate]`), 2/2 pairs (a1/b1 and b2/a2 medians), A/A spread
  <= 1%.
- Gate: `hc_host_ms` median -30% or more (6.10 -> <= 4.3 ms).
- Text byte equality: completion text in `f1-a*/result.json` vs
  `f1-b*/result.json` must be identical byte-for-byte (same deterministic
  decode protocol as the previous arms).
- Stronger bitwise arm (optional): rerun one A and one B with
  `AILA_DEBUG_Q4E_DUMP_DIR` set and compare `layer_*.bin` + `mixer.bin` +
  `logits.bin` bitwise. Note the existing interplay: the dump path
  materialises the pending MLP residual (`pending_mlp.Materialize()`), so the
  pending sites see `pending == false`; the fused arm still engages through
  the plain-norm and the mlp_hc update-relocation code paths, so the dumps
  remain a valid bit-exactness check for the fused kernels.
- Masking lesson (same as the 8192 regression): if `hc_host_ms` drops 30%+
  but `submitted_ms` moves less than 3%, the queue was hidden behind
  `router_wait` and the variant should be closed with the default left off.

## 4. Risks and TODOs

1. The fused kernel duplicates the update+mean phase in every work-group
   (21x). Latency-wise this is approx +0.7 us/call of ALU `[estimate]`; the
   phase-B N recomputation doubles the scalar multiply count but not the FMA
   count. First bench must compare `span_ms` of the fused kernel against the
   production norm+down pair.
2. SLM: 20 KiB (H') + 4 KiB (reduction) per work-group; if the compiler
   reports a resource error, lower the guard or split `groups` across
   work-groups.
3. `norm_w` is read `groups * group_dim` times per row (L2-served); if the
   fused kernel is bandwidth-bound in phase B, hoist `(1 + w)` into SLM once
   per work-group instead (bit-exact: same expression, same operand).
4. Relocation of the in-place update to the up kernel assumes nothing writes
   `H` between the two submissions; that holds in the decode loop today
   (verified: the only writer between them is the previously fused norm).
   Any future kernel inserted there must be re-audited.
5. The pending-residual consumption path needs a `MarkConsumed` helper; do
   not clear `pending` before the up submission succeeds, otherwise a throw
   can drop the residual.
