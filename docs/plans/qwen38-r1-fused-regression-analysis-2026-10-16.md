# R1 fused-HC + shared-fusion regression: root-cause study (2026-10-16)

Read-only source + receipt investigation of the NO-GO arm in
`docs/plans/qwen38-r1-fused-hc-shared-screen-2026-10-16.md`
(`AILA_Q4E_DECODE_FUSED_HC=1` + `AILA_Q4E_MOE_SHARED_FUSED=1` on the
production decode bundle: 14K continue rate 17.50 -> 14.03 tok/s, SSE content
sha changed, `cached_tokens` absent on all four B requests, +100 s wall).
No build, no GPU run, no src edit, no commit.

Anchor note: all `src/models/Qwen4ExpBnb4Backend.cpp` line numbers are against
the working tree as read for this study, which carried an uncommitted 3-line
insertion in `ensure_runtime_buffers` (an `AILA_Q4E_PREFILL_EXPERT_XMX_C1_F32`
env read at `:1225`) from a separate expert-XMX workstream. Everything after
`:1225` in that file is +2 lines relative to clean HEAD; re-verify before
quoting.

## 0. Answer in five lines

1. The B arms did not merely "change numbers": the generated text is
   **degenerate** (`Based\n\n\n\n\n\n The The\n\n...`), and the divergence starts
   at the **second generated token** - the first `count == 1` decode forward.
2. The pair is not needed to reproduce either symptom. `f1-b1`
   (`tmp/q4e-sprint-r1-20261014/f1-b1/result.json`) ran
   `AILA_Q4E_DECODE_FUSED_HC=1` **alone**, on a *different* bundle
   (h4 profile, `AILA_KV_QUANT` unset), and produced the **byte-identical**
   degenerate continuation of the 14K arm (`f1-b1` vs `fh-b1` 14K text:
   identical). `AILA_Q4E_MOE_SHARED_FUSED` is exonerated for the observed
   signature; `AILA_Q4E_DECODE_FUSED_HC` is the prime suspect.
3. The prefix-cache "side effect" is a *consequence* of #1, not an independent
   cache defect: reuse requires an **exact token-id prefix match**
   (`src/server/HttpServer.cpp:262-298`) between the cached
   prompt+generated ids and the *re-rendered* continuation prompt, and the
   degenerate reply re-renders to different/short ids. Mechanism narrowed to
   two branches; one `AILA_SERVER_PREFIX_DIAG=1` rerun settles it (section 3).
4. The -19.8% itself has two live mechanisms and they are not yet separated:
   (a) a device-side cost of the fused HC launches (24 KiB SLM/WG, per-WG
   duplicated norm) and (b) a *text-driven* effect - different tokens route to
   experts the hotset profile does not cover, so the CPU cold-expert path
   (`experts_host_ms`/`cpu_join_ms`) grows. Section 2 gives the cheap
   discriminator.
5. Recommendation: drop the *pair*, re-screen `MOE_SHARED_FUSED` alone only if
   a submit-count win is still wanted, and do **not** re-screen
   `DECODE_FUSED_HC` before the numerics defect below is fixed and a
   bit-exactness test exists.

## 1. What each flag actually changes at runtime

### 1.1 `AILA_Q4E_DECODE_FUSED_HC` (default 0)

Contract (whole file, `src/models/Qwen4ExpDecodeFusion.hpp:1-86`): replace the
production 3-submit HC call `(inject-add+grouped RMSNorm) -> hc_decode_down_inject
-> hc_decode_up_mix` with 2 submits - `(norm+down)` then `(up-mix + relocated
in-place residual update)` - for `count == 1` only. The header states the
functions are "bit-exact by construction" (`:18-22`), that `H` is read-only in
the first kernel and written by the second (`:44-56`), and documents the shape
guard: `groups == 4`, `group_dim % 8 == 0`,
`k_dim * 2 + groups * 256 * 4 <= 48 KiB` SLM, `lowrank >= 1`
(`:58-62`, enforced at `src/models/Qwen4ExpDecodeFusion.cpp:39-55`).

What the kernels do:
* `run_hc_norm_down_fused` (`Qwen4ExpDecodeFusion.cpp:24-198`): per work-group
  phase A1 materializes `H' = H + inj*y` into **SLM** (`:79-97`), A2 recomputes
  the grouped RMSNorm in SLM (`:102-124`), A3 writes `N` from work-group 0 only
  (`:129-138`), phase B runs the down GEMV `lowrank + groups` rows
  (`:147-193`) reading `nh_slm` instead of global `N` and re-deriving the
  per-element `scale = inv[g]*(1+w)` (`:166-168`, `:177-179`). The `2*sigmoid`
  inject rows / `bf16(v*s)` act rows / `n_aligned && row_is_aligned` gate mirror
  `ops::hc_decode_down_inject_impl` (`src/ops/HyperConnectionOps.cpp:367-433`).
* `run_hc_up_mix_with_hidden_update` (`:200-282`): the up-mix body is the
  production expression (`HyperConnectionOps.cpp:448-509`, incl. the
  `bf16(dot)` and `bf16(sigmoid)` rounding points), *plus* the relocated
  in-place update `H[row] = bf16(H[row] + inj[g]*y[d])` at `:270-273`. Production
  performs that same update inside the norm kernel
  (`HyperConnectionOps.cpp:327-334`) and the norm then reads the updated `H`
  (`:341`, `:352`).

Wiring and gating (`src/models/Qwen4ExpBnb4Backend.cpp`):
* flag read once, cached: `:2127`; entry gate `:2128-2129` requires
  `count == 1`, `AILA_Q4E_HC_FUSED` (default true, `:2118`), `mix_down_w`,
  `mix_up_w`, and `inject_w` when injecting.
* pending-residual handoff: `:2130-2143` chooses `H/y/inj` from
  `DecodePendingMlpResidual` when it is armed, else from the caller's
  `pending_y/pending_inj`; `MarkConsumed()` (`:2104-2111`, declared
  `Qwen4ExpBnb4Backend.hpp:400-404`) is called after the first submit succeeds
  (`:2154`).
* any guard rejection falls through to the unchanged production dispatch at
  `:2163-2210` (no partial submissions, no log line on fallback).
* `count != 1` (prefill, MTP verify rows, layer-major) never takes this path,
  so **prefill is untouched** by this flag.
* production bundle interaction: `AILA_Q4E_DECODE_PENDING_MLP_RESIDUAL=1` and
  `AILA_Q4E_MOE_FRONT_FUSED=1` are both set in
  `docs/plans/qwen38-server-mtp-2026-10-07.json` (`worker_environment`), i.e.
  every measured arm ran the deferred-residual + front-fused combination.

### 1.2 `AILA_Q4E_MOE_SHARED_FUSED` (default 0)

Read at two sites: `src/models/Qwen4ExpBnb4Backend.cpp:2510` (chunk-major
forward) and `src/models/Qwen4ExpLayerMajor.cpp:245` (layer-major prefill).

Runtime effect: the split kernel `add_shared_expert_to_y` is *skipped* -
`Qwen4ExpBnb4Backend.cpp:3026-3029` - and the shared-expert contribution is
folded into the MoE down-accumulate instead, by passing the shared buffers into
the merge kernels (`:3003` prefill, `:3008` decode -> `:1643-1650` decode merge,
`:1769-1770` prefill merge, `Qwen4ExpLayerMajor.cpp:687-688`).

Fused expression (`src/ops/MoeExpertOps.cpp`):
* ordered accumulate, decode/`tokens>=1`: `:6401-6406`
  `routed = bf16(total_val); gate_act = 1/(1+native::exp(-gate));
  y[gid] = bf16(float(routed) + gate_act*float(shared_out[gid]))`.
* CPU-host-row variant: `:6442-6448` (same expression, `shared_gate[0]`).
* compact/quantized finish: `:6348-6353`; other accumulate variants:
  `:1088-1096`, `:1568-1577`.
* split reference the flag replaces: `Qwen4ExpBnb4Backend.cpp:125-200`
  (`y_vec[v] = bf16(y_val + gate_act*s_val)`).

What the fused path assumes:
1. **Producer ordering.** `shared_gate_buf` is written by the MoE front
   (`:2732-2742`, front-fused gate) or `dense_rows` (`:2890`), and
   `shared_out_buf` by the shared expert block at `:2889-2926` - both *before*
   `RunAdaptivePrefill`/`RunResidentDecode` are submitted at `:3003`/`:3008` on
   the same in-order queue. If any caller reorders those submissions, the fused
   add silently reads a stale/near-zero shared row.
2. **Single application.** The add must happen exactly once per element: with
   the flag on, `add_shared_expert_to_y` must never also run (it does not).
3. **bf16 round-trip identity.** `bf16(total_val)` (`:6401`, `:6442`) must equal
   what the standalone float->bf16 conversion would have stored.
4. **Sigmoid cost / shape.** `gate_act` is recomputed **per output element**
   (`:6404`, `:6445`) with `sycl::native::exp`, whereas the split kernel
   computes it once per 8-element vector (`Qwen4ExpBnb4Backend.cpp:144`):
   8x more transcendental work per layer, inside the accumulate kernel.
   `shared_out[gid]` is additionally indexed by row (the split kernel indexes
   `s_ptr` with the same `gid`, so indexing matches).

Prior art in this repo says this flag is *not* numerically clean:
`docs/Qwen38FlashNext_A770_Handoff.md:244-246` records that it "changed some
BF16 layer-0 values and reduced the fixed bench from 10.91 to 7.52 tok/s in one
A/B run" and told us not to enable it by default. The 2026-10-16 audit's
expression-identity argument (`docs/plans/qwen38-r1-decode-submit-path-audit-2026-10-16.md:220-221`)
only covers the arithmetic, not assumption (1).

## 2. Falsifiable hypotheses for the -19.8%

Arms and raw evidence are in `tmp/q4e-sprint-r1-20261014/fh-*/result.json`
(A 17.445/17.549; B 13.734/14.328) and `f1-b1-worker.log`. Note
`tmp/q4e-sprint-r1-20261014/fh-*-worker.log` **does not exist**: `run.py` sets
`AILA_SERVER_WORKER_LOG` only under `--diag` (`tmp/q4e-sprint-r1-20261014/run.py:34-38`)
and the fh arms were run without it. The only worker-log evidence for a fused
arm is `f1-b1-worker.log` (`DECODE_FUSED_HC=1` alone, h4 profile, KV_QUANT off):
medians over 132 decode-token `[Q4E AdaptiveWork] count=1` lines -
`submitted_ms=63.49`, `hc_host_ms=5.66`, `experts_host_ms=26.79`,
`cpu_join_ms=16.23`, `router_wait_ms=16.27`, `shared_add_ms=0.67`. The production
reference for the same field is `hc_host_ms=6.10` with a -30% gate
(`docs/plans/qwen38-r1-decode-fusion-2026-10-15.md:201-206`, gate restated at `:288`).

| # | Hypothesis | What to check in a worker log | Cheapest discriminating measurement |
|---|---|---|---|
| H1 | **Fused pair is device-side slower** (24 KiB SLM/WG in `Qwen4ExpDecodeFusion.cpp:47-48` + per-WG duplicated norm `:102-138` cost more than the 1 submit/call saved; 97 HC calls/token) | `harness=diag` B arm: `submitted_ms` vs A arm at equal `pos`; `hc_host_ms` (f1-b1 shows only 5.66 vs 6.10 = -7%, NOT the -30% gate); look for a *device* stall (event wait) rather than host submit growth | A/A/B/A rerun of the *pair* with `--diag`, compare per-token `submitted_ms` medians at `pos>=14000` (1 extra ~7 min GPU round, no rebuild) |
| H2 | **Text-driven routing shift** - degenerate tokens route outside the `qwen38-nf4-agent6k-v1` hotset, so the CPU cold-expert path grows and the token becomes CPU-bound | per-token `hits`/`misses` deltas and `experts_host_ms`/`cpu_compute_ms`/`cpu_join_ms` in `[Q4E AdaptiveWork] count=1`; f1-b1 shows 110 hits / 370 misses per token and `cpu_join_ms=16.2`, i.e. 77% of ranks are CPU-served | run the **A** flags (agent6k + int8 KV) on a request that *starts* with the B-arm's degenerate assistant text (reuse `fh-b1/context-14000-continue-request.json` with `messages[1]` replaced by the B text) - if the A arm also drops to ~13.7 tok/s, the -19.8% is text-driven, not kernel-driven. ~1 arm, no rebuild |
| H3 | **Silent fallback** - `run_hc_norm_down_fused` rejects the real shape and the production dispatch runs, so the arm pays flag scaffolding without the fusion (no log exists for this) | nothing in the current logs distinguishes it; `hc_host_ms` is the only proxy (5.66 is consistent with the fusion being taken, ~97 x 4.5 us saved) | one-shot `AILA_LOG_INFO` at `Qwen4ExpBnb4Backend.cpp:2144` on taken/rejected, then a 1-request smoke (needs a build; defer until another build is scheduled) |
| H4 | **Missing/incorrect `H` update ordering** - the relocated update (`Qwen4ExpDecodeFusion.cpp:270-273`) lands one HC call later than production's `HyperConnectionOps.cpp:327-334`; the header's contract ("nothing writes H between the two submissions", `:44-56`) or the pending-residual handoff (`:2130-2154`) may not hold for the *attention* HC call that follows the MoE | `AILA_PROFILE_Q4E_DECODE=1` per-layer activation dumps (`AILA_DEBUG_Q4E_DUMP_DIR`) - compare `layer_*.bin`/`mixer.bin` A vs B for the **first differing layer** | CPU-only: run the path with `AILA_Q4E_HC_FUSED=1` + tiny shapes under the existing `Qwen4ExpDecodeProbe`/HC unit tests and diff fused vs split outputs bitwise (no GPU needed for the shape/order logic) |
| H5 | **Shared fusion double/zero add** - the fused add inside the merge kernel plus some other consumer of `shared_out_buf` (e.g. the front-fused path at `:2730-2740`, or MTP `Qwen4ExpMtpDraft.cpp:241-242`) | log the shared buffers A vs B (value compare) at layer 0 | run `MOE_SHARED_FUSED=1` **alone** on one arm and diff the 14K text vs the A text: if the text is identical to A, H5 is dead and the pair-only text change is pure HC |
| H6 | **int8-KV / hotset interaction** - the fusion changes the decode path only when int8 KV is on | compare the B-arm text with int8 KV on vs off | already answered: `f1-b1` (KV_QUANT unset, h4 profile) produces the *same* degenerate 14K text as `fh-b1` - H6 is **refuted** |

Standing result that any hypothesis must respect: the divergence begins at the
**2nd generated token**. Request 1 (`context-6000`, 5944 prompt tokens, no cache
in either arm) has the byte-identical request payload
(`messages_sha256=30c9efa7b597...`) and both arms start their reply with
`Based`; A continues `" on the provided ..."` while B continues
`"\n\n\n\n\n\n The The ..."`. Since the first token comes from the prefill's
final hidden state, the prefill (and therefore `RunAdaptivePrefill`) is
argmax-identical, and the defect is confined to the first `count == 1` decode
forward and later.

## 3. The prefix-cache interaction

Decision points (all in the pure-MSVC server):
* `tools/server/qwen38_context_probe.py:75-86` records `cached_prompt_tokens` per
  request and prints it; `result.json` carries it per request.
* The continuation request is built by re-rendering the previous answer into the
  conversation: `qwen38_context_probe.py:151-155`
  (`messages += [{"role":"assistant","content": answer["content"]}, {"role":"user", ...}]).
  **The cached token vector contains the worker's generated token ids, while the
  new prompt's ids come from re-tokenizing that text.** Reuse therefore requires
  a byte-exact re-tokenization round-trip of the generated text.
* `src/server/HttpServer.cpp:262-298` `TakeSession`: exact prefix match
  (`std::equal(tokens.begin(), tokens.end(), prompt.begin())`) and a **strict**
  `tokens.size() < prompt.size()`. A miss still consumes and erases the cached
  entry.
* `src/server/HttpServer.cpp:300-310` `KeepSession` (bounded by
  `maxSessions`, i.e. `--sessions 1` here) and `:893-911` the release-time gate
  `keep = !state.busy && state.contextLength && *state.contextLength == completedTokens.size()`.
* `:311-318` `ObserveCache` sets the per-record `cached_prompt_tokens` (`:224-232`)
  and the health counters printed at `:547-550`
  (`prefix_cache.hits`, `reused_tokens`, `idle_sessions`).
* Diagnostics exist and are env-gated only: `prefix_diag_enabled()`
  (`:33-42`, `AILA_SERVER_PREFIX_DIAG=1`) prints
  `[PfxDiag] release busy= ctx_len= completed= keep=` (`:900-905`) and
  `[PfxDiag] take cached_len= prompt_len= diverge_at= qualifying=` (`:273-291`).

Observed evidence: `fh-a1/report.json` =
`prefix_cache{hits:2, reused_tokens:19952, idle_sessions:1}` with per-request
`cached_prompt_tokens [0, 5976, 0, 13976]`; `fh-b1/report.json` =
`{hits:0, reused_tokens:0, idle_sessions:1}` with `[0,0,0,0]`. Both A and B kept
a session (`idle_sessions:1`), so `KeepSession` itself still works in the B arm;
the failure is in selection. The B-arm continuation payloads differ from the A
arm (`messages_sha256 070df5546387 -> a23456b2763a` and
`bf0679889b33 -> b4ed8e22b1e1`, with identical first-request payloads) and the
continue prompt lengths move (6000 -> 5976 and 14000 -> 13999). The B-arm
`context-6000-continue` prompt (5976 tokens) is exactly the length of the cached
vector (5944 prompt + 32 generated) that request 1 would have stored - in that
case `tokens.size() < prompt.size()` (`HttpServer.cpp:268`) alone rejects the
match, before any id comparison.

Conclusion: **unverified but narrowly scoped.** Two branches remain, and they
are distinguished by one env var, not by more tuning:
* (i) *take-side* mismatch: the degenerate reply re-tokenizes to a different
  id sequence (or to exactly the cached length), so `TakeSession` does not
  select it;
* (ii) *keep-side* rejection: `state.contextLength != completedTokens.size()`
  (`:895-899`) for the degenerate stream (e.g. a special/stop token counted on
  one side only).
Either way the cost is fixed and known: the two continuation requests fell back
to full prefill, `TTFT 31.5 s` and `71.5 s` (vs 1.56/1.44 s in A), i.e. the
~100 s wall gap, and `first_content_seconds` is excluded from the reported rate
metric, so the cache loss does **not** by itself explain the -19.8% decode rate.

Cheapest experiment to settle it (single arm, no rebuild, server-side env):
rerun one B arm with `AILA_SERVER_PREFIX_DIAG=1` and read
`[PfxDiag] release ... keep=` plus `[PfxDiag] take ... diverge_at=`. A single
4-request arm (~7 min GPU) answers (i) vs (ii) and gives the exact divergence
index. Do this *after* the numerics defect is fixed, or only as a one-off
diagnostic, since the arm itself is already known-bad.

## 4. Recommendations (max 4)

1. **Do not re-screen the pair.** Keep both flags at 0. The observed B text is
   degenerate (`f1-b1` reproduces it with `DECODE_FUSED_HC` alone on a different
   bundle), so this is a correctness regression, not a tuning question; the
   -19.8% and the cache loss are downstream symptoms. Expected e2e effect of
   doing nothing: 0 (baseline stays 17.50 tok/s A / 14.03 tok/s B).
2. **Fix the `DECODE_FUSED_HC` numerics first, and gate it on a bit-exactness
   test.** Before any further GPU time, add a CPU/GPU parity test that runs the
   fused pair against `grouped_rms_norm`/`try_hc_decode_inject_add_and_grouped_rms_norm`
   + `hc_decode_down_inject` + `hc_decode_up_mix` on the real shapes
   (`groups=4`, `hidden_size_=2560`, `cfg_.hc_lowrank`) and compares bitwise,
   including the pending-residual-armed case (`AILA_Q4E_DECODE_PENDING_MLP_RESIDUAL=1`).
   Expected e2e effect if it passes: +0.4-0.8% from the ~96 submissions/token
   (f1-b1 measured only 6.10 -> 5.66 ms `hc_host_ms`, so the submit-count win is
   ~0.4 ms/token, not the -1.0...-2.2 ms the audit projected).
3. **Re-screen `MOE_SHARED_FUSED` alone** only if the submit-count lever is
   still wanted, with the exact arms
   `run.py md-b1 --prime AILA_Q4E_DECODE_FUSED_HC=0 AILA_Q4E_MOE_SHARED_FUSED=1`
   (bundle unchanged: agent6k profile + int8 KV) plus an A arm, and read the
   `shared_add_ms` field (`0.67 ms/token` in `f1-b1`) and the 14K text vs A.
   Expected effect: `-48 submits/token` ~= -0.6 ms/token ~= **+1.0-1.2%** e2e at
   17.5 tok/s, and only if H5 (double/zero add) is dead. If the A/B text
   differs at all, treat it as a numerics defect (the handoff note already
   recorded BF16 value changes).
4. **If any GPU time is spent, spend it on the discriminator, not on a
   re-screen:** one B arm with `AILA_SERVER_PREFIX_DIAG=1` (settles section 3)
   and/or the H2 text-forcing arm (one A arm fed the degenerate assistant text).
   That converts "two live mechanisms" into one number and tells you whether the
   -20% is a kernel cost to fix or a hotset-profile coverage issue to fix
   (in the latter case the fix is profile/adoption, not the fusion).

## Appendix: evidence index

* Arms: `tmp/q4e-sprint-r1-20261014/fh-{a1,b1,b2,a2}/{result.json,report.json,launch.json}`,
  `.../fh-{a1,b1}-run.log`; arm script `.../run-fused-hc-shared.ps1` (bundle +
  the two flags).
* Solo-flag diag arm (the decisive one): `tmp/q4e-sprint-r1-20261014/f1-b1/`,
  `.../f1-b1-worker.log` (`worker_environment` in `f1-b1/result.json` shows
  `AILA_Q4E_DECODE_FUSED_HC=1` only).
* Bundle profile: `docs/plans/qwen38-server-mtp-2026-10-07.json`
  (`AILA_Q4E_DECODE_PENDING_MLP_RESIDUAL=1`, `AILA_Q4E_MOE_FRONT_FUSED=1`,
  `AILA_Q4E_HOTSET_PROFILE=...h4-v1.profile`; the fh arms override the profile
  and add `AILA_KV_QUANT=1`/`int8`).
* Prior numerics note for the shared fusion: `docs/Qwen38FlashNext_A770_Handoff.md:244-246`.
* Gate/expectation used by the fusion design:
  `docs/plans/qwen38-r1-decode-fusion-2026-10-15.md:195-305`.
