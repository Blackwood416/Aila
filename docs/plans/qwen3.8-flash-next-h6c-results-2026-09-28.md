# H6c: short-prefill interior checkpoint experiment

Base: 2f122cc (H6b). Windows, Arc A770, driver 32.0.101.8860,
oneAPI 2026.1 Release. All builds, tests and inference processes run serially.

## Status and use

An opt-in H6c prototype is implemented. It is **not promoted to default** and
does not complete the general H6c accuracy/shape acceptance gate.

Set AILA_Q4E_INLINE_CHECKPOINT=1 before starting the process to try it;
the default is 0. The request must retain snapshot_policy=reusable_prefix.
The existing snapshot_policy=none one-shot path remains independent.

Automatic selection is limited to Q4E text prefill, at most 64 tokens in the
current slice, both sides of the anchor at least two tokens, total context
at most 2052, and at least two snapshot slots. Other cases retain H6b splitting.
These are prototype bounds, not a validated universal dispatch rule. Arbitrary
prompts within the bounds are not guaranteed to match split-prefill logits.

## Implementation

- ForwardOptions carries a slice-relative checkpoint_after boundary. A backend
  capability query lets the engine keep the full slice instead of cutting it.
- All layer projections, attention, hyper-connections and routed MoE process
  the full slice. Only GDN recurrence and PLE convolution split at the anchor.
  No GPU kernel geometry, arithmetic formula, quantization or compiler flag
  changes are introduced.
- Each GDN layer enqueues prefix recurrence, D2H into its preallocated snapshot
  arrays, then suffix recurrence on the same in-order queue. This preserves
  source lifetime until copies finish. PLE captures convolution state/head and
  the correct previous two token IDs before consuming its suffix.
- Raw QSA K and full-attention KV remain full-prefix caches. The checkpoint
  records the number of complete compressed blocks visible at the anchor;
  future compressed entries are invalidated by the restored count and rebuilt
  on append. This does not implement the raw-K ring proposal.
- The snapshot length is published only after all layers and copies complete.
  Final-length capture still occurs. Storage is the bounded H6b host pool;
  there is no additional GPU recurrent-state allocation.

## Numerical findings

The final full-prefill logits match a full-prefill run without checkpointing.
The captured FP32 recurrent and convolution arrays restore exactly. However,
changing the projection/attention batch shapes from split to full can change
the state reached at the anchor. This is not fixed by accurate snapshot copies.

Story prompt, 30 tokens, 248320 BF16 logits:

| Anchor | Split-reference replay | Finding |
|---|---|---|
| 1 | Different | First prototype selected decode GDN for a one-token subrange; now internal GDN always uses prefill and one-token sides fall back |
| 2 | 233411 differing logit bit patterns, max absolute difference 0.25 | Known batch-shape numerical difference; prevents default promotion |
| 16 | 235541 differing logit bit patterns, max absolute difference 0.304688 | A minimum-prefix threshold does not prove accuracy |
| 26 | 0 differing bit patterns | Golden chat anchor passes |
| 28 | 0 differing bit patterns | Near-end multi-token anchor passes |

The test reports the known differences at 2/16 diagnostically, requires exact
full-prefill output and state restoration at every tested anchor, and requires
exact split-reference replay at 26/28. A successful experimental test run must
not be interpreted as all anchors passing the split-reference accuracy gate.
Failed initial runs remain in tmp/h6c-real.log, tmp/h6c-real-2.log,
tmp/h6c-real-3.log and tmp/h6c-real-final.log.

## Performance

Three serial alternating pairs, one process and standard warmup per run:
on-1/off-1/on-2/off-2/on-3/off-3. Same binary for the six measurements.
ReadFile reader, Q4E profile and token-ID logging enabled. Real model:
C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2. CLI parameters: max-seq 512,
greedy, thinking-budget 0, max-tokens 48, perf/qwen4exp/prompts/story.json.

| Sample | H6b split, ms | H6c inline, ms |
|---|---:|---:|
| 1 | 5663.32 | 5417.95 |
| 2 | 5667.66 | 5456.96 |
| 3 | 5655.22 | 5401.72 |
| Median | 5663.32 | 5417.95 |

GenerateMessages timer decreases by 245.37 ms, **4.33%** in this three-pair
sample. On range: 5401.72-5456.96 ms; off range: 5655.22-5667.66 ms.
Summed prefill profile median decreases from 2164.87 to 1934.62 ms (~10.64%).
All six runs match the original golden 48 token IDs (288/288 comparisons).
This is a small end-to-end paired experiment, not a 20-sample kernel benchmark.
The timer excludes load/warmup and final context alignment/reset. No claim is
made for untested models, long contexts, arbitrary prompts or cold OS pages.

Logs: tmp/h6c-story-on-{1,2,3}.log and tmp/h6c-story-off-{1,2,3}.log.
Final boundary checks change no computation for the measured 26+4 shape.

## Verification and remaining work

Full build succeeded; final Q4E/chat/SafeTensors/build-identity CTest selection
passed 11/11 in 63.16 seconds. Real-model acceptance additionally covers
full-prefill logits, state restore, two-turn reuse, edited-history rollback
and switching snapshot policies. The real test is invoked explicitly:

    AilaQwen4ExpGenerationTests.exe --model C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2

Use AILA_Q4E_EXPERT_READER=readfile and AILA_Q4E_INLINE_CHECKPOINT=1 for this
acceptance. Logs: tmp/h6c-build-delivery.log, tmp/h6c-ctest-delivery.log,
tmp/h6c-real-delivery.log. Existing compiler warnings remain.

Next gate before default promotion: isolate split/full batch numerical changes
at operator boundaries, then expand prompt/anchor/context coverage. H6c beyond
short dense text, original P3 state-shard residency, P4 raw-K ring and AOT are
still pending. No change to default behavior is justified by this experiment.
