# Qwen3.8 R1: Strata structural reassessment (2026-10-18)

Scope: canonical Qwen4Exp decode and 32K production policy. This combines
three read-only audits plus one measured cold-split probe. No production
default was changed.

Input report: `tmp/q4e-strata-principle/REPORT.md`.

## 0. Decision summary

| question | current answer | disposition |
|---|---|---|
| Are GR/HC, router, shared experts, LM head and state on GPU? | yes; GR is Aila's HC, not a second weight set | aligned |
| Is anything important left on host? | token embedding is host by default; MTP draft uses the same host table | the only explicit placement deviation |
| Should token embedding be uploaded as BF16? | no: 1.184 GiB = 501.7 expert slots, while measured `ple_host_ms` is 0.240-0.251 ms including PLE | keep host until a quantized device lookup exists |
| Is KV in host? | no; target KV and GDN hot state are device-resident. Host GDN copies are snapshots only | aligned; KV streaming is a >=64K option, not a 32K fix |
| Does canonical decode implement Strata's three-way cold split? | no; all misses go CPU because `CPU_MAX_EXPERTS=10 == top_k` | topology mismatch confirmed |
| Does a direct-DMA 40/256 split fix it? | no: `16.232 -> 12.245 tok/s` (`-24.6%`) despite `fill_ms` falling to 0.047 ms | NO-GO |
| Does MTP currently run at 32K? | no; `Model.cpp` still hard-gates MTP to `maxSequenceLength <= 2048` and forbids Q8 LM head | production MTP is inactive |
| Is full 512-expert MTP residency the next MTP fix? | no; it costs 1.208 GiB = 512 main expert slots in Aila's NF4 geometry | closed |

The important difference from Strata is quantitative as well as structural:
Strata's Q2_0 expert is about 1.32 MiB; Aila's production NF4 slot is
2.5344 MB. A GPU-residency decision that is affordable in Strata can therefore
cost hundreds of Aila expert slots.

## 1. Component residency

Evidence and detailed line references:
`docs/plans/qwen38-strata-residency-audit-2026-10-18.md`.

| component | Aila production placement | comparison with Strata |
|---|---|---|
| GR / Hyper-Connection / norm | device; uploaded at load, normal decode kernels are GPU | aligned |
| router weights and top-k | router weights device; top-k writes pinned host ids/weights for scheduling | aligned, with an unavoidable route readback |
| shared experts | device weights, GPU projection/SwiGLU | aligned |
| token embedding | host large-page table by default (`AILA_Q4E_HOST_EMBED=true`); per forward: token-id D2H, CPU row gather, activation H2D | not aligned as placement, but BF16 upload is not economic |
| LM head | device BF16 transposed/oneDNN head | aligned; Q8 is an opt-in memory variant |
| target KV | device; int8 is opt-in | stronger than Strata for current 32K |
| GDN recurrent/conv | device hot state; host vectors are checkpoint mirrors | aligned; host mirror can be lazily allocated |
| routed expert complement | native large-page host arena; 48 512-expert layers | aligned |

Production memory admission for the current int8 sprint is:
`4522.3 MB static + 622.2 MB session + 3584 MB reserve + 7204 MB expert
cache`, admitted as 2800 expert slots.

### 1.1 Token embedding

The source path is `Qwen4ExpBnb4Backend.cpp:540` and `:2066-2083`.
The embedding table is BF16 `[248320,2560]`:

```text
bytes = 1,271,398,400 = 1.184 GiB
expert-slot equivalent = 501.7
```

The measured PLE/host bucket is only 0.24-0.25 ms in steady decode. Since that
bucket also contains PLE work, even a free embedding gather can remove less
than 0.5% of token time while deleting about 502 expert slots. Moving the raw
BF16 table to GPU is therefore rejected without another GPU run.

The correct alignment path, if this is revisited, is a quantized GPU embedding
lookup (for example Q8 rows) whose saved data-movement cost is greater than the
hit-rate loss from the slots it consumes. It is not an unconditional move.

### 1.2 LM head and MTP coupling

The production LM head is already on device. The BF16 head is 1.184 GiB.
`AILA_Q4E_LM_HEAD_Q8` reduces it to about 0.629 GiB, saving about
235 expert slots, and has measured decode gains in the prior receipts.
It is currently opt-in and hard-incompatible with MTP.

## 2. Cold experts: topology vs measured economics

Detailed audit:
`docs/plans/qwen38-strata-cold-expert-split-audit-2026-10-18.md`.

### 2.1 What canonical code does

For each decode layer:

```text
hits  -> resident GPU slots
misses -> StartArenaCold(...)
StartArenaCold: cold_count = min(miss_count, CPU_MAX_EXPERTS)
production: CPU_MAX_EXPERTS = 10, top_k = 10
=> every miss goes to CPU; gpu_cold_count is zero
```

`COLD_POLICY=3` and `COLD_PCIE_NUM=141` are read only by the legacy
`StartResidentCold` path. They do not create a split in canonical adaptive
decode.

This confirms the user's topology concern: Aila has GPU-resident hits plus CPU
misses, not Strata's GPU-hit / PCIe-cold / CPU-cold three-way split.

### 2.2 Why the existing cap=4 split was rejected

The previous `dec-cpu4` arm moved about 24% of cold tasks to the current
staging path:

```text
15.50 -> 9.40 tok/s (-39.4%)
fill +24.4 ms/token
H2D +132.9 MB/token
cpu_join -7.0 ms but router_wait +12.8 ms
```

The path copied the large-page arena into a pinned staging bank first. That
5.45 GB/s coordinator copy and the copy-engine lane were additive, not
independent. This is a valid rejection of cap/fraction sweeps on the old
bounce implementation, not proof that every possible three-way design is
invalid.

### 2.3 Direct-arena 40/256 probe

One new implementation was tested:

```text
AILA_Q4E_COLD_SPLIT_NUM=40
AILA_Q4E_STAGE_DIRECT_ARENA=1
```

The split is token-scoped with a Bresenham credit, and the copy engine reads
the large-page arena directly. It bypasses `staging_host_` and the coordinator
memcpy:

```text
direct queue.memcpy(staging_device + i*slot_bytes, arena_slot_i, slot_bytes)
```

Source commit: `9fee30f`.

Production A/B, 14K continuation:

| arm | decode tok/s |
|---|---:|
| `dcs-a1` canonical all-CPU | **16.232** |
| `dcs-b1` direct 40/256 | **12.245** |

Delta: **-24.6%**. The mechanism was confirmed in `dcs-d1`:

| field | all-CPU `evd-b2` | direct 40/256 `dcs-d1` |
|---|---:|---:|
| `fill_ms` | 0.000 | **0.047** |
| `cpu_join_ms` | 12.030 | **0.978** |
| `cpu_compute_ms` | 16.255 | 16.364 |
| `router_wait_ms` | 16.838 | **31.329** |
| `bank_wait_ms` | 0.000 | **16.699** |
| `experts_host_ms` | 23.648 | 34.128 |
| `submitted_ms` | 58.835 | **83.336** |
| H2D | 0 MB | **113.6 MB/token** |
| staged | 0 | **47 experts/token** |

The host fill was eliminated, but the GPU-cold lane still serialized against
the main device queue and staging banks. The extra `router_wait` and
`bank_wait` are larger than the CPU-join saving. **C1 is NO-GO.** Do not scan
`COLD_SPLIT_NUM=8,16,24,32,...`; the lane is not independent.

The empirically correct policy for the current WDDM/copy-engine path remains
all cold misses on CPU. A future split needs a device-side or otherwise truly
independent cold lane; a numeric ratio alone is not a structural fix.

## 3. MTP path

Detailed audit:
`docs/plans/qwen38-strata-mtp-path-audit-2026-10-18.md`.

Current facts:

- `Model.cpp:800-802` hard-gates MTP to `maxSequenceLength <= 2048`,
  `GDN_CHUNK_MODE==0`, and `!AILA_Q4E_LM_HEAD_Q8`.
- The 32K production server therefore never constructs MTP.
- The draft layer uses a fixed 64-slot expert cache (154.7 MiB). The target
  still uses the normal routed-expert path.
- MTP width is 2 (one draft token + one verify row). Measured acceptance is
  about 74.8%.
- The measured verify window is 92.26 ms; full MTP window is about 112 ms
  versus roughly 57.7 ms for T1.

Why simply copying Strata's MTP layout is not viable:

- Aila NF4 MTP experts: `512 * 2,534,400 B = 1.208 GiB`, equal to 512 main
  expert slots. Strata's Q2 experts are about 1.32 MiB each, so its
  512-expert residency is a different memory trade.
- A 128-slot MTP residency experiment was already negative
  (`19.95 -> 15.38 tok/s`); full 512-slot residency is not the next step.
- Width 3/4 requires a marginal row of roughly 9-13 ms; measured marginal
  rows are 40-50 ms. Width expansion is NO-GO.
- The draft embedding row is only 5,120 bytes; its blocking H2D upper bound
  is about 0.04-0.14 ms/window. Uploading the 1.184 GiB table to fix it is
  economically wrong.
- Even if draft, capture, select and sync were free, the 92.26 ms verify
  window caps width-2 throughput at about 18.88 tok/s. MTP needs a faster
  verify path before it can target 20 tok/s.
- The current CPU verify batch does not deduplicate the same expert across
  multiple rows. The unknown cross-row overlap still bounds a possible
  saving, but width expansion and full draft residency are not justified by
  current receipts.

Best MTP structural candidate: share a `ForwardLmHeadRows` dispatch between
target and draft and support the Q8 head for both. It saves about 568 MiB /
235 main slots and can save roughly 2.3-3.0 ms/window. This is a MTP
availability/efficiency fix, not yet a demonstrated 20 tok/s path.

A per-request 2048 MTP window is also only an availability fix. A global
14K/32K draft window costs 4.2-18.2 s of prompt catch-up in current estimates
and is NO-GO without a batched catch-up design.

## 4. KV and GDN state

Current target KV and GDN hot state are on the GPU. At 32K, moving KV to host
would add per-layer PCIe traffic; Strata's KV-streaming gain is measured at
64K+ and does not justify a 32K default change.

One small memory cleanup remains: worker admission uses
`maxRecurrentSnapshots=0`, but each session still allocates a 112.2 MiB GDN
host mirror. Allocate it only when a snapshot is actually requested. This does
not change the steady-state GPU compute path.

## 5. What not to do next

- Do not upload the BF16 embedding table or draft embedding table.
- Do not retry `COLD_POLICY=3` or `CPU_MAX_EXPERTS=4`; neither is connected to
  the adaptive path in the way the old plans assumed, and both existing
  implementations are negative.
- Do not sweep cold split fractions after the direct-DMA result.
- Do not make MTP width 3/4, full 512-slot MTP residency, or a 14K/32K MTP
  window the next default candidate.
- Do not reopen whole-token graph + host route servicing; the A770/WDDM P0
  already showed that payload visibility occurs only at graph exit.

## 6. Next useful structural work

1. **MTP Q8 head compatibility + shared LM-head row dispatch.** It is the
   only MTP change with a positive measured upper bound and bounded memory
   benefit.
2. **Lazy GDN host mirror allocation** when snapshots are disabled. Small,
   low-risk memory cleanup.
3. **Prefill `4 x 4096` layer-major cold-union probe.** This remains the
   only prefill supply candidate with a credible multi-second upside; do not
   confuse it with the rejected cold decode split.
4. If decode cold experts are revisited, first remove the host-facing
   coordination/device boundary, not the ratio. The direct-arena receipt
   proves that bypassing the bounce fill alone is insufficient.

## 7. Receipts

- `docs/plans/qwen38-strata-residency-audit-2026-10-18.md`
- `docs/plans/qwen38-strata-cold-expert-split-audit-2026-10-18.md`
- `docs/plans/qwen38-strata-mtp-path-audit-2026-10-18.md`
- `tmp/q4e-sprint-r1-20261014/dcs-a1/result.json`
- `tmp/q4e-sprint-r1-20261014/dcs-b1/result.json`
- `tmp/q4e-sprint-r1-20261014/dcs-d1-worker.log`
- `tmp/q4e-sprint-r1-20261014/evd-b2-worker.log`
- source candidate commit `9fee30f`

Relevant audit commits: `ab28c3c`, `79aee36`, `6b54bcb`, `72bf6c2`.
