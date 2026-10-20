# QSA prompt XMX JIT failure: minimal geometry fix

Date: 2026-10-20
Workspace: `E:/RiderProjects/Aila`
Scope: standalone probe only; no production `src` edit, no GPU run, no bash/WSL.

## Observed failure

The first A770 run failed before numeric validation or timing:

```text
probe failed: The program was built for 1 devices
Build program log for 'Intel(R) Arc(TM) A770 Graphics':
IGC: Internal Compiler Error: Divide by zero
```

Receipt: `probe.err.log`.

## Minimal suspicious geometry

The probe's BF16 tiles were already XMX8-shaped:

```text
A: 8 x 16
B: 16 x 8
C: 8 x 8
```

but the work-item geometry was XMX16-shaped:

```text
kSubgroup  = 16
kSubgroups = 4
kWorkgroup = 64
sg_id      = lid / 16
```

The installed oneAPI definitions identify DG2/A770 BF16 as XMX8:

- `C:/Program Files (x86)/Intel/oneAPI/compiler/2026.1/include/sycl/ext/oneapi/matrix/static-query-use.hpp:151`
  accepts BF16 `M<=8, N=8, K=16`.
- The same header at line 209 selects `M=8, N=8, K=16` for
  `architecture::intel_gpu_dg2_g10`.

Therefore the primary static contradiction was an 8x8 XMX8 matrix tile carried by
a 16-lane subgroup. IGC's DPAS lowering has no valid row mapping for that pair;
the zero-derived geometry is the high-confidence trigger for the divide-by-zero
ICE. This is a geometry conclusion, not a claim about the exact IGC source line.

The other requested checks did not show a zero launch dimension:

- `count=64`, `kKvHeads=2`: 128 work-groups.
- Local range is now 32; global range is 4096.
- `local_accessor<QsaXmxShared, 1>` has range 1.
- `active_len` is positive for the 2052/4096 cases. An empty active set does not
  size the launch; it only skips the token loop.
- The PV tile bound is `kHeadDim / kXmxN = 32`, with four subgroups and stride 4.
- All compile-time tile dimensions are positive.

## Changes

### `probe_qsa_prompt_xmx.cpp`

1. Changed `kSubgroup` from 16 to the A770 BF16 XMX8 width of 8 at line 62.
   Four subgroups now produce a 32-thread work-group instead of 64.
2. Added static geometry gates at lines 88-93:

   ```text
   subgroup == 8
   subgroups > 0
   workgroup > 0 and divisible by subgroup
   workgroup / subgroup == subgroups
   ```

3. Added positive/range static checks for the XMX tile dimensions:

   ```text
   kXmxRows > 0
   kXmxK > 0
   kXmxN > 0
   kHeadDim % kXmxK == 0
   kHeadDim % kXmxN == 0
   0 < kRealRows <= kXmxRows
   ```

4. Added host-side guards for non-positive `count` and `max_seq_len` before
   constructing `nd_range` at lines 251-256.
5. Replaced manual `lid / kSubgroup` subgroup derivation with
   `sg.get_group_linear_id()` at lines 276-277.
6. Clamped device-side block and tail counts to non-negative values at lines
   281-285. The launch remains based on the fixed query-row and KV-head count.

The 12+4-row algorithm and the four-subgroup ownership scheme are unchanged.

### `build.ps1`

Added an opt-in `-Aot` build mode:

- Adds `-fsycl-targets=intel_gpu_acm_g10` to both compile and link.
- Locates the installed oneAPI 2026.1 `ocloc.exe` only for AOT builds.
- Forces IGC native device lowering during the link, without creating a SYCL
  device or launching the probe.
- `-CompileOnly` still stops before link.

### `aot-device-link.log`

Captured receipt for the A770 AOT device-link:

```text
Compilation from IR - skipping loading of FCL
Build succeeded.
Compilation from IR - skipping loading of FCL
Build succeeded.
built .\probe_qsa_prompt_xmx_aot.exe (not run)
```

## Verification

All commands were PowerShell. No GPU work was submitted.

| Check | Result |
|---|---|
| `build.ps1 -CompileOnly` | pass |
| Default JIT-style rebuild | pass, not run |
| `build.ps1 -Aot` A770 device-link | pass, not run |
| ASCII and trailing-whitespace scan | pass |

Artifacts:

| Artifact | Bytes | SHA256 |
|---|---:|---|
| `probe_qsa_prompt_xmx.exe` | 1,236,992 | `6B2ED088A0176381D87DD26AF6335FC63121961503EF3CDF756918A053DC22D6` |
| `probe_qsa_prompt_xmx_aot.exe` | 1,274,368 | `B8809E1B404C11A2A1854679CE84E9B86F73B0EB2FE6D5AC72679E826A29706D` |
| `probe_qsa_prompt_xmx.cpp` | - | `00D17D2D4B203BAF81E4683CDDCE1207373B7A2AC604FC214949E03FFEA50159` |
| `build.ps1` | - | `918DAE949E1F3D2524C516365A78F224158C999C36732B4FD28099BE9DD84F72` |

## Remaining risk

1. The JIT path itself was not executed. AOT device-link demonstrates that the
   installed IGC accepts the corrected geometry, but the driver's in-process JIT
   compiler is a separate execution path. The next GPU run is still required to
   confirm that the original ICE is gone.
2. `joint_matrix_apply` remains in both epilogues. AOT accepts it, so it was not
   changed preemptively. If a future JIT still reports an ICE, the next minimal
   isolation is to replace those callbacks with `joint_matrix_store` plus a
   shared-memory readback.
3. Numeric correctness, 2052/4096 timing, and the 40% speed gate remain unmeasured.
4. The probe assumes the DG2/A770 XMX8 subgroup model. Other accelerator families
   must query their matrix combinations rather than reuse this constant.
5. The host case builder rejects a zero selected-block count, so an actually empty
   active set is not exercised even though the kernel now handles `active_len=0`
   without a zero-sized launch.
