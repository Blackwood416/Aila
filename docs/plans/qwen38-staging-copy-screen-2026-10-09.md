# Qwen3.8 staging-copy screen — 2026-10-09

This report separates two measured A/B pairs from the post-commit official build archive and records the explicit retention disposition.

Parent commit: `c5d5805ef788802929b857a11a62f194dc635daf`. Runtime source patch: `tmp/q4e-stage-copy-screen-20261009/candidate-source.patch` (6 files, SHA-256 `ae7ed72b0f31722580cf3457cde171bf8314d66fe4029c66b427d4d292e91028`).

Only worker setting changed: `AILA_Q4E_STAGE_COPY_PARALLEL=0→1`. H4 profile, placement, swap budget/frequency, 2×16 ring, routing, CPU/GPU arithmetic, and all other worker environment values are gated as fixed.

Negative elapsed-time deltas mean the candidate was faster. Each row is one same-binary pair; full output, usage, cached-token and PI gates are listed in the JSON report.

| Pair | 14K total Δ | Continuation total Δ | Continuation TTFT Δ | Pair gates |
|---|---:|---:|---:|---|
| stage-a1→stage-b1 | +0.32% | -9.18% | -22.82% | pass |
| stage-a2→stage-b2 | +0.37% | -10.10% | -23.62% | pass |

| Request | Four-run same-work/output | Total median A→B | TTFT median A→B | Strict fixed-work qualified |
|---|---:|---:|---:|---|
| 14K first request | True | 72.619s → 72.869s (+0.34%) | 70.330s → 70.599s (+0.38%) | True |
| Continuation | True | 3.642s → 3.291s (-9.64%) | 1.467s → 1.126s (-23.22%) | True |

## Copy-only microbenchmark

16 × 2,534,400 B compact 2560×640 slots (40,550,400 B per pass), native large-page sources to Host USM, 20 interleaved pairs; full `memcmp` passed. Median: serial 5.73105 ms, parallel 1.57525 ms; copy-only ratio 3.64×.

The retained log contains 20-sample median/p10/p90 summaries, not individual timing values. This copy-only ratio is not an estimate of prefill or full-request speedup.

Normal adaptive-cache GPU-consumer test: PASS (`tmp/q4e-staging-parallel-20261009/cache-test.log`).

## Formal build archive and disposition

Decision: **retain** at commit `39625195bdadb996a77248b0d0136d01bcc3c849`.

保留依据：续轮总耗时两对分别降低 9.18%/10.10%，首字中位数降低 23.22%。14K 总耗时增加约 0.35%，不宣称长 prefill 改善。运行时默认关闭；Qwen3.8 launcher 默认开启并保留显式 flag=0。测量补丁为六文件，保留提交另含 launcher 一文件。

正式验收：Worker/Server 构建 exit=0，缓存 GPU 消费/owner/fence 专项通过；CPU/ABI 18 passed、1 conditional skip、0 failed；Server 4/4 passed。启动脚本语法通过。细节见 verification.json 和下面的日志。

The A1/B1 and A2/B2 entries above are the only timed measurement pairs. The official `retained/pair.json` below records post-commit Worker/Server build identities only; it is not a compare JSON, adds no timing sample, and does not require another production pair.

Official archive: `tmp/q4e-stage-copy-screen-20261009/retained/pair.json`; Worker build ID `aila-0.2.0-abi1-39625195bdadb996a77248b0d0136d01bcc3c849`, Server build ID `aila-0.2.0-abi1-39625195bdadb996a77248b0d0136d01bcc3c849`; archive commit matches disposition commit: `True`.

Check logs:
- `tmp/q4e-stage-copy-screen-20261009/build-worker.log` (SHA-256 `eaceb48205713a1440fbc991f6c2e52566a2eddbd87fd33e4139635569748968`, 34868 bytes)
- `tmp/q4e-stage-copy-screen-20261009/build-server.log` (SHA-256 `211bd011969bfc21eb29b955c6a9ac376d86f5fc1809d2979bed75edc8406bac`, 775 bytes)
- `tmp/q4e-stage-copy-screen-20261009/cache-formal-test.log` (SHA-256 `d2c4123967140bc020e0297ffc027330912aad75b80fb9bd18260f4899e9b217`, 241 bytes)
- `tmp/q4e-stage-copy-screen-20261009/ctest-cpu-abi.log` (SHA-256 `e15f9fd403b978d5c83eb6442bf24710546f1f2e41c954aecff31cfed8622da2`, 32402 bytes)
- `tmp/q4e-stage-copy-screen-20261009/ctest-server.log` (SHA-256 `eb9da67b68809f5e08cceb477eb48c69a602e45508aecdbb1d4dd8098268ca63`, 10714 bytes)
- `tmp/q4e-stage-copy-screen-20261009/verification.json` (SHA-256 `51026657b175bd508148de89fc0db665834ed97aef6624a37389a2046e734d4a`, 623 bytes)

## Range

Completed: A1/B1 and repeated A2/B2 flag-only paired screens; request and PI identity gates; raw result/monitor paths; staging copy and normal consumer validation logs.

Not established by this screen: broad stability beyond these two pairs or any CPU/GPU operator bitwise-equivalence claim.

Original run artifacts and all gate details are retained in the adjacent JSON report.
