# Qwen3.8 方向 3：swap 子系统净价值门控（FREQ=0 vs 1）

> **2026-10-10 修正：本屏幕的"FREQ=0"臂无效。** `aila::env::read_int`（src/utils/EnvUtils.hpp:72-84）对 ≤0 的值回退默认值（`parsed > 0 ? parsed : default_value`），因此 `AILA_Q4E_ADAPTIVE_SWAP_FREQ=0` 实际读到默认 1——两臂都是 swap 开启（修正后核对：两臂逐桶命中率、swap_ms≈2.7ms/token、输出全部一致）。本屏幕不能作为"swap 净价值"证据；等效禁用的长窗口对照见 [qwen38-swap-longwindow-2026-10-10.md](qwen38-swap-longwindow-2026-10-10.md)。`MAX_SWAPS` 轴（2 vs 8，均 >0）不受此影响，既有结论仍有效。

日期：2026-10-10。基点 `HEAD=6dc4c3d`，二进制对 `2596b999…/68d78431…`（同构建戳 `aila-0.2.0-abi1-6dc4c3d…`）。
目的：2a（MAX_SWAPS 2→8 为负）之后，回答"当前同步 swap（≤2/token）是否有净价值"，以决定是否为异步化（候选 2b）投入实现。

## 设计

顺序 A1→B1→B2→A2，全部 `run.py <tag> --agent`；唯一变量：`AILA_Q4E_ADAPTIVE_SWAP_FREQ` **1（A，默认）对 0（B，禁用）**；两臂固定 `AILA_Q4E_MAX_SWAPS=2` 与 `AILA_Q4E_4TOKEN_WGSHARE=4`（新生产默认）。代码确认 `s_adaptive_swap_freq > 0` 才触发 `PlanSwaps/ExecuteSwapsAsync`（`Qwen4ExpBnb4Backend.cpp:2133-2141`），FREQ=0 即完全禁用。

## 结果

四轮 `passed=true`、PI 与取消恢复通过；请求 payload/输入/**输出 hash 两侧完全一致**（14K 与续轮；swap 关停改变了放置轨迹，但本窗口内未产出可观测的转发差异——换入专家未被后续采样路径使用，或计划为空）。

| 指标（两轮中位数） | FREQ=1 | FREQ=0 | 变化 |
|---|---:|---:|---:|
| 14K 首字(s) | 69.809 | 69.365 | -0.64%（prefill，不受 swap 影响，属噪声） |
| 14K 总耗时(s) | 72.176 | 71.729 | -0.62% |
| 14K decode rate(tok/s) | 13.100 | 13.113 | +0.10% |
| 续轮首字(s) | 1.480 | 1.482 | +0.14% |
| 续轮总耗时(s) | 3.745 | 3.738 | -0.17% |
| 续轮 rate(tok/s) | 13.689 | 13.738 | +0.35% |

本屏幕同时给出当日生产环境噪声底：prefill 首字臂间波动 ±0.6%（约 ±0.5 s），远超 swap 可能的贡献。

## 结论

- 在当前生产窗口（32-token decode + 24-token 续轮）上，swap 子系统（≤2/token）的净价值在噪声内（±0.35%），**异步化（2b）的收益上界（隐藏 swap 的 H2D 尾部）≤ ~1–2%，低于噪声与本季验证成本** → 不实施 2b，方向 3 以此证据关闭。
- 2a + 本屏幕合并结论：同步换入的边际成本高（≈1.8 ms/swap）且短期收益不可见；swap 的价值只在更长解码窗口（命中率漂移）才可能显现，若未来重启该方向，应先构造长窗口（数百 token）证据，而不是继续在 32-token 窗口调参。

## 证据位置

`tmp/q4e-resident-speed-20261009/swf-{a1,b1,b2,a2}/{result.json,…}` 与同名 run.log；分析：`swap_analyze.py`（env diff 仅 `AILA_Q4E_ADAPTIVE_SWAP_FREQ`）。
