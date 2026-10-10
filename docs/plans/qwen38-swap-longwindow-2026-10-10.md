# Qwen3.8 swap 子系统：长窗口证据与"默认关闭"采纳

日期：2026-10-10。基点 `HEAD=6dc4c3d`。本文替代失效的 [swap-frequency 屏幕](qwen38-swap-frequency-screen-2026-10-10.md)，并记录一个环境变量语义缺陷的修复。

## 1. 关键发现：`read_int` 使 `FREQ=0` 不可表达

`aila::env::read_int`（`src/utils/EnvUtils.hpp:72-84`）对 ≤0 的解析值**回退默认值**（`parsed > 0 ? parsed : default_value`）。因此：

- `AILA_Q4E_ADAPTIVE_SWAP_FREQ=0` 实际读到默认 1 → **swap 无法通过 env 关闭**；此前的 swap-frequency 屏幕两个臂都是开启状态（逐桶命中率、swap_ms≈2.7ms/token、输出完全一致——已在该文档顶部标注无效）。
- `AILA_Q4E_MAX_SWAPS=0` 同理回退 2，而非文档所称"无上限"。
- 对照正确性：`MAX_SWAPS` 的 2 vs 8 屏幕不受影响（均 >0，结论仍有效）。

等效禁用实验用 `FREQ=1000000000`（`cached_len % 1e9 != 0`）完成；修复后 `FREQ=0` 直接可用（见 §4）。

## 2. 长窗口对照（256 token，诊断口径，同二进制单变量）

| 指标 | swap 开（FREQ=1） | swap 关（FREQ=1e9） |
|---|---:|---:|
| `swap_ms`/token | 2.70（256/256 非零） | **0.00** |
| decode 中位延迟 | 67.2 ms | **62.0 ms（-7.7%）** |
| 命中率 前半 / 后半 | 52.2% / 39.0% | 51.2% / 34.6% |
| 后半 CPU 任务/桶 | 282–301 | 298–331 |

[MEASURED] swap 每 token 实测 2.7 ms 同步执行，另有 ~2.5 ms 连带（提交/等待与传输竞争）；其收益仅为减缓命中率衰减（256 token 处 +4.4pp ≈ 少 20–30 个 CPU 任务 ≈ ~1 ms/token）——成本/收益比约 5:1。长窗口内 OFF 仍净胜。

## 3. 生产 A/B/B/A（run.py --agent，同二进制）

顺序 开→关→关→开（`swapoff-{a1,b1,b2,a2}`）。四轮 `passed=true`、PI 与取消恢复通过；请求 payload 与**输出 hash 两臂逐位一致**（swap 的放置变化未影响采样路径）。

| 指标（两轮中位数） | swap 开 | swap 关 | 变化 |
|---|---:|---:|---:|
| 14K decode rate (tok/s) | 13.732 | **14.835** | **+8.03%** |
| 14K 总耗时 (s) | 70.285 | 69.831 | -0.65% |
| 续轮 rate (tok/s) | 14.485 | **15.560** | **+7.42%** |
| 续轮总耗时 (s) | 3.619 | **3.479** | -3.86% |

## 4. 采纳

- `AILA_Q4E_ADAPTIVE_SWAP_FREQ` / `AILA_Q4E_MAX_SWAPS` 改用 `read_int_raw`；FREQ 默认值 **1 → 0（默认关闭同步 swap）**，MAX_SWAPS 默认 2（保持 `0=无上限` 的文档语义）。
- 重新启用（做长窗口/自适应实验）：`AILA_Q4E_ADAPTIVE_SWAP_FREQ=1`。
- 风险与边界：关闭后命中率随路由漂移自然衰减（256 token 实测约 -4pp 差距）；若未来出现 ≥1024 token 长生成的应用证据，需重测成本/收益并考虑"H4 离线重排 + 低频率 swap"组合。默认关闭在本窗口内是实测最优（0 vs 1 vs 8 三点：关闭最优）。
- 变更验证：默认配置生产复跑 `swapoff-c1` = **15.266 tok/s**（续轮 15.985；与 FREQ=1e9 臂同档、远离 ON 臂的 13.7），确认默认行为已切换为关闭。

## 证据位置

- 长窗口：`tmp/q4e-resident-speed-20261009/long-{on1,off1,off2}/`（off1 为无效对照）+ `long_analyze.py`
- 生产四轮：`tmp/q4e-resident-speed-20261009/swapoff-{a1,b1,b2,a2}/`（`swap_analyze.py` 输出：env diff 仅 FREQ）
- 无效屏幕更正：`docs/plans/qwen38-swap-frequency-screen-2026-10-10.md`（顶部修正说明）
