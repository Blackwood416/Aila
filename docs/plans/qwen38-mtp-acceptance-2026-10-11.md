# B4-③：MTP 开启的 128-token 验收（结果：未通过等价门）

日期：2026-10-11。基点 `HEAD=e4d4bc5`（pair `4e5efc6c…/7f8f05b2…`）。
装置：`tmp/q4e-resident-speed-20261009/run_mtp_ab.py`（probe 无 fixture 合成提示，context=2048、chunk=512、prompt 目标 1500、generate=128、reuse=1；A 臂 `AILA_Q4E_SERVER_MTP=0`，B 臂 `=1` + 对齐 env 组合）。

## 运行结果（A1 / B1 / B2）

| 臂 | warmup | context-1500（prompt 1424） | 续轮（cached 1543/1550） | MTP 统计 |
|---|---|---|---|---|
| A1（T1） | 15 tok / 14.08 | **126 tok，stop**，ttft 14.88，rate 15.60 | 128 tok，rate 15.75 | — |
| B1（MTP） | 15 tok / 13.41 | **119 tok，stop**，ttft 16.11，rate **15.30** | 128 tok，rate 15.53 | windows=149 accepted=111（**74.5%**） |
| B2（MTP） | 15 tok / 12.87 | **119 tok**，ttft 16.42，rate **15.01** | 128 tok，rate 15.42 | windows=149 accepted=111 |

## 判定

- **MTP 正确接入并运行**：`[Q4E Server MTP] enabled=1 width=2 capacity=2048 draft_slots=64`；两个 B 样本 **windows/accepted 完全相同（149/111，74.5%）**，内容 hash 完全相同（确定性），warmup 短请求与 T1 逐位一致。
- **T1 等价门失败**：1424-token 提示的回答从 **字符 66（约第 10+ 个 token）**起与 T1 分歧（A 开头 "a series of modules" vs B "a large number of modules"；完成长度 126 vs 119）；续轮随之不同。B1/B2 分歧点一致 → 确定性偏离。
- **无性能增益**：B 臂 rate 15.01–15.30 vs A1 15.60（-2~-4%）；对齐所需的逐行 verify 投影本身有吞吐代价。

## 与对齐驱动的矛盾（下一步的入口）

B4-② 的驱动器证明的是 **VerifyRows(4) == 4×VerifyRows(1)**（verify 路径内部自洽）；而验收显示 **VerifyRows 路径 ≠ 普通 T1 decode 路径**，或 MTP 运行时（draft 运行/接受与 bonus 记账/共享状态）引入了偏差。候选根因：

1. 普通 count=1 decode 与 `decode_verification` 单行路径的调度不同（需补一个"VerifyRows(1) vs 普通 decode"的驱动器用例）；
2. draft 模型运行对共享状态（adaptive 路由观察 → 驻留/CPU 放置、hotset 历史）的影响使后续窗口走了不同 FP 顺序；
3. 接受/bonus 记账路径与 T1 的 token 发布路径存在细微差别。

## 结论

- B4-③ **未通过**：MTP 可用（集成、确定性、74.5% 接受率）但不等价于 T1 且无速度收益；**不开启默认**。
- 下一步（B4-④）：活线分歧定位——在 MTP 与会话内记录首个分歧 token 的产生路径（哪个窗口/行），并先补 "VerifyRows(1) vs 普通 decode" 的等价性证据；再决定是修运行时状态影响还是改默认宽度/保持实验态。

## 证据

`tmp/q4e-resident-speed-20261009/mtp-{a1,b1,b2}/`（result.json、*-run.log、*-worker.log 含 `[Q4E Server MTP] windows/accepted`）；对比脚本 `mtp_compare.py`。
