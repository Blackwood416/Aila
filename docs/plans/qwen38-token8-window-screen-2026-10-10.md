# Qwen3.8 内核实验：8-token 权重复用窗口（负结果）

日期：2026-10-10。基点 `HEAD=3962519` + 候选补丁（`tmp/q4e-8token-20261010/candidate.patch`，5 文件 +1013/−27）。
候选二进制：Worker SHA-256 `aa4afa8133a53f0c01eb947062bdbaee57d446195441b8d2e5571f32aab221f0`（Archive: `tmp/q4e-8token-20261010/cand/AilaWorker-token8.exe`）；Server 保持官方 `ab04ae7d…`。

## 假设与实现

[HYPOTHESIS] VTune 显示专家 prefill ALU 路径占 GPU 时间 ~44%，其 4-token 窗口使每专家权重被 `ceil(n_e/4)` 次重读（n_e 平均 ~101 → ~25× 放大）；把窗口扩到 8 token 可减半重读，预计稳态块 −10%。

候选新增 `AILA_Q4E_8TOKEN_REUSE`（默认 false）：`launch_moe_expert_major_prefill_gate_up/down_8token`（16 个 gate/up 累加器）+ 分桶 eights→quads→pairs→singles + host 侧布局数组。每条 token 的 K 累加与归约顺序不变。
位精确验证：`AilaQwen4ExpKernelTests` env 开/关各跑一遍，全部 `[PASS]`，含 `G4 8-token` 直连位精确用例（0 mismatches，`tmp/q4e-8token-20261010/kernel-tests*.log`）。

## 测量（同一候选二进制，唯一切换 `AILA_Q4E_8TOKEN_REUSE` 0/1，`--diagnostic`，同输入）

| 指标 | 4-token（tok8d-a1, env off） | 8-token（tok8d-b1, env on） | 变化 |
|---|---:|---:|---:|
| block1 submitted | 19.644 s | 24.177 s | **+23.1%** |
| block2 submitted | 19.853 s | 24.382 s | **+22.8%** |
| block3 submitted | 10.918 s | 12.882 s | **+18.0%** |
| experts_host（块1） | 11.388 s | 15.385 s | +35.1% |
| bank_wait（块1） | 8.280 s | 12.236 s | +47.8% |
| fill（块1） | 2.444 s | 2.420 s | ~0 |
| 14K 首字（冷诊断口径） | 111.16 s | 126.15 s | +13.5% |

env off 臂稳态块 19.64/19.85/10.92 s 与此前基线（head-diag1/2 的 19.64/19.89/11.09–11.16 s）一致，证明默认路径无回归。

## 机制（VTune r8 采集，内核测试环境）

`tmp/q4e-vtune-20261010/r8`（`r8-gpu-tasks.csv`，解析见 `token_kernel_rows.py`）：

| 内核 | Spill 内存 | 平均实例时间 |
|---|---:|---:|
| `gate_up_4token` | 320/576 B | 476 µs |
| `gate_up_8token` | **3,904 B** | 1,373 µs |
| `down_4token` | 0 B | 261 µs |
| `down_8token` | 0 B | 412 µs |

16 个累加器令寄存器溢出暴增（~7–12×），本地内存流量与占用率损失超过权重重读减半的收益；`bank_wait` 增长与 GPU 消费变慢一致。

## 结论

- 8-token 窗口否决；连同此前的 2-token 判别（+20.9% 块耗时），窗口规模对块耗时的曲线在 4-token 处取极小值，`W4/G3` 分桶设计得到实测支持。
- 该实现已按 opt-in 处置提交为 `31a8b78`（默认关闭）；补丁副本与全部证据保留在 `tmp/q4e-8token-20261010/`。
- 后续若要继续削减专家 prefill 成本，需要改变数据流结构而非窗口参数：例如让同一专家的权重在一次内层遍历中服务更多 token 而不增加线程寄存器（分块/SLM/多行协作），或重新设计 grouped-XMX tile；此前 X3 的 M8 grouped XMX 已被实测否决，方向需另行论证。

## 证据位置

- 候选补丁/构建/测试：`tmp/q4e-8token-20261010/{candidate.patch,build*.log,kernel-tests*.log,cand/}`
- 两臂诊断：`tmp/q4e-resident-speed-20261009/tok8d-{a1,b1}-{run.log,worker.log}`
- VTune：`tmp/q4e-vtune-20261010/r8/`、`r8-gpu-tasks.csv`
