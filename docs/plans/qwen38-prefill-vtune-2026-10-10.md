# Qwen3.8 prefill GPU 热点剖析与 4→8-token 窗口判别（2026-10-10）

基点 `HEAD=3962519`，正式二进制对（Worker `18cea21c…`、Server `ab04ae7d…`）。本文记录：GPU 引擎计数器采样、VTune xpu-offload 采集、4/2-token 窗口判别实验，以及据此选定的内核实验。所有时间口径与限定条件分节标注。

## 1. GPU 引擎利用率采样（Windows GPU Engine 计数器）

诊断运行 `head-diag2`（14K 输入、8 输出、chunk 4096、H4 profile、并行 staging）期间按 1 s 采样 AilaWorker 进程的引擎利用率（`tmp/q4e-resident-speed-20261009/head-diag2-gpu.csv`，采样窗口 11:42:35–11:44:00，覆盖 block0 尾部 + block1–3）：

- **compute 引擎 97–103% 持续饱和**；copy 引擎 52–75%。
- block 对齐（首字 107.08 s，块 56.0/19.7/19.9/11.3 s）：稳态 4096-token prefill 块由 GPU 计算主导；主机侧 fill/等待只是被 GPU 消化后的余量。
- 这解释了此前全部主机侧候选（staging 并行 fill 被 bank_wait 完全吸收、PLE 不在关键路径、router wait 为队列追赶）为何无法改变 14K wall。

## 2. VTune xpu-offload（Level Zero 编程 API 追踪）

采集：VTune 2026.2，`-collect xpu-offload`，整个 probe 进程树；无管理员权限故无 GPU 硬件计数器（EU/DRAM/L2 指标缺席），内核耗时来自 L0 事件时间戳。运行耗时 310.3 s（未采集时 294–296 s），无显著扰动。结果：`tmp/q4e-vtune-20261010/r1/`（285 MB），导出 `summary-800.csv`、`gpu-tasks.csv`。

### 主机任务（Top）

| 任务 | 时间 | 次数 |
|---|---:|---:|
| `zeEventHostSynchronize` | 60.16 s (19.4%) | 19,812 |
| `zeModuleCreate` | 39.23 s (12.6%) | 38 |
| `tbb_parallel_for` | 26.91 s (8.7%) | 315,634 |
| `clBuildProgram` | 4.64 s (1.5%) | 3 |
| `zeCommandQueueExecuteCommandLists` | 2.28 s (0.7%) | 101,597 |

`zeModuleCreate` ≈1 s/模块为运行期着色器编译；引擎主动强制 `SYCL_CACHE_PERSISTENT=0`（已知运行时崩溃的回避，`src/utils/EnvUtils.hpp:143-162`、`src/worker/WorkerMain.cpp:498`），每次启动重复支付，属既有取舍，本轮不改。

### GPU 任务族（全轮合计 135.4 s）

| 族 | 总时间 | 实例 | 均值 |
|---|---:|---:|---:|
| `launch_moe_expert_major_prefill_gate_up_4token` | 41.34 s | 8,665 | 4.77 ms |
| `zeCommandListAppendMemoryCopy`（全部 H2D） | 34.62 s | 67,393 | 0.51 ms |
| `launch_moe_expert_major_prefill_down_4token` | 13.86 s | 8,665 | 1.60 ms |
| `run_linear_delta_prefill_gpu_batched` | 13.80 s | 360 | 38.3 ms |
| `attention_prefill_qsa_batched` | 9.99 s | 3,960 | 2.52 ms |
| `packed_nf4_gemm_bf16_joint_matrix_wide`（3 实例化） | 8.85 s | 1,166 | 7.6 ms |
| MoE 2/1-token 变体（4 个） | ~4.1 s | ~34,700 | ~0.12 ms |
| decode 及其余内核 | ~9 s | 其余 | — |

H2D 合计 332.5 GB（无 D2H）。copies 与计算重叠（compute 100% / copy 52–75%），不是 wall 决定项。

### 解释（事实 + 推断分列）

`[FACT]` 专家 prefill ALU 路径（4token 族 + 小窗口变体）≈59 s / 135 s GPU 时间（~44%），是最大单项。分桶逻辑（`Qwen4ExpBnb4Backend.cpp` RunGroupedExperts）按 4-token 窗口组织，每专家权重被 `ceil(n_e/4)` 次重读；该 14K 语料 n_e 均值约 101 → 约 25× 重读放大（按代码结构推算）。
`[FACT]` grouped XMX 备选路径在 `AILA_Q4E_GROUPED_XMX_MIN_ROWS=0` 下不可达（`min_rows=INT_MAX`，`Qwen4ExpBnb4Backend.cpp:1304-1306`）；2026-09-29 X3 轮已测得该实现（M8/N32/K32）prefill 慢 3.0%/14.6%、专家级慢约 25×（`docs/plans/qwen3.8-flash-next-x3-results-2026-09-29.md`），本轮不重复。
`[INFERENCE]` 该内核族有效权重读取速率 ~100–140 GB/s（按配置的 quads×rows×行字节 / 实测时间），显著低于 A770 显存能力；但受寄存器压力（4token 内核 spill 512 B）与逐迭代 scale 广播等指令开销影响，不能仅凭该数字断言纯带宽瓶颈。

## 3. 4-token vs 2-token 窗口判别（单变量诊断）

`AILA_Q4E_4TOKEN_REUSE=0` 将 ALU 路径从 4-token quads 改为 2-token pairs（重读 ×2，内核启动数不变）。诊断（同输入、同 env，唯一切换项）对比：

| 指标 | 4-token（head-diag1/2） | 2-token（tok2-diag） | 变化 |
|---|---:|---:|---:|
| block1 submitted | 19.64 / 19.65 s | 23.76 s | +20.9% |
| block2 submitted | 19.89 / 19.91 s | 23.97 s | +20.4% |
| block3 submitted | 11.09 / 11.16 s | 12.38 s | +10.9% |
| 14K 首字（冷诊断口径） | 107.08 / 107.08 s | 120.39 s | +12.4% |

`[MEASURED]` 块耗时对窗口大小高度敏感，支持"权重流量显著参与支配"；按线性外推，8-token 窗口（重读减半）预计稳态块约 **-2 s（~-10%）**、14K 首字约 **-7 s**。该外推为 `[HYPOTHESIS]`，需 8-token 候选实测证伪或确认。

## 4. 选定实验：8-token 窗口变体（已完成，负结果）

新增 `AILA_Q4E_8TOKEN_REUSE`（默认 off）与 `launch_moe_expert_major_prefill_gate_up/down_8token`，分桶 eights→quads→pairs→singles；每条 token 的 K 累加顺序与归约顺序不变。位精确内核测试 env 开/关两遍全部通过。
同二进制 A/B 诊断（唯一切换 env）：稳态块 submitted **+22.8~23.1%**、14K 首字 +13.5%；VTune r8 显示 `gate_up_8token` spill 3,904 B（4token 320/576 B），寄存器溢出吞掉了权重重读减半的收益。
结论：8-token 窗口否决；结合 2-token 判别（+20.9%），窗口规模曲线在 4-token 取极小值，`W4/G3` 设计获实测支持。完整报告见 [qwen38-token8-window-screen-2026-10-10.md](qwen38-token8-window-screen-2026-10-10.md)；实现已作为 opt-in 提交（`31a8b78`，默认关闭），补丁副本在 `tmp/q4e-8token-20261010/`。

## 5. 其余观察（暂不改）

- DeltaNet prefill 单实例仅 48 个 workgroup（global 6144 / local 128）、38 ms/实例，共 13.8 s；并行宽度可能是后续候选，但未做定向剖析前不动。
- `zeEventHostSynchronize` 60.2 s 与前述 host 区间一致（排队追赶），已由 GPU 饱和结论解释。
- 启动期 39 s 编译成本为已知 crash 规避的代价，不在本轮范围。

## 原始证据位置

- GPU 采样：`tmp/q4e-resident-speed-20261009/head-diag2-gpu.csv`、`gpu_sample.ps1`
- VTune：`tmp/q4e-vtune-20261010/r1/`、`summary-800.csv`、`gpu-tasks.csv`、`findings.md`、`moe_rows.py`
- 判别轮：`tmp/q4e-resident-speed-20261009/tok2-diag-run.log`、`tok2-diag-worker.log`
- 对照诊断：`tmp/q4e-resident-speed-20261009/head-diag1|2-{run,worker}.log`
