# Strata C3 筛选：16B 对齐装载（`load16_a2`）— NO-GO

日期：2026-10-13。范围：扫描文档 C3 的 Aila 最小实验。结论：Aila 的 NF4 权重布局本就
16B 对齐、decode GEMV 也不是装载宽度受限；唯一错位的 16B 装载（L1 常驻的激活向量）
拆成两个 8B 对齐装载后**无收益**（−0.5%，噪声内）。

## 1. 布局核查（为什么 Strata 的前提不成立）

- Aila 的 bnb-NF4 打包：`gate_up_packed [2I, H/2]`、`gate_up_absmax [2I, H/64]` **两个独立数组**
  （`src/ops/MoeExpertOps.hpp:24-31`）。行步长 H/2 = **1280 B**、down I/2 = **320 B**、
  块 = 32 B（64 权重）——全部是 16B 的整数倍；与 Strata 的 Q6_K（210 B 块步长）不同。
- decode GEMV（`launch_moe_decode_gate_up_swiglu_grouped`，`src/ops/MoeExpertOps.cpp:716`）的
  权重读取是 **uint32（4B，天然对齐）**；唯一的 16B 装载是激活向量
  `vec8<bf16>`，其地址 `byte_offset*2`（byte_offset = sg_lane*4 + 64i）使**一半 lane**
  落在 8B 对齐——即 Strata 记录的错位模式，但发生在只读一次/行、且 L1 常驻的激活流上。

## 2. 单变量实测（A770；`AilaQwen4ExpKernelTests --decode-gu-bench`，10 专家 × 640 行，
200 次 × 3 轮取中位；两个进程、只差一个 opt-in 开关）

| 变体 | median ms | 三轮 |
|---|---:|---|
| 原样（错位 16B 激活装载） | **0.12048** | 0.11907 / 0.12048 / 0.12817 |
| split 成两个 8B 对齐装载（位同值） | **0.11982** | 0.11960 / 0.11982 / 0.12419 |

- Δ = −0.5%，落在轮间散布内 → **不可判定为正收益**，按 C8 的 3% 发射规则否决。
- 佐证：6400 行 × 1280 B = 8.2 MB 权重在 0.12 ms 内读完 ≈ 68 GB/s ≪ VRAM 带宽（~500 GB/s）
  → 该核不是带宽/装载宽度受限（与 P4' 的解包/LUT 依赖链结论一致）。

变体补丁存档：`tmp/q4e-strata-c3/xsplit_variant_patch.diff`（已从内核回滚，生产核保持原样）。

## 3. 结论

**NO-GO**：不引入 `load16_a2` 类改造（无对应布局问题可修），不保留 xsplit 开关；
`--decode-gu-bench` 探针保留（kernel 级收据用），解码速率基线仍为 T1 ≈15.5 tok/s。
