# 给 A770 开 ESIMD 路径：可行性与分阶段方案（2026-10-12）

背景：`6b7286a` 修复了 int8 DPAS attention（根因是 **SG8**，不是工具链不可用），但实测 DPAS 版本
比向量 int8 路径 **慢 84.6%**（14k：1.925 vs 1.043 ms）→ 正确但默认关闭。由此提出：A770 是否值得
引入 ESIMD 路径？

## 1. 工具链事实（本机核对）

- oneAPI 2026.1 提供完整 ESIMD：`sycl/ext/intel/esimd.hpp` + `esimd/xmx/dpas.hpp`（公开
  `sycl::ext::intel::esimd::xmx::dpas<SystolicDepth, RepeatCount, T, CT, BT, AT>(C,B,A)`，
  s8 语义 = A(M×K)·B(K×N)+C(M×N)，K=32（SystolicDepth 8 × OpsPerChannel 4），N∈{8,16}，
  B 需 VNNI 布局）。
- 本仓库**零 ESIMD 使用**（仅 `Qwen4ExpDpasMoeProbe.hpp` 注释里声明"ordinary SYCL SG8, not ESIMD"）。
- int8 XMX 能力已在本机证实：SG8 joint_matrix 的 s8/s8/s32 走通（136/136 精确点积 + attention 8 组
  长度全过，稀疏/gated 用例 nrmse=0）。

## 2. 关键判断：DPAS 的问题不是"能不能用"，而是"M 太小"

- decode attention 是 **M=1**（单查询行）：硬件 DPAS 是 M8，7/8 算力天然闲置；`6b7286a` 的
  M1 实现因此比向量点积慢 84.6%。**ESIMD 不会改变这个形状事实。**
- 因此 ESIMD 的候选目标必须是 **M≥8 且指令/寄存器受限**的内核：
  1. **prefill MoE 专家 GEMM**（wgshare 4token 族）：占 prefill GPU ~44%，实测 100–140 GB/s 有效权重
     读、寄存器压力大（spill ~512 B）、每专家权重 ~25× 重读——M=quad×rows 天然 ≥8，是 ESIMD 寄存器分块
     + 显式 dpas 的最优候选；
  2. **chunked GDN 矩阵段**（T=16：三角求解 + 小 GEMM）：X4 的 joint_matrix/XMX 版比 FP32 chunk 慢
     （4.68 vs 3.38 ms），ESIMD 有机会翻转（M=16、K/N=128）；
  3. **prefill/多行 attention**（M=多行）：SG8 joint_matrix 已可用，ESIMD 的增量价值需用微基准证明。
- **明确非目标**：decode M=1 attention、CPU 专家路径（host DRAM 受限）、staging/H2D（DMA 受限）。

## 3. 分阶段方案（每阶段带门）

**Phase 0（0.5 天，先做）— 原始吞吐对照**
- 写一个 ESIMD DPAS 微基准（M8/N8/K32 与 M8/N16/K32，s8×s8→s32），量：有效 MAC/cycle、SLM 吞吐、
  寄存器打包开销；对照 (a) SG8 joint_matrix（`6b7286a` 路径）、(b) 现有向量 int8 点积。
- **门**：ESIMD 原始 DPAS 吞吐必须显著高于 joint_matrix 的同形状吞吐（≥20%），否则 ESIMD 无存在理由 → 停止。

**Phase 1（仅当 Phase 0 过门）— 单内核试点**
- 目标按预期收益排序：prefill MoE gate/up（44% × 潜在 −20~30%）> chunked GDN 矩阵段 > 多行 attention。
- 实现形态：新 `src/ops/*Esimd*.cpp` + env 开关（默认 off），保持现有 USM ledger/取消/位精确契约；
  验收 = 块级 ≥1% 或墙钟 ≥2%，否则归档为 opt-in/负结果。

**代价与风险**
- 新编程模型（`simd<>`/显式 SLM/`block_load`）与现有 54 个测试的位精确契约需要独立重建；
- 同一 IGC：ESIMD 是另一条 codegen 路径（对 DPAS 通常更直接），但仍有编译期风险；
- 维护分叉：即使试点成功，也应只覆盖 1–2 个热点内核，而不是把 Aila 迁到 ESIMD。

## 4. 与"继续用 joint_matrix"的关系

`6b7286a` 证明 SG8 joint_matrix 在本机可用——**若目标只是"用上 XMX"，先把 joint_matrix 用在 M≥8 的
形状上（prefill/multi-row）就够了**；ESIMD 的增量价值只在"需要寄存器级流水/显式 VNNI 打包，
joint_matrix 的 tile 约束挡路"时才成立。Phase 0 就是为这个判断服务的。

## 5. Phase 0 实测结果与判定（2026-10-12，已完成）

探针：`tests/ops/Qwen4ExpEsimdDpasProbe.hpp` + `--esimd-dpas-probe`（未接线生产、默认 off）。
证据：`tmp/q4e-esimd-dpas-20261012/{performance.md,probe-run-5.log}`。

- **正确性（全过）**：ESIMD `xmx::dpas` s8×s8→s32 在 M1..M8 × K32(链到 K256) × N8 与 CPU int64
  **逐位一致**；u8×s8 混合、A 经 SLM、MoE tile（m8×k256×n8 + per-64 scale）max_abs 1.86e-09。
- **N16：DG2 静默错误**（无异常、结果错）→ 任何接入必须**硬门 N=8**。
- **指令级微基准**：ESIMD rc=8/ilp=4 reg-resident 相对 SG8 joint_matrix **1.351×**（两人独立复核
  1.355×）→ 微基准门 GO；但该对照是寄存器驻留、无操作数流量的形态。
- **真实形状门（同形 ≥1.20×）：NO-GO**——K=2560/N=1280 与 K=640/N=2560 在 M8/32/64/128 网格上
  比值 **0.67–1.01**；计入 per-64 scale epilogue 后 **0.65/0.97**；两臂均只达 XMX 峰值 1–15%，
  瓶颈是 **staged 操作数从 global/L2 的流量与依赖链**，joint_matrix 的 2D block load 在该流量下
  ≥ ESIMD 的平坦 256B 载入（展开补偿无效）。
- **staging 成本（顺带量化）**：NF4→s8 解包 0.390 ms/launch（M=128，50.5 GB/s）、VNNI pack 0.104 ms、
  gate/up 三件套合计 0.876 ms；down 0.501 ms。
- **结论**：**不启动 ESIMD Phase-1 接入**；生产 prefill MoE 维持 SG8 joint_matrix/wgshare。
  下一步杠杆是**操作数流量/复用**（wgshare share 档、staging 槽复用、SLM 权重驻留），不是 DPAS 方言。
  本探针作为负结果归档（opt-in、无生产接线），N8/N16 规则与 staging 成本表留给后续复用。
