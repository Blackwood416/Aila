# Qwen3.8-Flash-Next (qwen4_exp) on Intel Arc A770：纯 GPU 深度优化 Brainstorm 路线与方案报告

> 后续评审（2026-09-29）：本文保留为 G1–G6 的历史设想，不代表所有预测收益已实现。关于“MoE XMX 不划算”的一概排除、SLM 层级、收益相加和 bitwise 准入，已在 [新一轮 XMX Prefill 与容差方案](/E:/RiderProjects/Aila/docs/plans/qwen3.8-flash-next-xmx-prefill-tolerance-plan-2026-09-29.md) 中纠正；后续实现以新方案为准。

**日期**: 2026-09-28  
**目标硬件与环境**: Intel Arc A770 16GB GDDR6 (DG2-512, Xe-HPG, 560 GB/s 带宽, 512 XMX, 16MB L2), Windows 11 25H2, oneAPI 2026.1 (`icx-cl`, SYCL, Level Zero), PowerShell  
**核心参考文档**:
- CPU 审查结论：[`docs/plans/qwen3.8-flash-next-cpu-route-value-review-2026-09-28.md`](file:///E:/RiderProjects/Aila/docs/plans/qwen3.8-flash-next-cpu-route-value-review-2026-09-28.md)
- 历史交接文档：[`docs/plans/qwen3.8-flash-next-a770-handoff-2026-09-27.md`](file:///E:/RiderProjects/Aila/docs/plans/qwen3.8-flash-next-a770-handoff-2026-09-27.md)
- 异构阶段成果：[`docs/plans/qwen3.8-flash-next-heterogeneous-progress-2026-09-28.md`](file:///E:/RiderProjects/Aila/docs/plans/qwen3.8-flash-next-heterogeneous-progress-2026-09-28.md) 与 [`docs/plans/qwen3.8-flash-next-h6c-results-2026-09-28.md`](file:///E:/RiderProjects/Aila/docs/plans/qwen3.8-flash-next-h6c-results-2026-09-28.md)
- W0/W4 首批实施记录：[`docs/plans/qwen3.8-flash-next-a770-w0-implementation-484b086.md`](file:///E:/RiderProjects/Aila/docs/plans/qwen3.8-flash-next-a770-w0-implementation-484b086.md)

---

## 1. 调研背景与前序假说纠偏 (Critical Review)

在分析 CPU 路线退出结论（端到端倒挂 53.33%：CPU 8.7s vs GPU 5.7s，因 3.1s 计算 + 2.8s 等待 + 1.0s 准备）并转向纯 GPU 优化的过程中，对部分潜在设想进行了严格的微架构与代码级复核，排除了 4 处重大误区：

1. **排除“HyperConnection SLM 缓存激活”误区**：
   - 违背 [`AGENTS.md`](file:///E:/RiderProjects/Aila/AGENTS.md) §7 明确记录的负优化规则（“SLM input caching in GEMV: Bank conflicts (32 banks cannot service 16 lanes × vec8)”）。
   - 激活值 $N$ 仅 20 KB，A770 拥有 16MB L2 Cache，命中率接近 100%。真正的瓶颈是 324 个 Workgroup 仅启动 5 次内循环并在 SLM 做 Barrier 规约。
   - **正解**：采用类似 `bf16_gemv_bf16` 的 **Subgroup-Cooperative 映射**，1 个 Subgroup 算 1 行，完全消除 SLM，利用寄存器级 `sycl::reduce_over_group(sg, ...)` 规约。
2. **纠正“MoE Router 主机同步等待 13~14 ms”的统计幻象**：
   - 检查真实无 profile 模式下的日志，48 层 `router_sync` 主机等待总和仅为 **1.54 ~ 1.94 ms**（平均每层 36 微秒）。
   - 调试模式下的 13.6 ms 实际包含了当前层/下一层推测 Router GEMV 与 Shared Expert 的完整 GPU 密集算子耗时，并非主机阻塞。
   - **正解**：不搞复杂的设备端槽位指针寻址，重点削减推测 GEMV 与融合 Shared Expert。
3. **纠正“Prefill MoE XMX 重构”误区**：
   - DG2 XMX 原生不支持 NF4，动态反量化为 BF16 并填充至 8 的开销对于短 Prefill（平均 $M_e \approx 3.7$）得不偿失。
   - W4 阶段已实现的 2-Token 权重复用算子（`launch_moe_expert_major_prefill_gate_up_2token` / `_down_2token`）在真机上已验证有最高 1.85x 加速。
   - **正解**：默认推广 W4 2-Token 复用，并扩展为 4-Token 矢量反量化寄存器复用。
4. **排除“GDN 状态矩阵 SLM 缓存”超限缺陷**：
   - 每个 Head 的 $S$ 状态矩阵为 $128 \times 128$ floats = 64 KB，加上局部参数已突破 A770 单 Xe-core 64 KB SLM 物理上限。
   - **正解**：采用 **单次写入重构（Single-Write Pass）**，循环 1 只读 DRAM 累加 `acc` 彻底消除回写，循环 2 唯一次写入，直接减半 50% 显存写回流量，且浮点等价零精度风险。

---

## 2. 六大 GPU 核心优化路线 (GPU Optimization Brainstorm)

### 路线一：HyperConnection (HC) Subgroup 协同 GEMV（最高 ROI 攻坚）
- **现状瓶颈**：48 层 Attn/MLP HC 及 Mixer HC 共 97 个模块，静态权重 1.28 GB，当前算子有效带宽仅 75 GB/s（利用率 13.4%），Decode 单步耗时 16.5~18.0 ms。
- **实施方案**：
  - 将 Workgroup 大小设为 256（16 个 Subgroups），每个 Subgroup 独立负责 1 个输出行，利用寄存器 Shuffle 规约（`sycl::reduce_over_group`），彻底消除 SLM 障碍与组内 Barrier；
  - 激活值 $N$（20 KB）直接从 16MB L2 Cache 并发读取；
  - 离线加载期重排 `w_up` 权重使 Subgroup 能连续发起 128-bit 矢量加载。
- **预期收益**：有效带宽跃升至 380~420 GB/s，耗时压缩至 **3.5 ~ 4.2 ms**，单步节省 **~12 - 14 ms**（解码提速 **~15%**）。

---

### 路线二：Gated Delta Net (GDN) 线性注意力单次写入（零风险、确定性）
- **现状瓶颈**：36 层线性注意力状态矩阵 $S$ 单步对 DRAM 发生 2 次读取和 2 次写入（单步流量 452 MB，Prefill 26 步流量 11.23 GB）。
- **实施方案**：
  - 循环 1（计算 `acc`）：保持计算不变，移除 `S[b] = sv0;` 写回；
  - 循环 2：计算 `sv0 = (S[b] * decay) + k * dval`，执行唯一次 DRAM 写回。
  - 浮点算术表达式 100% 对齐，零精度风险。
- **预期收益**：消除 50% 写回流量（Decode 每步减免 108 MB 写入，Prefill 减免 2.8 GB 写入）。Decode GDN 耗时从 11.8 ms 降至 **6.5 ~ 7.5 ms**（省 **~4.5 ms**）；Prefill GDN 耗时从 150 ms 降至 **~75 ms**。

---

### 路线三：Prefill MoE 多 Token 权重复用普及与 4-Token 扩展
- **现状瓶颈**：首段 26-token Prefill 按 Token 逐个串行执行 GEMV，重复加载 31.5 GB 权重，耗时 1,526 ms。
- **实施方案**：
  - 普及并默认启用既有的 W4 2-Token 权重复用流水线（`AILA_Q4E_EXPERT_MAJOR_PREFILL=1`）；
  - 拓展 4-Token Vector Engine 寄存器广播内核（单个 Subgroup 载入 1 次 NF4 权重，反量化后在寄存器中与 4 个 Token 的输入向量同时累加）。
- **预期收益**：26-token Prefill 纯 MoE 计算耗时从 728 ms 压缩至 **200 ~ 250 ms**（Prefill 总体提速 **~500 ms / ~25%**）。

---

### 路线四：MoE Router 推测退避与 Shared Expert 融合
- **现状瓶颈**：每层强制对下一层 Router 进行 GEMV 预测（47 次额外矩阵乘），而在稳态高命中率下收益微弱；Shared Expert 独立发射多次小内核。
- **实施方案**：
  - 运行时监控缓存命中率：当连续 2 层未发生 Cache Miss 或前序命中率 >90% 时，跳过下一层预测 GEMV；
  - 完善 `AILA_Q4E_MOE_SHARED_FUSED=1`，在 Routed Experts 累加核函数中直接融合就绪的 Shared Expert 输出按 Gate 缩放累加进张量 $y$。
- **预期收益**：Decode 单步节约 **~4.0 - 5.5 ms**，消减 100+ 次内核提交开销。

---

### 路线五：高频专家先验常驻与冷启动消除 (Static Hot-Pinning)
- **现状瓶颈**：3,714 个显存槽位在冷启动阶段全空，前 20 个 Token 产生密集 Cold Miss，导致大量 PCIe 停顿。
- **实施方案**：
  - 根据离线/预热统计的前 3%~5% 核心热专家（每层 18 个核心专家，48 层共 864 个专家，占用 2.19 GB 显存），在模型加载或 Warmup 期常驻固定在槽位 `0..863`，标记为 `pinned` 不予淘汰；
  - 剩余 2,850 个槽位留给动态 LRU 轮转。
- **预期收益**：首请求前 20 个 Token 的初始命中率从 50%~65% 跃升至 **85%~92%**，单请求端到端直接缩短 **~350 - 500 ms**。

---

### 路线六：H6c 对话模板白名单推广 (Anchor 26/28)
- **现状瓶颈**：首段 26-token Prefill 与尾段 4-token Prefill 被动切分，重复执行层遍历与 LM head。
- **实施方案**：
  - 基于已验证的白名单锚点集合 `{26, 28}`，在 Logits 位模式 0 差异的前提下受控启用单次全长 prefill 内捕获中间状态。
- **预期收益**：短对话端到端固定削减 **~245 ms (-4.33%)**。

---

## 3. 规划矩阵与综合收益预期

| 阶段 | 优化内容 | 预期单步 / 请求收益 | 工程复杂度 | 风险与验证 |
|---|---|---|---|---|
| **Phase G1** | HyperConnection Subgroup 协同连续访存 GEMV (HC-A) | Decode 单步省 **~12-14 ms**（提速 **~15%**） | 中等 | Bitwise 对齐，无精度风险 |
| **Phase G2** | GDN 状态单次写入重构与流量腰斩 (GDN-A) | Decode 单步省 **~4.5 ms**，Prefill 省 **~75 ms** | 低 | 纯算术等价，核验 Recurrent State |
| **Phase G3** | 推广 W4 2-Token 复用并拓展至 4-Token (MOE-P1/P2) | 26-token Prefill 省 **~500 ms** | 中等 | 已有 2-token 基石，4-token 扩展 |
| **Phase G4** | Router 动态推测退避与 Shared Expert 融合 | Decode 单步省 **~4-5 ms**，削减 100+ 次内核提交 | 低 | 验证稳态 Token 生成一致性 |
| **Phase G5** | 高频专家常驻与冷启动消除 (CACHE-A) | 首请求提速 **~350-500 ms**，稳态命中率 >95% | 低 | 纯 Cache 初始化策略调整 |
| **Phase G6** | H6c 对话模板白名单推广 (Anchor 26/28) | 短对话端到端固定削减 **~245 ms (-4.33%)** | 极低 | 仅对 0-diff 锚点生效，其他安全回退 |

### 复合预期性能推演
1. **稳态 Decode 生成速度**：从 **~79 - 95 ms / token** (10.5~12.6 tok/s) 降至 **50 - 55 ms / token** (**18.2 ~ 20.0 tok/s，解码吞吐提升 +60% ~ +80%**)。
2. **Prefill 首段耗时**：从 **~2,335 ms** 压缩至 **1,200 - 1,400 ms** (**Prefill 耗时降低 ~40% - 50%**)。
3. **端到端 48-Token Story 生成总耗时**：从目前的 **5,695 ~ 5,829 ms** 大幅跨越进入 **3,300 ~ 3,600 ms** 区间（**整体加速 ~1.6x ~ 1.7x**）。

---

## 4. 推荐优先推进计划

建议按以下顺序开展工程落地：
1. **第一波次（极速交付，确定性收益）**：
   - **Phase G2（GDN 状态单次写入重构）**：改动小、纯算术等价、零精度风险，立即消除 50% GDN 写回带宽（单步省 4.5 ms）。
   - **Phase G1（HyperConnection Subgroup 协同 GEMV）**：切中单步最大性能漏斗（17 ms -> 4 ms），直接找回 12~14 ms。
2. **第二波次（Prefill 与系统开销压缩）**：
   - **Phase G3（W4 2-Token / 4-Token 复用默认启用）**；
   - **Phase G6（H6c 锚点 26/28 白名单落地）**。
3. **第三波次（长序列与多轮稳定性）**：
   - **Phase G4（Router 推测退避与 Shared Expert 融合）**；
   - **Phase G5（静态高频专家 Pinning）**。
