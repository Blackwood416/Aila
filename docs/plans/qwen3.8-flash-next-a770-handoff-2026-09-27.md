# Qwen3.8-Flash-Next (qwen4_exp) on Arc A770 — 研发交接文档 (Handoff)

**交接日期**: 2026-09-29  
**当前状态**: Phase G1–G6 优化里程碑全量闭环，CTest 62/62 100% Passed，端到端 48 tokens 与黄金基线比特精确对齐。  
**目标硬件平台**: Intel Arc A770 16GB GDDR6 (DG2-512, Xe-HPG), Windows 11 25H2, oneAPI 2026.1 (`icx-cl`, SYCL, Level Zero)  

---

## 1. 优化里程碑交付总览 (H0-H5 与 G1-G6 已全量闭环)

按照最新设计规划 [`docs/plans/qwen3.8-flash-next-gpu-optimization-brainstorm-2026-09-28.md`](file:///E:/RiderProjects/Aila/docs/plans/qwen3.8-flash-next-gpu-optimization-brainstorm-2026-09-28.md)，已完成纯 GPU 优化全六个核心阶段落地：

### 1.1 H0–H5 基础交付
- **H0**: Warmup Manifest 覆盖、QSA JIT prime、消除首请求 JIT 停顿。
- **H1**: Ready/Fill/DMA 分离票据与 HostExpertLease 租约并发保护。
- **H2a/H2b/H3**: CPU NF4 专家算子原型、Ordered Rank Merge、有界 Credits 与 Straggler 监控（后续经定量评审已判定 CPU 异构端到端倒挂 53.33%，转向纯 GPU 主线）。
- **H4**: 可替换的 `ExpertReader` 架构，支持 Win32 异步 `OVERLAPPED ReadFile`，规避 mmap 突发缺页停顿。
- **H5**: 纳秒级 PCIe H2D DMA timing 诊断与就绪专家解耦派发。

### 1.2 Phase G2：GDN 状态矩阵单次写入重构 (提交: `b9e6471`, `8ac9568`)
- **显存写回减半**: 将 Decode 与 Prefill 状态矩阵更新由两次写回重构为**单次写入（Single-Write Pass）**：循环 1 仅从 DRAM 读 $S$ 累加标量 `acc`，彻底消除 `S[b] = sv0;` 中间写回；循环 2 唯一次写回更新后的 $S$；
- **纯算术等价**: 浮点算术表达式 100% 对齐，单步消除 108 MB 显存写入流量（Prefill 消除 2.8 GB 写入）；
- **CPU Reference 比对验证**: 注入独立的单步 GDN Decode CPU Reference，比对 GPU 回写状态矩阵与 CPU 参考输出（`max_diff <= 1.12e-8`，单精度极限误差）。

### 1.3 Phase G1：HyperConnection Subgroup 协同连续访存 GEMV (提交: `856ed6a`, `8ac9568`)
- **消除 SLM 32-Bank 冲突**: 严格遵守 `AGENTS.md` 铁律，彻底摒弃 SLM 缓存输入；
- **Subgroup-Cooperative 映射**: 1 个 Subgroup（16 线程）负责 1 个输出行，Workgroup 包含 16 个 Subgroup 计算 16 行；消除所有 SLM 障碍与组内 Barrier，使用寄存器级 `sycl::reduce_over_group(sg, ...)` 规约；
- **自适应地址对齐保护**: 引入 `is_aligned` 动态检测，在对齐时全速运行 128-bit 矢量加载与 FMA，在奇数行或非对称退化维度下优雅走 Subgroup-Cooperative 标量步进，消除 GPU 内存未对齐崩溃隐患。

### 1.4 Phase G3：Prefill MoE 多 Token 权重复用普及与 4-Token 扩展 (提交: `7977c14`, `2debba5`)
- **默认启用 S4 Expert-Major 流水线**: 淘汰逐 Token 串行循环，默认以专家为中心批处理发射；
- **扩展 4-Token Vector Engine 寄存器广播内核**: 在单个 Subgroup 内载入 1 次 NF4 权重，反量化后在寄存器中与 4 个 Token 的输入向量执行 FMA 累加，按 Quads + Pairs + Singles 阶梯分配；
- **解耦事件派发与消除别名指针**: 消除 `cur_dep = &last_ev` 自引用指针别名与强制串行化，Quads、Pairs 与 Singles 独立并行发射，终端统一合流事件，消除动态堆分配与驱动开销。

### 1.5 Phase G6：H6c 对话模板受控白名单推广 (提交: `0f9e1a1`, `2debba5`)
- **受控白名单保护机制**: 在 `supports_interior_checkpoint` 中显式限定必须在显式 opt-in 开关 `AILA_Q4E_INLINE_CHECKPOINT=1` 下才允许触发内部快照，且严格限制在已验证 Logits 位模式 0 差异的 `{26, 28}` 白名单内；
- **金标双轨防御**: 默认状态下严格回退至分段 Prefill，保持 100% 官方黄金基线对齐；显式启用加速模式时消除两段 Prefill 的重复层遍历与 LM head 计算，单次 Prefill 耗时缩减超 300 ms。

### 1.6 Phase G4：MoE Router 动态推测退避与 Shared Expert 融合
- **Predictive GEMV Gating**: 运行时实时监控缓存命中状态，在连续未发生 Cache Miss（`consecutive_no_miss_layers_ >= 2`）或命中率稳定（>90% 且 `consecutive_no_miss_layers_ > 0`）的稳态下，动态跳过下一层预测 GEMV `[512, 2560]` 与 topk 计算；仅在冷启动或任一层发生 Miss 时重置计数并重新唤醒推测；单步消除 47 次无消费者 GEMV 与 123 MB 显存带宽消耗；
- **默认启用 Shared Expert 融合 (`AILA_Q4E_MOE_SHARED_FUSED=1`)**: 在 Routed Experts 累加核函数中直接融合就绪的 Shared Expert 输出按 Gate 缩放累加进张量 $y$，彻底消除独立的 `add_shared_expert_to_y` 内核发射开销。

### 1.7 Phase G5：高频核心专家先验常驻 Static Hot-Pinning 与冷启动消除
- **先验核心热专家识别**: 基于多场景（故事、长文、代码、中文对话）真实端到端推理路由轨迹（统计超 12 万次激活），提取每层最核心的 18 个热专家（48 层共 864 个专家，占用 2,088.3 MB 显存）构建静态表 `kQwen4ExpDefaultHotExperts[48][18]`；
- **常驻槽位与 LRU 绝对豁免**: 模型初始化加载期通过批处理 DMA 直接将核心专家注入 GPU 显存槽位 `0..863`，标记为 `slot_pinned_ = true` 并从 LRU 淘汰链表中剥离；受害槽位筛选算法强制增加 `!slot_pinned_[s]` 绝对豁免过滤，确保常驻专家永不被淘汰；
- **安全容量自适应**: 严格保障保留至少 `num_experts_per_layer + 16` 个动态槽位给动态 LRU 轮转（A770 16GB 上动态槽位达 2,850 个）；小容量单元测试下自动安全跳过，零副作用。

---

## 2. 严苛工程验证记录 (CTest & 端到端真机)

所有测试均在 Intel Arc A770 真实物理设备上通过单任务串行执行完成：

1. **编译构建**: `pwsh -NoProfile -File build.ps1 -BuildDir build-refactor` **0 错误、0 警告**。
2. **全量关键 CTest 单元测试**:
   - `ctest --test-dir build-refactor --output-on-failure`: **62/62 100% Passed** (耗时 448.37 秒，零错误零崩溃)；
   - `AilaQwen4ExpKernelTests`: 100% Passed (4-Token MoE 算子与 CPU 参考实现 0 mismatch)；
   - `AilaQwen4ExpModuleTests`: 100% Passed (含新增静态 Pinning、LRU 豁免、白名单拦截机制与权重对齐)；
   - `AilaQwen4ExpGenerationTests`: 100% Passed (26+4 skip-logits 与回滚 248,320 BF16 logits 100% 相同；Anchor 26/28 内部快照位级一致；两轮复用与编辑回滚 Token IDs 100% 吻合)；
   - `AilaBuildIdentityCMakeTests`: 100% Passed。
3. **短上下文黄金基线端到端验证**:
   - 命令：`.\build-refactor\Aila.exe -m C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2 --max-seq 512 --greedy --thinking-budget 0 --max-tokens 48 --messages-json perf/qwen4exp/prompts/story.json`
   - **默认生产模式 (Overlapped ReadFile, G1-G5 全量启用, G6 安全金标防御)**:
     - 输出文本：
       `The horizon ignites with a soft, amber glow, piercing the lingering indigo of the pre-dawn sky. Gentle waves ripple beneath the emerging light, their crests catching the first golden rays like scattered diamonds. As the sun breaches the`
     - **48 tokens 与黄金基线保持 100% 比特精确对齐**。
     - **稳态性能**: 864 个核心专家常驻，初始命中率高达 85.4%，稳态单步耗时稳定进入 79~89 ms 区间，DMA wait 压缩至 0.00 ms。

---

## 3. 研发交接与后续建议

纯 GPU 优化规划 Phase G1 ~ G6 现已全量交付并闭环：
- Phase G1 (HC Subgroup GEMV)
- Phase G2 (GDN Single-Write Pass)
- Phase G3 (MoE 4-Token & 2-Token Reuse)
- Phase G4 (Router Predictive Gating & Fused Shared Expert)
- Phase G5 (Static Core Expert Hot-Pinning)
- Phase G6 (H6c Interior Checkpoint Whitelist)

### 后续建议方向：
1. **多流连续批处理 (Continuous Batching P7 集成)**：将 Qwen4Exp 架构接入上层 `AilaWorker` / `AilaServer` 连续批处理流水线；
2. **长序列扩展与 QSA 优化**：在更长上下文（>2k, 8k）下针对 QSA indexer 的注意力稀疏块压缩与检索展开专项微调。

---

## 4. 绝对红线与系统约束 (Strict Rules)

1. **单任务串行队列执行**: Windows WDDM 驱动对 Arc A770 并发显存占用极其敏感。**严禁并发运行多个构建、测试或推理进程**，必须等前一个进程彻底退出并释放显存后再启动下一个。
2. **热路径零堆分配 (Zero Allocations on Hot Path)**: 禁止在 `ForwardImpl`、算子调度或 decode 循环中调用 `new`、`malloc`、`std::vector::resize` 等动态堆操作。
3. **比特精确对齐黄金基线**: 任何改动后必须运行真实模型验证，默认生产模式下生成的 48 个 tokens 必须与黄金基线 **100% 比特精确对齐**。
4. **规范 Git 提交**: 分阶段推进，每个独立优化点通过 CTest 与端到端验证后方可提交。
