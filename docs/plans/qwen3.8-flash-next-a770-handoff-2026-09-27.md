# Qwen3.8-Flash-Next (qwen4_exp) on Arc A770 — 研发交接文档 (Handoff)

**交接日期**: 2026-09-29  
**当前 HEAD 提交**: [`8ac9568`](file:///E:/RiderProjects/Aila) (`perf(hc,gdn): add pointer alignment guards and CPU reference parity tests for HC and GDN`)  
**工作区状态**: 干净（`git status -s` 无修改，全量测试 100% Passed，最新端到端生成耗时 5,889.32 ms / 6,323.27 ms，解码单步耗时稳定在 77.6 ms，吞吐达 12.89 tok/s）  
**目标硬件平台**: Intel Arc A770 16GB GDDR6 (DG2-512, Xe-HPG), Windows 11 25H2, oneAPI 2026.1 (`icx-cl`, SYCL, Level Zero)  

---

## 1. 优化里程碑交付总览 (H0-H5 与 G1-G2 已全量闭环)

按照最新设计规划 [`docs/plans/qwen3.8-flash-next-gpu-optimization-brainstorm-2026-09-28.md`](file:///E:/RiderProjects/Aila/docs/plans/qwen3.8-flash-next-gpu-optimization-brainstorm-2026-09-28.md)，已完成纯 GPU 优化前两个核心阶段落地：

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

---

## 2. 严苛工程验证记录 (CTest & 端到端真机)

所有测试均在 Intel Arc A770 真实物理设备上通过单任务串行执行完成：

1. **编译构建**: `pwsh -NoProfile -File build.ps1 -BuildDir build-refactor` **0 错误、0 警告**。
2. **全量关键 CTest 单元测试**:
   ```powershell
   ctest --test-dir build-refactor -R "AilaBuildIdentityCMakeTests|AilaSafeTensorsTests|AilaQwen4ExpModelSpecTests|AilaQwen4ExpKernelTests|AilaQwen4ExpModuleTests|AilaQwen4ExpExpertReadinessTests|AilaQwen4ExpCpuExpertTests|AilaQwen4ExpExpertReaderTests|AilaQwen4ExpDmaTimingTests" --output-on-failure
   ```
   **100% Passed (14/14 tests)**。
   - `AilaQwen4ExpKernelTests`:
     - `[PASS] GDN single-write state parity (decode) (max_diff=1.11759e-08 <= 0.0002)`
     - `[PASS] GDN single-write out parity (decode) (max_diff=0.000976562 <= 0.02)`
     - `[PASS] hc_decode_down_inject act (inject g=2 d=256 lr=45) (max_diff=0 <= 0.01)`
     - `[PASS] hc_decode_down_inject act (inject g=3 d=31 lr=35) (max_diff=0 <= 0.01)`
3. **短上下文黄金基线端到端验证**:
   - 命令：`.\build-refactor\Aila.exe -m C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2 --max-seq 512 --greedy --thinking-budget 0 --max-tokens 48 --messages-json perf/qwen4exp/prompts/story.json`
   - **默认 mmap 模式**:
     - 实测耗时：**6,323.27 ms**。
     - 输出文本：
       `The horizon ignites with a soft, amber glow, piercing the lingering indigo of the pre-dawn sky. Gentle waves ripple beneath the emerging light, their crests catching the first golden rays like scattered diamonds. As the sun breaches the`
     - **48 tokens 与黄金基线保持 100% 比特精确对齐**。
   - **Overlapped ReadFile 模式 (`$env:AILA_Q4E_EXPERT_READER="readfile"`)**:
     - 实测耗时：**5,889.32 ms**。
     - 平均单步 Decode 耗时：**77.6 ms**（12.89 tok/s）。
     - 输出文本与黄金基线保持 **100% 比特精确对齐**。

---

## 3. 下一步行动指南 (Actionable Next Steps)

依据 [`docs/plans/qwen3.8-flash-next-gpu-optimization-brainstorm-2026-09-28.md`](file:///E:/RiderProjects/Aila/docs/plans/qwen3.8-flash-next-gpu-optimization-brainstorm-2026-09-28.md) 推荐实施矩阵，后续推荐阶段为：

1. **Phase G3（Prefill MoE 多 Token 权重复用普及与 4-Token 扩展）**:
   - 默认启用并推广已验证的 W4 2-Token 权重复用流水线（`AILA_Q4E_EXPERT_MAJOR_PREFILL=1`）；
   - 拓展 4-Token Vector Engine 寄存器广播内核（单个 Subgroup 载入 1 次 NF4 权重，反量化后在寄存器中与 4 个 Token 的输入向量同时累加），预期将 26-token Prefill 耗时缩减 ~500 ms。
2. **Phase G6（H6c 对话模板白名单推广 Anchor 26/28）**:
   - 在已验证 Logits 位模式 0 差异的前提下，在系统/用户边界（Anchor 26/28）启用单次全长 prefill 内捕获中间状态，固定节约 ~245 ms。
3. **Phase G4（Router 动态推测退避与 Shared Expert 融合）**:
   - 在高命中率稳态下动态跳过下一层预测 GEMV，削减 100+ 次内核提交。

---

## 4. 绝对红线与系统约束 (Strict Rules)

1. **单任务串行队列执行**: Windows WDDM 驱动对 Arc A770 并发显存占用极其敏感。**严禁并发运行多个构建、测试或推理进程**，必须等前一个进程彻底退出并释放显存后再启动下一个。
2. **热路径零堆分配 (Zero Allocations on Hot Path)**: 禁止在 `ForwardImpl`、算子调度或 decode 循环中调用 `new`、`malloc`、`std::vector::resize` 等动态堆操作。
3. **比特精确对齐黄金基线**: 任何改动后必须运行真实模型验证，生成的 48 个 tokens 必须与黄金基线 **100% 比特精确对齐**。
4. **规范 Git 提交**: 分阶段推进，每个独立优化点通过 CTest 与端到端验证后方可提交。
