# Qwen3.8-Flash-Next (qwen4_exp) on Arc A770 — 研发交接文档 (Handoff)

> **H6c 后续更新：** 默认关闭的短文本内部快照原型已验证 story 中位计时约降低 4.33%，并通过黄金 token、多轮复用和实际回滚对照。通用锚点数值一致性仍未解决，不能默认启用；详见 [H6c 实验与下一步](qwen3.8-flash-next-h6c-results-2026-09-28.md)。以下“H6c 尚未实施”是前一轮 H6b 交接时的历史状态。

> **2026-09-28 H6 更新：** H6a/b 已完成，H6a 提交 `ac5e7df`。默认聊天保留 26+4，非末段跳过 logits，token staging 与快照目标预分配；显式 one-shot 维持 30-token 单段。完整 logits 位模式、真实两轮复用/回滚及两种策略的 48-token ID 对照通过。最新代码和验收以 [H6 记录](qwen3.8-flash-next-h6-results-2026-09-28.md) 为准，本文下方 H0–H5 数据及原 H6 待办是接手时记录。H6c 尚未实施。

**交接日期**: 2026-09-28  
**当前 HEAD 提交**: [`d39c164`](file:///E:/RiderProjects/Aila) (`fix(qwen4_exp): resolve unnested float absmax reader crash, add >16 decoupled task dispatch and accurate DMA timing`)  
**工作区状态**: 干净（`git status -s` 无修改，全量测试 100% Passed，最新 Overlapped ReadFile 模式端到端生成耗时 **5,829.39 ms**，大幅刷新历史最快纪录）  
**目标硬件平台**: Intel Arc A770 16GB GDDR6 (DG2-512, Xe-HPG), Windows 11 25H2, oneAPI 2026.1 (`icx-cl`, SYCL, Level Zero)  

---

## 1. 异构协同优化里程碑交付总览 (H0, H1, H2a, H2b, H3, H4, H5 已全量闭环)

按照全新设计规划 [`docs/plans/qwen3.8-flash-next-heterogeneous-next-stage-2026-09-27.md`](file:///E:/RiderProjects/Aila/docs/plans/qwen3.8-flash-next-heterogeneous-next-stage-2026-09-27.md) 推荐交付顺序，已完成前七个阶段落地：

### 1.1 H0：Warmup Dispatch 覆盖与 JIT 首请求停顿排除 (提交: `1e1307f`, `f24dd34`)
- **预热清单落地**: 显式建立覆盖 2-token prefill (GEMM, score buffer)、1-token decode (GEMV, decode attention, MoE) 及状态安全 reset 的完整预热清单；在加载期完成 QSA 算子 JIT prime，彻底消除首请求内的动态编译停顿。

### 1.2 H1：专家就绪票据、CPU 租约与 GPU-Ready 规划 (提交: `a52e439`, `e26fbc2`)
- **解耦票据**: 实现 `prepare_host` (`FillTicket`) 与 `ensure_gpu` (`GpuExpertTicket`) 异步流水线解耦接口。
- **并发租约与淘汰保护**: `HostExpertLease` 引入 `host_list_mutex_` 与代次校验，实测 4/4 饱和池下严格保护被租约占用的槽位不被驱逐。
- **子集规划**: `Qwen4ExpExpertExecutor` 实现 GPU-ready 子集先发与 pending 异步绑定（`ensure_gpu_pending`）。

### 1.3 H2a：CPU NF4 专家计算原型与微基准 (提交: `264995b`, `49e667f`)
- **高性能 CPU 内核**: 在 `src/ops/Qwen4ExpCpuExpertOps.hpp/.cpp` 实现针对 $H=2560, I=640$ 的 AVX2/oneTBB 浮点内核原型，包含入口哨兵、非标维度安全降级与加权累加。
- **精度对齐**: 与 GPU 核函数比对，中间层与输出绝对误差均小于 $10^{-8}$（远超 $< 10^{-3}$ 要求）。
- **可行性微基准**: $m=1$ CPU 耗时仅 **0.2786 ms**（快于 PCIe H2D 2.53 MB 槽位传输时间 0.39 ms），证实小批次 CPU 异构计算具备极佳收益潜力。

### 1.4 H2b：Host-Ready Miss 专家分流与有序降维合并 (提交: `7ea0630`, `8bc99b6`)
- **异构并发重叠**: 实现 `x_dev -> x_host` 异步流式拷贝；GPU 先行入队发射 GPU-ready 专家，CPU 在主机侧并行计算 Host-Ready 专家，消除主机端阻塞等待，实现真正的物理硬件级并发。
- **Ordered Rank Merge**: 在 GPU 端新增 `down_to_rank_f32` 与有序累加核函数，CPU 结果经异步 H2D 写入对应 rank 槽位后与 GPU 结果精确累加，实测与熔断对比误差严格为 0（max_diff = 0）。
- **设备指针核函数重构**: 针对多 Token（$m=2..4, \text{top\_k}=10 \Rightarrow 20..40$ 任务）支持直接接收设备 USM 指针，杜绝主机解引用设备指针崩溃（0xC0000005），彻底消除 Decode 热路径上的动态堆分配。

### 1.5 H3：有界 CPU Credits、掉队者保护与动态节流 (提交: `7ea0630`, `8bc99b6`)
- **有界准入配额**: 严格限制单层 CPU 派发上限（`cpu_credit` 默认为 1..2），防止超额派发 CPU 租约耗尽主机内存池或引发线程争用。
- **精准掉队者监控**: 在 CPU 计算完成瞬间采样 GPU `down_ev` 的执行状态，若 CPU 计算超时落后于 GPU，触发惩罚机制（动态下调 credit 并连续抑制）；在连续准时完成后自动恢复配额。

### 1.6 H4：可替换的 ExpertReader 架构 (提交: `56e5255`, `d39c164`)
- **可替换架构**: 在 `src/models/Qwen4ExpExpertReader.hpp/.cpp` 抽象出 `ExpertReader` 纯虚接口与工厂方法，支持 `mmap`（默认）与 `readfile` 两种实现。
- **基于 Win32 Overlapped ReadFile 的异步重叠读取**:
  - 利用预打开的 safetensors shard 文件句柄，结合 Win32 异步 `OVERLAPPED` I/O，将 gate/up packed、gate/up codes、down packed、down codes 四段数据直接读入 Host Staging Slot；
  - 完善处理了取消恢复机制（`reset_cancelled`）、`slot_gen` 代次校验与短读校验；修复了针对非 compact unnested float32 absmax 权重的解析分支；
  - 实测真机推理中，相比 mmap 直接读取避免了突发缺页停顿，端到端耗时大幅降低至 **5,829.39 ms**。

### 1.7 H5：PCIe H2D 6.5 GB/s 传输细分诊断与局部 DMA 解耦触发 (提交: `56e5255`, `d39c164`)
- **细粒度 DMA Timing 监控**: 在 `Qwen4ExpExpertCache` 中建立以纳秒为单位的高精度指标捕获，包括 `last_host_submit_ns`、`last_copy_event_ns`、`total_dma_time_ns`、`total_consumer_gap_ns` 与 `total_dma_bytes`，精准量化硬件 Blitter 传输耗时；增加 `new_dma_batch_pending_` 守卫，杜绝全 Hit 层的虚假时间漂移累加。
- **局部解耦分派**: 在 `Qwen4ExpExpertExecutor` 中解耦 GPU ready 专家与 pending 专家的发射窗口；即使存在正在 DMA 传输的未就绪专家，已在显存中的 ready 专家立即可先行发射进入计算，避免整批专家串行等待最慢项。

---

## 2. 严苛工程验证记录 (CTest & 端到端真机)

所有测试均在 Intel Arc A770 真实物理设备上通过单任务串行执行完成：

1. **编译构建**: `pwsh -NoProfile -File build.ps1 -BuildDir build-refactor` **0 错误、0 警告**。
2. **全量关键 CTest 单元测试**:
   ```powershell
   ctest --test-dir build-refactor -R "AilaBuildIdentityCMakeTests|AilaSafeTensorsTests|AilaQwen4ExpModelSpecTests|AilaQwen4ExpKernelTests|AilaQwen4ExpModuleTests|AilaQwen4ExpExpertReadinessTests|AilaQwen4ExpCpuExpertTests|AilaQwen4ExpExpertReaderTests|AilaQwen4ExpDmaTimingTests" --output-on-failure
   ```
   **100% Passed (9/9 tests, 62.87s)**。
3. **短上下文黄金基线端到端验证**:
   - 命令：`.\build-refactor\Aila.exe -m C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2 --max-seq 512 --greedy --thinking-budget 0 --max-tokens 48 --messages-json perf/qwen4exp/prompts/story.json`
   - **默认 mmap 模式**:
     - 实测耗时：**6,362.77 ms**。
     - 输出文本：
       `The horizon ignites with a soft, amber glow, piercing the lingering indigo of the pre-dawn sky. Gentle waves ripple beneath the emerging light, their crests catching the first golden rays like scattered diamonds. As the sun breaches the`
     - **48 tokens 与黄金基线保持 100% 比特精确对齐**。
   - **Overlapped ReadFile 模式 (`$env:AILA_Q4E_EXPERT_READER="readfile"`)**:
     - 实测耗时：**5,829.39 ms**（**大幅刷新全历史最快纪录**，较此前 6,377 ms 减少 548 ms，提速 **~8.6%**）。
     - 输出文本：
       `The horizon ignites with a soft, amber glow, piercing the lingering indigo of the pre-dawn sky. Gentle waves ripple beneath the emerging light, their crests catching the first golden rays like scattered diamonds. As the sun breaches the`
     - **48 tokens 与黄金基线保持 100% 比特精确对齐**。

---

## 3. 下一步行动指南 (Actionable Next Steps)

依据 [`docs/plans/qwen3.8-flash-next-heterogeneous-next-stage-2026-09-27.md`](file:///E:/RiderProjects/Aila/docs/plans/qwen3.8-flash-next-heterogeneous-next-stage-2026-09-27.md) §11，后续待推进阶段为：

### H6 阶段（26+4 Prefill 切分优化与显式 One-Shot 模式）
1. **H6a（显式 One-Shot 模式）**:
   - 增加请求级 `snapshot_policy=none|reusable_prefix` 与 `need_logits` 参数；
   - 对承诺无需多轮增量复用的一次性生成请求跳过 anchor 拆分，避免在稳定前缀处（26 tokens）被动拆分为两段 prefill。
2. **H6b（非末段 Skip-Logits 与预分配复用）**:
   - 在必须保留 anchor 拆分的多轮对话模式下，前置非末段 forward 跳过冗余的 LM head / logits 算子计算；
   - 预分配 snapshot 目标与 token input staging 内存，消除前向辅助缓冲区的每段动态分配。

---

## 4. 绝对红线与系统约束 (Strict Rules)

1. **单任务串行队列执行**: Windows WDDM 驱动对 Arc A770 并发显存占用极其敏感。**严禁并发运行多个构建、测试或推理进程**，必须等前一个进程彻底退出并释放显存后再启动下一个。
2. **热路径零堆分配 (Zero Allocations on Hot Path)**: 禁止在 `ForwardImpl`、算子调度或 decode 循环中调用 `new`、`malloc`、`std::vector::resize` 等动态堆操作。
3. **比特精确对齐黄金基线**: 任何改动后必须运行真实模型验证，生成的 48 个 tokens 必须与黄金基线 **100% 比特精确对齐**。
4. **规范 Git 提交**: 分阶段推进，每个独立优化点通过 CTest 与端到端验证后方可提交。
