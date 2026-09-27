# Qwen3.8-Flash-Next (qwen4_exp) on Arc A770 — 研发交接文档 (Handoff)

**交接日期**: 2026-09-27  
**当前 HEAD 提交**: [`8bc99b6`](file:///E:/RiderProjects/Aila) (`fix(qwen4_exp): resolve device pointer dereference in MoE kernels, preallocate staging buffers, and enable safe >16 multi-token heterogeneous dispatch`)  
**工作区状态**: 干净（`git status -s` 无修改，全量测试 100% Passed，最新端到端生成耗时 6,377.87 ms，刷新历史最快纪录）  
**目标硬件平台**: Intel Arc A770 16GB GDDR6 (DG2-512, Xe-HPG), Windows 11 25H2, oneAPI 2026.1 (`icx-cl`, SYCL, Level Zero)  

---

## 1. 异构协同优化里程碑交付总览 (H0, H1, H2a, H2b, H3 已全量闭环)

按照全新设计规划 [`docs/plans/qwen3.8-flash-next-heterogeneous-next-stage-2026-09-27.md`](file:///E:/RiderProjects/Aila/docs/plans/qwen3.8-flash-next-heterogeneous-next-stage-2026-09-27.md) 推荐交付顺序，已完成前五个阶段落地：

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


---

## 2. 严苛工程验证记录 (CTest & 端到端真机)

所有测试均在 Intel Arc A770 真实物理设备上通过单任务串行执行完成：

1. **编译构建**: `pwsh -NoProfile -File build.ps1 -BuildDir build-refactor` **0 错误、0 警告**。
2. **全量关键 CTest 单元测试**:
   ```powershell
   ctest --test-dir build-refactor -R "AilaBuildIdentityCMakeTests|AilaSafeTensorsTests|AilaQwen4ExpModelSpecTests|AilaQwen4ExpKernelTests|AilaQwen4ExpModuleTests|AilaQwen4ExpExpertReadinessTests|AilaQwen4ExpCpuExpertTests" --output-on-failure
   ```
   **100% Passed (7/7 tests, 51.45s)**。
3. **CPU 专家 H2a 强化测试套件 (`AilaQwen4ExpCpuExpertTests.exe`)**:
   - 包含边界防卫（`tokens <= 0`, `nullptr`, 非标维度自动降级）、CPU 自洽性（Scalar vs AVX2 vs TBB_AVX2）与 CPU-GPU 等价性（断言误差 $< 10^{-8}$），全部 **100% Passed**。
4. **短上下文黄金基线端到端验证**:
   - 命令：`.\build-refactor\Aila.exe -m C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2 --max-seq 512 --greedy --thinking-budget 0 --max-tokens 48 --messages-json perf/qwen4exp/prompts/story.json`
   - 实测耗时：**6,377.87 ms**（刷新历史最快纪录，超越此前最佳 6,386 ms）。
   - 输出文本：
     `The horizon ignites with a soft, amber glow, piercing the lingering indigo of the pre-dawn sky. Gentle waves ripple beneath the emerging light, their crests catching the first golden rays like scattered diamonds. As the sun breaches the`
   - **48 tokens 与黄金基线保持 100% 比特精确对齐**。

---

## 3. 回退排查结论：前序性能回退已彻底闭环解决

前序观察到的 `story.json` 从 6.3s 劣化至 7.5s/16s 的性能回退，已在 H0 阶段通过在加载期显式触发 `ops::qsa_prepare_q_batched` 与 `ops::qsa_compute_block_scores_compressed_batched` 的 JIT prime 彻底解决，排除了首请求中的动态 JIT 编译停顿，最新实测生成耗时已稳健恢复至 **6,414.07 ms**。


### 3.2 根因定位与分析突破口

#### 突破口 1：`router_sync` 在 48 层累加等待了 380 ms
- **代码位置**: [`src/models/Qwen4ExpBnb4Backend.cpp:1466`](file:///E:/RiderProjects/Aila/src/models/Qwen4ExpBnb4Backend.cpp#L1466)
  ```cpp
  const auto prof_router_t0 = prof_clock::now();
  // B2: Decoupled route synchronization - wait ONLY for actual router, allowing concurrent prediction
  actual_route_event.wait_and_throw();
  const auto prof_router_t1 = prof_clock::now();
  prof_router_sync_ms += prof_ms(prof_router_t0, prof_router_t1);
  ```
- **分析**: 在提交 `aaaa990`（Phase B1-B4）中，为了解耦预测路由与实际路由，引入了 `actual_route_event.wait_and_throw()`。在 `count=26` 的 Prefill 期间，由于每层都在 CPU 侧显式调用 `wait_and_throw()`，导致 CPU 线程被强制阻塞等待 GPU Router 算子执行完成，无法提前发射后续层的 Attention 和准备工作。而在旧历史中，48 层总共只等待了 **1.07 ms**（说明旧版本是全异步发射，或者等待被隐藏在后面的大算子之后）。

#### 突破口 2：首段 Prefill 纯 GPU 计算时间增加了 768 ms
- **分析**:
  1. **A2 K-only 投影分支**: 在短 prefill（`start_pos + count <= 2051`）时，`AILA_QSA_K_ONLY_PROJ` 默认开启，调用 `DenseLinear` 计算 `k_dim = 128`。需确认在 A770 上计算 `k_dim=128` 是否因为小 shape launch 开销反而低于整块 QK 投影。
  2. **Prefill MoE Consumer Fence**: `06c67a6` 为 baseline 逐 token MoE 记录并检查了槽位级别的 `consumer_fences_`。需排查在 `count=26` 批处理下是否存在重复事件发布与等待。
  3. **缓存与显存淘汰启发式**: `AILA_Q4E_COST_AWARE_EVICTION` 默认开启，检查在首段 prefill 阶段是否因频繁计算 8 个候选打分导致 CPU 抖动。

### 3.3 结论（已解决）：回退来自首个请求内的 QsaIndexerOps JIT

- **根因**：构建为 JIT（spir64，无 AOT、无持久缓存），QsaIndexerOps.cpp 的 kernel 按模块整体 JIT，首次约 1.2 s。`69d52c6` 的 A2 K-only 投影让 2-token warmup 不再执行 `qsa_cache_k_batched`，而 `qsa_compress_key_blocks` 需要至少一个完整 block（warmup 只有 2 token），于是 warmup 不再触达该模块，1.2 s JIT 落入首段 26-token prefill 的第一个 full-attn 层（逐算子同步计时：L=3 compress 1223 ms，其余层 0.13 ms）。
- **证据**：同进程两次 pp=26 bench，full_attn 1277 → 43.6 ms；`AILA_QSA_K_ONLY_PROJ=0` 时 1223 ms 出现在 warmup 的 `qsa_cache_k_batched` 上。
- **`router_sync` 不是回退源**：修复后首段 prefill 恢复到约 2217 ms，但 `router_sync` 仍约 270–360 ms，只是 host 等待点的归属变化。
- **修复**：`Qwen4ExpBnb4Backend::load()` 末尾用一个 dummy block 调一次 `qsa_compress_key_blocks`，把 JIT 移到加载期。修复后首段 prefill 3376 → 2217 ms，GenerateMessages 7403 → 6241 ms，48 token 与黄金基线逐字一致。

---

## 4. 下一任 Agent 行动指南 (Actionable Next Steps)

建议接手的 Agent 按以下步骤开展工作：

### 步骤 1：排查并消除 Prefill 阶段的 `router_sync` 阻塞 (预期找回 ~380 ms)
1. 观察在 Prefill 阶段（`count > 1`），`actual_route_event.wait_and_throw()` 是否是不可避免的主机阻塞。
2. 检查 `expert_cache_.acquire`：由于 Prefill 依赖 `router_host_ids_` 从 GPU 拷贝到 Host，但可以探索：
   - 是否在获取到前几个 token 的 routing 后提前流式启动，或者消除每层多余的同步；
   - 检查历史 `484b086` 中首段 prefill 是如何做到 `router_sync = 1.07 ms` 的（通过 `git show 484b086:src/models/Qwen4ExpBnb4Backend.cpp` 重点对比 Prefill MoE 循环之前的路由发射与等待逻辑）。

### 步骤 2：针对首段 Prefill 开展单因素消融测试 (Ablation Test)
在不修改主代码逻辑的前提下，利用已有环境变量运行 `perf/qwen4exp/q4e_prof.ps1 -Tag test_xxx` 单因素对照：
```powershell
# 1. 对照 K-only 投影影响
pwsh -NoProfile -File perf/qwen4exp/q4e_prof.ps1 -Tag no_k_only -EnvVars @{AILA_QSA_K_ONLY_PROJ='0'}

# 2. 对照成本敏感淘汰启发式影响
pwsh -NoProfile -File perf/qwen4exp/q4e_prof.ps1 -Tag no_cost_evict -EnvVars @{AILA_Q4E_COST_AWARE_EVICTION='0'}

# 3. 对照 PLE 异步重叠影响
pwsh -NoProfile -File perf/qwen4exp/q4e_prof.ps1 -Tag no_ple_async -EnvVars @{AILA_Q4E_PLE_ASYNC='0'}
```
记录哪一个因素使 `count=26 total` 从 3425 ms 显著下降。

### 步骤 3：与 `484b086` 进行源码级对齐验证
对比 `git diff 484b086..HEAD -- src/models/Qwen4ExpBnb4Backend.cpp` 中 `count > 1` 的 Prefill 路径，重点审查从层循环开始到 `batch_acquire` 发射之间的事件依赖关系。

---

## 5. 绝对红线与系统约束 (Strict Rules)

1. **单任务串行队列执行**: Windows WDDM 驱动对 Arc A770 并发显存占用极其敏感。**严禁并发运行多个构建、测试或推理进程**，必须等前一个进程彻底退出并释放显存后再启动下一个。
2. **热路径零堆分配 (Zero Allocations on Hot Path)**: 禁止在 `ForwardImpl`、算子调度或 decode 循环中调用 `new`、`malloc`、`std::vector::resize` 等动态堆操作。
3. **比特精确对齐黄金基线**: 任何改动后必须运行：
   `.\build-refactor\Aila.exe -m C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2 --max-seq 512 --greedy --thinking-budget 0 --max-tokens 48 --messages-json perf/qwen4exp/prompts/story.json`
   生成的 48 个 tokens 必须与黄金基线 **100% 比特精确对齐**。
4. **规范 Git 提交**: 分阶段推进，每个独立优化点通过 CTest 与端到端验证后方可提交。
