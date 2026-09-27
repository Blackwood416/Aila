# Qwen3.8-Flash-Next (qwen4_exp) on Arc A770 — 研发交接文档 (Handoff)

**交接日期**: 2026-09-27  
**当前 HEAD 提交**: [`cd92eae`](file:///E:/RiderProjects/Aila) (`fix(qsa): resolve static multi-q config, complete R4b/R4c/R4d test coverage and verify >2048 sparse prefill`)  
**工作区状态**: 干净（`git status -s` 无修改，所有进展均已提交并闭环）  
**目标硬件平台**: Intel Arc A770 16GB GDDR6 (DG2-512, Xe-HPG), Windows 11 25H2, oneAPI 2026.1 (`icx-cl`, SYCL, Level Zero)  

---

## 1. 当前交付成果总览 (P0, P1, P2 已全量闭环)

基于 [`docs/plans/qwen3.8-flash-next-reference-brainstorm-2026-09-27.md`](file:///E:/RiderProjects/Aila/docs/plans/qwen3.8-flash-next-reference-brainstorm-2026-09-27.md) 规划的推荐交付顺序，前三阶段已全部完成高质量落地并提交：

### 1.1 P0：PLE 确定性流水线解耦与 `PleTicket` (提交: `4f3ad79`, `4bdf45d`)
- **流水线解耦**: 在 Token IDs 获准执行后预分配 `PleTicket` 双缓冲管线；主机在发射 Layer 0 时提前并发执行 hash/gather/dequant（利用 CPU 任务池异步提交 H2D 到独立 staging slot），Layer 1 仅消费就绪 ticket 依赖，彻底消除 Layer 1 的 exposed wait。
- **热路径零堆分配**: 将 `compute_ngram_ids` 重构为基于 `last_eos` 的单趟 $O(1)$ 空间状态机，彻底消除历史临时 `std::vector` 堆分配；纯 prepare 与 commit 分离，保障 Cancel/Retry 下 generation 阻断的异常安全。

### 1.2 P1：层内相邻 GR combine + norm 融合 (提交: `4f3ad79`)
- **内核级算子融合**: Attention 产生输出后暂存 `block_output + injection`，在下一次 GR read 时合并写回并一体化计算下一个 group RMSNorm（`hc_inject_add_and_grouped_rms_norm`）。
- **精度对齐**: 消除 48 个层内边界的独立内核发射与显存往返，严格保留 BF16 舍入契约与数值精度对齐。

### 1.3 P2：QSA Batch Arena + 共享 KV 多 Q Head 并行 + 补集选择 (提交: `25372ec`, `cd92eae`)
- **R4c 近阈值补集选择 (Complement Selection)**: 可见完整块数 $B = K + 1$ (513) 时，利用单 minimum-reduction 快速排除 1 个最低分块，严格保持 Radix 排序总序与同分小 ID 优先契约，避免多轮 Radix 展开。
- **R4a 有界 Batch Arena**: 预分配固定 query tile arena (`qsa_query_tile_` 默认为 64，`qsa_tile_block_scores`, `qsa_tile_active_blocks`, `qsa_tile_valid_counts`)，消除长上下文动态内存分配与越界。
- **R4b GQA 共享 KV 多 Q Head 并行**: 针对 Q/KV = 24/2 架构，在 SYCL 内核中实现一个 Workgroup 协作处理 4/2/1 个 Q Heads，重用同一 indexed K/V 读取并降低 partial/merge 显存流量。
- **R4d 重入式 Pre-indexer**: 融合短阶段 Raw K 写入、块边界压缩与 Q norm/RoPE，在非对齐 chunk 切分下保持 100% 比特精确可重入。

---

## 2. 严苛工程验证记录 (CTest & 端到端真机)

所有测试均在 Intel Arc A770 真实物理设备上通过单任务串行执行完成：

1. **编译构建**: `pwsh -NoProfile -File build.ps1 -BuildDir build-refactor` **0 错误、0 警告**。
2. **全量 CTest 单元测试**:
   ```powershell
   ctest --test-dir build-refactor -R "AilaSafeTensorsTests|AilaQwen4ExpModelSpecTests|AilaQwen4ExpKernelTests|AilaQwen4ExpModuleTests" --output-on-failure
   ```
   **100% Passed (4/4 tests, 47.76s)**。
3. **深上下文真实模型稀疏端到端验证**:
   - 输入 prompt = 2133 tokens（跨越 2052 稀疏阈值，触发 batched scoring、补集选择与 multi-Q attention）：
   - 真实输出：`"Ships relied on the lighthouse's"`，稀疏全链路端到端功能完全验证。
4. **短上下文黄金基线端到端验证**:
   - 命令：`.\build-refactor\Aila.exe -m C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2 --max-seq 512 --greedy --thinking-budget 0 --max-tokens 48 --messages-json perf/qwen4exp/prompts/story.json`
   - 输出文本：
     `The horizon ignites with a soft, amber glow, piercing the lingering indigo of the pre-dawn sky. Gentle waves ripple beneath the emerging light, their crests catching the first golden rays like scattered diamonds. As the sun breaches the`
   - **48 tokens 与黄金基线保持 100% 比特精确对齐**。

---

## 3. 待解决核心问题：`story.json` 端到端性能回退深度剖析

### 3.1 现象与基线数据对比
参考排查规划：[`docs/plans/qwen3.8-flash-next-story-regression-triage-2026-09-27.md`](file:///E:/RiderProjects/Aila/docs/plans/qwen3.8-flash-next-story-regression-triage-2026-09-27.md)。

| 指标 | 历史最优 (`s4_baseline_measured` / `484b086`) | 当前基线 (`cd92eae` warm run) | 差值与分析 |
|---|---:|---:|---|
| **端到端总耗时 (`GenerateMessages`)** | **6,386.75 ms** | **7,594.91 ms** | **+1,208.16 ms (变慢 ~1.2 秒)** |
| **首段 Prefill (`count=26`) 总耗时** | **2,217.11 ms** | **3,425.49 ms** | **+1,208.38 ms (几乎 100% 回退集中在此！)** |
| - `acquire` (GPU 专家加载/传输) | 1,449.17 ms | 1,510.04 ms | +60.87 ms (基本相当) |
| - `fill` (NVMe/Disk -> Host) | 1,129.90 ms | 1,037.19 ms | -92.71 ms (反而更快) |
| - **`router_sync` (CPU 等待 Router 事件)** | **1.07 ms** | **380.94 ms** | **+379.87 ms (严重停顿！)** |
| - **纯计算时间 (`total - acquire - router_sync`)** | **766.87 ms** | **1,534.51 ms** | **+767.64 ms (算子执行/发射开销翻倍)** |
| **Decode 48 步平均延迟** | **74.94 ms / token** | **75.90 ms / token** | **仅差 ~1.0 ms，Decode 性能基本稳定** |

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
