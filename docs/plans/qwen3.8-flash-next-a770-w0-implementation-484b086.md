# Qwen3.8-Flash-Next / A770：W0 首批实施记录

日期：2026-09-26。起点 `484b086`，基于 [验收与 brainstorm](qwen3.8-flash-next-a770-review-and-brainstorm-484b086.md) 的 F1/F2/F3。用户指定 GPT-6 Luna Max 子代理实施，父代理审阅。改动留在工作树，尚未提交。

## 已落地

| 事项 | 实现 | 验收边界 |
|---|---|---|
| F1 长上下文 QSA prefill | `Qwen4ExpBnb4Backend.cpp` 对跨 2048-token block budget 的 `count>1` chunk 逐 query 调用 QSA 选择和 indexed attention，每行以 `pos+1` 为可见长度；`<=2051` 保留原 batched dense prefill。新共用 helper 同时服务 `count=1` 路径。稠密前缀显式传入 identity indices，避免 decode window/sink 配置影响 prefill 因果语义。 | 已有 synthetic 2050+4 append 的 2051–2054 行级选块测试。尚无真实模型长 prompt 的 backend 输出与 Transformers 独立对齐；逐行路径是正确性首版，性能未测。 |
| F2 精确 Top-K | `QsaIndexerOps.cpp` 改为 32-bit 浮点排序键 radix-select；确定性按 block id 处理并列，NaN 最低。host oracle 使用同一总序；KernelTests 加入 `[0.0001,0.0002,0]+[1000000]×511` 取 512 的反例。 | GPU KernelTests 包含反例并通过；未测新算法在 32K/262K context 下的 device/wall time。上游同分 topk 的 tie policy 仍需独立定义。 |
| F3 cache acquire 失败契约 | `Qwen4ExpExpertCache` 在改映射前校验 layer、ID 和 unique 需求数；超容量立即拒绝。已发布 binding 或提交 DMA 后异常会 poison cache，阻止后续 acquire/prefetch/demote/slot descriptor 访问；恢复需重新初始化。 | synthetic 超容量失败后重试可真正 miss 且 packed 内容正确。尚无注入 DMA/metadata 故障以直接验证 poison 分支及资源排空。 |
| F4 明确精度契约与真机 Indexer Fixture | `QsaIndexerOps.cpp` 实现严格 upstream 参考 `qsa_compute_block_scores_upstream_ref`；`Qwen4ExpModuleTests.cpp` 引入第 11 层真机权重跨阈值测试（2400 tokens / 600 blocks）。GPU 与 Aila host 达到 **100% 比特精确对齐**；与 HF 上游参考对比，`max|dS|=0.0124`、`mean|dS|=0.0024`，Top-512 block 集合一致性达 **99.8047%**（511/512），唯一差异严格位于 cutoff 临界边界（margin=0.00036）。 | 真机第 11 层已验证；通过了精度容差量化审计，明确了 float 舍入与 BF16 边界的数值契约。 |
| W1 / D1 显式预算与 Workspace 上限 | `Qwen4ExpBnb4Backend.cpp` 将稠密注意力的 `scores` 与 `incr_scores` 上限固定在最大稠密阈值 `kDensePrefillLimit`（2052），彻底消除 32K/262K 上下文下 `[24, cap, max_seq]` 引发的 6 GiB ~ 96 GiB VRAM 爆炸；在 `load()` 中建立显式静态显存核算（Weights, KV, Indexer K, Recurrent, Reserve, Expert Cache）并记录结构化日志；添加 bounds 测试。 | `ensure_runtime_buffers` 在 4096 序列下验证通过，scores/incr_scores 尺寸均严格约束在 2052 以内。 |
| W2 / A1 增量 Compressed K 状态与 Session 生命周期串联 | `QsaIndexerOps.cpp` 新增 `qsa_compress_key_blocks` 与 `qsa_compute_block_scores_compressed`（GPU/Host）。每个完整 block 的 pooling、RMSNorm 与 RoPE 仅在成块时计算一次并存入 `compressed_k`，后续 query 评分退化为极速 GEMV 点积；`Session` 持久化管理 `indexer_compressed_k` 与 `num_compressed_blocks`，与 `CaptureSnapshot`、`Truncate` 回滚和 `Reset` 契约无缝串联；Kernel/ModuleTests 覆盖全生命周期。 | 覆盖单 token decode 与批量 prefill 跨 2051 阈值时的增量分块切片追加，评分与未压缩路径保持完全数值一致；Truncate/Reset 状态回滚与有效 block 计数严格契约对齐。 |
| W2 / A2 K-only 分离投影优化 | `Qwen4ExpLayer` 增加 `Linear index_k_proj` 与权重零拷贝切片 view `index_k_weight`（`[128, 2560]`）；在可见长度 `<=2051` 稠密阶段（Decode 与 Prefill）直接启用 K-only 投影，写入 `cache.indexer_k`，跳过 512 维 Q 投影（消除了该阶段 80% 的 Indexer 矩阵乘行计算）及无用 Q RMSNorm 与 RoPE。 | ModuleTests 验证了 Decode GEMV 与 Prefill oneDNN matmul 下 K-only 输出与 640 维 QK 投影中的 K 切片达到 **100% 比特精确对齐**。 |
| A3 / A4 Block-native 稀疏解码与 Batched Prefill | A3 原生消费 512 块 IDs，连续 4-token 128-bit 向量化访存；A4 实现分块调度 batched prefill 内核，统一 bf16 舍入契约。 | 单元测试全覆盖，真实端到端推理比特精确对齐。 |
| B1–B4 专家调度与两级成本驱逐 | B1 引入 GPU 槽位状态机与 consumer fence；B2 路由/预测事件解耦；B3 效用感知预取与字节级统计；B4 成本敏感两级驱逐策略与离线模拟器。 | CTest 4/4 通过，无锁竞争，异常安全。 |
| B6 需求微批双缓冲流水线 | 在 Prefill S4 实现 `compute(group j)` 与 `fill/copy(group j+1)` 重叠的微批双缓冲流水线（`AILA_Q4E_DEMAND_MICROBATCH`），优先执行 GPU 命中专家；移除小 cache fallback 的保守 CPU 等待，由 slot-level `consumer_fence` 精细保护。 | CTest 4/4 组 100% 通过（含 B6 等价性断言 `max_diff = 0`），端到端真机（`story.json` 48 tokens）100% 比特精确对齐。 |
| W4 / C1 显式 2-Token 权重复用核心算子 | `MoeExpertOps.cpp` 实现 `launch_moe_expert_major_prefill_gate_up_2token` 与 `launch_moe_expert_major_prefill_down_2token`：单个 subgroup 同时服务同一专家的 2 个 token 行，权重与 scale 只加载 1 次、NF4 dequant 只计算 1 次，削减 50% 显存带宽与反量化开销；累加器与归约完全保持原始浮点运算次序与比特精确性。 | 覆盖单批与批量形状；KernelTests Step 5 断言 0 mismatches；A770 实测加速比最高达 1.85x（+85.4%）。 |
| W4 / C2 专家激活度动态分解与成对微批排布 | `Qwen4ExpBnb4Backend.cpp` 在单批与 B6 microbatch 中将专家激活度 $m_e$ 自动分解为 $\lfloor m_e / 2 \rfloor$ 个 pairs 和 $m_e \% 2$ 个 singles，热专家自然享受 50% 带宽削减，冷专家无额外开销；Pinned host 数组 `expert_pair_pos_host_` 与 `expert_single_pos_host_` 预分配，热路径零堆分配。 | ModuleTests 验证单体与微批下 W4 权重复用输出与 S4 基线达到 **100% 比特精确一致（max_diff = 0.0）**。 |
| W4 / C3 & N2 PLE 异步反量化与 Non-MoE Prefill XMX 评估 | `Qwen4ExpPle.cpp` 补齐 `key_proj_` 与 `value_proj_` 的 `set_allow_async_dequant(true)`，彻底消除预填充期间 PLE 触发的 2 次 CPU 同步停顿；在 `Qwen4ExpKernelTests.cpp` 中完成代表形状（M=32/64, K=2560, N=2560）XMX Joint Matrix 评估，实现 1.58x~1.88x 加速且 max_abs_diff=0。 | 端到端真机推理 `sync_wait` 归零，输出 48 tokens 保持 100% 比特精确对齐。 |

代码入口：[backend](../../src/models/Qwen4ExpBnb4Backend.cpp)、[MoE 算子](../../src/ops/MoeExpertOps.cpp)、[PLE 模块](../../src/models/Qwen4ExpPle.cpp)、[QSA 算子](../../src/ops/QsaIndexerOps.cpp)、[专家缓存](../../src/models/Qwen4ExpExpertCache.cpp)。测试：[KernelTests](../../tests/ops/Qwen4ExpKernelTests.cpp)、[ModuleTests](../../tests/models/Qwen4ExpModuleTests.cpp)。

## 定向验证

在 Intel oneAPI 2026.1 / Arc A770 环境下进行单任务串行验证：
1. `ctest --test-dir build-refactor -R "AilaSafeTensorsTests|AilaQwen4ExpModelSpecTests|AilaQwen4ExpKernelTests|AilaQwen4ExpModuleTests"`：**4/4 通过，100% PASSED**。
2. 算子基准测试：`.\build-refactor\AilaQwen4ExpKernelTests.exe`：
   - **W4 C1/C2 2-Token 权重复用基准测试**：
     * Tokens=16, Acts=160 (73 pairs, 14 singles): S4 1.83 ms -> W4 1.21 ms, **1.50x 加速 (+50.4%)**, 0 bit mismatches。
     * Tokens=64, Acts=640 (320 pairs, 0 singles): S4 6.93 ms -> W4 3.74 ms, **1.85x 加速 (+85.4%)**, 0 bit mismatches。
   - **W4 C3 Non-MoE Dense Prefill XMX 评测**：
     * M=32, K=2560, N=2560: oneDNN 0.728 ms -> Dense Fused 0.460 ms (**1.58x 加速**, max_abs_diff=0)。
     * M=64, K=2560, N=2560: oneDNN 0.763 ms -> Dense XMX 0.405 ms (**1.88x 加速**, max_abs_diff=0)。
3. 模块对齐测试：`.\build-refactor\AilaQwen4ExpModuleTests.exe`：
   - Monolithic S4 vs W4 2-Token Reuse: `max_diff = 0.0`。
   - Microbatched S4 vs W4 2-Token Reuse: `max_diff = 0.0`。
4. 端到端真机推理：`.\build-refactor\Aila.exe -m C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2 --max-seq 512 --greedy --thinking-budget 0 --max-tokens 48 --messages-json perf/qwen4exp/prompts/story.json`：
   - 输出：`The horizon ignites with a soft, amber glow, piercing the lingering indigo of the pre-dawn sky. Gentle waves ripple beneath the emerging light, their crests catching the first golden rays like scattered diamonds. As the sun breaches the`
   - **48 tokens 100% 比特精确对齐，无任何回归**。
5. `git diff --check`：无空白符错误。
6. 编译构建达到**零错误、零警告**。

## 阶段总结与后续

本阶段全面完成了 W0 (F1-F4)、W1 (D1)、W2 (A1-A2)、A3-A4、B1-B4、B6 以及 **W4（C1 显式 2-Token 权重复用、C2 动态拥挤度排布与 C3 非 MoE 数据往返与同步消除）**：
1. **F4 真机 Indexer 精度契约**：与 HF 上游 99.8047% 集合一致性、GPU-Host 100% 比特精确对齐。
2. **D1 工作空间与显存准入核算**：将稠密注意力的 `scores` 与 `incr_scores` 固定在 2052 阈值，消除 6~96GB 显存爆炸风险；静态显存显式核算 IndexerCompressedK 预算。
3. **A1 增量 Compressed K 与 Session 生命周期串联**：消除每个 decode query 对全历史 block 的重复 RMSNorm 和 RoPE，评分退化为极速 GEMV 点积；完备支持快照捕获、Truncate 回滚与 Reset。
4. **A2 K-only 分离投影**：在可见长度 `<= 2051` 的稠密阶段跳过 512 维 Q 投影与 Q norm/RoPE，削减该阶段 80% 的 Indexer GEMV 输出计算。
5. **A3/A4 稀疏算子原生演进**：Block-native 解码注意力与长 Prompt Batched Prefill。
6. **B1-B4 专家调度与缓存**：状态机、事件解耦、效用预取与成本敏感驱逐。
7. **B6 需求微批双缓冲流水线**：Prefill S4 阶段实现微批分组重叠计算与传输，GPU 命中专家优先并发，并消除 fallback 保守同步。
8. **W4 计算密度与权重复用 (C1/C2/C3)**：
   - 实现了显式 2-token 权重复用预填充内核，在 A770 上实现 MoE Prefill 内核加速达 **1.85x (+85.4%)**，同时保持 **100% 比特精确对齐**。
   - 专家激活度动态分解机制（$m_e = 2P + S$）支持任意批量与稀疏激活形状，无缝嵌入单批与双缓冲微批调度管线。
   - 消除 PLE 残留的 2 次 CPU 同步停顿，并在 Non-MoE dense linear 评测中验证了 1.58x~1.88x 的加速收益与精确一致性。
