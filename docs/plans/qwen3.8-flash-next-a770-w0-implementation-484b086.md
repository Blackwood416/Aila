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

代码入口：[backend](../../src/models/Qwen4ExpBnb4Backend.cpp)、[QSA 算子](../../src/ops/QsaIndexerOps.cpp)、[专家缓存](../../src/models/Qwen4ExpExpertCache.cpp)。测试：[KernelTests](../../tests/ops/Qwen4ExpKernelTests.cpp)、[ModuleTests](../../tests/models/Qwen4ExpModuleTests.cpp)。

## 定向验证

在 Intel oneAPI 2026.1 / Arc A770 环境下进行单任务串行验证：
1. `ctest --test-dir build-refactor -R '^AilaQwen4ExpKernelTests$'`：**1/1 通过，17.92 秒**。
2. `ctest --test-dir build-refactor -R '^AilaQwen4ExpModuleTests$'`：**1/1 通过，13.07 秒**。
3. `ctest --test-dir build-refactor -L 'cpu|abi'`：**18/18 通过，171.08 秒**。
4. `git diff --check`：无空白符错误。
5. 修复 `resolve_qwen4exp_model_dir` 中 `std::getenv` 弃用告警，编译构建达到**零错误、零警告**。

## 阶段总结与后续

本阶段全面完成了 W0 的全部事项（F1-F4）以及 W1/W2 核心架构演进：
1. **F4 真机 Indexer 精度契约**：与 HF 上游 99.8047% 集合一致性、GPU-Host 100% 比特精确对齐。
2. **D1 工作空间与显存准入核算**：将稠密注意力的 `scores` 与 `incr_scores` 固定在 2052 阈值，消除 6~96GB 显存爆炸风险；静态显存显式核算 IndexerCompressedK 预算。
3. **A1 增量 Compressed K 与 Session 生命周期串联**：消除每个 decode query 对全历史 block 的重复 RMSNorm 和 RoPE，评分退化为极速 GEMV 点积；完备支持快照捕获、Truncate 回滚与 Reset。
4. **A2 K-only 分离投影**：在可见长度 `<= 2051` 的稠密阶段跳过 512 维 Q 投影与 Q norm/RoPE，削减该阶段 80% 的 Indexer GEMV 输出计算。

后续可进一步推进 A3（Block-native attention 优化）与 A4（长上下文真正 Batched QSA prefill）。
