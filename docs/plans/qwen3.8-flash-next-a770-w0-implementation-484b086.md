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
| W2 / A1 增量 Compressed K 状态 | `QsaIndexerOps.cpp` 新增 `qsa_compress_key_blocks` 与 `qsa_compute_block_scores_compressed`（GPU/Host）。每个完整 block 的 pooling、RMSNorm 与 RoPE 仅在成块时计算一次并存入 `compressed_k`，后续 query 评分退化为极速 GEMV 点积；KernelTests 覆盖压缩精度与评分对齐（`max_diff < 0.0008`）。 | KernelTests 覆盖 GPU vs Host 压缩与增量分块切片追加，评分与未压缩路径保持完全数值一致。 |

代码入口：[backend](../../src/models/Qwen4ExpBnb4Backend.cpp)、[QSA 算子](../../src/ops/QsaIndexerOps.cpp)、[专家缓存](../../src/models/Qwen4ExpExpertCache.cpp)。测试：[KernelTests](../../tests/ops/Qwen4ExpKernelTests.cpp)、[ModuleTests](../../tests/models/Qwen4ExpModuleTests.cpp)。

## 定向验证

在 Intel oneAPI 2026.1 / Arc A770 环境下进行单任务串行验证：
1. `ctest --test-dir build-refactor -R '^AilaQwen4ExpKernelTests$'`：**1/1 通过，20.45 秒**。
2. `ctest --test-dir build-refactor -R '^AilaQwen4ExpModuleTests$'`：**1/1 通过，12.09 秒**。
3. `ctest --test-dir build-refactor -L 'cpu|abi'`：**18/18 通过，170.92 秒**。
4. `git diff --check`：无空白符错误。
5. 编译构建零错误、零崩溃。

## 阶段总结与后续

本轮完成了 W0 的最后一项 F4（真机 Indexer 精度与边界审计，实现与 HF 上游 99.8% 集合一致性、GPU-Host 100% 比特精确对齐），同时完成了 W1 的 D1（工作空间上限与显存准入核算，排除了 6~96GB 显存溢出风险）以及 W2 的 A1（增量 Compressed K 算子与测试，消除每次 query 对全历史 block 的重复 RMSNorm 和 RoPE）。
后续可进一步在真实端到端推理流水线中将 `compressed_k` 作为 Session 持久化状态串联，并探索 A2 的 K-only 分离投影优化。
