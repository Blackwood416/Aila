# Qwen3.8-Flash-Next / A770：W0 首批实施记录

日期：2026-09-26。起点 `484b086`，基于 [验收与 brainstorm](qwen3.8-flash-next-a770-review-and-brainstorm-484b086.md) 的 F1/F2/F3。用户指定 GPT-6 Luna Max 子代理实施，父代理审阅。改动留在工作树，尚未提交。

## 已落地

| 事项 | 实现 | 验收边界 |
|---|---|---|
| F1 长上下文 QSA prefill | `Qwen4ExpBnb4Backend.cpp` 对跨 2048-token block budget 的 `count>1` chunk 逐 query 调用 QSA 选择和 indexed attention，每行以 `pos+1` 为可见长度；`<=2051` 保留原 batched dense prefill。新共用 helper 同时服务 `count=1` 路径。稠密前缀显式传入 identity indices，避免 decode window/sink 配置影响 prefill 因果语义。 | 已有 synthetic 2050+4 append 的 2051–2054 行级选块测试。尚无真实模型长 prompt 的 backend 输出与 Transformers 独立对齐；逐行路径是正确性首版，性能未测。 |
| F2 精确 Top-K | `QsaIndexerOps.cpp` 改为 32-bit 浮点排序键 radix-select；确定性按 block id 处理并列，NaN 最低。host oracle 使用同一总序；KernelTests 加入 `[0.0001,0.0002,0]+[1000000]×511` 取 512 的反例。 | GPU KernelTests 包含反例并通过；未测新算法在 32K/262K context 下的 device/wall time。上游同分 topk 的 tie policy 仍需独立定义。 |
| F3 cache acquire 失败契约 | `Qwen4ExpExpertCache` 在改映射前校验 layer、ID 和 unique 需求数；超容量立即拒绝。已发布 binding 或提交 DMA 后异常会 poison cache，阻止后续 acquire/prefetch/demote/slot descriptor 访问；恢复需重新初始化。 | synthetic 超容量失败后重试可真正 miss 且 packed 内容正确。尚无注入 DMA/metadata 故障以直接验证 poison 分支及资源排空。 |

代码入口：[backend](../../src/models/Qwen4ExpBnb4Backend.cpp)、[QSA 算子](../../src/ops/QsaIndexerOps.cpp)、[专家缓存](../../src/models/Qwen4ExpExpertCache.cpp)。测试：[KernelTests](../../tests/ops/Qwen4ExpKernelTests.cpp)、[ModuleTests](../../tests/models/Qwen4ExpModuleTests.cpp)。

## 定向验证

在项目 oneAPI 2026.1 环境下，子代理完成 `AilaQwen4ExpKernelTests`、`AilaQwen4ExpModuleTests` 两目标构建。`ctest --test-dir build-refactor -R '^AilaQwen4Exp(Module|Kernel)Tests$' --output-on-failure` 为 **2/2 通过，28.89 秒**。最后一次 backend 调整后又构建并运行 ModuleTests，**1/1 通过，11.13 秒**。`git diff --check` 无空白错误。没有全套 CTest、长 GPU benchmark 或真实模型长提示词生成。

最初裸 CMake 构建因缺 `libircmt.lib` 链接失败；加载项目所用 oneAPI 环境后重试成功。这是构建环境问题，不能计为代码失败。

## 父代理审阅与后续

父代理检查了 F1 的短 prompt 退化风险，要求并确认将 `<=2051` chunk 保留在原批量路径。逐行实现利用同一 in-order queue，当前行索引供 attention 消费后才被下一行覆盖。F2 反例已在 GPU 测试覆盖；F3 的超容量失败在任何 key 发布前发生。

**当前仍不宣称 M2 全面完成。** 最小下一步是一个真实 checkpoint 的跨阈值 prompt/append 对照，重点检查 F4 指出的 norm/RoPE 舍入与 selected set；随后优先处理 32K workspace 预算和 incremental compressed K。大尺寸 `scores/incr_scores` 的内存增长、逐行 sparse prefill 的 launch 数与精确 Top-K 的成本仍未解决。不要用本轮 2/2 的定向测试外推长上下文吞吐。
