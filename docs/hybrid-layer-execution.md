# Hybrid 公共层执行

`Qwen35HybridBnb4Backend` 与 `Qwen35HybridTextBackend` 共用 `HybridLayerExecutor::Execute`。加载后固定的 `HybridDecoderPlan<Layer>` / `HybridLayerPlan<Layer>` 持有具体投影、缓存索引及下一层 norm；`HybridLayerExecution` 显式携带阶段、token 行数、起始位置、缓存、MRoPE 和执行工作区。

投影槽位使用编译期 `HybridProjectionSlot`，存储后端各自提供 `ProjectionPolicy`，不在层循环中按模型名称或字符串选择实现。

| 部分 | 公共执行器 | 存储策略保留的行为 |
| --- | --- | --- |
| Full Attention | QKV 视图/拆分、QK norm/RoPE、KV 写入、decode/prefill/cached attention、输出 gate | NF4 投影；BF16 Linear / 专用 JM decode |
| DeltaNet | decode/prefill 分支、分组 host 路径、递归调用与对应 Session cache | 原后端的递归内核、卷积历史提交及调试对照 |
| FFN | gate/up、激活、down、阶段 profiling | NF4 融合 gate/up GEMV；BF16 专用 FFN；各自 fallback |
| Residual/norm | 层入口/出口契约及跨层融合 | NF4 原有可选 residual 参数；BF16 原有 add+norm |

NF4 旧 linear 模式仍沿用原 `DeltaNet` cache tag 和 checkpoint 行为，实际递归选择读取加载时固定的 `use_delta_linear_`。BF16 继续区分 `LegacyLinearAttention`。这次提取不改变旧模式的缓存分配/截断语义。共享 Session 仍要求 DeltaNet 和 BF16 KV。

`tokenCount` 是同一序列的 token 行数，不能解释为独立请求数。执行槽在现有设备锁及 in-order queue 下使用；模板没有引入逐层同步。只有显式启用 profiler 时，`StageProfiler` 才按原设置等待设备。

## 独立旧版本对照

`AilaLayerReferenceTests` 将完整的最后位置 float logits、输入 token、实际 context/checkpoint 长度和固定 seed/penalty 输出写入一个新目录。写入时拒绝覆盖已有目录；比较时检查版本、模型配置、元数据、记录完整性、有限值和逐位相等。默认 CTest 因缺少显式模型/参考返回 skip 77。

```powershell
# 必须在修改生产实现前，或用保留的旧可执行文件捕获。
./build-refactor/AilaLayerReferenceTests.exe --backend qwen35_nf4 --model ./models/qwen3.5-0.8B-bnb-nf4-offline --write ./tmp/old-layer-reference
# 重建候选版本后比较同一个不可覆盖的参考目录。
./build-refactor/AilaLayerReferenceTests.exe --backend qwen35_nf4 --model ./models/qwen3.5-0.8B-bnb-nf4-offline --compare ./tmp/old-layer-reference
```

实际 GPU 调用需使用机器守卫、owned Windows Job 和 watchdog。2026-09-09 本次执行记录位于 `tmp/refactor/p3-hybrid-layers-20260909`，详见 [验证文档第 19 节](refactor-validation.md#19-2026-09-09-hybrid-公共层执行与独立旧产物对照)。这组对照证明代码提取前后的数值保持，不替代独立参考引擎或长期稳定性验收。
