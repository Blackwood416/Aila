# Phase 7 Decode FFN Graph Fusion Status

## Summary

以 `2026-04-18` 当前稳定基线 `1c692d5` 为起点，已实际尝试在 `Qwen35HybridTextBackend` 中实现 Phase 7 计划里的 decode-only FFN graph 路线，并按串行流程完成最小验证。

本轮实际结论：

- `use_device_linear_decode_` 的生产语义已经从主类中移除
- `debug_compare_linear_delta_decode(...)` 不再依赖 runtime toggle
- `run_linear_delta_host(...)` 现在固定把 host state 同步回 device state
- oneDNN public Graph + SYCL interop 的 FFN graph 试作已验证
- 但在当前环境的 oneDNN `3.9.1` + A770 上，该 FFN graph 只得到 `4` 个 partition，而不是计划要求的 `1`
- 因此 graph helper 已按止损条件回撤，不保留在主线里

## Current Verified State

保留 cleanup 后的当前实测：

- `greedy_main`: `prefill 1571.40 tok/s`
- `greedy_main`: `decode 103.14 tok/s`

对应验证产物：

- `tmp/perf/phase7_cleanup_only_after_graph_stop/1c692d5/bench.json`

结论：

- 当前主线性能基本回到 `1c692d5` 稳定基线
- 本轮 graph 试探没有带来正收益
- 继续保留 graph 死路径只会增加维护噪音，不符合“失败即止损”的原则

## What Was Learned

这轮最关键的发现不是“graph 没写出来”，而是：

- 当前运行时 oneDNN 版本并不能把这条 Qwen3.5 decode FFN 图稳定收成单 partition
- 即使 public Graph pattern 在源码和文档里存在，当前安装版本/当前 GPU 路径并没有给出计划中需要的主线能力
- 所以下一轮不应该继续在 public Graph API 上打转，也不应该围绕 partition 数量继续做小修小补

## Next Direction

下一轮建议从下面几个方向里选一个单独开新计划，而不是继续混在 Phase 7 里：

1. 调研 oneDNN 当前安装版本里是否存在可直接调用的 `gated_mlp` primitive 或内部导出符号，绕开 public Graph partition 匹配限制
2. 重新评估 `Linear.cpp` 之外、但仍基于 oneDNN matmul primitive 的“小粒度 FFN 融合”，重点减少 `gate_up_proj -> swiglu -> down_proj` 的边界开销
3. 若 oneDNN 现成路径确认不可行，则直接转向 Qwen3.5 decode-only FFN custom kernel，严格控制融合粒度，只做最热固定形状

## Guardrails

下一轮继续遵守这些约束：

- 不再继续尝试 public Graph 单 partition 这条已证伪路径
- 不把 `linear_delta`、attention decode、通用 `Linear.cpp` 试探混进同一轮
- 每轮仍只做串行 build 和单 case benchmark
- 正向后再扩测，负向立即回撤
