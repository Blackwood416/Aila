# Phase 5 性能优化阶段计划

## Summary

当前基线以 `2026-04-18` 的工作区为准，锚点是 `0ebd90e` 上的当前修复态：

- 当前 `greedy_main`：`prefill 601.89 tok/s`，`decode 89.63 tok/s`
- 参考目标 `continue_ring_perf`：`prefill 663.13 tok/s`，`decode 101.2 tok/s`
- 当前 decode profile 热点依次为：`ffn_proj`、`down`、`linear_delta`、`linear_proj`、`linear_o`

本阶段不再做“看到哪慢改哪”，而是按 4 个阶段推进：先恢复到清理前性能，再做通用 decode matmul kernel，再做 `linear_delta` kernel，最后清理旧路径。

## Key Changes

### 阶段 1：回到清理前性能基线

目标：先追回 `continue_ring_perf` 的大盘，不引入新实验分支。

实施要点：

- 以当前 `gpu_iter_prefill_repair` 为唯一起点，不再基于更早的 host-prefill 回退态继续改
- 只对比两类实现差异：
  - `Qwen35HybridTextBackend` 的线性注意力热路径
  - `Linear.cpp` 中 decode 专用路径
- 所有验证只保留 1 条串行命令：
  - `.\build.ps1 -BuildDir build -Config Release`
  - `.\bench.ps1 -BuildDir build -Preset phase_gate_q35_text -Phase <phase_name> -CaseNames @('greedy_main')`
- 本阶段禁止继续加新的 env 调参矩阵；只允许结构性修复和旧路径梳理

验收标准：

- `greedy_main` prefill 恢复到 `>= 640 tok/s`
- `greedy_main` decode 恢复到 `>= 95 tok/s`
- 没有新的 host roundtrip 回到主热路径里

### 阶段 2：通用 decode matmul kernel 优化

目标：优先打掉 profile 里最重的 matmul 类阶段，一次覆盖多个热点，而不是单点调参。

实施范围：

- 重点对象：
  - `ffn_proj`
  - `down`
  - `linear_proj`
  - `linear_o`
- 统一在 `Linear.cpp` 做 `seq_len == 1` 的 decode 专用 kernel/dispatch
- 不再继续尝试 raw weight layout 主路径；保留当前转置后权重布局
- 设计原则：
  - 固定启发式选择，不依赖大规模 env 搜索
  - 让 `gate_up_proj`、`down_proj`、`linear_all_proj`、`linear_o_proj` 共享一套 decode 小矩阵策略
  - 优先减少 kernel launch 和无效中间 buffer，而不是堆更多开关

接口/类型变化：

- 不新增用户可见 API
- `Linear` 内部允许新增 decode 专用 dispatch 分支或内部 helper
- 新分支默认直接生效，不走“默认关闭”的实验态

验收标准：

- `decode_profile_main` 中 `ffn_proj + down + linear_proj + linear_o` 合计耗时下降 `>= 15%`
- `greedy_main` decode 达到 `>= 98 tok/s`
- `sample_main` decode 不低于当前已验证设备采样基线的 `97%`

### 阶段 3：`linear_delta` kernel 专项优化

目标：专门处理当前仍然偏重的 `linear_delta`，把剩余 decode 差距收口。

实施范围：

- 保留当前 ring-buffer conv state 设计
- 以现有 `run_linear_delta_decode_gpu(..., out_dst)` 为唯一主入口，继续围绕它做专用 kernel
- 保留“等维快路径”和“泛化回退路径”两层结构，但把默认热路径收敛到固定 specialization
- 优化重点：
  - 降低 local memory 占用
  - 减少 barrier 次数
  - 向量化 q/k/v/z 读写
  - 避免每 token 额外 split/copy
  - 保持 recurrent state 的现有内存布局不变，避免同时引入状态格式迁移

接口/类型变化：

- `Qwen35HybridTextBackend` 继续保留输出目标重载
- 不再新增新的 compare/debug 入口
- `AILA_Q35_LINEAR_DECODE_WG` 这类纯调参开关在本阶段结束后应准备移除或降级为调试专用

验收标准：

- `decode_profile_main` 中 `linear_delta` 平均耗时下降 `>= 15%`
- `greedy_main` decode 达到 `>= 100 tok/s`
- `greedy_main` prefill 保持 `>= 640 tok/s`

### 阶段 4：旧路径清理与主线收口

目标：在 A/B 已经证明准确性和性能后，把旧路径真正删掉，避免后续再误入。

清理对象：

- `run_linear_delta_host(...)` 主线依赖
- 仅用于对拍的 `debug_compare_linear_delta_decode(...)`
- 已证明负收益的 `Linear.cpp` 实验 decode JM 老分支
- 与最终默认实现无关的 decode 调参 env 开关
- 误导性的模式日志和过时注释

保留原则：

- 只保留一个生产默认热路径
- 如需 reference path，仅允许保留明确标注为 debug-only 的最小正确性路径
- 不保留“默认不用但还能跑”的半废弃实现

最终验收标准：

- `.\perf_suite.ps1 -Preset phase_gate_q35_text -Phase <final_phase>`
- `greedy_main` decode `>= 100 tok/s`
- `sample_main` decode 不回退到设备采样前
- preset 自带 smoke 全通过

## Test Plan

每个阶段只做最少必要验证，避免重新掉进调参循环：

- 结构性改动后的第一验证：
  - 串行 build
  - 单跑 `greedy_main`
- 只有当 `greedy_main` 有正向变化时，才补跑：
  - `decode_profile_main`
- 只有阶段收口时，才跑：
  - `sample_main`
  - `perf_suite.ps1`
  - smoke

固定验收命令：

```powershell
.\build.ps1 -BuildDir build -Config Release
.\bench.ps1 -BuildDir build -Preset phase_gate_q35_text -Phase <phase_name> -CaseNames @('greedy_main')
.\profile_decode.ps1 -Preset phase_gate_q35_text -Phase <phase_name>
.\bench.ps1 -BuildDir build -Preset phase_gate_q35_text -Phase <phase_name> -CaseNames @('sample_main')
.\perf_suite.ps1 -Preset phase_gate_q35_text -Phase <phase_name>
```

## Assumptions

- 本轮目标优先级是：`decode kernel 实现 > 结构性修复 > 小幅调参`
- 当前 `601.89 / 89.63` 是阶段起点，不再回到更差的 host-prefill 中间态
- `continue_ring_perf` 的 `663.13 / 101.2` 作为最近一次已知高水位目标
- 设备采样路径已验证有效，本阶段不把它当主攻方向，只要求不回退
- 本计划默认允许在通过 A/B 后删除旧路径，而不是长期并存
