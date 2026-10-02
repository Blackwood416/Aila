# Qwen3.8-Flash-Next：R1b 有界 MoE scratch 与顺序无关累加

日期：2026-10-01。继续 [R1b attention tile](qwen3.8-flash-next-r1b-attention-tile-results-2026-10-01.md)，完成默认关闭的紧凑 MoE down / window merge 原型。A770 16 GB、驱动 32.0.101.9033、oneAPI 2026.1 / oneDNN 3.11.2；GPU 专家计算、32 GiB host pool，模型 C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2。

**[MEASURED] 在 attention tile=256 的基础上再回收 94.73 MiB GPU USM，但没有获得端到端加速。** 同二进制交错对照的暖 TTFT 17.734→17.8335 s，16-token 完整请求 20.46642→20.59251 s，decode 5.15→5.10 tok/s。小幅变化不足以支持性能推广；保持默认关闭，preset chunk=512 不变。实际 decode 20 tok/s 尚未达到。

完整配置、日志哈希、进程和内存记录、数值检查及 VTune 指令 CSV 汇总见 [结果 JSON](qwen3.8-flash-next-r1b-compact-moe-results-2026-10-01.json)。本轮完成有界 MoE 输出与 merge；自适应路由窗口、内存重新分配给专家池、调度器层间让出仍未实施，不能称整个 R1b 已完成。

## 1. 实现与数学契约

新增 AILA_Q4E_MOE_COMPACT_ACTS，接受 0 / 256 / 512 / 1024，默认 0。本轮整模型测试为 1024。它要求 GPU ALU expert-major 路径、demand microbatch>0、grouped XMX 关闭；不满足条件时明确报错。count≤4 保留原来的 rank 输出路径，兼容 decode、小窗口验证和 CLI 尾段。

[MoeExpertOps.cpp](/E:/RiderProjects/Aila/src/ops/MoeExpertOps.cpp) 为原有 single/pair/quad down kernel 增加紧凑输出分支：输入仍从完整的 [token,rank,I] gate 激活读取，输出写到 [activation,H]。原有 NF4/nested scale、uint32 权重加载、FMA 和 reduction 不变。源码对照与真实专家逐元素检查都确认，改变的是目标地址。

[RunCompactExpertDown](/E:/RiderProjects/Aila/src/models/Qwen4ExpPrefill.cpp) 在已获取的专家 microbatch 内切分有界 activation tile，保持 4/2-token 复用单元完整。原始 token/rank/slot 描述符继续位于不可变的整窗口切片，inverse map 将 token/rank 映射回 activation。每个 tile 中，同一 token 的第一个 activation 独占其输出更新；没有浮点 atomic，重复 token 不会产生写竞争。

### 为什么没有直接使用 FP32 窗口累加

**[CORRECTNESS] 缓存驻留顺序、专家频次以及 quad/pair 分组会改变 tile 的提交顺序。** 若直接按到达顺序累加 FP32，同一前缀可能因未来 suffix 或缓存状态产生不同舍入结果。要严格按 rank 顺序求和，最坏情况下又必须保留接近完整的 rank 输出。

本原型因此采用 **int64 Q36 窗口累加**，而非原规划的 FP32 累加：每个 FP32 加权贡献乘 2^36，舍入到最近整数，再执行精确整数加法；最后转换为 FP32/BF16 输出。它不是专家权重重新量化，也没有引入 CPU 专家计算。

- Q36 转换对每项贡献的绝对舍入误差不超过 2^-37，top-10 的量化误差合计不超过约 7.28e-11。这个界限只描述 Q36 转换，不是相对旧 FP32 求和树的总差异，模型差异仍单独验收。
- 限制每项贡献绝对值≤2^20、top_k≤16，使任意提交次序下 |整数累加值|≤2^60，避免 int64 溢出。
- 由于项目启用 -ffast-math，范围保护检查浮点位模式，不能依赖可能被优化掉的 isfinite。超过范围、NaN 或 Inf 设置设备错误标志，在窗口状态发布前检查并报错，不静默截断。
- 报错不声称自动回滚所有已执行的 recurrent state；沿用既有失败请求的复位处理。正常路径每个大窗口只读取一次错误标志，decode 不增加这次检查。

窗口整数累加比原计划 FP32 多占一倍空间，但仍显著小于十份 rank 输出，并且与专家/tile 提交顺序无关。所有 buffer 预分配，继续使用 in-order compute queue 与既有 cache reader fences。没有启用全局 OOO、immediate command lists 或 GPU 自旋等待。

## 2. 容量变化

自然输入实际 Forward 为 1829+4，窗口 2048、attention tile=256，H=2560、top_k=10。

| 项目 | 原 rank 输出 | 紧凑候选 |
|---|---:|---:|
| 整窗口 FP32 rank 输出 | 187,289,600 B | 不再分配 |
| 紧凑 down scratch | — | 10,485,760 B，1024×H×4 |
| phase arena | 187,289,600 B | 50,429,952 B，由 attention scores 决定 |
| Q36 窗口累加 | — | 37,457,920 B，1829×H×8 |
| inverse map / 错误标志 | — | 73,160 B / 4 B |
| 观察到的 device ledger 最大值 | 13.69225 GiB | 13.59974 GiB |

净减少 **99,328,564 B = 94.7271 MiB**，与上表分配差一致。rank 输出消除后，attention scores 又成为 phase arena 的最大项。gate 激活和 hidden/HC bank 仍按窗口保存；专家槽均为 **3567**，节省空间未自动返还专家池。这里的 ledger 是 Forward 采样中的 USM 计数，不是 DXGI/WDDM 驻留量或所有瞬态分配的峰值。

## 3. 真实 CLI 性能

同 [自然文本](/E:/RiderProjects/Aila/perf/qwen4exp/prompts/archive-natural.json)，实际 1833 tokens、max-seq=2560、输出 16 tokens。四个独立进程按 control→compact→control→compact 执行，每进程两次清空会话后的相同请求；第二次为暖态。两组均为 attention tile=256、chunk=2048、32 GiB RAM。EXE SHA256 均为 09B1F7444C000ED8F519E2A84462EA00F585B55279E2C09F223EF04C6761F430。

| 指标 | control | compact=1024 |
|---|---:|---:|
| 首请求 TTFT | 50.394 / 39.583 s | 40.666 / 39.339 s |
| 暖 TTFT | 17.943 / 17.525 s | 17.855 / 17.812 s |
| 暖 TTFT 中位数 | 17.734 s | 17.8335 s（+0.56%） |
| 输入 tokens / 暖 TTFT | 103.36 tok/s | 102.78 tok/s |
| 暖 decode | 5.2 / 5.1 tok/s | 5.0 / 5.2 tok/s |
| 暖 decode 中位数 | 5.15 tok/s | 5.10 tok/s |
| 暖完整请求中位数 | 20.46642 s | 20.59251 s（+0.62%） |
| 暖 prefill demand 字节中位数 | 38.331533 GB | 38.330266 GB |
| 暖 decode host misses 中位数 | 375 | 375.5 |

八次请求生成文本相同。control 首进程的首请求明显偏慢，原样保留，不由此计算冷态加速；本轮没有隔离其原因。输入 tokens/TTFT 是暖态有效速率，不是单独 GPU prefill 计时。demand 字节为逻辑已调度 payload，不是物理总线带宽。普通性能进程仅启用不增加 GPU 等待的 WorkSummary，未开 stage profile 或 cache trace。

本轮控制组比上一轮 attention tile 对照的 16.5895 s 也偏慢，不能混用不同二进制/时段计算收益。本次一直使用 9033，没有新的跨驱动对照，无法把变化归因于驱动。最低系统可用 RAM 为 11.288 GiB，最高进程工作集约 37.277 GiB，未触发 8 GiB 内存门槛或 watchdog，也未出现 GPU device-loss/TDR。

## 4. 验证与局部成本

### 真实专家算子 fixture

[CompactMoeProbe](/E:/RiderProjects/Aila/tests/models/Qwen4ExpCompactMoeProbe.hpp) 从模型加载 layer 0 专家 7、113，使用确定性 BF16 激活，H=2560、I=640、每专家 32 行；两个非零 rank 权重各 0.5。它是两个真实专家的局部 fixture，不是捕获的全模型真实激活。

- single/pair/quad 混合调用覆盖 n=1/3/11/27、非零 map 偏移，down 结果与原路径逐元素相等；scratch guard 未被覆盖。
- Q36 累加与独立 CPU 整数参考完全相等。参考由原路径的 FP32 rank 输出与 route 权重构造，验证的是输出映射与 merge，而非重新认证 NF4 解量化。
- 正序和倒序提交四个 tile，最终输出位级一致。本 fixture 对旧 ALU 合并结果 errors=0、NRMSE=0、max abs=0；验收混合容差为 0.02 relative + 0.0002 absolute。
- 2^21、NaN、Inf 三种注入全部触发保护。循环无新增 USM allocation。

计时包含 gate/up、down、merge、清零与完成等待，不包含两专家 NF4 槽的首次 H2D；双方都预加载。候选刻意以 16 activation 一 tile 运行以覆盖重复复用，不能把其开销比例外推为生产 1024-act tile 的整模型比例。各模式 warmup 5 次，3 轮×20 个交错样本：

| 模式 | 三轮中位数（μs） | 三轮中位数的中位数 |
|---|---|---:|
| 原 ALU + rank merge | 542.6 / 545.7 / 549.8 | 545.7 μs |
| 紧凑 down + Q36 merge | 825.05 / 820.5 / 827.85 | 825.05 μs（约 +51.2%） |

### 模型状态与质量

最终构建的 [状态 canary](/E:/RiderProjects/Aila/tests/models/Qwen4ExpAttentionTileAcceptance.hpp) 全部通过：257-token 窗口、129 行可见前缀，修改未来 suffix 后，全窗口计算出的前缀 hidden 和 KV 位级一致；分段快照不受 suffix 影响，truncate 后重复重放一致。跨 2052 稀疏边界的真实 attention 层仍通过，NRMSE=0.000171792。8 步短 fixture 的两个 attention 切分方式均启用 compact，其 logits NRMSE=0；它不是 compact 与原 merge 的独立质量比较。

独立自然文本比较固定 2048 路由窗口和 attention tile256，切换 compact=0/1024，使用相同 48 个 teacher-forced tokens：mean KL=0.00481504、mean NLL delta=−0.00323072、max logits NRMSE=0.07347090，无 NLL 预警。负 NLL delta 不代表泛化质量改善。

真实 CLI 41-token、anchor=37 的 compact checkpoint canary 也通过：48 步 mean KL=0.00138532、mean NLL delta=+0.00749027、max logits NRMSE=0.05557242；快照、KV、恢复和重复重放检查通过。它覆盖了大于四行的紧凑路径与四行尾段旧路径的衔接，没有放宽 ≤64 行 interior checkpoint 限制。

### VTune

最终指令报告为 [CSV](/E:/RiderProjects/Aila/tmp/r1b-compact-vtune-final-report.csv)。新 inverse map、merge、finish，以及测到的紧凑 single/pair/quad down 均为 SIMD16、spill=0。正常 merge 的八个实例合计 2,317,312 instructions；三次异常注入另列，未混入正常性能计时。没有从“无 spill”推导加速；额外提交和 merge 的完整成本仍由上表与 CLI 结果判断。instrumented 时间未用于速度比较。

## 5. 复现、交付与下一步

~~~powershell
# 候选；去掉 -CompactMoe 1024 得到同 attention tile 的控制组，每次换 Tag。
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload generation -RuntimeDefaults -HostRamMB 32768 -PerformancePreset qwen38-a770 -WorkSummary -GenerationPromptFile perf/qwen4exp/prompts/archive-natural.json -GenerationMaxSeq 2560 -GenerationRepeats 2 -MaxTokens 16 -PrefillChunk 2048 -AttentionTile 256 -CompactMoe 1024 -Tag compact-repro

pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload compact -Tag compact-op-repro
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload tile -AttentionTile 256 -CompactMoe 1024 -Tag compact-state-repro
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload window -AttentionTile 256 -CompactMoe 1024 -Tag compact-quality-repro
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload preset -AttentionTile 256 -CompactMoe 1024 -Tag compact-checkpoint-repro
~~~

已构建 Aila、AilaWorker、AilaShared、AilaInference、AilaQwen4ExpToleranceTests；最终算子、状态、质量、checkpoint 与 VTune 运行通过。脚本按 driver / EXE / window / attention tile / compact acts 分组，不混算候选。未运行全量 CTest 或 HTTP 并发套件，未改生产 MTP 或调度器。

原始前缀：性能为 tmp/r1b-compact-{control-first,candidate-first,control-return,candidate-repeat}-20261001；最终验证为 tmp/r1b-compact-{operator-final,state-final,quality,checkpoint,vtune-final}-20261001。初版记录保留但不替代最终验证。

**决策：保留默认关闭的容量候选，不继续为此原型扫更多 tile 参数。** R1b 已有大路由窗口、小 attention tile 和有界 MoE 输出的可执行组件，但本轮专家需求字节和 host misses 没有下降，decode 也没有改善。下一项按架构规划推进 R2b 的 NF4 integer-LUT × Q8 activation，再做 R3-head；预算自适应和服务公平性仍需单独交付，不把局部组件当作完整执行器已完成。
