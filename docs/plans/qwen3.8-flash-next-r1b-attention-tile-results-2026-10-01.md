# Qwen3.8-Flash-Next：R1b attention tile 与 9033 驱动复测

日期：2026-10-01。继续 [架构规划](qwen3.8-flash-next-architecture-brainstorm-2026-09-30.md) 的 R1b。**本轮完成 attention/GDN tile 解耦这一部分，尚未完成紧凑 MoE merge、动态预算窗口或调度器层间让出。** GPU-only、32 GiB host pool；A770 16 GB，驱动 32.0.101.9033，oneAPI 2026.1 / oneDNN 3.11.2。模型仍为 C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2。

**[MEASURED] 新候选回收 258.27 MiB GPU USM，但完整请求没有加速。** 同新二进制四进程交错对照，暖态 TTFT 增加 1.7%，16-token 完整请求中位数增加 0.22%。候选的 decode 为 5.5 / 5.6 tok/s，控制组为 5.4 / 4.8 tok/s；不能由少量样本宣称稳定的普遍 decode 加速。保持 attention tile 默认关闭、preset chunk=512；实际 decode 20 tok/s 仍未达到。

配置、哈希、原始日志路径、计数器和内存采样见 [结果 JSON](qwen3.8-flash-next-r1b-attention-tile-results-2026-10-01.json)。

## 1. 驱动升级检查：同一个旧二进制

在任何重编译前，使用上一轮 Aila.exe 重跑相同自然输入。历史与升级后 EXE SHA256 均为 0BFB43F26E2F64CAA0F229CCAB76ED709006ABD63F2416001C33E6587B77FDDC；metadata 记录实际 GPU 驱动版本。输入 1833 tokens，max-seq=2560，3567 专家槽，输出 16 tokens，每进程两次请求。

| 窗口 | 8860 暖 TTFT | 9033 暖 TTFT | 8860 暖 decode | 9033 暖 decode |
|---|---:|---:|---:|---:|
| 512 | 22.607 s（两次中位数） | 23.198 s（一次） | 7.35 tok/s | 7.5 tok/s |
| 2048 | 16.412 s（两次中位数） | 16.877 s（一次） | 5.25 tok/s | 5.2 tok/s |

**[MEASURED] TTFT 分别比历史中位数慢 2.6% / 2.8%，decode 基本在此前范围附近。** 这是需要保留的小幅变慢信号，不是已经证明的驱动回退。跨天、样本少、未做驱动往返安装；后续新二进制未启用候选的 2048 控制组也测到 16.107 / 16.525 s。不能把这组不同二进制样本并入驱动对照，但它说明环境波动仍然存在。本轮没有 GPU device-loss、TDR 或 GPU 数值非有限错误。

前缀为 tmp/r1b-driver9033-{512,2048}-20261001。后续代码候选统一在 9033 上比较，不混入 8860 样本计算代码收益。

## 2. 实现边界

[Qwen4ExpPrefill.cpp](/E:/RiderProjects/Aila/src/models/Qwen4ExpPrefill.cpp) 抽出 RunAttentionLayerTile / RunAttentionWindow。每层仍保留整窗口 hidden/HC 输入输出，按时间顺序分 tile 执行 attention/GDN，随后对整窗口进行一次路由和现有专家分组。没有退回每个小 tile 跑完 48 层的旧小窗口顺序。

- AILA_Q4E_PREFILL_ATTN_TILE 接受 0 / 128 / 256 / 512，默认 0；本轮自然文本性能候选为 256。启动 ExecutionConfig 打印生效值。
- QKV、Q/Z/K/V、attention output、indexer 投影、GDN 输入及其可选 chunk scratch 按 tile 容量分配；dense scores 的 query 维按 tile、KV 维仍按可见前缀上界分配。
- KV、压缩 indexer K、GDN recurrent/conv state 继续属于传入的 Session。QSA 每个 query 使用自己的可见前缀，未共享选块列表。
- 末尾若只余一行，会把前一个 tile 缩一行以留下两行，避免误入 decode 的 window/sink 分支。count=1 的真正 decode 保持原路径。
- 所有 tile 使用原有 in-order 队列；非 owning 输入输出 view 不释放 backing store。现有 growth drain、Forward 末尾状态发布与错误处理不变，没有增加 decode USM 分配。
- 原有 interior checkpoint 的 ≤64 行限制保留。未为了通过新测试放宽生产条件。

**[CORRECTNESS] MoE 仍保留整窗口 [T,top_k,H] FP32 rank 输出，专家槽仍为 3567。** 释放的空间没有自动扩充专家池。1829 行时，phase arena 从 360,298,368 B 降到 187,289,600 B，后者现在由 MoE rank 输出决定；加上其他 attention bank 的缩减，总 ledger 减少 258.27 MiB。这是 R1b 的中间交付，不能称作整个 PrefillExecutor 已完成。

## 3. 同驱动、同新二进制性能对照

EXE SHA256 为 9AA568AFCCD6CCA0C1CA813CD6D594A11FE9A0CDB13CC06DC6BEADA6E1194DD7。仍使用 [自然文本](/E:/RiderProjects/Aila/perf/qwen4exp/prompts/archive-natural.json)，1833 tokens、max-seq=2560、32 GiB host pool、3567 专家槽。四个独立进程按 tile256→关闭→tile256→关闭交错，每个进程两次清空会话后的请求，各生成 16 tokens。

路由窗口均为 chunk=2048，实际 Forward 仍为 1829+4；tile256 在第一个 Forward 内将 attention 分成 7×256+37，最后 4 行 Forward 单独处理。输入低于 QSA 稀疏切换阈值，本表不测 QSA 稀疏收益。

| 指标 | 未分 attention tile | tile=256 |
|---|---:|---:|
| 首请求 TTFT | 40.582 / 40.058 s | 50.172 / 40.395 s |
| 暖 TTFT | 16.107 / 16.525 s | 16.713 / 16.466 s |
| 暖 TTFT 中位数 | 16.316 s | 16.5895 s（+1.7%） |
| 输入 tokens / 暖 TTFT | 112.34 tok/s | 110.49 tok/s |
| 暖 decode | 5.4 / 4.8 tok/s | 5.5 / 5.6 tok/s |
| 暖 decode 中位数 | 5.1 tok/s | 5.55 tok/s |
| 暖完整请求中位数 | 19.099125 s | 19.141105 s（+0.22%） |
| 暖 prefill demand 字节 | 38.317594 GB | 38.330266 GB |
| 暖 decode demand 字节（含末尾状态对齐） | 6.036941 GB | 6.057216 GB |
| 暖 decode host misses 中位数 | 373.5 | 375 |
| 峰值 device ledger | 13.94446 GiB | 13.69225 GiB（−258.27 MiB） |

八次请求生成的 16-token 文本相同。候选首进程的首请求 50.172 s 明显慢于其他首请求，不能删除；本轮未隔离 JIT、文件缓存或其他原因，不能将其归因于某一个因素。暖态 decode 中位数增加约 8.8%，其中控制组的 4.8 tok/s 回落影响很大；完整请求并未获益。没有继续扫更小 tile。

普通性能进程仅开启不增加 GPU 等待的 WorkSummary，没有 stage profile 或 cache trace 同步。逻辑 demand 字节不是裸 PCIe 带宽；USM ledger 不是 WDDM 驻留量。全模型性能进程最低系统可用 RAM 约 11.826 GiB，最高工作集约 37.358 GiB，未触发 8 GiB 停止门槛。

## 4. 数值、状态和因果检查

新增 [attention tile canary](/E:/RiderProjects/Aila/tests/models/Qwen4ExpAttentionTileAcceptance.hpp)：

1. 取自然输入 JSON 原文 token 化后的前 257 tokens（这是有界 fixture，不是正常 CLI 模板），比较未分 tile 与 tile128 的 8 步固定续写。logits NRMSE=0、NLL delta=0，KL 仅 1.76e−16 的计算噪声。本检查的预设门槛为 mean KL≤0.02、max logits NRMSE≤0.15。
2. 129+128 行的真实分段前缀快照，验证后续 suffix 改写不改变已保存的 recurrent/PLE 快照和前 129 行 KV；truncate 后两次重放位级一致。129 行前缀本身会触发 tile128 的单行尾部避让。这里没有声称存在 129 行 interior checkpoint。
3. 真实 layer 3 full-attention，初始可见前缀 KV/indexer K 为零，输入为确定性 BF16，start=1796、count=257，跨越 2052 稀疏边界。未分 tile 对 tile128 的输出 NRMSE=0.000171792、max abs=0.00390625，通过预设 NRMSE≤0.05；改变后续 tile 输入，前 128 行输出完全相同。

实际 1833-token CLI 模板的独立 48 步 teacher-forcing 比较，固定路由窗口 2048，对照未分 tile 与 tile256：mean KL=0.00589865、mean NLL delta=−0.01321465、max logits NRMSE=0.07879167，全部有限且无 NLL 预警。该测试比较同一 NF4 模型的执行舍入，不要求自由生成一致；负 NLL delta 不是泛化质量提高的证据。

首次状态 canary 错把 257 行输入请求为 interior checkpoint，触发现有 ≤64 行 guard 后退出，记录保留在 tmp/r1b-tile-state-20261001.*。修正为上述真实分段快照后，tmp/r1b-tile-state-v2-20261001 通过；这是测试调用错误，不是驱动或 GPU 故障。

## 5. 复现与后续

~~~powershell
# 同自然输入候选；去掉 -AttentionTile 256 即控制组。每次换 Tag。
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload generation -RuntimeDefaults -HostRamMB 32768 -PerformancePreset qwen38-a770 -WorkSummary -GenerationPromptFile perf/qwen4exp/prompts/archive-natural.json -GenerationMaxSeq 2560 -GenerationRepeats 2 -MaxTokens 16 -PrefillChunk 2048 -AttentionTile 256 -Tag r1b-repro

pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload tile -Tag r1b-state-repro
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload window -AttentionTile 256 -Tag r1b-quality-repro
~~~

[汇总脚本](/E:/RiderProjects/Aila/perf/qwen4exp/q4e_prefill_tile_summary.py) 校验日志哈希、进程退出、输入、模型、RAM 和槽数，并按 driver / EXE / window / tile 分组，避免把驱动变化混成代码收益。普通性能前缀为 tmp/r1b-2048-{tile256-first,control,tile256-repeat,control-repeat}-20261001；自然质量前缀为 tmp/r1b-natural-quality256-20261001。

已构建 Aila、AilaWorker、AilaShared、AilaInference、AilaQwen4ExpToleranceTests；针对性 GPU 检查、Python/PowerShell 语法、JSON 与 diff 空白检查通过。未运行全量 CTest 或 HTTP 并发套件。本轮复用原有算子，没有引入新 SIMD kernel、全局 immediate-list 或 out-of-order queue，也未重新采集 VTune。

**[HYPOTHESIS] 下一部分应让 MoE down 使用有界临时区和 [T,H] FP32 累加，替代十份整窗口 rank 输出，再检验 prefill→decode 容量交接。** 这比继续缩 attention tile 更直接：本次未减少专家需求字节、host misses 也未下降，说明只缩 attention 工作集不足以解决供给和计算成本。之后继续 R2b / R3-head；不将本轮中间结构当作已实现 20 tok/s 的方案。
