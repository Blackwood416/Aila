# Qwen3.8-Flash-Next H7：供给竞争治理与长文窗口实测

日期：2026-10-03。承接 `b9d3475` 的 H6 Ready 接管候选。本轮取得了真实 CLI 请求收益：三次独立进程中，首次长文请求中位数 **56.71→47.52 秒（减少16.2%）**，跨域返回长文 **40.50→26.70 秒（减少34.1%）**。五请求完整序列总耗时中位数 **145.36→118.71 秒（减少18.3%）**。这是同 H6 供给配置的改进，不能直接当作相对默认 GPU-only 路径的提升。

可直接运行的候选见 [q4e_supply_chat.ps1](../../perf/qwen4exp/q4e_supply_chat.ps1)，完整原始样本、输入/二进制/源码哈希、质量和 trace 证据见 [结果 JSON](qwen3.8-flash-next-supply-h7-results-2026-10-03.json)。默认 preset 未改变，32 GiB host pool 保持不变；100+ prefill / 20 decode 目标尚未完成。

## 1. 本轮实际改变

新增按阶段控制推测预取的 `AILA_Q4E_PREFETCH_DECODE` 和 `AILA_Q4E_PREFETCH_PREFILL`，两者默认均为1，并继续受原 `AILA_Q4E_PREFETCH` 主开关控制。使用 `prefill_remaining >= 0 || count > 1` 区分阶段，显式单行 prefill 尾片不会误用 decode 开关。需求加载仍正常执行，没有新增 kernel、量化格式、USM 热路径分配或编译参数。

在 H6 相同驻留热集、CPU 冷计算和 Ready 接管基础上，实测关闭两阶段推测预取、把在线 swap 预算设为0，并把 prefill chunk 从512增至2048。关闭在线 swap 仍保留 prompt 交接、借槽恢复和 Ready 接管，每个请求仍有约1.257 GB恢复流量；这里的 `off-static` 不表示热槽永远不变或没有 H2D。

新增顺序跑实验的 campaign、身份严格匹配的汇总器、旧 trace 的 refill 历史分析器，以及显式开启候选的 CLI launcher。launcher 临时清理与 probe 相同的运行参数前缀，注入已测配置，结束后恢复调用者环境，避免旧 hotset/fill/debug 参数污染结果。质量测试现在根据 swap 预算要求 epoch swap 增长或保持不变，原 KL/NRMSE、checkpoint、取消和内存要求保留。

## 2. 实验身份与样本口径

硬件：Intel Arc A770 16 GiB，驱动32.0.101.9033；Xeon E5-2696 v3，18个物理/逻辑核，DDR3。栈：oneAPI2026.1.0、oneDNN3.11.2、oneTBB2023.1。模型：`C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2`。

所有性能进程使用同一个 `Aila.exe`，SHA256：

```text
A4952A663206AEDCD07BDB32C5547577D43F8CF1A6AB826F2F8F6C413B9466AC
```

共同配置：`qwen38-a770` preset、max_seq2560、greedy、thinking_budget0、输出48、stream_chunk1、decode_chunk1、profile `qwen38-nf4-h4-v1`、总expert槽3567（decode热槽3551/服务槽16，prefill热槽3055/服务槽512）、borrow1、cold_queue2、resident_prefill1、CPU cold mode2/policy1/12线程/10 credits、prompt1/prompt_swaps0、Ready adopt1/limit4096、32 GiB host pool。质量工具使用同源码重建的独立测试二进制，主 CLI 哈希在测试头文件重建前后保持一致。

每进程固定顺序：story41 → archive1833 → code47 → archive1833 → story41；每次 `/clear` 清会话，模型和全局热集继续共享。保留启动后的首次请求、适应期和返回域请求。请求时间包括 prompt、生成、交接/epoch 维护，模型与 profile 初装在请求计时之外。真实 greedy 续写可能因 CPU/GPU贡献和路由变化而不同，JSON保留完整文本及哈希；独立 teacher forcing 用于数值验收。

主要三组各3进程，共9进程45请求；另保留3组单进程探索，共15请求。合计12进程60请求，全部退出0，没有 watchdog/memory-pressure 停止，没有删掉异常样本。汇总器检查 EXE、模型配置、输入历史、命令行、profile、stack/GPU、有效槽数及其他运行参数相同，只放行预取阶段、swap预算和明确允许的 chunk 差异。

| 组 | decode/prefill 预取 | 在线 swap预算 | chunk | 进程数 |
|---|---:|---:|---:|---:|
| all/512（基线） | 1/1 | 512 | 512 | 3 |
| off-static/512 | 0/0 | 0 | 512 | 3 |
| off-static/2048（本轮候选） | 0/0 | 0 | 2048 | 3 |
| prefill/512（探索） | 0/1 | 512 | 512 | 1 |
| off/512（探索） | 0/0 | 512 | 512 | 1 |
| off/2048（探索） | 0/0 | 512 | 2048 | 1 |

最初顺序为all512、prefill512、off512、static512、off2048、static2048；确认第2轮为static2048、static512、all512，第3轮相反。第一轮主要三组分散在探索阶段，仍全部保留；后两轮独立进程按逆序/正序确认。所有GPU压力任务串行运行，性能测量时没有并行构建或测试。

## 3. 请求、TTFT 与 decode

下表均为各位置三个独立进程的中位数，保持原顺序。单位：请求秒 / TTFT秒 / decode tok/s。

| 位置 | all/512 | off-static/512 | off-static/2048 |
|---|---:|---:|---:|
| 首次 story41 | 21.66 / 11.41 / 4.5 | 19.91 / 10.65 / 5.0 | 19.86 / 10.69 / 5.0 |
| 首次 archive1833 | 56.71 / 50.15 / 7.1 | 56.43 / 51.46 / 9.3 | 47.52 / 42.70 / 9.3 |
| code47 | 14.56 / 3.92 / 4.4 | 13.04 / 3.52 / 4.9 | 14.10 / 4.37 / 4.8 |
| 返回 archive1833 | 40.50 / 34.20 / 7.4 | 36.84 / 32.13 / 9.8 | 26.70 / 21.44 / 8.6 |
| 返回 story41 | 11.28 / 4.42 / 6.7 | 10.27 / 3.79 / 7.1 | 9.37 / 3.33 / 7.6 |

整段五请求耗时以每进程先求和再取中位数，未把每域中位数相加：

| 组 | 三个进程总耗时（秒） | 中位数（秒） |
|---|---|---:|
| all/512 | 134.00, 145.36, 146.13 | 145.36 |
| off-static/512 | 138.15, 143.77, 137.05 | 138.15 |
| off-static/2048 | 118.71, 112.74, 119.01 | 118.71 |

all/512的各位置请求CV均低于0.050，候选各位置低于0.041。static512返回story有一次约23.4秒、TTFT13.645秒的异常，CV约0.368，保留并标记variable；它不影响另外两组比较，也不能用该组三点中位数声称短输入稳定。

只改变窗口的 static512→static2048 对照，首次/返回长文请求减少15.8%/27.5%。但返回长文decode **9.8→8.6**，code请求与TTFT也较static512变慢，窗口2048不是每项指标都最佳。launcher默认2048面向本次长文请求延迟，允许 `-PrefillChunk 512` 做取舍。候选返回长文有效输入/TTFT约85.49 tok/s；没有达到100+/20，也没有接受HTTP、多会话、MTP或更长QSA稀疏区间。

## 4. 为什么这次请求真正缩短

长文窗口实际由 `512+512+512+293+4` 变为 `1829+4`，最后4行保留既有安全边界。减少重复访问专家供给的次数，效果比单纯改善热集计数更直接。计划提交字节的中位数如下（十进制GB，不是测量PCIe带宽）：

| 位置 / 流量 | all/512 | off-static/512 | off-static/2048 |
|---|---:|---:|---:|
| 首次长文 prefill需求 | 98.03 | 98.06 | 37.23 |
| 返回长文 prefill需求 | 99.55 | 97.89 | 37.27 |
| 首次长文 decode需求 | 2.08 | 1.52 | 1.46 |
| 返回长文 decode需求 | 2.02 | 1.50 | 1.65 |
| 首次长文 epoch/交接refill | 4.33 | 1.26 | 1.26 |
| 返回长文 epoch/交接refill | 4.35 | 1.26 | 1.26 |

预取与CPU冷计算都消耗 host backing、内存带宽和CPU资源；当前fill为7 worker加调用线程，CPU冷计算为12线程，而机器只有18个核。源码审阅发现matched prefetch项可能由主线程补齐，同时无关推测任务仍在运行；非全命中Acquire返回前还需取消并排空这些任务。关闭推测任务/在线swap的组合有实测净收益，但这并未证明所有模型和硬件都应关闭预取。

在线512 swap每4个forward重装，H6旧trace离线分析配对1134次retire/publish：此前窗口仅CPU观察848项、CPU+GPU244项、仅GPU0项、未观察42项。该分析只证明历史位置，不能证明换入浪费或未来GPU需求为零；成功GPU acquire还可能为缓存命中。保留这一边界后，实测零在线swap对照才支持本轮配置选择。

候选仍需每次长文供给约37 GB，CPU join和需求Acquire仍主导部分decode。普通trace显示长文诊断请求平均host forward约112 ms、Acquire约36 ms、CPU join约22 ms；这些区间相互嵌套/重叠，不能相加，也不是GPU纯kernel时间。后续有收益空间的是需求优先的供给任务、减少host搬运，以及按真实CPU替代/consumer等待成本决定付费GPU admission。

## 5. 质量、状态、资源和回归

代表性短41/长1833输入分别完成48步teacher forcing：

| 路径 | 短输入 | 长输入 |
|---|---|---|
| GPU冷计算 | 48/48逐位一致，NRMSE0，NLL差0 | 48/48逐位一致，NRMSE0，NLL差0 |
| CPU冷计算平均KL | 0.0015992 | 0.0026719 |
| CPU平均NLL差 | −0.0058690 | −0.0044932 |
| CPU最大logits NRMSE | 0.0563271 | 0.0596335 |
| CPU任务数 | 18527 | 12469 |

CPU路径通过既定容差门槛，无NLL预警，但不逐位一致。两组checkpoint/replay、取消恢复和live USM增长0检查通过。测试没有修改数值门槛。

launcher补验的检查脚本曾把日志前缀写为`[ExecutionConfig]`，实际为`[Q4E ExecutionConfig]`，导致最后一条配置断言失败；运行进程退出0、此前环境快照断言均已通过。已修正前缀并用保留日志完成复核，失败日志及复核日志一起纳入JSON，未排除性能样本。

汇总工具另完成8项CPU检查：接受同预算phase对照，拒绝命令行、profile、stack、GPU、在线预算、槽数污染及重复进程prefix；实际12进程的严格汇总通过。PowerShell AST与新增Python文件编译检查通过。

普通候选trace回放通过：30 forwards、5380次Ready接管、4次prompt交接、0在线epoch retire/publish配对，验证reader/generation/consumer顺序和槽位所有权；没有设备profiling，不能据此声称GPU重叠比例。6项CTest（ExpertReadiness、ExpertReader、CpuExpert、Generation、Module、ModelSpec）全部通过，15项已有Python回归通过。launcher DryRun与真实8-token smoke通过；额外注入旧HOTSET_SLOTS/FILL_THREADS/FILL_PARTS/DNNL_VERBOSE验证隔离和调用者环境恢复。messages-json smoke为30 token输入，只验证功能，性能结论采用上述interactive41/47/1833历史。

12个性能进程最高working set37.35 GiB，最低可用系统RAM11.33 GiB；最大device allocation ledger14,973,294,796字节（13.945 GiB）。device ledger不是DXGI dedicated/shared实测。chunk2048增加预分配arena，但本轮所有组总expert槽仍为3567，decode热槽3551/服务槽16、prefill热槽3055/服务槽512；没有通过降低热槽数换取窗口。更长输入和多会话内存行为尚未验证。

本轮没有重试H6同栈已失败的VTune instruction-count采集：改动为供给策略和已有窗口的重新测量，不涉及新kernel/codegen。失败记录保留在H6报告。

## 6. 使用与复现

进入项目目录后运行交互CLI：

```powershell
pwsh -NoProfile -File perf/qwen4exp/q4e_supply_chat.ps1
```

完整三轮对照顺序运行，仍保留所有首次和返回请求：

```powershell
pwsh -NoProfile -File perf/qwen4exp/q4e_supply_campaign.ps1 -WindowComparison -Pairs 3 -Tag h7-reproduce
```

读取一个JSON prompt可用 `-PromptFile`；使用较短窗口可加 `-PrefillChunk 512`。脚本不是默认生产dispatch，输出长度默认48，max_seq2560，单会话CLI。1024窗口参数可用于继续探索，本轮没有测量或宣称其性能。

接下来的实施优先级仍是需求供给和付费admission，保留R2b原型但继续暂停其生产集成。修复任务取消不能简单先调用CancelIoEx再drain：当前reader取消返回没有成功/部分读取状态，需要先建立成功/取消所有权，才能避免半读数据被发布为Ready。本轮未改这一安全协议。
