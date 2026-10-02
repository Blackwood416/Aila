# Strata S1–S2：KV-only 补齐与代表性窗口成本

2026-09-30，Intel Arc A770 16 GB，驱动 32.0.101.8860，oneAPI 2026.1 / oneDNN 3.11.2。此前累计 P0–P4/HC/MTP 与 Strata 评审已提交为 66e21e9。本轮新增改动基于该提交；专家算术全部在 GPU，host pool 仍为 16 GiB，Reader 仍为 ReadFile。

**S1 完成局部存储等价与收益检查；S2 完成四类文本的条件接受率、1–4 行成本和专家并集测量。S3 的收益前置条件未满足，不实现生产 MTP 状态提交。** 下一优先项是 target verifier 的专家执行与供给。20 generated tok/s 尚未达到。

原始逐位置/逐遍结果、输入 token、元数据、二进制/日志身份、阶段诊断及中断记录见[同名 JSON](qwen3.8-flash-next-strata-s1-s2-results-2026-09-30.json)。这些实验不等同于完整的 speculative generation 循环。

## S1：只补齐持久 K/V 和 indexer

Qwen4ExpMtpProbe::AppendTargetKv 复用现有 fused QKV、Q/K norm/RoPE 和 indexer 更新；完成持久缓存 append 后返回。仍有无用 Q 投影，第一版不改变算术边界。省去 attention、o_proj、后续 HC、router/shared/MoE 和输出 head；不会把中途的 hidden4 当作完整 draft hidden。

短输入和 512 输入分别检查 89 / 571 行目标对齐历史。所有 K/V、indexer 原始及压缩缓存位级一致，随后完整一步的 hidden/logits 位级一致。selected cycles 每一步的 logits 哈希也与原逐位置参考一致。KV-only 补齐没有访问专家，使用时 live USM 计数不增长；indexer 压缩四行边界包含在覆盖内。本测试范围仍为 dense-QSA、position <640。

两种路径按 AB/BA/AB 顺序测量，各三遍；每遍重新建立正确 MTP 历史，维持 64 个 MTP GPU slots、3520 个 target slots 和 3008 MiB reserve。相同 target greedy 轨迹仍为短输入 64 token、重复灯塔输入 64 token，分别选中 27 / 15 个周期。

| 输入 | 原完整补齐的 draft+catch-up ms/周期 | KV-only ms/周期 | 降幅 | 相同 target 验证成本组合后的 tok/s |
|---|---:|---:|---:|---:|
| 30-token 日出 | 23.418 | 21.319 | 9.0% | 11.69 |
| 512-token 重复灯塔 | 23.494 | 18.743 | 20.2% | 13.48 |

[MEASURED] 表内 draft 时间为提交到同步完成，不包含离线 logits D2H/hash/argmax；包括这些工作的 loop 时间在 JSON 保留。[DERIVED] 最后一列将独立阶段测量组合，未计 target 前缀提交、拒绝重放、服务采样或真实交替时的缓存干扰，不是生成吞吐。局部省算成立，但不支持单靠 draft 优化达到 20 tok/s。

既有两条 target/draft token 轨迹完全保留，短输入 52 个数值 reference 二进制产物与前轮一致。**此前整步 CPU reference 在位置 2/3 失败的 3% NRMSE 门槛仍未通过。** 本轮存储一致性不能替代该数值验收。

## S2：四类文本和窗口 1–4

固定 JSON 输入位于 perf/qwen4exp/prompts/mtp-{zh,en,code,prose}.json。render 后输入分别为 122、145、121、300 token。前三类生成 64 个 target greedy token、60 个 sliding anchor；非重复长文有效重跑生成 32 token、28 anchor。每个 anchor draft 三步，再按窗口 T=1/2/3/4 截取相应连续命中前缀，selected cycles 用 S1 的 target-aligned KV-only 补齐。每种窗口测三遍 draft 周期、四遍 target 工作，target 行宽按正反序交替。

target 测量重放同一条已知 greedy 轨迹，**没有输入错误 draft 后进行完整接受/拒绝状态机**。T=1 是相同驻留预算下的单行 target 工作。表内数据只适合检验条件收益空间；所有原始首遍与样本保留。

| 输入 | T=1 条件 tok/s | T=2 | T=3 | T=4 | 四行平均推进 token |
|---|---:|---:|---:|---:|---:|
| 中文分析 | 12.35 | 10.59 | 9.29 | 7.64 | 1.85 |
| 英文 memo | 10.58 | 10.42 | 10.25 | 9.24 | 2.70 |
| Python 代码 | 11.49 | 12.50 | 13.63 | 14.13† | 3.65 |
| 非重复长文，32-token 重跑 | 11.64 | 12.22 | 10.49 | 8.83 | 2.31 |

† 代码第一次 sweep 的四行耗时存在明显大幅波动，去首遍均值中位数为 495.10 ms/块，对应条件 7.05 tok/s。独立四行复测为 235.77 ms/块，对应 14.13 tok/s；其 target/draft token 轨迹一致。两组都保留，不能删除慢结果后声称稳定获得该吞吐，也不能把差额解释成代码优化收益。未隔离出这次墙钟波动的唯一原因。

长文首次完整 sweep 在仍持续输出日志时触发 240 秒 watchdog，exit=-1、未完成，未进入上表。最低可用 RAM 25.38 GiB，未触发 8 GiB 内存保护，没有设备丢失诊断。以较短续写在相同 240 秒限制内重跑完成；不得把较短样本外推为长续写稳定性能。

四行专家并集平均每层：中文 27.77、英文 28.31、代码 25.44、非重复长文 26.65。router 时 GPU cache resident 比例分别约 67.4%、56.4%、60.6%、60.9%；这是驻留观察，不表示对应 DMA 已结束。并集计数复用既有 histogram，观察模式有少量诊断开销。

同一窗口在各遍中的 logits 哈希一致；与单行 target 轨迹比较，中文的已知 63 个预测位置有 1 个 top-1 不同，英文有 2 个，代码和长文重跑没有不同。不能据此称普遍 greedy 等价或采样分布保持。原始 MTP 数值失败仍独立保留。

## 剩余 verifier 优先级与 S3 决策

另用中文前 24 个 target token、六个四行块做串行阶段诊断：[MEASURED] 总计约 280.0 ms/块，experts 182.8 ms（其中 acquire 67.8 ms）、HC 21.2、linear attention 27.4、full attention 13.7、router 阶段 22.9、shared 尾部 6.3、head 3.4。阶段强制同步会改变重叠，不能替代正常墙钟测速；router 阶段包括此前提交的 shared 投影，不是纯 top-k 耗时。acquire 已包含在 experts，不能重复相加。

专家执行与供给占这个诊断总时间约 65%，head 约 1%。优先研究 target 专家路径的实际等待、miss bytes 和小窗口 GPU 工作；不优先建立额外约 519 MiB 的 subset head，也不移植外部 GPU 的阈值。

[DERIVED] commit余量 = 50×平均推进token − verify − draft/catch-up。所有有效样本在把 commit/服务等成本设为零后仍达不到 20 tok/s；即使最有利的代码四行复测，每周期还差约 75.7 ms。独立阶段和条件接受率并非硬件上限，但目前没有足够证据支撑 S3 的大规模状态机投入。

因此按原规划的条件执行：**S3 暂缓，未实现 GDN tape、recurrence-only accepted-prefix commit 或生产 MTP。** 已有 device checkpoint 只证明恢复重放，没有被改称为前缀提交。其后的 S4 内存生命周期工作可独立继续。

有效 S1/S2 进程峰值 WS 21.47 GiB，系统最低可用 RAM 25.17 GiB；上述一次 watchdog 中断单列保留。没有新增 CPU 专家计算，没有更改生产 MTP 默认开关。

复现示例：

~~~powershell
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload mtp -MtpSlots 64 -MtpCompareKv -MtpAnchors 60 -MtpRepeats 3 -MtpVerifyPasses 4 -DemandMicrobatch 32 -Tag s1-mtp30
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload mtp -MtpSlots 64 -MtpWindowSweep -MtpPromptFile perf/qwen4exp/prompts/mtp-zh.json -MtpAnchors 60 -MtpRepeats 3 -MtpVerifyPasses 4 -DemandMicrobatch 32 -Tag s2-zh
python perf/qwen4exp/q4e_strata_summary.py tmp/s1-mtp30 tmp/s2-zh --output tmp/s1-s2-summary.json
~~~
