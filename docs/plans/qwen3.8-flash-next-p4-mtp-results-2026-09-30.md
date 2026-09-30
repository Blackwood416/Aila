# Qwen3.8-Flash-Next：GPU MTP 可行性探索

2026-09-30，A770 16 GB，驱动 32.0.101.8860，oneAPI 2026.1 / oneDNN 3.11.2。基于 0293b3b 和前轮未提交改动；计算保持 GPU 路线。

**本轮完成真实 MTP 权重执行、三步 draft 条件命中探测、两种显存分配策略，以及有界 device checkpoint 的恢复重放检查。当前候选不满足 decode 20 tok/s 的收益条件，未接入生产生成循环。** 64 槽策略优于全部驻留，但按测得成本组合的乐观估算仍只有约 11.4 / 13.3 tok/s。这不是已实现的推测生成吞吐。

## 模型语义和实现

本地 checkpoint 含 86 个 mtp.* 条目，存储载荷合计 1,378,709,721 字节，包含一个 full-attention MoE 层、三个 HC、fc_embedding 和 fc_hidden。共享主模型 embedding 和 LM head，没有另存两张大表。语义依据是本地 vLLM 3137ff07739f0cd19e06119506c6432a4a7e77ad 的 Qwen4Exp MTP；文件 SHA256、权重形状及分片 header 身份保存在结果 JSON。

这不是普通的 concat(embedding, hidden) MTP：

1. 输入主模型最终 HC mixer **之前**的 [4,2560] 隐藏状态；先在 10240 维整体做 GemmaRMSNorm，再对四路应用同一个 fc_hidden。
2. 新 token 的共享 embedding 经独立 norm 和 fc_embedding 后，加到每一路。
3. 运行一层 full attention + HC + MoE，PLE 关闭。最终 HC mixer 的单路输出用于共享 LM head，mixer 前的四路输出用于下一步 draft。
4. token ID 前移一位，position 保持 hidden 所在位置。MTP 有独立 KV；不读取主模型 KV。

实现位于测试专用 Qwen4ExpMtpProbe/Qwen4ExpMtpAcceptance，复用现有 packed NF4、HC、attention、MoE 算子。仅支持本次文本、位置 <640 的 dense-QSA 范围，不能外推至稀疏长上下文、多模态或服务并发。融合完成后释放多余源权重；只保留 Prefill 最后一行 logits，同时保存所需四路 hidden，避免为 512 个输入保留整段全词表 logits。

## 观测协议

短输入为 30-token 日出写作请求；长输入为 512-token 重复灯塔文本。主模型各生成 64 个 greedy token，遇 EOS 会停止。以这条真实目标轨迹建立 MTP 的目标对齐 KV，再在 60 个位置各递归预测三步。每种配置重复三遍，检查全部 logits 哈希重现、有限值和 live USM 计数。

另外按实际接受长度跳到下一个 anchor。每轮最多推进 1+连续命中数 个新 token；下一轮先用已确认的 target hidden 补齐 MTP KV。**这个 catch-up 成本已计入**，没有把前一轮近似 draft hidden 当作 target hidden 留在缓存里。所选轮次的接受长度与逐位置探针核对一致。

主模型验证在 MTP 权重/缓存仍驻留时单独测量：相同 greedy 轨迹、每块四行、共 64 token，重新 Prefill 后重复四遍。报告每遍总时间/块数，再取去首遍的三个值的中位数；JSON 保留所有首遍和逐块样本。验证输出重复哈希一致。host/GPU cache 没有强制恢复为相同快照，OS 文件缓存没有清空。

这里尚无生产接受/拒绝循环；命中参考来自单步 target，实际多行 target 的数值差异仍需最终集成时验证。未声称保持完全相同的 greedy 决策或采样分布。

## 显存取舍与成本

两种策略仅调整 MTP GPU LRU 容量和相应的显存预留，主模型和 MTP 的输入/预测 token 在对应配置间完全相同：

| 策略 | MTP GPU 专家槽 | MTP 专家池 | 主模型专家槽 | 额外 VRAM reserve |
|---|---:|---:|---:|---:|
| 全驻留 | 512 | 1237.5 MiB | 3069 | 1536 MiB |
| 小缓存 | 64 | 154.7 MiB | 3520 | 448 MiB |

全驻留加载末尾 device 计数增量约 1.29 GiB，额外 pinned host 约 1.37 GiB。小缓存分别约 239.3 MiB / 166.7 MiB；小缓存不启用额外 host expert pool，miss 经现有 ReadFile 和有界 staging 传输。两者都预触达全部 MTP 专家，再以 prompt 建立缓存。主模型 16 GiB host pool 不变，没有 CPU expert 算术。

[MEASURED] 同一输入下的条件式成本：

| 输入 / MTP 槽 | 四行 target 验证 ms/块 | draft + catch-up core ms/轮 | 平均推进 token/轮 | 乐观成本模型 tok/s |
|---|---:|---:|---:|---:|
| 30 / 512 | 192.71 | 15.74 | 2.33 | 11.19 |
| 30 / 64 | 181.91 | 23.47 | 2.33 | 11.36 |
| 512 / 512 | 313.55 | 19.10 | 4.00 | 12.02 |
| 512 / 64 | 278.09 | 23.39 | 4.00 | 13.27 |

模型计算为“所选轮次数 × 验证块成本 + 所选轮次 draft/catch-up core 总时间”，除以推进 token 总数。core 包含提交到同步完成，排除 logits D2H、CPU 哈希/argmax；含这些诊断工作的 loop 也保留在 JSON。还没有计入主模型回滚/前缀重放、服务、采样及真实交替执行的缓存影响，因此不能当作实际吞吐保证或严格物理上限。

减少 MTP 驻留量，长样本验证从 313.55 降至 278.09 ms，但 MTP 本身更慢。小缓存每遍 180 次 draft forward：短样本 608 次 miss、1.54 GB demand H2D，长样本 226 次 miss、0.573 GB；全驻留为零。长样本主模型每 64 token 的需求传输从约 31.0 GB 降至约 25.5 GB。此前 HC 报告的约 189 ms 来自**另一条固定 teacher-forcing 续写轨迹**，不能把本轮与它的差额全部归因于 MTP 或缓存容量。

## 接受情况与代表性

短样本在 60 个位置上，连续命中至少 1/2/3 步分别为 **38/24/14**；完整三步命中只有 23.3%。实际跳步选择 27 个轮次推进 63 token，平均 2.33；两种缓存策略的预测完全相同。即使不计 draft 成本，按本次四行验证成本也不足以支持 20 tok/s。

长样本 55 个位置三步全中、5 个位置首步失败。按起始位置和跳步顺序选择的 15 轮恰好全部三步命中，共推进 60 token。**它是在重复续写“灯塔—海洋—日出—日志”的容易文本，不能解释成一般文本的 91.7% 或 100% 接受率。** 尽管这个局部条件有利，主模型验证成本依然过高。

## 数值检查：保留失败，不更改阈值

独立 NumPy 数学参考逐个专家解量化、分块读取共享 LM head，仅用于离线校验。它遵守本机投影 BF16 权重舍入、融合 SwiGLU/MoE FP32 权重解量化及中间 BF16 边界；不替代生产 GPU 计算，也不是运行官方 vLLM 的整模型对齐。

前四个 MTP 位置中，四个 top-1 均相同，KL 最大约 0.00259。但原设的整步 hidden/mixed/logits NRMSE≤3% 检查在位置 2、3 **未通过**，保留 failed 状态和全部数据：位置 2 hidden NRMSE 12.26%、logits 9.03%。不能称为完整数值验收通过。

逐阶段定位结果：位置 2 的 Q/K NRMSE 约 0.146% / 0.318%，独立计算后 attention 输出差异扩大至 11.24%，随后 router 改变。使用**同一 GPU Q/K/V**再独立计算注意力，四个位置 NRMSE 为 0、0.0132%、0.0194%、0.0105%。位置 3 也受此前 K/V 小差异影响。证据支持 softmax 对舍入的放大，而不是直接把差异归因于 MoE 路由或通过放宽阈值消除它。采用 GPU route 的条件参考仍保留为诊断，未替代原失败检查。

释放融合源权重前后的短输入 reference artifacts 位级一致。由于上述敏感性和样本范围限制，当前结果仅用于评估本机 draft 候选，不作为 MTP 默认化质量证明。

## 有界 device checkpoint

测试专用单快照 arena 大小 **118,038,528 bytes = 112.57 MiB**，包括 36 层 GDN 状态、conv ring 和 PLE conv；同时保存位置、conv head、PLE token history、QSA 已完成块数等小型元数据。KV/indexer append 尾部由恢复的可见长度隐藏，不复制整段 KV 到主机。

在非四行对齐的 anchor **31 / 513** 捕获后，执行四行再单步；恢复后重放，四行 logits、随后单步 logits 和四路 hidden 全部位级一致。捕获/恢复各 73 次 D2D，3 次 warmup 后 20 个样本，中位数各约 **1.2 ms**，live USM 不增长。无每候选 112 MiB D2H host snapshot。

**这是整块之前的单快照恢复，不是接受前缀提交已完成。** 部分接受时仍需从该快照重放接受前缀，或进一步实现受预算限制的 token 内部状态保存；这些计算和可能增加的显存未包括在复制时间内。没有实现服务取消、采样 RNG 或多模态场景的完整推测状态机。

## 结论和后续选择

当前 NF4 MTP 的三步方案有两类问题：正常短写作样本的连续命中不足；容易长样本的 target 验证/专家供给成本过高。64 槽折中是更好的实验起点，但没有达到投入完整生产 MTP 状态机的收益条件。暂不继续堆叠接受前缀提交工程，不修改默认开关。

下一阶段若继续 MTP，应先降低同轨迹、同预算的 target 验证成本，并用更多非重复文本测接受情况；不能凭重复文本的高命中率承诺 20 tok/s。现有 draft、缓存取舍、CPU 数学参考和 device checkpoint 探针保留，供后续方案复用。中长 Prefill 100+ tok/s 的前轮 canary 结论没有因此变为默认性能保证。

本轮有效 GPU 探针全部退出 0，无 watchdog 停止。最终四组进程峰值 working set **22.64 GiB**，系统最低可用 RAM **25.11 GiB**，未触发 8 GiB 保护。构建通过；没有重复全量 CTest。原始失败的 CPU 数值门槛按上述说明保留。改动尚未提交。

复现：

~~~powershell
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload mtp -MtpSlots 512 -MtpAnchors 60 -MtpRepeats 3 -MtpVerifyPasses 4 -Tag mtp30
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload mtp -MtpSlots 64 -DecodePromptTokens 512 -DemandMicrobatch 32 -MtpAnchors 60 -MtpRepeats 3 -MtpVerifyPasses 4 -Tag mtp512-small
python perf/qwen4exp/q4e_mtp_summary.py tmp/mtp30 tmp/mtp512-small --output tmp/mtp-summary.json
python perf/qwen4exp/q4e_mtp_reference.py tmp/mtp30 --model C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2 --output tmp/mtp-reference.json
~~~

全部环境/二进制身份、每遍及逐轮样本、条件命中、数值失败、缓存策略、恢复成本与源文件 SHA256 见 [结果 JSON](qwen3.8-flash-next-p4-mtp-results-2026-09-30.json)。
