# Qwen3.8-Flash-Next：128 步 decode 基线与小批量验证路径

实验日期：2026-09-29 至 2026-09-30。基线为 `0293b3b` 加前几轮未提交改动。A770 16 GB、Windows 驱动 32.0.101.8860、oneAPI 2026.1 / oneDNN 3.11.2。计算仍全部在 GPU；ReadFile 和 16 GiB host expert pool 保持原配置。

本轮完成 P3 的固定续写测量，并为 P4 实现了默认关闭的小批量 NF4 投影候选。**单请求 decode 20 tok/s 尚未达到。** 中长 Prefill 100+ tok/s 的前轮结论及其设置、适用范围不变，见 [P2/P3 结果](qwen3.8-flash-next-p2-supply-p3-front-results-2026-09-29.md)。

新增 `--decode-model` canary：30-token chat 或 512-token lighthouse 输入，完整重做 Prefill，再消费相同的 128 个固定 continuation token。分别执行 128 次单行、64 次两行或 32 次四行 forward，每个输入 token 都产生全词表 logits。不是 128 个生成 token，也没有把 Prefill 的首个输出计入 decode。多行调用仅测完整主模型验证工作，尚无 MTP draft、接受/拒绝或状态回滚。

`core` 是每块 forward 提交至 GPU 完成的 wall 之和；`loop` 另外包含全词表 D2H、有限值检查、哈希及 CPU 参考 argmax。采样、HTTP/CLI 和 checkpoint 不包含在内。每轮检查全部 128 行 logits 的有限性和重复哈希，以及最终 context length。预留 token/logits 空间，解码前后 live USM 计数不变；这不是累计 allocation 计数，不能据此排除分配后立即释放的瞬时行为。旧日志中的 `new_usm_allocations` 字段已在汇总器中更正为 `live_usm_count_delta`。

[MEASURED] 30-token prompt；dense XMX=1、fill_parts=8、B6 关闭。每配置一个进程，首次完整轨迹单列，之后三次重复轨迹取中位数：

| 执行路径 | core ms/输入 token | loop ms/输入 token | core ms/块 |
|---|---:|---:|---:|
| 原有单步 decode | 70.26 | 71.41 | 70.26 |
| 单步关闭预测预取 | 69.55 | 70.69 | 69.55 |
| 原有两行验证 | 89.90 | 90.88 | 179.81 |
| 小批量 NF4 两行候选 | 64.11 | 65.09 | 128.23 |
| 原有四行验证 | 58.37 | 59.26 | 233.48 |
| 小批量 NF4 四行候选 | 46.07 | 46.97 | 184.28 |

关闭预取只有约 1% 的差异，全 128 行 logits 位级一致，但不是充分的默认变更证据，保持原设置。基线每 128 步约 8352 次 GPU miss、21.17 GB demand H2D；新候选没有显著降低这些传输量。原有单步首次轨迹 core 为 130.01 ms/步，首次 Prefill 为 6145.98 ms。首进程与后续进程的 OS 文件页状态不同，不能用首次轨迹相减计算优化收益。

512-token prompt 用 B6=32、max_seq=640，随后同样消费 128 个 token。首次采集后增加独立进程复测，各执行五次完整轨迹，明确报告前两次过渡和后三次：

| 路径 | 前两次 core ms/输入 token | 后三次 core ms/输入 token | 后三次中位数 |
|---|---|---|---:|
| 单步 decode | 132.20 / 82.87 | 76.85 / 75.83 / 75.57 | 75.83 |
| 小批量 NF4 四行候选 | 101.11 / 55.51 | 50.15 / 50.11 / 49.76 | 50.11 |

去掉首次轨迹、保留所有其余四次的中位数分别是 76.34 / 50.13 ms，JSON 同时保留这两种统计，未隐藏过渡样本。单步复测前两次 Prefill 为 17823.05 / 13530.38 ms，之后约 4.5 秒。因此这里是固定工作量的重复轨迹基线，**没有强制把 host/GPU/OS cache 状态恢复成完全相同的快照**，也不是新进程首请求的速度保证。

VTune 对两行路径各采集两次完整轨迹，保留第二次的 63 个完整块，丢弃 collector 缺失尾部事件的最后一块；导出采用高分辨率并检查重复事件。原有路径约 71.29 ms/块花在 NF4 XMX 投影，候选改为现有 packed NF4 `ForwardDecodeRows`，其对应 kernel 约 17.76 ms/块。记录到的 kernel union 均值从 136.71 降至 65.21 ms，copy union 仍约 36 ms；应用任务覆盖之外的时间约 32.33 / 40.69 ms。后者不是硬件 GPU idle。完整诊断 span 为 190.17 / 141.55 ms，说明不能直接把局部 kernel 时间差当作整模型收益。

实现仅增加 `AILA_Q4E_SMALL_ROWS_DECODE=1`：对 2–4 行 linear-attention/full-attention/shared-expert 的 NF4 投影选择既有多行 packed 算子。未更改权重、专家 cache、GDN、HC、attention 或 LM head 算术，也没有新增计算 kernel。默认关闭。它会影响所有启用开关后的 2–4-token 小段，不限于本测试；当前整模型测过 2/4 行、context≤640，3 行只有投影算子检查，不能将其提升为普遍生产分发规则。

在 `DecodeRowProjectionTests` 中补充五组 Q4E 真实形状（K→N）：2560→16480、6144→2560、2560→13312、2560→1280、640→2560。独立 double CPU 点积误差界通过；B=1/2/3/4/1 与原有 B1 投影位级一致，输入、输出和未用容量哨兵通过，释放后 USM 归零。这证明该投影算子的行为；整模型多行执行的数值变化仍需单独衡量。

[MEASURED] 相对于同输入逐 token 路径的全词表输出：

| 输入 / 路径 | top-1 相同 | mean KL | 最大 logits NRMSE | 127 个已知 teacher target 的 mean ΔNLL |
|---|---:|---:|---:|---:|
| 30 / 原有 2、4 行 | 126/128 | 0.002392 | 0.07927 | -0.01086 |
| 30 / 候选 2、4 行 | 124/128 | 0.002754 | 0.09295 | -0.00771 |
| 512 / 候选四行 | 123/128 | 0.006384 | 0.11381 | -0.01159 |

这些固定 teacher-forcing 观察不是 MTP 接受率。候选 2/4 分块的全 128 行彼此位级一致，独立进程长输入复测也一致。离线 NLL 的 tokenizer 输入 ID 哈希与 C++ 实际输入一致，仅用已消费 ids[1..127] 评分 logits[0..126]，排除未知的最后一个 next target。负 ΔNLL 仅表示该小段文本上目标损失较低，不能外推成质量提升。用户已允许合理容差，少量 top-1 改变不是自动否决条件，也不恢复旧的逐字一致要求；但不能声称当前批量路径提供完全不改变单步 greedy 决策的推测解码。默认化仍需更广文本的质量与状态验收。

P4 的预算现在可以具体计算。短输入四行块 core 约 184.28 ms，若全部四个 token 都提交，留给 draft、验证决策和 rollback/commit 的总余量最多约 **15.72 ms**；即使这些成本为零，每块也必须平均提交 **3.686** 个 token 才能达到 20 tok/s。512 输入的稳定验证块约 **200.43 ms**，零额外成本、全接受也已略超 200 ms。两行候选 128.23 ms 大于其 100 ms 预算。这些是乐观成本边界，不是实现后的吞吐或对接受率的预测。

下一步继续减少 2–4 行验证的剩余 HC/oneDNN 提交与算子成本，并按既定容差定位单行/多行数值分歧。之后再做有界的 GPU MTP draft acceptance 和 device state 回滚验证；每个候选都必须把额外权重挤占 expert cache、实际提交 token 数和恢复成本计入。当前结果不支持直接把普通多行 forward 包装成已完成的 MTP。

非 profiler 进程峰值 working set **21.27 GiB**，系统最低可用 RAM **24.85 GiB**，未触发 8 GiB 可用内存保护。所有有效模型/投影采集退出 0、无 watchdog 停止。相关 CTest 3/3 通过（Tolerance、ExpertReader、DmaTiming），投影 CPU/B1/guard 检查另行显式执行。没有更改默认 XMX、B6 或小批量开关。改动尚未提交。

完整每步时延、logits 哈希、输入/二进制/环境身份、内存采样摘要、VTune 摘要和测试输出见 [结果 JSON](qwen3.8-flash-next-p3-decode-p4-verify-results-2026-09-29.json)。复现示例：

```powershell
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload decode -Tag decode30
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload decode -VerifyRows 2 -Tag verify2-base
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload decode -VerifyRows 2 -SmallRowsDecode -Tag verify2-small
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload decode -VerifyRows 4 -SmallRowsDecode -Tag verify4-small
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload decode -DecodePromptTokens 512 -DemandMicrobatch 32 -DecodeRepeats 5 -Tag decode512
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload projection -Tag projection-q4e
python perf/qwen4exp/q4e_decode_summary.py tmp/decode30 tmp/verify2-base tmp/verify2-small tmp/verify4-small --output tmp/decode-summary.json
```
