# Qwen3.8-Flash-Next：32 GiB decode 预算验证

**后续本机性能探针采用 32 GiB 热专家 RAM 池。** 长上下文 decode 获得收益，但有一次明显回落；短输入没有进一步收益。32 GiB 是下一阶段实验基线，不是所有负载的最优配置或达到 20 tok/s 的声明。

本轮承接 [16/24 GiB 对照](qwen3.8-flash-next-ram-budget-results-2026-09-30.md)。设备与软件保持 Arc A770 16 GB、驱动 32.0.101.8860、oneAPI 2026.1 / oneDNN 3.11.2、约 64 GiB RAM，模型为 `C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2`。GPU-only 算术、ReadFile、NF4 权重和二进制未变。身份、逐遍数据及计算见 [JSON](qwen3.8-flash-next-ram32-decode-results-2026-09-30.json)。

## 方法

三个场景分别按 **32 → 24 → 24 → 32 GiB** 启动独立进程。同一场景只改变 `AILA_Q4E_HOT_RAM_MB`；GPU 槽数、输入、生成长度、其余环境变量和二进制哈希一致。每次请求清空会话，保留模型和专家缓存。全部为真实 CLI greedy 生成，stream/decode chunk=1、thinking budget=0，无 trace / stage profile。未清空 Windows 文件缓存，首遍单列，不把启动差异算成主要收益。

| 场景 | 实际输入 / 生成 token | 每进程请求数 | max-seq | GPU 专家槽 | Prefill chunk / phase scratch |
|---|---:|---:|---:|---:|---|
| 短 story | 41 / 48 | 4 | 512 | 3714 | 512 / 关闭 |
| 自然故事续写 | 307 / 128 | 4 | 512 | 3714 | 512 / 关闭 |
| 长日志背景续写 | 2146 / 64 | 3 | 2560 | 3567 | 512 / 开启 |

自然故事用 `mtp-prose.json`，长场景用 `ram-long.json` 的合成日志背景。CLI 输入换行替换为空格，原文件和实际 stdin 均保存哈希；307-token 场景不能与此前 MTP 的 300-token 格式混算。长场景经过 2052-token QSA 边界。

全部 12 个进程、44 次请求的输出长度符合预期，每个场景跨预算及各遍文本 SHA256 相同。CLI 回调数逐次核对等于实际生成 token 数。TPS 显示精度 0.1、TTFT 精度 1 ms；首个输出之后到请求返回的区间包含末尾上下文对齐。预算变化没有改变算术，本轮不代表完整模型质量评估。

## 结果与波动

[MEASURED] 取去首遍样本：短/自然故事每预算六个，长日志每预算四个。

| 场景 | 24 GiB decode 中位数 | 32 GiB decode 中位数 | 合并解码估算变化 | 完整请求中位数，24 → 32 GiB |
|---|---:|---:|---:|---:|
| 短 story | 14.8 | 14.6 | -2.2% | 4.867 → 4.897 s |
| 自然故事续写 | 12.2 | 12.35 | +1.3% | 14.552 → 14.952 s |
| 长日志背景续写 | 5.45 | 6.7 | +3.9% | 57.880 → 51.488 s |

[DERIVED] 每个场景输出长度固定，按耗时合并 TPS 用各次显示速率的调和平均估算：`n / sum(1/tps)`，不是算术平均。它受显示舍入影响，不是额外的高精度计时。长场景约 5.450 → 5.661 tok/s，因此不能把中位数 +22.9% 宣传成包含所有波动的稳定整体增益。

短 story 的后续样本：24 GiB 为 14.7、14.8、14.8 / 14.5、14.8、14.8；32 GiB 为 13.9、14.6、14.8 / 13.9、14.6、14.7。存在约 2% 小幅回退，主要在前面的重复请求。自然故事两预算最终一遍均为 12.6 tok/s，主要差别发生在预热过渡期。

长日志原始值：

| 独立进程 | 第一遍 decode | 第二遍 decode | 第三遍 decode |
|---|---:|---:|---:|
| 32-a | 6.3 | 6.9 | **3.8** |
| 24-a | 5.5 | 5.5 | 5.4 |
| 24-b | 5.5 | 5.5 | 5.4 |
| 32-b | 5.9 | 6.7 | 6.7 |

32-a 的回落原样保留，没有剔除。反向复测 32-b 的两个后续样本均保持 6.7 tok/s。四个后续样本中三个获益、一个明显退化。长场景 TTFT 中位数 46.524 → 39.345 秒；32 GiB 四个后续完整请求全部快于 24 GiB 的四个样本，中位数缩短 11.0%。按用户希望扩大预算继续推进的取向，采用 32 GiB 实验基线，并保留波动问题。

本轮没有 GPU residency、硬件事件计时或缓存 trace，不能把波动归因为已证实的 WDDM 换页、RAM 不足、热降频或某个 kernel。后续需定位长上下文的等待和驻留状态，并区分固定八步尾段与实际生成、无快照 Prefill 与 CLI 可复用前缀路径。

## 预算落地

峰值进程工作集约 29.22 → 37.27 GiB；32 GiB 组最低可用 RAM **10.775 GiB**。12 个进程串行执行，全部 exit=0，未触发 240 秒 watchdog 或 8 GiB 可用 RAM 停止线。

`q4e_execution_probe.ps1` 的 `HostRamMB` 默认改为 **32768**，仍可显式指定 16384/24576。新增生成 prompt 和 max-seq 参数。MTP 探针默认对齐此前 S1–S6 已选的 **64 个 draft 槽**，避免旧 512 槽默认在较大 RAM 预算下另启整池主机缓存。本轮未测 32 GiB 下的 MTP 收益；后续仍需完整周期测量。生产运行时的独立默认值未修改，本机研究流程由 probe 环境覆盖应用 32 GiB。

```powershell
# 当前探针默认 HostRamMB=32768；24 GiB 对照显式指定。
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload generation -GenerationRepeats 4 -MaxTokens 128 -GenerationPromptFile perf/qwen4exp/prompts/mtp-prose.json -DemandMicrobatch 32 -Tag ram32-prose-example
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload generation -GenerationRepeats 3 -MaxTokens 64 -GenerationPromptFile perf/qwen4exp/prompts/ram-long.json -GenerationMaxSeq 2560 -PrefillChunk 512 -PhaseScratch -DemandMicrobatch 32 -HostRamMB 24576 -Tag ram32-long-control
```

本轮只改变探针、输入 fixture 和报告，无新内核或运行时构建。Python / PowerShell 语法、摘要身份/输出验证、Git whitespace 检查通过。未重复无关 CTest 或服务并发验收。
