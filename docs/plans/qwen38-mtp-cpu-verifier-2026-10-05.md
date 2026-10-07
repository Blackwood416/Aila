# Qwen3.8 MTP CPU verifier iteration — 2026-10-05/06

结论：MTP 在高接受率代码样本中已达到约 20 tok/s，但原 archive-natural 场景没有突破 20。当前不适合默认开启固定宽度 MTP。保留收益明确的执行路径和实际生成探针，撤回无明确收益的 CPU 内核实验。

环境：Windows / Arc A770 16 GiB / driver 32.0.101.9033 / oneAPI 2026.1 / Xeon E5-2696 v3，CPU 18 线程。模型为 C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2。本轮均为筛选测量，不是三轮稳态认证。

## 实际生成结果

以下计入草稿、greedy 选择、目标验证、设备 checkpoint、拒绝恢复；排除 prompt prefill、MTP prompt KV 初始化和服务 IPC。每组生成并提交 64 个 token。后一列只剔除第一个循环，不能与原 CLI 的正式吞吐定义混用。

| 场景 / 路径 | 完整循环 tok/s | 剔除首循环 tok/s | 与同进程单 token 输出不一致数 |
|---|---:|---:|---:|
| 代码，T4，整段前缀重放 | 16.74 | 17.19 | 0/64 |
| 代码，T4，只重建循环状态 | 19.76 | 20.64 | 0/64 |
| 原 CLI 提示词，T1 控制组 | 17.18 | 18.12 | 0/64 |
| 原 CLI 提示词，T2，状态重建 | 14.10 | 14.67 | 0/64 |
| 原 CLI 提示词，T4，状态重建 | 13.02 | 13.45 | 0/64 |
| 原 CLI 提示词，T4，投影改用矩阵乘 | 10.79 | 11.07 | 0/64 |

代码样本为 121 prompt tokens。原 CLI 对照为 **1833 prompt tokens**：折叠原 JSON 用户内容中的换行，并加入引擎默认 system message “You are a helpful assistant.”。T1/T2/T4 在同一个进程中分别重建 prompt 后测量；它们都保留了 MTP 权重及显存预留，不等同于不加载 MTP 的历史 17.6 tok/s CLI 基线。当前测量仍是探针，未接入生产 scheduler/server。

原场景的实际瓶颈：

| 宽度 | 循环数 | 平均前进 token | 拒绝循环 | 草稿平均 ms | 验证平均 ms，剔除首循环 |
|---|---:|---:|---:|---:|---:|
| T2 | 35 | 1.829 | 5 | 8.46 | 115.24 |
| T4 | 25 | 2.560 | 18 | 23.42 | 161.78 |

按目前平均前进量，20 tok/s 允许 T2 每循环约 91 ms，T4 约 128 ms；验证本身已经超过预算。T4 扣除草稿和恢复后，需要把验证压到约 100 ms。继续增加草稿宽度不能自动解决这个差距。

## 保留的实现

1. Qwen4ExpExpertExecutor::ExecuteVerifyRows：2–4 token 验证直接复用冻结 GPU hotset 和 CPU mapped NF4-Q8/A64 冷专家。复用原有 40-task worker、输入、输出及 descriptor 缓冲；按原 token/rank 顺序合并。GPU lease 和 host reader fence 覆盖异常退出、全 GPU 层及 CPU 结果拷贝。
2. GPU 常驻专家按同一专家的 4/2/1 个 token 分组，调用已有 expert-major 内核共享权重读取，没有新写 GPU kernel。
3. 实际 MTP 循环：执行真实草稿、接受/拒绝及前缀恢复，不再把分别测量的 draft/verify 成本当成生成吞吐。
4. 状态重建：验证时保存各层 GDN 已投影输入，拒绝后从设备 checkpoint 用已有 GDN kernel 重建接受前缀；保留验证得到的 causal hidden/logits，恢复 PLE 接受行和 KV 可见长度。额外输入/PLE 缓冲约 6.19 MiB，设备 checkpoint 为 118,038,528 bytes。没有增加大规模 token-state 历史矩阵。

代码样本 64-token 的累计拒绝恢复成本从 **580.75 ms 降到 34.88 ms**；单次典型恢复由 90–115 ms 降至 4–7 ms。这是本轮实际循环的主要收益。

最初的 32-token 固定轨迹代码测量：

| 验证路径 | T2 平均 ms | T4 平均 ms |
|---|---:|---:|
| 正确 decode phase，GPU 冷专家搬运，上一轮 | 241.56 | 373.23 |
| CPU mapped 冷专家 | 118.34 | 170.14 |
| 再启用 GPU 双/四 token 权重复用 | 111.80 | 157.06 |

均包含首块。CPU 路径把这些验证窗口的冷专家 demand H2D 降到零；原 T4 约 13.0 GB / 32 tokens。GPU 权重复用前后，代码样本各宽度的所有 verifier logits hashes 相同。条件估算约 21.8 tok/s 不代表实际生成结果，实际结果以首表为准。

## 验证与边界

- 最初 2/3/4-token CPU 路径通过原 A64 shadow 逐位对照；实际模型 rollback replay 逐位一致，未见 live USM allocation 增长。
- 代码 64 tokens、中文 32 tokens以及最终 1833-token prompt 的 T1/T2/T4 对照，均与各自同进程单 token 生成结果一致。中文 T4 状态重建仅 10.42 tok/s，低接受率仍限制收益。
- 更早两个长提示词变体不是原 CLI：保留段落换行、无 system 的版本为 1838 tokens；只折叠换行、无 system 的版本为 1822 tokens。二者存在长输出分歧，不能引用为原 CLI 对照结果。
- 1838-token 固定轨迹下，T2/T3/T4 在预测行 35、60（零起始）有 top-1 差异。关闭状态重建、使用完整前缀重放，实际输出仍会偏离单 token。这说明分歧不只来自新恢复路径；也不能据此宣称长上下文已全面数值对齐。
- 探针仅允许容量 640 或 2048，并检查不超过模型的 dense-QSA indexer budget。没有实现稀疏 QSA MTP、生产接受 token 的 hotset 计数、scheduler 接入或并发服务。

## 撤回或不采用的尝试

- 提前 CPU 输入拷贝：T4 170.14 → 170.04 ms，没有明确收益，已撤回。
- CPU 双 token 共享 NF4 权重解码：独立 scalar oracle 覆盖 1–4 token / 40 tasks / 1、4、12 threads，h/out 逐位通过；代码完整循环 19.76 → 19.99 tok/s，约 1% 的单次变化不足以保留新内核。已撤回，补丁存于 tmp/mtp-cpu-tokenpair-experiment.patch。
- 小批量投影改用矩阵乘：1833-token 原场景 T4 13.02 → 10.79 tok/s，保留多行 GEMV 默认。
- 首版实际循环逐 logit 构造错误字符串，造成大量 CPU 分配，已修复。code-live / zh-live 两个最早日志保留，但不作为优化后吞吐。
- mtp-cpurows-code-pair64 在新 CPU pair 内核启用日志出现前发生 oneAPI “urEventWait must not be called for an internal event”，退出 -1073740791；没有有效性能结果。同配置独立复查成功，原因未查明，不把这次失败归因于 CPU pair 数学。

## 重现与材料

工作区现有 runner 继承当前 CPU18/A64、冻结 hotset、无 host pool、无 prefetch 基线：

    pwsh -NoProfile -File tmp/mtp-current-screen.ps1 -Tag REVIEW-code -Anchors 60 -VerifyCpuRows 4 -VerifyGpuReuse 4 -MtpLiveWidth 4 -MtpStateOnlyReplay 1 -Verify 0
    pwsh -NoProfile -File tmp/mtp-current-screen.ps1 -Tag REVIEW-archive -Prompt archive-cli-system -Anchors 60 -VerifyCpuRows 4 -VerifyGpuReuse 4 -MtpLiveWidth 4 -MtpLiveCompareWidths -MtpStateOnlyReplay 1 -Verify 0

tmp/mtp-archive-cli-system.json 由原 archive-natural 内容按 CLI 规则折叠换行，再前置默认 system message 得到。原始 .meta.json 保存完整实际 env、prompt hash、EXE hash、退出码和 watchdog 数据。

- 主要结果：tmp/mtp-cpurows-code-state64.*、tmp/mtp-cpurows-archive-matched-widths.*、tmp/mtp-cpurows-archive-matmul64.*。
- 初版 A64 对照：tmp/mtp-cpurows-shadow.*。
- 最终构建：tmp/mtp-final-build.log。
- 全部本轮结果和失败记录：[JSON](qwen38-mtp-cpu-verifier-2026-10-05.json)。可运行 python tmp/mtp-iteration-report.py 更新。

实验默认关闭：AILA_Q4E_VERIFY_CPU_ROWS=0、AILA_Q4E_MTP_LIVE_WIDTH=0、AILA_Q4E_MTP_STATE_ONLY_REPLAY=0。本轮没有改成生产默认 MTP，也没有宣称原 20+ 目标完成。
