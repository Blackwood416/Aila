# 当前异构版本的 MTP 小规模复测

结论：值得继续做 verifier 的 CPU 冷专家接入，尚不支持直接开启完整 MTP。代码样本的 draft 接受长度足够好；当前 2–4 行 verifier 仍走 GPU 冷专家搬运，无法继承单行 CPU NF4-Q8 的收益。

## 本次范围

A770 / driver 32.0.101.9033，oneAPI 2026.1，Qwen3.8-Flash-Next-BNB-NF4-v2；当前 hotset、HostRam0、CPU18/A64/row2/async/light-route、GDN history 配置。64 个 MTP GPU 专家槽，额外约 239.3 MiB device / 166.7 MiB pinned host；target 3520 总槽、3504 hot 槽，reserve 3008 MiB。MTP 自身仍使用 64 槽 GPU LRU。

复用现有 MTP probe，每个样本 28 个 anchor、32 个 target greedy tokens，一遍 draft、每种行宽一遍 verifier，含首块成本。代码 prompt 121 tokens、中文 prompt 122 tokens。仅用于初筛，没有广泛质量或重复性能结论，也没有把此前 archive1833 的 17.6 tok/s 直接代入这里。

## 发现与修正

1. 热集管理把 rows>1 隐式当作 prefill，导致验证重新收集 prompt、借出热槽。新增默认关闭的 ForwardOptions.decode_verification，让明确的多行 decode 完成 prompt handoff、保留 decode 热集；传统调用默认行为不变。
2. 修正阶段后暴露暂存容量问题：decode 只有 16 个 cold staging 槽，旧 verifier 仍使用 32 个专家的 prefill 批次。验证的供给批次现限制在实际 staging 容量内。

仅修阶段的实模尝试 mtp-phase-code-screen 以 Cold batch exceeds frozen staging capacity 失败，未当作有效性能样本。修复容量后代码、中文探针均完整退出 0，未触发 watchdog。readiness 套件通过，覆盖多行验证的首次 handoff、连续 2/3/4/1 行窗口不借槽，以及后续显式 prefill 仍可借槽。

代码样本修复前后 target tokens、draft tokens、每种行宽的全部 verifier logits hashes 一致。修复让执行阶段正确，但不是性能优化：固定 decode 热集与较小 staging 下 GPU 冷专家路径更慢。普通单行生成没有启用这个新选项。

## 测量结果

平均推进量为 1 + 连续接受 draft 数；T=4 含 pending token 和三个 draft。

| 样本 / 行宽 | 平均推进 token/轮 | draft + KV 补齐 ms/轮 | verifier ms/块 | 20 tok/s 留给 verifier + 其他开销的预算 ms |
|---|---:|---:|---:|---:|
| 代码 T=2 | 1.933 | 7.16 | 241.56 | 89.51 |
| 代码 T=4 | 3.875 | 20.36 | 373.23 | 173.39 |
| 中文 T=2 | 1.611 | 6.74 | 202.88 | 73.82 |
| 中文 T=4 | 2.071 | 19.38 | 327.52 | 84.19 |

公式：预算 = 50 ms × 平均推进量 − draft/catch-up 时间。其他开销包括接受决策和目标状态提交，因此 verifier 自身必须进一步低于此预算。代码 T=4 的旧隐式 prefill 探针为 316.16 ms，而非修复后的 373.23 ms；两种都达不到目标。

代码三步全命中 23/28 个 sliding anchors；中文为 4/28。按实际跳步选择轮次后得到表中推进量，不能将 23/28 直接作为逐 token 接受率。中文 T=3 和 T=4 都只推进 2.071 token/轮，T=4 多花约 5.91 ms draft 时间。

当前代码四行验证处理 32 tokens 仍产生 13.001 GB demand H2D，单行对应为 0；CPU 冷专家仅在单行路径工作。这是下一步最明确的接入缺口。CPU NF4-Q8 batch 算子已支持 1–4 tokens / 40 tasks，但 mapped cold 来源、按 rank 合并和 resident 调度尚未接入这个多行 verifier。

同预算 T=1 verifier 为代码 61.70 ms/token、中文 58.49 ms/token（包含首步 handoff）。阶段组合的条件估算仅约代码 T=4 9.85 tok/s、中文 T=4 5.97 tok/s。它们不是实际推测生成吞吐；尚未实现错误 draft 下的完整接受/拒绝、accepted-prefix commit 和交替执行循环。本次没有重跑旧的 MTP 独立 CPU 数值参考，其历史 NRMSE 失败记录仍保留。

## 下一步

先接两行 verifier 的 CPU mapped NF4-Q8 冷专家，沿用现有 batch 算子和预分配缓冲，不先扩展完整 MTP 状态机。代码样本以 verifier 接近 90 ms 为初步收益门槛；随后考虑高接受率的四行窗口。中文低接受率时应允许单行回退。上述预算是这两个短样本的派生值，不是所有任务的 dispatch 阈值。

复现入口：tmp/mtp-current-screen.ps1；有效日志前缀：mtp-current-code-screen（原探针）、mtp-phase-cap-code-screen、mtp-phase-cap-zh-screen。原始汇总：tmp/mtp-current-final-summary.json；构建和测试：tmp/mtp-phase-final-build.log、tmp/mtp-phase-final-readiness.log。每轮 meta 保存 EXE/config hash、有效环境和内存计数。
