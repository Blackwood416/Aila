# MTP verifier：阶段定位与 attention 小步迭代

日期：2026-10-06。A770 16 GiB / oneAPI 2026.1 / Xeon E5-2696 v3、CPU18，沿用当前 NF4/A64、CPU mapped cold、GPU expert 4/2/1 权重复用和 state-only rejection replay。

**原场景仍未达到 decode 20+。** 找到并验证一项长提示词收益：验证窗口的 dense full attention 可以复用现有单行 decode 内核。原 1833-token 场景 T2 完整循环达到 16.68 tok/s，但同进程 T1 为 17.28。短代码提示词退化，因此开关默认关闭，没有作为通用默认路径启用。

## 比较口径

- 原 CLI 文本已折叠换行并添加默认 system，输入为 tmp/mtp-archive-cli-system.json，实际 1833 tokens；没有使用另两个 1822/1838-token 变体替代。
- LIVE 包含实际 draft、全词表 greedy 选择、checkpoint、verification、拒绝恢复；不包含 prompt prefill、draft prompt KV 初始化或服务 IPC。
- 每个性能筛选生成 64 tokens。阶段诊断使用 32 tokens。一次只运行一个 GPU 进程，构建和计时分开。
- 下表是快速单次筛选，旧记录和新记录来自不同进程/构建，不作为多轮统计认证。带同步/日志的诊断单列，不能充当无诊断基线。
- “暖完整宽度”同时剔除首循环和不足标称宽度的收尾循环。post-first TPS 只剔首循环，保留收尾，二者不可混用。

## 实际循环结果

| 配置 / 场景 | T1 完整 / post-first | T2 完整 / post-first | T4 完整 / post-first |
|---|---:|---:|---:|
| 上轮原 1833-token 场景 | 17.18 / 18.12 | 14.10 / 14.67 | 13.02 / 13.45 |
| 本轮 decode attention，原场景 | 17.28 / 18.27 | 16.68 / 17.50 | 14.34 / 14.85 |
| 本轮另加工作组交错，原场景，已撤回 | 17.42 / 18.40 | 16.90 / 17.74 | 14.62 / 15.14 |
| 上轮代码样本 | — | — | 19.76 / 20.64 |
| 本轮 decode attention，代码样本 | — | — | 18.96 / 19.72 |

单位为 tok/s。以上本轮各 LIVE 均与各自同进程的单行 target 轨迹 64/64 相同。attention 算法改变了内部舍入，原场景 T4 接受轨迹有所改变：暖完整宽度平均推进由 2.6364 变成 2.5217；不能按固定接受率推算最终速度。

工作组交错的对照来自同一个二进制，只切换该开关，无额外诊断。收益仅约 1–2%，T1 也有约 0.8% 波动；已撤回这项代码和 runner 参数，不继续为小差异扩大测试。

## 成本边界

tmp/mtp-verifier-boundary：仅 LIVE 边界同步和 WorkSummary，backend 的 Profile 当时被 runner 的 MTP 分支强制关闭；这个 runner 问题现已修正。以下是原场景暖完整宽度均值：

| 区间 | T1 ms | T2 ms | T4 ms |
|---|---:|---:|---:|
| checkpoint + input | 0.260 | 1.310 | 1.335 |
| target forward，含 head | 56.065 | 112.140 | 149.999 |
| logits D2H，含 hidden copy 的队列尾部 | 0.408 | 0.778 | 1.408 |
| CPU finite 检查 + lowest-index argmax | 0.546 | 1.100 | 2.186 |

checkpoint 和选择合计无法解释几十毫秒差距，因此没有新写 argmax 内核。

相同诊断方式开启 decode attention 后：T2 forward 为 95.381 ms，T4 为 140.776 ms。T1 为 55.954 ms。这里不把 CPU compute 与 join 相加：二者相互重叠，且还与 GPU 执行重叠。

原路径 WorkSummary 的 T2/T4 暖完整宽度统计：

| 计数/区间 | T2 | T4 |
|---|---:|---:|
| CPU 逻辑 token-rank tasks / forward | 334.36 | 668.18 |
| CPU compute 累计 ms | 22.32 | 41.53 |
| 主线程 CPU join 累计 ms | 19.08 | 35.67 |
| worker activation wait 累计 ms | 7.29 | 6.35 |
| CPU result H2D bytes / forward | 3,423,884 | 6,842,182 |

这些数据没有独立量出 CPU/GPU overlap、逐层冷专家去重数或 GPU 尾部。不能由累计 compute/join 直接声称异步路由能节省多少时间。

较重的 tmp/mtp-verifier-stages 在每阶段同步，明显改变了执行重叠。其 full-attention 阶段 T1/T2/T4 约 6.85/30.08/35.18 ms，linear attention（含投影）约 13.46/21.01/31.92 ms，head 约 3.4 ms。这些仅用于选择优化点；它的吞吐不是正式基线，也没有将整个 linear-attention 区间误标为 GDN 内核耗时。

## 保留的修改

- src/models/Qwen4ExpPrefill.cpp：AILA_Q4E_VERIFY_DECODE_ATTN=1 时，在 2–4 行、decode 阶段、dense QSA、无 interior snapshot、未启用 window/sink 的条件下，逐查询复用 attention_decode。每行可见长度为 start_pos + row + 1，沿用 fused gate 和已分配的单行 scratch；in-order queue 保证 scratch 复用安全。没有新增 GPU kernel 或热路径 USM 分配。prompt prefill 和 sparse QSA 继续使用已有路径。
- tests/models/Qwen4ExpMtpProbe.hpp：AILA_Q4E_MTP_LIVE_PROFILE 可分别记录 checkpoint/input、forward、D2H 与 CPU argmax；输出首循环后的热集 identity 和循环结束 identity。
- perf/qwen4exp/q4e_execution_probe.ps1：增加对应参数 -VerifyDecodeAttention、-MtpLiveProfile；修复 MTP 工作负载忽略 -Profile 的问题。

两个新增开关均默认关闭。短代码样本已经测到退化，没有凭两个上下文样本发明自动长度阈值。长提示词实验显式开启即可。

## 正确性、热集和验证

- 原场景各宽度首循环后和结束时 FrozenIdentity 均为 7625625710176325289，跨宽度也一致；decode swaps=0。代码样本 identity 为 16984313793670183643，运行前后相同。
- ObserveHotsetRoute 仍观察验证窗口的所有行；这不等价于已实现 accepted-only 策略统计。当前被测热集没有变化，不能据此推及启用 adaptive swaps 的生产场景。
- 本轮所有有效 LIVE：64/64 token parity，live USM delta=0。
- 最后一次代码样本还执行了已有 T1–T4 固定轨迹 sweep 和设备 checkpoint/replay 检查；replay_bitwise=true，118,038,528-byte checkpoint，USM delta=0。这里只保证同宽度回放的 bitwise，一致 token 不代表跨宽度全部 logits bitwise 一致。
- 最终构建通过：tmp/mtp-verifier-final-build.log；修改文件 git diff --check 通过。没有跑全 CTest、服务并发套件或生产 scheduler 集成。
- 先前 1838-token 变体的跨宽度数值差异仍是遗留问题，本轮没有宣称解决。

## 剩余差距

保留 attention 修改后，原场景暖完整 T2：推进 1.8485、draft 8.041 ms、verify 96.777 ms、commit 0.650 ms。达到 20 tok/s，verify 需约 83.7 ms，还差约 13.0 ms。

T4：推进 2.5217、draft 23.148 ms、verify 142.298 ms、commit 3.922 ms；20 tok/s 留给 verify 约 99.0 ms，还差约 43.3 ms。当前 T2 更接近盈利线，但仍未胜过 T1。后续值得检查的仍是多行 MoE 的 host route/dispatch、CPU join 与 result-copy 临界路径，不是继续扩大草稿宽度或削减已很小的 checkpoint/恢复。

## 复现与资料

原场景保留路径：

~~~powershell
pwsh -NoProfile -File tmp/mtp-current-screen.ps1 -Tag NEW-archive -Prompt archive-cli-system -Anchors 60 -VerifyCpuRows 4 -VerifyGpuReuse 4 -MtpLiveWidth 4 -MtpLiveCompareWidths -MtpStateOnlyReplay 1 -Verify 0 -VerifyDecodeAttention 1
~~~

边界诊断增加 -MtpLiveProfile -WorkSummary；重阶段诊断单独增加 -Stages，勿混用其吞吐作为基线。机器可读记录见同名 JSON；包含上轮参考、本轮六个进程的实际环境和摘要。源日志及 metadata 均保留在 tmp/，没有覆盖先前报告。
