# MTP：窗口同步、草稿缓存与 checkpoint

2026-10-06，Intel Arc A770 / driver 32.0.101.8724 / oneAPI 2026.1，Xeon E5-2696 v3。模型 C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2。沿用原 CLI system prompt（1833 tokens）、128 outputs（Anchors=124）、CPU18 A64 mapped cold、冻结 hotset、T2 async、state-only replay、decode attention、NF4 projection reuse=3、decode prepare=1、draft hidden reuse=1、GPU greedy。

**保留批量 greedy 与无用尾部拷贝清理；四项失败候选全部撤回。** 本轮保留路径最好一次为 19.94968 tok/s（6416.1437 ms），最后同 EXE control 为 19.73595 tok/s（6485.6267 ms）。还未达到完整循环 20+，也未解决 T1/T2 长轨迹分歧。

本轮采用小改动、聚焦正确性检查和同 EXE 快速筛选。每组独立比较，不将跨 EXE 或不同时段的差异全部归因于代码。完整 decode 循环包含首轮、草稿、checkpoint、验证和拒绝恢复；prompt→输出还包含目标 prompt 和 draft KV 初始化，但不含模型加载、tokenization、IPC 或探针预分配。

## 已保留：合并选择与删除无用尾部拷贝

验证结果的 greedy 归约支持 B1/B2/B4：每行独立 scratch，两次 GPU 提交覆盖所有行，一次回传所有结果。保持最低 token ID tie 规则和 BF16 非有限值检测。仅在后续还需要 draft hidden 时拷贝它，删除阻塞选择之后的重复同步；width=1 尾部仍同步。

两项均在 LIVE 探针中，未集成生产 scheduler。MtpGreedyRows、MtpSkipDraftTail 默认关闭。原 GPU greedy、投影复用和 state-only replay 继续保留。

| 同 EXE 运行 | 完整 tok/s | 循环 ms | prompt→输出 ms |
|---|---:|---:|---:|
| 修正后的 T1，跳过 draft KV init | 18.72088 | 6837.2869 | 16954.5837 |
| T2，原逐行选择和尾部拷贝 | 19.10276 | 6700.6022 | 17985.9462 |
| T2，批量选择 + 删除尾部拷贝 | **19.91958** | **6425.8384** | **17665.8030** |

T2 pair 的完整 generated、advance 数组和 hotset identity 相同。直接观测到暖窗口 draft 减少约 0.104 ms，选择区间减少约 0.321 ms；forward 等区间也有变化，因此不能把整个 +4.28% 都归因于这两项同步改动。选择区间含排队中的 GPU 工作，不能当成纯传输时长。

T1 的 draft_kv_init_ms 现在为 0。它仍和 T2 共用相同的 MTP 权重、reserve 和 checkpoint 驻留布局，是共享驻留条件下的比较，不是完全剥离 MTP 的生产 T1。原 T2 相对该 T1 的循环收益仅 +2.04%，请求增加 1031.36 ms；优化后 T2 的请求仍慢约 711.22 ms。

聚焦测试 tmp/mtp-window-sync-unit.log 通过，覆盖 B2/B4 行隔离、跨工作组 ties、每行非有限值与 canary。

## 已撤回的三项尝试

| 尝试 | control → candidate tok/s | 循环变化 | 结论 |
|---|---:|---:|---|
| T2 pending MLP residual + norm | 19.89568 → 19.72361 | +56.13 ms | -0.865%，撤回 |
| 近似固定总显存，draft cache 64→128 | 19.94968 → 15.38379 | +1904.30 ms | 热集受损，撤回 |
| GDN 保存首 token 状态，缩减 checkpoint/replay | 19.89156 → 19.62932 | +85.97 ms | GDN 写回开销超过节省，撤回 |

pending residual 的 B1/B2 H、N 逐位检查通过，实际 token/advance 相同，但性能下降。临时运行时、内核、runner 和测试入口均撤回。原 generic fused 测试允许 1 ULP 的注释已纠正，不再误称 bitwise。候选存于 tmp/mtp-pending-rows-kernel.patch、tmp/mtp-pending-rows-backend.cpp，测试日志 tmp/mtp-pending-rows-unit.log。

draft128 增加 154.69 MiB，reserve 3008→3168，target slots 3418→3352（hot slots 3402→3336），总 GPU 分配略减约 4.8 MiB。草稿 misses 431→246，传输 1,092,326,400→623,462,400 bytes，暖 draft 7.5253→6.8430 ms；但 target CPU tasks 24397→26670，暖 verify 77.3776→99.0873 ms。输出从 index 16 就不同，hotset identity 和 advance 也不同。它是失败的缓存分配策略筛选，不能当作相同生成轨迹的算子 A/B。Mtp128 支持和 reserve mapping 已撤回；runner 仍只接受原有 64/512。

首 token 状态方案保持原 conv/PLE 快照，由 GDN 写出 row0 的 FP32 matrix，拒绝时恢复该 matrix 并提交 conv prefix。局部测试验证最终 T2 输出/状态、首 token matrix 和 conv prefix，与原 replay 逐位一致，覆盖两种 gate。完整 pair 的 generated、advance、hotset 也一致，但仍变慢：

| 暖完整窗口均值 | control ms | 首 token 状态 ms |
|---|---:|---:|
| verify | 77.5749 | 79.0759 |
| checkpoint+input | 1.28894 | 0.68697 |
| commit/replay | 0.79573 | 0.50489 |

checkpoint_copy_bytes 从 8,498,774,016 降至 345,047,040，但该计数是 capture 的 payload，未包含 restore 或新增的 GDN 写出；不能据此声称总内存流量下降同等比例。所有首状态内核、运行时、runner 和测试入口已撤回。候选保存在 tmp/mtp-prefix-state-{candidate.cpp,probe.hpp,unit.hpp}，测试日志 tmp/mtp-prefix-state-unit.log。

## 已撤回：批量 checkpoint 字节拷贝

保持原来的 pre-window snapshot 和 replay 语义，将 73 个跨度、118,038,528 bytes 的保存/恢复由多次 queue.memcpy 合为一次 SYCL 提交。描述符只在请求前分配、上传，SG16 / WG256，每跨度 64 groups，uchar16 分段拷贝；没有浮点计算。测试的 16、160、4096、122880、3145728 bytes 五个跨度，capture/restore 的字节与前后 canary 均完全一致，USM 不增长（tmp/mtp-state-copy-unit.log）。

| 同 EXE 快速筛选 | 原 memcpy | 批量 kernel |
|---|---:|---:|
| 完整 tok/s | **19.73595** | 18.59351 |
| 完整循环 ms | **6485.6267** | 6884.1228 |
| prompt→输出 ms | **17756.2520** | 18156.0531 |
| 暖 checkpoint+input ms | **1.303831** | 1.479332 |
| 暖 verify ms | **78.237162** | 83.767032 |
| 暖 commit/replay ms | **0.809207** | 0.892880 |
| 暖 draft ms | 7.579896 | 7.589296 |

候选吞吐 -5.789%，循环多 398.4961 ms，请求多 399.8011 ms。两次 EXE SHA256 均为 802E5724E0501BA8B8C47D8F9672795DF722083AFBDA4286D59B35C8D444826A，实际驱动均为 8724。generated、advance 全数组完全相同，hotset 首尾 identity 都是 7625625710176325289；draft misses 都为 431，target CPU tasks 都为 24397，LIVE USM delta 都为 0。

此处 checkpoint 区间用原 proposal H2D 阻塞边界计时，没有新增 synchronize；它不是纯 kernel device time。pair 中 verifier 也变慢，尚无有效 GPU 时间线，不能将全部退步精确归因于拷贝带宽、cache 污染或 GPU idle。该实现未展现收益，快速筛选后停止，不继续扫 group/vector 参数。

候选 helper、probe、测试、两个 runner 的完整副本保存在 tmp/mtp-state-copy-candidate/，原始结果前缀为 tmp/mtp-8724-state-copy-{live128,control128}。当前源码已删除 kernel、测试入口及 MtpBatchCheckpoint 开关，恢复原 memcpy 循环；保留 checkpoint_copy_bytes 的 capture payload 记录。

当前保留路径的复现命令：

~~~powershell
pwsh -NoProfile -File tmp/mtp-current-screen.ps1 -Tag NEW-window -Prompt archive-cli-system -Anchors 124 -VerifyCpuRows 4 -VerifyGpuReuse 4 -MtpLiveWidth 2 -MtpStateOnlyReplay 1 -Verify 0 -VerifyDecodeAttention 1 -VerifyAsyncRoute 1 -VerifyLinearReuse 3 -VerifyDecodePrepare 1 -MtpHiddenReuse 1 -MtpSelectProfile -MtpGpuGreedy 1 -MtpGreedyRows 1 -MtpSkipDraftTail 1
~~~

切回原选择对照将 MtpGreedyRows、MtpSkipDraftTail 同时置 0。T1 用 MtpLiveWidth=1。所有实验开关保持 opt-in。

## 边界与下一步

标准 64-slot T2 仍与 T1 在 output indices 123–127 不同。已保留改动保持原 T2 轨迹，但这不等于无损 MTP。此前相同 head 输入的 B2/B1 logits 一致，GDN 单行参考只消除了一个诊断窗口的 top1 差异；全模型长轨迹并未对齐。数值问题见 qwen38-mtp-greedy-8724-2026-10-06.md。

大头仍在 target verifier：暖窗口约 77 ms，draft 约 7.5 ms，checkpoint+input 约 1.3 ms。简单增加 draft cache 抢占 target hotset 已证实得不偿失，首状态写回也没有兑现流量节省。后续优先寻找 verifier 的具体复用或提交成本，同时推进 B1/B2 首次数值分歧定位；不继续堆叠这些已失败分支，也不以去掉首轮的速度宣称完整 20+。

只运行相关构建、聚焦 kernel 检查和有限 LIVE 筛选，没有全 CTest 或服务器验收，未改变生产默认配置。原始日志、EXE hash、实际驱动、完整 generated/advance 数组与对照关系由 tmp/mtp-window-cache-report.py 汇总到同名 JSON；tmp/mtp-verifier-report.py 读取实际 metadata，而非写死驱动。

撤回后的 AilaQwen4ExpToleranceTests、AilaQwen4ExpKernelTests 目标重新构建成功，日志 tmp/mtp-window-final-build.log，exit=0；相关 tracked diff 的 whitespace 检查通过。11 次有效 LIVE 都是 1833-token prompt、8724 驱动、exit=0、USM delta=0。失败分支的聚焦测试通过只证明其检查范围内正确，不能推翻实际性能退步。
