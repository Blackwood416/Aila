# Qwen3.8-Flash-Next / A770：story 端到端回退排查

> **后续状态：已定位并修复。** `f24dd34` 将遗漏的 QSA 首次编译提前到 load；提交记录报告 GenerateMessages 7403→6241 ms。此前 router_sync、缓存策略等仅为候选归因，不是根因；下文保留当时证据，不再执行其回退消融路线。下一阶段见 [异构执行与数据路径设计](qwen3.8-flash-next-heterogeneous-next-stage-2026-09-27.md)。本注记未新增性能复测。

日期：2026-09-27。触发：用户观察 `484b086` 曾低于 7000 ms，而后续版本回到 7000 ms 级别。

本轮只读历史日志与提交代码，没有构建、GPU 测试、修改运行时或打断 worker。读取时 HEAD 为 `7a0c3ce`，worker 有未提交改动；本文的新性能样本是 **`06c67a6` 的记录，不是当前工作树的验收结果**。沿用 A770 skill 的证据分级。

## 1. 结论与验收口径纠正

**已有日志支持“端到端发生明显变慢的观测”，主要增量在短 prefill；不支持“所有非实验性改动都失败”，也还不能确定致慢提交。** 后续工作的优先级应从增加优化方案转为恢复可重复的 story 基线。

之前的计划和验收应明确拆成三项：语义/生命周期正确性、实现完成度、实际性能收益。“非实验性”最多描述语义风险较低，不代表调度、缓存、线程或同步策略必然加速。算子测试通过和代码可接受，均不能给端到端性能背书。

`484b086` 是用户提供的候选性能锚点，可以用于后续对照；下面约 6387 ms 的旧日志时间早于该提交，不能冒充它的实测成绩，也不能据此宣布它是全历史最快版本。

## 2. 现有日志已经说明什么

输入为 story，以下日志均记录 Prompt=30、Generated=48，打印的回复文本相同。文本相同不是 token ID、logits 或长上下文精度一致性的证明。

| 指标 | 旧样本 `s4_baseline_measured` | 新样本 `test_w4` | 差值 |
|---|---:|---:|---:|
| GenerateMessages 耗时 | 6386.75 ms | 7217.25 ms | +830.50 ms，约 +13.0% |
| 首个 prefill，26 token | 2217.11 ms | 3208.90 ms | +991.79 ms |
| 第二个 prefill，4 token | 439.27 ms | 365.60 ms | −73.67 ms |
| 两段 prefill 合计 | 2656.38 ms | 3574.50 ms | **+918.12 ms** |
| 总耗时减去两段 prefill | 3730.37 ms | 3642.75 ms | −87.62 ms |
| decode 均值，跳过前 4 步 | 74.94 ms/token | 73.12 ms/token | 约 −2.4% |
| GPU 专家槽位 | 3778 | 3715 | −63 |

“总耗时减 prefill”仅是算术余量，包含 decode 和外层开销，不是独立测量的纯 GPU decode 时间。两段 prefill 来自实际日志，不把初始化 `count=2` warmup 算入请求。

旧样本不是孤立的最低点：`s4_enabled_measured` 为 6387.11 ms，`e1_warm` 为 6370.84 ms。但它们缺少完整构建与环境身份，仍不能当成严格受控 A/B。

第一段 prefill 的细节进一步缩小范围：

| 指标 | 旧样本 | 新样本 | 解读 |
|---|---:|---:|---|
| acquire 计时 | 1449.17 ms | 1483.43 ms | 只增加 34.26 ms，不能独自解释约 992 ms |
| fill 子计时 | 1129.90 ms | 1017.84 ms | 反而下降 |
| prefetch wait 子计时 | 274.23 ms | 399.69 ms | 上升，但属于 acquire 范围，不能重复相加 |
| hit | 299/4680 | 299/4680 | 相同 |
| xfer | 10588.8 MB | 10588.8 MB | 相同；是日志统计量，不等于物理总线全部流量 |
| router_sync | 1.07 ms | 372.54 ms | host 等待归属变化，不能直接说 router 算子慢了 371 ms |

异步反量化、实际路由和预测路由的等待位置都发生过变化，in-order 队列上的 router event 可能承担此前计算的等待。各项 host 计时也不是互不重叠的 GPU 时间分解。只能把“首段 prefill 的计算、提交和等待安排”列为重点嫌疑，不能通过上表相减认定具体算子根因。

## 3. 哪些改动值得查，哪些暂不能判失败

1. **首先检查真实短 prefill 执行路径及等待点。** 对比 `484b086`、`06c67a6` 和 worker 完成后的固定版本：dense projection/反量化提交、actual/prediction route event、shared expert 排队、consumer fence 发布与等待。`06c67a6` 中 baseline 逐 token MoE 也会记录 consumer fences；这些安全依赖不能粗暴删掉，需判断是否有重复事件管理或过宽等待。现有 acquire 数据尚不支持把 fence 等待当成约 1 秒的主要来源。
2. **默认缓存策略是可隔离候选，不是已确定根因。** `AILA_Q4E_COST_AWARE_EVICTION` 默认 true，最多检查 8 个候选；预取排序、淘汰选择和时机发生变化。先用现有开关做单因素对照。槽位减少可能影响驻留，但该样本 decode 更快，首段 hit/xfer 相同，不能只凭少了 63 个槽位解释回退。
3. **A2 K-only 会进入短 prompt 路径。** `AILA_QSA_K_ONLY_PROJ` 默认 true。减少投影输出维度不保证当前 oneDNN/反量化实现的实际延迟同比下降，应检查 dispatch 与形状，不先假定它必赚或必亏。A1 的 compressed K 维护也可能进入短路径，需与仅在长序列生效的稀疏选择分开。
4. **长上下文稀疏 attention 无法用 story 判断收益。** 30 输入加 48 输出远低于首次需要丢弃完整 block 的 2052 可见 token 边界。长上下文 QSA 修正、精确 Top-K 等有独立的正确性价值。
5. **`test_w4` 标签不能证明启用了 W4。** sidecar 明确记录 `baseline (token-by-token)`；该提交中 expert-major、rows、dense XMX 等开关默认关闭，demand microbatch 也依赖 expert-major。该日志没有验收两 token 权重复用或 XMX 的性能。有效环境记录仍不完整，最终以 dispatch 为准。
6. **P0 PLE / P1 GR 晚于这份日志。** 不能将 `06c67a6` 的 7217 ms 归咎于它们，也不能用实现完成说明它们已改善端到端。等 worker 完成后再固定构建比较。

## 4. 下一步：有限对照，而非大规模验证

此处是待执行方案，本轮未启动任何实验。

**第一步：固定两个可识别的构建。** worker 完成后，保留其成果；另用隔离的构建目录准备 `484b086` 对照，不在正在工作的目录 reset/revert。明确相同模型、story 内容、max-seq=512、48 输出、greedy、thinking-budget=0、GPU/CPU 负载条件与 profiling 开关。记录运行前的 exe SHA、源码 revision/dirty 状态、有效 `AILA_*` 环境、prompt hash、实际 dispatch 与缓存容量。现有脚本在运行后取 HEAD/哈希，且只保存传入的 EnvVars，不足以证明二进制来自该 HEAD 或没有继承环境变量。

**第二步：只做一个有上限的配对。** 每个构建采用相同预热流程，先各取一次可比的 warm story 计时；若仍出现约 0.8–1 秒差距，再逆序各补一次排除顺序影响。最多四次计入比较的请求，复用同一进程/模型热态必须双方一致，不扩大 prompt/参数矩阵。若差距不稳定，先保留“不确定”的结论，不急着做优化归因。主指标用整个 GenerateMessages 及两段 prefill，decode 作次指标；模型加载/进程启动时间单列。

**第三步：只有确认差距后才挑一项消融。** 优先基于首段 prefill 的实际 dispatch 和已有 timing 选择候选，先做一个已有开关的单因素对照；例如默认 cost-aware eviction，或 K-only projection。不要一次关掉所有优化，否则无法定位。若开关不覆盖真实嫌疑，再围绕 A 系列前后、B 系列前后做粗粒度提交二分，不逐提交全量跑测试。

`AILA_PROFILE_Q4E_STAGES` 会在每个 stage 后同步，改变原有重叠；如确需一次定位，只把它作诊断，不能将它的端到端成绩和普通路径比较，也不先引入逐层全同步的测量补丁。

## 5. 默认策略与后续 brainstorm 的收敛

- **保留正确性修复，性能策略分别验收。** 不为了 story 的低延迟恢复错误的长 prefill mask、非精确 Top-K 或不安全的 cache/DMA 生命周期。旧版本只是性能参考，不能自动成为生产回滚目标。
- **没有明确收益的性能策略先保持候选。** 增加线程、事件、buffer、预取和缓存启发式，会消耗 CPU、内存预算及提交时间；代码量或消除一个局部等待都不是成功指标。story 是必须守住的短请求目标；长上下文优化另用其目标场景证明收益，并标注短请求代价。
- **先解释首段 prefill 增量，再扩展路线图。** 当前最有价值的新工作是恢复“约 6.x 秒如何达成”的可重复解释，而不是再加入一批理论合理的默认优化。此前 reference brainstorm 中 PLE/GDN/QSA 等仍作为候选池，暂不因实现完成升级为性能结论。

## 6. 本轮证据索引

- `tmp/q4e_prof_s4_baseline_measured.log`：2026-09-26 12:55，早于 `484b086` 当日 22:03 的提交时间。
- `tmp/q4e_prof_s4_enabled_measured.log`、`tmp/q4e_prof_e1_warm.log`：另外两份约 6.37–6.39 秒记录。
- `tmp/q4e_prof_test_w4.log`、`tmp/q4e_prof_test_w4.meta.json`：2026-09-27 14:05；sidecar revision=`06c67a6bc7ac90def7ea5eb07eac7b6ccef08109`，exe SHA256=`B69C8786D1A76D0B2FB669D2686C68F2B8CF8B64600B2B340E0E3B0AEADDACF5`。
- `perf/qwen4exp/q4e_prof.ps1`：计时筛选和 sidecar 身份边界。
- `git show 06c67a6:src/models/Qwen4ExpBnb4Backend.cpp`、`git show 06c67a6:src/models/Qwen4ExpExpertCache.cpp`：历史版本实际分支与默认开关；避免用 worker 正在变化的代码解释旧日志。
- `docs/plans/qwen3.8-flash-next-a770-acceptance-2026-09-26-evidence.json`：旧日志汇总及原有证据限制。
