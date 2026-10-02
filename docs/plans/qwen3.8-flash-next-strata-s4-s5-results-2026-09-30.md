# Strata S4–S5：阶段临时区复用与专家供给门槛

2026-09-30，A770 16 GB、驱动 32.0.101.8860、oneAPI 2026.1 / oneDNN 3.11.2，基于 66e21e9 之后的工作区。先完成 [S1/S2 和 S3 条件判断](qwen3.8-flash-next-strata-s1-s2-results-2026-09-30.md)，再验证本工作包。保持 GPU-only 专家算术和 16 GiB host pool。

**S4 已实现默认关闭的临时区复用候选，在测试形状上减少 77–700 MiB GPU USM，未改变 Prefill/后续 decode 的 logits。S5 完成大块覆盖率与传输预算筛选，不直接移植 stream-all。** 尚未实现动态借用专家槽、自动 chunk/cache/ring 分配器或生产 MTP。

数据分别见 [S4 JSON](qwen3.8-flash-next-strata-s4-results-2026-09-30.json) 和 [S5 JSON](qwen3.8-flash-next-strata-s5-coverage-2026-09-30.json)。新实验代码及这些报告尚未另行提交；先前累计改动的提交为 66e21e9。

## S4：一个 arena，五个不同时存活的 view

AILA_Q4E_PHASE_SCRATCH=1 在 runtime capacity 增长时分配一个经 Context/USM ledger 记账的 arena，以下 Tensor view 都从其起点开始：

| 临时区 | 最后消费者 | 可以复用的原因 |
|---|---|---|
| full_qkv | packed prepare 或 split、norm/RoPE/KV append | attention 消费独立 Q/K/V 或持久 KV；不再读取原始融合输出 |
| linear_all | 所有 GDN/conv 段 | GDN 输出 attn_out 独立；o_proj 后不再读取投影输入 |
| scores / incr_scores | dense attention | start_pos=0 与 !=0 的分支互斥；attention 输出独立 |
| moe_down_acc | ordered weighted accumulation | 结果写入独立 y，随后 HC 与下一层不再读取累加区 |

这些消费者都按已有同一 in-order compute queue 顺序执行。PLE、预测、router、DMA staging、专家槽、持久 GDN/conv/KV/indexer、hidden4、x/y 和 attn_out 保持独立。没有取消或删除 reader/DMA fences。arena owner 在所有 view 之前声明，析构时后释放；增长前显式 drain，防止旧地址仍被 kernel 或 oneDNN 使用。decode 达到既有 capacity 后直接返回，不增加动态 USM 分配或新的 descriptor 绑定。

本轮仅改变存储生命周期，不改变量化、算术、kernel 或权重布局。下一层预测及 PLE 异步准备没有引用上述 alias 区域。将来如果 S3 保存 linear_all tape，必须在其最后消费者前另存需要的行，不能把 arena 当成跨阶段持久存储。

## 验证与统一预算记录

三个独立基线/候选对照都用相同二进制、模型、cache slots、XMX=1、B6=32 和固定灯塔 token；每个进程完整 replay 四遍，保留首遍和每遍数据。在每次 Prefill 后执行八个固定 decode token。

| 输入 / chunk | target slots | 基线 device ledger GiB | arena GiB | 减少 MiB | Prefill 末行与八个 decode logits |
|---|---:|---:|---:|---:|---|
| 512 / 512 | 3713 | 13.425 | 13.349 | 77.469 | 位级一致 |
| 1024 / 1024 | 3676 | 13.727 | 13.482 | 250.938 | 位级一致 |
| 2052 / 2048+4 | 3602 | 14.704 | 14.020 | 700.375 | 位级一致 |

第三项实际执行 2048-token chunk、接着四行，再 decode 至 position 2059，覆盖 QSA 的 2051/2052 切换区间。不是一次 2052-row GEMM。每个对照减少的 ledger 字节与五块独立存储总和减去最大块的公式完全相等；host pool 和 pinned-host 用量不变，decode live USM 计数不增长。

另外打开 arena 跑中文 MTP 小样本的 1/2/3/4 行 verifier：target/draft token 与原独立分配轨迹一致，各行宽的 top-1 一致，内部 device checkpoint 恢复重放位级一致。此补充探针没有跨模式保存完整 verifier logits 哈希，因此不将它描述为全部小行宽的跨模式 logits 位级验证。此前 MTP CPU reference 的数值失败仍然存在。

以上 slots 差异来自各案例的 max-seq/预算配置，对照对内完全相同。chunk 增大不能只看 Prefill 速度，必须同时计入 runtime scratch、persistent state、实际 cache slots 和后续 decode。没有把回收的 arena 字节直接加成新专家槽，没有自动扩大 ring。新分配仍需留出 WDDM/驱动和其它 GPU 使用的余量，不能将 16 GiB 减去 USM ledger 当成可全部使用的真实显存预算。

## 时间结果及其限制

[MEASURED] 每遍 Prefill 包括提交、同步和最终 logits 校验；表内取去首遍的三个总时间中位数。2052 的第二遍仍有较长过渡区，JSON 保留全部值，不能称三遍都已稳定。八步尾段为固定 teacher forcing 工作，不是 generated TPS。

| 输入 | 原分配 Prefill tok/s | arena Prefill tok/s | 原分配尾段 ms/步 | arena 尾段 ms/步 |
|---|---:|---:|---:|---:|
| 512 | 115.46 | 115.36 | 96.87 | 97.23 |
| 1024，大块 | 154.57 | 154.95 | 106.42 | 107.34 |
| 2052，首对照 | 118.39 | 173.15 | 184.31 | 110.26 |
| 2052，独立分配反向复测 | 154.43 | — | 185.84 | — |

512/1024 的差异接近噪声，可靠收益是内存回收。2052 的尾段差异在两次基线中保留；但 Prefill 基线反向复测明显变快，否定了把首对照直接宣传成稳定 46% 提速的做法。未清空 OS 文件缓存，未直接测量 WDDM paging，不能将全部差额归因为已证实的显存换页。该形状下显存占用更低与执行改善相伴，仍需更多交替控制再做性能默认化判断。

1024/2052 的首次 Prefill 约 24 / 43–46 秒，暖态 155/173 tok/s 不是冷启动或默认配置保证。跨行数/不同 prompt 的时序不能当作同一工作的 chunk 对照。现有默认 chunk 仍为 512，arena 默认关闭；这里给出可复现候选及其预算，不自动升级全局分发。

S4 七个主对照进程峰值 WS 21.72 GiB，最低可用 RAM 25.49 GiB。所有这些 GPU 进程 exit=0，没有 watchdog/TDR；S1/S2 中的一次长文总时限中断另有保留。构建通过；补充 Scheduler、SessionMemory、RuntimeFault 三项 CTest 全通过。未重复全套 CTest 或服务并发验收，因此没有宣布生产共享会话/cancel 路径完整验收。

## S5：先量化 stream-all 的代价

在 arena 候选上以 1024/2048 chunk、两遍 Prefill、相同 16 GiB host pool 采集 target-only cache trace。它们只用于路由/传输诊断，不作为无 trace 的性能比较。分析逐个 JSONL 记录，按每个 forward 的层与 expert ID 去重；不把微批次数当成独立专家数。

| 真实 chunk | 唯一 layer-expert | 覆盖率 | 第二遍实际需求 H2D GB | stream-all H2D 乐观下界 GB | 下界/当前需求 |
|---|---:|---:|---:|---:|---:|
| 256，既有 trace | 10341 | 42.1% | 25.86 | 52.87 | 2.04× |
| 1024，新 trace | 13826 | 56.3% | 30.86 | 52.97 | 1.72× |
| 2048，新 trace | 15717 | 64.0% | 34.99 | 53.16 | 1.52× |

[DERIVED] 下界 = (24576 − GPU slots) × 2534400 bytes，已乐观地假定所有初始驻留槽都能免掉一次传输，未计额外驱逐或借槽。不是一次实际 stream-all benchmark。第一遍对应比值分别约 2.11 / 1.57 / 1.39，所有数据保留。384 槽 NF4 ring 本身还需约 928 MiB，不能将 Strata 的槽数和 2048 阈值直接照搬。

当前数据不支持直接启用全量扫描，尚未证明额外传输能被提前供给的重叠收益抵消。既有 B6 已做有界 demand 微批和异步 reader；复制其机制不构成新增优化。有限跨层 issuer 仍是后续候选，但应先有可定位的未隐藏 host-submit/readiness 停顿及不额外放大传输的具体方案。

其余 S5 候选也没有盲目叠加：subset head 并非当前 target bottleneck，另存 BF16 子词表会新增约 519 MiB；即使整个 draft 免费，现有接受率/verify 成本仍不支持 20 tok/s。NF4 的非线性 LUT 不能当线性 INT4 直接送 DPAS，旧 grouped XMX 失败版本没有重跑。KV 压缩留待更长上下文的容量/质量独立问题；在当前小窗口和两千上下文，scratch 回收更直接。

因此 S5 本轮完成的是**测量门槛与方案筛选**，没有新增 stream-all、subset head、grouped XMX 或 KV 压缩内核。下一项计算性能工作应针对 S2 已定位的 target expert 执行与供给，继续以真实字节、接受率和完整周期预算决策。

复现：

~~~powershell
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload memory -MemoryTokens 1024 -MemoryRepeats 4 -MemoryDecodeTail 8 -PrefillChunk 1024 -DemandMicrobatch 32 -Tag s4-base
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload memory -MemoryTokens 1024 -MemoryRepeats 4 -MemoryDecodeTail 8 -PrefillChunk 1024 -DemandMicrobatch 32 -PhaseScratch -Tag s4-arena
python perf/qwen4exp/q4e_scratch_summary.py tmp/s4-base tmp/s4-arena --output tmp/s4-summary.json
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload memory -MemoryTokens 2052 -MemoryRepeats 2 -MemoryDecodeTail 8 -PrefillChunk 2048 -DemandMicrobatch 32 -PhaseScratch -Trace -Tag s5-coverage
python perf/qwen4exp/q4e_stream_coverage.py tmp/s5-coverage.cache.jsonl --output tmp/s5-coverage.json
~~~
