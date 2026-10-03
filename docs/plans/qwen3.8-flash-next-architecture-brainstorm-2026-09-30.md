# Qwen3.8-Flash-Next：执行架构、量化计算与数据供给的下一轮设计

2026-10-03 最新进展：[H8原始NF4异构供给与缓存预算](qwen3.8-flash-next-heterogeneous-h8-results-2026-10-03.md)。Q8 CPU、mmap冷来源、短forward分片及HostRam0落地，三轮同EXE完整序列 117.88→51.49秒（减少56.3%），仅pool/short1控制为84.44秒。长文有效输入/TTFT 145.6/150.2、decode 11.1/13.5，20 decode未完成。质量/状态、所有权、回归和launcher通过；保留mmap32内存保护停止、touch/预取回退及伪零预算排除。默认preset与H7 launcher保持，H8用`-Heterogeneous`开启；后续优先CPU来源/内存服务成本与cold admission，再看router同步，R2b仍待独立验收。以下保留历史结果。

2026-10-03 最新进展：[H7供给竞争治理与长文窗口](qwen3.8-flash-next-supply-h7-results-2026-10-03.md)。三轮独立进程确认关闭推测预取/在线swap、chunk2048在同H6配置下减少首次/返回长文请求16.2%/34.1%，完整五请求序列减少18.3%；全部12进程60请求保留。GPU逐位质量、CPU容差、状态与回归通过，专用launcher已实跑。仍存在CPU/需求Acquire和付费GPU换入成本，窗口对decode有取舍；下一步实施需求优先供给及成本感知admission。默认preset与32 GiB host pool不变，R2b仍暂停，100+/20未完成。以下保留历史记录。

2026-10-03 最新进展：[H6 Ready物理槽接管与供给复核](qwen3.8-flash-next-hotset-h6-ready-results-2026-10-03.md)。接管已付费取得的prefill冷槽、保持reader/generation和同层容量，避免接管H2D；补descriptor快照与独立cold LRU回归。三组30请求、长短质量和真实trace确认实现成立，但尚无稳定请求净收益。接管成本低不等于下游复用无等待，下一步结合consumer fence、host/CPU替代成本和prefetch竞争做admission。默认preset、32 GiB host pool不变，R2b仍暂停。以下保留历史记录。

2026-10-02 最新进展：[H4/C2策略迁移与验收完成](qwen3.8-flash-next-hotset-h4-c2-results-2026-10-02.md)。已实现profile预装、衰减热集自适应、异步换入、prefill借槽和独立冷区LRU，实际测试12线程CPU冷计算与Strata PCIe比例分流。长文暖decode 8.8→11.3 tok/s、整请求29.69→24.08秒；短后续请求达到15.6，但保留适应期和异常样本。质量/状态/回归通过，候选仍显式开启、32 GiB预算不变，R2b尚未恢复。以下为此前阶段记录。

2026-10-02 最新进展：[C1/H3 首轮完成](qwen3.8-flash-next-hotset-c1-h3-results-2026-10-02.md)。同热集 CPU cold 有小幅 decode 收益，固定集合却明显损害 decode 命中；H3 的 acquire 减少尚无稳定净收益。短/长质量、状态和回归通过，保留额外续写的NLL预警、吞吐异常及profiling内存失败。候选默认关闭，下一步 H4/C2 的 profile/epoch/admission，R2b继续暂停。

2026-10-02 按 [hotset新优先级规划](qwen3.8-flash-next-hotset-priority-plan-2026-10-02.md) 暂停R2b后续实施，先完成 [H1/H2动态驻留直接执行首轮](qwen3.8-flash-next-hotset-h1-h2-results-2026-10-02.md)。正确性与状态检查通过；当前GPU-only候选短decode由14.0降至12.6/13.1 tok/s，长decode由8.95降至8.4/8.75 tok/s，未默认开启。下一步是新架构下C1冷专家CPU/GPU分工的同热集对照；不再沿用本文件早期对该方向的低优先级安排。

**2026-10-02 优先级修订：先完成 [驻留专家直接执行、hot expert set 与 CPU 冷专家新分工](qwen3.8-flash-next-hotset-priority-plan-2026-10-02.md)。** Strata 的关键设计包含 profile 预装、设备 residency table 和绕过主机供给的命中计算；旧 G5 pinning/SLRU 负例不足以否定它。先拆分 resident binding 与 cold acquire，再做 GPU hit selection，并重启同一热集下的 CPU/GPU 冷计算对照。保留 R2b 已有成果，其进一步生产集成与数值细化后移。附带 Strata 热表在既有 Aila 路由上覆盖不足，不能直接照搬名单。本文件下面保留各阶段历史结果，后续实施顺序以新计划为准；第 11 节关于 CPU 路线的旧排序也由新计划替代。

2026-10-01 补齐 [R2b 真实隐藏态、单输入模型质量与状态检查](qwen3.8-flash-next-r2b-real-quality-results-2026-10-01.md)：六组真实专家通过整数 CPU/RTE 参考及连续 graph 参数表/scatter 检查；1833-token 输入的 48 步 teacher forcing 平均 KL=0.00701、NLL 差=−0.00115，42/48 首选 token 一致，通过预设模型 canary 门槛。257-token 前缀因果、65-token 快照/恢复回放及 3 项 CTest 通过。但原 NF4 逐点门槛在真实样本中有 18.9%–46.9% 超限，不能称全面质量认证。所有 count>1 prefill 专家统一整数格式，decode 仍 NF4，禁止热度决定数值格式；诊断重复计算 NF4，未产生生产 TPS 收益。32 GiB host pool 下最终最低可用 RAM 10.99 GiB。下一步先用真实语料验证 down per32 激活尺度及小桶一致算术，再去除重复计算接入；R2b 生产集成、decode 20 tok/s 尚未完成。

2026-10-01 继续完成 [R2b INT8 DPAS 热桶与静态 graph 实验](qwen3.8-flash-next-r2b-dpas-results-2026-10-01.md)：原生 dpas.8x8 已由实际 ISA 确认；两次独立运行中，含转码、两次 Q8、pack/merge 的完整链对同 graph NF4 基线，M32 延迟下降约 26%，M48 约 24%–26%，M64 约 47%。M8/M24 落后、M16 余量不足，WG32/stride padding 没有一致收益，不写泛化阈值。CPU 算子参考、修改输入/路由/专家描述符后的 graph 逐位检查、3 项 CTest 通过。新算术仍有少量逐点超限，未做模型 NLL/生成验收，保持测试入口；下一步真实隐藏态质量与可复用热桶 graph 集成，未宣称整机 TPS 提升。

2026-10-01 完成 [R2b NF4 integer-LUT × Q8 的有界 DP4A 实验](qwen3.8-flash-next-r2b-integer-results-2026-10-01.md)：真实专家整数参考、Q8 RTE/非有限保护、SwiGLU 和 merge 检查通过，实际 GPU ISA 确认原生 dp4a；四行权重复用及精确修正倒数量化把 M32 整数完整路径从约 953 降至 555 µs，但原 NF4 为 522 µs（仍慢 6.2%），M1 仍慢 20.8%。近似 NRMSE 1.65%–1.80%，少量元素超出原逐点门槛，未做模型质量认证，保持测试入口。R2b 的 INT8 DPAS/真实隐藏态验收尚未实现，未宣称整阶段完成；可继续 DPAS 热桶或原计划 R3-head，9033/32 GiB/GPU-only 不变。

实施更新（2026-10-01）：完成 [R1b 有界 MoE down 与顺序无关 merge](qwen3.8-flash-next-r1b-compact-moe-results-2026-10-01.md)。为避免缓存/分组顺序影响前缀舍入，采用 int64 Q36 窗口累加而非 FP32；真实专家地址检查、整数参考、倒序提交、整窗口未来无关性、checkpoint 和 48 步质量检查通过。attention tile256 上再回收 94.73 MiB GPU USM，但同轮暖 TTFT 17.734→17.8335 s、decode 5.15→5.10 tok/s，无端到端收益，默认关闭。预算自适应/层间让出未做，R1b 不是完整交付；下一项 R2b / R3-head。

实施更新（2026-10-01）：完成 [R1b attention/GDN tile 第一部分及 9033 驱动复测](qwen3.8-flash-next-r1b-attention-tile-results-2026-10-01.md)。整窗口路由与 attention tile 已解耦；自然输入 tile256 回收 258.27 MiB GPU USM，状态/因果及 48 步质量检查通过，但暖 TTFT 增加 1.7%、完整请求基本持平，保持默认关闭。MoE rank 临时区、预算窗口与层间让出尚未改造，R1b 未全部完成。9033 同旧二进制的首次暖 TTFT 比 8860 历史值慢约 2.6%–2.8%，decode 基本持平；尚不能确认驱动回退，原始数据保留。

实施更新（2026-09-30）：已完成 [R0 / R1a / R2a 首轮实施与实测](qwen3.8-flash-next-architecture-r0-r2a-results-2026-09-30.md)。显式 A770 CLI preset 可用；1833-token 自然输入的暖态 TTFT 从 22.607 降至 16.412 s（约 111.7 input tok/s），prefill demand 字节减少 68.5%，但 decode 从 7.35 回退至 5.25 tok/s，因此保留 chunk=512 默认并进入 R1b。M32 有限 BF16 staging + oneDNN 完整路径比旧 M8 快、仍比 ALU 慢约 19%，保留实验。R1b/R2b–R6 尚未实施，decode 20 tok/s 未达到。以下保留原始 brainstorm，原文的“本轮未运行”指规划勘察阶段。

日期：2026-09-30。代码基线 `66e21e9` **加本轮开始时已有的未提交改动**，包含 S1–S6、RAM 预算和 QSA selector 工作。本轮只勘察代码、阅读既有报告，并对已有 trace 和容量公式做 CPU 侧计算；没有构建、GPU 推理或重新跑验证。未改动现有 worker 的代码与报告。

源码哈希、参考仓库版本、计算输入和已有 trace 的重新汇总见 [evidence JSON](/E:/RiderProjects/Aila/docs/plans/qwen3.8-flash-next-architecture-brainstorm-evidence-2026-09-30.json)。本文件中的 `[CODE]` 是源码事实，`[REPORTED]` 是已有实验或外部仓库报告，`[DERIVED]` 是条件计算，`[HYPOTHESIS]` 是尚未在本机验证的方案。

## 1. 建议改变下一阶段的投入方式

**性能差距不能只归因于 PCIe，也不能仅靠继续微调现有 kernel 消除。最值得投入的是：减少一段 prompt 重复搬多少遍专家；让量化格式与计算指令相匹配；把现有 BF16 常驻权重占用换成更低的读取成本和更多专家容量。**

当前已经积累了足够多的负结果。下一轮应停止重复“更多 overlap、再换一个 graph、再扫一组 cache 比例”的循环，改成几项可以独立交付、作用机制明确的工作：

| 优先级 | 工作包 | 主要目标 | 获益依据与把握 |
|---|---|---|---|
| R0 | 把已实现能力组成一个显式 A770 性能配置，覆盖真正使用的 CLI 路径 | 先兑现已有成果 | 高把握消除配置落差；不是新算力收益 |
| R1 | 大路由窗口、分阶段小 tile 的 PrefillExecutor | 减少长 prefill 的重复专家传输 | 较高：先改变调用粒度，再逐步解耦内存规模；字节收益可提前估值 |
| R2 | 新的 MoE 计算基线：有限 BF16 staging + 成熟 GEMM；继而 NF4 integer-LUT × Q8 activation | 降低真实专家计算成本 | 前者实验成本低、可判定；后者中等把握、高上限，属于新算术候选 |
| R3 | HC 主投影、完整 LM head 的独立 Q8 路径 | 同时降低 decode 工作与显存压力 | 较高容量收益，速度和质量需小原型确认；先分开做 head 与 HC |
| R4 | 按 GQA 的 12 个 query heads 组织 XMX attention | 加速实际 QSA attention，而非再优化 selector | 具体参考实现已存在；A770 映射和短 decode 并行度待确认 |
| R5 | 非阻塞专家 ticket、按 readiness 提交、按 key 提升 I/O 优先级 | 削减真实暴露的供给停顿 | 条件性；有代码机会，但暖短 decode 已无 file fill，不能通吃所有场景 |
| R6 | 更低比特专家和分层表示 | 跨越 RAM/VRAM 容量边界 | 高潜力、较高质量风险，作为独立格式实验，不默认开启 |

第一轮不同时实现六项。先交付 R0/R1 的可执行边界，以及 R2 或 R3 中一个小原型；R4 可以作为独立算子工作包。长期目标是专用 `PrefillExecutor / DecodeExecutor / VerifyExecutor`，共享权重和状态格式，但不再强迫三种工作复用同一套执行顺序和 scratch 形状。

继续遵守当前 GPU 专家计算主线和 32 GiB 研究用 host pool；容差允许改变舍入顺序。CPU 用于 I/O、调度、离线转换和参考计算，不把此前失败的 CPU expert 路线换个名字重新开启。

## 2. 当前状态：哪些已经完成，哪些不能再当作空白

| 已有工作 | 证据与边界 | 下一轮处理 |
|---|---|---|
| dense XMX、direct I/O、shape dispatch | X0/X1 已实现；direct I/O 当前默认 true，dense XMX 仍显式 opt-in | 做运行配置整合，不重写一遍 |
| grouped NF4→BF16 XMX | X3 在 256-token prompt 有约 85.9% activation 进入候选，但 prefill 3633.54→4165.57 ms | 否定的是当前 M8 实现，不是所有矩阵化 MoE |
| chunked GDN | X4 FP32 chunk 算子约快 28%–31%，模型 prefill 只快约 1.2%；XMX product 更慢 | 保留，不作为下一轮主攻；不重复原版 tile sweep |
| MTP / 小窗口 verifier | KV-only catch-up、2–4-row 投影/HC、device checkpoint 等已做；完整生产 MTP 未完成 | 先降 target 周期成本，不投入大规模接受/拒绝状态机 |
| arena | S4 回收约 77–700 MiB；没有自动把空间还给专家池 | 作为新执行器的基础；“复用 scratch”本身不再作为新方案 |
| 驻留/缺失拆批 | S6 四行验证约改善 2.7%–3.2%，更大 overlap 同时伴随 kernel 区间增长 | 证明重叠不是免费收益；不盲目增加队列 |
| QSA selector | 32K Top-K 1.956→0.118 ms；2146-token CLI 暖 decode 仍约 6.15 tok/s | selector 已有成果，下一项是实际 attention 与供给 |
| RAM | 16→24 GiB 帮助短 decode；32 GiB 短输入无额外收益，长输入有收益也有回落 | 32 GiB 作为研究基线，不再将扩 RAM 当通用优化 |

这些结果分别来自 [X3](/E:/RiderProjects/Aila/docs/plans/qwen3.8-flash-next-x3-results-2026-09-29.md)、[X4](/E:/RiderProjects/Aila/docs/plans/qwen3.8-flash-next-x4-results-2026-09-29.md)、[S4/S5](/E:/RiderProjects/Aila/docs/plans/qwen3.8-flash-next-strata-s4-s5-results-2026-09-30.md)、[S6](/E:/RiderProjects/Aila/docs/plans/qwen3.8-flash-next-strata-s6-resident-results-2026-09-30.md)、[QSA selector](/E:/RiderProjects/Aila/docs/plans/qwen3.8-flash-next-qsa-selector-results-2026-09-30.md) 和 [RAM32](/E:/RiderProjects/Aila/docs/plans/qwen3.8-flash-next-ram32-decode-results-2026-09-30.md)。本轮没有重新认证其中的测速。

## 3. “离理论值很远”应如何拆开判断

### 3.1 有用的下界是数据流下界，而不是峰值 TFLOPS

[DERIVED] H=2560、I=640，每专家 `3HI=4,915,200` 权重；当前 NF4+nested code slot 为 2,534,400 B。48 层、每层 top-10 的单 token 涉及约 **1.2165 GB 专家 payload 的逻辑读取**，这不是每 token 都必须跨 PCIe 的字节数。全 24576 个专家约 62.285 GB，即 58.01 GiB。

在同一份 [24 GiB 暖 CLI trace](/E:/RiderProjects/Aila/tmp/ram24-generation-24-trace-20260930.cache.jsonl) 的第四请求，本轮重新汇总得：

| 区间 | token 行数 | demand H2D | host file fills | acquire 主机区间 | router wait 主机区间 |
|---|---:|---:|---:|---:|---:|
| Prefill，37+4 | 41 | 11.904 GB | 0 | 134.97 ms | 1329.68 ms |
| decode 含末尾状态对齐 | 48 | 8.812 GB | 0 | 230.95 ms | 1702.43 ms |

decode 平均约 183.6 MB demand H2D/行、4.81 ms acquire/行，trace 内 forward span 平均约 63.76 ms/行。它是旧的 24 GiB 带 trace 请求，**不是本轮 32 GiB 的新 benchmark，也不是 generated TPS**。但它足以排除“该暖短请求主要还在等磁盘”这一解释。

router wait 平均约 35.47 ms/行包括此前排队的 GPU 工作，不是可通过移除一次 D2H 立刻收回的 CPU 开销。时间线里的 copy/kernel union 也不宜与这些主机区间相加。

[REPORTED] 已有供给隔离实验搬 648.8 MB 加计算约需 69 ms，得到约 9.4 GB/s 的**完整路径有效速率**，不是裸 PCIe 带宽。若暂以此速率估算上述 183.6 MB，则传输服务量约 19.6 ms/行；假设能到 12 GB/s 则约 15.3 ms/行。这说明减少 PCIe 字节有价值，也说明“把链路填满”本身不能解释所有 60–70 ms 延迟。12 GB/s 是例算参数，并未测得；实际协商的 PCIe 代际/宽度也未在本轮确认。

真正的预算应分为 `文件→RAM 就绪 / H2D / GPU kernel / 提交与依赖间隙`，再按关键路径组合。纯硬件乐观下界可用 `max(bytes/BW, FLOPs/算力)` 帮助筛选；它不是有 48 层依赖、量化解码、低占用小 kernel 和 WDDM 开销的完整模型时延。

### 3.2 Strata 的速度并非同条件，但存在可迁移的执行设计

[REPORTED] Strata 本地版本 `a790805`（0.1.27）；[0.1.26 表格](/E:/RiderProjects/Aila/tmp/Strata/bench/results/2026-09-29-speed-0126/README.md) 使用 RTX 5070、Ryzen 7600/DDR5、文档所记 PCIe 5.0 x16、MTP、256-token 输出，且每格单次请求。Q2_0 的 4K decode 93 tok/s，IQ3_S 为 53.3 tok/s。其 decode 有 CPU experts，Aila 当前没有。

Q2_0 的专家编码按其源代码约为 18 B/64 权重，Aila 为 33 B/64，payload 比约 **1.83 倍**。此外更小权重能提高 GPU 驻留率，减少主机缺失，收益可能超过字节比例。不能把 93/14.6 直接解释为 GPU 算法差距，也不能以“不公平比较”为由停止寻找执行缺口。

最值得迁移的是：大窗口内跨 token 复用专家、成熟量化 GEMM、针对 GQA 的 attention 矩阵映射、小窗口 verifier 的专用组织。它们有具体代码依据，不依赖复制 CUDA graph 或 GPU 自旋等待。

## 4. R0：先解决实验配置与日常路径的落差

[CODE] 当前 [Q4E 初始化](/E:/RiderProjects/Aila/src/models/Qwen4ExpBnb4Backend.cpp:372) 中 `AILA_Q4E_PREFILL_XMX` 默认 false，`AILA_Q4E_DIRECT_PREFILL_IO` 默认 true；B6 默认 0，phase scratch 默认 false，small-row NF4/HC 默认 false。[研究 probe](/E:/RiderProjects/Aila/perf/qwen4exp/q4e_execution_probe.ps1:49) 则显式启用 dense XMX，并可加入 B6、arena 等。

[CODE] interior checkpoint 的自动条件仍只覆盖初始 `count=30, anchor=26`。结构检查本身已支持更宽范围，显式 opt-in 可进入其他合法短段；真实 CLI 的 41-token story 在旧 trace 仍为 `37+4`。因此 messages-json 的改进不能自动等同于用户日常 CLI 的改进。

[HYPOTHESIS] 增加一个可审计的 A770 性能 preset，打印最终生效的分发、chunk、host budget、cache slots 和 checkpoint 模式。将已测候选按形状组合，不一次性打开 grouped XMX/GDN/MTP 等负结果。对 37+4→41 做一次有针对性的 checkpoint continuation 检查后，推广按状态语义准入，减少对某个模板长度的特判。

这项工程便宜，优先做，但只改善“已实现却未使用”的部分；它不会修复长 prefill 的根本数据流问题。也不在本轮直接修改默认值。

## 5. R1：大路由窗口、小算子 tile 的 PrefillExecutor

### 5.1 当前瓶颈不只是 chunk 太小，而是几种粒度绑在一起

[CODE] [RequestState](/E:/RiderProjects/Aila/src/inference/Request.cpp:260) 按 `prefillChunkSize` 截取输入，每个 chunk 调用一次完整 session 执行；[Scheduler](/E:/RiderProjects/Aila/src/inference/Scheduler.hpp) 默认上限 512。实际 CLI/旧 engine 还有各自 chunk 入口，改造时必须核对用户使用的入口，不能仅修改 Request 的一个默认值。

当前模式近似：

```text
tokens 0..511:    layer 0 -> ... -> layer 47
tokens 512..1023: layer 0 -> ... -> layer 47
...
```

一层专家等到下一个 token chunk 重用时，已经穿过其余 47 层的工作集。GPU 只有约 3.6–3.7K 个槽，host 也放不下全部 NF4，重复搬运和缓存污染是自然结果。仅调 LRU 很难弥补执行顺序制造的重用距离。

[REPORTED] 已有不同窗口的唯一 layer-expert 数为 256→10341、1024→13826、2048→15717。增长显著慢于 token 数，说明较大窗口有合并需求的空间；这些来自特定重复文本，不能据此宣称自然文本一定获得某个加速倍数。

### 5.2 新执行方式

[HYPOTHESIS] 把“路由/权重复用窗口”与“attention、GDN、MoE 的计算 tile”分开：

```text
window 0..4095 的 R / hidden bank
  layer 0:
    attention/GDN 按 128/256 行小 tile 顺序处理整个窗口
    汇总这一层窗口内的精确路由
    按 expert 分组，流式执行所有属于该 expert 的 token
  layer 1 ... layer 47
提交窗口末态与最后一行 logits
```

相同专家在这个窗口内原则上只需供给一次；它的 m_e 也更大，更适合 GEMM。仍然只传实际选中的专家，不变成 Strata 的 stream-all。

因果语义是可实现的：某层所需的上一层整段输出先准备好；本层 GDN/conv 沿 token 时间推进，QSA 每行使用自己的可见前缀与选块。MoE 在获得本层输入和路由后可按专家重排 token，不改变数学依赖。不同 query 的 QSA IDs 仍独立，不能为了合批直接取同一组块。

### 5.3 用显存保存激活，交换更少的权重重传

[DERIVED] 4096 行、4×2560 的 BF16 hidden4，双缓冲仅 **160 MiB**。再加一个 FP32 `[T,2560]` 输出累加区约 40 MiB，top-10 路由 ID+FP32 权重约 0.31 MiB。它们只是工作集组成，不是完整显存预算。

现有 `[T,top_k,H]` FP32 `moe_down_acc` 约 100 KiB/输入行，4096 行约 400 MiB。新执行器可以让每个专家/微批输出先写小 scratch，再按明确的顺序累加到 `[T,H]`，将十份整窗口 rank buffer 改为一份输出加有界临时区。归约顺序可以改变，按用户允许的容差验收；初版用有序 merge，不引入浮点 atomic 争用。

GDN/attention scratch 按内层 tile 分配，复用 S4 arena；不因为路由窗口变为 4096，就把所有临时张量扩到 4096。所需 KV、PLE history 和 recurrent state 仍持久化。它比继续为当前单体 `ForwardImpl` 增加几个环境变量更值得投入。

### 5.4 分两步落地

1. **R1a，较小改动**：同一个自然文本、同一个 max-seq，固定预算比较现有 chunk=512 与 1024/2048+arena。允许缓存槽因预算发生变化，但必须记录，不能拿不同 prompt 或不同 warm 状态对比。已有 S4 数据不是这个同输入交错实验，仍不能作为确定速度结论。
2. **R1b，新执行器**：抽出 `RunAttentionLayerTile / RouteWindow / RunExpertGroups / MergeWindow`，先实现单 session、1024/2048 的受预算窗口。之后再扩大到 4096，并接入层边界让出执行权、cancel drain 和 session 隔离。

短请求直接原路径；多人服务的公平性不应被一个不可中断大窗口破坏。不要为首次原型先重写整个 scheduler，也不要将专用路径绑死在 defaultSession。

**停止/继续依据**：在同一输入上，需求字节和总 prefill 明显下降就继续；若字节已大降但时间没降，转向 R2/R4，而非继续扩大 chunk。预算不足时缩小窗口，不靠 WDDM 超额驻留硬撑。

## 6. R2：MoE 换计算方法，而非继续微调当前 M8 XMX

### 6.1 已失败原型的具体成本

[CODE] [MoeGroupedXmx.cpp](/E:/RiderProjects/Aila/src/ops/MoeGroupedXmx.cpp:28) 固定 BM8/BN32/BK32、WG32/SG8。gate/up 的 K=2560，需 **80 轮 K 循环、每轮两次 WG barrier**；down 的 K=640 也需 20 轮。每个 N tile 重做 activation gather，每个 M8 tile 重做对应权重解量化。scale lookup 在连续八权重粒度调用，同一 64-weight block 存在重复标量处理；实际编译器是否消去需查生成代码，不能凭源码断言每次都访问 DRAM。

[REPORTED] 真专家完整 gate/up→down 原型约 1.4–1.7 ms，明显慢于旧 ALU；这不是低位宽矩阵单元的物理极限。它说明“把 joint_matrix 放进去”不足以产生高效 GEMM。

### 6.2 R2a：先取得一个可用的矩阵乘基线

[HYPOTHESIS] 使用 2–4 个专家的 GPU BF16 staging（约 18.75–37.5 MiB，另计激活），每专家权重在本窗口展开一次；按 m_e bucket 预建 oneDNN batched GEMM，或采用已有成熟 DG2 GEMM 核心。将同专家 token gather 成连续矩阵，逐阶段完成 gate/up、SwiGLU、down、scatter。

专家尺寸一致，按 bucket padding 可以形成批量 primitive，避免每专家临时建 descriptor。descriptor/arena 在执行槽初始化时准备；先只做一个热桶，不需要一次覆盖所有 M。完整计时包含解量化、gather/scatter 和 merge，权重依然通过 PCIe 传 NF4。

参考 [Strata Gemm::quantized 的有界展开](/E:/RiderProjects/Aila/tmp/Strata/src/prefill/gemm.cu) 与 [vLLM Marlin 的分桶/排序](/E:/RiderProjects/Aila/tmp/vllm/vllm/model_executor/layers/fused_moe/experts/marlin_moe.py:342)。借用的是组织方法，不能把 CUDA/Triton 分发阈值移植到 DG2。

这条基线的价值是快速回答：现有 X3 究竟输在自写 kernel，还是目标 m_e 下连成熟矩阵乘加准备也不划算。若仍输，保留 ALU；不用先投入新一轮大规模 fused kernel 调参。

### 6.3 R2b：保留 NF4 payload 的 integer-LUT × Q8 activation

[CODE] [llama.cpp SYCL `vec_dot_iq4_nl_q8_1`](/E:/RiderProjects/Aila/tmp/llama.cpp/ggml/src/ggml-sycl/vecdotq.hpp:1694) 将非线性 4-bit code 查表成 int8，使用 `dpct::dp4a`，最后乘 block scale。[Strata MMQ](/E:/RiderProjects/Aila/tmp/Strata/src/prefill/moe_mmq.cu:128) 先量化 activation，再调用 llama.cpp 的 grouped quantized matmul。**非线性 codebook 不等于不能做整数点积；只是不能把 nibble 本身当作线性 INT4。**

[HYPOTHESIS] 对当前 NF4 LUT 定义 `q8[c]=round(127*NF4[c])`；GPU 仍加载原有 4-bit code 与 nested scale，tile 内查成有符号 int8。对 activation 做每 64/128 维的对称 Q8，整数点积后按 activation scale 与原权重 scale 转回 FP32，再完成 SwiGLU/merge。

[DERIVED] 本轮仅计算 16 个 LUT 值，得到：

```text
[-127,-88,-67,-50,-36,-23,-12,0,10,20,31,43,56,71,92,127]
max |NF4[c] - q8[c]/127| = 0.00367107
16-level unweighted RMSE = 0.00218868
```

这是归一化 LUT 的误差，未计 activation quantization、scale、异常值和层间放大，**不是模型误差保证**。不修改原权重文件，也不新增整份 INT8 专家缓存。PCIe 字节和 GPU NF4 cache 容量保持原样，先把计算链从大量浮点解码/FMA转为 packed lookup/dot。

decode 首版用一输出行/SG 的整数 dot，保持已有 GEMV 的良好访存方式；prefill 再用 INT8 DPAS。每 64 个 K 位置的权重 scale 不同，必须分 block 完成 INT32 dot 并做 FP32 scaled accumulation，不能把整个 K 的和统一乘一个 scale。activation scale 也必须匹配 block 边界。

DG2 的 INT8/INT4 DPAS 能力不等于该实现必然快；确认目标 `dp4a` 没有退成大量标量指令，再做一个真实专家原型。不要复用 IQ4_NL 的数值表代替 NF4 表，也不要误认为它们是相同量化格式。

与 R2a 比较完整成本后再决定主线。质量敏感时可对 outlier block 保留 BF16 或原 NF4 算术，但先不要引入复杂的动态混合逻辑；普通 Q8 canary 是否可接受应先明确。

## 7. R3：压缩两块被忽略的常驻 BF16 大权重

[CODE] [LM head 初始化](/E:/RiderProjects/Aila/src/models/Qwen4ExpBnb4Backend.cpp:433) 将完整 `248320×2560` 权重转置后以 BF16 常驻，用 oneDNN forward；它并不是 NF4。HC 也保留 BF16 down/up，不能把“模型是 NF4”理解为全部权重都已四位。

[DERIVED] 完整 head 为 **1212.5 MiB**。按已有 97 个 HC、每个 `10240×320` down 和 `320×10240` up 计算，两张主矩阵合计也为 **1212.5 MiB**，这里不计 injection 和 norm。

将这两部分分别变为 Q8，原始码字部分合计可少约 **1212.5 MiB**；扣除 scales、activation scratch 和其他预留后，净空间略小。未扣 metadata 的容量上限相当于 **501 个 NF4 专家槽**，约为当前 3714 槽的 13.5%。这不是命中率提升 13.5%，更不是速度提升 13.5%。

[HYPOTHESIS] 优先顺序：

1. **完整 head Q8**：保留完整词表，不做可能影响覆盖的 subset head；预量化只读权重，使用 Q8×Q8 dot 或 W8A16 解码计算候选。LM head 对排名敏感，用 teacher-forced logits/NLL 和 top-margin 观察，避免仅比较平均输出误差。
2. **HC down/up Q8**：先一组真实 HC，norm、sigmoid、SiLU、FP32 reduction 与 residual 状态仍保持现有精度。HC 重复 97 次，误差可能扩散到路由；不要 head/HC 同时改后再猜来源。
3. 通过局部性能/质量后，重新做静态显存预算，将部分净回收容量给专家池，另一部分保留为窗口执行器余量。先分别测纯 kernel 收益与扩大 cache 的收益，再看联合请求。

这是“更少每 token 必读数据 + 更大专家 cache”的双重作用。它不会像 static pinning 那样先扣掉 864 个动态槽，也不依赖猜对热专家。若新 INT8 kernel 算力不佳，可先让 Q8 只服务一部分形状；不因为保存的字节漂亮就全局切换。

技术报告中的 FP8 residual 是另一件事：它压的是四路残差激活，不是 HC 权重。报告还记录只读两条 residual branch 在后训练后退化，不能把删 branch 当作无损优化。见 [技术报告抽取文本 §2.2](/E:/RiderProjects/Aila/tmp/brainstorm-qwen38-20260927/report.txt:977)。FP8 residual 对大窗口的带宽有潜力，但 A770 没有因此自动获得原生 FP8 运算，当前优先级低于两块大权重 Q8。

## 8. R4：QSA 的 12-head 矩阵映射，兼顾速度与数值

### 8.1 在单 token 内找到真正的矩阵行

模型 `Hq=24, Hkv=2, D=256`，同一个 query 位置的 12 个 query heads 共享一个 KV head；该位置也共享选中的 block IDs。

[HYPOTHESIS] 对一个 `(query, kv_head)` 构造：

```text
Qgroup: [12,256]，tile 可补为 [16,256]
Ktile:  [selected_cells,256]
scores = Qgroup @ Ktile^T
P = 每个 query head 各自 softmax(scores)
Ogroup = P @ Vtile
```

即使 decode 的 token 数 M=1，也有 12 个有效矩阵行；这与强迫 M=1 的权重 GEMV 填成 8 行完全不同。每个 KV tile 在这些 heads 内共享，FP32 online-softmax 避免全量 scores 写回。

[CODE] 这不是凭空设想：[Strata `qsa_prompt_attn.cu`](/E:/RiderProjects/Aila/tmp/Strata/src/kernels/cuda/qsa_prompt_attn.cu:17) 明确 `G=12`、12 heads+4 padding、MMA；[prefill 入口](/E:/RiderProjects/Aila/tmp/Strata/src/prefill/prefill.cpp:1464) 使用一个 `(query,kv_head)` block 的 tensor-core attention。Strata 的实际入口是 prompt 路径，不能据此声称其单 token decode 也已验证同样收益。

Aila 当前 [QSA batched attention](/E:/RiderProjects/Aila/src/ops/AttentionOps.cpp:4285) 是 2/4 个 heads 的向量协作，仍有更进一步的矩阵映射空间。dense D256 的长 prefill 也有逐 query/head 的 scalar tiled 路径，不能把“存在若干 joint_matrix 函数”当成这些实际分支已走 XMX。

### 8.2 不必在 FP32 与低精度之间二选一

Q/K 已经是 BF16，先用 BF16 XMX、FP32 accumulate；max/sum/在线重标定保留 FP32。P×V 的 P 来自 FP32 softmax，可对照普通 BF16 P 与双分量形式：

`P_hi=BF16(P), P_lo=BF16(P-FP32(P_hi)); PV≈P_hi V+P_lo V`。

多一次矩阵乘换更小误差，不要求 bitwise 相同。Strata 对 Q/P 使用 hi/lo FP16 与幂次缩放提供了参考；Aila 已有 BF16 Q/K，无需机械复制所有步骤。这个方向特别适合处理此前 MTP 中 attention 对小误差放大的现象。

DG2 用 SG8、FP32 accumulator 和受限 SLM；不搬 CUDA `cp.async`、MMA fragment 或 PVC `load_2d`。不能越过因果边界，0–3 个尾部 token、非连续块、gate 和 KV stride 都要保持正确。不同 query 的 IDs 不同，按各自 tile gather，不共享错误的稀疏集合。

单 token 只有两个 KV groups，若每组一个 WG 会并行度不足。decode 候选应 split selected-key tiles，产出 FP32 `(max,sum,partial O)` 再合并；prefill 的 query 维通常已提供足够 WG。先做真实 attention 算子对照，再接入整请求，不能因 32K 合成形状赢就承诺 2146-token end-to-end 一定改善。

## 9. R5：把 I/O completion 与 kernel submission 的所有权拆开

### 9.1 现有异步不等于提交线程不会停

[CODE] 当前存在以下同步位置：

- [victim reader fence](/E:/RiderProjects/Aila/src/models/Qwen4ExpExpertCache.cpp:1500) 在主线程 `wait_and_throw` 后才复用槽。
- [matched prefetch](/E:/RiderProjects/Aila/src/models/Qwen4ExpExpertCache.cpp:1123) 主线程抢未完成 part，并 spin/yield 等 ready。
- [demand fill](/E:/RiderProjects/Aila/src/models/Qwen4ExpExpertCache.cpp:1535) 通过 `fill_pool_->run` 帮忙读数据，并等待该批任务都结束后 acquire 返回。
- [ReadFile reader](/E:/RiderProjects/Aila/src/models/Qwen4ExpExpertReader.cpp:526) 使用 OVERLAPPED，但每个 segment 的 worker 仍 `GetOverlappedResult(..., TRUE)` 等完成；它是线程池并发，不是完成端口式多请求 issuer。
- acquire 的非全命中路径末尾仍调用 `cancel_prefetch()`；取消先等活动 worker 停止。因此一个 batch 的生命周期与整批预测读任务仍有耦合。

这些是代码层面的重构点，不是已测得的延迟归因。暖短 trace 没有 file fills/prefetch wait，不能拿同一解释套到它身上。

### 9.2 目标接口和首版范围

[HYPOTHESIS] 定义按 `(layer,expert,generation)` 持有的 `ExpertTicket`，分为 host filling、host ready、device copying、device ready、consumed；demand 与 prefetch 只是同一 ticket 的优先级，不再创建后相互取消的两套批任务。

`PlanExperts` 只固定专家/槽位 generation 并返回 ticket；独立有界 issuer 处理 host misses 与 H2D。提交线程先发已经 ready 的同层工作；需要等待时等待最早能推进关键路径的一组 completion，不阻塞在预排序列表的第一个慢专家。不要为了 32 个算子槽必须等齐 32 个 file read。

设备 reader fence 尽量转成特定 copy 的依赖，而不是主线程等待；但旧 generation 的 descriptor、host source lease 与新 generation 的 publication 必须分开，不能先改映射再让旧 reader 读新 metadata。每个执行槽使用预分配 ticket/descriptor 数组，取消时 drain 所有已提交 reader/DMA 后再释放。

compute queue 初版仍可 in-order，靠 host 选择 ready 工作顺序，不需要全局 out-of-order 或 GPU 常驻自旋。S6 已证明拆批可能只改善几个百分点，所以设 DMA credits/水位，允许减少并发 copy 保护计算；不要追求最大 overlap 这个指标本身。

低成本前置是把现有 read/transform、wait-prefetch、consumer-fence 与 submit-gap 区间分清，复用已有 trace。只有长自然 prompt 的未隐藏停顿足够大才实现 issuer。**不再无差别重跑 copy slicing、bulk memcpy、全局 immediate-list 或旧 graph。**

一个可选的独立布局优化是生成只读 expert-major sidecar：每专家的 packed gate/up、down、scale 连续存储，保留原 nibble/LUT/scale 数值和模型哈希，一个逻辑 slot 不再跨多个 tensor 段读取。它改善 host-miss 路径，不降低 H2D，也不帮助已全在 RAM 的短 decode；只有 segment/整理成本显著时才优先做。

## 10. R6：更激进的容量路线，收益可能来自跨过阈值

[DERIVED] 只按专家 payload 和给定 scale 开销估算：

| 格式假设 | 全部 routed experts 容量 | 含义 |
|---|---:|---|
| 当前 NF4，4 bit + 每64权重1 byte scale code | 58.01 GiB | 32 GiB RAM + 约9 GiB GPU 放不下独占全集 |
| 3 bit + 每64权重1 byte scale | 43.95 GiB | 接近但仍超出当前 RAM+GPU 合计；格式仅为预算假设 |
| 2 bit + 每64权重FP16 scale，类似 Strata Q2_0 的字节率 | 31.64 GiB | 专家 payload 本身可能接近全部留在32 GiB RAM，尚未计索引、其他模型数据和系统余量 |

小幅压缩可能将供给从“host pool miss 后读文件”变成“全部可在 RAM/VRAM 中复用”，这类收益不是线性的。但两位专家的质量风险远大于 XMX 归约顺序变化，不能混用同一个“容差允许”理由直接发布。

[HYPOTHESIS] 可行探索顺序：保持 target 路由和专家个数不变，先对专家权重做少量层的 Q3 或混合 Q3/Q4；按敏感性保留某些 down projection/层/专家较高精度；用统一的 bytes+error 预算分配，而非简单“热专家四位、冷专家两位”。冷专家也可能承载关键能力。

优先从原始高精度权重生成新格式；若只有 NF4 checkpoint，NF4→Q3 是二次有损量化，需明确比较基准与额外损失。不通过删专家、减 top-k、跳层或缩目标词表获得伪装成同模型的速度。

另一个更远的方向是 coarse+residual expert：先用小型基础表示计算，按精度策略加载稀疏残差；它改变数值与数据格式，需要重新设计质量契约。只保留为研究分支，不放进前几轮的工程承诺。

## 11. 其他方向的取舍

- **缓存策略**：已有 protected/static pinning 负例，不再优先 sweep。新信号可用 prompt 末尾窗口而非整段频率为 decode 建立缓存候选；必须先在已有 route trace 上做未来不可见的回放，保持槽数和实际传输预算。它是小型候选，不能替代 R1。
- **MTP**：若 R2/R3/R5 降低 target verify，复用已做的 KV-only catch-up、真实接受率和小缓存 draft。生产 commit 的收益门槛重新计算；不拿重复灯塔的接受率作一般预测。lookup draft 只能作为重复文本专用候选，不能先承诺 20 tok/s。
- **GDN**：先保留现有较好的 FP32 chunk。更大 token tile、state 分片驻寄存器有理论空间，但须看 spill/并行度；当前已知模型收益小，暂不压过专家和执行窗口工作。
- **全局 graph/OOO**：当前 shared arena 与隐式 in-order 依赖不允许直接去掉队列属性。局部分支明确独立后再试；旧 front graph 和 immediate-list 负结果仍有效。
- **FP8 residual / KV 压缩**：对更长窗口有容量意义，当前先做计算和执行次序。技术报告的 FP8 结果不是 A770 性能保证，也不能将压 residual 的收益套到权重上。
- **CPU expert**：Strata 能用 Ryzen/DDR5 和更小权重获益，不代表本机旧 CPU/NF4 原型值得重启。除非压缩格式或硬件条件发生实质变化，不让 CPU 算术重新进入当前主线。

## 12. 有限验证与交付顺序

本轮不再建一套庞大验证框架。复用现有工具，每个原型最多选择真实短输入、一个自然中长输入及必要的单层 fixture。只在收益接近噪声时补少量交错样本；除状态机制改动外，不重复完整 lifecycle/全量 CTest。

| 第一轮可执行工作包 | 实现边界 | 首个决策结果 |
|---|---|---|
| R0 + R1a | 显式性能 preset；41-token checkpoint；同自然输入512/2048窗口对照 | 用户入口是否兑现现有能力；更大窗口到底减少多少字节和时间 |
| R2a | 一个热专家桶，预分配 BF16 staging + batched成熟GEMM | 给出与旧 ALU、失败 M8 XMX 完整路径可比的基线 |
| R3-head | 仅完整 head Q8，不改词表、不改 HC | head 时间、数值差异与可回收净容量 |
| R4 | 一个真实 QSA fixture，12-head XMX；普通P与双分量P对照 | 是否在算子层实现有用收益，短decode是否需要split-key |

随后按结果选主线：大窗口的字节收益强则进入 R1b；成熟 GEMM 仍输则先做 R2b；暖 decode 的量化 head 获益且质量可接受则扩到 HC；长请求明确卡在未隐藏 I/O 才进入 R5。

验收继续比较同一 NF4 基准与 teacher-forced 路径的分布误差/质量，不要求自由生成 token 完全一致。新重量化方案另记录格式误差；快照、causal mask、routing 索引和 lease/fence 属于结构正确性，不能靠放宽容差通过。普通性能路径与 profile 分开，明确 cold/warm、真实生成与固定 token、CLI 输入与 messages-json 输入。

性能成功以请求时间和实际提交 token 为准，不以 DPAS 指令出现、XMX 利用率提高、copy/kernel overlap 增加或 NLL 恰好下降作为替代。每个阶段只汇报新增可归因收益，不把候选百分比相加后承诺最终吞吐。

## 13. 本轮参考材料与勘察范围

- [官方技术报告 PDF](/D:/Downloads/Qwen3.8-Flash-Next-tech-report.pdf)，使用[已有文本抽取](/E:/RiderProjects/Aila/tmp/brainstorm-qwen38-20260927/report.txt)：混合 GDN/QSA、GR/FP8 residual、稀疏 branch 负例、PLE 与 MTP 语义。未把训练/数据中心 GPU 的速度外推为 A770 常数。
- vLLM `3137ff07739f0cd19e06119506c6432a4a7e77ad`：[Qwen4Exp 实现](/E:/RiderProjects/Aila/tmp/vllm/vllm/models/qwen4_exp/nvidia/model.py)、[low_latency_gemm](/E:/RiderProjects/Aila/tmp/vllm/vllm/models/qwen4_exp/nvidia/low_latency_gemm.py)、[HC 算子](/E:/RiderProjects/Aila/tmp/vllm/vllm/models/qwen4_exp/nvidia/ops/hc.py)、Marlin 分桶。启发是专用形状路径与分桶，TP=4/Hopper/Blackwell 的参数不复制。
- llama.cpp `95887577ab5fead779581a7030a83c7752ff3234`：SYCL 非线性量化查表、Q8 activation 整数点积；不能宣称它已直接支持 Aila NF4。
- Strata `a79080535d1b2a71a3419a0d97d8e7dca194b0f1`：实际 QSA attention 的 12-head MMA/hi-lo、prefill MMQ 与有限权重展开、host/GPU/CPU 专家分工、窗口与临时区布局。其代码和速度报告分别引用，不冒充本机复现。
- [A770 skill](/C:/Users/Administrator/.codex/skills/A770-Kernel-Optimization/SKILL.md) 的硬件、低比特 DPAS、执行模型和 SDPA 记录：只使用适用的 DG2 方法；no-mask D128 SDPA 的历史测量不是本模型 causal D256 sparse-attention 的分发依据。

**推荐给下一位实现者的起点：不要继续扩展现有大函数里的实验开关。先兑现用户入口的性能配置，再用同一个自然 prompt 证明大窗口能减少专家重传，同时建立一个成熟矩阵乘的 MoE 对照。接下来从完整 head Q8 与 12-head XMX attention 中选择已获局部证据的一项集成。**
