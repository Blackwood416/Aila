# Qwen3.8-Flash-Next：执行模型审计与 100 / 20 tok/s 路线

用户后续明确：100+ Prefill 面向中长输入，短输入不设相同吞吐门槛。最新进度见 [P2 供给隔离与 P3 固定段测试](qwen3.8-flash-next-p2-supply-p3-front-results-2026-09-29.md)：512/1024 暖 canary 已超过 100 tok/s，decode 20 tok/s 仍待实现。

实施记录：[P0 观测、P1 首候选筛选及 P2 小批对照](qwen3.8-flash-next-p0-p2-results-2026-09-29.md)。下文仍保留审计时的判断；最新测量与未完成的默认准入以实施记录为准。

日期：2026-09-29。代码基线：0293b3b。目标硬件：当前 A770 16 GB / Windows / oneAPI 2026.1 / 驱动 32.0.101.8860；模型为 Qwen3.8-Flash-Next-BNB-NF4-v2。推理计算继续放在 GPU，CPU 仅负责 I/O、缓存管理和提交。本轮读取实现及 X2–X4 证据，并运行三个串行、有超时与可用内存保护的诊断进程；没有修改运行时代码、重建或推广实验开关。

**结论：下一阶段应从“逐个算子提速”转向“减少专家重载和执行停顿”。256-token Prefill 到 100 tok/s 有明确的优化空间，但尚未证明能达标；decode 到 20 tok/s 需要同时优化数据供给与提交路径，MTP 推测解码是值得单独验证的结构性候选。不能承诺任意冷请求都达到这两个数字。**

## 1. 目标差距与测量口径

[MEASURED / DERIVED] 下面使用 X4 同进程预热、固定 continuation token 的三轮记录，逐轮计算 decode 时间再取中位数。dense XMX/direct I/O 开启，grouped expert XMX 与 chunked GDN 关闭，max_seq=512。是 backend wall time，不是服务 TTFT，也不是系统冷启动。

| 工作负载 | 现有时间 / 速度 | 目标 | 需要缩短的时间 |
|---|---:|---:|---:|
| story 30-token Prefill | 1370.18 ms / 21.89 tok/s | 300 ms / 100 tok/s | 78.1% |
| lighthouse 256-token Prefill | 3627.87 ms / 70.56 tok/s | 2560 ms / 100 tok/s | 29.4% |
| story 后 47 步 decode | 73.77 ms/token / 13.56 tok/s | 50 ms/token | 32.2% |
| lighthouse 后 7 步 decode | 96.63 ms/token / 10.35 tok/s | 50 ms/token | 48.3% |

长输入后的 decode 这里只观察了 7 步，不能称为长时间稳态。30-token 和 256-token 的差距也不能直接拟合成一个固定开销公式：路由集合、cache 状态和算子 dispatch 都不同。

后续主验收至少同时报告：256/512/1024 输入的 Prefill；相应上下文下固定 128 个生成 token 的 TPOT 均值和 p50/p95；新进程首请求与同进程暖请求。30-token 的 TTFT 独立报告。多 session 合计吞吐不能替代单请求 20 tok/s，缓存命中的历史 token 不能算成新处理的 Prefill token。

## 2. 当前真实执行模型

模型有 48 层：36 层 GDN + 12 层 full/QSA attention；hidden=2560，每层 512 experts、每 token top-10、expert intermediate=640。每个 routed expert 的 gate/up/down 权重共 4,915,200 个参数，packed NF4 + nested codes 的传输槽为 2,534,400 B。

每次 forward 的主要顺序如下：

~~~text
token IDs → host embedding gather / 异步 PLE prepare → GPU hidden4
  逐层 L=0..47：
    HC → attention（GDN 或 full/QSA）→ HC → 实际 router
    → 下一层 router 预测 → shared expert（已提交在 compute queue）
    → 主机等待实际 router，读取 pinned ids/weights
    → 主机查 cache、处理缺失专家；独立 copy queue 提交 NF4 H2D
    → routed expert gate/up、SwiGLU、down、merge → residual
  final HC → LM head → sampling；按策略捕获快照
~~~

[CORRECTNESS / CODE]

- Compute queue 是 in-order；copy queue 独立、同样 in-order。shared expert 可以覆盖一部分 host fill，copy 可与 GPU 计算重叠，但同一 compute queue 中的各分支并不自动并行。
- 每层实际 router 的 event 都要返回主机。即使专家全在 GPU，当前仍需要这次主机衔接才能构造后续调用。
- 现有 prefetch_async 主要把数据准备到 host pool。它过滤 GPU resident 和 host Ready 项，后台完成后发布 host Ready；没有在这个回调里把预测专家提前放入 GPU。真实 H2D 在 acquire 时触发。旧文档把它泛称为“跨层 DMA 预取”不准确。
- acquire 能逐个专家完成 fill 后提交 copy，但函数会等本批 host fill 完成才返回整批 DMA barrier。默认 Prefill 的 demand_microbatch=0，因此本层所有独特专家准备后才提交 expert-major 计算。B6 小批路径已存在且默认关闭，不能再把“有 microbatch”本身当作新成果。
- decode 主路径不再每个 expert/down 后无条件同步，但每层 router 边界仍在。Engine 也明确把 Q4E 的 decode_chunk_size 固定为 1；仅调大 CLI chunk 参数不会消除这些往返。
- 默认 dense prefill 仍是异步整矩阵解量化 + oneDNN，direct I/O 已默认开启。AILA_Q4E_PREFILL_XMX=1 才启用 X1 的混合 shape 策略。短 M 下仍有 48 次有意保留的解量化；M≥64 的大 N 也有 oneDNN 回退，不能把所有剩余调用都视为漏用 XMX。

主要代码位置：Context.hpp:25、Qwen4ExpBnb4Backend.cpp:372/1585/1647/1675/1700/2277、Qwen4ExpExpertCache.cpp:1190/1596/1663/1719、EngineGeneration.cpp:983。

## 3. 本轮诊断发现

[MEASURED] 三个进程都退出 0；未跑并行 GPU 压力。以下单次样本用于解释执行模型，不用来宣传优化加速比。

| 场景 | Prefill wall | GPU cache hit / request | demand H2D payload |
|---|---:|---:|---:|
| 新进程 story，默认 oneDNN，已有 2-token 初始化 warmup | backend 3841.85 ms | 336 / 4866 | 10.69 GiB |
| 新进程 lighthouse 256，显式 dense XMX，第一次 | 外围 11959.1 ms | 456 / 10341 | 23.33 GiB |
| 同进程相同 lighthouse 256，reset 后第二次 | 外围 3920.62 ms | 139 / 10341 | 24.08 GiB |

第二次 256 forward 的 backend 区间为 3915.97 ms，两个全词表 logits hash 相同，mmap fallback=0。它证明即使同输入重复执行，当前有限 cache 的逐层扫描仍产生大量重载。第一次更慢不能仅归因于 H2D 字节，因为它实际传得稍少；文件页、host pool、首次 shape 编译与 primitive 准备没有在本轮进一步隔离。

X3 的另一种暖缓存状态中，同一 256 输入有 1265/10341 次 resident hit；因此 139 不是这个 prompt 的固定命中率，更不能直接说 cache “坏了”。但两组记录都说明不能把 9 GB 的 GPU expert pool 当成已充分发挥作用。

新进程 story 的真实 47 步 decode 平均为 127.24 ms，平均每步 401.89/480 命中、搬运 188.78 MiB，整请求 9914.01 ms；输出末尾为“... As the sun”。这组是普通启动、自然生成、部分冷缓存，与 X4 的显式 XMX、反复预热、固定 token 路径不同，不是对 X4 的回归判定。前三个进程的初始化 warmup 本身约 25–28 秒，也不算进 Prefill；首次 shape 准备需要单独管理。

### 不能误读现有 profile

- 第二次 256 的 router_sync=2468.68 ms，**不是 router 算子耗时 2.47 秒**。host 在这个 event 上还会等此前排队的 attention/MoE/DMA 依赖；这里是完成时间的观察点。
- acquire=1320.92 ms、fill=1077.36 ms 也不等于可直接回收的 GPU idle。区间存在嵌套与重叠。
- 本轮额外开启逐 stage 强制同步，story decode 变为 160.47 ms/步；其中 HC=18.56、linear attention=12.25、router stage=13.47、expert stage=95.10 ms。router stage 还包含已先提交的预测与 shared expert。它们用于定位复查对象，不能相加成异步模式的归因或把差值称为纯 launch 开销。
- 本轮没有取得实际请求的逐 kernel / copy 重叠时间线，因此尚不能给出“GPU idle 占比”或“graph 能省多少 ms”。既有 fake-hit 模式会使用错误权重、改变后续路由，不是正确模型的可达性能下界。

## 4. 字节预算约束了哪些目标

[DERIVED] GPU expert pool：3714 槽 = 9,412,761,600 B；host pool：6778 槽，约 16 GiB。全部 48×512 个 experts 合计 62.29 GB，显然需要按需供给。GPU 与 host 都持有部分重复专家，二者容量不能简单相加成独特权重覆盖率。

### Prefill

X3 真实路由中，30/256 输入分别触及 4869/10341 个不同的 layer-expert。

- 完全冷 GPU expert cache 至少需要 12.34/26.21 GB H2D。
- 即使启动时的 3714 槽全部恰好有用，不计执行中的替换干扰，也至少需要 2.93/16.80 GB。这个容量下界不是已实现的 cache 算法，更不是可消除所有这些传输的承诺。
- 30 token 在 300 ms 内处理完，冷专家传输本身要求 41.13 GB/s。旧机器文档记录 PCIe 3.0 x16；即使只计编码、不计协议损失，上限也约 15.75 GB/s，12.34 GB 需要至少约 783 ms。**在这一硬件条件、该路由和当前 NF4 payload 下，30-token 冷请求 100 tok/s 不成立。** 本轮未重查链路协商状态；更窄链路只会进一步收紧限制。
- 当前第二次 256 forward 传输约 25.856 GB；保持字节数不变，要在 2.56 秒结束，要求平均供给至少 10.10 GB/s，而且这个预算还没留出未重叠的计算和调度时间。仅调快 GDN 无法解决它。

因此长 Prefill 应同时减少 byte/token、减少等待、提高 ready 数据上的计算速度。512/1024 token 可以摊薄同一专家首次加载，但只有实际路由复用才能产生收益；padding 或重算不能计作吞吐。

### Decode

每 token 有 480 次专家激活，约 1.217 GB 专家权重逻辑读取。GPU 命中时这部分走显存；仅 miss 需要 H2D。50 ms/token 的目标必须同时容纳静态权重读取、attention、host I/O、提交及采样。

以下只是敏感性预算，不是本机带宽测量：

| 假设有效 H2D | 若把全部 50 ms 都给传输：最低 hit rate | 若传输窗口预算为 20 ms：最低 hit rate |
|---|---:|---:|
| 6 GB/s | 75.3% | 90.1% |
| 8 GB/s | 67.1% | 86.8% |
| 10 GB/s | 58.9% | 83.6% |

传输可以与计算重叠，20 ms 是设计情景，不是物理硬门槛。实际还受逐层依赖、miss 分布和 host Ready 比例约束。不能说“平均命中率达到 90% 就一定 20 tok/s”；也不能拿 A770 560 GB/s 显存规格代替 PCIe 带宽。

## 5. 建议改变的执行模型，按投入顺序

### P0：建立可决策的关键路径记录，补齐默认配置准入

一次固定输入诊断只记录四类：实际 kernel、copy、GPU 等待间隙、host route/acquire/submit；用实际事件时间线识别重叠，避免逐 stage 全局 wait。同时记每层 GPU ready / copying / host ready / file fill、demand/prefetch bytes、命中及取消。计数在 request 级归零；区分实际请求与初始化 warmup。

dense XMX 的候选收益已有证据，可优先补齐代表形状与模型质量检查，决定是否把 X1 的 shape 策略用于默认启动。保留明确获胜的 oneDNN 形状。这个工作能兑现已实现能力，但不应预期它独自补齐约 30%–50% 的总时延差距。

### P1：给扫描型 Prefill 设计 cache admission，保留 decode 热专家

[HYPOTHESIS] 用同一 3714 槽划出可调整的热集与流式工作区。一次性的 Prefill 专家使用工作区，不因出现一次就淘汰跨 token 热集；明确 admission / bypass / promotion，并由真实复用、每层 miss 暴露代价决定热集，而非固定每层 77 个或扩大 RAM。

现有 prefill_demote 是使用后的 LRU 次序调整，不能等同于 scan-resistant admission。固定离线 static pinning 已有负结果，不原样重开；新候选必须在真实 trace 上解释它如何减少替换和传输。

同时检查 host/GPU 重复副本：host 优先保留可能产生昂贵 file miss 的 backing，允许在有安全 fallback 时淘汰部分 GPU 常驻项的 host 副本。保持 16 GiB host 总预算，不以恢复 mmap 大 working set 或加到 32–40 GiB 解决问题。

最小决策：同一真实路由的 trace 重放，对照当前策略和容量/离线最优参考，再做一个保持算术不变的实机 canary。现有 q4e_cache_sim.py 按 acquire 数目识别 decode，且未完整模拟当前 demote、异步 prefetch、host fences；须先修正模型，不能直接沿用其旧命中率结论。以 byte/token、file-miss 和总时延同时改善为准。

### P2：把同步 acquire 变成持续供给 GPU 的有界任务流水线

[HYPOTHESIS] 目标顺序为：实际路由得到后立即提交 ready 专家；缺失专家生成有界 demand ticket；I/O completion 发布 readiness，达到有效批量便提交下一批计算。将“本层全准备好才返回”的边界改为“已准备好即可消费”。

先利用现有 B6 骨架测一两个代表批量；若固定小批的额外 launch 抵消收益，就保留默认，转向 readiness 驱动的批量合并。不要让每个专家变成一次独立 GEMV launch。预分配描述符和 ticket，GPU reader / host DMA reader / generation 继续保护复用，取消时排空已提交任务。

主机供给有一个具体可优化点：默认 fill_parts=8，compact reader 每 part 依次读取 gate/up packed、其 codes、down packed、其 codes，即完整专家通常有 32 次 ReadFile；ERROR_IO_PENDING 后调用 GetOverlappedResult(TRUE)，单 worker 对这些片段仍是阻塞顺序。它提供线程并行，尚不是批量异步 I/O completion 管线。

首选按批量调整 part 粒度：大量独特专家时利用专家间并行，少量关键 miss 才考虑专家内切分；减少小 codes I/O 和反复 job/cancel 同步。另一个候选是离线生成连续的 expert-slot 文件，一次读入 GPU 所需 NF4 + codes 格式；保持原数值和原模型文件，以明确版本/哈希校验 sidecar，不做 host BF16 展开。现有 CPU dataflow 中“每完整专家一个任务”的 I/O 组织可独立复用，不恢复 CPU expert 点积。

预测 GPU DMA 曾有负面实验，不把它列为立即启用的开关。先优化真实 demand 的 ready 流水线，只有明确证明缺的是可覆盖的 PCIe 窗口，且 slot 版本和 reader 依赖完整，才单独重审更早的 H2D。

### P3：decode 缩短提交链，再判断是否需要 device 驱动的路由消费

[HYPOTHESIS] 先缓存固定形状的 layer 内静态计算段，使用固定地址参数块传入 position/state，减少大量小 kernel 的 host 提交。边界放在必须交给 host 的路由/缺页点。graph 不会消除真实数据依赖，也不把 in-order queue 直接改成 out-of-order 期待自动正确并行。

DG2 技能中存在 limited/static graph 与参数块的功能证据，但不代表当前驱动下整个 Q4E 能安全捕获。需重查 aspect、oneDNN 参与方式和具体多节点 replay。若 GPU 时间线显示提交间隙不大，应停止这条线，而不是全面重写。

更进一步的候选是 device resident directory：实际 router 后，GPU 直接查 slot/version，ready hit 在 GPU 消费；miss 才走 host ticket。当前每层 host wait 可以因此有机会减少，但需要有界的 miss 停止/恢复、分段 graph 或条件执行设计。它不是“把 pinned flag 放到 host 上轮询”这么简单；初版不做长驻自旋 kernel，也不承诺把 48 次边界一口气降到零。

G1 已有 HC 专用 decode kernel，G2 已有 GDN single-write，不再把它们重复算作未来收益。先从新时间线选真正昂贵的剩余 kernel；fused residual、全 BF16 expert decode、每 subgroup 两输出行等已有负方向继续避开。

### P4：MTP / 推测解码，改变每次主模型遍历产出的 token 数

[CODE] 当前模型 config 含 1 层 MTP，safetensors index 中有 86 个 mtp.* tensor 条目；Q4E backend 没有加载/执行 MTP 的路径。存在权重不等于已证明 draft 质量、可用吞吐或完整语义。

[HYPOTHESIS] GPU MTP 生成短 draft，再由主模型批量验证 2–4 行，有机会复用 NF4 专家权重并摊薄 48 层的路由/提交。这是接近或突破 20 tok/s 的另一条结构性路线，不能用普通 decode_chunk_size 代替。

先测一两个 token 的 draft acceptance、draft wall、完整 verification wall，再看：

~~~text
有效 ms/token = (draft + verify + rollback/commit 的完整时间) / 实际提交给用户的 token 数
准入目标：< 50 ms/token，且真实质量/分布契约满足要求
~~~

这里只给判断公式，不预设 acceptance。更改采样时需要正确的接受/拒绝算法；greedy 验收也要检查每步 target 决策。GDN/conv/PLE/KV 必须能恢复到接受前缀；不能每个候选都复制约 112 MiB state 到主机。应评估有预算的 device checkpoint / rollback，额外 MTP 权重与状态会挤占 expert cache，必须一并计入。若总验证成本或拒绝率抵消复用，即停止。

### P5：在供给与提交改善后，再攻 grouped XMX 和长上下文

X3 的首版在热专家占多数的 256 输入仍慢，因此不能用“矩阵太小”解释全部失败。若继续，应先分开 tile 解量化、SLM layout、barrier、寄存器/指令成本，并加入 2–4 expert 的 GPU BF16 staging + 已预建 GEMM 对照；包含 gather/expand/scatter 全路径，H2D 仍传 NF4。一次改变一个变量，不能把一版 grouped kernel 失败推广为 XMX 无望。

X4 FP32 chunk 的算子收益成立，整模型只有约 1.2%，长文 logits 又需要进一步质量定位，因此其默认化与 T32 等扩展不占本轮第一优先级。NF4 codebook 不是线性 INT4，不能直接把 nibble 交给 INT4 DPAS；重量化属于另一个质量项目。

当前 256 canary 不覆盖 QSA sparse 阈值。后续 2051/2052 边界及更长上下文要另列 attention/indexer/KV 预算。Engine 已有默认 512 的外层 chunk；不再重复提出“先支持512”。适当扩大 chunk 或改为层内算子分块时，同时测临时 buffer、调度公平性和 cache slots，避免为吞吐重新制造长文 RAM/VRAM 压力。

## 6. 执行门槛

建议第一个工作包止于 P0 + P1 的 trace/单候选，再进入 P2；不是同时铺开六条实现。先把 256 Prefill 从约 3.63 秒压到 2.56 秒内，把暖 story decode 从约 74 ms 压到 50 ms 内，同时验证长输入后的 decode。短冷请求和新进程首请求保持单独报告，不以暖 canary 隐藏它们。

P3/P4 的探索由实际剩余瓶颈决定；对 MTP 可先做小成本可行性检查，达到 acceptance/cost 门槛才投入完整状态实现。所有性能结论使用相同有效配置、输入、生成工作量及 cache 条件；存储/调度重构首先保证同权重同路由语义，算术变更继续使用已有 tolerance 和状态契约。允许的局部收益不能相加成未测量的端到端速度。

证据：[本轮 JSON](qwen3.8-flash-next-execution-model-audit-2026-09-29.json)、[X2](qwen3.8-flash-next-x2-results-2026-09-29.md)、[X3](qwen3.8-flash-next-x3-results-2026-09-29.md)、[X4](qwen3.8-flash-next-x4-results-2026-09-29.md)。JSON 保存本轮全部 profile 行、source/exe/stack 身份、日志哈希、内存摘要、计算公式及诊断 runner。三个原始日志前缀为 tmp/execution-{story,long,stages}-20260929；进程峰值 working set 约 21.1–21.2 GiB、系统最低可用 RAM 至少约 28 GiB。它们是进程/系统观测，不等同于 DXGI resident VRAM。
