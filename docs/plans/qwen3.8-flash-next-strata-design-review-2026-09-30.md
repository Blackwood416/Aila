# Qwen3.8-Flash-Next：Strata 执行设计对照与可迁移方案

后续执行状态：[S1/S2 与 S3 条件判断](qwen3.8-flash-next-strata-s1-s2-results-2026-09-30.md)、[S4 实现与 S5 筛选](qwen3.8-flash-next-strata-s4-s5-results-2026-09-30.md)。以下保留源码审阅时的设计与假设，实施结果以这些后续报告为准。

2026-09-30。审阅本地 `tmp/Strata`，commit `a79080535d1b2a71a3419a0d97d8e7dca194b0f1`（0.1.27），其工作区干净。Aila 为 `0293b3b` 加当前 P0–P4/HC/MTP 未提交改动。本轮只阅读源码、核对既有结果和计算预算，没有运行 Strata、GPU benchmark 或改变推理默认值。源码身份与计算输入见同名 JSON。

**Strata 值得借鉴，最有价值的是把“小窗口验证”作为独立执行路径，并分别优化 draft 历史补齐、接受前缀状态提交和 Prefill 内存生命周期。它使 MTP 出现了新的具体实验方向，但并未推翻 Aila 现有验证成本不足以达到 20 tok/s 的测量结论。**

本文的 `[CODE]` 是本次源码观察，`[REPORTED]` 是 Strata 仓库公布的数据，`[MEASURED/既有]` 是 Aila 前轮实测，`[DERIVED]` 是注明假设的计算，`[HYPOTHESIS]` 是尚未在 A770 上验证的设计。没有把外部 GPU 的性能常数当成 A770 分发规则。

**先分清其性能来自哪里。**

[REPORTED] [0.1.26 基准协议](/E:/RiderProjects/Aila/tmp/Strata/bench/results/2026-09-29-speed-0126/README.md:3) 为 RTX 5070 12 GB、文档所记 PCIe 5.0 x16、Ryzen 7600、64 GB DDR5，单请求 greedy 生成 256 token，MTP `--spec 4 --spec-min-p 0.5`，Prefill auto，大上下文启用量化 KV/streaming。每个表格格子只有一次请求。Q2_0 的 32K Prefill 为 2171 tok/s、4K 后 decode 为 93 tok/s；这些不是本机复现值。

[CODE] 其 decode 把专家分为 GPU resident、GPU PCIe staging、CPU 三部分，在 [expert_source.cpp](/E:/RiderProjects/Aila/tmp/Strata/src/core/expert_source.cpp:803) 构造分工，在 [verify.cpp](/E:/RiderProjects/Aila/tmp/Strata/src/core/verify.cpp:657) 消费。CPU 处理 miss 是核心路径的一部分。低比特 GGUF、较小专家 payload、整套 host resident 权重、CUDA 实现和 MTP 共同决定结果，不能拿 93/14 的比值声称 Aila 只差一个优化开关。大块 Prefill 的 routed experts 则在 GPU 上计算，因而其 Prefill 组织更适合直接参考。

用户的约束保持不变：专家算术全部在 GPU；CPU 只做 I/O、调度和离线校验；host expert pool 保持有界。Strata 文档中的 35–55 GB 常驻/锁页 RAM 和启动时系统卡顿，不能作为解决 Aila 长文内存问题的方式。其 README 也明确单请求服务；借用共享专家缓存的设计必须适配 Aila 的多 Session 调度。

**1. 最直接的新机会：MTP 历史补齐只算 K/V。**

[CODE] Strata 的 [MtpDrafter::record_forward](/E:/RiderProjects/Aila/tmp/Strata/src/core/mtp.cpp:332) 把执行分为完整一步和 K/V-only 两种模式：embedding/hidden norm → 两个 FC → attention HC → K/V projection、K norm/RoPE、KV append；[完成 append 即返回](/E:/RiderProjects/Aila/tmp/Strata/src/core/mtp.cpp:388)，不做 Q、attention、o_proj、FFN、router、MoE、mixer 和 head。[capture_round](/E:/RiderProjects/Aila/tmp/Strata/src/core/mtp.cpp:530) 先批量补 K/V，再从最后接受行跑完整 draft。

Aila 当前 [MTP Step](/E:/RiderProjects/Aila/tests/models/Qwen4ExpMtpProbe.hpp:149) 的 `head=false` 只跳过最后 mixer/head；此前完整 attention 和 MoE 仍执行。[selected-cycle catch-up](/E:/RiderProjects/Aila/tests/models/Qwen4ExpMtpAcceptance.hpp:133) 对每个补齐位置调用它。这是实在的可删工作，不是重复此前 512/64 槽 sweep。

[HYPOTHESIS] 第一项实验应增加测试专用 `AppendTargetKv`：先按原来的单行算术边界早退，再考虑 2–4 行批量补齐，避免同时改变多项结构。可先复用现有 fused QKV 投影然后在 append 后返回；第二步再为 K/V 增加受原权重生命周期保护的视图/分发，消除无用 Q 投影。当前 <640 的 dense 测试不需要消费 draft indexer，但扩展稀疏 draft 时必须重新定义其状态需求。

验收：同一目标 hidden/token/position 下 K/V 和后续 draft 对照；相同 greedy 轨迹、相同 64 槽与显存预算；保留首次与稳定区间；记录 draft+catch-up wall、expert miss、H2D、live USM。第一版若保留原算术可要求 K/V 存储一致，后续批量算术继续使用既定容差。既有 CPU reference 在位置 2/3 未通过 3% NRMSE 的记录继续保留，不能被这项局部一致性检查替代。

**2. 状态提交应重放递推，不应重跑整个目标前缀。**

[CODE] Strata 验证时保留每层 QKV、conv 后输入、gate/beta 等。其 [GDN multi kernel](/E:/RiderProjects/Aila/tmp/Strata/src/kernels/cuda/verify_kernels.cu:114) 读取初始 recurrent state，跨 T 个 token 递推；只有传入 `n_keep` 时才写回 state。[Verifier::capture_commit](/E:/RiderProjects/Aila/tmp/Strata/src/core/verify.cpp:881) 对接受前缀执行：

- 由保存的 QKV 提交 conv history；由保存的递推输入只更新 GDN state。
- 恢复 QSA indexer 的小 tail，重新 append 接受的 keys，避免拒绝部分污染仍会用到的块。
- PLE 选择接受末行的 history snapshot；普通 KV 的未接受尾部由可见长度排除。

它没有重做 48 层 HC、投影和专家计算。Aila 目前的 112.57 MiB device checkpoint 只证明“整块恢复后重放可重现”，尚未实现接受前缀提交；已有 interior checkpoint 是主机快照，也不能直接替代此路径。

[HYPOTHESIS] 可保存每个 GDN 层本窗口的 BF16 `linear_all`，验证在临时状态上进行，接受后以这些输入更新原始 state，逐步演进到不写原状态的 verifier。也可先复用已有 device snapshot 恢复初态，再做递推-only commit，降低首版复杂度。

[DERIVED] 本模型 36 层 × 4 行 × 16480 元素 × 2 B = **4.53 MiB**，只是原始投影输入 tape，不是完整 verifier 的内存；还需 PLE、indexer、临时状态、对齐和 descriptor 空间。它说明无需为每个候选位置额外保存一整份 112.57 MiB 状态。

不能照搬 CUDA 几何：Strata 此 kernel 的 block 是 `128×4=512` threads、每线程 32 个 state 元素；DG2 的 subgroup/GRF/SLM 和已有 TDR 负例不同。Aila 已有跨 token single-write GDN，不能把“跨行复用状态”再计一次收益；新增价值主要是推测执行的状态语义。

验收必须枚举 T=2/3/4、keep=1..T、非零初态、多轮 reject/continue、conv 环形边界、QSA 压缩块与 2051/2052 阈值、PLE history；之后才覆盖取消、会话交错和采样状态。

**3. Draft 词表与长度可自适应，但先看总成本。**

[CODE] [draft head](/E:/RiderProjects/Aila/tmp/Strata/src/core/mtp.cpp:309) 只计算一个 token 子集；[draft chain](/E:/RiderProjects/Aila/tmp/Strata/src/core/mtp.cpp:670) 在最后一步概率低于阈值时停止。实际生成循环使用 [DraftPolicy](/E:/RiderProjects/Aila/tmp/Strata/src/program/generate.cpp:4055)，在 MTP 首 token 与 suffix lookup 一致时，根据观测成本与接受率考虑 lookup。仓库另有 `spec/controller.cpp` 的通用成本模型，本次看到的生成入口调用的是 `DraftPolicy`，不能把二者混为已上线的同一控制器。其 RTX 5070 先验也不适用于 A770。

词表裁剪有实际负例：[draft_vocab.py](/E:/RiderProjects/Aila/tmp/Strata/tools/draft_vocab.py:1) 记录旧英文/代码词表只覆盖 55328 个汉字 token 中的 27 个，中文几乎无法 draft；0.1.27 加入 CJK。当前本地 `data/draft_vocab.bin` 有 **106299** 个 ID。

[DERIVED] 若在 Aila 上为它另建 BF16 head，`106299×2560×2` 需要 **519.04 MiB**，而 Aila 当前共享目标 head，并没有 draft head 副本。这会新增显存占用、挤占专家槽；不能把 Strata 的“subset 更小”直接解释为 Aila“节省 519 MiB”。间接读取共享 head 子集、紧凑副本或仅 draft 重新量化，都是不同候选，需分别计入成本和接受率。目标仍使用原 NF4 模型及完整 head。

[HYPOTHESIS] 先做代表性中文、英文、代码、非重复长文的接受率与 T=1/2/3/4 成本曲线，再决定是否缩词表/窗口。用实际提交 token 数除以完整周期时间选择长度，并允许退回 T=1；不要移植 `0.5` 概率阈值或 70%/86% 接受率先验。subset softmax 的概率也不是全词表概率。Lookup 是额外候选，重复灯塔文本的高接受率不能代表普通生成。

Greedy 时逐前缀核对目标 token；采样时需明确正确的目标采样/接受规则。Strata 生成循环比较 draft 与按位置 seed 得到的 target sample，不能将其描述成已经实现标准 `p/q` speculative rejection。Aila 的单行/多行数值差异与既定容差仍需独立评估。

**为何仍须优化目标 verifier。**

[MEASURED/既有] 采用前轮较好的 64 槽 MTP 配置、同一真实 greedy 轨迹的数据：

| 输入 | 目标四行验证 ms | draft+补齐 ms/选中周期 | 平均前进 token/周期 | 条件成本模型 tok/s | 若 draft 完全免费，模型 tok/s | 保持当前 draft 成本，20 tok/s 留给 verify 的 ms |
|---|---:|---:|---:|---:|---:|---:|
| 30-token 日出 | 181.908 | 23.466 | 2.333 | 11.36 | 12.83 | 93.20 |
| 512-token 重复灯塔 | 278.088 | 23.394 | 4.000 | 13.27 | 14.38 | 176.61 |

[DERIVED] `verify预算 = 50×平均前进token − draft成本 − commit及其它成本`；表中最后一列暂将 commit 等成本设为零。保持表内成本/接受率假设，verify 还需下降约 **48.8% / 36.5%**；即使 draft 免费，也分别需下降约 **35.9% / 28.1%**。这些是条件算术预算，不是不可突破的硬件上限，更不是新的生成测速。KV-only 会改变 cache interleaving，真实收益必须完整测量。

因此，先删掉 draft 冗余工作是低成本合理步骤，但单靠它无法让这两条轨迹达到 20 tok/s。接下来应把目标 2–4 行路径当独立 verifier，依实际时间线优化剩余 attention/GDN/router/shared/MoE/head，结合每层 expert union 与 ready/miss 成本选择下一项。Aila 已有 packed NF4 rows、HC rows、expert-major 4/2/1 复用；不再把“支持批量”“同专家合并”当作尚未实施的新方案。

**4. Prefill 与内存：先做生命周期复用，再选 chunk。**

[CODE] Strata [Prefill::carve](/E:/RiderProjects/Aila/tmp/Strata/src/prefill/prefill.cpp:571) 将 GDN、QSA 和 MoE 阶段临时区重叠分配，取三者最大值。[REPORTED] [其历史实验](/E:/RiderProjects/Aila/tmp/Strata/bench/results/2026-09-28-prefill-speed/README.md:9) 中 4096 chunk 从借用 2.84 GiB 降至 1.78 GiB；数值只适用于其 FP32/FP16 布局。

[CODE] [plan_lend](/E:/RiderProjects/Aila/tmp/Strata/src/program/generate.cpp:3018) 在 8192/6144/... 候选中按预算选 chunk；[每请求 lend](/E:/RiderProjects/Aila/tmp/Strata/src/program/generate.cpp:3899) 只借实际输入所需尾部专家槽，Prefill 结束再 refill。缓存与临时计算的显存份额随执行阶段改变。

Aila 的 [ensure_runtime_buffers](/E:/RiderProjects/Aila/src/models/Qwen4ExpBnb4Backend.cpp:680) 已按所需 cap 增长，并非全部按 max-seq 分配；但 `linear_all`、full attention scratch、MoE down accumulation 等分别保留。`moe_down_acc` 单项约 100 KiB/输入行。现有 PLE 已有异步 prepare/consume 与批量投影，Strata 的“批量 PLE”不是 Aila 的空白。

[HYPOTHESIS] 优先做分配清单与真实 lifetime 图，再用单个受 DeviceRuntime 记账的 arena/view 合并不同时存活的 scratch。后续才评估统一预算下 512/1024/2048 chunk 与 expert slots 的交换。PLE 异步准备、下一层预测、copy queue 消费以及 session 切换可能延长 lifetime，不能仅按函数调用先后覆盖。借槽必须撤销 residency、等待所有 reader/DMA、控制 refill，并由 scheduler 保证跨 session 安全；不在 decode 热循环动态 USM 分配。

这能把“长文内存”与“提升 Prefill”放在一个预算中决策。保持已修复的 ReadFile 和 16 GiB host pool，报告峰值 WS、系统可用 RAM、device ledger/实际 VRAM、有效 cache slots 与 Prefill→decode 切换成本。不能将 OS 可回收 mmap 页当成零内存，也不能把借出显存当成新增容量。

**5. Prefill 提前供给：值得借鉴机制，不能照搬全量预取。**

[CODE] Strata [stream_all](/E:/RiderProjects/Aila/tmp/Strata/src/prefill/prefill.cpp:1073) 在 chunk≥2048 时按 `(layer, expert)` 顺序，把所有非驻留专家送入有限 ring，不等待实际 router；独立 issuer 线程负责发 copy，reader 的 used event 控制槽复用。路由未使用的专家也已搬运，只在 [消费时释放](/E:/RiderProjects/Aila/tmp/Strata/src/prefill/prefill.cpp:1695)。其动机是大 chunk 几乎用遍专家，从而在 attention 期间持续供给。

这不同于准确预测下一层路由，也不是零额外流量的预取。其 ring 96/384 和 2048 阈值依赖权重大小、host pin 比例及硬件；Aila NF4 每槽约 2.42 MiB，384 槽本身约 **928 MiB**，全部 24576 专家约 62.29 GB。当前 256 样本只触及 10341 个 layer-expert，不能原样启用全扫描。

[HYPOTHESIS] 可先在既有 trace 上计算候选大 chunk 的“额外传输字节/真实需求字节”、host file-miss、重用距离，只有覆盖率支持时才试有限跨层流式读取。更近的迁移点是把缺失数据等待从 kernel 提交线程挪给有界 issuer，用 readiness/reader fences 连续供给。Aila 已有 B6=32 的分批 demand 与异步 reader，不是从零实现流水线；此前 copy slice/bulk 无 wall 收益，也不能被 Strata 的结果推翻。只针对时间线确认的提交停顿做新实验。

**6. 保留为后续课题的部分。**

| Strata 设计 | 对 Aila 的判断 |
|---|---|
| GGUF MMQ、Q8 activation、int8 tensor cores | 借鉴保持低比特存储和专家分组的组织；NF4 LUT 不是线性 INT4，不能把 nibble 直接送 DPAS。未来 grouped NF4→BF16 tile/XMX 或 NF4→INT8 另立实验；现有 X3 负结果不重跑原版。 |
| 整窗口 CUDA graph + doorbell/flag | 不能直接迁移。源码仍有每层 host 服务及 GPU 等待；不是完全 device autonomous。Aila 固定 front graph 已测慢；Strata 还记录 Windows DMA/driver lock 死锁后改 copy kernel。继续保留显式事件，不引入长驻 GPU 自旋来凑“零同步”。 |
| device resident-plan | 当前源码由环境变量显式启用，只对该组专家全部 resident 时 bypass host plan；不会自动解决 miss。先用实际路由计算 all-hit 层比例，再估值。 |
| 自适应专家热集 | 源码每层按 decayed usage 有条件 swap，缺失项可由 CPU 暂时计算。Aila 不能照搬该退路；已有 SLRU/static pinning 负例，须以 miss bytes、file-miss 和 wall 验证。 |
| host/GPU 互补存储 | FileExpertSource 有只常驻 GPU 补集的模式，但 Aila 全部 NF4 约 58 GiB，扣 GPU 后仍超过当前 16 GiB host 预算；只能借鉴消除无用重复副本的原则，不能全量常驻补集。 |
| K8V4 / KV streaming | 更长上下文再做。Aila 当前 BF16 KV+indexer 每 token 约 27.75 KiB，128K 约 3.47 GiB，可显著影响专家槽；512 上仅约 13.88 MiB，不是本轮小窗口瓶颈。压缩需单独质量验收，streaming 需计算 RAM/PCIe 与 QSA 访问工作集。 |
| experimental speed projection/control vector | 会改变目标行为；不纳入本次执行优化。 |

源码入口：[MMQ](/E:/RiderProjects/Aila/tmp/Strata/src/prefill/moe_mmq.cu:90)、[graph host loop](/E:/RiderProjects/Aila/tmp/Strata/src/core/verify.cpp:987)、[Windows DMA 负例](/E:/RiderProjects/Aila/tmp/Strata/src/program/generate.cpp:4685)、[resident plan](/E:/RiderProjects/Aila/tmp/Strata/src/core/verify.cpp:619)、[adaptive swaps](/E:/RiderProjects/Aila/tmp/Strata/src/program/generate.cpp:4715)、[host complement](/E:/RiderProjects/Aila/tmp/Strata/src/core/expert_source.cpp:458)。部分头文件仍保留早期“尚未接入”的注释，本报告以当前生成入口的调用链为准。

**建议的执行顺序。**

| 顺序 | 有界工作包 | 决策依据 |
|---|---|---|
| S1 | MTP KV-only catch-up；沿用 64 槽与原轨迹做局部等价/成本检查 | 少算完整 MoE 是否降低周期成本，是否改变 target cache 干扰；不承诺 20 tok/s |
| S2 | 代表性文本 + 小窗口成本/专家 union 记录；选择剩余 verifier 最贵的一项 | 以真正接受前缀和完整周期预算选 T，测清拒绝与 H2D；不重复无改动 fixed128 基线 |
| S3 | 若 S1/S2 显示足够收益空间，做 GDN 输入 tape + recurrence-only commit | 任意接受前缀的状态/continuation 成立，额外显存和 commit wall 计入；再决定生产 MTP 接入 |
| S4 | Prefill scratch lifetime 复用，之后统一 chunk/cache/ring 预算 | 在保持 RAM 上限下测中长 Prefill 与后续 decode；不把 short cold 100+ 作为门槛 |
| S5 | 只在对应瓶颈成立后，尝试子词表、有限跨层供给、grouped XMX 或 KV 压缩 | 每次一个结构变量；记录包含准备/恢复/拒绝成本的收益 |

本轮产物为此源码对照报告和预算 JSON，并链接回原规划；实现与测量留给上述后续工作包。已有未提交代码保持原状，未新增 CPU 专家计算、未提交或推送。
