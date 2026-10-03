# Qwen3.8-Flash-Next：以容差为契约的 XMX Prefill 优化方案

2026-10-03 最新进展：[H6 Ready物理槽接管与供给复核](qwen3.8-flash-next-hotset-h6-ready-results-2026-10-03.md)。接管无需额外H2D，短/长GPU teacher48逐位一致，CPU平均KL约0.00185/0.00225，checkpoint/取消恢复通过。三组同二进制跨域对照未获稳定端到端净收益；长文需求/epoch流量下降，GPU冷供给和prefetch等待仍需治理。候选默认关闭、32 GiB预算与原preset不变，R2b仍暂停，100+/20目标尚未完成。下文保留历史结果。

2026-10-02 最新进展：[H4/C2首轮迁移与验收完成](qwen3.8-flash-next-hotset-h4-c2-results-2026-10-02.md)。GPU teacher forcing逐位一致，CPU短/长平均KL约0.00188/0.00232，无NLL预警，checkpoint/取消恢复通过。长文暖TTFT约19.99秒、有效输入/TTFT约91.7 tok/s，decode中位数11.3；尚未达到100+/20。profile与自适应需要计入初装/借槽恢复/适应成本，候选显式开启、32 GiB预算不变，R2b尚未恢复。以下保留历史结果。

2026-10-02 最新进展：[C1/H3 首轮完成](qwen3.8-flash-next-hotset-c1-h3-results-2026-10-02.md)。同冻结热集 CPU cold 的短/长 decode 中位数7.86/7.63 tok/s，仍低于原动态13.90/8.75；H3 未取得稳定吞吐收益。冻结集合的同prompt暖长TTFT约20.3秒，但有前置校准成本，不能作为冷启动或跨域100+ prefill的结论。代表性质量与状态检查通过，候选默认关闭；下一步H4/C2，R2b继续暂停，保持32 GiB研究预算。

2026-10-02 按 [hotset新优先级规划](qwen3.8-flash-next-hotset-priority-plan-2026-10-02.md) 暂停R2b后续实施，先完成 [H1/H2动态驻留直接执行首轮](qwen3.8-flash-next-hotset-h1-h2-results-2026-10-02.md)。正确性与状态检查通过；当前GPU-only候选短decode由14.0降至12.6/13.1 tok/s，长decode由8.95降至8.4/8.75 tok/s，未默认开启。下一步是新架构下C1冷专家CPU/GPU分工的同热集对照，保持32 GiB研究预算。

2026-10-01 补齐 [R2b 真实隐藏态、单输入模型质量与状态检查](qwen3.8-flash-next-r2b-real-quality-results-2026-10-01.md)：六组真实专家通过整数 CPU/RTE 参考及连续 graph 参数表/scatter 检查；1833-token 输入的 48 步 teacher forcing 平均 KL=0.00701、NLL 差=−0.00115，42/48 首选 token 一致，通过预设模型 canary 门槛。257-token 前缀因果、65-token 快照/恢复回放及 3 项 CTest 通过。但原 NF4 逐点门槛在真实样本中有 18.9%–46.9% 超限，不能称全面质量认证。所有 count>1 prefill 专家统一整数格式，decode 仍 NF4，禁止热度决定数值格式；诊断重复计算 NF4，未产生生产 TPS 收益。32 GiB host pool 下最终最低可用 RAM 10.99 GiB。下一步先用真实语料验证 down per32 激活尺度及小桶一致算术，再去除重复计算接入；R2b 生产集成、decode 20 tok/s 尚未完成。

2026-10-01 继续完成 [R2b INT8 DPAS 热桶与静态 graph 实验](qwen3.8-flash-next-r2b-dpas-results-2026-10-01.md)：原生 dpas.8x8 已由实际 ISA 确认；两次独立运行中，含转码、两次 Q8、pack/merge 的完整链对同 graph NF4 基线，M32 延迟下降约 26%，M48 约 24%–26%，M64 约 47%。M8/M24 落后、M16 余量不足，WG32/stride padding 没有一致收益，不写泛化阈值。CPU 算子参考、修改输入/路由/专家描述符后的 graph 逐位检查、3 项 CTest 通过。新算术仍有少量逐点超限，未做模型 NLL/生成验收，保持测试入口；下一步真实隐藏态质量与可复用热桶 graph 集成，未宣称整机 TPS 提升。

2026-10-01 完成 [R2b NF4 integer-LUT × Q8 的有界 DP4A 实验](qwen3.8-flash-next-r2b-integer-results-2026-10-01.md)：真实专家整数参考、Q8 RTE/非有限保护、SwiGLU 和 merge 检查通过，实际 GPU ISA 确认原生 dp4a；四行权重复用及精确修正倒数量化把 M32 整数完整路径从约 953 降至 555 µs，但原 NF4 为 522 µs（仍慢 6.2%），M1 仍慢 20.8%。近似 NRMSE 1.65%–1.80%，少量元素超出原逐点门槛，未做模型质量认证，保持测试入口。R2b 的 INT8 DPAS/真实隐藏态验收尚未实现，未宣称整阶段完成；可继续 DPAS 热桶或原计划 R3-head，9033/32 GiB/GPU-only 不变。

2026-10-01 继续完成 [R1b 有界 MoE scratch / Q36 window merge](qwen3.8-flash-next-r1b-compact-moe-results-2026-10-01.md)：原 NF4 专家算术不变，以顺序无关整数累加保证缓存/分组重排后的前缀一致性，相关算子、因果、checkpoint 和质量检查通过。相对 attention tile256 再节省 94.73 MiB GPU USM，完整请求基本持平、decode 未改善，候选默认关闭。R1b 的预算窗口与服务公平性尚未实现；下一步 R2b / R3-head，GPU-only、32 GiB、9033 基线不变。

2026-10-01 完成 [R1b attention/GDN tile 第一部分](qwen3.8-flash-next-r1b-attention-tile-results-2026-10-01.md)：9033 驱动上，大路由窗口内分 tile 执行 attention，回收 258.27 MiB GPU USM；状态、跨稀疏阈值因果和自然文本 48 步质量检查通过。暖 TTFT 小幅增加、完整请求未加速，候选默认关闭；下一步是有界 MoE down scratch / window merge，R1b 尚未全部完成。驱动升级的同旧二进制复测显示约 3% TTFT 变慢信号，证据不足以归因于驱动；decode 20 tok/s 仍未达到。

根据架构 brainstorm 完成 [R0 / R1a / R2a 首轮](qwen3.8-flash-next-architecture-r0-r2a-results-2026-09-30.md)：新增显式 qwen38-a770 CLI preset，41-token checkpoint 状态与 48 步质量 canary 通过（保留前 16 步 NLL 预警）；自然 1833-token 大窗口暖态 TTFT 22.607→16.412 s、prefill demand 字节减少 68.5%，但 decode 7.35→5.25 tok/s，未将 2048 设为默认。成熟 GEMM M32 完整路径仍比 ALU 慢约 19%，仅保留测试原型。下一项进入 R1b 的大路由窗口/小 tile 解耦，随后 R2b/R3-head；32 GiB、GPU-only 不变，实际 decode 20 tok/s 尚未达到。

已完成 [精确 QSA selector 与提交方式实验](qwen3.8-flash-next-qsa-selector-results-2026-09-30.md)：A770/Level Zero 的 K=512、ratio=4、完整块数≤32768 默认使用 8-bit radix＋vec4＋并行稳定输出。32K Top-K 1.956→0.118 ms，组合 QSA 后半段 2.198→0.381 ms；1428 组精确检查及相关 attention 回归通过。2146-token 真实 ABBA 的暖态 decode 仍为 6.15 tok/s，未宣称整机提速。Immediate 的连续提交路径出现回退，未全局开启；OOO 留作有显式事件和独立 scratch 的分支候选。下一项优先专家供给等待，保持 32 GiB、GPU-only。

新增 [QSA 与长上下文执行差距审计](qwen3.8-flash-next-qsa-gap-audit-2026-09-30.md)：固定 Q/K/V 实测确认 selected attention 约 0.27 ms/层封顶，但通用 Top-K 从 4K 的约 0.39 ms 增至 32K 的约 1.95 ms，吞掉稀疏收益。同显存容量的模型分段诊断也显示，慢请求主要新增等待在专家 acquire（约 8.8 → 77.3 ms/forward），Full Attention 约 4.0 → 8.3 ms。下一步优先精确 radix/并行 compaction、长 Prefill 后供给与容量协调，再做 GQA 复用和 MTP 重估。保持 32 GiB、GPU-only；本轮增加诊断，未更换生产 kernel。

本机后续性能实验采用 **32 GiB host pool、MTP 64 个 draft 槽**。新增 [32 GiB decode 结果](qwen3.8-flash-next-ram32-decode-results-2026-09-30.md)：三个上下文的 24/32 GiB 正反向对照全部完成。2146-token 输入后实际 decode 中位数 5.45 → 6.7 tok/s；计入一次 3.8 tok/s 回落后，合并解码速度估算改善约 3.9%，完整请求中位数缩短 11.0%。短输入没有进一步收益，32 GiB 为实验基线而非全域最优；最低可用 RAM 10.775 GiB。probe 默认预算已切换，长上下文波动和 20 tok/s 目标仍待解决。

新增 [16/24 GiB RAM 预算对照](qwen3.8-flash-next-ram-budget-results-2026-09-30.md)：固定 GPU 容量、AB/BA 复测，短样本实际 decode 中位数 13.15 → 14.8 tok/s（约 +12.5%），主机池暖态填充归零；长 Prefill 未获得稳定收益。24 GiB 正式进程峰值工作集约 29.23 GiB，最低可用 RAM 18.09 GiB。保留全局 16 GiB 默认，24 GiB 作为本机 decode 性能候选，未达到 20 tok/s。

继续推进 [S6 驻留/缺失专家拆批](qwen3.8-flash-next-strata-s6-resident-results-2026-09-30.md)：修复同进程 MTP/target trace 文件冲突，补逐 token 路由和覆盖校验；默认关闭的小窗口拆批候选在中文/代码四行验证中分别降低约 3.2%/2.7%，所有对照 logits 位级一致。计算/拷贝重叠增大，但 kernel 区间也变长，尚未达到 20 generated tok/s；未更改生产 MTP 默认值。

最新进度（2026-09-30）：此前累计改动已提交为 66e21e9。[S1/S2 结果](qwen3.8-flash-next-strata-s1-s2-results-2026-09-30.md) 完成 KV-only 补齐（draft+catch-up 降约 9%/20%）及四类文本 1–4 行窗口测量；20 tok/s 的成本余量仍不足，按条件暂缓 S3 的前缀提交工程。[S4/S5 结果](qwen3.8-flash-next-strata-s4-s5-results-2026-09-30.md) 完成默认关闭的 scratch arena 候选，测试形状回收 77–700 MiB GPU USM；2052 边界和后续八步 logits 位级一致。S5 大块 trace 不支持直接移植 stream-all，未新增该供给路径。动态借槽和生产 MTP 均未实施，decode 20 tok/s 仍未达到。

设计起点 [Strata 源码对照](qwen3.8-flash-next-strata-design-review-2026-09-30.md) 给出 GPU-only 候选：MTP 历史补齐仅计算 K/V、保存 GDN 输入后只提交接受前缀的递推状态、复用 Prefill 临时区并统一 chunk/cache 预算。Strata 的 CUDA/CPU 混合吞吐不能外推到 A770/NF4；具体实施与未通过的门槛以上述后续结果为准。

此前 [P4 MTP 探索](qwen3.8-flash-next-p4-mtp-results-2026-09-30.md) 已完成真实 GPU draft、512/64 专家槽取舍和有界 device checkpoint 恢复重放检查。短样本连续命中不足，重复长样本验证成本过高；乐观条件成本模型约 11–13 tok/s，尚不支持生产 MTP 接入。CPU 数值参考的 softmax 敏感性与未通过门槛已保留，接受前缀提交未实现，默认不变。

此前 [P4 HC 融合结果](qwen3.8-flash-next-p4-hc-results-2026-09-30.md)：新增默认关闭的 2–4 行 HC 融合，512 输入四行验证由相邻控制的 199.95 ms/块降至 189.19 ms，独立复测 189.64 ms。算子与默认单行回归通过；仍有整模型数值差异，未默认启用。后续 MTP 探索见上文；20 tok/s 生成尚未实现。

X2–X4 之后，面向 Prefill 100+ tok/s / decode 20 tok/s 的执行模型审计、冷暖缓存诊断和下一阶段优先级见 [执行模型审计](qwen3.8-flash-next-execution-model-audit-2026-09-29.md)。候选的实际实施状态以随后各阶段结果为准，decode 目标尚未达到。

顺序推进记录见 [P0–P2 首轮结果](qwen3.8-flash-next-p0-p2-results-2026-09-29.md)：已补 trace/时间线，拒绝首版保护区候选；现有 B6 在固定长输入上缩短约 7.5%，默认仍不变。

后续 [P2/P3 结果](qwen3.8-flash-next-p2-supply-p3-front-results-2026-09-29.md) 已在 512/1024 暖输入验证 100+ tok/s 并拒绝固定 front graph。[P3/P4 继续结果](qwen3.8-flash-next-p3-decode-p4-verify-results-2026-09-29.md) 补齐 128 步固定续写与 2/4 行验证，新增默认关闭的小批量 NF4 投影；四行验证约 184 ms（短输入）/200 ms（512 输入），尚未计入 MTP draft/回滚，decode 20 tok/s 仍未达到。

实施进度（2026-09-29 更新）：X0/X1 已完成；X2 已完成有界验收并默认启用初始文本 `30/26` 形状，通用锚点保留 opt-in。快照、KV、非零起点和 256-token 回退检查见 [X2 结果](qwen3.8-flash-next-x2-results-2026-09-29.md)。长文 RAM 问题已将 Windows 专家 reader 默认改为 ReadFile。X3 已完成 histogram、grouped XMX 首版及短/256-token/B6 验证；首版性能和长文数值未满足默认准入，所有桶保留 ALU 默认，详见 [X3 结果](qwen3.8-flash-next-x3-results-2026-09-29.md)。X4 已完成 T16 chunked GDN 数学原型、DG2 FP32/XMX 映射和状态验收；GPU FP32 候选算子快约 28%–31%、模型 prefill 快约 1.2%，因长文 logits 差异较大仍保留 opt-in，详见 [X4 结果](qwen3.8-flash-next-x4-results-2026-09-29.md)。以下为原始设计依据，旧默认值描述应结合最新结果阅读。

日期：2026-09-29。勘察基线：`6091a42976c0863b96200314e92b0ffd5034d47d`，开始勘察时工作区干净。

本轮按用户要求以代码和已有日志为主，没有运行构建、GPU benchmark 或全量测试。本文件给出下一阶段设计，不宣称候选方案已经获得端到端收益。用户本轮已明确允许合理容差内的数值差异；历史文档中的“48 个生成 token 必须完全一致”不再作为新计算路径的准入条件。

## 1. 决策

**GPU 路线仍有值得投入的空间。优先把现有 dense NF4 fused XMX 路径做成按实际形状选择的生产路径，再做专家内多 token 的 grouped XMX，最后推进 chunked GDN。** 不需要为了旧归约顺序而拒绝矩阵加速器，也不需要先完成一次全后端重写。

下一轮具体交付建议：

1. 校正现有 dense XMX 对照条件，覆盖真实 `M=26/30`，保留 `M=4` 尾段对照。
2. 为 fused XMX 接入安全的 direct I/O；为 `M<64` 提供合理的 tile/BK 选择。形成 Q4E 局部的静态 shape dispatch，而非全局强制 XMX。
3. 在相同请求上单独评估已有 interior checkpoint，减少 `26+4` 两次遍历；按容差验收计算结果，按状态语义验收快照。
4. 为 MoE 建立按专家实际 token 数选择的 ALU/XMX 双路径；先完成一个可集成的小范围原型。

这里的目标是减少请求等待时间。XMX 利用率只是解释手段；提高利用率却增加 H2D、显存临时权重或调度时间，仍然应回退。

## 2. 当前成果验收与证据边界

| 项目 | 当前代码/记录 | 本轮判断 |
|---|---|---|
| G1/G2 | HC 的 decode 专用 subgroup 路径、GDN single-write 已在代码中 | 保留；不再重复规划成“待实现收益” |
| G3 | expert-major 默认开启；同专家 4/2/1-token 权重复用 | 已减少重复加载，但实现仍是向量 FMA，不是 grouped XMX |
| G4/G5 | 预测退避、shared fusion、static pinning 默认关闭 | 尊重近期端到端负收益结论，不重新捆绑开启 |
| Dense NF4 prefill | `AILA_Q4E_FUSED_PREFILL=false`；默认整矩阵解量化到 BF16，再调用 oneDNN | 已有 fused XMX 能力尚未成为 Q4E 默认；是最直接的机会 |
| 静态 BF16 层 | `DenseLinear(count>1)` 进入 `Linear::forward` 的 oneDNN matmul | 不能说整个 prefill 都不用 XMX；实际 oneDNN 指令路径未在本轮验证 |
| Interior checkpoint | 显式 opt-in，anchor 白名单 `{26,28}`，另有长度/状态限制 | “G6 已完成”不等于普通请求默认已经合并 prefill |
| 当前速度 | 最新 handoff 记录 48-token story 为 5692.38 ms | 作为既有报告引用，本轮未复测；不是本轮测量值 |

代码入口：

- [Q4E quant linear 初始化](/E:/RiderProjects/Aila/src/models/Qwen4ExpBnb4Backend.cpp:348)：异步解量化默认开启，direct I/O 与 fused prefill 默认关闭。
- [DenseLinear 与 HC](/E:/RiderProjects/Aila/src/models/Qwen4ExpBnb4Backend.cpp:817)、[oneDNN Linear](/E:/RiderProjects/Aila/src/ops/Linear.cpp:73)：prefill 与 decode 的实现不同，不能把 decode HC 的优化结果直接推广给 prefill。
- [Bnb4BitLinear prefill 分发](/E:/RiderProjects/Aila/src/ops/Bnb4BitLinear.cpp:2202)：默认按需展开整矩阵；fused 路径直接读取 packed NF4。
- [MoE 多 token 分发](/E:/RiderProjects/Aila/src/ops/MoeExpertOps.cpp:2598)：quads/pairs/singles；[backend 元数据排布](/E:/RiderProjects/Aila/src/models/Qwen4ExpBnb4Backend.cpp:1691)。
- [GDN token 循环](/E:/RiderProjects/Aila/src/ops/GatedDeltaNetOps.cpp:512)、[checkpoint 条件](/E:/RiderProjects/Aila/src/models/Qwen4ExpBnb4Backend.cpp:2525)。

### 2.1 现有日志能说明什么

[最近的详细 profile](/E:/RiderProjects/Aila/tmp/q4e_prof_test_g5.log:46) 配套 [metadata](/E:/RiderProjects/Aila/tmp/q4e_prof_test_g5.meta.json) 是 **`7eaeda8`**，早于当前 `6091a42`：

| 阶段 | forward total | acquire 主机计时 | 分发信息 |
|---|---:|---:|---|
| 26-token | 2021.78 ms | 1560.93 ms | on_demand 累计 194→388，fused=0 |
| 4-token | 368.02 ms | 62.66 ms | on_demand 累计 388→582，fused=0 |

194 是每段增量，计数器中还含先前 warmup。它不是 expert-miss 数，而是 Bnb4BitLinear 的整矩阵解量化调用数。这个日志证明旧构建确实在走默认路径；当前源码仍保留相同默认值。不能把该日志的 6594.68 ms 当作当前 HEAD 的端到端结果。

`acquire`、`router_sync` 是主机等待区间，可能等待此前排队工作并影响重叠；不能用 `total-acquire-router_sync` 得出纯 GPU 算力时间，也不能直接据此计算可靠的 Amdahl 上限。过去强制同步的 stage profile 同样不应与今天的异步路径混算。

### 2.2 需要纠正的旧规划结论

- **“MoE XMX 一定不划算”证据不足。** 平均每专家 token 少，足以支持保留小 M 路径，不能排除长尾热专家、长 prompt 或融合解量化后的 XMX。更不能推广为所有 dense projection 都不该走 XMX。
- **旧 dense C3 的 1.11–1.90 倍是候选信号。** [测试函数](/E:/RiderProjects/Aila/tests/ops/Qwen4ExpKernelTests.cpp:3638) 仅覆盖 M=32/64，使用随机权重；未设置 `allow_async_dequant=true`，而该字段[默认 false](/E:/RiderProjects/Aila/src/ops/Bnb4BitLinear.hpp:124)。在没有环境覆盖时，对照组有额外同步，当前 Q4E 则没有。测试还只打印 max diff，并未以其断言精度，因此不能直接继承速度或质量结论。
- **减少显式依赖不等于三个 kernel 并行执行。** [Context](/E:/RiderProjects/Aila/src/core/Context.hpp:32) 的 compute queue 是 in-order；MoE 的 quads/pairs/singles 仍按队列顺序执行。已有跨 copy/compute 队列重叠有价值，但不要将其与同一 compute queue 内的并行混淆。
- **SLM 限制应写准确。** A770 是每 Xe-core 128 KiB、每 workgroup 最多 64 KiB。完整 FP32 `128×128` 状态已占 64 KiB，无法再加普通 WG scratch；这不排除更小的 GEMM tile 或状态分块。
- 旧规划把 G1–G6 的理论收益相加推到 3.3–3.6 秒，缺乏联合运行证据；G4/G5 已出现反例。下一阶段不再使用这种加法承诺。

## 3. 新的精度契约：参考同一个 NF4 模型，允许不同运算顺序

允许：BF16/F16 矩阵乘输入、FP32 累加、不同 reduction tree、合理的 epilogue 融合，以及由微小 logits 差异导致的 greedy token 分叉。FP16 范围更小，首选与现有激活一致的 BF16；切换 FP16 应单独计为候选。

仍需正确：NF4 nibble/LUT、nested scale、矩阵转置与 stride、mask、token/专家/rank 映射、给定分数上的 Top-K 选择、快照位置、取消和内存生命周期。对这些错误不以“模型已经量化”作为解释。

NF4 codebook 非线性，不能把 4-bit code 直接当作 XMX 的线性 INT4 数值。当前可落地路线是 **packed NF4 存储和传输 → GPU tile 内 LUT/scale 解量化到 BF16 → XMX FP32 累加**。这通常是在同一 NF4 权重上改变计算舍入；再次重量化成 INT8/INT4 则属于另一类误差，后文单独处理。

最小数值观察方式：

1. **算子层**：固定真实输入和同一份 NF4 权重，比较输出的 NRMSE、绝对误差分位数、max abs 和输出 RMS。`NRMSE=RMS(candidate-reference)/max(RMS(reference),epsilon)`。近零元素不单独用 relative error 判错。小尺寸索引/解量化用例可沿用已有独立 CPU 参考，避免两条 GPU 路径共享同一个布局错误。
2. **模型层**：固定 prompt 与后续 teacher-forced token，观察 logits 分布差异、KL 与参考文本的平均 NLL 变化。自由生成的 48-token 序列不同，不单独判失败。
3. **有状态路径**：GDN/checkpoint 改动增加一次 restore 后继续 teacher forcing 的定向检查。相同实现的 snapshot copy/restore 必须保持原状态；不同计算算法产生的状态则用数值容差比较。

可以把算子 NRMSE 1%、teacher-forced 平均 NLL 增量 0.01 nat/token 作为**待校准的初始预警线**，不是已验证的统一合格线。超过预警先定位层和分支；低于预警也不代表完成通用质量认证。保留 BF16 路径的自然误差分布与代表文本记录后再固定门槛，不要求本轮先搭建大型质量评测系统。

性能比较固定输入 token、生成长度、warmup 和缓存条件；优先用 teacher forcing 保持工作负载可比。完整模型路径允许重新计算 routing；在 MoE 算子 A/B 中则复用同一组路由，只比较算子。不能用强制复用旧路由的测试宣称完整模型质量已通过。

## 4. 第一优先级：把已有 dense fused XMX 做对实际形状

### 4.1 当前路径及机会

默认：`NF4 -> 整矩阵 BF16 scratch -> oneDNN matmul -> 输出`。

候选：`NF4 -> SLM/寄存器中的 BF16 小 tile -> XMX FP32 accumulate -> BF16 输出`。

二者的区别包括中间权重显存流量、kernel 数、解量化复用和 tile 调度；不能简单称为“oneDNN 不用加速器”。已有 fused kernel 的入口是 [joint_matrix 实现](/E:/RiderProjects/Aila/src/ops/Bnb4BitLinear.cpp:677)，无需从零写 NF4 matmul。

首次候选使用 Q4E 局部已有开关 `AILA_Q4E_FUSED_PREFILL=1`，并记录 `AILA_BNB4_FUSED_PREFILL` / `AILA_BNB4_GEMM_XMX` 的全局覆盖值，防止配置污染。这个开关也影响 PLE，对比时应记录实际分发。最终生产策略应细化到形状；本轮不直接翻转默认值。

### 4.2 当前 dispatch 的真实缺口

[XMX 分发](/E:/RiderProjects/Aila/src/ops/Bnb4BitLinear.cpp:1086) 仅在 **M≥64 且 N≥64** 时考虑 BK32/BK64 和大 tile；否则固定 `BM32/BN32/BK16`。因此调整现有 BK 环境变量对 story 的 M=26/30 **没有作用**。

建议候选集保持很小：

| 实际 M | 首批候选 | 原因 |
|---|---|---|
| 1 | 原 packed NF4 GEMV | 避免 decode 权重膨胀和矩阵准备开销 |
| 2–7 | 原路径；必要时 BM8 的 XMX 对照 | 当前 BM32 padding 太多；不要默认强推 |
| 8–32 | BM16/32，BN32/64，BK32 为第一候选 | 匹配 26/30，减少 K16 的 barrier/装载轮数 |
| 33–64 | BM32/64，BN64，BK32/64 | 由真实形状选择，不仅看 M 的阈值 |
| 更大 M | 现有 fused XMX 与展开一次后 oneDNN 比较 | fused 会在不同 M tile 重做 B 解量化，大 M 时未必赢 |

这是候选空间，不要求扫描全部组合。先选一两个合理配置；出现收益后才扩大覆盖。DG2 首选已有 SG8、BF16 `8×8×16` fragment；不要直接移植 PVC 的 SG16、load_2d 或更大 fragment 假设。更大的 BK 应核算 SLM、寄存器和每 WG 占用，不能只按 barrier 减少推断收益。

### 4.3 去掉 fused 分支仍存在的输入/输出拷贝

[当前 fused 分支](/E:/RiderProjects/Aila/src/ops/Bnb4BitLinear.cpp:2212) 每次仍把 input 拷到 scratch，再把 scratch output 拷回去。后面的 oneDNN 分支已实现 disjoint/LoRA 条件下 direct I/O，fused 分支没有复用这套保护。

将安全判定提取为共用逻辑，非别名、布局支持且无冲突的 LoRA 条件下直接传入 input/output 指针；其他情况保持 scratch fallback。这主要减少提交与激活搬运，不能拿它解释全部 prefill 延迟。每段约 194 次调用意味着可减少很多小 copy，但实际收益仍应看端到端。

初始化预留所需 scratch、形状 primitive 和候选 warmup；不要在 decode 或执行槽热路径新分配 USM。warmup 覆盖最终选中的 M/tile 路径，并保留 decode count=1；不把大范围 JIT 枚举变成新的启动负担。

### 4.4 数值自由度可以换取哪些实现空间

- NF4 decode 后在 BF16 舍入，再由 XMX 累加；不恢复旧 scalar FMA 顺序。
- 合并合理的 scale、bias、gate/up epilogue，允许与原来“中间 BF16 写回再读取”不同的舍入点。
- shared expert 的 gate/up XMX 后先单独 SiLU×up，收益明确后再融合 epilogue；不同时改动多处数值边界。
- 静态 BF16 HC 的 prefill 后续可试 oneDNN post-op 或专用小 M epilogue 融合；先确认该部分占比，不把已经使用 oneDNN 的层全改成手写 XMX。

**停止条件**：真实 M 下 fused 总路径输给异步 oneDNN，则该形状保留 oneDNN。无需为了统一实现让所有层一起退化。

## 5. 第二优先级：按专家实际激活量使用 grouped XMX

### 5.1 先改变判断标准

设某层某专家接收 `m_e` 个 token。G3 把它拆成四行、两行、单行组：一组内复用权重，但当 `m_e` 较大时，同一专家仍被分成多组读取和解量化。XMX 应以 **同一专家的多行矩阵** 为单位组织。

不同专家权重不同，不能把它们的 token 堆到一块后当成共享 B 的普通 dense GEMM。Grouped launch 中，每个 expert/tile 必须有自己的 weight slot 和 activation 索引。

无需单开重型 profiling：现有 `expert_counts_host_` 已经拥有数据，聚合 `1/2/3–4/5–8/9–16/>16` 的计数、activation 数和 cache hit/miss 数即可。计数在 CPU 路由排布时顺手累加，不新增 GPU 同步。

以旧日志为例，26×10×48 个 activation 对应 4678 次 layer-expert 请求，平均约 2.67 行/被激活专家。这说明低 M 很多；**平均值不能给出 ≥8 行的实际工作占比**。不能据此永久禁止 XMX，也不能假定全部专家都能吃满矩阵引擎。

### 5.2 两条实现路线

**路线 A：tile 内解量化的 grouped XMX，作为主要设计。**

1. 复用现有路由与 expert slot，预分配 `ExpertTile{slot, token/rank offset, valid_rows}` 描述符；初版在现有 host 排布处生成，不先重写 GPU router。
2. 按 `m_e` 分桶，一次或少量 grouped launch 处理所有满足条件的专家，避免每专家逐个创建 oneDNN primitive。
3. 将同专家的 token gather 到小 A tile；在 GPU 内由 packed NF4 和 nested scale 生成 BF16 B tile；XMX 用 FP32 累加。
4. gate/up 分别保留累加结果，计算 SiLU×up，输出 `[token, rank, intermediate]`。先沿用现有 BF16 中间缓冲。
5. down XMX 写回各自 `[token, rank, hidden]` 的 FP32 结果；复用现有 merge。允许改变求和顺序，但初版没有必要引入浮点 atomic 竞争。
6. 每个桶的最后消费者 event 保护 slot；接入已有 B6 ready/fill/DMA 与 consumer fence，不让新 kernel 提前释放或复用专家权重。

启动策略：低 `m_e` 保留 4/2/1-token ALU；较高 `m_e` 尝试 M8/16/32 XMX。**8 是首个探索点，不是预先证明的 crossover**；A770 上 M4 XMX 若实际更快也可以采用。

**路线 B：小容量 GPU BF16 staging + 成熟 GEMM，作为有限对照。**

GPU 收到 NF4 后，只将当前 2–4 个专家展开进预算内的 staging arena，调用预建形状的 GEMM，再循环复用 arena。它能帮助判断瓶颈在自写 tile 解量化、矩阵引擎还是调度，开发成本可能更低。必须把展开、pack/scatter 与提交时间都算进去；短 M 大概率需要路线 A 或保留 ALU，不能只报 matmul 核心时间。

按当前 H=2560、I=640：

- 每专家 gate/up/down 共 `3HI=4,915,200` 个权重。
- packed NF4 + 当前 u8 scales 约 `2,534,400` B，约 2.42 MiB。
- 全 BF16 约 9.375 MiB/专家；2–4 个 staging 槽约 18.75–37.5 MiB，另计激活、描述符和对齐。

全专家缓存展开会把权重容量放大约 3.88 倍；不采用这种默认方案。任何 staging 都要进入现有 DeviceRuntime 显存账本，明确挤占多少 NF4 cache slots。H2D 继续传 NF4，不能靠提前在 host 展开来“喂饱 XMX”。

### 5.3 小成本模型，避免反复走错方向

每专家主要矩阵工作量约 `2*m_e*3HI` FLOPs。相对该专家 NF4+scale payload，算术强度约 `3.88*m_e` FLOP/B（未计激活和中间结果）。这个简单推导说明低 `m_e` 和 cold miss 很容易受传输限制；换上矩阵引擎不会消除首次权重到达时间。

实际选择比较以下**完整路径**：

`T_ALU(m_e)` 与 `T_gather + T_tile_dequant/expand + T_XMX(padded_m_e) + T_scatter/merge_extra`。

再把它放入 copy/compute 重叠后的完成时间，不能将单算子收益直接加到整请求上。G3 的“多组逻辑读取”也不等于全部都是 DRAM miss，L2 复用可能减小 XMX 的实际优势。

不要为了凑大矩阵等待尚未到达的专家。已 ready 的专家可先算；同一专家内的 token 已知时可以成组。全层等齐再 launch 会重演数据流优化中的等待问题。

## 6. 让 prefill 形成足够大的矩阵

### 6.1 先利用已有 interior checkpoint

story 的 `26+4` 意味着许多与 token 数无关的层遍历、解量化和提交成本执行两次。现在 26/28 白名单需要显式开启，新的容差契约允许重新评估将适用范围推广为状态语义条件，而非“恰好 0-diff 的长度”。

先只在已有支持条件内评价 30-token 单次 forward，并与 dense XMX 分开归因。26→30 本身没有改变当前 BM32 tile，却能省掉另一次 M4 路径；之后再讨论更多锚点与更长 chunk。

推广必须确认 prefix snapshot 只包含锚点及之前的 GDN/conv、KV、PLE 等状态；不能把未来 token 状态混入快照。较早 anchors 的已有数值偏差要区分算法舍入和状态捕获错误，不能仅靠放宽 logits tolerance 全部放行。

旧日志首段 `logits=0`、尾段 `logits=1`，所以收益主要来自重复层工作；不能继续声称一定省掉一次原本已被跳过的 LM head。

### 6.2 长 prompt 用算子 chunk，而非反复跑完整小段模型

对较长输入，尽量让一层能看到 128/256 行或预算允许的更大 token 块，使 dense projection 与热专家拥有复用空间。GDN 或 attention 内部可以用自己的小 tile；不要因此强迫整个后端每 2/4-token 重新遍历 48 层。

边界由既有调度和显存预算决定；不越过 causal 可见性、QSA 阈值或请求公平性约束。单请求 story 没有足够 token，不能靠 padding 到 256 假造复用。Continuous batching 有利于服务吞吐，但不应作为本轮单请求 TTFT 的前置工程。

## 7. 第三优先级：GDN 从 token recurrence 走向 chunked matrix 算法

当前 single-write 只是减少一次状态写回。[prefill kernel](/E:/RiderProjects/Aila/src/ops/GatedDeltaNetOps.cpp:433) 仍在每个 head 的 WG 内顺序循环 token，逐步更新 FP32 状态；不是矩阵化的 chunk 算法。

本地 vLLM 已带可参考的 FLA 实现：

- [chunk.py](/E:/RiderProjects/Aila/tmp/vllm/vllm/third_party/flash_linear_attention/ops/chunk.py:23)：gate cumsum → scaled KKT → triangular solve → WY → chunk state/output。
- [wy_fast.py](/E:/RiderProjects/Aila/tmp/vllm/vllm/third_party/flash_linear_attention/ops/wy_fast.py:93)：矩阵乘形成 W/U。
- [chunk_delta_h.py](/E:/RiderProjects/Aila/tmp/vllm/vllm/third_party/flash_linear_attention/ops/chunk_delta_h.py)、[chunk_o.py](/E:/RiderProjects/Aila/tmp/vllm/vllm/third_party/flash_linear_attention/ops/chunk_o.py)：跨 chunk 状态和输出。

设计方向：先将 conv、归一化、gate/beta 准备成 batched 数据，再以 T=16/32/64 的 chunk 形成数学等价的 WY/三角形式，将其中矩阵乘交给 XMX。保留 FP32 的 gate scan、关键求解和最终 recurrent state；BF16 只用于选择的矩阵乘输入，不同时把全部状态降精度。

需要显式处理 grouped Q/K head、已有 conv history、非零初始 state、最后不满 chunk 和中间 checkpoint。vLLM 的 Triton 代码可提供数学与布局参考，不能原样当成 DG2 的最优实现。

这条路线允许归约顺序变化，恰好是取消 bitwise 准入后值得重新打开的空间。但短 story 的 launch/准备成本可能压过收益，首个目标应是中长 prefill，decode 继续现有 recurrence。若希望先做更小原型，可探索每线程持有部分 FP32 状态跨多个 token 复用；必须核查 register spill，不能把整 head 状态塞满 SLM 后宣布成功。

## 8. 实验性备选及暂不优先的工作

| 候选 | 价值 | 本轮排序 |
|---|---|---|
| NF4→INT8 权重 + activation quant + INT8 DPAS | 可能减少矩阵输入/暂存量并提高计算吞吐 | 后续独立实验；增加了量化误差和动态 activation scale 成本，需完整路径赢过 BF16 XMX |
| 误差补偿的低精度 GDN 状态或 FP32 拆分矩阵乘 | 可能适配更大状态/更长上下文 | 更后；对 recurrent 误差积累更敏感，不与首轮 dense 优化捆绑 |
| AOT / 精简 JIT manifest | 改善启动与首触发时延 | 与 warm prefill 性能分开计；不把编译时间收益算作算力收益 |
| CUDA/PVC 风格 persistent mega-kernel | 减少 launch，有理论空间 | 先不做；当前 host 路由、slot 生命周期与 DG2 资源限制使改造范围过大 |
| 全局 BF16 专家缓存 / 静态 pin 更多专家 | 简化 GEMM 权重访问 | 不作为主线；容量和 miss 已有负收益先例 |
| 仅继续优化 1-token ALU GEMV | 改善 decode | 保留已有成果，本轮投入优先向 prefill 矩阵路径移动 |

不以 A770 的峰值 TFLOPS 推算某个 kernel 能快多少倍。Intel 官方说明 Xe-HPG 具有独立 vector/matrix engines 并支持 BF16/FP16 矩阵运算，但解量化、tile 占用与数据到达仍决定实现收益：[Intel Xe GPU Architecture](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/intel-xe-gpu-architecture.html)。本輪通过 tavily-search 查询此官方说明；具体 DG2 tile/SLM 限制同时参考本机 [A770 skill 硬件记录](/C:/Users/Administrator/.codex/skills/A770-Kernel-Optimization/references/hardware/hardware.md)。

## 9. 有界的实施与验证顺序

| 阶段 | 交付物 | 最少观察 | 继续/停止规则 |
|---|---|---|---|
| X0 | 校正 C3 的异步对照和容差报告，记录 shape/路径 | 真实 M26/30，加一个 M4 边界与一个较长 M；每形状少量 warm 后样本 | 不再拿旧同步 baseline 或随机 max diff 直接定默认 |
| X1 | dense fused direct I/O + 小 M tile/BK 分发 | 同形状完整算子时间；story 的 TTFT/prefill、固定 48-token 时间 | 仅对有收益的形状启用；弱收益才追加少量交错样本 |
| X2 | checkpoint 单次 prefill 接入评估 | 同一 prompt 的 26+4 与 30；一次 restore/continue | 状态语义正确、数值合格且端到端获益才推广 |
| X3 | expert histogram + grouped XMX 首版 | 现有小 M ALU 对照；真实专家/输入；ready/fence 一例 | 按桶保留赢家；story 热桶占比低时转向中长 prompt |
| X4 | chunked GDN 数学原型与 DG2 映射 | 一组非零初始状态的 chunk + 跨 chunk continuation | 局部和整体都获益才进入默认；不抢占 X1/X2 |

验证以目标算子和改动边界为限：固定短 story、一个约 256-token 的 prefill 样本，涉及状态时加一次 continuation；这些是 canary，不宣称代表完整模型质量。已有 warmup 后做少量交错 A/B，清楚区分冷/热 host expert cache。没有回归迹象不追加全量 CTest、多轮长上下文或 VTune 大扫荡。

记录最小身份：commit、exe hash、环境覆盖、prompt/token 数、冷/热状态、dispatch、prefill/TTFT、48-token 总时间、显存/slot 变化、数值指标。需要判断是否实际执行 DPAS 时，只查目标 kernel 的实现或一次代码生成记录；无须每轮反复检查所有 kernel。

**首个 worker 工作包应止于 X0/X1：修正对照、实现 fused direct I/O、给 M26/30 一条合适的 XMX 分发，并产出一次有身份信息的对照结果。** 不要求它同一轮同时实现 grouped MoE、GDN chunk、全 checkpoint 推广或 AOT。这样既开始利用容差换取加速空间，也能在第一轮明确判断投入是否产生请求级收益。
