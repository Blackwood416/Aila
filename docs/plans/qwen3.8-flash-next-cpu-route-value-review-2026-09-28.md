# Qwen3.8-Flash-Next：CPU 异构路线是否继续投入

日期：2026-09-28。勘察基线为 `f78732da1775f169b616f5bb8adbc0e191e16ba4` 加当前未提交工作树；不能把实验结果仅绑定到该 commit。当前 `Aila.exe` SHA256 为 `5CA0BAE474C8249C59A94EA84A9FD3C2AC1139F47BB86FFF0F14EBC352A8AB0A`，与最新 dataflow 配对一致。Windows / oneAPI 2026.1 / A770 16GB / Xeon E5-2696 v3 AVX2，目标为单请求 story 的延迟。

本轮只读代码、既有实验报告、原始日志与 sidecar，新增本判断和证据摘录。没有构建、运行模型、重跑测试、修改实现、停止 worker 或改变默认开关。

## 1. 决策

**当前机器和当前目标下，CPU routed-expert 计算应退出优先优化主线，保持关闭，停止继续扩大线程、任务数量和强制 CPU 放置。保留实现及实验资产；只在触发明确新条件时重启。**

这是投入优先级判断，不是证明所有 CPU/GPU 混合算法都不可能获益。现有证据已经足够停止无条件追进度；尚未覆盖的组合不构成必须再做一轮大矩阵的理由。

我调整 9 月 27 日“以 CPU 专家异构为下一阶段主线”的建议：当时提出的字节优势真实存在，但现在端到端证据表明，当前实现/硬件下的算术、数据准备、跨设备衔接及失去 GPU 复用的代价尚未被覆盖。继续以“又比上一版 CPU 快了多少”作为推进依据，会偏离 GPU-only 这个实际替代方案。

| 路线 | 当前决定 |
|---|---|
| 强制 CPU 接管 eligible routed experts | 暂停优化，不再扩大配额 |
| host-ready miss 选择性分流 | 保留 opt-in；无默认推广依据，不自动开启下一轮工程 |
| CPU 负责读取、填充、预取、PLE 与调度 | 保留；这是另一类协作，不因 CPU 点积失利一并回退 |
| Ready/lease/ticket、故障清理、测量与 CPU reference | 保留可独立复用的成果 |
| GPU ready-first / ReadFile / 批量 I/O | 分别用自己的结果决定，不以“属于异构计划”绑定升降级 |

## 2. 已核对的性能证据

本轮抽取 16 份原始日志，逐个核对 `.tokens.json` 的完整性、token 结果和其中候选日志 SHA256，全部 hash 对应。性能比较来自原实验，未重新推理。详细设置、binary hash 和结果在 [证据摘录](qwen3.8-flash-next-cpu-route-value-evidence-2026-09-28.json)。

| 配置与比较范围 | GPU-only | CPU 配置 | 结论及限制 |
|---|---:|---:|---|
| 较早的一配额、4 线程、GPU-order + overlap；同 binary 三组配对 | 中位 5682.80 ms | 中位 5865.32 ms | **+3.21%**；三对均回退、全部 token 精确相同。使用较早 CPU 算术，不代表最新 prepared kernel 的低配额表现 |
| 12 线程 prepared batch，miss-only 10；同 binary 单次筛选 | fill=8：5689.62 ms；fill=4：5898.85 ms | fill=4：6719.18 ms | 即使对相同 fill=4 仍 +13.91%；从 token index 26 开始不同，只能作可行性/负面筛选，不能作同输出性能结论 |
| 最新 data_wait，强制 eligible routed experts 放 CPU；三次候选、同 binary GPU 参考一次 | 5695.54 ms | 中位 8733.11 ms，范围 8547.99–8748.65 | **约 +53.33%**；候选三次 token 精确相同并有相应 stateful 验收。GPU 参考不是三次配对基线，但约 3 秒差距足以支持当前不推广 |

最新 CPU 路径要追平这份 GPU 参考，整个请求还需减少约 **34.78%** 时间。这不是再省少数 memcpy 提交即可自然达到的距离。

CPU 路径内部改善确实成立，但要使用正确分母：

- prepared input / AVX2 查表改写使固定 12 线程、40-task 算子约 20.45→4.13 ms；这是 CPU 算子改善。
- 大 CPU 放置的端到端旧/新中位数 19327.56→9192.97 ms，约 2.10×；两者仍是 CPU 实验配置。
- 最新批量准备、合并回传和等待预取组合 9258.08→8733.11 ms，约降低 5.67%；六次新旧黄金轨迹通过。

这些成果值得保留；它们均没有证明比当前 GPU-only 更快。早期交接文档以单专家约 0.2786 ms 小于估算 H2D 0.39 ms 来推断高收益潜力，现已不足以指导路线选择。

来源：[alignment](qwen3.8-flash-next-cpu-alignment-2026-09-28.md)、[overlap](qwen3.8-flash-next-cpu-overlap-2026-09-28.md)、[scale12](qwen3.8-flash-next-cpu-scale12-2026-09-28.md)、[dataflow](qwen3.8-flash-next-cpu-dataflow-2026-09-28.md)。同名 HEAD 下有多个 binary，不能跨实验直接合并为受控 A/B。

## 3. 当前失败机制，代码和计数能说明到哪一步

### 3.1 它尚未覆盖最初要解决的首段 26-token prefill

`Qwen4ExpBnb4Backend.cpp:1533–1536, 1935` 限定 CPU small-M/异构入口为 `count<=4`。story 的 26-token 首段仍用 GPU prefill；启用 small-M 后，4-token 后缀可以一次有 40 个 expert/token tasks，decode 每层最多 10 个。

因此，报告中的“40-task / all-CPU”是 eligible 小 M 的 routed expert 分支，不是整个模型在 CPU，也不是 26-token 首段已经实现 CPU/GPU 共同计算。它仍让 attention、GR、GDN 等在 GPU。用这个结果证明“所有冷 prefill 异构均无效”超出证据；但为首段另写高 M CPU 算法已经是新的项目。

### 3.2 重叠已经实现，不能继续把“没有并行”当成主要借口

`Qwen4ExpExpertExecutor.cpp:469` 左右已把 CPU job 提交给常驻 worker，然后 coordinator 处理 GPU 分支/acquire。低配额配对中有约 90–104 ms 的 CPU/acquire host 区间重叠，但仍保留约 141 ms 的 join 累计等待；这些旧计数含 warmup，仅用于说明并行确实发生。

最新全 eligible CPU 的 data_wait pair 1，在减掉 warmup 后记录：24960 个 CPU tasks、3134.02 ms CPU compute、2788.55 ms host join、1022.56 ms host preparation、107.46 ms result-copy submission。CPU/acquire 区间重叠为零，因为该配置相应小 M 层没有剩余 routed GPU acquisition 可以覆盖它。

join 是 host 等待，不能直接认作 GPU 完全空闲；compute、join、preparation 的区间也不能相加为总开销或分别当成可回收收益。但它们已经说明 CPU 算术/准备仍大量出现在必须完成的依赖链上。CPU core 利用率提高、worker 存在或拷贝字节减少，都不足以证明更短的层完成时间。

### 3.3 “一专家 CPU 算得比传得快”不是当前选择条件

GPU 已 Ready 的专家无需 H2D；GPU miss 的 copy 可能与其他工作重叠。CPU 则增加 activation D2H、工作唤醒、NF4 算术、FP32 result H2D 和单独 ordered merge。减少某个 transfer 的服务时间，不一定减少整层暴露等待。

此外，`plan_layer()` 将 CPU entry 标记 `gpu_slot=-1`，GPU acquisition 只处理 GPU subset；CPU 消费并不自动把该专家填入 GPU cache。因此反复选中的专家可能反复落在 CPU 路径。这个 cache-policy 反馈在代码上成立，**对当前总延迟的贡献还没有匹配路由测量**，不能把它当成已确定根因或保证收益的新优化。

CPU 核心还与填充池/PLE/提交线程共享主机资源。12-thread 试验把 fill 从 8 改为 4；其同 binary GPU-only 从 5689.62 变为 5898.85 ms，说明资源拆分有实际代价。该差异不能证明究竟是 DRAM 争用还是并行度不足，也解释不了全部 CPU 回退；相同 fill=4 的比较仍不支持选择性 CPU 已经获益。

### 3.4 动态选择仍是启发式，且数值契约尚不完整

当前 planner 按 token/rank 顺序、host Ready、GPU residency 和 credits 选择；straggler 用连续 join 阻塞触发 penalty/cooldown。它没有逐任务、可校准的完成时间模型，也没有用未来复用收益决定是否迁入 GPU。固定 credits 模式还主动绕过了 admission throttling。

这留下进一步设计空间，但也意味着所谓“再调 scheduler”不是一个小开关，而是要同时承担路由选择、cache 反馈和数值验证的维护成本。

CPU nested scale、FMA lane order、BF16 h 边界已经修过；16 个真实样本匹配及 prepared-versus-old CPU bitwise 相同，都不能证明任意动态 ownership 精确等价。最新 miss-only 10 的 token index 26 仍为 expected 23221 / actual 3808。全放置通过、动态混合失败，并不能把后者的剩余原因自动归为 native exp、merge 或某个 queue bug；当前材料尚未完整定位。

这里采用项目现有黄金 token 契约；不同 token 不等于文本质量一定下降。若以后允许近似数值，应单独定义质量/误差目标，不能通过撤掉验收门槛让本次路线“变成功”。而该混合筛选本身也比 GPU 更慢，放宽精度不会自动消除其性能问题。

## 4. 什么还值得保留，什么应停止

**保留：** CPU NF4 reference/AVX2 prepared kernel、真实权重诊断、小型 batch harness；host reader lease、generation、drain/失败清理；独立可用的专家 reader、fill batching、DMA timing、warmup 覆盖。把它们作为可复用组件和未来约束变化时的起点，不必删除实验代码或回滚已有正确性修复。

**停止：** 继续增加超过 12 线程/40 tasks 的同类扫描；继续拿全 CPU eligible 配置作为默认加速候选；为 0.2786 vs 0.39 ms 这个早期比较不断补微优化；重复证明 CPU 新版本比旧版本快，却不与同 build 的 GPU-only 比。

**主线回到 GPU/数据供给的已定位成本：** 例如已有 MoE rows 的实际权重复用和 prefill 固定开销。既有 progress 文档中 rows-only 的 story 中位数约 5518 ms，是独立场景的候选证据，不与本轮 5695 ms 冒充同次配对。CPU 缺乏优势不意味着所有 GPU 方案已经验证有效；每项仍独立判定。

原 P3 GDN 状态驻留、checkpoint/pre-fill 边界等可保留候选，但不建议因为结束 CPU 扩容，就同时启动多个新方向。

## 5. 未穷尽的机会与重新开启条件

### 当前机器，仅剩窄的选择性机会

较早低配额负面配对使用旧 CPU kernel；现有报告**没有完整证明“最新 prepared kernel + 低配额 + 相同资源预算”的每一种策略都更慢**。因此不能宣布 CPU 路线理论死亡。

但当前也没有证据值得直接投入复杂预测器。若确需保留最后一个调查窗口，要求依次满足：

1. 能定位并关闭动态混合的数值差异，或得到明确批准的另一套精度契约；不把整模型轨迹失败当普通噪声。
2. 从真实 trace 中找到持续存在、CPU 能赶上的 GPU-miss 暴露等待窗口，并计入 activation、join/merge 和未来驻留代价。仅有 aggregate acquire 或纯 copy 带宽不够。
3. 只做一个预先选定的低配额、最新 kernel 配置和 GPU-only 的少量同 binary 配对，记录实际 CPU tasks。收益须大于当前噪声且不牺牲既定输出/资源要求；没有清楚收益即结束，不自动扩成新矩阵。

这不是本轮要求继续跑的测试单，而是重启门槛；现状下我建议暂不支付这轮定位与调度开发成本。

### 改变约束后，重新评估才有意义

- **GPU cache 明显缩小或模型更大，确有更多无法隐藏的 PCIe miss。** CPU-ready 权重才可能更有相对价值。不能仅把 max-seq 拉大就假定成立，应看实际可用 slots 和暴露等待。
- **CPU 算力/内存子系统发生实质变化。** 在新硬件上重测 NF4 路径；不把 E5 的结论外推，也不因新 CPU 有某种 ISA 就保证成功。
- **目标改为容量保障或并发吞吐，允许不同的单请求延迟。** 可以重新评估整请求/更粗粒度分工；当前 story 延迟数据不回答那个目标。
- **长 prefill 按同专家多 token 真正复用权重解码。** 当前 flat batch 是多个 expert/token tasks，未实现跨 token 共享一个专家的算术 tile；这是不同计算设计，仍需与 GPU 对应批量路径比较，不是顺手把 count 上限从 4 改成 26。

以上均是条件性假设，没有现成加速承诺。本机冷文件页问题不会因 CPU 点积接管自动消失：CPU 同样需要拿到那些权重。

## 6. 对现有文档的解释优先级

本判断基于最新 CPU alignment → overlap → scale12 → dataflow 证据，优先于早期 handoff 中“CPU 原型证明极佳收益潜力”或“阶段完成即优化成功”的措辞。已有未提交文档和实现由 worker 持有，本轮没有覆盖它们。

**建议将 CPU routed-expert 计算标记为“实现可用、局部优化成立、目标场景收益未成立、默认关闭、暂停扩展”。** 这是当前最有证据支持的研发结论。
