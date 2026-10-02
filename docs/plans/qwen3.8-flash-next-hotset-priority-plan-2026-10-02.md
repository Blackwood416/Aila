# Qwen3.8-Flash-Next：优先实现驻留专家直接执行与 hot expert set

最新实施（2026-10-02）：已完成 [C1/H3 首轮实现与对照](qwen3.8-flash-next-hotset-c1-h3-results-2026-10-02.md)。同冻结集合下，CPU cold 的短/长 decode 中位数为7.57→7.86、7.45→7.63 tok/s；但原动态路径为13.90/8.75，固定集合仍损害 decode 命中。H3 减少 acquire，却没有稳定净收益。代表性短/长48步质量、checkpoint/取消恢复、6项CTest、7项Python测试和普通trace回放通过；GPU profiling 因系统RAM突增触发保护，设备时间未取得。所有候选继续默认关闭，32 GiB预算不变。**下一步 H4/C2：热集来源、跨域覆盖和有界 epoch/admission；R2b继续暂停。** 下文保留各阶段历史记录。

实施更新（2026-10-02）：已保存此前工作为5ea0171，并完成 [H1/H2动态驻留直接执行首轮](qwen3.8-flash-next-hotset-h1-h2-results-2026-10-02.md)。Ready租约与cold-only acquire分离、GPU device route/descriptor查表及提前热提交已实现；短/长输入48步logits、checkpoint回放逐位一致，6项CTest及真实trace回放通过。但当前短decode为原路径14.0、H1 12.6、H2 13.1 tok/s，长decode为8.95、8.4、8.75 tok/s，尚无稳定端到端净收益，默认仍为0。此版本没有固定热名单或CPU冷计算。**下一步继续C1的同热集、冻结admission对照，R2b继续暂停。** 以下保留原始规划与勘察记录。

日期：2026-10-02。基线为 `66e21e9` 加当前未提交工作；R0、R1b、R2b 的最新结果以各自报告为准。本轮只做源码勘察、既有 trace 计数和热表静态覆盖分析，没有构建或运行 GPU 测试，没有修改运行时代码。

**优先级修订：先做 hot/resident expert 的直接执行路径，并重新评估 CPU 只算冷专家的异构分工，再继续 R2b 的生产集成与数值细化。** 保留已有 R2b 原型和质量结果。这里调整的是计划，不代表已停止或更改其他 worker 的运行。

用户进一步指出，热集架构也改变了对 CPU 异构路线的判断。这里撤销“CPU routed expert 继续退出优先主线”的安排：此前结果证明旧实现未获益，不能否定新的驻留热集与冷计算分工；同时，架构缺口尚未被新实验确定为全部回退的原因。先建立 GPU 直接热路径作为共同基础，再做同架构下的 cold-on-CPU / cold-on-GPU 对照。

上一轮把 Strata 的热集主要归入缓存替换策略，并用 Aila 旧 static pinning 的负结果压低优先级，遗漏了它把“驻留专家的执行”从“主机侧专家供给”中分离的设计。旧 G5 不能覆盖这个问题。本计划取代 [9 月 30 日总计划](/E:/RiderProjects/Aila/docs/plans/qwen3.8-flash-next-architecture-brainstorm-2026-09-30.md) 中对此方向的低优先级安排。

证据见 [本轮 JSON](/E:/RiderProjects/Aila/docs/plans/qwen3.8-flash-next-hotset-evidence-2026-10-02.json)，包含源码/热表/trace 哈希和计数口径；[分析脚本](/E:/RiderProjects/Aila/tmp/q4e_hotset_audit_20261002.py) 可在 CPU 上重算。下文 `[CODE]` 表示源码事实，`[REPORTED]` 表示既有报告，`[DERIVED]` 表示本轮离线计算，`[HYPOTHESIS]` 表示待实现和测量的设计。

## 1. Strata 真正做了什么

本地 Strata 版本为 `a79080535d1b2a71a3419a0d97d8e7dca194b0f1`。需要沿调用链理解，部分头文件仍留有早期“hit kernel 尚未接入”的注释，不能将那些注释当作当前状态。

| 机制 | 当前源码依据 | 对 Aila 的意义 |
|---|---|---|
| 默认启动方案带热表 | [setup.py:1969](/E:/RiderProjects/Aila/tmp/Strata/setup.py:1969) 生成 `--expert-profile ... --expert-cache auto` | 用户正常启动就能使用；裸 `generate` 参数默认值不代表 setup 生成的配置 |
| 按 profile 排名预装 | [generate.cpp:2150](/E:/RiderProjects/Aila/tmp/Strata/src/program/generate.cpp:2150) 在推理前按排名填满可用 tier | 热表决定谁驻留，不只是供预取参考 |
| GPU residency table | [generate.cpp:2886](/E:/RiderProjects/Aila/tmp/Strata/src/program/generate.cpp:2886) 构造 `d_res[layer,expert]` | 地址和槽位映射可在一个执行周期内保持稳定 |
| GPU 选择命中项 | [s2_expert_grouped.cu:373](/E:/RiderProjects/Aila/tmp/Strata/src/kernels/cuda/s2_expert_grouped.cu:373) 从真实 router IDs 查表，生成 slot、原 rank 和 count | 命中的专家不需要由 CPU 重新决定 dispatch 参数 |
| 命中计算嵌入 token graph | [session.cpp:833](/E:/RiderProjects/Aila/tmp/Strata/src/core/session.cpp:833) 接 select、quantize、expert kernel，再等待冷路径完成 | 热路径可以先运行，冷路径返回后再合并 |
| CPU 只计算缺失贡献 | [expert_source.cpp:733](/E:/RiderProjects/Aila/tmp/Strata/src/core/expert_source.cpp:733) 跳过驻留专家的 CPU 计算，清零相应 miss 输出行 | 热/冷输出所有权明确；复用的输出不能残留上次数据 |
| 执行周期之间调整热集 | [generate.cpp:4715](/E:/RiderProjects/Aila/tmp/Strata/src/program/generate.cpp:4715) 用衰减 usage、收益阈值和换入上限调整；[默认参数](/E:/RiderProjects/Aila/tmp/Strata/src/program/generate.cpp:260) 为每 4 rounds、最多 96 次 swap | 是执行期间稳定、边界处适应的热集，不是永远不变的专家名单 |

这套机制仍然计算 router 实际选择的全部专家。高概率只决定**权重在哪里、走哪条执行路径**，不替代 router，也不允许省略冷专家。

还有两个移植边界：

- Strata 单 token 路径可让 CPU 算冷专家；其 verify 路径还有 CPU/PCIe 分工。Aila 的 GPU-only 对照必须保留冷专家的 GPU 供给空间；新异构候选允许 CPU 接管合适冷项，但仍保留 GPU 冷供给退路，不能用热集占满显存后假设 CPU 总能赶上。
- Strata 的整 token graph 包含 CUDA doorbell wait。A770 首版采用显式事件与主机服务冷请求，不复制设备自旋等待，也不把已有 limited graph 支持当成完整 CUDA 执行机制等价。

## 2. Aila 的实际缺口

### 2.1 已有命中处理，尚未形成独立热路径

[CODE] 当前普通 decode 的调用链为：

```text
router 输出 pinned host IDs/weights
  -> CPU 等 actual_route_event
  -> acquire(layer, 全部 top-k IDs)
  -> CPU 组装 MoeExpertSlotDev[top-k]
  -> gate/up + down
  -> 记录 slot reader fence
```

对应 [router 与等待](/E:/RiderProjects/Aila/src/models/Qwen4ExpBnb4Backend.cpp:1285)、[全 top-k acquire 与组装](/E:/RiderProjects/Aila/src/models/Qwen4ExpBnb4Backend.cpp:2001)。

[CODE] `acquire()` 已经区分 hit/miss，命中的权重不会被重新加载。其 [全命中快路](/E:/RiderProjects/Aila/src/models/Qwen4ExpExpertCache.cpp:1382) 会维护预留、检查 Copying/Ready、更新非 pinned LRU；全部命中且无未完成 DMA 时直接返回空 event。**“acquire 了全部专家”不能写成“重复搬了全部专家”。**

但混合命中时，普通 grouped decode 的一组专家仍统一依赖该批 `dma_event`。已就绪的贡献被冷项供给拦住；CPU 也仍参与每层路由消费和描述符组装。

### 2.2 G5 与 ready-first 都不等于 Strata 的设计

- [G5](/E:/RiderProjects/Aila/src/models/Qwen4ExpExpertCache.cpp:2173) 已有预装和 pinned 槽。默认表是每层 18 个、共 864 个；custom 文件的数量边界另按剩余动态容量限制。它改变驻留内容，但没有 GPU 端直接消费 hot table 的路径，正常调用仍进入 acquire。原型回退不能证明驻留执行架构无价值。
- `READY_FIRST` 已能先提交 GPU-ready 子集，且能保留正确的 reader fence。但 [ensure_gpu_pending](/E:/RiderProjects/Aila/src/models/Qwen4ExpExpertExecutor.cpp:333) 收集的是所有 GPU entries，最后仍一起 acquire；源码明确要保护已就绪槽不被同批缺失淘汰。**不能简单删掉 ready entries 而不补 lease/reservation。**
- [现有 device descriptor kernel](/E:/RiderProjects/Aila/src/ops/MoeExpertOps.cpp:517) 可作为基础，但目前工作量和描述符依然由主机提供。还需要按 device route/hit count 选择，以及正确保留原 token/rank 的能力。

因此，目标不应只是重新开启 `STATIC_PINNING` 或 `READY_FIRST`，而应建立明确的 `resident execution / cold supply` 两条路径。

## 3. 现有数据支持什么，不支持什么

### 3.1 大量 acquire 的输入已驻留，但函数本身不是全部性能差距

[DERIVED] 对既有 24 GiB host pool trace 的第四个 41-token CLI 请求，取 forward 155–202 的 48 个单行 forward：

| 指标 | 离线重算结果 |
|---|---:|
| acquire 调用 | 2304，即每步 48 次 |
| 请求的专家 ID | 23040 |
| acquire 开始时已有 GPU 槽的 ID | 19563，84.91% |
| 全部 ID 已驻留的 acquire | 778，33.77% |
| 全驻留 acquire 的主机区间总和 | 17.52 ms，即约 0.365 ms/步 |
| 所有 acquire 主机区间总和 | 230.95 ms，即约 4.81 ms/步 |
| demand H2D | 8.812 GB，即约 183.6 MB/步 |
| file fills | 0 |

“已有槽”不保证 DMA 已完成，不能直接当作 Ready 数；计时含 trace 开销，最后一步含状态对齐，不是本轮性能测试。全部 acquire 的 4.81 ms/步包含缺失服务，不能都算成热路径可回收开销。

这组数据支持**每层全 top-k 主机服务存在可拆分工作**；也说明如果只省掉全命中的 acquire 函数调用，收益很可能较小。应关注混合命中层的热计算启动时间、descriptor/提交开销、稳定 residency 对执行组织的帮助，以及新热集是否真正减少 H2D。

### 3.2 Strata 的架构可以借鉴，附带专家名单不能直接照搬

[DERIVED] 已检查 `expert-profile.bin` 的 `STRP/v1`、48×512 维度、24576 个唯一 pair 及反向表一致性。`expert-profile-coder.bin` 是 48×256，应拒绝用于当前 512-expert 模型。

在上述暖 CLI decode 的真实 router IDs 上做**静态集合覆盖**，没有做缓存替换模拟：

| 集合 | 覆盖专家贡献比例 | 全十项都在集合中的层 |
|---|---:|---:|
| Aila 旧 G5 864 项 | 38.89% | 0/2304 |
| Strata 排名前 864 项 | 1.72% | 0/2304 |
| Strata 排名前 1857 项 | 5.82% | 0/2304 |
| Strata 排名前 3202 项 | 12.11% | 0/2304 |
| Strata 排名前 3714 项 | 14.64% | 0/2304 |

另一份旧短 messages decode 也得到相近结果：G5 39.08%，Strata 前 3714 项 14.40%。两种表的前 864 项仅重合 37 项。

这不足以判断差异来自模型版本、专家编号、量化/路由差异还是语料；尚未证明两份权重的 expert identity 一致。也不能把 Aila 动态驻留的 84.91% 和静态表的 14.64% 当作等容量替换策略 A/B。3714 全静态槽只是覆盖诊断；GPU-only 真正运行还需要 cold slots。

**行动结论：优先做 Aila 的 resident fast path，并使用 Aila 自己的路由校准热集。** 不从 Strata 导入名单后直接 pin 满。全热层很少不阻止混合层获益，所以首版必须支持同一层的 hot+cold 合并，不能只优化 all-hot 分支。

## 4. 目标架构：热集稳定，冷供给独立，计算语义一致

[HYPOTHESIS] 在现有 expert pool 总预算内划分两部分，host pool 继续复用现有实现：

| 区域 | 内容与生命周期 | 执行方式 |
|---|---|---|
| Resident set | 在一个执行周期内不可淘汰的热点；NF4 payload、nested metadata、descriptor 一起固定 | router IDs 在 GPU 查表，直接运行；不调用通用 acquire |
| Dynamic/cold region | 其余专家，保留现有动态命中与缺失供给 | Ready 动态项用短期 lease；真正缺失按预算选择 CPU 冷计算或 GPU 冷供给 |

保留完整 top-10、router 权重和原专家权重。第一版沿用现有 NF4 算术，把驻留与供给变更的影响单独看清。R2b 的整数格式继续作为后续计算策略；hot/cold 身份不决定精度，避免会话缓存或未来 prompt 改变数值策略。

### 4.1 固定数据结构

```cpp
// 示意接口，非本轮已经提交的实现。
ResidentEpoch { generation, ready_event, table_dev, descriptors_dev, last_reader };
// table_dev: [48,512] -> resident descriptor index or -1，仅 96 KiB。
// descriptor: packed pointers + nested metadata；epoch 内不可变化。

RoutePlan { token, rank, expert_id, route_weight, resident_index, cold_index };
ColdRequest { epoch_generation, layer, token, rank, expert_id };
```

RoutePlan、冷请求 mailbox 和 scratch 由执行槽预分配。仅仅在 GPU 放一个 slots 表还不够，kernel 也必须从真实 route 和表中取得 descriptor，避免每次回到 CPU 重建数组。

### 4.2 先实现安全的主机分流，再把分流移到 GPU

**H1：主机分流、命中直接绑定。** 先保留现有 router wait：

1. `BindResidentAndReady` 取得固定热槽和 Ready 动态槽；动态项持 generation/reader lease。
2. `AcquireCold` 只接收缺失或尚未 Ready 的项目，同时显式知道本批受保护的动态槽。
3. 可先提交命中计算；冷项到达后算剩余贡献，按原 rank 合并。全命中直接执行，不创建冷服务事务。
4. 固定热槽以 epoch 的末次 reader fence 管生命周期；动态槽继续保留 per-consumer fence。

H1 用来隔离驻留绑定与冷事务的成本，并建立正确的所有权。它仍有 router host wait，不能宣称已经得到 Strata 的 GPU 直接调度收益。

**H2：GPU 分流、命中计算预先提交。** router 写 device IDs/weights；紧接着在 GPU 查 resident table，生成 hot/cold mask 或紧凑列表。在 CPU 等 cold mailbox 之前，就把 resident gate/up、down 提交到现有 compute queue。

首版 count=1 可按 top-10 原 rank 设置固定上界，不要求从 GPU 回读 hot_count 再决定 launch：每个 rank 检查是否命中，缺失项不解引用 descriptor。若 kernel 有工作组 barrier，屏蔽逻辑必须保证工作组参与规则，不能在 barrier 前按 lane 任意 return。紧凑列表可作为后续降低空 work-group 的版本。

CPU 得到冷列表后调用既有供给，在 compute queue 上提交依赖冷 DMA 的计算，最后 merge。这样维持 in-order 和 scratch 约束，也能让 hot work 在 host 服务缺失期间推进；不要求一次改成全局 out-of-order。

冷列表没有项目时，不做 acquire/H2D。首版主机仍可能需要观察 mailbox 才继续组织后续层；要消掉每个 all-hot 层的 host rendezvous，需要额外的可条件继续机制，不能把这一收益提前算进 H2。

### 4.3 合并与失效规则

- hot/cold 对同一个 `(token,rank)` 必须恰好产生一次贡献，复用缓冲先清理未写项；route weight 应恰好乘一次。
- 初版 decode 用现有 rank 输出区，保持 rank 顺序合并，避免 atomics 与新的数学差异。prefill 沿用已完成的有界 scratch/merge 机制。
- `contains()` 只表示有槽，不足以作为直接执行条件；必须有 Ready/event、正确 expert identity 和 generation。
- 热表发布前完成 payload 与 nested metadata 的上传。更换 epoch 先停止新读者、等旧读者结束，再改 descriptor/权重并发布新表；不能先更新 ID 后补权重。
- 冷 acquire 不能淘汰正在被直接路径读取的动态项。沿用 ready-first 的 fence 思路，补显式 reservation/lease 后才允许不把 ready IDs 交给 acquire。
- 多 session 共享模型缓存，epoch 属于模型/执行调度器，不能在一个 session 的请求边界覆盖另一个 session 仍在读的热槽。checkpoint 保存模型状态，不固定热缓存选择；restore 后重新绑定当前有效 epoch。

## 5. 热集怎么选，才能减少供给而不损失动态命中

### 5.1 第一版保留当前预算与工作集优势

[HYPOTHESIS] 第一个对照先复用当前 Ready 驻留内容和短期 lease，不大规模改变替换策略。这样先判断 bypass/提前计算的价值，避免把新热表引入的 miss 变化混在一起。固定 resident 子集的容量随后根据本机数据决定，不预设“全显存 pin 满”或“每层 18 个就是最优”。

引入固定子集时，需要保留足够 dynamic/cold 容量容纳一个最大合法服务批次，并考虑 prefill、R2b staging 与 KV/scratch。总容量不增加；热槽的机会成本就是被它挤出的动态命中。

### 5.2 Aila profile 的最小闭环

- 来源用真实 router trace，分别计 prefill token-frequency、decode frequency 和 unique-demand；不要把去重后的 acquire ID 数当作每 token 路由热度。
- 用已有多领域 traces 产生候选；评估留出的 prompt/conversation，不能用本次完整生成的未来路由反过来选择本次热集。旧二进制 trace 缺阶段信息的局限要保留。
- profile 附带模型/量化配置、专家布局或重排标识、维度、采集版本与语料身份；加载时至少验证兼容性、唯一性、范围和预算。不匹配则回退当前路径。
- Strata 附带排名只用于借鉴数据结构。本地 `make_profile.py` 默认保留 base 的全部既有排名，新的 trace 只追加未出现 pair；当前 base 已列全 24576 项，直接用默认参数追加 trace **不会改变已有顺序**。需要用 `--no-base` 或在 Aila 工具中明确做加权合并。
- 先以 frequency/bytes 为简单选择准则，保留跨层覆盖统计；后续才根据实际 H2D、暴露等待和执行收益做成本加权，不假设固定每层配额或纯全局排名一定最优。

### 5.3 适应发生在边界，不发生在每次访问

[HYPOTHESIS] 初版热 epoch 固定在一个 decode 区间内，计数只记录实际已发生的路由。下一版在 request、prefill→decode 或有界 token 窗口边界进行少量换入；需要明确 swap 字节预算和收益迟滞，避免换入开销超过减少的 misses。

Strata 的 4 rounds/96 swaps 是其实现参数，不能直接设成 Aila 默认。它还可用 CPU 暂算被换出的专家；Aila 首版用 dynamic/cold 区域保证 GPU 退路，接入 CPU 候选后也不能让表切换产生缺失或重复结果。

## 6. Prefill 与 R2b 的衔接

热集的静态选择对 decode 特别有意义；prefill 的专家覆盖广、每专家行数大，策略必须分开。上述短 CLI prefill 中，G5 864 项只覆盖约 10.58% 的路由贡献；不能据 decode 的 38.89% 预测同样收益。

**H3：prefill 先按实际 route 将 resident 与 nonresident 分组。** resident 专家直接绑定 device descriptor，仍按 expert-major 汇总 token；缺失项用 B6 的有界供给。当前 S6 已有 resident-first 拆批，应复用其中 rank/fence 处理，但最终让 resident 批不再进入通用 acquire。

初版不借用固定热槽作 scratch，明确记录固定热槽是否增加 prefill 的需求字节。若保留热槽代价过高，再做显式的 prefill/decode ownership handoff：撤销表、等待读者、借槽、恢复，并把 restore/refill 计入 TTFT。不能在报告中只计 prefill kernel，隐藏热集重建时间。

[REPORTED] R2b 已在 {32,48,64} 的有界 fixture 获得 DPAS/graph 收益，并通过一个真实输入的模型 canary；尚无可比较的生产端到端收益。见 [DPAS 结果](/E:/RiderProjects/Aila/docs/plans/qwen3.8-flash-next-r2b-dpas-results-2026-10-01.md) 与 [真实质量结果](/E:/RiderProjects/Aila/docs/plans/qwen3.8-flash-next-r2b-real-quality-results-2026-10-01.md)。

驻留表和可复用 device descriptor 能为 R2b 提供稳定入口，但第一版不把全部热专家展开成 INT8 长驻，那会消耗缓存容量。先保持 NF4 常驻、INT8 有界 staging；接入时独立评估转码复用的净收益，并保持既定数值契约，不按当前命中情况随机选择精度。

## 7. CPU 异构重新进入优先计划：比较新的冷路径，而非重复旧放置实验

### 7.1 撤销路线层面的否定，保留旧测量

[REPORTED] [9 月 28 日判断](/E:/RiderProjects/Aila/docs/plans/qwen3.8-flash-next-cpu-route-value-review-2026-09-28.md) 基于当时明确的负结果：较早低配额 overlap 请求回退约 3.21%，prepared miss-only 的筛选也回退，强制 eligible experts 放 CPU 的新 dataflow 仍慢约 53%。这些实测不会因发现新架构而失效。

但它们没有测到“GPU 按稳定 residency 自主消费热专家，CPU 冷计算独立提交，两者仅在贡献合并处汇合”的完整设计。当时 CPU/GPU 重叠确实已实现，不能说旧实现完全串行；问题是并行的起点、任务选择、可复用 GPU 路径与 cache admission 仍绑在主机逐层动态计划中。

[CODE] 旧计划还允许 host-ready GPU hits 放 CPU、同步准备未就绪权重，以及只按 credits/rank 决定放置；CPU 算过某专家又不会自动迁入 GPU。新设计必须将这些选择拆开，否则会重复制造 CPU 长尾或放弃已有 GPU 命中。

**现在有充分理由重启异构架构工作；尚无证据宣称新异构一定更快，也没有证据把全部旧回退都归因于 acquire。** 旧 CPU 准备、算术、join 与资源竞争数据继续作为预算约束。数值验收采用用户已允许的容差与模型质量指标，不再将 greedy token 不同单独作为拒绝依据；rank/lease/state 错误仍必须修复。

### 7.2 三种专家去向，独立决定计算与驻留

[HYPOTHESIS] router 仍选完整 top-k；placement 只决定每个已选专家的执行位置：

| 已选专家状态 | 首选路径 | 选择依据 |
|---|---|---|
| GPU resident/Ready | GPU 直接计算 | 已有权重，通常没有理由搬回 CPU；不占 CPU credits |
| GPU miss、host Ready、低 token 复用 | CPU cold 候选 | 避免整份权重 H2D，输入按层一次回传、结果按批合并 |
| GPU miss、多 token 复用高或 CPU 队列拥塞 | GPU cold staging | 可摊销 H2D 和转码，衔接 R2b 大 M 路径 |
| host 也未 Ready | 有界 reader 完成后再决定 | CPU 同样需要权重；不能把缺页/读盘从计时中移走 |

CPU 本次计算与将来 GPU admission 分开：某冷专家本步由 CPU 计算，不意味着永远留在 CPU。根据过去路由频次和实际重用，在 epoch 边界决定是否迁入热集；低复用项则可不污染 GPU cache。

[DERIVED] 当前单专家 NF4 slot 为 2,534,400 B；一个 FP32 输出向量仅 `2560×4=10,240 B`，payload 比为 247.5。BF16 输入每层每 token 5,120 B，可供本层所有 CPU 冷专家共享。CPU 路径确实可能显著降低 PCIe 需求，但仍在主机 DRAM 读取整份 NF4 权重，并增加输入就绪、计算与合并成本；这个字节比不是速度比。

### 7.3 优化目标是层完成时间

可用下式表达候选的关键路径关系，而非把计数器直接相加：

```text
layer MoE completion ≈ max(
    GPU shared/hot/cold 分支的完成时刻,
    CPU input-ready + queue + cold compute + result-ready 的完成时刻
) + merge
```

GPU cold 的 copy 与 compute、CPU compute 与 I/O 会争用资源；需要测完成时刻而非假设两者独立。CPU 预算应限制预测完成时间和队列积压，不能只设置“每层最多十个专家”。初版可用简单的按任务数/权重格式分桶 EWMA 与余量，不必建立复杂学习调度器。

GPU hot 在 route 后先入队；CPU worker 等一次有事件约束的输入就绪，批量执行 host Ready 冷项。主机同时服务选择在 GPU 上执行的冷项。CPU 输出仅在最终 merge 前需要等待，避免 coordinator 提前 join 阻止 GPU 分支提交。复用已有常驻 worker、prepared AVX2、lease、输出区和 drain，补足直接热路径，不重建线程系统。

预留 I/O/PLE/提交资源，不能让 CPU TBB 线程数吃满可用核心后，把供给停顿误判为 GPU 算力不足。暂不移植 CUDA persistent doorbell；事件驱动即可检验这一结构。

### 7.4 Prefill 的异构切分按专家桶

R2b 的收益集中在已有足够 token 行的大桶；CPU 可先探索 host Ready、低复用的小桶，把大桶留 GPU。CPU 小桶需要按 expert 复用解量化/输入准备，旧 flat batch 的多个 expert/token tasks 不等于这种复用。

首个异构对照先做 decode，随后最多一个自然 prefill。不要再以 forced all-CPU 作为主候选，也不要把小桶阈值直接写成未经测量的固定 dispatch。采用相同算法精度契约，不能让缓存是否命中决定使用未经校准的数值格式。

### 7.5 最小而有归因能力的对照

在同一个新热路径、同一初始 resident set、相同总预算下比较：

- **A：resident direct + cold GPU。** 这是 H1/H2 的基线。
- **B：resident direct + 有界 host-Ready cold CPU + 其余 cold GPU。** 复用最佳已有 prepared CPU 实现。第一轮冻结 resident set；GPU 冷项走可回收 staging，不把其临时槽变成下一步的新命中，两边都不做 promotion，先隔离冷计算去向。否则 A 的冷 GPU 计算会顺带建立未来驻留，而 B 不会，名义上相同的 admission 开关仍会混入缓存差异。

记录 CPU 实际任务数、开始时刻、queue/compute/result-ready、merge 暴露等待，以及 GPU hot/cold 完成时刻和 H2D bytes。先让任务 ID、贡献和状态正确，再比较总时延；任务太少或被 admission 全部拒绝的运行不能用来判断异构价值。

保留当前动态缓存 GPU-only 的整请求时间作为第三个参考，避免新 B 只击败一个变慢的新 A。之后才允许 admission 策略不同，单独看 CPU 留冷/GPU 留热带来的缓存收益。无需先重跑线程×credits×placement 大矩阵。若 B 获益就继续；若 B 不获益，结论限定为这台机器、这套冷任务与执行契约，仍不能泛化为“异构本身慢”。

## 8. 下一位实现者的工作包与最小验收

| 顺序 | 工作包 | 完成定义 |
|---|---|---|
| H0 | 本机热集来源与计数口径；记录当前驻留及混合命中 | 本轮离线审计已给出起点；profile identity 和跨域覆盖仍需实现时处理 |
| H1 | BindReady/lease 与 AcquireCold 分离；沿用 NF4 | resident 不再作为需求项交给通用 acquire；冷项正确供给；不改变实际 top-k |
| H2 | device residency/descriptor table、GPU hit selection、先提交热计算 | 冷服务前已有 hot kernel；主机不重建热 descriptor；全命中无冷事务 |
| C1 | 基于 H1/H2 的有界 CPU cold 路径 | 同热集对比 cold-on-CPU/GPU，GPU 命中保持直接执行，旧 worker/lease 复用 |
| H3 | prefill resident 批直接执行与 decode 热集交接 | 保留 R1b scratch 优势，记录 prefill 和热集重建的总成本 |
| H4/C2 | Aila profile、有界 epoch 自适应、compute/admission 解耦；随后接回 R2b | 在同预算下获得稳定的 miss bytes/总时延收益，再确定默认配置 |

不要将“已有旧 pinning 负例”作为跳过 H1/H2 的理由，也不要将“热路径代码写完”当作性能已经提升。

验证沿用现有工具，只增加有价值的边界：

1. 用一个小 fixture 覆盖全热、混合、全冷、Copying、跨 token 重复 expert、旧 generation；确认输出槽位/权重/贡献不丢不重，且 lease 释放与 reader fence 正确。
2. 一个真实短请求加一个自然长 prefill，在相同预算、精度和 warm 状态下对照。先保持驻留策略一致测 H1/H2，再改变热表；复用现有 continuation/cancel 检查，不再跑大范围 kernel sweep。
3. 记录 `resident_route_entries / direct_entries / cold_request_entries / acquire_calls / demand_bytes / resident_dispatch_gap / epoch_refill_bytes`。既有 hit-rate 统计原来在 acquire 内更新，bypass 后必须迁移，否则会出现命中率下降、prediction gating 误判的假象。旧 cache trace/replay 也要区分 route、direct binding 与 cold transaction。
4. 成功以整请求 prefill、decode、TTFT 和冷启动/切换成本判断。若只省约几百微秒而缺失字节上升，应缩小固定热区或继续使用动态 residency；若提前计算被 DMA 争用抵消，保留命中直接绑定但调整提交顺序。

**下一步明确先做 H1/H2，并以它作为 C1 异构冷计算的共同基线：让已在 GPU 的专家直接计算，CPU 与 GPU 共同承担合适的冷项。按实际层完成时间决定分工，再确定热集容量、自适应频率与 R2b 集成。**
