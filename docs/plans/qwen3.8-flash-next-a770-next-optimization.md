# Qwen3.8-Flash-Next / A770 后续优化计划

日期：2026-09-25。勘察基线：`91c9d8b`，开始勘察时工作树干净。

接续 [A770 Handoff](../Qwen38FlashNext_A770_Handoff.md)，保留 [最初适配计划](qwen3.8-flash-next-bnb-nf4.md) 的模型语义和 M1–M4 边界。本轮仅做代码勘察与方案设计，没有修改运行时代码，没有构建、运行 CTest、实模生成或 GPU benchmark。以下历史性能均引自交接文档，不是本轮复测。

**建议下一轮先做：prefill MoE 批量提交、cache 热路径去分配、预取按真实需求完成，而后再尝试更深的流水线和 XMX。** 当前瓶颈已不是“尚未有 host pool / 异步 DMA / prefill 预取”。不以完整 profiling 工程作为动手前置条件，也不承诺未实测的 tok/s。

## 1. 基线与证据口径

标记约定：

- `[CODE]`：当前提交可直接确认的实现事实；文内代码行号以本次基线为准，后续以函数名定位。
- `[MEASURED:handoff]`：交接记录的历史测量，保留其 workload、冷热状态及测量边界。
- `[DERIVED]`：由尺寸、控制流或公式推导，不能直接解释成实测耗时/DRAM 流量。
- `[HYPOTHESIS]`：待实现或待实验的设计，不是已验证 dispatch。
- `[CORRECTNESS]`：实施必须保留的语义、数值或生命周期约束。

目标环境沿用交接：A770 16 GB、显示输出共用显存、PCIe 3.0 x16、64 GB DDR3、NVMe 模型、oneAPI 2026.1、驱动 `32.0.101.8860`。本轮没有重新查询驱动。模型为 `C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2`，48 层，36 DeltaNet + 12 full attention，512 experts / top-10，hidden=2560、expert intermediate=640，BF16 激活、NF4 packed 权重与 nested u8 scales。

优化指标分开记录：进程启动、完整 TTFT（含所有 prefill chunk 和必要快照）、固定生成量的总耗时、decode 平均/分位耗时。局部 `fill`、`acquire`、kernel device time 只用于解释变化。

### 1.1 交接文档需要校正的地方

| 交接表述 | 当前判断 |
|---|---|
| “M1 后全部未提交” | 已过时。当前 HEAD 是 `91c9d8b`，Stage 4 等已有提交；不按未提交补丁重新应用。 |
| §7 仍把 prefill 跨层 fill overlap 列为未做 | Stage 4 与 `ForwardImpl` 已实现预测、频次排序、默认最多 48 个专家的 host-only 预取。后续是改粒度和调度。 |
| §3 描述 acquire 等待 DMA、copy 依赖 main barrier | 当前 acquire 返回 copy barrier event，MoE gate/up 建立依赖；原先的 main barrier 已移除，安全性依赖 router 处同步。 |
| `dma_wait=0`、“消除 GPU idle” | 只说明主要路径不再在 CPU 等待 DMA；device 仍可能等待 event，fill pool / prefetch 仍有 host 阻塞。 |
| Stage 3 移除 `qkv` / `ple_out` 分配 | 数据路径已不使用它们，但 `ensure_runtime_buffers` 仍分配 `buf_.qkv`、`buf_.ple_out`，以及未使用的 `moe_acc_f32`。 |
| `/boost` 中 `hc_decode_up_mix` 使用 act SLM cache | 当前函数仍直接 `vec8` 读取 act，没有该 SLM accessor；不能按交接假定优化还在，也不因此重试已知 GEMV SLM 负优化。 |
| “57/57 全通过” | 同文也注明 43 pass + 14 skip。不能把 skip 当成实模覆盖，本轮未独立核验。 |

`[MEASURED:handoff]` Stage 4 的 story 主 prefill `count=26`：fill 1876.90 → 525.90 ms，该 forward 总耗时约 2615.2 → 2442.74 ms；但 48-token 整次生成由 7268.6 ms 变为 7507.9 ms（另一条 7770.5 ms）。这些记录不足以认定整次加速，也不足以单凭不同运行状态断言回归。69.16 ms 是 peak decode token，不是持续平均。`q4e_prof.ps1` 的 prefill 摘要只抓 `pos=0` 大 chunk，不含后面的 4-token anchor，所以不能直接称作完整 TTFT。

旧的 fake-hit 48.9 ms floor 使用错误专家输出，且早于后续改动，只能作历史诊断。旧 isolated MoE 约 10–12 ms 和旧 `router_sync` 也不能直接拼成当前 token 的时间账。

## 2. 代码勘察地图与现有关键路径

| 路径 / 函数（基线行号） | 确认的事实与优化入口 |
|---|---|
| `src/models/Qwen4ExpBnb4Backend.cpp:945`，`ForwardImpl` | 当前和下一层预测 router 都在 `ctx.synchronize()` 前提交；每层一次 host 路由边界。 |
| 同文件 `:988–1105` | shared expert 提交后 acquire；prefill 批量 acquire，随后仍逐 token 调两次 decode MoE kernel。 |
| 同文件 `:1038` | prefill 预测在实际 acquire 返回后提交，使用频次 top-48；不是任意时间都可并行填充。 |
| `src/models/Qwen4ExpExpertCache.cpp:650`，`acquire` | 检查命中前先 join 整批 prefetch；fill 仍在 CPU 完成，仅 DMA completion 异步返回。 |
| 同文件 `:24`，`ExpertFillPool` | 单个 in-flight job；默认总线程数 8 包括调用线程，异步期间通常是 7 个后台线程，`wait()` 时调用线程也参与 Drain。 |
| 同文件 `:700–935` | 临时 vector、unordered_map 节点和 list erase/push；命中路径也有节点释放/分配。GPU reservation 在 acquire 返回前释放。 |
| 同文件 `:942`，`prefetch_async` | 只填 host，但会淘汰 host pool 的 LRU；“错误预测只花空闲 CPU、不发生驱逐”的注释不再完整。 |
| `src/ops/MoeExpertOps.cpp:256,390` | 现有内核为单 token；gate/up 按 expert×output 行，down 每个 output 行依次累加 top-k。 |
| `src/ops/Bnb4BitLinear.cpp:3067`，`Bnb4GroupedMoE::forward_batched_gpu` | 已有按专家聚合 / grouped XMX 结构可借鉴，但 GELU、BF16 累加、连续专家权重和分配方式与 Q4E 不兼容。 |
| 同文件 `:2163`，`Bnb4BitLinear::forward` | fused prefill 分支仍做输入/输出 D2D staging；fallback dequant 后同步，需按实际 dispatch 判断是否命中。 |
| `src/ops/HyperConnectionOps.cpp:224,268` | decode HC 已有 down+inject、up+mix；不是从未融合的常规 dense 层。 |
| `src/ops/GatedDeltaNetOps.cpp:443` | 本模型 geometry 命中 grouped prefill；单 kernel 内循环 token，反复访问 state。 |
| `src/models/Qwen4ExpBnb4Backend.cpp:1186`，`CaptureSnapshot` | prefill chunk 以及 decode 周期边界做 recurrent snapshot；每层 state、conv 分别阻塞回读，再复制到 snapshot vector。 |
| 同文件 `:488,534` | expert pool 先估算分配，之后 runtime buffer 扩容；新增 scratch 必须进入显存预算。 |
| 同文件 `:361` | LM head 原权重转置后已 erase，且关闭额外 GEMV copy；不存在可直接回收的第二份 1.27 GB LM head。 |
| `src/inference/Model.cpp:531` 附近 | shared-model loader 仍为 Qwen3 / Qwen3.5；Q4E 后端 scratch/cache 目前属于 backend，不能直接并发调用当作 M4。 |

现有 decode 的简化时序（按单层）：

```text
main in-order:  前层收尾 → HC/Attention/HC → 当前路由 → 预测路由 → wait
                                                            CPU 读 ids
main:                                                        shared → gate/up → down → add/inject
CPU:                                                         acquire(fill) → prefetch(L+1)
copy immediate:                                              H2D --------↑ dma_event
```

`[CORRECTNESS]` 当前 router 边界同时承担“路由结果可见”和“前层不再读取可驱逐 GPU slot”两项职责。不能只看到 wait 就删除；同一层小 cache 的逐 token acquire 也需要检查前 token 的 slot 使用期。

## 3. 推荐实施顺序

优先级表示实施顺序；工作量仅用于拆分任务，不是工期承诺。所有新实验开关默认关闭，成熟路径保留回退。

| 阶段 | 交付 | 主要收益来源 | 状态 / 提交 |
|---|---|---|---|
| S0 | 小范围耗时口径补齐 + host 热路径去分配 | 可比性、CPU 分配和提交成本 | 已完成 (`9abc996`) |
| S1 | prefill MoE 两个批量 kernel | 大幅减少 host 提交，保持计算顺序 | 已完成 (`b51a2cc`, `645b9d4`) |
| S2 | host slot 生命周期 + demand-first fill | 避免无关预取阻塞、保护 DMA source | 已完成 (`26838e1`, `cac125b`) |
| S3 | router event 边界与 shared 提前提交 + 2-tier SLRU | 用真实独立计算覆盖 CPU acquire 与抗抖动 | 已完成 (`49a69e8`, `3ae5d20`) |
| S5 | 快照异步批量回读、workspace 回收 | TTFT、尾延迟、内存余量 | 已完成 (`d8205c1`) |
| S4 | expert-major prefill / 权重多 token 复用 | 同专家多 token 权重复用、L2 高效命中 | 已完成 (`AILA_Q4E_EXPERT_MAJOR_PREFILL`, prefill 24.7s→9.6s, 100% 比特精确) |
| E1 | 预测轻量化 (Ids-only Prediction) | 降预测成本，跳过无消费者的 softmax/weights | 已完成 (`61a46c5`, decode 83.3ms→75.4ms, 100% 比特精确) |
| E2 | DeltaNet 分块、HC/LM head 专项、AOT | 第二阶段计算与启动优化 | 实验性候选 |

可以先独立实施 S1 与 S0；S4 不应作为 S1 的前置条件。S2/S3 先保留 layer router 的顺序依赖，不一次重写整个 scheduler。

## 4. S0：小改动与最少观测

### 4.1 消除确定存在的 host 热路径分配

`[CODE]` `acquire` 每次创建 `misses`、`batch_reserved_slots`、`to_fill`，fill chunk 创建 `assigned_hslots`、`parts_left`。GPU/host 两套 LRU 的命中也用 `erase + push_front`。Stage 4 预分配预测 vector 不代表整个热路径已经零分配。

`[HYPOTHESIS]` 最小改法：

1. 命中移动改为 `std::list::splice`，保持当前 LRU 次序，不更换淘汰策略。
2. 将 acquire 工作数组放入单执行槽 scratch，按最大单层 512 experts 预分配；decode 可用 top-k 有界数组。填充任务捕获的 scratch 必须活到任务结束。
3. `(layer, expert)` 空间固定为 `48×512`，可用直接索引表替代 GPU/host 的 unordered_map 节点操作。单张 int 表约 96 KiB；先做 splice / scratch，再决定是否值得换表。单纯 `reserve()` unordered_map 不会消除每次新节点分配。
4. `prefill_unique_`、`prefill_unique_slots_` 也在 load 预留容量。预测排序若要真正无堆分配，可按 512 个候选的预分配索引排序，显式保留同频次顺序；`stable_sort` 可能使用临时堆内存。

收益不估成毫秒；它减少已确认的分配与调度负担。使用现有 cache 单测验证 slot/LRU 行为即可，不为 splice 单独扩展整套测试。

### 4.2 只补足能指导选择的指标

`[HYPOTHESIS]` 复用 `ExpertCacheStats` 和 `q4e_prof.ps1`，增加 GPU hit、host-ready hit、host-fill miss、prefetch useful/unused、等待 required/非 required fill、host slot 因 DMA 尚未完成不能重用的次数。异步预取耗时不可只累计 acquire 内的 `fill_ns`，否则会误把“移出统计区间”当作消失。

只在选择 S3/S4 的候选时，保留 gate/up、down 和 copy 的少量 event，以 token/layer ring buffer 记录；取结果放在原有完成边界，不能新增逐 kernel wait。`Context` 已支持 opt-in profiling，但 cache copy queue 目前没有 `enable_profiling`，要在同一个可选模式中补齐。跨队列重叠用经过校准的时间线判断，不能随意相加 event span。

不要求先写全模型每个算子的 event 包装，也不要求先跑 VTune。只有出现“device 变快、整次变慢”且这些小指标无法解释时，再针对一个阶段取 trace。

## 5. S1：prefill MoE 从逐 token 提交改为批量提交

### 5.1 先改变提交粒度，保留原算术

`[CODE]` 即使 `batch_acquire=true`，`ForwardImpl:1056` 仍对每个 token 调 gate/up 与 down；`buf_.moe_h` 只有 `[10,640]`，导致 token 间必须复用串行 scratch。

`[DERIVED]` 对单个 N-token forward，48 层仅 routed MoE 就有 `96N` 次 kernel launch。N=26 为 2496 次，N=274 为 26304 次，尚不含 attention、HC 等。这是静态调用次数，不是 launch 时间测量。

`[HYPOTHESIS]` 新增 `moe_prefill_gate_up_rows` / `moe_prefill_down_rows`，直接适配 `MoeExpertSlotDev`：

- gate/up grid 增加 token 维度；每个 subgroup 仍负责一个 `(token, expert-rank, intermediate-row)`，复用当前 uint32 packed load、NF4 qmap、nested scale 与 reduce 顺序。
- down grid 增加 token 维度；每个 subgroup 仍负责一个 `(token, output-row)`，内部按照原 top-k rank 顺序累加 10 个专家。
- 一份 device slot descriptor 表，加 `[N,10]` slot-index 与 f32 weight 表；host 预分配 pinned 元数据，异步上传，main in-order 串联其使用。
- `h[N,10,640]` BF16 预分配。N=274 时约 3.35 MiB；不用 `[N,10,2560]` 的完整专家输出。
- 两个大 kernel 代替 `2N` 个小 kernel；gate/up 只依赖一次 `batch_dma_event`。权重读取量不保证下降，这一步的明确目标是提交粒度。
- 元数据上传、shared/预测预取等不能逐 token 引入新 wait。为减少 ring buffer 生命周期复杂度，首版仍保持每层原有 router 边界。

`[CORRECTNESS]` 保留以下计算：nested scale 的两次 f32 rounding；gate/up 的 f32 点积和 `bf16(silu(gate_sum) * up_sum)`；down 的 subgroup reduction、top-k 顺序以及最终一次 BF16 舍入。权重高 nibble 对应先出现的元素，不能套用技能中另一布局的 low-nibble-first 示例。

首版适用于当前 `num_slots >= num_experts + top_k` 的 batch-acquire 路径。小 pool 暂留原路径，同时补清其逐 token victim-read 依赖；以后再按可容纳 expert 集合分 wave。不能将一个层的 slot 提前驱逐，也不能让 descriptor 指向已换绑的 nested metadata。

建议实验开关（尚未存在）：`AILA_Q4E_PREFILL_MOE_ROWS=1`。首轮只需新旧 kernel 在 N=2 / 26 的代表输入、nested/f32 两种 scale 下比对，再用 lighthouse 看提交数和完整 prefill；短 story 作为小 N 回退判断。位级差异应先定位，不能用自然语言输出看起来相同替代算术检查。

### 5.2 为什么不直接接入现有 Bnb4GroupedMoE

`[CODE]` `Bnb4GroupedMoE::forward_batched_gpu` 提供了 route histogram、gather、group descriptor、grouped GEMM 的结构参考。但它使用 GELU-tanh，部分 accumulate 路径每个 contribution 都转 BF16，而 Q4E 使用 SiLU、f32 顺序和最终 BF16；它的权重所有权也不是可换绑的专家 slot。

可以提取中性的 route-layout 工具；不能直接换类或调用其累加内核作为“等价优化”。

## 6. S2：按实际需求完成 fill，显式保护 slot

### 6.1 去掉整批预取 join 的队头阻塞

`[CODE]` `acquire:685` 在 GPU 命中查询前等待所有预取结束。哪怕真实 top-10 已全在 GPU，或只需要其中一个已经填好的专家，也会 join 剩余预测任务。prefill 最多 48 个预测、单 fill job 的设计放大了该边界。

`[HYPOTHESIS]` 分两步实现：

1. GPU hit 查询前移；全命中路径无需为当前算子等待 host fill，但不能清空仍被 worker 引用的 `prefetch_items_`。保留任务对象直到其完成，下次预测也不强制覆盖旧对象。
2. 固定容量任务池，每个 expert 记录独立 `parts_remaining / ready`，支持 `WaitReady(required_ids)`；真实 miss 升为 demand，尚未开始的无关预测可取消，已经开始的部分完成后再回收。默认线程预算保持现有 8，不再用加线程抵消串行。

真实 miss 的 host-ready 数据立即 DMA；缺失部分优先填充，只有真正需要的专家阻塞当前计算。`start()` 只允许一个 job 的现有 pool 需要重构，不能只把 `wait_prefetch()` 改成条件 wait 后继续复用 lambda 捕获和数组。

### 6.2 生命周期是进一步 overlap 的前提

`[CODE]` host pool 在 fill 前就把 key 写入 map；当前整批 wait 充当“内容已 ready”的隐含证明。async DMA 后没有每个 host slot 的 read-completion 记录，prefetch 会从 host LRU 选 victim。GPU slot 的 `slot_reserved_` 只保护一次 acquire 的分配过程，返回前就释放。

这些是静态可见的依赖缺口，不等于本轮已经复现数据损坏。尤其小 host pool、同层逐 token fallback、多 wave prefill，以及后续移除 wait 时必须明确处理。

`[HYPOTHESIS]` 建立有界、带 generation 的元数据：

```text
HostSlot: key, generation, Free/Filling/Ready, parts_remaining, last_dma_read
GpuSlot:  key, generation, last_dma_write, last_compute_read, reservation_count
FillTask: key, generation, Demand/Speculative, remaining_parts, cancelled
```

`[CORRECTNESS]` 同 key 查到 slot 不代表 ready；worker 完成时校验 generation，发布 ready 使用 release/acquire；host victim 必须没有 fill writer 和 DMA reader；GPU victim 必须等旧计算结束，新消费者等 DMA write 完成。prefetch 中预测错的专家会占用 host cache 和 page cache，必须计入预算。

优先用事件/短生命周期 reservation，不引入 GPU 持久轮询。取消或异常时先排空已提交 GPU/copy 和填充任务，再释放 USM，异步异常需上抛到执行层。

### 6.3 预取从固定数量走向受预算约束

`[HYPOTHESIS]` 保留 48 作为当前回退上限；在已经得到的预测列表中先过滤 GPU-resident / host-ready，再按 token 频次、预测信心和近期有效性选择缺失专家。现实现先截 top-48、再由 cache 过滤，因此可能消耗了候选名额却没有填入有效新专家。

首版可只做“缺失候选优先 + 当层无 miss 时少预测/少发任务”；动态配额后做。一个可试的模型是：

```text
budget_bytes <= estimated_background_fill_bandwidth * next_use_window
quota <= min(current_cap, reclaimable_host_slots, budget_bytes / slot_bytes)
```

窗口、带宽均来自近期非阻塞统计；设置平滑与滞回。收益按“减少真实 demand stall”计算，不按预测 top-k overlap 或 host hit 单项最大化。全 GPU 命中时，昂贵的下一层预测本身也可能不值得做。

`[CORRECTNESS]` 不清空 OS page cache 做常规验收，不把 16 GiB pinned pool 继续扩到 32–40 GiB，不恢复已失败的预测专家 GPU DMA。这里预取始终是 host-only。

## 7. S3：router event 与 shared expert 提前提交

`[HYPOTHESIS]` 为 `router_softmax_topk_device` 增加返回 event 的接口或专用 overload：

1. 提交本层真实 router 和必要预测 router，记最后一个 route-ready event。
2. **在 CPU wait 之前**向 main queue 提交现有 shared expert 链。
3. CPU 仅等待 route-ready event；这个 event 仍涵盖 in-order queue 的所有前序层工作。
4. host 开始 acquire，shared 可继续执行；真实 miss 的 H2D 用 copy queue，routed MoE 保持 DMA event 依赖。

当前是 wait 后才提交 shared；新顺序把独立工作先放进队列，有机会改善 batched command-list 的提交间隙。保留生产主队列模式，不打开全进程 immediate。

`[CORRECTNESS]` shared 只读当前 x、写独立 shared buffers，acquire 不得覆盖这些 buffer。若 stage profiling 开着而额外同步，会掩盖此设计收益；选择候选时关闭该模式。若单个 event wait 没能带来预期设备执行 overlap，也不能仅据提交顺序声称成功。

下一步才考虑真实 route-ready 与 prediction-ready 分开：先处理真实 miss，预测完成后再提交下一层 fill。这需要独立预测结果 ring 和有效期管理，避免 CPU 读取未完成的 pinned ids。更激进的“无 router host round-trip”留给 E1，不能用预测 ids 替代真实 ids。

## 8. S4：按专家聚合 token，复用权重

`[HYPOTHESIS]` S1 稳定后，将 route 表变成 `expert_offsets[513] + routed(token,rank)`，按每个专家的实际 token 数 `m_e` 调度。

数学语义为 `X_e[m_e,2560] × W_gate/up^T`，再 `H_e[m_e,640] × W_down^T`；packed layout 为 gate/up `[1280,1280 bytes]`、down `[2560,320 bytes]`，bs=64，nested scales 引用该层 metadata。

### 8.1 三种候选，不预设 dispatch 门槛

- `m_e` 很少：继续 S1 或按专家的多行 packed GEMV；保留一输出行/SG，不尝试已有负结果“一个 SG 两输出行”。
- 中等 `m_e`：多 token tile 共享专家权重读取，每个 token 保持独立 accumulator；输入暂存和寄存器压力必须按新形状评估，不能套用 dense decode GEMV 的负优化。
- 热专家 `m_e` 较多：tile 内 NF4/nested dequant → BF16 → XMX，或有限的临时 BF16 专家 workspace。只转换当前处理的专家，不把整个 GPU expert pool 扩为 BF16。

初步候选行数 1/2/4/8/16/32 仅用于收集 crossover，不作为默认条件。N=26 即使 260 条 route 也可能分散给很多专家，所以小 prompt 不保证有 GEMM 收益；必须按 `m_e` 分布而非总 N 选择。

`[DERIVED]` 一个 compact expert 约 2,534,400 B；同一层所有路由按原路径的逻辑权重读取为 `N×10×slot_bytes`，理想一次读取每个唯一专家则为 `U×slot_bytes`，比值 `N×10/U` 只是复用机会的上界。它没有计入缓存、tiles 重读、scatter、PCIe 与填充，不能当作加速倍数。

### 8.2 有序还原输出与空间预算

expert-major 不能依靠无序 atomic add 输出到 token。保留 `(token, original_topk_rank)` 映射，先存 per-expert f32 down 结果，然后按照原 rank 顺序加权累加，最后转 BF16。

N=274 时，完整 f32 down scratch `[N,10,2560]` 约 26.76 MiB，另加 h 约 3.35 MiB；input gather 可再增加约 13.38 MiB。必要时按 token/output-column tile 分块，避免 workspace 挤出专家缓存。所有 scratch 预分配，不能在每个 expert 上重建 oneDNN descriptor。

`[CORRECTNESS]` XMX BF16 dequant 会引入不同于当前 f32 权重乘法的舍入，点积次序也可能变化。即使采用有序 f32 merge，也不保证 bit-exact。将此路线保持实验开关，报告误差、route/top-k 是否变化和 greedy 差异；不能在严格等价模式中静默替换。

本机已有 packed NF4 GEMM / grouped GEMM 可复用布局经验，但技能里的 oneDNN u4 baseline 是线性 INT4，不等价 NF4 codebook。不能因 DG2 支持 integer INT4 DPAS 就直接把 NF4 当 U4。

## 9. S5：快照与 workspace

### 9.1 批量异步回读 recurrent snapshot

`[CODE]` `CaptureSnapshot` 每层调用两次阻塞 `memcpy_d2h`，随后 `snapshot.layers[i] = host_vector` 复制；36 个线性层 state 合计 108 MiB，conv 约 4.22 MiB，另有 PLE conv 约 0.35 MiB。prefill 每个 forward 都触发，decode 默认每 64 token 触发。

`[HYPOTHESIS]` 给 snapshot slot 分配连续 pinned slab，保存所有 layer offset；提交 state/conv 的异步 D2H，末尾统一等待一次，再发布 snapshot。避免先写 host mirror 再拷贝完整 snapshot 的双份 CPU copy，需同时调整恢复逻辑对 mirror 的使用。第一版仍在 Forward 返回前完成，不跨下一次 recurrent 写入重叠。

这不会减少约 112.6 MiB 必需快照数据量，但有机会减少七十余次提交/等待边界和 CPU 复制。继续保留 N−4 + 4 的 recurrent anchor、Truncate 与多轮复用语义。减少 checkpoint 频率、只保存 state 差分或后台跨 token snapshot 属于后续独立设计，不能为单轮跑分快速关闭。

最小验证是一次“prefill → snapshot → decode → restore → 再 decode”状态和 logits 对比；不需要重跑所有模型服务用例。

### 9.2 明确长期内存与执行 workspace

`[HYPOTHESIS]` 回收已确认无消费者的 `qkv`、`ple_out`、`moe_acc_f32`；`hc_mix_up`、fallback scores 等仍有使用者，按 dispatch 延迟分配或复用前必须核查完整路径。

`[DERIVED]` 仅前两个废弃 BF16 buffer 在 cap=274 时约 11.77 MiB，实际 cap 按扩容规则，不能宣传为 GB 级收益。大头是 `scores[24,cap,cap]`、`incr_scores[24,cap,max_seq]`；cap=max_seq=2048 时两者合计约 768 MiB，即便部分 exact attention 路径不使用它们。先确定真实 fallback 需要，再按适用路径配置 scratch，不删除必要兜底。

pool sizing 应纳入 S1/S4、attention、snapshot staging 的明确预算，并记录实际 runtime 峰值。`global_mem_size` 和 Context ledger 不等于 WDDM 可用预算；保留现有 VRAM reserve，回收 workspace 优先形成余量，不立即把全部余量加回 expert pool。

`Bnb4BitLinear::forward` 的输入/输出 D2D staging 也可做单独低风险候选：只对已证明连续、对齐、不别名的 fused prefill 路径直接传入原指针。fallback/LoRA/scratch 所有权必须保留；没有确认实际 dispatch 前不估节省量。

## 10. 实验性方向

### E1. 静态子图、轻量预测、命中路径

`[HYPOTHESIS]` 先录制固定地址的层内段：HC → attention/DeltaNet → HC → router，或 route-ready 后的 shared/MoE/epilogue。host acquire 仍在图外。`MoeExpertSlotDev`、weights、position、conv_head 使用固定参数块；当前 MoE kernel 将最多 16 个 descriptor/weight 按值捕获，必须改为读 device 参数表后才可能复用静态图。

A770 技能中的 limited graph 功能证据来自另一驱动 `32.0.101.8974`，不是本基线 8860 的证明。实现时分别检查 limited/full graph，保留直接提交回退；静态 replay 不要求 full update 能力。oneDNN 节点、attention 长度 dispatch、PLE 和 checkpoint 可能需单独分段。host 修改参数前必须结束旧使用或使用双缓冲+依赖；不用后台轮询 persistent GPU kernel。

`[HYPOTHESIS]` 下一层预测只用于 host prefetch，因此可采用便宜的 ids-only top-k：在 BF16 logits 上排名，不计算完整 softmax 和无消费者的 predicted weights，或者按层 gate 预测计算。它可以改变缓存/耗时，不能改变真实路由、数值输出；tie-break 需固定。真实 router 若改成“先 top-k logits、再对 10 个值 softmax”，数学上可化简，但浮点顺序和归一会变化，必须另外列为数值实验。

`[HYPOTHESIS]` 更远期可以让 GPU 真实 router 查询 device resident-map，全命中时继续 MoE，只在 miss 时请求 host。这可能减少 48 次 host 往返，但牵涉 CPU 可见 miss 通知、slot version、resident-map 更新和 graph 分支，复杂度高。先从 host all-hit fast path 和静态段获益，不在第一轮实现完整 device-managed cache。

### E2. DeltaNet、HC、LM head 与启动

| 候选 | 代码依据与设计 | 限制 |
|---|---|---|
| DeltaNet recurrent state tile 驻留 | `run_linear_delta_prefill_gpu_batched_impl` 在 token loop 中先 decay/update S 再计算输出；尝试将一部分 state 在多个 token 间留在寄存器/SLM，或精确复用同一步已加载值 | 128×128 f32 每 head 为 64 KiB；每 lane 完整持有 128 个 f32 会增寄存器压力。限定小 tile，不直接套大 GRF；有 spill/设备丢失立即停该变体。 |
| DeltaNet chunkwise 矩阵算法 | 将若干 token 的 gated delta recurrence 转为块内矩阵运算，给 XMX 更多工作 | 高风险数值/状态设计；保持 16 K / 48 V GQA、sigmoid gate、conv history 和 snapshot 边界。仅保留研究项，不能直接用普通 prefix sum。 |
| HC 97 组权重专项 | 48×2 + final mixer，约 1.28 GB BF16 权重；先比较当前 down/up kernel，再试多 token HC epilogue | 已有融合和严格 rounding。96 次 grouped RMSNorm 不能改求和顺序后仍称 bit-exact；不重试普遍 SLM input caching。 |
| LM head 原生 BF16 GEMV | `lm_head_.forward` 当前保留 `[2560,248320]` oneDNN 权重；可独立试保留原始 `[vocab,hidden]` 一份并走 native GEMV | oneDNN 已有成熟实现，不能预测必胜。不得额外常驻第二份约 1.27 GB 表；不能为吞吐擅自量化 lm_head。 |
| AOT / warmup shape 覆盖 | `CMakeLists.txt:118` 仅 SYCL + per_kernel split；可加 opt-in DG2 AOT，保留 SPIR-V fallback | 目标是启动/JIT，不承诺 steady decode 提升。双目标参数需以安装编译器确认；二进制体积、构建时间一起记录，不启用已崩溃的 persistent cache。 |

技能参考已读：`A770-Kernel-Optimization/SKILL.md`、`references/weight-only-gemv.md`、`references/workflow/execution.md`、`dg2-lowbit-graph.md`、`codegen.md`。其中其他驱动/shape 的性能只用于生成候选，不转成此模型的默认阈值。若执行 ESIMD/新 XMX 几何，再读 robustness 参考并单进程短探针。

## 11. 两级缓存容量和路线边界

`[CODE]` GPU 和 host pool 目前都可以保留同一专家，两者容量不能简单相加当作唯一覆盖率。host 保留副本能避免 GPU 淘汰后的重新 mmap fill，因此直接改成 exclusive cache 也不保证更好。

`[HYPOTHESIS]` 若 S2 之后仍被 fill 限制，可在固定总内存内试“已在 GPU 且近期不需要的 host 副本优先作为 victim”，同时保护 DMA reader 和近期 GPU victim 的 host backing。先扩展已有 route simulator 加双层 residency；旧 trace 只有 acquire ids，无法还原预测到达/完成时序，不用它声称验证了异步预取策略。

本计划不增加以下工作：

- 普遍重做 LFU/SLRU/LRU-2；旧 trace 已无一致收益。新策略必须针对 host/GPU 重复覆盖或可观测 stall 提出明确变化。
- 任意增大 GPU pool、32–40 GiB pinned pool、超过 8 fill threads、全进程 immediate、`PrefetchVirtualMemory`、直接 GPU 预测预取。
- NF4 decode 全部反量化为 BF16、uint64 weight load、QMAP 私有寄存器、两输出行/SG、盲目 residual/GEMV 融合；shared/down 融合仍保持默认关闭。
- 减小 top-k、跳过专家、替换真实路由、低精度 recurrent state；这些改变模型语义，不属于默认优化。
- MTP：初始计划明确不做，本轮不引入。

M2 QSA 是大于 2048 context 的功能扩展，M3 vision、M4 shared sessions/server 仍独立排期。在 M4 前先把 S1/S2 的 scratch/任务归属写成执行槽概念，但不把本次性能计划扩大为后端迁移。Q4E 的 `count>1` 当前代表同一序列 prefill，不等于多会话 decode batch。

## 12. 最小验收与停止条件

本节是未来实施建议，不是本轮已执行操作。避免一个候选触发全 suite、多个长文本、冷缓存清理和全量 profiler。

| 改动类别 | 最少正确性证据 | 首选性能观察 |
|---|---|---|
| splice / scratch / 元数据 | 现有 Q4E cache/module 目标；LRU 和重复 ids 用例 | CPU acquire，整次 story；预期很小就不重复跑长文本 |
| S1 两 kernel prefill | 新旧 kernel 的代表 shape 与 nested/f32；一条跨 attention 边界的真实 prompt | lighthouse prefill 全 chunk；story 确认小 N 不明显退化 |
| S2/S3 async/slot | 最小 pool 强制复用，延迟 fill + DMA，部分命中、取消排空；检查内容/代次 | story 固定 48 token，总耗时、pf_wait、copy/compute overlap |
| S4 XMX / 算术变化 | 当前 kernel 对照，报告误差与 top-k/greedy 差异 | 包含 layout/dequant/gather/scatter 的整条 prefill，不只 GEMM |
| snapshot | 一条 restore/replay 状态与 logits 对比 | 完整 TTFT、64-token 边界尾延迟 |
| AOT | 一个短生成 smoke | 进程启动与 warmup，区别于 OS 模型页缓存冷热 |

仅在需要独立 kernel 收益结论时，使用现有 bench 的少量 warmup、每批至少 20 samples、3 个稳定批次，报 median 和离散度；不为每个容器清理改动跑 GPU campaign。end-to-end A/B 首轮保持同 workload / max-seq / 生成量 / 开关，沿用交接“每模式连续两次、第二次作 warm”规则，保留两次结果。只有收益接近噪声或 A/B 冲突才增加一组反序；自然冷启动若遇到则单独标记，不人为驱逐页面。

推荐复用入口（按实际改动选择一个目标，不默认全部执行）：

```powershell
ctest --test-dir build-refactor -R '^AilaQwen4ExpModuleTests$' --output-on-failure
ctest --test-dir build-refactor -R '^AilaQwen4ExpKernelTests$' --output-on-failure
pwsh perf/qwen4exp/q4e_prof.ps1 -Tag candidate_story -MaxTokens 48
pwsh perf/qwen4exp/q4e_prof.ps1 -Tag candidate_prefill -MaxTokens 24 -Messages perf/qwen4exp/prompts/lighthouse_274.json
```

脚本当前主动打开 profiling，报告结果时注明该模式；只有接近结论时补一次正常模式 smoke，以免把 profiling 边界收益误当作生产收益。增加实验开关时通过脚本 `-EnvVars` 显式传入；本文所有新开关名都是建议，尚无实现。

停止/回退规则：

1. 小阶段更快但完整 TTFT/生成总耗时更差，先保留实验开关，不升为默认；不持续扫参数寻找偶然最快值。
2. 改变算术的路线若不能达到当前严格等价要求，保留实验并说明取舍；不自动放宽模型的验收标准。
3. 新 memory/slot 设计出现代次错误、未完成拷贝被覆盖或异常未排空，先修该局部生命周期；不叠加优化来掩盖。
4. 对候选的一轮小规模 A/B 若无清晰收益，记录结果并转下一个优先项；全套回归只在准备合入跨模型公共算子时按实际影响扩大。

## 13. 下一次开发任务可直接采用的拆分

1. **S0a：热路径去分配。** 改 `Qwen4ExpExpertCache.{hpp,cpp}` 和后端预分配，保持算法、开关默认值与 trace 格式。
2. **S1：批量 MoE rows。** 改 `MoeExpertOps.{hpp,cpp}`、backend buffers/dispatch，补一个 new-vs-existing kernel 用例；默认先关闭。
3. **S2a：slot ready / in-flight 状态。** 先保护 host DMA source 与 GPU reader，加入小 pool 定向用例，再拆整批 wait。
4. **S2b/S3：demand-first 与 route event。** 不同时更改淘汰策略；以一个 story A/B 判断是否减少关键路径等待。
5. **S4 或 S5 二选一。** prefill MoE 仍显著占比则做 expert-major；TTFT/checkpoint 占比更突出则做 snapshot 批量回读。避免两条大改同时落地而无法归因。

每项交付都应包含一个可回退实现、一个与改动相称的正确性证据和一份简短结果说明。静态代码已足以支持上述第一轮工作；无需先重新跑交接文档的完整验证流程。
