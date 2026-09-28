# Qwen3.8-Flash-Next / A770：P2 之后的异构执行与数据路径设计

日期：2026-09-27。代码基线：`f24dd34e735c38d95009ce909fd85e38335fb58e`，开始/结束勘察时 HEAD 相同。面向 Windows、A770、BNB NF4；本轮没有构建、GPU benchmark、运行模型或修改运行时代码。

**下一阶段建议以“按实际就绪时间分配专家计算”为主线：GPU 处理常驻/高复用专家，CPU 处理少量 host-ready、GPU-miss 专家；同时把现有批量 acquire 拆成准备、传输、计算三个可衔接的阶段。** CPU 参与是合理候选，但不是所有 miss 的默认归宿。应比较整层完成时间，不能仅凭少搬了多少权重判断成功。

本文接续 [reference brainstorm](qwen3.8-flash-next-reference-brainstorm-2026-09-27.md) 的 P0–P2。标记：`[CODE]` 当前源码；`[RECORDED]` 用户/既有交接记录，非本轮复测；`[HOST]` 本轮只读硬件查询；`[DERIVED]` 尺寸与依赖推导；`[DESIGN]` 待实施假设；`[DOC]` 官方 API 文档。后续各 H 阶段是新编号，不覆盖原 P0–P5 的历史含义。

## 1. 先纠正基线，再决定优化对象

**2026-09-28 CPU 数据通路续验：** 批量准备、合并回传和等待所选专家预取完成的组合已完成三组实模配对：旧 CPU 配置总时延中位数 9258.08 ms，候选 8733.11 ms（降低 5.67%），六次黄金 token 均一致；候选三次均执行 26400 个 CPU 任务（含预热），并通过完整 logits、多轮复用和实际回滚验收。同期 GPU-only 为 5695.54 ms，强制 CPU 放置仍明显更慢。未等待预取的动态混合配置复测出现 token 偏差，不能计为有效加速；三个实验开关均保持默认关闭。详见 [CPU 数据通路实验](qwen3.8-flash-next-cpu-dataflow-2026-09-28.md)。

**2026-09-28 大规模 CPU 实验更新：** 按用户要求，不再限制于单专家/4 线程：CPU arena 放宽到 12 线程，执行槽支持 40 个 expert/token 任务，并增加固定配额、主动放置和 2..4-token prefill 实验入口。跨专家批量计算及输入预排布/寄存器 NF4 查表已实现；12 线程下 40 任务微基准由 20.45 降到 4.13 ms，保持原 CPU GPU-order 算术逐位一致。真实 40 任务放置也已跑通，但不能把 CPU 算子收益视为超越 GPU-only。完整矩阵、数值限制与最终复测见 [12 线程与大批专家实验](qwen3.8-flash-next-cpu-scale12-2026-09-28.md)。以下“暂停扩大”及少量分流建议保留为此前阶段记录，不再作为实验规模限制。

**2026-09-28 后续实施更新：** H1/H4/H5 的并发统计、DMA 计时和 ready-slot 生命周期已补强；GPU ready-first 已独立接入但默认关闭。修正 CPU 单配额永久停用后，真实 CPU 分流在 story 第 26 步出现 token 偏差，暂停扩大 CPU 份额，先做算子级数值对齐。direct prefill I/O 已完成实现与正确性验收：三组 story 对照总时延中位数改善约 0.49%，但与 MoE rows 组合未显示稳定叠加收益，保持默认关闭。最终 11 项聚焦测试、独立及组合真实模型验收通过，13 次短故事测试均与黄金 token 一致。详见 [异构后续进展](qwen3.8-flash-next-heterogeneous-progress-2026-09-28.md)。以下仍保留原始设计分析，不代表所有 H 阶段均已完成。

`[RECORDED + CODE]` 此前约 1.2 秒回退已定位为 QSA 首次编译未被 warmup 覆盖。A2 K-only 使 2-token warmup 不再走原 QSA cache kernel，也未组成完整 4-token block；首个真实 prefill 才触达相关编译工作。`f24dd34` 在 load 末尾用 dummy block 调用 `qsa_compress_key_blocks` 并完成同步。提交记录报告首段 prefill 3376→2217 ms、GenerateMessages 7403→6241 ms、打印的 48-token 回复不变。数字来自既有记录，本轮没有复现。

因此：

- 废止“约 1 秒是 B 系列调度或 K-only 算子稳态变慢”的推测；不再为该回退做逐提交消融。
- 交接文档 §3.3 已写明解决，但 §3.1–3.2 与 §4 仍有旧归因和“消除 router_sync 找回约 380 ms”的行动项，**后者不再适用**。host 等待点的计时不是可独立回收的 GPU 时间。
- `CMakeLists.txt:119` 已指定 `-fsycl-device-code-split=per_kernel`。事故记录中的“整个 cpp 模块 JIT”是现场解释，不能上升为所有构建均按翻译单元编译的规则；dummy prime 的有效性也不等于所有 variant、oneDNN shape 都已预热。
- P0/P1/P2 的实现完成与既有测试可以作为下一阶段起点；“打印文本相同”不等于内部张量 bitwise 一致。异构执行尤其需要明确这个边界。

### Opus 的四个问题：现状与设计判断

| 问题 | 当前可确认的内容 | 下一步判断 |
|---|---|---|
| 首段 acquire 1562 ms，fill 1059、pf_wait 438，约 10.6 GB | `[RECORDED]` 用户提供；`fill_ns` 范围包含 host slot 分配、任务调度、memcpy/scale 处理和提交等 | mmap fault 是候选，尚非根因。日志 xfer 统计的是 H2D slot 字节，不能直接当 mmap 实读字节或 SSD 吞吐 |
| host 已热时 7.1 GB / 1.1 s | `[RECORDED]` 约 6.5 GB/s 的路径吞吐 | 先分离 host enqueue、DMA、GPU 等待；不能直接认定是 PCIe 物理链路带宽 |
| 只做 2-token warmup，decode 首步 110 vs 75 ms | `[CODE]` EngineLoad 确实没有接一个 count=1 forward | 补 dispatch 覆盖值得先做；35 ms 差额可能含 cache miss 等，不能全部叫 JIT |
| prefill 26+4 | `[CODE]` 稳定聊天前缀产生 recurrent snapshot anchor，forward helper 在 anchor 切开 | 原因已找到。涉及回滚复用语义；第二段 370 ms 不全是可消除开销 |

`[HOST]` 本机 CIM 返回 Intel Xeon E5-2696 v3，18 cores / 18 logical processors，物理 RAM 68,622,077,952 B（约 63.9 GiB）；当前运行环境的 .NET ISA 检查返回 AVX2=true、FMA=true、AVX512F=false。CPU 原型按 AVX2/FMA 设计，不依赖 AVX-512/AMX/BF16 点积。没有测 DRAM 带宽、NUMA 实际访问代价或 CPU 持续算力。

本轮 PnP 返回的 A770 link speed/width 原始值均为 1（含 Max），与上述吞吐的字面解释明显冲突，不能用于链路定性。后续如需要，取可靠设备工具在负载时的 negotiated link 与 root-port 信息；本轮没有把它认定为 x1 或 PCIe Gen4 x16。

## 2. 现有路径的关键限制

`[CODE]` 当前并非完全串行的“全部读完才 DMA”：`ExpertFillPool` 分 part 填充，最后一个 part 完成后即可往独立 `copy_q_` 提交该 expert 的 memcpy。copy queue 是 in-order + immediate command list，compute queue 通过 event 依赖它。

但 `acquire()` 仍有两个较粗的边界：

1. 同一调用的 demand fill 经过 `fill_pool_->run()` 汇合后，调用者才能继续安排 routed expert compute。
2. 末尾 `copy_q_->ext_oneapi_submit_barrier()` 代表该批之前的所有 DMA；多个 GPU/host slot 共用这个完成 fence。baseline compute 因此难以只等待自己需要的 ready subset。

B6 已有 opt-in demand microbatch，不能重复宣称从零发明流水线；下一步应复用其 expert/token 映射，减少“该批最后一个 miss”对 ready expert 的阻塞。CPU 分流与此是互补关系，即使 CPU 原型无收益，ready-first GPU 调度仍可以独立保留。

现有 `contains()` 只说明 GPU key 有映射，`host_contains()` 也不保证 host Ready。异构 planner 必须查询状态、generation 和完成事件，不能把二者直接当成可读判据。

## 3. 为什么按专家划分 CPU/GPU，而不是按层划分

保持 attention、GDN recurrent state、GR、shared expert、router 和 LM head 在 GPU。初版仅分配真实 Top-K 中的 routed experts，不移动 recurrent state，不改变路由、不做跨 token 预测计算、不改量化格式。

`[DERIVED]` 当前一个专家 H=2560、I=640、gate/up/down 共 `3HI=4,915,200` 个权重。NF4 packed 为 2,457,600 B，加 block=64 的 u8 scale codes 76,800 B，总 slot 为 **2,534,400 B**（共享 nested metadata 另计）。一次 expert/token 的矩阵乘加约 `6HI=9,830,400 FLOP`，不含 NF4 解码、SiLU 和归约。

CPU 分流一个 expert/token 时，D2H 输入为 H 个 BF16，即 5120 B；H2D 返回 H 个 FP32 expert output，为 10,240 B。输入可被同 token 的多个 CPU experts 共享。相对搬 2.53 MB 权重，单 expert 的输入加结果仅约 15 KB，约小 165 倍；**这只是传输字节比，不是速度提升倍数**。

若暂把 6.5×10^9 B/s 作为路径有效吞吐，单 slot 约需 0.39 ms。这给出 CPU expert 的候选时间尺度，却不是“CPU 必须小于 0.39 ms”的硬规则：CPU 可以藏在 GPU 已有工作中，也可能被 D2H、线程唤醒、NF4 展开或内存竞争拖慢。

对于同一专家处理 m 个 token，权重 H2D 可以摊销，CPU 返回结果却随 m 增长；因此 prefill 中低复用专家更值得尝试 CPU，高复用专家更偏向 GPU。decode m=1 是清晰的起点，但必须设很小的 CPU 工作量上限，避免每层 CPU 尾部累积。

| 专家状态 | 初始策略候选 | 理由 |
|---|---|---|
| GPU Ready | GPU | 无需搬权重，通常不值得额外 D2H |
| GPU Copying，接近完成 | 等其 event 后 GPU | 已付出的传输应计为 sunk cost，不重复计算/搬运 |
| GPU miss，host Ready，少量 token | CPU 或 H2D→GPU，由完成时间估计选择 | 初版异构最有价值且最容易控制的集合 |
| GPU miss，host Filling | 默认沿原 GPU 路径，后续才允许 Ready 后选择 | CPU 同样不能读取未完成数据 |
| host miss、文件页冷 | 首版不交 CPU | CPU 不能消除读文件等待，还可能拖慢 GPU 的填充线程 |

llama.cpp 本地 `common/arg.cpp:2756` 的 `--cpu-moe` / `--n-cpu-moe` 展示的是 CPU 权重/层放置控制，可借鉴部署选择；它不是本方案的同层、按 miss 动态分流实现，也不证明 BNB NF4 CPU kernel 已存在。

## 4. H1：统一的 expert readiness 接口

`[DESIGN]` 保留 `Qwen4ExpBnb4Backend` 和现有公开 ABI；内部新增 `Qwen4ExpExpertExecutor`，把当前 cache 的三个职责明确拆开：

```cpp
// 接口草图，不是当前代码已有的 API。
ExpertResidency QueryResidency(ExpertKey); // Ready / Filling / Copying + generations
HostExpertLease AcquireHostReady(ExpertKey); // 不触发 GPU 分配；包含 CPU reader lease
FillTicket PrepareHost(ExpertKey, Priority);
GpuExpertTicket EnsureGpu(HostExpertLease, Priority); // 返回该 expert/sub-batch 的事件
LayerMoePlan PlanExperts(RouteBatch, ResidencyView, ResourceBudget);
```

`LayerMoePlan` 包含原始 token/rank、expert ID、route weight、CPU/GPU mask、generation、结果索引和有界任务数量。planner 位于真实路由就绪之后，只执行纯 CPU 控制计算；GPU 执行 slot 内预分配 route map、result arena 和输入 staging，避免每层创建 vector/future。

### host slot 必须新增 CPU reader ownership

现有 `HostSlotRecord::source_reader_fence` 只表达 DMA 读者。CPU 将持续读完整权重，因此需要独立 `cpu_readers`/lease 与 generation 检查；只有 fill writer 退出、CPU readers 为零、DMA readers 完成时才允许覆盖。generation 是防陈旧发布的辅助，**不能替代对写入和覆盖的排他保护**。

不要把 cached `MoeExpertSlotDev` 指针传给 CPU：nested metadata 中含 device 指针。增加明确的 `MoeExpertSlotHost`，引用 host packed weight、scale codes、layer-level nested absmax/qmap/offset，并持有其 model lifetime。GPU/CPU 分别读同一份不可变权重语义，不依赖 USM device pointer 可在主机解引用。

优先用现有 host pool 作 CPU 权重来源，避免首版再复制一套全模型 CPU cache。CPU 读频率应更新 host cache 热度，但不能让低收益 CPU 任务长期锁住大批 slot。任务 admission 先取得 bounded credit，再持有 lease，不让排队任务无限占内存。

取消/异常时停止新任务，已读 host 的 CPU job、已提交 H2D/D2H 和 GPU consumer 都必须 drain 后才能重用 scratch/slot。缓存原有 poison/failure contract 保留；未完成结果不能提交到下一层。future、event 或 completion mask 都按 execution generation 隔离。

## 5. H2：CPU NF4 原型与有序汇合

**2026-09-28 数值对齐更新：** 已修复 CPU nested scale 的融合乘加舍入差异，并新增默认关闭的 `AILA_Q4E_CPU_GPU_ORDER=1` 候选，以 GPU lane 顺序计算并显式执行非线性除法。16 个真实专家样本的中间值/down 对照、三次黄金 48-token 生成及多轮回滚验收通过；但端到端负性能筛选仍慢约 6%，不升默认、不扩大 CPU 配额。原 Auto CPU 路径仅修 scale 后仍有 token 偏差。详见 [CPU 对齐记录](qwen3.8-flash-next-cpu-alignment-2026-09-28.md)。

### 先实现小而专用的 CPU kernel

`[DESIGN]` 首个实现只支持本模型固定 H/I、BF16 activation、NF4 block=64 和当前 nested scale 格式；scalar 路径作 reference，AVX2/FMA 路径按输出行并行。

- 按 GPU 当前高 nibble→低 nibble 的顺序解析 packed bytes，使用实际 16-entry NF4 LUT；不能当线性 int4 用 VNNI dot。
- scale 解码遵循 `nested_qmap[code] * nested_absmax + offset` 及实际 block 对应关系。冷码/热 scale 均不改变计算语义。
- gate/up 使用 FP32 accumulate，SiLU 后的中间 h 按现有边界舍入 BF16，再做 down；返回未加权的 FP32 expert output。
- BF16 activation 可在任务入口展开为小块 FP32；权重按 tile 解码并立即计算，不将全部模型或每个 expert 常驻展开 FP32。每专家 FP32 权重约 19.66 MB，会抵消带宽/缓存优势。
- 多 token 时以同专家的小 activation tile 复用解码；初版只选一个小 m，不做线程数、tile、ISA 的大范围 sweep。oneDNN CPU/SGEMM 可作后续对照，但须把展开和 packing 成本计入。

### CPU/GPU 合作的执行顺序

1. GPU 产生本层 MoE 输入 x 和真实 route；安排有依赖的 x D2H，并正常发射与其独立的 shared expert / GPU-ready 专家工作。
2. host 拿到 route 后建立唯一 plan。host-ready CPU experts 持有 lease，x D2H 完成才启动点积；剩余 experts 继续异步 fill/H2D。
3. GPU 按 ready subset 计算，不依赖整批末尾 DMA。不要把等待冷数据的 kernel 先放在 in-order compute queue 头部，再把 ready work 排在后面。
4. CPU 输出放入固定 pinned result slot，再提交小 H2D；GPU expert 结果写入同一逻辑 `[token, original_rank, H]` 的 FP32 arena。
5. 所有必要 rank Ready 后，GPU 按**原 Top-K rank 顺序**逐项乘 route weight、累加，保持 routed BF16 舍入后再加 shared output/gate 的现有顺序，最后进入 GR。

初版使用 host coordinator 在已提交 GPU 分支后等待 CPU completion，再提交结果 H2D 和 merge 即可；不把等待 future 的 `host_task` 放进 compute queue，也不忙等抢占 fill 线程。结果 H2D 和输入 D2H 是短 deadline 数据，不应排在无关的下一层大权重 DMA 后。是否独立 activation copy queue 取决于实际 queue/engine 行为；多建 queue 不会增加 PCIe 物理带宽。

### 当前 GPU epilogue 不能原封不动复用

`MoeExpertOps.cpp` 的 decode down 当前循环所有 rank，直接输出合并后的 BF16，并可融合 shared expert。若把 CPU 总和事后加到此 BF16 结果，会改变求和顺序与舍入位置。异构路径需新增 `down_to_rank_f32` 和独立 ordered merge；GPU-only 的老路径保留，避免无分流时白白增加 scratch/launch。

`[DERIVED]` m=26、K=10 时完整 rank arena 为 `26×10×2560×4=2,662,400 B`，约 2.54 MiB；m=1 为 100 KiB。另有 BF16输入、CPU中间值、pinned输出及mask预算，均按 execution slot 计入 admission。首版只为实际 CPU rank 传回结果，避免把整个 arena 经 PCIe 往返。

这种汇合避免额外的跨设备乱序相加，但**不保证 CPU/GPU bitwise 相同**：AVX2 与 SYCL subgroup reduction、FMA 次序和 native exp 仍可能不同。先对同一权重/x 比较 gate、h、down、merge，再看一次完整生成。若项目坚持完全相同 token 输出，未满足时保持 hetero opt-in；不得以“NF4 格式没变”宣称数值无损。

## 6. H3：以整层完成时间为目标的调度

**2026-09-28 异步重叠更新：** 已增加默认关闭的 `AILA_Q4E_CPU_OVERLAP`、单任务槽常驻 worker 和 1–4 线程局部预算，实现 CPU 与 acquire/fill 重叠。修正异步汇合后的节流误判后，三组黄金 story 对照通过，但总时延中位数仍比 GPU-only 慢 3.21%，不升默认。初版大量 CPU 分流再次出现 token 偏差，通用数值一致性仍未解决。详见 [CPU 重叠记录](qwen3.8-flash-next-cpu-overlap-2026-09-28.md)。

`[DESIGN]` 初版只按真实 residency、m、CPU 队列长度和传输 backlog 作确定性选择，不训练 predictor、不依赖上一步 Top-K 猜测计算。

对一个候选集合 C，估计：

```text
T_cpu(C) = activation_D2H_ready + cpu_queue_wait + NF4_expert_compute(C) + result_H2D
T_gpu(G) = GPU queue 上 shared / ready experts / remaining fill+DMA+compute 的预计完成时间
T_layer  = max(T_cpu(C), T_gpu(G)) + ordered_merge
```

这里的项是依赖路径上的时间，不把重叠的 fill/DMA/GPU 时间简单相加。CPU 方案必须与原 GPU-only 完整路径比较，计入额外 launch、arena 写读、cache 驻留机会与 host 内存竞争，而非只比较一个 expert 点积。

首个可用策略：CPU 只接 host-ready 且 GPU-miss 的少量低 m experts；CPU credits 满或收益估计不明显，走 GPU。用少量既有/原型样本给出保守估计，给 CPU 完成 deadline 留余量；候选阈值不是已验证 dispatch。记录 CPU straggler 暴露时间，发现尾部变长就收缩份额。

本机 18 个逻辑处理器，已有填充池默认 8 线程且 caller 会参与，PLE 也有后台工作。不能另起 18 线程 CPU GEMM 再让 oneDNN 内部开满线程。统一预算划分 compute credits、fill credits 和提交/调度余量；可先试少量 CPU workers，但不把某个线程数写成通用最优。CPU compute、fill 和 DMA 都读写 host DRAM，必须看一起运行后的吞吐。

每个任务的 CPU/GPU owner 初版只选一次。暂不做超时后另一设备重复计算的竞速：这会增加读者、重复传输和“谁发布结果”的复杂度。下一 token/下一层再调整策略即可。

## 7. H4：mmap / ReadFile / host cache 的数据入口

### 先区分四种“冷”

进程 JIT 冷、OS file-page 冷、host expert-pool 冷、GPU expert-cache 冷是不同状态。10.6 GB 的日志 xfer 不能证明真的从 NVMe 读了 10.6 GB；页可能已在 standby cache，fill 还包含内存复制、任务唤醒和部分 scale 解码。

`[CODE]` `MemoryMappedFile.cpp` 使用 `CreateFileA(FILE_ATTRIBUTE_NORMAL)`、`CreateFileMappingA`、`MapViewOfFile`。`fill_expert_matrix()` 在 compact nested 模式主要执行 packed bytes 与 scale-code 的 memcpy，不是把所有 NF4 权重反量化到 CPU。一个 expert 有 gate/up packed、gate/up codes、down packed、down codes 四段来源。

### 可替换的 ExpertReader，而非一次重写模型加载器

1. **保留 mmap reader。** 对真实即将需求的文件区间做有界 `PrefetchVirtualMemory` 是一个更小的候选。该 API 可把预知范围组织为较大的并发 I/O，不能保证页持续驻留或消除后续 fault；不要预取全模型挤掉正在服务的页。[Microsoft 文档](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-prefetchvirtualmemory)
2. **buffered overlapped ReadFile reader。** 用预开的 shard handles、固定 `OVERLAPPED`/completion records，把上面四段分别读到 host slot 的目标 offsets；所有片段完成且字节数正确后才发布 Ready。它可以省掉应用显式 mmap→slot memcpy，但不保证绕过 Windows cache 或内部复制，更不是 SSD→GPU 零拷贝。[ReadFile 文档](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-readfile)
3. **最后才考虑 unbuffered。** `FILE_FLAG_NO_BUFFERING` 要处理文件 offset、长度和 buffer 对齐；safetensors 的 `8+header_size+tensor_offset` 及 scale 小段不天然满足扇区要求。需要 aligned read+bounce，或者离线建立按专家打包的文件，不能直接加 flag。[File Buffering 文档](https://learn.microsoft.com/en-us/windows/win32/fileio/file-buffering)

给 `HostTensorView` 的来源信息补一个持有 shard lifetime 的 `FileSpan{shard_id, absolute_offset, bytes}`；现有 `HostTensorEntry` 已有 shard index 和 tensor-relative offset，绝对 offset 必须包含 safetensors 的 8-byte header length 与 header payload。不要用当前裸指针地址反推一个不再受所有权保护的文件位置。

异步 ReadFile 返回 pending 不是失败；每次请求有唯一 completion 记录，必须处理短读、EOF、取消后完成和 slot generation。`CancelIoEx` 请求取消不等于立即可以释放接收 buffer；仅在终态完成、其他读者排空后复用。不要占着整个 fill pool 同步等待文件 I/O，避免 CPU compute 等 Ready、fill 又等空闲 compute worker 的循环等待。

可选后续实验：导出版本化 `expert-pack` sidecar，将四段合为一个对齐 record，共享 nested metadata 单存。保留 NF4 逐字节语义、checkpoint fingerprint、schema/endian/offset/校验和；只在加载时准备，热路径不转码。它改变磁盘排列而非 GPU GEMV 布局，不触碰已知有害的 blocked GPU weight layout。须计入生成时间、额外磁盘空间和旧 sidecar 失效，不作为 H1 的前置条件。

host pool 目前约 16 GiB，在约 64 GiB RAM 上还要给 mmap 页、PLE、OS 和 CPU scratch 留空间。初版复用 pool，不扩大到全模型 pinned，也不要求管理员清空 standby list 来制造冷启动。先做一组小的真实专家区间的首次/再次读取对照即可。

## 8. H5：H2D 6.5 GB/s 的可执行诊断与优化

先保留“应用路径吞吐约 6.5 GB/s”的描述。日志 MB 实际多处用 1024² 作除数，最终带宽统一保存 bytes/ns 再决定 GB/s 或 GiB/s，避免再混单位。

只增加能够决定下一步的少量时间：host submit 首/尾、copy event start/end（若 queue 开启 profiling）、批完成、首个 consumer 开始，以及 memcpy 次数/尺寸。不把多个重叠 event duration 累加后冒充整批 wall time，也不把等 compute 的尾部算成纯 DMA。

候选顺序：

- 先确认 preallocated host source 确实来自 `sycl::malloc_host`、source 已写入且处于 Ready、没有首次缺页/填充混入；当前 Context 已用 malloc_host，不要重新“引入 pinned”作为优化。
- 现有 slot 约 2.53 MB，一次 memcpy 并不很小；先判断问题是否为提交间隙、逐专家散布、copy/compute 依赖或实际链路，而非直接加更多 copy queues。
- `acquire` 目前对整批赋同一末尾 fence。按小批/专家给真实完成事件，可提前消费、提前释放 host readers，避免所有 lease 活到最慢项；需控制事件管理成本，不能把 32 个 part 都变成独立 fence。
- 能同时满足 source 与 destination 连续的多个 slot 才可合并 DMA；任意 scatter 不能仅因 ID 相邻就 memcpy 大块。为合并而额外 host pack 或改变 GPU placement，须计入成本及淘汰代价。
- 采用小而有界的待传字节窗口，ready expert 先算、demand 先于 speculative，activation/result 小传输有及时完成的机会。H1 的 CPU 输出不能排在很长的权重预取队列尾部。

若诊断确实需要底层上限，只做两个代表：一个较大连续 copy、一个当前真实 slot pattern，均用相同 pinned pool；不铺开尺寸/线程/队列大矩阵。即使较大连续 copy 很快，也不能把该数直接套给随机 slot / WDDM / 并发 CPU 场景。

## 9. H0：warmup 从 token 数改为 dispatch 覆盖

`[DESIGN]` 为当前 backend 建一个小的 warmup manifest：覆盖所选模式的 prefill、count=1 decode、QSA 完整 block、实际需要的 sparse selection/attention variant，以及 CPU hetero 路径准备。manifest 记录“触达了什么”，不依赖随便两个 token 恰好路由到全部关键代码。

短模式最低改动：保留 `f24dd34` 的 load-time prime；初始化 prefill 后再做一个 count=1 forward/采样，完成后统一 reset session、PLE history 和 snapshots。稀疏路径可用合成算子 prime，不必为了覆盖 JIT 跑 2052-token 全模型。oneDNN primitive shape/cache 与 GPU kernel 编译分开记录，count=2 不能自动覆盖 count=26 的 primitive 行为。

同时报告三种成本：load+warmup、首请求 TTFT/总时延、稳态 decode。把 JIT 移到 load 改善首请求，但不会自动缩短进程启动到首 token。

**先试持久编译缓存，再决定 AOT。** Intel runtime 文档提供 `SYCL_CACHE_PERSISTENT` 与 `SYCL_CACHE_DIR`，可作为不改全项目构建的候选；先核对部署 runtime 是否命中，路径/版本由应用管理，不能只设变量就宣布 20 秒消失。[Intel runtime 文档](https://intel.github.io/llvm/EnvironmentVariables.html)

AOT 单独建实验 target/preset：`-fsycl-targets=spir64_gen` 配合部署 DG2 的有效 OCLOC device 选项，编译与最终 device link 需一致；先针对 Q4E kernel 子目标和相关链接依赖评估，不直接改全局 `AilaSyclOptions` 让所有模型/测试一起变更。是否可混合 JIT/AOT 由该工具链小目标验证；本轮不提供未经编译验证的可直接执行命令。[Intel AOT 文档](https://www.intel.com/content/www/us/en/developer/articles/technical/ahead-of-time-compilation.html)

AOT 不会消除 oneDNN 运行期 primitive 准备、host 文件访问、专家 cache warming 或全部 20 秒初始化；也不承诺稳态 kernel 更快。现有 A770 skill 对部分简单 kernel 的 AOT 测量并不显示普遍收益。

## 10. H6：26+4 的来源与三档优化

**2026-09-28 实施更新：H6a/b 已完成并验证。** H6a 提交为 `ac5e7df`；H6b 保留 anchor、跳过非末段 logits，并预分配 token staging 和快照槽位。真实模型完整 logits 位模式、两轮复用、修改历史后的实际回滚及 48-token 黄金基线均通过。详见 [H6 验收记录](qwen3.8-flash-next-h6-results-2026-09-28.md)。下述来源分析描述实施前状态；H6c 仍为后续候选。

`[CODE]` 路径已确认：

1. `EngineGeneration.cpp:536–555` 重新渲染不带 generation prompt 的稳定聊天前缀，并可能加上 assistant header，形成 `recurrent_prefill_anchor`。
2. `:863–868` 把 anchor 传给 `forward_prompt_tokens`。
3. `:77–80` 强制 `next=snapshot_after_len`，因此 30-token prompt 被拆在稳定前缀处；默认 chunk limit 是 512，**26 不是 GPU tile 或固定 prefill 容量**。
4. Q4E `ForwardImpl` 每段都执行 LM head、wait_prefetch、CaptureSnapshot。`CaptureSnapshot` 对 tokenCount>1 默认捕获，同步并将 dirty recurrent/conv state D2H，再复制进 snapshot。由此得到 anchor=26 的可回滚状态。

两段 prefill 的真正额外成本包括重复遍历 48 层的固定提交/准备、重复尾部 LM head、可能重复专家搬运，以及多一个 snapshot；后 4 token 的 attention/GDN/MoE 计算在 count=30 合并后仍然需要。不能许诺完整省掉 370 ms。

**H6a：显式 one-shot 模式，最快可交付。** 增加请求级 `snapshot_policy=none|reusable_prefix` 和 `need_logits`，由调用方明确生命周期；仅对承诺无需多轮增量复用的一次性生成跳过 anchor。默认聊天保持原契约，不因 story 是短请求就偷偷关闭 snapshot。注意仅设 backend maxSnapshots=0 还不够，engine 仍会在 anchor 切分，两个层次要一致。

**H6b：保留 anchor，先去掉被丢弃的工作。** 非末段 forward 不计算 LM head/logits；同时将 snapshot destination、token input staging 预分配、减少当前 forward helper 的每段 alloc/free。这不改变 26+4 的模型语义，收益有界；不能跳过必要 recurrent-state checkpoint。

**H6c：单次全长 prefill 内捕获中间状态，较大设计。** 将 `CheckpointAt(position)` 传入 backend，使每层 GDN token loop 在 anchor 处保存 S、conv history/head，PLE 保存该位置的短历史、QSA 保存该边界的 valid counts/raw tail。其余 attention/GR/MoE 仍按完整 30 token batch 执行，所有层 checkpoint 完成后原子发布该长度。不是简单把最终 recurrent state 标成 26：递推不可倒推，必须在算到那个位置时捕获。

H6c 需要 GDN/conv 支持中途 snapshot，且各层位置一致；这是原 P3 state-shard 驻留优化的自然结合点，但先不耦合到 H2 原型。没有空间长期持有完整额外 GPU state 时可分层 D2H 到预分配 host snapshot；需要依赖保证 source 在拷贝完前不覆盖。只在其固定开销确实值得时推进。

**2026-09-28 H6c 实验更新：** 已加入默认关闭的短文本原型（AILA_Q4E_INLINE_CHECKPOINT=1）。三组交替 story 对照中，总生成计时中位数 5663→5418 ms（约 -4.33%），六次均与黄金 48-token ID 一致；常规锚点 26/28 和多轮复用/回滚通过。锚点 2/16 的分段参考 logits 存在数值差异，因此尚未满足通用 H6c 默认启用门槛，不推断安全长度阈值。实现范围、限制和原始测量见 [H6c 实验记录](qwen3.8-flash-next-h6c-results-2026-09-28.md)。P3/P4/AOT 仍待后续推进。

## 11. 推荐交付顺序、停止条件与最少验收

| 阶段 | 可独立评审的产物 | 最小验证/决策，不要求本轮执行 |
|---|---|---|
| H0 | warmup dispatch 清单、补 decode；给旧回退结论加已解决说明 | 一次启动+首请求，区分初始化和请求内首次编译；不重新全量回归 |
| H1 | Ready/Fill/DMA 分离的 expert tickets、CPU host-reader lease、GPU-ready subset 路径 | 一条混合 Ready/Copying 的小用例，加取消/slot 重用生命周期检查 |
| H2a | 固定 shape 的 CPU NF4 expert scalar/AVX2 原型 | 一个真实 expert，m=1 和一个小 m；比较 h/down；计 CPU compute 和资源占用 |
| H2b | 少量 host-ready miss 分流 + ordered rank merge | 一层 CPU/GPU 混合与原路径对照，再一条 story；若额外 merge/CPU 尾部抵消收益，保留 opt-in 或停止扩大 |
| H3 | 有界 CPU credits、简单完成时间策略、CPU straggler 指标 | 同一 story 少量配对；只看完整 TTFT/总时延/稳态，而非“CPU 利用率上升” |
| H4 | mmap reader 与 buffered ReadFile reader 可替换 | 同一小组 expert 首次/再次读取；仅在证据支持时加预取或 expert-pack |
| H5 | 分阶段 copy timing、细化 consumer event/提交窗口 | 一个真实 slot pattern；纯 copy 上限只在分解仍不清楚时补一个 |
| H6a/b | one-shot snapshot policy 与非末段 skip-logits | one-shot 一条短输入；reusable-prefix 一次两轮复用/回滚对照 |
| 后续 | H6c + 原 P3 GDN 中间状态；原 P4 raw K ring；AOT | 被实际剩余瓶颈触发时再进入，避免与异构主线同时重写状态契约 |

实施时 H2a 可以先作为独立小原型验证 CPU 是否值得，再投入完整 H1/H2b 接入；I/O reader 和计算分流分别提交，避免一次变更两个根本变量。性能无收益就不升默认；正确性问题先收敛，不扩大 shape 或线程组合。

原 reference brainstorm 的 P5（B=513 complement）已被 P2 实现吸收，不再重复排期。FP8、GDN 降精度、近似路由、MTP 不属于这轮异构 MVP。

最小新增运行统计只保留：每层 CPU/GPU expert/activation 数，saved H2D bytes，CPU queue/compute/暴露尾部，activation/result transfer bytes，GPU idle/等待的可解释区间，pinned/scratch 峰值。相同 token 数、warmup 覆盖、有效环境与线程预算下才比较，避免再次将初始化遗漏解释为结构性退化。

## 12. 接手源码地图

| 落点 | 阅读入口 | 设计关联 |
|---|---|---|
| load-time QSA prime | `src/models/Qwen4ExpBnb4Backend.cpp:627` | H0，已有修复保留 |
| 主机初始化 warmup | `src/engine/EngineLoad.cpp:593` | H0，补 count=1 与独立状态恢复 |
| 稳定前缀 anchor / 切片 | `src/engine/EngineGeneration.cpp:57`、`:536`、`:863` | H6，按请求明确复用策略 |
| routed MoE 分派 | `src/models/Qwen4ExpBnb4Backend.cpp` 的 actual_route_event、batch_acquire、B6 分支 | H1/H2/H3，保持 shared/GR 边界 |
| host/GPU cache ownership | `src/models/Qwen4ExpExpertCache.hpp:24`、`:34`、`:138` | H1，状态查询及 CPU reader lease |
| fill/DMA 汇合 | `src/models/Qwen4ExpExpertCache.cpp:1305`、`:1446`、`:1527` | H1/H5，子集就绪与批尾 fence |
| 原始四段 memcpy | `src/models/Qwen4ExpExpertCache.cpp:1745`、`:1800` | H2 host 视图、H4 reader |
| gate/SwiGLU/down/舍入 | `src/ops/MoeExpertOps.cpp:502` 及 `moe_decode_down_weighted_accumulate` 路径 | H2，FP32 rank output 与 ordered merge |
| shard 来源与 mmap | `src/utils/SafeTensors.hpp:73`、`src/utils/SafeTensors.cpp:901`、`src/utils/MemoryMappedFile.cpp:9` | H4，安全的 FileSpan/IO ownership |
| forward 尾部 / snapshot | `src/models/Qwen4ExpBnb4Backend.cpp:1993`、`:2076` | H6，skip-logits / 中间 checkpoint |
| 构建 SYCL flags | `CMakeLists.txt:116` | H0/AOT，避免全局无差别修改 |

建议新增内部文件为 `Qwen4ExpExpertExecutor.{hpp,cpp}`、`Qwen4ExpCpuExpertOps.{hpp,cpp}` 和局部 ExpertReader 实现；先保持模型特化，不为一个 backend 新造全项目通用任务图框架。当前设计的价值在于消除不必要的数据等待、降低必须迁移的权重总量，并保留可回退的 GPU-only 路径。
