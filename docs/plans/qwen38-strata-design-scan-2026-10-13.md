# Strata 设计扫描（只读调研）：可搬到 Aila 的机制候选清单

- 日期：2026-10-13
- Strata 仓库：`tmp/Strata`（独立 git 仓库，HEAD `a0116b27`，"Layer split + --batch: the default pipeline wins over --batch-mtp (#2038)"），本次只读，未修改、未构建、未跑基准。
- Aila 侧：仅新增本文件。
- 引用约定：所有 Strata 出处写成 `tmp/Strata/<相对路径>:<行>` 或 `docs/<文档>.md §<小节>`；Strata 的实测数字一律注明其测量硬件；凡不是直接测量、而是我据机制推出来的数字，标 `[待验证预估]`。
- Aila 现状引用（不在 `tmp/Strata` 内）：`AGENTS.md`、`docs/plans/qwen38-roadmap-status-2026-10-12.md`、`tests/ops/Qwen4ExpKernelTests.cpp`、`src/models/Qwen4Exp*.cpp` 中的 `AILA_Q4E_*` 开关（仅作为落地设施引用，不改动）。

## 0. 背景校准（读之前先对齐三件事）

1. **Strata 的目标机不是 A770。** 它的主线是 CUDA/HIP，专家全量放 RAM、热专家上卡、CPU 池并行算未命中专家（`docs/HOW_IT_WORKS.md:20-33`）。它的 SYCL 端口（`sycl/`）是"把整棵 CUDA 引擎迁到 oneAPI"（`docs/INTEL.md:156-159`），在 Arc Alchemist（A750/A770）上是**实验路径**：Intel 维护者没有 Arc 卡，所有 Arc 数字来自社区与 fork 的自测（`docs/INTEL_ARC.md:8-11`）。
2. **Strata 自己在 A770 上的结论是"引擎结构（每层 host↔GPU 握手 + CPU 专家池）是瓶颈，不是单个 kernel"**：A770（fork 自测机：PCIe 3.0 x4、实测 1.7-1.8 GB/s、Ryzen 5 5600、约 31 GiB RAM）上 `waitA/waitB`（GPU 空转等 CPU 专家池 flag）+ `copy+combine` 约占 verify 窗口的 **87%**，`gputop` 采样显示 A770 计算引擎在 decode 期间均值约 20%（`docs/A770_SPEED_SPEC.md:688-708, 734-761`）。Aila 的 A770 实测链路是 9.6 GB/s（比那台测试机快约 5 倍），且 Aila 已把 staging 归零、冷专家走 CPU——方向一致，但**所有 A770 数字都必须重测**。
3. **Strata 的工程纪律与 Aila 一致，可以直接互操作**：新功能一律 opt-in、默认路径字节不变、每项带关闭开关与启动日志、一次 PR 一件事、实测数字必须带硬件（`tmp/Strata/AGENTS.md:29-35`）。下面的候选尽量按此口径写最小实验。

## 1. 候选条目

### C1. prefill 双制度专家暂存：路由后小环 + 高覆盖率 stream-all 大环 + stream-ahead

1. **机制（一句话）**：prompt 分块读时按"本块 token 数"切换两种专家供给策略——小块先路由、再只把**被路由到的**非驻留专家经一个 8 槽小环拷上来；块足够大（默认 ≥1024 token）时改按固定顺序把**所有**非驻留专家经大环（按字节预算换算成槽，Q2_0 上 384/1024 槽）连续流式拷贝，让拷贝引擎在当前层 attention 期间持续工作；`STRATA_PREFILL_STREAM_AHEAD`（默认开）让下一层专家的拷贝提前发起。
2. **Strata 出处**：`tmp/Strata/src/prefill/prefill.cpp:107-153`（`STAGE = 8`、`stream_ahead_enabled()`、`stream_all_min()` 与它的实测依据）、`:199-210` 与 `:277-299`（环槽数/字节预算规则）、`:117-119`（"大块时 copy engine 不等每层路由"）；结论表 `docs/INTEL.md §The stream-all prompt walk is back (2026-10-03)`（`:613-626`）；机制概述 `docs/HOW_IT_WORKS.md:50-53`。
3. **Strata 实测**（Arc Pro B70 32 GB，`xe`，oneAPI 2026.1，engine 0.1.40-sycl，IQ3_S，11.5k/24.6k 专家驻留，PCIe 实测 13.3 GB/s H2D）：40K prompt routed-only 1,106 → stream-all 1,160 tok/s；128K int8（KV 流式）897 → 920；128K k8v4 1,006 → 1,041；256K 760 → 780。反向代价同样实测：覆盖率低时 stream-all 会拷数倍于被路由的专家，同一台的 IQ2_XS 2,184-token prompt 从约 560 掉到 254 tok/s，所以规则是"**驻留覆盖率 >90% 才开 stream-all**"。`stream_all_min()` 的 1024 阈值来自 RTX 5070（Q2_0/IQ2_XS，固定 cache）：1,500-token prompt 621→785 / 612→735 tok/s，2,000 token 727→934 / 712→892，4,000（末块）779→912 / 766→844。环大小：Q2_0、8192-token 块下 96 槽 1,153 tok/s vs 384 槽 1,294（该条注释未写卡型，同节上下文为 RTX 5070）。
4. **为什么可能适用于 A770 16GB+64GB RAM + NF4/SYCL**：这正是 Aila 自己列出的"剩余最大杠杆（B3-3）"的**同族问题**——Aila 现状是 chunk-major（14K prompt ≈ 3.4 个 4096 块 × 48 层 × 每层专家），冷专家每块重拷一遍，H2D 总量约 310 GB，Aila 自己的预估是改成层级序后 H2D ≈62 GB、ttft −13~−18%（`docs/plans/qwen38-roadmap-status-2026-10-12.md §B3`）。Strata 证明：在"一块内几乎命中全部专家"的规模上（4096 token 对 512 专家/层，等于 coupond collector 饱和），固定顺序 stream-all + 大环能把拷贝完全藏在计算后面，且**不需要**重写成层级序；反过来，Aila 若只想先拿部分收益，可以只改"块边界之间复用上一块已驻留的冷专家"这一半。差别声明：Strata 并**没有**实现 layer-major prefill（全仓库检索无 layer-major/expert-major 相关代码），它解决的是同一 H2D 浪费的另一半，二者可叠加。
5. **预期收益量级** `[待验证预估]`：只做"regime 切换 + 大环 + stream-ahead"：14K ttft −5~−12%；若同时做跨块冷专家复用（等价 Aila B3-3 的一半）：合计向 −13~−18% 靠拢。Aila 已有 `AILA_Q4E_EXPERT_MAJOR_PREFILL`（默认 true）、`AILA_Q4E_PREFETCH_PREFILL/_TOPN/_MAX`、`AILA_Q4E_STAGING_MB/_SLOTS`、`AILA_Q4E_STAGE_COPY_PARALLEL` 等开关，先量出当前 H2D 字节/块再谈。
6. **移植成本/风险**：中。Aila 已有 expert-major prefill 与 prefetch 开关，主要工作是"策略 + 计量"而不是新数据通路。风险：改变块的专家供给顺序会改 prefill 的浮点结合顺序（Strata 明说 `STRATA_PREFILL_FIRST` 会把 expert GEMM 的 rounding 移动、长 greedy 会在第 45 个 token 分叉，`docs/INTEL.md:291`），与 Aila 的逐位门冲突——必须作为 opt-in，并用 Aila 现成的续轮/前缀对比口径评估。与已否证项的关系：chunk 8192 已否证（Aila），本项不改 chunk 大小；"blocked 布局"无关。
7. **建议的 Aila 最小实验**：（a）探针：在 `tmp/q4e-*` runner 的 14K 生产序列上打 H2D 字节/层/块 + 每块"被路由且非驻留"专家数 vs "全部非驻留"专家数，用 `AILA_Q4E_CACHE_TRACE`/`AILA_Q4E_DMA_PROF` 现有通道；（b）用 `tmp/q4e_cache_sim.py`（replay `AILA_Q4E_DEBUG_ROUTE_TRACE`）离线算跨块复用上限；（c）门：`ctest -L 'cpu|abi'` + 同块顺序下 prompt hash 不变，ttft 落在 ±0.6% 噪声外；（d）实模：`run.py <tag> --agent` A/B/B/A，只动 `AILA_Q4E_EXPERT_MAJOR_PREFILL`/新加的 regime 阈值。

### C2. 用 GPU 的 copy-only 队列（blitter engine）承载专家 H2D，而不是 compute 队列

1. **机制（一句话）**：Level Zero "v2" 适配器把**所有** SYCL 队列（包括只做 H2D memcpy 的 in-order 队列）都放在 compute 引擎上，于是 2 MB 的专家拷贝会把后面的 kernel 顶住；Strata 用 `zeDeviceGetCommandQueueGroupProperties` 找到 copy-only 队列组（ordinal 1），在其上建 immediate command list，再用 `sycl::make_queue<ext_oneapi_level_zero>` 包成 SYCL 队列，事件/`ext_oneapi_submit_barrier` 跨队列照旧工作。
2. **Strata 出处**：`tmp/Strata/sycl/include/strata/copy_queue.hpp:1-26`（为什么/怎么做的注释即设计文档）、`tmp/Strata/sycl/src/core/copy_queue.cpp:30-70`（`ze_loader` 动态加载、按设备缓存队列、失败返回 nullptr 由调用方退回普通队列）；测量与默认规则 `docs/INTEL.md:812-817`。
3. **Strata 实测**（Arc Pro B70 32 GB，xe，oneAPI 2026.1）：开 copy engine 后 4K prompt 3.82 → 3.25 s、16K 15.5 → 13.5 s；unitrace 显示此前 **2,000 个拷贝中 0 个**与 kernel 重叠。默认只在 Battlemage 开；Alchemist（A750 8GB，i915，PCIe 4.0，10.6 GB/s）上只 +1.4%，并且"stager 的完成标志若是一次 copy-engine memcpy，prompt 会挂死一次"，因此 **A 系默认关闭**；`STRATA_COPY_ENGINE=1` 可在任意 Level Zero 设备强制打开。
4. **为什么可能适用于 A770 16GB+64GB RAM + NF4/SYCL**：Aila 的 prefill/冷专家数据流是 H2D 密集（C1 的 310 GB 就是这条通路），而 DG2 上如果 Aila 用 L0 v2 适配器，拷贝与 kernel 很可能同样串行。这与 Aila 已否证项不冲突（那些都是核内/权重布局项；这是队列放置）。注意 Aila 是 **Alchemist**，正是 Strata 明确"默认不开"的分支——所以这里搬的是**机制 + 那次挂死的根因修复**，不是默认值。
5. **预期收益量级** `[待验证预估]`：prefill ttft −5~−15%（若当前拷贝确实在 compute 队列上串行）；若 Aila 已经把 staging 拷贝放在独立队列/UMA 路径上，则接近 0。decode 侧收益取决于 staging 是否非 0（Aila 当前 staging 全 0）。
6. **移植成本/风险**：中。风险集中在 Strata 记录两次的 Level Zero v2 陷阱：（a）宿主线程一旦 `wait` 过某队列事件，另一队列的 barrier 可能永远不释放——修法是"copy 队列每完成一次 DMA 就往 page-locked 主机内存写序号，stager 轮询序号"，**不要让宿主线程等队列事件**（`docs/INTEL.md:577-587`）；（b）用 copy-engine memcpy 当 stager 完成标志会在 Alchemist 上挂死（`docs/INTEL.md:816-817`）。Aila 的"取消必须 drain 完 GPU 工作再释放 USM"（`AGENTS.md §6`）要求把这套序号协议接进 cancel 路径。
7. **建议的 Aila 最小实验**：（a）探针：仿 `tmp/Strata/sycl/probe/` 风格写一个独立 SYCL 探针（Aila 可挂在 `AilaQwen4ExpKernelTests` 里当 opt-in case）：同一段 2 MB × N 拷贝分别提交到默认 in-order 队列与 ordinal-1 copy 队列，比较"拷贝期间同队列 kernel 的墙钟"；同时用 `zeDeviceGetCommandQueueGroupProperties` 打印 DG2 是否暴露 copy-only 组；（b）门：探针显示拷贝确实顶住 kernel，且 copy 路径按字节一致（`AILA_Q4E_DMA_PROF` 已有字节校验思路）；（c）实模：只对 prefill staging 打开，`run.py <tag> --agent` A/B/B/A，重点看 cancel-recover 与 14K ttft。

### C3. 非 16B 块宽的 16B 装载（`load16_a2`）与宽核：把"错位向量装载"换成两次对齐装载 + 移位

1. **机制（一句话）**：量化块步长不是 16 字节整数倍时（Q6_K = 210 B），按块起始地址直接发 16B 向量装载会被硬件拆成碎片；改成"取两次 16B 对齐装载 + 按 2/4 字节位移拼出需要的 16 字节"，且第二次装载只在地址真的不对齐时发（不会越过分配末尾触发页错误）。
2. **Strata 出处**：`tmp/Strata/sycl/src/kernels/cuda/native_mmvq.dp.cpp:1505-1537`（`load16_a2` + `q6k_a2()` 开关）、`:3429-3447`（宽核 gate 从 `n_out>=4096` 放开到全部 shape）、`sycl/src/kernels/q6k_align_bench.cpp:1-20`（专门的 stride 210 vs 224 对照 bench）；结论 `docs/INTEL.md:448-456`（"相同权重按 224 字节步长重排后 2.3-4.7x"）、`:466-479`（宽核清单与开关 `STRATA_MMVQ_WIDE_32=0`、`STRATA_MMVQ_WIDE_K=0`）。
3. **Strata 实测**（Arc Pro B70 32 GB）：Q6_K 2560→10240 在 1/2/4/6 列 160/145/141/101 → 493/432/331/273 GB/s，248K 行 head 1 列 150 → 407 GB/s，**Coder decode 44.9 → 54 tok/s，输出 token 完全相同**；IQ4_XS 2 列 107 → 323 GB/s；Q8_0 与 IQ4_NL 的 640→2560 形状 4-6x（189 → 31 µs、28 → 6.6 µs）。对照数据：同卡 4 字节/lane 顺序流 395 GB/s，16 字节 596 GB/s（`native_mmvq.dp.cpp:1507-1508`）。
4. **为什么可能适用于 A770 16GB+64GB RAM + NF4/SYCL**：Aila 的 NF4 权重块 = 64 权重（32 B 打包）+ absmax/scale，块步长取决于布局：32 B 数据 + 4 B fp32 absmax = **36 B**（4 字节对齐），或数据/scale 分离数组（各自对齐）。只要步长不是 16 B 倍数，Aila 的 decode GEMV/前投影里任何 16B 装载都会走 Strata 记录的同一条硬件退化路径，而 Aila 的 decode 是带宽受限（AGENTS.md §3：NF4 的 0.5 B/权重是核心优势）。与已否证项的关系：**不同**——已否证的是"uint64 权重读""blocked 布局""half-unpack""字节对 LUT""2 rows/subgroup"，本项是"不改布局、只改装载方式"，而且 Strata 的对照里"重排布局"那条（224 B stride）只是基准对照，不是建议方案。
5. **预期收益量级** `[待验证预估]`：若 Aila 的 NF4 块步长确实非 16B 对齐且 kernel 已用 16B 装载：decode GEMV 带宽 +20~100%，端到端 +3~15%；若 Aila 当前只用 4B/8B（`vec8`）装载或布局已对齐：≈0。必须先量步长与装载宽度，不要先改 kernel。
6. **移植成本/风险**：低。`load16_a2` 是纯字节重排、位结果不变（Strata 用逐位 parity 测试），Aila 可直接复刻到自己的 NF4 GEMV；风险是地址在分配末尾的越界（Strata 的写法用"未对齐才读第二块"规避，见 `native_mmvq.dp.cpp:1520-1522`），以及 Aila 的 subgroup 尺寸（16/8）与 Strata 的 32 不同，寄存器/选择逻辑要按 Aila 几何重写。
7. **建议的 Aila 最小实验**：（a）探针：在 `AilaQwen4ExpKernelTests` 加一个 case，把真实的 NF4 gate/up/down 权重按 (i) 原步长、(ii) 16B 对齐重排步长 各测一遍同样的 GEMV/MoE decode 内核，报 GB/s（借 `tmp/Strata/sycl/src/kernels/q6k_align_bench.cpp` 的结构）；（b）门：逐位一致 + 现有 `AilaQwen4ExpKernelTests` 全绿；（c）实模：只在 decode GEMV 打开，`run.py <tag> --agent`（14K decode 15.5 tok/s 基线，噪声 ±0.6%）。

### C4. 路由画像工具链：设备侧计数 → 按 trace 归一化 mix → profile，并支持"自适应学到的热度回存"

1. **机制（一句话）**：三段工具链——(a) 引擎把每条 prompt 的 decode 路由计数从**设备侧**导出（`STRATA_USAGE_DUMP=PATH`，与 `--dump-routing` 同格式，SYCL 上也工作）；(b) `make_profile_mix.py` 把每条 trace 先归一化成分布再按权重求和（不是直接相加，避免长 trace 主导排名；一组 N 条给 1/N 权重使该组只算一次），生成 `--expert-profile` 用的排序文件；(c) `--expert-profile-save P` 把自适应 tier 学到的热度（带衰减）在退出与每 N 分钟落盘，作为下次启动的先验。
2. **Strata 出处**：`tmp/Strata/tools/collect_profile_traces.py:1-18`（45 prompt 语料、每 prompt 一次贪婪跑、取生成文本的路由、约 30 分钟）、`tools/make_profile_mix.py:1-17, 35-58`、`tools/make_profile.py:1-26`（profile 格式、排名语义、"基础 profile 会吞掉 trace"的坑）、`src/program/generate.cpp:492-494, 849-851, 8328-8333`（`--expert-profile-save` 语义与启动检查）；测量 `docs/INTEL.md:824-829`。
3. **Strata 实测**（Arc Pro B70 32 GB，IQ3_S，11.5k 缓存槽，250-300 token decode）：出厂 profile 命中聊天/代码/非英语路由的 60%、文档摘要的 85%；用 45 条聊天/代码/语言 prompt + 8 条摘要（权重 1:2）重生成后，对**两套都没见过**的 prompt 留一验证命中 71%/84%，12 条 prompt 上 decode **+5% ~ +64%（均值 +21%），tokens 完全相同**。
4. **为什么可能适用于 A770 16GB+64GB RAM + NF4/SYCL**：Aila 路线图把"动态重排 hotset（需 observe→export→重新生成 profile 的工具链）"列为 B3 的次选，但工具链还没建；Aila 已有 `AILA_Q4E_HOTSET_PROFILE/_EXPORT/_INTERVAL/_SWAPS`、`AILA_Q4E_DEBUG_ROUTE_TRACE` 与 `tmp/q4e_cache_sim.py`（可离线 replay trace 算命中率），所以缺的正是 Strata 的"归一化 mix + 留一验证 + 回存先验"三件套。Aila 的 agent 负载（PI agent、代码编辑、长上下文复用）与 Strata 的"聊天/代码/文档摘要"混合 corpus 同类。
5. **预期收益量级** `[待验证预估]`：Aila 已有 hotset（P3），若出厂 profile 是单域，重生成后 agent 负载 decode +3~+15%；若已混合良好则 0。注意 Aila 实测弹性只有 **0.45~0.5%/100 槽**，所以纯"多排 156 个 key"型收益上限 ~1%（Aila 已兑现），本项指的是**排序质量**（同样的槽装更对的专家）。
6. **移植成本/风险**：低。风险：(a) profile 改变驻留集合会改 rounding，Aila 必须保持 opt-in 并且不进逐位门；(b) Strata 明确提醒 i915 卡上"miss 由 CPU 算，residency 会移动 rounding"，故 A 系保留出厂 profile（`docs/INTEL.md:828-829`）——Aila 是同一分支，采数时要把"训 profile 的机器/配置"写进收据；（c) 别把 45 prompt 的采集开销塞进 CI。
7. **建议的 Aila 最小实验**：（a）探针：用 `AILA_Q4E_DEBUG_ROUTE_TRACE` 采 20-40 条 agent/代码/中文 prompt（沿用 `tmp/q4e-*` runner），用 `tmp/q4e_cache_sim.py` 做留一命中率与"top-2948 覆盖"对比：出厂 hotset vs 归一化 mix 排序；（b）门：命中率相对提升 ≥ 3pp 才进实模；（c）实模：`run.py <tag> --agent` B/A/B，14K decode + 续轮 rate，噪声 ±0.6%；（d）同时实现 `--expert-profile-save` 式回存（Aila 已有 `AILA_Q4E_HOTSET_EXPORT`，确认其是否走"下一次启动先验"语义）。

### C5. 自适应 tier 的重叠落地 + 交换旋转：把 swap 的拷贝藏进下一个窗口，并消掉一次 host memcpy

1. **机制（一句话）**：(a) 自适应换出/换入不再阻塞窗口——换表与上传在主线程、拷贝在 helper 线程/卡的 refill 流上，verify 窗口按层"等这一层的拷贝落地"再放行该层 plan（默认信号是一个 4 字节的 stream-ordered 拷贝写进 pinned 内存，不用宿主回调线程）；(b) 交换旋转：被换出的专家不再"拷回 RAM 旧槽"，而是让临时宿主缓冲直接成为常驻、旧槽变下一个临时缓冲，省掉每次交换的一次 host memcpy。
2. **Strata 出处**：`tmp/Strata/docs/BATCHING.md:144-163`（`--adapt-async` 语义与实测表）、`docs/DETAILS.md:1720-1743`（`STRATA_ADAPT_OVERLAP` 的落地语义 + per-layer landed 信号）、`docs/EXCHANGE_ROTATION.md:1-27, 39-47, 105-111`（旋转的前置条件与同步契约）；实现位置提示：`src/core/expert_source.cpp`（exchange）、`sycl/include/strata/core/exchange_storage.hpp`、`include/strata/core/foresight_swap.hpp`。
3. **Strata 实测**：
   - 重叠/异步（RTX A4000 16GB，6 对交错 whole-engine runs，decode 中位数）：Q2_0 16GB 62.5 → 66.2（+5.9%）、Q2_0 当成 12GB 51.3 → 54.2（+5.7%）、IQ3_XXS 16GB 48.6 → 52.6（+8.3%）、IQ3_XXS 12GB 39.3 → 41.5（+5.5%），6/6 对更快；Tesla P100 +4.4%；但 **RTX 5070 在 Windows 上 −2.6%**，**RTX 5090 Laptop(WSL2) −21%（84.6 vs 106.6 tok/s，5 次 256-token 生成的中位数）**——所以现在所有后端都是 opt-in。
   - 批窗口里的异步 tier（2x RTX 3080 20GB，UD-Q4_K_XL，两路 300-token 贪婪解码）：`--adapt-async 1 --adapt-every 2` 合计 89.2 tok/s（adapt wait 0.2 ms/窗口）vs 阻塞 74.7（8.4 ms）vs 冻结 69.5 / 45.2。
   - 交换旋转（RTX PRO 6000 Blackwell 96GB，Q2_0，GPU cache 人为限到 12,000 槽，1,024+1,024 token）：104.90 → 113.44 tok/s（+8.1%），3,802 次交换里避免 5,255,884,800 字节 host-copy payload，1,024 个输出 token ID 完全相同。前置条件：每层专家块等大、常驻补集与交换缓冲都 fully pinned + GPU-mapped。
4. **为什么可能适用于 A770 16GB+64GB RAM + NF4/SYCL**：Aila 实测"swap 默认关闭时 decode +8.03%"——即 Aila 的 swap 路径目前净亏 8%，与 Strata 在重叠之前的状态一样；Strata 的两条机制正好从两头削这个成本（把拷贝藏进窗口 + 每次交换省一次 memcpy），且 Aila 有 `AILA_Q4E_ADAPTIVE_SWAP_FREQ`、`AILA_Q4E_MAX_SWAPS`、`AILA_Q4E_CPU_ASYNC_ROUTE`、`AILA_Q4E_CPU_OVERLAP` 等开关可承载实验。差异必须明说：Strata 的交换旋转依赖 **pinned + GPU-mapped 的宿主专家补集**，而 Aila 已否证"全量 pinned 替代大页"（51 GiB 锁页 → 首请求 `UR_OUT_OF_RESOURCES`）；所以旋转只能在小块（如 Aila 当前 host complement 的某个子集）上做，或直接判定不可行——见 §3。
5. **预期收益量级** `[待验证预估]`：若换出输入仍是 Aila 的瓶颈，重叠 + 旋转合计可把 Aila 那 8% 的 swap 成本回收 30~70%，即 decode +2~+6%；命中率本身不变。
6. **移植成本/风险**：中高。三个具体风险：(a) per-layer landed 信号必须与 Aila 的取消/drain 纪律兼容（`AGENTS.md §6`：取消必须 drain 完 GPU 才释放 USM）；(b) Windows 上 Strata 自己测到过负收益（5070 −2.6%、5090 Laptop −21%），Aila 必须先在 Windows 上量；(c) 旋转的 pinned+mapped 前提与 Aila 已否证的 pinned 项正面相关，做之前先做 §3 里的小规模 pinned 预算探针。
7. **建议的 Aila 最小实验**：（a）探针：量 Aila 一次 swap 的分解（换表/上传/拷回/等待），确认 host memcpy 的字节数与占比（用 `AILA_Q4E_WORK_SUMMARY`/`AILA_Q4E_DMA_PROF` 现有通道）；（b）门：swap 与不 swap 的输出按 Aila 自己的口径比对（Strata 的 exactness 配方见 C10），且 swap 打开的 decode 不低于关闭时；（c）实模：只开"异步/重叠"一项，`run.py <tag> --agent` A/B/B/A（含 cancel-recover）。

### C6. verify 窗口的 route tail skip：跳过"全窗口都在低排名"的未命中专家

1. **机制（一句话）**：多 token verify 窗口里，一个**未命中**专家若被窗口内**每一个 token** 都排在 top-10 的第 7 名及以后（权重最小），就既不拷贝也不计算，贡献记 0、不重新归一化；任何一个 token 把它排在 7 名之前、或它已在 VRAM，就照旧供给。
2. **Strata 出处**：`tmp/Strata/docs/DETAILS.md:1609-1663`（规则、默认、禁用条件、KL 表、速度表、opt-out）；实现 `src/core/expert_source.cpp:3608-3622, 3691-3725, 3865-3867`、默认解析与启动行 `src/program/generate.cpp:8281-8315`。
3. **Strata 实测**（CUDA，5 轮交错、fresh engine，贪心 200 token；默认 vs `STRATA_ROUTE_TAIL_SKIP=0`，比值与"默认更快的轮数"）：RTX 3060 12GB IQ3_XXS resident experts 1.156（5/5）；RTX A4000 16GB Q2_0 1.129（5/5）；Tesla P100 16GB IQ3_XXS 1.146（5/5）；2x A4000 layer split（命中率 99.7%）0.998（1/5，无收益）。精度代价（teacher-forced KL，28 条 prompt ×220 token）：均值 0.0032-0.0058，top-1 一致 97.5-98.2%，p99 0.035-0.061；同引擎跑两次的噪声底是 0.0006-0.0007。当前**只在 CUDA（0.1.42 起）与 RDNA4 HIP 生效，SYCL 不生效**；覆盖率 ≥90% 时自动不启用。
4. **为什么可能适用于 A770 16GB+64GB RAM + NF4/SYCL**：Aila 的 2,948 槽 vs 24,576 专家意味着**每个 token 都有未命中**，未命中的成本是 CPU 计算 + host 往返（route 图里 router_wait 13.4 ms、experts_host 19.7 ms）；tail skip 恰好砍掉"贡献最小"的那部分。但**必须有 ≥2 行的窗口才有意义**：Aila 今天 T1（单行）没有可比较的"每一个 token"，所以这条与 Aila 的 MTP/批量 verify 工程绑定。差异声明：Strata 的 SYCL 端口**没有**实现它（列表里明确 off），所以这不是"抄代码"，而是"按 SYCL 语义补一个 CUDA 有、SYCL 没有的优化"。
5. **预期收益量级** `[待验证预估]`：decode 窗口 T=2~4 且未命中率高时 +8~+15%（照 CUDA 数字的 miss-heavy 端打折）；Aila 若维持 T=1，收益为 0。附带价值：它可能把 Aila"MTP 净负"（15.0-15.3 vs T1 15.6）翻正，因为 verify 窗口的未命中专家被砍掉。
6. **移植成本/风险**：中。需要在设备侧拿到"每个 token 的 rank"（Aila 已有 top-k ids/权重，`AilaQwen4ExpKernelTests` 里就有 router topk 探针）并按窗口取交集；输出会变（opt-in、KL 类门），与 Aila 的 MTP 等价门不兼容——只能作为独立 opt-in 档，不能默认。
7. **建议的 Aila 最小实验**：（a）探针：在 `AilaQwen4ExpKernelTests` 加"窗口内每 token 的 rank 统计"，先用 trace 离线算"若 window=4，多少未命中专家会被跳过"（`tmp/q4e_cache_sim.py` 的 trace 已有 count==top_k 的判定）；（b）门：跳过的专家集合与设备侧 rank 交集规则逐条一致；teacher-forced KL 与 Aila 现有 MTP 探针口径一致；（c）实模：仅在与 MTP/批量 verify 一起打开时测 14K decode 与任务质量。

### C7. 专家缓存的填充/预热流水线：profile 序 admission + 文件序批量读 + 拷贝重叠

1. **机制（一句话）**：启动时按 profile 顺序"准入"槽位（放置不变），但读取按**文件顺序**、最多 8 线程、每次 64 个 page-locked blob 一批；一批的 H2D 拷贝与下一批的读取重叠进行（`STRATA_FILL_SERIAL=1` 退回旧串行路径）。
2. **Strata 出处**：`docs/INTEL.md:605-611`（"Load time (2026-10-02)"）；实现：`sycl/src/core/gguf_expert_source.cpp`、`sycl/include/strata/core/gguf_expert_source.hpp`、`sycl/include/strata/core/expert_source.hpp`（`LoadStats`）。
3. **Strata 实测**（Arc Pro B70 32 GB，`--stream-experts`）：Coder 的 23.4 GiB 专家填充 76.2 s → 18.8 s（0.33 → 1.34 GB/s）；整机启动到首个 token 82 s → 26 s；IQ2_XS 120 s → 41 s（其中 8.2 GB host mirror 约 9 s）。输出不变。
4. **为什么可能适用于 A770 16GB+64GB RAM + NF4/SYCL**：Aila 的模型加载/专家填充与 per-request 的 staging refill 都是同构的"按需读盘 → H2D"路径；Aila 已有 `AILA_Q4E_FILL_THREADS`、`AILA_Q4E_FILL_PARTS`、`AILA_Q4E_FILL_SHORT_PARTS`、`AILA_Q4E_EXPERT_READER`、`AILA_Q4E_MMAP_PREFETCH`，所以这条大概率是**已有设施的调参/合并**而不是新设计；价值主要在缩短迭代周期与冷启动，不在 ttft。
5. **预期收益量级** `[待验证预估]`：模型加载时间 −50~−75%（若 Aila 目前是"读一个拷一个"的串行路径）；对 14K ttft 影响 ≈0（除非 prefill 的 staging refill 走同一路径）。
6. **移植成本/风险**：低。风险很小：批量/pinned 缓冲会占 RAM（Strata 的 64-blob 批在 2 MB blob 下约 128 MB），Aila 64 GB RAM 无压力；注意 Windows 上 pinned 预算（见 §3）。
7. **建议的 Aila 最小实验**：（a）探针：启动计时分解（读盘/拷贝/等待），在 `tmp/q4e-*` 里记成 receipt；（b）门：加载后 `AILA_Q4E_ARENA_VERIFY_ROWS` 抽查字节一致；（c）实模：连续 3 次冷启动取中位数。

### C8. 工程方法学包（可立即照抄的操作性细节）

1. **机制（一句话）**：把"怎么测、怎么判"写成硬规则，避免自欺——事件基准的输出 schema、固定 token 流的 ms/轮口径、逐卡实测表 + 保守发射规则、用 GPU 占用率反证"瓶颈是 kernel"的直觉。
2. **Strata 出处与可直接抄的细节**：
   - **事件基准**：队列必须 `in_order` + `enable_profiling` 且与设备/上下文一致；用 marker kernel 界定批次；只报 "device queue span"（含调度间隙）并同时报 host wall 与 empty-marker 开销；要求时间戳非零且单调，否则**报错而不是打印 0**；输出 schema 含源码修订、PCI ID、队列模式、编译选项、N/K/T、repeat/warmup、设备跨度中位/最小、host wall、空 marker 成本、有效带宽、正确性结果与输出 hash（`docs/A770_SPEED_SPEC.md:86-138`）。
   - **A/B 协议**：每臂 ≥3 次预热重复、优先交错 A/B/A/B、每个模式独立进程（选择器是 read-once 静态量，同进程 setenv 无效）；固定种子/贪心、比 **ms/轮** 与轮数而不只是 tok/s；记录"pairs faster"计数；先量 A/A 噪声底（Strata 用 KL 0.0006-0.0007 与重启 1.009/1.005 的比值底，`docs/DETAILS.md:1642`、`:1777-1778`）；明确"单 kernel 排名会误导"——A770 的 router 微基准 3.3x（395.47 → 118.62 µs/次，逐位一致）端到端为 0（`docs/A770_SPEED_SPEC.md:58-59, 214-218`）。
   - **逐卡实测表 + 保守发射**：一个格子只在"两张被测卡中较差的一张仍能 +3%"时非零；同类 shape 结论不一致就取 0；未测类沿用下方邻居；换卡无需重编译（`STRATA_MMVQ_IL_ROWS="type:ncols:r0,..,r4"` 环境变量覆盖），并提供 `--bench --emit-table` 打印可直接粘贴的表（`docs/MMVQ_IL_TABLE.md:18-32`；实现 `src/kernels/cuda/native_mmvq.cu:2543-2558, 2603-2606`）。
   - **反证法**：先用 `gputop`/unitrace 量 GPU 是否真的忙，再决定优化 kernel——A770 上 ccs 采样 0-20%（偶发 84%）推翻了"ms_wait = GPU 在算"的假设，随后 stage 表（`waitA/waitB/copy+combine` ≈87%）给出决定性证据（`docs/A770_SPEED_SPEC.md:688-708, 734-761`）。同理，Strata 把"80K prompt 的 75 s/1,065 tok/s"整体作废，因为发现 stream plan 在 512 槽环上越界复用了 blob 指针、测量部分用了错误专家（`docs/INTEL.md:318-324`）——**先证明测量本身正确**。
   - **收据**：A/B 收据 JSON 里放源码/二进制/模型/pack/profile 的 hash、参数、计时与诊断，token 对比单独存 raw ids 并附可独立运行的校验脚本（`docs/EXCHANGE_ROTATION.md:105-138`，`bench/results/exchange-rotation-ab.json`、`bench/results/exchange-rotation-token-ids.json`）。
   - **开关纪律**：每个新行为一个 `STRATA_*`/`--flag`、启动时打印一行状态与"怎么关"，未测量不放默认（`AGENTS.md:29-32`；例子遍布 `docs/DETAILS.md:1691-1743`）。
3. **Strata 实测**：以上规则的来源数字全部已在各条候选里列出（测量硬件见对应条目）。
4. **为什么可能适用于 A770 16GB+64GB RAM + NF4/SYCL**：Aila 已有相当好的纪律（`docs/plans/*.json` 收据、`ctest -L 'cpu|abi'`、14K 噪声 ±0.6%、A/B/B/A），但**缺** Strata 这套"事件基准 schema + 逐卡表发射规则 + GPU 占用反证 + raw ids 收据"的成品模板；且 Aila 正在做大量"微基准好看、端到端不动"的实验（W4A8、int8 KV、DPAS 都是 ±1~2% 量级），正好需要"3% 才发射""先证测量正确"这类闸门。
5. **预期收益量级**：不可直接量化（节省的是错误结论与返工）。`[待验证预估]`：把"微基准收益 × 端到端折算率"显式记进收据，可避免 1-2 次整轮campaign 走错方向。
6. **移植成本/风险**：低。唯一要注意的是 Strata 的部分条目绑定其自身工具链（CUDA events、unitrace、`PhaseA` 测试名），Aila 要换成 SYCL event profiling（Aila 已在用 VTune 目录体系 `tmp/*-vtune*`）。
7. **建议的 Aila 最小实验**：（a）把 `docs/A770_SPEED_SPEC.md:130-138` 的 schema 落成一张 Aila 的 benchmark JSON 模板（一个 `tmp/q4e-*` receipt）；（b）给 `AilaQwen4ExpKernelTests` 加一个 `--bench --emit-table` 式输出（每 shape×宽度 一行，机器可读）；（c）在下一轮 14K 实验里强制填 A/A 噪声与 pairs-faster 计数。

### C9. suffix-lookup drafter + DraftPolicy：不依赖权重、只验证不改变输出的草稿源

1. **机制（一句话）**：一个纯 CPU 的 n-gram（trigram 哈希，每 token 约 20 B，开放寻址、每 key 记 4 个最近位置）草稿器，在当前序列后缀的最长历史重现之后提议下 k 个 token；`DraftPolicy` 在线学习"MTP 窗口 vs lookup 窗口"的**每毫秒期望提交 token 数**（按 match 长度分 4 桶、带衰减计数、每窗口实测轮时），只在 lookup 期望值超出 margin 时才用 lookup 窗口——它只决定验证什么，输出不变；`--lookup-chain` 可在 MTP 窗口之后再串 lookup token。
2. **Strata 出处**：`tmp/Strata/include/strata/spec/suffix_drafter.hpp:1-41`（索引与匹配设计）、`include/strata/spec/draft_policy.hpp:1-62`（E/cost 公式与 `margin=0.03`）、`docs/HOW_IT_WORKS.md:43-46`（"prompt lookup 最多草拟 5 token；代码编辑 6-11% faster，其他文本不变；草稿与 MTP 同样被验证，所以输出相同"）、`docs/INTEL.md:492`（计划项 8：窗口是延迟受限的，双草稿分支近乎免费）、`docs/A770_SPEED_SPEC.md:254-258`（A770 上 `--spec-split`（草稿与 verify 重叠）**变慢**：9.9/10.2/10.3 vs 11.7/12.1/12.6 tok/s，引擎默认与 `--no-spec-split` 一致）。
3. **Strata 实测**：B70 上纯 suffix drafter（`--spec 2`）的 decode 20.2-20.8 tok/s、接受率约 9%，明显不如 MTP（42.7 tok/s）——即"只靠 lookup 不值，值在 lookup 与 MTP 的**择时**"（`docs/INTEL.md:213-223`）。代码编辑场景 Prompt lookup +6~11% 墙钟（`docs/HOW_IT_WORKS.md:44-46`，该行未标卡）。A770 上草稿/verify 重叠是负收益（上引）。
4. **为什么可能适用于 A770 16GB+64GB RAM + NF4/SYCL**：Aila 的 agent 序列（PI agent、代码编辑、长 context 复用；续轮已有 `cached_prompt_tokens=14072`、first content 1.486 s）正是 lookup 命中率最高的场景；suffix drafter 不需要模型权重、不占 VRAM、纯 CPU 微秒级，而且**输出由 verify 决定**，因此不像 MTP 那样直接撞 Aila 的等价门（Aila B4-3 的失败点在"被拒绝窗口的 row0"）。它与 Aila 的"T1 15.5 tok/s"互补：lookup 让窗口变长，多行的收益要由 verify 路径承担。
5. **预期收益量级** `[待验证预估]`：编辑/引用型回合 +6~11%；散文/闲聊 ≈0（策略会自动退回单 token）。风险收益比好，因为失败模式只是"多花 verify 时间"，可由 DraftPolicy 的 margin 自动关闭。
6. **移植成本/风险**：中低（算法清晰、可单元测试），但有两个前置：(a) Aila 的 verify 多行路径必须先可信（B4-3 的 1491 行分歧未定位）；(b) A770 上草稿与 verify 的**重叠**是负收益（Strata 实测），所以 Aila 的 lookup 必须走"串行窗口"而非双流重叠——这条与 Strata 自己的 `--spec-split` 结论直接冲突，必须按 A770 的实测走。与已否证项的关系：与"MTP 直接接入"不同（无 MTP 权重、无 draft head、无 draft KV），不重复。
7. **建议的 Aila 最小实验**：（a）探针：离线——把 `tmp/q4e-agent-corpus-v1` / `tmp/q4e-agent-pi-long*` 的 token 流喂给一个 Python 版 trigram lookup，报"match≥3 的命中率、平均可提议长度、被 verify 后接受率上界"；同时用 `AILA_Q4E_PREDICT_GATING/_STATS` 现有预测通道对照；（b）门：接受率上界 > 0.5 且平均 token/窗口 > 1.5 才实现 C++ 版；（c）实模：`run.py <tag> --agent`，只开 lookup（无 MTP），比 14K decode 与续轮 rate。

### C10. batch-MTP 的取舍规则 + batch 行"等价于单行"的配方

1. **机制（一句话）**：批窗口默认**不带** MTP 草稿（每槽每窗口 1 token）；`--batch-mtp` 让每个槽在窗口里多验证 1 个 MTP 提案（每个槽有自己的 drafter，共享权重、只拥有自己的 K/V 环）；按层切分时 `--batch` 默认把多个槽分组成流水线（pipeline），而流水线路径**不跑槽的 MTP 草稿**，所以 0.1.43.1 起默认是"流水线赢，--batch-mtp 关闭并打印原因"；要保留串行 MTP 窗口就用 `--batch-groups 1` 或 `STRATA_BATCH_MTP_SERIAL=1`。另有 exactness 配方：批行的算术必须与单行窗口一致（`STRATA_IQ_MT_MIN=1`、`--pcie-frac 0`、`--adapt-every 1000000`、读 prompt 时 `--no-prefill-borrow`），满足后可逐 token 验证 8 路并发 150 token 与单路完全相同。
2. **Strata 出处**：`docs/BATCHING.md:3-10, 22-28, 36-39, 96-110, 112-142, 144-163, 182-209`；HEAD 提交信息（`a0116b27`，0.1.43.1 的取舍与 2x A4000 数字）；`docs/MULTI_GPU.md:246-250`（`--pipeline-windows` 默认值）。
3. **Strata 实测**（每台机器都不同，务必带上）：2x RTX 3080 20GB + Xeon E5-2696 v4、UD-Q4_K_XL、`layer_split 23`、2 槽 300 token 贪婪：`--batch-mtp` 89.2 vs 串行无 MTP 77.3 vs 流水线默认 77.7 tok/s（+15.4%，95% bootstrap +12.5~+17.9%）；单流本身两者相同（81.7/82.0）。2x R9700（专家全在 VRAM）：80.2 → 62.7（−21.9%，0/5 对更快）——**只有专家不全驻 VRAM 时才赚**。RTX PRO 5000：2-4 客户端总吞吐 +31~+39%。HEAD 的层级流水线 vs `--batch-mtp`：2x RTX A4000 四流 62.2 vs 92.7（−33%，0/3 对），两流 55.7 vs 71.4（−22%）。4x R9700 8 客户端 `--batch-groups` 分组 166 vs 单组 86 tok/s。
4. **为什么可能适用于 A770 16GB+64GB RAM + NF4/SYCL**：Aila 有连续批量（`--decode-batch-size 1..4`）与多会话，2,948 槽意味着"专家不全驻 VRAM"——正好落在 Strata 说"batch-mtp 才赚"的区间；而 Strata 的取舍规则（流水线与 batch-MTP 互斥、按客户端数/长 prompt 到达选择）可以直接作为 Aila 的**默认策略文档**。但要承认：Aila 的 MTP 目前因等价门未过而关闭（B4-3），batch-MTP 只有在 MTP 恢复后才可用；本条的即时价值是"当默认策略冲突时，先按实测把两个默认互斥并打印原因"这套工程决定。
5. **预期收益量级** `[待验证预估]`：多会话聚合吞吐 +15~+35%（若 Aila 的多会话 decode 走批量且专家 miss 率高）；单会话 ≈0 或负。Aila 若最终恢复 MTP，还需 `--batch-mtp` 的 per-slot drafter 状态开销（Strata：每槽 50 MiB 私有状态 + 35-110 个专家槽）。
6. **移植成本/风险**：高（涉及会话态、drafter 状态复制、批量窗口行数）。风险：Strata 自己的 2x R9700 结果说明在"专家全驻"时会 −22%，Aila 的 slot/专家预算必须由 setup/启动日志按实测自动决定，不能默认开。
7. **建议的 Aila 最小实验**：（a）探针：把 batch 窗口的"行数/窗口"与 miss 率记进 `AILA_Q4E_WINDOW_ACCOUNT_LOG`（已有）；（b）门：先复刻 Strata 的 exactness 配方（固定 seed、禁 swap、禁 borrow、固定 PCIe 取向），验证"Aila 批量行 == 单行"逐 token；这正是 B4-3 需要的隔离手段；（c）实模：`tests/server/batching_socket_tests.py` + `tmp/q4e-server-mtp-*` 的 4 路并发场景，比聚合吞吐与首 token 等待。

### C11. 编译/稳健性小项（低成本，先探针后采纳）

1. **机制（一句话）**：一组各自独立的小修复/开关，都是"不测就不用"的类型：(a) JIT 构建加 `-ze-opt-greater-than-4GB-buffer-required`，避免 IGC 把 kernel 指针参数当地址 32 位偏移（>4 GiB 分配读错字节）；(b) `-fsycl-device-code-split=per_kernel` 缩短首次启动 kernel 模块构建；(c) 设备算子必须 `-cl-fp32-correctly-rounded-divide-sqrt`（Arc 默认 fp32 除法 2.5 ulp）+ 宿主 `-fp-model=precise`，否则量化算子与参考实现不逐位一致；(d) 大分配做"打 tag — 读回 — 重分配并加 spacer"的别名自检；(e) 设备侧对 host flag 的自旋必须有界（无界自旋会让 xe 驱动按节点重置 GT，卡到重启）；(f) KV append 的宿主拷贝用 4 字节 lane store 而不是 1 字节。
2. **Strata 出处**：`docs/INTEL.md:718-726`（>4 GiB；JIT 构建选项在 `sycl/CMakeLists.txt:65-68`）、`:287`（per_kernel split：首次 launch 245 ms → 1 ms）、`:184-188`（两个浮点 flag；warehouse `quantize_act_parity` 303k mismatches → 0）、`:747-756`（别名自检 `malloc_device_guarded`，每 64 KiB 打 tag、二次 kernel 读回、必要时加 spacer 重分配，<0.5 s）、`:708-717`（自旋上界 `STRATA_SYCL_SPIN_MAX`，修前 11/16 次首请求死/输出 token 0，修后 4/4 正确）、`:818-820`（kv_append 1 字节 store：4K prompt 里 2 s，改 4 字节 lane store）。
3. **Strata 实测**：>4 GiB（Arc A770 16GB，xe，Coder IQ1_M，`--stream-experts`，4.46 GiB 专家缓存）：cache 最后 0.46 GiB 的专家在 prompt 路径解量化成 NaN、prompt 全出 `!!!!`；加选项后 73-542 token 的 prompt 0 个非有限值（`STRATA_DBG_NAN=1`）；速度代价未测。别名（Arc Pro B70 32GB，xe，kernel 7.0）：22 GiB 专家 arena 里两个 2 MiB 页映射到同一物理页，prompt 在第 4 层 NaN；自检 <0.5 s 启动开销。自旋（A750 8GB，i915）：见上。
4. **为什么可能适用于 A770 16GB+64GB RAM + NF4/SYCL**：Aila 就是 Windows + DG2 + SYCL/oneDNN，其中 (a)(b)(c) 是**编译期选项**，代价极低、可能的收益是"消除一类随机 NaN/错字节"和"首请求更快"；(a) 尤其值得核对：Aila 有 USM ledger 与多 GiB 分配，若某个 kernel 参数指向分配内 >4 GiB 处，症状会与 Strata 一样是"特定槽位的权重解量化成 NaN"，极易被误判为量化 bug。差异：Strata 的别名自检发生在 **xe/Linux** 驱动，Windows 的 L0/IGC 行为不同，Aila 需要自己验证是否复现。
5. **预期收益量级** `[待验证预估]`：正确性类（消除一类隐性错值）：不可量化但风险高；(b) 首 token 延迟 −100~−250 ms；(c) 若 Aila 尚未加，可能解释现有量化边界 bug。
6. **移植成本/风险**：低。风险：`-ze-opt-greater-than-4GB-buffer-required` 在旧的 IGC 上可能不被接受（Strata 的规范要求"确认实际生效的编译选项并拒绝不支持的"），Aila 的 `build.ps1` 要打印实际使用的选项；(d)(e) 与 Aila 现有 USM 账本/取消纪律重叠，可能已经覆盖，先查再改。
7. **建议的 Aila 最小实验**：（a）探针：写一个 `AilaQwen4ExpKernelTests` case：在 5 GiB 分配的第 4.5 GiB 处放已知 bit pattern，kernel 读回比对（不用 UNIFIED；USM device 分配）；（b）门：现有 `ctest -L 'cpu|abi'` + 该探针；（c）实模：只改编译选项，`run.py` 看首 token 与 14K ttft（预期内）。

### C12. 短 chunk / 小 prompt 的 CPU 分担（`STRATA_PREFILL_CPU_SHARE`）

1. **机制（一句话）**：小块 prompt 的专家在 GPU 上要跨 PCIe 流式拷贝，而 decode 用的 CPU 池此刻空闲、RAM 里已经放着那些专家；引擎把"少数 token 会路由到"的专家交给 CPU 池，**逐层实测两侧耗时**（GPU 侧用 CUDA event 围住专家段、在下一次路由同步后读，CPU 侧计时），按 `g/(c+g)` 动态分配份额、只在该层"分担更快"时继续分担（`auto` 还会周期性做反向对照），也可以用固定份额。
2. **Strata 出处**：`docs/DETAILS.md:1564-1607`（规则、默认范围、精度代价、两张机器表）、`src/prefill/prefill.cpp:154-194`（`cpu_share_env/auto/max` 与默认的实测依据）、`:195-197`（分层切分时一个 chunk 只由一个 stage 拿池）。
3. **Strata 实测**（CUDA，10 轮交错中位数，`--expert-cache 1500`）：512 token 的 prompt −26%（RTX 5070/Ryzen 5 7600/Q2_0）、−28%（RTX 3060 IQ3_XXS）、−35%（Tesla P100 IQ3_XXS）；1,000 token −22%/−19%/−25%；2K/4K/16K 不变。RTX 5070 Ti（PCIe 3.0 x8）与 2 卡场景在 ≤3K token 上 −22~−49%。代价：首 token KL 均值 0.006/最大 0.026 nats，500/1000 token prompt 的 32-token 贪婪答案约一半完全相同、其余在约 23 个 token 后因近似平局分开；`=0` 恢复 0.1.40.3 的逐位输出。0.1.42 起在 RAM 紧张时自动不启用（含 Windows commit limit 检查）。
4. **为什么可能适用于 A770 16GB+64GB RAM + NF4/SYCL**：Aila 的 CPU 专家路径已经存在且"swap 关闭、staging 归零"，CPU 池在 prefill 期间本该空闲——这正是 Strata 的适用前提；Aila 有 64 GB RAM 与大量 CPU 线程（`AILA_Q4E_CPU_THREADS` 等），且 Aila 自测 CPU 专家内核"已近 DRAM 墙"，说明分摊的**效率**是有上限的，但小块 prompt 上"少拷一次"的收益可能更大。差异声明：Aila 的 prefill 是 4096 块（不是 512/1000 的小块），14K prompt 的**末块**（14K−3×4096≈1.7K）是唯一直接受益者；真要用在 4096 块上需要重测（Strata 的 `STRATA_PREFILL_CPU_SHARE_MAX` 扩展在 2,048 token 上输了 2%，见 `docs/DETAILS.md:1572`）。
5. **预期收益量级** `[待验证预估]`：14K ttft −0~−3%（只有末块受益，且 4096 块分摊未经验证）；若把 chunk 调小则与 Aila"chunk 4096 最优"的结论冲突，不划算。
6. **移植成本/风险**：中（要把 CPU 池接进 prefill 专家供给路径并做逐层计时仲裁）。风险：输出会因 CPU/GPU rounding 不同而变（opt-in）；Aila 的 4096 块与 Strata 验证过的 <1K 块不同，收益存疑。
7. **建议的 Aila 最小实验**：（a）探针：统计 14K prompt 各块 token 数（3×4096+~1.7K）与末块的未命中专家数，估算搬用 Strata 规则的上限；（b）门：`AILA_Q4E_CPU_*` 路径的逐位/容差口径；（c）实模：只对末块开，`run.py <tag> --agent`，ttft A/B/B/A。

### C13. KV 常驻/流式与"下一层 KV 预取"（只在长上下文才成立）

1. **机制（一句话）**：(a) 整条 KV 放 pinned 宿主内存、只有当前被 attend 的窗口留在 VRAM（`--kv-resident`）；(b) 预取：在当前层还在算的时候，用**另一条流**把下一层要用的 KV 缓存前缀上传，复用现有 staging 池（不新开 KV 缓冲），用 release event 保护 staging 池的上一个读者、ready event 排序下一个消费者。
2. **Strata 出处**：`docs/KV_PREFETCH.md:1-56`（机制、序、实测、未验证项）、`docs/INTEL.md:336-360`（`--kv-resident` 与 INT8/q4_0/k8v4 对照表）。
3. **Strata 实测**：KV 预取（2x modified RTX 3080 20GB，CUDA SM86/13.1，Linux，Xeon E5-2686 v4/94 GiB，PCIe 3.0 无 P2P，IQ3_S，131,072 ctx，INT8 KV，32,768 resident cells）：32K prompt 预填 −5.3%（16.989 → 16.082 s），64K −4.1%（32.604 → 31.259），8K 无收益、decode 无收益；profiling 显示约 939/1,553 ms 的串行 KV 上传等待被移出关键路径（拷贝仍发生，只是不挡算）。`--kv-resident`（Arc Pro B70 32GB）：256K decode 31 tok/s（此前 4-5），128K/256K 的 int8/q4_0/k8v4 对照表在 `docs/INTEL.md:340-347`。
4. **为什么可能适用于 A770 16GB+64GB RAM + NF4/SYCL**：Aila 的 KV 在 VRAM（int8 KV 已 opt-in），且 Aila 的槽弹性是 0.45-0.5%/100 槽——把 KV 挪去宿主能换出**十几个到几十个槽**（Aila 实测 int8 KV = 384 MB + 6 MB scales ≈ +156 槽 ≈ +0.7~0.9% decode），但代价是每层/每轮多走 PCIe；在 Aila 的 9.6 GB/s 链路上这与"staging 归零"的既有判断方向相反。因此**通用形式更值得搬**：把"下一层的宿主→设备数据搬运在另一条流上预取"用在**专家 staging**（C1/C2）上；KV 流式本身建议当作长上下文（≥64K）的独立 opt-in 备用方案。
5. **预期收益量级** `[待验证预估]`：14K 场景 ≈0 或负；若 Aila 支持 ≥64K 上下文，长 prompt 预填 −3~−5%（照 Strata 数字，需在 9.6 GB/s 链路上重测）。
6. **移植成本/风险**：中高（KV 布局与注意力 kernel 都要支持"部分在宿主"）。风险：与"全量 pinned"否证项相邻——KV 的宿主部分必须是小块且不锁全量；跨流预取的同步契约要与 Aila 的取消/drain 纪律一致（Strata 明说 early return 必须 drain pending copies）。
7. **建议的 Aila 最小实验**：（a）探针：量 14K/32K 下 KV 的 VRAM 字节与"若换成宿主可换回的槽数"，以及每轮 KV 的 H2D 字节；（b）门：预取流的正确性用现有 int8 KV 门（`q4e-int8kv-*`）复刻；（c）实模：≥32K context 的 agent 长读 + 短续轮。

## 2. 排序总表（价值 × 可行性）

| 序 | 候选 | 价值 | 可行性 | 主要依据（硬件） | 与已否证项的关系 |
|---|---|---|---|---|---|
| 1 | C1 prefill 双制度暂存 + stream-ahead | 高（Aila B3-3 的本体问题） | 中（Aila 已有 expert-major/prefetch 开关） | B70 40K 1106→1160 tok/s | 不改 chunk 大小；与 blocked 布局无关 |
| 2 | C4 路由画像工具链（mix + 留一 + 回存） | 中高（Aila 次选，工具链缺失） | 高（Aila 已有 trace/hotset/离线 sim） | B70 IQ3_S +5~+64%（均 +21%） | 与 P3 hotset 是增量，不是重复 |
| 3 | C8 方法学包（事件基准/A-B/逐卡表/反证/收据） | 高（防走错方向） | 高（零风险，立即用） | A770 微基准 3.3x→端到端 0 等 | 无 |
| 4 | C2 copy-engine 队列 | 中高（prefill 拷贝串行时） | 中（A 系要自测，有挂死史） | B70 4K 3.82→3.25 s；A750 仅 +1.4% | 与核内优化无关 |
| 5 | C3 16B 对齐装载（`load16_a2`） | 中高（取决于 NF4 步长） | 高（局部 kernel 改动） | B70 Coder decode 44.9→54 tok/s | 与 uint64/blocked/half-unpack 不同 |
| 6 | C6 route tail skip | 中（需 ≥2 行窗口） | 中（设备侧 rank 已有） | RTX 3060 1.156×；SYCL 未实现 | 与 MTP 门绑定，独立 opt-in |
| 7 | C5 adaptive 异步/重叠 + 交换旋转 | 中（回收 Aila 8% swap 成本） | 中（Windows 有负例） | A4000 +5.5~+8.3%；Blackwell +8.1% | 旋转的前置 pinned 与 Aila 否证项正面冲突 |
| 8 | C9 suffix-lookup + DraftPolicy | 中高（agent 编辑场景） | 中（受 B4-3 多行 verify 阻断） | B70 suffix-only 接受率 9%；编辑 +6~11% | 与"MTP 直接接入"不同 |
| 9 | C7 专家填充流水线 | 中（迭代速度/冷启动） | 高（大概率已有开关） | B70 76.2→18.8 s | 无 |
| 10 | C11 编译/稳健小项 | 中（正确性风险 + 首 token） | 高（编译选项级） | A770 >4 GiB NaN 等 | 无 |
| 11 | C10 batch-MTP 取舍 + batch exactness 配方 | 中（多会话） | 低（依赖 MTP 恢复） | 2x 3080 +15.4%；2x R9700 −21.9% | 不重复 MTP 直连；只搬取舍规则 |
| 12 | C12 CPU 分担（小 chunk） | 低（Aila 块太大） | 中 | RTX 5070/P100 −19~−35%（≤1K token） | 与 CPU 专家路径既有工作叠加 |
| 13 | C13 KV 常驻/预取 | 低（14K 场景） | 中高 | 2x 3080 32K −5.3% | 与"全量 pinned"相邻，须小块 |

## 3. 看过但判定不可搬（含原因）

1. **Batched DMA（`STRATA_DMA_BATCH`，CUDA 13 的 `cudaMemcpyBatchAsync`）**：`docs/BATCHED_DMA.md:3-13, 51-63, 88-93`。RTX 4070 Laptop 8GB/Windows/CUDA 13.4：16×2 MiB 的宿主提交中位数 0.0936 → 0.0162 ms（−82.7%），但**总传输只 −1.3%**；整机屏幕（Swift 1.5 IQ3_XXS, 64K）30.3 vs 30.0 tok/s，作者自己写"不构成改默认的证据"。SYCL 无对应 API，L0 immediate command list 已能一次挂多拷贝；Aila 侧等价收益应该在 C2 里量，而不是单独做一批 DMA。
2. **Foresight swap（`STRATA_FS_SLOTS`）**：`docs/DETAILS.md:1703-1711`。RTX 3060/Tesla P100/RTX 5070/2x RTX 3090 全部 ≤1.00（0.91-1.00），"更小的缓存比槽位收回的更多"。Aila 已把 swap 默认关闭。
3. **`STRATA_STAGE_PIN`（pinned stage buffers）**：`docs/DETAILS.md:1712-1719`。0.1.41 发布门发现 IQ3_S 在 4096-token prompt 后 decode 损坏（疑似 stage buffer 早于异步拷贝被复用），保持 opt-in。Aila 的"取消必须 drain"纪律下不值得冒这个风险。
4. **disjoint expert cache（`STRATA_DISJOINT_ADAPT`）**：`docs/DISJOINT_EXPERT_CACHE.md:1-19, 35-48`。需要第二个专家 tier（helper GPU/peer tier）；Aila 单卡，无此结构。可保留的细节：**"每个晋升候选向对方实时查询 `holds`，而不是复制一份加载时的掩码"**（掩码会过期）——若 Aila 未来加入第二个位置（如另一块卡/另一类内存），这条设计细节可复用。其历史数字（2x 改卡 RTX 3080 20GB，81.5 → 94.1 tok/s，+15.5%）文档自注"不是纯上游移植的实测加速"。
5. **`--pcie-frac` 与 PCIe 份额自动公式（L/(L+1.85R)）**：`docs/A770_SPEED_SPEC.md:205-242`（xe 上 `--pcie-frac 0.05` 被接受但 "PCIe 0.00"——计数器从不动，即该模式根本没接 PCIe 专家读路径；只有 pinned mirror 那条，而它在这台 A770 上因 PCIe 3.0 x4 慢约 5 倍）、`docs/DETAILS.md:1665-1689`（公式只在 CUDA/RDNA4 生效）。Aila 已有结论"staging 全 0、冷专家全走 CPU，在 9.6 GB/s 下是正确的"——**用 Strata 的公式验算会得到低份额，与 Aila 现状一致**，所以只需作一次书面验算，不建新路径。
6. **peer-device / 第二 GPU 专家 tier / SYCL 层切分**：`docs/INTEL_ARC.md:18-24`、`docs/A770_SPEED_SPEC.md:277-306`（`--peer-device` 在 SYCL 是 stub；`--layer-split auto` 配置成功但每次请求失败：`UR_RESULT_ERROR_DEVICE_LOST` + 跨 context 图提交错误；根因是端口只有单 context 图捕获）、`docs/INTEL.md:639-642`。Aila 单卡，且这条是"未移植 + 崩溃"。
7. **`STRATA_GDN_CHUNKED`**：`docs/DETAILS.md:1697-1702`。该内核要求一次波次 128 个 SM（RTX 5090 上 1.8x），小卡强制打开更慢（RTX 3060 0.73x、RTX 5070 0.87x）。Aila 已测过 GDN chunk mode 1（ttft −2.7% 但 decode −4~7%，维持 opt-in），结论同向。
8. **XMX / joint_matrix 全家**：专家点积 XMX（`STRATA_EXPERT_XMX`，3 个版本，1.4x/2-3x 慢，其中一个把 GPU 挂死，`docs/INTEL.md:427-431`）、融合 dequant+XMX GEMM（比 dequant + oneMKL 慢 4-5x，`docs/INTEL.md:299-305`）、Battlemage grouped XMX expert GEMM（慢 11-15%，`docs/INTEL.md:831`）、XMX prompt attention v2（比 FP32 fallback 慢约 2x，`docs/INTEL.md:362-372`）。与 Aila 已否证的"ESIMD DPAS 真实形状 / P4 SLM 解量化复用"同类；Aila 的 `AILA_Q4E_PREFILL_XMX`/`GROUPED_XMX_MIN_ROWS` 实验若还没收敛，这些数字可作为"预期为负"的先验。
9. **对齐/平面化 IQ 布局（llama.cpp SYCL PR 的 plane repack）**：`docs/A770_SPEED_SPEC.md:308-324`。现状的 Split<> loader 已经按字（不是按字节）读；PR 的变化只剩"plane 重排"，那是 loader+cache+验证三处改动，而且收益上限 ~2-3%（专家核只占窗口约 20%）。与 Aila 已否证的 "blocked 布局 / 字节对 LUT" 同类；本报告 C3（不改布局、只改装载）是另一条线。
10. **MMVQ_IL 表（interleaved small-batch matvec）**：`docs/MMVQ_IL_TABLE.md:1-32`、`src/kernels/cuda/native_mmvq.cu:2543-2606`。它是 CUDA/HIP 专用（按 sm_60/70/86/89/120 分表），**SYCL 端口没有移植**；Aila 的对应色（2 rows/subgroup）已被否证。只借方法论：逐卡实测、3% 发射规则、类内不一致取 0、`STRATA_MMVQ_IL_ROWS` 运行期覆盖、`--bench --emit-table`。
11. **WDDM 下 pinned arena 的 8 GiB 上限**：`docs/MULTI_GPU.md:217-220`（Windows/WSL2 下专家 arena 只 pin 8 GiB，更多会让 WDDM 拒绝分配；`STRATA_ARENA_PIN_GIB=N` 控制；Linux 无此限）。Aila 的"51 GiB 锁页 → 首请求 `UR_OUT_OF_RESOURCES`"与此完全同向——**把这条当作 Aila 否证结果的外部佐证**，不要再试全量 pinned；若将来要试，上限参考 8 GiB 并只 pin 子集。
12. **已测过的默认/开关结论（与 Aila 已否证列表重合，不重复建议）**：`--adapt-swaps 0` 更差（保持 8，`docs/A770_SPEED_SPEC.md:250-252`）；缩小专家缓存 neutral/更差（`:573-595`）；`--kv q4_0` 对该 A770 无益（`:266-269`）；`--spec-split`（草稿与 verify 重叠）在 A770 上是负收益（`:254-258`）；`STRATA_EXPERT_LANES` 8/16 无差（`:597-609`）；`STRATA_MMVF_ROWS=1` 1.4-6x 慢（`:510-519`）；GRF 128 no-op / 256 慢 10-58%（`:521-533`）。
13. **`STRATA_PREFILL_FIRST`（短首块先启动 GPU）**：`docs/INTEL.md:291`。它**移动 expert GEMM 的 rounding**，长贪婪续写会在约第 45 个 token 分叉；Aila 的逐位门与续轮口径会因此变红，且 Aila 已测 chunk 4096 为最优（chunk 变小 ttft 变差）。不搬。
14. **KV 格式变体 k8v4 / q4_0**：`docs/INTEL.md:340-347`。k8v4 在 B70 上 prompt 最快（128K：1,006 vs int8 888）但需要专用 kernel 且不支持 KV 流式；Aila 的 int8 KV 已 opt-in 且端到端约 1%，14K 场景下收益不足以支撑新 KV 格式 + 新注意力路径。留档备查。
15. **`--prompt-cache-tail` / `STRATA_CACHE_MESSAGE_BOUNDARY`（对话边界/尾检查点）**：`docs/PROMPT_CACHE_TAIL.md:1-33`、`docs/MESSAGE_BOUNDARY_CACHE.md:1-19`。它们解决"编辑最后一条 user 消息后复用长历史"，但 Aila 已有前缀缓存与续轮复用（实测续轮 `cached_prompt_tokens=14072`、first content 1.486 s），且这两项都是 opt-in 且改 chunk 边界/加快照成本（Strata 自注为非普遍加速）。不搬。
