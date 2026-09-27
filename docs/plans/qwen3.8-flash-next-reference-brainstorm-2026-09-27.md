# Qwen3.8-Flash-Next / A770：技术报告、vLLM 与 llama.cpp 驱动的第二轮 brainstorm

日期：2026-09-27。Aila 阅读基线：`69d52c67fd7c694fbf213086a70175e9f8eb3424`。本轮在 worker 运行期间进行，只读取代码与资料；未构建、运行模型、调用 GPU 测试、修改运行时代码或向 worker 下发任务。只新增本报告与来源清单，不改动 worker 正在维护的实施记录。

**建议把新一轮优先级放在：PLE 确定性提前搬运、相邻 GR 写回/归一化融合、GDN state 的片内驻留，以及 QSA 的有界批处理。** 压缩 K 和 K-only 投影已经有实现，不再把它们列为待从零开发的功能。更激进的 FP8、跨 token 索引复用和训练结构调整需要独立的精度边界，不能混入无损优化。

本报告是设计补充，不是对 worker 最新成果的全面验收。[上一轮 brainstorm](qwen3.8-flash-next-a770-review-and-brainstorm-484b086.md)中的缓存、需求优先预取、NF4/XMX 方向仍有效；本轮重点增加可由新资料支持的实现细节和取舍。

## 1. 来源及结论适用范围

| 来源 | 本轮读取版本 | 用途 |
|---|---|---|
| 官方技术报告 | 本地 PDF《On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability》，2026-08-26，28 页 | 模型语义、架构取舍、性能数据的实验条件；重点完整读取并查看第 5、6、9、14、15 页，文字核对第 7、11 页 |
| vLLM | 本地 `3137ff07739f0cd19e06119506c6432a4a7e77ad`，开始读取时干净 | QSA bounded prefill、paged/circular side cache、GQA Split-K、GR 跨块融合、PLE side-stream prefetch |
| llama.cpp | 本地 `95887577ab5fead779581a7030a83c7752ff3234`，开始读取时干净 | Qwen4Exp 图与 hybrid cache 生命周期、SYCL GDN state 分片、实现边界与反例 |
| Aila | `69d52c6` | 已有改动与剩余工作、DG2/SYCL/NF4 约束 |

PDF 原文件：`D:/Downloads/Qwen3.8-Flash-Next-tech-report.pdf`。来源哈希和精确文件入口保存在[来源清单](qwen3.8-flash-next-reference-brainstorm-2026-09-27-sources.json)。没有联网更新或切换两份参考仓库；这是用户提供的本地快照，不声称其为远端最新版本。

标记约定：`[REPORT]` 为报告所述；`[SOURCE]` 为参考实现事实；`[AILA]` 为当前代码事实；`[DERIVED]` 为尺寸/依赖推导；`[DESIGN]` 为尚未测量的候选。引用某引擎支持某功能，不代表已证明该引擎在 A770 上更快或更准确。

### 三个会改变方案优先级的事实

1. `[REPORT]` 第 14–15 页明确指出 PLE 的第 2 层放置有利于与第 1 层计算重叠；查表地址只由 token 的短历史决定。它比预测下一层专家更适合先做可确定的预取。
2. `[REPORT]` 第 14 页记录 GR 分支稀疏化虽在预训练指标上接近，后训练后质量明显退化，未被采用；FP8 residual storage 则被报告为质量损失很小。二者不能被统一归类为“省带宽就等价”。
3. `[REPORT]` 第 9 页图 6 的 7.6× prefill / 4.9× decode 是 1M context 的 attention-module 结果，含 indexer；prefill 为最后 16K chunk、BS=1，decode 为 BS=4、next_n=4（含三个 MTP steps）。不能拿来预测 A770、2K/32K、单 token、NF4 offload 的全模型收益。继续保持本项目“不做 MTP”的范围。

报告第 6 页定义的是先 raw K pooling、再 norm/RoPE，并且完整 block 才进入竞争、未满 block 的 tail 强制保留。不得把“先旋转再池化”、滑动窗口替代 Top-K 或缩小 token budget 当作实现优化。

## 2. 相对当前进度的差异

`[AILA]` 当前代码已有 causal QSA prefill 的逐 query 路径、radix Top-K、cache failure contract、稠密 workspace 边界、session compressed K 和短上下文 K-only。以下只讨论继续演进，不重复开同名工程。

| 当前路径 | 新资料提供的增量 |
|---|---|
| PLE 到 layer index 1 才计算 hash、CPU 解码、H2D | 将地址准备和 embedding 搬运前移，计算仍在原 PLE 层消费 |
| `hc_inject_add` 后下一块再 `group_rms_norm`；decode 已有 down+inject / up+mix 专项融合 | 融合相邻 GR 的写回与下一次 norm，而不是重复已有内部融合 |
| GDN prefill 已有单 kernel 的 token 循环，循环内多次读写 global S | 跨 token 保存 state 分片，或对长 M 引入 chunked delta recurrence |
| QSA prefill 逐 query 调 selection 和 attention | 有界 query tile、共享 KV 的多 Q head、随并行度变化的 Split-K |
| full raw indexer K 和 compressed K 同时保留 | raw K 缩成 compressor ring，但回滚需要单独的 tail checkpoint |
| 内存核算开始显式记录 KV/indexer/recurrent | 让 prefill 临时区、resident experts、snapshot、raw ring 共同竞争显式预算 |

精度边界仍须保留：当前 `tests/models/Qwen4ExpModuleTests.cpp:2670+` 的 F4 fixture 使用真实 checkpoint 的 Q/K norm 权重，但 QK 与 raw K 是随机生成；检查了 QK 投影权重存在，却没有通过真实 hidden state 计算整条 indexer。实施记录中的 511/512 集合一致不能解释为真实长 prompt 的端到端精确对齐，也不自动授权更多 FP8。这里只校准证据口径，不追加昂贵验证。

## 3. 候选 R1：把 PLE 拆为 prepare / transfer / consume

**证据。** vLLM 的 `nvidia/model.py:469–542` 提前启动下一层 PLE；`common/ngram_embedding.py:503–535` 在 side stream 查表，消费时等待完成。当前 Aila 的 `Qwen4ExpPle.cpp:370–467` 将 history 更新、分配 ID 数组、CPU NF4 解码和传输集中在 `forward`；backend `:962–974` 到 PLE 层才调用。

**[DESIGN] 第一版只移动确定性工作。**

在 token IDs 已知且本次请求获准执行后，建立一个固定大小 `PleTicket`：session/generation、起始长度、输入 IDs 身份、候选的新 n-gram history、固定 staging slot 和 ready event。主机线程开始 layer 0 提交时，让专用小任务执行 hash/gather/dequant；layer 1 只等待 ticket，然后执行依赖当前 hidden 的 PLE 投影与融合。

注意当前 `compute_ngram_ids(state,...)` 会直接修改 `prev_tokens`。提前执行必须分为纯 prepare 和成功后的 commit，否则取消、重试或 truncate 会把历史推进两遍。暂缓提交新 history，或明确失败会重置 session；绝不能“多调用一次 forward”来实现预取。

`[DERIVED]` 当前每 token 读取 16 行，每行 NF4 packed 80B + 5 个 f32 scale 20B，即约 1600B 有效数据（不计共享 qmap），解码后 H2D 是 2560×2=5120B。实际 mmap page fault 可远大于有效字节，最初应优化 page-ready deadline 和重叠，不为几 KB payload 先改量化格式。

**两个实现层次：**先沿用 CPU 解码到 BF16 pinned staging；只有该段成为瓶颈，才试 CPU gather packed row、GPU 解码或小范围 GPU hot-row cache。vLLM 是 pinned/UVA + BF16/FP8 表，Aila 是分片 mmap + NF4，不能直接照搬全表 pinning 或假设 Windows/SYCL 的直接主机读取同样划算。

预分配 2 个 staging slot，每个 slot 必须保留 H2D source-reader completion；layer0 专家 demand 与 PLE 共享 IO/CPU 资源时，PLE 只占一个有 deadline 的小 credit，不能为它让大量 speculative expert prefetch 抢先填满队列。最终只应隐藏 `min(可重叠时间, PLE 准备时间)`，不宣称消除全部 PLE 时间。

**最小决策。**先复用 EOS/hash 用例，补一次取消重试或跨 chunk history 对照；性能只看同一短 prompt 的 PLE exposed wait 和 TTFT/TPOT。若 CPU 任务唤醒成本高于 warm lookup，保留短路径同步执行，只对冷页或大 prefill 异步。

## 4. 候选 R2：跨相邻 GR 的 combine + norm

**证据。** vLLM `hyperconnection.py:151–189` 的 `combine_and_mix`，以及 `ops/hc.py:273–345` 的 combine/norm kernel。后者先把 combine 结果 cast 回输出 dtype，再参与 norm，以保留未融合路径的舍入边界。`model.py:290–298` 在 PLE 修改 widened residual 前显式物化 pending combine。

**[DESIGN] 建立最小 pending residual 状态。**

在 attention/MLP 完成后，暂存 `block_output + injection`，直到下一 GR read 时一起执行：加载旧 hidden4、计算写回、舍入 BF16、存回 hidden4，并从该 BF16 值直接算下一块 group RMSNorm。下游 down/inject 和 up/mix 仍使用现有优化。

先只融合层内 attention→MLP 的 48 个固定边界；稳定后再跨 MLP→下一层 attention。PLE、embedding override、多模态插入、debug dump、对外可见状态等需要完整 residual 的位置显式 flush。由于 `buf_.y` 和 injection scratch 会复用，pending descriptor 要在覆盖前消费；如引入 execution slot，多 slot 各自持有 pending 状态。

这与被否定的 GEMV residual-add 融合不同：不把 residual 写回塞进长点积 kernel，不增加 GEMV accumulator 的存活范围。融合点是相邻 elementwise/reduction。

**[DERIVED] 收益上限校准。**每 token hidden4 是 10240 个 BF16，单遍读取 20KiB。一次 combine/norm 融合主要省掉一个完整 residual 重读和一次 launch，仍需存 residual 与 norm 输出。HC down/up 权重每模块约 `2×10240×320×2=12.5MiB`；整个模型有 97 个 HC 模块，权重规模远大于单 token residual。不能因为论文说 GR 受 residual traffic 影响，就推断 A770 decode 的整体瓶颈也是 residual；小 batch 更可能先得益于 launch 减少，大 prefill 才更值得关注激活带宽。

**最小决策。**一个 GR 边界比较 BF16 materialized hidden 和 norm 输出；一条经过 PLE 边界的短序列确认 flush。若 register/SLM 增加抵消读写节省，缩小到层内融合，不把整条 layer loop 一次改成复杂惰性执行。

## 5. 候选 R3：GDN state 驻留，比重复“合并 launch”更进一步

**证据。** llama.cpp `ggml-sycl/gated_delta_net.cpp:54–67, 142–161` 将每列 state 的行分片放进 `s_shard[]`，token 循环后写回；可选短回滚 snapshot slot。该文件 `:193` 同时仍写有 chunked kernel 的 TODO，不能称其 SYCL 版已提供成熟的长 prefill chunk 算法。

`[AILA]` `GatedDeltaNetOps.cpp:443+` 已把 token 循环融合在同一 kernel 内，但 `:606+` 的 decay/dot 和 update/output 两段反复访问 global S。若一次矩阵扫描各读写一次，两段逻辑 traffic 约为每 token/层 `48×128×128×4×4=12MiB`，36 层约 432MiB；这是逻辑 load/store 数量推导，不能当成实测 DRAM 流量，缓存与编译器会影响实际代价。

**[DESIGN] 双层策略。**

- 中短 prefill：让 subgroup/多个 subgroup 合作处理少量 value columns，把每 lane 持有的 state shard 控制在小范围，跨 8/16 token 的候选 tile 复用，再统一写回。q/k/beta/decay 在安全范围内共享；保留 output gate、conv history、Q/K L2 norm 的现有数学顺序。
- 长 prefill：再考虑 chunked delta-rule，将块内变换与块间 recurrent carry 分开，利用矩阵运算密度。参考报告第 5 页 FlashQLA 的方向，但本轮没有读取 FlashQLA 仓库，也没有证据可直接移植其 TileLang 或性能常量。

不要简单把当前每 lane 的 128 个 f32 S 值全改成私有数组；GRF spilling 会把问题搬回内存。也不默认把完整 64KiB/head state 塞 SLM：容量和 occupancy 可能使其更慢。llama.cpp 的 warp mapping 要重算成 A770 的 subgroup 布局。

**最小决策。**只试一个 state tile、一个中等 M，比较最终 recurrent state、全部输出和后续一个 decode token。若 spill 或 wall time 无收益即停止该映射；不同时探索 GRF 模式、SG 大小、chunk size 的全排列。

## 6. 候选 R4：QSA 的计算接口按预算和并行度设计

### R4a. Batch query，按 byte budget 控制 logits

vLLM `ops/qsa_indexer.py:578–651` 用 `rows_per_chunk=max_logits_bytes/(logits_width×4)` 划分 prefill；width 随本 batch 最长历史的 compressed block 数变化，而非固定最大上下文。Aila 当前逐行 reference 可以保留为小规模 oracle，生产候选改为固定 arena 的 query tile。

`[DESIGN]` 分配 `[query_tile,max_visible_blocks]` 的 f32 score、`[query_tile,512]` 的 block IDs 和逐 query valid count；同一 tile 做 QK→score→精确 Top-K→sparse attention。每行独立限制 visible complete blocks 与 tail，不能让早期 query 使用本 chunk 后面才完成的 block。

`[DERIVED]` 32K context 有 8192 个 compressed block，128 query 的 f32 logits 仅 4MiB；256K 时相同 tile 为 32MiB。因此 query tile 应由预算而非固定 prompt size 决定。可从保守的少量 query 开始，不照搬 vLLM 的默认 MB、Triton tile 或运行时 `torch.empty`；Aila 在 load/admission 时预分配。

### R4b. GQA 以 KV head 为共享单位

vLLM `ops/qsa.py:543–591, 693–737` 的 grid 是 `(query rows, KV heads, splits)`，一个 program 处理共享 KV 的 Q heads，Split-K 随基础 program 数选择。其 dispatch 明确针对 GB300、SM120、Hopper，不能移植常量。

当前模型 Q/KV=24/2，意味着每个 KV head 对应 12 个 Q heads。Aila 可尝试每 tile 处理其中 2/4 个 Q heads，重用同一 indexed K/V 载入；query 很少时增加 KV 维度 splits 提升并行度，prefill query 足够多时优先降低 splits，减少 f32 partial/merge 流量。保留现有稳定 softmax 与 gate 舍入，不把全部 12 heads 强塞同一工作组。

原 head256 partial-merge 已存在，新收益应来自 **跨 Q head 的实际数据复用与按工作量选择 split 数**，不是重复实现 partial-merge。

### R4c. 近阈值采用精确“补集选择”

这是本轮独立 `[DESIGN]`，不是资料中的现成实现。若可见完整 block 数 B 仅略大于 K=512，找 top-512 等价于找 bottom-(B−512) 后保留其余项。例如 B=513 只需精确排除一个最小值，不必做通用多轮 radix selection。

使用与当前 radix 完全一致的总序：分数高者优先，同分小 ID 优先；补集应先删较大 ID。保证 NaN/Inf/±0 策略一致，选后按既有顺序展开，tail 独立追加。先只做 B=513 的单 minimum-reduction 特例，测不到收益就不扩大到更多排除数。

这仍需要计算所有 block scores，不是免去 indexer dot。收益在 selection 成本。远离阈值继续通用 radix，避免用浮点阈值近似重新引入上一轮已修的问题。

### R4d. 合并固定的 pre-indexer 工作

vLLM `indexer_qsa.py:282+` 将 Q norm/RoPE、raw cache 写入、边界 block compression 组织为 fused pre-indexer 路径。Aila 在 A1/A2 成熟后，可以把短阶段的 K 写入与“本 token 刚好封闭一个 block”的压缩合并；长阶段合并 Q normalize/rotate 与必要 cache update。

依赖纪律比 kernel 数更重要：跨 chunk 的旧 raw tail、新输入和 block-start position 必须一致；不能为了融合跨工作组读取尚未写完的 raw K。先让完成一个 block 的工作组直接读取旧 tail + 本次新输入，不通过全局 barrier 假设。

## 7. 候选 R5：raw indexer K 从完整历史缩为可恢复 ring

**证据。** vLLM `common/qsa_cache.py:837–869` 分离 `CircularBufferSpec` raw compressor state 与按 `tokens_per_state=4` 管理的 compressed cache。它对 speculative 写入还扩展 ring 防止覆盖已提交 tail；本项目不做 MTP，无需照搬 speculative 宽度，但 rollback 约束仍成立。

`[AILA]` 目前 session 保存 `[L,128]` BF16 raw K 和 `[L/4,128]` BF16 compressed K。压缩结果不可逆，因此删 raw 历史后，Truncate 到任意旧 block 中部无法凭 compressed K 恢复原始 1–3 行。

**[DESIGN] 与现有 bounded snapshot 结合。**每个可恢复 checkpoint 保存：logical length、compressed valid count、未满 block 的 raw K 行、对应 position、generation。正常 forward 只保留足够容纳当前 open group 的 ring；prefill 先从本次输入完成压缩，只把最后的尾部复制入 ring。Rollback 仅恢复已承诺的 checkpoint，再按现有 replay 契约处理目标位置。

不要以为 ring capacity=4 就自动正确。一个 batched write 可能先覆盖旧 open-group token，因此边界 block 必须在覆盖前消费，或使用暂存 tail。若未来要求任意位置零重算回滚，则需保留更多 raw 历史，不能无条件启用该节省。

`[DERIVED]` 12 层 raw K 每 token 3072B：32K 历史为 96MiB，256K 为 768MiB；ring+少量 snapshot tail 可释放其中绝大部分。以当前专家 slot 2,534,400B 估算，理论空间分别约相当于 39/317 个完整专家 slot（须向下取整、扣除新增 arena/headroom）。这只是容量换算，不能把新增 slots 直接换算为命中率或 tok/s。

llama.cpp 的 `llama-memory-hybrid-idx.cpp` 把 attention、recurrent、indexer 视为需要同步维护的序列状态；这一 ownership 思路值得借用。其 `set_input_qsa` 每 ubatch 遍历 cache cells 并构造映射，源码也注明 O(n_kv) 成本。Aila 当前连续单序列应维持简单的增量逻辑，不为了通用分页先引入每 token 的全历史 CPU 扫描。

## 8. 精度实验池：FP8 必须分清对象

| 对象 | 来源与潜在收益 | Aila 决策 |
|---|---|---|
| widened residual | 报告第 14 页支持 FP8 storage；相对 BF16 激活字节减半 | 实验项。先做 R2 无精度变化的融合；decode 每 token residual 很小，prefill 更可能受益。A770 上量化/反量化和指令代价必须计入。 |
| 主 attention KV | vLLM `nvidia/qsa.py` 支持 BF16/FP8-e4m3 main KV | 实验项。32K 时 12 层 main KV 约 768MiB，FP8 payload 可省约 384MiB，额外 scale/metadata 另算。需看 attention 误差与释放空间后的总收益。 |
| compressed indexer K/Q | vLLM `indexer_qsa.py:147+` 支持 BF16 或无 scale e4m3 indexer 表示 | 风险更高：很小 score 差异也会改变 selected set；当前已有近边界差异，不能把主 attention 的容差直接套到 Top-K。 |
| GDN recurrent state | 两份当前实现主要为 f32 state | 本轮不建议先降精度；长期递推误差与输出累积不同于只读 KV。先改变驻留方式。 |

任何 FP8 方案都不是把 NF4 权重换成线性 U4，也不代表 A770 有 CUDA FP8 tensor-core 同等能力。只借鉴存储布局/预算，算力路径单独选。

可以保留一个理论研究项：用低精度 score 粗筛、对 cutoff 附近候选以参考精度重算。但除非能给未重算候选提供**保守误差上界**，固定取一个“更宽 Top-K”仍不能保证精确集合。没有上界就必须标成近似算法，暂不作为默认方案。

## 9. 两个系统层面的设计补充

### 同时看专家缓存、计算 tile 和输出尾部

`[DESIGN]` workspace 节省的价值不仅是少分配内存：它可能增加可安全驻留的专家，降低每 token 的 H2D/IO。反过来，为 QSA/GDN/XMX 增大 tile 可能让专家 miss 增多。实验记录只需并列三项：wall time、实际 scratch bytes、专家 miss/transfer bytes，不给局部 kernel 单独授予“更优”结论。

长 prefill 可逐步引入层级 micro-tiling：在一个 layer 内用有限 activation tile 处理 query，保留该层专家热度；但避免将整个 transformer 按很小 token chunk 反复走完，导致相同专家跨 chunk 重复从 host 迁入。决定 tile 时同时考虑 QSA score arena、GDN carry、MoE 每专家 token 数和 snapshot 频率，而不是四个子系统各设一个互相冲突的 chunk 参数。

### 把执行资源合同固定下来，graph 留在后面

vLLM 的选择输出把 valid count 与 indices 放在同一 row，方便 compaction、padding 和消费端界定循环。Aila 可采用预分配 `SelectionBatch{indices,counts,logical_positions}`，明确每行长度；不要依赖旧 buffer 中没被本次写到的条目。

这是后续 static graph/execution slot 的基础。CPU PLE 准备、专家 IO 和动态 admission 保持独立；图只覆盖地址稳定、依赖明确的 GPU 片段。vLLM 的 CUDA PDL、Triton warmup profile 和 graph capture 约束不是 DG2 的可移植 API。

## 10. 不直接照抄的部分

- **llama.cpp QSA 的图兼容路径。** `qwen4exp.cpp:663–688, 730–764` 将 block scores 展开到 token 维并构造 mask 后交给通用 attention。可研究其 logical/physical cache 映射，但不能将这种物化大 mask 的方式当作 Aila 长上下文性能目标，也不能未经 tail/tie 边界比对就把其 token Top-K 当独立正确性 oracle。
- **vLLM 的 CUDA 阈值。** `base_programs<=24`、SM90/SM120 等仅表明派发维度值得借鉴，不是 A770 的边界。无需在本轮跑跨平台 benchmark。
- **GGUF 的 quantized GEMV/GEMM。** llama.cpp SYCL `mmvq/mmq` 处理 GGML Q4/IQ 等格式。它们与 BNB NF4 的 LUT、nested scale 及布局不同；不能移植名字就宣称支持同一种 4bit 量化。
- **whole-table pinned PLE。** vLLM 的硬件/RAM 条件不同；Aila 的 mmap 模型与专家、PLE、OS page cache 共同占用 RAM。先小 staging 和定向预取，不 pin 整个 51B embedding 的展开表示。
- **改训练结构。** 增大压缩比到 8/16、减少 QSA 层/heads、跳 GR branches、跨 backbone 层复用 Top-K 不是无损加速。报告的消融说明它们可能需要重新训练，且有质量折中。
- **以 cutoff 小解释一切。** 近阈值集合差异是一种机制说明，不是自动通过验收的理由。保留精确参考路径和明确的近似模式。

## 11. 推荐交付顺序与有限实验

| 顺序 | 可单独合入的交付 | 一个最小实验 | 推进条件 |
|---|---|---|---|
| P0 | `PleTicket` + 提前 hash/gather；历史提交与 DMA slot ownership | 一次跨 chunk/EOS 与取消重试；一组 warm/冷页有区分的时间记录 | layer1 exposed wait 下降，layer0 demand 没有被挤占 |
| P1 | 层内 attention→MLP combine/norm | 一个边界的 materialized BF16/norm 对照，再看短 decode wall time | 没有舍入漂移，launch 节省没有变成 GRF/SLM 回退 |
| P2 | QSA batch arena + 同 KV head 多 Q head + 少量 split 候选 | 一个跨界 chunk，随后一个中等长历史；分开 indexer/attention 时间 | causal sets、tail、gate 都正确，整体 prefill 获益 |
| P3 | GDN state shard 驻留原型 | 一个 M、一个 tile，比较最终 state 和后续 decode | 无 spill 回退且 state 精度契约成立 |
| P4 | raw K ring + checkpoint tail | truncate 到 block 内再 append | 释放真实预算且回滚与原路径一致 |
| P5 | B=513 精确补集特例 | 513 blocks，包括同分与 tail | 实测优于通用 radix，否则不扩展 |
| 研究池 | FP8 residual/KV、chunked delta、误差有界 score refinement | 被剩余瓶颈触发时才做 | 单独精度验收，未测不默认启用 |

此顺序是风险/可落地性排序，不是测得的性能排行榜。若 worker 接下来主要解决长 prefill，P2/P3 可以前移；若主要目标是短 decode，则先 P0/P1 和原计划的真实路由 event 拆分、demand-first cache。

保持验证克制：本轮不启动任何表中实验。未来一次只改变一个结构变量，先用已有小型 oracle；不因读到论文倍率就跑整模型长压测，不为每个候选创建独立庞大测试框架。参考仓库提供的是缩小搜索空间的设计证据。

## 12. 可直接接手的源码地图

| 主题 | 参考入口 | Aila 落点 |
|---|---|---|
| PLE overlap | [vLLM 模型调用点](../../tmp/vllm/vllm/models/qwen4_exp/nvidia/model.py)、[pinned lookup 与 side stream](../../tmp/vllm/vllm/models/qwen4_exp/common/ngram_embedding.py) | [PLE](../../src/models/Qwen4ExpPle.cpp)、[Backend](../../src/models/Qwen4ExpBnb4Backend.cpp) |
| GR 跨块融合 | [GatedResidual](../../tmp/vllm/vllm/models/qwen4_exp/nvidia/hyperconnection.py)、[combine/norm kernel](../../tmp/vllm/vllm/models/qwen4_exp/nvidia/ops/hc.py) | `RunHyperConnection` 与相邻 `hc_inject_add` |
| QSA batch / Split-K | [indexer](../../tmp/vllm/vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py)、[sparse GQA](../../tmp/vllm/vllm/models/qwen4_exp/nvidia/ops/qsa.py) | [QSA](../../src/ops/QsaIndexerOps.cpp)、[Attention](../../src/ops/AttentionOps.cpp) |
| Compressor ring | [QSA cache specs](../../tmp/vllm/vllm/models/qwen4_exp/common/qsa_cache.py)、[indexer lifecycle](../../tmp/vllm/vllm/models/qwen4_exp/nvidia/indexer_qsa.py) | Session raw/compressed cache、CaptureSnapshot/Truncate |
| GDN state | [llama.cpp SYCL](../../tmp/llama.cpp/ggml/src/ggml-sycl/gated_delta_net.cpp) | [GatedDeltaNetOps](../../src/ops/GatedDeltaNetOps.cpp) |
| Hybrid state contract | [llama.cpp hybrid indexer](../../tmp/llama.cpp/src/llama-memory-hybrid-idx.cpp)、[Qwen4Exp graph](../../tmp/llama.cpp/src/models/qwen4exp.cpp) | Session ownership / logical length / rollback |

本轮没有复制第三方实现进入运行时代码；将来移植代码时保留其文件的许可证和来源。当前可执行建议均已写成 Aila 的依赖、内存与精度约束，而非要求换成另一个推理引擎。
