# Qwen3.8-Flash-Next / A770：增量验收与系统优化 brainstorm

> 实施进展：F1/F2/F3 已在后续工作树中落地，定向测试通过。参见 [W0 首批实施记录](qwen3.8-flash-next-a770-w0-implementation-484b086.md)。下文仍是 `484b086` 原始验收与设计，不代表实施后的代码状态。

日期：2026-09-26。代码基线：`484b086c6c202a4b7ff2e4dd0f05eb48b2528154`；对照上一轮验收 `05055aa`。开始时工作树干净。

**结论：接受 N0 的主要正常路径修复与 N2 去同步实现；M2 只能验收为“decode 稀疏注意力原型及算子测试完成”，不能认定完整长上下文语义已经完成。** 下一轮首先补齐 QSA prefill 和精确选择，再把预算管理、专家数据路径与计算调度作为一个整体优化。S1/S4 仍保持候选，不凭旧日志升为默认。

本轮仅静态勘察代码、本地 Transformers 参考实现和已有日志，另做一次 CPU `single` 浮点阈值反例核对；没有构建、运行 CTest、实模生成或 GPU benchmark，没有修改运行时代码。沿用 A770-Kernel-Optimization 的证据约束，所有新优化均为设计假设，没有承诺加速倍数。

- `[CODE]`：当前代码可确认；`[LOG]`：既有日志；`[DERIVED]`：从代码/尺寸推导。
- `[CPU-CHECK]`：本轮轻量 CPU 核对，不能替代 GPU 复现。
- `[HYPOTHESIS]`：brainstorm 方案，需满足列出的条件后才推广。
- [上一轮验收](qwen3.8-flash-next-a770-acceptance-2026-09-26.md)保留历史结论；[本轮证据清单](qwen3.8-flash-next-a770-review-484b086-evidence.json)记录文件哈希、已有测试和反例。

## 1. 增量验收

| 项目 | 当前事实 | 判断 |
|---|---|---|
| N0 host DMA reader | 新增 `host_slot_in_batch_readers_`，prefetch 命中独立加入 batch reader；DMA 完成事件写回 host slot | 原 R1 的正常路径已修；原组合用例已加入。测试未人为卡住 DMA，不夸大为确定性覆盖所有交错。 |
| N0 move | move 前 drain 源对象；移动新增 reservation/event 成员，不再清空目标 LRU | 原 R2 已修正常 move；设备故障下仍有吞异常策略，另列故障契约。 |
| N0 64 parts | `atomic<uint64_t>`、`uint64_t{1} << p`；有边界及 64-part synthetic 用例 | 原 R3 已修。 |
| 小 cache fallback | 前 token down event 等待后才 acquire；层尾排空 | 正确性修补可接受，序列化只发生在 fallback。 |
| fill pool handoff | worker admission、任务领取和 active 计数统一受 mutex 保护，cancel 先关闭 admission 再 drain | 原竞态窗口已关闭；每 part 拷贝 `std::function` 和锁竞争仍值得后续减轻。 |
| 异常安全 | AcquireGuard 清 reservation，host victim 的正常等待不再吞错误 | 仍非事务安全；见 F3。 |
| N1 workspace | S1/S4 专用大 buffer 按 mode 分配，记录实际模式；profiling 脚本写 sidecar | 完成第一步；尚无统一长上下文预算，manifest 也非完整运行身份。 |
| N2 async dequant | Q4E 默认允许去除 on-demand dequant 后 host synchronize；fused rows 继承标志 | 同一 in-order queue 的 oneDNN stream 支持该依赖推理；取消的是 host wait，反量化本身仍执行。 |
| M2 QSA | 12 层 indexer 权重、raw K cache、decode block score/top-k/稀疏 gather attention 已接入 | 接口与代表算子测试可接受；prefill 缺失、Top-K 非精确、独立数值对齐不足。 |
| S4 | 仍为 expert-major 排列与每 activation 独立 GEMV | 仍未实现显式多 token 共享权重 tile，也没有 XMX。 |

### 已有证据的实际边界

`build-refactor/Testing/Temporary/LastTest.log` 记录 22:05 的四个目标通过：ModelSpec 0.04 s、Kernel 18.77 s、Module 11.26 s、SafeTensors 0.40 s。不是全部 57 个目标通过。本轮没有重跑。

QSA kernel tests 包括普通、并列/全零、较大 top-k，以及非连续索引、gate 和 baseline attention 路径，后两者确有数值参考比较。Module 中名为 end-to-end 的用例是**synthetic 算子串联**：score 和选中 token 对照自写 host 实现，最终 attention 检查有限值；真实模型部分检查权重存在和形状。不能把它解释为真实模型 backend 的 2400-token prefill/decode 与 Transformers 对齐。

`tmp/q4e_prof_test_verify.meta.json` 绑定 `256db15`、exe SHA、baseline MoE/ids-only，记录 16 个生成 token、decode 均值 95.70 ms、总生成 4162.77 ms。它早于后续 handoff 修复与 M2，不能给当前 HEAD 背书，也不能与旧 48-token warm run 直接比较。

同一日志的累计 dispatch 计数为 `194/2 → 388/4 → 582/6`（on-demand/sync）。这支持该旧构建每次记录区间约 194 次反量化只留下 2 次同步，**不支持“反量化已消除”或“同步为零”**。剩余两次应先定位 PLE 等未继承标志的调用者；当前计数是进程级累计，不是 per-layer 或 per-request 差值。

## 2. 本轮发现与处置

### F1 / P1：长上下文 prefill 没有 QSA mask

`src/models/Qwen4ExpBnb4Backend.cpp:1050–1083` 的 `count > 1` 分支只投影并保存 indexer raw K，随后调用普通 `attention_prefill` / `attention_prefill_cached`。没有每 query 的 block score、选块或稀疏 mask。

本地参考 `modeling_qwen4_exp.py:693–775, 847–862` 对每个 query 的可见 token 集合分块选择，再与 causal mask 合并，与 prefill/decode 无关。budget=2048、compress=4 时，可见长度 2052 首次出现第 513 个完整块，需要丢掉其中一个；2051 仍是 dense identity。

因此，长 prompt 中可见长度超过 2051 的行，以及在长历史后追加多个 token，都产生不同于参考的 attention。将同一输入分成 `count=1` 或 `count>1` 也会改变选择语义。这是实际功能缺口，不是仅缺 benchmark。

**修补顺序：**先给跨界 prefill 的每行复用已有 decode 稀疏原语作为有界正确路径，保持每行可见长度与 tail 正确；随后做 batched QSA。前一做法会慢，但可作为后续优化 oracle。不要以整个 chunk 的终点统一决定所有 query 可见块，避免未来信息泄漏。

**最小验收：**缓存长 2050 后追加 4 token，比较 chunk 与逐 token 的 selected block sets/attention；一条跨界用例即可覆盖 dense→sparse 和不完整 block 的变化。数值比较须先明确 F4 的精度边界。

### F2 / P2：固定 32 次浮点二分不是精确 Top-K

`src/ops/QsaIndexerOps.cpp:326–350, 369–397` 二分后将 `[low, high]` 内的候选按 block id 补齐。区间内可能包含多个**不同**分数，并非只有同分 ties；提前补齐会漏掉更大的分数。

`[CPU-CHECK]` 本轮用 PowerShell `single` 复写该比较/阈值逻辑：514 个分数为 `[0.0001, 0.0002, 0] + [1000000] × 511`，取 512。32 次之后 `low=0`、`high≈0.00023283064`，当前算法选择 block 0、排除 block 1；精确结果应反过来。全部是有限非负输入，正是 score API 的值域；本轮没有声称真实 checkpoint 已产生该分布。

**修补：**对未决区间做精确选择，或使用有确定性 index tie-break 的 float key radix-select。不要简单增加固定二分次数后宣称严格正确。非负有限 f32 可按 bit pattern 有序选择；要另外定义 NaN、Inf、±0 处理。原 host `partial_sort` 可作为此 API 的 oracle。最低限度补这一个反例，暂不扩大 sweep。

### F3 / P2：AcquireGuard 释放 reservation，但不回滚已发布的 GPU key

`Qwen4ExpExpertCache.cpp:1045–1061, 1095–1144` 在所有 miss 验证/填充前就修改 key→slot。若 pool=2、acquire 三个不同专家，前两个已被映射，第三个抛出 all-slots-reserved；guard 清 reservation 却保留映射，Phase 3 DMA 尚未开始。调用方捕获异常后再 acquire 第一个专家，会走 GPU hit，实际 slot 仍是旧内容。后续专家的 metadata 验证失败也有同类部分提交窗口。

**修补：**预检 unique demand 容量及 metadata，再建立暂存绑定；提交成功后发布 Ready，失败时回滚新映射或直接 poison cache，禁止继续使用。DMA 已部分提交的失败必须 drain 或保留 source reader event，不能仅清 reservation。明确选择“可恢复事务”还是“故障后销毁”，不要两者都不保证。

此问题不表示当前 backend 的正常 batch 必然越界；正常路径已有容量 guard。它阻碍的是公共 cache API 与“异常安全已完成”的验收。最小 CPU/小 synthetic 用例是“超容量失败→再次 acquire 原专家”，无需完整模型。

### F4 / 对齐门槛：QSA host reference 复刻了同一精度路径

`QsaIndexerOps.cpp:91–92, 211–245` 将 norm 后 Q/K 保持 f32 再做 RoPE；K 在 dot 前仍是 f32。参考 RMSNorm `:167–172` 返回输入 dtype，BF16 参考路径的 norm 输出先舍入；RoPE 的 cos/sin 与输出也有 BF16 边界。当前 `plus_one_norm` 又把 `(1+w)` 预先舍入 BF16，而参考在 f32 中加一和乘法后才 cast。

这不等于每个 token 一定选错；但 score 的近似误差经过 Top-K 是离散放大的，不能用 attention 的宽容差直接授权 indexer 近似。当前 host reference 也保留相同 f32 中间值，因此 GPU/host 高精度吻合不能证明吻合上游。

**处置：**先定义参考精度契约，保留精确/近似实现的明确分支。只取一层、一个跨界 query 的真实 QK/norm 输入，对照本地 Transformers 的 selected **set**（参考随后转成 mask，输出顺序无须完全相同），同时保存 cutoff 两侧 margin。若出现精确 ties，上游 topk 未保证按 id 稳定，必须单独约定 tie policy。没有独立对齐前不将 M2 宣称完整正确。

### 其余边界

- QSA kernel 固定 4 Q head、1 K head、128 dim、compress 4、eps=1e-6，而 backend 部分尺寸读取 config。当前 checkpoint 相符；加载阶段应拒绝不支持的配置，避免接口看似泛化。
- `wait_copy()`、无 host pool 的 `last_dma_event_` 等待和 move 仍有吞异常处。正常执行统一交给 Context/runtime fault 状态；析构 best-effort drain 与正常成功完成不能共用语义。
- 当前 raw indexer K 与 full KV 都是不可变有效前缀；Truncate/Reset 只缩小长度、重写新 suffix 原则上可行。**不把未清空整个 K cache 报成 bug。**新增 compressed K cache 后则必须管理其有效前缀和未满 block。

## 3. 系统目标：优化关键路径，而不是累计局部收益

`[DERIVED]` 端到端每层至少包含：前置计算→真实路由可见→必要专家 Ready→MoE 完成→下一层。CPU fill、H2D、shared compute 和预测有重叠机会。`router_sync` 往往等待整个此前 GPU 前缀，不等于 router kernel 自身耗时；`pf_wait` 又包含在 acquire 中，不能相加估算收益。

用两条关键约束筛选方案：

1. **延迟约束：**必要数据就绪时间是否早于消费者开始？把非关键路径 kernel 加速一倍，端到端可能不变。用 DAG 上最长路径判断，不能把独立 roofline 时间简单相加或统一取 max。
2. **容量约束：**新增 scratch/cache 是否挤掉 GPU 热专家、host 热专家或 OS mmap 页？局部算子加速可能因为额外 miss 反而降低吞吐。

数据下界可用 `D_H2D / 实测有效PCIe带宽` 和 `D_NVMe / 实测有效存储带宽` 辅助判断；这些只是资源服务时间下界，不等于 TTFT/TPOT。算子占比为 f 时，其加速 s 的理想整体上限是 `1 / (1-f+f/s)`；存在重叠时收益还可能更小。

建议只补一个轻量 trace：每层真实 route-ready、必要 fill-ready、DMA-ready、expert done 的事件/主机时间，以及 demand/prefetch 字节和 cache 命中层级。不默认全程 VTune；只有某候选需要解释 kernel 瓶颈时才抓一个代表形状。

## 4. Brainstorm A：QSA 从“反复计算历史”改成“增量状态”

以下均为 `[HYPOTHESIS]`，F1/F2/F4 是先决条件。

### A1. 完整 block 的 normalized + RoPE K 只计算一次

当前每个 decode query 都对历史 raw K 重新执行四 token pooling、RMSNorm、RoPE。对纯文本连续 causal prefix，完整 block 的 token、norm 权重和 block-start position 不再变化。

新增 session-owned `compressed_k[full_layer, max_blocks,128]`，每形成一个完整 block 时写入最终参考 dtype 的 K；score kernel 只做 Q×compressed-K 和 ReLU/head sum。prefill 批量建立完整 block，decode 每 4 token 补一个；tail 不压缩而是强制加入 attention。

32K context、12 层时 compressed BF16 K 约 24 MiB，现有 raw K 约 96 MiB。第一版可同时保留二者以简化回滚；成熟后只保留 compressed prefix 与未完成 block 的最多 3 个 raw keys。Truncate 到 block 内时必须恢复该 block 的 raw tail，不能仅缩短 compressed count。多模态位置由 session 提供，只有位置映射固定后才缓存。

**成立条件：**所有精度舍入点与 F4 契约一致；不能将缓存的 f32 K 擅自改成 BF16 当作无损。**最小实验：**一个含完整 block + tail + truncate/reappend 的 synthetic 输入。**停止条件：**收益被额外内存引发的专家 miss 吃掉，先回到保留 raw 的小范围版本评估。

### A2. 分离 K 建库与 Q 选择

当前 prefill 投影 `[2560→640]` 却只使用 128 维 K；decode 在 sparse 尚未启用时仍生成 Q、做 Q norm/RoPE。

对每行可见长度≤2051 的阶段只执行 indexer K 的 128 个输出行；超过阈值才算 512 个 Q 输出并选块。理论上可移除该阶段 indexer 投影 **80% 的输出行计算**，不是整体 80% 提速。若 session 最大容量本来≤2051，可以完全省去 indexer 数据路径；若可能跨界，就必须保存早期 K，不能临时缺历史。

K-only projector 与完整 projector 在加载/slot admission 时创建，避免 decode 中新建 oneDNN descriptor。分割投影可能多 launch，长序列应按净 wall time决定是否合并回 640 行。

### A3. 精确选块与 block-native attention

先修 F2，再考虑分层选择：tile 局部候选→全局精确选择→按 block id 整理输出。每 tile 保留 min(K,tile_size) 才能保证包含全局 Top-K；不能凭经验只取很少候选。可比较 radix-select，避免当前单 WG 反复扫全表、最后由单线程收集/排序的扩展性限制。

稀疏 attention 可直接消费 512 个 block IDs + tail，而非先展开为 2048+tail token IDs；保持块内连续四 token，沿连续维度向量化。继承现有 head256 partial/merge 的稳定 softmax，先改变索引接口，再考虑同 KV head 的多 Q head 复用。新 SLM 设计只针对 attention，不能混同被否定的 GEMV SLM input cache。

### A4. 真正的 batched QSA prefill

F1 的逐 query 正确路径稳定后，按有限 query tile 计算 `[query_tile,visible_blocks]` score，应用逐行 causal block count 和 tail，再将 block IDs 交给 tiled sparse attention。避免分配 `[whole_prompt,max_context]` score。

长 prompt 的收益不只是 token attention 从 L 缩到预算，还包括不物化大稠密 attention 矩阵。Indexer score 本身仍随历史长度增长，不能宣传“总注意力复杂度完全常数”。先比较 tile=小固定值的一个代表长度，不做全尺寸网格搜索。

## 5. Brainstorm B：专家路径按 deadline 和成本调度

### B1. 明确 host/GPU slot 的状态机

`[HYPOTHESIS]` 从布尔 reservation 进一步演进为预分配状态记录：key、generation、state、fill owner、H2D source-reader fence、GPU consumer-reader fence。操作最少需要区分 Empty/Filling/Copying/Ready/Faulted；generation 防止旧 completion 更新已复用 slot。

这既修 F3，也为后面的重叠建立边界：**producer completion 不等于 consumer completion，发布 key 不等于内容 Ready。**不要先加入更多队列再补 ownership。第一步可仅增加状态与断言，保留现有调度方式，避免一次重写整个 cache。

### B2. 真实路由与预测完成事件拆开

当前 `route_event` 被下一层 predictor event 覆盖，CPU acquire 仍等预测结束。保留 `actual_route_ready`、`prediction_ready` 两个事件：shared GPU 工作先提交，CPU 一旦 actual ready 就开始本层 demand；预测 buffer 只能在 prediction ready 后读取。

但当前 `acquire()` 是主线程同步调用，**只拆事件不保证预测能及时提交**。初版可在 demand 返回后消费已完成预测，量出“需求更早 vs 预取更晚”的得失；若被抵消，再让 fill 调度器支持非阻塞 demand ticket/完成回调，由独立 owner 按优先级提交后续预取。固定任务池，避免逐 token 创建 thread/future。

### B3. 预取按避免的 stall 排序，而非固定取最多 48 个

候选 utility 可先用低成本估计：`P(近期需要) × 预计避免的暴露等待 − 错误预取代价 − eviction代价`。代价包括文件页载入、CPU repack、H2D 和 host 池污染；单纯提高预测命中率可能仍变慢。

在截断 top-N **之前**过滤 GPU resident、host Ready、已排队项。对 GPU hit 不发 IO；host hit 是否提前 H2D 则由空闲安全 GPU slot 与 deadline 决定。使用固定 byte/part credit，demand 保留最低资源，background 仅消耗余量。取消无法抢占已陷入 page fault 的 memcpy，part 大小是取消粒度与锁开销的折中。

早期只记录 issued/useful/cancelled/wasted bytes 和 demand exposed wait，采用平滑的简单规则；暂不上 RL 或在线复杂模型。预测帮助小就自动降额，避免为了“预测更多”制造磁盘排队。

### B4. 两级 cache 的替换成本与离线 trace replay

GPU 热度按 layer/expert 的访问与缺失代价评估；host 层同时考虑恢复成本与复用距离。现在按 layer 平分保留量与 SLRU 可以作为 baseline，不预设更复杂策略必胜。

利用已有 `perf/qwen4exp/q4e_cache_sim.py` 思路，固定一条路由 trace，离线比较 LRU/SLRU/按成本 admission；输出每层 miss bytes、host miss 与替换次数。不要用只统计 hit-rate 的模拟推断 tok/s，也不要修改路由以迎合 cache。只将明显占优候选放进一轮实机短测。

### B5. 专家访问单位成为磁盘和 DMA 的共同单位

实验性离线 pack：将原 checkpoint 中一个专家的 GU/down/scale codes 按当前 slot 所需顺序存为连续 record，保留 NF4 bit pattern、nested metadata 和校验身份。按 layer/expert 建索引，fill 退化为顺序页读取或一次拷贝，减少散片 gather 与 CPU repack。

先只 pack 一个代表 layer；评估冷/暖页两种状态和额外磁盘空间，不一开始复制整个模型。不要默认把全部文件 pin 到 host RAM：64–96GB RAM 同时承载专家、PLE、OS 页缓存，双重驻留可能更差。进一步的 file prefetch/bounded async IO 是候选，并不假设 mmap 或 direct IO 哪个天然优胜。

### B6. Demand microbatch 的双缓冲专家流水线

完整按层 acquire 会等待所有专家就绪。可以将专家分成小 group：fill/copy(group j+1) 与 compute(group j) 重叠；先算 ready experts，但最终按原 token top-k rank 固定次序归并。

这是高收益也高风险方向：需要 B1 的 GPU reader fence，不能提前复用仍被 kernel 读取的 slot；还要防止为了提早 launch 增加太多小 kernel。先在 prefill S4 做候选，固定 scratch，不改变 decode 的稳定 GEMV。小 cache fallback 的保守等待在这条路径真正建立后再移除。

## 6. Brainstorm C：用“每专家 token 数”选择计算方式

### C1. 从排列局部性到显式权重复用

当前 S4 把同专家 activation 相邻放置，仍独立读取/解码权重。第一步尝试 output tile→activation group 的 dispatch swizzle，观察缓存复用；第二步尝试同一权重行服务少量 token accumulator，限制寄存器与占用率。

这不是禁用清单中的“一个 subgroup 算两个输出行”。仍保持一个权重输出行，改变 token 维复用；它依然可能因 GRF 增长变慢，故只能视为实验。第一轮只做一个形状和一个 2-token 候选，失败就停止，不展开穷举。

### C2. 按专家拥挤度而非 prompt 长度派发

关键量是本次每个专家的 token 数 `m_e`，不是全局 count。对热专家 m_e 大的 group，考虑 NF4 tile 解码→BF16 SLM/VNNI→XMX；稀疏专家继续 packed GEMV。采用固定 tile arena，避免为每个专家常驻 BF16 权重。

NF4 是非线性 LUT，不能把 codes 当线性 U4 直接套整数 DPAS。代价模型必须包括解码、布局转换、分组、tail、merge。只有 `reuse × 省下的重复解码/加载 > pack + launch + occupancy损失` 才值得升级。阈值来自本模型 A770 形状的少量代表点，不照搬技能中 INT4 大矩阵数据。

### C3. 非 MoE prefill：去掉 wait 之后再去掉多余数据往返

N2 已减少同步，但当前仍把 NF4 展开为完整 BF16 scratch 再由 oneDNN 读取。候选为现有 fused NF4 prefill 路径或 tiled dequant+GEMM；先覆盖大投影 `2560→16480`，再看 `6144→2560`、shared `2560→1280/640→2560`。按小、中 M 选两三个点即可，不做泛化 dispatcher sweep。

先把 PLE 的两次疑似残留同步核实为具体调用，不因“剩 2 次”而引入跨模块不安全依赖。进程级 atomic 计数改成可选的 per-forward 差值和类别计数即可，暂不建设庞大 telemetry 系统。

### C4. 低优先级研究池

DeltaNet prefill 分块/scan、HC 相邻 elementwise 融合、LM head 分片并只对 greedy 请求做设备端 argmax、按固定 slot 地址捕获局部 command graph，均可保留候选。前提分别是数值递推边界、融合寄存器代价、API 确实不需要完整 logits、图驱动能力及动态 slot 参数安全。

Graph 先覆盖没有 CPU expert miss 分支的小计算岛；不捕获跨 mmap/fill 的整 token 图。LM head 仍须计算所有候选 logits 才能精确 argmax，减少的是输出存储/回传；不要宣称省去整个 head 计算。继续遵守“不做 MTP”和既有 GEMV 负优化清单。

## 7. Brainstorm D：把长上下文容量作为优化的一部分

### D1. 显式 admission 预算与 workspace 上限

当前自动专家预算减去了常量 reserve + full KV，之后才建 runtime buffer；`scores=[24,cap,cap]`、`incr_scores=[24,cap,max_seq]` 仍无条件分配。解除 2048 检查不意味着任意长上下文可在 A770 容量内执行。

`[DERIVED]` f32 incr_scores 在 cap=2048、max_seq=32768 时为 **6 GiB**；cap=max_seq=32768 时 scores 和 incr_scores 各 **96 GiB**。这是这些配置的静态需求，不是本轮观察到的 OOM。把 prefill query tile 限为 256 时，同样 32K 历史的 incr_scores 为 768 MiB；最终应由 tiled/sparse attention 消除该矩阵，而不是仅缩小 cap。

Admission 先计算 `weights + KV + indexer + recurrent + selected-mode scratch + snapshot staging + expert cache + measured headroom`，再选 session 容量、prefill tile 和专家 slots。缺预算时减少 cache 或拒绝 admission，不用固定最低 1GiB cache 掩盖预算不足。运行中不迁移活跃 pointer，不在 decode 分配/绑定新描述符。

QSA 只降低被读取的 token 数，**不自动减少全部 KV 的存储**。32K context 的 12 层 full K/V 约 768 MiB，raw indexer K 约 96 MiB；必须计入整体预算。若尝试 FP8 KV，属于精度变更，单独验收。

### D2. Snapshot 从二次复制变成固定预算的数据路径

36 个 DeltaNet 层的 f32 recurrent state 合计 108 MiB，conv 约 4.22 MiB，单 snapshot 仅这两项就约 112.22 MiB；默认最多 4 个快照约 448.88 MiB host payload，另有 host mirror/PLE。本轮确认默认 session 已有数量上限，**不是无界增长**。

第一步建立预分配 snapshot slab/ring，直接 D2H 到目标 snapshot，完成事件后 publish，减少 mirror→snapshot CPU copy；pinned 只保留有预算的 transfer slots，不无差别 pin 全部历史。第二步才按回滚需求调 capture 频率。Rollback 要求、长 prefill TTFT 和带宽三者一起决定 checkpoint 间距；不能为少复制擅自破坏可恢复长度契约。

### D3. 为后续多 session 留出 ownership 边界

只读模型权重与专家 cache 可以共享，但 buffer、router pinned 输出、QSA state 和 recurrent state 必须属于 execution slot/session。当前单队列串行调度是合理起点；真正并行前先从 backend-global scratch 拆出固定 slot。共享 expert cache 的 reader fence 要涵盖所有消费者，不能用单 session 的 event 代表所有读取。

M4 可以受益于把同层多 session 的相同专家合批，但 layer-aligned microbatch 会引入排队延迟；目标是吞吐/TPOT 的可控折中，不默认做大 batch。M3 vision 会改变 position/visibility 边界，独立立项，不混入本轮纯文本性能结论。

## 8. 建议执行次序与最小决策门槛

| 波次 | 具体交付 | 最小证据 | 暂停/回退条件 |
|---|---|---|---|
| W0 正确性封口 | F1 跨界 prefill；F2 精确 Top-K；F3 cache 失败契约；F4 明确舍入/独立参考 | 一条跨界 chunk、一个 Top-K 反例、一次 acquire 失败重试、一层真实 indexer fixture；不要全模型矩阵验证 | 选择集合不符先修语义，不先调性能容差 |
| W1 容量与运行身份 | D1 预算/有限 prefill tile；模式和计数记录；保留 S1/S4 opt-in | 静态 bytes 表 + 一条目标长度 admission；已有小 prompt smoke 可复用 | WDDM paging/headroom 不够则减缓存/并发，不硬跑 |
| W2 增量 QSA | A1 compressed K + A2 K-only；保留可回退路径 | 一个 block/tail/rollback 用例 + 一个已需 sparse 的长度 | 数值边界改变或 cache miss 抵消收益则保留原路径 |
| W3 隐藏数据等待 | B2 event 分离 + B3 过滤/credit；随后 B6 小 group pipeline | 同 prompt/相同模式 warm A/B，必要等待与 bytes 同时下降 | 预取污染、取消等待或 launch 增长抵消收益 |
| W4 计算密度 | C1 单候选；获益后 C2/C3 tiled NF4/XMX | 一项代表 shape 数值比较与包含 pack/merge 的 wall time | 局部快而端到端无收益则停止推进 |
| W5 可选研究 | B5 离线 pack、D2 snapshot slab、C4 graph/DeltaNet、M4 | 由剩余瓶颈决定，一次一个方向 | 不为功能列表齐全而实现 |

**最值得先做的三个优化实验：**A1 完整 block K 缓存、B2/B3 demand 优先调度、D1 有限 workspace。它们直接减少重复历史计算、暴露的 IO 等待和容量挤占；C2 XMX 留在收益条件明确之后。这里的优先级是工程判断，不是已测出的收益排序。

### 有限验证规则

- 本轮只设计，不启动上述实验。下一轮先复用已有定向测试，仅增加触发新边界的用例。
- 性能决策先固定一个短 prompt 与一个刚跨 sparse 阈值的场景。冷启动/暖运行分开；相同 token 工作量、同一 exe 和 mode，交错 A/B，差异接近噪声就记录“无结论”，不反复扩大验证。
- sidecar 在启动前记录 exe/source identity、dirty flag、有效环境（含继承变量）、prompt/model config hash、GPU/驱动；完成后记录 exit code 和结果。当前脚本在运行后读取 HEAD/hash、仅记显式 EnvVars，仍可能发生身份错配。
- CPU/GPU 时长与端到端分开；不把 warm page cache 收益归给 S4，不把等待位移归为实际加速，不因 counter 为零就宣称全流程无同步。
- 只有推广默认、改变精度或声称完整长上下文时才增加对应范围的验证。单个局部修补无需全套 CTest、长压测或全模型 benchmark。

## 9. 代码索引与参考身份

- [Backend / QSA dispatch 与 buffer](../../src/models/Qwen4ExpBnb4Backend.cpp)：Session 208+，预算 548+，workspace 603+，QSA 1006+，MoE event 1101+，snapshot 1476+。
- [ExpertCache](../../src/models/Qwen4ExpExpertCache.cpp)：pool 17+，move 271+，batch-reader 923+，acquire 1007+，guard 1045+，binding 1124+。
- [QSA kernels 与自写 host reference](../../src/ops/QsaIndexerOps.cpp)：Q norm/RoPE 39+，block score 150+，selection 258+，host reference 430+。
- [Bnb4BitLinear](../../src/ops/Bnb4BitLinear.cpp)：async dequant 2258+；[Context](../../src/core/Context.hpp)：in-order queue 与 oneDNN stream 25–37。
- [Kernel tests](../../tests/ops/Qwen4ExpKernelTests.cpp)、[Module tests](../../tests/models/Qwen4ExpModuleTests.cpp)、[profiling 脚本](../../perf/qwen4exp/q4e_prof.ps1)。
- 上游语义参考是本机 `C:/Users/Administrator/AppData/Local/Programs/Python/Python313/Lib/site-packages/transformers/models/qwen4_exp/modeling_qwen4_exp.py`，本轮实际读取并在 evidence JSON 中记录 SHA-256；没有依赖未读取的网络版本或声称它是最新版本。
