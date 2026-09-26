# Qwen3.8-Flash-Next / A770：阶段验收与下一轮优化

> 后续增量验收：请以 [484b086 验收与系统 brainstorm](qwen3.8-flash-next-a770-review-and-brainstorm-484b086.md) 为当前结论。本文保留 `05055aa` 历史审查；原 R1–R3 的正常路径已修复，新发现与后续优先级见新报告。

日期：2026-09-26。审查基线：`05055aaf01d023364ca925e7029726b10567bceb`。对照基线：上一版计划编写时的 `91c9d8b`。开始审查时工作树干净。

**验收结论：有条件接受阶段成果，暂不认定全部完成，也不建议现在把 S1/S4 升为默认。** S0、S1、S3、S4、S5 与 ids-only 预测均有实质实现；已有定向测试和文本输出记录支持代表路径正确。但缓存还有三个可由代码确认的问题，性能记录也不足以归因到单个阶段。先完成小范围修补，再做有限的默认路径选择，比继续叠加大算法更合适。

本轮完成静态代码审查、历史日志重算和文件核对；没有修改运行时代码，没有重新构建、运行测试或实模生成。没有为了“验收”重跑全套验证。以下区分 `[CODE]` 实现事实、`[LOG]` 既有运行记录、`[DERIVED]` 静态推导及 `[HYPOTHESIS]` 后续设计。

入口：

- [第一轮设计与阶段表](qwen3.8-flash-next-a770-next-optimization.md)
- [本次日志证据摘要](qwen3.8-flash-next-a770-acceptance-2026-09-26-evidence.json)：保存 13 份日志的摘要、原文件 SHA-256、输出文本行 SHA-256 和已有 CTest 记录。
- [原交接](../Qwen38FlashNext_A770_Handoff.md)：仅作早期背景，不能代替当前实现状态。

## 1. 验收发现：先修补的三处问题

以下是静态确认的缺陷，不声称本轮已经复现 GPU 数据损坏或挂死。行号以 `05055aa` 为准。

### R1 / P1：预取命中的 host slot 没有转入本次 DMA 保护集合

位置：`src/models/Qwen4ExpExpertCache.cpp:1123–1136`，关联 `:939–977`、`:1319–1337`、`:1378–1408`。

完整触发链：

1. `prefetch_async` 给 host slot 设置 `host_slot_reserved_[h]=true`。
2. 真正 acquire 命中该 prefetch item，完成 fill 后提交从 h 发起的 H2D。
3. 代码只有在 `!host_slot_reserved_[h]` 时才把 h 加入 `batch_reserved_hslots_scratch_`。正常的 in-flight prefetch reservation 已经为 true，故这个分支跳过。
4. 后续 `cancel_prefetch()` 发现 h 不在 batch 集合，释放 reservation；而本次 DMA event 也只写给 batch 集合，因此 h 没有记录刚提交的 DMA reader。
5. 若同次 acquire 还有未预取的 miss，需要取消预取并填充新专家，或随后启动新的 prefetch，h 就可能在旧 H2D 完成前被重新写入。

小 host pool 可以很快触发复用；大 pool 只能降低概率，不能建立正确的生命周期。已有 contention 测试覆盖了普通 host hit + miss，但没有确定性地阻塞 copy 后覆盖“prefetch hit + 新 demand miss”这条组合。

修补建议：把“由 prefetch 保留”和“属于本次 DMA batch”分开记录。每次提交以 h 为 source 的 copy，都必须去重加入 batch-reader 集合，不能用单个 reserved 布尔值判断它是否已加入。取消预取只释放该任务的 ownership；DMA read 结束前 slot 仍不可写。用独立 membership 位图或预分配数组，避免新堆分配。

最小验收：一个 host slot、至少两个 GPU slot，先预取 A，随后 acquire `{A,B}`；人为延迟 A 的 DMA，确认 B 的 fill 不覆盖 A source。再确认 acquire 返回后立即预取 C 也受到同样保护。只补这一条有明确事件控制的用例，不用随机长压测代替。

### R2 / P2：cache move 破坏 host LRU 与新增生命周期成员

位置：`src/models/Qwen4ExpExpertCache.cpp:255–359`，尤其 `:316–330`。

`operator=(Qwen4ExpExpertCache&&)` 先 move 两条 host list 与 iterator 数组，随后调用了**目标对象**的 `host_probationary_list_.clear()` 和 `host_protected_list_.clear()`；保留下来的 iterator 因而失效。它还没有 move `host_slot_reserved_`、`host_slot_dma_events_` 和 `batch_reserved_hslots_scratch_`。移动构造调用同一个赋值逻辑。

结果：移动一个已初始化且有 host cache 内容的对象后，host hit 的 `splice` 可使用悬空 iterator；新对象的 reservation/event 数组也可能为空或残留旧尺寸。即便主后端当前不常移动该成员，这仍是公开 move API 的确定性破坏，不能把生命周期阶段完整验收。

修补建议：若项目没有实际移动需求，删除 move API 是最小选择；否则完整迁移所有状态，清理 moved-from 对象而不是目标 list。更关键的是，后台 fill lambda 捕获原 `this`，move 前必须停止并排空源对象工作，不能只转移 `fill_pool_` 指针。保留 `noexcept` 时也要明确排空失败策略。

最小验收：已初始化、包含 Ready host slot 的 cache，move 后执行一次 hit 与一次 miss；若支持 busy move，再单独验证该契约。不要把空对象 move 当作覆盖。

### R3 / P2：允许 64 个 fill part，但领取位图只有 32 位

位置：`src/models/Qwen4ExpExpertCache.hpp:191` 附近的 `PrefetchItem::claimed_parts`；`Qwen4ExpExpertCache.cpp:686`、`:912–915`、`:1425–1427`。

配置仍将 `AILA_Q4E_FILL_PARTS` 限定在 1..64；位图却为 `atomic<uint32_t>`，两处领取操作使用 `1u << p`。当 p>=32 时属于 C++ 非法移位，无法唯一领取后 32 个 part；`parts_remaining` 仍按完整数量等待，可能重复/漏填或永久等待 ready。

修补建议：在 CPU 任务位图上使用 `atomic<uint64_t>` 和 `uint64_t{1} << p`，或者明确把支持范围降到 32 并同步配置说明。这里是 CPU bitmap，与禁止在 GPU GEMV 中使用 uint64 权重加载无关。

最小验收：位图领取/计数的 CPU 边界检查 31/32/33/64；若保留 64，再用一条小 synthetic prefetch/acquire 检查全部 slot 字节并设有界超时。默认 8 的测试通过不能覆盖此问题。

## 2. 其余生命周期缺口与验收范围

这些项目也应进入下一轮修补清单，但需区分原有问题和新阶段回归。

- **小 GPU pool 的逐 token prefill fallback 仍未保护 GPU reader。** `ForwardImpl:1222–1258` 在同一层中连续 acquire / 提交 MoE，router event 只在该层循环之前等过。`acquire` 在返回前释放 slot reservation，copy queue 没有前 token 的 down-complete 依赖；发生驱逐时可能覆盖上一 token 尚在读取的专家。此问题在上一轮计划中已指出，不能归为本轮新增，却仍妨碍“小 cache 路径完整支持”的验收。先在 fallback 加保守的 token 完成边界，之后再做 per-slot reader event；默认大 pool 的按层路径不需要为此全面同步。
- **fill pool 的任务交接仍值得补一个确定性检查。** `ExpertFillPool::run_demand` / `cancel_prefetch_job` 等待 `active_workers==0` 后才关闭 `in_flight`；worker 在 mutex 外读取开关、增加 active 并领取任务，而 owner 可随后重用 `fn/total/next`。这里有“检查空闲后又有 worker 入场”及任务代次混淆的窗口。尚未做调度复现，不把已有测试通过当作排除了竞争。改为同一锁下的领取/关闭协议或 generation + 明确关闭 admission，再排空引用。
- **运行期 copy 错误不能被当成完成。** `pick_victim_hslot` 两个等待分支吞掉 `wait_and_throw` 异常后清 event 并返回可复用 slot。析构中吞异常可以理解，正常 acquire 的故障应进入 runtime fault 路径；不要在设备异常后继续绑定和覆盖专家。

本轮不实现这些修复，以保持“验收与后续规划”的范围；运行时代码仍保持原状。

## 3. 各阶段实际完成度

| 阶段 | 当前代码结果 | 验收判断 |
|---|---|---|
| S0 | 直接索引表、GPU/host list splice、主要 acquire scratch 预分配、自带 scratch 的稳定排序 | 主项完成。仍有 host list 初次插入/取消后重新插入、`std::function` 等潜在分配，不能称整个热路径严格零分配；观测指标未全面补齐。 |
| S1 | 两个 token-batched MoE kernel、device descriptor/weights、`h[N,10,640]` | 实现与代表形状正确性可接受；`AILA_Q4E_PREFILL_MOE_ROWS=false`，仍是 opt-in。 |
| S2 | 每 expert ready/parts、全 GPU hit 不等预取、调用线程补齐需求 part、取消非必要预取、host DMA events | 有实质进展；R1–R3 未修，slot 生命周期不能完整通过。demand 前仍会取消并 drain 正在执行的 background parts，不是任意抢占磁盘访问。 |
| S3 | 返回 router event，shared 在 wait 前提交；host cache 新增 protected/probationary SLRU 和 demote | event 重排完成。SLRU 是额外策略变化，不能将其收益与调度收益混在一起；需单独判断。 |
| S4 | host 专家分桶、expert-major gate/up、并行 expert down、按原 top-k 次序 f32 merge，共三个 kernel | 等价路径已实现；默认关闭。没有 XMX，也没有 tile 内显式跨 token 权重共享，后者只存在调度局部性带来的缓存机会。 |
| S5 | 删除 qkv/ple_out/moe_acc_f32；snapshot/restore 分批 enqueue，末尾统一等待 | 第一阶段完成。仍是 pageable host vector + mirror→snapshot 拷贝，未实现连续 pinned slab / 消除二次 CPU copy。 |
| E1 预测 | `AILA_Q4E_PRED_IDS_ONLY=true`；实际路由仍用 softmax/top-k，预测只返回 ids；`05055aa` 修 selection SLM 问题 | 边界测试已接入 `test_router_softmax_topk`；功能可接受，单独性能收益没有建立。 |
| E1 图 / E2 | 未见静态图、DeltaNet 分块、HC/LM head 专项或 AOT 改动 | 保留研究项，不写“已完成”。 |

### 3.1 S4 实现与“权重复用”的准确说法

`MoeExpertOps.cpp:936` / `:1076` 将 `global_row` 映射到单个 `act_idx`，每个 subgroup 仍为该 activation 独立读取 packed 权重、解码并点积。同专家 activation 被排到邻近位置，有机会提高缓存命中，但**没有在一个 tile 中只解码一次权重供多个 token accumulator 使用**。

相比 S1，S4 还把 down 的 top-k 串行循环拆成 expert 维度并行，再写 `[N,top_k,H]` f32 临时结果、有序归并。其潜在收益既可能来自并行度，也可能来自 locality，不能只归因为减少权重字节流量。

### 3.2 新增 workspace 会影响默认模式

`ensure_runtime_buffers:632–659` 无条件分配 S1/S4 的 descriptor、mapping 与 `moe_down_acc`，即使两个实验开关都关闭。仅 `moe_down_acc[cap,10,2560]` 就是 cap=274 约 26.76 MiB、cap=2048 为 200 MiB；实际容量可能因倍增大于当前 count。

因此“两个实验关闭”并不等于旧内存基线。下一轮应按 load 时确定的模式预分配 workspace，将增量纳入 pool budget；不要在 decode 热路径临时分配。S5 已回收部分闲置 buffer，但它的收益不能与 S4 无条件增量抵消后仍按原数值宣传。

## 4. 已有正确性记录：接受什么，不接受什么

`[LOG]` 本地 `build-refactor/Testing/Temporary/LastTest.log` 记录 2026-09-26 13:33 的四个目标通过：

| 目标 | 耗时 | 与本次验收的关系 |
|---|---:|---|
| AilaQwen4ExpModelSpecTests | 0.03 s | synthetic / 本地实际 config 解析 |
| AilaQwen4ExpKernelTests | 17.43 s | S1/S4、router、HC、attention 等定向算子 |
| AilaQwen4ExpModuleTests | 2.63 s | cache、host pool、prefetch、SLRU、snapshot，以及实际模型 PLE/cache |
| AilaSafeTensorsTests | 0.36 s | lazy loading / 131 shards 索引 |

这是四个目标的历史记录，不是本轮复跑，更不是“57 个测试全部执行通过”。日志未绑定 executable hash / commit；二进制时间早于提交时间不代表未包含修复，也不能作为构建来源的严格证明。

`[CODE + LOG]` MoE 新旧比较覆盖 N=2/26/64、k=4/10、nested/f32、不同专家和真实 expert-major 排列，记录 0 mismatch；ids-only 的普通、负值、极负值、tie 用例实际从 main 的 router test 调用。本轮确认了用例接线，未只看函数存在。

仍有边界：S4 nested fixture 使用较窄的 code 范围及零 offset；snapshot 测的是构造状态模式的 capture/restore，不是完整 logits replay。若后续改 nested arithmetic 或跨 token snapshot，再各补一个有针对性的案例，不为当前文档工作扩大整套测试。

13 份性能日志中，同长度 story 或 lighthouse 的 `Aila:` 输出行哈希一致。它支持已记录文本一致，**不证明所有层、logits 或 token ID 都 bit-exact**。不能把有限形状的 kernel 0 mismatch 推广成全模型的无条件数值保证。

## 5. 性能记录重新解读

下表单位 ms；decode 是按现有脚本口径，去掉最初四个 count=1 记录后的平均，不是峰值。`主 prefill + anchor` 是两个 forward 的 profile 总时长，不包括 tokenizer/其他框架开销，不能严格等同完整 TTFT。运行模式根据文件名推定，日志没有保存完整 env manifest，证据 JSON 中已标为未核实。

| 日志（`tmp/q4e_prof_` 后缀） | 生成量 | 主 prefill + 4-token anchor | decode 平均 | 整次生成 |
|---|---:|---:|---:|---:|
| baseline_warm | 48 | 2380.58 + 441.62 | 83.28 | 6955.96 |
| s1_rows | 48 | 2202.75 + 424.03 | 76.67 | 6431.94 |
| e1_warm | 48 | 2182.46 + 439.36 | 75.42 | 6370.84 |
| e1_fallback | 48 | 2212.55 + 443.70 | 75.17 | 6394.12 |
| s4_baseline_measured | 48 | 2217.11 + 439.27 | 74.94 | 6386.75 |
| s4_enabled_measured | 48 | 2203.36 + 427.15 | 75.42 | 6387.11 |
| s4_lighthouse_base | 20 | 24721.26 + 683.30 | 183.09 | 29364.46 |
| s4_lighthouse_s4 | 20 | 9636.20 + 533.20 | 131.10 | 12890.55 |
| s4_lighthouse_s4_m2 | 20 | 9650.19 + 619.80 | 118.94 | 12744.68 |
| lh_s4 | 20 | 24703.64 + 673.91 | 184.21 | 28944.90 |
| lh_s4_warm | 20 | 9690.40 + 527.00 | 131.50 | 12660.39 |

结论：

1. **S1 有值得保留的正向信号，但没有独立证明 decode 提速。** S1 只处理 count>1，表中的 decode 83.28→76.67 还包含页缓存、prefill 后专家分布与测量状态影响。正确结论是“这组整次结果更好”，不是“S1 kernel 令 decode 加速约 8%”。
2. **E1 的配对差异很小。** ids-only / fallback 的整次差约 23 ms（约 0.36%），decode 75.42 / 75.17 ms。原阶段表“83.3→75.4 ms”跨了不同条件，不能记为 E1 收益。
3. **S4 在较可比的短 story 记录中整次几乎持平。** 6386.75 / 6387.11 ms；prefill forward 合计减少约 25.87 ms，但没有转为整次收益。支持继续保留实验，不支持普遍默认开启。
4. **lighthouse 的 24.7→9.6 s 受冷热变化严重混淆。** `s4_lighthouse_base` 的主 prefill fill 为 19595.29 ms，`s4_lighthouse_s4` 为 6876.66 ms。更关键的是同样标为 S4 的 `lh_s4` / `lh_s4_warm` 自身就出现 24703.64→9690.40 ms，fill 为 20496.88→6953.43 ms。这足以否定“该差值可直接归因 S4”的说法；不能反过来断言 S4 完全无收益。
5. **最新大 prompt 仍有 host stall 与 router 边界成本。** `lh_s4_warm` 的 decode 平均 131.50 ms，其中 acquire 58.77、router wait 48.40、prefetch wait 19.51 ms（pfwait 已包含在 acquire，不能重复相加）。单 token 的 pf_wait 可过 100 ms，说明“fine-grained”并未消除 demand 等待。

不需要为清楚可见的冷热混淆立即重新跑一轮大 benchmark；先修 R1 等问题，并补运行身份记录，再做真正能作默认选择的小比较。

## 6. 下一轮实施顺序

| 顺序 | 任务 | 具体产物 | 完成条件 |
|---|---|---|---|
| N0 | 生命周期修补 | R1/R2/R3；小 cache fallback 的 reader 依赖；任务交接协议收口 | 每个问题一个最小触发用例，现有 Q4E module 目标通过 |
| N1 | 选定默认路径、控制 workspace | 在 baseline / S1 / S4 中选择；按模式预分配；补日志 manifest | 同模式同缓存状态下的完整请求收益，开关与结论一致 |
| N2 | 非 MoE prefill 的 dequant / 同步路径 | Q4E 专用 dispatch 计划、复用现有 NF4 GEMM 候选 | 消除或替换已确认的重复工作，保持依赖与数值契约 |
| N3 | S4 真正的跨 token 权重共享 | scheduling swizzle → 小 token tile → 仅热专家 XMX | 算子全链路与实模 prefill 都有收益，否则回退 S1 |
| N4 | 预取调度和真实 router 边界 | 分离 actual/pred event、过滤后再配额、demand 效用统计 | 降完整 decode/尾延迟，而非只改计数归属 |
| N5 | snapshot 与提交开销 | 一份 pinned staging/受限 snapshot slot；必要时静态子图 | 只在前述主项不再主导时执行 |

N0 是下一次开发的第一项。N2/N3/N4 是候选顺序，不要求一次全部实施；每次只改变一项结构变量。

## 7. N1：默认选择不需要大测试矩阵

### 7.1 先让结果可归因

给 `q4e_prof.ps1` 加简短 sidecar JSON：commit、exe SHA-256、完整命令行、本次相关环境变量、实际路由模式、driver/compiler 标识、max-seq、pool slots/bytes、是否首轮。当前日志缺少这些信息，不能只凭文件名判断 S4 是否覆盖了 S1。

补齐 `docs/Environment_Variables.md`：目前缺少 `AILA_Q4E_PREFILL_MOE_ROWS`、`AILA_Q4E_EXPERT_MAJOR_PREFILL`、`AILA_Q4E_PRED_IDS_ONLY`。明确 S4 优先于 S1、两者默认 false、预测 ids-only 默认 true，并记录启动时实际选择。

最小决策：

- 先用已有短 story 配对说明 S4 暂无整次优势，不再重复证明它。
- 修补后只针对 lighthouse 做 baseline 与一个候选（先 S1，再决定是否需要 S4）的各两次连续运行，第二次作 warm，记录全部结果。
- 只在差异接近噪声或方向矛盾时再做一组反序；不清 OS cache、不循环跑长篇 essay、不混用 24/48-token 生成量。
- 需要独立 device 结论时才扩现有 MoE bench，测 gate/up + down + merge + mapping copy 的全链路。microbenchmark 不承担整个模型的收益证明。

没有清楚胜者就保持实验开关关闭。默认 SLRU 的收益也需通过原 LRU 或固定 protected-ratio 的可控对照隔离；不要继续同时改策略与线程调度。

### 7.2 只为选定模式支付内存

load 时固定策略，按允许的 prefill cap 预分配对应 scratch；baseline 不分配 S4 down_acc，S1 不保留 S4 mapping。按 expert 存一份 descriptor、activation 存 compact expert/slot index，避免两套 `[N,10]` 大 descriptor 重复上传。

保留原显存 reserve，不立即把回收余量全部转为 expert slots。模式切换若需要扩容，应发生在请求 admission / prefill 准备边界，不能进入单 token decode 热路径。

## 8. N2：优先调查非 MoE prefill 的重复反量化与同步

`[CODE]` Q4E 的 `init_quant_linear` 和 fused 初始化路径没有设置 `allow_dense_xmx=true`；`Bnb4BitLinear` 该成员默认 false。`AILA_BNB4_FUSED_PREFILL` 默认 -1，因此按默认配置，多 token 投影走 `Bnb4BitLinear::forward:2245` 附近的路径：

```text
input D2D staging → dequantize_weight → ctx.synchronize()
→ oneDNN matmul → output staging / writeback
```

`cache_dequantized_weight` 默认关闭。就注意力输入/输出与 shared gate/up/down 而言，一次 prefill forward 有 `48×(2+2)=192` 次此类投影调用；另有 PLE 等。**这是默认源码路径的静态调用推导，不是历史日志中 192 次同步的实测确认**，外部 env 可能覆盖 dispatch。

该路径还限制 S3 的 prefill overlap：shared 在 router wait 前提交，但内部 dequant 的 `ctx.synchronize()` 可能已经把队列排空，导致之后 route-event wait 很小。不能看到 `router_sync≈1 ms` 就认为 prefill 的同步开销已经消失。

`[HYPOTHESIS]` 分三级推进：

1. 只打印每种投影实际选路次数，确认默认环境是否命中上述分支；不做全模型 event 工程。
2. 审查 oneDNN stream 与 SYCL queue 的顺序依赖、scratch 生存期，在它们确实共享 in-order 执行时，用已有队列/event 保证 dequant→matmul，避免不必要的 host 全队列 wait。公共算子改动影响其他模型，首版可限制为 Q4E 调用路径。
3. 复用现有 packed NF4 GEMM，选择实际形状 `M=4/26/274`、K/N 如 `2560→16480`、`6144→2560`、`2560→1280`、`640→2560` 做小候选；根据真实语义区分 SIMT fused / BF16 XMX。不能仅设置 `AILA_BNB4_FUSED_PREFILL=1` 就称为 XMX，也不能忽略其 BF16 舍入差别。

先消除同步/整权重临时展开的重复成本，再判断是否需要新 kernel。不要为所有静态投影永久缓存 BF16，避免挤出 compact experts。直接使用连续 input/output 而不经两次 D2D staging，可在这一项中作为后续单独变量测试。

## 9. N3：让 S4 超过“排序 + 更多并行度”

`[HYPOTHESIS]` 第一候选是**只调整 work-group 排列**：在同一专家内先按 output tile，再按该 tile 对应的多条 activation 排布，让相同 packed weight 行的消费者更靠近。保持一个 subgroup 一个 `(token,output-row)`，保持 f32 reduction 与 merge 顺序；不承诺硬件 work-group 执行次序，也不把可能的 cache 命中当作已测流量减少。

如果仍值得继续，再做小 token tile 的显式权重复用：

- 为同专家的 2/4 个 token 保留独立 accumulator，一次 packed load/scale reconstruct 供其消费；优先试有限大小，检查寄存器压力。
- 这不是已有负结果的“一 subgroup 两输出行”；也不重试 decode GEMV 的通用 SLM input cache。任何 SLM 方案按新 tile 重新证明 bank/layout 与同步。
- 冷专家 `m_e=1/2` 仍用 S1/SIMT；只有热专家才试 NF4→BF16 tile→XMX。阈值从少量 `m_e` 样本产生，不能照搬其他模型。
- 有序 merge 不使用 atomic add；若为了更强矩阵化改变点积舍入，作为独立数值实验，不混进等价模式。

利用现有 `expert_counts_host_` 即可输出 m_e 分布，不重新构建 routing 系统。收益分解同时记录 metadata upload、gate/up、down、merge。若大部分收益来自 S1 的提交压缩，则保留更小的 S1，不为“完成 S4”强行增加复杂度。

## 10. N4：预取效用与 actual/prediction 边界

`[CODE]` 当前 `route_event` 被下一层预测 event 覆盖；CPU 在获得实际路由后仍等待预测完成。`prefetch_async` 每次先取消旧任务，acquire 遇到新 demand miss 也会 cancel/drain，无法抢占已进入 mmap page fault 的 part。host-ready 命中、prefetch cancel 成本与 DMA victim 等待尚未在常规 profile 中分清。

`[HYPOTHESIS]` 修 N0 后再做：

1. 分开保留 actual-route 与 prediction-ready event。CPU 等 actual 后先处理 demand，预测结果未 ready 时不读 pinned ids；稍后只在需要开始下一层预取时观察/等待 prediction。仍保持前层 reader 的依赖关系。
2. 对预测候选先过滤 GPU-resident / host-ready，再取最多 48 个缺失候选；当前截断后过滤可能浪费名额。保留稳定同频 tie-break，不改真实 routing。
3. 增加少量计数：host-ready hit、真实 fill miss、required-pf wait、cancel/drain wait、host victim DMA wait、预取实际 bytes/useful bytes。`dma_wait=0` 目前没有涵盖所有新增 wait 路径，不能当作“没有 DMA 等待”的证明。
4. 近期预取效用低、取消浪费高或 fill 延迟大时，降低 quota；有充分后台窗口时再恢复。仍保持 host-only，不重新尝试已失败的 GPU 预测 DMA。

缓存容量固定，不扩 pinned pool，不加线程。若 zero-allocation 是下一步目标，host list empty/occupied 转换也改为固定节点池，不只统计 vector 的 reserve。

## 11. N5 与更远期候选

snapshot 后续采用一个可复用 pinned transfer slab，收齐后再发布 snapshot；可按最大 snapshot 数设计 ownership，逐步去掉 mirror→snapshot 的大拷贝。先保留同步发布边界、anchor 和 restore 语义，不把跨 token 异步状态回读与此项一起改。

静态图优先从稳定的层内段做，当前 router host 交互、expert slot 参数变化和 oneDNN fallback 都不适合直接录制整 token。先完成 N2 的同步收口，并确认本机 limited-graph 能力，不能把另一驱动的 A770 技能证据当作本机保证。

DeltaNet chunkwise/XMX、HC、LM head、AOT 保持上一版的实验定义：分别针对 prefill、已确认的 kernel 成本和启动/JIT；现阶段没有证据要求同时启动。M2 QSA、M3 vision、M4 shared sessions/server 仍独立排期，MTP 继续不做。

## 12. 下一次开发可直接使用的任务说明

> 以 `05055aa` 为基线，先修缓存生命周期：prefetch source 转为 demand DMA reader 时无条件建立 batch membership；修正或移除 cache move；统一 64-part bitmap；为小 GPU cache fallback 保留计算 reader 依赖。每项只补一个确定性边界用例。保持模型算术、默认开关、pool 容量不变。完成后再做按模式分配 workspace 和日志 manifest，选一个有限 A/B 决定 S1/S4 默认策略。随后优先勘察 Q4E 非 MoE prefill 的 dequant + 全队列同步，暂不进入全面 XMX/静态图重写。

验收关闭条件：R1–R3 的实现和定向案例可复查；已有成果与“待优化”边界写清；新的性能结论绑定实际模式及可比 warm 状态。没有必要再次跑完整 57 项、多个其他模型、长篇生成和全量 VTune 来完成这一步。
