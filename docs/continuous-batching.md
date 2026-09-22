# 连续 decode 合批

2026-09-12：最新独立 decode 行投影已完成构建，四种模型/权重配置的完整 logits、B1 对照、Request 重放、前向后 ResourceLimit 恢复及真实 GPU HTTP 均已通过本轮有限功能验收。最终主 CPU 组 20 项（含旧 ABI）、Server CPU 组 4 项、trace 38 项及 policy trace 6 项检查通过，MSVC SDK 和 HTTP Fake Worker 复核通过。集中压力、性能测量和长期稳定性仍留待后续阶段。

此前复用 prefill 多行投影的两轮实现未通过既定数值门禁。原始失败证据与计算语义差异见 [重构验证记录](refactor-validation.md) 第 33 节，不作为当前行为。最终独立 decode 路线的已验收结果见本页末节，原数值门限保持不变。

## 开关与适用范围

`SchedulerOptions::maxDecodeBatchSize` 和 `InferenceClientOptions::maxDecodeBatchSize` 默认均为 `1`。合法值为 `1..4`，同时不得超过 `maxActiveRequests`。`1` 保持原来的单请求步骤；大于 `1` 允许调度器将已就绪的兼容 decode 请求组成一批，上限不表示每步必须凑齐该数量，也不引入等待凑批的定时窗口。

服务通过 `--decode-batch-size N` 配置。Server 的活动请求上限取自 `--sessions`，因此该选项也不能超过会话槽数。`/health`、启动 ready 消息和退出报告中的 `max_decode_batch_size` 是配置上限，不是实际批次大小或吞吐统计。

本轮实现面向 Qwen3 NF4、Qwen3.5 Hybrid NF4 和 BF16 的独立文本 Session，KV 仍为 BF16。同一批必须拥有相同的 `LoadedModel` 实例；匹配依据是模型对象身份，不是 backend 名称、模型目录或相同的权重尺寸。不同模型、设备域及不支持合批的后端不能混入同批。

当前公共 `ModelOptions`、C ABI 模型选项和 Server 都没有 adapter 加载或选择参数。“固定 adapter 身份”描述的是后端权重/LoRA 在建立显式 Session 后不得改变的架构约束，不代表公共接口已经提供 LoRA 服务，也不代表任意 LoRA 在全部模型上的合批均已验收。运行中替换 LoRA、多 LoRA 共批、跨模型和多设备合批不在本接口内。

Qwen3 在 `AILA_PROFILE_Q3_DECODE=1` 时回退到完整的单请求路径，保留旧 profiling 行为。

`AILA_FUSE_RESIDUAL_ADD=1` 时，Qwen3 和两种 Hybrid 后端均回退到完整 B=1 路径，保留原残差融合语义。配置上限仍可为 2..4，但这些 Session 不进入多请求执行批次。

Hybrid 以下旧模式保持完整的单请求路径，不进入合批：

- `AILA_PROFILE_Q35_DECODE=1`。
- `AILA_DEBUG_Q35_LAYER_STATS=1`、非负的 `AILA_DEBUG_Q35_LAYER_DETAIL`，或 `AILA_DEBUG_Q35_LINEAR_COMPARE=1`。
- 层计划包含旧 `LegacyLinearAttention`，例如选择旧 linear attention 路径。

不能合批的请求继续按单请求步骤推进。`continuous_batching_v1` 表示 Worker 支持配置和调度契约，不保证任意后端、模式或某个请求必定执行多行投影。

## 每步执行与状态归属

| 请求状态 | 一个调度步骤的行为 |
| --- | --- |
| Queued / Prefill | 独立执行一个 prefill chunk；不把不同请求的 prompt 拼成同一条序列 |
| Decode | 每个参与 Session 消费自己的一个 pending token，使用自己的位置、历史和采样状态 |
| Finalizing | 消费最后一个已发布 token，使完成后的 Session 可以续写；该行不采样、不发布新 token |
| 已取消或终结 | 在安全边界移出执行集合；已终结请求不再推进缓存或生成第二个终态 |

最早就绪的请求作为当前调度步的起点，同模型的就绪 decode/finalizing 请求可以加入；本步结束后存活请求移到队尾。未选中的 prefill 或其他模型请求随后获得调度机会，已完成请求腾出的活动槽允许排队请求进入。混合长短请求不要求相同位置或相同 context length。

执行批次中的每一行代表一个独立 Session，不是单序列的相邻 token。QKV（Hybrid 包含 linear-all）、输出投影和 FFN down 通过独立的 `ForwardDecodeRows` 分发，NF4 与 BF16 的实际执行方式见下节。RMSNorm、残差加法＋RMSNorm 均逐行调用原 count=1 内核。Gate/Up 逐行调用原 fused helper；不可用时调用 count=1 的原投影和 SwiGLU fallback，保留单行 decode 的权重及激活舍入边界。Attention、LM head 和采样同样保留逐序列语义。

Hybrid 的卷积历史、DeltaNet 递归矩阵和 host mirror 仍属于各自 Session；行视图与回写必须保持该对应关系，没有 padding 行更新他人状态。

每个请求独立保存生成历史、seed/RNG、stop token、thinking budget 和强制关闭序列。Finalizing 与强制 thinking-close 行可以和普通 decode 行共存；不需要采样的行不推进 RNG。单行批次直接调用原来的单请求执行路径，保留 B=1 的内核选择和行为。

## 当前投影路由与工作区

NF4 的 `Bnb4BitLinear::ForwardDecodeRows` 专门处理独立 decode 行。符合原 packed decode 条件、输入宽度 K≥512、B=2..4，且 `AILA_BNB4_GEMV_WG` 为默认 0 或 256 时，在一个 kernel launch 中按序列划分 work-group。每个序列保留原 SG16、WG256 的 vec8/FMA dot 与归约顺序，直接读取 packed NF4 权重并在线去量化。Qwen3 的 QKV/O/down，以及 Hybrid 的 linear-all/linear-output/full-QKV/full-output/down 使用该入口；B 不作为 prefill token 数传入。

B=1、小 K、其他 work-group 覆盖或 packed 路径不可用时，入口逐行调用原 `forward(..., 1)`。是否使用原 cached/dequant 路径仍取决于原单行条件；不能仅凭打开 cache 配置就推断 packed 权重已不可用。后端已有的 LoRA 附加计算沿用逐行逻辑，不改变前述公共 API 的 adapter 范围。

Hybrid BF16 的 `ForwardDecodeRows` 在 packed JM 权重与自定义 decode 条件可用时使用多行 kernel（`run_decode_jm_matvec_rows`、`run_decode_ffn_gate_up_swiglu_rows`），否则逐行调用原 count=1 投影，保留 JM、自定义 decode 和 Linear fallback 的选择。多行 kernel 与逐行 kernel 逐位一致，细节与实测见下文“BF16 多行投影”。NF4 的单次 launch 组织方式也不等同于已经测得吞吐提升。

默认 packed NF4 decode 行投影不调用 `Bnb4BitLinearScratch::ensure`，不分配整权重 BF16 dequant scratch。无 adapter 的常规投影本身使用已有输入/输出缓冲；因此，全体请求只经过 1-token prompt 后首次进入合批，不会因把批宽误当作多 token prefill 而新建整权重暂存。回退路径和已有 adapter 中间区须按其原分配规则计量。

多 token prefill 仍使用原投影和 workspace 路由，包括其环境开关所选择的 packed GEMM 或去量化＋oneDNN 路径。它可能分配并保留 BF16 weight scratch；后续进入 decode 不会自动释放已经预热的 workspace。HTTP 模板化 prompt 应依据实际 prefill 分块计量，不能把常规 decode 无整权重暂存解释为整个请求没有这项分配。

批次的逐行激活和 token/logits 缓冲仍可能增长，四行 BF16 logits 的保留量为 `4 × vocab_size × sizeof(bf16)`。这些工作区与模型权重、KV/递归状态都受各自的现有限额约束；完整峰值还包括分配替换时的重叠、oneDNN 和驱动私有资源。当前没有以静态维度估算代替设备内存测量，也未宣称合批总显存或性能已经通过验收。

## 取消、背压、失败与资源

取消在完整 forward 后的请求状态检查点生效。步骤开始前已经观察到取消的行不提交新的 forward；执行期间到达的取消不能中断已提交的设备工作，在该检查点处理该行，并丢弃本次采样结果。其他行继续使用自己的输出状态。`Cancel` 不是流事件屏障：如果当前步骤已经越过取消检查点，准备好的 token 仍可能在 `Cancel` 返回后交付，例如在 deferred log callback 中取消的情况；之后的步骤不再生成 token。已交付和缓冲中的 token 仍可读取，终态表示执行收尾；释放 Request 则沿用既有的取消、排空/丢弃及累计 ACK 规则。

同一 Session 仍只允许一个活跃 Request，重叠提交返回 `Busy`。每个请求继续拥有独立的输出 ring、token credit 和保留终态槽。输出 ring 已满时，仅将需要继续发布 token 的对应请求标记为 `OutputBackpressure`；Finalizing 不产生 token，仍能完成缓存收尾。终态与 final 响应保留原来的关联、连续 sequence、eventCount 和一次交付约定。

参数、Busy、context 和准入错误在提交前按请求拒绝。一次已开始的共享批执行如果整体失败，则结束该次调用中尚未终结的成员；不会把无法确认状态的行静默重试。普通执行异常的影响止于该组请求，设备 `RuntimeFault` 则沿用整个设备域的故障传播，停止新执行并保留首因。发布任何行之前检查批次中的域故障；已经发布的终态不改写。取消或已经结束的行不因随后发现的组内错误得到第二个终态。

实际模型执行中遇到 `AllocationLimitExceeded` 时，先排空已提交工作，再将本次所选 Session 全部 reset 到空 context，随后报告 `ResourceLimit`。这些 Session 不保留该次调用前的缓存，复用前须重新提交完整上下文；未参与该组的 Session 保留原有 KV/递归状态。排空或重置中发现设备故障则按设备域故障处理。

合批不扩大活动/在途请求、Session、context、总 token、cache 或输出额度。安全收尾后，活动槽释放和终态可见在同一调度队列事务内完成；即使同批其他行的析构尚未结束，也不出现 active 请求数超过未完成请求数的状态。调度额度在读取终态或安全丢弃后归还，IPC 额度继续保留到其消费/ACK 条件满足。释放父句柄仍需等待子请求和执行引用安全结束，不能凭 release 应答认定设备分配已经回收。

新增投影 workspace、token/logits 暂存仍归模型/设备执行资源所有，显式 USM 分配受现有 device/pinned 限额控制。USM 账本不能代表 oneDNN、驱动私有分配或全部 WDDM/RSS；完整资源计量与准入仍是独立工作。

## IPC、C++ Client 与 C ABI

IPC 保留 protocol 2、public ABI 1 以及原有 `Limits` schema。`worker.configure` 输入和成功响应增加可选的 `max_decode_batch_size` 字段：

- 缺省值为 `1`；序列化值为 `1` 时省略字段，默认 Client 不要求新能力。
- 大于 `1` 时，Client 自动要求 `continuous_batching_v1`。旧 Worker 缺少能力时在发送 configure 前拒绝；直接发送缺少所需能力的扩展配置也会被拒绝。
- 字段必须是整数，范围为 `1..4` 且不超过选定的 `maxActiveRequests`；重复键、未知字段和错误类型仍严格拒绝。
- Worker 精确回显选定值，Client 拒绝缺失后退为 `1`、静默降低、提高或未请求的配置。配置在 Worker 生命周期内固定。

进程内 C++ Client 在启动时设置：

~~~cpp
aila::runtime::InferenceClientOptions options;
options.maxDecodeBatchSize = 4;
// 将 options 传给 InferenceClient::Start 或 Connect。
~~~

异步 C ABI 仅新增一个函数，旧 `AilaInferenceClientOptions` 仍为 104 字节，reserved 必须为零：

~~~c
int32_t aila_inference_client_start_with_batching(
    const char* runtime_directory,
    const char* worker_executable,
    const AilaInferenceClientOptions* options,
    uint32_t max_decode_batch_size,
    AilaInferenceClientHandle* out_handle,
    AilaInferenceError* error);
~~~

旧 `aila_inference_client_start` 等同于新函数传入 `1`。新函数复用旧的 UTF-8 路径、版本、错误和句柄约定，失败时清零有效的 out_handle；非法批上限在启动 Worker 前拒绝。旧文本适配器没有隐式启用此开关。详情见 [IPC 协议](inference-ipc-v2.md)、[异步 C ABI](async-c-api.md) 和 [C++ Server](inference-server.md)。

## 已验收结果与剩余检查

最终 CPU 复核为主组 20 项（包含旧 ABI）及 Server 组 4 项全部通过，另有 38 项 trace 与 6 项 policy trace 契约检查通过。MSVC C/C++ 头文件、布局、默认值、UTF-8 错误及未修饰导出检查通过：旧 `AilaShared.dll` 的 45 项导出完整保留，`AilaInference.dll` 从 27 项增加至 28 项，唯一新增导出为 `aila_inference_client_start_with_batching`。HTTP Fake Worker 的默认、显式 1、2、4 配置均通过；该后端验证配置和串行回退，原生合批另由以下设备用例验收。

独立 NF4 行投影组件通过 50 项逐位对照，覆盖 B=1..4；与 CPU 数值参考的 7 组比较共检查 24,848 个值，27 项非法输入拒绝保持输出和 USM 账本不变。完整模型使用独立 `Forward` Session 作为参考，按固定输入逐步比较全部 logits：

| 模型/权重配置 | 完整 logits 行 | 比较值数 | 最大绝对误差 | 最大归一化 RMS 误差 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3 0.6B NF4 | 72 | 10,939,392 | 0 | 0 |
| Qwen3.5 0.8B NF4 | 72 | 17,879,040 | 0 | 0 |
| Qwen3.5 Hybrid BF16 文本夹具 | 72 | 17,879,040 | 0 | 0 |
| Qwen3.5 4B NF4 | 72 | 17,879,040 | 0 | 0 |
| 合计 | 288 | 64,576,512 | 0 | 0 |

四种配置的 greedy 排名均未变化。独立冻结 B1 与当前产物的对照中，每种配置的参考侧和候选侧各运行 12 个请求、生成 96 个 token，结果完全一致。

Request 设备用例在四种配置上均通过 `4→3→2→1` 动态缩批、不同采样参数与独立 RNG、Finalizing、取消和后续重放；每种配置有 7 行独立重放，合计 28 行。批内重排以及取消一个请求后，其余行的完整输出和最终 context 均与对应参考一致。

四种配置还通过前向后的 pinned-host 额度拒绝用例：预热 B3 后，完整前向和第 0 行的 greedy 读取已完成，第 1 行首次 host sampling 在 1 字节 pinned 上限处得到 `ResourceLimit`。三条所选请求（含 Finalizing 行）均 reset 到空 context、释放 Busy，已经算出的第 0 行结果不发布，失败终态仅产生一次；非成员的完整生成及续写与独立 B1 oracle 一致。所选成员随后复现完整参考 logits，并完成新的参考请求。拒绝前后的 USM/pinned 账本不变，释放模型后显式 device/pinned 字节和分配数量归零，runtime 无 fault。原 trace 的 `sampling` 与 `decode_batch` 两处 error 状态完整保留，由专用负例契约核对前向、采样、排空、三次 reset 与关闭顺序。

四种配置的真实 Server/Worker HTTP 各通过 7 项功能用例，共 28 项：上限 1 与 4 下的单请求 JSON/SSE 结果一致，1/2/4 并发的请求各自完成；断连请求取消、其余请求继续完成、请求槽复用及活动请求关闭均通过，并检查每个请求唯一的保留 outcome。两个 Worker 的 trace 分别关联到 B1 参考与上限 4 的功能阶段。

## 吞吐、延迟与持续运行测量

2026-09-18 在同一台 A770（驱动 32.0.101.8860）上，用与上述数值验收完全相同的 Worker/Server 产物，通过真实 HTTP/SSE 服务测量了批上限 1 与 4 的差异。每个完成请求仍由 runner 与自己的单请求独立参考逐字段比对，因此打开批上限不会静默改变服务输出。测量命令使用新增的 `--decode-batch-size`，runner 会核对 health/ready/report 回报的配置上限与请求值一致。

并发 4 的 batch tok/s（同一进程内 4 个请求的完成 token 总量除以首请求开始至末 `[DONE]`，含请求准备与传输，不是 decode kernel 吞吐）：

| 模型/权重配置 | 输入 | 上限 1 | 上限 4 | 变化 |
| --- | --- | ---: | ---: | ---: |
| Qwen3 0.6B NF4 | 短 | 96.33 | 122.20 | +26.9% |
| Qwen3 0.6B NF4 | 短 | 95.13 | 118.70 | +24.8% |
| Qwen3.5 0.8B NF4 | 短 | 118.61 | 148.37 | +25.1% |
| Qwen3.5 0.8B NF4 | 短 | 118.74 | 149.69 | +26.1% |
| Qwen3.5 0.8B NF4 | 长短混合 | 85.16 | 96.83 | +13.7% |
| Hybrid 0.8B BF16 夹具 | 短 | 127.13 | 138.13 | +8.7% |
| Hybrid 0.8B BF16 夹具 | 长短混合 | 95.40 | 99.84 | +4.6% |
| Qwen3.5 4B NF4 | 短 | 45.09 | 50.25 | +11.4% |
| Qwen3.5 4B NF4 | 长短混合 | 36.17 | 39.40 | +8.9% |

表中前两行与后两行是重复测量（不同证据目录、各 10 轮），用于显示同一配置的批间波动。并发 2 与长短混合的完整分布见 `tmp/refactor/p7-batching-perf-20260918/p7-batching-performance-summary.json`。

上限 4 对 `B=1` 的执行路径没有影响：并发 1 时各配置的变化在 ±1.4% 内，与重复测量本身的波动同量级，且数值等价性已由 288 行完整 logits 零差异单独证明。表中 BF16 的 +8.7% 来自调度与行间交错：当时 `ForwardDecodeRows` 对 BF16 仍是逐行调用原投影。2026-09-21 补上 BF16 多行投影后，该配置的收益提高到 +57.0%，见下节。

实际执行批宽由 Worker 自己的执行 trace 直接证明，而不是由吞吐反推。2026-09-20 对四种配置分别运行同一套 HTTP 合批夹具，trace 记录的多行批次数分别为 Qwen3 0.6B NF4 48 次（宽 2×28、3×4、4×16）、0.8B NF4 58 次（宽 2×47、3×9、4×2）、Hybrid BF16 50 次（宽 2×30、3×4、4×16）、4B NF4 48 次（宽 2×28、3×4、4×16）；四轮夹具全部用例通过，两份 Worker trace 均完整。此前 4B 在 HTTP 路径未观察到宽 4，本轮已观察到，该缺口关闭。配置上限、并发请求数与某一步的实际执行批宽仍须分别解读。

近容量长 prompt 也通过验收：在 `--context 4096` 下用 190 次重复文本构造长输入，模板化后 Qwen3 实际 3725 token、0.8B NF4 为 3916 token，均远高于短输入的 27 token。批上限 1 与 4 各自作为独立进程运行，并发 4 的长短混合中上限 4 相对上限 1 为 Qwen3 +8.8%（31.27→34.01）、0.8B NF4 +11.7%（32.20→35.97）；单请求下两配置也各有 +5.6%/+9.0%。长 prompt 的 prefill 仍按原有分块轮转，合批只作用于 decode 行。

持续运行方面，四种配置各自在 `--sessions 4 --decode-batch-size 4` 下通过 512 次生命周期请求（完成/断连交替，4 并发），每进程预热后至最终空闲的 private bytes 增长 ≤ 37 MiB、句柄增长 ≤ 2，`failed`/`rejected`/`slow_client_cancellations` 均为 0。Qwen3 与 0.8B NF4 另有 150 轮 × 4 并发短输入扩展运行，两者均通过且无框架错误日志。

小时级长稳由背靠背的独立进程完成（runner 单进程 watchdog 上限 3600 秒，因此不把一次超长进程当作覆盖）。全部进程 `failed`/`rejected`/`slow_client_cancellations` 为 0、无框架错误日志、final report 正常关闭、Job 排空无强杀：

| 配置 | 进程 | 正式请求 | 累计时长 | 每进程 p50 tok/s |
| --- | ---: | ---: | ---: | --- |
| Qwen3 0.6B NF4 短输入 | 2 | 8024 | 1.09 h | 130.60 / 130.54 |
| Qwen3.5 0.8B NF4 短输入 | 6 | 7752 | 0.98 h | 153.45–154.77 |
| Hybrid 0.8B BF16 夹具短输入 | 6 | 7752 | 0.76 h | 200.70–204.09 |

同配置内的 p50 波动为 0.05%（Qwen3）、0.86%（NF4）与 1.68%（BF16），说明所测时长内无持续退化；这不等于已验证无上限的长期行为，也不是跨天运行。

本轮保留两项未通过证据，不作为达标样本：一次 Qwen3.5 0.8B NF4 的 5016 请求合批 soak 虽然全部请求达到预期结局（2512 完成、2504 断连、无失败/拒绝/慢客户端取消、final report 正常关闭、Job 排空），但被 runner 的框架 ERROR/FATAL 日志门禁判为失败——3 条 trantor 断连诊断与 [HTTP 探针](refactor-http-probe.md) 中已记录的 `0`/`10053` 同类；另一次 HTTP 合批夹具的 `disconnect_peer_and_slot_reuse` 用例超时，同产物重跑即通过，判定为间歇性时序敏感而非确定缺陷。两次证据原文保留，日志门禁未放宽。

以上是受限测量：小时级覆盖为每个配置约 0.76..1.09 小时、共三个配置，未覆盖跨天运行、更多配置的长时间连续运行及驱动私有资源；吞吐含服务传输与观测开销，不能当作 kernel 吞吐；显式 USM 观测也不代表驱动私有分配或全部 WDDM/RSS。

## BF16 多行投影

2026-09-21 为 Hybrid BF16 的 `ForwardDecodeRows` 增加多行 kernel：`run_decode_jm_matvec_rows`（QKV/O/linear-all/linear-output/down）与 `run_decode_ffn_gate_up_swiglu_rows`（gate+up+SwiGLU）。原实现逐行调用 count=1 kernel，24 层 × 每层 4 个投影下，4 行一步要发 384 次 launch 且每行重读整份权重；多行版本按 96 次 launch 完成，与 NF4 的路由方式一致。网格 group 编号为 `row * rows + sequence`，使读同一权重行的各序列相邻，便于 L2 复用。

数值上每个 `(row, sequence)` 仍使用原 subgroup size、原 vec8 载入、原 FMA 顺序和原 `reduce_over_group`，因此与逐行 kernel 逐位相同；`in_dim` 非 8 的倍数时回退逐行 kernel 以保证序列偏移保持 vec8 对齐，`rows==1` 仍走原入口。四种配置的完整 logits 对照（每配置 72 行、共 288 行，与独立 `Forward` Session 比较）全部 `failed_values=0`、`max_abs=0`、greedy 未变；`ctest` 54 项 100% 通过（14 项 GPU fixture 在 CPU 组跳过）；BF16 HTTP 合批夹具通过并观察到执行批宽 2/3/4（多行批次 58 次）。

同一 `--sessions 4`、10 轮测量下，BF16 上限 4 相对上限 1 的收益由改动前的 +7.5%/+8.7%（138.33 vs 127.13 tok/s）提高到 +57.0%（204.12 vs 130.00）；长短混合由 +4.5% 提高到 +35.0%（132.44 vs 98.11）。B=1 路径未变（短输入 91.05→91.04）。证据目录 `tmp/refactor/p7-batching-perf-20260918/`，汇总 `p7-bf16-rows-summary.json`（含源码与产物 SHA256）。

## 重复测量分布

此前各配置只有单轮进程，不能区分收益与批间波动。2026-09-21 对四种配置各运行三个独立进程（24 个进程、每个 10 轮，短输入），并发 4 下上限 4 相对上限 1 的均值：

| 模型/权重配置 | 上限 1 | 上限 4 | 变化 | 两臂跨进程极差 |
| --- | ---: | ---: | ---: | --- |
| Hybrid 0.8B BF16 夹具 | 129.46 | 200.77 | +55.1% | 1.0% / 0.3% |
| Qwen3.5 0.8B NF4 | 119.02 | 148.92 | +25.1% | 0.8% / 2.0% |
| Qwen3.5 4B NF4 | 45.97 | 51.87 | +12.8% | 4.0% / 0.1% |
| Qwen3 0.6B NF4 | 102.69 | 125.91 | +22.6% | 1.0% / 3.2% |

并发 1 时四配置的变化为 -0.8%/-0.7%/+0.2%/+0.7%，两臂跨进程极差 0.1%..4.0%；也就是说并发 4 的收益（+12.8%..+55.1%）远大于观测到的波动，而单请求路径在噪声内不变。原始 p50 值见 `p7-repeat-measurements.json`。

数值与功能阶段的原始运行、guard、最终 CPU 日志和各组 `audit.json` 位于 `tmp/refactor/p7-continuous-batching-20260912/`，汇总见 [重构验证记录](refactor-validation.md)；23 组正式 GPU 功能验证全部通过，审计确认对应 Job 已排空且未发现该阶段新增系统故障。2026-09-18 起的吞吐、集中运行、近容量与小时级证据位于 `tmp/refactor/p7-batching-perf-20260918/`，其中 `p7-batching-performance-summary.json` 与 `p7-bf16-rows-summary.json` 汇总全部通过轮次、执行批宽、保留失败与产物哈希。P7 的功能边界、吞吐/延迟收益、近容量长 prompt、小时级长稳与 BF16 多行投影均已有独立执行证据；batched/ragged attention、分页 KV、跨天长期运行及完整驱动私有资源计量仍未完成，本页不表示 P7 全阶段或整个重构完成。

## 多行 decode attention

2026-09-21 为 Hybrid decode 批次增加多行 attention：`attention_decode_rows`（实现在 `attention_decode_rows_baseline_impl`）把同一步的多个独立序列合并到一次 launch，取代每步每层每序列一次 `attention_decode`。每个 `(sequence, head)` 仍沿用原 baseline kernel body（原 vec8 K 点积、原 `reduce_over_group` 顺序、原 V 累加循环），因此单行输出与逐行 kernel 逐位相同。

覆盖范围刻意收窄到 head_dim 256、无 decode window/sink、且每行都低于 exact-partial-merge 阈值（1024）的情形。head_dim 128 优先使用 joint-matrix 或 exact partial merge，所以 Qwen3 保持逐行派发；达到阈值的行也回落逐行。多行 kernel 只复刻 baseline body，不会替换任何可能落到其他 kernel 的行。`AILA_ATTN_DECODE_ROWS=0` 可在同一二进制上恢复逐行调用。

同二进制交替 A/B（批上限 4、并发 4、各 10 轮、三轮交替）显示 Hybrid BF16 夹具 +3.9%（208.76 vs 200.90 tok/s）、0.8B NF4 +2.8%（155.32 vs 151.15）；六组配对全部倾向多行，幅度 +2.2%..+4.8%，两臂区间不重叠。收益幅度不大的原因是 attention 只占批次单步约 20%，且减少的是 launch 与主机提交开销而非算术量。四配置完整 logits 仍逐位一致、`ctest` 54 项全通过。自查中还修复了两处潜在错误：回落路径与多行 kernel 分别用传入的 `q_row_stride` 读输入、用独立的 `o_row_stride = num_heads*head_dim` 写输出。四个配置都启用了 `attn_output_gate`，两者恰好相等，所以原对照看不见该差异，但未启用门控的模型会错位；修复后四配置 logits 仍逐位一致。

## 多行精确合并覆盖

2026-09-22 把多行 decode attention 从 baseline 扩展到 exact-partial-merge：新增 `attention_decode_rows_exact_merge_impl<Quantized, 128|256>`，每个 `(sequence, head, tile)` 复用单行 merge 算术体，partials 区域改为每序列私有的 `[4, head, tile, stride]`。派发对齐单行调度：`head_dim 256` 低于阈值走多行 baseline，两个 head_dim 的达标行走多行 merge，其余（decode window/sink、head_dim 128 低于阈值时的 joint-matrix、no partials、跨阈值混合行）保持逐行。`AILA_ATTN_DECODE_ROWS=0` 仍可整体恢复逐行。

四配置新增近阈值长上下文批次（capacity 2176、prefix 2100..2115、B=4），每配置 12 行逐位相同、`greedy_changed=0`、无 `failed_values`；路径日志直接记录所选内核。期间修复一个真实缺陷：长行曾被按 baseline body 计算（partials 的 head 维误判），NF4 长上下文随即不再逐位相同；修正后四配置全部通过，`ctest` 54 项 100% 通过。原失败证据保留。见 [重构验证记录](refactor-validation.md) 第 35 节。

## 多行 Joint-Matrix Decode Attention

2026-09-22 补齐了此前注意力派发遗留的最后一条逐行回退路径：在 `head_dim == 128` 且所有序列长度低于 128 tokens 时，原先单序列采用 XMX Joint-Matrix，多序列批次由于未实现多行 JM 会退化为按行逐个派发。

新增 `attention_decode_rows_joint_matrix_impl<TM=1, TN=8, TK=16, SG=8, WG=128>`：
- **SLM 融合设计**：由于上下文长度 < 128，全序列 logits 最多仅 128 个 float（512 字节），完全分配在工作组本地内存（SLM `scores_slm`）中，彻底消除对全局 `scores_buf` 的 DRAM 写入与重读。
- **网格与硬件对齐**：派发网格为 `(row_count * num_heads * WG, WG)`，每个 work-group 独占一个 `(sequence, head)`。WG=128、SG=8 形成 16 个 sub-groups，恰好并行覆盖最多 16 个上下文 tile（128 / 8 = 16），Phase 1 矩阵乘（DPAS QK^T）全并行执行；Phase 2 的 Softmax 在 128 线程内协同归约；V 向量累加由前 16 个线程使用 `vec8` 向量载入并行写入 `output`。
- **派发集成**：在 `attention_decode_rows` 中接入检测与派发，路径日志显式输出 `[ATTN] decode rows=N head_dim=128 path=multi-row-joint-matrix`。将此前短上下文下的最多 8 次独立 Kernel 提交与全局中转，合并为 1 次全批次提交。

