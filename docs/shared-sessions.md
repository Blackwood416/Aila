# 共享模型文本会话

当前实现位于 `src/inference`，支持进程内的 Qwen3 与 Qwen3.5 Hybrid NF4 文本推理，共享会话使用 BF16 KV。旧 `InferenceEngine`、CLI 和 C ABI 保留原有入口；两个 NF4 后端的旧入口使用默认会话外观，缓存按需创建。Token API 支持手动步进和自动 [Scheduler](scheduler.md)。本文描述进程内的资源与故障语义；Hybrid 的详细约定见 [qwen35-shared-sessions.md](qwen35-shared-sessions.md)，跨进程接口约定见 [IPC v2](inference-ipc-v2.md)，服务层进度见 [重构路线图](refactor-roadmap.md)。

## 资源与执行约定

| 对象 | 当前实现 |
| --- | --- |
| `DeviceRuntime` | 活动句柄共享默认 SYCL 设备的 Context 和执行锁；覆盖所有设备资源的生命周期 |
| `LoadedModel` | 一次加载的原始/融合权重、模型配置和一个执行槽；创建会话不重新加载权重 |
| `SessionState` | 独立 KV、Hybrid 递归/卷积状态及快照、实际缓存长度和操作租约；持有模型及运行时资源 |
| `RequestState` | 输入进度、生成 token、采样配置、独立 RNG、取消和终态 |
| `Scheduler` | 专用提交线程、轮转队列、准入限制、有界 token 输出和故障传播 |
| 执行槽 | 共享输入/采样缓冲，以及暂时仍位于各模型 backend 内的激活、scratch 和可变 oneDNN 绑定 |

调用者可以在完整 `Step()` 之间手动切换请求，也可以让 Scheduler 自动轮转。每一步执行一个 prefill 分块或一个 decode token，返回前完成该步设备工作。两个后端均在加载时构造固定的 DecoderPlan/LayerPlan，绑定每层投影、缓存索引和下一层/最终归一化；Hybrid 的计划明确区分 Full Attention 与 DeltaNet。执行保留 in-order queue 与已有 NF4/attention/跨层 residual-norm 融合，不在每层之间增加同步。Scheduler 的输出 ring 约束进程内输出；IPC 和 HTTP 传输层需要各自的背压与资源预算。

`DeviceRuntime::Create()` 复用当前活动的默认设备域。这个版本不支持多设备执行，也不承诺与独立构造的旧 `InferenceEngine` 在同一进程内并发执行；部分旧算子仍有进程级缓存。

`ModelOptions::maxSessions` 限制会话数量，默认 4。创建 Session 前同时检查以下额度，最后一个 Session 强引用销毁后归还；计算与 backend 共用 `src/memory/SessionMemory.hpp`，包含维度和 `size_t` 溢出校验。

| 设置或查询 | 当前含义 |
| --- | --- |
| `maxSessionCacheBytes` | 模型下全部 Session 的 device payload 上限；Qwen3 为 BF16 KV，Hybrid 还包含 DeltaNet 递归矩阵和卷积状态；0 表示只用会话数限制 |
| `maxSessionHostBytes` | 全部 Session 的 host mirror 与最大保留快照的 float payload 额度；0 表示只用会话数限制 |
| `maxRecurrentSnapshots` | 每个 Hybrid Session 的快照上限，默认 4，0 禁用快照；Qwen3 不使用此设置 |
| `SessionReservationBytes()` / `SessionState::CacheBytes()` | 每 Session 的 device cache/state 字节数 |
| `SessionHostReservationBytes()` | 每 Session 的 host payload 预留；Qwen3 为 0，Hybrid 为常驻 mirror × `(1 + maxRecurrentSnapshots)` |

host 快照按需分配，额度按最大保留数量预留；禁用快照仍保留 mirror。这些额度不包含权重、执行 scratch、oneDNN/driver 分配、STL/allocator 元数据及网络缓冲，不能当作总内存或总显存准入。`DeviceRuntime::Memory()` 也只返回 Context 记账的 USM。

共享会话的权重仍为 NF4，KV 要求 BF16。Qwen3 检查 `AILA_KV_QUANT=0`；Qwen3.5 检查有效 `AILA_VLM_KV_QUANT`，未设置时回退到 `AILA_KV_QUANT`，显式 scoped 值 0 优先于全局值 1。新接口在加载及显式创建会话时拒绝量化 KV，返回 `InvalidArgument`；旧入口保留原有配置语义。既有 FP8 路径的验证证据见 [refactor-validation.md](refactor-validation.md)。

## 使用步骤

以下代码运行于链接 `AilaLib` 的进程；`promptA`、`promptB` 为已经准备好的模型 token。模板格式化与输出文本解析继续由上层完成。

```cpp
#include "inference/Model.hpp"
#include "inference/Request.hpp"

using namespace aila::inference;

auto runtime = DeviceRuntime::Create();
ModelOptions limits;
limits.maxSequenceLength = 1024;
limits.maxSessions = 2;
auto model = LoadedModel::LoadQwen3Nf4(runtime, modelDirectory, limits);
auto sessionA = model->CreateSession();
auto sessionB = model->CreateSession();

RequestOptions options;
options.prefillChunkSize = 128;
options.generation.max_new_tokens = 64;
options.generation.use_fixed_seed = true;
options.generation.sampling_seed = 42;
options.stopTokenIds = {eosTokenId};
auto requestA = RequestState::Create(sessionA, promptA, options);
options.generation.sampling_seed = 99;
auto requestB = RequestState::Create(sessionB, promptB, options);

while (!IsTerminal(requestA->Snapshot().status) ||
       !IsTerminal(requestB->Snapshot().status)) {
    for (const auto& request : {requestA, requestB}) {
        if (IsTerminal(request->Snapshot().status)) continue;
        const auto result = request->Step();
        // 消费 result.tokenId；result.becameTerminal 只在首次终结时为 true。
    }
}
```

Qwen3.5 使用 `LoadedModel::LoadQwen35Nf4`，其余 Request/Scheduler 接口相同；可通过 `maxRecurrentSnapshots` 和 `maxSessionHostBytes` 设置回滚及 host 预算。

`RequestState::Create()` 将输入追加到已有缓存，要求非空输入，并预留 `已有缓存 + 输入 + max_new_tokens` 的容量。它不推断聊天前缀、不复用其他会话历史。每个 Session 同时只能有一个请求；重复提交及活动期间的 `Reset/Truncate/ForwardLogits` 返回 `ErrorCode::Busy`。`Step()` 也拒绝并发或递归调用。

`ForwardLogits()` 和显式 backend `Forward()` 使用最后一个输入 token 对应的一行 logits。Hybrid 的旧入口仍保留小块增量执行时的多行 logits 约定；共享采样不会把张量容量误认为本次有效行数。

Qwen3 的 `Truncate(length)` 恢复到指定长度；Hybrid 向前截断时返回不大于指定长度的最近保留 checkpoint，没有合适快照时返回 0 并重置。请求长度等于当前长度时为 no-op。共享 Hybrid 默认保留最近 4 个快照，0 禁用；快照时机为多 token prefill 分块结束及既有 decode step 边界，旧默认会话仍使用原有快照策略。恢复包含递归矩阵、卷积历史、环形位置与 host/device 一致性，并移除恢复点之后的快照。调用者必须从返回的实际长度补算历史，不能直接使用请求截断的长度。

`stopTokenIds` 中的停止 token 不发布、不写入缓存。到达输出数量上限后，状态先进入 `Finalizing`；再调用一次 `Step()` 写入最后已发布 token，随后进入 `Completed`。因此正常完成的缓存覆盖全部已发布输出。

`Cancel()` 可以从其他线程调用，取消在提交边界或当前同步步骤结束时生效。取消/失败后保留实际已消费的缓存前缀，可能不包含最后一个已发布 token；上层应查询 `ContextLength()`，按自己的历史协议补算或 Reset。销毁 Request 释放会话租约，销毁 Session 在执行锁内排空设备工作后释放 KV。Session 和 Request 可在外部模型/运行时句柄释放后继续存活。

进程级日志通过有界 `ScopedLogDeferral` 在设备、会话和请求锁释放后投递，允许日志回调读取状态。`Step()` 在所有执行守卫释放前保留请求强引用，即使日志回调释放调用方最后一个句柄，也不会提前析构；回调中的递归 `Step()` 仍返回 Busy。

## 设备域故障与生命周期

执行步骤及 Reset/Truncate 的设备变更异常会归一为 `ErrorCode::RuntimeFault`，使当前设备域失效；其他会话的后续执行也会失败。设备域通过 `RuntimeFaultRecord` 在固定存储中保留首个诊断，最多 1024 字节并保持 UTF-8 边界。同步异常先记录再 drain，成功 drain 不会清除故障，后续 drain/析构异常也不会覆盖首因；空诊断使用默认设备故障消息。

`DeviceRuntime::Create()` 对 Context 构造时的 SYCL/oneDNN 异常使用相同分类；构造失败时尚无已发布的设备域可供排空。`LoadQwen3Nf4`、`LoadQwen35Nf4` 及显式 Session 初始化通过 `RunDeviceLifecycle` 处理已存在设备域上的创建工作：

| 故障位置或类型 | 处理 |
| --- | --- |
| `sycl::exception`、`dnnl::error`、已有的 `Error(RuntimeFault)` | 先记录首因，尝试 drain，再以首因抛 `RuntimeFault`；不会因清理成功而恢复健康 |
| 工作成功后的最终 `Context::synchronize()` | 任意异常都记录为设备域故障，资源不得作为初始化成功发布 |
| 参数、模型配置、文件或主机分配异常 | 本身只影响当前操作；已有 `ErrorCode` 保留，`std::invalid_argument` 映射为 `InvalidArgument`，其余离开生命周期边界的异常保持原类型和内容；若异常清理的 drain 失败，则以设备域故障收敛 |
| Model/Session 析构中的 drain | 不向析构调用者抛异常，将故障保留在设备域中供生命周期管理者检查 |

加载完成后先通过最终 drain，才返回 `LoadedModel`。Session 的临时 cache/state 也只有通过最终 drain 后才转为已发布状态并增加 `activeSessions`；初始化失败不会让未获得额度的 Session 在析构时扣减计数。Busy、ContextOverflow、模型不支持、配置错误和资源准入拒绝不应被调用方一律升级成设备域故障；普通错误后的 drain 正常时，同域已有会话仍可继续使用。这些规则不增加总显存准入能力，内存预算仍按前述 payload 范围计算。

权重加载的局部恢复路径保留上述分类。`SafeTensors.cpp` 的逐 tensor 容错加载会原样重抛 `sycl::exception`，停止该次加载，避免设备上传失败被吞掉后继续返回部分权重；这一分配/上传路径未执行 oneDNN primitive。Qwen3.5 Hybrid 的 tied `lm_head` 预处理同样原样传播 SYCL/oneDNN 异常，由外层加载生命周期归一为 `RuntimeFault`。普通非设备异常仍遵循既有逐 tensor 恢复或直接 embedding 权重回退逻辑。

`DeviceRuntime::FaultMessage() const` 返回 `std::optional<std::string>`：健康域为 `nullopt`，故障域为首因副本，包含不抛异常的析构 drain 留下的诊断。该查询取得设备执行锁，可能等待当前 GPU 步骤，应在负责创建/销毁资源的生命周期或清理线程上调用；transport reader 不应依赖它排空事件管道。它不会清除故障或自行通知 Scheduler。

使用 Scheduler 时，执行步骤的 `RuntimeFault` 会停止准入并终结尚未发布终态的活动/排队请求；加载、初始化或清理线程发现域故障后，应调用同域的 `Scheduler::ReportRuntimeFault()`。该方法不等待设备步骤，保留 `SchedulerSnapshot::faultMessage` 中的首因；已发布 token 保持可读，已发布 terminal 保持原状态，尚未发布的输出在安全边界停止。具体顺序及例子见 [调度器故障报告](scheduler.md#生命周期故障的外部报告)。手动使用 RequestState 时仍由调用者负责推进其他请求到终态。故障域内不自动重试部分输出，也不能通过再建一个 Scheduler 复用旧 Session；需释放旧域资源后重建并重新加载。

## 当前边界与验证

当前 Token API 支持 greedy、固定 seed 采样、已有 repetition/presence/frequency penalties，以及显式停止 token。它继承现有采样算子的行为；不改变旧生成接口的采样规则。`thinking_budget_tokens` 必须为 -1；聊天模板、thinking budget、工具调用和文本流解析尚未迁入这个低层驱动，旧接口继续支持原有能力。

`LoadQwen3Nf4` 与 `LoadQwen35Nf4` 分别接受对应模型的 NF4 文本权重。BF16 权重策略复用、视觉、ASR/TTS、Diffusion 等共享会话能力尚未迁移，继续使用旧入口；Hybrid 媒体 setter 只作用于旧默认 Session。发布显式 Session 后，backend 拒绝 reload 或 LoRA 替换。完整显存预算及 IPC/HTTP 服务层的进度以 [重构路线图](refactor-roadmap.md) 为准；接口约定见 [IPC v2 文档](inference-ipc-v2.md)，独立 HTTP 依赖探针的结果见 [refactor-http-probe.md](refactor-http-probe.md)。

本轮主任务已统一验证 `AilaRuntimeFaultTests` 的 7 组 CPU 用例通过，覆盖同步 SYCL/oneDNN 类型注入、首因不被后续 drain 覆盖、成功工作仍需通过最终 drain、普通 host/file/参数/准入失败保持局部、清理设备故障覆盖普通失败、析构与有界诊断、权重恢复边界。`AilaSchedulerTests` 新增的 3 组生命周期故障用例也已通过，整个测试目标为 22/22；覆盖已完成未读流保留、Submit 准入竞争回滚及日志回调重入报告。CPU 测试使用可注入的异常类型和 drain/gate，不创建 SYCL Context，不代表真实 GPU/driver 故障注入。

构建后运行 CPU 状态/日志/生命周期测试，以及真实模型会话测试：

```powershell
ctest --test-dir build-refactor -C Release -R '^Aila(RequestState|Scheduler|RuntimeFault|DeferredLogging|SessionMemory)Tests$' --output-on-failure
$env:AILA_QWEN3_TEST_MODEL = 'E:\RiderProjects\Aila\models\qwen3-0.6B-bnb-nf4-offline'
$env:AILA_KV_QUANT = '0'
ctest --test-dir build-refactor -C Release -R '^AilaQwen3(Session|Scheduler)Tests$' --output-on-failure

$env:AILA_QWEN35_TEST_MODEL = 'E:\RiderProjects\Aila\models\qwen3.5-0.8B-bnb-nf4-offline'
$env:AILA_VLM_KV_QUANT = '0'
ctest --test-dir build-refactor -C Release -R '^AilaQwen35SessionTests$' --output-on-failure
```

未提供模型时 GPU 用例以返回码 77 标记跳过。GPU 检查与 benchmark 必须串行运行；结果以 [refactor-validation.md](refactor-validation.md) 为准。会话用例使用相同的各自分块形状，对照独立执行与交错执行的每步 logits 和生成 token，并覆盖取消、回滚、资源回收、Busy 和所有者生命周期。Hybrid 额外覆盖旧入口有效末行、快照淘汰/禁用、host/device 额度和 scoped KV 配置。跨源码版本的 CLI 回归应另用 `perf/compare-token-traces.ps1` 比较完整 trace；测试不能用不完整日志宣称 token 一致。
