# IPC v2：共享推理的独立 Worker

日期：2026-09-09。当前实现覆盖 Qwen3 NF4 与 Qwen3.5 Hybrid NF4/BF16 token 推理；实际验证结果与构建身份见 [refactor-validation.md](refactor-validation.md)，BF16 远端接入见第 18 节。协议字段和完成判据见 [inference-ipc-v2.md](inference-ipc-v2.md)。

## 接入范围

`--ffi-worker-v2` 启动模式与 `src/runtime/InferenceClient.hpp` 内部 C++ API 使用一个设备域、一个 Scheduler 和最多一个 Qwen3 NF4 或 Qwen3.5 Hybrid NF4/BF16 模型，共享模型权重，最多创建四个独立 BF16 KV Session。Hybrid 各自持有 DeltaNet 递归矩阵、卷积历史及 host mirror；输入输出均为 token ID。

旧 `--ffi-worker`、`WorkerProcess` 和公共 C ABI 默认继续使用 protocol 1；显式启用 `AILA_LEGACY_TEXT_V2=1` 后，[旧文本适配器](legacy-text-adapter.md) 通过独立异步 C ABI 接入 v2。v2 的 protocol 为 2，ABI 仍为 1。聊天模板、reasoning 和工具解析由宿主处理；IPC 仍仅传 token。请求级 [thinking budget](thinking-budget.md) 通过可选能力传递可信标记元数据；音视频和运行中 adapter 替换尚未迁移。

## 代码边界

| 组件 | 职责 |
| --- | --- |
| `ipc/InferenceProtocol.*` | 独立的 v2 编解码、严格 schema、协商限额、LE token 编码与 terminal/final 核验；不依赖推理或设备头文件 |
| `runtime/InferenceClient.*` | 普通 C++ 宿主 API；常驻 command sender、response reader、event reader 和连接超时处理；每个请求独立路由、累计 ACK 与自动排空 |
| `runtime/InferencePipeTransport.cpp` | Windows 子进程、受限句柄继承、运行目录/PATH/build ID 校验、prefix 限长和可取消管道 IO |
| `worker/InferenceWorker.*` | 有界命令入口、对象 registry、生命周期执行器、Scheduler bridge、公平事件发送和消费额度 |
| `worker/SharedInferenceEngine.cpp` | 将真实 `DeviceRuntime/LoadedModel/SessionState` 接到 Worker 的模型工厂 |
| `worker/InferenceWorkerMain.*` | 每条 pipe 的独立 writer、持续 command reader、连接级写超时；同一外壳也用于无 GPU 的真实管道测试 |

`AilaInferenceClient`、`AilaInferenceProtocol` 和 `AilaInferenceWorkerCore` 是普通 C++ 静态目标。真实 Worker 链接 `AilaLib`；宿主 API 不链接它。CPU fake Worker 复用同一控制器、Scheduler 和管道外壳，只替换模型/Session 工厂。

## 宿主 API

```cpp
#include "runtime/InferenceClient.hpp"

using namespace std::chrono_literals;
using namespace aila::runtime;

auto client = InferenceClient::Start(runtimeDirectory, workerExecutable,
                                     expectedBuildId);
InferenceModelOptions modelOptions;
modelOptions.maxSequenceLength = 512;
auto model = client->LoadQwen3(modelDirectory, modelOptions);
// 对 Qwen3.5 Hybrid NF4，改用 client->LoadQwen35(modelDirectory, modelOptions)。
// 对未量化 Qwen3.5 BF16，改用 client->LoadQwen35Bf16(modelDirectory, modelOptions)。
auto session = model->CreateSession();

InferenceRequestOptions options;
options.generation.maxNewTokens = 32;
options.generation.doSample = false;
auto request = session->Submit(promptTokenIds, options);
for (;;) {
    auto read = request->Read(100ms);
    if (read.status == InferenceReadStatus::Timeout) continue;
    if (read.status == InferenceReadStatus::Closed) break;
    if (read.event->terminal) {
        // Inspect status/endReason and localFailure, then stop consuming.
        break;
    }
    ConsumeToken(read.event->tokenId); // Runs on this caller, never an IPC reader.
}
client->Shutdown();
```

对不同 Session 提交请求即可重叠执行。一个 Session 只允许一个执行租约；同 Session 的重叠 submit 返回 Busy。空闲时 `Session::Query()` 提供精确 contextLength，忙时返回 null。正常 Length 结束包含最后 token 的缓存写入，随后可以向原 Session 追加 prompt；取消或失败的会话可能缺少最后一个已发布 token，调用者不能把输出长度当作精确缓存长度。

| 宿主入口 | `backend` | 权重 / KV | 模型能力 |
| --- | --- | --- | --- |
| `LoadQwen3` | `qwen3_nf4` | NF4 / BF16 | `qwen3_nf4_bf16_kv` |
| `LoadQwen35` | `qwen35_nf4` | NF4 / BF16 | `qwen35_nf4_bf16_kv` |
| `LoadQwen35Bf16` | `qwen35_bf16` | BF16 / BF16 | `qwen35_bf16_bf16_kv` |

默认 requiredCapabilities 保留原 Qwen3 集合；hello/configure 可以额外发布 Hybrid 与 BF16 权重能力，加载前检查已配置的能力，模型返回值再核对所请求后端。BF16 复用相同的 wire 字段、资源预算和生命周期执行器；不修改 protocol/ABI 版本，也不自动转换权重。缺少可选能力时本次加载局部失败，不发送 `model.load`，已支持的 NF4 连接仍可使用；configure 返回 hello 未声明的能力或模型返回其他权重后端能力属于协议故障。实际进程连接仍要求匹配的 build ID，此处兼容描述不取消版本身份校验。

v2 通过可选 `session_context_v1` 提供 `InferenceSession::Reset()` / `Truncate()`，返回 `wire::SessionInfo`。两种权重格式的远端 Hybrid 仍固定 `maxRecurrentSnapshots=0`，非平凡截断可能恢复到 0；调用者须根据返回的实际 `contextLength` 重放缺失前缀。host mirror 必须保留，以最多四个 Session 约束其数量，并可通过 `allocationLimits.maxSessionHostBytes` 限制总 payload；`maxSessionCacheBytes` 只管 device cache/state payload。进程内 Hybrid 的快照和回滚 API 保持原行为。

上下文操作先在控制线程预留 Session，再由生命周期执行器调用原有 `SessionState::Reset/Truncate`。预留覆盖排队和响应发布，同 Session 的重复变更/submit 返回 Busy，query 在排队期间也返回 busy/null。操作及 completion 持有 Session/Model，父句柄提前释放不会使设备资源提前析构。变更占普通命令额度，RPC 超时只结束本地等待，不撤销已提交的变更。设备故障在生命周期线程立即广播，排队的后续变更不会再进入 backend；关闭需安全排空任务后才能完成。

`Request::Read(timeout)` 仅让本次等待超时。`Cancel()` 产生独立 RPC，保留可读输出；取消响应不等于请求终态。`Release()` 非阻塞，标记取消并由私有路由丢弃、累计确认剩余输出。请求执行器在完整 prefill/decode 步骤边界处理取消。

`InferenceSession::Prefill(ids, chunkSize)` 使用可选 `prefill_only_v1`，返回普通异步 Request，只执行输入而不采样。成功读到 `Completed/Length`、generatedTokens=0；取消后应查询真实 contextLength 再补算。普通 Submit 仍要求正数 maxNewTokens。

请求可独立选择 `thinkingBudget.closeOnce`、`forceHostSampling` 和 `legacyPrefillLogits`，分别协商 `thinking_budget_close_once_v1`、`host_sampling_v1`、`legacy_prefill_logits_v1`。旧文本适配器使用这些策略保留旧强制关闭、host sampling 和 Hybrid 短续写多行 logits 的首行采样行为；新 API/HTTP 默认仍使用最后有效行。策略随 Request 传递，不修改共享 Model 或其他 Session 的设置。

## 生命周期与完成

`InferenceModelOptions::allocationLimits` 可选请求 `allocation_limits_v1` 扩展，设置显式 device/pinned USM 以及全部 Session host mirror 的上限。Client 在加载前检查协商能力，并核对 response 的 `memory` 是否精确回显所有上限；没有请求时不会增加命令/响应字段。加载时 residency 与 Session 预留的含义、局部额度失败及兼容边界见 [执行资源](execution-resources.md)。

`InferenceClient::QueryMemory(timeout)` 使用独立的 `runtime_memory_v1` 查询当前账本。Worker生命周期忙时立即返回busy，其余查询在生命周期执行器调用 `DeviceRuntime::TryMemory()`；不等待执行锁或访问GPU队列。查询进入普通额度，取消/ACK保留容量保持。公开C接口和HTTP缓存采样复用这条路径，见 [运行时分配快照](runtime-memory.md)。

Model、Session、Request 宿主句柄均绑定 connection owner、128-bit 随机 epoch、对象类型与 ID。销毁 Client 外壳不提前关闭仍被派生句柄使用的 Worker。每次 Worker 重启生成新 epoch，不能把旧对象 ID 绑定到新进程。

释放父句柄只关闭该客户端引用。Worker 将最后设备引用交给生命周期执行器销毁；资源的实际存活数量包含已经关闭父句柄但仍被子对象使用的资源，也包含尚未完成的创建/回收任务。每条传输流继续保存原 Model/Session ID。

每个成功 submit 在 response pipe 上先有 accepted，再有 final；事件在独立 event pipe 上传输。宿主在写 submit 前登记路由，所以允许事件早于 accepted 到达。只有 accepted、final、terminal、连续 sequence、eventCount 和完整摘要全部一致，才能向用户交付 terminal。

Scheduler 的执行完成、bridge 读取 terminal、pipe 写完、宿主读取和用户消费分别记账。bridge 必须强持有 `ScheduledRequest`，不能通过析构它实现 release：析构会 detach 并丢弃 core ring。IPC 额度保留到 terminal ACK，且 Worker 的相关输出写入槽也已完成；消费终态后尚未 release 的小型快照继续占 retained-handle 额度。

## 有界输出与慢消费者

默认 tokenWindow 为 32，core token ring 为 128，最多 64 条未排空请求，最多 256 个保留请求句柄。默认每请求 prompt/generated/stop ID 上限为 4096/1024/64；总 prompt + maxNewTokens 预留上限为 32768。

显式设置 `InferenceClientOptions::limits.maxGeneratedTokens` 为 1025–32768 时，Client 自动要求 `extended_generation_v1`，在 configure 阶段申请准确额度。默认 hello/schema/限额保持原样，未支持的 peer 在 configure 前失败；Worker 降低的上限和所有其他资源边界仍保留。旧文本 adapter 按 context capacity 申请此扩展，从而让 1200-token 等请求通过一次完整 Request 保持 RNG、penalty history 和 thinking 状态，不拆成多个生成请求。

每端 ingress/transport 预算默认各 8 MiB，日志队列 64 KiB。Worker 为控制响应、日志与保留句柄预留预算，再接受新的流；每流最多有一个 staged/in-flight token 帧。日志回调只做有界入队，不写 pipe。超出日志预算时丢弃并计数。configure 不能把日志上限降到已经排队或正在发送的数据以下。

Client 通过 `ReadLog(timeout)` 提供单独的有界日志消费，按消息的保守字节数和固定开销记账，空消息也占额度。host 队列满时丢弃新日志；`Snapshot` 提供 queuedLogs、logQueueBytes、peakLogQueueBytes 和独立的 droppedLogs。调用者在自己的线程处理消息，host reader 不执行回调。Shutdown/连接故障后仍可排空，Client 释放则关闭日志流并唤醒读取，已有子句柄继续运行。`SetLogLevel` 使用可选 `worker_log_control_v1`，校验 0–3 与精确响应；无需改变默认 hello 或原日志事件形状。

窗口按用户实际消费或明确放弃的连续前缀累计 ACK。宿主 reader 持续排空 pipe，但不会因为刚收到 token 就确认消费。A 停止读取时最多占自己的窗口和 core ring；继续产 token 导致 ring 满会取消 A，B 仍可运行。短请求或只剩 Finalizing 的请求可正常完成，因此背压验证使用不设置 stop IDs 的 256-token 长请求。

这些是有界 payload 预留，不是总进程 RSS 或总显存测量。模型权重、共享执行 scratch、oneDNN/驱动分配、allocator 元数据和内核管道缓冲区不属于 Session 缓存额度。物理 pipe 的整个宿主停止读取属于连接故障；独立客户端的隔离依赖宿主专用 reader 持续工作。

## 故障与关闭

加载/初始化会保留首次同步 SYCL/oneDNN 异常，即使后续 drain 成功，也会归一为 RuntimeFault。加载器的 best-effort tensor 恢复不会吞掉这些设备异常。普通文件、配置或 host allocation 失败仍保持局部语义；若随后的设备 drain 也失败，则升级为域故障。

生命周期执行器捕获 RuntimeFault 后调用 `Scheduler::ReportRuntimeFault()`，停止准入并广播；析构 drain 的故障通过 `DeviceRuntime::FaultMessage()` 继续传播。已发布终态不改写，未结束请求在安全边界失败。domain.fault 事件不删除既有输出路由，ACK/release 仍可收尾。损坏 framing、计数失配、管道故障或 Worker 退出会失效整个连接，宿主为尚未交付终态的请求选择一次本地失败终态，不伪造未知的 wire eventCount。

`Shutdown()` 由并发调用共享同一次关闭过程。宿主停止新调用并放弃未消费 token，继续私有排空与 ACK；本地终态记录 `deliveryAbandoned`、实际交付数和丢弃数。Worker 等待 Scheduler、生命周期回收、全部 terminal ACK、短命令响应和事件写屏障，才返回含 lastEventOrdinal 的关闭结果。宿主再验证自己读到该 ordinal，并检查子进程退出。超过 grace 或子进程退出失败均报告连接失败，不报告优雅关闭成功。

默认连接写超时为 10 秒，最大 shutdown grace 为 30 秒。它们是连接级保护，不能保证抢占正在执行的 GPU kernel；单个 Request 等待超时不会杀掉共享 Worker。

宿主保留 I/O 线程句柄；显式 Shutdown 在解除连接状态锁后真正 join 线程，再返回关闭结果。初始化失败也等待线程退出，避免动态卸载 DLL 时线程仍在销毁捕获对象或退出 CRT trampoline。仅释放最后 owner 仍触发异步清理；要卸载 DLL 的调用者必须先 Shutdown，再释放全部句柄并结束在途 API 调用。

## 验证入口

- `AilaInferenceProtocolTests`：v1/v2 独立、prefix/UTF-8/JSON 整数/重复字段/附件/能力/摘要边界。
- `AilaRuntimeFaultTests`、`AilaSchedulerTests`：设备异常分类、首次原因、生命周期广播及既有调度行为。
- `AilaInferenceWorkerTests`：CPU Session 工厂下的准入/回滚、Busy、父子生命周期、背压、释放、故障、日志降限与关闭。
- `AilaInferenceClientTests`：受控双通道传输下的跨管道乱序、消费/释放/关闭状态机。
- `AilaInferencePipeTests`：实际 Windows 子进程和 pipe 的 CPU 集成，以及普通宿主的运行时隔离。
- `AilaQwen3IpcSessionTests`：真实 Qwen3 直接调用对照、1/2/4 会话、固定 seed/penalties、续写、取消、长请求停读、重启 epoch。
- `AilaQwen35IpcSessionTests`：同一可执行文件以 `--backend qwen35_nf4` 验证 Hybrid；使用 `AILA_QWEN35_TEST_MODEL`，另覆盖不同 prefill 块与四槽释放重建。可重复传 `--case concurrency|cancel-backpressure|session-recreate|parent-shutdown|epoch` 筛选，默认运行全部。
- `AilaQwen35Bf16IpcSessionTests`：同一 fixture 以 `--backend qwen35_bf16` 验证未量化 Hybrid，使用 `AILA_QWEN35_BF16_TEST_MODEL` 或 `--model`。直接参考与远端均选择 BF16 权重，模型能力还必须排除 NF4；用例选择、逐 token 相等和资源归零判据保持一致。

GPU 测试与 benchmark 必须串行执行。既有 IpcProtocol、WorkerDispatcher、RuntimeIsolation、ProxyAbi、ProxyLogging 与 BuildIdentity 保留为 v1 兼容门禁；文本/聊天与 HTTP/SSE 的实现和验收独立记录。
