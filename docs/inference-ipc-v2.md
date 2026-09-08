# 共享推理 IPC v2 实施协议

日期：2026-09-08。**状态：Qwen3 与 Qwen3.5 Hybrid NF4 token 入口均已接入 IPC v2。** codec、Worker controller/管道外壳、普通 C++ `InferenceClient`/Win32 transport 及设备故障处理共用一套实现，旧 Worker/Proxy 和公共 C ABI 保持 v1。普通 C++ [HTTP/SSE 服务](inference-server.md) 已使用这两种后端。实现入口见 [IPC v2 实现记录](inference-ipc-v2-implementation.md)，Qwen3 首片验证快照保留在第 10 节，各批实际证据见 [refactor-validation.md](refactor-validation.md) 第 10–12 节，阶段安排见 [refactor-roadmap.md](refactor-roadmap.md)。

本文保留 v2 的协议契约。实现位于 `src/ipc/InferenceProtocol.*`、`src/runtime/InferenceClient.*`、`src/runtime/InferencePipeTransport.cpp` 和 `src/worker/{InferenceWorker,InferenceWorkerMain}.*`，通过 `SharedInferenceEngine.cpp` 接入已有 Model/Session/Scheduler。进程内资源与调度语义见 [shared-sessions.md](shared-sessions.md) 和 [scheduler.md](scheduler.md)。协议枚举使用稳定字符串，不直接发送 C++ 枚举的整数值或结构体内存。

## 1. 首片边界及兼容模式

- 一个 v2 Worker 拥有一个 `DeviceRuntime` 设备域和一个 `Scheduler`，最多加载一个 Qwen3 或 Qwen3.5 Hybrid NF4 模型；共享 BF16 KV Session，Hybrid 同时隔离递归/卷积状态和 host mirror，按完整 prefill/decode 步骤轮转。已有 NF4 算子、融合和 in-order queue 不变。
- v1 `--ffi-worker`、`WorkerDispatcher`、`CApiWorkerEngine` 继续提供旧命令。新增私有启动模式 `--ffi-worker-v2`；同一进程启动后固定模式，不在已建立的管道上切换协议。
- v2 使用 `protocol=2`；`abi=1` 仍表示现有公共 C ABI 版本。IPC 版本与公共 ABI 版本分别管理。现有 `kProtocolVersion=1` 不应直接全局改为 2，让旧调用方无意中发送新协议。
- 新的宿主端内部 C++ API 选择 v2；旧公开函数先继续使用 v1。首片不把旧 `generate_chat_json` 翻译成 token 请求，也不声称已迁移聊天模板、reasoning、工具解析、音视频或 thinking budget。
- 旧文本流回调仍在发起旧同步调用的线程执行。ASR/TTS 派生句柄继续强持有 `ProxyEngine` owner；销毁 `AilaEngine` 外壳不等于提前销毁派生句柄的 Worker。既有 TTS 自身的线程约定也不改变。
- 将来旧文本 API 适配 v2 时，只等待自己的请求，并在原调用线程消费该请求的事件；不能把用户回调搬到 IPC reader、sender 或 Scheduler 线程。未迁移的旧 Engine 不与新 Scheduler 在同一进程内同时执行。

选择独立启动模式的依据是当前 `WorkerProcess` 严格校验每帧 protocol/ABI，且响应与事件有独立 reader。原地升级还需处理两条管道中遗留 v1 日志的切换屏障，首片没有引入这项复杂度的必要。新宿主遇到不支持 v2 的 Worker 明确返回 `UnsupportedProtocol`，不静默退化为串行推理；现有 build ID、运行目录、可执行文件和 PATH 隔离校验继续执行。

## 2. 握手与固定能力

Worker 在创建 GPU 资源前发送唯一 `kind=handshake, method=worker.hello, requestId=0` 的 v2 响应帧。payload 必含现有身份字段 `buildId/executable/runtimeDirectory/currentDirectory/path`，以及：

| 字段 | 类型与含义 |
| --- | --- |
| `epoch` | OS 生成的 128 位随机值，32 个小写十六进制字符；每次 Worker 启动重新生成 |
| `supportedProtocols` | 首片为 `[2]`；表示当前启动模式支持的协议 |
| `capabilities` | 实际可用的字符串集合；当前为 `token_requests_v1`、`qwen3_nf4_bf16_kv`、`qwen35_nf4_bf16_kv`、`session_lifecycle_v1`、`multi_inflight_v1`、`cumulative_ack_v1`、`domain_epoch_v1` |
| `hardLimits` | 本文准入表的上限；包含帧、对象、调度及传输限制 |

宿主先保留原身份校验，再发送 `worker.configure`，携带 `epoch`、`requiredCapabilities` 和不超过 hardLimits 的 `limits`。Worker 返回选定的完整 limits 和 capabilities；宿主确认能力齐全、数值合法后进入 Ready。未知必需能力返回 `UnsupportedCapability`，超上限参数返回 `InvalidArgument`，不能悄悄改变请求中的限制。未配置时只允许 configure、worker.query 和 shutdown；configure 成功后不能再次修改域的限额。

握手读取先使用现有有界 bootstrap 帧限制；configure 后按 v2 的较小限制读取。默认宿主必需能力保留原 Qwen3 集合，新增 Hybrid 是可选扩展；`LoadQwen35` 检查 configure 结果含 `qwen35_nf4_bf16_kv`，缺少时局部拒绝，不发送加载命令。configure 不能返回 hello 未声明的能力，model.load 的成功结果必须含所请求后端能力。能力协商不放宽 build ID 校验。当前不发布 FP8 KV、运行中 LoRA 替换、文本解码或 HTTP wire 能力；聊天与 HTTP 由宿主层完成。

## 3. 帧、关联与句柄标识

沿用现有二进制 framing：`u32_le headerBytes`、`u32_le attachmentBytes`、UTF-8 JSON header、attachment。保留现有 header 字段：

```json
{
  "protocol": 2,
  "abi": 1,
  "requestId": 27,
  "kind": "request",
  "method": "request.cancel",
  "payload": { "epoch": "0123456789abcdef0123456789abcdef", "inferenceId": 19 }
}
```

这里 header 的 `requestId=27` 是 **RPC 命令编号**，目标生成请求是 `inferenceId=19`。不得把一次 cancel 的 RPC 编号复用成 submit 的编号。

| 标识 | 分配方及约定 |
| --- | --- |
| RPC `requestId` | 宿主单一 command sender 分配；按实际发送顺序严格递增、非零，不重用。0 仅用于 hello、域事件和日志 |
| `inferenceId` | 等于创建该请求的 submit RPC 编号；宿主在写帧前建立对应请求路由。它是有独立类型的请求句柄，不能用作 model/session ID |
| `modelId`、`sessionId` | Worker 在成功创建对象时分配，各自类型内递增、非零，epoch 内不重用 |
| Scheduler `Id()` | Worker 内部标识；保留 `inferenceId -> ScheduledRequest` 映射，不将二者假定为相同数值 |
| 完整句柄 | 宿主本地 connection owner + `epoch` + 对象类型 + 对象 ID；只传数值 ID 不构成可用句柄 |

ID 使用 JSON 无符号 64 位整数，双方用整数解析，不经过 double；计数检查使用减法/除法防溢出。ID 用尽后停止准入并重启域，禁止回绕。epoch 放在每个 v2 payload，包含成功、错误、日志和事件；非当前 epoch 的命令返回 `StaleEpoch`。宿主不能把旧句柄重新绑定到一个恰好分配了相同数值 ID 的新 Worker。

每条 pipe 只有一个 writer，完整帧禁止字节级交错。command sender 发送前登记 pending，再写帧；并行调用通过队列进入，不能各自读 response pipe。有效 framing 中的无效参数属于命令错误；损坏 framing、非法 envelope 身份、重复 RPC 编号、错误 epoch 的响应和无法关联的结果属于连接协议故障。payload 重复字段、未知参数和错误类型明确拒绝，避免参数被静默忽略。

## 4. 对象所有权与完成的含义

Worker registry 区分客户端句柄引用和执行中的强引用：

| 对象 | 必须持有的资源 | release 行为 |
| --- | --- | --- |
| Model | 模型权重、设备域、共享执行槽 | 关闭客户端 Model 句柄，禁止由它创建新 Session；已有 Session/Request 继续持有资源 |
| Session | Model、独立 KV、会话元数据 | 关闭客户端 Session 句柄；已接受的 Request 继续执行，不因父句柄释放而隐式取消 |
| Request | `ScheduledRequest`、原 Model/Session 标识、传输额度、事件计数 | 关闭用户句柄，取消未结束的生成，进入内部自动排空；已投递事件仍由私有路由处理 |
| Connection | pipe、reader/sender、registry/pending、epoch | 所有派生宿主句柄及正在收尾的内部路由强持有 connection core |

关闭父句柄的响应只确认引用释放，不承诺 GPU 字节已回收。实际 Model/Session 数量和缓存额度要在最后强引用释放、设备工作结束后归还；不能通过反复 release 父句柄绕过 `maxLoadedModels/maxSessions`。

每个 Session 最多有一个提交中或执行中的 Request；同一 Session 的再次 submit、reset、truncate 返回 `Busy`。已经结束但输出未消费的 Request 不再持有执行租约，因此相同 Session 可以开始下一次生成，但旧流仍占传输额度。registry 为每个 Session 串行化 submit 与会话变更，避免先检查 Busy、后执行之间发生竞争。

分别维护以下状态，不能只用一个 `done`：

- `executionStatus`：`queued/prefill/decode/finalizing/completed/cancelled/failed`，来自 Scheduler。
- `deliveryStatus`：`pending_acceptance/open/terminal_received/drained/abandoning/released`，属于传输层。
- `domainStatus`：`starting/ready/stopping/faulted/closed`，影响整个 Worker。

Scheduler 的 `Read(terminal)` 会归还其调度额度。IPC sender 读到 terminal 仅代表输出已转移到传输层，所以 Worker 必须另持传输额度，直到收到终态消费确认。执行完成、发出最终响应、pipe 写入完成、宿主读取、用户消费是五个不同的时点。

## 5. 命令 schema

除 hello 外，以下 payload 都包含 `epoch`；下表省略它。所有方法都可在 command reader 持续运行时接收，耗时操作交给独立队列。除 submit 外，每个 RPC 恰有一个 `result` 或 `error` 响应，回显 method 和该命令的 header requestId。

| 方法 | 输入 payload | 成功响应的专用字段 |
| --- | --- | --- |
| `worker.configure` | `requiredCapabilities: string[]`, `limits: object` | `capabilities`, `limits`, `domainStatus: "ready"` |
| `worker.query` | 无 | `domainStatus`, Scheduler 的 accepting/stopping/faulted 和各额度计数，传输计数；不遍历所有请求输出 |
| `model.load` | `backend: "qwen3_nf4" / "qwen35_nf4"`, `modelDirectory: string`, `kvType: "bf16"`, `maxSequenceLength: u32`, `maxSessions: u32`, `maxSessionCacheBytes: u64` | `modelId`, `vocabSize`, `maxSequenceLength`, `sessionReservationBytes`, 所加载后端的实际能力 |
| `session.create` | `modelId` | `sessionId`, `modelId`, `capacity`, `cacheBytes`, `contextLength: 0` |
| `session.query` | `sessionId` | `modelId`, `capacity`, `cacheBytes`, `busy`, `contextLength: u32/null`；Busy 时长度为 null，不能假称得到精确缓存前缀 |
| `request.submit` | 见下文；header requestId 即 inferenceId | 一次 accepted，再一次最终 result/error，见下一节 |
| `request.cancel` | `inferenceId` | `cancelRequested: bool`, 最近 executionStatus；已终结时为 false，不改变原终态 |
| `request.query` | `inferenceId` | executionStatus、endReason、promptTokensProcessed、generatedTokens、bufferedTokens、lastSentSequence、lastAcknowledgedSequence、completionReady、eventCount 或 null、errorCode/message |
| `request.ack` | `inferenceId`, `consumedSequence: u64` | `acknowledgedSequence`；累计确认，不是新增额度数 |
| `handle.release` | `type: "model"/"session"/"request"`, `id: u64` | `releaseAccepted: true`；Request 可以仍处于内部收尾状态 |
| `worker.shutdown` | `graceMs: u32`，不得超过配置的关闭上限 | `domainStatus: "closed"`, `lastEventOrdinal`，仅在设备收尾及事件发送屏障后返回 |

`session.reset` 和 `session.truncate {sessionId, length}` 尚未加入。两者只允许空闲 Session；未来 truncate 响应必须含实际 `restoredLength`，不能将请求的 length 当作返回值。当前通过销毁并新建 Session 完成取消后的重用。Hybrid 远端固定关闭 recurrent snapshots，但保留续写所需的 host mirror；device reservation 含 Full KV、递归矩阵和卷积状态，host mirror 受 Session 数量限制，未纳入 v2 device 字节额度。

`session.query` 的 idle 精确读取也要先取得会话操作资格，不能让新 submit 与查询的长度读取交错。Busy 快照不等待 GPU。model.load/session.create/destruction 可能取得设备锁，必须在 lifecycle executor 上执行，禁止在 command reader 内分配 KV、加载权重或析构最后设备引用。

`request.submit` 的 payload 示例：

```json
{
  "epoch": "0123456789abcdef0123456789abcdef",
  "sessionId": 2,
  "promptTokenCount": 256,
  "promptEncoding": "i32le",
  "prefillChunkSize": 128,
  "stopTokenIds": [151645],
  "generation": {
    "maxNewTokens": 128,
    "doSample": false,
    "temperature": 0.6,
    "topK": 20,
    "topP": 0.95,
    "useFixedSeed": true,
    "samplingSeed": 42,
    "repetitionPenalty": 1.0,
    "presencePenalty": 0.0,
    "frequencyPenalty": 0.0
  }
}
```

attachment 必须恰为 `promptTokenCount * 4` 字节，逐个 little-endian int32 解码，且 token 在词表范围内。禁止复制本机 `int` 内存作为跨进程协议约定。首片只有 token IDs，没有文本、媒体或任意附件；stop IDs 放在有界 JSON 数组中。

generation 字段在 wire 上全部显式填写，宿主 convenience API 可以使用当前 `GenerationConfig` 的默认值补齐。映射到同名 C++ 参数，验证 finite、采样温度、topP/topK 和 penalties 的规则与 `RequestState` 一致。`prefillChunkSize` 必须在协商上限内，prompt 非空，maxNewTokens 正数，已有缓存 + prompt + maxNewTokens 不超过 capacity。thinking/decodeChunk/streamChunk 等未迁移字段不接受，内部 `thinking_budget_tokens` 固定为 -1。当前 mt19937 固定 seed 使用 samplingSeed 的低 32 位，v2 首片保留并记录这个行为，不擅自改变采样结果。

准入依次预留宿主路由与传输预算、Worker 传输预算、Scheduler 预算和 Session 租约。任何一层失败都回滚之前的预留。只有 `Scheduler::Submit` 成功且 Worker request entry 已发布后，才允许 accepted 或事件离开 Worker。解析和所有内存预留完成前不得发 accepted；已接受后发生的失败必须作为有终态的请求收尾。

## 6. accepted、事件和最终响应

成功 submit 的 response pipe 有两个不同阶段：

1. `kind=accepted, method=request.submit`，payload 含 `phase: "accepted"`、epoch、inferenceId、modelId、sessionId 和选定的 tokenWindow。它只确认准入。
2. 执行终结后，`kind=result/error, method=request.submit`，payload 含 `phase: "completed"` 和下述完整终态摘要。取消是正常请求结局，可使用 result；`executionStatus=failed` 使用 error。

准入拒绝只有 `kind=error, phase="rejected", eventCount=0`，带稳定的 errorCode/message，不分配生成流，不发 terminal。所有成功 accepted 的请求，包含排队取消和请求 release，内部协议流都恰有一个 terminal 和一个最终响应；明确的连接故障除外。

生成事件走 event pipe：`kind=event, method=request.event`，header requestId 等于 inferenceId。payload 至少包含：

| 字段 | token | terminal |
| --- | --- | --- |
| `epoch/inferenceId/modelId/sessionId` | 必须有，保持原归属标识 | 相同 |
| `event` | `"token"` | `"terminal"` |
| `sequence` | 从 1 连续递增 | 最后一个 token sequence + 1 |
| `eventOrdinal` | 此 pipe 中所有成功提交给 writer 的事件的全局递增编号 | 同左；仅用于关闭屏障，不替代请求 sequence |
| `tokenId` | 合法的单个 token ID | 不存在 |
| `executionStatus` | Scheduler 事件的状态 | completed/cancelled/failed |
| `promptTokensProcessed` | 当前已消费的 prompt 数 | 最终值 |
| `generatedTokens` | 该请求已发布的 token 数 | 最终发布数，包括尚未被远端消费的 token |
| `endReason/errorCode/message` | 不存在 | 原样映射 Scheduler 终态；message 原文最多 1024 字节 |
| `eventCount` | 不存在 | 必须等于自身 sequence，即 generatedTokens + 1 |

终态摘要还包含 `eventCount`、executionStatus、endReason、errorCode/message、promptTokensProcessed、generatedTokens；最终响应与 terminal 的摘要必须逐字段一致。它不重复返回整个 token 数组。endReason 为 `length/stop_token/cancelled/output_backpressure/scheduler_stopped/runtime_fault/failed`；errorCode 有 core 错误码时使用其稳定字符串，Failed 且无 core 错误码时使用 `RequestFailed`，无错误时为 null。停止 token 不发布、不缓存，正常 Length 完成包含 Finalizing，取消/失败后缓存可能不含最后一个已发布 token。

Worker bridge 可以从已经终结的 ScheduledRequestSnapshot 冻结摘要，并先排入最终 response；它随后仍需保留 ScheduledRequest，把所有剩余 token 和 terminal 按原 sequence 发送。不能在 result 发出后销毁它。snapshot 中的 `generatedTokens` 是已发布到 Scheduler 流的计数，不能以“已写入 pipe 的数量”代替。

response 与 event 两条 pipe 之间没有先后保证。宿主 pending 表在 submit 写入之前就存在，因此能够暂存先于 accepted 到达的事件；不能用“result pipe 的 accepted 已读到”作为建立事件路由的前提。response pipe 自身保持 accepted 先于最终响应。

每请求 reader 保存 `acceptedSeen/finalSeen/nextSequence/terminalSeen/expectedEventCount`。正常完成判断必须同时满足：

```text
acceptedSeen
AND finalSeen
AND terminalSeen
AND receivedSequence == terminal.sequence == final.eventCount
AND terminal.eventCount == final.eventCount
AND final.eventCount == final.generatedTokens + 1
AND terminal 摘要与 final 摘要相同
```

只在上述条件成立后才向用户暴露可消费的 terminal；token 可以提前消费。terminal 后 Read 为 Closed。final 先到、terminal 先到以及 A/B 任意交错都是正常情况，不等待其他请求完成。重复/跳号事件、terminal 后 token、计数越界、摘要不一致是协议故障，不能标为成功完成。日志和域事件使用 requestId=0，不计入任何请求的 eventCount；但它们同样携带 eventOrdinal，参与整个 event pipe 的关闭屏障。

例如收到 `A token 1、B token 1、A final(count=4)、B terminal 2、A token 2、A token 3、A terminal 4` 时，A 必须等到 sequence 4 且摘要匹配后完成。B 则独立等待自己的 final，不能因为 A 的 final 已到就结束 B。

## 7. 有界额度、累计 ACK 和公平发送

以下是首片实现的固定默认值，不是性能保证；握手宣告实际值。计数与字节限制同时生效。

| 限制 | 首片默认 | 归还或适用边界 |
| --- | ---: | --- |
| Scheduler active/outstanding | 4 / 64 | 沿用现有 Scheduler；outstanding 包含 core 终态尚未被读取的流 |
| prompt/generated/stop IDs | 4096 / 1024 / 64 | 每请求；生成事件总数最多 1025 |
| outstanding token reservation | 32768 | prompt + maxNewTokens；IPC 镜像额度直到终态 ACK，不因 core 提前归还而放宽 |
| prefill chunk/core token ring | 512 / 128 | 沿用现有 Scheduler |
| maxLoadedModels/maxSessions | 1 / 4 | 包含父句柄已关闭但仍被子对象强持有的资源 |
| Model/Scheduler cache byte limit | 配置值，默认 0 | 只计 Session 缓存，0 表示使用数量上限；不是完整显存预算 |
| IPC outstanding requests | 64 | accepted 前预留；终态消费 ACK 或连接关闭后的安全清理才归还 |
| retained Request handles | 256 | 已消费终态但用户未 release 的小型快照也占此额度；避免远程 registry 无界增长 |
| tokenWindow | 32 | 每请求最多 32 个已发送、尚未 ACK 的 token；host 预留同等容量 |
| encoded token/terminal/final bytes | 1 KiB / 8 KiB / 8 KiB | 每帧含 prefix/header；terminal/final 各有独立预留，不抢占 token 容量 |
| command header/attachment | 64 KiB / 16 KiB | 首片附件仅为最多 4096 个 token；读取 prefix 后、分配 body 前检查 |
| pending short RPC | 64，其中 16 个控制预留 | submit 在 accepted/rejected 后释放短 RPC 槽，最终响应由 Request entry 跟踪 |
| ingress/transport bytes | 每端各 8 MiB | ingress 与 output 分别记账；包括已解析输入、待发帧、路由 ring、终态/响应槽及未完成写入 |
| log queue bytes | 64 KiB | 超额丢弃并累计 droppedLogs，不能阻塞推理日志回调 |

取消、ACK、release、查询、关闭使用控制预留，加载和 submit 不能占满全部控制槽。每 Request 预留一个可合并的待 ACK 标记和取消/释放标记；event reader 更新标记，不等待 sender 空位。sender 将标记物化为普通有编号 RPC，兼顾控制与普通命令，不能让查询风暴永远饿死正常 submit。对端违反协商 in-flight 窗口时按协议故障收敛，不能无界积累拒绝响应。

### 数据通路

1. Scheduler 仅写自己的固定 token ring；它不调用 pipe、HTTP 或用户回调。
2. 一个 Worker bridge 作为每条 ScheduledRequest 的唯一消费者，轮询 `Read(0ms)`。只有该请求还有 token credit 和预留 staging 空间时才取 token；每请求最多暂存一个待写 token。无数据时用通知或短定时等待，不为每个请求创建阻塞 reader 线程。
3. 一个 event sender 对可发送请求轮转，每轮最多发送一个 token 帧。已取 token 必须保留直至写入成功或整个连接失败；不能用再一次 Read 重试。terminal 仅在同请求的前序 token 已提交 writer 后发送，使用独立的终态预留；没有 token credit 时也可发送已经没有前序 token 的 terminal。
4. 宿主专用 event reader 持续读 pipe，只校验和路由到每请求的预留 ring，绝不执行 HTTP socket 写入或用户回调。每请求窗口为 32，宿主至少给它保留 32 个 token 槽、一个 terminal 槽及一个 final 摘要槽；正常对端不可能让单条请求占满其他请求的空间。
5. 用户 Read 消费 token 后，宿主累计 `consumedSequence`，发送 `request.ack`。可每 16 个 token 或最多 10 ms 合并一次；消费 terminal 后立即 ACK。ACK 只有在本地实际消费或明确 abandon 丢弃后才能前进，不能在刚收到事件时就自动确认。

Worker 以 `sentTokenCount - acknowledgedTokenCount + stagedTokenCount < tokenWindow` 判断能否再从 core 取 token，并在 Read 前原子预留该 credit。这里 sentTokenCount 包含已经提交 writer 的在途 token，stagedTokenCount 为 0 或 1；staging 转交 writer 时只转移记账，不能重新增加窗口。ACK 重复或较旧值无副作用；大于已提交 writer 的 sequence 或越过未消费前缀的值非法。terminal 不需要新增 token credit，但它的 ACK 必须覆盖此前所有 token，而且宿主只有验证了 final 后才允许消费 terminal。

如果消费者停读且请求还需继续产 token，会先用完该请求的 32 个跨进程 token credit，再填满自己的 core ring；现有 Scheduler 在下一次产 token 步骤前以 `OutputBackpressure/ResourceLimit` 取消该请求。停读本身不保证触发取消：短请求可能在填满之前正常完成，已经进入 `Finalizing` 的请求即使 ring 满也会继续写入最后 token 的缓存并完成，因为这一步不发布新 token。bridge 不能继续把 ring 抽到一个无界 vector 中。已经 buffered 的 token 和独立 terminal 仍按序保留；这个请求会占额度，其他请求继续轮转。执行已完成但未读终态的流也占 IPC outstanding/token 额度。

当用户/HTTP 客户端主动断开时，`handle.release(request)` 将宿主路由转为 abandoning，立即停止向用户回调；私有 reader 丢弃该请求已收和后续 token，并以相同顺序 ACK。Worker 调用 Cancel，仍以有界窗口发送剩余已发布数据、terminal 和 final，直到正常排空判据成立；随后释放传输额度、请求 registry 和私有 owner。closing Request 仍接受清理用 ACK，新的用户操作返回 InvalidHandle；重复 release 在收尾期间无副作用，宿主对已经结束的本地 release 不再重复发命令。这样 release 不需要制造 sequence 缺口，也不会因两条 pipe 的在途帧而误报未知请求。局部 release 不等待用户读流，其他 Session 不受影响。

上述私有排空由 v2 宿主路由与 Worker bridge 协作完成。bridge 保留 `ScheduledRequest` 的强引用，Cancel 后继续读取并完成有界传输及终态确认，不能直接销毁该对象：`ScheduledRequest` 析构会调用 `Detach()`、丢弃 ring 并请求取消，不能再据此发送完整的原 sequence/eventCount 流。Worker 归还流额度时还需确认相关 terminal/final 写入槽已经完成。

显式 request.cancel 只请求停止生成，保留用户的可读流；cancel 响应不是 terminal。重复 cancel 与完成 racing 时，保留最先到达的原执行终态。丢弃请求句柄用 release/自动排空，不能等价为仅 cancel 后把路由删除。

**共享物理 pipe 的限制：** 公平 sender 本身不能绕过一个已经阻塞的 WriteFile。每请求隔离成立的前提是宿主 reader 始终从 pipe 排空并且不等待任何下游客户端。v2 写入需支持 overlapped I/O 或等价的可取消专用 writer，不能持 registry/设备锁做同步写入。若整个 Proxy 进程停读、pipe 破裂或写入无法在连接级超时内进展，这是连接故障，影响同域所有请求；不能将其谎称为只取消了某个 HTTP 客户端。

首片只有小型 token 帧，不允许把 v1 的 512 MiB attachment 上限带进共享 event pipe。后续文本/音频事件必须重新定义字节 credit 或分块上限，不能只按 token 数继续记账。

## 8. 线程与锁的实施划分

| 执行单元 | 可以做 | 不可以做 |
| --- | --- | --- |
| Worker command reader | 有界读取/解析、epoch/ID 校验、登记命令、通知队列 | 同步生成整个请求、等待 GPU、等待输出消费 |
| Worker control dispatcher | registry 短临界区、submit/cancel/query/ack、关闭句柄、发起生命周期工作 | 持 registry 锁调用会析构 GPU 资源的代码 |
| Worker lifecycle executor/reaper | model load、Session 分配、最后设备引用析构、关闭时 join | 占用 command reader；长操作期间禁止接收 cancel |
| Scheduler thread | 当前已有的完整步骤调度 | 写 pipe、socket、调用用户输出回调 |
| Worker bridge + event sender | 每请求 Read(0)、冻结终态、轮转小帧、维护信用 | 绕过窗口、阻塞 GPU 线程、每次读到一条流就整条读完 |
| Worker response sender | accepted、控制响应及最终摘要的有界独立发送 | 让慢 event 流持有 response writer 锁 |
| v2 InferenceClient command sender | 分配 RPC ID、登记 pending、序列化写入、合并 ACK | 持有等待最终结果的整请求互斥锁 |
| v2 InferenceClient response/event readers | 按 epoch + RPC/inference ID 路由、验证排空、唤醒本地等待者 | 调用用户/HTTP 回调，或等待某条请求腾出无预留空间 |
| API caller / HTTP adapter | 等待并消费自己的 token/terminal、把下游断开映射为 release | 直接使用 Worker 的裸 Model/Session 指针 |

registry 查找后取强引用，释放 registry 锁再调用 Submit/Cancel/Snapshot 或安排析构。拿到短引用后若 release 并发发生，已登记的命令按其登记次序执行；新命令检查句柄 closing，不能恢复它。Session 的提交/变更门和 core 的租约共同保证 Busy；读到 Busy 时无需再等待 ContextLength。

首片的 Worker 控制/bridge 循环按最多 64 个请求做有界 sweep，加可唤醒短等待；Scheduler 不提供多流 wait API。response sender、event sender 的队列和保留槽在接受请求前预留，避免已经接受的请求在终态时因没有分配空间而丢失完成信息。日志也走有界异步队列，不复用 v1 `EventWriter::log` 内的同步 pipe 写入。

## 9. 请求错误、域故障、epoch 与关闭

| 情况 | 处理范围与表现 |
| --- | --- |
| InvalidArgument、Busy、ContextOverflow、UnsupportedModel、ResourceLimit | 对应命令或请求；Submit 拒绝 eventCount=0，accepted 之后以该请求的 Failed/Cancelled terminal 收尾 |
| 正常 cancel、core 输出背压 | 仅对应 Request；保留已发布事件、一次 terminal 和 final |
| release 后访问、跨 Model/Session 类型使用、StaleEpoch | 返回 InvalidHandle/StaleEpoch；不重启健康 Worker |
| 执行或 Submit 观察到 typed RuntimeFault | 标记整个设备域 faulted，停止 load/create/submit，Scheduler 使未终结请求 Failed，并发 domain.fault 通知 |
| 模型加载或 Session 初始化发生设备异常 | 保存同步 SYCL/oneDNN 首因，完成异常清理并以 RuntimeFault 上报；成功初始化还需通过最终 drain，生命周期执行器负责域广播 |
| framing 损坏、事件计数失配、Worker 退出、连接级读写失败 | 关闭连接，宿主一次性终结所有尚未终结的本地请求并失效远端句柄 |
| 单个请求的等待超时 | 上层请求 Cancel 或 release；不得直接沿用 v1 的“该请求超时便杀 Worker” |

core 的 `Forward/Reset/Truncate` 执行失败会标记 `DeviceState` 并抛出 typed `ErrorCode::RuntimeFault`；Scheduler 在 Step 或 Submit 的 `RequestState::Create` 中观察到这类错误后停止准入并终结其余未结束请求。`DeviceState` 标记与 Scheduler 故障状态仍是不同层的状态：生命周期执行器通过 `Scheduler::ReportRuntimeFault()` 显式转发加载、初始化和清理故障，不能把一次普通加载异常直接等同于域广播。

`ModelSessionBackend::Initialize` 和 `LoadedModel::LoadNf4` 现通过 `RunDeviceLifecycle` 先记录同步 SYCL/oneDNN 异常，再尝试 drain；即使后续 synchronize 成功，也以首因抛出 `RuntimeFault`。首因使用有界固定存储，后续 drain/析构故障不覆盖它。加载在最终 drain 通过后才返回 `LoadedModel`；Session 也只有最终 drain 通过后才发布 cache/state 并增加活动会话计数。普通参数、配置、文件、主机分配和准入失败保持局部语义；若清理的 drain 也失败，则按域故障收敛。

SafeTensors 的逐 tensor 恢复会原样传播 SYCL 上传异常，Qwen3.5 tied projection 回退会原样传播 SYCL/oneDNN 异常，避免把设备错误吞掉后继续加载。析构 drain 不抛异常，其故障由生命周期/清理线程通过 `DeviceRuntime::FaultMessage()` 查询并转发给 Scheduler；该查询取得设备执行锁，不在 transport reader 上执行。`ReportRuntimeFault()` 不等待 GPU 步骤或 join，首因可由 `SchedulerSnapshot::faultMessage` 读取；已发布 terminal 保持原状态，尚未发布的 token/terminal 与故障报告按同一调度锁确定先后顺序。分类与广播的 CPU 注入门禁已经通过，详情见 [共享会话故障语义](shared-sessions.md#设备域故障与生命周期) 和第 10 节。

域 fault 通知为 requestId=0 的事件，带 epoch、`errorCode: "RuntimeFault"` 和有界 message；不占请求 sequence。它通知句柄禁止再次执行，不是“可以立即删除全部输出路由”的指令。已完成请求的终态不因之后域 fault 被改写；尚未完成的流尽量收到原 Scheduler 的 RuntimeFault terminal。旧 epoch 的只读 drain/ack/release 在这个已故障连接内仍可用于清理，但任何新执行都失败。

首片故障恢复为关闭整个 Worker 并重新启动，得到新 epoch、重新加载模型和创建 Session。不在原进程重建一个 Scheduler 后继续接收旧句柄，不自动重试已输出部分 token 的请求。

如果 Worker 已无法发 terminal，宿主为仍存活的用户句柄发布一次**本地** transport/domain 失败终态，包含已经交付的 token 数。不得伪造一个看似来自 Worker、符合未知 eventCount 的 wire terminal。此前已经向用户交付终态的请求不再发布第二个终态；尚未消费的本地终态槽由状态机一次性选定正常完成或连接失败。

`worker.shutdown` 的步骤为：

1. 宿主停止接受新的用户 RPC，把此前已经排队的命令写完后发 shutdown；其后 sender 只推进内部取消/ACK/收尾。Worker 原子设置 stopping，拒绝新的 load/create/submit；command reader 仍处理已经登记的 cancel/query/ack/release。多个 Shutdown 调用在宿主共享同一次关闭 future。尚未开始的 lifecycle 创建任务返回 SchedulerStopped 并回滚预留；已经执行的任务需在 grace 内安全收尾，新建但未交付的资源进入 reaper。
2. lifecycle executor 请求 Scheduler.Shutdown，等待当前同步设备步骤安全结束。join 发生在 Scheduler 线程之外，不持 registry/pipe 锁；关闭不等待用户消费。bridge 继续强持有 ScheduledRequest，不能通过析构句柄跳过后续私有排空。
3. 宿主将尚未排空的用户流切换为关闭排空模式，仍保留一次最终状态的交付能力；内部 reader 丢弃待交付 token 并持续 ACK，避免关闭因已停止消费的调用方永久等待。对已关闭的用户句柄只做私有排空。关闭丢弃的数据在宿主结果中另记 `deliveryAbandoned=true` 和 `discardedPublishedTokens`；不能假称这些 token 已交付用户，或把先前 Completed 的执行状态重写成 SchedulerStopped。
4. bridge 排空已终结的 ScheduledRequest，response sender 发完各 accepted 请求的 final，event sender 发完其 terminal；等待宿主私有排空产生的终态 ACK，将这些 ACK 及先前登记的短 RPC 响应排入 response sender。这里等待的是一直工作的内部 reader，不能等待用户消费。最后排空 lifecycle/reaper，冻结日志并完成最后 eventOrdinal 的写入，此后不再产生新事件。
5. Worker 发送 shutdown result，包含 `lastEventOrdinal`；宿主等待 event reader 读到此 ordinal，再完成健康关闭。response 和 event 的到达乱序不能让 shutdown result 提前截断最后事件。
6. 关闭句柄并回收资源。grace 到期且当前 GPU 步骤/连接仍无法收尾时，宿主按**整个域**故障终止 Worker；所有受影响请求明确失败，不能报告正常优雅关闭。

关闭模式丢弃 token 是 v2 传输层已实现的 Shutdown 输出放弃语义，不适用于普通 Cancel；正常运行的读者不能用这种方式越过未消费输出。`Scheduler::Shutdown()` 本身保留可读输出，句柄在 Scheduler 销毁后仍可读取；ACK、私有排空和 deliveryAbandoned 由 v2 宿主与 Worker bridge 负责，不属于 core 的输出丢弃行为。关闭期间终态的 wire 校验仍完整进行，然后在本地保留小型最终状态及交付计数供用户查询。需要全部输出的调用方应先消费请求终态，再执行 v2 Shutdown。grace 只是请求的收尾期限，不承诺 SYCL kernel 可以在该期限内被安全抢占。

## 10. 首个可交付纵向切片

最小命令集为第 5 节的 11 项，覆盖能力协商、加载/释放父对象、创建独立 Session、提交/取消/查询、消费确认和关闭。首片输入输出均为 token，普通 C++ 宿主不加载 oneAPI DLL。代码已接通；以下保留首片验收要求，运行状态单独列在后文，不能将已编译等同于 GPU 验收通过。

1. v2 codec/schema、private host connection 和 CPU fake Worker 已实现；验证两个 submit 可同时处于 accepted，在 A 仍生成时 B 能 accepted，cancel/query/release 有独立结果路由。客户端受控 transport 单测已通过；用 gate 控制步骤，不依赖 sleep 猜测交错。
2. 接入现有 Model/Session/Scheduler，运行同模型 A/B 和 1/2/4 Session token 测试；对照直接进程内请求，保持各请求分块形状、seed 和 penalties 一致。
3. 使用 `maxNewTokens > tokenWindow + coreRing` 且不会提前命中 stop 的长请求 A，例如默认窗口/ring 下生成 256 token，并确保 Session 容量足够。停住 A 的 token 消费，确认 A 用完窗口和 core ring 后被局部取消；B 持续完成。再 release A，确认 bridge 强持有请求完成私有排空并归还所有对象/传输额度。另测短请求和 Finalizing 保持正常完成，不能用前文 128-token 示例要求必然触发背压。
4. 完成 v1 ABI、日志线程、owner、运行时隔离的兼容门禁，才形成独立可回归提交。真实 HTTP/SSE 在这之后接 v2 host API，不在首片加入文本生成兼容宣称。

### 不能复用的旧整请求语义

- `WorkerProcess::operation_mutex` 当前覆盖完整 request/request_stream 等待；新路径只允许短生命周期/写入锁，不能把这把锁换名后继续持有到生成终结。
- 当前 request_stream 为当前调用临时启动 response reader，并把其他非日志 request ID 当异常；v2 必须使用固定 reader 和 pending/request 表。
- WorkerMain 的流式分支只接受当前 requestId 的 cancel，其余控制帧会中断当前生成；v2 command loop 不能再嵌套在“当前流完成”的等待中。
- 当前 cancel 是同一 requestId 的 one-way 控制；v2 cancel 是独立 RPC，有目标 inferenceId 和响应，仍不替代生成 terminal。
- 当前全局 1024 帧事件队列、`EventWriter` 同步写 pipe 和“慢 callback 阻塞事件消费”不能承担多客户端的背压语义。
- `ProxyEngine::stream_active_` 仍是旧引擎外观的兼容门；新路径用每 Session Busy 和每 Request 生命周期，不能将它推广成整个共享模型的全局 Busy。

### 当前验证快照

以下为本轮主任务已确认的状态；完整日志、构建身份及后续状态转换由 [refactor-validation.md](refactor-validation.md) 统一记录。实现说明与使用入口见 [inference-ipc-v2-implementation.md](inference-ipc-v2-implementation.md)。

| 门禁 | 当前状态 |
| --- | --- |
| `AilaInferenceProtocolTests` | 已通过，CPU 11 组 |
| `AilaInferenceWorkerTests` | 已通过，CPU 6 个测试函数 |
| `AilaRuntimeFaultTests` / `AilaSchedulerTests` | 已通过，CPU 7 组 / 22 组 |
| `AilaInferencePipeTests` | 已通过，真实 Windows 子进程 CPU pipe 7 组，6.96 秒 |
| v1 的 IpcProtocol、WorkerDispatcher、RuntimeIsolation、ProxyAbi、ProxyLogging、BuildIdentity | 6 个门禁已通过，14.00 秒 |
| `AilaInferenceClientTests` | 已通过，受控 transport 16 组，0.30 秒 |
| `AilaQwen3IpcSessionTests` | 真实 Qwen3 GPU IPC 首跑通过，61.41 秒；1/2/4 会话及续写逐 token 精确相同，seed/penalties、取消/背压、父释放、优雅关闭及新 epoch reload 通过 |
| 全 Release 构建 | 已通过，`build-refactor/ipc-v2-full-build.log` |
| 既有 Qwen3/Qwen3.5 模型/调度 GPU 回归 | Q3 Session 23.21 秒、Q3 Scheduler 26.33 秒、Q35 0.8B 36.93 秒，全部通过，总计 86.48 秒；`build-refactor/ipc-v2-model-gpu-regressions.log` |
| 最终基础/兼容/IPC 回归 | 23 项全部通过，60.67 秒；`build-refactor/ipc-v2-final-regressions.log` |
| 普通宿主运行时导入 | 无 SYCL 导入，`build-refactor/ipc-v2-host-dependents.log` |

CPU pipe 测试使用同一 Worker 控制器和管道外壳，以 CPU 工厂替换真实模型。GPU IPC 首跑详情记录于 `build-refactor/ipc-v2-qwen3-gpu-details.log`；这些测试耗时不作为推理吞吐或 HTTP 服务性能结果。Hybrid 远端、HTTP/SSE 及新增公共 C ABI 未在此切片发布。

### 验收矩阵

| 测试入口 | 具体通过条件；运行结论以上述快照为准 |
| --- | --- |
| `AilaInferenceProtocolTests` | 两种版本独立编解码；长度 prefix 在分配前拒绝；整数/乘法溢出、UTF-8 和转义后字节上限；重复字段和错误附件；未知能力失败 |
| `AilaInferenceWorkerTests` | 两个 Session submit；同 Session Busy；父先释放仍完成；模型/KV 实际存活数限制；submit 准入失败完整回滚；完成未 ACK 继续占额度 |
| `AilaInferenceClientTests` | accepted 前先到事件；final/terminal 任意跨 pipe 顺序；A/B 交错；cancel/query 各自响应；重复/跳号/多 terminal/摘要不符必须失败 |
| `AilaInferenceWorkerTests` / `AilaInferencePipeTests` | A 停 ACK，B 在有界时间完成；整个进程停读引发连接故障；send 部分写/断管；response/event 都不占 GPU 锁；所有队列峰值在限额内 |
| Worker/Client/Pipe 与 v1 owner 门禁 | Model/Session/Request 乱序 release；正在 callback 的旧 owner 生命周期；release 后在途事件私有排空；旧 epoch ID 即便数值相同也不可执行 |
| Worker/Client/Pipe 关闭用例 | 活动/排队/完成未读流同时关闭；已阻塞 lifecycle 操作；双 Shutdown；用户停止读仍收尾；终态后关闭结果乱序；超时必须全域失败 |
| `AilaRuntimeFaultTests` / `AilaSchedulerTests` 及 Worker/Pipe 故障用例 | 执行和 Submit 的 typed RuntimeFault 停止域准入并终结 peers；Initialize/LoadNf4 原同步设备异常在后续 drain 成功时仍保留首因并广播；普通参数/准入错误不影响 B；故障后旧输出可排空、新执行失败；Worker crash 不自动重试 |
| `AilaQwen3IpcSessionTests` | 真实 A/B、1/2/4 会话 token 对照；固定 seed/penalties、续写、取消与长请求停读；重复创建销毁与 KV 准入；与进程内基准的 shape/seed 一致 |
| 既有回归目标 | IpcProtocol、WorkerDispatcher、RuntimeIsolation、ProxyAbi、ProxyLogging、BuildIdentity；旧 caller-thread 回调、ASR/TTS owner 和 generation 校验全部保留 |

CPU fake transport 使用受控管道与 step gate 验证事件顺序；长期创建/释放仍需检查 handles、host bytes、remote records 的稳定性。GPU 测试和性能测量由主代理串行运行，不能与现有 benchmark 同时占用 Arc。后续测量至少分开记录 submit ACK 延迟、TTFT、token 到达间隔、总吞吐、取消/排空时间及缓冲峰值，不能以 CPU 控制器或 pipe 门禁代替真实 GPU/HTTP 结果。

### 实现文件归属与后续依赖

| 负责人 | 当前文件或后续组件 | 依赖及边界 |
| --- | --- | --- |
| 集成负责人 | v2 schema/limits/错误类型、WorkerMain 模式选择、CMake、build identity | 维护消息契约、旧公共头兼容和最终集成证据 |
| IPC/宿主实现 | `src/ipc/InferenceProtocol.*`、`src/runtime/InferenceClient.*`、`src/runtime/InferencePipeTransport.cpp` | 独立 pending/reader/额度；旧 ProxyEngine 保持 v1 |
| Worker 实现 | `src/worker/InferenceWorker.*`、`src/worker/InferenceWorkerMain.*`、`src/worker/SharedInferenceEngine.cpp` | 控制器/外壳与真实 Model/Session/Scheduler 适配；CApiWorkerEngine 保持 v1 |
| 验收实现 | `tests/runtime/FakeInferenceWorkerMain.cpp`、上述 CPU 测试与 `tests/inference/Qwen3IpcSessionTests.cpp` | CPU 工厂复用 Worker 外壳；本轮 CPU/GPU IPC 门禁及最终全量集成回归均已通过 |
| 后续服务适配 | `AilaServer`、tokenization/chat driver、HTTP/SSE bridge | 依赖 v2 drain/backpressure 门禁；不把 HTTP 探针当作端到端推理验收 |

表中的 v2 codec、宿主、Worker 与测试入口已存在；HTTP 服务适配仍为后续工作。各目标的链接边界和宿主 API 示例见 [IPC v2 实现记录](inference-ipc-v2-implementation.md)。实现完成、编译完成与验收通过分别记录；本轮 IPC 和集成回归通过不替代后续服务层验收。

本切片已采用**独立 v2 启动模式**与**仓库内部 C++ host API**。新增公共 C ABI 留待 token/句柄语义验收后另行设计 size/version 约定，现有 `aila_api.h` 布局保持。上文准入值作为首片默认实施，后续性能测量再通过协商配置调整。
