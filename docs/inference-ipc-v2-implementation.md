# IPC v2：共享推理的独立 Worker

日期：2026-09-07。当前为首个 token 推理切片的实现记录；最终验证结果与构建身份见 [refactor-validation.md](refactor-validation.md)。协议字段和完成判据见 [inference-ipc-v2.md](inference-ipc-v2.md)。

## 接入范围

新增 `--ffi-worker-v2` 启动模式与 `src/runtime/InferenceClient.hpp` 内部 C++ API。一个 Worker 使用一个设备域、一个 Scheduler 和最多一个 Qwen3 NF4 模型，共享模型权重，最多创建四个独立 BF16 KV Session。输入输出均为 token ID；已经迁移的 Qwen3.5 进程内会话暂不作为远端能力发布。

旧 `--ffi-worker`、`WorkerProcess`、Proxy 与公共 C ABI 继续使用 protocol 1。v2 的 protocol 为 2，ABI 仍为 1。此切片没有迁移聊天模板、reasoning、工具解析、thinking budget、音视频或运行中 adapter 替换，也没有提供 HTTP 端点。独立 Drogon 探针的结果见 [refactor-http-probe.md](refactor-http-probe.md)。

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

`Request::Read(timeout)` 仅让本次等待超时。`Cancel()` 产生独立 RPC，保留可读输出；取消响应不等于请求终态。`Release()` 非阻塞，标记取消并由私有路由丢弃、累计确认剩余输出。请求执行器在完整 prefill/decode 步骤边界处理取消。

## 生命周期与完成

Model、Session、Request 宿主句柄均绑定 connection owner、128-bit 随机 epoch、对象类型与 ID。销毁 Client 外壳不提前关闭仍被派生句柄使用的 Worker。每次 Worker 重启生成新 epoch，不能把旧对象 ID 绑定到新进程。

释放父句柄只关闭该客户端引用。Worker 将最后设备引用交给生命周期执行器销毁；资源的实际存活数量包含已经关闭父句柄但仍被子对象使用的资源，也包含尚未完成的创建/回收任务。每条传输流继续保存原 Model/Session ID。

每个成功 submit 在 response pipe 上先有 accepted，再有 final；事件在独立 event pipe 上传输。宿主在写 submit 前登记路由，所以允许事件早于 accepted 到达。只有 accepted、final、terminal、连续 sequence、eventCount 和完整摘要全部一致，才能向用户交付 terminal。

Scheduler 的执行完成、bridge 读取 terminal、pipe 写完、宿主读取和用户消费分别记账。bridge 必须强持有 `ScheduledRequest`，不能通过析构它实现 release：析构会 detach 并丢弃 core ring。IPC 额度保留到 terminal ACK，且 Worker 的相关输出写入槽也已完成；消费终态后尚未 release 的小型快照继续占 retained-handle 额度。

## 有界输出与慢消费者

默认 tokenWindow 为 32，core token ring 为 128，最多 64 条未排空请求，最多 256 个保留请求句柄。默认每请求 prompt/generated/stop ID 上限为 4096/1024/64；总 prompt + maxNewTokens 预留上限为 32768。

每端 ingress/transport 预算默认各 8 MiB，日志队列 64 KiB。Worker 为控制响应、日志与保留句柄预留预算，再接受新的流；每流最多有一个 staged/in-flight token 帧。日志回调只做有界入队，不写 pipe。超出日志预算时丢弃并计数。configure 不能把日志上限降到已经排队或正在发送的数据以下。

窗口按用户实际消费或明确放弃的连续前缀累计 ACK。宿主 reader 持续排空 pipe，但不会因为刚收到 token 就确认消费。A 停止读取时最多占自己的窗口和 core ring；继续产 token 导致 ring 满会取消 A，B 仍可运行。短请求或只剩 Finalizing 的请求可正常完成，因此背压验证使用不设置 stop IDs 的 256-token 长请求。

这些是有界 payload 预留，不是总进程 RSS 或总显存测量。模型权重、共享执行 scratch、oneDNN/驱动分配、allocator 元数据和内核管道缓冲区不属于 Session 缓存额度。物理 pipe 的整个宿主停止读取属于连接故障；独立客户端的隔离依赖宿主专用 reader 持续工作。

## 故障与关闭

加载/初始化会保留首次同步 SYCL/oneDNN 异常，即使后续 drain 成功，也会归一为 RuntimeFault。加载器的 best-effort tensor 恢复不会吞掉这些设备异常。普通文件、配置或 host allocation 失败仍保持局部语义；若随后的设备 drain 也失败，则升级为域故障。

生命周期执行器捕获 RuntimeFault 后调用 `Scheduler::ReportRuntimeFault()`，停止准入并广播；析构 drain 的故障通过 `DeviceRuntime::FaultMessage()` 继续传播。已发布终态不改写，未结束请求在安全边界失败。domain.fault 事件不删除既有输出路由，ACK/release 仍可收尾。损坏 framing、计数失配、管道故障或 Worker 退出会失效整个连接，宿主为尚未交付终态的请求选择一次本地失败终态，不伪造未知的 wire eventCount。

`Shutdown()` 由并发调用共享同一次关闭过程。宿主停止新调用并放弃未消费 token，继续私有排空与 ACK；本地终态记录 `deliveryAbandoned`、实际交付数和丢弃数。Worker 等待 Scheduler、生命周期回收、全部 terminal ACK、短命令响应和事件写屏障，才返回含 lastEventOrdinal 的关闭结果。宿主再验证自己读到该 ordinal，并检查子进程退出。超过 grace 或子进程退出失败均报告连接失败，不报告优雅关闭成功。

默认连接写超时为 10 秒，最大 shutdown grace 为 30 秒。它们是连接级保护，不能保证抢占正在执行的 GPU kernel；单个 Request 等待超时不会杀掉共享 Worker。

## 验证入口

- `AilaInferenceProtocolTests`：v1/v2 独立、prefix/UTF-8/JSON 整数/重复字段/附件/能力/摘要边界。
- `AilaRuntimeFaultTests`、`AilaSchedulerTests`：设备异常分类、首次原因、生命周期广播及既有调度行为。
- `AilaInferenceWorkerTests`：CPU Session 工厂下的准入/回滚、Busy、父子生命周期、背压、释放、故障、日志降限与关闭。
- `AilaInferenceClientTests`：受控双通道传输下的跨管道乱序、消费/释放/关闭状态机。
- `AilaInferencePipeTests`：实际 Windows 子进程和 pipe 的 CPU 集成，以及普通宿主的运行时隔离。
- `AilaQwen3IpcSessionTests`：真实 Qwen3 直接调用对照、1/2/4 会话、固定 seed/penalties、续写、取消、长请求停读、重启 epoch。

GPU 测试与 benchmark 必须串行执行。既有 IpcProtocol、WorkerDispatcher、RuntimeIsolation、ProxyAbi、ProxyLogging 与 BuildIdentity 保留为 v1 兼容门禁。服务层接入应在本切片门禁通过后进行；文本/聊天与 HTTP/SSE 的实现和验收独立记录。
