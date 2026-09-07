# Qwen3 token 请求调度器

`src/inference/Scheduler.hpp` 提供进程内的自动多会话调度。每个 Scheduler 有一个提交线程，使用已有 `SessionState / RequestState` 执行 Qwen3 NF4。该接口链接 `AilaLib`，尚未接入 Worker/Proxy 多请求协议或 HTTP 服务。

## 执行与公平性

一个调度 quantum 是完整的 prefill 分块、一个 decode token，或最后一次 `Finalizing`。每步返回前完成设备工作，随后轮转到下一个活动请求。长 prompt 的块大小被 `maxPrefillChunkSize` 限制；请求可以指定更小的块。单 token prompt 继续使用后端原有的 decode 快路径。

最多 `maxActiveRequests` 个请求参与轮转，其余按提交顺序等待。新获得活动名额的请求排在刚完成本轮的请求之前。取消的排队请求可以在步骤边界直接终结，无需等待活动名额；这类维护与正常执行交替，避免取消流量饿死 decode。

同一设备域使用一个 Scheduler。多个调用线程可以 Submit、读取自己的流、Cancel 和查询快照，输出消费不在提交线程上执行。Qwen3 的会话长度是由会话操作锁保护的私有主机元数据；提交空闲 Session 不需要等待另一个 Session 占用的设备执行锁。加载模型及预创建 Session 仍由调用者完成，设备互斥锁保护共享执行槽；此版本没有跨请求 GEMM 合批、分页 KV 或多 GPU 调度。

## 使用示例

输入必须已经按模型模板编码为 token；Scheduler 不执行聊天格式化、thinking budget 或文本/工具流解析。

```cpp
#include "inference/Model.hpp"
#include "inference/Scheduler.hpp"

using namespace aila::inference;
using namespace std::chrono_literals;

auto runtime = DeviceRuntime::Create();
ModelOptions modelOptions;
modelOptions.maxSessions = 4;
auto model = LoadedModel::LoadQwen3Nf4(runtime, modelDirectory, modelOptions);

SchedulerOptions limits;
limits.maxActiveRequests = 4;
limits.maxPrefillChunkSize = 128;
Scheduler scheduler(limits);

RequestOptions options;
options.generation.max_new_tokens = 64;
options.generation.use_fixed_seed = true;
options.generation.sampling_seed = 42;
options.stopTokenIds = {eosTokenId};

auto session = model->CreateSession();
auto request = scheduler.Submit(session, promptTokens, options);
for (;;) {
    auto next = request->Read(100ms);
    if (next.status == ReadStatus::Timeout) continue;
    if (next.status == ReadStatus::Closed) break;
    const auto& event = *next.event;
    if (event.kind == RequestEventKind::Token) {
        ConsumeToken(event.tokenId); // 在读取方线程处理/传输。
    } else {
        ConsumeTerminal(event.status, event.endReason, event.error);
        break;
    }
}
scheduler.Shutdown();
```

每个流只需一个消费者。不同 Session 可先分别 Submit，再由各自的线程或上层事件循环读取。`ScheduledRequest` 不暴露底层 `Step()`，因此调用者不能绕过调度器并发推进同一个请求。

## 准入与资源界限

| 设置 | 默认 | 约束 |
| --- | ---: | --- |
| `maxActiveRequests` | 4 | 参与轮转的请求数 |
| `maxOutstandingRequests` | 64 | 排队、活动和已结束但尚未消费终态的请求总数 |
| `maxPromptTokens` | 4096 | 每次新增输入，不含 Session 已有缓存 |
| `maxGeneratedTokens` | 1024 | 每个请求的 `max_new_tokens` 上限 |
| `maxStopTokenIds` | 64 | 每个请求的停止 token 列表长度，重复 ID 也计数 |
| `maxOutstandingTokens` | 32768 | 尚未归还额度的请求之 `prompt + max_new_tokens` 总和 |
| `maxPrefillChunkSize` | 512 | 每步最多消费的 prompt token 数 |
| `maxBufferedTokensPerRequest` | 128 | 每个流固定 token ring 的容量 |
| `maxSessionCacheBytes` | 0 | 可选的已准入 Session KV 额度；0 表示仅由模型的缓存限制管理 |

准入先预留额度，再分配输出 ring、复制紧凑输入并获取 Session 租约。无效输入、Busy、超额或与 Shutdown 的竞争都会回滚额度。请求不会保留调用方 vector 的多余容量。生成历史在开始前按上限预留；错误信息最多 1024 字节，截断保留 UTF-8 边界。

正常消费 terminal 后，或丢弃最后一个句柄且执行安全结束后，额度才归还。完成但无人读取的输出不能无限积累。归还额度前实际销毁 token ring；底层 Request 在发布终态前释放模型/Session 的所有权。终态被消费后的句柄仍可查询其小型快照；这些由调用方保留的句柄不再占用调度额度。

`maxSessionCacheBytes` 按请求所持 Session 的 `CacheBytes()` 保守计数，不合并同一个已完成 Session 的重复引用。Session 在 Submit 前已经创建，分配前的限制仍由 `ModelOptions::maxSessions / maxSessionCacheBytes` 负责。**这些设置不代表总显存准入**：权重、共享 scratch、oneDNN/直接 USM 分配及外部持有的 Session 都需要额外预算。HTTP/IPC 还需对输入原文、编码输出、socket 缓冲和连接数单独设限。

## 输出、取消和终态

事件使用每个 Scheduler 内递增的 request ID，每个流的 sequence 从 1 连续递增。token 事件之后恰好一个 terminal；消费 terminal 后 Read 返回 Closed，尚未有输出时可返回 Timeout。流快照报告最近完成步骤的状态、处理的 prompt token、已经发布的 token 数和 ring 占用，不等待当前 GPU 步骤。

ring 满时，在下一次会产生 token 的步骤之前取消该请求，终态原因为 `OutputBackpressure`，错误码 `ResourceLimit`。已缓冲 token 仍按序交付，terminal 使用独立的预留位置。`Finalizing` 只写入最后一个已发布 token，因此即使 ring 满也允许它正常完成。慢消费者不会让提交线程等待网络或输出消费。

`Cancel()` 非阻塞，在当前步骤的安全边界生效。丢弃最后一个 `ScheduledRequest` 句柄也触发取消，并丢弃无人消费的输出。其他请求继续运行。完成或取消 racing 时，以请求状态机已经到达的终态为准；不自动重试已经产生的输出。

| 原因 | 状态 | 说明 |
| --- | --- | --- |
| `Length` | Completed | 达到输出上限，已完成最后 token 的缓存写入 |
| `StopToken` | Completed | 停止 token 不发布，也不写入缓存 |
| `Cancelled` | Cancelled | 显式取消或丢弃句柄 |
| `OutputBackpressure` | Cancelled | 输出 ring 满，保留已有输出 |
| `SchedulerStopped` | Cancelled | 调度器关闭 |
| `RuntimeFault` | Failed | 设备域故障，整个调度器停止准入并终结其余请求 |
| `Failed` | Failed | 其他请求异常，只影响该请求 |

取消/失败后的缓存只保证保留实际消费的前缀，可能尚未包含最后一个已发布 token；复用长期 Session 前，应查询 `ContextLength()`，补算或 Reset。设备域故障后必须销毁相应旧域资源并重新加载，不能复用失效 Session。

## 关闭与回调

`Shutdown()` 停止准入，取消排队/活动请求，等待当前设备步骤及提交线程收尾；它不等待消费者读取输出。后续 Submit 抛 `SchedulerStopped`，设备故障后的 Submit 抛 `RuntimeFault`。重复或多个线程调用 Shutdown 可安全收敛。句柄在 Scheduler 销毁后仍可读取已有输出和终态。

日志仍通过 `ScopedLogDeferral` 在请求/设备锁外投递，因此日志回调可能运行在提交线程。回调可以读取快照、取消、提交其他 Session 或请求关闭；不要在回调中等待依赖该提交线程的流输出。若回调在提交线程调用 Shutdown 或销毁 Scheduler，关闭请求立即生效，实际收尾在回调返回后完成，避免线程等待自身。

此关闭语义的时间上界依赖当前同步 GPU 步骤能够返回。调度器不尝试强杀 SYCL 工作；超时回收失联设备进程属于后续 Worker 隔离层。

## 验证入口

CPU 的 `AilaSchedulerTests` 使用可控执行 gate，验证轮转、固定 seed、准入、排队/运行取消、背压、故障广播、输出寿命与回调关闭。它不创建 SYCL Context。

```powershell
ctest --test-dir build-refactor -R '^AilaSchedulerTests$' --output-on-failure
```

真实模型集成和性能结果统一记录在 [refactor-validation.md](refactor-validation.md)。GPU 测试与 benchmark 必须串行运行。

```powershell
$env:AILA_QWEN3_TEST_MODEL = 'E:\RiderProjects\Aila\models\qwen3-0.6B-bnb-nf4-offline'
$env:AILA_KV_QUANT = '0'
ctest --test-dir build-refactor -C Release -R '^AilaQwen3(Session|Scheduler)Tests$' --output-on-failure

pwsh -NoProfile -File perf/shared-session-bench.ps1 `
    -ModelDir models/qwen3-0.6B-bnb-nf4-offline -BuildDir build-refactor
```

性能入口调用 `AilaQwen3SchedulerTests --benchmark <json>`，跳过正确性用例，单独测量 1/2/4 会话，各预热 1 轮、测量 3 轮。固定 prompt 256、输出 128、prefill chunk 256、上下文 512、greedy；每请求独立读取线程，报告 Submit 到首 token 的 TTFT、实际 token 到达间隔和包含 prefill/finalization 的总吞吐。这些指标不包含 HTTP、tokenization、模型加载或 Session 创建。

脚本导入并校验 oneAPI 栈，持有全局 benchmark mutex，默认关闭 profiling，固定 BF16 KV，设置 600 秒超时。每次创建独立输出目录，保存原始日志、逐 token 时间与 ID、模型/二进制/源码哈希、tracked diff、环境和驱动信息；默认入口指针是 `tmp/refactor/q3-scheduler-bench/latest-run.json`。CTest 不使用该 mutex，仍需调用者串行安排 GPU 工作。
