# Qwen3 共享模型会话：首个实现切片

当前实现位于 `src/inference`，面向进程内的 Qwen3 NF4 文本推理。旧 `InferenceEngine`、CLI 和 C ABI 保留原有入口；Qwen3 后端的旧入口使用一个默认会话。新的 Token API 供后续 Worker 调度器接入，尚不是 HTTP 服务或新的公共 C ABI。

## 资源与执行约定

| 对象 | 当前实现 |
| --- | --- |
| `DeviceRuntime` | 活动句柄共享默认 SYCL 设备的 Context 和执行锁；覆盖所有设备资源的生命周期 |
| `LoadedModel` | 一次加载的原始/融合权重、模型配置和一个执行槽；创建会话不重新加载权重 |
| `SessionState` | 独立 KV、实际缓存长度和操作租约；持有模型及运行时资源 |
| `RequestState` | 输入进度、生成 token、采样配置、独立 RNG、取消和终态 |
| 执行槽 | 共享输入/采样缓冲，以及暂时仍位于 Qwen3 backend 内的激活、scratch 和可变 oneDNN 绑定 |

调用者在完整 `Step()` 之间切换请求。每一步执行一个 prefill 分块或一个 decode token，返回前完成该步设备工作。首版保留 in-order queue 和已有 NF4/attention/跨层 residual-norm 融合，不在每层之间同步。多个宿主线程的设备提交由同一执行锁串行化；专用调度线程、公平队列和传输背压属于后续阶段。

`DeviceRuntime::Create()` 复用当前活动的默认设备域。这个版本不支持多设备执行，也不承诺与独立构造的旧 `InferenceEngine` 在同一进程内并发执行；部分旧算子仍有进程级缓存。

`ModelOptions::maxSessions` 限制会话数量，默认 4。非零 `maxSessionCacheBytes` 在分配前按 BF16 KV 大小预留，会话销毁后归还额度。该额度只限制会话缓存；模型、执行 scratch 和 oneDNN 自有分配另计。`DeviceRuntime::Memory()` 返回 Context 记账的 USM，不能当作全部显存占用。

首批共享会话要求 BF16 KV（`AILA_KV_QUANT=0`，也是默认值），权重仍为 NF4。FP8 KV 的既有 decode 读写路径未通过本次正确性检查；新接口在加载及显式创建会话时返回 `InvalidArgument`，不会默默分配有问题的缓存。具体证据见 [refactor-validation.md](refactor-validation.md)。

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

`RequestState::Create()` 将输入追加到已有缓存，要求非空输入，并预留 `已有缓存 + 输入 + max_new_tokens` 的容量。它不推断聊天前缀、不复用其他会话历史。每个 Session 同时只能有一个请求；重复提交及活动期间的 `Reset/Truncate/ForwardLogits` 返回 `ErrorCode::Busy`。`Step()` 也拒绝并发或递归调用。

`stopTokenIds` 中的停止 token 不发布、不写入缓存。到达输出数量上限后，状态先进入 `Finalizing`；再调用一次 `Step()` 写入最后已发布 token，随后进入 `Completed`。因此正常完成的缓存覆盖全部已发布输出。

`Cancel()` 可以从其他线程调用，取消在提交边界或当前同步步骤结束时生效。取消/失败后保留实际已消费的缓存前缀，可能不包含最后一个已发布 token；上层应查询 `ContextLength()`，按自己的历史协议补算或 Reset。销毁 Request 释放会话租约，销毁 Session 在执行锁内排空设备工作后释放 KV。Session 和 Request 可在外部模型/运行时句柄释放后继续存活。

不可恢复的执行错误使当前设备域失效；该域其他会话的后续执行也会失败。当前没有调度器向所有等待请求主动广播终态，也不会自动重试部分输出。进程级日志通过有界 `ScopedLogDeferral` 在设备、会话和请求锁释放后投递，允许日志回调读取状态。`Step()` 在所有执行守卫释放前保留请求强引用，即使日志回调释放调用方最后一个句柄，也不会提前析构；回调中的递归 `Step()` 仍返回 Busy。

## 当前边界与验证

当前 Token API 支持 greedy、固定 seed 采样、已有 repetition/presence/frequency penalties，以及显式停止 token。它继承现有采样算子的行为；不改变旧生成接口的采样规则。`thinking_budget_tokens` 必须为 -1；聊天模板、thinking budget、工具调用和文本流解析尚未迁入这个低层驱动，旧接口继续支持原有能力。

`LoadedModel::LoadQwen3Nf4` 只接受 Qwen3 NF4；Qwen3.5、BF16、视觉、ASR/TTS、Diffusion 等继续使用旧入口。共享模型的运行中 LoRA/权重替换被拒绝。统一 LayerPlan、Hybrid 状态迁移、完整调度器、多请求 IPC 和 C++ HTTP/SSE 服务依照路线图继续推进。

构建后运行 CPU 状态/日志测试，以及真实模型会话测试：

```powershell
ctest --test-dir build-refactor -R 'Aila(RequestState|DeferredLogging)Tests' --output-on-failure
$env:AILA_QWEN3_TEST_MODEL = 'E:\RiderProjects\Aila\models\qwen3-0.6B-bnb-nf4-offline'
$env:AILA_KV_QUANT = '0'
ctest --test-dir build-refactor -R '^AilaQwen3SessionTests$' --output-on-failure
```

未提供模型时 GPU 用例以返回码 77 标记跳过。该用例用同一模型的两组不同输入、相同的各自分块形状，对照独立执行与交错执行的每步 logits 和生成 token，并覆盖取消、回滚、资源回收、Busy 和所有者生命周期。跨源码版本的 CLI 回归应另用 `perf/compare-token-traces.ps1` 比较完整 trace；测试不能用不完整日志宣称 token 一致。
