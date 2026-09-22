# 异步 token C ABI

`include/aila_inference.h` 提供独立的 `AilaInference.dll` 接口。它通过已有 IPC v2 Client 启动隔离的 Worker，支持 Qwen3 NF4、Qwen3.5 Hybrid NF4/BF16 的 Model、Session 和异步 Request。旧 `aila_api.h` / `AilaShared.dll` 保持原有入口，并可显式启用 [共享文本适配层](legacy-text-adapter.md)。

新 DLL 使用普通 C++ 编译，不链接 AilaLib、SYCL 或 oneDNN。宿主只依赖 Windows 与 C++ runtime。公开头文件可由 C99/C++17 使用；Windows x64 的布局和未修饰 C 符号已由独立 C 翻译单元与 Python ctypes 验证。

2026-09-12 本轮新增 `client_start_with_batching`，最终构建、CPU 与 MSVC SDK 检查已通过。四种模型/权重配置的冻结 B1 对照中，每种配置的参考侧和候选侧各运行 12 个请求、生成 96 个 token，结果完全一致；新合批的完整 logits、Request 重放和前向后额度拒绝恢复也已通过，具体范围见 [连续 decode 合批](continuous-batching.md)。四模型真实 HTTP 合计 28 项功能用例通过，包括上限 1/4 的单例结果一致、并发、断连恢复及活动请求关闭；尚无集中压力、吞吐或长期稳定性结论。

## 生命周期与线程约定

1. `client_start` 或显式的 `client_start_with_batching` 验证 UTF-8 运行目录和 Worker 身份，完成协议协商。Worker 必须与 DLL 的 `aila_inference_build_id()` 匹配。
2. `model_load` 选择权重后端；`session_create` 创建独立缓存状态。
3. `request_submit` 复制 token 和选项，在收到 accepted 后返回。生成继续在 Worker 内运行。
4. `request_read` 在调用线程交付一个 token 或终态。超时返回 `TIMEOUT`；消费终态后返回 `END_OF_STREAM`。超时不会取消请求。
5. `request_cancel` 请求取消，继续读可以排空已经发布的 token 和终态。`request_release` 立即使该句柄失效，并由私有连接执行取消、ACK 和丢弃；它不等待阻塞中的 reader。
6. `client_shutdown` 停止整个连接并等待关闭屏障、Worker 和 I/O 线程真正退出，包括线程捕获对象/CRT trampoline 的结束。它与仅释放父句柄不同；动态卸载 DLL 前应先完成 shutdown 并释放所有句柄。

不同 Request 可由不同线程读取。同一 Request 同时只能有一个 reader，重叠调用返回 `BUSY`。同一 Session 不能同时执行两个请求；另一个提交返回 Busy，不修改其缓存。取消、查询和释放可以与读取并行。

句柄是进程/DLL 实例内的 64 位整数，不是 C++ 指针。注册表限制为 4096 个已发布或正在创建的句柄；类型和代际校验拒绝旧句柄及类型错用。复用槽位推进代际，代际耗尽后停用该槽位。连接原有的请求、Session、token 和队列上限继续独立生效。

释放 Client 或 Model 不影响已经创建的子对象。原连接故障后的子句柄仍属于原 Worker epoch，不会自动绑定新 Worker。显式 shutdown 则会停止同一连接上的全部请求。若宿主要卸载 DLL，应保留 Client 句柄至 shutdown 完成，再结束所有 API 调用并释放全部句柄；仅释放最后一个 owner 发起的是异步关闭。

## 结构、错误和事件

所有顶层结构包含 `struct_size` 和 `abi_version`。当前版本是 `AILA_INFERENCE_ABI_VERSION == 1`，独立于 IPC 协议版本。三个 options 类型提供默认初始化函数；输入 reserved 字段必须为 0。版本 1 接受更大的结构，但只读写已知前缀，保留调用者的未知尾部；过小结构和未知版本会被拒绝。

输出结构及可选 Error 的头部在消费事件之前校验。调用者始终持有输入/输出内存，字符串和 token 数组只借用到函数返回，不要求跨 DLL 释放 C++ 对象。Error 为每次调用独立传入，不使用进程级或线程级 last-error 缓冲。

函数返回值与远端错误号是两套稳定编号，见头文件。正常 token/终态读取返回 `OK`；请求的生成失败记录在终态的 progress 中。`local_failure == 1` 表示宿主连接失败产生的本地通知，sequence 为 0，不伪造 wire terminal。生成完成摘要保留 prompt/generated token、事件数、终结原因和错误信息。

`client_snapshot` 查询宿主连接的状态和额度计量。`model_info` 返回模型、epoch、显式 USM/Session 预算；`session_query` 和 `request_query` 通过 RPC 查询远端。Session 活跃时 context length 可能不可用，调用者必须检查 `has_context_length`。

`client_query_memory(handle, timeout_ms, out_info, error)` 通过可选 `runtime_memory_v1` 读取运行时USM账本。独立的 `AilaInferenceRuntimeMemoryInfo` 包含状态、epoch、device/pinned当前字节与峰值、存活分配数量及上限；仅 `MEMORY_READY` 时数值有效，busy/未初始化仍返回OK。它需要新版DLL导出，Worker能力只在调用查询时要求。字段及Server采样边界见 [运行时分配快照](runtime-memory.md)。

`session_reset` 和 `session_truncate` 需要 Worker 提供可选 `session_context_v1`。它们阻塞至变更完成或本地超时，返回同一个 `AilaInferenceSessionInfo` 结构；reset 成功时长度为 0，truncate 成功时返回实际 checkpoint。Hybrid 关闭回滚快照时，实际长度可能为 0，调用者必须补算缺失的前缀。活动或已有变更排队的 Session 返回 `BUSY`，查询返回 busy 且没有精确长度。

输出结构和 Error 的头部在产生远端副作用前校验。RPC 超时不会撤销已提交的变更，应查询到 idle 后再决定下一步；释放父句柄也不撤销已在途的调用。现有结构布局、ABI version 1 和旧入口保持兼容；最终产物导出检查确认异步 DLL 原有 27 项完整保留，仅新增 `aila_inference_client_start_with_batching`，总计 28 项；旧 `AilaShared.dll` 的 45 项导出也完整保留。

`request_prefill(session, ids, count, chunk_size, timeout, out_request, error)` 需要可选 `prefill_only_v1`。它同样返回异步 Request，正常完成只交付一次 `Completed/Length` 终态，generated tokens 为 0，prompt processed 等于输入长度。它不采样、不输出 token；取消时保留已执行的输入分块，后续应查询实际 context length 后再补算。普通 `request_submit` 仍要求正数 max_new_tokens。

## 选项

Client 可降低活动/在途请求数、Session 数、prompt/输出 token、保留句柄、token window 和 buffer 上限，并设置握手、命令及关闭超时。默认保持原限制；唯一可提高的字段为 max_generated_tokens，显式设置为 1025–32768 时协商 `extended_generation_v1`。`AILA_INFERENCE_EXTENDED_MAX_GENERATED_TOKENS` 给出绝对上限，104 字节 ClientOptions 布局不变；旧 Worker 未支持该能力时初始化局部失败并关闭进程。prompt + max_new_tokens 仍占原总 token 预留，context、输出 ring/window、Session 和内存上限不会随之提高。RPC 调用的 timeout 0 使用已配置值；`request_read(..., 0, ...)` 则立即轮询。

`aila_inference_client_start_with_batching(runtime_directory, worker_executable, options, max_decode_batch_size, out_handle, error)` 使用额外的整数参数配置 decode 合批，不占用旧 ClientOptions 的 reserved 字段。上限必须为 1..4 且不超过 `options.max_active_requests`；NULL options 使用原默认值。传入 1 与原 `client_start` 相同，省略新的 configure 字段且不要求新能力；大于 1 协商 `continuous_batching_v1` 并核对精确回显。非法参数在启动 Worker 前拒绝，缺能力明确返回 unsupported，配置在 Worker 生命周期内固定。新入口复用原错误输出与失败清零 out_handle 规则，不改变其他对象、事件或额度结构。

合批只聚合兼容的独立 decode/finalizing 行；prefill 仍按请求分块轮转。同 LoadedModel 对象在加载时绑定的 adapter 组合固定，不能混入另一模型或 adapter。采样、thinking budget、取消、输出环和累计 ACK 仍归各请求；不支持合批的后端、Qwen3 的 `AILA_PROFILE_Q3_DECODE=1` 或 Hybrid 旧诊断模式回退为原单请求执行。`AILA_FUSE_RESIDUAL_ADD=1` 时，三个后端也均回退到完整 B=1 路径，保留原残差融合语义。原 `client_start` 以及旧文本适配器不会因为新 DLL 提供该导出而自动启用合批。

模型支持可选的 `allocation_limits_enabled` 及 device、pinned host、Session host 额度。显式 USM 和 mirror/checkpoint payload 的计量不等同于驱动分配或进程 RSS。限额为 0 表示没有该项字节上限。

Request 接收独立采样参数、seed、stop token 和 prefill chunk。`thinking_budget_tokens == -1` 关闭预算；非负值需要 tokenizer 提供不同的 open/close marker、以 close marker 开头的 1–16 token 关闭序列，以及 prompt 是否已经进入 thinking 的标记。强制关闭 token 的 RNG、usage、KV 与短容量约定复用 [Request thinking budget](thinking-budget.md)。token API 不自行解析聊天文本或推断 tokenizer 标记。

`request_submit_with_policy` 额外接受可空的 `AilaInferenceRequestPolicy*`。该新结构为 32 字节，填写 struct_size/ABI version，reserved 置零；旧 RequestOptions 仍为 152 字节。全零策略与原 submit 相同。`THINKING_CLOSE_ONCE` 通过 `thinking_budget_close_once_v1` 保留旧文本 API 的一次强制关闭语义；`SAMPLING_HOST` 通过 `host_sampling_v1` 明确选择 host sampler；`prefill_logits_policy=PREFILL_LOGITS_LEGACY` 通过 `legacy_prefill_logits_v1` 保留旧 Hybrid 短续写多行输出的首行采样行为。该字段占用原 offset 16 的 reserved 槽，结构大小不变。三者独立，Worker 不提供所需可选能力时在发送 RPC 前拒绝；默认仍使用最后有效行，旧 peer 的默认请求形状不变。

## Worker 日志

`client_read_log` 从独立的有界队列取出一个 `AilaInferenceLogEvent`，包含 Worker epoch、连接级 event ordinal、0–3 的日志级别、完整 UTF-8 消息及字节数。ordinal 与 token/terminal/fault 共用，所以日志之间可以有间隔。日志没有 Request 身份，不应归属到恰好正在读取的请求。

一个 Client 同时只能有一个日志 reader，重叠读取返回 `BUSY`，超时 0 表示非阻塞轮询。输出与 error 结构在消费前检查。默认队列上限为 64 KiB，满时丢弃新日志并累计 host dropped_logs；Worker 自身的丢弃由 Worker query 独立计数。慢日志消费者不消耗 token credit，也不阻塞 IPC reader 或 GPU 提交线程。

正常关闭或连接故障后，已经排队的日志仍可读，排空后返回 `END_OF_STREAM`；仍保持连接的设备域故障不会单独结束日志流。`client_release` 关闭日志消费、丢弃剩余日志并唤醒阻塞读取，同时保留已有 Model/Session/Request 的存活语义。任何调用方回调都由调用方线程自行执行。

`client_set_log_level(client, level, timeout, error)` 要求可选 `worker_log_control_v1`，0/1/2/3 分别为 debug/info/warning/error。设置作用于整个共享 Worker，在模型加载前调用可控制加载日志。旧 peer 在 RPC 发送前被拒绝；响应必须精确回显级别。RPC 超时不撤销已提交设置。新增两项导出及独立的 1096 字节日志结构，已有选项、事件和状态结构不变。

## 最小 C 调用顺序

以下片段省略应用自身的错误展示，所有路径都关闭连接并释放句柄。实际 prompt token 应由匹配模型的 tokenizer 生成。

```c
#include "aila_inference.h"

int generate(const char* runtime, const char* model, const int32_t* ids, uint32_t count) {
    AilaInferenceError error = {sizeof(error), AILA_INFERENCE_ABI_VERSION};
    AilaInferenceClientHandle client = 0;
    AilaInferenceModelHandle loaded = 0;
    AilaInferenceSessionHandle session = 0;
    AilaInferenceRequestHandle request = 0;
    AilaInferenceRequestOptions options;
    AilaInferenceEvent event = {sizeof(event), AILA_INFERENCE_ABI_VERSION};
    int32_t status = aila_inference_default_request_options(&options, sizeof(options));
    if (status != AILA_INFERENCE_OK) goto done;
    options.max_new_tokens = 32;
    status = aila_inference_client_start(runtime, 0, 0, &client, &error);
    if (status != AILA_INFERENCE_OK) goto done;
    status = aila_inference_model_load(client, AILA_INFERENCE_QWEN3_NF4, model, 0, 120000, &loaded, &error);
    if (status != AILA_INFERENCE_OK) goto done;
    status = aila_inference_session_create(loaded, 0, &session, &error);
    if (status != AILA_INFERENCE_OK) goto done;
    status = aila_inference_request_submit(session, ids, count, &options, 0, &request, &error);
    if (status != AILA_INFERENCE_OK) goto done;
    for (;;) {
        status = aila_inference_request_read(request, 100, &event, &error);
        if (status == AILA_INFERENCE_TIMEOUT) continue;
        if (status != AILA_INFERENCE_OK) break;
        if (event.kind == AILA_INFERENCE_EVENT_TOKEN) {
            /* Consume event.token_id on this caller thread. */
        } else {
            if (event.local_failure || event.progress.status != AILA_INFERENCE_COMPLETED)
                status = AILA_INFERENCE_REQUEST_FAILED;
            break;
        }
    }
done:
    aila_inference_request_release(request, 0);
    aila_inference_session_release(session, 0);
    aila_inference_model_release(loaded, 0);
    if (client) {
        int32_t close_status = aila_inference_client_shutdown(client, 15000, 0);
        if (status == AILA_INFERENCE_OK) status = close_status;
    }
    aila_inference_client_release(client, 0);
    return status;
}
```

## 构建与验证入口

常规 `pwsh build.ps1` 生成 DLL、import library 和 C/FFI fixture。release target 将 DLL 放入 `Release/bin`、头文件放入 `Release/include`、import library 放入 `Release/lib`，GPU Worker 和设备运行库仍放入 `bin/aila_runtime`。新 DLL 的依赖收集使用 host-only 模式，拒绝 GPU runtime 依赖。

```powershell
ctest --test-dir build-refactor -R '^AilaInference(CHeader|CApi|Client|Pipe)Tests$' --output-on-failure
```

`tests/runtime/inference_c_api_tests.py` 仅接受 CPU Fake Worker。真实 GPU fixture 为 `AilaInferenceCApiGpuTests`，比较独立内部 Client 运行与 C ABI 的 token、usage 和 cache length；必须通过既有机器守卫、Windows Job 和 watchdog 启动。它验证 ABI 与已验收引擎的执行一致性，不取代独立模型数值 oracle 或长稳测试。

GPU fixture 的 `--session-context` 选项进一步验证四个会话的 reset、truncate-to-zero、thinking/seed 重放、续写、Busy 和父释放；任意目标的实际 checkpoint/replay 由 `AilaQwen3IpcSessionTests --case session-context` 对照独立的直接调用结果。

旧文本 C ABI 的可选接线见 [适配层说明](legacy-text-adapter.md)；媒体驱动、完整兼容边界和后续资源工作继续按 [完成审计](refactor-completion-audit.md) 推进。
