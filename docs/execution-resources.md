# 执行资源与分配额度

`DeviceRuntime` 现在统一计量其 `Context` 显式分配的 device USM 和 pinned host USM。权重、KV/递归状态、激活、采样缓冲和用户分配的 oneDNN scratchpad 都使用同一账本。普通 host mirror/快照有单独的 Session payload 额度。

这些计数描述引擎拥有的分配，不是 WDDM 驻留显存或进程 RSS。oneDNN primitive/driver 内部、STL 容器及 allocator 元数据、HTTP/管道/内核缓冲不能从 USM 账本推算，服务长期测试分别记录进程工作集、private bytes 和句柄。

## 配置与查询

```cpp
auto runtime = aila::inference::DeviceRuntime::Create();
// 示例：显式 device 分配最多 8 GiB，pinned host 最多 8 MiB。
runtime->SetAllocationLimits(8ull << 30, 8ull << 20);
aila::inference::ModelOptions options;
options.maxSessions = 4;
options.maxSequenceLength = 2048;
options.maxSessionHostBytes = 256ull << 20;
options.maxRecurrentSnapshots = 0;
auto model = aila::inference::LoadedModel::LoadQwen35Nf4(runtime, directory, options);
auto usage = runtime->Memory();
```

每项额度为 0 时不启用相应上限。`SetAllocationLimits` 只能在没有存活模型时调用；同域多个 Runtime 句柄看到相同设置，存活模型期间修改返回 `Busy`。

`TryMemory()` 在执行锁忙时返回空值，提供不等待GPU步骤的账本读取。可选 `runtime_memory_v1`、公开C查询和Server后台采样见 [运行时分配快照](runtime-memory.md)；加载时的静态memory字段继续保留。

| `MemorySnapshot` 字段 | 含义 |
| --- | --- |
| `currentBytes` / `peakBytes` | 显式 device USM 当前字节数 / 域生命周期峰值 |
| `pinnedHostBytes` / `peakPinnedHostBytes` | pinned host USM 当前字节数 / 峰值 |
| `deviceAllocations` / `pinnedHostAllocations` | 当前存活分配数量 |
| `deviceLimitBytes` / `pinnedHostLimitBytes` | 配置的分配上限 |

每次 USM 分配先校验整数溢出和剩余额度，成功后才增加账本。更新缓冲区时，新旧分配短暂共存也计入峰值和额度；不能仅以更新后的最终大小判断是否可分配。创建 Session 时先检查整份 cache/state 预留，拒绝不会先部分分配 KV。`maxSessionCacheBytes`、`maxSessionHostBytes` 和会话数限制继续独立生效。

## 所有权和执行顺序

- `Context::workspace<T>()` 拥有采样和 masked attention 工作区；不同 Context、线程迁移和模型重新加载不会借用其他设备的缓存。共享入口在设备执行锁内访问它们。
- 采样的 device/pinned 缓冲与 CPU 排序、概率、penalty 向量都归工作区。每个 Request 仍持有自己的 RNG；旧无显式 RNG 的接口保留线程本地 RNG 语义。
- Linear 的四种非 decode primitive 及 NF4 oneDNN prefill 使用同一份按需扩大的用户 scratchpad，每次 execute 重新绑定。它们的形状缓存各最多 16 项；masked attention 也最多 16 项。
- 专用 decode scratchpad 仍由各 projection 持有并记账。现有 NF4 内核、投影存储与跨层融合没有被这次资源改造替换。
- LoRA 的中间结果由 projection 持续持有，在扩大前 drain；异步 kernel 提交后不会因局部 Tensor 析构而提前释放。
- 共享 scratch 只适用于现有单执行槽和 in-order queue。扩大、淘汰 primitive 或释放资源前先排空队列。`Context::release_execution_resources()` 的调用者必须保证已 drain。

最后一个模型及其 Session/Request 强引用销毁后，设备域在锁内排空队列、释放模型，再释放 Context 工作区。Runtime 句柄可继续用于下一次加载；历史 peak 保留。

## 拒绝与恢复

配置额度不足抛出 `AllocationLimitExceeded`，共享 API 将其转换为 `ResourceLimit`。权重加载的逐 tensor 容错和 tied LM-head fallback 必须传播该异常，不能忽略失败后发布部分模型。

若 Forward 在若干层已执行后遇到额度拒绝，先 drain，再 Reset 受影响的 Session，随后报告局部失败。其余 Session 可以继续执行。普通取消仍保留实际已消费前缀；调用者不能把额度失败后 Reset 的状态误当作取消前缀。drain 本身失败时仍按 `RuntimeFault` 使整个设备域失效。

## IPC 与 Server

可选能力 `allocation_limits_v1` 保留 protocol 2。`InferenceModelOptions::allocationLimits` 请求三项 `uint64_t` 上限：`maxDeviceBytes`、`maxPinnedHostBytes`、`maxSessionHostBytes`。只有显式请求扩展时，model-load command 含 `allocationLimits`，response 含 `memory`；旧 Qwen3 默认能力和旧消息形状保持不变。

response 的 `memory` 回显全部 limits，并给出加载完成时的 `residentDeviceBytes` 和每 Session 的 `sessionHostReservationBytes`。Client 检查已协商能力以及响应是否完整、精确匹配请求；缺失、改变或未经请求的确认均作为协议错误。`residentDeviceBytes` 是加载完成时的显式 device 分配，后续执行 workspace 增长不更新这个字段。

可选旧文本 adapter 也通过此协议传递额度。`AILA_LEGACY_TEXT_MAX_DEVICE_BYTES`、`AILA_LEGACY_TEXT_MAX_PINNED_HOST_BYTES`、`AILA_LEGACY_TEXT_MAX_SESSION_HOST_BYTES` 分别对应上述三项；另有 `AILA_LEGACY_TEXT_MAX_SESSION_CACHE_BYTES` 和 1–4 的 `AILA_LEGACY_TEXT_MAX_SESSIONS`。字节字段默认 0，非法或溢出值在启动 Worker 前拒绝。设置取自共享 owner 的 Win32 环境快照，默认旧 v1 路径不读取这些限额；详见 [适配器边界](legacy-text-adapter.md)。

Server 要求此能力，并提供：

```text
--max-device-bytes N
--max-pinned-host-bytes N
--max-session-host-bytes N
```

默认都为 0。`/health` 和最终 report 的 `memory` 提供上述设置、`model_resident_device_bytes_at_load`、`session_device_reservation_bytes` 和 `session_host_reservation_bytes`。远端 Hybrid 禁用回滚快照，host 额度约束其必要 mirror。

## 验证入口

`AilaSessionMemoryTests` 和 `AilaRuntimeFaultTests` 验证 CPU 账本边界及错误传播。`AilaExecutionResourcesTests` 在 GPU 上检查双预算、Context/线程隔离、增长失败、scratch 重新绑定、masked attention 的独立 CPU 数值参考与缓存淘汰。`AilaLoraExecutionFusedTests` / `AilaLoraExecutionDnnlTests` 检查 LoRA 数值、异步持有期、增长、复用及释放。

```powershell
./build-refactor/AilaModelAllocationTests.exe --backend qwen3_nf4 --model ./models/qwen3-0.6B-bnb-nf4-offline
./build-refactor/AilaModelAllocationTests.exe --backend qwen35_nf4 --model ./models/qwen3.5-0.8B-bnb-nf4-offline
./build-refactor/AilaModelAllocationTests.exe --backend qwen35_bf16 --model ./models/Qwen3.5-0.8B
```

真实模型 fixture 检查极小额度拒载后恢复、Session 准入/回收、pinned 采样拒绝与另一会话继续执行。大 prefill 的局部拒绝仅在实测内存可将该场景与加载峰值区分时执行；否则明确报告 `NOT_COVERED`。GPU 用例全部串行，实际通过范围以 [验证记录](refactor-validation.md) 为准。
