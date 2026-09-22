# 运行时分配快照

`runtime_memory_v1` 为当前 Worker 的设备执行域提供显式 USM 账本查询。它补充加载时的 `ModelMemoryInfo`，可观察 Session、采样和共享执行工作区的增长与释放。查询不创建 GPU、不提交 kernel，也不调用队列同步。

## 状态和范围

| 状态 | 含义 |
| --- | --- |
| `uninitialized` | 该 factory 尚未创建运行时；查询不会为观察而初始化设备 |
| `busy` | 生命周期操作正在进行，或无法立即取得执行锁；没有可用的本次快照 |
| `ready` | 返回一份在执行锁内读取的完整 CPU 账本；不表示设备健康或 GPU 工作已经完成 |

未初始化和busy的wire `memory`为null。C ABI返回OK并设置相应state，其他数值字段不表示零占用。设备健康继续由Client/Worker状态和故障事件判断。

快照包括device和pinned host的当前字节、历史峰值、存活分配数量以及当前上限。所有wire数值为严格uint64；当前值不得超过峰值或非零上限，零字节与零存活分配必须一致。模型释放后峰值保留；此后降低上限时，旧峰值可以超过新上限。

这些字段不包含WDDM驻留、oneDNN/驱动私有分配、普通host mirror/快照、STL/allocator元数据或网络缓冲。查询不是额外预留，也不能据此承诺后续分配一定成功。额度范围见 [执行资源](execution-resources.md)。

## C++ 和 C API

进程内 `DeviceRuntime::TryMemory()` 返回 `optional<MemorySnapshot>`，执行锁忙时立即返回空值。原有阻塞 `Memory()` 保留。IPC的 `InferenceClient::QueryMemory(timeout)` 是一次显式RPC；旧默认能力集合不变，调用查询时才检查可选的 `runtime_memory_v1`。

Worker把查询计入普通命令额度，保留取消/ACK等控制容量。生命周期队列有未完成操作时直接返回busy；否则在生命周期线程访问factory，并用try-lock读取账本。查询不会排在加载或Session初始化的GPU等待之后。

公开C接口增加独立结构和函数，既有结构布局与ABI版本1保持不变：

```c
AilaInferenceRuntimeMemoryInfo memory = {sizeof(memory), AILA_INFERENCE_ABI_VERSION};
AilaInferenceError error = {sizeof(error), AILA_INFERENCE_ABI_VERSION};
int32_t result = aila_inference_client_query_memory(client, 1000, &memory, &error);
if (result == AILA_INFERENCE_OK && memory.state == AILA_INFERENCE_MEMORY_READY) {
    /* device_bytes / pinned_host_bytes / *_allocations 属于 memory.epoch。 */
}
```

超时0使用配置的RPC超时，不是非阻塞poll。结构头在发送RPC前校验；较大的struct_size只写已知前缀。超时、关闭、能力缺失或其他RPC错误通过原有错误码返回，不用上一份快照替代失败。busy可以由调用者在合适的观测间隔重查，不能把查询用于等待GPU完成。

## Server 后台采样

`AilaServer --memory-sample-ms N` 支持0关闭或100–60000ms；默认0，不增加原有服务的查询流量，也不增加默认report字段。显式启用时才要求新Worker能力。

开启后，独立宿主线程发起最多一个在途内存RPC，每次1秒超时。`/health`和最终report只读取缓存，原来的`memory`加载信息保持不变，新增 `runtime_memory`：

- `state`、`memory`、`epoch`：最后一次查询结果；尚无结果时为pending，查询失败时为error且memory为null。
- `age_ms`：宿主收到最后一次查询结果/错误后经过的时间，不是GPU采样时刻或设备完成时间。
- `samples`、`busy_samples`、`failures`：查询结果计数。
- `sampling_stopped`：采样线程已停止；关闭后的缓存不代表Worker当前仍持有这些分配。
- `error`：timeout、query_failed或limit_mismatch。有效快照的上限必须与Server加载配置一致。

发生查询错误或超时后采样停止并保留错误状态，避免原RPC尚未退役时不断追加诊断命令。普通HTTP请求继续遵守自己的执行与故障语义。关闭时先停止并join采样线程，再关闭共享Client；HTTP事件循环不进行内存RPC或等待执行锁。

开启采样会增加宿主/IPC观测成本；不能直接把开启采样的耗时与默认关闭的 [服务性能基线](service-baselines.md) 当作同一测量条件。

## 验证

协议、Worker、Client和C ABI的CPU用例覆盖三种状态、精确整数、非法快照、能力缺失、结构布局/尾部保留、生命周期忙和控制容量。专用HTTP用例覆盖关闭、未初始化、ready/busy、错误与超时停止、HTTP可用性和推理结果保持：

```powershell
python tests/server/memory_socket_tests.py --server build-refactor-server/AilaServer.exe --worker build-refactor/AilaFakeInferenceWorker.exe --output tmp/refactor/memory-http
```

该命令仅接受CPU fake Worker。真实GPU验证使用机器守卫、Windows Job和watchdog，检查与冻结旧版本的完整token序列、并发查询、取消恢复、Session重复创建释放和最后模型释放后的账本归零；实际覆盖范围记录在 [重构验证](refactor-validation.md)。
