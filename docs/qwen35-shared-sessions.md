# Qwen3.5 Hybrid 共享文本会话

`LoadedModel::LoadQwen35Nf4` 将 Qwen3.5 Hybrid NF4 接入已有 `SessionState / RequestState / Scheduler`。权重与可变 oneDNN 绑定、激活和投影 scratch 共用一个执行槽，设备域的执行锁串行保护该槽。多个会话在完整 prefill 分块、decode 或 finalization 之间轮转。

该入口面向 token 文本推理，要求 BF16 KV。Qwen3.5 的旧 CLI、媒体入口和多行 logits 约定继续通过默认 Session 外观运行。共享 API 尚未提供图像输入、thinking budget 或新的公共 C ABI。[IPC v2](inference-ipc-v2-implementation.md) 的 `InferenceClient::LoadQwen35` 与 [HTTP/SSE 服务](inference-server.md) 的 `--backend qwen35_nf4` 已接入同一共享后端；聊天格式化在 HTTP 宿主完成。

## 独立状态与层执行

每个 Hybrid Session 独立持有：

- Full Attention 的 K/V 与实际上下文长度。
- 每个 DeltaNet 层的 float32 递归矩阵、卷积历史、环形位置、host mirror 和 device dirty 标记。
- 回滚快照及其 checkpoint 长度。
- MRoPE 位置、文本位置偏移和 embedding override 存储。现有媒体 setter 只操作旧默认 Session；这些状态不会流入显式文本 Session。

加载时生成具体 Full Attention / DeltaNet LayerPlan，绑定层投影、cache 索引和下一层或最终 norm。保留 NF4 投影内核、跨层 residual/norm 融合及 in-order queue，不在每层之间增加主机同步。profiling 汇总归模型执行槽。

4B 分组 DeltaNet 验证暴露了已有 fast path 的跨工作组竞争：多个 value head 的工作组读取同一份 Q/K 卷积历史，leader 提前覆盖历史时，工作组内 barrier 无法保护其他工作组。分组路径现在在计算期间只读初始历史，prefill 块内的前序 token 从投影输入读取；计算后在同一 in-order queue 上写回最后几行历史。非分组路径保留原有计算方式。这个正确性修复同时用于 NF4 后端的旧入口和显式 Session；BF16 权重后端不在本轮迁移范围内。

显式 `Forward` 返回最后输入 token 对应的一行 logits。旧入口在部分小块增量执行中返回多行，且张量容量可能大于本次有效行数；共享采样不根据容量猜测有效位置。新旧入口使用同一个执行实现，通过明确的返回模式区分这两种约定。

## 使用与资源预算

```cpp
#include "inference/Model.hpp"
#include "inference/Scheduler.hpp"

using namespace aila::inference;

auto runtime = DeviceRuntime::Create();
ModelOptions limits;
limits.maxSequenceLength = 512;
limits.maxSessions = 4;
limits.maxRecurrentSnapshots = 4;
// 示例预算适用于本次 Qwen3.5-0.8B、512 context 的四个会话。
limits.maxSessionCacheBytes = 128 * 1024 * 1024;
limits.maxSessionHostBytes = 512 * 1024 * 1024;
auto model = LoadedModel::LoadQwen35Nf4(runtime, modelDirectory, limits);
auto session = model->CreateSession();

RequestOptions options;
options.prefillChunkSize = 128;
options.generation.max_new_tokens = 64;
options.generation.use_fixed_seed = true;
options.generation.sampling_seed = 42;
options.stopTokenIds = {eosTokenId};

Scheduler scheduler;
auto request = scheduler.Submit(session, promptTokens, options);
// 使用 request->Read(...) 消费 token 和唯一终态，详见 scheduler.md。
```

创建会话前同时检查模型会话数、device cache/state 和 host mirror/snapshot 额度。最后一个 Session 强引用销毁后归还额度；丢弃外部 Model 或 Runtime 句柄不会提前释放存活请求使用的资源。权重发布为共享模型后不允许运行中 reload 或 LoRA 替换。

| 设置或查询 | 含义 |
| --- | --- |
| `maxSessionCacheBytes` | 模型下全部 Session 的 device cache/state payload 上限，0 表示只使用会话数限制 |
| `maxSessionHostBytes` | 全部 Session 的 host mirror 加最大保留快照的 float payload 额度，0 表示只使用会话数限制 |
| `maxRecurrentSnapshots` | 每 Session 最多保留的 checkpoint 数，默认 4，0 禁用快照；Qwen3 不使用此设置 |
| `SessionReservationBytes()` / `SessionState::CacheBytes()` | 每 Session 的 Full KV、DeltaNet 矩阵及卷积状态 device 字节数 |
| `SessionHostReservationBytes()` | 每 Session 的 host mirror 与快照最大 float payload 字节数 |

预算公式与 backend 共用 `src/memory/SessionMemory.hpp`，在分配前校验维度和 `size_t` 溢出。Qwen3.5-0.8B 在 context 512 下，每会话 device payload 是 **26,492,928 B**：Full KV 6,291,456 B，加递归/卷积状态 20,201,472 B；4 个快照时 host payload 额度是 **101,007,360 B**。禁用快照后仍保留 20,201,472 B 的 host mirror。

这些额度不包含权重、共享 workspace、oneDNN/driver 分配、STL/allocator 元数据或网络缓冲，因此不构成完整的进程内存或显存预算。host 快照按需分配，额度按上限预留；快照复制前先淘汰超额的最旧项，避免同时持有第 N+1 份快照 payload。

Forward 与 Reset/Truncate 的设备执行异常会失效共享设备域。加载和 Session 初始化已保留首次同步 SYCL/oneDNN 异常并归一为 `RuntimeFault`；成功发布前还要完成最终 drain。普通文件/配置/准入失败保持局部语义，清理 drain 失败才进一步升级为域故障。生命周期执行器查询析构 drain 留下的首因并向 Scheduler 广播，详细约定见 [IPC v2 故障语义](inference-ipc-v2.md)。

## 回滚、续写与取消

快照沿用已有时机：多 token prefill 分块结束，以及 decode 长度达到 `AILA_Q35_PREFILL_STEP` 边界（默认 64）。共享会话只保留最近 N 个 checkpoint；旧默认会话保留原有快照策略。

上述默认 4 个快照用于进程内 API。远端 v2 没有 Reset/Truncate 命令，Worker 适配器固定关闭快照，保留续写和必须的 host mirror；HTTP 每次创建独立 Session。以后开放远端回滚时还需要增加实际恢复长度、host 字节准入及对应验收，不能假设已有远端会话可回滚。

`SessionState::Truncate(requestedLength)` 返回实际恢复长度。它选择不大于 requestedLength 的最近保留 checkpoint；没有合适快照时重置为 0。调用者必须从返回位置补算自己的输入历史，不能把请求的长度当作实际缓存长度。例如已有 checkpoint 11、17，在长度 21 时请求截断到 20，实际返回 17。

每次恢复同时处理递归矩阵、卷积历史、环形位置和 host/device 一致性，然后排空状态复制，才允许后续操作或销毁。Full KV 按恢复后的逻辑长度覆盖续写。恢复点之后的快照被移除，其他 Session 的状态保持独立。

一个 Session 同时只允许一个生成租约，活动时 Reset/Truncate/直接 Forward 返回 Busy。取消与丢弃句柄在安全步骤边界生效；取消后的缓存可能不含最后一个已发布 token，应查询实际长度并按自己的历史协议补算或 Reset。请求状态、停止 token、错误与输出背压规则沿用 [scheduler.md](scheduler.md)。

## KV 配置与验证入口

Qwen3.5 使用有效 `AILA_VLM_KV_QUANT`，未设置时回退到 `AILA_KV_QUANT`（含现有 engine override）。显式 scoped 值 0 优先于全局值 1。共享模型加载和创建 Session 均检查有效设置，在权重或缓存分配前拒绝量化 KV；不改变旧入口的配置语义。

```powershell
pwsh -NoProfile -File build.ps1 -BuildDir build-refactor
ctest --test-dir build-refactor -C Release -R '^AilaSessionMemoryTests$' --output-on-failure

$env:AILA_QWEN35_TEST_MODEL = 'E:\RiderProjects\Aila\models\qwen3.5-0.8B-bnb-nf4-offline'
$env:AILA_KV_QUANT = '0'
$env:AILA_VLM_KV_QUANT = '0'
ctest --test-dir build-refactor -C Release -R '^AilaQwen35SessionTests$' --output-on-failure

# 串行验证 4B 的分组 DeltaNet，或单独复现混合形状的 logits 检查。
$env:AILA_QWEN35_TEST_MODEL = 'E:\RiderProjects\Aila\models\qwen3.5-4B-bnb-nf4-offline'
ctest --test-dir build-refactor -C Release -R '^AilaQwen35SessionTests$' --output-on-failure
./build-refactor/AilaQwen35SessionTests.exe --model $env:AILA_QWEN35_TEST_MODEL --case interleaved-logits
```

没有提供真实模型时 GPU fixture 返回 77，CTest 标记跳过。所有 GPU 检查、CLI 对照与 benchmark 串行运行。fixture 覆盖独立 legacy 有效末行参考、同形状逐步 logits、四会话生成/续写、快照恢复/淘汰/禁用、取消与资源回收、所有权、host/device 预算和 scoped/global KV 配置矩阵。执行结果及范围见 [refactor-validation.md](refactor-validation.md)。

`--case four-sessions`、`interleaved-logits` 和 `legacy-last-rows` 可独立运行对应场景；默认 `all` 保留完整门禁。筛选运行使用新的进程和工作区，比较仍要求每个 token / logit 精确一致，不放宽浮点容差。
