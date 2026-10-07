# Qwen4Exp Server 最小接入方案

状态（2026-10-07）：P1 普通 T1 已通过验收；P2/P3 已有实验性 MTP runtime/窗口提交实现，P4 的真实 JSON/SSE 和原长 prompt 请求已跑通，见 qwen38-server-mtp-2026-10-07.md/json。严格 T1 等价和 Server 20+ 未达成；本轮修正配置后的首字后输出约 14.98 tok/s。MTP 默认关闭，保留 T1 fallback。

目标：先让普通 Qwen4Exp NF4 T1 通过现有 LoadedModel → SessionBackend → Request/Scheduler → Worker IPC → HTTP/SSE 路径工作，再增加 opt-in T2 greedy。Server 保持纯 MSVC；SYCL、权重与 kernel bundle 留在 Worker 的 DeviceRuntime 执行域。

## P1：普通 T1 可工作的 fallback

| 修改位置 | 最小修改与完成条件 |
|---|---|
| src/inference/Model.hpp / Model.cpp | 增加 LoadQwen4ExpNf4；Load、reservation、backend 构造、CreateSession 都使用显式 Qwen4Exp family 分支。Q4E 不能进入 Qwen3 的默认 else。 |
| src/memory/SessionMemory.hpp | 新增有 overflow/维度检查的 qwen4exp_session_memory，覆盖 full K/V、indexer K/压缩 K、GDN/conv、PLE、host mirror 和指定数量的 snapshots。 |
| src/models/Qwen4ExpBnb4Backend.hpp / .cpp | 为 Session 提供 CacheBytes；先声明不支持跨 session decode batch。复用现有 CreateSession/Forward/Reset/Truncate 和最后有效行 logits 语义。 |
| src/inference/Model.cpp 的 ModelSessionBackend | Q4E 的 ForwardDecodeBatch 必须用 if constexpr/能力 trait 排除未实现的调用。仅让 SupportsDecodeBatch 返回 false 不够：虚函数模板实例化仍会引用缺失的 ForwardDecodeBatch。保留单行 fallback 和现有锁/异常 drain。 |
| src/worker/InferenceWorker.hpp / SharedInferenceEngine.cpp / InferenceWorker.cpp | 新增工厂 LoadQwen4Exp，调用 shared loader；拟用 backend 字符串 qwen4exp_nf4。模型加载一次、session 独立，沿用生命周期 reaper、内存限额与设备故障传播。 |
| src/ipc/InferenceProtocol.cpp | 增加 qwen4exp_nf4_bf16_kv capability、backend/kv_type 校验。握手能力、命令接受范围和实际 factory 一致。 |
| src/server/ServerMain.cpp / HttpServer.cpp | 放行该 backend；将 qwen3 ? Qwen3 : Qwen35 的 family 二选一改为明确分支。首版配置 decode-batch-size=1。 |
| src/server/ChatCodec.cpp | 使用模型 tokenizer 中的可信 chat template，明确 Q4E family；缺少模板时返回清楚的 unsupported 配置错误，不默认为 Qwen3.5。保持 OpenAI-compatible chat/completions、JSON 和 SSE 结构。 |

普通 greedy 的语义也要统一。Model.cpp 的 ReadOutput 当前使用 ops::argmax，新探针的两阶段 greedy 有最小 ID tie 与全量 BF16 finite guard。将后者从 tests/models/Qwen4ExpGreedyProbe.hpp 提取到内部 ops 组件，让 Q4E T1 和后续 MTP 共用；特殊采样继续走原请求 sampling/processor 路径。不能把测试探针整个 include 到 Worker。

P1 验收保持小而具体：一个真实文本请求、追加第二轮/Reset、内部 Truncate 的返回长度、两 session 交错的 KV/GDN/PLE 隔离、内存限额与先释放 model/runtime 句柄后的 session 生命周期；再跑一个 JSON 和一个 SSE 场景。协议/工厂变更用现有 CPU fake worker/socket 测试覆盖，不为每条配置新建框架。

## 内存预留：按现有分配计算

以下为当前固定 Q4E 几何、容量 C、host snapshots 数 S 的 payload；不是进程 RSS 或 WDDM 总占用：

- full K/V = 12 × 2 × 2 × C × 256 × sizeof(BF16)。
- indexer K = 12 × C × 128 × sizeof(BF16)。
- compressed indexer K = 12 × max(1, floor(C/4)) × 128 × sizeof(BF16)。
- R = 36 × (48 × 128 × 128 + 3 × 10240) × sizeof(float) = 117669888 bytes。
- PLE device state P = 9 × 10240 × sizeof(float) = 368640 bytes。
- device reservation = full KV + indexer K + compressed K + R + P。
- host reservation = R × (1+S) + P × S。PLE 当前态在 GPU，host PLE snapshot 每个 S 一份；小型整数元数据、vector/allocator 开销不在 payload 内。

C=2048 时 device=176234496 bytes（168.0703125 MiB）。S=4 时 host=589824000 bytes（562.5 MiB）；**现有 SharedFactory 为 IPC 明确设置 S=0**，对应 host mirror=117669888 bytes（112.21875 MiB），不能无意沿用 backend 的默认 4 个快照。IPC 没有 rollback API，保持这一现状；内部 Truncate 无可用快照时允许返回 reset 后的 0，并由调用者处理重放。

未来多模态 MRoPE/embedding overrides 不包含在首版文本资格中。模型共享权重、expert cache、PLE staging、执行 scratch 属于 model/执行域预算，不能乘 session 数或漏进 USM 总预算；session 只预留它实际独占的状态。MTP 临时 GPU checkpoint 与普通 host snapshots 是两类不同存储，不为 MTP 拒绝恢复把 S 隐式改成 4。

## P2：把 draft/verify/commit 迁成内部运行时组件

以当前有效配置为起点，不改变数学与调度策略：T2 CPU async、GPU expert 4/2/1 reuse、NF4 projection reuse、BF16 dual-query attention、batched greedy、state-only replay、prepare/hidden-reuse。公共 fallback 与数值检查保留。

所有权划分：

| 生命周期 | 内容 |
|---|---|
| model / 单一 SYCL Context | target/draft 权重、expert 权重 cache、可复用 kernel bundle、scratch/映射。DeviceRuntime mutex 串行保护可变执行资源。 |
| session | target KV/GDN/conv/PLE、draft KV/indexer/可见长度、必要历史和 generation 标记。Reset/Truncate/取消后的回放同时处理 draft。 |
| 执行窗口 | 预分配的 rollback state、捕获的 GDN 输入/PLE prefix、logits/greedy scratch、拒绝后缀元数据。热循环不分配 USM。 |

本次清理已把 bundle 从 probe 的局部 optional + borrowed pointer 改为 Qwen4ExpBnb4Backend 持有；PrepareDecodeVerification 校验调用 Context，初始化一次，backend 析构先 drain 再释放成员。后续直接复用这一所有权，不重新交给 Request/test 临时对象。

迁移时删除依赖 teacher 全量生成、扫描/预热所有专家、全量比较与文件 dump 的 probe 流程。生产请求不能重载模型或每次遍历 512 个专家。MTP 权重与缓存预算显式保留空间，避免压缩 target hotset；尚无这部分的服务器吞吐承诺。

首版资格：Q4E NF4、文本、greedy、T2、已验证的 dense-QSA/context 范围（先限制 2048）。存在 sampling、penalty/logit processor、需要 forced thinking close、vision、LoRA 或不支持的上下文时保留参数并走 T1。先不组合跨 session continuous batch × MTP；T2 是一个序列的两个连续位置，不是两个 Session。

## P3：窗口提交与 Request 状态

必须显式新增内部窗口执行/提交契约，不能伪装成 SessionState::Execute 的普通两个输入，也不能仅增加一个额外 token 队列。

目前 SessionState::Execute 校验 cache 恰好增长输入 token 数；Request 的 pendingToken 表示已经发布、尚未 forward 的 token，Finalizing 会消费一次。probe 则将窗口输入提交进 cache 后才发布它们，再留下未消费的 bonus。迁移要围绕现有 Request 的入口语义重新组织窗口：

| Request 入口 pending=t0（已发布、未缓存） | target 消费 | 新候选输出 | 待消费 token |
|---|---|---|---|
| 拒绝 draft d1 | t0 | 修正 token t1 | t1 |
| 接受 draft d1 | t0、d1 | d1、bonus t2 | t2 |

内部 result 至少明确 consumed input count、拟发布 token 前缀、哪些输出已在 cache 中、最终 cache length、未消费 bonus，以及可恢复/提交到指定可见前缀的窗口状态。RequestState 执行逐 token stop/EOS、max_new_tokens、thinking-budget、usage 和顺序检查后决定提交前缀。

例如只发布了已缓存的 d1，不能再把它交给旧 Finalizing 重复 forward；若 EOS/stop 阻止发布 d1，则不能保留其缓存后缀。仍未消费的 bonus 发布后才成为现有 pendingToken。剩余输出额度、上下文位置或输出缓冲空间不足两个时，直接走 T1，避免构造必然截短的窗口。

取消、断连、worker fault、slow consumer 后不再发布新 token；先排空已提交 GPU 工作。对已经执行但未被接纳的后缀恢复到明确前缀；若首版不能可靠保留取消时的可见历史，重置该 session 后才允许回池/重放，不复用超前的状态。stop-string/UTF-8 在 server streaming 层继续处理，取消信号必须走现有生命周期路径。

IPC 暂不需要 vector token 事件：窗口内部完成正确提交后，仍可逐 token 使用现有 sequence/stream 消息。HTTP 请求不修改 read-once 环境变量；性能配置收拢到 Worker/model 配置和资格检查，默认关闭 MTP，不暴露十几个实验参数。

P3 聚焦验收：接受/拒绝；EOS 在第 1/2 个输出位置；max_tokens=1 与奇数尾部；取消/断连/背压；两个 session 交错；完成后的下一轮追加。使用同一 greedy 实现对照 T1，检查 cache length、可见 token、pending 状态和恢复后的继续生成。

## P4：真实服务验收与交付口径

在 P1 已工作的 T1 上启用 MTP，使用相同 prompt/template/config，比较 JSON 与 SSE 输出、TTFT、decode、请求总时延和服务吞吐；不把探针 20.56 直接填作 Server 数据。再做两个 session 交错的状态隔离检查。保留默认 T1 与明确能力降级，Server 继续不链接 SYCL。

## 暂缓的性能 backlog

按块构造 draft KV（探针目前约 1.14 秒初始化）与首轮 target hotset handoff/初始化重叠，优先服务于 TTFT/总时延。去除 T2 GPU resident route 的主机依赖需要有效时间线与额外证据，放在后面。已失败的 cache/checkpoint/residual/FP8 分支不再作为下一轮候选。
