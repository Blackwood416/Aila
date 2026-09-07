# Aila 重构实施路线

日期：2026-09-07。状态：P1/P2 共享会话切片之后，P3 已接入 Qwen3 与 Qwen3.5 Hybrid NF4 的具体 LayerPlan 和共享 Session，P4 token 调度器可复用两类会话。P5 的 Qwen3 token IPC v2 与普通 MSVC C++ HTTP/SSE 服务均已接通；真实 socket 的 11 项 fake Worker 和 7 项 Qwen3 GPU 验收通过。此前的 CPU、真实子进程 pipe、GPU IPC 与兼容集成门禁保留。BF16 权重策略复用、Hybrid 远端、长期压力和完整资源预算仍待后续推进。

本路线基于当前源码、已有性能脚本和对 Worker/IPC/C API 的专项复核。目标是统一层执行、共享模型多会话，以及 C++ HTTP/SSE 服务。本文描述完整目标；当前交付边界见下文，进程内 API 见 [shared-sessions.md](shared-sessions.md)，跨进程实现见 [inference-ipc-v2-implementation.md](inference-ipc-v2-implementation.md)，服务构建与请求示例见 [inference-server.md](inference-server.md)，实际验证记录见 [refactor-validation.md](refactor-validation.md)。

当前交付包括：

- 进程内 `DeviceRuntime / LoadedModel / SessionState / RequestState` 支持 Qwen3 与 Qwen3.5 Hybrid NF4 权重加载一次。不同会话持有独立 BF16 KV，Hybrid 还持有 DeltaNet 递归矩阵、卷积历史、host mirror 和快照；采样 RNG 归各 Request。共享入口显式拒绝量化 KV。
- 保留旧 Qwen3/Qwen3.5 默认会话入口，缓存按需创建。增加 CPU 状态/日志/内存预算测试、真实 GPU 会话对照、同工具链完整 token 比较器与兼容回归脚本。
- 将 `CApiWorkerEngine` 从 `WorkerMain` 提取为独立适配器；旧 `--ffi-worker`、WorkerProcess、Proxy 与公共 C ABI 保持 v1，新 `--ffi-worker-v2` 使用独立通信路径。
- Qwen3 与 Hybrid 在加载时构造固定 DecoderPlan/LayerPlan；Hybrid 绑定 Full Attention/DeltaNet 分支、缓存索引和下一层/最终 norm，显式传递阶段、位置、状态视图及共享工作区，保留 NF4 投影内核与跨层融合。4B 验证发现的分组 DeltaNet 卷积历史竞争另作正确性修复。显式 Forward 返回最后有效位置的一行 logits，旧 Hybrid 多行布局保持原语义。
- Hybrid 共享会话默认保留最近 4 个快照，0 禁用；Truncate 返回实际 checkpoint 长度，无可用快照时重置为 0。`ModelOptions` 同时检查会话数、device cache/state 与 host mirror/snapshot payload 额度，公式与 backend 共用。详细接口见 [qwen35-shared-sessions.md](qwen35-shared-sessions.md)。
- [Scheduler](scheduler.md) 用专用线程轮转推进请求，限制活动/未消费请求数、token/device cache 额度和输出 ring；支持排队取消、慢消费者隔离、一次终态、收到 RuntimeFault 后的故障传播与关闭。
- 独立普通 C++ HTTP/SSE 探针已实际构建并验证。Drogon 1.9.11 在 Windows 上需要记录下来的 listener callback 补丁；探针使用受控新连接与合成输出，详见 [refactor-http-probe.md](refactor-http-probe.md)。
- [IPC v2 协议](inference-ipc-v2.md) 已落到独立 codec、Worker controller/管道外壳、真实 Model/Session/Scheduler 适配，以及普通 C++ `InferenceClient`/Win32 transport。内部宿主 API 支持 epoch 绑定的 Model/Session/Request 句柄、多个在途请求、独立响应/事件路由、累计 ACK、release 私有排空和关闭屏障；首片只发布 Qwen3 NF4/BF16 KV token 能力。
- 本轮 CPU protocol 11 组、Worker 6 个测试函数、RuntimeFault 7 组、Scheduler 22 组、受控 InferenceClient 16 组均通过；真实 Windows 子进程 CPU pipe 7 组通过（6.96 秒），6 个 v1 兼容门禁通过（14.00 秒）。客户端单测耗时 0.30 秒。真实 Qwen3 GPU IPC 首跑通过（61.41 秒）：1/2/4 会话及续写逐 token 精确相同，seed/penalties、取消/背压、父释放、优雅关闭与新 epoch reload 均通过。这些门禁耗时不是服务性能结果。
- 最终全 Release 构建通过；既有 Q3 Session（23.21 秒）、Q3 Scheduler（26.33 秒）、Q35 0.8B（36.93 秒）GPU 回归全部通过，总计 86.48 秒；23 项基础/兼容/IPC 回归全部通过，60.67 秒。普通宿主导入检查无 SYCL。日志分别为 `build-refactor/ipc-v2-full-build.log`、`ipc-v2-model-gpu-regressions.log`、`ipc-v2-final-regressions.log`、`ipc-v2-host-dependents.log`，最终证据见 [refactor-validation.md](refactor-validation.md)。
- 普通 C++ `AilaServer` 已通过独立 Worker 提供健康检查、模型列表和 JSON/SSE Chat Completions，首片为本机 Qwen3 NF4 文本、最多 4 个并发请求。复用 tokenizer/聊天模板/reasoning 解析，支持独立 seed、usage、严格参数校验、断连与慢读取消、Worker 故障和优雅关闭。普通 MSVC 的 ChatCodec 19 组、Client 16 组、真实 pipe 7 组通过；11 项 fake socket 与 7 项真实 Qwen3 GPU socket 全部通过，完整记录见验证文档第 11 节。

此阶段尚未实现 BF16 权重共享策略、Hybrid 远端能力、新公共异步 C ABI、连续合批和完整媒体服务。当前 device/host/HTTP payload 额度不包含完整进程与驱动分配，不代表总内存或总显存准入。聊天格式化/reasoning 已复用于 Qwen3 HTTP 路径；thinking budget 仍由旧驱动支持，Hybrid 媒体 setter 只操作旧默认 Session。历史独立 HTTP 探针保留，真实服务使用单独的依赖与构建目录。

执行步骤、Reset/Truncate、模型加载和 Session 初始化已接入设备域故障处理。加载/初始化先保留同步 SYCL/oneDNN 首因，再 drain；成功发布资源前还要通过最终 drain。普通 host/file/config 和准入失败保持局部语义，清理 drain 失败才进一步使域故障。生命周期执行器通过 `Scheduler::ReportRuntimeFault()` 广播，通过 `DeviceRuntime::FaultMessage()` 获取析构 drain 留下的首因；已发布终态保持原状态。分类与广播的 CPU 注入门禁已经通过，详细约定见 [共享会话](shared-sessions.md#设备域故障与生命周期) 和 [调度器](scheduler.md#生命周期故障的外部报告)。

## 1. 交付范围

第一版面向 Windows、单张 Intel Arc GPU。当前已迁移 Qwen3 NF4 与 Qwen3.5 Hybrid NF4 的文本会话；BF16 权重路径的策略复用仍是后续工作。默认保留现有 NF4/XMX 内核、融合方式和 in-order SYCL queue。

第一版需要做到：

- 一份加载完成的模型权重服务多个独立 Session。
- 多个请求按 token 或 prefill 分块交替推进，分别流式输出。
- C++ 服务提供健康检查、模型信息和兼容常用 OpenAI Chat Completions 字段的 HTTP/SSE 接口；支持范围需要明确记录。
- 每个请求有独立取消、错误和输出状态；排队、活动会话及输出缓冲区都有上限。
- 旧 CLI、C ABI 和未迁移模型通过兼容路径继续工作。

连续合批、分页 KV、跨请求前缀共享、多 GPU、多 LoRA 同批和完整音视频并发作为后续阶段。交替推进属于多会话服务能力，吞吐收益需要实测。

## 2. 资源所有权与接口约定

| 对象 | 持有内容 | 共享与生命周期 |
|---|---|---|
| DeviceRuntime | SYCL context/device、首版 queue/stream、设备内存管理、oneDNN engine | 模型和会话使用同一设备域；存活时间覆盖所有设备资源 |
| LoadedModel | 配置、权重、融合权重、层计划、加载时确定的能力 | 加载和预处理完成后逻辑只读；多个 Session 共享 |
| SessionState | 每层 KV/递归状态、卷积历史、实际缓存长度和回滚快照；Hybrid 的 MRoPE/embedding override 也归对应 backend Session | 会话独立；现有媒体 setter 只操作旧默认 Session |
| RequestState | 输入进度、采样 RNG、生成 token、惩罚参数、停止/取消/错误 | 一次生成独立；完成后按句柄和输出消费情况释放；thinking budget 和文本流解析尚未迁入 |
| ExecutionWorkspace | 激活值、投影 scratch、oneDNN 参数绑定及 scratchpad、采样临时区 | 按执行槽复用；设备工作完成前不得回收或错误重绑 |
| Scheduler | 等待队列、可运行请求、资源准入、公平性和状态转换 | 首版一个 GPU 提交线程，调度多个 Session |
| Transport | IPC 编解码、响应路由、有界输出队列；HTTP 适配仍待接入 | v2 已使用独立 sender/readers 和每请求信用；GPU 提交线程不直接等待慢客户端 |

具体约定：

1. 首版每个 Session 最多有一个活跃生成。新接口对同一 Session 的重入返回 Busy，上层可自行排队；不同 Session 可交替推进。旧接口保留原有同步和忙碌语义。
2. HTTP 计划默认根据完整 messages 创建独立的临时 Session。显式长期 Session 是单独扩展，不隐式复用其他客户端的聊天历史。
3. 共享权重不等于共享现有 Linear/Bnb4BitLinear 对象。它们内部可变的 memory handle、参数 map 和 scratchpad 必须归入执行资源，或在单执行槽中受明确约束地使用。
4. 采样 RNG 归 Request，结果不随其他请求插入或执行线程切换而改变；进程环境变量只作为初始化默认值。
5. Session/Request 持有必要的强引用。取消后先停止后续提交，等待已提交设备工作安全结束，再回收缓存；禁止仅因 HTTP 断线就释放 USM。
6. IPC v2 内部 C++ API 已提供 Model/Session/Request ID，并结合 connection owner、类型及 Worker epoch 检查句柄。Worker 重启后所有旧远端句柄失效；旧公共 C ABI 仍使用 v1，不隐式转换为新句柄。
7. 旧 AilaEngine 作为默认模型/默认 Session 的兼容外观。ASR/TTS 派生句柄延长 owner 生命周期的规则、旧流式回调的线程约定必须保留。
8. 新异步接口显式约定事件投递线程；HTTP/SSE 使用请求事件队列。每个请求只发布一次终态，区分排队取消、运行取消、正常结束和失败。

## 3. 层执行设计

加载阶段生成 DecoderPlan，其中包含有类型的 LayerPlan。按 Attention 类型、FFN 类型和投影存储格式选择实现，不按模型名称在热路径反复分支。

- Attention：GQA/Full Attention、Gated DeltaNet，以及后续的 Sliding/Masked Attention。
- FFN：Dense SwiGLU；后续加入 MoE。
- Projection：NF4 与 Dense 作为策略，保留当前合并 QKV、gate/up 和硬件快路径。
- 执行输入：明确区分 Prefill/Decode；显式携带序列位置、缓存视图和 logits 输出位置。
- 执行边界：保留 residual + 下一层 input norm 等跨层融合；定义 hidden/normed 的输入输出约定。
- 调度边界：首版在完整 forward step 或 prefill 分块之间切换请求。层分发本身不意味着每层都要抢占或同步。

当前 Qwen3 和 Qwen3.5 都使用具体层计划，共用 Model/Session 的执行锁、采样与生命周期框架。Hybrid 的 Full/DeltaNet 分发已经落地；NF4 与 BF16 权重的公共投影策略仍待验证，不引入覆盖所有模型的通用图系统。

后续 BatchView 必须区分 token 行与独立序列，携带 sequence ID、位置、有效长度、缓存映射和 logits gather 索引。现有 seq_len 表示单序列长度，不能直接当作请求 batch size。

AR、Diffusion、TTS、ASR 和检测保留各自的驱动/能力接口。共用执行资源和适合复用的层，不把不同生成算法塞入同一个 AR 步进循环。

## 4. 阶段与验收

| 阶段 | 主要交付 | 依赖 | 通过条件 |
|---|---|---|---|
| P0 基线与约定 | 固定回归模型/输入/配置；记录数值、速度、内存；确定资源所有权和最小接口 | 无 | 基线可复跑，公共接口与状态转换有明确说明 |
| P1 执行骨架 | DeviceRuntime、LoadedModel、Session/Request、Workspace 最小骨架；兼容旧入口；CPU fake executor | P0 | 旧路径行为保持，生命周期与请求状态可独立测试 |
| P2 Qwen3 多会话纵向切片 | 拆出 Qwen3 NF4 的模型资源与会话缓存；RNG 请求化；手动推进两个 Session | P1 | 一份权重下 A/B 交替生成，各自与独立运行一致；释放一个会话不损坏另一个 |
| P3 层执行与 Hybrid | 将 Qwen3 接入 LayerPlan；迁移 Qwen3.5 Full/DeltaNet、快照、位置状态；验证 BF16 策略 | P2 | 两类模型复用执行框架，增量 prefill/回滚正确，保持已有融合与快路径 |
| P4 步进生成与调度 | AR 生成状态机、有界队列、内存准入、分块 prefill、decode 调度、取消与错误 | P2；Hybrid 验收依赖 P3 | 2/4 个会话可持续输出；短请求不必等长请求完整生成结束；固定 seed 不受交错影响 |
| P5 IPC 与 C++ 服务 | 多在途请求路由、新句柄 API、旧 ABI 适配、AilaServer、HTTP/SSE | 接口设计可在 P1 后开始；真实推理依赖 P4 | 多客户端流式完成，断线只取消对应请求，慢客户端不拖住全部请求，旧 ABI 测试通过 |
| P6 首版集成门禁 | 文本模型端到端回归、取消/异常/内存压力测试、单请求与并发性能报告 | P3/P4/P5 | 无跨会话污染、持续资源增长和旧能力回归；有可复现性能记录 |
| P7 连续合批 | Batched projection、逐序列位置/采样/LM head、批次增删；再评估分页 KV | P6 | 批处理数值正确，吞吐/延迟收益可量化，B=1 快路径保持 |

上表保留完整阶段目标，不能将其中的目标视为全部已交付。当前 P3 的两类 NF4 会话/层计划与 P4 进程内调度已经接通；P5 的 Qwen3 token Worker/宿主链路已接入真实 Model/Session/Scheduler，并通过本轮 GPU IPC 与最终集成回归。P3 的 BF16 权重策略、完整资源预算、Hybrid 远端及 P5 HTTP 服务仍未完成；IPC 通过不能代替 HTTP/SSE 验收。

### P0：基线准备

- 小模型使用仓库现有 Qwen3-0.6B NF4、Qwen3.5-0.8B NF4；较大形状加入 Qwen3.5-4B NF4。
- 记录实际编译版本、驱动/设备、模型、环境变量、prompt、采样参数与 seed。
- 覆盖初始 prefill、缓存续写、greedy、sampling/penalty、流式、thinking budget 和长上下文。
- 复用 smoke.ps1、bench.ps1、perf/presets.json 与现有 Chat/IPC/Worker/C ABI/kernel 测试。
- GPU 测量串行执行；区分 warmup/JIT 与稳定运行。记录 TTFT、prefill/decode 吞吐、每 token 延迟、请求总时长、设备/主机内存。
- P0 根据重复测量噪声确定性能回归门限。没有实际基线前，不用未经测量的百分比宣称性能保持。

### P1–P3：先证明资源拆分正确

- 将 Engine 的加载、生成、媒体处理逐步移到各自实现文件；保留对外兼容外观。每次提取应能独立构建和回归。
- LoadedModel 在加载时完成权重融合、转换和必要的 LoRA 绑定，再供 Session 使用。首版固定模型/adapter 组合；运行中替换 adapter 不作为共享模型的隐式操作。
- 首版采用连续 KV 和有限 Session 数量。当前已在创建前检查会话数、device cache/state 与 host mirror/snapshot payload；最大执行工作区及非 Context 分配的预算仍待补齐。大激活/scores 缓冲区按执行槽共享，避免每个空闲 Session 都保留一整份 prefill scratch。
- Qwen3.5 已将 linear_state、conv_state、head、dirty flag、host mirror、snapshots、MRoPE/embedding override 与缓存长度迁入 Session。共享会话默认最多 4 个快照，0 禁用且仍保留 mirror；旧默认会话保持原有快照行为。回滚返回实际恢复长度，调用者补算剩余前缀。
- 检查全局 masked-attention cache、profile totals、采样 workspace、Tensor 的 Context 生命周期及所有直接 USM 分配；新缓存需要设备归属和容量策略。
- 未迁移的后端通过能力标记/兼容适配继续单会话运行，不默认授予共享模型多会话能力。

P2 是首个决定后续方向的验收点：两条不同 prompt、不同长度、不同 seed 的请求交错推进；取消/销毁 A 后，B 继续产生与独立运行一致的结果；再次创建 Session 能正常复用资源。

### P4：调度与完成语义

- 分离输入准备、prefill 进度、decode 进度、输出消费；每一步都能返回可运行/等待/完成/失败状态。
- 首版一个 GPU 提交线程。保持 in-order queue 的设备顺序；临时工作区是否可以复用由已提交工作的生命周期决定。
- 长 prompt 按块 prefill，在块之间给 decode 请求运行机会。decode quantum 可调整，单请求时保留已有 chunk 快路径。
- 限制排队请求数、活动 Session 数、上下文/输出 token 和输出队列字节数。当前模型准入覆盖 KV、递归/卷积状态与 host 快照 payload；完整预算还需纳入执行 scratch 和非 Context 记账的设备分配。
- 慢消费者达到队列上限后，对该请求实施明确的暂停或取消策略；首版优先取消并安全回收，避免无界积压。
- 取消在提交边界生效。已经提交的 SYCL 工作仍需安全收尾；定义并测试取消最坏延迟。
- Worker/GPU 致命错误属于整个设备执行域：停止新执行，未终结请求在安全边界失败；已发布终态保持，既有输出尽可能排空。重建域并重新加载后才能继续，中途输出不自动重试。

### P5：IPC 与服务落地

当前 [IPC v2 协议](inference-ipc-v2.md) 的 Qwen3 token 首片已实现，接口示例与文件边界见 [实现记录](inference-ipc-v2-implementation.md)。codec、Worker controller/管道外壳、普通 C++ 宿主与故障前提已接通，受控 CPU 与真实子进程 CPU pipe 门禁通过；真实 Qwen3 GPU IPC 的 1/2/4 会话、续写、取消/背压及生命周期首跑，以及全 Release 构建、既有模型/调度和基础/兼容/IPC 回归均已通过。Hybrid 的进程内能力仍需另经远端接入与验收后才能作为 Worker 能力发布。

- 旧 WorkerProcess 保留 v1 的整请求互斥/同步等待。新 InferenceClient 使用常驻发送、响应/事件 reader 与独立 pending/request 表，允许多个在途请求。
- 响应和流事件走不同管道，最终响应可能先于末尾事件到达。宿主已按请求校验 accepted、final、terminal、连续 sequence、eventCount 与摘要，只有排空条件成立才向用户交付终态。
- v2 Worker 外壳持续接收命令，由控制器与生命周期执行器对接调度器；生成期间可接收其他 Session 的 submit/cancel/query。CApiWorkerEngine 继续属于 v1 兼容路径。
- RPC requestId 与 inferenceId 分开关联，Model/Session/Request 绑定 Worker epoch；日志与推理事件分开路由。协议扩展与公共 C ABI 兼容分开管理。
- 独立 response/event writer、每请求 token credit 和终态/final 保留槽已接入，为取消、ACK、release 与关闭保留控制容量。host reader 不执行用户/网络回调；整条物理 pipe 停读按连接级故障处理。
- 普通输入错误、慢消费者和请求取消只影响对应 Request。设备域故障保留首因并停止新执行；损坏协议、Worker 退出和连接读写失败失效整个连接。单请求等待超时不直接杀掉共享 Worker。
- 加载/Session 初始化同步设备异常分类、最终 drain、析构故障查询及 Scheduler 广播已完成，CPU RuntimeFault 7 组与 Scheduler 22 组通过。局部 host/file/config 或准入失败不一律升级为设备域故障。

服务接入与后续验收：

- 服务层已接入并完成首轮真实 socket 验收；长时间创建/释放、完整资源预算与 HTTP/IPC 服务性能仍需独立测量。
- 旧同步 C API 保持原调用线程上的文本回调。以后适配 v2 时只等待自己的请求；新服务接口使用独立、明确的异步事件约定。
- AilaServer 是独立 MSVC C++17 子项目，通过 InferenceClient 访问 Worker，PE 导入检查无 SYCL/oneDNN。构建从 Worker metadata 绑定 hello 身份，锁定依赖与补丁，并检查 Ninja 的头文件依赖有效。
- Windows listener 回调补丁已沿用；另为 Trantor 加入实际 pending-byte 查询，以单请求 16 KiB handoff 和 64 KiB 默认网络队列门限回传压力。已验证停读后恢复、持续停读取消及其他请求隔离。固定 loopback HTTP/1.1，一连接一个应用请求；连接复用和 pipeline 关闭该连接。
- GET /health、GET /v1/models、POST /v1/chat/completions（普通 JSON 与 SSE）已实现。字段、finish_reason、usage 与错误映射见 [服务文档](inference-server.md)。
- 复用现有 tokenizer、可信模型模板和 reasoning parser，默认关闭 thinking。工具调用、媒体、thinking budget、请求侧自定义模板尚未接入；非法/未支持参数显式拒绝。

### P7 及其他模型

- 先将同层投影聚合成多行 GEMM，Attention 可暂时按序列调用现有内核；独立验证这一步是否带来吞吐收益。
- 后续再做 batched/ragged attention 与分页 KV。分页 KV 涉及缓存写入、读取和回收，不应只替换 KVCache 容器。
- Qwen3.5 的递归状态要按序列 gather/scatter，padding 或不同请求的 token 不能更新错误的状态。
- 视觉/ASR/强制对齐在文本状态模型稳定后迁移。TTS 需要单独管理 talker、predictor 与 vocoder 流状态；DiffusionGemma 需要 canvas/去噪步骤驱动和专门的内存预算；YOLO 使用检测能力接口。

## 5. 子代理分工与集成

本会话可用原生 subagent 工具：创建子代理、追加任务、收发消息、查看状态、等待和中断。总并发名额为 4，包含主代理。已实际分派 Qwen3 后端拆分/层计划、Qwen3.5 Session 与 Full/DeltaNet 层计划及 GPU fixture、CPU 状态/调度测试、Worker 提取/生命周期审查、IPC v2 设计及独立 HTTP 探针；主代理负责调度器、公用接口、内存预算、集成和串行 GPU 验证。能力以当前会话的工具状态为准。独立 shell 命令可并行执行，实现任务也可使用 git worktree 隔离。

| 执行者 | 主要责任 | 可并行工作 | 需要避免的冲突 |
|---|---|---|---|
| 主代理 | 公共接口、状态/资源约定、调度核心、Engine/ABI 集成、CMake、最终验收 | 其他代理开发模块时推进集成与验证 | 公共头文件和调度核心保持单一负责人 |
| 子代理 A | 模型迁移与层执行 | 先 Qwen3，再 Qwen3.5；接口稳定后扩展其他后端 | 同一后端文件不由多人同时修改 |
| 子代理 B | Worker/IPC/Proxy 适配与传输测试 | 用 fake executor 完成多请求路由、取消、epoch 和 ABI 兼容 | 公共 ABI 变更先交主代理集成 |
| 子代理 C | 验收用例与 HTTP/SSE 适配 | 先构建 CPU 调度/事件测试和 HTTP 假服务，后接真实推理 | 不与主代理并行运行 GPU 基准 |

执行规则：

1. 每个任务都附范围、可修改文件、公共接口版本、依赖和验收条件。新增目录名/类型名不能被不同代理各自定义。
2. 初期只读复核可以共享工作区。实现阶段按文件归属工作；跨模块的大任务使用独立 git worktree 与 codex/ 分支，并分配独立构建/临时目录。
3. 主代理先落地公共接口，再分派依赖接口的实现。使用 stub/fake 可提前开发传输和测试；真实后端集成必须等依赖就绪。
4. 每个代理回传改动摘要、文件清单、验证命令/结果和未解决问题。集成后重新执行相关检查，不把分支单测通过等同于集成通过。
5. CPU 只读分析和互不干扰的测试可并行。GPU 正确性/性能测试由主代理串行安排，避免显存争用和结果污染。
6. 每个阶段形成可独立回归的提交。保留旧路径作为迁移期对照，在新路径的数值、性能和兼容验证通过后再移除重复实现。

## 6. 首版验收矩阵

| 维度 | 必测行为 |
|---|---|
| 数值 | 单请求与 A/B 交错的 greedy token；固定 seed/penalty；关键层输出；prefill/decode 一致性 |
| 会话 | 不同长度/上下文；reset、续写、截断、回滚；Qwen3.5 checkpoint 恢复后的补算 |
| 取消 | 排队取消、prefill 中取消、decode 中取消、重复取消、取消后 Session/设备资源的生命周期 |
| 传输 | 两个 SSE 流同时输出、慢客户端、断线、事件终态恰好一次、日志交错、旧 callback 线程约定 |
| 故障 | Worker 退出/超时、旧 epoch 句柄、内存准入拒绝、句柄乱序销毁和安全关闭 |
| 兼容 | CLI、旧 C ABI、ASR/TTS 派生句柄存活规则、聊天模板/工具调用/thinking budget、未迁移模型 |
| 性能 | 1/2/4 并发的总吞吐、TTFT/每 token 延迟分布、端到端时长；长 prefill 与短 decode 混合 |
| 资源 | 权重只加载一份；会话释放后可复用；长时间创建/销毁后 CPU/GPU 内存和句柄数量稳定 |

数值对比需区分仅移动代码与改变计算形状：前者优先要求原始 token 和选定中间结果一致；后续合批若改变浮点归约顺序，使用既有数值容差并分析首次分歧，不能只凭最终文本大致相同通过。

## 7. 工作量与推进方式

沿用前次评估：首版 Qwen3/Qwen3.5 文本的资源拆分、层执行、多会话与 C++ 服务，按熟悉 C++/SYCL 的工程师约 8–12 人周量级规划；连续合批和相关缓存/算子适配额外约 6–12 人周起。完整音视频/Diffusion 迁移另计。

人周是工程量估计，不是本次助手运行时长。子代理能并行独立模块和审查，但接口决策、集成与单 GPU 验证处于关键路径，不能把工期直接除以代理数量。

Qwen3 与 Qwen3.5 NF4 的 Session/LayerPlan 已接入 token Scheduler；Qwen3 IPC v2 与 C++ HTTP/SSE 已形成真实推理纵向切片，并通过 CPU、真实 GPU IPC/HTTP、兼容回归和受控网络压力验收。下一步优先扩展 Hybrid 远端能力，并补齐更大模型/长上下文、长期创建释放、完整内存预算和服务延迟分布；BF16 权重策略复用、thinking budget/工具/媒体驱动与连续合批继续分阶段迁移。验证覆盖范围与数值由 [refactor-validation.md](refactor-validation.md) 持续记录。

## 8. 源码依据

- include/engine/Engine.hpp：初始化、聊天/媒体处理、整段生成循环与会话成员集中。
- src/models/IModelBackend.hpp：当前全模型 forward、缓存截断和多模态/扩散能力接口。
- src/models/Qwen3Bnb4Backend.cpp：逐层执行、跨层 residual/norm 融合、仅末位置 logits。
- src/models/Qwen35HybridBnb4Backend.hpp/.cpp：显式 Session 持有 Full KV、DeltaNet/卷积状态、媒体位置与快照；模型执行槽共享权重/临时区，具体 LayerPlan 保留融合；区分旧多行与显式末行 logits。
- src/inference/Model.hpp/.cpp、src/memory/SessionMemory.hpp：两类 NF4 模型的公共会话适配、执行锁、device/host payload 准入与生命周期。
- src/ops/Bnb4BitLinear.hpp、src/ops/Linear.cpp：可变 oneDNN 绑定、primitive cache 与 scratchpad。
- src/ops/AttentionOps.cpp、src/ops/SamplingOps.cpp：进程级 attention 缓存、线程级 RNG/workspace。
- src/ipc/InferenceProtocol.hpp/.cpp：v2 独立 schema/codec、协商限额、严格 token/摘要校验。
- src/runtime/InferenceClient.hpp/.cpp、src/runtime/InferencePipeTransport.cpp：普通 C++ v2 宿主句柄、pending/流路由、消费确认、Windows 子进程与关闭。
- src/worker/InferenceWorker.hpp/.cpp、src/worker/InferenceWorkerMain.hpp/.cpp、src/worker/SharedInferenceEngine.cpp：控制器、管道外壳、生命周期回收、Scheduler bridge 与真实模型适配。
- src/inference/RuntimeFault.hpp、src/utils/DeviceException.hpp：首因记录、加载/初始化最终 drain、局部恢复边界的设备异常传播。
- src/runtime/WorkerProcess.cpp：v1 的整个 request/request_stream 生命周期 operation_mutex 与同步响应匹配。
- src/worker/WorkerMain.cpp：启动模式分流；旧 v1 流式分支仍只接受当前请求的 cancel。
- src/proxy/AilaProxyApi.cpp、tests/runtime/ProxyAbiTests.cpp：派生句柄 owner、Worker generation 与旧回调线程语义。
- perf/presets.json、perf/oneapi-regression-cases.json、tests/：已有基线、功能和兼容性测试资源。
- docs/qwen35-shared-sessions.md、docs/inference-ipc-v2.md、docs/inference-ipc-v2-implementation.md：Hybrid 进程内约定、IPC v2 协议契约及已实现 token 切片的入口与边界。

子代理工具说明参考：[OpenAI 官方 Subagents 文档](https://developers.openai.com/codex/subagents)。
