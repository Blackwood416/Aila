# C++ Qwen3 / Qwen3.5 Hybrid HTTP/SSE 服务

**当前 GPU 验证已恢复：** 用户完成 DDU 清理、8860 WHQL 全新安装及重启；驱动/loader 核验和已完成的 Qwen3 压力测试见 [8860 验证记录](gpu-stability-8860-20260909.md)。后续 GPU 用例继续使用串行守卫、独立 Windows Job 和整轮超时。

`AilaServer` 是 Windows x64、普通 MSVC C++17 的本地文本服务，通过 [InferenceClient / IPC v2](inference-ipc-v2-implementation.md) 启动独立 `AilaWorker.exe`。Worker 加载一份 Qwen3 NF4 或 Qwen3.5 Hybrid NF4/BF16 权重，最多服务 4 个独立 BF16 KV Session；Hybrid 的 DeltaNet 递归矩阵、卷积历史和 host mirror 也归各 Session。每次 HTTP 请求创建新 Session，多轮聊天通过 `messages` 传入完整历史。

服务固定监听 `127.0.0.1`，支持 HTTP/1.1。宿主不链接 AilaLib、SYCL 或 oneDNN；设备运行时留在 Worker 进程中。默认由 Scheduler 交替推进独立请求；本轮新增可选的 [连续 decode 合批](continuous-batching.md)，其最终 CPU 集成、Fake Worker HTTP、四模型设备数值、B1 对照、Request/额度恢复及真实 HTTP 功能检查已通过。NF4 的历史验收及此前 BF16 范围见 [验证记录](refactor-validation.md) 第 11–12、18 节，本轮结果另记于第 33 节。

## 构建与运行

先按主项目要求安装 Intel oneAPI、Visual Studio C++ x64 工具链、CMake 和 Ninja。分别构建 Worker 与普通宿主：

```powershell
pwsh -NoProfile -File build.ps1 -BuildDir build-refactor -Config Release -Jobs 6
pwsh -NoProfile -File tools/server/build.ps1 -Parallelism 6
```

第二条命令首次会构建隔离的服务依赖。默认目录为 `build-refactor-server-deps` 和 `build-refactor-server`，不复用历史 HTTP 探针目录。依赖已构建且锁文件校验通过后，可使用：

```powershell
pwsh -NoProfile -File tools/server/build.ps1 -SkipDependencies -Parallelism 6
ctest --test-dir build-refactor-server -C Release --output-on-failure
```

`-WorkerBuildDir`、`-BuildRoot` 和 `-DependencyRoot` 可覆盖这三个目录。构建脚本从 Worker 的 `build_info.json` 读取完整 Git 身份，编译到宿主 hello 校验中；更新或重新提交 Worker 源码后，先重建 Worker，再重建 Server。构建信息与依赖哈希保存在 `server-build-info.json` 和 `dependency-lock.json`。

```powershell
$env:AILA_KV_QUANT = '0'
$env:AILA_VLM_KV_QUANT = '0'
./build-refactor-server/AilaServer.exe --worker ./build-refactor/AilaWorker.exe --model ./models/qwen3-0.6B-bnb-nf4-offline --port 8080
```

启动 Hybrid 时显式选择后端；默认 HTTP 模型名相应变为 `qwen3.5`：

```powershell
./build-refactor-server/AilaServer.exe --worker ./build-refactor/AilaWorker.exe --backend qwen35_nf4 --model ./models/qwen3.5-0.8B-bnb-nf4-offline --port 8080
```

未量化 Qwen3.5 BF16 checkpoint 使用独立后端名，仍复用 Qwen3.5 的聊天模板、reasoning、会话隔离与资源额度：

```powershell
./build-refactor-server/AilaServer.exe --worker ./build-refactor/AilaWorker.exe --backend qwen35_bf16 --model ./models/Qwen3.5-0.8B --port 8080
```

加载完成后，stdout 输出包含 `event=ready`、端口、模型名、backend 和 Worker epoch 的 JSON；`/health` 与最终 report 也提供 backend。后端与模型配置不匹配时启动失败，不自动切换。Ctrl+C 停止接入新请求、取消活动请求、关闭 socket，并等待 Worker 退出。也可指定 `--shutdown-file PATH`，由受控启动程序创建该文件触发关闭；`--report PATH` 将最终状态写成 JSON。没有 HTTP 关闭端点。

主要选项：

| 选项 | 默认值 | 约束 |
| --- | ---: | --- |
| `--backend` | `qwen3_nf4` | `qwen3_nf4`、`qwen35_nf4` 或 `qwen35_bf16`；一个进程加载一个模型 |
| `--model-id` | `qwen3` / `qwen3.5` | 按后端选择默认值；可覆盖为 1–128 个 ASCII 模型名字符 |
| `--port` | 8080 | 1–65535，固定 loopback |
| `--context` | 4096 | 32–4096 token |
| `--sessions` | 4 | 1–4 个同时占用的请求槽 |
| `--decode-batch-size` | 1 | 1–4，且不超过 sessions；同 LoadedModel/固定 adapter 的就绪 decode 行上限 |
| `--max-device-bytes` | 0 | Context 显式 device USM 上限，包含权重/缓存/执行 scratch；0 不启用 |
| `--max-pinned-host-bytes` | 0 | Context 显式 pinned host USM 上限；0 不启用 |
| `--max-session-host-bytes` | 0 | 全部 Session host mirror payload 上限；0 不启用 |
| `--max-new-tokens` | 1024 | 1–1024，超过 context 时缩至 context−1 |
| `--max-output-bytes` | 262144 | 4096–4194304，原始文本和序列化输出分别受限；预留 2048 字节错误收尾 |
| `--stream-buffer-bytes` | 65536 | 32768–1048576，Trantor 待写数据门限 |
| `--stream-stall-ms` | 5000 | 100–60000，持续无法接纳下一帧的时间 |
| `--request-timeout-ms` | 120000 | 1000–600000，从 HTTP 请求接入开始计算 |

Server 要求 Worker 的 `allocation_limits_v1` 能力，并检查额度确认。`/health` 与最终 report 的 `memory` 回显上限、加载完成时的模型 device 分配以及每 Session 的 device/host 预留；这些字段不表示当前 RSS 或 WDDM 驻留显存。配额范围、增长峰值及失败恢复见 [执行资源](execution-resources.md)。

可用 `--memory-sample-ms 100` 开启独立的运行时USM采样，默认0关闭，允许100–60000ms。启用后 `/health` 增加带状态、年龄和停止标记的 `runtime_memory` 缓存；HTTP线程不作内存RPC。busy/未初始化不表示零占用，查询失败停止采样并保留错误。字段和限制见 [运行时分配快照](runtime-memory.md)。

`--decode-batch-size 2` 或 `4` 在启动时要求 Worker 的 `continuous_batching_v1` 能力；不支持、数值越界或未精确回显时启动失败。默认 `1` 保持旧配置帧，不要求该能力。`/health`、ready 消息和最终 report 的 `max_decode_batch_size` 回显配置上限，不表示实际批次大小。prefill 仍独立分块轮转，取消、输出背压、请求槽和资源额度保持各请求的边界。Qwen3 的 `AILA_PROFILE_Q3_DECODE=1`、Hybrid 的旧单序列诊断及 legacy linear 模式回退为单请求执行；`AILA_FUSE_RESIDUAL_ADD=1` 时，三个后端也均回退到完整 B=1 路径，保留原残差融合语义。该选项属于服务配置，不是 HTTP 请求 JSON 字段；多 LoRA、分页 KV 和 batched attention 不由本开关启用。

## HTTP 接口

| 方法与路径 | 行为 |
| --- | --- |
| `GET /health` | 就绪返回 200；停止或 Worker 故障返回 503，附请求、IPC 和连接计数 |
| `GET /v1/models` | 返回当前单个模型的 OpenAI 风格模型列表 |
| `POST /v1/chat/completions` | `application/json` 请求；完整 JSON 或 SSE 响应 |

示例：

```powershell
$request = @{
    model = 'qwen3'
    messages = @(@{ role = 'user'; content = '用一句话解释为什么天空是蓝色的。' })
    max_tokens = 64
    temperature = 0
} | ConvertTo-Json -Depth 6
Invoke-RestMethod -Uri http://127.0.0.1:8080/v1/chat/completions -Method Post -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($request))
```

需要流式输出时，增加 `"stream":true`；增加 `"stream_options":{"include_usage":true}` 可在结束前获得 usage 帧。每条 SSE 记录使用 `data: <JSON>\n\n`，最后为 `data: [DONE]\n\n`。先发送 assistant role，再发送 `content` / `reasoning_content` 或 `tool_calls` 增量；正常终态 `finish_reason` 为 `stop`、`length` 或 `tool_calls`。终态之后仅发送可选 usage 和 `[DONE]`。启用 include_usage 时，普通 chunk 带 `usage:null`，最终 usage chunk 的 `choices` 为空数组；未启用时省略 usage 字段。

支持字段：

| 字段 | 语义 |
| --- | --- |
| `model`、`messages` | 必填；model 必须等于 `--model-id`，messages 为 1–256 条文本消息 |
| 消息 `role` | `system`、`developer`、`user`、`assistant`、`tool` |
| 消息 `content` | 字符串，或含 1–256 个 `{ "type":"text", "text":"..." }` 的数组；文本段顺序拼接；携带非空 tool_calls 的 assistant 可省略或传 null |
| 消息 `name`、`reasoning_content` | name 为 1–64 字节字符串；reasoning_content 仅允许 assistant 消息 |
| assistant `tool_calls` | 1–128 个 `{id,type:"function",function:{name,arguments}}`；arguments 必须是严格 JSON object 的字符串 |
| tool `tool_call_id` | 必填，关联上一轮尚未完成的 assistant 调用；历史中 ID 唯一、结果不可重复，全部结果返回后才能开始下一普通消息 |
| `tools` | 至多 128 个 function 定义；name 唯一，1–64 个 ASCII 字母、数字、下划线或连字符；可选 description 最多 16 KiB，parameters 为最多 64 KiB 的 JSON object，省略时为 `{}`；仍受总请求体额度约束 |
| `tool_choice` | `auto`（默认）、`none`、`required`，或 `{ "type":"function", "function":{"name":"..."} }`；required 必须提供工具，指定函数必须存在 |
| `parallel_tool_calls` | boolean，默认 true；false 时最多返回一个调用；这是输出约束，不控制 GPU 调度并发 |
| `stream` | boolean，默认 false |
| `stream_options.include_usage` | boolean，仅允许与 stream=true 同用；默认 false |
| `max_tokens` / `max_completion_tokens` | 正整数，二选一；默认 min(512, 服务生成上限)，prompt + 生成额度必须放入 context |
| `temperature` | 可表示为 float 的非负有限数，正数转换后不能为 0；0 为 greedy，默认 0.6 |
| `top_p`、`top_k` | top_p ∈ (0,1] 且转为 float 后不为 0，默认 0.95；top_k 为 0–INT32_MAX 整数，默认 20 |
| `seed` | 0–UINT64_MAX 整数；每个请求独立 RNG |
| `repetition_penalty` | 可表示为 float 的正有限数，转换后不能为 0；默认 1 |
| `presence_penalty`、`frequency_penalty` | 可表示为 float 的有限数，默认 0 |
| `n` | 仅支持 1 |
| `enable_thinking` | boolean，默认 false；控制可信模型模板的 thinking 开关 |
| `max_thinking_tokens` / `thinking_budget` / `thinking_budget_tokens` | 三选一；-1 禁用、0 关闭 thinking、正整数限定 reasoning 内容 token；未显式提供开关时正预算开启 thinking |

未知字段、重复键、错误类型、非法 Unicode、非有限数、额外 JSON 文本均拒绝。`stop` 字符串、媒体、logprobs、response_format 和请求侧模板覆盖尚未支持。模板优先使用 `tokenizer_config.json` 中的可信 `chat_template`；缺少时 Qwen3 使用固定 ChatML fallback，带工具或工具历史时使用工具感知版本，Hybrid 使用项目已有 `qwen35-fixed-v20` 模板。HTTP 不读取独立 `chat_template.jinja`，也不接受环境变量或请求参数覆盖模板；因此不承诺与该独立文件渲染的 prompt token 相同。检查点自带模板负责渲染相应工具能力；服务不尝试改写任意 Jinja 模板的语义。

工具定义传入可信模板，服务返回调用给客户端处理。例如：

```json
{
  "model": "qwen3",
  "messages": [{"role": "user", "content": "查询北京天气"}],
  "tools": [{"type": "function", "function": {
    "name": "weather",
    "description": "查询城市天气",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
  }}],
  "tool_choice": {"type": "function", "function": {"name": "weather"}},
  "parallel_tool_calls": false,
  "max_tokens": 192,
  "temperature": 0
}
```

支持 Qwen JSON 风格 `<tool_call>{"name":"weather","arguments":{"city":"北京"}}</tool_call>` 和已有 XML 函数/参数形式。JSON 形式保留参数对象的原始字节，包括高精度小数和负零；XML 沿用公共解析器的值转换规则（普通标量是字符串，JSON object/array/bool/null 等保留 JSON 值）。服务校验完整块、严格 JSON object 参数、声明的函数、指定函数与调用个数；不会校验参数是否满足 JSON Schema，也没有约束解码。`function.strict=true` 返回 400，`strict=false` 可以省略或显式提供。

响应 `message.tool_calls` 包含响应内稳定的 `id`、`type` 和 `function:{name,arguments}`；arguments 是 JSON 字符串。只有调用且无可见文本时 `content:null`，reasoning 仍独立。正常调用使用 `finish_reason:"tool_calls"`；已到生成上限且调用完整时保留 `length`。没有工具或选择 none 时沿用普通 tokenizer 解码与文本输出，解码后可见的意外工具块作为文本保留。Server 不执行函数。

启用工具时，HTTP 解码保留 tokenizer 中独立的 `<tool_call>` / `</tool_call>` token，也接受与普通 BPE 标签混合的输出。该处理只作用于工具 HTTP 请求；公共 Tokenizer 和旧 ABI 的解码规则保持原样。排查格式失败可在启动 Server 前设置 `AILA_SERVER_TOOL_DIAGNOSTICS=1`：WARN 日志包含请求 ID、具体原因及最多 4096 字节的 JSON 转义工具块，可能包含模型生成的参数内容，默认关闭。

SSE 在完整工具块验证后发布调用；较长参数再按最多 1024 个原始 UTF-8 字节拆分，保证每帧不超过 HTTP 的 16 KiB 交付上限。首个工具 delta 带 index/id/type/name，后续同 index 只带 arguments 片段，客户端按 index 拼接。所有帧计入既有序列化输出预算。未闭合或损坏的块、未声明函数返回 502 `invalid_tool_call`；未满足 required/指定函数或违反 parallel=false 返回 `tool_choice_violation`；超过 128 次调用返回 `tool_call_limit_exceeded`。后续错误仍走唯一的 error + `[DONE]` 终态，客户端应将该请求视为失败。

下一轮把 assistant 的 tool_calls 与每个调用对应的 `role:"tool"` 消息放回 messages。工具结果可以按任意顺序到达，但不能缺失、重复或引用不存在的调用；可选 name 必须匹配调用名称。历史函数不必出现在本轮 tools 中，允许工具集合变化。

thinking budget 的标记来自可信 tokenizer，结束序列计入总输出额度与 usage。剩余空间放不下完整结束序列时，可以提前以 `length` 完成；具体强制关闭、RNG 和 KV 语义见 [请求级预算](thinking-budget.md)。

Hybrid 固定模板沿用项目的 `preserve_thinking=true` 默认值，保留并规范化 assistant 历史中的 inline 或独立 reasoning；本轮未实现历史思维块自动剥离。`<|think_on|>` / `<|think_off|>` 文本开关可覆盖 `enable_thinking`，服务根据渲染后的实际后缀判断是否已进入 reasoning。0.8B 与 4B 的 HTTP 默认均显式关闭 thinking，开启后 reasoning token 同样计入生成额度。结束 token 从 tokenizer 获取并去重，不复用 Qwen3 的硬编码 ID；本地 Hybrid 检查点同时使用 `<|im_end|>`（248046）和 `<|endoftext|>`（248044）。具有 vision 元数据的模型仍只开放文本请求。

usage 的 `prompt_tokens` 包括完整模板化输入，`completion_tokens` 来自 Worker 的生成摘要，包含输出控制 token 和 reasoning token；它不等于文本字符数。正常生成因长度结束且最后 token 只构成合法 UTF-8 字符的前缀时，用一个 U+FFFD 收尾；非法 UTF-8、StopToken 时不完整字符及异常断流返回错误。

未开始 SSE 时，错误使用非 2xx JSON：例如 400 参数错误、404 模型名错误、413 body 超额、415 类型错误、429 请求槽占满、503 Worker 不可用。SSE 已开始后，Worker/输出解析失败发送 `data: {"error":...}` 与 `[DONE]`，没有命名的 `event: error`，也不伪造成功 finish_reason。客户端断开、持续慢读、总请求期限或服务关闭直接终止连接；此类客户端可能收不到最终事件。HTTP parser 自身的错误响应格式由 Drogon 决定。

## 并发、背压与生命周期

两个 HTTP IO loop 只处理 socket、少量 host 状态和计时器。固定大小的 job pool 负责 tokenization、创建 Session、提交请求与读取 token。IO loop 不执行模型推理、不等待 IPC RPC。

每个请求只有一个最多 16 KiB 的跨线程待交付帧，生产线程等它被接受后才读取下一条 IPC token。IO loop 用 Trantor `pendingWriteBytes()` 读取实际待写队列，检查下一帧的 HTTP chunk 开销，并预留 5 字节结束块。超过门限时暂停消费；持续超过 stall 时间则取消该请求。下游停止读取会经 IPC 的累计 ACK/credit 回传压力；其他请求拥有各自的通道状态和额度。

每条连接仅服务一个应用请求，所有响应声明 `Connection: close`。同连接复用、同包或延迟到达的第二个请求会关闭该连接。`Expect: 100-continue` 支持正常传入单个请求。Drogon 1.9.11 内部 close=true 会跳过 async stream callback，因此 SSE 使用应用持有的关闭路径：写完结束块后 half-close，等待连接关闭或最多 2 秒 drain。SSE 请求槽在 job 与 IO 清理都完成后归还。

总请求期限拥有独立 IO 计时器，CreateSession/Submit 的 RPC timeout 使用剩余期限；单次等待超时不主动杀掉共享 Worker。Session 异步释放后可能短暂占用容量，创建新 Session 遇到 ResourceLimit 时允许 2 秒重试窗口；单次 RPC 仍受总请求期限约束。取消与最终 JSON / `[DONE]` 接纳使用同一把锁，避免成功终态与取消同时被记录。

关闭顺序是停止准入、取消请求/关闭连接、等待 Worker Shutdown、join 请求线程池、确认请求和连接集合排空，再退出 IO loop。Worker 故障使未完成请求失败，后续推理返回 503；当前不会自动重启或自动重试部分输出。

## 依赖与额度边界

固定依赖为 Drogon 1.9.11、Trantor 1.5.24、JsonCpp 1.9.6、zlib 1.3.1；框架静态链接、CRT 使用 `/MD`，关闭 TLS/DNS/ORM。构建锁记录源 archive SHA512、补丁 SHA256 与产物 SHA256。三处本地补丁分别转发 Windows listener 回调、提供 IO-loop-only 待写字节查询，以及在写回调分发前重新检查写事件是否仍启用；源码和哈希随仓库保存。

`pendingWriteBytes()` 对 memory/async 节点求和；遇到未支持的 TLS、文件、同步流节点返回 SIZE_MAX，由服务关闭连接。它不把 bytesSent 当作客户端消费确认。Channel 补丁处理同批事件中读回调先禁用写入的情况，避免继续调用陈旧的写就绪回调；有效写事件仍正常分发。构建另使用保留原始输出字节的 `/showIncludes` 探针，修正本机 CMake/MSVC 输出编码导致的头文件依赖丢失；脚本检查 Ninja 依赖有效。

HTTP body 限制为 64 KiB，连接上限 128。正常输出以 `maxOutputBytes−2048` 分别限制原始生成文本和序列化输出，预留 2048 字节用于错误收尾；SSE 累计预算包括 `[DONE]`。输出预算、单帧额度和 Trantor 额度是各自的 payload 限制，**不是总 RSS 或显存上限**。它们不覆盖内核 socket buffer、Drogon 全部 header/parser 内存、allocator、权重、共享 scratch 和 oneDNN/driver 分配。非流式 JSON 有单独的输出预算，不使用 SSE 的 pending 门限。单独限制连接数并不能证明全部 HTTP 内存已精确有界。

远端 Hybrid 固定关闭 recurrent snapshots，必要 host mirror 仍保留。IPC 可协商 `session_context_v1` 执行 reset/truncate；Hybrid 没有可用 checkpoint 时 truncate 返回实际恢复长度0，调用方须重新补算上下文。`sessionReservationBytes` / `cacheBytes` 包含 Full KV、递归矩阵和卷积状态的 device payload；host mirror 受Session数及独立的 `--max-session-host-bytes` 限制，未纳入device字节额度，host上限为0时不启用字节限制。进程内 Hybrid 默认4个快照的API不受此设置影响。

本服务当前用于本机 HTTP/1.1 文本推理；鉴权、TLS、HTTP/2、代理环境、持久连接、远端媒体输入、完整内存准入和长期压力/性能分布仍需后续实现或验收。新增 decode 合批先完成独立数值与功能边界检查，再安排集中压力测试。

## Socket 验收

本轮新增 `tests/server/batching_socket_tests.py`，最终 CPU Fake Worker HTTP 复核已通过默认、显式 1、2、4 配置、健康报告、独立请求语义及非法参数检查。最终主 CPU 组 20 项（含旧 ABI）、Server CPU 组 4 项、trace 38 项、policy trace 6 项及 MSVC SDK 检查均通过。Fake 后端不具备原生合批执行器，该脚本验证配置与回退边界。既有 socket runner 新增 `--decode-batch-size`，可在限定用例中验证合批开启时的服务行为。

底层四种模型/权重配置已通过 288 行完整 logits、64,576,512 个值的对照，最大绝对误差和归一化 RMS 误差均为 0；冻结 B1 对照每种配置的每侧 12 个请求、96 个生成 token 完全一致。`4→3→2→1` 动态缩批、合计 28 行独立 Request 重放，以及前向后 pinned 额度拒绝时整批 reset、不发布部分结果、唯一终态和非成员恢复均通过。具体模型和证据见 [连续 decode 合批](continuous-batching.md)。

真实 Server/Worker 的四模型 HTTP 各通过 7 项功能用例，合计 28 项。上限 1 与 4 下单请求的 JSON/SSE 结果一致，1/2/4 并发、断连请求取消与同组请求完成、请求槽复用，以及活动请求关闭的唯一保留 outcome 均通过；两个 Worker 的 trace 已分别关联到参考阶段和上限 4 的功能阶段。

HTTP trace 在 Qwen3 0.6B NF4、Qwen3.5 0.8B NF4、Hybrid BF16 文本夹具上均观察到执行批宽 2、3、4；4B NF4 本轮为 2、3。4B 已通过四并发用例及上限 4 配置，其原生 B4 由后端和 Request 用例验证，本轮 HTTP 未观察到 B4。有限功能验证不提供吞吐、尾延迟或长期稳定性结论。

预热后的性能与生命周期测量使用 `tools/server/service_benchmark.py`。该工具为每个 prompt/seed 先建立独立单请求参考，再测 1/2/4 并发的短请求与长短混合请求；保存原始 SSE、逐请求时刻、health、进程内存/句柄采样和最终报告。示例：

```powershell
python tools/server/service_benchmark.py --server ./build-refactor-server/AilaServer.exe --worker ./build-refactor/AilaWorker.exe --model ./models/qwen3-0.6B-bnb-nf4-offline --output ./tmp/refactor/service-q3 --mode probe
```

默认 probe 只生成两条短串行请求，不提供性能或长期稳定性结论。它限制 context≤512、生成≤16 token、最多60秒，并先经过 `gpu_run_guard.py` 的系统故障检查；存在未审查证据时阻断启动。只有调查与低压力验证完成后才使用 `--allow-stress --mode benchmark|lifecycle|both`，此开关不会清除已有故障。压力模式保留 1/2/4 和 mixed 场景，默认整轮300秒，`--max-run-seconds`可设置明确上限；超过期限请求关闭服务，仍未结束则仅终止本次 Job。

runner 默认限制显式 device USM 为 probe 4 GiB / stress 8 GiB、pinned host 16 MiB、Session host payload 256 MiB，可用相应 `--max-*-bytes` 正整数覆盖。这些不是总显存/RSS 限制。机器固定目录的进程锁、启动/结束状态和系统证据阻止未收尾或故障后的自动重跑，不能控制其他不使用该守卫的 GPU 程序，也不能保证防止系统 bugcheck。JSONL fsync、原子 JSON 和 active-batch 记录纳入观测成本；不得把新日志路径测量与旧 runner 数据直接当作等价基准。

WER旧报告的重复投递现在按完整ReportId/payload和关联转储扫描信息识别，原始事件仍保存。新事故、扫描变化、已有待审失败和boot变化继续阻断；规则及CPU验证见 [GPU运行守卫](gpu-run-guard.md)。

收尾先关闭请求/采样并清理 owned Job，再关闭日志、校验报告并持久写入 `result.json`，最后提交 guard 的 `finished`。单个清理或诊断异常不会跳过 Job 与守卫锁释放；结果和 guard 同时写盘失败会保留未完成状态。验收必须同时满足 runner 退出码 0、结果 `passed=true`，以及 guard journal 中同一 `gpu_guard_run_id` 的 `finished` 记录与 `details.result_sha256` 匹配；仅一个提前落盘的 `passed=true` 不是验收凭据。成功提交后不再进行必需写盘；此后的锁释放异常通过终端 `post_commit_errors` 报告，不改写已完成的 GPU 验证和结果哈希。

失败收尾可能在guard记录提交前hash后，继续把 `result.json` 改为 `passed=false` 并加入诊断；失败guard的hash因此不必等于最终失败文件。分析时保留二者，核对失败阶段及实际写入顺序，不把重构出的提交前内容当作成功验收。

每次输出到唯一子目录；基线参考和 warmup 不计入性能统计。所有完成请求都校验 content/reasoning、usage 和结束原因；生命周期阶段交替完成与断连，默认 512 请求，必须越过 256 个 retained request handle 的边界。可用 `--max-private-growth-mib`、`--max-handle-growth` 设置明确的空闲期增长门限。

跨独立进程的性能汇总使用 `tools/server/service_baseline.py`。它校验 acceptance/守卫/原始记录的哈希、时刻和覆盖，分别统计 mixed 单并发的短、长输入，并要求每后端至少三轮。复现参数、观测成本和初步筛查边界见 [服务基线](service-baselines.md)。

TTFT 从客户端开始请求到首个非空 content/reasoning 事件，排除 role 事件；吞吐以确认的 `completion_tokens` 计算。SSE 内容事件可能拆分或合并 token，所以事件间隔单独记录，不能称作逐 token 延迟。有限轮数的 p95/p99 不构成服务尾延迟保证。工具的 Windows Job 仅管理本次启动的 Server/Worker，最终检查正常关闭、Job 为空、逻辑计数回收与可观察的错误日志。

Python runner 仅依赖标准库，持有自己的子进程句柄和唯一 shutdown 文件，保存原始 HTTP/SSE、启动参数、日志、health 样本及最终 report；框架 ERROR/FATAL 日志也使验收失败。CPU CTest 另包括真实 loopback Channel 的读写事件回归。先运行 CPU fake 组，再独占 GPU 运行真实模型组：

```powershell
python tests/server/socket_tests.py --server ./build-refactor-server/AilaServer.exe --worker ./build-refactor/AilaFakeInferenceWorker.exe --mode fake --output ./tmp/refactor/server-sockets-fake
python tests/server/socket_tests.py --server ./build-refactor-server/AilaServer.exe --worker ./build-refactor/AilaWorker.exe --model ./models/qwen3-0.6B-bnb-nf4-offline --mode gpu --output ./tmp/refactor/server-sockets-gpu
python tests/server/socket_tests.py --server ./build-refactor-server/AilaServer.exe --worker ./build-refactor/AilaWorker.exe --model ./models/qwen3.5-0.8B-bnb-nf4-offline --backend qwen35_nf4 --mode gpu --output ./tmp/refactor/server-sockets-hybrid-gpu
python tests/server/socket_tests.py --server ./build-refactor-server/AilaServer.exe --worker ./build-refactor/AilaWorker.exe --model ./models/Qwen3.5-0.8B --backend qwen35_bf16 --mode gpu --output ./tmp/refactor/server-sockets-hybrid-bf16-gpu
```

Fake 模式自动创建专用 tokenizer，用每 token 约 4 KiB 的文本稳定触发 socket 背压，并单独启动故障 Worker 验证失败终态。GPU 组验证真实推理的 JSON/SSE 一致性、1/2/4 并发独立 seed、连接行为、断连隔离和活动请求关闭。短文本 GPU 请求不保证填满网络缓冲区，因此慢读门限由 fake socket 组验收。记录的冷启动/JIT 与端到端用例耗时不是服务性能基准。

工具调用另有专用 socket 组，覆盖请求校验、JSON/XML 调用、结果历史、并发隔离、失败后恢复、长参数 SSE 分片以及 none/无工具兼容：

```powershell
python tests/server/tool_socket_tests.py --server ./build-refactor-server/AilaServer.exe --worker ./build-refactor/AilaFakeInferenceWorker.exe --mode fake --output ./tmp/refactor/server-tool-sockets-fake
python tests/server/tool_socket_tests.py --server ./build-refactor-server/AilaServer.exe --worker ./build-refactor/AilaFakeInferenceWorker.exe --mode fake --backend qwen35_nf4 --output ./tmp/refactor/server-tool-sockets-hybrid-fake
```

Fake 工具夹具同时覆盖独立标签 token 和普通 BPE 标签；真实模型组验证实际生成的调用与单请求参考，不能要求模型稳定生成专门的损坏输出。Qwen3.5 的 GPU 用例明确要求完整 XML 格式：0.8B NF4 在原较短请求中漏生成 `<function=weather>`，逐 token 记录与独立词表解码确认该错误来自模型输出，服务正确返回 502；原失败、格式明确后成功的诊断均保留。因此这些用例验收协议、解码与会话隔离，不代表任意自然语言请求的工具调用成功率。GPU 执行由外层机器守卫、Windows Job 和 watchdog 串行管理，独立旧 Server 的普通文本对照与工具验证分别留档。
