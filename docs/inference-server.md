# C++ Qwen3 / Qwen3.5 Hybrid HTTP/SSE 服务

`AilaServer` 是 Windows x64、普通 MSVC C++17 的本地文本服务，通过 [InferenceClient / IPC v2](inference-ipc-v2-implementation.md) 启动独立 `AilaWorker.exe`。Worker 加载一份 Qwen3 或 Qwen3.5 Hybrid NF4 权重，最多服务 4 个独立 BF16 KV Session；Hybrid 的 DeltaNet 递归矩阵、卷积历史和 host mirror 也归各 Session。每次 HTTP 请求创建新 Session，多轮聊天通过 `messages` 传入完整历史。

服务固定监听 `127.0.0.1`，支持 HTTP/1.1。宿主不链接 AilaLib、SYCL 或 oneDNN；设备运行时留在 Worker 进程中。并发请求由已有 Scheduler 交替推进，当前没有连续合批。实际验收范围见 [验证记录](refactor-validation.md) 第 11–12 节。

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

加载完成后，stdout 输出包含 `event=ready`、端口、模型名、backend 和 Worker epoch 的 JSON；`/health` 与最终 report 也提供 backend。后端与模型配置不匹配时启动失败，不自动切换。Ctrl+C 停止接入新请求、取消活动请求、关闭 socket，并等待 Worker 退出。也可指定 `--shutdown-file PATH`，由受控启动程序创建该文件触发关闭；`--report PATH` 将最终状态写成 JSON。没有 HTTP 关闭端点。

主要选项：

| 选项 | 默认值 | 约束 |
| --- | ---: | --- |
| `--backend` | `qwen3_nf4` | `qwen3_nf4` 或 `qwen35_nf4`；一个进程加载一个模型 |
| `--model-id` | `qwen3` / `qwen3.5` | 按后端选择默认值；可覆盖为 1–128 个 ASCII 模型名字符 |
| `--port` | 8080 | 1–65535，固定 loopback |
| `--context` | 4096 | 32–4096 token |
| `--sessions` | 4 | 1–4 个同时占用的请求槽 |
| `--max-new-tokens` | 1024 | 1–1024，超过 context 时缩至 context−1 |
| `--max-output-bytes` | 262144 | 4096–4194304，原始文本和序列化输出分别受限；预留 2048 字节错误收尾 |
| `--stream-buffer-bytes` | 65536 | 32768–1048576，Trantor 待写数据门限 |
| `--stream-stall-ms` | 5000 | 100–60000，持续无法接纳下一帧的时间 |
| `--request-timeout-ms` | 120000 | 1000–600000，从 HTTP 请求接入开始计算 |

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

需要流式输出时，增加 `"stream":true`；增加 `"stream_options":{"include_usage":true}` 可在结束前获得 usage 帧。每条 SSE 记录使用 `data: <JSON>\n\n`，最后为 `data: [DONE]\n\n`。先发送 assistant role，再发送 `content` / `reasoning_content` 增量；正常终态 `finish_reason` 为 `stop` 或 `length`。终态之后仅发送可选 usage 和 `[DONE]`。启用 include_usage 时，普通 chunk 带 `usage:null`，最终 usage chunk 的 `choices` 为空数组；未启用时省略 usage 字段。

支持字段：

| 字段 | 语义 |
| --- | --- |
| `model`、`messages` | 必填；model 必须等于 `--model-id`，messages 为 1–256 条文本消息 |
| 消息 `role` | `system`、`developer`、`user`、`assistant` |
| 消息 `content` | 字符串，或含 1–256 个 `{ "type":"text", "text":"..." }` 的数组；文本段顺序拼接 |
| 消息 `name`、`reasoning_content` | name 为 1–64 字节字符串；reasoning_content 仅允许 assistant 消息 |
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

未知字段、重复键、错误类型、非法 Unicode、非有限数、额外 JSON 文本均拒绝。`stop` 字符串、工具调用、媒体、logprobs、response_format、thinking budget 和请求侧模板覆盖尚未支持。模板优先使用 `tokenizer_config.json` 中的可信 `chat_template`；缺少时 Qwen3 使用固定 ChatML fallback，Hybrid 使用项目已有 `qwen35-fixed-v20` 模板。HTTP 不读取独立 `chat_template.jinja`，也不接受环境变量或请求参数覆盖模板；因此不承诺与该独立文件渲染的 prompt token 相同。模板引擎和 reasoning 解析复用现有聊天模块，模型意外输出的 tool 块按普通文本保留，不执行工具。

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

远端 Hybrid 固定关闭 recurrent snapshots，因为 v2 尚未提供回滚接口；必要 host mirror 仍保留。`sessionReservationBytes` / `cacheBytes` 包含 Full KV、递归矩阵和卷积状态的 device payload，host mirror 由最多 4 Session 限制，未纳入 v2 的 device 字节额度。进程内 Hybrid 默认 4 个快照的 API 不受此设置影响。

本服务当前用于本机 HTTP/1.1 文本推理；鉴权、TLS、HTTP/2、代理环境、持久连接、Qwen3.5 远端、连续合批、完整内存准入和长期压力/性能分布仍需后续实现或验收。

## Socket 验收

Python runner 仅依赖标准库，持有自己的子进程句柄和唯一 shutdown 文件，保存原始 HTTP/SSE、启动参数、日志、health 样本及最终 report；框架 ERROR/FATAL 日志也使验收失败。CPU CTest 另包括真实 loopback Channel 的读写事件回归。先运行 CPU fake 组，再独占 GPU 运行真实模型组：

```powershell
python tests/server/socket_tests.py --server ./build-refactor-server/AilaServer.exe --worker ./build-refactor/AilaFakeInferenceWorker.exe --mode fake --output ./tmp/refactor/server-sockets-fake
python tests/server/socket_tests.py --server ./build-refactor-server/AilaServer.exe --worker ./build-refactor/AilaWorker.exe --model ./models/qwen3-0.6B-bnb-nf4-offline --mode gpu --output ./tmp/refactor/server-sockets-gpu
python tests/server/socket_tests.py --server ./build-refactor-server/AilaServer.exe --worker ./build-refactor/AilaWorker.exe --model ./models/qwen3.5-0.8B-bnb-nf4-offline --backend qwen35_nf4 --mode gpu --output ./tmp/refactor/server-sockets-hybrid-gpu
```

Fake 模式自动创建专用 tokenizer，用每 token 约 4 KiB 的文本稳定触发 socket 背压，并单独启动故障 Worker 验证失败终态。GPU 组验证真实推理的 JSON/SSE 一致性、1/2/4 并发独立 seed、连接行为、断连隔离和活动请求关闭。短文本 GPU 请求不保证填满网络缓冲区，因此慢读门限由 fake socket 组验收。记录的冷启动/JIT 与端到端用例耗时不是服务性能基准。
