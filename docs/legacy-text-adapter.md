# 旧文本 C ABI 的共享会话适配

Windows 上设置 `AILA_LEGACY_TEXT_V2=1`，可将 `aila_api.h` 的旧文本入口接入新的 Model / Session / Request。默认值为 0，待完整兼容与资源边界验收后再切换默认路径。此模式不改变旧 C 结构或导出符号。

支持纯文本 Qwen3 NF4、Qwen3.5 Hybrid NF4，以及无 vision/audio metadata 的 Hybrid BF16。带视觉或音频能力、量化 KV、Hybrid legacy linear、Diffusion、TTS、ASR 和检测模型继续使用现有 v1 驱动；开关关闭时所有模型使用原路径。具有 vision 配置的 BF16 模型即使只输入文字，也不会自动改成纯文本模型。

## 宿主与 Worker

`LegacyTextSession` 在宿主复用现有 Tokenizer、Jinja/ChatFormatter、ChatSessionState、reasoning/tool parser 和 tool policy。`LegacyInferenceBridge` 通过动态加载的公开 C ABI 调用 `AilaInference.dll`，不跨 MT/MD CRT 传递 C++ 对象、异常或容器。

`AilaInference.dll` 必须位于 `AilaShared.dll` 旁，Worker 和 GPU 依赖可独立放在 `AILA_RUNTIME_DLL_DIR` 指向的目录。加载采用明确路径和受限 DLL 搜索，并验证 ABI/build ID；不会改宿主 cwd、PATH 或 DLL 搜索目录。

同一规范化模型目录、Worker 目录、API 目录、backend、context capacity 和完整继承环境共享一个 Model owner / Worker。每个旧 Engine 拥有独立 Session 和聊天历史；Worker 的 Scheduler 交错推进请求。环境只用于内存中的分组键，不写日志或证据文件。最后一个 owner 关闭连接、等待 Worker 和 I/O 线程退出，释放句柄后卸载推理 DLL。

## 文本与上下文契约

- 支持原始文本、多轮历史、messages JSON、assistant JSON、普通流和结构化流、reset/context length。
- 请求级模板优先于当前 Win32 环境模板，保留原 template name。显式读取进程环境避免 MT DLL 使用过期的 CRT 环境副本。
- 原始文本历史保留 assistant content/tool calls，后续 prompt 不保存 reasoning；溢出时按原规则丢弃最早完整轮次。
- 将新 prompt 与已缓存 token 做最长公共前缀匹配，truncate 后使用返回的实际长度补算。远端 Hybrid 当前关闭快照，可能回退到 0。
- 新 `request_prefill` 仅执行输入，不采样、不输出 token、不消耗采样 RNG；用于长 prompt 分段和取消后的前缀恢复。`max_new_tokens=0` 的旧调用也走此路径。
- 保留旧 ABI 的 thinking 语义：预算 0 只关闭模板默认思考；正预算使用 host sampler，最多强制关闭一次。正常新 token API 的默认策略保持每次重新进入 thinking 都受预算约束。
- Hybrid 使用显式的 `legacy_prefill_logits_v1` 请求策略：旧后端在非初始 2–16 token 分块返回多行 logits，旧生成循环从第一行采样。适配器保留该行为；新 Session API/HTTP 继续默认取最后有效行，投影和 attention 计算不变。
- 回调在发起调用的线程执行。回调重入同一 Engine 返回 Busy 对应的旧错误；独立 Engine 可继续执行。文本流调用保留 Proxy owner，允许回调中销毁外层 Engine 后安全返回。
- 提前中止时取消并排空远端，回滚多执行的 token，恢复调用者接受的前缀。该行为使用独立前缀参考验收：旧 v1 回调路径可能抛异常而绕过原本的上下文对齐，不能用那个缺陷作为恢复长度标准。
- 保留 48 次连续相同 token 的循环保护；第 48 个计入上下文，流回调只交付前 47 个。

## 当前边界

每个共享 Model 最多 4 个 Session，IPC prompt attachment 最多 4096 token，prefill chunk 最多 512。adapter 按 context capacity 申请 `extended_generation_v1`，输出上限为 max(1024, min(capacity, 32768))；实际生成仍受剩余 context 和 32768 的总 token 预留限制。单个 Request 持续保存 RNG、penalty 和 thinking 状态，不通过重新提交来拼接长输出。长 prompt 可用无采样补算拆分，生成超过协商上限会在修改上下文前明确报错。第 5 个同配置 Engine 初始化仍被拒绝。旧 decode/stream chunk 参数在此路径上不再控制 Worker 的批次形状。

生成请求至少携带 1 个输入 token。按剩余 context 裁剪后的输出额度若仍达到 32768，adapter 在 truncate/replay 之前拒绝，保留原有 context；1 个输入 token 加 32767 个输出额度可以恰好占满预留。这里描述的是请求准入，未保证生成完整长度：EOS、取消等仍能提前结束。同 owner 的其他请求也占 token 额度，实际 submit 仍可能因当前共享预留不足而失败；该提前检查不会预占 credit。

可通过 `AILA_LEGACY_TEXT_MAX_SESSIONS`（1–4）降低共享会话数；`AILA_LEGACY_TEXT_MAX_SESSION_CACHE_BYTES` 限制全部 Session 的 device cache/state 预留。`AILA_LEGACY_TEXT_MAX_DEVICE_BYTES`、`AILA_LEGACY_TEXT_MAX_PINNED_HOST_BYTES`、`AILA_LEGACY_TEXT_MAX_SESSION_HOST_BYTES` 接通已有 `allocation_limits_v1`，约束显式 device USM、pinned host USM 和 Session host payload。字节额度默认 0，不增加上限；设置任意后三项为非零会要求 Worker 确认该能力和完整额度。

这五项必须使用十进制数字，不接受符号、空白、单位或溢出值；初始化返回 InvalidArgument 并包含变量名。每个 owner 从与共享键相同的 Win32 环境快照读取一次，避免 DLL 的静态 CRT 环境过期。额度属于当前共享 owner，不代表不同配置 Worker 的合计显存，也不覆盖库/驱动私有分配。Session 准入拒绝不会新开 Worker；显式分配失败仍沿用 Worker 的局部失败、受影响 Session reset 和 peer 隔离语义。

宿主 tokenizer/模板日志和 v2 Worker 日志经现有异步日志分发器交付。每个共享 Model owner 有一个日志读取线程，Worker 日志只选择一个仍存活的 Engine source 转发，避免四个 Engine 重复交付同一日志；原 source 退休后选择剩余 Engine。日志不具备 Request 身份，共享加载只记录一次，因此不保证逐条等同四个独立旧 Worker。`aila_set_log_level` 在加载前及运行中同步到共享 Worker；已有 source 退休与回调等待规则继续管理用户数据寿命，日志读取线程在 DLL 卸载前 join。

IPC 和宿主日志队列均有界，慢回调只影响日志交付，溢出允许丢弃；既有异步分发器上限为 256 项。Worker 回调不执行用户代码或写 pipe。模型、模板、工具策略和媒体错误保持可区分，纯文本模型的媒体拒绝不会破坏已有 Session。

这些限制、完整媒体驱动迁移、旧分块性能、最坏取消延迟与近容量/长期压力仍属于后续工作。四会话并发不等同于 P7 连续合批。

## 验证入口

```powershell
ctest --test-dir build-refactor -R '^AilaLegacyText(Session|Api)Tests$' --output-on-failure
```

CPU seam 故意让后端在第一个回调前执行全部输出，验证精确截断、回退到 0、粗粒度 checkpoint 的恢复，以及异常、reasoning、工具/流事件、模板环境、短 V2 结构和历史溢出。公开 ctypes 测试在隔离的 host/runtime 目录中使用经过哈希校验的 CPU Worker，验证真实管道、四 Engine 共享一个进程、槽位复用、回调线程/重入/销毁、DLL 卸载和有限生命周期句柄计量。

真实 GPU 比较使用修改前完整独立 Worker/Proxy 产物，逐项比较文本、JSON、流事件、上下文和旧串行/新并发结果；取消另作 accepted-prefix 对照。运行必须使用机器守卫、Windows Job 和 watchdog。证据及实际通过范围见 [验证记录](refactor-validation.md)。
