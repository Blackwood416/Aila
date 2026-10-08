# Qwen3.8-Flash-Next：真实 PI 与长上下文 Server

后续单流优化已将本机启动脚本更新为 chunk4096 / reserve3584，并修正 PLE 工作区按上下文容量预分配的问题。最新性能与配置见 [单流长文报告](qwen38-single-stream-performance-2026-10-07.md)；以下保留本阶段原始测量。

2026-10-07。本轮打通真实 PI coding agent 的工具回合、长上下文服务及跨 HTTP 请求的精确前缀复用。吞吐测量使用 16K 容量；最终 PI 长背景组合验证使用真实 32K 容量。长上下文使用 T1；MTP 仍默认关闭，尚未达到 Server decode 20+。

## 改动

- IPC 新增可选 `extended_context_v1`，将 prompt/附件/prefill chunk/总 token reservation 的可协商上限提高到 262144 / 1 MiB / 2048 / 1179648。旧 hello/default 保持 4096 tokens、16 KiB 附件和 512 chunk；与 `extended_generation_v1` 独立协商。
- Server 支持 `--context`、`--prefill-chunk-size`、`--max-body-bytes` 和最高 32768 的 `--max-new-tokens`。ChatCodec 及最终 summary 校验同步扩容，`/health` 和 `/v1/models` 报告实际加载的容量。HTTP body 与渲染后的 prompt 分别受同一有界 byte limit 约束。
- 兼容 PI 的 `store:false` 和开头的 `developer` 指令消息；后者归一为 checkpoint 模板接受的 `system`。`store:true` 仍明确拒绝。工具 schema/history/XML 输出继续走现有验证与 SSE 转换。
- `--reuse-sessions 1` 保留至多 `--sessions` 个空闲 Session；新 prompt 必须包含完整且逐 token 相等的已缓存序列，才能只提交后缀。不匹配则 Reset；失败或取消的请求不入缓存。无需请求方提供 session ID，不进行任意位置的 DeltaNet 回退。
- HTTP usage 的 prompt 数仍是完整提示长度，命中量另外通过 `prompt_tokens_details.cached_tokens` 报告。健康报告保留 bounded cache counters 和每请求命中量。
- Q4E 的中间 prefill chunk 及最终 pending token 在没有 logits 消费者时省略 LM head。

## 实测

[MEASURED] Intel Arc A770，驱动 32.0.101.8724，oneAPI 2026.1，Xeon E5-2696 v3，CPU experts 18 threads。模型为 `C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2`。服务容量 16384，1 Session，BF16 KV，MTP=0。

每个长提示生成 64 tokens；速率定义为 `(completion_tokens - 1) / (SSE done - first content)`，包含 IPC 和 finalization。一次探索性测量，不是统计稳定性或通用质量保证。

| chunk | 实际 prompt | 复用 tokens | 首字秒 | 首字后 tok/s |
|---:|---:|---:|---:|---:|
| 512 | 5924 | 0 | 98.791 | 3.680 |
| 512 | 6012 | 5988 | 1.479 | 9.516 |
| 2048 | 5924 | 0 | 41.260 | 7.888 |
| 2048 | 6012 | 5988 | 1.429 | 10.411 |
| 2048 | 13924 | 0 | 96.105 | 10.360 |
| 2048 | 14012 | 13988 | 1.326 | 10.126 |

512/2048 两组来自不同服务进程，后者在长提示前先运行过 PI；驱动/文件/专家缓存热度没有控制。因此不能把全部差值归因于 chunk，也不宣称 2.4 倍因果加速。2048 作为本机启动脚本的实测候选，通用 Server 默认仍是 512。约 6K/14K 的续接均约 10 tok/s，首字约 1.3–1.5 秒。

16K 的 Session device reservation 为 583606272 bytes（556.57 MiB）；host mirror 为 117669888 bytes（112.22 MiB）。模型加载时显式 device ledger 为 13044897010 bytes，不含随后增长的 prefill scratch，不等于进程实际峰值。空闲复用 Session 会保留其状态直至替换或服务关闭。

短 PI 实测完成两次 HTTP 请求：第一次生成工具调用 26 tokens，执行 read 后第二次回答 7 tokens，并命中 950 个缓存 tokens。不是用 stub 响应冒充模型结果；stub 仅用于前期确认 PI 请求字段。

**长背景真实 PI 组合验证通过。** 32K 服务下，第一轮实际输入 14853 tokens，输出工具调用 64 tokens；PI 的 JSON 事件确认恰好一次成功 read 指定的随机标记文件。第二轮 usage 显示 `input=28, cacheRead=14917, output=10`，最终准确返回标记。总 CLI 时延 167.815 秒，包含冷启动长 prefill 和两次请求；不把此总时延当作 decode benchmark。结果及完整事件位于 `tmp/q4e-agent-pi-long-32k/pi/aila-pi-smoke-tvi2rx/`，Server 正常退出。

原始数据：`tmp/q4e-agent-context-512-v2/result.json`、`tmp/q4e-agent-context-2048/result.json`。精确配置沿用 `qwen38-server-mtp-2026-10-07.json` 的 `worker_environment`，覆盖 `AILA_Q4E_SERVER_MTP=0`。

## 使用

仓库根目录启动服务（32K、2048 chunk、单会话、精确前缀复用）：

```powershell
pwsh -NoProfile -File tools/server/start-qwen38.ps1
```

另一个终端创建项目专用 PI 配置，避免覆盖已有用户配置：

```powershell
New-Item -ItemType Directory -Force tmp/pi-qwen38 | Out-Null
Copy-Item tools/server/pi-models.example.json tmp/pi-qwen38/models.json
Copy-Item tools/server/pi-settings.example.json tmp/pi-qwen38/settings.json
$env:PI_CODING_AGENT_DIR = (Resolve-Path tmp/pi-qwen38).Path
pi --provider aila --model qwen4exp --thinking off
```

示例配置的 `contextWindow`/`maxTokens` 与启动脚本相同。`apiKey` 是本地服务的占位字符串。PI 1.0.1 在客户端固定预留 4096 安全 tokens；16K 配置下，本轮 14850-token 长背景曾被客户端压成 1 个输出 token，工具调用未发生（证据 `tmp/q4e-agent-pi-long`）。因此 agent 示例使用真实 32K 容量，且将该模型的 compaction reserve/keep-recent 配为 4096/8192，避免默认 16384 reserve 过早压缩。这里没有修改用户安装的 PI 或虚报服务容量。自动验证：

```powershell
python tools/server/pi_agent_smoke.py --endpoint http://127.0.0.1:8080/v1 --keep
python tools/server/qwen38_context_probe.py --model C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2 --output tmp/new-context-run --chunk 2048
```

## 验证及边界

Protocol、Worker、Client CPU 测试以及 ServerChat 测试通过；覆盖扩展能力协商、实际 64K-token wire 附件、长消息、summary、PI 字段和缓存 usage。真实长请求均正常 SSE 结束并优雅回收服务。

协议最大值不是模型实测范围：本轮吞吐针对 16K 容量、PI 长背景闭环针对 32K 容量，最大实际输入约 14.9K。填满 32K 或更大的输入、真实复杂仓库任务、多个并发长会话尚未实测。精确前缀缓存不处理历史删改、thinking 剥离、工具参数重排或重分词造成的前缀变化，这些情况会重算。

长上下文 MTP 的 sparse draft attention 与既有 T1 对齐问题尚未解决；未直接去掉 2048 门槛。下一步应围绕真实 agent 热路径和长上下文 MTP 做有针对性的测量，避免大范围 kernel 搜索。
