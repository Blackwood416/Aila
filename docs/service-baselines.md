# 服务性能基线与生命周期测量

`tools/server/service_benchmark.py` 测量完整 HTTP/SSE 服务。`tools/server/service_baseline.py` 离线汇总已通过审查的独立运行，校验参考输出、原始时刻、测量覆盖和文件身份；它不启动 Server、Worker 或 GPU，也不确认故障或重试。

## 固定工作负载

同一后端至少运行三个独立 Server/Worker 进程，每轮结束并核验机器守卫后再启动下一轮。参数、软件、模型文件清单、环境与 boot 身份必须一致。多个后端可交错测量，但 GPU 进程必须串行。

本轮 P6 使用以下配置，详细证据在 `tmp/refactor/p6-service-matrix-20260911`：

| 项目 | 配置 |
| --- | --- |
| 后端 | Qwen3 0.6B NF4、Qwen3.5 0.8B NF4、Qwen3.5 0.8B BF16 纯文本 fixture |
| context / Session 数 | 2048 / 4 |
| benchmark | short、mixed；1/2/4 并发；每组合预热 2 轮、正式 20 轮 |
| 输出 | 最多 64 token；保留实际生成量和结束原因 |
| 长输入 | `--long-prompt-repeat 90`；实际模板化后 Qwen3 为 1734 token，Hybrid 为 1825 token；短输入均为 27 token |
| 采样 | seed 42 加 lane 编号，temperature 0，thinking 关闭 |
| 生命周期 | 四并发、512 次正式请求；完成/首内容后断连交替，取消目标最多 128 token |
| 额度 | 显式 device USM 8 GiB、pinned host 16 MiB、Session host payload 256 MiB |
| watchdog | benchmark 每进程 600 秒；lifecycle 每进程 900 秒 |
| 生命周期增长门限 | 每进程预热后至最终空闲：private bytes 增长不超过 128 MiB，句柄增长不超过 64 |

每个 benchmark 有 8 次独立参考、28 次预热和 280 次正式请求，共 316 次。每个 lifecycle 有 8 次参考、8 次预热和 512 次正式请求，共 528 次。参考和预热不计入性能样本；其失败仍使整轮失败。每个完成请求必须与相同 payload 的独立参考逐字段一致，包括 content/reasoning 摘要、usage 和结束原因。

mixed 的单并发 20 轮交替两种输入，分开后每种只有 10 个批次样本；其余组合每轮有 20 个批次样本。报告保留实际 count，不把不同输入混在一起增加样本量。temperature=0 的本轮基线只验收 greedy 服务输出，采样 RNG 的隔离由其他固定 seed 用例验证。

单后端 benchmark 示例，`--output` 是父目录，runner 创建唯一运行子目录：

```powershell
python tools/server/service_benchmark.py --server build-refactor-server/AilaServer.exe --worker build-refactor/AilaWorker.exe --model models/qwen3-0.6B-bnb-nf4-offline --backend qwen3_nf4 --output tmp/refactor/service-q3-run1 --mode benchmark --allow-stress --context 2048 --sessions 4 --concurrency 1 2 4 --profiles short mixed --rounds 20 --warmup 2 --new-tokens 64 --long-prompt-repeat 90 --max-run-seconds 600 --max-device-bytes 8589934592 --max-pinned-host-bytes 16777216 --max-session-host-bytes 268435456
```

`--decode-batch-size` 默认 1，可设 1..4 且不得超过 `--sessions`；它直接传给 AilaServer，使同一进程可按该上限执行合批 decode。runner 会核对 health、ready 与最终 report 回报的 `max_decode_batch_size` 与请求值一致，因此漏传或错值不会静默退回默认单请求路径。该字段是配置上限，不是实际批宽；实际执行的批宽须由执行 trace 的 `decode_batch` 记录单独证明，不能由吞吐提高反推。

`--allow-stress` 不清除机器守卫的故障记录。退出非零、超时、设备丢失或新增系统证据后先审查并保留原始结果；不把失败轮从样本中静默剔除后继续重试。

## 离线汇总

输入为审查后保存的 acceptance JSON。每份记录绑定运行目录、`result.json` SHA256、守卫快照路径及 SHA256、run ID、后端和请求数；`artifacts` 至少绑定 `arguments.json`、`metadata.json`、`measurements.jsonl`、`batches.jsonl`、`final-report.json`。必须同时验证外层命令退出 0、日志完整、同轮守卫为 finished、结果哈希一致、Job 正常排空，以及前后系统故障快照没有增量。本轮使用证据目录中的 `audit-service-run.py` 在下一轮启动前保存这些记录。

三次 Qwen3 运行的汇总示例：

```powershell
python tools/server/service_baseline.py --acceptance tmp/refactor/p6-service-matrix-20260911/q3-benchmark-01-acceptance.json tmp/refactor/p6-service-matrix-20260911/q3-benchmark-02-acceptance.json tmp/refactor/p6-service-matrix-20260911/q3-benchmark-03-acceptance.json --output tmp/refactor/service-q3-baseline.json
```

可一次传入多个后端，每个后端仍须有至少三次独立进程。输出使用独占创建，不覆盖已有报告。哈希能检测已绑定文件的变化，不能替代记录来源的审查；本 runner 对权重只记录路径、大小及 mtime，模型内容身份须由另行保存的 checkpoint 清单绑定。

工具重新计算每组的 median、p10/p90、MAD、CV，保留每轮原始分布与实际输入/输出长度。mixed 的单并发轮交替使用短、长输入，两者分别统计；不能把不同输入的耗时差称为测量噪声。跨进程还检查测量参数、软件和环境身份、独立参考、分组覆盖一致，禁止重复计入同一运行。

初步筛查边界标为 **HEURISTIC**：以独立运行 median 的中位数为中心，吞吐下界减去两倍已观察到的 max−min 跨度并截断到 0，延迟上界加上两倍跨度。这个边界只触发审查，不是统计置信区间、通用回归百分比或自动放行规则。三轮样本不足以确定尾延迟保证。

## 指标含义与边界

- 吞吐为一个并发批次的实际 `completion_tokens` 总量除以首请求开始至末 `[DONE]` 的时长，包含请求准备和服务传输；它不等于 decode kernel 吞吐。
- TTFT 从客户端开始连接/发送至首个非空 content 或 reasoning 事件，排除 role。请求时长截至 `[DONE]`；EOF 单独记录。
- 内容事件间隔来自客户端读取观测。相邻事件可以共享一个读取时刻，因此 0 间隔合法；一个 SSE 事件也不必等于一个 token。这些分布不能称为逐 token 延迟。
- 长短混合同时保留“短请求先完成”与“短请求在所有长请求结束前出现首内容”。长输入可能更早遇到 EOS；完成顺序本身不能证明调度公平性，首内容观测也不构成最坏调度延迟保证。
- health、进程采样、原始 SSE 写盘和 JSONL 持久化在测量中保持开启，其观测成本包含在服务耗时内。执行 trace 关闭；开启 trace 的 probe 不参与此基线。
- Windows 工作集、private bytes 与进程句柄独立于 USM 账本。512 次生命周期跨过 IPC 的 256 个 retained request handle 边界，但仍是有限压力；不能证明长期无泄漏、近容量上下文稳定或驱动私有分配有界。

离线工具验证入口：

```powershell
python -m unittest discover -s tools/server -p test_service_baseline.py -v
```
