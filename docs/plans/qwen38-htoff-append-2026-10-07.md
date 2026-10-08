# Qwen3.8-Flash-Next：关闭超线程后的追加阶段实验

2026-10-07，Arc A770 16 GiB / driver 32.0.101.8724 / oneAPI 2026.1，Xeon E5-2696 v3 已确认为 18 核 / 18 逻辑处理器。目标仍是单流长上下文 agent；本轮没有接入并发、MTP 或 KV 精度变化。

结论：本轮没有得到可推广的稳定加速，未达到 20+。候选全部归档撤回，原来的单流启动配置和此前通过的 Server/PI 改动保留。

## 测量口径

32K 容量、BF16 KV、MTP off、CPU experts 18 线程、exact-prefix reuse。普通配置 chunk4096 / reserve3584 MiB；完整环境、各实验 overrides、二进制 hash、输出 hash 和实测 token 数见同名 JSON。每进程依次执行短 warmup、约 6K prompt、续轮、约 14K prompt、续轮，生成上限为 96 tokens。实际新长 prompt 为 5924 / 13924 tokens，续轮复用 6020 / 14020 tokens，只追加 24 tokens。

TTFT 是请求到首个内容片段，首字后速率为 `(completion_tokens-1)/(SSE done-first content)`。部分 6K 续轮提前 EOS，不能比较输出长度不同的整轮耗时。各项是高效初筛的单次样本，不是统计显著的算子加速结论；缓存变化会改变 CPU/GPU 专家 placement 和生成轨迹。

Windows 文件缓存未清理，重启后的首次 6K prompt 明显更冷。不能把基线 59.35 秒到后续约 35 秒的变化归功于代码。上一轮 HT-on 和本轮 HT-off 也不构成严格配对实验：14K 续轮约 12.03 和 11.98 tok/s，未证明超线程存在稳定回退。

## 对照与已否决方案

| 方案 | 6K 续轮 TTFT s | 首字后 tok/s | 14K 续轮 TTFT s | 首字后 tok/s | 14K 续轮整轮 s |
|---|---:|---:|---:|---:|---:|
| HT-off 基线 | 1.391 | 11.197 | 1.257 | 11.980 | 9.187 |
| 追加拆成 2 行 CPU/GPU 组 | 2.058 | 10.692 | 1.644 | 12.315 | 9.359 |
| 追加拆成 4 行 CPU/GPU 组 | 1.565 | 11.516 | 1.172 | 12.574 | 8.727 |
| 8K prefill / reserve4864 / attention tile256 | 1.534 | 11.025 | 1.326 | 11.322 | 9.717 |
| GPU 追加保留全部热集，16 个冷槽 | 1.606 | 10.897 | 1.471 | 12.379 | 9.145 |
| GPU 追加只保留 32 个冷槽 | 1.419 | 10.755 | 1.284 | 12.447 | 8.916 |

14K 续轮均生成 96 tokens。拆成 2/4 行确实把追加阶段的 demand/refill H2D 都降为零，但增加了按行组执行各层的成本。4 行样本在 14K 上约快 5%，6K 首字更慢，未作为默认实现保留。

8K prefill 将 14K 输入的 demand transfer 从 160.21 GB 降到 90.83 GB，TTFT 却从 72.70 增到 74.61 秒。为容纳新增 workspace，这个实验也提高 reserve 并改 attention tile，因此不能把差异归因于 chunk 单一变量；组合没有端到端收益，撤回 8K 协议上限和对应 buffer 增长试验。生产仍为 4096 上限。

保留全部热集的第一版在真实续轮触发 `Cold batch exceeds frozen staging capacity`：冷槽只有 16，原分组为 32。Worker 正常报告 SSE runtime_fault，Server 优雅退出；没有 TDR。将 resident prefill 的专家组按实际冷槽数限流后通过真实 HTTP。6K 追加 acquire 从 124 次增到 220 次，抵消了保留热集的收益；14K 续轮整轮仅从 9.187 到 9.145 秒。

随后只借出维持 32 个冷槽所需的热集尾部，让 6K 追加 acquire 回到 120 次。14K 追加 demand 为 7.428 GB，handoff 由基线 295 ms / 1.777 GB 降到约 157 ms；整轮快约 3%。但 6K 首字从 1.391 到 1.419 秒，速率从 11.197 到 10.755 tok/s，未稳定胜出。该方案也已撤回，不留下默认关闭的新参数。

## 定位到的成本

HT-off 基线 14K 输入 prefill 72.62 秒，其中 acquire 46.19 秒、fill 26.08 秒、demand 160.21 GB。这些计时有嵌套和重叠，不可相加。24-token 续轮 prefill 1.208 秒、demand 7.603 GB，接下来的首个 decode 还花 295 ms / 1.777 GB 重建热集。

热态续轮除去首个 decode 后，每次 forward 平均 79.28 ms；CPU join 26.11 ms，CPU compute 23.82 ms，二者相互重叠。即便只看剩余约 53 ms，也说明单靠 CPU 专家提速不足以稳达 20 tok/s。这轮没有把少量续轮收益当成 decode 20+ 的突破。

实验代码与原始输出保留在 `tmp/q4e-append-rows-and-chunk8k-experiment.patch`、`tmp/q4e-append-keep-full-experiment.patch` 和同名 JSON 指向的 `tmp/q4e-htoff-*` 目录。前者只归档本轮撤回的拆行与 8K 改动；后者是全热集候选相关文件相对 HEAD 的 diff，包含既有诊断 hunk。

最终 32 槽候选的独立补丁为 `tmp/q4e-append-staging32-experiment.patch`。开启/关闭候选的 expert readiness 检查、修正后两种 GPU 保留方案的真实 HTTP 请求及优雅关闭均通过。部分 Worker 日志有丢行，汇总将覆盖不全的 engine total 标成 null；端到端表使用完整 SSE 数据。所有候选都未推广，因此没有重复跑无关的 PI 全链路验收；此前正式配置的 PI 长文工具回合证据见单流性能报告。

清理后的构建及缓存检查记录为 `tmp/q4e-htoff-consolidated-build.log`、`tmp/q4e-htoff-consolidated-readiness.log`。没有扩大到全套测试。
