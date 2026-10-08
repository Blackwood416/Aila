# Qwen3.8：真实 Agent 基线与 Prefill 传输节奏否决

本轮按约定只测一个调度候选，在局部 10% 门槛处停止。限制跨组 H2D 的候选让 80 行完整专家链慢了 **9.56%**，因此没有接入主模型，没有新增运行时开关或修改启动配置。保留完整链测量入口、冻结输入工具和生产基线。此前未提交的 Server/PI 改动保留。

## 环境与测量边界

2026-10-07，Windows 显卡驱动 **32.0.101.8724**，Intel Arc A770，Xeon E5-2696 v3 **18 核/18 逻辑处理器**。编译器实测 oneAPI DPC++ 2026.1.0 Build 20260617；SYCL 返回的 driver 字符串是 Level Zero 1.15.37669，不拿它替代 Windows 显卡驱动。

真实请求沿用 32K 容量、chunk4096、reserve3584 MiB、CPU 18 线程、单会话、精确前缀复用、BF16 KV、MTP off。正式请求使用 `--production-timing` 清除 WorkSummary、stage/component profile、cache/debug trace、Worker 日志捕获和 verbose 等诊断环境变量。完整实际环境与二进制 hash 见同名 JSON。

输入从当前仓库冻结一次，正文分别为 5932/13932 tokens，附带文件和 token hash。6K 正文覆盖 AGENTS.md、Model 和 Session；14K 还覆盖 HttpServer、ExpertCache、Backend。冻结来源清单同时记录参考诊断资料及日志 hash；裁剪后的实测正文以代码和说明为主。正文不会随着后续代码编辑改变。文件在 `tmp/q4e-agent-corpus-v1/`。

PI 使用固定工作目录和 marker，配置及 session 保持独立。两次独立 stub 调用的完整请求 body 逐对象一致，排除了本测试的随机路径、随机标记和配置差异。自然代码输入与此前重复 module 句子的输入不同，不能横向计算加速比。Windows 文件缓存未清除，首次请求不等于冷磁盘测试。

## 无诊断的真实 Agent / HTTP 基线

Worker 启动后的第一个请求直接来自真实 PI，没有提前跑短 warmup。PI 首轮输入 **14915** tokens，生成 **44** tokens 并成功 read 一次；工具续轮复用 **14959** tokens，只新增 **32**，输出 **14** tokens，答案正好是固定 marker。

- 启动准备到 Server ready：**7.90 s**，包含测量 helper 的启动准备。
- PI CLI 工具任务：**212.67 s**，不含 helper 配置/launcher 查找时间。
- 上述两项相加：**220.57 s**。未单独采集 PI 的首字时间，不从消息时间戳推断。

随后同进程执行自然代码摘要及续轮，全部输出 96 tokens：

| 请求 | 实际输入 | 缓存命中 | 首字 s | 整轮 s | 首字后 tok/s |
|---|---:|---:|---:|---:|---:|
| 6K 新请求 | 5968 | 0 | 71.30 | 81.56 | 9.26 |
| 6K 摘要续轮 | 6090 | **0** | **60.60** | 69.30 | 10.92 |
| 14K 新请求 | 13979 | 0 | 135.00 | 142.43 | 12.79 |
| 14K 摘要续轮 | 14101 | **0** | **134.86** | 142.48 | 12.46 |

两条摘要续轮没有复用前缀，必须标为完整 prefill，不能称为 24-token 短追加性能。PI 工具续轮却能成功复用；这说明后续值得先定位具体历史内容下的复用失效，而不是继续把所有续轮都假设成命中。此次没有采集缓存 token 与重渲染 token 的第一处分歧，因此尚不能归因于 BPE、模板或缓存所有权，也未修改相关运行时逻辑。

主结果 `tmp/q4e-pacing-production-base/result.json`，四条完整 SSE、请求 JSON 与 PI events 同目录；Server 报告 6 个请求全部完成、无失败，优雅关闭。该数据是一次生产基线，不是优化后的成绩。

## 完整专家链：单候选、相同工作量

扩展既有 `AilaQwen4ExpPrefillSupplyBench`，增加 `--pacing-smoke` / `--pacing`。使用生产 NF4+nested codes 的 four-token gate/up、SiLU 和 down，H=2560、I=640、top-k=10；256 个专家，每组 32 个，每专家 8/80 行。token/rank 映射保证每个 token 的专家唯一。两种调度每轮均传输 **648,806,400 B**，权重、输入、精度及缓冲大小完全相同。

对照允许下一组 H2D 与当前组计算重叠；候选仅在 copy queue 上添加上一组 down 完成事件作为依赖，没有新增主线程等待。数据在计时前已填入 pinned host 缓冲，本实验不包含 ReadFile/CPU 源填充、router 或最终跨专家 rank merge。完整链 wall 从提交前到两个队列完成，事件时间在完成后读取。

每种形状、每个调度各 3×20 个交错样本，原始 60 个 wall 样本全部保留：

| 每专家行数 | 原重叠中位 ms | 限制节奏中位 ms | 完整链变化 |
|---|---:|---:|---:|
| 8 | 69.581 | 77.311 | **慢 11.11%** |
| 80 | 134.457 | 147.308 | **慢 9.56%** |

80 行的原重叠 p10/p90 为 133.988/135.394 ms，候选为 146.990/147.960 ms。三个重复的 wall 中位分别为 134.578/134.414/134.474 与 147.461/147.354/147.181 ms，差异不是一个异常样本造成的。

候选使 80 行 kernel 时间从约 125.49 降到 78.17 ms，但 copy 仍约 66.4–66.6 ms，减少重叠后 wall 更差。这支持并发传输对计算存在干扰，同时否定本候选有净收益。kernel/copy 计时有重叠，不相加充当旧路径墙钟分解。

两种形状均通过完整 h/down 输出逐位一致、未用 rank 和尾部 guard、有限值检查；独立 double CPU 参考包含 BF16 中间舍入，采用 abs=.001、rel=.006。既有内核 geometry 未改变。日志为 `tmp/q4e-pacing-smoke.log`、`tmp/q4e-pacing-bench.log`，源码补丁归档为 `tmp/q4e-pacing-experiment.patch`。

## 停止与交付

80 行未达到至少快 10% 的准入条件，按计划不做主模型 A/B、不新增运行时事件参数、不运行无关的取消/全套回归，也不追加其他调度或 kernel 候选。下一项建议是抓取自然摘要续轮缓存比较的首次 token 分歧，量清整段重算的触发条件；这里只记录，不开始实现。

工具改动包括冻结输入生成/校验、PI 固定外部工作目录、HTTP 探针的生产计时模式、启动/PI 时间与消息 hash 记录。Python 语法检查、固定 PI 请求稳定性、真实 PI/HTTP、相关 benchmark 构建和上述完整链检查通过。构建记录 `tmp/q4e-pacing-build.log`，编译器记录 `tmp/q4e-pacing-compiler.log`，机器可读汇总为同名 JSON。
