# Qwen3.8-Flash-Next：单流端到端瓶颈定位

2026-10-07。A770 / driver 32.0.101.8724、oneAPI 2026.1、Xeon E5-2696 v3；CPU experts 18 threads。32K容量、chunk4096、reserve3584 MiB、MTP=0、BF16 KV、单会话、精确前缀复用，沿用当前启动配置。

**结论：主要时间在Worker内部。真实agent应先处理首次长文prefill和短追加的专家供给；要突破decode 20+，CPU冷专家等待与固定GPU路径都需要缩短。Server/IPC不是当前主瓶颈。**

## 真实PI：端到端135.79秒，Worker占134.84秒

这次使用真实PI、约14.85K背景、一次read工具回合，没有先运行短warmup。工具调用和最终标记验证通过，两个请求的Forward记录完整。

| 阶段 | Worker时间 | 专家需求H2D |
|---|---:|---:|
| 第一轮，新增14853 tokens的prefill | **127.549 s** | **177.314 GB** |
| 第一轮，26 outputs的decode/收尾 | 4.222 s | 0 |
| 第二轮，复用14879，只新增29 tokens的prefill | **1.531 s** | **8.029 GB** |
| 第二轮，11 outputs的decode/收尾 | 1.539 s | 0 |

PI CLI总计135.787 s，Worker合计134.840 s，差值约 **0.947 s（0.70%）**，包含PI启动、工具读取、请求处理、IPC等全部外层时间。首次prefill占总时延约94%，这个短工具任务仅提升decode速度对总等待的作用有限。首次prefill包含编译、首次数据触达等成本，不能当作暖态prefill吞吐。

PI首轮去掉首个热集切换步骤后，decode forward均值151.31 ms、CPU join94.02 ms；第二轮为114.21/59.96 ms。它们明显慢于下面的暖摘要样本。不同输出轨迹和缓存状态都可能影响结果，不能把差值全部归因于某一种page fault；也不能把暖基准的12 tok/s当作agent首轮的通用速度。

## 暖HTTP：外层开销不足1%

另一进程按短warmup→6K摘要→续轮→14K摘要→续轮执行，每个长请求96 outputs。轻量汇总没有添加GPU barrier；逐请求核对所有prompt rows及decode forward数后，与同一次SSE时延相减。

| 14K请求 | HTTP总时间 | Worker Forward合计 | 其余时间 |
|---|---:|---:|---:|
| 输入13924，输出96 | 81.017 s | 80.932 s | 85.5 ms（0.11%） |
| 续轮14044，复用14020，输出96 | 9.146 s | 9.108 s | 37.6 ms（0.41%） |

差值包含模板/tokenization、调度、Session操作、IPC、SSE和日志递交，不能全部叫IPC。sample_ms包含前面尚未完成的GPU工作，不能当成采样算子的纯计算时间。

## 长文prefill：专家供给最大，attention次之

暖13924-token prefill用时 **72.351 s**，四块约20.16/20.43/20.65/11.11 s。累计专家需求H2D **160.215 GB**，每块仍搬运约38–43 GB。

- acquire主机区间累计 **46.081 s，约占64%**。
- 其中fill累计 **26.108 s**，包含主机供给、读取/整理staging。
- router_wait累计 **24.305 s**。它等待路由前的有序GPU工作，包含attention/HC等前序执行，不是top-k本身算了24秒。

fill属于acquire内部，不能相加。acquire还可能等待前一批GPU消费和缓冲复用，不能全部解释为SSD/PCIe耗时。ReadFile使用Windows文件缓存；此次没有物理磁盘读取量和PCIe硬件带宽counters。

独立串行stage诊断的四个prefill块记录完整：专家区间约46.90 s（其中acquire45.39 s）、线性注意力13.75 s、full attention9.72 s、HC1.05 s。它支持优先排序，但不能直接填回轻量运行做精确分解。

## 多轮的具体问题：短追加仍走大量权重加载

暖14K续轮命中14020 tokens，真正新增仅 **24 tokens**，仍搬运 **7.631 GB**，prefill **1.211 s**，其中acquire0.908 s、fill0.762 s。SSE首字1.250 s，主要等待已明确发生在引擎。

该24行prefill的CPU专家任务数为0，冷专家走既有GPU prefill供给。随后第一个decode还做prompt热集恢复/换入：本样本 **1.784 GB / 306.7 ms**，首个decode forward合计427 ms；这段发生在首字之后。后续普通decode的专家需求H2D为0。

因此命中前缀并不意味着下一轮几乎不用加载专家。应优先研究短追加是否需要借出/重建这么多热集，以及少量新增行的冷专家如何执行。尚未测过替代策略，不能提前保证改成逐token或CPU一定更快。

## Decode：CPU和固定GPU路径都要处理

暖14K续轮SSE速率 **12.03 tok/s**。去掉首个热集切换步骤后，95个完整decode forward均值：

| 指标 | 实测 |
|---|---:|
| Forward墙钟 | **78.63 ms** |
| CPU专家 | **273.6 / 480（约57%）** |
| CPU计算区间 | **23.01 ms** |
| 主线程CPU join | **24.48 ms** |
| 专家需求H2D | **0 bytes** |
| CPU结果回传 | 约2.8 MB/forward |

CPU计算会与GPU重叠，compute和join不能相加。按当前时间账本，即使完全隐藏24.48 ms的join，其余仍约 **54.15 ms**，而20 tok/s要求约50 ms。这说明只优化CPU一侧不够，并非对未来流水线重排的理论下界。

串行诊断中，14K续轮捕获的37个完整稳定样本：HC约22.29 ms、线性注意力12.21 ms、full attention7.93 ms、PLE7.19 ms、head3.47 ms。HC是值得优先调查的固定GPU阶段；现有证据不支持把KV cache访存当作唯一主瓶颈。串行模式的router区间还包含CPU专家/结果等待，不能当成router算子成本。

同一诊断样本的CPU子阶段约为gate/up22.06 ms、down9.53 ms、量化0.288 ms。CPU侧优先看gate/up及权重/scale访问，激活量化不是主要时间。记录到的进程级page faults没有区分软/硬缺页及具体来源，不能等同SSD瓶颈。

## 证据与边界

- 轻量HTTP的5个请求及PI的2个请求记录完整，host log drop=0，能够计算上述端到端差值。
- stage/CPU诊断会添加barrier，并强制materialize原本延迟融合的MLP residual，改变重叠，decode比正常路径慢。后半段还出现Worker日志缺失；host drop=0不保证Worker未丢日志。分析器检查位置/count配对与覆盖率，缺失请求的engine总时间、外层差值标为null，只用完整样本进行阶段排序。
- 没有控制Windows文件缓存，也没有固定不同运行的生成轨迹；各次结果是诊断样本，不是统计稳定性或质量保证。
- 没有修改执行策略。诊断默认关闭：AILA_Q4E_WORK_SUMMARY=1记录Forward/专家统计；AILA_SERVER_WORKER_LOG=绝对文件路径收集Worker日志。没有用每条强制刷盘的ExecutionTrace计算性能占比。

原始证据：tmp/q4e-bottleneck-light、tmp/q4e-bottleneck-stages、tmp/q4e-bottleneck-pi下的result.json和bottleneck.json；对应tmp/q4e-bottleneck-*-worker.log。解析器为tmp/q4e-bottleneck-summary.py，二进制hash见各目录launch.json。同名报告JSON保留配置、汇总与覆盖率。

**建议顺序：先短追加的专家供给/热集切换和首次长prefill，再一起压缩HC/线性注意力与CPU gate/up的decode时间。** 当前不优先增加HTTP并发、重写IPC或继续扫小型激活量化开关。
