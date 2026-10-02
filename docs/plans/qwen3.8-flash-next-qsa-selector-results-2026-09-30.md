# Qwen3.8-Flash-Next：精确 QSA selector 与提交方式实验

已在 A770 默认启用 8-bit radix、并行稳定输出和 vec4 分数加载。最终同一可执行文件复测：32K 上下文 Top-K 从 1.956 ms 降至 0.118 ms，score＋select＋attention 后半段从 2.198 ms 降至 0.381 ms。2146-token 完整请求 ABBA 对照的暖态 decode 仍为 6.15 tok/s，不能将算子收益当作整机收益。继续使用 32 GiB host pool、GPU-only 专家计算。

原始背景见 [QSA 差距审计](qwen3.8-flash-next-qsa-gap-audit-2026-09-30.md)，数据、散布、命令和哈希见 [JSON](qwen3.8-flash-next-qsa-selector-results-2026-09-30.json)。

## 实现与默认范围

[CODE] [QsaIndexerOps.cpp](../../src/ops/QsaIndexerOps.cpp) 的单行及 batched block-native selector 共用新实现：

- 将 32 轮单 bit 搜索替换成四轮 8-bit radix histogram；16 个 subgroup 各有一份 histogram，减少原子争用。WG=256，SG=16，histogram 占 16 KiB SLM。
- 每线程按 vec4 加载分数，末尾不足四个元素走掩码路径；支持非 16-byte 对齐的行起点，未通过 reinterpret_cast 引入额外对齐要求。
- 在组内扫描各 digit 数量、确定精确 cutoff；按连续 ID 区间计算大于 cutoff 和等于 cutoff 的数量，再并行压紧输出。去掉 thread 0 对所有候选的两遍串行扫描。
- 保持原整数 order key：NaN 最低、正负零同分、同分优先小 ID、输出 ID 升序、512 个完整块及 0–3 个尾部 token。不改变 scoring 浮点运算、主 KV、attention 数学或选块预算。单行 513→512 特例保留。

[DISPATCH] 默认仅用于 Intel Arc A770、Level Zero、K=512、compress_ratio=4、完整块数不超过 32768（约 128K 上下文）。设备识别按 Context 缓存；其他设备、预算和未覆盖的更长范围仍用原路径。正确性检查覆盖上限两侧，批量性能检查包含 4/64 行。A770 驱动 32.0.101.8860，oneAPI 2026.1，oneDNN 3.11.2。此范围内收益和正确性证据充分，整模型吞吐证据仅限下文的 2146-token 请求。

AILA_QSA_SELECTOR 可用于复现：-1/未设置为自动；0 为旧实现；1 仅替换单行原生归约；2 为 4-bit radix；3 为 8-bit scalar；4 为 8-bit vec4。显式值供实验使用，可越过默认设备及形状范围。旧实现和实验版本均保留。

## 算子结果

[MEASURED] 固定 BF16 Q/K/V，Hq=24、Hkv=2、D=256，indexer 4×128。每项预热五次，三轮，每轮二十次同步 wall 计时及二十次连续提交均摊 wall。表为三个中位数的中位数，单位 ms；不是硬件 event 时间。最后一列直接测量组合路径，不是相加独立中位数。

| 上下文 | 旧 Top-K | 新默认 Top-K | 旧组合路径 | 新默认组合路径 |
|---|---:|---:|---:|---:|
| 2146 | 0.2324 | 0.0976 | 0.4656 | 0.3312 |
| 4096 | 0.3556 | 0.0994 | 0.5942 | 0.3404 |
| 8192 | 0.6197 | 0.1015 | 0.8118 | 0.3425 |
| 32768 | 1.9561 | 0.1178 | 2.1979 | 0.3814 |

组合路径排除 projection、norm/RoPE、compression、gate 和 o_proj。32K 新默认的连续提交均摊约 0.279 ms，旧实现约 2.097 ms，收益不只来自逐次等待的 host 开销。QSA 阈值附近仍有评分/选择成本，不能保证上下文每增长一步整机就更快。

64 行 batched selector 的同构对照：

| 约等效上下文 | 旧实现 ms/批 | 新默认 ms/批 |
|---|---:|---:|
| 32K | 2.4271 | 0.2409 |
| 64K | 4.7357 | 0.3840 |
| 128K，最后一行完整块数=32768 | 9.3784 | 0.6488 |

边界外的 64 行（最多 32783 块）也验证了回退，默认返回旧路径约 9.337 ms；显式 mode=4 约 0.655 ms。此次不扩大自动范围。以上长上下文是 selector 合成输入，不是 64K/128K 整模型生成。

仅换原生归约时，32K Top-K 仍为约 1.938 ms。8-bit scalar 最终为 0.1351 ms，vec4 为 0.1173 ms，长形状降低约 13%；前一轮分别为 0.1341/0.1172 ms。中短形状的收益较小，8K 一次 vec4 wall 测量出现抖动，不将其解释成每个形状都稳定加速。

[MEASURED] VTune instruction-count：相同四行、四次 selector 调用，旧/vec4 的总指令数 12,869,300 / 1,667,512；Send 745,024 / 119,172；Synchronization 879,148 / 85,504。两者 SIMD16、spill=0。采集显著扰动执行时间，VTune 时间未用于速度结论。

## 真实请求

[MEASURED] 四个独立进程按 0→4→4→0 顺序运行，每进程两次相同请求，均生成 16 token。固定输入 2146 token、max-seq=2560、专家槽=3567、host pool=32768 MiB、chunk=512、phase scratch、demand microbatch=32，关闭阶段 profile/trace。可执行文件、模型配置、输入和除 selector 外的环境一致。

| 指标 | 旧实现 | vec4 |
|---|---:|---:|
| 两个暖请求 decode tok/s | 6.2、6.1 | 6.2、6.1 |
| 中位 decode tok/s | 6.15 | 6.15 |
| 中位完整请求 ms | 34716.45 | 35053.69 |
| 中位 TTFT ms | 32444.0 | 32795.5 |
| 峰值进程工作集 GiB | 37.246 | 37.263 |
| 最低系统可用 RAM GiB | 10.683 | 10.864 |

八次请求的生成文本哈希完全一致。当前样本只比 QSA 阈值长约一百 token，且专家供给等待显著，未测出整机改善；完整请求约 1% 的差异也不足以归因。每种模式只有两个暖请求，不能据此评价更长输入的真实吞吐。

## SIMD、immediate command list、乱序队列

SIMD 已落地在 selector 的 vec4 加载。下一项是 indexer scoring 的 BF16 连续加载与 subgroup 组织：当前每块 128 线程、四次 WG reduction，可研究 SG16 每 lane 处理多个维度、多个 block 共用 WG。这里会改变浮点归约顺序，需重新验收 score 误差及 cutoff 附近的 IDs；本轮未修改。NF4 GEMV 的已知失败布局不重试。

[MEASURED] 对 8-bit scalar QSA 固定其余条件，UR_L0_USE_IMMEDIATE_COMMANDLISTS=0/1 的 32K 组合路径逐次同步 wall 为 403.75/384.50 us；连续提交均摊却为 295.225/361.010 us，开启后慢约 22%。单 selector 连续提交也由约 60.87 增至 71.40 us。独立 UR debug 探针确认 0 使用普通 list＋queue execute，1 使用 immediate lists；不是只设置了无效变量。尚未在完整请求上评估，保持计算队列原提交策略。

[CODE] 专家 copy queue 已显式启用 immediate command list，现有 copy/compute 重叠不是缺失项。

[HYPOTHESIS] out-of-order 优先研究两个具体分支：

1. Indexer projection/prepare/score 与主 QKV projection/prepare 在共同输入就绪后分开，attention 汇合等待两边事件。需为并行投影拆开 linear_scratch/oneDNN scratch，补齐返回事件和跨队列依赖。
2. Shared expert 与 routed expert 的独立计算。共享专家目前已提前提交，与 CPU 路由等待/专家读取重叠；新增 GPU 并发需要证明有空闲执行资源，不能把已有重叠重复记为收益。驻留专家与缺失专家分支还必须保持 DMA 完成事件、消费 fence、最终有序归并。

Context 与多数算子仍依赖 in-order 的隐式顺序；移除该属性会遗漏生产者依赖，并造成共享临时区冲突。本轮未切换全局队列。下一阶段应先细分并减少专家 read/transform/prefetch wait，再测试有事件依赖的局部分支并发。

## 验证与复现

默认与显式 vec4 均通过 1428 组整数 IDs/数量/未写区域检查，包含 NaN、Inf、正负零、随机 float 位型、极近分数、同分、K=1/31/512、尾部 0–3、奇数行步长和 32768 块边界两侧。Batched attention、非连续 decode IDs 和 gate 回归通过；九种长度的 attention 检查两个 heads 的 CPU f64 参考及全部 heads 有限性。没有重新认证 scoring 浮点精度，因为本轮未修改它。热路径无新增 USM，退出 USM 归零。

Aila、AilaWorker、AilaShared、AilaInference、AilaQwen4ExpKernelTests 构建通过。已运行相关 GPU 算子验收和真实 CLI 对照，未运行无关全套 CTest 或 HTTP 服务验收。汇总脚本校验 31 份成功实验及两份提交方式 trace 的退出状态/日志哈希；Python/PowerShell 语法和 diff whitespace 检查通过。

~~~powershell
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload qsa_tests -Tag qsa-default-tests
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload qsa -QsaSelector 0 -Tag qsa-old
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload qsa -Tag qsa-default
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload qsa_batches -Tag qsa-batches
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload qsa_profile -QsaSelector 4 -VtuneInstructions -Tag qsa-vtune
~~~

首次尝试对完整成本曲线进行 VTune instruction-count 采集未完成，停止命令内部错误，后处理又遇日志句柄占用；tmp/qsa-selector-baseline-vtune-20260930 不用于结果。随后改成四次调用的短探针，旧/新两次采集均正常完成。早期重链接遇正在运行的探针占用 EXE/DLL，进程退出后构建通过；后续构建与 GPU 执行分开。原始失败文件保留。
