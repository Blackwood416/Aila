# Qwen3.8-Flash-Next：R0 / R1a / R2a 首轮实施结果

日期：2026-09-30。依据 [architecture brainstorm](qwen3.8-flash-next-architecture-brainstorm-2026-09-30.md) 第 12 节的首轮边界，完成显式 CLI 性能 preset、同自然输入的窗口对照、一个热专家桶的成熟 GEMM 对照。基线为 `66e21e9` 加已有 S1–S6、RAM、QSA 工作及本轮改动。**R1b 与 R2b–R6 尚未实施，实际 decode 20 tok/s 尚未达到。**

机器为 Intel Arc A770 16 GB，驱动 32.0.101.8860，oneAPI 2026.1 / oneDNN 3.11.2；模型为 C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2。全模型测试固定 32 GiB host pool，专家算术在 GPU 执行。原始日志、进程配置、输入和可执行文件哈希、内存采样、聚合值及 VTune 指令记录见 [结果 JSON](qwen3.8-flash-next-architecture-r0-r2a-results-2026-09-30.json)。

## 1. 本轮决策

| 工作包 | 实测结果 | 决策 |
|---|---|---|
| R0 显式 preset | 41-token CLI 暖态 TTFT 1911→1579.5 ms；快照和重放检查通过，48 步质量 canary 通过 | 可显式使用，普通启动默认不变 |
| R1a 大窗口 | 1833-token 自然文本暖态 TTFT 22.607→16.412 s；prefill demand 字节减少 68.5% | 进入 R1b，大窗口与算子 tile/scratch 解耦 |
| R1a 后续 decode | 同请求 16-token decode 7.35→5.25 tok/s | chunk=2048 不进入 preset 默认 |
| R2a 成熟 GEMM | 完整路径约 617 μs，旧 M8 约 1593 μs，ALU 约 517 μs | 保留 ALU；原型仅在测试中，后续转 R2b |

大窗口的字节收益已足够明确，无需继续盲目扩大 chunk。111.7 tok/s 是本自然输入的**暖态输入 token 数 / TTFT**，不是冷启动速度，也不是独立 GPU prefill 计时。它说明执行窗口值得改造，不能替代 decode 目标的验收。

## 2. R0：让真实 CLI 使用已有能力

新增 CLI 参数 `--performance-preset qwen38-a770`，在模型和队列初始化前校验 Qwen4Exp NF4 模型及 A770 / Level Zero 设备，然后配置已有分发。调用示例（已配置 oneAPI 运行环境的终端）：

~~~powershell
./build-refactor/Aila.exe -m C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2 --performance-preset qwen38-a770 --max-seq 2560
~~~

默认组合为 dense XMX、direct prefill I/O、demand microbatch=32、phase scratch arena、32 GiB host pool、ReadFile、chunk=512、通用合法 interior checkpoint。grouped M8 XMX、GDN chunk、小行 NF4/HC 和 static pinning 默认关闭；CPU expert dispatch 强制为 0。已有非空环境配置优先并打印来源。此 flag 只加入 CLI，未给 Worker 新增同名入口。

[Q4E ExecutionConfig] 打印实际生效的 chunk、RAM、专家槽数和关键分发；preset 下的 [Q4E PrefillSlice] 打印输入数组切片位置与 checkpoint。另增加 opt-in 的 AILA_Q4E_WORK_SUMMARY=1，复用现有计数器打印每次 Forward 的 demand 字节、host hit/miss 与 device ledger，没有新增 GPU 等待。

### 2.1 真实 CLI 对照

每个进程运行 3 次同一 41-token 请求，每次清空会话，生成 16 tokens；max-seq=512，均为 32 GiB host pool、3714 专家槽。第二、三次为进程内暖请求。普通配置对照未启用 dense XMX、arena、microbatch 或通用 checkpoint。

| 指标 | 普通配置，RAM 固定 32 GiB | preset |
|---|---:|---:|
| 首请求 TTFT | 11933 ms | 10680 ms |
| 两次暖 TTFT | 1917 / 1905 ms | 1587 / 1572 ms |
| 暖 TTFT 中位数 | 1911 ms | 1579.5 ms（−17.35%） |
| 暖 decode | 11.5 / 12.0 tok/s | 11.9 / 12.4 tok/s |
| 暖完整请求中位数 | 3131.485 ms | 2738.81 ms（−12.54%） |
| 峰值 device ledger | 13.3611 GiB | 13.1085 GiB |

preset 是多个已有能力的组合，不能将 decode 的小幅变化归因于 checkpoint。两组自由生成文本不同，数值验收另用固定 token canary。

### 2.2 41-token / checkpoint=37 状态与质量

采用真实 CLI 模板，验证合并 41 行与 37+4 行的差异。未来 suffix 改写后，checkpoint 与可见 KV 保持一致；truncate 恢复准确，重复重放位级一致。

| 固定续写 | mean KL | mean NLL delta | max logits NRMSE | 结论 |
|---|---:|---:|---:|---|
| 前 16 步 | 0.00195266 | +0.01370954 | 0.05568359 | NLL 超过 0.01 预警线 |
| 全 48 步 | 0.00159666 | +0.00559759 | 0.06540588 | 本 canary 的 KL/NLL≤0.01 门槛通过 |

最初 16 步运行因此失败，保留 tmp/r0-cli41-checkpoint-20260930.*；随后扩到既有 X2 的 48 步协议，仍打印完全相同的前 16 步预警。48 步通过不消除这个局部现象，也不是广泛质量认证。

三组交错 prefill 计时：split 为 1759.8452 / 1357.0173 / 1755.2695 ms；inline 为 1557.3534 / 1194.5027 / 1559.1587 ms。中位数 1755.2695→1557.3534 ms，降低约 11.3%。这些是 canary 内计时，与上表 CLI TTFT 分开报告。

## 3. R1a：自然文本上的路由复用窗口

输入为 [archive-natural.json](/E:/RiderProjects/Aila/perf/qwen4exp/prompts/archive-natural.json)，一份不靠重复句子扩长的档案现代化需求文本，约 1615 英文词。实际 CLI 含模板共 **1833 tokens**，max-seq=2560，3567 专家槽，32 GiB host pool。四个独立进程按 512→2048→512→2048 执行，每进程两次相同请求、每次生成 16 tokens；仅 chunk 不同。该顺序是交错复测，不是 ABBA。

chunk=2048 时实际 forward 形状为 **1829+4**，未填满 2048 行；chunk=512 为 **512+512+512+293+4**。该长度在 QSA 稀疏切换阈值以下，本结果来自窗口复用，不能归为 QSA 稀疏收益。性能进程没有启用同步 stage profile 或 cache trace；WorkSummary 只读取已有计数器。

| 指标 | chunk=512 | chunk=2048 |
|---|---:|---:|
| 两个首请求 TTFT | 60.847 / 61.755 s | 42.482 / 38.933 s |
| 两个暖请求 TTFT | 22.381 / 22.833 s | 16.689 / 16.135 s |
| 暖 TTFT 中位数 | **22.607 s** | **16.412 s（−27.4%）** |
| 输入数 / 暖 TTFT 中位数 | **81.08 tok/s** | **111.69 tok/s** |
| 暖 decode | 7.3 / 7.4 tok/s | 5.2 / 5.3 tok/s |
| 暖 decode 中位数 | **7.35 tok/s** | **5.25 tok/s（−28.6%）** |
| 暖完整请求中位数（输出 16 tokens） | 24.49647 s | 19.091895 s（−22.06%） |
| 暖 prefill demand 字节中位数 | 121.6347 GB | 38.3151 GB（−68.5%） |
| 暖 decode demand 字节（16 行） | 5.7759 GB | 6.0344 GB |
| 暖 decode host misses | 205 / 198 | 371 / 373 |
| 峰值 device ledger | 13.3961 GiB | 13.9445 GiB（+561.54 MiB） |

GB 按 10^9 B、GiB 按 2^30 B。demand 字节是已调度逻辑 payload，包含重新搬运；不是物理总线字节或裸 PCIe 带宽。decode 字节包含末尾状态对齐，CLI decode tok/s 使用正常生成统计，两者口径分别保留。

八次请求的 16-token 输出文本相同。大窗口后的 host misses 增多，同时缓存结束状态和保留的临时区发生变化；尚不能将 decode 回退单独归因于 scratch 或 WDDM paging。ledger 不是 DXGI 驻留量。本轮普通生成进程峰值工作集约 37.394 GiB，最低系统可用 RAM 10.286 GiB，未触发 8 GiB 内存门槛或 watchdog。

相同 1833-token 输入的 48 步固定续写检查得到 mean KL=0.00616275、mean NLL delta=−0.02170013、max logits NRMSE=0.08175737，全部有限且无 NLL 预警。负 NLL delta 只描述这个续写，不能视为模型整体质量改善。

**结论：保留 preset 的 512 默认，进入 R1b。** 目标是保留大窗口的专家需求合并，用小 tile 限制 attention/GDN/MoE scratch，并单独管理 prefill→decode 的缓存和容量交接。当前仅完成 R1a 对照，尚未实现专用 PrefillExecutor。

## 4. R2a：有限 BF16 staging + oneDNN batched GEMM

新增测试专用 --mature-moe-model：从原始 NF4 文件经正常 ReadFile/cache 加载 layer 0 的专家 7、113，E=2、每专家 M=32、H=2560、I=640。输入为固定种子的合成 BF16 激活，并非捕获的真实模型激活。top-k rank buffer 为 10，仅两路权重各 0.5 的贡献有效，merge 方式一致。

三个模式分别为现有 ALU G3 四 token 复用、旧 M8 grouped XMX、GPU NF4→BF16 staging + gather + 两次 oneDNN batched matmul + SwiGLU + scatter + ordered merge。每次计时都包含这些准备与收尾操作；NF4 槽预先加载，三路同样不计 H2D。fixture 只用两个槽并关闭 host pool，不是全模型 32 GiB 实验。

BF16 权重 staging 为 19,660,800 B（18.75 MiB）；总 device ledger 30,760,833 B。两个 oneDNN primitive 均为 jit:gemm:any，BF16 输入/权重、FP32 输出、strided 转置权重、strict fpmath；本形状 user scratch 为 0。descriptor、参数表和 USM 在循环前创建，计时循环的 USM allocation count 不增长。每模式 warmup 5 次，随后 3 轮，每轮 20 个交错样本。

最后一次独立普通计时如下；前次显式 SG16 构建的三路中位数为 518.85 / 1598.5 / 618.05 μs，结论一致。

| 模式 | 三轮中位数（μs） | 中位数（μs） |
|---|---|---:|
| ALU G3 | 517.1 / 516.7 / 511.15 | **516.7** |
| 旧 M8 XMX | 1603.1 / 1593.2 / 1588.3 | **1593.2** |
| BF16 staging + oneDNN | 616.9 / 614.9 / 626.75 | **616.9** |

成熟 GEMM 完整路径比旧 M8 缩短 **61.3%**，但仍比 ALU 增加 **19.4%**。本桶不接入生产分发；不能据此排除其他 m_e 桶，只是不再将当前 M32 BF16 staging 当作已获得的收益。

### 4.1 数值与指令证据

全部合并输出对照原始 NF4 ALU：M8 NRMSE=0.00380601；oneDNN NRMSE=0.00380570、max abs=0.0001220703125。两者逐元素混合条件 abs(error)≤0.02*abs(reference)+0.0002 均为 0 errors，NRMSE≤0.02。近零参考值使最大纯相对误差达 151.94，因此不将纯相对误差当作单独验收条件。

另对每专家两行、共四行，使用已展开 BF16 矩阵计算独立 CPU f64 参考，验证 oneDNN 的完整 gate/up→SwiGLU→down：NRMSE=1.82857e−7，max abs=8.42920e−9；abs(error)≤0.0001*abs(reference)+0.000001 为 0 errors。它验证 staged 算术，NF4→BF16 格式差异由前述原始 NF4 对照覆盖。这是 relaxed-matched 算子比较，不是模型级质量认证。

单独 VTune 指令运行正常完成，报告为 [VTune CSV](/E:/RiderProjects/Aila/tmp/r2a-m32-vtune-report.csv)。新 dequant kernel 约 151,756,800 instructions、spill=0、SIMD16；两次 oneDNN GEMM 分别约 2,424,320 / 1,937,920 instructions、spill=0、SIMD8；gather / activation / scatter 均 spill=0。旧 ALU gate / M8 gate 分别有 576 / 864 B spill。准备工作指令量值得关注，但不能据此直接推算其墙钟关键路径；instrumented 时间没有混入性能表。

## 5. 复现、检查和后续顺序

主要实现见 [preset](/E:/RiderProjects/Aila/src/utils/Qwen38A770Preset.hpp)、[R0/R1 canary](/E:/RiderProjects/Aila/tests/models/Qwen4ExpPresetAcceptance.hpp)、[R2 fixture](/E:/RiderProjects/Aila/tests/models/Qwen4ExpMatureGemmProbe.hpp)、[probe runner](/E:/RiderProjects/Aila/perf/qwen4exp/q4e_execution_probe.ps1)、[汇总脚本](/E:/RiderProjects/Aila/perf/qwen4exp/q4e_architecture_summary.py)。

~~~powershell
# R0 实际 CLI；去掉 preset 并加 -InlineCheckpoint 0 可复现普通配置对照
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload generation -RuntimeDefaults -HostRamMB 32768 -PerformancePreset qwen38-a770 -WorkSummary -GenerationRepeats 3 -MaxTokens 16 -Tag r0-repro

# R1a：分别以 512、2048 运行；每次使用新 Tag，按 512→2048→512→2048 交错
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload generation -RuntimeDefaults -HostRamMB 32768 -PerformancePreset qwen38-a770 -WorkSummary -GenerationPromptFile perf/qwen4exp/prompts/archive-natural.json -GenerationMaxSeq 2560 -GenerationRepeats 2 -MaxTokens 16 -PrefillChunk 512 -Tag r1-repro-512

pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload preset -Tag r0-quality-repro
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload window -Tag r1-quality-repro
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload mature -Tag r2-repro
~~~

已构建 Aila、AilaWorker、AilaShared、AilaInference 与 AilaQwen4ExpToleranceTests；CLI help、非法/缺失 preset 值和缺失模型 guard 已检查。最后一次仅重编测试的误差报告逻辑后，R2 完整 probe 再次通过。Python/PowerShell 语法、JSON 与 diff 空白检查通过。没有运行全量 CTest 或 HTTP 并发验收，未改变调度器或生产 MTP 状态机。

原始前缀：R0 为 tmp/r0-cli-{baseline,preset}-20260930、tmp/r0-cli41-checkpoint48-20260930；R1 为 tmp/r1-natural-{512-first,2048-large,512-return,2048-repeat,quality}-20260930；R2 为 tmp/r2a-m32-{baseline,final,vtune,contract}-20260930。JSON 保留成功记录的完整 metadata；初始 16 步失败单独保留如上，不计入性能聚合。

下一阶段按规划进入 **R1b**，先做单 session、受预算的 1024/2048 路由窗口与小算子 tile，保持因果、快照与取消 drain 的语义。算术侧随后做 **R2b 的 NF4 integer-LUT × Q8 activation** 和 **R3-head 的完整词表 Q8** 小原型。SIMD 加载以新路径的实际指令与时延筛选；沿用此前 immediate-list 负结果，OOO 仅在显式依赖与独立 scratch 条件满足后考虑。
