# R2b：NF4 integer-LUT × Q8 的 DP4A 原型结果

日期：2026-10-01。A770 16 GB / 驱动 32.0.101.9033 / oneAPI 2026.1；HEAD 66e21e9 加既有未提交工作及本轮测试改动。

**结论：[MEASURED] 原生 DP4A 确实生成，但本轮完整专家路径尚未赢过现有 NF4。** 加入四行权重复用和低成本、精确修正舍入的 Q8 量化后，M1 仍慢 20.8%，M32 仍慢 6.2%。保持测试入口，不接入生产。完成的是 R2b 的有界 DP4A 实验；INT8 DPAS、代表性隐藏态及模型质量尚未完成，不能将 R2b 整体标为交付。

机器可读数据、源码/二进制/日志哈希、逐轮 p10/p90、设备分阶段计时与 ISA 哈希见 [结果 JSON](qwen3.8-flash-next-r2b-integer-results-2026-10-01.json)。

## 1. 实现与实验边界

[CODE] 新增 tests/models/Qwen4ExpIntegerMoeProbe.hpp，通过 AilaQwen4ExpToleranceTests --integer-model 调用；runner 增加 integer workload 和 IntegerDeviceProfile。没有修改生产专家算术、默认 preset 或缓存预算。

- 真权重：Qwen3.8-Flash-Next-BNB-NF4-v2，第 0 层专家 7、113，H=2560、I=640。
- 每专家 M=1/32；rank 存储 top_k=10，但仅两位权重各 0.5，其余为零。输入是 seed=61003 的 BF16 正态合成激活，stddev=0.25。token 映射为 (m*7)%M，用于覆盖 gather/scatter。
- 原 NF4 payload 和 nested scales 不变。整数 LUT 是 round(127*NF4)，不是 IQ4_NL 的码表：[-127,-88,-67,-50,-36,-23,-12,0,10,20,31,43,56,71,92,127]。
- 激活每 64 维对称 Q8；每块分别应用原 NF4 weight scale 和 activation scale /127。uint32 权重加载、BNB 每字节高 nibble 在前、SLM LUT、SG16；每个 subgroup 对应一个权重输出行。四行版只复用同一专家权重于四个 token。
- 完整路径：输入量化 → gate/up → SwiGLU/BF16 → intermediate 量化 → down FP32 → 原有 ordered merge/BF16。所有 scratch 预分配。
- 新增 Q8 scratch：M1 4,080 B；M32 130,560 B。INT8 权重缓存为零。
- 这是两槽 resident 算子 fixture，host pool=0；不含 H2D 或模型加载。研究用 32 GiB host pool 约定保持不变，不能用该 fixture 的 RAM 值推断长文内存。

## 2. 数值验证

[CORRECTNESS] 所有候选均通过 CPU 整数参考、精确 Q8 字节检查、零块、RTE 半整数、NaN/Inf 原始位保护、SwiGLU/rank 地址及 ordered merge 检查。CPU 投影参考使用真实 packed 权重及 nested scale，按完整 64 维整数和、FP64 逐块缩放；M1 检查 7,680 个 gate/up/down 值，M32 每候选 15,360 个值。CPU 投影误差 NRMSE 约 1.05e-7，max_abs <8e-8，在 abs=1e-6、rel=1e-4 内零失败。

两个 fast-math 陷阱已在原型中处理：

1. BF16 转 float 后做 finite 位检查可能被优化掉；改成在转换前读取 BF16 原始位。测试 fixture 将非有限值置零；这不是生产推理的错误处理契约。
2. CPU/GPU 的倒数乘法都可能扰动精确半整数。CPU oracle 从 BF16 的尾数与指数建立整数分数，余数判定 ties-to-even，不依赖浮点除法。最初的 fdiv_rn 量化作为对照保留；低成本版本先用倒数估计，再以 BF16 × 小整数的精确 FP32 乘积比较半整数边界。阈值乘积除以 256 避免大数溢出，小块幅度 ≤2^-110 回退 fdiv_rn。已测的输入/中间激活中，两种量化的字节均与整数 oracle 一致；没有宣称覆盖所有 BF16 位模式。

[MEASURED] 相对**原 NF4 完整输出**的近似误差：

| M | NRMSE | max_abs | 超过 abs=0.0002 + rel=0.02 的元素 |
|---|---:|---:|---:|
| 1 | 0.0179581 | 0.000274658 | 3 / 2,560 |
| 32 | 0.0165318 | 0.000503540 | 29 / 81,920 |

单行、四行、修正量化候选在这些汇总量上相同。整数实现正确不等于近似质量通过：原来的逐点门槛未通过，未放宽该门槛，未宣称模型 NLL/生成质量可接受。没有将候选注入模型循环，也没有执行该候选的 checkpoint/自然文本质量验收。

## 3. 完整路径 wall time

[MEASURED] 同一最终 EXE 的交错比较；每模式 8 次 warmup，3 轮 × 20 样本，轮内轮换模式顺序；从提交到完成等待，包含两次量化。表中是三轮中位数的中位数。profiling 关闭，VTune 数据不用于此表。

| 模式 | M1 µs | M32 µs |
|---|---:|---:|
| 0：原 NF4（decode grouped / G3 四行） | 195.75 | 522.35 |
| 1：整数单行 + fdiv_rn 量化 | 250.00 | 952.95 |
| 2：整数四行 + fdiv_rn 量化 | 不适用 | 593.90 |
| 3：整数单/四行 + 修正倒数量化 | 236.55 | 554.55 |

最佳候选相对原 NF4 的延迟：M1 +20.8%，M32 +6.2%。M1 mode3 三轮中位数 236.85 / 233.30 / 236.55 µs；M32 为 556.50 / 554.55 / 552.45 µs，三轮均落后。部分 p90 有明显尾部波动，详见 JSON；没有把单次最低值当作收益。

这不是整机 decode tok/s 或真实 prefill tok/s。两个专家的比例也不能直接乘到模型 top-10。R2a 的 BF16 staging + oneDNN 是另一精度/算术基线，结果仍见既有报告；本轮未将历史 oneDNN 时间当作同二进制整数基线。

## 4. 设备时间与实际 ISA

[MEASURED] 另一次同 EXE、开启 queue profiling 的 20 样本中位数，单位 µs。各阶段的中位数不能当作完整请求延迟；它们用于区分 kernel 工作与启动成本。

| M / 模式 | Q8 输入 | gate/up | Q8 intermediate | down | merge |
|---|---:|---:|---:|---:|---:|
| 1 / 原 NF4 | — | 73.438 | — | 24.896 | 8.230 |
| 1 / 整数修正量化 | 8.438 | 69.062 | 6.458 | 19.844 | 8.542 |
| 32 / 原 NF4 | — | 251.094 | — | 140.938 | 21.146 |
| 32 / 整数四行修正量化 | 8.959 | 244.792 | 6.667 | 148.906 | 21.562 |

精确除法版两次量化合计约 65.6–65.8 µs，修正倒数版约 14.9–15.6 µs。M1 的投影局部节省被两次量化和额外提交抵消；M32 四行版两次投影合计基本与原 NF4 持平，没有足够计算收益覆盖量化和启动。不能继续把“用了整数指令”当作完整路径一定加速的证据。

[TOOLCHAIN/MEASURED] 使用 VTune 捕获运行时 zebin，再用 ocloc disasm 查看实际机器码：

| kernel | native dp4a 静态指令位置数 |
|---|---:|
| gate/up 单行 | 4 |
| down 单行 | 2 |
| gate/up 四行 | 16 |
| down 四行 | 8 |

例如实际指令为 dp4a (16|M0)，有符号 d 类型操作数。四行版不是把名称改成 DP4A；已有机器码证据。静态位置数不是动态总执行次数。

新增 perf/qwen4exp/q4e_integer_isa.py：Windows 下长 C++ kernel 名会使 ocloc 输出文件超出路径限制，因此仅在**分析副本**中缩短 ELF section 名，并逐段断言 executable section 字节完全不变，原始 zebin/机器码/反汇编均保存 SHA256。这些副本从不执行。

VTune 四行 gate/up 报告 spill=1024 B，对照 G3 为 1216 B；整数 down 四行报告无 spill。整数四行 gate 的动态 instruction count 缺失，不能当作零；完整原始 CSV 保留。这里仍有寄存器和 LUT/scale/索引成本，不是只剩四项整数点积。

进程环境 IGC_ShaderDumpEnable / DumpToCustomDir 在本机首次尝试未产生 dump；没有改注册表。最终 ISA 证据来自 VTune 实际捕获，非另一个编译器版本的离线重编译。

## 5. 验收和后续决策

- 两个相关 CTest 均通过：AilaQwen4ExpKernelTests、AilaQwen4ExpToleranceTests（2/2）。真实专家专项普通计时、event profiling、VTune 三次最终运行均 exit=0，未触发 watchdog；热路径 USM allocation count 不变。
- 最终三次运行使用同一 EXE SHA256：D7A6AC0374BD5293697049C2E26483DC182B1721C65DC719CE36856E9A328286。普通 fixture 峰值工作集约 186 MiB，系统最低可用约 52.4 GiB；这些值不代表全模型运行。
- 保留原型和失败对照，不改生产 dispatch。没有观察到本轮 GPU fault/TDR；本轮没有旧驱动同二进制 A/B，不能把候选性能差异归因于 9033，也不能据此否定之前约 3% 的 TTFT 回退信号。
- R2b 的下一候选应是 **INT8 DPAS 热桶矩阵路径**，按有限 tile 进行 NF4→integer-LUT 转换及 Q8/VNNI 布局，完整计入转码/量化/scale/merge。当前数据不足以否定它，也不构成推荐它上线的证据。
- 若继续尝试 decode DP4A，先证明可以把 Q8 输入量化与已有上游归一化合并、减少额外提交，并在真实隐藏态上满足质量门槛；单独再扫 SG/WG 不应是下一主线。按原规划另一可独立推进项是 R3-head。

## 6. 重现

~~~powershell
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload integer -Tag r2b-integer-corrected-20261001
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload integer -IntegerDeviceProfile -Tag r2b-integer-device-final-20261001
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload integer -VtuneInstructions -Tag r2b-integer-vtune-final-20261001
python perf/qwen4exp/q4e_integer_isa.py --result-dir tmp/r2b-integer-vtune-final-20261001-vtune --output tmp/r2b-integer-vtune-final-20261001-isa
~~~

重跑时更换 Tag/输出目录，VTune 不复用已有 result-dir。runner 默认 240s watchdog、可用 RAM 8 GiB 下限；GPU 压力运行按顺序执行，测量期间不重建/重链。
