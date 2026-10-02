# R2b：INT8 DPAS 热桶与静态 graph 实测

日期：2026-10-01。Intel Arc A770 / 驱动 32.0.101.9033 / oneAPI 2026.1。代码基线为 66e21e9 加此前未提交工作及本轮测试改动。

**[MEASURED] 找到有实测收益的算子候选：保留 NF4 payload，有限 INT8 staging + block64 DPAS + 静态 command graph。** 在两个真实驻留专家、合成 BF16 激活上，完整链包含转码、两次量化、pack 和 merge；与同样使用 graph 的原 NF4 对照相比，两次独立运行的 M32 延迟下降 26.0%–26.4%，M48 下降 24.4%–26.4%，M64 下降 46.8%–47.7%。

这不是整机 prefill/decode TPS，也不是生产 dispatch。M8/M24 落后，M16 余量很小；质量仍只有算子参考，逐点 NF4 近似门槛仍有少量超限。默认推理路径、32 GiB 研究 host pool 和 GPU 专家计算主线不变。

原始记录、源码/EXE/日志/机器码哈希、每轮 p10/p90、graph 构建及首次重放成本见 [结果 JSON](qwen3.8-flash-next-r2b-dpas-results-2026-10-01.json)。前一轮纯 DP4A 结果见 [DP4A 报告](qwen3.8-flash-next-r2b-integer-results-2026-10-01.md)。

## 1. 实现与算术

新增 tests/models/Qwen4ExpDpasMoeProbe.hpp；共享上一轮真实专家 fixture 的量化和 CPU oracle。入口为 AilaQwen4ExpToleranceTests --dpas-model，另有不加载模型的 --dpas-smoke，已注册 AilaQwen4ExpDpasSmoke CTest。

- 两个真实专家：第 0 层 7、113；H=2560、I=640。每专家 M=8/16/24/32/48/64，rank 存储宽度 10，但只有两个专家、路由权重各 0.5。
- NF4 bytes 和 nested scales 保持原格式。GPU 将当前两个专家的 payload 转成 round(127*NF4) 的 int8 行主序 W[N,K]，同时解出原 block64 weight scales。没有全模型 INT8 缓存。
- 激活使用上一轮通过精确整数 oracle 的 per64 Q8；VNNI B 布局为 [K/4,M,4]。gate 输入按实际 activation token map gather，down 输入按 token/rank gather。
- 使用普通 SYCL joint_matrix，显式 SG8；tile 为 8 输出通道 × 8 token × K32。每两次 K32 点积组成一个 K64 int32 partial，再用该块的 weight scale × activation scale /127 转为 FP32 累加。不能把整个 K 的整数和统一乘一个 scale。
- gate/up 在同一个 kernel 中累加，再执行 SwiGLU/BF16；中间激活重新量化后执行 down，沿用原 ordered merge。
- 普通 DPAS 链共九个 GPU 提交：Q8 输入、gather/pack、gate/up 权重转码、gate/up DPAS、Q8 intermediate、down pack、down 权重转码、down DPAS、merge。

INT8 staging 固定为 6,553,600 B，scale staging 409,600 B，gate/up 与 down 顺序复用同一空间。激活 pack/gather 缓冲随 M 有界增长；M64 三块合计 675,840 B。所有 scratch 和 oneDNN descriptors 在计时前建立，热循环 USM allocation count 不增长。

## 2. 匹配的成熟库对照

[CORRECTNESS] oneDNN 也按 K64 独立整数 matmul，不能用不匹配的全 K INT8 matmul 假装同等算术。把 [expert,Kblock] 当 batch，计算 s8×s8→s32 的 [N,64]×[64,M]，再逐 block 缩放归并。gate/up reduction 融合 SwiGLU，down reduction 写回原 rank 缓冲。

两种投影均报告 jit:gemm:any，显式 user scratchpad=1 B；必须落地的 int32 partial 在 M8/M32/M64 分别为 3,276,800 / 13,107,200 / 26,214,400 B，完整计入耗时。没有在热循环重绑 oneDNN memory descriptors。

此库对照数值检查通过，但整体落后；它的 partial materialization/reduction 成本不可忽略。

## 3. 完整链 wall time

[MEASURED] 两个独立进程、同一最终 EXE；每模式 8 次 warmup、3 轮 × 20 样本。轮内循环变换模式顺序，每次均等待完成。下表为六轮中位数的中位数，单位 µs。模型加载/H2D 不在计时内；每次 replay 内仍重新量化和转码，没有偷偷复用上一请求的 INT8 权重。

| 每专家行数 M | 普通 NF4 G3 | 普通 DPAS | oneDNN block64 | graph NF4 G3 | graph DP4A | graph DPAS |
|---|---:|---:|---:|---:|---:|---:|
| 8 | 287.95 | 484.38 | 566.55 | 273.13 | 237.83 | 346.55 |
| 16 | 385.95 | 512.28 | 667.35 | 365.70 | 364.03 | 358.65 |
| 24 | 432.68 | 524.08 | 729.65 | 408.58 | 421.00 | 445.93 |
| 32 | 526.85 | 529.78 | 748.18 | 506.83 | 487.13 | 374.80 |
| 48 | 678.80 | 563.30 | 839.50 | 630.58 | 646.80 | 470.35 |
| 64 | 827.83 | 584.38 | 920.30 | 775.00 | 766.30 | 408.13 |

普通 DPAS 与 graph DPAS 是相同九个 kernel、相同输入/输出/转码工作，只改变提交方式；原 NF4 和 DP4A 也给了匹配 graph 对照。M32 的收益相对较快的 graph DP4A 仍约 23%，并非只占了对照未使用 graph 的便宜。

M16 的约 2% 余量不够作选择依据，M24 有可重复的回退；不能从这些点写出“M≥16 全走 DPAS”。目前只有 {32,48,64} 是这些条件下有收益的已测点，未测试行数和真实不均匀热桶不包含在结论里。部分普通提交样本 p90 有明显尾部波动，原始离散度保留；没有选择最快一轮报告。

### 提交开销的证据

独立开启 event profiling，普通提交模式的设备阶段中位数如下，单位 µs；不是 graph wall time。

| M / 路径 | 两次量化 | gather + pack | 两次权重转码 | gate/up | down | merge |
|---|---:|---:|---:|---:|---:|---:|
| 32 / NF4 | — | — | — | 250.16 | 141.20 | 19.53 |
| 32 / DPAS | 16.25 | 11.98 | 81.41 | 193.07 | 32.40 | 19.38 |
| 64 / NF4 | — | — | — | 415.42 | 261.72 | 31.72 |
| 64 / DPAS | 16.46 | 16.56 | 82.14 | 196.35 | 51.51 | 31.15 |

DPAS 设备计算已经节约工作，但普通链有九次提交。静态 graph 让调用者一次提交整条依赖链，才把这些收益较充分地兑现到 wall time。这一解释基于同进程匹配对照；没有通过相减不匹配历史样本来推断一个固定 WDDM/UR 延迟下界。

## 4. graph 的有效性与成本

使用 ext_oneapi_limited_graph；固定形状、固定 USM 缓冲地址。每个已测 M，对 NF4、DP4A、DPAS、padded DPAS 都验证了：

1. 原输入下，真实计时所用 replay 路径的 h、FP32 rank down、最终 BF16 output 与普通提交逐字节一致。
2. 等待已有工作完成后，修改输入、token map，并交换两个专家的 device descriptors；同一个 graph 重放仍与普通路径逐字节一致。
3. 恢复原数据后再进行暖态交错计时。没有在 GPU 正读时改 host/device 参数，也没有重新构建 graph 冒充重放。

这证明了“固定形状/地址、更新设备缓冲内容”在该 fixture 内可用，不证明 ragged bucket、cache ticket 重排、多 session 或动态形状的生产集成。

graph 构建/finalize 和首次 replay 单独计时，未混入暖 replay 中位数。例如第二进程 DPAS graph 构建：首个 M8 为 2.256 ms，其余形状约 0.314–0.353 ms；首次重放（kernel 已暖）为 416–590 µs。生产应在执行槽生命周期内复用 graph；若每个热桶重新 finalize，不能套用上表。

## 5. 未确认或负面的变体

- WG8→WG32：四个独立 subgroup 放进一个 work-group，数值通过，但没有一致收益。保留 WG8 基线。
- VNNI 物理列跨度补齐到 32 token：逻辑 M、tile 数和有效输出不变，也没有多算 padding token。该变体未消除 M24/M48 的慢点，M16 有时更慢，不能把非单调形状行为归因为跨度对齐。保留实验，未作默认选择。
- oneDNN block64：语义匹配，但 partial 输出及 reduction 抵消收益。
- 没有重启 CPU expert 计算，没有改变全局 immediate-list/OOO 队列策略，没有因为 graph 可用就推断全模型可直接录成一个 graph。

## 6. 数值、ISA 与回归

小型独立 smoke：M=N=8、K=128，s8 正负值、非恒定的两组 weight/activation scale、置换 token 输出，CPU 检查零失败。真实专家所有已测形状/普通候选也通过以下检查：

- Q8 字节与 BF16 尾数/指数构造的整数 RTE oracle 一致；零块、NaN/Inf raw-bit guard 通过。
- 每候选检查第一/最后 activation 的全部 gate/up/down 值，共 15,360 值；FP64 block64 参考的 max_abs 约 1.33e-7，abs=1e-6、rel=1e-4 内零失败。
- 全部 activation 的 SwiGLU/rank 地址与 ordered merge 检查通过；graph 的全 h/down/output 精确对照通过。

相对原 NF4 的近似误差仍在：DPAS 输出 NRMSE 随 M 约 1.65%–1.68%；abs=0.0002 + rel=0.02 的超限数依次为 7/20,480、12/40,960、18/61,440、29/81,920、43/122,880、56/163,840。未放宽该门槛，未把整数实现正确当成模型质量认证；没有执行该新算术的全模型 teacher-forcing/NLL 或生成验收。

[TOOLCHAIN/MEASURED] 从 VTune 捕获的运行时 zebin 反汇编确认原生 dpas.8x8 (8|M0)，byte 输入、int32 accumulator。各 gate/up kernel 有 8 个 DPAS 静态位置，down 有 4 个（包含编译器生成的指令序列，不等于数学 API 调用数）。WG8/WG32/padded 三种版本均确认；代码不依赖 ESIMD。普通 SYCL gate/up 报告 480 B spill，down 为 0；没有把它写成 zero-spill 成果。

复用 ISA 提取脚本的 --symbol-contains integer_dpas_projection 选项。为处理 Windows 长符号文件名，只在分析副本中缩短 ELF section 名，断言机器码字节未变；原始 zebin、机器码和 ASM SHA256 均保留。

相关 CTest 3/3 通过：AilaQwen4ExpKernelTests、AilaQwen4ExpToleranceTests、AilaQwen4ExpDpasSmoke。四次最终基准/诊断均 exit=0、未触发 watchdog，EXE SHA256 均为 901407CC7CDBBB69485E896C473C9E2C076B3CD8201C64440F4F2B08B1676B7E。

普通 fixture 进程峰值工作集约 193–195 MiB，最低系统可用 RAM >51 GiB。此 fixture 只有两个 resident slots、host pool=0，不能用于证明长文 32 GiB 配置内存达标。本轮全在 9033 上，未做旧驱动 A/B，不能判断或否定此前约 3% TTFT 回退信号。

## 7. 下一步边界

[HYPOTHESIS] 有足够行数的真实专家热桶值得接入可复用 graph，但先完成两道关：

1. 用真实隐藏态和 teacher-forcing/NLL 评估 integer-LUT/Q8 误差，并区分 LUT 误差、输入量化和中间量化的贡献。当前逐点超限不支持默认上线。
2. 将固定 scratch、设备描述符更新、graph 生命周期接到实际 expert bucket/cache ticket；保留稀疏/小桶 NF4 fallback，验证等待/取消/缓存淘汰边界。随后才测整机 prefill、decode、TTFT 和峰值 RAM/VRAM。

R2b 的算子与执行候选已得到实测支持，生产集成与模型质量仍未交付。20 tok/s decode 目标未在本轮测量。

## 8. 重现

~~~powershell
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload dpas -Smoke -TimeoutSeconds 30 -Tag r2b-dpas-smoke-20261001
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload dpas -Tag r2b-dpas-pad-final-20261001
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload dpas -Tag r2b-dpas-repeat-final-20261001
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload dpas -IntegerDeviceProfile -Tag r2b-dpas-device-final-20261001
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload dpas -VtuneInstructions -Tag r2b-dpas-vtune-final-20261001
python perf/qwen4exp/q4e_integer_isa.py --result-dir tmp/r2b-dpas-vtune-final-20261001-vtune --output tmp/r2b-dpas-vtune-final-20261001-isa --symbol-contains integer_dpas_projection
~~~

实际重跑请换 Tag/输出目录，VTune 不能复用既有 result-dir。VTune profile-only 仅运行 M32；普通基准和 event profiling 覆盖六个 M。一次只跑一个 GPU 压力进程，测量时不重建；runner 默认 240s watchdog、8 GiB 可用 RAM 下限。
