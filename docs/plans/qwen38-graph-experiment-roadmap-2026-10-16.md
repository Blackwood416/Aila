# A770 Graph 实验研读 + Aila decode 提交路径落地路线图（2026-10-16）

研读对象：`E:\RiderProjects\A770-Graph-Experiment`（全程只读，未修改其中任何文件）。
用途：为 Aila（`E:\RiderProjects\Aila`）decode 路径的"减少 submit/launch 开销"提供可直接落地的设计参考。

证据标注约定：
`[GE]` = A770-Graph-Experiment 仓库内的文档或原始数据；`[Aila]` = Aila 仓库既有探针/文档
（`docs/plans/qwen38-r1-bottleneck-audit-2026-10-16.md` 简称 B16、`docs/plans/qwen38-r1-decode-submit-path-audit-2026-10-16.md`
简称 A16）；`[推断]` = 本文基于上述证据的推导，未实测；`[待验证]` = 明确未做过的实验。

来源简称（后文引用）：

| 简称 | 文件 |
|---|---|
| GE-R | `A770-Graph-Experiment/README.md` |
| GE-1 | `A770-Graph-Experiment/summary.md`（阶段一结论） |
| GE-2 | `A770-Graph-Experiment/phase2/summary.md`（阶段二结论） |
| GE-CAP | `A770-Graph-Experiment/capabilities.json` |
| GE-ENV | `A770-Graph-Experiment/environment.json`、`phase2/environment.json` |
| GE-BASE | `A770-Graph-Experiment/phase2/baseline_comparison.csv`、`phase2/results/P0_*/measurements.csv` |
| GE-P1 | `A770-Graph-Experiment/phase2/memory_analysis.md`、`results/P1_pointer_process2/measurements.csv` |
| GE-P2 | `A770-Graph-Experiment/phase2/pipeline_analysis.md`、`synchronization_design.md` |
| GE-P3 | `A770-Graph-Experiment/phase2/decode_analysis.md` |
| GE-P4 | `A770-Graph-Experiment/phase2/indirect_decision.md` |
| GE-P5 | `A770-Graph-Experiment/phase2/executor_design.md`、`api_design.md`、`integration_notes.md`、`graph_executor/graph_executor.hpp` |
| GE-T | `A770-Graph-Experiment/phase2/timing_methodology.md` |
| GE-CODE | `graph_experiment.cpp`、`native_graph_probe.cpp`、`mutable_probe.cpp`、`host_parameter_probe.cpp`、`graph_extension.hpp`、`common.hpp`、`phase2/parameters.hpp`、`phase2/async_replay.cpp`、`phase2/decode_microbench/benchmark.hpp` |

---

## 0. 结论速览

1. `[GE]` A770-Graph-Experiment 在 A770/Windows/oneAPI 2026.1/驱动 32.0.101.8724 上把
   "一次录制、多次重放 + 动态参数" 做到了可执行原型：Limited SYCL Graph、Level Zero regular command list、
   固定地址 USM 参数表、GPU Producer + Indirect Dispatch、Mutable Group Count/Size/Offset 全部实测通过，
   两阶段共 632 个配置 / 228.5 万次计时重放、正确性错误 0（GE-1、GE-2）。
2. `[GE]` 官方 Full SYCL Graph / updatable graph、Kernel Arguments/Instruction 更新在当前栈**不可用**
   （GE-CAP）；动态指针与标量只能走"kernel 自带参数表 ABI"这条绕行路（GE-1 §功能替代）。
3. `[GE]` 性能量级（64 个 ~5 μs kernel 的链）：阶段一 Queue 1500.9 / Limited Graph 252.8 / 原生 L0 list 134.1 μs；
   阶段二复测 474.6 / 253.9 / 135.7 μs（GE-1、GE-BASE）。**原生 regular list 是三者中最快的**，
   Graph 相对它没有优势；长链短 kernel 收益最大，kernel 变长（20/100 μs）收益收窄到 1.7×/1.1×（GE-1）。
4. `[GE]` 动态参数表的可行组合：64 B Host USM 表 + 原生 list，E2E p50 141.5 μs（Shared 268.7 μs，depth 64）；
   标量/指针/动态 n/指针随机交替全部逐元素验证通过（GE-BASE、GE-P1）。
5. `[Aila]` Aila 现状：每 token 984–987 次提交（GDN 层 19、QSA 层 25；A16 §1d），块计时口径 18.7 ms/token；
   合成探针里"每层一张静态图" 12.7 vs 14.6 ms/token（−13%），同一探针指出 DG2 不支持 kernel 参数更新、
   需要设备侧参数块（B16 §3.3、A16 §0.6）。
6. `[Aila]` 但 A16 §2 的关键保留条件：18.7 ms 是**块计时**，~30 μs/kernel 的 5× 膨胀还没归因
   （候选：wait 打断 L0 动态批、depends_on 提交、functor 宽负载、块内胶水），VTune 未跑。
   若膨胀主要在 waits/胶水而非 submit 本身，图重放的收益上限就是 −13% 探针那一档，而不是 18.7 ms 全额。
7. `[推断]` 落地主线应是 **Variant B：每类层一张图 + 设备侧参数表**（图数量与 48 层/会话数解耦），
   而不是"每层一张图、指针按值捕获"的 Variant A；机制证据来自 GE 的参数表实验与 Aila 自己的
   `tests/models/Qwen4ExpFrontGraphProbe.hpp` 录制范式。
8. `[推断]` 最短路径不是先写图，而是先用半天量级的 E0–E3 实验回答两件事：
   (a) 参数表 ABI 能否让真实 kernel 保持逐位一致；(b) 图重放在真实 Aila 形状（20 submit + 每层 wait）下
   是否仍有 −13% 量级收益。两者都通过才值得进入正式实现。

---

## 1. 该仓库到底测了什么、结论是什么

### 1.1 仓库形态（GE-R、GE-CODE）

- 阶段一（A–F）：环境/能力探测（`device_probe.cpp`）、执行路径矩阵（B：SYCL Queue / Limited Graph /
  原生 regular command list；C：USM 动态参数表；D：Indirect Dispatch；E：统一性能矩阵；F：Mutable 几何）。
- 阶段二（P0–P5）：计时口径重做与基线复现、Host USM 参数表、异步 Replay/双缓冲、
  真实 Decode 算子微基准（RMSNorm/Residual/RoPE/GEMV/Attention/组合链）、Indirect 收益边界、
  最小 `GraphExecutor` 原型（`phase2/graph_executor/graph_executor.hpp`）。
- 构建：`build.cmd` 调完整 oneAPI `setvars.bat`，用 `icx` 把 `kernels.cl`/`parameter_kernels.cl`/`operators.cl`
  编成 SPIR-V，`icx -fsycl /std:c++20` 编 C++ 驱动，链接 L0 SDK 1.28.2 的 `ze_loader.lib`。
  Kernel 通过 `zeModuleCreate` + `sycl::make_kernel_bundle<L0, executable>` 互操作在三条执行路径间共用
  （同一 context/module/几何/数据）。
- 运行器 `run_experiment.ps1` 保证一次只有一个 GPU 实验进程，保存 stdout/stderr/退出码/dump/显示驱动事件。
- 规模：阶段一 312 配置 × 5×1000 = **156 万次**计时重放，错误 0（GE-1）；阶段二合并 **320 配置、72.5 万次**
  （70 个 5×1000 + 250 个 3×500），验收错误 0（GE-2）。

### 1.2 结论与关键数字

**能力位（GE-CAP，实测原始值）：**

| 能力 | 本机结果 | 原始证据 |
|---|---|---|
| Limited SYCL Graph | 支持 | `limited_graph=true`，B/C/D 重放全部通过 |
| Full SYCL Graph / updatable | **不可用** | `full_graph=false`；updatable finalize 抛 `Device does not support Command Graph update` |
| Mutable Kernel Arguments | **未公布** | `mutable_command_flags_raw=62 (0x3e)`，bit0（Kernel Arguments）未设置 |
| Mutable Kernel Instruction | **未公布** | bit6 未设置 |
| Mutable Group Count / Size / Global Offset | 支持 | bits1/2/3；各 5000 次更新+重放通过（`results/F/correctness.json`） |
| 原生 Record & Replay Graph | 扩展存在，flags=0 | `zeCommandList*GraphCaptureExp` 单节点流程实测 5000 次通过，但**未公布能力** |

**性能（阶段一 E2E p50，单位 μs；64 层、每组 5×1000；GE-1）：**

|层数|目标/实测单 kernel|Queue|Limited Graph|原生 L0|Graph / L0 相对 Queue|
|---:|---:|---:|---:|---:|---:|
|1|5 / 7.19|74.5|136.7|51.5|0.54× / 1.45×|
|4|5 / 7.19|158.9|140.6|55.3|1.13× / 2.87×|
|16|5 / 7.19|457.7|131.0|71.2|3.49× / 6.43×|
|64|5 / 7.19|1500.9|252.8|134.1|5.94× / 11.19×|
|64|20 / 17.60|1539.6|921.3|808.7|1.67× / 1.90×|
|64|100 / 93.33|6313.1|5883.2|5774.1|1.07× / 1.09×|

- 同规模 CPU Submit p50：Queue 1031.1 / Graph 29.9 / L0 6.5 μs（GE-1）。注意这是 API 调用区间，
  可能包含 API 内部阻塞，不能当纯驱动开销。
- 单节点（depth=1）Graph 反而比 Queue 慢（136.7 vs 74.5）——图适合"长链、短 kernel"。

**阶段二复测（GE-2、GE-BASE，64 层 static，单位 μs）：**

- Queue 474.6 / Graph 253.9 / L0 135.7（p50）；CPU Submit p50 329.0 / 31.5 / 6.9。
- 早前阶段一可执行文件重跑同配置得到 Queue 523.35 μs —— 与阶段一 1500.9 差 2.9×，**该漂移未解释**（GE-2 §P0），
  因此跨阶段相对 Queue 的加速比不可直接沿用；Graph 与 L0 本身复现良好（252.8→253.9、134.1→135.7）。
- 动态表（depth 64）：Shared 表 L0 E2E 268.7 μs（其中 CPU 写表 p50 ~70 μs）；
  Host 表 L0 141.5 μs（CPU 写表 ~0.1 μs）（GE-BASE、GE-1）。

**参数表机制（GE-P1、GE-CODE）：**

- 16/64/256 B、Host/Shared/Device、一次/重复读、多 kernel 共用、不同块偏移、sequence-only、
  交替/随机指针全部通过；输入输出始终 Device Allocation，每次验证两份完整输出+未选中缓冲+哨兵。
- 64 B 随机指针 + 动态 n 第二确认进程 E2E p50：**Host 79.5 / Shared 202.5 / Device 251.8 μs**；
  Device 表的 copy+wait 约 175 μs，是它最贵的一环。
- 未验证（README 明确保留）：双缓冲并发协议、参数表内浮点、跨进程/跨队列流水化、
  "Host 表 + 指针切换"组合（指针切换只在 shared/device 表上验证）。

**异步/多流（GE-P2）：**

- 两个**相互独立**请求流 + 双 Slot（各自独立表/list/Fence）吞吐 1.62× / 1.60×（两次确认进程）；
  单条自回归流只有 +0.5% / +4.6%，低于 10% 门槛。同一 Slot/list 禁止并发重入；
  Slot 复用前必须 Fence 等待并 reset，完成后才能覆盖参数表。

**真实 Decode 微负载（GE-P3，七 kernel 链，E2E p50 μs）：**

| H / 类型 | Queue Shared | Graph Shared | L0 Shared | L0 Host | Queue→Host |
|---|---:|---:|---:|---:|---:|
|1024 FP32|410.5|356.6|260.7|**138.1**|2.97×|
|4096 FP32|747.1|729.1|656.2|535.6|1.39×|

- H1024 时后端收益明显（Native/Queue 1.57×），H4096 计算占比高、收益收窄；
  Host 表又额外把 E2E 压到 Shared 表的一半左右。
- 独立 profiling 的 H1024 Native span：Shared 89.4 / Host 92.3 / Device 88.8 μs（带插桩，
  表读取的额外成本被 E2E 收益抵消，但不代表无插桩纯访存延迟）。

**Indirect（GE-P4）：**

- Attention 定向比较（KV=1 第二确认进程，p50）：Direct 76.6 / Mask 80.0 / Indirect 80.0 / CPU 计数+Queue 131.0 μs；
  没有相对 mask 的稳定 ≥10% 改善。结论：CPU 已知 KV 时优先直接或固定范围+mask；Indirect 保留为功能选项。

**Mutable 几何与原生图（GE-1、`results/F`、`results/native_graph`）：**

- Group Count / Size / Offset 的 update+close 中位耗时 1.2 / 1.4 / 0.6 μs，各 5000 次重放通过。
- 原生 Record & Replay：capture/instantiate/append 全部返回 0x0、5000 次重放通过，但 `flags=0`，
  仅覆盖 shared USM + 1 个 work kernel + barrier 的串行重放；明确不提升为"受保证支持"。

**构图成本（GE-1 §构图成本，B 组 shared 短 kernel 64 层）：**

- Graph：capture 804.7 μs + finalize 723.7 μs；L0 list：capture 477.2 μs + close 1.1 μs。一次性成本，
  未摊入稳态。进程初始化约 173 ms（其中 module JIT ≈1.29 ms）；完整构建约 67.6 s。

### 1.3 已验证事实 vs 未完成的半成品

**已验证（可直接引用的机制）**

1. Limited SYCL Graph 一次录制多次重放，固定地址参数表在每次重放读取新内容（标量、sequence、输入/输出指针、动态 n）。
2. 原生 L0 regular command list 的 CPU 提交显著便宜（depth 64 时 ~6.9 μs/replay 的 API 区间）。
3. GPU Producer + barrier + Indirect Consumer 的设备侧动态工作量可用（真实调度数量回读验证）。
4. 小 Host USM 参数表 + 原生 list 是实测最省的动态参数组合（141.5 vs 268.7 μs，depth 64）。
5. 独立双流 + 多 Slot 的安全复用协议（Fence + 表覆盖次序）正确且吞吐 1.6×。
6. 一个 128 B ABI 的独立 `GraphExecutor` 原型：单实例单 in-flight、Fixture lease 防重入、
   同步/异步接口、生命周期测试（4 轮析构后 8 个参数/备用输出分配全部释放）。

**半成品 / 未完成**

1. 无 Full SYCL Graph，无 Kernel Arguments/Instruction 更新——**参数表不是通用替代**：未修改的库 kernel
   不会自动适配，需要 kernel 侧编程约定（GE-1 §对自研推理引擎的意义）。
2. 原生 Record & Replay flags=0，仅单节点+barrier+shared USM 串行重放；多节点依赖、buffer copy 混合、
   Event 依赖、实例隔离、重复实例化全部未测；P6 阶段未扩展（GE-2 §数据与限制）。
3. 双缓冲并发：阶段二只验证了"独立流 + 独立 Slot"；同表并发更新、host/GPU 同时访问、跨队列流水化未验证。
4. 真实模型负载：没有加载任何真实权重、没有 BF16、没有 batch 2/4、没有库 kernel 混排、没有多 host 调度线程；
   Executor 是依赖实验 Fixture 的头文件原型，不是 SDK（GE-P5、`integration_notes.md`）。
5. 无单元测试框架、无 SHA256/文件哈希校验；结果合并靠运行状态+正确性+样本完整性检查。
6. 数据可疑点保留：阶段一 vs 阶段二的 Queue 基线漂移（1500.9 / 474.6 / 523.35）未解释；
   176 个配置 CV>5% 标 NOISY；C 阶段一次超时、一次脚本竞态、一次实验重叠，均有补跑记录。
7. 合成算子局限：`kernels.cl` 是 uint32 仿射递推（mod 2^32），`scale` 实为加性整数；
   只验证机制，不代表推理模型性能。

### 1.4 与 Aila 侧探针数字的关系

- Aila 合成探针（B16 §3.3 `[Aila]`）：48 层 × 20 小 kernel + 每层一次 host wait = **13.8–14.6 ms/token**
  （设备忙只有 6.2–6.5 ms）；只保留末尾一次 wait = 8.3 ms/token；空 kernel+wait 往返 59 μs；
  **每层一张静态图 12.7 vs 14.6 ms（−13%）**，并注明"DG2 不支持 kernel 参数更新，需要设备侧参数块"。
- 生产账本（B16 §3.2）：内核提交块计时 18.7 ms/token（hc 6.6 + linear 3.3 + full 2.8 + shared 2.8 +
  front 1.8 + topk 0.9 + ple 0.25 + head 0.16）；A16 §2.1 按提交清单折算为 **12.9–35.5 μs/submit**，
  对照裸探针 5.9–7.4 μs。
- GE 仓库没有跑过 Aila 的"每层 wait + CPU join"结构；Aila 也没跑过 GE 的参数表/Executor 原型。
  两边是互补证据：GE 提供机制与常数，Aila 提供本机真实形状与收益上界。

---

## 2. 技术细节

### 2.1 用了哪个 API

| 用途 | API | 出处 |
|---|---|---|
| 静态图重放（主矩阵） | `sycl::ext::oneapi::experimental::command_graph<graph_state::executable>`，设备 aspect `ext_oneapi_limited_graph`；重放 `q.submit([&](handler& h){h.ext_oneapi_graph(*g);})`（Aila 探针另用 `q.ext_oneapi_graph(*g)` 简写） | `common.hpp`、`graph_experiment.cpp` `Pipeline`；Aila `Qwen4ExpFrontGraphProbe.hpp` |
| 原生列表 | SYCL-L0 互操作：`zeModuleCreate` → `sycl::make_kernel_bundle<L0,executable>`；`zeCommandListCreate`/`AppendLaunchKernel`/`AppendBarrier`，`zeCommandListClose` 一次，`zeCommandQueueExecuteCommandLists` 重放 | `graph_experiment.cpp` `Runtime`/`Pipeline` |
| 动态工作量 | `zeCommandListAppendLaunchKernelIndirect` + GPU Producer 写 `ze_group_count_t` | `kernels.cl` `producer`、D 阶段 |
| Mutable 几何 | `zeCommandListGetNextCommandIdExp` + `zeCommandListUpdateMutableCommandsExp`（`ZE_experimental_mutable_command_list`，扩展版本 1.1），**更新成功后必须再次 `zeCommandListClose`** | `mutable_probe.cpp` |
| 原生 Record&Replay | `zeDriverGetExtensionFunctionAddress` 动态取 `zeCommandListBeginGraphCaptureExp` / `EndGraphCaptureExp` / `InstantiateGraphExp` / `AppendGraphExp` / `zeGraphDestroyExp`；SDK 1.28.2 缺声明，`graph_extension.hpp` 手写最小 ABI（**不是**改能力位） | `native_graph_probe.cpp`、`graph_extension.hpp` |
| GPU 时间戳 | `zeEventQueryKernelTimestamp`、`zeCommandListAppendWriteGlobalTimestamp`、`ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2`（frequency 19.2 MHz，global/kernel 有效位 36/32，tick 0.052 μs） | `phase2/p0.cpp` `timestamps`、`timestamps_confirm/timestamp_validation.json` |

注：仓库**没有直接调用 UR command-buffer API**；SYCL Limited Graph 由 SYCL/UR 在内部落到 L0 命令缓冲。
官方 Full/updatable SYCL Graph 通道在这台机器上不可用（GE-CAP）。

### 2.2 工具链 / 驱动（GE-ENV）

| 组件 | 版本/路径 |
|---|---|
| GPU | Intel Arc A770 16 GB（device id 0x56a0，vendor 0x8086，显存 16 706 793 472 B） |
| OS | Windows 11 专业版，10.0.26200 |
| 显卡驱动 | 32.0.101.8724；SYCL L0 driver string 1.15.37669；Level Zero API 1.15 |
| 编译器 | Intel oneAPI DPC++/C++ 2026.1.0.20260617（`icx` + `llvm-spirv`） |
| Level Zero SDK | `C:\Program Files\LevelZeroSDK\1.28.2`；实际加载 `C:\WINDOWS\SYSTEM32\ze_loader.dll`（loader 文件版本 1.28.2） |
| 提交环境 | 未设置 `SYCL_UR_USE_LEVEL_ZERO_V2`/`UR_L0_USE_IMMEDIATE_COMMANDLISTS` 等；`ONEAPI_DEVICE_SELECTOR=level_zero:gpu` |
| 驱动扩展 | `ZE_experimental_mutable_command_list` v1.1、`ZE_experimental_record_replay_graph` v1.0、`ZE_experimental_immediate_command_list_append`、`ZE_experimental_global_offset` 等（GE-CAP 全列表） |

这套环境与 Aila 生产/探针文档记录的一致（B16：A770、驱动 32.0.101.8724、oneAPI 2026.1）。

### 2.3 DG2 的限制与绕过（GE-CAP、GE-1、GE-CODE）

| 限制 | 实测表现 | 仓库给出的绕过 |
|---|---|---|
| Kernel Arguments 不可在线更新（bit0=0） | 无法在重放前改写已录制 kernel 参数 | **固定地址 USM 参数表**：kernel 只捕获表地址，动态值由表内容提供；表可放在 Host/Shared/Device USM |
| Kernel Instruction 不可更新（bit6=0） | 不能换 kernel 指令 | 预建拓扑变体（每层/每分支一张图） |
| Full/updatable SYCL Graph 不可用 | `full_graph=false`；updatable finalize 报错 | 用 Limited Graph 或原生 regular list；`graph_state::modifiable` 录制 + `finalize()` |
| 原生 Record&Replay flags=0 | 驱动不承诺；SDK 1.28.2 中也没有正式声明 | 仅作研究；**不进入默认后端**（GE-2 明确） |
| Immediate command list 不占优（Aila 侧实测更差：41.5 ms/token） | — | 保持 regular list（`UR_L0_USE_IMMEDIATE_COMMANDLISTS=0`） |
| 单实例 command list 不可并发重入 | 同 Slot 并发执行未定义/被拒 | 每 Slot 独立 list/表/Fence；一实例一 in-flight（GE-P2、GE-P5） |
| 动态工作量需 GPU 侧决定时才划算 | Indirect 无稳定 ≥10% 收益 | CPU 已知 n 时用固定最大范围+mask（GE-P4） |

### 2.4 参数表机制细节（GE-CODE、GE-P1、GE-P5）

- 阶段一定义 `struct Params { const uint32_t* input; uint32_t* output; uint32_t* launch_report; uint32_t n, steps, scale, sequence; }`（40 B），
  kernel 端只读该表；每次重放前更新表内容（shared 直接写、device 走 memcpy+wait）。
- 阶段二 ABI v1：`struct alignas(64) Block { uint32_t sequence, scalar, n, version; uint64_t input, output, reserved[4]; }`（64 B，
  input 偏移 16、output 24）；16 B 变体只含标量头，不承载指针；256 B 变体四块、多 kernel 读不同偏移。
  Kernel 用 `__global const volatile` 读表，防止编译器把加载折叠掉（`parameter_kernels.cl`）。
- Executor ABI v2（`decode_microbench/benchmark.hpp`）：128 B、`alignas(64)`，
  `sequence/h/d/kv/position/dtype/max_kv/version` 位于 0/4/8/12/16/20/24/28，12 个设备地址从字节 32 起；
  `submit()` 先 `memcpy` 表（host 表）或 memcpy+wait（device 表），再异步提交，返回 ticket；
  `ready()` 只查询不释放，`wait()` 才 reset Fence/释放 lease。
- 同步要求（GE-P2 `synchronization_design.md`）：覆盖参数表前必须确认上一轮 GPU 已读完（上一轮完成后写，或 Fence 保护的 Slot 复用）；
  所有被表引用的分配必须活到 Fence 完成；同一流的下一 Token 提交前必须等前一次完成（单流不允许伪重叠）。
- 表大小在 16/64/256 B 范围内没有明显延迟拐点（61.4/61.4/61.3 μs Host 表，GE-P1）——因此 64–128 B 足够。

### 2.5 计时口径与已知陷阱（GE-T、GE-1、GE-2）

- `CPU Submit` 是提交 API 调用区间（可能内含阻塞）；`End-to-End` = 参数更新 + 提交 + 等待；验证/CSV/初始化/JIT/分配都在区间外。
- Graph `capture/finalize` 与 list `close` 是一次性成本，单独记录，不摊入稳态。
- 无插桩 GPU span 标 `UNAVAILABLE`；带 Event 插桩的结果（首尾 Event / 每 kernel Event）另计成本
  （64 层：无 Event E2E 135.4、首尾 Event 137.3/span 92、每 kernel Event 245.3/span 183 μs）。
- 统计规范：p50/p90/p99、每组中位数、CV；CV>5% 标 NOISY；路径顺序在同一进程内轮换。
- **不要跨阶段比 Queue 比值**：Queue 基线在两个阶段间从 1500.9 漂到 474.6～523.35 μs 且原因未定（GE-2 §P0）。

---

## 3. 对 Aila 的适用性分析

### 3.1 Aila decode 现状（证据：A16、B16、`src/core/Context.hpp`）

1. Compute queue 是 **in-order**：`sycl::queue& queue() { async_errors_->Rethrow(); CheckTrace(); return q_; }`
   （`src/core/Context.hpp:57`），构造时带 `sycl::property::queue::in_order()`，profiling 由构造参数可选开启。
2. 提交密度：**984–987 submissions/token = 每层 19（GDN）/25（QSA）**，48 层 = 36 GDN + 12 full（A16 §1d）。
3. 结构：每层 `[6 submits] → [3/9 submits] → D2H → front → top-k → (host wait) → shared → (CPU route) → 4 MoE submits → (CPU join) → accumulate → shared add`；
   每层 **2 个 host SYCL wait、4 个带显式 `depends_on` 的提交**（A16 §2.4）。
4. 提交成本块计时 18.7 ms/token；估算 per-submit 12.9–35.5 μs（裸提交 5.9–7.4 μs）；5× 膨胀未归因（A16 §2.2，候选 L1 batch-cold / L2 depends_on / L3 functor 宽 / L4 get_pointer_type / L5 块内胶水）。
5. 每 kernel 参数变化面（A16 §4 表）：
   - 每 token 变：`start_pos`、`cached_len`、`num_complete_blocks`、`num_active_blocks`；
   - 每 token 变（host 分类结果）：`resident_selected_[]`、`hit_ranks[]`、miss 列表、`MoeExpertSlotDev` 描述符数组；
   - 每层固定：权重、`buf_.x/y/hidden4`、`moe_h`、`resident_down`、shared 缓冲等基地址；
   - 路由 id/权重已由 top-k kernel 写在设备侧（host 只做命中分类）。
6. `count==1` decode 路径上**没有 oneDNN primitive 执行**（NF4/bf16 GEMV 全走自研 ops kernel；A16 §2.3），
   所以"图只捕获自研 SYCL kernel + memcpy"这件事在 decode 上成立；prefill 的 oneDNN/大内核不在本路线范围。
7. 已有可复用的图基础设施：`tests/models/Qwen4ExpFrontGraphProbe.hpp`（对 3 个 conv head 做
   `begin_recording` → 现有 `Access::Front()` → `end_recording` → `finalize`，逐位对比 state/history/output/router，
   并输出 graph vs 直连的 ms_per_front）、`Qwen4ExpIntegerQualityAcceptance.hpp`/`Qwen4ExpIntegerMoeProbe.hpp`
   （`ext_oneapi_graph` 重放/流式 replay 已在测试里跑过）。

### 3.2 "每层一张图 + 设备侧参数块"的可行性

**结论：机制上可行，但推荐的目标形态不是"每层一张图"，而是"每类拓扑一张图 + 参数表"。**

- Variant A（最小改动）：把现有 ops 调用原样录进图（`begin_recording` 包住 `Access::Front()`），指针按值捕获。
  由于权重/KV/激活指针每层不同且每会话不同，必须 **每层 + 每会话** 一张图（36/12 层 × session）。
  可行但图实例数与内存线性膨胀，且拓扑分支（env flag、索引器每 4 token、GDN fallback）都要翻倍。
- Variant B（GE 验证过的机制）：kernel 只接受固定表地址，动态指针/标量从表里读；
  图数量 = 拓扑种类（GDN 层、QSA 层、head/embed 等），与层数、会话数解耦；
  每层重放前写表（Host 表 CPU 写 ~0.1 μs；GE-P1），用完后按 GE-P2 的 Fence/顺序协议覆盖。
  这一形态同时解决 Aila 特有的"多会话共享同一 in-order queue"问题：图实例可跨会话共享，
  只要参数表按会话/Slot 分开。
- 层结构上的天然切分（沿用 A16 §4 的 S1/S2/S3 划分）：
  - **S1（GDN attention 子链，6 submits → 1 图）**：无 host 回读，只需把 `start_pos`/长度类标量放进表；36/48 层适用。
  - **S2（MoE 子链，4 submits）**：descriptor 数组与 rank 数组设备化；图边天然替代 4 个 `depends_on` 提交。
  - **CPU cold join 是硬切点**：一层至少两张图（或一张图 + host 段 + 一张图）；不要试图把 CPU join 包进图。
  - **S3（整层单图）** 只有在前两项落地且有"不需要 host 在中间"的冷专家策略后才值得尝试（A16 §4 同样结论）。
- 每层 wait 不会被图消除：GE 的参数表同步要求与 Aila 现有"每层 2 wait"节奏兼容（写表前需确认前一轮读完），
  但若目标是同时砍 wait，必须搭配 A16 §5 的 ready/release 类修复（例如 `AILA_Q4E_JOIN_RELEASE`、
  `AILA_Q4E_CPU_DEFER_ARENA_WAIT`），这两条与图路线互为补充而非替代。
- [推断] 提交成本算术（用 GE 常数）：Aila 一层 20 submits 若全部在图内，重放一次是 **1 次 `ExecuteCommandLists`**；
  GE 在 depth 64 的 L0 重放 API 区间 ~6.9 μs、Limited Graph ~31.5 μs（GE-BASE）。
  对照 Aila 生产 ~304 μs/层（13.8–14.6 ms ÷ 48），**上界**是每层降到几十 μs；但该上界只有在
  5× 膨胀确实来自 submit 频次时才成立（A16 指出也可能来自 waits 或块胶水，届时收益会显著低于此）。

### 3.3 需要改什么（落地清单）

1. **参数表 ABI（kernel 侧）**：为参与图的每个自研 kernel 增加"表指针"入口或新增表读变体，
   把 `start_pos`、`cached_len`、`num_complete_blocks`、`num_active_blocks`、权重/KV/激活/输出指针、
   descriptor 数组地址放进表；表布局版本化（参考 GE Executor ABI v2，128 B、64 B 对齐）。
   涉及 `src/ops/HyperConnectionOps.cpp`、`Bnb4BitLinear.cpp`（`packed_nf4_gemv_bf16*`、`bf16_gemv_bf16`）、
   `GatedDeltaNetOps.cpp`、`QsaIndexerOps.cpp`、`AttentionOps.cpp`、`ElementwiseOps.cpp`、`MoeExpertOps.cpp`。
   注意：不改数学表达式，只改"参数从哪里来"，才可能保住现有逐位门。
2. **录制钩子**：在 `ForwardImpl` 的层循环里加 env 门控的 `begin_recording/end_recording` 分支
   （范式照抄 `Qwen4ExpFrontGraphProbe.hpp`），把 S1/S2 子链录成图；录制期间必须避开 host 同步
   （D2H wait、event wait、`ctx.synchronize()`）。
3. **图缓存与失效**：按 (拓扑种类, 关键 env flag, h/d/KV 上限, dtype) 建缓存；env/形状变化即失效重建。
   初始化阶段一次性构图（GE 的 capture 量级：64 kernel 约 0.8–1.5 ms，48 层图组预计数十 ms，[推断]）。
4. **同步协议**：沿用 GE-P2——每 Slot 独立表+Fence，复用前 `zeFenceHostSynchronize` + reset；
   图重放返回的 `sycl::event` 作为后续 D2H/下一段的依赖；明确"最后读者完成后才可写表/释放分配"，
   与 Aila 既有取消不变式（取消请求先 drain GPU 再释放 USM）合并。
5. **事件依赖**：把 4 个 `depends_on` 提交改由图内边表达；跨图/跨 host 段用 SYCL event 串接
   （`q.ext_oneapi_graph(g)` 返回 event）。需要先验证 Limited Graph 录制状态下
   `cgh.depends_on(event)`/`ext_oneapi_submit_barrier` 的录制行为（[待验证]，见 §5）。
6. **数据设备化**：`resident_selected_`/`hit_ranks_` 等 host 分类数组若要进图，需要设备侧生成或由表提供；
   MoE 4 个提交（gate_up/down/accumulate/H2D 64 B）在 S2 图内需要 descriptor 数组设备化（A16 §4）。
7. **可观测性**：`AILA_Q4E_DECODE_STAGE_EVENTS`（`ext_oneapi_submit_barrier` + profiling queue）
   与图录制/重放的兼容性需单独验证；图重放自身的 profiling 需用图属性或在图内节点上取时间。
8. **回退开关**：图路径必须可一键回退到现有 Queue/L0 路径（GE-P5 的 Queue 回退范式），
   且所有图相关代码不进入默认配置，直到位精确门通过。

### 3.4 风险点

| # | 风险 | 依据 / 说明 |
|---|---|---|
| R1 | 收益被 waits/胶水吞掉 | A16 §2.4：13% 探针就是"图只去掉 per-submit host 成本、不去掉 wait/join"的证据；若 VTune 显示膨胀在块胶水（L5），图收益更小 |
| R2 | 位精确性 | 参数表变体改动 kernel 参数来源；Aila 采纳门槛是 SSE content sha 逐位一致 + A/A ≤1.5%（A16 §5、`qwen38-r1-fused-regression-analysis` 的教训） |
| R3 | 录制期 host 同步 | 录制中若发生 D2H+wait、event wait、`ctx.synchronize()`，捕获可能失败或产生错误图；需要逐一分类 Aila 每层 2 个 wait 的位置并移出图外 |
| R4 | 事件依赖录制 | 4 个 `depends_on` 提交、`dev_mark()` 的 `ext_oneapi_submit_barrier` 在 capture 下的语义未验证（[待验证]） |
| R5 | profiling | 生产可选 profiling queue（`Context` 构造参数）；图重放+profiling 的兼容性与开销未测（GE 明确"无插桩 GPU span UNAVAILABLE"） |
| R6 | prefix cache / 会话差异 | 前缀缓存的命中/未命中路径会走不同 kernel 序列；图必须按实际拓扑建，缓存命中快路径要计入；KV 指针每会话不同，只能用 Variant B 表指针解决 |
| R7 | 多会话并发 | 连续批处理下多个会话交替在同一 in-order queue 提交同一张图实例：GE 的 1.6× 多 Slot 吞吐证据来自**原生 list**（GE-P2），同一张 executable graph 的并发重放与跨会话共享未测（[待验证]） |
| R8 | 内存/账本 | 每个 executable graph 自带命令缓冲；48 层 × 会话数 × 图实例的显存未量，Aila 有 device_budget_ 账本与 VRAM admission，需要先测后算（GE 未测图显存） |
| R9 | 拓扑分支多 | env flag（`DECODE_FUSED_HC`、`MOE_SHARED_FUSED`、`CPU_DEFER_ARENA_WAIT`…）与每 4 token 的索引器压缩都会改拓扑，图缓存 key 容易漏项 → 建议把"是否可图化"作为显式声明 |
| R10 | 取消/释放 | 图引用的 USM 在重放期间必须存活；取消请求、会话销毁、实例重建时的 drain 规则要与现有"never free while kernels may still reference"不变式对齐（AGENTS.md §6） |
| R11 | oneDNN/第三方 kernel | decode count==1 无 oneDNN primitive（A16 §2.3），但 prefill/未来路径有；图路线只覆盖自研 kernel，不要在未分类路径上开启 |
| R12 | 原生 flags=0 的诱惑 | `zeCommandList*GraphCaptureExp` 单节点能跑但不承诺；GE 自己把它排除在默认后端外（GE-2）——Aila 不应把它作为主方案 |

### 3.5 预期收益边界（[推断]，标注为未实测）

- 上界来自 Aila 探针：同一提交/等待节奏下图重放 −13%（14.6 → 12.7 ms/token 的探针循环）；
  对应到 55 ms/token 的真实 token，若结构不变，单这一项约 −1.9 ms/token（−3.5%），与 B16 杠杆表 #3
  "估 −5…−10 ms/token（融合 + 精简提交 / 设备侧参数块 + 静态图）"相比是保守下界。
- 若同时按 A16 §5 先做 wait/release 类修复（已实现 flag），可叠加；图与 release 类修复解决的是不同成本项。
- 若 5× 膨胀最终被 VTune 判为"纯 UR/L0 submit 自时间"，则每层 20 submits → 1 次重放的收益可放大到
  GE 证明的量级（每 replay 6.9–31.5 μs API 区间 vs 20×~30 μs），但这一点当前**没有证据**。

---

## 4. 最小验证实验计划（半天到一天，尽量不碰 Aila 主源码）

原则：所有新代码放 `tmp/` 或 `perf/qwen4exp/`（新文件），Aila `src/` 不动；
复用 A770-Graph-Experiment 的探针与 Aila 现有测试范式；一次只跑一个 GPU 实验。

### E0（~15 分钟）：环境与探针复现

1. `cd E:\RiderProjects\A770-Graph-Experiment && cmd /c build.cmd && cmd /c phase2\build.cmd && cmd /c phase2\probe.cmd`
2. `.\run_experiment.ps1 -Program phase2 -ExtraArguments '--phase baseline --depth 64 --iterations 500 --repeats 3' -Output tmp/graph-repro`

**成功判据**：`limited=true`、`full=false`、`mutable flags=0x3e`；static Shared 64 层
L0 E2E p50 在 135.7±15% μs、Graph 253.9±15% μs 内（GE-BASE）；机器与 2026-10-14 无实质差异。
若不一致，先记录环境差异再谈后续。

### E1（2–4 小时）：Aila 形状的合成回归（先回答"−13% 是否可复现"）

1. 复用 Aila 探针 `tmp/q4e-bottleneck-audit-20261016/submit_probe.exe`（B16 §3.3 已测 12.7 vs 14.6），
   加两臂：
   - 臂 G1：每层一张 Limited Graph（20 个空转/微 kernel），每层 wait —— 复现 −13%；
   - 臂 G2：同 G1，但每层重放前改 Host USM 参数表（指针交替 + 标量 + sequence），验证"参数表 + 图"组合。
2. 顺带用同一代码量化：48 张图的 capture+finalize 总时间、device/host 分配增量
   （`zeMemGetAllocProperties` 前后对比或 Sysman）、每张图的常驻字节数。

**成功判据**：G1 复现 ≤14.6 ms 且相对普通臂 ≥10%；G2 正确性逐位一致、重放提交成本相对 G1 恶化 <2 μs/层；
48 图一次性构图 <100 ms、常驻增量 <50 MB（数值可调，但必须记录，作为 Variant B 图数量的预算）。
**停止规则**：G1 复现不出 −13% → 先修探针/环境，不要进入 E2。

### E2（2–3 小时）：真实 kernel 子链录制（利用现有 FrontGraphProbe）

1. 直接运行 Aila 现有容忍度测试的 front graph 探针（`tests/models/Qwen4ExpToleranceTests.cpp` 的
   `test_front_graph` 路径，或复制其代码到 `tmp/` 探针），确认 3 张 per-head 图的
   `[FRONT GRAPH CHECK] ... bitwise` 全部通过，并读取 `[FRONT GRAPH TIME] graph=0/1 ms_per_front`。
2. 若 graph 臂相对 direct 臂收益 <10%，说明该子链不是提交受限，记录后改选 S1（6 提交）或 S2（4 提交）作为目标。

**成功判据**：graph 臂输出与 direct 臂逐位一致；graph 臂 ms_per_front ≤0.6× direct 臂（提交受限才有意义）；
否则该子链不转化，只保留 S1/S2 候选。

### E3（2–3 小时）：参数表 ABI 的真实 kernel 变体 + 位精确门

1. 在 `tmp/`（或 `perf/qwen4exp/`）新增一个探针，把 **一个** 真实 kernel（建议
   `packed_nf4_gemv_bf16` 或 `hc_decode_down_inject`）复制/包装成"从 128 B 表读指针与标量"的变体，
   不修改 `src/`；用 Aila 现有 fixture（真实权重/激活）对比原 kernel 输出。
2. 校验：逐位一致（memcmp/hash）、Host 表写成本（预期 ~0.1 μs/层）、图重放 36 层正确。

**成功判据**：位精确通过；表写入不引入新的 host 同步；图重放 1000 次无错误。
**停止规则**：若位精确无法达成（例如编译器对表读的优化/量化差异），立即停止 Variant B，回到 E2 的 Variant A 评估。

### E4（可选，~1 小时）：决策会

汇总 E1–E3 后决定：进入 Variant B 实现（按 §3.3 清单）、或先做 A16 §5 的 wait/release 修复、
或维持现状。判据沿用 A16 §4 的 gate（S1：sha 一致 + hc_host+linear_host −40% + e2e +2% + A/A ≤1.5%）。

---

## 5. 未解问题清单

1. Aila 生产路径 12.9–35.5 μs/submit 的 5× 膨胀到底在 UR/L0 submit 自时间、块内胶水，还是 wait 打断动态批？
   （A16 §5 #4 的 VTune 臂未跑；它直接决定图路线的收益归属。）
2. Limited Graph 录制/重放与 Aila 现有构造的兼容边界：`cgh.depends_on(event)`、
   `ext_oneapi_submit_barrier`、`queue.memcpy` D2H、profiling queue、`ctx.synchronize()` 各自能否进图？
3. `q.ext_oneapi_graph(g)` 返回的 event 能否安全串接"图 → D2H → host → 下一张图"的真实层序列？
4. 同一张 executable graph 在 in-order queue 上连续重放（背靠背、不同参数表内容）是否安全、开销如何？
   （GE 只测了单实例单 in-flight 与独立 Slot。）
5. 多会话连续批处理下，跨会话共享图实例 + 每会话 Slot/表的正确性协议是什么？Fence 归属哪一个队列？
6. 48 层（或按拓扑 2–6 张）图组的显存与 capture 成本实测量级？GE 未测图实例常驻内存。
7. Aila 的 `resident_selected_`/`hit_ranks_`/`MoeExpertSlotDev` 设备化的最小改动面与位精确风险？
8. 图内 kernel 的 profiling/`AILA_Q4E_DECODE_STAGE_EVENTS` 方案（是否需要图属性 enable_profiling）？
9. prefix cache 命中路径是否走与 miss 相同的 kernel 序列？若不同，图缓存 key 是否覆盖？
10. 原生 Record&Replay（flags=0）是否支持多节点/多 buffer/event 依赖，值不值得 P6 式补充研究？
11. Indirect Dispatch 在 Aila 动态 KV 下（例如 attended length 随 token 增长）会不会出现与 GE 合成负载不同的收益？
12. Aila 的取消/会话销毁/重配置路径中，图实例的失效与 USM 释放顺序如何保证不违反"drain before free"不变式？

---

## 6. 附录：复现命令与源文件索引

```powershell
# 仓库构建与探针（只读使用，不修改该仓库）
cd E:\RiderProjects\A770-Graph-Experiment
cmd /c build.cmd
cmd /c phase2\build.cmd
cmd /c phase2\probe.cmd

# 基线复现（64 层）
.\run_experiment.ps1 -Program phase2 -ExtraArguments '--phase baseline --depth 64' -Output phase2/results/P0_new

# 参数表（64 B、Host/Shared/Device、指针随机）
.\run_experiment.ps1 -Program phase2 -ExtraArguments '--phase parameters --table-size 64' -Output phase2/results/P1_new

# 真实 decode 算子链
.\run_experiment.ps1 -Program phase2 -ExtraArguments '--phase decode --workload chain' -Output phase2/results/P3_new
```

关键源文件：`graph_experiment.cpp`（Queue/Graph/L0/Indirect 四路径与 E 矩阵）、`kernels.cl`（共享 ABI）、
`mutable_probe.cpp`（几何更新）、`native_graph_probe.cpp` + `graph_extension.hpp`（原生图 ABI 探测）、
`host_parameter_probe.cpp`（Host 表配对实验）、`phase2/parameters.hpp`（64 B 表/三种 USM/指针切换）、
`phase2/async_replay.cpp`（双 Slot+反馈流）、`phase2/indirect.hpp`、`phase2/decode_microbench/benchmark.hpp`
（七算子链与 128 B ABI）、`phase2/graph_executor/graph_executor.hpp`（最小 Executor 原型）。

Aila 侧配套证据：`docs/plans/qwen38-r1-bottleneck-audit-2026-10-16.md`（B16）、
`docs/plans/qwen38-r1-decode-submit-path-audit-2026-10-16.md`（A16）、
`tests/models/Qwen4ExpFrontGraphProbe.hpp`、`tests/models/Qwen4ExpIntegerQualityAcceptance.hpp`、
`src/core/Context.hpp:57`。
