# Qwen3.8 专家 prefill 内核：真正的瓶颈是 SLM dataport（2026-10-10，管理员 VTune）

日期：2026-10-10。提权后用 VTune GPU 硬件指标（`gpu-hotspots`，administrator）对当前生产配置（wgshare=4、HEAD `6dc4c3d`）采集：
- `r9`：`characterization-mode=global-memory-accesses`，全模型诊断运行（vtune-hw1），运行时长 258s 无额外开销。
- `r12`：`characterization-mode=lsc-slm`（多遍采集）于内核测试基准。
- `source-analysis` 模式（GTPin 插桩）在本驱动上崩溃（`latency_tool_helpers:621`），不可用。

## 测量事实（[MEASURED]）

MoE 专家 prefill 内核（gate_up/down，wgshare=4）：

| 指标 | 数值 |
|---|---|
| XVE Active / Stalled / Idle | ~50–64% / ~34–48% / ~2–4% |
| XVE 线程占用率 | ~90%（不是占用不足） |
| **GPU Shared Local Memory 读带宽** | **6,400–8,300 GB/s（≈dataport 上限的 70–85%）** |
| GPU Load Store Cache (L1) 读带宽 | 800–2,400 GB/s |
| GPU DRAM 读带宽 | 14–105 GB/s（远未饱和） |
| Spill 内存 | gate_up 576 B/线程 |

对照：DeltaNet prefill XVE **idle 75%**（并发锁死，见方向 2）；QSA attention 健康（active 48%，占用 72%）。

## 归因（[FACT] + [INFERENCE]）

[FACT] 内核内层循环每 lane 每迭代（4 权重/矩阵）做：2 次全局权重加载、4 次 `group_broadcast`（scale）、4×vec8 x 加载、**16 次 `qmap[...]` SLM gather**（NF4 反量化 LUT：gw0..gw7 + uw0..uw7，data-dependent 索引）、64 次 FMA。
[INFERENCE] SLM 读流量 ≈ 每 lane 每迭代 64 B（LUT）并叠加 576 B/线程 spill scratch；x 走 LSC（64 B/迭代）。SLM dataport 是该内核的**首要受限资源**；DRAM 完全不是瓶颈——此前"25× 权重重读 → DRAM 压力"的模型不成立（重读主要被 L2 吸收，且权重字节远小于 LUT/x 流量）。
这也解释了既往结果：8-token 变体把 spill 推到 3,904 B（SLM 洪泛）→ +23%；share=8 的 L1 失效与 SLM 无关但同样回退；wgshare=4 的 -3.7% 来自 LSC/L2 侧压力的减小。

## A1 下一步实验梯（按序，全部要求位精确）

1. **降寄存器压力（消 spill）**：内层循环改为"半窗口"消费（解开 gw0..3 → 16 FMA → gw4..7 → 16 FMA；x 同理），保持每累加器的 FMA 顺序逐字不变。目标：spill → 0，SLM 读流量即刻去掉 scratch 部分。
2. **减 LUT gather 次数**：改用 256 项字节对表（`table[byte] = (hi,lo)` 2×f32，8 B/次）把每字节 2 次 4B gather 合并为 1 次 8B gather（同样字节、事务减半）；或评估 `select_from_group` 动态 shuffle 替代（需先验证 IGC 不降级为 SLM 实现）。
3. **在上述两条成立后重测 8-token 窗口**：LUT 流量/FMA 减半的前提是寄存器不再爆炸；若 spill ≈0 且窗口 8 获胜，等价读降为 ~32 token/权重读。
4. 备选：x 侧工作集（每 wg 80 KB）超 L1 有效容量时的 share 自适应（4/2 按层切换）。

每步门槛：内核基准（`AilaQwen4ExpKernelTests` 位精确 + benchmark）→ 诊断块级（≥1%）→ 生产 A/B/B/A（噪声底 ±0.6%）。

## 实验梯执行结果（同日）

**矩阵实现与位精确**（补丁 `tmp/q4e-slmpack-20261010/candidate.patch`，`AILA_Q4E_UNPACK_HALF`，默认关）：{4tok,8tok}×{full,half} 四个组合全部与串行参考逐位一致（含 share 2/4/8、nested、不整除/尾形状），四组 env 测试全过、0 警告。

**内核基准**（同 shape，gate_up+down 中位数；tokens=64/acts=640）：

| 方案 | ms | 每 act |
|---|---:|---:|
| 4tok full | 2.24–2.27 | 3.53 µs |
| 4tok half | 2.63–2.65 | 4.14 µs（-16%，两次 vec4 加载的代价） |
| 8tok full | 3.42–3.43 | 6.68 µs（spill 灾难） |
| 8tok half | 1.99–2.00 | 3.89 µs（比 8tok full 快 1.71×，但仍为 4tok full 的 1.10×） |

**块级判别**（`--diagnostic`，同二进制单变量；稳态块 submitted）：

| 臂 | block1 | block2 | block3 |
|---|---:|---:|---:|
| A：wgshare4（补丁二进制 / 干净二进制） | 19.187 / 19.211 | 19.443 / 19.441 | 10.662 / 10.802 |
| B：wgshare4 + 8tok + half | **20.484** | **20.711** | **11.402** |

**结论**：8tok×half 比当前配置慢 **+6.7%**（experts_host/bank_wait 同步增长，与 GPU 变慢一致）→ **否决**；half-unpack 未能在生产窗口兑现 LUT/2 的收益（相邻两次 4tok-half 内核基准也显示额外加载指令的代价）。slmpack 补丁不采纳，仅保留补丁与证据。
**漂移记录**：同配置 A 臂今晨 18.918/19.163/10.538 → 下午 19.19–19.21/19.44/10.66–10.80（+1.4%），干净重建复现后确认是环境漂移而非补丁回归——块级对照必须同日同窗口进行。

**字节对 LUT 表实验（`AILA_Q4E_LUT_PAIR`，`tmp/q4e-lutpair-20261010/`）——已试，负结果**：256×float2 设备全局表 + 每字节 1 次 8B gather 替代 2 次 4B gather，位精确（share 2/4/8 × lut 0/1 × tokens{2,16,26,32,64}×k{4,10}×nested 全过），但内核基准 **-15~-16%（tokens=64）/ -9~-11%（tokens=16）**。汇编与计时探针确认：vec2 gather 真实生成（非标量化），而 IGC 本就融合相邻 4B 对（对照 3×d32x2+2×d32）；UGM 化反而更慢（0.103 vs SLM 0.0934 ms），且 **SLM 内 8B 配对相对 4B 标量收益为 0**（0.0933 vs 0.0934）。

**根因模型修正**：把 LUT 流量整体移出 SLM 或减半 gather 地址数都不改善内核 → "SLM dataport 70–85%" 是**相关性指标而非限制项**；结合 XVE stalled 34–48%，更可能受 **gather/scoreboard 依赖路径**（load→解包→查表→FMA 的串行链）支配。剩余未试的方向：② 预解量化权重到 bf16 的卸载结构改造（LUT/scratch 从 SM 移除，权重字节 ×3.2、bank 40MB→~130MB/对，属 B3 范畴）；③ `select_from_group`（需先验证不被 IGC 降级）；④ 面向 scoreboard 的结构调整（例如降低每迭代的依赖深度/交错解包与 FMA），但尚无该假设的直接证据，需 overview 采样或新探针先定性。

## 证据位置

- `tmp/q4e-vtune-20261010/r9/`（global-memory-accesses，含 `r9-gpu-tasks.csv`、`r9_summary.py`）
- `tmp/q4e-vtune-20261010/r12/`（lsc-slm，含 `r12-gpu-tasks.csv`、`r12_summary.py`）
- `tmp/q4e-vtune-20261010/r10-*`（source-analysis 崩溃记录）
- 内核源码：`src/ops/MoeExpertOps.cpp`（wgshare gate_up 内层循环，LUT gather 段）
