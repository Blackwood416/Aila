# Qwen4Exp 推理管线真实瓶颈诊断（2026-10-16）

> Versioned copy of `tmp/q4e-bottleneck-audit-20261016/REPORT.md` (untracked scratch:
> probe sources, scripts and raw logs remain in that directory). Authored by Claude
> Opus at the owner's request; no project source was modified by the audit.

范围：HEAD `f798c18`（工作树只有 docs 改动）、`build-refactor/AilaLib.lib`（10-16 12:39 构建）、A770 驱动
32.0.101.8724、oneAPI 2026.1。没有改任何项目源码，没有跑全模型；新增的只有本目录下的探针与脚本。
标注：`[实测]` = 本次在本机跑出；`[语料]` = 引自既有日志/文档（给出处）；`[估算]` = 推导值。

---

## 0. 结论

**Prefill（14K TTFT ≈ 66–71 s，每 4096 块 ≈ 19.9 s）的罪魁祸首是 MoE 专家阶段：ALU NF4 专家核 + PCIe H2D，二者在这台机器上是串行相加的。**

- 专家阶段 12.4 s/块（62%）可以被一条成本定律精确描述：`experts_dev = 0.251 ms × 上传专家数 + 3.57 µs × 激活数`，
  拟合 5 个 run、25 个块，误差 ±1–3% `[实测拟合，语料日志]`。即每块 ≈ **4.9 s H2D（9.9 GB/s，正好是 PCIe 速率）+ 7.0 s 专家计算（≈2.8 TFLOPS）**，没有重叠。
- 隔离探针在真实路由直方图上复现：生产专家核 138–160 ms/层，H2D 链 118–129 ms/层，两者同跑 = **严格相加**（overlap ≈ 0）`[实测]`。
- 专家计算慢的根因：每个 NF4 权重在每个 4-token quad 里都要重新解量化一次（一个专家一块里 ~20 次），核是 L1/加载管线受限的 FP32 FMA 链；
  这种访存型核又让 H2D 无法藏起来（Strata C1/C2 已测到"H2D 与内存型核相加"）。
- 同形状下"解量化一次 → bf16 → oneDNN 批量 GEMM（XMX）"的设计点：**39–42 ms/层，比生产路径快 3.4–4.0×；叠加 H2D 后 140–156 ms/层，快 1.9×，68% 的计算被藏到 H2D 下面** `[实测]`。
  推算 14K TTFT ≈ 69 s → **≈ 48 s（−30%）** `[估算]`，之后 prefill 的地板变成 PCIe（每块 4.9 s）+ GDN/attention（≈7 s）。

**Decode（≈ 55 ms/token，17–18 tok/s）的罪魁祸首是"每层一次 GPU→host→CPU→host→GPU 的串行往返 × 48 层"的控制面开销，而不是任何一个 kernel 或 CPU 专家本身。**

- 最佳组合（e10）每 token：后端 51.0 ms = 内核提交 ≈ 18.7 + MoE 协调 11.5 + 等 router 11.4 + 等 CPU join 6.8 + 杂项；sample 3.5 ms `[语料]`。
  单个协调线程 100% 忙（≈33 ms 是它自己的 CPU 工作），GPU 估计 ≈45% 空闲，18 个 CPU 专家线程 ≈80% 空闲。
- 生产队列配置（L0 v1 + 普通命令列表 + 动态批）会把已提交的 kernel 扣在批里直到下一个 SYCL wait：
  探针证实"resident 专家两只 kernel 在 CPU join（condvar 等待，不是 SYCL wait）期间不执行"，一次事件状态查询就能放行；
  在 decode 形状的合成循环里这一处就值 **−14%（29.2 → 25.4 ms/token）** `[实测]`。host↔GPU 一次往返固定 **59 µs** `[实测]`。
- 物理地板：CPU 专家 ≈199–275 个 miss × 2.53 MB ÷ ~47 GB/s ≈ 10.6–14.8 ms（4 通道 DDR4-1866 墙）；GPU 字节 ≈12–18 ms。
  当前 55 ms 与 ≈26–32 ms 的串行地板之间的 ~25 ms 主要是提交/同步/批扣留。

**另外发现一个潜在正确性 bug**：本工具链上 `ev != sycl::event{}` 恒为 true（默认构造事件各有独立 impl）`[实测]`，
专家分派器因此在"没有单激活尾巴"的 chunk 上返回一个无队列、已完成的默认事件，它被记为 staging bank 的读者栅栏 → 下一次 H2D 可能覆盖仍在被读的 bank。罕见、非确定性（见 §5）。

---

## 1. 平台约束（决定了哪些量是"物理"的）

| 项 | 值 | 来源 |
|---|---|---|
| CPU | Xeon E5-2696 v3（Haswell-EP），18 核 18 线程，2.3 GHz，AVX2，L3 45 MB | `[实测]` Get-CimInstance |
| 内存 | 4 × 16 GB DDR4-1866 RDIMM（4 通道，理论 59.7 GB/s；CPU 专家路径实测 45–53 GB/s） | `[实测]`；contention 探针 53.7 GB/s |
| PCIe | GPU 挂在 Haswell-EP 根端口（Root Port 2F08）→ **PCIe 3.0**；H2D 实测 9.9 GB/s（A770 本身支持 Gen4） | `[实测]` PnP 链 + 探针 H 臂 |
| GPU | A770 16 GB；oneDNN bf16 批量 GEMM 在专家形状达到 31–61 TFLOPS（含 padding） | `[实测]` 探针 X 臂 |
| 模型 | 48 层（36 GDN + 12 full），512 专家/层 top-10，专家槽 2.534 MB，总专家 59.4 GiB；GPU 常驻 2800 槽（6.6 GiB），RAM 补集 51.4 GiB | `[语料]` |
| VRAM | 15 933 MB = 静态 4 522（其中 bf16 HC 1 279 + bf16 LM head 1 271）+ session 622 + reserve 3 584 + 专家 7 204 | `[语料]` e10 Memory Admission |

---

## 2. Prefill：实际执行逻辑与真瓶颈

### 2.1 每层执行（chunk = 4096，`ForwardImpl` → `RunAdaptivePrefill` → `RunGroupedExperts`）

1. HC（norm+down+up）→ attention（GDN 递推 或 QSA full attention，稠密投影走 bf16 解量化 + oneDNN XMX）→ HC → router GEMM → top-k 写 pinned host。
2. host 阻塞等 top-k（`router_wait`，`Qwen4ExpBnb4Backend.cpp:2620`）——这期间 GPU 在跑本层 attention，所以它是"设备队列深度"，不是 host 工作。
3. host 统计每专家用量；常驻命中先跑（`RunGroupedExperts`）；冷专家按 16 个一组：host 从大页 arena memcpy 到 pinned bank（fill），
   在 `copy_q_` 上 H2D，chunk 的 kernel 依赖该 H2D 事件；bank 复用依赖上一个读者事件（`Qwen4ExpAdaptiveCache.cpp:606-660`）。
4. 每个 chunk 两个分派器调用（gate_up、down），内部按 quad/pair/single 分段各一个 kernel（`MoeExpertOps.cpp:5867-6256`）。
5. 有序累加回 y。下一层的 router 必须等本层 MoE 结果 → **层内 attention 与 MoE 严格串行，跨层无法预取（router 读 post-attention hidden）**。

### 2.2 成本定律（生产日志）

`fit_expert_span.py` 对 dpe-b1、c2-round2-b1/b2、d0-deferwait-b2、dec-instr-b2 共 25 个 prefill 块拟合：

```
experts_dev = 0.2511 ms × staged + 3.569 µs × acts      (staged → 10.1 GB/s；acts → 2.75 TFLOPS)
```

关键反证：1656-token 尾块 staged=17 431（接近整块）而 acts 只有 40%。若 H2D 与计算重叠应是 max(4.4, 2.9)=4.4 s，实测 7.1–7.6 s ≈ 两者之和。

一个 4096 块（dpe-b1）：experts 12.44 s（H2D 4.9 + 计算 7.0 + 少量）、GDN/linear 4.04 s、full 2.1–3.0 s、hc 0.27 s、shared 0.19 s。

### 2.3 隔离探针（`expert_shape_probe2.exe`，真实层直方图，7 个层）

生产 ALU 入口（`moe_expert_major_prefill_gate_up/down`，nested fixed-256）、生产布局（quads→pairs→singles）、16 专家 chunk、2 bank：

| 层 | ALU K | H2D H | ALU+H2D KH | XMX DX | XMX+H2D DXH | K/DX | KH/DXH |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 159.6 | 126.6 | 289.4 | 39.8 | 153.5 | 4.01 | 1.89 |
| 3 | 138.0 | 126.4 | 271.2 | 39.1 | 139.7 | 3.53 | 1.94 |
| 10 | 147.8 | 128.7 | 292.2 | 41.6 | 156.4 | 3.55 | 1.87 |
| 20 | 138.9 | 128.7 | 270.0 | 38.7 | 141.0 | 3.59 | 1.91 |
| 30 | 147.1 | 125.8 | 288.6 | 40.3 | 154.6 | 3.65 | 1.87 |
| 40（wgshare=4） | 148.2 | 125.0 | 305.2 | 41.8 | 155.8 | 3.54 | 1.96 |
| 47 | 137.7 | 118.4 | 282.4 | 40.7 | 148.6 | 3.39 | 1.90 |

（ms/层，每层 40 960 个激活；XMX 臂 = 向量化 NF4→bf16 解量化 + gather + oneDNN 批量 GEMM + SwiGLU + GEMM + scatter，按用量排序分块，padding 1.2–1.8×。）

- ALU K = 3.36–3.90 µs/act，与生产拟合 3.57 µs/act 一致 → 生产里的计算项就是这个核本身。
- KH ≈ K + H（overlap fraction −0.26…+0.16）→ H2D 与生产专家核**完全相加**；wgshare=4 时甚至超加性。
- DXH ≈ H + 12–30 ms → XMX 版计算 68% 左右能藏在 H2D 下（与 C1"ALU 型负载能重叠"一致）。

### 2.4 Prefill 推算 `[估算]`

| 每 4096 块 | 现在 | XMX 专家 |
|---|---:|---:|
| 专家计算 | 7.0 s | ≈1.9 s，约 2/3 被 H2D 藏住 |
| H2D（46.8 GiB，PCIe 3.0） | 4.9 s（相加） | 4.9 s（成为 MoE 地板） |
| GDN + linear 投影 | 4.0 s | 4.0 s |
| full attention | 2.1–3.0 s | 同 |
| hc/shared/router | 0.5 s | 0.5 s |
| **块** | **19.9 s** | **≈13–13.5 s** |

14K TTFT ≈ 3×13.5 + 尾块 ≈8 ≈ **48 s**（现 66–71 s）。下一层地板：每个请求 179.5 GiB H2D（≈19 s）来自 chunk-major 每块重传；
更大 chunk / layer-major 能砍字节，但它们此前的失败是 VRAM 超额（§4.4），且在 XMX 下 M 变大只会更有利。

数值：XMX 路径不是逐位一致（bf16 权重 + 不同累加顺序）。但生产 profile 已经 `AILA_Q4E_PREFILL_XMX=1`，
稠密投影本来就是"NF4→bf16 + oneDNN"（`Bnb4BitLinear.cpp:2515`），专家只是唯一没享受这条路的大 GEMM 族。需要用现有 teacher/KL 工具做质量门。

---

## 3. Decode：实际执行逻辑与真瓶颈

### 3.1 每层执行（count = 1，`ForwardImpl` 循环 + `RunResidentDecode`）

1. 提交约 20 个小 kernel：attn-HC、attention（GDN 或 QSA）、MLP-HC、resident-input D2H、MoE front（fused）、top-k、shared gate_up/down。
2. host 等 top-k（`router_wait`）。
3. host 分类 hit/miss → `StartArenaCold` 把 miss 交给 CPU worker（mutex+condvar，18 线程 TBB arena，NF4-Q8 AVX2）→ 提交 resident hit 的 2 个 kernel。
4. host 在 `JoinResidentCold` 阻塞等 CPU（condvar）。
5. 提交 weights memcpy + 有序累加（直接读 CPU 结果行）+ shared add / inject → 下一层。

### 3.2 每 token 账本（e10 bundle：front-fused + int8 KV + agent6k，n=128，无 device-event 观察者） `[语料]`

| 项 | ms | 性质 |
|---|---:|---|
| 内核提交（hc 6.6 + linear 3.3 + full 2.8 + shared 2.8 + front 1.8 + topk 0.9 + ple 0.25 + head 0.16） | 18.7 | host CPU 工作 |
| router_other | 1.9 | host CPU 工作 |
| MoE 协调（experts_host − cpu_join） | 11.5 | host CPU 工作 |
| router_wait | 11.4 | host 等 GPU |
| cpu_join | 6.8 | host 等 CPU（CPU 专家共 11.2 ms，199 miss/token） |
| sample | 3.5 | head + argmax 回读 |
| **ForwardWall** | **≈55** | |

### 3.3 探针（`submit_probe.exe`、`contention_probe.exe`，生产队列环境）`[实测]`

- 48 层 × 20 小 kernel + 每层一次 host wait：13.8–14.6 ms/token，设备忙只有 6.2–6.5 ms；同样 kernel 只在末尾 wait：8.3 ms。
  → 每层同步点让提交与执行近似串行（304 µs/层 ≈ 137 提交 + 135 执行 + 延迟）。
- 空 kernel + wait 的往返：**59 µs**。
- "router 同步 → 提交 2 个 resident kernel → host 自旋 230 µs（模拟 CPU join）→ wait"：420 µs/层 = 59 + 230 + 131，**kernel 没在自旋期间执行**；
  提交后查询一次事件状态：343 µs/层（放行）。固定 batch ≥4 同样扣留，batch 1–2 不扣留但单次提交贵（batch=1 时 43.9 ms/token）。
- immediate command list：41.5 ms/token（远差于 regular），与 QSA 文档的 −22% 结论一致 → 不是全局开关能解决的。
- 静态 graph 回放（每层一张图）：12.7 ms vs 14.6 ms（DG2 不支持 kernel 参数更新，需要设备侧参数块）。
- 加上每层一个 18 线程、10 MB 的 DRAM 流式 CPU 任务：29.0–29.5 ms/token；在 CPU join 前放行 resident kernel：**25.4 ms（−14%）**。
- CPU 争用只让每 kernel 提交从 5.9 µs 涨到 7.0–7.4 µs（+20%）→ 生产里 ~30 µs/kernel（E5：bench 里同样 194 个 HC kernel host 仅 1.24 ms，生产 hc_host 6.6 ms）的 ~5× 膨胀**不是**CPU 争用造成，在生产调用路径本身（需 VTune 定位）。

### 3.4 解读

- 一层 = 一个"提交 ~20 kernel → 等 GPU → 交给 CPU → 等 CPU → 再提交"的串行链；协调线程是关键路径上的唯一资源且永不空闲。
- attention 阶段 host 提交（≈20.6 ms/token）与 GPU 执行（≈21 ms）几乎相等 → 只砍 GPU（pair2 −16~21% kernel、int8 KV）e2e 只剩 1–3%，与语料一致；必须同时砍提交。
- MoE 阶段 CPU（≈0.23 ms/层）与 GPU resident（≈0.08 ms/层）理论可重叠，但 resident kernel 被批扣留，实际串行。
- 设备 span 账本（`AILA_Q4E_DECODE_STAGE_EVENTS`）按"上一 marker 结束 → 本 marker 开始"记账，把空闲记进下一个桶；
  decode 的 `experts_dev ≈ 30 ms` 主要是 GPU 在等 host flush，不是专家计算（C1 已测专家核忙只有 3.6–5.1 ms/token）。
- 残余地板：CPU DRAM ≈10.6 ms（agent6k）；驻留曲线很平：held-out 14K 上 2800 槽覆盖 H4 42.7% / agent6k 53.1% / oracle 62.8%，
  +525 槽（把 bf16 HC+head 量化省出的 1.27 GB）只让 agent6k 到 57.9%（miss 225→202，CPU 字节 −10%）`[实测，slot_curve.py]`。

---

## 4. 为什么之前大量工作没打中（证据链里的系统性误判）

1. **E1 prefill block bench 少算了 56× 激活**：`q4e_prefill_block_bench.cpp:89` `cold_acts = 593`/层，而生产是 1 600 267 miss / 48 层 ≈ 33 339/层（≈83/专家）。
   由此得出"专家核只占 2.8%、12 s 是供给"（cost model §6.25/§6.29），之后整整一轮工作都在供给侧（banks、并行 fill、layer stage、direct DMA、host-read），全部"计数器动了墙钟不动"。
2. **marker 到 marker 的设备 span 把等待记进下一个桶**：prefill 被判"device-bound"，decode 一度被判"GPU-span-bound"（§6.17），二者都是记账假象。
3. **XMX 被不合格的原型判死**：现有 grouped-XMX 是标 `[HYPOTHESIS]` 的 M8/N32/K32 小 tile（WG=32，每个 K 步 3 个 barrier、每 subgroup 只有 4 个 DPAS，`MoeGroupedXmx.cpp:25-127`）；
   ESIMD 门比的是 ESIMD vs joint_matrix，从没在真实 M（~83/专家，热专家 1 400+）上与 ALU 路径对比，且解包是单独一次 launch。"解量化一次、跨该专家全部 token 复用"的设计从未实现。
   再加上"内核类逐位一致优先"的门，这条唯一能动主导项的路被排除了——而稠密 prefill 早已接受同等数值类别。
4. **VRAM 超额被当成管线性质**：chunk-8192 实验里 runtime 容量按倍增长到 **cap=11 888**（6K 预热后），phase scratch 1.22 GB、其余 cap 级 buffer 全部翻倍，
   挤在 3 584 MB reserve 内 → WDDM 换页（TTFT 134.8 s）；reserve 2048 + 3 584 槽 ≈ 15.4/15.9 GB → decode 崩到 7.4 tok/s。"更大 chunk"与"更多驻留"因此被错误关闭。
5. **§6.33 CPU-MoE prefill 把每层当每块（48×）**——E9b 已更正，但 cost model 正文仍保留旧结论。

---

## 5. 正确性：`sycl::event{}` 比较恒不等

`event_eq_probe.exe` `[实测]`：`default == sycl::event{}` → 0，`default != sycl::event{}` → 1，两个默认事件互不相等；默认事件状态为 complete、无队列。

- `MoeExpertOps.cpp:6056-6058`（gate_up）与 `:6253-6255`（down）用 `if (single_ev != sycl::event{}) return single_ev;` 选"最后一个事件"——恒返回 `single_ev`。
  某 chunk 所有专家用量都为偶数（无 single 段）时返回的是默认事件。
- `RunGroupedExperts` → `RunAdaptivePrefill` 把它记为 `RecordStagingConsumer`（`Qwen4ExpBnb4Backend.cpp:1702`）；
  `StageColdExpertsAsync` 的 H2D `depends_on(staging_reader_[pp])`（`Qwen4ExpAdaptiveCache.cpp:653`）于是是空栅栏，
  新专家可能在旧 kernel 仍读 bank 时覆盖它。满 16 专家 chunk 触发概率 ≈2⁻¹⁶（每块 ~1 213 个 chunk → 约 2% 的块），层末小 chunk 被 router 同步掩护。
  探针在真实第 20 层分布上 6 个臂里观测到 48 个无队列事件。
- 同一写法还有约 80 处（`Qwen4ExpExpertExecutor.cpp` 24、`Qwen4ExpExpertCache.cpp` 16 等）；作为 `depends_on` 守卫是无害的，
  但 `Qwen4ExpExpertExecutor.cpp:1275` 这类"必须有真实事件"的检查永远不会触发。修法：用 `std::optional<sycl::event>`/bool 记录"是否提交过"，分派器返回最后一次真正提交的事件。

---

## 6. 杠杆排序（按"上界 × 证据强度"）

| # | 杠杆 | 上界 | 证据 | 成本/风险 |
|---:|---|---|---|---|
| 1 | Prefill 专家改为"解量化一次 + XMX 分组 GEMM"（按用量排序分块或真正的 grouped GEMM） | 块 19.9→≈13.5 s，14K TTFT −30% | 探针 7 层 3.4–4.0×（计算）/1.9×（含 H2D） | 中：新内核 + 质量门；非逐位一致 |
| 2 | Decode：在 CPU join 等非 SYCL 等待前显式放行（事件状态查询），并在 attention 阶段每几个 kernel 放行一次 | 估 −4…−10 ms/token | 合成循环 −14% | 低：几行代码，逐位一致 |
| 3 | Decode：削减每层 ~20 次提交（融合 + 精简提交路径；或设备侧参数块 + 静态 graph） | 估 −5…−10 ms/token | 生产 ~30 µs/kernel vs 裸 6–7 µs；graph −13%（合成） | 中：需先 VTune 定位 5× 膨胀 |
| 4 | 修 `sycl::event{}` 比较 | 正确性 | 探针复现 | 低 |
| 5 | 释放 VRAM（bf16 HC/LM head → Q8、down_acc 紧凑化、容量不按倍增长）后重测更大 chunk / 更多驻留 | prefill 少重传；decode miss −10% | slot 曲线、admission 账本 | 中 |
| 6 | PCIe/DDR 平台 | H2D 是 Gen3 上限；CPU 专家是 4 通道 DDR4-1866 上限 | 平台实测 | 硬件 |

---

## 7. 复现

```powershell
# 构建（复用 perf/PerfCommon.ps1 的 oneAPI 环境，链接 build-refactor/AilaLib.lib，不碰项目构建）
./tmp/q4e-bottleneck-audit-20261016/build_probe.ps1 -Source tmp/q4e-bottleneck-audit-20261016/expert_shape_probe2.cpp
# 运行（生产 L0 环境：copy engine 1、regular lists、L0 v1、dynamic batch；GPU 占用自检）
./tmp/q4e-bottleneck-audit-20261016/run_probe.ps1 -Exe ./tmp/q4e-bottleneck-audit-20261016/expert_shape_probe2.exe -Log out.log -ProbeArgs @('--layer','20')
# 日志分析（只读）
python -I tmp/q4e-bottleneck-audit-20261016/fit_expert_span.py <worker logs with dev events>
python -I tmp/q4e-bottleneck-audit-20261016/parse_adaptive_work.py <worker.log>
python -I tmp/q4e-bottleneck-audit-20261016/slot_curve.py tmp/q4e-strata-c4/heldout-14k.profile h4=... a6k=...
```

日志：`expert-A1.log`、`expert2-L{0,3,10,20,30,40}*.log`、`expert2-L47.log`、`submit-*.log`、`contention-*.log`、`event-eq.log`。
