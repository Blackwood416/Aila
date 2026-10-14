# Qwen3.8-Flash-Next / Arc A770 性能专项研究章程（2026-10-14）

用途：新对话/新团队的开场材料。回答三个问题：**① 怎么提速 ② 当前硬件的性能上界是多少
③ 怎么追**。所有数字都是本机实测，出处见括号；未实测的一律标 `[待验证]`。

---

## 0. 对象与生产形态

- 模型：`C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2`（48 层，12 full attention + 36 GDN
  linear；H=2560、I=640、每层 512 专家、top-k=10；24576 个 (layer, expert) key；专家槽
  ~2.417 MiB/个）。
- 负载：单会话 T1。agent 场景 = 14K-token 前缀 + PI 工具回合 + 128-token 续轮；服务场景 =
  OpenAI/Anthropic 兼容 HTTP/SSE（当前全测单会话、decode-batch=1）。
- 生产配置：chunk 4096、VRAM reserve 3584、专家槽 2792（bf16 KV）/2948（int8 KV opt-in）、
  W4A8/INT8 KV/GDN chunk/MTP 全部 opt-in 或关闭。
- 本机：Arc A770 16 GB（DG2）、64 GB RAM、oneAPI 2026.1、驱动 32.0.101.8724；
  PCIe H2D 实测 **9.2-9.8 GB/s**（pinned/大页）、D2D **95-129 GB/s**；
  模型盘（C:，LASTING NVMe）单文件顺序读 **~2.8 GB/s**、多分片并行 **8.6-9.8 GB/s**。

## 1. 事实基线（HEAD `c74296c`，pair `retained20`）

### 1.1 端到端（标准 14K+PI 序列，`tmp/q4e-resident-speed-20261009/run.py`）

| 指标 | 数值 | 备注 |
|---|---|---|
| 14K prefill TTFT | **~70.2 s** | 今天多次复现 70.17-70.80 s |
| 续轮 decode | **15.2-16.0 tok/s** | `(ctok-1)/(dur-first)` 口径 |
| decode 明细 | submitted 53.9 ms = router_wait 13.4 + experts_host 19.7（cpu_compute 12.9 + join 9.1）+ 其余 | 旧剖分（roadmap），需按当前代码重测 |
| 启动（warm） | 68 s（mmap 缺页） | 冷启动 116-119 s |
| 启动（opt-in 显式读 16 线程） | **63.8-64.4 s** | 冷缓存 −45%；热缓存反而 82 s → 默认关 |
| 14K prefill 每 4096 块（device 分段） | GDN 4.07 s + full attn 2.85 s + experts 12.58 s ≈ 19.8 s | experts 内：host fill 7.28 + bank_wait 3.76 + compute/其余 |
| 每块 staging 量 | 19.4-19.7k 专家 = **46-48 GiB**，H2D 实测 9.2 GB/s ⇒ ~5.1 s | 与内存型 compute **相加**（不可藏，见 §2） |

### 1.2 今天的增量发现（本轮会话）

| 发现 | 证据 |
|---|---|
| H2D 与流式 compute 相加（ALU 负载可重叠 4.0 vs 7.1 ms；内存型 6.70 ≈ 串行 7.16） | `qwen38-strata-c1/c2` 报告 |
| A770 有 copy-only 队列组（ordinal 1），但与默认队列同速（9.81 vs 9.76 GB/s） | `tmp/q4e-strata-c2/l0_copy_probe.cpp` |
| 加载慢的真因 = Windows 并行 mmap 缺页被同文件 section 串行（0.66 GB/s）+ 逐专家上传 | `qwen38-c7-deep-*` 报告 |
| DirectStorage：可获取（NuGet）且工作；单文件读 2.80-2.82 vs ReadFile×16 2.73-2.78 GB/s（+2-3%，CPU 减半） | `qwen38-directstorage-probe-*` |
| MTP（2048 上下文内）acceptance **113/151 = 74.8%**；decode +2.9%（首答 15.41 vs 14.98）/−0.3%（续轮）≈ 打平；门在载入时按 **context 容量 ≤2048**，32K 服务器永不生效 | `tmp/q4e-mtp-alignment-20261011/mtp-{A2,B2}*` |

### 1.3 现成工具（可直接复用）

- runner：`tmp/q4e-resident-speed-20261009/run.py`（标准 14K+PI）、`tmp/q4e-agent-plan-audit-20261012/run_val14k.py`（generate=128）、`tmp/q4e-strata-c1/run_ttft14k.py`（缩水版 6K 预热 + 14K，~5 min/臂）、`tmp/q4e-strata-c7/startup_only.py`（只启动 + 内存监控）。
- 内核探针：`AilaQwen4ExpKernelTests --prefill-bench-only|--queue-overlap-probe [--json]|--decode-gu-bench`（opt-in env 见 §5）。
- 收据工具：`tools/perf/ab_receipt.py` + `perf/templates/bench-receipt-v1.json`（median/p10/p90、pairs-won、A/A 噪声底）。
- 分析：`tmp/q4e-strata-c4/rank_headroom.py`（hotset 覆盖/上界）、`tmp/q4e_cache_sim.py`（旧 trace replay）、
  `tmp/q4e-strata-c7/nvme_read_bench.py`（盘带宽）。

## 2. 已知否证清单（不要重做；除非写清"这次机制有何不同"）

ESIMD 替换 SG8 DPAS（真实形状 0.67-1.01×）；INT8 attention DPAS（慢 84%）；prefill W4A8（上界 ≤0.7%）；
P4 SLM 解量化复用（−11~−18%）；half-unpack / 字节对 LUT；blocked 布局；uint64 权重读；
2 rows/subgroup；qmap 私有寄存器；fused residual；chunk 8192 / staging 64 / ring 4；
全量 pinned（51 GiB→`UR_OUT_OF_RESOURCES`）；BF16/INT8 权重副本；swap 默认开（decode −8.03%）；
GDN chunk 默认化（ttft −2.7% 但 decode −4~7%）；strata C1（stream-ahead）/C2（copy-only 队列）/
C3（16B 对齐）/C4（profile mix +1.1pp <3pp）/C6/C9/C10/C12/C13；DirectStorage 单文件读（+2-3%）。

## 3. 未解问题（按"预期价值 × 可验证性"排序，团队自行复核排序）

1. **prefill 的 layer-major 原型**：H2D 310 GB→~62 GB 推算、ttft −13~−18% `[待验证]`；
   工程量=prefill 驱动/调度/状态机（多日）。当前 chunk-major 每块把 46-48 GiB 专家重传一遍。
2. **decode 64 ms/token 的当代成本模型**（第一优先交付）：旧剖分需要按当前代码重测并细分到
   "device kernel / host 提交 / 专家供给（resident vs staging vs CPU）/ 路由同步 / 采样"。
   关键问题：13.4 ms 的 router_wait 是队列深度还是真实计算？19.7 ms 的 experts_host 里
   CPU 专家计算占多少、能否真重叠？
3. **MTP 扩展**：把门从"载入时 context ≤2048"改成"按会话长度逐请求生效"（短会话启用、长会话
   静默回退）是低风险第一步；acceptance 74.8% → 95% 需要什么（width=4？draft 训练/温度？
   只在特定内容上启用？）。注意 draft 要为 prompt 每个 token 做 Step ⇒ 长 prompt 有额外成本。
4. **专家供给层级与容量**：2948 槽 / 51 GiB RAM 补集 / CPU 路径；每 token 24.2 MiB 的专家需求
   里有多少走 CPU、多少走 staging；CPU 专家核是否已到 DRAM 墙（旧结论：接近）。
5. **prefill 的 GDN (4.07 s/块) 与 full attention (2.85 s/块)** 是否已到该形状的 kernel 上界；
   与 x-locality/ALU 口径的对照（P4 工具可复用）。
6. **服务模式吞吐上界**：多会话/`--decode-batch-size>1` 的聚合吞吐（目前全部单会话数据）；
   连续批处理是否能把 64 ms/token 的固定成本摊薄。
7. **启动与内存**：冷启动 116 s（显式读 63.8 s）、热启动 68 s；峰值内存 nightly 观察点；
   多分片并行预读（8.6-9.8 GB/s）是唯一已量化的大杠杆。
8. **内核族剩余空间**：在 §2 之外，按 A770 技能包（roofline、ESIMD、joint_matrix、GRF 模式）
   重新做一次"哪个核离上界最远"的审计（这次以测量为准，不预设结论）。

## 4. 建议团队结构（3-5 个子代理，按章程 §3 分工）

| 角色 | 范围 | 首轮交付 |
|---|---|---|
| **R 成本模型/roofline**（总协调） | decode+prefill 逐项成本模型与"若 X 免费"上界；收据管理 | 两张表（每 token / 每 4096 块）+ 上界排序；`tools/perf/ab_receipt.py` 收据 |
| **D 解码路径** | router_wait / experts_host / CPU 专家 / 采样 / MTP | decode 明细重测 + 2 个候选（如 MTP 按会话启用、CPU/GPU 分担重排） |
| **P prefill 与供给** | chunk 流水线、staging、layer-major 预研 | 每块分段复核 + layer-major 可行性/工作量评估（含 layer split 或跨块冷专家窗口） |
| **K 内核** | NF4 GEMV/attention/GDN 的 roofline 对照与候选 | "离上界最远"清单（含现有 negative 清单的复核） |
| **S 服务/多会话**（可选） | batch>1/多会话聚合吞吐 | 聚合吞吐基线与瓶颈定位 |

纪律：子代理只写自己名下文档/脚本；改 src 前先在报告里给出探针与门；GPU 同一时刻只允许一个占用者。

## 5. 工作纪律（硬性）

1. **证据优先**：oracle/上界探针 → 单变量判别 → 门 → 实模；禁止"看起来该快"直接实现。
2. **A/B/B/A + 收据**：`tools/perf/ab_receipt.py`；先跑 A/A 噪声底；报 median + pairs-won。
3. **默认值门槛**：kernel 级 ≥3%（两形状）、端到端 ≥+5%；否则 opt-in。
4. **GPU 互斥**：跑 GPU 前 `Get-Process AilaWorker,AilaServer`；同一时刻一个占用者（UI 自查看屏不受限）。
5. **禁止固定长 sleep**：1-2 s 轮询完成条件（进程/result.json/日志标记），上界 900 s。
6. **不误删**：`tmp/*-aborted` 与既有产物保留；清理前先确认。
7. **pair 纪律**：改 worker 必重建 server（identity 配对）；采纳=提交+重建+归档 `retainedN`；工作树保持干净。
8. **口径**：decode rate `(completion_tokens-1)/(duration-first_content)`；报告必须带测量硬件与 revision。
9. 输出会变的候选（tail skip/CPU 分担/量化）要另存 raw token ids 并附独立校验脚本。

## 6. 首个 Sprint（建议 2-3 天）

- **S1（半天）**：跑通 §1.3 工具链，复现 14K 70.2 s / 15.5 tok/s 两个基线；产出 A/A 噪声底收据。
- **S2（1 天）**：R 出两张成本模型表（decode 逐项、prefill 逐块），对每个 >5% 的项写"若为 0 的墙钟"上界
  （用现有 oracle 手法：env 短路/停拷贝/替代路径）。
- **S3（半天）**：按上界×可行性排序，选 2 个候选写探针（优先：MTP 按会话启用；prefill layer-major 的
  "跨块冷专家窗口"最小原型；decode 的 CPU/GPU 分担重排）。
- **S4**：Go/No-Go，实施第一个；每个决定落一份 receipt + 一行台账。

## 7. 新对话开场提示词（复制粘贴用）

> 从 `E:\RiderProjects\Aila` 继续。先读 `docs/plans/qwen38-performance-research-charter-2026-10-14.md`
> 与 `docs/plans/qwen38-qwen4exp-pipeline-status-2026-10-13.md`，按章程 §4 组建 3-5 个子代理
> （成本模型/解码/prefill 供给/内核/服务），按 §6 的 Sprint 执行：先复现基线并产出 A/A 噪声底，
> 再做 decode 与 prefill 的逐项成本模型与"若 X 免费"上界表，最后按上界排序选 2 个候选写探针。
> 全程遵守 §5 纪律（证据优先、A/B/B/A 收据、GPU 互斥、无固定长 sleep、pair 纪律、默认值门槛），
> 已否证清单见 §2，不要重做。
