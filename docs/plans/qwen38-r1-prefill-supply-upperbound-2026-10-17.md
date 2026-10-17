# Qwen3.8-Flash prefill 供给侧成本与上界（2026-10-17）

状态：只读研究；未跑 GPU、未构建、未改源码。基线为当前 canonical
Qwen4Exp prefill：chunk 4096，XMX expert path，`STAGE_COPY_PARALLEL=1`，
2 个 staging bank，int8 KV，agent6k 热集。硬件为 Arc A770 16 GB、
driver 32.0.101.8724、oneAPI 2026.1、Xeon E5-2696 v3 / 18 threads。

## 0. 结论

- 当前 14K（prompt 13944）TTFT 为 **53.202 s / 54.057 s**；A/A spread
  **1.61%**。基线取中位约 **53.6 s**，目标 35 s 对应约 **398 tok/s**。
- `router_wait` 为 **26.227 / 27.059 s**（49.4-50.2% TTFT），折合
  **1.881 / 1.941 ms/row**；4096-row block 为 **7.184-8.074 s**，
  8192-row 实测 **14.480 s / 8192 = 1.77 ms/row**。所以
  “4096 block ~7.1 s、约 1.8-2.0 ms/row”在当前 XMX 基线上成立。
- 这个等待不是 top-k 计算，也不是 state query：它是 in-order compute queue
  上 `actual_route_event.wait_and_throw()` 的设备队列深度，包含本层
  attention / HC / router 的前序工作。每块 `topk_submit_ms` 只有约 2.1 ms；
  per-stage sync 时 `router_wait` 会降到 3-21 ms，而 GDN/full-attention
  设备时间仍在。
- `experts_host` 为 **26.261 / 26.303 s**，其中 `fill` 9.2 s、
  `bank_wait` 8.1-8.3 s。H2D 字节为 **174.2-174.6 GiB/prompt**；
  按 9.8 GB/s 物理下限为 **18.5-18.6 s**。`experts_host` 已包含
  `fill` 和 `bank_wait`，不能再相加；H2D 也与 host interval 重叠。
- **供给侧单独打 35 s 没有余量**：把全部 H2D 物理成本变免费，理论墙钟为
  **34.6-35.5 s**，刚好贴着目标；而 14 GB 级 VRAM 不可能装下 51 GiB
  冷专家补集。任何真实候选都必须同时减少字节或设备侧工作。

## 1. 证据链

### 1.1 当前 14K canonical 逐块

来源：`tmp/q4e-sprint-r1-20261014/xpb-b1-worker.log:228,232,236,240`，
`xpb-b2-worker.log:230,234,238,242`；TTFT 取自同目录 `result.json`。

| block rows | TTFT / block wall (s) | router_wait (s) | experts_host (s) | fill (s) | bank_wait (s) | H2D (MiB) |
|---:|---:|---:|---:|---:|---:|---:|
| b1 4096 | 14.595 | 7.488 | 6.972 | 2.381 | 2.193 | 45,630.4 |
| b1 4096 | 14.847 | 7.911 | 6.802 | 2.327 | 2.097 | 44,479.9 |
| b1 4096 | 14.913 | 8.074 | 6.702 | 2.280 | 2.060 | 43,621.9 |
| b1 1656 | 9.564 | 3.586 | 5.827 | 2.182 | 1.926 | 40,830.2 |
| **b1 total** | **53.919** | **27.059** | **26.303** | **9.169** | **8.276** | **174,562.4** |
| b2 4096 | 14.227 | 7.184 | 6.905 | 2.369 | 2.108 | 45,246.1 |
| b2 4096 | 14.603 | 7.704 | 6.764 | 2.301 | 2.079 | 43,841.8 |
| b2 4096 | 14.677 | 7.827 | 6.716 | 2.295 | 2.040 | 43,841.8 |
| b2 1656 | 9.546 | 3.512 | 5.876 | 2.241 | 1.878 | 41,260.5 |
| **b2 total** | **53.053** | **26.227** | **26.261** | **9.206** | **8.106** | **174,190.2** |

`router_wait + experts_host` 占 53.919 s 的 98.9%（b1）和 53.053 s 的
98.9%（b2）；这两个 interval 是当前 prefill 墙钟的全部主项。`fill` 与
`bank_wait` 是 `experts_host` 的子项，表中只用于分解，不能再次求和。

### 1.2 `router_wait` 的证据

`ForwardImpl` 的顺序是 attention -> MLP-HC -> router GEMM -> top-k ->
`wait_and_throw`：

- attention: `src/models/Qwen4ExpBnb4Backend.cpp:2674`
- MLP-HC: `src/models/Qwen4ExpBnb4Backend.cpp:2719`
- router GEMM: `src/models/Qwen4ExpBnb4Backend.cpp:2791`
- top-k: `src/models/Qwen4ExpBnb4Backend.cpp:2814`
- route wait: `src/models/Qwen4ExpBnb4Backend.cpp:2857`
- prefill consumer: `src/models/Qwen4ExpBnb4Backend.cpp:3051`

这里的 top-k 直接写 pinned host；host 必须等 route event 才能构造 expert
map。没有轮询或 event-status query。event 在 in-order queue 上覆盖此前全部
提交，因此等待时间主要是 attention/HC/router 的设备队列深度，而不是
top-k kernel 的 2.1 ms 提交时间。

三条互相独立的算术一致：

1. 当前 4096 block：`7.184-8.074 s / 4096 = 1.754-1.971 ms/row`。
2. 当前 1656 tail：`3.512-3.586 s / 1656 = 2.121-2.165 ms/row`（固定
   per-block 成本使其略高于线性段）。
3. chunk-8192 反证：`14.480 s / 8192 = 1.77 ms/row`；旧 4096 记录
   `7.1 s / 4096 = 1.73 ms/row`，见
   `docs/plans/qwen38-r1-chunk8192-first-block-falsification-2026-10-16.md:32-34`。

stage-serialized 证据更直接：`r1-stages2` 的 4096-row 分块中 GDN
4.04-4.13 s、full attention 2.37-2.94 s、HC/router 约 0.5-0.8 s，
合计约 7.0-7.8 s；同一实验把 per-stage sync 打开后 `router_wait`
降到 3-21 ms。也就是说，生产里的 7-8 s 是这些设备工作的队列残差，
不是 router 独有成本。

### 1.3 `experts_host` 与 H2D 的证据

`Qwen4ExpAdaptiveCache::StageColdExpertsAsync` 先在 host 上等待同 bank 的
上一轮 DMA（`Qwen4ExpAdaptiveCache.cpp:615`），再从大页 arena 复制到 pinned
bank（并行路径已默认，`:627-639`），最后在 `copy_q_` 上提交 H2D
（`:657-658`）。下一轮 H2D 的依赖是上一消费者事件
`staging_reader_[pp]`，不是状态查询。

当前每块填 18.0-18.9k 个 2.417 MiB compact expert，约 43.6-45.6 GiB；
14K 共 72.1-72.2k slots、174.2-174.6 GiB。按 `xpb` 的 H2D 字节和
9.8 GB/s 量级，物理 DMA 为 **4.6-4.9 s/block、18.5-18.6 s/prompt**。
`bank_wait` 只记录同 bank 的上一轮 DMA/消费者等待，不是当前 chunk 的完整
DMA 时间，因此不能把 `bank_wait` 当作 H2D 总量。

## 2. 哪些等待是语义必需

| 项目 | 实际等待/工作 | 语义必需性 | 结论 |
|---|---|---|---|
| `router_wait` | event 覆盖 attention/HC/router + top-k 发布 | 对 host 读取 route 是必需；对设备侧消费不是必需 | 不能直接删；单独删 host wait 也不会消掉设备工作 |
| top-k pinned-host 交接 | top-k 写 `router_host_ids_/weights_` | 当前 host 构建 expert map 必需 | 只有改成 device-side route consumer 才能去掉 host round-trip |
| `fill` | 大页 arena -> pinned staging bank 的 CPU copy | 当前 copy-engine 源格式的必需步骤 | 可用 union/promotion 减少字节；不能当成“免费等待” |
| `bank_wait` | 同 bank 的上一消费者/DMA 事件 | 防止覆盖正在被消费者读取的 host bank | 必需的是 ownership，不是固定 2 s |
| H2D | 45 GiB/block 的冷专家字节 | 冷专家不在 VRAM 时的物理成本 | 9.8 GB/s 下限；queue 放置不改变字节 |
| `topk_submit` | top-k 提交/首次触达 | 当前实现成本 | prime 后约 2.1 ms/block，不是主项 |

因此，`router_wait` 是“语义上需要 route 可见，但测量上包含大量必要设备
工作”；`bank_wait` 是 host/device ownership wait；canonical prefill 中没有
独立的 status-query 开销。`topk_submit` 的 55 s 首块异常只出现在未 warm
的 8192 首块，不是本 14K canonical 路径。

## 3. “若 X 免费” TTFT 上界

基线 53.202-54.057 s；目标 35 s。下列均为诊断上界，不代表可直接实现。

| X -> 0 | 观测值 | TTFT 上界 | 解释 |
|---|---:|---:|---|
| `router_wait` interval | 26.227-27.059 s | **27.0 s** | 只适合当诊断天花板；其中约 23-25 s 是必要 attention/HC 设备工作，且 route 依赖仍在 |
| `experts_host` interval | 26.261-26.303 s | **26.9-27.6 s** | 物理 H2D/fill 仍在，不能把 host interval 归零当作供给侧收益 |
| H2D 字节全部免费 | 18.5-18.6 s | **34.6-35.5 s** | 刚好达到 35 s 附近；需要所有冷专家驻留或零拷贝，而 51 GiB 补集装不进 16 GB |
| `fill` 免费 | 9.17-9.21 s | **44.0-44.9 s** | conservation：时间会转移到 `bank_wait`；单独 free fill 不够 |
| `bank_wait` 免费 | 8.11-8.28 s | **45.1-45.8 s** | 仍保留 H2D 和 fill；单独 free bank wait 不够 |
| 完整 cold-key union（72.2k -> 21.8k slots） | H2D 约 -70% | **约 39.9-40.8 s** | 理论上界；当前 2792 个 resident slots 不可能容纳 21.8k 个 cold keys，实际会小很多 |
| measured host-USM kernel read | 9.76-9.85 GB/s | **约 48-49 s** | `hostread.log` 已测；direct host read 只与 DMA 同速，不能解决 35 s |
| supply 与 device 真正重叠 | 当前 law 为相加 | **约 30-35 s（条件性）** | 只有 route 依赖/队列 law 同时被改写才成立；C1/C2、G5 layer-stage 已实测未重叠 |

关键算术：**H2D-free 已经是最乐观的供给侧天花板，且只到 35 s。**
因此 35 s 不可能由“再调一个 staging 参数”得到；至少要做到接近全字节消除，
或者在字节减少的同时再削掉 attention/GDN 的设备时间。

## 4. 结构性候选（按上界 × 可行性排序）

### A. Layer-major 4 x 4096 的 cold-union 复用（首选 probe）

当前已有 layer-major 原型，但它只接受恰好 8192 rows，并在内部拆成两个
4096-row attention window：`src/models/Qwen4ExpLayerMajor.cpp:110-118`、
`src/models/Qwen4ExpBnb4Backend.hpp:412-426`。它每个 cold bank 只 stage
一次，供两个 4096-row half 消费；这已经证明了“每层 union”代码路径存在，
但 8192 e2e 因 32 s 级 `bank_wait` 和非 expert residual 实测到 62.3 s，
不能直接作为生产方案。

候选不是重跑 chunk-8192：保持 canonical 的 **4096 attention window**，
把 14K 的四个窗口放进同一 layer pass，并让每层 cold set 只上传一次。
理论上界：

```
14K staged 72.2k -> union 21.8k
H2D 174.6 GiB -> 52.7 GiB
节省约 13.3 s；TTFT 约 40.8 s
```

这个上界仍未到 35 s，但它是唯一能把 H2D 字节压到 1/4、且不依赖
51 GiB 常驻的供给侧结构。必须实测，不可从旧 8192 结果外推。

Probe A（下一次 GPU slot，读只读）：

- 把现有 `LayerMajorPrefillEligible` 的 `count == 8192` 探针扩成
  **4 个 4096 window** 的临时分支；不改默认值。
- 输入必须为 13944/14000 token 级真实 prompt，测试窗口不能短于 QSA
  启用门槛；读 `[Q4E LayerMajorLayer]` 的
  `staged/h2d_mb/fill_ms/bank_wait_ms/experts_ms/wall_ms`。
- Gate：H2D <= 60 GiB/prompt，TTFT <= 48 s（至少 -10%），输出 raw token
  与 chunk-major 逐 id 一致；`bank_wait` 或 non-expert residual 若仍占主导，
  该路径 NO-GO。

### B. Prompt-scoped cold-key promotion（低风险备份 probe）

不改变 4096 chunk 顺序，只把跨 block 重复出现的 cold key 在 prompt 生命周期
内提升到现有 resident 池，prefill 结束后回滚；不改变 decode hotset 的
长期内容。实际可用容量只有约 2792 slots，因此不能实现完整 21.8k union，
但可以减少后续 block 的 `staged`/H2D。上界是 15-40% staged 字节下降，
对应 TTFT 约 47-51 s，单独不达 35 s；与 A 叠加才有意义。

Probe B（CPU-only replay，先不跑 GPU 性能臂）：

- 从一次真实 14K prompt 的 route trace（或 `AILA_Q4E_HOTSET_EXPORT`
  的 4-block 记录）重放四个 4096 block。
- 用 2792 / 2400 slots 两种预算模拟 LRU + 只提升高频 cold key；记录
  `staged` 降幅、decode hotset 损失、每层命中分布。
- Gate：staged bytes >= -15%，且 decode continuation 不低于基线 15.4
  tok/s；达不到则只作为诊断，不进源码。

## 5. 不建议重做/不能作为候选

- **chunk 8192 / staging 64 / ring 4**：charter §2 已否证；当前 8192
  仍有首块 top-k first-touch 异常。
- **C1 stream-ahead / C2 copy-only queue**：已有四族 receipt 证明
  memory-bound H2D 与设备工作相加；不是本轮候选。
- **G5 whole-layer arena（`AILA_Q4E_LAYER_STAGE=1`）**：实测 NO-GO；
  bank_wait 被搬到 `router_wait`，块墙 19.87-20.59 s、decode 6.83/10.81
  tok/s。
- **host-USM direct expert read**：实测 9.76-9.85 GB/s，与 DMA 同速；
  只能得到约 48-49 s，不能单独解决 35 s。
- **topk-split patch**：第一个 half 的 event 仍覆盖整条 attention 链，
  预期 |delta| <= 0.2%；不作为候选。
- **CPU-MoE prefill**：1.966M rows/block x 53.7 us 约 105 s/block，
  已排除。

## 6. 建议的下一步

先写 Probe B 的 CPU replay（成本最低，可当天完成），同时把 Probe A 作为
下一次 GPU slot 的唯一 prefill 结构实验。A 的目标不是一次打到 35 s，而是
验证“4 x 4096 window + 每层 union”是否能把 H2D 从 174.6 GiB 真实压到
60 GiB 以下；若 A 的 non-expert residual 仍超过 10 s，则 35 s 需要另立
device-side attention/GDN 研究，不应继续在 staging 参数上迭代。
