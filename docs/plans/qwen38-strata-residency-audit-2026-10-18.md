# Qwen4Exp / Strata 组件驻留审计（2026-10-18）

范围：只读审计当前 `main`（审计起点 `ab28c3c`）。源码行号按该
`HEAD` 固定；审计期间出现的未提交并发实验改动不属于默认/生产路径，
也未纳入本报告。本文件对照
`tmp/q4e-strata-principle/REPORT.md` 的驻留原则，检查 Qwen3.8-Flash-Next
在 Qwen4Exp 路径上的 GR/HC、router、shared expert、token embedding、LM head、
MTP draft、KV 与 GDN state 的真实存储位置和默认/生产选择。没有运行 GPU，
没有修改源码。

日志里的 `MB` 是代码按 `1024 * 1024` 计算的 MiB；本文统一写 MiB/GiB。

## 0. 结论

1. **GR/HC、router 权重与计算、shared experts、LM head、目标模型 KV、
   GDN 热状态都在 GPU/USM device 上**，与 Strata 的“每 token 必经项优先
   驻留 VRAM”一致。Aila 的 GR 就是 `HyperConnectionOps`/HC，不是两套独立
   权重；见 `tmp/Strata/src/kernels/cuda/gr.cu:1` 与
   `src/models/Qwen4ExpBnb4Backend.cpp:584-596,613-614`。
2. **唯一明确的驻留偏差是 token embedding**：默认
   `AILA_Q4E_HOST_EMBED=1`，1.184 GiB 的表被复制到原生大页 host，
   `EmbedTokens()` 每个 forward 做一次阻塞的 token-id D2H、CPU gather、
   再 H2D；见 `src/models/Qwen4ExpBnb4Backend.cpp:540-545,784-787,2066-2083`。
   但它不是已经证明的瓶颈：84-136-token 日志的 `ple_host_ms` 中位约
   0.240-0.251 ms，且该桶还包含 PLE；按 2.5344 MB/slot 计算，原样上传
   BF16 表要占 **501.7 slots**，大概率负收益。不能把“结构上在 host”
   直接等同于“应该上传 BF16”。
3. **MTP 在生产 profile 中实际没有加载**。profile JSON 写了
   `AILA_Q4E_SERVER_MTP=1`，但生产启动器和 sprint runner 都覆盖为 0；
   32K 上下文也超过 `enableMtp` 的 2048 上限。若短上下文 opt-in：
   dense draft 权重在 GPU，但 512 个 MTP routed experts 只有固定的
   64 个 GPU slots（154.7 MiB），其余 1.083 GiB 仍从 mmap 按需填充；
   共享 embedding 仍在 host。这与 Strata “MTP 草稿层全量权重在 GPU”
   不同，见 `src/inference/Model.cpp:726-727,800-837` 与
   `src/models/Qwen4ExpMtpDraft.cpp:12-15,63-73`。
4. 目标模型的 24576 个 routed expert 仍按 Strata 式分层驻留：
   生产 h4/BF16 为 **2792 GPU + 21784 host = 24576**；当前 sprint
   agent6k/int8 为 **2800 GPU + 21776 host = 24576**。host 补集是原生
   大页，不是逐 token 的 mmap 后备路径；约 51.4 GiB。见
   `src/models/Qwen4ExpHostArena.cpp:135-146` 与
   `tmp/q4e-sprint-r1-20261014/xp-a1-worker.log:7-10`。
5. 目标模型 VRAM 账本（日志实测）：
   - 生产 h4 / 32K / BF16：`4522.3 static + 1000.2 session + 3584 reserve
     + 6826 expert cache = 15932.5 MiB`，admitted 2792 slots。
   - 当前 sprint agent6k / 32K / int8：`4522.3 static + 622.2 session +
     3584 reserve + 7204 expert cache = 15932.5 MiB`，admitted 2800 slots。

## 1. Strata 的驻留原则与 Aila 映射

Strata 的 GPU 固定项包括 attention/DeltaNet mixer、gated residual /
hyper-connection、router、shared expert、output head、MTP draft、KV 活跃窗口
和 GDN 状态；所有 routed experts 的完整集合在 host RAM，GPU 只保留热专家
cache。原文见 `tmp/Strata/docs/HOW_IT_WORKS.md:22-28`。

Aila 的映射如下：

| Strata 原则 | Aila 实现 | 实际判断 |
|---|---|---|
| gated residual / hyper-connection | `Qwen4ExpHyperConnectionLayer` + `HyperConnectionOps` | 同一组 HC 权重，全部 device |
| router | `router_gate` / `router_w` | 权重和 top-k 计算 device；top-k 结果写入 pinned host |
| shared expert | `shared_gate_up_proj` / `shared_down_proj` / `shared_expert_gate` | 全部 device |
| token embedding | `embed_host_` 大页 host 副本 | host，唯一明确偏差 |
| output head | `lm_head_weight_transposed_` 或 Q8 head | device；Q8 仅 opt-in |
| MTP draft | `Qwen4ExpMtpRuntime` / `Qwen4ExpMtpDraft` | 生产不加载；opt-in 时专家只部分驻留 |
| KV / GDN | `Qwen4ExpLayerCache` | 热状态 device；GDN 另有 pageable host mirror |

Strata 的 KV 在 64K 以上才开始把完整 KV 放 host、VRAM 只保留活跃窗口；
Aila 生产上下文是 32K，当前不实现 KV streaming 是合理的。当前不需要把
“KV 流转”作为本轮结构改动；见 `tmp/q4e-strata-principle/REPORT.md:7,31-35`。

## 2. 默认路径与生产 profile 的真实值

### 2.1 生产启动器

`tools/server/start-qwen38.ps1` 先加载
`docs/plans/qwen38-server-mtp-2026-10-07.json`，随后覆盖关键项：

| 项 | profile JSON | 启动器实际 | 结论 |
|---|---:|---:|---|
| `AILA_Q4E_SERVER_MTP` | 1（JSON:101） | **0**（`start-qwen38.ps1:19`） | MTP 不加载 |
| `AILA_Q4E_VRAM_RESERVE_MB` | 3008（JSON:8） | **3584**（`start-qwen38.ps1:22`） | 生产 reserve 3584 |
| `AILA_Q4E_HOTSET_PROFILE` | h4（JSON:80） | h4（`start-qwen38.ps1:23`） | 专家排序用 h4 |
| `AILA_KV_QUANT` | 0（JSON:102） | 0 | 生产是 BF16 KV |
| `AILA_Q4E_LM_HEAD_Q8` | 0（JSON:58） | 0 | BF16 输出头 |
| `AILA_Q4E_HOST_EMBED` | 未设置 | 代码默认 1（`Backend.cpp:540`） | host embedding |
| `AILA_Q4E_GDN_CHUNK_MODE` | 0（JSON:40） | 0 | 非 chunkwise |

生产 launcher 的 32K 配置实际得到：

```
Total VRAM=15933 MB, Static Weights=4522.3 MB,
Session State=1000.2 MB (KV=768.0, IndexerK=96.0,
IndexerCompressedK=24.0, Recurrent=112.2),
Reserve Headroom=3584 MB, Expert Cache=6826 MB
```

receipt：`tmp/q4e-sprint-r1-20261014/w4-committed-control-worker.log:7-10`。
该 cache 装下 **2792 hot + 32 staging** slots，host 补集 21784 experts /
51.46 GiB。

### 2.2 当前 sprint runner

`tmp/q4e-sprint-r1-20261014/run.py` 同样把 profile 的
`AILA_Q4E_SERVER_MTP` 覆盖成 0（`run.py:36`）。本轮 canonical bundle 另外
显式使用 agent6k hotset + int8 KV，见
`tmp/q4e-sprint-r1-20261014/xp-a1-worker.log:7-10,18`：

```
Static Weights=4522.3 MB
Session State=622.2 MB
  (KV=384.0, kv_scales=6.0, IndexerK=96.0,
   IndexerCompressedK=24.0, Recurrent=112.2)
Reserve=3584 MB, Expert Cache=7204 MB
admitted_slots=2800; native RAM complement=21776 experts, 51.44 GiB
```

这里 7204 MiB 是准入预算，不是全部被 slot payload 用掉：
`2800 hot + 32 staging` slots 的 payload 是 **6844.9 MiB**，因为
`AILA_Q4E_MAX_GPU_SLOTS=2800` 先撞到硬上限，剩余约 359 MiB 留作
cache 内部余量。slot 算术见 `Qwen4ExpBnb4Backend.cpp:802-810`。

### 2.3 结论

生产 h4 与 sprint agent6k 的差异主要是：

- KV：BF16 1000.2 MiB session vs int8 622.2 MiB session；
- expert slots：2792 vs 2800；
- hotset：h4 vs agent6k。

它们都没有改变 GR/HC、router、shared expert、embedding、LM head 的
组件级驻留方式。hotset profile 只决定哪些 `(layer, expert)` 进入 GPU
slot，不是一份“把这些组件搬到 host/device”的配置。

## 3. 组件逐项审计

### 3.1 GR / HC

**实际路径：device。**

`Qwen4ExpHyperConnectionLayer` 同时保存 `hc_norm`、low-rank down/up 和
inject 权重（`Qwen4ExpBnb4Backend.hpp:29-37`）。`load_hc()` 对每一项调用
`weights.upload(..., ctx)`；norm 再被 `ensure_f32()` 转成 device f32，
见 `Qwen4ExpBnb4Backend.cpp:584-596`。每层有 attention HC 和 MLP HC，
另有全局 mixer；没有 host 权重副本参与计算。

字节：

- 48 层 × 2 个 HC + 全局 mixer，共 97 个 HC 实例；
- BF16 down 6.554 MB + up 6.554 MB，96 个带 inject 的实例再加 0.082 MB；
- 源权重总计约 **1223.8 MiB**（含 97 个 norm 的 f32 化增量）；
- decode 每 token 约读 **1.279 GB**，见
  `docs/plans/qwen38-r1-decode-misc-2026-10-15.md:49-59`。

判断：**与 Strata 对齐**。现有 8.4 ms/token 级别的时间是 194 次
kernel submit 和 host pacing，不是 host 权重计算；不能靠“把 HC 搬回
GPU”改善，因为它本来就在 GPU。

### 3.2 Router

**实际路径：router 权重和 top-k 计算在 device；只有 route 结果在 pinned host。**

48 个 router 权重 [512,2560] BF16 共 **120.0 MiB**，加载和绑定见
`Qwen4ExpBnb4Backend.cpp:691-695`；router score 缓冲在 device
（`Qwen4ExpBnb4Backend.cpp:1139-1141`）。top-k kernel 把 ids/weights 写入
`router_host_ids_`/`router_host_weights_`，这两个是 `ctx.alloc_host`
（`Qwen4ExpBnb4Backend.cpp:1146-1149`），用于 host 侧专家分类。

生产实际：`AILA_Q4E_ROUTER_RESIDENT_FUSED=0`，所以走 device top-k，
host 只消费结果；这不是“router 在 host 算”。pinned route 缓冲在
cap=4096 时约：

- ids 4096 × 10 × 4 = 160 KiB；
- weights 4096 × 10 × 4 = 160 KiB；
- 预测镜像另约 320 KiB。

判断：**驻留对齐，控制面仍需优化**。11 ms/token 量级的 `router_wait`
是等待 device 发布 route，不是 host 权重；与 Strata 的 doorbell 对比
应研究重叠/唤醒，而不是把 router 权重搬到 host。

### 3.3 Shared experts

**实际路径：device。**

每层 `shared_expert.gate_proj`、`up_proj`、`down_proj` 通过
`init_quant_linear()` 上传，gate/up 再 fused rows；`shared_expert_gate`
单独上传，见 `Qwen4ExpBnb4Backend.cpp:697-709`。源 payload 共
**116.478 MiB**（48 层，含 nested NF4 metadata；不含运行时 scratch）。

判断：**与 Strata 对齐**。`AILA_Q4E_MOE_SHARED_FUSED=0` 只是 launch
形态，不是 host 驻留；shared expert 的 2.7 ms/token 是 144 次小 launch
和带宽/提交问题，不是权重不在 GPU。

### 3.4 Token embedding

**实际路径：原生大页 host + 每 forward CPU gather。**

默认 `AILA_Q4E_HOST_EMBED=1`：

1. `weights.host_view()` 从 mmap 取得 [248320,2560] BF16 表；
2. load 时 `ctx.alloc_large_host()` 复制一份到原生大页，见
   `Qwen4ExpBnb4Backend.cpp:540-545,784-787`；
3. `EmbedTokens()` 每个 forward 先 `ctx.memcpy_d2h()` 读 token ids，
   阻塞 queue（`Context.hpp:185-190`），CPU 逐行 memcpy，再 async H2D
   （`Qwen4ExpBnb4Backend.cpp:2066-2083`）。

字节与 slot 代价：

- 表：248320 × 2560 × 2 = **1,271,398,400 B = 1.184 GiB**；
- 2 MiB 大页对 block 的物理占用约 1.186 GiB；
- 按 2.5344 MB/expert slot，等价 **501.7 slots**；
- decode 每 token 只搬 5 KiB 行，但多一次阻塞 D2H；prefill 的 4096
  block 搬 20 MiB 行，另有一次 token-id D2H。

**不要把这条直接判成“应上传 BF16”。** 现有日志的 84-136-token
`ple_host_ms` 中位约 0.240-0.251 ms，虽然该桶包含 PLE，仍说明当前
embedding/PLE host 侧没有出现量级明显的尖峰；见
`tmp/q4e-sprint-r1-20261014/xp-a1-worker.log:39-63` 的采样
（0.242-0.281 ms）。在 isolate 出 `EmbedTokens` 自身之前，原样上传会
用掉约 502 个 expert slots，收益没有证据。

判断：**结构上偏离 Strata，当前不应直接改 BF16 GPU 表**。最小且可逆的
下一步是：

- 把 host 已知的 token ids 从 scheduler 传到 backend，消除 decode 的
  阻塞 D2H；这是没有数值变化的低风险改动；
- 若要试 GPU 表，只试量化/分阶段表并保持槽位预算相等；BF16 原表上传
  应作为负对照，不作为默认方案。

### 3.5 LM head

**实际路径：device；默认 BF16，Q8 仅 opt-in。**

默认把原始 BF16 `lm_head.weight` 上传，再转置成 device
[2560,248320]，保留 `lm_head_weight_transposed_` 并删除 raw device 副本，
见 `Qwen4ExpBnb4Backend.cpp:548-580`。生产 profile 明确
`AILA_Q4E_LM_HEAD_Q8=0`，所以实际是 **1212.5 MiB device head**。

Q8 opt-in 的实际大小：

- weight 248320 × 2560 = 606.25 MiB；
- scale 248320 × 40 × 4 = 37.89 MiB；
- 合计 **644.14 MiB = 0.629 GiB**；
- 相对 BF16 省 **568.36 MiB = 0.555 GiB**，等价 **235 slots**。

判断：**驻留与 Strata 对齐**。Q8 只是 opt-in 资源/带宽选项，不改变
“head 在 GPU”这一结论；且它和 MTP 互斥（`Model.cpp:800-802`），不能
为了省 235 slots 而同时宣称支持 MTP。load 时 raw+transposed BF16
会短暂同时存在，约 2.425 GiB 峰值；这是加载期风险，不是稳定态驻留。

### 3.6 MTP draft weights

**生产实际：没有加载。**

`AILA_Q4E_SERVER_MTP` 默认 false（`Model.cpp:726-727`），生产启动器和
runner 又显式设 0；同时 `Model.cpp:800-802` 要求
`maxSequenceLength <= 2048`、GDN chunk mode 0、Q8 head 关闭。生产 32K
请求不满足，故 `loaded->mtp` 不构造（`Model.cpp:836-837`）。
`AILA_Q4E_MTP_HIDDEN_REUSE=1` 出现在 profile JSON（`:51`）只对已加载的
MTP runtime 生效，不能拿它证明 MTP 已启用。

**短上下文 opt-in 时的实际路径：**

- dense draft 投影、HC、router、shared 均 `weights_.upload` 到 device；
  MTP 非专家源权重约 **76.75 MiB**；
- MTP 自身 512 experts 只分配固定 **64 个 GPU slots**
  （`Qwen4ExpMtpDraft.cpp:12-15,63-73`），每 slot 2,534,400 B，
  **154.69 MiB**；
- 其余 MTP expert 源 payload 仍在 mmap；源总专家约
  **1.209 GiB**，即约 **1.083 GiB 不在 GPU slot 中**；
- MTP 源权重总计约 **1.284 GiB**，其中非专家约 76.75 MiB；
- shared embedding 仍通过 host view，每个 draft Step 还做一次 blocking
  H2D（`Qwen4ExpMtpDraft.cpp:167-178`）；
- 每会话 draft KV+indexer 约 **4.5 MiB @2048**，是 device tensor
  （`Qwen4ExpMtpRuntime.cpp:35-38`；`Qwen4ExpMtpDraft.cpp:157-162`）；
- runtime 还分配 device `hiddenRows` 40 MiB、checkpoint 112.56 MiB、
  linearInputs 6.04 MiB 等（`Qwen4ExpMtpRuntime.cpp:16-25`）。

`Qwen4ExpMtpRuntime` 只有 width=2 的 proposal/verify 窗口
（`Qwen4ExpMtpRuntime.cpp:27,113-165`），不是 Strata 的多 token
宽窗口；draft Step 还在 `route.wait_and_throw()` 处同步（
`Qwen4ExpMtpDraft.cpp:234`）。

判断：**生产不是 MTP 问题；若重新启用，先补 residency 和 host 控制面**。
当前 Aila 的 MTP 不是 Strata 的“草稿层全量 512 experts 在 VRAM”，而是
“dense 全 GPU + 固定 64-slot 专家 cache + mmap 补齐”。这会在短会话
把 draft 冷专家供给重新变成 host/PCIe 问题；不应在 32K 生产 profile
上偷偷打开。

### 3.7 KV 与 GDN state

**目标模型热状态：device。**

Session 构造：

- full attention 12 层：K/V [2, capacity, 256]，BF16 或 opt-in int8；
- QSA indexer：`indexer_k` 与 4:1 `indexer_compressed_k`；
- 36 个 GDN 层：`linear_state` [48,128,128] f32 与
  `linear_conv_state` [3,10240] f32；
  见 `Qwen4ExpBnb4Backend.cpp:231-273`。

32K 目标模型：

| 项 | BF16 生产 | int8 sprint |
|---|---:|---:|
| full KV K/V | 768.0 MiB | 384.0 MiB |
| KV scale | 0 | 6.0 MiB |
| indexer K + compressed | 96.0 + 24.0 MiB | 同左 |
| GDN recurrent + conv | 112.2 MiB | 同左 |
| session total | 1000.2 MiB | 622.2 MiB |

KV 没有 host copy、没有 KV streaming；这与 Strata 在 32K 以下把活跃 KV
放 VRAM 的做法一致。

GDN 额外有一份 **pageable `std::vector<float>` host mirror**：
`linear_state` 108.0 MiB + conv 4.22 MiB = 112.2 MiB/session，见
`Qwen4ExpBnb4Backend.hpp:92-99` 和
`Qwen4ExpBnb4Backend.cpp:251-255`。hot token path 不读写它；只有
CaptureSnapshot/Truncate/Reset 使用（`:3627-3668,3738-3756`）。Worker
又设置 `maxRecurrentSnapshots=0`（`src/worker/SharedInferenceEngine.cpp:77-80`），
但 Session 构造仍无条件分配这份 mirror。判断：热状态正确，**worker
的 112.2 MiB/session mirror 是可省的冷内存**。

### 3.8 PLE 表（边界说明）

PLE 的 29.8 GiB ngram 表不参与上面的 device 账本；`Qwen4ExpPle` 保留
mmap shard 指针，并按 token 读取 16 行（`src/models/Qwen4ExpPle.cpp:221-240,484-505`）。
PLE 的非 ngram 投影/卷积约 16.26 MiB 是 device。Strata 也把大 lookup
表放 SSD，因此这不是本轮要修的驻留偏差。

## 4. VRAM 与 host 账本

### 4.1 目标模型 VRAM

| 项 | 生产 h4 / BF16 / 32K | sprint agent6k / int8 / 32K |
|---|---:|---:|
| static device weights | 4522.3 MiB | 4522.3 MiB |
| session state | 1000.2 MiB | 622.2 MiB |
| reserve | 3584 MiB | 3584 MiB |
| expert-cache budget | 6826 MiB | 7204 MiB |
| admitted hot slots | 2792 | 2800 |
| staging slots | 32 | 32 |
| slot payload | 6825.6 MiB | 6844.9 MiB |
| host complement | 21784 experts / 51.46 GiB | 21776 / 51.44 GiB |

静态 4522.3 MiB 的可核对构成（源 payload + 代码实际 f32 化/融合后口径）：

| 组件 | MiB | 占 static |
|---|---:|---:|
| GR/HC | 1223.8 | 27.06% |
| LM head BF16 | 1212.5 | 26.81% |
| linear-attention projections | 1028.8 | 22.75% |
| full-attention projections | 331.6 | 7.33% |
| router | 120.0 | 2.65% |
| shared experts | 116.5 | 2.58% |
| PLE dense（非 ngram） | 16.3 | 0.36% |
| NF4 f32 absmax/fused padding/其他 | 472.8 | 10.46% |

最后一行不是一个“未知大组件”：`Bnb4BitLinear::init()` 会为每 64 个
权重物化一组 f32 absmax（`src/ops/Bnb4BitLinear.cpp:1834-1857`），
fused rows 另有连续 packed/absmax 分配（`:1958-1959`）。精确拆到每个
Linear 需要 per-allocation 计数器；本次不再为 10% 的边角做过度验证。

### 4.2 host 账本

| host 项 | 大小 | 机制 |
|---|---:|---|
| routed expert complement | 51.44-51.46 GiB | `MEM_LARGE_PAGES`，不是 USM pinned |
| token embedding | 1.184 GiB nominal / ~1.186 GiB rounded | 原生大页副本 |
| GDN mirror | 112.2 MiB/session | pageable `std::vector<float>` |
| route ids/weights | ~0.3 MiB | USM pinned `alloc_host` |
| embed ids/rows | ~20 MiB @cap4096 | USM pinned `alloc_host` |
| adaptive host staging | 77.3 MiB | 2 banks × 16 slots，USM pinned |
| PLE ngram table | 29.8 GiB | mmap/按需，属于设计内 |
| MTP | 0 | 生产未加载 |

host 的 native expert 大页和 PLE mmap 是关键结构：前者是 Strata 的
“host RAM 保存 routed experts”，后者对应 Strata 的 SSD lookup table。
真正多出来的是 embedding 大页和 worker 不使用的 GDN cold mirror。

## 5. 风险与最小修法

按“证据强度 × 影响”排序：

### P1 — 先给 `EmbedTokens` 单独打点，不先搬 BF16 表

当前证据不足以证明 host embedding 是瓶颈。最小探针：

- 在 `EmbedTokens()` 外包一段可选统计，分别记录 D2H wait、CPU gather、
  H2D submit 的 ns 和字节；
- 用 84/136-token agent 续轮各一次，先看是否 >1 ms/token；
- 若 D2H 是主项，优先传 host token ids，保持表仍在 host；
- 只有隔离计时证明值得，才比较“量化 GPU 表”与“分阶段/热行表”，
  并保持 GPU slot 预算相等。

`AILA_Q4E_HOST_EMBED=0` 的 BF16 上传不是默认候选：会占约 501.7 slots，
在当前 2800-slot cap 下等价于把 hot expert 覆盖直接砍掉一大块。

### P1 — worker 在 snapshots=0 时不要分配 GDN mirror

`SharedInferenceEngine` 已明确设置 `maxRecurrentSnapshots=0`，但
`Qwen4ExpBnb4Backend::Session` 仍为每个 GDN 层分配 pageable host vector。
最小修法是让 host mirror 的 `assign()` 受 `maxSnapshots > 0` 控制，
并让 `qwen4exp_session_memory()` 的 host reservation 对 0 snapshots
返回 0；CaptureSnapshot 本就因 snapshots 为空提前返回，Truncate 走
Reset 路径，不需要 mirror。这样每 worker session 省 **112.2 MiB**，
不碰 token 热路径。

### P2 — 明确 MTP 的实际启用条件

保留默认关闭；在启动日志中打印“requested/实际 enabled/reason”，
避免 profile JSON 的 `SERVER_MTP=1` 被误读。若未来重新评估短会话
MTP，先决定是否接受：

- Strata 式全量 512 draft experts GPU（约 1.209 GiB 源 payload），或
- 当前 64-slot 方案（154.7 MiB）并把冷供给路径计入；
- width=2 与 host embedding D2H 的成本。

不能同时默认 Q8 head 与 MTP：`Model.cpp:800-802` 已把两者设为互斥。

### P2 — LM head 的加载峰值

稳定态 head 在 GPU，不需要 residency 迁移；但默认 load 会同时持有
raw BF16 与 transposed BF16，峰值约 2.425 GiB。若要降低启动峰值，
应做“直接转置上传”或 Q8 一次转换，而不是把 head 留 host。Q8 的
0.629 GiB 常驻值和 235 slots 收益只作为 opt-in 证据，未达到默认
门槛就不要默认化。

### P3 — router/shared/HC 不做 residency 迁移

它们的权重已经在 device；当前 `router_wait`、`shared_submit`、`hc_host`
是提交/同步和 kernel shape 问题。把权重搬到 host 只会把带宽和延迟
问题放大。后续若做，应走 Strata 式重叠/门铃/设备参数表，不是改
存储所有者。

### P3 — KV

32K 目标模型保持全量 device；当前没有必要做 host KV 流转。只有实际
进入 64K+ 生产上下文，才重新评估 Strata 的 `--kv-resident` 类窗口，
且必须先过 A/B 和输出一致性门。

## 6. 建议的下一步探针（本轮未执行）

1. `embed_ms` 三联计时（D2H/gather/H2D），用 84 与 136 token 两个
   生产长度；不要用合成短 prompt 归因。
2. 条件化 GDN mirror 的单元验证：`maxSnapshots=0` 时 host bytes 降为
   0，`Reset/Truncate` 仍通过；不跑 full GPU 也可先做 CPU/ABI。
3. MTP 启用日志断言：短上下文 opt-in 时打印实际 draft slot 数、
   host/mmap expert 字节、width 和 host-embedding 调用次数；长上下文
   必须明确打印 disabled reason。
4. 若以后做 embedding GPU A/B，固定 int8 KV、hotset、reserve、
   `MAX_GPU_SLOTS` 和 chunk；只把 placement 作为单变量，先看
   decode/prefill 与 slot 命中，不要只看单次 kernel。

所有这些探针都应遵守 GPU 互斥、无固定长 sleep、先 oracle 后实模的
纪律；本审计没有跑 GPU，因此以上均为后续执行项。

## 7. 证据索引

### Strata 原则

- `tmp/q4e-strata-principle/REPORT.md:7,27-35,42,60-63`
- `tmp/Strata/docs/HOW_IT_WORKS.md:22-28`
- `tmp/Strata/src/kernels/cuda/gr.cu:1`

### Qwen4Exp load / components

- `src/models/Qwen4ExpBnb4Backend.hpp:29-37,80-99,571-586,607-613`
- `src/models/Qwen4ExpBnb4Backend.cpp:231-273,540-580,584-596,613-614,691-709,722-810,1146-1149`
- `src/models/Qwen4ExpBnb4Backend.cpp:2066-2083`
- `src/ops/Bnb4BitLinear.cpp:1834-1857,1958-1959`
- `src/core/Context.hpp:66-80,185-190`
- `src/memory/SessionMemory.hpp:107-208`
- `src/models/Qwen4ExpHostArena.cpp:135-146`
- `src/worker/SharedInferenceEngine.cpp:77-80`

### MTP

- `src/inference/Model.cpp:726-727,800-837`
- `src/models/Qwen4ExpMtpDraft.cpp:12-15,63-73,148-164,167-178,234`
- `src/models/Qwen4ExpMtpRuntime.cpp:9-27,35-46,113-165`

### profile / receipts

- `docs/plans/qwen38-server-mtp-2026-10-07.json:8,40,51,58,80,101-102`
- `tools/server/start-qwen38.ps1:19,22,23,37-40`
- `tmp/q4e-sprint-r1-20261014/run.py:34-36`
- `tmp/q4e-sprint-r1-20261014/xp-a1-worker.log:7-10,18`
- `tmp/q4e-sprint-r1-20261014/w4-committed-control-worker.log:7-10`
- `docs/plans/qwen38-r1-chunk8192-vram-analysis-2026-10-16.md:297-309`

审计结论：**不要把“GPU 常驻项”与“GPU 计算耗时”混为一谈。** 当前
除了 token embedding 的 host gather 和 worker 的无用 GDN mirror，其他
被问到的关键组件在默认/生产路径上都已经是 device；下一步应先做
embedding 隔离计时和 mirror 条件化，而不是盲目把 BF16 embedding 或
router/shared/HC 搬上/搬下 GPU。
