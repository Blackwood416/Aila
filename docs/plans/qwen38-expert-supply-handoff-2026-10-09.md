# Qwen3.8 专家供给后续优化交接

日期：2026-10-09。当前源码 `HEAD=39625195bdadb996a77248b0d0136d01bcc3c849`，父提交为 H4 profile 接入 `c5d5805ef788802929b857a11a62f194dc635daf`。本文把 H4 初始驻留筛选与后续 staging copy 实验串成一条证据链；PCIe 按用户确认的 **3.0 x16** 讨论。本文件只作交接，不把待验证假设写成测量结论。

## 已有证据链

### H4：改变初始 GPU 专家集合

H4 通过 profile 选择初始驻留的 2792 个专家，RAM 中保留互补的大页补集。详细的实现、所有权、身份校验、原始对照和限制见[生产筛选报告](qwen38-h4-residency-screen-2026-10-09.md)及其 [JSON 结果](qwen38-h4-residency-screen-2026-10-09.json)，交换与路由审计见 `tmp/q4e-h4-residency-20261009/swap-route-audit.md`。

在独立 16-token decode 诊断中，profile 前后 GPU 命中率为 **14.49% → 50.81%**，CPU expert task 中位数为 **411 → 240**。这是供给位置变化的证据，不等于 prefill 会同比缩短：命中按路由出现计数，而 DMA 按每层去重后的专家计数。生产筛选中续轮总耗时改善，但 14K 新请求总耗时仅约 -0.53%；不能据此声称显著 prefill 加速，也不能把诊断数当自然语料的通用命中预测。

H4 的槽预算、动态 swap 预算/周期、PLE、router 和专家数学保持不变。GPU 与 native large-page RAM 仍是互补 owner；不做全量双份常驻。profile 身份采用采样 FNV 契约而非全模型密码学 digest。后续实验应固定 H4 profile 和已准入 key 集合，避免把 placement 变化与传输/计算实现变化混测。

### Staging 3a：保持 ring，并行化 host copy

完整生产筛选、反序复核、正式身份与验收见 [staging copy 报告](qwen38-staging-copy-screen-2026-10-09.md)及其 [JSON](qwen38-staging-copy-screen-2026-10-09.json)。

当前 HEAD `3962519` 将同一冷专家 batch 内各 slot 的 native large-page RAM → Host USM `memcpy` 放进已有受限 TBB arena。它**没有扩容 ring**：保留原 2×16 staging banks、同一批次 barrier、专家顺序、单次 H2D 提交点、H2D 字节数、bank reuse / staging-reader / consumer event fence；CPU expert math 仍由现有 TBB 路径执行，不应描述成单线程 CPU math。实现提交只改当前供给路径，见 HEAD 的 7 文件 diff。

微测使用 2560×640 几何、每专家 2,534,400 B、16 个专家，单批 40,550,400 B（约 40.55 MB），20 组交错 serial/parallel；源为 native large-page，目标为 Host USM，计时外对完整 payload 做 memcmp。serial median **5.73105 ms**（p10/p90 5.6887/5.7712），parallel median **1.57525 ms**（p10/p90 1.5343/1.6186）。该微测只支持 host copy 路径变快，不能单独代表 PCIe 或整轮收益。normal `AdaptiveCacheTests` 的 staging fence、owner exchange、GPU 实际消费数值检查通过；详见 `tmp/q4e-staging-parallel-20261009/README.md` 与 `microbench.log`。

生产 A/B 顺序为 **A1 → B1 → B2 → A2**，唯一切换项是同一二进制中的 `AILA_Q4E_STAGE_COPY_PARALLEL=0/1`；A1/A2 为 0，B1/B2 为 1。四份结果均 `passed=true`，输入序列、输出 hash、token usage、finish reason、prefix reuse 与取消恢复 gate 一致。四轮记录的 Worker SHA 都是 `e93be7e2b3623dc150ff4429cd0d5922fac9fcee4b4394199d093f0f392c1530`，Server SHA 都是 `5b08f3937e948197a2d97cd3533781e58538111d50f459f22f7920dd5da877da`。原始文件在 `tmp/q4e-resident-speed-20261009/stage-{a1,b1,b2,a2}/` 和对应 `stage-*-run.log`。

| 指标（A1/A2 与 B1/B2 中位数） | flag 0 | flag 1 | 变化 |
|---|---:|---:|---:|
| 14K 新请求总耗时 | 72.61899 s | 72.86948 s | +0.34494% |
| 14K 续轮首字延迟 | 1.46704 s | 1.12644 s | -23.2167% |
| 14K 续轮总耗时 | 3.64211 s | 3.29091 s | -9.6429% |

这组 screen 的续轮有一致改善，14K 新请求略慢，已独立保留。结论限定在本测试序列，不据此预测其他 prompt；运行时默认关闭，`start-qwen38.ps1` 在调用者未设置时启用 flag=1，明确传入 `AILA_Q4E_STAGE_COPY_PARALLEL=0` 会保留显式关闭。测量用的 A/B 是同 SHA 二进制，提交后的官方 pair 另列如下，不能把 screen SHA 冒充为最终官方 pair。

## 正式构建与验收

Worker 与 Server 官方构建 exit=0，两端 build ID 均为 `aila-0.2.0-abi1-39625195bdadb996a77248b0d0136d01bcc3c849`。归档：`tmp/q4e-stage-copy-screen-20261009/retained/`，含两个 exe、FakeWorker、build metadata 和 `pair.json`；它是身份归档，不增加生产计时样本。

| 二进制 | SHA-256 |
|---|---|
| Worker | `18cea21c4ba83ec858d04ebb2f96df40ba3efb2a535786539d56c8b96e647e8a` |
| Server | `ab04ae7de51de8e7982be0f8a5a6c3e7d0b03b64ca79501e2267f60e7b46bfd5` |

提交版缓存专项通过 GPU 数值来源、并行 staging、consumer fence、连续 RAM/GPU owner 交换与 2800 唯一槽检查。CPU/ABI：18 passed、1 conditional skip、0 failed（176.41s）；Server：4/4 passed（24.74s）。启动脚本语法通过。正式日志及 `verification.json` 留在同一归档父目录。

## 候选状态与隔离方式

一次只改变一个变量；若 placement、工作量或输出轨迹随策略变化，按策略对照报告，不称为固定工作量 kernel 加速。日志中的 `h2d_mb` 按 MiB 计，不是物理 PCIe 计数器。所有总线/传输估算均按 PCIe 3.0 x16 实测，不套用 PCIe 4.0 带宽。

| 候选 | 已有证据 | 假设 / 当前状态 | 隔离要求 |
|---|---|---|---|
| **2a：swap 预算或周期策略** | 当前 adaptive 每 token 默认最多计划 2 次 swap，周期默认 1；`MAX_SWAPS=0` 表示不设总量上限，不是禁用。 | 可能减少不经济的交换/等待；尚未测量，未实现新逻辑。 | 只改预算或只改周期，另一个保持固定；profile、swap 实现、router 和 compute 不动。记录计划数、两方向传输、swap wait、命中、输出 hash。 |
| **2b：仅异步 H2D 尾部** | 当前 swap 的 D2H 和 H2D 都在 host wait；统计字节数少计了 D2H。swap 收尾在最终 mixer/head 前，具备潜在重叠窗口。 | 固定 budget=2、freq=1，保留 D2H 同步 host commit，仅让 H2D 与 final mixer/head 重叠；尚未实现。最小方案和 ticket 生命周期见 `tmp/q4e-h4-residency-20261009/swap-route-audit.md`。 | 独立预分配两个 transfer ticket；incoming RAM source 在 owner 更换前复制到各自 ticket；D2H 完成后才读回/提交 victim RAM 和交换 owner；H2D 全部入队后一次性发布 slot map/table，`ResidentReady()` 覆盖 H2D 与 table 更新。最终路径、异常、复用、`Release`/析构、move assignment 都要 drain 或保留 ticket，不能提前释放 H2D source。只做这一变量，不同时改预算。 |
| **3a：现有 ring 内的并行 staging copy** | 已在 HEAD 实现并完成微测、缓存专项、四轮同二进制 flag A/B 和正式构建/验收；续轮改善、14K 新请求总耗时略退。 | 已独立保留；这不是扩大 ring 的收益证据。 | 后续复核只切 flag 0/1，保持其他变量不变，不叠加 ring 容量变量。 |
| **3b：ring 2→4** | 当前 3a 未增加 bank；更多 bank 可减少复用时的暴露等待仅为假设。 | 未实施，见 `tmp/q4e-stage-copy-screen-20261009/ring-next-audit.md`。4 bank 增加约 77.34 MiB Host USM 与 device staging；相同预算预计驻留槽从 2792 降至 2760。 | 先确认按 bank 的 host reuse wait，再固定相同 2760-slot H4 前缀比较 ring2/ring4；覆盖四 bank 的 descriptor、wrap、H2D source、reader fence、move、swap 与释放。不得将 14K 无收益解释为已证实 bank 瓶颈。 |
| **4：PLE gather** | 历史 PLE async/lookahead 已在当前路径；旧报告中的约 13 s 是旧时间线上的 PLE host 区间，含准备、等待及调用。H4 后当前 prefill 仍存在大块 expert fill/staging 工作。 | 仅并行化 PLE token gather 是待验证假设；没有当前关键路径证据证明旧 13 s 仍可节省，也没有本轮 gather 改动。 | 先在当前提交上分开计时 gather、ready wait、PLE consume 与 expert supply；之后若证据指向 gather，只改 gather 并核对完整输出/状态。不要拿旧 13 s 当当前收益预算。 |
| **5：router wait / next-layer route** | shared expert 已在实际 router host wait 前提交；adaptive 路由仍需等 router event 才能读 host top-k IDs/weights。旧 async-route / predict 分支受非-adaptive 条件约束。 | 再移动 shared 工作没有新的提前提交空间。预测下一层路由是否有效，取决于预测输入 `x` 是否与下一层 router 的真实输入一致；需先审清本层 MoE、residual/hyper-connection 的依赖。暂不优先实施。 | 先测 router host wait 的可隐藏窗口，并验证 next-layer `x` 依赖；没有这两项证据不改 fence 或 speculative route。与 swap/H2D、staging 或 PLE 实验分开。 |

## 后续交接检查

当前无未提交运行时代码，四轮原始结果与正式构建已归档。后续继续使用同一 prompt、profile、chunk、MTP 和缓存条件。正式 pair 是提交后的身份归档，不因提交身份变化重复整套生产测量。下一轮若转向 swap，须在 2a 与 2b 中只选一个。PLE 补丁/微测脚手架在 `tmp/q4e-ple-parallel-20261009/`，尚未应用、编译或测量。不要把 Strata 在其他 GPU、CPU 或量化格式上的吞吐数字换算为 A770/Haswell/NF4 目标；只借鉴有限预分配 transfer ring、明确 consumer fence、owner publish 边界和分项计时的方法。

## 2026-10-10 后续轮更新（在 `3962519` 上）

诊断与筛选全部完成，结论如下；证据见对应报告，本节替代上文"候选状态"表中相应行的待验证状态。

- **GPU 饱和结论**：稳态 4096-token prefill 块 compute 引擎 97–103%、copy 52–75%（Windows GPU Engine 采样）。14K prefill 是 GPU 计算瓶颈；主机侧候选（PLE 4、ring 3b、router 5、staging 再优化）不能改变 wall。VTune xpu-offload 见 [qwen38-prefill-vtune-2026-10-10.md](qwen38-prefill-vtune-2026-10-10.md)。
- **候选 2a（swap 预算）已测，负结果**：`MAX_SWAPS` 2→8，decode 速率 −10.4%/−13.4%，prefill 无变化；默认维持 2。见 [qwen38-swap-budget-screen-2026-10-10.md](qwen38-swap-budget-screen-2026-10-10.md)。
- **候选 4（PLE）否决**：稳态 `ple_reused=1`，PLE 不在关键路径；补丁不触及 reuse 路径。
- **候选 5（router）暂缓维持**：wait 为 in-order 队列追赶，无新可提前项，预测输入不等价。
- **候选 3b（ring 2→4）否决**：fill 并行被 bank_wait 完全吸收、copy 有余量、GPU 饱和；ring 深度不改变 GPU 工作量。
- **内核实验（8-token 窗口）否决**：稳态块 +22.8~23.1%（`gate_up_8token` spill 3,904 B vs 4token 320/576 B）；窗口曲线在 4-token 取极小，W4/G3 得到实测支持。已作为 opt-in 提交为 `31a8b78`（`AILA_Q4E_8TOKEN_REUSE` 默认关闭）；补丁与证据在 `tmp/q4e-8token-20261010/`。见 [qwen38-token8-window-screen-2026-10-10.md](qwen38-token8-window-screen-2026-10-10.md)。
- **VTune 观察（未动）**：主机 `zeEventHostSynchronize` 60 s、`zeModuleCreate` 39 s（38 模块，`SYCL_CACHE_PERSISTENT=0` 为既有 crash 规避）；DeltaNet prefill 13.8 s / 360 实例，仅 48 workgroup、38 ms/实例，并行宽度是未来候选但未做定向剖析；H2D 332 GB/34.6 s 与计算重叠。
- **后续可行方向（均需新证据启动）**：专家 prefill 数据流结构改造（在不增线程寄存器的前提下提高每 weights 遍历的 token 复用，或新 tile 设计）；DeltaNet prefill 定向剖析与并行宽度；decode 侧 hit-rate/CPU-miss 路径（2a 已证明预算轴无效；若做需先证明减少 swap 或异步 2b 的价值）。当前 HEAD=`31a8b78`（含 opt-in 8-token，默认关闭），工作区源码干净；`build-refactor/AilaWorker.exe` 为 `31a8b78` 构建（`aa4afa81…`，默认行为同 `3962519`），官方 `3962519` pair `18cea21c…/ab04ae7d…` 在 `tmp/q4e-stage-copy-screen-20261009/retained/`。

### 2026-10-10 方向 1 结果：WG-share 采纳（`e808f63`）

- 机制：workgroup 内 16 子组改为 `(16/ShareQuads) 行 × ShareQuads quad` 映射，多子组同时流式读同一权重行（L1/L2 时间共享）；除映射外逐字复制 4token 内核，**不增加累加器**；输出与 4token 路径逐位一致（内核测试 off/2/4/8 全过，含尾部形状）。
- 实测（同二进制单变量）：稳态块 -3.6~3.8%（share≥2 饱和、share=8 因工作集超 L1 失效）；生产 A/B/B/A median：14K 首字 70.115→68.054 s（**-2.94%**）、总耗时 72.440→70.376 s（-2.85%），四轮请求逐位一致，decode/续轮无回归。
- 采纳：`AILA_Q4E_4TOKEN_WGSHARE` 源码默认 off；`start-qwen38.ps1` 默认 4（保留显式覆盖）。报告 [qwen38-wgshare-screen-2026-10-10.md](qwen38-wgshare-screen-2026-10-10.md)；官方 pair `e808f63`（worker `86ca9dd1…`、server `54ebc853…`）在 `tmp/q4e-wgshare-20261010/retained/`。
- 空间未终极：按 4-token 单元的重读仍为 `ceil(n_e/4)`，WG-share 只摊薄了其中一部分（share=4 时等效 16 token/权重读）；"每权重一遍"的终态需要多行协作/SLM/新 tile 设计，未开始。
- 方向 2（DeltaNet prefill）侦察已完成：并行宽度被递推锁死为 48 WG/6144 线程（A770 的 9.4%），逐 token ~18.2 µs、状态流量 ~520 GB/s、延迟受限；首选零风险判别（WG/token 数扫描）与位精确候选（提高每 lane MLP/预取）见下一步。

### 2026-10-10 方向 2/3 结果与最终状态

- **方向 2（DeltaNet）**：D0 扫描（`--mlp-sweep`，测试参数）证明 per-head-token 成本随 WG 数单调下降且无拐点（48→96 WG：380.5→302.2 ns，-20.6%）——延迟受限、并发未饱和；但位精确的 MLP/预取候选（K1，`AILA_Q4E_GDN_MLP_PREFETCH`，`6dc4c3d` opt-in）**实测回退 1.27–1.95×**（生产几何最差），机理为手工分级把内核推向发射受限（编译器本已调度展开循环）。K1 作为负结果保留为 opt-in；报告 [qwen38-gdn-mlp-screen-2026-10-10.md](qwen38-gdn-mlp-screen-2026-10-10.md)。剩余选项（均未实施）：chunkwise（改数值，需资格验证）、dv 切分（需跨 WG 归约/epilogue）、多 session 网格维（逐位安全、单会话不可测）。
- **方向 3（decode/swap）——已采纳"默认关闭"（`d7b0649`）**：先用长窗口（256 token）证明同步 swap 净负（decode 62.0 vs 67.2 ms；成本 2.7 ms + 连带 ~2.5 ms，收益仅 ~1 ms 命中改善），再以生产 A/B/B/A 确认：14K decode rate **13.732→14.835（+8.03%）**、续轮 +7.42%，请求输出逐位一致；默认配置复跑 15.266 tok/s。发现并修复 env 语义缺陷：`read_int` 将 ≤0 回退默认值，使 `FREQ=0` 无法禁用（此前 swap-frequency 屏幕两臂实为同一配置、已标注无效）；现 FREQ/MAX_SWAPS 改用 `read_int_raw`，FREQ 默认 1→0，`FREQ=1` 可重新启用。报告 [qwen38-swap-longwindow-2026-10-10.md](qwen38-swap-longwindow-2026-10-10.md)、[qwen38-swap-frequency-screen-2026-10-10.md](qwen38-swap-frequency-screen-2026-10-10.md)（修正说明）。风险：长生成命中率自然衰减（256 token 处 -4pp 差距），≥1024 token 证据出现时重评。
- **当前状态**：HEAD=`d7b0649`（wgshare `e808f63` + K1 opt-in `6dc4c3d` + swap 默认关闭 `d7b0649`）；官方 pair `5426f055…/92fb22d1…`（戳记 `aila-0.2.0-abi1-d7b0649…`）在 `tmp/q4e-swapoff-20261010/retained/`。工作区源码干净（docs/plans 多份本轮报告未跟踪，按惯例保留）。
- **2026-10-10 追加：B1 INT8 KV 原型（`1f6c753`，opt-in 默认关闭）**——修复两处 FP8 缺陷（128 快路径半维、JM 量化守卫）＋ s8+per-token scale 的 INT8 KV（K 在 mRoPE 后量化）；三臂微测 int8 NRMSE ≈ FP8 的 1/5（0.0051 vs 0.0253），但读路径逐元素反量化使 14k attention 1.84 ms 慢于 bf16 1.57 ms → 不安排生产 LIVE，下一步为 DPAS 无解量读内核。报告 [qwen38-int8-kv-prototype-2026-10-10.md](qwen38-int8-kv-prototype-2026-10-10.md)。当前官方 pair `395dd97b…/4bf51966…`（戳记 `aila-0.2.0-abi1-1f6c753…`）在 `tmp/q4e-int8kv-20261010/retained/`；HEAD=`1f6c753`。
- **2026-10-11 追加：B2 W4A8 adaptive decode（`5ef4e1a`，opt-in 默认关闭）**——把 int8 激活内核从退役 resident_epoch 移植到 adaptive decode（`moe_decode_gate_up_swiglu_grouped_q8` 等，x/h per-64 scale）。内核级 T=1 慢 15.5%；端到端生产 A/B/B/A +1.61%（14K rate）/ +0.28%（续轮），输出逐位一致，容差 PASS；低于 +5% 采纳线 → opt-in。报告 [qwen38-w4a8-screen-2026-10-10.md](qwen38-w4a8-screen-2026-10-10.md)。当前官方 pair `0a2fb6a2…/273ba94c…`（戳记 `aila-0.2.0-abi1-5ef4e1a…`）在 `tmp/q4e-w4a8-20261010/retained/`；HEAD=`5ef4e1a`。
- **2026-10-11 追加：B4-① adaptive 多行 CPU verifier（`176aede`，opt-in 默认关闭）**——`AILA_Q4E_ARENA_VERIFY_ROWS`；rows 2..4 的 CPU 冷解从 host arena 解析，输出与逐行 T1 CPU 参考**逐位一致**（新测试 `AilaQwen4ExpArenaVerifyTests`），CPU/ABI 19/19；同时修复 16 项 cold 数组在多行 40 任务批下的越界。MTP 的"verifier 可达"前置已满足；下一步 B4-② GPU 多行核 T1 对齐。报告 [qwen38-arena-verify-2026-10-11.md](qwen38-arena-verify-2026-10-11.md)。当前官方 pair `204fa935…/0fc78b58…`（戳记 `aila-0.2.0-abi1-176aede…`）在 `tmp/q4e-arena-verify-20261011/retained/`；HEAD=`176aede`。
- **2026-10-11 追加：B4-② T1 对齐里程碑（`d701cf6`，opt-in 默认关闭）**——根因 = 调度（count≥2 走 prefill/oneDNN GEMM vs count=1 单行 decode 核），`AILA_Q4E_VERIFY_SMALL_ROWS` 修复后 **layer-0 全 4 行逐位一致**、残差推至 layer 1；`--mtp-align` 驱动与逐阶段探针入库；同时补齐 B2 提交遗漏的 `Qwen4ExpW4A8Tests.hpp`（树完整性）。第二段（layer≥1 + ≥1100 prompt）进行中。报告 [qwen38-mtp-align-2026-10-11.md](qwen38-mtp-align-2026-10-11.md)。当前官方 pair `59186185…/6e301d97…`（戳记 `aila-0.2.0-abi1-d701cf6…`）在 `tmp/q4e-mtp-alignment-20261011/retained/`；HEAD=`d701cf6`。
- **2026-10-11 追加：B4-② 全对齐（`e4d4bc5`）+ B4-③ 验收未过（`062b940`）**——`AILA_Q4E_VERIFY_SMALL_ROWS` 扩展到全部 count 相关阶段后，width 2/4 在 512/1182 prompt 上**全部 48 层逐位等于 4×T1**（含 attention-pair）；但 MTP 活线验收（context 2048、prompt 1500、128 token）从字符 66 起与 T1 分歧、速率 -2~-4%（windows=149/accepted=111 确定性复现）→ **MTP 保持默认关闭**；下一步 B4-④ 定位活线分歧（先证/否证 VerifyRows(1)==普通 decode）。报告 [qwen38-mtp-align-2026-10-11.md](qwen38-mtp-align-2026-10-11.md)、[qwen38-mtp-acceptance-2026-10-11.md](qwen38-mtp-acceptance-2026-10-11.md)。当前 pair `4e5efc6c…/7f8f05b2…`（`e4d4bc5`）在 `tmp/q4e-mtp-alignment-20261011/retained2/`；HEAD=`062b940`。
- **2026-10-11 追加：B4-④ 活线分歧修复（`362bc98`）**——候选 1/3 否证（新增 `AILA_Q4E_MTP_ALIGN_T1=decode` 证明 VerifyRows(4)==4×普通 decode 逐位；draft 无共享状态污染）；首个分歧 = **attention-pair 核**（`ATTN_PAIR=1` → 119 tok "large number"；=0 → 首请求与 T1 逐字节同 sha f72a0da0fb2f）；修复 = `VERIFY_SMALL_ROWS` 下 pair 核条件加 `&& !verify_rows_alignment_`。残余：续轮仍分歧（111 vs 128 tok，prompt/cached 相同）→ B4-⑤ 定位会话缓存态。报告同上。当前 pair `a03ae9ec…/20c721d5…`（`362bc98`）在 `tmp/q4e-mtp-alignment-20261011/retained3/`；HEAD=`362bc98`。另：Fermat 的测试编排已按用户反馈从固定 ~420s 等待改为 1–2s 轮询+前台运行。
- **2026-10-12 追加：B4-⑤ 状态对比（`b14cd64`）+ 构建完整性修复（`90b7931`）**——4 行 verify 窗口与 4×普通 decode 的持久状态**逐字节相同（differing_tensors=0）**；分歧触发收敛为 **stop-token 结束路径**（`--generate 8` length 结束：首请求+续轮全同；`--generate 128` stop 结束：首请求同、续轮 128 vs 111 tok 分歧，两臂 emitted=126/ctx=1550 相同）。记账探针一度用 `read_flag` 触发轻量目标链接主库符号失败，已改 `read_int_raw` 修复。B4-⑥（stop 路径二分：adaptive 驻留/路由计数 + draft mtpState 快照）进行中。当前 pair `ff367797…/61be4ffb…`（`90b7931`）在 `tmp/q4e-mtp-alignment-20261011/retained4/`；HEAD=`90b7931`。
