# Qwen3.8-Flash-Next H8：原始 NF4 CPU 供给与主机缓存预算

同最终 EXE 三轮对照，完整五请求序列中位数 **117.88→51.49 秒，减少56.3%**；相对仅关闭pool/短分片的控制，再减少39.0%。首次/返回长文请求 **47.40→16.74 / 27.43→15.62 秒**，decode **9.3→11.1 / 8.6→13.5 tok/s**。20 decode目标仍未完成。

本轮承接 H7 `291a43e`，实现并实测更激进的异构供给：CPU 直接读取稳定 mmap NF4 来源，增加 NF4→int8 LUT 与 activation Q8 点积，比较 32/16/8/0 GiB Host pool、冷热互补回收、短 forward 需求分片、CPU/PCIe 分流和批量页预取/触页。没有增加 GPU kernel 或改变模型文件。所有新算术/供给开关默认关闭，H7 launcher 默认保持；H8 通过 `-Heterogeneous` 显式开启。

最终同二进制三轮对照与启动命令见下文；完整数据、阶段身份、失败样本和源码快照索引见[结果 JSON](qwen3.8-flash-next-heterogeneous-h8-results-2026-10-03.json)。单进程筛选与正式确认分开，未把不同 EXE 的阶段直接合并比较。

## 1. 从 Strata 借用的机制与本轮边界

- [expert_source.cpp:444](../../tmp/Strata/src/core/expert_source.cpp) 的 `mapped_blob` 和 :1274 的 arena blob 让 CPU 直接消费稳定原始来源。Aila 的 `MappedExpertHostSource` 返回自己的 safetensors NF4 packed、scale、nested spans，保留 block64、dtype、geometry 与边界检查；没有移植 GGUF 布局或整模型 arena。
- Strata `expert_source.cpp:59,458–627` 构造 GPU cache 之外的完整 complement arena；`expert_cache.cpp:242` 的静态 cache 满后不驱逐。Aila 有 prompt 交接、借槽和 Ready 接管，本轮只在已有预算内做有界双 LRU 回收偏好，检查当前 frozen mask、generation、CPU readers、DMA fences；不能声称已经消除全部 GPU/host 重复。
- [ngram.cpp:237,329](../../tmp/Strata/src/kernels/ngram.cpp) 为 PLE table 一次提示多段 mmap 页。本轮把这种 advisory batch 方法用于已获准的 mapped CPU experts，正常读取仍决定正确性。Strata 的 expert `begin_layer` hook 本身仍是默认 no-op，不能把该实验说成移植了已有 expert 预取。touch 是 Aila 新增对照。
- Strata `expert_source.cpp:803–885` 将 resident 优先留在 GPU，按 `floor(nmiss*pcie_num/256)` 分配冷项。沿用 Aila 已有 policy3 做更低 PCIe 份额的对照，没有移植 CUDA mapped-host GPU 访问。
- Strata 读取 16 个相邻 experts 后 scatter 的连续加载方法尚未移植。本轮 short-parts 只降低 Aila 真实 demand 的任务/读分片数；Aila reader 已使用 `NORMAL|OVERLAPPED`，无需修改不存在的 `RANDOM_ACCESS`。

Strata 本地 revision `a79080535d1b2a71a3419a0d97d8e7dca194b0f1`。以上为源码机制参考，未把其 RTX/Ryzen/DDR5/Q2/MTP 速度当作本机复现。

## 2. 实际实现

CPU NF4-Q8 保留原 packed NF4 和 nested scale；用 signed int8 LUT、每32 activation Q8、AVX2 shuffle/maddubs/maddwd 完成点积，保持中间 h 的 BF16 边界。成对累加最大32258，未触发 int16 饱和；gate barrier 后才量化 h，固定 nearest-even，不依赖 MXCSR rounding。局部10专家/12线程微基准中位数0.94268→0.47998 ms，约1.96×；这是局部算术收益。

Mapped cold 只作用于 frozen resident 的冷项。Ready host lease 优先，GPU Ready/Copying 保持 GPU；mapped metadata 不合格退回原 GPU 供给。来源由 immutable ModelWeights 持有，Join/Abort/cleanup 先 drain worker 再释放 leases/映射引用；CPU observers 使用每任务独立 source slot。该路径减少 CPU 权重 slot fill 和对应 weight H2D，仍有 OS 缺页、CPU 计算、activation/result copy。

`FILL_SHORT_PARTS` 仅覆盖每次 forward rows1..64 的真实 demand fill，长 forward、profile/prewarm 和 prefetch 使用原 global parts。它以 forward 行数判断，长 prompt 的小尾片也会进入短路径。未新增热路径 USM 分配。

页预取对当前获准 mapped 来源做固定容量去重，最多40任务×8 spans，失败只记录；touch 只读 span 内首字节和后续 OS 页边界，不越界、不 pin 页。两个开关默认关闭；CPU policy2 的服务 EWMA 计入 compute+prefetch+touch，排除 activation/scheduler wait。

修复两个原有配置零值问题：`HOT_RAM_MB=0` 真正关闭 persistent Host pool，`CPU_MAX_EXPERTS=0` 真正禁用 CPU admission。原正值/默认行为保留。测试覆盖0、负 host budget、正预算、默认预算、完整 packed/scale payload、mapped 视图与显式0 credits。

## 3. 单进程筛选与负结果

各筛选进程输出24 tokens，顺序 story41 → archive1833 → code47，同阶段同 EXE。以下单位为各请求秒。

| 阶段/组 | story | archive | code | 解释 |
|---|---:|---:|---:|---|
| S1 H7基线 | 14.917 | 42.600 | 8.379 | 本阶段参考 |
| S1 Q8 | 15.134 | 44.331 | 8.389 | 算术快未自动变成净收益 |
| S1 Q8+complement | 15.373 | 44.555 | 8.116 | 有界回收扫描未取得明确净收益 |
| S1 全局fill1 | 11.427 | 45.344 | 7.453 | 短请求收益、长文回退 |
| S3 H7基线 | 15.543 | 45.049 | 8.834 | 本阶段参考 |
| S3 mapped16+short1 | 5.780 | 29.158 | 7.922 | 单进程不能证明稳定性 |
| S4 Q8+16GiB+short1 | 7.894 | 30.580 | 7.664 | 无 mmap CPU 的控制 |
| S4 mapped16+short1 | 8.168 | 29.785 | 18.348 | code 大幅回退，全样本保留 |
| S4 mapped16+预取 | 6.862 | 40.092 | 7.828 | compute下降、整长文回退 |
| S5 mapped16+touch | 12.481 | 31.095 | 11.137 | 等待移入触页，code decode2.8 |
| S5 mapped8+预取/touch | 7.736 | 26.744 | 7.051 | 不作为默认 |
| S5 8GiB+short1，无Q8/mmap | 5.730 | 25.129 | 6.445 | 缩小 pool 本身已有收益 |
| S5 mapped8+PCIe64/256 | 6.788 | 23.628 | 6.003 | policy3保留GPU冷供给 |
| S6 真mapped0+short1 | 6.131 | 16.968 | 4.458 | 长文TTFT14.617，decode9.4 |
| S6 mapped8+PCIe64/256 | 5.151 | 23.339 | 6.065 | 同最终EXE对照 |

S2 mapped32 在第三个 code 请求触发原8192 MiB空闲RAM保护：峰值工作集44,870,754,304 B、最低空闲8011.914 MiB。保留完整首请求、部分第三请求和 memory/meta/log，不作为成功性能样本；未降低保护阈值。它移除 weight staging，却扩大 mapped working set，与32GiB pool争抢物理页，CPU每token compute曾达200–450 ms。

S5 的 `HostRamMB=0` 曾实际回退16384 MiB。该样本是配置无效数据，全部排除；修复后 S6 的 ExecutionConfig 明确 host_ram_mb=0，测试和实模验证均确认关闭。零 pool 的 host_pool_misses=0不表示没有文件读取，direct staging 读不计入该计数。

Complement 的扫描/eviction 数不能等同于持续释放容量；prefetch bytes 是 advisory 请求，不是实际 I/O，touch pages 是范围访问次数，不是唯一缺页数。新的 warm 时间计入 cold queue，并可能与 GPU 重叠，不能与 compute/join 相加为独占总成本。

## 4. 最终三轮完整序列

表格单位：请求秒 / TTFT秒 / decode tok/s。

| 请求位置 | H7 32GiB | 仅关闭pool+short1 | Q8/mmap/零pool+short1 |
|---|---:|---:|---:|
| 首次story41 | 19.36 / 10.29 / 5.1 | 13.84 / 3.87 / 4.6 | 6.94 / 2.17 / 9.7 |
| 首次archive1833 | 47.40 / 42.58 / 9.3 | 29.00 / 22.28 / 6.8 | 16.74 / 12.59 / 11.1 |
| code47 | 14.12 / 4.35 / 4.7 | 11.47 / 2.20 / 5.0 | 6.71 / 1.98 / 9.8 |
| 返回archive1833 | 27.43 / 22.13 / 8.6 | 19.24 / 12.41 / 6.7 | 15.62 / 12.20 / 13.5 |
| 返回story41 | 9.56 / 3.38 / 7.5 | 10.61 / 1.88 / 5.3 | 5.52 / 1.83 / 12.5 |

| 组 | 三个进程五请求总耗时（秒） | 中位数（秒） | CV |
|---|---|---:|---:|
| baseline | 111.36, 117.88, 120.39 | 117.88 | 0.033 |
| baseline-ram0-short1 | 84.46, 71.26, 84.44 | 84.44 | 0.078 |
| q8-mapped-ram0-short1 | 51.46, 51.49, 51.64 | 51.49 | 0.001 |

每进程先求五请求总和再取中位数。相对同最终EXE的H7配置减少 **56.3%**；相对仅关闭pool/short1控制减少 **39.0%**。这是本轮同环境比较，未将历史H7的118.71秒与本轮样本混为同二进制对照。

独立进程每组3次，round2反序，全部输出48tokens：story41→archive1833→code47→archive1833→story41；每请求清会话，模型与热表持续共享。9进程45请求均保留，没有watchdog/内存保护停止。硬件A77016GiB/驱动32.0.101.9033、Xeon E5-2696v3/DDR3、oneAPI2026.1/oneDNN3.11.2，profile、3567GPU槽、borrow/Ready-adopt、chunk2048和H7其余配置固定。

最终 CLI EXE SHA256：`2F9171783BAC9633F39837B0C9B89CAC8DE7D16A59984B30C2E1F8B4513A7EF1`。六个探索阶段分别保存 EXE 与9项编译源码快照哈希，最终报告也分别列出编译快照和现有源文件身份。

请求CV>0.15的样本组：baseline-ram0-short1/首次archive1833: CV=0.176。原始样本全部保留，三次进程仍不足以建立生产场景普适分发。

主机内存：baseline peak WS=40.10 GB / min free=11764 MiB；baseline-ram0-short1 peak WS=5.66 GB / min free=44561 MiB；q8-mapped-ram0-short1 peak WS=30.39 GB / min free=23537 MiB。GPU槽数和设备ledger上限保持同配置；没有把更小host pool当成免费常驻整模型。


## 5. 数值、状态、所有权与回归

最终零 pool/Q8/mmap 配置短/长48步 teacher forcing：GPU mode1均48/48逐位一致；CPU mode2平均 KL **0.002022812 / 0.002704957**，最大 logits NRMSE **0.06699333 / 0.05957793**，mean NLL Δ **+0.00175637 / −0.00250156**，均无 NLL预警。原 mean KL≤0.01、max NRMSE≤0.15 门槛未放宽。CPU任务14532/8725，prefill direct2226/5717，checkpoint/replay/cancellation通过，live USM delta0。

其他阶段的短/长 canaries 也保存，包括 complement、mapped16、source-prefetch、mapped8+prefetch/touch；它们使用各阶段独立 tolerance test EXE，不能伪装成最终 CLI EXE 的性能样本。

NF4-Q8 单元覆盖4token/40task、nested与flat尺度、两步 scale rounding、1/4/12线程逐位批/单项一致和 zero overwrite，NRMSE0.0147053<0.03。Readiness 覆盖 mixed host/mapped、observer、CPU/GPU互斥、mapped几何/边界、无权重staging、prefetch去重/advisory failure、touch边界、credits0、取消、generation、move与reader fences。

48步 teacher canaries 不是广泛模型质量认证。Q8改变LUT、x和h量化，可能改变 logits、路由和自由生成文本；尚未验证 NaN/Inf 或极端幅度的传播语义。模型文件/默认路径保持原值。

最终构建42目标完成；CPU NF4-Q8专项与Readiness通过，6项CTest（Readiness/Reader/CpuExpert/Generation/Module/ModelSpec）、15项Python回归、PowerShell AST与Python编译检查通过。身份汇总保护验收覆盖配置白名单、EXE/模型/输入/profile/stack/GPU/槽/环境、token数量、重复进程、watchdog、伪零预算和源码快照不匹配，全部通过。

普通三历史8token trace 经cold_queue2 replay通过，共30 forwards，包含1个startup warmup；5378次Ready接管、4次prompt交接，在线swaps0，所以epoch retire/publish pairs为0。验证direct leases、reader fences与CPU/GPU冷项互斥；诊断trace不参与wall性能比较。launcher实际8token启动及污染环境DryRun/恢复/显式覆盖检查通过，具体日志身份见JSON。

## 6. 运行与下一瓶颈

```powershell
pwsh -NoProfile -File perf/qwen4exp/q4e_supply_chat.ps1 -Heterogeneous -PromptFile perf/qwen4exp/prompts/archive-natural.json
```

H8 开启 HostRam0、CpuNf4Q8=1、CpuMmapCold=1、FillShortParts=1；policy1、12线程/10credits、global fill8/threads8、源预取/touch/complement关闭，其余沿用H7。显式参数可覆盖单个实验项。省略 `-Heterogeneous` 继续运行H7的32GiB配置，新flags均0。launcher 清理运行环境、导入 oneAPI2026.1，结束恢复调用者环境。

launcher实跑采用messages-json，story本次为30 prompt tokens；性能probe为交互CLI输入，story为41 tokens。实跑证明启动参数/运行链路可用，不把两种模板路径说成同输入测速。

请求时间包括prompt/生成/交接/epoch维护，模型与profile初装在请求计时外。每请求清会话，模型与全局热表继续共享。进程间没有清空系统文件缓存，反序减少但不能消除顺序偏差；首次请求不表示机器冷启动。普通trace提供主机顺序/所有权证据，有trace开销，不是纯GPU执行或重叠测量。没有把逻辑payload当PCIe实测带宽、USM ledger当DXGI驻留。

最终候选首次/返回长文有效输入/TTFT为 **145.6 / 150.2 tok/s**，decode **11.1 / 13.5**；该输入/TTFT包含第一token与交接，并非纯prefill kernel速度。20 decode目标仍未完成。

零pool候选decode权重需求H2D为0。普通trace的三个正式8token请求，CPU compute为67.28/56.34/59.25 ms/token、join为68.52/57.46/60.48；router wait为10.84/12.67/10.97，resident GPU submit为4.54/4.31/4.39。expert_acquire_submit为75.17/63.60/66.62，包含或重叠上述CPU区间，不能相加为总成本。startup warmup的单forward另保留，不混入正式请求均值。

第三个正式性能进程五请求的CPU compute均值为49.52/35.04/47.92/19.38/25.73 ms/token，join为50.84/36.27/49.32/20.56/26.97；queue仅0.56–0.68、activation wait0.38–0.40。两次长文CPU任务数180.81→180.10/token几乎相同，compute却35.04→19.38：来源缓存/访问状态可能影响明显，现有证据不能归因为具体OS缺页。

下一步优先区分CPU实际算术、内存访问和软缺页成本，按实测服务成本调整cold admission或GPU热集，再检查逐层router回读/等待。继续加pool、显式触页或扩大补齐扫描已缺乏收益证据；完整Strata complement要求约49.6GiB冷专家容量，与本机OS/headroom一起不适宜直接整arena照搬。GPU提交图/R2b或更低字节权重格式需要独立实现和同条件验收，不能将这些未完成方向算成本轮成果。
