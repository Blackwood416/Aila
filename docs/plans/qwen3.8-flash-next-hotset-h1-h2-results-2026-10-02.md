# Hotset H1/H2：动态驻留直接执行首轮

日期：2026-10-02。A770 / 驱动 32.0.101.9033 / oneAPI 2026.1、oneDNN 3.11.2。按 [新优先级规划](qwen3.8-flash-next-hotset-priority-plan-2026-10-02.md) 暂停 R2b 的 per32 量化及生产接入，先实现 H1/H2。此前累计改动与新规划已保存为提交 **5ea0171**；原 Aila / tolerance EXE 另存为 build-refactor/AilaHotsetBaseline.exe、AilaQwen4ExpHotsetBaselineTests.exe。

本轮沿用 NF4、32 GiB host pool 和原有 GPU expert pool 容量、LRU/admission。CPU 冷专家计算已重新进入下一工作包 C1；本轮建立它所需的 GPU-only 直接执行对照，没有调用旧 CPU placement 开关。

**[MEASURED] H1/H2正确性通过，但当前实现没有稳定的端到端净收益。** 短输入暖decode中位数为原路径14.0、H1 12.6、H2 13.1 tok/s；自然长输入为8.95、8.4、8.75 tok/s。候选保持默认关闭，下一步进入C1的同热集冷计算对照，不把本轮结果泛化为热集或异构架构无效。

## 1. 实现边界

**H1，AILA_Q4E_RESIDENT_DECODE=1：** router 主机等待后，BindReady 按完整路由取得 Ready 槽的 generation/reader 租约，直接提交热贡献；AcquireCold 只接收冷项。全热层不创建冷事务。热/冷贡献各自写入原 rank 的 FP32 down 输出，按原 rank 顺序乘一次 router weight，再合并共享专家。

租约独立于 acquire 的批内 reservation，淘汰及 pin 新槽都跳过活动租约。正常结束把保护转交给有序 compute queue 的 reader fence；中途退出则先 drain 已提交工作。BindReady 保持原完整路由的首次出现 MRU 顺序，包括还在 Copying 的命中；后续冷事务不再次 touch 这些命中。已完成 DMA 的 Ready 状态被缓存，避免 bypass 后反复查询同一个完成事件。

**H2，AILA_Q4E_RESIDENT_DECODE=2：** router 写 device IDs/weights，GPU classifier 查驻留表，生成每个原 rank 的 slot 或 −1，并输出主机 mailbox。热 gate/up 和 down 在主机等待 mailbox 前入队；主机收到结果后只组装冷 descriptor。内核在 SLM barrier 后以整 subgroup 一致的条件屏蔽未命中 rank，保留原 NF4 uint32 / BF16 vec8 / SLM LUT 算术。

H2 首版采用**每个单行 forward 的 descriptor 快照**，每层发布仍符合 generation 且 Ready 的索引。发布前临时租用该层候选，GPU 分类后将选中项转交 route lease，释放未使用项。中途被其他层淘汰重用的槽不再符合快照 generation，会进入冷路径；不会让旧 descriptor 读取新专家 payload。

这仍是动态驻留方案，不是固定热名单或多 token 自适应热集。每 token 有一次 descriptor 上传和 48 次索引表上传，首版故意把这些成本纳入结果。短/长测试分别约 454,848 / 440,736 B metadata/forward；额外 device USM 约 0.53 MiB，不挤占既有专家槽数。没有 device spin wait、全局 OOO queue 或整 token graph。

默认值仍为 **0（原路径）**，A770 preset 未开启候选。目前仅 count=1 走直接路径；多行 prefill/verify 保持原实现。mode1/2 与 CPU dispatch 的组合暂时明确拒绝，防止把旧异构执行器误当成 C1。

## 2. 计数与 trace

直接绑定计入原 hit_count，Copying 项在 cold acquire 内完成原命中计数，避免 bypass 导致 hit-rate / prediction gating 假性下降。新增直接路由、驻留、直接执行、冷项、acquire_calls、metadata_bytes 计数；需求权重字节仍沿用原 demand 口径。

JSONL 增加 direct_bind / direct_release 和 cold transaction 的 touch_hits 标记。回放器同时验证 route = direct + cold、LRU、活动租约保护、victim 和 demand bytes；反事实缓存模拟也使用完整路由，不把冷子集当成全部工作。旧二进制 route trace 保留完整需求，未改成只记录冷项。

H2仍有每层 mailbox rendezvous；没有宣称消除了全部 router host wait。resident_gpu_submit 的主机时间戳能证明入队顺序，不能直接当作 GPU dispatch gap 或 GPU/CPU 实际重叠。H2 的 router_wait 主机区间还包含随后的租约绑定，不能全部解释成 GPU 等待。

## 3. 正确性

小 fixture 复用真实 cache、NF4 kernels 和有序 rank merge，覆盖全热、混合、全冷、Copying、跨 token 重复专家、受保护槽被放到 LRU 尾部、容量用尽、释放后复用、旧 generation、GPU mailbox、失效 descriptor 快照。热/冷输出与原参考逐位一致。新增中途退出检查在实际 weight reader 入队后取消，确认租约析构 drain 后槽位才可复用。

真实模型用同一引擎依次运行原路径、H1、H2；41-token短输入和1833-token archive-natural 各做48步 teacher forcing：

| 输入 | H1 对原路径 | H2 对原路径 | 状态检查 |
|---|---|---|---|
| 41 tokens | 48/48 logits 逐位一致 | 48/48 logits 逐位一致 | checkpoint 后缀重放两次均一致 |
| 1833 tokens | 48/48 logits 逐位一致 | 48/48 logits 逐位一致 | checkpoint 后缀重放两次均一致 |

NLL差和 logits NRMSE 均为0，KL仅有约1e−16的计算舍入。decode 的 live device/host USM 数量未增加。这是这些输入和状态场景的验收，不替代多会话服务、完整 cancellation HTTP 或 C1 CPU 算术验收。

诊断时长带逐步 logits 回读及额外恢复回放，明确 performance_comparable=0，没有用它计算 TPS 提升。

## 4. CLI 性能

每个输入按 **0→1→2、2→1→0** 顺序运行独立进程；每进程3次相同请求，每次生成48 tokens。首请求单列，后两次作为暖样本，每模式共4个暖请求。使用同一 Aila EXE、同一 prompt/template、32 GiB host pool、相同专家槽容量及 chunk=512。WorkSummary 只记录计数，不同步 GPU；候选多一行 resident 统计，其开销也保留。

全部模式的生成文本 SHA256 一致。这里的暖态指进程内重复请求，不声称磁盘或整个模型权重全驻留。每个 prompt 只有有限样本，不能外推为通用 dispatch 门槛。

下表均为4个暖请求的中位数。TTFT和完整请求单位为秒，decode为CLI报告的实际生成tok/s。

| 输入 | 模式 | TTFT | decode | 完整请求 | 完整请求相对原路径 |
|---|---|---:|---:|---:|---:|
| 41 | 0 原路径 | 1.546 | 14.00 | 4.828 | — |
| 41 | 1 H1 | 1.549 | 12.60 | 5.206 | +7.83% |
| 41 | 2 H2 | 1.548 | 13.10 | 5.049 | +4.58% |
| 1833 | 0 原路径 | 24.562 | 8.95 | 29.713 | — |
| 1833 | 1 H1 | 24.761 | 8.40 | 30.235 | +1.76% |
| 1833 | 2 H2 | 24.326 | 8.75 | 29.640 | −0.25% |

长输入H2的完整请求差异不稳定：正向一轮比对应原路径慢约4.2%，反向一轮快约4.8%；同轮暖prefill的TTFT范围约23.2–25.6秒。因此合并中位数的−0.25%不能写成稳定提速。长输入decode需求字节各模式约14.958–14.959 GB/请求，变化不足以解释大的传输收益。

首次请求、每个暖样本、prefill/decode需求字节、metadata字节、GPU USM、进程工作集及可用RAM均保留在 [结果 JSON](qwen3.8-flash-next-hotset-h1-h2-results-2026-10-02.json)。首次请求TTFT不包含已经结束的模型加载/warmup；加载总时长未单独打点，warmup时间保留于日志。

短输入各模式暖 decode demand 均为8.8399872 GB/请求。H1/H2直接计算19,552/23,040个贡献，约84.86%；acquire从2304降到1513次。**调用减少没有自动转化为净吞吐收益。**

H2相对H1收回了一部分完整请求时延，但仍增加classifier、分开的hot/cold kernels、rank merge和索引表发布。额外提交、表管理及预取提交时机是待进一步归因的成本项，不能仅用时长相减断言一个固定UR/WDDM开销。这个负结果不否定固定热集与CPU冷计算分工，后两者尚未测量。

## 5. 运行边界与证据

所有模型过程使用240秒 watchdog、8 GiB系统可用RAM下限，单GPU压力进程。没有提高预算以掩盖候选成本。所有性能对照都在9033上，本轮没有旧驱动A/B，候选回退不能归因于驱动。

12个CLI性能进程、2个真实模型验收进程和2个trace进程均exit=0，无watchdog/内存保护触发；合计峰值进程工作集37.51 GiB，最低系统可用RAM11.57 GiB。工作集不是GPU显存测量，也不是32 GiB host pool本身的大小；本轮没有新增RAM占用优化。

6项CTest通过：KernelTests、ModuleTests、GenerationTests、ToleranceTests、DpasSmoke、ExpertReadinessTests；取消drain标记保留于原始CTest详细日志。Python缓存回放测试6/6通过。

H1/H2各自的真实trace均通过完整LRU/victim/路由覆盖回放。取第二请求的16个单行forward，共768层，两条路径均直接执行5911项、冷执行1769项；161层全热、598层混合、9层全冷，实际冷事务607次。161个全热层均没有acquire。

H2在全部768层中都于主机开始等待router之前完成热kernel提交，主机时间提前量中位数102.5 µs。在混合层中，热提交到冷acquire开始的主机间隔中位数为705 µs，H1为5.6 µs。前者包含主机等待router的时间，只验证提交顺序，不用于计算GPU重叠收益；GPU dispatch gap/实际重叠仍未单独测得。

所有性能进程使用EXE SHA256：4DE373433C91476D58D433F9624B1D5FC54F424D87DA2EF3B14E574C5BC032E1；真实模型验收EXE SHA256：927ECFF72AE6D0CE6BE667619BEC156285B5BE1C9C45531E1151480C217D12E2。详细日志、内存样本、源码与trace哈希均收录于结果JSON。

主要源码：[cache与租约](../../src/models/Qwen4ExpExpertCache.hpp)、[后端接入](../../src/models/Qwen4ExpBnb4Backend.cpp)、[GPU分类/热kernel](../../src/ops/MoeExpertOps.cpp)、[小fixture](../../tests/models/Qwen4ExpExpertReadinessTests.cpp)、[模型验收](../../tests/models/Qwen4ExpResidentAcceptance.hpp)。复现入口为 [execution probe](../../perf/qwen4exp/q4e_execution_probe.ps1)，汇总及trace检查为 [hotset summary](../../perf/qwen4exp/q4e_hotset_summary.py)、[hotset trace](../../perf/qwen4exp/q4e_hotset_trace.py)。源码、EXE、输入和日志身份记录于JSON。

## 6. 后续优先级

H1/H2提供了可验证的直接执行/冷供给分界，尚未交付端到端净收益，也没有达到decode 20 tok/s。R2b后续实施继续暂停。

按新规划进入 **C1**：在同一初始热集、同一容量和冻结admission条件下，比较cold-on-GPU与有界host-Ready cold-on-CPU；GPU命中继续直接执行，复用prepared AVX2 worker和host leases。两边GPU冷槽均作为临时staging处理，避免GPU冷计算偷偷建立下一步命中而CPU计算没有对应admission。原动态缓存路径仍保留为第三个整请求参考。

固定热集、跨域profile identity/覆盖、epoch自适应、prefill热集交接和CPU/GPU admission解耦仍按H3/H4/C2推进。本轮没有从Strata导入专家名单，也没有重新开启旧static pinning来冒充新的执行架构。
