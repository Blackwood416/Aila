# Qwen3.8-Flash-Next：P0 观测、P1 筛选与 P2 首轮对照

后续隔离结果及 512/1024 验证见 [P2 供给与 P3 固定段结果](qwen3.8-flash-next-p2-supply-p3-front-results-2026-09-29.md)。并发 copy 已在无数据依赖的对照中复现 gate/up 增长；旧文中的等待时间戳假设已有进一步证据。

日期：2026-09-29。运行时代码基线 `0293b3b`；A770 16 GB、驱动 32.0.101.8860、oneAPI 2026.1。模型为 `Qwen3.8-Flash-Next-BNB-NF4-v2`。所有推理计算仍在 GPU，host pool 仍为约 16 GiB。

本轮实现了可核对的缓存 trace 与 GPU 时间线分析，并按顺序筛选一个缓存候选、测试读取粒度和既有 B6 小批路径。**没有推广新的默认策略，尚未达到 Prefill 100 / decode 20 tok/s。** dense XMX 的广泛质量/形状默认准入仍待完成；本轮所有模型对照都显式开启 XMX，不把它当作已有默认。

## 1. 已落地的观测能力

- `AILA_Q4E_CACHE_TRACE=<path.jsonl>`：每个 forward 的 position、token 数、是否产生 logits；每次 acquire 的原始专家顺序、GPU 槽和记录状态、host 状态、实际命中和搬运字节、demand file fill、prefetch 等待；实际淘汰槽及最多 8 个 cost-aware 候选；显式 demote 顺序；host router-wait 和 expert-acquire/submit 区间。prefetch 的 issued/useful/cancelled/wasted 字节在 forward 边界记录累计值，可以取差。
- trace 由 cache 实例持有，逐行写文件，在 forward 结束刷盘，不积累整请求的内存缓冲，也不增加 USM 分配。记录状态为 Copying 不等于当时 DMA 必然仍在执行；实际完成情况以事件时间线为准。
- `AILA_Q4E_TRACE_MARKERS=1` 配合 cache trace 才提交两个 GPU 标记 kernel，用于 VTune 区分初始化 warmup 和请求。普通执行关闭这些标记。它们围住 backend 原有 profile 区间，初始 embedding/PLE 提交不在该 GPU 区间内。
- `q4e_cache_replay.py` 精确检查实际 GPU key/slot 映射、先保护全部 batch hit 再分配 miss、8-candidate 搜索、demote、计数和 payload。host-ready 决策使用运行时记录，而不是离线猜测异步 prefetch 完成顺序。
- 新模拟器的反事实策略只预测 GPU miss/字节，不预测 host cache、文件读取量、DMA fence 等待或 wall time。旧二进制模拟器已标记为 legacy，不能以 `count==top_k` 充当当前运行时验证。

新工具：`perf/qwen4exp/q4e_execution_probe.ps1`、`q4e_cache_replay.py`、`q4e_export_timeline.ps1`、`q4e_timeline_summary.py`。runner 记录输入/二进制/模型 config 哈希、有效实验开关和栈信息；单 GPU 进程串行，240 秒超时、系统可用 RAM 低于 8 GiB 停止。VTune 父进程及其直接子进程的 working set 不能当作推理进程的完整内存；内存结论使用不带 VTune 的模型进程样本。

## 2. P0：实际瓶颈比 host 等待总和更明确

[MEASURED] 非 VTune 的 story 和重复 lighthouse256 trace 都逐条重放通过。两次 256-token 输入分别命中 `456/10341`、`139/10341`；第二次仍传 `25,855,948,800 B`。两次 logits FNV64 都为 `1271920800156241593`，ReadFile 的 mmap fallback 为 0。

VTune 的默认 interval 导出会合并事件；分析器拒绝 `duplicateCount!=0`。本轮导出将分辨率提高到 10^7，较长记录提高到 10^8，并验证标记配对。采集器会遗漏进程末尾的一部分事件：分析器明确排除末尾未闭合的 forward，不用 host 时间伪造结束点。内存 canary 新增有界重复次数（默认仍为两次），以第三次 forward 保证第二次暖 forward 可分析。

[MEASURED] `p0-marked-long3-20260929` 中第二次 256-token forward：

| GPU 标记区间指标 | 时间 |
|---|---:|
| 总跨度 | 3995.78 ms |
| kernel 区间并集 | 1102.28 ms |
| 所有方向 copy 区间并集 | 2676.60 ms |
| kernel/copy 重叠 | 0.134 ms |
| 没有记录到应用 kernel/copy 的区间 | 217.03 ms |

共有 2042 个 kernel、10395 个 copy 事件。最后一列意义是应用 trace 未覆盖，**不是硬件 idle/EU utilization**；本轮未取得详细硬件利用率。host acquire 约 1480 ms 与上述区间重叠，不能相加，也不能全部当作可回收停顿。

同配置不带 VTune 的三次暖模型 forward 中位数约 3948 ms，和该长输入时间线接近。相比之下，短 story 在 profiler 下被明显扰动：30-token Prefill 约 4.47 s，非 VTune trace 约 1.69 s；其 15 个真实 decode forward 每步约 1070 个 kernel、kernel 并集约 38 ms，但采集到的约 146–302 ms 总跨度不能当作普通 TPOT。初始化两次 forward 与最后只更新状态、不产生 logits 的 forward 均单列。

**决策：长 Prefill 首先需要让 copy 与专家计算重叠，并继续降低搬运字节。仅减少 host fill 或普通提交间隙，不足以解释需要回收的约 1.4 秒。** 此处不从 profiler 直接承诺任何加速比例。

## 3. P1：首个保护区候选不准入

[HYPOTHESIS / TRACE SIMULATION] 候选为 resident 再次使用后进入 75% protected 区、demote 不淘汰 protected、其余作为流式工作区。容量固定 3714 槽，batch 过大时允许安全退让。没有使用固定每层 pinning。

| 同一真实路由 | 实际当前策略 miss | 无 cost-aware 的 LRU+demote | 候选 SLRU75 | 离线未来已知参考 |
|---|---:|---:|---:|---:|
| story 的 47 个真实 decode（pos 30–76） | 3429 | 3427 | 3586 | 2462 |
| 第一次 lighthouse256 | 9885 | 9885 | 9890 | 9882 |
| 第二次 lighthouse256 | 10202 | 10149 | 9890 | 6627 |

为隔离保护策略，候选应先与同样关闭 cost-aware 的 LRU 比较。它在 story 增加约 4.6% miss；第二次长输入虽有小幅改善，不能抵消跨场景退步，**不增加该运行时实现、不推广默认**。

离线参考使用整条路由的未来信息并保留整个 batch，证明还有字节层面的空间；不是在线可达到的承诺。第二次长输入的 6627 miss 恰好对应 10341 次需求减去 3714 个全部有用的初始槽。后续缓存候选需要更好的 admission 信号，不能只调整保护区比例或恢复静态 pinning。

## 4. P2：先验证供给机制，再决定重构

[MEASURED] 固定 lighthouse256、max_seq=512、dense XMX=1、ReadFile、grouped XMX/GDN chunk=0；每个配置单独进程，首次 forward 单列，之后三次相同输入 reset/replay。下面是外围 wall（包含最终 logits 下载与哈希检查），同一测量口径。每组四次 logits 哈希完全相同；host/device 分配与 expert slot 数不变。

| 配置 | 三次暖 forward（ms） | 中位数 | tok/s |
|---|---|---:|---:|
| 当前 fill_parts=8，demand_microbatch=0 | 3948.21 / 3949.94 / 3937.80 | 3948.21 | 64.84 |
| fill_parts=1，demand_microbatch=0 | 3948.62 / 3939.48 / 3942.38 | 3942.38 | 64.94 |
| fill_parts=8，demand_microbatch=64 | 3665.71 / 3664.93 / 3665.88 | 3665.71 | 69.84 |
| fill_parts=8，demand_microbatch=32 | 3653.14 / 3657.42 / 3649.07 | 3653.14 | 70.08 |

减少 ReadFile 分片确实缩短了 fill：末次 fill 从约 1204 ms 降到 944 ms，但总时间仅变约 0.15%，没有准入依据。供给期间存在重叠，不能把 fill 减少量加到吞吐上。

**既有 B6 的 32 专家小批将本口径的 Prefill 时间缩短约 7.5%。** 它不是本轮新发明的流水线，也不是 X4 基线从 70 tok/s 再提升 7.5%：X4 的缓存预热方式和固定 continuation 不同。本轮的三个暖样本来自一个进程，尚不是三个独立进程的发布门槛。32 与 64 相差约 0.34%，不据此建立形状阈值。

### B6 的收益为什么只有约 7.5%

[MEASURED] `p2-mb32-profile-20260929` 第二次暖 Prefill 的时间线：

| 指标 | 整层 acquire | B6=32 |
|---|---:|---:|
| GPU 标记跨度 | 3995.78 ms | 3655.49 ms |
| kernel 区间并集 | 1102.28 ms | 3106.88 ms |
| copy 区间并集 | 2676.60 ms | 2670.14 ms |
| kernel/copy 重叠 | 0.134 ms | 2175.18 ms |
| trace 未覆盖 | 217.03 ms | 53.65 ms |
| kernel 数 | 2042 | 4085 |

小批已经实现大量重叠；但 kernel 区间显著变长，尤其 four-token expert gate/up 的累计区间从约 267 ms 增至 1642 ms。不能把它直接解释成算术量增加或纯提交成本：小批形状、并发 copy 的资源竞争和事件时间戳语义还没有隔离。**下一步先做 gate/up 的相同工作量、不同批量以及有/无并发 copy 对照，再决定 readiness 管线的批量合并方式。** 全面重写 acquire 或继续减小 batch 并没有现成的收益保证。

最终默认准入仍需 512/1024 输入、长输入后的固定 decode、不同 prompt 的质量/状态与冷暖分离检查。后续有界 readiness 驱动 acquire 必须保留 batch 合并及 reader/DMA 生命周期，并检查 metadata copy 和 consumer fence 的限制。

### 短请求反证：不能把文件页预热算成 decode 优化

依次运行 baseline → B6=32 → baseline，各为独立进程，story 输入 30 token、生成 48 token，三次输出文本完全一致：

| 运行顺序 | Prefill（ms） | 47 个真实 decode 的均值（ms/token） |
|---|---:|---:|
| baseline 首次 | 4189.60 | 137.00 |
| B6=32 | 1726.05 | 79.42 |
| baseline 复核 | 1698.52 | 77.96 |

第三次 baseline 也恢复了速度，说明第一组巨大差值受 OS/host 数据热度影响，不能归功于 B6。这里没有隔离文件页和 host prefetch 的各自贡献；不把它称为已经测得的某一种 I/O 成本。B6 对短 Prefill 也没有已证实的收益。**本轮不声称 decode 提速，更未达到 20 tok/s。**

## 5. 验证与内存

- 4 个相关 CTest 全通过（tolerance、expert readiness、expert reader、DMA timing），总计约 14.95 秒。Python 的 4 个测试覆盖 batch hit 预留、cost-aware 搜索、demote 与空槽语义、损坏/截断 trace 拒绝。
- 非 VTune 模型进程 working set 峰值约 21.11–21.18 GiB，系统最低可用 RAM 约 27.78 GiB。所有完成的模型采集进程退出 0，未触发 watchdog；没有以增加 RAM/VRAM 预算获得收益。最初旧临时 runner 的 VTune 参数拼接失败发生在应用启动前，已修复并由新 runner 取代，不纳入有效样本。
- P2 各组报告 device allocation `14,295,025,686 B`、pinned host `17,360,839,684 B`、host pool `17,178,163,200 B`；它们是引擎分配计数，不等同于 DXGI resident VRAM。
- 工作区保留源码和复现工具；本轮没有提交或推送。机器可读证据见 [结果 JSON](qwen3.8-flash-next-p0-p2-results-2026-09-29.json)。

## 6. 复现

```powershell
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Tag p0-story -Trace
python perf/qwen4exp/q4e_cache_replay.py tmp/p0-story.cache.jsonl --output tmp/p0-story.replay.json
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Tag p0-long -Workload memory -MemoryRepeats 3 -Vtune -Trace
pwsh -NoProfile -File perf/qwen4exp/q4e_export_timeline.ps1 -Prefix tmp/p0-long
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Tag p2-base -Workload memory -MemoryRepeats 4 -FillParts 8
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Tag p2-mb32 -Workload memory -MemoryRepeats 4 -FillParts 8 -DemandMicrobatch 32
python -m unittest discover -s perf/qwen4exp -p test_q4e_cache_replay.py
```
