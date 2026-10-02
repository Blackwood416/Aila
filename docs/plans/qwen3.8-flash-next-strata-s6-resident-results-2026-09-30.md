# S6：小窗口驻留专家与缺失专家拆批

2026-09-30，继续 S1–S5 后的目标 verifier 优化。A770 16 GB、32.0.101.8860、oneAPI 2026.1 / oneDNN 3.11.2。基于 66e21e9 后的未提交工作区，专家计算仍全部在 GPU。

**新增默认关闭的驻留边界拆批：中文、代码的四行目标验证分别降低约 3.2% / 2.7%，相同轨迹的全部 logits 位级一致。收益有限，20 generated tok/s 仍未达到。** 详见[逐遍结果和诊断 JSON](qwen3.8-flash-next-strata-s6-resident-results-2026-09-30.json)。

## 瓶颈证据与诊断修正

先修正一个 trace 问题：主模型和 MTP cache 读取同一个 trace 路径时，第二个 cache 的文件打开会截断前一个文件。现在 init 支持可选 suffix，MTP 写到同路径的 .mtp 后缀文件，target 保留原文件。此前没有用这种混合文件作 S1/S2 性能结论；本次新剖析先通过单 target cache 的完整回放校验。

新增逐 token 的 route 记录，保留层、行数、top-k 和原始 ID 顺序。校验器检查每行去重、ID 范围、每层记录完整性以及 route 与 acquire 专家集合一致；五项 Python 测试通过，包括损坏数据的拒绝检查。

[CODE] 原 B6 已把驻留专家排在前面，但仍按固定 batch size 截取。小窗口每层通常只有 20–30 个 unique experts，B6=32 经常将整层放进一个混合批次。acquire 返回的事件覆盖缺失专家的传输，于是已经驻留的专家也等这些数据后才执行。

[MEASURED] 中文前 24 token 的六个四行块、288 层实例中，共 362 个 acquire 批次，其中 283 个混合了驻留与缺失专家；这些批次内包含 5302 个已驻留专家、7755 个 activation，约占总 activation 的 67%。驻留不代表 DMA 已完成，这里只证明事件依赖过宽的调度机会，不能直接将该数量换算为时间收益。

## 改动边界与所有权

AILA_Q4E_SMALL_ROWS_RESIDENT_SPLIT=1 仅作用于 GPU ALU expert-major、2–4 行、已经启用 demand microbatch 的分支。在原有 stable partition 的驻留/缺失边界额外切一刀；描述符规划和实际 acquire 使用同一个批次大小函数。

- 保留每个专家全部 activation，继续用既有 4/2/1 行权重复用和按 rank 的确定顺序累加。
- 不引入新 kernel、权重格式、CPU 专家计算或动态 USM，也不主动预取额外专家；描述符拷贝被拆成更多次。
- 驻留批仍通过 acquire 的事件判断 DMA readiness；没有把 contains 当作 ready。
- 保留每批 reader fences 和不重叠的描述符切片。已有多个微批发生错误/取消时的 drain 约束不变。
- 较大 Prefill 和单行 decode 不走新分支；grouped XMX 分支不改。

测试增加 model 内部 override，使同一已加载模型可以按 AB/BA/AB/BA 对照，不通过运行时修改缓存的环境变量。这个 override 只由测试访问器设置。

## 正式对照

中文 122-token prompt 和代码 121-token prompt，各生成 64 个实际 target greedy token，然后用同一固定轨迹测四行 target 工作。MTP 64 槽、target 3520 槽、VRAM reserve 3008 MiB、B6=32、small-row NF4/HC 均相同；phase scratch 关闭，以隔离这一个结构变量。

每个配置四遍，按正反顺序交替；保留第一对数据。统计采用后三对逐遍平均块时间的中位数，比例取逐对降幅中位数，JSON 保留每块时间、全词表 logits 哈希、miss 和 H2D 字节。

| 输入 | 基线 ms/四行 | 拆批 ms/四行 | 配对降幅 | 后三遍基线/拆批 CV | 比较 |
|---|---:|---:|---:|---:|---|
| 中文 | 219.659 | 212.562 | 3.23% | 0.061% / 0.007% | 64 行 logits 全部位级一致 |
| 代码 | 233.683 | 227.333 | 2.66% | 0.185% / 0.261% | 64 行 logits 全部位级一致 |

中文最后两对和代码后三对的 miss、demand H2D 字节完全相同，因此这部分差异不是通过扩大缓存或省略专家获得。其它配对存在少量 prefetch/cache 状态差异，全部保留，未强制恢复缓存快照。

同一真实轨迹的 S1 KV-only draft/catch-up 成本单独测量后组合：[DERIVED] 中文约 7.74→7.97 条件 tok/s，代码约 14.25→14.61。没有 target 前缀提交、拒绝后真实 target 路径、采样/IPC 和完整交替循环，因此这些不是 generated TPS。代码仍缺约 67 ms/周期的 20 tok/s 预算；S3 的收益条件没有因此成立。此前 MTP CPU reference 的 3% NRMSE 失败仍保留。

## 为什么收益只有几个百分点

另用一个带 VTune 的 1–4 行 paired sweep 做诊断。各行宽的 24 行 logits 在拆批前后位级一致，单行是新分支不启用的控制。四行原有的 283 个混合批次在候选中变为 0。

| 四行诊断均值，六块/配置 | 基线 | 拆批 |
|---|---:|---:|
| marker span ms | 252.45 | 249.47 |
| kernel union ms | 91.94 | 134.90 |
| 所有方向 copy union ms | 132.60 | 131.14 |
| kernel/copy overlap ms | 10.36 | 54.18 |
| 无已记录任务覆盖的时间 ms | 38.28 | 37.61 |
| kernel 数量 | 1429 | 1547 |

重叠确实增加，但 kernel 总区间也明显增长，新增小批次及拷贝/计算的共享资源竞争可能共同抵消收益。本轮没有进一步隔离二者，不能把增长全部归因为某一种带宽瓶颈。它说明“制造更多重叠”不能直接等同于提速，下一项必须减少 GPU 工作、实际搬运字节或更具体的提交等待。

该表只有一个带收集器的 pair，不能替代无 trace 的正式时间表。“无覆盖”不是硬件 idle；copy 也不全是 demand H2D。GPU marker 不覆盖最初 embedding/PLE 提交，host/GPU 时钟没有直接相减。VTune wrapper/直接子进程的 WS 不能充当模型进程 WS。

## 验证与状态

正式两个模型进程峰值 WS 21.45 GiB、最低系统可用 RAM 26.22 GiB。全部本轮 GPU 探针 exit=0，未出现 watchdog 停止或设备丢失诊断；live USM 不增长，device checkpoint 恢复重放检查通过。构建了 tolerance 测试以及 CLI/Worker/DLL 目标，未重复全套 CTest，未执行服务取消/会话交错验收，也未宣称生产 MTP 验收完成。

默认保持关闭，源代码、诊断、正式对照及失败门槛均保留。没有重跑旧的 CPU 异构、SLRU、copy slicing 或失败 grouped XMX 候选。

复现：

~~~powershell
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload mtp -MtpSlots 64 -MtpPromptFile perf/qwen4exp/prompts/mtp-zh.json -MtpAnchors 60 -MtpRepeats 3 -MtpVerifyPasses 4 -MtpCompareKv -MtpCompareResident -DemandMicrobatch 32 -Tag s6-resident-zh
python perf/qwen4exp/q4e_resident_summary.py tmp/s6-resident-zh --output tmp/s6-resident-summary.json
~~~
