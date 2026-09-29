# X3：专家分桶与 grouped XMX 首版

2026-09-29，基线 commit 42b52181643ab3af054f01eb3b1777ba4fb797ed 加 X2/X3 工作树。

**首版实现和有界验证完成；所有桶继续默认使用原 ALU。** M8/N32/K32 grouped XMX 在本轮没有获胜：story prefill 慢约 3.0%，256-token prefill 慢约 14.6%，后者的模型 NLL 差异也超过方案初始预警线。保留实验实现与负结果，不推广为默认，也不据此否定其他 grouped XMX tile 或 staging 实现。

## 实现范围

- AILA_Q4E_EXPERT_HISTOGRAM=1：在已有 host 路由结果上统计 1/2/3–4/5–8/9–16/>16 六桶的 layer-expert 请求数、activation 数、驻留命中与未命中；没有增加 GPU 同步。日志是累计值；命中表示 acquire 前的 resident lookup，不等同于 DMA 已完成或事后的 acquire 统计。
- AILA_Q4E_GROUPED_XMX_MIN_ROWS 默认 0（关闭）；=8 是本轮候选。仅 A770、H/I 均为 64 的倍数、count>1、expert-major 且能批量 acquire 时进入候选分发。8 是探索点，没有成为已验证的 crossover。
- 复用现有 expert slot 和 token/rank map。低于阈值的专家保留 4/2/1-row ALU；较热专家生成 MoeExpertTile{first, rows}，每 tile 最多 8 行，尾部补零但不读写无效 token/rank。
- src/ops/MoeGroupedXmx.cpp：SG8、WG32、BF16 8×8×16 joint_matrix，FP32 累加；在小型 SLM B tile 内解 NF4/nested scale。gate/up 共享 gather 后的 A、分别累加，SwiGLU 后写已有 BF16 h；down 写已有 FP32 rank buffer，由原 merge 加权合并，没有浮点 atomic。
- 普通批量与 B6 demand microbatch 均接入。后者保留 ready 优先排序、每批 acquire/DMA 依赖、预取与 demote；不同批使用互不覆盖的 host/device 描述符切片，最后 down event 登记到 cache consumer fences。
- 描述符在 runtime buffer 扩容时预分配，decode 不新增 USM 分配。256 行容量增加 20 KiB device + 20 KiB pinned host；没有 BF16 专家权重 arena，GPU cache 仍为 3714 slots。

## 固定工作负载测量

[MEASURED] Intel Arc A770、驱动 32.0.101.8860、oneAPI 2026.1.0、oneDNN 3.11.2；模型 C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2，max_seq=512，Windows 默认 ReadFile reader。

两边固定 dense XMX/direct I/O，单次完整 prefill，只切换 grouped MoE。各预热一次，再做三组交错 A/B。继续正常重算 routing，后续 token 由外部固定；计时轮次不下载 logits。host cache 已预热但容量有限，不宣称所有 host/GPU 专家都命中。以下是 host wall 中位数，括号为 prefill MAD。

| 输入 | 原 ALU prefill ms | grouped prefill ms | 原 ALU 总时间 ms | grouped 总时间 ms |
|---|---:|---:|---:|---:|
| story 30 token | 1381.94 (0.13) | 1423.32 (3.32) | 4846.62 | 4892.20 |
| lighthouse 256 token | 3633.54 (2.57) | 4165.57 (0.57) | 4315.94 | 4844.57 |

story 总时间包含 prefill + 47 次固定 token decode；long 包含 prefill + 7 次固定 token decode。不能横向比较两行总时间，也不将它们冒充 Engine 文本 callback TTFT 或自由生成延迟。

### 真实分桶

以下来自 ALU 质量对照轮的一次 prefill，覆盖全部 48 层。列中为 activation 数，括号为 layer-expert 请求数。

| m_e 桶 | story 30 | long 256 |
|---|---:|---:|
| 1 | 2207 (2207) | 2096 (2096) |
| 2 | 1888 (944) | 2464 (1232) |
| 3–4 | 2913 (873) | 5124 (1486) |
| 5–8 | 3217 (530) | 10291 (1637) |
| 9–16 | 2792 (243) | 20468 (1713) |
| >16 | 1383 (72) | 82437 (2177) |
| 总计 | 14400 (4869) | 122880 (10341) |

grouped 候选轮实际交给 XMX 的 activation 分别是 4752/14400（33.0%）和 105564/122880（85.9%）。候选会重算路由，因此这两个比例不是从上表强行复用路由得到的。long 的热专家占比已经很高，仍没有请求级收益，不能把负结果全部归因于 story 太短。

### 真实专家算子对照

复用同一份真实权重、token/rank 和 activation，仅测候选专家的 gate/up → SwiGLU → down。包括 GPU gather、tile 解量化和 rank 写回；不包含权重 acquire/H2D、host 元数据复制与最终跨专家 merge。5 次预热，再做 3 组各 20 对交错 wall 样本，JSON 保留各组中位数、p10/p90、CV。

| prompt / layer / m_e | ALU ms | grouped XMX ms | down NRMSE |
|---|---:|---:|---:|
| 30 / 1 / 8 | 0.0437 | 1.4987 | 0.3462% |
| 30 / 0 / 9 | 0.3422 | 1.4222 | 0.2582% |
| 30 / 1 / 26 | 0.0624 | 1.7258 | 0.2631% |
| 256 / 0 / 8 | 0.0501 | 1.5673 | 0.3689% |
| 256 / 0 / 16 | 0.0502 | 1.6081 | 0.1950% |
| 256 / 0 / 59 | 0.6398 | 1.5988 | 0.2351% |

这里是各组中位数的中位数。多数 ALU 小任务 CV>10%，已标 noisy；不从离散样本推出连续阈值或精确倍数。没有新增 oneDNN staging 基线、VTune 全扫或 device-event 计时，不能据此宣称其他矩阵实现也会输。按本阶段停止规则保留现有赢家；若继续调 tile，应先对目标 kernel 做 device/host 时间分解和定向 profiling。

## 数值、生命周期与内存

模型质量轮独立于计时：

| 输入 / teacher-forced logits 行数 | 平均 KL(ref‖cand) | 平均 NLL 增量 nat/token | 最大 logits NRMSE |
|---|---:|---:|---:|
| story / 48 | 0.00161066 | -0.00930087 | 8.91% |
| long / 8 | 0.0192468 | +0.184535 | 12.99% |

全部 logits 有限；long 超过 +0.01 初始预警线，未满足默认准入条件。算子误差低于 1% 不代表经过多层路由和递归状态后模型误差也合格。这只是两个 canary，不能当作广泛质量评测。

独立算子 CTest 使用三个不同专家、8/9/17 行、非连续 token/rank、定制 NF4 LUT、非零 nested offset 和变化的二级 scales，对照 CPU double 累加/BF16 权重参考。普通/nested 两种路径 h NRMSE=0，down NRMSE=6.53e-8；未写入的 rank 和尾部 guards 不变。独立 DMA queue 填权重，XMX 等 ready；随后 copy queue 等 down event 才覆盖权重，验证 ready → consumer → reuse。

B6 microbatch=16 的真实 story + 3 次 continuation canary 正常结束，候选覆盖 4752 activation，平均 KL=0.000421444，NLL 增量=0.00333884。它只有一对计时样本，用于集成检查，不用于默认性能选择。

最终完整验收进程峰值 working set 21.29 GiB，最低系统可用 RAM 28.09 GiB，private commit 峰值约 31.43 GiB，未触发 8 GiB 可用 RAM 保护。无算子 probe 的独立 long A/B 完成后：device ledger=14,295,027,734 B，pinned host=17,360,839,684 B，仍为 3714 GPU slots。完整测试另有预分配的 29,777,920 B 测量专用 arena，这不是生产新增缓存。账本数不等同于 DXGI resident VRAM。

### 测量回调失败记录

初版 probe 在 live model 回调里反复分配/释放临时 USM。两次运行在 long layer 0 的 m_e=8 probe 处退出 0xC0000005；第二次定位到 ze_intel_gpu64.dll+0x308b4e。RAM 当时充足，查询中未发现相应 Display/TDR 系统事件。停下性能扩展，先运行关闭 probe 的相同 long A/B，完整结束；随后把测量 arena 改为预分配并复用，最终完整 canary 正常结束。证据支持规避这个触发方式，尚不能确定驱动内部根因或证明永不复现。

原始失败日志和身份保留在 tmp/x3-model.*、tmp/x3-probe-diagnostic.*；成功的隔离运行在 tmp/x3-long-diagnostic.*，最终数据在 tmp/x3-final.*。失败进程的部分数据不用于主结论。

## 验证与复现

Aila 和四个相关测试可执行文件已构建。GroupedXmx、Module、Generation、Tolerance、DmaTiming 五个定向 CTest 全部通过，31.60 秒；没有重复全量 CTest。测试退出成功表示结构/生命周期检查通过及指标采集完成，不表示 long 的质量预警被放行。

~~~powershell
ctest --test-dir build-refactor -R '^AilaQwen4Exp(GroupedXmx|Module|Generation|Tolerance|DmaTiming)Tests$' --output-on-failure -j 1
pwsh -NoProfile -File perf/qwen4exp/q4e_grouped_xmx_acceptance.ps1 -Mode kernel
pwsh -NoProfile -File perf/qwen4exp/q4e_grouped_xmx_acceptance.ps1 -Mode model -Tag x3-final
pwsh -NoProfile -File perf/qwen4exp/q4e_grouped_xmx_acceptance.ps1 -Mode model -Microbatch 16 -Tag x3-microbatch
~~~

最终验收 exe SHA256：67F5FF3DD9F16CB5CA9C66A799CF64ED8EF8483AC695A8D0BB7EB05D85540125。
样本、MAD、分桶命中/未命中、内存摘要、环境覆盖、源码和二进制身份以及失败记录见 [JSON evidence](qwen3.8-flash-next-x3-results-2026-09-29.json)。定向测试日志：tmp/x3-ctest.log、tmp/x3-ctest-detail.log。
