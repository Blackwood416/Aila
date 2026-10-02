# Qwen3.8-Flash-Next：QSA 与长上下文执行差距

后续实现见 [精确 QSA selector 与提交方式实验](qwen3.8-flash-next-qsa-selector-results-2026-09-30.md)：已在 A770 默认启用经过验证的 radix＋vec4 路径。下文保留修改前的诊断基线。

**已找到可测量的优化空间：QSA 的精确 Top-K 随上下文增长，吞掉大部分稀疏 attention 收益；当前约 2K 输入的整机降速，还伴随更大的专家供给等待。** 本轮完成源码审计、独立成本曲线和整模型分段诊断，未更换生产 kernel。继续 32 GiB host pool、GPU-only 专家算术。

设备为 Arc A770，驱动 32.0.101.8860，oneAPI 2026.1 / oneDNN 3.11.2。本地 Strata 为 a790805，工作区干净。数据与元信息见 [JSON](qwen3.8-flash-next-qsa-gap-audit-2026-09-30.json)。

## QSA 的收益边界

Aila 主 attention 为 Hq=24/Hkv=2/D=256；indexer 为 4 个 128 维 query heads，每 4 token 压缩一个索引 key，选择 512 个完整块，再加尾部 token。压缩的是 indexer key，主 K/V 仍保存原始 BF16，通过块索引访问。

2051 token 以内走 dense，2052 开始选择块。但 2146 token 时仍读取 2050 token，只减少约 4.5%；4K/8K/32K 时才分别约为完整上下文的 1/2、1/4、1/16。阈值附近并不必然立刻加速，筛选开销必须低于节省。

Strata 自带 [0.1.26 基准](../../tmp/Strata/bench/results/2026-09-29-speed-0126/README.md) 的 Q2_0 为 1K 87.3 → 4K 93.0 tok/s，但 IQ2_XS 为 79.6 → 78.6，Coder 为 58.9 → 55.1。该协议含 MTP、不同 KV 格式、CPU 专家计算，每格仅一次请求，文档说明接受率会影响结果。它不是 QSA 单变量消融，不能将所有增长归给 QSA。

此前 Aila 的 307/2146-token 对照也改变了输入、生成内容、max-seq、专家槽数和 scratch 配置，不能单凭该曲线证明 QSA 是全部原因。

## 独立算子证据

[MEASURED] 固定 BF16 Q/K/V、32768 容量，Hq=24/Hkv=2/D=256。每操作预热 5 次，三轮、每轮 20 个同步 wall 样本，另记 20 次连续提交均摊 wall。表为三个中位数的中位数，单位微秒，不是硬件事件时间或生成吞吐。

| KV 长度 | scoring | Top-K | dense attention | selected attention | 实际分支 score+select+attention |
|---|---:|---:|---:|---:|---:|
| 512 | 79.6 | 86.9 | 359.8 | 373.9 | 357.2 |
| 2048 | 78.7 | 82.5 | 256.4 | 316.1 | 256.4 |
| 2051 | 80.5 | 82.2 | 268.6 | 300.9 | 268.5 |
| 2052 | 81.8 | 79.4 | 268.9 | 268.9 | 324.4 |
| 2056 | 81.2 | 225.8 | 269.9 | 266.1 | 471.4 |
| 2146 | 80.8 | 269.6 | 270.7 | 281.2 | 468.8 |
| 4096 | 82.2 | 385.1 | 370.6 | 264.1 | 581.1 |
| 8192 | 85.4 | 589.4 | 558.1 | 267.8 | 812.3 |
| 32768 | 106.7 | 1954.6 | 2243.3 | 264.9 | 2194.3 |

各操作单独同步，不能把各列中位数相加代替最后一列。dense 阶段的实际分支不执行 score/Top-K；这些列只作独立测量。最后一列不含 projection、norm/RoPE、compression、gate、o_proj。

32K selected attention 已从约 2.24 ms 降至 0.265 ms，但 Top-K 花约 1.95 ms，几乎抵消收益。Top-K 连续提交均摊 wall 仍为 1.857 ms，不能把全部成本归给单次 host launch。512 比 2048 慢还涉及现有 head256 attention 在 1024 token 切换 partial-merge，说明执行策略能够改变曲线。

[CODE] [通用 selector](../../src/ops/QsaIndexerOps.cpp) 约 1170 行使用一个 256-thread WG：32 次逐 bit 筛选，每次手写 8 级 SLM reduction/barrier，最后 thread 0 两遍扫描候选并串行填充所选列表。513→512 已有“丢一个最差块”特例，514 开始回到通用路径，与 2052→2056 跃升吻合。

Strata 的 [selector](../../tmp/Strata/src/kernels/cuda/qsa_select.cu) 按 8-bit radix 做四轮 histogram，并行生成结果。算法组织值得借鉴，CUDA 常数仍需在 A770 重测。Aila 已有压缩 indexer K、块原生访问和稀疏 head256 partial-merge，不能把这些既有功能重新列为缺失项。

## 整机证据

[MEASURED/diagnostic] 两进程固定 RAM=32 GiB、max-seq=2560、专家槽=3567、chunk=512、phase scratch 开启，分别生成两次 16 token。表为第二请求 16 次单行 forward 均值，含最终 alignment，单位 ms；H2D 为 MiB。

| 项目 | 41-token story | 2146-token 日志背景 |
|---|---:|---:|
| forward 总计 | 111.07 | 177.84 |
| Full Attention，含 indexer/projections | 4.04 | 8.32 |
| Linear Attention，含 projections | 11.66 | 11.70 |
| 专家阶段 | 52.48 | 113.59 |
| 其中 acquire | 8.80 | 77.30 |
| fill | 0.00 | 35.88 |
| prefetch wait | 0.00 | 24.48 |
| 需求 H2D MiB/forward | 279.47 | 327.21 |

Stage profile 强制同步，改变重叠，不能当正式 TPS，也不能将 acquire 与专家阶段重复相加。输入和生成轨迹不同，路由会不同，不能断言 context length 自身导致所有 miss。但在当前慢请求中，QSA 显然不是全部差额；专家供给是另一项重要目标。

## 下一阶段顺序

1. **精确 QSA selector。** 先测 subgroup/native reduction 替换手写 SLM tree，再试 4/8-bit radix 与并行稳定 compaction，去掉 thread-0 全量串行扫描；近阈值扩展小补集选择。保持 512 块预算、尾部 token、NaN/有符号零规则、并列时较小 ID 优先及最终 ID 升序。覆盖 513/514/536/1024/2048/8192 blocks、尾部 0–3、并列与极近分数，先 CPU IDs 一致，再做整模型阈值两侧和 4K/8K 验证。不通过少选 token 伪造加速。

2. **长 Prefill 后供给与容量协调。** 固定 32 GiB，记录每层 host miss、prefetch useful/wasted、等待完成位置和 GPU residency，细分读取/转换/线程等待。当前 [ReadFile reader](../../src/models/Qwen4ExpExpertReader.cpp) 分别处理 gate/up、scale、down、scale，可测试上传槽布局的预打包连续记录或有界分段并发读取，先测细分成本再选方案。Host victim 主要看 LRU/lease/DMA，GPU victim 已感知 host readiness，可离线评估减少无用重复占用；必须计入 GPU eviction 后重读/回填，不将 D2H 当免费。不重跑失败的普通 protected-SLRU 或无条件全量预取。

3. **GQA 复用与融合。** [Aila attention](../../src/ops/AttentionOps.cpp) 约 267 行按 query head 分组；[Strata](../../tmp/Strata/src/kernels/cuda/qsa_decode_attn.cu) 在同一 KV head 的 12 个 query heads 之间共享 K/V 读取和转换。先测小 head-group 版本，再考虑 XMX QK/PV。缓存可能已经吸收部分重复读取，不承诺 12 倍带宽收益。保持 FP32 softmax，验收非连续块、尾部、gate。进一步融合 prepare、增量 compression、scoring，减少小 launch。

4. **真实 CLI 大块 Prefill，再重估 MTP。** 512-token chunks 会反复遍历专家集。S4 临时区复用需配合专家槽借还与实际显存预算，才能落到完整请求；固定预算比较 512/1024/2048 的需求字节、TTFT 和后续 decode，不能直接套用无快照 canary 的 100+ tok/s。target 成本下降后再测 verify/draft/commit 完整周期，MTP 继续 64 个 draft 槽。

selector 是最明确且局部的下一项实现；要把当前长输入真正推到 20 tok/s，供给问题也必须解决。两条线都保持 GPU 专家算术。

## 本轮交付与验证

新增测试专用 --qsa-probe 及汇总脚本，未改生产计算路径。9 个长度的选择 IDs 与 CPU oracle 一致；dense/selected attention 各检查两个 query heads 的 CPU f64 参考、全部 24 heads 的有限性，绝对误差阈值 2e-4。未在本轮重新认证 scoring 数值精度。探针无动态 USM 增长，退出 USM 归零；三次 GPU 进程正常退出，未触发 watchdog。测试目标构建通过，未跑无关全套 CTest。

~~~powershell
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload qsa -Tag qsa-cost-curve
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload generation -GenerationRepeats 2 -MaxTokens 16 -GenerationPromptFile perf/qwen4exp/prompts/ram-long.json -GenerationMaxSeq 2560 -PrefillChunk 512 -PhaseScratch -DemandMicrobatch 32 -Stages -Tag qsa-long-stages
python perf/qwen4exp/q4e_qsa_audit_summary.py --curve tmp/qsa-cost-curve-20260930 --short tmp/qsa-short-fixed-cap-stages-20260930 --long tmp/qsa-long-stages-20260930 --output tmp/qsa-audit.json
~~~
