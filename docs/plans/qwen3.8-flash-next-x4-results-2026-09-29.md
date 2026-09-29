# X4：chunked GDN 数学原型与 DG2 映射

日期：2026-09-29。基线 commit 42b52181643ab3af054f01eb3b1777ba4fb797ed 加 X2/X3/X4 工作树。

**X4 原型与有界验证完成。GPU FP32 chunk 是本轮较好的候选：GDN 算子快约 28%–31%，模型 prefill 快约 1.2%；默认仍保留原递推。** XMX 版已实现并验证，但完整路径不如 FP32 chunk。长文 logits 差异较大，两个 canary 的 NLL 未触发预警并不足以扩大默认准入，因此两条新路径都保留 opt-in。

## 实现与开关

AILA_Q4E_GDN_CHUNK_MODE 在模型加载时读取并缓存：

| 值 | 路径 |
|---|---|
| 0，默认 | 原逐 token GPU recurrence |
| 1 | T=16，GPU FP32 chunk 三角求解与矩阵乘 |
| 2 | 同样的 FP32 求解，最后矩阵乘的两个输入转 BF16，用 joint_matrix、FP32 累加 |

计算均在 GPU。单 token decode 不变；Q4E 只在 A770、128 维 K/V、kernel=4、合法 grouped heads 且 scratch 容量足够时进入新路径。其他情况在提交前回退。X3 grouped expert 路径在本轮对照中固定关闭，dense XMX/direct I/O 固定开启。

新增源文件 src/ops/GdnChunkOps.hpp/.cpp。先按 token/head 并行完成 conv、Q/K 归一化、V、beta/decay 准备；每 chunk 形成因果系数、计算初始 state 投影、解三角系统，再更新该 chunk 输出及最终 FP32 state。最后统一 norm/gate，并提交 conv ring history。没有把完整 head state 放进 SLM 或每线程寄存器。

FP32 投影和三角求解中，固定 T16 循环显式展开。XMX product 使用 SG8、WG64、BM16/BN32、8×8×16 fragment。其余 kernel 使用 SG16/WG128。FP32 gate 累积、求解、S0 投影、最终 recurrent state 均保留；没有把所有状态降到 BF16。

## 数学契约

令 a_i 为 decay，beta_i 为更新系数，q_i/k_i 为归一化向量。以 K×V 排列 state，原递推为：

~~~text
d_i = beta_i * (v_i - a_i * k_i^T S_(i-1))
S_i = a_i * S_(i-1) + k_i d_i^T
o_i = q_i^T S_i
~~~

chunk 初始状态为 S0，G_i=product(a_0..a_i)，R_ij=product(a_(j+1)..a_i)，对角 R_ii=1：

~~~text
L_ij = beta_i * R_ij * dot(k_i,k_j), j<i; 其他为 0
(I+L) D = beta * (V - G * K S0)
O_i = G_i * q_i^T S0 + sum(j<=i) R_ij * dot(q_i,k_j) * d_j
S_end = G_end * S0 + sum(i) R_end,i * k_i d_i^T
~~~

这是直接三角形式，没有物化 WY 的 W/U，也没有形成 exp(prefix) 的倒数。GPU 用非正 gate 对应的 decay 连乘求 G/R，decay 下溢为 0 时不会出现 0/0。最后两式合成 [T+128,T] × [T,128] 的矩阵乘：前 T 行产生输出，后 128 行更新 state。

CPU double 的独立 recurrence 与 T16/T32 chunk 公式，在非零初始 state/history、35 token 下 state/raw-output NRMSE 约 4e-16。DG2 首版只映射 T16，T32 仅完成 CPU 数学检查，不宣称已经实现或测量 GPU T32。

## 性能及归因

[MEASURED] Intel Arc A770，驱动 32.0.101.8860，oneAPI 2026.1.0，oneDNN 3.11.2。算子使用 q_heads=16、v_heads=48、K=V=128；随机但固定的 BF16 投影输入、非零 FP32 state/history。对照包含 conv、准备、求解、矩阵乘、norm/gate 和 history commit，排除 state reset/H2D。

每路径预热 3 次，随后 3 组各 20 次轮换样本。以下为各组中位数的中位数，单位 ms；GPU span 使用 event 时间戳，包含两端 marker 之间的队列间隙，不是纯 kernel 时间之和。

| token 数 | 路径 | wall | GPU span |
|---|---|---:|---:|
| 30 | 原递推 | 0.7796 | 0.6386 |
| 30 | FP32 chunk | 0.5593 | 0.4138 |
| 30 | XMX chunk | 0.6759 | 0.5286 |
| 256 | 原递推 | 4.9306 | 4.7669 |
| 256 | FP32 chunk | 3.3789 | 3.1981 |
| 256 | XMX chunk | 4.6831 | 4.4997 |

FP32 chunk 的 wall 分别减少 28.3%/31.5%。各轮 CV、p10/p90 见 JSON，最终候选无 CV>10% 的轮次。这里没有等价的单一 oneDNN GDN primitive 对照，也没有另建库版本的完整三角管线；结论限于三条已测实现，不宣称是 DG2 的最优算法。

### 定向 profile 与负结果

先对原递推采集 VTune instruction-count，再对 chunk 初版和循环展开版各采集一次。原递推、准备、求解及两个 product kernel 的 spill 字段均为 0。

- 初版 256-token FP32/XMX chunk 约 28.2/29.7 ms，显著回退。
- 只把 Q/K SLM 行跨度从 128 改成 129，结果约 28.0/29.4 ms，没有可重复的实质收益；不能将 bank conflict 猜测写成已确认根因，已撤回该改动。
- 显式展开固定 16 行循环后，相同 64 次 solve 调用的动态指令总数从 2,367,415,296 降到 690,935,808（减少 70.8%）；send 从 339,551,232 降到 131,490,816。没有引入 spill。未插桩 wall 才用于上表。

负版本保存在 perf/qwen4exp/experiments/gdn_chunk_t16_unpadded.cpp 与 gdn_chunk_t16_padded.cpp，均不参与常规构建。profile 的时间受插桩影响，不能把其时间作为生产加速比；CSV 与原始记录在 tmp/x4-{baseline,chunk,unrolled}-profile*。

## 模型 canary

模型 C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2，max_seq=512，ReadFile reader。每路径先预热，再进行三组轮换计时。host/GPU 专家缓存仍有容量限制；正常重算 routing，后续 token 由外部固定，计时轮次不下载 logits。

| 输入 | 路径 | prefill ms | prefill MAD ms | 总时间 ms |
|---|---|---:|---:|---:|
| story 30 | 原递推 | 1370.18 | 1.44 | 4838.36 |
| story 30 | FP32 chunk | 1354.31 | 2.28 | 4823.98 |
| story 30 | XMX chunk | 1351.34 | 0.35 | 4824.73 |
| lighthouse 256 | 原递推 | 3627.87 | 0.16 | 4304.26 |
| lighthouse 256 | FP32 chunk | 3585.45 | 4.34 | 4264.02 |
| lighthouse 256 | XMX chunk | 3619.06 | 0.08 | 4294.84 |

story 总时间为 prefill + 47 次固定 token decode，long 为 prefill + 7 次固定 token decode。两行总时间不能横向比较；它们不是文本 callback TTFT 或自由生成时延。每次候选 prefill 恰好执行 36 次 chunked GDN，decode 不计入新分发。routing 会因数值变化而改变，不能把全部模型差值归结为纯 GDN kernel 节省。

模型 NLL/KL 的采集轮次独立于计时：

| 输入 | 模式 | 行数 | 平均 KL(ref‖cand) | 平均 NLL 增量 nat/token | 最大 logits NRMSE |
|---|---|---:|---:|---:|---:|
| story | FP32 | 48 | 0.00196356 | -0.0125693 | 5.57% |
| story | XMX | 48 | 0.0022403 | -0.0150282 | 7.16% |
| long | FP32 | 8 | 0.0158226 | -0.0376396 | 23.51% |
| long | XMX | 8 | 0.0410126 | -0.0961328 | 28.81% |

全部 logits 有限，NLL 未超过 +0.01 的初始预警线。但 long 只有 8 个观察行，logits NRMSE 较大；NLL 为负不等于完整模型质量认证。本轮没有继续进行模型首分歧定位或长上下文质量扫描，因此不把这两条路径推广到默认。算子 FP32 误差很小，仍不能据此忽略多层路由后的模型差异。

## 状态语义与内存

- 独立 GPU 检查对照 CPU double reference，包含非零 state、非零 conv history、ring head=2、BF16 输出和 FP32 state。最终算子测试中 FP32 state NRMSE 约 7e-7；XMX 约 0.21%–0.24%，output 约 0.23%–0.31%。
- 35-token 输入在 1/15/16/17/32 处分段 continuation，检查输出、最终 state、精确 conv history 与尾部 guards。保存/恢复同一 prefix 后，重复 suffix 的 state/output 逐位一致。
- 相等 Q/V head 数、SiLU gate、a_log=10（decay 下溢）和 -20（近单位 decay）均通过；unsupported geometry/不足 scratch 在提交前拒绝。
- 真实模型先执行 4 token，再做 30-token forward，anchor=17 必须发布绝对位置 21。FP32/XMX 两模式均通过 snapshot restore/capture 逐位检查、重复 replay 逐位检查，以及改变 anchor 后所有 token 不影响 GDN/conv/PLE 快照的检查。复用已有 X2 的 run_range/D2H 排序，不把未来 state 放进快照。

workspace 只按执行槽容量预分配并复用，不按层或 chunk 反复分配。64 行容量为 7,545,024 B，256 行为 26,493,120 B（约 25.27 MiB），全部计入 device ledger；没有额外 pinned host arena，没有 BF16 专家缓存，也不改变 3714 个 GPU expert slots。默认模式 0 不分配该 workspace。临时容量随 token 数线性增长。

完整模型验收峰值进程 working set 约 21.33 GiB，最低系统可用 RAM 约 27.93 GiB，未触发 8 GiB 保护；256-token 后 device ledger=14,321,520,854 B、pinned host=17,360,839,684 B。账本不等同于 DXGI resident VRAM。本轮没有发生挂起、非零 GPU 测试退出或 device lost。

## 验证与复现

Aila 与相关测试目标已构建。GdnChunk、Module、Generation、Tolerance、GroupedXmx、DmaTiming 六项定向 CTest 全部通过，34.76 秒；没有重复全量 CTest。

~~~powershell
pwsh -NoProfile -File perf/qwen4exp/q4e_gdn_chunk_acceptance.ps1 -Mode kernel -Tag x4-kernel
pwsh -NoProfile -File perf/qwen4exp/q4e_gdn_chunk_acceptance.ps1 -Mode bench -Tag x4-ops
pwsh -NoProfile -File perf/qwen4exp/q4e_gdn_chunk_acceptance.ps1 -Mode model -Tag x4-model
pwsh -NoProfile -File perf/qwen4exp/q4e_gdn_chunk_acceptance.ps1 -Mode profile -Candidate -Tag x4-profile
ctest --test-dir build-refactor -R '^(AilaGdnChunkTests|AilaQwen4Exp(Module|Generation|Tolerance|GroupedXmx|DmaTiming)Tests)$' -j 1 --output-on-failure
~~~

实验使用方式：在启动新进程前设置 AILA_Q4E_GDN_CHUNK_MODE=1（GPU FP32）或 2（XMX）；清除变量或设 0 回到原递推。

主算子证据绑定 exe SHA256 5211A0508F8CCA1169EEF32607B6D451F2451C778DC4D1C6E3C26C14B72B43C8，后续仅扩展了边界测试。模型证据绑定 E46F6E11D05715DDEDE6F585A25F344097691442FD10CD03A0226AC3F58F86F2。source hash、全部轮次中位数/p10/p90/CV、模型样本/MAD、内存摘要、profile 表及回归日志摘要见 [JSON evidence](qwen3.8-flash-next-x4-results-2026-09-29.json)。
