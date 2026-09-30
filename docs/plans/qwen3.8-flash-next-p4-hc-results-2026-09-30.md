# Qwen3.8-Flash-Next：小批量 HC 融合验证

2026-09-30，基于 0293b3b 加前几轮未提交改动。Intel Arc A770 16 GB，驱动 32.0.101.8860，oneAPI 2026.1 / oneDNN 3.11.2。GPU 计算、ReadFile、16 GiB host expert pool 保持原配置。

本轮给 2–4-token 验证增加默认关闭的 HC（Hyper-Connection）融合路径。**512-token 输入的四行验证从相邻对照的 199.95 ms/块降到 189.19 ms，独立复测为 189.64 ms，约减少 5.2%–5.4%。单请求生成 20 tok/s 仍未完成。** 这是完整主模型验证成本，没有 MTP draft、接受/拒绝、回滚或采样。前轮中长 Prefill 100+ tok/s 的 canary 结论不变，见 [P2/P3 结果](qwen3.8-flash-next-p2-supply-p3-front-results-2026-09-29.md)。

## 实现与范围

新增 AILA_Q4E_SMALL_ROWS_HC=1，对 2–4 行 HC 调用两个融合 kernel：down + SiLU + injection，以及 up + sigmoid + mix reduce。原始 BF16 权重保持 [output,input] 行主序，FP32 点积归约及中间 BF16 舍入规则沿用单行实现；每个 subgroup 仍计算一个输出，不同 token 使用独立 work-group。没有转成 NF4 权重，也没有改变 NF4 解码格式。

单行接口保留独立模板实例；默认分发不变。批量路径使用已有 hc_N、hc_mix_down、hc_inject_f32、x 缓冲区，无新增 USM、权重副本、状态快照或 oneDNN descriptor 绑定。pending residual/injection 的读取仍先在同一 in-order queue 上完成。开关依赖原有 HC fusion 开启，适用于所有 2–4-token 小段，包括 warmup 和 Prefill 尾段，不仅是推测验证。整模型测过 2/4 行、context≤640；3 行只有算子验证。保持 opt-in。

## 成本归因

先使用前轮两行 NF4 小批量路径的 VTune 轨迹。按同一 in-order compute queue 的相邻消费者识别 HC down/up/injection GEMM，得到 97/97/96 次，与 48 层的两个 HC 和最终 mixer 吻合。原有剩余 GEMM 总计约 19.67 ms/块，其中 HC 14.59 ms，LM head 3.20 ms，其他 dense 约 1.89 ms。60 次投影仅归类为 dense-before-NF4，其中包含 shared gate 和 indexer；不全部当成 shared gate。

[MEASURED] 新旧两行轨迹均保留第二遍的 forward 69–131，共 63 个完整块；两次 collector 都缺失末尾 forward 132 的结束事件，明确排除。高分辨率 TSV 已检查 duplicateCount=0。这里是基于消费者顺序的归因，未新增逐算子 profiling range。

| 两行诊断指标，均值 | 原小批量 NF4 路径 | 加 HC 融合 |
|---|---:|---:|
| HC kernel 数，不含相同的 norm | 677 | 194 |
| HC 全部 kernel 时间 | 16.418 ms | 6.980 ms |
| 全部 kernel union | 65.208 ms | 55.790 ms |
| 全部 copy union | 35.744 ms | 35.411 ms |
| 应用任务未覆盖时间 | 40.689 ms | 36.928 ms |
| forward span | 141.548 ms | 128.063 ms |
| demand H2D 字节/块 | 327.90 MB | 327.46 MB |

收益来自计算融合和减少提交。数值变化会轻微改变专家选择，所以传输量并非逐字节一致，不能把全部 span 差额直接归入 HC。未覆盖时间不是硬件 idle，也不能全部视为可删除的 launch overhead。未重新尝试已否决的 graph、copy slicing 或保护区 cache 策略。

## 整模型测量

沿用固定 128-token teacher forcing，重新 Prefill 后，每四个输入 token 执行一次完整 forward，输出全部四行 logits；每遍 32 块。core 包含 forward 提交至 GPU 同步完成，loop 另含 logits D2H、有限值和 CPU 哈希/argmax。缓存容量受限，未强制恢复 host/GPU cache 快照，未清空 OS 文件缓存。首遍和第二遍过渡样本保留。

[MEASURED] 512-token 输入，B6=32，max_seq=640，各进程五遍：

| 路径 | pass 0 / 1，ms/输入 token | pass 2 / 3 / 4，ms/输入 token | 后三遍中位块时间 |
|---|---|---|---:|
| HC 关闭，相邻对照 | 103.240 / 56.379 | 50.144 / 49.987 / 49.741 | 199.949 ms |
| HC 开启 | 99.898 / 56.016 | 47.511 / 47.297 / 47.218 | 189.189 ms |
| HC 开启，独立复测 | 99.552 / 52.709 | 47.498 / 47.411 / 47.173 | 189.644 ms |

后三遍是与前轮相同的报告窗口，不隐藏 pass 1：全部去首遍统计及原始逐块数据同时写入 JSON。上述进程前两遍 Prefill 约 12.8–18.0 秒 / 13.6–14.6 秒，后三遍约 4.36–4.72 秒；不拿首次轨迹作加速比。

30-token 输入，四遍的 core 分别为 68.397 / 43.787 / 43.520 / 43.296 ms/输入 token。后三遍中位数 **174.080 ms/四行块**，loop 为 177.900 ms/块。前轮同配置但 HC 关闭的历史结果是 184.279 ms/块；短输入未做本轮相邻控制进程，收益判断以本轮长输入对照为主。

## 算子与数值验证

AilaQwen4ExpKernelTests --hc-only 测试：真实 groups=4、dim=2560、rank=320 的 inject/mixer，以及 (2,256,45)、(3,31,35) 的尾部和非对齐情况。每种配置执行 rows=1/2/3/4/1，输入、输出、未用容量和 null-injection 哨兵通过，批量结果与逐行融合结果位级一致，live USM 数量不变，释放 execution resources 后 device/pinned-host USM 均为零。live 计数不检测瞬时 allocate/free；候选的无分配结论另由调用路径检查支持。

独立 double CPU 点积包含模型规定的 BF16 中间舍入，沿用已有 HC 测试 abs_tol=0.01，输出、activation 和 injection 均通过。相同输入同时验证 oneDNN：GPU matmul jit:gemm:any，src/dst bf16 ab、weights bf16 ba，静态维度、user scratchpad、默认 fpmath、无 matmul post-op；激活在后续 kernel 完成。日志保留 verbose 实现信息。两者在此误差契约下为 matched；未将单次 verbose 时延当作独立算子加速比。

[MEASURED] 相对于前轮同输入逐 token 全词表结果：

| HC 四行候选 | top-1 相同 | mean KL | max logits NRMSE | 127 个已知 target 的 mean ΔNLL |
|---|---:|---:|---:|---:|
| 30-token 输入 | 123/128 | 0.002563 | 0.081879 | +0.002609 |
| 512-token 输入 | 121/128 | 0.005515 | 0.168152 | -0.009781 |

长输入 max NRMSE 比上一候选的 0.113807 更大，而平均 KL 更低；不能用其中一个指标替代完整质量判断。各遍全 128 行有限且重复哈希一致，独立长输入复测一致。少量 top-1 变化不触发逐字一致门槛；上述固定小文本也不能证明质量无损或代表 MTP 接受率。默认化仍需要更广文本和状态语义验收。NLL 汇总现已纳入可复用脚本，通过 tokenizer 的 128 个输入 ID 哈希核对后，仅评分 logits0..126 对 ids1..127。

关闭新开关的单 token 路径另做两遍 128 步回归，全 128 行 logits 与前轮基线位级一致；本轮四行控制进程也与前轮控制结果一致。HC 开启的短输入两行 profile 与四行非 profile 输出一致。所有本轮算子/整模型采集退出 0、无 watchdog 停止；没有重复全量 CTest。构建与 git diff --check 通过。

本轮非 profiler 整模型进程峰值 working set **21.266 GiB**，系统最低可用 RAM **26.213 GiB**。未触发 8 GiB 可用内存保护；此处没有测量 DXGI resident VRAM，也未扩大 context 到 640 以上。

## 预算与下一步

沿用每四行块平均提交至多四个新 token 的预算模型，长输入验证本身约 189–190 ms，20 tok/s 的 200 ms 预算仅余 **约 10–11 ms** 给 draft、接受决策和状态处理；短输入余量约 26 ms。长输入即使额外成本为零也需平均约 3.78–3.79 个提交 token/块。这不是实际 MTP 吞吐预测。

下一步应量出 GPU MTP 单层 draft 的真实成本和候选接受情况，再决定验证路径还需削减多少时间。必须明确 pending token/bonus token 的计数，计入新增 MTP 权重对专家 cache 的占用；GDN、conv、PLE、KV 回滚采用有界 device state 方案。当前剩余 kernel 与传输都不可忽略，不能仅凭 26 ms 的短输入余量承诺 20 tok/s。原有跨路径首次数值分歧和更广质量验证仍待推进。

复现：

~~~powershell
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload hc -Tag hc-ops
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload decode -DecodePromptTokens 512 -DemandMicrobatch 32 -DecodeRepeats 5 -VerifyRows 4 -SmallRowsDecode -Tag hc-control
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload decode -DecodePromptTokens 512 -DemandMicrobatch 32 -DecodeRepeats 5 -VerifyRows 4 -SmallRowsDecode -SmallRowsHc -Tag hc-candidate
python perf/qwen4exp/q4e_decode_summary.py tmp/hc-control tmp/hc-candidate --teacher-tokenizer C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2/tokenizer.json --output tmp/hc-summary.json
~~~

全部身份、样本、误差、profile 归因和内存数据见 [结果 JSON](qwen3.8-flash-next-p4-hc-results-2026-09-30.json)。改动尚未提交，未修改生产默认。
