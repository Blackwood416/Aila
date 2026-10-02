# R2b：真实隐藏态、模型质量与状态验证

日期：2026-10-01。Intel Arc A770 / 驱动 32.0.101.9033 / oneAPI 2026.1、oneDNN 3.11.2。基线为 66e21e9 加此前未提交工作及本轮诊断改动；研究 host pool 32 GiB，专家计算仍在 GPU。

**[MEASURED] INT8 DPAS 候选通过一个 1833-token 自然文本输入的 48 步 teacher-forcing 质量门槛，以及独立的前缀因果、快照和恢复回放检查。原 NF4 逐点误差门槛仍未通过，不能把模型 canary 通过写成全面质量认证。**

生产路径保持不变。本轮诊断先计算 NF4，再以候选覆盖所有 prefill 专家的 rank 输出，因此没有可比较的端到端性能结果。此前 M32/M48/M64 的算子收益仅适用于 [有界 DPAS/graph fixture](qwen3.8-flash-next-r2b-dpas-results-2026-10-01.md)，本报告没有将它外推为真实 TPS。

逐 token 指标、源码/EXE/日志/语料哈希及各进程内存记录见 [结果 JSON](qwen3.8-flash-next-r2b-real-quality-results-2026-10-01.json)。

## 1. 数值策略与诊断实现

- NF4 payload 保留原格式；按单专家、固定 M32 的有界 staging 解码到 integer LUT，输入及 SwiGLU 后 BF16 激活分别作 per64 Q8。每个权重 block64 由两次 K32 的 s8×s8 DPAS 累加成整数，再按该块的 scale 转为 FP32。
- **所有 count>1 的 prefill 专家贡献采用同一数值策略**，包括小桶及尾桶；小桶补零到 M32。count=1 / decode 仍采用 NF4。没有按整窗口专家热度在 NF4 与整数格式之间切换，否则前缀可能因未来 token 改变路由计数而选择另一种算术。
- 这只是本次诊断的策略边界，不保证任意 prefill/decode 切分等价，也没有验证完整窗口与分段窗口之间的逐位等价。未来热/冷桶可以选择不同执行方式，但须先定义并验证一致的数值契约。
- 每个执行槽预分配输入、中间结果、INT8 staging、描述符及最多 2048 条 tile 参数。固定 graph 从 device 参数表读取指针、token/rank、有效行数；一次表上传后连续 replay。没有创建全模型 INT8 权重缓存。
- 游标在独立的最后一个有序 node 中递增，避免 scatter 的一个工作组先递增、其他工作组仍在读取旧记录的竞态。复用参数表前检查游标并等待完成。
- 后端增加默认为 null 的测试 hook，其返回的实际 barrier event 纳入专家槽 reader fences；正常推理没有候选替换。析构、描述符复用和参数表更新前均等待读者完成。

相关代码：[诊断实现](../../tests/models/Qwen4ExpIntegerQualityAcceptance.hpp)、[测试入口](../../tests/models/Qwen4ExpToleranceTests.cpp)、[后端 hook](../../src/models/Qwen4ExpBnb4Backend.cpp)、[运行器](../../perf/qwen4exp/q4e_execution_probe.ps1)。

## 2. 六组真实隐藏态：实现正确与近似误差分开检查

从原 NF4 路径的第 0、23、47 层各取两个实际热专家，每组 32 行，H=2560、I=640。保存原始 BF16 X、SwiGLU 中间激活、FP32 down 输出、真实 NF4 gate/up/down payload 及解码后的 per64 scales；每份 3,297,312 B，无进程指针，六份共约 18.87 MiB。格式、专家 ID 和 SHA256 收录于 JSON 的 corpus 字段。

[CORRECTNESS] 六组均通过：

1. Q8 字节与从 BF16 尾数/指数构造的整数 RTE oracle 一致，scale 符合 per64 absmax/127。
2. 每组第一/最后行的全部 gate/up/down 值与独立 FP64 block64 整数参考比较，abs=1e-6 + rel=1e-4 内零错误；最大绝对误差约 1.672e-6，发生在允许相对误差的非零输出上。
3. 使用原 LUT 和 BF16 激活的 NF4 消融对照与模型原始 down **数值完全一致**：六组 NRMSE=0、max_abs=0。
4. 实际 stream graph 的完整 32 行、部分 3 行，以及连续两条参数记录（3 行后 32 行）均与独立 replay 输出逐位一致；游标计数正确。检查后恢复 NF4 输出，模型再消费这些数据。

下表为单项消融的 NRMSE 百分比；各列改变的量不同，不能相加。整数 LUT 列不量化激活，X Q8 列只量化输入，h Q8 列只量化 SwiGLU 后的激活。

| 层/专家 | 仅整数 LUT | 仅 X Q8 | 仅 h Q8 | 完整 DPAS | 超出原逐点阈值 / 81,920 |
|---|---:|---:|---:|---:|---:|
| 0 / 175 | 0.912% | 0.603% | 1.385% | 1.751% | 15,485 |
| 0 / 348 | 0.928% | 0.570% | 1.439% | 1.796% | 17,694 |
| 23 / 107 | 1.101% | 0.855% | 1.061% | 1.743% | 23,400 |
| 23 / 83 | 1.065% | 0.803% | 1.152% | 1.730% | 25,454 |
| 47 / 186 | 0.956% | 0.608% | 1.260% | 1.633% | 36,995 |
| 47 / 157 | 1.146% | 0.824% | 1.533% | 2.047% | 38,425 |

**原逐点阈值 abs=0.0002 + rel=0.02 未放宽，真实样本中超限约 18.9%–46.9%。** 不能沿用合成数据中“少量超限”的描述。最大绝对近似误差约 0.09971，发生在 47/186；该样本 h RMS≈0.89585、max=65.5，存在明显离群值。完整 DPAS 与相同近似的 float ALU 消融结果非常接近，具体误差保留于 JSON。

[HYPOTHESIS] down 输入采用 per32 激活尺度、权重仍保留 per64 scale，可能降低中间激活离群值的影响；现有 K32 DPAS 可以表达，但会增加 FP32 缩放工作。本轮没有实现或测量该变体，不能宣称收益。

## 3. 真实模型质量 canary

使用 archive-natural.json，经与 CLI 一致的换行展平及 chat template 得到 1833 tokens；max-seq=2560，prefill chunk=2048，实际按 1829+4 执行。随后对固定英文续写做 48 步 teacher forcing。

三个模式依次为默认 NF4、启用观察 hook 的 NF4、所有 prefill 专家贡献被整数候选覆盖。第二模式与默认 NF4 的 48 步 logits **全部逐位一致**，因此诊断采样及 scatter 恢复没有污染模型。

测试前固定门槛：平均 KL≤0.01、平均 NLL 差≤0.01、最大 logits NRMSE≤0.15。

| 指标 | 最终候选 |
|---|---:|
| 平均 KL（NF4 概率分布为参考） | 0.007011584149 |
| 平均 NLL 差（candidate − NF4） | −0.001149794072 |
| 最大 logits NRMSE | 0.07913760271 |
| 首选 token 一致 | 42 / 48 |
| 门槛结果 | 通过 |

最大单步 KL≈0.040309；有 6 步首选 token 不同，不能写成生成逐字一致。平均 NLL 略低只代表这段固定续写，不代表模型质量改善。最终复测与先前成功 stream 运行的上述指标相同。

候选覆盖 **879,840 / 879,840** 个 prefill 专家贡献（1833×48×10），39,242 个补齐 tile、1,255,744 个实际计算行；其中热桶≥32行的有效贡献为 793,858。不是只替换易通过的热专家。

最终候选 prefill 71,302.59 ms 包含原 NF4 重复计算、诊断同步、参数上传及补齐开销，日志明确 performance_comparable=0。不得与默认路径的时长相除形成 speedup。尚未运行自由生成、多语言/代码质量集或实际 decode TPS 测量。

## 4. 前缀因果、快照与回放

将原先过长的组合测试拆成两个独立进程，均使用同一最终 EXE 和全部 prefill 专家整数策略。

| 检查 | 输入与变化 | 结果 |
|---|---|---|
| 整窗口前缀因果 | 257 tokens，只改变后 128 tokens | 前 129 tokens 的最终 hidden4 及全注意力 K/V 逐位一致 |
| 前缀快照 | 33+32 tokens，两次使用不同后缀 | 第 33 token 的 PLE、递推状态、卷积历史、压缩块计数及前缀 K/V 一致 |
| 截断恢复 | 截断至 33，再重放原 32-token 后缀，共两次 | 两次 logits 均与原始续写逐位一致，前缀 K/V 不变 |
| 参数生命周期 | 每批重放后验证 device cursor | 已提交 tile 数全部消费，复用前完成等待 |

每次 65-token 分段 prefill 检查 31,200 个替换贡献；每次 32-token 回放检查 15,360 个。因果测试每次覆盖 123,360 个。未将快照/因果错误容忍为量化误差。

这不是所有 session/cancel/cache-eviction 调度组合的验收；未来生产集成仍需多会话、取消、单 token 边界和新的热/冷执行切换检查。

## 5. 超时、内存及可复现性

保留两次失败过程，不把部分完成当成整项通过：

- r2b-real-quality-first-20261001：每个 tile 三次阻塞 host 参数拷贝的首版触发 240 秒 watchdog；采样及 NF4 对照完成，候选模型阶段未完成。最低可用 RAM≈10.85 GiB。
- r2b-integer-state-20261001：原组合状态测试触发 240 秒 watchdog；257-token 因果子项已通过，后续 checkpoint 子项未完成。最低可用 RAM≈11.60 GiB。

这两次均未触发 8 GiB 内存保护，日志没有设备丢失错误。修复方式是参数表流式重放和拆分测试，没有提高超时或内存保护阈值。

最终三个独立进程均 exit=0、无 watchdog/内存保护触发：

| 最终运行 | 最后内存采样时间* | 峰值进程工作集 | 最低系统可用 RAM |
|---|---:|---:|---:|
| checkpoint | 210.570 s | 37.257 GiB | 11.717 GiB |
| causal | 164.175 s | 37.177 GiB | 11.562 GiB |
| quality | 203.139 s | 37.409 GiB | 10.991 GiB |

*运行器记录的最后一次内存采样时刻，不是精确总耗时或性能基准。host pool 预算是 32 GiB，不能把它解释为整个进程 RAM 上限；本轮没有新的 RAM 占用优化。GPU USM/显存峰值也没有由这些工作集数字测得。

最终三个进程的 EXE SHA256 均为 F6DA2BA2AFBD453D254F6A7A43D3E6FD632DF07A0F309DB278EA5A09C8F15872；最终诊断头文件 SHA256 为 44473C0299CAFDFD6E013F367B45B3E53B4963E41D81AB4553FA2F5C601639C5。先前成功 stream 运行的 EXE 为 31B4D5B0FFCDA672E1C4B26E1E8781E25AF4FB0E4C38E0CFB44A5FB5F5AF34FB，不能冒充包含最新两条记录检查的版本；各版本单独记入 JSON。

本轮全部运行于 9033，没有新的旧驱动 A/B；不能从这些诊断时长确认或否定之前约 3% TTFT 回退信号。

构建 Aila、AilaQwen4ExpToleranceTests、AilaQwen4ExpKernelTests 成功；相关 CTest 3/3 通过：AilaQwen4ExpKernelTests、AilaQwen4ExpToleranceTests、AilaQwen4ExpDpasSmoke。原始输出为 tmp/r2b-quality-ctest-final-20261001.log。

复现命令（顺序运行，一个 GPU 压力进程）：

~~~powershell
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload integer_state -IntegerStatePart checkpoint -Tag r2b-state-checkpoint-final-20261001
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload integer_state -IntegerStatePart causal -Tag r2b-state-causal-final-20261001
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload integer_quality -Tag r2b-real-quality-final-20261001
~~~

这些 tag 对应已有证据；再次运行应使用新 tag，避免覆盖原始记录。

## 6. 下一步

R2b 的真实激活/单输入模型 canary/本报告状态子项完成，生产集成未完成。

1. 利用已保存的六份真实语料，独立测试 down per32 激活尺度的误差与完整链成本；不靠放宽原逐点门槛消除失败记录。
2. 明确 count=1、prefill、decode 和热/冷桶的数值契约，验证 DPAS 与小桶 DP4A 的 block64 整数和、FP32 缩放/累加次序。热度只决定执行方式，不能决定是否量化。
3. 再去除诊断中的重复 NF4，按多专家批次组织有界 staging 与 graph，避免大量小桶 M32 padding。量化/转码/gather/scatter/H2D 均计入真实成本。
4. 补多输入质量、cache/cancel/session 验收后，测实际 Prefill / decode TPS。decode 20 tok/s 目标尚未达到；R3-head 仍是可独立推进的下一工作包。
