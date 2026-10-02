# Hotset C1/H3：冷计算分工与 prefill 直接执行

日期：2026-10-02；基于 de5f943，A770 / 32.0.101.9033 / oneAPI 2026.1、oneDNN 3.11.2，32 GiB host pool。**C1/H3 首轮实现和对照完成，候选继续默认关闭，R2b 继续暂停。** CPU 冷计算在相同热集下有小幅收益，但当前冻结集合损害 decode 命中；H3 的 acquire 减少没有转化为稳定吞吐收益。没有达到通用 prefill 100+ / decode 20 tok/s。

原始样本、模型配置/EXE/源码身份、排除的实验、内存及回归证据见 [结果 JSON](qwen3.8-flash-next-hotset-c1-h3-results-2026-10-02.json)。后续优先 H4/C2 的热集选择、跨域验证和有界 epoch 调整。

## 实现

C1 增加 AILA_Q4E_COLD_MODE=1/2：相同冻结热集下，1 为 cold GPU staging，2 为有界 host-Ready CPU cold 加其余 GPU staging。原动态模式 0 作为第三对照。冻结从当前 Ready/MRU 内容取得，保留 512 个 staging 槽；冷 GPU 项读完后撤销逻辑映射但保留 DMA/consumer fence，不建立下一 token 的命中。冻结热槽不能被冷供给淘汰。固定 descriptor/index table 发布一次后复用。

GPU 热 kernel 在 host route wait 之前入队。CPU 输入回传在热 kernel 之前入队；复用既有常驻 worker、prepared AVX2、host lease 和结果缓冲。单 worker 批次、无任务积压；初版按批任务数维护从 submit 到 compute-complete 的 EWMA，叠加任务上限。实验配置为 4 个 CPU 线程、最多 2 项、1000 µs 完成预算；这些是候选参数，不是已验证的默认 dispatch。

AILA_Q4E_COLD_LOW_RANK=1 在这个预算内优先选择低路由权重的冷项，保留原 rank 输出与一次权重乘法，不省略任何专家。该选项是针对初始数值分歧的单独实验，原 rank-first 候选仍可复现。

H3 增加 AILA_Q4E_RESIDENT_PREFILL=1：在 B6/expert-major 的 Ready 边界拆批，Ready 批 BindReady 后直接计算，只有冷子集进入 acquire。复用现有 pair/quad、grouped 和 R1b 有界 scratch/merge。冻结热集跨 prefill 保留，冷 staging 不 promotion；其容量机会成本及完整 prefill 时间必须纳入结果。没有借用热槽做 scratch，也没有隐藏的热集重建。动态模式下 H3 亦可独立开启。

初始化 warmup 显式禁止冻结，避免把 2-token dummy 路由当成实际热集。真实请求首个 decode 才允许自动冻结。对照 harness 则在 prompt 校准之后显式冻结，校准耗时单列；A/B 共享同一模型和热集 identity。

## 数值与诊断记录

初始错误冻结发生在 warmup，只留下 646 个热槽。该运行记录在 hotset-c1-h3-short-quality-v2-20261002，不参与性能判断。修正后，短输入质量对照冻结 3055 个槽，GPU-only cold 对原动态路径的 48 步 logits 逐位一致。

初版对短故事使用了 archive 续写作为 teacher targets。rank-first CPU cold 的平均 KL=0.010064、NLL 增量=0.045178、最大 logits NRMSE=0.12450；低权重优先为 0.003701、0.017182、0.063003。它们是保留的负例/预警，不能写成完整质量通过。最终短/长验收改用 H1/H2 原有相应续写文本，并单独报告。

真实 CPU 专家 spot check 的两个样本，输入、BF16 h 和 FP32 down 均与对应 GPU 专家逐位一致。进一步逐层追踪定位到第 5 个续写 token、第 12 层：x 一致，只有 rank1 的 down 不同，相对 L2=0.000265；合并 y 的相对 L2=0.000634，其余九个 rank 逐位一致。这符合局部算术差异逐步放大的证据，未观察到贡献丢失/重复，不能据两个样本宣布所有 CPU 算术精确等价。

原始容差计划将 +0.01 nat/token NLL 定义为待校准的预警线。验收同时记录 KL、NLL、NRMSE、checkpoint 重放、取消恢复和 USM 数量；NLL 预警必须显式保留，不因为 greedy 输出相同就清除。候选仍默认关闭。

使用 H1/H2 原有续写样本的最终验收：每个配置48步 teacher forcing，24步处 checkpoint、后缀重复回放、实际 CPU result boundary 取消及 reset 恢复均通过，decode live USM 数量不增加。

| 输入 | 配置 | 原路径逐位一致行数 | 平均 KL | 平均 NLL 增量 | 最大 logits NRMSE | CPU 冷任务 |
|---|---|---:|---:|---:|---:|---:|
|41|冻结 GPU，H3 off/on|48/48|舍入噪声|0|0|0|
|41|CPU 低权重优先，H3 off|7/48|0.0012511|−0.0009977|0.051034|2243|
|41|CPU 低权重优先，H3 on|7/48|0.0012511|−0.0009977|0.051034|2243|
|1833|冻结 GPU，H3 off/on|48/48|舍入噪声|0|0|0|
|1833|CPU 低权重优先，H3 off|5/48|0.0021507|−0.0025140|0.048477|2107|
|1833|CPU 低权重优先，H3 on|5/48|0.0021880|−0.0037621|0.048477|2104|

这两个相应续写样本没有触发 NLL 预警，不替代跨域质量评估，也不抹去前述不相关 teacher continuation 的预警。CPU 回放按同样 KL/NRMSE 检查，GPU 回放要求逐位一致；保存的 checkpoint 内容在两次回放之间逐位保持不变。

短/长 H3 质量进程分别直接绑定6110、23478个 prefill 唯一专家项（按窗口、层、batch计数，不是 token-frequency）。小型 cache fixture 已通过冻结/非 promotion/读者 fence/容量超限拒绝/解冻；Python replay 7/7、相关 CTest 6/6 通过（66.37秒）。CTest 为 KernelTests、ModuleTests、GenerationTests、ToleranceTests、DpasSmoke、ExpertReadinessTests；本轮没有扩大到服务端 HTTP/多会话验收。

一次长质量进程因测试筛选器用正整数读取函数解析 H3=0，被误当作“跑全部组合”，在额外组合期间触发240秒 watchdog。改为 raw integer 读取并分成 H3 off/on 两个独立进程后都完整通过。该超时不作为驱动回退或性能结论。

## 性能结果

所有性能样本 max-seq=2560、chunk=512、相同3567个 GPU expert 槽和32 GiB host pool，preset=qwen38-a770。冻结3055个槽，余512个作为 staging。短输入也用2560容量，以便与对照匹配。16个性能进程：10个同模型冻结 A/B 进程、6个原动态/独立 H3 的 CLI 进程。

冻结 A/B 在每个进程中共享同一个初始集合，顺序为 GPU→CPU→CPU→GPU 或反序；每请求实际生成48 tokens。每模式的首请求单列，后一个为暖样本。短每组合2个暖样本；长 H3 off 因一条异常增加了平衡复测，每模式4个暖样本，H3 on 每模式2个。CLI 每进程3请求，首个单列，每配置共4个暖样本。每个样本及异常均保留，没有按结果删样本。

冻结基准在计时前做一次相同 prompt 的校准：短11.37–11.70秒，长58.88–65.96秒（均不含模型加载）。**这是有校准成本、重复相同 prompt 的暖态实验，不是冷启动或留出语料的提速。** 质量基准还保留了 reference 的缓存历史，其时间完全不用于 TPS。

decode 按“(生成数−1)/(生成调用结束−首 token)”汇总；冻结 harness 还 drain 末尾 GPU 工作。JSON 另留最后文本 callback 的时间，初期 callback-only 数字不作为最终 TPS。冻结组“完整请求”包含最终状态对齐；CLI 的 GenerateMessages 日志时长截止于对齐前，JSON 区分两种口径，不混算完整请求百分比。

| 输入 | 配置 | 暖 TTFT 秒 | 实际 decode tok/s | 冻结组完整请求秒 |
|---|---|---:|---:|---:|
|41|原动态 GPU|1.558|13.90|—|
|41|冻结 GPU，H3 off|1.071|7.57|7.283|
|41|冻结 CPU cold，H3 off|1.087|7.86|7.072|
|41|冻结 GPU，H3 on|1.096|7.60|7.279|
|41|冻结 CPU cold，H3 on|1.097|7.89|7.058|
|1833|原动态 GPU|25.879|8.75|—|
|1833|原动态 GPU + H3|25.923|8.55|—|
|1833|冻结 GPU，H3 off|20.369|7.45|26.695|
|1833|冻结 CPU cold，H3 off|20.307|7.63|26.483|
|1833|冻结 GPU，H3 on|20.524|7.42|26.858|
|1833|冻结 CPU cold，H3 on|20.541|7.70|26.642|

同冻结集合、H3 off：CPU 分支短 decode 中位数约 +3.9%、完整请求 −2.9%；长约 +2.5%、完整请求 −0.8%。样本有限，长完整请求差异小于部分 prefill 波动，不能称稳定普适收益。长 CPU 四个暖 decode 样本为7.62、3.61、7.65、7.64 tok/s；3.61的异常没有删除，后两次平衡复测未复现。该异常的 CPU compute/任务数/需求字节没有相应倍增，原因未定位。

短 CPU 每请求约2289项、长约2068项，确实进入冷计算；不是空跑或全部 admission 拒绝。短整请求需求权重约41.43→35.76 GB，长128.67→123.40 GB。CPU FP32 回传短约23.4 MB/请求；输入按层一次回传。生成文本在 CPU/GPU 两组间可能不同，质量使用独立 teacher-forced 验收，字节差异不全部归因于放置选择。

代价也明确：当前集合按 prompt 后 Ready/MRU 冻结，暖 decode 直接贡献短约9355/23040、长11159/23040；远低于原动态路径的命中覆盖。短原动态 decode 需求约8.84 GB/请求，而冻结 GPU 的 decode 需求显著更大。冻结集合保护了重复 prompt 的 prefill 内容，长期阻止新 decode 热项进入，导致 prefill 更快、decode 更慢的权衡。暖长输入按 tokens/TTFT 约90 tok/s，不能外推为100+；本轮1833-token输入也未覆盖更长的 QSA 阶段曲线。

H3 本身：同冻结 GPU 集合时，短/长分别减少98/385次 acquire/请求，需求字节完全一致，但 TTFT/完整请求没有净改善。原动态 H3 的长 TTFT 中位数25.923秒与原路径25.879秒接近，其中一条暖样本为36.775秒；其需求字节与其余样本基本相同，不能用这条异常推导固定的 H3 开销，也没有证据据此开启默认。

## Trace、内存与测量边界

所有 GPU 过程串行执行，240秒 watchdog、系统可用 RAM 8 GiB 下限。正常性能运行不启用 trace/device profiling。16个性能、3个最终质量及普通 trace 的模型进程均 exit=0，正常实验最低可用 RAM 约10.76 GiB，峰值进程工作集约37.52 GiB；32 GiB 是 host pool，不是整个进程的工作集。本轮没有新的 RAM 占用优化声明。

普通 trace 完整回放通过：36次 forward、1510个 CPU 冷任务，验证全路由 = GPU direct + CPU cold + GPU cold，冻结槽不被淘汰、临时槽不 promotion、原 rank 归属正确。它验证提交顺序与主机区间，不提供 GPU 实际重叠比例。

新增的整模型 GPU event profiling 配置未完成验收：warmup 期间约50.7秒，系统可用 RAM 从约15 GiB突然降到1.13 GiB，触发内存保护终止；这不是240秒超时。被测进程工作集没有同幅增加，具体分配来源未定位。终止后可用 RAM 恢复到约55 GiB。没有继续在32 GiB完整模型上堆叠 profiling 探测，设备时间记录为未取得。该配置仍仅为显式 opt-in 的诊断原型，默认关闭；后续应先用小型已捕获专家 fixture 验证设备计时与内存行为。

本轮全部正常对照在9033上，没有旧驱动 A/B；吞吐异常和 profiling 内存事件不能归因于驱动版本。普通 trace 的推理进程成功后，脚本末尾曾因缺省 memory_pressure 字段报错，该 guard 已改为先检查字段存在；没有把包装器异常当作推理失败或丢弃 trace。

运行时 EXE SHA256：Aila F491E2F8D43766F93EACAD10F10030FBBBC07C24D65E79A5C41E046212336907；冻结 benchmark/最终长质量 EXE F4F2B2BBEC9054E4B824D56C78EA89E70C35DA253688FEADC7C0445B95925AE4。短质量使用前一版测试 EXE，区别为测试组合筛选器修复，运行时实现相同；各次身份完整保存在 JSON。

## 下一步

按计划进入 H4/C2：用独立历史路由建立带模型身份的热表，并在留出的短/长、多领域输入上评估；给 prefill 与 decode 明确的热集交接/有界 epoch 更新，避免本轮固定集合长期拒绝 decode 新热项。之后再调整 CPU 完成预算、被初次慢样本抑制的任务数桶及 compute/admission 解耦。当前4线程/2项/1000 µs配置是首轮候选，不是CPU能力上限。

R2b 暂不恢复。先解决热集覆盖与动态 admission 的机会成本，再决定是否将这些候选纳入 preset；不把 CPU 冷路径的小幅收益解释成已达到20 tok/s，也不将旧 pinning 的负例或本轮固定集合的回退泛化为异构架构无效。

相关入口：perf/qwen4exp/q4e_execution_probe.ps1 的 ColdQuality / ColdBench / ResidentPrefill / ColdLowRank；结果汇总 q4e_cold_summary.py，缓存/冷贡献回放 q4e_cache_replay.py。
