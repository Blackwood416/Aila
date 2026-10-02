# H4/C2：迁移 Strata 热集自适应与冷专家分流

基线：5534a02；A770、32.0.101.9033、oneAPI 2026.1、32 GiB host pool。**H4/C2 首轮策略迁移和验收完成，所有候选仍需显式开启。** R2b 保持暂停。原始数据、EXE/源码身份和所有样本见[结果 JSON](qwen3.8-flash-next-hotset-h4-c2-results-2026-10-02.json)。

本轮将 Strata 的 profile 预装、分层衰减 usage、迟滞换入、异步发布、prefill 借槽以及 CPU/PCIe 冷项比例分流移植到 NF4 管线。C1 的 prompt/MRU 一次性冻结不再代表本轮候选。所有专家贡献仍按原 rank 合并，不截断 top-k、不改变 router 权重。

## 实现与 Strata 对应关系

对照源码版本为 tmp/Strata 的 a79080535d1b2a71a3419a0d97d8e7dca194b0f1。
迁移策略保留 [Strata MIT 许可与版权声明](../third-party/Strata-LICENSE.txt)。

| Strata 机制 | 本轮实现 |
|---|---|
| generate.cpp profile fill | 启动按 Aila 自身 profile 排名预装；加载时验证维度、完整唯一 ID 集合、NF4 布局与 source fingerprint |
| generate.cpp adaptive refill | 同层冷候选 usage≥2；候选与最冷驻留项配对，gain≥1.5；全局按收益取最多96项；每4个 decode forward 调整，usage乘0.7 |
| retire / pending / apply_pending | 旧读者 fence 完成后撤销旧映射、异步 H2D；持有 host lease；只在后续 forward 边界确认 DMA 完成后发布新映射和 device tables |
| prefill borrowing | decode 3551热槽+16冷槽；prefill 暂借初始profile顺序末尾496槽（换入项继承原位置），保留3055热槽，形成512个服务槽；归还时先保护所有幸存目标，只补齐丢失目标 |
| resident GPU + CPU misses | H2 device route 和热 kernel 提前提交；冷 CPU 复用常驻 worker / prepared AVX2；GPU 剩余冷项提交后才 join；CPU 算过的项也按相同 usage 参与 admission |
| expert_source.cpp PCIe fraction | policy=3 将路由顺序末尾 floor(nmiss×141/256) 项送 GPU，前部 host-Ready 冷项送 CPU；141是 native pack默认0.55的离散值，作为候选而非已定 dispatch |

Strata rounds 可能包含多个 speculative tokens；这里的 interval 是单 token decode forward，不能把两个“4”当作相同时间跨度。prefill 使用已有 R1b scratch 和 NF4 expert-major 执行，借槽用于专家供给容量，没有声称实现 CUDA scratch alias 或完整 token graph。

CPU policy=1 测试全部 host-Ready 冷项（12线程、最多10项），作为冷分工的直接对照；未 Ready 的 host 项仍走 GPU 供给。policy=0保留C1完成预算；policy=2增加 compute EWMA、较大任务数桶的恢复与每32批有界重试，避免初次慢样本永久关闭某个桶。CPU 计算位置与未来 GPU 驻留是独立决策。

所有 epoch 变化发生在模型的串行执行边界，checkpoint不保存热集。取消恢复允许使用新的有效 epoch。换入失败会排空 DMA、释放 host leases 并标记缓存故障，不能带着半发布映射继续执行。

## Profile 来源与验证范围

可复现 profile 位于 perf/qwen4exp/profiles/qwen38-nf4-h4-v1.profile，SHA256为3f0994bc4356675ee23d6c8e3557bdaa9afff618c98b484f3dde993c1ee9a77c；相邻JSON保存语料及模型配置身份。

校准输入为独立的 C++ 队列、整数规划和社区花园三类问题，各重复2次、每次生成48 tokens，共138240个 decode路由贡献和220800个prefill贡献。分别保留token-frequency与每层每forward的unique-demand，排名按decode频次、prefill频次、原expert ID排序。merge工具会重新计算已有项的排名，不沿用Strata make_profile默认base不重排的行为。

故事和1833-token档案长文未参与profile校准；在线只观察已经执行的实际路由。profile不包含它们未来的生成路由。3055项在校准decode上覆盖68.14%，在上轮独立故事trace上约19.24%，说明初始表的跨域覆盖仍有限。在线适应成本必须作为结果的一部分。
最终3551项分别覆盖73.11%和21.86%；仍需更广的校准语料，不能把同一prompt反复运行后的热集覆盖当作跨域初始覆盖。

identity guard覆盖几何、nested NF4元数据及每个投影source首尾采样FNV64；这是兼容性检查，不是完整checkpoint的密码学哈希，也不能保证检测所有中间expert重排。probe另外记录model config SHA256、EXE SHA256、profile SHA256和原始日志身份。显式给错profile会报错，避免实验静默回退。

采集元数据的旧input.sha256字段曾继承未使用的默认story路径；实际stdin已逐行核对，只包含声明的6个校准请求，SHA也与原始记录一致。sidecar保存这项说明，probe已修正多输入模式的字段。原始实验元数据没有被改写。

## 测量口径

固定 max-seq=2560、chunk=512、总3567 expert slots、32 GiB host pool，每个性能请求实际生成48 tokens。GPU过程串行，240秒watchdog、8 GiB系统可用RAM下限；本轮不启用曾失败的整模型GPU event profiling。

CLI TTFT包含实际prefill。当前热槽归还在第一个单行decode入口执行，因此归还成本计入decode及整请求，尚未移到首token回调之前；不得把这段成本从总时延中扣除。profile初装在model load阶段，3551项最多8,999,654,400 B；每次prefill借出496槽，全部丢失时恢复最多1,257,062,400 B。需求H2D与epoch refill分别记账，二者相加才是专家总传输量。

不把第二次请求自动视为策略已经收敛。保留每次请求的时间、路由直接覆盖、CPU任务、epoch swaps和refill bytes，区分缓存变暖与热集适应。不同CPU/GPU算术会导致生成文本变化；质量用独立teacher forcing验证。

## 已完成的数值验收

每个配置48步teacher forcing，在第24步checkpoint，两次后缀重放；真实CPU结果边界取消及reset恢复通过，decode live USM数量不增加。

| 输入 | 冷计算 | 与原路径逐位一致行数 | 平均KL | 平均NLL增量 | 最大logits NRMSE | CPU任务 |
|---|---|---:|---:|---:|---:|---:|
|41|GPU|48/48|舍入噪声|0|0|0|
|41|全部host-Ready CPU，12线程|1/48|0.00187531|−0.0119851|0.0563271|9607|
|1833|GPU|48/48|舍入噪声|0|0|0|
|1833|全部host-Ready CPU，12线程，最终版本|2/48|0.00232274|−0.00226909|0.0423677|8506|
|41|Strata比例分流，12线程|1/48|0.00171303|−0.0130560|0.0563271|5351|

两种CPU样本均未触发原计划的NLL预警；这是代表性canary，不等于大语料质量认证。GPU重放要求逐位一致，CPU重放按既定KL/NRMSE检查；checkpoint内容在重放之间不变。模型结果见tmp/hotset-h4-c2-{short,long}-quality-20261002。
最终冷区LRU与维护统计修正后的长文复验见tmp/hotset-h4-c2-final-long-quality-20261002；质量指标与此前一致，实际CPU任务数随host readiness略有差别。

## 短输入：必须保留适应曲线

第一组按动态GPU→热集GPU→CPU→CPU→热集GPU→动态GPU的反向顺序复测，每进程3次独立请求，48个实际生成tokens。

| 配置 | 两进程第2次decode tok/s | 两进程第3次decode tok/s | 第2/3次合并中位数 | 第3次整请求秒 |
|---|---|---|---:|---:|
|动态GPU|13.5、13.6|14.2、14.2|13.90|4.802、4.804|
|热集GPU|12.6、12.6|13.7、13.7|13.15|4.860、4.851|
|热集CPU，12线程|10.8、10.5|15.0、14.9|12.85|4.567、4.594|

CPU两个进程的epoch swaps都为第1次1075、第2次400、第3次8。第3次直接贡献达到20103/23040（87.25%），CPU冷贡献2933项；GPU需求权重接近清零，但仍有约1.277 GB的epoch恢复/换入流量，不能只报告demand=0。第3次速度改善不应掩盖适应期：第2/3次混合的中位数还没有击败动态GPU。

比例分流初测4个连续请求，第2/3/4次为13.4、12.3、15.1 tok/s；后续仍需依据完整曲线判断稳定性，不能取最后一点代替所有暖样本。

## 独立冷槽修正

第一轮自适应迁移仍让GPU冷staging沿原LRU选槽，虽然冻结项不能被淘汰，搜索却可能反复经过数千个热槽。这是实现遗留，不能用这种开销否定热集架构。

后续增加独立冷槽ring：只在decode的16槽或prefill的512槽中寻找未被本批保留、没有direct reader的槽。payload覆盖前仍等待consumer fence，DMA与nested metadata规则不变。epoch初装/恢复继续走有保护的装载流程；普通动态缓存和C1冻结对照不改。AILA_Q4E_HOTSET_RING=0保留相同二进制中的旧搜索以便对照。trace replay同时验证ring顺序、保护集合和原rank归属。

FIFO原型改变了冷槽重用顺序，长文同二进制对照的暖TTFT中位数从20.289升至29.603秒，没有获得稳定收益。该原型保留为AILA_Q4E_HOTSET_COLD_QUEUE=1。不能据此将延迟变化全部归因于队列顺序，但它没有隔离“省掉热槽扫描”这个变量。

最终采用冷区独立LRU（COLD_QUEUE=2，profile候选内默认），同步原访问的touch/demote，保留原cost-aware选择；0为原完整LRU。实际trace逐次验证了victim与原完整LRU计算一致，20次forward、3427个CPU冷任务、384次retire/publish全部闭合，包含prefill借槽归还。该trace也实际执行了policy=2的可恢复预算分支，未出现空跑或永久拒绝。

同一个最终算法二进制的长文对照，冷区LRU与完整LRU的decode中位数均为11.25 tok/s，整请求24.319与24.179秒；**没有证据称冷区LRU本身带来净提速**。保留它的理由是有界的冷服务与等价的victim顺序。主要端到端改善来自整个profile/admission/冷计算组合，而不是这次队列改动。

维护装载的H2D计入epoch_refill_bytes；初装/恢复的hit、miss、acquire不再混进真实路由命中率，避免影响90%命中阈值的预测预取判断。该边界有小fixture断言；最后复验验证质量和状态不受影响。

## 最终性能与边界

共保存19个性能进程、61个完整请求，另有校准、4个质量进程和2个普通trace。JSON按EXE身份分组，不将不同实现的样本隐式合并。以下动态基线来自本轮新跑的相同预算/容量CLI对照，版本差异和每个样本都在JSON中保留。

| 1833-token长文，进程内第2/3次请求 | 动态GPU | 最终profile + CPU冷计算 |
|---|---:|---:|
|decode tok/s|8.6 / 9.0|10.8 / 11.8|
|decode中位数|8.8|11.3（约+28.4%）|
|TTFT中位数|24.476秒|19.994秒|
|整请求中位数|29.691秒|24.075秒（约−18.9%）|
|有效输入tokens / TTFT|约74.9 tok/s|约91.7 tok/s|
|decode需求权重中位数|14.959 GB|1.076 GB，另计epoch流量|

此前独立CPU长文复测第2/3/4次decode为10.5/11.7/12.1 tok/s，整请求中位数24.106秒；最终冷区LRU的另一进程第2/3次为10.8/11.7，整请求中位数24.319秒。支持“适应后长文获得收益”，仍不等于达到20 tok/s或所有输入都获益。

最终短输入第2/3/4次decode为11.1/14.9/15.6 tok/s，整请求5.562/4.568/4.434秒。动态对照第2次13.5/13.6，第3次14.2/14.2；因此不能把15.6当作从第二次请求起就稳定可得的收益。策略尚有适应成本。

异常保留：早期CPU长文有31.276秒的暖prefill，后续独立复测未持续出现；比例分流长文曾有4.2 tok/s样本，另一暖样本10.8。没有删掉它们或用单点宣告比例分流无效。FIFO长文的回退也保留为独立实现。所有这些运行均正常退出，原因没有被归因于9033；本轮没有旧驱动A/B。

冷启动/适应开销没有消失。最终长文首请求TTFT50.304秒、整请求56.217秒；动态同轮首请求为64.178/70.419秒，但profile预装额外发生在model load，不能用两行请求时间直接宣称冷启动收益。

正常性能进程峰值工作集37.29 GiB、最低系统可用RAM12.64 GiB；包含质量进程后峰值37.52 GiB、最低可用12.30 GiB。没有触发RAM保护或watchdog，没有新增RAM优化声明。19个性能进程、4个质量进程均exit=0。仍沿用32 GiB host pool，不是把整个进程限定在32 GiB。

6项相关CTest全部通过（66.83秒），维护统计修正后补跑ExpertReadiness通过（6.13秒）；Python replay 8项、profile工具2项通过。Aila、AilaWorker、AilaShared和AilaInference已构建。没有声称完成HTTP多会话/MTP验收；当前自适应针对单行decode，多行验证仍需明确的阶段标识和交接策略。

## 使用与后续

实验配置示例（相同预算，全部host-Ready冷项交给CPU）：

```powershell
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 `
  -Workload generation -PerformancePreset qwen38-a770 `
  -GenerationMaxSeq 2560 -PrefillChunk 512 -GenerationRepeats 3 -MaxTokens 48 `
  -HotsetProfile perf/qwen4exp/profiles/qwen38-nf4-h4-v1.profile `
  -HotsetBorrow -HotsetColdQueue 2 -ResidentPrefill `
  -ColdMode 2 -ColdPolicy 1 -ColdCredits 10 -ColdThreads 12 -WorkSummary `
  -Tag hotset-user-check
```

ColdPolicy=3、ColdPcieNum=141可复现Strata比例分流；ColdPolicy=2启用可恢复预算。候选没有写入默认preset。初版profile只有三类校准输入，优先补广泛语料与跨域/多轮验证，减少适应开销；随后把稳定descriptor用于更完整的GPU提交路径和R2b接入。当前尚未实现通用100+ prefill / 20 decode，也未覆盖更长的QSA上下文曲线。这里的收益和限制属于当前迁移实现，不能外推为对Strata架构本身的否定。
