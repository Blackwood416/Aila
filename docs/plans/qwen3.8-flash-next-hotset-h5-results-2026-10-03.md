# H5：由当前 prompt 驱动热集，并重新定位供给瓶颈

基线 126d2de，A770 / 32.0.101.9033 / oneAPI 2026.1，32 GiB host pool。
**已实现并验证 prompt 热度、交接和 chunk 间更新候选；尚无稳定端到端净收益，候选未进入默认 preset。** R2b 继续暂停。原始逐请求数据、排除原因、EXE 身份、质量和时间线见[结果 JSON](qwen3.8-flash-next-hotset-h5-results-2026-10-03.json)。

本轮没有将这些结果外推为 Strata 架构无效。它们暴露了当前迁移实现中“观察路由”和“廉价取得对应物理驻留”之间的缺口。

## 重新确认的瓶颈

- [CODE] H4 的 prefill 路由只进入累计导出统计，没有进入在线 usage；decode 开始时恢复旧逻辑热表，再等待单 token 路由学习。
- [MEASURED] 在独立故事 trace 上，只看已经完成的 prefill、按层原容量选表，前 8 个 decode token 的静态覆盖从约 22% 提高到约 55%；但需要替换约 2643 个 profile 专家。这是因果覆盖回放，不是延迟预测。校准语料上的离线 profile 本来就更好，不能据此主张全量覆盖它。
- [CODE] 冷 staging 在层内计算后调用 DiscardTransient；chunk 结束时的 InstallHotset 往往重新装入刚算过的专家。H5 复用幸存目标，但没有实现层内临时槽的直接 admission/所有权接管。
- [MEASURED] 频率命中提升主要减少了 host-Ready CPU 冷项，昂贵的 GPU 冷供给没有相应减少；额外 epoch 传输反而抵消收益。

例如同二进制的首次跨域对照，代码请求前 8 个 decode token 的 direct 覆盖从 740/3840（19.3%）提高到 2044/3840（53.2%），CPU join 从 0.829 降到 0.594 秒，但 decode demand 从 5.175 变成 5.297 GB，epoch 从 4.177 增到 6.052 GB，整请求 12.654→13.772 秒。仅报告 hit-rate 会误判这组结果。

## 实现及开关

以下为本轮新假设，不是 Strata 的已测常数；Strata 对照源码仍为 a79080535d1b2a71a3419a0d97d8e7dca194b0f1，未修改其源码。

1. AILA_Q4E_HOTSET_PROMPT=1：保存当前 prefill 的逐层实际路由，按 token 距离衰减，半衰期 128 tokens。跨 chunk 累积与切片方式无关，不使用未来生成 token。
2. 交接时用逐层平方余弦相似度混合上一 prompt 的 decode usage 与当前 prompt 的归一化频率；种子质量为 4/(1−0.7)，对应现有 4-forward 衰减周期的量级。相同 prompt 保留已学到的 usage，跨域时削弱旧信息。
3. AILA_Q4E_HOTSET_PROMPT_SWAPS（默认实验预算 1024）限制交接替换；预算随相似度下降。先决定新逻辑目标，再归还借槽，省掉“先恢复旧目标、再换出”的重复装载。0 只初始化热度并继续在线 admission。
4. AILA_Q4E_HOTSET_PROMPT_PREFILL=1：在 chunk 边界更新临时 prefill 热表。它与学习得到的 decode 目标分离，保留 3055 热槽及 512 服务槽，之后再交接到 3551/16。最后不足 32 个 prompt tokens 时不重建；32 是实验摊销保护，不是 kernel dispatch 定律。
5. 对照在线 swaps=96 与 512，interval 均为 4。提高上限没有显示稳定收益。全部 expert/rank/route weight 保留，compute placement 仍沿用 H4 的 CPU/GPU 分工。

新增 ForwardOptions.prefill_remaining，由真实 prompt 分片传递；单行 prefill 尾部与生成结束时的 KV 对齐分开。后一条是验收时发现并修复的边界问题。MTP 和 HTTP 多会话仍未认证；未知阶段的多行 forward 不执行本轮 chunk 更新。

仍只保存上一 prompt 的统计，不是多 prompt 热集库；返回更早的领域仍需重新适应。逐层逻辑容量仍受原 profile 配额约束。本轮没有把 checkpoint 变成缓存快照，也没有改变权重格式或 kernel 数学。

## 性能实验与未获益的候选

9 个性能进程、43 个完整请求，每次实际生成 48 tokens。固定 max-seq=2560、chunk=512、3567 总槽、12 CPU 线程及 10 credits、cold policy=1。跨域顺序为故事41→档案1833→代码47→档案1833→故事41，每次 /clear，热集与 host pool 保留。档案和故事未加入离线 profile 校准。

所有首请求、回退和异常均保留；不把不同 prompt 混成“warm 中位数”。两个阶段标识尚有生成尾部误分类问题的进程（seed-a、control-b，共10请求）明确排除最终性能判断。旧原型按 EXE SHA 区分。

- 首轮仅交接候选：首次档案 decode 7.8→7.5 tok/s、整请求47.763→49.244秒；返回档案7.5→7.2、33.156→34.735秒。
- chunk 更新原型：首次档案 prefill demand 从参考104.420降到98.299 GB，但整请求 epoch 增到16.578 GB，完整请求50.943秒。这里的参考不是相同二进制，作为诊断记录，不计算独立算法加速比。该原型还对最后4-token尾部做了无益刷新；最终版本已修正并通过质量/trace，但没有额外宣称修正后的性能收益。
- 相同故事重复3次的初版为5.2/11.3/15.1 tok/s；它仍有重复 prompt 尾部重排问题，不能拿15.1代表跨域收益。

最终同二进制反向顺序对照（103CC7363305…）：

| 请求位置 | H4 decode tok/s | 轻量 seed + 512 swaps | H4 整请求秒 | seed 整请求秒 |
|---|---:|---:|---:|---:|
|1 故事|2.3|4.6|31.262|20.881|
|2 档案|7.5|7.5|51.315|51.538|
|3 代码|4.6|4.5|13.607|14.073|
|4 档案|7.5|6.9|34.743|36.660|
|5 故事|7.4|7.1|9.577|10.240|

基线首请求出现较大的 decode 慢样本，保留且不归因于驱动；模型预装也在请求计时之外。这一异常不能用来宣称 seed 的总体收益。两次档案的有效输入/TTFT也未达到100 tok/s，decode未达到20。最终样本是筛选性对照，没有足够证据发布新性能 dispatch。

## 时间线定位

普通 trace（没有整模型 SYCL event profiling）通过33次forward、919对retire/publish、3975个CPU任务的所有权/原rank/受保护victim验证。包含3次真实prompt交接、1次warmup交接、3次chunk交接；最后4-token尾部没有重建。

该 trace 的档案前8个decode forward，平均每token：

| 观测范围 | 毫秒 |
|---|---:|
|完整forward host span|129.17|
|expert_acquire_submit范围|83.92|
|其中acquire调用span|48.27|
|acquire内prefetch等待统计|15.81|
|CPU join|16.82|
|router等待及绑定|6.73|
|resident GPU提交|5.04|

上述范围有嵌套/并发，不能直接相加；不是纯GPU时间，epoch重建在forward span之外，trace本身也有开销。它定位的是暴露的供给/提交等待，不能用129.17ms换算成未经trace的基准吞吐。尤其不能把CPU join之外的全部时间都归为GPU kernel计算。

## 验证与资源

| 48步teacher forcing | GPU逐位一致 | CPU平均KL | CPU平均NLL增量 | CPU最大NRMSE | CPU任务 |
|---|---:|---:|---:|---:|---:|
|41-token故事|48/48|0.00177012|+0.00198337|0.0562232|7320|
|1833-token档案|48/48|0.00243918|−0.00943411|0.0609069|5739|

两组均启用chunk候选和512在线swap上限，checkpoint第24步、后缀重复重放、真实CPU结果边界取消/恢复通过，decode live USM数量不增长，NLL预警均为0。质量canary不是大语料认证。

6项相关CTest通过（65.80秒）；对齐边界修复后Generation/Readiness两项复验通过（6.00秒）。Python replay/profile/因果回放共13项通过。一个新增fixture最初错误假定expert 0不可能被替换，已修正为检查预算外的保留项；失败日志保留。主运行时四个产物已构建。

12个模型进程（9性能、2质量、1trace）均exit=0，无watchdog或8 GiB保护终止。最高工作集37.52 GiB，最低系统可用11.45 GiB；32 GiB仍指host pool，未声称改善总RAM。没有旧驱动A/B，也没有更长QSA区间、HTTP/MTP认证。

## 后续优先级

1. 在层内丢弃冷staging之前，依据当前prompt路由接管有价值的Ready槽；保持reader fence、generation、物理/逻辑表原子发布，避免epoch重装。不要把本轮高成本InstallHotset当作这一路线的最终实现。
2. Admission从频率收益改为预计节省的GPU需求/host fill/等待成本，区别便宜的host-Ready CPU冷项与昂贵的GPU冷项；将refill成本和prefetch阻塞纳入同一预算。先用捕获fixture验证并发，再做同域切换对照。
3. 随后验证有界多prompt历史/领域返回，以及固定逐层配额的影响。稳定物理驻留后再接GPU提交图和R2b。100+ prefill / 20 decode及长QSA曲线仍是未完成目标。
