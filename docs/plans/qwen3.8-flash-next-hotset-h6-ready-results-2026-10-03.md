# H6：prefill Ready 槽接管与供给成本复核

基线 `737375c`，Intel Arc A770 / 32.0.101.9033 / oneAPI 2026.1，32 GiB host pool。
**已实现显式开启的物理槽接管；三组跨域对照尚未证明稳定端到端净收益。默认 preset 不变，R2b 继续暂停。**
原始逐请求、EXE/源码身份、质量和时间线以[结果 JSON](qwen3.8-flash-next-hotset-h6-ready-results-2026-10-03.json)为准。

## 实现与有效范围

- `AILA_Q4E_HOTSET_ADOPT_READY=1` 在 prefill microbatch 退休前尝试接管已完成 DMA 的冷槽，默认关闭。自动启用当前 prompt 统计，但不自动开启 H5 的 chunk 全表重装。
- `AILA_Q4E_HOTSET_ADOPT_LIMIT` 每个 forward 的接管上限，默认实验值96、允许0..4096；本轮性能/质量候选用4096。该值尚不是生产 dispatch。
- [HYPOTHESIS] 当前层的衰减 prompt usage≥2、相对最低热项 gain≥1.5。只比较当前 prompt 已观察的路由；不使用未来 decode。这里只把“权重已由需求付费取得”作为成本限制，**尚未实现完整的供给成本评分**。
- 接管只交换热/冷槽归属，并更新逻辑目标与冷队列成员；incoming/outgoing 的物理地址、key、generation、reader/DMA fence 保留。没有接管 H2D，没有增加 USM。未完成 DMA 和活跃 lease 不可接管；同层容量、总热槽数和服务槽数不变。
- 接管写入 decode 逻辑目标，借槽归还不会重新装回被它替代的旧专家。device descriptor/indices 失效后在下一快照发布；新回归测试还发现并修复了“旧 snapshot 构造的 layer mask 在 descriptor 更新后继续被复用”的问题。
- 接管计数/维护耗时独立于 `epoch_swaps/refill_bytes`。`ready_adoption_ns` 只计接管方法，不等于所有下游成本已经消失。

[CODE] 降出的槽仍受 reader fence 保护，后续 acquire 复用它时会等待该 fence。无接管拷贝并不保证复用无等待；consumer 完成状态、剩余 chunk 路由、host backing/CPU 替代成本仍需进入 admission 预算。

## 三组同二进制跨域对照

每个进程依次故事41→档案1833→代码47→档案1833→故事41，每个请求48生成 tokens，每次 `/clear` 保留模型/缓存。6进程、30完整请求；执行顺序 control/adopt、adopt/control、control/adopt。

共同配置为 max-seq2560、chunk512、3567总槽、prefill3055热槽/512服务槽、decode3551/16、cold queue2、12 CPU线程/10 credits、cold policy1。控制是 **H5轻量 seed（prompt swaps0、online swaps512）**；两臂只改变 Ready 接管开关。没有与旧二进制或 H4 默认配置混算速度比。

下表为相同序列位置的3进程中位数，秒与 tok/s：

|位置|控制整请求|接管整请求|控制 decode|接管 decode|
|---|---:|---:|---:|---:|
|1 故事|23.56|21.98|4.2|4.3|
|2 首次档案|56.30|57.06|7.4|7.2|
|3 代码|15.19|14.75|4.3|4.3|
|4 返回档案|38.67|40.53|6.9|7.3|
|5 返回故事|10.72|10.70|6.8|6.9|

首次档案 prefill demand中位数103.46→98.04 GB，epoch refill5.77→4.37 GB，但TTFT50.03→50.71秒。返回档案 prefill demand107.95→99.54 GB，epoch6.20→4.33 GB，TTFT32.01→34.24秒。**流量减少没有转化为稳定请求收益，decode demand也没有同步减少。**

接管候选首次/返回档案中位数4415/5256次接管，累计方法耗时33.29/34.11毫秒；不是数秒回退的直接维护耗时解释。旧高成本全表重装被绕开后，仍需测供给竞争、fence复用等待和后续 epoch 成本。

首请求控制23.56/41.19/20.92秒（CV0.315）；返回档案控制57.35/38.67/37.52秒（CV0.204），超过本轮筛选阈值0.15。全部保留。首组单独比较会得到返回档案57.35→40.53的假性稳定结论，反向和第三组明确否定了这种报告方式。不归因于驱动；没有旧驱动A/B。

这些是筛选样本，不能发布新 dispatch、普适冷启动加速或100+ prefill / 20 decode结论。

## 验证与诊断

短/长48步 GPU teacher forcing均逐位一致。CPU短/长平均KL分别约0.001853/0.002253，最大NRMSE约0.06058/0.06306，平均NLL增量为−0.00502/−0.00757，NLL预警0；checkpoint重放、取消恢复与decode USM增长检查纳入验收。

Ready fixture覆盖零接管拷贝、generation不变、预算/同层容量、未完成DMA拒绝、活跃lease保护、三种冷队列、旧/新descriptor快照、真实reader fence和借槽归还。trace回放新增独立cold LRU节点接管顺序，区别全局LRU；读取时使用probe metadata的cold queue2。

描述符快照回归最初失败的CTest日志保留，修复后Readiness复验通过。首个campaign漏设GenerationRepeats1，误用默认4次重复，已人工停止并保留14个完成请求及未完成尾部；它不进入上述中位数。

最终6项相关CTest全部通过（32.67秒），Python replay/profile/因果回放15项通过。Aila、AilaWorker、AilaShared、AilaInference已构建，源码/EXE哈希与逐进程元数据保留。正常性能、质量和trace进程均exit0；没有HTTP/MTP或更多上下文形状认证。

真实trace验证33次forward、7405次Ready接管、1134对retire/publish、4次prompt交接（含warmup），没有chunk全表重装。档案前8个decode forward平均host span130.19ms、acquire43.35ms、其中prefetch等待14.04ms，expert_acquire_submit83.89ms、CPU join19.44ms。范围嵌套/并发，不能相加，也不能转成无trace基准吞吐。reader/generation由GPU fixture验证；trace验证的是host映射/原rank/冷队列顺序，不是设备级并发时间。

## Prefetch探索与profiling限制

额外1个同二进制接管进程关闭prefetch，仍按相同五请求序列运行48tokens：

|位置|无prefetch整请求秒|无prefetch decode tok/s|
|---|---:|---:|
|1 故事|21.21|4.8|
|2 首次档案|57.49|8.8|
|3 代码|13.56|4.7|
|4 返回档案|38.55|8.9|
|5 返回故事|9.88|7.6|

长文decode比接管主候选7.2/7.3有改善信号，但首次档案TTFT52.25秒、整请求57.49秒仍未改善。只做了1进程，不作稳定净提速或默认关闭prefetch的建议；需求优先、取消/合并粒度与有用prefetch保留策略需要后续对照。该样本单独放在JSON的exploratory，不混入三组主中位数。

VTune instruction-count短输入采集长时间停留于warmup/首次prefill，未完成完整请求。调用其stop接口返回internal error，最终600秒watchdog终止（exit−1）；第一次日志哈希被未释放句柄阻塞，原metadata保留，句柄关闭后另存日志哈希与错误记录。没有取得可用GPU设备时间，不据此归因于driver，也未继续堆叠profiling探测。随后正常GPU回归通过。VTune内存采样只覆盖即时子进程，不代表整个进程树的峰值。

32 GiB仍是host pool预算；正常性能峰值约37.35 GiB工作集，包含质量进程后37.52 GiB、最低系统可用12.33 GiB，完整资源字段见JSON，没有新增RAM优化结论。首个误配campaign明确排除，VTune失败单列diagnostics。上述主实验加prefetch探索共7性能进程、35完整请求；另有2质量、1普通trace进程。

## 复现与下一步

```powershell
pwsh -NoProfile -File perf/qwen4exp/q4e_ready_adoption_campaign.ps1 -Pairs 3 -AdoptLimit 4096 -Tag h6-ready-final
```

1. 用实际成本替代单独频率差：区别host-Ready CPU冷项与昂贵GPU需求，计入降出项的重新供给、consumer fence和prefetch竞争。
2. 将在线refill预算与Ready接管统一；目前每4-forward的512-swap候选仍可对便宜CPU项主动付费搬运。
3. 保留阶段/所有权验证，再研究多prompt历史和稳定提交路径。当前无HTTP多会话/MTP、长QSA区间或默认启用认证。
