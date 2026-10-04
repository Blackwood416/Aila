# Qwen3.8 Flash Next H10：有界 CPU NF4 缓存及 prompt 重填

日期：2026-10-04。接续 [H9](qwen3.8-flash-next-heterogeneous-h9-results-2026-10-04.md)，基于提交 `d5f053c`。**本轮两种缓存均不推广，默认仍采用 H9 的 pending residual 融合。20 tok/s 未达到。**

借用 Strata 有界 host arena 的供给思路，实现 CPU 专用的原 NF4 packed 权重缓存，再尝试用当前 prefill 路由重填它。缓存确实减少了部分原始 mapped 读取，但更高命中率没有转化为整体加速。静态 4 GiB 在正式三轮中慢约13%；prompt 版本提高覆盖率，却引入不可忽略的复制停顿。没有继续将这些负结果扩大成推广测试。

机器可读 [H10 结果](qwen3.8-flash-next-heterogeneous-h10-results-2026-10-04.json)保存全部14个性能进程的逐请求结果、源码/EXE/模型配置/profile/软件栈身份、日志 hash、诊断、canaries 和检查器失败。Stage1 与 Stage2 使用不同 EXE，独立比较。

## 实现与比较边界

- `AILA_Q4E_CPU_PACKED_CACHE_MB` 默认0，上限8192 MiB。只复制 gate/up 和 down 的原 packed 字节，原 nested codes、absmax、offset 和 LUT 继续引用模型映射。真实每专家 packed payload 为 **2,457,600 B**，不同于 GPU slot 的2,534,400 B。
- 缓存由 cache 对象持有，是普通 pageable CPU heap；不是 pinned USM，也不保证物理驻留。它有独立的显式 RAM 上限，不并入 expanded host pool 或 USM ledger。生产配置要求独立 offline profile、mapped CPU source 和 expanded host pool=0。
- Stage1 按独立 H4 profile 的排序选取初始 GPU 热集以外的专家，初始化后只读。后续 GPU ready adoption 可能与静态 CPU 缓存重叠；收益依据真实 `cpu_cache_tasks`，不是 oracle 覆盖率。初始化复制在请求计时之外，耗时仍完整记录。
- Stage2 的 `AILA_Q4E_CPU_PACKED_CACHE_PROMPT` 默认0。每次完成 prefill 后，用该 prompt 的归一化路由频率排序，排除当时 GPU Ready/Copying 专家。重填前由 backend 排空 CPU readers；复用原 arena 和保留专家原槽，只复制新选中的专家。没有第二个多 GiB arena、USM 分配或 descriptor 重绑。
- Prompt 重填在首个 decode forward 的请求时间内，可能发生在首 token 已产生之后；不得把该停顿仅用 TTFT 评价。请求总耗时和 decode 速度包含其成本。权重拷贝期间无 CPU reader，普通后续 decode 不改写缓存。
- CPU 算术仍为 H8 NF4-Q8。diagnostic shadow 对实际缓存任务重新使用**原始 mmap 来源**及旧 enum6 计算，检查中间 BF16 与最终 FP32 输出逐字节一致，避免仅用同一缓存数据作自身对照。

机器仍为 A770 16 GiB、driver32.0.101.9033、Xeon E5-2696v3、oneAPI2026.1。固定 H9 pending=1、HC WG256/up串行、CPU12线程/10credits、mmap=1/Q8=1、host-pool=0、prefetch/touch/complement=0、H4 profile、chunk2048、ready-adopt4096、swaps0。GPU auto3567 slots，decode 热集3551、暂存16。单 GPU 工作负载串行运行；正式测量不与编译或 CPU microbench 重叠。保留8192 MiB系统 free RAM watchdog。

## Stage1：静态 cache

完整编译及六项 CTest 通过（38.83秒），源码快照16文件。[初筛](../../tmp/h10-stage1-screen-summary.json)为每臂一个新进程，24 outputs × story41/archive1833/code47 三请求：

| CPU budget | 实际 payload / slots | 初始化复制 | 三请求总耗时 | 各请求 CPU 冷任务命中率 |
|---|---|---|---|---|
| 0 | 0 | 无 | 24.55502 s | 0 / 0 / 0 |
| 1024 MiB | 1,071,513,600 B / 436 | 1.039 s | 24.00458 s | 3.28% / 4.10% / 2.03% |
| 2048 MiB | 2,145,484,800 B / 873 | 2.048 s | 23.96933 s | 6.53% / 8.11% / 4.53% |
| 4096 MiB | 4,293,427,200 B / 1747 | 4.017 s | 23.82130 s | 12.33% / 15.89% / 10.13% |

基线先跑，没有清理 Windows file cache；初筛约3%的差值未作为收益证据。4 GiB 的实际原映射 shadow8-output/3-request检查通过，1181批 bitwise，缓存任务观察753次，单独标为 diagnostic，排除性能比较。

随后 [正式三轮](../../tmp/h10-stage1-formal-summary.json)：每臂三个新进程，每个48 outputs × 五请求 `41→1833→47→1833→41`，第二轮反序，所有6个进程 exit0。

| 配置 | 五请求总耗时三样本 | 中位数 | CV |
|---|---|---|---|
| H9 pending，无 CPU cache | 53.55093 / 51.14312 / 51.08883 s | **51.14312 s** | 2.21% |
| 静态 CPU cache4096 | 57.81235 / 53.97027 / 59.17547 s | **57.81235 s** | 3.87% |

**总耗时中位数 +13.04%，不推广。** 每轮 paired 都变慢，差值4.26142 / 2.82715 / 8.08664秒。

| 请求位置 | 无缓存 request中位数 / decode | 静态4 GiB request中位数 / decode |
|---|---|---|
| 首次 story41 | 6.93402 s / 9.7 tok/s | 6.81316 s / 9.9 tok/s |
| 首次 archive1833 | 16.62798 s / 11.3 tok/s | 21.70412 s / 10.7 tok/s |
| code47 | 6.75026 s / 9.6 tok/s | 8.39143 s / 7.3 tok/s |
| 返回 archive1833 | 15.49084 s / 13.6 tok/s | 15.46127 s / 14.1 tok/s |
| 返回 story41 | 5.50765 s / 12.4 tok/s | 5.59974 s / 12.1 tok/s |

返回长请求的一点 decode 改善抵不过其他位置的退化。静态4 GiB在正式五个位置仅约7%–17%命中。峰值 WS35,769,823,232 B，最低 free RAM20,273.07 MiB，无 watchdog/pressure，decode demand H2D均0。没有采集可归因的 hard faults、Windows文件缓存驻留或 WDDM paging 数据，不将退化归因于某一种分页机制。

Stage1 CLI SHA256：`1D0CAB79C2324D8FA873FDFFACF8B98E8BEB83FFF2301EAB874B22EA8DE5CCD2`。[源码身份](../../tmp/h10-stage1-source.json)与归档 EXE、每个样本相符。

## Stage2：复用 arena，用 prefill 选专家

完整编译及六项 CTest再次通过（37.05秒）。新增测试覆盖重排不复制已有专家、替换后原字节一致、预算不变、删除旧键、空选择、空后重新填充、非法/重复/无效来源保持旧缓存、move/ref rebind、原 nested metadata 和无 USM 分配。

[24-output三请求初筛](../../tmp/h10-stage2-screen-summary.json)，每臂一个新进程：

| 配置 | 三请求总耗时 | 相对本阶段基线 | 三请求实际 CPU 冷任务命中率 | 三次重填耗时 |
|---|---|---|---|---|
| 无 CPU cache | 23.93912 s | — | 0 / 0 / 0 | 无 |
| prompt1024 | 25.21788 s | +5.34% | 26.10% / 20.02% / 19.82% | 557 / 450 / 393 ms |
| prompt2048 | 26.17621 s | +9.34% | 30.20% / 28.54% / 28.56% | 1114 / 844 / 717 ms |
| prompt4096 | 28.04138 s | +17.14% | 48.02% / 45.24% / 45.64% | 1984 / 1581 / 1251 ms |

1/2/4 GiB三请求重填总字节分别约3.074 / 5.802 / 11.000 GB。命中率提高是可观测事实，但包含复制成本的 decode 速度同样退化。此初筛不能建立可重复收益，未继续将它们作为48-output推广候选；更长生成可能摊薄成本，**本轮没有该证据，不推出长生成 dispatch**。

峰值 WS31,868,567,552 B，最低 free RAM24,650.46 MiB，4个进程均exit0，无 watchdog/pressure。Stage2 CLI SHA256：`3D3860C82BA6F1CE55446E8CDE6AF47A4D776EE20E2394AC4FDEA43D30E987C7`。[源码快照](../../tmp/h10-stage2-source.json)在初筛前保存；CLI随后经同 hash校验归档。阶段身份不合并。

## 正确性、生命周期及入口

4 GiB prompt cache在固定 teacher48、原 HC reference开关和原映射 CPU shadow开启下验收，两次均exit0：

| Canary | GPU1原HC一致 | CPU2 KL / NLL delta / 最大NRMSE | CPU teacher任务 | bitwise shadow批数 |
|---|---|---|---|---|
| [story41](../../tmp/h10-stage2-canary-story.log) | 48/48 | .001778392 / −.010557030 / .054861156 | 15867 | 4606 |
| [archive1833](../../tmp/h10-stage2-canary-archive.log) | 48/48 | .003990639 / −.001242300 / .056646893 | 8764 | 4420 |

GPU的48行包含初次 prefill logits及后续47行 decode logits；最终 teacher forward另验 replay。CPU2仍是近似算术，非48行严格GPU等价。两次KL≤.01、NRMSE≤.15，NLL无warning；replay、checkpoint、cancellation通过，live_usm_delta=0。shadow批次包含重放/其他验收 forwards，与 teacher任务数不是同一统计口径；所有 shadow均bitwise=1，包含真实缓存命中。live USM检查不代表单独的 pageable CPU arena计量。

[最终默认入口8-token smoke](../../tmp/h10-stage2-default-smoke.log)及 caller环境逐项恢复通过。继承的 cache8192/prompt1、CPUVerify和HC reference污染被清理，默认 cache0/prompt0、pending1。仍可使用：

```powershell
pwsh -NoProfile -File perf/qwen4exp/q4e_supply_chat.ps1 -Heterogeneous -PromptFile perf/qwen4exp/prompts/archive-natural.json
```

显式 cache实验参数已保留用于复现，**不是推荐启动参数**。[campaign](../../perf/qwen4exp/q4e_cpu_cache_campaign.ps1)和 [summary](../../perf/qwen4exp/q4e_cpu_cache_summary.py)锁定H9其余环境、同 EXE/model/profile/input/stack/GPU身份；验证真实payload预算、cache+mapped任务计数、非零命中、每请求恰一次prompt重填及对应prefill行数、raw log hash，拒绝诊断/verify/失败/pressure混入性能。

[parser guards](../../tmp/h10-parser-guards.log)、[7臂campaign mock/逆序/AST](../../tmp/h10-campaign-guards.log)、[default/explicit/poison/restore/参数拒绝](../../tmp/h10-launcher-guards.log)、[15 Python tests](../../tmp/h10-python-tests.log)、[immutable bundle检查](../../tmp/h10-bundle-validation.log)通过。三项控制/检查器失败均保留：native `-File -Cases baseline,cpu-cache4096`被当作单字符串而拒绝，改为driver中的真实数组；初版guard的Windows Path拼写比较未施加模拟日志替换，修正Path equality；源码快照helper没有复制Stage2 CLI，bundle拒绝缺失归档，核对当前CLI与已记录hash一致后归档。均未作为性能样本或GPU正确性失败。

本轮新增的是可验证的缓存/重填实验基础，以及它们在代表性请求上无收益的证据。继续推进应优先减少需要搬运的数据或把供给工作真正重叠进既有工作；不能把提高 cache hit、返回长文14.1 tok/s或单次短筛下降当作整条管线已变快。
