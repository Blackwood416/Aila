# MTP：非 MoE 权重复用与首轮成本

日期：2026-10-06。设备和配置沿用 A770 / oneAPI 2026.1 / CPU18 A64 mapped cold / prompt hotset / T2 async / state-only replay。输入仍是原 CLI 的 1833-token prompt。

**保留的优化是 NF4 投影对两个输入 token 复用解量化权重。** 64-token 同二进制对照：完整循环 17.521 → 18.694 tok/s，约 +6.7%；post-first 为 19.722。20+ 尚未达成。扩到 128 outputs 后还发现了已有的 T1/多行数值分歧，不能把这组速度当作无损 MTP 已达标的证据。

## 保留实现

src/ops/Bnb4BitLinear.cpp 的 packed_nf4_decode_rows_bf16 增加编译期 ReuseTwo 分支：

- 每个 subgroup 仍负责一个输出通道，为两个输入 token 累加两个独立结果。
- packed uint32、absmax、SLM NF4 map lookup 和 BF16 权重舍入复用一次；各 token 的 vec8 FMA 顺序及 subgroup reduction 不变。
- 未改变权重布局、精度或 residual；不展开整份 BF16 权重，不缓存输入到 SLM，不增加热路径 USM 分配。
- 原路径保留为模板的默认分支。开关默认关闭，只有 rows=2 和指定形状才走候选。

AILA_Q4E_VERIFY_LINEAR_REUSE / runner 的 VerifyLinearReuse：

| 值 | 覆盖形状，按输出 N × 输入 K |
|---|---|
| 0 | 原路径 |
| 1 | linear_all：16480 × 2560 |
| 2 | 上述形状，加 attention output：2560 × 6144 |
| 3 | 上述形状，加 full QKV：13312 × 2560 |

模式 2 有完整的 64-token 同二进制对照；模式 3 有 128-token 同二进制对照。没有采集 GRF/spill 数量，性能结论以实际 wall time 为准。

另保留两项诊断改进：

- ProfileComponents 在 T2 verifier 内分别同步测量 linear input projection / GDN recurrence / linear output projection / router / shared gate。默认关闭，数据用于拆分，不能当无扰动关键路径。
- LIVE 现在输出 target_prompt_ms、draft_kv_init_ms、prompt_to_output_ms。该总区间覆盖目标 prompt、draft KV 初始化和生成，不包含模型加载、tokenization、IPC 或预先分配的探针存储。
- MtpAnchors 上限由 60 扩到 124，以支持原 CLI 的 128-token 输出口径。仍受 640/2048 context capacity 检查约束；本场景末位置 1961，没有扩展 sparse QSA。

## 真实生成对照

LIVE 均包含真实草稿、greedy 选择、checkpoint、验证和恢复。以下都是快速单次筛选，非多轮统计认证。post-first 仅排除首循环；“暖完整窗口”还排除了不足标称宽度的收尾。

| 1833-token prompt / T2 | outputs | 完整 tok/s | post-first tok/s | 暖完整 verify ms |
|---|---:|---:|---:|---:|
| 仅复用 linear input，模式 1 | 64 | 18.238 | 19.198 | 87.083 |
| 模式 2 的同 EXE control，模式 0 | 64 | 17.521 | 18.425 | 91.225 |
| 模式 2 | 64 | 18.694 | 19.722 | 84.480 |
| 模式 3 的同 EXE control，模式 0 | 128 | 17.139 | 17.550 | 90.834 |
| 模式 3 | 128 | 18.097 | 18.545 | 85.213 |

64-token 模式 2 对照中，T1 完整速度为 17.195 / 17.162，T4 为 14.270 / 14.262，基本不变。T2 generated token 数组和 advance 数组在开关两边完全相同，且各自与 T1 64/64 一致。热集 identity 保持 7625625710176325289。

128-token 模式 3 和 control 的 T2 token/advance 数组也完全相同。但两者均在输出 index 123–127（第 124–128 个 token）与各自 T1 不同；T4 也是同样的位置。新权重复用没有引入本例的分歧，原多行路径的长输出对齐问题仍需定位。

128-token 的暖完整 T2 advance 降至 1.76056，64-token 为 1.84848，因此不能将短样本的接受率直接外推。

## 拆分诊断与撤回项

tmp/mtp-components-t2 使用 16 outputs；8 个暖完整窗口的同步区间均值（每个完整 forward 汇总）为：

| 区间 | ms |
|---|---:|
| linear input projection | 12.306 |
| GDN recurrence | 9.013 |
| linear output projection | 7.107 |
| router BF16 projection | 7.395 |
| shared gate BF16 projection | 12.724 |

这些同步会破坏重叠，不能把各区间与 CPU join 直接相加，也不能把一个大区间等同于可节省的时间。shared gate 的真实 A/B 就没有兑现诊断中的大区间。

已撤回以下候选和对应运行时开关：

1. **已有 multi-row exact-merge attention + 独立 sigmoid gate。** 预分配了两行私有 scratch，保留逐行 causal length，日志确认选中 multi-row-exact-merge。实际 gated BF16 shadow 比较失败；无 shadow 筛选 17.449 tok/s，也没有明确收益。64 个 greedy token 一致不能推翻逐位失败。静态代码显示单行 BF16 QK 已有 vec8 K 读取，多行仍是标量读取，但尚未证明这就是差异原因；未继续改写整个 attention 内核。
2. **shared gate 改为两次已有 BF16 GEMV。** 同 EXE T2 为 17.385，对照为 17.475 tok/s，没有收益；输出和 advance 相同。
3. **合并多行 logits D2H + 缓存 CPU argmax 最大值。** 保留了最低索引 tie 和全量 finite 检查，但筛选为 18.676 tok/s，未见清晰净收益，恢复原选择实现。
4. **直接 borrow=0。** 该开关同时将热集从 3402 缩到 2906 槽，改变了 hotset identity，并出现 15/64 个 T1 mismatch；不是保持热集的有效对照，不能用它得出借槽优化结论。
5. **保持 3402 槽，只禁止 prefill 借槽。** 单独加过实验开关并把 microbatch 限到 16，保持相同逻辑热集。首循环变快，但总区间明显更慢，因此开关及临时 dispatch 改动均撤回。

attention shadow 失败日志为 tmp/mtp-attn-rows-shadow.err.log，exit=1；其它无 shadow 吞吐不是该候选数值正确的证明。失败日志没有删除。

## 首轮和总区间：没有把成本移出计时

| 64-token T2 配置 | target prompt ms | draft KV init ms | 首循环 ms | 生成 loop ms | prompt→输出 ms |
|---|---:|---:|---:|---:|---:|
| 模式 2 control | 11684.387 | 1251.788 | 287.823 | 3652.858 | 16589.040 |
| 模式 2，正常借槽 | 11691.800 | 1250.223 | 279.682 | 3423.471 | 16365.502 |
| 模式 2，保持热槽候选，已撤回 | 12852.756 | 1249.522 | 104.643 | 3401.537 | 17503.864 |

保持热槽省下约 175 ms 首循环，但 prompt 增加约 1.16 秒，暖阶段也变慢；总区间增加约 1.14 秒。相同逻辑 hotset hash 并不保证相同物理 slot 布局，未将暖阶段变化归因于某个未经测量的因素。

旧 async-live64 首轮有 1,257,062,400 bytes / 176.254 ms 的热集归还。swaps=0 不等于没有回填。

当前模式 2：暖完整窗口 advance=1.848485、draft=8.2307 ms、verify=84.4800 ms、commit=0.6469 ms。暖完整窗口达到 20 只差约 0.93 ms。但完整 64-token loop 为 3423.471 ms，目标为 3200 ms，仍差 223.471 ms；如果只改 33 个暖完整窗口，需要每窗约 6.77 ms。二者不能混为一个目标。

## 验证、构建与复现

- 模式 1/2/3 均运行现有 AilaDecodeRowProjectionTests --run，未新增测试框架。目标 B2 输出分别为 32960、5120、26624 个 BF16 值，全部与原 B1 逐位一致；现有 CPU reference、canary、容量/shape、USM teardown 检查通过。各次完整日志在 tmp/mtp-linear-reuse*-unit.log。
- 所有有效 LIVE 报告 tracked USM delta=0。64-token 推荐路径 64/64 parity；128-token 与原 T2 相同，但有上述 T1 分歧。
- 最终构建：tmp/mtp-linear-reuse3-build.log；相关 git diff --check 通过。没有全 CTest、服务器并发、生产 scheduler 集成或硬件 GRF/spill 采样。
- 之前保留的 CPU verifier、GPU expert 权重复用、T2 async、state-only replay 和可选长上下文 decode attention 均继续保留。新投影复用默认 0，失败候选均已撤回。

64-token 已验证配置：

~~~powershell
pwsh -NoProfile -File tmp/mtp-current-screen.ps1 -Tag NEW-reuse -Prompt archive-cli-system -Anchors 60 -VerifyCpuRows 4 -VerifyGpuReuse 4 -MtpLiveWidth 2 -MtpStateOnlyReplay 1 -Verify 0 -VerifyDecodeAttention 1 -VerifyAsyncRoute 1 -VerifyLinearReuse 2
~~~

128-token 用 Anchors=124、VerifyLinearReuse=3，须同时保留数值分歧说明。模式 0 是同 EXE control。机器可读记录、环境及比较结果见同名 JSON；原始日志、metadata 和失败日志均保留在 tmp/。
