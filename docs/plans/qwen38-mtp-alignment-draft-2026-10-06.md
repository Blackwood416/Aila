# MTP：数值边界、草稿置信度与四 stream 权重复用

2026-10-06，A770 / driver 32.0.101.9033 / oneAPI 2026.1。沿用 CPU18 A64、mapped cold、冻结 prompt hotset、T2 async、state-only replay、decode attention、NF4 projection reuse=3。原 CLI prompt 为 1833 tokens，本轮完整输出 128 tokens。

**本轮没有找到可靠的 decode 20+ 突破。** 新增的草稿 FC 权重复用只改善了初始化；准备阶段和单阈值策略的循环收益不足 1%。不要将不同构建/批次的 18.10、18.59、18.75 tok/s 差异全部归因于优化。

## 同二进制对照

均为快速单次筛选，未做多轮统计认证。完整循环包含草稿、checkpoint、验证、拒绝恢复和首轮热集归还；初始化不计入 decode tok/s。

| 候选 | control → candidate tok/s | 循环节省 | 结论 |
|---|---:|---:|---|
| 两次现成单行 QKV prepare | 18.479 → 18.590 | 41.22 ms，+0.60% | 弱信号；保留为可选对齐参考 |
| draft margin < 1.5 时改用 T1 验证 | 18.597 → 18.782 | 67.96 ms，+1.00% | 收益小且仅本样本；已撤回策略 |
| draft fcHidden 四行分两对 | 18.754 → 18.726 | -10.14 ms，-0.15% | decode 无收益；只保留初始化候选 |

三组都核对了 EXE SHA256、完整 generated 数组和热集 identity（7625625710176325289）。prepare 和 hidden-pairs 的 advance 数组也相同；confidence 策略主动改变窗口宽度，advance 不同是预期行为。全部有效 LIVE 的 tracked USM delta=0。

hidden-pairs 的草稿 KV 初始化为 **1239.965 → 1203.750 ms，节省 36.215 ms（2.92%）**；prompt→输出为 19724.300 → 19680.693 ms。暖窗口 draft 为 8.3156 → 8.3074 ms，基本不变。该完整请求区间仍不含模型加载、tokenization、IPC。

所有本轮多行版本仍在 output index 123–127 与 T1 不同，不能声称无损 MTP 已对齐。历史 T1 探针的请求总时间仍包含本不需要的 MTP KV 初始化，不能直接视为普通 T1 请求基线。

## 关键分歧诊断

`tmp/mtp-prepare-shadow128` 在 target pos=1954、count=2，从同一份当前 T2 checkpoint 和相同两个 token 分别执行 B2 与连续 B1，然后恢复并重跑实际 B2。该探针的循环计时被额外工作污染，不用于性能结论。

- row 0：229120 个 logits 不同，最大绝对差 0.265625；top1 都是 1902。
- row 1（pos=1955，决定输出第 124 个 token）：两路的 logits[1118] 和 logits[1870] 都为 19.625，margin=0，均按最低编号规则选择 1118。
- 因此，仅在这个窗口切成连续 B1，并不能恢复原始 T1 轨迹的 1870；进入窗口之前的状态也需要考察。
- 首层 GDN 最终状态有 123948 个 FP32 值不同，max_abs=1.3411e-7。后续层输入已经可能不同，不能将其更大的状态差异全归因于 GDN 自身。

当前 shadow 也支持从 T1 的已知前缀做同窗口对照，但本轮只运行了上述 T2 入口版本。没有采集 router/placement 或完整逐层张量，首个造成整模型漂移的算子仍未定位。

## GDN 与 QKV prepare

源码确实存在 GDN 单行八项、多行四项累加，以及 RoPE 单行 native sin/cos、多行普通 sin/cos 的差异。它们是嫌疑点，不是已证明的完整根因。

1. 尝试将短 batched GDN 改成八项累加，保持大 prompt 路径不变。128-token 混合宽度逐位检查失败，未跑性能筛选，已撤回。失败日志 `tmp/mtp-gdn-order-unit.log` 和源码 `tmp/mtp-gdn-eight-term-candidate.cpp` 保留。
2. `VerifyGdnOrder=1` 现在只作参考路径：对 1–4 行调用真实单行 GDN 内核；verifier 与 state-only replay 同时使用。测试覆盖非零初始 FP32 状态、非零卷积历史、混合 B1/B2/B4、Sigmoid/Silu，128-token 输出、状态和历史逐位一致，USM 不增长。
3. 整模型使用该 GDN 参考和 QKV prepare 后仍在第 124 个 token 分歧，筛选为 18.385 tok/s，也改变了部分草稿接受窗口；未证明性能收益或整模型无损对齐。
4. `VerifyDecodePrepare=1` 仅在 dense-QSA 的 T2 verifier、start_pos>0 使用两次现成 `decode_prepare_qgkv_packed_partial`。沿用原 packed stride、MRoPE、KV 写入和逐行 attention；不重写 attention 内核，不新增 USM scratch。

## margin 策略：实测后撤回

`MtpDraftMargins` 在既有 CPU logits 扫描中记录 top1-top2 差值，不增加 D2H 或 softmax。GDN 参考样本分歧前的暖窗口中，接受/拒绝草稿的 margin 中位数为 4.0625 / 1.125，但低于 1.5 的窗口各有 12 个接受、12 个拒绝，误判明显。

真实动态实验是在完成 draft Step 后才选择 T1，计入已经发生的草稿成本；此时当前位置 draft KV 已建立，后续切回 T2 不会因 lastCorrect 跳过必要 KV。暖阶段 28 个单行窗口、52 个双行窗口，后者仍有 6 次拒绝。没有采用离线 oracle 速度作为实测。

约 1% 的单样本收益不足以继续引入策略复杂度。阈值开关和执行分支已撤回，保留 margin 采样。候选探针源码存于 `tmp/mtp-confidence-candidate.hpp`；未做阈值搜索或跨提示推广。

## 草稿 FC 成对复用

`MtpHiddenReuse=1` 对 `fcHidden[N=2560,K=2560]` 的四个 hidden stream 分成两对，沿用已验证的 ReuseTwo：每个 subgroup 仍一个输出通道、两个输入累加器。修正 pair 的 input/output base、尾部和第二行写入；原 T2 投影分支保持相同数学。所有开关默认关闭。

现有 `AilaDecodeRowProjectionTests --run` 增加该形状，B4 的 10240 个 BF16 输出与原 B1 逐位一致；原有形状、CPU 参考、canary、容量、LoRA、USM teardown 检查通过。没有采集 GRF/spill，不能声称无 spill。

复现当前初始化候选：

```powershell
pwsh -NoProfile -File tmp/mtp-current-screen.ps1 -Tag NEW -Prompt archive-cli-system -Anchors 124 -VerifyCpuRows 4 -VerifyGpuReuse 4 -MtpLiveWidth 2 -MtpStateOnlyReplay 1 -Verify 0 -VerifyDecodeAttention 1 -VerifyAsyncRoute 1 -VerifyLinearReuse 3 -VerifyDecodePrepare 1 -MtpHiddenReuse 1
```

`MtpHiddenReuse=0` 为对照；`VerifyGdnOrder=1` 为 GDN 对齐参考；`MtpDraftMargins` 打开采样；`MtpShadowPosition=1954` 会污染计时。

原始日志与 metadata 均在 `tmp/mtp-{prepare,gdn-decode,confidence,hidden-pairs}-*`；本报告同名 JSON 由 `tmp/mtp-alignment-report.py` 汇总。最终构建日志 `tmp/mtp-alignment-final-build.log`，两项聚焦测试日志为 `tmp/mtp-gdn-decode-unit.log`、`tmp/mtp-hidden-pairs-unit.log`；相关 diff whitespace 检查通过。未扩展到全 CTest、服务器或 scheduler 集成。

下一步值得检查首层 GDN 之后的 MoE/HC 等单行、多行差异；性能上，简单 margin 阈值和 draft FC 已显示空间有限。大段 draft prefix KV 的批量初始化尚未尝试，它的收益应归入初始化/请求总时间。
