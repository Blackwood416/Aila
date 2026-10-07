# MTP 两行 MoE 重叠：保留收益与撤回候选

日期：2026-10-06。延续原 CLI 的 1833-token prompt、64 outputs、A770 / oneAPI 2026.1 / CPU18 / A64 mapped cold / frozen prompt hotset / state-only replay。

**保留的收益是 T2 异步路由：完整循环 16.87 → 17.65 tok/s，约 +4.6%。原场景仍未达到 20+。** T4 提前启动 CPU 反而退化，最终入口限定为两行，默认关闭。

## 无重诊断的原场景对照

| 配置 | T1 完整 / post-first | T2 完整 / post-first | T4 完整 / post-first |
|---|---:|---:|---:|
| 同二进制 control，已有 attention 优化 | 17.523 / 18.514 | 16.868 / 17.704 | 14.436 / 14.959 |
| 初版异步候选，允许 T2/T4 | 17.501 / 18.480 | 17.649 / 18.546 | 13.690 / 14.142 |
| 再加结果 packet，异步已限定 T2，已撤回 packet | 17.463 / 18.411 | 17.786 / 18.680 | 14.914 / 15.490 |

单位 tok/s。前两行使用相同 EXE，只切换 VerifyAsyncRoute。所有 LIVE 都包含真实 draft、选择、checkpoint、verification 和拒绝恢复，不含 prompt prefill、draft prompt KV 初始化和服务 IPC。post-first 剔除首循环但保留收尾。

T2 暖完整宽度窗口（同时剔除首循环和不足两行的收尾）平均 advance=1.848485，接受轨迹不变：

- control：draft 8.059 ms，verify 95.506 ms，commit 0.646 ms。
- async：draft 8.055 ms，verify 90.734 ms，commit 0.644 ms。
- verify 减少约 4.77 ms；同进程 T1 吞吐差异约 0.1%，因此这个收益比跨轮旧基线更可信。
- 达到 20 tok/s，当前接受率留给 verify 约 83.72 ms，仍需再削约 7.01 ms。首次循环开销还会进一步影响完整请求吞吐。

这些是快速单次筛选，不是多轮置信区间或生产服务认证；T2 当前与 T1 基本持平，尚不能宣称普遍胜过单 token decode。

## 定位与实现

先运行 tmp/mtp-moe-t2-profile，启用已有 Profile、WorkSummary 和 LIVE 边界同步，不开逐阶段序列化。33 个 T2 暖完整窗口的累计区间均值：

| 区间 / 计数 | 均值 |
|---|---:|
| 主线程 router wait | 28.86 ms |
| CPU compute | 23.29 ms |
| 主线程 CPU join | 19.91 ms |
| worker activation wait | 6.21 ms |
| CPU result copies | 187.36 / forward |
| result copy host submit | 3.30 ms |

这些区间会重叠，不能相加，也不能把 router wait 当作纯路由内核耗时。它包含等待之前 GPU 工作的时间。

保留修改只复用现有 CPU worker：

1. 多行 activation D2H 放到 router 前；router event 提交后立即向 worker 提交任务。
2. worker 等待 event 后构造冷专家 token/rank 任务，此时主线程继续提交 shared/GPU 工作。worker 只读取已捕获的 eligibility 和 immutable mapped weights，不操作 cache/lease。
3. 主线程仍按原方式绑定 GPU lease、分组执行热专家和合并结果；finish 后逐项检查 worker 的 cold count、expert ID 和 flattened rank 与主线程绑定完全相同。
4. 保留输入、前轮回传和 GPU metadata 的 fence；异常沿用 drain guard。没有新增线程、GPU kernel 或热路径 USM 分配。

修改位置：src/models/Qwen4ExpExpertExecutor.hpp/.cpp、src/models/Qwen4ExpBnb4Backend.cpp 和 runner。worker 结果数组扩为已有 40-task 容量；单行路径显式复制其有效 rank 范围。

AILA_Q4E_VERIFY_ASYNC_ROUTE 默认 0。开启后，backend 只在 count=2、frozen mapped cold、无预取/换入换出且既有 async-source 条件满足时启用。T4 负收益已经收窄在 dispatch 入口，不保留为默认宽度策略。

## 结果合并 packet：通过正确性检查，但已撤回

候选将同层离散 CPU rank spans 打成一次 H2D，使用 40-entry rank map 和现有 ordered merge。增加的临时 host/device arena 均在初始化分配。48 个小型合并用例覆盖 1–4 行、257/2560 hidden、无/部分/全部 CPU ranks、shared gate；对应 GPU rank 槽填 NaN，结果仍与 canonical 合并逐位相同。

性能上 T2 仅多约 0.8%，T4 原场景约 +3.3%，但后者仍不可盈利。于是只补了高接受率代码样本的同二进制 A/B，关闭短上下文曾退化的 attention 开关：

| 121-token 代码场景 / T4 | 完整 tok/s | post-first tok/s |
|---|---:|---:|
| packet 开 | 20.185 | 21.089 |
| packet 关，同 EXE 对照 | 20.410 | 21.365 |

**代码样本单次越过 20，不能归功于 packet。** 关闭候选反而略快；也不能把这一结果等同于原 1833-token 场景达到 20。该 A/B 来自含 packet 扩展的候选二进制，不是最终回退版本的性能认证。

已撤回 packet arena、merge 扩展、runner 参数及专用测试扩展，恢复这些位置的原有实现。保留原始日志和这份反例，避免下次因一次跨阈值就错误归因。

## 验证与限制

- tmp/mtp-moe-async-shadow：32-token 短场景包含 T1/T2/T3/T4 固定轨迹 sweep，2–4 行 CPU A64 shadow 逐位一致，worker/main placement 检查通过，已有 checkpoint/replay bitwise 检查通过，LIVE 与单行 target 32/32 相同。
- 原场景所有有效 64-token LIVE：输出 64/64 一致，live USM delta=0。各宽度首循环后的热集与结束热集均为 7625625710176325289。
- 代码 A/B：64/64 输出一致，无 LIVE USM 增长。此前其他 prompt 变体的跨宽度数值问题并未因此消失。
- 最终构建日志 tmp/mtp-moe-final-build.log。只做本次相关的短验证、真实循环和构建，没有扩展到 scheduler、服务器并发或全 CTest。
- 最终不包含 packet，当前可复现优化开关为 VerifyAsyncRoute=1，仍默认关闭；既有 CPU verifier、GPU 权重复用、state-only replay 与可选长上下文 attention 保留。

## 复现

~~~powershell
pwsh -NoProfile -File tmp/mtp-current-screen.ps1 -Tag NEW-async -Prompt archive-cli-system -Anchors 60 -VerifyCpuRows 4 -VerifyGpuReuse 4 -MtpLiveWidth 2 -MtpStateOnlyReplay 1 -Verify 0 -VerifyDecodeAttention 1 -VerifyAsyncRoute 1
~~~

将 VerifyAsyncRoute 改为 0 是当前二进制的 control；加 MtpLiveCompareWidths 可检查同进程 T1/T2/T4。实际环境、prompt identity、退出状态与完整/暖窗口均值见同名 JSON。所有进程串行运行，构建与 GPU 计时没有重叠。
