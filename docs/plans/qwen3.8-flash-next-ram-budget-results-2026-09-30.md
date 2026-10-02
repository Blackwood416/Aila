# Qwen3.8-Flash-Next：16 / 24 GiB 热专家 RAM 缓存对照

后续按用户要求继续测试了 [32 GiB decode 预算](qwen3.8-flash-next-ram32-decode-results-2026-09-30.md)，并将本机性能探针默认预算更新为 32 GiB；以下保留原 16/24 GiB 对照数据。

2026-09-30，Arc A770 16 GB，驱动 32.0.101.8860，oneAPI 2026.1 / oneDNN 3.11.2，约 64 GiB 主机 RAM；基于 66e21e9 后的 S1–S6 工作区。模型为 `C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2`，继续 GPU-only 专家算术和 buffered ReadFile。只改变 `AILA_Q4E_HOT_RAM_MB=16384/24576`。

**[MEASURED] 额外 8 GiB RAM 对短样本实际 decode 有收益，对本轮长 Prefill 没有稳定收益。** 正反向对照的实际 decode 中位数为 13.15 → 14.8 tok/s（约 +12.5%），完整 48-token 请求时间减少约 8.3%。仍未达到 20 tok/s。24 GiB 可作为这台 64 GiB 机器的 decode 性能候选；本报告仅覆盖 16/24 GiB，未修改运行时的 16 GiB 默认值。

完整身份、逐遍数据及独立缓存诊断见 [JSON](qwen3.8-flash-next-ram-budget-results-2026-09-30.json)。

## 正式无 trace 对照

每种负载按 16 → 24 → 24 → 16 GiB 顺序启动四个独立进程。每个进程执行四遍同一工作，保留首遍，主表取去首遍后的六个样本中位数。没有清空 Windows 文件缓存；首遍不称为冷 SSD 测量，不据此宣称启动提速。各组固定同一二进制、模型、GPU 容量和全部其余开关。

| 指标 | 16 GiB | 24 GiB | 判断 |
|---|---:|---:|---|
| 实际 decode tok/s | 13.15 | 14.8 | +12.5%；反向复测保留 |
| 48-token 完整请求 ms | 5290.3 | 4853.6 | -8.3% |
| 首个流式文本 TTFT ms | 1764.5 | 1743.0 | 差异较小 |
| 2052-token Prefill ms | 11615.3 | 12324.6 | 扩容没有收益，含过渡期 |
| 同一 Prefill tok/s | 176.7 | 166.5 | 不能解释成稳定态必然下降 5.8% |
| 八步固定尾段 ms/步 | 108.9 | 104.4 | 约 -4.1%，非 generated TPS |
| 正式进程峰值工作集 GiB | 21.21 | 29.23 | 增加约 8 GiB |
| 正式运行最低可用 RAM GiB | 25.98 | 18.09 | 均高于 8 GiB 停止线 |

生成负载使用 CLI 流式接口，每遍 `/clear` 后重新生成，greedy、thinking budget=0、max-seq=512、48 个生成 token，stream/decode chunk 均为 1。实际格式化输入为 **41 tokens**，包含 CLI 默认系统上下文；不能与此前 messages-json 的 30-token story 直接混算。GPU 专家槽固定 **3714**，B6=32，phase scratch 关闭。流式回调数逐遍验证等于引擎记录的 48 个生成 token，因此使用 CLI 的首个文本回调至请求返回区间估算实际生成速度，包含末尾上下文对齐等收尾工作。显示精度为 0.1 tok/s / 1 ms。

16 GiB 六个后续 decode 样本为 12.4、12.9、13.2、13.1、13.2、13.2 tok/s；24 GiB 为 14.8、14.9、14.8、14.8、14.8、14.9 tok/s。两组 16 次完整生成的文本 SHA256 相同。16 GiB 首个进程第一请求为 15.50 秒、反向复测第一请求为 6.02 秒；24 GiB 两次为约 5.98 / 5.97 秒。这个顺序效应说明不能拿 15.50 → 5.98 秒当作扩容的冷启动收益。

长 Prefill 沿用 2052 个重复灯塔 token，chunk=2048+4，max-seq=2060，B6=32，两组均开启 phase scratch，GPU 专家槽固定 **3602**。最终 device USM ledger 均为 15054377266 bytes（14.020 GiB）。每遍后追加八个固定 token，Prefill 最后一行和八行续写 logits 哈希在全部对照中一致，decode live USM 不增长，mmap fallback 为零。

第二遍仍在过渡，耗时分别约 20.7–21.4 / 18.5–19.2 秒。第四遍 16 GiB 为 11313.8 / 11324.6 ms，24 GiB 为 11346.6 / 11379.0 ms，两组约 181 tok/s。因此可靠结论是长 Prefill 没有稳定改善，不能将主表的有限次数混合中位数外推成所有长输入的稳定回退。

## 独立缓存诊断

各预算另跑一个长 Prefill 进程和一个真实生成进程，每个仍为四遍，打开 cache trace，不把这些耗时并入正式速度。两组逐 token 路由哈希完全一致，GPU 槽数相同；主机槽数为 6778 / 10168。下表取第四遍；短 decode 的 48 次单行 forward 包括请求末尾上下文对齐。

| 需求供给计数 / 主机计时 | 16 GiB | 24 GiB |
|---|---:|---:|
| 长 Prefill 主机缓存填充次数 | 8037 | 5667 |
| 长 Prefill fill ms | 2969.6 | 2338.4 |
| 长 Prefill 需求 H2D GB | 36.338 | 36.338 |
| 短 decode 主机缓存填充次数 | 1323 | 0 |
| 短 decode fill ms | 561.7 | 0 |
| 短 decode acquire ms | 934.1 | 229.9 |
| 短 decode 需求 H2D GB | 8.799 | 8.812 |

24 GiB 下短请求第二至第四遍的 Prefill 和 decode 主机缓存填充均为零，说明该样本的热权重可以留在更大的主机池内。16 GiB 下这三遍 decode 仍有 1330 / 1328 / 1323 次填充。GPU 命中与 H2D 没有因此明显改善，支持本轮 decode 收益主要来自减少主机供给等待的判断；现有 cost-aware GPU 驱逐依赖 host readiness，不能承诺不同 RAM 预算的 GPU miss 总是严格相同。

长 Prefill 填充次数减少约 29.5%，却仍需搬运同样的 36.3 GB 权重，正式总时间没有改善。`fill` 和 `acquire` 为异步管线中的主机区间，不能直接相加到总时间或解释为纯 SSD / 纯 GPU 时间。ReadFile 使用 Windows 文件缓存，`file_fills` 表示主机池缺失后的读取/整理，不等于物理 SSD I/O；扩大的 pinned pool 也会与 Windows 文件缓存竞争。

## 复现与范围

`q4e_execution_probe.ps1` 新增显式 `-HostRamMB` 和 `generation` 负载；正式运行关闭逐 forward profile，仅显式 `-Profile` 打开。`q4e_ram_summary.py` 校验 exit/watchdog、日志 SHA256、输入/二进制/模型身份、环境变量差异、GPU 槽数和输出一致性。没有修改内核、生产缓存算法或全局默认，也无需重编译运行时；本轮不代表服务并发、采样、MTP 接受率或完整质量评估。

全部八个正式进程、四个诊断进程 exit=0，未触发 240 秒 watchdog 或低于 8 GiB 可用 RAM 的停止条件。报告与 runner 的 Python / PowerShell 语法检查和 Git whitespace 检查通过。本轮只改变 benchmark 工具及文档，未重复无关 CTest。

```powershell
# 分别以 16384/24576 运行，并按 AB/BA 顺序复测。
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload memory -MemoryTokens 2052 -MemoryRepeats 4 -MemoryDecodeTail 8 -PrefillChunk 2048 -DemandMicrobatch 32 -PhaseScratch -HostRamMB 24576 -Tag ram24-prefill-example
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload generation -GenerationRepeats 4 -MaxTokens 48 -DemandMicrobatch 32 -HostRamMB 24576 -Tag ram24-generation-example
# 独立加 -Trace 采集缓存诊断；不把其耗时并入正式比较。
```
