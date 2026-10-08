# Qwen3.8-Flash-Next：单流长文与多轮性能

2026-10-07，Arc A770 16 GiB / driver 32.0.101.8724 / oneAPI 2026.1，Xeon E5-2696 v3、CPU experts 18 threads。模型 `C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2`。本轮只测单会话，MTP 关闭，BF16 KV；没有接入并发路径。尚未达到 Server decode 20+。

本机启动配置采用按需 PLE、chunk4096、3584 MiB reserve。相同请求顺序的 14K 样本 TTFT 从 89.24 降到 73.29 秒（约 -18%），续轮 11.72→11.55 tok/s，decode 基本持平。6K TTFT 40.53→38.51 秒。这里改善的主要是长文首字等待，未把它写成 decode 的突破。

## 保留的实现

- PLE 六个计算中间缓冲及两组投影 scratch 不再按整个上下文容量分配。加载时预留 64 行，实际 prefill 更大时扩容；扩容前排空计算队列，异步 gather/DMA 的 staging banks 不移动。普通 decode 直接复用。32K 容量下，这部分原先固定占 4.375 GiB，现在加载时为 8.75 MiB，2048/4096 行时为 280/560 MiB。数值来自 Tensor payload 尺寸计算，不是 DXGI 峰值。自动显存分配会把加载时省下的空间用于专家缓存，因此整个模型加载账本不会相应减少 4 GiB。
- Shared Session 将实际 prefill 的 remaining-token 信息传给 Q4E，区分单 token prompt 与普通 decode，并接通已有末段处理。MTP 存在时保留旧行为，避免清理 AfterForward 仍需读取的 hidden rows。`DECODE_TRIM_CAP` 继续关闭。
- 可选 IPC `extended_context_v1` 的 prefill chunk 上限由 2048 提到 4096，Server 参数校验一致。旧 hello/default 的 512 上限保持。

## 测量口径

每个独立进程都加载 32K 容量、先运行一个短 warmup，再按 6K 输入→续轮→14K 输入→续轮执行。部分初筛仅运行 6K。实际长 prompt 为 5924/13924 tokens，每轮 96 outputs。续轮 prompt 为 6044/14044，复用 6020/14020 tokens；只需 prefill 24 个新 tokens。

TTFT 为 HTTP 请求到第一个内容片段；首字后速率为 `(completion_tokens-1)/(SSE done-first content)`，包括 IPC 和 finalization。JIT warmup 通常约 33 秒，单独保留，不计作优化后的 decode。未清 Windows 文件缓存，各项是快速筛选的单次样本；扩容改变 CPU/GPU 专家 placement，完整生成文本可能不同，不能将这些数解释为固定轨迹的算子加速或质量评估。原始结果、输出 hash、二进制 hash 和环境见同名 JSON。

| 配置 | 6K 首字 s | 6K 续轮 tok/s | 14K 首字 s | 14K 续轮 tok/s |
|---|---:|---:|---:|---:|
| 改 PLE 前，chunk2048，同顺序复跑 | 40.53 | 10.17 | 89.24 | 11.72 |
| 按需 PLE，chunk2048 | 47.13 | 10.40 | 85.19 | 12.48 |
| 按需 PLE，chunk4096 | 51.65 | 9.66 | 73.50 | 11.40 |
| 按需 PLE，chunk4096，reserve3584 | **38.51** | **10.21** | **73.29** | **11.55** |

新 PLE + 2048 在 14K 样本上 TTFT 减少约 4.5%、续轮吞吐提高约 6.5%；6K 首次 prefill 反而退步。4096 改善本次 14K prefill，但不是所有输入的统一赢家。不能从一项节省显存直接推出同等比例的 decode 提升。

在相同新版/chunk4096 下，仅把 reserve 从 3008 增至 3584 MiB，就使 6K TTFT 从 51.65 降为 38.51 秒。按需 PLE 的额外增长需要留空间，不能把加载时释放的显存全部交给更多专家。这支持显存余量影响性能的解释，但没有 DXGI/VTune counters，不据此断言具体 WDDM page-fault 数。3584 是本机样本配置，不是其他设备的通用阈值。

最终长样本续轮复用 14020 tokens，只提交 24 个新 tokens，首字 1.29 秒。`chunk2048/reserve3008` 的 12.48 tok/s 仍是备选测量，不能把它与默认 chunk4096 的 73.29 秒拼成同一配置。

```powershell
pwsh -NoProfile -File tools/server/start-qwen38.ps1
```

脚本默认 32K 容量、4096 chunk、单会话、exact-prefix reuse、reserve3584。通用 Server 默认 chunk512 不变。复现实验仍以同名 JSON 的完整环境和 overrides 为准。

## 初筛与否决

以下均在修改 PLE 前、32K 容量下运行；对照为 chunk2048、trim0、prompt swaps512、online swaps0。

| 6K 配置 | 首字 s | 续轮 tok/s | 处理 |
|---|---:|---:|---|
| 对照 | 40.70 | 10.19 | 基线 |
| decode trim64 | 39.79 | 9.91 | 不启用 |
| prompt swaps2048 | 39.71 | 10.11 | 不启用 |
| online swaps32，临时 no-pool fallback | 43.22 | 9.55 | 撤回 fallback |
| chunk4096 | 39.64 | 10.45 | 保留手动参数 |
| chunk4096 + attention tile256 | 41.63 | 9.69 | 不作为默认 |

online swaps32 同时受既有 gate 影响，会关闭 CPU async route，因此不能用它证明换入策略本身退步。代码检查确认 async 的真实路由已有 Observe，CPU job 在本层 join；后续若重试，应分离 async 和换入两个变量。本轮临时 fallback 与专项测试归档在 `tmp/q4e-single-hotset-no-pool-experiment.patch`，未保留到运行时代码。

一次 16K 容量、5924-token 输入的 CLI stage attribution 表明：2048-token prefill 块需搬运约 36–39 GB 专家权重，acquire 占很大比重；decode 样本有 403/480 个专家在 CPU 上执行。stage profile 会同步并扰动时序，仅作为调查方向，不能拼成上述 32K 容量、无 profile 的 HTTP 时延。原始日志 `tmp/q4e-single-stages.log`。

## 验证与范围

最终配置下，真实 PI 长背景工具回合通过：第一轮实际输入 **14852 tokens**，生成 63 tokens 并恰好一次成功读取随机标记文件；第二轮复用 **14915 tokens**，只提交 **26 个新 tokens**，输出 8 tokens，准确返回标记。完整 PI CLI 时延 142.54 秒，包含首次长 prefill 和两次请求，不是 decode benchmark。随机标记/路径长度不同，不能把它与上轮 167.82 秒直接当作固定输入加速比。证据 `tmp/q4e-single-ple-pi/pi/aila-pi-smoke-o1507u/result.json`、同目录 `events.jsonl`，以及 `tmp/q4e-single-ple-pi/result.json`；Server 优雅关闭。

`AilaQwen4ExpModuleTests --ple-only` 在真实权重上检查 4→65→4→1 行，验证扩容前后同一输入逐位相等、ticket 与 forward 一致。常规和 `PREFILL_XMX=1` 两条路径均通过；最终 XMX 检查还在单 token 回合把设备/host allocation limit 收紧到当前占用。只运行相关 RequestState/Protocol 检查与真实 HTTP 探针，没有重跑全套 CTest。

构建/验证日志：`tmp/q4e-single-phase-build.log`、`tmp/q4e-single-phase-cpu.log`、`tmp/q4e-single-adapt-tests.log`、`tmp/q4e-single-final-build.log`、`tmp/q4e-single-ple-test.log`、`tmp/q4e-single-ple-xmx-test.log`。

PLE 权重、投影数学、KV/GDN/PLE recurrent state 精度不变；其异步 staging 仍按最大容量分配，本轮只改计算工作区。完整 32K 输入、更高容量和复杂仓库任务不在此次验收范围。长上下文 MTP 的 sparse draft 与 T1 对齐问题仍未解决。
