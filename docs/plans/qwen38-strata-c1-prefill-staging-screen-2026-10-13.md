# Strata C1 筛选：prefill 专家暂存（双制度 / stream-ahead）— NO-GO

日期：2026-10-13。范围：`docs/plans/qwen38-strata-design-scan-2026-10-13.md` 候选 C1 的
Aila 最小实验（探针 → 门）。结论：**transfer 不可藏**，C1 机制（更深的预取/stream-ahead）
在这台 DG2/Windows/L0 上不能兑现扫描文档里 B70 的收益。C2（copy-only 队列）成为自然的
后续候选（见 §6）。

## 1. 现有结构（对照扫描文档的"当前是否已重叠"）

- `RunAdaptivePrefill`（`src/models/Qwen4ExpBnb4Backend.cpp:1417-1468`）：按 16 专家一批走
  `StageColdExpertsAsync`：host 先把 arena 大页槽 memcpy 到 pinned `staging_host_[pp]`（bounce），
  再在独立 in-order `copy_q_`（`Qwen4ExpBnb4Backend.cpp:832-833`）上 H2D 到 ping-pong 设备 bank；
  消费方（grouped experts）依赖 `batch.ready_event`，反向由 `staging_reader_[pp]` 保护 bank 复用。
- worker 启动即设 `UR_L0_USE_COPY_ENGINE=1`/`SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE=1`
  （`src/utils/EnvUtils.hpp:179-201`，`src/worker/WorkerMain.cpp:494`）——即"copy engine"开关
  **本来就开着**；扫描文档 C2 的机制在 Aila 里已部分存在。
- 现有的跨层 prefetch 预测（`AILA_Q4E_PREFETCH_PREFILL`，Backend.cpp:1966）只在 legacy
  `expert_cache_` 路径（`predict_next` 前置 `!use_adaptive_pipeline_`），adaptive 生产管线不可达。

## 2. 测量（同一 14K fixture：`--fixture tmp/q4e-agent-corpus-v1`，chunk 4096，

先 6K 预热再测 14K；`tmp/q4e-strata-c1/run_ttft14k.py`；基线复现标准 70.5 s）

| 臂 | 14K TTFT | chunk 内 host fill | bank_wait | 说明 |
|---|---:|---:|---:|---|
| a04 基线（串行 fill） | **70.23 s** | 7.20-7.32 s/块 | 3.77-3.85 s/块 | 与标准序列 70.5 s 一致 |
| b04 oracle=1（跳过 device memcpy） | 52.71 s | — | — | 但 oracle 把路由也改了（staged 47.3→25.4 GiB/块），只作上界 |
| d2 `STAGE_COPY_PARALLEL=1`（并行 fill） | 70.80 s | **2.42-2.46 s** | **8.56-8.63 s** | fill+bank_wait 守恒（11.0 s）→ 瓶颈是 device 拷贝 |
| e1 `STAGE_DIRECT_DMA=1`（跳过 bounce，直拷大页） | 72.23 s | **0.75 s** | **0.05 s** | experts_host 11.6→4.9 s，chunk wall 不变（20.3 s） |

每 4096 块：staged 19.4-19.7k 专家 = **46.2-47.6 GiB**；`AILA_PROFILE_Q4E_STAGES=1` 的
device 分段（pos=4096）：GDN 4.07 s + full attn 2.85 s + experts 12.58 s（其中 host fill 7.28 +
bank_wait 3.76）≈ 19.8 s ≈ chunk wall。

## 3. 微探针（`AilaQwen4ExpKernelTests --queue-overlap-probe`，本机 A770）

| 场景 | 单轮耗时 | 对照 |
|---|---:|---|
| 纯 ALU burn（3.60 ms）+ H2D copy（3.48 ms） | — | 串行和 7.07 ms |
| dep（compute 等 copy） | 5.15-5.34 ms | ≈ max |
| **两 bank ping-pong（含反向依赖，等价真实结构）** | **4.04-4.18 ms** | ≈ max → ALU 负载下**能重叠** |
| 内存型 burn（256 MiB×6 pass，3.70 ms）+ 同 ping-pong | **6.70 ms** | 串行和 7.16、max 3.70 → 近似**相加** |
| 大页直拷源 | 9.05-9.23 GB/s | pinned 9.83 GB/s；D2D 117-129 GB/s |

即：队列结构本身没有串行化（ALU 负载下重叠良好），但**当 compute 是流式访存负载（真实 MoE
权重核的形状）时，H2D 与 compute 在 memory path 上相加**——这解释了 e1 去掉全部 host 侧成本后
chunk wall 完全不动。

## 4. 结论（对 C1）

1. 现管线里 host fill（7.2 s/块）与 bank 等待都被 device 工作掩盖（并行 fill 只是把时间从
   fill 搬到 bank_wait；直拷大页把两者清零后 wall 不变）→ **C1 的"stream-ahead/大环"无法产生
   收益**；14K 的 TTFT 不由 host 侧转账成本决定。
2. 每块 ~5 s（47 GiB @ 9.2 GB/s）的 H2D 是真实 device 时间，且与流式 compute 相加；把它藏起来
   需要一条真正的独立拷贝引擎路径（C2），或让 compute 不与拷贝争用 memory path。
3. oracle（完全去掉 device memcpy）的 17.5 s 上界不可达：那等于不传字节。

## 5. 顺带否证/确认

- `AILA_Q4E_STAGE_COPY_PARALLEL=1`：把 fill 从 7.2 s 降到 2.4 s/块，但 TTFT 70.80 vs 70.23 s
  （+0.8%，噪声内）——**不是 TTFT 杠杆**（与旧"staging 64 无净收益"一致）。
- `AILA_Q4E_STAGE_DIRECT_DMA=1`（本次实验实现，已回滚）：去掉 bounce、fill≈0、bank_wait≈0，
  TTFT 72.23 s（+2.9%）→ 大页直拷可行但逐专家 2.4 MiB 拷贝抵消了收益；结论=不值得留代码。
- 6K 预热 ≠ 17 token 预热：17 token 预热后 14K 仍带 ~21 s 一次性开销（91.4 vs 70.2 s）；
  新的 reduced runner 必须用 fixture 内的 6K 先热身（`tmp/q4e-strata-c1/run_ttft14k.py`）。

## 6. 对后续候选的输入

- **C2（copy-only 队列）**：机制正确且未被否证——本次只验证了"SYCL+L0 默认/pinned 路径"下的
  行为；Strata 的做法是用 `zeDeviceGetCommandQueueGroupProperties` 找 copy-only 组 + immediate
  command list + `make_queue<ext_oneapi_level_zero>`。下一步：在 KernelTests 加同一微探针的
  ordinal-1 版本；若 DG2 暴露该组且内存型负载下仍重叠 → GO，否则 C2 也 NO-GO。
- **C7/C5**：bounce fill 与 bank 等待已被证明"被设备掩盖"，优先级下调；C5 的交换旋转前提
  （pinned+mapped 全量）与既有否证冲突，仍按扫描文档处理。
- **C13（KV 预取）**：同一 memory-path 争用结论可用于预判其长上下文收益上限。

## 7. 复现

```powershell
python tmp/q4e-strata-c1/run_ttft14k.py c1-a04 --diag                 # 基线（6K 预热 + 14K）
python tmp/q4e-strata-c1/run_ttft14k.py c1-d2  --diag AILA_Q4E_STAGE_COPY_PARALLEL=1
build-refactor\AilaQwen4ExpKernelTests.exe --queue-overlap-probe      # 微探针（含 ping-pong/mem 变体）
```

日志：`tmp/q4e-strata-c1/`（`c1-{a04,b04,d0,d1,d2,e1,s0}*`、`run_ttft14k.py`）。
