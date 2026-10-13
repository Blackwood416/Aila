# Strata C2 筛选：copy-only 队列（blitter engine）— A770 上 NO-GO

日期：2026-10-13。范围：扫描文档 C2 的最小实验（探针 → 门）。

## 1. 机制与 Aila 现状

- Strata 机制：L0 v2 适配器把所有 SYCL 队列放 compute 引擎，专家 H2D 会顶住 kernel；
  解法是用 `zeDeviceGetCommandQueueGroupProperties` 找 copy-only 组（ordinal 1），在其上建
  immediate command list，再用 `sycl::make_queue<ext_oneapi_level_zero>` 包成 SYCL 队列。
- Aila 现状：worker 启动即设 `UR_L0_USE_COPY_ENGINE=1` / `SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE=1`
  （`src/utils/EnvUtils.hpp:179-201`、`src/worker/WorkerMain.cpp:494`），staging 拷贝走独立
  in-order `copy_q_`（`src/models/Qwen4ExpBnb4Backend.cpp:832-833`）。即"把拷贝放到独立队列/
  copy engine"这条已经在跑。

## 2. 证据

**（a）copy-engine 开关的真实效果**（`AilaQwen4ExpKernelTests --queue-overlap-probe`，A770，
同一二进制，仅环境变量不同）：

| 场景（单轮） | copy engine ON（worker 默认） | OFF |
|---|---:|---:|
| 纯 ALU burn（3.60-3.63 ms）+ H2D copy（3.45-3.48 ms） | 串行和 7.07 ms | 串行和 7.08 ms |
| dep（compute 等 copy） | 5.15-5.34 ms | **7.09 ms** |
| 两 bank ping-pong（含反向依赖） | 4.04-4.18 ms | **7.08 ms** |
| 内存型 burn（256 MiB×6 pass，3.70-3.73 ms）+ ping-pong | **6.70 ms** | 7.17 ms |

- 结论 1：copy engine 确实生效——ALU 负载下重叠良好（4.0 vs 7.1 串行）。
- 结论 2：**内存型负载下重叠消失**（6.70 ≈ 串行和 7.16，远大于 max 3.73）→ 拷贝与 compute 在
  memory path 上相加。这与 C1 报告的整链观测一致（去掉全部 host 侧成本后 chunk wall 不变）。

**（b）工具链**：oneAPI 2026.1 只装 `unified-runtime/ur_api.h`，**没有 `ze_api.h`/`ze_loader.lib`**
（只有驱动自带的 `C:\Windows\System32\ze_loader.dll`）。Strata 的原版做法需要 vendored L0 头 +
导入库（或手工 GetProcAddress 声明一组 L0 过程并自建 interop 类型），属于新增工具链依赖。

**（c）外部佐证**：Strata 自己在这条路的 Alchemist 分支上也只测到 **+1.4%**（A750 8GB、i915、
PCIe 4.0、10.6 GB/s），并记录"用 copy-engine memcpy 当 stager 完成标志会在 Alchemist 上挂死"，
因此 **A 系默认关闭**（扫描文档 C2 条目第 3 点引 `docs/INTEL.md:812-817`）。

## 3. 结论

**NO-GO**，不实现：

1. 该机制在 Aila 里已经以 UR 的形式默认开着，且在内存型负载（真实专家核形状）下不产生重叠；
   上 copy-only 硬件队列（若 DG2 暴露）在 Strata 的 A750 上也只有 +1.4%，并有挂死史。
2. 收益上限被 C1 报告量化：14K 每块 ~5 s 的 H2D 与 compute 相加，能藏起来才是收益；当前证据
   表明这台 DG2/Windows/L0(UR) 上藏不住。
3. 不新增 L0 头/库依赖（与 Aila 现有 UR/SYCL 栈一致），也不引入 Strata 记录的挂死风险。

## 4. 对后续候选的影响

- C13（KV 预取）：同一 memory-path 争用会对"预取藏进计算"给出同样的上限，先按此预期设计探针。
- C5（异步 swap/旋转）：swap 的 H2D 走同一路径，收益上限同样受此约束（另有 pinned 前提问题）。
- C7（填充流水线）：其价值在冷启动/加载时间，与 TTFT 的 memory-path 争用无关，不受此影响。

## 5. 复现

```powershell
build-refactor\AilaQwen4ExpKernelTests.exe --queue-overlap-probe            # copy engine ON（默认）
$env:UR_L0_USE_COPY_ENGINE='0'; $env:SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE='0'
build-refactor\AilaQwen4ExpKernelTests.exe --queue-overlap-probe            # OFF 对照
```

## 6. 追加：驱动自带 L0 SDK 的硬件级复核（2026-10-13 晚）

前几节用的是 UR/SYCL 路径；补做 Strata 原版路径（`zeDeviceGetCommandQueueGroupProperties` +
immediate command list），SDK 来自驱动自带安装：`C:\Program Files\LevelZeroSDK\1.28.2`
（include\level_zero + lib\ze_loader.lib；oneAPI 2026.1 自身只带 UR 头）。
探针：`tmp/q4e-strata-c2/l0_copy_probe.cpp`（独立 exe，cl + ze_loader.lib）。

**实测（A770，32 MiB H2D ×16，中位数）**：

| ordinal | flags | 每 32 MiB | 带宽 |
|---|---|---:|---:|
| 0 | COMPUTE+COPY | 3.436 ms | 9.76 GB/s |
| 1 | **COPY only** | 3.422 ms | 9.81 GB/s |

- A770 **确实暴露 copy-only 队列组**（这修正了"2026.1 无 L0 头/库"带来的不确定）；
- 但 copy-only 引擎与默认队列的 H2D 速率**完全相同** → 上一节的"拷贝与内存型 compute 相加"
  不是队列/引擎放置问题，而是整条 host→VRAM 通路的属性；Strata 的 A750 实测（+1.4%、有挂死）
  与此一致。
- 结论不变（**NO-GO**）。唯一未做的确认是"在 copy-only ordinal 上跑 SYCL interop 队列再测
  重叠"——需要 `zeCommandQueueCreate(ordinal=1)` + `sycl::make_queue<backend::ext_level_zero>`；
  鉴于两条 ordinal 的独立带宽一致、且 UR copy-engine 路径已证明"ALU 负载能重叠、内存型不能"，
  预计不改变结论，列为可选复核（成本：中高）。
