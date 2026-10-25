# Aila Prefill 优化接手文档

> 状态：**未提交**（用户要求暂不 commit）。写完直接留在工作区即可。
> 生成时间：2026-10-25
> 上一阶段目标：prefill 达到 400 tok/s —— **未达成**，当前 357.1 tok/s，差 1.12×。
> 最新提交：`3687c28`

---

## 1. 目标与当前实测状态

**目标**：prefill（prompt 预填充）吞吐达到 400 tok/s。

**实测（出厂默认配置，无任何覆盖）**：

| prompt 长度 | 墙钟 | ms/token | tok/s |
|---:|---:|---:|---:|
| 2026 | 11,695 ms | 5.77 | **173.2** |
| 4952 | 16,545 ms | 3.34 | **299.3** |
| 7878 | 22,059 ms | 2.79 | **357.1** |

**起点**：本目标开始时为 150.3 tok/s（2026 token）。8K 长度相对提升约 +138%。

> 注意：更早文档里出现过"prefill 16 tok/s"的说法，那是把 **decode A/B runner 的墙钟**误当成 prefill 造成的，已被推翻（见第 10 节"已推翻的结论"）。

**要达标还差**：22.06 s → 19.69 s，即再省 **2.37 s**。

---

## 2. 硬件与软件环境

| 项 | 值 |
|---|---|
| GPU | Intel Arc A770 Graphics，16 GB VRAM（引擎自报 15,933 MB；WMI 的 `AdapterRAM` 只报 4 GB，不可信） |
| GPU 驱动 | 32.0.101.9150（历史记录里出现过 8724，升级过） |
| 平台 | 关键：**Haswell-EP 只有 PCIe 3.0**，所以 A770 跑在 PCIe 3.0 x16（理论 15.75 GB/s） |
| CPU | Intel Xeon E5-2696 v3 @ 2.30 GHz，18 核 / 18 线程，AVX2（无 AVX-512） |
| 内存 | 63.9 GiB（4 × 16 GB DDR3-1866，四通道；理论 59.7 GB/s，实测可持续 ~45–52 GB/s） |
| OS | Windows 11 专业版 10.0.26200 |
| 工具链 | Intel oneAPI 2026.1（DPC++ / oneDNN 3.11 / oneTBB / Level Zero SDK 1.33.1）、VS 2022+ (MSVC v143/v145)、CMake ≥3.24、Ninja、PowerShell 7 |
| VTune | `C:\Program Files (x86)\Intel\oneAPI\vtune\latest\bin64\vtune.exe`（2026.2，**不在 PATH**） |
| Level Zero SDK | `C:/Program Files/LevelZeroSDK/1.33.1`（头文件在 `include/level_zero/`，库 `lib/ze_loader.lib`） |

**模型**：`C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2`
- Qwen3.8-Flash MoE，NF4 量化，48 层 = 36 GDN（线性注意力）+ 12 QSA（块稀疏全注意力，head_dim 256）
- 512 专家/层，top-10，专家权重合计 60 GB（512 × 48 × 2.46 MB）
- 隐藏维 2560，MoE 中间维 640，hc_dim 10240（hyper-connection）

**为什么这些硬件事实重要**：
- PCIe 3.0 x16 是 prefill 的物理天花板。专家权重 60 GB、单次 pass 必须搬 ~42.6 GB（路由饱和），以 arena 实测 8.3 GB/s 计 ≈ 5.1 s。
- 宿主 DDR3 带宽 ~47.8 GB/s，决定了任何 host 侧拷贝都是 6.4 GB/s 量级。
- 18 核 Haswell 决定 CPU 侧无法承担专家计算（`experts_host_ms` 只有几百 ms，专家算力在 GPU）。

---

## 3. 仓库结构与关键文件

```
E:\RiderProjects\Aila\
├─ build.ps1                     官方构建入口（pwsh -File build.ps1 -BuildDir build-refactor）
├─ CMakeLists.txt                探针/测试目标定义（含 AilaH2DBandwidthProbe 等）
├─ src/
│  ├─ core/Context.hpp           设备/队列/分配器；唯一主队列 q_ 是 in_order()
│  ├─ memory/LargePages.cpp      VirtualAlloc(MEM_LARGE_PAGES) 分配器
│  ├─ models/
│  │  ├─ Qwen4ExpBnb4Backend.cpp **prefill 主控流**（~4000 行，本阶段改动集中处）
│  │  ├─ Qwen4ExpBnb4Backend.hpp
│  │  ├─ Qwen4ExpPrefill.cpp     单层注意力/投影驱动（RunAttentionLayerTile 在 :65）
│  │  ├─ Qwen4ExpAdaptiveCache.cpp/.hpp  暂存 bank 与 DMA（StageColdExpertsAsync :622-690）
│  │  ├─ Qwen4ExpExpertCache.cpp/.hpp    常驻专家池、load_expert_to_staging :3036
│  │  ├─ Qwen4ExpWholeTokenGraph.cpp     decode 侧的 whole-token graph（与本目标无关）
│  │  └─ Qwen4ExpLayerMajor.cpp          层主序路径（默认关）
│  ├─ ops/
│  │  ├─ MoeExpertXmx.cpp/.hpp   专家 XMX/DPAS 路径、submit_dequant、oneDNN matmul 描述符
│  │  ├─ GdnChunkOps.cpp         GDN chunked prefill（T=16 在 :6，循环在 :180-196）
│  │  ├─ AttentionOps.cpp        QSA prefill/decode 注意力（6000+ 行，多份相似实现，编辑极易歧义）
│  │  ├─ Bnb4BitLinear.cpp       NF4 线性层（packed_nf4_gemv_* / joint matrix）
│  │  └─ QsaIndexerOps.cpp       QSA 索引器（block scores / topk / compress）
│  └─ utils/Qwen38A770Preset.hpp **性能 preset**（默认值实际生效地，见第 5 节）
├─ tests/ops/
│  ├─ H2DBandwidthProbe.cpp      H2D 带宽 / 双队列重叠 / 大页 DMA / L0 导入 探针
│  ├─ QsaAttentionProbe.cpp      QSA decode 注意力微探针（cell 数走 argv）
│  └─ KernelSubmitProbe.cpp      裸提交成本探针
├─ tools/
│  └─ prefill-bench.ps1          **prefill tok/s 记分板**（本阶段新增，判定的唯一依据）
├─ docs/plans/
│  └─ qwen38-r1-token-graph-round1-characterization-2026-10-23.md
│                                主记录文档，3,814 行 / 195 KB，第 44–83 节是本阶段
└─ tmp/                          工作产物（gitignored）
   ├─ build-aila.bat              增量构建包装脚本（见第 4 节，绕开工具链问题）
   ├─ prefill-bench/              bench 输出、prompt-*.json、vtune 结果
   └─ q4e-token-graph-default-20261021/run_ab.ps1   decode A/B runner（**不要用于 prefill**）
```

---

## 4. 构建与验证命令（可直接复制）

### 4.1 增量构建

```powershell
cd E:\RiderProjects\Aila
cmd /c "tmp\build-aila.bat Aila.exe"                 # 只构建 CLI
cmd /c "tmp\build-aila.bat AilaH2DBandwidthProbe"    # 只构建探针
cmd /c "tmp\build-aila.bat"                          # 全量（脚本内已分支处理空参数）
```

**为什么用 `tmp\build-aila.bat` 而不是直接 cmake**：这台机器的 LIB 路径组合有坑。
`vcvars64.bat` 单独用会丢 Intel 编译器库（LNK1104 `libircmt.lib`），只把 Intel lib 加到 LIB 又会丢
Windows SDK 库（LNK1181 `windowscodecs.lib`）。脚本的做法是：先调
`C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat`，
再把 `C:\Program Files (x86)\Intel\oneAPI\compiler\2026.1\lib` 前置到 LIB。

脚本内容要点（若丢失可重建）：

```bat
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
set "LIB=C:\Program Files (x86)\Intel\oneAPI\compiler\2026.1\lib;%LIB%"
cmake --build build-refactor --target %*
```

**遇到 LNK1104 / LNK1181 / LNK2019 时**：先怀疑 toolchain 路径或 Level Zero SDK 升级（历史上
1.28.2→1.33.1 曾把 `CMakeCache.txt` 的 `AILA_LEVEL_ZERO_INCLUDE_DIR` / `AILA_LEVEL_ZERO_LOADER_LIBRARY`
指到已删除的目录），而不是怀疑源码。

### 4.2 测试

```powershell
ctest --test-dir build-refactor -L 'cpu|abi' --output-on-failure   # 约 176 s，应全绿
```
`AilaInferenceCApiGpuTests`（#66）会 skip，属正常。

### 4.3 prefill 记分板（判定目标是否达成的唯一依据）

```powershell
pwsh -NoProfile -Command "& 'tools/prefill-bench.ps1' -Tokens 2026,4952,7878 -Chunk 8192"
```

输出 tok/s、ms/token，并用线性拟合给出**边际 ms/token** 与每次运行的**固定成本**。
`-Chunk` 对应 `AILA_Q35_PREFILL_CHUNK`。

**注意**：脚本内部会构造约 size/19.3 句重复文本作为 prompt，引擎会打印实际 token 数
（`[GenerateMessages] Full prefill: N tokens`），吞吐按 N 计算。

### 4.4 单次直接跑（调试用）

```powershell
$env:UR_L0_USE_COPY_ENGINE='1'; $env:SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE='1'
$env:UR_L0_USE_IMMEDIATE_COMMANDLISTS='0'; $env:SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS='0'
$env:SYCL_UR_USE_LEVEL_ZERO_V2='0'; $env:SYCL_CACHE_PERSISTENT='1'

& build-refactor\Aila.exe -m 'C:\QwenModels\Qwen3.8-Flash-Next-BNB-NF4-v2' `
  --performance-preset qwen38-a770 --max-seq 12288 --greedy --thinking-budget 0 `
  --max-tokens 1 --messages-json tmp\prefill-bench\prompt-8k.json `
  --no-stream --chat-output-json --log-level info
```

- `--max-tokens 1` 保证只测 prefill。
- `--max-seq` 必须 ≥ prompt 长度，否则会被截断。
- 进程墙钟 ≈ 183 s（模型加载） + 生成时间；**只看 `GenerateMessages` 行**，不要看进程墙钟。

### 4.5 探针

```powershell
& build-refactor\AilaH2DBandwidthProbe.exe 256 2400
# 参数：总 MB、分块 KB。输出：分块带宽曲线、单大块带宽、轮转目标、
#       大页源（ctx.memcpy_h2d / queue.memcpy）、arena 导入前后入队耗时、
#       双队列 overlap 判定
```

```powershell
& build-refactor\AilaQsaAttentionProbe.exe 2052   # QSA decode 注意力，cell 数
& build-refactor\AilaKernelSubmitProbe.exe        # 裸提交成本
```

### 4.6 VTune（observer-free 归因，prefill-only）

```powershell
$v='C:\Program Files (x86)\Intel\oneAPI\vtune\latest\bin64\vtune.exe'
& $v -collect xpu-offload -result-dir tmp\prefill-bench\vtune-x -- build-refactor\Aila.exe <同上参数>
# 报告必须对**结果目录的副本**执行，否则报 "Cannot re-finalize a read-only result"，
# 直接在原目录重跑会报 "Insufficient permissions ... sqlite-db"
Copy-Item -Recurse tmp\prefill-bench\vtune-x tmp\prefill-bench\vtune-x-copy
& $v -report hotspots -r tmp\prefill-bench\vtune-x-copy -group-by computing-task `
     -format csv -csv-delimiter tab -report-output tmp\prefill-bench\hot.csv
```

CSV 是 **TAB 分隔**，时间列单位是**秒**；`Import-Csv -Delimiter "`t"` 才能正确解析。
`gpu-hotspots -knob characterization-mode=...` 在这台机器上不可用（53 分钟级），只用 `xpu-offload`（~4 分钟）。

---

## 5. 配置默认值：以 `Qwen38A770Preset.hpp` 为准（重要陷阱）

`src/utils/Qwen38A770Preset.hpp` 在模型/队列构造前用 `_putenv_s` 写入一批环境变量，
**只在该变量当前为空时写入**。因此：

- 用户在命令行前显式设置的变量**优先**（preset 不覆盖）；
- 代码里的 `read_int("X", default)` 只有在 preset 也没写 X 时才生效。

当前 preset 内容（`--performance-preset qwen38-a770`）：

```
AILA_Q4E_PREFILL_XMX          = 1
AILA_Q4E_DIRECT_PREFILL_IO    = 1
AILA_Q4E_DEMAND_MICROBATCH    = 32
AILA_Q4E_PHASE_SCRATCH        = 1
AILA_Q4E_HOT_RAM_MB           = 32768
AILA_Q4E_EXPERT_READER        = readfile
AILA_Q35_PREFILL_CHUNK        = 8192      <- 本阶段由 512 改来
AILA_Q4E_INLINE_CHECKPOINT    = 1
AILA_Q4E_STATIC_PINNING       = 0
AILA_Q4E_GROUPED_XMX_MIN_ROWS = 0
AILA_Q4E_GDN_CHUNK_MODE       = 0         <- **坑，见 5.1**
AILA_Q4E_SMALL_ROWS_DECODE    = 0
AILA_Q4E_SMALL_ROWS_HC        = 0
```

### 5.1 ⚠️ 已知未修：`GDN_CHUNK_MODE` 的默认值被 preset 吃掉

- 实测（显式设环境变量做 A/B）：mode 0 = 311.7 tok/s，**mode 1 = 319.7 tok/s（+2.6%）**，mode 2 = 317.7。
- 我把 `Qwen4ExpBnb4Backend.cpp:594` 的代码默认改成了 `1`。
- **但 preset 显式写入 `0`，所以代码默认值根本不生效**，出厂实际仍跑 mode 0。
- **修复**：把 `src/utils/Qwen38A770Preset.hpp` 里 `{"AILA_Q4E_GDN_CHUNK_MODE", "0"}` 改成 `"1"`，
  然后复测。这是本阶段留下的**最便宜的一个待办**（预期 +2.6%，约 5 s 工作量）。
- 顺带解释：第 68 节记录的"322.9 tok/s"很可能不是 mode 1 带来的，而是运行波动——复测时要注意。

---

## 6. prefill 管线控制流（从 CLI 到内核）

### 6.1 顶层切片

`EngineGeneration.cpp` 按 `AILA_Q35_PREFILL_CHUNK`（现 8192）把 prompt 切片，逐片调用
backend 的 forward。日志形如：

```
[Q4E PrefillSlice] start=0 count=2048 checkpoint_after=0
```

对 2108 token 的 prompt，在 chunk=8192 下是 `2048 + 56 + 4`；在旧的 512 下会切成 5–6 片，
**每片都要重付一次专家暂存与权重读取**，这是 `PREFILL_CHUNK` 从 512 提到 8192 的收益来源。

### 6.2 每层顺序（`Qwen4ExpBnb4Backend.cpp` 主循环）

```
for 每层 L:
    HC(attn_hc)                      hyper-connection，BF16，hc_dim=10240 / lowrank=320
    if layer.is_linear:  GDN 路径     RunAttentionLayerTile -> try_linear_delta_prefill_chunked
    else:                QSA 路径     每 tile 三发：block scores / topk / 稀疏注意力
    HC(mlp_hc)
    bf16_moe_front_fused             router 打分（hidden -> 512+1）
    router topk (512 专家, 单 WG)    + 可选 classify
    【专家执行】                     见 6.3
    shared expert (gate_up + down)
    moe_ordered_accumulate_down_rows 合并专家输出到 y
```

### 6.3 专家执行段（`RunGroupedExperts`，约 `:2007`）

```
const int top_k = cfg_.num_experts_per_tok;      // 10
if (prefill_expert_xmx_ && count >= GATE_ROWS && <call-shape 条件>)
    → XMX/DPAS 路径（banded，16 专家/带，oneDNN batched matmul）
else
    → ALU 路径（每专家独立 GEMM）
```

- `GATE_ROWS` = `AILA_Q4E_PREFILL_EXPERT_XMX_GATE_ROWS`，代码默认 **4096**（本阶段新增）。
  小 M 时 DPAS 亏 13%，大 M 时赚 26%，交叉点在 2995~4952 之间。
- 外层 chunk 循环（`:2013` 起）：

```
for c in 0..num_chunks-1:                        # num_chunks = ceil(unique_experts / 16)
    batch = staged_window[c % staging_banks]
    last = RunGroupedExperts(..., batch.ready_event)     # 依赖拷贝事件，不等宿主
    adaptive_cache_.RecordStagingConsumer(bank, last)
    stage_chunk(c + lookahead)                            # 预取下一 chunk
```

- `staging_banks` 默认 2（`AILA_Q4E_PREFILL_STAGING_BANKS`），`lookahead = banks-1`。
- 每 chunk 16 个专家（`kMaxExpertsPerChunk`，编译期常量，`src/ops/MoeExpertXmx.cpp:21`）。

### 6.4 暂存（`Qwen4ExpAdaptiveCache::StageColdExpertsAsync`，`:622-690`）

三个分支：

```cpp
if (parallel_host_copy && count>1 && copy_executor)  → 执行器多线程拷贝到 pinned staging
else if (!direct_arena)                              → 逐专家 std::memcpy 到 pinned staging
if (direct_arena)                                    → 直接从大页 arena 发 queue::memcpy
```

`direct_arena` 由 `AILA_Q4E_STAGE_DIRECT_ARENA` 控制（本阶段改为代码默认 **true**），
并且**必须在调用点传第 8 个参数**（本阶段修复：`Qwen4ExpBnb4Backend.cpp:2005` 之前漏传，
导致该路径长期不生效）。

### 6.5 队列结构（决定能否重叠）

- `Context` 只有**一个主队列**，属性为 `sycl::property::queue::in_order()`，命令严格串行。
- 另有 `copy_q_`（`Qwen4ExpExpertCache.cpp:769-782`，in-order + immediate command list），
  暂存 DMA 走在上面，靠 `batch.ready_event` 做依赖。
- 探针已证明**双队列可以完全重叠**（kernel 155.7 ms + copies 37.9 ms，串行 193.6 ms，并发 154.5 ms）。

---

## 7. 测量工具与"哪些数字可信"

### 7.1 可信

| 指标 | 来源 | 说明 |
|---|---|---|
| tok/s、ms/token | `tools/prefill-bench.ps1` / `GenerateMessages` 行 | 唯一判定依据 |
| `h2d_mb`、`staged`、`hits`、`misses`、`fill_ms`、`bank_wait_ms` | `[Q4E AdaptiveWork]` 行 | 与独立探针自洽（如 `fill_ms=284` ↔ 1.84 GB / 6.4 GB/s） |
| per-kernel 设备时间 | VTune `xpu-offload` CSV | observer-free |
| 探针数值 | H2D / QSA / submit 探针 | 秒级迭代 |

### 7.2 **不可信**（务必别再用）

- `submitted_ms`、`full_host_ms`、`linear_host_ms`、`experts_host_ms`、`hc_host_ms`：
  **这些字段会超过整次运行的墙钟**（例：20 token 的运行墙钟 1,485 ms，而 `submitted_ms=117,140`），
  说明它们是按层累加/GPU 时间戳差，**不是 elapsed 毫秒**。第 53–55 节曾据此推出"QSA 占 76%"等结论，已作废。
- VTune `gpu-hotspots` 模式的**时间**（replay 放大 15×，只有 geometry/counter 可用），且该模式本身不可用。
- NodeProf 的 in-code marks（barrier 间跨度，短 kernel 放大 2.5–5×；且默认令牌图路径下无量）。

### 7.3 可用的自建仪器（默认关闭）

| 开关 | 作用 |
|---|---|
| `AILA_Q4E_PREFILL_EXPERT_TIMING=1` | 每 chunk 打印 `gemm_host_ms` / `stage_host_ms`（宿主墙钟分解） |
| `AILA_Q4E_WORK_SUMMARY=1` | 打印 `AdaptiveWork` 行（注意 7.2 的坑） |

### 7.4 当前 observer-free 归因（8K prompt，25.4 s 设备时间，698 行）

| kernel | 时间 | 实例数 | 备注 |
|---|---:|---:|---|
| `zeCommandListAppendMemoryCopy` | 6.48 s | 37,465 | 平均 0.17 ms |
| `packed_nf4_gemm_bf16_joint_matrix_wide` | 3.18 s | 97 | 专家 GEMM，32.8 ms/次 |
| `solve_chunk` | 3.08 s | 17,800 | GDN chunked，0.17 ms/次 |
| `attention_prefill_qsa_batched_impl` | 2.50 s | 84 | QSA |
| `zeCommandListAppendMemoryCopy`（第二行） | 0.77 s | 3,817 | |
| `product_f32` | 0.74 s | 17,632 | GDN chunked，0.04 ms/次 |

**关键观察**：当前配置下 `h2d_mb = 1,851 MB`、`staged = 766`，即专家只暂存了 1.85 GB，
而拷贝总数是 37,465 —— 说明**约 36,700 次是小拷贝**（≈ 6.2 s），拷贝成本主要来自"大量小传输"，
而非"流送 42.6 GB 专家权重"。小拷贝的确切来源尚未定位（候选：ALU 路径 per-segment 上传
`Qwen4ExpBnb4Backend.cpp:2142-2146`；但 XMX 路径本不该走那里）。

---

## 8. 已落地的收益（全部在默认路径，用户无需配置）

| 改动 | 位置 | 效果 |
|---|---|---|
| `PREFILL_CHUNK` 512 → 8192 | `Qwen38A770Preset.hpp` | 8K：+11~25%（视长度） |
| XMX/DPAS 按行数 ≥4096 门控（新增 `..._GATE_ROWS`） | `Qwen4ExpBnb4Backend.cpp:~2015` | 2K：151→171；8K：238→299 |
| `QSA_QUERY_TILE` 64（被 4 MB 预算钳死）→ 1024（16 MB 预算） | `Qwen4ExpBnb4Backend.cpp:~1276` | +3.9% |
| `GDN_CHUNK_MODE` 代码默认 0 → 1 | `Qwen4ExpBnb4Backend.cpp:594` | +2.6%（**但被 preset 覆盖，未生效**，见 5.1） |
| `direct_arena` 参数补传到 prefill 暂存 | `Qwen4ExpBnb4Backend.cpp:2005` | **+10.8%** |

---

## 9. 已排除的路径（避免重复投入）

**配置类（全部实测中性或更差）**：
`MIN_M`（是填充带参数不是门控；设 4096 会劣化到 39.6 s）、`LARGE_PAGE_PIPELINE`、`GDN_HISTORY_FUSED`、
`LAYER_MAJOR`、`RESIDENT_PREFILL`、`FUSED_PREFILL`、`XMX_ROWS` 2048/8192、
`STAGING_SLOTS` 32/64、`PREFILL_STAGING_BANKS` 3/4、`VRAM_RESERVE_MB` 2048/1024、`W_F16`、
`DIRECT_PREFILL_IO=0`（346.1）、`PREFILL_XMX=0`（291.0）。

**实现类（写过后回退）**：
- 权重布局改成 `[K][N]`（`W_KN`）：oneDNN 自己会重排，完全中性。
- 一个 chunk 的 16 次 arena 拷贝合并成 1 次：351.7 vs 357.1，更差，已回退。
- 把 `stage_chunk` 放 worker 线程（加互斥锁保护 `adaptive_cache_`）：356.3 vs 355.4，中性，已回退。
- GDN 块大小 `T=16`：T=32 慢 13%（求解是 O(T²)），T=64 显存不足。**16 已是权衡最优。**

**已推翻的结论（别再复活）**：
- "prefill 16 tok/s"（是 runner 假象，真实起点 150.3）。
- "42.6 GB 专家流送是主项"（当前配置只有 1.85 GB）。
- "大页直接 DMA 慢 1.7×"（那是 `ctx.memcpy_h2d` 入口；`queue.memcpy` 是 8.3 GB/s，比 pinned 7.1 更快）。
- "`submitted_ms`/`*_host_ms` 可作分解"（超过墙钟，不可用）。
- "QSA 占 prefill 76%"（基于上面这些坏字段）。

---

## 10. 下一步建议（按证据强度排序）

### 9-A 立刻可做的两件小事

1. **修 preset 的 `GDN_CHUNK_MODE`**（第 5.1 节）：预期 +2.6%，5 分钟。
2. **给 36,700 次小拷贝加计数**：在 `Qwen4ExpBnb4Backend.cpp:2142-2146`（ALU per-segment 上传）与
   `:2292-2295`（XMX plan 上传）各加一个 `static std::atomic<uint64_t>` 计数，prefill 结束打印。
   若命中 36k，则把它们合并成每层一次上传，**预期回收 3–5 s，直接达标**。这是当前最高性价比的探索。

### 9-B Argon 的架构建议（已评估，置信度高但工作量大）

| 排名 | 改动 | 预期 | 置信度 |
|---|---|---|---|
| 1 | 把 GDN `solve_chunk` 跨 493 个 chunk 批成 36 次启动 + 跨 chunk 扫描融成每 head 一个持久内核（状态常驻寄存器/SLM） | −2.2~−2.9 s | 90% |
| 2 | 用 `zexDriverImportExternalPointer` 在初始化时把大页 arena 一次性导入 L0 | −1.8~−2.8 s | 85% |
| 3 | pre-router 投机预取冷专家（在注意力/GDN 运行时用 `copy_q_`） | −1.5~−2.4 s | 75% |
| 4 | cache-aware tail routing（近似，需困惑度验证） | −0.9~−1.4 s | 65% |

Argon 明确否掉两条：无损减少 42.6 GB（coupon-collector：>95% 专家至少被一个 token 选中）、
多 pass / 按层排序（chunk 8192 下每层已只访问一次，重新分段数学等价且开销更大）。

### 9-C 已由探针验证可行（本阶段最后一项，结论新鲜）

```
arena, before import: 109 copies, enqueue 2.6 ms (23.7 µs/copy)
zexDriverImportExternalPointer: OK (0x0)
arena, after import:  109 copies, enqueue 0.4 ms ( 4.0 µs/copy)     ← 6×
```

即 Argon Rank 2 的**机制成立**：未导入的大页指针对 L0 是"外来指针"，每次 `queue::memcpy` 都要做
host-pointer 注册；一次性导入后入队成本降 6 倍。探针代码在
`tests/ops/H2DBandwidthProbe.cpp`（含 `zeDriverGetExtensionFunctionAddress` 取函数指针的写法，
注意 L0 1.33 的签名是 `(driver, name, void** pfn)`）。

**落地方式**：在 `LargePages.cpp` 分配 arena 后（或 `Qwen4ExpExpertCache.cpp:3592` 之后）
拿到 `ze_driver_handle_t` → `zexDriverImportExternalPointer(driver, arena, bytes)`，
并在析构时 `zexDriverReleaseImportedPointer`。需要 CMake 链接 `ze_loader.lib`
（`CMakeLists.txt` 里 `AilaH2DBandwidthProbe` 已加，主库需在 `AilaSyclOptions` 或 `AilaLib` 上加）。

---

## 11. 工作区纪律（重要，请保持）

1. **一次只改一个结构变量**，改完立即构建 + 单次 A/B。
2. **不要相信跨运行的绝对数值**：这台机器漂移明显（同一配置 unrolled 曾从 49 ms 漂到 55 ms）。
   A/B 结论只取**同一次会话内的配对差值**。
3. **效果 < 1 ms 的改动不可判定**，不要为此反复跑；要么合成一组一起测，要么放弃。
4. **实测更差的改动必须回退**（`git checkout -- <file>`），不要留在默认关闭的 flag 后面
   ——除非该 flag 有明确诊断价值。本阶段回退了 4 个这样的改动。
5. **写代码前先查"参数有没有传下去"**。本阶段最大的单个收益（+10.8%）来自发现 prefill 调用点
   漏传第 8 个参数；在此之前连续三轮 flag 试探全部落空。
6. **配置项优先级**：显式环境变量 > preset（`Qwen38A770Preset.hpp`）> 代码默认值。
   改默认值前先确认 preset 没写死它。
7. 文档：每次有结论就追加到
   `docs/plans/qwen38-r1-token-graph-round1-characterization-2026-10-23.md`（编号小节），
   包括**阴性结果**，然后 commit（通常只提交文档）。
8. `tmp/` 是 gitignored，可随意放中间产物；`tools/`、`tests/` 下的新增工具应当提交。
9. 提交信息用 `perf(q4e):` / `experiment(q4e):` / `docs(q4e):` / `test(q4e):` 前缀。
10. 合理使用 opencode-go/deepseek-v4.1-flash max 子代理辅助工作，但不要让子代理跑重CPU/GPU任务，如grep/find大范围扫工作区、构建和GPU实跑，耗时任务由接手的主代理负责。
11. `ctest` 只有在固化经过 A/B 验证的变量为生产配置时才跑相关的单元测试。

---

## 12. 已知陷阱清单

### 12.1 测量
- `Aila.exe` 的**进程墙钟 ≈ 183 s 模型加载 + 生成时间**，只能看 `GenerateMessages` 行。
- 2000 token 以上的 prompt 才进入 token graph（decode 侧），prefill 不受影响，但**别把 decode 数字混进来**。
- VTune 报告必须对结果目录**副本**执行（见 4.6）。
- 探针跑之前**必须确认构建真的成功了**（`Get-ChildItem build-refactor\*.exe | Select LastWriteTime`），
  否则测的是旧二进制。
- 探针 CSV/输出解析注意：`Import-Csv` 要 `-Delimiter "`t"`；PowerShell 里 `[int]$x/16` 会被解析成
  `[int]($x/16)`，字符串转 int 失败会变成 0 并触发除零异常，请显式 `[int]$x` 后单独算。

### 12.2 编辑
- `edit` 工具：多个 `oldText` **重叠或嵌套会整体拒绝**；`oldText` 必须逐字符匹配（缩进、`const` 位置）。
- **锚点吞并**：如果 `oldText` 跨过一行 `#include` 或 `parallel_for(...) {` 而 `newText` 没有重新写出它，
  该行会被静默删除。插入 include 时要锚在相邻行上，并在改完后 `read` 回来确认。
- `AttentionOps.cpp`（6000+ 行）里有 2/3/4/5 处重复片段，多行 `oldText` 经常报 "Found N occurrences"；
  可靠办法是**用唯一单行内容定位后再按行号范围重写**，并核对行数变化。
- `Qwen4ExpBnb4Backend.cpp` 里 `for (int i=0;i<count;++i) {` 这类短行也重复，需要带上下文。
- PowerShell 的 `-replace` 用 `[regex]::Escape` 且字符串里含 `` `r`n `` 时**不会匹配文件行尾**，
  结果是"部分生效、部分不生效"。安全做法：`$t.Replace('<单行>', ...)`（不带换行、不走正则）。
- 用大括号计数脚本批量改 `AttentionOps.cpp` 曾把文件改坏，恢复方式是 `git checkout`。

### 12.3 运行时
- `Start-Process` **不能绑定 hashtable 形式的 EnvironmentOverrides**，需要写一个 wrapper `.ps1`
  构造 `$over = @{...}` 再调用 runner。
- 若以后用 A/B runner：它 gates 在 `Global\AilaGpuSingleton` 互斥量与 54 GiB 空闲内存上；
  它设的 `AILA_Q4E_HOTSET_PROFILE` 会改变专家常驻，**使 prefill 的 staged 量差 20 倍以上**，
  所以 prefill 一律用 `tools/prefill-bench.ps1`，不要用 runner。

---

## 13. 一句话交接

> prefill 现在 357.1 tok/s（8K），离目标 400 差 2.37 s。最容易的两步：先把 preset 里被写死的
> `GDN_CHUNK_MODE` 改成 1（预期 +2.6%），再给那 36,700 次小拷贝加计数找出源头（可能一次就够）。
> 若要动架构，Argon 的排序是 GDN 批式 solve + 持久扫描 > arena 导入 L0 > 投机预取冷专家。
> 别再用 decode runner 测 prefill，也别再信 `submitted_ms` 一类的字段。
