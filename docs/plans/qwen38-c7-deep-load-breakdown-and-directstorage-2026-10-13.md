# C7 深挖：加载阶段分解 + DirectStorage 可行性评估（2026-10-13）

## 1. 加载分解（A770 本机，`AILA_Q4E_LOAD_TIMING=1`，`tmp/q4e-strata-c7/startup_only.py`）

总 startup 137-144 s（该次开了逐层日志；无日志基线 116-119 s）。阶段：

| 阶段 | 时间 | 说明 |
|---|---:|---|
| weights+embedding prep | 6.5 s | safetensors 引用建立（**不读数据**：测得 0.000 s/layer） |
| adaptive_cache.Allocate+InitializeResidency | 0.1 s | 纯记账 |
| **host_arena.Allocate** | 5.1-13.0 s | 51 GiB 大页 VirtualAlloc（含 OS 清零） |
| **host_arena.LoadFromModelWeights** | **120.6 s** | 逐层 2.4-2.7 s，48 层，均匀（无升温趋势） |
| ├─ 4 段 memcpy（RAM→大页 arena） | **95.7 s**（2.0 s/层） | 源是 mmap 懒映射 → 拷贝时触发缺页读，实测 **~0.43 GB/s** |
| └─ GPU 归属专家上传（逐专家 staging+wait） | ~24 s（0.5 s/层） | 每层 ~58 专家，**逐个 host memcpy→H2D→wait** |
| FinishInitialLoad + runtime buffers | <0.05 s | — |

对照（同机、同盘、`tmp/q4e-strata-c7/nvme_read_bench.py`，4 MiB 块）：

| 读法 | 带宽 |
|---|---:|
| 单线程顺序读 | 1.97-2.19 GB/s |
| 8 线程 | 5.1-7.8 GB/s |
| 16 线程 | **8.6-9.8 GB/s** |

**结论：fill 不是盘受限，而是"mmap 缺页驱动的读 + 逐专家串行上传"两处受限**
（0.43 GB/s 只有单线程裸读的 1/5）。这解释了为什么 `ARENA_FILL_THREADS` 16 线程只赢
10-12%：多线程仍在同一条 mmap 缺页路径上争用。

### 1.1 追加实验：并行"预触"预热页（`AILA_Q4E_ARENA_FILL_PREFETCH`，默认 0）

在每层拷贝前用 N 线程按 4 KiB 步长扫过该层源区间，强制 OS 并行取页：

| 配置（无逐层日志） | startup |
|---|---:|
| 基线 | 116.6 / 118.7 s |
| `FILL_PREFETCH=8` | 108.1 s（−7~−9%） |
| `FILL_PREFETCH=16` | **103.8 s（−11~−13%）** |
| `ARENA_FILL_THREADS=8/16`（另一机制） | 105.5 / 102.9 s |

带 `LOAD_TIMING=1` 的分解（prefetch=16，共 99.1 s）：

| 项 | 时间 | 速率 |
|---|---:|---:|
| **源预触（16 线程，同文件 mmap 缺页）** | **62.4 s** | **0.66 GB/s** |
| arena RAM 拷贝（页已驻留） | **8.6 s** | **6.7 GB/s** ← memcpy 本身没问题 |
| GPU 上传（逐专家 wait） | ~24 s（由层内差值） | ~2.4 GiB/s |

**根因确认**：并行缺页读**同一个文件映射**被 Windows section 串行化（0.66 GB/s，与线程数无关）；
而"不同文件并发读"可到 8.6 GB/s。进一步实测**同一文件 16 线程显式读**：3.2 GB/s
（`nvme_read_bench.py --single-file`，0.87 GB 分片、含缓存成分），单线程 1.9-2.2 GB/s。

### 1.2 显式并行读（已实现，**opt-in**：`AILA_Q4E_ARENA_READ_THREADS`，默认 0）

实现：默认 0 = 原 mmap 路径不变；>0 = 用 N 个线程对每层的 4 个源区间做显式
`CreateFileW + SetFilePointerEx + ReadFile`（每线程独立句柄、4 MiB 读块）读入一个可复用的
bounce 缓冲，随后沿用原有逐专家拷贝；每层 working-set trim **保持默认开启**（内存曲线与基线一致）。

| 场景 | 旧 mmap 路径 | 显式读 8 线程 | 显式读 16 线程 |
|---|---:|---:|---:|
| 冷缓存（长时间未读模型后） | 116.6 / 118.7 s | 82.8 s（−29~−30%） | **63.8-64.4 s（−45~−46%）** |
| 热缓存（连续开发跑） | **68.0 s** | — | 82.2 / 83.3 s（**更慢**） |
| 峰值内存（min available / worker RSS） | 2.13 GiB / 5.70 GiB | — | 1.14-2.97 GiB / 5.72-6.43 GiB |

- 热缓存下 mmap 缺页由缓存服务，显式读的额外 bounce 拷贝反而更慢；冷缓存下 mmap 的单文件
  section 串行（0.66 GB/s）是硬伤，显式 16 线程可到 ~1.2-3 GB/s。
- **结论**：该路径是冷启动优化，默认保持 mmap（不改变默认行为与内存曲线）；需要冷启动加速时
  `AILA_Q4E_ARENA_READ_THREADS=16`。峰值内存没有改善（两条路径都会把 arena + 模型页带入内存）。
- 未落地：`FILE_FLAG_NO_BUFFERING`（对齐窗口直读、零系统缓存占用）——首版在"边缘 slice 回退"时
  句柄未重开导致整层读失败并退回 mmap（170 s + 缓存压力），修复版（对齐窗口 + 文件尾小回退）
  未继续验证（用户指示先停止压缩 startup）；如需"冷启动快且峰值低"，这是首选续作。

### 1.3 尚未落地的后续项

1. **`FILE_FLAG_NO_BUFFERING` 对齐窗口直读**：见 §1.2 末条（修好回退句柄问题的版本值得再验一次）。
2. **GPU 上传流水线化**：一轮 ring(6 槽) 实验未带来收益（64.3 s vs 63.8 s，逐专家 DMA 延迟不是
   当时的瓶颈）；若要再动，先分别计量"staging memcpy"与"wait"两段。
3. 两者合计上限仍指向 startup ≈ 40-55 s（冷缓存口径）。

优先级顺序（按性价比）：
（见 §1.2；`FILL_PREFETCH` 与 `ARENA_FILL_THREADS` 两个 opt-in 已入库，先各自省 10-13%，
但真正的解法是显式读。）

## 2. DirectStorage 可行性（回答"NVMe 直读 GPU"）

**环境事实**：
- 本机 **没有 DirectStorage 运行时**（全盘无 `dstorage.dll`/`dstoragecore.dll`）；Windows SDK
  （`D:\Windows Kits\10`，含 d3d12.h）**不含** `dstorage.h` —— 头/导入库只随 NuGet 包
  `Microsoft.Direct3D.DirectStorage` 分发（运行时 DLL 亦在该包 redist 内，可再分发）。
- 盘：C: = "LASTING in NVMe i35 1TB"（NVMe），并行读上限 ~8.6-9.8 GB/s（上表）。

**DS 能带来什么 / 不能带来什么**：

| 形态 | 可行性 | 对当前瓶颈的价值 |
|---|---|---|
| NVMe → pinned RAM（DS 到 MEMORY 目标） | 高（需装 NuGet 运行时） | 替代 mmap 缺页读；但**普通并行 ReadFile 已可到 8.6 GB/s**，DS 的额外价值主要是低 CPU 占用与更深队列——对本项目（一次性 fill）收益有限 |
| NVMe → **VRAM 直读**（DS 到 D3D12 resource） | 未知（需 D3D12↔L0 共享内存 interop 探针） | 理论上省掉 RAM 中转；但 PCIe/RAM 两跳都是 ~9.8 GB/s，**总时间不会更低**，唯一收益是省 RAM（当前 51 GiB arena 存在的意义正是 RAM 已很紧张） |

**D3D12↔L0 interop 的风险**：Aila 的显存是 L0 USM；DS 的 GPU 目标必须是 D3D12 资源。
两条候选路径都没有现成支持：
- `D3D12_HEAP_FLAG_SHARED` + `CreateSharedHandle` → `zeMemOpenIpcHandle` 导入为 USM（跨 API 共享
  句柄，Intel Windows 驱动未承诺支持，需要写探针验证）；
- 反向（L0 分配 → D3D12 打开）同样无承诺。
本仓没有 DS 头/运行时，先验证 interop 需要额外引入 NuGet 依赖与一次 API 探针。

**建议顺序（证据驱动）**：
1. 先做 §1 的 1+2（显式批量读 + 上传流水线）——不需要新依赖，预期收益最大。
2. 若批量读后仍受"读"限制（例如源必须保持 mmap 语义），再评估引入 DS（NuGet）做
   NVMe→pinned RAM 的读路径替换。
3. 只有在"RAM 成为硬约束（例如 >16 GiB 模型的 GPU 常驻集）"时，才投入 D3D12↔L0 interop 探针
   去验证 NVMe→VRAM 直读；届时以"DS 直读 vs 现成 pread+H2D 两跳"的实测对比做门。

## 3. 复现

```powershell
python tmp/q4e-strata-c7/startup_only.py t-diag AILA_Q4E_LOAD_TIMING=1   # 阶段分解（看 *-worker.log）
python tmp/q4e-strata-c7/nvme_read_bench.py --gib 8 --threads 16 --offset 125829120
```
