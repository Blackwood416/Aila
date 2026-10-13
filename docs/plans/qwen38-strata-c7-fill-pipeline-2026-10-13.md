# Strata C7 筛选：专家填充流水线 — 部分采纳（opt-in，−10~−12% 启动时间）

日期：2026-10-13。范围：扫描文档 C7 的 Aila 最小实验。结论：Aila 的 58 GiB 专家 RAM 填充
单线程 ≈0.5 GB/s（缺页/IO 受限），按 Strata 的"批量并行读取"精神做线程化后
**启动时间 116-119 s → 103-105 s**；默认路径保持不变，新开关 opt-in。

## 1. 现状与结构

- 启动 = `Qwen4ExpHostArena::LoadFromModelWeights` → 逐层 `LoadLayerDirect`：每层 512 专家 ×
  ~2.42 MiB，**单线程**逐专家 4 次 memcpy（packed/absmax × gu/dn），GPU 归属的专家再经
  `load_staging_` 逐个上传（串行 + 每次 wait）。48 层合计 ~58 GiB。
- 实测：`worker_startup_seconds`（probe 口径）**116.6 / 118.7 s**（两次连续启动，文件缓存不改变
  → 不是简单冷盘问题）；折算 ~0.5 GB/s，远低于盘/内存带宽 → 单线程缺页路径受限。
- 既有 `AILA_Q4E_FILL_THREADS/FILL_PARTS` 属于 **legacy ExpertCache**（生产 adaptive 不可达），
  本项因此需要一个真正作用于 arena 的旋钮。

## 2. 实现（`AILA_Q4E_ARENA_FILL_THREADS`，默认 1）

`LoadLayerDirect` 把"RAM 归属专家"的 4 段 memcpy 按专家区间分给 N 个 `std::thread`；
GPU 归属的专家保持原有串行 staging 上传路径（语义与顺序不变，默认 1 线程 = 历史路径逐字节相同）。
`src/models/Qwen4ExpHostArena.cpp`（`AILA_Q4E_ARENA_FILL_THREADS`，clamp 1..16）。

## 3. 实测（同机同模型；`tmp/q4e-strata-c7/startup_only.py`，只启动不推理）

| 配置 | 启动时间 | 相对基线 |
|---|---:|---:|
| 1 线程（默认） | 116.6 / 118.7 s | — |
| `ARENA_FILL_THREADS=8` | **105.5 s** | −9.5% ~ −11.1% |
| `ARENA_FILL_THREADS=16` | **102.9 s** | −11.8% ~ −13.3% |

- 8→16 线程收益递减（−2.6 s），说明剩余 ~103 s 由别的串行路径主导（大概率是 mmap 源文件的
  单句柄缺页路径与 GPU 归属上传的逐条 wait）；已足以止损，未继续深挖。
- 该改动不影响任何输出（拷贝语义不变），与 TTFT/decode 无关，收益在**迭代速度与冷启动**。

## 4. 结论

- **opt-in 保留**（默认 1）；建议 launcher 在下一轮验证后设 `ARENA_FILL_THREADS=8`
  （16 线程收益有限且占满 IO 队列）。
- 未做（后续可选）：文件序批量读 + H2D 与读取重叠（Strata 的完整机制）；若要把 103 s 再砍半，
  先探针定位"mmap 缺页 vs 上传 wait"的占比。

## 5. 复现

```powershell
python tmp/q4e-strata-c7/startup_only.py base
python tmp/q4e-strata-c7/startup_only.py t8 AILA_Q4E_ARENA_FILL_THREADS=8
python tmp/q4e-strata-c7/startup_only.py t16 AILA_Q4E_ARENA_FILL_THREADS=16
```
