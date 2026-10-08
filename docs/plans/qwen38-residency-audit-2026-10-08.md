# Qwen3.8：容量、物理驻留与真实读盘审计

用户提出改为优先利用 16 GiB VRAM + 64 GiB RAM、让 PLE 留盘。本轮核对容量，并用原运行时做有界文件页驻留验证，没有重写 offload、扩大旧 Host pool 或修改模型精度。结论是：**容量方向成立，当前依赖 Windows 文件缓存；需要解决的是可解释的驻留预算与重复/淘汰问题，不能从 Host pool=0 或较小进程工作集直接推断 RAM 没有利用。**

## 模型文件的实际容量

只读取 131 个 safetensors header，按 tensor data_offsets 计数，没有加载全部权重。

| 内容 | GiB |
|---|---:|
| 48 层 routed experts，含 NF4/nested 元数据 | 58.0354 |
| 其余层内 backbone | 2.7405 |
| Backbone 合计 | **60.7759** |
| Embed tokens | 1.1841 |
| LM head | 1.1841 |
| 上述合计 | **63.1441** |
| 最终 HC mixer，另计 | 0.0122 |
| PLE | **29.8184** |
| MTP，本配置未启用 | 1.2840 |
| Vision，本测试未使用 | 0.2225 |

用户的 60.78/63.14 GiB 数字正确。当前配置的总专家 cache 为 2824 槽、6825.6 MiB，decode 稳定热集 2808 槽；不是全部 16 GiB 都能用于权重，还需 KV、workspace、PLE 计算缓冲和显示余量。Embed 默认走 host mmap，LM head 及主体投影在 GPU。按初始 profile 排除 GPU 热专家，RAM 侧原始补集加 embedding 的页对齐范围约 **52.63 GiB**。这尚不是计入系统和运行时开销后的实测可驻留预算。

## 原配置实际用了 Windows 文件缓存

同一 Worker、同一冻结自然代码输入 5968 prompt tokens、每次 16 outputs，关闭 prefix reuse，使每次都完整 prefill。硬件、32K 容量、chunk4096、reserve3584、CPU18线程和 NF4/BF16 算术不变。正式请求关闭诊断日志。

物理读盘来自 Windows PhysicalDisk 原始累计读取字节的差值；模型在磁盘 0 的 C:，因此列出该物理盘。它包含该盘其他进程的读取，不是逐文件 ETW 归因，也不是 ReadFile 的逻辑字节或 H2D payload。

| 阶段 | 首字 s | 整请求 s | 物理盘 0 读取 GiB |
|---|---:|---:|---:|
| 原配置，第一个完整请求 | 101.56 | 103.86 | **54.925** |
| 原配置，第二次相同请求 | **32.08** | **33.37** | **0.794** |

加载阶段另外读取约 10.76 GiB，启动约 25.43 秒，不混入上表。首次请求还含 JIT/首次数据触达，不能把全部耗时差解释为 SSD 时间。至少在这个同输入暖态样本里，主体权重已经大量由 RAM/file cache 服务；160 GB 等旧统计是累计 H2D 供给，不能据此断言重复从 SSD 读了 160 GB。

第二次请求时 Worker 工作集峰值仅约 11.79 GiB，系统 available 最低仍有约 42.60 GiB；但读盘已不足 1 GiB。Available 包含可回收的 standby file cache，不是全部空白内存。进程退出后的系统快照也印证了这一点：Available 60646 MiB，而 Standby normal+reserve 合计约 59.22 GiB。该快照是全系统指标，不把所有缓存都归因于模型。

## 原始文件页锁驻原型

为验证较大 RAM 驻留，使用独立、随父进程退出的 helper 映射原始文件并 VirtualLock；不分配另一份 NF4 权重副本。只选择初始 GPU profile 前 2808 项之外的专家来源和 embedding，排除 PLE、MTP、Vision。主 Worker 继续走原 mmap/ReadFile 路径。后续 GPU ready adoption 会改变热集，因此这不是动态互补所有权实现。

先以 64 MiB 验证 Windows API 和页状态，再在两次基线之后按 8 GiB available admission 余量逐块增加驻留。最终锁定 **42.05 GiB**，在余量处停止，未锁满 52.63 GiB 计划。QueryWorkingSetEx 抽查 47,952 页，全部 valid、locked。预加载约 **45.13 秒**，期间额外物理读盘约 **9.31 GiB**，单列记录。

| 阶段 | 首字 s | 整请求 s | 物理盘 0 读取 GiB |
|---|---:|---:|---:|
| 文件页锁驻，第一次 | 53.18 | 54.55 | **19.297** |
| 文件页锁驻，第二次 | 43.76 | 45.05 | **19.167** |

锁驻后 Worker 工作集峰值仍约 12 GiB，运行时 available 最低约 7.92 GiB。四次请求的输入 hash、输出 16-token 文本 hash 均相同。这个部分锁驻方案反而增加了未锁部分的读盘；不推广，也不把它作为否定完整 GPU/RAM 互补驻留的证据。它同时改变页可回收性、保留余量、冷页覆盖；没有逐文件 cache priority 和页级重复归因。

原型采用先基线后锁驻的顺序，未做反序或清空文件缓存。这是可行性与读盘诊断，不是统计稳定的性能推广测试。Server 四次请求全部成功、优雅关闭，helper 已释放全部锁页和映射；生产配置未变。

## 调整后的实施方向

1. 以独立权重覆盖计量：GPU 常驻集合、RAM 原始补集、临时传输区、KV/workspace、OS/其他进程分别记账；Host pool、mmap 工作集和 standby cache 不能简单相加。
2. Windows/运行时已有物理占用和额外可用余量分开。固定再保留 8 GiB available 会限制锁驻规模，不能把这个试验阈值误当成硬件最多只能容纳 42 GiB 权重。
3. 优先保留 backbone 与 host embedding，PLE 表按需读取并研究较低缓存保留优先级；不锁整份约 95 GiB 模型，也不增加与 mmap 并行消费的重复大 Host pool。
4. 必须处理 GPU 热集变化/借槽后的覆盖，不能用静态初始 profile 宣称动态零重复。先观察真实文件页与物理读盘，再决定是否需要专有 RAM arena；不能因旧 4/32 GiB 局部缓存失败而否定统一驻留。
5. 收益验收使用不同自然长输入和工具多轮下的物理读盘、首次总成本、暖态整轮时间；当前同输入暖态已经低读盘，后续不能承诺单靠扩大 RAM 会突破 decode 20。

证据：`tmp/q4e-residency-inventory.json`（逐 tensor 容量）、`tmp/q4e-residency-probe1/result.json`（请求、页状态、内存采样、物理读盘、清理）、`tmp/q4e-residency-probe1.log`。原型脚本为 `tmp/q4e-resident-pages.py` 与 `tmp/q4e-residency-probe.py`，仅作本次验证，不是推荐生产 launcher。
