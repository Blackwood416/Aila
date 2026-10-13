# Strata C6 / C11 判定（2026-10-13）

## C6 verify 窗口 route tail skip — NO-GO（前置未满足；机制本身未被否证）

1. **前置缺失**：tail skip 的意义在"窗口内多个 token 都把某专家排在 7-10 名"；Aila 生产是
   `--sessions 1 --decode-batch-size 1` + MTP 关闭（等价门未过，`40d6bf6`），**没有 ≥2 行的
   verify 窗口**。T=1 时该规则退化为"从不跳过"。
2. **代价**：跳过的专家贡献记 0（不重归一化）→ 输出会变，需要独立的精度门；Aila 的 MTP 门
   正是因为"verify 行与单行非逐位等价"而失败，tail skip 与那条门天然冲突，只能作为**另一个
   opt-in 档**（KL/任务质量口径）。
3. **重启条件**：出现可用的多行 verify 路径（B4-3 修复、或 decode-batch>1 进入默认路径）时，
   先做设备侧"窗口内每 token 排名"统计，再按扫描文档的 KL 口径评估；覆盖率 ≥90% 时该规则
   按 Strata 自己的规则自动不启用（Aila 的 decode 覆盖 53%，属 miss 重区间，届时值得测）。

## C11 编译/稳健性小项 — 已核实（无需改动；两项已在位）

| 项 | 结论 | 依据 |
|---|---|---|
| (a) `-ze-opt-greater-than-4GB-buffer-required` | **保持关闭**：CMake 已列出该选项并注明"在 Arc A770 上 decode 回归 ~10%"；Aila 大块设备池按 **≤1 GiB** 分块（`Qwen4ExpAdaptiveCache.cpp:154-161` 注释 + `slots_per_chunk = 1 GiB / slot_bytes`），无 >4 GiB 单缓冲 | `CMakeLists.txt:123-127`、`src/models/Qwen4ExpAdaptiveCache.cpp:154-161` |
| (b) `-fsycl-device-code-split=per_kernel` | **已在位** | `CMakeLists.txt:119` |
| (c) fp32 除法/开方 correctly-rounded + 宿主 precise | **不采纳**：Aila 全局 `-ffast-math` 是刻意选择，且位精确内核门（对 CPU 参考）全绿；改全局 FP 语义 = 改变全部数值与性能，须先有具体 parity 失败案例 | `CMakeLists.txt:112`；`ctest -L 'cpu|abi'` 20/20、P4/C3 位精确门 |
| (d) 大分配别名自检（tag/读回/spacer） | **仅记录**：Strata 的案例发生在 xe/Linux 驱动；Windows/L0 无证据，做一次全池自检成本高（22 GiB），收益不可量化 | 扫描文档 §1-C11 第 3 点 |
| (e) 设备侧 host-flag 自旋有界 | **已满足**：Aila 的等待是 host 侧有界自旋（`AILA_Q4E_CPU_HANDOFF_SPIN_US`，clamp 0-5000 µs）与事件等待；热路径无设备侧无界自旋 | `src/models/Qwen4ExpExpertExecutor.cpp:109-185` |
| (f) KV append 的 4B lane store | **已满足**：int8 KV 读写已向量化（`6deaefc`：per-token scale 提出循环 + 8B 向量装载，14k 内核 1.854→1.053 ms） | `qwen38-int8kv-vec-2026-10-12.md` |

结论：C11 无可实施增量；(a)(c) 保留"重启条件"——出现 >4 GiB 单缓冲需求或具体 parity 失败时再评。
