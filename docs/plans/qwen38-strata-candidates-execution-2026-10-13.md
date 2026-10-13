# Strata 候选执行台账（2026-10-13 起）

对象：`docs/plans/qwen38-strata-design-scan-2026-10-13.md` 的 13 个候选，按该文档
§2 排序总表顺序实施：C1 → C4 → C8 → C2 → C3 → C6 → C5 → C9 → C7 → C11 → C10 → C12 → C13。
纪律：先探针/上界，再门，再实模；每项落独立报告，负结果归档为 opt-in 或回滚。

| # | 候选 | 状态 | 结论 | 证据 |
|---|---|---|---|---|
| C1 | prefill 双制度暂存 + stream-ahead | **完成** | **NO-GO**：transfer 与流式 compute 在 memory path 上相加，不可藏；并行 fill/直拷大页均不改变 chunk wall | [C1 报告](qwen38-strata-c1-prefill-staging-screen-2026-10-13.md) |
| C2 | copy-only 队列（blitter engine） | **完成（含硬件级复核）** | **NO-GO**：UR copy-engine 已默认启用且 ALU 负载下重叠（4.0 vs 7.1 ms），内存型负载下相加（6.70 vs 7.16）；**驱动自带 L0 SDK 复核**：A770 确有 copy-only ordinal 1，但其 H2D 与默认队列同速（9.81 vs 9.76 GB/s）→ 是 host→VRAM 通路属性而非队列放置 | [C2 报告](qwen38-strata-c2-copy-queue-screen-2026-10-13.md) §6 |
| C3 | 16B 对齐装载（`load16_a2`） | **完成** | **NO-GO**：权重布局本就 16B 对齐（行 1280/320 B、块 32 B、scale 独立数组）；唯一错位的激活 16B 装载拆成 2×8B 后 −0.5%（噪声内），内核非装载宽度受限（68 GB/s ≪ 500） | [C3 报告](qwen38-strata-c3-aligned-loads-screen-2026-10-13.md) |
| C4 | 路由画像工具链（mix + 留一 + 回存） | **完成** | **NO-GO（增量）**：导出/回存/留一已在 P3；多源 mix 仅 +1.1pp（门 3pp）。oracle 覆盖 62.74% vs agent6k 53.06%，剩余 headroom 是实例特异 | [C4 报告](qwen38-strata-c4-profile-mix-screen-2026-10-13.md) |
| C5 | adaptive 异步/重叠 + 交换旋转 | **完成** | **NO-GO**：传输与内存型 compute 相加（C1/C2 证据）→ 藏不住；旋转的前置（全量 pinned+mapped）与既有 `UR_OUT_OF_RESOURCES` 否证冲突 | [C5/C10/C12/C13 判定](qwen38-strata-c5-c10-c12-c13-verdicts-2026-10-13.md) |
| C6 | verify 窗口 route tail skip | **完成** | **NO-GO（前置未满足）**：生产无 ≥2 行 verify 窗口（MTP 关、batch=1）；输出会变需独立精度门；重启条件=多行 verify 生效后先做窗口内排名统计 | [C6/C11 判定](qwen38-strata-c6-c11-verdicts-2026-10-13.md) |
| C7 | 专家填充流水线 | **完成（含分解与 DS 评估）** | 部分采纳（opt-in）：`AILA_Q4E_ARENA_FILL_THREADS`（默认 1）启动 −10~−12%；深挖分解：**fill 95.7 s 受 mmap 缺页限（0.43 GB/s）+ 上传 24 s 逐专家串行**，盘可达 8.6-9.8 GB/s；DirectStorage 需 NuGet 运行时且"直读 VRAM"需 D3D12↔L0 interop 探针 | [C7 报告](qwen38-strata-c7-fill-pipeline-2026-10-13.md)、[分解+DS 评估](qwen38-c7-deep-load-breakdown-and-directstorage-2026-10-13.md) |
| C8 | 工程方法学包 | **完成** | 已落地：收据 schema `perf/templates/bench-receipt-v1.json`、生成器 `tools/perf/ab_receipt.py`（A/A 噪声底示例 0.08%）、探针 `--json` 机器可读输出、3% 发射规则等 6 条纪律 | [C8 方法](qwen38-strata-c8-methods-2026-10-13.md) |
| C9 | suffix-lookup drafter + DraftPolicy | **完成** | **NO-GO**：离线门未过——接受率 32.3%/35.5%（门 50%），token/窗口 1.66-1.80；且需多行 verify（前置缺失） | [C9 报告](qwen38-strata-c9-suffix-lookup-screen-2026-10-13.md) |
| C10 | batch-MTP 取舍 + batch exactness 配方 | **完成** | **NO-GO（前置未满足）**：MTP 等价门未过（pos=1492）+ 旧口径净负；仅保留"When 两条默认冲突时先按实测互斥"的规则记录 | [C5/C10/C12/C13 判定](qwen38-strata-c5-c10-c12-c13-verdicts-2026-10-13.md) |
| C11 | 编译/稳健小项 | **完成** | 逐项核实：(a) >4GB flag 刻意关（−10% decode）且大块池 ≤1 GiB 分块；(b) per_kernel split 已在位；(c) fast-math 刻意保留（位精确门全绿）；(d)(e)(f) 不适用/已满足 | [C6/C11 判定](qwen38-strata-c6-c11-verdicts-2026-10-13.md) |
| C12 | 小 chunk CPU 分担 | **完成** | **NO-GO**：Strata 收益区间 ≤1K token，与 Aila "chunk 4096 最优"（2048/4096/8192=83.9/70.6/89.2 s）冲突；外推上界 −0~3% 且改输出 | [C5/C10/C12/C13 判定](qwen38-strata-c5-c10-c12-c13-verdicts-2026-10-13.md) |
| C13 | KV 常驻/预取 | **完成** | **NO-GO（≤32K）**：KV 在 VRAM、int8 已 opt-in（+156 槽/~1%）；宿主化与 staging 归零判断相反、且传输不可藏（C1/C2）；重启条件=实际使用 ≥64K 上下文 | [C5/C10/C12/C13 判定](qwen38-strata-c5-c10-c12-c13-verdicts-2026-10-13.md) |

## 已产生的可复用设施

- `tmp/q4e-strata-c1/run_ttft14k.py`：14K TTFT reduced runner（fixture 6K 预热 + 14K + 续轮，
  `--diag` 时保留 `[Q4E AdaptiveWork]` 统计；~5 min/臂）。
- `AilaQwen4ExpKernelTests --queue-overlap-probe`：队列重叠微探针（burn/copy/dep/ping-pong/
  内存型变体/三种 DMA 源速率），opt-in、零生产引用。
- `AILA_Q4E_STAGE_ORACLE` / `AILA_Q4E_STAGE_DIRECT_DMA`：C1 实验用 staging 探针，
  **未保留**（已回滚，生成补丁在 `tmp/q4e-strata-c1/`）。

## 总结论（13/13 完成，2026-10-13）

- **采纳 1 项（opt-in）**：C7 `AILA_Q4E_ARENA_FILL_THREADS`（默认 1 不变；8 线程启动 −10%）。
- **采纳设施 1 项**：C8 方法学包（收据 schema/工具、3% 发射规则、探针 `--json`）。
- **其余 11 项 NO-GO**，其中三类原因：
  1. **物理/平台层**：C1/C2（DMA 与内存型 compute 相加，藏不住）、C3（布局本就 16B 对齐）、
     C5（交换旋转的 pinned 前提与既有否证冲突）。
  2. **门未过**：C4（mix +1.1pp < 3pp）、C9（接受率 32-36% < 50%）。
  3. **前置缺失/与既有实测冲突**：C6、C10（多行 verify/MTP 未恢复）、C11（全部已在位或刻意关闭）、
     C12、C13（与 chunk 4096 / VRAM KV 的既有最优冲突）。
- 顺序说明：C2 紧接 C1 完成（C1 的微探针直接回答了 C2 的机制问题，且该机制在 worker 中已默认开启）；
  其余按扫描文档顺序执行。
- 横切结论：Aila 的 prefill 成本是"**流式 compute + 不可藏的 DMA**"主导（C1/C2），
  decode 的进一步收益被"多行 verify 前置"卡住（C6/C9/C10），而**启动/迭代时间**是本轮唯二
  可直接兑现的优化面（C7；另一面是已完成的 P1 修复）。
