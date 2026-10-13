# Strata 候选执行台账（2026-10-13 起）

对象：`docs/plans/qwen38-strata-design-scan-2026-10-13.md` 的 13 个候选，按该文档
§2 排序总表顺序实施：C1 → C4 → C8 → C2 → C3 → C6 → C5 → C9 → C7 → C11 → C10 → C12 → C13。
纪律：先探针/上界，再门，再实模；每项落独立报告，负结果归档为 opt-in 或回滚。

| # | 候选 | 状态 | 结论 | 证据 |
|---|---|---|---|---|
| C1 | prefill 双制度暂存 + stream-ahead | **完成** | **NO-GO**：transfer 与流式 compute 在 memory path 上相加，不可藏；并行 fill/直拷大页均不改变 chunk wall | [C1 报告](qwen38-strata-c1-prefill-staging-screen-2026-10-13.md) |
| C2 | copy-only 队列（blitter engine） | **完成** | **NO-GO**：UR copy-engine 已默认启用且 ALU 负载下重叠（4.0 vs 7.1 ms），但内存型负载下相加（6.70 vs 7.16）；2026.1 无 L0 头/库；Strata 自家 A750 仅 +1.4% 且有挂死 | [C2 报告](qwen38-strata-c2-copy-queue-screen-2026-10-13.md) |
| C3 | 16B 对齐装载（`load16_a2`） | **完成** | **NO-GO**：权重布局本就 16B 对齐（行 1280/320 B、块 32 B、scale 独立数组）；唯一错位的激活 16B 装载拆成 2×8B 后 −0.5%（噪声内），内核非装载宽度受限（68 GB/s ≪ 500） | [C3 报告](qwen38-strata-c3-aligned-loads-screen-2026-10-13.md) |
| C4 | 路由画像工具链（mix + 留一 + 回存） | **完成** | **NO-GO（增量）**：导出/回存/留一已在 P3；多源 mix 仅 +1.1pp（门 3pp）。oracle 覆盖 62.74% vs agent6k 53.06%，剩余 headroom 是实例特异 | [C4 报告](qwen38-strata-c4-profile-mix-screen-2026-10-13.md) |
| C5 | adaptive 异步/重叠 + 交换旋转 | 未开始 | — | — |
| C6 | verify 窗口 route tail skip | 未开始 | — | — |
| C7 | 专家填充流水线 | **完成** | 部分采纳（opt-in）：`AILA_Q4E_ARENA_FILL_THREADS`（默认 1）；启动 116.6/118.7 → 105.5(8T)/102.9(16T) s，−10~−12%；默认路径不变 | [C7 报告](qwen38-strata-c7-fill-pipeline-2026-10-13.md) |
| C8 | 工程方法学包 | **完成** | 已落地：收据 schema `perf/templates/bench-receipt-v1.json`、生成器 `tools/perf/ab_receipt.py`（A/A 噪声底示例 0.08%）、探针 `--json` 机器可读输出、3% 发射规则等 6 条纪律 | [C8 方法](qwen38-strata-c8-methods-2026-10-13.md) |
| C9 | suffix-lookup drafter + DraftPolicy | 未开始 | — | — |
| C10 | batch-MTP 取舍 + batch exactness 配方 | 未开始 | — | — |
| C11 | 编译/稳健小项 | 未开始 | — | — |
| C12 | 小 chunk CPU 分担 | 未开始 | — | — |
| C13 | KV 常驻/预取 | 未开始 | — | — |

## 已产生的可复用设施

- `tmp/q4e-strata-c1/run_ttft14k.py`：14K TTFT reduced runner（fixture 6K 预热 + 14K + 续轮，
  `--diag` 时保留 `[Q4E AdaptiveWork]` 统计；~5 min/臂）。
- `AilaQwen4ExpKernelTests --queue-overlap-probe`：队列重叠微探针（burn/copy/dep/ping-pong/
  内存型变体/三种 DMA 源速率），opt-in、零生产引用。
- `AILA_Q4E_STAGE_ORACLE` / `AILA_Q4E_STAGE_DIRECT_DMA`：C1 实验用 staging 探针，
  **未保留**（已回滚，生成补丁在 `tmp/q4e-strata-c1/`）。
