# Strata 候选执行台账（2026-10-13 起）

对象：`docs/plans/qwen38-strata-design-scan-2026-10-13.md` 的 13 个候选，按该文档
§2 排序总表顺序实施：C1 → C4 → C8 → C2 → C3 → C6 → C5 → C9 → C7 → C11 → C10 → C12 → C13。
纪律：先探针/上界，再门，再实模；每项落独立报告，负结果归档为 opt-in 或回滚。

| # | 候选 | 状态 | 结论 | 证据 |
|---|---|---|---|---|
| C1 | prefill 双制度暂存 + stream-ahead | **完成** | **NO-GO**：transfer 与流式 compute 在 memory path 上相加，不可藏；并行 fill/直拷大页均不改变 chunk wall | [C1 报告](qwen38-strata-c1-prefill-staging-screen-2026-10-13.md) |
| C2 | copy-only 队列（blitter engine） | 进行中 | 探针实现中（DG2 是否暴露 copy-only 组 + ordinal-1 队列下内存型负载是否仍重叠） | — |
| C3 | 16B 对齐装载（`load16_a2`） | 未开始 | — | — |
| C4 | 路由画像工具链（mix + 留一 + 回存） | 未开始 | — | — |
| C5 | adaptive 异步/重叠 + 交换旋转 | 未开始 | — | — |
| C6 | verify 窗口 route tail skip | 未开始 | — | — |
| C7 | 专家填充流水线 | 未开始 | — | — |
| C8 | 工程方法学包 | 未开始 | — | — |
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
