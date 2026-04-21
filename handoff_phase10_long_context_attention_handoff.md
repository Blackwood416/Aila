# Phase 10 Long-Context Attention Handoff

## Summary

以 `2026-04-20` 当前工作区 `9df88e4` 为阶段性收口点，这几轮性能优化的主线已经从“decode 小 matmul 去 oneDNN 化”继续推进到：

- exact long-context decode attention
- exact long-context initial prefill attention
- exact long-context cached prefill attention
- Qwen3.5 文本 `generate_messages(...)` 的真实前缀复用

这一轮没有走 windowed decode / windowed prefill 这类质量 tradeoff 路线，而是继续保持 exact 语义。

当前可以比较明确地说：

- 长上下文 exact decode 的阶梯式退化已经被大幅缓解
- 长上下文 initial prefill 也已经有结构性提速
- 文本多轮消息路径终于真正命中了 cached prefill，而不是每轮都 full prefill

## Current Head

- 当前 HEAD：`9df88e4`
- 当前工作区：仅新增本 handoff 文件

补充说明：

- 写入本 handoff 之前，工作区是 clean

最近几个关键 checkpoint：

- `8c5f4ff` `Add decode subgroup matvec custom path`
- `2326c8d` `Replace decode qkv projections with subgroup kernels`
- `ab23ad7` `Implement tiled exact decode attention merge`
- `cbc7ae8` `Add Qwen3.5 prefill stage profiling`
- `c21ad25` `Optimize long prefill exact attention`
- `9df88e4` `Enable exact cached prefill reuse`

## What Was Implemented

### 1. Exact long-context decode attention merge

主文件：

- `src/ops/AttentionOps.cpp`

实现摘要：

- 为 `attention_decode(...)` 增加了 `head_dim == 256` 的 exact tiled partial-merge 路径
- 不再走“全量 logits 落地到 global score buffer + 单 workgroup softmax/V accumulation”的结构
- 改成先按 tile 计算精确 partial `(m_i, l_i, acc_i)`，再做 exact merge

关键意义：

- 避免了 context-sized local scratch 带来的 occupancy cliff
- 避免了 decode 长上下文下 full-attn 成为单组串行瓶颈

当前路由条件：

- `head_dim == 256`
- `effective_len >= 1024`
- decode partial workspace 可用

### 2. Qwen3.5 prefill stage profiling

主文件：

- `src/models/Qwen35HybridTextBackend.cpp`

新增内容：

- `AILA_PROFILE_Q35_PREFILL=1`
- `AILA_PROFILE_Q35_PREFILL_EVERY=<N>`
- 让 `seq_len > 1` 的 full-attn / linear / FFN 路径都进入统一 stage profile

输出形式：

- `[Q35PrefillProfile] ...`

这一步的价值非常直接：

- 它证明了长文 prefill 的主矛盾确实是 full-attn prefill，而不是别的边角热点

### 3. Exact long-context initial prefill attention

主文件：

- `src/ops/AttentionOps.cpp`

实现摘要：

- 为 `attention_prefill(...)` 增加了 `head_dim == 256` 的 exact online softmax kernel
- 对每个 `(head, row)` 采用 tiled online merge，边扫 K/V 边维护精确 `(m, l, acc)`
- 不再依赖大 `scores_buf` 三段式作为长 prompt 主热路径

当前路由条件：

- `head_dim == 256`
- `seq_len >= 1024`

关键意义：

- 直接打掉 initial prefill 中 full-attn 的大工作集和重复访存成本
- 明显改善长 prompt prefill 退化

### 4. Exact long-context cached prefill attention

主文件：

- `src/ops/AttentionOps.cpp`

实现摘要：

- 为 `attention_prefill_cached(...)` 增加了 `head_dim == 256` 的 exact online softmax 路径
- 对增量 prefill 的每个 query row，精确扫描 `[0, start_pos + qi]` 范围
- 保持语义不变，不使用 window tradeoff

当前路由条件：

- `head_dim == 256`
- `total_len >= 1024`

### 5. `generate_messages(...)` 文本路径前缀复用

主文件：

- `include/engine/Engine.hpp`

实现摘要：

- 之前 `generate()` 有前缀复用，但 `generate_messages(...)` 仍然每轮：
  - `backend_->reset()`
  - `cached_ids_.clear()`
  - full prefill
- 现在文本路径会：
  - 计算 `cached_ids_` 与 `full_ids` 的最长公共前缀
  - `truncate_kv_cache(reusable_prefix)`
  - 只 prefill 后缀

当前保守限制：

- 只对 `total_vision_tokens == 0` 的文本路径启用
- 多模态消息仍然走 full prefill，避免把 vision mRoPE / image token 位置逻辑一起卷进本轮

## Validated Results

### A. 这一轮开始前的短上下文强基线

保留 artifact：

- `tmp/perf/phase10_subgroup_linearall_stage6/8c5f4ff/bench.json`
- `tmp/perf/phase10_subgroup_linearall_stage6/8c5f4ff/decode_profile_summary.json`

短基线：

- `sample_main`
  - prefill `2844.08 tok/s`
  - decode `116.18 tok/s`

decode profile 平均热点：

- `linear_proj 3.927 ms`
- `ffn_proj 3.564 ms`
- `down 3.255 ms`
- `linear_delta 3.040 ms`
- `attn 1.353 ms`

说明：

- 这时短上下文 decode 已经很强
- 后续主目标不再是继续抠短 decode，而是把收益推进到 exact 长上下文

### B. Exact long-context decode baseline before attention rewrite

保留 artifact：

- `tmp/perf/phase10_longctx_probe_before/2326c8d/bench.json`
- `tmp/perf/phase10_longctx_profile_baseline/2326c8d/bench.json`

基线：

- `3072 / 128`
  - prefill `1636.57 tok/s`
  - decode `67.15 tok/s`

profile baseline：

- `3072 / 64`
  - decode `30.68 tok/s`
- `attn` 单项约 `9.2 ms/token`

结论：

- full-attn decode 就是长上下文退化的主瓶颈

### C. Exact long-context decode after tiled partial-merge

保留 artifact：

- `tmp/perf/phase10_longctx_exact_partial_merge_v1/ab23ad7/bench.json`

保留结果：

- `3584 / 128`
  - prefill `1551.18 tok/s`
  - decode `125.30 tok/s`

同阶段保留短基线：

- `tmp/perf/phase10_longctx_exact_partial_merge_v1/2326c8d/bench.json`
  - `sample_main`
  - prefill `2863.15 tok/s`
  - decode `116.45 tok/s`

额外说明：

- `3072 / 128` exact decode 在控制台实测达到过 `127.70 tok/s`
- 但同 phase 下后续 manual run 覆盖了 `manual_greedy` artifact，所以该值没有单独保留成最终 json
- retained `3584 / 128` artifact 已足够说明 exact long decode cliff 基本被打掉

### D. Long-prefill bottleneck identification

保留 artifact：

- `tmp/perf/phase10_prefill_profile_probe/ab23ad7/bench.json`
- `tmp/perf/phase10_prefill_profile_probe/ab23ad7/bench_logs/manual_greedy.log`

探针结果：

- `3072 / 1`
  - prefill `1328.75 tok/s`

关键 profile：

- total `0.745922 ms/token`
- `attn 0.420967 ms/token`
- `linear_delta 0.241900 ms/token`

结论：

- 长文 prefill 的主矛盾是 full-attn prefill
- `attn` 大约占总 prefill token cost 的 `56%`

### E. Exact long-context initial prefill after online merge kernel

保留 artifact：

- `tmp/perf/phase10_prefill_exact_online_v1/cbc7ae8/bench_logs/manual_greedy.log`

保留结果：

- `3584 / 128`
  - prefill `2318.71 tok/s`
  - decode `123.82 tok/s`

额外说明：

- `3072 / 128` 在控制台实测达到：
  - prefill `2475.47 tok/s`
  - decode `127.31 tok/s`
- 但该 manual run 后来被同 phase 下的 `3584` run 覆盖，没有保留最终独立 json

profile 改善：

- `3072` prefill 中 `attn` 从约 `0.420967 ms/token` 降到约 `0.130153 ms/token`
- 这组 profile 数字在控制台确认过，但因为后续 manual run 覆盖，同样没有最终留成独立 artifact

### F. Text cached prefill reuse sanity on real `generate_messages(...)`

这一步没有独立 benchmark case，当前是运行时 sanity，而不是 perf-suite artifact。

验证方式：

- 双轮文本输入
- 第一轮先堆长上下文
- 第二轮发送较短 follow-up，要求命中 cached prefill

控制台确认：

- 第一轮：
  - `Full prefill: 1723 tokens`
  - 整轮约 `720 ms`
- 第二轮：
  - `Incremental prefill: reusing 1719 cached tokens, prefilling 135 new tokens`
  - 整轮约 `326 ms`

结论：

- `generate_messages(...)` 已经真正走到了 cached prefill
- 本轮收益主要来自“只 prefill 后缀”而不是每轮 full prefill

## Important Caveats

### 1. `phase10_prefill_exact_online_v1` 的 artifact commit id 看起来会“滞后”

需要特别注意：

- `phase10_prefill_exact_online_v1/cbc7ae8/...` 这批 artifact 是在 dirty worktree 上生成的
- 当时代码已经包含后来提交成 `c21ad25` 的 long-prefill kernel
- 因此 bench 目录里的 `shortCommit = cbc7ae8` 并不代表没有 long-prefill 改动

换句话说：

- 这批 artifact 的“代码语义”更接近后来的 `c21ad25`
- 只是 build metadata 记录的还是提交前的 HEAD

### 2. `generate_messages(...)` 的 cached prefill 目前只在文本路径启用

当前保守行为：

- `total_vision_tokens == 0` 时启用前缀复用
- 多模态消息仍然 full prefill

原因：

- 本轮不把 vision embedding override / mRoPE 位置复用一起混改

### 3. 还没有正式的 cached-prefill benchmark case

目前标准 bench 路径主要覆盖：

- raw prefill benchmark
- raw decode benchmark

尚未正式覆盖：

- 多轮 `generate_messages(...)`
- 文本前缀复用后的 cached prefill 统计

所以：

- `9df88e4` 的核心收益目前主要来自运行时 sanity，而不是完整脚本化 benchmark

## Failed / Rejected Directions

### 1. Global-score-buffer exact decode attempt

状态：

- 试过
- 明确负收益
- 已回退

现象：

- `3072 / 128` exact decode 从 baseline `67.15 tok/s` 掉到 `63.88 tok/s`

结论：

- “QK tiled + global score buffer + 单组 softmax/V”这条路不值得继续

### 2. Engine-layer exact chunked prefill

状态：

- 试过
- 明确负收益
- 已回退

典型结果：

- `3072 / 128`
  - prefill 约 `1453 tok/s`
- `3584 / 128`
  - prefill 约 `1338 tok/s`

结论：

- 这更像 working-set 保守策略，不是有效提速
- 真正有效的是 attention kernel 本身的 exact online 化

## Current Code State

当前已经具备的长上下文 exact attention 路径：

- `attention_decode(...)`
  - `head_dim == 256`
  - long-context exact partial-merge decode
- `attention_prefill(...)`
  - `head_dim == 256`
  - long initial prefill exact online attention
- `attention_prefill_cached(...)`
  - `head_dim == 256`
  - long cached prefill exact online attention

当前新增但非热路径行为：

- `AILA_PROFILE_Q35_PREFILL`
- `AILA_PROFILE_Q35_PREFILL_EVERY`

当前仍未展开的热点：

- `qk_rope`
- `kv_copy`
- full-attn prefill/decode 的边界准备成本
- cached-prefill 的正式 benchmark / correctness A/B

## Recommended Next Steps

### 1. 先补正式 cached-prefill benchmark / profile case

这是最推荐的下一步。

建议新增一类 benchmark：

- 两轮文本消息
- 第一轮长 prompt 建 cache
- 第二轮短 follow-up
- 明确统计：
  - reusable prefix
  - incrementally prefilling suffix length
  - second-turn wall time

最好同时支持：

- greedy
- sample
- `AILA_PROFILE_Q35_PREFILL`

### 2. 做 long-prefill / cached-prefill correctness A/B

建议至少补以下一种：

- 固定 seed，对比旧 checkpoint 与当前 checkpoint 的生成 token 序列
- 或做单层 attention 输出 compare

当前阶段更关注性能推进，所以 correctness 还没有系统化落盘。

### 3. 继续优化 `qk_rope + kv_copy`

在 full-attn 长文 prefill 被打下来以后，下一批更值得看的热点就是：

- `qk_rope`
- `kv_copy`

可以考虑的方向：

- 减少 Q/K 准备阶段的中间 materialize
- 进一步融合 RoPE + cache write
- 避免 full-attn prefill 前后的多次边界搬运

### 4. 等 cached-prefill benchmark 稳定后，再考虑清理旧长文 baseline

目前还不建议立刻删掉 old path，因为：

- 初始 prefill / cached prefill 的 exact online 路径虽然已经明显正向
- 但 cached-prefill 还缺正式 benchmark / correctness A/B

更稳妥的流程是：

1. 先把 cached-prefill benchmark / correctness 补齐
2. 再决定是否收口旧 baseline

## Quick Reproduction Commands

### 1. 短基线

```powershell
.\build.ps1 -BuildDir build -Config Release
.\bench.ps1 -BuildDir build -Preset phase_gate_q35_text -Phase phase10_prefill_profile_cleanup -CaseNames @('greedy_main')
```

### 2. 长上下文 exact decode baseline

```powershell
.\bench.ps1 -BuildDir build -ModelDir Qwen3.5-0.8B -Phase phase10_longctx_probe_before -PromptTokens 3072 -GenTokens 128 -BenchIters 1 -WarmupIters 1
```

### 3. 长上下文 prefill profile probe

```powershell
.\bench.ps1 -BuildDir build -ModelDir Qwen3.5-0.8B -Phase phase10_prefill_profile_probe -PromptTokens 3072 -GenTokens 1 -BenchIters 1 -WarmupIters 0 -EnvOverrides @{AILA_PROFILE_Q35_PREFILL='1';AILA_PROFILE_Q35_PREFILL_EVERY='1'}
```

### 4. 长上下文 initial prefill / decode sanity

```powershell
.\bench.ps1 -BuildDir build -ModelDir Qwen3.5-0.8B -Phase phase10_prefill_exact_online_v1 -PromptTokens 3584 -GenTokens 128 -BenchIters 1 -WarmupIters 1
```

### 5. 文本 `generate_messages(...)` cached prefill runtime sanity

思路：

- 运行 `Aila.exe`
- 先发一条足够长的文本消息
- 第二轮再发一条较短 follow-up
- 打开 `AILA_PROFILE_Q35_PREFILL=1`
- 查看是否出现：
  - `Incremental prefill: reusing ... cached tokens, prefilling ... new tokens`
