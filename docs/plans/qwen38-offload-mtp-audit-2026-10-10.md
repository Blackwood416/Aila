# 显存/内存卸载结构与 MTP 前置审计（2026-10-10，只读）

审计基准 `HEAD=1f6c753`。本文归档 B3/B4 侦察结论，供后续实现引用。

## 1. 显存预算（生产 chunk=2048、reserve 3584）

准入：`expert_cache = 设备上限 − (已分配 + reserve)`，`reserve = AILA_Q4E_VRAM_RESERVE_MB + total_session_state`（`Qwen4ExpBnb4Backend.cpp:663-699`；代码默认 2560、JSON 3008、launcher 3584）。
实测（h4-ad1 日志）：VRAM 上限 15933 MiB = static 4522.3 + session 1000.2 + reserve 3584 + expert cache 6826；槽 2.417 MiB → `min(2800, …−32)` = 2792 常驻 + 2×16 staging = 预算吃满、零余量。
reserve 3584 分解（chunk=2048）：logits 970 + phase 403 + PLE staging 320 + PLE compute 220 + HC/QKV ~225 + staging 77 + 其他 ~60 ≈ **2275 MiB**，余量 ~1.3 GiB；chunk=4096 时前述合计 ≈3530 MiB（这就是 launcher 固定 3584 的原因）。**KV 增长不占 reserve**（session_state 已按 max_seq 全额预付）。

情景：KV int8 省 384 MiB → 槽上限若抬到 2949（+5.6%），需解禁共享会话守卫并做 DPAS 读内核；8192 chunk 需 +4.6 GiB（须先做"专家窗口 vs forward 行数"解耦）；ring 2→4 已否决；reserve 3008→3584 已生效（−238 槽）。

## 2. 传输管线候选（现状约束下）

- **adaptive 无跨层专家预取**：`predict_next` 恒 false（`Qwen4ExpBnb4Backend.cpp:2027-2029`）；冷专家每层在 `wait_actual_route()` 后才突发上传。注意 10-10 router 审计已证 legacy 预测输入 `buf_.x` 与下一层真实 router 输入**不等价**——若做只能作启发式预取提示（不影响正确性）且须先给证据。
- 大 chunk 8192：4096 块已触达 ~19.5k distinct（群占比 ~80%），8192 预计 H2D −43~46%，但墙钟收益受"GPU 计算主导 + 窗口未解耦"限制。
- 预解量化 bf16 bank：字节 ×3.2、常驻 −68%，16 GB 下高风险低可行（除非只覆盖热集）。
- pinned 替代 native large-page：降启动失败风险，CPU TLB 代价待测（`Qwen4ExpHostArena.cpp:72-73`）。

## 3. MTP 结构与前置（关键校正）

- **MTP 结构上要求 dense ≤2048**（`Qwen4ExpMtpRuntime.cpp:12`；launcher 32K 下硬置 `SERVER_MTP=0`）——其经济性由"≤2048 GPU 常驻 + 多行 CPU verifier"决定，**与长上下文卸载结构无关**。
- 已就绪：draft 权重 + 64 槽（155 MiB）、session KV 4.87 MiB、`{2,vocab}` logits、GPU 验证路径（`ForwardImpl(count=2, decode_verification=true)` + pair bundle）、窗口零设备分配。
- **缺口①**：多行 CPU verifier 在 adaptive 不可达（`ExExecutor.cpp:825/:1059/:1069` 绑 legacy cache；唯一喂入口是死分支 `cpu_verify_async`）。**缺口②**：T1 对齐（多行 GDN 8vs4 累加、RoPE、QKV prepare、dense-QSA；`mtp-alignment-draft-2026-10-06.md` 记录末态 123,948 个 FP32 差、max 1.34e-7）。
- 实施顺序：① verifier 重接 adaptive（bit-exact 参考路径，本轮进行中）→ ② T1 对齐资格 → ③ `AILA_Q4E_SERVER_MTP=1` 的 128-token 验收（接受率、tok/s、容差契约）。

## 4. 口径修正（相对 10-10 版路线图）

- VTune 全轮 H2D 聚合 ≈ **9.6 GB/s**（332.5 GB/34.6 s）；"4.6 GB/s" 是单批 fence 限速，两者勿混。
- "更深 ring（4–8 bank）"应标为已验证负项（ring 2→4 已否决）。
- "B3 决定 MTP 供给经济性"不成立（见 §3）。
