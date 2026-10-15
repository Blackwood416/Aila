# NF4 全链路 W4A8：prefill 侧收益上界测量（B2 收口，2026-10-12）

基点 `e43ead5`（含 int8 KV 读取优化 `6deaefc`）。问题：把 decode 已验证的 q8 激活扩展到 prefill 专家核
（4-token quad 族，`MoeExpertOps.cpp:2417/2846`）是否值得？

## 1. 结论：不值得（有界负结果）

**x 激活流量整体只有 ~3% 权重**，砍半收益 ≤1.5%；而 ALU 族只占 prefill GPU 时间 ~44% → **端到端上界
≤0.7%**，远低于 +5% 采纳线。因此不实现 prefill q8 变体。

## 2. 证据（同二进制、同输入，单变量）

对 4token wgshare 家族（gate_up+down 成对计时，20 次中位数）：

| 场景（tokens=64, quads=160, share=4） | 时间 | 相对真实 token_ids |
|---|---:|---:|
| 真实 token_ids（x = 64 行 = 327 KB） | 2.1317 ms | — |
| 全部 quad 指向 token 0（x 全 L1 命中） | 2.0698 ms | **−2.9%** |
| 4096 行流式 x（20 MiB，生产尺度） | 2.1078 ms | **+1.1%** |

- 把 x 从"生产尺度流式"换成"完全缓存"，整个 ALU 族只快 ~3% → **该家族不是激活带宽受限**，
  而是指令/寄存器受限（权重 ≤25× 重读在 share=4 下由 L1/L2 承担）。
- 补充证据（W4A8 decode 同形状）：T=4 时 gate_up q8 比 bf16 慢 3.8%、down 快 4.6%（`q8` 侧净 +3.5% 慢），
  说明 `moe_q8_dot_acc` 惯用式没有带来数量级的 ALU 指令削减（未观察到 dp4a 级收益）。

## 3. 新增诊断工具（默认关闭，可复用）

- `AilaQwen4ExpKernelTests.exe --prefill-bench-only`：只跑 prefill MoE 基准（跳过全量套件）。
- `AILA_Q4E_BENCH_XLOCALITY=1`：在同一基准里追加 token0（全 L1）与 4096 行流式 x 两个对照。
- 证据：`tmp/q4e-int8kv-dpas-20261012/{prefill-xlocality.log,prefill-xlocality2.log}`。

## 4. B2 总账

- decode 侧 q8 激活（`AILA_Q4E_GPU_Q8_ADAPTIVE`，`5ef4e1a`）：内核 T=1 慢 15.5%，端到端 +1.61%（低于线）
  → 维持 opt-in；精度：内核 NRMSE ≤1.44%、全模型 cos 0.998/0.9995、输出逐位一致。
- prefill 侧：本报告给出 ≤0.7% 的端到端上界 → 不实现。稠密路径同理（其瓶颈是权重/指令，不是激活字节）。
- 若未来 NPU/XMX 侧引入真正的 int8 点积（dp4a/dpas）或激活走 DRAM 的形态改变（如更长 block、更低 L2 命中），
  用本报告的两个对照重测即可。
