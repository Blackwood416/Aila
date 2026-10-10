# Qwen3.8 DeltaNet prefill：并发判别（D0）与 MLP/预取候选（K1，负结果）

日期：2026-10-10。基点 `HEAD=e808f63`。D0 为纯测试参数扫描；K1 为 opt-in 内核候选（`AILA_Q4E_GDN_MLP_PREFETCH`，默认关闭）。

## 背景（已归因）

VTune：`run_linear_delta_prefill_gpu_batched_impl<(GateActivation)1>` 全轮 13.80 s / 360 实例（38.3 ms/实例），工作分解为 `nd_range = 48 v_heads × 128 lanes`（48 WG / 6144 work-item = A770 的 9.4% 硬件线程），token 维因递推串行；逐 token ~18.2 µs、状态流量 ~520 GB/s、~30 cycle/指令，疑似内存延迟受限。

## D0：workgroup 数 / token 数扫描（测试参数，无源码改动）

`AilaGdnChunkTests --mlp-sweep`（新增，`tests/ops/GdnChunkTests.cpp`）：`vh ∈ {32,48,64,96} × n ∈ {256..4096}`，各 5 样本 round-robin（日志 `tmp/q4e-gdn-sweep-20261010/sweep.log`）。

| vh | per_head_token_ns（n=4096） | 相对 vh=48 |
|---:|---:|---:|
| 32 | 554.5 | +45.7% |
| 48 | 380.5 | — |
| 64 | 327.5 | -13.9% |
| 96 | 302.2 | -20.6% |

每 head 每 token 成本随独立 WG 数**单调下降且无拐点**（48→64 -12.7%、64→96 -10.3%）→ 延迟受限、并发未饱和，支持"提高每 lane 在途访存"的位精确候选（K1），而非先动用改数值的 chunkwise。

## K1：MLP/预取变体（位精确，性能负结果）

实现（`src/ops/GatedDeltaNetOps.cpp:437/606/664/791`、`.hpp:111`）：`template <GateActivation, bool Prefetch>`，两个 S 循环显式 8 元素双缓冲预取、地址运算提出内层，乘加表达式与分组逐字保持；env 门控默认关闭，另加 `force_prefetch` 测试开关。

逐位一致：`--mlp-parity` 同进程双跑 memcmp（state/out/conv）在 `(16,48)/(16,96) × n∈{256,1024,4096}` 全部 `bitwise=pass`；默认 correctness env 开/关一致。

性能对照（同二进制 env off → on，`--baseline-bench`/`--mlp-sweep`）：

| vh | n | off ms | on ms | 比值 |
|---:|---:|---:|---:|---:|
| 48 | 256 | 5.042 | 9.265 | 1.84× |
| 48 | 1024 | 18.937 | 36.026 | 1.90× |
| 48 | 4096 | 74.817 | 143.305 | 1.92× |
| 96 | 4096 | 119.031 | 151.803 | 1.28× |

块宽扫描 n=256：blk4 6.95 / blk8 9.26 / blk16 10.34 ms（基线 5.00）——代价随块宽单调上升，指向寄存器压力/指令发射受限。

**结论**：K1 否决。手工软件流水给每元素增加装载与寄存器搬移，把内核推向发射受限；编译器本就对全展开的 `j+=4` 循环做了取数调度。该内核在 48 WG 下虽属延迟受限，但**通过每 lane 手工 MLP 无法兑现**。

## 方向 2 现状与剩余选项（均未实施）

- **chunkwise（已实现、改数值）**：`GdnChunkOps` T=16 路径，X4 报告实测 -28~31% wall，但 KL/NLL 偏差至今只作为 opt-in（`AILA_Q4E_GDN_CHUNK_MODE=0`）；若要做，核心是数值资格而非性能改动。
- **dv 切分（2–4 WG/head）**：增加 WG 数（改善 Xe-core 均衡），但需把 `out_sq` 归一化/gate 移到 epilogue、`q_sq/k_sq` 冗余重算；不确定能否保持逐位，收益上限受总硬件线程数不变限制。
- **多 session 同批 prefill（grid 维）**：逐位安全，但单会话探针语料无法体现收益。

## 证据位置

- D0：`tmp/q4e-gdn-sweep-20261010/{candidate.patch,sweep.log,baseline.log,correctness.log}`
- K1：`tmp/q4e-gdn-k1-20261010/{candidate.patch,REPORT.md,parity.log,correctness-{off,on}.log,baseline-{off,on}.log,baseline-on-blk{4,8,16}.log,sweep-{off,on}.log}`
