# Phase 10 Full Decode Custom Kernel Plan

## Summary

以 `2026-04-20` 当前工作区 `7380bc2` 为起点，但性能锚点继续使用 `2026-04-19` 的稳定 benchmark `9326084`：

- `greedy_main`: `prefill 3113.48 tok/s`, `decode 103.25 tok/s`
- `sample_main`: `prefill 3132.33 tok/s`, `decode 97.80 tok/s`
- `decode_profile_main` 平均总耗时：`33.239 ms/token`
- 当前 decode 前五热点：
  - `ffn_proj 4.8585 ms`
  - `down 4.4695 ms`
  - `linear_proj 3.7805 ms`
  - `linear_o 3.1450 ms`
  - `linear_delta 2.7895 ms`

Phase 9 的核心结论已经足够明确：

- GPU 实际计算时间约 `2.23 ms/token`
- wall time 约 `9.7 ms/token`
- decode 主线每 token 仍有 `96` 次 oneDNN `matmul::execute()`/等价小 GEMM 调度
- oneDNN / oneMKL 互换不能解决瓶颈

因此下一阶段不再继续做“换库 / 换 primitive / 调 env”试探，而是正式转向：

**Qwen3.5-0.8B 文本 decode-only 全链路 custom kernel 化**

目标不是把每个小 matmul 都重写成一个独立替代品，而是把 decode 热路径改造成少数几个 backend-local 固定形状融合 kernel，直接减少 dispatch 数量。

建议落盘名固定为：`handoff_phase10_full_decode_custom_kernel_plan.md`

## Key Changes

### 阶段 1：确立 decode-only custom backend 骨架

目标：先把“decode 主线”和“prefill / generic 路径”彻底分开，避免后续实现始终被 `Linear::forward(seq_len==1)` 拖回 oneDNN。

实施内容：

- 在 `Qwen35HybridTextBackend` 内新增 decode-only helper 层，不改 `Linear` 对外接口
- 只覆盖固定目标：
  - `Qwen3.5-0.8B` 文本路径
  - `seq_len == 1`
  - `hidden_size == 1024`
  - `ff_dim == 3584`
- 所有 custom decode kernel 都只做 backend-local 固定形状 specialization
- prefill、非目标模型、非固定形状继续保留现有 oneDNN/generic 路径
- decode 主线不再新增 env 开关；新路径一旦接入就是默认热路径

设计原则：

- 不在 `Linear.cpp` 主线里继续堆 decode 特判
- 不把半成品 custom path 长期和 oneDNN decode 双轨并存
- 一旦某段 custom path A/B 通过，就让目标形状 decode 默认走它

验收标准：

- 代码结构上可以明确区分：
  - decode fixed-shape custom path
  - generic / prefill path
- 不改变 prefill 当前性能锚点
- 不引入新的 host roundtrip

### 阶段 2：FFN decode 全手写融合

目标：先吃掉最重的 `ffn_proj + ffn_act + down + post_mlp`，因为这一段合计已经超过 `14 ms/token`，也是最适合按固定形状做专用 kernel 的部分。

实施内容：

- 不再继续尝试纯 FMA 风格 FFN kernel
- 必须直接按 **XMX / joint_matrix / subgroup mat-vec** 方向实现
- 拆成两波，不强求单 kernel 吃完整个 FFN：

1. `gate_up_proj + swiglu`
   - 输入：`buf_.normed`
   - 权重：现有 fused `gate_up` `[hidden, 2 * ff_dim]`
   - 输出：激活后的 `ffn_hidden`
   - 不再 materialize 旧的 decode `buf_.gate_up`

2. `down_proj + post_mlp residual/rmsnorm`
   - 输入：上一波输出
   - 权重：现有 `down` `[ff_dim, hidden]`
   - 输出：直接写下一层 `buf_.normed`
   - decode 热路径中不再单独 materialize `buf_.up`

实现要求：

- 固定形状专用，不做通用 GEMV 框架
- 优先减少 dispatch 数，而不是追求单个 kernel 绝对最优
- kernel 内只允许极小 shared/local 用于收口，不回到大块 local staging
- 若做归约，优先 subgroup reduction / no-shmem 风格

验收标准：

- `decode_profile_main` 中：
  - `ffn_proj + ffn_act + down + post_mlp` 合计下降 `>= 30%`
- `greedy_main` decode 达到 `>= 112 tok/s`
- `greedy_main` prefill 保持 `>= 3000 tok/s`

### 阶段 3：线性注意力 decode 投影链全手写融合

目标：去掉线性层里的 oneDNN decode `linear_all_proj` 和 `linear_o_proj`。

当前线性层 decode 逻辑是：

- `linear_all_proj`
- `run_linear_delta_decode_gpu`
- `linear_o_proj`
- `post_attn rmsnorm`

实施内容：

- 新增 decode-only `linear_all_proj` custom kernel：
  - 固定输出 fused row
  - 继续复用当前 `run_linear_delta_decode_gpu(...)` 输入布局
  - 不修改现有 recurrent state / ring-buffer state 布局
- 新增 decode-only `linear_o_proj + post_attn residual/rmsnorm` kernel：
  - 输入：`linear_delta` 输出
  - 输出：直接写下一步 `buf_.normed`
  - decode 热路径中不再单独 materialize `buf_.gate`

设计要求：

- 不回到 Phase 8 那类纯 BF16/F32 scalar FMA GEMV
- 必须按固定形状做 XMX/joint-matrix/subgroup tile specialization
- 保持 `linear_delta` 当前状态布局不动，不混入状态格式迁移

验收标准：

- `decode_profile_main` 中：
  - `linear_proj + linear_o + post_attn` 合计下降 `>= 30%`
- `greedy_main` decode 达到 `>= 118 tok/s`
- `sample_main` decode 不低于 `97 tok/s`

### 阶段 4：全注意力 decode 投影链全手写融合

目标：去掉 full-attention 层里的 oneDNN decode `qkv_proj` 和 `o_proj`。

当前 full-attention decode 逻辑是：

- `qkv_proj`
- `decode_prepare_qkv_partial(...)`
- `attention_decode(...)`
- `attn_gate`
- `o_proj`
- `post_attn rmsnorm`

实施内容：

- 新增 decode-only `qkv_proj` custom kernel：
  - 直接生成当前 packed layout
  - 保持 `decode_prepare_qkv_partial(...)` 的输入契约不变，先不混改 RoPE / KV 写入
- 新增 decode-only `o_proj + residual/rmsnorm` kernel：
  - 在 attention / gate 之后直接写下一步 `buf_.normed`
- 保留当前 attention kernel，不在本阶段同时重写 attention 核心

验收标准：

- `decode_profile_main` 中：
  - `full_qkv + full_o + post_attn` 合计下降 `>= 25%`
- `greedy_main` decode 达到 `>= 122 tok/s`
- `greedy_main` prefill 保持 `>= 3000 tok/s`

### 阶段 5：主线收口，decode 不再依赖 oneDNN matmul

目标：让目标模型的 decode 主线真正不再经过 `Linear::forward(seq_len==1)`。

清理对象：

- `Qwen3.5-0.8B` decode 热路径中对以下 oneDNN decode matmul 的依赖：
  - `linear_all_proj.forward`
  - `linear_o_proj.forward`
  - `qkv_proj.forward`
  - `o_proj.forward`
  - `gate_up_proj.forward`
  - `down_proj.forward`
- 与 decode-only 老实验路径相关的 compare / 临时日志 / 注释

保留原则：

- 只保留：
  - fixed-shape decode custom path
  - generic fallback（prefill / 非固定形状 / 非目标模型）
- 不保留“目标形状下默认不用但还能走”的 oneDNN decode 老路径

最终验收标准：

- `Qwen3.5-0.8B` 文本 decode 主线不再调用 `Linear::forward(..., seq_len=1)`
- `greedy_main` decode `>= 125 tok/s`
- `sample_main` decode `>= 100 tok/s`
- `perf_suite` 与 smoke 全通过

## Test Plan

固定继续按“少测、串行、正向后扩测”执行：

1. 每次结构性修改后只跑：

```powershell
.\build.ps1 -BuildDir build -Config Release
.\bench.ps1 -BuildDir build -Preset phase_gate_q35_text -Phase <phase_name> -CaseNames @('greedy_main')
```

2. 只有 `greedy_main` 正向时才补：

```powershell
.\profile_decode.ps1 -BuildDir build -Preset phase_gate_q35_text -Phase <phase_name>
```

3. 只有阶段收口时才补：

```powershell
.\bench.ps1 -BuildDir build -Preset phase_gate_q35_text -Phase <phase_name> -CaseNames @('sample_main')
.\perf_suite.ps1 -Preset phase_gate_q35_text -Phase <phase_name>
.\smoke.ps1 -Preset phase_gate_q35_text -Phase <phase_name>
```

4. 正确性要求：

- `greedy_main` 固定 seed 的 token 序列必须一致
- 临时对拍只允许保留在实现阶段
- 阶段收口时不保留 compare 入口在正常热路径中

## Assumptions

- 当前 decode 瓶颈首要矛盾是 dispatch overhead，而不是 GPU 纯算力不足
- 继续保留 oneDNN 作为 prefill / generic 路径实现是合理的
- 下一阶段的重点不是“手写所有通用 GEMM”，而是“手写 Qwen3.5 decode 固定形状专用融合 kernel”
- 当前 repo 已经具备 `joint_matrix` / subgroup kernel 经验，可直接复用现有 attention kernel 的实现风格
- 若某一波 custom kernel 仍明显打不过 oneDNN 且无法降低 dispatch 数，应立即止损，不长期把负收益路径留在主线
