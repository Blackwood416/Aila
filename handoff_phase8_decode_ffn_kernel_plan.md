# Phase 8 Decode FFN Custom Kernel Plan

## Summary

本轮按 Phase 8 思路验证了 `Qwen3.5-0.8B` 文本 decode-only FFN custom kernel，但当前结论是：

- `gate_up + swiglu` custom kernel 数值上基本正确
- 纯手写 BF16/F32 accumulate 的 FFN custom kernel 在当前 oneDNN + SYCL 栈下没有打过现有主线
- 已将热路径回退到原实现，避免把负收益路径留在主线上

当前回退后基线复核：

- `phase8_reverted_baseline_check`
- `greedy_main`: `prefill 1428.65 tok/s`, `decode 92.59 tok/s`

## Implementation Notes

本轮实现边界：

- 只在 `Qwen35HybridTextBackend` 内尝试
- 不修改 `Linear` 对外接口
- 不新增用户可见 API
- 不新增运行时 env 开关

### 实验 1：完整 custom FFN（首版）

- Phase: `phase8_ffn_custom_kernel_v1`
- 路径：`gate_up + swiglu` custom + `down_proj` custom
- 结果：
  - `greedy_main`: `prefill 1429.29 tok/s`
  - `greedy_main`: `decode 24.60 tok/s`

结论：

- 明显负收益，不能保留

### 实验 2：4 路 K 分片归约版

- Phase: `phase8_ffn_custom_kernel_v2`
- 路径：`gate_up + swiglu` custom + `down_proj` custom
- 结果：
  - `greedy_main`: `prefill 1454.38 tok/s`
  - `greedy_main`: `decode 54.65 tok/s`

对应 profile：

- `ffn_proj 6.568 ms`
- `ffn_act 0.564 ms`
- `down 9.333 ms`

结论：

- 比首版好很多，但 `down_proj` custom kernel 仍明显慢于现有 oneDNN decode 路径
- 当前最主要的负收益来源是 `down`

### 实验 3：只保留 `gate_up + swiglu` custom

- Phase: `phase8_ffn_gateup_only_v1`
- 路径：`gate_up + swiglu` custom + `down_proj.forward`
- 结果：
  - `greedy_main`: `prefill 1421.75 tok/s`
  - `greedy_main`: `decode 67.03 tok/s`

临时单层对拍结果：

- `phase8_ffn_gateup_debugcmp`
- `layer0 gate_up_swiglu max_abs=0.00390625`
- `layer0 gate_up_swiglu mean_abs=0.00003982`

结论：

- `gate_up + swiglu` custom kernel 不是明显算错
- 但即使只替换前半段，整体仍打不过现有主线

## Final Decision

本轮不合入 FFN custom kernel 热路径。

原因：

- 数值对拍基本正常，但性能持续负收益
- `down_proj` custom kernel 负收益最明显
- `gate_up + swiglu` custom kernel 即使单独启用也没有赢过当前 oneDNN 路径

因此本轮主线已回退，避免后续 benchmark 被失败实验污染。

## Next Suggestions

下一轮如果还要继续 FFN 专用优化，建议按下面顺序推进：

1. 不再继续保留当前这类“每线程多输出、纯 FMA”的 FFN custom kernel 版本
2. 若继续自定义 FFN，只尝试更接近 `llama.cpp` `mul_mat_vec` 的组织方式：
   - subgroup-only reduction
   - no-shmem 或极小 shared-memory 收口
   - 固定小列块 mat-vec，而不是当前写法
3. `down_proj` 不要再做纯 BF16 标量/FMA kernel 试探
   - 除非引入更接近 XMX/joint-matrix 的实现
   - 否则优先保留 oneDNN 现有 decode matmul
4. 后续所有 FFN 实验继续限制在 `Qwen35HybridTextBackend` 内，不扩散到 `Linear.cpp` 主线

## Validation Commands

本轮实际用到的验证命令：

```powershell
.\build.ps1 -BuildDir build -Config Release
.\bench.ps1 -BuildDir build -Preset phase_gate_q35_text -Phase phase8_ffn_custom_kernel_v1 -CaseNames @('greedy_main')
.\bench.ps1 -BuildDir build -Preset phase_gate_q35_text -Phase phase8_ffn_custom_kernel_v2 -CaseNames @('greedy_main')
.\profile_decode.ps1 -BuildDir build -Preset phase_gate_q35_text -Phase phase8_ffn_custom_kernel_v2
.\bench.ps1 -BuildDir build -Preset phase_gate_q35_text -Phase phase8_ffn_gateup_debugcmp -CaseNames @('greedy_main')
.\bench.ps1 -BuildDir build -Preset phase_gate_q35_text -Phase phase8_reverted_baseline_check -CaseNames @('greedy_main')
```
