# Qwen3.8 INT8 KV cache：原型、三臂微测与结论（B1）

日期：2026-10-10。基点 `HEAD=d7b0649` + 原型补丁（`tmp/q4e-int8kv-20261010/candidate.patch`，9 文件 +702/−175）。
env：`AILA_KV_QUANT=1` + `AILA_KV_QUANT_FORMAT=int8`（方案 B：保留布尔开关，新增格式变量；默认 fp8/未启用时路径逐字节不变）。

## S0：先修两处既有 FP8 缺陷（已在本次一并修复）

- `ElementwiseOps.cpp:237`、`:1091`：`head_dim==128` 快路径只写 K/V 前 64 维，后 64 维从未落盘（decode 与 partial 两个变体）。修复后编码完整。
- `AttentionOps.cpp:1503`：单行 `attention_decode` 缺量化守卫，会误入 bf16-only 的 joint-matrix 内核；现改为回退 baseline。
- 证据：`[S0 PASS]`（canary 填充的 fp8 缓存 + bf16/fp8 code 等值比对）、`[FP8 CODEC PASS]`、`kv-run-final.log`；负向对照（仅移除后半维写入）以 `mism=128` 失败，证明测试有判别力。

## 三臂微测（A770，Hq=24/Hkv=2/D=256，B=2，交错计时中位数）

| len | 指标 | fp8 | int8 | bf16 参考 |
|---|---:|---:|---:|---:|
| 333 | NRMSE / cos / ms | .0264 / .99965 / 0.594 | .0052 / .99999 / 0.457 | — / — / 0.461 |
| 1834 | NRMSE / cos / ms | .0227 / .99975 / 0.797 | .0047 / .99999 / 0.515 | — / — / 0.449 |
| 14336 | NRMSE / cos / ms | .0253 / .99968 / 3.288 | **.0051 / .99999 / 1.840** | — / — / 1.574 |

结论：**同字节下 INT8 精度约为 FP8 的 5×**（NRMSE 0.005 vs 0.025；cos 0.99999），完全满足既有容差契约（cos ≥ 0.995）；但当前"读时逐元素反量化"实现使 **int8 比 bf16 慢 17%**（14k：1.84 vs 1.57 ms），FP8 更慢（2.1×）。就速度而言 INT8 KV **不适合**以此形态上线；其价值在（a）精度显著优于 FP8、（b）KV 显存 768→384 MB、（c）为未来无解量 int8（DPAS）读路径铺路。

## 覆盖与显式拒绝（原型范围）

- 支持（opt-in）：`copy_to_cache`、`attention_decode`（baseline + exact-256 merge）、`attention_prefill_cached`、`decode_prepare_qgkv_packed_partial`；存储 = `s8 {2,cap,256}` + **per-token** `f32 {2,cap}` scale（per-head 标量在增量 decode 下会失效）。
- 显式 `invalid_argument` 拒绝：`attention_decode_rows`、`decode_prepare_qkv(_partial)`、QSA 稀疏 prefill、`q4e_attention_decode_pair`、缺 scale 的 int8 读取。默认（env 关）保持 bf16 且逐字节一致。

## 结论与下一步

- 维持 **opt-in、默认关闭**；暂不安排生产三臂 LIVE（内核级已显示读取路径净负）。
- 若继续 B1：实现**无解量 int8 读内核**（`sint8` DPAS，Device.cpp:16 已报告支持；Q 同步 int8 化 → W8A8 attention），再按"NRMSE ≤ FP8 且不慢于 bf16"复测；通过后再做生产三臂。
- 记录限制：端到端 sanity 未做（代理环境未定位到 Qwen3.8 模型目录；主模型路径为 `C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2`，后续可直接使用）。

## 验收与证据

- CPU/ABI 19/19 通过（`ctest-cpu-abi.log`）；`AilaQwen4ExpKernelTests` 全过（`kernel-regression.log`）。
- 原始证据：`tmp/q4e-int8kv-20261010/{candidate.patch,REPORT.md,kv-run-final.log,s0-teeth-negative.log,ctest-subset.log}`。
