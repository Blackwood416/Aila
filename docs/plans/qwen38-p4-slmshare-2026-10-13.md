# P4：prefill 解量化复用（同专家多 quad 共享 FP32 tile）— NO-GO

日期：2026-10-13。范围：计划 §5-P4 的单一候选（gate/up，share=4）。结论按门否决：
同数学完整专家链 wall **−11%~−18%**（需 ≥10% 正收益），out-of-tree 保留为 opt-in 诊断。

## 1. 判据设计（三级，先否定后实现）

1. **oracle 上界**（`_nounpack`）：复制 wgshare gate/up 内核，把 16 行
   `qmap[nib] * scale` 换成 `= scale`（运行期未知常数）。packed 字节读、nibble 提取、
   LUT gather、mul 全部成为死代码，**仅保留全部 FMA**。任何"同数学解量化复用"方案
   的理论上界（且它还额外拿掉了不可共享的 packed-load 链，属刻意过高估计）。
2. **单变量判别**（`_nolut`）：只把 16 次 LUT gather 换成 `static_cast<float>(nib)`，
   packed 读与整条依赖链保留——分离"可共享的 LUT 部分"与"不可共享的 packed 链"。
3. **逐位精确实现**（`_slm`）：share=4 行组内 64 lane 协作解包当前 64B chunk 的
   128 gate + 128 up 权重到 SLM FP32 tile（16 lane × 9 float 加垫、高 nibble 优先、
   与原式相同的 `qmap[·] * scale`），4 个 subgroup 消费；混专家行组、行尾、blocksize≠64、
   per-lane 余块全部回退原解包路径。barrier 全程 WG-uniform（统一 64B chunk 行程），
   FMA/归约/ordered-write 序列与原内核逐字节相同（生成器从原文抽取这些 span）。

## 2. 结果（real shape：H=2560, I=640, E_pool=32, top-k=10, block64；gate/up only；
3 轮交错 A/B/B/A，每样本 20 次 launch，取中位数；`AilaQwen4ExpKernelTests --prefill-bench-only`）

| 臂 | tokens=16 (32 quads) | tokens=64 (160 quads) |
|---|---:|---:|
| 原 wgshare share=4 | 0.4017 ms | 1.3809 ms |
| `_nounpack` oracle | 0.1300 ms（−68.1%） | 0.6136 ms（−54.4%） |
| `_nolut` 判别 | 0.3469 ms（−13.6%） | 0.9244 ms（−33.1%） |
| `_slm` 实现 | 0.4472 ms（−11.1%） | 1.6041 ms（−17.7%） |

- 逐位一致：`_slm` vs wgshare share=4 全缓冲 bit-exact PASS（tokens=64：409,600/409,600；
  tokens=16：64 个遗留单 act 槽位两臂都不写，预填充同值后 81,920/81,920）。零新 USM（仅 SLM）。
- 完整链参考（同 run，gu+dn+ordered，share=0/4）：tokens=64 = 2.175 ms（share=4）。
  gu 占 62%；`_slm` 的 gu 负收益换算到完整链约 −11%，**未过 ≥10% 正收益门**。

## 3. 机理（为什么 SLM 复用收不回成本）

1. 消费者每 lane 每 chunk 仍需 16 次 SLM 读（tile 读），与原 16 次 LUT gather **同量**；
   SLM 方案只能把 3/4 的 nibble 解码 + scale mul 移到生产者（4 次/ lane）。
2. `_nolut` 给出"可共享部分"的真实上限：tokens=64 时 33%（gu 内），tokens=16 时仅 13.6%。
   其余差值属于 packed 字节读+nibble 提取+依赖链，任何共享方案都必须保留。
3. 生产者/消费者相位分离 + 每 tile 2 次 barrier 的固定开销（tokens=64 摊到每 WG ≈2.9 µs）
   超过其省下的解码/mul；不试 double-buffer 救援：即便开销清零，16-token 形状的上限
   （13.6% gu ≈ 8% 链）也低于门。
4. 由此关闭"减少重复解量化"方向；同族旧负结果（half-unpack、字节对 LUT、qmap 私有寄存器）
   互相印证：本 GPU 上该内核受 packed-load/解码依赖链支配，而非纯 ALU 表查。

## 4. 复现

```powershell
# 生成三个诊断内核（幂等：先 git checkout src/ops/MoeExpertOps.cpp 再跑）
python tmp/q4e-agent-plan-audit-20261012/gen_nounpack_variant.py
python tmp/q4e-agent-plan-audit-20261012/gen_slm_variant.py
python tmp/q4e-agent-plan-audit-20261012/gen_nolut_variant.py
pwsh -NoProfile -File tmp/q4e-w4a8-20261010/build-target.ps1 -Targets AilaQwen4ExpKernelTests
$env:AILA_Q4E_NOUNPACK_PROBE='1'; $env:AILA_Q4E_NOLUT_PROBE='1'; $env:AILA_Q4E_SLM_REUSE='1'
build-refactor\AilaQwen4ExpKernelTests.exe --prefill-bench-only
```

- 日志：`tmp/q4e-agent-plan-audit-20261012/nounpack-probe-run1.log`、`slm-run1.log`、
  `nolut-run1.log`；生成片段：同目录 `slm_variant_generated.cpp.txt`。
- 代码：`src/ops/MoeExpertOps.cpp` 三个 `*_wgshare_{nounpack,nolut,slm}` 内核与 dispatcher，
  头文件声明注明"timing diagnostic only, never used for real inference"；生产代码零引用，
  仅 `tests/ops/Qwen4ExpKernelTests.cpp` 的 env-gated 基准臂调用。
