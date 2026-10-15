# Qwen3.8 INT8 KV：读取路径向量化 + DPAS 复查记录（B1，2026-10-12）

基点 `40d6bf6` + 补丁（`6deaefc`，仅 `src/ops/AttentionOps.cpp`）。env：沿用 `AILA_KV_QUANT=1
AILA_KV_QUANT_FORMAT=int8`；新增 `AILA_KV_QUANT_VEC`（int8 路径默认 1，设 0 回到逐元素 accessor 读取做 A/B）。

## 1. 动机与改动

`1f6c753` 的 int8 KV 是"读时逐元素反量化"：每个 d 都重新算下标、读 scale（global load）、读 1 字节码、
再 I2F+FMUL。14k 因此比 bf16 慢 17%（1.845 vs 1.573 ms）。

本次只改**取数**，不改数值：

- K 行：per-token scale 提出 d 循环（每次 token 1 次 scale load）；码用 `sycl::vec<int8_t,8>` 8 字节
  向量加载；每项仍是 `q * (float(code) * scale)`，累加顺序不变。
- V 累加：改为 token-major（`head_dim == 2 * tile_wg` 时每线程 2 个 d 累加器），每 token 一次 scale
  load，其余同上。

## 2. 内核级结果（A770，Hq24/Hkv2/D256，B=2，20 样本中位数，同二进制单变量）

| len | bf16 | int8 VEC=0 | int8 VEC=1 | VEC 收益 | vs bf16 |
|---:|---:|---:|---:|---:|---:|
| 1834 | 0.4491 ms | 0.5293 ms | **0.3466 ms** | **−34.5%** | **−22.8%** |
| 14336 | 1.5833 ms | 1.8543 ms | **1.0532 ms** | **−43.2%** | **−33.5%** |
| fp8 14336（参考） | — | — | 3.2996 ms | — | （int8 比 FP8 快 3.1×） |

精度行（`[KV ACC]`，14336）：int8 开/关两臂 **逐字相同**（max_abs 6.1e-5、NRMSE 0.00510548、
cos 0.999987；FP8 同长度 NRMSE 0.0252639 / cos 0.999682）→ 数值不变，改动是纯读取优化。

证据：`tmp/q4e-int8kv-dpas-20261012/{kv-vec-off.log,kv-vec-on.log,kv-vec-on2.log}`；三臂回归
`--kv-fp8-only` 全过（`[KV THREE-ARM PASS]`）。

## 3. 生产启用现状（阻塞与归属）

- 服务器/worker 明确拒绝非 BF16 KV：`src/inference/Model.cpp:702`（"Qwen4Exp shared sessions require
  BF16 KV"）+ `src/worker/InferenceWorker.cpp:546`（capability 为空）。生产 A/B 首启即失败
  （`tmp/q4e-resident-speed-20261009/int8kv-b1-aborted/`，4s，server 退出码 1）。
- **2026-10-12 已落地（`8101788`）**：IPC capability `qwen4exp_nf4_int8_kv` + `--kv-type int8`/`AILA_SERVER_KV_TYPE`
  + `allowQuantizedKv` 显式放行 + **reservation 按 int8 布局重算**（含 per-token F32 K/V scales，逐字节对齐 session
  分配；此前 "Session cache reservation did not match allocation" 会拒绝每个 session）+ `AILA_Q4E_MAX_GPU_SLOTS`
  放开 2800 硬上限。默认 bf16 生产冒烟（14K+PI+cancel-recover）通过。
- **剩余阻塞（B3-2）**：`attention_prefill_qsa_batched requires bf16 KV cache`（`AttentionOps.cpp:4516`，
  生产 prefill 的 QSA 批量内核仍只读 bf16）。INT8 run 现在能加载/建 session/跑到 PI 请求才在此抛错。
  需把该内核（以及 `attention_decode_rows`/pair/QSA 稀疏路径）按 `KVCacheAccessor<Fmt>` 模式补齐 int8。

## 4. DPAS 复查：SG 配置错误，工具链阻塞结论撤回

- 原探针始终使用 SG16，attention 补丁则继承构建的默认 SG16；没有测试 SG8。
  `IGC: Internal Compiler Error: Divide by zero` 是该配置的失败，不能推导出所有 int32 joint_matrix 不可用。
  原始失败证据仍保存在 `tmp/q4e-int8kv-dpas-20261012/`。
- 同一 A770（驱动 `32.0.101.8724`）和 oneAPI 2026.1 上，显式 SG8 的 s8×s8→s32 探针
  已通过 M=8/M=1、K=256（8×32）、N=8、global/SLM 和直接 col-major Kᵀ 的 136 项整数精确比较。
- 归档 WIP 另有分块累加结果覆盖和尾部 N8 越界读取问题。已修复为每个 SG8 一块 N8、独立 base/residual
  结果区、短尾块标量回退；Q 使用双遍残差，沿用既有 KV scale 和 partial merge。
- 新入口 `AILA_KV_QUANT_DPAS=1` 默认关闭，仅覆盖 D256 exact decode 的连续 KV 读取。
  正确性回归通过；三轮复测 14k wall 为 1.922～1.931 ms，现有向量读取为 1.043 ms，
  DPAS 慢约 84.6%，因此保持默认关闭。
  完整复查、适用范围与复测数据见 [DPAS 修复记录](qwen38-int8-dpas-repair-2026-10-12.md)。
