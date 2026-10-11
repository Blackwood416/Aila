# B4-②：MTP 多行核 T1 对齐（里程碑：根因定位 + layer-0 逐位）

日期：2026-10-11。基点 `HEAD=176aede` + 补丁（`tmp/q4e-mtp-alignment-20261011/candidate.patch`；另补 B2 提交遗漏的 `tests/ops/Qwen4ExpW4A8Tests.hpp` 跟踪）。
env：`AILA_Q4E_VERIFY_SMALL_ROWS`（默认 0=关，仅影响 2..4 行 `decode_verification` forward）。

## 根因（重要）

T1 与多行 verify 的分歧**主要不是算子舍入，而是调度**：`count≥2`（含 verify 窗口）把 attention/shared 投影、HC、PLE、LM head 走 prefill/oneDNN GEMM 路径；`count=1` 走单行 NF4 decode GEMV。两者从 **layer 0 row 0** 起就逐位不同；历史记录的"首层 GDN 末态差异"是其下游投影产物。width=2 与 width=4 在所有探针上逐位相同 → 分歧只与 `count≥2` 相关。

## 修复（opt-in）

`AILA_Q4E_VERIFY_SMALL_ROWS=1`：在 2..4 行 verify forward 内把 attention/shared 投影、HC（`hc_*_rows`）、PLE key/value 投影、LM head 切到**单行 decode 核**；并把 `VERIFY_DECODE_PREPARE` 的覆盖从 `count==2` 放宽到 2..4。默认路径零改动。

## 证据（prompt=512、context=2048、rows=4）

| 首个分歧层 | row0 | row1 | row2 | row3 |
|---|---|---|---|---|
| baseline（仅参考开关） | 0 | 0 | 0 | 0 |
| `SMALL_ROWS_DECODE=1` | 1 | 0 | 0 | 0 |
| `SMALL_ROWS_DECODE=1 + SMALL_ROWS_HC=1` | 1 | 1 | 1 | 1 |
| **`VERIFY_SMALL_ROWS=1`（本补丁）** | **1** | **1** | **1** | **1** |

- layer-0 GDN 末态：473,268 个元素 diff / max_abs 2.12e-4 → **0（逐位）**；
- layer-1 残差 max_abs 1.25e-3 → 7.46e-4；全部 4 行 top-1 与 T1 贪心一致；
- CPU/ABI ctest 19/19（`ctest-cpu-abi.log`）。

## 未覆盖（B4-③ 仍未开）

- layer≥1 的 router gate、shared-expert gate、MoE rank 合并、indexer 投影仍是 count 相关 GEMM → 修复后首个残差从 layer 1 开始。要全逐位需把这些也纳入 `VERIFY_SMALL_ROWS`（或等价逐位路径）。
- `VERIFY_ATTN_PAIR`（`start_pos+1≥1024 && count==2`）prompt=512 未触发；需 ≥1100 token prompt 覆盖。
- 未跑生产 harness/server，未做性能回归。

## 移交

- 继续项：① layer≥1 阶段纳入 small-rows；② 长 prompt（≥1100）覆盖 attention-pair/dense-QSA；③ 全过后开 B4-③（协议见路线图）。
- 工具：`tests/models/Qwen4ExpMtpAlign.hpp` + `--mtp-align` 驱动；逐阶段探针仅在 `verification_stage_rows_for_test_>0` 时触发，普通 forward 零改动。
- 证据：`tmp/q4e-mtp-alignment-20261011/{candidate.patch,REPORT.md,align-*.log,ctest-cpu-abi.log,run-*.ps1}`。
