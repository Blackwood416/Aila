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

## 第二段：全 48 层逐位对齐（完成）

把 layer≥1 仍走 count 相关 GEMM 的阶段全部纳入 `VERIFY_SMALL_ROWS`：
- `Qwen4ExpBnb4Backend.cpp`：router gate、shared-expert gate 在 verify 窗口内逐行走 `bf16_gemv_bf16`；shared gate_up/down 逐行走 count==1 融合链（count==1 分支代码原样）。
- `Qwen4ExpPrefill.cpp`：indexer k / qk 投影在 verify 窗口内逐行。

### 结果（两种 prompt、两种宽度，全部逐位）

| prompt | width | 结果 |
|---|---|---|
| 512 | 4 / 2 | rows 0..3 全部 `logit_different=0`、所有 stage `first_diff_layer=-1`、无任何 `[MTP ALIGN STATE]` 行（全 48 层 GDN 末态逐位） |
| 1182（触发 `VERIFY_ATTN_PAIR` active=1） | 4 / 2 | 同上全部逐位（dense-QSA 索引分支同） |

### B4-③ 开启条件（实测最小集，均可逐位复现）

必需：`AILA_Q4E_VERIFY_SMALL_ROWS=1`、`AILA_Q4E_VERIFY_GDN_ORDER=1`、`AILA_Q4E_VERIFY_DECODE_PREPARE=1`、`AILA_Q4E_VERIFY_DECODE_ATTN=1`、`AILA_Q4E_ARENA_VERIFY_ROWS=1`。
推荐：`AILA_Q4E_VERIFY_ATTN_PAIR=1`（长上下文 count==2 pair 核；关掉退回逐行 `attention_decode`，同样逐位）、`AILA_Q4E_VERIFY_ASYNC_ROUTE=1`、`AILA_Q4E_VERIFY_CPU_ROWS=4`。

## 残余风险（未覆盖）

- 每个 prompt 只测一个窗口位置；多窗口连续 verify 的跨窗口状态累积未测。
- >2048 sparse-QSA 分支未覆盖（MTP 资格为 dense ≤2048）。
- 未跑生产 harness/server，未做性能回归——verify 窗口投影逐行执行**预期降吞吐**（B4-③ 需实测）。

## B4-④：活线分歧定位与修复（2026-10-11 晚）

三个候选逐一裁决：

| 候选 | 结论 | 证据 |
|---|---|---|
| `VerifyRows(1) != 普通 decode` | **否证** | 新增 `AILA_Q4E_MTP_ALIGN_T1=decode`；prompt=512 下 `VerifyRows(4) == 4×普通 decode` 逐位（logits + 48 层 GDN 态 + 全 stage 探针），叠加 B 臂数值开关后仍逐位 |
| draft 共享状态污染 | **否证** | 进程内活线 MTP（含拒绝 + state-only replay）`single_row_target_mismatches=[]`；小规模真实 A/B 内容 hash 相同 |
| 活线首个分歧 | **定位 = attention-pair 核** | `ATTN_PAIR=1` → 119 tok "a large number"；`ATTN_PAIR=0` → 首请求与 T1 **逐字节相同**（sha `f72a0da0fb2f`，126 tok "a series"）。prompt=1447 复现：width=4（逐行 `attention_decode`）逐位；width=2（`q4e_attention_decode_pair`）从 layer 15/16 起分歧 → **长 KV 下 pair 核 partial-merge 与两次单行不等价** |

**修复（opt-in）**：`Qwen4ExpPrefill.cpp:513-517` pair 核条件追加 `&& !verify_rows_alignment_`——`VERIFY_SMALL_ROWS=1` 时 2 行 verify 窗口回避 pair 核。修复后 prompt=1447 width=2/4 全 rows 逐位、全尺寸 A/B 首请求与 T1 逐字节相同。

**残余阻塞（B4-⑤ 目标）**：续轮（prefix reuse）仍分歧（111 tok vs T1 128 tok），尽管续轮 prompt_tokens/cached 与 T1 相同（1574/1550）→ 差异在 MTP 请求结束后的**会话缓存态**（KV/状态内容），非提示词或 pair 核。

**工作方式修正**（用户反馈）：已移除测试命令中的固定 ~420s 硬等待，改为 1–2s 轮询完成条件（进程/`result.json`/日志结束标记，上界 900s），长测试前台运行；受影响的运行已用新方式补齐。

## 移交

- 继续项：① layer≥1 阶段纳入 small-rows；② 长 prompt（≥1100）覆盖 attention-pair/dense-QSA；③ 全过后开 B4-③（协议见路线图）。
- 工具：`tests/models/Qwen4ExpMtpAlign.hpp` + `--mtp-align` 驱动；逐阶段探针仅在 `verification_stage_rows_for_test_>0` 时触发，普通 forward 零改动。
- 证据：`tmp/q4e-mtp-alignment-20261011/{candidate.patch,REPORT.md,align-*.log,ctest-cpu-abi.log,run-*.ps1}`。
