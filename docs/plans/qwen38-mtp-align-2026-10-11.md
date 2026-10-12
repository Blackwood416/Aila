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

## B4-⑤：会话缓存态定位（结果：状态相同、触发条件收敛到 stop 结束路径）

- **持久状态逐字节对比**（扩展 `--mtp-align`）：一次 4 行 verify 窗口 vs 4 次普通 count=1 decode，结束后 24 层 KV、indexer k/compressed、GDN state+conv、PLE conv_state（含 `conv_head`/`prev_tokens`）、`session no_miss`、每层 `num_compressed_blocks` → **differing_tensors = 0**：窗口算术在等长度下与 T1 状态完全一致，分歧不在窗口内部。
- **触发条件 = stop-token 结束**（全尺寸 A/B 只改 `--generate`）：

| 运行 | 请求 1 结束 | 首请求 | 续轮 |
|---|---|---|---|
| `--generate 8` | length（ctx=1432） | 逐字节相同（sha `b2bd7b018f8d`） | **逐字节相同** |
| `--generate 128` | stop（126 tok，ctx=1550） | 逐字节相同（sha `f72a0da0fb2f`） | **分歧（128 vs 111 tok）** |

- opt-in 记账日志（`AILA_Q4E_WINDOW_ACCOUNT_LOG`）显示两臂请求 1 均为 emitted=126 / ctx=1550 → 不是长度 off-by-one，而是 **stop 结束分支的会话边界/内容差异**（"MTP 窗口 + bonus/接受提交 + `PublishSample` stop 分支跳过 Finalizing flush" 记账链）。
- 结论：等价门仍未通过（首请求相等、stop 场景续轮分歧）；MTP 保持默认关闭。下一步（B4-⑥）：把快照扩到 (a) adaptive 驻留槽/路由计数、(b) draft `mtpState`（KV 行 + `lastHidden` + 可见长度），dump 续轮第一个 forward 前后；`--generate 128`（stop）复现、`--generate 8`（length）对照——若是 stop 分支 boundary 则小修可闭；若是 session 复用 draft 态则需失效/重建。

## B4-⑥：状态快照扩展 + 活线 boundary 探针（结果：等价门未过，根因钉到 verify 行）

- 新探针 `AILA_Q4E_MTP_BOUNDARY_DUMP`（提交 `40d6bf6`，默认关闭）：每次 `ForwardImpl`/`ForwardWindow` 前后 dump
  session 指纹（24 层 KV、indexer k/压缩、GDN state+conv、PLE、计数器）+ adaptive 槽表/路由计数 + draft `mtpState`。
- 结果（全尺寸 A/B，`--generate 128` stop 场景，两轮独立复现 b6/b6b）：
  - 两臂请求 1 文本逐字节相同（sha `f72a0da0fb2f`，126 tok，ctx=1550），但**请求 1 结束时的持久状态已逐位不同**；
    续轮（A `d76fb23bc0` 128 tok vs B `86c43a4f98` 111 tok）是下游后果——**B4-⑤ 的 "stop 结束路径" 假设被否证**。
  - 首分歧 pos=1492（1491 相同）= 请求 1 内 start=1491 的**被拒绝 MTP 窗口 row0**：该行由 2 行 verify forward 写出，
    token 选择一致，但整条残差流与其下游张量与单行 decode 不逐位相同；漂移永久保留到续轮 resume 点 1550 与 1574。
  - 分歧集合 `kv/kvv/idxk/idxc/gdn/gdnc` 同时不同；`idxn`/`ple`/`scal` 全程相同 → 是逐行数值差，非长度/计数记账。
    逐层：full-attention 12/12 层 KV 全分歧，linear 层 GDN 从 layer 2 起分歧（layer 0/1 相同）。
- 排除项（全尺寸差分否证）：stop/bonus/Finalizing 记账、draft 态复用、adaptive 驻留槽集合、压缩块计数，以及
  `GDN_HISTORY_FUSED`/`DECODE_PENDING_MLP_RESIDUAL`（b6c 关闭后两臂状态逐位不变、首分歧仍在 1492）。
- 结论：残留唯一解释 = **verify 窗口的单行路径与单行 decode 不逐位等价**（状态相关的稀有行）。闭环选项：
  (a) env 化 live stage 探针在 1491 行定位具体算子；(b) verify 窗口按逐行 `count==1` 完整 decode 体执行
  （构造上逐位等价，代价 = 放弃核内多行收益，B4-③ 吞吐需重做）。
- B4-③ 判定：**等价门未通过 → MTP 维持默认关闭**（仅 `AILA_Q4E_SERVER_MTP=1` opt-in）；吞吐沿用 MTP ~15.0–15.3 vs T1 15.6 tok/s。
- 证据：`tmp/q4e-mtp-alignment-20261011/{candidate-b6.patch,run_mtp_ab_b6.py,analyze_bnd*.py,summarize_bnd.py,content_sha.py,b6-*,b6b-*,b6c-*,analysis-*.txt,b6-harness-prod.log,ctest-cpu-abi-b6.log}`；
  官方 pair `retained5`（`40d6bf6`，worker `436719a8…` / server `82336e2f…`）；CPU/ABI 19/19。

## 移交

- 继续项（按路线图顺序）：① B1 int8 KV DPAS（设计见 `tmp/q4e-int8kv-dpas-20261012/DESIGN.md`，s8 组合已探明 n=8/k=32）；
  ② B2 prefill W4A8；③ B3 卸载结构重整；④ MTP 闭环只在 B3 之后按上面 (a)/(b) 选项重评（当前多行 verify 既慢又不等价，不投入）。
- 工具：`tests/models/Qwen4ExpMtpAlign.hpp` + `--mtp-align` 驱动；逐阶段探针仅在 `verification_stage_rows_for_test_>0` 时触发，普通 forward 零改动。
- 证据：`tmp/q4e-mtp-alignment-20261011/{candidate.patch,REPORT.md,align-*.log,ctest-cpu-abi.log,run-*.ps1}`。
