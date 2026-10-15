# Qwen3.8 ESIMD/SG8-DPAS prefill MoE：过门后接入清单（2026-10-12）

配套文档：接入审计 `docs/plans/qwen38-esimd-integration-audit-2026-10-12.md`（下称"审计"，
章节号引用它）；探针交接 `tmp/q4e-esimd-dpas-20261012/handoff-draft-notes.md`。

范围：本清单只覆盖**探针 P0 过门之后**的动作。P0 未过门（ESIMD 同形状吞吐 <1.20× SG8）→
直接走 §6 分支，不进入 §2–§5。纪律：默认 off；每步单变量；不动数值策略；不放宽既有断言；
任何"方案/待验证"动作都必须先有证据文件再改代码。

## 0. 触发条件（先判定，再动手）

- 探针结论必须包含：选定 geometry（M/N/K）、operand 顺序与 VNNI 公式证据、SG8 与 ESIMD
  **同形状**吞吐比（≥1.20×）、编译/JIT/ISA 记录、int32 精确性证据（对应审计 P0 行）。
- 结论缺任一项 → 退回探针补测，不进入接入。
- 参与对比的 SG8 实现本身也要在探针里过同一正确性口径（审计 §5.1：能力检查不能照抄
  attention 的 M1 guard）。

## 1. 冻结与记录（0.5 天）

- 1.1 探针结论写入 `docs/plans/`（文件名含日期），附原始日志与可复现命令。
- 1.2 固化被测形状表（如 M=8/16/32/…、N=8、K=32 链表）；N16 保持"未验证"，不得因头文件
  支持而进生产。
- 1.3 记录 IGC/驱动/编译器版本，以及 `-fsycl-default-sub-group-size=16` 对 ESIMD 内核的
  实测影响（本构建该 flag 是 target 级全局项）。

## 2. Kernel 级接入（默认 off；tests 由探针代理负责，主会话只复核）

- 2.1 `tests/ops`：新入口与现有 wgshare bit-exact 测试**并列**，不改旧断言（审计 §6.1
  原 wgshare 行 / §4.1 三层契约表）。
- 2.2 `tests/ops`：新增 stage/pack/projection 的整数 oracle 对照（审计 K1/K2 行）；
  MoE tile 端到端 epilogue 与独立 FP64 参考对照（冒烟）。
- 2.3 `src/ops`：候选实现放独立 TU（SG8 与 ESIMD 分开，如 `MoeExpertDpasOps.cpp` /
  `MoeExpertEsimdOps.cpp`），共享同一量化/LUT/scale/epilogue 契约；不把 tests 原型头
  拖进生产 TU（审计 §5.2 第 2 行）。
- 2.4 env 只加两个核心 read-once：`AILA_Q4E_PREFILL_MOE_W4A8`（0/1/2）与
  `AILA_Q4E_PREFILL_MOE_DPAS`（0/1/2）；研究用 `..._MIN_ROWS` 默认保守并声明"待测起点"
  （审计 §5.1）。

## 3. Backend 接入（单变量、可回退）

- 3.1 scratch 预算：XQ8/XS、权重 stage、VNNI、H Q8 分别核算；**先进 cache fit（
  `backend:723` 同一预算口径）再定 resident 数**；预分配落在
  `ensure_runtime_buffers` 容量边界，热路径零分配/零 descriptor 重绑（审计 §3.4/§7）。
- 3.2 接入点：`RunAdaptivePrefill → RunGroupedExperts` 的单专家桶边界构 tile；保留
  token/rank map 与 ordered merge、map slice 不重叠约定（审计 §2.3/§5.2）。
- 3.3 生命周期：stage/pack/projection/down 全部纳入 last-event 链；取消/trim 走既有
  drain 路径（审计 §7 前四行）。
- 3.4 compact output：第一阶段**整请求拒绝候选**（审计 §2.2/§5.1；wgshare wrapper 固定
  `Compact=false`）。
- 3.5 小桶与尾行（live rows <8、M 尾部）：同算术 vector 或零 padding tile，不允许退回
  NF4/BF16 混算术；否则重新引入"桶形状影响前缀"的风险（审计 §5.1）。
- 3.6 能力判定：startup 一次性查询 s8×s8→s32、N8/K32、SG8 组合并按 thread/device 缓存；
  不允许在热循环查询 env/capability。

## 4. 测试矩阵（逐层全过，或明确记录失败）

- 4.1 按审计 §6.1 顺序执行：K1→K2→K3→M1→M2→L1→Q1→B1→B2；每层保留原始日志。
- 4.2 Q1 必须**强制 A/B 两臂**（独立 setter/override + 重置 session），不能用同进程静态
  env 切换冒充（审计 §4.2 的警告）。
- 4.3 K3/Q1 已知历史难点（逐点 abs/rel 超限）不因新 ISA 过门而豁免：先修，或另立经产品
  接受的质量契约（审计 §4.1/§6.1 K3 行）。
- 4.4 所有 bit-exact 回归（原 wgshare、prefill 输出、状态对比）保持不放宽。

## 5. 生产口径验收（B1/B2）

- 5.1 块级：`build-refactor/AilaQwen4ExpKernelTests.exe --prefill-bench-only`（先 correctness
  后 bench）；新 direct-ops bench 必须打印 median/分位与采样定义（审计 §6.2）。
- 5.2 生产：`tmp/q4e-int8kv-dpas-20261012/run_chunk2048.py` 三臂（W4A8=1/DPAS=2、SG8=1、
  同算术 vector=0 控制）+ off/on 交错 ≥3 稳定轮；主门 = 同 label `first_content_seconds`/
  TTFT 下降 ≥2%（审计 §6.3）。
- 5.3 同期记录：USM device/pinned 峰值、DXGI 实际占用、resident/cold 计数、cold H2D 增量、
  stage 命中计数、输出 hash/tokenIDs/usage/finish_reason 一致性；首次请求与 reuse 请求分开
  统计。

## 6. 决策规则（与审计 §6.4 一致）

| 结果 | 动作 |
|---|---|
| ESIMD raw ≥1.20× SG8，块级 ≥1%，墙钟 ≥2%，全部质量/状态门过 | 默认启用（保留 SG8 备选） |
| ESIMD raw ≥1.20× 但块级 <1% 或墙钟 <2% | 默认 off，opt-in 归档负结果 |
| ESIMD raw <1.20× SG8 | 关闭 ESIMD 分叉；若 SG8/vector 已过门则只保留它们 |
| 任一质量门（K3/Q1）或前缀因果失败 | 停止推广；修同算术或退出候选，不得调阈值 |

## 7. 归档与回滚

- 7.1 负结果归档：`docs/plans/` + 默认 off + opt-in env 保留；探针证据目录不删。
- 7.2 回滚路径：两个 env 全 0 应回到原 NF4 路径，并在同 build 上复跑既有 bit-exact 回归。
- 7.3 采纳节奏：沿用工程惯例——每个采纳改动单独提交，随后重建官方 worker/server pair
  并归档 build-id。

## 8. 分工与边界（本次口径）

- 探针与 tests 归属：Hooke（探针代理）；主会话不写 `tests/`、不跑 GPU，只做审计/验收复核
  与文档。
- 主会话交付：本清单 + 审计复核记录（审计 §9）+ 探针交接页
  `tmp/q4e-esimd-dpas-20261012/handoff-draft-notes.md`。
