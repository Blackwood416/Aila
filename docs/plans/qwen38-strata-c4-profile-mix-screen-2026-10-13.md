# Strata C4 筛选：路由画像工具链（多源归一化 mix）— NO-GO（增量不足）

日期：2026-10-13。范围：扫描文档 C4 的 Aila 最小实验。结论：导出/回存/留一已具备（P3），
但**多源归一化 mix 相对 agent6k 只有 +1.1pp**，低于"≥3pp 才进实模"的门。

## 1. 已有部分（P3，`4f09e87`）

- `AILA_Q4E_HOTSET_EXPORT`（adaptive，teardown 全量 24576 key，格式
  `AILA_Q4E_HOTSET 1 / identity / L E N / key decode prefill unique`）= Strata 的
  `--expert-profile-save` 等价物；profile 的**文件顺序**就是初始驻留排名
  （`Qwen4ExpHotsetPolicy::Save/Load`，`src/models/Qwen4ExpHotsetPolicy.hpp:135-165`）。
- P3 单源校准（agent6k）+ held-out A/B/B/A 已验证：misses −31%、experts_host −19%、
  submitted p50 −3.5%（端到端持平）。

## 2. 本次新增：离线 headroom（held-out 导出 + 排名覆盖）

对标准 14K+PI 工作负载做一次导出（`c4-export2-20261013`，绝对路径导出成功；
相对路径会因 worker cwd 不同而写失败——记录为坑），得到该工作的真实路由分布；再用
`tmp/q4e-strata-c4/rank_headroom.py`、`mix_eval.py` 计算各排名在 top-2792 槽下的
**decode route-slot 覆盖率**（该工作 decode 槽位总数 149,280）：

| 排名 | decode 覆盖 | 相对 agent6k |
|---|---:|---:|
| oracle（= 该工作自身分布排序，上界） | **62.74%** | +9.68pp |
| agent6k（P3 单源校准） | 53.06% | — |
| H4（出厂 profile） | 42.58% | −10.48pp |
| mix h4:0.15 / agent6k:0.85 | **54.17%** | **+1.11pp** |
| mix h4:0.30 / agent6k:0.70 | 53.60% | +0.54pp |
| mix h4:0.50 / agent6k:0.50 | 51.65% | −1.41pp |
| mix agent6k:0.5 / calib6k:0.5（同域另一条导出） | 54.07% | +1.01pp |
| mix agent6k:0.5 / calib6k256:0.5 | 53.04% | −0.02pp |

## 3. 结论

1. **门未过**：本仓库可用的校准源（agent 域导出 + H4 出厂）做归一化 mix，最好只有 +1.1pp；
   同域再混（不同 token 预算/不同运行的导出）≈ 0。
2. 剩余 headroom（9.68pp）主要是**实例特异**而非域特异：oracle 自己（用被评估工作训练）
   也只能覆盖 62.74%——2792/24576 = 11.4% 的容量是主约束；换排名能拿到的第 2 项收益
   ≈ 每 311 个 decode 步多 ~5 个 slot 命中，远低于端到端口径噪声（P3 实测 −31% misses 时
   端到端已持平）。
3. 与 Strata 结论不冲突：他们的 mix 之所以 +21%，是因为出厂 profile 与目标负载**不同域**
   （chat/code vs 摘要/多语言）。Aila 当前 agent6k 已是同域，等同域语料齐备后此项自然收敛。
4. **重新评估的前置条件**（非阻塞）：出现真正异域的常用负载语料（中文长文/文档摘要/代码仓库）
   时，按同一脚本重跑 mix 评估；若 mix 相对当时基线 ≥3pp 再做实模。

## 4. 复现

```powershell
python tmp/q4e-agent-plan-audit-20261012/run_val14k.py c4-export2-20261013 --agent --diagnostic `
  AILA_Q4E_HOTSET_EXPORT=E:/RiderProjects/Aila/tmp/q4e-strata-c4/heldout-14k.profile
python tmp/q4e-strata-c4/rank_headroom.py --heldout tmp/q4e-strata-c4/heldout-14k.profile `
  --candidate "h4=perf/qwen4exp/profiles/qwen38-nf4-h4-v1.profile" `
  --candidate "agent6k=perf/qwen4exp/profiles/qwen38-nf4-agent6k-v1.profile"
python tmp/q4e-strata-c4/mix_eval.py --heldout tmp/q4e-strata-c4/heldout-14k.profile `
  --source "h4=perf/qwen4exp/profiles/qwen38-nf4-h4-v1.profile" `
  --source "agent6k=perf/qwen4exp/profiles/qwen38-nf4-agent6k-v1.profile" --weights 0.15,0.85
```
