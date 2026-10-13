# Qwen3.8 优化路线图：2026-10-12 状态汇总（含未完成项与前置）

基点 `HEAD=2bbf628`；官方 pair `retained8/c67edc7` + 本文件提交后重建 `retained9`。
口径：生产序列 `run.py <tag> --agent`（14K prompt + PI agent + 32-token 续轮 + cancel-recover），
单变量 A/B/B/A，14K decode 噪声底约 ±0.6%。

## A 组（本轮在办）

| 项 | 状态 | 结论/数字 |
|---|---|---|
| A1 专家 prefill 数据流 | 部分采纳 | WG-share=4 已默认（14K ttft −2.94%，`e808f63`）；share=8/8-token 窗口已否证；二维分工未做 |
| A2 CPU 专家路径 | 采纳 | swap 默认关闭（decode +8.03%，`d7b0649`）；CPU 内核已近 DRAM 墙；join 尾部调度未做（收益有限） |
| A3 DeltaNet prefill | ① 判定完成（opt-in） | chunk mode 1：容差过、ttft −2.7%（68.72×2）、decode −4~7%（路由漂移）→ 不默认；②/③ 未开工 |

## B1 INT8 KV

- 读取优化 `6deaefc`：per-token scale 提出循环 + 8B 向量加载 → 14k 内核 **1.854→1.053 ms**（bf16 1.583，
  比 bf16 快 **33%**），1834 fast 23%，精度行逐字不变（NRMSE 0.0051/cos 0.999987，优于 FP8 的 0.0253/0.99968）。
- 生产使能 `8101788`+`c67edc7`：IPC capability/`--kv-type int8`/守卫放行/reservation 重算/QSA prefill int8 化；
  生产序列首次全绿。**端到端 +1.8% decode / −1.1% ttft → opt-in**（低于 +5% 线）。
- 容量未兑现：hotset profile 只排 2792 key（`resident=min(slots, ranking.size)`）；+157 槽上限 ≈ +0.7~0.9%。
- DPAS 复查：原 SG16 配置触发 IGC 错误，SG8 在同一工具链通过；已修复 D256 连续 decode 的
  QK 双遍残差路径及分块/尾块问题。正确性通过，首次内核测量慢于现有向量读取路径，默认关闭；
  见 [修复记录](qwen38-int8-dpas-repair-2026-10-12.md)。

## B2 NF4 全链路 W4A8

- decode（`5ef4e1a`）：内核 T=1 −15.5%，端到端 +1.61% → opt-in；精度 NRMSE ≤1.44%、全模型 cos 0.998/0.9995。
- prefill：实测 4token ALU 族**不是激活带宽受限**（20 MiB 流式 x vs 全 L1 只差 ~3%），砍半上限 ≤1.5%，
  端到端 ≤0.7% → **不实现**（`93562fd` + 报告）。

## B3 显存/内存卸载结构

- decode 解剖（活线）：`submitted 53.9 ms` = router_wait 13.4（in-order 队列深度，非路由器本身）
  + experts_host 19.7（cpu_compute 12.9 读 ~0.65 GB/token DRAM + join 9.1）；staging 全 0 → 冷专家全走 CPU，
  在 PCIe 9.6 GB/s 下是正确的。
- 弹性：常驻槽 **0.45~0.5%/100 槽**；chunk **每多一个 ≈4.4 s ttft**（4096→2048：70.6→83.9 s，+18.8%）。
- 已否证：pinned 替代大页（51 GiB 锁页 → 首请求 UR_OUT_OF_RESOURCES）；ring 2→4（旧）。
- **剩余最大杠杆（B3-3，未做）**：把 prefill 从 chunk-major 改成 layer-major（或等效的跨 chunk 冷专家窗口），
  H2D 310 GB→~62 GB，ttft 预计 **−13~−18%**；工程量：prefill 驱动、调度/取消路径、状态机重构（多日）。
  次选：动态重排 hotset（需 observe→export→重新生成 profile 的工具链）。

## B4 MTP 接入

- B4-③ 等价门未过：首分歧 = 被拒绝 verify 窗口 row0（pos=1492）——verify 单行路径与单行 decode 非逐位等价，
  stop 记账/draft/槽位/压缩计数全部排除（`e43ead5` + `40d6bf6` 探针）。
- 吞吐 MTP ~15.0–15.3 vs T1 15.6 → 即使等价也净负。**维持默认关闭**。
- 闭环选项：(a) env 化 live stage 探针定位 1491 行算子；(b) verify 窗口退化为逐行 count==1（放弃多行收益）。

## 2026-10-12 追加测量（B3 大窗口 / L 系列 / A3-1）

| 实验 | 14K ttft | 14K decode | 结论 |
|---|---:|---:|---|
| chunk 2048 / 4096 / 8192 | 83.89 / **70.57–70.64** / 89.22 | — | 非单调，4096 最优 |
| chunk 4096 + staging 64 | 71.16 | 15.50 | 无益（挤占常驻槽） |
| chunk 8192 + staging 64 | 89.51 | 13.86 | 最差 |
| GDN chunk mode 1 | **68.72 ×2** | 14.83/14.90（续轮 14.92/15.27） | ttft −2.7% 但 decode −4~7%（路由漂移），维持 opt-in |
| pinned host arena | 首请求失败 | — | UR_OUT_OF_RESOURCES，否证 |

### 2026-10-13 更正与新增（`548b0d8`）

- **撤回**："H4 profile 只排 2792、需重生成 ≥2949"——profile 实为全量 24576 key；`ranked_keys=` 打印的是预算
  截取后的前 N。**真正缺口**是 backend 专家缓存 admission 仍按 bf16 计 KV。已修（单源 `qwen4exp_kv_bytes`）：
  INT8 臂实测 `KV=384 MB + scales 6 MB`、**slots 2792→2948（+156）**。
- **撤回**："int8 两臂全程逐位一致"——首轮 hash 同、**续轮 hash 不同**（量化差异，需按容差/任务质量口径看）。
- **口径更正**：decode rate = `(completion_tokens-1)/(duration-first)`（旧解析用 32 当分子，绝对数偏高 ~3%）。
- P2 端到端 A/B/B/A：14K ttft INT8 −0.8%；续轮 rate +0.8~+1.7%（区间不重叠）；14K 32-token 落在方差内
  → **账目与 156 槽已兑现，端到端 ~1%**，int8 KV 维持 opt-in。

## 当前基线

- HEAD `b835180`；官方 pair `retained11`（worker `5fc5147f…` / server `9fa536ac…`）。
- 收口验证：`ctest -L 'cpu|abi'` **19/19 通过**（含本次补的 capability 精确列表守卫；期间一次
  `AilaInferenceCApiTests` 失败为抖动，单跑与复跑均通过，85.07 s）。
- 默认生产配置：bf16 KV、W4A8 关、MTP 关、chunk 4096、reserve 3584、槽 2792（profile 上限）。

## 2026-10-13 收口（P4 判定 + P1 修复）

- **P4 NO-GO**：share-4 SLM 解量化复用（`_slm`）逐位精确、零新 USM，但 gate/up 为 −11%（t16）/−18%（t64）；
  `_nounpack` oracle −54~−68%、`_nolut` 判别显示"可共享部分"上限仅 13.6%（t16）/33.1%（t64），完整链
  ≥10% 门不可达（消费者 tile 读次数与原 LUT gather 同量）。三个诊断内核保留为 opt-in 测试路径
  （生产零引用）。详见 [P4 报告](qwen38-p4-slmshare-2026-10-13.md)。
- **P1 修复**（`c1e9e60` + `2c51cc5`，均在 `src/utils/Tokenizer.cpp`）：① merges 只把 `#version` 当注释；
  ② `pretokenize` 对齐参考 `\s+(?!\S)`。离线逐 id 对齐 content 128/128、prompt1 13944/13944、
  prompt2 14096/14096；live g128 续轮恢复复用 **cached_prompt_tokens=14072、first content 71.0→1.486 s**
  （pair `retained16`）。新增 `AilaTokenizerRegressionTests`，`ctest -L 'cpu|abi'` 20/20。
  详见 [P1 定位](qwen38-p1-reencode-divergence-2026-10-13.md)。
- **pair**：retained16＝worker `7bad7286…` / server `916d4b56…`（build `2c51cc5`，live 验证用）；
  retained17＝同一 worker/server 源状态在测试/文档提交后的重建。
