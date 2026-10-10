# Qwen3.8 专家 prefill：workgroup 内权重行共享（WG-share）

日期：2026-10-10。基点 `HEAD=4bbf849` + 候选补丁（`tmp/q4e-wgshare-20261010/candidate.patch`，3 文件 +953/−2）。
候选二进制（测量版）：Worker SHA-256 `bdc203ebf264a5bdbe6adcca6ae9c52a4f06f135a05f5c055ff7543f52b4e07c`、Server `7b84457e8c476abe12df4b1f7f7d1f45a816c640fc5d4cf1695abfb6c581f01a`（同构建戳 `aila-0.2.0-abi1-4bbf849…`）。

## 假设与设计

[HYPOTHESIS] 专家 prefill ALU 路径（VTune 占全轮 GPU 时间 ~44%）按 4-token quad 组织，同一 workgroup 内 16 个子组处理同一 quad 的 16 个权重行；每个专家的权重被 `ceil(n_e/4)` 次重读（~25×）。把 workgroup 的 16 个子组改为 `(16/ShareQuads)` 行 × `ShareQuads` 个 quad 的映射后，多个子组在同一时间窗内流式读取相同的权重行（相同 cache line），可用 L1 时间共享摊薄 DRAM/L2 缺失，且**不增加任何每线程累加器**（与已否决的 8-token 方案的关键区别）。

实现：`launch_moe_expert_major_prefill_gate_up/down_4token_wgshare<..., ShareQuads>`；除映射 4 行外逐字复制原 4token 内核（同 stride/字节步进/scale 广播/归约顺序/写出公式）。env `AILA_Q4E_4TOKEN_WGSHARE`（默认 0=off；2/4/8 选择 share 系数）只在 quads 段改道。

位精确验证：`AilaQwen4ExpKernelTests` 在 env 未设/=2/=4/=8 四遍全部 `[PASS]`（每次 191 项，0 失败），含新边界用例（Q=5、I=13、H=7 对 2/4/8 均非整除）与哨兵填充比对，wgshare 输出与 4token 直连输出**逐位一致**（0 mismatches），并经原有链路与串行参考一致。

## 诊断扫描（同候选二进制，唯一切换 env；--diagnostic；稳态块 submitted_ms）

| 臂 | block1 | block2 | block3 | 对照 off |
|---|---:|---:|---:|---:|
| off（wgs-a1） | 19.665 s | 19.879 s | 10.939 s | — |
| share=2（wgs-b3） | 18.915 s | 19.135 s | 10.553 s | -3.8% / -3.7% / -3.5% |
| share=4（wgs-b1） | 18.918 s | 19.163 s | 10.538 s | -3.8% / -3.6% / -3.7% |
| share=8（wgs-b2） | 19.598 s | 19.918 s | 10.826 s | -0.3% / +0.2% / -1.0% |

share≥2 后收益饱和（share4≈share2），share=8 因 x/权重工作集超出 L1 有效驻留而失效——与"缓存时间共享"机制一致。选定 **share=4** 进入生产筛选。
（内核测试内的小尺寸 benchmark 不具代表性：share2 +2.3% / share4 +1.5% / share8 −1.1%，其工作集远小于生产 staging bank。）

## 生产 A/B/B/A（同二进制单一变量；--agent 全序列）

顺序 off → share4 → share4 → off。全部 `passed=true`、PI 与取消恢复通过；四轮请求 payload/输入/输出 hash **完全一致**（14K 与续轮，均逐位相同——本改动数学不变）。

| 指标（两轮中位数） | off | share4 | 变化 |
|---|---:|---:|---:|
| 14K 首字(s) | 70.115 | 68.054 | **-2.94%** |
| 14K 总耗时(s) | 72.440 | 70.376 | **-2.85%** |
| 14K decode rate(tok/s) | 13.334 | 13.354 | +0.15% |
| 24-token 续轮 首字(s) | 1.120 | 1.127 | +0.6% |
| 24-token 续轮 总耗时(s) | 3.329 | 3.345 | +0.5% |
| 续轮 rate(tok/s) | 14.036 | 13.971 | -0.5% |

四个交叉对比（B1/B2 对 A1/A2）全部为负收益方向（B 更快）：-1.05 s…-2.07 s。A 臂内差 ~1.0 s 属当日环境波动，但两臂区间不重叠。最低可用 RAM 2.08–2.61 GiB。

## 结论与采纳

- 稳态块 -3.6~3.8%（诊断口径）；生产 14K 首字 **-2.94%**、总耗时 **-2.85%**，输出逐位不变、decode/续轮无回归。
- 采纳：源码默认关闭；`tools/server/start-qwen38.ps1` 默认置 `AILA_Q4E_4TOKEN_WGSHARE=4`（保留调用者显式覆盖，含 '0'）。
- 机理说明：收益来自同一 workgroup 内多个子组对相同权重行的 L1/L2 时间共享；share=8 的失效给出 L1 有效容量上界。这只是第一条结构改造（消掉约一半可摊薄的重读），不是"每权重一遍"的终态。

## 验收与证据

- 内核测试（最终构建，env off / =4）：`tmp/q4e-wgshare-20261010/final-kernel-tests-{off,share4}.log`，均 exit 0、全部 PASS。
- CPU/ABI：`ctest -L 'cpu|abi'` 19 项 100% 通过（1 项条件跳过）；Server：4/4 通过（`ctest-{cpu-abi,server}.log`）。
- 生产原始证据：`tmp/q4e-resident-speed-20261009/wgs-p-{a1,b1,b2,a2}/` 与同名 run.log；诊断扫描：`wgs-{a1,b1,b2,b3}-*`；分析：`swap_analyze.py` 输出（同二进制、env diff 仅 `AILA_Q4E_4TOKEN_WGSHARE`）。
- 候选补丁与构建/测试日志：`tmp/q4e-wgshare-20261010/`（`candidate.patch`、`kernel-tests-env-*.log`、`build*.log`）。
