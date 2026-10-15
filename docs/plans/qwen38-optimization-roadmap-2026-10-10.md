# Qwen3.8 优化路线图（2026-10-10）

当前基点：`HEAD=90b7931`（本轮链路 `e808f63`→`6dc4c3d`→`d7b0649`→`1f6c753`→`5ef4e1a`→`176aede`→`d701cf6`/`e4d4bc5`→`062b940`→`362bc98`→`b14cd64`→`90b7931`）；官方 pair `ff367797…/61be4ffb…`（`tmp/q4e-mtp-alignment-20261011/retained4/`）。历史 pair `395dd97b…/4bf51966…` 等保留在各 `tmp/q4e-*/retained*/` 归档。
统一度量口径：生产序列 `run.py <tag> --agent`（PI + 14K + 24-token 续轮 + 取消恢复），单变量 A/B/B/A，输出 hash 门；当日运行噪声底 ≈ ±0.6%（ttft）。内核类改动优先位精确（同二进制 env 开关 + off/on 双跑校验）。

## 0. 当前瓶颈基线（已归因）

- **14K prefill（首字 ~68 s）＝ GPU 计算主导**：稳态块 18.9–19.2 s ≈ router_wait 8.1 s + experts_host 10.7 s（compute 引擎 97–103% 持续饱和）。GPU 时间占比：MoE 专家 ALU 内核族 ~44% > H2D 34.6 s（与计算重叠）> DeltaNet 13.8 s > QSA 10 s > dense XMX 8.9 s。
- **decode（~72 ms/token，13.3–13.4 tok/s）＝ 主机侧主导**：experts_host 22–28 ms（含 CPU 专家 calculate 12–18 ms + join 7–13 ms，~50% 命中、200–290 任务/token）> router_wait 8–10 ms > 其它逐层小段 ~20 ms；swap 2.6–4 ms（已证净价值≈0）。

## A. 本轮在办方向（先做完）

### A1 专家 prefill 数据流终态（权重读取 ~1 遍/专家）
- 现状：4-token quad 结构使每专家权重被 `ceil(n_e/4)` 次重读（~25×）；WG-share 只把等效读降到 ~16 token/权重读（-2.94% ttft），share=8 因 L1 工作集上限失效；8-token 窗口因寄存器 spill 否决。
- 候选（按可行性排序，均要求寄存器中性、优先位精确）：
  1. workgroup 内多行×多 quad 的二维分工扩展（例如 8 行×2 quad 与 4×4 组合的按层自适应，按 staging bank 与 L1 容量选择 share）；
  2. SLM 权重暂存 + 多子组消费（规避 L1 政策不确定）；
  3. 新 tile 设计的 grouped GEMM（M=640×N=token 块，K 分块累加；注意 X3 的 M8 grouped-XMX 已被实测否决，设计需先论证 M/N 形状与 XMX 效率）。
- 门槛：块级 submitted 收益 ≥1%；位精确或按策略对照（数学不变才可声称算子加速）。

### A2 Decode CPU 专家路径（compute+join 20–30 ms/token）
- 路径 a：**swap 成本**——✅ 已完成并采纳（`d7b0649`）：长窗口+生产 A/B/B/A 证明同步 swap 净负，默认关闭后 14K decode rate **+8.03%**（13.732→14.835），输出逐位一致；同时修复 `read_int` 使 `FREQ=0` 不可表达的问题（改用 `read_int_raw`）。见 [qwen38-swap-longwindow-2026-10-10.md](qwen38-swap-longwindow-2026-10-10.md)。
- 路径 b：**CPU 内核本身**——审计结论（2026-10-10）：热循环为 `nf4_q8_dot_activation64`（~24 向量 uop/32B 权重）；每 token 权重 ~0.65 GB ÷ compute 12–18 ms ≈ 35–55 GB/s，**已近桌面 DRAM 带宽墙**，纯指令缩减收益有限。Strata 的"一次 scale 修正"恒等式**不可移植**（NF4 非仿射、无 offset 项；Aila 已在结构上实现等价形式）。位精确开关组合探针（`BYTE_SHUFFLE=1 + SCALE_REGISTER=1 + SCALE_CACHE_MB=16`）生产两轮：rate 15.301 vs 基线 15.210（+0.6%）、续轮 -0.6% → **中性，拒绝**。剩余可选项仅剩调度类（join 尾部按任务完成即回传，位精确但收益有限）与 CPU 侧预取探索（无直接证据，暂缓）。
- 路径 c：**swap 关闭后的长生成监控**——命中率随路由漂移自然衰减；若出现 ≥1024 token 生成的应用证据，重评"H4 离线重排 + 低频 swap"。

### A3 DeltaNet prefill（13.8 s，延迟受限、并行锁死 48 WG）
- MLP/预取路线已实测否定（K1，1.27–1.95× 回退）。剩余（均需先立数值/收益证据）：
  1. chunkwise（`GdnChunkOps` T=16，已实现，X4 实测 -28~31% wall，改浮点顺序 → 需 KL/NLL/logits 资格化）；
  2. dv 切分（2–4 WG/head + epilogue 归一化，逐位性待论证，总硬件线程数不变）；
  3. 多 session 网格维（逐位，单会话不可测，随服务并发启用）。
- **2026-10-12 A3-① 判定（有界，维持 opt-in）**：`AILA_Q4E_GDN_CHUNK_MODE=1`（T=16 GPU FP32 chunk）
  容差契约通过（logits NRMSE 7.79%/4.30%、cos 0.996978/0.999091、top-1 全中）；生产两轮 **14K ttft
  68.719/68.720 s vs 基线 70.636/70.574（−2.7%，完全复现）**；但 **decode 一致回退**：14K decode
  14.825/14.895 vs 15.583/15.360，续轮 14.921/15.272 vs 16.627/16.012（区间不重叠，−4~7%）。
  归因：chunk 路径只用于 prefill（`Qwen4ExpPrefill.cpp:111`），decode 回退来自**数值变化 → 路由漂移 → 解码期
  专家命中率下降**。结论：prompt 重/短答场景净赚（省 1.9 s ttft，32 token 解码只多 ~0.1 s），长生成净亏；
  **维持 opt-in，不做默认**；若未来要默认，需先按工作负载混合比与路由稳定性再评估。
  （A3-② dv 切分、③ 多 session 网格维：仍未开工的研究项，门槛分别为逐位性论证/并发服务验证。）
- **A3-②/③ 收口判定（2026-10-12，基于已测弹性，不新开工程）**：
  - ② **dv 切分**：数学上逐元素独立（`k_iᵀS` 的 k 维归约不变、norm/gate 是逐 v 维 epilogue），可做到**逐位**；
    但收益上界由实测弹性给出：GDN op ≈13.8 s / 135 s GPU，X4 chunk（GDN op −28~31%）只换来 14K ttft **−2.7%**
    → 弹性 ≈0.09（ttft/GDN-op）。dv 切分乐观给 GDN op −20% ⇒ **ttft ≲2%**；且它只影响 prefill，
    与"adopted 线"相比属边际收益，**不排期**（若将来 GDN 占比上升再评估）。
  - ③ **多 session 网格维**：生产场景为 `--sessions 1`（单会话 agent 服务），且 decode 瓶颈是专家供给
    （cpu_compute 12.9 ms 读 0.65 GB/token），线性注意力段在 decode 中占比很小 → **无证据支持，不排期**。

## B. 用户指定方向（A 完成后按序）

### B1 INT8 KV cache（真正的 INT8，不是沿用 FP8）
- **已实现原型并完成三臂微测（`1f6c753`，opt-in 默认关闭）**：`AILA_KV_QUANT=1 AILA_KV_QUANT_FORMAT=int8`；s8 存储 + per-token f32 scale（per-head 标量在增量 decode 下会失效）；K 在 mRoPE 后量化；覆盖 copy_to_cache/attention_decode/attention_prefill_cached/稠密 partial prepare，其余路径显式拒绝。
- **结果**：同字节下 **INT8 精度约为 FP8 的 5×**（NRMSE 0.0051 vs 0.0253，cos 0.99999 vs 0.99968）；但"读时逐元素反量化"使 **int8 比 bf16 慢 17%**（14k：1.84 vs 1.57 ms），FP8 更慢（2.1×）→ 速度形态**不达标，维持 opt-in、不安排生产 LIVE**。
- 同时修复两处既有 FP8 缺陷（128 快路径半维漏写、JM 量化守卫）。**下一步**：无解量 int8（DPAS/W8A8）读内核，设计已定稿于 [tmp/q4e-int8kv-dpas-20261012/DESIGN.md](../../tmp/q4e-int8kv-dpas-20261012/DESIGN.md)（Stage A = QKᵀ s8 DPAS + per-token scale epilogue；Stage B = P/V 双 DPAS，s_v[j] 折进 P；先探 `AILA_PRINT_MATRIX_COMBOS` s8 组合），达标（不慢于 bf16 且 NRMSE≤FP8）后再生产三臂。
- **2026-10-12 结果（`6deaefc`，达标）**：读取路径向量化（K 的 per-token scale 提出 d 循环 + 8 字节向量加载；V 改 token-major 每 token 一次 scale）→ 同二进制 A/B：14k int8 **1.854→1.053 ms**（bf16 1.583），1834 **0.529→0.347**（bf16 0.449）→ **int8 首次快于 bf16**，精度行逐字不变（NRMSE 0.00510548/cos 0.999987，FP8 同长度 0.0253/0.999682）。报告 [qwen38-int8kv-vec-2026-10-12.md](qwen38-int8kv-vec-2026-10-12.md)。
- **生产启用**仍被共享会话守卫挡住（`src/inference/Model.cpp:702` "Qwen4Exp shared sessions require BF16 KV" + `src/worker/InferenceWorker.cpp:546` capability 检查；生产 A/B 4s 即退出，见 `tmp/q4e-resident-speed-20261009/int8kv-b1-aborted/`）→ 解禁 + kvType 管线 + 覆盖缺口（rows/pair/QSA 稀疏 prefill）并入 **B3**。
- **DPAS 路径工具链否证**：A770 设备组合确认 `n=8 k=32 s8×s8→s32`，但 DPC++ 2026.1/本机 IGC 对**任何 int32 累加器 joint_matrix** 编译期崩溃（`IGC ICE: Divide by zero`，30 行最小复现，已排除 M=1/layout/SLM store 等因素）→ 暂不可实现，待工具链升级后按 DESIGN.md 重试。

### B2 NF4 全链路 W4A8（T=1、MTP 关闭）
- **已实现并测量（`5ef4e1a`，opt-in 默认关闭）**：env `AILA_Q4E_GPU_Q8_ADAPTIVE`；decode 路径 int8 激活变体（x/h 各带 per-64 scale），非生产几何自动回退 bf16。
- 结果：内核级 T=1 **慢 15.5%**（NF4 权重已 0.5B/权重、decode 专家路径为带宽/延迟主导）；端到端生产 A/B/B/A **+1.61%**（14K decode rate）/ +0.28%（续轮），**两请求输出逐位一致**，全模型容差 PASS（NRMSE 5.89%/3.31%、cos 0.9983/0.9995）。**低于 +5% 采纳线 → 保持 opt-in**。
- 覆盖缺口：仅 decode；prefill 专家内核与稠密路径未覆盖（prefill 侧 x 走 LSC 的主导流量若 int8 化理论上可减半，未实施）。报告 [qwen38-w4a8-screen-2026-10-10.md](qwen38-w4a8-screen-2026-10-10.md)。
- **2026-10-12 全链路收口（负结果，有界）**：prefill 4-token ALU 族实测**不是激活带宽受限**——生产尺度
  （4096 行=20 MiB 流式 x）vs 全 L1 命中只差 ~3%，即砍半 x 最多 ≤1.5%；ALU 族占 prefill 44% → **端到端上界
  ≤0.7%**，低于 +5% 线 → 不实现 prefill q8 变体。T=4 decode 同形状 q8/bf16 持平也印证点积惯用式无 ALU 数量级收益。
  报告 [qwen38-w4a8-prefill-bound-2026-10-12.md](qwen38-w4a8-prefill-bound-2026-10-12.md)；诊断开关 `--prefill-bench-only`
  + `AILA_Q4E_BENCH_XLOCALITY` 已入库备复用。

### B3 显存/内存卸载结构重整（2026-10-10 审计校正版）
- **2026-10-12 活线诊断（`b3-diag-a1`，`AILA_Q4E_WORK_SUMMARY=1`，decode 8 行中位数）**：
  `submitted_ms=53.9`，其中 `router_wait=13.4`、`experts_host=19.7`（`cpu_compute=12.9` + `cpu_join=9.1`）；
  **staging `fill/bank_wait/staged/h2d` 全为 0** → 冷专家全部走 CPU 路径（每 token ~236 misses × 2.42 MiB
  ≈ 0.65 GB DRAM 读，~50 GB/s 级）；PCIe 实测 9.6 GB/s，把冷专家搬到 GPU staging 反而更差（68 ms）。
  **结论：B3 的主杠杆是"命中率"，不是 staging。**
- **下一步实现清单（int8 KV → 更多常驻槽）**：① `InferenceProtocol.cpp:630` capability 目前 `kvType != "bf16" → {}`，
  需允许 int8；② `Model.cpp:702` 共享会话守卫按显式 opt-in 放开；③ `Qwen4ExpBnb4Backend.cpp:724`
  槽上限 `min(2800, bytes/slot-32)` 的**硬 2800** 需改为按字节推导（KV 省 384 MiB → ~2949，+5.3%）；
  ④ 覆盖核对：`attention_decode`/`attention_prefill_cached` 已支持 int8，`attention_decode_rows`/pair/QSA 稀疏
  prefill 仍显式拒绝（生产 B=1 不触发，但需守卫/补齐）。预期把 ~30% 的 CPU 冷读搬到 VRAM（每 token 数 ms）。
- **2026-10-12 B3-1 已落地（`8101788`）**：①②③ 全部实现（capability `qwen4exp_nf4_int8_kv`、`--kv-type int8`/
  `AILA_SERVER_KV_TYPE`、`allowQuantizedKv` 显式放行、reservation 按 int8 布局重算含 scales、`AILA_Q4E_MAX_GPU_SLOTS`
  默认 2800 可抬）；默认 bf16 生产冒烟（14K+PI+取消恢复）通过。**B3-2 剩余阻塞**：
  `attention_prefill_qsa_batched requires bf16 KV cache`（`AttentionOps.cpp:4516`）——生产 prefill 主路径仍需
  int8 化（连同 `attention_decode_rows`/pair/QSA 稀疏路径），完成前 int8 生产 A/B 无法跑完。
- **2026-10-12 B3-2 完成 + 生产 A/B/B/A（`c67edc7`）**：QSA prefill 已 int8 化（内核测试 int8 臂 max_diff 3.05e-05，
  bf16 臂不变），int8 生产序列首次全绿。结果：14K decode **+1.8%**（15.75×2 vs 15.36/15.58）、ttft **−1.1%**、
  续轮噪声内 → **低于 +5% 线，int8 KV 维持 opt-in**。容量收益未兑现：hotset profile 只排 2792 key
  （`resident = min(slots, ranking.size)`），384 MiB 没变成常驻槽；诊断 hits/misses 与 bf16 相同（staging=0）。
  报告 [qwen38-int8kv-prod-2026-10-12.md](qwen38-int8kv-prod-2026-10-12.md)；若续做：重生成 ≥2949 键 profile 再 A/B。
- **2026-10-12 B3 弹性定量（两个反向实验，`8101788` 的 env 开关使能）**：
  - **常驻槽弹性**：`AILA_Q4E_MAX_GPU_SLOTS=2000`（−28% 槽）→ 14K decode **−3.6%**（14.916 vs 15.47 中位）、
    续轮 **−4.5%** → 约 **0.45–0.5%/100 槽**。2580→2949 的 +157 槽上限 ⇒ 仅 **+0.7~0.9%**（且需重生成 profile）。
  - **chunk/专家窗口弹性**：`--chunk 2048`（refill 次数 ×1.75）→ 14K ttft **70.60→83.89 s（+18.8%）**，
    每多一个 chunk ≈ **4.4 s**；反推 8192（chunks 4→2）理论上 **−9~−13 s（−13~−18%）**。
    诊断佐证：prefill 10 行里 `fill_ms=47.7 s`、`bank_wait_ms=17.7 s`、H2D 310 GB——每 chunk 都要重传
    ~454/512 个冷专家（2.42 MB/个 ×48 层）。
  - **结论**：B3 剩余最大杠杆 = 让冷专家窗口跨越多个 forward chunk（解耦），
    但 8192 行 forward 需 +4.6 GiB reserve（16 GB 不可行）；可行形态是把 ~1.1 GB 从"常驻槽"挪成
    "单层冷专家暂存"（交换率：−455 槽 ≈ decode −2%，换 ttft −13~−18%）——留作 B3-3 设计题，需先出预算方案。
- **候选③ pinned vs 大页（2026-10-12 否证）**：`AILA_Q4E_HOST_ARENA_PINNED=1`（新 env 门控，默认大页）→
  worker 能在 ~51 GB 锁页下加载，但**首次请求**即 `level_zero backend failed with error: 40
  (UR_RESULT_ERROR_OUT_OF_RESOURCES)`，随后 shutdown 宽限超时 → 生产 A/B 无法进行。结论：51 GB 级 host
  complement 不能用 pinned 替代原生大页（大页既是启动风险来源也是当前唯一可行解）；小模型/小 arena 场景仍可用该 env 复测 TLB 影响。
- **B3-3 大窗口/深 staging（2026-10-12 全部否证，结论性）**：
  - 协议/CLI 上限抬到 8192（`kExtendedMaxPrefillChunkSize`）后实测 14K ttft：chunk2048 **83.89 s**、
    4096 **70.57/70.64 s（最优）**、8192 **89.22 s**——**非单调**。
  - 8192 的诊断账目证明 H2D 收益真实存在（310→179 GB、fill 47.7→27.4 s），但被
    `bank_wait 17.7→34.0 s`、router_wait +9.7 s 与逐层 host gap（ple_host 0.1→8.0 s 的归因区间）吃掉。
  - 加深 staging（`AILA_Q4E_STAGING_SLOTS` 1..64，默认 16）无效：chunk4096 + stg64 **71.16 s（−0.7%）**，
    chunk8192 + stg64 **89.51 s**（且 decode 13.86，最差）。深 staging 还会按 2.42 MB/slot 挤掉常驻槽（64 槽 ≈ −96 槽）。
  - **结论**：4096 chunk + 16 槽 staging 是当前内核集下的局部最优；"layer-major/大窗口"重构必须先修
    bank_wait 与逐层 host 串行，否则 H2D 的 131 GB 节省到不了墙钟。B3 大项到此证据关闭，除非先修这两处。
- 生效值澄清（2026-10-12 复核）：`AILA_Q4E_VRAM_RESERVE_MB` **生产与所有 MTP/对齐 runner 都是 3584**
  （`tools/server/start-qwen38.ps1:22` 与 `run_mtp_ab*.py` 在加载 profile 后显式覆盖）；profile JSON 的 3008
  与代码默认 2560 只在"env 完全缺失"时才有意义。做预算审计时以 3584 为基准，若要动 JSON 值须同步改覆盖点。
- 目标：服务长上下文供给与结构性缺口；预算与约束（审计实测）：reserve launcher 默认 3584（代码 2560/JSON 3008），其中 logits 970 MiB + phase 403 + PLE 540 + HC 225 + staging 77 + misc 60 ≈ 2275 MiB，余量 ~1.3 GiB；KV 已按 max_seq 全额预付（不占 reserve）；8192 chunk 需 +4.6 GiB（须先做"专家窗口 vs forward 行数"解耦）；KV int8 可省 384 MiB（上限 2800→2949 槽可行，+5.6%，但需解禁共享会话守卫与 DPAS 读内核）。
- 候选（按新证据排序）：① adaptive 路径**跨层路由预取缺失**（`predict_next` 恒 false，`Qwen4ExpBnb4Backend.cpp:2027`）——注意 10-10 router 审计已证 legacy 预测输入 `buf_.x` 与下一层真实 router 输入不等价，若做须把它当**启发式预取提示**（不允许影响正确性），并先给证据；② 窗口/大块解耦（专家窗口 8192、forward 仍 2048）H2D/fill 预计 -43~46%；③ pinned vs native large-page 补集 A/B（降低启动失败风险，CPU TLB 代价待测）；④ ring 2→4 已否决（负结果，勿重做）；⑤ 预解量化 bf16 bank 高风险低可行（字节 ×3.2、常驻 -68%），除非只覆盖热集。
- 口径修正：VTune 全轮 H2D 332.5 GB/34.6 s ≈ 9.6 GB/s 聚合；"4.6 GB/s"是单批 fence 限速，两者勿混。
- 门槛：A/B/B/A + 预算审计；供给类改动要求输出逐位一致。

### B4 MTP 接入（2026-10-10 审计校正版）
- 关键校正：MTP **结构上要求 dense ≤2048**（`Qwen4ExpMtpRuntime.cpp:12`，32K 长上下文不可用）——其经济性由"≤2048 下的 GPU 常驻 + 多行 CPU verifier"决定，**与 B3 的长上下文卸载结构无关**。
- 已就绪：draft 权重常驻 + 固定 64 槽（155 MiB）、session 级 KV（4.87 MiB/会话）、target `{2,vocab}` logits、GPU 验证路径（`ForwardImpl(count=2, decode_verification=true)` + attention pair bundle）、窗口零设备分配。**缺口**：① 多行 CPU verifier 在 adaptive 路径不可达（`Qwen4ExpExpertExecutor.cpp:825` 的 `cache_` 仍绑退役 cache；verify 分支只喂给死路径 `cpu_verify_async`）——需把 cold/verify 绑到 adaptive cache/host_arena（**这是任何 CPU-verifier 路线与 MTP 的共同前置**）；② T1 对齐：多行核（GDN 8-vs-4 累加、RoPE native、QKV prepare、dense-QSA）与 T1 存在数值分歧（`docs/plans/qwen38-mtp-alignment-draft-2026-10-06.md`：末态 123,948 个 FP32 不同、max 1.34e-7）；`VerifyGdnOrder/VerifyDecodePrepare/VerifyDecodeAttention` 开关已存在（默认关）。
- 实施顺序：① ✅ 多行 verifier 重接 adaptive（`176aede`）→ ② ✅ 全 48 层逐位对齐（`d701cf6`+`e4d4bc5`）→ ③ 首轮验收未过（`062b940`）→ ④ ✅ 活线分歧定位+修复（`362bc98`）：首个分歧 = attention-pair 核（长 KV 下 partial-merge ≠ 两次单行 decode），`VERIFY_SMALL_ROWS` 下回避后**首请求与 T1 逐字节相同**（sha f72a0da0fb2f）→ ⑤ ✅ 状态对比与触发条件收敛（`b14cd64`）：verify 窗口结束态与 4×decode **逐字节相同（differing=0）**；分歧触发 = **stop-token 结束路径**（length 结束两请求全同；stop 结束仅续轮分歧，两臂 emitted/ctx 相同）→ ⑥ **进行中**：扩展快照（adaptive 驻留/路由计数 + draft mtpState）二分 stop 路径的会话边界/记账链；修好后复跑 B4-③ 等价门与吞吐（当前 MTP ~15.0–15.3 vs T1 15.6 tok/s）。构建完整性修复 `90b7931`（记账探针改 `read_int_raw`，避免轻量目标链接主库符号）。
- ⑥ ✅ 完成（`40d6bf6`，2026-10-12）：`AILA_Q4E_MTP_BOUNDARY_DUMP` 指纹探针把首分歧钉到 **pos=1492 = 被拒绝 MTP 窗口 row0（2 行 verify forward）**——请求 1 文本逐字节相同但结束态已逐位不同（B4-⑤ 的 stop 假设否证）；stop/bonus/Finalizing、draft 复用、adaptive 槽、压缩计数、GDN_HISTORY_FUSED/DECODE_PENDING_MLP 全部排除。残留唯一解释 = verify 单行路径与单行 decode 不逐位等价（状态相关稀有行）。**B4-③ 等价门未过 → MTP 维持默认关闭/opt-in**（吞吐 ~15.0–15.3 vs T1 15.6 tok/s，即使等价也净负）。闭环选项：(a) live stage 探针定位 1491 行算子；(b) verify 窗口退化为逐行 count==1（构造等价但放弃多行收益）——按用户执行序，等 B3 之后再重评。pair `retained5`（worker `436719a8…`/server `82336e2f…`）。
- **B4-③ 验收协议（固化自 `qwen38-server-mtp-2026-10-07.md` 实测流程）**：启动参数 `--context 2048 --sessions 2 --decode-batch-size 1`，先加载 profile JSON 的 worker_environment，再置 `AILA_Q4E_SERVER_MTP=1`（启动时读一次）+ 本轮对齐开关与 `AILA_Q4E_ARENA_VERIFY_ROWS=1`；门禁 = ① 交错 MTP/T1 输出逐一一致（历史 7/8、6/8 未过，本轮必须全过）② CPU RequestState/Scheduler 测试 ③ 128-token 请求无流错误、正常回收、C=2048/2 会话内存账目一致 ④ 报告首字/总时延与首字后 tok/s（口径 127/(done−first content)，注明含 IPC 与末 token）。达标后再谈与 B2/B1 的叠加与默认化。
- 验收：接受率、decode tok/s（按本机重测，不用 Strata 数字外推）、容差契约 + 采样一致性。

## 执行顺序与门禁

### 2026-10-12 用户确认的执行序（A 收尾后）

1. **B1 INT8 KV（先做）**：目标从"原型三臂"升级为**可上线形态评估**——补齐无解量 int8 读内核（s8 KV × s8 Q，DPAS s8×s8→s32，W8A8 attention），门槛 = 不慢于 bf16 且 NRMSE ≤ FP8；未达标则保留 opt-in + 记录负结论，KV 省内存（-384 MiB）并入 B3 预算。
2. **B2 NF4 全链路 W4A8（T=1、MTP 关闭）**：decode 侧已有 `AILA_Q4E_GPU_Q8_ADAPTIVE`（+1.61%，低于采纳线）；本轮补齐 **prefill 专家内核（wgshare/quad）与稠密路径**，量化端到端性能与精度损失（NRMSE/cos + 逐位/容差），采纳线 ≥+5%。
3. **B3 显存/内存卸载结构重整**：窗口/大块解耦、pinned vs 大页 A/B、路由预取启发式（仅提示、不影响正确性）、KV int8 预算；每步预算审计 + 输出位精确优先。
4. **B4 MTP 接入（B3 之后）**：B4-⑥ 已收口（`40d6bf6`）——等价门未过（首分歧 = 被拒绝 verify 窗口 row0，pos=1492），MTP 维持默认关闭；闭环需 (a) live stage 探针或 (b) verify 逐行 decode（放弃多行收益），等 B3 之后按吞吐证据重评。

口径与纪律（2026-10-12 追加）：
- **运行测试的命令里禁止固定长 sleep**（曾出现 `Start-Sleep 240/420` 拖慢进度）：一律 1–2s 轮询完成条件（`result.json`/日志 mtime/`Get-Process`），长测试前台运行并给足 yield_ms。
- 已获管理员权限，可做更细粒度 VTune 采样；GPU 运行前互斥检查 `AilaWorker/AilaServer`，不误删 `tmp/*-aborted` 等未决证据。

| 序 | 项 | 类型 | 前置 | 关键判据 |
|---|---|---|---|---|
| 1 | A1 专家数据流终态 | 内核重构 | 无 | 块级 ≥1%，寄存器中性、位精确优先 |
| 2 | A2 CPU 专家路径 | 内核/策略 | 路径 a 已完成（+8.03%） | 路径 b：compute -15~25% |
| 3 | A3 DeltaNet 备选 | 内核/数值 | 数值资格（chunkwise）或逐位论证（dv 切分） | wall -15% 且过容差/逐位 |
| 4 | B1 INT8 KV | 新功能 | 无 | 精度优于 FP8 或等损更快；单变量三臂 |
| 5 | B2 W4A8 | 内核/数值 | 无（探针先行） | decode ≥+5% 且容差内 |
| 6 | B3 卸载结构 | 架构 | A1/B1/B2 结论 | 每 token 冷字节/等待下降，预算审计过 |
| 7 | B4 MTP | 架构 | B3（+B2 若成立） | 接受率与吞吐、容差契约 |

每项完成后按既有惯例产出：`docs/plans/qwen38-*-screen-*.md`（或 `-results-`）+ 原始证据归档 + 官方 pair 身份重建（若采纳）。
