# Qwen3.8 R1: 乱序队列 / submit 开销解剖（2026-10-16）

范围：**只读源码调研 + 既有日志取证**。没有跑 GPU（GPU 被其它工作占用），没有构建，没有
改 `src/`，没有 commit；本文件是唯一写入。

树状态（编写中变动过一次，见下）：本研究的证据扫描开始于 `HEAD 6498159` + 3 个未提交文件
（`HyperConnectionOps.{hpp,cpp}` 的 `HcSubmitTiming` 插桩 + `Qwen4ExpBnb4Backend.cpp` 的计数字段）；
扫描结束前另一条 workstream 把它们提交了（`ffb2b08`、`6f9a41e`），并**新增了与本任务直接重叠的
OOO 探针**（`8f4950e`，见下）。本文所有行号已按 `HEAD 8f4950e`（= 当前磁盘）复核过：
`src/ops/HyperConnectionOps.cpp` 的 4 个 decode submit 仍在 `:324/:397/:467/:549`（+`:612` streams 变体），
`src/models/Qwen4ExpBnb4Backend.cpp` 因 XMX 采纳多了 6 行，`ensure_prediction_ready` 现为 `:2943-2946`。

> **并发变更提示**：`8f4950e docs(q4e): OOO-vs-in-order probe (OOO is 7% slower, +27% submit cost;
> waits are the lever) + A770 graph experiment roadmap` 在本次调研期间落地，给出了
> **OOO vs in-order 的实测**（`docs/plans/qwen38-r1-ooo-queue-probe-2026-10-16.md`、
> `tmp/q4e-structural-20261016/ooo-probe-L20.log`）。本文 §0.1/§1.7/§2/§3/§4 已按该实测重写；
> 其余章节（提交链清单、`depends_on` 密度、profiling 开销、`Context::queue()` 成本）不受影响。

证据标注：`[源码]`=本次直接读到；`[日志]`=本次读的既有日志/探针输出；`[语料]`=引既有文档
（B16=`docs/plans/qwen38-r1-bottleneck-audit-2026-10-16.md`、A16=`docs/plans/qwen38-r1-decode-submit-path-audit-2026-10-16.md`、
HC16=`docs/plans/qwen38-r1-hc-submit-attribution-2026-10-16.md`）；`[推断]`=本文推导、未实测。

术语：**提交（submission）** = 一条 queue-level 命令（`queue::submit` / `memcpy` / `memset` /
`single_task` / `ext_oneapi_submit_barrier`）。**块计时** = 相邻 `stage()` 之间的 host wall time。

---

## 0. 结论速览

1. **生产里没有任何 out-of-order 队列，而且刚被实测证明"改用 OOO 是负收益"。** compute queue
   永远是 in-order（`Context.hpp:29-40`），profiling 是可选的第二个属性；其余 5 个 queue 实例
   全是 in-order 或 compute queue 的拷贝（§1.1）。`8f4950e` 的 OOO 探针（同形状合成链、
   生产 L0 开关）实测：显式 `depends_on` 的 OOO 链 host submit **6.45 → 8.17 µs/kernel（+27%）**、
   wall **13.92 → 14.93 ms/token（+7%）**；两条真正独立的链在 OOO 下 overlap 0.42 vs in-order 0.43
   （无额外重叠）；"放行/扣留"行为两种队列完全一致（337→278 vs 325→278 µs）。
   **结论：OOO 不是杠杆；每层 wait 节奏才是**（同探针去掉 per-layer wait：14.43 → 8.61 ms，−40%）。
   该结论与本文 §3 的静态 blast-radius 分析一致。
2. **提交密度与依赖密度严重不对称**：`src/` 共 **203 处 `.submit(`**（另有 24 处 `.memset(`、
   139 处 `.memcpy(`），生成约 987 submissions/token（A16 §1d：GDN 层 19 / QSA 层 25），
   而全仓只有 **33 处 `depends_on`**（≈3%），且集中在 prefill 的 MoE/XMX 路径；decode 的依赖
   100% 靠隐式 in-order，源码里有 ~30 处注释明写 "the in-order queue …"（§1.2）。
   这是 OOO 改造 blast radius 的直接度量。
3. `src/ops/Bnb4BitLinear.cpp:2709` 的 `in_order` 检查是 **Q8 LM-head 路径的前置条件**：
   该路径每行提交 2 个 kernel（量化激活 → 受量化 GEMV），量化返回的 event 被丢弃，
   两 kernel 靠 in-order 排序，且复用同一份 `xq/xs` scratch。违反时
   `bf16_q8_head_supported()` 返回 false → 后端**静默回退**到 bf16+oneDNN head
   （`Qwen4ExpBnb4Backend.cpp:573` 打一行 `requested=1 actual_enabled=0`），
   直接调用则抛 `std::invalid_argument`（`Bnb4BitLinear.cpp:2744-2751`）。不会静默算错，但会
   悄悄关掉一个性能路径。
4. **`Context::queue()` 不是 30 µs 的来源**：`async_errors_->Rethrow()` 是 1 次 acquire load
   （`AsyncErrorState.hpp:31-33`），`CheckTrace()` 在 worker 上是 1 次空指针判断
   （`Context.hpp:224`，`AILA_EXECUTION_TRACE_DIR` 未设时 `trace_==nullptr`）。
   decode 每层 30-60 次调用 → 全部加起来 <0.1 ms/token。
5. **一层 decode 的真实提交链**（e10 bundle，`count==1`）＝ 16 个共享提交
   （14 kernel ＋ **2 个 compute-queue memcpy**：5 KB D2H 与 64 B H2D）＋ 3/9 个注意力 kernel；
   kernel 总 19（GDN）/25（QSA）。这 2 个 memcpy **不走 `copy_q_`**，与 kernel 同在
   in-order compute queue 上；另有 **≥2 次 host SYCL wait**（`resident_input_event_` 与
   `actual_route_event`）。decode 热路径没有 USM 分配、没有 oneDNN descriptor 绑定（§1.5）。
6. **"30 µs vs 7 µs"的口径需要修正**：按层归一化后，生产 host-submit 块 ≈ 18.7 ms/48 =
   390 µs/层 ÷ 20.5 submits ≈ **18.5 µs/submit**，探针同类循环是 6.5-6.8 µs/submit，
   即 **2.7-2.8×**，不是 5×；5× 是把 291 个 HC 提交放进 "97 次调用"分母的产物
   （A16 分母说明、HC16 §0）。这 13 ms 的绝对差额仍值 ~+30% e2e，值得定位，但排序时要按
   2.7× 起步。
7. **合成回路的"减去一次 wait = −14%"没有迁移到 e2e**：`AILA_Q4E_JOIN_RELEASE` 实测
   **+1.80%**（A/A spread 2.24%）、`AILA_Q4E_CPU_DEFER_ARENA_WAIT` 实测 **−1.08%**、
   pair2 **−1.6%**——三个控制面候选都在噪声里。任何以"放行 L0 动态批"为机制的新改造
   都必须先解释这条负结果，或在探针里给出比 ±100 µs/层更大的、可解释的量。
   （OOO 探针给出的"wait 节奏 −40%"是纯合成回路；生产 decode 的每层 wait 是 router 读回，
   host 必须等，所以结构方向只能是减少/摊薄同步点——见 §3.4。）
8. **不建议把 compute queue 直接翻成 out-of-order（已有实测否决，见 §0.1）**：31 个受影响的翻译单元、~30 处
   注释级隐式依赖、共享 scratch（`attn_xq/xs`、`lora_scratch_`、oneDNN user-mode scratch、
   `linear_scratch_`）、以及 `tests/inference/ExecutionResourcesTests.cpp:108` 的显式断言。
   可行的是**分域**：copy 走 copy queue、独立分支走第二 compute queue，两者都用显式 event
   串接；这与既有的 `AILA_Q4E_JOIN_RELEASE`/`CPU_DEFER_ARENA_WAIT` 属同一层控制面。

---

## 1. 事实清单（file:line / 日志出处）

### 1.1 compute queue 与全部 queue 实例

| # | 位置 | 属性 | 用途 |
|---:|---|---|---|
| 1 | `src/core/Context.hpp:29-40` `Context(Trace*, bool enableProfiling=false)` | **必定 `property::queue::in_order()`**；`enableProfiling` 时才追加 `enable_profiling()` | 全局 compute queue；`queue()` 访问器 `:57` |
| 2 | `src/inference/Model.cpp:35-58` `MakeDeviceContext()` | 读一次 `AILA_Q4E_DEVICE_QUEUE_PROFILING`（默认 0）；为 1 时 `Context(trace, true)`，构造被拒则 warn + 回退纯 in-order | worker/CLI 的设备队列工厂 |
| 3 | `src/models/Qwen4ExpBnb4Backend.cpp:892-893` `copy_q_` | `in_order` only（无 immediate list、无 profiling） | 冷专家 H2D / swap 的 DMA；用户 `:1511`、`:1757`、`:2674`、`:3081`；析构 `:314-317` 等它 drain |
| 4 | `src/models/Qwen4ExpExpertCache.cpp:770-779`（第二个 `copy_q_`） | `in_order` + `ext::intel::property::queue::immediate_command_list()`（profiling 可选） | legacy expert cache 的 DMA 路径（decode 已并入 #3 的调用者） |
| 5 | `src/models/Qwen4ExpPle.cpp:327-329` `copy_q_` | `in_order` | PLE embed/table 的 H2D |
| 6 | `src/models/Qwen4ExpHostArena.cpp:104` `standalone_queue_` | `make_unique<sycl::queue>(q)` — 拷贝 compute queue 的属性 | arena 自持队列（`Qwen4ExpHostArena.hpp:87` 的 `Queue()`） |
| 7 | `src/models/Qwen4ExpAdaptiveCache.cpp:194` `standalone_queue_` | 同上，拷贝属性 | adaptive cache 无 ctx 时的兜底队列 |
| 8 | `src/profile/Device.cpp:32` | 局部临时 queue | 仅用于查询设备信息，非热路径 |

补充事实：
- **copy engine 是强制的**：`aila::env::enable_l0_copy_engine()`（`src/utils/EnvUtils.hpp:179-201`）
  在 device 初始化前把 `UR_L0_USE_COPY_ENGINE=1` / `SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE=1`
  写进进程环境；`src/main.cpp:97`、`src/worker/WorkerMain.cpp:494`、多个 `perf/qwen4exp/*.cpp`
  都会调用。因此 compute queue 上的每次 `memcpy` 都会落到**另一个硬件引擎**，in-order 语义要求
  跨引擎的事件/fence 串接。
- **跨队列的强约束是源码写死的**：`Qwen4ExpAdaptiveCache.cpp:565-566` 与
  `Qwen4ExpLayerStage.cpp:182-183` 在 copy queue 不是 in-order 时直接 `throw`。
- **worker 看不到 `UR_*` 环境变量**：`src/runtime/WorkerProcess.cpp:943` 用
  `build_isolated_environment()` 生成子进程环境，前缀表在
  `src/runtime/ChildEnvironment.cpp:28-39`（`XPTI_`、**`UR_`**、`ZES_`、`ZET_`、
  `ZE_ENABLE_`、`OCL_ICD_`、`SYCL_CACHE_`、`__KMP_REGISTERED_LIB_`），擦除发生在
  `:258-265`；`docs/Environment_Variables.md:59-73` 明文记录。`WorkerMain` 只读 `PATH`
  （`WorkerMain.cpp:176-188`、`:510`），不应用任何 env dict。
  → e10 bundle 的 `worker_environment`（`tmp/q4e-sprint-r1-20261014/e10-bundle/result.json`）
  里 `UR_L0_BATCH_SIZE/UR_L0_DEVICE_SCOPE_EVENTS/UR_L0_USE_IMMEDIATE_COMMANDLISTS=0`
  **只到 AilaServer，不到 AilaWorker**；worker 实际能看到的同义开关是
  `SYCL_PI_LEVEL_ZERO_BATCH_SIZE=0` / `SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=0` /
  `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0` / `SYCL_UR_USE_LEVEL_ZERO_V2=0`
  （这些不匹配擦除前缀，会继承）。探针是独立 exe，`run_probe.ps1:18-22` 两组都设了。
  这是"探针 == 生产 queue 环境"这一前提里唯一未经确认的环节（§2 E-C）。

### 1.2 提交密度 vs 依赖密度（in-order blast radius 的度量）

| 指标 | 数值 | 出处 |
|---|---:|---|
| `src/` 的 `.submit(` 站点 | **203**（另有 `memcpy` 139、`memset` 24） | `rg -c "\.submit\(" src -g '*.cpp' -g '*.hpp'` 求和（本次） |
| `src/` 的 `depends_on` 站点 | **33** | `rg -n depends_on src`：`MoeExpertOps.cpp` 24、`MoeExpertXmx.cpp` 3、`MoeDecodeVariant.cpp` 2、`Qwen4ExpAdaptiveCache.cpp` 2、`Qwen4ExpLayerStage.cpp` 1、`MoeGroupedXmx.cpp` 1 |
| 显式 in-order 断言（测试） | 1 | `tests/inference/ExecutionResourcesTests.cpp:108-109` "Execution Context must retain an in-order queue" |
| 源码里"依赖 in-order"的注释 | ~30 处 | `rg -n "in-order" src -g '*.cpp' -g '*.hpp'`（本次；清单见下） |
| 每层提交数（e10，`count==1`） | GDN 19 / QSA 25 | A16 §1a-1d（本次逐条复核了其中的站点存在性） |
| 每 token 提交数 | 984-987 | A16 §1d |
| 每层 host SYCL wait | 2（`resident_input_event_`、`actual_route_event`） | `Qwen4ExpExpertExecutor.cpp:745`、`Qwen4ExpBnb4Backend.cpp:2816` |

注释级隐式依赖的关键几条（这些是 OOO 会静默破坏的地方）：

| 位置 | 内容 |
|---|---|
| `src/models/Qwen4ExpPrefill.cpp:12-13` | "All submitted work shares the in-order compute queue; no state is published until ForwardImpl completes" |
| `src/models/Qwen4ExpBnb4Backend.cpp:1124-1127` | `attn_xq`/`attn_xs` "Shared by successive decode projections on the in-order queue" |
| `src/models/Qwen4ExpBnb4Backend.cpp:2849` | "The in-order queue preserves router publication → quantized resident projections → shared work → ordered rank merge" |
| `src/models/Qwen4ExpBnb4Backend.cpp:1665` | "in-order merge covers both reader subsets and metadata uploads"（死重载，见 A16） |
| `src/ops/Bnb4BitLinear.cpp:2368-2369` | LoRA 中间结果复用 "The in-order queue safely reuses the retained single-row adapter intermediate" |
| `src/ops/Bnb4BitLinear.cpp:3278` | Q8 输入复制的顺序由 in-order 保证 |
| `src/ops/MoeExpertOps.hpp:155` | "Caller provides dependencies through the engine's in-order compute queue" |
| `src/ops/MoeExpertXmx.hpp:98-104` | 整块 MoE pipeline 排序 + "Context always submits on an in-order queue" |
| `src/core/Context.hpp:137-139` | oneDNN user-mode scratch "shared by sequential submissions on the in-order queue"（`scratchpad_memory()` 会随 primitive 增长而**重分配并 free 旧指针**） |
| `src/models/Qwen4ExpExpertExecutor.cpp:744`、`:1052`、`:1248`、`:1479` | CPU 激活/结果包/行合并的 GPU 写序 |

### 1.3 `src/ops/Bnb4BitLinear.cpp:2709` 的 in-order 检查

```cpp
bool bf16_q8_head_supported(Context& ctx, int hidden, int vocab) {
    if (hidden <= 0 || hidden % 64 != 0 || vocab <= 0 || vocab % 16 != 0 ||
        !ctx.queue().has_property<sycl::property::queue::in_order>()) return false;   // :2707-2709
    ... // 之后是 device aspect / SG16 / WG256 的能力缓存（:2710-2724）
}
```

- 它保护的是 **Q8 LM head**：`bf16_q8_head_forward`（`:2741-2791`）对每一行
  **提交 2 个 kernel**——`quantize_bf16_head_blocks`（返回值被丢弃，`:2764-2766`）与
  受量化 GEMV（`:2768`，无 `depends_on`），并且两行以上时**共用同一份 `xq/xs`**
  （`Bnb4BitLinear.hpp:21-23` 明写 "one-row xq[H] and xs[H/64] scratch is reused only on
  the in-order queue"）。
- 违反（OOO ctx）时的行为：
  1. 加载期 gate `Qwen4ExpBnb4Backend.cpp:547-573` 的 `use_q8_head` 直接为 false →
     `AILA_Q4E_LM_HEAD_Q8=1` 被**静默降级**为 bf16 转置权重 + oneDNN head，只留一行
     `[Q4E LM Head Q8 Load] requested=1 actual_enabled=0 unsupported_shape_or_context=1`（`:573`）。
  2. 已经走过 `bf16_q8_head_forward` / `quantize_bf16_weights_q8_64` 的调用会抛
     `std::invalid_argument("… unsupported Context, shape or tensors")`（`:2727-2739`、`:2741-2751`）。
     现存的调用点只有 `Qwen4ExpLayerMajor.cpp:453`（layer-major prefill）和
     `Qwen4ExpBnb4Backend.cpp:3102`（chunk-major head），两者都以 `lmHeadQ8Enabled_` 为前提，
     所以 OOO 下只会"路径消失"，不会算错。
- 推论：**这是全仓唯一一处"把 in-order 当 API 契约写进代码"的地方**；它是 OOO 迁移的
  最小必需改动点之一（改成显式 event 链 + 每行独立 scratch，或保留该 gate 并在 OOO 下计入
  已知损失）。

### 1.4 `Context::queue()` 的成本

| 组成 | 实现 | 成本 |
|---|---|---|
| `async_errors_->Rethrow()` | `src/core/AsyncErrorState.hpp:31-40`：`if (!Failed()) return;`，`Failed()` 是 `atomic<bool>` 的 acquire load；只有真出错才拿 mutex | ~1-2 ns（预测分支 + 一次 load） |
| `CheckTrace()` | `src/core/Context.hpp:224`：`if (trace_) trace_->Check();`；`ExecutionTrace::Check()`（`ExecutionTrace.cpp:172-174`）= `Failed()` = 又一次 atomic load，失败才 throw | worker 默认 `trace_==nullptr`（`AILA_EXECUTION_TRACE_DIR` 未设，`ExecutionTrace::FromEnvironment`），所以只有 1 次空指针判断 |
| `return q_;` | 引用返回 | 0 |

- 调用密度：一次提交 1-3 次 `ctx.queue()`（典型是"算 range 时取一次、submit 时再取一次"），
  decode 每层 30-60 次、每 token ~1500-3000 次 → **合计 <0.1 ms/token**。HC16 §3 已用同口径
  判过。结论：即使把访问器优化成零成本，也动不了 18.7 ms 的块计时；**它不是 5× 的候选**。
- 唯一要注意的是 HC 插桩本身（未提交改动）：每个 submit 前后各 1 次 `steady_clock::now()`
  （gate off 时只测一次 cached bool），291 次/token ≈ 0.02-0.03 ms，可作为"测量偏差"忽略。

### 1.5 逐 token / 逐层的提交链路（e10 bundle，`count==1`）

e10 相关开关（`tmp/q4e-sprint-r1-20261014/e10-bundle/result.json` 本次核对）：
`MOE_FRONT_FUSED=1`、`GDN_HISTORY_FUSED=1`、`DECODE_PENDING_MLP_RESIDUAL=1`、
`KV_QUANT=1(int8)`、`ROUTER_RESIDENT_FUSED=0`、`ATTN_NF4_Q8=0`、`GPU_NF4_Q8=0`、
`MOE_SHARED_FUSED`/`DECODE_FUSED_HC`/`JOIN_RELEASE`/`LAYER_STAGE`/`LM_HEAD_Q8=0`、
`CPU_DEFER_ARENA_WAIT=0`、`WORK_SUMMARY=1`（diag 臂）。

**共享块（每层 16 个提交，A16 §1a）**

| # | 调用点 | 提交站点 | 原语 / 规模 | 结果事件去向 |
|---:|---|---|---|---|
| 1 | `Qwen4ExpBnb4Backend.cpp:2628` attn HC norm（pending 路径） | `HyperConnectionOps.cpp:397` | `submit`，`nd_range 4*256`，SLM 256 f32，~12 captures | 丢弃 |
| 2 | 同上 down | `HyperConnectionOps.cpp:467` | `submit`，`ceil(324/16)*256` 线程 | 丢弃 |
| 3 | 同上 up | `HyperConnectionOps.cpp:549` | `submit`，`160*256` 线程 | 丢弃 |
| 4 | `Qwen4ExpExpertExecutor.cpp:746`（调用点 `Qwen4ExpBnb4Backend.cpp:2701`/`:2725`/`:2841`） | 自身 | **`memcpy` D2H ~5 KB**（`x_host_`） | 存 `resident_input_event_`；下一层 `:745` host wait；作为 CPU job 的 `activation`（`:784`/`:886`/`:936`） |
| 5 | `Qwen4ExpBnb4Backend.cpp:2740` | `Bnb4BitLinear.cpp:2811` | `submit`，router GEMM + shared-gate 合一 | 丢弃 |
| 6 | `Qwen4ExpBnb4Backend.cpp:2766`/`:2773` | `MoeExpertOps.cpp:123`（`launch_router_softmax_topk`） | `submit`，softmax+top-k，kernel 内直写 pinned host ids/weights | `actual_route_event` → `:2816` host wait |
| 7 | `Qwen4ExpBnb4Backend.cpp:2902` | `Bnb4BitLinear.cpp:1493`（`packed_nf4_gemv_bf16_impl`，gate-up+SwigLU 变体） | `submit` | 丢弃 |
| 8 | `Qwen4ExpBnb4Backend.cpp:2908` shared down | `Bnb4BitLinear.cpp:1493` | `submit` | 丢弃 |
| 9 | `Qwen4ExpBnb4Backend.cpp:2680` mlp HC norm | `HyperConnectionOps.cpp:324` | `submit`，SLM 256 f32 | 丢弃（残差所有权转移给 `pending_mlp`） |
| 10 | 同上 down | `HyperConnectionOps.cpp:467` | 同 #2 | 丢弃 |
| 11 | 同上 up | `HyperConnectionOps.cpp:549` | 同 #3 | 丢弃 |
| 12 | `Qwen4ExpBnb4Backend.cpp:1525`（`RunResidentDecode:1286` 起） | `MoeExpertOps.cpp:873`（`launch_moe_decode_gate_up_swiglu_grouped_dev`） | `submit`，16×`MoeExpertSlotDev`（约 6 ptr/项）+ 16 ranks + token 数 ≈ 0.8-1 KB by-value | 记为 `last` |
| 13 | `:1531` | `MoeExpertOps.cpp:1233`（`launch_moe_decode_down_to_rank_f32_dev`） | `submit` + **`depends_on(last)`** | `last`；`RecordResidentConsumer` |
| 14 | `:1577` | 自身 | **`memcpy` H2D 64 B**（`resident_weights_`） | 丢弃 |
| 15 | `:1580`/`:1582` | `MoeExpertOps.cpp:6379`（或 `:6428` cpu-host-rows） | `submit` + `depends_on(&last)` | `merge` → `FinishResidentCpuRead`（守 `:745` 的下一层 D2H） |
| 16 | `:3034`（`MOE_SHARED_FUSED=0`） | `Qwen4ExpBnb4Backend.cpp:137/161/188` 之一 | `submit` | 丢弃 |

**注意力增量（GDN 层 +3，A16 §1b）**：`Qwen4ExpPrefill.cpp:64` all_proj GEMV
（`Bnb4BitLinear.cpp:1493`）→ `:77` `run_linear_delta_decode_gpu`（submit 站点
`GatedDeltaNetOps.cpp:115`，fastpath，4 个小 SLM 数组）→ `:81`
`TryForwardDecodeCommitHistory`（`Bnb4BitLinear.cpp:2389`→`packed_nf4_gemv_bf16<true>` 的
`:1493`，o_proj+conv-history 合一）。fallback 对（`:88` commit + `:90` project）不取。

**QSA（full）层增量（+9，A16 §1c）**：`:144` qkv GEMV、`:150`
`decode_prepare_qgkv_packed_partial`（submit `ElementwiseOps.cpp:1312`）、`:209`
`DenseLinear(index_qk_proj)`→`bf16_gemv_bf16`、`:213` `qsa_prepare_causal_query_and_select_blocks`
（内部 3 个 submit：`QsaIndexerOps.cpp:204` Q-norm+RoPE+KV 写、`:1082` block scores、`:1301` top-k 选择，
后者返回值是 **host** 值）、`:248` `attention_decode`→`AttentionOps.cpp:2210-2225` 分支
（phase-1 `:476` + merge `:686`）、`:256` o_proj GEMV。每 4 token 多 1 个
`qsa_compress_key_blocks`（`QsaIndexerOps.cpp:944`）。

**每 token 收尾**：mixer 3 个提交（`Qwen4ExpBnb4Backend.cpp:3090`→同一个
`RunHyperConnectionCombined`）＋ head（`:3122` `forward_head` → e10 走 `:3117`
`lm_head_.forward`（bf16 转置 + oneDNN）；Q8 变体在 `:3102`，每行 2 submit，本例关闭）
＋ `EmbedTokens` 的 1 次阻塞 D2H（`:2312`）。

**host 侧构造（每层/每 token，均为小对象）**

| 构造 | 位置 | 规模 |
|---|---|---|
| `std::array<ops::MoeExpertSlotDev, 16> hot{}`（含 `cold{}`、staging 版） | `:1521`、`:1540` | 各 ~1 KB 零初始化 |
| `ResidentColdWork cpu_work{expert_executor_}`（RAII 守卫） | `Qwen4ExpBnb4Backend.cpp:1305` | 32 B 级 |
| `std::array<int,16>` 分类数组、`d1_stash_*`（thread_local，仅 `CPU_ROUTE_EARLY`） | `:1445-1448`、`:2967` | ~128 B |
| `std::optional<GpuResidentLayerLease>` | `:2707-2711` | e10 为 `use_adaptive_pipeline_` → 不构造 |
| `std::vector<DevMark> dev_marks` + 13 处 `dev_mark()` | `:2574-2584` | **只在 `DECODE_STAGE_EVENTS` 臂**：每层 ~11-12 次 `ext_oneapi_submit_barrier()` 提交 + 末尾 1 次 wait + 每个 mark 2 次 `get_profiling_info`（`:3198-3205`） |
| `hsplit_*` 的 `steady_clock` 对 | `:2537-2546` 及 5 对调用（`:2738`、`:2763`、`:2896`、`:3032`、`:3040`） | `WORK_SUMMARY=1` 时 ~12 次 `now()`/层 |
| USM 分配 | `ensure_runtime_buffers(ctx, count)`（`:2293`） | 稳态 decode 全部命中容量判断；**无分配** |
| oneDNN descriptor/primitive | decode `count==1` 走 `packed_nf4_gemv_bf16`（`Bnb4BitLinear.cpp:2408-2411`）与 `bf16_gemv_bf16`（`Qwen4ExpBnb4Backend.cpp:2065-2068`） | **无绑定/无 primitive cache 查询** |
| `Context::scratchpad_memory()` | `Context.hpp:137-148` | decode 不进入（只在 oneDNN prefill 路径） |

### 1.6 profiling 环境在提交路径上加的东西

| 环境变量 | 读取点 | 对提交路径的影响 | 量级 |
|---|---|---|---|
| `AILA_Q4E_WORK_SUMMARY` | `Qwen4ExpBnb4Backend.cpp:2455` | 打开 `stage()` 与 5 对 `hsplit_*` 计时 + 每 forward 1 行日志；**不** synchronize（`:2525-2531` 里只有 `s_stages` 才 sync） | ~12 次 `now()`/层，10^-2 ms/token |
| `AILA_PROFILE_Q4E_STAGES` | `:2489` | `stage()` 每次调用先 `ctx.synchronize()` → 每层 8 次全队列 drain | **每 token +384 次 drain，必须关闭** |
| `AILA_PROFILE_Q4E_DECODE` | `:2454` | forward 结尾 1 次 `ctx.synchronize()`（`:3402`）+ 1 行日志 | ~1 次 drain/token（阻断跨 token 流水） |
| `AILA_Q4E_PROFILE_COMPONENTS` | `:2267-2269` | 仅在 `decode_verification && count==2` 生效（`:2070-2078`、`Qwen4ExpPrefill.cpp:93-101`）；`count==1` 只有 1 个 bool + `fill(0)` | 0（T1） |
| `AILA_Q4E_DECODE_STAGE_EVENTS` | `:2554-2566` | 每层 ~11-12 次 `ext_oneapi_submit_barrier()`（`dev_mark`）+ 末尾 1 次 wait + 每 mark 2 次 `get_profiling_info` + `std::vector` push_back；并要求 profiling queue | **+~576 提交/token（+58%）**，且整个进程 queue 属性改变 |
| `AILA_Q4E_DEVICE_QUEUE_PROFILING` | `Model.cpp:42-45` | 给 compute queue 加 `enable_profiling` 属性（全局，进程生命周期） | L0 侧每命令时间戳；是 `dev_ms` 的前提 |
| `AILA_DEBUG_Q4E_DUMP_DIR` | `:2453`、`:3056`、`:3094`、`:3143` | 每层 `dump_hidden_row()`（`:200-208`）= **阻塞 `memcpy_d2h` + 文件写**，48 次/token | 每层一次全 drain，量级同 PROFILE_Q4E_STAGES |
| `AILA_Q4E_TRACE_MARKERS` + `AILA_Q4E_CACHE_TRACE` | `:2461`、`:2479-2480` | 1 次 `single_task` 标记 + 若干 host interval 记录 | 小 |
| `AILA_Q4E_ROUTE_REPLAY_COUNT` | `:2467`、`:2791-2812` | 每层 16 次 slot 查询 + 比较 | 小 |
| `AILA_EXECUTION_TRACE_DIR` | `DeviceState.trace`（`Model.cpp:66`）→ `Context::CheckTrace` | `queue()` 每次多一次 atomic load；`ExecutionScope` 在 alloc/free/阻塞 copy/synchronize 上写+flush JSONL | 每层 1 次阻塞 D2H（`EmbedTokens`）时才会写；默认关 |
| `AILA_Q4E_HC_SUBMIT_TIMING`（未提交） | `HyperConnectionOps.cpp` 顶部 + 4 个 submit | gate off：每次 submit 多 1 个 cached bool 判断；gate on：2 次 `now()`/submit | 0.02-0.03 ms/token |

**读法**：任何用于讨论 "30 µs" 的臂，必须核对 `dev_events=0`、无 `PROFILE_Q4E_STAGES`、
无 `DUMP_DIR`；`[Q4E AdaptiveWork]` 行本身要求 `WORK_SUMMARY=1`（即 B16/A16 引用的 e10 日志
全部来自 diag 臂），这是可接受的，但 `PROFILE_Q4E_DECODE=1`（1 次 drain/token）会把
`submitted_ms` 抬高一点点，跨臂比较时要记住。

### 1.7 探针已测 / 未测（避免重复实验）

已测 `[日志]`（除最后一行外都在 `tmp/q4e-bottleneck-audit-20261016/`；最后一行来自
`tmp/q4e-structural-20261016/`，由并发落地的 `8f4950e` 产出）：

| 实验 | 结果 | 出处 |
|---|---|---|
| 48×20 kernel + 每层 wait（in-order，regular list，copy engine on，BATCH_SIZE=0） | wall 13.842-14.575 ms/token，host submit **6.512-6.839 µs/kernel**，device busy 6.215-6.473 ms | `submit-regular.log`、`submit-regular-2.log` |
| 同上一律只在末尾 wait | wall 8.292 ms，host submit 7.775-7.869 µs/kernel | 同上 |
| 空 kernel + wait 往返 | **59.1-59.8 µs/次** | 同上（`sync_only`） |
| 静态 graph（每层 1 张，20 kernel） | 12.671-12.755 ms/token（−13%），submit 33.7-34.2 µs/层 | 同上 |
| BATCH_SIZE=1/2/4/8/16 | 1: 43.9 ms + 17.3 µs/kernel；16: 15.13 ms + 5.07 µs/kernel；4/8 介于中间 | `submit-batch*.log` |
| immediate command lists | 41.546 ms/token（**最差**） | `submit-immediate.log` |
| "router sync → 2 kernel → host 自旋(condvar 模拟) → wait"，batch≥4 | **kernel 被扣在批里**：no_flush 313/476/639 µs vs status_query 213/327/521（spin=100/230/400） | `submit-batch4/8/16.log` 的 `[OVERLAP ARM]` |
| batch=1/2 | 两种模式无差别（212-221 / 329-333 µs） | `submit-batch1/2.log` |
| 18 线程 DRAM 流（CPU 争用） | submit 5.9 → 7.0-7.4 µs（+20%），token 19.7 → 29.0 ms；join 前放行 → 25.4 ms | `contention-A1.log`、`contention-flush.log` |
| `sycl::event{}` 比较 | `default != sycl::event{}` 恒为 true；默认事件 complete、无队列 | `event-eq.log`、`event_eq_probe.cpp` |
| **in-order vs out-of-order**（同一形状，模板 `probe_ooo.cpp`） | in-order 链 13.915 ms / 6.454 µs·kernel⁻¹；OOO+`depends_on` **14.926 ms / 8.171 µs（+7% / +27%）**；双独立链 in-order 14.433（overlap 0.431）/ OOO 15.252（overlap 0.419）；OOO no-wait 8.607 ms（overlap 0.684）；hold 模式 in-order 337/278 µs、OOO 325/278 µs | `docs/plans/qwen38-r1-ooo-queue-probe-2026-10-16.md`、`tmp/q4e-structural-20261016/ooo-probe-L20.log` |

**仍未测（本次确认，OOO 已由 `8f4950e` 补上）**：带 per-layer `memcpy`（D2H/H2D + copy engine）
的提交成本；大 by-value functor 的提交成本；48 组不同 USM 指针轮换的提交成本；
`UR_L0_*` 与 `SYCL_PI_*` 两组开关的可分辨影响；"等待节奏"在生产 decode 的可压缩量
（OOO 探针的 −40% 来自纯合成回路，生产每层 wait 是 router 读回硬依赖）。

已否证 / 不要重复 `[语料]`：CPU 争用不是 5× 主因（+20%）；immediate lists 更差；
execution trace / 日志 / 未缓存 env 读 / `data_as` / oneDNN descriptor 绑定都已排除（A16 §2.3）；
`dev_ms` 的 marker 记账把空闲记进下一个桶（B16 §3.4），不能用它反推"设备忙"。

OOO 的历史前作只到 [HYPOTHESIS] 级：`qwen3.8-flash-next-qsa-selector-results-2026-09-30.md:70-75`
（两个候选分支 + "Context 与多数算子仍依赖 in-order 的隐式顺序；移除该属性会遗漏生产者依赖，
并造成共享临时区冲突"）、`qwen3.8-flash-next-architecture-brainstorm-2026-09-30.md:274`、
`qwen3.8-flash-next-execution-model-audit-2026-09-29.md:138`、
`qwen38-r1-expert-xmx-ab-quality-round1-2026-10-16.md:85`（prefill XMX 想"overlap dequant with
GEMMs (needs an out-of-order path)"）、`perf/qwen4exp/q4e_qsa_selector_summary.py:112`
（"Out-of-order compute queues were audited only"）。**`8f4950e` 是第一条实测**（合成 decode 形状，
结论为负）；prefill 侧"解量化与 GEMM 重叠"是否值得局部 OOO 仍**没有**实测。

### 1.8 块计时口径（后面所有算术的前提）

`stage(int)`（`Qwen4ExpBnb4Backend.cpp:2525-2531`）：`if (!s_stages && !s_work_summary) return;
if (s_stages) ctx.synchronize(); now; stage_ms[s] += delta;`。所以
`hc_host_ms` = **相邻 stage 边界之间的 host wall time**，包含 3 次 `submit` + 分支判断，
不含任何 SYCL wait（除非 `s_stages`）。e10 生产稳态（`e10-bundle-worker.log`，
pos 5945-5950）：`hc_host_ms` 6.45-6.71、`linear_host_ms` 3.26-3.42、
`full_host_ms` 2.73-3.06、`experts_host_ms` 19.3-20.4、`moe_front_ms` 1.70-1.80、
`topk_submit_ms` 0.88-0.96、`shared_submit_ms` 2.88-3.16、`shared_add_ms` 0.62-0.73、
`router_wait_ms` 10.0-11.5、`cpu_compute_ms` 12.3-13.3、`cpu_join_ms` 8.0-9.2、
`submitted_ms` 51.8-55.2（另见 `tmp/q4e-strata-c1/c1-e1-worker.log` 同构样本）。
注意第一个 decode token 的 `hc_host_ms` 是 223 ms（`f1-b1-worker.log` pos=5944）——
one-time 预热（JIT/primitive/第一次触碰）真实存在，**任何"30 µs"结论都不该包含 pos 首 token**。

---

## 2. "30 µs vs 7 µs" 假设列表（按证据强度排序）与最便宜的证伪实验

先固定分母（否则会做错实验）：**HC 块 = 97 次调用 × 3 submit = 291 提交/token**（A16 §1、
HC16 §1）；**整层 host 提交块 = 18.7 ms/48 ≈ 390 µs/层 ÷ 20.5 提交 ≈ 18.5 µs/提交**；
探针同类循环 = 6.5-6.8 µs/提交。所以真正要解释的是 **~12 µs/提交 × 987 ≈ 12.8 ms/token**
（约占 55 ms token 的 23%，若全部消掉 ≈ +30% e2e 上界）。

| # | 假设 | 机制 | 证据强度 | 最便宜证伪（★=不需要改 `src/`） |
|---:|---|---|---|---|
| H1 | **块计时 ≠ submit 之和（归因假象）** | `hc_host_ms` 是 stage-to-stage wall，可能含非 submit 的 host 工作（分支/`pending_mlp` 状态机/`stage()` 本体/日志）；291 × 6.4 µs = 1.86 ms，而块是 6.6 ms，差额 4.7 ms 至今无候选 | **强**：HC16 §1/§3 已证明 HC 调用点自身只有 3 次 submit + 纯分支；A16 §2.1 明说这些是 block cost | **已在树里**：跑一个 `AILA_Q4E_HC_SUBMIT_TIMING=1` 的 decode 臂，读 `hc_submit_*_us` 与 `hc_host_ms` 的差（每 op 的 sum/min/max 区分"普遍贵"与"少数极贵"）。需要一次 worker 构建（插桩未编译过） |
| H2 | **生产每层多出来的 memcpy 把 L0 批切开，边界的 submit 付 standalone 启动价** | 生产每层在 compute queue 上有 2 次 `memcpy`（5 KB D2H、64 B H2D）＋2 次 host wait；copy engine 打开后每次 copy 都要跨引擎事件链，等于一次批边界。探针**一次 copy 都没有**、只有 1 次 wait。批内 submit ≈ 6 µs，边界后第一个 submit ≈ 20-60 µs（空 kernel+wait=59 µs） | **中强**：`submit-*.log` 的 `[OVERLAP ARM]` 证明"批被扣住/放行"确实存在（status_query 省 60-120 µs/层），BATCH_SIZE=4/8/16 与 =1 是两种完全不同的机制 | ★**探针扩展**（tmp-only）：给 submit_probe 加"每层 D2H 5 KB + H2D 64 B，copy engine on、位置照抄生产"的臂，重读 host µs/kernel。若仍 6-7 µs，H2 死；若跳到 15-30 µs，机制坐实（但仍须过 H2'） |
| H2' | **H2 的 e2e 反证** | 若批边界是主因，去掉一个 wait 应显著加速；实测 `JOIN_RELEASE` +1.80%、`DEFER_ARENA_WAIT` −1.08%、pair2 −1.6% | **强（反向）**：三份 A/B/B/A 收据 | 不需要新实验：把 H2 的预期收益按"最多 = 已被移除的边界数占比"重算。若 H2 成立但 e2e 不动，说明这些边界已与 CPU join/router 重叠，**收益是 0**，改造只能进"顺手清理"清单 |
| H3 | **submit 自身成本 ∝ functor/参数宽度** | MoE 两发各带 16×`MoeExpertSlotDev`（≈0.8-1 KB by-value）+ 16 ranks；kernel 参数 marshal 走 `zeKernelSetArgumentValue`×N；探针 kernel 只有 ~32 B | **中**：A16 §2.2 L3；本次确认 `sizeof` 级（`hot{}` 1 KB 零初始化/层） | ★**探针臂**：把 submit 的 functor 换成携带 1 KB `std::array` 的结构，对比 host µs/kernel |
| H4 | **USM 指针工作集大 → 每次提交做分配查表** | 生产每 token marshal 数千个不同 USM 分配（48 层权重、KV、scratch）；探针只有 8 个指针反复复用 | **中**：A16 §2.2 C5；无人测过 | ★**探针臂**：预分配 48×N 组指针，在 97 次调用里轮换（bench-only，0 行 `src/` 改动） |
| H5 | **`Context::queue()` 访问器** | 2 次 atomic load + 空指针判断 × 1500-3000 次/token | **弱（已基本否证）** | 不需要实验（`AsyncErrorState.hpp:31-33` + `Context.hpp:224` 已够；上限 <0.1 ms/token） |
| H6 | **CPU 争用** | 18 线程 DRAM 流抢 coordinator | **已否证**：+20%（5.9→7.0-7.4 µs） | 不要重测 |
| H7 | **immediate command list** | 每次 submit 直接走立即列表 | **已否证**：41.5 ms/token | 不要重测 |
| H8 | **测量臂本身的 profiling 偏差** | `PROFILE_Q4E_DECODE=1` 加 1 次 drain/token；`DECODE_STAGE_EVENTS` 会 +576 提交/token 并改 queue 属性 | **中**（本次新增） | ★不需要新 GPU：逐臂核对 env（`result.json.worker_environment` + `dev_events=` 字段）。**禁止**用 `DECODE_STAGE_EVENTS`/`DEVICE_QUEUE_PROFILING` 的臂测 submit 成本 |
| H9 | **首 token / 预热** | 第一个 decode forward 的 `hc_host_ms=223 ms`（steady 6.6）；短臂的均值会被拉高 | **强**（`f1-b1-worker.log` vs `e10-bundle-worker.log`） | ★只读日志：用 pos≥2 的样本做中位数；任何结论先剔除 pos 首 token |
| H10 | **稳态 submit 真的 ~20 µs（CPU 在 queue 内部阻塞）** | 生产每层 device 工作量 ≈ 1 ms，host 提交块 ≈ 390 µs；L0 command list 池有上限（in-flight list 数 × batch 阈值），pool 用尽时 `submit` 内部要等 fence → 直接计入 submit 时间 | **中**：探针 host/device 是 137/130 µs（同量级），生产是 390/1000 µs；若池上限是"命令条数"，两者行为会不同 | ★**探针臂**：把每 kernel 的设备时间从 ~7 µs 拉到 ~50 µs（加循环），其余不变，重读 host µs/kernel。若 host 变 20-30 µs → 机制成立，H1 与 H2 都只是它的表现 |

**OOO 探针（`8f4950e`）对本章的两条直接输入**：

- 显式 `depends_on` 的 OOO 链比同形状 in-order 链贵 **+1.7 µs/提交（6.45→8.17，+27%）**，
  wall 慢 7%；双独立链也不产生额外重叠（overlap 0.43 vs 0.42）→ "改用 OOO + 显式依赖"
  在 submit 侧是**加法**。H2 里"边界后 submit 更贵"的机制与它同源：每次提交的固定开销
  随队列状态/依赖数变化，而不是随"队列类型"变化。
- 每层 wait 的数量是两种队列**共同**的节流阀（去 wait：14.43 → 8.61 ms，−40%；hold 行为
  in-order 337/278 µs vs OOO 325/278 µs 完全一致）→ H2 的"批边界"名单里
  **host SYCL wait ≥ compute-queue memcpy**；H2 的实验应把 wait 与 copy 拆成两个因变量
  （先测 copy，再加 wait）。

补充建议（一次性 instrumentation，需要改 `src/`，优先级低于上面三颗星）：
把 HC 插桩的同一模式套到 **每个阶段的第一个/最后一个 submit**（attention、shared、front），
或按 H2 的邻居位置给 submit 打 tag（"批边界后第 k 个"），一次性拿到 987 个提交的
"冷/热"直方图——这能把 H1/H2/H3/H4/H10 一次分开。VTune 的 GPU-Offload/Level-Zero
attribution（A16 §5 rank 4）仍是判定"UR/L0 自时间 vs 我们的胶水"的替代路径，但需要 GPU。

---

## 3. in-order → out-of-order 改造：成本 / 风险 / 收益

**实测先行（`8f4950e`，2026-10-16 21:43，同形状合成链、生产 L0 开关）**：

| arm | wall ms/token | host submit µs/kernel | 结论 |
|---|---:|---:|---|
| in-order 链，wait/layer | 13.915 | 6.454 | 复现 B16 的 13.8-14.6 |
| **OOO 链 + `depends_on`，wait/layer** | **14.926** | **8.171** | **+7% wall / +27% submit** |
| in-order 两条独立链，wait/layer | 14.433 | 6.924 | overlap 0.431 |
| OOO 两条独立链，wait/layer | 15.252 | 8.526 | overlap 0.419（没有多出来的重叠） |
| OOO 两条独立链，**无 per-layer wait** | **8.607** | 8.280 | 去掉 wait 才有效（−40%） |
| hold 模式（2 kernel + 230 µs host spin） | in-order 337/278 µs；OOO 325/278 µs | — | 放行机制与队列类型无关 |

出处：`docs/plans/qwen38-r1-ooo-queue-probe-2026-10-16.md`、
`tmp/q4e-structural-20261016/ooo-probe-L20.log`、`tmp/q4e-structural-20261016/probe_ooo.cpp`。
**判定：全局 OOO 已被实测否决（本 GPU/驱动/形状下为负收益），`Context` 保持 in-order。**
下面 §3.1 的 blast-radius 是"为什么它本来也不划算"的静态证据；§3.2 保留的是局部替代路径。

### 3.1 blast radius（全部受影响的模块与文件）

| 类别 | 文件 / 位置 | 为什么受影响 |
|---|---|---|
| **提交站点（要逐个补依赖）** | `src/ops/AttentionOps.cpp` 31、`MoeExpertOps.cpp` 29、`Bnb4BitLinear.cpp` 18、`NormOps.cpp` 15、`QsaIndexerOps.cpp` 15、`HyperConnectionOps.cpp` 13、`ElementwiseOps.cpp` 10、`SamplingOps.cpp` 9、`ConvOps.cpp` 7、`GatedDeltaNetOps.cpp` 3+2、`MoeExpertXmx.cpp` 4、`MoeDecodeVariant.cpp` 2、`HcDecodeVariant.cpp` 2、`GreedyOps.hpp` 2、`Transpose.cpp` 1、`VisionOps.cpp` 1、`MoeGroupedXmx.cpp` 1 | 现在靠"提交顺序 = 执行顺序"；OOO 后必须给每个消费者补 event |
| **模型/编排层** | `Qwen4ExpBnb4Backend.cpp`（11 submit + 19 memcpy）+ `Qwen4ExpPrefill.cpp`（0 直接 submit，全部经 ops；1 个 `memcpy` + 2 个 `memcpy_d2h_async`）+ `Qwen4ExpLayerMajor.cpp`（4 submit + 6 memcpy）+ `Qwen4ExpDecodeFusion.cpp`（2）+ `Qwen4ExpAdaptiveCache.cpp`（2 + 4 memcpy）+ `Qwen4ExpLayerStage.cpp`（1）+ `Qwen35HybridBnb4Backend.cpp` / `Qwen3Bnb4Backend.cpp` / `Qwen3ASRBnb4Backend.cpp`（都用同一 compute queue 与 `linear_scratch_`） | host 编排本身假定"提交即排队"，多处直接用 host 读回的 slot/rank 数组再提交 |
| **共享 scratch（必须分域或加事件）** | `Context.hpp:137-148` oneDNN user-mode scratchpad；`Bnb4BitLinear.hpp:49-61` + `Qwen4ExpBnb4Backend.hpp:672` `linear_scratch_`；`Bnb4BitLinear.cpp:2209-2258` `lora_scratch_`（两步 `parallel_for` 靠 in-order）；`Qwen4ExpBnb4Backend.cpp:1124-1127` `attn_xq/attn_xs`；`gdn_chunk_scratch_`、`xmx_scratch_`、`buf_.*` 激活 bank | 同一缓冲被先后 kernel 复用（WAR/RAW），OOO 下会静默互相覆盖 |
| **跨队列/跨引擎依赖** | `Qwen4ExpBnb4Backend.cpp:892`（copy_q_）、`Qwen4ExpAdaptiveCache.cpp:565`/`:611`、`Qwen4ExpLayerStage.cpp:182`、`Qwen4ExpHostArena.hpp:87` | 现在是 "in-order compute + 显式 H2D event"；OOO 后每条跨队列边都要显式，且 copy queue 的 in-order **断言要保留** |
| **测试/契约** | `tests/inference/ExecutionResourcesTests.cpp:108-109`（断言 in-order）；`tests/ops/Qwen4ExpKernelTests.cpp:850`、`:4384`、`tests/models/*Tests.cpp`（自建 in-order copy queue）；`Qwen4ExpKernelTests` 的 cross-queue 用例 | 断言要改成"仍保持声明属性"，否则 OOO 迁移立刻红 |
| **正确性不变式** | `src/ops/Bnb4BitLinear.cpp:2707-2725`（§1.3）；取消路径 `AGENTS.md §6` "canceled requests must drain submitted GPU work before freeing USM" | OOO 下"队列位置"不再代表完成顺序，释放/取消需要真 event 等待 |
| **文档** | `docs/plans/qwen3.8-flash-next-*` 里的 in-order 设计前提（多篇） | 需要一并修订 |

量级判断：**31 个翻译单元、203 个 `.submit(` 站点（+139 `memcpy` / 24 `memset`）、33 个已有 `depends_on`、~30 处注释级隐式依赖、
至少 4 个共享 scratch、2 个测试契约**。这是一次"重写执行模型的依赖表达"，不是开关级改动。

### 3.2 最小可行路径（按 收益/成本 排序）

**(M1) 保持 in-order，只在"即将进入非 SYCL 等待"前做一次批放行（现状 + 扩展）**
- 现状：`AILA_Q4E_JOIN_RELEASE`（`Qwen4ExpBnb4Backend.cpp:1311-1312`、release at `:1563-1566`）
  实测 +1.80%（噪声内，默认仍 0）。
- 扩展思路：把放行点从"CPU join 前"推广到**每个已知会阻塞的 host 段开头**
  （arena 等待、`JoinResidentCold`、layer-stage 等待），且**严格一处一次**——event status query
  本身也是一次批边界，放行次数多了会反过来切成更多 standalone 提交。
- 成本：低（多个 if + 1 行调用）；风险：低；预期：+0-2%（H2' 已给上界）。

**(M2) 把 compute queue 上的 2 次 memcpy 移到既有 `copy_q_` + 显式 event（"copy queue 并行"）**
- 目标站点：`Qwen4ExpExpertExecutor.cpp:746`（5 KB D2H）、`Qwen4ExpBnb4Backend.cpp:1577`
  （64 B H2D）；可选 `cpu_result_packet` 的 H2D（`Qwen4ExpExpertExecutor.cpp:1059`）。
- 需要的显式依赖（现成的机制就在同一文件里）：D2H 的消费者是
  `job.activation`（`:784`/`:886`/`:936`/`:1110`，本来就是 event 参数）；H2D 的消费者是
  `moe_ordered_accumulate_*`（`MoeExpertOps.cpp:6379`，已经接受 `&last`，可再加一个 event）。
- 反向依赖：copy 必须在产生源数据之后（resident kernel / hidden 写）→ copy queue 需要一个
  来自 compute queue 的 event（`ext_oneapi_submit_barrier` 或源 kernel 的返回 event）。
- 成本：中低（4-6 处，1 个 pinned 双缓冲可选）；风险：中（跨队列事件漏一条就是静默数据错）；
  预期：**先按 H2 的探针结论决定是否值得**；若探针显示 copy 邻居不是贵的那部分，收益 ≈ 0。

**(M3) 多 queue 分域（第二 compute queue）用于真正独立的分支**
- 候选分支（来自 `qsa-selector-results-2026-09-30.md:70-75`）：
  (a) QSA indexer 链（`Qwen4ExpPrefill.cpp:209-246`）与主 QKV/attention 链；
  (b) shared expert（`:2897-2932`）与 routed expert（`RunResidentDecode`）。
- **(b) 已经被现有结构覆盖**：shared expert 的提交在 router wait 之前（`:2896-2935`），
  现在靠"提交顺序 + host 等待 router"重叠；把它挪到第二队列只有在
  "router wait 期间 GPU 实际空闲"时才额外收益，而 `join-release`/`defer-wait` 两个
  实验说明这个窗口对 e2e 不敏感。
- **(a) 有明确依赖边界**（两边都以 `buf_.x` 为输入，attention 汇合处等两边），但需要为
  indexer 链拆出独立 scratch，并给 attention 一个真正的 `depends_on`。
- 成本：中高（新 scratch 生命周期 + 会话级事件管理 + 取消路径）；风险：中高；
  预期：`[推断]` ≤1-2 ms/token（设备侧本来就 45% 空闲，这不是 binder）。

**(M4) 全局 out-of-order + 显式依赖图**
- **已被实测否决**（§3 表）：+27% submit、+7% wall、无额外 overlap。虽然它的静态成本（§3.1）
  本来就高（31 个 TU、共享 scratch、33/203 的依赖覆盖率），现在也不需要再论证收益了。
- 唯一仍开放的是 **prefill 局部 OOO**（"解量化与 GEMM 重叠"，`qwen38-r1-expert-xmx-ab-quality-round1-2026-10-16.md:85`），
  那是一条独立链路、可以用独立 scratch + 显式事件隔离，且不改变 `Context` 的队列属性；
  在 decode 之外单独评估。

**(M5) "等待节奏"方向（OOO 探针点名的杠杆，且不需要改队列属性）**
- 探针结论：`去掉 per-layer wait：14.43 → 8.61 ms（−40%）`；生产里这个 wait 就是 router top-k
  读回（`Qwen4ExpBnb4Backend.cpp:2816`），host 必须等。结构选项：
  (a) **更少/更便宜的同步点**：把 router 读回与其它 host 工作对齐（现有 `JOIN_RELEASE`、
  `CPU_EARLY_START`、`CPU_ROUTE_EARLY`、`STAGEIN_HOIST` 都是这个方向，但单点收益都 <2%）；
  (b) **更少提交**：融合 / 图重放（`qwen38-graph-experiment-roadmap-2026-10-16.md`、A16 §3）；
  (c) **把一次读回摊到更多 token**：MTP / verify batching（`AILA_Q4E_VERIFY_*`、pair2 已验证
  −1.6% 且要过数值门）。
- 成本：方向 (a) 低、(b) 中高、(c) 高；风险：(c) 数值；预期：`[推断]` (a) 已近耗尽，
  (b)/(c) 是唯一还有量级空间的方向。

### 3.3 收益上界（写死三个数，避免后面再争）

| 项 | 值 | 来源 |
|---|---:|---|
| 若 987 个提交都降到探针的 6.5 µs | 18.7 → 6.4 ms/token，≈ +25-30% e2e 上界 | 本次算术，A16 §2.1 的块计时 |
| 图/参数块路线（去 per-submit 成本、保留 wait/join） | −13%（探针），≈ −1.9 ms/token | `submit-regular*.log`、graph roadmap §3.5 |
| **OOO 路线** | **负**：+27% submit、+7% wall（合成）；生产 e2e 未测但无理由更好 | `docs/plans/qwen38-r1-ooo-queue-probe-2026-10-16.md` |
| 去 per-layer wait（合成回路） | −40%（14.43 → 8.61 ms），但生产每层 wait 是 router 硬依赖 | 同上；`qwen38-r1-ooo-queue-probe-2026-10-16.md:Conclusions 2` |
| 已经试过的控制面（join-release / defer-wait / pair2） | +1.80% / −1.08% / −1.6% | jr-*/`result.json`、da-*/`result.json`、attn-pair2 文档 |

---

## 4. 不建议做（明确清单）

1. **不要把 `Context` 的 compute queue 全局改成 out-of-order**：已由 `8f4950e` 实测否决
   （+27% submit / +7% wall / 无额外 overlap），静态成本见 §3.1（31 个 TU、共享 scratch、
   测试断言、取消不变式）。这条不要再排 GPU 时间。
2. **不要用会改变 queue 属性/profiling 的臂去测 submit 成本**：`AILA_Q4E_DEVICE_QUEUE_PROFILING=1`、
   `AILA_Q4E_DECODE_STAGE_EVENTS=1`（+~576 提交/token）、`AILA_PROFILE_Q4E_STAGES=1`
   （每层 8 次 drain）、`AILA_DEBUG_Q4E_DUMP_DIR`（每层阻塞 D2H）。要测 submit 只能用
   `steady_clock`-only 的插桩。
3. **不要再用 `dev_ms`（marker-to-marker span）反推"设备忙/空闲"**（B16 §3.4 的记账假象）；
   它只能做上界。
4. **不要重测**：immediate command lists（41.5 ms/token）、CPU 争用（+20%）、oneDNN descriptor
   绑定、`data_as`、未缓存 env 读、日志/异常路径（A16 §2.3、本次 §1.7）。
5. **不要在修好数值缺陷前重测 `AILA_Q4E_DECODE_FUSED_HC`**（`qwen38-r1-fused-regression-analysis-2026-10-16.md`：
   第二个 token 起文本退化，−19.8% 是正确性回归不是调优问题）。
6. **不要把 `UR_L0_*` 环境变量当作 worker 的有效开关**：worker 环境按前缀擦除
   （`ChildEnvironment.cpp:28-39`），要么用 `SYCL_PI_LEVEL_ZERO_*` 别名，要么把名字加进
   `AILA_WORKER_ENV_PASSTHROUGH`（§1.1 最后一条）。
7. **不要把"每层 wait / 放行"继续当作独立杠杆反复调参**：三个候选都落在噪声内
   （+1.80% / −1.08% / −1.6%），而 OOO 探针里"去 wait"的 −40% 是合成回路的量；除非先给出
   生产侧"哪个 wait 可去/可摊薄"的具体机制（M5 的 (b)/(c)），否则不要再排 GPU 时间。
8. **不要在没有独立 scratch 和显式 event 的情况下，把任意两条链丢进同一个 OOO 队列**
   （`attn_xq/xs`、`lora_scratch_`、oneDNN user-mode scratch、`linear_scratch_` 四条是明确的雷）。
9. **不要假设 `bf16_q8_head_supported()` 的 in-order gate 是"多余检查"**（§1.3）：删掉它会
   在 OOO 上给出未定义读序，或在 OOO 上静默禁用 head-Q8 路径。
10. **不要在没有 `pos≥2` 过滤的情况下引用任何 `hc_host_ms`/submit 均值**（首 token 223 ms）。

---

## 5. 建议的下一步实验队列（按 信息量/成本 排序）

| # | 实验 | 需要改 `src/`? | 需要 GPU? | 成本 | 能回答什么 |
|---:|---|---|---|---:|---|
| E-A | 跑一个 `AILA_Q4E_HC_SUBMIT_TIMING=1`（e10 bundle，`WORK_SUMMARY=1`）decode 臂，读 `hc_submit_{norm,normfb,down,up}_{us,n,min_us,max_us}` | **插桩已在 HEAD `ffb2b08`**，只差一次 worker 构建 | 是（1 臂，~15 min） | 中 | **H1 vs H2/H10**：若 sum≈1.8-2 ms 且 min≈5-7 µs、max 40-60 µs → 5× 是归因 + 少数冷提交；若四类都 20-35 µs → UR/L0 自时间 |
| E-B | 扩展 `tmp/q4e-bottleneck-audit-20261016/submit_probe.cpp`：新增 (i) 每层 5 KB D2H + 64 B H2D（copy engine on，位置照抄生产）、(ii) 48 组指针轮换、(iii) 1 KB by-value functor、(iv) 每 kernel 设备时间 ×7 四个臂 | 否（tmp-only） | 是（~30 min） | 低 | H2/H3/H4/H10 四选一/多 |
| E-C | 同一 decode 臂设 `AILA_WORKER_ENV_PASSTHROUGH=UR_L0_BATCH_SIZE;UR_L0_USE_IMMEDIATE_COMMANDLISTS;UR_L0_DEVICE_SCOPE_EVENTS`（带 `UR_L0_BATCH_SIZE=0/16` 各一次）| 否 | 是（2-3 臂） | 低 | 生产 worker 实际吃到的 L0 开关；同时验证 `SYCL_PI_*` 别名是否已等价 |
| E-D | 读现有 `e10-bundle-worker.log`、`c1-e1-worker.log`、`f1-b1-worker.log`，按 pos 分组统计 `hc_host_ms` 的分布与首 token 影响 | 否 | 否 | 极低 | H9；同时给 E-A 的读法提供基线 |
| E-E | VTune GPU-Offload/Level-Zero attribution on a decode 臂（无 `--diag`） | 否 | 是（~6 min） | 低 | UR/L0 自时间 vs 我们的胶水（H1/H3 的独立证据） |
| E-F | 仅在 E-A/E-B 指向 H2 时才做 M2（copy 移出 compute queue） | 是（4-6 处） | 是 | 中 | 用真实臂验证 M2，而不是用合成回路 |
| E-G | M5 方向的价格：统计生产 decode 的 wait 数量与可摊薄量（router 读回 / MTP / verify batching），先用只读日志 + `qwen38-r1-decode-tail-p7`、`...-attn-pair2-gate` 的收据，再决定是否排 GPU | 否（先做纸面/日志） | 否 | 极低 | OOO 探针点名的"wait 节奏"在生产还剩多少量 |

实验门槛沿用 A16 §5：SSE content sha 逐位一致 + A/A ≤1.5%，采纳门槛 ≥ +3% e2e；
任何"控制面"类改动在没有 E-A/E-B 的机制证据前不排 GPU。E-A 与 OOO 探针（`8f4950e`）是
当前信息量最高的两项，且都不需要新的 `src/` 设计。

---

## 6. 已知风险与遗留

1. 树状态在本次调研期间变了两次：`HyperConnectionOps.{hpp,cpp}` 的插桩与
   `Qwen4ExpBnb4Backend.cpp` 的 HC 计数字段先以未提交形式存在（我按它读行号），随后被
   `ffb2b08`/`6f9a41e` 提交，`6f9a41e` 又在 backend 里 +6 行（XMX 采纳）→
   `Qwen4ExpBnb4Backend.cpp:2943+` 的锚点比 HC16/A16 的时代后移约 6 行；
   `8f4950e` 另行新增 OOO 探针与 graph roadmap。本文已按 `HEAD 8f4950e` 复核主要锚点，
   引用其它文档时注意其 base。
2. `UR_L0_*` 是否真的不影响 worker 行为尚未用实验确认（E-C）；若确有差异，A16/B16 里
   "探针与生产同 queue 环境"的假设需要重新表述。
3. `sycl::event{}` 比较恒不等（B16 §5）使 `ensure_prediction_ready()`
   （`Qwen4ExpBnb4Backend.cpp:2943-2946`）在 `use_adaptive_pipeline_` 下每层对**默认构造事件**
   调一次 `wait_and_throw()`。default event 无队列、状态 complete，预期 ~ns 级，但它每层都在
   提交路径上；建议在 E-B 里用探针确认"对 default event 的 wait"是否为 0 成本
   （若它会触发队列 flush，就是一个未被记账的批边界）。
4. `count==2..4`（verify 窗口 / P7 批）与 layer-major/prefill 的提交清单不同（A16 §6.4），
   本文件的逐层表只覆盖 `count==1`。
5. 收益上界（+25-30%）假设"提交成本可独立消除"；H2' 的 e2e ≈ 0 已部分否定这一点，
   实际应以 E-A/E-B 的机制结论重算。

---

## 附：本文用到的关键锚点速查

- 队列：`src/core/Context.hpp:29-40,57,137-148,224-233`；`src/inference/Model.cpp:35-58,66-75`；
  `src/models/Qwen4ExpBnb4Backend.cpp:314-317,892-893,1511,1757,2674,3081`；
  `src/models/Qwen4ExpExpertCache.cpp:770-779`；`src/models/Qwen4ExpPle.cpp:327-329`；
  `src/models/Qwen4ExpHostArena.cpp:104`；`src/models/Qwen4ExpAdaptiveCache.cpp:194,565-566,611`；
  `src/models/Qwen4ExpLayerStage.cpp:182-183`；`src/utils/EnvUtils.hpp:179-201`；
  `src/main.cpp:97`；`src/worker/WorkerMain.cpp:494`；
  `src/runtime/ChildEnvironment.cpp:28-39,258-265`；`src/runtime/WorkerProcess.cpp:943`。
- in-order gate：`src/ops/Bnb4BitLinear.cpp:2707-2725,2727-2791`；`src/ops/Bnb4BitLinear.hpp:21-27`；
  `src/models/Qwen4ExpBnb4Backend.cpp:547-579`；`src/models/Qwen4ExpLayerMajor.cpp:452-457`；
  `tests/inference/ExecutionResourcesTests.cpp:106-109`。
- 提交站点：见 §1.5 表；HC=[`src/ops/HyperConnectionOps.cpp:324,397,467,549,612`]、
  MoE=[`src/ops/MoeExpertOps.cpp:123,873,1233,6379,6428`]、
  GEMV=[`src/ops/Bnb4BitLinear.cpp:1493,2389`]、GDN=[`src/ops/GatedDeltaNetOps.cpp:115`]、
  重排/合并=[`src/models/Qwen4ExpBnb4Backend.cpp:137,3034,1577,1580-1583`]、
  D2H=[`src/models/Qwen4ExpExpertExecutor.cpp:745-746`]。
- 计时/环境：`src/models/Qwen4ExpBnb4Backend.cpp:2267-2269,2453-2584,3198-3205,3219-3245,3401-3402`；
  `src/core/AsyncErrorState.hpp:29-40`；`src/core/ExecutionTrace.cpp:169-174`。
- 日志：`tmp/q4e-bottleneck-audit-20261016/submit-*.log`、`contention-*.log`、`event-eq.log`、
  `expert2-fixed-L20.log`；`tmp/q4e-structural-20261016/ooo-probe-L20.log`（OOO 实测）、
  `.../probe_ooo.cpp`、`.../probe_paramtable.cpp`；`tmp/q4e-sprint-r1-20261014/e10-bundle-worker.log`、
  `.../c1-e1-worker.log`（`tmp/q4e-strata-c1/`）、`.../f1-b1-worker.log`、
  `.../jr-*/result.json`、`.../da-*/result.json`。
