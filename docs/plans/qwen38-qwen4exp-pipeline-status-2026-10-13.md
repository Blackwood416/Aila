# Qwen4Exp 推理管线现状快照（2026-10-13）

对象：`C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2`（48 层，12 full / 36 linear；H=2560、I=640、512 experts/layer、top-k=10；48×512=24576 个 (layer, expert) key）。环境：A770 16 GiB、驱动 `32.0.101.8724`、oneAPI 2026.1、oneDNN 3.11.2（[qwen38-a770-agent-performance-plan-2026-10-12.md](qwen38-a770-agent-performance-plan-2026-10-12.md):18-21）。

纪律：本文**零 GPU**（未启动 Worker/Server、未跑 GPU ctest/基准），只读源码、docs、tmp 日志。每条结论带出处：源码给 `文件:行`，数字给文档/日志路径；无出处者标 `[未核验]`。本文只描述现状与已有证据，不含新测量，因此不写无出处的性能断言。

---

## 1. 架构与组件

### 1.1 进程与 ABI 分层

```
Client / OpenAI API
  → AilaServer.exe            纯 MSVC C++17 + Drogon 1.9.11（tools/server/CMakeLists.txt:3-16 强制 cl.exe/x64//MD；
                              98 行 find_package(Drogon 1.9.11)；177 行链 AilaServerChat+Drogon）
  → Win32 Named Pipe / IPC v2 (src/ipc/Win32Pipe.cpp, src/ipc/InferenceProtocol.cpp)
  → AilaWorker.exe            DPC++/SYCL，加载 LoadedModel、跑 Scheduler 与 Qwen4ExpBnb4Backend
  → SYCL in-order queue        adaptive GPU 常驻 / staging / 原生大页 RAM 三级专家供给
```

- 能力协商/守卫：`src/ipc/InferenceProtocol.cpp:627` 注册 `qwen4exp_nf4_int8_kv`；`model_capability()` 在 630-635 对 `kvType=="int8"` 只放行 qwen4exp_nf4，非 bf16/int8 返回空；协议 prefill chunk 上限 `kExtendedMaxPrefillChunkSize = 8192`（`src/ipc/InferenceProtocol.hpp:110`）。Worker 侧在 `src/worker/InferenceWorker.cpp:544-559` 校验 capability 并加载。
- KV 类型守卫：`src/inference/Model.cpp:698-707` 读 `AILA_Q4E_SERVER_MTP`、`AILA_KV_QUANT`；量化 KV 默认因 "Qwen4Exp shared sessions require BF16 KV" 被拒，仅 `options.allowQuantizedKv`（server `--kv-type int8`）放行。
- Scheduler：每 Scheduler 一个提交线程、在完整 forward 边界轮转（`src/inference/Scheduler.hpp:102-107`）；`Scheduler::Step` 把 prefillChunkSize 截到 `limits.maxPrefillChunkSize`（`src/inference/Scheduler.cpp:654`）。RequestState/SessionState 分别在 `src/inference/Request.hpp:84`、`src/inference/Session.hpp:134`；职责表见 AGENTS.md §2。
- prefill 分块：`InferenceEngine::forward_prompt_tokens` 按 `AILA_Q35_PREFILL_CHUNK`（代码默认 512）切 chunk（`src/engine/EngineGeneration.cpp:72-95`）；生产用 4096（见 §2.3）。

### 1.2 Qwen4ExpBnb4Backend 对象与加载

- `LoadedModel`（共享、只读）持有 fused NF4 权重与静态层计划；`Session` 独立 KV/DeltaNet 状态；`Request` 单次生成生命周期；`Scheduler` 连续批处理（AGENTS.md §2 资源表）。
- 后端只走 adaptive 管线：`const bool use_adaptive_pipeline_ = true`（`src/models/Qwen4ExpBnb4Backend.hpp:493`）；老 `expert_cache_` 路径在生产已不可达——`src/models/Qwen4ExpBnb4Backend.cpp:2302-2304` 在 adaptive cache 未分配时直接抛 "legacy expert_cache pipeline has been removed"。
- `load()`（`Qwen4ExpBnb4Backend.cpp:375-925`）关键顺序：专家缓存准入 → host arena → adaptive residency：
  - KV 布局判定 `AILA_KV_QUANT && kv_quant_format_is_int8()`（688-690），`kv_element_bytes = 1/2`；
  - `AILA_Q4E_EXPERT_CACHE_MB`（默认 0=自动）或自动差额 `vram − (已分配 + reserve)`，`reserve = AILA_Q4E_VRAM_RESERVE_MB（默认 2560）+ session state`，下限 1024 MiB（691-713）；
  - staging 深度 `AILA_Q4E_STAGING_SLOTS`（默认 16，clamp 1..64）与槽高 `slot_bytes=3*H*I*33/64`（743-746）；常驻槽 `min(AILA_Q4E_MAX_GPU_SLOTS(默认2800), expert_cache_bytes/slot_bytes − 2*staging_slots)`（751-754）；
  - hotset profile/export 接线（759-820）；`adaptive_cache_.Allocate` + `InitializeResidency` + `host_arena_.Allocate` + `LoadFromModelWeights` + `FinishInitialLoad`（821-838）；
  - cold/resident 模式（841-861）；专用 `copy_q_` 拷贝队列（832-833）。
- 执行分派：`count>4` → `RunAdaptivePrefill`，否则 `RunResidentDecode`（2298-2301）；即 **prefill chunk 走 4-token quad 族，decode 走单行/多行 GEMV 族**。

### 1.3 adaptive 专家缓存链路

**Prefill（`RunAdaptivePrefill`，`Qwen4ExpBnb4Backend.cpp:1417-1468`）**
1. 观测路由（1419；exporter 1420），统计每专家使用次数与 unique 集合（1421-1438）。
2. 常驻命中批先跑：`RunGroupedExperts(...,0,hits,...)` + `RecordResidentConsumer`（1444-1448）。
3. 冷专家按 staging 批上传：`StageColdExpertsAsync(...)`，每层/窗口每 key 只上传一次，ping-pong 交替（1451-1461）；`RecordStagingConsumer`（1459）。
4. 全部 rank 完成后 `moe_ordered_accumulate_down_rows` 合并进 y（1464-1467）。

**RunGroupedExperts（1470-1575）**：把本 microbatch 的 unique 专家按可用宽度分段——8token（默认关）、4token quad、2token、单行（1477-1494）；quad 段由 dispatcher 决定 plain 还是 wgshare（见 §1.7）；gate/up 与 down 各一次专家段调用（1548-1555）；`AILA_Q4E_GROUPED_XMX_MIN_ROWS>0` 且设备是 A770 时另有 grouped XMX tile 路径（391、1439-1441、1557-1566，默认 0 关闭）。

**Decode（`RunResidentDecode` adaptive 分支，1153-1339）**：每 token 在 host 侧分流 hit/miss（1248-1260）；miss 先可选交给 CPU（`StartArenaCold`，1268-1270），其余进 GPU staging（`StageColdExpertsAsync`，1272-1279）；resident/staging 各跑 `moe_decode_gate_up_swiglu_grouped` + `moe_decode_down_to_rank_f32`（1288-1321）；最后由 `moe_ordered_accumulate_down_rows` / `moe_ordered_accumulate_cpu_host_rows` 写回（1330-1333）。

### 1.4 Qwen4ExpHostArena（原生大页 RAM 补集）

- 槽几何（`src/models/Qwen4ExpHostArena.cpp:54-64`）：`gu_packed=2I*H/2` + `gu_absmax=2I*H/64`(1B/码) + `dn_packed=H*I/2` + `dn_absmax=H*I/64` → `slot_bytes`；H2560/I640 时 = **2,534,400 B = 2.416992 MiB**（[plan](qwen38-a770-agent-performance-plan-2026-10-12.md):58）。
- chunk 分配：每 chunk `min(chunk_bytes, 256 MiB)/slot_bytes` 个专家，循环走 `ctx_->alloc_large_host`（`AILA_Q4E_HOST_ARENA_PINNED=1` 时改用 pinned，默认 0，73-84）；分配失败**绕过 deferred log 直接写 stderr** 并带 chunks/allocated/bytes/resident_experts/host_avail/host_total/pinned（98-108，P0 落地）。
- 加载：`LoadLayerDirect` 逐层把 4 个 view memcpy 进 RAM 槽（GPU owner 的经 `load_staging_` 中转，162-223）；`load_staging_` 为单槽 host 缓冲（121），`DmaTransferToGpu` 用它做 RAM→GPU memcpy（152-160）。
- 所有权交换：`ExchangeOwner(layer, incoming, outgoing)` 只改专家指针表（245-251）；host 端 slot 描述符由 `GetExpertSlotHost`（253-269）产出，nested blocksize 固定 256。

### 1.5 专家来源三级

| 来源 | 载体 | 证据 |
|---|---|---|
| resident | GPU 常驻槽（`num_gpu_slots`，槽表 + 反向表）| `src/models/Qwen4ExpAdaptiveCache.hpp:97-101,148-200`；初始化 `InitializeResidency`（204-206） |
| staging | GPU 双 ping-pong bank，每 bank `staging_slots_per_buffer` 槽（默认 16，上限 64）| `AdaptiveCache.hpp:28,32,141,326-331`；`StageColdExpertsAsync`（hpp:295，cpp:594-637）；并行 staging memcpy 由 `AILA_Q4E_STAGE_COPY_PARALLEL` 控制（cpp:616） |
| cold host | 大页 RAM 补集 → CPU 专家计算（`Qwen4ExpExpertExecutor`）；或经 staging 转 GPU | `Qwen4ExpExpertExecutor.hpp:193-201`（`StartArenaCold`/`JoinResidentCold`）；Backend.cpp:1268-1270、1323-1325 |

量级（历史实测，非本文新测）：decode ~240.5 CPU 任务/token、~0.65 GB/token 逻辑读（plan:56-58；[roadmap](qwen38-optimization-roadmap-2026-10-10.md):70-71）；14K prefill 每 4096 块上传 ~19.3k-19.6k 冷 unique 专家、约 46 GiB/块（[H4 报告](qwen38-h4-residency-screen-2026-10-09.md):46）。

### 1.6 hotset profile（Qwen4ExpHotsetPolicy）

- identity：`MakeIdentity`（`src/models/Qwen4ExpHotsetPolicy.hpp:28-47`）对每层 gate_up/down 的 packed/absmax/quant_map/nested_absmax/nested_quant_map 取**头尾各 ≤4096 B 采样**做 FNV-64，拼成 `nf4-original-expert-order-<L>x<E>-<H>x<I>-b64-nested1-sample-fnv64-<hash>`——是身份守卫，不是全模型密码学校验（注释 25-27、48-49 同源）。
- `Load`（145-165）：校验 magic/version/identity/`n==layers*experts`/key 唯一且范围合法/无尾随数据；`Save`（135-144）：按 decode desc → prefill desc → key asc 排序输出全量 key + 三列计数，头 `AILA_Q4E_HOTSET 1`。
- backend 接线：`AILA_Q4E_HOTSET_PROFILE` 决定初始常驻（759-760、784-804），截取 `min(num_gpu_slots, ranking.size())` 项进 `InitializeResidency`（793-799）；`AILA_Q4E_HOTSET_EXPORT` 仅在 adaptive 生效，在 decode 两处（1208-1209、1263-1264）与 prefill 一处（1419-1420）观测，teardown `Save` 全量 24576 key（808-819；P3 落地见 plan §8-P3）。
- 现有 profile：`perf/qwen4exp/profiles/qwen38-nf4-h4-v1.profile`（H4，头 `48 512 24576`，identity `...fnv64-14151072976976870495`）与 `qwen38-nf4-agent6k-v1.profile`（4f09e87 校准，同 identity）。
- 老 cache 的等价入口仍在源码但生产不可达：`Qwen4ExpExpertCache::ConfigureHotset`（`src/models/Qwen4ExpExpertCache.cpp:1475-1508`，含 `AILA_Q4E_HOTSET_SLOTS/INTERVAL/SWAPS/RING/...`）。

### 1.7 MoE 内核族与现役选择

- 声明族见 `src/ops/MoeExpertOps.hpp`：2token（416/433）、4token（451/468）、wgshare gate_up（485-503，`share_quads ∈ {2,4,8}`）、`_nounpack`/`_slm`/`_nolut` 诊断（506-569）、wgshare down（572-587）、8token（591-608）、`moe_ordered_accumulate_down_rows`（634）、`moe_ordered_accumulate_cpu_host_rows`（649）。
- 选路（`src/ops/MoeExpertOps.cpp`）：env 只读一次 `AILA_Q4E_4TOKEN_WGSHARE`，仅 2/4/8 生效、其余回退 plain 4token（5652-5658）；gate/up quad 段 5727-5756、down quad 段 5874-5902 各自分派；kernel 实体在 2417（wgshare gate/up）与 3849（wgshare down）。
- **现役选择**：库默认 `AILA_Q4E_4TOKEN_WGSHARE=0`（plain 4token）；生产 launcher 强制 `=4`（`tools/server/start-qwen38.ps1:29-34`）。8token 默认关（`AILA_Q4E_8TOKEN_REUSE=false`，Backend.cpp:1476）；4token/2token 复用默认开（1474-1475）。decode 用 `moe_decode_gate_up_swiglu_grouped`（hpp:195 声明；调用 1288、1309）+ down-to-rank + ordered accumulate（6027、6085 定义）。

---

## 2. 开关与生产默认

### 2.1 AILA_Q4E_*（当前源码仍在用）

| 变量 | 默认（代码） | 作用 | 出处 |
|---|---|---|---|
| `AILA_Q4E_EXPERT_CACHE_MB` | 0=自动 | >0 时覆盖专家缓存字节预算 | `Qwen4ExpBnb4Backend.cpp:691-693` |
| `AILA_Q4E_VRAM_RESERVE_MB` | **代码 2560**；生产 3584 | reserve=该值+session state | 709；`start-qwen38.ps1:22`；`tmp/q4e-resident-speed-20261009/run.py:25` |
| `AILA_Q4E_STAGING_SLOTS` | 16（1..64） | ping-pong staging 深度/缓冲 | 743-745；`AdaptiveCache.hpp:28` |
| `AILA_Q4E_MAX_GPU_SLOTS` | 2800 | 常驻槽硬上限（P2 后可抬） | 751-754 |
| `AILA_Q4E_HOTSET_PROFILE` | "" | 初始驻留排名（identity 校验） | 759-760、784-804 |
| `AILA_Q4E_HOTSET_EXPORT` | "" | 仅 adaptive；teardown 导出全量 profile | 761-762、808-819 |
| `AILA_Q4E_RESIDENT_DECODE` | 0 | 旧 cache 常驻 decode 模式 0/1/2 | 841、846-851 |
| `AILA_Q4E_COLD_MODE` | 0（有 profile 且为 0 时→1） | 0 动态 / 1 冻结+GPU cold / 2 冻结+CPU cold | 842-846 |
| `AILA_Q4E_RESIDENT_PREFILL` | false | 冻结 prefill 驻留 | 844 |
| `AILA_Q4E_CPU_EXPERT_DISPATCH` | false | 与 resident decode 互斥 | 848-851 |
| `AILA_Q4E_GROUPED_XMX_MIN_ROWS` | 0 | >0 才走 grouped XMX tile | 1439-1441 |
| `AILA_Q4E_WEIGHT_REUSE` / `4TOKEN_REUSE` / `8TOKEN_REUSE` | true / true / false | 宽度复用段开关 | 1474-1476 |
| `AILA_Q4E_4TOKEN_WGSHARE` | **库 0**；生产 4 | quad 段 WG-share 2/4/8 | `MoeExpertOps.cpp:5652-5658`；`start-qwen38.ps1:32-34` |
| `AILA_Q4E_GDN_CHUNK_MODE` | 0（0..2） | chunkwise DeltaNet，opt-in | 393；`Model.cpp:773` |
| `AILA_Q4E_LM_HEAD_Q8` | false | 会禁用 MTP | `Model.cpp:774` |
| `AILA_Q4E_SERVER_MTP` | false | MTP 开关（加载时读一次） | `Model.cpp:698-699,772-774`；`start-qwen38.ps1:19` 显式 0 |
| `AILA_Q4E_ADAPTIVE_SWAP_FREQ` | 0（read_int_raw，0 有效） | 0=完全禁用 swap | 2347；`EnvUtils.hpp:135` |
| `AILA_Q4E_MAX_SWAPS` | 2 | 每 token swap 上限 | 2348 |
| `AILA_Q4E_STAGE_COPY_PARALLEL` | false；生产 1 | 并行 staging memcpy | `AdaptiveCache.cpp:616`；`start-qwen38.ps1:26-28` |
| `AILA_Q4E_HOST_ARENA_PINNED` | false | pinned 替代大页（已否证，见 §3.3） | `HostArena.cpp:73-84` |
| `AILA_Q4E_PREFILL_ATTN_TILE` | 0（0/128/256/512） | attention tile 上限 | 387-390 |
| `AILA_Q4E_PREFILL_MOE_ROWS` / `PRED_IDS_ONLY` | false / true | 旧 batched-rows 仅日志分支；router ids-only | 904-912 |
| `AILA_Q4E_PREFILL_XMX` / `FUSED_PREFILL` / `DIRECT_PREFILL_IO` | false / false / true | 稠密 prefill XMX 与直接 IO | 450-457、907-909 |
| `AILA_Q4E_INLINE_CHECKPOINT` | "auto-30/26" | 中间 checkpoint | 923 |
| `AILA_Q4E_PHASE_SCRATCH` | false | 阶段 scratch 复用 | 921、939 |
| `AILA_Q4E_ASYNC_DEQUANT` | true | bnb linear 异步解量化 | 433-448 |
| `AILA_Q4E_DENSE_GEMV` | true | count==1 原生 GEMV | 1600-1605 |
| `AILA_Q4E_GPU_Q8_ADAPTIVE`（W4A8） | false | decode 激活 Q8 变体，opt-in | `Qwen4ExpExpertExecutor.cpp:684` |
| `AILA_Q4E_NOUNPACK_PROBE` / `NOLUT_PROBE` / `SLM_REUSE` | false | 仅 KernelTests 诊断臂，生产零引用 | `tests/ops/Qwen4ExpKernelTests.cpp:4166,4220,4270`；[P4](qwen38-p4-slmshare-2026-10-13.md):61-63 |
| `AILA_Q4E_BENCH_XLOCALITY` | false | 基准追加 x 局部性对照 | `Qwen4ExpKernelTests.cpp:4124` |
| `AILA_Q4E_WORK_SUMMARY` / `PROFILE_COMPONENTS` / `AILA_PROFILE_Q4E_DECODE` | false | 诊断/profiling | 1930、1742、1929 |
| `AILA_Q4E_VERIFY_SMALL_ROWS` | false | 2..4 行 verify 走单行 decode 核 | `Qwen4ExpBnb4Backend.hpp:468-473`；Backend.cpp:1890 |
| `AILA_Q4E_ARENA_VERIFY_ROWS` / `VERIFY_CPU_ROWS` | false / 0 | arena 多行验证与 CPU rows | `Qwen4ExpExpertExecutor.cpp:1263`；Backend.cpp:2066 |
| `AILA_Q4E_MTP_BOUNDARY_DUMP` / `WINDOW_ACCOUNT_LOG` | false / 0 | MTP 边界/窗口记账探针 | `Model.cpp:271-274`；`Request.cpp:176` |

注意：`aila::env::read_int` 对 **≤0 的值回退默认值**（`src/utils/EnvUtils.hpp:72-89`），需要表达 0 的开关必须用 `read_int_raw`（`EnvUtils.hpp:135`）——历史上 `ADAPTIVE_SWAP_FREQ=0` 曾被误读成默认 1（[swap 频率屏幕](qwen38-swap-frequency-screen-2026-10-10.md):3）。

### 2.2 AILA_SERVER_* 与 KV/int8 相关

| 变量 | 默认 | 作用 | 出处 |
|---|---|---|---|
| `AILA_SERVER_KV_TYPE` | "bf16" | server 侧 KV 类型；int8 仅 qwen4exp_nf4 | `src/server/ServerMain.cpp:107-113`（用法说明 39 行） |
| `AILA_SERVER_PREFIX_DIAG` | off | 只读 prefix-reuse 诊断（TakeSession/KeepSession/releaseInference WARN 日志） | `src/server/HttpServer.cpp:33-42`（7d361f2） |
| `AILA_SERVER_WORKER_LOG` | "" | 捕获 worker 日志到文件（诊断） | `HttpServer.cpp:53-61` |
| `AILA_SERVER_TOOL_DIAGNOSTICS` | 0 | 工具解析失败时输出诊断块 | `src/server/ChatCodec.cpp:432-435` |
| `AILA_KV_QUANT` / `AILA_KV_QUANT_FORMAT` | 0 / 未设=int8 语义由格式函数判定 | worker 侧 int8 KV 前提 | `Model.cpp:702`；`EnvUtils.hpp:108-113` |
| `AILA_KV_QUANT_VEC` / `AILA_KV_QUANT_DPAS` | true / false | int8 读取向量化 / attention DPAS | `src/ops/AttentionOps.cpp:306,311` |

生产 int8 用法：worker `AILA_KV_QUANT=1 AILA_KV_QUANT_FORMAT=int8` + server `--kv-type int8`/`AILA_SERVER_KV_TYPE=int8`（`src/server/HttpServer.hpp:21`；[int8kv-prod](qwen38-int8kv-prod-2026-10-12.md):8-13）。

### 2.3 生产 launcher 固定值（`tools/server/start-qwen38.ps1`）

加载 `docs/plans/qwen38-server-mtp-2026-10-07.json.worker_environment`（12-16）后显式覆盖：`Model=C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2`（3）、`Port=8080`（4）、`Context=32768`（5）、**`PrefillChunk=4096`**（6）、`MaxNewTokens=2048`（7）、`AILA_Q4E_SERVER_MTP=0`（19）、**`AILA_Q4E_VRAM_RESERVE_MB=3584`**（22）、**`AILA_Q4E_HOTSET_PROFILE=perf/qwen4exp/profiles/qwen38-nf4-h4-v1.profile`**（23）、`AILA_Q4E_STAGE_COPY_PARALLEL=1`（26-28）、**`AILA_Q4E_4TOKEN_WGSHARE=4`**（32-34）；server 参数 `--sessions 1 --decode-batch-size 1 --reuse-sessions 1 --request-timeout-ms 600000`（37-40）。adaptive swap 默认 0、GDN chunk/GPU W4A8/MTP 关、BF16 KV（[plan](qwen38-a770-agent-performance-plan-2026-10-12.md):119 同口径）。

正式 runner 叠加同值：`MTP=0 / RESERVE=3584 / CHUNK=4096`（`tmp/q4e-resident-speed-20261009/run.py:25`，`tmp/q4e-agent-plan-audit-20261012/run_val14k.py:17`、`run_calib6k.py:17`）。profile JSON 自身的 `AILA_Q4E_VRAM_RESERVE_MB=3008` 与代码默认 2560 只在无覆盖时才有意义（roadmap:111-113 复核）。

---

## 3. 已验证结论表

采纳/拒绝线：内核类以逐位一致优先；端到端采纳线 **≥+5%**（roadmap:132；[w4a8-prefill](qwen38-w4a8-prefill-bound-2026-10-12.md):9）；噪声底：14K ttft ±0.6%（roadmap:4-5）。

### 3.1 采纳（生产默认已生效）

| 结论 | 关键数字 | commit | 文档 |
|---|---|---|---|
| nested-256 scale 固定布局 | 14K 完整 75.295→73.486 s（−2.40%）、首字 −2.60%；反序复核复现 | `98cc037` | [nested-div256](qwen38-nested-div256-screen-2026-10-09.md):9 |
| H4 初始驻留（profile→常驻槽） | 续轮首字 −14.85%、续轮总耗时 −11.32%；14K 仅 −0.32% | `c5d5805` | [h4](qwen38-h4-residency-screen-2026-10-09.md):24-31 |
| staging 并行 memcpy | 续轮总 −9.18%/−10.10%、首字中位 −23.22%；copy-only 3.64× | `3962519` | [staging](qwen38-staging-copy-screen-2026-10-09.md):13-23 |
| 同步 swap 默认关闭 | 14K decode **+8.03%**（13.732→14.835） | `d7b0649` | roadmap:22 |
| WG-share=4 | 14K ttft −2.94%（位精确） | `e808f63` | roadmap:11；[status](qwen38-roadmap-status-2026-10-12.md):11 |
| P1 tokenizer 修复（`#` merges + `\s+(?!\S)`） | 续轮 cached=14072、首字 71.0→1.486 s；离线逐 id EQUAL | `c1e9e60`+`2c51cc5` | [P1](qwen38-p1-reencode-divergence-2026-10-13.md):142-173 |
| P1 回归测试入库 | `AilaTokenizerRegressionTests`；cpu\|abi 20/20 | `3f81de0` | [status](qwen38-roadmap-status-2026-10-12.md):86 |
| P2 预算修正（单源 KV 字节 + 实际 staging 预留） | INT8 slots **2792→2948（+156）**；ttft −0.8%、续轮 +0.8~+1.7% | `548b0d8` | plan:259-277；日志 `tmp/q4e-resident-speed-20261009/p2-int8-worker.log:7-8` |
| P0 arena 失败日志（仅日志，未改分配策略） | 失败路径直写 stderr + error string | `548b0d8` | plan:254-257 |

### 3.2 opt-in（已实现/已验证，未达默认化门槛）

| 结论 | 关键数字 | commit | 文档 |
|---|---|---|---|
| INT8 KV 全链路 | 内核 14336 长度 1.854→1.053 ms（bf16 1.583，快 33%）；生产 +1.8% decode / −1.1% ttft / 续轮噪声内 → **低于 +5% 线** | `6deaefc`→`8101788`→`c67edc7` | [int8kv-vec](qwen38-int8kv-vec-2026-10-12.md):18-30；[int8kv-prod](qwen38-int8kv-prod-2026-10-12.md):18-27 |
| W4A8 decode（q8 激活） | 内核 T=1 慢 15.5%、e2e +1.61%；输出逐位一致 | `5ef4e1a` | roadmap:58-59 |
| GDN chunk mode 1 | 14K ttft −2.7%（68.719/68.720 双跑复现）；decode −4~7%（路由漂移） | — | roadmap:31-37 |
| MTP | 等价门未过（首分歧=被拒 verify 窗口 row0，pos=1492）；吞吐 15.0-15.3 vs T1 15.6 | `40d6bf6` 等 | [mtp-align](qwen38-mtp-align-2026-10-11.md):80-96 |
| agent6k hotset profile | CPU misses −31%、experts_host −19%、submitted p50 −3.5%（同槽数 A/B/B/A） | `4f09e87` | plan:335-361 |
| MTP 对齐开关族（VERIFY_SMALL_ROWS 等 5+3 项） | prompt 512/1182 × width 2/4 全 48 层逐位 | 见 doc | [mtp-align](qwen38-mtp-align-2026-10-11.md):40-43,61 |

### 3.3 NO-GO（已实测否决，勿重做）

| 方向 | 关键数字 | commit | 文档 |
|---|---|---|---|
| P4 SLM 解量化复用（share=4） | `_slm` −11.1%（t16）/−17.7%（t64）vs 门 +10%；`_nolut` 上限 13.6%/33.1% | `53aa56d` | [P4](qwen38-p4-slmshare-2026-10-13.md):20-45 |
| prefill W4A8 | x 全 L1 vs 生产流式仅差 ~3% → e2e 上界 ≤0.7% | — | [w4a8-prefill](qwen38-w4a8-prefill-bound-2026-10-12.md):6-24 |
| ESIMD 替换 SG8 DPAS | 真实网格 8 形状 0.67-1.01×（含 gate/up 0.65×）；微基准 GO 不迁移 | — | `tmp/q4e-esimd-dpas-20261012/performance.md:154`；plan:203 |
| INT8 attention DPAS | 14k wall 1.922-1.931 vs 向量 1.043 ms（慢 ~84.6%） | — | [int8kv-vec](qwen38-int8kv-vec-2026-10-12.md):45-56 |
| 8-token 窗口 / share=8 | 寄存器 spill / L1 工作集上限 | — | roadmap:14；status:11 |
| swap 预算 2→8；swap 频率轴 | decode −10~13%；边际 ≈1.8 ms/swap；FREQ=0 臂曾因 read_int 误读无效 | — | [swap-budget](qwen38-swap-budget-screen-2026-10-10.md):24-26；[swap-freq](qwen38-swap-frequency-screen-2026-10-10.md):3,29 |
| chunk 8192 / staging 64 / ring 4 | 83.89（2048）/70.6（4096）/89.22（8192）非单调；stg64 无益且挤占常驻槽 | — | roadmap:104-110 |
| pinned 替代大页 | 51 GiB 锁页首请求 `UR_OUT_OF_RESOURCES` | — | roadmap:98-101 |
| half-unpack / 字节对 LUT | 生产/小核回退；字节对 LUT 回退 9-16% | — | plan:113-115 |
| CPU 指令变体（BYTE_SHUFFLE+SCALE_REGISTER 等） | 生产两轮 +0.6%/−0.6%（中性） | — | roadmap:23 |
| 全量 pinned / BF16 / INT8 权重副本 | 破坏 16G/64G 容量优势 | — | plan:210 |

---

## 4. 测试与基准设施

### 4.1 CPU/ABI 套件（现 20 项）

`ctest --test-dir build-refactor -N -L 'cpu|abi'` 本次列出 **20 项**（Test #2..#69），含 3f81de0 新增的 `AilaTokenizerRegressionTests`（#50）：AilaRequestStateTests、AilaSchedulerTests、AilaSessionBatchTests、AilaSessionMemoryTests、AilaDeferredLoggingTests、AilaTokenTraceComparisonTests、AilaQwen4ExpModelSpecTests、AilaQwen4ExpExpertReaderTests、AilaTokenizerRegressionTests、AilaInferenceProtocolTests、AilaInferenceWorkerTests、AilaRuntimeFaultTests、AilaExecutionTraceTests、AilaInferenceClientTests、AilaInferenceCHeaderTests、AilaInferenceCApiGpuTests、AilaInferenceCApiTests、AilaLegacyTextApiTests、AilaLegacyTextSessionTests、AilaInferencePipeTests。
历史门禁：`19/19`（status:73-74）→ `20/20`（status:86；plan:329）。

### 4.2 Server 套件

- `tests/server/socket_tests.py`（86 KiB；fake 模式用 `AilaFakeInferenceWorker.exe`，gpu 模式用 `AilaWorker.exe`，AGENTS.md §5）。
- `tests/server/batching_socket_tests.py`（fake 并发）、`tests/server/batching_gpu_tests.py`（`--server/--worker/--model/--backend/--output`，:415-419）。
- 另有 `memory_socket_tests.py`、`tool_socket_tests.py`。

### 4.3 标准生产序列（runner 纪律）

`tmp/q4e-resident-speed-20261009/run.py`：清理 AILA_/SYCL_/UR_L0_ 环境 → 加载 profile JSON `worker_environment` → 覆盖 MTP=0/reserve3584/chunk4096 → `python tools/server/qwen38_context_probe.py --context 32768 --chunk 4096 --generate 8|32 --reuse 1 --fixture tmp/q4e-agent-corpus-v1 --prompt-tokens 14000`（:11-36）；非诊断加 `--production-timing`，`--agent` 追加 PI 6K 回合 + cancel-recover（:37-40）；2 s 采样 available RAM、1200 s 超时、拒绝覆盖既有 label 目录（:6-8、44-61）；产物 `*-run.log`/`*-monitor.json`（:62-65）。
同构变体：`tmp/q4e-agent-plan-audit-20261012/run_val14k.py`（generate 128）、`run_calib6k.py`（generate 256、6K prompt，用于 P3 校准）。

### 4.4 KernelTests opt-in 探针

`build-refactor/AilaQwen4ExpKernelTests.exe`：`--prefill-bench-only`（`tests/ops/Qwen4ExpKernelTests.cpp:4676`）、`--qsa-prefill-only`（4686）、`--kv-fp8-only`（4649）；env 门控诊断臂 `AILA_Q4E_NOUNPACK_PROBE`（4166）/`NOLUT_PROBE`（4220）/`SLM_REUSE`（4270）/`BENCH_XLOCALITY`（4124）。P4 三内核声明为 "timing diagnostic only, never used for real inference"，生产零引用（[P4](qwen38-p4-slmshare-2026-10-13.md):61-63）。对齐探针：`tests/models/Qwen4ExpToleranceTests.cpp:719` `--mtp-align`，`AILA_Q4E_MTP_ALIGN_T1` 默认 "verify1"（`tests/models/Qwen4ExpMtpAlign.hpp:12`）。

### 4.5 速率口径与 pair/identity 纪律

- 速率口径：`decode_rate = (completion_tokens-1)/(duration_seconds-first_content_seconds)`（`tools/server/qwen38_context_probe.py:78-79`；plan:99-107 修正说明）。旧表用 32 当分子偏高 ~3%。
- 每轮启动实测 exe SHA256 并核对 pair record（h4:20；nested:68）；pair.json 记 worker/server 的 path/sha256/build_id。
- `retainedN` 归档在 `tmp/q4e-mtp-alignment-20261011/`（现至 retained17）；当前两个：`retained16/pair.json`（build `2c51cc5e…`，worker `7bad7286…` / server `916d4b56…`）、`retained17/pair.json`（build `3f81de01…`，worker `3e6b976d…` / server `2490bdc8…`）。

---

## 5. 已知缺口与下一步候选

1. **layer-major prefill 原型（未开工，最大的结构杠杆）**：现 chunk-major（每 4096 块全 48 层重传冷专家）；改为 layer-major 后 H2D 310 GB→~62 GB 为推算，ttft 预计 −13~−18%，但 8192 大窗口实验已证明 bank_wait/router_wait/逐层 host 串行会吃掉流量收益（roadmap:39-41、104-110）。工程量：prefill 驱动、调度/取消、状态机重构，多日（plan:215）。
2. **4token share=0 vs wgshare 现状**：库默认仍 0；生产永远 4。历史生产收益 −2.94% ttft（`e808f63`），但当前审计微测 quads=160 时 share0/4 = 2.20065/2.20056 ms（中性，plan:71）——小形状不能反证生产，也提示该收益点是形状敏感的。同族候选（二维分工 8×2/4×4、grouped GEMM M640×N）未做（roadmap:16-19）。
3. **P2 尾项**：计划 §5-P2 第 4 条要求"另起诊断轮用 `--memory-sample-ms` + DXGI 监测峰值，覆盖 4096 块/短追加/取消恢复；若余量充分再做 reserve 3584→3328 单变量"（plan:166）。执行记录 §8 只覆盖第 1-3 条（plan:259-277），本文未找到该尾项结果 `[未核验是否已另行执行]`。
4. **MTP 闭环选项**：(a) env 化 live stage 探针定位 pos=1491 行算子；(b) verify 窗口退化为逐行 count==1（构造逐位等价、放弃多行收益）。两者均要求先于 B3 之后重评（mtp-align:93-95；roadmap:134）。
5. **agent6k profile 默认化路径**：先 opt-in；扩大真实代码/工具语料校准并再跑 held-out，再决定是否替换 launcher 的 H4 默认（plan:360-361）。
6. **swap 重评条件**：仅当出现 ≥1024 token 长生成证据时重启"H4 重排 + 低频 swap"（roadmap:24）。
7. **INT8 KV 默认化**：结构收益已兑现（+156 槽）但 e2e ~1%，维持 opt-in；默认化需要更大证据（int8kv-prod:26-27；plan:170）。
8. **GDN chunk 若默认化**：需先按工作负载混合比与路由稳定性评估（roadmap:36-37）。P2 收尾的 reserve 单变量实验（见 3）也是其前置同类问题。

---

## 6. 当前状态快照（2026-10-13）

- HEAD：`3f81de01e5761718c102a9a1071ee89e298c317f`（"test(tokenizer): lock the P1 merge/pretokenize regressions and record P4/P1 results"）；前一串关键提交：`2c51cc5`（pretokenize 修复）、`c1e9e60`（merges 修复）、`53aa56d`（P4 探针）、`7d361f2`（PREFIX_DIAG）、`4f09e87`（hotset exporter）。
- pair `retained16`（build `2c51cc5e9b99601f35e4a5795a104e9cd14ea738`，P1 live 验证用）：worker `7bad72864336c86dcb2383984512c2461f6911fedaecf2a7775a04f2510fa5b9` / server `916d4b56171104ae32478cecafb189a23de300c9dc6d48c4338abacef04f4c99`（`tmp/q4e-mtp-alignment-20261011/retained16/pair.json`）。
- pair `retained17`（build `3f81de01e5761718c102a9a1071ee89e298c317f`，测试/文档提交后重建）：worker `3e6b976db8b31b57b2b9ad17c975f42dd17d8ec2ce2a00b33867adcbb7c27417` / server `2490bdc8d2b47f3f25861a7042ec910dd4293e9f3e12558d7e9c24efb7b41055`（`retained17/pair.json`）。
- 工作树：无已跟踪文件改动；未跟踪 = `docs/plans/` 22 个文件（含本文件、各 screen/status 报告与 JSON）+ 仓库根 `p1_tokenizer_probe.obj`；`tmp/` 被 `.gitignore:15` 忽略。注意用户所述"只有 docs/tmp"严格说漏了根目录的 `p1_tokenizer_probe.obj`（P1 探针产物）。
- P1 live 现场（retained16）：`tmp/q4e-agent-plan-audit-20261012/p1-g128-pretok-20261013/` — `[PfxDiag] take slot=0 cached_len=14072 prompt_len=14096 diverge_at=14072 qualifying=1`（stderr.log:15）、续轮 `first_content_seconds=1.4855`、`cached_prompt_tokens=14072`（run.log:4）。

---

## 7. 未核验清单

- `[未核验]` P2 尾项（§5-3）的 DXGI/USM 峰值实测与 reserve 3584→3328 实验是否已在别处执行并记录。
- `[未核验]` ESIMD 审计（`qwen38-esimd-integration-audit-2026-10-12.md`）未通读；本文 ESIMD 结论只引用 plan:203 与 `tmp/q4e-esimd-dpas-20261012/performance.md:154`。
- `[未核验]` 老 cache 系环境变量（`AILA_Q4E_STAGING_MB`=128、`AILA_Q4E_EXPERT_CHUNK_MB`=1024、`AILA_Q4E_FILL_*`、`AILA_Q4E_HOT_RAM_MB` 等）现状：源码仍在（`Qwen4ExpExpertCache.cpp:726,744` 等）但仅测试/死分支可达，未逐一确认无其他调用方。
- `[未核验]` 多会话/并发路径（sessions>1、decode-batch>1）在 Qwen4Exp adaptive 管线的行为：本文所有数字均为 `--sessions 1 --decode-batch-size 1`。
- `[未核验同源]` §1.5 的 240.5 任务/token、0.65 GB/token 为历史诊断口径，本次未复测。
- `[未核验]` SSD/PCIe 实际协商速率与带宽（plan:19 标明未测）。
