# 重构验证入口与 P0 基线

核对日期：2026-09-08。本文记录当前源码的实际入口，区分已执行结果与待补门禁。第 6 节保留 P1/P2 首批结果，第 8 节保留 Qwen3 LayerPlan、Scheduler 与 HTTP 探针结果，第 9 节记录 Hybrid 共享会话及分组状态竞争修复，第 10–11 节保留 Qwen3 IPC/HTTP 首片，第 12 节记录 Hybrid 远端与服务接入。GPU 验证由集成负责人串行执行；子代理仅执行明确分配的 CPU 验证。

## 1. 固定输入与构建

| 项目 | 值 |
|---|---|
| 基线源码 | `1956de9ed445accc33604b6e98ae3e3d7fddc813` |
| 不可变源码目录 | `E:\RiderProjects\Aila\.worktrees\refactor-baseline-1956de9` |
| 基线构建目录 | `E:\RiderProjects\Aila\.worktrees\refactor-baseline-1956de9\build` |
| 实现目录 | `E:\RiderProjects\Aila`，分支 `codex/shared-model-sessions` |
| 候选构建目录 | `E:\RiderProjects\Aila\build-refactor` |
| 编译配置 | Release，Ninja，`build.ps1` 的 `oneapi-2026.1` |
| 已记录工具链版本 | compiler 2026.1.0，oneDNN 3.11.2，TBB 2023.1，UMF 1.1，`sycl9.dll`；来自基线 `build/build_info.json` |
| GPU / 驱动 | Intel Arc A770，`32.0.101.8974`（驱动日期 2026-08-11） |
| 模型最大上下文 | 以下最小 smoke/bench 固定 4096，BF16 KV（`AILA_KV_QUANT=0`） |

模型来自主工作区，worktree 不需要另拷权重：

| 模型 | 绝对目录 | 形状 | `model.safetensors` 字节数 |
|---|---|---|---:|
| Qwen3-0.6B NF4 | `E:\RiderProjects\Aila\models\qwen3-0.6B-bnb-nf4-offline` | 28 层，hidden 1024，16 Q / 8 KV heads，head_dim 128 | 538883408 |
| Qwen3.5-0.8B NF4 | `E:\RiderProjects\Aila\models\qwen3.5-0.8B-bnb-nf4-offline` | 24 层，hidden 1024，18 DeltaNet / 6 Full Attention | 766662032 |

两份 `config.json` 当前 SHA256：

- Qwen3：`107E29CC28C78CEFF2259BCD318C3694B5EF31EACC95B67E459CAE2AA4EC0B4E`
- Qwen3.5：`FDAA22884228DBDD50582A1223EB08417BE72DDFA4A14A3396D7B4FEE9D46CEB`

这两个配置哈希不替代完整权重身份。本次 `perf/refactor-legacy-check.ps1` 记录了 Qwen3 的模型文件、输入、二进制 SHA256 和实际调用参数，见 `tmp/refactor/legacy-parity/report.json`。现有 `regress.ps1` 也支持递归权重 manifest，但其 metadataNames 不包含独立 `chat_template.jinja`，使用该入口时需要额外记录。

固定文本输入：

- `E:\RiderProjects\Aila\perf\fixtures\smoke_text_qa.json`：system + “请简短回答：1+1等于几？ /no_think”。SHA256 `20288E77651FA139DD033949CCD8BFD72BF433E495867456FD28D1907504F8A7`。
- `E:\RiderProjects\Aila\perf\fixtures\smoke_multi_turn.json`：预置 Alice 历史，再询问姓名。SHA256 `7E7FBAA0D3DEBFA68DB5AFC6F71656B5247FEE7B8231015B9FF9AF197F986681`。这是一次带历史的完整输入，不是同一存活 Session 的增量缓存复用测试。

测量时同时保留构建日志、可执行文件哈希、源码 diff、设备/驱动、全部影响执行的 AILA/SYCL/UR/ZE 环境设置。现有脚本的 Git 信息只有 commit/branch，不记录 dirty diff；未提交候选改动不能仅凭同一 commit 字段声称源码相同。

## 2. CPU CTest 白名单

在基线 `1956de9` 的 Windows/Ninja 配置中共注册 20 个 CTest。以下 12 项不会创建 SYCL Context 或提交 GPU 工作：

| CTest 名称 | 覆盖范围 |
|---|---|
| AilaChatTests | 模板、JSON、工具策略、输出/流解析、thinking budget |
| AilaRuntimeDllSelectionCMakeTests | DLL 选择规则，CMake 文件测试 |
| AilaYolo26ModelSpecTests | JSON 配置/manifest 校验 |
| AilaSafeTensorsTests | 空 Tensor 下的 ModelWeights 键删除规则 |
| AilaLoraLoaderTests | PEFT 权重名称映射 |
| AilaQwen35VisionWeightsTests | 视觉 patch 权重形状与释放判定 |
| AilaIpcProtocolTests | 协议帧与 Windows pipe |
| AilaBuildIdentityCMakeTests | 只读 Git、Ninja 依赖和 Worker 二进制 build ID，不重建、不改 HEAD |
| AilaWorkerDispatcherTests | fake engine 的参数转发和结果校验 |
| AilaRuntimeIsolationTests | fake worker 的进程/环境/管道生命周期 |
| AilaProxyAbiTests | fake worker 下的旧 C ABI、回调线程、重入、派生句柄生命周期 |
| AilaProxyLoggingTests | Proxy 日志/回调生命周期 |

基线命令（PowerShell）：

```powershell
$refactorBaseline = 'E:\RiderProjects\Aila\.worktrees\refactor-baseline-1956de9'
$refactorBuild = Join-Path $refactorBaseline 'build'
$refactorCpuTests = '^Aila(ChatTests|RuntimeDllSelectionCMakeTests|Yolo26ModelSpecTests|SafeTensorsTests|LoraLoaderTests|Qwen35VisionWeightsTests|IpcProtocolTests|BuildIdentityCMakeTests|WorkerDispatcherTests|RuntimeIsolationTests|ProxyAbiTests|ProxyLoggingTests)$'
ctest --test-dir $refactorBuild -C Release --output-on-failure --parallel 4 -R $refactorCpuTests --output-log (Join-Path $refactorBaseline 'baseline-cpu-ctest.log')
```

候选新增四个 CPU CTest：`AilaRequestStateTests`（会话租约、请求状态/取消/独立 RNG、输入校验和生命周期）、`AilaDeferredLoggingTests`（延迟日志、重入及有界缓存）、`AilaTokenTraceComparisonTests`（完整 trace 与失败判定）和 `AilaSchedulerTests`（轮转、准入、取消、背压、故障和输出生命周期，19 组）。它们可与上面 12 项在同一个 CTest 命令中并行运行；不要同时启动两个 CTest 进程写入同一构建目录的 `Testing/Temporary/LastTest.log`。新增的 `AilaQwen3SessionTests` 和 `AilaQwen3SchedulerTests` 属于 GPU 串行队列。

另有 `AilaMimiEncoderTests` 为 CPU 卷积/Transformer/RVQ 实现，但加载真实 TTS 权重并依赖 `tests/audio/data/ref_codes_golden.json`、`tmp/icl_oracle/ref_test_wav24k.bin` 和 `ref_test.wav`。它较重且不能在缺数据的 worktree 直接复跑，因此不列入快速白名单。

其余 7 项需要纳入 GPU 串行队列：`AilaBnb4BitLinearKernelTests`、`AilaConvOpsKernelTests`、`AilaDiffusionGemmaModelSpecTests`、`AilaDiffusionGemmaFusionTests`、`AilaDiffusionGemmaVisionTests`、`AilaEnvUtilsTests`、`AilaTTSVoiceCloneAlignmentTests`。尤其 EnvUtils 会构造 Context，DiffusionGemmaModelSpec 会执行 NF4 projection / masked attention / RoPE，不能按名字归类为 CPU。

未注册为 CTest 的补充入口包括 `tests/perf/PerfCommonTests.ps1`（脚本/fake executable 自测）和 `compare_accuracy.ps1 -SelfTest`。`tests/runtime/ProxyDependencyTests.ps1 -BuildDir <dir>` 只检查发布依赖，但要求已存在 `Release/bin` staging；这三项不是本次 12 项结果的一部分。

## 3. 最快文本 smoke 与 token 采集

以下命令使用基线目录中的脚本，使结果里的源码身份来自基线。两个命令顺序运行。`smoke.ps1` 的真实入口是 `-BuildDir`，没有 `-AilaExe`。

```powershell
$refactorRepo = 'E:\RiderProjects\Aila'
$refactorBaseline = Join-Path $refactorRepo '.worktrees\refactor-baseline-1956de9'
$refactorBuild = Join-Path $refactorBaseline 'build'
$refactorTraceEnv = @{ AILA_DEBUG_TOKEN_IDS = '1'; AILA_LOG_LEVEL = 'info'; AILA_KV_QUANT = '0' }

& (Join-Path $refactorBaseline 'smoke.ps1') -BuildDir $refactorBuild -Preset phase_gate_q3_bnb4 -CaseNames q3_06_bnb_nf4_smoke -ModelDir (Join-Path $refactorRepo 'models\qwen3-0.6B-bnb-nf4-offline') -MaxSeqLen 4096 -OneApiStack oneapi-2026.1 -EnvOverrides $refactorTraceEnv -OutputDir (Join-Path $refactorBaseline 'tmp\refactor\q3-smoke')
& (Join-Path $refactorBaseline 'smoke.ps1') -BuildDir $refactorBuild -Preset phase_gate_q35_text -CaseNames q35_08_bnb_nf4_smoke -ModelDir (Join-Path $refactorRepo 'models\qwen3.5-0.8B-bnb-nf4-offline') -MaxSeqLen 4096 -OneApiStack oneapi-2026.1 -EnvOverrides $refactorTraceEnv -OutputDir (Join-Path $refactorBaseline 'tmp\refactor\q35-smoke')
```

两例都是 greedy、最多 16 generated tokens；Qwen3 expectation 是非空，Qwen3.5 还检查包含 `2` / `二` / `两`。结果在各自 `smokes.json` 和 `smoke_logs/<case>.log`。`AILA_DEBUG_TOKEN_IDS` 仅记录前 64 个生成 step，默认 Info 日志必须开启。比较时要求 step 从 0 连续、每个 id 非负、step 数量等于 `[GenerateMessages] ... Generated=N`，再比较完整 ID 数组。不能只比较解码文本或前 64 token 前缀。

如需现有 runner 自动保存完整 token provenance 和模型 manifest，可以生成只包含这两个 chat case 的临时 cases JSON，再调用基线 `regress.ps1`：

```powershell
$refactorCasesDir = Join-Path $refactorBaseline 'tmp\refactor'
New-Item -ItemType Directory -Path $refactorCasesDir -Force | Out-Null
$refactorCasesPath = Join-Path $refactorCasesDir 'text-cases.json'
@{
    schemaVersion = 1
    cases = @(
        @{ name = 'q3_06_nf4_text'; kind = 'chat'; model = (Join-Path $refactorRepo 'models\qwen3-0.6B-bnb-nf4-offline'); input = (Join-Path $refactorRepo 'perf\fixtures\smoke_text_qa.json'); maxTokens = 16; mode = 'greedy' },
        @{ name = 'q35_08_nf4_text'; kind = 'chat'; model = (Join-Path $refactorRepo 'models\qwen3.5-0.8B-bnb-nf4-offline'); input = (Join-Path $refactorRepo 'perf\fixtures\smoke_text_qa.json'); maxTokens = 16; mode = 'greedy' }
    )
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $refactorCasesPath -Encoding utf8
$refactorAccuracyEnv = @{ AILA_MAX_SEQ_LEN = '4096'; AILA_KV_QUANT = '0'; AILA_LOG_LEVEL = 'info' }
$refactorPreviousEnv = @{}
foreach ($refactorKey in $refactorAccuracyEnv.Keys) {
    $refactorPreviousEnv[$refactorKey] = [Environment]::GetEnvironmentVariable($refactorKey, 'Process')
    [Environment]::SetEnvironmentVariable($refactorKey, $refactorAccuracyEnv[$refactorKey], 'Process')
}
try {
    & (Join-Path $refactorBaseline 'regress.ps1') -BuildDir $refactorBuild -OneApiStack oneapi-2026.1 -CasesFile $refactorCasesPath -OutputDir (Join-Path $refactorBaseline 'tmp\refactor\text-accuracy') -CaseTimeoutSeconds 600
} finally {
    foreach ($refactorKey in $refactorPreviousEnv.Keys) {
        [Environment]::SetEnvironmentVariable($refactorKey, $refactorPreviousEnv[$refactorKey], 'Process')
    }
}
```

`regress.ps1` 只支持 greedy chat，不接受采样参数或 MaxSeqLen 参数。运行该入口前必须固定 `AILA_MAX_SEQ_LEN=4096`、`AILA_KV_QUANT=0` 和 Info 日志，并在外层记录/恢复原环境。它自动在子进程开启 `AILA_DEBUG_TOKEN_IDS`，生成 `accuracy.json`、`case_results`、原始 stdout/stderr 和输出文本；默认 cases 文件未包含 Qwen3 文本例，所以此处使用单独文件。候选运行应使用同一个 cases 文件，改用候选脚本/构建和候选 `tmp` 输出目录。

## 4. 最小性能记录

使用 manual 模式可直接给绝对模型目录。`bench.ps1 -Preset ...` 会从 presets 的模型别名解析路径，并忽略 `-ModelDir`、`-PromptTokens` 等手动 case 参数；worktree 没有模型目录时尤其需要避免混用。

```powershell
$refactorBenchEnv = @{ AILA_KV_QUANT = '0' }
& (Join-Path $refactorBaseline 'bench.ps1') -BuildDir $refactorBuild -ModelDir (Join-Path $refactorRepo 'models\qwen3-0.6B-bnb-nf4-offline') -PromptTokens 256 -GenTokens 128 -BenchIters 3 -WarmupIters 1 -Seed 42 -OneApiStack oneapi-2026.1 -EnvOverrides $refactorBenchEnv -OutputDir (Join-Path $refactorBaseline 'tmp\refactor\q3-bench')
& (Join-Path $refactorBaseline 'bench.ps1') -BuildDir $refactorBuild -ModelDir (Join-Path $refactorRepo 'models\qwen3.5-0.8B-bnb-nf4-offline') -PromptTokens 256 -GenTokens 128 -BenchIters 3 -WarmupIters 1 -Seed 42 -OneApiStack oneapi-2026.1 -EnvOverrides $refactorBenchEnv -OutputDir (Join-Path $refactorBaseline 'tmp\refactor\q35-bench')
```

不传 `-Sample` 时为 greedy。输入是 `src/bench/Benchmark.cpp` 固定英文段落编码后重复至指定长度。每次 decode 测量前重新 prefill，warmup 仅解码 `min(GenTokens,16)`。保存 `bench.json` 和每轮原始日志；3 轮足以建立最小样本，但不足以预先承诺某个回归百分比。

现有 `ttft_ms_avg = prefill_ms_avg + decode_ms_avg / gen_length` 是推算值，不包含模型加载、HTTP/IPC、排队、tokenization 或实际首字节观测。它不能充当服务 TTFT，也没有 p95 token latency。当前 Context 内存计数只覆盖经 Context 分配的内存，不代表进程或设备总峰值。

`bench.ps1` 有全局 benchmark mutex，但 smoke/CTest 不参与此锁，因此仍由集成负责人统一串行所有 GPU 工作。数值 trace 与性能运行分开，profile 环境不得残留。

## 5. 首批数值验证实现

基线仓库没有 Qwen3/Qwen3.5 的逐步 backend 测试，也没有固定文本模型完整 logits golden 文件。本次新增 `tests/inference/Qwen3SessionTests.cpp`，在真实 Qwen3 NF4 模型上比较独立与交错执行。现有两个 smoke JSON 仍是输入 fixture；Mimi/audio 与 DiffusionGemma/YOLO 的 golden/InferRef 工具不能替代文本 Session 验证。

`compare_accuracy.ps1` 的现有比较模式限定 baseline `oneapi-2025.3`、candidate `oneapi-2026.1`，要求两边同 commit、tracked tree/index 干净，并拒绝模型源码变化后的旧产物。因此它不能直接比较本次同 oneAPI、不同源码的重构结果；不得修改产物 metadata 来绕过检查。

本次组合两层门禁：

1. P0/兼容入口：新增 `perf/compare-token-traces.ps1`，要求完整连续 token 记录和唯一终态，再比较完整 token 数组；不完整日志和差异都返回非零。`perf/refactor-legacy-check.ps1` 在同一 oneAPI 下顺序运行基线与候选，固定 seed 并覆盖采样、惩罚、流式、thinking budget、带历史输入，同时保存输入/模型/二进制身份。不改变现有 oneAPI 升级比较器的含义。
2. P2/资源隔离：真实 GPU fixture 使用同一份权重，A 输入 24 token（8 + 16 分块），B 输入 144 token（96 + 48 分块），随后执行 decode。每步在复用 workspace 前保存 host logits，与各自相同形状的独立执行逐元素精确比较，并检查实际 cache length。另验证 greedy、固定 seed、采样惩罚、取消、截断/重算、会话释放与重新创建、默认设备域共享和日志回调重入。

复跑完整兼容对照：

```powershell
pwsh -NoProfile -File perf/refactor-legacy-check.ps1 -BaselineBuild .worktrees/refactor-baseline-1956de9/build -CandidateBuild build-refactor -ModelDir models/qwen3-0.6B-bnb-nf4-offline
```

真实会话测试的模型环境设置与命令见 [shared-sessions.md](shared-sessions.md)。Qwen3.5 后续加入递归状态/快照回滚，按照实际恢复长度补算。本次旧 Qwen3.5 smoke 不代表它已支持新 Session API。

## 6. P1/P2 首批结果与当时的剩余门禁

| 检查 | 当前结果 | 证据 |
|---|---|---|
| 基线完整 Release 构建 | 通过（集成负责人执行） | 基线 worktree `baseline-build.log`、`build/build_info.json` |
| 候选完整 Release 构建 | 通过，包含新会话接口、测试和 Worker 适配器 | `build-refactor/kv-capability-build.log`、`build_info.json` |
| 基线 12 项 CPU 白名单 | 全部通过，0 失败，11.18 秒（并行 4） | 基线 worktree `baseline-cpu-ctest.log` |
| 候选 15 项 CPU CTest | 全部通过，0 失败，24.58 秒（并行 4） | `build-refactor/final-cpu-ctest.log`、`final-cpu-details.log` |
| Qwen3 greedy smoke/token trace | prompt 34、generated 16，全部 token 相同 | `tmp/refactor/q3-smoke-comparison.json` |
| Qwen3.5 greedy smoke/token trace | prompt 37、generated 9，全部 token 相同 | `tmp/refactor/q35-smoke-comparison.json` |
| Qwen3 旧 CLI 扩展对照 | 五组全部相同：固定 seed 32、惩罚 32、流式 32、thinking budget 13、带历史 8 token | `tmp/refactor/legacy-parity/report.json`，`passed: true` |
| 同栈精确 token comparator | 27 组离线案例通过；拒绝不完整 trace 和错误终态 | `AilaTokenTraceComparisonTests`，`tests/inference/TokenTraceComparisonTests.ps1` |
| Request / 日志 CPU 回归 | 请求 15 组、日志 9 组通过，包括日志回调释放最后一个外部请求句柄 | `AilaRequestStateTests`、`AilaDeferredLoggingTests` |
| 两 Session 逐步 logits/生命周期 | BF16 KV 通过，独立/交错每个 logit 精确一致；含 FP8 准入拒绝的最后复测 23.54 秒 | `build-refactor/kv-capability-details.log` |
| 旧 CLI 单请求性能 | 两模型完成；Qwen3 额外完成紧邻复测，数据见下表 | 各自 `tmp/refactor/q3-bench/bench.json`、`q35-bench/bench.json`、`q3-bench-repeat/bench.json` |

所有性能运行使用 256 prompt / 128 generated、greedy、warmup 1、测量 3 轮、BF16 KV、同一 oneAPI，并串行使用 GPU。吞吐单位为 token/s，变化为候选相对基线：

| 运行 | 阶段 | 基线 | 候选 | 变化 |
|---|---|---:|---:|---:|
| Qwen3 首组 | prefill | 3375.60 | 3282.55 | -2.76% |
| Qwen3 首组 | decode | 139.96 | 135.38 | -3.27% |
| Qwen3 紧邻复测 | prefill | 3286.05 | 3303.54 | +0.53% |
| Qwen3 紧邻复测 | decode | 135.53 | 135.65 | +0.09% |
| Qwen3.5 | prefill | 2183.10 | 2215.89 | +1.50% |
| Qwen3.5 | decode | 138.28 | 137.76 | -0.38% |

Qwen3 首组有约 3% 差异，因此保留原记录并紧邻重跑基线/候选。复测 decode 平均为 944.42 / 943.57 ms，每组标准差 4.87 / 3.09 ms；prefill 为 77.91 / 77.49 ms，标准差 0.90 / 0.59 ms。复测没有重现稳定下降，基线自身也有跨运行漂移。这些短测量不足以设定覆盖所有负载的固定回归阈值，更不代表新 Session API 的并发吞吐或 HTTP TTFT 已测。

基准和兼容 trace 在候选提交前生成，部分 metadata 的 commit 字段仍为 `1956de9`；实际源码差异与二进制 SHA256 另保存在 `tmp/refactor/candidate-source.patch`、`candidate-provenance.json`。判断产物身份时应同时使用这些记录，不能仅凭 Git 字段把基线与候选视为同一源码。

Qwen3 fixture 的容量固定 512；每个 BF16 KV 会话新增 **58,720,256 字节（56 MiB）**，释放后归还，第二会话不复制权重或执行工作区。较大测试形状后 Context 记账的模型/工作区常驻量为 **1,278,658,202 字节**。这些数字不是设备总显存；首批只验证了会话数准入与实际 KV 增减，非零字节预算的新增集成用例见第 8 节。

主项目启用 kernel 用的 fast-math；`Request.cpp` 和相关 host 验证目标单独使用 `/fp:precise`（非 MSVC 模式为 `-fno-fast-math`），保留 NaN/Inf 输入校验。没有更改设备算子优化选项。

首批结果验证了进程内 Qwen3 手动多会话切片与所列旧入口。Qwen3 LayerPlan、自动调度及独立 HTTP 探针随后完成的验证见第 8 节；Hybrid 共享状态、IPC 多请求、真实 HTTP 推理、4B/长上下文、长期压力与服务延迟分布仍待后续阶段验收。

## 7. FP8 KV 探索结果与当前限制

额外使用 `AILA_KV_QUANT=1` 运行同一真实会话 fixture，17.16 秒失败于 `interleaved A 2: non-finite logit at 0`。原始失败保存在 `build-refactor/qwen3-fp8-session-details.log`；这不是 FP8 正确性通过记录。

源码对照发现两处既有问题：

- `src/ops/AttentionOps.cpp` 的 `attention_decode` 在默认 JM 路径不检查 KV dtype，`attention_decode_joint_matrix_tiled` 把 K/V 当作 `bf16*`，而缓存可能实际为 FP8。A 的首次 decode（cached length 25）符合该分支条件。
- `src/ops/ElementwiseOps.cpp` 的 head_dim 128 量化融合写入只写 `k0/v0` 的低 64 维，漏掉 `k1/v1` 的高 64 维；部分旋转变体也有相同问题。仅关闭 JM 不能证明整个 FP8 路径正确。

上述两个 ops 文件及 `src/memory/KVCache.cpp` 与基线 `1956de9` 无差异。这里保留了失败与源码证据，没有额外运行基线 FP8 logits 诊断，也没有在本次资源拆分中修改这些算子。

新共享会话 API 限定 BF16 KV：加载前检查有效 `AILA_KV_QUANT` 设置，显式 `CreateSession` 再次检查以覆盖加载后设置变化，均以 `InvalidArgument` 拒绝 FP8。旧默认会话入口保持原有行为。GPU fixture 已通过加载前/加载后开启量化的拒绝回归：错误提示明确、会话计数和 Context current/peak 分配不变，恢复设置后可继续 BF16 推理。FP8 重新开放必须先修复读取/写入问题并单独通过数值验证。

## 8. LayerPlan / Scheduler 续批验证

本轮基于 `ad25cd3` 的共享会话实现继续开发，基线仍为不可变 `1956de9`。使用第 1 节同一工具链、A770 和驱动。Qwen3 加载时绑定具体 LayerPlan，执行时显式传入阶段、位置、KV 视图和工作区；保留 NF4 内核、跨层 residual/norm 融合与 in-order queue，没有新增逐层同步。Scheduler 在完整 prefill 分块、decode 或 finalization 之间轮转，所有 GPU 验证和测量顺序执行。

### 构建、正确性与资源回收

| 检查 | 已执行结果 | 证据 |
| --- | --- | --- |
| 完整 Release 构建 | 通过；最终构建包含去除 `ContextLength()` 设备全局锁后的源码 | `build-refactor/scheduler-final-build.log` |
| 16 项 CPU CTest | 全部通过，24.36 秒；Scheduler 内部 19 组 | `build-refactor/scheduler-cpu-ctest.log`、`scheduler-cpu-details.log` |
| Qwen3 Session GPU | 通过，23.17 秒；每步 logits 精确一致 | `build-refactor/scheduler-gpu-ctest.log`、`scheduler-gpu-details.log` |
| Qwen3 Scheduler GPU | 通过，26.14 秒；四会话 token 及最终探针 logits 与独立运行精确一致 | 同上，两项总计 49.32 秒 |
| 旧 Qwen3 CLI 完整 token 对照 | 五组全部一致，明细如下 | `tmp/refactor/layer-scheduler-legacy-parity/report.json` |
| 同会话与跨会话取消/回收 | 192/256-token 首块后取消或丢弃 A，B 继续正确生成；A 的 KV 精确归还 | `AilaQwen3SchedulerTests` |
| 输出背压 | 4-token ring 满时只取消 A；B 完成；独立终态位置保留 | `AilaSchedulerTests`、`AilaQwen3SchedulerTests` |
| 非零模型 KV 字节预算 | `maxSessions=4`、预算 `2 * cacheBytes - 1` 时，第二 Session 在分配前被拒绝；销毁后额度可复用 | `AilaQwen3SchedulerTests` |
| 生命周期与错误 | 排队/运行取消、已完成未读流占用额度、RuntimeFault 广播、回调重入/销毁、并发 Shutdown、Submit/Shutdown 竞争均通过 | `AilaSchedulerTests` |

CPU 16 项执行后，最后仅调整了 `Model.cpp` 的主机长度 getter 和 `Session.hpp` 的同步说明；随后完整重建并通过上述两项 GPU 测试。getter 由 Session 操作锁或尚未发布的构造期保护，不再等待另一个会话的设备执行锁；执行、分配和销毁仍保留设备锁。

| 旧 CLI 对照案例 | Prompt tokens | Generated tokens | 完整 ID 数组 |
| --- | ---: | ---: | --- |
| fixed_seed | 34 | 32 | 相同 |
| penalties | 34 | 32 | 相同 |
| stream | 34 | 32 | 相同 |
| thinking_budget | 34 | 13 | 相同 |
| history | 54 | 8 | 相同 |

此对照测量时的 CLI SHA256 为 `F357FCA6018D798AFC78F195728D144B7C2CC06D8F55E45B743E0455FFA5B0CA`，基线为 `7E082C11C94476D870ACCF3642F1D603B97FF22440199E04A8475810A36FD43B`。最终接口重建后 CLI 发生重新链接，不把较早哈希当作最终产物；提交后构建身份、二进制哈希和补充检查统一保存在 `tmp/refactor/layer-scheduler-final-provenance.json`，最终 CLI 对照另存 `tmp/refactor/layer-scheduler-final-legacy-parity/report.json`，保留上述原始结果。

### 旧 CLI 相邻性能对照

固定 256 prompt / 128 generated、greedy、warmup 1、测量 3 轮、最大上下文 4096、BF16 KV。基线后紧接候选运行；这一组用于检查 LayerPlan 提取对旧路径的影响。

| 阶段 | 基线 token/s | 候选 token/s | 基线平均 ms | 候选平均 ms |
| --- | ---: | ---: | ---: | ---: |
| Prefill | 3404.04 | 3405.98 | 75.20 | 75.16 |
| Decode | 142.54 | 141.53 | 897.98 | 904.42 |

Decode 标准差分别为 4.69 / 2.72 ms，候选吞吐约低 0.71%。保留该短测量差异，不据此设定通用回归门限。原始数据位于基线 worktree 与主工作区各自的 `tmp/refactor/q3-layer-bench/bench.json`，构建目录保留 `layer-baseline-bench.log` 和 `layer-candidate-bench.log`。前一阶段的性能数据未覆盖或删除。

### 进程内 1/2/4 会话性能

入口为 `perf/shared-session-bench.ps1`。本次运行目录：`tmp/refactor/q3-scheduler-bench/20260907-180522-641b5122`。`benchmark.json` 保存每次 Submit 和实际 Read 到达的时间、全部输出 ID 与逐 token 时间；`run.json` 保存模型/二进制/源码哈希、构建/驱动、环境和退出码，`tracked-source.patch` 保存当时 tracked diff。

每个并发数预热 1 轮、测量 3 轮，固定上下文容量 512、prompt 256、生成 128、prefill chunk 256、greedy、BF16 KV，每请求一个读取线程。模型加载、tokenization 和 Session 创建在计时外。TTFT 从各请求 Submit 前计时到其读取线程实际读到首 token；间隔取同一流相邻 token 到达时间。总吞吐是所有测量轮的输出 token 总数除以各轮“最早 Submit 到最后 terminal Read”的时长之和，包含 prefill 与 finalization。分位数用 nearest-rank，跨轮汇集请求/间隔样本。

| 会话数 | 总吞吐 token/s | TTFT p50 / p95 ms | Token 间隔 p50 / p95 ms | 测量轮总时长范围 ms |
| --- | ---: | ---: | ---: | ---: |
| 1 | 121.28 | 76.60 / 76.92 | 7.61 / 7.93 | 1049.89–1062.62 |
| 2 | 120.34 | 75.52 / 151.44 | 15.32 / 16.11 | 2110.82–2156.89 |
| 4 | 116.47 | 159.57 / 317.59 | 30.98 / 34.81 | 4242.12–4694.09 |

四会话三轮吞吐分别为 120.69、120.42、109.07 token/s；最后一轮较慢，原样纳入汇总，没有删掉该样本或推断其原因。每个并发数仅有 3/6/12 个 TTFT 样本，p95 不代表稳定的服务尾延迟。请求交替推进改善多会话可用性；当前没有合批，本次测量未显示并发吞吐提升。这些数字是进程内 token 流观测，不能当作 HTTP/SSE 延迟，也不能直接与旧 CLI 的 decode-only token/s 比较。

在各轮请求结束、Session 仍存活时，Context current 字节数分别为 1,306,903,194 / 1,365,623,450 / 1,483,063,962。同一并发数三轮数值相同，差值符合每会话 58,720,256 字节 KV；这只证明所测形状下的 Context 记账稳定，不是总显存或长期泄漏门禁。JSON 中 peak 是整个测试进程截至该时刻的峰值，不能视为每轮重置后的独立峰值。

### HTTP 探针与剩余范围

独立普通 C++17 探针在 Drogon 1.9.11 + Trantor 1.5.24 及本地 Windows listener callback 补丁下，真实 loopback 的六个场景全部通过：增量 UTF-8 SSE、双客户端、断线隔离、慢读、完成竞争及退出清理。未打补丁的失败记录也已保留。此探针没有加载模型或调用 GPU；结果、复现命令、固定依赖及连接复用/背压限制见 [refactor-http-probe.md](refactor-http-probe.md)。

本轮交付 Qwen3 的层计划和进程内自动调度；共享接口仍只接受 NF4 权重与 BF16 KV。Hybrid Full/DeltaNet 状态与回滚、BF16 权重策略、总显存预算、多在途 Worker/Proxy ID/epoch、真实 HTTP 推理、长上下文/4B 和持续服务压力均保留为后续验收项。

## 9. Qwen3.5 Hybrid 共享会话续批验证

本批在 `f85790d` 上继续开发，基线仍为不可变 `1956de9`，沿用第 1 节 A770 / oneAPI / 驱动。`LoadedModel::LoadQwen35Nf4` 接入已有 Session/Request/Scheduler，Full KV、DeltaNet 矩阵、卷积历史、host mirror 和快照均归会话；NF4 投影、共享执行槽及跨层融合保留。具体 API、实际回滚长度与预算说明见 [qwen35-shared-sessions.md](qwen35-shared-sessions.md)。

### 构建、数值和资源门禁

下表为包含分组竞争修复后的最终源码验证。此前通过和失败的原始日志均保留，没有覆盖失败记录。

| 检查 | 实际结果 | 证据 |
| --- | --- | --- |
| 完整 Release 构建 | 通过，包含最终 backend 和诊断 fixture | `build-refactor/q35-grouped-fix-build.log` |
| 17 项 CPU CTest | 全部通过，24.38 秒；新增 SessionMemory 预算边界测试 | `build-refactor/q35-final-cpu-ctest.log`、`q35-final-cpu-details.log` |
| Qwen3 Session / Scheduler GPU 回归 | 通过，23.62 / 26.43 秒，总计 50.06 秒 | `build-refactor/q35-final-q3-gpu-ctest.log`、`q35-final-q3-gpu-details.log` |
| Qwen3.5-0.8B 完整 GPU fixture | 通过，37.41 秒 | `build-refactor/q35-final-08b-gpu-ctest.log`、`q35-final-08b-gpu-details.log` |
| Qwen3.5-4B 完整 GPU fixture | 通过，81.37 秒 | `build-refactor/q35-grouped-fixed-4b-gpu-ctest.log`、`q35-grouped-fixed-4b-gpu-details.log` |
| 4B 混合形状 logits 复现 | 修复后首次通过，另用三个新进程重复，三次均精确通过 | `build-refactor/q35-grouped-after-interleaved.log`、`q35-grouped-repeat-1.log` 至 `q35-grouped-repeat-3.log` |

0.8B 与 4B 使用同一完整 fixture，context 固定 512，快照上限 4，BF16 KV。覆盖内容包括：

- 独立 legacy backend 对 `[11,3,7,16,19,1,3,1]` 步形状读取实际最后有效 logits 行，再与共享 API 逐元素精确比较，覆盖小块增量的多行输出与旧容量残留。
- 两 Session 不同分块形状交错的每步 logits，以及四 Session 的 initial/continuation token、greedy、固定 seed、penalties 和最终状态探针。独立参考保持各请求的分块形状。
- 截断到 20 实际恢复 checkpoint 17，截断到 65 恢复 decode checkpoint 64；无 checkpoint 时返回 0；最近四份快照的保留/淘汰以及禁用快照后的重算。
- 192/256-token prefill 安全边界的取消与丢弃，B 继续正常完成；反复创建/销毁和外部 Model/Runtime 先释放的所有权验证。
- device 与 host 预算分别设为 `2 * reservation - 1`、`maxSessions=4`，第二个 Session 在分配前拒绝；释放后额度可复用。有效 scoped/global/engine KV 覆盖矩阵在加载和 Session 创建前拒绝量化 KV。

CPU 内存预算测试独立于 SYCL，校验已知模型字节数、上下文增长、分组 heads、显式/部分 layer_types、kernel=1、非法维度、`size_t` 溢出及最大可表示快照额度。模型准入与 backend 分配使用同一个计算模块。

| 模型，context 512 | 每 Session device payload | Host mirror + 4 snapshots 额度 |
| --- | ---: | ---: |
| Qwen3.5-0.8B NF4 | 26,492,928 B | 101,007,360 B |
| Qwen3.5-4B NF4 | 69,468,160 B | 263,454,720 B |

这两个 device 数值与 fixture 中实际新增/释放的 Context 记账一致。host 列是 float payload 最大预留，不是进程 RSS 实测；不包含 STL/allocator 开销。禁用快照仍需要一份 host mirror。两列都不包含权重、共享 workspace 或 oneDNN/driver 分配。

### 4B 验证暴露的分组 DeltaNet 竞争

首次 4B 完整测试在 44.40 秒失败于 session 0 的第二个追加 token。该 Session 使用 greedy、无惩罚，初始和追加分块在独立/调度路径相同。缩小到 `--case interleaved-logits` 后，首次 8-token prefill 就有 **232,280 / 248,320** 个 logits 不同，`max_abs=0.359375`，说明问题并非采样随机数。失败证据是 `build-refactor/q35-shared-4b-gpu-details.log` 和 `q35-grouped-before-interleaved.log`。

源码原因是一个 value head 对应一个工作组，分组时两个或更多工作组共用同一 Q/K 卷积历史。leader 在组内 barrier 后原地写回，其他工作组可能还没读完旧值；prefill 各工作组的 token 循环也不能互相同步。这段 decode/prefill helper 在修复前与基线 `1956de9` 的 31,392 字符区域完全一致，比较证据位于 `tmp/refactor/q35-grouped-before/baseline-kernel-comparison.json`。该目录还保留修复前源码和二进制哈希。

修复仅针对 grouped fast path：计算 kernel 期间历史只读；prefill 的历史 tap `k=t-3+j` 在 k<0 时读初始物理 ring，否则读取本块第 k 行投影输入。计算结束后，用后置 kernel 按原 ring/head 格式写回最后 `min(T,3)` 行；in-order queue 提供跨 kernel 顺序，不添加逐层主机等待。非 grouped / generic fallback 和 NF4 GEMV/GEMM 保持原计算方式。代码复核确认 T=1/2 的混合历史、长块末三行与 input stride；随后通过上表中的精确数值、回滚与重复复现门禁。

这个修复同时作用于 NF4 后端的新旧入口。未把含竞争的 4B 旧输出当作正确性 oracle，也未用它计算性能加速比。BF16 权重后端尚未迁移，其相似 grouped 实现仍需单独验证和处理。

### 旧 CLI 完整 token 与相邻性能对照

使用 `perf/refactor-legacy-check.ps1` 对 Qwen3.5-0.8B 比较不可变基线与候选；五组全部精确一致，`tmp/refactor/q35-shared-legacy-parity/report.json` 的 `passed` 为 true。

| 案例 | Prompt tokens | Generated tokens | 完整 ID 数组 |
| --- | ---: | ---: | --- |
| fixed_seed | 37 | 24 | 相同 |
| penalties | 37 | 9 | 相同 |
| stream | 37 | 9 | 相同 |
| thinking_budget | 37 | 9 | 相同 |
| history | 54 | 5 | 相同 |

性能继续使用此前的短测协议：256 prompt / 128 generated、greedy、最大上下文 4096、warmup 1、测量 3 轮、BF16 KV，关闭 Q3/Q35 profiling 和 oneDNN verbose。首组按基线→候选运行；因为 prefill 有 3.54% 差异，又在源码和二进制不变的情况下按候选→基线复测，保留两组结果。

| 运行顺序 | 阶段 | 基线 token/s | 候选 token/s | 候选相对变化 | 基线 / 候选平均 ms |
| --- | --- | ---: | ---: | ---: | --- |
| 基线→候选 | Prefill | 2254.93 | 2175.12 | -3.54% | 113.53 / 117.69 |
| 基线→候选 | Decode | 143.32 | 143.52 | +0.14% | 893.11 / 891.87 |
| 候选→基线 | Prefill | 2255.35 | 2234.64 | -0.92% | 113.51 / 114.56 |
| 候选→基线 | Decode | 142.96 | 145.18 | +1.55% | 895.33 / 881.64 |

首组 prefill/decode 的标准差为基线 0.28 / 3.06 ms、候选 0.05 / 1.01 ms；复测为基线 0.44 / 3.47 ms、候选 0.44 / 6.37 ms。prefill 首轮差异在反序复测中缩小，但两组都略低于基线，不能据此宣称完全无性能下降或设定通用阈值。原始结果分别位于基线 worktree 与主工作区各自的 `tmp/refactor/q35-shared-bench/bench.json` 和 `q35-shared-bench-repeat/bench.json`，未覆盖历史基准。这些是旧 CLI 的阶段吞吐，不是 Hybrid 多会话或 HTTP 性能。

本批源码快照（包含新增且当时未跟踪的文件）、完整模型文件哈希、构建信息、二进制/运行时 DLL 和日志身份保存于 `tmp/refactor/q35-shared-precommit-provenance.json`；提交后的身份、重建和补充检查对应 `q35-shared-final-provenance.json`。候选测量时的 build metadata 仍写 `f85790d`，须结合源码快照与二进制 SHA256，不能把该字段当作已提交的完整源码身份。

### 本批边界

本批证明 Qwen3 与 Qwen3.5 NF4 文本会话可复用进程内执行/调度框架，未实现连续合批。Hybrid 新接口仍是 token API；媒体 setter、聊天模板、reasoning 和 thinking budget 继续属于旧驱动。4B 的上述检查固定 context 512，不代表长上下文或长期服务压力门禁已经完成。

[IPC v2 文档](inference-ipc-v2.md) 是可实施的协议设计，尚未改动 Worker/Proxy 多在途通信或增加真实 HTTP 推理。初始化/加载中的同步设备异常尚未可靠统一为 `RuntimeFault`；v2 接入前需补齐分类、原始故障记录、生命周期故障广播和注入测试。新 host/device 额度是 payload 准入，完整内存预算和 Qwen3.5 的 1/2/4 会话性能分布仍待后续阶段验证。

## 10. IPC v2：Qwen3 独立 Worker 纵向切片

本批从 `78bcdd1` 的协议设计继续实现。新增严格 v2 codec、Worker 控制器/生命周期执行器、独立管道 reader/writer、普通 C++ `InferenceClient` 和真实 Qwen3 模型工厂。v1 Worker、Proxy 与公共 C ABI 保留。具体接口和所有权见 [inference-ipc-v2-implementation.md](inference-ipc-v2-implementation.md)。本节的验证对象是 token API；HTTP/SSE 的实现和验证单独记录。

### 构建与回归

| 检查 | 实际结果 | 证据 |
| --- | --- | --- |
| 完整 Release 构建 | 通过，oneAPI 2026.1 / oneDNN 3.11.2 | `build-refactor/ipc-v2-full-build.log` |
| 23 项基础、兼容与 IPC 回归 | 全部通过，60.67 秒，串行执行 | `build-refactor/ipc-v2-final-regressions.log`、`ipc-v2-final-regressions-details.log` |
| v2 codec | 11 组通过；严格类型/重复字段/UTF-8/附件/限额/完成摘要 | `AilaInferenceProtocolTests` |
| 设备故障分类 / Scheduler | 7 / 22 组通过；同步故障、首因保留、最终 drain、生命周期广播 | `AilaRuntimeFaultTests`、`AilaSchedulerTests` |
| Worker 控制器 | 6 个测试函数通过，覆盖资源准入、父子所有权、释放/背压、fault、日志和关闭 | `AilaInferenceWorkerTests` |
| 客户端受控双通道 | 16 组通过，首次完整通过 0.30 秒，最终回归 0.28 秒 | `ipc-v2-client-tests-fixed.log`、最终回归明细 |
| 真实 Windows 子进程 / CPU pipe | 7 组通过，首次 6.96 秒，最终回归 7.76 秒 | `ipc-v2-pipe-tests.log`、最终回归明细 |
| 真实 Qwen3 IPC GPU | 全部通过，61.41 秒 | `ipc-v2-qwen3-gpu.log`、`ipc-v2-qwen3-gpu-details.log` |
| Qwen3 Session / Scheduler GPU 回归 | 23.21 / 26.33 秒，全部通过 | `ipc-v2-model-gpu-regressions.log`、`ipc-v2-model-gpu-regressions-details.log` |
| Qwen3.5-0.8B Session GPU 回归 | 36.93 秒，通过；三项模型回归总计 86.48 秒 | 同上 |

客户端单测首次在第 9 组失败：CPU 假响应的 SchedulerStopped 摘要遗漏协议要求的 errorCode，codec 正确拒绝；修复 fixture 后 16 组通过，未放宽 codec 或修改宿主行为。原失败保存在 `ipc-v2-client-tests.log`。RuntimeFault 单测此前还修正了 MSVC 上不可依赖的 exception_ptr 身份断言，改为检查异常动态类型、消息和错误 payload；生产故障策略未放宽。

`AilaInferencePipeTests` 实际运行同一个 Worker 外壳和 Scheduler，只替换 CPU 模型工厂；包括双 Session、取消/释放后的私有排空、未读输出关闭、异常子进程退出、新 epoch，以及五次启动中每次三轮创建/释放与模型槽复用。检查宿主句柄回到起始数，PATH/当前目录不变且未载入 oneAPI 模块。`dumpbin /dependents` 的普通宿主产物只包含 Windows、MSVC/UCRT 依赖，没有 SYCL/oneDNN DLL；原始清单是 `ipc-v2-host-dependents.log`。此处是有限轮次的句柄稳定性验证，不是长期 RSS/显存泄漏结论。

### 真实 Qwen3 的覆盖范围

模型为 `models/qwen3-0.6B-bnb-nf4-offline`，BF16 KV，context 容量 512；设备与工具链沿用前文。直接推理先产生参考结果并销毁全部设备 owner，再启动独立 Worker，避免同时运行两个 GPU 测试进程。

- 四条不同 prompt 长度为 40/43/52/50，生成 20/23/26/29 token，prefill chunk 为 16/32。前两条 greedy，后两条固定独立 seed，含 repetition/presence/frequency penalties。
- 1/2/4 个 IPC Session 的完整生成 ID 与直接参考精确相同；各追加 5 token 的续写也精确相同。空闲 Session 的 contextLength 包含最后生成 token，并观察到多个已接受请求重叠执行/排队。
- 长 A 首 token 后取消，B 仍与参考相同；另一次完全停读 A 的 256-token 请求触发 A 的 OutputBackpressure，B 完成。A 的 wire 窗口不超过 32，core ring 不超过 128，生成数不超过 160；未消费 terminal 继续占额度，Release 私有排空后归还。
- 请求执行中先释放 Session/Model，请求仍正确结束。Shutdown 完成 eventOrdinal 屏障与子进程退出；新 Worker 采用新 epoch，重新加载后的输出仍相同。

fixture 中 `caller_first_token_ms` 是主测试线程轮询读到首 token 的时刻，包含冷启动/JIT、各请求提交和读取顺序；首次单会话记录为 15365.6 ms。它不是预热后的 TTFT 基准，不据此报告 IPC 吞吐或服务性能。本批未重复前阶段的 NF4 内核性能测量，未改动 NF4 GEMV/GEMM 算子。

### 故障与当前边界

加载/Session 初始化的同步 SYCL 和 oneDNN 异常现在会记录首次原因并归一为 RuntimeFault，即使后续 drain 成功也不漏报。普通文件、配置和 host allocation 失败仍可局部返回；最终 drain 失败会升级为设备域故障。生命周期执行器向 Scheduler 报告 fault，停止准入、保留已经发布的终态并使未完成请求在安全边界失败。CPU 注入验证了这些策略；本批没有故意触发真实 GPU 设备挂起。

v2 当前只发布 Qwen3 NF4/BF16 KV token 请求能力。Qwen3.5 远端、公共异步 C ABI、聊天/thinking budget 和 HTTP/SSE 不属于这个 token 切片。所有 payload 额度仍不代表整个进程 RSS 或设备总显存；连续合批、长期服务压力和更大上下文另行验收。

源码快照、当时的 diff、模型与二进制哈希、build metadata 和日志身份保存在 `tmp/refactor/ipc-v2-precommit-provenance.json`；提交后的匹配构建身份与补充检查保存在 `tmp/refactor/ipc-v2-final-provenance.json`。提交前 metadata 的 Git 字段仍为 `78bcdd1`，需要结合源码快照与 SHA256 判断实际产物。

## 11. C++ HTTP/SSE 真实推理服务

本批在 `cd57b46` 的 IPC v2 基础上实现普通 MSVC C++17 `AilaServer`，通过独立 Worker 服务 Qwen3 NF4 文本请求。提供 `/health`、`/v1/models` 和 `/v1/chat/completions`，支持 JSON 与 SSE。字段、构建/运行命令、传输语义和限制见 [inference-server.md](inference-server.md)。历史依赖探针目录、IPC 证据和旧基线均保留；本批使用独立 `build-refactor-server` 与 `build-refactor-server-deps`。

### 构建、CPU 与依赖检查

| 检查 | 实际结果 | 证据 |
| --- | --- | --- |
| 普通 MSVC Release 服务构建 | 通过；Drogon 1.9.11、Trantor 1.5.24、JsonCpp 1.9.6、zlib 1.3.1，静态框架与 `/MD` | `build-refactor-server/server-build-deps-fixed.log`、`logs/build-server.txt` |
| ChatCodec | 19 组通过，0.08 秒；严格字段/Unicode、reasoning、usage、输出预算、结束与故障语义 | `server-cpu-deps-fixed-details.log` |
| 普通 MSVC InferenceClient | 16 组通过，0.28 秒 | 同上 |
| 普通 MSVC 真实 CPU pipe | 7 组通过，7.83 秒；三个 CTest 总计 8.20 秒 | `server-cpu-deps-fixed.log` 与明细 |
| Server PE 导入 | 仅 Windows、MSVC/UCRT，无 SYCL/oneDNN/OpenSSL | `server-dependents-fixed.log` |
| Ninja header dependencies | 31 个 C++ 对象全部非空且 VALID；HttpServer.cpp 记录 221 项依赖 | `logs/server-header-dependencies.txt`、`server-build-info.json` |

真实构建复核发现本机 CMake 4.1.2-msvc8 的 MSVC `/showIncludes` 自动探测将 UTF-8 中文前缀按 ANSI 936 解码，Ninja 因原始字节不匹配记录了 0 项依赖。设置 `VSLANG=1033` 未改变该编译器的输出。修复采用独立小编译探针、`ENCODING NONE` 保留原字节，从已知 header 行提取前缀；前缀哈希进入编译命令，使旧 0-deps 对象自动重编译。构建脚本额外校验所有对象的依赖有效。

隔离复现先证明旧配置修改 header 后漏编译；同一探针目录重配后依赖从 0 变为 4，包含直接/间接自有 header 和 UCRT header，修改间接 header 正常重编译，再次无改动构建为 no-op。证据保存在 `tmp/refactor/server-compiler-probe/raw-prefix-1/result.json`，未改写历史 HTTP 探针。

### 实际 socket 验收

`tests/server/socket_tests.py` 使用 Python 标准库，实际启动 Server 与 Worker，并通过 loopback socket 读取 HTTP/SSE。每个 runner 持有自己的 Popen 句柄，先创建唯一 shutdown 文件、等待有界退出，失败时只终止其拥有的进程。两个 suite 串行运行，未与其他 GPU 进程并跑。

| Suite | 结果 | 用例合计耗时 | 结果文件 |
| --- | --- | ---: | --- |
| CPU fake Worker + 实际 HTTP/SSE | 11/11 通过 | 21.07 秒 | `tmp/refactor/server-sockets-fake/result.json` |
| Qwen3-0.6B NF4 + 实际 HTTP/SSE | 7/7 通过 | 24.44 秒 | `tmp/refactor/server-sockets-gpu/result.json` |

GPU 模型为 `models/qwen3-0.6B-bnb-nf4-offline`，BF16 KV，context 512；服务配置 4 个请求槽、生成上限 256 token、1 MiB 输出预算、64 KiB Trantor 门限与 5 秒慢读期限。单条 GPU 内容验收生成至多 24 token，并发固定 seed 案例各至多 16 token。每条 JSON/SSE 比较包括完整 content、reasoning_content、finish_reason 和 usage；HTTP 接口不输出原始 token ID，逐 token 的独立参考比较由第 10 节的 IPC GPU 门禁提供。

两组共同验证：

- 健康与模型端点、未知字段、非法 JSON/seed、超生成上限、n>1、错误 stream_options 和媒体输入拒绝。
- JSON 与实际增量 SSE 完全一致；分段 socket 读取保持 UTF-8/帧边界，usage 可选，finish 与 `[DONE]` 各一次。
- 1/2/4 并发的活动请求峰值实际为 1/2/4，各输出与其独立固定 seed 参考一致。
- 客户端 `Connection: close`、实际 `100 Continue`、同包 pipeline、延迟第二请求和连接复用关闭。
- 确认 A/B 同时 active 后断开 A，B 仍与参考一致；GPU 用例观察到 A 约 15.95 ms 后进入断连终态，但这是单次验收观测，不是取消最坏延迟保证。
- 带活动请求关闭服务，无需 runner 强制 kill；请求、IPC outstanding/pending 与连接集合均归零，accepted 与 closed 计数一致。

Fake 模式额外生成约 4056 字节/token 的合法 UTF-8 文本，以 160-token 响应与 4096 字节接收缓冲区稳定制造网络压力。观察到持续不读时 pause_ticks 增长，独立 B 正常完成；恢复读取后 A 完成，持续停读则只取消 A，随后槽位可复用。四个占用请求得到明确 429，断开一个后可立即再次接入。

| Fake 传输观测 | 实际值 | 配置上限 |
| --- | ---: | ---: |
| Trantor 最大实际 pending bytes | 63,855 B | 65,536 B |
| 单请求最大 handoff frame | 4,249 B | 16,384 B |
| 恢复读取的 A | 160 个内容帧，680,445 B SSE payload | 1 MiB 输出预算 |
| SO_SNDBUF 回调读回 | 16,384 B；388 次回调、0 次失败 | 请求值 16,384 B |

Fake 主进程最终关闭 388/388 个连接；GPU 关闭 141/141 个连接。GPU 短文本仅达到 1,620 B pending、未触发网络暂停，因此不拿它证明慢读门限。Fake health 344 个样本中位 15.81 ms、最大 549.28 ms；GPU health 108 个样本中位 14.50 ms、最大 27.08 ms。这些仅是本次 runner 的观测，受客户端轮询、Windows 调度和首次 JIT 影响；GPU 增量用例本身耗时 16.41 秒，不能当作预热 TTFT 或服务吞吐基准。

独立故障用例用专用 token 使 fake Worker 退出，SSE 收到单个 error payload 与 `[DONE]`，没有成功 finish_reason，随后推理返回 503。故障服务关闭的退出码为 2、`shutdown.graceful=false`，明确保留 Worker 故障；`shutdown.idle=true`、5/5 socket 关闭且所有请求/命令归零，无强制 kill。原始响应、stderr 和故障 report 在 `server-sockets-fake/worker-fault`。

### 当前边界与产物身份

本批复用现有 NF4 内核、in-order queue、Scheduler 和 IPC 所有权，没有改变 GEMV/GEMM 计算。真实服务当前只发布 Qwen3 NF4、BF16 KV、纯文本、本机 HTTP/1.1、一连接一个应用请求。Hybrid 远端、鉴权/TLS、持久连接、工具/媒体/thinking budget、连续合批、长上下文、长期压力和性能分布仍需后续验收。

Trantor 扩展读取实际 memory/async write 节点，包含普通 HTTP 输出；不是用累计 bytesSent 推算消费确认。该快照加上单请求 handoff 仅约束被计入的数据载荷，不涵盖内核 socket buffer、所有 Drogon header/parser 内存、allocator、权重或驱动分配，不能作为总 RSS/显存上限结论。

提交前源码快照（含当时未跟踪的服务文件）、模型/产物/依赖/日志 SHA256 与 socket 证据清单保存在 `tmp/refactor/server-precommit-provenance.json`；此时 Worker metadata 的提交身份为 `cd57b46`，实际源码须结合快照判读。提交后匹配重建及补充验收的身份记录保存在 `tmp/refactor/server-final-provenance.json`；每个快照目录另复制当时的构建日志，避免后续重建覆盖历史内容。

### 提交后日志复核：Trantor 陈旧写事件

服务首片提交 `b8a5086` 后，匹配重建、BuildIdentity/pipe/Proxy ABI 三项（19.70 秒）、普通宿主三个 CPU CTest（8.48 秒）及 GPU socket 7 项仍通过。随后复核 stdout，发现初次 fake 与此次 GPU 各有一条 `FATAL no writing but write callback called`，原 socket runner 仅检查响应和资源收尾，未将框架日志纳入失败条件。这两次记录只证明当时的功能断言，不能据此声称框架无错误日志。

源码路径是 `Channel::handleEventSafely()`：同批 IN|OUT 先执行 read callback；该 callback 可因普通 FIN 进入 `handleClose()`，清除写订阅。Channel 随后仍根据旧 revents 分发 write，触发 `TcpConnectionImpl::writeCallback()` 的 LOG_SYSERR。Windows Wepoll 将普通 FIN 映射为 IN/RDHUP，现有 `!HUP` 判断无法排除这条路径；LOG_SYSERR 在该版 Logger 中标成 FATAL 但不会终止进程。

新增真实 loopback `AilaServerChannelTests`，通过公共 API 预置可读字节和写订阅，必须实际观察同一批 `revents=0x5`。未补丁库稳定复现：read 中 disableAll、disableWriting 两组仍错误执行 write（顺序 rwq），另两组 keepWriting、disable→reenableWriting 正常通过，合计 2/4。原库、Server 和日志身份保存在 `tmp/refactor/server-channel-before/provenance.json`，失败详情为 `build-refactor-server/server-channel-before-tests-details.log`。

修复以独立 `trantor-channel-interest.patch` 在分发 write callback 前重新检查 `isWriting()`，不更改真实写事件、读回调、HTTP 取消流程或日志等级。依赖锁新增补丁哈希，构建会拒绝缺少补丁的旧前缀。Socket runner 同时检查框架 ERROR/FATAL 日志；它能够检出上述两份原始日志，旧证据完整保留。

补丁后同一 Channel 测试 4/4 通过：两种禁用场景都是 read=1/write=0/order=rq，两种保留写订阅的对照仍为 read=1/write=1/order=rwq，全部实际观察到 revents=0x5。完整普通宿主四个 CTest 通过，总计 7.59 秒，32 个编译对象的头文件依赖均有效。证据为 `build-refactor-server/server-channel-fixed-tests.log` 及其明细。

加入日志门禁后，完整 fake socket 11/11（用例合计 21.01 秒）与 GPU socket 7/7（22.70 秒）均通过，`framework_errors=[]`，分别在 `tmp/refactor/server-sockets-channel-fake` 与 `server-sockets-channel-gpu`。Fake 仍观察到 63,855 B pending / 4,249 B handoff；主进程 388/388、GPU 143/143 个连接正常收尾，请求/IPC 计数归零。独立故障 Worker 用例保持退出码 2 与 idle=true，未出现框架错误。补丁前后原始输出与失败证据均未覆盖；修复提交前快照为 `tmp/refactor/server-channel-precommit-provenance.json`。

## 12. Qwen3.5 Hybrid 远端与 HTTP/SSE

本批在 `8ec6a3e` 基础上增加 `qwen35_nf4` / `qwen35_nf4_bf16_kv` 能力，复用已有 IPC v2 schema、Worker 生命周期、Scheduler 和 Hybrid 共享 Session。普通 C++ 客户端新增 `LoadQwen35`，Server 新增 `--backend qwen35_nf4`，默认 HTTP 模型名为 `qwen3.5`。原 Qwen3 参数、默认握手、v1 Worker/Proxy 和公共 C ABI 保留。本批没有修改 NF4/DeltaNet 算子或 in-order queue。

### 构建与 CPU 门禁

| 检查 | 实际结果 | 证据 |
| --- | --- | --- |
| 完整 Worker Release 构建 | 通过，oneAPI 2026.1.0 / oneDNN 3.11.2 | `build-refactor/hybrid-remote-worker-build.log` |
| 根构建目录的协议、调度、故障、Client、真实 pipe 与兼容门禁 | 12/12，通过，27.47 秒 | `build-refactor/hybrid-remote-cpu-tests.log`、`hybrid-remote-cpu-details.log` |
| 普通 MSVC C++17 Server 构建 | 通过，32 个 C++ 对象的 Ninja 头文件依赖有效 | `build-refactor-server/hybrid-remote-server-build.log`、`hybrid-history-corrected-build.log` |
| 普通宿主完整 CPU CTest | 4/4，通过，10.61 秒；Channel 4 组、ChatCodec 25 组、Client 18 组、真实 pipe 8 组 | `build-refactor-server/hybrid-remote-cpu-corrected-tests.log`、对应 `details.log` |
| 宿主 PE 导入 | 仅 Windows/MSVC/UCRT，无 SYCL/oneDNN 导入 | `build-refactor-server/hybrid-remote-dependents.log` |

协议测试共 12 组，确认新增 backend 的严格编解码、默认 Qwen3 能力兼容、未知后端与 KV 的局部拒绝。Worker 的 7 个后端场景分别对 Q3/Q35 执行，连同公共控制场景共 15 个；覆盖正确工厂分发、加载失败恢复、独立 Session/续写、准入、背压和生命周期。Client 验证 Qwen3-only peer、hello/configure 能力子集、缺少可选能力不发送命令、不污染连接，以及错误模型能力触发协议故障。真实 CPU 子进程测试增加 Hybrid 分发/续写，重复创建关闭轮换两种后端。

首次 ChatCodec 结果为 24/25：新增历史测试错误地假设会剥离旧 assistant reasoning。实际 `ChatTemplateOptions::preserve_thinking` 默认 true，且模板引擎显式传入该值；本批没有改变它。测试改为逐 token 精确验证 inline 与独立 reasoning 字段产生相同的保留历史 prompt。原失败和 Actual/Expected 诊断保存在 `hybrid-remote-cpu-details.log`、`hybrid-history-diagnostic-cpu.log`；修正后单独 25/25 和完整四项 CPU 门禁均通过，没有放宽匹配要求。

### 真实 GPU IPC

所有 GPU 运行串行执行，设备仍为 Intel Arc A770、驱动 `32.0.101.8974`，两个 KV 环境设置均为 0，context 固定 512。共用 `AilaQwen3IpcSessionTests` 可执行文件；CTest 的 Hybrid alias 传 `--backend qwen35_nf4`。

| 模型 | 全套结果 | 耗时 | 证据 |
| --- | --- | ---: | --- |
| Qwen3.5-0.8B NF4 | 通过 | 76.72 秒 | `build-refactor/hybrid-remote-ipc-08b-tests.log`、`hybrid-remote-ipc-08b-details.log` |
| Qwen3.5-4B NF4 | 通过 | 116.12 秒 | `build-refactor/hybrid-remote-ipc-q3-4b-tests.log`、`hybrid-remote-ipc-q3-4b-details.log` |
| Qwen3-0.6B NF4 回归 | 通过 | 75.12 秒 | 同上 Q3/4B 合并日志 |

每个后端先用直接 `RequestState::Step()` 生成四个独立参考，释放全部直接模型/会话资源后才启动 Worker。Hybrid 初始 prefill 块为 1/8/16/32，续写块为 1/3，生成 20/23/26/29 token 后各追加 5 token；混合 greedy、独立 seed 和 penalties。1/2/4 并发的全部生成与续写 token 必须和各自参考精确一致，同时核验容量、device reservation 和实际 contextLength。

另验证取消 A 后 B 保持参考结果、未读的 256-token 请求背压与私有排空、两轮填满四槽后全部释放重建并轮换 prompt/seed、父句柄释放后请求继续完成，以及旧 Worker 退出后新 epoch reload 的生成/续写。该对照证明远端传输、排程与直接执行一致；它不是另一个独立算子实现或 Transformers 的数值参考。

| Hybrid 模型，context 512 | 每 Session device cache/state payload | 每 Session host mirror payload，零快照 |
| --- | ---: | ---: |
| 0.8B | 26,492,928 B | 20,201,472 B |
| 4B | 69,468,160 B | 52,690,944 B |

远端没有 Reset/Truncate 命令，因此固定 `maxRecurrentSnapshots=0`。这些数字来自模型公共预算查询；host mirror 仍必需且仅受最多四个 Session 的数量限制，不在 IPC 的 device 字节额度内。进程内默认四快照的回滚行为保持原状。payload 不含权重、共享 scratch、allocator 和 oneDNN/driver 分配，不代表总 RSS 或显存。

### 真实 HTTP/SSE socket

Server 使用 context 512、四请求槽、最多 256 生成 token、1 MiB 输出预算、64 KiB Trantor 门限。扩展后的 runner 默认执行全套，新增多轮文本与 thinking false/true 的 JSON/SSE 对照；下面各报告均为 `full_suite=true`，没有排除用例。

| 模式/模型 | 结果 | 用例合计耗时 | 证据目录 |
| --- | --- | ---: | --- |
| Hybrid fake Worker | 12/12 | 22.32 秒 | `tmp/refactor/hybrid-http-fake` |
| Hybrid 0.8B GPU | 8/8 | 27.05 秒 | `tmp/refactor/hybrid-http-08b` |
| Hybrid 4B GPU | 8/8 | 36.29 秒 | `tmp/refactor/hybrid-http-4b` |
| Qwen3-0.6B GPU 回归 | 8/8 | 23.78 秒 | `tmp/refactor/hybrid-http-q3` |

全部覆盖健康/模型/参数、分片 UTF-8 SSE、JSON/SSE 的完整 content/reasoning/finish/usage 一致、1/2/4 并发独立 seed、multi-turn 历史和 text part 归一化、thinking 模式切换后新请求的隔离、Connection close/100 Continue、复用/pipeline、断连隔离与活动请求关闭。HTTP 不暴露原始 token ID，逐 token 参考来自前述 IPC 门禁；没有把 JSON 与 SSE 相等解读为答案质量评估。

Fake 额外验证慢读暂停后恢复、持续停读取消且不影响 B、429 与槽位复用，以及 Worker 崩溃的单次错误终态。观察到最大 **63,885 B pending / 4,261 B handoff**，分别低于 65,536 / 16,384 B 门限。主进程最后关闭 408/408 个连接；0.8B、4B、Q3 分别关闭 146/146、235/235、146/146 个连接。全部 `framework_errors=[]`，正常运行无强制 kill，活动请求与 IPC outstanding/pending 均归零。独立 fake 故障用例仍以非优雅关闭保留 Worker 崩溃语义。

本地 Hybrid checkpoint 的 `tokenizer_config.json` 没有内嵌模板，服务显式使用项目 `qwen35-fixed-v20`；不读取独立 `.jinja`，也不宣称与其 prompt token 相同。默认 `preserve_thinking=true`、文本开关覆盖、checkpoint 模板优先、双 EOS（248046/248044）和纯文本边界有独立 CPU 测试。vision 元数据用例只验证 Tokenizer/HTTP 的文本边界，不代表媒体或 vision 权重的远端验收。

### 身份、证据与后续边界

提交前 metadata 的 Git 身份仍为 `8ec6a3e`，实际候选源码、diff、三模型文件哈希、二进制、依赖、驱动与日志记录在 `tmp/refactor/hybrid-remote-precommit-provenance.json`。提交后的匹配重建身份与补充验收另存 `tmp/refactor/hybrid-remote-final-provenance.json`；每个快照复制相应构建日志，原第 9–11 节证据和本轮失败诊断均保留。

以上耗时包含启动、冷/JIT 和验证工作，不是吞吐或延迟性能基准。本批仍为本机纯文本、一个共享模型、最多四个交替推进请求；长上下文、长期创建释放、完整内存预算和服务延迟分布属于后续 P6，BF16 权重策略、工具/媒体/thinking budget、远端回滚和连续合批尚未开放。
