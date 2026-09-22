# 重构验证入口与 P0 基线

**当前状态：** 用户在 2026-09-09 晚间完成 DDU 清理、全新安装 8860 WHQL 及重启后，GPU 验证已恢复。第 17 节记录 Qwen3 0.6B 六轮、788 个请求通过及无新增故障的有限结果。此前 8974 上的 `0x116/0x139` 和压测退出后复发根因仍未定位，详见 [系统崩溃调查](stability-incident-20260908.md)；P6 跨模型长稳与 P7 设备验收尚未完成。各节结果仅适用于对应源码、驱动和场景。

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
| GPU / 驱动 | 历史基线为 Intel Arc A770 / `32.0.101.8974`；2026-09-09 晚间 DDU 后的 8860 / Level Zero 1.28.2 验证见第 17 节 |
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

## 13. 2026-09-08 稳定性事件与 CPU 修复验证

后续 P6 压测期间的四份转储已完成采集、SHA-256/事件参数核对及 CDB 离线分析：三次 `0x116 VIDEO_TDR_FAILURE`，一次 `0x139/2` stack-cookie fail-fast，四份均涉及 Intel 显示内核驱动 `igdkmdnd64.sys`，对应版本 `32.0.101.8974`。用户当时仅在 Aila 压测遇到同类故障；9 月 9 日压测退出后的新复发见第 16 节。故障模块与 bucket 不能直接证明独立驱动缺陷，Minidump 未恢复具体 Aila kernel、提交序列或释放动作。

实际转储时间（北京时间）为 14:18:28.859、17:00:21.517、17:45:01.931 和 18:06:05.926，不能用重启后的 WER 报告时间替代。后两次与两轮 Qwen3 HTTP 压测有绝对时间关联；两轮都未完成 512 请求及最终清理门禁，不能计为长稳通过。14:18 的 Hybrid 0.8B 未完成 soak 是未确认候选，17:00 尚无确定的对应运行。综合证据为 `tmp/refactor/stability-20260908/dump-analysis.json`；具体边界、修复和分步恢复方案见 [稳定性调查](stability-incident-20260908.md)。

暂停 GPU 后完成了显式 SYCL 异步错误交付、NF4 融合加载异常排空、机器守卫和 runner 收尾修复。Release Worker/全部目标与普通 MSVC Server 编译成功；RuntimeFault 9 组、根目录 7 个 CPU CTest、Server 4 个 CPU CTest 及 Python 72 项均通过。精确日志和源码/二进制哈希见调查文档；这些结果不等同于系统崩溃根因已消除。管理员采集后仅分析本地证据并继续静态审查，没有启动 GPU、解除隔离或更改驱动。P6 仍未验收，P7 设备验证暂停。

## 14. 2026-09-09 执行阶段记录与 CPU 验证

共享推理增加默认关闭的持久阶段记录，覆盖 Model/Session、Forward、现有队列等待、阻塞复制、工作区和 USM 释放；不增加 GPU 同步点。`backend_forward` end 仅表示主机返回；最终 `trace_close` 在原生 Context 成员完全析构后记录。写入失败停止追加、拒绝后续工作，但不跳过已有排空和释放。最终关闭行可读而 flush 失败时，进程级失败标记在 writer 析构后继续存活，Worker 在 Stop 后返回非零。实现约定与完整测试清单见 [执行阶段记录](execution-tracing.md)。

| 验证 | 实际结果 |
| --- | --- |
| Release 全部已配置目标 / 普通 MSVC Server | 88 / 41 个构建步骤成功，Server 32 个 object 的 header 依赖有效 |
| 根目录 CPU CTest / 普通宿主 CPU CTest | 9/9（12.23 秒）/ 4/4（10.64 秒） |
| 新增 writer / fake Worker pipe 故障注入 | 10 / 9 组通过；pipe 在两种宿主构建下均通过 |
| Python 离线解析 / guard 与 runner | 37/37 + 81/81，共 118 项 |
| C++ → Python 真实日志交叉验证 | 正常 18 行通过；中断 3 行拒绝；最终 flush 失败 4 行虽然完整，仍按退出码 5 拒绝 |

本轮只运行 CPU 与 fake Worker，依赖检查未发现验证程序导入 SYCL/oneDNN/Level Zero。所有记录保存在 `tmp/refactor/stability-20260909`，候选源码、二进制、测试与原始日志 SHA-256 见 `implementation-validation.json`；首次链接失败日志和旧 11 份验证日志均保留。机器守卫及暂停标记内容保持，未运行真实 GPU probe。尚不能确认 `0x116/0x139` 的具体 Aila 触发点或根因已经消除，P6/P7 的设备验收继续暂停。

## 15. 2026-09-09 GPU 有界恢复与取消诊断

用户授权恢复 GPU 后，在管理员 CLI 中完成实时系统证据审阅和 89 项源码/产物/模型校验，沿用第 14 节的 C++ 二进制、A770 / `32.0.101.8974` / oneAPI 2026.1.0。恢复范围为 Qwen3 0.6B NF4、context 512、每次完整生成最多 8 token；每轮最多 60 秒，清理另限 15 秒，device/pinned/session-host 分配上限为 4 GiB/16 MiB/256 MiB。

| 阶段 | 结果 | 请求数 | 执行 trace |
| --- | --- | ---: | --- |
| 首轮串行探针，应用期限 15 秒 | 失败；首个 prefill 加采样耗时 15.754 秒，服务按期限取消 | 1 | 8,744 条，完整，无错误 scope |
| 审阅后相同探针，应用期限 45 秒 | 两次生成与独立参考一致 | 2 | 9,250 条，完整 |
| 单次断连取消与新 Session 复用 | 两次完成、一次断连取消；复用输出与原参考一致 | 3 | 9,644 条，完整 |
| 默认路径 1 并发短回归 | 通过 | 4 | 关闭 |
| 默认路径 2 并发短回归 | 通过，核验请求时间重叠 | 8 | 关闭 |
| 默认路径 4 并发短回归 | 通过，核验请求时间重叠 | 16 | 关闭 |

首轮失败及其 guard 记录保留，未计为通过；仅在确定这次应用期限失败、确认设备清理完成和系统快照不变之后，显式记录审阅并修正期限。整轮上限和工作负载未扩大。五个通过运行共 33 条请求，其中 32 条完整、1 条有意断连；各次均零退出、无强制 kill、guard run id/结果哈希匹配、HTTP/IPC/Job 归零，事后快照可读且没有新增故障。取消 trace 按串行创建顺序确认中间 Session 析构结束后才创建后续 Session。

新增 `--probe-cancel` 仅允许带执行跟踪的单会话 probe，默认取消目标上限 16 token，不进入长生命周期循环；目标提前完成、被超时取消及复用语义改变都拒绝通过。新增 7 项 CPU 检查后，guard/runner 88/88、解析器 37/37，共 125 项通过。此次只更改 Python runner/测试和文档，没有重新修改或构建 C++ 推理二进制。

证据目录：`tmp/refactor/gpu-resume-20260909T063259Z`。其中 `request-timeout-review.json` 保留首轮失败审阅，`successful-probe-validation.json`、`cancellation-validation.json`、`cancellation-trace-lifecycle.json` 与 `bounded-concurrency-{1,2,4}-validation.json` 保存验收，最终身份与原始产物摘要见 `resume-validation.json`。9 月 8 日转储、9 月 9 日初始 CPU 验证和本轮失败均未覆盖。

三档短回归各自只有两轮 warmup 和一轮测量，记录可用于有限正确性/收尾检查，不能建立可靠性能分布或长期内存稳定性结论。没有恢复长上下文、512 次生命周期、Hybrid/BF16 或 P7 GPU 验证；P6 长稳仍未通过，原始整机崩溃根因仍未确认。

## 16. 2026-09-09 压测退出后的黑屏复发

最后一次 runner 于北京时间 15:18:03.819888 零退出，owned Job 为空、服务优雅关闭；15:48:18 的系统快照尚未发现新增证据。新的 Live dump 将 GPU 引擎超时定位到 15:51:11.555，系统 Minidump 将 `0x116` 定位到 15:51:21.190，距 runner 退出 33 分 17.370112 秒。两份转储的 TDR context 相同，均涉及 `igdkmdnd64.sys 32.0.101.8974`。

LiveKernelEvent 在崩溃前生成的 WER 附件含 226 个进程，没有 Aila、AilaWorker、AilaServer、Python 或最后一轮两个 owned PID。它支持当时没有运行本轮压测，但不能排除早先负载的延迟影响。转储缺少关联进程对象页面，具体触发进程及最终根因仍未知。

机器守卫已重新持久阻断，GPU 暂停标记为 true，新的故障证据没有 acknowledge。本轮以现有 CDB、本地符号和 CPU 文件校验完成取证；16 个文件副本校验一致，崩溃前验证快照的 437 项引用在更新事故文档前全部重新核验通过。原快照和短运行结果保持原样，当前 P6 系统稳定性未通过，P7 GPU 验证继续暂停。

证据目录为 `tmp/refactor/stability-recurrence-20260909T081320Z`；机器可读结论见 `incident-analysis.json`，详细边界和下一步见 [复发调查](stability-recurrence-20260909.md)。

## 17. 2026-09-09 DDU 重装 8860 后恢复压力验证

用户完成安全模式 DDU 清理、全新安装 8860 WHQL 并重启。本机驱动、内核文件和 Worker 实际加载的设备运行库为 `32.0.101.8860`，系统 loader / tracing / validation DLL 及 SDK 为 `1.28.2`；旧 8974 DriverStore 目录不存在。114 项非文档源码、产物、应用依赖和模型文件匹配此前验证，沿用 oneAPI 2026.1.0，没有修改或重新构建推理代码。

Qwen3 0.6B NF4 六轮全部通过：串行探针 2 请求、取消/复用 3 请求、1/2/4 并发短输入 53 请求、混合输入 106 请求、生命周期 528 请求、接近上下文上限的 4 并发 96 请求。合计 **788 请求、527 完成、261 主动断连取消、0 失败、31,808 个完成 token**。正式生命周期部分为 512 请求，完成/取消各 256；前后空闲取样中 Worker 私有内存增加 16.313 MiB、句柄减少 6，Server 增加 1.828 MiB、句柄增加 2，均通过事先设置的 128 MiB / 64 句柄增长门禁。

上下文上限为 512/2048，实际输入为 27、632、1848 token，完整生成上限为 8/32/64。每轮独立参考输出一致、优雅退出、owned Job 排空、guard/result 哈希匹配，各轮和最终快照无新增故障。两轮 trace 分别为 9,250 / 9,634 条且完整；相同短探针的输出哈希及 usage 与 8974 下的历史结果一致。六轮 runner 累计 613.768 秒，包含启动及收尾，不是连续 GPU 忙碌时长。

最终守卫为 `finished`，暂停标记为 false。有限验证不能独立确定此前黑屏根因，也不能代替较大模型、Hybrid/BF16 和持续数小时的系统稳定性验收。证据目录为 `tmp/refactor/gpu-8860-20260909T103826Z`，汇总见 `validation-summary.json`，完整范围和安装诊断记录见 [8860 验证结果](gpu-stability-8860-20260909.md)。

## 18. 2026-09-09 Qwen3.5 BF16 接入 IPC 与 C++ HTTP 服务

本轮将已有的 `LoadedModel::LoadQwen35Bf16` 共享会话接到 Worker 工厂、普通 C++ `InferenceClient::LoadQwen35Bf16` 和 Server 的 `--backend qwen35_bf16`。新增独立能力 `qwen35_bf16_bf16_kv`，复用原有 model-load 字段、设备/host 预算、生命周期执行器和 Scheduler。远端 Hybrid 的 recurrent snapshots 仍固定为 0，保留独立 host mirror。HTTP 使用 Qwen3.5 模板、reasoning 和模型名 `qwen3.5`。

默认 NF4 入口、protocol 2 / ABI 1 以及默认 requiredCapabilities 保持原约定。CPU 用例验证 Qwen3-only 和 Hybrid NF4-only 旧能力集合均拒绝 BF16 加载且不发送 `model.load`；连接仍能加载其支持的 NF4 后端。模型响应给出错误权重能力时，Client 判定协议故障。没有更改推理内核、驱动、SDK 或 TDR 设置。

Worker 完整 Release 构建与独立 MSVC Server 构建通过。Server 的 32 个 C++ 对象具有有效 Ninja 头文件依赖，PE 导入不含 AilaLib/SYCL/oneDNN。CPU CTest **21/21** 通过：根项目 CPU 11 项、Server 4 项、v1 兼容门禁 6 项；guard/benchmark Python **88/88** 通过。BF16 fake HTTP socket **12/12** 通过，覆盖背压、慢读恢复/超时、429/槽位复用、断连隔离及故障关闭。

真实 GPU 继续使用 A770 / `32.0.101.8860` / loader `1.28.2`，应用构建为 oneAPI 2026.1。下列测试独占机器守卫，进程及其子进程在启动前归入 Windows Job，并带整轮 watchdog：

| 验证 | 模型 | 结果 | 整轮秒数 |
| --- | --- | --- | ---: |
| IPC 默认五组 | Qwen3.5 0.8B BF16 | 5/5；1/2/4 会话、续写、取消/背压、四槽重建、父释放与新 epoch | 73.178 |
| HTTP/SSE | Qwen3.5 0.8B BF16 | 8/8；JSON/SSE、固定 seed 并发、多轮/reasoning、连接及关闭 | 31.199 |
| IPC 原有后端回归 | Qwen3.5 0.8B NF4 | 默认五组 5/5 | 73.926 |
| HTTP 默认后端回归 | Qwen3 0.6B NF4 | 8/8 | 30.452 |

整轮时间包括启动、冷 JIT 与收尾，不是纯推理性能。两轮 IPC 都以各自权重格式的直接执行作为参考，生成和续写逐 token 精确相同；直接模型释放后的 Context device/pinned USM 字节数及 allocation 数量归零。这些对照验证远端执行与会话隔离，未替代独立参考引擎的模型数值验收。

另运行 BF16 HTTP 混合输入 benchmark：context 2048，实际 prompt 为 **27 / 665 token**，每请求生成 32 token，1/2/4 并发，每种 profile/并发先 warmup 2 轮再测量 3 轮。**78/78 请求完成、0 失败、2496 个完成 token**，其中 reference 8、warmup 28、measured 42；整轮 54.115 秒。所有输出与独立串行参考的内容/reasoning 哈希及 usage 一致。

显式额度为 device **8 GiB**、pinned host **16 MiB**、全部 Session host **256 MiB**，HTTP memory 正确回显。模型加载完成时的显式 device residency 为 **4,761,273,536 B**；context 2048 下每 Session 的 device/host 预留分别为 **45,367,296 / 20,201,472 B**。这些是 Context/Session payload 计量，不包含完整进程与驱动分配。

| 输入 profile | 并发 | 测量请求 | TTFT p50，ms | 端到端 p50，ms | 聚合完成 token/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| short | 1 | 3 | 69.961 | 394.318 | 82.463 |
| short | 2 | 6 | 69.938 | 544.296 | 116.320 |
| short | 4 | 12 | 116.137 | 993.674 | 117.181 |
| mixed | 1 | 3 | 86.031 | 420.990 | 69.220 |
| mixed | 2 | 6 | 264.158 | 795.475 | 78.365 |
| mixed | 4 | 12 | 478.258 | 1577.007 | 80.497 |

聚合速率为测量阶段的完成 token 总数除以 batch 时间跨度之和，包含 prefill、生成和 HTTP/IPC 交付；reference、warmup 及冷启动不计入表格。每档仅 3 轮，适合作为可复现的服务基线，不能建立稳定的尾延迟分布。

各轮正常退出、owned Job 排空，没有强制结束；guard 与结果哈希一致。各轮及最终收尾快照无新增 GPU 故障，最终守卫为 `finished`，没有残留 Aila 或测试进程。证据目录：`tmp/refactor/p6-bf16-service-20260909`；`validation-summary.json` 汇总通过条件，`before` 保留本轮改动前的源码和已验证二进制，`candidate` / `candidate-manifest.json` 保存最终源码与产物，`implementation.diff` 区分本轮改动，`working-tree.patch` 保留累计 tracked 改动，`evidence-manifest.json` 记录证据文件哈希。

本轮完成 BF16 远端接线及有限集成验收。公共投影策略的进一步统一、较大模型与接近容量上限的跨模型输入、长期资源稳定性、远端 Reset/Truncate 和 P7 连续合批仍待推进。

## 19. 2026-09-09 Hybrid 公共层执行与独立旧产物对照

两个 Hybrid 存储后端现在共用有类型的 `HybridLayerPlan`、执行视图、阶段 profiler 和约 430 行层执行模板。NF4/BF16 各自保留投影策略：NF4 原融合 gate/up GEMV 与 scratch/residual 参数，BF16 原 JM decode、专用 FFN 和 Linear fallback。Full Attention、DeltaNet、RoPE/KV、跨层 residual/norm 的执行顺序保持。源码边界和使用方式见 [公共层执行说明](hybrid-layer-execution.md)。

新增 `AilaLayerReferenceTests`，在生产实现变更前使用旧 AilaLib 编译并保存可执行文件，捕获不可覆盖的完整 float logits 参考。候选版本逐位比较全部 logit、实际 cache/checkpoint 长度及固定 seed/penalty 的生成 token；两次运行的模型文件 SHA-256 和相关环境一致。该测试不在候选版本中重新生成 oracle。

| 模型 | 比较行数 | 逐位一致的 float 数 | 旧/新显式 device 峰值，B |
| --- | ---: | ---: | ---: |
| Qwen3.5 0.8B NF4 | 34 | 8,442,880 | 1,962,253,610 |
| Qwen3.5 0.8B BF16 | 34 | 8,442,880 | 4,816,942,532 |
| Qwen3.5 4B NF4，分组 DeltaNet | 34 | 8,442,880 | 3,967,047,655 |
| Qwen3 0.6B NF4，回归对照 | 33 | 5,013,888 | 1,315,125,146 |

合计 **135 行 / 30,342,528 个 float**，零逐位差异；容差为 0。场景包括两个 Session 的 1/3/7/11/16/64 行交錯 prefill/decode，reset，64+7 行后向 66 截断并从实际 checkpoint 补算，缺失 checkpoint 回退，单 token 跨 64 边界的快照与回放，独立 seed/penalties、续写及销毁重建。每个用例模型及 Session 释放后，显式 device 分配回到 0；峰值保持不等于完整 WDDM/RSS 计量。

Worker/CLI 全 Release 与独立 MSVC Server 构建通过。CPU/兼容 CTest **21/21**：根 CPU 11、旧 Chat/IPC/Worker/Proxy 6、Server 4。BF16 真实 HTTP/SSE **8/8**，包括 1/2/4 并发、reasoning、多轮、断连隔离和带活动请求关闭。没有用运行整轮耗时宣称吞吐提升或性能保持。

初次启动前守卫发现 48 条新写入的 WER 事件，测试未启动。逐条核查发现 ReportId、EventName、P1–P10、AttachedFiles、StorePath 全部与已有历史事件一致，启动标识及转储清单不变，判定为旧报告补写后进行了带证据 ID 的显式确认。原阻断和审查记录分别保存在 `guard-blocked.json` / `guard-evidence-review.json`。随后的 8 次参考 capture/compare 与 HTTP 验证全部正常结束、owned Job 排空，无新增 GPU 故障。

环境继续为 A770 / 8860 WHQL / Level Zero 1.28.2 / oneAPI 2026.1。证据目录 `tmp/refactor/p3-hybrid-layers-20260909` 保留旧源码、旧可执行文件、不可覆盖的 reference、逐阶段日志与守卫结果。此次完成公共 Hybrid 执行提取；P6 长稳、完整资源准入与性能分布、P7 连续合批及其他模型驱动仍按 [完成审计](refactor-completion-audit.md) 继续推进。

该目录现已补齐 `validation-summary.json` 与 149 项文件的 `evidence-manifest.json`；保存的 16 项 candidate 源码/产物、四份 reference 与所有成功 run/guard 的哈希重新核验一致。

## 20. 2026-09-09 执行资源额度与 BF16 分组回归

8860 上串行完成 8 个 GPU fixture。共用机器守卫、启动前 Windows Job 归属与整轮 watchdog；各轮零退出、Job 排空，守卫正常完成。

| 验证 | 实际结果 |
| --- | --- |
| 执行资源 | 7 组通过：Context 额度与顺序、typed workspace、采样隔离/复用、共享 scratch、Linear scratch 重绑、masked-attention cache、runtime memory |
| Qwen3 0.6B NF4 额度 | 加载拒绝后恢复、活动模型改额度 Busy/原子性、Session device 上限与回收、pinned sampling 拒绝后 peer 继续、大 prefill 局部拒绝后 peer 继续 |
| Qwen3.5 0.8B NF4 额度 | 同上五组均通过，large prefill 实际触发额度拒绝 |
| Qwen3.5 0.8B BF16 额度 | 同上五组均通过，large prefill 实际触发额度拒绝 |
| LoRA fused / oneDNN | 两条路径均通过独立 CPU 数值、scratch 持有、增长/复用及析构回收 |
| BF16 grouped DeltaNet | 16 key / 32 value heads、d128、kernel4、hidden128；host 递归参考与 GPU 对照共 86 步，changed logits=0、max abs=0 |

三个模型额度 fixture 均无 `NOT_COVERED` 分支，模型/会话析构后显式 device/pinned allocations 归零。BF16 grouped fixture 的 projection 仍在 GPU 执行，仅递归状态计算采用独立 host oracle；覆盖非专用 JM decode 的 Linear fallback、prefill/decode、Session 隔离、reset/替换和续写。history-order 负对照实际改变 127 个 logits，证明用例对历史顺序敏感。

证据目录 `tmp/refactor/p6-resource-validation-20260909`，8 个 result 与守卫引用哈希校验一致，`validation-summary.json` 和 51 项 `evidence-manifest.json` 已生成。范围为显式资源额度和上述有限正确性场景；库/驱动私有分配、长期增长、近容量上下文和性能分布仍需要后续验收。

## 21. 2026-09-09 Request → IPC → HTTP/SSE thinking budget

请求级状态机现已支持独立 thinking 预算与 tokenizer 结束序列，IPC 新增可选 `thinking_budget_v1`，HTTP 接受三个互斥别名。强制 token 不消耗 RNG，正常计入输出、penalty history 和 KV；无法容纳完整结束序列时提前 Length，并正确传递到 Scheduler、Client 和 HTTP usage。接口与边界见 [请求级预算](thinking-budget.md)。

完整 Worker/CLI 与独立 MSVC Server 构建通过。核心 CPU CTest 11/11、Server CTest 4/4；新增 CPU 状态/协议/Worker/Client 用例覆盖零预算、自然关闭后再次开启、强制关闭、有限输出、取消、非法 marker、旧 peer 能力拒绝及资源回收。socket fake 全套 13/13，另有预算专用用例通过；fake 使用真实 Request/Worker 控制器和独立 tokenizer，精确校验 2/5 token reasoning、关闭序列的 usage，以及 3 token 上限提前输出 2 token 后结束。

| 真实后端 | IPC | HTTP/SSE |
| --- | --- | --- |
| Qwen3 0.6B NF4 | 四预算与独立执行精确 token 对照、续写/KV/seed，普通 1/2/4 会话回归通过 | 0/2/5/-1 四预算并发与独立 JSON 完整语义/usage 一致，短输出与非法别名通过 |
| Qwen3.5 0.8B NF4 | 同上 | 同上 |
| Qwen3.5 0.8B BF16 | 同上 | 同上 |

IPC 零预算的短容量场景实际以 Length/0 输出结束，完整 prompt 已消费，随后追加请求与独立执行一致。三个直接参考的模型/会话释放后，显式 device/pinned 字节数和 allocation 数量归零。HTTP 关闭标记经过现有 reasoning parser，未泄漏到 content/reasoning 文本。

默认无预算路径再与第 19 节四个模型的不可覆盖旧产物参考比较：135 行、30,342,528 个 float 逐位相同，fixed seed/penalty 和 checkpoint/cache 状态一致。10 个 GPU run 全部零退出、owned Job 排空、guard finished，无新增 GPU 故障。本轮没有改 kernel，不以这些有限回归宣称长期稳定或性能提升。

证据目录：`tmp/refactor/p5-thinking-budget-20260909`。保留初次编译的测试字段名错误与能力清单断言失败日志，修正后完整构建及测试通过；未将失败日志覆盖为成功。

## 22. 2026-09-09 Engine 兼容外观实现提取

`Engine.hpp` 中 74 个函数体移到 `src/engine/` 的 8 个实现文件，公共头文件缩为 457 行；ASR 文本 helper 的声明保留在独立头文件。逐函数核验移动前后函数体 SHA-256 相同，未更改计算、生成循环、缓存或回调行为。默认参数保留在声明，类成员布局与原 heavy include 暂时保留。

完整 Worker/CLI Release 构建通过。兼容 CTest 日志显示 10/10，其中正确计入 CPU/旧 ABI 的是 9 项：Chat、Yolo26ModelSpec、EnvUtils、LoraLoader、IpcProtocol、WorkerDispatcher、RuntimeIsolation、ProxyAbi 和 ProxyLogging。另一个 DiffusionGemmaModelSpec 实际执行 GPU，当时误归类且未使用守卫；已补 `gpu;models;ops` 标签和 timeout，并在守卫下独立重跑通过。该误运行不计作受守卫验收。

| 独立旧 CLI 对照 | 结果 |
| --- | --- |
| Qwen3 0.6B NF4 | 完整 assistant JSON 完全一致 |
| Qwen3.5 0.8B NF4 | 完整 assistant JSON 完全一致 |
| Qwen3.5 0.8B BF16 | 完整 assistant JSON 完全一致 |
| Qwen3.5 0.8B NF4 structured JSONL | 24 条记录完全一致 |

各文本对照固定 seed 921、24 输出 token 上限、context 512、thinking budget 4 和同一多轮输入，比较 content、reasoning、raw_text、finish_reason、warnings 与 metadata。此门禁验证提取保持行为，不用于评价生成答案质量。Diffusion fixture 的 empty quant-state loader max abs diff 为 0.0341584，proportional RoPE 通过，masked attention max relative diff 为 0.000301152；不代表完整 Diffusion 或音视频端到端已验收。

旧 CLI 的首次 NF4 运行以 `0xC0000135 / STATUS_DLL_NOT_FOUND` 退出，stdout/stderr 均为空，具体缺失 DLL 未捕获。当时保存的 exe 旁缺少运行库，且与 library build 有时间重叠。核查没有新增 GPU OS evidence 或 dump，boot 未变，Job 正常排空；据此不能把它认定为驱动 TDR。随后复制并逐个核验 34 个运行库至旧 exe 旁，保留原 exe 哈希，停止构建与 GPU 基线并行。显式审查证据 ID `ac20b77008d9ffb5576792540bdaf4dd20ef303fa9505656d2de530fb006eedd` 后，单次重试 `before-nf4-02` 通过。

四个旧参考、四个候选比较和修正后的 Diffusion fixture 共 9 次成功 GPU run，全部零退出、Job 排空、守卫 finished。原失败目录、审查与 acknowledge 记录保留；最终成功守卫为 `5a092fd95d7e428487a3c7476a82943d`。环境仍为 A770 / 8860 / Level Zero 1.28.2 / oneAPI 2026.1。没有性能或长期稳定性结论。

证据目录：`tmp/refactor/p1-engine-extraction-20260909`。`body-movement.json` 记录原位置及函数体哈希，旧/新产物、运行日志、候选 manifest 和 `validation-summary.json` 保留本切片的边界与验证结果。

## 23. 2026-09-10 异步 token C ABI

新增 `include/aila_inference.h` 和 host-only `AilaInference.dll`，复用已有 Client → IPC → Worker → Scheduler。20 个未修饰 C 导出提供 Client/Model/Session/Request、独立选项和事件读取。4096 槽注册表在远端副作用之前预留句柄，代际与类型校验拒绝旧句柄；读取先校验输出结构，重叠 reader 返回 Busy，release 不等待 reader 并立即触发私有取消/排空。内部 Client 的终态诊断字符串复制移到消费标志之前，保留分配异常后的读取机会。

完整 Release 构建通过。核心 CPU CTest **13/13**、旧 Chat/IPC/Worker/Proxy 兼容 **6/6**；新增 ctypes fixture 内含 **11 组**独立场景：三后端四会话及续写、预算 -1/0/2/5、短容量、非法结构不丢 token、NaN/无穷/非法 token、Busy、阻塞读取与释放、慢消费者、父先释放、句柄额度/槽位复用、Worker crash/旧 epoch，以及反复生命周期的 OS handle 回收。

公开结构布局由独立 C 翻译单元与 Python ctypes 两端校验，头文件恢复调用者的 packing。另使用原生 MSVC 的 C11/C++17 编译并链接同一 import library；在仅含 Windows 系统路径的 PATH 下，从隔离 SDK 目录运行通过。该目录只有测试 exe、AilaInference.dll 和三个 MSVC runtime DLL，没有 SYCL/oneDNN/Level Zero；运行真实 CPU Worker 的 ctypes 进程也检查了未加载 GPU 模块。release target 已接入 DLL/header/import library 和 host-only 依赖收集，实际依赖收集在独立 SDK 目录验证。

| 真实 GPU 后端 | 结果 | 加载完成时显式 device bytes |
| --- | --- | ---: |
| Qwen3 0.6B NF4 | 五组通过 | 1,192,083,094 |
| Qwen3.5 0.8B NF4 | 五组通过 | 1,821,201,446 |
| Qwen3.5 0.8B BF16 | 五组通过 | 4,760,966,336 |

每轮先通过内部 Client 独立运行参考，确认 Worker 退出后再启动 C ABI Worker。context 512、四种不同长度输入、seed 921–924、sampling/penalties、独立 thinking budget、16–19 个初次输出及续写，所有 token、usage 和实际 cache length 完全一致；取消 A 后 B 的输出仍与参考一致，释放后复用 Session 槽及父先释放后的子请求通过。显式限额 device 8 GiB、pinned host 16 MiB、Session host 256 MiB。表中的 device bytes 不覆盖库/驱动私有分配。

三轮 GPU 均由机器守卫独占、启动前归入 Windows Job，并带整轮 watchdog；全部零退出、Job 排空、无强杀。设备仍为 A770 / 8860 / Level Zero 1.28.2 / oneAPI 2026.1。此次验证 ABI 对已验收引擎的执行保持，不代替独立模型 oracle、性能分布或长期稳定性验收。

首次 Qwen3 启动被守卫阻断，GPU 进程没有启动：新增 240 条 WER 写入和 boot ID 变化。逐条核验 240 条 payload 与之前的 48 份历史报告完全相同，169 项 dump 清单不变；System 日志记录 9 月 10 日 00:35 开始菜单关机、6006/Kernel-General 正常关闭，11:22 正常启动，没有新的 41/6008/系统 bugcheck/Display reset 事件。保存审查后显式确认证据 ID `453b127e9dad3fc91e1c8d8f2e24b2f03943dd66e751d8d7dbaa8947a4b1a0f1`，在新目录 `gpu-q3-02` 执行成功；原阻断目录与状态没有覆盖。

证据目录：`tmp/refactor/p5-async-c-abi-20260909`，保留旧 Worker/Proxy 产物、所有 build/CPU/GPU 日志、PE 导出/依赖、MSVC 消费者、独立 SDK、守卫审查及候选哈希。接口约定见 [异步 C API](async-c-api.md)。远端 Session reset/truncate、旧 ABI 适配和工具/媒体语义仍未完成。

## 24. 2026-09-10 远端 Session reset/truncate 与重放

新增可选 `session_context_v1`、`session.reset` / `session.truncate`，复用现有 `SessionState` 操作与 `SessionInfo` 响应。C++ Client 和公开 C ABI 均返回实际恢复长度；公开导出由 20 个增至 22 个，结构布局、ABI version 1 和原 11 项 wire 命令形状保持。默认必需能力不变，旧 peer 的缺失能力在发送 RPC 前拒绝。

Worker 在排入生命周期执行器前预留 Session，预留持续到结果发布；同 Session 的 submit/重复变更被拒绝，query 返回 busy/null 并保持控制线程响应。变更占普通命令额度，保留 cancel/query/ACK/release 的控制额度。RPC 超时不撤销已提交操作；在途操作持有父资源，父释放和 shutdown 必须安全排空。生命周期线程发现设备故障即广播并停止后续排队工作；本地参数错误清除预留后仍可继续使用。

Release Worker/CLI、独立普通 MSVC Server 构建通过。根 CPU CTest **13/13**、旧兼容 **6/6**、Server CPU **4/4**，合计 23 项目标；设备故障发布顺序的最后修正后重跑的 4 项包含在上述范围内，不重复加总。新增门禁覆盖严格 wire 字段、实际 checkpoint 小于目标、非法响应身份/额度、未协商能力零 RPC、超时后的响应路由、排队即 Busy、独立 peer、父释放、关闭和故障后不再进入下一个 backend 操作。

ctypes 消费者现有 **13 组**场景，新增三后端 reset/truncate/replay、错误输出结构无副作用、Busy、父释放和代际句柄拒绝。独立 MSVC C11/C++17 消费者、仅 Windows PATH 的隔离 SDK 运行通过，22 个未修饰 C 导出；DLL 无 SYCL/oneDNN/Level Zero 导入。HTTP fake 完整 **13/13** 通过，包括 JSON/SSE、thinking budget、慢读、断连、准入、优雅关闭和 Worker fault。旧 `AilaShared.dll` 二进制未改变。

| 模型 | IPC 对照直接调用 | 公开 C ABI 对照独立 Client |
| --- | --- | --- |
| Qwen3 0.6B NF4 | 通过；目标/实际 0/0、21/21、51/51、79/79 | 通过 |
| Qwen3.5 0.8B NF4 | 通过；目标/实际 0/0、20/0、52/0、80/80 | 通过 |
| Qwen3.5 0.8B BF16 | 通过；目标/实际同 Hybrid NF4 | 通过 |
| Qwen3.5 4B NF4 | 通过；包含 grouped recurrent 状态；目标/实际同上 | 本切片未额外运行 |

四轮 IPC 使用 context512、四个独立 Session、不同 prefill chunk、固定 seed 和 penalties。先单独执行原进程内 `SessionState::Reset/Truncate` 并保存参考，再销毁全部直接设备 owner，启动隔离 Worker 对照。按实际 checkpoint 补算前缀后，完整生成 token、usage、实际缓存长度一致；再次 reset 的生成/续写、活动请求 Busy、取消后恢复、peer 隔离和四个槽位释放重建均通过。直接参考结束时 device/pinned 字节和 allocation 记录均归零。Hybrid 关闭快照时回退到 0 的诊断符合既定语义。

三轮 C ABI 验证四个 Session 的 reset、truncate-to-zero、thinking budget -1/0/2/5、seed 921–924、续写、Busy、取消 peer、父释放和槽位重用。其参考为独立 Client 运行；任意 checkpoint 的重放正确性由上述 IPC/直接执行门禁覆盖。这些对照验证调用与状态行为，不是外部模型 oracle 或完整 logits 数值验收。

7 轮全部零退出、owned Job 排空、守卫 finished，无新增 GPU OS evidence。A770 / 驱动 `32.0.101.8860` / Level Zero `1.28.2` / oneAPI `2026.1`，boot `2026-09-10T03:22:16.5000000Z` 保持；最终守卫 `6eca874dd1f341af820a2d501cd3c974`。本切片没有修改 GPU kernel、投影策略或 NF4 格式，不宣称性能提升、近容量或长期稳定。

证据目录：`tmp/refactor/p5-session-context-20260910`。保存各轮日志/launch/result/guard、CPU/HTTP/MSVC 记录、原始与候选源码/产物、独立差异和哈希清单。旧 C ABI 的请求/上下文适配、工具/媒体语义、跨模型近容量和长稳仍按 [完成审计](refactor-completion-audit.md) 推进。

## 25. 2026-09-10 旧文本 ABI → 共享 Session 的可选适配器

新增 `LegacyTextSession` 和只通过公开 C POD ABI 跨 MT/MD CRT 的 `LegacyInferenceBridge`。显式设置 `AILA_LEGACY_TEXT_V2=1` 后，纯文本 Qwen3 NF4、Hybrid NF4/BF16 复用 Model/Session/Request；宿主沿用 Tokenizer、ChatFormatter、历史、reasoning、工具策略和原调用线程回调。相同模型/运行目录/API 目录/容量/继承环境的四个 Engine 共享一个 Worker。默认开关为 0，完整兼容边界见 [适配器说明](legacy-text-adapter.md)。

新增可选 `prefill_only_v1`，输入补算不采样、不消耗 RNG、无 token 事件，成功为 Completed/Length、generated=0。取消后从查询到的实际长度补算。32 字节 `AilaInferenceRequestPolicy` 提供一次 thinking 强制关闭、host sampler 和旧 Hybrid prefill 输出行策略，原 152 字节 RequestOptions 不变；异步 C DLL 共 24 个入口。

### 独立旧产物与发现的兼容差异

修改前保存完整源码和可独立运行的 Worker/Proxy/设备 DLL，`before-manifest.json` 共 357 文件、514,298,922 字节。参考由该完整旧产物执行，候选使用新产物；两者分别启动并串行验证，不用新实现生成旧路径的期望值。

第一轮 Hybrid NF4 比较有 51 个字段不一致，尽管 GPU 运行与关闭本身通过。源码定位到原 Hybrid 对非初始 2–16 token 分块返回多行 logits，旧生成循环从第一行采样；显式 Session API 返回最后有效行。适配器新增 `legacy_prefill_logits_v1` 请求策略保留旧行为，NF4/BF16 仍使用原投影与 attention 计算。默认新 API/HTTP 不启用该策略。失败输出、364 文件的候选快照和诊断保存在 `legacy-comparison-nf4.json`、`hybrid-mismatch-candidate`、`hybrid-mismatch-review.json`；修复后的 NF4 比较为零差异。

### 最终验证

| 项目 | 结果与范围 |
| --- | --- |
| 最终 Worker/CLI 构建 | 102/102 增量步骤完成，`build-adapter-05.log` |
| 普通 MSVC Server | 构建通过，32 个 C++ 对象的 Ninja 头文件依赖有效 |
| CPU CTest | 15/15，66.01 秒；含 Request、Scheduler、协议/Client/Worker/真实 pipe、C header、async ctypes 15 组、legacy ctypes 10 组及 legacy C++ 11 组 |
| 旧兼容门禁 | 6/6，12.41 秒；Chat、v1 IPC、WorkerDispatcher、RuntimeIsolation、ProxyAbi、ProxyLogging |
| Server CTest | 4/4，16.58 秒 |
| HTTP fake | 13/13；thinking、JSON/SSE、1/2/4 并发、慢读恢复/超时、429、断连隔离、正常关闭和故障 Worker；正常服务退出码 0，无强杀、无框架错误 |
| 独立 SDK | 普通 MSVC C11/C++17 编译；仅 Windows PATH 的 host-only 目录运行通过；45 个旧 Proxy 导出集合完全一致，24 个未修饰异步 C 导出；两宿主 DLL 无 GPU runtime 导入 |

最终采用 9 轮有守卫的 GPU 运行：三种后端各 1 轮旧产物 capture、1 轮最终候选 capture、1 轮默认异步 prefill/policy 检查。设备为 A770，驱动 **32.0.101.8860**、系统 Level Zero **1.28.2**、oneAPI **2026.1**，context=512。最终候选目录为 `gpu-legacy-after-q3-02`、`gpu-legacy-after-nf4-02`、`gpu-legacy-after-bf16`；默认 API 目录为 `gpu-prefill-q3-03`、`gpu-prefill-nf4-02`、`gpu-prefill-bf16`。

| 后端 | 旧 ABI 完整对照 | 新适配器附加恢复检查 | 默认异步 API |
| --- | --- | --- | --- |
| Qwen3 0.6B NF4 | 11 比较项零差异 | 接受 6 token 后取消、上下文及后续 12 token 与独立参考一致 | greedy、host/close-once、实际 checkpoint 补算、四 Session prefill |
| Qwen3.5 0.8B NF4 | 11 比较项零差异 | 同上，包括 Hybrid 回退到 0 后补算 | 同上 |
| Qwen3.5 0.8B BF16 | 11 比较项零差异 | 同上 | 同上 |

每后端的 11 比较项包含 raw/history/reset、6 种 JSON 请求、完整结构化事件序列、四个独立 peer 的 JSON/上下文；旧 peer 串行参考与新四 Engine 共享 Worker 并发执行逐项相等。额外槽位重建也与原 peer 一致。BF16 使用从原多模态 checkpoint 派生的纯文本 config，权重为只读使用的 hardlink；原 config/权重未修改，来源和哈希见 `bf16-text-fixture-manifest.json`。这不构成视觉路径迁移验收。

CPU 回调测试故意让假后端在首次回调前执行完全部输出，覆盖精确截断、回退到 0、粗 checkpoint、回调取消/异常及 48-token 循环保护。真实 DLL 用隔离的 `host/AilaShared.dll`、`host/AilaInference.dll`、`host/aila_runtime/AilaWorker.exe` 布局，核验四 Engine 共享、4210-token 以上输入补算、Busy/peer 重入、回调内销毁、环境模板变更、Worker 环境隔离和 DLL 卸载。

关闭实现改为保留 I/O 线程句柄并真正 join，避免计数归零后 DLL 提前卸载而线程仍在退出。第一次生命周期夹具把 CRT 冷初始化句柄误判为泄漏：实测冷值 107，首次成功后 113 并在后续 5 次成功保持，首次失败后 121 并在后续失败保持。现先预热成功/失败，再验证 5 组混合循环共 10 次，要求不超过基线 +2；原失败与诊断日志保留。SDK 准备脚本首次因 Windows 环境键大小写查询失败，修正后独立运行通过，未涉及设备执行。

首轮启动前守卫阻止了 48 条新写入 WER；完整 EventData、provider/log/event ID 与已有历史报告逐条相同，只有记录号/写入时间变化，boot、转储 metadata 不变。审查记录后作一次显式 acknowledgement，未改守卫规则、未自动重试。见 `guard-blocked-delayed-wer.json`、`delayed-wer-review.json`、`delayed-wer-acknowledgement.json`。所有实际 GPU 运行均在独立 Windows Job 和 watchdog 下串行结束，无强杀、无剩余 owned PID，各轮 guard 为 finished，未出现新增故障证据。

证据目录：`tmp/refactor/p5-legacy-text-adapter-20260910`。`validation-summary.json` 汇总最终选择、旧/新源码产物及所有原始结果；历史失败不计为通过。归档后成功行写入导致 `finalize.log` 的原清单哈希过期，其余 1402 文件哈希核验不变；追加的 `evidence-manifest-sealed.json` 是包含 1405 文件的权威最终清单，审查见 `evidence-seal-review.json`。旧 Qwen3 capture 的 runner 记录了当时工作区 metadata 哈希，该临时版本未归档；实际启动的旧 Worker/Proxy/DLL 及其自身 build metadata 均保留并逐哈希核对，汇总明确区分这两种 metadata。此切片没有验收长期稳定性、性能分布或 P7 连续合批。当时旧 adapter 的 1024 输出上限在第 26 节扩展；4 Session 数量限制、decode/stream chunk 性能与 Worker 日志消费仍待补齐，其他项目继续按 [完成审计](refactor-completion-audit.md) 推进。

## 26. 2026-09-10 可选长输出额度与旧 adapter 接入

新增可选 `extended_generation_v1`，默认 hello 和生成上限仍为 1024。Client 显式申请更大额度时要求该能力，双方精确协商至最多 32768；不支持的旧 peer 在发送 configure 前拒绝，Worker 自定义的较低上限不会被扩展覆盖。协商响应不能静默降限。四 Session、总 outstanding tokens 32768、prompt 4096、prefill chunk 512、core ring 128 和 token window 32 均不变。

旧文本 adapter 按 `max(1024, min(context_capacity, 32768))` 申请，实际输出仍受剩余 context 与总预留额度约束。同一长 Request 保留 RNG、penalty history 和 thinking 状态，不靠多次提交拼接输出。公开 ClientOptions 仍为 104 字节，默认 wire schema、45 个旧 Proxy 导出和 24 个异步 C 导出保持不变。

| 验证 | 实际结果 |
| --- | --- |
| Worker/CLI 构建 | 41/41 增量步骤完成；普通 MSVC Server 构建及 32 对象头依赖检查通过 |
| 核心 CPU 门禁 | 15 项唯一 CTest 通过：首轮 14 项通过，修正夹具后单独复验 C API 1 项通过；未重新宣称完整第二轮 15/15 |
| 异步 ctypes | 16 组通过；三后端 fake 的 1100 token 精确 oracle、context 1102、续写；超限拒绝不改上下文、并发 token 预留拒绝与取消/ACK 后归还 |
| 旧文本 adapter CPU | 10 组通过；长输出提升到 1200 token，精确 oracle 及 context 1208 通过 |
| 兼容与服务 | 旧兼容 6/6、Server 4/4、HTTP fake 13/13 |
| 独立 SDK | MSVC C11/C++17 消费者通过；host-only 隔离目录与 GPU import 检查通过，原导出集合不变 |

首轮 async 长用例的 5 秒夹具超时不够：FakeBackend 每步 sleep 1 ms 在本机累计 1100 步实测约 12.2–12.4 秒，仅此长用例的 drain 改为 30 秒。另一处将协商额度拒绝误期待为 InvalidArgument，已改为既有的 ResourceLimit。`cpu-01.log`、`c-api-02.log` 保留失败，最终 `c-api-03.log` 与完整 16 组明细记录成功；没有修改生产超时或错误码来通过测试。

| 真实 GPU 后端 | 独立旧产物对照 | 新版完整请求与续写 |
| --- | --- | --- |
| Qwen3 0.6B NF4 | 固定 seed 921 采样，前 1000 token 零差异 | 1100 token 与第二个独立 Session 完全相同，两会话追加 2-token prompt 后的 16-token 续写一致 |
| Qwen3.5 0.8B NF4 | greedy，前 1000 token 零差异 | 同上 |
| Qwen3.5 0.8B BF16 | 固定 seed 921 采样，前 1000 token 零差异 | 同上 |

旧参考来自第 25 节冻结的完整 runtime，373 文件先核验哈希；本切片每轮保留实际运行库的 build metadata。context=1536，最多两 Session，未设置 stop IDs；逐事件验证连续 sequence、唯一终态、完整生成数和实际 context，关闭后 outstanding/reserved/pending 为零。BF16 沿用只读 hardlink 的纯文本 fixture，原权重和配置未改。额外的最后 100 token 与续写由第二个候选 Session 对照，不声称它们拥有旧产物 oracle。

6 轮合计生成 **9696 token**，均由机器守卫、Windows Job 与 watchdog 串行运行，零退出、无强杀、Job 排空、guard finished，各轮及最终快照无新增故障证据。环境仍为 A770 / 32.0.101.8860 / Level Zero 1.28.2 / oneAPI 2026.1。本切片不改 GPU kernel 或计算形状，不声称性能、长期稳定性或 32768-token 设备边界已经验收。

证据目录：`tmp/refactor/p5-extended-generation-20260910`。独立 baseline 引用、三份完整 prefix 比较、六轮原始日志/守卫、CPU 失败与复验、SDK、候选源码/产物和哈希清单均归档。旧 adapter 默认仍关闭；完整资源与最大上下文边界、Worker 日志桥接、分块性能以及其余路线图继续推进。

## 27. 2026-09-11 Worker 日志读取与旧 ABI 桥接

新增可选 `worker_log_control_v1` 和 `worker.set_log_level`。Client 提供独立的有界日志队列，默认 64 KiB，按消息最大转义开销保守记账；满时丢弃新日志并计数，不占 token credit。关闭或连接故障后允许排空，设备域 fault 本身不提前结束仍存活的连接日志。控制线程设置日志级别，无须等待当前 GPU Request 完成。

公开 C ABI 新增 `client_read_log` / `client_set_log_level` 两个入口和独立的 1096 字节 LogEvent；原 24 个导出、旧结构布局和默认必需能力保持。日志具有 Worker epoch 和连接级 ordinal，没有 Request 身份。同 Client 的重叠 reader 返回 Busy，输出结构在消费前校验；client release 唤醒阻塞日志 reader，已有子句柄的存活语义不变。

旧 adapter 每个共享 Model owner 使用一个日志读取线程，选择仍 active 的 Engine source 只转发一次。source 退休后由剩余 Engine 接管，沿用 ProxyLogging 的回调 lease 与退休等待规则。`aila_set_log_level` 在加载前及运行中同步到 Worker，同 owner 去重；关闭时先 shutdown，随后停止并 join 日志线程，最后释放句柄与 DLL。

| 验证 | 实际结果 |
| --- | --- |
| Worker/CLI 构建 | `build-01.log`，47/47；随后未修改生产 C++ |
| 核心 CPU CTest | 14 项唯一门禁通过：首轮 13/14，修正夹具后单独复验 C API 1/1 |
| 异步 ctypes | 完整 18 组通过；新增日志布局/级别、输出先校验、重叠读取、阻塞唤醒、排空与日志溢出隔离 |
| 旧 adapter ctypes | 13 组通过；四 Engine 日志不重复、级别切换/source 退休接管、阻塞 callback 不阻止推理、销毁等待 lease、callback 内销毁最后 Engine |
| 旧兼容与服务 | 旧兼容 6/6、普通 MSVC Server 构建及 32 对象头依赖通过、Server CTest 4/4、HTTP fake 13/13 |
| SDK/ABI | MSVC C11/C++17 消费者及 host-only 隔离运行通过；45 个旧 Proxy 导出不变、原 24 个异步 C 导出全保留，新增 2 个；两宿主 DLL 无 GPU imports |

首轮 C API 测试在检查 quiet 前遗漏释放两个已完成 Request；quiet 同时检查 retained handles，因此失败。补齐两次 release 后完整 C API 18 组通过，76.03 秒。未修改生产逻辑来适应夹具。原失败版本与日志保留。`legacy-compat-01.log` 误用 cpu label，实际 **0 tests，不计通过**；移除 label 并加 `--no-tests=error` 后 `legacy-compat-02.log` 才是有效的 6/6。CTest 详细日志含 CP936 时区，按原始字节保留。

| GPU 后端 | 新 C API | 旧 ABI 完整对照 | 真实 Worker 日志 |
| --- | --- | --- | --- |
| Qwen3 0.6B NF4 | 两次 128-token 固定 seed 采样一致；首次与第 26 节冻结产物前缀一致 | 11 项零差异，附加取消前缀/续写通过 | API 14 条；旧桥接含 7 条后端日志 |
| Qwen3.5 0.8B NF4 | 两次 128-token greedy 一致；首次与冻结产物前缀一致 | 同上 | API 15 条；旧桥接含 16 条后端日志 |
| Qwen3.5 0.8B BF16 | 两次 128-token 固定 seed 采样一致；首次与冻结产物前缀一致 | 同上 | API 19 条；旧桥接含 18 条后端日志 |

新 C API 使用 context1536、两 Session。第二次生成中确认 Session busy 后切换到 warning，再排空输出；完整 token、最终 context143、唯一终态、日志线程 join 和 outstanding/reserved/pending 归零均通过。三后端共生成 768 token，其中 384 token 有独立上一产物前缀对照。旧 ABI context512 fixture 保留全部 raw/history/reset、JSON/工具策略、结构化事件、四 peer 与槽位重建断言；11 项完整结果与第 25 节冻结的独立旧 runtime 输出逐字段比较，诊断字段不参与语义比较。BF16 仍只读使用原纯文本 hardlink fixture。

六轮全部由机器守卫、Windows Job、600 秒 watchdog 串行运行，零退出、Job 排空、无强杀、guard finished。实际 Worker 加载系统 `ze_loader.dll`，宿主不加载 GPU modules。A770 / 驱动 **32.0.101.8860** / Level Zero **1.28.2** / oneAPI **2026.1**，本切片未修改 GPU kernel 或计算形状，不将这些有限用例解释为性能提升或长期稳定性结论。

GPU 预检曾因 boot 变化及 240 条 WER 新写入阻断，未启动设备进程。审查确认 9/10 23:30 正常关机、9/11 13:13 正常启动，无新增 41/6008/系统 bugcheck/Display reset；240/240 WER 的完整 EventData/provider/log/event ID 与历史一致，169 份 dump metadata 不变。保存审查后显式确认 `1f21d736ba1e2df66eb414916324a1ad95fe36d44f353b74706e608a509cb2a9`，未修改守卫规则或自动重试。

证据目录：`tmp/refactor/p5-worker-log-bridge-20260910`。保存独立 baseline 引用、六轮 launch/result/guard/日志、三份旧 ABI 比较、CPU 失败及复验、SDK、重启审查和候选源码/产物清单。旧 adapter 默认仍关闭；剩余额度/资源边界、分块性能、HTTP 工具与媒体、P6 长稳和 P7 连续合批继续按完成审计推进。

## 28. 2026-09-11 旧 adapter 的额度提前检查与分配预算

旧文本 adapter 新增五项可选环境配置：共享 Session 数（1–4）、Session cache/state 总预留、显式 device USM、pinned host USM 和 Session host payload 额度。后三项接通已有 `allocation_limits_v1`；所有值从共享键使用的同一 Win32 环境快照读取，避免静态 CRT 环境过期。严格十进制与整数溢出校验在启动 Worker 前执行。默认仍为 4 Session、字节额度 0，旧 v1 路径与可选开关默认值保持不变。

### 复现并修正的两处边界

`boundary-before-01.log` 用冻结产物和 CPU Worker 复现：已有 context14，请求 32768 个输出 token、另含 1 个输入 token，超出总预留32768。旧 adapter 先 truncate 到0，随后 submit 才拒绝，丢失可用 context。现在按剩余 context 裁剪输出额度后，先拒绝连最少1个输入 token 都无法容纳的请求，保留上下文。32769/INT_MAX 的超过协商限额情况同样不改状态；短 context 对大 max_new_tokens 的原有裁剪行为保留。

新增非法配置检查还发现 `ProxyEngine::init` 把 `LegacyTextError` 统一转换成 runtime error。现保留原错误分类，非法限额返回 InvalidArgument；设备/缓存额度拒绝仍为原有 runtime error。失败产物对应源码、首轮日志和修改前复现脚本均保留。

### CPU 与 ABI

9 项相关唯一 CTest 门禁通过：首轮8/9，修复初始化错误码后对 LegacyTextApi、ProxyAbi、ProxyLogging、RuntimeIsolation 复验4/4。后四项中有三项属于重复兼容检查，不重复计数。旧 ctypes 完整 **17 组**、C++ seam **12 组**通过。

新增用例覆盖：32768/32769/INT_MAX 拒绝后 context 保持；1个输入加32767输出额度恰好准入，并立即取消以验证回收；短 context 裁剪；默认4个/显式2个 Session 的下一次拒绝；两份 cache 额度拒绝第三个、释放后槽位复用、peer 继续；所有分配额度经过真实 pipe 到达 CPU factory 的独立 sentinel 校验。28 组非法数值（符号、空白、单位、Session范围、uint64溢出）被拒绝，0和uint64最大字节值解析通过。CPU fixture 不模拟实际 USM 分配。

最终构建为首轮10步、错误映射修复后2步。43 项执行/公开接口产物与第27节冻结候选逐字节相同，包括 AilaWorker、GPU运行库、AilaInference.dll/import library/公开头和 AilaServer。只重建宿主 Proxy 及受影响 CPU fixture。45个旧 Proxy 导出不变，26个异步 C 导出不变，两宿主 DLL 的 PE 导入无 GPU runtime。新 C API及其头未变，沿用已核验哈希的第27节 MSVC/隔离 SDK 证据；本切片没有重复宣称重新构建 Server、重跑 HTTP 或 SDK。

### 真实 GPU 准入与恢复

每后端先独立加载并读取实际 Session 预留，再关闭该 Worker，依次运行各额度场景。始终只有一个 GPU Worker；context512，显式 device 上限8GiB、pinned host16MiB、Session host256MiB，测试目标额度逐项降低。

| 后端 | 每 Session device cache/state | 每 Session host payload | 结果 |
| --- | ---: | ---: | --- |
| Qwen3 0.6B NF4 | 58,720,256 B | 0 | device、cache、Session数及 pinned 采样额度拒绝/恢复通过；无独立 host Session payload |
| Qwen3.5 0.8B NF4 | 26,492,928 B | 20,201,472 B | 同上；一份 host 额度拒绝第二个 Session，已有 peer 正常 |
| Qwen3.5 0.8B BF16 | 26,492,928 B | 20,201,472 B | 同上 |

device=1 在加载时返回明确的分配额度错误；cache=两份预留、Session数仍为4时拒绝第三个 Engine，不新增 Worker，销毁一个后可复用。pinned=1 时先验证 greedy peer，再强制带 penalties 的 host sampler；采样局部失败、受影响 context归零，peer 原 context 不变，两个 Session 随后都可正常生成。另在该阶段验证显式 Session数2的第三个 Engine 被拒绝。Hybrid host 额度为一份实际预留时，第二个 Session 在分配前被拒绝，已有会话继续运行。

各阶段共 **17 次完整 JSON/上下文比较**，全部与第25节独立旧 runtime 的 `chat_greedy` 结果一致，比较原始文本、结构化结果/metadata和context，不只检查进程成功。三轮最终目录为 `gpu-admission-q3-02`、`gpu-admission-nf4`、`gpu-admission-bf16`，全部零退出、Job排空、无强杀、guard finished，无新增GPU OS故障证据。设备仍为 A770 / **32.0.101.8860** / Level Zero **1.28.2** / oneAPI **2026.1**。

首次 Qwen3 预检被48条新写入 WER 阻断，未启动 GPU进程。完整payload/provider/log/event ID全部与冻结历史一致，169份dump metadata、boot和主机未变，无读取错误及新增系统故障信号。独立审查后显式确认 `be86e74a55f39e72ca4d9f61715602ec14283130e738559eabd168279ce8fcb5`，新增attempt目录后才继续；原阻断状态、原runner和launch保留。审查脚本首次把守卫的持久化envelope当成payload读取，随后改为校验journal并读取payload；原脚本与原始envelope保留，未改守卫规则。

证据目录：`tmp/refactor/p5-legacy-admission-20260911`。此切片只验收旧入口的额度传递、有限资源拒绝与恢复；32768边界是CPU准入/提前取消测试，未运行32768-token GPU生成。局部提前检查不预占共享credit，其他请求仍可能使submit额度不足。预算不覆盖驱动/库私有分配，不代表跨owner的机器总量；近容量输入、长期压力、分块性能与HTTP工具/媒体继续推进，旧 adapter 默认仍关闭。

## 29. 2026-09-11 HTTP 工具调用、独立 token 解码与真实模型格式边界

HTTP 宿主新增严格的 `tools`、`tool_choice`、`parallel_tool_calls` 和工具历史校验。历史 call ID 唯一，结果必须对应尚未完成的调用，顺序可调整但不能缺失或重复；arguments 必须是严格 JSON object 字符串。工具定义和历史交给可信模板，Qwen3 无模板时使用工具感知 fallback，Hybrid 沿用固定模板。无工具/none 的普通 prompt 和解码保持兼容。

JSON 与 SSE 支持 Qwen JSON 和既有 XML 调用。完整块通过校验后才发布，调用 ID 绑定响应及 index；参数按最多1024原始UTF-8字节拆分，使帧符合16KiB交付上限。JSON对象/字符串参数保留原始数值拼写，避免小数或负零经 DOM 重写；原始文本及序列化输出继续各自受额度限制。未声明函数、损坏块、required/指定函数/调用数量违例产生明确错误和唯一终态。`strict=true` 返回400；不宣称 JSON Schema 约束解码，也不在服务内执行工具。

### CPU、socket 与产物

- Server **4/4 CTest**，其中 ChatCodec **35/35组**；主工程 **10/10相关兼容 CTest**，含18组新 C API、17组旧 ctypes 和12组旧 C++ seam。CTest详细日志按原始字节保留。
- 最终 Qwen3/Hybrid fake 工具 socket 各 **7组**；普通 HTTP/SSE **13组**，覆盖1/2/4并发、断线隔离、慢读恢复/取消、槽位回收、活动请求关闭和Worker故障。
- 工具用例覆盖 JSON/XML、分片、关联历史、required/指定/并行策略、24KiB参数分帧、错误后恢复、取消及无工具兼容。独立token与BPE开闭标签的四种组合通过。
- 仅生产 `ChatCodec.cpp/.hpp`、`HttpServer.cpp` 相对第28节变化；Worker/运行库、公共头、公共模板/解析器等45项产物字节一致，45个旧导出与26个异步C导出不变。Server的PE imports无SYCL/oneDNN/Level Zero；没有修改GPU内核。

保留的非GPU失败：最初fake context8192超过服务上限4096；控制标记放在空格后未被夹具按独立token识别；诊断环境读取误用了不存在的`read_bool`；独立诊断程序首次缺少user32链接。均修复后复验，原日志与相关失败源保留，未计作通过。

### 两层格式问题与隔离诊断

首次NF4工具请求返回502 `invalid_tool_call`，Worker正常生成41 token并关闭。只读审查确认两份执行trace完整、Job排空、无强杀及新增OS故障后，显式确认证据`352090bc…f4c44f7a7`。

代码检查找到工具标签的解码边界：Hybrid的`<tool_call>`/`</tool_call>`分别为248058/248059。公共Tokenizer把added token纳入特殊token表后会过滤它们。HTTP现在只在提供工具且choice不为none时恢复这两个标记；公共Tokenizer及旧ABI规则保留。CPU用例同时复现旧解码对BPE开标签/独立闭标签组合的失败。

修复后第二轮NF4的外层标签完整，但诊断WARN显示缺少`<function=weather>`，仍正确拒绝。再次审查后显式确认`2cd822e3…47c3f95e`，执行一次受守卫的三请求诊断，直接复用当前Server宿主静态库及未变Worker。保存每个token ID和解码字节，再由Python按checkpoint词表独立解码：原短请求的41 token确实没有函数头，且是BPE开标签配248059闭标签；两份原始块一致。明确XML和JSON格式的两个请求各40 token，都生成了可解析的XML调用。诊断正常退出、trace完整、Job为空，无新增OS故障；它是定位证据，不算正式HTTP通过。

该原请求的模型格式错误仍保留，服务未补造函数名或放宽错误块校验。正式Hybrid用例改为明确XML格式，验收传输、解析及独立会话；不能由此推断任意提示的工具调用成功率。最初两轮未保存逐token记录，不能宣称它们与诊断逐token完全相同；首次旧Server候选仅记录SHA256，后续轮次才逐轮冻结候选exe和三份生产源码。

### 三后端最终 GPU 验收

每轮先单独启动第28节冻结的旧Server，采集两份普通请求的完整JSON/SSE；正常关闭后再启动候选Server，验证普通对照、参数拒绝、工具JSON/SSE、工具结果历史和北京/巴黎并发隔离。reference与candidate及三个后端严格串行。

| 后端 | 最终目录 | 候选完成请求 / 生成token | 结果 |
| --- | --- | ---: | --- |
| Qwen3 0.6B NF4 | `gpu-q3-02` | 13 / 454 | 5组候选、1组独立旧参考通过 |
| Qwen3.5 0.8B NF4 | `gpu-nf4-03` | 13 / 320 | 同上，明确XML fixture |
| Qwen3.5 0.8B BF16 | `gpu-bf16` | 13 / 218 | 同上，使用既有只读纯文本hardlink fixture |

每候选另含5次预期400参数拒绝，最终report中的failed=5来自这些输入，不是GPU推理失败。三个候选共39次完成、992生成token，6组普通JSON/SSE语义与独立旧Server完全一致。工具响应的ID随请求变化，比较时仅剔除这个身份字段；两城市参数仍分别明确等于Beijing/Paris，不只与自身参考互比。

三轮均零退出、Job排空、无强杀、guard finished，六份Worker执行trace完整，最终系统快照无新增故障。环境为A770 / **32.0.101.8860** / System32 Level Zero **1.28.2** / oneAPI **2026.1**，context2048，工具生成上限192，每轮600秒watchdog。早期修复前Qwen3通过轮保留为历史，最终验收使用复验目录。

证据目录：`tmp/refactor/p5-http-tools-20260911`。保存失败与审查、逐token独立解码、所有最终launch/result/guard/trace、旧Server完整对照、CPU原始日志，以及冻结candidate和证据清单。本切片不提供性能或长期稳定性结论，P6完整资源计量、近容量与持续生命周期、P7连续合批和媒体驱动继续推进。

## 30. 2026-09-11/12 三后端重复服务基线与有限生命周期

本轮复用第29节的 Server/Worker 和全部推理源码，新增离线基线分析工具，不改变计算路径。九轮benchmark和三轮最终lifecycle通过，共4428次正式验收请求；另保留一轮因历史WER补写而未通过守卫的528次请求，实际共13次GPU进程运行、4956次请求。

### 工作负载与观测

三个后端各使用三个独立 Server/Worker 进程，按 NF4→BF16→Qwen3 交错，GPU 严格串行。每进程 context2048、4 Session，short/mixed、1/2/4并发；每组合预热2轮、正式20轮，最多生成64 token。长提示经实际 Server tokenizer/template 测得 Qwen3 为1734 token、Hybrid为1825 token，短提示均27 token。temperature=0、thinking关闭，seed为42加lane编号；本轮不能替代 sampling RNG 的独立验收。

每轮8次独立单请求参考、28次预热和280次正式请求，共316次；九轮共 **2844次请求，其中2520次计入性能**。所有完成请求的content/reasoning摘要、usage和结束原因均与相同payload的独立参考一致，跨进程参考也一致。mixed单并发交替短/长输入，拆分后每种每轮只有10个批次；其他组合各20个批次。

`service_baseline.py` 校验审查记录、guard/result与原始文件的SHA256，重新计算时刻、实际token数、吞吐、参考匹配和完整批次覆盖；禁止重复运行或失败样本被静默删除。输出每轮median、p10/p90、MAD和CV。13项离线测试通过，涵盖缺失/重复证据、参考不一致、时刻被改写、warmup失败、混合输入分组及合法的零SSE事件间隔。真实数据的九轮离线汇总通过。

### 短请求服务性能

下表为三个独立运行的批次吞吐median的中位数，单位tok/s；TTFT为同样方式汇总的客户端首内容时间，单位ms。不是kernel或纯decode耗时。

| 后端 | 1并发吞吐 / TTFT | 2并发吞吐 / TTFT | 4并发吞吐 / TTFT | 三档吞吐的跨进程max−min跨度 / 中位数 |
| --- | ---: | ---: | ---: | --- |
| Qwen3 0.6B NF4 | 82.48 / 128.80 | 95.38 / 160.04 | 96.57 / 252.57 | 2.01% / 0.73% / 0.15% |
| Qwen3.5 0.8B NF4 | 84.19 / 121.45 | 117.25 / 147.13 | 119.42 / 239.40 | 0.74% / 1.25% / 0.22% |
| Qwen3.5 0.8B BF16 | 88.56 / 78.91 | 126.72 / 77.20 | 128.57 / 125.49 | 0.77% / 0.21% / 0.65% |

本工作负载中1到2并发有收益，2到4的增幅有限且TTFT增加。此结果不自动改变调度参数，也不能当作连续合批的预期收益。三个checkpoint的输出和存储不同，不能由此给NF4/BF16内核排名。

### 长短混合与有限进度观测

| 后端 | 2并发吞吐 | 4并发吞吐 | 短请求TTFT，2 / 4并发（ms） | 短请求在所有长请求结束前开始输出，2 / 4并发 |
| --- | ---: | ---: | ---: | --- |
| Qwen3 NF4 | 63.46 | 63.86 | 141.84 / 176.47 | 60/60 / 117/120 |
| Hybrid NF4 | 56.03 | 60.05 | 130.51 / 179.44 | 58/60 / 120/120 |
| Hybrid BF16 | 70.41 | 71.76 | 77.52 / 107.87 | 60/60 / 119/120 |

短输入均生成64 token，长输入分别生成64/24/51 token，因此完成顺序不能单独判定公平性。表中首内容观测没有隐去未满足的批次；NF4两次例外另保留逐请求时刻。执行trace关闭，不能仅用客户端时刻定位具体GPU调度原因或宣称最坏取消/调度延迟。

初步性能筛查线标为 **HEURISTIC**：独立运行median的中位数，加/减两倍已观察到的max−min跨度，吞吐下界截断到0。它只触发审查，不是置信区间、固定百分比的通用回归门限或尾延迟保证。HTTP/health/进程采样、原始SSE写盘及JSONL持久化成本包含在数据内；SSE事件间隔单列，不能称为逐token延迟。完整参数及分析命令见 [服务基线](service-baselines.md)。

离线资源复核初次把断连请求的 `generated_tokens=null` 当成整数比较而抛出TypeError；`resource-audit-nf4-01` 和修改前脚本已保留。HTTP退役可能早于Worker最终摘要，此字段本来允许未知。汇总现保留未知计数，不补零、不推算取消请求的实际生成量；同一份已通过的NF4 GPU数据经 `resource-audit-nf4-02` 复核通过，没有重跑GPU或修改服务实现。

### Qwen3 收尾的历史WER补写

首轮Qwen3生命周期完成全部528次请求，进程零退出、Job排空，但guard.finish检测到48条新增WER而返回失败。事件写入时间为2026-09-11 16:37 UTC，完整EventData/provider/log/event ID全部与第29节冻结历史一致；169份dump metadata、host和boot不变，无读取错误或新System故障信号。证据ID为 `5946eae4a57a0c50e0eb2dec394ce97ee152b4fe79f77d51c8dec0214fc7ad39`。独立保存审查后才显式确认该ID，并使用新目录 `q3-lifecycle-02` 复验；未更改守卫规则。

原 `q3-lifecycle-01` 的failed结果、guard、全部请求和资源记录保留，不计作正式通过。离线另验证它的256次正常完成/256次断连、参考输出和增长门限，标记为workload checks通过、GPU guard未通过。该轮不会增加性能基线样本。

首次审查脚本误要求失败guard的hash等于最终失败报告，断言失败并保留 `delayed-wer-review-01`。runner本来先写待提交报告，再在guard失败后写入 `passed=false` 和diagnostic；只在成功路径保证最终hash匹配。修正后用原序列化规则重构提交前内容，SHA256与guard记录精确一致，最终失败文件保持原样；`delayed-wer-review-02` 完成48/48历史记录核验。

### 生命周期、资源与最终身份

每后端正式运行512次请求，四并发128轮，其中256次完成、256次首内容后主动断连；另有8次参考和8次预热，不计入512。正式生命周期共1536次，每个后端都越过256个retained request handle边界。逐批验证reference一致、取消归属、active/outstanding/pending归零及槽位复用；三个最终运行的全部已完成请求均与独立参考一致。

| 后端 / 最终目录 | Server private增长 | Worker private增长 | Server / Worker句柄变化 |
| --- | ---: | ---: | ---: |
| Hybrid NF4 / `nf4-lifecycle-01` | 2.80 MiB | 26.29 MiB | +1 / −6 |
| Hybrid BF16 / `bf16-lifecycle-01` | 2.51 MiB | 15.80 MiB | +1 / −1 |
| Qwen3 NF4 / `q3-lifecycle-02` | 3.37 MiB | 25.91 MiB | +1 / −5 |

增长比较预热后的baseline与最终空闲，均低于事先设定的每进程128 MiB和64句柄门限。另按连续32轮分四段保留分布：NF4 Worker private的中位数仍有小幅上升，BF16/Qwen3后段较平；不能由门限通过推断长期无泄漏。断连摘要中的未知生成量保留为null，不纳入完成token总量。

十二轮正式验收均runner退出0、guard finished及SHA256绑定匹配，Server/Worker正常退出、Job为空、无强杀。最后快照无新增系统故障；前述历史WER补写的失败与显式审查/确认另行保存。GPU始终串行，benchmark watchdog600秒、lifecycle900秒；显式device USM8GiB、pinned host16MiB、Session host payload256MiB。环境仍为A770 / **32.0.101.8860** / System32 Level Zero **1.28.2** / oneAPI **2026.1**，同一boot。

三模型26个文件在全部13次启动中路径/大小/mtime一致，测试结束后另记录完整SHA256；runner没有在每轮启动前对权重内容做hash。本轮未重建C++、未改变GPU/推理实现；53个已冻结构建产物与第29节字节一致，因此旧ABI与执行产物沿用已有兼容证据。新增离线工具13项测试和本节真实数据重算属于本轮新增验证。

证据目录：`tmp/refactor/p6-service-matrix-20260911`，含 `service-baseline.json`、`resource-validation.json`、`checkpoint-identities.json`、全部成功/失败运行及审查、candidate和证据清单。P6仍需近容量/较大模型上下文、持续创建释放与回滚、运行时显式分配观测、未计量资源的准入策略和更长压力；P7及媒体迁移继续推进。当前HTTP memory只反映加载时记录，下一切片补充不会阻塞HTTP事件循环的运行时分配查询。

封存工作在下一次启动后继续。Windows记录显示9/12 00:57开始菜单关机、正常停止日志服务，10:32再次启动；测试此前已于00:51正常结束。恢复时144条新写入WER均与最后成功快照的完整历史payload一致，169份dump metadata未变，无新增41/6008/System1001/Display4101信号；其他启动事件也保留。首次封存因新boot及补写记录暂停，审查后以最后成功guard中的原始收尾快照作为 `crash-after.json`，另存 `post-resume-*` 记录，未将十小时后的归档时刻当作GPU测试结束时刻。这项归档审查不清除机器GPU守卫，也不授权跳过下一次启动预检。

## 31. 2026-09-12 运行时USM查询、HTTP缓存采样与释放验证

新增 `DeviceRuntime::TryMemory()`，执行锁忙时立即返回空值，读取CPU账本而不初始化设备、提交kernel或同步队列。IPC增加可选 `runtime_memory_v1` 和独立的 `worker.memory_query`；uninitialized/busy的memory为null，ready返回完整uint64字节、峰值、存活分配及上限。生命周期操作未结束时Worker直接回复busy，避免诊断查询排在模型加载、Session创建/回收的GPU等待后面。查询使用普通命令额度，取消/ACK的保留容量不变。

普通C++ Client、新的 `aila_inference_client_query_memory` 和Server复用这条路径。原ABI结构与版本保持，旧Proxy45个导出不变，异步API保留原26个并新增1个。Server可选 `--memory-sample-ms`，默认0关闭；启用时HTTP只读缓存，报告状态、结果年龄、busy/失败计数和停止状态。查询错误或1秒超时后停止采样、清空本次有效memory，不自动重复提交；关闭先join采样线程再关闭Client。详情见 [运行时分配快照](runtime-memory.md)。

### CPU、HTTP 与 SDK

- Worker首次构建67步通过；补上生命周期忙的快速回复后6步、调整测试生命周期后2步通过。没有修改GPU kernel、计算形状或编译优化参数。
- 主工程13项相关CTest全部通过，覆盖协议、Worker、Client、C头、异步C API、旧接口、调度/故障与隔离；最终Worker及C API复验2/2。异步ctypes完整19组，旧ctypes17组通过。CTest详细日志按原始字节保存。
- 独立MSVC Server构建、32对象头依赖检查和4/4 CTest通过。普通HTTP/SSE13组回归通过。
- 新HTTP CPU场景5/5通过：关闭采样、未初始化、ready/busy、查询异常、查询超时；5组非法interval被拒绝。查询失败后采样停止，HTTP仍能完成与关闭采样参考一致的JSON/SSE。异常和超时用例明确由fake factory注入，不是GPU错误。
- C11/C++17独立MSVC消费者及只有宿主DLL/CRT的隔离运行通过；新结构120字节、计数offset16、epoch offset80、未知尾部保留。两宿主DLL及Server的PE依赖无SYCL/oneDNN/Level Zero。

### 三后端 GPU

每轮顺序运行冻结的第30节旧API产物、新API产物、旧Server、新Server，四个Worker严格串行。context512、4 Session，8GiB显式device、16MiB pinned、256MiB Session host额度，600秒watchdog。API验证4组greedy/固定seed host sampling独立输出，再验证四请求交错；两种执行方式都逐token匹配冻结旧版本。候选随后完成取消前缀对照、reset后恢复、8轮×4 Session创建释放，检查预热后的当前字节与分配数量不增长；最后释放模型，device/pinned存活字节和数量均归零，历史peak保留。

| 后端 / 最终运行 | API并发阶段busy / 查询数 | device生命周期峰值（B） | pinned生命周期峰值（B） | 最后存活分配 |
| --- | ---: | ---: | ---: | --- |
| Qwen3 NF4 / `gpu-q3` | 74 / 76 | 1,491,320,534 | 303,872 | device/pinned字节及数量均0 |
| Hybrid NF4 / `gpu-nf4-02` | 65 / 67 | 2,021,581,926 | 496,640 | 同上 |
| Hybrid BF16 / `gpu-bf16` | 39 / 40 | 4,868,511,172 | 496,640 | 同上 |

三个候选共24组完整API token序列与旧产物一致，并完成取消/恢复和96次额外Session创建释放。HTTP每后端两种请求的JSON/SSE与旧Server一致，另四个同时进行的SSE请求分别匹配旧参考。开启采样后的HTTP记录都有ready/busy观测、上限正确、失败数0，并在关闭后标记采样已停止。真实RPC耗时和客户端观测不用于性能或最坏延迟结论。

三个最终运行均guard finished、结果SHA256匹配、进程零退出、Job排空、无强杀，12份Worker执行trace完整，无新增GPU OS故障。环境为A770 / **32.0.101.8860** / System32 Level Zero **1.28.2** / oneAPI **2026.1**。所有GPU测试期间没有构建任务。

### 保留的启动/收尾审查

本切片开始前审查了正常关机/再次启动及144条历史WER补写，169份dump metadata未变。第一次显式确认因尚未登记待审状态而被守卫拒绝，未启动GPU；随后运行仅预检的guard登记，再确认精确ID `83dcbd0a06938d85143e024e912a81d81909eb6e9e22906cc5f72e32f19fe8b3`。没有修改守卫规则或自动确认逻辑。

首轮NF4的API/HTTP功能和四份trace通过，但收尾出现48条WER补写，guard返回失败。完整EventData/provider/log/event ID与运行前历史全部一致，boot和169份dump metadata不变。保存 `nf4-wer-review.json`，显式确认 `86d26d2c72b5c8cb2906e12fd7f64a3b08e669088c5cb6180b4debdebb91203d` 后，才在新目录复验NF4。原 `gpu-nf4` 不计入三个最终通过运行，原结果、failed guard、日志、trace和源码/产物副本保留；该wrapper的功能passed不能替代guard验收。

证据目录：`tmp/refactor/p6-runtime-memory-20260912`。本切片完成显式USM的运行时观测和有限释放验证，不把账本归零解释为WDDM/驱动私有分配归零，也不宣称长期无泄漏。近容量上下文、更长期压力、未计量资源准入策略、P7连续合批与其他模型能力仍按完成审计推进。

## 32. 2026-09-12 四模型4096-token边界、持续回收与WER报告身份

本轮推理源码、C++二进制和公开ABI与第31节冻结产物一致。新增验证覆盖Qwen3 0.6B NF4、Hybrid 0.8B NF4/BF16，以及Hybrid 4B NF4。每模型context4096、4 Session，显式device8GiB、pinned16MiB、Session host256MiB，整轮900秒watchdog。先用独立进程逐个采集greedy与固定seed host sampling参考，再运行四请求交错。

### 容量与状态

每模型四组请求中两组为4032输入+64生成，精确达到4096；另两组短输入作交错对照。16组并发完整token序列均与独立串行参考一致。达到容量后再提交被拒绝，句柄输出保持0、Session长度和其他peer状态不变，USM当前量也不增加。

随后向3584截断，按返回的实际恢复长度用prefill补齐，再生成最后448输入对应的64 token；结果与原完整请求一致。Qwen3实际恢复3584，三个Hybrid因IPC关闭快照而恢复0，补算路径同样通过。四模型均在prefill已处理1024 token后完成取消，保留该实际前缀，续算后恢复原始输出；3968输入+128生成目标的decode取消前缀匹配独立参考，再次回滚补算也一致。释放满容量Session、确认回收后重建槽位，不改变其他peer。

### 持续创建释放

长循环在单独Worker运行，先预热同一组长/短输入与host sampling，另做两轮生命周期预热。正式256轮，每轮创建4个capacity4096的Session，完成一个16-token请求、取消另一个128-token目标请求，再释放四个Session。每个完整输出和取消前缀均与独立参考比较，逐轮保存实际token、USM和Windows进程数据。

四模型合计 **4096次正式Session创建释放、1024次正常完成和1024次取消**，不含预热及功能阶段。每轮空闲USM字节和存活分配数量都回到同一预热基线；最终模型释放后，device/pinned存活字节与分配数量均为0。

| 模型 / 最终运行 | device峰值（B） | Worker private增长 | 宿主private增长 | Worker句柄变化 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3 NF4 / `gpu-q3-03` | 3,407,165,082 | 9.22 MiB | 1.20 MiB | −4 |
| Hybrid NF4 / `gpu-nf4` | 2,482,777,130 | 9.90 MiB | 0.02 MiB | −4 |
| Hybrid BF16 / `gpu-bf16` | 5,351,500,996 | 5.55 MiB | 0.18 MiB | −4 |
| Hybrid 4B NF4 / `gpu-nf4_4b` | 4,816,754,407 | 16.10 MiB | 1.17 MiB | −1 |

四轮宿主句柄变化均为0，均通过预先设置的每进程128MiB/64句柄增长门限。这里的宿主是C API测试进程，不能视为HTTP Server RSS。所有最终运行正常退出、Job排空、guard finished，无新增OS故障。功能阶段保留完整trace，长循环关闭trace，仍由机器守卫、Job和watchdog保护；没有构建与GPU并行。

### 两个保留的失败与修正

首轮Qwen3通过容量、回滚和取消后，在 `handle.release` 应答后立即创建新Session，收到正确的ResourceLimit。应答只关闭引用，不代表生命周期执行器已释放设备资源。修正夹具，在无其他活动请求的受控条件下等待账本显示相应cache已回收，再创建新槽位；没有修改生产释放语义。原失败和两份完整trace保留，审查后显式确认 `006a1ce41284a472ff9f57302f46352c95ca3168e34cd4738c2b93b5dfcf146f`。

第二轮长循环完成133轮后，trace达到67,108,731字节，仅距64MiB上限133字节。诊断写入限制按设计使后续执行失败，Worker退出5；没有新增OS故障。原不完整trace、请求和内存记录保留，不能计作完整通过。审查后显式确认 `c93f927a6b81a16d951f45b27610936fbe2aa5c081bd4694cac994282a523298`，将短功能诊断和长循环拆成独立Worker；生产trace上限保持。Qwen3成功参考按SHA256复用，修正候选独立复验，不覆盖旧数据。新的请求日志保留失败终态的错误文本与代码。

### WER重投递识别

此前多次手动审查确认同一历史WER被重新写入日志。守卫现仅对有效ReportId、完整EventData/provider/log/event ID完全一致、且关联转储存在状态/元数据未变的记录作重复分类；原始记录保留。持续不存在的转储必须由前后相同、完整且无读取错误的扫描范围证明，新快照增加dump_roots/full_dump_path。旧快照缺少扫描范围时保持保守，不推断不存在。已有failed/blocked证据仍须显式确认，分类器不会把过去失败轮改为通过。

41项守卫CPU测试和58项runner CPU测试通过，包括新ReportId、payload变化、dump新增/消失/变化、扫描范围变化、System事件、boot和日志回退、读写失败及已有失败不被清除。对真实历史数据的首次离线检查发现43份关联dump已经不存在；未放宽为仅按ReportId去重，补齐扫描覆盖条件后，用真实48份payload模拟重投递通过。旧原始快照仍保守留下待审项。该验证是离线模拟，不是新触发了48次系统事件；详情见 [GPU运行守卫](gpu-run-guard.md)。

环境为A770 / **32.0.101.8860** / System32 Level Zero **1.28.2** / oneAPI **2026.1**，证据目录 `tmp/refactor/p6-capacity-lifecycle-20260912`。此轮证明所列4096-token工作负载和有限持续回收，不代表32768-token设备执行、任意上下文/模型、小时级长期稳定性或驱动私有资源完全有界。P6剩余边界、P7和其他模型迁移继续按完成审计推进。

## 33. 2026-09-12 连续decode合批的功能边界与数值验收

按用户要求，本轮先完成合批功能、兼容性和失败恢复，再进入集中压力阶段。三个subagent分别负责后端、调度/证据及服务/协议工作，GPU验证由主执行流程严格串行运行，并与构建错开。新入口默认批宽1，显式配置上限为1..4；同批必须属于同一LoadedModel对象，不能仅凭模型目录或backend名称匹配。同一Session只允许一个活动请求，重复/非法成员在全组预检中拒绝，锁和Busy lease完整回滚。完整行为见[连续decode合批](continuous-batching.md)。

### 执行、调度与兼容边界

Session、Model和Request增加独立decode批次入口；每行保留自己的位置、KV、DeltaNet递归/卷积状态、RNG、history、thinking budget及Finalizing状态。调度器按队首公平选取就绪成员，prefill仍分块轮转，不等待凑满批；批内前向和结果校验完成后才发布输出。修复了终态可见与active slot扣减之间的竞态，两者在同一调度队列事务中完成。活动/在途请求、context、token、cache和输出ring额度保持既有约束。

取消在步骤检查点处理，不中断已提交的设备工作；已越过检查点的输出仍可能在Cancel返回后交付，Cancel不是事件屏障。背压、释放、慢消费者和父句柄关闭保留原来的排空/ACK及唯一终态约定。一般执行异常结束本组尚未终结的成员，设备RuntimeFault按设备域传播。前向后出现AllocationLimitExceeded时先排空队列，再将所选Session reset为空context并报告ResourceLimit；非成员状态保留，不自动重试该组。

IPC通过可选 `continuous_batching_v1` 协商 `max_decode_batch_size`，严格校验1..4、maxActiveRequests及精确回显。缺省1省略扩展字段，旧Worker和旧Client保持原路径。C ABI新增 `aila_inference_client_start_with_batching`，原options仍为104字节、reserved为零，旧start等价于批宽1。Server新增 `--decode-batch-size`，不得超过 `--sessions`；health、ready及退出报告展示的是配置上限，不是实际批宽。旧文本adapter没有隐式启用合批，公共模型API没有新增LoRA加载/切换能力。

### 独立decode投影与保留的两次数值失败

此前将请求数当作seq_len复用prefill GEMM，改变了decode的投影累加、归一化与FFN中间值舍入边界。第一轮 `gpu-backend-q3` 的归一化RMS误差为0.0175053928，超过0.015625；恢复逐行原gate/up计算后，第二轮 `gpu-backend-q3-02` 仍有66个值未通过、最大绝对误差0.21875、归一化RMS误差0.0204347377。两次均为数值断言失败，没有新增系统故障，不能计作成功。

最终新增 `Bnb4BitLinear::ForwardDecodeRows`。默认packed NF4、K≥512、B2..4且WG为默认/256时使用独立多行decode kernel，每个序列拥有独立的补齐output group网格，保持SG16/WG256及原vec8、BF16反量化、FMA和reduction算术体。原B1 kernel body未改；B1、小K、不兼容packed或其他WG override逐行调用原forward。输入dtype、context、形状与容量在提交前检查，logical span不得重叠；LoRA按行沿用原attachment顺序，不重复添加，不引入整权重BF16 scratch。

Qwen3与Hybrid NF4的QKV/O/Down等投影接入新入口；norm、residual+norm和gate/up逐行沿用原decode内核。Hybrid BF16保留逐行原JM/custom/Linear选择，没有实现跨请求BF16 GEMM。`AILA_FUSE_RESIDUAL_ADD=1` 及原profiling/debug/legacy条件按文档回退完整B1。另修复三后端gate/up物理buffer宽度为 `max(ff_dim, hidden_size)`，逻辑row stride不变，避免窄FFN时O/Down输出越界；fused gate/up遇LoRA adapter时拒绝融合，保留原adapter计算。

预先固定的数值门限未放宽：绝对0.125、相对0.03125、归一化RMS 0.015625，B1要求逐位一致。组合修正后全部选中模型比较误差为0；这些结果不独立量化GEMM与norm各自对旧误差的贡献。失败原产物分别保存在 `pre-ffn-correction-artifacts/` 和 `pre-decode-rows-artifacts/`，原日志、failed guard、review及精确acknowledgement均保留：

- 第一次证据ID：`12e00435c4b26b78a3db9bd2a567a74fc41c77b32a8578921aab217b9aa82499`，审查为 `numerical-review-01.json`。
- 第二次证据ID：`1bf00c7f70407008f06be47ef685a27175d93c6e059acc42478d93a41ec91e62`，审查为 `numerical-review-02.json`。

### 构建、CPU与独立宿主

最终生产构建为 `build-main-06`，Server为 `build-server-03`；后续 `build-policy-01` 只重编确定性负例所在的AilaBatchInferenceTests，没有改变已验证的Worker或其他组件产物。所有构建和正式命令回执保留。

- 主CPU最终20/20，包含16个CPU组与4个旧ABI门禁；Server CPU 4/4。
- trace离线检查38项、policy trace正反契约6项通过。普通analyzer的错误判定没有放宽。
- HTTP Fake Worker的默认、显式1/2/4配置、1/2/4并发及7类非法参数通过；fake后端验证配置和逐行回退，原生批次由GPU用例验证。
- 独立MSVC C11/C++17消费者编译、链接和隔离运行通过；旧45个导出全部保留，异步API由27增至28，唯一新增为batching start。普通宿主DLL和Server无SYCL/oneDNN依赖。

### 四模型完整数值与Request重放

每个后端以独立Forward Session为参考，覆盖B1/2/3/4、不同位置、动态换序/暂停/替换、非法组拒绝、context capacity为1而批宽为4、reset重放；Hybrid还覆盖snapshot恢复和淘汰。比较全部logits，不以少量抽样或最终greedy token代替数值门禁。

| 模型/权重配置 | 完整logits行 | 比较值数 | 最大绝对误差 | 最大归一化RMS误差 | 独立逐位比较事件数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3 0.6B NF4 | 72 | 10,939,392 | 0 | 0 | 46 |
| Qwen3.5 0.8B NF4 | 72 | 17,879,040 | 0 | 0 | 50 |
| Qwen3.5 Hybrid BF16文本夹具 | 72 | 17,879,040 | 0 | 0 | 50 |
| Qwen3.5 4B NF4 | 72 | 17,879,040 | 0 | 0 | 50 |
| 合计 | 288 | 64,576,512 | 0 | 0 | 196 |

逐位比较事件与完整数值比较行是不同计数，不相加当作样本数。另使用第32节冻结旧产物作独立B1兼容对照，每种配置的参考/候选两侧各12个请求、96个输出token，完整tokens、status和context一致。

Request用例覆盖4→3→2→1动态缩批、独立采样/RNG、Finalizing与取消；每模型4行换序重放加3行取消后peer重放，合计28行完整输出与context对照通过。四模型trace均实际出现decode_batch宽度2/3/4，最终模型释放后显式device/pinned字节与分配数量为0。

独立NF4投影组件覆盖7种形状、24,848个值的CPU double参考，以及50组B1～B4逐位对照、27组非法输入拒绝、尾部/padding/canary和USM不变。LoRA另在dnnl与fused两配置验证gate-only/up-only/both、B1回退和B4投影；不将组件覆盖扩展为公共LoRA服务或任意模型adapter验收。

### 前向后确定性ResourceLimit与恢复

四模型各自从初始化起设置8GiB device及1字节pinned-host上限，先获得独立B1 oracle并释放，再创建四个Session。预热精确B3前向形状后，三个成员经synthetic thinking close延迟首次采样，其中row2为Finalizing。正式B3前向已完成，row0的greedy与D2H回读成功；row1首次host sampling申请vocab×2 pinned字节时被额度拒绝。因此这是真实前向之后的确定性分配失败，不依赖测量ready字节后重载来猜测拒绝位置。

三条所选请求精确返回ResourceLimit，无新增token，终态仅一次，context重置为0、Busy lease释放；重复调用不再次提交或发布。非成员的完整请求与续写保持B1一致；三个成员无须额外显式Reset即可重建上下文，完整logits和新请求tokens均与独立参考一致。拒绝前后显式USM/pinned账本不变，最终释放后归零，runtime无fault。

普通trace分析结果仍保留 `state=error`、`complete=false` 和sampling/decode_batch两条scope_error。专用 `policy-trace-contract.py` 只接受精确这两处错误，并验证成功forward、row0回读、host drain、catch drain、三次reset、完整hash及trace_close。6项离线正反用例确认缺失/错误顺序会拒绝；没有将任意错误trace改判为正常完成。

### HTTP、正式选择与结论边界

四模型真实Server/Worker各7项，共28项：批上限1和4的singleton JSON/SSE一致，1/2/4并发各自完成，断线peer隔离、slot reuse及活动关闭保留唯一outcome。Qwen3、0.8B NF4和BF16的HTTP trace观察到实际批宽2/3/4；4B HTTP本轮只观察到2/3，四并发和配置上限4通过，不能据此声称HTTP实际执行B4。4B原生B4由backend及Request用例验证。

`accepted-runs.json` 显式选中23轮正式GPU功能验收：四模型各backend/B1/requests/policy/HTTP五类，加两轮LoRA及一轮projection。所有正式轮正常退出、Job排空、无强杀、guard finished且无新增OS故障。全部28次尝试保留，包括3轮被新版产物替代的历史成功和上述2轮已审查失败；失败不计入正式通过数。封存校验按每种用例实际执行的产物匹配最终哈希，不把独立fixture未使用的Worker/Server重建误判为其已执行依赖变化。

环境为A770 / **32.0.101.8860** / System32 Level Zero **1.28.2** / oneAPI **2026.1**。证据目录为 `tmp/refactor/p7-continuous-batching-20260912`，包含正式选择、CPU/ABI回执、完整原始尝试与失败审查、源码/产物candidate、implementation.diff及候选/证据清单。第32节冻结基线继续只读保留。

本轮完成的是上述合批功能边界。默认仍为1，显式USM归零不能推导驱动私有分配或全部WDDM/RSS归零；NF4多行入口没有在本轮证明吞吐收益，BF16仍按行投影。集中压测、吞吐/尾延迟、小时级稳定性及P6剩余资源边界继续按路线图推进，不宣称P7全阶段或整个重构完成。

## 34. 2026-09-18 连续合批吞吐/延迟与集中运行

本节补齐第33节功能边界之外的 P7 量化验收。全部使用**与第33节数值验收完全相同的 Worker/Server 产物**（AilaWorker.exe / AilaServer.exe 哈希与 P7 candidate-manifest 一致），因此性能与数值证据绑定同一二进制。测量经真实 HTTP/SSE 服务，每条完成请求仍与自己的单请求独立参考逐字段比对（content/reasoning 摘要、usage、finish_reason），打开批上限不会静默改变服务输出。

### 机器守卫与准入审查

本节开工前机器守卫为 blocked，原因是本会话最初以 Medium 完整性运行：boot_id 因操作者重启而改变，且受 ACL 保护的 LiveKernelReports PoW32kWatchdog 无法枚举，使该次快照的 dump 清单与已存记录不一致并附带 read error；其余为新增内核 TDR/watchdog 上报（50 个 WER 内核报告，LiveKernelEvent 141/117/1a8/1b8 与 BlueScreen 0x116/0x139，另有 2026-09-13 的一份 7.08 GiB WATCHDOG dump）。操作者确认该 TDR 证据与本项目无关、不作为项目阻塞项。取得管理员权限后重读快照 read_errors 归零，经守卫自身的显式复核接口（先绑定陈旧 evidence id 触发重算，再确认新 id）记为 acknowledged，理由与复核记录写入 guard-review-20260918.json、guard-acknowledgement-20260918.json 与 guard-acknowledgement-note-20260918.json。过程中没有放宽任何守卫规则、没有重试失败运行，也没有清除任何 Aila 侧失败。该状态机在隔离目录做过一次 dry run（dryrun-guard-acknowledge.py，运行后删除沙箱）确认只有显式确认后才允许准入。

### benchmark runner 的合批复权

tools/server/service_benchmark.py 新增 --decode-batch-size（默认 1，1..4 且不超过 --sessions），直接传给 AilaServer；runner 同时校验 health、ready 消息与最终 report 回显的 max_decode_batch_size 与请求值一致，因此漏传或错值不会静默退回默认单请求路径。该改动为 Python 侧，未触碰推理源码；test_service_benchmark.py 60 项（含新增 3 项边界/回显负例）与 test_service_baseline.py 13 项通过，test_execution_trace.py 38 项、test_gpu_run_guard.py 41 项不变通过。

### 吞吐与延迟

以下为同一进程内 N 个并发请求的完成 token 总量除以首请求开始至末 [DONE] 的时长（含请求准备与服务传输，不是 kernel 吞吐），单位 tok/s。除另有说明为 10 轮；--new-tokens 64、temperature 0、thinking 关闭、context 2048、sessions 4。

| 模型/权重配置 | 输入 | 并发 | 上限 1 | 上限 4 | 变化 | TTFT p50 (1→4) |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Qwen3 0.6B NF4 | 短 | 4 | 96.33 | 122.20 | +26.9% | 0.249→0.267 |
| Qwen3 0.6B NF4 | 短 | 4 | 95.13 | 118.70 | +24.8% | 0.246→0.282 |
| Qwen3.5 0.8B NF4 | 短 | 4 | 118.61 | 148.37 | +25.1% | 0.238→0.230 |
| Qwen3.5 0.8B NF4 | 短 | 4 | 118.74 | 149.69 | +26.1% | 0.238→0.243 |
| Qwen3.5 0.8B NF4 | 混合 | 4 | 85.16 | 96.83 | +13.7% | 0.585→0.585 |
| Hybrid 0.8B BF16 夹具 | 短 | 4 | 127.13 | 138.13 | +8.7% | 0.130→0.128 |
| Hybrid 0.8B BF16 夹具 | 混合 | 4 | 95.40 | 99.84 | +4.6% | 0.305→0.250 |
| Qwen3.5 4B NF4 | 短 | 4 | 45.09 | 50.25 | +11.4% | 0.825→0.831 |
| Qwen3.5 4B NF4 | 混合 | 4 | 36.17 | 39.40 | +8.9% | 1.822→1.880 |
| Qwen3 0.6B NF4 | 混合 | 4 | 88.89 | 107.03 | +20.4% | 0.459→0.327 |

并发 2 的混合输入为 Qwen3 +9.9%、NF4 +5.5%、BF16 +4.4%、4B NF4 +5.3%。并发 1 的四个短输入配置变化为 +1.4%/-0.9%/-0.6%/+0.2%，处在重复测量波动内；上限 4 不改变 B=1 执行路径，其等价性由第33节 288 行完整 logits 零差异单独证明，不能由并发 1 的吞吐差反推。150 轮的扩展运行沿用同一方向：Qwen3 +26.9%（91.05–97.54 → 109.51–125.92）、0.8B NF4 +25.1%（116.71–119.54 → 118.77–150.70）。

BF16 的提高来自调度与行间交错，而不是把多个 BF16 投影合并成一次 kernel launch：ForwardDecodeRows 对 BF16 仍是逐行调用原投影。

### 实际执行批宽

配置上限与并发请求数都不能证明某一步真的执行了多行。本节用 Worker 自身执行 trace 直接计数：在 --decode-batch-size 4 下运行第33节同一套 HTTP 合批夹具，trace 记录 decode_batch 宽 2 共 28 次、宽 3 共 4 次、宽 4 共 16 次（48 次多行批次），两份 Worker trace 完整、无 scope error，且该轮夹具全部用例通过。同轮 stdout 出现 3 条 trantor 断连诊断（见下）。

### 集中运行

四种配置在 --sessions 4 --decode-batch-size 4 下各通过 512 次生命周期（宽 4、混合输入、完成与断连交替）：

| 配置 | 请求 | server private Δ | worker private Δ | 句柄 Δ | failed/rejected/slow |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3 0.6B NF4 | 512 | +2.39 MiB | +33.67 MiB | ≤2 | 0/0/0 |
| Qwen3.5 0.8B NF4 | 512 | +3.88 MiB | +28.76 MiB | ≤2 | 0/0/0 |
| Hybrid 0.8B BF16 夹具 | 512 | +2.90 MiB | +19.75 MiB | ≤2 | 0/0/0 |
| Qwen3.5 4B NF4 | 512 | +3.19 MiB | +36.87 MiB | ≤2 | 0/0/0 |

Qwen3 与 0.8B NF4 另有 150 轮 × 4 并发的扩展短输入运行（各 600+ 请求），两者通过且 stdout 无任何框架 ERROR/FATAL。

### 两项保留的未通过证据

两项均按原样保留，**不计入达标样本**，门禁未放宽：

1. soak-nf4-b4/run-20260918T070024-f20eb8294337：5016 请求合批 soak，全部请求达到预期结局（2512 完成、2504 断连，failed/rejected/slow_client_cancellations 均为 0，final report 正常关闭、Job 排空无强杀，growth 在门限内），但被 runner 的框架 ERROR/FATAL 日志门禁判为失败：3 条 TcpConnectionImpl.cc:61 发送路径诊断（0 与两次 10053）。该类诊断已在 [HTTP 探针](refactor-http-probe.md) 中作为刻意断连期的既有现象记录，且未中止进程、未遗留生产者、未阻止关闭。审查见 soak-log-gate-review-20260918.json。结论边界：该轮证明请求结局与资源收尾，但因日志门禁未通过，不作为通过的长稳样本。
2. trace-widths-02/q3：同一 HTTP 合批夹具的 disconnect_peer_and_slot_reuse 用例在 8 秒内未观察到 disconnect_cancellations 增长而超时。同产物重跑（trace-widths-03/q3）全部用例通过且批宽分布完全一致（2×28、3×4、4×16），四次第33节正式 HTTP 运行该用例的取消延迟为 0.019/0.028/0.047/0.063 秒。判定为间歇性时序敏感（该轮 victim 首内容到达偏晚，疑在客户端断开前已完成终结），**不是确定缺陷，也未当作已通过**。

另有一次自致驱动缺陷：新驱动提前创建了 fixture 要求的 --output 目录，fixture 在启动任何 Aila 进程前即退出 1（stdout 为 Cannot create an independent output directory），无 trace 产生。已修正并保留审查（harness-failure-review-20260918.json）。

### 结论边界

本节完成 P7 的吞吐/延迟量化与集中运行。输入为短 prompt 与长短混合（decode 主导），未覆盖长 prompt prefill、近容量上下文与小时级长稳；吞吐含服务准备与传输开销及常开的 health/进程采样/写盘观测，不能当作 kernel 吞吐；显式 USM 观测也不代表驱动私有分配或全部 WDDM/RSS。证据目录 tmp/refactor/p7-batching-perf-20260918，汇总 p7-batching-performance-summary.json（SHA256 1ab1afd5dbb7c382493d4238aca4c0425a01c5b8b52a39f2630516fb962eca92）。不宣称 P7 全阶段或整个重构完成。

## 35. 2026-09-21 BF16 多行投影与 P7 剩余项收口

本节完成第33/34节之后 P7 的剩余验收：小时级长稳、近容量长 prompt，以及此前列为"另行评估"的 BF16 投影聚合。

### BF16 多行投影实现与数值

Hybrid BF16 的 `ForwardDecodeRows` 原实现逐行调用 count=1 kernel，24 层 × 每层 4 个投影下，4 行一步发出 384 次 launch，且每行重新读取整份权重。本轮新增 `run_decode_jm_matvec_rows`（QKV/O/linear-all/linear-output/down）与 `run_decode_ffn_gate_up_swiglu_rows`（gate+up+SwiGLU，含逐行 SiLU），经 `ProjectionPolicy::ForwardDecodeRows` 与新增的 `TryFusedGateUpRows` 接入；NF4 策略显式返回不适用，其路由完全未变。多行版本按 96 次 launch 完成同一工作，与 NF4 的组织方式一致。

每个 `(row, sequence)` 仍使用原 subgroup size、原 vec8 载入、原 FMA 顺序与原 `reduce_over_group`，因此与逐行 kernel 逐位相同。`in_dim` 非 8 的倍数时回退逐行 kernel（保证序列偏移保持 vec8 对齐），`rows==1` 仍走原入口。网格 group 编号取 `row * rows + sequence`，使读同一权重行的各序列相邻，便于 L2 复用。

四种配置的完整 logits 对照（每配置 72 行、共 288 行，与独立 `Forward` Session 比较，B=1 要求逐位一致）全部 `failed_values=0`、`max_abs=0`、`normalized_rms=0`、greedy 未变；容量为 1 的 B=4 用例与 snapshot 重放同样通过。`ctest --test-dir build-refactor` 54 项 100% 通过（0 失败；14 项 GPU fixture 在 CPU 组按设计跳过）。

### 吞吐

同一 `--sessions 4`、10 轮、context 2048、new-tokens 64 下，改动前后对比（batch tok/s p50，含服务传输与观测开销）：

| 输入 | 并发 | 上限 1（前/后） | 上限 4（前/后） | 上限 4 相对上限 1（前→后） |
| --- | ---: | --- | --- | --- |
| 短 | 4 | 128.67 / 130.00 | 138.33 / 204.12 | +7.5% → +57.0% |
| 长短混合 | 4 | 95.50 / 98.11 | 99.82 / 132.44 | +4.5% → +35.0% |
| 短 | 1 | 91.05 / 91.04 | 89.24 / 90.85 | 不变（B=1 路径未改） |

上限 4 绝对吞吐提升 +47.6%（短）/ +32.7%（混合）；上限 1 的变化在 ±2.7% 内，与既有批间波动同量级。BF16 HTTP 合批夹具通过并观察到执行批宽 2/3/4（多行批次 58 次）。

### 近容量长 prompt

context 4096、`--long-prompt-repeat 190`，模板化后实测 Qwen3 3725 token、0.8B NF4 3916 token（短输入 27 token）。上限 1 与 4 各为独立进程，并发 4 的长短混合中上限 4 相对上限 1：Qwen3 +8.8%（31.27→34.01）、0.8B NF4 +11.7%（32.20→35.97）；单请求 +5.6%/+9.0%。长 prompt prefill 仍按原分块轮转。

### 小时级长稳

runner 单进程 watchdog 上限 3600 秒，因此以背靠背独立进程累加，不把一次超长进程当作覆盖：

| 配置 | 进程 | 正式请求 | 测量批次 | 累计时长 | 每进程 p50 tok/s |
| --- | ---: | ---: | ---: | ---: | --- |
| Qwen3 0.6B NF4 短输入 | 2 | 8024 | 4000 | 1.09 h | 130.60 / 130.54 |
| Qwen3.5 0.8B NF4 短输入 | 6 | 7752 | 3840 | 0.98 h | 153.45–154.77 |
| Hybrid 0.8B BF16 夹具短输入 | 6 | 7752 | 3840 | 0.76 h | 200.70–204.09 |

全部进程 `failed`/`rejected`/`slow_client_cancellations` 为 0、无框架日志错误、final report 正常关闭、Job 排空无强杀。同配置内 p50 波动为 0.05%（Qwen3）、0.86%（NF4）与 1.68%（BF16），说明所测时长内无持续退化；BF16 是本轮唯一改动的执行路径，其小时级覆盖与吞吐提升使用同一批产物。

### 执行批宽

四种配置在 `--decode-batch-size 4` 下运行 HTTP 合批夹具，Worker trace 记录的多行批次：Qwen3 0.6B NF4 48 次（2×28、3×4、4×16）、0.8B NF4 58 次（2×47、3×9、4×2）、Hybrid BF16 50 次（2×30、3×4、4×16）、4B NF4 48 次（2×28、3×4、4×16）。第34节遗留的 4B 在 HTTP 未观察到宽 4 已关闭。

### 保留的未通过证据

两次客户端传输失败按原样保留、不计入达标样本，门禁未放宽：

1. `hour-soak-nf4/nf4-b4-mixed-r1000-p1`：460 请求后客户端 `OSError WinError 10055`（WSAENOBUFS）加 3 次 `BrokenPipeError 10058`；引擎侧 0 失败/0 拒绝、456 完成/3 取消、优雅关闭。审查见 `nf4-soak-failure-review-20260920.json`。
2. `hour-soak-nf4b/nf4-b4-short-r1000-p1`：采样器同样命中 `WinError 10055`。审查见 `socket-exhaustion-review-217e84dec72d.json`。

两次都不是推理或设备故障：本 harness 按服务要求为每个请求和每次 health 探测新建 loopback 连接，在这台同时运行桌面应用的机器上，累计连接 churn 会间歇耗尽主机套接字缓冲。复核后改用更小进程（每次 320 轮）完成小时级覆盖，全部通过。

### 机器守卫

本轮开始时守卫因非提权快照读取受阻（受 ACL 保护的 LiveKernelReports PoW32kWatchdog 无法枚举，而守卫在存在读取错误时既拒绝放行也拒绝确认）。经与操作者确认，改为通过 `elevate.ps1` 在提权子进程中执行受守卫保护的运行；守卫规则未修改，提权重读后 `read_errors` 归零，放行经守卫自身的显式复核接口完成（`guard-review-20260920.json`、`guard-acknowledgement-20260920.json`、`guard-acknowledgement-20260920b.json`）。操作者确认新增的 TDR/BSOD 上报由 ComfyUI 产生、与本项目无关。所有确认均先断言证据仅含 Windows 内核报告、且无 Aila 二进制或进程参与。

### 重复测量分布

单轮进程无法区分收益与批间波动，因此 2026-09-21 对四种配置各运行三个独立进程（共 24 个进程，每进程 10 轮短输入），并发 4 下上限 4 相对上限 1 的均值：

| 模型/权重配置 | 上限 1 | 上限 4 | 变化 | 两臂跨进程极差 |
| --- | ---: | ---: | ---: | --- |
| Hybrid 0.8B BF16 夹具 | 129.46 | 200.77 | +55.1% | 1.0% / 0.3% |
| Qwen3.5 0.8B NF4 | 119.02 | 148.92 | +25.1% | 0.8% / 2.0% |
| Qwen3.5 4B NF4 | 45.97 | 51.87 | +12.8% | 4.0% / 0.1% |
| Qwen3 0.6B NF4 | 102.69 | 125.91 | +22.6% | 1.0% / 3.2% |

并发 1 时四配置的变化为 -0.8%/-0.7%/+0.2%/+0.7%。即并发 4 的收益（+12.8%..+55.1%）远大于各臂自身的跨进程极差（0.1%..4.0%），而单请求路径的差异落在噪声内，与"上限 4 不改动 B=1"一致。原始值见 `p7-repeat-measurements.json`。
### 附带修复：legacy CLI bench 的上下文越界崩溃

为测量 attention 占比而运行 legacy CLI bench 时，发现 `Aila.exe --bench --bench-pp 4096 --bench-tg 8` 以 `0xC0000409`（STATUS_STACK_BUFFER_OVERRUN）在 warmup 阶段终止，紧接在 `seq_cap=4096` 的缓冲扩容日志之后。边界是精确的：总量 4096（pp 4088 + tg 8）通过，4097（pp 4089 + tg 8）崩溃，无 profiling 时同样复现。

根因在 `src/bench/Benchmark.cpp`：它先 `benchmark_prefill(prompt)` 再 `benchmark_decode(gen_length)`，但从不检查 prompt 与生成 token 之和是否超过模型上下文。decode 循环持续追加到同一 KV cache，因此总量超过 `max_seq_len` 时越界写入。该路径是 legacy 单行 CLI bench，不属于 P7 合的共享 Model/Session 路径——同一 3725-token 模板化 prompt 经服务路径运行正常（`tmp/refactor/p7-batching-perf-20260918/prefill-4096-shared`）。

修复：在 warmup 之前用 `engine.max_context_length()` 比较 `actual_pp + cfg.gen_length`，超限时记录可操作的错误并返回携带被拒尺寸的结果，而不是继续执行。复验：pp 4089 + tg 8 现在退出 0 并输出上下文上限错误（`fix-verify-4089.log`），pp 4088 + tg 8 正常完成（`fix-verify-4088.log`）；`ctest` 54 项仍 100% 通过。完整证据与边界测量见 `finding-cli-bench-4096-crash.json`。
### 多行 decode attention

此前 decode 批次中 attention 仍逐行调用：每步每层对每个独立序列各发一次 `attention_decode`。本轮新增 `attention_decode_rows` 与 `attention_decode_rows_baseline_impl`，为 Hybrid 路径把同一步的多个序列合并到一次 launch。每个 `(sequence, head)` 仍使用原 baseline kernel body：原 vec8 K 点积、原 `reduce_over_group` 顺序、原 V 累加循环，因此单行输出与逐行 kernel 逐位相同。

覆盖范围经过刻意收窄：只有 head_dim 256、无 decode window/sink、且每行都低于 exact-partial-merge 阈值（1024）时才走多行 kernel。head_dim 128 会优先选择 joint-matrix 或 exact partial merge，因此 Qwen3 保留逐行派发；达到阈值的行同样回落逐行。多行 kernel 只复刻 baseline body，所以任何可能落到其他 kernel 的行都不会被它替换。新增 `AILA_ATTN_DECODE_ROWS=0` 可在同一二进制上恢复逐行调用，用于 A/B。

数值：四配置完整 logits 对照仍全部 `failed_values=0`、`max_abs=0`、greedy 未变（q3/nf4/nf4_4b 各 144 行 `[NUMERICS]`，bf16 72 行）；`ctest` 54 项 100% 通过；四配置 HTTP 合批夹具通过并观察到执行批宽 2/3/4（`trace-widths-09`）。

收益用同二进制交替 A/B 测量（`AILA_ATTN_DECODE_ROWS=1` 与 `=0`，三轮交替，批上限 4、并发 4、各 10 轮）：

| 配置 | 多行 p50 (n=3) | 逐行 p50 (n=3) | 变化 |
| --- | --- | --- | ---: |
| Hybrid 0.8B BF16 夹具 | 208.76 [207.81–210.34] | 200.90 [200.28–201.64] | +3.9% |
| Qwen3.5 0.8B NF4 | 155.32 [154.72–156.50] | 151.15 [150.47–151.62] | +2.8% |

六组配对（3 轮 × 2 配置）全部倾向多行，幅度 +2.2%..+4.8%，两臂区间不重叠。

自查该改动时发现并修复两处潜在错误：回落路径按 `num_heads*head_dim` 而非常数参数 `q_row_stride` 构造逐行 q 视图；多行 kernel 用 `q_row_stride` 写输出，而 attention 输出始终是 `num_heads*head_dim` 的稠密行栈。二者在本次四个配置下不可见，因为它们都启用了 `attn_output_gate`，此时两个 stride 恰好相等，所以逐位对照全部通过；但未启用输出门控的模型会读到或写到错误偏移。现已分别改为 `q_row_stride`（输入）与独立的 `o_row_stride = num_heads*head_dim`（输出）。修复后复验：四配置完整 logits 仍全部 `failed_values=0`、`max_abs=0`、greedy 未变（q3/nf4/nf4_4b 各 216 行、bf16 72 行），`ctest` 54 项 100% 通过。记录见 `p7-attention-rows-stride-fix.json`。收益只有 2–5% 的原因是可解释的：attention 约占批次单步的 20%，且多行版本减少的是 launch 与主机提交开销，不是算术量本身。证据见 `p7-attention-rows-summary.json`。
#### 多行精确合并覆盖（2026-09-22）

上一轮多行 attention 只覆盖 head_dim 256 且低于 exact-partial-merge 阈值的行，joint-matrix（head_dim 128）与两个 head_dim 的 partial-merge 达标行仍逐行派发。本轮新增 `attention_decode_rows_exact_merge_impl<Quantized, HeadDim 128|256>`：每个 `(sequence, head, tile)` group 复用单行 merge 的算术体、`native::exp` 与 `reduce_over_group` 顺序，phase 1 的扁平网格按各行的 tile 数展开，phase 2 每 `(sequence, head)` 一个 work-group。

partials 区域改为 `[4, max_attn_heads_, tile, stride]`，每条序列一段私有区间，因此多行 merge 不需要跨行仲裁；batch 上限固定为 4，这也与最多 4 条并发 decode 序列的 scratch 需求一致，且与上下文长度无关。单行 merge 现从 `ndim` 推导 tile 数，4 维时取倒数第二维。

派发规则对齐单行调度：`head_dim 256` 且 partials 不可用或未到阈值 → 多行 baseline；`head_dim 128/256` 且 partials 可用并已到阈值 → 多行 exact merge；其余（decode window/sink、head_dim 128 低于 128 时优先 joint-matrix、no partials、跨阈值的混合行）→ 逐行。`AILA_ATTN_DECODE_ROWS=0` 仍可整体恢复逐行。

验收在四个配置上新增近阈值长上下文批次（capacity 2176，prefix 2100..2115，B=4，每步 3 次），oracle 仍是独立单序列 Session，要求逐位相同：每配置 12 行 `bitwise=true`、`greedy_changed=0`、无 `failed_values`。stderr 一次性路径日志直接记录所选择内核（如 `[ATTN] decode rows=4 head_dim=256 path=multi-row-exact-merge`），不再只靠耗时推断。证据 `p7-longctx-multirow-merge.json`。

期间发现并修复一个真实缺陷：初版把「能否为每行切出 4 维视图」与「单行是否会走 merge」混为一谈，并假设 partials 的 head 维等于本层 head 数。Hybrid 的该区域按最宽 attention 层分配（`max_attn_heads_=16`，本层 `num_heads=8`），于是判据失败，长行被错误送入 baseline body，NF4 长上下文第一行即非逐位相同。修正为：baseline 判据只看 partials 是否可用，多行访问用缓冲区自身的 head 维作每行 block 步长。该失败已保留为独立证据（`guard-review-33896ade586e.json`、`guard-ack-33896ade586e.json`、`backend-rows-longctx/nf4/`），未重分类为通过。
### 结论边界

吞吐为 runner 的批次完成 token 率（含请求准备与服务传输），不是 kernel 吞吐；小时级覆盖约 1 小时、集中在两个配置、非跨天运行；显式 USM 观测不代表驱动私有分配或全部 WDDM/RSS。batched/ragged attention 与分页 KV 仍未实现。证据目录 `tmp/refactor/p7-batching-perf-20260918/`，汇总 `p7-batching-performance-summary.json` 与 `p7-bf16-rows-summary.json`。不宣称 P7 全阶段或整个重构完成。

### 36. 2026-09-22：多行 Joint-Matrix Decode Attention、宽3合批扩展与端到端实测

本轮对连续合批的最后一条注意力逐行回退路径（head_dim 128、未量化 KV、序列长度 < 128 tokens）实施多行内核与端到端验收：

1. **多行 Joint-Matrix Decode Attention 内核实现与数值验证**：
   - 新增 `attention_decode_rows_joint_matrix_impl<TM=1, TN=8, TK=16, SG=8, WG=128>`。在短上下文（< 128 tokens）下，全序列 logits 不超过 128 个 float（512 字节），完全分配在工作组本地 SLM（`scores_slm`）中，彻底消除中间分数对全局 DRAM `scores_buf` 的往返读写。
   - 提取硬件级统一检测与配置 `get_jm_decode_config`，在 `attention_decode_rows` 中接入安全派发。
   - 内核质量加固：在 sub-group 矩阵存储后增加 `sycl::group_barrier(sg)` 规避 RAW 风险；增加 `t < 128` 防御性上界保护；将 `max_val` 初始值置为 `-1.0e38f` 规避数值下溢。
   - **真实 Arc A770 GPU 验证**：执行 `AilaQwen3BatchTests.exe --model models/qwen3-0.6B-bnb-nf4-offline`，控制台显式确认多行 JM 内核激活：
     ```text
     [ATTN] decode rows=2 head_dim=128 path=multi-row-joint-matrix
     [NUMERICS] fixed B=2 sequence=0 position=0 values=151936 failed_values=0 max_abs=0 normalized_rms=0 greedy_changed=0
     [NUMERICS] fixed B=2 sequence=1 position=1 values=151936 failed_values=0 max_abs=0 normalized_rms=0 greedy_changed=0
     [NUMERICS] fixed B=3 sequence=0..2 values=151936 failed_values=0 max_abs=0 normalized_rms=0 greedy_changed=0
     [NUMERICS] fixed B=4 sequence=0..3 values=151936 failed_values=0 max_abs=0 normalized_rms=0 greedy_changed=0
     ```
     与独立单序列参考相比，所有行全部 `failed_values=0`、`max_abs=0`、`greedy_changed=0`，贪婪解码零漂移。

2. **服务端非2幂次并发（宽3）扩展与套接字验证**：
   - 在 `service_benchmark.py`、`batching_socket_tests.py` 与 `socket_tests.py` 中支持并发宽 3（`(1, 2, 3, 4)`）。
   - 在编译出的服务端 `AilaServer.exe` 上实测运行，全部 5 组批宽配置（default、explicit_one、batch_two、batch_three、batch_four）全量通过，流式与独立基准完全对齐。

3. **真实 GPU 端到端 HTTP / SSE 调用与小型并发压测**：
   - 单请求测试（`incremental_utf8_json_sse`）：耗时 19.9 秒（含冷启动加载），生成的 JSON/SSE 语义连贯准确，逐字节对齐。
   - 并发流式测试（`concurrent_1_2_4_fixed_seed`）：
     - 宽度 1：峰值并发 1，耗时 0.279s；
     - 宽度 2：峰值并发 2，耗时 0.490s / 0.493s（双流完全并行）；
     - 宽度 3：峰值并发 3，耗时 0.725s / 0.725s / 0.725s（三流完全同步）；
     - 宽度 4：峰值并发 4，耗时 0.482s / 0.478s / 0.962s / 0.963s。
   - 完整合批功能套件（`batching_gpu_tests.py`）：单请求等价、宽 1/2/3/4 并发、中途断线隔离复用（`disconnect_peer_and_slot_reuse`）与服务器带请求优雅关闭（`graceful_shutdown_with_active_requests`）全量通过。17 请求准入，14 成功，3 受控取消，0 失败，0 拒绝。

4. **思维链历史污染清洗（Chat Template）**：
   - `strip_thinking_blocks` 成功剔除导入多轮历史中的旧思考内容，231 项聊天套件 100% 通过。

