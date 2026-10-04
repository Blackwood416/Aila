# Qwen3.8 Flash Next 原始 NF4 异构推理 H9 结果（2026-10-04）

状态：Stage1–7 已完成验收与测量，实验入口启用单独的跨层融合，HC 并行保持关闭。本文按保存的 JSON、原始日志及源码快照记录结果，不合并不同 EXE 的测量。

H9 尚未达到 20 tok/s。CPU row4、静态 chunks、resident router fusion、block64、HC 并行及缩小 GPU 专家缓存均没有获得可推广的端到端收益。单独 pending MLP residual 在两个独立的同 EXE 三轮活动中分别改善 0.64%（Stage6）和 0.42%（Stage7），不合并不同 EXE 的样本。最新 Stage7 五请求总耗时中位数从 51.398 s 降到 51.181 s，返回长文 decode 从 13.6 到 14.0 tok/s。实验入口启用单独融合；这些是小幅改善，尚不足以完成目标。

## 1. 测量口径与可复现身份

沿用 [H8 结果与限制](qwen3.8-flash-next-heterogeneous-h8-results-2026-10-03.md)：原始 NF4 模型，A770，Windows/oneAPI，mapped CPU cold source，CPU NF4-Q8，host RAM pool 真正为 0，short fill parts=1。H9 性能活动固定 CPU threads=12、credits=10、cold policy=1、global fill parts=8、fill threads=8；mapped prefetch/touch、host complement 关闭。模型、输入、profile、参数、工具链和 GPU 身份以每条 `.meta.json` 和汇总器 `comparison_identity` 为准。

CPU screen 每 arm 是一个独立进程，三个历史请求（prompt 41 / 1833 / 47 tokens），每请求生成 24 tokens。正式活动每 arm 三个独立进程，每进程五个历史请求（41 / 1833 / 47 / 1833 / 41），每请求生成 48 tokens，偶数轮逆序运行 arms。总耗时是请求耗时之和，不包含模型加载；decode tok/s 和 TTFT 独立报告。单样本 CV=0 不表示稳定。诊断、shadow verification、同步 stage profiling、watchdog 失败和配置不符样本不进入性能组。

各阶段源码与 CLI 已冻结，不能把当前工作树或当前 EXE 当作历史测量来源：

| 阶段 | EXE SHA256 | 保存文件与源码清单 |
|---|---|---|
| S1 | `58675F23B982F1C3557297E0899A322F825269C7694742B98AA076C4E41C308C` | [CLI](../../tmp/h9-stage1-Aila.exe)、[9 项 source manifest](../../tmp/h9-stage1-source.json) |
| S2 | `473765A4939227FD13B4CD1B61F3988FFDC38B39C77870948945A26B08CC1F55` | [CLI](../../tmp/h9-stage2-Aila.exe)、[9 项 manifest](../../tmp/h9-stage2-source.json) |
| S3 | `93977C71437107923869924705B250413D8888A98CF39AA8D606069F7529A12E` | [CLI](../../tmp/h9-stage3-Aila.exe)、[12 项 manifest](../../tmp/h9-stage3-source.json) |
| S4 | `C317928B53D34A69011041CF5E16074F05485E1B7CFA9418A517EAEC8CCC2E1B` | [CLI](../../tmp/h9-stage4-Aila.exe)、[12 项 manifest](../../tmp/h9-stage4-source.json) |
| S5 | `31BCE7F6C3360BB64551753831E70B94A22E5A1EE1B525D986DA67AFFD57B927` | [失败候选 CLI](../../tmp/h9-stage5-Aila.exe)、[15 项 manifest](../../tmp/h9-stage5-source.json) |
| S6 | `3C5783EAA0D9854D0578E5AC84B219D339ACA43302866E678F43AC418AC19F24` | [CLI](../../tmp/h9-stage6-Aila.exe)、[15 项 manifest](../../tmp/h9-stage6-source.json) |

每项 manifest 保存编译相关源码的路径、快照路径及 SHA256。临时 microbench 的 EXE/库身份另外保存在各 micro JSON 中，不能用上表 CLI 哈希替代它们。Stage6 cold acceptance 使用独立测试 EXE，身份与局限见第 6 节。

## 2. CPU Stage1–4：正确性修复与负结果

### 2.1 尝试与修复

Stage1 尝试一次处理相邻输出行的 NF4-Q8 row4 路径，并增加固定逻辑 chunks 调度。合成 CPU 测试在 nested/flat scales、1/4/12 threads、混合四 token/最多 40 task 等场景通过，见 [chunks0](../../tmp/h9-row4-chunks0.log)、[chunks3](../../tmp/h9-row4-chunks3.log)、[chunks8](../../tmp/h9-row4-chunks8.log)。但真实权重 microbench 在 layer19 的 h 出现 bitwise 差异，[原始失败日志](../../tmp/h9-micro-c0-b0.log)保留；不能用合成测试通过代替真实来源验证。

Stage2 修复 row4 SwiGLU 输出循环：在 fast-math 下阻止编译器将两个标量 `expf` 合并为 SVML vector exp，保留 H8 标量路径的 BF16 舍入。修复可在 S1/S2 冻结 CPU ops 源码中逐项核对。随后 [真实来源检查](../../tmp/h9-micro-diff.log)与 [S2 micro 原始日志](../../tmp/h9-stage2-micro-c0-b1.log)的所有选中层 h/out bitwise 通过，没有放宽阈值。[S2 build](../../tmp/h9-stage2-build.log)及 [readiness](../../tmp/h9-readiness.log)保留。

Stage3 将 row4 热循环改为 template-inline，并加入 resident router fused 路径；后者合并 resident 路由处理的部分 launch，保持分类与数值约束。首轮构建因测试中的 `const sycl::event` 调用非 const `wait_and_throw()` 失败；[失败 build](../../tmp/h9-stage3-build.log)及 [修复 build](../../tmp/h9-stage3-build-repair.log)分别保留。[S3 CTest](../../tmp/h9-stage3-ctest.log)六项通过；fused resident router 测试覆盖 624 个 bitwise case、host/device mirrors 和边界。

Stage4 block64 用 256-bit 操作解码 64 个 NF4 权重，减少 packed load/shuffle 指令，保持原来的 half/FMA 顺序。该候选与 row4 互斥，默认关闭。真实运行的 [shadow verification 原始日志](../../tmp/h9-stage4-actual-verify.err.log)和 [诊断解释](../../tmp/h9-stage4-review-summary.json)记录候选 h/out 与 enum6 H8 基线在发布前 bitwise 一致。这里计数是每层 CPU batch，并非 expert task；三请求各八次 forward，启动 warmup 不计入请求总数。这证明被执行 batch 的等价性，不能替代完整生成质量测试。

### 2.2 同阶段 screen

百分比为总耗时相对同阶段 baseline，正值表示更慢。每 arm 仅一个进程，细小差异不构成重复性证据。

| 阶段 | arm | 三请求总耗时 s | 相对 baseline |
|---|---|---:|---:|
| S2 | baseline | 24.012 | — |
| S2 | row4 | 24.137 | +0.52% |
| S2 | chunks3 | 23.957 | −0.23% |
| S2 | row4-chunks3 | 24.145 | +0.55% |
| S3 | baseline | 23.896 | — |
| S3 | row4 | 24.014 | +0.49% |
| S3 | fused | 23.882 | −0.06% |
| S3 | row4-fused | 24.010 | +0.47% |
| S4 | baseline | 23.957 | — |
| S4 | block64 | 24.112 | +0.65% |
| S4 | block64-fused | 23.960 | +0.01% |
| S4 | fused | 23.963 | +0.03% |

原始 request 指标、metadata、日志哈希和来源见 [S2 summary](../../tmp/h9-stage2-screen-summary.json)、[S3 summary](../../tmp/h9-stage3-screen-summary.json)、[S4 summary](../../tmp/h9-stage4-screen-summary.json)。不跨这些 EXE 求平均，不改变默认 CPU flags。

早期 [H8 EXE thread screen](../../tmp/h9-h8-thread-screen-summary.json)记录 cpu8 / cpu6 / cpu12valid 总耗时 58.611 / 24.955 / 24.275 s。先运行的 cpu8 有明显冷暖/顺序干扰，这不是“58% 提速”的证据，也不足以确认 12 threads 最优。另一个 cpu12 启动被 RuntimeDefaults 清除了预期 H8 flags，已停止并 [明确排除](../../tmp/h9-screen-cpu12-excluded.txt)。正式 H9 保持 12 threads，未完成公平的线程数多轮比较。

### 2.3 真实 mapped 来源 microbench

[临时源码](../../tmp/h9-real-source-bench.cpp)使用 lazy model 的原 NF4 mapped views，owner 贯穿计算；[task 文本](../../tmp/h9-real-source-tasks.txt)来自一次真实 decode 的 48 层选中冷 CPU 专家。每 pass 逻辑来源 649,637,888 bytes，超过 L3；没有 USM/GPU 或整个 expert 复制。输入是确定性合成 BF16 hidden，**不是真实 hidden 激活**。计时前旧/新路径逐层 bitwise 检查会预热来源，因此此结果是 warm micro，不能解释成 cold file cache 性能。

| 快照 | threads / arm | 中位 GU+quantize+down ms | 相对该快照基线 |
|---|---|---:|---:|
| S2 | 12 / baseline chunks0 | 21.337 | — |
| S2 | 12 / row4 chunks0 | 23.502 | +10.15% |
| S2 | 12 / baseline chunks3 | 22.760 | +6.67% |
| S2 | 12 / row4 chunks3 | 25.315 | +18.64% |
| S3 | 12 / baseline | 21.014 | — |
| S3 | 12 / row4 | 24.116 | +14.76% |
| S3 | 18 / baseline | 22.417 | +6.68% |
| S3 | 18 / row4 | 19.616 | −6.66% |
| S4 | 12 / baseline | 20.875 | — |
| S4 | 12 / block64 | 20.351 | −2.51% |

见 [S2 micro JSON](../../tmp/h9-micro-summary.json)、[S3 micro JSON](../../tmp/h9-stage3-micro-summary.json)、[S4 micro JSON](../../tmp/h9-stage4-micro-summary.json)。各 JSON 保留原始日志哈希及 benchmark/库身份；代表原始日志为 [S2](../../tmp/h9-stage2-micro-c0-b0.log)、[S3](../../tmp/h9-stage3-micro-t12-b1.log)、[S4](../../tmp/h9-stage4-micro-t12-b2.log)。CPU stage wall time 与包含 hashes/checksum、指标和输出的 diagnostic loop wall time 不混用。

S3 的 18-thread isolated row4 结果不能证明固定 12-thread 推理活动获益。S4 block64 在 warm micro 小幅变快，screen 总耗时却没有改善，说明不能仅靠孤立 kernel 结果推广。

最初 micro JSON parser 截断了十六进制 h/out hash，S4 checksum 也丢失精度。已用 [修复脚本](../../tmp/h9-repair-micro-json.py)从原始 hash-verified 日志重新解析，保留 timing 和原始 artifact 身份，三个 JSON 均有 `parser_correction`。本文引用修正后的 JSON；数值 bitwise 依据是预检查和测试，不是截断的历史 JSON。

## 3. CPU 与同步 stage 诊断的边界

[S4 CPU diagnostic summary](../../tmp/h9-stage4-cpu-diagnostic-summary.json)明确为诊断样本。五个 48-token 历史位置的 CPU compute/token 中位数依次为 47.915 / 35.492 / 61.960 / 21.708 / 28.123 ms；对应 CPU 区间进程 PageFaultCount 增量中位数为 30,921 / 22,731 / 32,982 / 6,832.5 / 4,351.5。GU 约占测得 compute 的 64–65%，quantize <1%，down 约 34–35%，其余 arena/LUT/dispatch <0.3%。这使实际来源驻留/供给与 CPU/GPU 重叠比单独削减量化更值得继续诊断。

PageFaultCount 同时含 soft/hard fault，且属于整个进程；进程 user/kernel 时间也包括并发线程。它不能区分 CPU mapped 页、磁盘读取或 WDDM paging。当前证据不支持把差异归因于硬盘或 GPU 物理驻留，进一步归因需要 ETW 等带 fault 类型与地址映射的证据。

[同步 stages 原始日志](../../tmp/h9-stage4-wall-stages.err.log)及 [解释 JSON](../../tmp/h9-stage4-review-summary.json)保留 HC/linear/router/experts 等 host wall buckets。`AILA_PROFILE_Q4E_STAGES` 在边界同步 GPU，改变正常重叠；router 同时 drain 前面的 resident/shared GPU 工作，experts 包含 CPU/cold/acquire/merge。这些不是正常流水线的互斥 device operator 时间，不能把独立中位数相加，也不能当作正常执行的占比或可直接节省的时间。

## 4. Stage4 GPU 专家缓存预算

沿用 S4 同一个 EXE，固定全部 CPU/融合实验关闭，仅改变 `EXPERT_CACHE_MB` 和对应有效 slots。slot bytes=2,534,400；显式预算 slots 必须是 `floor(MB*1048576/2534400)`，baseline auto=3567，8192=3389，7168=2965，6144=2542。profile 排名与其它运行条件保持一致，热集数量适应容量。

[四 arm screen](../../tmp/h9-gpu-budget-screen-summary.json)总耗时为 auto 57.494、8192 24.342、7168 24.618、6144 24.852 s。首先 auto 明显冷慢；这些单进程数值保留，但**不能据此声称缩预算约 58% 提速**。随后的三轮逆序正式比较才是本轮判断依据。

| 正式 arm | round1 / round2 / round3 总耗时 s | 中位数 s | CV |
|---|---|---:|---:|
| auto, 3567 slots | 53.645 / 51.376 / 51.384 | 51.384 | 2.05% |
| 8192 MiB, 3389 slots | 52.198 / 51.678 / 51.463 | 51.678 | 0.60% |

[正式 summary](../../tmp/h9-gpu-budget-formal-summary.json)记录 8192 总耗时 +0.57%，返回长文 decode 13.6 → 13.1 tok/s。减少预算没有成为性能改善，不推广 8192；7168/6144 仅有 screen，未完成同等正式重复。formal 的每条 log/meta 路径与哈希可从 summary 追溯。

### GPU memory diagnostic

[诊断 summary](../../tmp/h9-gpu-memory-summary.json)按真实 PID、adapter LUID、physical adapter 索引保留 PDH counters；[auto 原始 JSON](../../tmp/h9-gpu-memory-auto.gpu-memory.json)和 [8192 原始 JSON](../../tmp/h9-gpu-memory-8192.gpu-memory.json)属于独立诊断运行，排除性能比较。

| 进程 peak bytes | auto | 8192 |
|---|---:|---:|
| dedicated usage | 15,329,267,712 | 14,880,010,240 |
| shared usage | 226,697,216 | 226,631,680 |
| committed | 15,555,997,696 | 15,106,674,688 |

A770 adapter dedicated peaks 为 16,343,769,088 / 15,879,094,272 bytes。减预算确实降低了所记录 usage/committed，但这些 PDH counters **不是物理驻留、DXGI budget、分页量或 GPU event time**；adapter 和 process peaks 也未必来自同一采样时刻。auto 34 个采样中 33 个 available，8192 34/34；进程尚未出现时 unavailable 不当作 0。采样会干扰 wall time。不能把这里的高 committed 或较小 usage 单独解释成吞吐瓶颈的原因。

## 5. Stage5 失败与 Stage6 pending MLP residual

从本地 Strata 的 delayed residual/下一层消费结构借鉴调度思路，落到 Aila 需要保持 BF16 中间舍入和实际 buffer 所有权，而非直接搬用 FP32 数学。参考 [Strata layer.cpp](../../tmp/Strata/src/core/layer.cpp)，本轮 Aila 实现在 S5/S6 source manifest 的 Backend 与 HyperConnectionOps 中保存。

候选仅 count=1 decode、默认关闭 `AILA_Q4E_DECODE_PENDING_MLP_RESIDUAL=0`。上一层 MLP residual 在下一次 HC norm 消费，使 ordinary logits=1 token 最多省去 47 次独立 residual launch。PLE、最终 mixer/head、no-logits 返回、snapshot/observer、异常边界必须物化；消费必须先于 y/inject scratch 被下一层覆盖，无热路径动态分配。

S5 build 通过，但 [S5 六项 CTest](../../tmp/h9-stage5-ctest.log)出现一项失败：dim257 的 hidden bitwise 不同。之前 scalar fused residual/norm 的近似容差证据不能证明旧 vec8 的 BF16 bitwise 等价。失败候选和日志已保存，未放宽断言。

S6 为可安全 vec8 shape 使用专门 primitive：沿用旧 vec8 residual expression，显式 BF16 store，WG 内 global/local barrier 后按旧 grouped norm 的 lid+256 遍历与 reduction 顺序计算。dim257 等非 vec8 shape 使用原 residual add + grouped norm 两次 launch fallback。materialize 先清 ticket 所有权，防止 norm 提交异常后 double inject。

[局部严格测试](../../tmp/h9-stage6-ctest-module-verbose.log)比较更新 hidden 与 norm 输出 uint16 bits，并覆盖 BF16 tie/cancellation、多层 y/inject 覆盖顺序、final mixer、重复 materialize/异常退出；[局部 test manifest](../../tmp/h9-stage6-module-source.json)不是完整 CLI manifest。之后 [完整 S6 CTest](../../tmp/h9-stage6-ctest.log)六项通过，36.40 s；CLI 用第 1 节 S6 完整快照身份。

普通 forward 的 `[Q4E PendingMlp]` 是 host dispatch 计数，只有 opt-in + WorkSummary 才记录，不添加 GPU event profiling。正式 candidate 日志中 positive logits=1 decode 为 deferred=48、fused=47、materialized=1；日志=0状态 forward 为 fused=46、materialized=2。这证明普通路径实际消费 pending，没有仅因 dump/stage 观察模式而全部退回即时物化；它不等于 47 次 launch 的 device 时间收益。

## 6. Stage6 正式性能与 teacher canary

[单轮 screen](../../tmp/h9-stage6-screen-summary.json)的 32.128 → 24.166 s 有冷暖顺序干扰，**不报告 25% 提速**。下面采用 [同 EXE 正式三轮 JSON](../../tmp/h9-stage6-formal-summary.json)：

| arm | round1 / round2 / round3 总耗时 s | 中位数 s | CV |
|---|---|---:|---:|
| pending=0 baseline | 53.873 / 51.356 / 51.122 | 51.356 | 2.39% |
| pending=1 | 51.127 / 51.026 / 50.924 | 51.026 | 0.16% |

中位总耗时 −0.64%；round2/3 paired 差值为 −0.331 / −0.197 s。第一轮 baseline 较慢的 53.873 s 完整保留，不删样本放大收益。不同请求并非一致改善：

| 请求位置 | baseline → pending 耗时中位数 s | decode tok/s |
|---|---|---|
| 首次 story, prompt41 | 6.840 → 6.886 | 9.8 → 9.7 |
| 首次 archive, prompt1833 | 16.714 → 16.622 | 11.1 → 11.2 |
| code, prompt47 | 6.700 → 6.716 | 9.8 → 9.7 |
| 返回 archive, prompt1833 | 15.539 → 15.402 | 13.6 → 14.1 |
| 返回 story, prompt41 | 5.563 → 5.396 | 12.3 → 12.9 |

### Canary 范围

短文和长文各 48 个 teacher rows，原始 [story log](../../tmp/h9-stage6-canary-story.log)、[story metadata](../../tmp/h9-stage6-canary-story.meta.json)、[archive log](../../tmp/h9-stage6-canary-archive.log)、[archive metadata](../../tmp/h9-stage6-canary-archive.meta.json)保留。测试 EXE SHA256 为 `354959D62E802E3CA0DDDC789BD07D95368310B339D2EDEFFBBD92B1A6E02811`，与性能 CLI 不同；没有单独冻结该 S6 测试 EXE。其构建相关源码来自 S6 完整 manifest，但该 manifest 的 15 项没有独立保存当时的 cold acceptance harness，因此历史测试 EXE 的完整源码复现还存在限制，不能用 Stage7 当前 harness 代替历史身份。

| CPU mode2 对 GPU reference | mean KL | mean NLL delta | max logits NRMSE | CPU tasks |
|---|---:|---:|---:|---:|
| story | 0.001966874 | −0.008564113 | 0.05570596 | 15,227 |
| archive | 0.003007233 | −0.005476215 | 0.06035771 | 8,735 |

两项 `COLD MODEL PASS`，nll_warning=0，replay/checkpoint/cancellation 通过，live_usm_delta=0。GPU mode1 对其 GPU reference 48/48 exact rows，max logits NRMSE=0。CPU mode2 各 1/48 exact rows，以既定 KL/NRMSE 质量门槛验收，不能称 CPU logits bitwise 等价。

**S6 GPU mode0 reference、GPU mode1 和 CPU mode2 都启用 pending。** 这些 canaries 验证采用 candidate HC 时的既有 GPU/CPU cold quality，不能证明 candidate pending HC 对原始 HC 数学的全模型等价；当前原始 HC 等价证据仅是严格局部 Module fixture。Stage7 增加显式 reference hook，需用旧 HC math 的 GPU mode0 做新的真实模型对照。48-step teacher canary 也不代表广泛自由生成质量。

## 7. Stage7 HC 并行及跨层融合组合

候选为 decode/small-row HC down WG64/128（默认256），以及四 stream 并行 up gate（默认关闭；SLM 仅保存最终 BF16 gate，保留旧 SG16/vec8/FMA/reduction 和 g0..3 mixer 顺序）。局部严格测试覆盖真实 lowrank320/H2560/groups4、rows1/2/4、多 seed、tie/adversarial、inject=null、tails 和 sentinels。非四组配置沿用原 up；生产路径无新增 USM 分配。WG 缩小只改变输出 subgroup 的装箱，不拆分一个 dot 的归约。

测试专用 `SetDecodeHcReferenceForTest(true)` 在 drain 后强制旧 down WG256、旧 up、pending关闭；candidate 分支按环境选择。默认 runtime 不调用 hook。probe 的 `HcReferenceCanary` 只允许 `ColdQuality`，性能汇总器即使遇到伪造的 diagnostic=false 也拒绝该环境标记。

Stage7 CLI SHA256 `6B4268CCF9FA72AEB263B6C77D5E3B8269C7D4C6FB57A7A4670AF06E79BC260E`；[16 项源码快照](../../tmp/h9-stage7-source.json)包含 cold acceptance harness。独立 [canary EXE](../../tmp/h9-stage7-AilaQwen4ExpToleranceTests.exe) SHA256 `7C729D6CE9533FB9A08A1BD87D977E706B3947DC77B9A6C68E10B6AC91625CA9`。 [完整增量构建](../../tmp/h9-stage7-build.log)通过；[六项 CTest](../../tmp/h9-stage7-ctest.log)全部通过，总计 37.78 s。

### 原始 HC 真实模型对照

两组各 48 teacher rows，reference 为原 HC，candidate 为 WG64/up-parallel/pending 全开。GPU mode1 各 48/48 logits bitwise，NRMSE=0。48 行包含首次 prefill logits 与后续 47 行 decode logits；最后一个 teacher forward 验 replay 自一致性，没有额外宣称其状态与原 HC 全量逐位一致。

| CPU mode2 对原 GPU reference | mean KL | mean NLL delta | max logits NRMSE | CPU tasks |
|---|---:|---:|---:|---:|
| story | 0.002072159 | −0.008033313 | 0.06190598 | 15,020 |
| archive | 0.003007601 | −0.013118222 | 0.05126336 | 8,805 |

两组均 `COLD MODEL PASS`、nll_warning=0，replay/checkpoint/cancellation 通过，live_usm_delta=0。CPU shadow 仍使用原 NF4-Q8 enum6，不将 CPU 相对 GPU 的近似结果称为 bitwise。原始 [story](../../tmp/h9-stage7-canary-story.log)/[metadata](../../tmp/h9-stage7-canary-story.meta.json)、[archive](../../tmp/h9-stage7-canary-archive.log)/[metadata](../../tmp/h9-stage7-canary-archive.meta.json)保留。

串行控制命令有一次 PowerShell 外层展开造成的 `-ne` 错误；这是两个独立模型 probe 之间的返回码检查错误，两组 probe 自身 metadata、退出码和原始日志 hash 均单独通过。保留 [控制日志](../../tmp/h9-stage7-canaries-console.log)，后续正式活动采用独立 `.ps1` 的严格返回码检查。

### 七组短筛选

同 Stage7 EXE，24 tokens × 3 历史请求，每 arm 一个进程，[原始汇总](../../tmp/h9-stage7-screen-summary.json)：

| arm | 三请求总耗时 s |
|---|---:|
| baseline | 25.28243 |
| HC down64 | 24.05312 |
| HC down128 | 24.03196 |
| HC up parallel | 23.98567 |
| down64 + up | 24.04080 |
| down128 + up | 24.05988 |
| down64 + up + pending | 23.94697 |

首个 baseline 缓存状态较冷，不能归因为组合带来 5.28% 加速；其余 HC arms 之间只有几十毫秒，单进程筛选不支持推广。正式三轮五请求对照分别比较 baseline、pending-only、down64/up/pending，以分离跨层融合与 HC 组合的实际收益。

### 三轮正式对照及决定

同 Stage7 EXE，各 arm 三个进程 × 5 历史请求 × 48 outputs，第二轮逆序，[完整汇总](../../tmp/h9-stage7-formal-summary.json)：

| arm | 三次总耗时 s | 中位数 s | CV |
|---|---|---:|---:|
| baseline | 54.28158 / 51.29265 / 51.39757 | 51.39757 | 2.65% |
| pending-only | 51.29266 / 51.18119 / 51.00132 | 51.18119 | 0.23% |
| down64 + up + pending | 51.22527 / 51.37822 / 51.01232 | 51.22527 | 0.29% |

单独融合相对 baseline 中位数改善 0.42%，第二/三轮 paired 分别 −111.46 / −396.25 ms；第一轮较冷的 baseline 全部保留。HC 组合比单独融合中位数慢 0.086%；其 paired 差值 −67.39 / +197.03 / +11.00 ms，没有稳定叠加收益，不推广 HC 并行。

| 请求位置 | baseline / pending / HC组合 耗时中位数 s | decode tok/s |
|---|---|---|
| 首次 story, 41 | 6.88204 / 6.88502 / 6.80046 | 9.8 / 9.7 / 9.9 |
| 首次 archive, 1833 | 16.71494 / 16.63063 / 16.61422 | 11.0 / 11.2 / 11.2 |
| code, 47 | 6.66863 / 6.68674 / 6.74257 | 9.8 / 9.8 / 9.7 |
| 返回 archive, 1833 | 15.55086 / 15.42342 / 15.43177 | 13.6 / 14.0 / 14.1 |
| 返回 story, 41 | 5.55601 / 5.42520 / 5.49267 | 12.3 / 12.7 / 12.5 |

峰值 WS 30,746,468,352 B，最低系统 free RAM 27,465.52 MiB；九个进程均 exit0，无 watchdog/pressure，decode demand H2D 为 0。所有原始请求位置、输出文本及其 hash、CPU tasks、placement/热集日志保留。CPU NF4-Q8 是近似算术，ready adoption 和生成路由可以变化；这些数字不是固定 trace 的算子加速，也不保证其他对话获得同等改善。

`q4e_supply_chat.ps1 -Heterogeneous` 现在默认 pending=1；显式 `-DecodePendingMlpResidual 0` 可复现 H8 配置。普通启动入口及底层 runtime 的 pending 默认仍为0。CPU row4/block64/chunks、HC WG64/128、up并行、router fusion 与 GPU budget override 均保持默认关闭/原值。便捷使用：

```powershell
pwsh -NoProfile -File perf/qwen4exp/q4e_supply_chat.ps1 -Heterogeneous -PromptFile perf/qwen4exp/prompts/archive-natural.json
```

[8-output/3-request trace](../../tmp/h9-stage7-trace.cache.jsonl)通过 cold_queue=2 的现有 ownership/parser 检查，只作诊断。21 次常规 decode 及一次 warmup 各 deferred48/fused47/materialized1，三次请求末尾 no-logits forward 为48/46/2，位置为48/1840/54；这是 host dispatch 计数。[启动参数与环境恢复](../../tmp/h9-stage7-launcher-check.log)、[HC campaign AST/mock](../../tmp/h9-stage7-campaign-guards.log)、[parser 拒绝条件](../../tmp/h9-stage7-parser-guards.log)、[15 项 Python tests](../../tmp/h9-stage7-python-tests.log)通过。

最终 [默认实验入口 8-token smoke](../../tmp/h9-stage7-default-smoke-validation.log)通过：pending1/HC256-serial，清理继承的诊断及 HC 开关，退出后逐项恢复 caller 环境。smoke 使用 CLI 原始 chat 格式（story prompt30），与性能 corpus 自动插入 system 后的 prompt41 不混为同一测速样本。两个检查器错误分别是写死 prompt41、期待默认关闭的 WorkSummary 日志；模型退出与环境恢复都先通过，随后 marker 断言失败。保留 [初次计数检查失败](../../tmp/h9-stage7-default-smoke-initial-marker-failure.log)及 [日志开关检查失败](../../tmp/h9-stage7-default-smoke-instrumentation-marker-failure.log)，修正后重跑通过。

机器可读的 [H9 记录](qwen3.8-flash-next-heterogeneous-h9-results-2026-10-04.json)收录各阶段原始比较身份、结果、源码快照 hash、canaries、诊断、失败/排除原因及最终校验日志，不重新解析或用当前库身份覆盖旧 microbench。

## 8. 当前判断与继续工作的约束

保留所有失败、修复和负结果；CPU 新 flags、HC 并行与 GPU budget 尚不推广。单独 pending 在两个各自独立的同 EXE 活动中取得小幅整体收益，实验入口启用它；不合并两个阶段的样本。20 tok/s 目标未达到，Stage7 单独融合返回长文正式中位数 14.0 tok/s 不能替代首次短文约 9.7 tok/s 的结果。

H8 已否定的 prefetch/touch、大 host pool/重复 mapped 来源，以及此前两次变慢的 front graph，不作为本轮新的提速依据。Strata 提供的是具体供给/调度结构启发；它的 CUDA、CPU 内存平台或不同 HC tensor 数学不能用于预测 A770 吞吐。继续优化应维持原 NF4 来源、BF16 中间值和所有权约束，先通过严格参考验证，再做相同 EXE 的逆序重复。诊断计数与同步 profiling 只能帮助定位，必须与不扰动的正式 wall time 分开。

脚本及轻量校验记录：[CPU campaign](../../perf/qwen4exp/q4e_cpu_supply_campaign.ps1)、[CPU summary](../../perf/qwen4exp/q4e_cpu_supply_summary.py)、[GPU budget campaign](../../perf/qwen4exp/q4e_gpu_budget_campaign.ps1)、[GPU budget summary](../../perf/qwen4exp/q4e_gpu_budget_summary.py)、[最终 parser guards](../../tmp/h9-final-parser-validation.log)、[campaign mock validation](../../tmp/h9-final-campaign-validation.log)、[budget guards](../../tmp/h9-gpu-budget-parser-validation.log)、[budget mock validation](../../tmp/h9-gpu-budget-campaign-validation.log)。汇总器限制同 EXE/model/input/arguments/profile/stack/GPU/非变量环境，拒绝 diagnostic、verify=1、失败样本、重复 prefix 和有效 host RAM/slots 不符；这些脚本验证不是性能结果。
