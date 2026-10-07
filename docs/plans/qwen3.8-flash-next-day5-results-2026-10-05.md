# Qwen3.8 Flash Next：第五天管线优化记录

目标是 A770 单请求 decode ≥20 tok/s。当前保留的配置在 archive1833、128 outputs、第二次请求上重复达到17.6 tok/s；首次请求约12–13 tok/s。目标尚未达到。

硬件/栈：A770 driver32.0.101.9033，Xeon E5-2696v3，oneAPI2026.1 / oneDNN3.11.2。模型 `C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2`。每次只运行一个 GPU 工作负载，测量不与编译或 CPU microbench 重叠，保留8192 MiB系统可用RAM watchdog。

## 保留的精确优化

Stage13 将 GDN 环形历史行写回合入后续原 NF4 投影的独立 workgroups，保持投影原FMA与归约顺序。开关 `AILA_Q4E_GDN_HISTORY_FUSED`，核心默认关闭，实验入口的 `-Heterogeneous` 开启。

11项相关CTest通过；真实story48在原GPU/HC reference下48/48 logits逐位一致，实际调用1728次（48×36）；CPU供给质量、checkpoint/replay/cancellation、无hot USM检查通过。

Stage14同EXE长生成对照：GDN1/spin0第二请求17.5、重复17.4；GDN0/spin0为16.0。5000µs有界CPU交接spin同为17.5，保持关闭。

## Stage15：Q8输出头与CPU结果直读

两项开关均默认关闭：`AILA_Q4E_LM_HEAD_Q8`、`AILA_Q4E_CPU_RESULT_HOST_READ`。Stage15源快照与EXE归档在 `tmp/day5-s15-source.json`、`tmp/day5-s15-source/`、`tmp/day5-s15-Aila.exe`。CLI SHA256：`6E69511067A09E44AC993B628985C6BEE7B2020469DEBBDC074715C4B6AAFE75`。

Q8头将BF16头替换为signed-byte/block64 scales，减少595,968,000 B，热集由3551增加至3786专家。独立真实输出头测试中，原oneDNN中位3.2262ms，Q8含输入量化2.0145ms；三组输入NRMSE约.0082–.0084，KL约4.2e-6，argmax相同，无hot USM。

CPU直读保留原有序merge，CPU写回arena受真实GPU reader event保护。全冷、混合、重排、复用、canary及非法描述符测试通过；11项相关CTest通过（182.69秒）。真实story48运行的CPU结果H2D字节为0。

输出头质量对照先在原BF16模型导出teacher，再单独加载Q8模型导入。原始teacher `tmp/day5-original-story48.bin`；SHA256 `7B951FB6B9FD0BDC1CE237F8CF2FDF4A3814BBF3E640C31A4C65A0B150006D0D`，sidecar记录模型配置与源EXE身份。

组合story48验收：GPU KL .00079563 / 最大NRMSE .0291559；CPU KL .00232587 / 最大NRMSE .0543352；checkpoint/replay/cancel及live USM检查通过。原BF16导出运行CPU NLLdelta +.0106013触发已有alert，组合运行NLLdelta −.0113615无alert。archive原头teacher对照尚未运行。

同Stage15 EXE、GDN1/spin0/front0/GPU专家Q8关闭、18 CPU线程、archive1833、两次128-output请求，暖第二请求：

| Q8头 | CPU直读 | tok/s | 日志前缀 |
|---|---|---:|---|
| 1 | 1 | 17.1 | `day5-s15-head1-host1-spin0-gdn1-front0-fast1-gpuq80-row21-t18-trim0-spec1-scale0-a641-r1` |
| 1 | 0 | 16.4 | `day5-s15-head1-host0-spin0-gdn1-front0-fast1-gpuq80-row21-t18-trim0-spec1-scale0-a641-r1` |
| 0 | 1 | 15.9 | `day5-s15-head0-host1-spin0-gdn1-front0-fast1-gpuq80-row21-t18-trim0-spec1-scale0-a641-r1` |
| 0 | 0 | 16.9 | `day5-s15-head0-host0-spin0-gdn1-front0-fast1-gpuq80-row21-t18-trim0-spec1-scale0-a641-r1` |

这些是完整生成测量，不把独立头速度或消除memcpy数量作为管线收益。自然生成的近似CPU算术与热集admission可改变后续路由。组合17.1与全关16.9仅单轮0.2tok/s差值，单开两项均下降，目前两项均不推广。

## 下一项精确候选

Stage16通过模板去掉单行NF4 GEMV重复的activation prefetch，保留weight prefetch和算术。`AILA_BNB4_GEMV_INPUT_PREFETCH_OFF` 默认关闭，支持GDN CommitHistory。旧完整decode设备时间线NF4 GEMV合计7.71–7.99ms/token，GDN状态合计1.38–1.55ms/token；因此本轮先测投影路径。

Stage16 CLI SHA256 `F85BC98590C4B1387D71B2E5A8A422BFFFA8242EC4EB3B2B19FF0D0D9FD37E61`，源码与EXE归档 `tmp/day5-s16-source.json`、`tmp/day5-s16-Aila.exe`。打开输入预取移除开关后，NF4 kernel、Qwen4 kernel、GDN chunk三项CTest通过（62.70秒），包含history逐位/canary检查的GpuQ8测试通过（17.91秒）。长提示archive1833的GPU teacher48逐位一致，CPU KL .00347213 / NRMSE .0629546 / NLLdelta −.00206271，实际GDN融合1728次，重放/checkpoint/live USM检查通过。

输入预取移除长生成初筛：首请求12.7、暖第二请求17.2 tok/s；取消检查也通过。暂无重复收益，开关保持默认关闭。

容量候选单独降低load-time显存reserve至2048 MiB，专家hotset3551→3763。诊断运行暖第二请求15.8 tok/s，不推广。显存计数51个采样：adapter dedicated峰16137.53 MiB，process dedicated峰15135.02 MiB，adapter shared峰282.37 MiB，process shared峰216.51 MiB。样本有监测开销，标记diagnostic_only；不混入正式性能收益。`tmp/day5-s16-reserve2048-memory-summary.json`保存计数器汇总，原始数据保留于对应`.gpu-memory.json`。实际device advertised global memory为15933 MiB；warm显式USM ledger约13432 MiB，该差值不是DXGI可用budget。

## Stage17：CPU按专家流水候选

`AILA_Q4E_CPU_PER_EXPERT_DAG` 默认关闭，仅A64 backend。每个专家仍使用原gate16/down32 row tiles；该专家最后一个gate tile通过acq_rel倒计数获取完整BF16 h，执行原量化，再派发其down tiles；其他专家不必到达整批gate屏障。原算术不改，scratch仍在同步batch生命周期内。异常时取消并排空全部后代任务，batch之间使用隔离取消context。旧串行phase诊断非空timings时回退旧路径，避免把重叠阶段作为串行耗时汇总。正常pipeline实际进入后输出一次`CpuExpertDagEnabled actual=1`标记。

专项 `--nf4-q8-dag` 通过：1/3/10/40任务、1/4/18线程，flat/nested256/generic255、独立spec/row2/shift、BF16 h与FP32 out逐位一致、canary、并发隔离、异常排空、立即复用、其他backend/诊断回退。真实A64 shadow增加显式DAG=false，防止对照继承实验ENV。Stage17 CLI SHA256 `61EB97232596022D28E8900CC160963A5CBDC261457C24850FE5ACF49ED68BAE`；源码与EXE已归档。

archive1833 teacher48验收PASS：实际进入DAG，4271个A64 shadow批次与原单行/旧调度h/out逐位一致；GPU48/48 logits exact，CPU KL .00347213 / NRMSE .0629546 / NLLdelta −.00206271，实际GDN1728次，checkpoint/replay/cancellation通过，live USM delta0。shadow包含warmup、重放与验收forward，与teacher专家任务7826是不同口径。验收有双算开销，明确diagnostic_only。

Stage17同EXE长生成：DAG1暖第二请求16.8，两次相同；DAG0同EXE16.1，首次请求DAG1为11.7、DAG0为12.9。自然生成调度改变hotset ready admission，完整生成不是固定teacher路由实验。尚未超过Stage14已重复17.4–17.5，默认不推广。

## Stage18：会话常驻CPU tile pool

`AILA_Q4E_CPU_TILE_POOL` 默认关闭。借鉴Strata固定线程复用调度思路，自行实现同步tile执行器：load-time创建threads−1工作线程，coordinator参与；原gate/down两阶段tiles按4个tile领取；两次phase barrier仍保留，算术不改。1ms有界idle spin后睡眠，mutex与generation发布避免丢唤醒；每代全部worker ack后才释放借用的stack callback/scratch。异常保存首个错误、停止新领取、排空全部worker再抛出。与DAG互斥选择pool路径。

Stage18编译与专项测试通过：100代任务恰一次覆盖，异常drain/reuse，1/4/18线程，1/3/10/40 A64任务h/out逐位一致。真实shadow通过thread-local scope清空pool并显式DAG=false，继续与旧TBB调度单行A64对照。CLI SHA256 `F97977BCE7E0ABB2E983CCA6A0D163ABAF19AD6277A0C4BAE799A50C3578FE8C`，source/EXE归档。

archive1833 teacher48验收PASS：4271个实际A64 shadow批次逐位一致；GPU48/48 exact，CPU KL .00347213 / NRMSE .0629546 / NLLdelta −.00206271，无NLLalert；checkpoint/replay/cancel通过，live USM delta0。长生成暖第二请求18线程10.6、16线程16.1 tok/s；均未超现有配置，默认关闭。

## Stage19：异步路由轻量residency查询

`AILA_Q4E_CPU_LIGHT_ROUTE` 默认关闭。旧异步route每层对全部512专家构建ExpertResidency，复制SYCL event和host元数据，只用gpu_ready/gpu_copying两位。新 `HasGpuRouteSource` 直接读相同cache slot state；Ready与Copying的union始终true，非法/缺失/故障仍false。原选择与CPU/GPU数值保持，避免每token约24576次完整对象构建。Readiness CTest通过（7.63秒），新增缺失/非法/host-only/ready/copying行为对照。

长生成暖第二请求两次均17.6 tok/s，尚未推广。CPU pool、DAG、headQ8、host-read、input-prefetch-off均关闭。Stage19 CLI SHA256 `1F7DB913FCB1789EC3F73E2E4522FEEA1E38F483AE36FDD833CDB082247BAC1A`，源码与EXE归档。

## Stage20：A64 nested scale寄存器舍入候选

`AILA_Q4E_CPU_SCALE_REGISTER` 默认关闭。仅nested256/spec/shift准入，用显式scalar SSE mul/add替代每block volatile product栈store/load，保持两步f32舍入。原Generic和Nested256代码保留，真实shadow明确关闭spec/shift仍执行原volatile加载。

初版局部fp pragmas未阻止inline intrinsic融合，独立scalar测试在bs256/task0的F32 out失败；反汇编确认scalar FMA，失败日志与Stage20源/EXE保留。修复加opaque register屏障阻止FMA、保留非Clang volatile回退；修复反汇编为独立vmulss/vaddss、无中间stack traffic，独立A64 scalar-bitwise测试通过（1..4 tokens/40 tasks、1/4/12线程、signed/ties/zero/saturation、bs256/255、独立shift/spec/row2）。Stage20r CLI SHA256 `B7A64BA5A2AAE2D90573529E582A873A8954B8735E3A4FDBD1AA3196BF38E18D`，真实模型质量与速度待测。

真实archive48验收通过，4271个A64旧路径shadow批次逐位一致，GPU48/48 exact，CPU KL .00347213/NRMSE .0629546，checkpoint/replay/cancellation和live USM通过。同修复版EXE长生成，register1/light1暖16.6，register0/light1暖17.6 tok/s，因此register保持关闭。light1的17.6已在Stage19两次与Stage20r一次重复。

## Stage21：NF4 GEMV工作组

使用已有 `AILA_BNB4_GEMV_WG=512` 单变量改变workgroup内SG数量，仍SG16/每SG一行，保持权重布局、SLM QMAP、FMA与归约。无新增实现，沿用Stage20r EXE；数值/速度进行中。

GpuQ8专项CTest通过（11.88秒），长生成暖17.3tok/s，保留WG256。Stage22同Stage20r EXE/TBB16线程/light1暖17.6，18线程也为17.6；保留18默认，没有线程数净收益。

light1/register0短story48质量验收PASS：GPU48/48 exact，CPU KL .00148405 / 最大NRMSE .0549614 / NLLdelta −.0114601，checkpoint/replay/cancellation通过，live USM0，真实A64旧路径对照通过。长archive48在Stage20r/light1验收PASS。实验launcher `-Heterogeneous` 开启light route；独立显式参数可关闭。继承pool/DAG/register/input-prefetch/reserve环境污染的DryRun确认被固定清理，所有负收益项关闭。Stage23已有router/classify融合开关叠加light route速度测量中。

实验入口：

```powershell
pwsh -NoProfile -File perf/qwen4exp/q4e_supply_chat.ps1 -Heterogeneous -MaxTokens 128 -PromptFile perf/qwen4exp/prompts/archive-natural.json
```

该入口使用验证过的GDN历史融合与CPU18/A64/row2/async/light-route供给，Q8头、CPU直读、spin、MoE前端融合、GPU专家/attention Q8、packed/scale缓存、DAG、tile pool、register scale均关闭。此入口仍为实验配置。
