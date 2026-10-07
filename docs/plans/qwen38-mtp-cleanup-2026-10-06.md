# MTP 实验收敛与清理验收

2026-10-06。本轮只做已授权的实验清理、当前源码构建/复测和 Server 接入方案，没有新增 kernel 或参数性能筛选。

## 保留 / 删除 / 诊断

| 分类 | 处理 |
|---|---|
| 保留有效配置 | T2 CPU async、GPU expert 4/2/1 reuse、NF4 projection reuse、BF16 dual-query attention、batched GPU greedy、删除无用 draft tail copy、state-only replay。prepare/hidden-reuse 也沿用已验证配置。 |
| 保留 bundle 初始化并迁移所有权 | Qwen4ExpBnb4Backend 新增 PrepareDecodeVerification(Context&)，校验模型 Context，缓存一个 executable bundle；后端析构先 synchronize 再释放成员。probe 仅在原 requestStart 之后调用该方法并计时，不再持有 optional 或向后端借临时指针。 |
| 删除失效采集链 | MtpTimelinePosition / AILA_Q4E_MTP_TIMELINE_POSITION、LIVE timeline 容器和 profiling 输出、executor VerifyTimeline 及各事件 hook、MoE gate-up 唯一用于该链的 first_kernel 回调。该诊断两次因 internal event wait 错误中止，未获得有效时间线。 |
| 保留数值依据 | shadow、T1 reference、GDN reference、finite guard、状态恢复/拒绝重放、局部正确性检查。Ops.hpp 的标准 E4M3FN 修复和 Qwen4ExpKvFp8Tests.hpp 独立保留。 |
| 已撤回候选核对 | pending residual rows、draft128、prefix-state、batch-checkpoint、Q4E FP8 KV/向量读取、后台 bundle 线程：本轮核对 src/tests/perf/CMake 无对应死入口。旧报告与 tmp 负结果档案保留。 |
| 其它共享工作区修改 | 不按默认关闭或 test/experimental 名称批量删除。CMake 的已有 Q8/模型测试入口仍有实际调用，本轮没有删除这些公共验证资产。 |

清理前相关文件快照在 tmp/mtp-cleanup-before/。本轮未改变 kernel 数学、热集容量或请求采样参数；删除的 inactive timeline 不再使公共 gate-up 接口携带无消费者的事件输出参数。

## 当前二进制复测

构建目标 Aila、AilaWorker、AilaQwen4ExpToleranceTests、AilaQwen4ExpKernelTests 全部成功（tmp/mtp-cleanup-build.log，exit=0）。聚焦检查全部通过：

- --attention-pair-only：5 个因果长度的两行 gated BF16 输出逐位匹配原 B1，partial/output canary 与 USM 检查通过。
- --mtp-greedy-only：B1/B2/B4、极值、尾部、跨 WG ties、signed zero、逐行 NaN/Inf、canary 通过。
- --kv-fp8-only：标准值表、中点、全部 65536 种 BF16 位模式的 CPU/GPU 编解码，以及 append/cache/reference 检查通过。

没有重跑所有失败候选或全 CTest。diff whitespace 检查按原有 CRLF 行尾通过。

原 1833-token CLI system prompt / 128 outputs、A770 32.0.101.8724、oneAPI 2026.1，保持已验证配置：

| 记录 | 清理前 D | 清理后 |
|---|---:|---:|
| 完整 decode tok/s | 20.556847 | **20.592890** |
| 完整循环 ms | 6226.6359 | **6215.7375** |
| 首窗口 ms（包含在循环内） | 269.4538 | **268.9396** |
| 首次 bundle 准备 ms | 0.1428 | **275.7270** |
| prompt→128 outputs ms | 17531.1980 | **17770.0362** |
| T1 token mismatch | 0 | 0 |
| LIVE USM delta | 0 | 0 |

清理后 EXE SHA256 为 2A10BC20735ABE6DFD1A326E3ACD6421DD211C1B13C9CFD6ABAAA39F45795674，exit=0。生成和 advance 全数组与 D 完全相同，hotset 首尾均为 7625625710176325289。旧二进制的成绩没有用来冒充清理后成绩；两次构建不同，不宣称 -10.8984 ms 是严格同 EXE 的性能收益。

**请求总时间实际增加了 238.8382 ms。** 新记录的首次 bundle 准备为 275.727 ms，已包含在 target_prompt_ms=10413.3645 和总请求中，未隐藏或删掉。bundle 现在缓存于模型后端，不随 Request/session reset 重建；这项首次准备成本在 Server 模型初始化/首请求验收中必须单列。此次没有为解释跨构建的程序准备/缓存变化再开性能 campaign，也不将它当成已经彻底定位的编译问题。

清理后完整循环仍过 20，本样本 128/128 与 T1 相同；没有扩展为普遍无损 MTP 或 Server 吞吐保证。

原始记录 tmp/mtp-8724-cleanup128.*，机器可读数据和完整 token/advance 见同名 JSON，由 tmp/mtp-cleanup-report.py 生成。复现继续使用 qwen38-mtp-kv-pair-8724-2026-10-06.md 的命令，只改 Tag。

## Server 下一步

具体代码修改清单、内存公式、窗口提交语义与分阶段验收在 qwen38-server-minimal-integration-2026-10-06.md。本轮没有修改 HTTP/IPC 的能力声明，没有让未接通的后端被提前广告为可用。

实施顺序：普通 Q4E shared-model T1 → 最小 MTP 运行时与资格检查 → Request 窗口提交/取消恢复 → 同配置真实 JSON/SSE 验收。draft KV 按块初始化和首轮 hotset handoff 是接入后的 TTFT backlog，不继续重试已失败性能候选。
