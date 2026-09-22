# 2026-09-08 GPU 压测系统崩溃调查

状态（2026-09-09 晚间更新）：用户 DDU 清理并全新安装 8860 / Level Zero 1.28.2 后，已恢复有限 GPU 验证，六轮结果见 [8860 验证记录](gpu-stability-8860-20260909.md)。9 月 8 日四份 Minidump 的离线结论保留；9 月 9 日在最后一次有界压测退出约 33 分钟后又发生 `0x141 → 0x116`，仍涉及 `igdkmdnd64.sys 32.0.101.8974`。崩溃前 WER 进程清单没有 Aila 或 Python 压测进程，具体触发点及最终根因仍未知。P6 跨模型长稳和 P7 设备验收尚未完成。事故证据见 [压测退出后的复发调查](stability-recurrence-20260909.md)。

## 已确认的事实

9 月 8 日用户报告多次突然黑屏、自动重启，当时反馈只在 Aila 压测中遇到。9 月 9 日的新复发发生在压测进程退出之后，崩溃前 Windows 进程快照支持该判断，因此已知场景已扩大。Aila 提交、资源生命周期、驱动/runtime 状态和日常图形负载的交互仍需调查；不能单凭故障模块名将根因认定为独立驱动缺陷。

以下实际时间来自调试器初始输出和显式 `.time`，两处一致。所有时间均为 2026-09-08 北京时间：

| 实际转储时间 | 系统 uptime | 后续 WER 1001 报告时间 | Bugcheck | 转储 |
| --- | --- | --- | --- | --- |
| 14:18:28.859 | 3:05:03.710 | 16:36:20.548 | `0x116 VIDEO_TDR_FAILURE` | `090826-6390-01.dmp` |
| 17:00:21.517 | 0:24:07.370 | 17:38:06.368 | `0x116 VIDEO_TDR_FAILURE` | `090826-6203-01.dmp` |
| 17:45:01.931 | 0:07:01.789 | 17:46:08.153 | `0x139 KERNEL_SECURITY_CHECK_FAILURE`，参数 1 为 2 | `090826-5984-01.dmp` |
| 18:06:05.926 | 0:10:49.778 | 18:06:57.266 | `0x116 VIDEO_TDR_FAILURE` | `090826-6109-01.dmp` |

前两次 WER 报告分别晚约 2 小时 17 分、37 分钟。WER 时间、转储文件写入时间和测试日志修改时间均不能替代实际崩溃时间。

`0x116` 表示 GPU 超时后的恢复失败；`0x139/2` 表示内核检测到 stack-cookie 检查失败。两者均为系统级故障，不能把测试进程、工具句柄消失解释为普通任务中断并直接重试。`Kernel-Power 41` 和 `6008` 本身不提供根因，也不证明电源故障。

设备为 Intel Arc A770，驱动 `32.0.101.8974`（驱动日期 2026-08-11，Intel 公开发布日期 2026-08-14）；oneAPI compiler 2026.1.0，oneDNN 3.11.2。采集系统版本为 `10.0.26200`，转储中的 kernel/dxgkrnl/dxgmms2 文件版本为 `10.0.26100.7920`，不能仅凭这一差异判断为 Insider 或不支持的组合。系统同时枚举到 GameViewer Virtual Display Adapter；没有证据将故障归因于它。注册表未显示自定义 TdrDelay/TdrDdiDelay/TdrLevel，本次没有修改这些设置、驱动或系统自动重启设置。

原始事件 XML 和元数据保存于 `tmp/refactor/stability-20260908/system-events.json`。最初普通权限读取受保护的 `.dmp` 和 LiveKernelReports 被 Windows 拒绝，历史 `dumps.json` 中空 SHA256 保留这一失败。用户随后在管理员 PowerShell 执行 `tools/diagnostics/collect-system-crash.ps1`，成功取得四份副本，总计 **37,545,589 字节**；文件大小、SHA-256、`PAGEDU64` 头部与采集清单及事件参数全部匹配。副本位于 `tmp/refactor/stability-20260908/admin-collection/Minidump`，没有覆盖原始转储。

## 转储分析结果与边界

使用从 [Microsoft 官方包](https://windbg.download.prss.microsoft.com/dbazure/prod/1-2606-22001-0/windbg.msixbundle) 解出的 WinDbg `1.2606.22001.0` / CDB `10.0.29617.1000` 离线读取转储；只解包到项目临时目录，没有安装调试器或上传转储。`cdb.exe` 的 Microsoft 签名已核验，Windows 符号来自 Microsoft symbol server。四份主要分析及两份补充分析保存在 `tmp/refactor/stability-20260908/windbg-analysis`。

| 转储组 | 确认的故障证据 | 能支持的判断 |
| --- | --- | --- |
| 三份 `0x116` | `FAILURE_BUCKET_ID: 0x116_IMAGE_igdkmdnd64.sys`；参数 3 为 `0xffffffffc0000001`（`STATUS_UNSUCCESSFUL`） | GPU 超时恢复失败，责任驱动指针位于 Intel 显示内核模块 |
| 一份 `0x139/2` | `FAILURE_BUCKET_ID: 0x139_MISSING_GSFRAME_igdkmdnd64!unknown_function`；异常 `0xc0000409`、子码 `FAST_FAIL_STACK_COOKIE_CHECK_FAILURE` | 驱动内栈 cookie 校验失败并触发 fail-fast；不是 LIST_ENTRY 校验失败 |

三份 TDR 的系统栈一致经过 `dxgkrnl!TdrBugcheckOnTimeout`、`DXGADAPTER::PrepareToReset`、`DXGADAPTER::Reset`、`TdrResetFromTimeout` 和 `TdrResetFromTimeoutWorkItem`。参数 2 对应 `igdkmdnd64+0x11cc0`，这是责任模块指针，**不是已定位的故障 GPU kernel 或 Aila 调用指令**。

`0x139` 的异常点为 `igdkmdnd64+0x4812a5`；补充反汇编可见前一条 `mov ecx,2` 和该处 `int 29h`。返回地址 `+0x46ff36` 前有读取、异或和调用 stack-cookie 检查的指令。再往后的栈包含无效/全 1 地址，不能当作可靠的完整调用链。Intel 私有符号不可用，四份转储的 `PROCESS_NAME` 均为 `System`；它们没有恢复 Aila 用户态提交序列，也不能对应到唯一的用户进程 PID。

四份 loaded-module 的 timestamp `6A7AB478`、checksum `01DE870B`、SizeOfImage `01DD9000` 均与本机 Intel 驱动文件一致。本机文件版本为 `32.0.101.8974`，WHQL 签名有效，SHA-256 为 `0602d6003d0c97e6812ca2d96235fc8b0fb4afde89664bbf8c2a7942c6598658`。个别初次 CDB 输出未显示版本资源，版本通过匹配的本机 PE 和补充 `lmv` 解析核对；这不等于取得了整份驻留驱动内存的哈希。

现有符号无法解析 `_TDR_RECOVERY_CONTEXT`，因此没有猜测其中的字段。`!blackboxpnp` 中的 Wintun 条目没有证据与本次故障相关。转储不足以区分具体算子、提交积压、释放次序、runtime 或驱动内部错误；上述异常处理与生命周期修复是否消除触发条件，仍未经过 GPU 验证。

综合报告 `tmp/refactor/stability-20260908/dump-analysis.json` 将转储副本哈希、CDB 行号/栈、驱动身份和测试时间关联在一起。生成脚本 `consolidate-dump-analysis.py` 仅重读已有证据，四份再次核验通过；可用 `--output <新文件名>` 复核，脚本拒绝覆盖历史报告。`implementation-validation.json` 是管理员采集前的编译/CPU 验证快照，其中的当时状态不覆盖本次新结论。

## 与现有测试证据的对应

以下两次 Qwen3 HTTP 测试均没有完整 `result.json`，不得计为长稳通过：

| Run | 开始时间 | 最后可解析的请求开始时间 | 最后完整生命周期波次 | 实际崩溃 |
| --- | --- | --- | --- | --- |
| `run-20260908T094056-c58090d1ac0c` | 17:40:56.911 | 17:44:58.397 | 0–121，共 488 个生命周期请求 | 17:45:01.931，`0x139` |
| `run-20260908T100349-04fefc3e3578` | 18:03:49.855 | 18:05:51.996 | 0–23，共 96 个生命周期请求 | 18:06:05.926，`0x116` |

目录在 `tmp/refactor/p6-service-q3`。两次都是 Qwen3 NF4 0.6B、context 2048、四个 Session、短/长 prompt 实际 27/936 token；每波两条完成、两条在收到首个内容事件后断连。故障前各 140 个正式 benchmark 响应可用原始 SSE 对照独立参考，但这只支持这些响应的正确性，不支持服务稳定性或完整 512 请求门禁通过。

已保存的 lifecycle baseline 到最后完整波次的 idle 样本，没有显示显著持续 private bytes 或 handles 增长：Worker 分别约 +0.176/+2.598 MiB，handles −2/+2。这些是进程统计，不包含实时 WDDM/GPU 内存，不能据此排除驱动资源问题或认定超显存。

第二次末尾有 22 个 health 样本，约 11 秒内仍有两条 active/outstanding 请求，completed/cancelled 计数不变，HTTP health 仍为 `ok`。说明 HTTP 能响应不等于推理有进展；现有记录不足以指出最后执行的 GPU kernel。尾部部分 JSON/SSE 为空或全 NUL。分析脚本只读取原始文件，报告和哈希清单见 `tmp/refactor/stability-20260908/trace-analysis.json` / `.md`。

14:18 的候选记录是 `build-refactor/p6-ipc-q35-08b-4096-soak.log`：Hybrid NF4 0.8B、context 4096、四 Session，direct 20 轮与 Worker long-context 通过，Worker soak 第 3 轮尾部中断且缺少 suite PASS。其修改时间距转储约 17 秒，但缺少绝对执行区间和 PID，关联尚未确认。日志在整波 `Drain(requests)` 完成后才逐 Session 打印 `MEASURE`，不能将尾部 `request=2` 解释为 request 3 的 GPU kernel 卡住。

17:00 的具体触发测试尚无法确定。附近 Hybrid 4B long-context/20 轮 soak 和 Qwen3 long-context 文件都保存了完整 PASS / exit 0，却没有绝对执行区间；不能仅凭修改时间归因，也不能推断卸载后发生了延迟 TDR。详细关联、缺失信息及文件哈希见 `dump-timeline-correlation.json` / `.md`。先前短用例和数值结果继续保留，但不能覆盖本次长稳失败；四次崩溃的精确 GPU 提交序列都尚未取得。

## 官方驱动资料核对

[8974 官方发行说明](https://downloadmirror.intel.com/926177/ReleaseNotes_101.8974__WHQL.pdf) 支持 A770 和 Windows 11 24H2/25H2，列有 Procyon AI / Geekbench AI 的应用崩溃，但没有直接对应本次 `0x116`、`0x139/2` 或上述驱动偏移的已知问题/修复声明。不能把这些应用崩溃条目等同于整机 bugcheck。

[oneAPI 2026.1 要求表](https://www.intel.com/content/www/us/en/developer/articles/release-notes/oneapi-toolkit/2026.html) 列出 Arc Windows 驱动 `32.0.101.8826`，[历史驱动页](https://www.intel.com/content/www/us/en/download/857252/historical-intel-arc-graphics-drivers.html) 也确认其 WHQL 版本。因此 8826 是未来固定其他变量时有依据的对照候选，尚未核验安装包或验证本机稳定性；该表不表示只有 8826 兼容，更不证明 8974 就是根因。

本轮提取的 A770 产品下载目录显示 8992 / 2026-09-02，链接指向的主下载页却仍返回 8974 / 2026-08-14；已记录这一差异，不能将 8992 称为已确认修复。8974 说明中的回退 8864 建议针对 Core Ultra NPU，不适用于本次 A770 调查。官方来源和提取证据保存在 `intel-driver-research.md` / `.json`。本轮没有改驱动、切换 SYCL backend 或改变系统超时。

## 当前代码的取消与释放路径复核

本轮由独立子代理只读检查当前工作区，主代理复核关键调用点。有限范围内，未发现正常 HTTP 断连直接导致提前释放 USM、跨线程同时修改 Context 或重用失效 scratch 的确定缺陷。该结论是当前源码的静态审查，不替代当时崩溃二进制的执行证据，也不表示资源/runtime 路径已被排除。

| 路径 | 当前屏障与所有权 | 源码依据 |
| --- | --- | --- |
| HTTP 断连 | 请求线程调用 Request/Session Release，客户端安排取消与私有排空；socket 回调不直接析构 GPU 资源 | `src/server/HttpServer.cpp:728`、`src/runtime/InferenceClient.cpp:413` |
| 执行中的取消 | Cancel 置标记，Forward 在 `device->mutex` 内同步完成后返回；Step 再处理取消并释放 Session lease | `src/inference/Model.cpp` 的 `ModelSessionBackend::Forward`，`src/inference/Request.cpp:148`、`:190`、`:199` |
| Worker 回收 | Scheduler 先清执行引用再发布终态；Worker 随后清 `executingSession`。Session 只有 registry 持有最后引用才交 lifecycle thread，Model 还需 `sessions == 0` | `src/inference/Scheduler.cpp:138`，`src/worker/InferenceWorker.cpp:821`、`:915`、`:932` |
| 正常退出 | 先等待 control thread 与 Scheduler，再在 lifecycle thread 释放 Request/Session/Model/factory | `src/worker/InferenceWorker.cpp:1042`、`:1056` |
| 设备资源 | Session/Model 析构同样持执行锁并先排空；已检查的 scratch 增长与 primitive cache 淘汰前有同步 | `src/inference/Model.cpp` 的 `LoadedModelState` / `ModelSessionBackend` 析构；`src/core/Context.hpp` 的 `scratchpad_memory` / `make_primitive_cache_room` |

HTTP 每个请求都会创建并释放 Session；正常断连走 Cancel/Release，**不必经 `SessionState::Reset`**。Reset 属于直接 API 或资源上限错误后的恢复路径，后续诊断应单独覆盖。底层 Tensor 析构自身不等待，仍依赖上述拥有者的排空约定；已查看的 Forward 局部 Tensor 是非拥有 view。当前证据无法区分故障发生在 Forward 的提交/等待中，还是 Session USM 回收阶段，下一步先记录这两个边界。

## 已实施的可靠性修复

1. `Context` 使用显式 SYCL async handler 保存首个 `exception_ptr`。本机 SDK 的默认 handler 会 `std::terminate()`，无法可靠进入上层 RuntimeFault。现在同步边界始终先尝试排空，再重抛首因；已知故障后的提交/分配被阻止，资源释放仍允许执行。blocking copy 使用 `wait_and_throw()` 并检查已保存异常。
2. NF4 融合权重加载在函数内部捕获异常并排空。调用方的局部源投影在退出前继续存活，消除后续 metadata/初始化异常时先析构 USM、外层才 drain 的窗口。正常成功路径和算子数学保持。
3. HTTP benchmark 接入机器固定目录的进程锁及持久运行记录。已有崩溃、未收尾运行、新系统故障或证据不可读时拒绝启动；没有自动清除或重试。实际只读 preflight 已确认阻断当前机器，未启动 Worker。
4. 默认 runner 仅提供两条短串行请求的 `probe` 配置（context 512、生成 8 token、60 秒总期限）；长压测需显式 `--allow-stress`，仍不能绕过故障隔离。增加正数的 device/pinned/session-host 分配上限、独立整轮 watchdog、原子 JSON 与 fsync 增量记录、提交前 active-batch 记录。
5. runner 的异常收尾先保证尝试清理 owned Job，再关闭日志、汇总与持久写入结果，最后提交 guard 成功。即使结果和 guard 日志同时写入失败，也不会把该运行持久标为成功；锁由最外层 `finally` 释放。结果文件的 `passed` 必须结合零退出码、相同 run id 的 guard `finished` 和结果哈希验收，不能单独使用。成功提交后的锁释放错误单独报告为 `post_commit_errors`，不依赖可能再次失败的文件回滚。
6. 2026-09-09 增加默认关闭的执行阶段持久记录和离线解析，短 probe 可选择 `--execution-trace`。写盘失败阻止后续工作，但继续已有清理。最终关闭行已写入而 flush 失败时，进程级失败标记仍存活；Worker 在运行时完全清理后非零退出，防止文件看似完整却误计通过。细节见 [执行阶段记录](execution-tracing.md)。

上述修复分别解决已证实的异常交付、生命周期及测试止损/证据问题，**不构成 `0x116/0x139` 根因已经消除的证明**。Windows 用户态 watchdog 与 Job 无法保证拦住内核驱动导致的系统崩溃；文件持久化也不能保证硬件忽略 flush 时的数据安全。机器锁只覆盖使用同一守卫的工具，不约束手工启动的其他 GPU 进程。

## 验证与恢复条件

CPU `AilaRuntimeFaultTests` 已通过 9 组，包括并发错误记录、首因保持、已知故障仍排空、拒绝新工作，以及向 RuntimeFault 传播。Python guard/runner 共 72 项 CPU 测试通过，覆盖真实 Python 子进程锁/Job、异常退出后阻断、超时止损、日志失败、输入边界、故障隔离先于进程启动，以及结果/guard 同时写盘失败时保留未完成状态。它们没有执行 GPU，不证明整机已恢复稳定。

| 检查 | 结果 | 日志 |
| --- | --- | --- |
| Release Worker/CLI 与全部已配置目标编译 | 75 个构建步骤成功 | `build-refactor/stability-worker-build.log` |
| 普通 MSVC Server 编译 | 41 个构建步骤成功；32 个 C++ object 的 header 依赖有效 | `build-refactor-server/stability-server-build.log`；旧 `build-server.txt` 已按原哈希归档到 `tmp/refactor/stability-20260909/previous-validation-logs/build-refactor-server/logs/` |
| RuntimeFault CPU 用例 | 9/9 | `build-refactor/stability-cpu-fault-tests.log` |
| Request/Scheduler/预算/IPC/Worker/Client/pipe CPU CTest | 7/7，12.53 秒 | `build-refactor/stability-inference-cpu-tests.log` |
| Server Channel/Chat/Client/pipe CPU CTest | 4/4，11.38 秒 | `build-refactor-server/stability-server-cpu-tests.log` |
| Python guard/runner CPU 单测 | 72/72 | `build-refactor-server/stability-finalization-cpu-tests.log` |

所有测试都使用 CPU 状态机、注入错误或 fake Worker；没有运行真实推理、GPU kernel 或压力用例。编译身份仍为 `bbd007aaf7a2b8edc55d383659196dea8edd77b3`，本次修复与先前 P6 源码尚未提交，不能只用 HEAD 重建本次二进制；实际源码/二进制/日志哈希保存在 `tmp/refactor/stability-20260908/implementation-validation.json`。此前的 63 项 Python 通过记录和初次 fixture 失败日志均保留，不覆盖历史证据。

2026-09-09 的阶段记录补充验证另存：Release / MSVC Server 重建成功，根 CPU CTest 9/9、普通宿主 4/4、Python 118 项通过。CPU 交叉验证能够拒绝中断日志，以及文件完整但最终 flush 失败的退出；后者的 Worker→客户端关闭失败传播也由 fake pipe 验证。证据位于 `tmp/refactor/stability-20260909/implementation-validation.json`，完整边界见 [执行阶段记录](execution-tracing.md)。本轮构建和测试不替代真实 GPU 验证。

管理员采集已完成，无需重复先前命令。采集成功不等于清除了机器守卫，也不代表普通权限现在能读取受保护的 LiveKernelReports。此次只读检查持久状态仍为 `blocked`，证据 ID 为 `848bc3209097ac64d78d13fc344efaee461aa58863373e420c129dfe06e03fe7`；没有调用 acknowledgement，也没有为取证启动 GPU。

## 原定定位顺序（恢复结果见下节）

取消与析构链的本轮静态复核已完成。2026-09-09 已在源码中接入 Forward、已有排空、阻塞复制、Session/Model 析构和 USM 释放的持久阶段记录，以及离线解析和 runner 验收。开关默认关闭，启用时记录本轮 Worker PID、线程、诊断 Session ID、实际 token 数与位置，保持现有 GPU 同步点。语义与验证边界见 [执行阶段记录](execution-tracing.md)。尚未细化到每层/算子，也尚未运行真实 GPU 诊断；当前结论仍是根因未知。

后续实验按以下次序逐项审查和执行，每步单独保存结果，不自动递增负载：

| 步骤 | 固定范围与待回答问题 | 当前准备状态 |
| --- | --- | --- |
| 1. 短串行请求 | Qwen3 NF4 0.6B，context 512，1 Session，greedy，最多 8 generated tokens；独立参考 + 相同请求共两次，整轮含启动上限 60 秒；device 4 GiB / pinned host 16 MiB / session host 256 MiB 上限 | `service_benchmark.py --mode probe --execution-trace` 可承载；守卫仍阻断，尚未执行 |
| 2. 单次取消与复用 | 沿用短输入/形状，一次首内容事件后断连，确认取消终态与 GPU 排空，再创建 Session 完成一次参考请求；区分取消/释放与纯完成路径 | 需要独立小用例；不能调用现有至少 257 请求的 lifecycle 模式来代替 |
| 3. 生命周期分离 | 分开检查同 Session reset、Session 销毁重建、Worker 正常卸载；每项仅一次转换，固定相同模型和请求 | 待实现有边界日志的小用例，不循环 soak |
| 4. 隔离变量 | 只有前面得到足够证据后，才比较基线/候选源码，或固定源码切换有官方依据的驱动；一次只改变一个变量 | 基线 worktree 保留；8826 仅是版本候选，没有下载/安装或宣称修复 |

恢复任何 GPU 小用例前，应先复核新系统事件、转储可读性、对应源码/二进制/模型和环境身份，并对现存故障证据作显式审阅记录。已有失败证据不能通过换输出目录、换进程、重启系统或宽松快照清除。发生超时、device lost、非零退出、新系统错误或记录不完整即停止本轮，不自动重试。用户态 watchdog 只能限制进程运行时间，不能保证避免再次系统崩溃。

短请求成功也不能验收 P6。未完成低压力分步定位前，不恢复 1/2/4 并发、长上下文、512 次生命周期或 P7 GPU 测试。驱动对照与任何显示系统变更都属于后续独立步骤，本次没有执行。

## 2026-09-09 有界恢复结果

用户明确授权恢复 GPU 测试，并从管理员 PowerShell 启动 Codex CLI。当前进程已实际确认管理员权限，系统事件、5 份 Minidump 与 164 份 LiveKernelReports 的元数据均可读取。新可见的 LiveKernelReports 文件修改时间范围为 2025-10-11 至 2026-04-30 UTC，早于本次事故；新增 256 条 WER 报告的转储引用均已出现于原快照，System 故障记录未增加。这是证据可见性恢复，不代表新增了 164 次故障，也没有重新分析这些旧转储内容。

在保持驱动、TDR、内核和 C++ 二进制不变的前提下，89 项源码/产物/Qwen3 模型 SHA-256 匹配初始验证快照。显式审阅并记录守卫 evidence ID `1e830ddcf71eb6b9173aed39aca0b81e52120feb7b7cfb6a1f967e1b6c04c938` 后才启动 GPU。所有原始记录另存 `tmp/refactor/gpu-resume-20260909T063259Z`，9 月 8 日转储和 9 月 9 日初始 CPU 快照保持原样。

第一轮探针失败：应用请求期限设为 15 秒，而首个 27-token prefill 加采样的 trace 耗时为 15.754 秒；服务明确报告 `request_timeout`，客户端收到没有成功终态的 SSE EOF。8,744 条 trace 完整且无错误 scope，Server/Worker 零退出，Job 与请求资源排空，系统快照没有变化。失败以 evidence ID `3294aa4b7fd989e9d2fe300e2405d421091c87b9346bb9de4a08bf71902564a8` 保留。长耗时主要落在未细分的后端主机段，不能仅凭这份 trace 归因于具体 GPU kernel、JIT 或日志写盘。

完成这次超时的明确审阅后，将应用请求期限修正为 45 秒，整轮 watchdog 仍为 60 秒、清理期限仍为 15 秒，模型/输入/context/生成量不变。复测两条 8-token 请求成功且输出一致，trace 共 9,250 条。随后实现并通过 CPU 验证的 `--probe-cancel` 用例执行三条请求：参考、首内容事件后断连、创建新 Session 并对照参考。服务确认一次断连取消、两次完成，9,644 条 trace 完整；取消 Session 的等待和释放先于后续 Session 创建。上述日志不能单独证明取消发生前所有内核的具体顺序，记录边界仍以 [执行跟踪约定](execution-tracing.md) 为准。

在短串行与取消用例通过后，分别审阅并执行默认无 trace 路径的 1/2/4 并发短回归，共 4/8/16 条请求全部完成并匹配各自独立参考。每档仅一个 profile、两轮 warmup 和一轮测量，context 512、生成上限 8、整轮上限 60 秒，device/pinned/session-host 额度保持 4 GiB/16 MiB/256 MiB。每个 GPU 进程结束并核对结果后才进入下一档。五个通过的运行均为零退出、对应 guard `finished` 与结果哈希匹配，HTTP/IPC/Job 全部排空，事后快照仍可读且无新增故障。

本轮只新增 Python 取消诊断入口与 7 项 CPU 检查，guard/runner 88 项、解析器 37 项通过；C++ 推理实现没有再改动。没有运行长上下文、512 次生命周期、Hybrid/BF16 恢复或 P7 设备测试；也没有修改驱动或显示系统设置。当前结论是这组有限负载通过，原始整机崩溃的具体触发点和根因仍未确认。

## 2026-09-09 压测结束后复发

上述最后一次 runner 于 15:18:03.819888 退出。15:51:11.555 出现新的 `0x141`，15:51:21.190 触发 `0x116`；两份转储对应相同 TDR context，均指向相同 Intel 驱动版本。崩溃前 LiveKernelEvent 附带的 226 进程清单没有 Aila 或 Python。机器守卫已重新阻断，GPU 暂停标记已恢复，新的故障证据没有 acknowledge。完整时间线、转储哈希、取证边界和后续路径见 [复发调查](stability-recurrence-20260909.md)。文中各阶段的通过仅限当时用例，不表示本机已恢复稳定。
