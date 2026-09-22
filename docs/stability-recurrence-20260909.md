# 2026-09-09 压测退出后的 GPU 黑屏复发

本次直接故障已确认是 Intel 显卡引擎超时后恢复失败，最终触发 `0x116 VIDEO_TDR_FAILURE`；具体触发应用、GPU 命令及最终根因尚未确认。复发后曾暂停 GPU 并将机器守卫置为 `blocked`。2026-09-09 晚间用户完成 DDU 清理、全新安装 8860 并重启后，已按授权恢复验证，结果见 [8860 验证记录](gpu-stability-8860-20260909.md)。以下时间线和阻断证据保留事故当时状态。

## 时间线与进程证据

以下均为北京时间。故障时刻来自转储内的 `.time`，不使用重启后的事件写入时间或文件修改时间代替。

| 时间 | 已核实的证据 |
| --- | --- |
| 15:18:03.819888 | 最后一档 4 并发短回归的 runner 退出，exit 0；此前服务已优雅关闭，owned Job 为空，无强制终止、遗留 PID 或清理错误 |
| 15:48:18.474066 | 最后一份崩溃前系统快照可读，尚未观察到新增故障证据 |
| 15:51:11.555 | `WATCHDOG-20260909-1551.dmp`：`0x141 VIDEO_ENGINE_TIMEOUT_DETECTED` |
| 15:51:18.763132 | 该 LiveKernelEvent 的 WER 报告创建；附带进程清单包含 226 个进程，没有 Aila、AilaWorker、AilaServer 或 Python，也没有最后一轮的 Server PID 8936 / Worker PID 7640 |
| 15:51:21.190 | `090926-10734-01.dmp`：`0x116 VIDEO_TDR_FAILURE` |
| 16:09:35.500 | 后续系统启动；Kernel-Power 41、System 1001 和 WER 补报在此后写入 |

引擎超时到 bugcheck 相隔 **9.635 秒**，两份转储的 TDR context 指针相同。最后一次 runner 退出到 bugcheck 相隔 **33 分 17.370112 秒**。本轮没有在该 runner 之后启动新的 GPU 工作。

进程清单来自**崩溃前的 LiveKernelEvent 附件**；同次 BlueScreen 报告附带的另一份 174 进程清单是在重启后采集，不能拿它证明崩溃时没有 Aila。前者的 WER metadata 创建时间为 `07:51:18Z`，所列进程的最晚创建时间为 15:51:11.592337，早于 bugcheck。CSV 没有独立的采样时刻字段，所以不能把报告创建时间当成精确的进程采样瞬间。

这些记录共同支持用户“当时没有在跑压测”的判断。它们不排除早先 GPU 负载对驱动状态产生延迟影响，也不构成系统全程无 GPU 活动的证明。清单包含桌面合成、Edge/WebView、GameViewer 等进程；同时存在不能证明其中任何一个就是触发者。

## 已定位的故障路径与边界

两份转储均指向 `igdkmdnd64.sys`，版本 **32.0.101.8974**：

| 转储 | Bucket / 状态 |
| --- | --- |
| LiveKernelEvent | `LKD_0x141_IMAGE_igdkmdnd64.sys`，责任模块指针 `+0x12b70` |
| 系统 bugcheck | `0x116_IMAGE_igdkmdnd64.sys`，责任模块指针 `+0x11cc0`，参数 3 为 `0xc0000001 STATUS_UNSUCCESSFUL` |

系统栈经过 `dxgkrnl!TdrBugcheckOnTimeout`、`DXGADAPTER::PrepareToReset`、`DXGADAPTER::Reset`、`TdrResetFromTimeout`。这与 [9 月 8 日三次 0x116](stability-incident-20260908.md) 的故障路径一致。loaded-module 的 timestamp/checksum/SizeOfImage 与本机已核验驱动一致，驱动文件及 CDB 文件哈希也仍匹配此前记录；没有获得整份驻留驱动镜像的哈希。

`PROCESS_NAME: System` 是执行恢复/采集的系统线程上下文。LiveKernelEvent 提供的 `PROCESS_OBJECT` 所在页面没有收进小型转储；`!process` 和使用公开类型读取 PID/ImageFileName 均失败，因此不能识别该对象的应用名称。现有证据不能在驱动自身缺陷、先前负载留下的异常、其他图形负载/显示组件交互或硬件问题之间作最终区分。

15:15–16:15 的 System 事件查询未发现 WHEA 或 Display 4101；这不排除硬件或驱动故障。Kernel-Power 41 已携带 bugcheck 278（`0x116`），不能仅凭事件 41 认定为电源断电。重启后另有 volmgr 161 和 162；小型转储可读且 CDB 报告完成，但没有完整 `MEMORY.DMP`。当前配置是 Small dump（`CrashDumpEnabled=3`），不能把缺失页面当作所有图形状态均已检查。

Security 进程创建/结束审计未启用，无 4688/4689 历史记录。DxgKrnl 已启用的日志在本时段只记录了 12:02 的虚拟 GPU 创建/打开成功，没有提供超时对应的用户态提交链。EventLog 6008 中的 12:00:28 与转储时刻不一致，未用于故障定时。重启后的多条 WER 记录含旧事故补报，不能计为多个新崩溃。

## 证据与验证状态

本次证据位于 `tmp/refactor/stability-recurrence-20260909T081320Z`：

- `incident-analysis.json`：离线汇总、时间差、CDB 行号、文件身份和判断边界；生成脚本为 `analyze-recurrence.py`。
- `dumps/`、`wer/`：新 Minidump 与 LiveKernelEvent/BlueScreen 附件。Live dump 已被 WER 移到 ReportQueue，原 LiveKernelReports 路径不存在；从队列取得的副本也已校验。共 16 个保存的文件副本核验一致，其中 Minidump 在两个来源中相同。
- 三份 `*.cdb.log`：两份主分析及无法解析关联进程对象的补充分析；使用现有 CDB 与本地符号缓存。
- `events-1515-1615.json`、`dxgkrnl-events.json`、`wer-process-snapshots.json`：事件 XML 与进程清单摘要；原始 WER CSV 一并保留。
- `guard-after-block.json`：新故障和 boot change 已持久阻断；evidence ID 为 `515702a480bc25e33f345e42927aea503e1436b829ed5674e44b76918d90c04d`，没有 acknowledge。
- `pre-crash-snapshot-verification.json`：修改本次事故文档之前，崩溃前快照引用的源码、源码副本、产物、运行库、模型和报告等 **437 项全部重新校验通过**。发生暂停/阻断后变化的 live marker/guard 使用事先保存的原样副本核对。

新系统 Minidump 的 SHA-256 为 `a2fb4f2a8dfa048f0d415ad2e346a47015dd6e6e0b5bff15c2270889ec44c792`；Live dump 为 `c1f6492f11bad87c433cdf704c9caa41b59a47a7c51dd2751f9c3d906856fa67`。原 `gpu-resume-20260909T063259Z/resume-validation.json` 保持原样，SHA-256 为 `842683766d339debd00d6a131ca7b962bcf9f74ca070de1af3fda05039c3cf97`。

此前五个短运行的有限正确性与进程收尾结果保留，但**系统稳定性未通过**，P6 长稳不能验收，P7 GPU 验证继续暂停。守卫是运行前后检查，无法保证进程退出后的桌面使用不发生驱动故障；扩大应用请求期限也不能修复内核 TDR。

## 下一步调查路径

先建立不启动 Aila 的桌面观察基线，保留应用/显示环境和时间线，再安排单变量对照。现有小型转储缺少关联进程页面，后续需评估更完整的内核/图形诊断采集。驱动版本、远程显示及浏览器图形负载应分别比较，避免同时变更多项后失去归因依据；此前资料中的 8826 仅为有来源的驱动对照候选，尚未验证本机稳定性。

复发后的初始阶段仅开展 CPU 取证与源码审查。晚间已以本次复发和 DDU 重装后的新环境为起点完成审阅并恢复有限 GPU 压力测试；详细参数、结果和未覆盖范围见 [8860 验证记录](gpu-stability-8860-20260909.md)，没有将原 8974 短回归直接视为系统稳定性通过。
