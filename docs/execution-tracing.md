# 执行阶段持久记录

2026-09-09：为 9 月 8 日系统崩溃调查增加诊断记录。用户授权恢复后，Qwen3 0.6B 的短串行和单次取消/复用 GPU 探针已通过；结果见文末。该功能用于收窄故障阶段，不代表已修复 `0x116` / `0x139`，开关本身不解除机器守卫。

## 覆盖范围

`AILA_EXECUTION_TRACE_DIR` 非空时，共享推理 `DeviceRuntime` 在创建 SYCL Context 前创建记录器，直到 Context 的 stream/queue 全部析构后才关闭记录器。每个设备执行域独占创建一份 `execution-<PID>-<UTC ns>-<序号>.jsonl`，不会追加到旧文件。开关在进程内只读取一次，应在创建运行时前设置。

记录覆盖共享 Model/Session API 下的 Qwen3 NF4、Qwen3.5 Hybrid NF4/BF16，包括使用这些 API 的 IPC Worker 和 HTTP 服务。旧 CLI/旧 ABI 自行创建的 Context 默认没有记录器；不能用没有记录的旧路径代替本诊断入口。尚未增加每一层/每一个 kernel 的标记。

默认不创建记录器或文件。启用后使用独立 CPU 文件写入，不经过可延迟的 AILA 日志回调；Windows 使用独占新文件、write-through 和逐条 `FlushFileBuffers`，允许其他工具只读查看。没有增加 GPU 同步点、修改内核或改变数值计算。但写盘会延缓主机提交，诊断运行不能作为性能数据，也不能用“不再复现”单独证明故障已消除。

每个记录器有固定64MiB上限；到达上限也属于记录失败，会阻止新设备工作，并使Worker关闭返回失败。长期创建释放测试应与短功能诊断拆成独立Worker：短阶段保留完整trace，长循环在启动前移除 `AILA_EXECUTION_TRACE_DIR`，继续保留请求结果、USM/进程采样、机器守卫和Job/watchdog。不能让trace耗尽冒充设备计算故障，也不能通过无限扩大记录上限来维持长稳运行。

## 记录的含义

| operation | begin / end 的边界 |
| --- | --- |
| `model_load` / `session_create` | 共享模型/会话初始化及已有的最终队列排空 |
| `forward` | 一个实际 token chunk 的执行；正常 end 前已完成现有最终排空 |
| `decode_batch` | 一个含 2..4 个独立 Session 的 decode 步骤；`token_count` 为实际行数，正常 end 前已排空；不是单条序列的 prompt 长度 |
| `backend_forward` | 后端调用的主机入口/返回；**end 不表示 GPU 已完成** |
| `sampling` / `logits_read` | 采样或读取 logits 的主机调用，内部阻塞复制另有记录 |
| `queue_wait` | 现有 `Context::synchronize()` 的入口与成功返回 |
| `copy_h2d_wait` / `copy_d2h_wait` | 现有 blocking copy、event wait 和异步异常检查 |
| `session_reset` / `session_truncate` | 显式状态变更及已有排空；不是 HTTP 断连的必经路径 |
| `session_destroy` / `model_destroy` | 持执行锁后的排空与拥有资源的释放 |
| `usm_free` | `sycl::free` 调用；记录记账字节数和 device/pinned-host 类别 |
| `workspace_release` | Context 工作区、primitive cache 和共享 scratch 的释放 |
| `context_initialize` / `context_release` | Context 构造体内的 oneDNN 初始化 / 析构体内的资源清理；不包围 SYCL queue 的成员构造/析构 |
| `device_shutdown` | DeviceState 析构体的最终 drain；之后仍会执行 Context 成员析构 |

每行包含 `schema_version=1`、本文件 `trace_id`、连续 `seq`、PID/TID、UTC/单调时钟纳秒、scope/parent ID、operation、event/status，以及可继承的模型/Session ID、phase、token_count、position、target_position。`phase` 跟随后端形状：一个 token 的 prompt 也使用 decode 路径。`-1` 表示未知/不适用；model/session 的 `0` 表示未绑定。

合批外层只绑定共享模型，Session ID 与位置未绑定；每行的 `sampling` / `logits_read` 子 scope 重新绑定该 Session、它自己的位置和 `token_count=1`。B=1 仍记录原 `forward`，不能仅凭配置上限推断发生了多行执行。分析器接受新增的 `decode_batch` 操作并验证 scope 的字段继承与配对；未需要读取 logits 或采样的 Finalizing 行不会产生对应子 scope。

模型/Session ID 属于诊断记录，不是 IPC 句柄；不会写 prompt、生成 token ID、权重或张量内容。没有全局逐 kernel 完成序列。单调时钟用于本文件排序，UTC 用于与 runner/系统事件关联；系统校时可能让 UTC 回退。

`begin` 后必须显式完成才记录 `end/status=ok`；异常或未完成的 scope 析构记录 `end/status=error`。`trace_open` 和 `trace_close` 表示记录器的生命周期。最末 begin、缺少 end 或缺少 close 只能说明现有证据未完成，不能直接认定该操作就是根因；驱动延迟恢复和未持久化的尾部仍可能影响判断。

## 记录失败与验收

记录器在写入/flush 失败或超过每文件 64 MiB 后锁存失败并停止追加，不重写、不重试部分记录。失败在持有写锁时锁存，其他线程不能继续追加进损坏的尾部。scope 记录与析构不抛异常，确保已有排空和释放仍会尝试；Context 的提交/分配访问点及共享推理边界拒绝后续工作，原有设备异常优先保留。

失败同时保存在进程级原子标记中，直到进程结束都不清除。最终 `trace_close` 有可能整行可读，但其 flush 返回失败；此时仅解析文件无法辨别。IPC Worker 在 `Stop()` 释放运行时、Context 和记录器之后检查该标记，以非零退出码报告失败；宿主已有的进程退出检查将其传播为关闭失败。不能只凭文件完整或关闭响应成功验收，也不从析构抛异常。

HTTP runner 的 `--execution-trace` 只允许与短 `--mode probe` 一起使用。它在本次新 run 目录创建 `execution-trace` 子目录并设置 Worker 环境；未指定该选项时移除继承的 `AILA_EXECUTION_TRACE_DIR`，防止旧路径混入证据或静默改变性能测量。

`--probe-cancel` 进一步要求 `--mode probe --execution-trace`，固定执行三个串行步骤：完整独立参考、首个内容事件后断连、创建新 Session 并对照原参考。普通 probe 仍只执行两次完整请求。取消目标默认最多 16 个生成 token，必须满足 `new-tokens < cancel-new-tokens <= 16`；会话数仍为 1、context 不超过 512、整轮 watchdog 不超过 60 秒。此入口复用 batch 的取消计数、具体请求 retirement、idle 与参考一致性检查，不进入至少 257 次请求的 lifecycle 循环。目标提前完成、被超时取消或复用输出改变都算失败。

runner 先清理 owned Job，再读取 trace，最后才可提交 guard 成功。需满足：至少一份完整 trace、文件名和 trace_id 一致、PID 等于本轮已识别的 Worker、无缺号/非法 JSON/错误 scope/未关闭 scope，并有最终 trace_close。解析结果及原始文件 SHA-256 写入 `result.json`；已有的退出码、服务最终报告和 guard/result 哈希门禁继续生效。读取失败同样不能提交成功。

`tools/diagnostics/execution_trace.py` 可离线读取已有文件。它限制每文件 64 MiB、每条物理记录 1536 字节（含换行），按 C++ 整数范围验证字段，并检查每线程嵌套、父子关系与连续序号，保留有效前缀摘要和未关闭 scope。截断、NUL、重复 JSON 键、错误 end、缺少 close 均不能视为完整通过。文件逻辑关闭不等于进程已成功退出，进程退出及服务最终报告由 runner 单独核实。

## 验证边界

本批已完成编译和 CPU writer/解析器/runner 故障注入。Release 全部已配置目标 88 步重建成功；普通 MSVC Server 41 步编译成功，32 个 C++ object 的 header 依赖有效。根目录指定 CPU CTest 9/9（12.23 秒），普通宿主 CPU CTest 4/4（10.64 秒）。其中 writer 10 组、RuntimeFault 9 组、fake Worker pipe 9 组均通过；后者在两种宿主编译器下都确认最终 flush 失败会在关闭响应之后经 Worker 非零退出传回客户端，重复 Shutdown 保留失败且宿主额度清零。Python 解析器 37 项、guard/runner 81 项通过，共 118 项。

使用真实 C++ encoder 输出与 Python CLI/函数进行交叉检查，结果为：

| CPU 模拟 | 原始记录 | helper 退出码 | 文件逻辑完整 | 结合进程退出的验收 |
| --- | --- | --- | --- | --- |
| 正常退出 | 18 行 | 0 | 是 | 通过 |
| `std::_Exit(0)` 中断 | 3 行；保留 `forward`、`queue_wait` 两个未结束 scope | 0 | 否 | 拒绝 |
| 最终 close 整行写入后注入 flush 失败 | 4 行；含 `trace_close` | 5 | 是 | 拒绝 |

三份原始文件、分析结果及退出码保存在 `tmp/refactor/stability-20260909/cross-language/validation.json` 所列路径；分析前后原始文件 SHA-256 一致。`dumpbin /dependents` 确认 writer 测试、fake Worker 和 pipe 宿主没有 SYCL/oneDNN/Level Zero 导入，pipe 测试还检查宿主加载模块。这些模拟不提交 GPU 工作。

实际构建、测试日志及新源码/产物身份记录在 `tmp/refactor/stability-20260909/implementation-validation.json`。首次构建因独立 kernel fixture 缺少日志实现的链接依赖失败，已补齐该依赖并完整重建；失败日志也保留。原 9 月 8 日的 11 份验证日志在 Server 重建前按旧快照核对 SHA-256，复制到本目录 `previous-validation-logs`，原转储、分析和验证快照未修改。上述记录属于初始 CPU 阶段，后续 GPU 恢复另存证据。

## 2026-09-09 受控 GPU 恢复

在用户启动的管理员 CLI 中重新核对了 89 项源码/产物/模型身份，并读取完整的系统快照。沿用 A770、驱动 `32.0.101.8974` 和第 14 节验证过的 C++ 二进制。所有运行独立持有机器守卫和 Job/watchdog；原始失败和审阅记录保存在 `tmp/refactor/gpu-resume-20260909T063259Z`。

首轮使用 15 秒应用请求期限，首个 27-token prefill 加采样耗时 15.754 秒，服务记录 `request_timeout`。这轮仍失败，8,744 条 trace 无错误并完整关闭，进程和设备清理完成，系统快照未增加故障。长间隔位于尚未细分的后端主机段，不能据此把耗时确定归因于 JIT、具体 kernel 或写盘。

明确审阅失败后，仅将应用请求期限改为 45 秒，保留 60 秒整轮限制、8-token 上限和原负载，复测两次请求均完成且输出一致，trace 为 9,250 条。新增取消入口的 GPU 用例随后通过：两次完整请求一致、一次被服务确认为断连取消，9,644 条 trace 完整；按串行创建顺序，取消 Session 的等待/释放全部结束后才创建后续 Session。两轮均零退出、guard `finished`/结果哈希匹配，HTTP/IPC/Job 排空，事后快照可读且无新故障。

新增 7 项 CPU 行为/参数测试后，guard/runner 88 项与解析器 37 项全部通过。默认无 trace 路径的 1/2/4 并发短回归另通过 28 条请求；它们不提供逐阶段 trace，也没有足够样本建立性能分布。逐层记录、显式 reset、长上下文和长期压力仍未完成此次恢复验证。

## 2026-09-09 复发后的验收边界

最后一次 runner 零退出、owned Job 清空约 33 分钟后，系统仍发生 `0x141 → 0x116`；崩溃前 WER 进程清单没有 Aila 或 Python。GPU 已再次暂停，机器守卫已重新阻断。完整 trace、正常退出和即时系统快照支持对应运行的收尾检查，无法证明驱动随后在桌面使用中持续稳定。根因及具体触发进程仍未知，系统稳定性未通过，P6 长稳未验收，P7 设备验证暂停；新证据与旧快照的保存方式见 [复发调查](stability-recurrence-20260909.md)。

## 2026-09-09 DDU 重装后的再次验证

晚间用户完成 DDU 清理、全新安装 8860 WHQL 并重启后，驱动 / 实际 Level Zero 加载版本核验为 8860 / 1.28.2，GPU 已按授权恢复。相同二进制的串行与取消/复用探针通过，分别留下 9,250 / 9,634 条完整 trace；默认非 trace 路径继续通过 1/2/4 并发、512 次正式生命周期请求及 1848-token 输入测试。六轮合计 788 请求，各轮及最终系统快照无新增故障，守卫最终为 `finished`。这些有限结果及未覆盖的长期系统行为见 [8860 验证记录](gpu-stability-8860-20260909.md)。
