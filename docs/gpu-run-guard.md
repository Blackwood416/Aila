# GPU运行守卫与WER重复投递

`tools/server/gpu_run_guard.py` 在机器固定目录保存运行状态、系统快照、带校验的journal和待审证据。一个合作运行持有进程级文件锁；watchdog和Windows Job由调用者负责。守卫本身不启动GPU，不修改驱动或TDR设置，也不自动确认失败或重试。

## 哪些情况继续阻断

首次快照已有故障/转储、运行未正确结束、非零退出、设备错误、新的故障报告或dump、启动会话变化、日志回退、证据读取/写入失败都会阻断。已登记的failed/blocked证据不会因为记录消失、后续出现重复报告或分类器更新而清除；仍须先审查，再按精确evidence ID调用显式确认接口。

Windows会在重启或稍后重新提交ReportQueue中的旧报告。Application日志的记录编号与写入时间可能都是新的，但报告对应的仍是旧事件。此前多次隔离诊断确认的48份历史报告存在这种重复投递。

## 严格的重复报告识别

只有同时满足以下条件，新增日志记录才归为同一报告的再次投递：

1. 来源必须是Application / Windows Error Reporting / event1001，EventName为BlueScreen或LiveKernelEvent，并含有效且非零的GUID ReportId。
2. 完整EventData及log/provider/event ID与运行前已有报告相同。P1–P10、ReportStatus、AttachedFiles、StorePath等任何字段变化都不能略去。
3. 报告必须引用绝对路径的dmp。每个关联转储要么在前后快照中有完全相同的路径/大小/mtime元数据，要么在两次完整扫描中均不存在。
4. 对“持续不存在”的判断，前后必须声明相同扫描目录和MEMORY.DMP路径、覆盖该文件位置、没有读取错误，且整个转储清单未变。旧快照没有扫描范围时，不推断文件不存在；这些记录仍保守阻断。

新快照保存 `dump_roots` 和 `full_dump_path`；已知扫描范围发生变化也会阻断。新ReportId、不同payload、关联文件新出现/消失/变化、其他新增dump、System故障事件、boot变化和读取错误仍作为待审证据。

原始事件始终留在snapshot中。运行启动和结束时的识别结果另写入guard的 `historical_wer_replays`，包括before_start/during_run阶段、完整payload摘要、匹配记录及已存在/持续不存在的关联dump路径。显式确认过程仍保存其完整新快照。这个分类不改变已有失败轮的状态或验收结果。

这些规则用于区分报告投递与新的事故，不能保证发现所有驱动问题，也不能代替GPU进程的退出码、数值检查、队列收尾和设备错误检查。

## 验证入口

```powershell
python -m unittest discover -s tools/server -p test_gpu_run_guard.py -v
python -m unittest discover -s tools/server -p test_service_benchmark.py -v
```

CPU用例注入快照，不调用Windows故障探测或GPU。它们覆盖已有失败不被清除、dump/ReportId/payload/扫描范围变化、日志回退、读写错误、进程锁和失去进程后的隔离。历史失败数据只作只读分析，不改写旧guard或result。
