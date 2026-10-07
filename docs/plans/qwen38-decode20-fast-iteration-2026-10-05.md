# Decode 20+：继续任务的快速迭代

承接 chat 01a1009d-05d1-7cb3-8dd4-64d41e3dd205。目标仍是 A770 单请求 decode ≥20 tok/s。

**结果：本轮没有达到 20。原配置复现 17.4–17.6 tok/s，所有新运行时代码已撤回，18 CPU 线程 / reserve 2560 MiB 保持。** 新阶段没有保留额外线程池、kernel 变体或实验开关。逐项结果见下表，机器可读记录见同名 JSON。

机器：A770 / driver 32.0.101.9033、Xeon E5-2696 v3（18 核/18 逻辑处理器）、oneAPI 2026.1。模型：C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2。

测量沿用 archive-natural（1833 prompt tokens）、同进程两次请求、每次 128 outputs。下表为正常生成初筛；第一请求与暖第二请求分开。一次只运行一个 GPU 工作负载，测速不与编译重叠。不同调度会改变 ready-adoption 以及 CPU/GPU 近似算术的后续路由，单次跨二进制比较不用于声称精确的内核加速比例。

| 实验 | 首次 tok/s | 暖第二次 tok/s | 处理 / 日志前缀 |
|---|---:|---:|---|
| 原配置、CPU byte-shuffle 关闭 | 13.0 | 17.6 | 本轮复现；day5-s26-shuffle0…r12 |
| 上轮 CPU byte-shuffle 候选 | 13.0 | 16.1 | 不启用；day5-s26-shuffle1…r11 |
| HC 直接写 pinned CPU 输入 | 11.1 | 14.9 | 已撤回；day6-mirror1-r1 |
| classify 连续写 pinned CPU 输入 | 12.9 | 17.4 | 无收益，已撤回；day6-classify1-r1 |
| CPU 12 线程 | 12.2 | 14.5 | 保留 18；day6-threads12-r1 |
| TBB static_partitioner | 12.1 | 14.3 | 已撤回；day6-static1-r1 |
| decode 共享专家加法并入有序合并 | 13.1 | 15.9 | 已撤回；day6-shared1-r1 |
| 当前二进制全关对照 | 13.1 | 17.4 | 与 shared 候选同 EXE；day6-control-current |
| CPU A64 rows2 内循环展开两次 | 13.1 | 17.4 | 无收益，已撤回；day6-unroll2-r1 |
| reserve 2816 MiB | 12.9 | 17.4 | 无收益；day6-reserve2816-r1 |
| reserve 2304 MiB | 12.8 | 16.6 | 无收益；day6-reserve2304-r1 |

HC mirror 复用原 pinned arena，由生产 kernel 的完成事件发布输入；专项 --hc-only 逐位/guard/USM 释放检查通过。classify mirror 通过现有 readiness 套件（含复用、rank 合并、取消）；TBB static 通过既有 A64 独立 scalar-bitwise 对照。三项均因整条管线没有达到更好的速度而撤回，没有引入新的线程池或长期保留实验分支。早期误启动的全 kernel suite 未完整结束，不能计为全套通过。

WorkSummary 诊断为 day6-baseline-summary：首次/暖 10.6/12.7 tok/s，不能混入正式速度。127 个有效 decode forward 的暖均值为 CPU tasks 240.89、compute 27.396 ms、join 39.393 ms、queue 0.590 ms。join 包含等待前置 GPU 路由的部分，不能与 compute 相加；日志运行的热集和生成内容与正常生成不同，因此不能把这些数直接当作 17.6 tok/s 轨迹的分解。

原始每轮 .log/.err.log/.meta.json 保留在 tmp；meta 保存实际参数、EXE hash、退出码和内存 watchdog 结果。失败实现留档为 tmp/day6-hc-mirror.patch 和 tmp/day6-rejected-experiments.patch；后者包含当时未构建的 shared 开关草稿，不能将整份 patch 视为 archived EXE 的精确源码身份。

A64 rows2 内循环 unroll_count(2) 通过既有独立 scalar-bitwise 测试，但完整生成仍为 17.4，因此撤回这一行提示。共享专家融合也未保留新开关。

最后两个纯参数候选 VramReserveMB=2816 / 2304 使用 day6-control-Aila.exe（与当前全关对照相同）。2304 增加 hotset 到 3657 个专家，却未改善整条 decode；2816 也无净收益。因此保留 2560 MiB / 3551 hot experts。

从 17.4–17.6 到 20，需要把约 56.8–57.5 ms/token 压到 50 ms，仍差 6.8–7.5 ms。此次没有证据支持靠小拷贝融合、TBB 分块、线程数或相邻显存预算点补齐差距。后续应复用已有 teacher/replay 固定 CPU/GPU placement 与真实输入，定位冷专家计算的主要开销；避免再为这些已否决的小开关增加测试或架构。上述诊断的 CPU compute/等待是调查线索，不是已确认可节省的时间。

收尾核对：本轮修改过的 backend、executor、MoE ops、CPU expert ops、readiness tests、probe 脚本均与接续时 Stage26 源码逐文本一致（忽略换行编码）。原任务未提交的改动全部保留。本轮新增交付只有本报告和 JSON；恢复构建日志为 tmp/day6-restored-build.log。
