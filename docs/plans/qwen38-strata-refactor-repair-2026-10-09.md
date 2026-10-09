# Qwen3.8-Flash-Next：Strata 风格重构接手修复

接手基于 `35f750a` 的已暂存重构。原版本可加载权重，但首次 2-token forward 报 Level Zero Error 40，随后以 0xC0000409 退出。本轮完成了实模型、单流 Server、PI 工具回合和长上下文续轮的功能闭环。**尚未证明端到端提速，decode 20+ 仍未达到。**

原重构把全部 24576 个专家（62,285,414,400 B）放进 Host USM，并在显存重复保存热专家；进程 Working Set 的下降不能证明这些内存已不可分页驻留。复现期间系统可用物理内存最低约 0.06 GiB。仅分配但不完整触达大 Host USM 的小探针能够执行 device-only kernel，因此不把 Error 40 解释为任意大 USM 分配都会失败，也没有依据把它归因于两个 token 共用顺序队列上的 down 缓冲。

修复后的 RAM arena 使用现有原生 MEM_LARGE_PAGES 分配器，只保存 GPU 缓存的补集，按完整专家分块且每块最多 256 MiB。GPU 初始槽数上限 2800，并扣除实际会话、工作区预留和双 staging bank 后确定；32K 使用该准入结果。4K CLI 样本为 GPU 独占 2800 个专家、RAM 21776 个专家，专家大页约 51.44 GiB，另有 embedding 大页。分配失败显式报错。每层加载后回收普通 mmap 工作集，大页不受影响；运行时没有专家文件读取后备路径。小型 Host USM staging 两组各最多 16 个专家，避免把整套 RAM 权重注册为 Host USM。

缓存替换先保留新入专家的 H2D 源，再把被替换专家的唯一 GPU 副本搬入腾出的 RAM 槽，最后交接所有权。copy/consumer 事件由缓存持有，跨 forward 保留；设备 bank 的下一次写入依赖前次 down 消费完成。CPU 专家复用原执行器和 18 线程配置，直接读取当前 RAM 所有者，Join 完成后才允许交换所有权。没有新增线程池或推理并发路径。

同时修复了以下遗漏：GPU resident/staging 描述符按专家编号偏移 nested scales，CPU 描述符填写 blocksize=256；GPU 元数据分配失败不再回退到普通 vector 指针；初始热集不再产生重复 key；移除默认跳过 31 个 PLE 分片的代码，实际完整索引 131 分片、4114 个文本张量。长 prefill 恢复既有 expert-major W4/G3/XMX 分发与按 rank 合并，每层每个窗口的冷专家只上传一次。warmup 异常时先排空再释放临时缓冲，释放路径不再重复抛出已记录的异步错误。

两个定向 CTest 共 9.38 秒通过。检查包括原生 LargePage 位、页对齐账本、完整 DMA 字节和尾部 guard、GPU/RAM 补集、两个相邻调用复用同一 staging bank、连续所有权交换，以及 48×512 key 空间中 2800 个槽的唯一性。H2560/I640 的 GPU NF4 输出与独立原始权重副本逐位一致；CPU A64 输出在交换前后都与原始权重源逐位一致。没有运行全套测试；原先单测中额外分配 58 GiB 的 warmup 被替换为小容量的实际计算检查。

实模型修复过程中，CLI 的 2-token prefill、1-token decode 和短生成通过，退出码 0，最低可用内存约 4.59 GiB。该 CLI 样本启用了诊断且早于 CPU 接入收尾，不作为最终性能样本。

最终生产式验证使用同一冻结自然语料 tmp/q4e-agent-corpus-v1、32K 容量、chunk4096、reserve3584、18 CPU 专家线程、BF16 KV、MTP 关闭；诊断和 Worker 日志捕获全部关闭。Server 和 Worker 使用匹配的构建标识。输入内容、token hash、请求 hash、输出数量与输出 hash 保存在同名 JSON 和原始记录中。

| 阶段 | 结果 |
|---|---|
| 服务启动至 Worker 就绪 | 149.05 秒 |
| 就绪后第一个真实 PI 工具回合 | 88.36 秒；计入启动共 237.42 秒 |
| PI 输入/输出 | 首轮 6931/26；工具续轮复用 6957、新增输入 32、输出 14 |
| PI 任务 | 恰好一次 read(smoke-marker.txt)，正确返回 AILA_FIXED_AGENT_READ_20261007 |
| 取消后新请求 | 正确返回 READY，总耗时 1.70 秒 |
| 14K 长请求 | 输入 13979、输出 32；首字 99.82 秒、总耗时 102.28 秒 |
| 14K 续轮 | 输入 14035、复用 14011、新增 24、输出 32 |
| 续轮速度 | 首字 1.73 秒、总耗时 4.12 秒、首字后 12.99 tok/s |
| 系统可用 RAM 最低值 | 加载阶段约 4.95 GiB，运行阶段约 4.91 GiB（2 秒采样） |
| Server 收尾 | 完成 5、取消 1、失败 0、拒绝 0；正常退出 0，未强杀 |

这些数据确认功能恢复和物理驻留有效，不能据此宣称加速。旧报告的部分输出长度为 64，本轮长请求输出为 32；未做三对同二进制性能比较。当前启动仍需 149 秒，是明确保留在账本中的成本。后续性能工作应在这个可运行版本上单独比较，不能用本次故障修复代替性能准入。

使用已配齐的本机二进制可直接运行 tools/server/start-qwen38.ps1。新的自适应执行路径使用原生大页补集，已删除跳过 PLE 的入口；不需要打开旧 LARGE_PAGE_RESIDENT 或 ADAPTIVE_PIPELINE 开关。重编 Worker 后，也需用 tools/server/build.ps1 -SkipDependencies 配齐 Server 的构建标识。

原始记录位于 tmp/q4e-refactor-repair-20261009/：handover.patch 保存接手时暂存内容，final-source.patch 保存最终代码差异；original/ 是故障复现，repaired-first/ 是中间 CLI 检查，targeted-tests-cpu.log 是最终定向检查，agent3/ 和 agent3-monitor.json 是最终真实 Agent/HTTP 验证。agent/ 保留旧 Server 构建标识不匹配的失败，agent2/ 保留固定槽数不满足 32K 显存准入的失败。未覆盖或清理这些失败记录。
