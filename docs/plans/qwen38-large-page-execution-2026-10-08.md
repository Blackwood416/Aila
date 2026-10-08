# Qwen3.8-Flash-Next Large Page 执行模型

本轮在当前 Qwen4Exp 路径上参考 `tmp/Strata`，保留既有 NF4 算子和 CPU/GPU 专家分工，重构驻留专家的供给。Strata 的有限传输环、按消费事件回收和协调线程顺序提交适用于此处；没有移植 CUDA arena 注册或跨层猜测预取。

- 大页源已经是完整 slot 布局，每个专家一个任务复制全部 packed NF4 和 nested codes，取消原来的八份填充任务及末任务原子计数。复用原有填充线程作为生产者，协调线程复制第一个专家；专家完成后通过 CPU release/acquire 通知发布，协调线程立即顺序提交其 H2D，与其余专家填充重叠。
- 复用现有 pinned staging，两个半区各保存跨 acquire 的完成事件；只在复用该半区前等待。协调线程顺序提交逐专家 H2D，原有整批 ready barrier、GPU slot reader fence 和所有权交换仍生效。
- `RESIDENT_PREFILL=1`、`EXPERT_MAJOR_PREFILL=1`、正数 `DEMAND_MICROBATCH` 时，GPU prefill 借用区按两批需求配置。实际 microbatch32 从 512 槽缩至 64 槽，GPU 独占专家从 2312 增至 2760，RAM 专家从 22264 减至 21816；全部 24576 个专家仍有所有者。
- pinned staging 扩至两批的大小，microbatch32 为 162201600 B；释放大页模式不再使用的预取 staging。分配只发生在初始化，CPU 解码读取实时大页所有权表。
- 专家大页块默认由最多 256 MiB 改成最多 128 MiB。实际每块 52 个专家，页对齐后 126 MiB。本轮原 256/128 MiB 大补集启动均遇到过 Win32 1450；缩小 RAM 补集后完整模型启动成功。仍使用原生 `MEM_LARGE_PAGES`。

新容量的专家 payload 为 55290470400 B，元数据和 embedding 不变；128 MiB 分块下大页总分配为 56732155904 B（52.84 GiB）。这与原 53.81 GiB 容量不同，不能把后续对照全部归因于 staging。

`AILA_Q4E_LARGE_PAGE_PIPELINE=0/1` 只切换旧/新 staging 路径，两边使用相同新容量；默认关闭新路径，只有设置为 1 才使用生产者/协调线程模型。容量重构和 128 MiB 分块默认生效。`AILA_Q4E_LARGE_PAGE_CHUNK_MB` 可设 16..256，默认 128。完整 6K 请求对照沿用原自然代码 fixture、32K 会话容量、chunk4096、reserve3584、18 个 CPU 专家线程、BF16 KV、MTP 关闭和 prefix reuse 关闭。每条生成 64 tokens，关闭性能诊断；不清空 OS 文件缓存。

全半区填完再提交的初版在相同二进制对照中没有胜出：6K 重复请求首字/总耗时为旧路径 32.28/36.36 s、候选 32.77/36.95 s；两边输出 hash 一致且正常退出。因此调整为上述专家 ready 即提交，恢复批内重叠。初版数据保存在 `tmp/q4e-largepage-execution-a2/b2`，不将其报告为加速。

最终同二进制对照为 A3（旧 staging）和 C3（专家 ready 即提交），实际各请求 5968 prompt tokens、64 output tokens、cached_tokens=0。两边使用相同新容量，因此不能从此表隔离容量重构的收益。

| 请求 | 旧 staging 首字 / 总耗时 | 候选首字 / 总耗时 | 旧 / 候选首字后 tok/s |
|---|---:|---:|---:|
| Worker ready 后首次 6K | 73.678 / 78.504 s | 75.531 / 80.577 s | 13.053 / 12.485 |
| 重复完整 6K prefill | 32.226 / 36.301 s | 32.233 / 36.399 s | 15.458 / 15.123 |

重复项首字只差 0.0069 s，候选总耗时慢 0.27%；没有支持默认启用新流水线的提速证据。启动为 A3 83.294 s、C3 79.150 s，受到源页与物理内存状态影响，不解释为流水线收益。两边所有请求输出 hash 一致，Server 和 Worker 均正常退出；运行可用 RAM 最低分别约 5.38 / 5.28 GiB。

默认启用的实际改善是有限预填充借用区：专家 RAM payload 减少 1135411200 B（1.057 GiB），包括页对齐的总分配从原 53.81 GiB 降到 52.84 GiB，保留的 GPU 核心增加 448 个专家。原 512 槽容量本轮在当前物理内存状态下分配失败，没有获得可用的隔离性能对照，因此不声称这一变化带来确定的 tok/s 加速。

三个已有定向检查通过：原生大页/净账本/所有权/释放，完整 48×512 专家编号热集交接，以及 H2560/I640/E512 的真实几何异步 CPU 专家结果逐位一致。最终把流水线默认值改为关闭，重新编译 Worker 和 readiness executable，再次通过默认路径的 `--large-pages-only`。性能对照使用显式开关，不受此默认值变化影响。未运行全套测试。性能结果与定向检查记录保存在同名 JSON；A2/B2 与 A3/C3 的完整原始结果保留在 `tmp/q4e-largepage-execution-*`。
