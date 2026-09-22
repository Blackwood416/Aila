# 路线图完成审计

本表对应 [完整路线图](refactor-roadmap.md)，跟踪用户要求的全部计划。某个纵向切片通过不等于所在阶段全部完成。数值、服务和资源证据记录在 [验证文档](refactor-validation.md)；历史失败证据继续保留。

| 范围 | 已有交付 | 剩余工作与验收证据 |
| --- | --- | --- |
| P0 可复跑基线 | 固定模型、驱动/运行时身份、CPU/GPU/IPC/HTTP fixture；重构前完整 logits 捕获程序；三后端各三次独立 HTTP 进程、共2844请求/2520正式请求，原始分布与跨进程波动、明确标注HEURISTIC的审查线及离线分析工具 | 扩展近容量/较大模型基线并验证可迁移的回归门限；关键层首次分歧定位和长期输入基线 |
| P1/P2 执行资源和共享 Session | DeviceRuntime、LoadedModel、Session、Request；Qwen3 NF4 单权重多会话；请求 RNG 和生命周期；Engine 74 个函数原样提取至 8 个实现文件，三后端 CLI 及流事件与旧产物一致 | 收窄 Engine 转递依赖；覆盖所有显式和库内执行分配的准入策略；媒体并发迁移另行验收 |
| P3 Hybrid 层执行 | Qwen3 与 Hybrid 固定层计划，Hybrid NF4/BF16 独立状态与 checkpoint；公共 Hybrid 执行模板及投影策略通过四模型旧产物逐位对比 | 继续覆盖旧媒体、legacy linear 和调试路径；更完整的层输出门禁 |
| P4 调度 | token/分块 prefill 轮转、有界输出、独立取消、设备故障传播；context2048长短混合下三后端的首内容进度与完成顺序分开记录，保留全部例外批次 | 取消最坏延迟和更广泛的调度公平性，完整资源准入及异常压力 |
| P5 IPC/服务 | 三种文本后端通过 IPC v2、C++ HTTP/SSE 和新异步 C ABI；thinking budget、reset/truncate/prefill 与生命周期门禁；旧文本 ABI 可选适配历史、模板、工具/结构化流、四 Engine 共享 Worker、回调销毁和取消前缀恢复；三后端与独立旧产物一致，45 个旧导出不变、当前28个异步 C 导出和 MSVC 消费者通过；可选长输出额度、1100-token 请求/续写与有界 Worker 日志及级别控制通过；旧入口五项资源额度拒绝/恢复与初始化错误分类通过；HTTP 工具定义/选择/历史、JSON/XML 与有界 SSE 参数分片、独立工具 token 恢复及三后端调用/历史/并发通过，6组普通文本与旧 Server 一致；NF4 模型格式失败保留为能力边界 | 旧文本 adapter 默认切换前的近容量/32768-token设备执行、动态共享额度压力与完整资源计量、decode/stream chunk 性能；HTTP 媒体执行驱动 |
| P6 集成 | 多模型有限 IPC/HTTP 回归；额度拒绝/恢复、LoRA scratch、BF16 grouped、远端reset/truncate/replay与运行时USM查询通过；三后端context2048服务基线及生命周期通过；四模型（含4B NF4）context4096满容量、回滚和prefill/decode取消恢复、4096次正式Session创建释放/1024完成/1024取消通过，USM逐轮稳定且最终归零；WER完整报告身份及扫描覆盖识别通过41+58项CPU测试；全部失败/审查/复验保留 | 32768等更高容量设备执行、小时级持续压力、完整内存压力和驱动私有资源策略、扩大可复现性能门限验证域 |
| P7 连续合批 | 同模型decode批次、显式1..4上限且默认1、逐序列位置/KV/采样/LM head、动态增删、取消/错误边界及IPC/C ABI/HTTP配置通过；NF4 decode rows保留原GEMV算术。四种配置288行完整logits零差异，冻结旧产物B1、请求重放、前向后ResourceLimit恢复及HTTP通过，共23轮正式GPU验收；CPU/ABI/MSVC通过，见验证第33节。2026-09-18/21 补齐量化验收：并发4短输入上限4相对上限1为 Qwen3 +24.8%/+26.9%、0.8B NF4 +25.1%/+26.1%、4B NF4 +11.4%，长短混合 +4.6%..+20.4%，并发1在±1.4%内；BF16 补齐多行投影后由 +8.7% 提升到 +57.0%（短）/ +35.0%（混合），四种配置288行logits仍逐位一致且ctest 54项全通过；近容量长prompt（context 4096、实际3725/3916 token）上限4相对上限1为 +8.8%/+11.7%；四配置各512次宽4生命周期通过；多行decode attention已扩展到 baseline 与 exact-partial-merge 两条路径（head_dim 128/256；同二进制交替A/B：BF16 +3.9%、0.8B NF4 +2.8%，六组配对同向），并新增近阈值长上下文批次（capacity 2176、B=4），四配置各12行逐位相同且路径日志直接记录 multi-row-exact-merge；decode window/sink 与 head_dim 128 低于阈值时的 joint-matrix 仍逐行；小时级长稳以背靠背进程完成（Qwen3 8024请求/1.09h、0.8B NF4 7752请求/0.98h），0失败且同配置p50波动<1%；实际批宽2/3/4由执行trace在四配置上分别证明，见 [合批边界](continuous-batching.md) 与验证第34/35节 2026-09-22 补齐 head_dim 128 低于128 tokens 的多行 Joint-Matrix decode attention（`attention_decode_rows_joint_matrix_impl`），使用工作组本地 SLM 融合执行并消除全局 scores_buf 中转；扩展基准测试与服务端套接字测试支持非2幂次并发（宽3，cases 1..4全覆盖）；闭环 mm_history_ 外部导入思维链剥离与单元测试。 | 分页 KV 读写/回收与前缀共享（已完成32-token架构规范设计），驱动私有显存计量（DXGI/Sysman双通道方案已设计）；跨天长期运行验证；保留历史网络传输偶发失败记录，门禁未放宽。 |
| 后续 Attention/KV | 保留既有连续 KV 和 attention 快路径 | batched/ragged attention、分页 KV 读写/回收及性能评估，跨请求前缀共享 |
| 后续设备/adapter | 首版单 A770、加载时固定 LoRA 组合 | 多 GPU 资源域与调度、多 LoRA 同批隔离和真实设备验证 |
| 视觉/语音/对齐 | 旧兼容外观与派生 owner 生命周期保留 | 显式媒体 Session 能力；vision、ASR、强制对齐执行驱动和并发隔离 |
| TTS | 旧 talker/predictor/vocoder 路径 | 单独的流状态驱动、取消/资源预算、各音频阶段的会话隔离与并发 |
| Diffusion | 旧 DiffusionGemma 能力和回归保留 | canvas/去噪步骤驱动、MoE 执行资源与专门内存预算、共享服务验收 |
| 检测 | 旧 YOLO 能力与 InferRef adapter | 检测能力接口、执行资源归属、服务与并发验收 |

完成标准：每项必须有实现、适合该变更的数值/生命周期/兼容验证以及可复跑证据。现有机器无法证明多 GPU 的真实行为；可先实现和验证 CPU 调度契约，真实设备验收仍须单独记录。此限制不阻断其他实现。
