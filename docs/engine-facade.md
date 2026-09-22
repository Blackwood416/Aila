# Engine 兼容外观与实现边界

`include/engine/Engine.hpp` 保留原 `InferenceEngine` 和派生流对象的接口、成员及默认参数。加载、上下文、生成和媒体函数体移到 `src/engine/`，由 `AilaLib` 和非隔离共享库共用。

| 文件 | 职责 |
| --- | --- |
| `EngineLoad.cpp` | 模型检测、加载、tokenizer、LoRA 与媒体组件初始化 |
| `EngineContext.cpp` | 上下文重置、容量和历史统计 |
| `EngineGeneration.cpp` | prompt 对齐、聊天模板、thinking、生成与结构化流事件 |
| `EngineAudio.cpp` | ASR、对齐、TTS、speaker embedding 和 `TranscribeStream` |
| `EngineAudioCache.cpp` | 参考音频的内存/磁盘缓存与键计算 |
| `EngineBenchmark.cpp` | 旧 prefill/decode benchmark 入口 |
| `EngineDetection.cpp` | 检测入口及其统计查询 |
| `AsrText.cpp` | UTF 转换、重复消除、语言与 ASR 输出解析；声明在 `engine/AsrText.hpp` |

此次共移动 74 个函数，头文件缩为 457 行。函数体按 UTF-8 和统一换行后逐个 SHA-256 核验原样保留，类布局、隐式析构/移动行为和回调执行位置没有调整。原转递 include 暂时保留，后续可在独立兼容门禁下收窄。

这是旧兼容外观的编译边界提取。新共享模型、Session、Request 和 Scheduler 仍由 `src/inference/` 提供；媒体方法尚未迁入新并发执行驱动。使用这些旧方法仍须遵守原来的单会话约定。

验证包含完整 Release 构建、9 项 CPU/旧 ABI 兼容测试、三种文本后端与旧独立 CLI 的完整 JSON 对照，以及 Hybrid NF4 的 24 条结构化流事件对照。Diffusion kernel fixture 已另行在机器守卫下通过；没有将这项算作 CPU 测试或完整媒体端到端验收。证据及失败修正见 [验证记录第 22 节](refactor-validation.md#22-2026-09-09-engine-兼容外观实现提取)。
