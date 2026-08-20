# Aila

<p align="center">
  <b>充分发挥 Arc 显卡性能的推理引擎。</b><br>
  <a href="README.md">English</a>
</p>

---

> [!NOTE]
> 该项目仍在积极开发中，并不能兼顾支持的所有模型。部分模型的性能可能并不理想。

基于 **SYCL + oneDNN** 构建的高性能 LLM 推理引擎，专为 **Intel Arc 显卡** 设计。针对 bitsandbytes 4-bit (NF4) 量化模型提供手写优化 kernel，包括融合反量化+矩阵乘法、GEMV 解码，以及 Qwen3.5 混合架构的 GPU DeltaNet 循环加速。

## ✨ 功能特性

- **⚡ Bitsandbytes 4-bit (NF4) 推理** — 融合反量化+矩阵乘 kernel、手写 GEMV 解码、融合 gate+up+SiLU 投影，直接在 Intel Arc 上运行量化模型
- **🔢 Bfloat16 推理** — 通过 oneDNN 矩阵乘法原语支持密集（非量化）模型
- **🏗️ Qwen3.5 Hybrid 架构** — 完整支持双重注意力（GQA + DeltaNet 线性注意力），GPU 加速 delta 循环计算
- **📐 Qwen3 Dense 架构** — 标准 Transformer，支持 GQA、QK-norm 和 SwiGLU FFN
- 👁️ 视觉理解 (Qwen3.5) — 支持图像输入，CPU 预处理 + GPU 视觉 Transformer
- 🎯 **YOLO26 目标检测** — 支持 n/s/m/l/x、固定 640×640、NMS-free one-to-one head、FP16 oneDNN/SYCL 推理，以及 UTF-8 类名
- 🎙️ 语音转录 (Qwen3-ASR) — 基于音频预处理与 GPU 加速音频编码器的语音转文本（ASR）。支持离线 wav 转录和实时流式输入转录
- 🎯 **强制对齐** — 音频 + 文本 → 词级时间戳对齐，支持 CJK 逐字分词和 LIS 时间戳修正
- 🔊 语音合成 (Qwen3-TTS) — 基于 Mimi Vocoder 的文本转语音（TTS），支持原生零样本语音克隆。支持直接生成原始音频 WAV、通过参考音频进行音色克隆，以及离线 Mimi 译码
- 🎤 **CustomVoice / VoiceDesign** — 命名说话人预设（vivian、ryan 等）快速选择音色，以及通过 VoiceDesign 文本描述音色风格进行创意 TTS 控制
- 🔉 **流式 TTS** — 实时原始 24kHz 单声道 f32 PCM 音频输出到 stdout，实现低延迟流式语音合成
- ⚡ **原生 bf16 GEMV kernel** — 手写优化的 SG=16 vec8+FMA bf16 GEMV 用于 TTS 解码，TTS 速度提升 9 倍（0.6B 模型 RTF 从 8.08 降至 0.89）
- 🔄 流式输出 — token 级别流式回调，支持中止生成
- **💬 交互式 CLI** — 多轮对话，支持运行时命令（`/clear`、`/greedy`、`/sample` 等）
- **📊 性能基准测试** — 分别测量 prefill 和 decode 吞吐量
- **🔌 C API** — 稳定的 C FFI 接口（Python、C#、Rust、Go、Java）— 见 [docs/C_API.md](docs/C_API.md)
- **💭 Chat Formatting** — llama.cpp 风格 Jinja 渲染、修复版 Qwen3.5 模板、结构化 reasoning/tool-call 解析，以及 JSONL chat 流事件

## 📦 支持的模型

| 模型 | 架构 | 量化 | 视觉 | 语音 (ASR / TTS) |
|------|------|------|------|------------------|
| [Qwen3.5-0.8B](https://huggingface.co/Blackwood416/Qwen3.5-0.8B-BNB-NF4-with-vision) | Hybrid (GQA + DeltaNet) | BNB NF4, dense | ✅ | ❌ |
| [Qwen3.5-4B](https://huggingface.co/Blackwood416/Qwen3.5-4B-BNB-NF4-with-vision) | Hybrid (GQA + DeltaNet) | BNB NF4, dense | ✅ | ❌ |
| [Qwen3-0.6B](https://huggingface.co/Blackwood416/Qwen3-0.6B-BNB-NF4) | Dense (GQA) | BNB NF4, dense | ❌ | ❌ |
| [Qwen3-4B](https://huggingface.co/Blackwood416/Qwen3-4B-BNB-NF4) | Dense (GQA) | BNB NF4, dense | ❌ | ❌ |
| [Qwen3-ASR-1.7B](https://huggingface.co/Blackwood416/Qwen3-ASR-0.6B-BNB-NF4) | Dense + Audio Encoder | BNB NF4, dense | ❌ | ✅ (ASR) |
| [Qwen3-ASR-1.7B](https://huggingface.co/Blackwood416/Qwen3-ASR-1.7B-BNB-NF4) | Dense + Audio Encoder | BNB NF4, dense | ❌ | ✅ (ASR) |
| [Qwen3-ForcedAligner-0.6B](https://huggingface.co/Blackwood416/Qwen3-ForceAligner-0.6B-BNB-NF4) | Dense + Audio Encoder | BNB NF4, dense | ❌ | ✅（对齐） |
| [Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) | Talker + Mimi Vocoder | BF16 | ❌ | ✅ TTS（语音克隆） |
| [Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | Talker + Mimi Vocoder | BF16 | ❌ | ✅ TTS（语音克隆） |
| [Qwen3-TTS-12Hz-0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) | Talker + Mimi Vocoder | BF16 | ❌ | ✅ TTS（预置音色） |
| [Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) | Talker + Mimi Vocoder | BF16 | ❌ | ✅ TTS（预置音色） |
| [Qwen3-TTS-12Hz-1.7B-VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) | Talker + Mimi Vocoder | BF16 | ❌ | ✅ TTS（指令设计） |
| [YOLO26 n/s/m/l/x](https://docs.ultralytics.com/models/yolo26/) | 目标检测、one-to-one 端到端 head | FP16 | ✅（检测） | ❌ |

其他符合支持架构模式的 Qwen3 / Qwen3.5 模型大小理论上也可运行。

YOLO26 权重需要先转换为 Aila 稳定格式，发布包不会附带这些资产。请参阅
[YOLO26 准备与验证指南](docs/YOLO26.md)。

## 🔧 系统要求

### 🖥️ 硬件
- **Intel Arc A770** (16 GB) — 主要开发和测试平台
- 其他 Intel Arc 独立显卡（A750、A580、A380、B580），≥8 GB 显存
- 集成显卡（Xe-LP、Xe-LPG）可能支持小模型，但未经测试

### 💿 操作系统
- **Windows 10 22H2** 或更高版本 / **Windows 11**

### 💻 软件
- [Intel Arc显卡驱动](https://www.intel.cn/content/www/cn/zh/products/docs/discrete-gpus/arc/software/drivers.html)
- `Aila-vX.Y.Z-win64.zip` 发行包已包含所有必需的运行时 DLL

## 📥 安装

1. 安装 **Intel Arc显卡驱动**。
2. 从 [Releases](https://github.com/Blackwood416/Aila/releases) 页面下载 `Aila-vX.Y.Z-win64.zip`。
3. 解压到任意目录。
4. 将模型文件放入目录中（如 `./models/qwen3.5-0.8B-bnb-nf4-offline/`）。

Windows 发行包采用拆分运行时布局：

```text
integration_root/
|-- AilaShared.dll
`-- aila_runtime/
    |-- AilaWorker.exe
    |-- Aila.exe
    `-- <oneAPI 及其他运行时 DLL>
```

`AilaShared.dll` 是不导入 oneAPI 推理运行时的 C ABI 转接层。每个成功初始化
的引擎都在独立的 `AilaWorker.exe` worker process（工作进程）中执行推理。
通过 C API 集成时，请在加载转接层前设置
`AILA_RUNTIME_DLL_DIR=aila_runtime`。相对路径以 `AilaShared.dll` 所在目录为
基准；也支持绝对路径，内部会统一规范化为绝对路径。如果变量未设置或为空，
转接层会使用自身所在目录，从而兼容旧的平铺布局。此回退只用于兼容旧部署，
不提供宿主 DLL 搜索隔离：平铺目录一旦暴露给宿主，转接层和私有运行时都会被
暴露。Python、ComfyUI 及其他嵌入式宿主必须使用拆分布局，并显式设置
`AILA_RUNTIME_DLL_DIR`。非 Windows 构建仍在进程内执行推理。

Python 示例：

```python
import ctypes
import os

os.environ["AILA_RUNTIME_DLL_DIR"] = "aila_runtime"
lib = ctypes.CDLL(r".\AilaShared.dll")
lib.aila_engine_create.restype = ctypes.c_void_p
engine = lib.aila_engine_create()
```

宿主的 DLL 搜索路径只能加入转接层所在目录。不要调用
`os.add_dll_directory("aila_runtime")`，也不要把 `aila_runtime` 加入宿主
`PATH`；否则会让 Python 看到 Aila 私有的 oneAPI DLL，破坏隔离。现有生成、
ASR、强制对齐以及 Aila 自有内存释放接口保持源码兼容。0.1.7 的 TTS 流式
接口有一项 ABI 变更：现在返回 `AilaTTSStream*`，wait/destroy 也改为接收该
stream handle；迁移方法见 [docs/C_API.md](docs/C_API.md)。

工作进程以 runtime 目录作为工作目录，并使用隔离的子进程 `PATH`。缺少
`AilaWorker.exe`、转接层与 worker build ID 不匹配、启动超时或工作进程退出
时，现有 C API 错误接口会返回诊断信息；失败操作不会自动重试。请始终部署
同一发行包中的 `AilaShared.dll` 与 `AilaWorker.exe`，不保证跨版本 worker
兼容。下方 CLI 示例假定从 `aila_runtime/` 中运行 `Aila.exe`，或使用其完整
路径。

ASR 和 TTS stream handle 会共享其引擎的所有权。销毁 `AilaEngine` 前，应先
销毁 ASR stream，并等待或取消后销毁 TTS stream；否则 worker 的关闭可能
延迟到最后一个 stream handle 被销毁。

## 📊 性能基准

基于 Intel Arc A770 16 GB，Qwen3.5-4B，pp=2048 tg=1024 测试：

| 引擎 | 后端 | 模型 | Prefill | Decode |
|------|------|------|---------|--------|
| **Aila 0.1.3** | SYCL + oneDNN | Qwen3.5-4B BNB NF4 | **1649 tok/s** | 58 tok/s |
| llama.cpp b8996 | SYCL | Qwen3.5-4B Q4_K_XL | 1290 tok/s | 28 tok/s |
| llama.cpp b8996 | Vulkan | Qwen3.5-4B Q4_K_XL | 700 tok/s | **60 tok/s** |

Aila 提供最高的 prefill 吞吐量，同时在保留视觉能力的 NF4 量化下实现接近 Vulkan 的 decode 性能。

基于 Intel Arc A770 16 GB，Qwen3-TTS-12Hz-0.6B-Base 的 TTS 性能测试：

| 引擎 | 模型 | RTF | 说明 |
|------|------|-----|------|
| **Aila** | Qwen3-TTS-12Hz-0.6B-Base | **0.89** | 原生 bf16 GEMV kernel |
| Aila（优化前） | Qwen3-TTS-12Hz-0.6B-Base | 8.08 | 仅 oneDNN matmul |

实时率（RTF）< 1 表示引擎合成语音的速度快于实时播放。原生 bf16 GEMV kernel 带来 9 倍的 TTS 速度提升。

## 🚀 使用方法

### ⌨️ CLI 快速开始

```powershell
# 交互式对话
Aila.exe -m ./models/Qwen3.5-4B-BNB-NF4-with-vision

# 离线语音转录 (ASR)
Aila.exe -m ./models/Qwen3-ASR-1.7B --transcribe input.wav

# YOLO26 检测（稳定单行 JSON，可选原尺寸标注 PNG）
Aila.exe -m ./models/yolo26/aila/n --detect image.jpg --conf 0.25 --max-det 300 --save-detect result.png

# 强制对齐（词级时间戳）
Aila.exe -m ./models/Qwen3-ForcedAligner-0.6B-BNB-NF4 --align-text "你好世界" --align-audio input.wav --align-lang Chinese

# 文本转语音合成 (TTS)
Aila.exe -m ./models/Qwen3-TTS-12Hz-0.6B-Base --synthesize "你好世界" --output-wav output.wav

# TTS 语音克隆（零样本）
Aila.exe -m ./models/Qwen3-TTS-12Hz-0.6B-Base --synthesize "今天天气真好。" --ref ./reference_speaker.wav --output-wav cloned.wav

# 从 JSON 文件读取单条 prompt
Aila.exe -m ./models/Qwen3.5-4B-BNB-NF4-with-vision --messages-json prompt.json

# 从 stdin 读取 prompt
echo '{"messages":[{"role":"user","content":"你好"}]}' | Aila.exe -m ./models/qwen3.5-0.8B-bnb-nf4-offline --messages-json -

# 性能基准测试（贪心解码）
Aila.exe -m ./models/Qwen3.5-4B-BNB-NF4-with-vision --bench --bench-pp 2048 --bench-tg 1024 --bench-iters 3

# 性能基准测试（采样解码）
Aila.exe -m ./models/Qwen3.5-4B-BNB-NF4-with-vision --bench --sample
```

### ⚙️ CLI 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-m, --model <path>` | 模型目录 | `AILA_MODEL_DIR` 环境变量 |
| `-s, --max-seq <N>` | 最大序列长度 | 4096 |
| `-t, --temperature <F>` | 采样温度 | 0.7 |
| `-k, --top-k <N>` | Top-K 采样 | 15 |
| `-p, --top-p <F>` | Top-P (nucleus) 采样 | 0.95 |
| `--seed <N>` | 采样随机种子 | 无 |
| `--greedy` / `--sample` | 解码模式 | sample |
| `--stream` / `--no-stream` | 强制流式输出开/关 | 自动 |
| `--max-tokens <N>` | 最大生成 token 数 | 1024 |
| `--thinking-budget <N>` | 思考预算：-1 关闭，0 禁止思考，>0 限制 token | -1 |
| `--decode-chunk <N>` | 解码块大小 | 12 |
| `--stream-chunk <N>` | 流式块大小 | 4 |
| `--rep-penalty <F>` | 重复惩罚 | 1.0 |
| `--pres-penalty <F>` | 存在惩罚 | 0.0 |
| `--freq-penalty <F>` | 频率惩罚 | 0.0 |
| `--detect <image>` | 对单张 JPEG/PNG 执行 YOLO26 检测 | 关闭 |
| `--conf <F>` | `[0,1]` 内的检测置信度 | 0.25 |
| `--max-det <N>` | `[1,300]` 内的最大检测数 | 300 |
| `--save-detect <png>` | 保存原尺寸标注 PNG | 关闭 |
| `--bench` | 基准测试模式 | 关闭 |
| `--bench-pp <N>` | 基准测试 prompt 长度 | 512 |
| `--bench-tg <N>` | 基准测试生成长度 | 128 |
| `--bench-iters <N>` | 基准测试迭代次数 | 5 |
| `--bench-warmup <N>` | 基准测试预热次数 | 1 |
| `--bench-sample` / `--bench-greedy` | 基准测试解码模式 | greedy |
| `--log-level <level>` | 最低日志级别（verbose/debug/info/warning/error） | info |
| `--messages-json <path>` | JSON prompt 文件（`-` = stdin） | 无 |
| `--chat-output-json` | 与 `--messages-json` 配合，输出结构化 assistant JSON 而非原始文本 | 关闭 |
| `--chat-stream-jsonl` | 与 `--messages-json` 配合，以 JSONL 输出结构化流事件 | 关闭 |
| `--lora <path>` | LoRA adapter 目录 | `AILA_LORA_DIR` 环境变量 |
| `--transcribe <path>` | 语音 WAV 音频转录模式 | 无 |
| `--synthesize <text>` | TTS 文本转语音合成 | 无 |
| `--output-wav <path>` | TTS 输出 WAV 文件路径 | `output.wav` |
| `--ref <path>` | TTS 语音克隆参考音频 | 无 |
| `--speaker <name>` | TTS 命名音色（如 vivian、ryan） | 无 |
| `--instruct <text>` | VoiceDesign 音色风格描述（如 "深沉温暖的声音"） | 无 |
| `--language <lang>` | TTS 语言：chinese、english、japanese、korean、auto | auto |
| `--ref-cache-dir <dir>` | 说话人嵌入缓存目录 | `AILA_REF_CACHE_DIR` 环境变量 |
| `--stream-tts` | 流式输出原始 24kHz 单声道 f32 PCM 到 stdout | 关闭 |
| `--stream-batch <N>` | 流式 TTS 每批帧数 | 4 |
| `--align-text <text>` | 强制对齐：要对齐的文本 | 无 |
| `--align-audio <path>` | 强制对齐：音频文件路径 | 无 |
| `--align-lang <lang>` | 强制对齐：语言（默认 Chinese） | Chinese |
| `--forced-lang <lang>` | 强制指定 ASR 转录语种（如 Chinese, English） | 无 |
| `--asr-system <prompt>` | ASR 偏置的系统提示词（system prompt） | 无 |
| `--asr-segment <sec>` | ASR 音频分段切片秒数大小 | 0.0 (禁用) |
| `--asr-past` / `--no-asr-past` | 启用/禁用 ASR 分段上文偏置（past-text conditioning） | no-asr-past |
| `--q35-prefill-step <N>` | Qwen3.5 循环状态 checkpoint 间隔 | 64 |
| `--kv-quant` | 启用 FP8 E4M3 KV cache 量化 | 关闭 |
| `-h, --help` | 显示帮助 | — |
| `-v, --version` | 显示版本 | — |

### 🎮 交互命令

| 命令 | 说明 |
|------|------|
| `/help` | 显示可用命令 |
| `/quit`、`/exit` | 退出程序 |
| `/transcribe <path>` | 转录音频文件（ASR） |
| `/align text="..." audio=<path> [language="..."]` | 强制对齐（词级时间戳） |
| `/tts <text> [--ref <path>]` | 语音合成（TTS），支持可选语音克隆 |
| `/synthesize <text> [--ref <path>]` | `/tts` 的别名 |
| `/voice <name>` | 设置 TTS 音色（vivian、ryan 等） |
| `/clear` | 清除对话历史 |
| `/context` | 显示上下文用量 |
| `/greedy` | 切换到贪心解码 |
| `/sample` | 切换到采样解码 |
| `/seed <N>` | 设置采样种子 |
| `/stream_on` / `/stream_off` | 切换流式输出 |
| `/decode_chunk <N>` | 设置解码块大小 |
| `/stream_chunk <N>` | 设置流式块大小 |
| `/thinking_budget <N|off>` | 设置思考预算 |
| `/log_level <level>` | 设置日志级别（verbose/debug/info/warning/error） |
| `/config` | 显示当前配置 |

### 💭 Chat Formatting 与工具调用

`--messages-json` 接受 OpenAI 风格 chat request，支持 `messages`、`tools`、
`tool_choice`、`chat_template_kwargs` 和生成参数。Qwen3.5 Hybrid 模型在未显式
覆盖模板时会使用 Aila 内置的修复版 Qwen3.5 Jinja 模板。

默认情况下，`--messages-json` 输出原始 assistant 文本。添加
`--chat-output-json` 后会输出结构化 assistant JSON，包括 `content`、
`reasoning_content`、`tool_calls`、`raw_text`、`finish_reason`、`warnings` 和
`metadata`。
也可以改用 `--chat-stream-jsonl`，以 newline-delimited JSON 输出结构化流事件。

Aila 只负责格式化 prompt 并解析模型输出的工具调用，不在推理引擎内部执行工具。
调用方应在外部执行返回的 `tool_calls`，再把工具结果作为 `tool` 消息传回 Aila。
`tool_policy` 支持 `"warn"` 和 `"strict"`；strict 会用
`finish_reason: "tool_policy"` 标记策略违背。reasoning 可通过
`reasoning_budget` / `--thinking-budget` 限制。C API 调用方可使用
`aila_generate_chat_json_ex` 和 `aila_generate_chat_json_stream_ex` 搭配
`AilaGenConfigV2` 获取 ABI-safe 的 chat 选项。
流式调用方可以把带有 `finish_reason: "tool_calls"` 的 final JSONL / C API
事件作为工具执行交接点，然后发起下一轮请求。

### 🎤 TTS 语音克隆

Aila 为 Qwen3-TTS 模型提供**零样本语音克隆**支持。说话人嵌入通过原生 C++ ECAPA-TDNN 编码器提取（不依赖 Python）。

**参考音频要求：**

| 要求 | 说明 |
|------|------|
| **格式** | WAV、MP3 或 FLAC |
| **采样率** | 任意（自动重采样到 24kHz） |
| **声道** | 推荐单声道（多声道自动平均为单声道） |
| **时长** | 建议 ≥ 3 秒清晰语音 |
| **内容** | 目标说话人的清晰语音，背景噪音尽量小 |
| **编码** | 16-bit PCM 或 32-bit float |

```powershell
# 从参考音频克隆音色
Aila.exe -m ./models/Qwen3-TTS-12Hz-0.6B-Base `
    --synthesize "你好世界" `
    --ref ./reference_speaker.wav `
    --output-wav cloned_output.wav

# CustomVoice — 使用命名说话人预设
Aila.exe -m ./models/Qwen3-TTS-12Hz-0.6B-CustomVoice `
    --speaker vivian `
    --synthesize "你好世界" `
    --output-wav vivian_output.wav

# VoiceDesign — 用自然语言描述音色风格
Aila.exe -m ./models/Qwen3-TTS-12Hz-0.6B-Base `
    --instruct "深沉温暖的声音，语速较慢" `
    --synthesize "你好世界" `
    --output-wav styled_output.wav

# VoiceDesign 配合语言指定
Aila.exe -m ./models/Qwen3-TTS-12Hz-0.6B-CustomVoice `
    --speaker ryan `
    --language english `
    --instruct "whispering softly" `
    --synthesize "Hello" `
    --output-wav ryan_english.wav

# 流式 TTS — 实时 PCM 音频输出
Aila.exe -m ./models/Qwen3-TTS-12Hz-0.6B-Base `
    --stream-tts --synthesize "你好！" 2>/dev/null | pcm_play

# 流式 TTS 自定义批大小
Aila.exe -m ./models/Qwen3-TTS-12Hz-0.6B-Base `
    --stream-tts --stream-batch 8 --synthesize "你好！" 2>/dev/null | pcm_play
```

`--rep-penalty` 参数控制重复惩罚（TTS 模式下自动设为 1.1）。如果输出出现重复伪影可调高（如 `--rep-penalty 1.3`），需要更多变化时可调低（如 `--rep-penalty 1.0`）。

说话人嵌入会**自动缓存**：会话期间缓存于内存中，并持久化到磁盘（`<audio_path>.ref.bin`），避免重复提取。使用 `--ref-cache-dir <dir>` 或 `AILA_REF_CACHE_DIR` 环境变量可将缓存文件集中存储到指定目录。

### 📄 Messages JSON 格式

支持标准 OpenAI 兼容的 JSON 对象，以 `"messages"` 作为顶层键，同时也允许直接传递采样/生成参数（例如 `temperature`, `max_tokens`/`max_new_tokens`, `seed`/`do_sample` 等）：

```json
{
  "messages": [
    {"role": "system", "content": "你是一个简洁的助手。"},
    {"role": "user",   "content": [{"type": "text", "text": "介绍你自己"}]}
  ],
  "temperature": 0.7,
  "max_tokens": 128
}
```

同时也向下兼容直接以 JSON 数组作为顶层对象的格式：`[{"role": "user", "content": "..."}]`。

支持 `text`、`image`、`audio` 和 `video` 内容类型。图像部分支持 `image`、`image_url`（支持 base64 Data URI 或本地文件路径）或 `{"image_url":{"url":"..."}}` 格式。音频部分支持 `audio`、`audio_url`（支持 base64 Data URI 或本地文件路径）或 `{"input_audio": {"data": "base64", "format": "wav"}}` 格式。

### 🔌 C API

完整 C API 文档见 **[docs/C_API.md](docs/C_API.md)**（支持 Python ctypes、C# P/Invoke、Rust FFI 等）。

### 🌐 环境变量

完整环境变量列表见 **[docs/Environment_Variables.md](docs/Environment_Variables.md)**。

## 📦 模型导出

使用 `export_bnb_nf4.py` 将 Hugging Face 模型量化为 BNB NF4 格式：

```powershell
# 纯文本模型
python export_bnb_nf4.py \
    --source Qwen/Qwen3.5-0.8B \
    --output ./models/Qwen3.5-0.8B-BNB-NF4

# 视觉模型
python export_bnb_nf4.py \
    --source Qwen/Qwen3.5-4B \
    --output ./models/Qwen3.5-4B-BNB-NF4-with-vision \
    --vision

# ASR模型
python export_bnb_nf4.py \
    --source Qwen/Qwen3-ASR-1.7B \
    --output ./models/Qwen3-ASR-1.7B-BNB-NF4

# ForceAligner 模型（自动检测，classify_head 保持密集）
python export_bnb_nf4.py \
    --source Qwen/Qwen3-ForcedAligner-0.6B \
    --output ./models/Qwen3-ForcedAligner-0.6B-BNB-NF4

# 从本地目录导出，覆盖已有导出
python export_bnb_nf4.py \
    --source ./Qwen3-0.6B \
    --output ./models/Qwen3-0.6B-BNB-NF4 \
    --overwrite
```

需要：`torch`、`transformers`、`bitsandbytes`（Intel XPU 后端）。

## 🛠️ 从源码构建

```powershell
# 需要：Intel oneAPI Base Toolkit 2026.1+、CMake 3.24+、Ninja
.\build.ps1

# 清理构建
.\build.ps1 -Clean

# Debug 构建
.\build.ps1 -Config Debug
```

输出：
| 文件 | 说明 |
|------|------|
| `build/Aila.exe` | CLI 可执行文件 |
| `build/AilaShared.dll` | 动态链接库（C API） |
| `build/AilaWorker.exe` | Windows 隔离推理工作进程 |
| `build/AilaLib.lib` | 静态库 |
| `build/build_info.json` | 本地构建与 oneAPI 工具链来源信息 |

在 Windows 上，`cmake --build build --target release` 会把集成布局 staging
到 `build/Release/bin`：根目录只包含转接层，CLI、工作进程与运行时 DLL 位于
`aila_runtime/`。构建来源信息仍保存在 `build/build_info.json`，不属于发行包。

## 📁 项目结构

```
Aila/
├── include/
│   ├── aila_api.h              # 公共 C API 头文件
│   └── engine/Engine.hpp       # InferenceEngine 类
├── src/
│   ├── main.cpp                 # CLI 入口
│   ├── api/aila_api.cpp         # C API 实现
│   ├── cli/                     # CLI 参数解析和交互循环
│   ├── core/                    # SYCL 上下文和张量管理
│   ├── memory/                  # KV 缓存
│   ├── models/                  # 模型后端（Qwen3、Qwen3.5、BNB4）
│   ├── ops/                     # SYCL kernel（注意力、RMSNorm、Bnb4BitLinear 等）
│   ├── profile/                 # 日志、性能分析和设备信息
│   ├── audio/                   # 音频预处理（ASR）+ 说话人编码器（TTS 语音克隆）
│   ├── utils/                   # 分词器、SafeTensors、内存映射 I/O
│   └── vision/                  # 视觉编码器（Qwen3.5）
├── docs/
│   ├── C_API.md                 # C API 文档
│   └── Environment_Variables.md # 环境变量参考
├── third_party/simdjson/        # JSON 解析
├── build.ps1                    # 构建脚本
├── bench.ps1                    # 基准测试脚本
├── smoke.ps1                    # 冒烟测试脚本
└── CMakeLists.txt
```

---

## 🙏 致谢

- **[oneDNN](https://github.com/oneapi-src/oneDNN)** — Intel 深度神经网络库，提供 bf16 推理所用的矩阵乘法原语
- **[bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)** — NF4 量化格式和反量化参考实现
- **[simdjson](https://github.com/simdjson/simdjson)** — 高性能 JSON 解析器，用于模型配置和分词器元数据
- **[dr_libs](https://github.com/mackron/dr_libs)** — 单文件头音频解码库（`dr_wav`、`dr_mp3`、`dr_flac`），用于 ASR 音频预处理
- **[llama.cpp jinja module](https://github.com/ggml-org/llama.cpp/tree/master/common/jinja)** — jinja对话模板解析模块，用于对话模板解析和构建

## 📄 许可证

详见 [LICENSE](LICENSE)。
