# Aila

<p align="center">
  <b>充分发挥 Arc 显卡性能的推理引擎。</b><br>
  <a href="README.md">English</a>
</p>

---

> [!NOTE]
> 该项目仍在积极开发中，并不能兼顾支持的所有模型。目前的重点在 **Qwen3.5** 模型上，Qwen3 模型的性能可能并不理想。

基于 **SYCL + oneDNN** 构建的高性能 LLM 推理引擎，专为 **Intel Arc 显卡** 设计。针对 bitsandbytes 4-bit (NF4) 量化模型提供手写优化 kernel，包括融合反量化+矩阵乘法、GEMV 解码，以及 Qwen3.5 混合架构的 GPU DeltaNet 循环加速。

## ✨ 功能特性

- **⚡ Bitsandbytes 4-bit (NF4) 推理** — 融合反量化+矩阵乘 kernel、手写 GEMV 解码、融合 gate+up+SiLU 投影，直接在 Intel Arc 上运行量化模型
- **🔢 Bfloat16 推理** — 通过 oneDNN 矩阵乘法原语支持密集（非量化）模型
- **🏗️ Qwen3.5 Hybrid 架构** — 完整支持双重注意力（GQA + DeltaNet 线性注意力），GPU 加速 delta 循环计算
- **📐 Qwen3 Dense 架构** — 标准 Transformer，支持 GQA、QK-norm 和 SwiGLU FFN
- **👁️ 视觉理解 (Qwen3.5)** — 支持图像输入，CPU 预处理 + GPU 视觉 Transformer
- **🔄 流式输出** — token 级别流式回调，支持中止生成
- **💬 交互式 CLI** — 多轮对话，支持运行时命令（`/clear`、`/greedy`、`/sample` 等）
- **📊 性能基准测试** — 分别测量 prefill 和 decode 吞吐量
- **🔌 C API** — 稳定的 C FFI 接口（Python、C#、Rust、Go、Java）— 见 [docs/C_API.md](docs/C_API.md)
- **💭 Chat 模板** — ChatML 格式，支持 `<think>` 块生成和 `/no_think` 抑制

## 📦 支持的模型

| 模型 | 架构 | 量化 | 视觉 |
|------|------|------|------|
| [Qwen3.5-0.8B](https://huggingface.co/Blackwood416/Qwen3.5-0.8B-BNB-NF4-with-vision) | Hybrid (GQA + DeltaNet) | BNB NF4, dense | ✅ |
| [Qwen3.5-4B](https://huggingface.co/Blackwood416/Qwen3.5-4B-BNB-NF4-with-vision) | Hybrid (GQA + DeltaNet) | BNB NF4, dense | ✅ |
| [Qwen3-0.6B](https://huggingface.co/Blackwood416/Qwen3-0.6B-BNB-NF4) | Dense (GQA) | BNB NF4, dense | ❌ |
| [Qwen3-4B](https://huggingface.co/Blackwood416/Qwen3-4B-BNB-NF4) | Dense (GQA) | BNB NF4, dense | ❌ |

其他符合支持架构模式的 Qwen3 / Qwen3.5 模型大小理论上也可运行。

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
2. 从 [Releases](https://github.com/Blackwood416/releases) 页面下载 `Aila-vX.Y.Z-win64.zip`。
3. 解压到任意目录。
5. 将模型文件放入目录中（如 `./models/qwen3.5-0.8B-bnb-nf4-offline/`）。

## 📊 性能基准

基于 Intel Arc A770 16 GB，Qwen3.5-4B，pp=2048 tg=1024 测试：

| 引擎 | 后端 | 模型 | Prefill | Decode |
|------|------|------|---------|--------|
| **Aila 0.1.0** | SYCL + oneDNN | Qwen3.5-4B BNB NF4 | **1600 tok/s** | 50 tok/s |
| llama.cpp b8996 | SYCL | Qwen3.5-4B Q4_K_XL | 1290 tok/s | 28 tok/s |
| llama.cpp b8996 | Vulkan | Qwen3.5-4B Q4_K_XL | 700 tok/s | **60 tok/s** |

Aila 提供最高的 prefill 吞吐量，同时在保留视觉能力的 NF4 量化下实现接近 Vulkan 的 decode 性能。

## 🚀 使用方法

### ⌨️ CLI 快速开始

```powershell
# 交互式对话
Aila.exe -m ./models/qwen3.5-0.8B-bnb-nf4-offline

# 从 JSON 文件读取单条 prompt
Aila.exe -m ./models/qwen3.5-0.8B-bnb-nf4-offline --messages-json prompt.json

# 从 stdin 读取 prompt
echo '{"messages":[{"role":"user","content":"你好"}]}' | Aila.exe -m ./models/qwen3.5-0.8B-bnb-nf4-offline --messages-json -

# 性能基准测试（贪心解码）
Aila.exe -m ./models/qwen3.5-0.8B-bnb-nf4-offline --bench --bench-pp 512 --bench-tg 128

# 性能基准测试（采样解码）
Aila.exe -m ./models/qwen3.5-0.8B-bnb-nf4-offline --bench --sample
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
| `--decode-chunk <N>` | 解码块大小 | 12 |
| `--stream-chunk <N>` | 流式块大小 | 4 |
| `--rep-penalty <F>` | 重复惩罚 | 1.0 |
| `--pres-penalty <F>` | 存在惩罚 | 0.0 |
| `--freq-penalty <F>` | 频率惩罚 | 0.0 |
| `--bench` | 基准测试模式 | 关闭 |
| `--bench-pp <N>` | 基准测试 prompt 长度 | 512 |
| `--bench-tg <N>` | 基准测试生成长度 | 128 |
| `--bench-iters <N>` | 基准测试迭代次数 | 5 |
| `--bench-warmup <N>` | 基准测试预热次数 | 1 |
| `--bench-sample` / `--bench-greedy` | 基准测试解码模式 | greedy |
| `--messages-json <path>` | JSON prompt 文件（`-` = stdin） | 无 |
| `-h, --help` | 显示帮助 | — |
| `-v, --version` | 显示版本 | — |

### 🎮 交互命令

| 命令 | 说明 |
|------|------|
| `/help` | 显示可用命令 |
| `/quit`、`/exit` | 退出程序 |
| `/clear` | 清除对话历史 |
| `/context` | 显示上下文用量 |
| `/greedy` | 切换到贪心解码 |
| `/sample` | 切换到采样解码 |
| `/seed <N>` | 设置采样种子 |
| `/stream_on` / `/stream_off` | 切换流式输出 |
| `/decode_chunk <N>` | 设置解码块大小 |
| `/stream_chunk <N>` | 设置流式块大小 |
| `/config` | 显示当前配置 |

### 🤫 `/no_think` 与 `/think` 后缀

在消息末尾添加 `/no_think` 可抑制模型的思考过程，添加 `/think` 可强制思考（对默认为非思考模式的 Qwen3.5-0.8B 有用）：

```
User: 1+1等于几？ /no_think
Aila: 1+1等于2。

User: 解释量子计算。 /think
Aila: <think>
让我逐步分析...
</think>
量子计算是...
```

交互模式和 `--messages-json` 均支持。

### 📄 Messages JSON 格式

```json
[
  {"role": "system", "content": "你是一个简洁的助手。"},
  {"role": "user",   "content": [{"type": "text", "text": "介绍你自己"}]}
]
```

支持 `text`、`image` 和 `video` 内容类型。图像部分支持 `image`、`image_url` 或 `{"image_url":{"url":"..."}}` 格式。

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
    --output ./models/qwen3.5-0.8B-bnb-nf4-offline

# 视觉模型
python export_bnb_nf4.py \
    --source Qwen/Qwen3.5-4B \
    --output ./models/qwen3.5-4B-bnb-nf4-vision-offline \
    --vision

# 从本地目录导出，覆盖已有导出
python export_bnb_nf4.py \
    --source ./Qwen3-0.6B \
    --output ./models/qwen3-0.6B-bnb-nf4-offline \
    --overwrite
```

需要：`torch`、`transformers`、`bitsandbytes`（Intel XPU 后端）。

## 🛠️ 从源码构建

```powershell
# 需要：Intel oneAPI Base Toolkit、CMake 3.24+、Ninja
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
| `build/AilaLib.lib` | 静态库 |

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

## 📄 许可证

详见 [LICENSE](LICENSE)。
