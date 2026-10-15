# Qwen3.8-Flash-Next 优化工作交接

> 接手更新：本文的 `88be93c` 为历史基点。nested-256 已保留为 `98cc037`，H4 初始驻留接入为 `c5d5805`，staging 并行拷贝为当前 `3962519`；官方 Worker/Server 双构建与检查已完成。最新状态、生产测量和后续隔离方案见 [专家供给优化交接](qwen38-expert-supply-handoff-2026-10-09.md)。

交接日期：2026-10-09。来源 thread：`01a10bdd-b9a5-75c0-bc44-107fb2e66f45`，其前序 thread 为 `01a1009d-05d1-7cb3-8dd4-64d41e3dd205`。

代码快照：`main@88be93cf79cc95e4efec1ed95ba0853000d23d48`，项目目录 `E:/RiderProjects/Aila`。整理交接前工作区干净，没有待接续的未提交运行时代码。本文根据原 thread、本地源码、Git 提交、报告 JSON 和原始结果整理；本次交接没有重新编译或执行 GPU 性能实验。

## 1. 接手时先知道这六件事

1. **当前目标是这个模型在本机上的真实 agent 单流长文、多轮体验。** 长输入到工具任务完成的总耗时是主指标；decode 20+ 继续追求，但不能替代整轮体验。暂不做并发推理。
2. **主体专家已经改成 GPU 与原生大页 RAM 的互补所有权。** 当前专家运行路径没有磁盘读取后备；PLE 约 29.82 GiB 仍按需从文件取数。继续优化时，先区分 PLE 读取、RAM staging、H2D、GPU 消费和 CPU 计算。
3. **最后保留的优化是跨 prompt chunk 提前准备 PLE。** `[MEASURED]` 两对交错请求中，14K 暖态长请求完整时间中位数 `102.23 → 74.80 s`，缩短 26.8%；真实 PI 工具回合在两对内部各缩短 12.0% / 10.3%。
4. **当前 Server decode 仍约 13 tok/s，启动仍约 130 秒。** 历史 MTP 探针的 20.5929 tok/s 属于旧执行路径、短上下文和特定提示；它不代表当前 32K Server 的成绩。
5. **新驻留路径绕过了部分旧优化分支。** 启动 JSON 中有旧 profile/async/hotset 参数，不代表它们都在当前路径生效。尤其不能只改环境变量，就认为已经恢复旧的多行 CPU verifier 或 learned hotset。
6. **下一轮只选一个有证据的候选。** 先做小范围筛选，有整轮收益才补反序复测和相关功能检查；失败归档撤回，不积累默认关闭的参数，不反复跑全套测试。

建议先读本文第 4、6、10、12 节，再读 [最新 PLE 对照报告](qwen38-ple-lookahead-2026-10-09.md) 和 [驻留重构修复报告](qwen38-strata-refactor-repair-2026-10-09.md)。早期报告用于理解历史，不能覆盖当前源码事实。

本文的证据标记：`[MEASURED]` 是指定配置的已有测量；`[CORRECTNESS]` 是所有权、状态或数值约束；`[TOOLCHAIN]` 是本机工具链事实；`[HYPOTHESIS]` 是还没有收益证明的候选；`[HEURISTIC]` 是建议的起点，不是通用 dispatch 规则。

## 2. 用户目标与已明确的工作约束

原 thread 的要求先后发生了以下变化，接手应遵从后期的具体要求：

| 时间 | 用户要求 | 对后续工作的影响 |
|---|---|---|
| 10-05 | 探索 decode 20+，不要过度设计和测试，高效迭代 | 小实验、明确否决、及时收敛 |
| 10-05 / 10-06 | 深挖 MTP；可以尝试单字节 KV | 两者已尝试；现有负结果见第 9、10 节 |
| 10-07 | 让真实 agent，例如 PI，接入并支持长上下文 | 保留 HTTP/SSE、工具往返、精确前缀复用的功能闭环 |
| 10-07 | 继续以 agent 单流长文/多轮为目标，暂不接并发 | `sessions=1`、`decode-batch-size=1` 是当前测量范围 |
| 10-07 | 先定位实际瓶颈；之后按规划只验证一个调度假设 | 不再同时扫线程、chunk、缓存和精度开关 |
| 10-08 | 充分利用 16G 显存和 64G RAM，减少硬盘参与 | 主体驻留方向已实现，不能继续把 H2D 字节当成 SSD 字节 |
| 10-08 | “不需要向前兼容性……当前目标就是为了把 Qwen3.8-Flash-Next 这个模型跑得越快越好……底层也可以改动” | 不必为旧模型兼容性额外设计；用户会安排原有模型的兼容工作 |
| 10-09 | 接手其他 agent 的 Strata 风格未完成重构，修好后继续提速 | 重构已修好；后续从 `88be93c` 的可运行状态继续 |

兼容性要求放宽不等于可以忽略取消排空、实际数学语义、资源释放或测试结果。用户要求的是高效完成优化，不是凭内核名称或局部速度宣称收益。

10-08 用户还明确告知：已关闭 Windows 内存压缩、固定虚拟内存容量，并给当前用户锁定内存页权限。这些是原 thread 中的环境说明；本次交接没有重新修改系统设置。当前 CPU 的 18 核 / 18 逻辑处理器和显卡驱动已重新只读核实。历史 HT-on/off 没有构成严格配对实验，不宣称关闭 HT 本身已经证明提速。

## 3. 当前仓库、设备与二进制身份

### 3.1 提交边界

| 提交 | 含义 | 接手判断 |
|---|---|---|
| `f5a1ece` | 实验性 Server MTP、decode 优化及相关接入 | 有价值的历史实现；不是当前长上下文默认路径 |
| `78806f1` | 大页互补驻留与单槽 agent 服务交接 | 驻留方向成立；运行期读盘明显降低，启动成本增加 |
| `35f750a` | 缩小大页 prefill 借用区，实验 staging 流水线 | 历史路径；其流水线没有整轮收益，未默认启用 |
| `a83cd80` | 完成其他 agent 的 adaptive resident 重构修复 | 功能基点：真实模型、PI、取消和 14K 续轮恢复 |
| **`88be93c`** | **跨块提前准备 PLE** | **当前接手续写基点** |

其他 agent 的原交接文件为：

`C:/Users/Administrator/.gemini/antigravity/brain/da5f320e-424f-4bba-8570-1775f17996a4/HANDOVER_QWEN4EXP_REFACTOR.md`

**该文件的若干成功判断和根因推测已被后续实跑修正。** 它声称整套 Host USM 可物理驻留、跳过 PLE 可接受，并怀疑 2-token 共享 down 缓冲；接手修复实际发现内存最低仅剩约 0.06 GiB，并恢复了完整 PLE。以 `a83cd80` 报告和当前代码为准，原文件只保留为故障背景。

### 3.2 已核对的本机环境

| 项目 | 当前事实 / 最后测量记录 |
|---|---|
| OS | Windows 11 专业版，`10.0.26200` |
| GPU | Intel Arc A770，16 GiB；Windows 驱动 **`32.0.101.8724`** |
| CPU | Xeon E5-2696 v3，**18 physical cores / 18 logical processors** |
| RAM | 约 63.91 GiB OS 可见内存，标称 64 GiB |
| 编译器 | Intel oneAPI DPC++ `2026.1.0 Build 20260617`，`icx-cl` |
| oneDNN | `3.11.2`；安装根 `C:/Program Files (x86)/Intel/oneAPI/dnnl/2026.0` |
| oneTBB | `2023.1` |
| Server 编译器 | 本机 `Visual Studio/18/Community`，MSVC `14.50.35717` |
| 模型 | `C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2` |
| Strata 参考源码 | `tmp/Strata/`，本机目录仍存在 |
| 调试器 | `D:/Windows Kits/10/Debuggers/x64/cdb.exe` |

`[TOOLCHAIN]` 根 AGENTS.md 的工具链概述与实际机器版本可能不同，以构建元数据和实查为准。早期 10-05 和部分 10-06 报告使用驱动 `32.0.101.9033`；不要将其速度与 8724 的差异归因于代码。

交接时没有发现运行中的 `Aila` / `AilaWorker` / `AilaServer` 进程。后续 GPU 实验仍一次只跑一个压力目标，编译也不要与正式性能测量重叠。

### 3.3 本机可直接使用的最终二进制

下列 SHA256 在本次交接时重新读取，和最后取消验证记录相同：

```text
build-refactor/AilaWorker.exe
8ce1efd990cb64332ebde8421989160e18a86810e916d88eed005c530d2a653d

build-refactor-server/AilaServer.exe
020c8a89cef7146999d1bb5de2e680f84b8a74bf4f891bf3f30b4835e48662e9
```

`[TOOLCHAIN]` **源码 HEAD 是 `88be93c`，本机最后验证二进制的构建标识仍是 `a83cd80`。** 原因是修改在该基点上构建、验证后才提交；不能因此把二进制误认成“未包含 PLE lookahead 的旧版”。最终 Worker hash 才是其实际身份。

`build-refactor/build_info.json` 与 `build-refactor-server/server-build-info.json` 的 Worker build ID 当前匹配：`aila-0.2.0-abi1-a83cd80a421fbe3d3dc789bcd6fe911445a9b9d5`。Server 的构建报告中记录的 Worker executable hash 是当时较早的构建，不等于后来重编的最终 Worker hash；区分握手 build ID、当前文件 hash 和历史报告 hash。

**重新构建后配齐 Server，不能只改 build_info.json 来绕过握手。** 已经两次因旧 Server 与新 Worker 构建标识不一致浪费了完整加载尝试。见第 12 节。

## 4. 当前实际执行模型，及与旧报告的差异

### 4.1 服务调用链

```text
PI / HTTP OpenAI-compatible client
  → AilaServer（纯 MSVC，模板、tokenizer、tools、SSE、prefix reuse）
  → InferenceClient / IPC v2 named pipe
  → AilaWorker / LoadedModel / Request / Session
  → Qwen4ExpBnb4Backend::ForwardImpl
  → adaptive GPU/RAM resident experts + PLE + KV/GDN/HC
```

Server/IPC 目前已有实用的功能闭环：真实 PI 的 `store:false`、开头 `developer → system`、工具消息、长 body、长 context、SSE usage、单槽连续工具轮、精确前缀复用。不要因为旧文件中有“最小接入计划”而重新实现一次。

`[CORRECTNESS]` 前缀复用只在完整已缓存 token 序列逐 token 相等时成立，不是任意语义相似、任意位置的 GDN rollback。usage 的 prompt 数是完整 prompt；复用量单独看 `prompt_tokens_details.cached_tokens`。失败或取消的 session 不应当作为正常前缀缓存回池。

当前实测配置为容量 32768、chunk4096；实际最长正式性能输入约 14K。协议能协商更大的上限，不代表完整 32K/64K 输入已经验收或有性能保证。`extended_context_v1` 当前 prefill chunk 上限为 4096；旧报告的 2048 是历史值。通用 Server 默认 chunk512 没有改成4096，本机 launcher 显式覆盖。

### 4.2 GPU / RAM 的互补所有权

当前 `Qwen4ExpBnb4Backend.hpp` 中 `use_adaptive_pipeline_` 是 **`const bool = true`**。普通加载直接初始化 `Qwen4ExpAdaptiveCache + Qwen4ExpHostArena`；不是通过打开旧 `ADAPTIVE_PIPELINE` 或 `LARGE_PAGE_RESIDENT` 环境开关才启用。

加载顺序与容量逻辑：

1. 加载 dense/shared projections、head、PLE 索引与基础工作区。
2. 将 host embedding 优先复制到原生大页，避免大块连续分配被后续专家块碎片化。
3. 扣除已分配设备内存、会话状态与 `VRAM_RESERVE_MB=3584` 余量。
4. GPU resident 专家槽数最多 2800，并扣除双 staging bank 的 32 槽；长容量最终槽数要按准入结果，**不能把 4K CLI 的 2800 当作 32K 必然槽数**。
5. 建立唯一的初始 GPU expert key 集合，再只为其余专家分配 RAM arena。
6. RAM 以原生 `MEM_LARGE_PAGES`、完整专家为单位、每块最多 256 MiB 分配。每层加载后回收普通 mmap 工作集；大页权重不受此回收影响。
7. runtime staging 使用两组小 Host USM，每组最多 16 专家。原生大页指针是 CPU 权重存储，不把约 50+ GiB 原生 RAM 全部注册成 Host USM 给 Level Zero。

`[CORRECTNESS]` 每个专家在 GPU 或 RAM 有一个权重所有者；临时 staging 是受事件约束的副本。GPU 热专家不能在 RAM arena 再常驻一份。换入时先保存 incoming H2D 源，再将 victim 唯一 GPU payload 回写到 incoming 腾出的 RAM 槽，最后交换所有权映射。必须同时保持 payload、nested scale 描述符和 key 映射一致。

当前模型几何：48 层，每层 512 routed experts，实际 top-k=10，hidden=2560，expert intermediate=640；36 层 GDN / 12 层 full attention；full attention Hq=24、Hkv=2、D=256。不要沿用原外部交接文件的“Top-K=8/16”来配置本模型。

量化几何要分清：专家主 NF4 block=64；nested scale block=256；PLE lookup 表的 NF4 block=32。compact expert slot payload 为 `2,534,400 B`，含 gate/up、down packed bytes 和 nested codes。元数据中的 per-expert nested offsets 必须正确偏移，已经因遗漏修过一次。

### 4.3 当前 prefill 与 decode 分支

| 条件 | 当前执行 | 接手注意事项 |
|---|---|---|
| `count > 4` | `RunAdaptivePrefill` | 按 expert 聚合 rows，GPU hits 先执行，冷专家每层每窗口只上传一次；每批最多16专家 |
| `count <= 4` | adaptive `RunResidentDecode` | 按 token 分开做 hits/misses 和 rank 有序合并 |
| 单行且 CPU overlap 启用 | `StartArenaCold` | CPU 从实时 RAM owner 读权重，和已提交的 shared/resident GPU 工作重叠 |
| 两至四行 | 当前不调用 `StartArenaCold` | **不是旧版本优化过的多行 CPU verifier**；缺失专家会走 GPU staging |
| forward 结束且 `count==1` | adaptive usage / `PlanSwaps` / ownership exchange | 默认最多2 swaps，频率默认1；它不受旧 `HOTSET_SWAPS=0` 同一套 gate 控制 |

`[CORRECTNESS]` 冷专家 CPU 工作必须 Join 后才允许交换 RAM owner。跨 forward 的 staging reader、resident reader 和 copy 事件必须存活；双 bank 下一次写入依赖上次消费完成。compute queue in-order 的顺序保障与 copy queue 的跨队列依赖都不能删掉。

已有注释中“XMX fused kernels”不是 dispatch 证据。当前 `GROUPED_XMX_MIN_ROWS=0`，不能据函数名推断所有专家都在 XMX 上执行；以 `RunGroupedExperts` 的实际条件和 kernel 为准。

### 4.4 启动参数保留，不代表旧分支仍生效

`tools/server/start-qwen38.ps1` 读取 `qwen38-server-mtp-2026-10-07.json.worker_environment`，随后覆盖 MTP=0、reserve3584 和 profile 路径。这个 JSON 有大量历史开关。

本次源码核实发现：

- 旧 `cpu_async_route` / `cpu_verify_async` 的条件包含 `!use_adaptive_pipeline_`；在当前固定 true 的路径中不会进入。`CPU_ASYNC_ROUTE=1` 不等于旧早发 GPU resident route 已接上。
- `adaptive_cache_.InitializeResidency()` 当前没有传入外部 ranked profile；其默认使用 `Qwen4ExpDefaultHotExperts`，然后按 expert/layer 补齐唯一 key。**launcher 设置的 H4 profile 没有通过这次调用用于初始 GPU residency。** 这不是性能根因结论，但值得单独调查。
- 旧 cache 的 Begin/EndHotsetForward、prompt borrowing、ready adoption 与新 adaptive cache 是两套实现；不要混用统计字段或直接套用旧冷槽预算。
- `HOT_RAM_MB=0` 是旧扩展 host pool 的参数，不能理解成当前大页 RAM complement 为零。
- `AILA_Q4E_PLE_LOOKAHEAD` 实验开关已经删除；当前自动行为由已有 `AILA_Q4E_PLE_ASYNC` 控制，并要求调用传入已知下一块。

这些差异解释了为什么接手应从当前调用链看问题，而不能把旧 JSON 当成优化功能清单。也不要因此批量删除旧实现或测试资产：本次任务是继续提速，清理应只针对已确定没有用途且妨碍本轮工作的部分。

## 5. 内存与驻留的已知账本

### 5.1 模型容量

`[MEASURED]` safetensors header 按 data_offsets 的清点，见 [驻留审计](qwen38-residency-audit-2026-10-08.md) 和 `tmp/q4e-residency-inventory.json`：

| 内容 | GiB |
|---|---:|
| Routed experts，含 NF4/nested 元数据 | 58.0354 |
| 其余层内 backbone | 2.7405 |
| Backbone 合计 | **60.7759** |
| Embed tokens | 1.1841 |
| LM head | 1.1841 |
| 上述合计 | **63.1441** |
| 最终 HC mixer，另计 | 0.0122 |
| PLE | **29.8184** |
| MTP，当前未启用 | 1.2840 |
| Vision，当前文本测试未使用 | 0.2225 |

主体专家 RAM 补集与 GPU cache 是覆盖关系。不要将 mmap 工作集、standby file cache、arena 和 Host USM 的数值不加区分地相加；Working Set 下降也不能证明某个普通 Host USM 分配不可分页。

### 5.2 当前32K服务的账本

`[MEASURED]` 最后一轮服务报告：

| 项目 | bytes / 解释 |
|---|---|
| model device ledger at load | `11,963,686,138 B`，不等于 DXGI 峰值 |
| Session device reservation | `1,049,174,016 B`，约1000.57 MiB |
| Session host reservation | `117,669,888 B`，约112.22 MiB，IPC snapshots=0 |
| 10-09修复验证最低 available RAM | 加载约4.95 GiB，运行约4.91 GiB，2秒采样 |
| 4K CLI 修复样本 | GPU2800专家，RAM21776专家，专家大页约51.44 GiB，embedding另计 |

最后一行不是32K的精确槽数。后续需要槽数时，应读取那次初始化日志或映射，而不是从4K报告拷贝。

此前 PLE 工作区已经由按最大上下文预分配改成加载预留64 rows、prefill按需扩容：32K时原4.375 GiB固定计算工作区，加载时降至8.75 MiB；2048/4096 rows时280/560 MiB。异步 staging banks仍按最大容量分配。这里是 tensor payload计算，不是显存实测节省；省下的加载空间还会影响专家cache准入。

`[CORRECTNESS]` PLE工作区扩容前排空compute queue，不移动异步gather/DMA还在使用的staging。decode复用现有缓冲。当前3584 MiB reserve用于容纳prefill增长和设备余量，是本机测量配置，不是所有A770的通用阈值。

## 6. 最后一轮保留的优化：跨块提前准备 PLE

### 6.1 触发原因与实现边界

驻留后重新做了少量轻量汇总。`[MEASURED]` 直接14K诊断中，第2、第3个4096-token块的PLE主机区间为13.07/13.71秒；准备直到本块开始才启动，但输入prompt及前两个token早已可知。这个区间含数据准备、就绪等待和调用开销，不能直接叫“SSD用了13秒”。

`88be93c` 做的改动：

- Request将一个已知的后续prompt chunk经Session和ForwardOptions传到Q4E。
- `ple_.prepare`立即将输入token复制到已有两个staging slot中的另一槽，由已有后台线程完成hash、gather、解量化和H2D。
- 下一次Forward使用未来ticket前核对session指针、generation、start_pos、前两个token、输入长度与完整实际token内容。
- ticket只提前准备输入数据；**不提前推进KV、GDN、PLE卷积状态或session长度**。状态更新仍发生在真实Forward/consume时。
- reset/truncate使旧ticket不能复用。取消的后台任务只依赖模型持有的token副本、权重和staging；析构先等待后台准备/传输，再释放资源。
- 复用已有线程与双缓冲，没有新增线程池、权重副本或GPU/Host USM buffer。HTTP/IPC接口没有改变。
- 实验开关从运行时代码移除，普通Server prefill在既有PLE async打开时自动使用。未传下一块的CLI调用仍逐块准备，不宣称CLI获得相同收益。

这是一块lookahead，不是跨层路由预测，也不预知模型未来生成的token。不要将它扩成无限队列或再复制大块权重，除非新证据能说明必要性。

### 6.2 正式性能结果

`[MEASURED]` A1→B1→B2→A2；A=本块开始才准备，B=提前准备下一块。新Worker就绪后的第一个业务请求均是PI，然后取消恢复、14K摘要、精确前缀续轮。正式测量关闭诊断，没有先做短业务warmup。

| 运行 | 启动至就绪 s | 首个PI任务 s | 启动+PI s | 后续14K请求 s | 14K续轮 s | 续轮tok/s |
|---|---:|---:|---:|---:|---:|---:|
| A1 | 128.09 | 90.40 | 218.49 | 102.39 | 4.15 | 12.79 |
| B1 | 127.99 | 79.51 | 207.50 | 74.85 | 4.15 | 12.89 |
| B2 | 133.34 | 79.90 | 213.23 | 74.76 | 4.18 | 12.59 |
| A2 | 132.28 | 89.02 | 221.30 | 102.06 | 4.17 | 12.77 |

14K长请求均输入13979、输出32，cached=0；输出文本相同。首字中位`99.73→72.30 s`，完整请求中位`102.23→74.80 s`。这是PI之后的暖态新长请求，不是服务启动后第一个14K业务请求。

14K续轮输入14035，其中复用14011、新增24、输出32。总时间中位`4.1587→4.1652 s`，约+0.16%；首字约1.73秒，首字后约12.74 tok/s。续轮没有明显提速，这符合lookahead主要隐藏大块准备时间的预期。

PI首轮输入均6931，但两对首轮输出分别39/26 tokens；工具续轮均新增32、输出14。每对内部输入、助手文本、工具参数和用量一致，忽略服务生成tool-call ID后输出hash一致。**PI收益只按各自配对解释，不能把四次中位当成严格同工作量实验。** 每次恰好一次`read(smoke-marker.txt)`，最终答案均为`AILA_FIXED_AGENT_READ_20261007`。

启动约130秒，没有启动提速。启动+PI中位`219.89→210.36 s`，约缩短4.3%；仍需同时报告启动和首个任务，不能只给12%的PI改善隐藏预加载成本。

A1使用候选前逻辑加关闭的轻量汇总；B1/B2/A2是同一Worker二进制切换已删除的实验开关，B2/A2提供同二进制反序对照。收益约27%后按高效迭代要求停止，不再追加性能轮次。此结果不保证所有prompt或完整32K输入都改善相同百分比。

主证据：[报告MD](qwen38-ple-lookahead-2026-10-09.md)、[报告JSON](qwen38-ple-lookahead-2026-10-09.json)，原始记录`tmp/q4e-resident-speed-20261009/{a1,b1,b2,a2}/`。

### 6.3 最终功能检查与取消限制

`[CORRECTNESS]` `AilaRequestStateTests`通过，约0.12秒；真实权重的`AilaQwen4ExpModuleTests --ple-only`通过，约9.13秒。PLE新增覆盖65→7→1三块、EOS历史、双staging交替、完整输出和卷积状态逐位一致、丢弃旧输入后重试、无新增设备分配。

四次生产式序列均完成PI、取消后READY恢复及14K续轮，正常退出，没有强杀。移除实验开关并重编后，还额外在新长prefill首字前断开，排空后新请求2.04秒返回READY。

**取消仍按chunk边界生效。** 该次首次大块排空用了52.41秒；2.04秒是排空后恢复请求的时间，不是用户断开到恢复的总时间。取消请求的generated_tokens可能为null，表示没有最终Worker用量快照。初版探针将null误判为失败，之后核对disconnect、未排队内容和实际排空后通过；不能把这次探针断言错误描述为运行时算错。

最终失败/成功记录分别在`final-cancel/`和`final-cancel2/`，均保留，不进入性能统计。没有重跑全CTest、复杂仓库修改任务、并发验收或完整32K输入验收。

## 7. 当前可以相信的瓶颈证据

### 7.1 新驻留版本的最新线索

以下来自`a83cd80`上、lookahead接入前的`diagnostic2`轻量记录。它用于决定先改PLE，不是修改后正式性能的精确分解：

| 观察 | 证据与含义 |
|---|---|
| PLE后两大块主机区间13秒以上 | 已通过lookahead得到整轮收益；修改后要重新测，不能继续当作完整可节省时间 |
| staging fill约7.30秒/完整4096块 | RAM到小型Host USM的填充仍有成本；不是专家磁盘读取 |
| bank等待约4.4秒/完整块 | 与GPU消费者和bank复用有关；不能直接当PCIe纯耗时 |
| 单token Forward约72–74ms | 当前完整路径还高于20tok/s要求的约50ms/token |
| 每token CPU专家约408–421/480 | 当前GPU命中覆盖较低；和旧路径约274/480的诊断有差异，但不是同轨迹因果比较 |
| CPU compute约21–23ms，join约15–18ms | 存在CPU/GPU重叠，不能相加或从Forward减join推导纯GPU成本 |
| 24行短追加：约1.77秒，CPU tasks=0 | prefill仍走GPU供给；staging约7874.56MiB，约7.69GiB，fill约1.21秒 |

`[HYPOTHESIS]` 当前初始静态hotset、较晚的resident GPU调度、单行CPU冷专家接入、nested scale解码和短追加供给都可能有空间。当前没有证明某一个是新的最大瓶颈。

轻量统计的`*_host_ms`往往是提交函数区间；只在首个大块有扩容/JIT时出现的成本不能算作steady-state。事件异步完成的kernel不会因为主机提交只有几十毫秒就只用了几十毫秒。

### 7.2 老路径的定位，只保留作为方向

10-07 [端到端瓶颈报告](qwen38-bottleneck-2026-10-07.md)记录：真实PI135.79秒，Worker134.84秒，外层差约0.70%；暖HTTP外层差约0.11%/0.41%。Server/IPC不是当时的主要瓶颈。

该报告的prefill acquire/fill占比、HC/attention排序、24token短追加仍需大量供给等，帮助形成后续实验；但它早于原生大页adaptive重构。更关键的是报告曾用“Forward−CPU join”列出剩余区间；**该剩余不能当成纯GPU时间或未来流水线的理论下界**。后续只使用原始区间和完整墙钟，不重复这种归因。

### 7.3 哪些统计不能混用

- H2D累计160GB/177GB说明专家权重重复供给，不证明从SSD读了同样字节。驻留后专家供给来源是RAM。
- Windows PhysicalDisk累计差含同盘其他进程和分页文件，不是模型逐文件归因，也不是PCIe counters。
- `available`包含可回收standby file cache；不是所有字节都从未使用。
- 阶段profile会插barrier并强制materialize延迟MLP residual，改变执行重叠。其速度和阶段排序不能拼回无诊断生产数据。
- CPU join、CPU compute、copy和GPU kernel存在重叠，禁止相加充当总成本。
- 没有完整日志覆盖的请求，engine total应为null；host log drop=0不保证Worker日志完整。

## 8. 关键源码与资产导航

路径相对仓库根；行号以`88be93c`为准，后续编辑会移动。先按符号搜索，再看附近实现。

| 文件 / 符号 | 接手用途 |
|---|---|
| `src/models/Qwen4ExpBnb4Backend.cpp`：init约643；`ForwardImpl`；`RunResidentDecode`约1036；`RunAdaptivePrefill`约1235 | 显存准入、实际dispatch、CPU/GPU分工、PLE lookahead、统计及swap收尾 |
| `src/models/Qwen4ExpBnb4Backend.hpp` | 固定adaptive开关、scratch与session所有权、future PLE ticket |
| `src/models/Qwen4ExpAdaptiveCache.cpp/.hpp`：`InitializeResidency`、`StageColdExpertsAsync`、`ExecuteSwapsAsync` | GPU key唯一性、双bank、copy/consumer事件、动态所有权 |
| `src/models/Qwen4ExpHostArena.cpp/.hpp` | 原生大页RAM补集、块分配、加载回收、CPU描述符与owner exchange |
| `src/models/Qwen4ExpExpertExecutor.cpp/.hpp`：`StartArenaCold`约835、`JoinResidentCold` | CPU任务输入/输出的事件、RAM权重源、18线程执行器 |
| `src/models/Qwen4ExpPle.cpp/.hpp`：`prepare`、`MatchesInput`、`consume` | PLE线程、两个staging slot、token副本、ticket与conv state |
| `src/inference/Request.cpp`、`Session.cpp/.hpp`、`Model.cpp` | prefill剩余量、下一chunk输入、pending token、Model执行锁 |
| `src/models/IModelBackend.hpp` | ForwardOptions中的next_prefill_tokens/count；借用输入只在调用期间有效 |
| `src/models/Qwen4ExpPrefill.cpp`与`Qwen4ExpBnb4Backend.cpp` | GDN、attention、projection、短行与验证路径 |
| `src/ops/MoeExpertOps.cpp`：`read_nested_absmax`约408 | NF4+nested scale解码、gate/up/down多行复用与rank合并 |
| `src/ops/Qwen4ExpCpuExpertOps.cpp` | CPU NF4-Q8 / A64算术、gate/up与down热点 |
| `src/ops/Qwen4ExpAttentionPair.hpp`、`src/ops/GreedyOps.hpp` | 历史T2双查询与GPU greedy实现；有限形状资格 |
| `src/models/Qwen4ExpMtpDraft.*`、`Qwen4ExpMtpRuntime.*` | 实验性draft、checkpoint、verify/commit；当前<=2048资格 |
| `src/memory/SessionMemory.hpp` | Session device/host reservation和overflow/几何检查 |
| `src/memory/LargePages.cpp/.hpp`、`src/core/Context.hpp` | 原生大页分配与实际账本；Context实现位于头文件 |
| `src/server/HttpServer.cpp`、`ChatCodec.cpp` | 单槽准入交接、前缀复用、工具历史、SSE与PI字段 |
| `src/ipc/InferenceProtocol.*`、`src/runtime/InferenceClient.cpp` | build ID握手、可协商长上下文上限 |
| `tools/server/start-qwen38.ps1` | 本机默认启动入口 |
| `tools/server/qwen38_context_probe.py` | 自建并关闭Server的真实PI/HTTP/取消探针 |
| `tools/server/freeze_qwen38_agent_inputs.py` | 固定自然语料生成与hash校验 |
| `tools/server/pi_agent_smoke.py` | 真实PI隔离配置、固定workspace、read marker验证 |
| `tests/models/Qwen4ExpModuleTests.cpp` | `--ple-only`真实权重检查 |
| `tests/models/Qwen4ExpHostArenaTests.cpp`、`Qwen4ExpAdaptiveCacheTests.cpp` | 现行大页/所有权/真实几何payload检查 |
| `tests/models/Qwen4ExpExpertReadinessTests.cpp` | 较早大页路径的handoff/async资产；不等于现行adaptive覆盖 |
| `tests/ops/Qwen4ExpKernelTests.cpp`及相关hpp | attention-pair、greedy、KV-FP8聚焦检查 |
| `tests/inference/RequestStateTests.cpp`、`Qwen4ExpSessionTests.cpp` | Request阶段传参、窗口状态、shared session生命周期 |

旧`Qwen4ExpExpertCache`实现、learned profile和teacher/replay仍可作为参考资产。若移植某段优化，应接到上述现行所有权和实际dispatch，而不是重新引入全量mmap消费、重复RAM权重或第二套线程池。

## 9. MTP 已做过什么，为什么当前关闭

### 9.1 历史有效结果

`[MEASURED]` 10-06清理版探针：1833-token原CLI system prompt、128 outputs、T2、CPU18/A64、冻结hotset、64 draft slots、BF16 KV，完整循环 **20.592890 tok/s / 6215.7375ms**，包含首窗口268.94ms，128/128输出与该次T1参考相同，LIVE USM delta=0。

同期保留过的组合包括：多行CPU冷专家、GPU expert 4/2/1 reuse、NF4 projection reuse、BF16双查询KV复用、batched GPU greedy、去掉无用draft tail copy、state-only replay、decode prepare、draft hidden reuse。kernel bundle归后端持有，初始化成本计入账本，不需要后台bundle线程。

证据：[MTP清理验收](qwen38-mtp-cleanup-2026-10-06.md)、[BF16双查询结果](qwen38-mtp-kv-pair-8724-2026-10-06.md)、原始`tmp/mtp-8724-cleanup128.*`。

### 9.2 服务集成不等于无损20+

实验MTP已迁入共享模型、Session、Request和Worker，内部窗口显式报告consumedInputs、cacheLength、firstTokenCached及未缓存bonus；IPC仍逐token发布。接受draft后不能再次forward同一已缓存token，拒绝/stop必须恢复已执行后缀；取消用过MTP的session会reset以避免复用超前状态。

`[MEASURED]` 服务侧1833 prompt / 128 outputs，补回hidden reuse后首字50.51秒、总58.99秒、首字后14.984 tok/s。并且T1对照出现7/8和后来6/8相同的情况，**严格T1等价没有通过**。见 [Server MTP报告](qwen38-server-mtp-2026-10-07.md)。

当前32K服务强制MTP=0；MTP runtime/draft要求dense context<=2048，非默认GDN chunk或Q8 head也不合资格。稀疏长上下文draft、首次分歧算子与普遍数值对齐均未解决。

更重要的是，adaptive重构后`count=2..4`已经不会走现行`StartArenaCold`，旧verifier CPU优化不能通过重新设置环境变量恢复。**若重启MTP工作，先核实现行target verifier的路由、CPU支持与成本，再谈历史20+组合。** 不建议直接在默认32K launcher开启MTP。

### 9.3 数值定位留下的线索

历史同checkpoint的B2/连续B1比较显示差异发生在head之前；相同最终hidden输入的head-only对照逐位一致。router少量scores差异在该窗口未改变top-k IDs，不能据此认定全模型router完全等价。单行GDN参考能消除一部分局部状态差异，却没有修复整模型长轨迹。

这些是定位资产，不是结论。涉及B1/B2近似算术、CPU/GPU placement和历史状态时，用同checkpoint、同输入、同placement寻找第一处分歧；不要用一个prompt的token parity宣布普遍无损。

## 10. 重要负结果与已踩过的坑

### 10.1 本thread已否决的性能方向

除非新路径或新形状带来明确的新假设，不应原样再跑一次。

| 候选 | 已有结果 / 处理 | 主要证据 |
|---|---|---|
| HC直接写pinned CPU输入 | 暖decode14.9，基线17.6；撤回 | `qwen38-decode20-fast-iteration-2026-10-05.md` |
| classify写CPU输入、A64小展开 | 约17.4，未胜出；撤回 | 同上 |
| CPU12线程、TBB static_partitioner | 暖14.5/14.3；保留18线程 | 同上 |
| shared expert加法并入合并 | 15.9 vs同EXE17.4；撤回 | 同上 |
| 小范围reserve2304/2816 | 未改善；历史2560保留，但不是当前3584的默认依据 | 同上 |
| short append拆2/4行 | H2D下降但整轮不稳定；4行6K首字更慢；撤回 | `qwen38-htoff-append-2026-10-07.md` |
| 8Kchunk+更高reserve+tile256 | H2D160→91GB，TTFT反而增加；组合撤回 | 同上 |
| short append保留全热集/仅借32冷槽 | acquire数量、6K退步等抵消收益；撤回 | 同上 |
| 限制下一组H2D等待上一组down | 80行完整链wall134.457→147.308ms，慢9.56%；未接主模型 | `qwen38-agent-pacing-2026-10-07.md` |
| 部分文件页VirtualLock原型 | 只锁42.05GiB，未锁部分读盘增加；不推广 | `qwen38-residency-audit-2026-10-08.md` |
| 同microbatch D2H批量回写 | 无有说服力的整轮收益；撤回 | `qwen38-large-page-residency-2026-10-08.md` |
| 大页staging整半区填完再提交 | 6K重复36.36→36.95s；无收益 | `qwen38-large-page-execution-2026-10-08.md` |
| 大页专家ready即提交流水线 | 重复6K36.301→36.399s；未默认启用 | 同上；早于当前adaptive实现 |
| target FP8 KV | 完整MTP19.776→17.055tok/s；轨迹改变，分支撤回 | `qwen38-mtp-kv-fp8-8724-2026-10-06.md` |
| FP8 K向量读取 | 改善FP8微测但仍慢于BF16；撤回 | 同上 |
| MTP简单margin阈值策略 | 约1%单样本收益，撤回 | `qwen38-mtp-alignment-draft-2026-10-06.md` |
| shared FFN额外双token复用 | 约0.176%，撤回模式4 | `qwen38-mtp-greedy-8724-2026-10-06.md` |
| 后台kernel bundle线程 | 同EXE关闭更快；不保留 | `qwen38-mtp-kv-pair-8724-2026-10-06.md` |

FP8尝试是E4M3FN单字节格式，不是动态scale/zero-point的UINT8方案；其负结果不覆盖所有8-bit KV，也不覆盖更长context。但本机当前没有证据优先重试。标准FP8编解码正确性修复和聚焦检查保留，Q4E当前KV仍BF16。

根AGENTS.md还有DG2历史负优化：GEMV SLM输入缓存、BF16替代NF4 decode权重、blocked weight布局、每SG两输出行、uint64权重load、QMAP改private、在GEMV里融合residual。不要因“理论带宽/融合”直接复活这些方案；原thread已有证据反复显示局部省流量不等于整轮加速。

### 10.2 故障与正确性经验

- **全专家Host USM+GPU热副本失败。** 重构原版将全部24576专家放进Host USM，GPU再留热专家，系统available最低约0.06GiB，第一次2-token forward Error40随后0xC0000409。现行原生大页补集修好后运行最低约4.91GiB。不要回到重复覆盖方案。
- **Working Set下降不能证明物理不可分页驻留。** 原外部交接文件过早宣布“51GB锁死”；不能据此跳过完整触达与真实forward。
- **大页分块与碎片有关。** 54GiB单次、1GiB分块都曾Win32 1450；256MiB分块探针成功，当前arena最多256MiB。单次成功不是任意系统状态的保证；分配失败明确报错，不隐藏回退到可分页/磁盘。
- **加载mmap源页会造成双占用。** 逐层回收普通工作集解决了加载阶段available长期不足及RPC超时；不可删掉后又把结果叫“预热不稳定”。加载deadline扩大不代表启动提速。
- **nested metadata要按专家偏移。** GPU descriptor漏偏移、CPU nested blocksize漏填256都修过；GPU元数据分配失败不能用普通vector指针冒充device pointer。
- **初始expert key不能重复。** GPU/RAM补集要覆盖48×512完整key空间；默认hot keys与补齐循环曾产生重复，现已修好。
- **不能跳过PLE分片。** 已移除跳过31个PLE safetensors的入口；完整索引131个文件分片、4114个文本张量，PLE模块自身有128张lookup shard表，两个“分片数”不是同一概念。
- **SYCL internal event wait。** 旧时间线探针、旧逐槽重复DMA等待曾触发`urEventWait must not be called for an internal event`，0xC0000409。无有效时间线的失败不能当成profile结果；旧重复等待移除但消费者fence保留。不要无差别删wait。
- **单槽429曾是服务交接问题。** 完成推理后TCP排空仍占推理准入，PI快速工具续轮被拒；已在正常结束帧前释放推理句柄并完成prefix交接。当前仍单槽，无需增加并发绕过。
- **错误析构须先排空再释放，避免重复抛异步异常。** 模型warmup失败时缓冲释放曾加剧终止；当前释放路径记录错误后drain，不在析构再次抛。

原始失败记录都在tmp中保留；不要清理目录以便“重新跑同一个tag”。

## 11. 下一轮建议：先选一个候选，再动手

这些是交接建议，不是已授权之外的新性能结论。本次没有实现或测试下面的候选。

### 11.1 第一步：确定当前基线和实际路径

先确认HEAD/二进制身份、固定语料和环境，使用已有runner跑一次生产式序列。不要拿10-05旧CLI17.6、MTP20.59、10-07串行profile或不同prompt的样本当当前对照。若当前基线明显偏离最新B结果，先查输入/输出用量、诊断遗留、driver、驻留槽、RAM状态、prefix复用和握手；不要直接重写内核。

需确认下一轮到底优化哪个成本：14K新prefill、首个PI任务、24token追加、暖decode、启动。每项单独记账。已有启动约130秒是很大的成本，如果用户主要反复启动worker，启动优化可能比暖decode更重要；当前任务方向仍以长任务整轮为主。

### 11.2 候选A：nested scale固定几何去掉运行时整数除法

`[HYPOTHESIS]` `src/ops/MoeExpertOps.cpp::read_nested_absmax`使用`size_t index / nested.blocksize`；本模型nested block固定256，代码疑似生成运行时64位整数除法。原thread已经发现但因PLE更大而未修改。

先看实际codegen确认除法是否还存在以及调用覆盖哪些gate/up/down路径。然后在**不改量化精度、BF16舍入和累加顺序**的前提下，做本模型固定几何的最小特化，比较完整expert链墙钟。它比再开一轮广泛thread/chunk扫描更容易隔离，但目前没有收益数据。

局部要覆盖GPU resident和staging描述符、nested offset、完整gate/up+SiLU+down与rank输出；通过后才接真实HTTP序列。若只有kernel小胜、整轮无收益，则归档撤回。

### 11.3 候选B：恢复有效的GPU热集与较早resident调度

`[HYPOTHESIS]` 当前init没有使用外部ranked H4 profile，最新诊断CPU约408–421/480专家；旧优化路径的早发GPU resident route也被adaptive gate绕过。可能存在“驻留容量已足够，但热专家选择/调度没继承”的缺口。

先核对当前GPU key集合、初始化profile覆盖、真实自然prompt的hit/CPU比例、swap改变与所有权事件，再选**profile接入**或**早调度**其中一个变量。不能同时改二者后声称已定位。历史prompt/hotset改变会改变CPU/GPU近似算术和生成轨迹，需要记录placement、token与输出hash；若轨迹不同，只能称策略比较，不能计算固定工作量的算子加速。

不得重新引入RAM/GPU双份常驻、共享可变slot而没有consumer fence，或只打开旧CPU_ASYNC_ROUTE=1就宣称接通。

### 11.4 候选C：24token短追加的供给与两至四行CPU支持

`[HYPOTHESIS]` 当前短追加仍RunAdaptivePrefill、CPU=0、约7.69GiB staging；现行count2..4也缺失CPU冷专家。多轮大量短追加时值得调查。

先按实际所有权和完整层成本设计小对照。历史拆2/4行、保留全部热集、32冷槽均已失败或不稳定；**不应原样重试**。只有新adaptive执行方式能避开旧候选的逐组层开销、冷槽限制等具体问题时，再尝试。每轮保持同样输出数量和prefix命中量。

### 11.5 延后项目

- 重新开启长上下文MTP：先解决现行多行verifier支持、dense<=2048资格及T1对齐，才有意义。
- 大范围HC/GDN/attention搜索：先有有效轻量时间线/完整链证据，不能复用旧profile排序作为当前最大瓶颈。
- 增大chunk或减少reserve：旧组合没有整轮收益，且会改变placement和workspace；不优先扫参数。
- 新并发、HTTP/IPC重写、其他模型兼容性、大规模清理框架：不属于当前目标。

`[HEURISTIC]` 若需要选择第一项，优先从A或B中挑一个能最快得到隔离证据的实验。对新算子微测，沿用A770技能的CPU参考/数值资格、warmup、至少20个交错batch样本和中位数/离散度；对真实整轮先一对screen，有约5%以上收益再补反序。小于噪声或不改善整轮就停止，避免为弱信号建立新架构。

## 12. 可复制的接手与复现命令

### 12.1 核对快照，启动现有服务

```powershell
Set-Location E:/RiderProjects/Aila
git status --short
git rev-parse HEAD
Get-FileHash build-refactor/AilaWorker.exe, build-refactor-server/AilaServer.exe -Algorithm SHA256
pwsh -NoProfile -File tools/server/start-qwen38.ps1
```

launcher默认模型路径、port8080、context32768、chunk4096、max-new-tokens2048、单session、decode-batch1、max-body4MiB、prefix reuse启用、request-timeout600000ms；加载JSON后明确覆盖MTP=0和reserve3584。读第4.4节后再解释其他历史环境参数。

### 12.2 完整官方构建与Server配齐

```powershell
pwsh -NoProfile -File build.ps1 -BuildDir build-refactor -Jobs 8
pwsh -NoProfile -File tools/server/build.ps1 -SkipDependencies
```

构建不等于执行全套测试。正式提交/换构建身份后，用这条流程生成一致的构建元数据和Server握手标识。不要运行`-Clean`或删除既有tmp记录来解决普通编译问题。

同一已配置实验基点上，原thread还有一个本地定向构建helper，可只编本轮目标：

```powershell
& ./tmp/q4e-resident-speed-20261009/build.ps1 -Targets @(
    'AilaWorker', 'AilaFakeInferenceWorker',
    'AilaRequestStateTests', 'AilaQwen4ExpModuleTests'
)
pwsh -NoProfile -File tools/server/build.ps1 -SkipDependencies
```

这个helper实际文件已核实存在，初始化oneAPI环境并记录构建信息；它是本机tmp资产，若移动仓库未携带tmp，使用官方构建入口。编缓存/arena时加相应目标即可，不必为文档或低影响配置编辑重跑所有GPU测试。

### 12.3 校验冻结输入，不重新生成旧baseline语料

```powershell
python tools/server/freeze_qwen38_agent_inputs.py `
  --model C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2 `
  --output tmp/q4e-agent-corpus-v1 --verify
```

fixture正文约5932/13932 tokens；chat template后普通摘要实际5968/13979。PI加自身指令与工具schema，最新首轮6931。不要将正文token数当最终HTTP prompt数。

```text
context-6000.txt content SHA256:
6437d29f146179e55e71c0d08da55dfd905c0e672dbfd9fb3bfe9310735221be
context-14000.txt content SHA256:
aa3169bf2689bedcfcad3e0e0f9c30142a4dfae0bf8a215b7fe1a39259417ea3
tokenizer.json SHA256:
0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3
```

其token hash和来源excerpt hash在`manifest.json`。PI固定workspace是`tmp/q4e-agent-corpus-v1/pi-work`，marker为`AILA_FIXED_AGENT_READ_20261007`。保留隔离PI配置，不写全局PI账号或认证文件。

### 12.4 最接近最新正式序列的现有runner

```powershell
python tmp/q4e-resident-speed-20261009/run.py handoff-next-baseline --agent
```

runner创建子进程环境，清掉继承的AILA/SYCL/UR_L0参数后读取固定worker_environment，再覆盖MTP0、reserve3584、chunk4096；正式运行加production-timing。顺序为首个真实PI6K工具任务→取消恢复→14K摘要32outputs→前缀续轮32outputs，记录启动和RAM监控，自动关闭Server。

输出在`tmp/q4e-resident-speed-20261009/handoff-next-baseline/`，同名目录已存在会拒绝覆盖。实际运行时使用新的唯一tag。该命令是接手复现指引，本次交接没有执行。

**不要再把`AILA_Q4E_PLE_LOOKAHEAD=0/1`传给当前二进制做A/B。** 开关已经删除，会得到相同运行逻辑。历史A/B对应归档的candidate source和Worker；新的实验应建立自己的明确对照身份。

若不依赖tmp runner，可在独立PowerShell进程中加载相同配置后运行仓库探针：

```powershell
$q4eProfile = Get-Content docs/plans/qwen38-server-mtp-2026-10-07.json -Raw | ConvertFrom-Json
foreach ($q4eEntry in $q4eProfile.worker_environment.PSObject.Properties) {
    [Environment]::SetEnvironmentVariable($q4eEntry.Name, [string]$q4eEntry.Value, 'Process')
}
$env:AILA_Q4E_SERVER_MTP = '0'
$env:AILA_Q4E_VRAM_RESERVE_MB = '3584'
$env:AILA_Q35_PREFILL_CHUNK = '4096'
python tools/server/qwen38_context_probe.py `
  --model C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2 `
  --output tmp/q4e-handoff-next-baseline `
  --context 32768 --chunk 4096 --generate 32 --reuse 1 `
  --fixture tmp/q4e-agent-corpus-v1 --prompt-tokens 14000 `
  --production-timing `
  --pi-context-file tmp/q4e-agent-corpus-v1/context-6000.txt --cancel-recover
```

此探针有固定worker/server路径并管理服务生命周期；不要同时手动启动另一个GPU服务。默认probe参数context16384/chunk512不是最新正式配置，必须显式传上述参数。

### 12.5 只运行与改动相关的检查

PLE / Request改动：

```powershell
./build-refactor/AilaRequestStateTests.exe
./build-refactor/AilaQwen4ExpModuleTests.exe --ple-only
```

现行arena/cache/所有权改动：

```powershell
ctest --test-dir build-refactor `
  -R '^(AilaQwen4ExpHostArenaTests|AilaQwen4ExpAdaptiveCacheTests)$' --output-on-failure
```

这两个目标含实际GPU检查，不能因为旧log文件名叫`targeted-tests-cpu.log`就把它们称CPU-only；最后报告标签为gpu/models/qwen4_exp，总约9.38秒。

若确实修改相应历史算子，可以用已有聚焦入口：

```powershell
./build-refactor/AilaQwen4ExpKernelTests.exe --attention-pair-only
./build-refactor/AilaQwen4ExpKernelTests.exe --mtp-greedy-only
./build-refactor/AilaQwen4ExpKernelTests.exe --kv-fp8-only
```

它们不应成为每一轮prefill调度的固定测试负担。MTP改动才补Request窗口/真实session检查；协议变化才补Protocol/Client/ChatCodec；不变的路径不重复验收。

长prefill取消检查可在已初始化的同一runner里加`--cancel-prefill`；当前排空可能几十秒。检查disconnect、没有排队输出、真实恢复与正常退出，不要求所有取消usage字段非null。

### 12.6 正式对照的最小记录

每轮保存：source commit/patch、Worker/Server hash、完整有效环境、fixture/token/request hash、prompt/复用/新增/output tokens、完整输出hash、启动、TTFT、整轮、首字后速率、任务结果、内存余量与退出/取消状态。

首字后速率口径固定为：

```text
(completion_tokens - 1) / (SSE done - first content)
```

它包括IPC和finalization；不是纯kernel decode，也不是128/完整MTP循环的同一个口径。20+必须在定义清楚的口径、足够输出长度、包含约定首用成本的条件下达成。

正式测速关闭WorkSummary、stage/component profile、trace、Worker日志捕获、verbose。需要诊断时单独有限运行，保留其扰动范围；不把诊断速度混入生产表格。输入/输出工作量不同的PI只做配对内解释。

## 13. 原始资产与报告索引

### 13.1 最重要的本机目录

| 目录 / 文件 | 内容 |
|---|---|
| `tmp/q4e-resident-speed-20261009/` | 最新优化完整档案；a1/b1/b2/a2、launch/result/report、PI events、监控和patch |
| `tmp/q4e-resident-speed-20261009/diagnostic2/`及`diagnostic2-worker.log` | 驻留后PLE/staging/CPU诊断；不用于正式速度 |
| `tmp/q4e-resident-speed-20261009/final-cancel2/` | 最终二进制的长prefill取消与READY恢复 |
| `tmp/q4e-resident-speed-20261009/AilaWorker-baseline.exe` | 基线归档，hash以最新报告JSON为准 |
| `tmp/q4e-resident-speed-20261009/AilaWorker-lookahead-reference.exe` | 带实验开关的历史同二进制A/B候选 |
| `tmp/q4e-resident-speed-20261009/candidate-source.patch`、`final-source.patch` | 测量候选和移除开关后的源码档案 |
| `tmp/q4e-refactor-repair-20261009/` | 重构故障复现、接手patch、最终所有权检查和agent3真实验证 |
| `tmp/q4e-refactor-repair-20261009/agent/`、`agent2/` | build ID失配、32K槽数准入失败；保留失败原样 |
| `tmp/q4e-agent-corpus-v1/` | 固定正文、manifest和PI工作目录；必须随交接一起保留 |
| `tmp/mtp-current-screen.ps1`、`tmp/mtp-8724-*` | 历史MTP复现脚本和raw性能/数值记录；旧路径成绩 |
| `tmp/q4e-largepages-*`、`tmp/q4e-largepage-execution-*` | 历史大页、事件故障、读盘和staging负结果 |
| `tmp/Strata/` | 成熟参考源码；采用有限ring、consumer fence等思想，未直接移植CUDA注册 |

tmp通常不随Git克隆携带。**将优化交给另一台机器/另一份clone时，至少同时携带fixture、最新四次结果、final-cancel2、相关patch和报告JSON。** 二进制只在同环境复现时有用；源码提交与报告是主要可移植交付。

本机原thread rollout：

`C:/Users/Administrator/.codex/sessions/2026/10/05/rollout-2026-10-05T19-40-42-01a10bdd-b9a5-75c0-bc44-107fb2e66f45.jsonl`

本次已按用户要求读取其消息和最后完成记录，没有把认证文件或无关thread纳入交接。下一agent通常无需再读约64MB的整份日志；本文和以下报告已覆盖关键状态。

### 13.2 建议阅读顺序

1. [10-09 PLE lookahead](qwen38-ple-lookahead-2026-10-09.md)及同名JSON：最新保留成果、二进制身份、A/B证据。
2. [10-09 Strata重构修复](qwen38-strata-refactor-repair-2026-10-09.md)及同名JSON：现行内存所有权和故障边界。
3. [10-08大页驻留](qwen38-large-page-residency-2026-10-08.md)、[大页执行模型](qwen38-large-page-execution-2026-10-08.md)、[容量审计](qwen38-residency-audit-2026-10-08.md)：为何减少读盘后仍不等于整轮提速。
4. [10-07传输节奏否决](qwen38-agent-pacing-2026-10-07.md)、[短追加负结果](qwen38-htoff-append-2026-10-07.md)、[PLE按需工作区](qwen38-single-stream-performance-2026-10-07.md)：避免重复已否决方案。
5. [10-07 Server长上下文/PI](qwen38-server-agent-long-context-2026-10-07.md)、[Server MTP](qwen38-server-mtp-2026-10-07.md)：服务能力和资格。
6. [10-06 MTP清理](qwen38-mtp-cleanup-2026-10-06.md)、[BF16 pair](qwen38-mtp-kv-pair-8724-2026-10-06.md)、[FP8负结果](qwen38-mtp-kv-fp8-8724-2026-10-06.md)：仅在重启MTP/算子工作时深入。
7. [10-05 decode快速迭代](qwen38-decode20-fast-iteration-2026-10-05.md)和更早H系列：历史baseline，不用其最佳数代替当前Server。

如需继续kernel工作，读取本机`C:/Users/Administrator/.codex/skills/A770-Kernel-Optimization/SKILL.md`，再按实际算子路由读取weight-only GEMV、execution、codegen/robustness等相应reference。该技能中的driver/shape实测有效域也需遵守；不要把其他campaign的阈值直接搬成本项目生产dispatch。

## 14. 可直接交给下一agent的起始任务

> 在`E:/RiderProjects/Aila`承接`main@88be93c`。先读`docs/plans/qwen38-optimization-handoff-2026-10-09.md`，以当前原生大页GPU/RAM补集、Server单流32K/chunk4096/reserve3584/CPU18/BF16 KV/MTP0为基点。目标是缩短真实PI长背景工具任务、长prefill与多轮总耗时，继续追求Server decode20+，不做并发或其他模型兼容设计。
>
> 最新PLE lookahead已保留，14K暖态长请求约74.8秒，续轮约4.16秒/13tok/s，启动约130秒。不要重新实现这项改动，也不要把旧MTP探针20.59当当前成绩。核实现行adaptive调用链，尤其旧profile/async分支的有效性与多行CPU缺口；从nested scale固定256特化、初始GPU热集/调度或短追加供给中只选一个能取得隔离证据的候选。
>
> 使用已有冻结语料、生产计时和真实PI helper。先小范围筛选，有完整任务收益才补反序复测和相关所有权/取消检查；失败归档撤回，不追加开关扫描，不过度设计和测试。保留启动/首用成本，记录同输入输出用量、hash、二进制身份和退出状态。代码变更后配齐Server/Worker构建标识，不能靠改metadata绕过握手。

交接完成时的未完成工作是后续性能优化，而非修复未收尾的实现。当前仍明确未解决：Server20+、启动预加载时间、完整32K输入验收、长prefill即时取消、复杂真实仓库任务表现，以及MTP长上下文/普遍T1对齐。
