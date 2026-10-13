# Qwen3.8-Flash-Next / Qwen4Exp：A770 单流 agent 性能优化方案

日期：2026-10-12。对象：`C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2`。
本次审计源码 HEAD：`b463802ba8a07b7bd7c4f058140994b92434f86f`。

优先做**启动稳定性、真实 agent 前缀复用、专家缓存预算修正、prefill 解量化数据复用**。继续使用单会话 T1；INT8 KV 保留选项；MTP、ESIMD 接入和 layer-major prefill 暂不排进第一轮。现有证据不支持承诺 20 tok/s，也不支持把 PCIe 流量下降直接换算成 TTFT 收益。

本文交付实施方案及探测证据，没有修改生产源码、默认配置或既有未跟踪文档。标记含义：`[本次实测]` 为此次工具运行，`[源码]` 为当前实现，`[历史实测]` 为可追踪旧结果，`[推算]` 为明确假设下的计算，`[待验证]` 为下一步实验。

## 1. 目标和当前实际环境

目标是 agent 的**完整工作回合耗时**：新代码上下文的首字延迟、工具返回后的追加延迟、生成时间，以及模型可持续驻留。单独提高算子吞吐或短输出 tok/s 不等于 agent 更快。

| 项目 | 探测结果 / 使用范围 |
|---|---|
| CPU | E5-2696 v3，18 核 / 18 逻辑处理器；无 AVX-512，继续用已有 AVX2 路线 |
| RAM | 4×16 GiB Micron，配置频率 1866 MHz；OS 可见 68,622,086,144 B |
| GPU | Intel Arc A770，Windows 驱动 `32.0.101.8724`；runtime admission 日志总显存 15,933 MiB |
| 总线 / SSD | 采用用户给定 PCIe 3.0 x16 / PCIe 3.0 x4 NVMe；本次未测实际协商速率或 SSD 带宽 |
| 工具链 | CMake Release，icx-cl 路径 oneAPI 2026.1；本地 oneDNN 头文件版本 3.11.2 |
| 模型 | 48 层，12 full / 36 linear；H=2560，I=640，512 experts/layer，top-k=10；131 个 safetensors，合计 101,449,082,207 B |
| 生产路径 | `HttpServer → IPC v2 → LoadedModel/Qwen4ExpBnb4Backend → adaptive GPU/RAM complement` |

本次运行产物按 SHA256 识别，不能把源码 HEAD 自动当作所有 exe 的构建身份：

- Worker：`42bdd5cd88cc8578658b3f4f5ddacda301a07ce6532f64c00899bbe01ca19c9d`
- Server：`9fa536ac1d78e82a536c87b80a0612e2c4a8cf76ddf47ab326b78eb6fee653cc`
- 完整路径、修改时间、KernelTests SHA、输入/config/profile SHA 和复现信息见配套 JSON。

## 2. 本次运行与瓶颈证据

### 2.1 当前产物实模探测

第一次无诊断生产序列在 ready 前退出：`MEM_LARGE_PAGES allocation failed: bytes=125829120 win32_error=1450`，18.01 s；RAM monitor 最低 available 6.69 GiB。**仅凭错误 1450 不能判定是碎片、总量、还是其他系统资源约束。**

相同产物随后带诊断重试成功：32K capacity / chunk4096 / BF16 KV / 8 输出，14K 新请求和续轮通过，优雅退出，无 forced kill。ready 113.85 s，完整探针 254.07 s，RAM available 最低 0.96 GiB。两次观察只说明启动不稳定，不能估计失败概率。

| 带诊断请求 | 实际 prompt | 已缓存 | TTFT | 首字后速率 |
|---|---:|---:|---:|---:|
| 14K 新请求 | 13,979 | 0 | 134.224 s | 12.415 tok/s |
| 续轮 | 14,011 | 13,987 | 1.180 s | 14.238 tok/s |

该轮**没有 PI 暖机且开诊断**，不可与历史 70.6 s 的暖态生产首字计算回退比例；8-token 窗口也不足以给稳态吞吐结论。用途是确认当前数据流、复用与资源状态。最终无诊断 PI/HTTP/取消恢复复核结果见 §7。

### 2.2 当前 decode 与 prefill 的实际工作

本次诊断 16 个单行样本包含生成及 finalization，粗粒度中位数为：

| 字段 | 中位数 | 含义 |
|---|---:|---|
| submitted | 59.335 ms | backend 提交区间，不是纯 GPU 时间 |
| experts_host | 21.213 ms | 专家协调区间，包含 CPU/GPU 重叠 |
| cpu_compute | 13.459 ms | 冷专家 CPU 工作 |
| cpu_join | 9.396 ms | 主机等待 CPU 完成；不能与 compute 直接相加 |
| router_wait | 11.924 ms | 等队列中的路由发布，包含前序 GPU 工作；不是路由器独占成本 |
| CPU tasks | 240.5/token | 约一半路由落在 CPU；单行 staging 全部为 0 |

`[推算]` 一个 compact nested expert 为 **2,534,400 B = 2.416992 MiB**。若每个 CPU expert 的 packed payload 读取一遍，240.5 个任务对应约 0.61 GB/token / 13.46 ms ≈45 GB/s 的逻辑读取率。它不是 DRAM 硬件计数器结果，但结合既有 CPU 指令变体的中性结果，足以把“继续大改 AVX2 点积”降到低优先级。

本次**单独 13,979-token prompt** 的四块 prefill：冷专家上传累计 75,935 个、179.233 GiB；中间两个 4096 块 submitted 为 19.18 / 19.44 s。CPU experts 不参与这些大块。旧 VTune 已发现 MoE ALU 族约占全轮 GPU 时间 44%，compute 引擎持续饱和；因此大块 prefill 首选计算数据流优化。见 [旧剖析](qwen38-prefill-vtune-2026-10-10.md)。

**H2D/fill/bank_wait/router_wait 都有重叠，不相加。** 历史“310 GB”及 VTune“332.5 GB”是各自整轮口径，不能直接作为本次单个 14K 请求的传输量。

### 2.3 已执行专项检查

- CTest：SessionMemory、Qwen4ExpModelSpec、ExpertReader、InferenceProtocol，**4/4 通过**。
- `KernelTests --prefill-bench-only`：现有 prefill 路径及 WG-share 尾部映射逐位检查通过；teardown USM zero。
- `KernelTests --qsa-prefill-only`：BF16 / INT8 / multi-tile 通过；INT8 max_diff `3.05176e-05`；teardown USM zero。
- `CpuExpertTests --nf4-q8-activation64`：scalar-bitwise、1..4 token / 40 tasks、1/4/12 线程等现有契约通过。

微测 quads=160 时 share0/share4 分别为 2.20065 / 2.20056 ms，仅作当前工具可运行证据；不以这个小形状否定历史生产 share4 的收益。本次没有对新内核作性能比较。

## 3. 必须先修正的历史归因

### 3.1 H4 profile 是完整排名；INT8 缓存预算没有真正打通

`[本次实测/源码]` profile header 为 `48 512 24576`，本次验证 **24,576 个 key 全量、唯一、完整覆盖**。`Qwen4ExpHotsetPolicy::Load` 也要求 `n == layers*experts`。

`Qwen4ExpBnb4Backend.cpp` 的 `ranked_keys` 日志打印的是 `initial_ranked_profile.size()`，即**预算选出的前 N 项**，不是 profile 总行数。实际预算先把槽位定成 2792，再截取 2792 项。此前“profile 只排 2792，必须扩写到 2949”的解释错误；不需要补 key。

真正的遗漏在 `load()` 内专家缓存 admission：`kv_bytes` 仍固定 `sizeof(bf16)`。`Model.cpp` 的 session reservation 已支持 INT8，**这不等于 backend 专家预算已支持 INT8**。旧 INT8 诊断仍打印 `KV=768 MiB / slots=2792`，与源码吻合。

| Capacity | BF16 K/V | INT8 K/V + FP32 scales | 净省 | 理论额外专家槽 |
|---|---:|---:|---:|---:|
| 8K | 192 MiB | 97.5 MiB | 94.5 MiB | 39 |
| 16K | 384 MiB | 195 MiB | 189 MiB | 78 |
| 32K | 768 MiB | 390 MiB | **378 MiB** | **156** |

这是 payload 计算，尚未扣其他新增 metadata。约 2792→2948 是可测起点，不保证正好 2948；此前“省 384 MiB / 多 157 槽”漏了 INT8 scales。

另有一个同处的小问题：槽位公式固定减 32，而实际 staging 是 `2*staging_slots_per_buffer`。默认 16 时正确；研究配置 64 时少预留 96 槽、约 232 MiB。计划修预算时一并按实际字节计算。

### 3.2 reserve 的旧 logits 大项已过时

当前 `ensure_runtime_buffers()` 初始 logits 是 `[1,248320] BF16`，约 **0.474 MiB**；普通生成走 `lastTokenOnly`，中间 chunk 不算 head。完整多行 logits 只在显式请求该接口时增长。

因此旧资料“普通 agent logits 970 MiB”的 reserve 分解不能继续沿用。但 reserve3584 还包含 workspace 生长峰值、PLE、桌面/WDDM 余量，**不能据此直接减掉 970 MiB**。必须看当前实测峰值。

### 3.3 统一首字后速率公式与数值质量口径

以原始 `result.json` 和 `qwen38_context_probe.py` 为准：

```text
decode_rate = (completion_tokens - 1) / (duration_seconds - first_content_seconds)
```

旧 INT8 表的 15.583 等数字相当于使用 32 而非 31 作分子。按上述公式，四臂新14K速率是 BF16 **15.096/14.880**、INT8 **15.261/15.254**；相对增益仍约 1.8%，绝对 tok/s 应校正。

这组原始 SSE 还显示：首轮文本 hash 相同，**INT8 的续轮 hash 与 BF16 不同**。量化允许经过资格化的差异，但不能再称两臂全程逐位一致。详见配套 JSON 抽取与 [旧 INT8 报告](qwen38-int8kv-prod-2026-10-12.md)。

### 3.4 prefill 的后续硬件证据覆盖早期带宽归因

[管理员VTune与后续判别](qwen38-prefill-slmbound-2026-10-10.md) 的最终结论比早期“权重重读25遍”更具体：生产MoE DRAM读仅14–105 GB/s，线程占用约90%，SLM读高；但把LUT gather配对或移出SLM并未提速。因此**高SLM带宽也不足以证明SLM容量/带宽就是限制项**，load→解包→查表→FMA的依赖链仍只是待验证解释。

该记录已否定half-unpack、8token×half以及256项字节对LUT：前两者生产/小核回退，后者即使生成真实vec2 gather仍回退约9–16%。不重做这些候选。P4针对减少重复解量化及依赖，不以减少DRAM读取作为充分理由。

## 4. 先保留的运行配置

沿用 `tools/server/start-qwen38.ps1`：context32768、chunk4096、sessions1、decode-batch-size1、reuse-sessions1、CPU18、reserve3584、H4 profile、parallel staging copy、WG-share4；adaptive swap 默认 0，GDN chunk / GPU W4A8 / MTP 关闭，BF16 KV。

已保留改动包括 nested256、H4 初始驻留、staging 并行 memcpy、WG-share4、关闭同步 swap。不要把它们重复列成待开发；其中 swap-off 的既有生产收益约 +8.0%。分别追踪 [H4](qwen38-h4-residency-screen-2026-10-09.md)、[staging](qwen38-staging-copy-screen-2026-10-09.md)、[swap](qwen38-swap-longwindow-2026-10-10.md)。

Worker/模型持续运行，agent 用固定 endpoint；工具回合追加历史。已有 exact-prefix 复用的价值已经兑现：旧 PI 工具续轮缓存 6957 token，仅新增 32；旧 14K 续轮缓存 14011 token，仅新增 24。不要为每个工具回合重启模型。

`[源码]` Qwen4Exp 跨会话 decode batching 仍明确拒绝；不能套用 AGENTS 中其他 backend 的 B=4 收益。当前 RAM 峰值也不适合直接增加 sessions。

## 5. 实施顺序与可执行改动

| 优先级 | 工作 | 预估投入 | 采纳依据 |
|---|---|---:|---|
| P0 | 定位并改善大页启动失败；保证持久 Worker | 半天初筛 | 可稳定 ready、失败可解释、释放完整 |
| P1 | 守住真实工具回合的 exact-prefix 命中 | 半天 | 已命中路径不增加成本；遇到失效时有明确首分歧 |
| P2 | 修 INT8 / staging budget，再测当前 reserve | 1 天 | 账目正确；实际额外槽与端到端价值可量化 |
| P3 | 更新 agent 热集排名，保持 swap-off | 半天至 1 天 | held-out CPU miss 下降且回合更快 |
| P4 | 单个生产形状的 prefill 解量化复用探针 | 1–2 天 | 同数学完整专家链 ≥10%，再进实模 |

P0→P2 先完成；P1 若当前回合一直命中则只留轻量诊断，不另造缓存。P3/P4 分别初筛，失败即归档，不扩建调度框架。

### P0：启动稳定性先于微小 tok/s 收益

落点：`Qwen4ExpHostArena::AllocateStorage`、`memory/LargePages.cpp`、启动日志。

1. 让失败日志包含已分配 arena bytes/chunks、失败块 bytes、系统 available；失败路径立即输出，避免整段日志因 deferred logging 丢失。保持 GPU/RAM complementary owner 和 Context ledger。
2. 只测一个小改动：arena chunk256 MiB→64 MiB，仍用原生大页。比较完整加载耗时/峰值和连续启动；不预设缩块必然修好错误1450。embedding 的大页分配单独记录。
3. 若缩块仍失败，停止参数扫描。再评估**显式、记录数量的少量普通页尾部 fallback**；优先做 availability 选项，不能暗中把全 arena 变成 pageable RAM。需维持 native CPU-only 指针、小 USM staging、VirtualFree 和 rounded host budget；局部 CPU decode A/B 通过后才保留。

先跑小 HostArena ownership/release 检查，再跑完整模型启动和一次请求；GPU 压力进程串行。全量 pinned arena 已实测首请求 `UR_OUT_OF_RESOURCES`，不能作为本问题的替代方案。现阶段持久 Worker 可以减少重启暴露，但不是启动缺陷的代码修复。

### P1：复用现有缓存，定位具体失效

落点：`HttpServer::Service::TakeSession` / `RequestRun::releaseInference`，以及 agent 端历史构造。

当前成功路径不重写。只在诊断模式记录缓存长度、重渲染长度、首个不同 token 的位置、是否有 completed lease。用一次 read 工具回合、一次代码摘要续轮、一次含 reasoning 历史的续轮确认。日志留 token ID/索引即可，无需复制全文。

若 miss 来自模板/历史重建，修该处序列稳定性；若中途混入另一会话导致唯一 lease 被重置，先在单 agent 客户端串行化。**不要把任意 LCP 截断当成可用 KV**：DeltaNet/PLE 需要对应 recurrent/checkpoint state。暂停 prefix trie、多分支快照池等设计，直到有真实失效证据。

旧 [agent pacing](qwen38-agent-pacing-2026-10-07.md) 的 96-token 摘要续轮曾失去缓存；当前 8/32-token 续轮及 PI 已命中。这是覆盖缺口，不是已确认仍存在的同一个 bug。

### P2：用已有内存计算器修预算，不新增内存管理层

落点：`Qwen4ExpBnb4Backend::load` 的 cache fit、`memory/SessionMemory.hpp::qwen4exp_session_memory`；保留 Model.cpp 现有 reservation。

1. 调用同一份 session device payload 计算，传实际 BF16/INT8 元素宽度；包括 INT8 scales、indexer、recurrent、PLE state。以实际 `2*staging_slots*slot_bytes` 和 slot table 计 staging；避免固定减32。
2. 日志分别打印 `profile_total_keys / admitted_slots / initial_selected_keys / kv_bytes`，消除本次归因歧义。
3. A/B 两臂都设 `MAX_GPU_SLOTS=3200`，BF16 对 INT8；reserve/chunk/staging/profile 同一，预计 INT8 约多156槽。跑既有 SessionMemory、AdaptiveCache ownership/稳态零 USM 检查及生产序列。不要通过缩减 reserve 冒充 KV 释放。
4. 另起诊断轮用已有 `--memory-sample-ms` 和 DXGI 监测实际显存峰值，覆盖4096块、短追加、取消及恢复；USM 与 DXGI 分别记录。若余量充分，再做 reserve3584→3328 的单变量实验。预算 cap 必须在对照两臂都充分放开。

`[推算]` 历史槽位反向实验给约0.45–0.5%/100槽的局部弹性，156槽可能再贡献约0.7–0.8% decode。它不是实测增益，且不同热集边际价值不同。profile 训练集覆盖只从65.13%→66.94%，不能据此宣称把30% CPU流量搬走。P2 是小而明确的预算修正，不是 20 tok/s 的突破。

INT8 的新首字旧测略慢约1.1%，续轮已有文本差异；分别检查工具调用、停止条件和任务正确性。总回合没有稳定收益则继续 opt-in，不默认打开。reserve 的改动也以实测峰值及端到端结果决定。

### P3：按真实 agent 工作量更新离线排名

现有 H4 来源是3类 prompt、每类2次、每次48输出。它有效，但不能代表长期代码 agent；当前 decode 仍约240个 CPU任务/token。

在 adaptive 实际 routing 观察点收集 decode/prefill 两组计数，启动预分配、收尾导出，复用当前 identity / 完整24576-key profile 格式。现有 legacy `HotsetPolicy::Save` 不能直接当成 adaptive 已接通的 exporter。

只选少量真实代码/工具输入校准，再用独立14K和PI工具回合验证；A/B相同槽数、相同swap-off，防止把扩容混进排名实验。可以在请求边界停服务后加载新profile，第一轮不做在线迁移/后台换入。

采纳看 held-out CPU miss、decode p50/p90 和工具回合耗时。CPU/GPU当前数学路线有差异，placement 改变不能自动继承逐位等价：需要检查工具行为、固定输入 logits/生成质量；不得根据生成得更短声称更快。只有命中率升、墙钟不升时不继续加机制。

### P4：只做一个同数学 prefill 解量化复用候选

落点：`RunAdaptivePrefill → RunGroupedExperts → moe_expert_major_prefill_gate_up/down`。优先 gate/up，它在旧 VTune 中比 down 更重。

保持 NF4+nested、BF16激活、每线程4-token累加、SG16、原 K 累加/归约/ordered rank 次序。候选只改变**同专家多个quad的解量化供给**：先保证参与共享的quad确实同专家，由一组线程完成NF4解包、LUT查表、原scale重建及FP32权重乘法，把小tile的**FP32解量化结果**放入SLM，多个quad消费。只缓存packed权重并不会消掉查表依赖，不作为候选。

`[待验证]` 从4输出行×4quad、K128开始，gate/up的FP32 tile约4KiB；按lane交错存储，不能照搬已失败的vec8输入SLM布局。每quad保留原累加器与FMA顺序，小桶/尾行沿用同数学原路径。局部FP32 tile不改变cache格式，不生成全局BF16/INT8权重副本。新增barrier、SLM读取和占用可能抵消减少的解包/LUT工作，**先测完整链再决定是否存在收益**。

生产布局必须遵守：high nibble first；gate/up `[1280,1280 bytes]`、down `[2560,320 bytes]`；block64；nested block256；scale保留既有两次FP32 rounding。技能中其他 NF4 示例的 low-nibble 布局不能照抄。

第一轮只测 E=16、H2560/I640、top-k10、每专家约80/128行，包含 packed source→staging→gate/up→SwiGLU→down，resident/cold 与 ragged 尾行都有参考校验。新 kernel 与已有同数学 ALU 路线比较；若加入库对照，oneDNN affine u4 不等同 NF4，只能用同一反量化参考建立可比性。

先要求全输出逐位一致、零新 USM、完整链 wall ≥10% 改善；再做大块 submitted ≥1% 和生产 TTFT ≥2% 的实模验证。新实验3预热、至少20交错样本、3轮、median/p10/p90；小核同时看event device和submit→wait wall。编译 spill/VTune 可解释收益后才扩展第二个shape。失败即结束这个候选。

如果完整链不达门，仍有一个较小备选：`RunGroupedExperts` 当前每批重新扫 `count*top_k` 全路由；可在本层一次建立按expert的token/rank桶并让microbatch切片。只保持原map顺序、不改数学。先量主机map耗时，处于GPU饱和的重叠区就不做；不把理论扫描次数减少当收益。

## 6. 第一轮不做的方向及原因

| 方向 | 决定 / 可追踪依据 |
|---|---|
| MTP | 续轮状态等价门未过，已有15.0–15.3对T1约15.6的旧口径也净负；保持关闭。[对齐](qwen38-mtp-align-2026-10-11.md)、[状态](qwen38-roadmap-status-2026-10-12.md) |
| ESIMD 替换 SG8 DPAS | **微基准GO已被真实shape NO-GO覆盖**：8个网格仅0.67–1.01×，含scale gate/up 0.65×；不执行旧“过门后接入清单”。[最终探针](../../tmp/q4e-esimd-dpas-20261012/performance.md) §9 |
| INT8 attention DPAS | 正确性修复不等于性能采纳；14K慢于现有vector INT8，保留vector路线。[修复](qwen38-int8-dpas-repair-2026-10-12.md) |
| prefill ALU 改Q8激活 | 当前不是激活流量瓶颈，历史估计收益≤0.7%；不继续。[上界](qwen38-w4a8-prefill-bound-2026-10-12.md) |
| 8-token / share8 / grouped XMX旧实现 | 前两者spill/缓存局限、后者旧shape回退；不原样重试。[token8](qwen38-token8-window-screen-2026-10-10.md)、[WG-share](qwen38-wgshare-screen-2026-10-10.md)、[X3](qwen3.8-flash-next-x3-results-2026-09-29.md) |
| half-unpack / 字节对LUT | 已有位精确实现仍回退；高SLM读量不能单独指导重构。[最终判别](qwen38-prefill-slmbound-2026-10-10.md) |
| GDN chunk默认启用 | 旧TTFT−2.7%，decode−4~7%，数值改变后的路由影响；短答可opt-in，通用agent不默认。[状态](qwen38-roadmap-status-2026-10-12.md) |
| chunk8192 / staging64 / ring4 | 旧70.6→89.2s、加深staging无净收益、ring旧负结果；保留4096/16。[路线图](qwen38-optimization-roadmap-2026-10-10.md) B3 |
| 全量 pinned / BF16 / INT8 权重副本 | 约51GiB pinned首请求失败；解量化或扩展权重格式破坏16G/64G容量优势。保持GPU/RAM互补 |
| 通用graph、跨层预测router、跨会话batch | 当前动态专家和真实依赖不支持直接移植旧小核/其他模型结果；先做可测单点 |

**layer-major 是后续设计题，不是已证明可赚13–18%的实施项。** 当前chunk-major按“chunk0全48层→chunk1全48层”运行；只新增一个1.1GiB单层expert bank，无法跨中间47层留住所有层权重，必须改变forward驱动。旧8192实验已经证明流量减少可被bank等待/计算等成本抵消。

若P4完成后仍以长prompt为主要成本，再单独立一个“两块4096、保持每块原attention/GDN数学”的layer-major原型：只复用一层冷专家供给，保存必要HC hidden块，量实测峰值与完整墙钟。当前单14K H2D179GiB，RAM补集51.46GiB只是“一次传每个cold key”的理论量级；节省多少最终要按每层union测量。涉及调度/取消/状态提交的多日重构，第一轮不启动。

## 7. 验收、复现与交接

方案只保留少量可判定实验。计算数值不变的候选先逐位检查；INT8/placement变化另按任务及logits质量验证，不能放宽旧断言。生产对照仍用固定fixture、单变量A/B/B/A；需要推广时再补第3稳定轮。计时关闭diagnostic/profile/trace，诊断另跑。

验收关注四个数：新14K TTFT、真实工具续轮新增token及TTFT、256-token decode延迟分布、完整PI工具任务耗时。再加启动/peak RAM/DXGI和取消恢复即可，不增加全套边界矩阵。32-token旧样本用于初筛，不能支撑长期生成采纳。

当前无需修改配置即可运行：

```powershell
# 实际服务入口：持续运行模型，单agent顺序追加历史
pwsh -NoProfile -File tools/server/start-qwen38.ps1

# 无诊断生产基线；使用新tag，runner保护旧目录
python tmp/q4e-resident-speed-20261009/run.py <new-tag> --agent `
  AILA_Q4E_STAGE_COPY_PARALLEL=1 AILA_Q4E_4TOKEN_WGSHARE=4 `
  AILA_Q4E_ADAPTIVE_SWAP_FREQ=0 AILA_Q4E_GPU_Q8_ADAPTIVE=0 AILA_Q4E_GDN_CHUNK_MODE=0

# INT8对照：预算修复后，两臂都用相同MAX_GPU_SLOTS=3200
# BF16臂不传AILA_SERVER_KV_TYPE；INT8臂增加如下两个键
python tmp/q4e-resident-speed-20261009/run.py <int8-tag> --agent `
  AILA_Q4E_STAGE_COPY_PARALLEL=1 AILA_Q4E_4TOKEN_WGSHARE=4 `
  AILA_Q4E_MAX_GPU_SLOTS=3200 AILA_SERVER_KV_TYPE=int8

# 小范围CPU与内存计数器检查
ctest --test-dir build-refactor -R '^(AilaSessionMemoryTests|AilaQwen4ExpModelSpecTests|AilaQwen4ExpExpertReaderTests|AilaInferenceProtocolTests)$' --output-on-failure

# 此次临时探针：串行执行既有3个专项入口；不替代生产A/B
python tmp/q4e-agent-plan-audit-20261012/run_operator_checks.py
python tmp/q4e-agent-plan-audit-20261012/audit_inventory.py
```

`run_operator_checks.py` 使用独立log的`xb`保护旧结果；再次执行请换证据目录或新脚本副本。env在子进程清理并重新建立，read-once开关用独立进程A/B。

## 8. 执行记录（2026-10-13，主线程；提交 `548b0d8`，pair `retained12`）

用户裁决：P0 的启动失败很可能是启动瞬间运行 mem-reduct 清理碎片造成的 → **只加日志，不做 chunk 大小扫描**；其余按计划启动。

### P0（日志，已落地）
`Qwen4ExpHostArena::AllocateStorage` 的分配失败路径现在**绕过 deferred logging 直接写 stderr**，并带：
`chunks / allocated / chunk_bytes / per_chunk / count / bytes / resident_experts / host_avail / host_total / pinned`；
同一组数字也追加进返回给调用方的 error 字符串。**未做任何 chunk/分配策略改动**（256 MiB 保持）。

### P2（预算修正，已落地并实测）
1. `SessionMemory.hpp` 新增单源 `qwen4exp_kv_bytes(config, capacity, kvElementBytes)`（K/V + int8 的 per-token
   F32 scale 对），`qwen4exp_session_memory` 与 backend admission 共用；backend 判定与 session builder 完全一致
   （`AILA_KV_QUANT && kv_quant_format_is_int8()`）。
2. staging 预留从固定 `-32` 改为 `2 * staging_slots_per_buffer`；residency 日志拆成
   `profile_total_keys / ranked_keys / admitted_slots / kv_element_bytes`。
3. 实测（`p2-bf16b` / `p2-int8`，两臂同 `MAX_GPU_SLOTS=3200`）：

| 臂 | Session State | KV | scales | slots |
|---|---:|---:|---:|---:|
| BF16 | 1000.2 MB | 768.0 MB (element_bytes=2) | 0 | 2792 |
| INT8 | 622.2 MB | 384.0 MB (element_bytes=1) | 6.0 MB | **2948** |

   `profile_total_keys=24576` 证实 profile 本就是全量；此前"只排 2792、需重生成"的结论作废。
4. 端到端 A/B/B/A（修正口径 `(ctok-1)/(dur-first)`）：14K ttft BF16 70.586/70.579 vs INT8 71.180/71.145
   （**−0.8%**）；续轮 rate 15.590/15.599 vs **15.720/15.862（+0.8~+1.7%，区间不重叠）**；14K 32-token decode
   落在自身方差内。即 **+156 槽的结构收益兑现，但端到端只有 ~1%**（与 §5-P2 的 0.7–0.8% 推演一致）。
5. 附带更正：INT8 与 BF16 首轮文本 hash 相同、**续轮 hash 不同**（此前"两臂全程逐位一致"作废）；
   绝对速率按 `(ctok-1)` 口径重算。

### P4（prefill 解量化复用，已完成 2026-10-13，结论 NO-GO）

按计划 §5-P4 只做"同专家多 quad 共享 FP32 解量化 tile"一个候选，三级判据全部落地：
oracle 探针 → 单变量判别探针 → 逐位精确实现。完整证据见
[P4 SLM 解量化复用探针](qwen38-p4-slmshare-2026-10-13.md)。

| 臂（share=4，gate/up only，median of 3 轮交错 ×20 次） | tokens=16 | tokens=64 |
|---|---:|---:|
| 原始 wgshare | 0.4017 ms | 1.3809 ms |
| `_nounpack` oracle（去掉 packed 读+解包+LUT+mul） | 0.1300 ms（−68%） | 0.6136 ms（−54%） |
| `_nolut` 判别（只把 LUT gather 换成 int→float） | 0.3469 ms（−13.6%） | 0.9244 ms（−33.1%） |
| `_slm` 实现（逐位精确，share-4 SLM tile） | 0.4472 ms（**−11.1%**） | 1.6041 ms（**−17.7%**） |

1. 逐位一致 ✓（tokens=64 全覆盖 PASS；tokens=16 遗留单 act 槽位两臂均不写，预填充后 PASS），
   零新 USM ✓（只用 SLM），但**完整专家链 wall 为负**，未过 ≥10% 门。
2. 机理：消费者仍要每 lane 每 chunk 发 16 次 SLM 读（与原 16 次 LUT gather 同量），SLM 方案只能省
   掉 3/4 的 nibble 解码+mul；而 `_nolut` 证明在 tokens=64 时"可共享部分"上限≈33%（gu 内），
   tokens=16 时仅 13.6%（低于 10% 链门本身），其余是被保留的 packed-load 依赖链（不可共享）。
   生产者/消费者分相 + 每 tile 2 次 barrier 的固定开销直接把净收益压成负值。
3. 不做后续：不试 double-buffer/phase 交织救援（上限也不能过门）、不试 share=8/16、不接入
   `RunGroupedExperts`。三个诊断内核保留为 opt-in 测试路径（`AILA_Q4E_NOUNPACK_PROBE` /
   `AILA_Q4E_NOLUT_PROBE` / `AILA_Q4E_SLM_REUSE`，生产代码零引用）。
4. 由此关闭"靠减少重复解量化提升 prefill"方向；若 prefill 仍是主成本，按计划改走
   layer-major 原型（§5 末）或 P1 客户端/服务端路径收益。

### P1（prefix-reuse 轻量诊断，已完成 2026-10-13）

- **实现**：`AILA_SERVER_PREFIX_DIAG=1`（默认 off，只读诊断、不改复用决策）——服务端在
  `TakeSession` / `KeepSession` / `releaseInference` 三处打 WARN 级日志：缓存长度、prompt 长度、
  **首个分叉 index 与两侧 token id**、`qualifying`、`idle` 数、`max_sessions`、keep 判定输入。
- **现场对照**（同 fixture、单变量只改 `--generate`）：

| 场景 | 14K 完成时 | 续轮 take | 结果 |
|---|---|---|---|
| `--generate 32` | `ctx=14011 == completed=14011, keep=1` | prompt 14035 vs cached 14011 → **diverge_at=14011=len（qualifying）** | 复用 ✓（cached 14011） |
| `--generate 128` | `ctx=14107 == completed=14107, keep=1` | prompt 14133 vs cached 14107 → **diverge_at=14047**（cached_tok=13962, prompt_tok=2） | 不复用（cached=0）→ 续轮整段重跑 14K prefill |
- **根因（2026-10-13 修正）**：服务端 Take/Keep 判定链正确，客户端也无责（续轮 assistant 文本与 SSE
  逐字节相同）。失效点是**服务端自带 tokenizer 的 merges 加载缺陷**：`src/utils/Tokenizer.cpp:449`
  把 `merges.txt` 中以 `#` 开头的 83 条合法合并规则当注释跳过（247587→247504 条），`##`/`###`/`# include`
  等无法合并；模型生成的 `'###'`(13962) 被重编码成 `'#'(2)×3` → `prompt[14047] ≠ cached[14047]` → 不复用。
  最小修复=只跳过 `#version` 头行；tmp 修复副本已验证 `###`→13962、assistant 重编码 130→128 token。
  另有独立的 `pretokenize` 空白/标点边界与 tokenizer.json 不一致（等长换 id），生成区内命中 content
  120-121 一处，是否影响本次复用需修复后 live 复跑判定。详见
  [P1 分叉定位](qwen38-p1-reencode-divergence-2026-10-13.md)。
- **影响与路径**：agent 在长生成后立刻续轮会整段重跑 prefill（本机 14K ≈70 s）。当前路径：
  ① 采纳 `Tokenizer.cpp:449` 一行修复并重建 pair，按同一 fixture 复跑确认 `qualifying=1`；
  ② 若仍在生成区 miss，按新文档的残余清单继续对齐 `pretokenize`。边界不变：不引入 prefix trie /
  多分支快照池，不对 DeltaNet/PLE 做任意 LCP 截断。原始现场：
  `tmp/q4e-agent-plan-audit-20261012/p1-g128/{stdout,stderr}.log` 与
  `tmp/q4e-resident-speed-20261009/p1-g32/…`。

### P3（agent 工作负载热集排名，已完成 2026-10-13）

1. **exporter 接线**（`Qwen4ExpBnb4Backend`）：`AILA_Q4E_HOTSET_EXPORT=<path>`（默认空、仅 adaptive 生效）在读 profile 的
   同一处计算 identity；在三个真实路由观测点（decode per-token ×2、prefill per-layer ×1）旁调用
   `Qwen4ExpHotsetPolicy::Observe`；teardown 时 `Save` 出**全量 24576-key** profile（格式/identity 与 H4 一致，
   `Load` 可直接读）。策略对象启动时预分配，观测为纯计数。
2. **校准**（6K prompt + 256-token 生成，独立于验证工作负载）：导出 `calib-6k-g256.profile`——
   decode 观测 245,760、prefill 5,864,640、decode>0 的 key 13,958；top key 389 次。已入库为
   `perf/qwen4exp/profiles/qwen38-nf4-agent6k-v1.profile`（与 H4 同 identity）。
3. **held-out 验证 A/B/B/A**（标准 14K+PI，128-token 解码样本，**两臂槽数同为 2792**、swap-off）：

| 指标（decode 中位） | A1 H4 | B1 agent6k | B2 agent6k | A2 H4 |
|---|---:|---:|---:|---:|
| CPU misses/token | 255.5 | **175.0** | **180.0** | 255.5 |
| hits/token | 224.5 | 305.0 | 300.0 | 224.5 |
| experts_host ms | 21.00 | 17.00 | 17.11 | 21.02 |
| submitted p50 ms | 60.17 | 58.14 | 58.11 | 60.24 |
| submitted p90 ms | 64.11 | 62.49 | 62.64 | 65.08 |
| PI 工具回合 s | 232.09 | 233.40 | 233.34 | 233.62 |

   **misses −31%、experts_host −19%、submitted p50 −3.5%（区间不重叠、两轮复现）**；工具回合耗时无变化
   （prefill 主导）。满足 P3 的 held-out 采纳观察项。
4. **附带发现（转 P1）**：把生成上限从 32 提到 128 后，续轮 prefix 复用失效（`cached_prompt_tokens` 14011→0），
   即 `TakeSession` 的"整段缓存 token 必须是新 prompt 前缀"在更长生成下被破坏（疑似 detokenize→re-encode 边界）。
   本 P3 的 per-token decode 统计不受影响（两臂同等条件），但续轮时延指标需用 32-token 口径复测。
5. **建议**：`agent6k` profile 先作为 **opt-in**（`AILA_Q4E_HOTSET_PROFILE=perf/qwen4exp/profiles/qwen38-nf4-agent6k-v1.profile`）；
   默认仍用 H4。扩大校准语料（真实代码/工具回合）并再做一轮 held-out 验证后，再考虑切换 `start-qwen38.ps1` 的默认 profile。

本次原始现场：

- 首次启动失败：`tmp/q4e-resident-speed-20261009/audit-20261012-b463802-baseline/`及同名monitor/run log。
- 实模诊断通过：`.../audit-20261012-b463802-startup-diag/`及`-worker.log`；不是生产计时。
- 当前无诊断PI/HTTP/取消恢复复核：`.../audit-20261012-b463802-production/`；结果在配套JSON。
- 专项原始日志和临时探针：`tmp/q4e-agent-plan-audit-20261012/`。
- 可随方案保存的机器可读证据：[配套JSON](qwen38-a770-agent-performance-plan-2026-10-12.json)。

### 本次无诊断生产复核最终结果

同一Worker/Server，固定fixture，32K capacity / chunk4096 / BF16 KV / 32输出；先PI、再取消恢复、再新14K与续轮。**全序列通过**，进程exit0，graceful shutdown，无forced kill、无残留Worker/Server。ready120.19s，PI工具任务107.05s，全探针310.08s；系统available最低3.21GiB。

| 请求 | 实际prompt | 缓存token | TTFT | 首字后速率 | 完整请求时间 |
|---|---:|---:|---:|---:|---:|
| 新14K | 13,979 | 0 | **68.137s** | **14.702 tok/s** | 70.246s |
| 续轮 | 14,035 | **14,011** | **1.165s** | **15.627 tok/s** | 3.149s |

PI成功read一次并返回固定marker；工具续轮复用6957、新增32。取消后的新请求正确返回READY。两条摘要文本hash分别与旧BF16对照相同；最终cache hits=2、reused_tokens=20968。这是**一次当前基线复核**，没有新生产实现，不宣称相对旧70.6s加速，也不把32-token速率外推为长生成吞吐。

后续完成一个改动就记录：原始命令、exe SHA、样本、输出/任务质量、是否保留。先把P0/P2的确定缺口关掉，再决定P3/P4；没有数据支持时不堆叠更多新开关。
