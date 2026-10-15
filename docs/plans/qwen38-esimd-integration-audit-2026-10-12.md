# Qwen3.8 prefill MoE：ESIMD / SG8-DPAS 接入审计与验收方案

日期：2026-10-12。审计目录 `E:\RiderProjects\Aila`；核对 HEAD 为 `6b7286a4636b00ae25c98066c551b97eeb8ad6b3`。下列行号均为本次工作树读取位置，后续补丁应按函数名重新定位。

本次只读源码、文档与已有日志；没有构建、执行测试、运行 GPU 或提交。唯一写入为本文件。工作树原有未跟踪方案文档保留。另一个代理负责 `tests/ops/Qwen4ExpEsimdDpasProbe.hpp`；初次查读时该文件尚不存在，收尾时已观察到该文件及 KernelTests 的并行探针改动，均未由本审计修改。本文不审查其未完成的实现，也不声称其布局、正确性或性能已过门；KernelTests 引用采用本次先前读取的位置，合并探针后需按函数名重定位。

先读了 [ESIMD 分阶段方案](qwen38-esimd-path-2026-10-12.md) 与 [SG8 INT8 attention 修复](qwen38-int8-dpas-repair-2026-10-12.md)。下文 `[源码]` 是当前实现事实，`[已有记录]` 是历史结果，`[方案/待验证]` 是未来补丁及门，不是本次测量。

## 1. 接入结论

推荐在 **`RunAdaptivePrefill → RunGroupedExperts` 的单专家桶边界**引入独立、默认关闭的 prefill W4A8/DPAS 路由，先 gate/up、后 down；保持 NF4 原始权重与 adaptive cache 主格式，采用执行槽拥有的有界 INT8 scratch。不能直接把 wgshare workgroup 改为一个 M8 GEMM：相邻 quad 可以来自不同专家，原型的等长专家数组也不等于生产 ragged map。

必须先分开两种契约：

- **执行实现正确性**：s8×s8→s32、BNB nibble、VNNI、每 K64 的 scales、token/rank scatter 与独立整数/FP64 参考一致。
- **相对 NF4 的数值质量**：整数 LUT 与 Q8 激活均引入近似；不能继承 wgshare 的位精确契约。DPAS 原始吞吐过门不足以允许生产启用。

另一个关键门是前缀因果：**专家行数、resident/cold 状态、microbatch 切分只能选择执行方式，不能决定该前缀是否量化**。同一已启用数值策略中的小桶/尾桶需用同算术向量实现或安全补零 tile，不能简单退回 BF16/NF4；不同执行方式的 K64 缩放与累加次序也要对齐。关闭候选或整请求不支持时仍使用完整原路径。

采纳以完整成本计：块级耗时下降 ≥1% 或生产墙钟下降 ≥2% 是前置文档的 **OR** 门；达到块级门但墙钟没有收益，只足以保留 opt-in 算子候选。建议默认生产推广额外要求墙钟 ≥2%、质量/状态/生命周期全部通过且其他场景无可重复回退，不把裸 DPAS 速度当采纳依据。

## 2. 生产真实 ABI 与布局

### 2.1 描述符不是 INT8 权重描述符

`src/ops/MoeExpertOps.hpp:16–31`：

```cpp
struct MoeNestedScaleDev {
    const uint8_t* codes;
    const float* nested_absmax;
    const float* nested_quant_map;
    float offset;
    int blocksize;
};
struct MoeExpertSlotDev {
    const uint8_t* gate_up_packed;
    const float* gate_up_absmax;
    const uint8_t* down_packed;
    const float* down_absmax;
    MoeNestedScaleDev gate_up_nested;
    MoeNestedScaleDev down_nested;
};
```

这里是内部 C++/device 参数 ABI，不是公共 C ABI。设 H=hidden、I=expert intermediate、R=top_k、C=窗口 token 数：

| 字段/缓冲 | 布局和语义 |
|---|---|
| `gate_up_packed` | NF4 行主序 `[2I,H/2]`；前 I 行 gate，后 I 行 up；每 byte 的高 nibble 是先一个逻辑权重 |
| `gate_up_absmax` | 非 nested 路径的 FP32 `[2I,H/64]` |
| `down_packed` | NF4 行主序 `[H,I/2]` |
| `down_absmax` | 非 nested 路径的 FP32 `[H,I/64]` |
| `*_nested.codes` | 每 64 个权重一个 scale code；分别 `[2I,H/64]`、`[H,I/64]`，不是权重 nibble |
| nested map/absmax/offset | map 有 256 个 FP32 元素；nested absmax 对每 256 个 scale code 分组，外加 scalar offset |
| `x` | BF16 `[C,H]`，gate/up 按 `token_ids[a]` gather，与 rank 无关 |
| `h` | BF16 `[C,R,I]`，存储 SwiGLU 后结果；地址 `(token*R+rank)*I+n` |
| `down_acc` | 默认 FP32 `[C,R,H]`；没有在投影里乘路由权重，也没有原子累加 |
| maps / slots | 每个 activation 一条 token、rank、slot；传入指针可以是一个 microbatch 的切片 |

生产 adaptive 描述符由 `Qwen4ExpAdaptiveCache.cpp:310–339` 的 `make_slot/GetResidentSlotDev/GetStagingSlotDev` 构造：只设置 packed 指针和 nested codes，FP32 `*_absmax` 保持 null；nested_absmax 指针偏移到当前专家，map/offset 取 layer metadata，blocksize 强制 256。`Qwen4ExpHostArena.cpp:129–143` 在加载时验证权重 blocksize=64、nested blocksize=256、codes 长度可按 256 切分。

**不要混淆 compact nested payload 与 compact activation output。** 当前权重确实是 compact nested；`moe_compact_tile_` 在 `Qwen4ExpBnb4Backend.cpp:366` 被固定为 0，生产输出仍是 ordered rank buffer。

### 2.2 两个 launch 模板的全部参数

`src/ops/MoeExpertOps.cpp:2416–2430`：

```cpp
template<bool UseNested, bool FixedNested256=false, int ShareQuads=2>
sycl::event launch_moe_expert_major_prefill_gate_up_4token_wgshare(
    Context& ctx, const MoeExpertSlotDev* slots,
    const int* token_ids, const int* ranks, const float* quant_map,
    const bf16* x, bf16* h, int total_quads, int top_k,
    int in_features, int intermediate_size, int blocksize,
    const sycl::event* dep_event=nullptr);
```

`src/ops/MoeExpertOps.cpp:2845–2859`：

```cpp
template<bool UseNested, bool Compact=false,
         bool FixedNested256=false, int ShareQuads=2>
sycl::event launch_moe_expert_major_prefill_down_4token_wgshare(
    Context& ctx, const MoeExpertSlotDev* slots,
    const int* token_ids, const int* ranks, const float* quant_map,
    const bf16* h, float* down_acc, int total_quads, int top_k,
    int in_features, int out_features, int blocksize,
    const sycl::event* dep_event=nullptr);
```

Gate/up 生产值是 in_features=2560、intermediate_size=640；down 是 in_features=640、out_features=2560；两者 blocksize=64。公开 wrappers 在 `.hpp:488–523`、`.cpp:4278–4416`，附带 `use_nested`、`share_quads`、`nested_fixed256` 参数，share 2/4/8 选择模板，否则回落普通 4token。模板本身没有强制 H=2560/I=640 或 blocksize=64 的 runtime guard，不能把默认参数误称已执行的几何守卫。

WG=256、SG=16，`rows_per_wg=16/ShareQuads`；gate/up 的 groups 为 `ceil(I/rows_per_wg)*ceil(total_quads/ShareQuads)`，down 将 I 换为输出 H。每 SG 独立负责一个 `(quad,输出通道)`；quad 从 `slots[4*quad]` 取权重（gate/up `.cpp:2477`、down `2902`）。**只有 quad 内 4 行同专家是调用方保证；不同 SG/quad 不保证同专家。** WG-share 是重排映射以改善复用，不是显式 SLM 缓存整个权重 tile。

权重 uint32 每次取 8 个 NF4 值；输入 BF16 vec8；SG16 lane 从 `lane*4` packed-byte offset 起，stride=64 bytes。blocksize=64 时每 32 bytes 共用一个 absmax，lane 0/8 分别读取两个 K64 scale，再向半组 broadcast（gate/up `2492–2525`，down `2920–2939`）。顺序 FMA 与 `reduce_over_group` 保留原 4token 次序；gate/up 执行 native exp 的 SiLU×up 后写 BF16（`2624–2660`），down 写 rank FP32（`2994–3019`）。

`load_expert_absmax<UseNested,FixedNested256>` 在 `.cpp:405–425`：

```text
j = output_row*(K/64) + k/64
nested_block = FixedNested256 ? (j >> 8) : j/nested.blocksize
product = fma(nested_quant_map[codes[j]], nested_absmax[nested_block], 0)
absmax = fma(product, 1, offset)
```

非 nested 返回 `absmax_f32[j]`。两步 FP32 rounding 是现有契约，不能改成一次 `fma(map,absmax,offset)`；源码明确说明 1 ULP 可经 48 层影响 greedy routing。`quant_map=nullptr` 使用默认 16 项 NF4 LUT（`2461–2463/2890–2892`）。

Down 模板 `Compact=true` 时输出 `[local_activation,H]`（`3003/3008/3013/3018`），h 输入仍 `[token,rank,I]`。但公开 wgshare wrapper 的 down 实例均是 `Compact=false`（`4368–4401`）；generic compact dispatcher 的 quad 分支却调用这个 wrapper（`4707–4713`）。因此 **wgshare+compact output 没有可信支持证据，存在 dense scatter 写入 compact arena 的风险**；当前 `moe_compact_tile_=0` 使其不是本次生产主路径。新候选第一阶段必须拒绝 compact output，不将此组合列为已支持，也不在本次改源码。

### 2.3 选择链、env、几何与 microbatch

`ForwardImpl` 在 `Qwen4ExpBnb4Backend.cpp:2249–2254`：adaptive allocated 且 count>4 调 `RunAdaptivePrefill`；count≤4 调已有 `RunResidentDecode`；不支持的 legacy 分支抛异常。生产 `s_expert_major=true` 固定在 `ensure_runtime_buffers:894`，旧 `AILA_Q4E_EXPERT_MAJOR_PREFILL/DEMAND_MICROBATCH` 不能当当前选择主路径的有效开关。

`RunAdaptivePrefill:1369–1418`：

1. 统计整个 layer/window 中每专家的 routing use count；先 resident hits，再 cold misses，构造 unique experts。
2. resident 等 `ResidentReady()`；cold 每批 `min(StagingSlotsPerBuffer(),剩余unique)`，`StageColdExpertsAsync` 返回 ready_event，两个 staging bank 交替。**每个 cold 专家每 layer/window 上传一次，不按每个 quad 上传。**
3. 每批 `RunGroupedExperts` 后记录 resident/staging consumer；最终复制 routing weights，`moe_ordered_accumulate_down_rows` 按原 rank 次序合并，保持 shared expert 融合参数。

`RunGroupedExperts:1421–1525`：

| env/守卫 | 当前作用 |
|---|---|
| `AILA_Q4E_WEIGHT_REUSE` 默认 true | 允许 pair/quad/eight 分桶 |
| `AILA_Q4E_4TOKEN_REUSE` 默认 true | 默认 quad 路径 |
| `AILA_Q4E_8TOKEN_REUSE` 默认 false | 开启后先取每专家完整 8 行，再 4/2/1 尾段 |
| `AILA_Q4E_GROUPED_XMX_MIN_ROWS` 默认 0 | 0 不选 grouped BF16 XMX；正值且 `grouped_xmx_supported_` 才将该专家所有行分给 XMX |
| `grouped_xmx_supported_` | load:378 按设备名含 `Arc(TM) A770`；它是 BF16 grouped 路由条件，不是 INT8 DPAS capability 认证 |
| `AILA_Q4E_4TOKEN_WGSHARE` 默认 0 | ops:4482–4487 read-once，仅 2/4/8 有效；generic quad 分支 gate/up:4560、down:4707 选择它 |
| `AILA_Q4E_STAGING_SLOTS` 默认 16 | load:720–722 控制 cold unique experts 批容量；不是 token 数，且不等于旧 demand_microbatch=32 |

ALU 布局为 **eights→quads→pairs→singles**，每段内按专家连续；XMX 段位于 ALU 之后，`MoeExpertTile{first,rows}` 每 tile≤8（`1451–1455`）。map 内容通过原 token-major router 顺序填写，rank=`i%top_k`、token=`i/top_k`（`1467–1485`）。每 microbatch 使用不重叠 host/device slice，host 描述符在 DMA 完成前不可覆盖（`1488–1494`）。

generic ops dispatcher `4491–4634/4637–4791` 将 sections 分开提交，并让后续 section 继承已提交 event。现有 wgshare 没有 DPAS capability 或 M≥8 检查；几何合法性依赖生产配置。新的 DPAS 应在 backend 保留单专家边界后形成独立 tile，不能用 `total_quads` 聚合值作单专家 M。

`Qwen4ExpPrefill.cpp:557–575` 是 attention tiling；`578–600/RunCompactExpertDown` 是按 quad/pair/single 单位切 compact down tiles 与 merge。**它不是当前 adaptive prefill 专家 dispatch 所在文件。** 当前 compact 固定关闭，首期接入不应把该 helper 当主入口。

## 3. 原型的数据流、额外成本及生产落点

### 3.1 现有 SG8 原型到底做了什么

`tests/models/Qwen4ExpDpasMoeProbe.hpp` 是 ordinary SYCL SG8，不是 ESIMD：

| 函数/行号 | 数据流与限制 |
|---|---|
| `integer_dpas_stage<Gate>:7–25` | 每专家取 `slots[e*rows]`；uint32 NF4 load→SLM integer LUT→8 个 s8，写 W `[E,N,K]` row-major；每 offset%8==0 重建一项 per64 FP32 scale `[E,N,K/64]`；Gate 的 N=2I，down N=H；只支持 nested scale，没有非 nested 分支、显式 dep 或泛化尾守卫 |
| `integer_dpas_pack<Gather,Pad>:29–43` | uint32 读 4 个连续 Q8，写 `[E,K/4,stride_rows,4]`；Gate Gather 用 tokens[a] 取 xq；可同时写 dense 镜像（仅库比较需要）；scales 按 `[E*rows,K/64]` gather；Pad 仅把物理 row stride 向 32 对齐，并未初始化 padding 或多算 padded rows |
| `integer_dpas_projection<Gate,WG,Pad>:45–101` | rows%8==0、K%64==0、outputs%8==0；SG8 M8/K32/N8；矩阵 A 是 8 个输出通道的 W，B 是 8 个 activation 的 VNNI；算出 `[output_channel,activation]`，最后 scatter 回 `[token,rank,outputs]` |
| projection:64–87 | 每 K64 先清 s32 accumulator，做两次 K32；随后 `float(dot)*ws*xs/127` 加到 FP32，gate/up 分别取 scale；不允许全 K 整数和后统一缩放 |
| projection:89–99 | 同核 SiLU×up，BF16 h 边界，down FP32；route weights 不在这里 |
| smoke:103–123 | M8/N8/K128、非恒定 scales、正负 s8、置换 token；独立 CPU reference，abs≤1e-5+1e-5*abs(ref)；不是 NF4 模型质量认证 |

integer LUT 不是 nibble 值本身：`Qwen4ExpIntegerMoeProbe.hpp:4–6` 为 `{-127,-88,-67,-50,-36,-23,-12,0,10,20,31,43,56,71,92,127}`，即 round(NF4×127)；`79–85` 的 unpack 保持 high nibble first。

激活量化在该文件 `23–70`：每 64 BF16 求 absmax=a，存 **xs=a/127**，RTE 编码到 ±127；raw BF16 bits 先过滤 NaN/Inf，避免 fast-math 抹掉 finite 检查；tiny/zero 块走正确除法，校正 reciprocal 处理 ties。生产已有 `quantize_moe_input_q8_64_rows`（`MoeExpertOps.cpp:3726–3730`）可复用接口，生产 quantizer 在 `452–475` 还处理极大 BF16 防溢出。于是实际重建系数是 **`weight_absmax * activation_absmax / 127²`**，因为 xs 已除 127，projection 再为 integer NF4 LUT 除一次；禁止重复或遗漏因子。

`Qwen4ExpIntegerMoeProbe.hpp:185–190/226–248` 的完整链是：

```text
BF16 X → per64 Q8 → token gather/VNNI pack
NF4 gate/up → s8 stage + decoded scales → SG8 projection → BF16 SwiGLU H
H → token/rank gather + per64 Q8 → VNNI pack
NF4 down → s8 stage + decoded scales → SG8 down → FP32 rank buffer → ordered merge
```

普通链九次提交（包括 merge），gate/up 与 down 重用 staged weight/scale/VNNI 最大容量。dense 镜像和 oneDNN `[E,Kblock,N,M]` s32 partial 是匹配库比较的成本（DpasMoeProbe:129–175），**不应搬进生产 DPAS 主链**。原型依赖同一 in-order queue；生产跨 copy queue 必须补显式 ready dependency。

### 3.2 原型已有成绩不能省略的边界

`[已有记录]` [R2b DPAS 报告](qwen3.8-flash-next-r2b-dpas-results-2026-10-01.md)，驱动 32.0.101.9033、oneAPI 2026.1、两个 resident experts、合成激活：普通 NF4/DPAS wall μs 分别 M8=287.95/484.38，M32=526.85/529.78，M64=827.83/584.38。匹配 graph 的 M32/48/64 有收益，但 M8/M24 落后，M16 余量很小；**M≥8 是硬件形状最低要求，不是收益 dispatch 阈值**。不得从 {32,48,64} 推出所有 M≥32 已验证。

同报告 M32 device 分阶段：两次量化16.25 μs、gather/pack11.98 μs、两次权重转码81.41 μs、gate/up193.07 μs、down32.40 μs、merge19.38 μs。只是这组历史 fixture 的读数，不能直接移作当前生产常数。普通 SYCL gate/up 有480 B spill、down 0；ESIMD 是否消除需探针证据。

原始 `tmp/r2b-dpas-boundary-final-20261001.log:7–21` 同时显示 integer_errors=0 与 NF4 pointwise 超限。真实质量记录见第4节。没有当前驱动/生产整机 ESIMD 性能结论。

### 3.3 放在哪个生命周期

`[方案/待验证]` 首期采用 **GPU 解包，有界临时 staging**：

1. **X 量化每 layer/window 一次**，在 MoE 输入 `buf_.x` 已就绪后提交，覆盖 C 行；所有 resident/cold experts 共用。切忌每 unique expert 或 microbatch 重量化整个 X。
2. **权重 stage 每 unique expert、每 layer/window 一次/每 projection 一次**，位于 resident ready 或 cold H2D ready 之后、projection 之前；同专家的全部 M8 tiles 复用。cold 不增加 H2D 格式，仍只上传 NF4；不每 tile 反复解包。
3. **VNNI pack 每 expert bucket/microbatch 的有效行一次**，输入 XQ8 按 token map gather；可按固定8/32行 tile 构造带零 padding 的 VNNI，并记录 live_rows。只需 VNNI 与 gathered scales，删除 dense 镜像。
4. 双投影阶段再对 H 的 **候选 expert activation** 量化与 pack，必须从 `[token,rank,I]` gather；它依赖 gate/up BF16 完成。不能像 X 一样在 layer 前量化，更不能用 tokens alone 索引。
5. resident hits 目前可能一次全部提交，数量可能远大于 cold bank16；新 scratch 不能默认足以容纳全部 hits。保留原不可变 map slices，将候选 resident 也按预分配最大专家容量分组，或逐专家 stage；在同一专家所有 tiles 消费完成后才覆盖 scratch。

`buf_.moe_x_q8/moe_x_q8_scale/moe_h_q8/moe_h_q8_scale` 在 `.hpp:143–146` 和 `.cpp:1025–1031` **只预分配 `[H]`、`[H/64]`、`[R,I]`、`[R,I/64]`**，且受 decode Q8 开关/几何支持控制。不能当 C 行 prefill arena；建议新 `prefill_moe_xq/xs` 以 cap 预分配，避免扩大这些长期保留的 decode buffer。可复用量化函数和数学规则。

可直接复用：`buf_.x`、`buf_.moe_h`、`buf_.moe_down_acc`、existing token/rank maps、tiles 元数据思想、batch_weights、NF4 adaptive resident/cold source pointers、layer nested metadata、ordered merge、ready/consumer fence 链。新增 INT8 stage 描述符宜单独定义，不往 `MoeExpertSlotDev` 塞长期可变缓存状态。

adaptive 现有 pinned/device bank 仅容纳 NF4：geometry 在 `AdaptiveCache.cpp:104–119`，生产 slot bytes=`3HI*33/64`（backend:723）。StageColdExpertsAsync:603 只等待 pinned source 的前次 H2D 完成；device 写入在633依赖前次 consumer。**可复用 bank 的运输、地址和 fence，不能原地把 payload 覆盖成 s8**：原路径/后续 down/owner exchange 仍需要 NF4，而且容量不足。

若以后尝试 CPU unpack→pinned s8→H2D，需要额外 pinned 输出 bank 或改变所有 source/geometry/swap 假设，会增加 host 带宽与 H2D，第一阶段不采用。若做 resident INT8 常驻缓存，必须按 `(layer,expert,slot_generation,quantizer_version)` 标记有效性、swap/重载时失效，并额外计显存；不能按裸 device 指针缓存，也不能承诺“每 layer 只解包一次永久复用”。

### 3.4 显存及带宽账

设 B=同时 stage 的专家数、A=候选有效 activation 数、P=这些专家 padding 后的 activation 总数（P≥A），Ccap=预留窗口容量，H=2560/I=640。下面是新增量，现有 NF4/h/down arena 另计：

| 新缓冲/流量 | 容量公式（bytes） | 生产形状示例 |
|---|---|---|
| gate/up s8 + FP32 scales | `B*2IH*(1+4/64)` | 每专家3.3203125 MiB；B16单 bank53.125 MiB |
| down s8 + scales | `B*HI*(1+4/64)` | 每专家1.66015625 MiB |
| 两投影顺序复用 weight scratch | `B*max(2IH,HI)*(1+4/64)` | 与 gate/up 相同；若双 bank，再×2 |
| 两投影都常驻 expanded scratch | `B*3HI*(1+4/64)` | 每专家4.98046875 MiB；不推荐首期 |
| XQ8 + XS | `Ccap*H*(1+4/64)` | Ccap2048 为5.3125 MiB |
| gate VNNI + gathered XS | `P*H + A*H/64*4` | padding scales若被读则必须也补足P；无dense镜像 |
| H Q8 + HS（双投影） | `A*I*(1+4/64)` | 或分 tile/microbatch容量；C2048/R10全窗为13.28125 MiB |
| down VNNI | `P*I` | 可顺序复用更大的 gate VNNI bank，前次消费者需完成 |
| descriptors/live rows/events/graphs | O(B+tile_count) 元数据、driver graph资源 | 要分别报 device/pinned/native/driver，不遗漏 |

NF4 原始每专家 payload=`3HI*(1/2+1/64)`=2,534,400 B（约2.417 MiB），不含独立 layer metadata。以上 B16 是 scratch 设计例，不是要求建立16专家并发计算；scratch 容量/调度需测吞吐。B1更省显存但增加提交，B16利于合并但减少 resident cache budget。

每次 stage N×K 至少：读 NK/2 NF4 bytes、NK/64 scale codes 与 nested元数据，写 NK s8 + NK/64*4 scale；projection 随后读 expanded s8/scales。因此展开后的权重主读宽度是 NF4 的2倍，虽然换来更高 tile reuse/更少指令。转码读写、缓存未命中和额外launch不能遗漏。

X量化至少读2CH BF16、写CH Q8+4CH/64 scales；pack再读AH Q8，写PH VNNI及gathered scales；down同理并且多读BF16 H。CPU展开会把主权重H2D由0.5 byte/weight提升到1 byte/weight（另有scale表示变化），不能默认 DMA免费。

`backend:723–729` 当前 cache fit 以 NF4 slot bytes计算且扣固定32槽；新增 scratch要在**cache admission前**扣预算，同时审计实际 `2*staging_slots_per_buffer` 与默认32是否一致。只在 Tensor allocation时发现OOM不足以构成可接受接入；比较因减少resident数量而增加cold H2D的端到端代价。

## 4. 数值契约和现有测试的真实覆盖

### 4.1 三层契约

| 层级 | 必须验证 | 当前证据/限制 |
|---|---|---|
| 原 wgshare | 与plain 4token的 h/BF16和down/FP32位精确，share2/4/8、nested和tails | `.hpp:485–487`、KernelTests:1556–1613/1833–1988；保持原测试不放宽 |
| integer实现 | LUT/量化字节/RTE位精确；每K64 signed dot整数精确；缩放/SiLU/scatter与独立参考在声明容差内 | Dpas smoke；IntegerMoeProbe:282–317（FP64 abs1e-6+rel1e-4）；原型不能证明BF16/NF4位精确 |
| NF4模型质量 | 原BF16/NF4为基线，比较h、down、hidden/logits、生成、状态与HTTP输出 | W4A8 decoder是先例，非prefill批准；R2b仅单输入canary、逐点仍失败 |

`tests/ops/Qwen4ExpW4A8Tests.hpp:9–11/316–326` 明确近似；实际断言 gate/up 与 down cosine≥0.995、down NRMSE≤0.05。不要把历史最好NRMSE当已写入断言。其量化/CPU参考与zero-hot-allocation/teardown可复用；新prefill宜使用同样或更严门，并另报每层/专家/离群样本最大误差。

`[已有记录]` [W4A8 prefill收益上界](qwen38-w4a8-prefill-bound-2026-10-12.md) 第4节记录decode opt-in：内核NRMSE≤1.44%、全模型cos0.998/0.9995、输出逐位一致；这只证明相关固定样本上的输出一致，不能推导所有prompt或DPAS一致。该报告的“只减激活带宽≤0.7%端到端”也不是DPAS指令削减的上界。

真实R2b六组隐藏态在 [质量报告](qwen3.8-flash-next-r2b-real-quality-results-2026-10-01.md) 与 `tmp/r2b-real-quality-final-20261001.log:5/12/19/26/33/40`：整数实现检查通过，但相对NF4 NRMSE约1.63%–2.05%，abs2e-4+rel0.02逐点超限15,485–38,425/81,920，最高abs0.099711。同报告单1833-token输入、48步teacher-forcing：平均KL0.00701158、NLL差−0.00114979、最大logits NRMSE0.0791376；通过其固定canary门 `KL≤0.01/NLL差≤0.01/max logits NRMSE≤0.15`，top1只有42/48一致。**此门不替代逐点门与自由生成、代码/中文/多输入质量。** 性能包含NF4重复计算与diagnostic同步，标记performance_comparable=0。

### 4.2 `AilaQwen4ExpToleranceTests --model` 怎么用、不能证明什么

未来独占GPU时可运行（本审计未执行）：

```powershell
build-refactor/AilaQwen4ExpToleranceTests.exe --model C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2
build-refactor/AilaQwen4ExpToleranceTests.exe --dpas-smoke
build-refactor/AilaQwen4ExpToleranceTests.exe --dpas-model C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2
build-refactor/AilaQwen4ExpToleranceTests.exe --integer-quality C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2
```

入口 `.cpp:673–689/723–737`。`--model` 先跑mock tolerance，再跑真模型；mock断言NRMSE≤0.015、cos≥0.999（`346–361`）。真模型初始化maxseq512，用M26/M30比较 **dense oneDNN vs dense XMX**（`400–407/449–505`）；门为cos≥0.995、top1一致、M30 top5 overlap≥0.8（`511–516`），**没有真模型NRMSE上限断言**。生成token overlap是打印，末尾只断言非空/长度/词汇（`561–586`），不是完整输出逐位一致门。

因此仅设置新env后跑一次 `--model`，可能基线与candidate两臂都启用同一个新MoE路径；通过不能证明“DPAS vs旧MoE”质量。未来要增加独立test-only setter/override，分别强制baseline/candidate、重置session，明确记录被替换activation数，禁止静态env在进程中临时切换冒充A/B。`--dpas-model`是原型算子fixture；`--integer-quality`是诊断hook，不是生产dispatch acceptance。

新增模型验收应覆盖实际候选数值策略的gate-only/both，离散层0/23/47真实expert corpus、自然文本/中文/代码/结构化输出、多prompt长度；先冻结上述算子和模型门。历史pointwise失败不能通过放宽阈值消失：默认推广前解决失败（例如独立研究更细activation scale），或明确另立经产品接受的质量契约；本方案不声称可自动豁免。固定greedy基准要比较token IDs和content/reasoning/toolcalls，不以“包含READY”等语义测试代替逐位一致门。

## 5. 接入形态与按文件/函数/行号的补丁计划

以下全为未来计划，本次未修改这些文件。

### 5.1 开关和 dispatch

建议只用两个核心read-once env，分离数值策略和执行方式：

| 拟新增env（目前不存在） | 值与目的 |
|---|---|
| `AILA_Q4E_PREFILL_MOE_W4A8` | 0默认off；1仅gate/up近似、down原NF4；2 gate/up+down。在声明的prefill窗口范围内所有专家同策略 |
| `AILA_Q4E_PREFILL_MOE_DPAS` | 0同算术vector；1 SG8；2 ESIMD。W4A8=0时不起作用；明确请求ESIMD不可用则用已验收SG8或同算术vector并记录原因 |

可在研究阶段加一个 `AILA_Q4E_PREFILL_MOE_DPAS_MIN_ROWS`，但 M≥8只是硬guard；阈值默认保守≥32只是**待测起点**，不是当前validated rule。最终应固化被测形状表/收益区间，避免配置组合长期膨胀。

第一阶段请求/几何条件：current adaptive pipeline、count>4、H=2560/I=640、1≤R≤16、weight blocksize=64、compact_nested=true且nested_blocksize=256、默认NF4 LUT（加载时验证其16项身份）、rank输出、预分配scratch容量/对齐满足。count≤4仍走原路径；这定义一个明确prefill数值边界，**不保证任意窗口切分或prefill/decode跨边界位精确**，需单独验证。以后泛化non-nested FP32 scale与K/N尾形状要新增门。

每专家满足M≥8、K%64=0、outputs%8=0，构造同专家M8 tile。完整M8之外的live尾行走同算术vector或零padding tile，不能load越界再mask。空expert不submit。small buckets应保留同算术integer vector，K64 s32和、FP32scale multiplication/add顺序、native exp及BF16边界与DPAS对齐；执行选择变化要求h/down/最终output位精确，否则会重新引入桶形状影响前缀的风险。对原NF4的近似误差仍用容差门。

SG8能力检查参考 `AttentionOps.cpp:310–330`，但 **不能直接复用M1 guard**：MoE要查询m=8或dynamic(msize=0)、n=8/k=32、sint8/sint8/sint32/sint32，加SG8支持并按thread/device缓存。ESIMD只能启用探针在目标设备/驱动/编译器通过的组合；N16不可因为头文件支持就上生产。原型将W作为A、activation作为packed B，ESIMD探针可能选择常规X×W方向；要写显式layout adapter、用非对称/带标签数据检查转置，不能复制SG8指针算法而假定ESIMD operand顺序相同。

whole-shape unsupported或禁用W4A8：完整原NF4 fallback；DPAS能力/行数不支持：同算术vector fallback。compact output首期整请求拒绝候选；custom LUT或nested异常在提交前处理。IGC/runtime device loss发生后进入runtime fault/drain，**不能捕获已部分写output的失败后继续NF4重跑，伪装正常fallback**。

### 5.2 具体补丁落点

| 文件/函数/当前行号 | 未来改动与可审查结果 |
|---|---|
| `src/ops/MoeExpertOps.hpp:16–54,158–190,488–523` | 保持现有slot及wgshare ABI；声明独立DPAS stage/pack/projection接口、显式scratch view、bucket/live_rows、ready event，注释每个scale是否已/127 |
| 拟新增 `src/ops/MoeExpertDpasOps.cpp` / `MoeExpertEsimdOps.cpp` | 分离SG8与ESIMD TU，默认off；共用定义好的量化/LUT/scale/epilogue契约，stage/pack支持ragged和显式dep；不包含tests原型头、不把vendor非inline dot_acc header复制到多TU |
| `MoeExpertOps.cpp:405–475,500–525,3726–3730` | 复用/抽取nested两步rounding、Q8 RTE/finite/extreme规则；现有read-once逻辑先初始化；新增同算术small-bucket vector，不放宽旧wgshare断言 |
| `Qwen4ExpBnb4Backend.hpp:143–146,420,489,549–571` | 新prefill scratch owner与测试override/dispatch counters；固定tile参数/bucket描述符；明确backend execution slot持有，不能是session共享的可变权重副本 |
| `Qwen4ExpBnb4Backend.cpp:load 362–385,cache fit 667–729` | 能力/模型LUT/geometry预判，env读取与startup诊断；计算scratch budget后再决定resident slots；unsupported提前选baseline |
| `ensure_runtime_buffers:885–906,995–1065; TrimDecodeRuntimeBuffers:1079–1100` | 在capacity边界预分配XQ8、VNNI、stage、descriptor；growth/trim等consumer完成；steady layer/microbatch/tile绝无USM alloc/oneDNN rebinding |
| `RunAdaptivePrefill:1369–1418` | X按layer/window量化一次；resident和cold共用逻辑，候选resident也有界分组；保留cold只传NF4及原ordered merge；返回覆盖所有候选消费者的last event |
| `RunGroupedExperts:1421–1525` | 在unique expert count/边界已知处构造候选bucket；保留非重叠map slice、rank、XMX/ALU剩余段；W4A8策略固定，DPAS只决定执行；ready→stage/pack→projection→down→hook链纳入last |
| `ForwardImpl:2249–2257` | 明确count≤4与prefill候选范围；阶段统计涵盖额外转换；避免dense XMX/grouped BF16开关悄悄覆盖候选策略 |
| `Qwen4ExpPrefill.cpp:578–600` | 首期不改；compact output拒绝候选。若后续支持，单列scatter/merge/index offsets验收，先解决wgshare wrapper的Compact=false风险 |
| `Qwen4ExpAdaptiveCache.cpp:122–173,260–294,310–339,552–639; .hpp:208–210` | 首期保持NF4geometry；候选stage作为reader，最终event登记；若后续persistent INT8 cache，增generation/invalidation/dual-bank ownership、move/Release事件，独立变更 |
| `tests/ops/Qwen4ExpKernelTests.cpp:1833–1988,3819–4140,4468–4474` | 新kernel-only入口与完整成本bench，保留旧bit-exact；现有bench不会自动走backend新dispatch，必须直接调用新ops加conversion |
| `tests/models/Qwen4ExpModuleTests.cpp:5166–5219,5238–5705` | 在既有monolithic/microbatch、4/2/1场景上新增candidate同算术执行等价、ragged/tails、resident/cold两类fixture；必须调用实际adaptive dispatch，旧legacy cache fixture不充分 |
| `tests/models/Qwen4ExpToleranceTests.cpp:400–586,673–689,723–737` | 加baseline/candidate override与生产kernel命中计数、预冻结的MoE tolerance/输出比较；复用real corpus与quality oracle，移除诊断重复NF4才可计性能 |
| `CMakeLists.txt:111–123,155–177,787–793` | 注册新TU和gpu ops/module tests；保持per-kernel split/现有flags，先探针compile/JIT再进runtime；不全局改SG默认16去“修”INT8 |

graph不是第一阶段必要条件，也不能复用固定E/M地址原型后立即宣称动态生产支持。若普通提交无墙钟收益，再在同一执行槽中预建固定地址/固定容量graph，device参数描述live_rows/weight_ptr/map offsets；参数表复用前等待，graph销毁先drain。cache更新、输入变化、pointer rebinding、模型reload、session切换和取消分别验收；graph finalize/first replay/driver memory单列并计入首次请求成本。

## 6. 测试矩阵、性能口径与采纳门

### 6.1 必须逐层过门的矩阵

| 门 | 用例/比较 | 通过标准 |
|---|---|---|
| P0 探针 | SG8与ESIMD同形状M8/N8/K32；正负极值、零、非对称布局、非恒定scales；N16独立测 | s32点积精确；layout/canary零错误；ESIMD相对同形状SG8原始吞吐≥20%，否则不扩ESIMD分叉 |
| K1 stage/quant/pack | 全16 nibble、high/low顺序；nested scale跨255/256、offset非零；custom/非nested拒绝；Q8 ties/zero/subnormal/NaN/Inf/极大BF16；P≠A | 字节与独立oracle一致；两步FP32absmax正确；padding/所有被读取scale初始化；内存边界零错误 |
| K2 projection | gate-only和both；M={1,4,7,8,9,15,16,24,31,32,33,48,63,64,65}，N8，多个K64 scales；两专家相邻、置换token与不同rank；支持/不支持shape | M<8/尾走声明的同算术fallback；CPU整数dot精确，FP64 abs1e-6+rel1e-4；所有h/down/canary全量检查，不只首尾两行 |
| K3 NF4误差 | 合成+层0/23/47真实corpus、outliers；baseline原NF4、同算术vector、SG8、ESIMD分别比较 | cosine≥0.995、down NRMSE≤0.05作为现有W4A8最低门；逐点abs2e-4+rel0.02失败仍记录并阻止宣称该门通过；不得以aggregate掩盖离群值 |
| M1 分桶和merge | zero-count/mixed {9,8,7,8}/all odd/no full8；resident all-hit/all-miss/mixed，staging bank边界16/17/32/33；完整桶vsmicrobatch、R=1/10/16 | 每activation恰写一次；router权重/rank ordered merge和shared结果正确；同数值策略改变执行实现/冷热/切分，h/down/output位精确 |
| M2 因果/状态 | 相同prefix不同future suffix；threshold前后M变化；257-token前缀129；checkpoint33+32/truncate33/replay；窗口增长/trim/append/session交替 | 相同前缀hidden/KV/GDN/PLE逐位一致，snapshot/replay无污染；跨声明数值策略边界另按契约测，不假称任意切分位精确 |
| L1 生命周期 | ready之前不得读；同bank连续reuse；swap evict/reload；取消stage/pack/projection期间；partial startup failure、async fault | 不释放/覆盖in-flight USM；正常取消后新请求可用；fault保留第一cause；warm hot allocation counts不增；最后device/pinned计数零 |
| Q1 真模型 | `--model`旧回归+新增强制A/B；多语种/代码/结构化/自然文本、多个prompt长度/seed、固定greedy输出 | 旧cos/top1/top5门不退化；R2b固定KL/NLL/maxlogits门逐输入通过；报告所有token parity；输出逐位一致需求不允许用canary豁免 |
| B1 块级 | 同routes/nested/每专家M/完整gate-only或both链，stage+quant+pack+scatter+merge+submit/wait | 暖pipeline耗时下降≥1%，≥20批样本、≥3独立稳定运行，报median+p10/p90；额外预处理完整计入 |
| B2 生产HTTP A/B | 相同build/profile/fixture、32K context、chunk2048、14Kprompt、MTP0、同KV格式/reserve；off/on交错；短prompt/reuse/append/cancel恢复也测 | 主场景墙钟下降≥2%，无可重复其他场景回退；usage/content/token正确、稳定，无OOM/明显memory膨胀；比较首次与reuse请求分别，不混startup |

生产推广全部质量/内存/状态门为AND。表中点误差门目前历史候选未通过，不因新ISA探针通过而自动消除。SG8与ESIMD对同量化算术若存在FP32差异，要定位/对齐后才满足前缀状态位精确门；这是实现工作，不是将所有状态检查改成容差。

### 6.2 块级：`--prefill-bench-only`

未来独占GPU且候选已构建后，先跑当前基线入口：

```powershell
build-refactor/AilaQwen4ExpKernelTests.exe --prefill-bench-only
$env:AILA_Q4E_BENCH_XLOCALITY='1'
build-refactor/AilaQwen4ExpKernelTests.exe --prefill-bench-only
```

它不是“只打印速度”：入口4468–4474先跑MoE correctness与wgshare mapping edges，再bench，最后检查USM teardown。现有wgshare bench `4082–4110` 直接调用ops wrappers、3次warmup、20次提交后一次wait，打印**整批平均wall time/iteration，不是20次独立样本median**；前置报告的20次中位数记录不能反向改变源码含义。新bench需明确采样定义并打印median/分位数。

目前新backend env不会让此direct-ops benchmark自动测DPAS。需增加对应新ops直接调用与完整conversion chain；同时保留plain4token/share2/4/8为baseline，nested实际权重、真实ragged route distributions为主。token0/4096行X locality只解释瓶颈，不构成代表性生产性能或模型质量。

分别记录 device(event profiling)、host submit→wait、完整pipeline；禁止只拿projection device time抵baseline wall。先保持普通提交同队列；graph与非graph、first-JIT与warm、cache HIT与coldH2D分开。增加运算强度/指令/GRF/spill证据时使用独立diagnostic进程，不把profile计时混进墙钟。

### 6.3 生产口径：`run_chunk2048.py`

`tmp/q4e-int8kv-dpas-20261012/run_chunk2048.py:13–43` 清理AILA/SYCL/UR_L0环境后导入 `docs/plans/qwen38-server-mtp-2026-10-07.json` 的worker_environment；固定MTP0、VRAM_RESERVE_MB3584、chunk2048，context32768、prompt14000、reuse1、fixture `tmp/q4e-agent-corpus-v1`。非diagnostic生成32并传 `--production-timing`；diagnostic生成8并启用work_summary/worker log。arg中的KEY=VALUE覆盖profile，**运行者shell env不会自动穿过清理**。

未来拟用以下不同label（目前env尚不存在；本次未运行）：

```powershell
python tmp/q4e-int8kv-dpas-20261012/run_chunk2048.py audit-moe-off-01 AILA_Q4E_PREFILL_MOE_W4A8=0 AILA_Q4E_PREFILL_MOE_DPAS=0
python tmp/q4e-int8kv-dpas-20261012/run_chunk2048.py audit-moe-esimd-01 AILA_Q4E_PREFILL_MOE_W4A8=1 AILA_Q4E_PREFILL_MOE_DPAS=2
python tmp/q4e-int8kv-dpas-20261012/run_chunk2048.py audit-moe-sg8-01 AILA_Q4E_PREFILL_MOE_W4A8=1 AILA_Q4E_PREFILL_MOE_DPAS=1
```

这些未来命令输出在旧runner目录，不是本次准许写入范围；本次交付只有本文。正式实施时应复制等价runner到批准的新campaign目录，不能在本次审计执行命令。

三臂可区分“量化/DPAS净收益”与“ESIMD比SG8的增量收益”；另加W4A8=1/DPAS=0同算术vector控制。baseline retain现场有效wgshare值；不要只用share0弱基线。确认legacy profile里的设置实际在HEAD生效，staging默认16与旧demand32不是同一设置。固定KV相关值，`AILA_KV_QUANT_DPAS=0`避免decode attention负收益混入；需测INT8 KV时两臂同格式/同vector设置。记录实际运行的ExecutionConfig、driver/compiler与所有命中计数。

`tools/server/qwen38_context_probe.py:43–62` 的production-timing去诊断env并保存effective worker_environment；`75–86` 记录first_content_seconds与after-first tokens/s。墙钟主门用 **同label请求的first_content_seconds/TTFT**；after-first decode speed不是prefill收益。startup/loading、initial/reused/cached_prompt_tokens、整个SSE duration分别记录，不能把prefix cache差异当kernel收益。诊断work_summary阶段时间只是定位（backend:1939–1940中stages会sync），不作生产墙钟依据。

off/on/off/on/off/on独立进程交错，至少3稳定paired轮；1%/2%接近噪声时增加样本、报paired分布/置信范围，不选最快轮。输出hash或tokenIDs、usage/finish_reason一致后才比较速度。检查monitor系统available RAM、explicit USM峰值、DXGI实际显存和resident slots/cold bytes；显存ledger与WDDM占用分别报告。

`--agent`增加pi-context/cancel-recover；`--cancel-prefill`中断长prefill并验证恢复（context_probe:156–187）。取消/异常为独立验收进程，不混普通timing。runner的`--cancel-only`只删除prompt值14000，并非天然选择全部取消用例，需要显式配合cancel标志，不能误读为完整取消验收。

### 6.4 决策记录

以耗时下降定义收益：`1 - Tcandidate/Tbaseline`，避免吞吐增长百分比混用。ESIMD raw≥20%是P0存在价值门；块级≥1%/墙钟≥2%是后续采纳门，均不能跳过质量。若ESIMD不优于已过门SG8，保留SG8与vector，关闭ESIMD分叉；若raw优但full pipeline失败，归档负结果；若只块级过门，默认off；默认生产启用还需B2及所有状态门。

## 7. 风险与控制

| 风险 | 具体事实和控制 |
|---|---|
| USM ledger与预算 | `Context.hpp:62–67/89–110/238–255`持有device/host预算和allocation map；新增scratch必须经Context/Tensor，禁止生产裸malloc_device绕账。startup/capacity边界预分配并计cache fit，partial allocation失败清理已分配对象 |
| 取消/异常后in-flight对象 | compute与copy是不同队列。AdaptiveCache Release:260–270分别draincopy/readers/resident事件；backend destructor:272–281等待copy再释放cache/arena；Context synchronize:203–210始终尝试drain并保留async fault；Scheduler:148–150依赖Step已drain才释放ownership。新stage/graph读者必须进这些fences，不依赖returned event在host被忽略仍安全的侥幸 |
| scratch或host参数覆盖 | StageColdExpertsAsync:603/633区分host DMA consumer与device compute consumer；新增weight/VNNI banks及graph参数表要同样区分。map mirror下一层router wait才可重写；指针生命周期延伸到最终consumer，而非H2D ready |
| 热路径分配/描述符重绑 | `ensure_runtime_buffers:890–892`growth前drain；trim:1090–1095同样drain。所有max-cap scratch、bucket参数、graph/primitive事先reserve；不得每layer/expert allocation、unordered_map动态扩容或oneDNN descriptor rebinding。首次env/capability查询放startup，热循环allocation counters保持不增 |
| adaptive交换/缓存污染 | NF4是GPU/nativeRAM互补唯一owner，swap:571–585会保留victim sole copy并交换owner；不得将现有NF4slot原地换成s8。persistent扩展缓存必须generation失效；首期temporary staging降低维护面 |
| 前缀因果/执行差异 | 从M/count/future routing选择数值格式会改变旧prefix；固定W4A8策略，M只选择同算术执行。整数dot精确不等于浮点scale/SiLU累加位精确，SG8/vector/ESIMD结果须跨tile选择一致并做prefix/change-future测试 |
| IGC/JIT/codegen | SG16 INT8失败不能推断SG8无能力；只用s8/s8/s32 N8/K32已验证组合。ESIMD新TU仍用同IGC，register/spill、large GRF、模板组合可能编译/JIT崩溃。保留per-kernel split，compile→单进程watchdog smoke→stress→模型；不全局改SG或flags；runtime device fault不可用NF4掩盖 |
| ESIMD布局/API外推 | 显式simd、operand顺序、VNNI orientation需探针adapter；不从N16或通用ESIMD API存在推出A770支持。技能中其他driver/attention的TDR记录只作风险提示，不能直接转为MoE阈值；每个选定geometry独立验收 |
| graph状态与首请求 | 固定地址可以更新contents≠动态weights/session全自动安全；finalize与firstJIT/firstReplay单列；不能每microbatch build graph而套用暖replay收益；graph/nodecursor必须等所有scatter消费者结束再推进 |
| compact分叉 | 当前compact output关闭，generic wgshare down wrapper固定Compact=false；首期明确reject，后续单独修/验输出offset，不将compact nested与compact output混同 |
| 维护分叉 | NF4、integer vector、SG8、ESIMD四种实现容易重复量化/scale规则；共享内部contract/helpers/reference，测试独立oracle。只覆盖gate/up、必要时down两个热点；ESIMD缺净收益就不保留长期生产TU/复杂env组合 |

## 8. 关键已读位置清单

以下是直接读过或检索后按函数范围核对的关键位置，可作为后续补丁review索引；其中已存在日志数值是历史证据，本次没有重跑。

| 文件 | 行号/函数/用途 |
|---|---|
| `AGENTS.md` | 139–147：subgroup/env/零热USM分配与取消drain不变量 |
| `docs/plans/qwen38-esimd-path-2026-10-12.md` | 7–15工具链；17–28目标；30–44分阶段≥20%与≥1%/≥2%门 |
| `docs/plans/qwen38-int8-dpas-repair-2026-10-12.md` | 7–44根因/修复/guard；46–70正确性；74–93时间与84.6%回退 |
| `src/ops/MoeExpertOps.hpp` | 16–39slot/nested/tile；450–5234token/wgshare ABI；158–190Q8/grouped接口 |
| `src/ops/MoeExpertOps.cpp` | 405–425scale；428–525integer/quantizer；2410–2664gate/up wgshare；2844–3023down wgshare；3684–3730Q8 rows；4278–4416wrappers；4480–4791section dispatcher |
| `src/ops/AttentionOps.cpp` | 310–330INT8 capability cached M1/N8/K32，MoE不能照抄M1 |
| `src/models/Qwen4ExpBnb4Backend.hpp` | 143–146decodeQ8容量；420/489grouped/adaptive入口；549–571map/hook生命周期 |
| `src/models/Qwen4ExpBnb4Backend.cpp` | 259–281析构；362–385compact/device；650–729预算/staging；738–760NF4identity；885–906growth/phase；995–1065MoE/maps容量；1079–1115trim/nested；1369–1525真实生产expert chain；1693/1881/1905/1939–1940诊断；2249–2257选择；2395–2398trim边界 |
| `src/models/Qwen4ExpPrefill.cpp` | 1–47attention decode量化；557–575attention tile；578–600compact down helper |
| `src/models/Qwen4ExpAdaptiveCache.hpp` | 100–142生命周期/容量；208–210reader登记；327–331ping-pong状态 |
| `src/models/Qwen4ExpAdaptiveCache.cpp` | 104–119geometry；122–173Contextalloc；260–294Release；310–339nesteddescriptor；552–586swap；594–640coldstaging |
| `src/models/Qwen4ExpHostArena.cpp` | 129–143LoadLayerDirectnested64/256验证 |
| `src/core/Context.hpp` | 62–115ledger；203–210drain；231–255allocation owner/预算 |
| `src/inference/Scheduler.cpp` | 138–150terminalownership与Stepdrain约定 |
| `tests/models/Qwen4ExpDpasMoeProbe.hpp` | 1–101stage/pack/projection；103–123smoke；129–175匹配oneDNNpartial/reduction |
| `tests/models/Qwen4ExpIntegerMoeProbe.hpp` | 4–85LUT/quant/RTE/unpack；155–198realfixture与容量；226–270完整链；282–317oracle；319–350graph一致；353–364容量/计时 |
| `tests/models/Qwen4ExpIntegerQualityAcceptance.hpp` | 110–117projection链；138–153scatter/error；171–179CPU整数参考；353–354canary门与production_enabled=0 |
| `tests/ops/Qwen4ExpW4A8Tests.hpp` | 1–11近似声明；316–331实际cos/NRMSE门与activation成本 |
| `tests/ops/Qwen4ExpKernelTests.cpp` | 1556–1613wgshare精确比较；1819/1833–1988edges；3819bench入口；4060–4123map/直接调用/采样；4468–4474prefill-bench-only与teardown |
| `tests/models/Qwen4ExpModuleTests.cpp` | 5166–5219monolithic/microbatch；5238/5313/5363/5420/5494/5559reuse路径；5654–5705routing场景 |
| `tests/models/Qwen4ExpToleranceTests.cpp` | 346–361mock门；400–430fixture；477–516real门；561–586generation不强制全token一致；673–689prototype入口；723–741--model解析 |
| `CMakeLists.txt` | 111–123fastmath/per-kernel split；155–177source list；787–793tolerance/DPASCTest |
| `tmp/q4e-int8kv-dpas-20261012/run_chunk2048.py` | 13–43profile/覆盖/production-timing；44–66watchdog/monitor，仅读取 |
| `tools/server/qwen38_context_probe.py` | 22–33选项；43–62清诊断/effectiveenv；75–86TTFT；156–187取消prefill与恢复 |
| `docs/plans/qwen38-server-mtp-2026-10-07.json` | 5worker_environment；20groupedXMX0；35legacydemand32；其他env只作固定profile参考 |
| `docs/plans/qwen38-w4a8-prefill-bound-2026-10-12.md` | 6–27激活带宽上界；29–42benchmark入口与decode质量先例 |
| `docs/plans/qwen3.8-flash-next-r2b-dpas-results-2026-10-01.md` | 实现/九submit/容量、普通/graphwall、device分阶段、graph边界、ISA/spill/质量限制 |
| `docs/plans/qwen3.8-flash-next-r2b-real-quality-results-2026-10-01.md` | 13–21统一数值策略与cursor；29–44realcorpus误差；54–69teacherforcing与42/48；71–85状态；最终production未接入 |
| `tmp/r2b-dpas-boundary-final-20261001.log` | 7–21integer正确/NF4超限/stage容量；52–66M16边界 |
| `tmp/r2b-real-quality-final-20261001.log` | 5/12/19/26/33/40六组真实隐藏态；150canarygate_pass1但production_enabled0 |
| `tmp/q4e-int8kv-dpas-20261012/tolerance-int8-vec.log` | 237–250cos/NRMSE历史读数；276checkpointpass，不是新MoE证据 |

交付边界：这是可执行的接入/验收补丁计划，不是生产启用批准。下一步等待探针代理给出选定geometry、operand/VNNI定义、正确性/编译/JIT/ISA与同形状吞吐记录，再按上述门逐层接入。

## 9. 复核记录（2026-10-12，主会话抽查）

本节由主会话在交付后抽查写入；未改动正文结论。抽查方式：对承重引用逐一在当前工作树用
函数名/环境名/门槛值检索比对（HEAD=`6b7286a`，工作树含未提交改动）：

- `MoeExpertOps.hpp:16/24`（两 struct）；`.cpp:406` `load_expert_absmax`；`.cpp:2417/2846`
  gate-up/down wgshare 模板；`.hpp:488/506` wrappers；`.cpp:4484` `AILA_Q4E_4TOKEN_WGSHARE`
  read-once；`.hpp:178`/`.cpp:3726` `quantize_moe_input_q8_64_rows`；
- `Qwen4ExpBnb4Backend.cpp:366` `moe_compact_tile_=0`；`:378` `grouped_xmx_supported_`；`:721`
  staging slots；`:723` `slot_bytes=3HI*33/64`；`:891` growth 前 `ctx.synchronize()`；`:1079`
  `TrimDecodeRuntimeBuffers`；`:1369/:1421` `RunAdaptivePrefill`/`RunGroupedExperts`；
  `:1425–1427` 三个 reuse env；`:2249` 附近 `ForwardImpl` 分支；
- `Qwen4ExpAdaptiveCache.cpp:310/332/336` `make_slot`/`GetResidentSlotDev`/`GetStagingSlotDev`；
  `Qwen4ExpHostArena.cpp:137–139` blocksize/quant_state 验证；
- `Qwen4ExpW4A8Tests.hpp:323–326` 实际门槛（gate/up+down `cos≥0.995`、down `NRMSE≤0.05`）；
  `Qwen4ExpDpasMoeProbe.hpp:8/30/46/103` 原型函数；
- `AttentionOps.cpp:310–330` INT8 capability 查询（m=1/N8/K32，MoE 不能照抄 M1）。

结论：抽查的 20+ 项行号/环境名/门槛引用全部命中，未发现需要更正的事实错误；正文对
“源码事实 / 已有记录 / 方案待验证”的标注与当前实现一致。

探针侧补充（主会话核对本机头文件，供正文 §3/§5.1 参考）：

- 上游 e2e（intel/llvm sycl 分支 `sycl/test-e2e/ESIMD/dpas/dpas_common.hpp`）的 VNNI
  （vertically packed）公式：`idx(k,n)=(k/4)*N*4+n*4+(k%4)`；参考副本已存
  `tmp/q4e-esimd-dpas-20261012/upstream-dpas_common.hpp`。
- `<sycl/ext/intel/esimd.hpp>` 内部包含 `esimd/xmx/dpas.hpp`；`sycl/ext/intel/info/device.hpp`
  在本机 oneAPI 2026.1 安装中**不存在**，不能用它取 `gpu_eu_simd_width` 探测执行尺寸。
- 探针草稿要点与两个已知坑：`tmp/q4e-esimd-dpas-20261012/handoff-draft-notes.md`。
- 过门后执行清单（本节的配套操作版）：`docs/plans/qwen38-esimd-post-gate-checklist-2026-10-12.md`。

## 10. Phase 1 接入清单（probe-verified 版）

本节依据探针实测（Hooke：`tmp/q4e-esimd-dpas-20261012/performance.md`，日志 `probe-run-1/2.log`；
用户独立复跑核验）与既有审计结论合并。**P0 判定：GO**；本节把先前的“方案/待验证”条目
升级为可执行口径，并标注少数需要修改的假设。写本节时未改任何源码；Phase 1 动作待确认后执行。

### 10.1 冻结的实测输入

| 项 | 实测值 | 来源 |
|---|---|---|
| ESIMD rc=8/ilp=4 reg-resident 吞吐 | 93,406 GMAC/s（run-1 94,840） | `performance.md` §3 |
| SG8 joint_matrix 8×8×32 ilp=4 同形状 | 69,113 GMAC/s（run-1 68,806） | 同上 |
| 比值（P0 门 ≥1.20×） | **1.351×**（run-1 1.378×；用户复跑 1.355×） | 同上 + 独立复跑 |
| ESIMD rc=8 权重逐轮 global 重载 | 85,056 GMAC/s（仍 **1.23×** SG8） | 同上 |
| ESIMD rc=4 / rc=1 | 94,254 / 58,806 GMAC/s（rc=1≈63%） | 同上 |
| 标量 int8 点积（wgshare 类数学） | 31,078 GMAC/s（ESIMD 为其 **3.0×**） | 同上 |
| 正确性 | m1/2/4/8×k32、k64/128/256 链、u8×s8 混合、SLM 路径全部 int32 逐位精确 | `performance.md` §1 |
| MoE tile（m8×k256×n8，per-64 scale epilogue） | max_abs=1.86e-09，max_rel=5.53e-07 | 同上 |
| N16 | **静默错误**（got=96117 vs ref=29609，无异常、无 TDR） | 同上 |
| NF4→s8 解包 | 10,485,760 weights / 0.2277 ms = 69.1 GB/s（r+w，50% occupancy） | `performance.md` §2 |
| 激活 VNNI pack | 512×2560 / 0.0894 ms = 29.3 GB/s（r+w） | 同上 |
| 布局契约 | A 行主序 `a[m*32+k]`；B VNNI `v(k,n)=(k/4)*N*4+n*4+(k%4)`；C 行主序 int32；K=32/次；**DG2 N=8** | `performance.md` §0 |

### 10.2 对审计假设的逐条复核（成立 / 需改）

| 审计中的假设 | 判定 | 新口径 |
|---|---|---|
| ESIMD 同形状 raw ≥1.20× SG8 才有存在价值（§6.1 P0） | **成立，已过门** | 1.35–1.38×；这是“存在价值门”，**不是** dispatch 阈值（§6.4 维持） |
| 整数部分必须逐位精确（K1/K2） | **成立** | 含 SLM staging 路径；K 链 8/4/2 块全精确 |
| MoE tile 容差设计门 max_rel ≤1e-4（探针设计） | **成立，裕度巨大** | 实测 5.53e-07；epilogue 可按“int 精确 + 逐块 fp32 乘加”实现 |
| N16“不可因为头文件支持就上生产”（§5.1） | **需改：升级为硬门** | DG2 N16 是**静默错误**（无异常）；编译期不得实例化 N16 形状，运行时 dispatch 只允许 B tile N=8；探针 N16 用例保留为负证据并断言继续失败 |
| “转码读写/额外 launch 不能遗漏”（§3.4，定性） | **成立，升级为定量** | 解包 69.1 GB/s → 每专家外推：gate/up（2IH=3.2768M weights）≈0.0711 ms、down（HI=1.6384M）≈0.0356 ms；VNNI pack ≈0.0894 ms/512 行（≈0.175 µs/行），且 pack 实测仅 29.3 GB/s，**是主链薄弱项**，需真实形状复测或与 projection 重叠 |
| scratch 尺寸公式 `B*2IH*(1+4/64)` 等（§3.4） | **成立，结构不变** | 仍按 P≥A padding 与 gathered scales 记账；实测只改变时间预算，不改变容量公式 |
| “raw 过门后仍需真实形状对比”（§6.4/§8 结尾） | **成立，且 Hooke 已列为前置** | 进入 B1 前先补：K=2560/640（80/20 块链）grid 上 ESIMD vs SG8 同形 + epilogue 计入的同形对比（口径沿用 3 预热+20 采样 median） |
| rc=1 可用（decode 形状） | **需改：限制形状** | rc=1 仅 63% 吞吐 → 生产 tile 用 rc≥4；rc=4 与 rc=8 同速（±1%），尾桶 4–7 行用 rc=4 tile（编译期形状族 {8,4}），<4 行走同算术 vector |
| 优势依赖权重寄存器常驻 | **需改：不成立** | A 逐轮 global 重载仍有 1.23× → 优势来自指令/流水；staging 常驻可放宽，但 staging 时间仍必须计入块级预算（见上一行） |

### 10.3 可执行补丁清单

#### A. env（全部 read-once、startup 解析、默认 off）

| env | 默认 | 值域/语义 |
|---|---|---|
| `AILA_Q4E_PREFILL_MOE_W4A8` | 0 | 0=off；1=仅 gate/up 量化；2=gate/up+down。声明窗口内全专家同策略 |
| `AILA_Q4E_PREFILL_MOE_DPAS` | 0 | 0=同算术 vector；1=SG8；2=ESIMD。W4A8=0 时无效；显式请求 2 而 ESIMD 不可用 → 降到已验收的 1/0 并记录原因 |
| `AILA_Q4E_PREFILL_MOE_DPAS_MIN_ROWS` | 0 | 研究用；0=不额外限制。**M≥8 是硬 guard 而非此开关**；阈值最终以实测形状表固化 |

#### B. dispatch 条件（全部 AND；任一条不满足走 fallback）

- 路径：adaptive prefill（`ForwardImpl:2249–2257`，count>4）；H=2560、I=640；weight blocksize=64、
  nested_blocksize=256、默认 16 项 NF4 LUT 身份校验通过；
- 几何：**每专家 live rows ≥8**（M=RC）；K%64==0（K=2560/640 ⇒ 80/20 条 K32 链）；
  outputs%8==0；**B tile N=8 only（硬门）**；
- 形状族：M8 tile（rc=8）+ 尾桶 4–7 行 rc=4 tile；<4 行、空专家（不提交）→ 同算术 vector；
  任何 padding 行必须以 0 码/0 scale 补齐并登记 `live_rows`；
- 输出：compact output 首期**整请求拒绝候选**（任一等价检查失败即拒绝，不静默降级）；
- 能力：startup 一次性查 s8×s8→s32、N8/K32 组合 + 一次 watchog smoke JIT；按 thread/device 缓存；
- fallback 矩阵：几何/行数不满足 → 同算术 vector；capability 失败或 env=0 → 原 NF4 全路径；
  IGC/JIT/device fault → runtime fault/drain（**不允许**捕获部分写输出后重跑 NF4 伪装成功）。

#### C. scratch 所有权（USM ledger、零热路径分配）

- 全部经 `Context`/`Tensor` 分配，记录在 ledger（禁止裸 `sycl::malloc_device`）；owner =
  backend execution slot（非 session 共享的可变权重副本）；
- 预分配点：`ensure_runtime_buffers` 容量边界（`885–906/995–1065`），**先进 cache fit**
  （`723` 同口径：`slot_bytes` 折算后 ff 定 resident 数），`growth/trim` 沿既有 drain
  （`891/1090`）；热路径（每 layer/microbatch/tile）alloc 计数必须不增、oneDNN descriptor 不重绑；
- 建议缓冲（沿用 §3.4 公式）：`prefill_moe_xq/xs`（[Ccap,H]、[Ccap,H/64]）、
  `prefill_moe_stage_w8`（[B,max(2IH,HI)] s8 + scales，双 bank）、`prefill_moe_vnni`（[P,H]、
  down 复用 [P,I]）、`prefill_moe_hq/hs`（[A,I]、[A,I/64]）、元数据（tiles/live_rows/events）；
- 复用 adaptive 运输/地址/fence 结构（`StageColdExpertsAsync:603/633` 的 host-DMA vs device-consumer
  区分），**不得原地把 NF4 payload 覆盖成 s8**；冻结的 NF4 bank 与新增 INT8 scratch 分离。

#### D. 数值契约（K64 scale 与前缀因果）

- 每 K64 = 2 次 K32 dpas 的 s32 部分和，整数精确；per-64 nested scale 解码保持两步 FP32
  rounding（`load_expert_absmax:405–425`：`fma(map,absmax,0)`→`fma(product,1,offset)`）；
- epilogue 固定顺序：`partial * (w_absmax/127) * (x_absmax/127)`（xs 已除 127，生产禁止重复
  或遗漏因子）；pre-probe 用预生成 combined scale 向量的做法只作为探针实现，生产要与
  vector/SG8 实现**同次序**，否则同策略跨执行实现不位精确；
- SiLU（native exp）×up、BF16 h 边界、down FP32、route 权重与 ordered merge 全部不动；
- 策略边界由“请求 + 声明窗口”决定，**不由桶形状/冷热/切分选择**：同策略内 vector/SG8/ESIMD
  的 h/down/final output 必须位精确（前缀因果）；近似误差只走 K3/Q1 容差门，历史逐点超限
  不解禁、不放宽。

#### E. 目标文件/函数 hook（行号按 HEAD=6b7286a 工作树）

| 文件/函数 | 改动 |
|---|---|
| `src/ops/MoeExpertOps.hpp:16–54,158–190,488–523` | 保持现有 slot/wgshare ABI；新增 ESIMD ops 声明（stage/pack/projection、scratch view、live_rows、dep event、每个 scale 是否已/127 的注释） |
| 新 `src/ops/MoeExpertEsimdOps.cpp`（+ 共享契约头） | ESIMD TU（默认 off）；复用 `quantize_moe_input_q8_64_rows`（`.hpp:178`/`.cpp:3726–3730`）与 `load_expert_absmax`；不 include tests 原型头 |
| `src/ops/MoeExpertOps.cpp:405–475,500–525` | 抽取/复用二步 rounding 与 Q8 RTE/finite/extreme 规则；small-bucket 同算术 vector（不改旧 wgshare 断言） |
| `src/models/Qwen4ExpBnb4Backend.hpp:143–146,420,489,549–571` | 新 scratch owner、env、capability gate、override/计数（测试用） |
| `src/models/Qwen4ExpBnb4Backend.cpp:362–385 / 650–729 / 885–906 / 995–1065 / 1079–1115` | startup 诊断、cache fit 预算、预分配/trim/growth drain |
| `RunAdaptivePrefill:1369–1418` / `RunGroupedExperts:1421–1525` | X 每 layer/window 量化一次；单专家桶边界构 tile；stage/pack/projection/down 全进 last-event 链；ordered merge 不动 |
| `ForwardImpl:2249–2257` | count≤4 与原路径保持；候选范围显式 |
| `CMakeLists.txt:155–177` | 注册新 TU；**不改** `111–123` 的全局 flags/SG 默认 |

### 10.4 测试矩阵（Phase 1）

**Kernel 级**
- 回归：现有 `--esimd-dpas-probe` 全绿（N8 族）；**N16 负用例保留且必须继续失败**；
- 新增：整数精确（rc∈{4,8}、K 链 80/20、u8 混合）；MoE tile 容差（设计门 max_abs ≤1e-4，
  实测 1.86e-09 为裕度证据）；stage/pack 字节精确（high-nibble 顺序、cross-255/256、
  offset≠0、P≥A padding、被读 scales 全初始化）；负向对照（错布局/漏乘 N 的参考 bug 类
  必须失败）；canary + 热路径 alloc 计数零增量 + teardown USM 归零。

**块级（`--prefill-bench-only`）**
- 现有入口先 correctness 后 bench；新增 direct-ops bench：3 预热 + ≥20 采样 median/p10/p90，
  **完整转换链（stage+quant+pack+scatter+merge+submit）计入**；与 wgshare baseline 同 process
  交替；保留 plain/share2/4/8 对照行；真实 ragged 分布为主，token0 类局部分析不作代表值。

**生产 A/B（`tmp/q4e-int8kv-dpas-20261012/run_chunk2048.py`）**
- 臂：off / `W4A8=1,DPAS=2` / `W4A8=1,DPAS=1`（SG8）/+ `W4A8=1,DPAS=0`（同算术 vector 控制）；
- 固定：MTP0、KV 格式一致、`AILA_KV_QUANT_DPAS=0`（避免 decode 负收益混入）、
  chunk2048/context 32768/prompt 14000/reuse1；off/on 交错 ≥3 稳定轮；
- 主门：同 label `first_content_seconds`/TTFT；同时记录 after-first tokens/s、USM device/pinned、
  DXGI、cold H2D 增量、stage 命中计数；输出 hash/tokenIDs/usage 一致后才比较速度。

**回归清单（每次采纳前）**
- CPU/ABI：`ctest --test-dir build-refactor -L 'cpu|abi' --output-on-failure`；
- GPU 现有：`AilaQwen4ExpKernelTests`（含 wgshare bit-exact、prefill bench）、
  `AilaQwen4ExpDpasSmoke`、`AilaQwen4ExpInt8DpasAttention`、`AilaQwen4ExpToleranceTests --model`、
  Module tests；USM teardown 计数零；
- 采纳改动照惯例单独提交 + 重建官方 worker/server pair 归档；
- 顺手项：探针 TU 的 `data_as<AT>()` 缺 `template` 告警（2 处）在提交前一并清理。

### 10.5 验收门、判定与回退

| 门 | 判据 | 未过处理 |
|---|---|---|
| P0（已过） | 同形 ≥1.20×（实测 1.35–1.38×） | —（存在价值门，非 dispatch 阈值） |
| 补测（B1 前置） | 真实形状 K=2560/640、epilogue 计入的同形 ESIMD ≥ SG8 | <1.0× 时只保留 SG8/vector；不影响块级门判定 |
| B1 块级 | 完整链 median 耗时下降 **≥1%**（≥20 批、≥3 轮、含全部转换/提交） | 未过 → 默认 off，opt-in 归档负结果 |
| B2 生产 | 同 label TTFT 下降 **≥2%**（交错 ≥3 轮稳定）且质量/状态/生命周期全绿 | 只过 B1 → 默认 off；质量门失败 → 停止推广 |

负结果归档：默认 off + opt-in env 保留 + `docs/plans` 记录 + 探针证据目录不删；回退 = env
全 0 即刻回原 NF4 路径，并在同 build 复跑既有 bit-exact 回归确认。决策记录以
`1 - Tcandidate/Tbaseline` 计时口径书写（§6.4），不混用吞吐百分比。

——本节为计划，不是启用批准；源码零改动，Phase 1 动作待确认。
