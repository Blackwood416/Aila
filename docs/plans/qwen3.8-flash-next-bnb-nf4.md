# Plan: 适配 Qwen3.8-Flash-Next-BNB-NF4-v2 (`qwen4_exp`) 到 Aila

## Context

`F:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2` 是 Qwen3.8-Flash-Next（`model_type=qwen4_exp`，125B 总参 / 6B 激活 / 51B n-gram 嵌入）的定制 BNB-NF4 checkpoint，131 个分片共 94.5 GiB。Aila 目前只认 `qwen3_5`/`qwen3`/`qwen3_asr`/`diffusion_gemma`，`LoadModelWeightsFromDir` 会把全部权重上传 GPU（A770 16 GB 装不下），且现有 Hybrid 层执行器（`HybridLayerExecutor`）的层结构（pre/post RMSNorm + dense FFN）与该模型不同。

参考实现已在本机：`C:\Users\Administrator\AppData\Local\Programs\Python\Python313\Lib\site-packages\transformers\models\qwen4_exp\modeling_qwen4_exp.py`（下文引用行号均指此文件）。

用户已确认：
- 主机 RAM 64–96 GB：专家（62 GB）+ PLE 表（32 GB）无法全部常驻，靠 mmap 页缓存按需读。
- 专家策略：**GPU 计算 + 主机 mmap 驻留 + GPU LRU 专家缓存**。
- 分期：M1 文本正确性（ctx ≤ 2048，QSA 等价稠密）→ M2 QSA 稀疏索引 → M3 vision → M4 共享会话/批处理/HTTP。**MTP 不做**。

## 模型结构（来自 config.json + 参考实现）

| 项 | 值 |
|---|---|
| hidden / layers / vocab | 2560 / 48 / 248320；`embed_tokens`、`lm_head` 为 BF16，不共享 |
| layer_types | `[linear×3, full]×12`（`full_attention` 在参考实现中即 `qwen_sparse_attention`） |
| 残差流 | **hc_count=4** 条并行残差流，`hidden=[tokens, 4×2560]`；初始化为 embedding 重复 4 份（L1480） |
| 无独立层 norm | 没有 `input_layernorm/post_attention_layernorm/model.norm`；归一化全部在 hyper-connection 的 `hc_norm` 内 |
| GatedDeltaNet | 16 K 头 / 48 V 头 / head 128 / conv 4；in_proj_qkv(10240)/z(6144)/a(48)/b(48)/out_proj NF4；**output gate = sigmoid**（`output_gate_type`，L499–501），Qwen3.5 是 silu |
| Full attn | 24 Q / 2 KV / head 256；q_proj 输出 2×24×256（每头 `[q(256), gate(256)]`，L867–870，与 Qwen3.5 `attn_output_gate` 布局相同）；q/k norm 为 `(1+w)`；rotary 64 维、mrope 交错 `[11,11,10]`、theta 1e7；输出 `attn*sigmoid(gate)` |
| QSA indexer | `index_qk_proj` BF16 `[640,2560]`（4 Q 头 + 1 K 头 ×128），budget 2048 / compress 4 → 选 512 块。可见 token ≤ 2051 时全部块被选中，**等价稠密注意力**（L734–763） |
| MoE | 512 专家 top-10，softmax→topk→重归一（L969–978）；`experts.gate_up_proj` `[512,1280,2560]`（前 640 行 gate 后 640 行 up，L951），`experts.down_proj` `[512,2560,640]`，NF4 bs=64 双量化；act silu；shared_expert (gate/up/down 640, NF4) × `sigmoid(shared_expert_gate·x)`（L992–998）；`mlp.gate.weight` BF16 `[512,2560]` |
| Hyper-connection | 每层 2 个（attn/mlp）+ 模型级 `hyper_connection_mixer`（无 block_inject）。数学见下 |
| PLE | 仅 layer_idx=1（`ple_layer_ids=[2]` 一索引）。n-gram 哈希嵌入 16 头 × 160 维 = 2560；`ngram_embedding` 分 128 shard，NF4 bs=32 单量化，每行 80 B + 5 f32 scale；`key_proj [10240,2560]`、`value_proj [2560,2560]` NF4；`norm_key/query/conv [10240]`；`conv1d [10240,1,4]` depthwise dilation=3 |
| RoPE 位置 | 文本 3D 位置相同（mrope），vision 用 `get_rope_index` |

### 关键数学（实现时逐条对照）

**GatedResidual（L1014–1031）**，输入 `H=[4×2560]`：
```
N   = hc_norm(H)                     # 分组 RMSNorm，每流独立 rms，权重 (1+w[s*2560+d])
mix = sigmoid(Wup · silu(Wdown · N / 4))          # [10240] → [320] → [10240]
x   = mean_s( mix[s,:] * N[s,:] )                 # [2560]，块输入
inj = 2 * sigmoid(Winject · N / 4)                # [4]（mixer 无此项）
块输出 y 写回: H[s,:] += inj[s] * y
```
**PLE（L1131–1180, L1235–1255）**：
```
ids: 上下文 = 前 2 个 token（初始为 eos=248044；遇 eos 分段重置，L1115–1129）
mixed = tok0*m0 ^ tok1*m1 (^ tok2*m2)   # int64 回绕乘法 + xor；m 来自 layer_multipliers
head h: id = python_mod(mixed, vocab_sizes[h]) + offset[h]   # 2-gram 头 0..7, 3-gram 头 8..15
emb = concat_h dequant_row(id)  → [2560]
K = norm_key(key_proj(emb)) → [4×2560];  V = value_proj(emb) → [2560];  Q = norm_query(H)
g[s] = Σ_d K[s,d]Q[s,d] / sqrt(2560);  g = sign(g)*sqrt(max(|g|,1e-6))
GV[s,:] = sigmoid(g[s]) * V ;  out = GV + silu(conv1d_dilated3(norm_conv(GV)))  (需 9 行历史)
H += out
```
**MoE 路由**：`p = softmax(W_gate · x)` fp32 → top10 → `p_k / Σp_k`；`out = Σ_k p_k·down_k(silu(gate_k x)·up_k x) + sigmoid(w_sg·x)·shared(x)`。

### 本机硬件特性与约束分析 (4通道 DDR3-1800 + PCIe 3.0 x16 + A770 16GB)
1. **系统内存 (Host RAM)**: 4 通道 DDR3-1800 (实测连续读取带宽 ~38–45 GB/s)。94.5 GB 总权重无法全部常驻 64–96 GB 物理内存，必须依靠 Windows OS mmap 页缓存。未固定页直传 GPU 会触发驱动昂贵的动态锁页/双缓冲，导致 PCIe 传输降速至 5–7 GB/s。因此主机端必须维护预固定的 Pinned Host Staging Ring Buffer (64–128 MB, sycl::malloc_host)，以达满速 PCIe 3.0 DMA 传输。
2. **总线 (PCIe 3.0 x16)**: 单向有效吞吐约 12.5–13.2 GB/s。传输专家必须结合流水线异步预取: 在第 L 层执行 Attention/DeltaNet 计算期间，提前通过独立 SYCL 传输队列异步 DMA 下一步所需的专家数据，隐藏 PCIe 3.0 传输时延。
3. **显存预算 (A770 16GB)**: 静态常驻 5.6 GB，动态工作区 (KV/DeltaNet/Scratchpad) 0.9 GB，GPU LRU 专家缓存 6.5 GB (约 2320 槽)，Windows WDDM/DXGI 桌面合成安全余量 3.0 GB (避免驱动 TDR)。


## 内存预算（A770 16 GB / 主机 64–96 GB）

- GPU 常驻：embed 1.27 GB + lm_head 1.27 GB + 48 层 HC BF16 ≈ 1.26 GB + attention/DeltaNet NF4 ≈ 1.4 GB + shared expert 0.12 GB + router/indexer 0.16 GB + nested_absmax 0.06 GB ≈ **5.6 GB**。
- 专家：每专家 gate_up 1.64 MB + down 0.82 MB + f32 absmax 0.3 MB ≈ **2.8 MB**；LRU 缓存默认 6 GB ≈ 2100 槽（总 24576 专家的 ~9%）。decode 每 token 480 专家 ≈ 1.3 GB 权重流量，PCIe 4.0 实测 ~20 GB/s → **理论上限约 15 tok/s，v1 预期 5–10 tok/s**。prefill 每 chunk（512 token）几乎触及全部专家 → 每 chunk ≈ 62 GB 传输 ≈ 3–4 s。
- 主机：专家 62 GB + PLE 32 GB 全靠 mmap 页缓存；不做 pinned 全驻留。

## 实施步骤

### Step 0：权重懒加载基础设施（`src/utils/SafeTensors.{hpp,cpp}`）
1. 新增 `LoadModelWeightsLazy(model_dir, ctx)`：打开全部分片 `MemoryMappedFile`（只映射不触页），解析 header 建 `name → {shard, TensorMeta}`，不上传任何张量。`ModelWeights` 增加：`std::vector<std::unique_ptr<MemoryMappedFile>> mmaps_`、`host_index_`、`bool has_host(name)`、`HostTensorView host_view(name)`（`{const uint8_t* ptr, shape, dtype, bytes}`）、`Tensor& upload(name)`（按需上传并 `put`）。现有 `LoadSafetensors/LoadModelWeightsFromDir` 不变。
2. 把匿名命名空间中的 `parse_bnb_4bit_quant_state_json` 暴露为公开函数；新增 `Bnb4BitHostWeightRef`（packed/absmax_codes/nested_absmax/nested_quant_map/quant_map 的主机指针 + `Bnb4BitQuantState`）和 `LoadBnb4BitHostRef(weights, name, out, err)`，复用与 `LoadBnb4BitWeightRef`（`SafeTensors.cpp:462`）相同的一致性校验。
3. 单测 `tests/utils/SafeTensorsTests.cpp` 追加：懒加载后 `size()==0`、`host_view` 形状正确、`upload` 后可 `get`。

### Step 1：ModelSpec / 类型（`include/engine/Types.hpp`, `src/utils/ModelSpec.cpp`）
- `ModelFamily::Qwen4Exp`；`struct Qwen4ExpTextConfig`（在 `Qwen35TextConfig` 字段基础上加：`moe_intermediate_size, shared_expert_intermediate_size, num_experts, num_experts_per_tok, norm_topk_prob=true, hc_count, hc_lowrank, ple_layer_ids, ple_embed_dim, ple_conv_kernel_size, ngram_size, heads_per_ngram, ngram_vocab_size_base, make_ngram_vocab_size_divisible_by, split_ngram_parts, indexer_{n_heads,kv_heads,head_dim,budget,compress_ratio}, output_gate_type, eos_token_id(248044)`）；`ModelSpec::qwen4exp_text`。
- `parse_qwen4exp(root, spec)`：仿 `parse_qwen35`（`ModelSpec.cpp:303`），`model_type == "qwen4_exp" || "qwen4_exp_text"` 分派；vision_config 同 Qwen3.5 解析。
- 测试 `tests/models/Qwen4ExpModelSpecTests.cpp`（仿 `DiffusionGemmaModelSpecTests.cpp`，用真实 config.json 片段作 fixture），加入 CMake。

### Step 2：抽取 GatedDeltaNet 内核并加 gate 激活参数（`src/ops/GatedDeltaNetOps.{hpp,cpp}`）
- 把 `Qwen35HybridBnb4Backend::run_linear_delta_decode_gpu`（`Qwen35HybridBnb4Backend.cpp:1201–1545`）与 `run_linear_delta_prefill_gpu_batched`（`:1547–1797`）的内核体移为自由函数，参数打包为 `DeltaNetGeometry{q_heads,v_heads,head_k,head_v,kernel,qkv_dim,z_dim,...}` + `DeltaNetWeights{conv,norm,A_log,dt_bias}` + `DeltaNetState{state,conv_state,conv_head}` + `GateActivation{Silu,Sigmoid}`；`commit_linear_conv_history` 等辅助一起迁移。Qwen3.5 后端改为转调，`GateActivation::Silu`。
- 回归验证：`AilaLayerReferenceTests --backend qwen35_nf4 --write/--compare`（`docs/hybrid-layer-execution.md`）必须逐位一致。
- 48/16 分组（group=3）走 `supports_default_grouped_linear_delta_fastpath`（`:63`），与 Qwen3.5-9B（32/16）同一路径。

### Step 3：新内核（`src/ops/`）
在 `NormOps.cpp` / `ElementwiseOps.cpp` 或新文件 `HyperConnectionOps.cpp` 中：
1. `grouped_rms_norm(x[rows, G*H], w[G*H], eps, out)`：每 (row, g) 一个 work-group，权重按 `(1+w)` 已在加载时预处理（复用 `plus_one_norm_weight` 逻辑）。
2. `hc_mix_reduce(N, mix, out[rows,H], G)`：`out = mean_g(mix*N)`。
3. `hc_inject_add(H[rows,G*H], y[rows,H], inj[rows,G])`：原地残差写回。
4. `ple_gate_value(K, Q, V, GV)` 与 `ple_dilated_conv_silu_add(GVn, conv_state[9 rows], w[10240,4], GV, out)`：decode 单 token 路径 + prefill 多 token 路径（历史来自 state + 前序行）。
5. `router_topk_device` 泛化：现有实现（`NormOps.cpp:719`）WG=128 硬编码；新增 `router_softmax_topk_device(scores, num_experts=512, top_k, ids, weights, per_expert_scale=nullptr)`，WG=512（或每线程 4 专家）；主机参考版从 `DiffusionGemmaBnb4Backend.cpp:42` 的 `router_topk_from_bf16_scores` 提到 `ops` 命名空间供测试对拍。

新文件 `src/ops/MoeExpertOps.{hpp,cpp}`（专家槽位布局：每槽 `packed_gate_up[1280×1280B] | packed_down[2560×320B] | absmax_gu_f32[51200] | absmax_down_f32[25600]`，absmax 已在填充时反量化为 f32，因此内核与 `packed_nf4_gemv_bf16`（`Bnb4BitLinear.cpp:1319`）同结构，SG=16、uint32 加载、SLM qmap）：
1. `moe_decode_gate_up_swiglu_grouped(slots[k], x[1,2560], h[k,640])`：nd_range `(k × 640/16 组)`，每子组一行；gate 行 r 与 up 行 r+640 同一子组内算，输出 `silu(g)*u`（参考 `packed_nf4_gemv_gate_up_swiglu` `:1182`）。
2. `moe_decode_down_weighted_accumulate(slots[k], weights[k], h[k,640], acc_f32[2560])` → 转 bf16 加到 MoE 输出。
3. prefill：复制 `run_grouped_moe_gemm_single_launch`（`Bnb4BitLinear.cpp:2811`）为槽位版：`GroupInfo{expert, slot, token_offset, count}`，权重指针按 `slot`，absmax 取 f32，激活改 silu（`moe_gelu_tanh_gate_up_mul_flat` → silu 版），复用 `moe_gather_rows_flat`/`run_moe_accumulate_gpu`。
4. 单测 `tests/ops/Qwen4ExpKernelTests.cpp`：随机 NF4 数据，对拍 CPU 参考（grouped norm、hc、router512 vs 主机版、decode GEMV vs CPU dequant 矩阵乘、prefill GEMM vs decode 路径逐行一致）。

### Step 4：专家 LRU 缓存（`src/models/Qwen4ExpExpertCache.{hpp,cpp}`）
- `init(ctx, weights, layers, cache_bytes)`：为每层构造 gate_up/down 的 `Bnb4BitHostWeightRef`；分配槽位大张量；nested_absmax/nested_quant_map/quant_map 每层上传 GPU 常驻（用于校验，槽内已 f32）。
- `acquire(layer, expert_ids[]) → slots[]`：LRU（`std::list` + `unordered_map<(layer<<16|expert), slot>`），缺失则填充：主机侧从 mmap 取 packed 行块 + absmax codes → 用 `nested_qmap[code]*nested_absmax[blk/256]+offset` 反量化到 pinned staging（`ctx.alloc_host`，64 MB 环形），`memcpy_h2d_async` 到槽；同一批 acquire 内不淘汰刚命中的槽。
- 统计命中率/字节数，日志 `[Q4E MoE] layer=%d hit=%d/%d bytes=%zu`；env `AILA_Q4E_EXPERT_CACHE_MB`（默认 6144）。
- 单测：容量 3 槽的命中/淘汰序列；填充后槽内容与主机反量化一致。

### Step 5：PLE（`src/models/Qwen4ExpPle.{hpp,cpp}`）
- 加载：读 `layer_multipliers`（I64[3]）、`ngram_heads_vocab_sizes`（I64[16]）→ 计算 offsets；128 个 shard 的 `host_view`（packed + f32 absmax），`rows_per_shard = shard_packed_bytes/80`；校验 `Σvocab ≤ padded_total`。
- `compute_ngram_ids(token_history, out[tokens×16])` 严格复刻 L1115–1174（int64 回绕、python 取模、eos 分段）。
- `lookup_dequant(ids) → bf16 [tokens,2560]`：每行 80 B → 160 值，5 个 block 各 32 值；`quant_map[hi/lo]*absmax[row*5+blk]`；上传到 `buf_.ple_embed`。
- 会话状态：`prev_tokens[2]`、`conv_state[9×10240] f32`（GPU）+ host 镜像用于快照。
- 单测：用 `tools/qwen4exp_reference/dump_ple_golden.py`（直接 import `modeling_qwen4_exp` 的 `_build_layer_multipliers`/`Qwen4ExpTextNGramEmbedding` 逻辑）导出若干 token 序列的 ngram ids 和前 4 行反量化值为 golden JSON，C++ 对比。

### Step 6：后端 `src/models/Qwen4ExpBnb4Backend.{hpp,cpp}`
以 `Qwen35HybridBnb4Backend` 为模板（Session/快照/Truncate/MRoPE/embedding override 接口照搬），但 **自写层循环**，不复用 `HybridLayerExecutor`。

`load()`：
- 校验 family/quant（`bnb_4bit_compute_dtype=bfloat16`）；`max_seq_len` M1 硬上限 2048（超出报错，提示 M2）。
- 用 `weights.upload()` 上传：embed、lm_head（`transpose_weight` 后 `Linear::init(preprocessed=true)`，同 `:958`）、`hyper_connection_mixer.*`、每层 `attn/mlp_hyper_connection.*`（`hc_norm` 走 `(1+w)`；`input_mix_weight_down/up`、`block_inject_weight` 各建 `Linear`）、`mlp.gate.weight`（`Linear`）、`shared_expert_gate.weight`、shared_expert 三个 NF4（gate/up `init_fused_rows`）、DeltaNet 五个 NF4 + `norm/A_log/dt_bias/conv1d`（复用 `:807–888` 逻辑，融合 `linear_all_proj`）、full attn q/k/v/o NF4（`init_fused_rows`）+ `q_norm/k_norm (1+w)`、layer 1 的 PLE 投影/norm/conv。不上传 `experts.*`、`ngram_embedding.*`、`self_attn.indexer.*`（M1）、`mtp.*`。
- `ExpertCache::init`、`Ple::init`。

`ForwardImpl(session, ids, count)`：
```
hidden4 = repeat(embed(ids) [+override], 4)          # [n, 10240]
for L in layers:
    if L==1: hidden4 += PLE(hidden4, ids, session)
    (x, N, inj) = HC_attn(hidden4)                   # grouped_rms_norm + Linear×3 + mix_reduce
    y = DeltaNet(x) | FullAttn(x)                    # 复用 Step2 内核 / ops::attention_* (head 256, 24/2)
    hidden4 = hc_inject_add(hidden4, y, inj)
    (x, N, inj) = HC_mlp(hidden4)
    y = MoE(x) = routed(x) + sigmoid(sg·x)*shared(x)
    hidden4 = hc_inject_add(hidden4, y, inj)
x = HC_mixer(hidden4)                                # 无 inject
logits = lm_head(x[last])
```
- Full attn 复用 `HybridLayerExecutor.hpp:206–345` 的调用序列（`decode_prepare_qgkv_packed_partial`、`attention_decode(+partials)`、`attention_prefill(_cached)`、`sigmoid_mul`），KV cache `[2, max_seq, 256]` bf16。
- MoE：router `Linear` → `router_softmax_topk_device` → d2h ids/weights → `ExpertCache::acquire` → decode 走 grouped GEMV，prefill 走槽位 XMX GEMM；shared expert decode 用 `try_forward_decode_gate_up_swiglu`。
- 快照：DeltaNet 状态（同 Qwen3.5 `CaptureSnapshot` `:2355`）+ PLE conv/token 状态；`Truncate` 恢复同上。
- Profiling：`AILA_PROFILE_Q4E_DECODE` 分段（hc / attn / router / expert_fill / expert_gemm / shared / ple / lm_head）。
- 调试：`AILA_DEBUG_Q4E_DUMP_DIR` 导出每层 `hidden4` 最后一行 bf16 到 `.bin`，供 Step 8 对拍。

### Step 7：引擎集成
- `src/engine/EngineLoad.cpp`：`validate_quantization` 放行 `Qwen4Exp`（bf16 compute dtype）；family 分支填 `config_`；**权重加载改为** `family==Qwen4Exp ? LoadModelWeightsLazy : LoadModelWeightsFromDir`；工厂 `make_unique<Qwen4ExpBnb4Backend>()`；`run_init_warmup=false`；vision 仅在 M3 接入。
- `src/engine/EngineGeneration.cpp`：`family_name()` 加 `"qwen4_exp"`；Qwen3.5 专属的采样调优（`:293`）、`<think>` 结尾检测（`:530`）、递归 prefill anchor（`:535`）、`AILA_Q35_PREFILL_CHUNK`（`:64`）改为 `family==Qwen35Hybrid || family==Qwen4Exp` 视情况共享（chunk/anchor/think 检测共享；采样调优不共享，默认 T=1.0/top_p 0.95/top_k 20 由 generation_config 决定）。
- `src/chat/ChatFormatter.cpp:90`：`Qwen4Exp` 不走 `qwen35_fixed_chat_template`，落到 `tokenizer_chat_template`（tokenizer_config.json 已含；模板用到的 `[::-1]`、`loop.previtem/nextitem`、`tojson`、`items`、`startswith` 均已由 `src/templates/jinja` 支持）；`ChatTemplateEngine.cpp:205` 追加 `reasoning_effort`（默认 `"xhigh"`，可由 `chat_template_kwargs` 覆盖）。`enable_thinking` 默认 true。
- 停止 token：Tokenizer 已把 `<|im_end|>`(248046) 作 eos，`<|endoftext|>`(248044) 作 eot，无需改。
- `CMakeLists.txt`：新增 4 个源文件到 `LIB_SOURCES`，新增 3 个测试可执行文件（仿 `AilaDiffusionGemmaModelSpecTests` 段）。
- `docs/Qwen38FlashNext.md`（结构、内存预算、限制）、`docs/Environment_Variables.md` 追加 `AILA_Q4E_*`；CLAUDE.md 后端表加一行。

### InferRef 深度对齐指南 ($inferref)
本项目强制使用 InferRef (0.11.0 + PyTorch 2.14.0+xpu) 进行数值对齐：
1. **适配器实现**: 在 tests/inferref/ 下建立 Qwen4ExpInferRefAdapter.cpp 与 qwen4_exp.adapter.json，支持 irtensor 输入输出，对照 DiffusionGemmaNf4InferRefAdapter.cpp 标准实现。
2. **对齐矩阵**:
   - 算子级: GatedDeltaNet (Sigmoid 门控), HyperConnection (grouped norm 1+w, mix_reduce, inject_add), MoE Router Top-10, PLE (n-gram hash, 128 shard NF4, conv1d dilation=3)。
   - 模块与单层级: Layer 0 (Linear), Layer 1 (PLE Linear), Layer 3 (Full Attention 24/2, head_dim 256)。
   - 端到端级: 16-token / 64-token trace 对齐 (align run --boundary all)，余弦相似度 > 0.999。


### Step 8：数值参考与 InferRef 自动化对齐（`tools/qwen4exp_reference/`）
- `reference_layers.py`：用 `safetensors` + `bitsandbytes.functional.dequantize_4bit` 在 CPU 反量化前 4 层（含 layer 1 PLE，PLE 行按需从 shard 反量化，不加载整张表），用 `modeling_qwen4_exp` 的模块类（`Qwen4ExpTextDecoderLayer`、`Qwen4ExpTextGatedResidual`）跑 16-token 提示，导出每层 `hidden4` 最后一行与最终 logits（仅前 4 层，无 lm_head 对拍时对 `hyper_connection_mixer` 输出）。
- `compare_layers.py`：读取 `AILA_DEBUG_Q4E_DUMP_DIR` 与 golden，报告每层 max|Δ|/cosine；目标 cosine > 0.999（bf16 残差流误差累积），路由专家 id 集合一致。

## 里程碑
- **M1**（本计划）：Step 0–8，文本 greedy/sampling，ctx ≤ 2048。
- **M2**：QSA indexer（`index_qk_proj` + q/k layernorm + RoPE + 块均值池化 + relu 打分 top-512 + attention mask），indexer key cache `[12, max_seq, 128]`，解除 2048 上限。
- **M3**：vision（此 checkpoint 的 `model.visual.blocks.*` 是 NF4 双量化，`Qwen35VisionEncoder` 只吃 bf16 → 加载时反量化为 bf16 或换 `Bnb4BitLinear`；merger 用 LayerNorm、pos_embed 双线性插值、axial rope 与 Qwen3.5 对齐性需核对）。
- **M4**：`qwen4exp_session_memory`、`Model.cpp` 共享会话/`ForwardDecodeBatch`/IPC/HTTP。

## 验证
1. `pwsh build.ps1`；运行 `AilaQwen4ExpModelSpecTests`、`AilaQwen4ExpKernelTests`、`AilaSafeTensorsTests`、`AilaBnb4BitLinearKernelTests`。
2. Qwen3.5 回归：`AilaLayerReferenceTests --backend qwen35_nf4 --model <0.8B模型> --write/--compare` 逐位一致；`pwsh smoke.ps1 -ModelDir <Qwen3.5>`。
3. `python tools/qwen4exp_reference/reference_layers.py` 生成 golden；`$env:AILA_DEBUG_Q4E_DUMP_DIR=...; .\build\Aila.exe -m F:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2 --max-seq 2048 --greedy --max-tokens 1`；`compare_layers.py` 通过。
4. 端到端：`.\build\Aila.exe -m ... --greedy --max-tokens 64` 对 "1+1=? /no_think" 类提示给出连贯答案；多轮对话 + `truncate_kv_cache` 回滚正确。
5. `pwsh bench.ps1 -ModelDir ... -PromptTokens 512 -GenTokens 64`，记录 tok/s 与专家缓存命中率，写入 `docs/Qwen38FlashNext.md`。

## 已知限制 / 风险
- decode 受 PCIe 带宽限制（~15 tok/s 上限）；后续可评估 CPU 直接算专家或更大缓存。
- prefill 每 512-token chunk 流式全部专家（~62 GB），长提示很慢；M2 后可考虑按层批量预取。
- PLE/专家依赖 OS 页缓存，冷启动首个 token 会有明显 NVMe 延迟。
- 24/2 GQA（group 12）在 head_dim=256 exact-partial 注意力内核上未验证，需在 Step 6 用单测确认。
- 残差流以 bf16 存储（与参考一致），48 层 × 2 次注入的误差与 HF 相当但不逐位相等。
