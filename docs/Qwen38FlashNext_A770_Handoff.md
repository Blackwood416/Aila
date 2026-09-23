# Qwen3.8-Flash-Next (qwen4_exp) A770 Handoff

## 1. Scope & Current State

- **Target Model**: `F:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2`
  - **Architecture**: `qwen4_exp` (125B total parameters / 6B active parameters / 51B n-gram PLE embedding table).
  - **Quantization**: bitsandbytes NF4 (nested double quantization block_size=64 / single quantization block_size=32), 131 safetensors shards, 94.5 GiB total size on disk.
  - **Reference Implementation**: `C:\Users\Administrator\AppData\Local\Programs\Python\Python313\Lib\site-packages\transformers\models\qwen4_exp\modeling_qwen4_exp.py`.
- **Milestone Status**:
  - **M1 (Completed & Verified)**:
    - Text generation correctness for context length <= 2048 (QSA acts as dense attention).
    - End-to-end greedy and sampling generation via `Aila.exe` on Intel Arc A770 16GB.
    - Full Transformer execution: Layer 1 PLE, 48-layer Hyper-Connection, 36-layer GatedDeltaNet (Sigmoid gate), 12-layer Full Attention (24/2 GQA + MRoPE), 512-expert MoE (top-10 dynamic LRU swap + shared expert), final Mixer reduction, and `lm_head`.
    - 100% zero-regression on existing backends (Qwen3.5, DiffusionGemma).
  - **Next Milestones**:
    - **M2**: QSA Indexer hardware kernel & sparse KV indexing, lifting sequence length limit.
    - **M3**: Vision tower (Vision Tower `model.visual.blocks.*` BNB-NF4 decoding & multimodal forward).
    - **M4**: Multi-session sharing, continuous batching, IPC v2 worker & Drogon HTTP/SSE server deployment.

---

## 2. Hardware Characteristics & Physical Constraints

This platform has very specific hardware constraints that governed all design choices:
1. **Host RAM & Storage**:
   - **Specs**: 4-channel DDR3-1800 MHz (measured continuous read bandwidth ~38–45 GB/s), 64–96 GB capacity.
   - **Weight Management**: The 94.5 GB checkpoint **cannot be fully resident in RAM**. It is mapped via OS page cache using `SafeTensors::LoadModelWeightsLazy`.
   - **Pinned Memory Buffer**: Passing unpinned virtual memory pages from mmap directly to Level-Zero / DirectX GPU drivers causes dynamic page-locking and double-buffering, dropping PCIe throughput to 5–7 GB/s. Therefore, a pinned host staging ring buffer (based on `sycl::malloc_host`) is maintained in `Qwen4ExpExpertCache`, achieving full line-rate PCIe 3.0 DMA (~12.5 GB/s).
2. **GPU & Bus (Intel Arc A770 16GB)**:
   - **Bus**: **PCIe 3.0 x16** (effective single-direction bandwidth ~12.5–13.2 GB/s, not PCIe 4.0).
   - **Transfer Bottleneck**: Each expert slot takes 2.76 MB (gate_up 1.64 MB + down 0.82 MB + f32 absmax 0.3 MB). 48 layers * top-10 requires up to 480 expert slots (~1.34 GB) per token decode in cold-start.
   - **VRAM Allocation Budget (16 GB total)**:
     - GPU Static Weights (Embedding, lm_head, 48-layer Hyper-Connection, Attention/DeltaNet, Shared Expert, Router): ~5.4 GB.
     - Dynamic Runtime Workspace (KV Cache, DeltaNet recurrence states, scratchpads): ~0.9 GB.
     - GPU LRU Expert Cache: ~4.0 GB (~1450 slots, configurable via `AILA_Q4E_EXPERT_CACHE_MB`).
     - Total resident VRAM after model load: **~9422 MB (~59.1% VRAM usage)**.
     - Remaining safety headroom: **~6.5 GB free**, completely eliminating Windows WDDM TDR driver resets.

---

## 3. Implemented Modules & Architecture

### Step 0: Lazy Loading Infrastructure (`src/utils/SafeTensors.{hpp,cpp}`)
- `LoadModelWeightsLazy(model_dir, ctx)`: Memory-maps all 131 shard files, parsing only JSON headers to construct host tensor metadata indices without uploading anything to the GPU.
- `Bnb4BitHostWeightRef` & `LoadBnb4BitHostRef`: Supports nested quantization state parsing directly from mmap host pointers with safety boundary checks.
- `ModelWeights::upload(name, ctx)`: Enables on-demand GPU upload of static weights.

### Step 1: Types & ModelSpec (`include/engine/Types.hpp`, `src/utils/ModelSpec.{hpp,cpp}`)
- Registered `ModelFamily::Qwen4Exp`.
- Added `struct Qwen4ExpTextConfig` supporting `hc_count=4`, `hc_lowrank=320`, 512 experts with top-10 routing, Sigmoid gate activations, PLE layer IDs, and MRoPE sections `[11, 11, 10]`.

### Step 2: GatedDeltaNet Decoupling (`src/ops/GatedDeltaNetOps.{hpp,cpp}`)
- Extracted `run_linear_delta_decode_gpu` and `run_linear_delta_prefill_gpu_batched` into standalone free operators.
- Added `enum class GateActivation { Silu, Sigmoid }`. Qwen3.5 uses `Silu` (verified 100% bitwise regression passing in `AilaQwen35GroupedBf16Tests`), while Qwen4Exp uses `Sigmoid`.

### Step 3: Core Qwen4Exp Operators
- **Hyper-Connection (`src/ops/HyperConnectionOps.{hpp,cpp}`)**:
  - `grouped_rms_norm`: Grouped RMSNorm across 4 residual streams with `(1 + w)` weight scaling.
  - `hc_mix_reduce`: Channel mixing reduction `mean_g(mix * N)`.
  - `hc_inject_add`: In-place residual injection `H[s] += inj[s] * y`.
  - `ple_gate_value` & `ple_dilated_conv_silu_add`: PLE dot-product gating (`sign(g)*sqrt(max(|g|, 1e-6))`) and 9-step history dilated 1D depthwise convolution (kernel=4, dilation=3).
- **MoE Experts (`src/ops/MoeExpertOps.{hpp,cpp}`)**:
  - `router_softmax_topk_device`: 512-expert top-10 router on GPU with Softmax and probability re-normalization.
  - `moe_decode_gate_up_swiglu_grouped`: Subgroup-cooperative NF4 GEMV calculating gate and up simultaneously with uint32 vector loads and SLM qmap.
  - `moe_decode_down_weighted_accumulate`: Single-kernel down projection and weighted accumulation directly into register accumulators.

### Step 4: Expert LRU Cache (`src/models/Qwen4ExpExpertCache.{hpp,cpp}`)
- Contiguous GPU slot memory pool with double-quantized NF4 absmax expansion to float32 on CPU.
- Pinned staging ring buffer for high-speed DMA uploads.
- Doubly-linked LRU list with same-step reservation flags to prevent intra-batch eviction collision.

### Step 5: PLE Module (`src/models/Qwen4ExpPle.{hpp,cpp}`)
- Zero-copy indexing across 128 `ngram_embedding` shards.
- Bit-exact replication of 64-bit int rolling hash with EOS (248044) context resets.
- State snapshot capture & rollback for multi-turn conversations.

### Step 6 & 7: Model Backend & Engine Integration
- `src/models/Qwen4ExpBnb4Backend.{hpp,cpp}`: Complete 48-layer execution pipeline, MRoPE position encoding, GQA 24/2 Full Attention, and Session snapshot management.
- `src/engine/EngineLoad.cpp`, `EngineGeneration.cpp`, `Engine.hpp`: Engine loader branching, thinking tag streaming controller, and prefill chunk coordination.
- `tests/inferref/Qwen4ExpInferRefAdapter.cpp`: Standard InferRef adapter supporting `.irtensor` numerical validation.

---

## 4. Build, Test & Run Guide

### Build
```powershell
pwsh -NoProfile -File build.ps1 -BuildDir build-refactor
```

### Run CTest Suites
```powershell
ctest --test-dir build-refactor -R "AilaSafeTensorsTests|AilaQwen4ExpModelSpecTests|AilaQwen4ExpKernelTests|AilaQwen4ExpModuleTests" --output-on-failure
```
All 4 test suites pass cleanly on Arc A770 in ~2 seconds.

### Run End-to-End CLI Inference
```powershell
# Direct factual / no-think mode
.\build-refactor\Aila.exe -m F:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2 --max-seq 256 --thinking-budget 0 --max-tokens 32 --messages-json tmp/test_msg.json

# Interactive chat mode
.\build-refactor\Aila.exe -m F:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2 --max-seq 512
```

### Verified Outputs
1. **Math Query**: `"1+1=?"`
   - With thinking: `<think>\nWe need answer simple user: "1+1=?" Need final concise.\n</think>\n\n2`
   - Without thinking (`--thinking-budget 0`): `1 + 1 = **2**`
2. **Factual Query**: `"What is the capital of France?"`
   - Output: `The capital of France is **Paris**.`

---

## 5. Notes for Successor Agents (Next Steps)

1. **Milestone M2 (QSA Indexer Hardware Kernel)**:
   - Location: `self_attn.indexer` (`index_qk_proj` BF16 `[640, 2560]`).
   - Task: Implement `QSAIndexer` kernel that compresses 4 tokens per block, computes block-level similarity, and selects top-512 blocks when sequence length exceeds 2048.
   - Reference: `modeling_qwen4_exp.py` L734–763 (`Qwen4ExpTextQSAIndexer`).
2. **PCIe 3.0 Pipeline Overlap**:
   - Currently, expert swapping occurs synchronously during layer execution.
   - Win: In layer L, while running Attention / DeltaNet on the compute queue, use a dedicated SYCL in-order copy queue to prefetch missing experts for layer L+1 into the staging buffer and DMA to GPU, hiding transfer latency.
3. **Milestone M3 (Vision Tower)**:
   - The vision tower in this checkpoint (`model.visual.blocks.*`) uses BNB-NF4 double quantization.
   - Integrate with `Qwen35VisionEncoder` / `Bnb4BitLinear` to support multimodal image inputs.
4. **Milestone M4 (Server & Batching)**:
   - Wire `Qwen4ExpBnb4Backend` into `src/inference/Model.cpp` for shared-model multi-session serving and Drogon HTTP/SSE server (`AilaServer.exe`).
