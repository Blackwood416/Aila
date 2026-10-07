#pragma once

#include "ops/MoeExpertOps.hpp"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <atomic>

namespace ops {
namespace cpu {

constexpr int kHiddenDim = 2560;
constexpr int kIntermediateDim = 640;
constexpr int kBlockSize = 64;
constexpr int kMaxExpertBatchTasks = 40;
struct CpuExpertTask { MoeExpertSlotHost slot; int token = 0; };

enum class CpuKernelBackend {
    Auto = 0,
    Scalar = 1,
    Avx2 = 2,
    TbbAvx2 = 3,
    GpuOrderAvx2 = 4, // experimental DG2 lane order, not a cross-device exactness guarantee
    GpuOrderPreparedAvx2 = 5, // batch-only, shared lane-major inputs and register NF4 lookup
    Nf4Q8Avx2 = 6, // experimental batch-only NF4 LUT + Q8 activations; approximate arithmetic
    Nf4Q8Row4Avx2 = 7, // batch-only four independent dots; preserves the Nf4Q8Avx2 row arithmetic
    Nf4Q8Block64Avx2 = 8, // batch-only 256-bit NF4 block decode; preserves Nf4Q8Avx2 half order
    Nf4Q8Activation64Avx2 = 9 // independent approximation: one activation scale and integer dot merge per 64
};

struct CpuExpertPhaseTimings {
    uint64_t gate_up_ns = 0, quantize_ns = 0, down_ns = 0;
    // Zero denotes TBB's original automatic partitioning; otherwise these are
    // logical coarse chunks, not observed scheduler task counts.
    int gate_chunks = 0, down_chunks = 0;
};

// Fault injection for task lifetime tests only. Production callers leave this
// null; storage must outlive the synchronous batch call, including its drain.
struct CpuExpertDagTestHooks {
    int fail_gate_tile = -1;
    std::atomic<int>* active_tiles = nullptr;
};

// Two flat parallel stages over all expert rows. Caller supplies [tasks,I] h,
// [tasks,H] out; no nested per-expert parallel_for or repeated x conversion.
// nf4_scale_shift only affects Q8 backends: effective nested blocksize 256
// uses an exact integer shift. Default false retains division for CPU shadows.
// A64 specialization defaults to the cached AILA_Q4E_CPU_A64_SPECIALIZE flag;
// A64 row2 defaults to cached AILA_Q4E_CPU_A64_ROW2. Explicit overrides
// let tests compare each independent path without changing ENV. The cached,
// default-off AILA_Q4E_CPU_PER_EXPERT_DAG applies only to A64 and only when
// timings is null: overlapping phases cannot use the old serial timing fields.
// It returns only after every descendant task drains, including exceptions.
void cpu_expert_batch_forward(const CpuExpertTask* tasks, int count,
    const bf16* x, int tokens, bf16* h, float* out, const float* quant_map = nullptr,
    CpuKernelBackend backend = CpuKernelBackend::GpuOrderAvx2,
    CpuExpertPhaseTimings* timings = nullptr, bool nf4_scale_shift = false,
    std::optional<bool> a64_specialize_override = std::nullopt,
    std::optional<bool> a64_row2_override = std::nullopt,
    std::optional<bool> per_expert_dag_override = std::nullopt,
    const CpuExpertDagTestHooks* dag_test_hooks = nullptr);

// Gate/Up SwiGLU:
// Inputs:
//   slot: Host expert slot containing packed gate/up weights and nested/unnested scales
//   x: [tokens, in_features] in bf16
// Output:
//   h: [tokens, intermediate_size] in bf16, where h[t, r] = bf16(silu(gate[t, r]) * up[t, r])
void cpu_expert_gate_up_swiglu(
    const MoeExpertSlotHost& slot,
    const bf16* x,
    bf16* h,
    int tokens,
    const float* quant_map = nullptr,
    int in_features = kHiddenDim,
    int intermediate_size = kIntermediateDim,
    int blocksize = kBlockSize,
    CpuKernelBackend backend = CpuKernelBackend::Auto);

// Down projection:
// Inputs:
//   slot: Host expert slot containing packed down weights and nested/unnested scales
//   h: [tokens, intermediate_size] in bf16
// Output:
//   out: [tokens, out_features] in float (FP32)
// Optional weight and accumulate:
//   if accumulate is true, out[t, r] += weight * down_sum; else out[t, r] = weight * down_sum;
void cpu_expert_down(
    const MoeExpertSlotHost& slot,
    const bf16* h,
    float* out,
    int tokens,
    const float* quant_map = nullptr,
    float weight = 1.0f,
    bool accumulate = false,
    int intermediate_size = kIntermediateDim,
    int out_features = kHiddenDim,
    int blocksize = kBlockSize,
    CpuKernelBackend backend = CpuKernelBackend::Auto);

// Full expert forward:
// Combines Gate/Up SwiGLU and Down projection.
// scratch_h: optional caller-provided buffer of size [tokens * intermediate_size] (bf16).
// If scratch_h is nullptr, uses an internal stack buffer for tokens <= 4 (zero heap allocations).
void cpu_expert_forward(
    const MoeExpertSlotHost& slot,
    const bf16* x,
    float* out,
    int tokens,
    bf16* scratch_h = nullptr,
    const float* quant_map = nullptr,
    float weight = 1.0f,
    bool accumulate = false,
    int in_features = kHiddenDim,
    int intermediate_size = kIntermediateDim,
    int out_features = kHiddenDim,
    int blocksize = kBlockSize,
    CpuKernelBackend backend = CpuKernelBackend::Auto);

// Explicit scalar reference kernels
void cpu_expert_gate_up_swiglu_scalar(
    const MoeExpertSlotHost& slot,
    const bf16* x,
    bf16* h,
    int tokens,
    const float* quant_map = nullptr,
    int in_features = kHiddenDim,
    int intermediate_size = kIntermediateDim,
    int blocksize = kBlockSize);

void cpu_expert_down_scalar(
    const MoeExpertSlotHost& slot,
    const bf16* h,
    float* out,
    int tokens,
    const float* quant_map = nullptr,
    float weight = 1.0f,
    bool accumulate = false,
    int intermediate_size = kIntermediateDim,
    int out_features = kHiddenDim,
    int blocksize = kBlockSize);

void cpu_expert_forward_scalar(
    const MoeExpertSlotHost& slot,
    const bf16* x,
    float* out,
    int tokens,
    bf16* scratch_h = nullptr,
    const float* quant_map = nullptr,
    float weight = 1.0f,
    bool accumulate = false,
    int in_features = kHiddenDim,
    int intermediate_size = kIntermediateDim,
    int out_features = kHiddenDim,
    int blocksize = kBlockSize);

// Explicit AVX2 kernels (single-threaded)
void cpu_expert_gate_up_swiglu_avx2(
    const MoeExpertSlotHost& slot,
    const bf16* x,
    bf16* h,
    int tokens,
    const float* quant_map = nullptr,
    int in_features = kHiddenDim,
    int intermediate_size = kIntermediateDim,
    int blocksize = kBlockSize);

void cpu_expert_down_avx2(
    const MoeExpertSlotHost& slot,
    const bf16* h,
    float* out,
    int tokens,
    const float* quant_map = nullptr,
    float weight = 1.0f,
    bool accumulate = false,
    int intermediate_size = kIntermediateDim,
    int out_features = kHiddenDim,
    int blocksize = kBlockSize);

void cpu_expert_forward_avx2(
    const MoeExpertSlotHost& slot,
    const bf16* x,
    float* out,
    int tokens,
    bf16* scratch_h = nullptr,
    const float* quant_map = nullptr,
    float weight = 1.0f,
    bool accumulate = false,
    int in_features = kHiddenDim,
    int intermediate_size = kIntermediateDim,
    int out_features = kHiddenDim,
    int blocksize = kBlockSize);

// Explicit oneTBB + AVX2 parallel kernels
void cpu_expert_gate_up_swiglu_tbb_avx2(
    const MoeExpertSlotHost& slot,
    const bf16* x,
    bf16* h,
    int tokens,
    const float* quant_map = nullptr,
    int in_features = kHiddenDim,
    int intermediate_size = kIntermediateDim,
    int blocksize = kBlockSize);

void cpu_expert_down_tbb_avx2(
    const MoeExpertSlotHost& slot,
    const bf16* h,
    float* out,
    int tokens,
    const float* quant_map = nullptr,
    float weight = 1.0f,
    bool accumulate = false,
    int intermediate_size = kIntermediateDim,
    int out_features = kHiddenDim,
    int blocksize = kBlockSize);

void cpu_expert_forward_tbb_avx2(
    const MoeExpertSlotHost& slot,
    const bf16* x,
    float* out,
    int tokens,
    bf16* scratch_h = nullptr,
    const float* quant_map = nullptr,
    float weight = 1.0f,
    bool accumulate = false,
    int in_features = kHiddenDim,
    int intermediate_size = kIntermediateDim,
    int out_features = kHiddenDim,
    int blocksize = kBlockSize);

} // namespace cpu
} // namespace ops
