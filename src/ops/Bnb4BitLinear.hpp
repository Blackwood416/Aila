#pragma once

#include "../core/Context.hpp"
#include "../core/Tensor.hpp"
#include "../utils/SafeTensors.hpp"
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"
#include <string>
#include <unordered_map>
#include <vector>

struct LoraAttachment {
    Tensor lora_a;         // (r, in_features) bf16 GPU
    Tensor lora_b;         // (rows, r) bf16 GPU
    float scaling = 1.0f;
    int output_offset = 0; // starting row offset in the output tensor
    int output_rows = 0;   // number of output rows this attachment covers
};

struct Bnb4BitLinearScratch {
    Tensor input_bf16;
    Tensor output_bf16;
    Tensor weight_bf16;

    int seq_capacity = 0;
    int input_dim_capacity = 0;
    int output_dim_capacity = 0;
    int weight_in_capacity = 0;
    int weight_out_capacity = 0;

    void ensure(Context& ctx, int seq_len, int in_features, int out_features, bool need_weight);
};

class Bnb4BitLinear {
public:
    Bnb4BitLinear() = default;
    Bnb4BitLinear(Bnb4BitLinear&& other) noexcept = default;
    Bnb4BitLinear& operator=(Bnb4BitLinear&& other) noexcept = default;
    Bnb4BitLinear(const Bnb4BitLinear&) = delete;
    Bnb4BitLinear& operator=(const Bnb4BitLinear&) = delete;

    bool init(Context& ctx, const Bnb4BitWeightRef& weight, std::string* error_message = nullptr);
    bool init_fused_rows(Context& ctx,
                         const Bnb4BitLinear& first,
                         const Bnb4BitLinear& second,
                         std::string* error_message = nullptr);
    bool init_fused_rows(Context& ctx,
                         const Bnb4BitLinear& first,
                         const Bnb4BitLinear& second,
                         const Bnb4BitLinear& third,
                         std::string* error_message = nullptr);
    bool init_fused_rows(Context& ctx,
                         const Bnb4BitLinear& first,
                         const Bnb4BitLinear& second,
                         const Bnb4BitLinear& third,
                         const Bnb4BitLinear& fourth,
                         std::string* error_message = nullptr);
    void forward(Context& ctx,
                 Bnb4BitLinearScratch& scratch,
                 Tensor& input,
                 Tensor& output,
                 int seq_len,
                 const sycl::ext::oneapi::bfloat16* add_residual = nullptr);
    // Independent decode rows, not a prefill GEMM. Buffers may have larger
    // capacity shapes; logical row strides are in_features()/out_features().
    // Accepts 1..4 BF16 rows and preserves each row's single-decode arithmetic.
    // Input and output logical spans must not overlap.
    void ForwardDecodeRows(Context& ctx,
                           Bnb4BitLinearScratch& scratch,
                           Tensor& input,
                           Tensor& output,
                           int rows);
    static bool try_forward_decode_gate_up_swiglu(Context& ctx,
                                                  Bnb4BitLinear& fused_gate_up,
                                                  Tensor& input,
                                                  Tensor& output,
                                                  int ff_dim);

    int in_features() const { return in_features_; }
    int out_features() const { return out_features_; }

    // Allow the dense prefill path to select the A770 XMX Joint Matrix GEMM.
    // Off by default; backends opt in after validating numerics for their
    // weight format. The process-wide AILA_BNB4_GEMM_XMX override still wins.
    void set_allow_dense_xmx(bool value) { allow_dense_xmx_ = value; }
    bool allow_dense_xmx() const { return allow_dense_xmx_; }

    void set_allow_async_dequant(bool value) { allow_async_dequant_ = value; }
    bool allow_async_dequant() const { return allow_async_dequant_; }

    struct PrefillDispatchStats {
        uint64_t fused_gemm = 0;
        uint64_t cached_weight = 0;
        uint64_t on_demand_dequant = 0;
        uint64_t sync_wait = 0;
    };
    static PrefillDispatchStats get_prefill_dispatch_stats();
    static void reset_prefill_dispatch_stats();

    void set_lora(std::vector<LoraAttachment>&& attachments) {
        lora_attachments_ = std::move(attachments);
    }
    bool has_lora() const { return !lora_attachments_.empty(); }

private:
    bool init_fused_rows_impl(Context& ctx,
                              const std::vector<const Bnb4BitLinear*>& sources,
                              std::string* error_message);
    void finish_init(Context& ctx);
    void ensure_primitive(Context& ctx, int seq_len);
    void dequantize_weight(Context& ctx, Tensor& weight_bf16_view);

    Bnb4BitWeightRef weight_{};
    Tensor owned_packed_weight_;
    Tensor absmax_f32_;
    Tensor cached_weight_bf16_;
    int in_features_ = 0;
    int out_features_ = 0;
    bool force_dequant_cache_ = false;
    bool release_quant_after_cache_ = false;
    bool cache_dequantized_weight_ = false;
    bool cached_weight_ready_ = false;
    bool allow_dense_xmx_ = false;
    bool allow_async_dequant_ = false;
    dnnl::matmul decode_prim_;
    dnnl::memory::desc decode_src_md_;
    dnnl::memory::desc decode_weight_md_;
    dnnl::memory::desc decode_dst_md_;
    bool decode_inited_ = false;

    dnnl::memory decode_src_mem_;
    dnnl::memory decode_weight_mem_;
    dnnl::memory decode_dst_mem_;
    dnnl::memory decode_scratchpad_mem_;
    Tensor decode_scratchpad_;
    bool decode_mem_inited_ = false;
    void* decode_src_ptr_ = nullptr;
    void* decode_weight_ptr_ = nullptr;
    void* decode_dst_ptr_ = nullptr;
    std::unordered_map<int, dnnl::memory> decode_args_;

    struct CachedPrimitive {
        dnnl::matmul prim;
        dnnl::memory::desc src_md;
        dnnl::memory::desc weight_md;
        dnnl::memory::desc dst_md;
        dnnl::memory src_mem;
        dnnl::memory weight_mem;
        dnnl::memory dst_mem;
        dnnl::memory::desc scratchpad_md;
        bool mem_inited = false;
        void* src_ptr = nullptr;
        void* weight_ptr = nullptr;
        void* dst_ptr = nullptr;
        std::unordered_map<int, dnnl::memory> args;
    };
    std::unordered_map<int, CachedPrimitive> prim_cache_;

    std::vector<LoraAttachment> lora_attachments_;
    Tensor lora_scratch_;
    float* ensure_lora_scratch(Context& ctx, size_t count);
    void apply_lora_decode(Context& ctx, Tensor& input, Tensor& output);
    void apply_lora_prefill(Context& ctx, Tensor& input, Tensor& output, int seq_len);
};

// Grouped Mixture-of-Experts over one packed NF4 tensor per layer.
// Expert weights are stored concatenated along the row axis:
//   gate_up rows: num_experts * 2 * moe_intermediate_size
//   down    rows: num_experts * moe_intermediate_size
// forward() is host-driven for v1 correctness: tokens are grouped by expert on
// the host, per-expert rows are dequantized on the GPU, and per-group GEMMs
// run through oneDNN.
class Bnb4GroupedMoE {
public:
    bool init(Context& ctx, const Bnb4BitWeightRef& gate_up,
              const Bnb4BitWeightRef& down, int hidden_size,
              int moe_intermediate_size, int num_experts,
              std::string* error_message = nullptr);

    // input: [tokens * hidden_size] bf16 host
    // expert_ids: [tokens * top_k] int (selected expert indices)
    // expert_weights: [tokens * top_k] float (renormalized, per-expert scaled)
    // output: [tokens * hidden_size] bf16 host
    void forward(Context& ctx,
                 const std::vector<sycl::ext::oneapi::bfloat16>& input,
                 const std::vector<int>& expert_ids,
                 const std::vector<float>& expert_weights,
                 std::vector<sycl::ext::oneapi::bfloat16>& output,
                 int tokens, int top_k);

    // Batched-transfer MoE forward, numerically identical to forward().
    // input is [tokens * hidden] bf16 on the device.  transfer is one
    // device buffer with capacity [tokens * top_k, hidden] that aliases the
    // gathered per-expert source rows and the per-expert down outputs (each
    // expert uses a disjoint row range, and the in-order queue separates the
    // read of a range from its later overwrite).  All expert groups share a
    // single gather kernel and a single d2h instead of one blocking h2d/d2h
    // pair per group.  token_ids_dev is [tokens * top_k] int32 scratch.
    void forward_batched(Context& ctx, Tensor& input, Tensor& transfer,
                         Tensor& token_ids_dev,
                         const std::vector<int>& expert_ids,
                         const std::vector<float>& expert_weights,
                         std::vector<sycl::ext::oneapi::bfloat16>& output,
                         int tokens, int top_k);
    void forward_batched_impl(
        Context& ctx, Tensor& input, Tensor& transfer, Tensor& token_ids_dev,
        const std::vector<int>& expert_ids,
        const std::vector<float>& expert_weights,
        std::vector<sycl::ext::oneapi::bfloat16>& output, int tokens,
        int top_k);

    void forward_batched(
        Context& ctx, Tensor& input, Tensor& transfer, Tensor& token_ids_dev,
        Tensor& weights_dev, Tensor& output,
        const std::vector<int>& expert_ids,
        const std::vector<float>& expert_weights, int tokens, int top_k);
    void forward_batched_impl(
        Context& ctx, Tensor& input, Tensor& transfer, Tensor& token_ids_dev,
        Tensor& weights_dev, Tensor& output,
        const std::vector<int>& expert_ids,
        const std::vector<float>& expert_weights, int tokens, int top_k);

    // GPU-resident MoE forward.  input is [tokens * hidden] bf16 on the
    // device; output is [tokens * hidden] bf16 on the device and is zeroed at
    // the start.  The per-expert dequant and oneDNN matmul steps are the same
    // as forward(); only the gather and the final bf16 accumulation move on
    // device, preserving the sorted-expert accumulation order bitwise.
    void forward_gpu(Context& ctx, Tensor& input, Tensor& output,
                     const std::vector<int>& expert_ids,
                     const std::vector<float>& expert_weights,
                     int tokens, int top_k);

    void forward_batched_gpu(
        Context& ctx, Tensor& input, Tensor& transfer, Tensor& token_ids_dev,
        Tensor& output, const std::vector<int>& expert_ids,
        const std::vector<float>& expert_weights, int tokens, int top_k);

    struct GroupInfo {
        int expert_id;
        int token_offset;
        int token_count;
    };

    int hidden_size() const { return hidden_; }
    int moe_intermediate_size() const { return moe_intermediate_; }
    int num_experts() const { return num_experts_; }

private:
    struct ExpertWeightSet {
        Bnb4BitWeightRef ref{};
        bool ready = false;

        bool init(Context& ctx, const Bnb4BitWeightRef& weight_ref,
                  std::string* error_message);
    };

    struct MatmulPrim {
        bool inited = false;
        dnnl::matmul prim{};
        dnnl::memory::desc src_md{};
        dnnl::memory::desc w_md{};
        dnnl::memory::desc dst_md{};
    };

    ExpertWeightSet gate_up_;
    ExpertWeightSet down_;
    int hidden_ = 0;
    int moe_intermediate_ = 0;
    int num_experts_ = 0;
    int gate_up_in_features_ = 0;
    int gate_up_rows_per_expert_ = 0;
    int down_in_features_ = 0;
    int down_rows_per_expert_ = 0;

    std::unordered_map<uint64_t, MatmulPrim> prims_;
    Tensor src_;
    Tensor gate_up_out_;
    Tensor proj_;
    Tensor down_out_;
    Tensor gate_up_w_;
    Tensor down_w_;
    Tensor tokens_dev_;
    Tensor weights_dev_;



    Tensor batched_gate_up_out_;
    Tensor batched_proj_;
    Tensor group_infos_dev_;
    Tensor transfer_indices_dev_;
    Tensor transfer_weights_dev_;
    int accumulate_scratch_rows_ = 0;
    int batched_scratch_rows_ = 0;
    int batched_scratch_groups_ = 0;

    // oneDNN memory handles from run_matmul must outlive their asynchronous
    // executions on this A770/Level-Zero stack; destroying them while a batch
    // of matmuls is still queued can deadlock the driver.  They are released
    // only after the queue is drained.
    std::vector<dnnl::memory> pending_memories_;
    int scratch_rows_ = 0;
    std::vector<int> route_counts_;
    std::vector<int> route_offsets_;
    std::vector<int> route_cursors_;
    std::vector<int> route_token_ids_;
    std::vector<float> route_weights_;
    std::vector<int> active_experts_;
    int route_max_rows_ = 0;

    void build_route_layout(const std::vector<int>& expert_ids,
                            const std::vector<float>& expert_weights,
                            int tokens, int top_k);
    void ensure_scratch(Context& ctx, int rows);
    void run_matmul(Context& ctx, int rows, int in_n, int out_n,
                    uint64_t prim_key, Tensor& src, Tensor& w, Tensor& dst);
    static void dequant_subset(Context& ctx, const ExpertWeightSet& set,
                               int expert, int rows_per_expert, Tensor& out);
};

void run_fused_softmax_soft_embed_xmx(
    Context& ctx,
    const float* logits_ptr,
    const float* row_max_ptr,
    const float* row_inv_sum_ptr,
    const sycl::ext::oneapi::bfloat16* embed_weight,
    sycl::ext::oneapi::bfloat16* output,
    int canvas_len, int hidden, int vocab, float embed_scale = 1.0f);

