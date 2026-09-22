#pragma once

#include "IModelBackend.hpp"
#include "HybridLayerExecution.hpp"
#include "../ops/Ops.hpp"
#include <cstddef>
#include <memory>
#include <vector>
#include <unordered_map>
#include <sycl/sycl.hpp>

class Qwen35HybridTextBackend : public IModelBackend {
    struct SessionOwner {};

    struct LayerCache {
        Tensor k;
        Tensor v;
        Tensor linear_state; // f32 [value heads, key dim, value dim]
        Tensor linear_conv_state; // f32 [kernel - 1, conv channels]
        std::vector<float> host_linear_state;
        std::vector<float> host_linear_conv_state;
        bool device_state_dirty = false;
        int linear_conv_head = 0;
    };
    struct LayerStateSnapshot {
        std::vector<float> linear_state_host;
        std::vector<float> linear_conv_state_host;
        int linear_conv_head = 0;
    };
    struct StateSnapshot {
        std::vector<LayerStateSnapshot> layers;
    };

public:
    // Weights, mutable oneDNN bindings, activations and logits use one
    // serialized execution slot. Every Session owns independent model state.
    class Session {
    public:
        ~Session();
        Session(const Session&) = delete;
        Session& operator=(const Session&) = delete;
        Session(Session&&) = delete;
        Session& operator=(Session&&) = delete;

        int CurrentLength() const { return currentLength_; }
        int Capacity() const { return capacity_; }
        std::size_t CacheBytes() const { return cacheBytes_; }
        // Float payload only: persistent host mirrors and bounded snapshots.
        std::size_t HostReservationBytes() const { return hostReservationBytes_; }

    private:
        friend class Qwen35HybridTextBackend;
        friend struct aila::models::HybridDecodeBatchExecutor;
        Session(Context& ctx, int capacity, std::shared_ptr<const SessionOwner> owner,
                bool boundedSnapshots, std::size_t maxSnapshots);
        Context* context_ = nullptr;
        std::shared_ptr<const SessionOwner> owner_;
        int currentLength_ = 0;
        int capacity_ = 0;
        std::size_t cacheBytes_ = 0;
        std::size_t hostReservationBytes_ = 0;
        bool boundedSnapshots_ = true;
        std::size_t maxSnapshots_ = 4;
        std::vector<LayerCache> layerCaches_;
        std::unordered_map<int, StateSnapshot> snapshots_;
        Tensor mropePosT_, mropePosH_, mropePosW_;
        int mropePromptLength_ = 0;
        int mropeTextPositionDelta_ = 0;
        std::vector<int> embeddingOverridePositions_;
        std::vector<sycl::ext::oneapi::bfloat16> embeddingOverrideValues_;
        int embeddingOverrideHiddenSize_ = 0;
    };

    Qwen35HybridTextBackend();
    ~Qwen35HybridTextBackend() override;

    bool load(Context& ctx,
              ModelWeights& weights,
              const ModelSpec& spec,
              int max_seq_len,
              std::string* error_message) override;

    Tensor& forward(Context& ctx, const int* token_ids_device, int seq_len) override;
    void reset() override;
    bool truncate_kv_cache(int new_len) override;
    int get_current_context_len() const override {
        return defaultSession_ ? defaultSession_->CurrentLength() : 0;
    }
    int max_seq_len() const override { return max_seq_len_; }
    int vocab_size() const override { return cfg_.vocab_size; }
    ModelFamily family() const override { return ModelFamily::Qwen35Hybrid; }
    bool supports_vision_embedding_override() const override { return true; }

    // Explicit sessions require BF16 KV and the DeltaNet path. The legacy
    // default session retains its cache/storage and multi-row logits behavior.
    std::unique_ptr<Session> CreateSession(Context& ctx, std::size_t maxSnapshots = 4);
    // Borrowed last-token row, consumed before another operation on the slot.
    Tensor& Forward(Context& ctx, Session& session, const int* tokenIdsDevice, int count);
    bool SupportsDecodeBatch() const noexcept;
    Tensor& ForwardDecodeBatch(Context& ctx, const std::vector<Session*>& sessions,
                               const int* tokenIdsDevice);
    // Legacy text ABI compatibility: retain the original multi-row layout.
    Tensor& ForwardLegacyLogits(Context& ctx, Session& session, const int* tokenIdsDevice, int count);
    void Reset(Session& session);
    // Restore the nearest retained checkpoint <= length, or reset to zero.
    int Truncate(Session& session, int length);

    void set_embedding_overrides(const std::vector<int>& positions,
                                 const std::vector<sycl::ext::oneapi::bfloat16>& embeddings,
                                 int hidden_size) override;
    void clear_embedding_overrides() override;
    void set_mrope_positions(Context& ctx,
                             const std::vector<int>& pos_t,
                             const std::vector<int>& pos_h,
                             const std::vector<int>& pos_w,
                             int text_pos_delta) override;
    void clear_mrope_positions() override;

private:
    struct Layer {
        bool is_linear = false;

        Tensor* input_ln_weight = nullptr;
        Tensor* post_attn_ln_weight = nullptr;

        // full attention branch
        Linear q_proj, k_proj, v_proj, o_proj;
        Linear qkv_proj;
        Tensor* qkv_weight_jm = nullptr; // bf16 [full_fused_qkv_dim, hidden]
        Tensor* o_weight_jm = nullptr; // bf16 [hidden, full_q_dim]
        Tensor* q_norm_weight = nullptr;
        Tensor* k_norm_weight = nullptr;

        // linear attention branch
        Linear linear_qkv_proj, linear_z_proj, linear_o_proj;
        Linear linear_all_proj;
        Tensor* linear_all_weight_jm = nullptr; // bf16 [linear_all_dim, hidden]
        Tensor* linear_o_weight_jm = nullptr; // bf16 [hidden, linear_kv_dim]
        Linear linear_a_proj, linear_b_proj;
        Tensor* linear_norm_weight = nullptr; // f32
        Tensor* linear_A_log = nullptr;       // f32
        Tensor* linear_dt_bias = nullptr;     // bf16
        Tensor* linear_conv1d_weight = nullptr; // bf16 [channels, 1, kernel]

        // FFN
        Linear gate_proj, up_proj, down_proj;
        Linear gate_up_proj;
        Tensor* gate_up_weight = nullptr; // bf16 [hidden, 2 * ff_dim]
        Tensor* down_weight = nullptr;    // bf16 [ff_dim, hidden]
        Tensor* gate_up_weight_jm = nullptr; // bf16 [2 * ff_dim, hidden]
        Tensor* down_weight_jm = nullptr;    // bf16 [hidden, ff_dim]

        std::vector<float> host_linear_A_negexp;
        std::vector<float> host_linear_dt_bias;
        std::vector<float> host_linear_conv;      // [channels * kernel]
        std::vector<float> host_linear_norm;      // [head_dim]
    };

    struct Buffers {
        Tensor hidden;
        Tensor normed;
        Tensor qkv;
        Tensor linear_all;
        Tensor q;
        Tensor k;
        Tensor v;
        Tensor z;
        Tensor a;
        Tensor b;
        Tensor attn_out;
        Tensor full_qkv;
        Tensor gate_up;
        Tensor gate;
        Tensor up;
        Tensor logits;
        Tensor decode_scores;
        Tensor decode_attn_partials;
        Tensor scores;
        Tensor incr_scores;
    } buf_;

    friend struct aila::models::HybridLayerExecutor;
    friend struct aila::models::HybridDecodeBatchExecutor;
    using ExecutionPhase = aila::models::HybridExecutionPhase;
    using LayerKind = aila::models::HybridLayerKind;
    using LayerPlan = aila::models::HybridLayerPlan<Layer>;
    using DecoderPlan = aila::models::HybridDecoderPlan<Layer>;
    using LayerExecution = aila::models::HybridLayerExecution<
        Buffers, LayerCache, aila::models::Bf16ProjectionWorkspace>;
    struct ProjectionPolicy;
    struct ProfileState;

    void BuildDecoderPlan();
    void ExecuteLayer(const LayerPlan& plan, const LayerExecution& execution);
    Tensor& ForwardImpl(Context& ctx, Session& session, const int* tokenIdsDevice,
                        int count, bool lastTokenOnly);
    std::unique_ptr<Session> CreateSessionImpl(Context& ctx, bool quantizedKv,
                                              bool boundedSnapshots, std::size_t maxSnapshots);
    Session& DefaultSession(Context& ctx);
    void ValidateSession(const Session& session) const;
    void CaptureSnapshot(Context& ctx, Session& session, int nextLength, int tokenCount);

    void ensure_runtime_buffers(Context& ctx, int seq_len);
    void ensure_prefill_scores(Context& ctx, int seq_len);
    void ensure_incr_prefill_scores(Context& ctx, int seq_len, int total_len);
    void run_linear_delta_decode_gpu(Context& ctx, Layer& layer, LayerCache& cache,
                                     Tensor& qkv_src, Tensor& z_src,
                                     Tensor& a_src, Tensor& b_src);
    void run_linear_delta_decode_gpu(Context& ctx, Layer& layer, LayerCache& cache,
                                     Tensor& qkv_src, Tensor& z_src,
                                     Tensor& a_src, Tensor& b_src,
                                     Tensor& out_dst);
    void run_linear_delta_prefill_gpu_batched(Context& ctx, Layer& layer, LayerCache& cache,
                                              Tensor& fused_all, Tensor& out_dst, int seq_len);
    bool use_decode_jm_custom_path(int seq_len) const;
    bool use_decode_ffn_custom_path(int seq_len) const;
    void run_decode_ffn_gate_up_swiglu_custom(Context& ctx, const Layer& layer,
                                              Tensor& input, Tensor& output);
    void run_decode_ffn_down_custom(Context& ctx, const Layer& layer,
                                    Tensor& input, Tensor& output);
    void run_decode_jm_matvec_custom(Context& ctx, Tensor& input,
                                     const Tensor& weight_jm,
                                     int in_dim, int out_dim, Tensor& output);
    // Multi-row variants for independent decode sequences. Each row keeps the
    // original per-row vec8/FMA order and reduction; only the launch count and
    // weight re-reads are reduced by mapping rows onto one grid.
    void run_decode_jm_matvec_rows(Context& ctx, Tensor& input,
                                   const Tensor& weight_jm,
                                   int in_dim, int out_dim, Tensor& output, int rows);
    void run_decode_ffn_gate_up_swiglu_rows(Context& ctx, const Layer& layer,
                                            Tensor& input, Tensor& output, int rows);
    void debug_compare_linear_delta_decode(Context& ctx, int layer_idx,
                                           Layer& layer, LayerCache& cache,
                                           Tensor& qkv_src, Tensor& z_src,
                                           Tensor& a_src, Tensor& b_src);
    void run_linear_delta_host(Context& ctx, Layer& layer, LayerCache& cache,
                               Tensor& qkv_src, Tensor& z_src, Tensor& a_src, Tensor& b_src,
                               int seq_len);
    void run_linear_delta_host(Context& ctx, Layer& layer, LayerCache& cache,
                               Tensor& qkv_src, Tensor& z_src, Tensor& a_src, Tensor& b_src,
                               int seq_len, Tensor& out_dst);
    static float softplus(float x);
    static float silu(float x);
    static void head_l2_norm_inplace(std::vector<float>& x, int seq_len, int num_heads, int head_dim, float eps);
    static void head_rms_norm_and_silu_gate(std::vector<float>& x,
                                            const std::vector<float>& norm_weight,
                                            const std::vector<float>& z,
                                            int seq_len, int num_heads, int head_dim, float eps);

    ModelSpec spec_{};
    Qwen35TextConfig cfg_{};
    Tensor* embed_weight_ = nullptr;
    Tensor* final_norm_weight_ = nullptr;
    Linear lm_head_;
    Tensor batchLogits_, batchLogitsView_;

    std::vector<Layer> layers_;
    std::vector<Tensor> fused_weights_;
    Context* context_ = nullptr;
    std::shared_ptr<const SessionOwner> sessionOwner_;
    std::unique_ptr<Session> defaultSession_;
    std::unique_ptr<const DecoderPlan> decoderPlan_;
    std::unique_ptr<ProfileState> profile_;
    Tensor lastTokenLogits_;
    bool explicitSessionsCreated_ = false;
    bool legacyQuantizedKv_ = false;

    int max_seq_len_ = 0;
    int runtime_seq_capacity_ = 0;
    int prefill_scores_capacity_ = 0;
    int incr_prefill_seq_cap_ = 0;
    int incr_prefill_total_cap_ = 0;

    int full_q_heads_ = 0;
    int full_kv_heads_ = 0;
    int full_head_dim_ = 0;
    int full_q_dim_ = 0;
    int full_kv_dim_ = 0;
    int full_q_proj_dim_ = 0;
    int full_fused_qkv_dim_ = 0;

    int linear_q_heads_ = 0;
    int linear_kv_heads_ = 0;
    int linear_head_dim_ = 0;
    int linear_q_dim_ = 0;
    int linear_kv_dim_ = 0;
    int linear_qkv_dim_ = 0;
    int linear_all_dim_ = 0;
    int linear_z_dim_ = 0;
    int linear_conv_kernel_dim_ = 4;
    int linear_conv_channels_ = 0;

    int hidden_size_ = 0;
    int ff_dim_ = 0;
    int max_attn_heads_ = 0;
    int max_qkv_dim_ = 0;
    int max_attn_dim_ = 0;
    bool use_delta_linear_ = false;
    bool decode_ffn_custom_enabled_ = false;

    struct LinearDeltaScratch {
        std::vector<sycl::ext::oneapi::bfloat16> h_qkv;
        std::vector<sycl::ext::oneapi::bfloat16> h_z;
        std::vector<sycl::ext::oneapi::bfloat16> h_a;
        std::vector<sycl::ext::oneapi::bfloat16> h_b;
        std::vector<float> conv_in;
        std::vector<float> z;
        std::vector<float> a;
        std::vector<float> b;
        std::vector<float> conv_out;
        std::vector<float> q;
        std::vector<float> k;
        std::vector<float> v;
        std::vector<float> out;
        std::vector<float> sk;
        std::vector<float> d;
        std::vector<sycl::ext::oneapi::bfloat16> out_bf16;
    } linear_delta_scratch_;

};
