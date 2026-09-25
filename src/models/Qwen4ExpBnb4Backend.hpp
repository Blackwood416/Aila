#pragma once

#include "IModelBackend.hpp"
#include "../ops/Bnb4BitLinear.hpp"
#include "../ops/GatedDeltaNetOps.hpp"
#include "../ops/HyperConnectionOps.hpp"
#include "../ops/MoeExpertOps.hpp"
#include "../ops/Ops.hpp"
#include "../core/Tensor.hpp"
#include "Qwen4ExpExpertCache.hpp"
#include "Qwen4ExpPle.hpp"
#include "engine/Types.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

struct Qwen4ExpHyperConnectionLayer {
    Tensor* hc_norm_weight = nullptr; // f32, preprocessed for grouped_rms_norm (1+w)
    Linear input_mix_weight_down;     // (10240, 320)
    Linear input_mix_weight_up;       // (320, 10240)
    Linear block_inject_weight;       // (10240, 4)
    // Raw [out, in] bf16 weights for the native decode GEMV path.
    Tensor* mix_down_w = nullptr;
    Tensor* mix_up_w = nullptr;
    Tensor* inject_w = nullptr;
};

struct Qwen4ExpLayer {
    bool is_linear = false;

    // Hyper-connections
    Qwen4ExpHyperConnectionLayer attn_hc;
    Qwen4ExpHyperConnectionLayer mlp_hc;

    // Attention
    // Linear Attention (36 layers):
    Bnb4BitLinear linear_all_proj; // in_proj_qkv (10240), in_proj_z (6144), a (48), b (48) -> 16480
    Bnb4BitLinear linear_o_proj;   // 2560 rows, in_features = 6144
    Tensor* linear_norm_weight = nullptr; // [128] f32
    Tensor* linear_A_log = nullptr;       // [48] f32
    Tensor* linear_dt_bias = nullptr;     // [48] bf16
    Tensor* linear_conv1d_weight = nullptr; // [10240, 4] bf16
    ops::DeltaNetGeometry geom;

    // Full Attention (12 layers):
    Bnb4BitLinear qkv_proj;        // q (12288 = 2*24*256), k (512), v (512) -> 13312
    Bnb4BitLinear o_proj;          // 2560 rows, in_features = 6144
    Tensor* q_norm_weight = nullptr; // [256] f32 (1+w)
    Tensor* k_norm_weight = nullptr; // [256] f32 (1+w)

    // MoE
    Linear router_gate;            // (2560, 512) bf16
    Linear shared_expert_gate;     // (2560, 1) bf16
    Tensor* router_w = nullptr;          // raw [512, 2560]
    Tensor* shared_expert_gate_w = nullptr; // raw [1, 2560]
    Bnb4BitLinear shared_gate_up_proj; // (1280, 2560)
    Bnb4BitLinear shared_down_proj;    // (2560, 640)
};

struct Qwen4ExpLayerCache {
    // Full Attention
    Tensor k;
    Tensor v;

    // Linear Attention
    Tensor linear_state;
    Tensor linear_conv_state;
    std::vector<float> host_linear_state;
    std::vector<float> host_linear_conv_state;
    int linear_conv_head = 0;
    bool device_state_dirty = false;
};

struct Qwen4ExpBuffers {
    Tensor hidden4;       // [max_seq, 10240]
    Tensor hc_N;          // [max_seq, 10240]
    Tensor hc_mix_down;   // [max_seq, 320]
    Tensor hc_mix_up;     // [max_seq, 10240]
    Tensor hc_inject_buf; // [max_seq, 4]
    Tensor hc_inject_f32; // [max_seq, 4]
    Tensor x;             // [max_seq, 2560]
    Tensor y;             // [max_seq, 2560]

    // Full Attention
    Tensor full_qkv;      // [max_seq, 13312]
    Tensor q;             // [max_seq, 6144]
    Tensor z;             // [max_seq, 6144]
    Tensor k;             // [max_seq, 512]
    Tensor v;             // [max_seq, 512]
    Tensor attn_out;      // [max_seq, 6144]
    Tensor decode_scores; // [24, max_seq]
    Tensor decode_attn_partials; // [24, 64 * 256]
    Tensor scores;        // [24, max_seq, max_seq]
    Tensor incr_scores;   // [24, max_seq, max_seq]

    // Linear Attention
    Tensor linear_all;    // [max_seq, 16480]

    // MoE
    Tensor router_scores;      // [max_seq, 512]

    Tensor moe_h;              // [10, 640]
    Tensor shared_gate_buf;    // [max_seq, 1]
    Tensor shared_gate_up_buf; // [max_seq, 1280]
    Tensor shared_inter_buf;   // [max_seq, 640]
    Tensor shared_out_buf;     // [max_seq, 2560]

    // Output
    Tensor logits;
};

class Qwen4ExpBnb4Backend : public IModelBackend {
    struct SessionOwner {};

    struct LayerStateSnapshot {
        std::vector<float> linear_state_host;
        std::vector<float> linear_conv_state_host;
        int linear_conv_head = 0;
    };
    struct StateSnapshot {
        std::vector<LayerStateSnapshot> layers;
        Qwen4ExpPleSnapshot ple_snapshot;
    };

public:
    class Session {
    public:
        ~Session();
        Session(const Session&) = delete;
        Session& operator=(const Session&) = delete;
        Session(Session&&) = delete;
        Session& operator=(Session&&) = delete;

        int CurrentLength() const { return currentLength_; }
        int Capacity() const { return capacity_; }

    private:
        friend class Qwen4ExpBnb4Backend;
        Session(Context& ctx, int capacity,
                std::shared_ptr<const SessionOwner> owner,
                bool boundedSnapshots, std::size_t maxSnapshots);

        Context* context_ = nullptr;
        std::shared_ptr<const SessionOwner> owner_;
        int currentLength_ = 0;
        int capacity_ = 0;
        bool boundedSnapshots_ = true;
        std::size_t maxSnapshots_ = 4;

        std::vector<Qwen4ExpLayerCache> layerCaches_;
        Qwen4ExpPleState ple_state_;
        std::unordered_map<int, StateSnapshot> snapshots_;

        Tensor mropePosT_, mropePosH_, mropePosW_;
        int mropePromptLength_ = 0;
        int mropeTextPositionDelta_ = 0;
        std::vector<int> embeddingOverridePositions_;
        std::vector<sycl::ext::oneapi::bfloat16> embeddingOverrideValues_;
        int embeddingOverrideHiddenSize_ = 0;
    };

    Qwen4ExpBnb4Backend();
    ~Qwen4ExpBnb4Backend() override;

    bool load(Context& ctx, ModelWeights& weights, const ModelSpec& spec,
              int max_seq_len, std::string* error_message) override;
    Tensor& forward(Context& ctx, const int* token_ids_device, int seq_len) override;
    void reset() override;
    bool truncate_kv_cache(int new_len) override;
    int get_current_context_len() const override {
        return defaultSession_ ? defaultSession_->CurrentLength() : 0;
    }
    int max_seq_len() const override { return max_seq_len_; }
    int vocab_size() const override { return cfg_.vocab_size; }
    ModelFamily family() const override { return ModelFamily::Qwen4Exp; }

    std::unique_ptr<Session> CreateSession(Context& ctx, std::size_t maxSnapshots = 4);
    Tensor& Forward(Context& ctx, Session& session,
                    const int* tokenIdsDevice, int count);
    Tensor& ForwardLegacyLogits(Context& ctx, Session& session,
                                const int* tokenIdsDevice, int count);
    void Reset(Session& session);
    int Truncate(Session& session, int newLength);

    bool supports_vision_embedding_override() const override { return true; }
    void set_embedding_overrides(const std::vector<int>& positions,
                                 const std::vector<sycl::ext::oneapi::bfloat16>& embeddings,
                                 int hidden_size) override;
    void clear_embedding_overrides() override;
    void set_mrope_positions(Context& ctx, const std::vector<int>& pos_t,
                             const std::vector<int>& pos_h,
                             const std::vector<int>& pos_w,
                             int text_pos_delta) override;
    void clear_mrope_positions() override;

private:
    Tensor& ForwardImpl(Context& ctx, Session& session,
                        const int* tokenIdsDevice, int count, bool lastTokenOnly);
    Session& DefaultSession(Context& ctx);
    void ValidateSession(const Session& session) const;
    void CaptureSnapshot(Context& ctx, Session& session, int nextLength, int tokenCount);
    void ensure_runtime_buffers(Context& ctx, int seq_len);
    void free_pinned_host_buffers();
    // Writes count embedding rows into buf_.x, from VRAM or gathered from the host mmap.
    void EmbedTokens(Context& ctx, const int* tokenIdsDevice, int count);
    // bf16 Linear with raw [out, in] weight: native GEMV for one row, oneDNN otherwise.
    void DenseLinear(Context& ctx, Linear& linear, Tensor* raw_weight,
                     Tensor& input, Tensor& output, int count, int out_features, int in_features);
    // grouped norm -> low-rank mix -> x = mean_s(mix * N); optionally inject = 2*sigmoid(W N / 4).
    void RunHyperConnection(Context& ctx, Qwen4ExpHyperConnectionLayer& hc, int count, bool with_inject);

    ModelSpec spec_;
    Qwen4ExpTextConfig cfg_;
    int hidden_size_ = 2560;
    int max_seq_len_ = 2048;
    int full_q_heads_ = 24, full_kv_heads_ = 2, full_head_dim_ = 256;
    int full_q_dim_ = 6144, full_kv_dim_ = 512, full_q_proj_dim_ = 12288, full_fused_qkv_dim_ = 13312;
    int linear_q_heads_ = 16, linear_kv_heads_ = 48, linear_head_dim_ = 128;
    int linear_q_dim_ = 2048, linear_kv_dim_ = 6144, linear_qkv_dim_ = 10240;
    int linear_z_dim_ = 6144, linear_all_dim_ = 16480;

    Context* context_ = nullptr;
    std::shared_ptr<const SessionOwner> sessionOwner_;
    std::unique_ptr<Session> defaultSession_;

    std::vector<Qwen4ExpLayer> layers_;
    Qwen4ExpHyperConnectionLayer mixer_;
    Qwen4ExpPle ple_;
    Qwen4ExpExpertCache expert_cache_;

    Tensor* embed_weight_ = nullptr; // resident copy, only when AILA_Q4E_HOST_EMBED=0
    // Otherwise rows are gathered from the mmap per forward: decode needs one 5 KB row, and the
    // 1.3 GB table left in VRAM would displace ~460 expert slots.
    HostTensorView embed_host_{};
    int* embed_host_ids_ = nullptr;      // pinned [runtime_seq_capacity_]
    uint8_t* embed_host_rows_ = nullptr; // pinned [runtime_seq_capacity_, hidden] bf16
    Tensor lm_head_weight_transposed_;
    Linear lm_head_;
    Tensor lastTokenLogits_;

    Qwen4ExpBuffers buf_;
    // Pinned host [runtime_seq_capacity_, top_k] router outputs written directly by the top-k kernel.
    int* router_host_ids_ = nullptr;
    float* router_host_weights_ = nullptr;
    // S1 Batched prefill MoE slot descriptors and weights (host pinned buffers and device mirrors).
    ops::MoeExpertSlotDev* batch_slots_host_ = nullptr;
    ops::MoeExpertSlotDev* batch_slots_dev_ = nullptr;
    float* batch_weights_dev_ = nullptr;

    // Prefill batched expert acquire scratch (expert -> slot / token uses for the current layer).
    std::vector<int> prefill_expert_slot_;
    std::vector<int> prefill_expert_uses_;
    std::vector<int> prefill_unique_;
    std::vector<int> prefill_unique_slots_;

    // Prefill speculative next-layer prefetch scratch.
    std::vector<int> prefill_pred_unique_;
    std::vector<int> prefill_pred_uses_;
    std::vector<int> sort_scratch_;

    // AILA_Q4E_PREDICT_STATS diagnostic: next-layer expert prediction quality.
    Tensor pred_scores_;
    int* pred_host_ids_ = nullptr;
    float* pred_host_weights_ = nullptr;
    uint64_t pred_total_ = 0, pred_overlap_ = 0, pred_miss_total_ = 0, pred_miss_hit_ = 0;
    Bnb4BitLinearScratch linear_scratch_;
    int runtime_seq_capacity_ = 0;
};
