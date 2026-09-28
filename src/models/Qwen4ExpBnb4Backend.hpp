#pragma once

#include "IModelBackend.hpp"
#include "../ops/Bnb4BitLinear.hpp"
#include "../ops/GatedDeltaNetOps.hpp"
#include "../ops/HyperConnectionOps.hpp"
#include "../ops/MoeExpertOps.hpp"
#include "../ops/Ops.hpp"
#include "../core/Tensor.hpp"
#include "Qwen4ExpExpertCache.hpp"
#include "Qwen4ExpExpertExecutor.hpp"
#include "Qwen4ExpPle.hpp"
#include "engine/Types.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

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

    // QSA Indexer (M2 & W2)
    Linear index_qk_proj;          // (2560, 640) bf16
    Linear index_k_proj;           // (2560, 128) bf16 (A2: K-only projection for <= 2051)
    Tensor* index_qk_weight = nullptr; // raw [640, 2560]
    Tensor index_k_weight;         // view [128, 2560] pointing to K rows of index_qk_weight
    Tensor* index_q_norm_weight = nullptr; // [128] f32 (1+w)
    Tensor* index_k_norm_weight = nullptr; // [128] f32 (1+w)

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
    Tensor indexer_k; // [capacity, 128] bf16
    Tensor indexer_compressed_k; // [capacity / 4, 128] bf16 (A1)
    int num_compressed_blocks = 0; // Number of complete blocks compressed (A1)

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

    // QSA Indexer
    Tensor indexer_qk;           // [max_seq, 640]
    Tensor indexer_q;            // [max_seq, 512]
    Tensor indexer_block_scores; // [max_blocks] f32
    Tensor indexer_active_tokens;// [2051] s32
    Tensor indexer_active_blocks;// [516] s32 (Phase A3)
    Tensor qsa_tile_block_scores;// [kQsaQueryTile, max_blocks] f32 (Phase A4)
    Tensor qsa_tile_active_blocks;// [kQsaQueryTile, 516] s32 (Phase A4)
    Tensor qsa_tile_valid_counts;// [qsa_query_tile_, 2] s32 (Phase P2)


    // Linear Attention
    Tensor linear_all;    // [max_seq, 16480]

    // MoE
    Tensor router_scores;      // [max_seq, 512]

    Tensor moe_h;              // [10, 640]
    Tensor shared_gate_buf;    // [max_seq, 1]
    Tensor shared_gate_up_buf; // [max_seq, 1280]
    Tensor shared_inter_buf;   // [max_seq, 640]
    Tensor shared_out_buf;     // [max_seq, 2560]
    Tensor moe_down_acc;       // [max_seq, top_k, 2560] f32 (Phase S4 expert-major down accumulation)

    // Output
    Tensor logits;
};

class Qwen4ExpBnb4Backend : public IModelBackend {
    struct SessionOwner {};

    struct LayerStateSnapshot {
        std::vector<float> linear_state_host;
        std::vector<float> linear_conv_state_host;
        int linear_conv_head = 0;
        int num_compressed_blocks = 0;
    };
    struct StateSnapshot {
        int length = -1;
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
        uint64_t generation() const { return generation_; }
        void bump_generation() { ++generation_; }

        const std::vector<StateSnapshot>& snapshot_slots_for_testing() const { return snapshots_; }
        const StateSnapshot* snapshot_for_testing(int length) const {
            for (const auto& snapshot : snapshots_) {
                if (snapshot.length == length) return &snapshot;
            }
            return nullptr;
        }
        std::vector<Qwen4ExpLayerCache>& layer_caches_for_testing() { return layerCaches_; }
        int& current_length_for_testing() { return currentLength_; }

    private:
        friend class Qwen4ExpBnb4Backend;
        Session(Context& ctx, int capacity,
                std::shared_ptr<const SessionOwner> owner,
                std::size_t maxSnapshots);

        Context* context_ = nullptr;
        std::shared_ptr<const SessionOwner> owner_;
        int currentLength_ = 0;
        int capacity_ = 0;
        uint64_t generation_ = 1;

        std::vector<Qwen4ExpLayerCache> layerCaches_;
        Qwen4ExpPleState ple_state_;
        std::vector<StateSnapshot> snapshots_;

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
    bool supports_interior_checkpoint(int count, int checkpoint_after) const override;
    Tensor& forward(Context& ctx, const int* token_ids_device, int seq_len,
                    const ForwardOptions& options) override;
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
                    const int* tokenIdsDevice, int count,
                    const ForwardOptions& options = {});
    Tensor& ForwardLegacyLogits(Context& ctx, Session& session,
                                const int* tokenIdsDevice, int count);
    void Reset(Session& session);
    int Truncate(Session& session, int newLength);

    void init_for_testing(Context& ctx, int num_layers = 4, int max_seq_len = 64) {
        context_ = &ctx;
        max_seq_len_ = max_seq_len;
        layers_.resize(num_layers);
        for (int i = 0; i < num_layers; ++i) {
            layers_[i].is_linear = ((i + 1) % 4 != 0);
        }
    }
    void capture_snapshot_for_testing(Context& ctx, Session& session, int nextLength, int count = 1,
                                      bool force = false) {
        CaptureSnapshot(ctx, session, nextLength, count, force);
        session.currentLength_ = nextLength;
    }
    void ensure_runtime_buffers_for_testing(Context& ctx, int seq_len) {
        ensure_runtime_buffers(ctx, seq_len);
    }
    const Qwen4ExpBuffers& buf_for_testing() const { return buf_; }
    int qsa_query_tile() const { return qsa_query_tile_; }

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

    const aila::HeteroExecutionStats& hetero_stats() const { return expert_executor_.stats(); }
    aila::Qwen4ExpExpertExecutor& expert_executor() { return expert_executor_; }

private:
    Tensor& ForwardImpl(Context& ctx, Session& session,
                        const int* tokenIdsDevice, int count, bool lastTokenOnly,
                        const ForwardOptions& options);
    Session& DefaultSession(Context& ctx);
    void ValidateSession(const Session& session) const;
    void CaptureSnapshot(Context& ctx, Session& session, int nextLength, int tokenCount, bool force = false);
    StateSnapshot& ReserveSnapshot(Session& session, int nextLength);
    friend struct Qwen4ExpGenerationTestAccess;
    void ensure_runtime_buffers(Context& ctx, int seq_len);
    void free_pinned_host_buffers();
    // Writes count embedding rows into buf_.x, from VRAM or gathered from the host mmap.
    void EmbedTokens(Context& ctx, const int* tokenIdsDevice, int count);
    // bf16 Linear with raw [out, in] weight: native GEMV for one row, oneDNN otherwise.
    void DenseLinear(Context& ctx, Linear& linear, Tensor* raw_weight,
                     Tensor& input, Tensor& output, int count, int out_features, int in_features);
    // grouped norm -> low-rank mix -> x = mean_s(mix * N); optionally inject = 2*sigmoid(W N / 4).
    void RunHyperConnection(Context& ctx, Qwen4ExpHyperConnectionLayer& hc, int count, bool with_inject);
    // P1: Fused residual add + grouped norm across attention -> MLP boundary
    void RunHyperConnectionCombined(Context& ctx, Qwen4ExpHyperConnectionLayer& hc, int count, bool with_inject,
                                    const sycl::ext::oneapi::bfloat16* pending_y, const float* pending_inj);

    ModelSpec spec_;
    Qwen4ExpTextConfig cfg_;
    int hidden_size_ = 2560;
    int max_seq_len_ = 2048;
    int full_q_heads_ = 24, full_kv_heads_ = 2, full_head_dim_ = 256;
    int full_q_dim_ = 6144, full_kv_dim_ = 512, full_q_proj_dim_ = 12288, full_fused_qkv_dim_ = 13312;
    int linear_q_heads_ = 16, linear_kv_heads_ = 48, linear_head_dim_ = 128;
    int linear_q_dim_ = 2048, linear_kv_dim_ = 6144, linear_qkv_dim_ = 10240;
    int linear_z_dim_ = 6144, linear_all_dim_ = 16480;
    int qsa_query_tile_ = 64;

    Context* context_ = nullptr;
    std::shared_ptr<const SessionOwner> sessionOwner_;
    std::unique_ptr<Session> defaultSession_;

    std::vector<Qwen4ExpLayer> layers_;
    Qwen4ExpHyperConnectionLayer mixer_;
    Qwen4ExpPle ple_;
    PleTicket ple_ticket_;
    Qwen4ExpExpertCache expert_cache_;
    aila::Qwen4ExpExpertExecutor expert_executor_;

    Tensor* embed_weight_ = nullptr; // resident copy, only when AILA_Q4E_HOST_EMBED=0
    // Otherwise rows are gathered from the mmap per forward: decode needs one 5 KB row, and the
    // 1.3 GB table left in VRAM would displace ~460 expert slots.
    HostTensorView embed_host_{};
    int* embed_host_ids_ = nullptr;      // pinned [runtime_seq_capacity_]
    uint8_t* embed_host_rows_ = nullptr; // pinned [runtime_seq_capacity_, hidden] bf16
    Tensor lm_head_weight_transposed_;
    Linear lm_head_;
    Tensor lastTokenLogits_;
    Tensor unusedLogits_; // Invalid output for forwards that only advance cached state.

    Qwen4ExpBuffers buf_;
    // Pinned host [runtime_seq_capacity_, top_k] router outputs written directly by the top-k kernel.
    int* router_host_ids_ = nullptr;
    float* router_host_weights_ = nullptr;
    // S1 Batched prefill MoE slot descriptors and weights (host pinned buffers and device mirrors).
    ops::MoeExpertSlotDev* batch_slots_host_ = nullptr;
    ops::MoeExpertSlotDev* batch_slots_dev_ = nullptr;
    float* batch_weights_dev_ = nullptr;

    // Phase S4 & W4 (C1/C2): Expert-major prefill bucket sort and mapping tables (host pinned and device mirrors).
    int* expert_offsets_host_ = nullptr;
    int* expert_counts_host_ = nullptr;
    int* expert_curr_pos_host_ = nullptr;
    int* expert_pair_pos_host_ = nullptr;
    int* expert_single_pos_host_ = nullptr;
    int* expert_map_token_host_ = nullptr;
    int* expert_map_rank_host_ = nullptr;
    ops::MoeExpertSlotDev* expert_map_slot_host_ = nullptr;
    int* expert_map_token_dev_ = nullptr;
    int* expert_map_rank_dev_ = nullptr;
    ops::MoeExpertSlotDev* expert_map_slot_dev_ = nullptr;

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
