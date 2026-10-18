#pragma once

#include "IModelBackend.hpp"
#include "../ops/Bnb4BitLinear.hpp"
#include "../ops/GatedDeltaNetOps.hpp"
#include "../ops/GdnChunkOps.hpp"
#include "../ops/HyperConnectionOps.hpp"
#include "../ops/MoeExpertOps.hpp"
#include "../ops/MoeExpertXmx.hpp"
#include "../ops/Ops.hpp"
#include "../core/Tensor.hpp"
#include "Qwen4ExpExpertCache.hpp"
#include "Qwen4ExpExpertExecutor.hpp"
#include "Qwen4ExpHostArena.hpp"
#include "Qwen4ExpAdaptiveCache.hpp"
#include "Qwen4ExpLayerStage.hpp"
#include "Qwen4ExpPle.hpp"
#include "Qwen4ExpHotsetPolicy.hpp"
#include "engine/Types.hpp"

#include <cstddef>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
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
    // Opt-in INT8 KV: per-(kv_head, token) f32 dequantization scale for k/v.
    // Only allocated when AILA_KV_QUANT=1 AILA_KV_QUANT_FORMAT=int8.
    Tensor k_scale; // [2, capacity]
    Tensor v_scale; // [2, capacity]
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
    // Owns the opt-in phase scratch. Declared before views so it dies last.
    Tensor phase_scratch;
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
    Tensor attn_xq;       // decode-only [max attention projection input] signed bytes
    Tensor attn_xs;       // decode-only [max attention projection input / 64] f32

    // MoE
    Tensor router_scores;      // [max_seq, 512]

    Tensor moe_h;              // [10, 640]
    // Decode-only NF4/Q8 activation scratch; allocated before the first forward.
    Tensor moe_x_q8;           // [hidden] signed bytes in u8 storage
    Tensor moe_x_q8_scale;     // [hidden / 64] f32
    Tensor moe_h_q8;           // [top_k, intermediate] signed bytes in u8 storage
    Tensor moe_h_q8_scale;     // [top_k, intermediate / 64] f32
    Tensor shared_gate_buf;    // [max_seq, 1]
    Tensor shared_gate_up_buf; // [max_seq, 1280]
    Tensor shared_inter_buf;   // [max_seq, 640]
    Tensor shared_out_buf;     // [max_seq, 2560]
    Tensor moe_window_sum; // compact path [window,H] int64 Q36, independent of expert submission order
    Tensor moe_compact_invalid; // fail closed on nonfinite/out-of-range contributions
    Tensor moe_inverse_map; // [window,top_k] -> global activation index
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
        // Exact device payload of the tensors this Session owns (Full K/V,
        // QSA indexer keys, DeltaNet state/conv history, PLE conv state).
        // Excludes model weights, expert cache and shared execution scratch.
        std::size_t CacheBytes() const { return cacheBytes_; }

        const std::vector<StateSnapshot>& snapshot_slots_for_testing() const { return snapshots_; }
        const StateSnapshot* snapshot_for_testing(int length) const {
            for (const auto& snapshot : snapshots_) {
                if (snapshot.length == length) return &snapshot;
            }
            return nullptr;
        }
        std::vector<Qwen4ExpLayerCache>& layer_caches_for_testing() { return layerCaches_; }
        int& current_length_for_testing() { return currentLength_; }
        int& consecutive_no_miss_layers_for_testing() { return consecutive_no_miss_layers_; }
        Qwen4ExpPleState& ple_state_for_testing() { return ple_state_; }
        int consecutive_no_miss_layers() const { return consecutive_no_miss_layers_; }

    private:
        friend class Qwen4ExpBnb4Backend;
        friend struct Qwen4ExpMtpProbe;
        friend class Qwen4ExpMtpRuntime;
        Session(Context& ctx, int capacity,
                std::shared_ptr<const SessionOwner> owner,
                std::size_t maxSnapshots);

        Context* context_ = nullptr;
        std::shared_ptr<const SessionOwner> owner_;
        int currentLength_ = 0;
        int capacity_ = 0;
        std::size_t cacheBytes_ = 0;
        uint64_t generation_ = 1;
        int consecutive_no_miss_layers_ = 0;

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
    // Qwen4Exp has no cross-session decode batch path; the Model session
    // backend also excludes ForwardDecodeBatch at compile time.
    bool SupportsDecodeBatch() const noexcept { return false; }
    Tensor& Forward(Context& ctx, Session& session,
                    const int* tokenIdsDevice, int count,
                    const ForwardOptions& options = {});
    Tensor& ForwardLegacyLogits(Context& ctx, Session& session,
                                const int* tokenIdsDevice, int count);
    void Reset(Session& session);
    int Truncate(Session& session, int newLength);
    void PrepareDecodeVerification(Context& ctx);

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

    void set_prefill_xmx(bool enabled) {
        for (auto& layer : layers_) {
            if (layer.is_linear) {
                layer.linear_o_proj.set_allow_dense_xmx(enabled);
                layer.linear_all_proj.set_allow_dense_xmx(enabled);
            } else {
                layer.o_proj.set_allow_dense_xmx(enabled);
                layer.qkv_proj.set_allow_dense_xmx(enabled);
            }
            layer.shared_gate_up_proj.set_allow_dense_xmx(enabled);
            layer.shared_down_proj.set_allow_dense_xmx(enabled);
        }
        ple_.key_proj().set_allow_dense_xmx(enabled);
        ple_.value_proj().set_allow_dense_xmx(enabled);
    }

    void set_direct_prefill_io(bool enabled) {
        for (auto& layer : layers_) {
            if (layer.is_linear) {
                layer.linear_o_proj.set_direct_prefill_io(enabled);
                layer.linear_all_proj.set_direct_prefill_io(enabled);
            } else {
                layer.o_proj.set_direct_prefill_io(enabled);
                layer.qkv_proj.set_direct_prefill_io(enabled);
            }
            layer.shared_gate_up_proj.set_direct_prefill_io(enabled);
            layer.shared_down_proj.set_direct_prefill_io(enabled);
        }
        ple_.key_proj().set_direct_prefill_io(enabled);
        ple_.value_proj().set_direct_prefill_io(enabled);
    }

    const aila::HeteroExecutionStats& hetero_stats() const { return expert_executor_.stats(); }
    bool LmHeadQ8Enabled() const { return lmHeadQ8Enabled_; }
    uint64_t LmHeadQ8ForwardCount() const { return lmHeadQ8ForwardCount_; }
    uint64_t LmHeadQ8RowCount() const { return lmHeadQ8RowCount_; }
    aila::Qwen4ExpExpertExecutor& expert_executor() { return expert_executor_; }
    void SetColdEpochEnabled(bool enabled) { cold_epoch_enabled_=enabled; }
    // Cold acceptance reference only. Drain GPU/CPU work and stop forwards
    // before changing this hook; it is false throughout normal runtime use.
    void SetDecodeHcReferenceForTest(bool enabled) { decodeHcReferenceForTest_ = enabled; }
    // Forces original resident and attention GPU projection arithmetic. Drain
    // work before toggling this hook; normal execution leaves it false.
    void SetGpuNf4Q8ReferenceForTest(bool enabled) { gpuNf4Q8ReferenceForTest_ = enabled; }
    uint64_t GdnHistoryFusionCount() const { return gdnHistoryFusionCount_; }
    // Actual submitted projections, ordered linear_all, linear_o, full_qkv, full_o.
    const std::array<uint64_t, 4>& AttentionNf4Q8ProjectionCounts() const { return attentionNf4Q8ProjectionCounts_; }
    uint64_t AttentionNf4Q8ProjectionCount() const {
        return attentionNf4Q8ProjectionCounts_[0] + attentionNf4Q8ProjectionCounts_[1] +
            attentionNf4Q8ProjectionCounts_[2] + attentionNf4Q8ProjectionCounts_[3];
    }

    const aila::Qwen4ExpHostArena& host_arena() const noexcept { return host_arena_; }
    aila::Qwen4ExpHostArena& host_arena() noexcept { return host_arena_; }
    const aila::Qwen4ExpAdaptiveCache& adaptive_cache() const noexcept { return adaptive_cache_; }
    aila::Qwen4ExpAdaptiveCache& adaptive_cache() noexcept { return adaptive_cache_; }
    bool use_adaptive_pipeline() const noexcept { return use_adaptive_pipeline_; }

private:
    bool decodeHcReferenceForTest_ = false;
    bool gpuNf4Q8ReferenceForTest_ = false;
    bool gpuNf4Q8Supported_ = false;
    bool gpuNf4Q8Logged_ = false;
    bool attentionNf4Q8Enabled_ = false;
    uint64_t gdnHistoryFusionCount_ = 0;
    bool attentionNf4Q8Logged_ = false;
    std::array<uint64_t, 4> attentionNf4Q8ProjectionCounts_{};
    bool moeFrontFusedGpu_ = false;
    bool moeFrontFusedLogged_ = false;
    // Owns the resident CPU job until Join, including exceptions between its
    // early submission and the later GPU/cold merge. Declare after route leases.
    struct ResidentColdWork {
        aila::Qwen4ExpExpertExecutor& executor;
        bool active = false;
        bool completed = false;
        explicit ResidentColdWork(aila::Qwen4ExpExpertExecutor& value) : executor(value) {}
        ResidentColdWork(const ResidentColdWork&) = delete;
        ResidentColdWork& operator=(const ResidentColdWork&) = delete;
        ~ResidentColdWork() noexcept;
    };
    // One forward owns this ticket. y/inject remain in their existing buffers
    // until the next HC norm consumes them on the in-order compute queue.
    struct DecodePendingMlpResidual {
        Context& ctx;
        sycl::ext::oneapi::bfloat16* hidden;
        const sycl::ext::oneapi::bfloat16* y;
        const float* inject;
        int hiddenSize;
        bool pending = false;
        uint64_t deferred = 0;
        uint64_t consumedNorm = 0;
        uint64_t materialized = 0;
        DecodePendingMlpResidual(Context& context, sycl::ext::oneapi::bfloat16* hiddenPtr,
            const sycl::ext::oneapi::bfloat16* yPtr, const float* injectPtr, int dim)
            : ctx(context), hidden(hiddenPtr), y(yPtr), inject(injectPtr), hiddenSize(dim) {}
        DecodePendingMlpResidual(const DecodePendingMlpResidual&) = delete;
        DecodePendingMlpResidual& operator=(const DecodePendingMlpResidual&) = delete;
        ~DecodePendingMlpResidual() noexcept;
        void Defer();
        void Materialize();
        void ConsumeNorm(const float* normWeight, float eps,
            sycl::ext::oneapi::bfloat16* normalized);
        // F1 fused HC path (AILA_Q4E_DECODE_FUSED_HC=1): the pending residual
        // was consumed by the fused (inject-add + norm) + down submission; its
        // in-place update runs in the following up kernel, so no norm may be
        // re-submitted on unwind. Mirrors ConsumeNorm's clear-after-submit.
        void MarkConsumed();
    };
    Tensor& ForwardImpl(Context& ctx, Session& session,
                        const int* tokenIdsDevice, int count, bool lastTokenOnly,
                        const ForwardOptions& options);
    // Layer-major prefill prototype (AILA_Q4E_LAYER_MAJOR=1). Only the exact
    // two-full-chunk prefill shape is accepted; everything else falls back to
    // the chunk-major ForwardImpl path unchanged.
    bool LayerMajorPrefillEligible(const Session& session, int count,
                                   const ForwardOptions& options) const;
    Tensor& ForwardLayerMajorPrefill(Context& ctx, Session& session,
                                     const int* tokenIdsDevice, int count,
                                     bool lastTokenOnly, const ForwardOptions& options);
    sycl::event RunCompactExpertDown(Context& ctx, int first, int acts,
        int pairs, int quads, const sycl::event& ready);
    void RunAttentionLayerTile(Context& ctx, Session& session, int layer,
        Tensor& input, Tensor& output, int count, int startPos,
        StateSnapshot* interior, int checkpointAfter, bool allowQ8 = false);
    void RunAttentionWindow(Context& ctx, Session& session, int layer, int count,
        int startPos, StateSnapshot* interior, int checkpointAfter, bool allowQ8 = false);
    // Same tiling as RunAttentionWindow, but input/output rows start at
    // row_offset in buf_.x / buf_.y. Layer-major runs chunk 0 and chunk 1 as
    // two independent 4096-row attention windows inside one layer pass.
    void RunAttentionWindowAt(Context& ctx, Session& session, int layer, int count,
        int startPos, int row_offset, bool allowQ8 = false);
    Session& DefaultSession(Context& ctx);
    void ValidateSession(const Session& session) const;
    void CaptureSnapshot(Context& ctx, Session& session, int nextLength, int tokenCount, bool force = false);
    StateSnapshot& ReserveSnapshot(Session& session, int nextLength);
    bool CanCaptureInteriorCheckpoint(const Session& session, int count, int checkpointAfter) const;
    friend struct Qwen4ExpGenerationTestAccess;
    friend struct Qwen4ExpMtpProbe;
    friend class Qwen4ExpMtpDraft;
    friend class Qwen4ExpMtpRuntime;
    void ensure_runtime_buffers(Context& ctx, int seq_len);
    void TrimDecodeRuntimeBuffers(Context& ctx);
    void free_pinned_host_buffers();
    sycl::event RunGroupedExperts(Context& ctx, int layer, int count,
        int firstUnique, int numUnique, int& actStart, int& tileStart,
        int minRows, const sycl::event& ready,
        int route_base = 0, int token_base = 0, const int* use_counts = nullptr,
        const sycl::ext::oneapi::bfloat16* x_base = nullptr,
        sycl::ext::oneapi::bfloat16* moe_h_base = nullptr,
        float* down_acc_base = nullptr);
    // R1 expert-XMX consumer (AILA_Q4E_PREFILL_EXPERT_XMX=1): usage-sorted
    // chunks of <=16 experts, dequantize-once bf16 weights, oneDNN batched
    // GEMMs. Replaces the ALU bucketing consumer for this segment only.
    sycl::event RunGroupedExpertsXmx(Context& ctx, int layer, int count,
        int firstUnique, int numUnique, int& actStart, const sycl::event& ready);
    // Layer-major variant of RunAdaptivePrefill: same union/resident ordering
    // and staging ping-pong, but every expert launch is split into two
    // 4096-row halves that share one staged bank, and resident hits are
    // processed in bounded microbatches so a single large launch cannot stall
    // the staging pipeline.
    sycl::event RunAdaptivePrefillLayerMajor(Context& ctx, int layer,
        bool fuse_shared, int count,
        const sycl::ext::oneapi::bfloat16* x_base = nullptr,
        sycl::ext::oneapi::bfloat16* y_base = nullptr,
        sycl::ext::oneapi::bfloat16* moe_h_base = nullptr,
        float* down_acc_base = nullptr,
        const sycl::ext::oneapi::bfloat16* shared_out_base = nullptr,
        const sycl::ext::oneapi::bfloat16* shared_gate_base = nullptr);
    // Writes count embedding rows into buf_.x, from VRAM or gathered from the host mmap.
    void EmbedTokens(Context& ctx, const int* tokenIdsDevice, int count);
    // bf16 Linear with raw [out, in] weight: native GEMV for one row, oneDNN otherwise.
    void DenseLinear(Context& ctx, Linear& linear, Tensor* raw_weight,
                     Tensor& input, Tensor& output, int count, int out_features, int in_features);
    // grouped norm -> low-rank mix -> x = mean_s(mix * N); optionally inject = 2*sigmoid(W N / 4).
    // Returns the x-writing (up-mix) submission event when this path produces
    // one so the caller can order the activation D2H behind it without a
    // compute-queue barrier. Only the count==1 fused decode branch reports it;
    // all other shapes return nullopt (callers keep their old behavior).
    std::optional<sycl::event> RunHyperConnection(Context& ctx, Qwen4ExpHyperConnectionLayer& hc,
                                                  int count, bool with_inject);
    // P1: Fused residual add + grouped norm across attention -> MLP boundary
    std::optional<sycl::event> RunHyperConnectionCombined(
        Context& ctx, Qwen4ExpHyperConnectionLayer& hc, int count, bool with_inject,
        const sycl::ext::oneapi::bfloat16* pending_y, const float* pending_inj,
        DecodePendingMlpResidual* pending_mlp = nullptr);

    ModelSpec spec_;
    Qwen4ExpTextConfig cfg_;
    // P3: opt-in routing collector for offline hotset profile export
    // (AILA_Q4E_HOTSET_EXPORT). Populated on the adaptive pipeline and written
    // as a full 24576-key profile at teardown.
    Qwen4ExpHotsetPolicy export_policy_;
    bool export_enabled_ = false;
    std::string export_path_;
    std::string export_identity_;
    int hidden_size_ = 2560;
    int max_seq_len_ = 2048;
    int full_q_heads_ = 24, full_kv_heads_ = 2, full_head_dim_ = 256;
    int full_q_dim_ = 6144, full_kv_dim_ = 512, full_q_proj_dim_ = 12288, full_fused_qkv_dim_ = 13312;
    int linear_q_heads_ = 16, linear_kv_heads_ = 48, linear_head_dim_ = 128;
    int linear_q_dim_ = 2048, linear_kv_dim_ = 6144, linear_qkv_dim_ = 10240;
    int linear_z_dim_ = 6144, linear_all_dim_ = 16480;
    // Borrowed MTP window scratch: [layers,4,linear_all_dim]. The runtime or
    // diagnostic caller owns it through verification and prefix replay.
    bf16* verification_linear_inputs_for_test_ = nullptr;
    bf16* verification_router_inputs_for_test_ = nullptr;
    // Opt-in alignment probe (B4-2). When set together with
    // verification_stage_rows_for_test_ > 0, decode_verification forwards with
    // count <= rows copy the per-layer attention output (post o_proj, pre
    // MLP-HC) and per-layer router input (post MLP-HC, pre MoE) for every row
    // into caller-owned [layers, rows, hidden] bf16 buffers. These copies ride
    // the in-order queue and never change arithmetic.
    bf16* verification_attn_out_for_test_ = nullptr;
    bf16* verification_router_x_for_test_ = nullptr;
    bf16* verification_hidden_in_for_test_ = nullptr;
    bf16* verification_attn_in_for_test_ = nullptr;   // HC output = attention module input, [layers, rows, hidden]
    bf16* verification_attn_raw_for_test_ = nullptr;  // attention module raw output (pre o_proj), [layers, rows, 6144]
    int verification_stage_rows_for_test_ = 0;
    // Opt-in alignment (AILA_Q4E_VERIFY_SMALL_ROWS): while a 2..4 row
    // decode_verification forward runs, route the attention/shared projections,
    // the hyper-connections, PLE and the LM head through the single-row decode
    // kernels so the window matches sequential B1 decode. Default false keeps
    // the legacy multi-row GEMM schedule byte-for-byte unchanged.
    bool verify_rows_alignment_ = false;
    std::optional<sycl::kernel_bundle<sycl::bundle_state::executable>> verification_attention_pair_bundle_;
    bool component_profile_active_for_test_ = false;
    std::array<double, 5> component_profile_ms_for_test_{};
    int qsa_query_tile_ = 64;

    Context* context_ = nullptr;
    std::shared_ptr<const SessionOwner> sessionOwner_;
    std::unique_ptr<Session> defaultSession_;

    std::vector<Qwen4ExpLayer> layers_;
    Qwen4ExpHyperConnectionLayer mixer_;
    Qwen4ExpPle ple_;
    PleTicket ple_ticket_;
    PleTicket ple_next_ticket_;
    Qwen4ExpExpertCache expert_cache_;
    aila::Qwen4ExpExpertExecutor expert_executor_;
    aila::Qwen4ExpHostArena host_arena_;
    aila::Qwen4ExpAdaptiveCache adaptive_cache_;
    std::unique_ptr<sycl::queue> copy_q_;
    // V3 whole-layer staging (role G3, 2026-10-15): one device arena holding
    // whole-layer nonresident experts, filled during the attention window.
    // Allocated only when AILA_Q4E_LAYER_STAGE=1; empty otherwise.
    aila::Qwen4ExpLayerStage layer_stage_;
    std::vector<int> layer_stage_ids_;
    uint64_t layer_stage_epoch_ = 0;
    const bool use_adaptive_pipeline_ = true;
    int ping_pong_index_ = 0;
    void* resident_embedding_ = nullptr;
    std::vector<ops::MoeExpertSlotDev> adaptive_prefill_slots_;
    sycl::event RunAdaptivePrefill(Context& ctx,int layer,bool fuse_shared,int count);
    int resident_decode_mode_ = 0; // 0 legacy, 1 host Ready binding, 2 GPU resident selection
    // Token-scoped Bresenham credit for the opt-in adaptive cold split.
    int cold_split_credit_ = 0;
    int cold_split_num_ = 0;
    bool direct_arena_stage_ = false;
    bool cold_split_this_forward_ = false;
    int cold_mode_ = 0; // 0 dynamic, 1 frozen + GPU cold, 2 frozen + bounded CPU cold
    bool resident_prefill_ = false;
    bool cold_epoch_enabled_ = true; // disabled explicitly during synthetic engine warmup
    Tensor resident_down_, resident_weights_, resident_route_ids_, resident_selected_;
    // Opt-in adaptive multi-row verification scratch: device [4*top_k, hidden]
    // rank rows and [4*top_k] routed weights. Allocated only by
    // PrepareDecodeVerification, so ordinary decode never grows device memory.
    Tensor verify_down_, verify_weights_;
    int* resident_selected_host_ = nullptr;
    sycl::event resident_hot_reader_{};
    void* resident_probe_data_ = nullptr;
    void (*resident_probe_)(void*,Qwen4ExpBnb4Backend&,int) = nullptr;
    sycl::event RunResidentDecode(Context& ctx, int layer, bool fuse_shared, int count,
                                  bool verify_rows = false);
    sycl::event RunResidentDecode(Context& ctx, int layer, bool fuse_shared,
                                  GpuReadyLease* device_lease = nullptr,
                                  ResidentColdWork* early_cold = nullptr);

    Tensor* embed_weight_ = nullptr; // resident copy, only when AILA_Q4E_HOST_EMBED=0
    // Otherwise rows are gathered from the mmap per forward: decode needs one 5 KB row, and the
    // 1.3 GB table left in VRAM would displace ~460 expert slots.
    HostTensorView embed_host_{};
    int* embed_host_ids_ = nullptr;      // pinned [runtime_seq_capacity_]
    uint8_t* embed_host_rows_ = nullptr; // pinned [runtime_seq_capacity_, hidden] bf16
    Tensor lm_head_weight_transposed_;
    Linear lm_head_;
    // Experimental head replacement. Original host mmap remains owned by the
    // model weights; no resident BF16 head/oneDNN head exists when enabled.
    HostTensorView lm_head_host_{};
    Tensor lm_head_q8_weight_;
    Tensor lm_head_q8_scale_;
    Tensor lm_head_q8_input_;
    Tensor lm_head_q8_input_scale_;
    bool lmHeadQ8Enabled_ = false;
    uint64_t lmHeadQ8ForwardCount_ = 0, lmHeadQ8RowCount_ = 0;
    Tensor lastTokenLogits_;
    Tensor unusedLogits_; // Invalid output for forwards that only advance cached state.
    // Layer-major v3 carries: 8192-row activation/window buffers that must
    // outlive the two 4096-row halves while the standard runtime scratch stays
    // at 4096 rows. Allocated on the first layer-major prefill and reused.
    Tensor lm_hidden_, lm_x_, lm_y_, lm_inject_, lm_moe_h_, lm_down_;
    Tensor lm_shared_out_, lm_shared_gate_;
    std::vector<int> lm_ids_host_;
    int layer_major_map_capacity_ = 0;
    void EnsureLayerMajorCarries(Context& ctx, int count);
    // The layer-major expert pass indexes activations over the full 8192-row
    // window, so the route/map host and device mirrors need count slots even
    // when the standard runtime scratch stays at the 4096-row chunk size.
    void EnsureLayerMajorMaps(Context& ctx, int count);

    Qwen4ExpBuffers buf_;
    ops::GdnChunkScratch gdn_chunk_scratch_;
    int gdn_chunk_mode_ = 0; // 0 recurrence, 1 FP32 chunk control, 2 BF16 XMX products
    uint64_t gdn_chunk_calls_ = 0;
    // Pinned host [runtime_seq_capacity_, top_k] router outputs written directly by the top-k kernel.
    int* router_host_ids_ = nullptr;
    float* router_host_weights_ = nullptr;
    // S1 Batched prefill MoE slot descriptors and weights (host pinned buffers and device mirrors).
    ops::MoeExpertSlotDev* batch_slots_host_ = nullptr;
    ops::MoeExpertSlotDev* batch_slots_dev_ = nullptr;
    float* batch_weights_dev_ = nullptr;

    // Phase S4, W4 & G3: Expert-major prefill bucket sort and mapping tables (host pinned and device mirrors).
    int* expert_offsets_host_ = nullptr;
    int* expert_counts_host_ = nullptr;
    int* expert_curr_pos_host_ = nullptr;
    int* expert_eight_pos_host_ = nullptr;
    int* expert_quad_pos_host_ = nullptr;
    int* expert_pair_pos_host_ = nullptr;
    int* expert_single_pos_host_ = nullptr;
    int* expert_map_token_host_ = nullptr;
    int* expert_map_rank_host_ = nullptr;
    ops::MoeExpertSlotDev* expert_map_slot_host_ = nullptr;
    int* expert_map_token_dev_ = nullptr;
    int* expert_map_rank_dev_ = nullptr;
    ops::MoeExpertSlotDev* expert_map_slot_dev_ = nullptr;
    ops::MoeExpertTile* expert_tiles_host_ = nullptr;
    Tensor expert_tiles_dev_;
    bool grouped_xmx_supported_ = false;
    int grouped_xmx_min_rows_override_ = -1; // test harness only; -1 uses cached environment
    // R1 expert-XMX path. The plan staging holds the per-chunk bases/tokens/
    // ranks/slots; the ops scratch owns the bf16 weight banks and oneDNN
    // primitives and is allocated only when the flag is on.
    struct XmxPlanMember { int expert = 0; int offset = 0; int rows = 0; };
    std::vector<XmxPlanMember> xmx_members_;
    std::vector<unsigned char> xmx_plan_host_[2];
    unsigned char* xmx_plan_dev_[2] = {nullptr, nullptr};
    int xmx_plan_bytes_ = 0;
    int xmx_tokens_off_ = 0;
    int xmx_ranks_off_ = 0;
    int xmx_slots_off_ = 0;
    int xmx_bank_ = 0;
    bool prefill_expert_xmx_ = false;
    ops::MoeExpertXmxScratch xmx_scratch_{};
    // Test-only override permits paired scheduling trials in one loaded model.
    int small_rows_resident_split_override_ = -1;
    bool expert_histogram_enabled_ = false;
    // Buckets 1,2,3-4,5-8,9-16,>16: requests, activations, resident hits, misses.
    std::array<std::array<uint64_t, 4>, 6> expert_histogram_{};
    uint64_t grouped_xmx_tiles_ = 0, grouped_xmx_acts_ = 0;
    // Optional bounded real-input acceptance hook; null in normal inference.
    void (*grouped_expert_probe_)(Qwen4ExpBnb4Backend&, int, int, int, int, int,
                                  const sycl::event&) = nullptr;
    // Test-only real-input/replacement seam. The returned event must cover all
    // extra slot readers before cache eviction; null for normal inference.
    void* expert_quality_probe_data_ = nullptr;
    sycl::event (*expert_quality_probe_)(void*, Qwen4ExpBnb4Backend&, int, int,
                                        int, int, const sycl::event&) = nullptr;

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
    int moe_compact_tile_ = 0; // bounded activation rows, separate from router window
    int prefill_attention_tile_ = 0; // opt-in; independent of the routing window
};
