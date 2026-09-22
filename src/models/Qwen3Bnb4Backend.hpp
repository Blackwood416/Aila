#pragma once

#include "IModelBackend.hpp"
#include "../ops/Bnb4BitLinear.hpp"
#include "../ops/Ops.hpp"
#include "../core/Tensor.hpp"
#include "../memory/KVCache.hpp"
#include "engine/Types.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct Qwen3Bnb4Layer {
    Tensor* input_ln_weight = nullptr;
    Tensor* post_attn_ln_weight = nullptr;
    Tensor* q_norm_weight = nullptr;
    Tensor* k_norm_weight = nullptr;

    Bnb4BitLinear qkv_proj;      // fused Q/K/V (3-column horizontal concat)
    Bnb4BitLinear o_proj;
    Bnb4BitLinear gate_up_proj;  // fused gate/up (2-column horizontal concat)
    Bnb4BitLinear down_proj;
};

class Qwen3Bnb4Backend : public IModelBackend {
    struct SessionOwner {};

public:
    // Session resources are independent; the backend's projections, scratch
    // and logits remain one execution slot. Callers serialize all operations,
    // keep Context/model resources alive, and drain the queue before destroying
    // a Session. No weights or execution buffers are copied into a Session.
    class Session {
    public:
        ~Session() = default;
        Session(const Session&) = delete;
        Session& operator=(const Session&) = delete;
        Session(Session&&) = delete;
        Session& operator=(Session&&) = delete;

        int CurrentLength() const { return cache_.current_length(); }
        int Capacity() const { return cache_.max_length(); }
        std::size_t CacheBytes() const { return cacheBytes_; }

    private:
        friend class Qwen3Bnb4Backend;
        Session(Context& ctx, const Qwen3Config& config, int capacity,
                std::shared_ptr<const SessionOwner> owner);

        Context* context_ = nullptr;
        std::shared_ptr<const SessionOwner> owner_;
        KVCache cache_;
        std::size_t cacheBytes_ = 0;
    };

    ~Qwen3Bnb4Backend() override;

    bool load(Context& ctx, ModelWeights& weights, const ModelSpec& spec,
              int max_seq_len, std::string* error_message) override;
    bool apply_lora(Context& ctx, const aila::lora::LoraAdapter& adapter,
                    std::string* error_message = nullptr) override;
    Tensor& forward(Context& ctx, const int* token_ids_device, int seq_len) override;
    void reset() override;
    bool truncate_kv_cache(int new_len) override;
    int get_current_context_len() const override {
        return defaultSession_ ? defaultSession_->CurrentLength() : 0;
    }
    int max_seq_len() const override { return max_seq_len_; }
    int vocab_size() const override { return cfg_.vocab_size; }
    ModelFamily family() const override { return ModelFamily::Qwen3Dense; }

    // Explicit sessions require BF16 KV (AILA_KV_QUANT=0).
    std::unique_ptr<Session> CreateSession(Context& ctx);
    // The returned logits belong to the execution slot and must be consumed
    // before the next forward call (including calls through the legacy API).
    Tensor& Forward(Context& ctx, Session& session,
                    const int* tokenIdsDevice, int count);
    // One token for each distinct Session, in the supplied row order. B=1
    // retains Forward's original decode path; B=2..4 shares each layer's
    // projections while positional/KV attention stays per Session. The
    // returned contiguous [B, vocab] BF16 view uses the execution slot.
    // Validate the whole group before submission. After an execution failure
    // the caller must invalidate every participating Session, as with Forward.
    bool SupportsDecodeBatch() const noexcept;
    Tensor& ForwardDecodeBatch(Context& ctx, const std::vector<Session*>& sessions,
                               const int* deviceTokenIds);
    void Reset(Session& session);
    // Returns the restored length. Invalid lengths leave the Session intact.
    int Truncate(Session& session, int newLength);

private:
    enum class ExecutionPhase { Prefill, Decode };

    // A concrete full-GQA + dense-SwiGLU NF4 layer. Bindings are fixed at
    // load time; mutable projection state stays in the single execution slot.
    struct LayerPlan {
        Qwen3Bnb4Layer* layer = nullptr;
        Tensor* postMlpNormWeight = nullptr;
        int cacheIndex = 0;
    };

    struct DecoderPlan {
        Tensor* initialNormWeight = nullptr;
        std::vector<LayerPlan> layers;
    };

    struct LayerExecution;

    void BuildDecoderPlan();
    void ExecuteLayer(const LayerPlan& plan, const LayerExecution& execution);
    std::unique_ptr<Session> CreateSessionImpl(Context& ctx);
    void ValidateSession(const Session& session) const;
    void ensure_runtime_buffers(Context& ctx, int seq_len);
    void ensure_prefill_scores(Context& ctx, int seq_len);
    void ensure_incr_prefill_scores(Context& ctx, int seq_len, int total_len);

    Qwen3Config cfg_{};
    int hidden_size_ = 0;
    int ff_dim_ = 0;
    int max_seq_len_ = 0;

    int q_dim_ = 0;
    int kv_dim_ = 0;
    int fused_qkv_dim_ = 0;  // q_dim_ + 2 * kv_dim_

    Context* context_ = nullptr;
    std::shared_ptr<const SessionOwner> sessionOwner_;
    std::unique_ptr<Session> defaultSession_;
    bool explicitSessionsCreated_ = false;

    std::vector<Qwen3Bnb4Layer> layers_;
    std::unique_ptr<const DecoderPlan> decoderPlan_;
    std::vector<Tensor> fused_weights_;
    Bnb4BitLinearScratch linear_scratch_;

    Tensor* embed_weight_ = nullptr;
    Tensor* final_norm_weight_ = nullptr;
    Linear lm_head_;

    struct Buffers {
        Tensor hidden, normed, qkv;
        Tensor q, k, v, attn_out;
        Tensor gate_up, gate, up, logits;
        Tensor decode_batch_logits;
        Tensor decode_scores, scores, incr_scores;
        Tensor rope_freq;
        Tensor decode_attn_partials; // cache for head_dim=256 decode exact partials
    } buf_;

    Tensor decodeBatchLogitsView_;

    int runtime_seq_capacity_ = 0;
    int prefill_scores_capacity_ = 0;
    int incr_prefill_seq_cap_ = 0;
    int incr_prefill_total_cap_ = 0;
};
