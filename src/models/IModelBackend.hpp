#pragma once

#include "../core/Context.hpp"
#include "../core/Tensor.hpp"
#include "../utils/SafeTensors.hpp"
#include "../lora/LoraLoader.hpp"
#include "engine/Types.hpp"
#include <string>
#include <vector>
#include <sycl/sycl.hpp>

struct ForwardOptions {
    SnapshotPolicy snapshot_policy = SnapshotPolicy::ReusablePrefix;
    bool need_logits = true;
    bool capture_snapshot = false; // Force a checkpoint at an explicit prefix boundary.
    int checkpoint_after = 0; // Tokens within this forward; zero disables the interior checkpoint.
    int prefill_remaining = -1; // Known prompt tokens after this slice; -1 for decode/unspecified.
    bool decode_verification = false; // Explicit multi-token decode, not a new prompt slice.
};

class IModelBackend {
public:
    virtual ~IModelBackend() = default;

    virtual bool load(Context& ctx,
                      ModelWeights& weights,
                      const ModelSpec& spec,
                      int max_seq_len,
                      std::string* error_message) = 0;

    virtual bool apply_lora(Context&, const aila::lora::LoraAdapter&,
                            std::string* error_message = nullptr) {
        if (error_message) {
            *error_message = "LoRA is not supported by this model backend";
        }
        return false;
    }

    virtual Tensor& forward(Context& ctx, const int* token_ids_device, int seq_len) = 0;
    virtual bool supports_interior_checkpoint(int, int) const { return false; }
    // Backends may skip the output head when logits are not consumed.
    virtual Tensor& forward(Context& ctx, const int* token_ids_device, int seq_len,
                            const ForwardOptions&) {
        return forward(ctx, token_ids_device, seq_len);
    }
    virtual void reset() = 0;
    // Truncate cached state to new_len positions.  Returns true if the
    // truncation was clean (state is consistent at new_len so incremental
    // prefill can continue).  Returns false if a full reset was necessary
    // (caller must do a full prefill of all prompt tokens).
    virtual bool truncate_kv_cache(int new_len) = 0;
    virtual int get_current_context_len() const { return 0; }
    virtual int max_seq_len() const = 0;
    virtual int vocab_size() const = 0;
    virtual ModelFamily family() const = 0;

    virtual bool supports_vision_embedding_override() const { return false; }
    virtual void set_embedding_overrides(
        const std::vector<int>& positions,
        const std::vector<sycl::ext::oneapi::bfloat16>& embeddings,
        int hidden_size) {
        (void)positions;
        (void)embeddings;
        (void)hidden_size;
    }
    virtual void clear_embedding_overrides() {}
    virtual void set_mrope_positions(Context& ctx,
                                     const std::vector<int>& pos_t,
                                     const std::vector<int>& pos_h,
                                     const std::vector<int>& pos_w,
                                     int text_pos_delta) {
        (void)ctx;
        (void)pos_t;
        (void)pos_h;
        (void)pos_w;
        (void)text_pos_delta;
    }
    virtual void clear_mrope_positions() {}

    // Block-diffusion generation support (DiffusionGemma).
    virtual bool supports_block_diffusion() const { return false; }
    // Causal encoder prefill/decode: embeds tokens, runs all layers, appends
    // K/V to the internal cache. No logits are produced.
    virtual bool diffusion_encode(Context& ctx, const int* token_ids_device,
                                  int seq_len, std::string* error_message) {
        (void)ctx;
        (void)token_ids_device;
        (void)seq_len;
        if (error_message) {
            *error_message = "block diffusion is not supported by this backend";
        }
        return false;
    }
    // Bidirectional canvas decode over [cached KV + canvas KV]. Reads the KV
    // cache without writing; returns softcapped fp32 logits [canvas_len, vocab].
    virtual Tensor& diffusion_decode_canvas(
        Context& ctx, const int* canvas_ids_device,
        const sycl::ext::oneapi::bfloat16* sc_embeds_device, int canvas_len,
        float temperature = 1.0f) {
        (void)ctx;
        (void)canvas_ids_device;
        (void)sc_embeds_device;
        (void)canvas_len;
        (void)temperature;
        throw std::runtime_error(
            "block diffusion is not supported by this backend");
    }
    // Optional device buffers computed during fused post-logits decode
    virtual int* diffusion_argmax_device() { return nullptr; }
    virtual float* diffusion_entropy_device() { return nullptr; }

    // Computes soft embeddings = softmax(logits) @ embed^T * embed_scale
    // (bf16). logits: [canvas_len, vocab] fp32; out: [canvas_len, hidden] bf16.
    virtual Tensor& diffusion_compute_soft_embeds(
        Context& ctx, Tensor& logits_f32, int canvas_len) {
        (void)ctx;
        (void)logits_f32;
        (void)canvas_len;
        throw std::runtime_error(
            "block diffusion is not supported by this backend");
    }
};

