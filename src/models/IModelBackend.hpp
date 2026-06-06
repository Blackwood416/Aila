#pragma once

#include "../core/Context.hpp"
#include "../core/Tensor.hpp"
#include "../utils/SafeTensors.hpp"
#include "../lora/LoraLoader.hpp"
#include "engine/Types.hpp"
#include <string>
#include <vector>
#include <sycl/sycl.hpp>

class IModelBackend {
public:
    virtual ~IModelBackend() = default;

    virtual bool load(Context& ctx,
                      ModelWeights& weights,
                      const ModelSpec& spec,
                      int max_seq_len,
                      std::string* error_message) = 0;

    virtual bool apply_lora(Context& ctx, const aila::lora::LoraAdapter& adapter,
                            std::string* error_message = nullptr) { return true; }

    virtual Tensor& forward(Context& ctx, const int* token_ids_device, int seq_len) = 0;
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
    virtual Tensor* forward_mtp(Context& ctx, int next_token_id) { return nullptr; }
    virtual bool has_mtp() const { return false; }

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
};

