#pragma once

#include "../core/Context.hpp"
#include "../core/Tensor.hpp"
#include "../utils/SafeTensors.hpp"
#include "../lora/LoraLoader.hpp"
#include "engine/Types.hpp"
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <sycl/sycl.hpp>

class ModelBackendCancelled final : public std::runtime_error {
public:
    ModelBackendCancelled()
        : std::runtime_error("model backend inference cancelled") {}
};

class IModelBackend {
public:
    struct TtsBackendTiming {
        double codes_ms = -1.0;
        double mimi_init_ms = -1.0;
        double first_audio_ms = -1.0;
        double total_ms = -1.0;
        int total_frames = 0;
        int callback_count = 0;
        int first_audio_samples = 0;
        int stream_batch_frames = 0;
        int initial_stream_batch_frames = 0;
        int steady_stream_batch_frames = 0;
        int steady_batch_callback_count = 0;
        int playback_aware_steady_batch = 0;
    };

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

    virtual void set_cancellation_checker(std::function<bool()> should_cancel) {
        cancellation_checker_ = std::move(should_cancel);
    }

    virtual void set_internal_prefill_state_snapshots_override(
        std::optional<bool> enabled) {
        (void)enabled;
    }

    virtual bool synthesize_tts_stream(
        Context& ctx,
        const std::vector<int>& text_tokens,
        const GenerationConfig& gen_config,
        int stream_batch_frames,
        std::function<void(const std::vector<float>&)> audio_callback,
        std::string* error_message = nullptr,
        std::function<bool()> should_cancel = {}) {
        (void)ctx;
        (void)text_tokens;
        (void)gen_config;
        (void)stream_batch_frames;
        (void)audio_callback;
        (void)should_cancel;
        if (error_message) {
            *error_message = "backend does not support TTS streaming";
        }
        return false;
    }

    virtual TtsBackendTiming last_tts_backend_timing() const {
        return {};
    }

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

protected:
    bool cancellation_requested() const {
        return cancellation_checker_ && cancellation_checker_();
    }

    void throw_if_cancelled() const {
        if (cancellation_requested()) {
            throw ModelBackendCancelled();
        }
    }

private:
    std::function<bool()> cancellation_checker_;
};
