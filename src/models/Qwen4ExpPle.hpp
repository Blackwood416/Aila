#pragma once

#include "core/Context.hpp"
#include "core/Tensor.hpp"
#include "ops/Bnb4BitLinear.hpp"
#include "ops/HyperConnectionOps.hpp"
#include "utils/SafeTensors.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct Qwen4ExpPleState {
    static constexpr int NGRAM_CONTEXT_LEN = 2;
    static constexpr int CONV_CHANNELS = 10240;
    static constexpr int CONV_KERNEL_SIZE = 4;
    static constexpr int CONV_DILATION = 3;
    static constexpr int CONV_STATE_LEN = (CONV_KERNEL_SIZE - 1) * CONV_DILATION; // 9

    std::array<int, NGRAM_CONTEXT_LEN> prev_tokens = {248044, 248044}; // eos initialized
    int conv_head = 0;
    Tensor conv_state; // [9, 10240] f32 on GPU
};

struct Qwen4ExpPleSnapshot {
    std::array<int, 2> prev_tokens = {248044, 248044};
    int conv_head = 0;
    std::vector<float> conv_state; // [9 * 10240] f32 on host
};

struct PleShardRef {
    const uint8_t* packed_weight = nullptr; // [rows_per_shard * 80]
    const float* absmax = nullptr;          // [rows_per_shard * 5]
    const float* quant_map = nullptr;       // [16]
    int64_t rows = 0;
};

class Qwen4ExpPle {
public:
    static constexpr int EOS_TOKEN_ID = 248044;
    static constexpr int NGRAM_SIZE = 3;
    static constexpr int HEADS_PER_NGRAM = 8;
    static constexpr int NUM_HEADS = 16;
    static constexpr int HEAD_DIM = 160;
    static constexpr int EMBED_DIM = 2560; // 16 * 160
    static constexpr int HC_COUNT = 4;
    static constexpr int HC_HIDDEN_SIZE = 10240; // 4 * 2560
    static constexpr int NUM_SHARDS = 128;
    static constexpr int CONV_KERNEL_SIZE = 4;
    static constexpr int CONV_DILATION = 3;
    static constexpr int CONV_STATE_LEN = 9;

    Qwen4ExpPle() = default;
    ~Qwen4ExpPle();

    Qwen4ExpPle(const Qwen4ExpPle&) = delete;
    Qwen4ExpPle& operator=(const Qwen4ExpPle&) = delete;

    Qwen4ExpPle(Qwen4ExpPle&& other) noexcept;
    Qwen4ExpPle& operator=(Qwen4ExpPle&& other) noexcept;

    bool init(Context& ctx,
              ModelWeights& weights,
              int ple_layer_id = 1,
              int max_tokens = 2048,
              float eps = 1e-6f,
              std::string* error_message = nullptr);

    // Compute n-gram IDs from token history.
    // token_history: full sequence including initial context tokens [prev_tokens, input_ids]
    // total_tokens: length of token_history
    // num_tokens: number of tokens to produce IDs for (from the end of token_history)
    // out_ids: output buffer of size [num_tokens * NUM_HEADS]
    static void compute_ngram_ids(
        const int* token_history,
        int total_tokens,
        int num_tokens,
        const int64_t* layer_multipliers, // [3]
        const int64_t* vocab_sizes,       // [16]
        const int64_t* offsets,           // [16]
        int64_t* out_ids,
        int eos_token_id = EOS_TOKEN_ID);

    // Convenience method using internal tables and updating state.prev_tokens
    void compute_ngram_ids(
        Qwen4ExpPleState& state,
        const int* input_ids,
        int num_tokens,
        int64_t* out_ids) const;

    // Lookup row in 128 shards, dequantize NF4 (blocksize=32) into pinned staging buffer,
    // and DMA copy to out_embed_dev [num_tokens, 2560] bf16 on GPU.
    void lookup_and_dequant(
        Context& ctx,
        const int64_t* ids,
        int num_tokens,
        sycl::ext::oneapi::bfloat16* out_embed_dev);

    // Forward pass:
    // hidden_states: [num_tokens, 10240] bf16 on device
    // input_ids: [num_tokens] int on host
    // out: [num_tokens, 10240] bf16 on device (residual to be added to hidden_states)
    void forward(
        Context& ctx,
        Qwen4ExpPleState& state,
        const int* input_ids,
        int num_tokens,
        const sycl::ext::oneapi::bfloat16* hidden_states,
        sycl::ext::oneapi::bfloat16* out);

    // Session state helpers
    void init_state(Context& ctx, Qwen4ExpPleState& state) const;
    void capture_snapshot(Context& ctx, const Qwen4ExpPleState& state, Qwen4ExpPleSnapshot& snapshot) const;
    void restore_snapshot(Context& ctx, Qwen4ExpPleState& state, const Qwen4ExpPleSnapshot& snapshot) const;

    // Set shards explicitly (useful for unit tests)
    void set_shards(const std::vector<PleShardRef>& shards, int64_t rows_per_shard);
    void set_multipliers_and_vocab(
        const std::array<int64_t, 3>& multipliers,
        const std::array<int64_t, 16>& vocab_sizes,
        const std::array<int64_t, 16>& offsets);

    bool is_initialized() const { return initialized_; }

    const std::array<int64_t, 3>& layer_multipliers() const { return layer_multipliers_; }
    const std::array<int64_t, 16>& ngram_heads_vocab_sizes() const { return ngram_heads_vocab_sizes_; }
    const std::array<int64_t, 16>& ngram_heads_offsets() const { return ngram_heads_offsets_; }

    Bnb4BitLinear& key_proj() { return key_proj_; }
    Bnb4BitLinear& value_proj() { return value_proj_; }
    Tensor& norm_key_weight() { return norm_key_w_; }
    Tensor& norm_query_weight() { return norm_query_w_; }
    Tensor& norm_conv_weight() { return norm_conv_w_; }
    const sycl::ext::oneapi::bfloat16* conv1d_weight_ptr() const { return conv1d_w_ptr_; }

private:
    void cleanup();

    Context* ctx_ = nullptr;
    int ple_layer_id_ = 1;
    int max_tokens_ = 2048;
    float eps_ = 1e-6f;

    std::array<int64_t, 3> layer_multipliers_{};
    std::array<int64_t, 16> ngram_heads_vocab_sizes_{};
    std::array<int64_t, 16> ngram_heads_offsets_{};

    std::vector<PleShardRef> shards_;
    int64_t rows_per_shard_ = 0;

    Bnb4BitLinear key_proj_;
    Bnb4BitLinear value_proj_;
    Tensor norm_key_w_;   // [10240] f32
    Tensor norm_query_w_; // [10240] f32
    Tensor norm_conv_w_;  // [10240] f32
    Tensor owned_conv1d_w_; // [10240, 4] bf16
    const sycl::ext::oneapi::bfloat16* conv1d_w_ptr_ = nullptr;

    // Pre-allocated pinned host staging buffer for n-gram dequantization
    void* pinned_staging_embed_ = nullptr;
    size_t staging_embed_bytes_ = 0;

    // Pre-allocated device tensors for forward execution (zero allocations on hot path)
    Tensor dev_embed_;    // [max_tokens, 2560] bf16
    Tensor key_buf_;      // [max_tokens, 10240] bf16
    Tensor value_buf_;    // [max_tokens, 2560] bf16
    Tensor key_normed_;   // [max_tokens, 10240] bf16
    Tensor query_normed_; // [max_tokens, 10240] bf16
    Tensor gv_buf_;       // [max_tokens, 10240] bf16
    Tensor gv_normed_;    // [max_tokens, 10240] bf16

    Bnb4BitLinearScratch scratch_key_;
    Bnb4BitLinearScratch scratch_val_;

    bool initialized_ = false;
};
