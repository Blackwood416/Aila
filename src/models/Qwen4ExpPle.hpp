#pragma once

#include "core/Context.hpp"
#include "core/Tensor.hpp"
#include "ops/Bnb4BitLinear.hpp"
#include "ops/HyperConnectionOps.hpp"
#include "utils/SafeTensors.hpp"

#include <array>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
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
    // Optional graph-only [1] s32 ring head. Allocated by create_graph_view();
    // once valid, graph consume is the authority and host consume must not run.
    Tensor conv_head_dev;
};

struct Qwen4ExpPleSnapshot {
    std::array<int, 2> prev_tokens = {248044, 248044};
    int conv_head = 0;
    std::vector<float> conv_state = std::vector<float>(
        Qwen4ExpPleState::CONV_STATE_LEN * Qwen4ExpPleState::CONV_CHANNELS);
};

struct PleShardRef {
    const uint8_t* packed_weight = nullptr; // [rows_per_shard * 80]
    const float* absmax = nullptr;          // [rows_per_shard * 5]
    const float* quant_map = nullptr;       // [16]
    int64_t rows = 0;
};

// P0 Candidate R1: PleTicket tracking session, generation, staging slot, and async DMA event.
struct PleTicket {
    const void* session_ptr = nullptr;
    uint64_t generation = 0;
    int start_pos = 0;
    int num_tokens = 0;
    std::array<int, 2> prev_tokens_in{248044, 248044};
    std::array<int, 2> prev_tokens_next{248044, 248044};
    int staging_slot = 0;
    uint64_t task_id = 0;
    sycl::event h2d_event{};
    bool valid = false;
};

// Fixed-address view for one-token PLE graph replay. It deliberately contains
// only device pointers: preparation, waits, commit, tickets and host state stay
// outside the captured graph.
struct Qwen4ExpPleGraphView {
    bf16* dev_embed = nullptr;
    const bf16* hidden_states = nullptr;
    bf16* out = nullptr;
    bf16* residual = nullptr;
    bf16* key_buf = nullptr;
    bf16* value_buf = nullptr;
    bf16* key_normed = nullptr;
    bf16* query_normed = nullptr;
    bf16* gv_buf = nullptr;
    bf16* gv_normed = nullptr;
    float* conv_state = nullptr;
    int* conv_head_dev = nullptr;
    const float* norm_key_weight = nullptr;
    const float* norm_query_weight = nullptr;
    const float* norm_conv_weight = nullptr;
    const bf16* conv_weight = nullptr;
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
    static constexpr int kNumStagingSlots = 2; // Double buffered staging slots

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

    // Pure n-gram IDs calculation (does NOT mutate state.prev_tokens)
    void compute_ngram_ids_pure(
        const Qwen4ExpPleState& state,
        const int* input_ids,
        int num_tokens,
        int64_t* out_ids,
        std::array<int, 2>& candidate_prev_tokens) const;

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

    // P0 Candidate R1 Decoupled Pipeline API:
    // Pure prepare: kicks off background hash, gather, dequant and H2D DMA to staging slot.
    PleTicket prepare(
        Context& ctx,
        const Qwen4ExpPleState& state,
        const int* input_ids,
        int num_tokens,
        const void* session_ptr = nullptr,
        uint64_t generation = 0,
        int start_pos = 0);

    // Graph path: always uses staging slot 0. Call wait_prepared() on the host
    // before recording/launching; consume_graph never waits or advances any
    // host-side ticket/state.
    PleTicket prepare_fixed_slot(
        Context& ctx,
        const Qwen4ExpPleState& state,
        const int* input_ids,
        int num_tokens,
        const void* session_ptr = nullptr,
        uint64_t generation = 0,
        int start_pos = 0);
    void wait_prepared(Context& ctx, PleTicket& ticket);

    // The two staging slots own their token copies. A cancelled/replaced
    // request must not reuse a ticket merely because its position matches.
    bool MatchesInput(const PleTicket& ticket, const int* tokens, int count) const;

    // Consume: waits for ticket ready and executes projection & fusion kernels.
    void consume(
        Context& ctx,
        PleTicket& ticket,
        Qwen4ExpPleState& state,
        const sycl::ext::oneapi::bfloat16* hidden_states,
        sycl::ext::oneapi::bfloat16* out,
        sycl::ext::oneapi::bfloat16* residual = nullptr,
        int checkpoint_after = 0,
        Qwen4ExpPleSnapshot* checkpoint = nullptr);

    // Builds a fixed-address one-token view. This may allocate/grow workspace
    // and initialize state.conv_head_dev, so call it outside graph recording.
    Qwen4ExpPleGraphView create_graph_view(
        Context& ctx,
        Qwen4ExpPleState& state,
        const bf16* hidden_states,
        bf16* out,
        bf16* residual = nullptr);

    // Queues only fixed one-row kernels. No wait, capacity growth, ticket
    // validation, host-head update or commit occurs here.
    void consume_graph(Context& ctx, const Qwen4ExpPleGraphView& view);

    // Commit: updates state.prev_tokens only after successful execution.
    void commit(
        PleTicket& ticket,
        Qwen4ExpPleState& state);

    // Forward pass (legacy convenience wrapper calling prepare -> consume -> commit):
    // hidden_states: [num_tokens, 10240] bf16 on device
    // input_ids: [num_tokens] int on host
    // out: [num_tokens, 10240] bf16 on device (residual to be added to hidden_states)
    void forward(
        Context& ctx,
        Qwen4ExpPleState& state,
        const int* input_ids,
        int num_tokens,
        const sycl::ext::oneapi::bfloat16* hidden_states,
        sycl::ext::oneapi::bfloat16* out,
        sycl::ext::oneapi::bfloat16* residual = nullptr);

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

    // Opt-in alignment (B4-2): treat a 2..4 token verify window as independent
    // single-row projections so it matches sequential B1 decode bit-for-bit.
    void set_rows_decode(bool value) { rows_decode_ = value; }

private:
    PleTicket prepare_slot(
        Context& ctx,
        const Qwen4ExpPleState& state,
        const int* input_ids,
        int num_tokens,
        const void* session_ptr,
        uint64_t generation,
        int start_pos,
        int slot);

    void cleanup();
    void ensure_compute_capacity(Context& ctx, int rows);
    void prefetch_gather_pages(const int64_t* ids, int num_tokens) const;
    void dequant_to_staging(const int64_t* ids, int num_tokens, int slot);
    void execute_prepare_slot(int slot, int num_tokens, const std::array<int, 2>& prev_tokens);
    void worker_loop();
    void wait_task(uint64_t task_id);

    Context* ctx_ = nullptr;
    int ple_layer_id_ = 1;
    int max_tokens_ = 2048;
    int compute_capacity_ = 0;
    float eps_ = 1e-6f;
    bool rows_decode_ = false;

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

    // Double-buffered staging slots (pinned host memory & device tensors)
    size_t staging_embed_bytes_ = 0;
    void* pinned_staging_embed_ = nullptr; // alias to slot 0 for legacy compatibility
    std::array<void*, kNumStagingSlots> pinned_staging_embed_slots_{nullptr, nullptr};
    std::array<Tensor, kNumStagingSlots> dev_embed_slots_;
    std::array<sycl::event, kNumStagingSlots> slot_h2d_events_;
    std::array<std::vector<int64_t>, kNumStagingSlots> slot_ngram_ids_;
    std::array<std::vector<int>, kNumStagingSlots> slot_input_ids_;
    std::array<std::vector<int>, kNumStagingSlots> slot_history_;

    // Dedicated background worker and copy queue
    std::unique_ptr<sycl::queue> copy_q_;
    std::thread worker_thread_;
    std::mutex worker_mutex_;
    std::condition_variable worker_cv_;
    std::mutex done_mutex_;
    std::condition_variable done_cv_;
    bool stop_worker_ = false;
    uint64_t current_task_id_ = 0;
    uint64_t completed_task_id_ = 0;
    mutable int next_staging_slot_ = 0;

    struct WorkerTask {
        uint64_t task_id = 0;
        int slot = 0;
        int num_tokens = 0;
        std::array<int, 2> prev_tokens_in{};
    };
    std::deque<WorkerTask> task_queue_;
    std::array<uint64_t, kNumStagingSlots> slot_task_ids_{0, 0};

    // Pre-allocated device tensors for forward execution (zero allocations on hot path)
    Tensor dev_embed_;    // alias / view to dev_embed_slots_[0] for legacy compatibility
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
