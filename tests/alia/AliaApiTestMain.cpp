#include "alia_api.h"
#include "alia/AliaContext.hpp"
#include "alia/AliaBackgroundPipeline.hpp"
#include "alia/AliaAsrPipeline.hpp"
#include "alia/AliaTtsPipeline.hpp"
#include "alia/RuntimeContext.hpp"
#include "core/Tensor.hpp"
#include "models/IModelBackend.hpp"
#include "utils/Tokenizer.hpp"

#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <condition_variable>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void expect_true(bool condition, const char* expression, int line) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL line " << line << ": expected " << expression << "\n";
    }
}

#define ALIA_EXPECT_TRUE(expr) expect_true((expr), #expr, __LINE__)
#define ALIA_EXPECT_EQ(actual, expected) \
    expect_true(((actual) == (expected)), #actual " == " #expected, __LINE__)

std::filesystem::path make_temp_model_dir(const std::string& name, const std::string& config_json) {
    auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::filesystem::path dir = std::filesystem::temp_directory_path() /
        ("aila_alia_api_tests_" + name + "_" + std::to_string(stamp));
    std::filesystem::create_directories(dir);
    std::ofstream out(dir / "config.json", std::ios::binary);
    out << config_json;
    return dir;
}

std::filesystem::path make_temp_tokenizer_dir(const std::string& name,
                                              const std::vector<std::string>& normal_tokens) {
    auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::filesystem::path dir = std::filesystem::temp_directory_path() /
        ("aila_alia_api_tests_" + name + "_" + std::to_string(stamp));
    std::filesystem::create_directories(dir);

    const int tool_result_open_id = static_cast<int>(normal_tokens.size());
    const int tool_result_close_id = tool_result_open_id + 1;
    const int im_start_id = tool_result_open_id + 2;
    const int im_end_id = tool_result_open_id + 3;
    const int endoftext_id = tool_result_open_id + 4;

    std::ofstream out(dir / "tokenizer.json", std::ios::binary);
    out << "{\"added_tokens\":[";
    out << "{\"id\":" << tool_result_open_id << ",\"content\":\"<tool_result>\"},";
    out << "{\"id\":" << tool_result_close_id << ",\"content\":\"</tool_result>\"},";
    out << "{\"id\":" << im_start_id << ",\"content\":\"<|im_start|>\"},";
    out << "{\"id\":" << im_end_id << ",\"content\":\"<|im_end|>\"},";
    out << "{\"id\":" << endoftext_id << ",\"content\":\"<|endoftext|>\"}";
    out << "],\"model\":{\"vocab\":{";
    bool first = true;
    auto write_vocab_entry = [&](const std::string& token, int id) {
        if (!first) {
            out << ",";
        }
        first = false;
        out << "\"" << aila::alia::AliaBackgroundPipeline::json_escape(token)
            << "\":" << id;
    };
    for (size_t i = 0; i < normal_tokens.size(); ++i) {
        write_vocab_entry(normal_tokens[i], static_cast<int>(i));
    }
    write_vocab_entry("<tool_result>", tool_result_open_id);
    write_vocab_entry("</tool_result>", tool_result_close_id);
    write_vocab_entry("<|im_start|>", im_start_id);
    write_vocab_entry("<|im_end|>", im_end_id);
    write_vocab_entry("<|endoftext|>", endoftext_id);
    out << "},\"merges\":[]}}";
    return dir;
}

void remove_temp_dir(const std::filesystem::path& dir) {
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

std::string qwen35_nf4_config() {
    return R"({
        "model_type":"qwen3_5",
        "quantization_config":{
            "quant_method":"bitsandbytes",
            "load_in_4bit":true,
            "bnb_4bit_quant_type":"nf4",
            "bnb_4bit_compute_dtype":"float16",
            "bnb_4bit_quant_storage":"uint8"
        }
    })";
}

struct BlockingAudioCallbackState {
    std::mutex mutex;
    std::condition_variable cv;
    bool entered = false;
    bool release = false;
    int call_count = 0;
    int sample_count = 0;
};

struct BlockingBackgroundCallbackState {
    std::mutex mutex;
    std::condition_variable cv;
    bool entered = false;
    bool release = false;
    int call_count = 0;
    std::string result_json;
    void* user_data = reinterpret_cast<void*>(0x1);
};

BlockingBackgroundCallbackState* g_background_callback_state = nullptr;

struct RecordingBackgroundCallbackState {
    int call_count = 0;
    std::string result_json;
    void* user_data = reinterpret_cast<void*>(0x1);
};

RecordingBackgroundCallbackState* g_recording_background_callback_state = nullptr;

struct ToolCallbackState {
    int call_count = 0;
    std::string tool_json;
    std::string result_to_write = R"({"ok":true,"text":"window title is Settings"})";
};

struct CountingAudioCallbackState {
    int call_count = 0;
    int total_samples = 0;
};

struct RecordingAudioCallbackState {
    int call_count = 0;
    std::vector<float> samples;
};

class SessionCountingBackend final : public IModelBackend {
public:
    explicit SessionCountingBackend(
        Context& ctx,
        int vocab_size = 3,
        std::vector<std::vector<int>> generation_sequences = {{0}, {1}})
        : logits_(Tensor::allocate(ctx, {vocab_size}, dnnl::memory::data_type::bf16)),
          generation_sequences_(std::move(generation_sequences)),
          vocab_size_(vocab_size),
          fallback_token_id_(std::max(0, vocab_size - 1)) {}

    bool load(Context&,
              ModelWeights&,
              const ModelSpec&,
              int,
              std::string*) override {
        return true;
    }

    Tensor& forward(Context& ctx, const int*, int seq_len) override {
        context_len_ += seq_len;
        if (seq_len > 1) {
            prefill_lengths_.push_back(seq_len);
            active_sequence_index_ = std::min(
                static_cast<size_t>(prefill_count_),
                generation_sequences_.empty() ? size_t{0} : generation_sequences_.size() - 1);
            active_sequence_offset_ = 0;
            prefill_count_++;
        }
        const int token_id = next_token_id();

        using Bf16 = sycl::ext::oneapi::bfloat16;
        std::vector<Bf16> host_logits(static_cast<size_t>(vocab_size_), Bf16(-100.0f));
        host_logits[static_cast<size_t>(token_id)] = Bf16(100.0f);
        ctx.memcpy_h2d(logits_.data(), host_logits.data(),
                       host_logits.size() * sizeof(Bf16));
        return logits_;
    }

    void reset() override {
        reset_count_++;
        context_len_ = 0;
    }

    bool truncate_kv_cache(int new_len) override {
        context_len_ = new_len;
        return true;
    }

    int get_current_context_len() const override { return context_len_; }
    int max_seq_len() const override { return 4096; }
    int vocab_size() const override { return vocab_size_; }
    ModelFamily family() const override { return ModelFamily::Qwen35Hybrid; }

    int reset_count() const { return reset_count_; }
    int prefill_count() const { return prefill_count_; }
    int prefill_length(size_t index) const {
        return index < prefill_lengths_.size() ? prefill_lengths_[index] : -1;
    }
    const std::vector<int>& sampled_token_ids() const { return sampled_token_ids_; }

private:
    int next_token_id() {
        if (generation_sequences_.empty()) {
            sampled_token_ids_.push_back(fallback_token_id_);
            return fallback_token_id_;
        }
        const auto& sequence = generation_sequences_[active_sequence_index_];
        int token_id = fallback_token_id_;
        if (active_sequence_offset_ < sequence.size()) {
            token_id = sequence[active_sequence_offset_++];
        }
        token_id = std::clamp(token_id, 0, std::max(0, vocab_size_ - 1));
        sampled_token_ids_.push_back(token_id);
        return token_id;
    }

    Tensor logits_;
    std::vector<std::vector<int>> generation_sequences_;
    int vocab_size_ = 3;
    int fallback_token_id_ = 2;
    size_t active_sequence_index_ = 0;
    size_t active_sequence_offset_ = 0;
    int reset_count_ = 0;
    int prefill_count_ = 0;
    int context_len_ = 0;
    std::vector<int> prefill_lengths_;
    std::vector<int> sampled_token_ids_;
};

class StreamingTtsBackend final : public IModelBackend {
public:
    explicit StreamingTtsBackend(Context& ctx)
        : logits_(Tensor::allocate(ctx, {1}, dnnl::memory::data_type::bf16)) {}

    bool load(Context&,
              ModelWeights&,
              const ModelSpec&,
              int,
              std::string*) override {
        return true;
    }

    Tensor& forward(Context&, const int*, int) override {
        return logits_;
    }

    void reset() override {
        reset_count_++;
    }

    bool truncate_kv_cache(int) override {
        return true;
    }

    int max_seq_len() const override { return 4096; }
    int vocab_size() const override { return 32; }
    ModelFamily family() const override { return ModelFamily::Qwen3TTS; }

    bool synthesize_tts_stream(Context&,
                               const std::vector<int>& text_tokens,
                               const GenerationConfig& gen_config,
                               int stream_batch_frames,
                               std::function<void(const std::vector<float>&)> audio_callback,
                               std::string*,
                               std::function<bool()> should_cancel) override {
        if (should_cancel && should_cancel()) {
            return false;
        }
        stream_call_count_++;
        last_text_tokens_ = text_tokens;
        last_config_ = gen_config;
        last_stream_batch_frames_ = stream_batch_frames;
        audio_callback(std::vector<float>{0.25f, -0.5f});
        audio_callback(std::vector<float>{0.75f});
        return true;
    }

    int reset_count() const { return reset_count_; }
    int stream_call_count() const { return stream_call_count_; }
    const std::vector<int>& last_text_tokens() const { return last_text_tokens_; }
    const GenerationConfig& last_config() const { return last_config_; }
    int last_stream_batch_frames() const { return last_stream_batch_frames_; }

private:
    Tensor logits_;
    int reset_count_ = 0;
    int stream_call_count_ = 0;
    std::vector<int> last_text_tokens_;
    GenerationConfig last_config_;
    int last_stream_batch_frames_ = 0;
};

class CancellableStreamingTtsBackend final : public IModelBackend {
public:
    explicit CancellableStreamingTtsBackend(Context& ctx)
        : logits_(Tensor::allocate(ctx, {1}, dnnl::memory::data_type::bf16)) {}

    bool load(Context&,
              ModelWeights&,
              const ModelSpec&,
              int,
              std::string*) override {
        return true;
    }

    Tensor& forward(Context&, const int*, int) override {
        return logits_;
    }

    void reset() override {}
    bool truncate_kv_cache(int) override { return true; }
    int max_seq_len() const override { return 4096; }
    int vocab_size() const override { return 32; }
    ModelFamily family() const override { return ModelFamily::Qwen3TTS; }

    bool synthesize_tts_stream(Context&,
                               const std::vector<int>&,
                               const GenerationConfig&,
                               int,
                               std::function<void(const std::vector<float>&)>,
                               std::string*,
                               std::function<bool()> should_cancel) override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stream_call_count_++;
            entered_ = true;
        }
        cv_.notify_all();

        for (int i = 0; i < 200; ++i) {
            if (should_cancel && should_cancel()) {
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    observed_cancel_ = true;
                }
                cv_.notify_all();
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        return true;
    }

    bool wait_until_entered_for(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, timeout, [&]() { return entered_; });
    }

    bool observed_cancel() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return observed_cancel_;
    }

    int stream_call_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return stream_call_count_;
    }

private:
    Tensor logits_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool entered_ = false;
    bool observed_cancel_ = false;
    int stream_call_count_ = 0;
};

void blocking_audio_callback(const float* samples, int sample_count, void* user_data) {
    auto* state = static_cast<BlockingAudioCallbackState*>(user_data);
    if (!state) {
        return;
    }

    std::unique_lock<std::mutex> lock(state->mutex);
    state->entered = true;
    state->call_count++;
    state->sample_count = sample_count;
    ALIA_EXPECT_TRUE(samples != nullptr);
    ALIA_EXPECT_TRUE(sample_count > 0);
    state->cv.notify_all();
    state->cv.wait(lock, [&]() { return state->release; });
}

void counting_audio_callback(const float* samples, int sample_count, void* user_data) {
    auto* state = static_cast<CountingAudioCallbackState*>(user_data);
    if (!state) {
        return;
    }

    ALIA_EXPECT_TRUE(samples != nullptr);
    ALIA_EXPECT_TRUE(sample_count > 0);
    state->call_count++;
    state->total_samples += sample_count;
}

void recording_audio_callback(const float* samples, int sample_count, void* user_data) {
    auto* state = static_cast<RecordingAudioCallbackState*>(user_data);
    if (!state) {
        return;
    }

    ALIA_EXPECT_TRUE(samples != nullptr);
    ALIA_EXPECT_TRUE(sample_count > 0);
    state->call_count++;
    state->samples.insert(state->samples.end(), samples, samples + sample_count);
}

void blocking_background_callback(const char* extracted_json, void* user_data) {
    auto* state = g_background_callback_state;
    if (!state) {
        return;
    }

    std::unique_lock<std::mutex> lock(state->mutex);
    state->entered = true;
    state->call_count++;
    state->result_json = extracted_json ? extracted_json : "";
    state->user_data = user_data;
    state->cv.notify_all();
    state->cv.wait(lock, [&]() { return state->release; });
}

void recording_background_callback(const char* extracted_json, void* user_data) {
    auto* state = g_recording_background_callback_state;
    if (!state) {
        return;
    }
    state->call_count++;
    state->result_json = extracted_json ? extracted_json : "";
    state->user_data = user_data;
}

int recording_tool_callback(const char* tool_json,
                            char* out_result_buf,
                            int max_result_len,
                            void* user_data) {
    auto* state = static_cast<ToolCallbackState*>(user_data);
    if (!state) {
        return 1;
    }

    state->call_count++;
    state->tool_json = tool_json ? tool_json : "";
    if (!out_result_buf || max_result_len <= 0) {
        return 1;
    }

    const std::string& result = state->result_to_write;
    const int copy_len = std::min(static_cast<int>(result.size()), max_result_len - 1);
    std::memcpy(out_result_buf, result.data(), static_cast<size_t>(copy_len));
    out_result_buf[copy_len] = '\0';
    return 0;
}

void test_context_init_rejects_null_out_pointer() {
    int rc = alia_context_init(nullptr, "", "", "", "", 2048);
    ALIA_EXPECT_EQ(rc, ALIA_ERR_INVALID_ARGUMENT);
}

void test_context_init_and_destroy_allocates_handle_without_models() {
    AliaContext* ctx = nullptr;
    int rc = alia_context_init(&ctx, "", "", "", "", 2048);

    ALIA_EXPECT_EQ(rc, ALIA_OK);
    ALIA_EXPECT_TRUE(ctx != nullptr);

    alia_context_destroy(ctx);
}

void test_context_init_rejects_non_positive_sequence_length() {
    AliaContext* ctx = reinterpret_cast<AliaContext*>(0x1);
    int rc = alia_context_init(&ctx, "", "", "", "", 0);

    ALIA_EXPECT_EQ(rc, ALIA_ERR_INVALID_ARGUMENT);
    ALIA_EXPECT_TRUE(ctx == nullptr);
}

void test_runtime_functions_reject_null_context() {
    float sample = 0.0f;
    char* stable = reinterpret_cast<char*>(0x1);
    char* partial = reinterpret_cast<char*>(0x1);

    ALIA_EXPECT_EQ(alia_abort_inference(nullptr, ALIA_PIPELINE_ALL), ALIA_ERR_INVALID_ARGUMENT);
    ALIA_EXPECT_EQ(alia_vlm_rollback_kv_cache(nullptr, 1), ALIA_ERR_INVALID_ARGUMENT);
    ALIA_EXPECT_EQ(alia_asr_feed_audio(nullptr, &sample, 1), ALIA_ERR_INVALID_ARGUMENT);
    ALIA_EXPECT_EQ(alia_asr_get_text(nullptr, &stable, &partial), ALIA_ERR_INVALID_ARGUMENT);
    ALIA_EXPECT_EQ(alia_trigger_background_processing(nullptr, "turn"), ALIA_ERR_INVALID_ARGUMENT);
    ALIA_EXPECT_EQ(alia_start_conversation_turn(nullptr, nullptr, nullptr, nullptr, nullptr),
                   ALIA_ERR_INVALID_ARGUMENT);

    alia_asr_reset(nullptr);
    alia_register_background_callback(nullptr, nullptr);
}

void test_asr_feed_rejects_bad_audio_arguments() {
    AliaContext* ctx = nullptr;
    int init_rc = alia_context_init(&ctx, "", "", "", "", 2048);
    ALIA_EXPECT_EQ(init_rc, ALIA_OK);
    ALIA_EXPECT_TRUE(ctx != nullptr);

    float sample = 0.0f;
    ALIA_EXPECT_EQ(alia_asr_feed_audio(ctx, nullptr, 1), ALIA_ERR_INVALID_ARGUMENT);
    ALIA_EXPECT_EQ(alia_asr_feed_audio(ctx, &sample, 0), ALIA_ERR_INVALID_ARGUMENT);

    alia_context_destroy(ctx);
}

void test_asr_pipeline_owns_feed_and_reset_state() {
    AliaContext* ctx = nullptr;
    int init_rc = alia_context_init(&ctx, "", "", "", "", 2048);
    ALIA_EXPECT_EQ(init_rc, ALIA_OK);
    ALIA_EXPECT_TRUE(ctx != nullptr);
    ALIA_EXPECT_TRUE(ctx->asr_pipeline != nullptr);

    float samples[3] = {0.1f, -0.1f, 0.0f};
    ALIA_EXPECT_EQ(alia_asr_feed_audio(ctx, samples, 3), ALIA_OK);
    ALIA_EXPECT_EQ(ctx->asr_pipeline->buffered_sample_count(), static_cast<size_t>(3));

    alia_asr_reset(ctx);
    ALIA_EXPECT_EQ(ctx->asr_pipeline->buffered_sample_count(), static_cast<size_t>(0));

    alia_context_destroy(ctx);
}

void test_asr_pipeline_reports_readiness_from_loaded_slot() {
    AliaContext* ctx = nullptr;
    int init_rc = alia_context_init(&ctx, "", "", "", "", 2048);
    ALIA_EXPECT_EQ(init_rc, ALIA_OK);
    ALIA_EXPECT_TRUE(ctx != nullptr);
    ALIA_EXPECT_TRUE(ctx->asr_pipeline != nullptr);
    ALIA_EXPECT_TRUE(!ctx->asr_pipeline->ready());

    alia_context_destroy(ctx);
}

void test_asr_pipeline_process_pending_is_safe_without_loaded_model() {
    AliaContext* ctx = nullptr;
    int init_rc = alia_context_init(&ctx, "", "", "", "", 2048);
    ALIA_EXPECT_EQ(init_rc, ALIA_OK);
    ALIA_EXPECT_TRUE(ctx != nullptr);

    float samples[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    ALIA_EXPECT_EQ(alia_asr_feed_audio(ctx, samples, 4), ALIA_OK);
    ALIA_EXPECT_TRUE(!ctx->asr_pipeline->process_pending());
    ALIA_EXPECT_EQ(ctx->asr_pipeline->buffered_sample_count(), static_cast<size_t>(4));

    alia_context_destroy(ctx);
}

void test_asr_output_parser_extracts_language_and_text() {
    std::string language;
    std::string text;
    aila::alia::parse_asr_output("language Chinese\n<asr_text>hello Alia", "", language, text);

    ALIA_EXPECT_TRUE(language == "Chinese");
    ALIA_EXPECT_TRUE(text == "hello Alia");

    aila::alia::parse_asr_output("plain transcript", "japanese", language, text);
    ALIA_EXPECT_TRUE(language == "Japanese");
    ALIA_EXPECT_TRUE(text == "plain transcript");
}

void test_background_trigger_requires_registered_callback() {
    AliaContext* ctx = nullptr;
    int init_rc = alia_context_init(&ctx, "", "", "", "", 2048);
    ALIA_EXPECT_EQ(init_rc, ALIA_OK);
    ALIA_EXPECT_TRUE(ctx != nullptr);

    alia_register_background_callback(ctx, nullptr);
    ALIA_EXPECT_EQ(alia_trigger_background_processing(ctx, "hello"), ALIA_ERR_INVALID_STATE);

    alia_context_destroy(ctx);
}

void test_background_system_prompt_is_alia_json_extraction_prompt() {
    const std::string prompt = aila::alia::background_system_prompt();

    ALIA_EXPECT_TRUE(prompt.find("Alia") != std::string::npos);
    ALIA_EXPECT_TRUE(prompt.find("JSON") != std::string::npos);
    ALIA_EXPECT_TRUE(prompt.find("memory") != std::string::npos);
    ALIA_EXPECT_TRUE(prompt.find("conversation") != std::string::npos);
}

void test_background_schema_accepts_valid_memory_result() {
    const std::string valid =
        R"({"summary":"user likes concise replies","memory_candidates":[],"preferences":[],"tasks":[]})";

    const std::string enforced = aila::alia::enforce_background_result_schema(
        valid, "chat turn");

    ALIA_EXPECT_TRUE(enforced == valid);
}

void test_background_schema_repairs_malformed_json_with_required_key_names() {
    const std::string malformed =
        R"({"summary":"partial","memory_candidates":[],"preferences":[],"tasks":[])";

    const std::string enforced = aila::alia::enforce_background_result_schema(
        malformed, "fallback summary");

    ALIA_EXPECT_TRUE(enforced.find("\"summary\":\"fallback summary\"") != std::string::npos);
    ALIA_EXPECT_TRUE(enforced.find("\"memory_candidates\":[]") != std::string::npos);
    ALIA_EXPECT_TRUE(enforced.find("\"preferences\":[]") != std::string::npos);
    ALIA_EXPECT_TRUE(enforced.find("\"tasks\":[]") != std::string::npos);
    ALIA_EXPECT_TRUE(enforced.find("\"raw_model_output\"") != std::string::npos);
    ALIA_EXPECT_TRUE(enforced.find("\"source\":\"schema_repair\"") != std::string::npos);
}

void test_background_schema_repairs_wrong_required_field_types() {
    const std::string wrong_types =
        R"({"summary":42,"memory_candidates":"oops","preferences":{},"tasks":null})";

    const std::string enforced = aila::alia::enforce_background_result_schema(
        wrong_types, "typed fallback");

    ALIA_EXPECT_TRUE(enforced.find("\"summary\":\"typed fallback\"") != std::string::npos);
    ALIA_EXPECT_TRUE(enforced.find("\"memory_candidates\":[]") != std::string::npos);
    ALIA_EXPECT_TRUE(enforced.find("\"preferences\":[]") != std::string::npos);
    ALIA_EXPECT_TRUE(enforced.find("\"tasks\":[]") != std::string::npos);
    ALIA_EXPECT_TRUE(enforced.find("\"raw_model_output\"") != std::string::npos);
    ALIA_EXPECT_TRUE(enforced.find("\"source\":\"schema_repair\"") != std::string::npos);
}

void test_background_pipeline_invokes_callback_and_rejects_busy_trigger() {
    AliaContext* ctx = nullptr;
    int init_rc = alia_context_init(&ctx, "", "", "", "", 2048);
    ALIA_EXPECT_EQ(init_rc, ALIA_OK);
    ALIA_EXPECT_TRUE(ctx != nullptr);
    ALIA_EXPECT_TRUE(ctx->background_pipeline != nullptr);

    BlockingBackgroundCallbackState callback_state;
    g_background_callback_state = &callback_state;
    alia_register_background_callback(ctx, blocking_background_callback);
    ALIA_EXPECT_EQ(alia_trigger_background_processing(ctx, "turn text"), ALIA_OK);

    {
        std::unique_lock<std::mutex> lock(callback_state.mutex);
        ALIA_EXPECT_TRUE(callback_state.cv.wait_for(
            lock, std::chrono::seconds(2), [&]() { return callback_state.entered; }));
    }

    ALIA_EXPECT_EQ(alia_trigger_background_processing(ctx, "second turn"),
                   ALIA_ERR_INVALID_STATE);

    {
        std::lock_guard<std::mutex> lock(callback_state.mutex);
        callback_state.release = true;
    }
    callback_state.cv.notify_all();
    ALIA_EXPECT_TRUE(ctx->background_pipeline->wait_until_idle_for(std::chrono::seconds(2)));
    ALIA_EXPECT_EQ(callback_state.call_count, 1);
    ALIA_EXPECT_TRUE(callback_state.result_json.find("turn text") != std::string::npos);
    ALIA_EXPECT_TRUE(callback_state.result_json.find("alia_background_stub") ==
                     std::string::npos);
    ALIA_EXPECT_TRUE(callback_state.result_json.find("\"summary\"") != std::string::npos);
    ALIA_EXPECT_TRUE(callback_state.result_json.find("\"memory_candidates\"") !=
                     std::string::npos);
    ALIA_EXPECT_TRUE(callback_state.result_json.find("\"preferences\"") !=
                     std::string::npos);
    ALIA_EXPECT_TRUE(callback_state.result_json.find("\"tasks\"") != std::string::npos);
    ALIA_EXPECT_TRUE(ctx->background_pipeline->last_decode_mode() ==
                     aila::alia::BackgroundDecodeMode::NoModelFallback);
    ALIA_EXPECT_TRUE(ctx->background_pipeline->last_prompt_text().find("turn text") !=
                     std::string::npos);
    ALIA_EXPECT_TRUE(ctx->background_pipeline->last_prompt_text().find("JSON") !=
                     std::string::npos);
    ALIA_EXPECT_TRUE(callback_state.user_data == nullptr);

    g_background_callback_state = nullptr;
    alia_context_destroy(ctx);
}

void test_background_loaded_vlm_retries_invalid_json_before_schema_repair() {
    AliaContext* ctx = nullptr;
    int init_rc = alia_context_init(&ctx, "", "", "", "", 2048);
    ALIA_EXPECT_EQ(init_rc, ALIA_OK);
    ALIA_EXPECT_TRUE(ctx != nullptr);
    ALIA_EXPECT_TRUE(ctx->runtime != nullptr);
    ALIA_EXPECT_TRUE(ctx->background_pipeline != nullptr);

    const std::string invalid_json =
        R"({"summary":42,"memory_candidates":"bad","preferences":[],"tasks":[]})";
    const std::string valid_json =
        R"({"summary":"learned preference","memory_candidates":[],"preferences":[],"tasks":[]})";
    std::vector<std::string> normal_tokens{invalid_json, valid_json, "unused"};
    const int eos_token_id = static_cast<int>(normal_tokens.size()) + 3;
    std::filesystem::path tokenizer_dir =
        make_temp_tokenizer_dir("background_retry", normal_tokens);

    auto tokenizer = std::make_unique<Tokenizer>();
    ALIA_EXPECT_TRUE(tokenizer->load(tokenizer_dir.string()));
    auto backend = std::make_unique<SessionCountingBackend>(
        ctx->runtime->background(),
        tokenizer->vocab_size(),
        std::vector<std::vector<int>>{{0, eos_token_id}, {1, eos_token_id}});
    SessionCountingBackend* backend_observer = backend.get();
    ctx->background_vlm.configure_loaded_for_tests(
        aila::alia::ModelRole::BackgroundVlm,
        &ctx->runtime->background(),
        std::move(tokenizer),
        std::move(backend),
        aila::alia::BackendKind::Qwen35HybridBnb4);

    RecordingBackgroundCallbackState callback_state;
    g_recording_background_callback_state = &callback_state;
    alia_register_background_callback(ctx, recording_background_callback);
    ALIA_EXPECT_EQ(alia_trigger_background_processing(ctx, "user likes concise answers"), ALIA_OK);
    ALIA_EXPECT_TRUE(ctx->background_pipeline->wait_until_idle_for(std::chrono::seconds(2)));

    ALIA_EXPECT_EQ(callback_state.call_count, 1);
    ALIA_EXPECT_EQ(backend_observer->prefill_count(), 2);
    ALIA_EXPECT_TRUE(callback_state.result_json.find("\"summary\":\"learned preference\"") !=
                     std::string::npos);
    ALIA_EXPECT_TRUE(callback_state.result_json.find("\"source\":\"schema_repair\"") ==
                     std::string::npos);
    ALIA_EXPECT_TRUE(ctx->background_pipeline->last_prompt_text().find("Repair") !=
                     std::string::npos);
    ALIA_EXPECT_EQ(ctx->background_pipeline->last_schema_retry_count(), 1);
    ALIA_EXPECT_TRUE(!ctx->background_pipeline->last_schema_repair_applied());
    ALIA_EXPECT_TRUE(ctx->background_pipeline->last_schema_diagnostic().find("retry accepted") !=
                     std::string::npos);

    g_recording_background_callback_state = nullptr;
    alia_context_destroy(ctx);
    remove_temp_dir(tokenizer_dir);
}

void test_background_loaded_vlm_reports_schema_repair_after_retry_failure() {
    AliaContext* ctx = nullptr;
    int init_rc = alia_context_init(&ctx, "", "", "", "", 2048);
    ALIA_EXPECT_EQ(init_rc, ALIA_OK);
    ALIA_EXPECT_TRUE(ctx != nullptr);
    ALIA_EXPECT_TRUE(ctx->runtime != nullptr);
    ALIA_EXPECT_TRUE(ctx->background_pipeline != nullptr);

    const std::string first_invalid =
        R"({"summary":42,"memory_candidates":"bad","preferences":[],"tasks":[]})";
    const std::string second_invalid =
        R"({"summary":"still bad","memory_candidates":[],"preferences":{},"tasks":[]})";
    std::vector<std::string> normal_tokens{first_invalid, second_invalid, "unused"};
    const int eos_token_id = static_cast<int>(normal_tokens.size()) + 3;
    std::filesystem::path tokenizer_dir =
        make_temp_tokenizer_dir("background_retry_repair", normal_tokens);

    auto tokenizer = std::make_unique<Tokenizer>();
    ALIA_EXPECT_TRUE(tokenizer->load(tokenizer_dir.string()));
    auto backend = std::make_unique<SessionCountingBackend>(
        ctx->runtime->background(),
        tokenizer->vocab_size(),
        std::vector<std::vector<int>>{{0, eos_token_id}, {1, eos_token_id}});
    ctx->background_vlm.configure_loaded_for_tests(
        aila::alia::ModelRole::BackgroundVlm,
        &ctx->runtime->background(),
        std::move(tokenizer),
        std::move(backend),
        aila::alia::BackendKind::Qwen35HybridBnb4);

    RecordingBackgroundCallbackState callback_state;
    g_recording_background_callback_state = &callback_state;
    alia_register_background_callback(ctx, recording_background_callback);
    ALIA_EXPECT_EQ(alia_trigger_background_processing(ctx, "user asked for broken schema"), ALIA_OK);
    ALIA_EXPECT_TRUE(ctx->background_pipeline->wait_until_idle_for(std::chrono::seconds(2)));

    ALIA_EXPECT_EQ(callback_state.call_count, 1);
    ALIA_EXPECT_TRUE(callback_state.result_json.find("\"source\":\"schema_repair\"") !=
                     std::string::npos);
    ALIA_EXPECT_EQ(ctx->background_pipeline->last_schema_retry_count(), 1);
    ALIA_EXPECT_TRUE(ctx->background_pipeline->last_schema_repair_applied());
    ALIA_EXPECT_TRUE(ctx->background_pipeline->last_schema_diagnostic().find("retry failed") !=
                     std::string::npos);

    g_recording_background_callback_state = nullptr;
    alia_context_destroy(ctx);
    remove_temp_dir(tokenizer_dir);
}

void test_foreground_pipeline_rejects_second_turn_while_worker_is_running() {
    AliaContext* ctx = nullptr;
    int init_rc = alia_context_init(&ctx, "", "", "", "", 2048);
    ALIA_EXPECT_EQ(init_rc, ALIA_OK);
    ALIA_EXPECT_TRUE(ctx != nullptr);
    ALIA_EXPECT_TRUE(ctx->foreground_pipeline != nullptr);

    BlockingAudioCallbackState callback_state;
    AliaGenConfig config{0.2f, 0.9f, 8};
    int start_rc = alia_start_conversation_turn(
        ctx, &config, nullptr, blocking_audio_callback, &callback_state);
    ALIA_EXPECT_EQ(start_rc, ALIA_OK);

    {
        std::unique_lock<std::mutex> lock(callback_state.mutex);
        ALIA_EXPECT_TRUE(callback_state.cv.wait_for(
            lock, std::chrono::seconds(2), [&]() { return callback_state.entered; }));
    }

    ALIA_EXPECT_EQ(alia_start_conversation_turn(
                       ctx, &config, nullptr, blocking_audio_callback, &callback_state),
                   ALIA_ERR_INVALID_STATE);

    {
        std::lock_guard<std::mutex> lock(callback_state.mutex);
        callback_state.release = true;
    }
    callback_state.cv.notify_all();
    ALIA_EXPECT_TRUE(ctx->foreground_pipeline->wait_until_idle_for(std::chrono::seconds(2)));
    ALIA_EXPECT_EQ(callback_state.call_count, 1);
    ALIA_EXPECT_TRUE(callback_state.sample_count > 0);

    alia_context_destroy(ctx);
}

void test_foreground_turn_rejects_invalid_generation_config() {
    AliaContext* ctx = nullptr;
    int init_rc = alia_context_init(&ctx, "", "", "", "", 2048);
    ALIA_EXPECT_EQ(init_rc, ALIA_OK);
    ALIA_EXPECT_TRUE(ctx != nullptr);

    AliaGenConfig zero_tokens{0.2f, 0.9f, 0};
    ALIA_EXPECT_EQ(alia_start_conversation_turn(ctx, &zero_tokens, nullptr, nullptr, nullptr),
                   ALIA_ERR_INVALID_ARGUMENT);
    ALIA_EXPECT_EQ(ctx->foreground_pipeline->state(), aila::alia::ForegroundTurnState::Idle);

    AliaGenConfig bad_top_p{0.2f, 1.5f, 8};
    ALIA_EXPECT_EQ(alia_start_conversation_turn(ctx, &bad_top_p, nullptr, nullptr, nullptr),
                   ALIA_ERR_INVALID_ARGUMENT);
    ALIA_EXPECT_EQ(ctx->foreground_pipeline->state(), aila::alia::ForegroundTurnState::Idle);

    alia_context_destroy(ctx);
}

void test_vlm_rollback_requires_loaded_foreground_anchor() {
    AliaContext* ctx = nullptr;
    int init_rc = alia_context_init(&ctx, "", "", "", "", 2048);
    ALIA_EXPECT_EQ(init_rc, ALIA_OK);
    ALIA_EXPECT_TRUE(ctx != nullptr);

    ALIA_EXPECT_EQ(alia_vlm_rollback_kv_cache(ctx, 0), ALIA_OK);
    ALIA_EXPECT_EQ(alia_vlm_rollback_kv_cache(ctx, 1), ALIA_ERR_INVALID_STATE);

    alia_context_destroy(ctx);
}

void test_foreground_turn_captures_asr_text_and_generation_config() {
    AliaContext* ctx = nullptr;
    int init_rc = alia_context_init(&ctx, "", "", "", "", 2048);
    ALIA_EXPECT_EQ(init_rc, ALIA_OK);
    ALIA_EXPECT_TRUE(ctx != nullptr);
    ALIA_EXPECT_TRUE(ctx->asr_pipeline != nullptr);
    ALIA_EXPECT_TRUE(ctx->foreground_pipeline != nullptr);

    ctx->asr_pipeline->append_stable_text("please summarize the active window");

    BlockingAudioCallbackState callback_state;
    AliaGenConfig config{0.35f, 0.72f, 33};
    ALIA_EXPECT_EQ(alia_start_conversation_turn(
                       ctx, &config, nullptr, blocking_audio_callback, &callback_state),
                   ALIA_OK);

    {
        std::unique_lock<std::mutex> lock(callback_state.mutex);
        ALIA_EXPECT_TRUE(callback_state.cv.wait_for(
            lock, std::chrono::seconds(2), [&]() { return callback_state.entered; }));
    }

    const GenerationConfig translated = ctx->foreground_pipeline->last_generation_config();
    ALIA_EXPECT_EQ(translated.max_new_tokens, 33);
    ALIA_EXPECT_TRUE(translated.temperature == 0.35f);
    ALIA_EXPECT_TRUE(translated.top_p == 0.72f);
    ALIA_EXPECT_TRUE(translated.do_sample);
    ALIA_EXPECT_TRUE(ctx->foreground_pipeline->last_user_text() ==
                     "please summarize the active window");
    ALIA_EXPECT_TRUE(ctx->foreground_pipeline->last_decode_mode() ==
                     aila::alia::ForegroundDecodeMode::NoModelFallback);
    ALIA_EXPECT_TRUE(ctx->foreground_pipeline->last_assistant_text() ==
                     "please summarize the active window");

    {
        std::lock_guard<std::mutex> lock(callback_state.mutex);
        callback_state.release = true;
    }
    callback_state.cv.notify_all();
    ALIA_EXPECT_TRUE(ctx->foreground_pipeline->wait_until_idle_for(std::chrono::seconds(2)));

    alia_context_destroy(ctx);
}

void test_foreground_system_prompt_is_alia_specific() {
    const std::string prompt = aila::alia::foreground_system_prompt();

    ALIA_EXPECT_TRUE(prompt.find("Alia") != std::string::npos);
    ALIA_EXPECT_TRUE(prompt.find("local companion") != std::string::npos);
    ALIA_EXPECT_TRUE(prompt.find("tool call") != std::string::npos);
    ALIA_EXPECT_TRUE(prompt.find("generic") == std::string::npos);
}

void test_foreground_tool_call_invokes_callback_and_keeps_spoken_text_native() {
    AliaContext* ctx = nullptr;
    int init_rc = alia_context_init(&ctx, "", "", "", "", 2048);
    ALIA_EXPECT_EQ(init_rc, ALIA_OK);
    ALIA_EXPECT_TRUE(ctx != nullptr);
    ALIA_EXPECT_TRUE(ctx->asr_pipeline != nullptr);
    ALIA_EXPECT_TRUE(ctx->foreground_pipeline != nullptr);

    ctx->asr_pipeline->append_stable_text(
        "Let me check."
        "<tool_call>\n"
        "<function=inspect_window>\n"
        "<parameter=target>active</parameter>\n"
        "</function>\n"
        "</tool_call>"
        "Done.");

    ToolCallbackState tool_state;
    AliaGenConfig config{0.0f, 1.0f, 16};
    ALIA_EXPECT_EQ(alia_start_conversation_turn(
                       ctx, &config, recording_tool_callback, nullptr, &tool_state),
                   ALIA_OK);
    ALIA_EXPECT_TRUE(ctx->foreground_pipeline->wait_until_idle_for(std::chrono::seconds(2)));

    ALIA_EXPECT_EQ(tool_state.call_count, 1);
    ALIA_EXPECT_TRUE(tool_state.tool_json.find("inspect_window") != std::string::npos);
    ALIA_EXPECT_TRUE(tool_state.tool_json.find("target") != std::string::npos);
    ALIA_EXPECT_TRUE(tool_state.tool_json.find("active") != std::string::npos);
    ALIA_EXPECT_TRUE(ctx->foreground_pipeline->last_tool_call_json().find("inspect_window") !=
                     std::string::npos);
    ALIA_EXPECT_TRUE(ctx->foreground_pipeline->last_tool_result_text() ==
                     tool_state.result_to_write);
    ALIA_EXPECT_TRUE(ctx->foreground_pipeline->last_assistant_text() ==
                     "Let me check.Done.");

    alia_context_destroy(ctx);
}

void test_foreground_tool_result_is_promoted_to_resume_prompt_state() {
    AliaContext* ctx = nullptr;
    int init_rc = alia_context_init(&ctx, "", "", "", "", 2048);
    ALIA_EXPECT_EQ(init_rc, ALIA_OK);
    ALIA_EXPECT_TRUE(ctx != nullptr);
    ALIA_EXPECT_TRUE(ctx->asr_pipeline != nullptr);
    ALIA_EXPECT_TRUE(ctx->foreground_pipeline != nullptr);

    ctx->asr_pipeline->append_stable_text(
        "What is on my screen?"
        "<tool_call>\n"
        "<function=inspect_window>\n"
        "<parameter=target>active</parameter>\n"
        "</function>\n"
        "</tool_call>");

    ToolCallbackState tool_state;
    tool_state.result_to_write = R"({"ok":true,"title":"Settings","control":"Privacy"})";
    AliaGenConfig config{0.0f, 1.0f, 16};
    ALIA_EXPECT_EQ(alia_start_conversation_turn(
                       ctx, &config, recording_tool_callback, nullptr, &tool_state),
                   ALIA_OK);
    ALIA_EXPECT_TRUE(ctx->foreground_pipeline->wait_until_idle_for(std::chrono::seconds(2)));

    const std::string resume_prompt =
        ctx->foreground_pipeline->last_tool_resume_prompt_text();
    ALIA_EXPECT_TRUE(resume_prompt.find("What is on my screen?") != std::string::npos);
    ALIA_EXPECT_TRUE(resume_prompt.find("inspect_window") != std::string::npos);
    ALIA_EXPECT_TRUE(resume_prompt.find("Settings") != std::string::npos);
    ALIA_EXPECT_TRUE(resume_prompt.find("Continue the response") != std::string::npos);

    alia_context_destroy(ctx);
}

void test_foreground_tool_resume_appends_result_without_chat_scaffold() {
    AliaContext* ctx = nullptr;
    int init_rc = alia_context_init(&ctx, "", "", "", "", 2048);
    ALIA_EXPECT_EQ(init_rc, ALIA_OK);
    ALIA_EXPECT_TRUE(ctx != nullptr);
    ALIA_EXPECT_TRUE(ctx->runtime != nullptr);
    ALIA_EXPECT_TRUE(ctx->foreground_pipeline != nullptr);

    const std::string first_token =
        "Checking."
        "<tool_call>\n"
        "<function=inspect_window>\n"
        "<parameter=target>active</parameter>\n"
        "</function>\n"
        "</tool_call>";
    const std::string second_token = "The Settings window is open.";
    std::filesystem::path tokenizer_dir =
        make_temp_tokenizer_dir("foreground_loaded_resume",
                                {first_token, second_token, "unused"});

    auto tokenizer = std::make_unique<Tokenizer>();
    ALIA_EXPECT_TRUE(tokenizer->load(tokenizer_dir.string()));
    auto backend = std::make_unique<SessionCountingBackend>(ctx->runtime->foreground());
    SessionCountingBackend* backend_observer = backend.get();
    ctx->foreground_vlm.configure_loaded_for_tests(
        aila::alia::ModelRole::ForegroundVlm,
        &ctx->runtime->foreground(),
        std::move(tokenizer),
        std::move(backend),
        aila::alia::BackendKind::Qwen35HybridBnb4);

    ToolCallbackState tool_state;
    tool_state.result_to_write = R"({"ok":true,"title":"Settings"})";
    AliaGenConfig config{0.0f, 1.0f, 1};
    ALIA_EXPECT_EQ(alia_start_conversation_turn(
                       ctx, &config, recording_tool_callback, nullptr, &tool_state),
                   ALIA_OK);
    ALIA_EXPECT_TRUE(ctx->foreground_pipeline->wait_until_idle_for(std::chrono::seconds(2)));

    ALIA_EXPECT_EQ(tool_state.call_count, 1);
    ALIA_EXPECT_EQ(backend_observer->prefill_count(), 2);
    ALIA_EXPECT_EQ(backend_observer->reset_count(), 1);
    ALIA_EXPECT_EQ(backend_observer->prefill_length(1), 2);
    ALIA_EXPECT_TRUE(ctx->foreground_pipeline->last_decode_mode() ==
                     aila::alia::ForegroundDecodeMode::LoadedVlm);
    ALIA_EXPECT_TRUE(ctx->foreground_pipeline->last_assistant_text().find("Checking.") !=
                     std::string::npos);
    ALIA_EXPECT_TRUE(ctx->foreground_pipeline->last_assistant_text().find("Settings window") !=
                     std::string::npos);

    alia_context_destroy(ctx);
    remove_temp_dir(tokenizer_dir);
}

void test_foreground_loaded_vlm_pauses_initial_decode_on_complete_tool_call() {
    AliaContext* ctx = nullptr;
    int init_rc = alia_context_init(&ctx, "", "", "", "", 2048);
    ALIA_EXPECT_EQ(init_rc, ALIA_OK);
    ALIA_EXPECT_TRUE(ctx != nullptr);
    ALIA_EXPECT_TRUE(ctx->runtime != nullptr);
    ALIA_EXPECT_TRUE(ctx->foreground_pipeline != nullptr);

    const std::string spoken_before_tool = "Checking.";
    const std::string tool_call_text =
        "<tool_call>\n"
        "<function=inspect_window>\n"
        "<parameter=target>active</parameter>\n"
        "</function>\n"
        "</tool_call>";
    const std::string leaked_after_tool = "SHOULD_NOT_BE_INITIAL";
    const std::string resumed_text = "The Settings window is open.";
    std::vector<std::string> normal_tokens{
        spoken_before_tool,
        tool_call_text,
        leaked_after_tool,
        resumed_text,
        "unused"
    };
    const int eos_token_id = static_cast<int>(normal_tokens.size()) + 3;
    std::filesystem::path tokenizer_dir =
        make_temp_tokenizer_dir("foreground_tool_pause", normal_tokens);

    auto tokenizer = std::make_unique<Tokenizer>();
    ALIA_EXPECT_TRUE(tokenizer->load(tokenizer_dir.string()));
    const int vocab_size = tokenizer->vocab_size();
    auto backend = std::make_unique<SessionCountingBackend>(
        ctx->runtime->foreground(),
        vocab_size,
        std::vector<std::vector<int>>{{0, 1, 2}, {3, eos_token_id}});
    ctx->foreground_vlm.configure_loaded_for_tests(
        aila::alia::ModelRole::ForegroundVlm,
        &ctx->runtime->foreground(),
        std::move(tokenizer),
        std::move(backend),
        aila::alia::BackendKind::Qwen35HybridBnb4);

    ToolCallbackState tool_state;
    tool_state.result_to_write = R"({"ok":true,"title":"Settings"})";
    AliaGenConfig config{0.0f, 1.0f, 3};
    ALIA_EXPECT_EQ(alia_start_conversation_turn(
                       ctx, &config, recording_tool_callback, nullptr, &tool_state),
                   ALIA_OK);
    ALIA_EXPECT_TRUE(ctx->foreground_pipeline->wait_until_idle_for(std::chrono::seconds(2)));

    ALIA_EXPECT_EQ(tool_state.call_count, 1);
    ALIA_EXPECT_TRUE(ctx->foreground_pipeline->last_assistant_text().find(spoken_before_tool) !=
                     std::string::npos);
    ALIA_EXPECT_TRUE(ctx->foreground_pipeline->last_assistant_text().find(resumed_text) !=
                     std::string::npos);
    ALIA_EXPECT_TRUE(ctx->foreground_pipeline->last_assistant_text().find(leaked_after_tool) ==
                     std::string::npos);

    alia_context_destroy(ctx);
    remove_temp_dir(tokenizer_dir);
}

void test_foreground_loaded_vlm_streams_sentence_to_tts_before_decode_finishes() {
    AliaContext* ctx = nullptr;
    int init_rc = alia_context_init(&ctx, "", "", "", "", 2048);
    ALIA_EXPECT_EQ(init_rc, ALIA_OK);
    ALIA_EXPECT_TRUE(ctx != nullptr);
    ALIA_EXPECT_TRUE(ctx->runtime != nullptr);
    ALIA_EXPECT_TRUE(ctx->foreground_pipeline != nullptr);

    std::vector<std::string> normal_tokens{
        "First sentence.",
        "Second sentence.",
        "unused"
    };
    const int eos_token_id = static_cast<int>(normal_tokens.size()) + 3;
    std::filesystem::path tokenizer_dir =
        make_temp_tokenizer_dir("foreground_token_tts", normal_tokens);

    auto tokenizer = std::make_unique<Tokenizer>();
    ALIA_EXPECT_TRUE(tokenizer->load(tokenizer_dir.string()));
    const int vocab_size = tokenizer->vocab_size();
    auto backend = std::make_unique<SessionCountingBackend>(
        ctx->runtime->foreground(),
        vocab_size,
        std::vector<std::vector<int>>{{0, 1, eos_token_id}});
    SessionCountingBackend* backend_observer = backend.get();
    ctx->foreground_vlm.configure_loaded_for_tests(
        aila::alia::ModelRole::ForegroundVlm,
        &ctx->runtime->foreground(),
        std::move(tokenizer),
        std::move(backend),
        aila::alia::BackendKind::Qwen35HybridBnb4);

    BlockingAudioCallbackState callback_state;
    AliaGenConfig config{0.0f, 1.0f, 3};
    ALIA_EXPECT_EQ(alia_start_conversation_turn(
                       ctx, &config, nullptr, blocking_audio_callback, &callback_state),
                   ALIA_OK);

    {
        std::unique_lock<std::mutex> lock(callback_state.mutex);
        ALIA_EXPECT_TRUE(callback_state.cv.wait_for(
            lock, std::chrono::seconds(2), [&]() { return callback_state.entered; }));
    }

    ALIA_EXPECT_EQ(backend_observer->get_current_context_len(),
                   backend_observer->prefill_length(0) + 1);

    {
        std::lock_guard<std::mutex> lock(callback_state.mutex);
        callback_state.release = true;
    }
    callback_state.cv.notify_all();

    ALIA_EXPECT_TRUE(ctx->foreground_pipeline->wait_until_idle_for(std::chrono::seconds(2)));
    ALIA_EXPECT_EQ(callback_state.call_count, 2);
    ALIA_EXPECT_TRUE(ctx->foreground_pipeline->last_assistant_text().find("First sentence.") !=
                     std::string::npos);
    ALIA_EXPECT_TRUE(ctx->foreground_pipeline->last_assistant_text().find("Second sentence.") !=
                     std::string::npos);

    alia_context_destroy(ctx);
    remove_temp_dir(tokenizer_dir);
}

void test_foreground_spoken_text_is_streamed_to_tts_as_chunks() {
    AliaContext* ctx = nullptr;
    int init_rc = alia_context_init(&ctx, "", "", "", "", 2048);
    ALIA_EXPECT_EQ(init_rc, ALIA_OK);
    ALIA_EXPECT_TRUE(ctx != nullptr);
    ALIA_EXPECT_TRUE(ctx->asr_pipeline != nullptr);
    ALIA_EXPECT_TRUE(ctx->foreground_pipeline != nullptr);

    ctx->asr_pipeline->append_stable_text("First sentence. Second sentence.");

    CountingAudioCallbackState audio_state;
    AliaGenConfig config{0.0f, 1.0f, 16};
    ALIA_EXPECT_EQ(alia_start_conversation_turn(
                       ctx, &config, nullptr, counting_audio_callback, &audio_state),
                   ALIA_OK);
    ALIA_EXPECT_TRUE(ctx->foreground_pipeline->wait_until_idle_for(std::chrono::seconds(2)));

    ALIA_EXPECT_EQ(audio_state.call_count, 2);
    ALIA_EXPECT_TRUE(audio_state.total_samples > 0);
    ALIA_EXPECT_EQ(ctx->tts_pipeline->pending_text_count(), static_cast<size_t>(0));

    alia_context_destroy(ctx);
}

void test_foreground_abort_during_tts_chunk_stops_remaining_audio() {
    AliaContext* ctx = nullptr;
    int init_rc = alia_context_init(&ctx, "", "", "", "", 2048);
    ALIA_EXPECT_EQ(init_rc, ALIA_OK);
    ALIA_EXPECT_TRUE(ctx != nullptr);
    ALIA_EXPECT_TRUE(ctx->asr_pipeline != nullptr);
    ALIA_EXPECT_TRUE(ctx->foreground_pipeline != nullptr);

    ctx->asr_pipeline->append_stable_text("First sentence. Second sentence.");

    BlockingAudioCallbackState callback_state;
    AliaGenConfig config{0.0f, 1.0f, 16};
    ALIA_EXPECT_EQ(alia_start_conversation_turn(
                       ctx, &config, nullptr, blocking_audio_callback, &callback_state),
                   ALIA_OK);

    {
        std::unique_lock<std::mutex> lock(callback_state.mutex);
        ALIA_EXPECT_TRUE(callback_state.cv.wait_for(
            lock, std::chrono::seconds(2), [&]() { return callback_state.entered; }));
    }

    ALIA_EXPECT_EQ(alia_abort_inference(ctx, ALIA_PIPELINE_ALL), ALIA_OK);

    {
        std::lock_guard<std::mutex> lock(callback_state.mutex);
        callback_state.release = true;
    }
    callback_state.cv.notify_all();

    ALIA_EXPECT_TRUE(ctx->foreground_pipeline->wait_until_idle_for(std::chrono::seconds(2)));
    ALIA_EXPECT_EQ(callback_state.call_count, 1);
    ALIA_EXPECT_EQ(ctx->foreground_pipeline->state(), aila::alia::ForegroundTurnState::Aborted);

    alia_context_destroy(ctx);
}

void test_tts_pipeline_owns_text_queue_and_audio_callback() {
    AliaContext* ctx = nullptr;
    int init_rc = alia_context_init(&ctx, "", "", "", "", 2048);
    ALIA_EXPECT_EQ(init_rc, ALIA_OK);
    ALIA_EXPECT_TRUE(ctx != nullptr);
    ALIA_EXPECT_TRUE(ctx->tts_pipeline != nullptr);
    ALIA_EXPECT_TRUE(!ctx->tts_pipeline->ready());

    ALIA_EXPECT_TRUE(ctx->tts_pipeline->enqueue_text("hello from foreground"));
    ALIA_EXPECT_EQ(ctx->tts_pipeline->pending_text_count(), static_cast<size_t>(1));

    BlockingAudioCallbackState callback_state;
    {
        std::lock_guard<std::mutex> lock(callback_state.mutex);
        callback_state.release = true;
    }
    AliaGenConfig config{0.2f, 0.9f, 8};
    ALIA_EXPECT_TRUE(ctx->tts_pipeline->synthesize_pending(
        config, blocking_audio_callback, &callback_state));
    ALIA_EXPECT_EQ(ctx->tts_pipeline->pending_text_count(), static_cast<size_t>(0));
    ALIA_EXPECT_EQ(callback_state.call_count, 1);
    ALIA_EXPECT_TRUE(callback_state.sample_count > 0);

    alia_context_destroy(ctx);
}

void test_tts_pipeline_uses_loaded_backend_streaming_path() {
    AliaContext* ctx = nullptr;
    int init_rc = alia_context_init(&ctx, "", "", "", "", 2048);
    ALIA_EXPECT_EQ(init_rc, ALIA_OK);
    ALIA_EXPECT_TRUE(ctx != nullptr);
    ALIA_EXPECT_TRUE(ctx->runtime != nullptr);
    ALIA_EXPECT_TRUE(ctx->tts_pipeline != nullptr);

    std::vector<std::string> normal_tokens{
        "a", "s", "i", "t", "n", "h", "e", "l", "o", "f", "r", "m"
    };
    std::filesystem::path tokenizer_dir =
        make_temp_tokenizer_dir("tts_backend_streaming", normal_tokens);

    auto tokenizer = std::make_unique<Tokenizer>();
    ALIA_EXPECT_TRUE(tokenizer->load(tokenizer_dir.string()));

    auto backend = std::make_unique<StreamingTtsBackend>(ctx->runtime->foreground());
    StreamingTtsBackend* backend_observer = backend.get();
    ctx->tts.configure_loaded_for_tests(
        aila::alia::ModelRole::Tts,
        &ctx->runtime->foreground(),
        std::move(tokenizer),
        std::move(backend),
        aila::alia::BackendKind::Qwen3Tts);

    ALIA_EXPECT_TRUE(ctx->tts_pipeline->ready());
    ALIA_EXPECT_TRUE(ctx->tts_pipeline->enqueue_text("hello"));

    RecordingAudioCallbackState audio_state;
    AliaGenConfig config{0.3f, 0.8f, 12};
    ALIA_EXPECT_TRUE(ctx->tts_pipeline->synthesize_pending(
        config, recording_audio_callback, &audio_state));

    ALIA_EXPECT_EQ(backend_observer->stream_call_count(), 1);
    ALIA_EXPECT_EQ(backend_observer->last_config().max_new_tokens, 12);
    ALIA_EXPECT_EQ(backend_observer->last_stream_batch_frames(), 6);
    ALIA_EXPECT_TRUE(!backend_observer->last_text_tokens().empty());
    ALIA_EXPECT_EQ(audio_state.call_count, 2);
    ALIA_EXPECT_EQ(audio_state.samples.size(), static_cast<size_t>(3));
    ALIA_EXPECT_TRUE(audio_state.samples[0] == 0.25f);
    ALIA_EXPECT_TRUE(audio_state.samples[1] == -0.5f);
    ALIA_EXPECT_TRUE(audio_state.samples[2] == 0.75f);
    ALIA_EXPECT_EQ(ctx->tts_pipeline->pending_text_count(), static_cast<size_t>(0));

    alia_context_destroy(ctx);
    remove_temp_dir(tokenizer_dir);
}

void test_foreground_abort_cancels_loaded_tts_backend_stream() {
    AliaContext* ctx = nullptr;
    int init_rc = alia_context_init(&ctx, "", "", "", "", 2048);
    ALIA_EXPECT_EQ(init_rc, ALIA_OK);
    ALIA_EXPECT_TRUE(ctx != nullptr);
    ALIA_EXPECT_TRUE(ctx->runtime != nullptr);
    ALIA_EXPECT_TRUE(ctx->foreground_pipeline != nullptr);
    ALIA_EXPECT_TRUE(ctx->tts_pipeline != nullptr);

    std::vector<std::string> foreground_tokens{"hello.", "unused"};
    const int eos_token_id = static_cast<int>(foreground_tokens.size()) + 3;
    std::filesystem::path foreground_tokenizer_dir =
        make_temp_tokenizer_dir("foreground_abort_tts_cancel", foreground_tokens);
    auto foreground_tokenizer = std::make_unique<Tokenizer>();
    ALIA_EXPECT_TRUE(foreground_tokenizer->load(foreground_tokenizer_dir.string()));

    const int foreground_vocab_size = foreground_tokenizer->vocab_size();
    auto foreground_backend = std::make_unique<SessionCountingBackend>(
        ctx->runtime->foreground(),
        foreground_vocab_size,
        std::vector<std::vector<int>>{{0, eos_token_id}});
    ctx->foreground_vlm.configure_loaded_for_tests(
        aila::alia::ModelRole::ForegroundVlm,
        &ctx->runtime->foreground(),
        std::move(foreground_tokenizer),
        std::move(foreground_backend),
        aila::alia::BackendKind::Qwen35HybridBnb4);

    std::vector<std::string> tts_tokens{
        "a", "s", "i", "t", "n", "h", "e", "l", "o"
    };
    std::filesystem::path tts_tokenizer_dir =
        make_temp_tokenizer_dir("tts_cancel_streaming", tts_tokens);
    auto tts_tokenizer = std::make_unique<Tokenizer>();
    ALIA_EXPECT_TRUE(tts_tokenizer->load(tts_tokenizer_dir.string()));

    auto tts_backend = std::make_unique<CancellableStreamingTtsBackend>(
        ctx->runtime->foreground());
    CancellableStreamingTtsBackend* tts_observer = tts_backend.get();
    ctx->tts.configure_loaded_for_tests(
        aila::alia::ModelRole::Tts,
        &ctx->runtime->foreground(),
        std::move(tts_tokenizer),
        std::move(tts_backend),
        aila::alia::BackendKind::Qwen3Tts);

    RecordingAudioCallbackState audio_state;
    AliaGenConfig config{0.0f, 1.0f, 2};
    ALIA_EXPECT_EQ(alia_start_conversation_turn(
                       ctx, &config, nullptr, recording_audio_callback, &audio_state),
                   ALIA_OK);
    ALIA_EXPECT_TRUE(tts_observer->wait_until_entered_for(std::chrono::seconds(2)));

    ALIA_EXPECT_EQ(alia_abort_inference(ctx, ALIA_PIPELINE_ALL), ALIA_OK);
    ALIA_EXPECT_TRUE(ctx->foreground_pipeline->wait_until_idle_for(std::chrono::seconds(2)));

    ALIA_EXPECT_EQ(tts_observer->stream_call_count(), 1);
    ALIA_EXPECT_TRUE(tts_observer->observed_cancel());
    ALIA_EXPECT_EQ(audio_state.call_count, 0);
    ALIA_EXPECT_EQ(ctx->foreground_pipeline->state(),
                   aila::alia::ForegroundTurnState::Aborted);

    alia_context_destroy(ctx);
    remove_temp_dir(foreground_tokenizer_dir);
    remove_temp_dir(tts_tokenizer_dir);
}

void test_runtime_context_creates_two_lanes_on_one_sycl_context() {
    aila::alia::RuntimeContext runtime;

    ALIA_EXPECT_TRUE(runtime.foreground().queue().get_context() ==
                     runtime.background().queue().get_context());
    ALIA_EXPECT_TRUE(runtime.foreground().queue().get_device() ==
                     runtime.background().queue().get_device());
    ALIA_EXPECT_TRUE(&runtime.foreground() != &runtime.background());
}

void test_runtime_context_lanes_can_allocate_and_copy_device_memory() {
    aila::alia::RuntimeContext runtime;

    int host_in = 42;
    int host_out = 0;

    void* fg_ptr = runtime.foreground().alloc_device(sizeof(int));
    runtime.foreground().memcpy_h2d(fg_ptr, &host_in, sizeof(int));
    runtime.foreground().memcpy_d2h(&host_out, fg_ptr, sizeof(int));
    runtime.foreground().free_device(fg_ptr);
    ALIA_EXPECT_EQ(host_out, 42);

    host_in = 77;
    host_out = 0;
    void* bg_ptr = runtime.background().alloc_device(sizeof(int));
    runtime.background().memcpy_h2d(bg_ptr, &host_in, sizeof(int));
    runtime.background().memcpy_d2h(&host_out, bg_ptr, sizeof(int));
    runtime.background().free_device(bg_ptr);
    ALIA_EXPECT_EQ(host_out, 77);
}

void test_context_initializes_four_model_slots_on_expected_lanes() {
    auto asr_dir = make_temp_model_dir("asr", R"({"model_type":"qwen3_asr"})");
    auto fg_dir = make_temp_model_dir("fg", qwen35_nf4_config());
    auto bg_dir = make_temp_model_dir("bg", qwen35_nf4_config());
    auto tts_dir = make_temp_model_dir("tts", R"({"model_type":"qwen3_tts"})");

    AliaContext ctx(4096);
    ctx.asr_model_dir = asr_dir.string();
    ctx.vlm_4b_model_dir = fg_dir.string();
    ctx.vlm_0_8b_model_dir = bg_dir.string();
    ctx.tts_model_dir = tts_dir.string();
    ctx.configure_model_slots();

    ALIA_EXPECT_TRUE(ctx.load_model_metadata());
    ALIA_EXPECT_EQ(ctx.asr.role(), aila::alia::ModelRole::Asr);
    ALIA_EXPECT_EQ(ctx.foreground_vlm.role(), aila::alia::ModelRole::ForegroundVlm);
    ALIA_EXPECT_EQ(ctx.background_vlm.role(), aila::alia::ModelRole::BackgroundVlm);
    ALIA_EXPECT_EQ(ctx.tts.role(), aila::alia::ModelRole::Tts);

    ALIA_EXPECT_TRUE(ctx.asr.context() == &ctx.runtime->foreground());
    ALIA_EXPECT_TRUE(ctx.foreground_vlm.context() == &ctx.runtime->foreground());
    ALIA_EXPECT_TRUE(ctx.tts.context() == &ctx.runtime->foreground());
    ALIA_EXPECT_TRUE(ctx.background_vlm.context() == &ctx.runtime->background());

    ALIA_EXPECT_TRUE(ctx.asr.model_dir() == asr_dir.string());
    ALIA_EXPECT_TRUE(ctx.foreground_vlm.model_dir() == fg_dir.string());
    ALIA_EXPECT_TRUE(ctx.background_vlm.model_dir() == bg_dir.string());
    ALIA_EXPECT_TRUE(ctx.tts.model_dir() == tts_dir.string());
    ALIA_EXPECT_EQ(ctx.asr.state(), aila::alia::ModelSlotState::MetadataLoaded);
    ALIA_EXPECT_EQ(ctx.foreground_vlm.state(), aila::alia::ModelSlotState::MetadataLoaded);
    ALIA_EXPECT_EQ(ctx.background_vlm.state(), aila::alia::ModelSlotState::MetadataLoaded);
    ALIA_EXPECT_EQ(ctx.tts.state(), aila::alia::ModelSlotState::MetadataLoaded);

    remove_temp_dir(asr_dir);
    remove_temp_dir(fg_dir);
    remove_temp_dir(bg_dir);
    remove_temp_dir(tts_dir);
}

void test_context_init_fails_when_required_model_path_is_missing() {
    auto missing_dir = std::filesystem::temp_directory_path() /
        "aila_alia_api_tests_missing_model_dir";
    remove_temp_dir(missing_dir);

    AliaContext* ctx = reinterpret_cast<AliaContext*>(0x1);
    int rc = alia_context_init(&ctx, missing_dir.string().c_str(), "", "", "", 4096);

    ALIA_EXPECT_EQ(rc, ALIA_ERR_MODEL_LOAD);
    ALIA_EXPECT_TRUE(ctx == nullptr);
}

void test_context_init_fails_when_model_assets_are_incomplete() {
    auto asr_dir = make_temp_model_dir("asr_incomplete_assets",
                                       R"({"model_type":"qwen3_asr"})");

    AliaContext* ctx = reinterpret_cast<AliaContext*>(0x1);
    int rc = alia_context_init(&ctx, asr_dir.string().c_str(), "", "", "", 4096);

    ALIA_EXPECT_EQ(rc, ALIA_ERR_MODEL_LOAD);
    ALIA_EXPECT_TRUE(ctx == nullptr);

    remove_temp_dir(asr_dir);
}

void test_model_slot_records_metadata_load_errors() {
    aila::alia::RuntimeContext runtime;
    aila::alia::ModelSlot slot;
    slot.configure(aila::alia::ModelRole::Asr, "Z:/definitely/missing/alia/model",
                   &runtime.foreground());

    ALIA_EXPECT_TRUE(!slot.load_metadata());
    ALIA_EXPECT_EQ(slot.state(), aila::alia::ModelSlotState::Failed);
    ALIA_EXPECT_TRUE(slot.last_error().find("config.json") != std::string::npos);
}

void test_model_slot_load_model_fails_before_loaded_when_tokenizer_is_missing() {
    auto asr_dir = make_temp_model_dir("asr_no_tokenizer", R"({"model_type":"qwen3_asr"})");

    aila::alia::RuntimeContext runtime;
    aila::alia::ModelSlot slot;
    slot.configure(aila::alia::ModelRole::Asr, asr_dir.string(), &runtime.foreground());

    ALIA_EXPECT_TRUE(!slot.load_model(2048));
    ALIA_EXPECT_EQ(slot.state(), aila::alia::ModelSlotState::Failed);
    ALIA_EXPECT_TRUE(slot.last_error().find("tokenizer") != std::string::npos ||
                     slot.last_error().find("Tokenizer") != std::string::npos);
    ALIA_EXPECT_TRUE(slot.backend() == nullptr);
    ALIA_EXPECT_TRUE(slot.weights() == nullptr);

    remove_temp_dir(asr_dir);
}

void test_model_slots_select_alia_backend_kinds_from_metadata() {
    auto asr_dir = make_temp_model_dir("asr_nf4", R"({
        "model_type":"qwen3_asr",
        "quantization_config":{
            "quant_method":"bitsandbytes",
            "load_in_4bit":true,
            "bnb_4bit_quant_type":"nf4",
            "bnb_4bit_compute_dtype":"float16",
            "bnb_4bit_quant_storage":"uint8"
        }
    })");
    auto fg_dir = make_temp_model_dir("fg_nf4", qwen35_nf4_config());
    auto bg_dir = make_temp_model_dir("bg_nf4", qwen35_nf4_config());
    auto tts_dir = make_temp_model_dir("tts_dense", R"({"model_type":"qwen3_tts"})");

    AliaContext ctx(4096);
    ctx.asr_model_dir = asr_dir.string();
    ctx.vlm_4b_model_dir = fg_dir.string();
    ctx.vlm_0_8b_model_dir = bg_dir.string();
    ctx.tts_model_dir = tts_dir.string();
    ctx.configure_model_slots();

    ALIA_EXPECT_TRUE(ctx.load_model_metadata());
    ALIA_EXPECT_EQ(ctx.asr.backend_kind(), aila::alia::BackendKind::Qwen3AsrBnb4);
    ALIA_EXPECT_EQ(ctx.foreground_vlm.backend_kind(), aila::alia::BackendKind::Qwen35HybridBnb4);
    ALIA_EXPECT_EQ(ctx.background_vlm.backend_kind(), aila::alia::BackendKind::Qwen35HybridBnb4);
    ALIA_EXPECT_EQ(ctx.tts.backend_kind(), aila::alia::BackendKind::Qwen3Tts);

    remove_temp_dir(asr_dir);
    remove_temp_dir(fg_dir);
    remove_temp_dir(bg_dir);
    remove_temp_dir(tts_dir);
}

}  // namespace

int main() {
    test_context_init_rejects_null_out_pointer();
    test_context_init_and_destroy_allocates_handle_without_models();
    test_context_init_rejects_non_positive_sequence_length();
    test_runtime_functions_reject_null_context();
    test_asr_feed_rejects_bad_audio_arguments();
    test_asr_pipeline_owns_feed_and_reset_state();
    test_asr_pipeline_reports_readiness_from_loaded_slot();
    test_asr_pipeline_process_pending_is_safe_without_loaded_model();
    test_asr_output_parser_extracts_language_and_text();
    test_background_trigger_requires_registered_callback();
    test_background_system_prompt_is_alia_json_extraction_prompt();
    test_background_schema_accepts_valid_memory_result();
    test_background_schema_repairs_malformed_json_with_required_key_names();
    test_background_schema_repairs_wrong_required_field_types();
    test_background_pipeline_invokes_callback_and_rejects_busy_trigger();
    test_background_loaded_vlm_retries_invalid_json_before_schema_repair();
    test_background_loaded_vlm_reports_schema_repair_after_retry_failure();
    test_foreground_pipeline_rejects_second_turn_while_worker_is_running();
    test_foreground_turn_rejects_invalid_generation_config();
    test_vlm_rollback_requires_loaded_foreground_anchor();
    test_foreground_turn_captures_asr_text_and_generation_config();
    test_foreground_system_prompt_is_alia_specific();
    test_foreground_tool_call_invokes_callback_and_keeps_spoken_text_native();
    test_foreground_tool_result_is_promoted_to_resume_prompt_state();
    test_foreground_tool_resume_appends_result_without_chat_scaffold();
    test_foreground_loaded_vlm_pauses_initial_decode_on_complete_tool_call();
    test_foreground_loaded_vlm_streams_sentence_to_tts_before_decode_finishes();
    test_foreground_spoken_text_is_streamed_to_tts_as_chunks();
    test_foreground_abort_during_tts_chunk_stops_remaining_audio();
    test_tts_pipeline_owns_text_queue_and_audio_callback();
    test_tts_pipeline_uses_loaded_backend_streaming_path();
    test_foreground_abort_cancels_loaded_tts_backend_stream();
    test_runtime_context_creates_two_lanes_on_one_sycl_context();
    test_runtime_context_lanes_can_allocate_and_copy_device_memory();
    test_context_initializes_four_model_slots_on_expected_lanes();
    test_context_init_fails_when_required_model_path_is_missing();
    test_context_init_fails_when_model_assets_are_incomplete();
    test_model_slot_records_metadata_load_errors();
    test_model_slot_load_model_fails_before_loaded_when_tokenizer_is_missing();
    test_model_slots_select_alia_backend_kinds_from_metadata();

    if (failures != 0) {
        std::cerr << failures << " Alia API test assertion(s) failed\n";
        return EXIT_FAILURE;
    }

    std::cout << "Alia API tests passed\n";
    return EXIT_SUCCESS;
}
