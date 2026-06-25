#include "AliaForegroundPipeline.hpp"

#include "AliaAsrPipeline.hpp"
#include "ModelSlot.hpp"
#include "AliaTtsPipeline.hpp"
#include "../chat/AssistantOutputParser.hpp"
#include "../chat/ChatJson.hpp"
#include "../chat/StructuredStreamParser.hpp"
#include "../models/IModelBackend.hpp"
#include "../ops/Ops.hpp"
#include "../utils/Tokenizer.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <cstring>
#include <cctype>
#include <functional>
#include <vector>

namespace aila::alia {
namespace {

class DeviceAllocation {
public:
    DeviceAllocation(Context& ctx, size_t bytes)
        : ctx_(&ctx),
          ptr_(ctx.alloc_device(bytes)) {}

    ~DeviceAllocation() {
        if (ctx_ && ptr_) {
            ctx_->free_device(ptr_);
        }
    }

    DeviceAllocation(const DeviceAllocation&) = delete;
    DeviceAllocation& operator=(const DeviceAllocation&) = delete;

    template <typename T>
    T* as() const {
        return static_cast<T*>(ptr_);
    }

private:
    Context* ctx_ = nullptr;
    void* ptr_ = nullptr;
};

Tensor* forward_token_span(Context& context,
                           IModelBackend& backend,
                           const int* token_ids,
                           int token_count,
                           bool use_decode_path) {
    if (token_count <= 0) {
        return nullptr;
    }

    Tensor* logits = nullptr;
    if (use_decode_path) {
        DeviceAllocation token_device(context, sizeof(int));
        for (int i = 0; i < token_count; ++i) {
            context.memcpy_h2d(token_device.as<int>(), token_ids + i, sizeof(int));
            logits = &backend.forward(context, token_device.as<int>(), 1);
        }
        return logits;
    }

    DeviceAllocation prompt_device(context, static_cast<size_t>(token_count) * sizeof(int));
    context.memcpy_h2d(prompt_device.as<int>(),
                       token_ids,
                       static_cast<size_t>(token_count) * sizeof(int));
    return &backend.forward(context, prompt_device.as<int>(), token_count);
}

class BackendCancellationScope {
public:
    BackendCancellationScope(IModelBackend* backend, std::function<bool()> should_cancel)
        : backend_(backend) {
        if (backend_) {
            backend_->set_cancellation_checker(std::move(should_cancel));
        }
    }

    ~BackendCancellationScope() {
        if (backend_) {
            backend_->set_cancellation_checker({});
        }
    }

    BackendCancellationScope(const BackendCancellationScope&) = delete;
    BackendCancellationScope& operator=(const BackendCancellationScope&) = delete;

private:
    IModelBackend* backend_ = nullptr;
};

AliaGenConfig default_alia_generation_config() {
    AliaGenConfig config{};
    config.temperature = 0.6f;
    config.top_p = 0.9f;
    config.max_tokens = 256;
    return config;
}

std::string build_tool_resume_prompt_text(const std::string& user_text,
                                          const std::string& spoken_text,
                                          const std::string& tool_call_json,
                                          const std::string& tool_result_text) {
    std::string prompt;
    prompt += "User request:\n";
    prompt += user_text.empty() ? "(no user text captured)" : user_text;
    prompt += "\n\nAssistant spoken text before tool result:\n";
    prompt += spoken_text.empty() ? "(none)" : spoken_text;
    prompt += "\n\nAssistant tool call JSON:\n";
    prompt += tool_call_json;
    prompt += "\n\nHost tool result:\n";
    prompt += tool_result_text;
    prompt += "\n\nContinue the response to the user in concise spoken text. "
              "Do not repeat the tool call JSON.";
    return prompt;
}

std::string build_tool_result_continuation_text(const std::string& tool_result_text) {
    std::string text;
    text += "\n<tool_result>\n";
    text += tool_result_text.empty() ? "{}" : tool_result_text;
    text += "\n</tool_result>\n";
    text += "Continue the response in concise spoken text.\n";
    return text;
}

const std::vector<std::string>& tts_chunk_boundary_markers() {
    static const std::vector<std::string> markers = {
        ".", "!", "?", ";", "\n",
        "。", "！", "？", "；", "…",
    };
    return markers;
}

bool ends_with_tts_chunk_boundary(const std::string& text) {
    for (const std::string& marker : tts_chunk_boundary_markers()) {
        if (text.size() >= marker.size() &&
            text.compare(text.size() - marker.size(), marker.size(), marker) == 0) {
            return true;
        }
    }
    return false;
}

size_t last_tts_chunk_boundary(const std::string& text) {
    size_t cutoff = std::string::npos;
    for (const std::string& marker : tts_chunk_boundary_markers()) {
        size_t pos = text.find(marker);
        while (pos != std::string::npos) {
            cutoff = std::max(cutoff == std::string::npos ? 0 : cutoff,
                              pos + marker.size());
            pos = text.find(marker, pos + marker.size());
        }
    }
    return cutoff;
}

std::string trim_ascii(std::string value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) {
        value.erase(0, 1);
    }
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) {
        value.pop_back();
    }
    return value;
}

std::string lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

bool is_explicit_tool_request(const std::string& user_text) {
    const std::string lower = lower_ascii(user_text);
    return lower.find("host tool") != std::string::npos ||
           lower.find("tool call") != std::string::npos ||
           lower.find("inspect_window") != std::string::npos ||
           lower.find("<tool_call") != std::string::npos;
}

std::string strip_structured_artifacts_from_spoken_text(std::string text) {
    const char* markers[] = {
        "<tool_call>",
        "</tool_call>",
        "<function=",
        "</function>",
        "<parameter=",
        "</parameter>",
        "<think>",
        "</think>",
    };

    size_t cutoff = std::string::npos;
    for (const char* marker : markers) {
        const size_t pos = text.find(marker);
        if (pos != std::string::npos &&
            (cutoff == std::string::npos || pos < cutoff)) {
            cutoff = pos;
        }
    }
    if (cutoff != std::string::npos) {
        text.erase(cutoff);
    }
    return trim_ascii(std::move(text));
}

void append_encoded(std::vector<int>& ids, Tokenizer* tokenizer, const std::string& text) {
    std::vector<int> encoded = tokenizer->encode(text);
    ids.insert(ids.end(), encoded.begin(), encoded.end());
}

void append_alia_system_turn(std::vector<int>& ids,
                             Tokenizer* tokenizer,
                             const std::string& system_prompt) {
    if (system_prompt.empty()) {
        return;
    }
    ids.push_back(tokenizer->im_start_id());
    append_encoded(ids, tokenizer, "system\n" + system_prompt);
    ids.push_back(tokenizer->im_end_id());
    append_encoded(ids, tokenizer, "\n");
}

std::vector<int> build_alia_chat_prompt(
    Tokenizer* tokenizer,
    const std::string& system_prompt,
    const std::string& user_message) {
    std::vector<int> prompt_ids;
    append_alia_system_turn(prompt_ids, tokenizer, system_prompt);
    prompt_ids.push_back(tokenizer->im_start_id());
    append_encoded(prompt_ids, tokenizer, "user\n" + user_message);
    prompt_ids.push_back(tokenizer->im_end_id());
    append_encoded(prompt_ids, tokenizer, "\n");
    prompt_ids.push_back(tokenizer->im_start_id());
    append_encoded(prompt_ids, tokenizer, "assistant\n");
    append_encoded(prompt_ids, tokenizer, "<think>\n\n</think>\n\n");
    return prompt_ids;
}

std::vector<int> build_alia_user_prefix_prompt(
    Tokenizer* tokenizer,
    const std::string& system_prompt,
    const std::string& user_prefix) {
    std::vector<int> prompt_ids;
    append_alia_system_turn(prompt_ids, tokenizer, system_prompt);
    prompt_ids.push_back(tokenizer->im_start_id());
    append_encoded(prompt_ids, tokenizer, "user\n" + user_prefix);
    return prompt_ids;
}

std::string combine_asr_text(const std::string& stable_text,
                             const std::string& partial_text) {
    std::string stable = trim_ascii(stable_text);
    std::string partial = trim_ascii(partial_text);
    if (stable.empty()) {
        return partial;
    }
    if (partial.empty()) {
        return stable;
    }
    if (!std::isspace(static_cast<unsigned char>(stable.back())) &&
        !std::isspace(static_cast<unsigned char>(partial.front())) &&
        !std::ispunct(static_cast<unsigned char>(partial.front()))) {
        stable += " ";
    }
    stable += partial;
    return stable;
}

bool is_token_prefix(const std::vector<int>& prefix, const std::vector<int>& value) {
    return prefix.size() <= value.size() &&
           std::equal(prefix.begin(), prefix.end(), value.begin());
}

size_t common_token_prefix_size(const std::vector<int>& a, const std::vector<int>& b) {
    const size_t n = std::min(a.size(), b.size());
    size_t i = 0;
    while (i < n && a[i] == b[i]) {
        ++i;
    }
    return i;
}

std::vector<std::string> take_ready_tts_chunks(std::string& buffer, bool force) {
    size_t cutoff = std::string::npos;
    if (force) {
        cutoff = buffer.size();
    } else {
        cutoff = last_tts_chunk_boundary(buffer);
    }

    if (cutoff == std::string::npos || cutoff == 0) {
        return {};
    }

    std::string ready = buffer.substr(0, cutoff);
    buffer.erase(0, cutoff);
    return split_spoken_text_for_tts(ready);
}

}  // namespace

std::string foreground_system_prompt() {
    return "You are Alia, a local companion running inside the native Aila "
           "engine. Answer with concise spoken text for real-time voice "
           "interaction. Do not answer in JSON, Markdown code fences, or "
           "schemas. No host tools are available unless the user message "
           "explicitly provides a tool request, so ordinary turns must be "
           "answered as natural language only. For explicit host tool "
           "requests, emit only "
           "<tool_call><function=name><parameter=name>value</parameter>"
           "</function></tool_call>.";
}

std::vector<std::string> split_spoken_text_for_tts(const std::string& text) {
    std::vector<std::string> chunks;
    std::string current;
    auto flush = [&]() {
        size_t begin = 0;
        while (begin < current.size() &&
               std::isspace(static_cast<unsigned char>(current[begin]))) {
            ++begin;
        }
        size_t end = current.size();
        while (end > begin &&
               std::isspace(static_cast<unsigned char>(current[end - 1]))) {
            --end;
        }
        if (end > begin) {
            chunks.push_back(current.substr(begin, end - begin));
        }
        current.clear();
    };

    for (char ch : text) {
        current.push_back(ch);
        if (ends_with_tts_chunk_boundary(current)) {
            flush();
        }
    }
    flush();
    return chunks;
}

bool is_valid_generation_config(const AliaGenConfig& config) {
    return std::isfinite(config.temperature) &&
           std::isfinite(config.top_p) &&
           config.temperature >= 0.0f &&
           config.top_p > 0.0f &&
           config.top_p <= 1.0f &&
           config.max_tokens > 0;
}

GenerationConfig translate_generation_config(const AliaGenConfig* config) {
    const AliaGenConfig source = config ? *config : default_alia_generation_config();

    GenerationConfig translated;
    translated.max_new_tokens = source.max_tokens;
    translated.temperature = source.temperature;
    translated.top_p = source.top_p;
    translated.do_sample = source.temperature > 0.0f;
    return translated;
}

AliaForegroundPipeline::AliaForegroundPipeline(
    ModelSlot* vlm_slot,
    AliaTtsPipeline* tts_pipeline,
    AliaAsrPipeline* asr_pipeline)
    : vlm_slot_(vlm_slot),
      tts_pipeline_(tts_pipeline),
      asr_pipeline_(asr_pipeline) {}

AliaForegroundPipeline::~AliaForegroundPipeline() {
    request_abort();
    join();
}

bool AliaForegroundPipeline::start_turn(
    const AliaGenConfig* config,
    AliaToolCallCallback tool_cb,
    AliaAudioCallback audio_cb,
    void* user_data) {
    if (config && !is_valid_generation_config(*config)) {
        return false;
    }

    AliaGenConfig captured_config{};
    if (config) {
        captured_config = *config;
    } else {
        captured_config = default_alia_generation_config();
    }

    std::unique_lock<std::mutex> lock(mutex_);
    if (is_busy_locked()) {
        return false;
    }

    if (worker_.joinable()) {
        lock.unlock();
        worker_.join();
        lock.lock();
    }

    abort_requested_ = false;
    last_error_.clear();
    state_ = ForegroundTurnState::Running;

    worker_ = std::thread([this, captured_config, tool_cb, audio_cb, user_data]() {
        run_turn(captured_config, tool_cb, audio_cb, user_data);
    });
    return true;
}

AliaErrorCode AliaForegroundPipeline::prefill_asr_text(
    const std::string& stable_text,
    const std::string& partial_text) {
    const std::string user_prefix = combine_asr_text(stable_text, partial_text);
    if (user_prefix.empty()) {
        return ALIA_OK;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (is_busy_locked()) {
            return ALIA_ERR_INVALID_STATE;
        }
    }

    if (!can_generate_with_loaded_vlm()) {
        return ALIA_ERR_INVALID_STATE;
    }

    Context* context = vlm_slot_->context();
    Tokenizer* tokenizer = vlm_slot_->tokenizer();
    IModelBackend* backend = vlm_slot_->backend();
    if (!context || !tokenizer || !backend) {
        return ALIA_ERR_INVALID_STATE;
    }

    constexpr int kRetokenizeGuardTokens = 16;
    constexpr int kFinalPromptSuffixGuardTokens = 16;
    constexpr int kMinIncrementalPrefillSuffixTokens = 16;
    constexpr int kMaxDecodePathPrefillSuffixTokens = 64;
    const auto started = std::chrono::steady_clock::now();
    std::vector<int> target_ids =
        build_alia_user_prefix_prompt(tokenizer, foreground_system_prompt(), user_prefix);
    const std::vector<int> full_candidate_ids =
        build_alia_chat_prompt(tokenizer, foreground_system_prompt(), user_prefix);
    if (static_cast<int>(target_ids.size()) > kRetokenizeGuardTokens) {
        target_ids.resize(target_ids.size() - kRetokenizeGuardTokens);
    } else {
        target_ids.clear();
    }
    if (full_candidate_ids.size() <=
            static_cast<size_t>(kFinalPromptSuffixGuardTokens) ||
        target_ids.size() + static_cast<size_t>(kFinalPromptSuffixGuardTokens) >
            full_candidate_ids.size()) {
        const size_t capped_size = full_candidate_ids.size() >
                static_cast<size_t>(kFinalPromptSuffixGuardTokens)
            ? full_candidate_ids.size() - static_cast<size_t>(kFinalPromptSuffixGuardTokens)
            : 0;
        target_ids.resize(std::min(target_ids.size(), capped_size));
    }
    while (!target_ids.empty() && !is_token_prefix(target_ids, full_candidate_ids)) {
        target_ids.pop_back();
    }
    if (target_ids.empty()) {
        return ALIA_OK;
    }

    const int max_seq_len = backend->max_seq_len();
    if (max_seq_len > 0 && static_cast<int>(target_ids.size()) > max_seq_len) {
        return ALIA_ERR_CONTEXT_OVERFLOW;
    }

    auto lane_lock = context->lock_execution();
    std::vector<int> current_prefill;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        current_prefill = asr_prefill_prompt_ids_;
    }

    bool reset_required = false;
    size_t already_prefilled = 0;
    int reused_tokens = 0;
    const int current_len = backend->get_current_context_len();
    if (!current_prefill.empty() &&
        current_len == static_cast<int>(current_prefill.size())) {
        if (is_token_prefix(current_prefill, target_ids)) {
            already_prefilled = current_prefill.size();
        } else {
            const size_t common_prefix = common_token_prefix_size(current_prefill, target_ids);
            if (common_prefix > 0 &&
                backend->truncate_kv_cache(static_cast<int>(common_prefix)) &&
                backend->get_current_context_len() == static_cast<int>(common_prefix)) {
                already_prefilled = common_prefix;
            } else {
                reset_required = true;
            }
        }
    } else {
        reset_required = true;
    }

    reused_tokens = static_cast<int>(already_prefilled);
    if (already_prefilled >= target_ids.size()) {
        const long long elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - started).count();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            asr_prefill_prompt_ids_ = std::move(target_ids);
            asr_prefill_text_ = user_prefix;
            last_asr_prefill_token_count_ =
                static_cast<int>(asr_prefill_prompt_ids_.size());
            last_asr_prefill_reused_token_count_ = reused_tokens;
            last_asr_prefill_suffix_token_count_ = 0;
            last_asr_prefill_ms_ = elapsed_ms;
        }
        return ALIA_OK;
    }

    if (reset_required) {
        backend->reset();
        already_prefilled = 0;
        reused_tokens = 0;
    }

    const size_t suffix_count = target_ids.size() - already_prefilled;
    const bool small_append =
        !reset_required &&
        !current_prefill.empty() &&
        already_prefilled == current_prefill.size() &&
        suffix_count > 0 &&
        suffix_count < static_cast<size_t>(kMinIncrementalPrefillSuffixTokens) &&
        is_token_prefix(current_prefill, target_ids);
    if (small_append) {
        const long long elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - started).count();
        std::lock_guard<std::mutex> lock(mutex_);
        ++last_asr_prefill_skipped_small_suffix_count_;
        last_asr_prefill_token_count_ = static_cast<int>(current_prefill.size());
        last_asr_prefill_reused_token_count_ = reused_tokens;
        last_asr_prefill_suffix_token_count_ = static_cast<int>(suffix_count);
        last_asr_prefill_ms_ = elapsed_ms;
        return ALIA_OK;
    }

    const int suffix_tokens = static_cast<int>(suffix_count);
    // For small cached suffixes, the single-token decode kernels are faster
    // than the Qwen3.5 batch prefill path while preserving causal equivalence.
    const bool decode_path_suffix =
        already_prefilled > 0 && suffix_tokens <= kMaxDecodePathPrefillSuffixTokens;
    forward_token_span(*context,
                       *backend,
                       target_ids.data() + already_prefilled,
                       suffix_tokens,
                       decode_path_suffix);

    const long long elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started).count();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (already_prefilled == 0) {
            last_asr_prefill_skipped_small_suffix_count_ = 0;
        }
        asr_prefill_prompt_ids_ = std::move(target_ids);
        asr_prefill_text_ = user_prefix;
        last_asr_prefill_token_count_ = static_cast<int>(asr_prefill_prompt_ids_.size());
        last_asr_prefill_reused_token_count_ = reused_tokens;
        last_asr_prefill_suffix_token_count_ = static_cast<int>(suffix_count);
        last_asr_prefill_ms_ = elapsed_ms;
    }
    return ALIA_OK;
}

void AliaForegroundPipeline::request_abort() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        abort_requested_ = true;
        if (state_ == ForegroundTurnState::Running) {
            state_ = ForegroundTurnState::Aborting;
        }
    }
    cv_.notify_all();
}

AliaErrorCode AliaForegroundPipeline::rollback_kv_cache(int rollback_tokens) {
    if (rollback_tokens < 0) {
        return ALIA_ERR_INVALID_ARGUMENT;
    }
    if (rollback_tokens == 0) {
        return ALIA_OK;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (is_busy_locked()) {
            return ALIA_ERR_INVALID_STATE;
        }
        if (generation_start_context_len_ < 0 || last_generated_token_count_ <= 0) {
            return ALIA_ERR_INVALID_STATE;
        }
    }

    if (!can_generate_with_loaded_vlm()) {
        return ALIA_ERR_INVALID_STATE;
    }

    Context* context = vlm_slot_->context();
    IModelBackend* backend = vlm_slot_->backend();
    if (!context || !backend) {
        return ALIA_ERR_INVALID_STATE;
    }
    auto lane_lock = context->lock_execution();
    const int current_len = backend->get_current_context_len();
    int anchor = 0;
    std::vector<int> anchor_prompt_ids;
    std::vector<int> generated_token_ids;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        anchor = generation_start_context_len_;
        anchor_prompt_ids = generation_anchor_prompt_ids_;
        generated_token_ids = generation_token_ids_;
    }

    const int replayable_len = anchor + static_cast<int>(generated_token_ids.size());
    if (anchor < 0 || anchor_prompt_ids.empty() || current_len != replayable_len) {
        return ALIA_ERR_INVALID_STATE;
    }

    int target_len = std::max(anchor, current_len - rollback_tokens);
    if (target_len > current_len) {
        target_len = current_len;
    }

    auto replay_to_target = [&](int replay_target_len) -> bool {
        if (!vlm_slot_ || !vlm_slot_->context() || anchor_prompt_ids.empty()) {
            return false;
        }
        const int replay_generated_count = replay_target_len - anchor;
        if (replay_target_len < anchor ||
            replay_generated_count < 0 ||
            replay_generated_count > static_cast<int>(generated_token_ids.size())) {
            return false;
        }

        backend->reset();

        DeviceAllocation prompt_device(*context, anchor_prompt_ids.size() * sizeof(int));
        context->memcpy_h2d(prompt_device.as<int>(), anchor_prompt_ids.data(),
                            anchor_prompt_ids.size() * sizeof(int));
        backend->forward(*context, prompt_device.as<int>(),
                         static_cast<int>(anchor_prompt_ids.size()));

        if (replay_generated_count > 0) {
            DeviceAllocation token_device(*context, sizeof(int));
            for (int i = 0; i < replay_generated_count; ++i) {
                const int token_id = generated_token_ids[static_cast<size_t>(i)];
                context->memcpy_h2d(token_device.as<int>(), &token_id, sizeof(int));
                backend->forward(*context, token_device.as<int>(), 1);
            }
        }

        return backend->get_current_context_len() == replay_target_len;
    };

    const bool truncated = backend->truncate_kv_cache(target_len);
    const int truncated_len = backend->get_current_context_len();
    if (!truncated || truncated_len != target_len) {
        if (!replay_to_target(target_len)) {
            return ALIA_ERR_RUNTIME;
        }
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        last_generated_token_count_ = std::max(0, target_len - generation_start_context_len_);
        if (static_cast<int>(generation_token_ids_.size()) > last_generated_token_count_) {
            generation_token_ids_.resize(static_cast<size_t>(last_generated_token_count_));
        }
    }
    return ALIA_OK;
}

void AliaForegroundPipeline::join() {
    if (worker_.joinable()) {
        worker_.join();
    }
}

ForegroundTurnState AliaForegroundPipeline::state() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_;
}

bool AliaForegroundPipeline::wait_until_idle_for(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(lock, timeout, [this]() { return !is_busy_locked(); });
}

GenerationConfig AliaForegroundPipeline::last_generation_config() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_generation_config_;
}

std::string AliaForegroundPipeline::last_user_text() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_user_text_;
}

std::string AliaForegroundPipeline::last_assistant_text() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_assistant_text_;
}

std::string AliaForegroundPipeline::last_tool_call_json() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_tool_call_json_;
}

std::string AliaForegroundPipeline::last_tool_result_text() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_tool_result_text_;
}

std::string AliaForegroundPipeline::last_tool_resume_prompt_text() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_tool_resume_prompt_text_;
}

ForegroundDecodeMode AliaForegroundPipeline::last_decode_mode() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_decode_mode_;
}

int AliaForegroundPipeline::last_prompt_token_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_prompt_token_count_;
}

int AliaForegroundPipeline::last_generated_token_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_generated_token_count_;
}

int AliaForegroundPipeline::last_asr_prefill_token_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_asr_prefill_token_count_;
}

int AliaForegroundPipeline::last_asr_prefill_reused_token_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_asr_prefill_reused_token_count_;
}

int AliaForegroundPipeline::last_asr_prefill_suffix_token_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_asr_prefill_suffix_token_count_;
}

int AliaForegroundPipeline::last_asr_prefill_skipped_small_suffix_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_asr_prefill_skipped_small_suffix_count_;
}

long long AliaForegroundPipeline::last_asr_prefill_ms() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_asr_prefill_ms_;
}

long long AliaForegroundPipeline::last_first_content_delta_ms() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_first_content_delta_ms_;
}

long long AliaForegroundPipeline::last_first_tts_enqueue_ms() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_first_tts_enqueue_ms_;
}

AliaForegroundMetrics AliaForegroundPipeline::last_metrics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return metrics_;
}

void AliaForegroundPipeline::run_turn(
    AliaGenConfig config,
    AliaToolCallCallback tool_cb,
    AliaAudioCallback audio_cb,
    void* user_data) {
    try {
        const auto turn_start = std::chrono::steady_clock::now();
        GenerationConfig generation_config = translate_generation_config(&config);
        std::string stable_text;
        std::string partial_text;
        if (asr_pipeline_) {
            asr_pipeline_->get_text(stable_text, partial_text);
        }
        const std::string user_text = combine_asr_text(stable_text, partial_text);
        std::vector<int> prefetched_prompt_ids;
        std::vector<int> prompt_override_ids;
        int prefilled_prompt_tokens = 0;
        if (tts_pipeline_) {
            tts_pipeline_->begin_turn_metrics();
        }
        bool tts_async_started = false;
        auto finish_tts_async = [&]() -> bool {
            if (!tts_async_started || !tts_pipeline_) {
                return true;
            }
            const bool ok = tts_pipeline_->finish_async_turn();
            tts_async_started = false;
            return ok;
        };
        struct TtsAsyncTurnGuard {
            std::function<bool()> finish;
            ~TtsAsyncTurnGuard() {
                if (finish) {
                    finish();
                }
            }
        } tts_async_guard{finish_tts_async};

        {
            std::lock_guard<std::mutex> lock(mutex_);
            last_generation_config_ = generation_config;
            last_user_text_ = user_text;
            last_assistant_text_.clear();
            last_tool_call_json_.clear();
            last_tool_result_text_.clear();
            last_tool_resume_prompt_text_.clear();
            generation_start_context_len_ = -1;
            last_prompt_token_count_ = 0;
            last_generated_token_count_ = 0;
            last_first_content_delta_ms_ = -1;
            last_first_tts_enqueue_ms_ = -1;
            metrics_ = AliaForegroundMetrics{};
            generation_anchor_prompt_ids_.clear();
            generation_token_ids_.clear();
            last_decode_mode_ = ForegroundDecodeMode::None;
            prefetched_prompt_ids = asr_prefill_prompt_ids_;
            if (abort_requested_) {
                state_ = ForegroundTurnState::Aborted;
                cv_.notify_all();
                return;
            }
        }

        if (tts_pipeline_ && audio_cb) {
            tts_async_started = tts_pipeline_->start_async_turn(
                config,
                audio_cb,
                user_data,
                [this]() { return abort_requested(); });
            if (!tts_async_started) {
                std::lock_guard<std::mutex> lock(mutex_);
                last_error_ = "failed to start asynchronous TTS worker";
                state_ = ForegroundTurnState::Failed;
                cv_.notify_all();
                return;
            }
        }

        std::string assistant_text;
        ForegroundDecodeMode decode_mode = ForegroundDecodeMode::LoadedVlm;
        if (!can_generate_with_loaded_vlm()) {
            finish_tts_async();
            std::lock_guard<std::mutex> lock(mutex_);
            last_error_ = "foreground VLM slot is not loaded";
            state_ = ForegroundTurnState::Failed;
            cv_.notify_all();
            return;
        }
        if (!user_text.empty() && !prefetched_prompt_ids.empty()) {
            Tokenizer* tokenizer = vlm_slot_->tokenizer();
            IModelBackend* backend = vlm_slot_->backend();
            std::vector<int> full_prompt =
                build_alia_chat_prompt(tokenizer, foreground_system_prompt(), user_text);
            if (backend &&
                backend->get_current_context_len() == static_cast<int>(prefetched_prompt_ids.size()) &&
                is_token_prefix(prefetched_prompt_ids, full_prompt)) {
                prefilled_prompt_tokens = static_cast<int>(prefetched_prompt_ids.size());
                prompt_override_ids = std::move(full_prompt);
            } else {
                prefetched_prompt_ids.clear();
                prefilled_prompt_tokens = 0;
            }
        }
        if (!generate_with_loaded_vlm(user_text, generation_config, assistant_text,
                                      prefilled_prompt_tokens <= 0, true, true, true,
                                      prompt_override_ids.empty() ? nullptr : &prompt_override_ids,
                                      prefilled_prompt_tokens,
                                      &config, audio_cb, user_data, turn_start)) {
            finish_tts_async();
            std::lock_guard<std::mutex> lock(mutex_);
            if (last_error_.empty()) {
                last_error_ = "foreground VLM generation failed";
            }
            state_ = ForegroundTurnState::Failed;
            cv_.notify_all();
            return;
        }

        std::string spoken_text;
        if (!process_tool_calls(assistant_text, user_text, tool_cb, user_data, spoken_text)) {
            finish_tts_async();
            std::lock_guard<std::mutex> lock(mutex_);
            if (last_error_.empty()) {
                last_error_ = "foreground tool call failed";
            }
            state_ = ForegroundTurnState::Failed;
            cv_.notify_all();
            return;
        }

        if (decode_mode == ForegroundDecodeMode::LoadedVlm) {
            const std::string resume_prompt = last_tool_resume_prompt_text();
            if (!resume_prompt.empty()) {
                std::string resumed_text;
                const std::string continuation_text =
                    build_tool_result_continuation_text(last_tool_result_text());
                if (!generate_with_loaded_vlm(continuation_text, generation_config, resumed_text,
                                              false, false, false, false,
                                              nullptr, 0,
                                              &config, audio_cb, user_data, turn_start)) {
                    finish_tts_async();
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (last_error_.empty()) {
                        last_error_ = "foreground VLM tool-result resume failed";
                    }
                    state_ = ForegroundTurnState::Failed;
                    cv_.notify_all();
                    return;
                }
                if (!resumed_text.empty()) {
                    spoken_text += resumed_text;
                }
            }
        }

        if (!finish_tts_async() && !abort_requested()) {
            std::lock_guard<std::mutex> lock(mutex_);
            last_error_ = "asynchronous TTS worker failed";
            state_ = ForegroundTurnState::Failed;
            cv_.notify_all();
            return;
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            last_assistant_text_ = spoken_text;
            last_decode_mode_ = decode_mode;
            if (abort_requested_) {
                state_ = ForegroundTurnState::Aborted;
                cv_.notify_all();
                return;
            }
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            state_ = abort_requested_ ? ForegroundTurnState::Aborted
                                      : ForegroundTurnState::Completed;
        }
    } catch (const ModelBackendCancelled&) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (abort_requested_) {
            state_ = ForegroundTurnState::Aborted;
        } else {
            last_error_ = "foreground VLM backend cancelled without an abort request";
            state_ = ForegroundTurnState::Failed;
        }
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(mutex_);
        last_error_ = e.what();
        state_ = ForegroundTurnState::Failed;
    } catch (...) {
        std::lock_guard<std::mutex> lock(mutex_);
        last_error_ = "unknown foreground turn failure";
        state_ = ForegroundTurnState::Failed;
    }

    cv_.notify_all();
}

bool AliaForegroundPipeline::can_generate_with_loaded_vlm() const {
    return vlm_slot_ &&
           vlm_slot_->state() == ModelSlotState::Loaded &&
           vlm_slot_->context() &&
           vlm_slot_->tokenizer() &&
           vlm_slot_->backend();
}

bool AliaForegroundPipeline::generate_with_loaded_vlm(
    const std::string& user_text,
    const GenerationConfig& config,
    std::string& assistant_text,
    bool reset_session,
    bool record_generation_anchor,
    bool use_chat_template,
    bool stop_on_tool_call,
    const std::vector<int>* prompt_override_ids,
    int prefilled_prompt_tokens,
    const AliaGenConfig* tts_config,
    AliaAudioCallback audio_cb,
    void* user_data,
    std::chrono::steady_clock::time_point turn_start) {
    assistant_text.clear();
    if (!can_generate_with_loaded_vlm()) {
        return false;
    }

    const auto model_started = std::chrono::steady_clock::now();
    auto elapsed_ms = [](std::chrono::steady_clock::time_point start) -> long long {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();
    };

    Context* context = vlm_slot_->context();
    Tokenizer* tokenizer = vlm_slot_->tokenizer();
    IModelBackend* backend = vlm_slot_->backend();
    if (!context || !tokenizer || !backend) {
        last_error_ = "foreground VLM slot is incomplete";
        return false;
    }
    BackendCancellationScope cancellation_scope(
        backend,
        [this]() { return abort_requested(); });

    const std::string prompt_text = user_text.empty()
        ? "Continue the current conversation."
        : user_text;
    std::vector<int> prompt_ids = prompt_override_ids
        ? *prompt_override_ids
        : (use_chat_template
            ? build_alia_chat_prompt(tokenizer, foreground_system_prompt(), prompt_text)
            : tokenizer->encode(prompt_text));
    const long long prompt_build_ms = elapsed_ms(model_started);
    if (prompt_ids.empty()) {
        last_error_ = use_chat_template
            ? "foreground VLM prompt encoded to zero tokens"
            : "foreground VLM continuation encoded to zero tokens";
        return false;
    }

    const int vocab_size = backend->vocab_size() > 0
        ? backend->vocab_size()
        : tokenizer->vocab_size();
    if (vocab_size <= 0) {
        last_error_ = "foreground VLM vocab size is invalid";
        return false;
    }

    int prompt_tokens_to_forward = 0;
    int effective_prefilled_prompt_tokens = prefilled_prompt_tokens;
    long long prompt_prefill_ms = -1;
    Tensor* logits = nullptr;
    {
        auto lane_lock = context->lock_execution();
        if (reset_session) {
            backend->reset();
            prefilled_prompt_tokens = 0;
        }
        if (prefilled_prompt_tokens < 0 ||
            prefilled_prompt_tokens > static_cast<int>(prompt_ids.size()) ||
            (!reset_session && prefilled_prompt_tokens > 0 &&
             backend->get_current_context_len() != prefilled_prompt_tokens)) {
            backend->reset();
            prefilled_prompt_tokens = 0;
        }

        const int max_seq_len = backend->max_seq_len();
        const int context_len_before_prompt = reset_session ? 0 : backend->get_current_context_len();
        prompt_tokens_to_forward =
            static_cast<int>(prompt_ids.size()) - prefilled_prompt_tokens;
        effective_prefilled_prompt_tokens = prefilled_prompt_tokens;
        if (max_seq_len > 0 &&
            context_len_before_prompt + prompt_tokens_to_forward +
                config.max_new_tokens > max_seq_len) {
            last_error_ = use_chat_template
                ? "foreground VLM prompt would exceed max sequence length"
                : "foreground VLM continuation would exceed max sequence length";
            return false;
        }

        if (prompt_tokens_to_forward <= 0) {
            last_error_ = "foreground VLM prompt was fully prefetched without cached logits";
            return false;
        }

        constexpr int kMaxDecodePathPromptSuffixTokens = 64;
        // Keep full initial prompts on the batch path; use decode kernels only
        // for small suffixes after a validated cached prefix.
        const bool decode_path_prompt_suffix =
            !reset_session &&
            prefilled_prompt_tokens > 0 &&
            prompt_tokens_to_forward <= kMaxDecodePathPromptSuffixTokens;
        const auto prompt_prefill_started = std::chrono::steady_clock::now();
        logits = forward_token_span(*context,
                                    *backend,
                                    prompt_ids.data() + prefilled_prompt_tokens,
                                    prompt_tokens_to_forward,
                                    decode_path_prompt_suffix);
        prompt_prefill_ms = elapsed_ms(prompt_prefill_started);
    }

    if (record_generation_anchor) {
        std::lock_guard<std::mutex> lock(mutex_);
        generation_start_context_len_ = backend->get_current_context_len();
        last_prompt_token_count_ = static_cast<int>(prompt_ids.size());
        last_generated_token_count_ = 0;
        last_first_content_delta_ms_ = -1;
        last_first_tts_enqueue_ms_ = -1;
        generation_anchor_prompt_ids_ = prompt_ids;
        generation_token_ids_.clear();
    }

    DeviceAllocation one_token_device(*context, sizeof(int));
    std::vector<int> generated_ids;
    aila::chat::StructuredStreamParser stream_parser;
    const bool allow_tool_calls = stop_on_tool_call && is_explicit_tool_request(user_text);
    std::string raw_stream_text;
    std::string pending_tts_text;
    bool paused_on_tool_call = false;
    long long first_token_delta_ms = -1;
    const auto decode_started = std::chrono::steady_clock::now();
    auto flush_tts = [&](bool force) {
        if (!tts_pipeline_ || !audio_cb || !tts_config) {
            return;
        }
        for (const std::string& chunk : take_ready_tts_chunks(pending_tts_text, force)) {
            if (abort_requested()) {
                return;
            }
            if (tts_pipeline_->enqueue_text(chunk)) {
                if (record_generation_anchor) {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (last_first_tts_enqueue_ms_ < 0) {
                        last_first_tts_enqueue_ms_ =
                            std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - turn_start).count();
                    }
                }
            }
        }
    };

    const int max_new_tokens = std::max(config.max_new_tokens, 1);
    for (int step = 0; step < max_new_tokens; ++step) {
        if (abort_requested()) {
            break;
        }

        int next_token = 0;
        {
            auto lane_lock = context->lock_execution();
            next_token = ops::sample_with_config(
                *context, *logits, vocab_size, config, generated_ids);
        }
        if (tokenizer->is_eos(next_token)) {
            break;
        }

        if (first_token_delta_ms < 0) {
            first_token_delta_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - turn_start).count();
        }
        generated_ids.push_back(next_token);
        const std::string token_text = tokenizer->decode(next_token);
        raw_stream_text += token_text;
        const bool ordinary_turn_started_tool_markup =
            stop_on_tool_call && !allow_tool_calls &&
            raw_stream_text.find("<tool_call>") != std::string::npos;
        bool complete_tool_call = false;
        std::vector<aila::chat::StructuredStreamEvent> events;
        stream_parser.push(token_text, events);
        for (const auto& event : events) {
            if (event.type == aila::chat::StructuredStreamEventType::ContentDelta) {
                pending_tts_text += event.text;
                if (record_generation_anchor && !event.text.empty()) {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (last_first_content_delta_ms_ < 0) {
                        last_first_content_delta_ms_ =
                            std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - turn_start).count();
                    }
                }
            }
            if (event.type == aila::chat::StructuredStreamEventType::ToolCallDelta &&
                !event.tool_calls.empty()) {
                complete_tool_call = true;
            }
        }
        {
            auto lane_lock = context->lock_execution();
            context->memcpy_h2d(one_token_device.as<int>(), &next_token, sizeof(int));
            logits = &backend->forward(*context, one_token_device.as<int>(), 1);
        }
        flush_tts(false);
        if (ordinary_turn_started_tool_markup) {
            flush_tts(true);
            break;
        }
        if (allow_tool_calls && complete_tool_call) {
            paused_on_tool_call = true;
            flush_tts(true);
            break;
        }
    }

    if (!paused_on_tool_call) {
        std::vector<aila::chat::StructuredStreamEvent> final_events;
        stream_parser.finish(final_events);
        for (const auto& event : final_events) {
            if (event.type == aila::chat::StructuredStreamEventType::ContentDelta) {
                pending_tts_text += event.text;
                if (record_generation_anchor && !event.text.empty()) {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (last_first_content_delta_ms_ < 0) {
                        last_first_content_delta_ms_ =
                            std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - turn_start).count();
                    }
                }
            }
        }
        flush_tts(true);
    }

    const long long decode_ms = elapsed_ms(decode_started);
    assistant_text = tokenizer->decode(generated_ids);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (record_generation_anchor) {
            last_generated_token_count_ = static_cast<int>(generated_ids.size());
            generation_token_ids_ = generated_ids;
            metrics_.prompt_tokens = static_cast<int>(prompt_ids.size());
            metrics_.prefilled_prompt_tokens = effective_prefilled_prompt_tokens;
            metrics_.prompt_suffix_tokens = prompt_tokens_to_forward;
            metrics_.generated_tokens = static_cast<int>(generated_ids.size());
            metrics_.prompt_build_ms = prompt_build_ms;
            metrics_.prompt_prefill_ms = prompt_prefill_ms;
            metrics_.first_token_delta_ms = first_token_delta_ms;
            metrics_.first_content_delta_ms = last_first_content_delta_ms_;
            metrics_.first_tts_enqueue_ms = last_first_tts_enqueue_ms_;
            metrics_.decode_ms = decode_ms;
            metrics_.model_ms = elapsed_ms(model_started);
        } else {
            last_generated_token_count_ += prompt_tokens_to_forward +
                                           static_cast<int>(generated_ids.size());
            generation_token_ids_.insert(generation_token_ids_.end(),
                                         prompt_ids.begin() + prefilled_prompt_tokens,
                                         prompt_ids.end());
            generation_token_ids_.insert(generation_token_ids_.end(),
                                         generated_ids.begin(),
                                         generated_ids.end());
        }
    }
    last_error_.clear();
    return true;
}

bool AliaForegroundPipeline::process_tool_calls(
    const std::string& raw_assistant_text,
    const std::string& user_text,
    AliaToolCallCallback tool_cb,
    void* user_data,
    std::string& spoken_text) {
    spoken_text.clear();

    aila::chat::AssistantChatResult parsed =
        aila::chat::parse_assistant_output(raw_assistant_text);
    spoken_text = strip_structured_artifacts_from_spoken_text(parsed.content);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        last_tool_call_json_.clear();
        last_tool_result_text_.clear();
        last_tool_resume_prompt_text_.clear();
    }

    if (parsed.tool_calls.empty()) {
        return true;
    }

    if (!tool_cb) {
        std::lock_guard<std::mutex> lock(mutex_);
        last_error_ = "foreground VLM emitted a tool call but no Alia tool callback was registered";
        return false;
    }

    std::string combined_call_json;
    std::string combined_result;
    for (size_t i = 0; i < parsed.tool_calls.size(); ++i) {
        const std::string call_json = aila::chat::tool_call_to_json(parsed.tool_calls[i]);
        std::vector<char> result_buffer(8192, '\0');
        const int callback_rc = tool_cb(call_json.c_str(),
                                        result_buffer.data(),
                                        static_cast<int>(result_buffer.size()),
                                        user_data);
        if (callback_rc != 0) {
            std::lock_guard<std::mutex> lock(mutex_);
            last_error_ = "Alia tool callback returned failure";
            return false;
        }

        if (!combined_call_json.empty()) {
            combined_call_json += "\n";
        }
        combined_call_json += call_json;

        if (!combined_result.empty()) {
            combined_result += "\n";
        }
        combined_result += result_buffer.data();
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        last_tool_resume_prompt_text_ =
            build_tool_resume_prompt_text(user_text, spoken_text,
                                          combined_call_json, combined_result);
        last_tool_call_json_ = std::move(combined_call_json);
        last_tool_result_text_ = std::move(combined_result);
        last_error_.clear();
    }
    return true;
}

bool AliaForegroundPipeline::abort_requested() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return abort_requested_;
}

bool AliaForegroundPipeline::is_busy_locked() const {
    return state_ == ForegroundTurnState::Running ||
           state_ == ForegroundTurnState::Aborting;
}

}  // namespace aila::alia
