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

bool is_tts_chunk_boundary(char ch) {
    return ch == '.' || ch == '!' || ch == '?' || ch == ';' || ch == '\n';
}

std::vector<std::string> take_ready_tts_chunks(std::string& buffer, bool force) {
    size_t cutoff = std::string::npos;
    if (force) {
        cutoff = buffer.size();
    } else {
        for (size_t i = buffer.size(); i > 0; --i) {
            if (is_tts_chunk_boundary(buffer[i - 1])) {
                cutoff = i;
                break;
            }
        }
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
           "interaction. When an external action is required, emit a compact "
           "JSON tool call and wait for the host tool result before continuing.";
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
        if (ch == '.' || ch == '!' || ch == '?' || ch == ';' || ch == '\n') {
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

    IModelBackend* backend = vlm_slot_->backend();
    const int current_len = backend->get_current_context_len();
    int anchor = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        anchor = generation_start_context_len_;
    }

    int target_len = std::max(anchor, current_len - rollback_tokens);
    if (target_len > current_len) {
        target_len = current_len;
    }

    if (!backend->truncate_kv_cache(target_len)) {
        return ALIA_ERR_RUNTIME;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        last_generated_token_count_ = std::max(0, target_len - generation_start_context_len_);
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

void AliaForegroundPipeline::run_turn(
    AliaGenConfig config,
    AliaToolCallCallback tool_cb,
    AliaAudioCallback audio_cb,
    void* user_data) {
    (void)tool_cb;
    (void)vlm_slot_;

    try {
        GenerationConfig generation_config = translate_generation_config(&config);
        std::string stable_text;
        std::string partial_text;
        if (asr_pipeline_) {
            asr_pipeline_->get_text(stable_text, partial_text);
        }
        const std::string user_text = !stable_text.empty() ? stable_text : partial_text;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            last_generation_config_ = generation_config;
            last_user_text_ = user_text;
            last_assistant_text_.clear();
            last_tool_call_json_.clear();
            last_tool_result_text_.clear();
            last_tool_resume_prompt_text_.clear();
            generation_start_context_len_ = -1;
            last_generated_token_count_ = 0;
            last_decode_mode_ = ForegroundDecodeMode::None;
            if (abort_requested_) {
                state_ = ForegroundTurnState::Aborted;
                cv_.notify_all();
                return;
            }
        }

        std::string assistant_text;
        ForegroundDecodeMode decode_mode = ForegroundDecodeMode::NoModelFallback;
        if (can_generate_with_loaded_vlm()) {
            decode_mode = ForegroundDecodeMode::LoadedVlm;
            if (!generate_with_loaded_vlm(user_text, generation_config, assistant_text,
                                          true, true, true, true,
                                          &config, audio_cb, user_data)) {
                std::lock_guard<std::mutex> lock(mutex_);
                if (last_error_.empty()) {
                    last_error_ = "foreground VLM generation failed";
                }
                state_ = ForegroundTurnState::Failed;
                cv_.notify_all();
                return;
            }
        } else {
            assistant_text = user_text.empty() ? "Alia foreground turn" : user_text;
        }

        std::string spoken_text;
        if (!process_tool_calls(assistant_text, user_text, tool_cb, user_data, spoken_text)) {
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
                                              &config, audio_cb, user_data)) {
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

        if (decode_mode != ForegroundDecodeMode::LoadedVlm &&
            tts_pipeline_ && audio_cb && !spoken_text.empty()) {
            synthesize_spoken_text(spoken_text, config, audio_cb, user_data);
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            state_ = abort_requested_ ? ForegroundTurnState::Aborted
                                      : ForegroundTurnState::Completed;
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
    const AliaGenConfig* tts_config,
    AliaAudioCallback audio_cb,
    void* user_data) {
    assistant_text.clear();
    if (!can_generate_with_loaded_vlm()) {
        return false;
    }

    Context* context = vlm_slot_->context();
    Tokenizer* tokenizer = vlm_slot_->tokenizer();
    IModelBackend* backend = vlm_slot_->backend();
    if (!context || !tokenizer || !backend) {
        last_error_ = "foreground VLM slot is incomplete";
        return false;
    }

    const std::string prompt_text = user_text.empty()
        ? "Continue the current conversation."
        : user_text;
    std::vector<int> prompt_ids = use_chat_template
        ? tokenizer->apply_chat_template(foreground_system_prompt(), prompt_text)
        : tokenizer->encode(prompt_text);
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

    if (reset_session) {
        backend->reset();
    }

    const int max_seq_len = backend->max_seq_len();
    const int context_len_before_prompt = reset_session ? 0 : backend->get_current_context_len();
    if (max_seq_len > 0 &&
        context_len_before_prompt + static_cast<int>(prompt_ids.size()) +
            config.max_new_tokens > max_seq_len) {
        last_error_ = use_chat_template
            ? "foreground VLM prompt would exceed max sequence length"
            : "foreground VLM continuation would exceed max sequence length";
        return false;
    }

    DeviceAllocation prompt_device(*context, prompt_ids.size() * sizeof(int));
    context->memcpy_h2d(prompt_device.as<int>(), prompt_ids.data(),
                        prompt_ids.size() * sizeof(int));
    Tensor* logits = &backend->forward(*context, prompt_device.as<int>(),
                                       static_cast<int>(prompt_ids.size()));

    if (record_generation_anchor) {
        std::lock_guard<std::mutex> lock(mutex_);
        generation_start_context_len_ = backend->get_current_context_len();
        last_generated_token_count_ = 0;
    }

    DeviceAllocation one_token_device(*context, sizeof(int));
    std::vector<int> generated_ids;
    aila::chat::StructuredStreamParser stream_parser;
    std::string pending_tts_text;
    bool paused_on_tool_call = false;
    auto flush_tts = [&](bool force) {
        if (!tts_pipeline_ || !audio_cb || !tts_config) {
            return;
        }
        for (const std::string& chunk : take_ready_tts_chunks(pending_tts_text, force)) {
            if (abort_requested()) {
                return;
            }
            if (tts_pipeline_->enqueue_text(chunk)) {
                tts_pipeline_->synthesize_pending(
                    *tts_config,
                    audio_cb,
                    user_data,
                    [this]() { return abort_requested(); });
            }
        }
    };

    const int max_new_tokens = std::max(config.max_new_tokens, 1);
    for (int step = 0; step < max_new_tokens; ++step) {
        if (abort_requested()) {
            break;
        }

        int next_token = ops::sample_with_config(
            *context, *logits, vocab_size, config, generated_ids);
        if (tokenizer->is_eos(next_token)) {
            break;
        }

        generated_ids.push_back(next_token);
        bool complete_tool_call = false;
        std::vector<aila::chat::StructuredStreamEvent> events;
        stream_parser.push(tokenizer->decode(next_token), events);
        for (const auto& event : events) {
            if (event.type == aila::chat::StructuredStreamEventType::ContentDelta) {
                pending_tts_text += event.text;
            }
            if (event.type == aila::chat::StructuredStreamEventType::ToolCallDelta &&
                !event.tool_calls.empty()) {
                complete_tool_call = true;
            }
        }
        context->memcpy_h2d(one_token_device.as<int>(), &next_token, sizeof(int));
        logits = &backend->forward(*context, one_token_device.as<int>(), 1);
        flush_tts(false);
        if (stop_on_tool_call && complete_tool_call) {
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
            }
        }
        flush_tts(true);
    }

    assistant_text = tokenizer->decode(generated_ids);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (record_generation_anchor) {
            last_generated_token_count_ = static_cast<int>(generated_ids.size());
        } else {
            last_generated_token_count_ += static_cast<int>(generated_ids.size());
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
    spoken_text = parsed.content;

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

void AliaForegroundPipeline::synthesize_spoken_text(
    const std::string& spoken_text,
    const AliaGenConfig& config,
    AliaAudioCallback audio_cb,
    void* user_data) {
    if (!tts_pipeline_ || !audio_cb) {
        return;
    }

    for (const std::string& chunk : split_spoken_text_for_tts(spoken_text)) {
        if (abort_requested()) {
            break;
        }
        if (tts_pipeline_->enqueue_text(chunk)) {
            tts_pipeline_->synthesize_pending(
                config,
                audio_cb,
                user_data,
                [this]() { return abort_requested(); });
        }
    }
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
