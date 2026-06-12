#include "AliaBackgroundPipeline.hpp"

#include "ModelSlot.hpp"
#include "../models/IModelBackend.hpp"
#include "../ops/Ops.hpp"
#include "../utils/Tokenizer.hpp"

#include "simdjson.h"

#include <algorithm>
#include <cctype>
#include <exception>
#include <sstream>
#include <utility>

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

bool has_string_field(simdjson::dom::object object, const char* key) {
    simdjson::dom::element element;
    std::string_view value;
    return object.at_key(key).get(element) == simdjson::SUCCESS &&
           element.get_string().get(value) == simdjson::SUCCESS;
}

bool has_array_field(simdjson::dom::object object, const char* key) {
    simdjson::dom::element element;
    simdjson::dom::array value;
    return object.at_key(key).get(element) == simdjson::SUCCESS &&
           element.get_array().get(value) == simdjson::SUCCESS;
}

std::vector<int> apply_alia_chat_template(
    Tokenizer* tokenizer,
    const std::string& system_prompt,
    const std::string& user_message) {
    std::vector<int> prompt_ids =
        tokenizer->apply_chat_template(system_prompt, user_message);
    const std::vector<int> closed_think_ids =
        tokenizer->encode("<think>\n\n</think>\n\n");
    prompt_ids.insert(prompt_ids.end(),
                      closed_think_ids.begin(),
                      closed_think_ids.end());
    return prompt_ids;
}

std::string trim(std::string value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) {
        value.erase(0, 1);
    }
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) {
        value.pop_back();
    }
    return value;
}

std::string strip_markdown_json_fence(const std::string& value) {
    std::string trimmed = trim(value);
    if (trimmed.rfind("```", 0) != 0) {
        return trimmed;
    }

    const size_t first_newline = trimmed.find('\n');
    if (first_newline == std::string::npos) {
        return trimmed;
    }

    size_t body_begin = first_newline + 1;
    size_t body_end = trimmed.rfind("```");
    if (body_end == std::string::npos || body_end <= body_begin) {
        body_end = trimmed.size();
    }
    return trim(trimmed.substr(body_begin, body_end - body_begin));
}

}  // namespace

std::string background_system_prompt() {
    return "You are Alia's local background memory extraction model. Read one "
           "conversation turn and return strict JSON only. Extract durable "
           "memory candidates, user preferences, unresolved tasks, and a short "
           "summary. Do not include prose outside JSON.";
}

std::string build_background_extraction_prompt(const std::string& chat_turn_text) {
    std::string prompt;
    prompt += "Conversation turn text:\n";
    prompt += chat_turn_text;
    prompt += "\n\nReturn JSON with keys: summary, memory_candidates, preferences, tasks.";
    return prompt;
}

std::string build_background_schema_repair_prompt(const std::string& chat_turn_text,
                                                  const std::string& invalid_output) {
    std::string prompt;
    prompt += "Repair the previous background memory extraction output for Alia.\n";
    prompt += "Conversation turn text:\n";
    prompt += chat_turn_text;
    prompt += "\n\nPrevious invalid output:\n";
    prompt += invalid_output.empty() ? "(empty)" : invalid_output;
    prompt += "\n\nReturn strict JSON only with this schema: ";
    prompt += "{\"summary\":\"string\",\"memory_candidates\":[],\"preferences\":[],\"tasks\":[]}";
    return prompt;
}

std::string make_background_fallback_json(const std::string& chat_turn_text) {
    return std::string("{") +
        "\"summary\":\"" + AliaBackgroundPipeline::json_escape(chat_turn_text) + "\"," +
        "\"memory_candidates\":[]," +
        "\"preferences\":[]," +
        "\"tasks\":[]," +
        "\"source_turn_text\":\"" + AliaBackgroundPipeline::json_escape(chat_turn_text) + "\"," +
        "\"source\":\"no_model_fallback\"" +
        "}";
}

std::string enforce_background_result_schema(const std::string& raw_result_json,
                                             const std::string& chat_turn_text) {
    if (AliaBackgroundPipeline::has_required_schema_keys(raw_result_json)) {
        return raw_result_json;
    }

    return std::string("{") +
        "\"summary\":\"" + AliaBackgroundPipeline::json_escape(chat_turn_text) + "\"," +
        "\"memory_candidates\":[]," +
        "\"preferences\":[]," +
        "\"tasks\":[]," +
        "\"raw_model_output\":\"" + AliaBackgroundPipeline::json_escape(raw_result_json) + "\"," +
        "\"source\":\"schema_repair\"" +
        "}";
}

std::string normalize_background_model_json(const std::string& raw_result_json) {
    if (AliaBackgroundPipeline::has_required_schema_keys(raw_result_json)) {
        return raw_result_json;
    }

    const std::string unfenced = strip_markdown_json_fence(raw_result_json);
    if (AliaBackgroundPipeline::has_required_schema_keys(unfenced)) {
        return unfenced;
    }

    const size_t begin = unfenced.find('{');
    const size_t end = unfenced.rfind('}');
    if (begin != std::string::npos && end != std::string::npos && begin < end) {
        const std::string object_candidate = unfenced.substr(begin, end - begin + 1);
        if (AliaBackgroundPipeline::has_required_schema_keys(object_candidate)) {
            return object_candidate;
        }
    }

    return raw_result_json;
}

AliaBackgroundPipeline::AliaBackgroundPipeline(ModelSlot* slot)
    : slot_(slot) {}

AliaBackgroundPipeline::~AliaBackgroundPipeline() {
    request_abort();
    join();
}

void AliaBackgroundPipeline::register_callback(AliaBackgroundResultCallback callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    callback_ = callback;
}

bool AliaBackgroundPipeline::trigger(std::string chat_turn_text) {
    AliaBackgroundResultCallback callback = nullptr;

    std::unique_lock<std::mutex> lock(mutex_);
    if (is_busy_locked()) {
        return false;
    }

    if (worker_.joinable()) {
        lock.unlock();
        worker_.join();
        lock.lock();
    }

    callback = callback_;
    if (!callback) {
        last_error_ = "background callback is not registered";
        return false;
    }

    abort_requested_ = false;
    last_error_.clear();
    last_prompt_text_.clear();
    last_result_json_.clear();
    last_schema_retry_count_ = 0;
    last_schema_repair_applied_ = false;
    last_schema_diagnostic_.clear();
    last_decode_mode_ = BackgroundDecodeMode::None;
    state_ = BackgroundJobState::Running;
    worker_ = std::thread([this, text = std::move(chat_turn_text), callback]() mutable {
        run_job(std::move(text), callback);
    });
    return true;
}

void AliaBackgroundPipeline::request_abort() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        abort_requested_ = true;
        if (state_ == BackgroundJobState::Running) {
            state_ = BackgroundJobState::Aborting;
        }
    }
    cv_.notify_all();
}

void AliaBackgroundPipeline::join() {
    if (worker_.joinable()) {
        worker_.join();
    }
}

BackgroundJobState AliaBackgroundPipeline::state() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_;
}

bool AliaBackgroundPipeline::wait_until_idle_for(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(lock, timeout, [this]() { return !is_busy_locked(); });
}

BackgroundDecodeMode AliaBackgroundPipeline::last_decode_mode() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_decode_mode_;
}

std::string AliaBackgroundPipeline::last_prompt_text() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_prompt_text_;
}

std::string AliaBackgroundPipeline::last_result_json() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_result_json_;
}

int AliaBackgroundPipeline::last_schema_retry_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_schema_retry_count_;
}

bool AliaBackgroundPipeline::last_schema_repair_applied() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_schema_repair_applied_;
}

std::string AliaBackgroundPipeline::last_schema_diagnostic() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_schema_diagnostic_;
}

void AliaBackgroundPipeline::run_job(
    std::string chat_turn_text,
    AliaBackgroundResultCallback callback) {
    try {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (abort_requested_) {
                state_ = BackgroundJobState::Completed;
                cv_.notify_all();
                return;
            }
        }

        const std::string prompt_text = build_background_extraction_prompt(chat_turn_text);
        std::string final_prompt_text = prompt_text;
        std::string result_json;
        int schema_retry_count = 0;
        bool schema_repair_applied = false;
        std::string schema_diagnostic;
        BackgroundDecodeMode decode_mode = BackgroundDecodeMode::NoModelFallback;
        if (can_generate_with_loaded_vlm()) {
            decode_mode = BackgroundDecodeMode::LoadedVlm;
            if (!generate_with_loaded_vlm(prompt_text, result_json)) {
                std::lock_guard<std::mutex> lock(mutex_);
                if (last_error_.empty()) {
                    last_error_ = "background VLM generation failed";
                }
                state_ = BackgroundJobState::Failed;
                cv_.notify_all();
                return;
            }
            result_json = normalize_background_model_json(result_json);
            if (!AliaBackgroundPipeline::has_required_schema_keys(result_json)) {
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (abort_requested_) {
                        state_ = BackgroundJobState::Completed;
                        cv_.notify_all();
                        return;
                    }
                }

                const std::string repair_prompt =
                    build_background_schema_repair_prompt(chat_turn_text, result_json);
                schema_retry_count = 1;
                final_prompt_text = repair_prompt;
                std::string repaired_json;
                if (!generate_with_loaded_vlm(repair_prompt, repaired_json)) {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (last_error_.empty()) {
                        last_error_ = "background VLM schema repair generation failed";
                    }
                    state_ = BackgroundJobState::Failed;
                    cv_.notify_all();
                    return;
                }
                result_json = normalize_background_model_json(repaired_json);
                schema_diagnostic = AliaBackgroundPipeline::has_required_schema_keys(result_json)
                    ? "retry accepted schema-valid background JSON"
                    : "retry failed schema validation; applying schema repair wrapper";
            } else {
                schema_diagnostic = "initial background JSON accepted";
            }
        } else {
            result_json = make_background_fallback_json(chat_turn_text);
            schema_diagnostic = "no-model fallback produced schema-valid background JSON";
        }
        schema_repair_applied = !AliaBackgroundPipeline::has_required_schema_keys(result_json);
        result_json = enforce_background_result_schema(result_json, chat_turn_text);

        {
            std::lock_guard<std::mutex> lock(mutex_);
            last_prompt_text_ = final_prompt_text;
            last_result_json_ = result_json;
            last_schema_retry_count_ = schema_retry_count;
            last_schema_repair_applied_ = schema_repair_applied;
            last_schema_diagnostic_ = schema_diagnostic;
            last_decode_mode_ = decode_mode;
            if (abort_requested_) {
                state_ = BackgroundJobState::Completed;
                cv_.notify_all();
                return;
            }
        }

        callback(result_json.c_str(), nullptr);

        {
            std::lock_guard<std::mutex> lock(mutex_);
            state_ = BackgroundJobState::Completed;
        }
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(mutex_);
        last_error_ = e.what();
        state_ = BackgroundJobState::Failed;
    } catch (...) {
        std::lock_guard<std::mutex> lock(mutex_);
        last_error_ = "unknown background pipeline failure";
        state_ = BackgroundJobState::Failed;
    }

    cv_.notify_all();
}

bool AliaBackgroundPipeline::can_generate_with_loaded_vlm() const {
    return slot_ &&
           slot_->state() == ModelSlotState::Loaded &&
           slot_->context() &&
           slot_->tokenizer() &&
           slot_->backend();
}

bool AliaBackgroundPipeline::generate_with_loaded_vlm(
    const std::string& prompt_text,
    std::string& result_json) {
    result_json.clear();
    if (!can_generate_with_loaded_vlm()) {
        return false;
    }

    Context* context = slot_->context();
    Tokenizer* tokenizer = slot_->tokenizer();
    IModelBackend* backend = slot_->backend();
    if (!context || !tokenizer || !backend) {
        last_error_ = "background VLM slot is incomplete";
        return false;
    }

    std::vector<int> prompt_ids =
        apply_alia_chat_template(tokenizer, background_system_prompt(), prompt_text);
    if (prompt_ids.empty()) {
        last_error_ = "background VLM prompt encoded to zero tokens";
        return false;
    }

    const int max_new_tokens = 384;
    const int max_seq_len = backend->max_seq_len();
    if (max_seq_len > 0 &&
        static_cast<int>(prompt_ids.size()) + max_new_tokens > max_seq_len) {
        last_error_ = "background VLM prompt would exceed max sequence length";
        return false;
    }

    const int vocab_size = backend->vocab_size() > 0
        ? backend->vocab_size()
        : tokenizer->vocab_size();
    if (vocab_size <= 0) {
        last_error_ = "background VLM vocab size is invalid";
        return false;
    }

    GenerationConfig config;
    config.max_new_tokens = max_new_tokens;
    config.temperature = 0.0f;
    config.top_p = 1.0f;
    config.do_sample = false;

    backend->reset();
    DeviceAllocation prompt_device(*context, prompt_ids.size() * sizeof(int));
    context->memcpy_h2d(prompt_device.as<int>(), prompt_ids.data(),
                        prompt_ids.size() * sizeof(int));
    Tensor* logits = &backend->forward(*context, prompt_device.as<int>(),
                                       static_cast<int>(prompt_ids.size()));

    DeviceAllocation one_token_device(*context, sizeof(int));
    std::vector<int> generated_ids;
    for (int step = 0; step < config.max_new_tokens; ++step) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (abort_requested_) {
                break;
            }
        }

        int next_token = ops::sample_with_config(
            *context, *logits, vocab_size, config, generated_ids);
        if (tokenizer->is_eos(next_token)) {
            break;
        }

        generated_ids.push_back(next_token);
        context->memcpy_h2d(one_token_device.as<int>(), &next_token, sizeof(int));
        logits = &backend->forward(*context, one_token_device.as<int>(), 1);
    }

    result_json = tokenizer->decode(generated_ids);
    last_error_.clear();
    return true;
}

bool AliaBackgroundPipeline::is_busy_locked() const {
    return state_ == BackgroundJobState::Running ||
           state_ == BackgroundJobState::Aborting;
}

std::string AliaBackgroundPipeline::json_escape(const std::string& value) {
    std::ostringstream out;
    for (char ch : value) {
        switch (ch) {
            case '\\':
                out << "\\\\";
                break;
            case '"':
                out << "\\\"";
                break;
            case '\n':
                out << "\\n";
                break;
            case '\r':
                out << "\\r";
                break;
            case '\t':
                out << "\\t";
                break;
            default:
                out << ch;
                break;
        }
    }
    return out.str();
}

bool AliaBackgroundPipeline::has_required_schema_keys(const std::string& value) {
    simdjson::dom::parser parser;
    simdjson::dom::element root;
    if (parser.parse(value).get(root) != simdjson::SUCCESS) {
        return false;
    }

    simdjson::dom::object object;
    if (root.get_object().get(object) != simdjson::SUCCESS) {
        return false;
    }

    return has_string_field(object, "summary") &&
           has_array_field(object, "memory_candidates") &&
           has_array_field(object, "preferences") &&
           has_array_field(object, "tasks");
}

}  // namespace aila::alia
