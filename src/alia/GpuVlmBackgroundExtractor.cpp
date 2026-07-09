#include "GpuVlmBackgroundExtractor.hpp"

#include "AliaBackgroundPipeline.hpp"
#include "ModelSlot.hpp"
#include "../models/IModelBackend.hpp"
#include "../ops/Ops.hpp"
#include "../utils/Tokenizer.hpp"

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

}  // namespace

GpuVlmBackgroundExtractor::GpuVlmBackgroundExtractor(ModelSlot* slot)
    : slot_(slot) {}

bool GpuVlmBackgroundExtractor::ready() const {
    return slot_ &&
           slot_->state() == ModelSlotState::Loaded &&
           slot_->context() &&
           slot_->tokenizer() &&
           slot_->backend();
}

BackgroundExtractionResult GpuVlmBackgroundExtractor::extract(
    const BackgroundExtractionRequest& request,
    const std::atomic_bool& abort_requested) {
    BackgroundExtractionResult out;
    const std::string prompt_text =
        build_background_extraction_prompt(request.chat_turn_text);
    std::string result_json;
    std::string error;
    if (!generate_once(prompt_text, abort_requested, result_json, error)) {
        out.error = error.empty() ? "background VLM generation failed" : error;
        return out;
    }

    result_json = normalize_background_model_json(result_json);
    std::string final_prompt_text = prompt_text;
    std::string schema_diagnostic;
    int retry_count = 0;
    bool schema_repair_applied = false;

    if (!AliaBackgroundPipeline::has_required_schema_keys(result_json)) {
        if (abort_requested.load()) {
            out.ok = true;
            out.result_json = result_json;
            out.prompt_text = final_prompt_text;
            return out;
        }

        const std::string repair_prompt =
            build_background_schema_repair_prompt(request.chat_turn_text, result_json);
        retry_count = 1;
        final_prompt_text = repair_prompt;
        std::string repaired_json;
        if (!generate_once(repair_prompt, abort_requested, repaired_json, error)) {
            out.error = error.empty()
                ? "background VLM schema repair generation failed"
                : error;
            return out;
        }
        result_json = normalize_background_model_json(repaired_json);
        schema_diagnostic = AliaBackgroundPipeline::has_required_schema_keys(result_json)
            ? "retry accepted schema-valid background JSON"
            : "retry failed schema validation; applying schema repair wrapper";
    } else {
        schema_diagnostic = "initial background JSON accepted";
    }

    schema_repair_applied = !AliaBackgroundPipeline::has_required_schema_keys(result_json);
    result_json = enforce_background_result_schema(result_json, request.chat_turn_text);
    if (!schema_repair_applied) {
        const std::string cleaned_json =
            cleanup_background_result_json(result_json, request.chat_turn_text);
        if (cleaned_json != result_json &&
            AliaBackgroundPipeline::has_required_schema_keys(cleaned_json)) {
            result_json = cleaned_json;
            if (!schema_diagnostic.empty()) {
                schema_diagnostic += "; post cleanup applied";
            } else {
                schema_diagnostic = "post cleanup applied";
            }
        }
    }

    out.ok = true;
    out.result_json = result_json;
    out.prompt_text = final_prompt_text;
    out.schema_retry_count = retry_count;
    out.schema_repair_applied = schema_repair_applied;
    out.schema_diagnostic = schema_diagnostic;
    return out;
}

bool GpuVlmBackgroundExtractor::generate_once(
    const std::string& prompt_text,
    const std::atomic_bool& abort_requested,
    std::string& result_json,
    std::string& error) {
    result_json.clear();
    error.clear();
    if (!ready()) {
        error = "background VLM slot is not loaded";
        return false;
    }

    Context* context = slot_->context();
    Tokenizer* tokenizer = slot_->tokenizer();
    IModelBackend* backend = slot_->backend();
    if (!context || !tokenizer || !backend) {
        error = "background VLM slot is incomplete";
        return false;
    }

    std::vector<int> prompt_ids =
        apply_alia_chat_template(tokenizer, background_system_prompt(), prompt_text);
    if (prompt_ids.empty()) {
        error = "background VLM prompt encoded to zero tokens";
        return false;
    }

    const int max_new_tokens = 384;
    const int max_seq_len = backend->max_seq_len();
    if (max_seq_len > 0 &&
        static_cast<int>(prompt_ids.size()) + max_new_tokens > max_seq_len) {
        error = "background VLM prompt would exceed max sequence length";
        return false;
    }

    const int vocab_size = backend->vocab_size() > 0
        ? backend->vocab_size()
        : tokenizer->vocab_size();
    if (vocab_size <= 0) {
        error = "background VLM vocab size is invalid";
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
        if (abort_requested.load()) {
            break;
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
    return true;
}

}  // namespace aila::alia
