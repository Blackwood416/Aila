#include "CpuQ35BackgroundExtractor.hpp"

#include "AliaBackgroundPipeline.hpp"
#include "../utils/ModelSpec.hpp"
#include "../utils/Tokenizer.hpp"
#include "../utils/EnvUtils.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <utility>
#include <vector>

namespace aila::alia {
namespace {

void set_error(std::string* error_message, const std::string& message) {
    if (error_message) {
        *error_message = message;
    }
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

int argmax_token(const std::vector<float>& logits, int vocab_size) {
    int best = 0;
    float best_value = logits.empty() ? 0.0f : logits[0];
    const int limit = std::min<int>(vocab_size, static_cast<int>(logits.size()));
    for (int i = 1; i < limit; ++i) {
        if (logits[static_cast<size_t>(i)] > best_value) {
            best_value = logits[static_cast<size_t>(i)];
            best = i;
        }
    }
    return best;
}

}  // namespace

CpuQ35BackgroundExtractor::CpuQ35BackgroundExtractor(std::string model_dir, int max_seq_len)
    : model_dir_(std::move(model_dir)),
      max_seq_len_(max_seq_len) {}

CpuQ35BackgroundExtractor::~CpuQ35BackgroundExtractor() = default;

bool CpuQ35BackgroundExtractor::load(std::string* error_message) {
    loaded_ = false;
    tokenizer_.reset();
    model_.reset();
    last_error_.clear();

    if (model_dir_.empty()) {
        last_error_ = "native CPU Qwen3.5 background model dir is empty";
        set_error(error_message, last_error_);
        return false;
    }
    if (max_seq_len_ <= 0) {
        last_error_ = "native CPU Qwen3.5 max_seq_len must be positive";
        set_error(error_message, last_error_);
        return false;
    }

    ModelSpec loaded_spec;
    std::string spec_error;
    if (!aila::modelspec::load_from_dir(model_dir_, loaded_spec, &spec_error)) {
        last_error_ = spec_error.empty()
            ? "native CPU Qwen3.5 model spec load failed"
            : "native CPU Qwen3.5 model spec load failed: " + spec_error;
        set_error(error_message, last_error_);
        return false;
    }
    if (loaded_spec.family != ModelFamily::Qwen35Hybrid) {
        last_error_ = "native CPU background extractor requires a Qwen3.5 hybrid model";
        set_error(error_message, last_error_);
        return false;
    }
    if (!loaded_spec.is_bitsandbytes_4bit()) {
        last_error_ = "native CPU background extractor requires a bitsandbytes 4-bit checkpoint";
        set_error(error_message, last_error_);
        return false;
    }
    if (!is_supported_qwen35_hybrid_0p8b_spec(loaded_spec.qwen35_text)) {
        last_error_ = "native CPU background extractor currently supports only Qwen3.5 0.8B";
        set_error(error_message, last_error_);
        return false;
    }

    auto tokenizer = std::make_unique<Tokenizer>();
    if (!tokenizer->load(model_dir_)) {
        last_error_ = "native CPU Qwen3.5 tokenizer load failed: " + model_dir_;
        set_error(error_message, last_error_);
        return false;
    }

    auto model = std::make_unique<CpuQ35HybridModel>();
    std::string model_error;
    if (!model->load(model_dir_, loaded_spec, max_seq_len_, &model_error)) {
        last_error_ = model_error.empty()
            ? "native CPU Qwen3.5 model load failed"
            : "native CPU Qwen3.5 model load failed: " + model_error;
        set_error(error_message, last_error_);
        return false;
    }

    spec_ = loaded_spec;
    tokenizer_ = std::move(tokenizer);
    model_ = std::move(model);
    loaded_ = true;
    set_error(error_message, "");
    return true;
}

BackgroundExtractionResult CpuQ35BackgroundExtractor::extract(
    const BackgroundExtractionRequest& request,
    const std::atomic_bool& abort_requested) {
    BackgroundExtractionResult out;
    if (abort_requested.load()) {
        out.error = "native CPU Qwen3.5 background extraction aborted";
        return out;
    }
    if (!loaded_) {
        out.error = "native CPU Qwen3.5 background extractor is not loaded";
        return out;
    }

    const std::string prompt_text =
        build_background_extraction_prompt(request.chat_turn_text);
    std::string result_json;
    std::string error;
    if (!generate_once(prompt_text, abort_requested, result_json, error)) {
        out.error = error.empty() ? "native CPU Qwen3.5 background generation failed" : error;
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
                ? "native CPU Qwen3.5 schema repair generation failed"
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

bool CpuQ35BackgroundExtractor::generate_once(
    const std::string& prompt_text,
    const std::atomic_bool& abort_requested,
    std::string& result_json,
    std::string& error) {
    result_json.clear();
    error.clear();
    if (!loaded_ || !tokenizer_ || !model_) {
        error = "native CPU Qwen3.5 background extractor is not loaded";
        return false;
    }

    std::vector<int> prompt_ids =
        apply_alia_chat_template(tokenizer_.get(), background_system_prompt(), prompt_text);
    if (prompt_ids.empty()) {
        error = "native CPU Qwen3.5 background prompt encoded to zero tokens";
        return false;
    }

    const int kMaxNewTokens =
        std::max(1, aila::env::read_int("AILA_CPU_Q35_BACKGROUND_MAX_NEW_TOKENS", 384));
    if (max_seq_len_ > 0 &&
        static_cast<int>(prompt_ids.size()) + kMaxNewTokens > max_seq_len_) {
        error = "native CPU Qwen3.5 background prompt would exceed max sequence length";
        return false;
    }

    const int vocab_size = tokenizer_->vocab_size();
    if (vocab_size <= 0) {
        error = "native CPU Qwen3.5 tokenizer vocab size is invalid";
        return false;
    }

    model_->reset();
    std::vector<float> logits;
    const bool profile_cpu =
        aila::env::read_flag("AILA_PROFILE_CPU_Q35_BACKGROUND", false);
    const auto prompt_t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < prompt_ids.size(); ++i) {
        if (abort_requested.load()) {
            error = "native CPU Qwen3.5 background extraction aborted";
            return false;
        }
        const int token_id = prompt_ids[i];
        const bool is_last_prompt_token = (i + 1 == prompt_ids.size());
        const bool ok = is_last_prompt_token
            ? model_->forward_one(token_id, logits, &error)
            : model_->consume_one(token_id, &error);
        if (!ok) {
            return false;
        }
        if (profile_cpu && (((i + 1) % 16) == 0 || i + 1 == prompt_ids.size())) {
            const auto now = std::chrono::high_resolution_clock::now();
            const double ms =
                std::chrono::duration<double, std::milli>(now - prompt_t0).count();
            std::cout << "cpu_background_prompt_tokens=" << (i + 1)
                      << " cpu_background_prompt_ms=" << static_cast<int>(ms)
                      << "\n";
        }
    }

    std::vector<int> generated_ids;
    generated_ids.reserve(kMaxNewTokens);
    const auto decode_t0 = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < kMaxNewTokens; ++step) {
        if (abort_requested.load()) {
            break;
        }

        const int next_token = argmax_token(logits, vocab_size);
        if (tokenizer_->is_eos(next_token)) {
            break;
        }

        generated_ids.push_back(next_token);
        if (!model_->forward_one(next_token, logits, &error)) {
            return false;
        }
        if (profile_cpu && (((step + 1) % 8) == 0 || step + 1 == kMaxNewTokens)) {
            const auto now = std::chrono::high_resolution_clock::now();
            const double ms =
                std::chrono::duration<double, std::milli>(now - decode_t0).count();
            std::cout << "cpu_background_generated_tokens=" << (step + 1)
                      << " cpu_background_decode_ms=" << static_cast<int>(ms)
                      << "\n";
        }
    }

    result_json = tokenizer_->decode(generated_ids);
    return true;
}

}  // namespace aila::alia
