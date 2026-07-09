#include "CpuQ35BackgroundExtractor.hpp"

#include "../utils/ModelSpec.hpp"
#include "../utils/Tokenizer.hpp"

#include <utility>

namespace aila::alia {
namespace {

void set_error(std::string* error_message, const std::string& message) {
    if (error_message) {
        *error_message = message;
    }
}

}  // namespace

CpuQ35BackgroundExtractor::CpuQ35BackgroundExtractor(std::string model_dir, int max_seq_len)
    : model_dir_(std::move(model_dir)),
      max_seq_len_(max_seq_len) {}

CpuQ35BackgroundExtractor::~CpuQ35BackgroundExtractor() = default;

bool CpuQ35BackgroundExtractor::load(std::string* error_message) {
    loaded_ = false;
    tokenizer_.reset();
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

    spec_ = loaded_spec;
    tokenizer_ = std::move(tokenizer);
    loaded_ = true;
    set_error(error_message, "");
    return true;
}

BackgroundExtractionResult CpuQ35BackgroundExtractor::extract(
    const BackgroundExtractionRequest&,
    const std::atomic_bool& abort_requested) {
    BackgroundExtractionResult result;
    if (abort_requested.load()) {
        result.error = "native CPU Qwen3.5 background extraction aborted";
        return result;
    }
    if (!loaded_) {
        result.error = "native CPU Qwen3.5 background extractor is not loaded";
        return result;
    }

    result.error = "native CPU Qwen3.5 background extractor inference is not implemented";
    return result;
}

}  // namespace aila::alia
