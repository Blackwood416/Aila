#include "BackgroundExtractorFactory.hpp"

#include "CpuQ35BackgroundExtractor.hpp"
#include "GpuVlmBackgroundExtractor.hpp"
#include "../utils/EnvUtils.hpp"

#include <algorithm>
#include <cctype>
#include <utility>

namespace aila::alia {
namespace {

std::string normalize_kind_value(std::string value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) {
        value.erase(value.begin());
    }
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) {
        value.pop_back();
    }
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        char out = static_cast<char>(std::tolower(ch));
        return out == '_' ? '-' : out;
    });
    return value;
}

void set_error(std::string* error_message, const std::string& message) {
    if (error_message) {
        *error_message = message;
    }
}

}  // namespace

BackgroundExtractorKind background_extractor_kind_from_string(std::string value) {
    value = normalize_kind_value(std::move(value));
    if (value == "cpu" ||
        value == "native-cpu" ||
        value == "native-cpu-q35" ||
        value == "native-cpu-qwen35") {
        return BackgroundExtractorKind::NativeCpuQ35;
    }
    return BackgroundExtractorKind::GpuLoadedVlm;
}

BackgroundExtractorKind read_background_extractor_kind_from_env() {
    return background_extractor_kind_from_string(
        aila::env::read_string("AILA_BACKGROUND_EXTRACTOR", "gpu"));
}

const char* background_extractor_kind_name(BackgroundExtractorKind kind) {
    switch (kind) {
        case BackgroundExtractorKind::GpuLoadedVlm:
            return "gpu";
        case BackgroundExtractorKind::NativeCpuQ35:
            return "native-cpu-q35";
    }
    return "gpu";
}

bool should_load_gpu_model_slot(ModelRole role, BackgroundExtractorKind background_kind) {
    return role != ModelRole::BackgroundVlm ||
           background_kind == BackgroundExtractorKind::GpuLoadedVlm;
}

std::unique_ptr<IBackgroundMemoryExtractor> create_background_memory_extractor(
    BackgroundExtractorKind kind,
    ModelSlot* background_slot,
    const std::string& background_model_dir,
    int max_seq_len,
    std::string* error_message) {
    if (kind == BackgroundExtractorKind::GpuLoadedVlm) {
        set_error(error_message, "");
        return std::make_unique<GpuVlmBackgroundExtractor>(background_slot);
    }

    auto extractor =
        std::make_unique<CpuQ35BackgroundExtractor>(background_model_dir, max_seq_len);
    std::string load_error;
    if (!extractor->load(&load_error)) {
        set_error(error_message, load_error.empty()
            ? "native CPU background extractor load failed"
            : load_error);
        return nullptr;
    }
    set_error(error_message, "");
    return extractor;
}

}  // namespace aila::alia
