#pragma once

#include "BackgroundMemoryExtractor.hpp"
#include "ModelSlot.hpp"

#include <memory>
#include <string>

namespace aila::alia {

enum class BackgroundExtractorKind {
    GpuLoadedVlm,
    NativeCpuQ35
};

BackgroundExtractorKind background_extractor_kind_from_string(std::string value);
BackgroundExtractorKind read_background_extractor_kind_from_env();
const char* background_extractor_kind_name(BackgroundExtractorKind kind);
bool should_load_gpu_model_slot(ModelRole role, BackgroundExtractorKind background_kind);

std::unique_ptr<IBackgroundMemoryExtractor> create_background_memory_extractor(
    BackgroundExtractorKind kind,
    ModelSlot* background_slot,
    const std::string& background_model_dir,
    int max_seq_len,
    std::string* error_message = nullptr);

}  // namespace aila::alia
