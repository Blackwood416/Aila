#pragma once

#include "BackgroundMemoryExtractor.hpp"

#include <string>

namespace aila::alia {

class ModelSlot;

class GpuVlmBackgroundExtractor final : public IBackgroundMemoryExtractor {
public:
    explicit GpuVlmBackgroundExtractor(ModelSlot* slot);

    bool ready() const override;
    const char* backend_name() const override { return "LoadedVlm"; }
    BackgroundExtractionResult extract(
        const BackgroundExtractionRequest& request,
        const std::atomic_bool& abort_requested) override;

private:
    bool generate_once(const std::string& prompt_text,
                       const std::atomic_bool& abort_requested,
                       std::string& result_json,
                       std::string& error);

    ModelSlot* slot_ = nullptr;
};

}  // namespace aila::alia
