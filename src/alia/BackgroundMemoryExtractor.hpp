#pragma once

#include <atomic>
#include <string>

namespace aila::alia {

struct BackgroundExtractionRequest {
    std::string chat_turn_text;
};

struct BackgroundExtractionResult {
    bool ok = false;
    std::string result_json;
    std::string prompt_text;
    std::string error;
    int schema_retry_count = 0;
    bool schema_repair_applied = false;
    std::string schema_diagnostic;
};

class IBackgroundMemoryExtractor {
public:
    virtual ~IBackgroundMemoryExtractor() = default;
    virtual bool ready() const = 0;
    virtual const char* backend_name() const = 0;
    virtual BackgroundExtractionResult extract(
        const BackgroundExtractionRequest& request,
        const std::atomic_bool& abort_requested) = 0;
};

}  // namespace aila::alia
