#pragma once

#include "BackgroundMemoryExtractor.hpp"
#include "engine/Types.hpp"
#include "models/cpu/CpuQ35HybridModel.hpp"

#include <memory>
#include <string>

class Tokenizer;

namespace aila::alia {

class CpuQ35BackgroundExtractor final : public IBackgroundMemoryExtractor {
public:
    CpuQ35BackgroundExtractor(std::string model_dir, int max_seq_len);
    ~CpuQ35BackgroundExtractor() override;

    bool load(std::string* error_message = nullptr);
    bool ready() const override { return loaded_; }
    const char* backend_name() const override { return "NativeCpuQ35"; }
    BackgroundExtractionResult extract(
        const BackgroundExtractionRequest& request,
        const std::atomic_bool& abort_requested) override;

    const std::string& last_error() const { return last_error_; }

private:
    bool generate_once(const std::string& prompt_text,
                       const std::atomic_bool& abort_requested,
                       std::string& result_json,
                       std::string& error);

    std::string model_dir_;
    int max_seq_len_ = 0;
    ModelSpec spec_{};
    std::unique_ptr<Tokenizer> tokenizer_;
    std::unique_ptr<CpuQ35HybridModel> model_;
    bool loaded_ = false;
    std::string last_error_;
};

}  // namespace aila::alia
