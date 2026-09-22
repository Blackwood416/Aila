#pragma once

#include "worker/WorkerDispatcher.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace aila::worker {

// Adapts the in-process C API to the worker dispatcher.
class CApiWorkerEngine final : public WorkerEngineApi {
public:
    CApiWorkerEngine();

    ~CApiWorkerEngine() override;

    int init(const std::string& model, int max_seq_len) override;

    void reset_context() override;

    int context_length() const override;

    int last_error_code() const override;

    std::string last_error_message() const override;

    void set_log_level(int level) override;

    bool generate_text(
        const aila::worker::TextGenerationRequest& request,
        std::string& output) override;

    int generate_stream(
        const aila::worker::TextGenerationRequest& request,
        const aila::worker::TokenStreamCallback& token_callback,
        const aila::worker::StructuredStreamCallback& structured_callback) override;

    bool transcribe(
        const aila::worker::AsrRequest& request,
        const std::function<void(std::string_view)>& token_callback,
        std::string& transcript,
        std::string& language) override;

    bool transcribe_stream_create(
        uint64_t id, const aila::worker::AsrStreamConfig& config) override;

    bool transcribe_stream_feed(uint64_t id, const float* samples, size_t count) override;

    bool transcribe_stream_get_text(
        uint64_t id, std::string& stable, std::string& partial) override;

    bool transcribe_stream_destroy(uint64_t id) noexcept override;

    bool process_audio(
        const aila::worker::AudioRequest& request,
        std::vector<float>& output) override;

    int synthesize_stream(
        const aila::worker::AudioRequest& request,
        const aila::worker::AudioStreamCallback& callback) override;

    bool align(
        const aila::worker::AlignmentRequest& request,
        std::vector<aila::worker::AlignedWordResult>& output) override;

    bool detect(const aila::worker::DetectionRequest& request,
                std::vector<aila::worker::DetectionResult>& output) override;

private:
    AilaEngine* engine_ = nullptr;
    std::unordered_map<uint64_t, AilaTranscribeStream*> asr_streams_;
};

} // namespace aila::worker
