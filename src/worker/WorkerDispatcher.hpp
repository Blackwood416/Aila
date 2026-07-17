#pragma once

#include "aila_api.h"
#include "ipc/IpcProtocol.hpp"

#include <memory>
#include <atomic>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_set>

namespace aila::worker {

namespace detail {

inline constexpr bool stream_data_event_can_emit(uint64_t emitted_data_events) noexcept {
    return emitted_data_events < ipc::kMaxStreamEventCount - 1;
}

} // namespace detail

enum class TextGenerationMethod {
    Generate,
    GenerateMessages,
    GenerateChatJson,
    GenerateChatJsonEx,
    GenerateStream,
    GenerateMessagesStream,
    GenerateChatJsonStreamEx,
};

using TokenStreamCallback = std::function<bool(std::string_view)>;
using StructuredStreamCallback = std::function<bool(const AilaChatStreamEvent&)>;
using WorkerStreamEmitter = std::function<bool(const ipc::Frame&)>;

struct AsrRequest {
    std::string wav_path;
    bool has_config = false;
    AilaGenConfig config{};
    bool has_forced_language = false;
    std::string forced_language;
    bool has_system_prompt = false;
    std::string system_prompt;
    float segment_sec = 0.0f;
    int past_text_conditioning = 0;
};

struct AsrStreamConfig {
    bool has_config = false;
    AilaGenConfig config{};
    bool has_forced_language = false;
    std::string forced_language;
    bool has_system_prompt = false;
    std::string system_prompt;
};

struct TextGenerationRequest {
    TextGenerationMethod method = TextGenerationMethod::Generate;
    std::string input;
    bool has_config = false;
    AilaGenConfig config{};
    bool has_v2_config = false;
    AilaGenConfigV2 config_v2{};
};

class WorkerEngineApi {
public:
    virtual ~WorkerEngineApi() = default;

    virtual int init(const std::string& model, int max_seq_len) = 0;
    virtual void reset_context() = 0;
    virtual int context_length() const = 0;
    virtual int last_error_code() const = 0;
    virtual std::string last_error_message() const = 0;
    virtual bool generate_text(
        const TextGenerationRequest& request,
        std::string& output) = 0;
    virtual int generate_stream(
        const TextGenerationRequest& request,
        const TokenStreamCallback& token_callback,
        const StructuredStreamCallback& structured_callback) = 0;
    virtual bool transcribe(
        const AsrRequest& request,
        const std::function<void(std::string_view)>& token_callback,
        std::string& transcript,
        std::string& language) = 0;
    virtual bool transcribe_stream_create(uint64_t id, const AsrStreamConfig& config) = 0;
    virtual bool transcribe_stream_feed(uint64_t id, const float* samples, size_t count) = 0;
    virtual bool transcribe_stream_get_text(
        uint64_t id, std::string& stable, std::string& partial) = 0;
    virtual bool transcribe_stream_destroy(uint64_t id) noexcept = 0;
};

class WorkerDispatcher {
public:
    explicit WorkerDispatcher(std::unique_ptr<WorkerEngineApi> engine);
    ~WorkerDispatcher() noexcept;

    ipc::Frame dispatch(const ipc::Frame& request, bool& should_shutdown);
    ipc::Frame dispatch_stream(
        const ipc::Frame& request,
        const WorkerStreamEmitter& emit,
        const std::atomic_bool& cancelled);
    static bool is_stream_method(std::string_view method) noexcept;

private:
    std::unique_ptr<WorkerEngineApi> engine_;
    bool initialized_ = false;
    uint64_t next_asr_stream_id_ = 1;
    std::unordered_set<uint64_t> asr_stream_ids_;
};

} // namespace aila::worker
