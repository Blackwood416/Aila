#pragma once

#include "chat/ChatTypes.hpp"
#include "runtime/InferenceClient.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

class Tokenizer;

namespace aila::server {

class ApiError : public std::runtime_error {
public:
    ApiError(int status, std::string errorCode, std::string message, std::string parameter = {});
    std::string ToJson() const;

    int httpStatus;
    std::string code;
    std::string param;
};

struct ChatSettings {
    std::string modelId;
    uint32_t maxContextTokens = 4096;
    uint32_t maxPromptTokens = 4096;
    uint32_t maxNewTokens = 1024;
    // Prefill work per scheduler step. Bounded by the prompt limit at render time.
    uint32_t prefillChunkSize = 512;
    // Per-request body and rendered-prompt byte ceiling. 1 MiB is the legacy
    // default; the server CLI may raise it up to 4 MiB.
    size_t maxBodyBytes = 1024 * 1024;
    size_t maxOutputBytes = 262144;
    double defaultTemperature = 0.6;
    ModelFamily family = ModelFamily::Qwen3Dense;
};

struct PreparedChat {
    bool stream = false;
    bool includeUsage = false;
    bool parallelToolCalls = true;
    std::vector<int32_t> promptIds;
    runtime::InferenceRequestOptions options;
    chat::ChatRequest chatRequest;
    // A generation prompt can already contain an opening <think> marker.
    bool startsInReasoning = false;
    uint64_t cachedPromptTokens = 0; // Server-owned exact prefix reuse, not a request field.
};

// Strict text and function-tool OpenAI-style input. Unknown fields and unsupported features
// are rejected before template rendering or inference admission. The full
// prompt + requested generation must fit the Session, including reused tokens.
PreparedChat PrepareChat(std::string_view body, const Tokenizer& tokenizer,
                         const ChatSettings& settings);

// Tool markers can be added-token IDs that legacy Tokenizer::decode hides.
// Preserve them only for an HTTP request that actually offers function tools.
std::string DecodeChatToken(const Tokenizer& tokenizer, const PreparedChat& prepared, int32_t tokenId);

struct ChatIdentity {
    std::string id;
    std::string model;
    int64_t createdUnixSeconds = 0;
};

struct ChatCompletion {
    std::string json;
    // Complete "data: ...\n\n" SSE frames, including the final [DONE].
    std::vector<std::string> sseFrames;
    int httpStatus = 200;
    bool failed = false;
    // Bounded diagnostics, e.g. an unsolicited tool block retained as text.
    std::vector<std::string> warnings;
};

// Single-consumer adapter. Tokenizer::decode may split UTF-8 scalars or control
// markers between calls; only complete UTF-8 is emitted. Normal serialized
// output and raw input text are bounded by maxOutputBytes, with a reserved
// failure envelope. No socket or inference callbacks run in this class.
class ChatOutput final {
public:
    ChatOutput(ChatIdentity identity, const PreparedChat& prepared, size_t maxOutputBytes = 262144);
    ~ChatOutput();
    ChatOutput(ChatOutput&&) noexcept;
    ChatOutput& operator=(ChatOutput&&) noexcept;
    ChatOutput(const ChatOutput&) = delete;
    ChatOutput& operator=(const ChatOutput&) = delete;

    std::vector<std::string> Begin();
    std::vector<std::string> PushText(std::string_view bytes);
    ChatCompletion Finish(const ipc::v2::Summary& summary, bool localFailure = false);
    // Use after a thrown ApiError or another HTTP-layer failure; stream output
    // receives an error payload and [DONE], never a successful finish_reason.
    ChatCompletion Fail(const ApiError& error);

private:
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace aila::server
