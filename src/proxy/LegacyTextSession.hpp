#pragma once

#include "aila_api.h"
#include "aila_inference.h"
#include "chat/ChatFormatter.hpp"
#include "chat/ChatSessionState.hpp"
#include "chat/StructuredStreamParser.hpp"
#include "proxy/ProxyLogging.hpp"
#include "utils/Tokenizer.hpp"

#include <filesystem>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string_view>

namespace aila::proxy {

class LegacyTextError : public std::runtime_error {
public:
    LegacyTextError(int code, std::string message) : std::runtime_error(std::move(message)), code(code) {}
    int Code() const noexcept { return code; }
private:
    int code;
};

// Host-only seam. Production crosses the MT/MD boundary through the C ABI;
// CPU tests supply a deterministic token stream without loading a GPU runtime.
class LegacyTokenStream {
public:
    virtual ~LegacyTokenStream() = default;
    virtual AilaInferenceEvent Read() = 0;
    virtual void Cancel() = 0;
};
class LegacyTokenSession {
public:
    virtual ~LegacyTokenSession() = default;
    virtual uint32_t Capacity() const = 0;
    virtual uint32_t MaxPromptTokens() const = 0;
    virtual uint32_t MaxGeneratedTokens() const = 0;
    virtual uint64_t MaxOutstandingTokens() const = 0;
    virtual uint32_t MaxPrefillChunk() const = 0;
    virtual uint32_t ContextLength() = 0;
    virtual void Reset() = 0;
    virtual void SetLogLevel(int) {}
    virtual uint32_t Truncate(uint32_t length) = 0;
    virtual void Prefill(const std::vector<int>& tokens, uint32_t chunk) = 0;
    virtual std::unique_ptr<LegacyTokenStream> Submit(
        const std::vector<int>& tokens, const AilaInferenceRequestOptions& options,
        const AilaInferenceRequestPolicy& policy) = 0;
};

std::unique_ptr<LegacyTokenSession> CreateLegacyTokenSession(
    const std::filesystem::path& runtimeDirectory, const std::filesystem::path& apiDirectory,
    const std::string& modelDirectory,
    uint32_t backend, uint32_t capacity, logging::Source source = {});

struct LegacyTextResult {
    std::string text;
    bool aborted = false;
};

class LegacyTextSession final {
public:
    using TokenCallback = std::function<bool(const std::string&)>;
    using ChatCallback = std::function<bool(const aila::chat::StructuredStreamEvent&)>;
    // Null means this model/mode still belongs to the existing capability
    // driver (media, quantized KV or an unsupported weight backend).
    static std::unique_ptr<LegacyTextSession> TryCreate(
        const std::filesystem::path& runtimeDirectory, const std::filesystem::path& apiDirectory,
        const std::string& modelDirectory,
        int capacity, logging::Source source);
    LegacyTextSession(ModelSpec spec, Tokenizer tokenizer, std::unique_ptr<LegacyTokenSession> session,
                      uint32_t prefillChunk, logging::Source source = {});
    LegacyTextResult Generate(std::string_view method, const std::string& input,
                              GenerationConfig config, TokenCallback callback = {}, ChatCallback structured = {});
    void Reset();
    void SetLogLevel(int level) { session->SetLogLevel(level); }
    int ContextLength();
    static GenerationConfig Config(const AilaGenConfig* config);
    static GenerationConfig Config(const AilaGenConfigV2* config);
    static int Deliver(const aila::chat::StructuredStreamEvent& event,
                       AilaChatStreamCallback callback, void* userData);

private:
    struct Generated {
        std::string text;
        std::string templateName;
        std::string finishReason = "stop";
        bool aborted = false;
        bool forcedClose = false;
        bool truncatedClose = false;
    };
    aila::chat::ChatFormatInput FormatInput(const aila::chat::ChatRequest& request) const;
    Generated GenerateChat(const aila::chat::ChatRequest& request, const TokenCallback& callback);
    aila::chat::AssistantChatResult Result(const aila::chat::ChatRequest& request, const Generated& generated) const;
    void Replay(const std::vector<int>& sequence, size_t first, size_t end);
    void Align(const std::vector<int>& prompt, const std::vector<int>& generated);

    ModelSpec spec;
    Tokenizer tokenizer;
    std::unique_ptr<LegacyTokenSession> session;
    uint32_t prefillChunk;
    logging::Source source;
    aila::chat::ChatFormatter formatter;
    aila::chat::ChatSessionState history;
    std::vector<int> cachedIds;
    std::string systemPrompt;
};

} // namespace aila::proxy
