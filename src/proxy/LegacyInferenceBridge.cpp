#include "proxy/LegacyTextSession.hpp"

#include <windows.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <thread>

namespace aila::proxy {
namespace {
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

struct AdmissionLimits {
    uint32_t maxSessions = 0; // Unspecified: use the public Client default.
    uint64_t sessionCacheBytes = 0, deviceBytes = 0, pinnedHostBytes = 0, sessionHostBytes = 0;
};

// Read once from the same Win32 environment snapshot used by the pool key.
// The Proxy's static CRT environment can be stale after a host changes its
// settings. Do not use it, log the environment block, or persist its values.
std::optional<uint64_t> ReadAdmissionLimit(std::wstring_view environment, const char* name) {
    const std::wstring key(name, name + std::strlen(name));
    size_t offset = 0;
    while (offset < environment.size()) {
        const size_t end = environment.find(L'\0', offset);
        const auto entry = environment.substr(offset,
            (end == std::wstring_view::npos ? environment.size() : end) - offset);
        offset += entry.size() + 1;
        const auto separator = entry.find(L'=');
        if (separator != key.size() || CompareStringOrdinal(entry.data(), static_cast<int>(separator),
            key.data(), static_cast<int>(key.size()), TRUE) != CSTR_EQUAL) continue;
        const auto text = entry.substr(separator + 1);
        uint64_t value = 0;
        const auto invalid = [&] {
            throw LegacyTextError(AILA_ERR_INVALID_ARGUMENT, std::string(name) + " must be an unsigned decimal integer");
        };
        if (text.empty()) invalid();
        for (const wchar_t digit : text) {
            if (digit < L'0' || digit > L'9' || value > ((std::numeric_limits<uint64_t>::max)() - (digit - L'0')) / 10)
                invalid();
            value = value * 10 + (digit - L'0');
        }
        return value;
    }
    return {};
}

AdmissionLimits ParseAdmissionLimits(std::wstring_view environment) {
    AdmissionLimits limits;
    if (const auto count = ReadAdmissionLimit(environment, "AILA_LEGACY_TEXT_MAX_SESSIONS")) {
        if (!*count || *count > (std::numeric_limits<uint32_t>::max)())
            throw LegacyTextError(AILA_ERR_INVALID_ARGUMENT, "AILA_LEGACY_TEXT_MAX_SESSIONS must be a positive supported Session count");
        limits.maxSessions = static_cast<uint32_t>(*count);
    }
    limits.sessionCacheBytes = ReadAdmissionLimit(environment, "AILA_LEGACY_TEXT_MAX_SESSION_CACHE_BYTES").value_or(0);
    limits.deviceBytes = ReadAdmissionLimit(environment, "AILA_LEGACY_TEXT_MAX_DEVICE_BYTES").value_or(0);
    limits.pinnedHostBytes = ReadAdmissionLimit(environment, "AILA_LEGACY_TEXT_MAX_PINNED_HOST_BYTES").value_or(0);
    limits.sessionHostBytes = ReadAdmissionLimit(environment, "AILA_LEGACY_TEXT_MAX_SESSION_HOST_BYTES").value_or(0);
    return limits;
}

template <class T> T Output() {
    T value{};
    value.struct_size = sizeof(T);
    value.abi_version = AILA_INFERENCE_ABI_VERSION;
    return value;
}
void Check(int32_t code, const AilaInferenceError& error) {
    if (code == AILA_INFERENCE_OK) return;
    const int legacy = code == AILA_INFERENCE_INVALID_ARGUMENT ? AILA_ERR_INVALID_ARGUMENT
        : code == AILA_INFERENCE_CONTEXT_OVERFLOW ? AILA_ERR_CONTEXT_OVERFLOW
        : code == AILA_INFERENCE_UNSUPPORTED ? AILA_ERR_MODEL_CAPABILITY : AILA_ERR_RUNTIME;
    throw LegacyTextError(legacy, error.message[0] ? error.message : "Inference C ABI operation failed");
}

// Explicit path and restricted DLL search keep GPU and host dependency lookup
// independent of the embedding application's cwd and PATH. All functions use
// caller-owned POD buffers, never C++ allocations across the two CRTs.
class Api final {
public:
    explicit Api(const fs::path& runtime) {
        library = LoadLibraryExW((runtime / L"AilaInference.dll").c_str(), nullptr,
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!library) throw LegacyTextError(AILA_ERR_RUNTIME,
            "Could not load AilaInference.dll from the runtime directory (Win32 " + std::to_string(GetLastError()) + ")");
        try {
#define BIND(name) name = reinterpret_cast<decltype(name)>(GetProcAddress(library, "aila_inference_" #name)); \
    if (!name) throw LegacyTextError(AILA_ERR_RUNTIME, "AilaInference.dll is missing " #name)
            BIND(abi_version); BIND(build_id);
            BIND(default_client_options); BIND(default_model_options);
            BIND(client_start); BIND(client_shutdown); BIND(client_release);
            BIND(client_read_log); BIND(client_set_log_level);
            BIND(model_load); BIND(model_info); BIND(model_release);
            BIND(session_create); BIND(session_query); BIND(session_reset); BIND(session_truncate); BIND(session_release);
            BIND(request_submit_with_policy); BIND(request_prefill); BIND(request_read); BIND(request_cancel); BIND(request_release);
#undef BIND
            if (abi_version() != AILA_INFERENCE_ABI_VERSION || std::string(build_id()) != AILA_BUILD_ID)
                throw LegacyTextError(AILA_ERR_RUNTIME, "Proxy and AilaInference.dll have different ABI/build identities");
        } catch (...) { FreeLibrary(library); library = nullptr; throw; }
    }
    ~Api() { if (library) FreeLibrary(library); }
#define ENTRY(name) decltype(&aila_inference_##name) name = nullptr
    ENTRY(abi_version); ENTRY(build_id); ENTRY(default_client_options); ENTRY(default_model_options);
    ENTRY(client_start); ENTRY(client_shutdown); ENTRY(client_release);
    ENTRY(client_read_log); ENTRY(client_set_log_level);
    ENTRY(model_load); ENTRY(model_info); ENTRY(model_release);
    ENTRY(session_create); ENTRY(session_query); ENTRY(session_reset); ENTRY(session_truncate); ENTRY(session_release);
    ENTRY(request_submit_with_policy); ENTRY(request_prefill); ENTRY(request_read); ENTRY(request_cancel); ENTRY(request_release);
#undef ENTRY
private:
    HMODULE library = nullptr;
};

struct ModelOwner final {
    explicit ModelOwner(const fs::path& runtime, const fs::path& apiDirectory,
                        const std::string& directory, uint32_t backend, uint32_t capacity,
                        const AdmissionLimits& admission, const logging::Source& source)
        : api(apiDirectory) {
        auto error = Output<AilaInferenceError>();
        Check(api.default_client_options(&limits, sizeof(limits)), error);
        limits.max_generated_tokens = (std::max)(limits.max_generated_tokens,
            (std::min)(capacity, AILA_INFERENCE_EXTENDED_MAX_GENERATED_TOKENS));
        if (admission.maxSessions) {
            if (admission.maxSessions > limits.max_sessions)
                throw LegacyTextError(AILA_ERR_INVALID_ARGUMENT, "AILA_LEGACY_TEXT_MAX_SESSIONS exceeds the supported Session count");
            limits.max_sessions = admission.maxSessions;
            limits.max_active_requests = (std::min)(limits.max_active_requests, limits.max_sessions);
        }
        limits.max_session_cache_bytes = admission.sessionCacheBytes;
        // Match the old synchronous command deadline for model loading.
        limits.command_timeout_ms = 600000;
        limits.handshake_timeout_ms = 5000;
        const auto runtimeUtf8 = runtime.u8string();
        try {
            AttachLogSource(source);
            Check(api.client_start(runtimeUtf8.c_str(), nullptr, &limits, &client, &error), error);
            SetLogLevel(logging::level());
            logPump = std::thread([this] { PumpLogs(); });
            auto options = Output<AilaInferenceModelOptions>();
            Check(api.default_model_options(&options, sizeof(options)), error);
            options.max_sequence_length = capacity;
            options.max_sessions = limits.max_sessions;
            options.max_session_cache_bytes = admission.sessionCacheBytes;
            options.allocation_limits_enabled = admission.deviceBytes || admission.pinnedHostBytes || admission.sessionHostBytes;
            options.max_device_bytes = admission.deviceBytes;
            options.max_pinned_host_bytes = admission.pinnedHostBytes;
            options.max_session_host_bytes = admission.sessionHostBytes;
            Check(api.model_load(client, backend, directory.c_str(), &options, 600000, &model, &error), error);
            info = Output<AilaInferenceModelInfo>();
            Check(api.model_info(model, &info, &error), error);
        } catch (...) { Close(); throw; }
    }
    ~ModelOwner() { Close(); }
    void Close() noexcept {
        if (client) api.client_shutdown(client, 10000, nullptr);
        stopLogs.store(true);
        if (logPump.joinable()) logPump.join();
        if (model) api.model_release(model, nullptr);
        if (client) api.client_release(client, nullptr);
        model = client = 0;
    }
    void AttachLogSource(const logging::Source& source) {
        std::lock_guard<std::mutex> lock(logSourcesMutex);
        logSources.erase(std::remove_if(logSources.begin(), logSources.end(), [](const auto& entry) {
            return !logging::is_active(entry.lock());
        }), logSources.end());
        if (!logging::is_active(source)) return;
        for (const auto& entry : logSources) if (entry.lock() == source) return;
        logSources.emplace_back(source);
    }
    void RouteLog(int level, std::string_view message) noexcept {
        try {
            logging::Source destination;
            {
                std::lock_guard<std::mutex> lock(logSourcesMutex);
                for (const auto& entry : logSources) {
                    auto source = entry.lock();
                    if (logging::is_active(source)) { destination = std::move(source); break; }
                }
            }
            // Worker logs have no Request identity. Deliver once through a
            // live Engine lease; retirement and callback quiescence stay with
            // ProxyEngine, outside its operation/pool locks. Never fan out.
            if (destination) logging::enqueue(destination, level, message);
        } catch (...) {}
    }
    void PumpLogs() noexcept {
        while (!stopLogs.load()) {
            auto event = Output<AilaInferenceLogEvent>();
            const auto result = api.client_read_log(client, 100, &event, nullptr);
            if (result == AILA_INFERENCE_TIMEOUT) continue;
            if (result == AILA_INFERENCE_END_OF_STREAM || result == AILA_INFERENCE_CLOSED) return;
            if (result != AILA_INFERENCE_OK) {
                RouteLog(3, "[Inference] Worker log reader failed");
                return;
            }
            RouteLog(static_cast<int>(event.level), {event.message, event.message_bytes});
        }
    }
    void SetLogLevel(int level) {
        std::lock_guard<std::mutex> lock(logLevelMutex);
        if (appliedLogLevel == level) return;
        auto error = Output<AilaInferenceError>();
        Check(api.client_set_log_level(client, static_cast<uint32_t>(level), 30000, &error), error);
        appliedLogLevel = level;
    }
    Api api;
    AilaInferenceClientOptions limits{};
    AilaInferenceModelInfo info{};
    AilaInferenceClientHandle client = 0;
    AilaInferenceModelHandle model = 0;
    std::atomic<uint32_t> sessions{0};
    std::mutex logSourcesMutex;
    std::vector<std::weak_ptr<logging::SourceState>> logSources;
    std::mutex logLevelMutex;
    int appliedLogLevel = -1;
    std::atomic<bool> stopLogs{false};
    std::thread logPump;
};

class TokenStream final : public LegacyTokenStream {
public:
    TokenStream(std::shared_ptr<ModelOwner> owner, AilaInferenceRequestHandle request)
        : owner(std::move(owner)), request(request), deadline(Clock::now() + std::chrono::minutes(10)) {}
    ~TokenStream() override { owner->api.request_release(request, nullptr); }
    AilaInferenceEvent Read() override {
        auto error = Output<AilaInferenceError>();
        for (;;) {
            auto event = Output<AilaInferenceEvent>();
            const auto result = owner->api.request_read(request, 100, &event, &error);
            if (result == AILA_INFERENCE_TIMEOUT && Clock::now() < deadline) continue;
            Check(result, error);
            if (event.kind == AILA_INFERENCE_EVENT_TERMINAL &&
                (event.local_failure || event.progress.status == AILA_INFERENCE_FAILED))
                throw LegacyTextError(AILA_ERR_RUNTIME, event.progress.message[0] ? event.progress.message : "Inference request failed");
            return event;
        }
    }
    void Cancel() override {
        auto error = Output<AilaInferenceError>();
        uint32_t requested = 0;
        Check(owner->api.request_cancel(request, 30000, &requested, &error), error);
    }
private:
    std::shared_ptr<ModelOwner> owner;
    AilaInferenceRequestHandle request;
    Clock::time_point deadline;
};

class TokenSession final : public LegacyTokenSession {
public:
    explicit TokenSession(std::shared_ptr<ModelOwner> owner, const logging::Source& source) : owner(std::move(owner)) {
        auto error = Output<AilaInferenceError>();
        this->owner->AttachLogSource(source);
        Check(this->owner->api.session_create(this->owner->model, 600000, &session, &error), error);
    }
    ~TokenSession() override {
        owner->api.session_release(session, nullptr);
        --owner->sessions;
    }
    uint32_t Capacity() const override { return owner->info.max_sequence_length; }
    uint32_t MaxPromptTokens() const override { return owner->limits.max_prompt_tokens; }
    uint32_t MaxGeneratedTokens() const override { return owner->limits.max_generated_tokens; }
    uint64_t MaxOutstandingTokens() const override { return owner->limits.max_outstanding_tokens; }
    uint32_t MaxPrefillChunk() const override { return owner->limits.max_prefill_chunk_size; }
    uint32_t ContextLength() override {
        auto error = Output<AilaInferenceError>();
        auto info = Output<AilaInferenceSessionInfo>();
        Check(owner->api.session_query(session, 30000, &info, &error), error);
        if (info.busy || !info.has_context_length) throw LegacyTextError(AILA_ERR_RUNTIME, "Inference Session is busy");
        return info.context_length;
    }
    void Reset() override {
        auto error = Output<AilaInferenceError>();
        auto info = Output<AilaInferenceSessionInfo>();
        Check(owner->api.session_reset(session, 600000, &info, &error), error);
    }
    void SetLogLevel(int level) override { owner->SetLogLevel(level); }
    uint32_t Truncate(uint32_t length) override {
        auto error = Output<AilaInferenceError>();
        auto info = Output<AilaInferenceSessionInfo>();
        Check(owner->api.session_truncate(session, length, 600000, &info, &error), error);
        return info.context_length;
    }
    void Prefill(const std::vector<int>& tokens, uint32_t chunk) override {
        auto error = Output<AilaInferenceError>();
        AilaInferenceRequestHandle request = 0;
        Check(owner->api.request_prefill(session, tokens.data(), static_cast<uint32_t>(tokens.size()),
                                        chunk, 30000, &request, &error), error);
        TokenStream stream(owner, request);
        const auto event = stream.Read();
        if (event.kind != AILA_INFERENCE_EVENT_TERMINAL || event.progress.status != AILA_INFERENCE_COMPLETED ||
            event.progress.generated_tokens || event.progress.prompt_tokens_processed != tokens.size())
            throw LegacyTextError(AILA_ERR_RUNTIME, "Prefix replay did not complete the supplied prefix");
    }
    std::unique_ptr<LegacyTokenStream> Submit(const std::vector<int>& tokens,
                                             const AilaInferenceRequestOptions& options,
                                             const AilaInferenceRequestPolicy& policy) override {
        auto error = Output<AilaInferenceError>();
        AilaInferenceRequestHandle request = 0;
        Check(owner->api.request_submit_with_policy(session, tokens.data(), static_cast<uint32_t>(tokens.size()),
            &options, &policy, 30000, &request, &error), error);
        try { return std::make_unique<TokenStream>(owner, request); }
        catch (...) { owner->api.request_release(request, nullptr); throw; }
    }
private:
    std::shared_ptr<ModelOwner> owner;
    AilaInferenceSessionHandle session = 0;
};

std::wstring EnvironmentKey() {
    wchar_t* block = GetEnvironmentStringsW();
    if (!block) throw LegacyTextError(AILA_ERR_RUNTIME, "Could not snapshot Worker environment");
    struct Release { wchar_t* p; ~Release() { FreeEnvironmentStringsW(p); } } release{block};
    const wchar_t* end = block;
    while (*end) end += wcslen(end) + 1;
    // Not logged or persisted. Reusing a model requires the exact inherited
    // environment so changing a kernel/adapter setting never reuses old weights.
    return {block, static_cast<size_t>(end - block)};
}
} // namespace

std::unique_ptr<LegacyTokenSession> CreateLegacyTokenSession(
    const fs::path& runtimeDirectory, const fs::path& apiDirectory,
    const std::string& modelDirectory, uint32_t backend, uint32_t capacity, logging::Source source) {
    const auto runtime = fs::weakly_canonical(runtimeDirectory);
    const auto libraryDirectory = fs::weakly_canonical(apiDirectory);
    const auto model = fs::weakly_canonical(fs::u8path(modelDirectory));
    const auto environment = EnvironmentKey();
    const auto admission = ParseAdmissionLimits(environment);
    const std::wstring key = runtime.wstring() + L'\0' + libraryDirectory.wstring() + L'\0' + model.wstring() + L'\0' +
        std::to_wstring(backend) + L":" + std::to_wstring(capacity) + L'\0' + environment;
    static std::mutex poolMutex;
    static std::map<std::wstring, std::weak_ptr<ModelOwner>> pool;
    std::shared_ptr<ModelOwner> owner;
    {
        std::lock_guard<std::mutex> lock(poolMutex);
        for (auto it = pool.begin(); it != pool.end();) {
            if (it->second.expired()) it = pool.erase(it); else ++it;
        }
        const auto found = pool.find(key);
        if (found != pool.end()) owner = found->second.lock();
        if (!owner) {
            owner = std::make_shared<ModelOwner>(runtime, libraryDirectory, model.u8string(), backend, capacity, admission, source);
            pool[key] = owner;
        }
        if (owner->sessions >= owner->limits.max_sessions)
            throw LegacyTextError(AILA_ERR_RUNTIME, "Shared legacy text model reached its Session limit");
        ++owner->sessions;
    }
    try { return std::make_unique<TokenSession>(owner, source); }
    catch (...) { --owner->sessions; throw; }
}
} // namespace aila::proxy
