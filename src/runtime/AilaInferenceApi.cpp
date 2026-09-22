#include "aila_inference.h"
#include "runtime/InferenceClient.hpp"

#include <Windows.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string_view>
#include <utility>

#ifndef AILA_BUILD_ID
#error "The public inference DLL requires the Worker build identity"
#endif

namespace {
namespace host = aila::runtime;
namespace wire = aila::ipc::v2;
using Milliseconds = std::chrono::milliseconds;

struct ApiError {
    int32_t code;
    const char* message;
};
void Require(bool value, const char* message) {
    if (!value) throw ApiError{AILA_INFERENCE_INVALID_ARGUMENT, message};
}

template <typename T>
bool ValidHeader(const T* value) noexcept {
    return value && value->struct_size >= sizeof(T) && value->abi_version == AILA_INFERENCE_ABI_VERSION;
}
template <typename T>
void CheckHeader(const T* value) {
    Require(ValidHeader(value), "Invalid struct pointer, struct_size or abi_version");
}
template <size_t N>
void CheckReserved(const uint32_t (&value)[N]) {
    for (const auto field : value) Require(field == 0, "Reserved input fields must be zero");
}
template <typename T>
T Output(const T* destination) noexcept {
    T result{};
    result.struct_size = destination->struct_size;
    result.abi_version = AILA_INFERENCE_ABI_VERSION;
    return result;
}
template <size_t N>
void CopyText(char (&destination)[N], std::string_view text) noexcept {
    auto count = (std::min)(text.size(), N - 1);
    // Do not cut a valid UTF-8 code point when bounding diagnostics.
    if (count < text.size()) {
        while (count && (static_cast<unsigned char>(text[count]) & 0xc0) == 0x80) --count;
    }
    if (count) std::memcpy(destination, text.data(), count);
    destination[count] = '\0';
}

uint32_t RemoteCode(wire::ErrorCode code) noexcept {
    switch (code) {
    case wire::ErrorCode::InvalidArgument: return AILA_INFERENCE_REMOTE_INVALID_ARGUMENT;
    case wire::ErrorCode::Busy: return AILA_INFERENCE_REMOTE_BUSY;
    case wire::ErrorCode::ContextOverflow: return AILA_INFERENCE_REMOTE_CONTEXT_OVERFLOW;
    case wire::ErrorCode::UnsupportedModel: return AILA_INFERENCE_REMOTE_UNSUPPORTED_MODEL;
    case wire::ErrorCode::ResourceLimit: return AILA_INFERENCE_REMOTE_RESOURCE_LIMIT;
    case wire::ErrorCode::RuntimeFault: return AILA_INFERENCE_REMOTE_RUNTIME_FAULT;
    case wire::ErrorCode::SchedulerStopped: return AILA_INFERENCE_REMOTE_SCHEDULER_STOPPED;
    case wire::ErrorCode::UnsupportedProtocol: return AILA_INFERENCE_REMOTE_UNSUPPORTED_PROTOCOL;
    case wire::ErrorCode::UnsupportedCapability: return AILA_INFERENCE_REMOTE_UNSUPPORTED_CAPABILITY;
    case wire::ErrorCode::StaleEpoch: return AILA_INFERENCE_REMOTE_STALE_EPOCH;
    case wire::ErrorCode::InvalidHandle: return AILA_INFERENCE_REMOTE_INVALID_HANDLE;
    case wire::ErrorCode::ProtocolError: return AILA_INFERENCE_REMOTE_PROTOCOL_ERROR;
    case wire::ErrorCode::RequestFailed: return AILA_INFERENCE_REMOTE_REQUEST_FAILED;
    case wire::ErrorCode::TransportError: return AILA_INFERENCE_REMOTE_TRANSPORT_ERROR;
    }
    return AILA_INFERENCE_REMOTE_PROTOCOL_ERROR;
}
int32_t ResultCode(const host::InferenceClientError& error) noexcept {
    switch (error.Code()) {
    case host::ClientErrorCode::Timeout: return AILA_INFERENCE_TIMEOUT;
    case host::ClientErrorCode::InvalidHandle: return AILA_INFERENCE_INVALID_HANDLE;
    case host::ClientErrorCode::ResourceLimit: return AILA_INFERENCE_RESOURCE_LIMIT;
    case host::ClientErrorCode::Closed: return AILA_INFERENCE_CLOSED;
    case host::ClientErrorCode::ProtocolError: return AILA_INFERENCE_PROTOCOL_ERROR;
    case host::ClientErrorCode::RemoteError: break;
    }
    if (error.RemoteCode()) {
        switch (*error.RemoteCode()) {
        case wire::ErrorCode::InvalidArgument: return AILA_INFERENCE_INVALID_ARGUMENT;
        case wire::ErrorCode::Busy: return AILA_INFERENCE_BUSY;
        case wire::ErrorCode::ContextOverflow: return AILA_INFERENCE_CONTEXT_OVERFLOW;
        case wire::ErrorCode::UnsupportedModel:
        case wire::ErrorCode::UnsupportedProtocol:
        case wire::ErrorCode::UnsupportedCapability: return AILA_INFERENCE_UNSUPPORTED;
        case wire::ErrorCode::ResourceLimit: return AILA_INFERENCE_RESOURCE_LIMIT;
        case wire::ErrorCode::RuntimeFault: return AILA_INFERENCE_RUNTIME_FAULT;
        case wire::ErrorCode::SchedulerStopped:
        case wire::ErrorCode::TransportError: return AILA_INFERENCE_CLOSED;
        case wire::ErrorCode::StaleEpoch: return AILA_INFERENCE_STALE_EPOCH;
        case wire::ErrorCode::InvalidHandle: return AILA_INFERENCE_INVALID_HANDLE;
        case wire::ErrorCode::ProtocolError: return AILA_INFERENCE_PROTOCOL_ERROR;
        case wire::ErrorCode::RequestFailed: return AILA_INFERENCE_REQUEST_FAILED;
        }
    }
    return AILA_INFERENCE_INTERNAL_ERROR;
}
int32_t Report(AilaInferenceError* error, int32_t code, std::string_view message = {},
               uint32_t remote = AILA_INFERENCE_REMOTE_NONE) noexcept {
    if (error) {
        auto result = Output(error);
        result.code = code;
        result.remote_error = remote;
        CopyText(result.message, message);
        *error = result;
    }
    return code;
}
template <typename Function>
int32_t Boundary(AilaInferenceError* error, Function&& function) noexcept {
    if (error && !ValidHeader(error)) return AILA_INFERENCE_INVALID_ARGUMENT;
    try {
        return Report(error, function());
    } catch (const ApiError& failure) {
        return Report(error, failure.code, failure.message);
    } catch (const host::InferenceClientError& failure) {
        return Report(error, ResultCode(failure), failure.what(),
                      failure.RemoteCode() ? RemoteCode(*failure.RemoteCode()) : AILA_INFERENCE_REMOTE_NONE);
    } catch (const std::bad_alloc&) {
        return Report(error, AILA_INFERENCE_OUT_OF_MEMORY, "Host allocation failed");
    } catch (const std::invalid_argument& failure) {
        return Report(error, AILA_INFERENCE_INVALID_ARGUMENT, failure.what());
    } catch (const std::exception& failure) {
        return Report(error, AILA_INFERENCE_INTERNAL_ERROR, failure.what());
    } catch (...) {
        return Report(error, AILA_INFERENCE_INTERNAL_ERROR, "Unknown host exception");
    }
}

std::string Utf8Path(const char* text) {
    Require(text != nullptr, "A UTF-8 path is required");
    constexpr size_t limit = 32 * 1024;
    size_t size = 0;
    while (size <= limit && text[size]) ++size;
    Require(size && size <= limit, "Path is empty or exceeds 32768 UTF-8 bytes");
    Require(MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text, static_cast<int>(size), nullptr, 0) > 0,
            "Path is not valid UTF-8");
    return {text, size};
}

enum class Kind : uint64_t { Client = 1, Model = 2, Session = 3, Request = 4 };
struct Entry {
    virtual ~Entry() = default;
    virtual void Release() noexcept = 0;
};
struct ClientEntry final : Entry {
    std::shared_ptr<host::InferenceClient> value;
    wire::Limits limits;
    std::mutex logReader;
    void Release() noexcept override { if (value) value->CloseLogStream(); }
};
struct ModelEntry final : Entry {
    std::shared_ptr<host::InferenceModel> value;
    wire::Limits limits;
    uint32_t backend = 0;
    void Release() noexcept override { if (value) value->Release(); }
};
struct SessionEntry final : Entry {
    std::shared_ptr<host::InferenceSession> value;
    wire::Limits limits;
    void Release() noexcept override { if (value) value->Release(); }
};
struct RequestEntry final : Entry {
    std::shared_ptr<host::InferenceRequest> value;
    std::mutex reader;
    void Release() noexcept override { if (value) value->Release(); }
};

// Reserve before a remote side effect; commit cannot allocate. A slot with an
// in-progress reservation is invisible. Recycling advances a 48-bit generation;
// generation exhaustion retires the slot rather than revalidating stale IDs.
class Registry final {
    struct Slot {
        uint64_t generation = 0;
        Kind kind = Kind::Client;
        bool occupied = false;
        std::shared_ptr<Entry> value;
    };
public:
    class Reservation final {
    public:
        Reservation(Registry& registry, size_t index, uint64_t id)
            : registry(registry), index(index), id(id) {}
        Reservation(const Reservation&) = delete;
        Reservation& operator=(const Reservation&) = delete;
        ~Reservation() {
            if (!committed) {
                std::lock_guard<std::mutex> lock(registry.mutex);
                registry.slots[index].occupied = false;
            }
        }
        uint64_t Commit(std::shared_ptr<Entry> value) {
            std::lock_guard<std::mutex> lock(registry.mutex);
            registry.slots[index].value = std::move(value);
            committed = true;
            return id;
        }
    private:
        Registry& registry;
        size_t index;
        uint64_t id;
        bool committed = false;
    };
    Reservation Reserve(Kind kind) {
        std::lock_guard<std::mutex> lock(mutex);
        for (size_t index = 0; index < slots.size(); ++index) {
            auto& slot = slots[index];
            if (slot.occupied || slot.generation == (uint64_t{1} << 48) - 1) continue;
            ++slot.generation;
            slot.kind = kind;
            slot.occupied = true;
            return Reservation(*this, index, (slot.generation << 16) | ((index + 1) << 3) | static_cast<uint64_t>(kind));
        }
        throw ApiError{AILA_INFERENCE_RESOURCE_LIMIT, "Public inference handle registry is full"};
    }
    template <typename T>
    std::shared_ptr<T> Acquire(uint64_t id, Kind kind) {
        std::lock_guard<std::mutex> lock(mutex);
        return std::static_pointer_cast<T>(Find(id, kind).value);
    }
    void Release(uint64_t id, Kind kind) {
        if (id == 0) return;
        std::shared_ptr<Entry> value;
        {
            std::lock_guard<std::mutex> lock(mutex);
            auto& slot = Find(id, kind);
            value = std::move(slot.value);
            slot.occupied = false;
        }
        // Never wait for Read or destroy a connection under the registry lock.
        value->Release();
    }
private:
    Slot& Find(uint64_t id, Kind kind) {
        const auto number = (id >> 3) & 0x1fff;
        if (number == 0 || number > slots.size() || (id & 7) != static_cast<uint64_t>(kind))
            throw ApiError{AILA_INFERENCE_INVALID_HANDLE, "Invalid or wrong-type inference handle"};
        auto& slot = slots[number - 1];
        if (!slot.occupied || !slot.value || slot.kind != kind || slot.generation != (id >> 16))
            throw ApiError{AILA_INFERENCE_INVALID_HANDLE, "Stale or released inference handle"};
        return slot;
    }
    std::mutex mutex;
    std::array<Slot, AILA_INFERENCE_MAX_HANDLES> slots;
};
Registry& Handles() {
    static Registry registry;
    return registry;
}

host::InferenceClientOptions ClientOptions(const AilaInferenceClientOptions* input, uint32_t maxDecodeBatchSize = 1) {
    host::InferenceClientOptions result;
    result.maxDecodeBatchSize = maxDecodeBatchSize;
    if (input) {
        CheckHeader(input);
        CheckReserved(input->reserved);
        Require(input->handshake_timeout_ms && input->command_timeout_ms && input->shutdown_timeout_ms,
                "Configured timeouts must be positive");
        result.handshakeTimeout = Milliseconds(input->handshake_timeout_ms);
        result.commandTimeout = Milliseconds(input->command_timeout_ms);
        result.shutdownTimeout = Milliseconds(input->shutdown_timeout_ms);
        auto& limits = result.limits;
        limits.maxActiveRequests = input->max_active_requests;
        limits.maxOutstandingRequests = input->max_outstanding_requests;
        limits.maxPromptTokens = input->max_prompt_tokens;
        limits.maxGeneratedTokens = input->max_generated_tokens;
        limits.maxSessions = input->max_sessions;
        limits.maxRetainedRequestHandles = input->max_retained_request_handles;
        limits.maxBufferedTokensPerRequest = input->max_buffered_tokens_per_request;
        limits.tokenWindow = input->token_window;
        limits.maxPrefillChunkSize = input->max_prefill_chunk_size;
        limits.maxOutstandingTokens = input->max_outstanding_tokens;
        limits.maxSessionCacheBytes = input->max_session_cache_bytes;
    }
    wire::ValidationError error;
    if (!wire::validate_limits(result.limits, error, wire::extended_generation_limits()) ||
        !wire::validate_decode_batch_size(result.maxDecodeBatchSize, result.limits, error))
        throw host::InferenceClientError(host::ClientErrorCode::RemoteError, error.message, wire::ErrorCode::InvalidArgument);
    return result;
}
host::InferenceModelOptions ModelOptions(const AilaInferenceModelOptions* input) {
    host::InferenceModelOptions result;
    if (!input) return result;
    CheckHeader(input);
    CheckReserved(input->reserved);
    Require(input->reserved0 == 0 && input->allocation_limits_enabled <= 1, "Invalid model option flags");
    result.maxSequenceLength = input->max_sequence_length;
    result.maxSessions = input->max_sessions;
    result.maxSessionCacheBytes = input->max_session_cache_bytes;
    if (input->allocation_limits_enabled) {
        result.allocationLimits = wire::AllocationLimits{
            input->max_device_bytes, input->max_pinned_host_bytes, input->max_session_host_bytes};
    } else {
        Require(!(input->max_device_bytes || input->max_pinned_host_bytes || input->max_session_host_bytes),
                "Allocation limits require allocation_limits_enabled");
    }
    return result;
}
std::vector<int32_t> Tokens(const int32_t* ids, uint32_t count, uint32_t maximum) {
    Require(count <= maximum, "Token count exceeds the configured limit");
    Require(count == 0 || ids != nullptr, "A nonempty token array requires a pointer");
    if (!count) return {};
    return {ids, ids + count};
}
host::InferenceRequestOptions RequestOptions(const AilaInferenceRequestOptions* input, const wire::Limits& limits) {
    host::InferenceRequestOptions result;
    if (!input) return result;
    CheckHeader(input);
    CheckReserved(input->reserved);
    Require(input->do_sample <= 1 && input->use_fixed_seed <= 1 && input->prompt_opens_think <= 1,
            "Boolean request options must be 0 or 1");
    auto& generation = result.generation;
    generation.maxNewTokens = input->max_new_tokens;
    generation.doSample = input->do_sample != 0;
    generation.useFixedSeed = input->use_fixed_seed != 0;
    generation.samplingSeed = input->sampling_seed;
    generation.temperature = input->temperature;
    generation.topK = input->top_k;
    generation.topP = input->top_p;
    generation.repetitionPenalty = input->repetition_penalty;
    generation.presencePenalty = input->presence_penalty;
    generation.frequencyPenalty = input->frequency_penalty;
    result.prefillChunkSize = input->prefill_chunk_size;
    result.stopTokenIds = Tokens(input->stop_token_ids, input->stop_token_count, limits.maxStopTokenIds);
    Require(input->thinking_budget_tokens >= -1, "Thinking budget must be -1 or nonnegative");
    if (input->thinking_budget_tokens >= 0) {
        wire::ThinkingBudgetOptions thinking;
        thinking.maxTokens = static_cast<uint32_t>(input->thinking_budget_tokens);
        thinking.openTokenId = input->thinking_open_token_id;
        thinking.closeTokenId = input->thinking_close_token_id;
        thinking.promptOpensThink = input->prompt_opens_think != 0;
        thinking.closeTokenIds = Tokens(input->thinking_close_token_ids, input->thinking_close_token_count,
                                       static_cast<uint32_t>(aila::generation::kMaxThinkingCloseTokens));
        result.thinkingBudget = std::move(thinking);
    } else {
        Require(!input->prompt_opens_think && !input->thinking_close_token_count && !input->thinking_close_token_ids &&
                input->thinking_open_token_id == -1 && input->thinking_close_token_id == -1,
                "Disabled thinking budget must not specify marker metadata");
    }
    return result;
}
uint32_t Status(wire::RequestStatus value) noexcept {
    switch (value) {
    case wire::RequestStatus::Queued: return AILA_INFERENCE_QUEUED;
    case wire::RequestStatus::Prefill: return AILA_INFERENCE_PREFILL;
    case wire::RequestStatus::Decode: return AILA_INFERENCE_DECODE;
    case wire::RequestStatus::Finalizing: return AILA_INFERENCE_FINALIZING;
    case wire::RequestStatus::Completed: return AILA_INFERENCE_COMPLETED;
    case wire::RequestStatus::Cancelled: return AILA_INFERENCE_CANCELLED;
    case wire::RequestStatus::Failed: return AILA_INFERENCE_FAILED;
    }
    return AILA_INFERENCE_FAILED;
}
uint32_t Reason(wire::EndReason value) noexcept {
    switch (value) {
    case wire::EndReason::None: return AILA_INFERENCE_END_NONE;
    case wire::EndReason::Length: return AILA_INFERENCE_END_LENGTH;
    case wire::EndReason::StopToken: return AILA_INFERENCE_END_STOP_TOKEN;
    case wire::EndReason::Cancelled: return AILA_INFERENCE_END_CANCELLED;
    case wire::EndReason::OutputBackpressure: return AILA_INFERENCE_END_OUTPUT_BACKPRESSURE;
    case wire::EndReason::SchedulerStopped: return AILA_INFERENCE_END_SCHEDULER_STOPPED;
    case wire::EndReason::RuntimeFault: return AILA_INFERENCE_END_RUNTIME_FAULT;
    case wire::EndReason::Failed: return AILA_INFERENCE_END_FAILED;
    }
    return AILA_INFERENCE_END_FAILED;
}
uint32_t DomainStatus(wire::DomainStatus value) noexcept {
    switch (value) {
    case wire::DomainStatus::Starting: return AILA_INFERENCE_DOMAIN_STARTING;
    case wire::DomainStatus::Ready: return AILA_INFERENCE_DOMAIN_READY;
    case wire::DomainStatus::Stopping: return AILA_INFERENCE_DOMAIN_STOPPING;
    case wire::DomainStatus::Faulted: return AILA_INFERENCE_DOMAIN_FAULTED;
    case wire::DomainStatus::Closed: return AILA_INFERENCE_DOMAIN_CLOSED;
    }
    return AILA_INFERENCE_DOMAIN_FAULTED;
}
AilaInferenceProgress Progress(const wire::Summary& summary) noexcept {
    AilaInferenceProgress result{};
    result.status = Status(summary.status);
    result.end_reason = Reason(summary.endReason);
    result.remote_error = summary.errorCode ? RemoteCode(*summary.errorCode) : AILA_INFERENCE_REMOTE_NONE;
    result.prompt_tokens_processed = summary.promptTokensProcessed;
    result.generated_tokens = summary.generatedTokens;
    result.event_count = summary.eventCount;
    CopyText(result.message, summary.message);
    return result;
}
void SessionInfo(AilaInferenceSessionInfo* output, const wire::SessionInfo& info,
                 std::string_view epoch) noexcept {
    auto result = Output(output);
    result.session_id = info.sessionId;
    result.model_id = info.modelId;
    result.capacity = info.capacity;
    result.busy = info.busy;
    result.cache_bytes = info.cacheBytes;
    result.has_context_length = info.contextLength.has_value();
    result.context_length = info.contextLength.value_or(0);
    CopyText(result.epoch, epoch);
    *output = result;
}
void CheckHandleOutput(uint64_t* output) {
    Require(output != nullptr, "An output handle pointer is required");
    *output = 0;
}
} // namespace

uint32_t AILA_INFERENCE_CALL aila_inference_abi_version(void) noexcept { return AILA_INFERENCE_ABI_VERSION; }
const char* AILA_INFERENCE_CALL aila_inference_build_id(void) noexcept { return AILA_BUILD_ID; }

int32_t AILA_INFERENCE_CALL aila_inference_default_client_options(AilaInferenceClientOptions* options, uint32_t size) noexcept {
    if (!options || size < sizeof(*options)) return AILA_INFERENCE_INVALID_ARGUMENT;
    AilaInferenceClientOptions result{};
    result.struct_size = size;
    result.abi_version = AILA_INFERENCE_ABI_VERSION;
    result.handshake_timeout_ms = 2000;
    result.command_timeout_ms = 30000;
    result.shutdown_timeout_ms = 10000;
    const wire::Limits defaults;
    result.max_active_requests = defaults.maxActiveRequests;
    result.max_outstanding_requests = defaults.maxOutstandingRequests;
    result.max_prompt_tokens = defaults.maxPromptTokens;
    result.max_generated_tokens = defaults.maxGeneratedTokens;
    result.max_sessions = defaults.maxSessions;
    result.max_retained_request_handles = defaults.maxRetainedRequestHandles;
    result.max_buffered_tokens_per_request = defaults.maxBufferedTokensPerRequest;
    result.token_window = defaults.tokenWindow;
    result.max_prefill_chunk_size = defaults.maxPrefillChunkSize;
    result.max_outstanding_tokens = defaults.maxOutstandingTokens;
    result.max_session_cache_bytes = defaults.maxSessionCacheBytes;
    *options = result;
    return AILA_INFERENCE_OK;
}
int32_t AILA_INFERENCE_CALL aila_inference_default_model_options(AilaInferenceModelOptions* options, uint32_t size) noexcept {
    if (!options || size < sizeof(*options)) return AILA_INFERENCE_INVALID_ARGUMENT;
    AilaInferenceModelOptions result{};
    result.struct_size = size;
    result.abi_version = AILA_INFERENCE_ABI_VERSION;
    const host::InferenceModelOptions defaults;
    result.max_sequence_length = defaults.maxSequenceLength;
    result.max_sessions = defaults.maxSessions;
    result.max_session_cache_bytes = defaults.maxSessionCacheBytes;
    *options = result;
    return AILA_INFERENCE_OK;
}
int32_t AILA_INFERENCE_CALL aila_inference_default_request_options(AilaInferenceRequestOptions* options, uint32_t size) noexcept {
    if (!options || size < sizeof(*options)) return AILA_INFERENCE_INVALID_ARGUMENT;
    AilaInferenceRequestOptions result{};
    result.struct_size = size;
    result.abi_version = AILA_INFERENCE_ABI_VERSION;
    const wire::GenerationOptions defaults;
    result.max_new_tokens = defaults.maxNewTokens;
    result.prefill_chunk_size = 512;
    result.do_sample = defaults.doSample;
    result.use_fixed_seed = defaults.useFixedSeed;
    result.sampling_seed = defaults.samplingSeed;
    result.temperature = defaults.temperature;
    result.top_k = defaults.topK;
    result.top_p = defaults.topP;
    result.repetition_penalty = defaults.repetitionPenalty;
    result.presence_penalty = defaults.presencePenalty;
    result.frequency_penalty = defaults.frequencyPenalty;
    result.thinking_budget_tokens = -1;
    result.thinking_open_token_id = -1;
    result.thinking_close_token_id = -1;
    *options = result;
    return AILA_INFERENCE_OK;
}

int32_t AILA_INFERENCE_CALL aila_inference_client_start(
    const char* runtime, const char* executable, const AilaInferenceClientOptions* options,
    AilaInferenceClientHandle* output, AilaInferenceError* error) noexcept {
    return aila_inference_client_start_with_batching(runtime, executable, options, 1, output, error);
}
int32_t AILA_INFERENCE_CALL aila_inference_client_start_with_batching(
    const char* runtime, const char* executable, const AilaInferenceClientOptions* options,
    uint32_t maxDecodeBatchSize, AilaInferenceClientHandle* output, AilaInferenceError* error) noexcept {
    return Boundary(error, [&] {
        CheckHandleOutput(output);
        const auto directory = std::filesystem::u8path(Utf8Path(runtime));
        const auto worker = executable ? std::filesystem::u8path(Utf8Path(executable)) : std::filesystem::path{};
        Require(directory.is_absolute(), "Runtime directory must be absolute");
        auto settings = ClientOptions(options, maxDecodeBatchSize);
        auto entry = std::make_shared<ClientEntry>();
        entry->limits = settings.limits;
        auto reservation = Handles().Reserve(Kind::Client);
        entry->value = host::InferenceClient::Start(directory, worker, AILA_BUILD_ID, std::move(settings));
        *output = reservation.Commit(std::move(entry));
        return AILA_INFERENCE_OK;
    });
}
int32_t AILA_INFERENCE_CALL aila_inference_client_snapshot(
    AilaInferenceClientHandle handle, AilaInferenceClientInfo* output, AilaInferenceError* error) noexcept {
    return Boundary(error, [&] {
        CheckHeader(output);
        const auto entry = Handles().Acquire<ClientEntry>(handle, Kind::Client);
        const auto snapshot = entry->value->Snapshot();
        auto result = Output(output);
        result.domain_status = DomainStatus(snapshot.status);
        result.outstanding_requests = snapshot.outstandingRequests;
        result.retained_request_handles = snapshot.retainedRequestHandles;
        result.pending_commands = snapshot.pendingCommands;
        result.queued_commands = snapshot.queuedCommands;
        result.reserved_tokens = snapshot.reservedTokens;
        result.ingress_bytes = snapshot.ingressBytes;
        result.transport_bytes = snapshot.transportBytes;
        result.peak_ingress_bytes = snapshot.peakIngressBytes;
        result.peak_transport_bytes = snapshot.peakTransportBytes;
        result.dropped_logs = snapshot.droppedLogs;
        CopyText(result.epoch, snapshot.epoch);
        CopyText(result.message, snapshot.error);
        *output = result;
        return AILA_INFERENCE_OK;
    });
}
int32_t AILA_INFERENCE_CALL aila_inference_client_query_memory(
    AilaInferenceClientHandle handle, uint32_t timeout, AilaInferenceRuntimeMemoryInfo* output,
    AilaInferenceError* error) noexcept {
    return Boundary(error, [&] {
        CheckHeader(output);
        const auto entry = Handles().Acquire<ClientEntry>(handle, Kind::Client);
        const auto memory = entry->value->QueryMemory(Milliseconds(timeout));
        auto result = Output(output);
        switch (memory.state) {
        case wire::RuntimeMemoryState::Uninitialized: result.state = AILA_INFERENCE_MEMORY_UNINITIALIZED; break;
        case wire::RuntimeMemoryState::Busy: result.state = AILA_INFERENCE_MEMORY_BUSY; break;
        case wire::RuntimeMemoryState::Ready: result.state = AILA_INFERENCE_MEMORY_READY; break;
        }
        if (memory.usage) {
            const auto& usage = *memory.usage;
            result.device_bytes = usage.deviceBytes;
            result.peak_device_bytes = usage.peakDeviceBytes;
            result.pinned_host_bytes = usage.pinnedHostBytes;
            result.peak_pinned_host_bytes = usage.peakPinnedHostBytes;
            result.device_allocations = usage.deviceAllocations;
            result.pinned_host_allocations = usage.pinnedHostAllocations;
            result.device_limit_bytes = usage.deviceLimitBytes;
            result.pinned_host_limit_bytes = usage.pinnedHostLimitBytes;
        }
        CopyText(result.epoch, entry->value->Snapshot().epoch);
        *output = result;
        return AILA_INFERENCE_OK;
    });
}

int32_t AILA_INFERENCE_CALL aila_inference_client_read_log(
    AilaInferenceClientHandle handle, uint32_t timeout, AilaInferenceLogEvent* output,
    AilaInferenceError* error) noexcept {
    return Boundary(error, [&] {
        static_assert(AILA_INFERENCE_MESSAGE_BYTES > wire::kMaxErrorBytes, "Log output must fit the entire wire message");
        CheckHeader(output);
        const auto entry = Handles().Acquire<ClientEntry>(handle, Kind::Client);
        std::unique_lock<std::mutex> consumer(entry->logReader, std::try_to_lock);
        if (!consumer.owns_lock()) throw ApiError{AILA_INFERENCE_BUSY, "Client already has an active log reader"};
        const auto read = entry->value->ReadLog(Milliseconds(timeout));
        auto result = Output(output);
        result.dropped_logs = read.droppedLogs;
        if (read.status != host::InferenceReadStatus::Event) {
            *output = result;
            return read.status == host::InferenceReadStatus::Timeout ? AILA_INFERENCE_TIMEOUT : AILA_INFERENCE_END_OF_STREAM;
        }
        const auto& event = *read.event;
        result.event_ordinal = event.eventOrdinal;
        result.level = static_cast<uint32_t>(event.level);
        result.message_bytes = static_cast<uint32_t>(event.message.size());
        CopyText(result.epoch, event.epoch);
        CopyText(result.message, event.message);
        *output = result;
        return AILA_INFERENCE_OK;
    });
}
int32_t AILA_INFERENCE_CALL aila_inference_client_set_log_level(
    AilaInferenceClientHandle handle, uint32_t level, uint32_t timeout, AilaInferenceError* error) noexcept {
    return Boundary(error, [&] {
        Require(level <= 3, "Log level must be in 0..3");
        Handles().Acquire<ClientEntry>(handle, Kind::Client)->value->SetLogLevel(level, Milliseconds(timeout));
        return AILA_INFERENCE_OK;
    });
}
int32_t AILA_INFERENCE_CALL aila_inference_client_shutdown(
    AilaInferenceClientHandle handle, uint32_t timeout, AilaInferenceError* error) noexcept {
    return Boundary(error, [&] {
        Handles().Acquire<ClientEntry>(handle, Kind::Client)->value->Shutdown(Milliseconds(timeout));
        return AILA_INFERENCE_OK;
    });
}
int32_t AILA_INFERENCE_CALL aila_inference_model_load(
    AilaInferenceClientHandle client, uint32_t backend, const char* directory,
    const AilaInferenceModelOptions* options, uint32_t timeout,
    AilaInferenceModelHandle* output, AilaInferenceError* error) noexcept {
    return Boundary(error, [&] {
        CheckHandleOutput(output);
        const auto path = Utf8Path(directory);
        const auto settings = ModelOptions(options);
        Require(backend >= AILA_INFERENCE_QWEN3_NF4 && backend <= AILA_INFERENCE_QWEN35_BF16, "Unknown model backend");
        const auto parent = Handles().Acquire<ClientEntry>(client, Kind::Client);
        auto entry = std::make_shared<ModelEntry>();
        entry->limits = parent->limits;
        entry->backend = backend;
        auto reservation = Handles().Reserve(Kind::Model);
        if (backend == AILA_INFERENCE_QWEN3_NF4)
            entry->value = parent->value->LoadQwen3(path, settings, Milliseconds(timeout));
        else if (backend == AILA_INFERENCE_QWEN35_NF4)
            entry->value = parent->value->LoadQwen35(path, settings, Milliseconds(timeout));
        else
            entry->value = parent->value->LoadQwen35Bf16(path, settings, Milliseconds(timeout));
        *output = reservation.Commit(std::move(entry));
        return AILA_INFERENCE_OK;
    });
}
int32_t AILA_INFERENCE_CALL aila_inference_model_info(
    AilaInferenceModelHandle handle, AilaInferenceModelInfo* output, AilaInferenceError* error) noexcept {
    return Boundary(error, [&] {
        CheckHeader(output);
        const auto entry = Handles().Acquire<ModelEntry>(handle, Kind::Model);
        const auto& info = entry->value->Info();
        auto result = Output(output);
        result.model_id = info.modelId;
        result.vocab_size = info.vocabSize;
        result.max_sequence_length = info.maxSequenceLength;
        result.backend = entry->backend;
        result.session_reservation_bytes = info.sessionReservationBytes;
        if (info.memory) {
            result.has_memory_info = 1;
            result.resident_device_bytes = info.memory->residentDeviceBytes;
            result.session_host_reservation_bytes = info.memory->sessionHostReservationBytes;
            result.max_device_bytes = info.memory->limits.maxDeviceBytes;
            result.max_pinned_host_bytes = info.memory->limits.maxPinnedHostBytes;
            result.max_session_host_bytes = info.memory->limits.maxSessionHostBytes;
        }
        CopyText(result.epoch, entry->value->Epoch());
        *output = result;
        return AILA_INFERENCE_OK;
    });
}
int32_t AILA_INFERENCE_CALL aila_inference_session_create(
    AilaInferenceModelHandle model, uint32_t timeout, AilaInferenceSessionHandle* output,
    AilaInferenceError* error) noexcept {
    return Boundary(error, [&] {
        CheckHandleOutput(output);
        const auto parent = Handles().Acquire<ModelEntry>(model, Kind::Model);
        auto entry = std::make_shared<SessionEntry>();
        entry->limits = parent->limits;
        auto reservation = Handles().Reserve(Kind::Session);
        entry->value = parent->value->CreateSession(Milliseconds(timeout));
        *output = reservation.Commit(std::move(entry));
        return AILA_INFERENCE_OK;
    });
}
int32_t AILA_INFERENCE_CALL aila_inference_session_query(
    AilaInferenceSessionHandle handle, uint32_t timeout, AilaInferenceSessionInfo* output,
    AilaInferenceError* error) noexcept {
    return Boundary(error, [&] {
        CheckHeader(output);
        const auto entry = Handles().Acquire<SessionEntry>(handle, Kind::Session);
        SessionInfo(output, entry->value->Query(Milliseconds(timeout)), entry->value->Epoch());
        return AILA_INFERENCE_OK;
    });
}
int32_t AILA_INFERENCE_CALL aila_inference_session_reset(
    AilaInferenceSessionHandle handle, uint32_t timeout, AilaInferenceSessionInfo* output,
    AilaInferenceError* error) noexcept {
    return Boundary(error, [&] {
        CheckHeader(output);
        const auto entry = Handles().Acquire<SessionEntry>(handle, Kind::Session);
        SessionInfo(output, entry->value->Reset(Milliseconds(timeout)), entry->value->Epoch());
        return AILA_INFERENCE_OK;
    });
}
int32_t AILA_INFERENCE_CALL aila_inference_session_truncate(
    AilaInferenceSessionHandle handle, uint32_t length, uint32_t timeout,
    AilaInferenceSessionInfo* output, AilaInferenceError* error) noexcept {
    return Boundary(error, [&] {
        CheckHeader(output);
        const auto entry = Handles().Acquire<SessionEntry>(handle, Kind::Session);
        SessionInfo(output, entry->value->Truncate(length, Milliseconds(timeout)), entry->value->Epoch());
        return AILA_INFERENCE_OK;
    });
}
int32_t AILA_INFERENCE_CALL aila_inference_request_submit(
    AilaInferenceSessionHandle session, const int32_t* prompt, uint32_t count,
    const AilaInferenceRequestOptions* options, uint32_t timeout,
    AilaInferenceRequestHandle* output, AilaInferenceError* error) noexcept {
    return aila_inference_request_submit_with_policy(session, prompt, count, options,
        nullptr, timeout, output, error);
}
int32_t AILA_INFERENCE_CALL aila_inference_request_submit_with_policy(
    AilaInferenceSessionHandle session, const int32_t* prompt, uint32_t count,
    const AilaInferenceRequestOptions* options, const AilaInferenceRequestPolicy* policy, uint32_t timeout,
    AilaInferenceRequestHandle* output, AilaInferenceError* error) noexcept {
    return Boundary(error, [&] {
        CheckHandleOutput(output);
        if (policy) {
            CheckHeader(policy);
            Require(policy->thinking_policy <= AILA_INFERENCE_THINKING_CLOSE_ONCE, "Unknown thinking policy");
            Require(policy->sampling_policy <= AILA_INFERENCE_SAMPLING_HOST, "Unknown sampling policy");
            Require(policy->prefill_logits_policy <= AILA_INFERENCE_PREFILL_LOGITS_LEGACY, "Unknown prefill logits policy");
            for (uint32_t field : policy->reserved) Require(field == 0, "Reserved policy fields must be zero");
        }
        const auto parent = Handles().Acquire<SessionEntry>(session, Kind::Session);
        Require(count != 0, "Prompt must contain at least one token");
        auto tokens = Tokens(prompt, count, parent->limits.maxPromptTokens);
        auto settings = RequestOptions(options, parent->limits);
        if (policy) {
            if (settings.thinkingBudget)
                settings.thinkingBudget->closeOnce = policy->thinking_policy == AILA_INFERENCE_THINKING_CLOSE_ONCE;
            settings.forceHostSampling = policy->sampling_policy == AILA_INFERENCE_SAMPLING_HOST;
            settings.legacyPrefillLogits = policy->prefill_logits_policy == AILA_INFERENCE_PREFILL_LOGITS_LEGACY;
        }
        auto entry = std::make_shared<RequestEntry>();
        auto reservation = Handles().Reserve(Kind::Request);
        entry->value = parent->value->Submit(std::move(tokens), std::move(settings), Milliseconds(timeout));
        *output = reservation.Commit(std::move(entry));
        return AILA_INFERENCE_OK;
    });
}
int32_t AILA_INFERENCE_CALL aila_inference_request_prefill(
    AilaInferenceSessionHandle session, const int32_t* prompt, uint32_t count,
    uint32_t chunkSize, uint32_t timeout, AilaInferenceRequestHandle* output,
    AilaInferenceError* error) noexcept {
    return Boundary(error, [&] {
        CheckHandleOutput(output);
        const auto parent = Handles().Acquire<SessionEntry>(session, Kind::Session);
        Require(count != 0, "Prompt must contain at least one token");
        Require(chunkSize != 0 && chunkSize <= parent->limits.maxPrefillChunkSize,
                "Prefill chunk size is outside the negotiated limits");
        auto tokens = Tokens(prompt, count, parent->limits.maxPromptTokens);
        auto entry = std::make_shared<RequestEntry>();
        auto reservation = Handles().Reserve(Kind::Request);
        entry->value = parent->value->Prefill(std::move(tokens), chunkSize, Milliseconds(timeout));
        *output = reservation.Commit(std::move(entry));
        return AILA_INFERENCE_OK;
    });
}
int32_t AILA_INFERENCE_CALL aila_inference_request_read(
    AilaInferenceRequestHandle handle, uint32_t timeout, AilaInferenceEvent* output,
    AilaInferenceError* error) noexcept {
    return Boundary(error, [&] {
        CheckHeader(output); // Never lose a token because an output buffer is invalid.
        const auto entry = Handles().Acquire<RequestEntry>(handle, Kind::Request);
        std::unique_lock<std::mutex> consumer(entry->reader, std::try_to_lock);
        if (!consumer.owns_lock()) throw ApiError{AILA_INFERENCE_BUSY, "Request already has an active reader"};
        const auto read = entry->value->Read(Milliseconds(timeout));
        auto result = Output(output);
        result.token_id = -1;
        if (read.status != host::InferenceReadStatus::Event) {
            *output = result;
            return read.status == host::InferenceReadStatus::Timeout ? AILA_INFERENCE_TIMEOUT : AILA_INFERENCE_END_OF_STREAM;
        }
        const auto& event = *read.event;
        result.kind = event.terminal ? AILA_INFERENCE_EVENT_TERMINAL : AILA_INFERENCE_EVENT_TOKEN;
        result.local_failure = event.localFailure;
        result.sequence = event.sequence;
        result.token_id = event.tokenId;
        result.delivery_abandoned = event.deliveryAbandoned;
        result.delivered_tokens = event.deliveredTokens;
        result.discarded_published_tokens = event.discardedPublishedTokens;
        result.progress = Progress(event.summary);
        *output = result;
        return AILA_INFERENCE_OK;
    });
}
int32_t AILA_INFERENCE_CALL aila_inference_request_cancel(
    AilaInferenceRequestHandle handle, uint32_t timeout, uint32_t* output, AilaInferenceError* error) noexcept {
    return Boundary(error, [&] {
        Require(output != nullptr, "A cancellation output pointer is required");
        *output = 0;
        *output = Handles().Acquire<RequestEntry>(handle, Kind::Request)->value->Cancel(Milliseconds(timeout));
        return AILA_INFERENCE_OK;
    });
}
int32_t AILA_INFERENCE_CALL aila_inference_request_query(
    AilaInferenceRequestHandle handle, uint32_t timeout, AilaInferenceRequestInfo* output,
    AilaInferenceError* error) noexcept {
    return Boundary(error, [&] {
        CheckHeader(output);
        const auto entry = Handles().Acquire<RequestEntry>(handle, Kind::Request);
        const auto info = entry->value->Query(Milliseconds(timeout));
        auto result = Output(output);
        result.inference_id = entry->value->Id();
        result.buffered_tokens = info.bufferedTokens;
        result.last_sent_sequence = info.lastSentSequence;
        result.last_acknowledged_sequence = info.lastAcknowledgedSequence;
        result.completion_ready = info.completionReady;
        CopyText(result.epoch, entry->value->Epoch());
        result.progress = Progress(info.summary);
        *output = result;
        return AILA_INFERENCE_OK;
    });
}

#define AILA_DEFINE_RELEASE(name, type, kind) \
    int32_t AILA_INFERENCE_CALL name(type handle, AilaInferenceError* error) noexcept { \
        return Boundary(error, [&] { Handles().Release(handle, Kind::kind); return AILA_INFERENCE_OK; }); \
    }
AILA_DEFINE_RELEASE(aila_inference_client_release, AilaInferenceClientHandle, Client)
AILA_DEFINE_RELEASE(aila_inference_model_release, AilaInferenceModelHandle, Model)
AILA_DEFINE_RELEASE(aila_inference_session_release, AilaInferenceSessionHandle, Session)
AILA_DEFINE_RELEASE(aila_inference_request_release, AilaInferenceRequestHandle, Request)
#undef AILA_DEFINE_RELEASE
