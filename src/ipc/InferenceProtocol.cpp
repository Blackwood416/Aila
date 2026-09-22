#include "ipc/InferenceProtocol.hpp"

#include "simdjson.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>

namespace aila::ipc::v2 {
namespace {

using Object = simdjson::dom::object;
using Element = simdjson::dom::element;
constexpr size_t MaxCapabilities = 32;
constexpr size_t MaxCapabilityBytes = 64;
constexpr size_t MaxModelPathBytes = 32 * 1024;
constexpr size_t MaxIdentityPathBytes = 128 * 1024;

constexpr std::pair<Method, std::string_view> MethodNames[] = {
    {Method::WorkerConfigure, "worker.configure"}, {Method::WorkerQuery, "worker.query"},
    {Method::ModelLoad, "model.load"}, {Method::SessionCreate, "session.create"},
    {Method::SessionQuery, "session.query"}, {Method::RequestSubmit, "request.submit"},
    {Method::RequestCancel, "request.cancel"}, {Method::RequestQuery, "request.query"},
    {Method::RequestAck, "request.ack"}, {Method::HandleRelease, "handle.release"},
    {Method::WorkerShutdown, "worker.shutdown"},
    {Method::SessionReset, "session.reset"}, {Method::SessionTruncate, "session.truncate"},
    {Method::WorkerSetLogLevel, "worker.set_log_level"},
    {Method::WorkerMemoryQuery, "worker.memory_query"},
};
constexpr std::pair<ObjectType, std::string_view> ObjectNames[] = {
    {ObjectType::Model, "model"}, {ObjectType::Session, "session"}, {ObjectType::Request, "request"},
};
constexpr std::pair<RequestStatus, std::string_view> StatusNames[] = {
    {RequestStatus::Queued, "queued"}, {RequestStatus::Prefill, "prefill"},
    {RequestStatus::Decode, "decode"}, {RequestStatus::Finalizing, "finalizing"},
    {RequestStatus::Completed, "completed"}, {RequestStatus::Cancelled, "cancelled"},
    {RequestStatus::Failed, "failed"},
};
constexpr std::pair<EndReason, std::string_view> ReasonNames[] = {
    {EndReason::None, "none"}, {EndReason::Length, "length"}, {EndReason::StopToken, "stop_token"},
    {EndReason::Cancelled, "cancelled"}, {EndReason::OutputBackpressure, "output_backpressure"},
    {EndReason::SchedulerStopped, "scheduler_stopped"}, {EndReason::RuntimeFault, "runtime_fault"},
    {EndReason::Failed, "failed"},
};
constexpr std::pair<ErrorCode, std::string_view> ErrorNames[] = {
    {ErrorCode::InvalidArgument, "InvalidArgument"}, {ErrorCode::Busy, "Busy"},
    {ErrorCode::ContextOverflow, "ContextOverflow"}, {ErrorCode::UnsupportedModel, "UnsupportedModel"},
    {ErrorCode::ResourceLimit, "ResourceLimit"}, {ErrorCode::RuntimeFault, "RuntimeFault"},
    {ErrorCode::SchedulerStopped, "SchedulerStopped"}, {ErrorCode::UnsupportedProtocol, "UnsupportedProtocol"},
    {ErrorCode::UnsupportedCapability, "UnsupportedCapability"}, {ErrorCode::StaleEpoch, "StaleEpoch"},
    {ErrorCode::InvalidHandle, "InvalidHandle"}, {ErrorCode::ProtocolError, "ProtocolError"},
    {ErrorCode::RequestFailed, "RequestFailed"}, {ErrorCode::TransportError, "TransportError"},
};
constexpr std::pair<DomainStatus, std::string_view> DomainNames[] = {
    {DomainStatus::Starting, "starting"}, {DomainStatus::Ready, "ready"},
    {DomainStatus::Stopping, "stopping"}, {DomainStatus::Faulted, "faulted"},
    {DomainStatus::Closed, "closed"},
};
constexpr std::pair<RuntimeMemoryState, std::string_view> MemoryStateNames[] = {
    {RuntimeMemoryState::Uninitialized, "uninitialized"}, {RuntimeMemoryState::Busy, "busy"},
    {RuntimeMemoryState::Ready, "ready"},
};

template <typename T, size_t N>
std::string_view Name(T value, const std::pair<T, std::string_view> (&names)[N]) noexcept {
    for (const auto& item : names) if (item.first == value) return item.second;
    return {};
}
template <typename T, size_t N>
bool ParseName(std::string_view value, T& output, const std::pair<T, std::string_view> (&names)[N]) noexcept {
    for (const auto& item : names) {
        if (item.second == value) { output = item.first; return true; }
    }
    return false;
}

bool Fail(ValidationError& error, ErrorCode code, std::string_view message) {
    error.code = code;
    error.message.assign(message);
    return false;
}
template <typename Function>
bool Guard(ValidationError& error, Function&& function) {
    error.message.clear();
    try {
        return function();
    } catch (const std::bad_alloc&) {
        return Fail(error, ErrorCode::ResourceLimit, "Memory allocation failed while processing an IPC frame");
    } catch (const std::exception&) {
        return Fail(error, ErrorCode::ProtocolError, "Unexpected exception while processing an IPC frame");
    } catch (...) {
        return Fail(error, ErrorCode::ProtocolError, "Unknown exception while processing an IPC frame");
    }
}

bool Finite(double value) noexcept {
    static_assert(sizeof(double) == sizeof(uint64_t) && std::numeric_limits<double>::is_iec559);
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    // This validation must also work if a surrounding target enables fast math.
    return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}
bool FloatRepresentable(double value) noexcept {
    return Finite(value) && value <= static_cast<double>((std::numeric_limits<float>::max)()) &&
           value >= -static_cast<double>((std::numeric_limits<float>::max)());
}
bool PositiveId(uint64_t value) noexcept {
    return value != 0 && value != (std::numeric_limits<uint64_t>::max)();
}
bool Utf8(std::string_view value) noexcept {
    return value.empty() || simdjson::validate_utf8(value.data(), value.size());
}
bool SafeText(std::string_view value, size_t maximum, bool allowNul = false) noexcept {
    return value.size() <= maximum && Utf8(value) && (allowNul || value.find('\0') == std::string_view::npos);
}

// Every key is visited, including duplicates that object.at_key() alone would
// hide. DOM keys are already unescaped, so "epoch" and "\u0065poch" collide.
bool Fields(Object object, std::initializer_list<std::string_view> required,
            std::initializer_list<std::string_view> optional, ValidationError& error,
            ErrorCode code = ErrorCode::InvalidArgument) {
    std::array<std::string_view, 64> allowed{};
    if (required.size() > allowed.size() || optional.size() > allowed.size() - required.size()) {
        return Fail(error, ErrorCode::ProtocolError, "Internal schema has too many fields");
    }
    size_t count = 0;
    for (auto key : required) allowed[count++] = key;
    for (auto key : optional) allowed[count++] = key;
    uint64_t seen = 0;
    for (const auto field : object) {
        size_t index = 0;
        while (index < count && allowed[index] != field.key) ++index;
        if (index == count) return Fail(error, code, "JSON object contains an unknown field");
        const uint64_t bit = UINT64_C(1) << index;
        if ((seen & bit) != 0) return Fail(error, code, "JSON object contains a duplicate field");
        seen |= bit;
    }
    const uint64_t requiredMask = required.size() == 64 ? (std::numeric_limits<uint64_t>::max)()
                                                       : (UINT64_C(1) << required.size()) - 1;
    if ((seen & requiredMask) != requiredMask) return Fail(error, code, "JSON object is missing a required field");
    return true;
}
bool GetObject(Element element, Object& object, ValidationError& error, ErrorCode code) {
    if (element.get_object().get(object) != simdjson::SUCCESS) {
        return Fail(error, code, "JSON value must be an object");
    }
    return true;
}
bool ChildObject(Object object, const char* name, Object& child, ValidationError& error, ErrorCode code) {
    Element element;
    if (object.at_key(name).get(element) != simdjson::SUCCESS) return Fail(error, code, "Missing object field");
    return GetObject(element, child, error, code);
}
template <typename T>
bool Unsigned(Object object, const char* name, T& output, ValidationError& error, ErrorCode code) {
    static_assert(std::is_unsigned_v<T>);
    uint64_t value = 0;
    if (object.at_key(name).get_uint64().get(value) != simdjson::SUCCESS ||
        value > static_cast<uint64_t>((std::numeric_limits<T>::max)())) {
        return Fail(error, code, std::string(name) + " must be an unsigned integer in range");
    }
    output = static_cast<T>(value);
    return true;
}
bool Boolean(Object object, const char* name, bool& output, ValidationError& error, ErrorCode code) {
    if (object.at_key(name).get_bool().get(output) != simdjson::SUCCESS) {
        return Fail(error, code, std::string(name) + " must be a boolean");
    }
    return true;
}
bool Number(Object object, const char* name, double& output, ValidationError& error, ErrorCode code) {
    if (object.at_key(name).get_double().get(output) != simdjson::SUCCESS || !FloatRepresentable(output)) {
        return Fail(error, code, std::string(name) + " must be a finite float-representable number");
    }
    return true;
}
bool Text(Object object, const char* name, std::string& output, size_t maximum,
          ValidationError& error, ErrorCode code, bool allowNul = false) {
    std::string_view value;
    if (object.at_key(name).get_string().get(value) != simdjson::SUCCESS || !SafeText(value, maximum, allowNul)) {
        return Fail(error, code, std::string(name) + " must be a bounded UTF-8 string");
    }
    output.assign(value);
    return true;
}
template <typename T, typename Parser>
bool Enum(Object object, const char* name, T& output, Parser parse, ValidationError& error, ErrorCode code) {
    std::string_view value;
    if (object.at_key(name).get_string().get(value) != simdjson::SUCCESS || !parse(value, output)) {
        return Fail(error, code, std::string(name) + " has an unknown enum value");
    }
    return true;
}
bool Epoch(Object object, std::string& epoch, ValidationError& error, ErrorCode code) {
    if (!Text(object, "epoch", epoch, 32, error, code)) return false;
    return valid_epoch(epoch) || Fail(error, code, "epoch must contain exactly 32 lowercase hexadecimal digits");
}
bool Id(Object object, const char* name, uint64_t& value, ValidationError& error, ErrorCode code) {
    return Unsigned(object, name, value, error, code) &&
           (PositiveId(value) || Fail(error, code, "Object IDs must be positive and cannot use the reserved maximum"));
}

class Json final {
public:
    explicit Json(size_t maximum = 64 * 1024) : maximum(maximum) { Raw("{"); }
    void String(const char* key, std::string_view value) { Key(key); Quoted(value); }
    void U64(const char* key, uint64_t value) { Key(key); Integer(value); }
    void Bool(const char* key, bool value) { Key(key); Raw(value ? "true" : "false"); }
    void Null(const char* key) { Key(key); Raw("null"); }
    void Number(const char* key, double value) {
        Key(key);
        if (!Finite(value)) { good = false; return; }
        char storage[128];
        const auto result = std::to_chars(storage, storage + sizeof(storage), value,
                                           std::chars_format::general, std::numeric_limits<double>::max_digits10);
        if (result.ec != std::errc{}) { good = false; return; }
        Raw(std::string_view(storage, static_cast<size_t>(result.ptr - storage)));
    }
    void ObjectValue(const char* key, std::string_view value) { Key(key); Raw(value); }
    void Strings(const char* key, const std::vector<std::string>& values) {
        if (!good || values.size() > maximum) { good = false; return; }
        Key(key); Raw("[");
        bool initial = true;
        for (const auto& value : values) { if (!good) break; if (!initial) Raw(","); initial = false; Quoted(value); }
        Raw("]");
    }
    template <typename T>
    void Integers(const char* key, const std::vector<T>& values) {
        if (!good || values.size() > maximum) { good = false; return; }
        Key(key); Raw("[");
        bool initial = true;
        for (const auto value : values) {
            if (!good) break;
            if (!initial) Raw(",");
            initial = false;
            char storage[32];
            const auto result = std::to_chars(storage, storage + sizeof(storage), value);
            if (result.ec != std::errc{}) { good = false; return; }
            Raw(std::string_view(storage, static_cast<size_t>(result.ptr - storage)));
        }
        Raw("]");
    }
    bool Finish(std::string& value, ValidationError& error) {
        Raw("}");
        if (!good) return Fail(error, ErrorCode::InvalidArgument, "JSON encoding exceeded its bound or contained invalid text/numbers");
        value = std::move(output);
        return true;
    }

private:
    void Raw(std::string_view value) {
        if (!good || value.size() > maximum - output.size()) { good = false; return; }
        output.append(value);
    }
    void Key(const char* key) { if (!first) Raw(","); first = false; Quoted(key); Raw(":"); }
    void Integer(uint64_t value) {
        char storage[32];
        const auto result = std::to_chars(storage, storage + sizeof(storage), value);
        if (result.ec != std::errc{}) { good = false; return; }
        Raw(std::string_view(storage, static_cast<size_t>(result.ptr - storage)));
    }
    void Quoted(std::string_view value) {
        if (!good || value.size() > maximum - output.size() || maximum - output.size() - value.size() < 2) {
            good = false;
            return;
        }
        if (!Utf8(value)) { good = false; return; }
        Raw("\"");
        constexpr char Hex[] = "0123456789abcdef";
        for (const unsigned char byte : value) {
            if (!good) break;
            if (byte == '"') Raw("\\\"");
            else if (byte == '\\') Raw("\\\\");
            else if (byte < 0x20) {
                const char escaped[] = {'\\', 'u', '0', '0', Hex[byte >> 4], Hex[byte & 15]};
                Raw(std::string_view(escaped, sizeof(escaped)));
            } else {
                const char character = static_cast<char>(byte);
                Raw(std::string_view(&character, 1));
            }
        }
        Raw("\"");
    }
    std::string output;
    size_t maximum;
    bool first = true;
    bool good = true;
};

#define AILA_V2_LIMIT_FIELDS(X) \
    X(maxActiveRequests) X(maxOutstandingRequests) X(maxPromptTokens) X(maxGeneratedTokens) \
    X(maxStopTokenIds) X(maxOutstandingTokens) X(maxPrefillChunkSize) X(maxBufferedTokensPerRequest) \
    X(maxLoadedModels) X(maxSessions) X(maxSessionCacheBytes) X(maxRetainedRequestHandles) X(tokenWindow) \
    X(maxTokenFrameBytes) X(maxTerminalFrameBytes) X(maxFinalFrameBytes) X(maxHeaderBytes) X(maxAttachmentBytes) \
    X(maxPendingCommands) X(reservedControlCommands) X(maxIngressBytes) X(maxTransportBytes) X(maxLogQueueBytes) \
    X(maxShutdownGraceMs) X(connectionWriteTimeoutMs)

bool LimitsJson(const Limits& limits, std::string& output, ValidationError& error) {
    Json json;
#define AILA_WRITE_LIMIT(field) json.U64(#field, limits.field);
    AILA_V2_LIMIT_FIELDS(AILA_WRITE_LIMIT)
#undef AILA_WRITE_LIMIT
    return json.Finish(output, error);
}
bool ReadLimits(Object object, Limits& limits, ValidationError& error, ErrorCode code) {
#define AILA_LIMIT_NAME(field) #field,
    if (!Fields(object, {AILA_V2_LIMIT_FIELDS(AILA_LIMIT_NAME)}, {}, error, code)) return false;
#undef AILA_LIMIT_NAME
#define AILA_READ_LIMIT(field) if (!Unsigned(object, #field, limits.field, error, code)) return false;
    AILA_V2_LIMIT_FIELDS(AILA_READ_LIMIT)
#undef AILA_READ_LIMIT
    return true;
}

bool AllocationLimitsJson(const AllocationLimits& limits, std::string& output, ValidationError& error) {
    Json json;
    json.U64("maxDeviceBytes", limits.maxDeviceBytes);
    json.U64("maxPinnedHostBytes", limits.maxPinnedHostBytes);
    json.U64("maxSessionHostBytes", limits.maxSessionHostBytes);
    return json.Finish(output, error);
}

bool ReadAllocationLimits(Object object, AllocationLimits& limits, ValidationError& error, ErrorCode code) {
    return Fields(object, {"maxDeviceBytes", "maxPinnedHostBytes", "maxSessionHostBytes"}, {}, error, code) &&
        Unsigned(object, "maxDeviceBytes", limits.maxDeviceBytes, error, code) &&
        Unsigned(object, "maxPinnedHostBytes", limits.maxPinnedHostBytes, error, code) &&
        Unsigned(object, "maxSessionHostBytes", limits.maxSessionHostBytes, error, code);
}

bool ReadOptionalAllocationLimits(Object object, std::optional<AllocationLimits>& limits,
                                  ValidationError& error, ErrorCode code) {
    if (object.at_key("allocationLimits").error() == simdjson::NO_SUCH_FIELD) return true;
    Object child;
    AllocationLimits value;
    if (!ChildObject(object, "allocationLimits", child, error, code) ||
        !ReadAllocationLimits(child, value, error, code)) return false;
    limits = value;
    return true;
}

bool ModelMemoryJson(const ModelMemoryInfo& memory, std::string& output, ValidationError& error) {
    std::string limits;
    if (!AllocationLimitsJson(memory.limits, limits, error)) return false;
    Json json;
    json.ObjectValue("limits", limits);
    json.U64("residentDeviceBytes", memory.residentDeviceBytes);
    json.U64("sessionHostReservationBytes", memory.sessionHostReservationBytes);
    return json.Finish(output, error);
}

bool ReadOptionalModelMemory(Object object, std::optional<ModelMemoryInfo>& memory,
                             ValidationError& error, ErrorCode code) {
    if (object.at_key("memory").error() == simdjson::NO_SUCH_FIELD) return true;
    Object child, limits;
    ModelMemoryInfo value;
    if (!ChildObject(object, "memory", child, error, code) ||
        !Fields(child, {"limits", "residentDeviceBytes", "sessionHostReservationBytes"}, {}, error, code) ||
        !ChildObject(child, "limits", limits, error, code) ||
        !ReadAllocationLimits(limits, value.limits, error, code) ||
        !Unsigned(child, "residentDeviceBytes", value.residentDeviceBytes, error, code) ||
        !Unsigned(child, "sessionHostReservationBytes", value.sessionHostReservationBytes, error, code)) return false;
    if (value.limits.maxDeviceBytes && value.residentDeviceBytes > value.limits.maxDeviceBytes)
        return Fail(error, code, "Model residency exceeds its device allocation limit");
    memory = value;
    return true;
}

#define AILA_V2_MEMORY_FIELDS(X) \
    X(deviceBytes) X(peakDeviceBytes) X(pinnedHostBytes) X(peakPinnedHostBytes) \
    X(deviceAllocations) X(pinnedHostAllocations) X(deviceLimitBytes) X(pinnedHostLimitBytes)

bool ValidMemoryUsage(const RuntimeMemoryUsage& value, ValidationError& error, ErrorCode code) {
    if (value.deviceBytes > value.peakDeviceBytes || value.pinnedHostBytes > value.peakPinnedHostBytes ||
        (value.deviceLimitBytes && value.deviceBytes > value.deviceLimitBytes) ||
        (value.pinnedHostLimitBytes && value.pinnedHostBytes > value.pinnedHostLimitBytes) ||
        (value.deviceBytes == 0) != (value.deviceAllocations == 0) ||
        (value.pinnedHostBytes == 0) != (value.pinnedHostAllocations == 0) ||
        value.deviceAllocations > value.deviceBytes || value.pinnedHostAllocations > value.pinnedHostBytes)
        return Fail(error, code, "Inconsistent runtime allocation counters");
    // A previous model can have had a higher limit: lifetime peaks are not
    // constrained by the current limit after all models have been released.
    return true;
}

bool WriteRuntimeMemory(Json& json, const RuntimeMemoryInfo& memory, ValidationError& error) {
    if (Name(memory.state, MemoryStateNames).empty() ||
        (memory.state == RuntimeMemoryState::Ready) != memory.usage.has_value())
        return Fail(error, ErrorCode::InvalidArgument, "Runtime memory state/usage mismatch");
    json.String("state", Name(memory.state, MemoryStateNames));
    if (!memory.usage) { json.Null("memory"); return true; }
    if (!ValidMemoryUsage(*memory.usage, error, ErrorCode::InvalidArgument)) return false;
    Json data;
#define AILA_WRITE_MEMORY(field) data.U64(#field, memory.usage->field);
    AILA_V2_MEMORY_FIELDS(AILA_WRITE_MEMORY)
#undef AILA_WRITE_MEMORY
    std::string encoded;
    if (!data.Finish(encoded, error)) return false;
    json.ObjectValue("memory", encoded);
    return true;
}

bool ReadRuntimeMemory(Object object, RuntimeMemoryInfo& memory, ValidationError& error, ErrorCode code) {
    if (!Fields(object, {"epoch", "state", "memory"}, {}, error, code) ||
        !Enum(object, "state", memory.state, parse_runtime_memory_state, error, code)) return false;
    if (memory.state != RuntimeMemoryState::Ready) {
        Element element;
        if (object.at_key("memory").get(element) != simdjson::SUCCESS || !element.is_null())
            return Fail(error, code, "Unavailable runtime memory must be null");
        return true;
    }
    Object data;
    RuntimeMemoryUsage usage;
#define AILA_MEMORY_NAME(field) #field,
    if (!ChildObject(object, "memory", data, error, code) ||
        !Fields(data, {AILA_V2_MEMORY_FIELDS(AILA_MEMORY_NAME)}, {}, error, code)) return false;
#undef AILA_MEMORY_NAME
#define AILA_READ_MEMORY(field) if (!Unsigned(data, #field, usage.field, error, code)) return false;
    AILA_V2_MEMORY_FIELDS(AILA_READ_MEMORY)
#undef AILA_READ_MEMORY
    if (!ValidMemoryUsage(usage, error, code)) return false;
    memory.usage = usage;
    return true;
}
#undef AILA_V2_MEMORY_FIELDS

bool CapabilityName(std::string_view name) noexcept {
    if (name.empty() || name.size() > MaxCapabilityBytes) return false;
    for (const char value : name) {
        if (!((value >= 'a' && value <= 'z') || (value >= '0' && value <= '9') || value == '_')) return false;
    }
    return true;
}
bool Capabilities(const std::vector<std::string>& values, ValidationError& error, ErrorCode code) {
    if (values.size() > MaxCapabilities) return Fail(error, code, "Too many capabilities");
    for (size_t index = 0; index < values.size(); ++index) {
        if (!CapabilityName(values[index])) return Fail(error, code, "Invalid capability name");
        if (std::find(values.begin(), values.begin() + index, values[index]) != values.begin() + index) {
            return Fail(error, code, "Duplicate capability");
        }
    }
    return true;
}
bool ReadCapabilities(Object object, const char* key, std::vector<std::string>& values,
                      ValidationError& error, ErrorCode code) {
    simdjson::dom::array array;
    if (object.at_key(key).get_array().get(array) != simdjson::SUCCESS || array.size() > MaxCapabilities) {
        return Fail(error, code, "Capabilities must be a bounded array");
    }
    for (const auto element : array) {
        std::string_view value;
        if (element.get_string().get(value) != simdjson::SUCCESS || !CapabilityName(value)) {
            return Fail(error, code, "Capabilities must contain bounded capability strings");
        }
        values.emplace_back(value);
    }
    return Capabilities(values, error, code);
}

bool Envelope(const FrameHeader& header, Channel channel, ValidationError& error) {
    if (header.protocol != kProtocolVersion) return Fail(error, ErrorCode::UnsupportedProtocol, "IPC protocol must be 2");
    if (header.abi != kPublicAbiVersion) return Fail(error, ErrorCode::ProtocolError, "IPC ABI must be 1");
    if (!SafeText(header.method, 64) || header.method.empty()) return Fail(error, ErrorCode::ProtocolError, "Invalid frame method");
    if (channel == Channel::Command) {
        if (header.kind != "request" || !PositiveId(header.request_id)) {
            return Fail(error, ErrorCode::ProtocolError, "Command frames require request kind and a positive RPC ID");
        }
    } else if (channel == Channel::Response) {
        if (header.kind == "handshake") {
            if (header.request_id != 0 || header.method != "worker.hello") {
                return Fail(error, ErrorCode::ProtocolError, "Handshake identity is invalid");
            }
        } else if ((header.kind != "result" && header.kind != "error" && header.kind != "accepted") ||
                   !PositiveId(header.request_id)) {
            return Fail(error, ErrorCode::ProtocolError, "Response identity is invalid");
        }
    } else {
        if (header.kind != "event" || header.request_id == (std::numeric_limits<uint64_t>::max)()) {
            return Fail(error, ErrorCode::ProtocolError, "Event identity is invalid");
        }
    }
    return true;
}
bool InferChannel(const FrameHeader& header, Channel& channel, ValidationError& error) {
    if (header.kind == "request") channel = Channel::Command;
    else if (header.kind == "event") channel = Channel::Event;
    else if (header.kind == "result" || header.kind == "error" || header.kind == "accepted" || header.kind == "handshake") {
        channel = Channel::Response;
    } else return Fail(error, ErrorCode::ProtocolError, "Unknown frame kind");
    return Envelope(header, channel, error);
}
bool SerializeHeader(const FrameHeader& header, std::string_view payload, std::string& output,
                     ValidationError& error, size_t maximum) {
    Json json(maximum);
    json.U64("protocol", header.protocol);
    json.U64("abi", header.abi);
    json.U64("requestId", header.request_id);
    json.String("kind", header.kind);
    json.String("method", header.method);
    json.ObjectValue("payload", payload);
    return json.Finish(output, error);
}
bool Payload(const Frame& frame, Channel channel, const FrameBounds& bounds,
             simdjson::dom::parser& parser, Object& object, ValidationError& error) {
    if (!Envelope(frame.header, channel, error)) return false;
    if (frame.header.payload_json.size() > bounds.maxHeaderBytes || frame.attachment.size() > bounds.maxAttachmentBytes) {
        return Fail(error, ErrorCode::ProtocolError, "Frame exceeds its channel bounds");
    }
    Element root;
    const ErrorCode code = channel == Channel::Command ? ErrorCode::InvalidArgument : ErrorCode::ProtocolError;
    if (parser.parse(frame.header.payload_json).get(root) != simdjson::SUCCESS || !GetObject(root, object, error, code)) {
        return Fail(error, code, "Frame payload must be a valid JSON object");
    }
    std::string header;
    if (!SerializeHeader(frame.header, simdjson::minify(root), header, error, bounds.maxHeaderBytes)) return false;
    if (static_cast<uint64_t>(header.size()) + frame.attachment.size() + kFramePrefixBytes > bounds.maxFrameBytes) {
        return Fail(error, ErrorCode::ProtocolError, "Encoded frame exceeds its channel byte bound");
    }
    return true;
}
Frame BaseFrame(std::string_view kind, std::string_view method, uint64_t rpcId) {
    Frame frame;
    frame.header.protocol = kProtocolVersion;
    frame.header.abi = kPublicAbiVersion;
    frame.header.kind.assign(kind);
    frame.header.method.assign(method);
    frame.header.request_id = rpcId;
    return frame;
}

} // namespace

std::string_view to_string(Method value) noexcept { return Name(value, MethodNames); }
std::string_view to_string(ObjectType value) noexcept { return Name(value, ObjectNames); }
std::string_view to_string(RequestStatus value) noexcept { return Name(value, StatusNames); }
std::string_view to_string(EndReason value) noexcept { return Name(value, ReasonNames); }
std::string_view to_string(ErrorCode value) noexcept { return Name(value, ErrorNames); }
std::string_view to_string(DomainStatus value) noexcept { return Name(value, DomainNames); }
std::string_view to_string(RuntimeMemoryState value) noexcept { return Name(value, MemoryStateNames); }
bool parse_method(std::string_view value, Method& output) noexcept { return ParseName(value, output, MethodNames); }
bool parse_object_type(std::string_view value, ObjectType& output) noexcept { return ParseName(value, output, ObjectNames); }
bool parse_request_status(std::string_view value, RequestStatus& output) noexcept { return ParseName(value, output, StatusNames); }
bool parse_end_reason(std::string_view value, EndReason& output) noexcept { return ParseName(value, output, ReasonNames); }
bool parse_error_code(std::string_view value, ErrorCode& output) noexcept { return ParseName(value, output, ErrorNames); }
bool parse_domain_status(std::string_view value, DomainStatus& output) noexcept { return ParseName(value, output, DomainNames); }
bool parse_runtime_memory_state(std::string_view value, RuntimeMemoryState& output) noexcept { return ParseName(value, output, MemoryStateNames); }
bool is_terminal(RequestStatus value) noexcept {
    return value == RequestStatus::Completed || value == RequestStatus::Cancelled || value == RequestStatus::Failed;
}
bool valid_epoch(std::string_view value) noexcept {
    if (value.size() != 32) return false;
    for (const char byte : value) if (!((byte >= '0' && byte <= '9') || (byte >= 'a' && byte <= 'f'))) return false;
    return true;
}
Limits hard_limits() noexcept { return {}; }
Limits extended_generation_limits() noexcept {
    auto limits = hard_limits();
    limits.maxGeneratedTokens = kExtendedMaxGeneratedTokens;
    return limits;
}
Limits configure_limits_ceiling(const Limits& offered,
                                const std::vector<std::string>& requiredCapabilities) {
    auto ceiling = offered;
    if (ceiling.maxGeneratedTokens == hard_limits().maxGeneratedTokens &&
        std::find(requiredCapabilities.begin(), requiredCapabilities.end(), "extended_generation_v1") != requiredCapabilities.end()) {
        ceiling.maxGeneratedTokens = kExtendedMaxGeneratedTokens;
    }
    return ceiling;
}
std::vector<std::string> default_capabilities() {
    return {"token_requests_v1", "qwen3_nf4_bf16_kv", "session_lifecycle_v1", "multi_inflight_v1",
            "cumulative_ack_v1", "domain_epoch_v1"};
}
std::vector<std::string> supported_capabilities() {
    auto capabilities = default_capabilities();
    capabilities.emplace_back("qwen35_nf4_bf16_kv");
    capabilities.emplace_back("allocation_limits_v1");
    capabilities.emplace_back("qwen35_bf16_bf16_kv");
    capabilities.emplace_back("thinking_budget_v1");
    capabilities.emplace_back("session_context_v1");
    capabilities.emplace_back("prefill_only_v1");
    capabilities.emplace_back("thinking_budget_close_once_v1");
    capabilities.emplace_back("host_sampling_v1");
    capabilities.emplace_back("legacy_prefill_logits_v1");
    capabilities.emplace_back("extended_generation_v1");
    capabilities.emplace_back("worker_log_control_v1");
    capabilities.emplace_back("runtime_memory_v1");
    capabilities.emplace_back("continuous_batching_v1");
    return capabilities;
}
std::string_view model_capability(std::string_view backend, std::string_view kvType) noexcept {
    if (kvType != "bf16") return {};
    if (backend == "qwen3_nf4") return "qwen3_nf4_bf16_kv";
    if (backend == "qwen35_nf4") return "qwen35_nf4_bf16_kv";
    if (backend == "qwen35_bf16") return "qwen35_bf16_bf16_kv";
    return {};
}

bool validate_decode_batch_size(uint32_t size, const Limits& limits, ValidationError& error) {
    return Guard(error, [&] {
        if (size == 0 || size > kMaxDecodeBatchSize || size > limits.maxActiveRequests)
            return Fail(error, ErrorCode::InvalidArgument,
                        "max_decode_batch_size must be in 1..4 and not exceed maxActiveRequests");
        return true;
    });
}

bool validate_limits(const Limits& requested, ValidationError& error, const Limits& ceiling) {
    return Guard(error, [&] {
        const Limits absolute = extended_generation_limits();
#define AILA_CHECK_LIMIT(field) \
        if constexpr (std::string_view(#field) != std::string_view("maxSessionCacheBytes")) { \
            if (requested.field == 0 || requested.field > ceiling.field || ceiling.field > absolute.field) \
                return Fail(error, ErrorCode::InvalidArgument, #field " must be positive and within its hard limit"); \
        }
        AILA_V2_LIMIT_FIELDS(AILA_CHECK_LIMIT)
#undef AILA_CHECK_LIMIT
        if (ceiling.maxSessionCacheBytes != 0 &&
            (requested.maxSessionCacheBytes == 0 || requested.maxSessionCacheBytes > ceiling.maxSessionCacheBytes)) {
            return Fail(error, ErrorCode::InvalidArgument, "Requested cache budget exceeds its offered ceiling");
        }
        if (requested.maxActiveRequests > requested.maxOutstandingRequests ||
            requested.maxOutstandingRequests > requested.maxRetainedRequestHandles ||
            requested.reservedControlCommands >= requested.maxPendingCommands) {
            return Fail(error, ErrorCode::InvalidArgument, "Inconsistent request/handle/control limits");
        }
        if (requested.maxTokenFrameBytes < 768 || requested.maxTerminalFrameBytes < 8192 ||
            requested.maxFinalFrameBytes < 8192 || requested.maxHeaderBytes < 8192 ||
            requested.maxPrefillChunkSize > requested.maxPromptTokens ||
            requested.maxPromptTokens > requested.maxAttachmentBytes / sizeof(int32_t)) {
            return Fail(error, ErrorCode::InvalidArgument, "Frame bounds cannot represent the negotiated request/error sizes");
        }
        const uint64_t frameBytes = kFramePrefixBytes + static_cast<uint64_t>(requested.maxHeaderBytes) + requested.maxAttachmentBytes;
        const uint64_t streamBytes = static_cast<uint64_t>(requested.tokenWindow + uint64_t{1}) * requested.maxTokenFrameBytes +
                                     requested.maxTerminalFrameBytes + requested.maxFinalFrameBytes;
        if (requested.maxIngressBytes < frameBytes || requested.maxTransportBytes < streamBytes ||
            requested.maxOutstandingTokens < 2) {
            return Fail(error, ErrorCode::InvalidArgument, "Byte/token budgets cannot admit one bounded request");
        }
        return true;
    });
}
bool negotiate(const Limits& requested, const std::vector<std::string>& requiredCapabilities,
               const Limits& offered, const std::vector<std::string>& availableCapabilities,
               Limits& selected, ValidationError& error) {
    return Guard(error, [&] {
        if (!validate_limits(offered, error, extended_generation_limits()) ||
            !Capabilities(requiredCapabilities, error, ErrorCode::InvalidArgument) ||
            !Capabilities(availableCapabilities, error, ErrorCode::InvalidArgument)) return false;
        for (const auto& required : requiredCapabilities) {
            if (std::find(availableCapabilities.begin(), availableCapabilities.end(), required) == availableCapabilities.end()) {
                return Fail(error, ErrorCode::UnsupportedCapability, "A required capability is not offered by the Worker");
            }
        }
        if (!validate_limits(requested, error, offered)) return false;
        selected = requested;
        return true;
    });
}

FrameBounds frame_bounds(Channel channel, const Limits& limits) {
    if (channel == Channel::Event) {
        const uint32_t maximum = (std::max)(limits.maxTokenFrameBytes, limits.maxTerminalFrameBytes);
        return {(std::min)(limits.maxHeaderBytes, maximum >= 8 ? maximum - 8 : 0u), 0, maximum};
    }
    const uint32_t attachment = channel == Channel::Command ? limits.maxAttachmentBytes : 0;
    return {limits.maxHeaderBytes, attachment, kFramePrefixBytes + static_cast<uint64_t>(limits.maxHeaderBytes) + attachment};
}
FrameBounds bootstrap_bounds() noexcept { return {1024 * 1024, 0, kFramePrefixBytes + 1024 * 1024}; }
bool inspect_prefix(const std::byte* bytes, size_t count, FrameSizes& sizes,
                    ValidationError& error, const FrameBounds& bounds) {
    return Guard(error, [&] {
        if (!bytes || count < kFramePrefixBytes) return Fail(error, ErrorCode::ProtocolError, "Incomplete eight-byte frame prefix");
        FrameSizes parsed;
        parsed.headerBytes = read_u32_le(bytes);
        parsed.attachmentBytes = read_u32_le(bytes + 4);
        const uint64_t total = kFramePrefixBytes + static_cast<uint64_t>(parsed.headerBytes) + parsed.attachmentBytes;
        if (parsed.headerBytes == 0 || parsed.headerBytes > bounds.maxHeaderBytes ||
            parsed.attachmentBytes > bounds.maxAttachmentBytes || total > bounds.maxFrameBytes ||
            total > static_cast<uint64_t>((std::numeric_limits<size_t>::max)())) {
            return Fail(error, ErrorCode::ProtocolError, "Declared frame sizes exceed their channel bounds");
        }
        parsed.totalBytes = static_cast<size_t>(total);
        sizes = parsed;
        return true;
    });
}
bool encode_frame(const Frame& frame, std::vector<std::byte>& bytes,
                  ValidationError& error, const FrameBounds& bounds) {
    return Guard(error, [&] {
        Channel channel;
        if (!InferChannel(frame.header, channel, error)) return false;
        simdjson::dom::parser parser;
        Object object;
        if (!Payload(frame, channel, bounds, parser, object, error)) return false;
        Element root;
        if (parser.parse(frame.header.payload_json).get(root) != simdjson::SUCCESS) return Fail(error, ErrorCode::ProtocolError, "Invalid payload");
        std::string header;
        if (!SerializeHeader(frame.header, simdjson::minify(root), header, error, bounds.maxHeaderBytes)) return false;
        std::vector<std::byte> encoded(kFramePrefixBytes + header.size() + frame.attachment.size());
        write_u32_le(encoded.data(), static_cast<uint32_t>(header.size()));
        write_u32_le(encoded.data() + 4, static_cast<uint32_t>(frame.attachment.size()));
        std::memcpy(encoded.data() + kFramePrefixBytes, header.data(), header.size());
        if (!frame.attachment.empty()) std::memcpy(encoded.data() + kFramePrefixBytes + header.size(), frame.attachment.data(), frame.attachment.size());
        bytes = std::move(encoded);
        return true;
    });
}
bool decode_frame(const std::vector<std::byte>& bytes, Frame& frame,
                  ValidationError& error, const FrameBounds& bounds) {
    return Guard(error, [&] {
        FrameSizes sizes;
        if (!inspect_prefix(bytes.data(), bytes.size(), sizes, error, bounds)) return false;
        if (sizes.totalBytes != bytes.size()) return Fail(error, ErrorCode::ProtocolError, "Frame is truncated or contains trailing bytes");
        simdjson::dom::parser parser;
        Element root;
        if (parser.parse(reinterpret_cast<const char*>(bytes.data() + kFramePrefixBytes), sizes.headerBytes).get(root) != simdjson::SUCCESS) {
            return Fail(error, ErrorCode::ProtocolError, "Header is not valid UTF-8 JSON");
        }
        Object object;
        constexpr auto code = ErrorCode::ProtocolError;
        if (!GetObject(root, object, error, code) ||
            !Fields(object, {"protocol", "abi", "requestId", "kind", "method", "payload"}, {}, error, code)) return false;
        Frame parsed;
        if (!Unsigned(object, "protocol", parsed.header.protocol, error, code) ||
            !Unsigned(object, "abi", parsed.header.abi, error, code) ||
            !Unsigned(object, "requestId", parsed.header.request_id, error, code) ||
            !Text(object, "kind", parsed.header.kind, 16, error, code) ||
            !Text(object, "method", parsed.header.method, 64, error, code)) return false;
        Channel channel;
        if (!InferChannel(parsed.header, channel, error)) return false;
        Element payload;
        Object payloadObject;
        if (object.at_key("payload").get(payload) != simdjson::SUCCESS || !GetObject(payload, payloadObject, error, code)) return false;
        parsed.header.payload_json = simdjson::minify(payload);
        parsed.attachment.assign(bytes.begin() + static_cast<std::ptrdiff_t>(kFramePrefixBytes + sizes.headerBytes), bytes.end());
        frame = std::move(parsed);
        return true;
    });
}

bool encode_token_ids(const std::vector<int32_t>& tokens, std::vector<std::byte>& bytes,
                      ValidationError& error, const Limits& limits) {
    return Guard(error, [&] {
        if (tokens.size() > limits.maxPromptTokens || tokens.size() > limits.maxAttachmentBytes / 4) {
            return Fail(error, ErrorCode::ResourceLimit, "Token attachment exceeds its limit");
        }
        for (const auto token : tokens) if (token < 0) return Fail(error, ErrorCode::InvalidArgument, "Token IDs cannot be negative");
        std::vector<std::byte> encoded(tokens.size() * 4);
        for (size_t index = 0; index < tokens.size(); ++index) write_u32_le(encoded.data() + index * 4, static_cast<uint32_t>(tokens[index]));
        bytes = std::move(encoded);
        return true;
    });
}
bool decode_token_ids(const std::vector<std::byte>& bytes, uint64_t tokenCount,
                      std::vector<int32_t>& tokens, ValidationError& error, const Limits& limits) {
    return Guard(error, [&] {
        if (tokenCount > limits.maxPromptTokens || tokenCount > limits.maxAttachmentBytes / 4) {
            return Fail(error, ErrorCode::ResourceLimit, "Token attachment count exceeds its limit");
        }
        if (bytes.size() != tokenCount * 4) return Fail(error, ErrorCode::InvalidArgument, "Token count does not match the exact i32le attachment size");
        std::vector<int32_t> decoded(static_cast<size_t>(tokenCount));
        for (size_t index = 0; index < decoded.size(); ++index) {
            const uint32_t value = read_u32_le(bytes.data() + index * 4);
            if (value > static_cast<uint32_t>((std::numeric_limits<int32_t>::max)())) return Fail(error, ErrorCode::InvalidArgument, "Token attachment contains a negative token ID");
            decoded[index] = static_cast<int32_t>(value);
        }
        tokens = std::move(decoded);
        return true;
    });
}

namespace {

bool GenerationJson(const GenerationOptions& generation, std::string& output, ValidationError& error) {
    Json json;
    json.U64("maxNewTokens", generation.maxNewTokens);
    json.Bool("doSample", generation.doSample);
    json.Number("temperature", generation.temperature);
    json.U64("topK", generation.topK);
    json.Number("topP", generation.topP);
    json.Bool("useFixedSeed", generation.useFixedSeed);
    json.U64("samplingSeed", generation.samplingSeed);
    json.Number("repetitionPenalty", generation.repetitionPenalty);
    json.Number("presencePenalty", generation.presencePenalty);
    json.Number("frequencyPenalty", generation.frequencyPenalty);
    return json.Finish(output, error);
}
bool ReadGeneration(Object object, GenerationOptions& generation, ValidationError& error, const Limits& limits,
                    bool prefillOnly) {
    constexpr auto code = ErrorCode::InvalidArgument;
    if (!Fields(object, {"maxNewTokens", "doSample", "temperature", "topK", "topP", "useFixedSeed",
                         "samplingSeed", "repetitionPenalty", "presencePenalty", "frequencyPenalty"}, {}, error, code) ||
        !Unsigned(object, "maxNewTokens", generation.maxNewTokens, error, code) ||
        !Boolean(object, "doSample", generation.doSample, error, code) ||
        !Number(object, "temperature", generation.temperature, error, code) ||
        !Unsigned(object, "topK", generation.topK, error, code) ||
        !Number(object, "topP", generation.topP, error, code) ||
        !Boolean(object, "useFixedSeed", generation.useFixedSeed, error, code) ||
        !Unsigned(object, "samplingSeed", generation.samplingSeed, error, code) ||
        !Number(object, "repetitionPenalty", generation.repetitionPenalty, error, code) ||
        !Number(object, "presencePenalty", generation.presencePenalty, error, code) ||
        !Number(object, "frequencyPenalty", generation.frequencyPenalty, error, code)) return false;
    if ((prefillOnly ? generation.maxNewTokens != 0 : generation.maxNewTokens == 0) || generation.temperature < 0 ||
        (generation.doSample && static_cast<float>(generation.temperature) <= 0) ||
        generation.topK > static_cast<uint32_t>((std::numeric_limits<int32_t>::max)()) ||
        generation.topP <= 0 || generation.topP > 1 || static_cast<float>(generation.topP) <= 0 ||
        generation.repetitionPenalty <= 0 || static_cast<float>(generation.repetitionPenalty) <= 0) {
        return Fail(error, code, "Invalid generation sampling parameters");
    }
    if (generation.maxNewTokens > limits.maxGeneratedTokens) return Fail(error, ErrorCode::ResourceLimit, "Generation token limit exceeded");
    return true;
}
bool ThinkingBudgetJson(const ThinkingBudgetOptions& budget, const std::vector<int32_t>& stops,
                        std::string& output, ValidationError& error) {
    if (const char* message = aila::generation::ValidateThinkingBudget(budget, uint32_t{1} << 31, stops))
        return Fail(error, ErrorCode::InvalidArgument, message);
    Json json;
    json.U64("maxTokens", budget.maxTokens);
    json.U64("openTokenId", static_cast<uint32_t>(budget.openTokenId));
    json.U64("closeTokenId", static_cast<uint32_t>(budget.closeTokenId));
    json.Bool("promptOpensThink", budget.promptOpensThink);
    json.Integers("closeTokenIds", budget.closeTokenIds);
    if (budget.closeOnce) json.Bool("closeOnce", true);
    return json.Finish(output, error);
}

bool ReadOptionalThinkingBudget(Object object, std::optional<ThinkingBudgetOptions>& budget,
                                const std::vector<int32_t>& stops, ValidationError& error) {
    if (object.at_key("thinkingBudget").error() == simdjson::NO_SUCH_FIELD) return true;
    constexpr auto code = ErrorCode::InvalidArgument;
    Object child;
    ThinkingBudgetOptions value;
    uint32_t open = 0, close = 0;
    if (!ChildObject(object, "thinkingBudget", child, error, code) ||
        !Fields(child, {"maxTokens", "openTokenId", "closeTokenId", "promptOpensThink", "closeTokenIds"}, {"closeOnce"}, error, code) ||
        !Unsigned(child, "maxTokens", value.maxTokens, error, code) ||
        !Unsigned(child, "openTokenId", open, error, code) ||
        !Unsigned(child, "closeTokenId", close, error, code) ||
        !Boolean(child, "promptOpensThink", value.promptOpensThink, error, code)) return false;
    if (child.at_key("closeOnce").error() != simdjson::NO_SUCH_FIELD &&
        (!Boolean(child, "closeOnce", value.closeOnce, error, code) || !value.closeOnce))
        return Fail(error, code, "closeOnce must be true when present");
    if (open >= (uint32_t{1} << 31) || close >= (uint32_t{1} << 31))
        return Fail(error, code, "Thinking marker ID exceeds int32 range");
    value.openTokenId = static_cast<int32_t>(open);
    value.closeTokenId = static_cast<int32_t>(close);
    simdjson::dom::array array;
    if (child.at_key("closeTokenIds").get_array().get(array) != simdjson::SUCCESS ||
        array.size() > aila::generation::kMaxThinkingCloseTokens)
        return Fail(error, code, "closeTokenIds must be an array of at most 16 token IDs");
    for (const auto element : array) {
        uint64_t token = 0;
        if (element.get_uint64().get(token) != simdjson::SUCCESS || token >= (uint64_t{1} << 31))
            return Fail(error, code, "closeTokenIds must contain nonnegative int32 token IDs");
        value.closeTokenIds.push_back(static_cast<int32_t>(token));
    }
    if (const char* message = aila::generation::ValidateThinkingBudget(value, uint32_t{1} << 31, stops))
        return Fail(error, code, message);
    budget = std::move(value);
    return true;
}

bool ReadStopIds(Object object, std::vector<int32_t>& values, ValidationError& error, const Limits& limits) {
    simdjson::dom::array array;
    if (object.at_key("stopTokenIds").get_array().get(array) != simdjson::SUCCESS) {
        return Fail(error, ErrorCode::InvalidArgument, "stopTokenIds must be an array");
    }
    if (array.size() > limits.maxStopTokenIds) return Fail(error, ErrorCode::ResourceLimit, "Too many stop token IDs");
    values.reserve(array.size());
    for (const auto element : array) {
        uint64_t value = 0;
        if (element.get_uint64().get(value) != simdjson::SUCCESS || value > static_cast<uint64_t>((std::numeric_limits<int32_t>::max)())) {
            return Fail(error, ErrorCode::InvalidArgument, "stopTokenIds must contain nonnegative int32 token IDs");
        }
        values.push_back(static_cast<int32_t>(value));
    }
    return true;
}
bool NoAttachment(const Frame& frame, ValidationError& error, ErrorCode code) {
    return frame.attachment.empty() || Fail(error, code, "This method does not accept an attachment");
}

bool ReadDecodeBatchSize(Object object, uint32_t& size, const Limits& limits,
                        const std::vector<std::string>& capabilities,
                        ValidationError& error, ErrorCode code) {
    if (object.at_key("max_decode_batch_size").error() != simdjson::NO_SUCH_FIELD &&
        !Unsigned(object, "max_decode_batch_size", size, error, code)) return false;
    if (!validate_decode_batch_size(size, limits, error)) { error.code = code; return false; }
    if (size > 1 && std::find(capabilities.begin(), capabilities.end(), "continuous_batching_v1") == capabilities.end())
        return Fail(error, code == ErrorCode::InvalidArgument ? ErrorCode::UnsupportedCapability : code,
                    "Decode batching requires continuous_batching_v1");
    return true;
}

} // namespace

bool make_command(const Command& command, Frame& frame, ValidationError& error, const Limits& limits) {
    return Guard(error, [&] {
        if (!validate_limits(limits, error, extended_generation_limits())) return false;
        Frame candidate = BaseFrame("request", to_string(command.method), command.rpcId);
        Json json(limits.maxHeaderBytes);
        json.String("epoch", command.epoch);
        switch (command.method) {
        case Method::WorkerConfigure: {
            if (!Capabilities(command.requiredCapabilities, error, ErrorCode::InvalidArgument) ||
                !validate_decode_batch_size(command.maxDecodeBatchSize, command.limits, error)) return false;
            std::string configuration;
            if (!LimitsJson(command.limits, configuration, error)) return false;
            json.Strings("requiredCapabilities", command.requiredCapabilities);
            json.ObjectValue("limits", configuration);
            if (command.maxDecodeBatchSize > 1) json.U64("max_decode_batch_size", command.maxDecodeBatchSize);
            break;
        }
        case Method::WorkerQuery:
        case Method::WorkerMemoryQuery:
            break;
        case Method::WorkerSetLogLevel:
            json.U64("level", command.logLevel);
            break;
        case Method::ModelLoad:
            json.String("backend", command.backend);
            json.String("modelDirectory", command.modelDirectory);
            json.String("kvType", command.kvType);
            json.U64("maxSequenceLength", command.maxSequenceLength);
            json.U64("maxSessions", command.maxSessions);
            json.U64("maxSessionCacheBytes", command.maxSessionCacheBytes);
            if (command.allocationLimits) {
                std::string value;
                if (!AllocationLimitsJson(*command.allocationLimits, value, error)) return false;
                json.ObjectValue("allocationLimits", value);
            }
            break;
        case Method::SessionCreate:
            json.U64("modelId", command.modelId);
            break;
        case Method::SessionQuery:
        case Method::SessionReset:
            json.U64("sessionId", command.sessionId);
            break;
        case Method::SessionTruncate:
            json.U64("sessionId", command.sessionId);
            json.U64("contextLength", command.contextLength);
            break;
        case Method::RequestSubmit: {
            if (command.stopTokenIds.size() > limits.maxStopTokenIds ||
                !encode_token_ids(command.promptIds, candidate.attachment, error, limits)) {
                if (error.message.empty()) Fail(error, ErrorCode::ResourceLimit, "Too many stop token IDs");
                return false;
            }
            std::string generation;
            if (!GenerationJson(command.generation, generation, error)) return false;
            json.U64("sessionId", command.sessionId);
            json.U64("promptTokenCount", command.promptIds.size());
            json.String("promptEncoding", "i32le");
            json.U64("prefillChunkSize", command.prefillChunkSize);
            json.Integers("stopTokenIds", command.stopTokenIds);
            json.ObjectValue("generation", generation);
            if (command.prefillOnly) json.Bool("prefillOnly", true);
            if (command.forceHostSampling) json.Bool("hostSampling", true);
            if (command.legacyPrefillLogits) json.Bool("legacyPrefillLogits", true);
            if (command.thinkingBudget) {
                std::string budget;
                if (!ThinkingBudgetJson(*command.thinkingBudget, command.stopTokenIds, budget, error)) return false;
                json.ObjectValue("thinkingBudget", budget);
            }
            break;
        }
        case Method::RequestCancel:
        case Method::RequestQuery:
            json.U64("inferenceId", command.inferenceId);
            break;
        case Method::RequestAck:
            json.U64("inferenceId", command.inferenceId);
            json.U64("consumedSequence", command.consumedSequence);
            break;
        case Method::HandleRelease:
            json.String("type", to_string(command.objectType));
            json.U64("id", command.objectId);
            break;
        case Method::WorkerShutdown:
            json.U64("graceMs", command.graceMs);
            break;
        default:
            return Fail(error, ErrorCode::InvalidArgument, "Unknown command method");
        }
        if (!json.Finish(candidate.header.payload_json, error)) return false;
        Command checked;
        if (!parse_command(candidate, checked, error, limits)) return false;
        frame = std::move(candidate);
        return true;
    });
}

bool parse_command(const Frame& frame, Command& command, ValidationError& error, const Limits& limits) {
    return Guard(error, [&] {
        constexpr auto code = ErrorCode::InvalidArgument;
        simdjson::dom::parser parser;
        Object object;
        if (!Payload(frame, Channel::Command, frame_bounds(Channel::Command, limits), parser, object, error)) return false;
        Command parsed;
        parsed.rpcId = frame.header.request_id;
        if (!parse_method(frame.header.method, parsed.method)) return Fail(error, code, "Unsupported command method");
        if (!Epoch(object, parsed.epoch, error, code)) return false;
        if (parsed.method != Method::RequestSubmit && !NoAttachment(frame, error, code)) return false;
        switch (parsed.method) {
        case Method::WorkerConfigure: {
            Object configuration;
            if (!Fields(object, {"epoch", "requiredCapabilities", "limits"}, {"max_decode_batch_size"}, error, code) ||
                !ReadCapabilities(object, "requiredCapabilities", parsed.requiredCapabilities, error, code) ||
                !ChildObject(object, "limits", configuration, error, code) ||
                !ReadLimits(configuration, parsed.limits, error, code) ||
                !validate_limits(parsed.limits, error, configure_limits_ceiling(limits, parsed.requiredCapabilities)) ||
                !ReadDecodeBatchSize(object, parsed.maxDecodeBatchSize, parsed.limits,
                                    parsed.requiredCapabilities, error, code)) return false;
            break;
        }
        case Method::WorkerQuery:
        case Method::WorkerMemoryQuery:
            if (!Fields(object, {"epoch"}, {}, error, code)) return false;
            break;
        case Method::WorkerSetLogLevel:
            if (!Fields(object, {"epoch", "level"}, {}, error, code) ||
                !Unsigned(object, "level", parsed.logLevel, error, code)) return false;
            if (parsed.logLevel > 3) return Fail(error, code, "Log level must be in 0..3");
            break;
        case Method::ModelLoad:
            if (!Fields(object, {"epoch", "backend", "modelDirectory", "kvType", "maxSequenceLength", "maxSessions", "maxSessionCacheBytes"}, {"allocationLimits"}, error, code) ||
                !Text(object, "backend", parsed.backend, 64, error, code) ||
                !Text(object, "modelDirectory", parsed.modelDirectory, MaxModelPathBytes, error, code) ||
                !Text(object, "kvType", parsed.kvType, 16, error, code) ||
                !Unsigned(object, "maxSequenceLength", parsed.maxSequenceLength, error, code) ||
                !Unsigned(object, "maxSessions", parsed.maxSessions, error, code) ||
                !Unsigned(object, "maxSessionCacheBytes", parsed.maxSessionCacheBytes, error, code) ||
                !ReadOptionalAllocationLimits(object, parsed.allocationLimits, error, code)) return false;
            if (model_capability(parsed.backend).empty())
                return Fail(error, ErrorCode::UnsupportedModel, "The v2 token API supports qwen3_nf4, qwen35_nf4 and qwen35_bf16");
            if (model_capability(parsed.backend, parsed.kvType).empty() || parsed.modelDirectory.empty() || parsed.maxSequenceLength == 0 ||
                parsed.maxSequenceLength > static_cast<uint32_t>((std::numeric_limits<int32_t>::max)()) ||
                parsed.maxSessions == 0 || parsed.maxSessions > limits.maxSessions ||
                (limits.maxSessionCacheBytes != 0 && (parsed.maxSessionCacheBytes == 0 || parsed.maxSessionCacheBytes > limits.maxSessionCacheBytes))) {
                return Fail(error, code, "Invalid model options or options exceed negotiated limits");
            }
            break;
        case Method::SessionCreate:
            if (!Fields(object, {"epoch", "modelId"}, {}, error, code) || !Id(object, "modelId", parsed.modelId, error, code)) return false;
            break;
        case Method::SessionQuery:
        case Method::SessionReset:
            if (!Fields(object, {"epoch", "sessionId"}, {}, error, code) || !Id(object, "sessionId", parsed.sessionId, error, code)) return false;
            break;
        case Method::SessionTruncate:
            if (!Fields(object, {"epoch", "sessionId", "contextLength"}, {}, error, code) ||
                !Id(object, "sessionId", parsed.sessionId, error, code) ||
                !Unsigned(object, "contextLength", parsed.contextLength, error, code)) return false;
            if (parsed.contextLength > static_cast<uint32_t>((std::numeric_limits<int32_t>::max)()))
                return Fail(error, code, "Truncate length exceeds int32 range");
            break;
        case Method::RequestSubmit: {
            uint64_t tokenCount = 0;
            std::string encoding;
            Object generation;
            if (!Fields(object, {"epoch", "sessionId", "promptTokenCount", "promptEncoding", "prefillChunkSize", "stopTokenIds", "generation"}, {"thinkingBudget", "prefillOnly", "hostSampling", "legacyPrefillLogits"}, error, code) ||
                !Id(object, "sessionId", parsed.sessionId, error, code) ||
                !Unsigned(object, "promptTokenCount", tokenCount, error, code) ||
                !Text(object, "promptEncoding", encoding, 16, error, code) ||
                !Unsigned(object, "prefillChunkSize", parsed.prefillChunkSize, error, code) ||
                !ReadStopIds(object, parsed.stopTokenIds, error, limits) ||
                !ChildObject(object, "generation", generation, error, code)) return false;
            if (object.at_key("prefillOnly").error() != simdjson::NO_SUCH_FIELD &&
                (!Boolean(object, "prefillOnly", parsed.prefillOnly, error, code) || !parsed.prefillOnly))
                return Fail(error, code, "prefillOnly must be true when present");
            if (object.at_key("hostSampling").error() != simdjson::NO_SUCH_FIELD &&
                (!Boolean(object, "hostSampling", parsed.forceHostSampling, error, code) || !parsed.forceHostSampling))
                return Fail(error, code, "hostSampling must be true when present");
            if (object.at_key("legacyPrefillLogits").error() != simdjson::NO_SUCH_FIELD &&
                (!Boolean(object, "legacyPrefillLogits", parsed.legacyPrefillLogits, error, code) || !parsed.legacyPrefillLogits))
                return Fail(error, code, "legacyPrefillLogits must be true when present");
            if (!ReadGeneration(generation, parsed.generation, error, limits, parsed.prefillOnly) ||
                !ReadOptionalThinkingBudget(object, parsed.thinkingBudget, parsed.stopTokenIds, error)) return false;
            if (parsed.prefillOnly && (!parsed.stopTokenIds.empty() || parsed.thinkingBudget))
                return Fail(error, code, "Prefill-only requests cannot have stop tokens or a thinking budget");
            if (encoding != "i32le" || tokenCount == 0 || parsed.prefillChunkSize == 0 || parsed.prefillChunkSize > limits.maxPrefillChunkSize) {
                return Fail(error, code, "Invalid token encoding, empty prompt, or prefill chunk size");
            }
            if (!decode_token_ids(frame.attachment, tokenCount, parsed.promptIds, error, limits)) return false;
            if (tokenCount > limits.maxOutstandingTokens || parsed.generation.maxNewTokens > limits.maxOutstandingTokens - tokenCount) {
                return Fail(error, ErrorCode::ResourceLimit, "Request exceeds the total token reservation limit");
            }
            parsed.inferenceId = parsed.rpcId;
            break;
        }
        case Method::RequestCancel:
        case Method::RequestQuery:
            if (!Fields(object, {"epoch", "inferenceId"}, {}, error, code) || !Id(object, "inferenceId", parsed.inferenceId, error, code)) return false;
            break;
        case Method::RequestAck:
            if (!Fields(object, {"epoch", "inferenceId", "consumedSequence"}, {}, error, code) ||
                !Id(object, "inferenceId", parsed.inferenceId, error, code) ||
                !Unsigned(object, "consumedSequence", parsed.consumedSequence, error, code)) return false;
            if (parsed.consumedSequence > static_cast<uint64_t>(limits.maxGeneratedTokens) + 1) {
                return Fail(error, code, "ACK sequence exceeds the stream event limit");
            }
            break;
        case Method::HandleRelease:
            if (!Fields(object, {"epoch", "type", "id"}, {}, error, code) ||
                !Enum(object, "type", parsed.objectType, parse_object_type, error, code) ||
                !Id(object, "id", parsed.objectId, error, code)) return false;
            break;
        case Method::WorkerShutdown:
            if (!Fields(object, {"epoch", "graceMs"}, {}, error, code) || !Unsigned(object, "graceMs", parsed.graceMs, error, code)) return false;
            if (parsed.graceMs == 0 || parsed.graceMs > limits.maxShutdownGraceMs) {
                return Fail(error, code, "Shutdown grace must be positive and within the negotiated limit");
            }
            break;
        }
        command = std::move(parsed);
        return true;
    });
}

namespace {

void WriteIdentity(Json& json, const Identity& identity) {
    json.U64("inferenceId", identity.inferenceId);
    json.U64("modelId", identity.modelId);
    json.U64("sessionId", identity.sessionId);
}
bool ReadIdentity(Object object, Identity& identity, ValidationError& error) {
    constexpr auto code = ErrorCode::ProtocolError;
    return Id(object, "inferenceId", identity.inferenceId, error, code) &&
           Id(object, "modelId", identity.modelId, error, code) &&
           Id(object, "sessionId", identity.sessionId, error, code);
}
void WriteError(Json& json, const std::optional<ErrorCode>& code, const std::string& message) {
    if (code) json.String("errorCode", to_string(*code));
    else json.Null("errorCode");
    json.String("message", message);
}
bool ReadError(Object object, std::optional<ErrorCode>& code, std::string& message,
               ValidationError& error, ErrorCode validationCode = ErrorCode::ProtocolError) {
    Element element;
    if (object.at_key("errorCode").get(element) != simdjson::SUCCESS) return Fail(error, validationCode, "Missing errorCode");
    if (element.is_null()) code.reset();
    else {
        ErrorCode parsed;
        std::string_view value;
        if (element.get_string().get(value) != simdjson::SUCCESS || !parse_error_code(value, parsed)) {
            return Fail(error, validationCode, "Unknown errorCode");
        }
        code = parsed;
    }
    if (!Text(object, "message", message, kMaxErrorBytes, error, validationCode, true)) return false;
    return code || message.empty() || Fail(error, validationCode, "A null errorCode requires an empty error message");
}
void WriteSummary(Json& json, const Summary& summary, bool nullableCount = false) {
    json.String("executionStatus", to_string(summary.status));
    json.String("endReason", to_string(summary.endReason));
    json.U64("promptTokensProcessed", summary.promptTokensProcessed);
    json.U64("generatedTokens", summary.generatedTokens);
    if (nullableCount && summary.eventCount == 0) json.Null("eventCount");
    else json.U64("eventCount", summary.eventCount);
    WriteError(json, summary.errorCode, summary.message);
}
bool ReadSummary(Object object, Summary& summary, ValidationError& error, const Limits& limits, bool nullableCount = false) {
    constexpr auto code = ErrorCode::ProtocolError;
    if (!Enum(object, "executionStatus", summary.status, parse_request_status, error, code) ||
        !Enum(object, "endReason", summary.endReason, parse_end_reason, error, code) ||
        !Unsigned(object, "promptTokensProcessed", summary.promptTokensProcessed, error, code) ||
        !Unsigned(object, "generatedTokens", summary.generatedTokens, error, code) ||
        !ReadError(object, summary.errorCode, summary.message, error)) return false;
    Element count;
    if (object.at_key("eventCount").get(count) != simdjson::SUCCESS) return Fail(error, code, "Missing eventCount");
    if (nullableCount && count.is_null()) summary.eventCount = 0;
    else if (count.get_uint64().get(summary.eventCount) != simdjson::SUCCESS) return Fail(error, code, "eventCount must be an unsigned integer");
    if (nullableCount && is_terminal(summary.status) == count.is_null()) {
        return Fail(error, code, "Only nonterminal query snapshots may have a null eventCount");
    }
    return validate_summary(summary, error, limits);
}
bool SpecificFrameSize(const Frame& frame, uint64_t maximum, ValidationError& error) {
    size_t headerBytes = 0;
    if (!aila::ipc::encoded_header_size(frame.header, headerBytes) ||
        kFramePrefixBytes + static_cast<uint64_t>(headerBytes) + frame.attachment.size() > maximum) {
        return Fail(error, ErrorCode::ProtocolError, "Frame exceeds its event/result byte limit after JSON escaping");
    }
    return true;
}

} // namespace

bool validate_summary(const Summary& summary, ValidationError& error, const Limits& limits) {
    return Guard(error, [&] {
        constexpr auto code = ErrorCode::ProtocolError;
        if (to_string(summary.status).empty() || to_string(summary.endReason).empty() ||
            (summary.errorCode && to_string(*summary.errorCode).empty()) ||
            !SafeText(summary.message, kMaxErrorBytes, true) ||
            summary.promptTokensProcessed > limits.maxPromptTokens || summary.generatedTokens > limits.maxGeneratedTokens) {
            return Fail(error, code, "Invalid or out-of-bounds request summary");
        }
        if (!is_terminal(summary.status)) {
            if (summary.endReason != EndReason::None || summary.eventCount != 0 || summary.errorCode || !summary.message.empty()) {
                return Fail(error, code, "Nonterminal summaries cannot contain completion or error fields");
            }
            return true;
        }
        if (summary.eventCount != summary.generatedTokens + 1) return Fail(error, code, "Terminal eventCount must equal generatedTokens plus one");
        if (summary.status == RequestStatus::Completed) {
            if ((summary.endReason != EndReason::Length && summary.endReason != EndReason::StopToken) ||
                summary.errorCode || !summary.message.empty()) {
                return Fail(error, code, "Completed summary has an incompatible reason or error");
            }
        } else if (summary.status == RequestStatus::Cancelled) {
            const bool cancelled = summary.endReason == EndReason::Cancelled && !summary.errorCode && summary.message.empty();
            const bool backpressure = summary.endReason == EndReason::OutputBackpressure && summary.errorCode == ErrorCode::ResourceLimit;
            const bool stopped = summary.endReason == EndReason::SchedulerStopped && summary.errorCode == ErrorCode::SchedulerStopped;
            if (!cancelled && !backpressure && !stopped) return Fail(error, code, "Cancelled summary has an incompatible reason or error");
        } else {
            const bool domainFault = summary.endReason == EndReason::RuntimeFault && summary.errorCode == ErrorCode::RuntimeFault;
            const bool failed = summary.endReason == EndReason::Failed && summary.errorCode && summary.errorCode != ErrorCode::RuntimeFault;
            if (!domainFault && !failed) return Fail(error, code, "Failed summary has an incompatible reason or error");
        }
        return true;
    });
}

bool make_event(const Event& event, Frame& frame, ValidationError& error, const Limits& limits) {
    return Guard(error, [&] {
        const bool requestEvent = event.kind == EventKind::Token || event.kind == EventKind::Terminal;
        const std::string_view method = requestEvent ? "request.event" : event.kind == EventKind::DomainFault ? "domain.fault" : "log";
        Frame candidate = BaseFrame("event", method, requestEvent ? event.identity.inferenceId : 0);
        Json json(limits.maxHeaderBytes);
        json.String("epoch", event.epoch);
        json.U64("eventOrdinal", event.eventOrdinal);
        if (requestEvent) {
            WriteIdentity(json, event.identity);
            json.U64("sequence", event.sequence);
            if (event.kind == EventKind::Token) {
                if (event.tokenId < 0) return Fail(error, ErrorCode::InvalidArgument, "Token ID cannot be negative");
                json.String("event", "token");
                json.U64("tokenId", static_cast<uint32_t>(event.tokenId));
                json.String("executionStatus", to_string(event.summary.status));
                json.U64("promptTokensProcessed", event.summary.promptTokensProcessed);
                json.U64("generatedTokens", event.summary.generatedTokens);
            } else {
                json.String("event", "terminal");
                WriteSummary(json, event.summary);
            }
        } else if (event.kind == EventKind::DomainFault) {
            WriteError(json, event.errorCode, event.message);
        } else if (event.kind == EventKind::Log) {
            if (event.logLevel < 0) return Fail(error, ErrorCode::InvalidArgument, "Log level cannot be negative");
            json.U64("level", static_cast<uint32_t>(event.logLevel));
            json.String("message", event.message);
        } else return Fail(error, ErrorCode::InvalidArgument, "Unknown event kind");
        if (!json.Finish(candidate.header.payload_json, error)) return false;
        Event checked;
        if (!parse_event(candidate, checked, error, limits)) return false;
        frame = std::move(candidate);
        return true;
    });
}

bool parse_event(const Frame& frame, Event& event, ValidationError& error, const Limits& limits) {
    return Guard(error, [&] {
        constexpr auto code = ErrorCode::ProtocolError;
        simdjson::dom::parser parser;
        Object object;
        if (!Payload(frame, Channel::Event, frame_bounds(Channel::Event, limits), parser, object, error) ||
            !NoAttachment(frame, error, code)) return false;
        Event parsed;
        if (!Epoch(object, parsed.epoch, error, code) ||
            !Unsigned(object, "eventOrdinal", parsed.eventOrdinal, error, code) || !PositiveId(parsed.eventOrdinal)) {
            if (error.message.empty()) Fail(error, code, "Event ordinal must be positive");
            return false;
        }
        if (frame.header.method == "log") {
            uint32_t level = 0;
            parsed.kind = EventKind::Log;
            if (frame.header.request_id != 0 ||
                !Fields(object, {"epoch", "eventOrdinal", "level", "message"}, {}, error, code) ||
                !Unsigned(object, "level", level, error, code) || level > 3 ||
                !Text(object, "message", parsed.message, kMaxErrorBytes, error, code, true)) {
                if (error.message.empty()) Fail(error, code, "Invalid log event identity or level");
                return false;
            }
            parsed.logLevel = static_cast<int32_t>(level);
        } else if (frame.header.method == "domain.fault") {
            parsed.kind = EventKind::DomainFault;
            if (frame.header.request_id != 0 ||
                !Fields(object, {"epoch", "eventOrdinal", "errorCode", "message"}, {}, error, code) ||
                !ReadError(object, parsed.errorCode, parsed.message, error) || parsed.errorCode != ErrorCode::RuntimeFault) {
                if (error.message.empty()) Fail(error, code, "Domain fault requires request ID zero and RuntimeFault");
                return false;
            }
        } else if (frame.header.method == "request.event") {
            std::string kind;
            if (!ReadIdentity(object, parsed.identity, error) || frame.header.request_id != parsed.identity.inferenceId ||
                !Unsigned(object, "sequence", parsed.sequence, error, code) ||
                !Text(object, "event", kind, 16, error, code)) {
                if (error.message.empty()) Fail(error, code, "Request event identity does not match its header");
                return false;
            }
            if (kind == "token") {
                uint32_t tokenId = 0;
                parsed.kind = EventKind::Token;
                if (!Fields(object, {"epoch", "eventOrdinal", "inferenceId", "modelId", "sessionId", "sequence", "event", "tokenId",
                                     "executionStatus", "promptTokensProcessed", "generatedTokens"}, {}, error, code) ||
                    !Unsigned(object, "tokenId", tokenId, error, code) ||
                    !Enum(object, "executionStatus", parsed.summary.status, parse_request_status, error, code) ||
                    !Unsigned(object, "promptTokensProcessed", parsed.summary.promptTokensProcessed, error, code) ||
                    !Unsigned(object, "generatedTokens", parsed.summary.generatedTokens, error, code)) return false;
                if (tokenId > static_cast<uint32_t>((std::numeric_limits<int32_t>::max)()) ||
                    parsed.sequence == 0 || parsed.sequence > limits.maxGeneratedTokens ||
                    parsed.summary.generatedTokens != parsed.sequence ||
                    parsed.summary.promptTokensProcessed == 0 || parsed.summary.promptTokensProcessed > limits.maxPromptTokens ||
                    (parsed.summary.status != RequestStatus::Decode && parsed.summary.status != RequestStatus::Finalizing)) {
                    return Fail(error, code, "Invalid token event sequence, progress, or status");
                }
                parsed.tokenId = static_cast<int32_t>(tokenId);
                if (!SpecificFrameSize(frame, limits.maxTokenFrameBytes, error)) return false;
            } else if (kind == "terminal") {
                parsed.kind = EventKind::Terminal;
                if (!Fields(object, {"epoch", "eventOrdinal", "inferenceId", "modelId", "sessionId", "sequence", "event", "executionStatus", "endReason",
                                     "promptTokensProcessed", "generatedTokens", "eventCount", "errorCode", "message"}, {}, error, code) ||
                    !ReadSummary(object, parsed.summary, error, limits) || !is_terminal(parsed.summary.status) ||
                    parsed.sequence != parsed.summary.eventCount) {
                    if (error.message.empty()) Fail(error, code, "Terminal event requires a terminal summary and matching sequence/count");
                    return false;
                }
            } else return Fail(error, code, "Unknown request event kind");
        } else return Fail(error, code, "Unknown event method");
        event = std::move(parsed);
        return true;
    });
}

namespace {

#define AILA_V2_WORKER_COUNTS(X) \
    X(outstandingRequests) X(queuedRequests) X(activeRequests) X(reservedTokens) X(reservedSessionCacheBytes) \
    X(ipcOutstandingRequests) X(retainedRequestHandles) X(pendingCommands) X(ingressBytes) X(transportBytes) X(droppedLogs)

void WriteWorker(Json& json, const WorkerInfo& worker) {
    json.String("domainStatus", to_string(worker.domainStatus));
    json.Bool("accepting", worker.accepting);
    json.Bool("stopping", worker.stopping);
    json.Bool("faulted", worker.faulted);
#define AILA_WRITE_WORKER_COUNT(field) json.U64(#field, worker.field);
    AILA_V2_WORKER_COUNTS(AILA_WRITE_WORKER_COUNT)
#undef AILA_WRITE_WORKER_COUNT
}
bool ReadWorker(Object object, WorkerInfo& worker, ValidationError& error, const Limits& limits) {
    constexpr auto code = ErrorCode::ProtocolError;
#define AILA_WORKER_COUNT_NAME(field) #field,
    if (!Fields(object, {"epoch", "domainStatus", "accepting", "stopping", "faulted", AILA_V2_WORKER_COUNTS(AILA_WORKER_COUNT_NAME)}, {}, error, code)) return false;
#undef AILA_WORKER_COUNT_NAME
    if (!Enum(object, "domainStatus", worker.domainStatus, parse_domain_status, error, code) ||
        !Boolean(object, "accepting", worker.accepting, error, code) ||
        !Boolean(object, "stopping", worker.stopping, error, code) ||
        !Boolean(object, "faulted", worker.faulted, error, code)) return false;
#define AILA_READ_WORKER_COUNT(field) if (!Unsigned(object, #field, worker.field, error, code)) return false;
    AILA_V2_WORKER_COUNTS(AILA_READ_WORKER_COUNT)
#undef AILA_READ_WORKER_COUNT
    if (worker.outstandingRequests > limits.maxOutstandingRequests || worker.activeRequests > limits.maxActiveRequests ||
        worker.queuedRequests > worker.outstandingRequests || worker.activeRequests > worker.outstandingRequests - worker.queuedRequests ||
        worker.reservedTokens > limits.maxOutstandingTokens || worker.ipcOutstandingRequests > limits.maxOutstandingRequests ||
        worker.retainedRequestHandles > limits.maxRetainedRequestHandles || worker.pendingCommands > limits.maxPendingCommands ||
        worker.ingressBytes > limits.maxIngressBytes || worker.transportBytes > limits.maxTransportBytes ||
        (limits.maxSessionCacheBytes != 0 && worker.reservedSessionCacheBytes > limits.maxSessionCacheBytes) ||
        (worker.accepting && (worker.stopping || worker.faulted || worker.domainStatus != DomainStatus::Ready))) {
        return Fail(error, code, "Worker snapshot exceeds negotiated bounds or has inconsistent state");
    }
    return true;
}
bool ReadContextLength(Object object, SessionInfo& session, ValidationError& error) {
    Element element;
    if (object.at_key("contextLength").get(element) != simdjson::SUCCESS) return Fail(error, ErrorCode::ProtocolError, "Missing contextLength");
    if (element.is_null()) session.contextLength.reset();
    else {
        uint64_t length = 0;
        if (element.get_uint64().get(length) != simdjson::SUCCESS || length > session.capacity) {
            return Fail(error, ErrorCode::ProtocolError, "Session contextLength exceeds its capacity");
        }
        session.contextLength = static_cast<uint32_t>(length);
    }
    if (session.busy == session.contextLength.has_value()) {
        return Fail(error, ErrorCode::ProtocolError, "Only a busy session may have an unknown contextLength");
    }
    return true;
}
bool ValidModelInfo(const ModelInfo& model, ValidationError& error) {
    if (!PositiveId(model.modelId) || model.vocabSize == 0 || model.maxSequenceLength == 0 ||
        model.vocabSize > static_cast<uint32_t>((std::numeric_limits<int32_t>::max)()) ||
        model.maxSequenceLength > static_cast<uint32_t>((std::numeric_limits<int32_t>::max)()) || model.sessionReservationBytes == 0) {
        return Fail(error, ErrorCode::ProtocolError, "Invalid model metadata");
    }
    if (model.memory && std::find(model.capabilities.begin(), model.capabilities.end(), "allocation_limits_v1") == model.capabilities.end())
        return Fail(error, ErrorCode::ProtocolError, "Memory metadata requires allocation_limits_v1");
    return true;
}

} // namespace

bool make_response(const Response& response, Frame& frame, ValidationError& error, const Limits& limits) {
    return Guard(error, [&] {
        if (response.kind != ResponseKind::Result && response.kind != ResponseKind::Error && response.kind != ResponseKind::Accepted) {
            return Fail(error, ErrorCode::InvalidArgument, "Unknown response kind");
        }
        const std::string_view kind = response.kind == ResponseKind::Accepted ? "accepted" : response.kind == ResponseKind::Error ? "error" : "result";
        Frame candidate = BaseFrame(kind, to_string(response.method), response.rpcId);
        Json json(limits.maxHeaderBytes);
        json.String("epoch", response.epoch);
        if (response.method == Method::RequestSubmit) {
            if (response.phase == SubmissionPhase::Accepted && response.kind == ResponseKind::Accepted) {
                json.String("phase", "accepted");
                WriteIdentity(json, response.identity);
                json.U64("tokenWindow", response.tokenWindow);
            } else if (response.phase == SubmissionPhase::Rejected && response.kind == ResponseKind::Error) {
                json.String("phase", "rejected");
                json.U64("eventCount", 0);
                WriteError(json, response.errorCode, response.message);
            } else if (response.phase == SubmissionPhase::Completed && response.kind != ResponseKind::Accepted) {
                json.String("phase", "completed");
                WriteIdentity(json, response.identity);
                WriteSummary(json, response.summary);
            } else return Fail(error, ErrorCode::InvalidArgument, "Submit response kind/phase is invalid");
        } else if (response.phase != SubmissionPhase::None || response.kind == ResponseKind::Accepted) {
            return Fail(error, ErrorCode::InvalidArgument, "Only submit has accepted/completed phases");
        } else if (response.kind == ResponseKind::Error) {
            WriteError(json, response.errorCode, response.message);
        } else {
            switch (response.method) {
            case Method::WorkerConfigure: {
                std::string configuration;
                if (!Capabilities(response.capabilities, error, ErrorCode::InvalidArgument) ||
                    !validate_decode_batch_size(response.maxDecodeBatchSize, response.limits, error) ||
                    !LimitsJson(response.limits, configuration, error)) return false;
                json.String("domainStatus", to_string(response.worker.domainStatus));
                json.Strings("capabilities", response.capabilities);
                json.ObjectValue("limits", configuration);
                if (response.maxDecodeBatchSize > 1) json.U64("max_decode_batch_size", response.maxDecodeBatchSize);
                break;
            }
            case Method::WorkerQuery:
                WriteWorker(json, response.worker);
                break;
            case Method::WorkerMemoryQuery:
                if (!WriteRuntimeMemory(json, response.runtimeMemory, error)) return false;
                break;
            case Method::WorkerSetLogLevel:
                json.U64("level", response.logLevel);
                break;
            case Method::ModelLoad:
                if (!Capabilities(response.model.capabilities, error, ErrorCode::InvalidArgument)) return false;
                json.U64("modelId", response.model.modelId);
                json.U64("vocabSize", response.model.vocabSize);
                json.U64("maxSequenceLength", response.model.maxSequenceLength);
                json.U64("sessionReservationBytes", response.model.sessionReservationBytes);
                json.Strings("capabilities", response.model.capabilities);
                if (response.model.memory) {
                    std::string value;
                    if (!ModelMemoryJson(*response.model.memory, value, error)) return false;
                    json.ObjectValue("memory", value);
                }
                break;
            case Method::SessionCreate:
            case Method::SessionQuery:
            case Method::SessionReset:
            case Method::SessionTruncate:
                json.U64("sessionId", response.session.sessionId);
                json.U64("modelId", response.session.modelId);
                json.U64("capacity", response.session.capacity);
                json.U64("cacheBytes", response.session.cacheBytes);
                if (response.method != Method::SessionCreate) json.Bool("busy", response.session.busy);
                if (response.session.contextLength) json.U64("contextLength", *response.session.contextLength);
                else json.Null("contextLength");
                break;
            case Method::RequestCancel:
                json.Bool("cancelRequested", response.cancelRequested);
                json.String("executionStatus", to_string(response.summary.status));
                break;
            case Method::RequestQuery:
                WriteSummary(json, response.request.summary, true);
                json.U64("bufferedTokens", response.request.bufferedTokens);
                json.U64("lastSentSequence", response.request.lastSentSequence);
                json.U64("lastAcknowledgedSequence", response.request.lastAcknowledgedSequence);
                json.Bool("completionReady", response.request.completionReady);
                break;
            case Method::RequestAck:
                json.U64("acknowledgedSequence", response.acknowledgedSequence);
                break;
            case Method::HandleRelease:
                json.Bool("releaseAccepted", response.releaseAccepted);
                break;
            case Method::WorkerShutdown:
                json.String("domainStatus", to_string(response.worker.domainStatus));
                json.U64("lastEventOrdinal", response.lastEventOrdinal);
                break;
            default:
                return Fail(error, ErrorCode::InvalidArgument, "Unknown response method");
            }
        }
        if (!json.Finish(candidate.header.payload_json, error)) return false;
        Response checked;
        if (!parse_response(candidate, checked, error, limits)) return false;
        frame = std::move(candidate);
        return true;
    });
}

bool parse_response(const Frame& frame, Response& response, ValidationError& error, const Limits& limits) {
    return Guard(error, [&] {
        constexpr auto code = ErrorCode::ProtocolError;
        simdjson::dom::parser parser;
        Object object;
        if (!Payload(frame, Channel::Response, frame_bounds(Channel::Response, limits), parser, object, error) ||
            !NoAttachment(frame, error, code)) return false;
        Response parsed;
        parsed.rpcId = frame.header.request_id;
        if (!parse_method(frame.header.method, parsed.method) || !Epoch(object, parsed.epoch, error, code)) {
            if (error.message.empty()) Fail(error, code, "Unknown response method");
            return false;
        }
        if (frame.header.kind == "result") parsed.kind = ResponseKind::Result;
        else if (frame.header.kind == "error") parsed.kind = ResponseKind::Error;
        else if (frame.header.kind == "accepted") parsed.kind = ResponseKind::Accepted;
        else return Fail(error, code, "Unexpected response kind");

        if (parsed.method == Method::RequestSubmit) {
            std::string phase;
            if (!Text(object, "phase", phase, 16, error, code)) return false;
            if (phase == "accepted" && parsed.kind == ResponseKind::Accepted) {
                parsed.phase = SubmissionPhase::Accepted;
                if (!Fields(object, {"epoch", "phase", "inferenceId", "modelId", "sessionId", "tokenWindow"}, {}, error, code) ||
                    !ReadIdentity(object, parsed.identity, error) ||
                    !Unsigned(object, "tokenWindow", parsed.tokenWindow, error, code)) return false;
                if (parsed.tokenWindow == 0 || parsed.tokenWindow > limits.tokenWindow || parsed.identity.inferenceId != parsed.rpcId) {
                    return Fail(error, code, "Accepted identity or token window is invalid");
                }
            } else if (phase == "rejected" && parsed.kind == ResponseKind::Error) {
                parsed.phase = SubmissionPhase::Rejected;
                uint64_t count = 0;
                if (!Fields(object, {"epoch", "phase", "eventCount", "errorCode", "message"}, {}, error, code) ||
                    !Unsigned(object, "eventCount", count, error, code) || count != 0 ||
                    !ReadError(object, parsed.errorCode, parsed.message, error) || !parsed.errorCode) {
                    if (error.message.empty()) Fail(error, code, "Rejected submit requires an error and zero eventCount");
                    return false;
                }
                parsed.identity.inferenceId = parsed.rpcId;
            } else if (phase == "completed" && parsed.kind != ResponseKind::Accepted) {
                parsed.phase = SubmissionPhase::Completed;
                if (!Fields(object, {"epoch", "phase", "inferenceId", "modelId", "sessionId", "executionStatus", "endReason",
                                     "promptTokensProcessed", "generatedTokens", "eventCount", "errorCode", "message"}, {}, error, code) ||
                    !ReadIdentity(object, parsed.identity, error) ||
                    !ReadSummary(object, parsed.summary, error, limits)) return false;
                if (!is_terminal(parsed.summary.status) || parsed.identity.inferenceId != parsed.rpcId ||
                    (parsed.summary.status == RequestStatus::Failed) != (parsed.kind == ResponseKind::Error)) {
                    return Fail(error, code, "Completed submit kind, identity or status is inconsistent");
                }
                if (!SpecificFrameSize(frame, limits.maxFinalFrameBytes, error)) return false;
            } else return Fail(error, code, "Invalid submit response phase/kind combination");
        } else if (parsed.kind == ResponseKind::Accepted) {
            return Fail(error, code, "Only submit can be accepted asynchronously");
        } else if (parsed.kind == ResponseKind::Error) {
            if (!Fields(object, {"epoch", "errorCode", "message"}, {}, error, code) ||
                !ReadError(object, parsed.errorCode, parsed.message, error) || !parsed.errorCode) {
                if (error.message.empty()) Fail(error, code, "Error response requires a nonnull errorCode");
                return false;
            }
        } else {
            switch (parsed.method) {
            case Method::WorkerConfigure: {
                Object configuration;
                if (!Fields(object, {"epoch", "domainStatus", "capabilities", "limits"}, {"max_decode_batch_size"}, error, code) ||
                    !Enum(object, "domainStatus", parsed.worker.domainStatus, parse_domain_status, error, code) ||
                    !ReadCapabilities(object, "capabilities", parsed.capabilities, error, code) ||
                    !ChildObject(object, "limits", configuration, error, code) ||
                    !ReadLimits(configuration, parsed.limits, error, code) ||
                    !validate_limits(parsed.limits, error, limits) ||
                    !ReadDecodeBatchSize(object, parsed.maxDecodeBatchSize, parsed.limits,
                                        parsed.capabilities, error, code)) return false;
                if (parsed.worker.domainStatus != DomainStatus::Ready) return Fail(error, code, "Configured Worker must be ready");
                break;
            }
            case Method::WorkerQuery:
                if (!ReadWorker(object, parsed.worker, error, limits)) return false;
                break;
            case Method::WorkerMemoryQuery:
                if (!ReadRuntimeMemory(object, parsed.runtimeMemory, error, code)) return false;
                break;
            case Method::WorkerSetLogLevel:
                if (!Fields(object, {"epoch", "level"}, {}, error, code) ||
                    !Unsigned(object, "level", parsed.logLevel, error, code)) return false;
                if (parsed.logLevel > 3) return Fail(error, code, "Invalid applied log level");
                break;
            case Method::ModelLoad:
                if (!Fields(object, {"epoch", "modelId", "vocabSize", "maxSequenceLength", "sessionReservationBytes", "capabilities"}, {"memory"}, error, code) ||
                    !Id(object, "modelId", parsed.model.modelId, error, code) ||
                    !Unsigned(object, "vocabSize", parsed.model.vocabSize, error, code) ||
                    !Unsigned(object, "maxSequenceLength", parsed.model.maxSequenceLength, error, code) ||
                    !Unsigned(object, "sessionReservationBytes", parsed.model.sessionReservationBytes, error, code) ||
                    !ReadCapabilities(object, "capabilities", parsed.model.capabilities, error, code) ||
                    !ReadOptionalModelMemory(object, parsed.model.memory, error, code) ||
                    !ValidModelInfo(parsed.model, error)) return false;
                break;
            case Method::SessionCreate:
            case Method::SessionQuery:
            case Method::SessionReset:
            case Method::SessionTruncate: {
                if (parsed.method == Method::SessionCreate) {
                    if (!Fields(object, {"epoch", "sessionId", "modelId", "capacity", "cacheBytes", "contextLength"}, {}, error, code)) return false;
                } else if (!Fields(object, {"epoch", "sessionId", "modelId", "capacity", "cacheBytes", "busy", "contextLength"}, {}, error, code) ||
                           !Boolean(object, "busy", parsed.session.busy, error, code)) return false;
                if (!Id(object, "sessionId", parsed.session.sessionId, error, code) ||
                    !Id(object, "modelId", parsed.session.modelId, error, code) ||
                    !Unsigned(object, "capacity", parsed.session.capacity, error, code) ||
                    !Unsigned(object, "cacheBytes", parsed.session.cacheBytes, error, code)) return false;
                if (parsed.session.capacity == 0 || parsed.session.capacity > static_cast<uint32_t>((std::numeric_limits<int32_t>::max)()) ||
                    parsed.session.cacheBytes == 0 || !ReadContextLength(object, parsed.session, error)) {
                    if (error.message.empty()) Fail(error, code, "Invalid session metadata");
                    return false;
                }
                if ((parsed.method == Method::SessionCreate || parsed.method == Method::SessionReset) &&
                    parsed.session.contextLength != std::optional<uint32_t>{0}) {
                    return Fail(error, code, "New or reset sessions must have zero context length");
                }
                if ((parsed.method == Method::SessionReset || parsed.method == Method::SessionTruncate) &&
                    (parsed.session.busy || !parsed.session.contextLength))
                    return Fail(error, code, "A completed context operation requires an idle Session and actual context length");
                break;
            }
            case Method::RequestCancel:
                if (!Fields(object, {"epoch", "cancelRequested", "executionStatus"}, {}, error, code) ||
                    !Boolean(object, "cancelRequested", parsed.cancelRequested, error, code) ||
                    !Enum(object, "executionStatus", parsed.summary.status, parse_request_status, error, code)) return false;
                break;
            case Method::RequestQuery: {
                auto& query = parsed.request;
                if (!Fields(object, {"epoch", "executionStatus", "endReason", "promptTokensProcessed", "generatedTokens", "eventCount", "errorCode", "message",
                                     "bufferedTokens", "lastSentSequence", "lastAcknowledgedSequence", "completionReady"}, {}, error, code) ||
                    !ReadSummary(object, query.summary, error, limits, true) ||
                    !Unsigned(object, "bufferedTokens", query.bufferedTokens, error, code) ||
                    !Unsigned(object, "lastSentSequence", query.lastSentSequence, error, code) ||
                    !Unsigned(object, "lastAcknowledgedSequence", query.lastAcknowledgedSequence, error, code) ||
                    !Boolean(object, "completionReady", query.completionReady, error, code)) return false;
                const uint64_t maximumSequence = is_terminal(query.summary.status) ? query.summary.eventCount : query.summary.generatedTokens;
                if (query.completionReady != is_terminal(query.summary.status) || query.bufferedTokens > limits.maxBufferedTokensPerRequest ||
                    query.lastAcknowledgedSequence > query.lastSentSequence || query.lastSentSequence > maximumSequence) {
                    return Fail(error, code, "Inconsistent request progress/transport counters");
                }
                break;
            }
            case Method::RequestAck:
                if (!Fields(object, {"epoch", "acknowledgedSequence"}, {}, error, code) ||
                    !Unsigned(object, "acknowledgedSequence", parsed.acknowledgedSequence, error, code)) return false;
                if (parsed.acknowledgedSequence > static_cast<uint64_t>(limits.maxGeneratedTokens) + 1) {
                    return Fail(error, code, "Acknowledged sequence exceeds stream bounds");
                }
                break;
            case Method::HandleRelease:
                if (!Fields(object, {"epoch", "releaseAccepted"}, {}, error, code) ||
                    !Boolean(object, "releaseAccepted", parsed.releaseAccepted, error, code) || !parsed.releaseAccepted) {
                    if (error.message.empty()) Fail(error, code, "Release result must acknowledge release");
                    return false;
                }
                break;
            case Method::WorkerShutdown:
                if (!Fields(object, {"epoch", "domainStatus", "lastEventOrdinal"}, {}, error, code) ||
                    !Enum(object, "domainStatus", parsed.worker.domainStatus, parse_domain_status, error, code) ||
                    !Unsigned(object, "lastEventOrdinal", parsed.lastEventOrdinal, error, code)) return false;
                if (parsed.worker.domainStatus != DomainStatus::Closed || parsed.lastEventOrdinal == (std::numeric_limits<uint64_t>::max)()) {
                    return Fail(error, code, "Shutdown result must represent a closed domain and a valid event fence");
                }
                break;
            default:
                return Fail(error, code, "Unknown result method");
            }
        }
        response = std::move(parsed);
        return true;
    });
}

bool match_terminal_final(const Event& terminal, const Response& final, ValidationError& error, const Limits& limits) {
    return Guard(error, [&] {
        constexpr auto code = ErrorCode::ProtocolError;
        if (terminal.kind != EventKind::Terminal || final.method != Method::RequestSubmit ||
            final.phase != SubmissionPhase::Completed || final.kind == ResponseKind::Accepted ||
            !valid_epoch(terminal.epoch) || terminal.epoch != final.epoch ||
            !PositiveId(terminal.identity.inferenceId) || !PositiveId(terminal.identity.modelId) || !PositiveId(terminal.identity.sessionId) ||
            terminal.identity.inferenceId != final.identity.inferenceId || terminal.identity.modelId != final.identity.modelId ||
            terminal.identity.sessionId != final.identity.sessionId || final.rpcId != final.identity.inferenceId ||
            !is_terminal(terminal.summary.status) || !is_terminal(final.summary.status) ||
            !validate_summary(terminal.summary, error, limits) || !validate_summary(final.summary, error, limits)) {
            if (error.message.empty()) Fail(error, code, "Terminal/final identity, phase or summary is invalid");
            return false;
        }
        const auto& left = terminal.summary;
        const auto& right = final.summary;
        if (terminal.sequence != left.eventCount || left.eventCount != right.eventCount ||
            left.status != right.status || left.endReason != right.endReason ||
            left.promptTokensProcessed != right.promptTokensProcessed || left.generatedTokens != right.generatedTokens ||
            left.errorCode != right.errorCode || left.message != right.message ||
            (right.status == RequestStatus::Failed) != (final.kind == ResponseKind::Error)) {
            return Fail(error, code, "Terminal and final response disagree");
        }
        return true;
    });
}

bool make_hello(const Hello& hello, Frame& frame, ValidationError& error) {
    return Guard(error, [&] {
        if (!Capabilities(hello.capabilities, error, ErrorCode::InvalidArgument) ||
            !validate_limits(hello.hardLimits, error)) return false;
        Frame candidate = BaseFrame("handshake", "worker.hello", 0);
        Json json(bootstrap_bounds().maxHeaderBytes);
        json.String("epoch", hello.epoch);
        json.String("buildId", hello.buildId);
        json.String("executable", hello.executable);
        json.String("runtimeDirectory", hello.runtimeDirectory);
        json.String("currentDirectory", hello.currentDirectory);
        json.String("path", hello.path);
        json.Integers("supportedProtocols", hello.supportedProtocols);
        json.Strings("capabilities", hello.capabilities);
        std::string limits;
        if (!LimitsJson(hello.hardLimits, limits, error)) return false;
        json.ObjectValue("hardLimits", limits);
        if (!json.Finish(candidate.header.payload_json, error)) return false;
        Hello checked;
        if (!parse_hello(candidate, checked, error)) return false;
        frame = std::move(candidate);
        return true;
    });
}

bool parse_hello(const Frame& frame, Hello& hello, ValidationError& error) {
    return Guard(error, [&] {
        constexpr auto code = ErrorCode::ProtocolError;
        if (frame.header.protocol != kProtocolVersion) return Fail(error, ErrorCode::UnsupportedProtocol, "IPC protocol must be 2");
        if (frame.header.kind != "handshake" || frame.header.method != "worker.hello" || frame.header.request_id != 0) {
            return Fail(error, code, "Expected the unique worker.hello handshake");
        }
        simdjson::dom::parser parser;
        Object object;
        if (!Payload(frame, Channel::Response, bootstrap_bounds(), parser, object, error) || !NoAttachment(frame, error, code) ||
            !Fields(object, {"epoch", "buildId", "executable", "runtimeDirectory", "currentDirectory", "path", "supportedProtocols", "capabilities", "hardLimits"}, {}, error, code)) return false;
        Hello parsed;
        Object limits;
        if (!Epoch(object, parsed.epoch, error, code) ||
            !Text(object, "buildId", parsed.buildId, 256, error, code) ||
            !Text(object, "executable", parsed.executable, MaxIdentityPathBytes, error, code) ||
            !Text(object, "runtimeDirectory", parsed.runtimeDirectory, MaxIdentityPathBytes, error, code) ||
            !Text(object, "currentDirectory", parsed.currentDirectory, MaxIdentityPathBytes, error, code) ||
            !Text(object, "path", parsed.path, MaxIdentityPathBytes, error, code) ||
            !ReadCapabilities(object, "capabilities", parsed.capabilities, error, code) ||
            !ChildObject(object, "hardLimits", limits, error, code) ||
            !ReadLimits(limits, parsed.hardLimits, error, code) || !validate_limits(parsed.hardLimits, error)) return false;
        if (parsed.buildId.empty() || parsed.executable.empty() || parsed.runtimeDirectory.empty() ||
            parsed.currentDirectory.empty() || parsed.path.empty()) {
            return Fail(error, code, "Handshake identity fields must not be empty");
        }
        simdjson::dom::array protocols;
        if (object.at_key("supportedProtocols").get_array().get(protocols) != simdjson::SUCCESS || protocols.size() != 1) {
            return Fail(error, ErrorCode::UnsupportedProtocol, "The v2 launch mode must offer exactly protocol 2");
        }
        uint64_t protocol = 0;
        for (const auto element : protocols) {
            if (element.get_uint64().get(protocol) != simdjson::SUCCESS || protocol != kProtocolVersion) {
                return Fail(error, ErrorCode::UnsupportedProtocol, "The v2 launch mode must offer exactly protocol 2");
            }
        }
        parsed.supportedProtocols = {kProtocolVersion};
        hello = std::move(parsed);
        return true;
    });
}

#undef AILA_V2_WORKER_COUNTS
#undef AILA_V2_LIMIT_FIELDS

} // namespace aila::ipc::v2
