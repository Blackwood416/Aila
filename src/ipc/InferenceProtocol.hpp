#pragma once

#include "ipc/IpcProtocol.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

// Host-only wire types. Do not include inference, SYCL, or oneDNN headers here.
namespace aila::ipc::v2 {

inline constexpr uint32_t kProtocolVersion = 2;
inline constexpr uint32_t kPublicAbiVersion = 1;
inline constexpr size_t kFramePrefixBytes = 8;
inline constexpr size_t kMaxErrorBytes = 1024;

enum class Method {
    WorkerConfigure, WorkerQuery, ModelLoad, SessionCreate, SessionQuery,
    RequestSubmit, RequestCancel, RequestQuery, RequestAck, HandleRelease,
    WorkerShutdown,
};
enum class ObjectType { Model, Session, Request };
enum class RequestStatus { Queued, Prefill, Decode, Finalizing, Completed, Cancelled, Failed };
enum class EndReason { None, Length, StopToken, Cancelled, OutputBackpressure, SchedulerStopped, RuntimeFault, Failed };
enum class ErrorCode {
    InvalidArgument, Busy, ContextOverflow, UnsupportedModel, ResourceLimit,
    RuntimeFault, SchedulerStopped, UnsupportedProtocol, UnsupportedCapability,
    StaleEpoch, InvalidHandle, ProtocolError, RequestFailed, TransportError,
};
enum class DomainStatus { Starting, Ready, Stopping, Faulted, Closed };
enum class ResponseKind { Result, Error, Accepted };
enum class SubmissionPhase { None, Accepted, Rejected, Completed };
enum class EventKind { Token, Terminal, DomainFault, Log };
enum class Channel { Command, Response, Event };

std::string_view to_string(Method value) noexcept;
std::string_view to_string(ObjectType value) noexcept;
std::string_view to_string(RequestStatus value) noexcept;
std::string_view to_string(EndReason value) noexcept;
std::string_view to_string(ErrorCode value) noexcept;
std::string_view to_string(DomainStatus value) noexcept;
bool parse_method(std::string_view value, Method& output) noexcept;
bool parse_object_type(std::string_view value, ObjectType& output) noexcept;
bool parse_request_status(std::string_view value, RequestStatus& output) noexcept;
bool parse_end_reason(std::string_view value, EndReason& output) noexcept;
bool parse_error_code(std::string_view value, ErrorCode& output) noexcept;
bool parse_domain_status(std::string_view value, DomainStatus& output) noexcept;
bool is_terminal(RequestStatus value) noexcept;
bool valid_epoch(std::string_view value) noexcept;

struct ValidationError {
    ErrorCode code = ErrorCode::ProtocolError;
    std::string message;
};

// The first wire revision permits lowering these bounds, not increasing them.
// maxSessionCacheBytes == 0 means no byte ceiling beyond the session count.
// The offered hard limit may also be zero; then a finite requested ceiling is valid.
struct Limits {
    uint32_t maxActiveRequests = 4;
    uint32_t maxOutstandingRequests = 64;
    uint32_t maxPromptTokens = 4096;
    uint32_t maxGeneratedTokens = 1024;
    uint32_t maxStopTokenIds = 64;
    uint64_t maxOutstandingTokens = 32768;
    uint32_t maxPrefillChunkSize = 512;
    uint32_t maxBufferedTokensPerRequest = 128;
    uint32_t maxLoadedModels = 1;
    uint32_t maxSessions = 4;
    uint64_t maxSessionCacheBytes = 0;
    uint32_t maxRetainedRequestHandles = 256;
    uint32_t tokenWindow = 32;
    uint32_t maxTokenFrameBytes = 1024;
    uint32_t maxTerminalFrameBytes = 8192;
    uint32_t maxFinalFrameBytes = 8192;
    uint32_t maxHeaderBytes = 64 * 1024;
    uint32_t maxAttachmentBytes = 16 * 1024;
    uint32_t maxPendingCommands = 64;
    uint32_t reservedControlCommands = 16;
    uint64_t maxIngressBytes = 8 * 1024 * 1024;
    uint64_t maxTransportBytes = 8 * 1024 * 1024;
    uint64_t maxLogQueueBytes = 64 * 1024;
    uint32_t maxShutdownGraceMs = 30000;
    uint32_t connectionWriteTimeoutMs = 10000;
};

Limits hard_limits() noexcept;
// Keep existing Qwen3 callers compatible with Workers that predate Hybrid.
std::vector<std::string> default_capabilities();
// All capabilities published by current real and CPU fixture Workers.
std::vector<std::string> supported_capabilities();
// Empty for unsupported backend/storage combinations; values have static lifetime.
std::string_view model_capability(std::string_view backend, std::string_view kvType = "bf16") noexcept;
bool validate_limits(const Limits& requested, ValidationError& error,
                     const Limits& ceiling = hard_limits());
bool negotiate(const Limits& requested, const std::vector<std::string>& requiredCapabilities,
               const Limits& offered, const std::vector<std::string>& availableCapabilities,
               Limits& selected, ValidationError& error);

struct FrameBounds {
    uint32_t maxHeaderBytes = 64 * 1024;
    uint32_t maxAttachmentBytes = 16 * 1024;
    uint64_t maxFrameBytes = 8 + 64 * 1024 + 16 * 1024;
};
struct FrameSizes {
    uint32_t headerBytes = 0;
    uint32_t attachmentBytes = 0;
    size_t totalBytes = 0;
};
FrameBounds frame_bounds(Channel channel, const Limits& limits = {});
FrameBounds bootstrap_bounds() noexcept;
// Call on the eight-byte prefix BEFORE allocating a body in a pipe reader.
bool inspect_prefix(const std::byte* bytes, size_t count, FrameSizes& sizes,
                    ValidationError& error, const FrameBounds& bounds = {});
// Strict envelope parsing, including duplicate/unknown keys. Payload method
// validation is separate so a valid command with bad arguments can be rejected
// locally without treating it as corrupted framing.
bool encode_frame(const Frame& frame, std::vector<std::byte>& bytes,
                  ValidationError& error, const FrameBounds& bounds = {});
bool decode_frame(const std::vector<std::byte>& bytes, Frame& frame,
                  ValidationError& error, const FrameBounds& bounds = {});

// Enforces nonnegative int32 IDs and byte/count bounds. The Worker still checks
// the loaded vocabulary, Session capacity, ownership, and aggregate admission.
bool encode_token_ids(const std::vector<int32_t>& tokens, std::vector<std::byte>& bytes,
                      ValidationError& error, const Limits& limits = {});
bool decode_token_ids(const std::vector<std::byte>& bytes, uint64_t tokenCount,
                      std::vector<int32_t>& tokens, ValidationError& error,
                      const Limits& limits = {});

struct GenerationOptions {
    uint32_t maxNewTokens = 512;
    bool doSample = true;
    double temperature = 0.6;
    uint32_t topK = 20;
    double topP = 0.95;
    bool useFixedSeed = false;
    uint64_t samplingSeed = 42;
    double repetitionPenalty = 1.0;
    double presencePenalty = 0.0;
    double frequencyPenalty = 0.0;
};

// Only fields belonging to method are serialized. Parsing validates each
// method's exact schema and returns a fresh value, never partly changes output.
struct Command {
    Method method = Method::WorkerQuery;
    uint64_t rpcId = 0;
    std::string epoch;
    Limits limits;
    std::vector<std::string> requiredCapabilities;
    std::string backend = "qwen3_nf4";
    std::string modelDirectory;
    std::string kvType = "bf16";
    uint32_t maxSequenceLength = 4096;
    uint32_t maxSessions = 4;
    uint64_t maxSessionCacheBytes = 0;
    uint64_t modelId = 0;
    uint64_t sessionId = 0;
    uint64_t inferenceId = 0;
    uint32_t prefillChunkSize = 512;
    std::vector<int32_t> promptIds;
    std::vector<int32_t> stopTokenIds;
    GenerationOptions generation;
    ObjectType objectType = ObjectType::Request;
    uint64_t objectId = 0;
    uint64_t consumedSequence = 0;
    uint32_t graceMs = 30000;
};

struct Identity {
    uint64_t inferenceId = 0;
    uint64_t modelId = 0;
    uint64_t sessionId = 0;
};
struct Summary {
    RequestStatus status = RequestStatus::Queued;
    EndReason endReason = EndReason::None;
    uint64_t promptTokensProcessed = 0;
    uint64_t generatedTokens = 0;
    uint64_t eventCount = 0; // Zero only for a nonterminal query snapshot.
    std::optional<ErrorCode> errorCode;
    std::string message;
};
struct ModelInfo {
    uint64_t modelId = 0;
    uint32_t vocabSize = 0;
    uint32_t maxSequenceLength = 0;
    uint64_t sessionReservationBytes = 0;
    std::vector<std::string> capabilities;
};
struct SessionInfo {
    uint64_t sessionId = 0;
    uint64_t modelId = 0;
    uint32_t capacity = 0;
    uint64_t cacheBytes = 0;
    bool busy = false;
    std::optional<uint32_t> contextLength;
};
struct RequestInfo {
    Summary summary;
    uint64_t bufferedTokens = 0;
    uint64_t lastSentSequence = 0;
    uint64_t lastAcknowledgedSequence = 0;
    bool completionReady = false;
};
struct WorkerInfo {
    DomainStatus domainStatus = DomainStatus::Starting;
    bool accepting = false;
    bool stopping = false;
    bool faulted = false;
    uint64_t outstandingRequests = 0;
    uint64_t queuedRequests = 0;
    uint64_t activeRequests = 0;
    uint64_t reservedTokens = 0;
    uint64_t reservedSessionCacheBytes = 0;
    uint64_t ipcOutstandingRequests = 0;
    uint64_t retainedRequestHandles = 0;
    uint64_t pendingCommands = 0;
    uint64_t ingressBytes = 0;
    uint64_t transportBytes = 0;
    uint64_t droppedLogs = 0;
};
struct Response {
    Method method = Method::WorkerQuery;
    uint64_t rpcId = 0;
    std::string epoch;
    ResponseKind kind = ResponseKind::Result;
    SubmissionPhase phase = SubmissionPhase::None;
    // For generic errors and submit rejection. Accepted request failures use
    // summary.errorCode/message, together with phase == Completed.
    std::optional<ErrorCode> errorCode;
    std::string message;
    Identity identity;
    Summary summary; // Final submit or cancel's status.
    uint32_t tokenWindow = 32;
    Limits limits; // worker.configure result
    std::vector<std::string> capabilities; // worker.configure result
    WorkerInfo worker; // Also supplies domainStatus for configure/shutdown.
    ModelInfo model;
    SessionInfo session;
    RequestInfo request;
    bool cancelRequested = false;
    uint64_t acknowledgedSequence = 0;
    bool releaseAccepted = true;
    uint64_t lastEventOrdinal = 0;
};
struct Event {
    EventKind kind = EventKind::Token;
    std::string epoch;
    Identity identity;
    uint64_t sequence = 0;
    uint64_t eventOrdinal = 0;
    int32_t tokenId = -1;
    Summary summary; // Token progress or terminal summary.
    int32_t logLevel = 0;
    // DomainFault uses errorCode/message; Log uses message. These do not use
    // summary or per-request identity/sequence and have header requestId zero.
    std::optional<ErrorCode> errorCode;
    std::string message;
};
struct Hello {
    std::string epoch;
    std::string buildId;
    std::string executable;
    std::string runtimeDirectory;
    std::string currentDirectory;
    std::string path;
    std::vector<uint32_t> supportedProtocols{kProtocolVersion};
    std::vector<std::string> capabilities;
    Limits hardLimits = hard_limits();
};

bool make_command(const Command& command, Frame& frame, ValidationError& error, const Limits& limits = {});
bool parse_command(const Frame& frame, Command& command, ValidationError& error, const Limits& limits = {});
bool make_response(const Response& response, Frame& frame, ValidationError& error, const Limits& limits = {});
bool parse_response(const Frame& frame, Response& response, ValidationError& error, const Limits& limits = {});
bool make_event(const Event& event, Frame& frame, ValidationError& error, const Limits& limits = {});
bool parse_event(const Frame& frame, Event& event, ValidationError& error, const Limits& limits = {});
bool make_hello(const Hello& hello, Frame& frame, ValidationError& error);
bool parse_hello(const Frame& frame, Hello& hello, ValidationError& error);

bool validate_summary(const Summary& summary, ValidationError& error, const Limits& limits = {});
// Cross-pipe identity, sequence/count and full-summary equality. The connection
// still tracks acceptedSeen and every preceding sequence independently.
bool match_terminal_final(const Event& terminal, const Response& final,
                          ValidationError& error, const Limits& limits = {});

} // namespace aila::ipc::v2
