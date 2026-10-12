#include "ipc/InferenceProtocol.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace wire = aila::ipc::v2;
using aila::ipc::Frame;

namespace {

constexpr const char* Epoch = "0123456789abcdef0123456789abcdef";

void Expect(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
void Success(bool condition, const wire::ValidationError& error) {
    Expect(condition, error.message.empty() ? "Expected success" : error.message);
}
void Replace(std::string& text, const std::string& from, const std::string& to) {
    const size_t position = text.find(from);
    Expect(position != std::string::npos, "Test replacement target was absent: " + from);
    text.replace(position, from.size(), to);
}
void NumberField(std::string& text, const char* key, const char* value) {
    const std::string prefix = std::string("\"") + key + "\":";
    const size_t begin = text.find(prefix);
    Expect(begin != std::string::npos, "Test numeric field was absent");
    const size_t number = begin + prefix.size();
    const size_t end = text.find_first_of(",}", number);
    Expect(end != std::string::npos, "Test numeric field has no delimiter");
    text.replace(number, end - number, value);
}
std::vector<std::byte> RawFrame(const std::string& header, const std::vector<std::byte>& attachment = {}) {
    std::vector<std::byte> bytes(8 + header.size() + attachment.size());
    aila::ipc::write_u32_le(bytes.data(), static_cast<uint32_t>(header.size()));
    aila::ipc::write_u32_le(bytes.data() + 4, static_cast<uint32_t>(attachment.size()));
    std::memcpy(bytes.data() + 8, header.data(), header.size());
    if (!attachment.empty()) std::memcpy(bytes.data() + 8 + header.size(), attachment.data(), attachment.size());
    return bytes;
}
std::string Header(const std::vector<std::byte>& bytes) {
    return {reinterpret_cast<const char*>(bytes.data() + 8), aila::ipc::read_u32_le(bytes.data())};
}
wire::Command Command(wire::Method method) {
    wire::Command command;
    command.method = method;
    command.rpcId = 77;
    command.epoch = Epoch;
    command.requiredCapabilities = wire::default_capabilities();
    command.modelDirectory = "C:\\models\\qwen3";
    command.modelId = 3;
    command.sessionId = 7;
    command.inferenceId = 21;
    command.promptIds = {2, 17, 42};
    command.stopTokenIds = {9, 12};
    command.prefillChunkSize = 2;
    command.generation.maxNewTokens = 8;
    command.generation.useFixedSeed = true;
    command.generation.samplingSeed = (std::numeric_limits<uint64_t>::max)();
    command.objectType = wire::ObjectType::Session;
    command.objectId = 7;
    command.consumedSequence = 3;
    command.graceMs = 1000;
    return command;
}
Frame Make(const wire::Command& command) {
    Frame frame;
    wire::ValidationError error;
    Success(wire::make_command(command, frame, error), error);
    return frame;
}
wire::Summary CompleteSummary() {
    wire::Summary summary;
    summary.status = wire::RequestStatus::Completed;
    summary.endReason = wire::EndReason::Length;
    summary.promptTokensProcessed = 3;
    summary.generatedTokens = 4;
    summary.eventCount = 5;
    return summary;
}
wire::Response Final(const wire::Summary& summary = CompleteSummary()) {
    wire::Response response;
    response.method = wire::Method::RequestSubmit;
    response.rpcId = 21;
    response.epoch = Epoch;
    response.kind = summary.status == wire::RequestStatus::Failed ? wire::ResponseKind::Error : wire::ResponseKind::Result;
    response.phase = wire::SubmissionPhase::Completed;
    response.identity = {21, 3, 7};
    response.summary = summary;
    return response;
}
wire::Event Terminal(const wire::Summary& summary = CompleteSummary()) {
    wire::Event event;
    event.kind = wire::EventKind::Terminal;
    event.epoch = Epoch;
    event.identity = {21, 3, 7};
    event.sequence = summary.eventCount;
    event.eventOrdinal = 16;
    event.summary = summary;
    return event;
}
void RejectCommand(const Frame& frame) {
    wire::Command parsed;
    parsed.rpcId = 999;
    wire::ValidationError error;
    Expect(!wire::parse_command(frame, parsed, error), "Malformed command was accepted");
    Expect(!error.message.empty(), "Rejected command omitted an error message");
    Expect(parsed.rpcId == 999, "Failed command parse partly changed output");
}

void TestVersionsAndNames() {
    static_assert(aila::ipc::kProtocolVersion == 1);
    static_assert(wire::kProtocolVersion == 2 && wire::kPublicAbiVersion == 1);
    Expect(wire::to_string(wire::EndReason::OutputBackpressure) == "output_backpressure", "Unstable end reason spelling");
    Expect(wire::to_string(wire::ErrorCode::RuntimeFault) == "RuntimeFault", "Unstable error spelling");
    wire::Method method;
    Expect(wire::parse_method("request.ack", method) && method == wire::Method::RequestAck, "ACK method mapping failed");
    Expect(!wire::parse_method("request.step", method), "Unimplemented method was accepted");
    wire::RequestStatus status;
    Expect(!wire::parse_request_status("Completed", status), "Status parsing was not case sensitive");
    Expect(wire::valid_epoch(Epoch), "Valid epoch rejected");
    Expect(!wire::valid_epoch("0123456789ABCDEF0123456789abcdef") && !wire::valid_epoch("1"), "Malformed epoch accepted");

    Frame legacy;
    legacy.header.request_id = 1;
    legacy.header.kind = "request";
    legacy.header.method = "engine.init";
    const auto bytes = aila::ipc::encode_frame(legacy);
    Frame decoded;
    std::string legacyError;
    Expect(aila::ipc::decode_frame(bytes, decoded, legacyError), "Existing v1 codec stopped working");
    wire::ValidationError error;
    Expect(!wire::decode_frame(bytes, decoded, error) && error.code == wire::ErrorCode::UnsupportedProtocol,
           "v2 silently accepted a v1 frame");
    legacy.header.kind = "handshake";
    legacy.header.method.clear();
    legacy.header.request_id = 0;
    wire::Hello hello;
    Expect(!wire::parse_hello(legacy, hello, error) && error.code == wire::ErrorCode::UnsupportedProtocol,
           "Direct handshake parsing lost the v1 rejection reason");
}

void TestLimitsAndNegotiation() {
    wire::ValidationError error;
    wire::Limits defaults;
    Success(wire::validate_limits(defaults, error), error);
    Expect(defaults.maxActiveRequests == 4 && defaults.maxOutstandingRequests == 64 && defaults.tokenWindow == 32,
           "Scheduler/transport defaults changed");
    Expect(defaults.maxShutdownGraceMs == 30000 && defaults.connectionWriteTimeoutMs == 10000, "Timeout defaults changed");
    wire::Limits lower = defaults;
    lower.maxActiveRequests = 1;
    lower.maxSessions = 2;
    lower.maxOutstandingRequests = 8;
    lower.tokenWindow = 4;
    lower.maxSessionCacheBytes = UINT64_C(123456789);
    wire::Limits selected;
    Success(wire::negotiate(lower, {"token_requests_v1"}, defaults, wire::default_capabilities(), selected, error), error);
    Expect(selected.maxActiveRequests == 1 && selected.maxSessionCacheBytes == lower.maxSessionCacheBytes, "Negotiation silently changed limits");
    const auto preserved = selected;
    Expect(!wire::negotiate(lower, {"continuous_batching_v1"}, defaults, wire::default_capabilities(), selected, error) &&
           error.code == wire::ErrorCode::UnsupportedCapability, "Missing capability was not rejected");
    Expect(selected.maxActiveRequests == preserved.maxActiveRequests, "Failed negotiation changed selected limits");
    Expect(!wire::negotiate(lower, {"token_requests_v1", "token_requests_v1"}, defaults, wire::default_capabilities(), selected, error),
           "Duplicate required capabilities accepted");

    std::vector<wire::Limits> invalid;
    auto value = defaults; value.maxActiveRequests = 5; invalid.push_back(value);
    value = defaults; value.tokenWindow = 0; invalid.push_back(value);
    value = defaults; value.reservedControlCommands = value.maxPendingCommands; invalid.push_back(value);
    value = defaults; value.maxOutstandingRequests = 1; invalid.push_back(value);
    value = defaults; value.maxRetainedRequestHandles = 8; invalid.push_back(value);
    value = defaults; value.maxAttachmentBytes = 4; invalid.push_back(value);
    value = defaults; value.maxIngressBytes = 1; invalid.push_back(value);
    value = defaults; value.maxTransportBytes = 1; invalid.push_back(value);
    value = defaults; value.connectionWriteTimeoutMs = 10001; invalid.push_back(value);
    value = defaults; value.maxShutdownGraceMs = 0; invalid.push_back(value);
    for (const auto& bad : invalid) Expect(!wire::validate_limits(bad, error), "Inconsistent limits accepted");
    wire::Limits finite = defaults;
    finite.maxSessionCacheBytes = 10;
    Expect(!wire::validate_limits(defaults, error, finite), "Unlimited requested cache escaped a finite ceiling");
    wire::Limits widenedCeiling = defaults;
    widenedCeiling.maxSessions = 5;
    Expect(!wire::validate_limits(widenedCeiling, error, widenedCeiling), "A caller-supplied ceiling bypassed absolute hard limits");
}

void TestPrefixAndEnvelope() {
    wire::ValidationError error;
    std::array<std::byte, 8> prefix{};
    aila::ipc::write_u32_le(prefix.data(), 65537);
    wire::FrameSizes sizes;
    sizes.totalBytes = 999;
    Expect(!wire::inspect_prefix(prefix.data(), prefix.size(), sizes, error), "Oversized header prefix accepted");
    Expect(sizes.totalBytes == 999, "Prefix failure changed output");
    aila::ipc::write_u32_le(prefix.data(), 10);
    aila::ipc::write_u32_le(prefix.data() + 4, 16385);
    Expect(!wire::inspect_prefix(prefix.data(), prefix.size(), sizes, error), "Oversized attachment prefix accepted");
    Expect(!wire::inspect_prefix(prefix.data(), 7, sizes, error), "Partial prefix accepted");
    aila::ipc::write_u32_le(prefix.data() + 4, 4);
    Expect(!wire::inspect_prefix(prefix.data(), prefix.size(), sizes, error, wire::frame_bounds(wire::Channel::Response)),
           "Response pipe accepted attachment bytes");

    const Frame command = Make(Command(wire::Method::WorkerQuery));
    std::vector<std::byte> bytes;
    Success(wire::encode_frame(command, bytes, error), error);
    const std::string goodHeader = Header(bytes);
    std::vector<std::string> invalid;
    auto header = goodHeader; Replace(header, "\"requestId\":77", "\"requestId\":77,\"requestId\":78"); invalid.push_back(header);
    header = goodHeader; Replace(header, "\"requestId\":77", "\"requestId\":77,\"request\\u0049d\":78"); invalid.push_back(header);
    header = goodHeader; header.insert(header.size() - 1, ",\"extension\":true"); invalid.push_back(header);
    header = goodHeader; Replace(header, "\"abi\":1", "\"abi\":2"); invalid.push_back(header);
    header = goodHeader; Replace(header, "\"requestId\":77", "\"requestId\":77.0"); invalid.push_back(header);
    header = goodHeader; Replace(header, "\"requestId\":77", "\"requestId\":18446744073709551616"); invalid.push_back(header);
    header = goodHeader; Replace(header, "\"requestId\":77", "\"requestId\":0"); invalid.push_back(header);
    header = goodHeader; Replace(header, "\"kind\":\"request\"", "\"kind\":false"); invalid.push_back(header);
    header = goodHeader; Replace(header, "\"protocol\":2,", ""); invalid.push_back(header);
    for (const auto& bad : invalid) {
        Frame parsed;
        parsed.header.request_id = 999;
        Expect(!wire::decode_frame(RawFrame(bad), parsed, error), "Malformed envelope accepted");
        Expect(parsed.header.request_id == 999, "Failed envelope parse changed output");
    }
    auto truncated = bytes; truncated.pop_back();
    Frame parsed;
    Expect(!wire::decode_frame(truncated, parsed, error), "Truncated body accepted");
    bytes.push_back(std::byte{0});
    Expect(!wire::decode_frame(bytes, parsed, error), "Trailing byte accepted");
}

void TestLittleEndianTokens() {
    const std::vector<int32_t> tokens{0, 1, 0x01020304, (std::numeric_limits<int32_t>::max)()};
    const std::vector<std::byte> expected{
        std::byte{0}, std::byte{0}, std::byte{0}, std::byte{0},
        std::byte{1}, std::byte{0}, std::byte{0}, std::byte{0},
        std::byte{4}, std::byte{3}, std::byte{2}, std::byte{1},
        std::byte{0xff}, std::byte{0xff}, std::byte{0xff}, std::byte{0x7f},
    };
    wire::ValidationError error;
    std::vector<std::byte> bytes;
    Success(wire::encode_token_ids(tokens, bytes, error), error);
    Expect(bytes == expected, "Token IDs are not encoded explicitly as i32 little endian");
    std::vector<int32_t> decoded;
    Success(wire::decode_token_ids(expected, 4, decoded, error), error);
    Expect(decoded == tokens, "LE token decoding changed values");
    Expect(!wire::decode_token_ids(bytes, (std::numeric_limits<uint64_t>::max)(), decoded, error), "Count overflow escaped bounds");
    Expect(!wire::decode_token_ids(bytes, 3, decoded, error), "Mismatched count accepted");
    Expect(!wire::decode_token_ids({std::byte{0xff}, std::byte{0xff}, std::byte{0xff}, std::byte{0xff}}, 1, decoded, error),
           "Negative encoded token accepted");
    Expect(!wire::encode_token_ids({-1}, bytes, error), "Negative input token accepted");
    std::vector<int32_t> excessive(4097, 1);
    Expect(!wire::encode_token_ids(excessive, bytes, error), "Prompt token hard limit ignored");
}

void TestAllCommands() {
    const wire::Method methods[]{wire::Method::WorkerConfigure, wire::Method::WorkerQuery, wire::Method::ModelLoad,
        wire::Method::SessionCreate, wire::Method::SessionQuery, wire::Method::RequestSubmit, wire::Method::RequestCancel,
        wire::Method::RequestQuery, wire::Method::RequestAck, wire::Method::HandleRelease, wire::Method::WorkerShutdown,
        wire::Method::SessionReset, wire::Method::SessionTruncate, wire::Method::WorkerSetLogLevel, wire::Method::WorkerMemoryQuery};
    for (const auto method : methods) {
        const auto input = Command(method);
        const Frame frame = Make(input);
        wire::ValidationError error;
        std::vector<std::byte> bytes;
        Success(wire::encode_frame(frame, bytes, error), error);
        Frame wireFrame;
        Success(wire::decode_frame(bytes, wireFrame, error), error);
        wire::Command parsed;
        Success(wire::parse_command(wireFrame, parsed, error), error);
        Expect(parsed.method == method && parsed.rpcId == input.rpcId && parsed.epoch == input.epoch, "Command identity changed");
        if (method == wire::Method::RequestSubmit) {
            Expect(parsed.inferenceId == input.rpcId && parsed.promptIds == input.promptIds && parsed.stopTokenIds == input.stopTokenIds,
                   "Submit identity or token data changed");
            Expect(parsed.generation.samplingSeed == (std::numeric_limits<uint64_t>::max)(), "uint64 seed lost precision");
        }
    }
}

void TestRequestPolicyExtensions() {
    auto command = Command(wire::Method::RequestSubmit);
    command.thinkingBudget = wire::ThinkingBudgetOptions{2, 10, 11, true, {11, 3}, true};
    command.forceHostSampling = true;
    command.legacyPrefillLogits = true;
    const auto frame = Make(command);
    wire::Command parsed;
    wire::ValidationError error;
    Success(wire::parse_command(frame, parsed, error), error);
    Expect(parsed.forceHostSampling && parsed.legacyPrefillLogits && parsed.thinkingBudget && parsed.thinkingBudget->closeOnce,
           "Request policies changed in transit");
    for (const std::string field : {"closeOnce", "hostSampling", "legacyPrefillLogits"}) {
        for (const char* replacement : {"false", "null", "1", "\"true\""}) {
            auto bad = frame;
            Replace(bad.header.payload_json, "\"" + field + "\":true", "\"" + field + "\":" + replacement);
            Expect(!wire::parse_command(bad, parsed, error), "Invalid request policy value was accepted");
        }
    }
}

void TestPrefillOnlyExtension() {
    auto command = Command(wire::Method::RequestSubmit);
    Expect(Make(command).header.payload_json.find("prefillOnly") == std::string::npos,
           "Ordinary generation changed its wire schema");
    command.prefillOnly = true;
    command.generation.maxNewTokens = 0;
    command.stopTokenIds.clear();
    const auto frame = Make(command);
    wire::Command parsed;
    wire::ValidationError error;
    Success(wire::parse_command(frame, parsed, error), error);
    Expect(parsed.prefillOnly && parsed.generation.maxNewTokens == 0 && parsed.promptIds == command.promptIds,
           "Replay mode or prefix changed in transit");
    for (const char* replacement : {"false", "null", "1", "\"true\""}) {
        auto bad = frame;
        Replace(bad.header.payload_json, "\"prefillOnly\":true", std::string("\"prefillOnly\":") + replacement);
        Expect(!wire::parse_command(bad, parsed, error), "Invalid replay mode was accepted");
    }
    for (int invalid = 0; invalid < 5; ++invalid) {
        auto bad = command;
        if (invalid == 0) bad.generation.maxNewTokens = 1;
        if (invalid == 1) bad.prefillOnly = false;
        if (invalid == 2) bad.promptIds.clear();
        if (invalid == 3) bad.stopTokenIds = {9};
        if (invalid == 4) bad.thinkingBudget = wire::ThinkingBudgetOptions{1, 10, 11, true, {11}};
        Frame ignored;
        Expect(!wire::make_command(bad, ignored, error), "Contradictory replay options were encoded");
    }
}

void TestModelCapabilities() {
    const std::vector<std::string> legacy{
        "token_requests_v1", "qwen3_nf4_bf16_kv", "session_lifecycle_v1", "multi_inflight_v1",
        "cumulative_ack_v1", "domain_epoch_v1"};
    Expect(wire::default_capabilities() == legacy, "Legacy Qwen3 required capabilities changed");
    auto current = legacy;
    current.emplace_back("qwen35_nf4_bf16_kv");
    current.emplace_back("allocation_limits_v1");
    const auto nf4Only = current;
    current.emplace_back("qwen35_bf16_bf16_kv");
    // A Worker that offers both Hybrid storages but not Qwen4Exp must still
    // reject a Qwen4Exp requirement rather than silently accepting it.
    const auto withoutQwen4Exp = current;
    current.emplace_back("thinking_budget_v1");
    current.emplace_back("session_context_v1");
    current.emplace_back("prefill_only_v1");
    current.emplace_back("thinking_budget_close_once_v1");
    current.emplace_back("host_sampling_v1");
    current.emplace_back("legacy_prefill_logits_v1");
    current.emplace_back("extended_generation_v1");
    current.emplace_back("extended_context_v1");
    current.emplace_back("worker_log_control_v1");
    current.emplace_back("runtime_memory_v1");
    current.emplace_back("continuous_batching_v1");
    current.emplace_back("qwen4exp_nf4_bf16_kv");
    current.emplace_back("qwen4exp_nf4_int8_kv");
    Expect(wire::supported_capabilities() == current, "Worker capabilities omitted a backend or changed Qwen3");
    Expect(wire::model_capability("qwen3_nf4") == "qwen3_nf4_bf16_kv" &&
           wire::model_capability("qwen35_nf4", "bf16") == "qwen35_nf4_bf16_kv" &&
           wire::model_capability("qwen35_bf16", "bf16") == "qwen35_bf16_bf16_kv" &&
           wire::model_capability("qwen4exp_nf4", "bf16") == "qwen4exp_nf4_bf16_kv" &&
           wire::model_capability("qwen4exp_nf4", "int8") == "qwen4exp_nf4_int8_kv",
           "Backend capability mapping was incorrect");
    Expect(wire::model_capability("qwen35_nf4", "fp8").empty() &&
           wire::model_capability("qwen35_bf16", "fp8").empty() &&
           wire::model_capability("qwen4exp_nf4", "fp8").empty() &&
           wire::model_capability("qwen3_nf4", "int8").empty() &&
           wire::model_capability("qwen35_nf4", "int8").empty() &&
           wire::model_capability("qwen35_bf16", "int8").empty() &&
           wire::model_capability("qwen3_nf4", "BF16").empty() &&
           wire::model_capability("future_nf4").empty(), "Unknown backend or KV type was advertised");

    wire::Limits selected;
    wire::ValidationError error;
    Expect(!wire::negotiate({}, {"qwen35_nf4_bf16_kv"}, {}, legacy, selected, error) &&
           error.code == wire::ErrorCode::UnsupportedCapability, "Legacy Worker accepted Hybrid requirements");
    Success(wire::negotiate({}, legacy, {}, current, selected, error), error);
    Success(wire::negotiate({}, {"qwen35_nf4_bf16_kv"}, {}, current, selected, error), error);
    Expect(!wire::negotiate({}, {"qwen35_bf16_bf16_kv"}, {}, nf4Only, selected, error) &&
           error.code == wire::ErrorCode::UnsupportedCapability, "NF4 Worker accepted BF16 weights");
    Success(wire::negotiate({}, nf4Only, {}, current, selected, error), error);
    Success(wire::negotiate({}, {"qwen35_bf16_bf16_kv"}, {}, current, selected, error), error);
    Expect(!wire::negotiate({}, {"qwen4exp_nf4_bf16_kv"}, {}, withoutQwen4Exp, selected, error) &&
           error.code == wire::ErrorCode::UnsupportedCapability, "Worker without Qwen4Exp accepted a Qwen4Exp requirement");
    Success(wire::negotiate({}, {"qwen4exp_nf4_bf16_kv"}, {}, current, selected, error), error);
    for (const char* backend : {"qwen3_nf4", "qwen35_nf4", "qwen35_bf16", "qwen4exp_nf4"}) {
        auto command = Command(wire::Method::ModelLoad);
        command.backend = backend;
        Frame sent = Make(command);
        std::vector<std::byte> bytes;
        Success(wire::encode_frame(sent, bytes, error), error);
        Frame received;
        Success(wire::decode_frame(bytes, received, error), error);
        wire::Command parsed;
        Success(wire::parse_command(received, parsed, error), error);
        Expect(parsed.backend == backend && parsed.kvType == "bf16" &&
               received.header.protocol == 2 && received.header.abi == 1,
               "Model command round trip changed backend or wire version");

        auto invalid = sent;
        Replace(invalid.header.payload_json, "\"bf16\"", "\"fp8\"");
        parsed.rpcId = 999;
        Expect(!wire::parse_command(invalid, parsed, error) && error.code == wire::ErrorCode::InvalidArgument &&
               parsed.rpcId == 999, "Invalid KV type lost local error semantics or changed output");
    }
    auto invalid = Make(Command(wire::Method::ModelLoad));
    Replace(invalid.header.payload_json, "\"qwen3_nf4\"", "\"future_nf4\"");
    wire::Command parsed;
    Expect(!wire::parse_command(invalid, parsed, error) && error.code == wire::ErrorCode::UnsupportedModel,
           "Unknown model backend lost local UnsupportedModel semantics");
}

void TestExtendedGenerationLimits() {
    auto configure = Command(wire::Method::WorkerConfigure);
    configure.limits.maxGeneratedTokens = 2048;
    wire::ValidationError error;
    Frame frame;
    Expect(!wire::make_command(configure, frame, error), "Larger generation limit did not require its capability");
    configure.requiredCapabilities.emplace_back("extended_generation_v1");
    frame = Make(configure);
    wire::Command parsed;
    Success(wire::parse_command(frame, parsed, error), error);
    Expect(parsed.limits.maxGeneratedTokens == 2048 && parsed.limits.maxSessions == 4 &&
               parsed.limits.maxOutstandingTokens == 32768 && parsed.limits.tokenWindow == 32,
           "Generation extension changed unrelated reservations");
    Expect(!wire::validate_limits(configure.limits, error), "Default validation silently expanded its limits");
    auto reduced = wire::hard_limits();
    reduced.maxGeneratedTokens = 512;
    Expect(wire::configure_limits_ceiling(reduced, configure.requiredCapabilities).maxGeneratedTokens == 512,
           "Extension bypassed a lower offered ceiling");
    configure.limits.maxGeneratedTokens = wire::kExtendedMaxGeneratedTokens;
    Success(wire::make_command(configure, frame, error), error);
    ++configure.limits.maxGeneratedTokens;
    Expect(!wire::make_command(configure, frame, error), "Extended limit overflow was accepted");

    auto limits = wire::hard_limits();
    limits.maxGeneratedTokens = 2048;
    auto submit = Command(wire::Method::RequestSubmit);
    submit.generation.maxNewTokens = 1100;
    Success(wire::make_command(submit, frame, error, limits), error);
    Success(wire::parse_command(frame, parsed, error, limits), error);
    Expect(!wire::parse_command(frame, parsed, error), "Default connection accepted an extended request");
    auto summary = CompleteSummary();
    summary.generatedTokens = 1100;
    summary.eventCount = 1101;
    wire::Event event;
    Success(wire::make_event(Terminal(summary), frame, error, limits), error);
    Success(wire::parse_event(frame, event, error, limits), error);
    Expect(event.sequence == 1101 && !wire::parse_event(frame, event, error),
           "Terminal count was truncated or accepted outside negotiated limits");
    auto ack = Command(wire::Method::RequestAck);
    ack.consumedSequence = 1101;
    Success(wire::make_command(ack, frame, error, limits), error);
    Success(wire::parse_command(frame, parsed, error, limits), error);
    Expect(parsed.consumedSequence == 1101 && !wire::parse_command(frame, parsed, error),
           "Extended terminal ACK lost its bound");
}

void TestExtendedContextLimits() {
    wire::ValidationError error;
    // The absolute combination ceiling keeps every field at its own design
    // bound: generation from one extension, context from the other.
    const auto absolute = wire::absolute_limits();
    Expect(absolute.maxPromptTokens == wire::kExtendedMaxPromptTokens &&
               absolute.maxAttachmentBytes == wire::kExtendedMaxAttachmentBytes &&
               absolute.maxPrefillChunkSize == wire::kExtendedMaxPrefillChunkSize &&
               absolute.maxOutstandingTokens == wire::kExtendedMaxOutstandingTokens &&
               absolute.maxGeneratedTokens == wire::kExtendedMaxGeneratedTokens,
           "Absolute combination ceiling lost an extended field");
    // The aggregate budget must admit four maximal prompt+output requests, and
    // the attachment ceiling must be able to carry the maximum prompt.
    Expect(wire::kExtendedMaxOutstandingTokens ==
               4ULL * (wire::kExtendedMaxPromptTokens + wire::kExtendedMaxGeneratedTokens),
           "Outstanding token ceiling cannot hold four long requests");
    Expect(wire::kExtendedMaxAttachmentBytes / sizeof(int32_t) >= wire::kExtendedMaxPromptTokens,
           "Attachment bytes cannot carry the maximum prompt");

    // Default validation still refuses each widened field; the capability is
    // what unlocks them, exactly as with the generation extension.
    for (int field = 0; field != 4; ++field) {
        auto widened = wire::hard_limits();
        if (field == 0) widened.maxPromptTokens = wire::kExtendedMaxPromptTokens;
        if (field == 1) widened.maxAttachmentBytes = wire::kExtendedMaxAttachmentBytes;
        if (field == 2) widened.maxPrefillChunkSize = wire::kExtendedMaxPrefillChunkSize;
        if (field == 3) widened.maxOutstandingTokens = wire::kExtendedMaxOutstandingTokens;
        Expect(!wire::validate_limits(widened, error), "Unqualified context ceiling was accepted");
    }

    // Configure carrying both capabilities negotiates the full combination
    // without disturbing any unrelated limit.
    auto configure = Command(wire::Method::WorkerConfigure);
    configure.requiredCapabilities.emplace_back("extended_generation_v1");
    configure.requiredCapabilities.emplace_back("extended_context_v1");
    configure.limits.maxGeneratedTokens = wire::kExtendedMaxGeneratedTokens;
    configure.limits.maxPromptTokens = wire::kExtendedMaxPromptTokens;
    configure.limits.maxAttachmentBytes = wire::kExtendedMaxAttachmentBytes;
    configure.limits.maxPrefillChunkSize = wire::kExtendedMaxPrefillChunkSize;
    configure.limits.maxOutstandingTokens = wire::kExtendedMaxOutstandingTokens;
    Frame frame = Make(configure);
    wire::Command parsed;
    Success(wire::parse_command(frame, parsed, error), error);
    Expect(parsed.limits.maxGeneratedTokens == wire::kExtendedMaxGeneratedTokens &&
               parsed.limits.maxPromptTokens == wire::kExtendedMaxPromptTokens &&
               parsed.limits.maxAttachmentBytes == wire::kExtendedMaxAttachmentBytes &&
               parsed.limits.maxPrefillChunkSize == wire::kExtendedMaxPrefillChunkSize &&
               parsed.limits.maxOutstandingTokens == wire::kExtendedMaxOutstandingTokens &&
               parsed.limits.maxSessions == 4 && parsed.limits.tokenWindow == 32,
           "Combined extension perturbed unrelated limits");
    Expect(!wire::validate_limits(configure.limits, error), "Combined ceiling leaked past default validation");

    // One field past its own design bound is rejected even for a qualified peer.
    auto overflow = configure;
    overflow.limits.maxAttachmentBytes = wire::kExtendedMaxAttachmentBytes + 4;
    Expect(!wire::make_command(overflow, frame, error), "Attachment ceiling overflow was accepted");
    overflow = configure;
    overflow.limits.maxPromptTokens = wire::kExtendedMaxPromptTokens + 1;
    Expect(!wire::make_command(overflow, frame, error), "Prompt ceiling overflow was accepted");
    overflow = configure;
    overflow.limits.maxPrefillChunkSize = wire::kExtendedMaxPrefillChunkSize + 1;
    Expect(!wire::make_command(overflow, frame, error), "Prefill ceiling overflow was accepted");
    overflow = configure;
    overflow.limits.maxOutstandingTokens = wire::kExtendedMaxOutstandingTokens + 1;
    Expect(!wire::make_command(overflow, frame, error), "Outstanding ceiling overflow was accepted");

    // A Worker deliberately offering a smaller ceiling keeps it, and the two
    // extensions compose independently.
    auto reduced = wire::hard_limits();
    reduced.maxPromptTokens = 8192;
    reduced.maxGeneratedTokens = 512;
    auto both = configure.requiredCapabilities;
    const auto ceiling = wire::configure_limits_ceiling(reduced, both);
    Expect(ceiling.maxPromptTokens == 8192 && ceiling.maxGeneratedTokens == 512 &&
               ceiling.maxAttachmentBytes == wire::kExtendedMaxAttachmentBytes &&
               ceiling.maxPrefillChunkSize == wire::kExtendedMaxPrefillChunkSize &&
               ceiling.maxOutstandingTokens == wire::kExtendedMaxOutstandingTokens,
           "Extension widened a deliberately reduced ceiling");
    const auto generationOnly = wire::configure_limits_ceiling(
        wire::hard_limits(), {std::string(wire::kExtendedGenerationCapability)});
    Expect(generationOnly.maxGeneratedTokens == wire::kExtendedMaxGeneratedTokens &&
               generationOnly.maxPromptTokens == wire::hard_limits().maxPromptTokens &&
               generationOnly.maxOutstandingTokens == wire::hard_limits().maxOutstandingTokens,
           "Generation extension silently widened a context field");
    const auto contextOnly = wire::configure_limits_ceiling(
        wire::hard_limits(), {std::string(wire::kExtendedContextCapability)});
    Expect(contextOnly.maxGeneratedTokens == wire::hard_limits().maxGeneratedTokens &&
               contextOnly.maxPromptTokens == wire::kExtendedMaxPromptTokens,
           "Context extension silently widened the generation field");

    // Submit round trips an actually long attachment the default limit could
    // never carry, and the negotiated limits bound both reservation terms.
    const auto negotiated = configure.limits;
    auto submit = Command(wire::Method::RequestSubmit);
    submit.promptIds.assign(65536, 5);
    submit.promptIds.back() = 3;
    submit.generation.maxNewTokens = wire::kExtendedMaxGeneratedTokens;
    submit.prefillChunkSize = wire::kExtendedMaxPrefillChunkSize;
    Success(wire::make_command(submit, frame, error, negotiated), error);
    Expect(frame.attachment.size() == submit.promptIds.size() * sizeof(int32_t),
           "Long attachment was truncated");
    wire::Command recovered;
    Success(wire::parse_command(frame, recovered, error, negotiated), error);
    Expect(recovered.promptIds == submit.promptIds && recovered.prefillChunkSize == wire::kExtendedMaxPrefillChunkSize,
           "Long attachment changed in transit");
    Expect(!wire::parse_command(frame, recovered, error), "Default connection accepted a long attachment");
    // One token beyond the attachment ceiling still fails on the qualified pipe.
    auto beyond = submit;
    beyond.promptIds.assign(wire::kExtendedMaxPromptTokens + 1, 1);
    Expect(!wire::make_command(beyond, frame, error, negotiated), "Attachment exceeded its negotiated ceiling");
}

void TestStrictCommandPayloads() {
    const Frame submit = Make(Command(wire::Method::RequestSubmit));
    std::vector<Frame> invalid;
    auto bad = submit; bad.header.payload_json.insert(bad.header.payload_json.size() - 1, ",\"unknown\":1"); invalid.push_back(bad);
    bad = submit; Replace(bad.header.payload_json, "\"generation\":{", "\"generation\":{\"topK\":0,"); invalid.push_back(bad);
    bad = submit; Replace(bad.header.payload_json, "\"generation\":{", "\"generation\":{\"\\u0074opK\":0,"); invalid.push_back(bad);
    bad = submit; Replace(bad.header.payload_json, "\"generation\":{", "\"generation\":{\"thinkingBudgetTokens\":5,"); invalid.push_back(bad);
    bad = submit; NumberField(bad.header.payload_json, "temperature", "true"); invalid.push_back(bad);
    bad = submit; NumberField(bad.header.payload_json, "temperature", "0"); invalid.push_back(bad);
    bad = submit; NumberField(bad.header.payload_json, "temperature", "1e309"); invalid.push_back(bad);
    bad = submit; NumberField(bad.header.payload_json, "temperature", "1e-300"); invalid.push_back(bad);
    bad = submit; NumberField(bad.header.payload_json, "topP", "0"); invalid.push_back(bad);
    bad = submit; NumberField(bad.header.payload_json, "topK", "-1"); invalid.push_back(bad);
    bad = submit; NumberField(bad.header.payload_json, "topK", "2147483648"); invalid.push_back(bad);
    bad = submit; NumberField(bad.header.payload_json, "samplingSeed", "42.0"); invalid.push_back(bad);
    bad = submit; NumberField(bad.header.payload_json, "repetitionPenalty", "0"); invalid.push_back(bad);
    bad = submit; NumberField(bad.header.payload_json, "maxNewTokens", "0"); invalid.push_back(bad);
    bad = submit; NumberField(bad.header.payload_json, "maxNewTokens", "1025"); invalid.push_back(bad);
    bad = submit; NumberField(bad.header.payload_json, "prefillChunkSize", "513"); invalid.push_back(bad);
    bad = submit; NumberField(bad.header.payload_json, "promptTokenCount", "18446744073709551615"); invalid.push_back(bad);
    bad = submit; Replace(bad.header.payload_json, "\"i32le\"", "\"native_int\""); invalid.push_back(bad);
    bad = submit; bad.attachment.push_back(std::byte{0}); invalid.push_back(bad);
    bad = submit; bad.header.payload_json.insert(bad.header.payload_json.size() - 1, std::string(",\"e\\u0070och\":\"") + Epoch + "\""); invalid.push_back(bad);
    for (const auto& frame : invalid) RejectCommand(frame);

    bad = Make(Command(wire::Method::WorkerConfigure));
    Replace(bad.header.payload_json, "\"limits\":{", "\"limits\":{\"maxSessions\":1,");
    RejectCommand(bad);
    bad = Make(Command(wire::Method::WorkerQuery));
    bad.attachment = {std::byte{0}};
    RejectCommand(bad);
    bad = Make(Command(wire::Method::ModelLoad));
    Replace(bad.header.payload_json, "\"qwen3_nf4\"", "\"future_nf4\"");
    RejectCommand(bad);
    bad = Make(Command(wire::Method::ModelLoad));
    Replace(bad.header.payload_json, "\"bf16\"", "\"fp8\"");
    RejectCommand(bad);

    wire::ValidationError error;
    Frame unchanged;
    unchanged.header.request_id = 999;
    auto invalidInput = Command(wire::Method::RequestSubmit);
    invalidInput.generation.temperature = (std::numeric_limits<double>::quiet_NaN)();
    Expect(!wire::make_command(invalidInput, unchanged, error) && unchanged.header.request_id == 999, "NaN input was encoded or changed output");
    invalidInput = Command(wire::Method::RequestSubmit);
    invalidInput.promptIds.clear();
    Expect(!wire::make_command(invalidInput, unchanged, error), "Empty prompt was accepted");
    invalidInput = Command(wire::Method::RequestSubmit);
    invalidInput.stopTokenIds.resize(65);
    Expect(!wire::make_command(invalidInput, unchanged, error), "Stop token count limit was ignored");
}

wire::Response Result(wire::Method method) {
    wire::Response response;
    response.method = method;
    response.rpcId = 77;
    response.epoch = Epoch;
    response.worker.domainStatus = wire::DomainStatus::Ready;
    response.worker.accepting = true;
    response.capabilities = wire::default_capabilities();
    response.model = {3, 1000, 512, 1024, {"qwen3_nf4_bf16_kv"}};
    response.session = {7, 3, 512, 1024, false, uint32_t{0}};
    response.cancelRequested = true;
    response.summary.status = wire::RequestStatus::Decode;
    response.request.summary.status = wire::RequestStatus::Decode;
    response.request.summary.promptTokensProcessed = 3;
    response.request.summary.generatedTokens = 2;
    response.request.bufferedTokens = 1;
    response.request.lastSentSequence = 1;
    response.acknowledgedSequence = 3;
    if (method == wire::Method::WorkerShutdown) {
        response.worker.domainStatus = wire::DomainStatus::Closed;
        response.lastEventOrdinal = 16;
    }
    return response;
}

void TestDecodeBatchingConfiguration() {
    wire::ValidationError error;
    auto command = Command(wire::Method::WorkerConfigure);
    const auto ordinary = Make(command);
    wire::Command parsed;
    Success(wire::parse_command(ordinary, parsed, error), error);
    Expect(parsed.maxDecodeBatchSize == 1 &&
           ordinary.header.payload_json.find("max_decode_batch_size") == std::string::npos,
           "Default configure changed the legacy payload");
    command.requiredCapabilities.emplace_back("continuous_batching_v1");
    for (uint32_t size = 1; size <= 4; ++size) {
        command.maxDecodeBatchSize = size;
        const auto frame = Make(command);
        Success(wire::parse_command(frame, parsed, error), error);
        Expect(parsed.maxDecodeBatchSize == size, "Decode batch bound changed in transit");
    }
    Frame frame;
    for (uint32_t size : {uint32_t{0}, uint32_t{5}, (std::numeric_limits<uint32_t>::max)()}) {
        command.maxDecodeBatchSize = size;
        Expect(!wire::make_command(command, frame, error) && error.code == wire::ErrorCode::InvalidArgument,
               "Out-of-range decode batch bound was encoded");
    }
    command.maxDecodeBatchSize = 3;
    command.limits.maxActiveRequests = 2;
    Expect(!wire::make_command(command, frame, error), "Decode batch bound exceeded active admission");
    command.maxDecodeBatchSize = 2;
    Success(wire::make_command(command, frame, error), error);
    command.requiredCapabilities = wire::default_capabilities();
    Expect(!wire::make_command(command, frame, error) && error.code == wire::ErrorCode::UnsupportedCapability,
           "Decode batching was enabled without a required capability");
    command.requiredCapabilities.emplace_back("continuous_batching_v1");
    const auto valid = Make(command);
    for (const char* value : {"0", "3", "5", "-1", "1.5", "true", "null", "\"2\"", "4294967296"}) {
        auto bad = valid;
        NumberField(bad.header.payload_json, "max_decode_batch_size", value);
        parsed.maxDecodeBatchSize = 99;
        Expect(!wire::parse_command(bad, parsed, error) && parsed.maxDecodeBatchSize == 99,
               "Invalid configure batch field was accepted or partly changed output");
    }
    auto duplicate = valid;
    Replace(duplicate.header.payload_json, "\"max_decode_batch_size\":2",
            "\"max_decode_batch_size\":2,\"max_decode_batch_size\":2");
    RejectCommand(duplicate);
    auto unrelated = Make(Command(wire::Method::WorkerQuery));
    unrelated.header.payload_json.insert(1, "\"max_decode_batch_size\":2,");
    RejectCommand(unrelated);

    auto result = Result(wire::Method::WorkerConfigure);
    Success(wire::make_response(result, frame, error), error);
    wire::Response response;
    Success(wire::parse_response(frame, response, error), error);
    Expect(response.maxDecodeBatchSize == 1 &&
           frame.header.payload_json.find("max_decode_batch_size") == std::string::npos,
           "Default configure response changed its legacy payload");
    result.maxDecodeBatchSize = 2;
    Expect(!wire::make_response(result, frame, error), "Batch response omitted its capability");
    result.capabilities.emplace_back("continuous_batching_v1");
    Success(wire::make_response(result, frame, error), error);
    Success(wire::parse_response(frame, response, error), error);
    Expect(response.maxDecodeBatchSize == 2, "Configure acknowledgement lost its batch bound");
    for (const char* value : {"0", "5", "-1", "true", "null", "\"2\"", "4294967296"}) {
        auto bad = frame;
        NumberField(bad.header.payload_json, "max_decode_batch_size", value);
        response.maxDecodeBatchSize = 99;
        Expect(!wire::parse_response(bad, response, error) && response.maxDecodeBatchSize == 99 &&
               error.code == wire::ErrorCode::ProtocolError,
               "Invalid batch acknowledgement was accepted or partly changed output");
    }
    result.limits.maxActiveRequests = 1;
    Expect(!wire::make_response(result, frame, error), "Batch acknowledgement exceeded selected active limit");
}

void TestResponsesAndHandshake() {
    wire::ValidationError error;
    const wire::Method methods[]{wire::Method::WorkerConfigure, wire::Method::WorkerQuery, wire::Method::ModelLoad,
        wire::Method::SessionCreate, wire::Method::SessionQuery, wire::Method::RequestCancel,
        wire::Method::RequestQuery, wire::Method::RequestAck, wire::Method::HandleRelease, wire::Method::WorkerShutdown,
        wire::Method::SessionReset, wire::Method::SessionTruncate, wire::Method::WorkerSetLogLevel, wire::Method::WorkerMemoryQuery};
    for (const auto method : methods) {
        auto response = Result(method);
        Frame frame;
        Success(wire::make_response(response, frame, error), error);
        wire::Response parsed;
        Success(wire::parse_response(frame, parsed, error), error);
        Expect(parsed.method == method && parsed.rpcId == response.rpcId && parsed.epoch == Epoch, "Result identity changed");
        response.kind = wire::ResponseKind::Error;
        response.errorCode = wire::ErrorCode::InvalidArgument;
        response.message = "invalid input";
        Success(wire::make_response(response, frame, error), error);
        Success(wire::parse_response(frame, parsed, error), error);
        Expect(parsed.errorCode == response.errorCode && parsed.message == response.message, "Error response changed");
    }
    auto busy = Result(wire::Method::SessionQuery);
    busy.session.busy = true;
    busy.session.contextLength.reset();
    Frame frame;
    Success(wire::make_response(busy, frame, error), error);
    busy.session.contextLength = 10;
    Expect(!wire::make_response(busy, frame, error), "Busy session reported an exact length");

    for (const char* backend : {"qwen3_nf4", "qwen35_nf4", "qwen35_bf16", "qwen4exp_nf4"}) {
        auto result = Result(wire::Method::ModelLoad);
        result.model.capabilities = {std::string(wire::model_capability(backend))};
        Success(wire::make_response(result, frame, error), error);
        wire::Response parsed;
        Success(wire::parse_response(frame, parsed, error), error);
        Expect(parsed.model.capabilities == result.model.capabilities,
               "Model result changed the selected backend capability");
    }

    wire::Hello hello;
    hello.epoch = Epoch;
    hello.buildId = "test-build";
    hello.executable = "C:\\runtime\\AilaWorker.exe";
    hello.runtimeDirectory = "C:\\runtime";
    hello.currentDirectory = "C:\\runtime";
    hello.path = "C:\\runtime;C:\\Windows\\System32";
    hello.capabilities = wire::supported_capabilities();
    Success(wire::make_hello(hello, frame, error), error);
    std::vector<std::byte> bytes;
    Success(wire::encode_frame(frame, bytes, error, wire::bootstrap_bounds()), error);
    Frame received;
    Success(wire::decode_frame(bytes, received, error, wire::bootstrap_bounds()), error);
    wire::Hello parsed;
    Success(wire::parse_hello(received, parsed, error), error);
    Expect(parsed.path == hello.path && parsed.capabilities == hello.capabilities, "Handshake metadata changed");
    auto malformed = received;
    Replace(malformed.header.payload_json, "\"hardLimits\":{", "\"hardLimits\":{\"unknown\":1,");
    Expect(!wire::parse_hello(malformed, parsed, error), "Unknown hard limit accepted");
    hello.supportedProtocols = {1, 2};
    Expect(!wire::make_hello(hello, frame, error), "v2 launch mode advertised a live protocol switch");
}

void TestWorkerLogControl() {
    wire::ValidationError error;
    for (uint32_t level = 0; level <= 3; ++level) {
        auto command = Command(wire::Method::WorkerSetLogLevel);
        command.logLevel = level;
        wire::Command parsed;
        Success(wire::parse_command(Make(command), parsed, error), error);
        Expect(parsed.logLevel == level, "Log control level changed");
        auto response = Result(wire::Method::WorkerSetLogLevel);
        response.logLevel = level;
        Frame frame;
        Success(wire::make_response(response, frame, error), error);
        wire::Response decoded;
        Success(wire::parse_response(frame, decoded, error), error);
        Expect(decoded.logLevel == level, "Log control response changed");
    }
    for (const char* value : {"-1", "4", "1.5", "null", "true", "4294967296"}) {
        auto frame = Make(Command(wire::Method::WorkerSetLogLevel));
        NumberField(frame.header.payload_json, "level", value);
        RejectCommand(frame);
    }
    auto missing = Make(Command(wire::Method::WorkerSetLogLevel));
    Replace(missing.header.payload_json, ",\"level\":1", "");
    RejectCommand(missing);
    auto response = Result(wire::Method::WorkerSetLogLevel);
    response.logLevel = 4;
    Frame output;
    Expect(!wire::make_response(response, output, error), "Invalid applied log level accepted");
}

void TestAcceptedAndRejected() {
    wire::ValidationError error;
    auto accepted = Final();
    accepted.kind = wire::ResponseKind::Accepted;
    accepted.phase = wire::SubmissionPhase::Accepted;
    Frame frame;
    Success(wire::make_response(accepted, frame, error), error);
    wire::Response parsed;
    Success(wire::parse_response(frame, parsed, error), error);
    Expect(parsed.phase == wire::SubmissionPhase::Accepted && parsed.identity.inferenceId == 21, "Accepted response changed");
    auto bad = frame;
    NumberField(bad.header.payload_json, "inferenceId", "22");
    Expect(!wire::parse_response(bad, parsed, error), "Accepted identity/header mismatch accepted");
    bad = frame;
    bad.header.kind = "result";
    Expect(!wire::parse_response(bad, parsed, error), "Accepted phase with result kind accepted");

    auto rejected = Final();
    rejected.kind = wire::ResponseKind::Error;
    rejected.phase = wire::SubmissionPhase::Rejected;
    rejected.errorCode = wire::ErrorCode::Busy;
    rejected.message = "Session is busy";
    Success(wire::make_response(rejected, frame, error), error);
    Success(wire::parse_response(frame, parsed, error), error);
    Expect(parsed.phase == wire::SubmissionPhase::Rejected && parsed.identity.inferenceId == rejected.rpcId, "Rejected submit mapping lost");
    NumberField(frame.header.payload_json, "eventCount", "1");
    Expect(!wire::parse_response(frame, parsed, error), "Rejected request promised a stream event");
}

void TestTerminalAndFinalAgreement() {
    wire::ValidationError error;
    Frame finalFrame;
    Frame terminalFrame;
    Success(wire::make_response(Final(), finalFrame, error), error);
    Success(wire::make_event(Terminal(), terminalFrame, error), error);
    wire::Response final;
    wire::Event terminal;
    // Final first is allowed: the connection retains it until the other pipe arrives.
    Success(wire::parse_response(finalFrame, final, error), error);
    Success(wire::parse_event(terminalFrame, terminal, error), error);
    Success(wire::match_terminal_final(terminal, final, error), error);
    auto mismatch = final;
    mismatch.identity.modelId = 4;
    Expect(!wire::match_terminal_final(terminal, mismatch, error), "Cross-model stream corruption accepted");
    mismatch = final; mismatch.summary.promptTokensProcessed = 2;
    Expect(!wire::match_terminal_final(terminal, mismatch, error), "Terminal/final progress mismatch accepted");
    mismatch = final; mismatch.epoch[0] = 'f';
    Expect(!wire::match_terminal_final(terminal, mismatch, error), "Old epoch final accepted");
    mismatch = final; mismatch.kind = wire::ResponseKind::Error;
    Expect(!wire::match_terminal_final(terminal, mismatch, error), "Completed status with error kind accepted");
    auto wrongTerminal = terminal; wrongTerminal.sequence = 4;
    Expect(!wire::match_terminal_final(wrongTerminal, final, error), "Missing terminal sequence accepted");
    wrongTerminal = terminal; wrongTerminal.kind = wire::EventKind::Token;
    Expect(!wire::match_terminal_final(wrongTerminal, final, error), "Token was accepted as terminal");
    NumberField(terminalFrame.header.payload_json, "eventCount", "6");
    Expect(!wire::parse_event(terminalFrame, terminal, error), "Inconsistent terminal count accepted");
}

void TestEveryTerminalReasonAndEscaping() {
    wire::ValidationError error;
    std::vector<wire::Summary> summaries;
    summaries.push_back(CompleteSummary());
    auto summary = CompleteSummary(); summary.endReason = wire::EndReason::StopToken; summaries.push_back(summary);
    summary = CompleteSummary(); summary.status = wire::RequestStatus::Cancelled; summary.endReason = wire::EndReason::Cancelled; summaries.push_back(summary);
    summary.endReason = wire::EndReason::OutputBackpressure; summary.errorCode = wire::ErrorCode::ResourceLimit; summary.message = "buffer full"; summaries.push_back(summary);
    summary.endReason = wire::EndReason::SchedulerStopped; summary.errorCode = wire::ErrorCode::SchedulerStopped; summary.message = "shutdown"; summaries.push_back(summary);
    summary.status = wire::RequestStatus::Failed; summary.endReason = wire::EndReason::RuntimeFault; summary.errorCode = wire::ErrorCode::RuntimeFault; summary.message = "device failed"; summaries.push_back(summary);
    summary.endReason = wire::EndReason::Failed; summary.errorCode = wire::ErrorCode::RequestFailed; summary.message = "request failed"; summaries.push_back(summary);
    for (const auto& value : summaries) {
        Frame frame;
        Success(wire::make_event(Terminal(value), frame, error), error);
        wire::Event event;
        Success(wire::parse_event(frame, event, error), error);
        Frame response;
        Success(wire::make_response(Final(value), response, error), error);
        wire::Response parsed;
        Success(wire::parse_response(response, parsed, error), error);
        Success(wire::match_terminal_final(event, parsed, error), error);
    }
    summary = summaries.back();
    summary.message.assign(1024, '\0');
    Frame escaped;
    Success(wire::make_event(Terminal(summary), escaped, error), error);
    std::vector<std::byte> bytes;
    Success(wire::encode_frame(escaped, bytes, error, wire::frame_bounds(wire::Channel::Event)), error);
    Expect(bytes.size() <= 8192, "Worst-case JSON error escaping exceeded the reserved terminal size");
    wire::Event parsed;
    Success(wire::parse_event(escaped, parsed, error), error);
    Expect(parsed.summary.message == summary.message, "Control-character error message changed");
    summary.message.push_back('\0');
    Expect(!wire::make_event(Terminal(summary), escaped, error), "Error text exceeded 1024 raw bytes");
    summary.message = std::string("\xc0\xaf", 2);
    Expect(!wire::make_event(Terminal(summary), escaped, error), "Malformed UTF-8 error message was serialized");
    summary = CompleteSummary(); summary.errorCode = wire::ErrorCode::RuntimeFault;
    Expect(!wire::validate_summary(summary, error), "Successful completion contained a runtime error");
}

void TestTokenLogDomainAndQuery() {
    wire::ValidationError error;
    wire::Event token;
    token.epoch = Epoch;
    token.identity = {21, 3, 7};
    token.eventOrdinal = 5;
    token.sequence = 1;
    token.tokenId = 42;
    token.summary.status = wire::RequestStatus::Decode;
    token.summary.promptTokensProcessed = 3;
    token.summary.generatedTokens = 1;
    Frame frame;
    Success(wire::make_event(token, frame, error), error);
    wire::Event parsed;
    Success(wire::parse_event(frame, parsed, error), error);
    Expect(parsed.tokenId == 42 && parsed.sequence == 1, "Token event changed");
    auto malformed = frame;
    NumberField(malformed.header.payload_json, "generatedTokens", "2");
    Expect(!wire::parse_event(malformed, parsed, error), "Token used a later snapshot's total instead of its own sequence");
    malformed = frame;
    malformed.header.payload_json.insert(malformed.header.payload_json.size() - 1, ",\"eventCount\":2");
    Expect(!wire::parse_event(malformed, parsed, error), "Token event contained terminal metadata");

    wire::Event log;
    log.kind = wire::EventKind::Log;
    log.epoch = Epoch;
    log.eventOrdinal = 6;
    log.logLevel = 1;
    log.message = "ready\n\"quoted\"\\text";
    Success(wire::make_event(log, frame, error), error);
    Expect(frame.header.request_id == 0, "Log consumed a request event identity");
    Success(wire::parse_event(frame, parsed, error), error);
    Expect(parsed.message == log.message, "Log escaping changed text");
    log.kind = wire::EventKind::DomainFault;
    log.errorCode = wire::ErrorCode::RuntimeFault;
    Success(wire::make_event(log, frame, error), error);
    log.errorCode = wire::ErrorCode::InvalidArgument;
    Expect(!wire::make_event(log, frame, error), "Local argument failure became a domain fault event");

    auto query = Result(wire::Method::RequestQuery);
    query.request.summary = CompleteSummary();
    query.request.completionReady = true;
    query.request.bufferedTokens = 0;
    query.request.lastSentSequence = 5;
    query.request.lastAcknowledgedSequence = 4;
    Success(wire::make_response(query, frame, error), error);
    wire::Response response;
    Success(wire::parse_response(frame, response, error), error);
    Expect(response.request.summary.eventCount == 5 && response.request.completionReady, "Completed unread query lost its terminal metadata");
    query.request.lastAcknowledgedSequence = 6;
    Expect(!wire::make_response(query, frame, error), "Query acknowledged events not sent");
}

void TestThinkingBudgetExtension() {
    auto command = Command(wire::Method::RequestSubmit);
    auto plain = Make(command);
    Expect(plain.header.payload_json.find("thinkingBudget") == std::string::npos,
           "Default submit changed its old-peer schema");
    command.thinkingBudget = wire::ThinkingBudgetOptions{2, 10, 11, true, {11, 3}};
    const auto frame = Make(command);
    wire::Command parsed;
    wire::ValidationError error;
    Success(wire::parse_command(frame, parsed, error), error);
    Expect(parsed.thinkingBudget && parsed.thinkingBudget->maxTokens == 2 &&
               parsed.thinkingBudget->openTokenId == 10 && parsed.thinkingBudget->closeTokenId == 11 &&
               parsed.thinkingBudget->promptOpensThink && parsed.thinkingBudget->closeTokenIds == std::vector<int32_t>({11, 3}),
           "Thinking token metadata changed in transit");
    for (const char* number : {"-1", "1.5", "4294967296", "true", "null"}) {
        auto bad = frame;
        NumberField(bad.header.payload_json, "maxTokens", number);
        RejectCommand(bad);
    }
    for (const auto& edit : std::vector<std::pair<std::string, std::string>>{
             {"\"openTokenId\":10", "\"openTokenId\":2147483648"},
             {"\"closeTokenId\":11", "\"closeTokenId\":10"},
             {"\"promptOpensThink\":true", "\"promptOpensThink\":1"},
             {"\"closeTokenIds\":[11,3]", "\"closeTokenIds\":[]"},
             {"\"closeTokenIds\":[11,3]", "\"closeTokenIds\":[3,11]"},
             {"\"closeTokenIds\":[11,3]", "\"closeTokenIds\":[11,10]"},
             {"\"closeTokenIds\":[11,3]", "\"closeTokenIds\":[11,9]"},
             {"\"closeTokenIds\":[11,3]", "\"closeTokenIds\":[11,-1]"},
             {"\"maxTokens\":2", "\"maxTokens\":2,\"maxTokens\":1"},
             {"\"maxTokens\":2", "\"maxTokens\":2,\"extra\":true"}}) {
        auto bad = frame;
        Replace(bad.header.payload_json, edit.first, edit.second);
        RejectCommand(bad);
    }
    command.thinkingBudget->closeTokenIds.resize(17, 3);
    Expect(!wire::make_command(command, plain, error), "Unbounded thinking close sequence was encoded");
    const auto defaults = wire::default_capabilities(), supported = wire::supported_capabilities();
    Expect(std::find(defaults.begin(), defaults.end(), "thinking_budget_v1") == defaults.end() &&
               std::find(supported.begin(), supported.end(), "thinking_budget_v1") != supported.end(),
           "Thinking capability changed required defaults or was not offered");
}

void TestSessionContextExtension() {
    const auto defaults = wire::default_capabilities(), supported = wire::supported_capabilities();
    Expect(std::find(defaults.begin(), defaults.end(), "session_context_v1") == defaults.end() &&
           std::find(supported.begin(), supported.end(), "session_context_v1") != supported.end(),
           "Context capability changed required defaults or was not offered");
    auto command = Command(wire::Method::SessionTruncate);
    command.contextLength = 129;
    auto frame = Make(command);
    wire::Command parsed;
    wire::ValidationError error;
    Success(wire::parse_command(frame, parsed, error), error);
    Expect(parsed.sessionId == command.sessionId && parsed.contextLength == 129,
           "Truncate identity/target changed in transit");
    for (const char* value : {"-1", "1.5", "2147483648", "4294967296", "true", "null"}) {
        auto bad = frame;
        NumberField(bad.header.payload_json, "contextLength", value);
        RejectCommand(bad);
    }
    for (const char* field : {"\"contextLength\":129,\"contextLength\":129", "\"contextLength\":129,\"extra\":0"}) {
        auto bad = frame;
        Replace(bad.header.payload_json, "\"contextLength\":129", field);
        RejectCommand(bad);
    }
    command.contextLength = UINT32_MAX;
    Expect(!wire::make_command(command, frame, error), "Unrepresentable truncate length was encoded");
    auto reset = Make(Command(wire::Method::SessionReset));
    Replace(reset.header.payload_json, "\"sessionId\":7", "\"sessionId\":7,\"contextLength\":0");
    RejectCommand(reset);
    auto query = Make(Command(wire::Method::SessionQuery));
    Expect(query.header.payload_json == Make(Command(wire::Method::SessionReset)).header.payload_json,
           "Reset changed the identity-only payload shape");
    for (const auto method : {wire::Method::SessionReset, wire::Method::SessionTruncate}) {
        auto response = Result(method);
        Success(wire::make_response(response, frame, error), error);
        wire::Response received;
        Success(wire::parse_response(frame, received, error), error);
        Expect(received.session.contextLength == 0 && !received.session.busy, "Completed mutation lost actual length");
        auto bad = frame;
        NumberField(bad.header.payload_json, "contextLength", "null");
        Expect(!wire::parse_response(bad, received, error), "Completed mutation allowed unknown length");
        response.session.busy = true;
        response.session.contextLength.reset();
        Expect(!wire::make_response(response, frame, error), "Completed mutation claimed Busy");
        response.session.busy = false;
        response.session.contextLength = 5;
        if (method == wire::Method::SessionReset)
            Expect(!wire::make_response(response, frame, error), "Reset returned a nonzero length");
        else Success(wire::make_response(response, frame, error), error);
    }
}

void TestAllocationLimitsExtension() {
    auto command = Command(wire::Method::ModelLoad);
    Expect(Make(command).header.payload_json.find("allocationLimits") == std::string::npos,
           "Legacy load schema acquired an unrequested extension");
    command.allocationLimits = wire::AllocationLimits{1234567890123ULL, 4096, 20201472};
    auto frame = Make(command);
    wire::Command parsed;
    wire::ValidationError error;
    Success(wire::parse_command(frame, parsed, error), error);
    Expect(parsed.allocationLimits && *parsed.allocationLimits == *command.allocationLimits,
           "Allocation byte limits did not survive a round trip");
    for (const char* invalid : {"-1", "1.5", "null", "18446744073709551616"}) {
        auto bad = frame;
        NumberField(bad.header.payload_json, "maxDeviceBytes", invalid);
        RejectCommand(bad);
    }
    auto bad = frame;
    Replace(bad.header.payload_json, "\"maxPinnedHostBytes\":4096", "\"maxPinnedHostBytes\":4096,\"maxPinnedHostBytes\":4096");
    RejectCommand(bad);
    bad = frame;
    Replace(bad.header.payload_json, "\"maxPinnedHostBytes\":4096", "\"unknownBudget\":4096");
    RejectCommand(bad);

    auto result = Result(wire::Method::ModelLoad);
    result.model.capabilities.emplace_back("allocation_limits_v1");
    result.model.memory = wire::ModelMemoryInfo{*command.allocationLimits, 8192, 1024};
    Success(wire::make_response(result, frame, error), error);
    wire::Response received;
    Success(wire::parse_response(frame, received, error), error);
    Expect(received.model.memory && received.model.memory->limits == *command.allocationLimits &&
           received.model.memory->residentDeviceBytes == 8192 && received.model.memory->sessionHostReservationBytes == 1024,
           "Model allocation acknowledgement lost its accounting");
    result.model.memory->limits.maxDeviceBytes = 8191;
    Expect(!wire::make_response(result, frame, error), "Over-budget model residency was accepted");
}

void TestRuntimeMemory() {
    auto command = Make(Command(wire::Method::WorkerMemoryQuery));
    Expect(command.header.payload_json == std::string("{\"epoch\":\"") + Epoch + "\"}",
           "Memory observation acquired mutation fields");
    auto badCommand = command;
    Replace(badCommand.header.payload_json, "}", ",\"synchronize\":true}");
    RejectCommand(badCommand);
    wire::ValidationError error;
    wire::Response parsed;
    Frame frame;
    auto result = Result(wire::Method::WorkerMemoryQuery);
    for (const auto state : {wire::RuntimeMemoryState::Uninitialized, wire::RuntimeMemoryState::Busy}) {
        result.runtimeMemory = {state, std::nullopt};
        Success(wire::make_response(result, frame, error), error);
        Success(wire::parse_response(frame, parsed, error), error);
        Expect(parsed.runtimeMemory.state == state && !parsed.runtimeMemory.usage, "Unknown usage became zero usage");
        Replace(frame.header.payload_json, "\"memory\":null", "\"memory\":{}");
        Expect(!wire::parse_response(frame, parsed, error), "Unavailable state accepted memory payload");
    }
    const wire::RuntimeMemoryUsage usage{4096, (uint64_t{1} << 53) + 1, 16, 128, 4, 1, 8192, 64};
    result.runtimeMemory = {wire::RuntimeMemoryState::Ready, usage};
    Success(wire::make_response(result, frame, error), error);
    Success(wire::parse_response(frame, parsed, error), error);
    Expect(parsed.runtimeMemory.usage && parsed.runtimeMemory.usage->peakDeviceBytes == usage.peakDeviceBytes &&
           parsed.runtimeMemory.usage->deviceBytes == 4096 && parsed.runtimeMemory.usage->pinnedHostBytes == 16,
           "Memory bytes lost precision or identity");
    const auto good = frame;
    for (const char* key : {"deviceBytes", "peakDeviceBytes", "pinnedHostBytes", "peakPinnedHostBytes",
                            "deviceAllocations", "pinnedHostAllocations", "deviceLimitBytes", "pinnedHostLimitBytes"}) {
        for (const char* invalid : {"-1", "1.5", "true", "null", "18446744073709551616"}) {
            auto bad = good;
            NumberField(bad.header.payload_json, key, invalid);
            parsed.rpcId = 999;
            Expect(!wire::parse_response(bad, parsed, error) && parsed.rpcId == 999,
                   "Invalid memory count was accepted or changed parse output");
        }
    }
    for (int invalid = 0; invalid != 8; ++invalid) {
        auto bad = good;
        if (invalid == 0) NumberField(bad.header.payload_json, "deviceBytes", "8193");
        if (invalid == 1) NumberField(bad.header.payload_json, "peakDeviceBytes", "4095");
        if (invalid == 2) NumberField(bad.header.payload_json, "deviceAllocations", "0");
        if (invalid == 3) NumberField(bad.header.payload_json, "pinnedHostAllocations", "17");
        if (invalid == 4) NumberField(bad.header.payload_json, "pinnedHostLimitBytes", "15");
        if (invalid == 5) Replace(bad.header.payload_json, "\"state\":\"ready\"", "\"state\":\"cached\"");
        if (invalid == 6) Replace(bad.header.payload_json, "\"deviceBytes\":4096", "\"deviceBytes\":4096,\"extra\":0");
        if (invalid == 7) Replace(bad.header.payload_json, "\"deviceBytes\":4096", "\"deviceBytes\":4096,\"deviceBytes\":4096");
        Expect(!wire::parse_response(bad, parsed, error), "Inconsistent memory observation was accepted");
    }
    result.runtimeMemory.usage = wire::RuntimeMemoryUsage{0, 4096, 0, 128, 0, 0, 1, 1};
    Success(wire::make_response(result, frame, error), error); // Peak may exceed a later lower limit.
    result.runtimeMemory.state = wire::RuntimeMemoryState::Busy;
    Expect(!wire::make_response(result, frame, error), "Busy response published an old snapshot");
}

} // namespace

int main() {
    const std::pair<const char*, void (*)()> tests[]{
        {"versions and stable names", TestVersionsAndNames},
        {"limits and capability negotiation", TestLimitsAndNegotiation},
        {"explicit extended generation, reservation and terminal bounds", TestExtendedGenerationLimits},
        {"optional extended context ceiling, combination and long attachment", TestExtendedContextLimits},
        {"decode batching opt-in, strict schema and exact bounds", TestDecodeBatchingConfiguration},
        {"prefix and strict envelope", TestPrefixAndEnvelope},
        {"little endian token attachment", TestLittleEndianTokens},
        {"all supported commands", TestAllCommands},
        {"model backends and capability compatibility", TestModelCapabilities},
        {"optional prefill-only mode and strict schema", TestPrefillOnlyExtension},
        {"single-close and host sampler request policy schema", TestRequestPolicyExtensions},
        {"optional allocation limits and acknowledgement", TestAllocationLimitsExtension},
        {"optional thinking budget and strict token metadata", TestThinkingBudgetExtension},
        {"optional Session reset/truncate and strict actual length", TestSessionContextExtension},
        {"strict command payloads", TestStrictCommandPayloads},
        {"result schemas and handshake", TestResponsesAndHandshake},
        {"accepted and rejected submit", TestAcceptedAndRejected},
        {"terminal and final agreement", TestTerminalAndFinalAgreement},
        {"terminal reasons and escaped bounds", TestEveryTerminalReasonAndEscaping},
        {"token log domain and query events", TestTokenLogDomainAndQuery},
        {"optional Worker log level and strict schema", TestWorkerLogControl},
        {"optional runtime memory states, exact integers and invariants", TestRuntimeMemory},
    };
    for (const auto& test : tests) {
        try {
            test.second();
            std::cout << "PASS " << test.first << '\n';
        } catch (const std::exception& error) {
            std::cerr << "FAIL " << test.first << ": " << error.what() << '\n';
            return 1;
        }
    }
    return 0;
}
