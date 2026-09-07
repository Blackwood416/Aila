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
        wire::Method::RequestQuery, wire::Method::RequestAck, wire::Method::HandleRelease, wire::Method::WorkerShutdown};
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
    Replace(bad.header.payload_json, "\"qwen3_nf4\"", "\"qwen35_nf4\"");
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

void TestResponsesAndHandshake() {
    wire::ValidationError error;
    const wire::Method methods[]{wire::Method::WorkerConfigure, wire::Method::WorkerQuery, wire::Method::ModelLoad,
        wire::Method::SessionCreate, wire::Method::SessionQuery, wire::Method::RequestCancel,
        wire::Method::RequestQuery, wire::Method::RequestAck, wire::Method::HandleRelease, wire::Method::WorkerShutdown};
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

    wire::Hello hello;
    hello.epoch = Epoch;
    hello.buildId = "test-build";
    hello.executable = "C:\\runtime\\AilaWorker.exe";
    hello.runtimeDirectory = "C:\\runtime";
    hello.currentDirectory = "C:\\runtime";
    hello.path = "C:\\runtime;C:\\Windows\\System32";
    hello.capabilities = wire::default_capabilities();
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

} // namespace

int main() {
    const std::pair<const char*, void (*)()> tests[]{
        {"versions and stable names", TestVersionsAndNames},
        {"limits and capability negotiation", TestLimitsAndNegotiation},
        {"prefix and strict envelope", TestPrefixAndEnvelope},
        {"little endian token attachment", TestLittleEndianTokens},
        {"all eleven commands", TestAllCommands},
        {"strict command payloads", TestStrictCommandPayloads},
        {"result schemas and handshake", TestResponsesAndHandshake},
        {"accepted and rejected submit", TestAcceptedAndRejected},
        {"terminal and final agreement", TestTerminalAndFinalAgreement},
        {"terminal reasons and escaped bounds", TestEveryTerminalReasonAndEscaping},
        {"token log domain and query events", TestTokenLogDomainAndQuery},
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
