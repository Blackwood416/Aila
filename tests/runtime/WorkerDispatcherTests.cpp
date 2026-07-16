#include "worker/WorkerDispatcher.hpp"

#include "aila_api.h"
#include "simdjson.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using aila::ipc::Frame;
using aila::worker::WorkerDispatcher;
using aila::worker::WorkerEngineApi;

[[noreturn]] void fail(const char* test_name, const std::string& message) {
    throw std::runtime_error(std::string("FAILED: ") + test_name + ": " + message);
}

void expect(bool condition, const char* test_name, const std::string& message) {
    if (!condition) {
        fail(test_name, message);
    }
}

Frame request(uint64_t id, std::string method, std::string payload = "{}") {
    Frame frame;
    frame.header.request_id = id;
    frame.header.kind = "request";
    frame.header.method = std::move(method);
    frame.header.payload_json = std::move(payload);
    return frame;
}

simdjson::dom::element parse_payload(
    const Frame& frame,
    simdjson::dom::parser& parser,
    const char* test_name) {
    simdjson::dom::element root;
    const simdjson::error_code error = parser.parse(frame.header.payload_json).get(root);
    expect(error == simdjson::SUCCESS, test_name, "response payload is not valid JSON");
    return root;
}

int64_t payload_integer(const Frame& frame, std::string_view field, const char* test_name) {
    simdjson::dom::parser parser;
    const simdjson::dom::element root = parse_payload(frame, parser, test_name);
    int64_t value = 0;
    expect(root[field].get_int64().get(value) == simdjson::SUCCESS,
           test_name,
           std::string("missing integer field: ") + std::string(field));
    return value;
}

std::string payload_string(const Frame& frame, std::string_view field, const char* test_name) {
    simdjson::dom::parser parser;
    const simdjson::dom::element root = parse_payload(frame, parser, test_name);
    std::string_view value;
    expect(root[field].get_string().get(value) == simdjson::SUCCESS,
           test_name,
           std::string("missing string field: ") + std::string(field));
    return std::string(value);
}

bool payload_boolean(const Frame& frame, std::string_view field, const char* test_name) {
    simdjson::dom::parser parser;
    const simdjson::dom::element root = parse_payload(frame, parser, test_name);
    bool value = false;
    expect(root[field].get_bool().get(value) == simdjson::SUCCESS,
           test_name,
           std::string("missing boolean field: ") + std::string(field));
    return value;
}

void expect_response_identity(
    const Frame& response,
    const Frame& source,
    const char* expected_kind,
    const char* test_name) {
    expect(response.header.protocol == source.header.protocol, test_name, "protocol was not echoed");
    expect(response.header.abi == source.header.abi, test_name, "ABI was not echoed");
    expect(response.header.request_id == source.header.request_id, test_name, "request id was not echoed");
    expect(response.header.kind == expected_kind, test_name, "response kind is incorrect");
    expect(response.header.method == source.header.method, test_name, "method was not echoed");
    expect(response.attachment.empty(), test_name, "response unexpectedly contained an attachment");
}

struct EngineState {
    int init_calls = 0;
    int reset_calls = 0;
    int context_length = 37;
    int init_result = AILA_OK;
    int last_error_code = AILA_OK;
    std::string last_error_message;
    std::string model;
    int max_seq_len = 0;
    bool throw_on_context_length = false;
};

class FakeEngine final : public WorkerEngineApi {
public:
    FakeEngine(EngineState& state, int& destruction_count)
        : state_(state), destruction_count_(destruction_count) {}

    ~FakeEngine() override { ++destruction_count_; }

    int init(const std::string& model, int max_seq_len) override {
        ++state_.init_calls;
        state_.model = model;
        state_.max_seq_len = max_seq_len;
        return state_.init_result;
    }

    void reset_context() override { ++state_.reset_calls; }

    int context_length() const override {
        if (state_.throw_on_context_length) {
            throw std::runtime_error("synthetic context failure");
        }
        return state_.context_length;
    }

    int last_error_code() const override { return state_.last_error_code; }
    std::string last_error_message() const override { return state_.last_error_message; }

private:
    EngineState& state_;
    int& destruction_count_;
};

std::unique_ptr<WorkerEngineApi> fake_engine(EngineState& state, int& destruction_count) {
    return std::make_unique<FakeEngine>(state, destruction_count);
}

void test_init_forwards_utf8_model_and_sequence_length() {
    constexpr const char* name = "engine.init forwards UTF-8 model and maxSeqLen";
    EngineState state;
    int destructions = 0;
    WorkerDispatcher dispatcher(fake_engine(state, destructions));
    const Frame command = request(
        41,
        "engine.init",
        R"({"model":"C:\\模型\\qwen","maxSeqLen":8192})");

    bool should_shutdown = false;
    const Frame response = dispatcher.dispatch(command, should_shutdown);

    expect_response_identity(response, command, "result", name);
    expect(payload_boolean(response, "ok", name), name, "init result did not report success");
    expect(state.init_calls == 1, name, "engine init was not called exactly once");
    expect(state.model == "C:\\模型\\qwen", name, "UTF-8 model path changed");
    expect(state.max_seq_len == 8192, name, "maxSeqLen changed");
    expect(!should_shutdown, name, "init requested shutdown");
}

void test_context_reset_and_second_init_are_deterministic() {
    constexpr const char* name = "context, reset, and deterministic second init";
    EngineState state;
    int destructions = 0;
    WorkerDispatcher dispatcher(fake_engine(state, destructions));
    bool should_shutdown = false;

    const Frame first_init = request(50, "engine.init", R"({"model":"model-a","maxSeqLen":4096})");
    expect(dispatcher.dispatch(first_init, should_shutdown).header.kind == "result",
           name,
           "first init failed");

    const Frame context_command = request(51, "engine.context_length");
    const Frame context_response = dispatcher.dispatch(context_command, should_shutdown);
    expect_response_identity(context_response, context_command, "result", name);
    expect(payload_integer(context_response, "contextLength", name) == 37,
           name,
           "context length changed");

    const Frame reset_command = request(52, "engine.reset");
    const Frame reset_response = dispatcher.dispatch(reset_command, should_shutdown);
    expect_response_identity(reset_response, reset_command, "result", name);
    expect(payload_boolean(reset_response, "ok", name), name, "reset did not report success");
    expect(state.reset_calls == 1, name, "reset was not called exactly once");

    const Frame second_init = request(53, "engine.init", R"({"model":"model-b","maxSeqLen":2048})");
    const Frame second_response = dispatcher.dispatch(second_init, should_shutdown);
    expect_response_identity(second_response, second_init, "error", name);
    expect(payload_integer(second_response, "code", name) == AILA_ERR_INVALID_ARGUMENT,
           name,
           "second init returned the wrong code");
    expect(payload_string(second_response, "message", name).find("already initialized") != std::string::npos,
           name,
           "second init error was not useful");
    expect(state.init_calls == 1, name, "second init reached the engine");
}

void test_invalid_requests_return_structured_errors() {
    constexpr const char* name = "invalid requests return structured errors";
    EngineState state;
    int destructions = 0;
    WorkerDispatcher dispatcher(fake_engine(state, destructions));
    bool should_shutdown = false;

    const Frame malformed = request(60, "engine.init", "{");
    const Frame malformed_response = dispatcher.dispatch(malformed, should_shutdown);
    expect_response_identity(malformed_response, malformed, "error", name);
    expect(payload_integer(malformed_response, "code", name) == AILA_ERR_JSON_PARSE,
           name,
           "malformed JSON returned the wrong code");
    expect(!payload_string(malformed_response, "message", name).empty(),
           name,
           "malformed JSON returned an empty message");

    const Frame missing_model = request(61, "engine.init", R"({"maxSeqLen":4096})");
    const Frame missing_model_response = dispatcher.dispatch(missing_model, should_shutdown);
    expect_response_identity(missing_model_response, missing_model, "error", name);
    expect(payload_integer(missing_model_response, "code", name) == AILA_ERR_INVALID_ARGUMENT,
           name,
           "missing model returned the wrong code");
    expect(payload_string(missing_model_response, "message", name).find("model") != std::string::npos,
           name,
           "missing model message was not useful");

    const Frame bad_length = request(62, "engine.init", R"({"model":"m","maxSeqLen":0})");
    const Frame bad_length_response = dispatcher.dispatch(bad_length, should_shutdown);
    expect(payload_integer(bad_length_response, "code", name) == AILA_ERR_INVALID_ARGUMENT,
           name,
           "invalid maxSeqLen returned the wrong code");
    expect(payload_string(bad_length_response, "message", name).find("maxSeqLen") != std::string::npos,
           name,
           "invalid maxSeqLen message was not useful");

    Frame bad_kind = request(63, "ping");
    bad_kind.header.kind = "event";
    const Frame bad_kind_response = dispatcher.dispatch(bad_kind, should_shutdown);
    expect_response_identity(bad_kind_response, bad_kind, "error", name);
    expect(payload_integer(bad_kind_response, "code", name) == AILA_ERR_INVALID_ARGUMENT,
           name,
           "invalid kind returned the wrong code");

    const Frame unknown = request(64, "engine.not_real");
    const Frame unknown_response = dispatcher.dispatch(unknown, should_shutdown);
    expect_response_identity(unknown_response, unknown, "error", name);
    expect(payload_integer(unknown_response, "code", name) == AILA_ERR_INVALID_ARGUMENT,
           name,
           "unknown method returned the wrong code");
    expect(state.init_calls == 0, name, "invalid requests reached init");
}

void test_init_failure_propagates_engine_error() {
    constexpr const char* name = "init failure propagates engine last error";
    EngineState state;
    state.init_result = -1;
    state.last_error_code = AILA_ERR_RUNTIME;
    state.last_error_message = "GPU unavailable: synthetic";
    int destructions = 0;
    WorkerDispatcher dispatcher(fake_engine(state, destructions));
    const Frame command = request(70, "engine.init", R"({"model":"m","maxSeqLen":1024})");

    bool should_shutdown = false;
    const Frame response = dispatcher.dispatch(command, should_shutdown);

    expect_response_identity(response, command, "error", name);
    expect(payload_integer(response, "code", name) == AILA_ERR_RUNTIME,
           name,
           "engine error code changed");
    expect(payload_string(response, "message", name) == state.last_error_message,
           name,
           "engine error message changed");
}

void test_shutdown_sets_flag_and_destroys_engine_once() {
    constexpr const char* name = "shutdown sets flag and destroys engine once";
    EngineState state;
    int destructions = 0;
    {
        WorkerDispatcher dispatcher(fake_engine(state, destructions));
        const Frame command = request(80, "shutdown");
        bool should_shutdown = false;
        const Frame response = dispatcher.dispatch(command, should_shutdown);

        expect_response_identity(response, command, "result", name);
        expect(payload_boolean(response, "ok", name), name, "shutdown did not report success");
        expect(should_shutdown, name, "shutdown flag was not set");
        expect(destructions == 0, name, "engine was destroyed before dispatcher destruction");
    }
    expect(destructions == 1, name, "engine was not destroyed exactly once");
}

void test_protocol_and_abi_mismatches_are_rejected() {
    constexpr const char* name = "protocol and ABI mismatches are rejected";
    EngineState state;
    int destructions = 0;
    WorkerDispatcher dispatcher(fake_engine(state, destructions));
    bool should_shutdown = false;

    Frame protocol = request(90, "ping");
    ++protocol.header.protocol;
    const Frame protocol_response = dispatcher.dispatch(protocol, should_shutdown);
    expect_response_identity(protocol_response, protocol, "error", name);
    expect(payload_integer(protocol_response, "code", name) == AILA_ERR_INVALID_ARGUMENT,
           name,
           "protocol mismatch returned the wrong code");
    expect(payload_string(protocol_response, "message", name).find("protocol") != std::string::npos,
           name,
           "protocol mismatch message was not useful");

    Frame abi = request(91, "ping");
    ++abi.header.abi;
    const Frame abi_response = dispatcher.dispatch(abi, should_shutdown);
    expect_response_identity(abi_response, abi, "error", name);
    expect(payload_integer(abi_response, "code", name) == AILA_ERR_INVALID_ARGUMENT,
           name,
           "ABI mismatch returned the wrong code");
    expect(payload_string(abi_response, "message", name).find("ABI") != std::string::npos,
           name,
           "ABI mismatch message was not useful");
}

void test_ping_and_exception_containment() {
    constexpr const char* name = "ping and exception containment";
    EngineState state;
    int destructions = 0;
    WorkerDispatcher dispatcher(fake_engine(state, destructions));
    bool should_shutdown = false;

    const Frame ping = request(100, "ping");
    const Frame ping_response = dispatcher.dispatch(ping, should_shutdown);
    expect_response_identity(ping_response, ping, "result", name);
    expect(payload_boolean(ping_response, "pong", name), name, "ping did not return pong");

    state.throw_on_context_length = true;
    const Frame throwing = request(101, "engine.context_length");
    const Frame throwing_response = dispatcher.dispatch(throwing, should_shutdown);
    expect_response_identity(throwing_response, throwing, "error", name);
    expect(payload_integer(throwing_response, "code", name) == AILA_ERR_RUNTIME,
           name,
           "exception returned the wrong code");
    expect(payload_string(throwing_response, "message", name).find("synthetic context failure") !=
               std::string::npos,
           name,
           "exception message was not preserved");
}

} // namespace

int main() {
    try {
        test_init_forwards_utf8_model_and_sequence_length();
        test_context_reset_and_second_init_are_deterministic();
        test_invalid_requests_return_structured_errors();
        test_init_failure_propagates_engine_error();
        test_shutdown_sets_flag_and_destroys_engine_once();
        test_protocol_and_abi_mismatches_are_rejected();
        test_ping_and_exception_containment();
        std::cout << "AilaWorkerDispatcherTests passed\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
