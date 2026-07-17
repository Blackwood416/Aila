#include "worker/WorkerDispatcher.hpp"

#include "aila_api.h"
#include "simdjson.h"

#include <cstdint>
#include <atomic>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using aila::ipc::Frame;
using aila::worker::WorkerDispatcher;
using aila::worker::WorkerEngineApi;
using aila::worker::TextGenerationMethod;
using aila::worker::TextGenerationRequest;

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
    int generate_calls = 0;
    bool generate_result = true;
    std::string generated_text = u8"worker 输出";
    TextGenerationRequest generation_request;
    int stream_calls = 0;
    uint64_t asr_stream_id = 0;
    std::vector<float> asr_samples;
    aila::worker::AsrRequest asr_request;
    aila::worker::AsrStreamConfig asr_stream_config;
    int asr_transcribe_calls = 0;
    int asr_create_calls = 0;
    int asr_destroy_calls = 0;
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

    bool generate_text(
        const TextGenerationRequest& request,
        std::string& output) override {
        ++state_.generate_calls;
        state_.generation_request = request;
        output = state_.generated_text;
        return state_.generate_result;
    }

    int generate_stream(
        const TextGenerationRequest& request,
        const aila::worker::TokenStreamCallback& token_callback,
        const aila::worker::StructuredStreamCallback& structured_callback) override {
        ++state_.stream_calls;
        state_.generation_request = request;
        if (request.method == TextGenerationMethod::GenerateChatJsonStreamEx) {
            AilaChatStreamEvent event{};
            event.struct_size = sizeof(event);
            event.type = AILA_CHAT_STREAM_TOOL_CALL_DELTA;
            event.text = u8"工具";
            event.tool_call_id = nullptr;
            event.tool_name = "search";
            event.arguments_delta = "";
            event.finish_reason = nullptr;
            event.warnings_json = "[]";
            event.tool_calls_json = nullptr;
            return structured_callback(event) ? 0 : 1;
        }
        if (!token_callback(u8"第一")) return 1;
        return token_callback(" second") ? 0 : 1;
    }

    bool transcribe(
        const aila::worker::AsrRequest& request,
        const std::function<void(std::string_view)>& callback,
        std::string& transcript,
        std::string& language) override {
        ++state_.asr_transcribe_calls;
        state_.asr_request = request;
        callback(u8"词");
        transcript = u8"转录";
        language = "Chinese";
        return true;
    }
    bool transcribe_stream_create(
        uint64_t id, const aila::worker::AsrStreamConfig& config) override {
        ++state_.asr_create_calls;
        state_.asr_stream_id = id;
        state_.asr_stream_config = config;
        return true;
    }
    bool transcribe_stream_feed(uint64_t id, const float* samples, size_t count) override {
        if (id != state_.asr_stream_id) return false;
        state_.asr_samples.assign(samples, samples + count);
        return true;
    }
    bool transcribe_stream_get_text(
        uint64_t id, std::string& stable, std::string& partial) override {
        if (id != state_.asr_stream_id) return false;
        stable = u8"稳定";
        partial = u8"部分";
        return true;
    }
    bool transcribe_stream_destroy(uint64_t id) noexcept override {
        ++state_.asr_destroy_calls;
        return id == state_.asr_stream_id;
    }

private:
    EngineState& state_;
    int& destruction_count_;
};

std::unique_ptr<WorkerEngineApi> fake_engine(EngineState& state, int& destruction_count) {
    return std::make_unique<FakeEngine>(state, destruction_count);
}

void initialize(WorkerDispatcher& dispatcher, const char* test_name) {
    bool should_shutdown = false;
    const Frame command = request(1, "engine.init", R"({"model":"m","maxSeqLen":1024})");
    const Frame response = dispatcher.dispatch(command, should_shutdown);
    expect(response.header.kind == "result", test_name, "test engine init failed");
}

std::string attachment_string(const Frame& frame) {
    return std::string(
        reinterpret_cast<const char*>(frame.attachment.data()),
        frame.attachment.size());
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

void test_init_rejects_embedded_nul_model() {
    constexpr const char* name = "engine.init rejects embedded NUL model path";
    EngineState state;
    int destructions = 0;
    WorkerDispatcher dispatcher(fake_engine(state, destructions));
    const Frame command = request(
        75,
        "engine.init",
        R"({"model":"safe\u0000suffix","maxSeqLen":1024})");

    bool should_shutdown = false;
    const Frame response = dispatcher.dispatch(command, should_shutdown);

    expect_response_identity(response, command, "error", name);
    expect(payload_integer(response, "code", name) == AILA_ERR_INVALID_ARGUMENT,
           name,
           "embedded NUL returned the wrong code");
    expect(payload_string(response, "message", name).find("NUL") != std::string::npos,
           name,
           "embedded NUL message was not useful");
    expect(state.init_calls == 0, name, "embedded NUL model reached engine init");
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

void test_generation_methods_forward_unicode_and_return_exact_attachment() {
    constexpr const char* name = "generation methods forward Unicode and exact attachment";
    struct Case {
        const char* wire_method;
        TextGenerationMethod engine_method;
    };
    constexpr Case cases[] = {
        {"generate", TextGenerationMethod::Generate},
        {"generate.messages", TextGenerationMethod::GenerateMessages},
        {"generate.chat_json", TextGenerationMethod::GenerateChatJson},
        {"generate.chat_json_ex", TextGenerationMethod::GenerateChatJsonEx},
    };

    for (const Case& test_case : cases) {
        EngineState state;
        int destructions = 0;
        WorkerDispatcher dispatcher(fake_engine(state, destructions));
        initialize(dispatcher, name);
        bool should_shutdown = false;
        const Frame command = request(
            120,
            test_case.wire_method,
            u8R"({"input":"你好 🌍","config":null})");

        const Frame response = dispatcher.dispatch(command, should_shutdown);

        expect(response.header.kind == "result", name, "generation did not return result");
        expect(response.header.method == test_case.wire_method, name, "method was not echoed");
        expect(attachment_string(response) == state.generated_text, name, "output bytes changed");
        expect(payload_integer(response, "byteCount", name) ==
                   static_cast<int64_t>(state.generated_text.size()),
               name,
               "byte count changed");
        expect(state.generate_calls == 1, name, "adapter was not called exactly once");
        expect(state.generation_request.method == test_case.engine_method,
               name,
               "adapter method changed");
        expect(state.generation_request.input == u8"你好 🌍", name, "Unicode input changed");
        expect(!state.generation_request.has_config, name, "NULL legacy config became present");
        expect(!state.generation_request.has_v2_config, name, "NULL V2 config became present");
    }
}

void test_generation_forwards_every_legacy_config_field() {
    constexpr const char* name = "generation forwards every legacy config field";
    EngineState state;
    int destructions = 0;
    WorkerDispatcher dispatcher(fake_engine(state, destructions));
    initialize(dispatcher, name);
    bool should_shutdown = false;
    const Frame command = request(
        130,
        "generate.chat_json",
        R"({"input":"request","config":{"max_new_tokens":17,"temperature":0.25,"top_k":7,"top_p":0.75,"repetition_penalty":1.125,"presence_penalty":0.375,"frequency_penalty":0.625,"do_sample":0,"decode_chunk_size":3,"stream_chunk_size":5}})");

    const Frame response = dispatcher.dispatch(command, should_shutdown);

    expect(response.header.kind == "result", name, "legacy config request failed");
    expect(state.generation_request.has_config, name, "legacy config was not present");
    expect(!state.generation_request.has_v2_config, name, "legacy config became V2");
    const AilaGenConfig& config = state.generation_request.config;
    expect(config.max_new_tokens == 17, name, "max_new_tokens changed");
    expect(config.temperature == 0.25f, name, "temperature changed");
    expect(config.top_k == 7, name, "top_k changed");
    expect(config.top_p == 0.75f, name, "top_p changed");
    expect(config.repetition_penalty == 1.125f, name, "repetition_penalty changed");
    expect(config.presence_penalty == 0.375f, name, "presence_penalty changed");
    expect(config.frequency_penalty == 0.625f, name, "frequency_penalty changed");
    expect(config.do_sample == 0, name, "do_sample changed");
    expect(config.decode_chunk_size == 3, name, "decode_chunk_size changed");
    expect(config.stream_chunk_size == 5, name, "stream_chunk_size changed");
}

void test_generation_forwards_size_gated_v2_config_without_reserved_fields() {
    constexpr const char* name = "generation forwards size-gated V2 config";
    EngineState state;
    int destructions = 0;
    WorkerDispatcher dispatcher(fake_engine(state, destructions));
    initialize(dispatcher, name);
    bool should_shutdown = false;
    const Frame command = request(
        140,
        "generate.chat_json_ex",
        R"({"input":"request","config":{"struct_size":64,"max_new_tokens":31,"temperature":0.5,"top_k":11,"top_p":0.875,"repetition_penalty":1.25,"presence_penalty":0.125,"frequency_penalty":0.25,"do_sample":1,"decode_chunk_size":9,"stream_chunk_size":6,"thinking_budget_tokens":77,"sampling_seed":123456789,"use_fixed_seed":1}})");

    const Frame response = dispatcher.dispatch(command, should_shutdown);

    expect(response.header.kind == "result", name, "V2 config request failed");
    expect(!state.generation_request.has_config, name, "V2 config became legacy");
    expect(state.generation_request.has_v2_config, name, "V2 config was not present");
    const AilaGenConfigV2& config = state.generation_request.config_v2;
    expect(config.struct_size == 64, name, "struct_size changed");
    expect(config.max_new_tokens == 31, name, "max_new_tokens changed");
    expect(config.temperature == 0.5f, name, "temperature changed");
    expect(config.top_k == 11, name, "top_k changed");
    expect(config.top_p == 0.875f, name, "top_p changed");
    expect(config.repetition_penalty == 1.25f, name, "repetition_penalty changed");
    expect(config.presence_penalty == 0.125f, name, "presence_penalty changed");
    expect(config.frequency_penalty == 0.25f, name, "frequency_penalty changed");
    expect(config.do_sample == 1, name, "do_sample changed");
    expect(config.decode_chunk_size == 9, name, "decode_chunk_size changed");
    expect(config.stream_chunk_size == 6, name, "stream_chunk_size changed");
    expect(config.thinking_budget_tokens == 77, name, "thinking budget changed");
    expect(config.sampling_seed == 123456789, name, "sampling seed changed");
    expect(config.use_fixed_seed == 1, name, "fixed seed changed");
    for (int reserved : config.reserved) {
        expect(reserved == 0, name, "reserved field was accepted from the wire");
    }

    const Frame prefix = request(
        141,
        "generate.chat_json_ex",
        R"({"input":"prefix","config":{"struct_size":12,"max_new_tokens":44,"temperature":0.375}})");
    expect(dispatcher.dispatch(prefix, should_shutdown).header.kind == "result",
           name,
           "prefix-sized V2 config failed");
    expect(state.generation_request.config_v2.struct_size == 12, name, "prefix size changed");
    expect(state.generation_request.config_v2.max_new_tokens == 44,
           name,
           "prefix max_new_tokens changed");
    expect(state.generation_request.config_v2.temperature == 0.375f,
           name,
           "prefix temperature changed");
    expect(state.generation_request.config_v2.top_k == 0,
           name,
           "absent later V2 field was synthesized");

    const int calls_before_zero_size = state.generate_calls;
    const Frame zero_size = request(
        142,
        "generate.chat_json_ex",
        R"({"input":"zero","config":{"struct_size":0}})");
    const Frame zero_response = dispatcher.dispatch(zero_size, should_shutdown);
    expect(zero_response.header.kind == "error", name, "zero-sized V2 config succeeded");
    expect(payload_integer(zero_response, "code", name) == AILA_ERR_INVALID_ARGUMENT,
           name,
           "zero-sized V2 config returned the wrong code");
    expect(state.generate_calls == calls_before_zero_size,
           name,
           "zero-sized V2 config reached adapter");
}

void test_generation_rejects_malformed_payload_before_adapter() {
    constexpr const char* name = "generation rejects malformed payload before adapter";
    const char* payloads[] = {
        R"([])",
        R"({"input":3,"config":null})",
        R"({"input":"safe\u0000suffix","config":null})",
        R"({"input":"x","config":[]})",
        R"({"input":"x","config":{"max_new_tokens":"bad"}})",
        "{\"input\":\"\xc3\x28\",\"config\":null}",
    };
    for (const char* payload : payloads) {
        EngineState state;
        int destructions = 0;
        WorkerDispatcher dispatcher(fake_engine(state, destructions));
        initialize(dispatcher, name);
        bool should_shutdown = false;
        const Frame command = request(150, "generate", payload);
        const Frame response = dispatcher.dispatch(command, should_shutdown);
        expect(response.header.kind == "error", name, "malformed generation payload succeeded");
        expect(payload_integer(response, "code", name) == AILA_ERR_INVALID_ARGUMENT,
               name,
               "malformed payload returned the wrong code");
        expect(state.generate_calls == 0, name, "malformed payload reached adapter");
    }
}

void test_generation_propagates_adapter_error_and_rejects_embedded_nul_output() {
    constexpr const char* name = "generation propagates adapter errors and rejects NUL output";
    EngineState state;
    state.generate_result = false;
    state.last_error_code = AILA_ERR_CONTEXT_OVERFLOW;
    state.last_error_message = "synthetic context overflow";
    int destructions = 0;
    WorkerDispatcher dispatcher(fake_engine(state, destructions));
    initialize(dispatcher, name);
    bool should_shutdown = false;
    const Frame command = request(160, "generate", R"({"input":"x","config":null})");

    const Frame failed = dispatcher.dispatch(command, should_shutdown);
    expect(failed.header.kind == "error", name, "adapter failure became success");
    expect(payload_integer(failed, "code", name) == AILA_ERR_CONTEXT_OVERFLOW,
           name,
           "adapter error code changed");
    expect(payload_string(failed, "message", name) == state.last_error_message,
           name,
           "adapter error message changed");

    state.generate_result = true;
    state.generated_text = std::string("left\0right", 10);
    const Frame malformed = dispatcher.dispatch(command, should_shutdown);
    expect(malformed.header.kind == "error", name, "embedded NUL output succeeded");
    expect(payload_integer(malformed, "code", name) == AILA_ERR_RUNTIME,
           name,
           "embedded NUL output returned the wrong code");
    expect(malformed.attachment.empty(), name, "embedded NUL output was attached");

    state.generated_text = std::string("\xc3\x28", 2);
    const Frame invalid_utf8 = dispatcher.dispatch(command, should_shutdown);
    expect(invalid_utf8.header.kind == "error", name, "invalid UTF-8 output succeeded");
    expect(payload_integer(invalid_utf8, "code", name) == AILA_ERR_RUNTIME,
           name,
           "invalid UTF-8 output returned the wrong code");
    expect(invalid_utf8.attachment.empty(), name, "invalid UTF-8 output was attached");
}

void test_stream_dispatcher_emits_correlated_token_and_structured_events() {
    constexpr const char* name = "stream dispatcher emits correlated events";
    EngineState state;
    int destructions = 0;
    WorkerDispatcher dispatcher(fake_engine(state, destructions));
    initialize(dispatcher, name);
    std::atomic_bool cancelled = false;

    for (const auto& entry : std::vector<std::pair<const char*, TextGenerationMethod>>{
             {"generate.stream", TextGenerationMethod::GenerateStream},
             {"generate.messages_stream", TextGenerationMethod::GenerateMessagesStream}}) {
        std::vector<Frame> events;
        const Frame command = request(170, entry.first, u8R"({"input":"流式","config":null})");
        const Frame response = dispatcher.dispatch_stream(
            command,
            [&](const Frame& event) { events.push_back(event); return true; },
            cancelled);
        expect(response.header.kind == "result", name, "token stream failed");
        expect(payload_integer(response, "status", name) == 0, name, "token status changed");
        expect(payload_integer(response, "eventCount", name) == 3,
               name, "token eventCount did not include terminal event");
        expect(events.size() == 2, name, "token event count changed");
        expect(events[0].header.request_id == 170 && events[0].header.method == entry.first,
               name, "token event identity changed");
        expect(payload_string(events[0], "event", name) == "token", name, "token kind changed");
        expect(payload_integer(events[0], "byteCount", name) ==
                   static_cast<int64_t>(events[0].attachment.size()),
               name, "token byteCount changed");
        expect(attachment_string(events[0]) == u8"第一", name, "token bytes changed");
        expect(state.generation_request.method == entry.second, name, "stream method changed");
    }

    std::vector<Frame> structured;
    const Frame structured_command = request(
        171,
        "generate.chat_json_stream_ex",
        R"({"input":"{}","config":{"struct_size":4}})");
    const Frame structured_response = dispatcher.dispatch_stream(
        structured_command,
        [&](const Frame& event) { structured.push_back(event); return true; },
        cancelled);
    expect(payload_integer(structured_response, "status", name) == 0,
           name, "structured status changed");
    expect(payload_integer(structured_response, "eventCount", name) == 2,
           name, "structured eventCount did not include terminal event");
    expect(structured.size() == 1, name, "structured event count changed");
    expect(payload_string(structured[0], "event", name) == "structured",
           name, "structured event kind changed");
    expect(payload_integer(structured[0], "type", name) == AILA_CHAT_STREAM_TOOL_CALL_DELTA,
           name, "structured type changed");
    simdjson::dom::parser parser;
    const simdjson::dom::element payload = parse_payload(structured[0], parser, name);
    expect(payload["toolCallId"].is_null(), name, "NULL string became empty");
    std::string_view arguments;
    expect(payload["argumentsDelta"].get_string().get(arguments) == simdjson::SUCCESS &&
               arguments.empty(),
           name, "empty string became NULL");

    cancelled.store(true);
    const Frame cancelled_response = dispatcher.dispatch_stream(
        request(172, "generate.stream", R"({"input":"cancel","config":null})"),
        [](const Frame&) { return true; },
        cancelled);
    expect(payload_integer(cancelled_response, "status", name) == 1,
           name, "pre-observed cancellation status changed");
    expect(payload_integer(cancelled_response, "eventCount", name) == 1,
           name, "cancelled eventCount omitted terminal event");
}

void test_stream_event_limit_reserves_terminal_slot() {
    constexpr const char* name = "stream event limit reserves terminal slot";
    expect(
        aila::worker::detail::stream_data_event_can_emit(
            aila::ipc::kMaxStreamEventCount - 2),
        name,
        "last legal data event was rejected");
    expect(
        !aila::worker::detail::stream_data_event_can_emit(
            aila::ipc::kMaxStreamEventCount - 1),
        name,
        "data event consumed the reserved terminal slot");
}

void test_asr_wire_methods_are_dispatched() {
    constexpr const char* name = "ASR wire methods are dispatched";
    EngineState state;
    int destructions = 0;
    WorkerDispatcher dispatcher(fake_engine(state, destructions));
    initialize(dispatcher, name);
    bool should_shutdown = false;
    const Frame create_response = dispatcher.dispatch(
        request(201, "asr.stream.create",
                R"({"config":{"max_new_tokens":8,"temperature":0.5,"top_k":9,"top_p":0.8,"repetition_penalty":1.1,"presence_penalty":0.2,"frequency_penalty":-0.3,"do_sample":0,"decode_chunk_size":7,"stream_chunk_size":2},"forcedLanguage":"","systemPrompt":null})"),
        should_shutdown);
    expect(create_response.header.kind == "result", name,
           "ASR stream create was not handled by the dispatcher");
    expect(payload_integer(create_response, "streamId", name) == 1,
           name, "ASR remote stream ID did not start at one");
    expect(state.asr_create_calls == 1 && state.asr_stream_config.has_config &&
               state.asr_stream_config.has_forced_language &&
               state.asr_stream_config.forced_language.empty() &&
               !state.asr_stream_config.has_system_prompt,
           name, "ASR stream nullable/empty config semantics changed");

    const float samples[] = {1.25f, -2.5f, 0.0f};
    Frame feed = request(
        202, "asr.stream.feed",
        R"({"streamId":1,"sampleCount":3,"elementSize":4,"byteCount":12})");
    feed.attachment.resize(sizeof(samples));
    std::memcpy(feed.attachment.data(), samples, sizeof(samples));
    expect(dispatcher.dispatch(feed, should_shutdown).header.kind == "result",
           name, "valid ASR float attachment was rejected");
    expect(state.asr_samples.size() == 3 &&
               std::memcmp(state.asr_samples.data(), samples, sizeof(samples)) == 0,
           name, "ASR float attachment bytes changed");

    Frame malformed = feed;
    malformed.header.request_id = 203;
    malformed.header.payload_json =
        R"({"streamId":1,"sampleCount":18446744073709551615,"elementSize":4,"byteCount":12})";
    expect(dispatcher.dispatch(malformed, should_shutdown).header.kind == "error",
           name, "overflowing ASR sample count reached the adapter");

    const Frame text_response = dispatcher.dispatch(
        request(204, "asr.stream.get_text", R"({"streamId":1})"), should_shutdown);
    expect(text_response.header.kind == "result" &&
               attachment_string(text_response) == std::string(u8"稳定部分"),
           name, "ASR stable/partial attachment changed");
    expect(payload_integer(text_response, "stableBytes", name) == 6 &&
               payload_integer(text_response, "partialBytes", name) == 6,
           name, "ASR stable/partial lengths changed");

    expect(dispatcher.dispatch(
               request(205, "asr.stream.destroy", R"({"streamId":1})"),
               should_shutdown).header.kind == "result",
           name, "ASR stream destroy failed");
    expect(state.asr_destroy_calls == 1, name, "ASR stream was not destroyed exactly once");
    expect(dispatcher.dispatch(
               request(206, "asr.stream.get_text", R"({"streamId":1})"),
               should_shutdown).header.kind == "error",
           name, "destroyed ASR stream ID remained usable");

    std::vector<Frame> events;
    std::atomic_bool cancelled = false;
    const Frame transcription = dispatcher.dispatch_stream(
        request(207, "asr.transcribe",
                R"({"wavPath":"C:\\音频\\说话.wav","config":null,"forcedLanguage":null,"systemPrompt":"","segmentSec":12.5,"pastTextConditioning":3})"),
        [&](const Frame& event) { events.push_back(event); return true; }, cancelled);
    expect(transcription.header.kind == "result" && events.size() == 1,
           name, "offline ASR did not emit token/result frames");
    expect(payload_integer(transcription, "transcriptBytes", name) == 6 &&
               payload_integer(transcription, "languageBytes", name) == 7 &&
               payload_integer(transcription, "eventCount", name) == 2,
           name, "offline ASR result lengths/event barrier changed");
    expect(attachment_string(transcription) == std::string(u8"转录Chinese"),
           name, "offline ASR transcript/language attachment changed");
    expect(state.asr_transcribe_calls == 1 &&
               state.asr_request.wav_path == u8R"(C:\音频\说话.wav)" &&
               !state.asr_request.has_forced_language &&
               state.asr_request.has_system_prompt && state.asr_request.system_prompt.empty() &&
               state.asr_request.segment_sec == 12.5f &&
               state.asr_request.past_text_conditioning == 3,
           name, "offline ASR arguments were not forwarded exactly");
}

} // namespace

int main() {
    try {
        test_init_forwards_utf8_model_and_sequence_length();
        test_context_reset_and_second_init_are_deterministic();
        test_invalid_requests_return_structured_errors();
        test_init_failure_propagates_engine_error();
        test_init_rejects_embedded_nul_model();
        test_shutdown_sets_flag_and_destroys_engine_once();
        test_protocol_and_abi_mismatches_are_rejected();
        test_ping_and_exception_containment();
        test_generation_methods_forward_unicode_and_return_exact_attachment();
        test_generation_forwards_every_legacy_config_field();
        test_generation_forwards_size_gated_v2_config_without_reserved_fields();
        test_generation_rejects_malformed_payload_before_adapter();
        test_generation_propagates_adapter_error_and_rejects_embedded_nul_output();
        test_stream_dispatcher_emits_correlated_token_and_structured_events();
        test_stream_event_limit_reserves_terminal_slot();
        test_asr_wire_methods_are_dispatched();
        std::cout << "AilaWorkerDispatcherTests passed\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
