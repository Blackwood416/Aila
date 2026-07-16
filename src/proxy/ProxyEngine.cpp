#include "proxy/ProxyEngine.hpp"

#include "aila_api.h"
#include "runtime/RuntimeDirectory.hpp"
#include "simdjson.h"

#include <windows.h>

#include <chrono>
#include <cstddef>
#include <cstring>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace aila::proxy {
namespace {

void proxy_module_anchor() {}

HMODULE this_proxy_module() {
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&proxy_module_anchor),
            &module)) {
        throw std::runtime_error(
            "GetModuleHandleExW could not identify the loaded AilaShared proxy");
    }
    return module;
}

std::string json_string(std::string_view value) {
    constexpr char hex[] = "0123456789abcdef";
    std::string result = "\"";
    for (const unsigned char character : value) {
        switch (character) {
            case '"': result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\b': result += "\\b"; break;
            case '\f': result += "\\f"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default:
                if (character < 0x20) {
                    result += "\\u00";
                    result += hex[character >> 4];
                    result += hex[character & 0x0f];
                } else {
                    result += static_cast<char>(character);
                }
                break;
        }
    }
    result += '"';
    return result;
}

std::runtime_error malformed_response(std::string_view detail) {
    return std::runtime_error("worker response was invalid: " + std::string(detail));
}

std::string json_float(float value) {
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << std::setprecision((std::numeric_limits<float>::max_digits10)) << value;
    return stream.str();
}

std::string legacy_config_json(const AilaGenConfig* config) {
    if (!config) {
        return "null";
    }
    return std::string("{") +
        "\"max_new_tokens\":" + std::to_string(config->max_new_tokens) + "," +
        "\"temperature\":" + json_float(config->temperature) + "," +
        "\"top_k\":" + std::to_string(config->top_k) + "," +
        "\"top_p\":" + json_float(config->top_p) + "," +
        "\"repetition_penalty\":" + json_float(config->repetition_penalty) + "," +
        "\"presence_penalty\":" + json_float(config->presence_penalty) + "," +
        "\"frequency_penalty\":" + json_float(config->frequency_penalty) + "," +
        "\"do_sample\":" + std::to_string(config->do_sample) + "," +
        "\"decode_chunk_size\":" + std::to_string(config->decode_chunk_size) + "," +
        "\"stream_chunk_size\":" + std::to_string(config->stream_chunk_size) + "}";
}

bool v2_has_field(uint32_t struct_size, size_t offset, size_t field_size) {
    return struct_size >= offset + field_size;
}

#define AILA_PROXY_V2_HAS(config, field) \
    v2_has_field((config)->struct_size, offsetof(AilaGenConfigV2, field), sizeof((config)->field))

std::string v2_config_json(const AilaGenConfigV2* config) {
    if (!config) {
        return "null";
    }
    const uint32_t struct_size = config->struct_size;
    std::string result = "{\"struct_size\":" + std::to_string(struct_size);
    if (AILA_PROXY_V2_HAS(config, max_new_tokens)) {
        result += ",\"max_new_tokens\":" + std::to_string(config->max_new_tokens);
    }
    if (AILA_PROXY_V2_HAS(config, temperature)) {
        result += ",\"temperature\":" + json_float(config->temperature);
    }
    if (AILA_PROXY_V2_HAS(config, top_k)) {
        result += ",\"top_k\":" + std::to_string(config->top_k);
    }
    if (AILA_PROXY_V2_HAS(config, top_p)) {
        result += ",\"top_p\":" + json_float(config->top_p);
    }
    if (AILA_PROXY_V2_HAS(config, repetition_penalty)) {
        result += ",\"repetition_penalty\":" + json_float(config->repetition_penalty);
    }
    if (AILA_PROXY_V2_HAS(config, presence_penalty)) {
        result += ",\"presence_penalty\":" + json_float(config->presence_penalty);
    }
    if (AILA_PROXY_V2_HAS(config, frequency_penalty)) {
        result += ",\"frequency_penalty\":" + json_float(config->frequency_penalty);
    }
    if (AILA_PROXY_V2_HAS(config, do_sample)) {
        result += ",\"do_sample\":" + std::to_string(config->do_sample);
    }
    if (AILA_PROXY_V2_HAS(config, decode_chunk_size)) {
        result += ",\"decode_chunk_size\":" + std::to_string(config->decode_chunk_size);
    }
    if (AILA_PROXY_V2_HAS(config, stream_chunk_size)) {
        result += ",\"stream_chunk_size\":" + std::to_string(config->stream_chunk_size);
    }
    if (AILA_PROXY_V2_HAS(config, thinking_budget_tokens)) {
        result += ",\"thinking_budget_tokens\":" +
            std::to_string(config->thinking_budget_tokens);
    }
    if (AILA_PROXY_V2_HAS(config, sampling_seed)) {
        result += ",\"sampling_seed\":" + std::to_string(config->sampling_seed);
    }
    if (AILA_PROXY_V2_HAS(config, use_fixed_seed)) {
        result += ",\"use_fixed_seed\":" + std::to_string(config->use_fixed_seed);
    }
    result += "}";
    return result;
}

#undef AILA_PROXY_V2_HAS

} // namespace

ProxyEngine::~ProxyEngine() noexcept {
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        shutdown_locked();
    } catch (...) {
    }
}

bool ProxyEngine::init(std::string_view model_directory, int max_seq_len) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized_) {
        set_error_locked(AILA_ERR_INVALID_ARGUMENT, "engine is already initialized");
        return false;
    }
    if (model_directory.empty()) {
        set_error_locked(AILA_ERR_INVALID_ARGUMENT, "model directory must not be empty");
        return false;
    }
    if (model_directory.find('\0') != std::string_view::npos) {
        set_error_locked(
            AILA_ERR_INVALID_ARGUMENT,
            "model directory must not contain an embedded NUL");
        return false;
    }
    if (max_seq_len <= 0) {
        set_error_locked(AILA_ERR_INVALID_ARGUMENT, "max_seq_len must be positive");
        return false;
    }

    shutdown_locked();
    try {
        const std::filesystem::path module_path =
            runtime::proxy_module_path(this_proxy_module());
        const std::filesystem::path runtime_directory = runtime::resolve_runtime_directory(
            module_path, runtime::runtime_directory_override());
        const std::filesystem::path worker_executable =
            runtime::require_worker_executable(runtime_directory);
        worker_.start(
            runtime_directory,
            worker_executable,
            runtime::ExpectedHandshake{
                ipc::kProtocolVersion,
                ipc::kPublicAbiVersion,
                AILA_BUILD_ID,
            });

        const std::string payload =
            std::string("{\"model\":") + json_string(model_directory) +
            ",\"maxSeqLen\":" + std::to_string(max_seq_len) + "}";
        const ipc::Frame response = request_locked("engine.init", payload);
        if (!accept_lifecycle_response_locked(response, "engine.init")) {
            shutdown_locked();
            return false;
        }
        initialized_ = true;
        clear_error_locked();
        return true;
    } catch (const std::exception& exception) {
        shutdown_locked();
        set_error_locked(AILA_ERR_RUNTIME, exception.what());
        return false;
    } catch (...) {
        shutdown_locked();
        set_error_locked(AILA_ERR_RUNTIME, "unknown proxy initialization failure");
        return false;
    }
}

void ProxyEngine::reset_context() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) {
        set_error_locked(AILA_ERR_INVALID_ARGUMENT, "engine is not initialized");
        return;
    }
    try {
        const ipc::Frame response = request_locked("engine.reset", "{}");
        if (accept_lifecycle_response_locked(response, "engine.reset")) {
            clear_error_locked();
        }
    } catch (const std::exception& exception) {
        shutdown_locked();
        set_error_locked(AILA_ERR_RUNTIME, exception.what());
    } catch (...) {
        shutdown_locked();
        set_error_locked(AILA_ERR_RUNTIME, "unknown proxy reset failure");
    }
}

int ProxyEngine::context_length() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) {
        set_error_locked(AILA_ERR_INVALID_ARGUMENT, "engine is not initialized");
        return 0;
    }
    try {
        const ipc::Frame response = request_locked("engine.context_length", "{}");
        if (!accept_lifecycle_response_locked(response, "engine.context_length")) {
            return 0;
        }
        simdjson::dom::parser parser;
        simdjson::dom::element payload;
        int64_t length = 0;
        if (parser.parse(response.header.payload_json).get(payload) != simdjson::SUCCESS ||
            payload["contextLength"].get_int64().get(length) != simdjson::SUCCESS ||
            length < 0 || length > (std::numeric_limits<int>::max)()) {
            throw malformed_response("contextLength must be a non-negative integer");
        }
        clear_error_locked();
        return static_cast<int>(length);
    } catch (const std::exception& exception) {
        shutdown_locked();
        set_error_locked(AILA_ERR_RUNTIME, exception.what());
        return 0;
    } catch (...) {
        shutdown_locked();
        set_error_locked(AILA_ERR_RUNTIME, "unknown proxy context-length failure");
        return 0;
    }
}

bool ProxyEngine::generate_text(
    std::string_view method,
    std::string_view input,
    const AilaGenConfig* config,
    std::string& output) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) {
        set_error_locked(AILA_ERR_INVALID_ARGUMENT, "engine is not initialized");
        return false;
    }
    if (input.find('\0') != std::string_view::npos) {
        set_error_locked(AILA_ERR_INVALID_ARGUMENT, "generation input must not contain an embedded NUL");
        return false;
    }
    if (!simdjson::validate_utf8(input)) {
        set_error_locked(AILA_ERR_INVALID_ARGUMENT, "generation input must be valid UTF-8");
        return false;
    }
    try {
        return generate_payload_locked(
            method,
            std::string("{\"input\":") + json_string(input) +
                ",\"config\":" + legacy_config_json(config) + "}",
            output);
    } catch (const std::exception& exception) {
        shutdown_locked();
        set_error_locked(AILA_ERR_RUNTIME, exception.what());
        return false;
    } catch (...) {
        shutdown_locked();
        set_error_locked(AILA_ERR_RUNTIME, "unknown proxy generation failure");
        return false;
    }
}

bool ProxyEngine::generate_text_v2(
    std::string_view method,
    std::string_view input,
    const AilaGenConfigV2* config,
    std::string& output) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) {
        set_error_locked(AILA_ERR_INVALID_ARGUMENT, "engine is not initialized");
        return false;
    }
    if (config && config->struct_size == 0) {
        set_error_locked(
            AILA_ERR_INVALID_ARGUMENT,
            "generation V2 config struct_size must not be zero");
        return false;
    }
    if (input.find('\0') != std::string_view::npos) {
        set_error_locked(AILA_ERR_INVALID_ARGUMENT, "generation input must not contain an embedded NUL");
        return false;
    }
    if (!simdjson::validate_utf8(input)) {
        set_error_locked(AILA_ERR_INVALID_ARGUMENT, "generation input must be valid UTF-8");
        return false;
    }
    try {
        return generate_payload_locked(
            method,
            std::string("{\"input\":") + json_string(input) +
                ",\"config\":" + v2_config_json(config) + "}",
            output);
    } catch (const std::exception& exception) {
        shutdown_locked();
        set_error_locked(AILA_ERR_RUNTIME, exception.what());
        return false;
    } catch (...) {
        shutdown_locked();
        set_error_locked(AILA_ERR_RUNTIME, "unknown proxy generation failure");
        return false;
    }
}

int ProxyEngine::generate_stream(
    std::string_view method,
    std::string_view input,
    const AilaGenConfig* config,
    AilaTokenCallback callback,
    void* user_data) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_ || !callback) {
        set_error_locked(AILA_ERR_INVALID_ARGUMENT,
                         !initialized_ ? "engine is not initialized" : "stream callback must not be NULL");
        return -1;
    }
    if (input.find('\0') != std::string_view::npos || !simdjson::validate_utf8(input)) {
        set_error_locked(AILA_ERR_INVALID_ARGUMENT,
                         "stream input must be C-string-safe valid UTF-8");
        return -1;
    }
    try {
        return stream_payload_locked(
            method,
            std::string("{\"input\":") + json_string(input) +
                ",\"config\":" + legacy_config_json(config) + "}",
            callback,
            nullptr,
            user_data);
    } catch (const std::exception& exception) {
        shutdown_locked();
        set_error_locked(AILA_ERR_RUNTIME, exception.what());
        return -1;
    } catch (...) {
        shutdown_locked();
        set_error_locked(AILA_ERR_RUNTIME, "unknown proxy streaming failure");
        return -1;
    }
}

int ProxyEngine::generate_stream_v2(
    std::string_view method,
    std::string_view input,
    const AilaGenConfigV2* config,
    AilaChatStreamCallback callback,
    void* user_data) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_ || !callback) {
        set_error_locked(AILA_ERR_INVALID_ARGUMENT,
                         !initialized_ ? "engine is not initialized" : "stream callback must not be NULL");
        return -1;
    }
    if (config && config->struct_size == 0) {
        set_error_locked(AILA_ERR_INVALID_ARGUMENT,
                         "generation V2 config struct_size must not be zero");
        return -1;
    }
    if (input.find('\0') != std::string_view::npos || !simdjson::validate_utf8(input)) {
        set_error_locked(AILA_ERR_INVALID_ARGUMENT,
                         "stream input must be C-string-safe valid UTF-8");
        return -1;
    }
    try {
        return stream_payload_locked(
            method,
            std::string("{\"input\":") + json_string(input) +
                ",\"config\":" + v2_config_json(config) + "}",
            nullptr,
            callback,
            user_data);
    } catch (const std::exception& exception) {
        shutdown_locked();
        set_error_locked(AILA_ERR_RUNTIME, exception.what());
        return -1;
    } catch (...) {
        shutdown_locked();
        set_error_locked(AILA_ERR_RUNTIME, "unknown proxy structured streaming failure");
        return -1;
    }
}

int ProxyEngine::last_error_code() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return error_code_;
}

const char* ProxyEngine::last_error_message() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return error_message_.c_str();
}

void ProxyEngine::record_invalid_argument(std::string message) {
    std::lock_guard<std::mutex> lock(mutex_);
    set_error_locked(AILA_ERR_INVALID_ARGUMENT, std::move(message));
}

void ProxyEngine::record_runtime_error(std::string message) {
    std::lock_guard<std::mutex> lock(mutex_);
    set_error_locked(AILA_ERR_RUNTIME, std::move(message));
}

ipc::Frame ProxyEngine::request_locked(std::string method, std::string payload_json) {
    ipc::Frame request;
    request.header.protocol = ipc::kProtocolVersion;
    request.header.abi = ipc::kPublicAbiVersion;
    request.header.request_id = next_request_id_++;
    if (next_request_id_ == 0 ||
        next_request_id_ == (std::numeric_limits<uint64_t>::max)()) {
        next_request_id_ = 1;
    }
    request.header.kind = "request";
    request.header.method = std::move(method);
    request.header.payload_json = std::move(payload_json);
    return worker_.request(request, std::chrono::minutes(10));
}

bool ProxyEngine::accept_lifecycle_response_locked(
    const ipc::Frame& response,
    std::string_view expected_method) {
    if (!response.attachment.empty()) {
        throw malformed_response("lifecycle response contained an attachment");
    }
    if (response.header.kind == "result") {
        if (response.header.method != expected_method) {
            throw malformed_response("method did not match the request");
        }
        return true;
    }
    return accept_error_response_locked(
        response,
        expected_method,
        "worker lifecycle request failed");
}

bool ProxyEngine::accept_error_response_locked(
    const ipc::Frame& response,
    std::string_view expected_method,
    std::string_view fallback_message) {
    if (response.header.method != expected_method) {
        throw malformed_response("method did not match the request");
    }
    if (response.header.kind != "error") {
        throw malformed_response("kind was neither result nor error");
    }
    if (!response.attachment.empty()) {
        throw malformed_response("error response contained an attachment");
    }

    simdjson::dom::parser parser;
    simdjson::dom::element payload;
    int64_t code = 0;
    std::string_view message;
    if (parser.parse(response.header.payload_json).get(payload) != simdjson::SUCCESS ||
        payload["code"].get_int64().get(code) != simdjson::SUCCESS ||
        payload["message"].get_string().get(message) != simdjson::SUCCESS ||
        code < AILA_OK || code > (std::numeric_limits<int>::max)()) {
        throw malformed_response("error payload requires integer code and string message");
    }
    set_error_locked(
        code == AILA_OK ? AILA_ERR_RUNTIME : static_cast<int>(code),
        message.empty() ? std::string(fallback_message) : std::string(message));
    return false;
}

bool ProxyEngine::generate_payload_locked(
    std::string_view method,
    std::string payload_json,
    std::string& output) {
    const ipc::Frame response = request_locked(std::string(method), std::move(payload_json));
    if (response.header.kind == "error") {
        return accept_error_response_locked(response, method, "worker generation failed");
    }
    if (response.header.method != method || response.header.kind != "result") {
        throw malformed_response("generation result identity did not match request");
    }
    simdjson::dom::parser parser;
    simdjson::dom::element payload;
    uint64_t byte_count = 0;
    if (parser.parse(response.header.payload_json).get(payload) != simdjson::SUCCESS ||
        payload["byteCount"].get_uint64().get(byte_count) != simdjson::SUCCESS ||
        byte_count != response.attachment.size()) {
        throw malformed_response("generation byteCount did not match attachment");
    }
    for (const std::byte byte : response.attachment) {
        if (byte == std::byte{0}) {
            throw malformed_response("generation attachment contained an embedded NUL");
        }
    }
    if (!response.attachment.empty() &&
        !simdjson::validate_utf8(
            reinterpret_cast<const char*>(response.attachment.data()),
            response.attachment.size())) {
        throw malformed_response("generation attachment was not valid UTF-8");
    }
    if (response.attachment.empty()) {
        output.clear();
    } else {
        output.assign(
            reinterpret_cast<const char*>(response.attachment.data()),
            response.attachment.size());
    }
    clear_error_locked();
    return true;
}

int ProxyEngine::stream_payload_locked(
    std::string_view method,
    std::string payload_json,
    AilaTokenCallback token_callback,
    AilaChatStreamCallback structured_callback,
    void* user_data) {
    ipc::Frame request;
    request.header.protocol = ipc::kProtocolVersion;
    request.header.abi = ipc::kPublicAbiVersion;
    request.header.request_id = next_request_id_++;
    if (next_request_id_ == 0 ||
        next_request_id_ == (std::numeric_limits<uint64_t>::max)()) next_request_id_ = 1;
    request.header.kind = "request";
    request.header.method = std::string(method);
    request.header.payload_json = std::move(payload_json);

    bool aborted = false;
    const ipc::Frame response = worker_.request_stream(
        request,
        [&](const ipc::Frame& event) {
            if (event.header.kind != "event" || event.header.method != method ||
                event.header.protocol != ipc::kProtocolVersion ||
                event.header.abi != ipc::kPublicAbiVersion) {
                throw malformed_response("stream event identity did not match request");
            }
            simdjson::dom::parser parser;
            simdjson::dom::element root;
            simdjson::dom::object object;
            std::string_view event_kind;
            if (!simdjson::validate_utf8(event.header.payload_json) ||
                parser.parse(event.header.payload_json).get(root) != simdjson::SUCCESS ||
                root.get_object().get(object) != simdjson::SUCCESS ||
                object["event"].get_string().get(event_kind) != simdjson::SUCCESS) {
                throw malformed_response("stream event payload schema was invalid");
            }
            size_t field_count = 0;
            for (auto field : object) { (void)field; ++field_count; }
            if (event_kind == "end") {
                if (field_count != 1 || !event.attachment.empty()) {
                    throw malformed_response("stream terminal event schema was invalid");
                }
                return runtime::WorkerProcess::StreamEventAction::End;
            }
            if (event_kind == "token") {
                uint64_t byte_count = 0;
                if (!token_callback || field_count != 2 ||
                    object["byteCount"].get_uint64().get(byte_count) != simdjson::SUCCESS ||
                    byte_count != event.attachment.size()) {
                    throw malformed_response("token stream event byteCount or method was invalid");
                }
                for (std::byte byte : event.attachment) {
                    if (byte == std::byte{0}) {
                        throw malformed_response("token stream event contained an embedded NUL");
                    }
                }
                const std::string token(
                    reinterpret_cast<const char*>(event.attachment.data()),
                    event.attachment.size());
                if (!simdjson::validate_utf8(token)) {
                    throw malformed_response("token stream event was not valid UTF-8");
                }
                if (!aborted && token_callback(token.c_str(), user_data) != 0) {
                    aborted = true;
                    return runtime::WorkerProcess::StreamEventAction::Cancel;
                }
                return runtime::WorkerProcess::StreamEventAction::Continue;
            }
            if (event_kind != "structured" || !structured_callback ||
                field_count != 10 || !event.attachment.empty()) {
                throw malformed_response("structured stream event schema was invalid");
            }
            uint64_t struct_size = 0;
            int64_t type = 0;
            if (object["structSize"].get_uint64().get(struct_size) != simdjson::SUCCESS ||
                struct_size != sizeof(AilaChatStreamEvent) ||
                object["type"].get_int64().get(type) != simdjson::SUCCESS ||
                type < AILA_CHAT_STREAM_REASONING_DELTA || type > AILA_CHAT_STREAM_FINAL) {
                throw malformed_response("structured stream event size or type was invalid");
            }
            auto optional_string = [&](const char* name) -> std::optional<std::string> {
                simdjson::dom::element value;
                if (object.at_key(name).get(value) != simdjson::SUCCESS) {
                    throw malformed_response("structured stream event omitted a field");
                }
                if (value.is_null()) return std::nullopt;
                std::string_view text;
                if (value.get_string().get(text) != simdjson::SUCCESS ||
                    text.find('\0') != std::string_view::npos || !simdjson::validate_utf8(text)) {
                    throw malformed_response("structured stream string field was invalid");
                }
                return std::string(text);
            };
            const auto text = optional_string("text");
            const auto tool_call_id = optional_string("toolCallId");
            const auto tool_name = optional_string("toolName");
            const auto arguments_delta = optional_string("argumentsDelta");
            const auto finish_reason = optional_string("finishReason");
            const auto warnings_json = optional_string("warningsJson");
            const auto tool_calls_json = optional_string("toolCallsJson");
            AilaChatStreamEvent abi_event{};
            abi_event.struct_size = sizeof(abi_event);
            abi_event.type = static_cast<int>(type);
            abi_event.text = text ? text->c_str() : nullptr;
            abi_event.tool_call_id = tool_call_id ? tool_call_id->c_str() : nullptr;
            abi_event.tool_name = tool_name ? tool_name->c_str() : nullptr;
            abi_event.arguments_delta = arguments_delta ? arguments_delta->c_str() : nullptr;
            abi_event.finish_reason = finish_reason ? finish_reason->c_str() : nullptr;
            abi_event.warnings_json = warnings_json ? warnings_json->c_str() : nullptr;
            abi_event.tool_calls_json = tool_calls_json ? tool_calls_json->c_str() : nullptr;
            if (!aborted && structured_callback(&abi_event, user_data) != 0) {
                aborted = true;
                return runtime::WorkerProcess::StreamEventAction::Cancel;
            }
            return runtime::WorkerProcess::StreamEventAction::Continue;
        },
        std::chrono::minutes(10));
    if (response.header.kind == "error") {
        accept_error_response_locked(response, method, "worker streaming generation failed");
        return -1;
    }
    if (response.header.kind != "result" || response.header.method != method ||
        !response.attachment.empty()) {
        throw malformed_response("stream result identity was invalid");
    }
    simdjson::dom::parser parser;
    simdjson::dom::element root;
    simdjson::dom::object object;
    int64_t status = -1;
    if (parser.parse(response.header.payload_json).get(root) != simdjson::SUCCESS ||
        root.get_object().get(object) != simdjson::SUCCESS ||
        object["status"].get_int64().get(status) != simdjson::SUCCESS ||
        status < 0 || status > 1) {
        throw malformed_response("stream result status was invalid");
    }
    size_t field_count = 0;
    for (auto field : object) { (void)field; ++field_count; }
    if (field_count != 1 || (aborted && status != 1)) {
        throw malformed_response("stream result did not acknowledge callback cancellation");
    }
    clear_error_locked();
    return static_cast<int>(status);
}

void ProxyEngine::set_error_locked(int code, std::string message) {
    error_code_ = code;
    error_message_ = std::move(message);
}

void ProxyEngine::clear_error_locked() {
    error_code_ = AILA_OK;
    error_message_.clear();
}

void ProxyEngine::shutdown_locked() noexcept {
    initialized_ = false;
    try {
        worker_.shutdown(std::chrono::seconds(2));
    } catch (...) {
    }
}

} // namespace aila::proxy
