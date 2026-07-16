#include "proxy/ProxyEngine.hpp"

#include "aila_api.h"
#include "runtime/RuntimeDirectory.hpp"
#include "simdjson.h"

#include <windows.h>

#include <chrono>
#include <limits>
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
    return std::runtime_error("worker lifecycle response was invalid: " + std::string(detail));
}

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
    if (response.header.method != expected_method) {
        throw malformed_response("method did not match the request");
    }
    if (!response.attachment.empty()) {
        throw malformed_response("lifecycle response contained an attachment");
    }
    if (response.header.kind == "result") {
        return true;
    }
    if (response.header.kind != "error") {
        throw malformed_response("kind was neither result nor error");
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
        message.empty() ? "worker lifecycle request failed" : std::string(message));
    return false;
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
