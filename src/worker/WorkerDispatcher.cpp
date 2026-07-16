#include "worker/WorkerDispatcher.hpp"

#include "aila_api.h"
#include "simdjson.h"

#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace aila::worker {
namespace {

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

bool require_c_string_safe(std::string_view value) {
    return value.find('\0') == std::string_view::npos;
}

ipc::Frame response_base(const ipc::Frame& request, std::string kind) {
    ipc::Frame response;
    response.header.protocol = request.header.protocol;
    response.header.abi = request.header.abi;
    response.header.request_id = request.header.request_id;
    response.header.kind = std::move(kind);
    response.header.method = request.header.method;
    return response;
}

ipc::Frame result(const ipc::Frame& request, std::string payload_json) {
    ipc::Frame response = response_base(request, "result");
    response.header.payload_json = std::move(payload_json);
    return response;
}

ipc::Frame error(const ipc::Frame& request, int code, std::string_view message) {
    ipc::Frame response = response_base(request, "error");
    response.header.payload_json =
        std::string("{\"code\":") + std::to_string(code) +
        ",\"message\":" + json_string(message) + "}";
    return response;
}

} // namespace

WorkerDispatcher::WorkerDispatcher(std::unique_ptr<WorkerEngineApi> engine)
    : engine_(std::move(engine)) {
    if (!engine_) {
        throw std::invalid_argument("worker dispatcher requires an engine API");
    }
}

ipc::Frame WorkerDispatcher::dispatch(const ipc::Frame& request, bool& should_shutdown) {
    should_shutdown = false;
    try {
        if (request.header.kind != "request") {
            return error(request, AILA_ERR_INVALID_ARGUMENT, "frame kind must be 'request'");
        }
        if (request.header.protocol != ipc::kProtocolVersion) {
            return error(request, AILA_ERR_INVALID_ARGUMENT, "unsupported worker protocol version");
        }
        if (request.header.abi != ipc::kPublicAbiVersion) {
            return error(request, AILA_ERR_INVALID_ARGUMENT, "unsupported public ABI version");
        }

        simdjson::dom::parser parser;
        simdjson::dom::element payload;
        const simdjson::error_code parse_error =
            parser.parse(request.header.payload_json).get(payload);
        if (parse_error != simdjson::SUCCESS) {
            return error(
                request,
                AILA_ERR_JSON_PARSE,
                std::string("request payload JSON parse failed: ") +
                    simdjson::error_message(parse_error));
        }

        if (request.header.method == "ping") {
            return result(request, "{\"pong\":true}");
        }

        if (request.header.method == "engine.init") {
            if (initialized_) {
                return error(request, AILA_ERR_INVALID_ARGUMENT, "engine is already initialized");
            }

            simdjson::dom::object object;
            if (payload.get_object().get(object) != simdjson::SUCCESS) {
                return error(request, AILA_ERR_INVALID_ARGUMENT, "engine.init payload must be an object");
            }

            std::string_view model;
            if (object["model"].get_string().get(model) != simdjson::SUCCESS || model.empty()) {
                return error(
                    request,
                    AILA_ERR_INVALID_ARGUMENT,
                    "engine.init field 'model' must be a non-empty UTF-8 string");
            }
            if (!require_c_string_safe(model)) {
                return error(
                    request,
                    AILA_ERR_INVALID_ARGUMENT,
                    "engine.init field 'model' must not contain an embedded NUL");
            }

            int64_t max_seq_len = 0;
            if (object["maxSeqLen"].get_int64().get(max_seq_len) != simdjson::SUCCESS ||
                max_seq_len <= 0 || max_seq_len > (std::numeric_limits<int>::max)()) {
                return error(
                    request,
                    AILA_ERR_INVALID_ARGUMENT,
                    "engine.init field 'maxSeqLen' must be a positive integer");
            }

            const int init_result =
                engine_->init(std::string(model), static_cast<int>(max_seq_len));
            if (init_result != 0) {
                int code = engine_->last_error_code();
                if (code == AILA_OK) {
                    code = AILA_ERR_RUNTIME;
                }
                std::string message = engine_->last_error_message();
                if (message.empty()) {
                    message = "engine initialization failed";
                }
                return error(request, code, message);
            }
            initialized_ = true;
            return result(request, "{\"ok\":true}");
        }

        if (request.header.method == "engine.context_length") {
            return result(
                request,
                std::string("{\"contextLength\":") +
                    std::to_string(engine_->context_length()) + "}");
        }

        if (request.header.method == "engine.reset") {
            engine_->reset_context();
            return result(request, "{\"ok\":true}");
        }

        if (request.header.method == "shutdown") {
            should_shutdown = true;
            return result(request, "{\"ok\":true}");
        }

        return error(
            request,
            AILA_ERR_INVALID_ARGUMENT,
            std::string("unsupported worker method: ") + request.header.method);
    } catch (const std::exception& exception) {
        return error(request, AILA_ERR_RUNTIME, exception.what());
    } catch (...) {
        return error(request, AILA_ERR_RUNTIME, "unknown worker dispatcher failure");
    }
}

} // namespace aila::worker
