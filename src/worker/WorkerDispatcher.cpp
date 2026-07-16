#include "worker/WorkerDispatcher.hpp"

#include "aila_api.h"
#include "simdjson.h"

#include <exception>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
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

bool generation_method(std::string_view method, TextGenerationMethod& result) {
    if (method == "generate") {
        result = TextGenerationMethod::Generate;
    } else if (method == "generate.messages") {
        result = TextGenerationMethod::GenerateMessages;
    } else if (method == "generate.chat_json") {
        result = TextGenerationMethod::GenerateChatJson;
    } else if (method == "generate.chat_json_ex") {
        result = TextGenerationMethod::GenerateChatJsonEx;
    } else {
        return false;
    }
    return true;
}

bool stream_generation_method(std::string_view method, TextGenerationMethod& result) {
    if (method == "generate.stream") {
        result = TextGenerationMethod::GenerateStream;
    } else if (method == "generate.messages_stream") {
        result = TextGenerationMethod::GenerateMessagesStream;
    } else if (method == "generate.chat_json_stream_ex") {
        result = TextGenerationMethod::GenerateChatJsonStreamEx;
    } else {
        return false;
    }
    return true;
}

bool object_integer(
    simdjson::dom::object object,
    const char* name,
    int& value) {
    int64_t parsed = 0;
    if (object[name].get_int64().get(parsed) != simdjson::SUCCESS ||
        parsed < (std::numeric_limits<int>::min)() ||
        parsed > (std::numeric_limits<int>::max)()) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

bool object_float(
    simdjson::dom::object object,
    const char* name,
    float& value) {
    double parsed = 0.0;
    if (object[name].get_double().get(parsed) != simdjson::SUCCESS ||
        !std::isfinite(parsed) ||
        parsed < -(std::numeric_limits<float>::max)() ||
        parsed > (std::numeric_limits<float>::max)()) {
        return false;
    }
    value = static_cast<float>(parsed);
    return true;
}

bool parse_legacy_config(simdjson::dom::object object, AilaGenConfig& config) {
    return object_integer(object, "max_new_tokens", config.max_new_tokens) &&
        object_float(object, "temperature", config.temperature) &&
        object_integer(object, "top_k", config.top_k) &&
        object_float(object, "top_p", config.top_p) &&
        object_float(object, "repetition_penalty", config.repetition_penalty) &&
        object_float(object, "presence_penalty", config.presence_penalty) &&
        object_float(object, "frequency_penalty", config.frequency_penalty) &&
        object_integer(object, "do_sample", config.do_sample) &&
        object_integer(object, "decode_chunk_size", config.decode_chunk_size) &&
        object_integer(object, "stream_chunk_size", config.stream_chunk_size);
}

bool v2_has_field(uint32_t size, size_t offset, size_t field_size) {
    return size >= offset + field_size;
}

#define AILA_WORKER_V2_FIELD(config, field) \
    v2_has_field((config).struct_size, offsetof(AilaGenConfigV2, field), sizeof((config).field))

bool parse_v2_config(simdjson::dom::object object, AilaGenConfigV2& config) {
    uint64_t size = 0;
    if (object["struct_size"].get_uint64().get(size) != simdjson::SUCCESS ||
        size > (std::numeric_limits<uint32_t>::max)()) {
        return false;
    }
    config = {};
    config.struct_size = static_cast<uint32_t>(size);
    if (config.struct_size == 0) {
        return false;
    }
    if (AILA_WORKER_V2_FIELD(config, max_new_tokens) &&
        !object_integer(object, "max_new_tokens", config.max_new_tokens)) return false;
    if (AILA_WORKER_V2_FIELD(config, temperature) &&
        !object_float(object, "temperature", config.temperature)) return false;
    if (AILA_WORKER_V2_FIELD(config, top_k) &&
        !object_integer(object, "top_k", config.top_k)) return false;
    if (AILA_WORKER_V2_FIELD(config, top_p) &&
        !object_float(object, "top_p", config.top_p)) return false;
    if (AILA_WORKER_V2_FIELD(config, repetition_penalty) &&
        !object_float(object, "repetition_penalty", config.repetition_penalty)) return false;
    if (AILA_WORKER_V2_FIELD(config, presence_penalty) &&
        !object_float(object, "presence_penalty", config.presence_penalty)) return false;
    if (AILA_WORKER_V2_FIELD(config, frequency_penalty) &&
        !object_float(object, "frequency_penalty", config.frequency_penalty)) return false;
    if (AILA_WORKER_V2_FIELD(config, do_sample) &&
        !object_integer(object, "do_sample", config.do_sample)) return false;
    if (AILA_WORKER_V2_FIELD(config, decode_chunk_size) &&
        !object_integer(object, "decode_chunk_size", config.decode_chunk_size)) return false;
    if (AILA_WORKER_V2_FIELD(config, stream_chunk_size) &&
        !object_integer(object, "stream_chunk_size", config.stream_chunk_size)) return false;
    if (AILA_WORKER_V2_FIELD(config, thinking_budget_tokens) &&
        !object_integer(object, "thinking_budget_tokens", config.thinking_budget_tokens)) return false;
    if (AILA_WORKER_V2_FIELD(config, sampling_seed) &&
        object["sampling_seed"].get_uint64().get(config.sampling_seed) != simdjson::SUCCESS) return false;
    if (AILA_WORKER_V2_FIELD(config, use_fixed_seed) &&
        !object_integer(object, "use_fixed_seed", config.use_fixed_seed)) return false;
    return true;
}

#undef AILA_WORKER_V2_FIELD

ipc::Frame generation_result(
    const ipc::Frame& request,
    std::string_view output) {
    ipc::Frame response = result(
        request,
        std::string("{\"byteCount\":") + std::to_string(output.size()) + "}");
    response.attachment.reserve(output.size());
    for (const unsigned char byte : output) {
        response.attachment.push_back(static_cast<std::byte>(byte));
    }
    return response;
}

std::string nullable_json_string(const char* value) {
    return value == nullptr ? "null" : json_string(value);
}

ipc::Frame token_event(const ipc::Frame& request, std::string_view token) {
    ipc::Frame event = response_base(request, "event");
    event.header.payload_json =
        std::string("{\"event\":\"token\",\"byteCount\":") +
        std::to_string(token.size()) + "}";
    event.attachment.reserve(token.size());
    for (const unsigned char byte : token) {
        event.attachment.push_back(static_cast<std::byte>(byte));
    }
    return event;
}

ipc::Frame structured_event(const ipc::Frame& request, const AilaChatStreamEvent& value) {
    ipc::Frame event = response_base(request, "event");
    event.header.payload_json =
        std::string("{\"event\":\"structured\",\"structSize\":") +
        std::to_string(value.struct_size) +
        ",\"type\":" + std::to_string(value.type) +
        ",\"text\":" + nullable_json_string(value.text) +
        ",\"toolCallId\":" + nullable_json_string(value.tool_call_id) +
        ",\"toolName\":" + nullable_json_string(value.tool_name) +
        ",\"argumentsDelta\":" + nullable_json_string(value.arguments_delta) +
        ",\"finishReason\":" + nullable_json_string(value.finish_reason) +
        ",\"warningsJson\":" + nullable_json_string(value.warnings_json) +
        ",\"toolCallsJson\":" + nullable_json_string(value.tool_calls_json) + "}";
    return event;
}

} // namespace

WorkerDispatcher::WorkerDispatcher(std::unique_ptr<WorkerEngineApi> engine)
    : engine_(std::move(engine)) {
    if (!engine_) {
        throw std::invalid_argument("worker dispatcher requires an engine API");
    }
}

bool WorkerDispatcher::is_stream_method(std::string_view method) noexcept {
    TextGenerationMethod ignored;
    return stream_generation_method(method, ignored);
}

ipc::Frame WorkerDispatcher::dispatch_stream(
    const ipc::Frame& request,
    const WorkerStreamEmitter& emit,
    const std::atomic_bool& cancelled) {
    try {
        TextGenerationMethod method;
        if (!is_stream_method(request.header.method) ||
            !stream_generation_method(request.header.method, method)) {
            return error(request, AILA_ERR_INVALID_ARGUMENT, "unsupported streaming method");
        }
        if (request.header.kind != "request" ||
            request.header.protocol != ipc::kProtocolVersion ||
            request.header.abi != ipc::kPublicAbiVersion) {
            return error(request, AILA_ERR_INVALID_ARGUMENT, "invalid streaming request identity");
        }
        if (!initialized_) {
            return error(request, AILA_ERR_INVALID_ARGUMENT, "engine is not initialized");
        }
        if (!simdjson::validate_utf8(request.header.payload_json)) {
            return error(request, AILA_ERR_INVALID_ARGUMENT, "request payload must be valid UTF-8");
        }
        simdjson::dom::parser parser;
        simdjson::dom::element payload;
        if (parser.parse(request.header.payload_json).get(payload) != simdjson::SUCCESS) {
            return error(request, AILA_ERR_JSON_PARSE, "streaming payload JSON parse failed");
        }
        simdjson::dom::object object;
        std::string_view input;
        if (payload.get_object().get(object) != simdjson::SUCCESS ||
            object["input"].get_string().get(input) != simdjson::SUCCESS ||
            !require_c_string_safe(input) || !simdjson::validate_utf8(input)) {
            return error(request, AILA_ERR_INVALID_ARGUMENT, "streaming input must be a C-string-safe UTF-8 string");
        }
        simdjson::dom::element config_element;
        if (object.at_key("config").get(config_element) != simdjson::SUCCESS) {
            return error(request, AILA_ERR_INVALID_ARGUMENT, "streaming field 'config' is required");
        }
        TextGenerationRequest generation;
        generation.method = method;
        generation.input.assign(input.data(), input.size());
        if (!config_element.is_null()) {
            simdjson::dom::object config_object;
            if (config_element.get_object().get(config_object) != simdjson::SUCCESS) {
                return error(request, AILA_ERR_INVALID_ARGUMENT, "streaming config must be null or an object");
            }
            if (method == TextGenerationMethod::GenerateChatJsonStreamEx) {
                generation.has_v2_config = true;
                if (!parse_v2_config(config_object, generation.config_v2)) {
                    return error(request, AILA_ERR_INVALID_ARGUMENT, "streaming V2 config is malformed");
                }
            } else {
                generation.has_config = true;
                if (!parse_legacy_config(config_object, generation.config)) {
                    return error(request, AILA_ERR_INVALID_ARGUMENT, "streaming config is malformed");
                }
            }
        }

        bool emitter_failed = false;
        auto can_emit = [&] { return !cancelled.load(std::memory_order_acquire) && !emitter_failed; };
        const int status = engine_->generate_stream(
            generation,
            [&](std::string_view token) {
                if (!can_emit()) return false;
                if (token.find('\0') != std::string_view::npos ||
                    !simdjson::validate_utf8(token) || token.size() > ipc::kMaxAttachmentBytes) {
                    emitter_failed = true;
                    return false;
                }
                if (!emit(token_event(request, token))) {
                    emitter_failed = true;
                    return false;
                }
                return can_emit();
            },
            [&](const AilaChatStreamEvent& event) {
                if (!can_emit()) return false;
                if (event.struct_size != sizeof(AilaChatStreamEvent) ||
                    event.type < AILA_CHAT_STREAM_REASONING_DELTA ||
                    event.type > AILA_CHAT_STREAM_FINAL) {
                    emitter_failed = true;
                    return false;
                }
                const char* fields[] = {event.text, event.tool_call_id, event.tool_name,
                                        event.arguments_delta, event.finish_reason,
                                        event.warnings_json, event.tool_calls_json};
                for (const char* field : fields) {
                    if (field != nullptr &&
                        !simdjson::validate_utf8(std::string_view(field))) {
                        emitter_failed = true;
                        return false;
                    }
                }
                if (!emit(structured_event(request, event))) {
                    emitter_failed = true;
                    return false;
                }
                return can_emit();
            });
        if (emitter_failed) {
            return error(request, AILA_ERR_RUNTIME, "streaming event was invalid or could not be written");
        }
        if (cancelled.load(std::memory_order_acquire)) {
            return result(request, "{\"status\":1}");
        }
        if (status < -1 || status > 1) {
            return error(request, AILA_ERR_RUNTIME, "streaming adapter returned an invalid status");
        }
        if (status < 0) {
            int code = engine_->last_error_code();
            if (code == AILA_OK) code = AILA_ERR_RUNTIME;
            std::string message = engine_->last_error_message();
            if (message.empty()) message = "streaming generation failed";
            return error(request, code, message);
        }
        return result(request, std::string("{\"status\":") +
                                  std::to_string(cancelled.load() ? 1 : status) + "}");
    } catch (const std::exception& exception) {
        return error(request, AILA_ERR_RUNTIME, exception.what());
    } catch (...) {
        return error(request, AILA_ERR_RUNTIME, "unknown streaming dispatcher failure");
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
        if (!simdjson::validate_utf8(request.header.payload_json)) {
            return error(request, AILA_ERR_INVALID_ARGUMENT, "request payload must be valid UTF-8");
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

        TextGenerationMethod text_method;
        if (generation_method(request.header.method, text_method)) {
            if (!initialized_) {
                return error(request, AILA_ERR_INVALID_ARGUMENT, "engine is not initialized");
            }
            simdjson::dom::object object;
            if (payload.get_object().get(object) != simdjson::SUCCESS) {
                return error(request, AILA_ERR_INVALID_ARGUMENT, "generation payload must be an object");
            }
            std::string_view input;
            if (object["input"].get_string().get(input) != simdjson::SUCCESS) {
                return error(request, AILA_ERR_INVALID_ARGUMENT, "generation field 'input' must be a string");
            }
            if (!require_c_string_safe(input)) {
                return error(request, AILA_ERR_INVALID_ARGUMENT, "generation input must not contain an embedded NUL");
            }
            if (!simdjson::validate_utf8(input)) {
                return error(request, AILA_ERR_INVALID_ARGUMENT, "generation input must be valid UTF-8");
            }

            simdjson::dom::element config_element;
            if (object.at_key("config").get(config_element) != simdjson::SUCCESS) {
                return error(request, AILA_ERR_INVALID_ARGUMENT, "generation field 'config' is required");
            }
            TextGenerationRequest generation;
            generation.method = text_method;
            generation.input.assign(input.data(), input.size());
            if (!config_element.is_null()) {
                simdjson::dom::object config_object;
                if (config_element.get_object().get(config_object) != simdjson::SUCCESS) {
                    return error(request, AILA_ERR_INVALID_ARGUMENT, "generation config must be null or an object");
                }
                if (text_method == TextGenerationMethod::GenerateChatJsonEx) {
                    generation.has_v2_config = true;
                    if (!parse_v2_config(config_object, generation.config_v2)) {
                        return error(request, AILA_ERR_INVALID_ARGUMENT, "generation V2 config is malformed");
                    }
                } else {
                    generation.has_config = true;
                    if (!parse_legacy_config(config_object, generation.config)) {
                        return error(request, AILA_ERR_INVALID_ARGUMENT, "generation config is malformed");
                    }
                }
            }

            std::string output;
            if (!engine_->generate_text(generation, output)) {
                int code = engine_->last_error_code();
                if (code == AILA_OK) {
                    code = AILA_ERR_RUNTIME;
                }
                std::string message = engine_->last_error_message();
                if (message.empty()) {
                    message = "generation failed";
                }
                return error(request, code, message);
            }
            if (output.find('\0') != std::string::npos) {
                return error(request, AILA_ERR_RUNTIME, "generation result contained an embedded NUL");
            }
            if (!simdjson::validate_utf8(output)) {
                return error(request, AILA_ERR_RUNTIME, "generation result was not valid UTF-8");
            }
            if (output.size() > ipc::kMaxAttachmentBytes) {
                return error(request, AILA_ERR_RUNTIME, "generation result exceeded attachment limit");
            }
            return generation_result(request, output);
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
