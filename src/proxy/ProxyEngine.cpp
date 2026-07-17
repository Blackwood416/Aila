#include "proxy/ProxyEngine.hpp"

#include "aila_api.h"
#include "runtime/RuntimeDirectory.hpp"
#include "simdjson.h"

#include <windows.h>

#include <chrono>
#include <cmath>
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

std::string attachment_text(
    const std::vector<std::byte>& attachment, size_t offset, size_t count) {
    if (count == 0) return {};
    return std::string(
        reinterpret_cast<const char*>(attachment.data() + offset), count);
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

std::string nullable_c_string_json(const char* value) {
    return value ? json_string(value) : "null";
}

bool valid_optional_c_string(const char* value) {
    return value == nullptr || simdjson::validate_utf8(std::string_view(value));
}

__declspec(noinline) bool finite_float_bits(const volatile float& value) {
    const volatile unsigned char* bytes =
        reinterpret_cast<const volatile unsigned char*>(&value);
    const uint32_t bits = static_cast<uint32_t>(bytes[0]) |
        (static_cast<uint32_t>(bytes[1]) << 8) |
        (static_cast<uint32_t>(bytes[2]) << 16) |
        (static_cast<uint32_t>(bytes[3]) << 24);
    return (bits & 0x7f800000u) != 0x7f800000u;
}

bool valid_asr_legacy_config(const AilaGenConfig* config) {
    return config == nullptr ||
        (finite_float_bits(config->temperature) &&
         finite_float_bits(config->top_p) &&
         finite_float_bits(config->repetition_penalty) &&
         finite_float_bits(config->presence_penalty) &&
         finite_float_bits(config->frequency_penalty));
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
    if (stream_active_) {
        set_stream_busy_error_locked();
        return false;
    }
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
        ++worker_session_generation_;
        if (worker_session_generation_ == 0) ++worker_session_generation_;
        last_remote_asr_id_ = 0;
        active_asr_stream_ids_.clear();
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
    if (stream_active_) {
        set_stream_busy_error_locked();
        return;
    }
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
    if (stream_active_) {
        set_stream_busy_error_locked();
        return 0;
    }
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
    if (stream_active_) {
        set_stream_busy_error_locked();
        return false;
    }
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
    if (stream_active_) {
        set_stream_busy_error_locked();
        return false;
    }
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
    std::unique_lock<std::mutex> lock(mutex_);
    if (stream_active_) {
        set_stream_busy_error_locked();
        return -1;
    }
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
    stream_active_ = true;
    try {
        const int result = stream_payload_locked(
            method,
            std::string("{\"input\":") + json_string(input) +
                ",\"config\":" + legacy_config_json(config) + "}",
            callback,
            nullptr,
            user_data,
            lock);
        stream_active_ = false;
        return result;
    } catch (const std::exception& exception) {
        stream_active_ = false;
        shutdown_locked();
        set_error_locked(AILA_ERR_RUNTIME, exception.what());
        return -1;
    } catch (...) {
        stream_active_ = false;
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
    std::unique_lock<std::mutex> lock(mutex_);
    if (stream_active_) {
        set_stream_busy_error_locked();
        return -1;
    }
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
    stream_active_ = true;
    try {
        const int result = stream_payload_locked(
            method,
            std::string("{\"input\":") + json_string(input) +
                ",\"config\":" + v2_config_json(config) + "}",
            nullptr,
            callback,
            user_data,
            lock);
        stream_active_ = false;
        return result;
    } catch (const std::exception& exception) {
        stream_active_ = false;
        shutdown_locked();
        set_error_locked(AILA_ERR_RUNTIME, exception.what());
        return -1;
    } catch (...) {
        stream_active_ = false;
        shutdown_locked();
        set_error_locked(AILA_ERR_RUNTIME, "unknown proxy structured streaming failure");
        return -1;
    }
}

bool ProxyEngine::transcribe(
    std::string_view wav_path,
    const AilaGenConfig* config,
    const char* forced_language,
    const char* system_prompt,
    float segment_sec,
    int past_text_conditioning,
    AilaTokenCallback callback,
    void* user_data,
    std::string& transcript,
    std::string& language) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (stream_active_) {
        set_stream_busy_error_locked();
        return false;
    }
    if (!initialized_ || wav_path.empty() || wav_path.find('\0') != std::string_view::npos ||
        !simdjson::validate_utf8(wav_path) || !valid_optional_c_string(forced_language) ||
        !valid_optional_c_string(system_prompt) || !valid_asr_legacy_config(config) ||
        !finite_float_bits(segment_sec) || segment_sec < 0.0f) {
        set_error_locked(AILA_ERR_INVALID_ARGUMENT, "ASR transcription arguments are invalid");
        return false;
    }
    stream_active_ = true;
    try {
        ipc::Frame request;
        request.header.protocol = ipc::kProtocolVersion;
        request.header.abi = ipc::kPublicAbiVersion;
        request.header.request_id = next_request_id_++;
        if (next_request_id_ == 0 || next_request_id_ == (std::numeric_limits<uint64_t>::max)()) {
            next_request_id_ = 1;
        }
        request.header.kind = "request";
        request.header.method = "asr.transcribe";
        request.header.payload_json =
            std::string("{\"wavPath\":") + json_string(wav_path) +
            ",\"config\":" + legacy_config_json(config) +
            ",\"forcedLanguage\":" + nullable_c_string_json(forced_language) +
            ",\"systemPrompt\":" + nullable_c_string_json(system_prompt) +
            ",\"segmentSec\":" + json_float(segment_sec) +
            ",\"pastTextConditioning\":" + std::to_string(past_text_conditioning) + "}";
        bool terminal_seen = false;
        const ipc::Frame response = worker_.request_stream(
            request,
            [&](const ipc::Frame& event) {
                if (event.header.protocol != ipc::kProtocolVersion ||
                    event.header.abi != ipc::kPublicAbiVersion ||
                    event.header.kind != "event" || event.header.method != "asr.transcribe" ||
                    event.header.request_id != request.header.request_id || terminal_seen) {
                    throw malformed_response("ASR token event identity was invalid");
                }
                simdjson::dom::parser parser;
                simdjson::dom::element root;
                simdjson::dom::object object;
                std::string_view kind;
                if (!simdjson::validate_utf8(event.header.payload_json) ||
                    parser.parse(event.header.payload_json).get(root) != simdjson::SUCCESS ||
                    root.get_object().get(object) != simdjson::SUCCESS ||
                    object["event"].get_string().get(kind) != simdjson::SUCCESS) {
                    throw malformed_response("ASR token event schema was invalid");
                }
                size_t fields = 0;
                for (auto field : object) { (void)field; ++fields; }
                if (kind == "end") {
                    uint64_t count = 0;
                    if (fields != 2 || !event.attachment.empty() ||
                        object["eventCount"].get_uint64().get(count) != simdjson::SUCCESS ||
                        count == 0 || count > ipc::kMaxStreamEventCount) {
                        throw malformed_response("ASR terminal event schema was invalid");
                    }
                    terminal_seen = true;
                    return runtime::WorkerProcess::StreamEventAction::End;
                }
                uint64_t bytes = 0;
                if (kind != "token" || fields != 2 ||
                    object["byteCount"].get_uint64().get(bytes) != simdjson::SUCCESS ||
                    bytes != event.attachment.size()) {
                    throw malformed_response("ASR token event attachment was invalid");
                }
                const std::string token = attachment_text(
                    event.attachment, 0, event.attachment.size());
                if (token.find('\0') != std::string::npos || !simdjson::validate_utf8(token)) {
                    throw malformed_response("ASR token event was not valid UTF-8");
                }
                if (callback) {
                    lock.unlock();
                    try { (void)callback(token.c_str(), user_data); }
                    catch (...) { lock.lock(); throw; }
                    lock.lock();
                }
                return runtime::WorkerProcess::StreamEventAction::Continue;
            },
            std::chrono::minutes(10));
        flush_deferred_asr_destroys_locked();
        stream_active_ = false;
        if (response.header.kind == "error") {
            accept_stream_error_response_locked(response, "asr.transcribe", "ASR transcription failed");
            return false;
        }
        simdjson::dom::parser parser;
        simdjson::dom::element root;
        simdjson::dom::object object;
        uint64_t transcript_bytes = 0, language_bytes = 0, event_count = 0;
        if (response.header.kind != "result" || response.header.method != "asr.transcribe" ||
            parser.parse(response.header.payload_json).get(root) != simdjson::SUCCESS ||
            root.get_object().get(object) != simdjson::SUCCESS ||
            object["transcriptBytes"].get_uint64().get(transcript_bytes) != simdjson::SUCCESS ||
            object["languageBytes"].get_uint64().get(language_bytes) != simdjson::SUCCESS ||
            object["eventCount"].get_uint64().get(event_count) != simdjson::SUCCESS ||
            transcript_bytes > response.attachment.size() ||
            language_bytes != response.attachment.size() - transcript_bytes || !terminal_seen) {
            throw malformed_response("ASR result lengths were invalid");
        }
        size_t fields = 0;
        for (auto field : object) { (void)field; ++fields; }
        if (fields != 3) throw malformed_response("ASR result contained unexpected fields");
        transcript = attachment_text(
            response.attachment, 0, static_cast<size_t>(transcript_bytes));
        language = attachment_text(
            response.attachment, static_cast<size_t>(transcript_bytes),
            static_cast<size_t>(language_bytes));
        if (transcript.find('\0') != std::string::npos || language.find('\0') != std::string::npos ||
            !simdjson::validate_utf8(transcript) || !simdjson::validate_utf8(language)) {
            throw malformed_response("ASR result strings were invalid");
        }
        clear_error_locked();
        return true;
    } catch (const std::exception& exception) {
        stream_active_ = false;
        shutdown_locked();
        set_error_locked(AILA_ERR_RUNTIME, exception.what());
        return false;
    } catch (...) {
        stream_active_ = false;
        shutdown_locked();
        set_error_locked(AILA_ERR_RUNTIME, "unknown ASR transcription failure");
        return false;
    }
}

bool ProxyEngine::transcribe_stream_create(
    const AilaGenConfig* config,
    const char* forced_language,
    const char* system_prompt,
    uint64_t& worker_session,
    uint64_t& stream_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    worker_session = 0;
    stream_id = 0;
    if (stream_active_) { set_stream_busy_error_locked(); return false; }
    if (!initialized_ || !valid_asr_legacy_config(config) ||
        !valid_optional_c_string(forced_language) ||
        !valid_optional_c_string(system_prompt)) {
        set_error_locked(AILA_ERR_INVALID_ARGUMENT, "ASR stream arguments are invalid");
        return false;
    }
    try {
        const ipc::Frame response = request_locked(
            "asr.stream.create",
            std::string("{\"config\":") + legacy_config_json(config) +
                ",\"forcedLanguage\":" + nullable_c_string_json(forced_language) +
                ",\"systemPrompt\":" + nullable_c_string_json(system_prompt) + "}");
        if (response.header.kind == "error") {
            return accept_error_response_locked(response, "asr.stream.create", "ASR stream creation failed");
        }
        simdjson::dom::parser parser;
        simdjson::dom::element root;
        uint64_t id = 0;
        if (response.header.kind != "result" || response.header.method != "asr.stream.create" ||
            !response.attachment.empty() ||
            parser.parse(response.header.payload_json).get(root) != simdjson::SUCCESS ||
            root["streamId"].get_uint64().get(id) != simdjson::SUCCESS || id == 0 ||
            id <= last_remote_asr_id_) {
            throw malformed_response("ASR stream create response was invalid");
        }
        simdjson::dom::object object;
        if (root.get_object().get(object) != simdjson::SUCCESS) {
            throw malformed_response("ASR stream create result was not an object");
        }
        size_t fields = 0;
        for (auto field : object) { (void)field; ++fields; }
        if (fields != 1) throw malformed_response("ASR stream create result contained extra fields");
        for (const uint64_t active_id : active_asr_stream_ids_) {
            if (active_id == id) {
                throw malformed_response("ASR stream create reused an active remote ID");
            }
        }
        active_asr_stream_ids_.push_back(id);
        last_remote_asr_id_ = id;
        worker_session = worker_session_generation_;
        stream_id = id;
        clear_error_locked();
        return true;
    } catch (const std::exception& exception) {
        shutdown_locked(); set_error_locked(AILA_ERR_RUNTIME, exception.what()); return false;
    } catch (...) {
        shutdown_locked(); set_error_locked(AILA_ERR_RUNTIME, "unknown ASR stream creation failure"); return false;
    }
}

int ProxyEngine::transcribe_stream_feed(
    uint64_t worker_session, uint64_t stream_id, const float* samples, int sample_count) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stream_active_) { set_stream_busy_error_locked(); return AILA_ERR_INVALID_ARGUMENT; }
    if (!asr_stream_is_active_locked(worker_session, stream_id) ||
        !samples || sample_count <= 0 ||
        static_cast<size_t>(sample_count) > ipc::kMaxAttachmentBytes / sizeof(float)) {
        set_error_locked(AILA_ERR_INVALID_ARGUMENT, "ASR stream audio arguments are invalid");
        return AILA_ERR_INVALID_ARGUMENT;
    }
    try {
        const size_t bytes = static_cast<size_t>(sample_count) * sizeof(float);
        std::vector<std::byte> attachment(bytes);
        std::memcpy(attachment.data(), samples, bytes);
        const ipc::Frame response = request_locked(
            "asr.stream.feed",
            "{\"streamId\":" + std::to_string(stream_id) +
                ",\"sampleCount\":" + std::to_string(sample_count) +
                ",\"elementSize\":" + std::to_string(sizeof(float)) +
                ",\"byteCount\":" + std::to_string(bytes) + "}",
            std::move(attachment));
        if (response.header.kind == "error") {
            accept_error_response_locked(response, "asr.stream.feed", "ASR stream feed failed");
            return error_code_;
        }
        simdjson::dom::parser parser;
        simdjson::dom::element root;
        simdjson::dom::object object;
        bool ok = false;
        if (response.header.kind != "result" || response.header.method != "asr.stream.feed" ||
            !response.attachment.empty() ||
            parser.parse(response.header.payload_json).get(root) != simdjson::SUCCESS ||
            root.get_object().get(object) != simdjson::SUCCESS ||
            object["ok"].get_bool().get(ok) != simdjson::SUCCESS || !ok) {
            throw malformed_response("ASR stream feed response was invalid");
        }
        size_t fields = 0;
        for (auto field : object) { (void)field; ++fields; }
        if (fields != 1) throw malformed_response("ASR stream feed response contained extra fields");
        clear_error_locked();
        return AILA_OK;
    } catch (const std::exception& exception) {
        shutdown_locked(); set_error_locked(AILA_ERR_RUNTIME, exception.what()); return AILA_ERR_RUNTIME;
    } catch (...) {
        shutdown_locked(); set_error_locked(AILA_ERR_RUNTIME, "unknown ASR stream feed failure"); return AILA_ERR_RUNTIME;
    }
}

int ProxyEngine::transcribe_stream_get_text(
    uint64_t worker_session, uint64_t stream_id,
    std::string& stable, std::string& partial) {
    std::lock_guard<std::mutex> lock(mutex_);
    stable.clear(); partial.clear();
    if (stream_active_) { set_stream_busy_error_locked(); return AILA_ERR_INVALID_ARGUMENT; }
    if (!asr_stream_is_active_locked(worker_session, stream_id)) {
        set_error_locked(AILA_ERR_INVALID_ARGUMENT, "ASR stream handle is invalid");
        return AILA_ERR_INVALID_ARGUMENT;
    }
    try {
        const ipc::Frame response = request_locked(
            "asr.stream.get_text", "{\"streamId\":" + std::to_string(stream_id) + "}");
        if (response.header.kind == "error") {
            accept_error_response_locked(response, "asr.stream.get_text", "ASR stream get_text failed");
            return error_code_;
        }
        simdjson::dom::parser parser;
        simdjson::dom::element root;
        uint64_t stable_bytes = 0, partial_bytes = 0;
        if (response.header.kind != "result" || response.header.method != "asr.stream.get_text" ||
            parser.parse(response.header.payload_json).get(root) != simdjson::SUCCESS ||
            root["stableBytes"].get_uint64().get(stable_bytes) != simdjson::SUCCESS ||
            root["partialBytes"].get_uint64().get(partial_bytes) != simdjson::SUCCESS ||
            stable_bytes > response.attachment.size() ||
            partial_bytes != response.attachment.size() - stable_bytes) {
            throw malformed_response("ASR stream text lengths were invalid");
        }
        simdjson::dom::object object;
        if (root.get_object().get(object) != simdjson::SUCCESS) {
            throw malformed_response("ASR stream text result was not an object");
        }
        size_t fields = 0;
        for (auto field : object) { (void)field; ++fields; }
        if (fields != 2) throw malformed_response("ASR stream text result contained extra fields");
        stable = attachment_text(response.attachment, 0, static_cast<size_t>(stable_bytes));
        partial = attachment_text(
            response.attachment, static_cast<size_t>(stable_bytes),
            static_cast<size_t>(partial_bytes));
        if (stable.find('\0') != std::string::npos || partial.find('\0') != std::string::npos ||
            !simdjson::validate_utf8(stable) || !simdjson::validate_utf8(partial)) {
            throw malformed_response("ASR stream text was invalid UTF-8");
        }
        clear_error_locked();
        return AILA_OK;
    } catch (const std::exception& exception) {
        shutdown_locked(); set_error_locked(AILA_ERR_RUNTIME, exception.what()); return AILA_ERR_RUNTIME;
    } catch (...) {
        shutdown_locked(); set_error_locked(AILA_ERR_RUNTIME, "unknown ASR stream get_text failure"); return AILA_ERR_RUNTIME;
    }
}

void ProxyEngine::transcribe_stream_destroy(
    uint64_t worker_session, uint64_t stream_id) noexcept {
    if (worker_session == 0 || stream_id == 0) return;
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!asr_stream_is_active_locked(worker_session, stream_id)) {
            if (!stream_active_) {
                set_error_locked(AILA_ERR_INVALID_ARGUMENT, "ASR stream handle is stale or invalid");
            }
            return;
        }
        remove_asr_stream_locked(stream_id);
        if (stream_active_) {
            for (const uint64_t deferred_id : deferred_asr_destroy_ids_) {
                if (deferred_id == stream_id) return;
            }
            deferred_asr_destroy_ids_.push_back(stream_id);
            return;
        }
        destroy_remote_asr_stream_locked(stream_id);
        clear_error_locked();
    } catch (const std::exception& exception) {
        std::lock_guard<std::mutex> lock(mutex_);
        shutdown_locked(); set_error_locked(AILA_ERR_RUNTIME, exception.what());
    } catch (...) {
        try { std::lock_guard<std::mutex> lock(mutex_); shutdown_locked(); }
        catch (...) {}
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
    if (stream_active_) {
        set_stream_busy_error_locked();
        return;
    }
    set_error_locked(AILA_ERR_INVALID_ARGUMENT, std::move(message));
}

void ProxyEngine::record_runtime_error(std::string message) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stream_active_) {
        set_stream_busy_error_locked();
        return;
    }
    set_error_locked(AILA_ERR_RUNTIME, std::move(message));
}

ipc::Frame ProxyEngine::request_locked(std::string method, std::string payload_json) {
    return request_locked(std::move(method), std::move(payload_json), {});
}

ipc::Frame ProxyEngine::request_locked(
    std::string method,
    std::string payload_json,
    std::vector<std::byte> attachment) {
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
    request.attachment = std::move(attachment);
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

bool ProxyEngine::accept_stream_error_response_locked(
    const ipc::Frame& response,
    std::string_view expected_method,
    std::string_view fallback_message) {
    if (response.header.method != expected_method || response.header.kind != "error" ||
        !response.attachment.empty()) {
        throw malformed_response("stream error response identity was invalid");
    }
    simdjson::dom::parser parser;
    simdjson::dom::element payload;
    simdjson::dom::object object;
    int64_t code = 0;
    uint64_t event_count = 0;
    std::string_view message;
    if (parser.parse(response.header.payload_json).get(payload) != simdjson::SUCCESS ||
        payload.get_object().get(object) != simdjson::SUCCESS ||
        object["code"].get_int64().get(code) != simdjson::SUCCESS ||
        object["message"].get_string().get(message) != simdjson::SUCCESS ||
        object["eventCount"].get_uint64().get(event_count) != simdjson::SUCCESS ||
        code < AILA_OK || code > (std::numeric_limits<int>::max)() ||
        event_count == 0 || event_count > ipc::kMaxStreamEventCount) {
        throw malformed_response(
            "stream error payload requires code, message, and bounded eventCount");
    }
    size_t field_count = 0;
    for (auto field : object) { (void)field; ++field_count; }
    if (field_count != 3) {
        throw malformed_response("stream error payload contained unexpected fields");
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
    void* user_data,
    std::unique_lock<std::mutex>& engine_lock) {
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
    bool terminal_seen = false;
    const ipc::Frame response = worker_.request_stream(
        request,
        [&](const ipc::Frame& event) {
            if (event.header.kind != "event" || event.header.method != method ||
                event.header.request_id != request.header.request_id ||
                event.header.protocol != ipc::kProtocolVersion ||
                event.header.abi != ipc::kPublicAbiVersion) {
                throw malformed_response("stream event identity did not match request");
            }
            if (terminal_seen) {
                throw malformed_response("stream emitted data or a duplicate end after terminal event");
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
                uint64_t terminal_count = 0;
                if (field_count != 2 || !event.attachment.empty() ||
                    object["eventCount"].get_uint64().get(terminal_count) !=
                        simdjson::SUCCESS ||
                    terminal_count == 0 || terminal_count > ipc::kMaxStreamEventCount) {
                    throw malformed_response("stream terminal event schema was invalid");
                }
                terminal_seen = true;
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
                const std::string token = attachment_text(
                    event.attachment, 0, event.attachment.size());
                if (!simdjson::validate_utf8(token)) {
                    throw malformed_response("token stream event was not valid UTF-8");
                }
                int callback_result = 0;
                if (!aborted) {
                    engine_lock.unlock();
                    try {
                        callback_result = token_callback(token.c_str(), user_data);
                    } catch (...) {
                        engine_lock.lock();
                        throw;
                    }
                    engine_lock.lock();
                }
                if (!aborted && callback_result != 0) {
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
            int callback_result = 0;
            if (!aborted) {
                engine_lock.unlock();
                try {
                    callback_result = structured_callback(&abi_event, user_data);
                } catch (...) {
                    engine_lock.lock();
                    throw;
                }
                engine_lock.lock();
            }
            if (!aborted && callback_result != 0) {
                aborted = true;
                return runtime::WorkerProcess::StreamEventAction::Cancel;
            }
            return runtime::WorkerProcess::StreamEventAction::Continue;
        },
        std::chrono::minutes(10));
    flush_deferred_asr_destroys_locked();
    if (response.header.kind == "error") {
        accept_stream_error_response_locked(
            response, method, "worker streaming generation failed");
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
    uint64_t event_count = 0;
    if (parser.parse(response.header.payload_json).get(root) != simdjson::SUCCESS ||
        root.get_object().get(object) != simdjson::SUCCESS ||
        object["status"].get_int64().get(status) != simdjson::SUCCESS ||
        object["eventCount"].get_uint64().get(event_count) != simdjson::SUCCESS ||
        event_count == 0 || event_count > ipc::kMaxStreamEventCount ||
        status < 0 || status > 1) {
        throw malformed_response("stream result status was invalid");
    }
    size_t field_count = 0;
    for (auto field : object) { (void)field; ++field_count; }
    if (field_count != 2 || !terminal_seen || (!aborted && status != 0)) {
        throw malformed_response("stream result cancellation status did not match the host callback");
    }
    clear_error_locked();
    return aborted ? 1 : 0;
}

void ProxyEngine::set_error_locked(int code, std::string message) {
    error_code_ = code;
    error_message_ = std::move(message);
}

void ProxyEngine::set_stream_busy_error_locked() {
    set_error_locked(
        AILA_ERR_INVALID_ARGUMENT,
        "engine operation is unavailable while a streaming callback is active");
}

void ProxyEngine::clear_error_locked() {
    error_code_ = AILA_OK;
    error_message_.clear();
}

bool ProxyEngine::asr_stream_is_active_locked(
    uint64_t worker_session, uint64_t stream_id) const {
    if (!initialized_ || worker_session == 0 ||
        worker_session != worker_session_generation_ || stream_id == 0) {
        return false;
    }
    for (const uint64_t active_id : active_asr_stream_ids_) {
        if (active_id == stream_id) return true;
    }
    return false;
}

void ProxyEngine::remove_asr_stream_locked(uint64_t stream_id) {
    for (auto iterator = active_asr_stream_ids_.begin();
         iterator != active_asr_stream_ids_.end(); ++iterator) {
        if (*iterator == stream_id) {
            active_asr_stream_ids_.erase(iterator);
            return;
        }
    }
}

void ProxyEngine::destroy_remote_asr_stream_locked(uint64_t stream_id) {
    const ipc::Frame response = request_locked(
        "asr.stream.destroy", "{\"streamId\":" + std::to_string(stream_id) + "}");
    if (response.header.kind == "error") {
        (void)accept_error_response_locked(
            response, "asr.stream.destroy", "ASR stream destroy failed");
        throw std::runtime_error(error_message_.empty()
                                     ? "ASR stream destroy failed"
                                     : error_message_);
    }
    simdjson::dom::parser parser;
    simdjson::dom::element root;
    simdjson::dom::object object;
    bool ok = false;
    if (response.header.kind != "result" || response.header.method != "asr.stream.destroy" ||
        !response.attachment.empty() ||
        parser.parse(response.header.payload_json).get(root) != simdjson::SUCCESS ||
        root.get_object().get(object) != simdjson::SUCCESS ||
        object["ok"].get_bool().get(ok) != simdjson::SUCCESS || !ok) {
        throw malformed_response("ASR stream destroy response was invalid");
    }
    size_t fields = 0;
    for (auto field : object) { (void)field; ++fields; }
    if (fields != 1) {
        throw malformed_response("ASR stream destroy response contained extra fields");
    }
}

void ProxyEngine::flush_deferred_asr_destroys_locked() {
    std::vector<uint64_t> pending;
    pending.swap(deferred_asr_destroy_ids_);
    for (const uint64_t stream_id : pending) {
        destroy_remote_asr_stream_locked(stream_id);
    }
}

void ProxyEngine::shutdown_locked() noexcept {
    initialized_ = false;
    active_asr_stream_ids_.clear();
    deferred_asr_destroy_ids_.clear();
    last_remote_asr_id_ = 0;
    try {
        worker_.shutdown(std::chrono::seconds(2));
    } catch (...) {
    }
}

} // namespace aila::proxy
