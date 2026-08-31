#include "worker/WorkerDispatcher.hpp"

#include "aila_api.h"
#include "simdjson.h"

#include <exception>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <iomanip>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <unordered_set>

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

std::string json_float(float value) {
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << std::setprecision((std::numeric_limits<float>::max_digits10)) << value;
    return stream.str();
}

std::string detection_result_json(const std::vector<DetectionResult>& detections) {
    std::string output = "{\"detections\":[";
    for (size_t index = 0; index < detections.size(); ++index) {
        const auto& value = detections[index];
        if (index) output += ',';
        output += "{\"x1\":" + json_float(value.x1) +
                  ",\"y1\":" + json_float(value.y1) +
                  ",\"x2\":" + json_float(value.x2) +
                  ",\"y2\":" + json_float(value.y2) +
                  ",\"confidence\":" + json_float(value.confidence) +
                  ",\"classId\":" + std::to_string(value.class_id) +
                  ",\"className\":" + json_string(value.class_name) + "}";
    }
    output += "]}";
    return output;
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

bool optional_string_field(
    simdjson::dom::object object,
    const char* name,
    bool& present,
    std::string& value) {
    simdjson::dom::element element;
    if (object.at_key(name).get(element) != simdjson::SUCCESS) return false;
    if (element.is_null()) {
        present = false;
        value.clear();
        return true;
    }
    std::string_view text;
    if (element.get_string().get(text) != simdjson::SUCCESS ||
        !require_c_string_safe(text) || !simdjson::validate_utf8(text)) return false;
    present = true;
    value.assign(text.data(), text.size());
    return true;
}

bool parse_optional_legacy_config(
    simdjson::dom::object object,
    bool& has_config,
    AilaGenConfig& config) {
    simdjson::dom::element element;
    if (object.at_key("config").get(element) != simdjson::SUCCESS) return false;
    if (element.is_null()) {
        has_config = false;
        config = {};
        return true;
    }
    simdjson::dom::object config_object;
    if (element.get_object().get(config_object) != simdjson::SUCCESS ||
        !parse_legacy_config(config_object, config)) return false;
    has_config = true;
    return true;
}

ipc::Frame two_text_result(
    const ipc::Frame& request,
    std::string_view first_name,
    std::string_view first,
    std::string_view second_name,
    std::string_view second,
    std::string suffix = {}) {
    ipc::Frame response = result(
        request,
        std::string("{\"") + std::string(first_name) + "\":" +
            std::to_string(first.size()) + ",\"" + std::string(second_name) +
            "\":" + std::to_string(second.size()) + suffix + "}");
    response.attachment.reserve(first.size() + second.size());
    for (unsigned char byte : first) response.attachment.push_back(static_cast<std::byte>(byte));
    for (unsigned char byte : second) response.attachment.push_back(static_cast<std::byte>(byte));
    return response;
}

ipc::Frame float_result(const ipc::Frame& request, const std::vector<float>& values) {
    ipc::Frame response = result(
        request,
        std::string("{\"sampleCount\":") + std::to_string(values.size()) +
            ",\"elementSize\":" + std::to_string(sizeof(float)) +
            ",\"byteCount\":" + std::to_string(values.size() * sizeof(float)) + "}");
    response.attachment.resize(values.size() * sizeof(float));
    if (!values.empty()) std::memcpy(response.attachment.data(), values.data(), response.attachment.size());
    return response;
}

ipc::Frame alignment_result(
    const ipc::Frame& request,
    const std::vector<AlignedWordResult>& words) {
    if (words.size() > static_cast<size_t>((std::numeric_limits<int>::max)())) {
        throw std::runtime_error("alignment result contains too many words");
    }
    size_t text_bytes = 0;
    std::string entries = "[";
    for (size_t index = 0; index < words.size(); ++index) {
        const AlignedWordResult& word = words[index];
        if (!require_c_string_safe(word.text) || !simdjson::validate_utf8(word.text) ||
            word.text.size() > ipc::kMaxAttachmentBytes - text_bytes) {
            throw std::runtime_error("alignment result text is malformed");
        }
        if (index != 0) entries += ',';
        entries += std::string("{\"textOffset\":") + std::to_string(text_bytes) +
            ",\"textBytes\":" + std::to_string(word.text.size()) +
            ",\"startMs\":" + std::to_string(word.start_ms) +
            ",\"endMs\":" + std::to_string(word.end_ms) + "}";
        text_bytes += word.text.size();
    }
    entries += ']';
    ipc::Frame response = result(
        request,
        std::string("{\"wordCount\":") + std::to_string(words.size()) +
            ",\"textByteCount\":" + std::to_string(text_bytes) +
            ",\"words\":" + entries + "}");
    size_t header_bytes = 0;
    if (!ipc::encoded_header_size(response.header, header_bytes)) {
        ipc::Frame failure = error(
            request,
            AILA_ERR_RUNTIME,
            "alignment result metadata exceeded the IPC header limit");
        size_t failure_header_bytes = 0;
        if (!ipc::encoded_header_size(failure.header, failure_header_bytes)) {
            throw std::runtime_error("bounded alignment error header could not be encoded");
        }
        return failure;
    }
    response.attachment.reserve(text_bytes);
    for (const AlignedWordResult& word : words) {
        for (const unsigned char byte : word.text) {
            response.attachment.push_back(static_cast<std::byte>(byte));
        }
    }
    return response;
}

ipc::Frame audio_event(const ipc::Frame& request, const float* samples, size_t count) {
    ipc::Frame event = response_base(request, "event");
    event.header.payload_json =
        std::string("{\"event\":\"audio\",\"sampleCount\":") + std::to_string(count) +
        ",\"elementSize\":" + std::to_string(sizeof(float)) +
        ",\"byteCount\":" + std::to_string(count * sizeof(float)) + "}";
    event.attachment.resize(count * sizeof(float));
    if (count != 0) std::memcpy(event.attachment.data(), samples, event.attachment.size());
    return event;
}

bool finite_values(const std::vector<float>& values) {
    for (const float value : values) {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        if ((bits & 0x7f800000u) == 0x7f800000u) return false;
    }
    return true;
}

bool parse_tts_text_request(simdjson::dom::object object, AudioRequest& audio) {
    std::string_view text;
    if (object["text"].get_string().get(text) != simdjson::SUCCESS ||
        !require_c_string_safe(text) || !simdjson::validate_utf8(text) ||
        !optional_string_field(object, "referenceAudioPath", audio.has_reference_audio_path,
                               audio.reference_audio_path) ||
        !optional_string_field(object, "speakerName", audio.has_speaker_name,
                               audio.speaker_name) ||
        !optional_string_field(object, "instructText", audio.has_instruct_text,
                               audio.instruct_text) ||
        !optional_string_field(object, "language", audio.has_language, audio.language) ||
        (object.at_key("referenceText").error() == simdjson::SUCCESS && !optional_string_field(object, "referenceText", audio.has_reference_text, audio.reference_text)) ||
        !parse_optional_legacy_config(object, audio.has_config, audio.config)) return false;
    int64_t mode_val = 0;
    if (object["voiceCloneMode"].get_int64().get(mode_val) == simdjson::SUCCESS) {
        audio.voice_clone_mode = static_cast<int>(mode_val);
    }
    audio.text.assign(text.data(), text.size());
    return true;
}

} // namespace

WorkerDispatcher::WorkerDispatcher(std::unique_ptr<WorkerEngineApi> engine)
    : engine_(std::move(engine)) {
    if (!engine_) {
        throw std::invalid_argument("worker dispatcher requires an engine API");
    }
}

WorkerDispatcher::~WorkerDispatcher() noexcept {
    for (const uint64_t id : asr_stream_ids_) {
        engine_->transcribe_stream_destroy(id);
    }
}

bool WorkerDispatcher::is_stream_method(std::string_view method) noexcept {
    TextGenerationMethod ignored;
    return method == "asr.transcribe" || method == "tts.synthesize_stream" ||
        stream_generation_method(method, ignored);
}

ipc::Frame WorkerDispatcher::dispatch_stream(
    const ipc::Frame& request,
    const WorkerStreamEmitter& emit,
    const std::atomic_bool& cancelled) {
    try {
        if (request.header.method == "tts.synthesize_stream") {
            if (request.header.kind != "request" ||
                request.header.protocol != ipc::kProtocolVersion ||
                request.header.abi != ipc::kPublicAbiVersion || !initialized_ ||
                !request.attachment.empty() ||
                !simdjson::validate_utf8(request.header.payload_json)) {
                return error(request, AILA_ERR_INVALID_ARGUMENT, "invalid TTS stream request");
            }
            simdjson::dom::parser parser;
            simdjson::dom::element root;
            simdjson::dom::object object;
            AudioRequest audio;
            audio.method = AudioMethod::SynthesizeStream;
            if (parser.parse(request.header.payload_json).get(root) != simdjson::SUCCESS ||
                root.get_object().get(object) != simdjson::SUCCESS ||
                !parse_tts_text_request(object, audio)) {
                return error(request, AILA_ERR_INVALID_ARGUMENT, "TTS stream payload is malformed");
            }
            bool event_failed = false;
            uint64_t event_count = 0;
            const int status = engine_->synthesize_stream(
                audio,
                [&](const float* samples, size_t count) {
                    if (cancelled.load(std::memory_order_acquire)) return false;
                    if (!samples || count == 0 ||
                        count > ipc::kMaxAttachmentBytes / sizeof(float) ||
                        !detail::stream_data_event_can_emit(event_count)) {
                        event_failed = true;
                        return false;
                    }
                    for (size_t index = 0; index < count; ++index) {
                        uint32_t bits = 0;
                        std::memcpy(&bits, &samples[index], sizeof(bits));
                        if ((bits & 0x7f800000u) == 0x7f800000u) {
                            event_failed = true; return false;
                        }
                    }
                    if (!emit(audio_event(request, samples, count))) {
                        event_failed = true;
                        return false;
                    }
                    ++event_count;
                    return !cancelled.load(std::memory_order_acquire);
                });
            if (event_failed) {
                return error(request, AILA_ERR_RUNTIME, "TTS stream emitted invalid audio");
            }
            if (status < -1 || status > 1) {
                return error(request, AILA_ERR_RUNTIME, "TTS stream adapter returned invalid status");
            }
            if (status < 0) {
                int code = engine_->last_error_code();
                if (code == AILA_OK) code = AILA_ERR_RUNTIME;
                return error(request, code, engine_->last_error_message());
            }
            return result(request, std::string("{\"status\":") +
                std::to_string(cancelled.load() ? 1 : status) +
                ",\"eventCount\":" + std::to_string(event_count + 1) + "}");
        }
        if (request.header.method == "asr.transcribe") {
            if (request.header.kind != "request" ||
                request.header.protocol != ipc::kProtocolVersion ||
                request.header.abi != ipc::kPublicAbiVersion || !initialized_ ||
                !request.attachment.empty() ||
                !simdjson::validate_utf8(request.header.payload_json)) {
                return error(request, AILA_ERR_INVALID_ARGUMENT, "invalid ASR transcription request");
            }
            simdjson::dom::parser parser;
            simdjson::dom::element root;
            simdjson::dom::object object;
            std::string_view wav_path;
            double segment = 0.0;
            int past = 0;
            AsrRequest asr;
            if (parser.parse(request.header.payload_json).get(root) != simdjson::SUCCESS ||
                root.get_object().get(object) != simdjson::SUCCESS ||
                object["wavPath"].get_string().get(wav_path) != simdjson::SUCCESS ||
                wav_path.empty() || !require_c_string_safe(wav_path) ||
                !simdjson::validate_utf8(wav_path) ||
                object["segmentSec"].get_double().get(segment) != simdjson::SUCCESS ||
                !std::isfinite(segment) || segment < 0.0 ||
                segment > (std::numeric_limits<float>::max)() ||
                !object_integer(object, "pastTextConditioning", past) ||
                !parse_optional_legacy_config(object, asr.has_config, asr.config) ||
                !optional_string_field(object, "forcedLanguage", asr.has_forced_language,
                                       asr.forced_language) ||
                !optional_string_field(object, "systemPrompt", asr.has_system_prompt,
                                       asr.system_prompt)) {
                return error(request, AILA_ERR_INVALID_ARGUMENT, "ASR transcription payload is malformed");
            }
            asr.wav_path.assign(wav_path.data(), wav_path.size());
            asr.segment_sec = static_cast<float>(segment);
            asr.past_text_conditioning = past;
            bool event_failed = false;
            uint64_t event_count = 0;
            std::string transcript;
            std::string language;
            const bool ok = engine_->transcribe(
                asr,
                [&](std::string_view token) {
                    if (event_failed || cancelled.load(std::memory_order_acquire) ||
                        token.find('\0') != std::string_view::npos ||
                        !simdjson::validate_utf8(token) ||
                        !detail::stream_data_event_can_emit(event_count) ||
                        !emit(token_event(request, token))) {
                        event_failed = true;
                        return;
                    }
                    ++event_count;
                },
                transcript,
                language);
            if (event_failed) {
                return error(request, AILA_ERR_RUNTIME, "ASR token event was invalid or could not be written");
            }
            if (!ok) {
                int code = engine_->last_error_code();
                if (code == AILA_OK) code = AILA_ERR_RUNTIME;
                std::string message = engine_->last_error_message();
                return error(request, code, message.empty() ? "ASR transcription failed" : message);
            }
            if (transcript.find('\0') != std::string::npos ||
                language.find('\0') != std::string::npos ||
                !simdjson::validate_utf8(transcript) || !simdjson::validate_utf8(language) ||
                language.size() > ipc::kMaxAttachmentBytes ||
                transcript.size() > ipc::kMaxAttachmentBytes - language.size()) {
                return error(request, AILA_ERR_RUNTIME, "ASR result was invalid");
            }
            return two_text_result(
                request, "transcriptBytes", transcript, "languageBytes", language,
                ",\"eventCount\":" + std::to_string(event_count + 1));
        }
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
        uint64_t event_count = 0;
        auto can_emit = [&] { return !cancelled.load(std::memory_order_acquire) && !emitter_failed; };
        auto emit_counted = [&](const ipc::Frame& event) {
            if (!detail::stream_data_event_can_emit(event_count)) {
                emitter_failed = true;
                return false;
            }
            if (!emit(event)) return false;
            ++event_count;
            return true;
        };
        const int status = engine_->generate_stream(
            generation,
            [&](std::string_view token) {
                if (!can_emit()) return false;
                if (token.find('\0') != std::string_view::npos ||
                    !simdjson::validate_utf8(token) || token.size() > ipc::kMaxAttachmentBytes) {
                    emitter_failed = true;
                    return false;
                }
                if (!emit_counted(token_event(request, token))) {
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
                if (!emit_counted(structured_event(request, event))) {
                    emitter_failed = true;
                    return false;
                }
                return can_emit();
            });
        if (emitter_failed) {
            return error(request, AILA_ERR_RUNTIME, "streaming event was invalid or could not be written");
        }
        if (cancelled.load(std::memory_order_acquire)) {
            return result(
                request,
                std::string("{\"status\":1,\"eventCount\":") +
                    std::to_string(event_count + 1) + "}");
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
                                  std::to_string(cancelled.load() ? 1 : status) +
                                  ",\"eventCount\":" + std::to_string(event_count + 1) + "}");
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

        if (request.header.method == "log.set_level") {
            simdjson::dom::object object;
            int64_t level = -1;
            if (payload.get_object().get(object) != simdjson::SUCCESS ||
                object["level"].get_int64().get(level) != simdjson::SUCCESS ||
                level < 0 || level > 3) {
                return error(
                    request, AILA_ERR_INVALID_ARGUMENT,
                    "log.set_level requires an integer level from 0 through 3");
            }
            size_t fields = 0;
            for (auto field : object) { (void)field; ++fields; }
            if (fields != 1) {
                return error(
                    request, AILA_ERR_INVALID_ARGUMENT,
                    "log.set_level payload contained extra fields");
            }
            engine_->set_log_level(static_cast<int>(level));
            return result(request, "{\"ok\":true}");
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

        if (request.header.method == "vision.detect_file" ||
            request.header.method == "vision.detect_encoded" ||
            request.header.method == "vision.detect_pixels") {
            if (!initialized_) {
                return error(request, AILA_ERR_INVALID_ARGUMENT, "engine is not initialized");
            }
            simdjson::dom::object object;
            DetectionRequest detection;
            float confidence = 0.0f;
            int max_detections = 0;
            if (payload.get_object().get(object) != simdjson::SUCCESS ||
                !object_float(object, "confidenceThreshold", confidence) ||
                !object_integer(object, "maxDetections", max_detections) ||
                confidence < 0.0f || confidence > 1.0f ||
                max_detections < 1 || max_detections > 300) {
                return error(request, AILA_ERR_INVALID_ARGUMENT, "detection config is malformed");
            }
            detection.config = {};
            detection.config.struct_size = sizeof(AilaDetectionConfig);
            detection.config.confidence_threshold = confidence;
            detection.config.max_detections = max_detections;
            if (request.header.method == "vision.detect_file") {
                std::string_view path;
                if (!request.attachment.empty() ||
                    object["path"].get_string().get(path) != simdjson::SUCCESS || path.empty() ||
                    !require_c_string_safe(path)) {
                    return error(request, AILA_ERR_INVALID_ARGUMENT, "detection file path is malformed");
                }
                detection.method = DetectionMethod::File;
                detection.path.assign(path.data(), path.size());
            } else {
                uint64_t byte_count = 0;
                if (object["byteCount"].get_uint64().get(byte_count) != simdjson::SUCCESS ||
                    byte_count == 0 || byte_count != request.attachment.size()) {
                    return error(request, AILA_ERR_INVALID_ARGUMENT, "detection attachment size is malformed");
                }
                detection.bytes = request.attachment;
                if (request.header.method == "vision.detect_encoded") {
                    detection.method = DetectionMethod::Encoded;
                } else {
                    int format = 0;
                    if (!object_integer(object, "width", detection.width) ||
                        !object_integer(object, "height", detection.height) ||
                        !object_integer(object, "rowStride", detection.row_stride) ||
                        !object_integer(object, "pixelFormat", format) ||
                        detection.width <= 0 || detection.height <= 0 || detection.row_stride <= 0 ||
                        format < AILA_PIXEL_RGB8 || format > AILA_PIXEL_BGRA8) {
                        return error(request, AILA_ERR_INVALID_ARGUMENT, "raw detection image shape is malformed");
                    }
                    detection.method = DetectionMethod::Pixels;
                    detection.pixel_format = static_cast<AilaPixelFormat>(format);
                }
            }
            std::vector<DetectionResult> detections;
            if (!engine_->detect(detection, detections)) {
                int code = engine_->last_error_code();
                if (code == AILA_OK) code = AILA_ERR_RUNTIME;
                std::string message = engine_->last_error_message();
                if (message.empty()) message = "object detection failed";
                return error(request, code, message);
            }
            if (detections.size() > 300) {
                return error(request, AILA_ERR_RUNTIME, "detection result exceeded limit");
            }
            for (const auto& value : detections) {
                if (!simdjson::validate_utf8(value.class_name)) {
                    return error(request, AILA_ERR_RUNTIME, "detection class name is not UTF-8");
                }
            }
            return result(request, detection_result_json(detections));
        }

        if (request.header.method == "align" || request.header.method == "align.words") {
            if (!initialized_) {
                return error(request, AILA_ERR_INVALID_ARGUMENT, "engine is not initialized");
            }
            simdjson::dom::object object;
            uint64_t sample_count = 0, element_size = 0, byte_count = 0;
            int sample_rate = 0;
            AlignmentRequest alignment;
            if (payload.get_object().get(object) != simdjson::SUCCESS ||
                object["sampleCount"].get_uint64().get(sample_count) != simdjson::SUCCESS ||
                !object_integer(object, "sampleRate", sample_rate) ||
                object["elementSize"].get_uint64().get(element_size) != simdjson::SUCCESS ||
                sample_count == 0 || sample_count > (std::numeric_limits<int>::max)() ||
                element_size != sizeof(float) ||
                sample_count > ipc::kMaxAttachmentBytes / sizeof(float)) {
                return error(request, AILA_ERR_INVALID_ARGUMENT, "alignment audio shape is malformed");
            }
            const size_t audio_bytes = static_cast<size_t>(sample_count) * sizeof(float);
            alignment.sample_rate = sample_rate;
            if (request.header.method == "align") {
                std::string_view text;
                std::string_view language;
                if (object["byteCount"].get_uint64().get(byte_count) != simdjson::SUCCESS ||
                    byte_count != audio_bytes || byte_count != request.attachment.size() ||
                    object["text"].get_string().get(text) != simdjson::SUCCESS ||
                    object["language"].get_string().get(language) != simdjson::SUCCESS ||
                    !require_c_string_safe(text) || !require_c_string_safe(language) ||
                    !simdjson::validate_utf8(text) || !simdjson::validate_utf8(language)) {
                    return error(request, AILA_ERR_INVALID_ARGUMENT, "alignment request is malformed");
                }
                alignment.method = AlignmentMethod::Text;
                alignment.text.assign(text.data(), text.size());
                alignment.language.assign(language.data(), language.size());
            } else {
                uint64_t declared_audio_bytes = 0, word_count = 0;
                simdjson::dom::array lengths;
                if (object["audioByteCount"].get_uint64().get(declared_audio_bytes) != simdjson::SUCCESS ||
                    object["wordCount"].get_uint64().get(word_count) != simdjson::SUCCESS ||
                    object["wordByteLengths"].get_array().get(lengths) != simdjson::SUCCESS ||
                    object["byteCount"].get_uint64().get(byte_count) != simdjson::SUCCESS ||
                    declared_audio_bytes != audio_bytes || word_count == 0 ||
                    word_count > (std::numeric_limits<int>::max)() ||
                    byte_count != request.attachment.size() || byte_count < audio_bytes) {
                    return error(request, AILA_ERR_INVALID_ARGUMENT, "alignment word request is malformed");
                }
                alignment.method = AlignmentMethod::Words;
                size_t offset = audio_bytes;
                size_t parsed_word_count = 0;
                for (simdjson::dom::element length_element : lengths) {
                    uint64_t length = 0;
                    if (length_element.get_uint64().get(length) != simdjson::SUCCESS ||
                        parsed_word_count >= word_count ||
                        length > request.attachment.size() - offset) {
                        return error(request, AILA_ERR_INVALID_ARGUMENT, "alignment word lengths are malformed");
                    }
                    ++parsed_word_count;
                    offset += static_cast<size_t>(length);
                }
                if (parsed_word_count != word_count || offset != request.attachment.size()) {
                    return error(request, AILA_ERR_INVALID_ARGUMENT, "alignment word count is malformed");
                }
                alignment.words.reserve(static_cast<size_t>(word_count));
                offset = audio_bytes;
                for (simdjson::dom::element length_element : lengths) {
                    uint64_t length = 0;
                    if (length_element.get_uint64().get(length) != simdjson::SUCCESS) {
                        return error(request, AILA_ERR_INVALID_ARGUMENT, "alignment word lengths are malformed");
                    }
                    const std::string_view word(
                        reinterpret_cast<const char*>(request.attachment.data() + offset),
                        static_cast<size_t>(length));
                    if (!require_c_string_safe(word) || !simdjson::validate_utf8(word)) {
                        return error(request, AILA_ERR_INVALID_ARGUMENT, "alignment word text is malformed");
                    }
                    alignment.words.emplace_back(word);
                    offset += static_cast<size_t>(length);
                }
            }
            alignment.samples.resize(static_cast<size_t>(sample_count));
            std::memcpy(alignment.samples.data(), request.attachment.data(), audio_bytes);
            std::vector<AlignedWordResult> output;
            if (!engine_->align(alignment, output)) {
                int code = engine_->last_error_code();
                if (code == AILA_OK) code = AILA_ERR_RUNTIME;
                return error(request, code, engine_->last_error_message());
            }
            return alignment_result(request, output);
        }

        if (request.header.method == "tts.synthesize_wav") {
            if (!initialized_) return error(request, AILA_ERR_INVALID_ARGUMENT, "engine is not initialized");
            simdjson::dom::object object;
            uint64_t token_count = 0, embedding_count = 0, token_offset = 0;
            uint64_t embedding_offset = 0, element_size = 0, byte_count = 0;
            AudioRequest audio;
            audio.method = AudioMethod::SynthesizeWav;
            if (payload.get_object().get(object) != simdjson::SUCCESS ||
                object["tokenCount"].get_uint64().get(token_count) != simdjson::SUCCESS ||
                object["embeddingCount"].get_uint64().get(embedding_count) != simdjson::SUCCESS ||
                object["tokenOffset"].get_uint64().get(token_offset) != simdjson::SUCCESS ||
                object["embeddingOffset"].get_uint64().get(embedding_offset) != simdjson::SUCCESS ||
                object["elementSize"].get_uint64().get(element_size) != simdjson::SUCCESS ||
                object["byteCount"].get_uint64().get(byte_count) != simdjson::SUCCESS ||
                !parse_optional_legacy_config(object, audio.has_config, audio.config) ||
                token_count == 0 || token_count > (std::numeric_limits<int>::max)() ||
                embedding_count > (std::numeric_limits<int>::max)() ||
                element_size != sizeof(int32_t) || token_offset != 0 ||
                token_count > (std::numeric_limits<size_t>::max)() / sizeof(int32_t) ||
                embedding_count > (std::numeric_limits<size_t>::max)() / sizeof(float) ||
                embedding_offset != token_count * sizeof(int32_t) ||
                embedding_offset > ipc::kMaxAttachmentBytes ||
                embedding_count * sizeof(float) > ipc::kMaxAttachmentBytes - embedding_offset ||
                byte_count != embedding_offset + embedding_count * sizeof(float) ||
                byte_count != request.attachment.size()) {
                return error(request, AILA_ERR_INVALID_ARGUMENT, "TTS token/embedding attachment is malformed");
            }
            audio.text_tokens.resize(static_cast<size_t>(token_count));
            audio.embedding.resize(static_cast<size_t>(embedding_count));
            std::memcpy(audio.text_tokens.data(), request.attachment.data(),
                        audio.text_tokens.size() * sizeof(int32_t));
            if (!audio.embedding.empty()) {
                std::memcpy(audio.embedding.data(), request.attachment.data() + embedding_offset,
                            audio.embedding.size() * sizeof(float));
            }
            std::vector<float> output;
            if (!engine_->process_audio(audio, output)) {
                int code = engine_->last_error_code(); if (code == AILA_OK) code = AILA_ERR_RUNTIME;
                return error(request, code, engine_->last_error_message());
            }
            if (output.size() > (std::numeric_limits<int>::max)() ||
                output.size() > ipc::kMaxAttachmentBytes / sizeof(float) || !finite_values(output)) {
                return error(request, AILA_ERR_RUNTIME, "TTS audio result is malformed");
            }
            return float_result(request, output);
        }

        if (request.header.method == "tts.synthesize_text_to_wav") {
            if (!initialized_) return error(request, AILA_ERR_INVALID_ARGUMENT, "engine is not initialized");
            simdjson::dom::object object;
            std::string_view text;
            uint64_t embedding_count = 0, element_size = 0, byte_count = 0;
            AudioRequest audio;
            audio.method = AudioMethod::SynthesizeTextToWav;
            if (payload.get_object().get(object) != simdjson::SUCCESS ||
                object["text"].get_string().get(text) != simdjson::SUCCESS ||
                !require_c_string_safe(text) || !simdjson::validate_utf8(text) ||
                object["embeddingCount"].get_uint64().get(embedding_count) != simdjson::SUCCESS ||
                object["elementSize"].get_uint64().get(element_size) != simdjson::SUCCESS ||
                object["byteCount"].get_uint64().get(byte_count) != simdjson::SUCCESS ||
                !parse_optional_legacy_config(object, audio.has_config, audio.config) ||
                embedding_count > (std::numeric_limits<int>::max)() ||
                element_size != sizeof(float) ||
                embedding_count > ipc::kMaxAttachmentBytes / sizeof(float) ||
                byte_count != embedding_count * sizeof(float) ||
                byte_count != request.attachment.size()) {
                return error(request, AILA_ERR_INVALID_ARGUMENT, "TTS text/embedding request is malformed");
            }
            audio.text.assign(text.data(), text.size());
            audio.embedding.resize(static_cast<size_t>(embedding_count));
            if (!audio.embedding.empty()) std::memcpy(audio.embedding.data(), request.attachment.data(), request.attachment.size());
            std::vector<float> output;
            if (!engine_->process_audio(audio, output)) {
                int code = engine_->last_error_code(); if (code == AILA_OK) code = AILA_ERR_RUNTIME;
                return error(request, code, engine_->last_error_message());
            }
            if (output.size() > (std::numeric_limits<int>::max)() ||
                output.size() > ipc::kMaxAttachmentBytes / sizeof(float) || !finite_values(output)) {
                return error(request, AILA_ERR_RUNTIME, "TTS audio result is malformed");
            }
            return float_result(request, output);
        }

        if (request.header.method == "tts.synthesize") {
            if (!initialized_ || !request.attachment.empty())
                return error(request, AILA_ERR_INVALID_ARGUMENT, "invalid TTS file request");
            simdjson::dom::object object;
            std::string_view output_path;
            AudioRequest audio;
            audio.method = AudioMethod::SynthesizeFile;
            if (payload.get_object().get(object) != simdjson::SUCCESS ||
                !parse_tts_text_request(object, audio) ||
                object["outputWavPath"].get_string().get(output_path) != simdjson::SUCCESS ||
                !require_c_string_safe(output_path) ||
                !simdjson::validate_utf8(output_path)) {
                return error(request, AILA_ERR_INVALID_ARGUMENT, "TTS file request is malformed");
            }
            audio.output_wav_path.assign(output_path.data(), output_path.size());
            std::vector<float> ignored;
            if (!engine_->process_audio(audio, ignored)) {
                int code = engine_->last_error_code(); if (code == AILA_OK) code = AILA_ERR_RUNTIME;
                return error(request, code, engine_->last_error_message());
            }
            return result(request, "{\"ok\":true}");
        }

        if (request.header.method == "mimi.decode") {
            if (!initialized_) return error(request, AILA_ERR_INVALID_ARGUMENT, "engine is not initialized");
            simdjson::dom::object object;
            uint64_t frames = 0, codes = 0, element_size = 0, byte_count = 0;
            AudioRequest audio;
            audio.method = AudioMethod::DecodeMimi;
            if (payload.get_object().get(object) != simdjson::SUCCESS ||
                object["frameCount"].get_uint64().get(frames) != simdjson::SUCCESS ||
                object["codeCount"].get_uint64().get(codes) != simdjson::SUCCESS ||
                object["elementSize"].get_uint64().get(element_size) != simdjson::SUCCESS ||
                object["byteCount"].get_uint64().get(byte_count) != simdjson::SUCCESS ||
                frames == 0 || frames > (std::numeric_limits<int>::max)() ||
                frames > (std::numeric_limits<uint64_t>::max)() / 16 || codes != frames * 16 ||
                element_size != sizeof(int32_t) || codes > ipc::kMaxAttachmentBytes / sizeof(int32_t) ||
                byte_count != codes * sizeof(int32_t) || byte_count != request.attachment.size()) {
                return error(request, AILA_ERR_INVALID_ARGUMENT, "Mimi code attachment is malformed");
            }
            audio.frame_count = static_cast<int>(frames);
            audio.codes.resize(static_cast<size_t>(codes));
            std::memcpy(audio.codes.data(), request.attachment.data(), request.attachment.size());
            std::vector<float> output;
            if (!engine_->process_audio(audio, output)) {
                int code = engine_->last_error_code(); if (code == AILA_OK) code = AILA_ERR_RUNTIME;
                return error(request, code, engine_->last_error_message());
            }
            if (output.size() > (std::numeric_limits<int>::max)() ||
                output.size() > ipc::kMaxAttachmentBytes / sizeof(float) || !finite_values(output)) {
                return error(request, AILA_ERR_RUNTIME, "Mimi audio result is malformed");
            }
            return float_result(request, output);
        }

        if (request.header.method == "tts.extract_embedding") {
            if (!initialized_ || !request.attachment.empty())
                return error(request, AILA_ERR_INVALID_ARGUMENT, "invalid embedding request");
            simdjson::dom::object object;
            std::string_view path;
            AudioRequest audio;
            audio.method = AudioMethod::ExtractEmbedding;
            if (payload.get_object().get(object) != simdjson::SUCCESS ||
                object["audioPath"].get_string().get(path) != simdjson::SUCCESS ||
                !require_c_string_safe(path) || !simdjson::validate_utf8(path)) {
                return error(request, AILA_ERR_INVALID_ARGUMENT, "embedding audio path is malformed");
            }
            audio.audio_path.assign(path.data(), path.size());
            std::vector<float> output;
            if (!engine_->process_audio(audio, output)) {
                int code = engine_->last_error_code(); if (code == AILA_OK) code = AILA_ERR_RUNTIME;
                return error(request, code, engine_->last_error_message());
            }
            if (output.size() > (std::numeric_limits<int>::max)() ||
                output.size() > ipc::kMaxAttachmentBytes / sizeof(float) || !finite_values(output)) {
                return error(request, AILA_ERR_RUNTIME, "speaker embedding result is malformed");
            }
            return float_result(request, output);
        }

        if (request.header.method == "asr.stream.create") {
            if (!initialized_) {
                return error(request, AILA_ERR_INVALID_ARGUMENT, "engine is not initialized");
            }
            if (!request.attachment.empty()) {
                return error(request, AILA_ERR_INVALID_ARGUMENT,
                             "ASR stream create must not contain an attachment");
            }
            simdjson::dom::object object;
            AsrStreamConfig config;
            if (payload.get_object().get(object) != simdjson::SUCCESS ||
                !parse_optional_legacy_config(object, config.has_config, config.config) ||
                !optional_string_field(object, "forcedLanguage", config.has_forced_language,
                                       config.forced_language) ||
                !optional_string_field(object, "systemPrompt", config.has_system_prompt,
                                       config.system_prompt)) {
                return error(request, AILA_ERR_INVALID_ARGUMENT, "ASR stream config is malformed");
            }
            if (next_asr_stream_id_ == 0 ||
                next_asr_stream_id_ == (std::numeric_limits<uint64_t>::max)()) {
                return error(request, AILA_ERR_RUNTIME, "ASR stream ID space is exhausted");
            }
            const uint64_t id = next_asr_stream_id_++;
            if (!engine_->transcribe_stream_create(id, config)) {
                int code = engine_->last_error_code();
                if (code == AILA_OK) code = AILA_ERR_RUNTIME;
                return error(request, code, engine_->last_error_message());
            }
            asr_stream_ids_.insert(id);
            return result(request, "{\"streamId\":" + std::to_string(id) + "}");
        }

        if (request.header.method == "asr.stream.feed") {
            simdjson::dom::object object;
            uint64_t id = 0, sample_count = 0, element_size = 0, byte_count = 0;
            if (payload.get_object().get(object) != simdjson::SUCCESS ||
                object["streamId"].get_uint64().get(id) != simdjson::SUCCESS ||
                object["sampleCount"].get_uint64().get(sample_count) != simdjson::SUCCESS ||
                object["elementSize"].get_uint64().get(element_size) != simdjson::SUCCESS ||
                object["byteCount"].get_uint64().get(byte_count) != simdjson::SUCCESS ||
                id == 0 || sample_count == 0 || element_size != sizeof(float) ||
                sample_count > (std::numeric_limits<size_t>::max)() / sizeof(float) ||
                byte_count != sample_count * sizeof(float) ||
                byte_count != request.attachment.size()) {
                return error(request, AILA_ERR_INVALID_ARGUMENT, "ASR stream audio attachment is malformed");
            }
            if (asr_stream_ids_.find(id) == asr_stream_ids_.end()) {
                return error(request, AILA_ERR_INVALID_ARGUMENT, "unknown ASR stream ID");
            }
            std::vector<float> samples(static_cast<size_t>(sample_count));
            std::memcpy(samples.data(), request.attachment.data(), request.attachment.size());
            if (!engine_->transcribe_stream_feed(id, samples.data(), samples.size())) {
                int code = engine_->last_error_code();
                if (code == AILA_OK) code = AILA_ERR_RUNTIME;
                return error(request, code, engine_->last_error_message());
            }
            return result(request, "{\"ok\":true}");
        }

        if (request.header.method == "asr.stream.get_text") {
            simdjson::dom::object object;
            uint64_t id = 0;
            if (payload.get_object().get(object) != simdjson::SUCCESS ||
                object["streamId"].get_uint64().get(id) != simdjson::SUCCESS || id == 0) {
                return error(request, AILA_ERR_INVALID_ARGUMENT, "ASR stream ID is malformed");
            }
            if (!request.attachment.empty() || asr_stream_ids_.find(id) == asr_stream_ids_.end()) {
                return error(request, AILA_ERR_INVALID_ARGUMENT, "unknown ASR stream ID");
            }
            std::string stable, partial;
            if (!engine_->transcribe_stream_get_text(id, stable, partial)) {
                int code = engine_->last_error_code();
                if (code == AILA_OK) code = AILA_ERR_RUNTIME;
                return error(request, code, engine_->last_error_message());
            }
            if (stable.find('\0') != std::string::npos || partial.find('\0') != std::string::npos ||
                !simdjson::validate_utf8(stable) || !simdjson::validate_utf8(partial) ||
                partial.size() > ipc::kMaxAttachmentBytes ||
                stable.size() > ipc::kMaxAttachmentBytes - partial.size()) {
                return error(request, AILA_ERR_RUNTIME, "ASR stream text was invalid");
            }
            return two_text_result(request, "stableBytes", stable, "partialBytes", partial);
        }

        if (request.header.method == "asr.stream.destroy") {
            simdjson::dom::object object;
            uint64_t id = 0;
            if (payload.get_object().get(object) != simdjson::SUCCESS ||
                object["streamId"].get_uint64().get(id) != simdjson::SUCCESS || id == 0 ||
                !request.attachment.empty() || asr_stream_ids_.find(id) == asr_stream_ids_.end()) {
                return error(request, AILA_ERR_INVALID_ARGUMENT, "unknown ASR stream ID");
            }
            asr_stream_ids_.erase(id);
            if (!engine_->transcribe_stream_destroy(id)) {
                int code = engine_->last_error_code();
                if (code == AILA_OK) code = AILA_ERR_RUNTIME;
                return error(request, code, engine_->last_error_message());
            }
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
