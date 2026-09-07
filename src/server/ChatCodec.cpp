#include "server/ChatCodec.hpp"

#include "chat/AssistantOutputParser.hpp"
#include "chat/ChatFormatter.hpp"
#include "chat/ChatJson.hpp"
#include "chat/StructuredStreamParser.hpp"
#include "utils/Tokenizer.hpp"

#include <json/json.h>
#include <simdjson.h>

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <optional>
#include <utility>

namespace aila::server {
namespace {
namespace wire = ipc::v2;

constexpr size_t kMaxBodyBytes = 1024 * 1024;
constexpr size_t kMaxMessages = 256;
constexpr size_t kErrorReserveBytes = 2048;
constexpr size_t kMinimumOutputBytes = 4096;
constexpr const char* kDoneFrame = "data: [DONE]\n\n";

[[noreturn]] void Invalid(const std::string& param, const std::string& message) {
    throw ApiError(400, "invalid_parameter", message, param);
}

// Returns a scalar's length, zero for a valid incomplete suffix, or -1 for an
// invalid sequence. Surrogates, overlong encodings and values above U+10FFFF
// are rejected before JsonCpp or the structured output parser sees them.
int ScalarLength(std::string_view value, size_t offset) {
    const auto first = static_cast<unsigned char>(value[offset]);
    if (first < 0x80) return 1;
    const int length = first >= 0xc2 && first <= 0xdf ? 2 : first >= 0xe0 && first <= 0xef ? 3
        : first >= 0xf0 && first <= 0xf4 ? 4 : -1;
    if (length < 0) return -1;
    for (int index = 1; index < length; ++index) {
        if (offset + static_cast<size_t>(index) >= value.size()) return 0;
        const auto byte = static_cast<unsigned char>(value[offset + index]);
        if (byte < 0x80 || byte > 0xbf) return -1;
        if (index == 1 && ((first == 0xe0 && byte < 0xa0) || (first == 0xed && byte > 0x9f) ||
                           (first == 0xf0 && byte < 0x90) || (first == 0xf4 && byte > 0x8f))) return -1;
    }
    return length;
}

bool ValidUtf8(std::string_view value) {
    for (size_t offset = 0; offset < value.size();) {
        const int length = ScalarLength(value, offset);
        if (length <= 0) return false;
        offset += static_cast<size_t>(length);
    }
    return true;
}

std::string SafeDiagnostic(std::string_view value, size_t maximum) {
    std::string output;
    output.reserve(std::min(value.size(), maximum));
    for (size_t offset = 0; offset < value.size();) {
        const int length = ScalarLength(value, offset);
        if (length <= 0) {
            if (output.size() == maximum) break;
            output.push_back('?');
            ++offset;
            continue;
        }
        if (static_cast<size_t>(length) > maximum - output.size()) break;
        if (length == 1 && static_cast<unsigned char>(value[offset]) < 0x20) output.push_back(' ');
        else output.append(value.data() + offset, static_cast<size_t>(length));
        offset += static_cast<size_t>(length);
    }
    return output;
}

std::string JsonText(const Json::Value& value) {
    Json::StreamWriterBuilder builder;
    builder["indentation"] = "";
    builder["commentStyle"] = "None";
    builder["emitUTF8"] = true;
    builder["useSpecialFloats"] = false;
    return Json::writeString(builder, value);
}

void ValidateJsonScalars(const Json::Value& value) {
    if (value.isString()) {
        if (!ValidUtf8(value.asString())) throw ApiError(400, "invalid_json", "JSON strings must contain valid Unicode");
    } else if (value.type() == Json::realValue) {
        if (!std::isfinite(value.asDouble())) throw ApiError(400, "invalid_json", "JSON numbers must be finite");
    } else if (value.isArray()) {
        for (const auto& child : value) ValidateJsonScalars(child);
    } else if (value.isObject()) {
        for (const auto& name : value.getMemberNames()) {
            if (!ValidUtf8(name)) throw ApiError(400, "invalid_json", "JSON member names must contain valid Unicode");
            ValidateJsonScalars(value[name]);
        }
    }
}

Json::Value ParseBody(std::string_view body) {
    if (body.size() > kMaxBodyBytes) throw ApiError(413, "request_too_large", "Chat request body exceeds 1 MiB");
    if (body.empty() || !ValidUtf8(body)) throw ApiError(400, "invalid_json", "Request body must be a UTF-8 JSON object");
    // JsonCpp's strict settings still accept some non-JSON lexical forms,
    // including unescaped controls in strings and leading-zero numbers.
    // The engine's existing simdjson dependency validates the entire grammar;
    // JsonCpp below additionally rejects duplicate keys after unescaping.
    simdjson::dom::parser syntaxParser;
    simdjson::dom::element syntaxRoot;
    if (syntaxParser.parse(body.data(), body.size()).get(syntaxRoot) != simdjson::SUCCESS) {
        throw ApiError(400, "invalid_json", "Request body must contain valid JSON strings, numbers and structure");
    }
    Json::CharReaderBuilder builder;
    builder["collectComments"] = false;
    builder["allowComments"] = false;
    builder["allowTrailingCommas"] = false;
    builder["strictRoot"] = true;
    builder["allowDroppedNullPlaceholders"] = false;
    builder["allowNumericKeys"] = false;
    builder["allowSingleQuotes"] = false;
    builder["stackLimit"] = 64;
    builder["failIfExtra"] = true;
    builder["rejectDupKeys"] = true;
    builder["allowSpecialFloats"] = false;
    builder["skipBom"] = false;
    const std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    Json::Value root;
    std::string errors;
    bool parsed = false;
    try { parsed = reader->parse(body.data(), body.data() + body.size(), &root, &errors); }
    catch (const Json::Exception&) { throw ApiError(400, "invalid_json", "Request body exceeds the JSON structure limits"); }
    if (!parsed || !root.isObject()) {
        throw ApiError(400, "invalid_json", "Request body must be one strict JSON object without duplicate keys");
    }
    ValidateJsonScalars(root);
    return root;
}

void Fields(const Json::Value& object, std::initializer_list<std::string_view> allowed, const std::string& prefix = {}) {
    if (!object.isObject()) Invalid(prefix, "Expected a JSON object");
    for (const auto& name : object.getMemberNames()) {
        if (std::find(allowed.begin(), allowed.end(), name) == allowed.end()) {
            const auto parameter = prefix.empty() ? name : prefix + "." + name;
            throw ApiError(400, "unsupported_parameter", "This chat endpoint does not support the supplied field", parameter);
        }
    }
}

std::string String(const Json::Value& value, const std::string& param, bool empty = true, size_t maxBytes = kMaxBodyBytes) {
    if (!value.isString()) Invalid(param, "Expected a string");
    auto text = value.asString();
    if ((!empty && text.empty()) || text.size() > maxBytes) Invalid(param, "String is empty or exceeds its allowed length");
    return text;
}

bool Boolean(const Json::Value& value, const std::string& param) {
    if (!value.isBool()) Invalid(param, "Expected a boolean");
    return value.asBool();
}

uint64_t Unsigned(const Json::Value& value, const std::string& param, uint64_t maximum, bool zero = true) {
    if ((value.type() != Json::intValue && value.type() != Json::uintValue) || !value.isUInt64()) {
        Invalid(param, "Expected an unsigned integer without a fractional part");
    }
    const auto number = value.asUInt64();
    if ((!zero && number == 0) || number > maximum) Invalid(param, "Integer is outside the supported range");
    return number;
}

double Number(const Json::Value& value, const std::string& param) {
    if (!value.isNumeric() || value.isBool()) Invalid(param, "Expected a finite number");
    const double number = value.asDouble();
    if (!std::isfinite(number) || std::abs(number) > static_cast<double>((std::numeric_limits<float>::max)())) {
        Invalid(param, "Number cannot be represented by the inference engine");
    }
    return number;
}

void ValidateMessages(const Json::Value& messages) {
    if (!messages.isArray() || messages.empty() || messages.size() > kMaxMessages) {
        Invalid("messages", "Expected between 1 and 256 text messages");
    }
    for (Json::ArrayIndex index = 0; index < messages.size(); ++index) {
        const auto& message = messages[index];
        const std::string prefix = "messages[" + std::to_string(index) + "]";
        Fields(message, {"role", "content", "name", "reasoning_content"}, prefix);
        const auto role = String(message["role"], prefix + ".role", false, 16);
        if (role != "system" && role != "developer" && role != "user" && role != "assistant") {
            throw ApiError(400, "unsupported_message_role", "Only system, developer, user and assistant text messages are supported", prefix + ".role");
        }
        if (message.isMember("name")) String(message["name"], prefix + ".name", false, 64);
        if (message.isMember("reasoning_content")) {
            if (role != "assistant") Invalid(prefix + ".reasoning_content", "Only assistant messages may carry reasoning_content");
            String(message["reasoning_content"], prefix + ".reasoning_content");
        }
        const auto& content = message["content"];
        if (content.isString()) continue;
        if (!content.isArray() || content.empty() || content.size() > 256) {
            Invalid(prefix + ".content", "Message content must be text or an array of text parts");
        }
        for (Json::ArrayIndex partIndex = 0; partIndex < content.size(); ++partIndex) {
            const auto& part = content[partIndex];
            const auto parameter = prefix + ".content[" + std::to_string(partIndex) + "]";
            Fields(part, {"type", "text"}, parameter);
            if (String(part["type"], parameter + ".type", false, 32) != "text") {
                throw ApiError(400, "unsupported_content_type", "Only text content parts are supported", parameter + ".type");
            }
            String(part["text"], parameter + ".text");
        }
    }
}

GenerationConfig GenerationConfigFor(const wire::GenerationOptions& wireOptions) {
    GenerationConfig result;
    result.max_new_tokens = static_cast<int>(wireOptions.maxNewTokens);
    result.do_sample = wireOptions.doSample;
    result.temperature = static_cast<float>(wireOptions.temperature);
    result.top_k = static_cast<int>(wireOptions.topK);
    result.top_p = static_cast<float>(wireOptions.topP);
    result.use_fixed_seed = wireOptions.useFixedSeed;
    result.sampling_seed = wireOptions.samplingSeed;
    result.repetition_penalty = static_cast<float>(wireOptions.repetitionPenalty);
    result.presence_penalty = static_cast<float>(wireOptions.presencePenalty);
    result.frequency_penalty = static_cast<float>(wireOptions.frequencyPenalty);
    result.thinking_budget_tokens = -1;
    return result;
}

const char* FallbackTemplate() {
    return "{%- for message in messages -%}"
           "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' -}}"
           "{%- endfor -%}"
           "{%- if add_generation_prompt -%}"
           "{{- '<|im_start|>assistant\\n' -}}"
           "{%- if enable_thinking -%}{{- '<think>\\n' -}}"
           "{%- else -%}{{- '<think>\\n</think>\\n' -}}{%- endif -%}"
           "{%- endif -%}";
}

bool StartsInReasoning(const std::string& prompt) {
    const auto open = prompt.rfind("<think>");
    const auto close = prompt.rfind("</think>");
    if (open == std::string::npos || (close != std::string::npos && close > open)) return false;
    return prompt.find_first_not_of(" \t\r\n", open + 7) == std::string::npos;
}

Json::Value Usage(uint64_t prompt, uint64_t generated) {
    Json::Value usage(Json::objectValue);
    usage["prompt_tokens"] = Json::UInt64(prompt);
    usage["completion_tokens"] = Json::UInt64(generated);
    usage["total_tokens"] = Json::UInt64(prompt + generated);
    return usage;
}

ApiError CompletionError(const wire::Summary& summary, bool localFailure) {
    if (localFailure) return ApiError(502, "worker_connection_failed", "Inference output was interrupted or abandoned");
    const auto message = summary.message.empty() ? "Inference did not complete successfully" : summary.message;
    if (summary.endReason == wire::EndReason::RuntimeFault) return ApiError(503, "runtime_fault", message);
    if (summary.endReason == wire::EndReason::SchedulerStopped) return ApiError(503, "server_stopping", message);
    if (summary.endReason == wire::EndReason::OutputBackpressure) return ApiError(503, "output_backpressure", message);
    if (summary.status == wire::RequestStatus::Cancelled) return ApiError(409, "request_cancelled", message);
    return ApiError(500, "generation_failed", message);
}
} // namespace

ApiError::ApiError(int status, std::string errorCode, std::string message, std::string parameter)
    : std::runtime_error(std::move(message)), httpStatus(status), code(std::move(errorCode)), param(std::move(parameter)) {}

std::string ApiError::ToJson() const {
    Json::Value result(Json::objectValue);
    Json::Value error(Json::objectValue);
    error["message"] = SafeDiagnostic(what(), 512);
    error["type"] = httpStatus >= 500 ? "server_error" : "invalid_request_error";
    error["code"] = SafeDiagnostic(code, 64);
    if (param.empty()) error["param"] = Json::nullValue;
    else error["param"] = SafeDiagnostic(param, 128);
    result["error"] = std::move(error);
    return JsonText(result);
}

PreparedChat PrepareChat(std::string_view body, const Tokenizer& tokenizer, const ChatSettings& settings) {
    const auto hard = wire::hard_limits();
    if (settings.modelId.empty() || settings.modelId.size() > 256 || !ValidUtf8(settings.modelId) ||
        settings.maxContextTokens == 0 || settings.maxContextTokens > static_cast<uint32_t>((std::numeric_limits<int>::max)()) ||
        settings.maxPromptTokens == 0 || settings.maxPromptTokens > hard.maxPromptTokens ||
        settings.maxNewTokens == 0 || settings.maxNewTokens > hard.maxGeneratedTokens || settings.maxOutputBytes < kMinimumOutputBytes) {
        throw ApiError(500, "invalid_server_configuration", "Server chat limits or model identity are invalid");
    }
    const auto root = ParseBody(body);
    Fields(root, {"model", "messages", "stream", "stream_options", "max_tokens", "max_completion_tokens",
                  "temperature", "top_p", "top_k", "seed", "repetition_penalty", "presence_penalty",
                  "frequency_penalty", "n", "enable_thinking"});
    const auto requestedModel = String(root["model"], "model", false, 256);
    if (requestedModel != settings.modelId) throw ApiError(404, "model_not_found", "Requested model is not served by this process", "model");
    ValidateMessages(root["messages"]);
    PreparedChat prepared;
    if (root.isMember("stream")) prepared.stream = Boolean(root["stream"], "stream");
    if (root.isMember("stream_options")) {
        if (!prepared.stream) Invalid("stream_options", "stream_options requires stream=true");
        Fields(root["stream_options"], {"include_usage"}, "stream_options");
        if (root["stream_options"].isMember("include_usage")) {
            prepared.includeUsage = Boolean(root["stream_options"]["include_usage"], "stream_options.include_usage");
        }
    }
    if (root.isMember("n") && Unsigned(root["n"], "n", (std::numeric_limits<uint64_t>::max)()) != 1) {
        throw ApiError(400, "unsupported_parameter", "Only n=1 is supported", "n");
    }
    auto& generation = prepared.options.generation;
    generation.maxNewTokens = std::min(512u, settings.maxNewTokens);
    if (root.isMember("max_tokens") && root.isMember("max_completion_tokens")) {
        Invalid("max_completion_tokens", "max_tokens and max_completion_tokens are mutually exclusive");
    }
    for (const char* field : {"max_tokens", "max_completion_tokens"}) {
        if (root.isMember(field)) generation.maxNewTokens = static_cast<uint32_t>(Unsigned(root[field], field, settings.maxNewTokens, false));
    }
    if (root.isMember("temperature")) generation.temperature = Number(root["temperature"], "temperature");
    if (generation.temperature < 0 || (generation.temperature > 0 && static_cast<float>(generation.temperature) <= 0)) {
        Invalid("temperature", "Temperature must be nonnegative and representable as a float");
    }
    generation.doSample = generation.temperature != 0;
    if (root.isMember("top_p")) generation.topP = Number(root["top_p"], "top_p");
    if (generation.topP <= 0 || generation.topP > 1 || static_cast<float>(generation.topP) <= 0) Invalid("top_p", "top_p must be in (0, 1]");
    if (root.isMember("top_k")) generation.topK = static_cast<uint32_t>(Unsigned(root["top_k"], "top_k", (std::numeric_limits<int32_t>::max)()));
    if (root.isMember("seed")) {
        generation.samplingSeed = Unsigned(root["seed"], "seed", (std::numeric_limits<uint64_t>::max)());
        generation.useFixedSeed = true;
    }
    if (root.isMember("repetition_penalty")) generation.repetitionPenalty = Number(root["repetition_penalty"], "repetition_penalty");
    if (generation.repetitionPenalty <= 0 || static_cast<float>(generation.repetitionPenalty) <= 0) {
        Invalid("repetition_penalty", "repetition_penalty must be positive");
    }
    if (root.isMember("presence_penalty")) generation.presencePenalty = Number(root["presence_penalty"], "presence_penalty");
    if (root.isMember("frequency_penalty")) generation.frequencyPenalty = Number(root["frequency_penalty"], "frequency_penalty");
    const bool thinking = root.isMember("enable_thinking") && Boolean(root["enable_thinking"], "enable_thinking");

    // Only this normalized object reaches the existing permissive ChatJson
    // reader; request-provided template kwargs and overrides never do.
    Json::Value normalized(Json::objectValue);
    normalized["messages"] = root["messages"];
    for (auto& message : normalized["messages"]) {
        if (!message["content"].isArray()) continue;
        std::string content;
        for (const auto& part : message["content"]) content += part["text"].asString();
        // Qwen's text templates expect message.content to be one string.
        message["content"] = std::move(content);
    }
    normalized["chat_template_kwargs"]["enable_thinking"] = thinking;
    std::string error;
    if (!chat::parse_chat_request_json(JsonText(normalized), GenerationConfigFor(generation), prepared.chatRequest, &error)) {
        throw ApiError(400, "invalid_messages", "Could not parse validated chat messages", "messages");
    }
    if (tokenizer.vocab_size() <= 0) throw ApiError(500, "tokenizer_unavailable", "Server tokenizer is not loaded");

    // Select the trusted checkpoint template (or the fixed service fallback)
    // explicitly so environment/request paths cannot override the HTTP route.
    auto renderRequest = prepared.chatRequest;
    renderRequest.template_options.template_override_text = tokenizer.chat_template().empty() ? FallbackTemplate() : tokenizer.chat_template();
    chat::ChatFormatInput input;
    input.request = &renderRequest;
    input.family = ModelFamily::Qwen3Dense;
    input.tokenizer_chat_template = tokenizer.chat_template();
    input.bos_token = tokenizer.bos_token();
    input.eos_token = tokenizer.eos_token();
    chat::ChatFormatTextResult rendered;
    if (!chat::ChatFormatter{}.render_text(input, true, rendered, &error) || !ValidUtf8(rendered.text)) {
        throw ApiError(500, "chat_template_failed", "The model chat template could not render this text request");
    }
    if (rendered.text.size() > kMaxBodyBytes) throw ApiError(413, "prompt_too_large", "Rendered prompt exceeds the server byte limit", "messages");
    prepared.startsInReasoning = StartsInReasoning(rendered.text);
    const auto ids = tokenizer.encode(rendered.text);
    if (ids.empty() || ids.size() > settings.maxPromptTokens) {
        Invalid("messages", "Rendered prompt token count exceeds the server limit or is empty");
    }
    if (ids.size() > settings.maxContextTokens || generation.maxNewTokens > settings.maxContextTokens - ids.size()) {
        throw ApiError(400, "context_length_exceeded", "Prompt and requested completion do not fit the configured context", "max_completion_tokens");
    }
    prepared.promptIds.reserve(ids.size());
    for (const int token : ids) {
        if (token < 0 || token >= tokenizer.vocab_size()) throw ApiError(500, "tokenizer_failed", "Tokenizer returned an invalid prompt token");
        prepared.promptIds.push_back(static_cast<int32_t>(token));
    }
    for (const int candidate : {tokenizer.eos_id(), tokenizer.im_end_id(), tokenizer.special_token_id("<|endoftext|>")}) {
        if (candidate >= 0 && candidate < tokenizer.vocab_size() && tokenizer.is_eos(candidate) &&
            std::find(prepared.options.stopTokenIds.begin(), prepared.options.stopTokenIds.end(), candidate) == prepared.options.stopTokenIds.end()) {
            prepared.options.stopTokenIds.push_back(candidate);
        }
    }
    if (prepared.options.stopTokenIds.empty()) throw ApiError(500, "tokenizer_failed", "Qwen3 tokenizer has no valid EOS token");
    prepared.options.prefillChunkSize = std::min(hard.maxPrefillChunkSize, settings.maxPromptTokens);
    return prepared;
}

struct ChatOutput::State {
    ChatIdentity identity;
    bool stream;
    bool includeUsage;
    bool startsInReasoning;
    uint64_t promptTokens;
    uint32_t maxGenerated;
    size_t maximum;
    size_t normalLimit;
    size_t emittedBytes = 0;
    bool begun = false;
    bool finished = false;
    bool toolBlockSeen = false;
    std::optional<ApiError> failure;
    chat::StructuredStreamParser parser;
    std::string pendingUtf8;
    std::string rawText;
    std::string content;
    std::string reasoning;
    std::vector<std::string> warnings;
    ChatCompletion finalResult;

    State(ChatIdentity value, const PreparedChat& prepared, size_t limit)
        : identity(std::move(value)), stream(prepared.stream), includeUsage(prepared.includeUsage),
          startsInReasoning(prepared.startsInReasoning), promptTokens(prepared.promptIds.size()),
          maxGenerated(prepared.options.generation.maxNewTokens), maximum(limit), normalLimit(limit - kErrorReserveBytes) {
        if (startsInReasoning) {
            std::vector<chat::StructuredStreamEvent> ignored;
            parser.push("<think>", ignored);
        }
    }

    Json::Value Envelope(bool chunk) const {
        Json::Value result(Json::objectValue);
        result["id"] = identity.id;
        result["object"] = chunk ? "chat.completion.chunk" : "chat.completion";
        result["created"] = Json::Int64(identity.createdUnixSeconds);
        result["model"] = identity.model;
        result["choices"] = Json::arrayValue;
        if (chunk && includeUsage) result["usage"] = Json::nullValue;
        return result;
    }

    void AddFrame(const Json::Value& value, std::vector<std::string>& frames) {
        auto frame = "data: " + JsonText(value) + "\n\n";
        if (frame.size() > normalLimit - emittedBytes) throw ApiError(502, "output_limit_exceeded", "Serialized chat output exceeds the configured byte limit");
        emittedBytes += frame.size();
        frames.push_back(std::move(frame));
    }

    void AddDelta(const Json::Value& delta, std::vector<std::string>& frames, const char* finishReason = nullptr) {
        if (!stream) return;
        auto value = Envelope(true);
        Json::Value choice(Json::objectValue);
        choice["index"] = 0;
        choice["delta"] = delta;
        if (finishReason) choice["finish_reason"] = finishReason;
        else choice["finish_reason"] = Json::nullValue;
        value["choices"].append(std::move(choice));
        AddFrame(value, frames);
    }

    std::vector<std::string> Begin() {
        std::vector<std::string> frames;
        if (begun || finished) return frames;
        if (failure) throw *failure;
        Json::Value delta(Json::objectValue);
        delta["role"] = "assistant";
        delta["content"] = "";
        AddDelta(delta, frames);
        begun = true;
        return frames;
    }

    void FlushDelta(std::string& text, const std::string& field, std::vector<std::string>& frames) {
        if (text.empty()) return;
        Json::Value delta(Json::objectValue);
        delta[field] = text;
        AddDelta(delta, frames);
        text.clear();
    }

    void AppendEvents(const std::vector<chat::StructuredStreamEvent>& events, std::string& deltaText,
                      std::string& deltaField, std::vector<std::string>& frames) {
        for (const auto& event : events) {
            if (event.type != chat::StructuredStreamEventType::ContentDelta &&
                event.type != chat::StructuredStreamEventType::ReasoningDelta &&
                event.type != chat::StructuredStreamEventType::ToolCallDelta) continue;
            const bool isReasoning = event.type == chat::StructuredStreamEventType::ReasoningDelta;
            const std::string field = isReasoning ? "reasoning_content" : "content";
            if (!deltaText.empty() && deltaField != field) FlushDelta(deltaText, deltaField, frames);
            deltaField = field;
            deltaText += event.text;
            (isReasoning ? reasoning : content) += event.text;
            if (event.type == chat::StructuredStreamEventType::ToolCallDelta) toolBlockSeen = true;
        }
    }

    void Events(const std::vector<chat::StructuredStreamEvent>& events, std::vector<std::string>& frames) {
        std::string text;
        std::string field;
        AppendEvents(events, text, field, frames);
        FlushDelta(text, field, frames);
    }

    std::vector<std::string> Push(std::string_view bytes) {
        if (finished) throw ApiError(500, "completion_finished", "Chat output has already finished");
        if (failure) throw *failure;
        if (bytes.size() > normalLimit - rawText.size()) throw ApiError(502, "output_limit_exceeded", "Generated text exceeds the configured byte limit");
        if (!bytes.empty()) rawText.append(bytes.data(), bytes.size());
        std::string input = std::move(pendingUtf8);
        pendingUtf8.clear();
        if (!bytes.empty()) input.append(bytes.data(), bytes.size());
        auto frames = Begin();
        std::vector<chat::StructuredStreamEvent> events;
        std::string deltaText;
        std::string deltaField;
        // Isolate each tag delimiter so completing a marker never also strips
        // whitespace from the following tokenizer fragment. Ordinary UTF-8
        // runs are batched to avoid quadratic rescans of a long tool block.
        for (size_t offset = 0; offset < input.size();) {
            const int length = ScalarLength(input, offset);
            if (length < 0) throw ApiError(502, "invalid_utf8_output", "Model output contains invalid UTF-8");
            if (length == 0) { pendingUtf8.assign(input, offset, std::string::npos); break; }
            size_t end = offset + static_cast<size_t>(length);
            if (input[offset] != '<' && input[offset] != '>') {
                while (end < input.size() && input[end] != '<' && input[end] != '>') {
                    const int nextLength = ScalarLength(input, end);
                    if (nextLength < 0) throw ApiError(502, "invalid_utf8_output", "Model output contains invalid UTF-8");
                    if (nextLength == 0) break;
                    end += static_cast<size_t>(nextLength);
                }
            }
            events.clear();
            parser.push(input.substr(offset, end - offset), events);
            AppendEvents(events, deltaText, deltaField, frames);
            offset = end;
        }
        FlushDelta(deltaText, deltaField, frames);
        return frames;
    }

    ChatCompletion Snapshot() const {
        auto result = finalResult;
        result.sseFrames.clear();
        return result;
    }

    ChatCompletion Fail(const ApiError& error) {
        if (finished) return Snapshot();
        ChatCompletion result;
        result.json = error.ToJson();
        result.httpStatus = error.httpStatus;
        result.failed = true;
        result.warnings = warnings;
        if (result.json.size() + 8 + std::char_traits<char>::length(kDoneFrame) > kErrorReserveBytes) {
            result.json = ApiError(500, "server_error", "Chat generation failed").ToJson();
        }
        if (stream) {
            result.sseFrames.push_back("data: " + result.json + "\n\n");
            result.sseFrames.emplace_back(kDoneFrame);
            for (const auto& frame : result.sseFrames) emittedBytes += frame.size();
        }
        finished = true;
        finalResult = result;
        finalResult.sseFrames.clear();
        return result;
    }

    ChatCompletion Finish(const wire::Summary& summary, bool localFailure) {
        if (finished) return Snapshot();
        if (failure) return Fail(*failure);
        if (localFailure || summary.status != wire::RequestStatus::Completed) return Fail(CompletionError(summary, localFailure));
        wire::ValidationError validation;
        if (!wire::validate_summary(summary, validation) || summary.promptTokensProcessed != promptTokens || summary.generatedTokens > maxGenerated) {
            return Fail(ApiError(502, "invalid_completion_summary", "Worker completion counts or status do not match the request"));
        }
        if (!pendingUtf8.empty() && summary.endReason != wire::EndReason::Length) {
            return Fail(ApiError(502, "invalid_utf8_output", "Model output ended inside a UTF-8 character"));
        }
        try {
            auto frames = Begin();
            if (!pendingUtf8.empty()) {
                // A token limit can stop byte-level BPE halfway through a
                // valid scalar. Replace that suffix once, without inventing a
                // token, and use the regular parser/quota path for both SSE
                // and the final JSON. Invalid sequences already failed Push.
                rawText.resize(rawText.size() - pendingUtf8.size());
                pendingUtf8.clear();
                auto replacementFrames = Push("\xef\xbf\xbd");
                for (auto& frame : replacementFrames) frames.push_back(std::move(frame));
            }
            std::vector<chat::StructuredStreamEvent> events;
            parser.finish(events);
            Events(events, frames);
            // Reuse the complete-output parser for tool recovery diagnostics;
            // canonical deltas remain the authoritative visible strings so a
            // recognized tool block is never removed from a text-only API.
            auto parsed = chat::parse_assistant_output(startsInReasoning ? "<think>" + rawText : rawText);
            if (toolBlockSeen || !parsed.tool_calls.empty()) warnings.emplace_back("Unsolicited tool_call output was retained as text; no tool was executed");
            for (const auto& warning : parsed.warnings) {
                if (warnings.size() == 4) break;
                warnings.push_back(SafeDiagnostic(warning, 256));
            }
            const char* reason = summary.endReason == wire::EndReason::Length ? "length" : "stop";
            auto value = Envelope(false);
            Json::Value message(Json::objectValue);
            message["role"] = "assistant";
            message["content"] = content;
            message["reasoning_content"] = reasoning;
            Json::Value choice(Json::objectValue);
            choice["index"] = 0;
            choice["message"] = std::move(message);
            choice["finish_reason"] = reason;
            value["choices"].append(std::move(choice));
            value["usage"] = Usage(promptTokens, summary.generatedTokens);
            ChatCompletion result;
            result.json = JsonText(value);
            if (result.json.size() > normalLimit) throw ApiError(502, "output_limit_exceeded", "Serialized chat output exceeds the configured byte limit");
            if (stream) {
                AddDelta(Json::Value(Json::objectValue), frames, reason);
                if (includeUsage) {
                    auto usageChunk = Envelope(true);
                    usageChunk["usage"] = Usage(promptTokens, summary.generatedTokens);
                    AddFrame(usageChunk, frames);
                }
                const size_t doneBytes = std::char_traits<char>::length(kDoneFrame);
                if (doneBytes > normalLimit - emittedBytes) throw ApiError(502, "output_limit_exceeded", "Serialized chat output exceeds the configured byte limit");
                emittedBytes += doneBytes;
                frames.emplace_back(kDoneFrame);
            }
            result.sseFrames = std::move(frames);
            result.warnings = warnings;
            finished = true;
            finalResult = result;
            finalResult.sseFrames.clear();
            return result;
        } catch (const ApiError& error) { return Fail(error); }
    }
};

ChatOutput::ChatOutput(ChatIdentity identity, const PreparedChat& prepared, size_t maxOutputBytes) {
    if (maxOutputBytes < kMinimumOutputBytes || identity.id.empty() || identity.id.size() > 128 ||
        identity.model.empty() || identity.model.size() > 256 || identity.createdUnixSeconds < 0 ||
        !ValidUtf8(identity.id) || !ValidUtf8(identity.model) || prepared.promptIds.empty()) {
        throw ApiError(500, "invalid_server_configuration", "Invalid chat response identity or output budget");
    }
    state_ = std::make_unique<State>(std::move(identity), prepared, maxOutputBytes);
}
ChatOutput::~ChatOutput() = default;
ChatOutput::ChatOutput(ChatOutput&&) noexcept = default;
ChatOutput& ChatOutput::operator=(ChatOutput&&) noexcept = default;

std::vector<std::string> ChatOutput::Begin() {
    try { return state_->Begin(); }
    catch (const ApiError& error) { state_->failure = error; throw; }
}
std::vector<std::string> ChatOutput::PushText(std::string_view bytes) {
    try { return state_->Push(bytes); }
    catch (const ApiError& error) { state_->failure = error; throw; }
}
ChatCompletion ChatOutput::Finish(const wire::Summary& summary, bool localFailure) { return state_->Finish(summary, localFailure); }
ChatCompletion ChatOutput::Fail(const ApiError& error) { return state_->Fail(error); }

} // namespace aila::server
