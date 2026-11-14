#include "server/ChatCodec.hpp"

#include "chat/AssistantOutputParser.hpp"
#include "chat/BuiltinTemplates.hpp"
#include "chat/ChatFormatter.hpp"
#include "chat/ChatJson.hpp"
#include "chat/StructuredStreamParser.hpp"
#include "profile/Profiling.hpp"
#include "utils/EnvUtils.hpp"
#include "utils/Tokenizer.hpp"

#include <json/json.h>
#include <simdjson.h>

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace aila::server {
namespace {
namespace wire = ipc::v2;

// Absolute protocol-floor ceiling for a request body and its rendered prompt.
// The per-request limit is ChatSettings::maxBodyBytes, which the server CLI may
// raise from the 64 KiB floor up to this ceiling.
constexpr size_t kProtocolMaxBodyBytes = 4 * 1024 * 1024;
constexpr size_t kMinimumBodyBytes = 64 * 1024;
constexpr size_t kMaxMessages = 256;
constexpr size_t kMaxTools = 128;
constexpr size_t kMaxToolArgumentBytes = 64 * 1024;
constexpr size_t kToolArgumentDeltaBytes = 1024;
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

Json::Value ParseBody(std::string_view body, size_t maxBytes = kProtocolMaxBodyBytes) {
    if (body.size() > maxBytes) throw ApiError(413, "request_too_large", "Chat request body exceeds the configured byte limit");
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

std::string String(const Json::Value& value, const std::string& param, bool empty = true, size_t maxBytes = kProtocolMaxBodyBytes) {
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

std::string Identifier(const Json::Value& value, const std::string& param, size_t maximum = 64) {
    auto name = String(value, param, false, maximum);
    for (const unsigned char character : name) {
        if (!(character >= 'a' && character <= 'z') && !(character >= 'A' && character <= 'Z') &&
            !(character >= '0' && character <= '9') && character != '_' && character != '-') {
            Invalid(param, "Identifiers may contain only ASCII letters, digits, underscores and hyphens");
        }
    }
    return name;
}

std::string ToolArguments(const Json::Value& value, const std::string& param) {
    const auto text = String(value, param, false, kMaxToolArgumentBytes);
    try { (void)ParseBody(text); return text; }
    catch (const ApiError&) { Invalid(param, "Function arguments must encode one strict JSON object"); }
}

void ValidateTools(const Json::Value& root) {
    std::unordered_set<std::string> names;
    if (root.isMember("tools")) {
        const auto& tools = root["tools"];
        if (!tools.isArray() || tools.size() > kMaxTools) Invalid("tools", "Expected at most 128 function tools");
        for (Json::ArrayIndex index = 0; index < tools.size(); ++index) {
            const auto& tool = tools[index];
            const auto prefix = "tools[" + std::to_string(index) + "]";
            Fields(tool, {"type", "function"}, prefix);
            if (String(tool["type"], prefix + ".type") != "function") Invalid(prefix + ".type", "Only function tools are supported");
            const auto& function = tool["function"];
            const auto fn = prefix + ".function";
            Fields(function, {"name", "description", "parameters", "strict"}, fn);
            if (!names.insert(Identifier(function["name"], fn + ".name")).second) Invalid(fn + ".name", "Tool names must be unique");
            if (function.isMember("description")) String(function["description"], fn + ".description", true, 16384);
            if (function.isMember("parameters") && (!function["parameters"].isObject() ||
                JsonText(function["parameters"]).size() > kMaxToolArgumentBytes)) {
                Invalid(fn + ".parameters", "Function parameters must be a JSON object of at most 64 KiB");
            }
            if (function.isMember("strict") && Boolean(function["strict"], fn + ".strict")) {
                throw ApiError(400, "unsupported_parameter", "Schema-constrained generation (strict=true) is not supported", fn + ".strict");
            }
        }
    }
    if (!root.isMember("tool_choice")) return;
    const auto& choice = root["tool_choice"];
    if (choice.isString()) {
        const auto text = choice.asString();
        if (text != "auto" && text != "none" && text != "required") Invalid("tool_choice", "Expected auto, none, required, or a named function");
        if (text == "required" && names.empty()) Invalid("tool_choice", "A required call needs at least one declared tool");
    } else {
        Fields(choice, {"type", "function"}, "tool_choice");
        if (String(choice["type"], "tool_choice.type") != "function") Invalid("tool_choice.type", "Only function choices are supported");
        Fields(choice["function"], {"name"}, "tool_choice.function");
        const auto name = Identifier(choice["function"]["name"], "tool_choice.function.name");
        if (!names.count(name)) Invalid("tool_choice.function.name", "The chosen function must be declared in tools");
    }
}

void ValidateMessages(const Json::Value& messages) {
    if (!messages.isArray() || messages.empty() || messages.size() > kMaxMessages) {
        Invalid("messages", "Expected between 1 and 256 text messages");
    }
    std::unordered_set<std::string> callIds;
    std::unordered_map<std::string, std::string> pendingCalls;
    for (Json::ArrayIndex index = 0; index < messages.size(); ++index) {
        const auto& message = messages[index];
        const std::string prefix = "messages[" + std::to_string(index) + "]";
        Fields(message, {"role", "content", "name", "reasoning_content", "tool_calls", "tool_call_id"}, prefix);
        const auto role = String(message["role"], prefix + ".role", false, 16);
        if (role != "system" && role != "developer" && role != "user" && role != "assistant" && role != "tool") {
            throw ApiError(400, "unsupported_message_role", "Only system, developer, user, assistant and tool messages are supported", prefix + ".role");
        }
        if (role != "tool" && !pendingCalls.empty()) Invalid(prefix + ".role", "All pending tool calls require results before the next message");
        if (message.isMember("name")) String(message["name"], prefix + ".name", false, 64);
        if (message.isMember("reasoning_content")) {
            if (role != "assistant") Invalid(prefix + ".reasoning_content", "Only assistant messages may carry reasoning_content");
            String(message["reasoning_content"], prefix + ".reasoning_content");
        }
        if (role == "tool") {
            const auto id = Identifier(message["tool_call_id"], prefix + ".tool_call_id", 160);
            const auto found = pendingCalls.find(id);
            if (found == pendingCalls.end()) Invalid(prefix + ".tool_call_id", "Tool result must match one pending assistant call");
            if (message.isMember("name") && message["name"].asString() != found->second) Invalid(prefix + ".name", "Tool result name differs from its call");
            pendingCalls.erase(found);
        } else if (message.isMember("tool_call_id")) {
            Invalid(prefix + ".tool_call_id", "Only tool messages may carry tool_call_id");
        }
        const bool hasCalls = message.isMember("tool_calls");
        if (hasCalls) {
            const auto parameter = prefix + ".tool_calls";
            const auto& calls = message["tool_calls"];
            if (role != "assistant" || !calls.isArray() || calls.empty() || calls.size() > kMaxTools) {
                Invalid(parameter, "Only assistant messages may carry between 1 and 128 function calls");
            }
            for (Json::ArrayIndex callIndex = 0; callIndex < calls.size(); ++callIndex) {
                const auto& call = calls[callIndex];
                const auto path = parameter + "[" + std::to_string(callIndex) + "]";
                Fields(call, {"id", "type", "function"}, path);
                const auto id = Identifier(call["id"], path + ".id", 160);
                if (!callIds.insert(id).second) Invalid(path + ".id", "Tool call IDs must be unique throughout the history");
                if (String(call["type"], path + ".type") != "function") Invalid(path + ".type", "Only function calls are supported");
                Fields(call["function"], {"name", "arguments"}, path + ".function");
                const auto name = Identifier(call["function"]["name"], path + ".function.name");
                ToolArguments(call["function"]["arguments"], path + ".function.arguments");
                pendingCalls.emplace(id, name);
            }
        }
        const auto& content = message["content"];
        if (hasCalls && content.isNull()) continue;
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
    if (!pendingCalls.empty()) Invalid("messages", "Tool history ends with unresolved assistant calls");
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

const char* ToolFallbackTemplate() {
    return R"JINJA(
{%- if tools -%}
{{- '<|im_start|>system\n# Tools\n\n<tools>' -}}
{%- for tool in tools -%}{{- '\n' + (tool | tojson) -}}{%- endfor -%}
{{- '\n</tools>\nReturn each function call as <tool_call>{"name":"function_name","arguments":{}}</tool_call>. Use only declared functions.\n' -}}
{%- if tool_choice == 'required' -%}{{- 'You must call at least one tool.\n' -}}
{%- elif tool_choice == 'function' -%}{{- 'You must call ' + tool_choice_function_name + '.\n' -}}{%- endif -%}
{{- '<|im_end|>\n' -}}
{%- endif -%}
{%- for message in messages -%}
{{- '<|im_start|>' + message.role + '\n' -}}
{%- if message.role == 'tool' -%}
{{- '<tool_response>\n{"tool_call_id":' + (message.tool_call_id | tojson) + ',"content":' + (message.content | tojson) + '}\n</tool_response>' -}}
{%- else -%}
{{- message.content -}}
{%- if message.tool_calls is defined -%}
{%- for call in message.tool_calls -%}
{{- '\n<tool_call>\n{"id":' + (call.id | tojson) + ',"name":' + (call.function.name | tojson) + ',"arguments":' + (call.function.arguments | tojson) + '}\n</tool_call>' -}}
{%- endfor -%}{%- endif -%}
{%- endif -%}
{{- '<|im_end|>\n' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{- '<|im_start|>assistant\n' -}}
{%- if enable_thinking -%}{{- '<think>\n' -}}
{%- else -%}{{- '<think>\n</think>\n' -}}{%- endif -%}
{%- endif -%}
)JINJA";
}

std::string Trim(std::string_view text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string_view::npos) return {};
    return std::string(text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1));
}

chat::ChatToolCall ParseToolEvent(const chat::StructuredStreamEvent& event, const std::string& requestId) {
    try {
        // The legacy parser deliberately recovers missing outer tags. HTTP
        // requires a complete block before publishing a callable object.
        constexpr std::string_view open = "<tool_call>", close = "</tool_call>";
        const auto& text = event.text;
        if (text.size() < open.size() + close.size() || text.compare(0, open.size(), open) != 0 ||
            text.compare(text.size() - close.size(), close.size(), close) != 0) Invalid("", "Incomplete tool block");
        const auto inner = Trim(std::string_view(text).substr(open.size(), text.size() - open.size() - close.size()));
        chat::ChatToolCall call;
        if (!inner.empty() && inner.front() == '{') {
            const auto json = ParseBody(inner, kProtocolMaxBodyBytes);
            Fields(json, {"id", "name", "arguments"});
            if (json.isMember("id")) Identifier(json["id"], "id", 160);
            call.function.name = Identifier(json["name"], "name");
            // Qwen JSON templates use object arguments. Also accept a string
            // containing that object, as in a Chat Completions history call.
            if (json["arguments"].isObject()) {
                // Validate with the strict reader, but forward the original
                // object bytes. Re-serializing a DOM would round decimal
                // arguments to double precision or normalize signed zero.
                simdjson::ondemand::parser rawParser;
                const simdjson::padded_string padded(inner);
                simdjson::ondemand::document document;
                std::string_view arguments;
                if (rawParser.iterate(padded).get(document) != simdjson::SUCCESS ||
                    document["arguments"].raw_json().get(arguments) != simdjson::SUCCESS) Invalid("arguments", "Missing raw arguments");
                call.function.arguments_json = ToolArguments(Json::Value(std::string(arguments)), "arguments");
            } else {
                call.function.arguments_json = ToolArguments(json["arguments"], "arguments");
            }
        } else {
            if (event.tool_calls.size() != 1) Invalid("", "Malformed XML tool block");
            call = event.tool_calls.front();
            Identifier(Json::Value(call.function.name), "name");
            if (inner.rfind("<function=", 0) == 0) {
                const auto end = inner.find("</function>");
                if (end == std::string::npos || end + 11 != inner.size()) Invalid("", "Extra content in XML tool block");
            } else {
                // Reject permissive legacy prefix/suffix recovery, including
                // text after a self-closing named tag.
                const auto start = "<" + call.function.name;
                const auto end = inner.find('>');
                if (inner.rfind(start, 0) != 0 || end == std::string::npos ||
                    (end > 0 && inner[end - 1] == '/' && end + 1 != inner.size())) Invalid("", "Extra content in XML tool block");
            }
            call.function.arguments_json = ToolArguments(Json::Value(call.function.arguments_json), "arguments");
        }
        return call;
    } catch (const ApiError& error) {
        static const bool diagnostics = aila::env::read_int_raw("AILA_SERVER_TOOL_DIAGNOSTICS", 0) != 0;
        if (diagnostics) {
            const auto block = SafeDiagnostic(JsonText(Json::Value(event.text)), 4096);
            AILA_LOG_WARN("[HTTP tools] request=%s reason=%s block=%s", requestId.c_str(), error.what(), block.c_str());
        }
        throw ApiError(502, "invalid_tool_call", "Model tool output is invalid: " + SafeDiagnostic(error.what(), 256));
    }
}

bool StartsInReasoning(const std::string& prompt) {
    const auto open = prompt.rfind("<think>");
    const auto close = prompt.rfind("</think>");
    if (open == std::string::npos || (close != std::string::npos && close > open)) return false;
    return prompt.find_first_not_of(" \t\r\n", open + 7) == std::string::npos;
}

Json::Value Usage(uint64_t prompt, uint64_t generated, uint64_t cached) {
    Json::Value usage(Json::objectValue);
    usage["prompt_tokens"] = Json::UInt64(prompt);
    usage["completion_tokens"] = Json::UInt64(generated);
    usage["total_tokens"] = Json::UInt64(prompt + generated);
    if (cached) usage["prompt_tokens_details"]["cached_tokens"] = Json::UInt64(cached);
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
    // The published absolute profile is the elementwise maximum of both
    // optional extensions, so a server may be configured for either or both.
    const auto absolute = wire::absolute_limits();
    if (settings.modelId.empty() || settings.modelId.size() > 256 || !ValidUtf8(settings.modelId) ||
        settings.maxContextTokens == 0 || settings.maxContextTokens > static_cast<uint32_t>((std::numeric_limits<int>::max)()) ||
        settings.maxPromptTokens == 0 || settings.maxPromptTokens > absolute.maxPromptTokens ||
        settings.maxNewTokens == 0 || settings.maxNewTokens > absolute.maxGeneratedTokens ||
        settings.prefillChunkSize == 0 || settings.prefillChunkSize > absolute.maxPrefillChunkSize ||
        settings.maxBodyBytes < kMinimumBodyBytes || settings.maxBodyBytes > kProtocolMaxBodyBytes ||
        settings.maxOutputBytes < kMinimumOutputBytes) {
        throw ApiError(500, "invalid_server_configuration", "Server chat limits or model identity are invalid");
    }
    const auto root = ParseBody(body, settings.maxBodyBytes);
    Fields(root, {"model", "messages", "stream", "stream_options", "max_tokens", "max_completion_tokens",
                  "temperature", "top_p", "top_k", "seed", "repetition_penalty", "presence_penalty",
                  "frequency_penalty", "n", "enable_thinking", "max_thinking_tokens",
                  "thinking_budget", "thinking_budget_tokens", "tools", "tool_choice", "parallel_tool_calls",
                  "store", "reasoning_effort"});
    // The PI coding agent (openai-completions) always sends store:false, which
    // this stateless server ignores. Storing a conversation is not supported,
    // so store:true is rejected rather than silently dropped. reasoning_effort
    // only switches the template's enable_thinking channel; it is not a
    // thinking-token budget.
    if (root.isMember("store") && !root["store"].isNull() && Boolean(root["store"], "store")) {
        throw ApiError(400, "unsupported_parameter", "Server-side response storage is not supported", "store");
    }
    if (root.isMember("reasoning_effort") && !root["reasoning_effort"].isNull()) {
        static const std::unordered_set<std::string> efforts = {
            "off", "none", "minimal", "low", "medium", "high", "xhigh", "max"};
        if (!efforts.count(String(root["reasoning_effort"], "reasoning_effort", false, 16))) {
            Invalid("reasoning_effort", "Expected off, minimal, low, medium, high, xhigh or max");
        }
    }
    const auto requestedModel = String(root["model"], "model", false, 256);
    if (requestedModel != settings.modelId) throw ApiError(404, "model_not_found", "Requested model is not served by this process", "model");
    ValidateMessages(root["messages"]);
    ValidateTools(root);
    PreparedChat prepared;
    if (root.isMember("parallel_tool_calls")) prepared.parallelToolCalls = Boolean(root["parallel_tool_calls"], "parallel_tool_calls");
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
    generation.temperature = settings.defaultTemperature;
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
    const char* budgetField = nullptr;
    std::optional<uint32_t> thinkingBudget;
    for (const char* field : {"max_thinking_tokens", "thinking_budget", "thinking_budget_tokens"}) {
        if (!root.isMember(field)) continue;
        if (budgetField) Invalid(field, "Supply only one thinking budget field");
        budgetField = field;
        const auto& value = root[field];
        if (value.type() == Json::intValue && value.isInt64() && value.asInt64() == -1) continue;
        thinkingBudget = static_cast<uint32_t>(Unsigned(value, field, settings.maxNewTokens));
    }
    bool thinking = root.isMember("enable_thinking") ? Boolean(root["enable_thinking"], "enable_thinking")
        : thinkingBudget && *thinkingBudget > 0;
    // An explicit request effort enables reasoning unless it is explicitly off,
    // matching the checkpoint template's enable_thinking gate.
    if (!root.isMember("enable_thinking") && root.isMember("reasoning_effort") && !root["reasoning_effort"].isNull()) {
        const auto effort = root["reasoning_effort"].asString();
        thinking = effort != "off" && effort != "none";
    }
    if (thinkingBudget && *thinkingBudget == 0) thinking = false;

    // Only this normalized object reaches the existing permissive ChatJson
    // reader; request-provided template kwargs and overrides never do.
    Json::Value normalized(Json::objectValue);
    normalized["messages"] = root["messages"];
    for (const char* field : {"tools", "tool_choice"}) if (root.isMember(field)) normalized[field] = root[field];
    for (auto& message : normalized["messages"]) {
        if (message["content"].isNull()) message["content"] = "";
        if (!message["content"].isArray()) continue;
        std::string content;
        for (const auto& part : message["content"]) content += part["text"].asString();
        // Qwen's text templates expect message.content to be one string.
        message["content"] = std::move(content);
    }
    normalized["chat_template_kwargs"]["enable_thinking"] = thinking;
    std::string error;
    auto generationConfig = GenerationConfigFor(generation);
    if (thinkingBudget) generationConfig.thinking_budget_tokens = static_cast<int>(*thinkingBudget);
    if (!chat::parse_chat_request_json(JsonText(normalized), generationConfig, prepared.chatRequest, &error)) {
        throw ApiError(400, "invalid_messages", "Could not parse validated chat messages", "messages");
    }
    // PI's openai-completions client labels the leading instruction message
    // "developer". Only the leading one is an instruction: the trusted Qwen
    // checkpoint template rejects a later developer message, and the service
    // fallbacks accept a leading system message. A leading developer therefore
    // becomes that system message; an interior developer stays a developer for
    // the Hybrid template and keeps its original rejection elsewhere.
    if (!prepared.chatRequest.messages.empty() &&
        prepared.chatRequest.messages.front().role == chat::ChatRole::Developer) {
        prepared.chatRequest.messages.front().role = chat::ChatRole::System;
    }
    if (tokenizer.vocab_size() <= 0) throw ApiError(500, "tokenizer_unavailable", "Server tokenizer is not loaded");

    // Select the trusted checkpoint template (or the fixed service fallback)
    // explicitly so environment/request paths cannot override the HTTP route.
    auto renderRequest = prepared.chatRequest;
    const bool hasToolHistory = std::any_of(renderRequest.messages.begin(), renderRequest.messages.end(),
        [](const chat::ChatMessage& message) { return !message.tool_calls.empty() || message.role == chat::ChatRole::Tool; });
    const bool useToolFallback = hasToolHistory || (!renderRequest.tools.empty() && renderRequest.tool_choice != chat::ToolChoice::None);
    // Qwen4Exp requires its checkpoint template; there is no service fallback.
    if (settings.family == ModelFamily::Qwen4Exp && tokenizer.chat_template().empty())
        throw ApiError(500, "invalid_server_configuration",
                       "Qwen4Exp requires the checkpoint chat_template; the service has no Qwen4Exp fallback");
    renderRequest.template_options.template_override_text = !tokenizer.chat_template().empty() ? tokenizer.chat_template()
        : settings.family == ModelFamily::Qwen35Hybrid ? chat::qwen35_fixed_chat_template()
        : useToolFallback ? ToolFallbackTemplate() : FallbackTemplate();
    chat::ChatFormatInput input;
    input.request = &renderRequest;
    input.family = settings.family;
    input.tokenizer_chat_template = tokenizer.chat_template();
    input.bos_token = tokenizer.bos_token();
    input.eos_token = tokenizer.eos_token();
    chat::ChatFormatTextResult rendered;
    if (!chat::ChatFormatter{}.render_text(input, true, rendered, &error) || !ValidUtf8(rendered.text)) {
        throw ApiError(500, "chat_template_failed", "The model chat template could not render this text request");
    }
    if (rendered.text.size() > settings.maxBodyBytes) throw ApiError(413, "prompt_too_large", "Rendered prompt exceeds the server byte limit", "messages");
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
    if (prepared.options.stopTokenIds.empty()) throw ApiError(500, "tokenizer_failed", "Model tokenizer has no valid EOS token");
    if (thinkingBudget) {
        wire::ThinkingBudgetOptions budget;
        budget.maxTokens = *thinkingBudget;
        budget.openTokenId = tokenizer.special_token_id("<think>");
        budget.closeTokenId = tokenizer.special_token_id("</think>");
        budget.promptOpensThink = prepared.startsInReasoning;
        budget.closeTokenIds.push_back(budget.closeTokenId);
        const auto suffix = tokenizer.encode("\n\n");
        budget.closeTokenIds.insert(budget.closeTokenIds.end(), suffix.begin(), suffix.end());
        if (aila::generation::ValidateThinkingBudget(budget, static_cast<uint32_t>(tokenizer.vocab_size()),
                                                    prepared.options.stopTokenIds)) {
            throw ApiError(400, "unsupported_thinking_budget", "Model tokenizer cannot encode the required thinking markers", budgetField);
        }
        prepared.options.thinkingBudget = std::move(budget);
    }
    prepared.options.prefillChunkSize = std::min(settings.prefillChunkSize, settings.maxPromptTokens);
    return prepared;
}

std::string DecodeChatToken(const Tokenizer& tokenizer, const PreparedChat& prepared, int32_t tokenId) {
    if (!prepared.chatRequest.tools.empty() && prepared.chatRequest.tool_choice != chat::ToolChoice::None) {
        for (const char* marker : {"<tool_call>", "</tool_call>"}) {
            const int markerId = tokenizer.special_token_id(marker);
            if (markerId >= 0 && tokenId == markerId) return marker;
        }
    }
    return tokenizer.decode(tokenId);
}

struct ChatOutput::State {
    ChatIdentity identity;
    bool stream;
    bool includeUsage;
    bool startsInReasoning;
    uint64_t promptTokens;
    uint64_t cachedPromptTokens;
    uint32_t maxGenerated;
    size_t maximum;
    size_t normalLimit;
    size_t emittedBytes = 0;
    bool begun = false;
    bool finished = false;
    bool toolBlockSeen = false;
    bool extractTools;
    bool parallelToolCalls;
    bool requireTool;
    std::string forcedFunction;
    std::unordered_set<std::string> declaredTools;
    Json::Value toolCalls{Json::arrayValue};
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
          cachedPromptTokens(prepared.cachedPromptTokens),
          maxGenerated(prepared.options.generation.maxNewTokens), maximum(limit), normalLimit(limit - kErrorReserveBytes),
          extractTools(!prepared.chatRequest.tools.empty() && prepared.chatRequest.tool_choice != chat::ToolChoice::None),
          parallelToolCalls(prepared.parallelToolCalls),
          requireTool(prepared.chatRequest.tool_choice == chat::ToolChoice::Required || prepared.chatRequest.tool_choice == chat::ToolChoice::Function),
          forcedFunction(prepared.chatRequest.tool_choice == chat::ToolChoice::Function ? prepared.chatRequest.tool_choice_function_name : "") {
        for (const auto& tool : prepared.chatRequest.tools) declaredTools.insert(tool.name);
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

    void AddTool(const chat::StructuredStreamEvent& event, std::vector<std::string>& frames) {
        const auto call = ParseToolEvent(event, identity.id);
        if (!declaredTools.count(call.function.name)) throw ApiError(502, "invalid_tool_call", "Model called a function that was not declared in tools");
        if ((!forcedFunction.empty() && call.function.name != forcedFunction) || (!parallelToolCalls && !toolCalls.empty())) {
            throw ApiError(502, "tool_choice_violation", "Model output violates tool_choice or parallel_tool_calls");
        }
        if (toolCalls.size() >= kMaxTools) throw ApiError(502, "tool_call_limit_exceeded", "Model returned more than 128 function calls");
        const auto index = toolCalls.size();
        Json::Value value(Json::objectValue);
        value["id"] = "call_" + identity.id + "_" + std::to_string(index);
        value["type"] = "function";
        value["function"]["name"] = call.function.name;
        value["function"]["arguments"] = call.function.arguments_json;
        toolCalls.append(value);
        if (!stream) return;
        // Validate the complete arguments before streaming. Bound each delta
        // independently of model tokenization and JSON escaping so it fits
        // the HTTP handoff's 16 KiB frame limit. Only the first carries ID/name.
        const auto& arguments = call.function.arguments_json;
        for (size_t offset = 0; offset < arguments.size();) {
            size_t end = offset;
            while (end < arguments.size()) {
                const auto length = static_cast<size_t>(ScalarLength(arguments, end));
                if (end + length - offset > kToolArgumentDeltaBytes) break;
                end += length;
            }
            Json::Value deltaCall(Json::objectValue);
            deltaCall["index"] = index;
            if (offset == 0) {
                deltaCall["id"] = value["id"];
                deltaCall["type"] = "function";
                deltaCall["function"]["name"] = call.function.name;
            }
            deltaCall["function"]["arguments"] = arguments.substr(offset, end - offset);
            Json::Value delta(Json::objectValue);
            delta["tool_calls"].append(std::move(deltaCall));
            AddDelta(delta, frames);
            offset = end;
        }
    }

    void AppendEvents(const std::vector<chat::StructuredStreamEvent>& events, std::string& deltaText,
                      std::string& deltaField, std::vector<std::string>& frames) {
        for (const auto& event : events) {
            if (event.type != chat::StructuredStreamEventType::ContentDelta &&
                event.type != chat::StructuredStreamEventType::ReasoningDelta &&
                event.type != chat::StructuredStreamEventType::ToolCallDelta) continue;
            if (event.type == chat::StructuredStreamEventType::ToolCallDelta && extractTools) {
                FlushDelta(deltaText, deltaField, frames);
                AddTool(event, frames);
                continue;
            }
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
        // A long-context request exceeds the default/hello wire bounds, so the
        // summary is validated against the prepared prompt and generation
        // capacities instead of the legacy 4096/1024 defaults.
        wire::Limits summaryLimits;
        summaryLimits.maxPromptTokens = static_cast<uint32_t>(std::min<uint64_t>(promptTokens,
            (std::numeric_limits<uint32_t>::max)()));
        summaryLimits.maxGeneratedTokens = maxGenerated;
        if (!wire::validate_summary(summary, validation, summaryLimits) || summary.promptTokensProcessed != promptTokens || summary.generatedTokens > maxGenerated) {
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
            if (requireTool && toolCalls.empty()) throw ApiError(502, "tool_choice_violation", "Model did not return the required function call");
            if (!extractTools) {
                // Preserve the existing no-tools/none response, including raw
                // unsolicited tool blocks and bounded recovery diagnostics.
                auto parsed = chat::parse_assistant_output(startsInReasoning ? "<think>" + rawText : rawText);
                if (toolBlockSeen || !parsed.tool_calls.empty()) warnings.emplace_back("Unsolicited tool_call output was retained as text; no tool was executed");
                for (const auto& warning : parsed.warnings) {
                    if (warnings.size() == 4) break;
                    warnings.push_back(SafeDiagnostic(warning, 256));
                }
            }
            const char* reason = summary.endReason == wire::EndReason::Length ? "length" : !toolCalls.empty() ? "tool_calls" : "stop";
            auto value = Envelope(false);
            Json::Value message(Json::objectValue);
            message["role"] = "assistant";
            message["content"] = content.empty() && !toolCalls.empty() ? Json::Value(Json::nullValue) : Json::Value(content);
            message["reasoning_content"] = reasoning;
            if (!toolCalls.empty()) message["tool_calls"] = toolCalls;
            Json::Value choice(Json::objectValue);
            choice["index"] = 0;
            choice["message"] = std::move(message);
            choice["finish_reason"] = reason;
            value["choices"].append(std::move(choice));
            value["usage"] = Usage(promptTokens, summary.generatedTokens, cachedPromptTokens);
            ChatCompletion result;
            result.json = JsonText(value);
            if (result.json.size() > normalLimit) throw ApiError(502, "output_limit_exceeded", "Serialized chat output exceeds the configured byte limit");
            if (stream) {
                AddDelta(Json::Value(Json::objectValue), frames, reason);
                if (includeUsage) {
                    auto usageChunk = Envelope(true);
                    usageChunk["usage"] = Usage(promptTokens, summary.generatedTokens, cachedPromptTokens);
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
