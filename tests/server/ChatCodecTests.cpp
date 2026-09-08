#include "server/ChatCodec.hpp"
#include "utils/Tokenizer.hpp"

#include <json/json.h>
#include <simdjson.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
using namespace aila::server;
namespace wire = aila::ipc::v2;
namespace fs = std::filesystem;

void Expect(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Function>
ApiError ExpectError(Function function, int status, const std::string& code = {},
                     const std::string& param = {}) {
    try { function(); }
    catch (const ApiError& error) {
        Expect(error.httpStatus == status, "Unexpected HTTP status for " + error.code + ": " + error.what());
        Expect(code.empty() || error.code == code, "Unexpected API error code: " + error.code + ", expected " + code);
        Expect(param.empty() || error.param == param, "Unexpected error parameter: " + error.param);
        return error;
    }
    throw std::runtime_error("Expected ApiError was not thrown");
}

std::string Serialize(const Json::Value& value) {
    Json::StreamWriterBuilder builder;
    builder["indentation"] = "";
    builder["emitUTF8"] = true;
    return Json::writeString(builder, value);
}

Json::Value Parse(std::string_view text) {
    simdjson::dom::parser syntax;
    simdjson::dom::element element;
    Expect(syntax.parse(text.data(), text.size()).get(element) == simdjson::SUCCESS,
           "Output is not valid UTF-8 JSON");
    Json::CharReaderBuilder builder;
    builder["failIfExtra"] = true;
    builder["rejectDupKeys"] = true;
    const std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    Json::Value value;
    std::string error;
    Expect(reader->parse(text.data(), text.data() + text.size(), &value, &error), "Could not read output JSON: " + error);
    return value;
}

std::string Utf8(unsigned codepoint) {
    std::string text;
    if (codepoint <= 0x7f) text.push_back(static_cast<char>(codepoint));
    else if (codepoint <= 0x7ff) {
        text.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
        text.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else if (codepoint <= 0xffff) {
        text.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
        text.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        text.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else {
        text.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
        text.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
        text.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        text.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    }
    return text;
}

std::string Bytes(std::initializer_list<unsigned> bytes) {
    std::string result;
    for (unsigned byte : bytes) result.push_back(static_cast<char>(byte));
    return result;
}

class TempDirectory {
public:
    TempDirectory() : parent_(fs::canonical(fs::temp_directory_path())) {
        std::random_device random;
        for (int attempt = 0; attempt < 32; ++attempt) {
            const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
            const auto candidate = parent_ / ("aila-chat-codec-" + std::to_string(stamp) + "-" + std::to_string(random()));
            if (fs::create_directory(candidate)) {
                path = fs::canonical(candidate);
                return;
            }
        }
        throw std::runtime_error("Could not create an isolated tokenizer test directory");
    }
    ~TempDirectory() {
        // Only remove the uniquely created, owned directory under the resolved
        // temp root; never derive deletion targets from model/request paths.
        std::error_code error;
        if (!path.empty() && path.is_absolute() && path.parent_path() == parent_ &&
            path.filename().string().rfind("aila-chat-codec-", 0) == 0) fs::remove_all(path, error);
    }
    TempDirectory(const TempDirectory&) = delete;
    TempDirectory& operator=(const TempDirectory&) = delete;
    fs::path path;
private:
    fs::path parent_;
};

void Write(const fs::path& path, const std::string& text) {
    std::ofstream output(path, std::ios::binary);
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    Expect(output.good(), "Could not write tokenizer fixture " + path.string());
}

const char* TrustedTemplate() {
    return "TRUSTED|{%- for message in messages -%}"
           "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' -}}"
           "{%- endfor -%}{%- if add_generation_prompt -%}"
           "{{- '<|im_start|>assistant\\n' -}}"
           "{%- if enable_thinking -%}{{- '<think>\\n' -}}"
           "{%- else -%}{{- '<think>\\n</think>\\n' -}}{%- endif -%}{%- endif -%}";
}

void WriteTokenizer(const fs::path& path, const std::string& chatTemplate, bool withEos = true) {
    fs::create_directory(path);
    Json::Value vocab(Json::objectValue);
    unsigned extra = 256;
    for (unsigned byte = 0; byte < 256; ++byte) {
        const bool visible = (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) || byte >= 174;
        vocab[Utf8(visible ? byte : extra++)] = byte;
    }
    Json::Value tokenizer(Json::objectValue);
    tokenizer["added_tokens"] = Json::arrayValue;
    const std::vector<std::string> special = {"<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>"};
    for (size_t index = 0; index < special.size(); ++index) {
        if (!withEos && (index == 1 || index == 2)) continue;
        Json::Value token(Json::objectValue);
        token["content"] = special[index];
        token["id"] = static_cast<Json::UInt>(256 + index);
        token["special"] = true;
        tokenizer["added_tokens"].append(std::move(token));
    }
    Json::Value config(Json::objectValue);
    config["chat_template"] = chatTemplate;
    config["eos_token"] = "<|im_end|>";
    config["added_tokens_decoder"] = Json::objectValue;
    Write(path / "vocab.json", Serialize(vocab));
    Write(path / "merges.txt", "#version: 0.2\n");
    Write(path / "tokenizer.json", Serialize(tokenizer));
    Write(path / "tokenizer_config.json", Serialize(config));
}

void WriteHybridTokenizer(const fs::path& path, bool withVision) {
    WriteTokenizer(path, "");
    // Retain the real Qwen3.5-0.8B/4B marker IDs in an otherwise tiny byte
    // vocabulary. This catches accidental use of Qwen3's hard-coded IDs
    // without requiring checkpoints or a GPU in the CPU test suite.
    Json::Value tokenizer(Json::objectValue);
    tokenizer["added_tokens"] = Json::arrayValue;
    const std::vector<std::pair<const char*, int>> markers = {
        {"<|endoftext|>", 248044}, {"<|im_start|>", 248045}, {"<|im_end|>", 248046},
        {"<|vision_start|>", 248053}, {"<|vision_end|>", 248054}, {"<|image_pad|>", 248056},
        {"<think>", 248068}, {"</think>", 248069}
    };
    for (const auto& [content, id] : markers) {
        Json::Value token(Json::objectValue);
        token["content"] = content;
        token["id"] = id;
        token["special"] = id != 248068 && id != 248069;
        tokenizer["added_tokens"].append(std::move(token));
    }
    Write(path / "tokenizer.json", Serialize(tokenizer));

    // Metadata mirrors the text-only and vision-dense export layouts. These
    // tests cover the tokenizer/HTTP boundary, not Worker weight loading.
    Json::Value textConfig(Json::objectValue);
    textConfig["model_type"] = "qwen3_5_text";
    textConfig["eos_token_id"] = 248044;
    Json::Value config = textConfig;
    if (withVision) {
        config = Json::objectValue;
        config["model_type"] = "qwen3_5";
        config["architectures"].append("Qwen3_5ForConditionalGeneration");
        config["text_config"] = textConfig;
        config["vision_config"]["model_type"] = "qwen3_5_vision";
    } else {
        config["architectures"].append("Qwen3_5ForCausalLM");
    }
    Write(path / "config.json", Serialize(config));
}

struct Fixture {
    TempDirectory directory;
    Tokenizer tokenizer;
    Tokenizer fallback;
    Tokenizer noEos;
    Tokenizer brokenTemplate;
    Tokenizer hybrid;
    Tokenizer hybridVision;
    ChatSettings settings;

    Fixture() {
        settings.modelId = "tiny-qwen";
        WriteTokenizer(directory.path / "trusted", TrustedTemplate());
        WriteTokenizer(directory.path / "fallback", "");
        WriteTokenizer(directory.path / "no-eos", "safe", false);
        WriteTokenizer(directory.path / "broken", "{{");
        WriteHybridTokenizer(directory.path / "hybrid-text", false);
        WriteHybridTokenizer(directory.path / "hybrid-vision", true);
        Expect(tokenizer.load((directory.path / "trusted").string()), "Could not load trusted tokenizer fixture");
        Expect(fallback.load((directory.path / "fallback").string()), "Could not load fallback tokenizer fixture");
        Expect(noEos.load((directory.path / "no-eos").string()), "Could not load tokenizer without EOS");
        Expect(brokenTemplate.load((directory.path / "broken").string()), "Could not load tokenizer with invalid template");
        Expect(hybrid.load((directory.path / "hybrid-text").string()), "Could not load Hybrid text tokenizer fixture");
        Expect(hybridVision.load((directory.path / "hybrid-vision").string()), "Could not load Hybrid vision tokenizer fixture");
        Expect(tokenizer.vocab_size() == 261, "Byte tokenizer fixture has the wrong vocabulary size");
    }

    PreparedChat Prepare(const Json::Value& body) const { return PrepareChat(Serialize(body), tokenizer, settings); }
    PreparedChat Raw(std::string_view body) const { return PrepareChat(body, tokenizer, settings); }
};

Json::Value Request() {
    Json::Value root(Json::objectValue);
    root["model"] = "tiny-qwen";
    Json::Value message(Json::objectValue);
    message["role"] = "user";
    message["content"] = "Hi";
    root["messages"].append(std::move(message));
    return root;
}

std::string RawRequest(const std::string& fields = {}) {
    return std::string("{\"model\":\"tiny-qwen\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]") +
        (fields.empty() ? "" : "," + fields) + "}";
}

std::string RawContent(const std::string& text) {
    auto body = RawRequest();
    body.replace(body.find("Hi"), 2, text);
    return body;
}

PreparedChat OutputRequest(bool stream = true, bool usage = false, bool reasoning = false) {
    PreparedChat prepared;
    prepared.stream = stream;
    prepared.includeUsage = usage;
    prepared.startsInReasoning = reasoning;
    prepared.promptIds = {10, 20, 30};
    prepared.options.generation.maxNewTokens = 32;
    return prepared;
}

ChatIdentity Identity() { return {"chatcmpl-fixture", "tiny-qwen", 1800000000}; }

wire::Summary Summary(const PreparedChat& prepared, uint64_t generated = 5,
                      wire::EndReason reason = wire::EndReason::StopToken) {
    wire::Summary summary;
    summary.status = wire::RequestStatus::Completed;
    summary.endReason = reason;
    summary.promptTokensProcessed = prepared.promptIds.size();
    summary.generatedTokens = generated;
    summary.eventCount = generated + 1;
    return summary;
}

void Append(std::vector<std::string>& destination, std::vector<std::string> source) {
    for (auto& frame : source) destination.push_back(std::move(frame));
}

size_t FrameBytes(const std::vector<std::string>& frames) {
    size_t bytes = 0;
    for (const auto& frame : frames) bytes += frame.size();
    return bytes;
}

struct StreamView {
    std::string content;
    std::string reasoning;
    std::string finishReason;
    size_t done = 0;
    size_t roles = 0;
    size_t finishes = 0;
    size_t usageChunks = 0;
    size_t errors = 0;
    Json::Value usage;
    Json::Value error;
};

StreamView Inspect(const std::vector<std::string>& frames, bool includeUsage) {
    StreamView result;
    for (const auto& frame : frames) {
        Expect(result.done == 0, "An SSE frame was emitted after [DONE]");
        if (frame == "data: [DONE]\n\n") { ++result.done; continue; }
        Expect(frame.size() >= 8 && frame.rfind("data: ", 0) == 0 && frame.substr(frame.size() - 2) == "\n\n",
               "Malformed SSE event framing");
        const auto value = Parse(std::string_view(frame).substr(6, frame.size() - 8));
        if (value.isMember("error")) {
            ++result.errors;
            result.error = value["error"];
            Expect(!value.isMember("choices") && !value.isMember("usage"), "Error frame fabricated choices or usage");
            continue;
        }
        const auto identity = Identity();
        Expect(value["id"].asString() == identity.id && value["model"].asString() == identity.model &&
               value["created"].asInt64() == identity.createdUnixSeconds &&
               value["object"].asString() == "chat.completion.chunk", "SSE identity changed between chunks");
        Expect(value["choices"].isArray(), "SSE choices must be an array");
        if (value["choices"].empty()) {
            Expect(includeUsage && result.finishes == 1 && value["usage"].isObject(), "Unexpected usage-only SSE chunk");
            result.usage = value["usage"];
            ++result.usageChunks;
            continue;
        }
        Expect(result.finishes == 0, "A non-usage chunk followed the finish chunk");
        Expect(value["choices"].size() == 1, "Text chat must have exactly one choice");
        const auto& choice = value["choices"][0];
        Expect(choice["index"].asInt() == 0 && choice["delta"].isObject() && !choice.isMember("message"),
               "Stream choice must contain a delta, not a message");
        const auto& delta = choice["delta"];
        Expect(!delta.isMember("tool_calls"), "Text-only SSE exposed a tool invocation");
        if (delta.isMember("role")) {
            Expect(result.roles == 0 && result.content.empty() && result.reasoning.empty(), "Assistant role was not first");
            Expect(delta["role"].asString() == "assistant" && delta["content"].asString().empty(), "Invalid initial role delta");
            ++result.roles;
        }
        if (delta.isMember("content")) result.content += delta["content"].asString();
        if (delta.isMember("reasoning_content")) result.reasoning += delta["reasoning_content"].asString();
        if (!choice["finish_reason"].isNull()) {
            Expect(delta.empty(), "Finish chunk must have an empty delta");
            result.finishReason = choice["finish_reason"].asString();
            ++result.finishes;
        }
        if (includeUsage) Expect(value.isMember("usage") && value["usage"].isNull(), "Ordinary SSE chunks require usage:null");
        else Expect(!value.isMember("usage"), "Unrequested SSE usage was emitted");
    }
    return result;
}

void CheckFailure(const ChatCompletion& completion, const std::vector<std::string>& frames,
                  const std::string& code, int status, bool includeUsage = false) {
    Expect(completion.failed && completion.httpStatus == status, "Failed generation was reported as successful");
    const auto value = Parse(completion.json);
    Expect(value["error"]["code"].asString() == code && !value.isMember("choices") && !value.isMember("usage"),
           "Failure JSON has the wrong schema/code");
    const auto stream = Inspect(frames, includeUsage);
    Expect(stream.done == 1 && stream.errors == 1 && stream.finishes == 0 && stream.usageChunks == 0,
           "Failed SSE must contain one error and [DONE], without a success finish or usage");
    Expect(stream.error["code"].asString() == code, "SSE and nonstream error codes differ");
}

struct OutputRun {
    ChatCompletion completion;
    std::vector<std::string> frames;
};

OutputRun RunOutput(const PreparedChat& prepared, const std::string& text, size_t chunkSize,
                    wire::EndReason reason = wire::EndReason::StopToken) {
    ChatOutput output(Identity(), prepared);
    OutputRun run;
    Append(run.frames, output.Begin());
    for (size_t offset = 0; offset < text.size(); offset += chunkSize) {
        Append(run.frames, output.PushText(std::string_view(text).substr(offset, chunkSize)));
    }
    run.completion = output.Finish(Summary(prepared, 5, reason));
    Append(run.frames, run.completion.sseFrames);
    Expect(!run.completion.failed && run.completion.httpStatus == 200, "Valid output failed: " + run.completion.json);
    return run;
}

OutputRun RunOutput(const std::string& text, size_t chunkSize, bool stream = true,
                    bool usage = true, bool reasoning = false) {
    return RunOutput(OutputRequest(stream, usage, reasoning), text, chunkSize);
}

void DefaultsAndTemplate(const Fixture& fixture) {
    const auto prepared = fixture.Prepare(Request());
    const auto& generation = prepared.options.generation;
    Expect(!prepared.stream && !prepared.includeUsage && !prepared.startsInReasoning, "Wrong default stream/thinking behavior");
    Expect(generation.maxNewTokens == 512 && generation.doSample && generation.temperature == 0.6 &&
           generation.topP == 0.95 && generation.topK == 20 && !generation.useFixedSeed &&
           generation.repetitionPenalty == 1 && generation.presencePenalty == 0 && generation.frequencyPenalty == 0,
           "HTTP defaults differ from the generation contract");
    Expect(prepared.chatRequest.template_options.enable_thinking == false &&
           prepared.chatRequest.generation_config.thinking_budget_tokens == -1, "Thinking defaults were not explicit");
    Expect(prepared.chatRequest.template_options.template_override_text.empty() &&
           prepared.chatRequest.template_options.template_override_path.empty(), "Internal template override escaped into the request");
    const std::string expected = "TRUSTED|<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n";
    const auto ids = fixture.tokenizer.encode(expected);
    Expect(std::vector<int32_t>(ids.begin(), ids.end()) == prepared.promptIds, "Trusted checkpoint template was not used exactly");
    Expect(prepared.options.stopTokenIds == std::vector<int32_t>({257, 258}), "EOS IDs were not validated/deduplicated");
    Expect(prepared.options.prefillChunkSize == 512, "Prefill chunk exceeds the wire limit");
    const auto whitespace = fixture.Raw(" \r\n\t" + RawRequest() + "\n");
    Expect(whitespace.promptIds == prepared.promptIds, "Legal JSON whitespace changed the request");
}

void SupportedParameters(const Fixture& fixture) {
    auto body = Request();
    body["stream"] = true;
    body["stream_options"]["include_usage"] = true;
    body["max_completion_tokens"] = 17;
    body["temperature"] = 0;
    body["top_p"] = 0.875;
    body["top_k"] = 0;
    body["seed"] = Json::UInt64((std::numeric_limits<uint64_t>::max)());
    body["repetition_penalty"] = 1.5;
    body["presence_penalty"] = -2.25;
    body["frequency_penalty"] = 3.25;
    body["n"] = 1;
    body["enable_thinking"] = true;
    const auto prepared = fixture.Prepare(body);
    const auto& generation = prepared.options.generation;
    Expect(prepared.stream && prepared.includeUsage && prepared.startsInReasoning, "Stream or prompt-open reasoning flags were lost");
    Expect(generation.maxNewTokens == 17 && !generation.doSample && generation.temperature == 0 &&
           generation.topP == 0.875 && generation.topK == 0 && generation.useFixedSeed &&
           generation.samplingSeed == (std::numeric_limits<uint64_t>::max)() && generation.repetitionPenalty == 1.5 &&
           generation.presencePenalty == -2.25 && generation.frequencyPenalty == 3.25, "Generation options were not mapped exactly");
    const auto& config = prepared.chatRequest.generation_config;
    Expect(config.sampling_seed == generation.samplingSeed && config.use_fixed_seed && !config.do_sample &&
           config.max_new_tokens == 17 && config.top_k == 0 && config.repetition_penalty == 1.5f,
           "Legacy ChatRequest lost the uint64 seed or explicit generation settings");
    body.removeMember("max_completion_tokens");
    body["max_tokens"] = 1;
    body["seed"] = 0;
    body["top_k"] = (std::numeric_limits<int32_t>::max)();
    body["top_p"] = 1;
    body["temperature"] = 3;
    const auto zeroSeed = fixture.Prepare(body);
    Expect(zeroSeed.options.generation.useFixedSeed && zeroSeed.options.generation.samplingSeed == 0 &&
           zeroSeed.options.generation.maxNewTokens == 1 && zeroSeed.options.generation.doSample,
           "Zero seed, legacy token limit, or valid engine sampling range was rejected");
    auto settings = fixture.settings;
    settings.maxNewTokens = 16;
    Expect(PrepareChat(Serialize(Request()), fixture.tokenizer, settings).options.generation.maxNewTokens == 16,
           "Default generation length did not obey the service cap");
}

void MessagesAndContentParts(const Fixture& fixture) {
    auto body = Request();
    body["messages"] = Json::arrayValue;
    for (const char* role : {"system", "developer", "user", "assistant"}) {
        Json::Value message(Json::objectValue);
        message["role"] = role;
        message["content"] = std::string(role) + " text";
        if (std::string(role) == "assistant") message["reasoning_content"] = "previous reasoning";
        body["messages"].append(message);
    }
    auto& user = body["messages"][2];
    user["name"] = "visitor";
    user["content"] = Json::arrayValue;
    Json::Value part(Json::objectValue);
    part["type"] = "text";
    part["text"] = "hello ";
    user["content"].append(part);
    part["text"] = Utf8(0x4e16) + Utf8(0x754c);
    user["content"].append(part);
    const auto prepared = fixture.Prepare(body);
    Expect(prepared.chatRequest.messages.size() == 4 && prepared.chatRequest.messages[1].role == aila::chat::ChatRole::Developer,
           "Supported message roles were not retained");
    const auto& normalized = prepared.chatRequest.messages[2];
    Expect(normalized.name == "visitor" && normalized.content.size() == 1 &&
           normalized.content[0].text == "hello " + Utf8(0x4e16) + Utf8(0x754c),
           "Text parts were not concatenated for the text-only template");
    Expect(prepared.chatRequest.messages[3].reasoning_content == "previous reasoning", "Assistant reasoning history was lost");
    const auto escaped = fixture.Raw(RawContent("\\ud83d\\ude42"));
    Expect(escaped.chatRequest.messages[0].content[0].text == Utf8(0x1f642), "Valid surrogate pair did not decode to one scalar");
    auto empty = Request();
    empty["messages"][0]["content"] = "";
    Expect(!fixture.Prepare(empty).promptIds.empty(), "Empty text content should still produce a generation prompt");
}

void StrictJson(const Fixture& fixture) {
    std::vector<std::string> invalid = {
        "", "null", "[]", "true", RawRequest() + "{}", RawRequest() + " garbage",
        RawRequest("\"model\":\"tiny-qwen\""), RawRequest("\"\\u006dodel\":\"tiny-qwen\""),
        RawRequest("\"temperature\":NaN"), RawRequest("\"temperature\":Infinity"),
        RawRequest("\"temperature\":01"), RawRequest("\"temperature\":+1"),
        RawRequest("\"temperature\":1."), RawRequest("\"temperature\":.5"),
        RawRequest("\"temperature\":1e999"), RawRequest("\"seed\":18446744073709551616"),
        RawContent("\\ud800"), RawContent("\\udc00"), RawContent("\\ud800\\u0041"),
        RawContent("\\x41"), RawContent("line\nbreak"), RawContent("tab\tcharacter"),
        RawContent(std::string(1, '\0')), RawContent(Bytes({0xed, 0xa0, 0x80})),
        RawContent(Bytes({0xc0, 0xaf})), Bytes({0xef, 0xbb, 0xbf}) + RawRequest(),
        RawRequest("\"top_k\":" + std::string(80, '[') + "0" + std::string(80, ']')),
        "{\"model\":\"tiny-qwen\",\"messages\":[{\"role\":\"user\",\"role\":\"assistant\",\"content\":\"Hi\"}]}",
        "{\"model\":\"tiny-qwen\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"a\",\"\\u0074ext\":\"b\"}]}]}"
    };
    auto comments = RawRequest(); comments.insert(1, "/* comment */"); invalid.push_back(comments);
    auto trailingComma = RawRequest(); trailingComma.insert(trailingComma.size() - 1, ","); invalid.push_back(trailingComma);
    auto singleQuotes = RawRequest(); std::replace(singleQuotes.begin(), singleQuotes.end(), '"', '\''); invalid.push_back(singleQuotes);
    for (size_t index = 0; index < invalid.size(); ++index) {
        try { ExpectError([&] { fixture.Raw(invalid[index]); }, 400, "invalid_json"); }
        catch (const std::exception& error) { throw std::runtime_error("Strict JSON case " + std::to_string(index) + ": " + error.what()); }
    }
    ExpectError([&] { fixture.Raw(std::string(1024 * 1024 + 1, ' ')); }, 413, "request_too_large");
}

void UnsupportedFields(const Fixture& fixture) {
    for (const char* field : {"tools", "tool_choice", "stop", "max_thinking_tokens", "thinking_budget", "thinking_budget_tokens",
                              "chat_template_kwargs", "template_override", "template_override_path", "response_format", "logprobs", "user"}) {
        auto body = Request(); body[field] = Json::nullValue;
        ExpectError([&] { fixture.Prepare(body); }, 400, "unsupported_parameter", field);
    }
    for (const char* field : {"tool_calls", "tool_call_id", "extra"}) {
        auto body = Request(); body["messages"][0][field] = "unsupported";
        ExpectError([&] { fixture.Prepare(body); }, 400, "unsupported_parameter", std::string("messages[0].") + field);
    }
    auto body = Request();
    body["stream"] = true; body["stream_options"]["extra"] = false;
    ExpectError([&] { fixture.Prepare(body); }, 400, "unsupported_parameter", "stream_options.extra");
    body = Request();
    body["messages"][0]["content"] = Json::arrayValue;
    Json::Value part(Json::objectValue); part["type"] = "image"; part["text"] = "payload";
    body["messages"][0]["content"].append(part);
    ExpectError([&] { fixture.Prepare(body); }, 400, "unsupported_content_type");
    body["messages"][0]["content"][0]["type"] = "text";
    body["messages"][0]["content"][0]["image_url"] = "https://example.invalid/image.png";
    ExpectError([&] { fixture.Prepare(body); }, 400, "unsupported_parameter");
}

void InvalidMessages(const Fixture& fixture) {
    std::vector<Json::Value> invalidMessages = {Json::Value(), Json::Value("text"), Json::Value(Json::arrayValue)};
    Json::Value tooMany(Json::arrayValue);
    for (size_t index = 0; index < 257; ++index) tooMany.append(Request()["messages"][0]);
    invalidMessages.push_back(std::move(tooMany));
    for (const auto& invalid : invalidMessages) {
        auto body = Request(); body["messages"] = invalid;
        ExpectError([&] { fixture.Prepare(body); }, 400, "invalid_parameter", "messages");
    }
    for (const char* role : {"tool", "invalid", "USER"}) {
        auto body = Request(); body["messages"][0]["role"] = role;
        ExpectError([&] { fixture.Prepare(body); }, 400, "unsupported_message_role", "messages[0].role");
    }
    for (const Json::Value& invalid : {Json::Value(), Json::Value(7), Json::Value(Json::objectValue), Json::Value(Json::arrayValue)}) {
        auto body = Request(); body["messages"][0]["content"] = invalid;
        ExpectError([&] { fixture.Prepare(body); }, 400, "invalid_parameter", "messages[0].content");
    }
    auto body = Request(); body["messages"][0]["reasoning_content"] = "not an assistant";
    ExpectError([&] { fixture.Prepare(body); }, 400, "invalid_parameter", "messages[0].reasoning_content");
    body = Request(); body["messages"][0]["name"] = std::string(65, 'x');
    ExpectError([&] { fixture.Prepare(body); }, 400, "invalid_parameter", "messages[0].name");
    body = Request(); body["messages"][0].removeMember("role");
    ExpectError([&] { fixture.Prepare(body); }, 400, "invalid_parameter", "messages[0].role");
    body = Request(); body["messages"][0] = "not an object";
    ExpectError([&] { fixture.Prepare(body); }, 400, "invalid_parameter", "messages[0]");
    body = Request(); body["messages"][0]["content"] = Json::arrayValue;
    Json::Value part(Json::objectValue); part["type"] = "text";
    body["messages"][0]["content"].append(part);
    ExpectError([&] { fixture.Prepare(body); }, 400, "invalid_parameter", "messages[0].content[0].text");
}

void InvalidParameters(const Fixture& fixture) {
    const std::vector<std::pair<const char*, std::string>> invalid = {
        {"stream", "1"}, {"stream", "null"}, {"enable_thinking", "\"false\""},
        {"max_tokens", "0"}, {"max_tokens", "-1"}, {"max_tokens", "1.5"}, {"max_tokens", "1.0"}, {"max_tokens", "1025"},
        {"max_completion_tokens", "false"}, {"temperature", "-1"}, {"temperature", "1e39"}, {"temperature", "1e-50"},
        {"temperature", "\"0.5\""}, {"top_p", "0"}, {"top_p", "1.01"}, {"top_p", "1e-50"},
        {"top_k", "-1"}, {"top_k", "2147483648"}, {"top_k", "1.0"},
        {"seed", "-1"}, {"seed", "1.5"}, {"seed", "1e2"}, {"seed", "null"},
        {"repetition_penalty", "0"}, {"repetition_penalty", "-0.5"}, {"repetition_penalty", "1e-50"},
        {"presence_penalty", "1e39"}, {"frequency_penalty", "false"}, {"n", "2"}, {"n", "0"}
    };
    for (const auto& [field, raw] : invalid) {
        ExpectError([&] { fixture.Raw(RawRequest("\"" + std::string(field) + "\":" + raw)); }, 400, {}, field);
    }
    auto body = Request(); body["max_tokens"] = 1; body["max_completion_tokens"] = 1;
    ExpectError([&] { fixture.Prepare(body); }, 400, "invalid_parameter", "max_completion_tokens");
    body = Request(); body["stream_options"]["include_usage"] = true;
    ExpectError([&] { fixture.Prepare(body); }, 400, "invalid_parameter", "stream_options");
    body["stream"] = true; body["stream_options"]["include_usage"] = "true";
    ExpectError([&] { fixture.Prepare(body); }, 400, "invalid_parameter", "stream_options.include_usage");
    body["stream_options"] = Json::nullValue;
    ExpectError([&] { fixture.Prepare(body); }, 400, "invalid_parameter", "stream_options");
    body = Request(); body["model"] = "another-model";
    ExpectError([&] { fixture.Prepare(body); }, 404, "model_not_found", "model");
    body.removeMember("model");
    ExpectError([&] { fixture.Prepare(body); }, 400, "invalid_parameter", "model");
}

void PromptAndServerLimits(const Fixture& fixture) {
    auto body = Request(); body["max_tokens"] = 1;
    const auto baseline = fixture.Prepare(body);
    auto settings = fixture.settings;
    settings.maxPromptTokens = static_cast<uint32_t>(baseline.promptIds.size());
    settings.maxContextTokens = settings.maxPromptTokens + 1;
    const auto exact = PrepareChat(Serialize(body), fixture.tokenizer, settings);
    Expect(exact.promptIds == baseline.promptIds, "Exact prompt/context boundary should be accepted without truncation");
    --settings.maxContextTokens;
    ExpectError([&] { PrepareChat(Serialize(body), fixture.tokenizer, settings); }, 400, "context_length_exceeded");
    settings.maxContextTokens = 4096; --settings.maxPromptTokens;
    ExpectError([&] { PrepareChat(Serialize(body), fixture.tokenizer, settings); }, 400, "invalid_parameter", "messages");
    const std::vector<std::function<void(ChatSettings&)>> invalid = {
        [](auto& s) { s.modelId.clear(); }, [](auto& s) { s.modelId = std::string(257, 'x'); },
        [](auto& s) { s.modelId = Bytes({0x80}); }, [](auto& s) { s.maxContextTokens = 0; },
        [](auto& s) { s.maxContextTokens = (std::numeric_limits<uint32_t>::max)(); },
        [](auto& s) { s.maxPromptTokens = 0; }, [](auto& s) { s.maxNewTokens = 0; },
        [](auto& s) { s.maxPromptTokens = wire::hard_limits().maxPromptTokens + 1; },
        [](auto& s) { s.maxNewTokens = wire::hard_limits().maxGeneratedTokens + 1; },
        [](auto& s) { s.maxOutputBytes = 4095; }
    };
    for (const auto& mutate : invalid) {
        settings = fixture.settings; mutate(settings);
        ExpectError([&] { PrepareChat(Serialize(body), fixture.tokenizer, settings); }, 500, "invalid_server_configuration");
    }
    Tokenizer unloaded;
    ExpectError([&] { PrepareChat(Serialize(body), unloaded, fixture.settings); }, 500, "tokenizer_unavailable");
    ExpectError([&] { PrepareChat(Serialize(body), fixture.noEos, fixture.settings); }, 500, "tokenizer_failed");
    ExpectError([&] { PrepareChat(Serialize(body), fixture.brokenTemplate, fixture.settings); }, 500, "chat_template_failed");
    const auto fallback = PrepareChat(Serialize(body), fixture.fallback, fixture.settings);
    const auto expected = fixture.fallback.encode("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n");
    Expect(fallback.promptIds == std::vector<int32_t>(expected.begin(), expected.end()), "Fixed fallback template was not applied");
}

ChatSettings HybridSettings(const Fixture& fixture) {
    auto settings = fixture.settings;
    settings.family = ModelFamily::Qwen35Hybrid;
    return settings;
}

std::string HybridGenerationPrompt(bool thinking) {
    return std::string("<|im_start|>assistant\n") + (thinking ? "<think>\n" : "<think>\n</think>\n");
}

void ExpectPrompt(const Tokenizer& tokenizer, const PreparedChat& prepared, const std::string& expected,
                  const std::string& message) {
    const auto ids = tokenizer.encode(expected);
    if (prepared.promptIds == std::vector<int32_t>(ids.begin(), ids.end())) return;
    std::string actual;
    for (const int32_t id : prepared.promptIds) {
        const auto raw = tokenizer.raw_token(id);
        actual += tokenizer.special_token_id(raw) == id ? raw : tokenizer.decode(id);
    }
    throw std::runtime_error(message + "\nExpected prompt: " + Serialize(Json::Value(expected)) +
                             "\nActual prompt: " + Serialize(Json::Value(actual)));
}

void HybridFallbackAndEos(const Fixture& fixture) {
    const auto settings = HybridSettings(fixture);
    Expect(fixture.settings.family == ModelFamily::Qwen3Dense, "Existing service settings no longer default to Qwen3");
    Expect(fixture.hybrid.chat_template().empty(), "Hybrid fallback fixture unexpectedly has a checkpoint template");
    Expect(fixture.hybrid.decode(248068) == "<think>" && fixture.hybrid.decode(248069) == "</think>" &&
           fixture.hybrid.decode(248044).empty() && fixture.hybrid.decode(248046).empty(),
           "Hybrid tokenizer must expose reasoning markers to ChatOutput and suppress terminal tokens");
    auto body = Request();
    body["messages"][0]["content"] = "  Hi  ";
    const auto omitted = PrepareChat(Serialize(body), fixture.hybrid, settings);
    Expect(!omitted.startsInReasoning && omitted.chatRequest.template_options.enable_thinking == false,
           "Hybrid HTTP thinking must default to false independent of model size");
    for (bool thinking : {false, true}) {
        body["enable_thinking"] = thinking;
        const auto prepared = PrepareChat(Serialize(body), fixture.hybrid, settings);
        ExpectPrompt(fixture.hybrid, prepared, "<|im_start|>user\nHi<|im_end|>\n" + HybridGenerationPrompt(thinking),
                     "Hybrid fallback did not render the fixed family template exactly");
        Expect(prepared.startsInReasoning == thinking && prepared.chatRequest.template_options.enable_thinking == thinking,
               "Hybrid thinking option and generation prefix disagree");
        if (!thinking) Expect(prepared.promptIds == omitted.promptIds, "Omitted and false Hybrid thinking differ");
        Expect(prepared.options.stopTokenIds == std::vector<int32_t>({248046, 248044}),
               "Hybrid must stop on both im_end and endoftext using checkpoint marker IDs");
        Expect(std::find(prepared.promptIds.begin(), prepared.promptIds.end(), 248068) != prepared.promptIds.end(),
               "Hybrid opening marker was split despite being an added token");
        Expect((std::find(prepared.promptIds.begin(), prepared.promptIds.end(), 248069) == prepared.promptIds.end()) == thinking,
               "Hybrid closing marker did not follow the thinking flag");
        Expect(prepared.chatRequest.template_options.template_override_text.empty() &&
               prepared.chatRequest.template_options.template_override_path.empty(), "Trusted Hybrid fallback leaked into public request options");
    }
}

void HybridHistoryAndTextParts(const Fixture& fixture) {
    const auto settings = HybridSettings(fixture);
    auto body = Request();
    body["messages"] = Json::arrayValue;
    const auto addMessage = [&](const char* role, const std::string& content) {
        Json::Value message(Json::objectValue);
        message["role"] = role;
        message["content"] = content;
        body["messages"].append(std::move(message));
    };
    addMessage("developer", "  Be brief.  ");
    addMessage("user", " First ");
    addMessage("assistant", "<think>\nprevious reasoning\n</think>\n\nFirst answer");
    addMessage("user", "Next");
    addMessage("assistant", "Partial answer");
    body["messages"][4]["reasoning_content"] = "current reasoning";
    const auto unicode = Utf8(0x4e16) + Utf8(0x754c);
    auto& parts = body["messages"][3]["content"];
    parts = Json::arrayValue;
    Json::Value part(Json::objectValue);
    part["type"] = "text";
    part["text"] = "  Next ";
    parts.append(part);
    part["text"] = unicode + "  ";
    parts.append(part);
    const std::string expectedHistory =
        "<|im_start|>system\nBe brief.<|im_end|>\n"
        "<|im_start|>user\nFirst<|im_end|>\n"
        "<|im_start|>assistant\n<think>\nprevious reasoning\n</think>\n\nFirst answer<|im_end|>\n"
        "<|im_start|>user\nNext " + unicode + "<|im_end|>\n"
        "<|im_start|>assistant\n<think>\ncurrent reasoning\n</think>\n\nPartial answer<|im_end|>\n";
    for (bool thinking : {false, true}) {
        body["enable_thinking"] = thinking;
        const auto prepared = PrepareChat(Serialize(body), fixture.hybrid, settings);
        ExpectPrompt(fixture.hybrid, prepared, expectedHistory + HybridGenerationPrompt(thinking),
                     "Hybrid history must preserve supplied assistant reasoning and normalize developer/text parts");
        // ChatTemplateOptions defaults to preserving history, and the engine
        // always passes that value to Jinja. The template's undefined-variable
        // fallback does not override the HTTP/CLI request default.
        Expect(prepared.chatRequest.template_options.preserve_thinking,
               "Hybrid HTTP history no longer follows the shared preserve_thinking default");
        Expect(prepared.chatRequest.messages[2].content[0].text.find("previous reasoning") != std::string::npos &&
               prepared.chatRequest.messages[3].content[0].text == "  Next " + unicode + "  " &&
               prepared.chatRequest.messages[4].reasoning_content == "current reasoning",
               "Rendering must not mutate normalized input history");
        Expect(prepared.startsInReasoning == thinking, "Historical reasoning changed the next generation channel");
    }
    body["messages"][2]["content"] = "First answer";
    body["messages"][2]["reasoning_content"] = "previous reasoning";
    const auto separate = PrepareChat(Serialize(body), fixture.hybrid, settings);
    ExpectPrompt(fixture.hybrid, separate, expectedHistory + HybridGenerationPrompt(true),
                 "Inline and separate assistant reasoning history must produce the same complete prompt");

    auto literal = Request();
    literal["messages"][0]["content"] = "Quote <think>user text</think> exactly.";
    const auto quoted = PrepareChat(Serialize(literal), fixture.hybrid, settings);
    ExpectPrompt(fixture.hybrid, quoted,
                 "<|im_start|>user\nQuote <think>user text</think> exactly.<|im_end|>\n" + HybridGenerationPrompt(false),
                 "Hybrid history rendering altered literal tags in a user message");
    Expect(!quoted.startsInReasoning, "A user-quoted reasoning marker became the output channel");
}

void HybridTrustedTemplate(const Fixture& fixture) {
    const auto settings = HybridSettings(fixture);
    for (bool thinking : {false, true}) {
        auto body = Request();
        body["enable_thinking"] = thinking;
        const auto prepared = PrepareChat(Serialize(body), fixture.tokenizer, settings);
        ExpectPrompt(fixture.tokenizer, prepared,
                     "TRUSTED|<|im_start|>user\nHi<|im_end|>\n" + HybridGenerationPrompt(thinking),
                     "A trusted checkpoint template must take precedence over the Hybrid fallback");
    }

    const std::string templatePrefix =
        "{%- for message in messages -%}"
        "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' -}}"
        "{%- endfor -%}{{- '<|im_start|>assistant\\n' -}}";
    struct Suffix {
        const char* source;
        const char* rendered;
        bool thinking;
        bool startsInReasoning;
    };
    const std::vector<Suffix> suffixes = {
        {"{{- '<think>\\n' -}}", "<think>\n", false, true},
        {"{{- '<think>\\n</think>\\n' -}}", "<think>\n</think>\n", true, false},
        {"", "", true, false}
    };
    TempDirectory directory;
    for (size_t index = 0; index < suffixes.size(); ++index) {
        const auto& suffix = suffixes[index];
        const auto path = directory.path / std::to_string(index);
        WriteTokenizer(path, templatePrefix + suffix.source);
        Tokenizer tokenizer;
        Expect(tokenizer.load(path.string()), "Could not load checkpoint generation-prefix fixture");
        auto body = Request();
        body["messages"][0]["content"] = "literal <think>";
        body["enable_thinking"] = suffix.thinking;
        const auto prepared = PrepareChat(Serialize(body), tokenizer, settings);
        ExpectPrompt(tokenizer, prepared,
                     std::string("<|im_start|>user\nliteral <think><|im_end|>\n<|im_start|>assistant\n") + suffix.rendered,
                     "Checkpoint generation suffix changed during Hybrid preparation");
        Expect(prepared.startsInReasoning == suffix.startsInReasoning,
               "Output reasoning state must come from the rendered suffix, not the request flag or a history marker");
    }
}

void HybridPreparedOutputParity(const Fixture& fixture) {
    const auto settings = HybridSettings(fixture);
    const std::string thought = "\n" + Utf8(0x63a8) + Utf8(0x7406) + Utf8(0x1f642) + "  \n";
    const std::string visible = "\n" + Utf8(0x7b54) + Utf8(0x6848) + " <tag>ok</tag>";
    struct Scenario {
        bool thinking;
        const char* control;
        bool startsInReasoning;
        std::string output;
        std::string reasoning;
        std::string content;
        wire::EndReason reason = wire::EndReason::StopToken;
    };
    const std::vector<Scenario> scenarios = {
        {false, "", false, visible, "", visible},
        {true, "", true, thought + "</think>" + visible, thought, visible},
        {false, "<|think_on|>", true, thought + "</think>" + visible, thought, visible},
        {true, "<|think_off|>", false, visible, "", visible},
        {true, "", true, thought, thought, "", wire::EndReason::Length}
    };
    for (const auto& scenario : scenarios) {
        auto body = Request();
        body["max_tokens"] = 32;
        body["enable_thinking"] = scenario.thinking;
        body["messages"][0]["content"] = std::string("Hi ") + scenario.control;
        const auto plainRequest = PrepareChat(Serialize(body), fixture.hybrid, settings);
        body["stream"] = true;
        body["stream_options"]["include_usage"] = true;
        const auto streamRequest = PrepareChat(Serialize(body), fixture.hybrid, settings);
        Expect(plainRequest.startsInReasoning == scenario.startsInReasoning &&
               streamRequest.startsInReasoning == scenario.startsInReasoning && plainRequest.promptIds == streamRequest.promptIds,
               "Prepared Hybrid stream/plain requests disagree on the rendered reasoning state");
        const auto nonstream = RunOutput(plainRequest, scenario.output, scenario.output.size(), scenario.reason);
        Expect(nonstream.frames.empty(), "Nonstream Hybrid completion emitted SSE");
        const auto expected = Parse(nonstream.completion.json);
        const auto& message = expected["choices"][0]["message"];
        Expect(message["content"].asString() == scenario.content && message["reasoning_content"].asString() == scenario.reasoning,
               "Hybrid generation prefix leaked reasoning into visible content or discarded text");
        for (size_t chunk : {size_t(1), size_t(2), size_t(7), scenario.output.size()}) {
            const auto streamed = RunOutput(streamRequest, scenario.output, chunk, scenario.reason);
            const auto view = Inspect(streamed.frames, true);
            Expect(view.content == scenario.content && view.reasoning == scenario.reasoning &&
                   view.roles == 1 && view.done == 1 && view.finishes == 1 && view.usageChunks == 1 && view.errors == 0,
                   "Token-fragmented Hybrid output differs from its prepared reasoning contract");
            Expect(view.finishReason == (scenario.reason == wire::EndReason::Length ? "length" : "stop") &&
                   view.usage == expected["usage"] && Parse(streamed.completion.json) == expected,
                   "Hybrid JSON and SSE completion, finish reason, or usage disagree");
        }
    }
}

void HybridVisionTextBoundary(const Fixture& fixture) {
    const auto settings = HybridSettings(fixture);
    auto body = Request();
    body["messages"][0]["content"] = Json::arrayValue;
    Json::Value text(Json::objectValue);
    text["type"] = "text";
    text["text"] = "Describe this text.";
    body["messages"][0]["content"].append(text);
    const auto textOnly = PrepareChat(Serialize(body), fixture.hybrid, settings);
    const auto visionExport = PrepareChat(Serialize(body), fixture.hybridVision, settings);
    Expect(visionExport.promptIds == textOnly.promptIds && visionExport.options.stopTokenIds == textOnly.options.stopTokenIds,
           "Vision architecture metadata changed the pure-text HTTP tokenizer path");
    for (const int marker : {248053, 248054, 248056}) {
        Expect(std::find(visionExport.promptIds.begin(), visionExport.promptIds.end(), marker) == visionExport.promptIds.end(),
               "Pure-text Hybrid request received an implicit vision marker");
    }
    for (const char* type : {"image", "image_url", "video", "audio", "input_audio"}) {
        auto media = body;
        media["messages"][0]["content"][0]["type"] = type;
        ExpectError([&] { PrepareChat(Serialize(media), fixture.hybridVision, settings); },
                    400, "unsupported_content_type", "messages[0].content[0].type");
    }
    auto image = body;
    image["messages"][0]["content"][0].removeMember("text");
    image["messages"][0]["content"][0]["type"] = "image_url";
    image["messages"][0]["content"][0]["image_url"]["url"] = "file:///not-loaded.png";
    ExpectError([&] { PrepareChat(Serialize(image), fixture.hybridVision, settings); },
                400, "unsupported_parameter", "messages[0].content[0].image_url");
}

void HybridLimitsAndTemplateFailures(const Fixture& fixture) {
    auto settings = HybridSettings(fixture);
    auto body = Request();
    body["max_tokens"] = 1;
    const auto baseline = PrepareChat(Serialize(body), fixture.hybrid, settings);
    settings.maxPromptTokens = static_cast<uint32_t>(baseline.promptIds.size());
    settings.maxContextTokens = settings.maxPromptTokens + 1;
    Expect(PrepareChat(Serialize(body), fixture.hybrid, settings).promptIds == baseline.promptIds,
           "Exact Hybrid prompt/context boundary was rejected");
    --settings.maxContextTokens;
    ExpectError([&] { PrepareChat(Serialize(body), fixture.hybrid, settings); }, 400, "context_length_exceeded");
    settings = HybridSettings(fixture);
    ExpectError([&] { PrepareChat(Serialize(body), fixture.brokenTemplate, settings); }, 500, "chat_template_failed");
    ExpectError([&] { PrepareChat(Serialize(body), fixture.noEos, settings); }, 500, "tokenizer_failed");
    body["chat_template_kwargs"]["enable_thinking"] = true;
    ExpectError([&] { PrepareChat(Serialize(body), fixture.hybrid, settings); },
                400, "unsupported_parameter", "chat_template_kwargs");
    body.removeMember("chat_template_kwargs");
    body["enable_thinking"] = 1;
    ExpectError([&] { PrepareChat(Serialize(body), fixture.hybrid, settings); },
                400, "invalid_parameter", "enable_thinking");
}

void NonstreamAndSseSchema() {
    const auto nonstream = RunOutput("hello", 1, false, false);
    Expect(nonstream.frames.empty(), "Nonstream output emitted SSE");
    const auto value = Parse(nonstream.completion.json);
    const auto& choice = value["choices"][0];
    Expect(value["id"].asString() == Identity().id && value["object"].asString() == "chat.completion" &&
           value["created"].asInt64() == Identity().createdUnixSeconds && value["model"].asString() == Identity().model,
           "Nonstream response identity/schema is incorrect");
    Expect(value["choices"].size() == 1 && choice["index"].asInt() == 0 && choice["finish_reason"].asString() == "stop" &&
           choice["message"]["role"].asString() == "assistant" && choice["message"]["content"].asString() == "hello" &&
           choice["message"]["reasoning_content"].asString().empty() && !choice.isMember("delta"), "Nonstream message is incorrect");
    Expect(value["usage"]["prompt_tokens"].asUInt64() == 3 && value["usage"]["completion_tokens"].asUInt64() == 5 &&
           value["usage"]["total_tokens"].asUInt64() == 8, "Usage must come from prompt IDs and verified summary counts");
    for (bool usage : {false, true}) {
        const auto streamed = RunOutput("hello", 2, true, usage);
        const auto view = Inspect(streamed.frames, usage);
        Expect(view.roles == 1 && view.done == 1 && view.finishes == 1 && view.finishReason == "stop" && view.errors == 0 &&
               view.content == "hello" && view.reasoning.empty() && view.usageChunks == (usage ? 1 : 0), "SSE sequence is incorrect");
        if (usage) Expect(view.usage == value["usage"], "SSE and nonstream usage differ");
        Expect(Parse(streamed.completion.json) == value, "Stream aggregation differs from nonstream output");
    }
}

void FragmentationAndReasoning() {
    const std::string thought = "\n" + Utf8(0x63a8) + Utf8(0x7406) + Utf8(0x1f642) + "  \n";
    const std::string visible = "\nB <tag>" + Utf8(0x4fdd) + Utf8(0x7559) + "</tag> <thi";
    const std::string text = "A <think>" + thought + "</think>" + visible;
    for (size_t chunk : {size_t(1), size_t(2), size_t(5), size_t(17), text.size()}) {
        const auto run = RunOutput(text, chunk);
        const auto stream = Inspect(run.frames, true);
        const auto value = Parse(run.completion.json);
        Expect(stream.content == "A " + visible && stream.reasoning == thought, "UTF-8/marker splitting changed visible whitespace or channels");
        Expect(value["choices"][0]["message"]["content"].asString() == stream.content &&
               value["choices"][0]["message"]["reasoning_content"].asString() == stream.reasoning,
               "Canonical stream and final message disagree");
        const auto initial = RunOutput(thought + "</think>" + visible, chunk, true, true, true);
        const auto initialView = Inspect(initial.frames, true);
        Expect(initialView.content == visible && initialView.reasoning == thought, "Prompt-open reasoning leaked into content");
    }
    const auto partial = RunOutput("literal <thi", 1);
    Expect(Inspect(partial.frames, true).content == "literal <thi", "Incomplete marker suffix was dropped at finish");
    const auto unfinished = RunOutput("prefix <think>\nunfinished ", 2);
    const auto view = Inspect(unfinished.frames, true);
    Expect(view.content == "prefix " && view.reasoning == "\nunfinished ", "Unclosed thinking did not preserve its complete text");
    const std::string scalars = Utf8(0x80) + Utf8(0x7ff) + Utf8(0x800) + Utf8(0xd7ff) + Utf8(0xe000) + Utf8(0x10000) + Utf8(0x10ffff);
    Expect(Inspect(RunOutput(scalars, 1).frames, true).content == scalars, "Valid UTF-8 boundary scalars were altered");
}

void ToolBlocksRemainText() {
    const std::vector<std::string> blocks = {
        "<tool_call><function=calc><parameter=x>1</parameter></function></tool_call>",
        "<tool_call>{\"name\":\"calc\",\"arguments\":{\"x\":1}}</tool_call>",
        "<tool_call>malformed</tool_call>", "<tool_call>unfinished",
        "<tool_call>{\"text\":\"<think>literal</think>\"}</tool_call>"
    };
    for (const auto& block : blocks) {
        const std::string text = "before " + block + " after";
        for (size_t chunk : {size_t(1), size_t(7), text.size()}) {
            const auto run = RunOutput(text, chunk);
            Expect(Inspect(run.frames, true).content == text, "Tool-like output was dropped, modified, or executed");
            const auto message = Parse(run.completion.json)["choices"][0]["message"];
            Expect(message["content"].asString() == text && !message.isMember("tool_calls"), "Tool output escaped the text-only response contract");
            Expect(!run.completion.warnings.empty() && run.completion.warnings.size() <= 4, "Tool recovery warnings are missing or unbounded");
            for (const auto& warning : run.completion.warnings) Expect(warning.size() <= 256, "Tool diagnostic exceeded its limit");
        }
    }
}

void OutputUtf8Errors() {
    const std::vector<std::string> invalid = {
        Bytes({0x80}), Bytes({0xc0, 0xaf}), Bytes({0xc2, 0x20}), Bytes({0xe0, 0x80, 0xaf}),
        Bytes({0xed, 0xa0, 0x80}), Bytes({0xf0, 0x80, 0x80, 0x80}), Bytes({0xf4, 0x90, 0x80, 0x80}),
        Bytes({0xf5, 0x80, 0x80, 0x80}), Bytes({0xff})
    };
    for (const auto& bad : invalid) {
        const auto prepared = OutputRequest();
        ChatOutput output(Identity(), prepared);
        auto frames = output.Begin();
        ExpectError([&] { output.PushText(bad); }, 502, "invalid_utf8_output");
        const auto completion = output.Finish(Summary(prepared, 5, wire::EndReason::Length));
        Append(frames, completion.sseFrames);
        CheckFailure(completion, frames, "invalid_utf8_output", 502);
    }
    for (const auto& suffix : {Bytes({0xc2}), Bytes({0xe4, 0xb8}), Bytes({0xf0, 0x9f, 0x99})}) {
        const auto prepared = OutputRequest();
        ChatOutput output(Identity(), prepared);
        auto frames = output.PushText("prefix " + suffix);
        const auto completion = output.Finish(Summary(prepared));
        Append(frames, completion.sseFrames);
        CheckFailure(completion, frames, "invalid_utf8_output", 502);
        Expect(Inspect(frames, false).content == "prefix ", "Incomplete UTF-8 leaked into a delta");
    }
    const std::string controls = std::string("A\0", 2) + "\"\\\n\tB";
    const auto run = RunOutput(controls, 1);
    Expect(Inspect(run.frames, true).content == controls, "Legal output controls were not JSON-escaped losslessly");
}

void LengthTruncatedUtf8() {
    const std::string replacement = Utf8(0xfffd);
    const std::vector<std::string> suffixes = {
        Bytes({0xe4}), Bytes({0xe4, 0xb8}), Bytes({0xf0}), Bytes({0xf0, 0x9f}), Bytes({0xf0, 0x9f, 0x99})
    };
    for (const auto& suffix : suffixes) {
        for (bool streaming : {false, true}) {
            for (bool reasoning : {false, true}) {
                for (size_t chunk : {size_t(1), suffix.size()}) {
                    auto prepared = OutputRequest(streaming, true, reasoning);
                    prepared.options.generation.maxNewTokens = 8;
                    ChatOutput output(Identity(), prepared);
                    auto frames = output.PushText("prefix ");
                    for (size_t offset = 0; offset < suffix.size(); offset += chunk) {
                        Append(frames, output.PushText(std::string_view(suffix).substr(offset, chunk)));
                    }
                    const auto completion = output.Finish(Summary(prepared, 8, wire::EndReason::Length));
                    Append(frames, completion.sseFrames);
                    Expect(!completion.failed, "A valid UTF-8 prefix at the token limit should not fail");
                    const auto value = Parse(completion.json);
                    const std::string expectedContent = reasoning ? "" : "prefix " + replacement;
                    const std::string expectedReasoning = reasoning ? "prefix " + replacement : "";
                    Expect(value["choices"][0]["finish_reason"].asString() == "length" &&
                           value["choices"][0]["message"]["content"].asString() == expectedContent &&
                           value["choices"][0]["message"]["reasoning_content"].asString() == expectedReasoning &&
                           value["usage"]["completion_tokens"].asUInt64() == 8,
                           "Length replacement changed its channel, finish reason, or actual token count");
                    if (streaming) {
                        const auto view = Inspect(frames, true);
                        Expect(view.content == expectedContent && view.reasoning == expectedReasoning &&
                               view.finishes == 1 && view.finishReason == "length" && view.done == 1 &&
                               view.usage["completion_tokens"].asUInt64() == 8,
                               "Length-truncated UTF-8 differs between SSE and final JSON");
                    } else Expect(frames.empty(), "Nonstream length replacement emitted SSE");
                    Expect(output.Finish(Summary(prepared, 8, wire::EndReason::Length)).sseFrames.empty(),
                           "Repeated finish emitted another replacement character");
                }
            }
        }
    }
    const auto prepared = OutputRequest();
    ChatOutput interrupted(Identity(), prepared);
    auto frames = interrupted.PushText("prefix " + suffixes.back());
    const auto completion = interrupted.Finish(Summary(prepared, 32, wire::EndReason::Length), true);
    Append(frames, completion.sseFrames);
    CheckFailure(completion, frames, "worker_connection_failed", 502);
    Expect(Inspect(frames, false).content == "prefix ", "Interrupted delivery must not become a successful replacement");
}

void LengthReplacementQuotas() {
    const auto prepared = OutputRequest();
    const auto summary = Summary(prepared, 32, wire::EndReason::Length);
    {
        // Hide most of the raw text in empty marker pairs so the final visible
        // output would fit; the additional replacement bytes alone exceed raw
        // capacity. Both modes must enforce that bound before serialization.
        std::string hidden;
        while (hidden.size() + 15 <= 2047) hidden += "<think></think>";
        hidden.append(2047 - hidden.size(), 'x');
        for (bool stream : {false, true}) {
            const auto request = OutputRequest(stream);
            ChatOutput output(Identity(), request, 4096);
            auto frames = output.PushText(hidden + Bytes({0xe4}));
            const auto completion = output.Finish(Summary(request, 32, wire::EndReason::Length));
            Append(frames, completion.sseFrames);
            Expect(completion.failed && Parse(completion.json)["error"]["code"].asString() == "output_limit_exceeded",
                   "UTF-8 replacement bypassed the raw output limit");
            if (stream) CheckFailure(completion, frames, "output_limit_exceeded", 502);
            Expect(FrameBytes(frames) <= 4096, "Length replacement overflow escaped the reserved error budget");
        }
    }
    {
        const auto prefixFrames = [](ChatOutput& output) {
            auto frames = output.Begin();
            for (char character : std::string("0123456789abcdef")) {
                Append(frames, output.PushText(std::string(1, character)));
            }
            return frames;
        };
        ChatOutput reference(Identity(), prepared);
        auto referenceFrames = prefixFrames(reference);
        const auto referenceCompletion = reference.Finish(summary);
        Expect(!referenceCompletion.failed, "SSE quota reference unexpectedly failed");
        Append(referenceFrames, referenceCompletion.sseFrames);
        const size_t exactBudget = FrameBytes(referenceFrames) + 2048;
        Expect(exactBudget >= 4096, "SSE quota fixture must exceed the minimum output budget");
        ChatOutput output(Identity(), prepared, exactBudget);
        auto frames = prefixFrames(output);
        Append(frames, output.PushText(Bytes({0xf0, 0x9f})));
        const auto completion = output.Finish(summary);
        Append(frames, completion.sseFrames);
        CheckFailure(completion, frames, "output_limit_exceeded", 502);
        Expect(FrameBytes(frames) <= exactBudget, "Replacement SSE frame was omitted from quota accounting");
    }
    {
        const auto request = OutputRequest(false);
        ChatOutput reference(Identity(), request);
        reference.PushText(std::string(2000, 'x'));
        const auto baseline = reference.Finish(summary);
        Expect(!baseline.failed, "JSON quota reference unexpectedly failed");
        const size_t exactBudget = baseline.json.size() + 2048;
        Expect(exactBudget >= 4096, "JSON quota fixture must exceed the minimum output budget");
        ChatOutput output(Identity(), request, exactBudget);
        output.PushText(std::string(2000, 'x') + Bytes({0xe4, 0xb8}));
        const auto completion = output.Finish(summary);
        Expect(completion.failed && completion.sseFrames.empty() && completion.json.size() <= exactBudget &&
               Parse(completion.json)["error"]["code"].asString() == "output_limit_exceeded",
               "Replacement UTF-8 bytes were omitted from JSON quota accounting");
    }
}

void RawAndSerializedQuotas() {
    const auto prepared = OutputRequest();
    {
        ChatOutput output(Identity(), prepared, 4096);
        auto frames = output.Begin();
        ExpectError([&] { output.PushText(std::string(3000, 'x')); }, 502, "output_limit_exceeded");
        const auto completion = output.Finish(Summary(prepared));
        Append(frames, completion.sseFrames);
        CheckFailure(completion, frames, "output_limit_exceeded", 502);
        Expect(FrameBytes(frames) <= 4096, "Raw overflow error escaped the reserved SSE budget");
    }
    {
        ChatOutput output(Identity(), prepared, 4096);
        auto frames = output.Begin();
        size_t pushes = 0;
        bool failed = false;
        for (; pushes < 100; ++pushes) {
            try { Append(frames, output.PushText("x")); }
            catch (const ApiError& error) { Expect(error.code == "output_limit_exceeded", "Unexpected quota failure"); failed = true; break; }
        }
        Expect(failed && pushes > 1, "Serialized SSE quota did not count envelope overhead");
        const auto completion = output.Finish(Summary(prepared));
        Append(frames, completion.sseFrames);
        CheckFailure(completion, frames, "output_limit_exceeded", 502);
        Expect(FrameBytes(frames) <= 4096, "SSE total bytes exceed the configured cap including failure and [DONE]");
    }
    {
        ChatOutput output(Identity(), OutputRequest(false), 4096);
        Expect(output.PushText(std::string(500, '\1')).empty(), "Nonstream quota test emitted SSE");
        const auto completion = output.Finish(Summary(prepared));
        Expect(completion.failed && completion.sseFrames.empty() && completion.json.size() <= 4096 &&
               Parse(completion.json)["error"]["code"].asString() == "output_limit_exceeded",
               "Nonstream quota ignored JSON escaping expansion");
    }
    {
        ChatOutput output(Identity(), prepared, 4096);
        auto frames = output.Begin();
        const ApiError huge(503, std::string(20000, '"'), Bytes({0xff}) + std::string(20000, '\1'), std::string(20000, '\\'));
        const auto completion = output.Fail(huge);
        Append(frames, completion.sseFrames);
        const auto error = Parse(completion.json)["error"];
        Expect(error["message"].asString().size() <= 512 && error["code"].asString().size() <= 64 &&
               error["param"].asString().size() <= 128 && FrameBytes(frames) <= 4096,
               "Failure diagnostics did not stay inside the sanitized error reserve");
        Expect(Inspect(frames, false).errors == 1, "Sanitized error is not a valid SSE JSON frame");
    }
}

void RemoteAndLocalFailures() {
    struct FailureCase { wire::RequestStatus status; wire::EndReason reason; std::optional<wire::ErrorCode> wireCode; const char* apiCode; int http; };
    const std::vector<FailureCase> cases = {
        {wire::RequestStatus::Cancelled, wire::EndReason::Cancelled, {}, "request_cancelled", 409},
        {wire::RequestStatus::Cancelled, wire::EndReason::OutputBackpressure, wire::ErrorCode::ResourceLimit, "output_backpressure", 503},
        {wire::RequestStatus::Cancelled, wire::EndReason::SchedulerStopped, wire::ErrorCode::SchedulerStopped, "server_stopping", 503},
        {wire::RequestStatus::Failed, wire::EndReason::RuntimeFault, wire::ErrorCode::RuntimeFault, "runtime_fault", 503},
        {wire::RequestStatus::Failed, wire::EndReason::Failed, wire::ErrorCode::RequestFailed, "generation_failed", 500}
    };
    const auto prepared = OutputRequest(true, true);
    for (const auto& failure : cases) {
        ChatOutput output(Identity(), prepared);
        auto frames = output.PushText("partial");
        auto summary = Summary(prepared);
        summary.status = failure.status; summary.endReason = failure.reason; summary.errorCode = failure.wireCode;
        const auto completion = output.Finish(summary);
        Append(frames, completion.sseFrames);
        CheckFailure(completion, frames, failure.apiCode, failure.http, true);
    }
    ChatOutput output(Identity(), prepared);
    auto frames = output.PushText("partial");
    const auto completion = output.Finish(Summary(prepared), true);
    Append(frames, completion.sseFrames);
    CheckFailure(completion, frames, "worker_connection_failed", 502, true);
}

void InvalidCompletionSummaries() {
    const auto prepared = OutputRequest();
    const std::vector<std::function<void(wire::Summary&)>> mutations = {
        [](auto& s) { ++s.promptTokensProcessed; }, [](auto& s) { --s.promptTokensProcessed; },
        [](auto& s) { s.generatedTokens = 33; s.eventCount = 34; }, [](auto& s) { ++s.eventCount; },
        [](auto& s) { s.endReason = wire::EndReason::None; }, [](auto& s) { s.errorCode = wire::ErrorCode::RuntimeFault; },
        [](auto& s) { s.message = "unexpected error on success"; }
    };
    for (const auto& mutate : mutations) {
        ChatOutput output(Identity(), prepared);
        auto frames = output.PushText("partial");
        auto summary = Summary(prepared); mutate(summary);
        const auto completion = output.Finish(summary);
        Append(frames, completion.sseFrames);
        CheckFailure(completion, frames, "invalid_completion_summary", 502);
    }
}

void EmptyLengthAndIdempotence() {
    auto prepared = OutputRequest(true, true);
    ChatOutput empty(Identity(), prepared);
    const auto zero = empty.Finish(Summary(prepared, 0));
    const auto zeroView = Inspect(zero.sseFrames, true);
    Expect(!zero.failed && zeroView.content.empty() && zeroView.roles == 1 && zeroView.done == 1 &&
           zeroView.usage["completion_tokens"].asUInt64() == 0, "Empty successful output did not complete correctly");
    Expect(empty.Finish(Summary(prepared, 0)).sseFrames.empty() && empty.Begin().empty(), "Repeated finish/begin duplicated SSE");
    const auto afterSuccess = empty.Fail(ApiError(500, "later", "later"));
    Expect(afterSuccess.json == zero.json && !afterSuccess.failed && afterSuccess.sseFrames.empty(), "Late failure rewrote a completed result");
    prepared.options.generation.maxNewTokens = 5;
    ChatOutput length(Identity(), prepared);
    auto frames = length.PushText("abc");
    const auto complete = length.Finish(Summary(prepared, 5, wire::EndReason::Length));
    Append(frames, complete.sseFrames);
    Expect(Inspect(frames, true).finishReason == "length", "Length termination was reported as a stop token");
    ExpectError([&] { length.PushText("late"); }, 500, "completion_finished");
    Expect(length.Finish(Summary(prepared, 5)).json == complete.json, "Repeated finish changed the cached result");
    ChatOutput failed(Identity(), prepared);
    const auto first = failed.Fail(ApiError(503, "unavailable", "worker missing"));
    const auto second = failed.Fail(ApiError(500, "different", "different"));
    Expect(first.failed && second.json == first.json && second.sseFrames.empty() &&
           failed.Finish(Summary(prepared)).sseFrames.empty(), "Repeated errors duplicated terminal SSE");
}

void MoveAndConstruction() {
    const auto prepared = OutputRequest();
    ChatOutput original(Identity(), prepared);
    auto frames = original.PushText("a");
    ChatOutput moved(std::move(original));
    Append(frames, moved.PushText("b"));
    const auto completion = moved.Finish(Summary(prepared));
    Append(frames, completion.sseFrames);
    Expect(Inspect(frames, false).content == "ab", "Moving the output adapter lost parser state");
    ExpectError([&] { ChatOutput output(Identity(), prepared, 4095); }, 500, "invalid_server_configuration");
    auto identity = Identity(); identity.id.clear();
    ExpectError([&] { ChatOutput output(identity, prepared); }, 500, "invalid_server_configuration");
    identity = Identity(); identity.model = Bytes({0x80});
    ExpectError([&] { ChatOutput output(identity, prepared); }, 500, "invalid_server_configuration");
    identity = Identity(); identity.createdUnixSeconds = -1;
    ExpectError([&] { ChatOutput output(identity, prepared); }, 500, "invalid_server_configuration");
    auto noPrompt = prepared; noPrompt.promptIds.clear();
    ExpectError([&] { ChatOutput output(Identity(), noPrompt); }, 500, "invalid_server_configuration");
}
} // namespace

int main() {
    try {
        const Fixture fixture;
        const std::vector<std::pair<const char*, std::function<void()>>> tests = {
            {"defaults and trusted template", [&] { DefaultsAndTemplate(fixture); }},
            {"supported parameters and uint64 seeds", [&] { SupportedParameters(fixture); }},
            {"message roles and text parts", [&] { MessagesAndContentParts(fixture); }},
            {"strict JSON and Unicode input", [&] { StrictJson(fixture); }},
            {"unsupported feature rejection", [&] { UnsupportedFields(fixture); }},
            {"message shape validation", [&] { InvalidMessages(fixture); }},
            {"sampling and option validation", [&] { InvalidParameters(fixture); }},
            {"prompt and service configuration limits", [&] { PromptAndServerLimits(fixture); }},
            {"Hybrid fallback, thinking, and checkpoint EOS IDs", [&] { HybridFallbackAndEos(fixture); }},
            {"Hybrid history and text parts", [&] { HybridHistoryAndTextParts(fixture); }},
            {"Hybrid trusted templates and rendered reasoning state", [&] { HybridTrustedTemplate(fixture); }},
            {"Hybrid prepared JSON and SSE parity", [&] { HybridPreparedOutputParity(fixture); }},
            {"Hybrid vision metadata and text-only boundary", [&] { HybridVisionTextBoundary(fixture); }},
            {"Hybrid limits and template failures", [&] { HybridLimitsAndTemplateFailures(fixture); }},
            {"nonstream and SSE schema", NonstreamAndSseSchema},
            {"fragmentation and reasoning channels", FragmentationAndReasoning},
            {"unsolicited tools retained as text", ToolBlocksRemainText},
            {"UTF-8 output validation", OutputUtf8Errors},
            {"UTF-8 length-boundary replacement", LengthTruncatedUtf8},
            {"UTF-8 replacement output quotas", LengthReplacementQuotas},
            {"raw and serialized output quotas", RawAndSerializedQuotas},
            {"remote and local failures", RemoteAndLocalFailures},
            {"completion summary validation", InvalidCompletionSummaries},
            {"empty output, length, and idempotence", EmptyLengthAndIdempotence},
            {"move and construction invariants", MoveAndConstruction}
        };
        size_t failed = 0;
        for (const auto& [name, test] : tests) {
            try { test(); std::cout << "[PASS] " << name << '\n'; }
            catch (const std::exception& error) { ++failed; std::cerr << "[FAIL] " << name << ": " << error.what() << '\n'; }
        }
        std::cout << "Chat codec: " << tests.size() - failed << '/' << tests.size() << " groups passed\n";
        return failed == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Chat codec fixture failed: " << error.what() << '\n';
        return 1;
    }
}
