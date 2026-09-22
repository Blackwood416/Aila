#include "proxy/LegacyTextSession.hpp"
#include "chat/ChatJson.hpp"

#include <windows.h>
#include <simdjson.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

namespace {
using namespace aila::proxy;
namespace chat = aila::chat;
namespace fs = std::filesystem;
void Expect(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}
template<class F> void Error(int code, F action) {
    try { action(); }
    catch (const LegacyTextError& error) {
        Expect(error.Code() == code, "Unexpected legacy error: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("Expected legacy error");
}
void Write(const fs::path& path, const std::string& text) {
    std::ofstream output(path, std::ios::binary);
    output << text;
    Expect(output.good(), "Could not write tokenizer fixture");
}
std::string Utf8(unsigned point) {
    if (point < 128) return {static_cast<char>(point)};
    return {static_cast<char>(0xc0 | (point >> 6)), static_cast<char>(0x80 | (point & 63))};
}
std::string Template() {
    return "{%- for message in messages -%}"
           "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' -}}"
           "{%- endfor -%}{%- if add_generation_prompt -%}"
           "{{- '<|im_start|>assistant\\n' -}}"
           "{%- if enable_thinking -%}{{- '<think>\\n' -}}{%- endif -%}{%- endif -%}";
}
class Directory {
public:
    Directory() : parent(fs::canonical(fs::temp_directory_path())) {
        path = parent / ("aila-legacy-text-" + std::to_string(GetCurrentProcessId()) + "-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        Expect(fs::create_directory(path), "Could not create owned temporary fixture");
        path = fs::canonical(path);
        std::string vocab = "{";
        unsigned extra = 256;
        for (unsigned byte = 0; byte < 256; ++byte) {
            if (byte) vocab += ',';
            const bool visible = (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) || byte >= 174;
            vocab += chat::json_escape_string(Utf8(visible ? byte : extra++)) + ':' + std::to_string(byte);
        }
        vocab += '}';
        Write(path / "vocab.json", vocab);
        Write(path / "merges.txt", "#version: 0.2\n");
        Write(path / "tokenizer.json", R"({"added_tokens":[
            {"content":"<|im_start|>","id":256,"special":true},
            {"content":"<|im_end|>","id":257,"special":true},
            {"content":"<|endoftext|>","id":258,"special":true},
            {"content":"<think>","id":259,"special":true},
            {"content":"</think>","id":260,"special":true}]})");
        Write(path / "tokenizer_config.json", "{\"chat_template\":" + chat::json_escape_string(Template()) +
            ",\"eos_token\":\"<|im_end|>\",\"added_tokens_decoder\":{}}");
    }
    ~Directory() {
        std::error_code error;
        if (path.is_absolute() && path.parent_path() == parent &&
            path.filename().string().rfind("aila-legacy-text-", 0) == 0) fs::remove_all(path, error);
    }
    fs::path path;
private:
    fs::path parent;
};
class Environment {
public:
    Environment(const wchar_t* name, const wchar_t* value) : name(name) {
        const auto size = GetEnvironmentVariableW(name, nullptr, 0);
        if (size) {
            before.resize(size);
            before.resize(GetEnvironmentVariableW(name, before.data(), size));
            present = true;
        }
        Expect(SetEnvironmentVariableW(name, value), "Could not configure test environment");
    }
    ~Environment() { SetEnvironmentVariableW(name.c_str(), present ? before.c_str() : nullptr); }
private:
    std::wstring name, before;
    bool present = false;
};
struct Record {
    std::vector<int> prompt, emitted, closes, stops;
    AilaInferenceRequestOptions options{};
    AilaInferenceRequestPolicy policy{};
};
class FakeSession;
class Stream final : public LegacyTokenStream {
public:
    Stream(FakeSession& state, std::vector<int> tokens, bool length) : state(state), tokens(std::move(tokens)), length(length) {}
    AilaInferenceEvent Read() override;
    void Cancel() override;
private:
    FakeSession& state;
    std::vector<int> tokens;
    size_t first = 0;
    bool terminal = false, length;
};
class FakeSession final : public LegacyTokenSession {
public:
    uint32_t capacity = 4096, maxPrompt = 32, maxGenerated = 512, maxChunk = 16;
    uint64_t maxOutstanding = 32768;
    // -1: exact truncation; 0: no Hybrid checkpoint; positive: coarse checkpoint.
    int checkpoint = -1;
    int resets = 0, cancels = 0, prefills = 0;
    std::vector<int> context, next;
    std::vector<Record> submitted;
    std::vector<std::pair<uint32_t, uint32_t>> truncations;
    uint32_t Capacity() const override { return capacity; }
    uint32_t MaxPromptTokens() const override { return maxPrompt; }
    uint32_t MaxGeneratedTokens() const override { return maxGenerated; }
    uint64_t MaxOutstandingTokens() const override { return maxOutstanding; }
    uint32_t MaxPrefillChunk() const override { return maxChunk; }
    uint32_t ContextLength() override { return static_cast<uint32_t>(context.size()); }
    void Reset() override { context.clear(); ++resets; }
    uint32_t Truncate(uint32_t target) override {
        Expect(target <= context.size(), "Truncate past actual context");
        const auto actual = target == context.size() || checkpoint < 0 ? target :
            checkpoint == 0 ? 0 : target / static_cast<uint32_t>(checkpoint) * static_cast<uint32_t>(checkpoint);
        context.resize(actual);
        truncations.emplace_back(target, actual);
        return actual;
    }
    void Prefill(const std::vector<int>& tokens, uint32_t chunk) override {
        Expect(!tokens.empty() && tokens.size() <= maxPrompt && chunk && chunk <= maxChunk, "Invalid replay chunk");
        Expect(context.size() + tokens.size() <= capacity, "Replay overflow");
        context.insert(context.end(), tokens.begin(), tokens.end());
        ++prefills;
    }
    std::unique_ptr<LegacyTokenStream> Submit(const std::vector<int>& tokens,
        const AilaInferenceRequestOptions& options, const AilaInferenceRequestPolicy& policy) override {
        Expect(!tokens.empty() && tokens.size() <= maxPrompt, "Invalid submit prefix");
        Expect(tokens.size() + static_cast<uint64_t>(options.max_new_tokens) <= maxOutstanding, "Submit token reservation overflow");
        Record record;
        record.options = options;
        record.policy = policy;
        if (options.thinking_close_token_count) record.closes.assign(options.thinking_close_token_ids,
            options.thinking_close_token_ids + options.thinking_close_token_count);
        if (options.stop_token_count) record.stops.assign(options.stop_token_ids,
            options.stop_token_ids + options.stop_token_count);
        context.insert(context.end(), tokens.begin(), tokens.end());
        record.prompt = context;
        for (int token : next) {
            if (record.emitted.size() == options.max_new_tokens ||
                std::find(record.stops.begin(), record.stops.end(), token) != record.stops.end()) break;
            record.emitted.push_back(token);
        }
        // Execute all tokens before the first callback. Cancellation cannot
        // undo this overrun: the adapter must restore the accepted prefix.
        context.insert(context.end(), record.emitted.begin(), record.emitted.end());
        Expect(context.size() <= capacity, "Submit overflow");
        submitted.push_back(record);
        return std::make_unique<Stream>(*this, record.emitted, record.emitted.size() == options.max_new_tokens);
    }
};
AilaInferenceEvent Stream::Read() {
    Expect(!terminal, "Read after terminal");
    AilaInferenceEvent event{};
    if (first < tokens.size()) {
        event.kind = AILA_INFERENCE_EVENT_TOKEN;
        event.token_id = tokens[first++];
    } else {
        terminal = true;
        event.kind = AILA_INFERENCE_EVENT_TERMINAL;
        event.progress.status = AILA_INFERENCE_COMPLETED;
        event.progress.end_reason = length ? AILA_INFERENCE_END_LENGTH : AILA_INFERENCE_END_STOP_TOKEN;
    }
    return event;
}
void Stream::Cancel() { ++state.cancels; }
struct Fixture {
    Directory directory;
    Tokenizer tokenizer;
    FakeSession* fake;
    std::unique_ptr<LegacyTextSession> adapter;
    explicit Fixture(bool hybrid = false, uint32_t capacity = 4096) {
        Expect(tokenizer.load(directory.path.u8string()), "Tokenizer fixture did not load");
        ModelSpec spec;
        spec.family = hybrid ? ModelFamily::Qwen35Hybrid : ModelFamily::Qwen3Dense;
        spec.qwen3.vocab_size = spec.qwen35_text.vocab_size = 261;
        auto value = std::make_unique<FakeSession>();
        fake = value.get();
        fake->capacity = capacity;
        Tokenizer copy;
        Expect(copy.load(directory.path.u8string()), "Tokenizer copy did not load");
        adapter = std::make_unique<LegacyTextSession>(spec, std::move(copy), std::move(value), 7);
        Output("OK");
    }
    void Output(const std::string& text) { fake->next = tokenizer.encode(text); }
    std::string Prompt() { return tokenizer.decode(fake->submitted.back().prompt); }
    GenerationConfig Config(int maximum = 64) {
        GenerationConfig config;
        config.max_new_tokens = maximum;
        config.do_sample = false;
        return config;
    }
    std::string Json(const std::string& text, const std::string& extra = "") {
        return "{\"messages\":[{\"role\":\"user\",\"content\":" + chat::json_escape_string(text) + "}]" + extra + "}";
    }
};
std::string StringField(const std::string& text, const char* field) {
    simdjson::dom::parser parser;
    return std::string(parser.parse(text)[field].get_string().value());
}
void HistoryAndReset() {
    Fixture f;
    f.Output("<think>private-work</think>answer");
    f.adapter->Generate("generate", "first-question", f.Config());
    f.Output("second-answer");
    f.adapter->Generate("generate", "second-question", f.Config());
    const auto prompt = f.Prompt();
    Expect(prompt.find("private-work") == std::string::npos, "Reasoning leaked into a later history prompt");
    Expect(prompt.find("answer") != std::string::npos && prompt.find("first-question") != std::string::npos &&
        prompt.find("second-question") != std::string::npos, "Conversation history was lost");
    f.adapter->Reset();
    Expect(f.adapter->ContextLength() == 0, "Reset left context");
    f.adapter->Generate("generate", "fresh-question", f.Config());
    Expect(f.Prompt().find("first-question") == std::string::npos, "Reset left history");
}
void AcceptedPrefixCancellation() {
    for (int checkpoint : {-1, 0, 16}) {
        Fixture f;
        f.fake->checkpoint = checkpoint;
        f.Output("abcdefghijklmnopqrst");
        int calls = 0;
        const auto result = f.adapter->Generate("generate.messages_stream", f.Json("cancel"), f.Config(),
            [&](const std::string&) { return ++calls < 3; });
        auto expected = f.fake->submitted.back().prompt;
        expected.insert(expected.end(), {'a', 'b', 'c'});
        Expect(result.aborted && calls == 3 && result.text == "abc", "Callback abort semantics changed");
        Expect(f.fake->context == expected && f.fake->cancels == 1, "Cancellation did not restore accepted tokens");
        if (checkpoint >= 0) Expect(f.fake->prefills > 0, "Hybrid checkpoint replay was skipped");
        f.Output("next");
        Expect(f.adapter->Generate("generate.messages", f.Json("after cancel"), f.Config()).text == "next",
            "Session could not be used after cancellation");
    }
}
void CallbackExceptionAndReuse() {
    Fixture f;
    f.fake->checkpoint = 0;
    f.Output("abcdefghijk");
    int calls = 0;
    try {
        f.adapter->Generate("generate.messages_stream", f.Json("throw"), f.Config(), [&](const std::string&) {
            if (++calls == 3) throw std::runtime_error("caller exception");
            return true;
        });
        Expect(false, "Callback exception was swallowed");
    } catch (const std::runtime_error& error) {
        Expect(std::string(error.what()) == "caller exception", "Original callback failure was replaced");
    }
    auto expected = f.fake->submitted.back().prompt;
    expected.insert(expected.end(), {'a', 'b', 'c'});
    Expect(f.fake->context == expected, "Exception cleanup lost accepted prefix");
    f.Output("ok");
    Expect(f.adapter->Generate("generate.messages", f.Json("reuse"), f.Config()).text == "ok", "Failed to reuse Session");
}
void LoopGuardAndStops() {
    for (bool stream : {false, true}) {
        Fixture f;
        f.Output(std::string(64, 'x'));
        int calls = 0;
        auto callback = stream ? LegacyTextSession::TokenCallback([&](const std::string&) { ++calls; return true; }) :
            LegacyTextSession::TokenCallback{};
        const auto result = f.adapter->Generate("generate.chat", f.Json("loop"), f.Config(64), callback);
        Expect(StringField(result.text, "finish_reason") == "loop_guard", "Missing loop guard finish reason");
        Expect(StringField(result.text, "content") == std::string(stream ? 47 : 48, 'x'), "Legacy loop output changed");
        Expect(calls == (stream ? 47 : 0), "Loop guard callback count changed");
        Expect(f.fake->context.size() == f.fake->submitted.back().prompt.size() + 48, "Loop guard context changed");
    }
    Fixture f;
    f.Output("A<|im_end|>B");
    Expect(f.adapter->Generate("generate.messages", f.Json("stop"), f.Config()).text == "A", "EOS appeared in text");
    Expect(f.fake->submitted.back().stops == std::vector<int>({257, 258}), "Tokenizer EOS set was lost");
}
void HybridAnchorAndThinkingPolicy() {
    Fixture f(true);
    auto config = f.Config();
    config.do_sample = true;
    config.thinking_budget_tokens = 1;
    f.Output("x</think>\n\n<think>yz</think>ok");
    const auto request = f.Json("reason", ",\"chat_template_kwargs\":{\"enable_thinking\":true,\"chat_template\":" +
        chat::json_escape_string(Template()) + "}");
    const auto result = f.adapter->Generate("generate.chat", request, config);
    const auto& first = f.fake->submitted.back();
    Expect(first.options.prompt_opens_think && first.options.thinking_budget_tokens == 1, "Missing prompt thinking state");
    Expect(first.policy.thinking_policy == AILA_INFERENCE_THINKING_CLOSE_ONCE &&
        first.policy.sampling_policy == AILA_INFERENCE_SAMPLING_HOST &&
        first.policy.prefill_logits_policy == AILA_INFERENCE_PREFILL_LOGITS_LEGACY,
        "Legacy reasoning/sampling/logit-row policy changed");
    Expect(first.closes == std::vector<int>({260, 10, 10}), "Legacy thinking suffix changed");
    Expect(first.options.repetition_penalty > 1.1 && first.options.top_k >= 40, "Hybrid sampling tuning was lost");
    Expect(f.fake->prefills > 0, "Recurrent stable prefix was not prefetched");
    Expect(result.text.find("\"reasoning_budget_forced_close\":true") != std::string::npos, "Forced close metadata lost");
    config.max_new_tokens = 2;
    f.Output("xy");
    const auto truncated = f.adapter->Generate("generate.chat", request, config);
    Expect(truncated.text.find("\"reasoning_budget_truncated\":true") != std::string::npos, "Missing truncated close metadata");
    config.max_new_tokens = 64;
    config.thinking_budget_tokens = 0;
    f.Output("<think>unbounded");
    f.adapter->Generate("generate.messages", request, config);
    Expect(f.fake->submitted.back().options.thinking_budget_tokens == -1, "Budget zero changed legacy generation");
}
void HybridPrefixAbort() {
    Fixture f(true);
    f.fake->checkpoint = 0;
    f.Output("overrun");
    std::vector<std::string> calls;
    const auto result = f.adapter->Generate("generate.messages_stream",
        f.Json("reason", ",\"chat_template_kwargs\":{\"enable_thinking\":true,\"chat_template\":" +
            chat::json_escape_string(Template()) + "}"), f.Config(),
        [&](const std::string& piece) { calls.push_back(piece); return false; });
    auto expected = f.fake->submitted.back().prompt;
    expected.push_back('o');
    Expect(calls == std::vector<std::string>({"<think>\n"}) && result.aborted, "Hybrid prefix callback was lost");
    Expect(f.fake->context == expected, "Prefix callback abort committed unaccepted overrun");
}
void TemplateEnvironmentAndMetadata() {
    Fixture f;
    const auto file = f.directory.path / "override.jinja";
    Write(file, "FILE");
    Environment path(L"AILA_CHAT_TEMPLATE_PATH", file.c_str());
    {
        Environment text(L"AILA_CHAT_TEMPLATE", L"ENV");
        auto result = f.adapter->Generate("generate.chat", f.Json("ignored"), f.Config());
        Expect(f.Prompt() == "ENV" && result.text.find("\"template_name\":\"AILA_CHAT_TEMPLATE\"") != std::string::npos,
            "Environment template precedence or metadata changed");
        f.adapter->Generate("generate.messages", f.Json("ignored",
            ",\"chat_template_kwargs\":{\"chat_template\":\"REQUEST\"}"), f.Config());
        Expect(f.Prompt() == "REQUEST", "Request template did not override the environment");
    }
    {
        Environment text(L"AILA_CHAT_TEMPLATE", nullptr);
        const auto result = f.adapter->Generate("generate.chat", f.Json("ignored"), f.Config());
        Expect(f.Prompt() == "FILE" && result.text.find(chat::json_escape_string(file.u8string())) != std::string::npos,
            "Environment file template or metadata changed");
    }
    Environment noPath(L"AILA_CHAT_TEMPLATE_PATH", nullptr), noText(L"AILA_CHAT_TEMPLATE", nullptr);
    f.adapter->Generate("generate.messages", f.Json("tokenizer-again"), f.Config());
    Expect(f.Prompt().find("tokenizer-again") != std::string::npos, "Removing template overrides left stale CRT values");
}
void PrefillOnlyAndValidation() {
    Fixture f;
    f.fake->maxPrompt = 11;
    const auto result = f.adapter->Generate("generate.chat", f.Json(std::string(130, 'p')), f.Config(0));
    Expect(StringField(result.text, "content").empty() && StringField(result.text, "finish_reason") == "length",
        "Zero generation must prefill without output");
    Expect(f.fake->submitted.empty() && f.fake->prefills > 10 && f.adapter->ContextLength() > 130,
        "Zero generation sampled or skipped bounded prefix replay");
    const auto before = f.fake->context;
    auto config = f.Config();
    config.temperature = std::numeric_limits<float>::quiet_NaN();
    Error(AILA_ERR_INVALID_ARGUMENT, [&] { f.adapter->Generate("generate.messages", f.Json("bad"), config); });
    Error(AILA_ERR_RUNTIME, [&] { f.adapter->Generate("generate.messages", f.Json("large"), f.Config(600)); });
    Expect(f.fake->context == before, "Invalid request changed context");
    AilaGenConfigV2 shortConfig{};
    shortConfig.struct_size = sizeof(uint32_t);
    shortConfig.max_new_tokens = -999;
    Expect(LegacyTextSession::Config(&shortConfig).max_new_tokens == GenerationConfig{}.max_new_tokens,
        "Short V2 config read beyond struct_size");
    shortConfig.struct_size = offsetof(AilaGenConfigV2, use_fixed_seed);
    shortConfig.max_new_tokens = 7;
    shortConfig.thinking_budget_tokens = -9;
    Expect(LegacyTextSession::Config(&shortConfig).max_new_tokens == 7 &&
        LegacyTextSession::Config(&shortConfig).thinking_budget_tokens == -1, "Versioned config conversion changed");
    shortConfig.struct_size = 0;
    Error(AILA_ERR_INVALID_ARGUMENT, [&] { LegacyTextSession::Config(&shortConfig); });
}
void GenerationReservationBoundary() {
    Fixture f(false, 70000);
    f.fake->maxGenerated = 32768;
    f.adapter->Generate("generate.messages", f.Json("baseline"), f.Config(6));
    const auto before = f.fake->context;
    const auto truncated = f.fake->truncations.size();
    const auto prompt = f.Json("ignored", ",\"chat_template_kwargs\":{\"chat_template\":\"p\"}");
    for (const int maximum : {32768, 32769, (std::numeric_limits<int>::max)()}) {
        Error(AILA_ERR_RUNTIME, [&] { f.adapter->Generate("generate.messages", prompt, f.Config(maximum)); });
        Expect(f.fake->context == before && f.fake->truncations.size() == truncated,
            "Impossible generation reservation mutated the previous context");
    }
    Expect(f.adapter->Generate("generate.messages", prompt, f.Config(32767)).text == "OK",
        "A generation reservation exactly at the total token ceiling was rejected");
    Expect(f.fake->submitted.back().options.max_new_tokens == 32767 &&
        f.fake->submitted.back().prompt == std::vector<int>({'p'}), "Exact boundary changed the request");
    Fixture small(false, 16);
    Expect(small.adapter->Generate("generate.messages", prompt, small.Config((std::numeric_limits<int>::max)())).text == "OK" &&
        small.fake->submitted.back().options.max_new_tokens == 15,
        "Legacy max_new_tokens must still clamp to the remaining context");
}
void StructuredToolsAndLocalErrors() {
    Fixture f;
    f.Output("<think>why</think>hello<tool_call><function=lookup><parameter=q>x</parameter></function></tool_call>");
    std::vector<chat::StructuredStreamEvent> events;
    const auto result = f.adapter->Generate("generate.chat_stream", f.Json("tools",
        ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\"}}}]"),
        f.Config(200), {}, [&](const chat::StructuredStreamEvent& event) { events.push_back(event); return true; });
    Expect(!events.empty() && events.back().type == chat::StructuredStreamEventType::Final, "Missing structured final");
    Expect(events.back().finish_reason == "tool_calls" && events.back().tool_calls.size() == 1 &&
        events.back().tool_calls[0].function.name == "lookup", "Structured tool result changed");
    Expect(StringField(result.text, "reasoning_content") == "why" && StringField(result.text, "content") == "hello",
        "Structured reasoning/content split changed");
    const auto before = f.fake->context;
    Error(AILA_ERR_JSON_PARSE, [&] { f.adapter->Generate("generate.chat", "{", f.Config()); });
    Error(AILA_ERR_VISION_NOT_ENABLED, [&] { f.adapter->Generate("generate.chat",
        R"({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"x.png"}}]}]})", f.Config()); });
    Expect(f.fake->context == before, "Local JSON/media failure changed context");
    f.Output("still-usable");
    Expect(f.adapter->Generate("generate.messages", f.Json("text"), f.Config()).text == "still-usable",
        "Local failure broke the Session");
}
void HistoryOverflowDropsWholeTurn() {
    Fixture f(false, 210);
    f.Output("answer");
    f.adapter->Generate("generate", std::string(70, 'a'), f.Config(10));
    f.adapter->Generate("generate", std::string(90, 'b'), f.Config(10));
    Expect(f.Prompt().find(std::string(70, 'a')) == std::string::npos &&
        f.Prompt().find(std::string(90, 'b')) != std::string::npos, "Overflow did not drop oldest whole turn");
    Error(AILA_ERR_CONTEXT_OVERFLOW, [&] { f.adapter->Generate("generate", std::string(400, 'z'), f.Config(10)); });
    f.adapter->Generate("generate", "small", f.Config(10));
    Expect(f.Prompt().find("zzzz") == std::string::npos, "Failed user message stayed in history");
}
void DeliverEventLayout() {
    chat::StructuredStreamEvent final;
    final.type = chat::StructuredStreamEventType::Final;
    final.finish_reason = "length";
    final.warnings = {"quote \" and newline\n"};
    int calls = 0;
    const auto callback = [](const AilaChatStreamEvent* event, void* value) -> int {
        ++*static_cast<int*>(value);
        Expect(event->struct_size == sizeof(*event) && event->type == AILA_CHAT_STREAM_FINAL &&
            std::string(event->finish_reason) == "length" && !event->text, "Public event layout or nullability changed");
        simdjson::dom::parser parser;
        Expect(parser.parse(std::string(event->warnings_json)).get_array().value().size() == 1, "Warnings are invalid JSON");
        return 7;
    };
    Expect(LegacyTextSession::Deliver(final, callback, &calls) == 7 && calls == 1, "Callback return value changed");
}
}
int main() {
    Environment noText(L"AILA_CHAT_TEMPLATE", nullptr), noPath(L"AILA_CHAT_TEMPLATE_PATH", nullptr);
    const std::pair<const char*, void(*)()> tests[] = {
        {"history and reset", HistoryAndReset},
        {"accepted prefix after overrun", AcceptedPrefixCancellation},
        {"callback exception and reuse", CallbackExceptionAndReuse},
        {"loop guard and EOS", LoopGuardAndStops},
        {"Hybrid anchor and legacy thinking policy", HybridAnchorAndThinkingPolicy},
        {"Hybrid synthetic prefix abort", HybridPrefixAbort},
        {"template environment and metadata", TemplateEnvironmentAndMetadata},
        {"zero generation, bounded prefill, invalid configs", PrefillOnlyAndValidation},
        {"generation reservation and context clamp", GenerationReservationBoundary},
        {"structured tools and local error recovery", StructuredToolsAndLocalErrors},
        {"history overflow", HistoryOverflowDropsWholeTurn},
        {"public structured event layout", DeliverEventLayout},
    };
    int failures = 0;
    for (const auto& test : tests) {
        try { test.second(); std::cout << "PASS " << test.first << '\n'; }
        catch (const std::exception& error) { ++failures; std::cerr << "FAIL " << test.first << ": " << error.what() << '\n'; }
    }
    std::cout << std::size(tests) - failures << "/" << std::size(tests) << " legacy text groups passed\n";
    return failures ? 1 : 0;
}
