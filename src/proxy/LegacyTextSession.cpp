#include "proxy/LegacyTextSession.hpp"

#include "chat/AssistantOutputParser.hpp"
#include "chat/ChatJson.hpp"
#include "chat/ChatStreamJson.hpp"
#include "chat/ThinkingBudgetController.hpp"
#include "chat/ToolPolicy.hpp"
#include "profile/Profiling.hpp"
#include "utils/ModelSpec.hpp"

#include <windows.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <optional>

namespace aila::proxy {
namespace {
namespace chat = aila::chat;
thread_local logging::Source textLogSource;
class LogScope {
public:
    explicit LogScope(logging::Source value) : previous(std::move(textLogSource)) {
        static std::once_flag installed;
        std::call_once(installed, [] {
            aila::set_log_callback([](int level, const char* message, void*) {
                logging::enqueue(textLogSource, level, message ? message : "");
            });
        });
        textLogSource = std::move(value);
    }
    ~LogScope() { textLogSource = std::move(previous); }
private:
    logging::Source previous;
};

std::optional<std::string> Environment(const wchar_t* name) {
    DWORD length = GetEnvironmentVariableW(name, nullptr, 0);
    if (!length) return std::nullopt;
    std::wstring value(length, L'\0');
    const auto copied = GetEnvironmentVariableW(name, value.data(), length);
    if (copied >= length) throw LegacyTextError(AILA_ERR_RUNTIME, "Environment changed during text adapter configuration");
    value.resize(copied);
    const int count = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.data(),
        static_cast<int>(value.size()), nullptr, 0, nullptr, nullptr);
    std::string utf8(static_cast<size_t>(count), '\0');
    if (count) WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.data(),
        static_cast<int>(value.size()), utf8.data(), count, nullptr, nullptr);
    return utf8;
}
bool Flag(const wchar_t* name, bool fallback = false) {
    const auto value = Environment(name);
    return value ? std::atoi(value->c_str()) != 0 : fallback;
}
bool EndsWith(const std::string& value, const char* suffix) {
    const std::string_view tail(suffix);
    return value.size() >= tail.size() && value.compare(value.size() - tail.size(), tail.size(), tail) == 0;
}
void ValidateConfig(const GenerationConfig& config) {
    if (!std::isfinite(config.temperature) || !std::isfinite(config.top_p) ||
        !std::isfinite(config.repetition_penalty) || !std::isfinite(config.presence_penalty) ||
        !std::isfinite(config.frequency_penalty) || config.max_new_tokens < 0 ||
        (config.do_sample && config.repetition_penalty <= 0))
        throw LegacyTextError(AILA_ERR_INVALID_ARGUMENT, "Generation configuration contains invalid numeric values");
}
GenerationConfig Tune(GenerationConfig config, bool hybrid) {
    if (hybrid && config.do_sample) {
        if (config.repetition_penalty <= 1.0f) config.repetition_penalty = 1.12f;
        if (config.presence_penalty == 0.0f) config.presence_penalty = 0.05f;
        if (config.frequency_penalty == 0.0f) config.frequency_penalty = 0.10f;
        if (config.top_k < 30) config.top_k = 40;
        if (config.temperature < 0.75f) config.temperature = 0.80f;
    }
    return config;
}
const char* Choice(chat::ToolChoice choice) {
    switch (choice) {
    case chat::ToolChoice::None: return "none";
    case chat::ToolChoice::Required: return "required";
    case chat::ToolChoice::Function: return "function:";
    default: return "auto";
    }
}
} // namespace

std::unique_ptr<LegacyTextSession> LegacyTextSession::TryCreate(
    const std::filesystem::path& runtimeDirectory, const std::filesystem::path& apiDirectory,
    const std::string& modelDirectory,
    int capacity, logging::Source source) {
    LogScope logs(source);
    if (!Flag(L"AILA_LEGACY_TEXT_V2")) return {};
    ModelSpec spec;
    std::string error;
    if (!aila::modelspec::load_from_dir(modelDirectory, spec, &error)) return {};
    const bool hybrid = spec.family == ModelFamily::Qwen35Hybrid;
    if ((!hybrid && spec.family != ModelFamily::Qwen3Dense) || spec.vision.enabled || spec.audio_token_id >= 0)
        return {};
    const bool nf4 = spec.is_bitsandbytes_4bit() && spec.quantization.bnb_4bit_quant_type == "nf4";
    if ((!nf4 && (!hybrid || spec.is_quantized())) ||
        (hybrid && !Flag(L"AILA_Q35_LINEAR_DELTA", true))) return {};
    const auto scopedKv = Environment(L"AILA_VLM_KV_QUANT");
    if (hybrid && scopedKv ? std::atoi(scopedKv->c_str()) != 0 : Flag(L"AILA_KV_QUANT")) return {};
    Tokenizer tokenizer;
    if (!tokenizer.load(modelDirectory))
        throw LegacyTextError(AILA_ERR_RUNTIME, "Tokenizer could not be loaded from the model directory");
    uint32_t chunk = 512;
    if (hybrid) {
        const auto configured = Environment(L"AILA_Q35_PREFILL_CHUNK");
        if (configured && std::atoi(configured->c_str()) > 0) chunk = static_cast<uint32_t>(std::atoi(configured->c_str()));
        if (Flag(L"AILA_Q35_PREFILL_TOKENWISE")) chunk = 1;
    }
    const uint32_t backend = !hybrid ? AILA_INFERENCE_QWEN3_NF4
        : nf4 ? AILA_INFERENCE_QWEN35_NF4 : AILA_INFERENCE_QWEN35_BF16;
    auto session = CreateLegacyTokenSession(runtimeDirectory, apiDirectory, modelDirectory, backend, static_cast<uint32_t>(capacity), source);
    logging::enqueue(source, 1, "[Inference] Legacy text ABI uses shared Model/Session/Request over IPC v2");
    return std::make_unique<LegacyTextSession>(std::move(spec), std::move(tokenizer), std::move(session), chunk, std::move(source));
}

LegacyTextSession::LegacyTextSession(ModelSpec value, Tokenizer tokenizer,
    std::unique_ptr<LegacyTokenSession> session, uint32_t chunk, logging::Source source)
    : spec(std::move(value)), tokenizer(std::move(tokenizer)), session(std::move(session)),
      prefillChunk(std::max(1u, std::min(chunk, this->session->MaxPrefillChunk()))), source(std::move(source)),
      systemPrompt(spec.family == ModelFamily::Qwen35Hybrid && !spec.vision.enabled
          ? "You are a helpful text assistant. Reply in the user's language when possible. Do not assume images or videos unless explicitly provided."
          : "You are a helpful assistant.") {}

GenerationConfig LegacyTextSession::Config(const AilaGenConfig* input) {
    GenerationConfig result;
    if (!input) return result;
#define COPY(field) result.field = input->field
    COPY(max_new_tokens); COPY(temperature); COPY(top_k); COPY(top_p);
    COPY(repetition_penalty); COPY(presence_penalty); COPY(frequency_penalty);
    COPY(decode_chunk_size); COPY(stream_chunk_size);
#undef COPY
    result.do_sample = input->do_sample != 0;
    return result;
}
GenerationConfig LegacyTextSession::Config(const AilaGenConfigV2* input) {
    GenerationConfig result;
    if (!input) return result;
    if (!input->struct_size) throw LegacyTextError(AILA_ERR_INVALID_ARGUMENT, "Generation V2 struct_size must be positive");
#define COPY(field) if (input->struct_size >= offsetof(AilaGenConfigV2, field) + sizeof(input->field)) result.field = input->field
    COPY(max_new_tokens); COPY(temperature); COPY(top_k); COPY(top_p);
    COPY(repetition_penalty); COPY(presence_penalty); COPY(frequency_penalty);
    COPY(decode_chunk_size); COPY(stream_chunk_size); COPY(thinking_budget_tokens); COPY(sampling_seed);
#undef COPY
    if (input->struct_size >= offsetof(AilaGenConfigV2, do_sample) + sizeof(input->do_sample)) result.do_sample = input->do_sample != 0;
    if (input->struct_size >= offsetof(AilaGenConfigV2, use_fixed_seed) + sizeof(input->use_fixed_seed)) result.use_fixed_seed = input->use_fixed_seed != 0;
    result.thinking_budget_tokens = std::max(-1, result.thinking_budget_tokens);
    return result;
}

chat::ChatFormatInput LegacyTextSession::FormatInput(const chat::ChatRequest& request) const {
    chat::ChatFormatInput input;
    input.request = &request;
    input.family = spec.family;
    input.is_qwen35_0p8b = spec.family == ModelFamily::Qwen35Hybrid && is_exact_qwen35_hybrid_0p8b_spec(spec.qwen35_text);
    input.tokenizer_chat_template = tokenizer.chat_template();
    input.bos_token = tokenizer.bos_token();
    input.eos_token = tokenizer.eos_token();
    input.environment_template_text = Environment(L"AILA_CHAT_TEMPLATE").value_or("");
    input.environment_template_path = Environment(L"AILA_CHAT_TEMPLATE_PATH").value_or("");
    return input;
}

void LegacyTextSession::Reset() {
    session->Reset();
    cachedIds.clear();
    history.clear();
    logging::enqueue(source, 1, "[Context] Conversation reset");
}
int LegacyTextSession::ContextLength() { return static_cast<int>(session->ContextLength()); }

void LegacyTextSession::Replay(const std::vector<int>& sequence, size_t first, size_t end) {
    while (first < end) {
        const size_t next = std::min(end, first + session->MaxPromptTokens());
        session->Prefill({sequence.begin() + first, sequence.begin() + next}, prefillChunk);
        first = next;
    }
}
void LegacyTextSession::Align(const std::vector<int>& prompt, const std::vector<int>& generated) {
    const size_t accepted = prompt.size() + generated.size();
    uint32_t actual = session->ContextLength();
    if (actual == accepted) return;
    if (actual > prompt.size()) actual = session->Truncate(static_cast<uint32_t>(prompt.size()));
    if (actual < prompt.size()) Replay(prompt, actual, prompt.size());
    actual = session->ContextLength();
    Replay(generated, actual - prompt.size(), generated.size());
    if (session->ContextLength() != accepted)
        throw LegacyTextError(AILA_ERR_RUNTIME, "Could not restore the accepted generation prefix");
}

LegacyTextSession::Generated LegacyTextSession::GenerateChat(const chat::ChatRequest& request, const TokenCallback& callback) {
    LogScope logs(source);
    const bool hybrid = spec.family == ModelFamily::Qwen35Hybrid;
    chat::ChatRequest renderedRequest = request;
    auto& config = renderedRequest.generation_config;
    config = Tune(config, hybrid);
    ValidateConfig(config);
    for (const auto& message : request.messages) {
        for (const auto& part : message.content) {
            if (part.type == ContentType::Image) throw LegacyTextError(AILA_ERR_VISION_NOT_ENABLED, "Vision content is not enabled for this backend");
            if (part.type == ContentType::Video) throw LegacyTextError(AILA_ERR_TEMPLATE, "Video content is not enabled yet in this backend");
            if (part.type == ContentType::Audio) throw LegacyTextError(AILA_ERR_RUNTIME, "Audio encoder is not initialized");
        }
    }
    chat::ChatFormatTextResult rendered;
    std::string error;
    if (!formatter.render_text(FormatInput(renderedRequest), true, rendered, &error))
        throw LegacyTextError(AILA_ERR_TEMPLATE, error);
    auto full = tokenizer.encode(rendered.text);
    if (full.empty()) throw LegacyTextError(AILA_ERR_TEMPLATE, "Chat template encoded an empty prompt");
    if (full.size() >= session->Capacity()) throw LegacyTextError(AILA_ERR_CONTEXT_OVERFLOW, "Prompt exceeds context window");
    const uint32_t maximum = std::min(static_cast<uint32_t>(config.max_new_tokens), session->Capacity() - static_cast<uint32_t>(full.size()));
    if (maximum > session->MaxGeneratedTokens())
        throw LegacyTextError(AILA_ERR_RUNTIME, "Generation exceeds the shared runtime's negotiated token limit");
    // Every generation Request carries at least one prompt token. Reject an
    // impossible reservation before Truncate/Replay changes useful context.
    // Peers may still exhaust the shared credit at Submit; this is a static
    // capacity check, not a reservation of currently available credit.
    if (maximum && maximum >= session->MaxOutstandingTokens())
        throw LegacyTextError(AILA_ERR_RUNTIME, "Generation leaves no shared token credit for its required prompt");

    const bool restoreThink = hybrid && (EndsWith(rendered.text, "<think>\n") || EndsWith(rendered.text, "<think>\r\n") || EndsWith(rendered.text, "<think>"));
    size_t anchor = 0;
    if (hybrid) {
        chat::ChatFormatTextResult stable;
        if (formatter.render_text(FormatInput(renderedRequest), false, stable, &error)) {
            const auto stableIds = tokenizer.encode(stable.text);
            if (!stableIds.empty() && stableIds.size() < full.size() && std::equal(stableIds.begin(), stableIds.end(), full.begin())) {
                anchor = stableIds.size();
                auto header = tokenizer.encode("assistant\n");
                header.insert(header.begin(), tokenizer.im_start_id());
                if (anchor + header.size() < full.size() && std::equal(header.begin(), header.end(), full.begin() + anchor)) anchor += header.size();
            }
        }
    }
    AilaInferenceRequestOptions options{};
    options.struct_size = sizeof(options);
    options.abi_version = AILA_INFERENCE_ABI_VERSION;
    options.max_new_tokens = maximum;
    options.prefill_chunk_size = prefillChunk;
    options.do_sample = config.do_sample;
    options.use_fixed_seed = config.use_fixed_seed;
    options.sampling_seed = config.sampling_seed;
    options.temperature = std::max(config.temperature, 1e-6f);
    options.top_k = static_cast<uint32_t>(std::max(0, config.top_k));
    options.top_p = std::clamp(config.top_p, 1e-6f, 1.0f);
    options.repetition_penalty = config.do_sample ? config.repetition_penalty : 1.0f;
    options.presence_penalty = config.presence_penalty;
    options.frequency_penalty = config.frequency_penalty;
    // Zero affects the chat template only in the legacy ABI.
    options.thinking_budget_tokens = config.thinking_budget_tokens > 0 ? config.thinking_budget_tokens : -1;
    options.thinking_open_token_id = tokenizer.special_token_id("<think>");
    options.thinking_close_token_id = tokenizer.special_token_id("</think>");
    std::vector<int> closes;
    if (options.thinking_budget_tokens > 0) {
        if (options.thinking_open_token_id < 0 || options.thinking_close_token_id < 0)
            throw LegacyTextError(AILA_ERR_TEMPLATE, "Reasoning budget requires tokenizer thinking markers");
        closes = {options.thinking_close_token_id};
        const auto suffix = tokenizer.encode("\n\n");
        closes.insert(closes.end(), suffix.begin(), suffix.end());
        options.thinking_close_token_ids = closes.data();
        options.thinking_close_token_count = static_cast<uint32_t>(closes.size());
        options.prompt_opens_think = restoreThink;
    } else {
        options.thinking_open_token_id = options.thinking_close_token_id = -1;
    }
    std::vector<int> stops;
    const int vocab = spec.family == ModelFamily::Qwen35Hybrid ? spec.qwen35_text.vocab_size : spec.qwen3.vocab_size;
    for (int id = 0; id < vocab; ++id) if (tokenizer.is_eos(id)) stops.push_back(id);
    options.stop_token_ids = stops.data();
    options.stop_token_count = static_cast<uint32_t>(stops.size());
    AilaInferenceRequestPolicy policy{};
    policy.struct_size = sizeof(policy);
    policy.abi_version = AILA_INFERENCE_ABI_VERSION;
    policy.thinking_policy = AILA_INFERENCE_THINKING_CLOSE_ONCE;
    if (hybrid) policy.prefill_logits_policy = AILA_INFERENCE_PREFILL_LOGITS_LEGACY;
    // Legacy positive-budget decoding samples on the host even when the
    // device sampler could otherwise handle this top-k configuration.
    if (config.do_sample && (config.thinking_budget_tokens > 0 || config.temperature <= 0 || config.top_p <= 0))
        policy.sampling_policy = AILA_INFERENCE_SAMPLING_HOST;

    size_t common = 0;
    while (common < std::min(cachedIds.size(), full.size()) && cachedIds[common] == full[common]) ++common;
    const auto actual = session->ContextLength();
    if (actual != cachedIds.size()) common = 0;
    size_t first = session->Truncate(static_cast<uint32_t>(std::min(common, static_cast<size_t>(actual))));
    cachedIds.resize(first);
    if (first == full.size()) { session->Reset(); cachedIds.clear(); first = 0; }
    if (anchor > first && anchor < full.size()) { Replay(full, first, anchor); first = anchor; }
    // Bound command attachments without inventing new generation/RNG steps.
    while (full.size() - first > session->MaxPromptTokens()) {
        const size_t next = first + session->MaxPromptTokens();
        Replay(full, first, next);
        first = next;
    }
    Generated generated;
    generated.templateName = rendered.template_name;
    if (!maximum) {
        Replay(full, first, full.size());
        cachedIds = std::move(full);
        generated.finishReason = "length";
        return generated;
    }
    auto stream = session->Submit({full.begin() + first, full.end()}, options, policy);
    std::vector<int> accepted;
    accepted.reserve(maximum);
    chat::ThinkingBudgetController thinking;
    thinking.start(restoreThink, config.thinking_budget_tokens,
        tokenizer.special_token_id("<think>"), tokenizer.special_token_id("</think>"));
    bool prefixDelivered = false;
    bool stop = false;
    int last = -1, run = 0;
    const auto cancelAndDrain = [&] {
        stream->Cancel();
        while (stream->Read().kind != AILA_INFERENCE_EVENT_TERMINAL) {}
    };
    try {
        for (;;) {
            const auto event = stream->Read();
            if (event.kind == AILA_INFERENCE_EVENT_TERMINAL) {
                if (event.progress.status != AILA_INFERENCE_COMPLETED)
                    throw LegacyTextError(AILA_ERR_RUNTIME, "Text request ended without caller cancellation");
                generated.finishReason = event.progress.end_reason == AILA_INFERENCE_END_LENGTH ? "length" : "stop";
                break;
            }
            if (event.kind != AILA_INFERENCE_EVENT_TOKEN || event.token_id < 0 || event.token_id >= vocab)
                throw LegacyTextError(AILA_ERR_RUNTIME, "Text request returned an invalid token event");
            if (tokenizer.is_eos(event.token_id)) { stop = true; break; }
            accepted.push_back(event.token_id);
            run = event.token_id == last ? run + 1 : 1;
            last = event.token_id;
            if (run >= 48) { generated.finishReason = "loop_guard"; stop = true; break; }
            if (callback) {
                const auto piece = tokenizer.decode(event.token_id);
                if (restoreThink && !prefixDelivered) {
                    prefixDelivered = true;
                    if (piece != "<think>") {
                        generated.text += "<think>\n";
                        if (!callback("<think>\n")) { generated.aborted = stop = true; break; }
                    }
                }
                generated.text += piece;
                if (!callback(piece)) { generated.aborted = stop = true; break; }
            }
            thinking.observe_generated_token(event.token_id);
            if (thinking.needs_forced_close()) {
                if (maximum - accepted.size() < closes.size()) generated.truncatedClose = true;
                else { generated.forcedClose = true; thinking.mark_forced_close(); }
            }
        }
        if (stop) cancelAndDrain();
    } catch (...) {
        const auto cause = std::current_exception();
        // A failed connection may reject drain. Preserve the original failure;
        // request release still asks its private connection to stop this stream.
        try { cancelAndDrain(); Align(full, accepted); } catch (...) { cachedIds.clear(); }
        std::rethrow_exception(cause);
    }
    stream.reset();
    if (!callback && !accepted.empty()) generated.text = tokenizer.decode(accepted);
    if (restoreThink && !generated.text.empty() && generated.text.compare(0, 7, "<think>") != 0)
        generated.text.insert(0, "<think>\n");
    Align(full, accepted);
    cachedIds = std::move(full);
    cachedIds.insert(cachedIds.end(), accepted.begin(), accepted.end());
    return generated;
}

chat::AssistantChatResult LegacyTextSession::Result(const chat::ChatRequest& request, const Generated& generated) const {
    auto result = chat::parse_assistant_output(generated.text);
    result.finish_reason = chat::decode_finish_reason(generated.finishReason == "loop_guard", generated.finishReason == "length", !result.tool_calls.empty());
    if (generated.forcedClose) result.warnings.emplace_back("reasoning budget exhausted; forced </think>");
    if (generated.truncatedClose) result.warnings.emplace_back("reasoning budget exhausted but max_new_tokens prevented forced close");
    const auto policy = chat::validate_tool_policy(request, result);
    result.warnings.insert(result.warnings.end(), policy.warnings.begin(), policy.warnings.end());
    if (policy.hard_error) result.finish_reason = "tool_policy";
    result.metadata.template_name = generated.templateName;
    result.metadata.model_family = spec.family == ModelFamily::Qwen35Hybrid ? "qwen3_5_hybrid" : "qwen3";
    result.metadata.reasoning_budget_tokens = request.generation_config.thinking_budget_tokens;
    result.metadata.reasoning_budget_forced_close = generated.forcedClose;
    result.metadata.reasoning_budget_truncated = generated.truncatedClose;
    result.metadata.tool_policy = request.tool_policy == chat::ToolPolicyMode::Strict ? "strict" : "warn";
    result.metadata.tool_choice = Choice(request.tool_choice);
    if (request.tool_choice == chat::ToolChoice::Function) result.metadata.tool_choice += request.tool_choice_function_name;
    return result;
}

LegacyTextResult LegacyTextSession::Generate(std::string_view method, const std::string& input,
    GenerationConfig config, TokenCallback callback, ChatCallback structured) {
    LogScope logs(source);
    const bool stateful = method == "generate" || method == "generate.stream";
    if (stateful) {
        history.set_system_prompt(systemPrompt);
        history.add_user_text(input);
        try {
            for (;;) {
                auto request = history.to_request_without_reasoning();
                request.generation_config = config;
                try {
                    auto generated = GenerateChat(request, callback);
                    history.add_assistant_result(chat::parse_assistant_output(generated.text), false);
                    return {std::move(generated.text), generated.aborted};
                } catch (const LegacyTextError& error) {
                    if (error.Code() != AILA_ERR_CONTEXT_OVERFLOW || !history.drop_oldest_turn()) throw;
                }
            }
        } catch (...) { history.remove_last_user_message(); throw; }
    }
    chat::ChatRequest request;
    std::string error;
    if (!chat::parse_chat_request_json(input, config, request, &error)) throw LegacyTextError(AILA_ERR_JSON_PARSE, error);
    chat::StructuredStreamParser parser;
    if (structured) callback = [&](const std::string& piece) {
        std::vector<chat::StructuredStreamEvent> events;
        parser.push(piece, events);
        for (const auto& event : events) if (!structured(event)) return false;
        return true;
    };
    auto generated = GenerateChat(request, callback);
    if (generated.aborted) return {std::move(generated.text), true};
    if (method == "generate.messages" || method == "generate.messages_stream") return {std::move(generated.text), false};
    const auto result = Result(request, generated);
    if (structured) {
        std::vector<chat::StructuredStreamEvent> tail;
        parser.finish(tail);
        for (const auto& event : tail) if (!structured(event)) return {generated.text, true};
        if (!structured(chat::final_stream_event_from_result(result))) return {generated.text, true};
    }
    return {chat::assistant_result_to_json(result), false};
}

int LegacyTextSession::Deliver(const chat::StructuredStreamEvent& input, AilaChatStreamCallback callback, void* userData) {
    AilaChatStreamEvent event{};
    event.struct_size = sizeof(event);
    switch (input.type) {
    case chat::StructuredStreamEventType::ReasoningDelta: event.type = AILA_CHAT_STREAM_REASONING_DELTA; break;
    case chat::StructuredStreamEventType::ContentDelta: event.type = AILA_CHAT_STREAM_CONTENT_DELTA; break;
    case chat::StructuredStreamEventType::ToolCallDelta: event.type = AILA_CHAT_STREAM_TOOL_CALL_DELTA; break;
    case chat::StructuredStreamEventType::Warning: event.type = AILA_CHAT_STREAM_WARNING; break;
    case chat::StructuredStreamEventType::Final: event.type = AILA_CHAT_STREAM_FINAL; break;
    }
    const auto nullable = [](const std::string& text) { return text.empty() ? nullptr : text.c_str(); };
    event.text = nullable(input.text);
    event.tool_call_id = nullable(input.tool_call_id);
    event.tool_name = nullable(input.tool_name);
    event.arguments_delta = nullable(input.arguments_delta);
    event.finish_reason = nullable(input.finish_reason);
    std::string warnings, calls;
    if (!input.warnings.empty()) {
        warnings = "[";
        for (const auto& warning : input.warnings) {
            if (warnings.size() > 1) warnings += ',';
            warnings += chat::json_escape_string(warning);
        }
        warnings += ']';
    }
    if (input.type == chat::StructuredStreamEventType::Final && !input.tool_calls.empty()) calls = chat::tool_calls_to_json(input.tool_calls);
    event.warnings_json = nullable(warnings);
    event.tool_calls_json = nullable(calls);
    return callback(&event, userData);
}
} // namespace aila::proxy
