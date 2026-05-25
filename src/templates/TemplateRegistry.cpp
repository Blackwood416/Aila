#include "TemplateRegistry.hpp"
#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"
#include "jinja/value.h"
#include "profile/Profiling.hpp"
#include <cctype>

namespace aila {
namespace templating {

namespace {

void set_error(std::string* err, const std::string& msg) {
    if (err) *err = msg;
}

void rtrim(std::string& text) {
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) {
        text.pop_back();
    }
}

bool strip_no_think_suffix(std::string& text) {
    rtrim(text);
    constexpr const char* kNoThinkCmd = "/no_think";
    constexpr size_t kNoThinkCmdLen = 9;
    if (text.size() < kNoThinkCmdLen) {
        return false;
    }

    size_t pos = text.size() - kNoThinkCmdLen;
    if (text.compare(pos, kNoThinkCmdLen, kNoThinkCmd) != 0) {
        return false;
    }

    bool boundary_ok = (pos == 0) || std::isspace(static_cast<unsigned char>(text[pos - 1]));
    if (!boundary_ok) {
        return false;
    }

    text.erase(pos);
    rtrim(text);
    return true;
}

bool strip_think_suffix(std::string& text) {
    rtrim(text);
    constexpr const char* kThinkCmd = "/think";
    constexpr size_t kThinkCmdLen = 6;
    if (text.size() < kThinkCmdLen) {
        return false;
    }

    size_t pos = text.size() - kThinkCmdLen;
    if (text.compare(pos, kThinkCmdLen, kThinkCmd) != 0) {
        return false;
    }

    // Reject if preceded by "no_" — /no_think ends with "_think", not "/think",
    // but the boundary check already handles this correctly since '_' is not space.
    bool boundary_ok = (pos == 0) || std::isspace(static_cast<unsigned char>(text[pos - 1]));
    if (!boundary_ok) {
        return false;
    }

    text.erase(pos);
    rtrim(text);
    return true;
}

void append_encoded(const Tokenizer& tokenizer, std::vector<int>& ids, const std::string& text) {
    auto t = tokenizer.encode(text);
    ids.insert(ids.end(), t.begin(), t.end());
}

bool collect_text_content(const std::vector<ContentPart>& parts,
                          bool allow_vision_placeholders,
                          const Tokenizer& tokenizer,
                          std::string& out,
                          std::string* error_message) {
    out.clear();
    for (const auto& p : parts) {
        if (p.type == ContentType::Text) {
            out += p.text;
            continue;
        }
        if (!allow_vision_placeholders) {
            set_error(error_message, "Vision content is not enabled for this backend");
            return false;
        }
        if (p.type == ContentType::Image) {
            out += "<|vision_start|><|image_pad|><|vision_end|>";
        } else if (p.type == ContentType::Video) {
            out += "<|vision_start|><|video_pad|><|vision_end|>";
        } else {
            set_error(error_message, "Unknown content part type");
            return false;
        }
    }

    // Ensure placeholder tokens are known by tokenizer when vision is enabled.
    if (allow_vision_placeholders) {
        if (tokenizer.special_token_id("<|vision_start|>") < 0 ||
            tokenizer.special_token_id("<|vision_end|>") < 0) {
            set_error(error_message, "Tokenizer does not contain vision special tokens");
            return false;
        }
    }
    return true;
}

bool render_chatml(const Tokenizer& tokenizer,
                   const std::vector<Message>& messages,
                   bool allow_vision_placeholders,
                   bool add_generation_prompt,
                   bool qwen35_style_prompt,
                   bool qwen35_closed_think_prompt,
                   std::vector<int>& out_ids,
                   std::string* error_message) {
    out_ids.clear();
    if (messages.empty()) {
        set_error(error_message, "No messages provided");
        return false;
    }

    int im_start = tokenizer.im_start_id();
    int im_end = tokenizer.im_end_id();
    if (im_start < 0 || im_end < 0) {
        set_error(error_message, "Tokenizer does not provide <|im_start|>/<|im_end|> IDs");
        return false;
    }

    bool disable_thinking = false;
    bool force_thinking = false;
    std::string merged;
    for (size_t i = 0; i < messages.size(); ++i) {
        const auto& m = messages[i];
        if (m.role != "system" && m.role != "user" && m.role != "assistant" && m.role != "tool") {
            set_error(error_message, "Invalid message role: " + m.role);
            return false;
        }
        if (!collect_text_content(m.content, allow_vision_placeholders, tokenizer, merged, error_message)) {
            return false;
        }
        // Strip suffix from ALL user messages so the model never sees raw
        // /no_think or /think in multi-turn context. Only use detection
        // result from the last user message for think-mode control.
        if (m.role == "user") {
            bool is_last = add_generation_prompt && (i + 1 == messages.size());
            if (is_last) {
                disable_thinking = strip_no_think_suffix(merged);
                if (!disable_thinking) {
                    force_thinking = strip_think_suffix(merged);
                }
            } else {
                strip_no_think_suffix(merged);
                strip_think_suffix(merged);
            }
        }

        out_ids.push_back(im_start);
        append_encoded(tokenizer, out_ids, m.role + "\n" + merged);
        out_ids.push_back(im_end);
        append_encoded(tokenizer, out_ids, "\n");
    }

    if (add_generation_prompt) {
        out_ids.push_back(im_start);
        append_encoded(tokenizer, out_ids, "assistant\n");
        if (qwen35_style_prompt) {
            // /no_think → closed think; /think → force open think
            bool use_closed = (qwen35_closed_think_prompt || disable_thinking) && !force_thinking;
            if (use_closed) {
                append_encoded(tokenizer, out_ids, "<think>\n\n</think>\n\n");
            } else {
                append_encoded(tokenizer, out_ids, "<think>\n");
            }
        } else if (disable_thinking) {
            int end_think_id = tokenizer.special_token_id("</think>");
            if (end_think_id >= 0) {
                out_ids.push_back(end_think_id);
                append_encoded(tokenizer, out_ids, "\n");
            }
            append_encoded(tokenizer, out_ids, "Please answer directly and briefly.\n");
        }
    }
    return true;
}

bool render_jinja(const Tokenizer& tokenizer,
                  const std::vector<Message>& messages,
                  bool vision_enabled,
                  bool add_generation_prompt,
                  std::vector<int>& out_ids,
                  std::string* error_message) {
    try {
        const std::string& chat_template = tokenizer.chat_template();
        if (chat_template.empty()) {
            set_error(error_message, "chat_template is empty");
            return false;
        }

        jinja::lexer lex;
        jinja::lexer_result lex_res;
        try {
            lex_res = lex.tokenize(chat_template);
        } catch (const std::exception& e) {
            set_error(error_message, std::string("Jinja tokenization failed: ") + e.what());
            return false;
        }

        jinja::program program;
        try {
            program = jinja::parse_from_tokens(lex_res);
        } catch (const std::exception& e) {
            set_error(error_message, std::string("Jinja parsing failed: ") + e.what());
            return false;
        }

        jinja::context ctx(chat_template);

        ctx.set_val("bos_token", jinja::mk_val<jinja::value_string>(tokenizer.bos_token()));
        ctx.set_val("eos_token", jinja::mk_val<jinja::value_string>(tokenizer.eos_token()));
        ctx.set_val("add_generation_prompt", jinja::mk_val<jinja::value_bool>(add_generation_prompt));

        auto messages_arr = jinja::mk_val<jinja::value_array>();
        for (const auto& msg : messages) {
            auto msg_obj = jinja::mk_val<jinja::value_object>();
            msg_obj->insert("role", jinja::mk_val<jinja::value_string>(msg.role));

            std::string text_content;
            std::string err;
            if (!collect_text_content(msg.content, vision_enabled, tokenizer, text_content, &err)) {
                set_error(error_message, err);
                return false;
            }

            auto val_content = jinja::mk_val<jinja::value_string>(text_content);
            val_content->mark_input();

            msg_obj->insert("content", val_content);
            messages_arr->push_back(msg_obj);
        }
        ctx.set_val("messages", messages_arr);

        jinja::runtime rt(ctx);
        jinja::value_array res_arr;
        try {
            res_arr = rt.execute(program);
        } catch (const std::exception& e) {
            set_error(error_message, std::string("Jinja execution failed: ") + e.what());
            return false;
        }

        jinja::value_string final_str = jinja::runtime::gather_string_parts(res_arr);
        std::string rendered = jinja::render_string_parts(final_str);

        out_ids = tokenizer.encode(rendered);
        return true;
    } catch (const std::exception& e) {
        set_error(error_message, std::string("Jinja render failed with exception: ") + e.what());
        return false;
    }
}

} // namespace

bool TemplateRegistry::render(const ModelSpec& spec,
                              const Tokenizer& tokenizer,
                              const std::vector<Message>& messages,
                              bool vision_enabled,
                              bool add_generation_prompt,
                              std::vector<int>& out_ids,
                              std::string* error_message) const {
    if (!tokenizer.chat_template().empty()) {
        std::string jinja_err;
        if (render_jinja(tokenizer, messages, vision_enabled, add_generation_prompt, out_ids, &jinja_err)) {
            return true;
        }
        AILA_LOG_WARN("[TemplateRegistry] Jinja render failed, falling back to legacy ChatML renderer. Error: %s",
                      jinja_err.c_str());
    }

    if (spec.family == ModelFamily::Qwen35Hybrid) {
        bool closed_think_prompt = is_exact_qwen35_hybrid_0p8b_spec(spec.qwen35_text);
        return render_chatml(tokenizer, messages, vision_enabled, add_generation_prompt, true,
                             closed_think_prompt, out_ids, error_message);
    }
    return render_chatml(tokenizer, messages, false, add_generation_prompt, false,
                         false, out_ids, error_message);
}

} // namespace templating
} // namespace aila
