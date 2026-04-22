#include "TemplateRegistry.hpp"
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
        if (add_generation_prompt && i + 1 == messages.size() && m.role == "user") {
            disable_thinking = strip_no_think_suffix(merged);
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
            if (qwen35_closed_think_prompt || disable_thinking) {
                append_encoded(tokenizer, out_ids, "<think>\n\n</think>\n\n");
            } else {
                append_encoded(tokenizer, out_ids, "<think>\n");
            }
        }
    }
    return true;
}

} // namespace

bool TemplateRegistry::render(const ModelSpec& spec,
                              const Tokenizer& tokenizer,
                              const std::vector<Message>& messages,
                              bool vision_enabled,
                              bool add_generation_prompt,
                              std::vector<int>& out_ids,
                              std::string* error_message) const {
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
