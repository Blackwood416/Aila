#include "Tokenizer.hpp"
#include "simdjson.h"
#include <fstream>
#include <sstream>
#include "profile/Profiling.hpp"
#include <algorithm>
#include <regex>
#include <cassert>
#include <cctype>

// ============================================================
// GPT-2 style byte-to-unicode mapping
// ============================================================

std::unordered_map<uint8_t, std::string> Tokenizer::byte_to_unicode() {
    std::unordered_map<uint8_t, std::string> b2u;
    std::vector<int> bs;
    for (int i = 33; i <= 126; i++) bs.push_back(i);
    for (int i = 161; i <= 172; i++) bs.push_back(i);
    for (int i = 174; i <= 255; i++) bs.push_back(i);
    std::vector<int> cs(bs.begin(), bs.end());
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }
    for (size_t i = 0; i < bs.size(); i++) {
        uint8_t byte_val = static_cast<uint8_t>(bs[i]);
        int cp = cs[i];
        std::string utf8_char;
        if (cp < 0x80) {
            utf8_char = std::string(1, static_cast<char>(cp));
        } else if (cp < 0x800) {
            utf8_char += static_cast<char>(0xC0 | (cp >> 6));
            utf8_char += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            utf8_char += static_cast<char>(0xE0 | (cp >> 12));
            utf8_char += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            utf8_char += static_cast<char>(0x80 | (cp & 0x3F));
        }
        b2u[byte_val] = utf8_char;
    }
    return b2u;
}

std::unordered_map<std::string, uint8_t> Tokenizer::unicode_to_byte() {
    auto b2u = byte_to_unicode();
    std::unordered_map<std::string, uint8_t> u2b;
    for (auto& p : b2u) {
        u2b[p.second] = p.first;
    }
    return u2b;
}

// ============================================================
// Load vocab.json + merges.txt
// ============================================================

bool Tokenizer::load(const std::string& model_dir) {
    std::string vocab_path = model_dir + "/vocab.json";
    std::string merges_path = model_dir + "/merges.txt";

    // --- Load vocab.json ---
    try {
        std::ifstream vf(vocab_path, std::ios::binary);
        if (!vf.is_open()) {
            AILA_LOG_ERROR("[Tokenizer] Cannot open %s", vocab_path.c_str());
            return false;
        }
        std::string vocab_str((std::istreambuf_iterator<char>(vf)),
                               std::istreambuf_iterator<char>());
        vf.close();

        simdjson::padded_string padded(vocab_str);
        simdjson::ondemand::parser parser;
        simdjson::ondemand::document doc = parser.iterate(padded);

        int max_id = 0;
        {
            simdjson::padded_string padded2(vocab_str);
            simdjson::ondemand::parser parser2;
            simdjson::ondemand::document doc2 = parser2.iterate(padded2);
            for (auto field : doc2.get_object()) {
                int64_t id = field.value().get_int64();
                if (id > max_id) max_id = static_cast<int>(id);
            }
        }

        id_to_token_.resize(max_id + 1);

        for (auto field : doc.get_object()) {
            std::string_view key = field.unescaped_key();
            int64_t id = field.value().get_int64();
            std::string token_str(key);
            token_to_id_[token_str] = static_cast<int>(id);
            if (id >= 0 && id < static_cast<int64_t>(id_to_token_.size())) {
                id_to_token_[static_cast<int>(id)] = token_str;
            }
        }

        AILA_LOG_INFO("[Tokenizer] Loaded vocab: %zu tokens (max_id=%d)",
                      token_to_id_.size(), max_id);

    } catch (const std::exception& e) {
        AILA_LOG_ERROR("[Tokenizer] Failed to load vocab: %s", e.what());
        return false;
    }

    // --- Load merges.txt ---
    try {
        std::ifstream mf(merges_path);
        if (!mf.is_open()) {
            AILA_LOG_ERROR("[Tokenizer] Cannot open %s", merges_path.c_str());
            return false;
        }

        std::string line;
        int rank = 0;
        while (std::getline(mf, line)) {
            if (line.empty() || line[0] == '#') continue;
            auto space_pos = line.find(' ');
            if (space_pos == std::string::npos) continue;

            MergeRule rule;
            rule.a = line.substr(0, space_pos);
            rule.b = line.substr(space_pos + 1);
            rule.merged = rule.a + rule.b;

            merges_.push_back(rule);
            merge_rank_[rule.a + " " + rule.b] = rank;
            rank++;
        }
        mf.close();

        AILA_LOG_INFO("[Tokenizer] Loaded %zu merge rules", merges_.size());

    } catch (const std::exception& e) {
        AILA_LOG_ERROR("[Tokenizer] Failed to load merges: %s", e.what());
        return false;
    }

    // --- Load special tokens from tokenizer.json ---
    try {
        std::string tokenizer_json_path = model_dir + "/tokenizer.json";
        std::ifstream tf(tokenizer_json_path, std::ios::binary);
        if (tf.is_open()) {
            std::string tj_str((std::istreambuf_iterator<char>(tf)),
                                std::istreambuf_iterator<char>());
            tf.close();

            simdjson::padded_string tj_padded(tj_str);
            simdjson::ondemand::parser tj_parser;
            simdjson::ondemand::document tj_doc = tj_parser.iterate(tj_padded);

            auto added_tokens = tj_doc["added_tokens"].get_array();
            for (auto token_obj : added_tokens) {
                std::string_view content = token_obj["content"].get_string();
                int64_t id = token_obj["id"].get_int64();
                std::string tok_str(content);

                // Add to vocab maps
                token_to_id_[tok_str] = static_cast<int>(id);
                if (id >= static_cast<int64_t>(id_to_token_.size())) {
                    id_to_token_.resize(static_cast<size_t>(id) + 1);
                }
                id_to_token_[static_cast<int>(id)] = tok_str;
                special_tokens_[tok_str] = static_cast<int>(id);
            }

            AILA_LOG_INFO("[Tokenizer] Loaded %zu special tokens from tokenizer.json",
                          special_tokens_.size());
        }
    } catch (const std::exception& e) {
        AILA_LOG_WARN("[Tokenizer] Could not load tokenizer.json: %s", e.what());
    }

    // Set key special token IDs
    auto find_special = [&](const std::string& token) -> int {
        auto it = special_tokens_.find(token);
        return (it != special_tokens_.end()) ? it->second : -1;
    };

    im_start_id_ = find_special("<|im_start|>");
    im_end_id_ = find_special("<|im_end|>");
    eot_id_ = find_special("<|endoftext|>");
    eos_id_ = im_end_id_;

    AILA_LOG_INFO("[Tokenizer] Special tokens: im_start=%d im_end=%d eot=%d total_special=%zu",
                  im_start_id_, im_end_id_, eot_id_, special_tokens_.size());

    return true;
}


// ============================================================
// Pretokenize: split text into words
// ============================================================

std::vector<std::string> Tokenizer::pretokenize(const std::string& text) const {
    std::vector<std::string> result;
    if (text.empty()) return result;

    std::string remaining = text;
    while (!remaining.empty()) {
        // Check for special tokens first
        bool found_special = false;
        for (auto& sp : special_tokens_) {
            if (remaining.size() >= sp.first.size() &&
                remaining.substr(0, sp.first.size()) == sp.first) {
                result.push_back(sp.first);
                remaining = remaining.substr(sp.first.size());
                found_special = true;
                break;
            }
        }
        if (found_special) continue;

        size_t end = 1;
        unsigned char c0 = static_cast<unsigned char>(remaining[0]);

        if (c0 <= 0x7F) {
            if (std::isalpha(c0)) {
                while (end < remaining.size() && std::isalpha(static_cast<unsigned char>(remaining[end])))
                    end++;
            } else if (std::isdigit(c0)) {
                while (end < remaining.size() && std::isdigit(static_cast<unsigned char>(remaining[end])))
                    end++;
            } else if (c0 == ' ') {
                while (end < remaining.size() && std::isalpha(static_cast<unsigned char>(remaining[end])))
                    end++;
                if (end == 1) {
                    while (end < remaining.size() && remaining[end] == ' ')
                        end++;
                }
            } else {
                end = 1;
            }
        } else {
            if ((c0 & 0xE0) == 0xC0) end = 2;
            else if ((c0 & 0xF0) == 0xE0) end = 3;
            else if ((c0 & 0xF8) == 0xF0) end = 4;
            if (end > remaining.size()) end = remaining.size();
        }

        result.push_back(remaining.substr(0, end));
        remaining = remaining.substr(end);
    }

    return result;
}

// ============================================================
// Split word to byte-level chars
// ============================================================

std::vector<std::string> Tokenizer::split_to_chars(const std::string& word) const {
    static auto b2u = byte_to_unicode();
    std::vector<std::string> chars;
    for (unsigned char c : word) {
        auto it = b2u.find(c);
        if (it != b2u.end()) {
            chars.push_back(it->second);
        } else {
            chars.push_back(std::string(1, static_cast<char>(c)));
        }
    }
    return chars;
}

// ============================================================
// BPE encode: apply merge rules
// ============================================================

std::vector<std::string> Tokenizer::bpe_encode(const std::vector<std::string>& tokens) const {
    if (tokens.size() <= 1) return tokens;
    std::vector<std::string> word(tokens);

    while (word.size() > 1) {
        int best_rank = INT_MAX;
        int best_idx = -1;

        for (size_t i = 0; i + 1 < word.size(); i++) {
            std::string pair_key = word[i] + " " + word[i + 1];
            auto it = merge_rank_.find(pair_key);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = static_cast<int>(i);
            }
        }

        if (best_idx < 0) break;

        std::vector<std::string> new_word;
        for (size_t i = 0; i < word.size(); ) {
            if (static_cast<int>(i) == best_idx) {
                new_word.push_back(word[i] + word[i + 1]);
                i += 2;
            } else {
                new_word.push_back(word[i]);
                i += 1;
            }
        }
        word = new_word;
    }

    return word;
}

// ============================================================
// Encode: text -> token IDs
// ============================================================

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> ids;
    auto words = pretokenize(text);

    for (const auto& word : words) {
        auto sp_it = special_tokens_.find(word);
        if (sp_it != special_tokens_.end()) {
            ids.push_back(sp_it->second);
            continue;
        }

        auto chars = split_to_chars(word);
        auto bpe_tokens = bpe_encode(chars);

        for (const auto& tok : bpe_tokens) {
            auto it = token_to_id_.find(tok);
            if (it != token_to_id_.end()) {
                ids.push_back(it->second);
            }
        }
    }

    return ids;
}

// ============================================================
// Decode
// ============================================================

std::string Tokenizer::decode(int token_id) const {
    if (token_id < 0 || token_id >= static_cast<int>(id_to_token_.size())) {
        return "";
    }

    const std::string& token = id_to_token_[token_id];

    if (special_tokens_.count(token)) {
        // Keep think markers visible by default so users can observe CoT boundaries.
        if (token == "<think>" || token == "</think>") {
            return token;
        }
        return "";
    }

    static auto u2b = unicode_to_byte();
    std::string result;

    size_t i = 0;
    while (i < token.size()) {
        bool matched = false;
        for (int len = 4; len >= 1; len--) {
            if (i + len <= token.size()) {
                std::string sub = token.substr(i, len);
                auto it = u2b.find(sub);
                if (it != u2b.end()) {
                    result += static_cast<char>(it->second);
                    i += len;
                    matched = true;
                    break;
                }
            }
        }
        if (!matched) {
            result += token[i];
            i++;
        }
    }

    return result;
}

std::string Tokenizer::decode(const std::vector<int>& token_ids) const {
    std::string result;
    for (int id : token_ids) {
        result += decode(id);
    }
    return result;
}

// ============================================================
// Apply ChatML template
// ============================================================

std::vector<int> Tokenizer::apply_chat_template(
    const std::string& system_prompt,
    const std::string& user_message) const {

    std::vector<int> ids;
    std::string cleaned_user_message = user_message;
    bool disable_think = false;

    auto rtrim = [](std::string& s) {
        while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) {
            s.pop_back();
        }
    };

    rtrim(cleaned_user_message);
    const std::string kNoThinkCmd = "/no_think";
    if (cleaned_user_message.size() >= kNoThinkCmd.size()) {
        size_t pos = cleaned_user_message.size() - kNoThinkCmd.size();
        if (cleaned_user_message.compare(pos, kNoThinkCmd.size(), kNoThinkCmd) == 0) {
            bool boundary_ok = (pos == 0) ||
                               std::isspace(static_cast<unsigned char>(cleaned_user_message[pos - 1]));
            if (boundary_ok) {
                disable_think = true;
                cleaned_user_message.erase(pos);
                rtrim(cleaned_user_message);
            }
        }
    }

    if (!system_prompt.empty()) {
        ids.push_back(im_start_id_);
        auto sys_tokens = encode("system\n" + system_prompt);
        ids.insert(ids.end(), sys_tokens.begin(), sys_tokens.end());
        ids.push_back(im_end_id_);
        auto nl = encode("\n");
        ids.insert(ids.end(), nl.begin(), nl.end());
    }

    ids.push_back(im_start_id_);
    auto user_tokens = encode("user\n" + cleaned_user_message);
    ids.insert(ids.end(), user_tokens.begin(), user_tokens.end());
    ids.push_back(im_end_id_);
    auto nl = encode("\n");
    ids.insert(ids.end(), nl.begin(), nl.end());

    ids.push_back(im_start_id_);
    auto asst_tokens = encode("assistant\n");
    ids.insert(ids.end(), asst_tokens.begin(), asst_tokens.end());

    // Only attempt to skip thinking when user explicitly appends "/no_think".
    if (disable_think) {
        auto end_think_it = special_tokens_.find("</think>");
        if (end_think_it != special_tokens_.end()) {
            auto nl_tokens = encode("\n");
            ids.push_back(end_think_it->second);
            ids.insert(ids.end(), nl_tokens.begin(), nl_tokens.end());

            // Reinforce no-think behavior with an explicit assistant prefix.
            auto direct_tokens = encode("Please answer directly and briefly.\n");
            ids.insert(ids.end(), direct_tokens.begin(), direct_tokens.end());
        }
    }

    return ids;
}

// ============================================================
// Check EOS
// ============================================================

bool Tokenizer::is_eos(int token_id) const {
    return token_id == eos_id_ || token_id == eot_id_;
}

// ============================================================
// Apply ChatML template (multi-turn)
// ============================================================

std::vector<int> Tokenizer::apply_chat_template(
    const std::string& system_prompt,
    const ChatHistory& history) const {

    std::vector<int> ids;

    // System prompt
    if (!system_prompt.empty()) {
        ids.push_back(im_start_id_);
        auto sys_tokens = encode("system\n" + system_prompt);
        ids.insert(ids.end(), sys_tokens.begin(), sys_tokens.end());
        ids.push_back(im_end_id_);
        auto nl = encode("\n");
        ids.insert(ids.end(), nl.begin(), nl.end());
    }

    // All conversation turns
    for (size_t i = 0; i < history.messages().size(); ++i) {
        const auto& msg = history.messages()[i];
        bool is_last_user = (msg.role == "user" && i == history.messages().size() - 1);

        ids.push_back(im_start_id_);

        if (msg.role == "user" || msg.role == "assistant") {
            auto role_tokens = encode(msg.role + "\n" + msg.content);
            ids.insert(ids.end(), role_tokens.begin(), role_tokens.end());
        }

        if (is_last_user) {
            // Last user message: close user turn, open assistant turn
            ids.push_back(im_end_id_);
            auto nl = encode("\n");
            ids.insert(ids.end(), nl.begin(), nl.end());
            ids.push_back(im_start_id_);
            auto asst_tokens = encode("assistant\n");
            ids.insert(ids.end(), asst_tokens.begin(), asst_tokens.end());

            // Check /no_think
            std::string cleaned = msg.content;
            auto rtrim = [](std::string& s) {
                while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
            };
            rtrim(cleaned);
            const std::string kNoThinkCmd = "/no_think";
            if (cleaned.size() >= kNoThinkCmd.size()) {
                size_t pos = cleaned.size() - kNoThinkCmd.size();
                if (cleaned.compare(pos, kNoThinkCmd.size(), kNoThinkCmd) == 0) {
                    auto end_think_it = special_tokens_.find("</think>");
                    if (end_think_it != special_tokens_.end()) {
                        auto nl_tokens = encode("\n");
                        ids.push_back(end_think_it->second);
                        ids.insert(ids.end(), nl_tokens.begin(), nl_tokens.end());
                        auto direct_tokens = encode("Please answer directly and briefly.\n");
                        ids.insert(ids.end(), direct_tokens.begin(), direct_tokens.end());
                    }
                }
            }
        } else {
            // Completed turns: close with im_end
            ids.push_back(im_end_id_);
            auto nl = encode("\n");
            ids.insert(ids.end(), nl.begin(), nl.end());
        }
    }

    return ids;
}

