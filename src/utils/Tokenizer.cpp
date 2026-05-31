#include "Tokenizer.hpp"
#include "simdjson.h"
#include <fstream>
#include <sstream>
#include "profile/Profiling.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cstring>

// ============================================================
// GPT-2 style byte-to-unicode mapping
// ============================================================

namespace {

struct Utf8Char {
    uint32_t cp = 0;
    size_t bytes = 1;
};

Utf8Char decode_utf8_char(const std::string& text, size_t offset) {
    Utf8Char out{};
    if (offset >= text.size()) {
        return out;
    }

    const unsigned char c0 = static_cast<unsigned char>(text[offset]);
    out.cp = c0;
    out.bytes = 1;

    if (c0 < 0x80) {
        return out;
    }

    auto cont = [&](size_t idx) -> uint32_t {
        return static_cast<unsigned char>(text[offset + idx]) & 0x3Fu;
    };

    if ((c0 & 0xE0) == 0xC0 && offset + 1 < text.size()) {
        out.cp = ((c0 & 0x1Fu) << 6) | cont(1);
        out.bytes = 2;
    } else if ((c0 & 0xF0) == 0xE0 && offset + 2 < text.size()) {
        out.cp = ((c0 & 0x0Fu) << 12) | (cont(1) << 6) | cont(2);
        out.bytes = 3;
    } else if ((c0 & 0xF8) == 0xF0 && offset + 3 < text.size()) {
        out.cp = ((c0 & 0x07u) << 18) | (cont(1) << 12) | (cont(2) << 6) | cont(3);
        out.bytes = 4;
    }

    return out;
}

bool is_newline_cp(uint32_t cp) {
    return cp == '\n' || cp == '\r';
}

bool is_whitespace_cp(uint32_t cp) {
    switch (cp) {
    case '\t':
    case '\n':
    case '\v':
    case '\f':
    case '\r':
    case ' ':
        return true;
    default:
        break;
    }

    return cp == 0x0085 || cp == 0x00A0 || cp == 0x1680 ||
           cp == 0x2028 || cp == 0x2029 || cp == 0x202F ||
           cp == 0x205F || cp == 0x3000;
}

bool is_mark_cp(uint32_t cp) {
    return (cp >= 0x0300 && cp <= 0x036F) ||
           (cp >= 0x1AB0 && cp <= 0x1AFF) ||
           (cp >= 0x1DC0 && cp <= 0x1DFF) ||
           (cp >= 0x20D0 && cp <= 0x20FF) ||
           (cp >= 0xFE20 && cp <= 0xFE2F);
}

bool is_digit_cp(uint32_t cp) {
    return (cp >= '0' && cp <= '9') ||
           (cp >= 0x0660 && cp <= 0x0669) ||
           (cp >= 0x06F0 && cp <= 0x06F9) ||
           (cp >= 0x0966 && cp <= 0x096F) ||
           (cp >= 0xFF10 && cp <= 0xFF19);
}

bool is_ascii_punct_cp(uint32_t cp) {
    return cp < 0x80 && std::ispunct(static_cast<unsigned char>(cp)) != 0;
}

bool is_punctuation_or_symbol_cp(uint32_t cp) {
    if (is_ascii_punct_cp(cp)) {
        return true;
    }
    return (cp >= 0x2000 && cp <= 0x206F) ||
           (cp >= 0x20A0 && cp <= 0x214F) ||
           (cp >= 0x2190 && cp <= 0x27BF) ||
           (cp >= 0x2E00 && cp <= 0x2E7F) ||
           (cp >= 0x3000 && cp <= 0x303F) ||
           (cp >= 0xFE10 && cp <= 0xFE1F) ||
           (cp >= 0xFE30 && cp <= 0xFE6F) ||
           (cp >= 0xFF01 && cp <= 0xFF0F) ||
           (cp >= 0xFF1A && cp <= 0xFF20) ||
           (cp >= 0xFF3B && cp <= 0xFF40) ||
           (cp >= 0xFF5B && cp <= 0xFF65) ||
           (cp >= 0x1F300 && cp <= 0x1FAFF);
}

bool is_letter_cp(uint32_t cp) {
    if (cp < 0x80) {
        return std::isalpha(static_cast<unsigned char>(cp)) != 0;
    }
    if (is_whitespace_cp(cp) || is_newline_cp(cp) || is_digit_cp(cp) ||
        is_mark_cp(cp) || is_punctuation_or_symbol_cp(cp)) {
        return false;
    }
    return cp >= 0x80;
}

bool is_letter_or_mark_cp(uint32_t cp) {
    return is_letter_cp(cp) || is_mark_cp(cp);
}

bool is_ascii_case_insensitive_match(const std::string& text,
                                     size_t offset,
                                     const char* literal) {
    for (size_t i = 0; literal[i] != '\0'; ++i) {
        if (offset + i >= text.size()) {
            return false;
        }
        unsigned char a = static_cast<unsigned char>(text[offset + i]);
        unsigned char b = static_cast<unsigned char>(literal[i]);
        if (std::tolower(a) != std::tolower(b)) {
            return false;
        }
    }
    return true;
}

} // namespace

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
    const std::string vocab_path = model_dir + "/vocab.json";
    const std::string merges_path = model_dir + "/merges.txt";
    const std::string tokenizer_json_path = model_dir + "/tokenizer.json";

    token_to_id_.clear();
    id_to_token_.clear();
    merges_.clear();
    merge_rank_.clear();
    special_tokens_.clear();
    special_token_order_.clear();
    chat_template_.clear();
    bos_token_.clear();
    eos_token_.clear();

    auto load_special_tokens_from_tokenizer_json = [&]() {
        try {
            std::ifstream tf(tokenizer_json_path, std::ios::binary);
            if (!tf.is_open()) {
                return;
            }

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

                token_to_id_[tok_str] = static_cast<int>(id);
                if (id >= static_cast<int64_t>(id_to_token_.size())) {
                    id_to_token_.resize(static_cast<size_t>(id) + 1);
                }
                id_to_token_[static_cast<int>(id)] = tok_str;
                special_tokens_[tok_str] = static_cast<int>(id);
            }

            AILA_LOG_INFO("[Tokenizer] Loaded %zu special tokens from tokenizer.json",
                          special_tokens_.size());
        } catch (const std::exception& e) {
            AILA_LOG_WARN("[Tokenizer] Could not load tokenizer.json special tokens: %s", e.what());
        }
    };

    auto load_added_tokens_from_tokenizer_config_json = [&]() {
        try {
            const std::string tokenizer_config_path = model_dir + "/tokenizer_config.json";
            std::ifstream tf(tokenizer_config_path, std::ios::binary);
            if (!tf.is_open()) {
                return;
            }

            std::string tc_str((std::istreambuf_iterator<char>(tf)),
                                std::istreambuf_iterator<char>());
            tf.close();

            simdjson::dom::parser tc_parser;
            simdjson::dom::element tc_root = tc_parser.parse(tc_str);

            // 1. 读取 chat_template
            simdjson::dom::element chat_template_val;
            if (tc_root.at_key("chat_template").get(chat_template_val) == simdjson::SUCCESS) {
                std::string_view chat_template_sv;
                if (chat_template_val.get_string().get(chat_template_sv) == simdjson::SUCCESS) {
                    chat_template_ = std::string(chat_template_sv);
                    AILA_LOG_INFO("[Tokenizer] Loaded chat_template from tokenizer_config.json (length: %zu)",
                                  chat_template_.size());
                }
            }

            // 2. 读取 bos_token
            simdjson::dom::element bos_token_val;
            if (tc_root.at_key("bos_token").get(bos_token_val) == simdjson::SUCCESS) {
                std::string_view bos_sv;
                if (bos_token_val.get_string().get(bos_sv) == simdjson::SUCCESS) {
                    bos_token_ = std::string(bos_sv);
                } else {
                    simdjson::dom::object bos_obj;
                    if (bos_token_val.get_object().get(bos_obj) == simdjson::SUCCESS) {
                        std::string_view content_sv;
                        if (bos_obj.at_key("content").get_string().get(content_sv) == simdjson::SUCCESS) {
                            bos_token_ = std::string(content_sv);
                        }
                    }
                }
            }

            // 3. 读取 eos_token
            simdjson::dom::element eos_token_val;
            if (tc_root.at_key("eos_token").get(eos_token_val) == simdjson::SUCCESS) {
                std::string_view eos_sv;
                if (eos_token_val.get_string().get(eos_sv) == simdjson::SUCCESS) {
                    eos_token_ = std::string(eos_sv);
                } else {
                    simdjson::dom::object eos_obj;
                    if (eos_token_val.get_object().get(eos_obj) == simdjson::SUCCESS) {
                        std::string_view content_sv;
                        if (eos_obj.at_key("content").get_string().get(content_sv) == simdjson::SUCCESS) {
                            eos_token_ = std::string(content_sv);
                        }
                    }
                }
            }

            // 4. 读取 added_tokens_decoder
            simdjson::dom::element decoder_val;
            if (tc_root.at_key("added_tokens_decoder").get(decoder_val) != simdjson::SUCCESS) {
                return;
            }

            simdjson::dom::object decoder_obj;
            if (decoder_val.get_object().get(decoder_obj) != simdjson::SUCCESS) {
                return;
            }

            size_t loaded_count = 0;
            for (auto field : decoder_obj) {
                std::string_view id_str = field.key;
                simdjson::dom::object token_obj;
                if (field.value.get_object().get(token_obj) != simdjson::SUCCESS) {
                    continue;
                }
                
                std::string_view content_view;
                if (token_obj.at_key("content").get_string().get(content_view) != simdjson::SUCCESS) {
                    continue;
                }
                std::string tok_str(content_view);
                
                int64_t id = std::stoll(std::string(id_str));
                
                token_to_id_[tok_str] = static_cast<int>(id);
                if (id >= static_cast<int64_t>(id_to_token_.size())) {
                    id_to_token_.resize(static_cast<size_t>(id) + 1);
                }
                id_to_token_[static_cast<int>(id)] = tok_str;

                bool is_special = false;
                simdjson::dom::element special_val;
                if (token_obj.at_key("special").get(special_val) == simdjson::SUCCESS) {
                    if (special_val.get_bool().get(is_special) != simdjson::SUCCESS) {
                        is_special = false;
                    }
                }

                if (is_special || tok_str == "<asr_text>" || tok_str == "<think>" || tok_str == "</think>") {
                    special_tokens_[tok_str] = static_cast<int>(id);
                }
                loaded_count++;
            }
            AILA_LOG_INFO("[Tokenizer] Loaded %zu added tokens from tokenizer_config.json (total special tokens: %zu)",
                          loaded_count, special_tokens_.size());
        } catch (const std::exception& e) {
            AILA_LOG_WARN("[Tokenizer] Could not load tokenizer_config.json added tokens: %s", e.what());
        }
    };

    bool loaded_base_tokenizer = false;

    try {
        std::ifstream vf(vocab_path, std::ios::binary);
        std::ifstream mf(merges_path);
        if (vf.is_open() && mf.is_open()) {
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
            loaded_base_tokenizer = true;
        }
    } catch (const std::exception& e) {
        AILA_LOG_ERROR("[Tokenizer] Failed to load legacy vocab/merges: %s", e.what());
        return false;
    }

    if (!loaded_base_tokenizer) {
        try {
            std::ifstream tf(tokenizer_json_path, std::ios::binary);
            if (!tf.is_open()) {
                AILA_LOG_ERROR("[Tokenizer] Cannot open %s (legacy vocab.json/merges.txt also missing)",
                               tokenizer_json_path.c_str());
                return false;
            }

            std::string tj_str((std::istreambuf_iterator<char>(tf)),
                                std::istreambuf_iterator<char>());
            tf.close();

            simdjson::padded_string tj_padded(tj_str);
            simdjson::ondemand::parser tj_parser;
            simdjson::ondemand::document tj_doc = tj_parser.iterate(tj_padded);
            auto model = tj_doc["model"].get_object();

            std::vector<std::pair<std::string, int>> vocab_entries;
            int max_id = 0;
            for (auto field : model["vocab"].get_object()) {
                std::string_view key = field.unescaped_key();
                int64_t id = field.value().get_int64();
                vocab_entries.emplace_back(std::string(key), static_cast<int>(id));
                if (id > max_id) {
                    max_id = static_cast<int>(id);
                }
            }

            id_to_token_.resize(max_id + 1);
            for (const auto& [token, id] : vocab_entries) {
                token_to_id_[token] = id;
                if (id >= 0 && id < static_cast<int>(id_to_token_.size())) {
                    id_to_token_[id] = token;
                }
            }

            int rank = 0;
            for (auto merge_entry : model["merges"].get_array()) {
                std::string first;
                std::string second;
                int parts = 0;
                for (auto item : merge_entry.get_array()) {
                    std::string_view piece = item.get_string();
                    if (parts == 0) {
                        first = std::string(piece);
                    } else if (parts == 1) {
                        second = std::string(piece);
                    }
                    parts++;
                }
                if (parts != 2) {
                    continue;
                }

                MergeRule rule;
                rule.a = std::move(first);
                rule.b = std::move(second);
                rule.merged = rule.a + rule.b;
                merges_.push_back(rule);
                merge_rank_[rule.a + " " + rule.b] = rank;
                rank++;
            }

            AILA_LOG_INFO("[Tokenizer] Loaded vocab from tokenizer.json: %zu tokens (max_id=%d)",
                          token_to_id_.size(), max_id);
            AILA_LOG_INFO("[Tokenizer] Loaded %zu merge rules from tokenizer.json", merges_.size());
        } catch (const std::exception& e) {
            AILA_LOG_ERROR("[Tokenizer] Failed to load tokenizer.json model: %s", e.what());
            return false;
        }
    }

    load_special_tokens_from_tokenizer_json();
    load_added_tokens_from_tokenizer_config_json();

    special_token_order_.reserve(special_tokens_.size());
    for (const auto& [token, _] : special_tokens_) {
        special_token_order_.push_back(token);
    }
    std::sort(special_token_order_.begin(), special_token_order_.end(),
              [](const std::string& a, const std::string& b) {
                  if (a.size() != b.size()) return a.size() > b.size();
                  return a < b;
              });

    auto find_special = [&](const std::string& token) -> int {
        auto it = special_tokens_.find(token);
        return (it != special_tokens_.end()) ? it->second : -1;
    };

    im_start_id_ = find_special("<|im_start|>");
    im_end_id_ = find_special("<|im_end|>");
    eot_id_ = find_special("<|endoftext|>");
    eos_id_ = im_end_id_;

    if (eos_token_.empty() && eos_id_ >= 0) {
        eos_token_ = raw_token(eos_id_);
    }

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

    auto try_match_special = [&](size_t offset, std::string& matched) -> bool {
        for (const auto& token : special_token_order_) {
            if (offset + token.size() <= text.size() &&
                text.compare(offset, token.size(), token) == 0) {
                matched = token;
                return true;
            }
        }
        return false;
    };

    size_t i = 0;
    while (i < text.size()) {
        std::string special;
        if (try_match_special(i, special)) {
            result.push_back(special);
            i += special.size();
            continue;
        }

        const size_t start = i;
        const Utf8Char cur = decode_utf8_char(text, i);

        if (cur.cp == '\'' || cur.cp == 0x2019) {
            static const std::array<const char*, 6> contractions = {
                "s", "t", "re", "ve", "m", "ll"
            };
            for (const char* suffix : contractions) {
                if (is_ascii_case_insensitive_match(text, i + cur.bytes, suffix)) {
                    size_t len = cur.bytes + std::strlen(suffix);
                    result.push_back(text.substr(i, len));
                    i += len;
                    special.clear();
                    break;
                }
            }
            if (i != start) {
                continue;
            }
            if (is_ascii_case_insensitive_match(text, i + cur.bytes, "d")) {
                result.push_back(text.substr(i, cur.bytes + 1));
                i += cur.bytes + 1;
                continue;
            }
        }

        if (!is_newline_cp(cur.cp) && !is_letter_cp(cur.cp) && !is_digit_cp(cur.cp)) {
            size_t j = i + cur.bytes;
            if (j < text.size()) {
                Utf8Char next = decode_utf8_char(text, j);
                if (is_letter_or_mark_cp(next.cp)) {
                    while (j < text.size()) {
                        Utf8Char cp = decode_utf8_char(text, j);
                        if (!is_letter_or_mark_cp(cp.cp)) {
                            break;
                        }
                        j += cp.bytes;
                    }
                    result.push_back(text.substr(start, j - start));
                    i = j;
                    continue;
                }
            }
        }

        if (is_letter_or_mark_cp(cur.cp)) {
            size_t j = i + cur.bytes;
            while (j < text.size()) {
                Utf8Char cp = decode_utf8_char(text, j);
                if (!is_letter_or_mark_cp(cp.cp)) {
                    break;
                }
                j += cp.bytes;
            }
            result.push_back(text.substr(start, j - start));
            i = j;
            continue;
        }

        if (is_digit_cp(cur.cp)) {
            result.push_back(text.substr(i, cur.bytes));
            i += cur.bytes;
            continue;
        }

        if (cur.cp == ' ') {
            size_t j = i + cur.bytes;
            if (j < text.size()) {
                Utf8Char next = decode_utf8_char(text, j);
                if (is_punctuation_or_symbol_cp(next.cp) &&
                    !is_letter_or_mark_cp(next.cp) &&
                    !is_digit_cp(next.cp) &&
                    !is_whitespace_cp(next.cp)) {
                    while (j < text.size()) {
                        Utf8Char cp = decode_utf8_char(text, j);
                        if (is_whitespace_cp(cp.cp) || is_letter_or_mark_cp(cp.cp) || is_digit_cp(cp.cp)) {
                            break;
                        }
                        j += cp.bytes;
                    }
                    while (j < text.size()) {
                        Utf8Char cp = decode_utf8_char(text, j);
                        if (!is_newline_cp(cp.cp)) {
                            break;
                        }
                        j += cp.bytes;
                    }
                    result.push_back(text.substr(start, j - start));
                    i = j;
                    continue;
                }
            }
        }

        if (is_punctuation_or_symbol_cp(cur.cp) &&
            !is_letter_or_mark_cp(cur.cp) &&
            !is_digit_cp(cur.cp) &&
            !is_whitespace_cp(cur.cp)) {
            size_t j = i + cur.bytes;
            std::string dummy;
            while (j < text.size()) {
                // Stop merging if a special token (like <|im_end|>) starts here
                if (try_match_special(j, dummy)) {
                    break;
                }
                Utf8Char cp = decode_utf8_char(text, j);
                if (is_whitespace_cp(cp.cp) || is_letter_or_mark_cp(cp.cp) || is_digit_cp(cp.cp)) {
                    break;
                }
                j += cp.bytes;
            }
            while (j < text.size()) {
                Utf8Char cp = decode_utf8_char(text, j);
                if (!is_newline_cp(cp.cp)) {
                    break;
                }
                j += cp.bytes;
            }
            result.push_back(text.substr(start, j - start));
            i = j;
            continue;
        }

        if (is_whitespace_cp(cur.cp)) {
            size_t j = i;
            while (j < text.size()) {
                Utf8Char cp = decode_utf8_char(text, j);
                if (!is_whitespace_cp(cp.cp) || is_newline_cp(cp.cp)) {
                    break;
                }
                j += cp.bytes;
            }
            if (j < text.size()) {
                Utf8Char cp = decode_utf8_char(text, j);
                if (is_newline_cp(cp.cp)) {
                    while (j < text.size()) {
                        Utf8Char nl = decode_utf8_char(text, j);
                        if (!is_newline_cp(nl.cp)) {
                            break;
                        }
                        j += nl.bytes;
                    }
                    result.push_back(text.substr(start, j - start));
                    i = j;
                    continue;
                }
            }
            if (j == start) {
                j += cur.bytes;
            }
            while (j < text.size()) {
                Utf8Char cp = decode_utf8_char(text, j);
                if (!is_whitespace_cp(cp.cp) || is_newline_cp(cp.cp)) {
                    break;
                }
                j += cp.bytes;
            }
            result.push_back(text.substr(start, j - start));
            i = j;
            continue;
        }

        result.push_back(text.substr(i, cur.bytes));
        i += cur.bytes;
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
        // Keep think markers and ASR text tags visible so users and downstreams can parse them.
        if (token == "<think>" || token == "</think>" || token == "<asr_text>") {
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
    auto rtrim = [](std::string& s) {
        while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
    };

    for (size_t i = 0; i < history.messages().size(); ++i) {
        auto msg = history.messages()[i];  // copy — we may strip /no_think
        bool is_last_user = (msg.role == "user" && i == history.messages().size() - 1);
        bool no_think = false;

        if (is_last_user) {
            rtrim(msg.content);
            const std::string kNoThinkCmd = "/no_think";
            if (msg.content.size() >= kNoThinkCmd.size()) {
                size_t pos = msg.content.size() - kNoThinkCmd.size();
                if (msg.content.compare(pos, kNoThinkCmd.size(), kNoThinkCmd) == 0) {
                    bool boundary_ok = (pos == 0) ||
                        std::isspace(static_cast<unsigned char>(msg.content[pos - 1]));
                    if (boundary_ok) {
                        no_think = true;
                        msg.content.erase(pos);
                        rtrim(msg.content);
                    }
                }
            }
        }

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

            if (no_think) {
                auto end_think_it = special_tokens_.find("</think>");
                if (end_think_it != special_tokens_.end()) {
                    auto nl_tokens = encode("\n");
                    ids.push_back(end_think_it->second);
                    ids.insert(ids.end(), nl_tokens.begin(), nl_tokens.end());
                    auto direct_tokens = encode("Please answer directly and briefly.\n");
                    ids.insert(ids.end(), direct_tokens.begin(), direct_tokens.end());
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
