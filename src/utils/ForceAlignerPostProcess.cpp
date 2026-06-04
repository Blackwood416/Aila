#include "ForceAlignerPostProcess.hpp"
#include <algorithm>
#include <cctype>
#include <sstream>

namespace aila {

static bool is_cjk_char(uint32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF)    // CJK Unified Ideographs
        || (cp >= 0x3400 && cp <= 0x4DBF)     // Extension A
        || (cp >= 0x20000 && cp <= 0x2A6DF)   // Extension B
        || (cp >= 0xF900 && cp <= 0xFAFF);    // Compatibility Ideographs
}

static bool is_kept_char(uint32_t cp) {
    if (cp == '\'') return true;
    // ASCII letters and digits
    if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z') || (cp >= '0' && cp <= '9'))
        return true;
    // Latin-1 Supplement letters (e.g., é, ü, ñ)
    if ((cp >= 0xC0 && cp <= 0xD6) || (cp >= 0xD8 && cp <= 0xF6) || (cp >= 0xF8 && cp <= 0xFF))
        return true;
    // Latin Extended, IPA, spacing modifiers (U+0100–U+02FF)
    if (cp >= 0x0100 && cp <= 0x02FF) return true;
    // Greek & Coptic (U+0370–U+03FF)
    if (cp >= 0x0370 && cp <= 0x03FF) return true;
    // Cyrillic (U+0400–U+04FF)
    if (cp >= 0x0400 && cp <= 0x04FF) return true;
    // Arabic, Hebrew, Thai, etc. — broad Letter/Number blocks
    if (cp >= 0x0600 && cp <= 0x06FF) return true;  // Arabic
    if (cp >= 0x0E00 && cp <= 0x0E7F) return true;  // Thai
    // Hiragana (U+3040–U+309F)
    if (cp >= 0x3040 && cp <= 0x309F) return true;
    // Katakana (U+30A0–U+30FF)
    if (cp >= 0x30A0 && cp <= 0x30FF) return true;
    // Hangul Jamo (U+1100–U+11FF)
    if (cp >= 0x1100 && cp <= 0x11FF) return true;
    // Hangul Compatibility Jamo (U+3130–U+318F)
    if (cp >= 0x3130 && cp <= 0x318F) return true;
    // Hangul Syllables (U+AC00–U+D7AF)
    if (cp >= 0xAC00 && cp <= 0xD7AF) return true;
    // Fullwidth Latin letters/digits (U+FF01–U+FF5E)
    if (cp >= 0xFF21 && cp <= 0xFF3A) return true;  // A-Z fullwidth
    if (cp >= 0xFF41 && cp <= 0xFF5A) return true;  // a-z fullwidth
    if (cp >= 0xFF10 && cp <= 0xFF19) return true;  // 0-9 fullwidth
    // CJK Radicals Supplement, Kangxi (catch-all for other CJK-adjacent)
    if (cp >= 0x2E80 && cp <= 0x2FDF) return true;
    if (cp >= 0x31C0 && cp <= 0x31EF) return true;  // CJK Strokes
    return false;
}

static std::vector<uint32_t> utf8_to_codepoints(const std::string& text) {
    std::vector<uint32_t> out;
    size_t i = 0;
    while (i < text.size()) {
        unsigned char c0 = static_cast<unsigned char>(text[i]);
        uint32_t cp = c0;
        size_t bytes = 1;
        if ((c0 & 0xE0) == 0xC0 && i + 1 < text.size()) { cp = ((c0 & 0x1F) << 6) | (text[i+1] & 0x3F); bytes = 2; }
        else if ((c0 & 0xF0) == 0xE0 && i + 2 < text.size()) { cp = ((c0 & 0x0F) << 12) | ((text[i+1] & 0x3F) << 6) | (text[i+2] & 0x3F); bytes = 3; }
        else if ((c0 & 0xF8) == 0xF0 && i + 3 < text.size()) { cp = ((c0 & 0x07) << 18) | ((text[i+1] & 0x3F) << 12) | ((text[i+2] & 0x3F) << 6) | (text[i+3] & 0x3F); bytes = 4; }
        out.push_back(cp);
        i += bytes;
    }
    return out;
}

static std::string codepoint_to_utf8(uint32_t cp) {
    std::string utf8_char;
    if (cp < 0x80) utf8_char = static_cast<char>(cp);
    else if (cp < 0x800) { utf8_char += static_cast<char>(0xC0 | (cp >> 6)); utf8_char += static_cast<char>(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) { utf8_char += static_cast<char>(0xE0 | (cp >> 12)); utf8_char += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); utf8_char += static_cast<char>(0x80 | (cp & 0x3F)); }
    else { utf8_char += static_cast<char>(0xF0 | (cp >> 18)); utf8_char += static_cast<char>(0x80 | ((cp >> 12) & 0x3F)); utf8_char += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); utf8_char += static_cast<char>(0x80 | (cp & 0x3F)); }
    return utf8_char;
}

static std::vector<std::string> tokenize_cjk(const std::string& text) {
    // CJK: one character per token, collapse non-CJK alphanumeric runs into words
    std::vector<std::string> tokens;
    std::vector<uint32_t> cps = utf8_to_codepoints(text);
    std::string latin_buf;

    auto flush_latin = [&]() {
        if (!latin_buf.empty()) {
            tokens.push_back(latin_buf);
            latin_buf.clear();
        }
    };

    for (size_t i = 0; i < cps.size(); ++i) {
        uint32_t cp = cps[i];
        if (is_cjk_char(cp)) {
            flush_latin();
            tokens.push_back(codepoint_to_utf8(cp));
        } else {
            if (is_kept_char(cp)) {
                latin_buf += codepoint_to_utf8(cp);
            } else {
                flush_latin();
            }
        }
    }
    flush_latin();
    return tokens;
}

static std::vector<std::string> tokenize_space(const std::string& text) {
    // Space-delimited: split on whitespace, also split CJK within each word
    std::vector<std::string> tokens;
    std::istringstream iss(text);
    std::string word;
    while (iss >> word) {
        auto sub = tokenize_cjk(word);
        tokens.insert(tokens.end(), sub.begin(), sub.end());
    }
    return tokens;
}

std::pair<std::vector<std::string>, std::string>
ForceAlignerPostProcess::encode_input(const std::string& text,
                                      const std::string& language) {
    std::vector<std::string> word_list;
    std::string lang_lower = language;
    std::transform(lang_lower.begin(), lang_lower.end(), lang_lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    // Naive v1: all CJK-relevant languages use char-by-char; everything else space-delimited.
    // Japanese (nagisa) and Korean (soynlp) deferred to v2.
    bool is_cjk_lang = (lang_lower == "chinese" || lang_lower == "cantonese" ||
                        lang_lower == "japanese" || lang_lower == "korean");

    if (is_cjk_lang) {
        word_list = tokenize_cjk(text);
    } else {
        word_list = tokenize_space(text);
    }

    // Return word_list + plain text (no timestamp markers — those are added as
    // explicit token IDs in Engine to ensure correct tokenization).
    std::string prompt;
    for (size_t i = 0; i < word_list.size(); ++i) {
        prompt += word_list[i];
    }

    return {word_list, prompt};
}

std::vector<int> ForceAlignerPostProcess::extract_timestamps(
    const float* logits, int64_t seq_len, int classify_num,
    const int* input_ids, int input_len,
    int timestamp_token_id, int timestamp_segment_time) {

    std::vector<int> timestamps;
    for (int i = 0; i < input_len; ++i) {
        if (input_ids[i] == timestamp_token_id) {
            const float* row = logits + static_cast<int64_t>(i) * classify_num;
            int best_class = 0;
            float best_val = row[0];
            for (int c = 1; c < classify_num; ++c) {
                if (row[c] > best_val) {
                    best_val = row[c];
                    best_class = c;
                }
            }
            timestamps.push_back(best_class * timestamp_segment_time);
        }
    }
    return timestamps;
}

std::vector<int> ForceAlignerPostProcess::fix_timestamp(const std::vector<int>& data) {
    int n = static_cast<int>(data.size());
    if (n == 0) return {};

    // Longest Increasing Subsequence to find "normal" indices
    std::vector<int> dp(n, 1);
    std::vector<int> parent(n, -1);
    for (int i = 1; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            if (data[j] <= data[i] && dp[j] + 1 > dp[i]) {
                dp[i] = dp[j] + 1;
                parent[i] = j;
            }
        }
    }

    int max_len = 0, max_idx = 0;
    for (int i = 0; i < n; ++i) {
        if (dp[i] > max_len) { max_len = dp[i]; max_idx = i; }
    }

    std::vector<bool> is_normal(n, false);
    {
        int idx = max_idx;
        while (idx != -1) {
            is_normal[idx] = true;
            idx = parent[idx];
        }
    }

    std::vector<int> result = data;
    int i = 0;
    while (i < n) {
        if (!is_normal[i]) {
            int j = i;
            while (j < n && !is_normal[j]) ++j;
            int anomaly_count = j - i;

            int left_val = -1;
            for (int k = i - 1; k >= 0; --k) {
                if (is_normal[k]) { left_val = result[k]; break; }
            }
            int right_val = -1;
            for (int k = j; k < n; ++k) {
                if (is_normal[k]) { right_val = result[k]; break; }
            }

            if (anomaly_count <= 2) {
                for (int k = i; k < j; ++k) {
                    if (left_val < 0) result[k] = right_val;
                    else if (right_val < 0) result[k] = left_val;
                    else result[k] = ((k - (i - 1)) <= (j - k)) ? left_val : right_val;
                }
            } else {
                if (left_val >= 0 && right_val >= 0) {
                    double step = static_cast<double>(right_val - left_val) / (anomaly_count + 1);
                    for (int k = i; k < j; ++k) {
                        result[k] = left_val + static_cast<int>(step * (k - i + 1));
                    }
                } else if (left_val >= 0) {
                    for (int k = i; k < j; ++k) result[k] = left_val;
                } else if (right_val >= 0) {
                    for (int k = i; k < j; ++k) result[k] = right_val;
                }
            }
            i = j;
        } else {
            ++i;
        }
    }
    return result;
}

std::vector<AlignedWord> ForceAlignerPostProcess::build_output(
    const std::vector<std::string>& words,
    const std::vector<int>& timestamps) {

    std::vector<AlignedWord> output;
    size_t n_words = words.size();
    for (size_t i = 0; i < n_words && (i * 2 + 1) < timestamps.size(); ++i) {
        output.push_back({words[i], timestamps[i * 2], timestamps[i * 2 + 1]});
    }
    return output;
}

} // namespace aila
