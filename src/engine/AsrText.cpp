#include "engine/AsrText.hpp"

#include <algorithm>
#include <cctype>

namespace aila_asr {

std::vector<uint32_t> utf8_to_utf32(const std::string& text) {
    std::vector<uint32_t> out;
    size_t offset = 0;
    while (offset < text.size()) {
        unsigned char c0 = static_cast<unsigned char>(text[offset]);
        uint32_t cp = c0;
        size_t bytes = 1;
        if (c0 >= 0x80) {
            auto cont = [&](size_t idx) -> uint32_t {
                if (offset + idx < text.size()) {
                    return static_cast<unsigned char>(text[offset + idx]) & 0x3Fu;
                }
                return 0;
            };
            if ((c0 & 0xE0) == 0xC0 && offset + 1 < text.size()) {
                cp = ((c0 & 0x1Fu) << 6) | cont(1);
                bytes = 2;
            } else if ((c0 & 0xF0) == 0xE0 && offset + 2 < text.size()) {
                cp = ((c0 & 0x0Fu) << 12) | (cont(1) << 6) | cont(2);
                bytes = 3;
            } else if ((c0 & 0xF8) == 0xF0 && offset + 3 < text.size()) {
                cp = ((c0 & 0x07u) << 18) | (cont(1) << 12) | (cont(2) << 6) | cont(3);
                bytes = 4;
            }
        }
        out.push_back(cp);
        offset += bytes;
    }
    return out;
}

std::string utf32_to_utf8(const std::vector<uint32_t>& u32) {
    std::string out;
    for (uint32_t cp : u32) {
        if (cp < 0x80) {
            out += static_cast<char>(cp);
        } else if (cp < 0x800) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (cp >> 18));
            out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }
    return out;
}

std::vector<uint32_t> fix_char_repeats(const std::vector<uint32_t>& s, int thresh) {
    std::vector<uint32_t> res;
    int i = 0;
    int n = static_cast<int>(s.size());
    while (i < n) {
        int count = 1;
        while (i + count < n && s[i + count] == s[i]) {
            count++;
        }
        if (count > thresh) {
            res.push_back(s[i]);
        } else {
            for (int k = 0; k < count; ++k) {
                res.push_back(s[i + k]);
            }
        }
        i += count;
    }
    return res;
}

std::vector<uint32_t> fix_pattern_repeats(const std::vector<uint32_t>& s, int thresh, int max_len) {
    int n = static_cast<int>(s.size());
    int min_repeat_chars = thresh * 2;
    if (n < min_repeat_chars) {
        return s;
    }

    int i = 0;
    std::vector<uint32_t> result;
    bool found = false;

    while (i <= n - min_repeat_chars) {
        found = false;
        for (int k = 1; k <= max_len; ++k) {
            if (i + k * thresh > n) {
                break;
            }

            bool valid = true;
            for (int rep = 1; rep < thresh; ++rep) {
                int start_idx = i + rep * k;
                for (int p_idx = 0; p_idx < k; ++p_idx) {
                    if (s[start_idx + p_idx] != s[i + p_idx]) {
                        valid = false;
                        break;
                    }
                }
                if (!valid) break;
            }

            if (valid) {
                int total_rep = thresh;
                int end_index = i + thresh * k;
                while (end_index + k <= n) {
                    bool match = true;
                    for (int p_idx = 0; p_idx < k; ++p_idx) {
                        if (s[end_index + p_idx] != s[i + p_idx]) {
                            match = false;
                            break;
                        }
                    }
                    if (!match) break;
                    total_rep++;
                    end_index += k;
                }

                for (int p_idx = 0; p_idx < k; ++p_idx) {
                    result.push_back(s[i + p_idx]);
                }

                std::vector<uint32_t> rest(s.begin() + end_index, s.end());
                std::vector<uint32_t> rest_fixed = fix_pattern_repeats(rest, thresh, max_len);
                result.insert(result.end(), rest_fixed.begin(), rest_fixed.end());

                i = n;
                found = true;
                break;
            }
        }

        if (found) {
            break;
        } else {
            result.push_back(s[i]);
            i++;
        }
    }

    if (!found) {
        for (int idx = i; idx < n; ++idx) {
            result.push_back(s[idx]);
        }
    }

    return result;
}

std::string detect_and_fix_repetitions(const std::string& text, int threshold) {
    std::vector<uint32_t> u32 = utf8_to_utf32(text);
    std::vector<uint32_t> fixed_char = fix_char_repeats(u32, threshold);
    std::vector<uint32_t> fixed_pattern = fix_pattern_repeats(fixed_char, threshold);
    return utf32_to_utf8(fixed_pattern);
}

std::string normalize_language_name(const std::string& language) {
    if (language.empty()) return "";
    std::string s = language;
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(0, 1);
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
    if (s.empty()) return "";

    s[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(s[0])));
    for (size_t i = 1; i < s.size(); ++i) {
        s[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(s[i])));
    }
    return s;
}

std::string to_lowercase(const std::string& str) {
    std::string s = str;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return std::tolower(c);
    });
    return s;
}

void parse_asr_output(const std::string& raw, const std::string& user_language, std::string& language_out, std::string& text_out) {
    language_out = "";
    text_out = "";
    if (raw.empty()) return;

    std::string s = raw;
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(0, 1);
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
    if (s.empty()) return;

    s = detect_and_fix_repetitions(s);

    if (!user_language.empty()) {
        language_out = normalize_language_name(user_language);
        text_out = s;
        return;
    }

    const std::string tag = "<asr_text>";
    size_t tag_pos = s.find(tag);

    std::string meta_part;
    std::string text_part;
    if (tag_pos != std::string::npos) {
        meta_part = s.substr(0, tag_pos);
        text_part = s.substr(tag_pos + tag.length());
    } else {
        text_out = s;
        while (!text_out.empty() && std::isspace(static_cast<unsigned char>(text_out.front()))) text_out.erase(0, 1);
        while (!text_out.empty() && std::isspace(static_cast<unsigned char>(text_out.back()))) text_out.pop_back();
        return;
    }

    std::string meta_lower = to_lowercase(meta_part);

    if (meta_lower.find("language none") != std::string::npos) {
        std::string t = text_part;
        while (!t.empty() && std::isspace(static_cast<unsigned char>(t.front()))) t.erase(0, 1);
        while (!t.empty() && std::isspace(static_cast<unsigned char>(t.back()))) t.pop_back();
        if (t.empty()) {
            return;
        }
        text_out = t;
        return;
    }

    std::string lang = "";
    std::string prefix = "language ";
    size_t line_start = 0;
    while (line_start < meta_part.size()) {
        size_t line_end = meta_part.find('\n', line_start);
        if (line_end == std::string::npos) {
            line_end = meta_part.size();
        }
        std::string line = meta_part.substr(line_start, line_end - line_start);
        line_start = line_end + 1;

        while (!line.empty() && std::isspace(static_cast<unsigned char>(line.front()))) line.erase(0, 1);
        while (!line.empty() && std::isspace(static_cast<unsigned char>(line.back()))) line.pop_back();
        if (line.empty()) continue;

        std::string low = to_lowercase(line);
        if (low.rfind(prefix, 0) == 0) {
            std::string val = line.substr(prefix.length());
            while (!val.empty() && std::isspace(static_cast<unsigned char>(val.front()))) val.erase(0, 1);
            while (!val.empty() && std::isspace(static_cast<unsigned char>(val.back()))) val.pop_back();
            if (!val.empty()) {
                lang = normalize_language_name(val);
            }
            break;
        }
    }

    while (!text_part.empty() && std::isspace(static_cast<unsigned char>(text_part.front()))) text_part.erase(0, 1);
    while (!text_part.empty() && std::isspace(static_cast<unsigned char>(text_part.back()))) text_part.pop_back();

    language_out = lang;
    text_out = text_part;
}

} // namespace aila_asr
