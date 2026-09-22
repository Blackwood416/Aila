#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace aila_asr {

std::vector<uint32_t> utf8_to_utf32(const std::string& text);

std::string utf32_to_utf8(const std::vector<uint32_t>& u32);

std::vector<uint32_t> fix_char_repeats(const std::vector<uint32_t>& s, int thresh);

std::vector<uint32_t> fix_pattern_repeats(const std::vector<uint32_t>& s, int thresh, int max_len = 20);

std::string detect_and_fix_repetitions(const std::string& text, int threshold = 20);

std::string normalize_language_name(const std::string& language);

std::string to_lowercase(const std::string& str);

void parse_asr_output(const std::string& raw, const std::string& user_language, std::string& language_out, std::string& text_out);

} // namespace aila_asr
