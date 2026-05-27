#include "Base64.hpp"
#include <algorithm>

namespace aila {
namespace utils {

bool is_data_uri(const std::string& str) {
    return str.rfind("data:", 0) == 0;
}

bool parse_data_uri(const std::string& uri, std::string& out_format, std::vector<uint8_t>& out_data) {
    if (!is_data_uri(uri)) return false;

    // Format: data:[mime];base64,[payload]
    size_t base64_pos = uri.find(";base64,");
    if (base64_pos == std::string::npos) return false;

    std::string mime = uri.substr(5, base64_pos - 5);
    std::string_view payload = std::string_view(uri).substr(base64_pos + 8);

    // Map MIME to extension format
    std::string fmt = "bin";
    if (mime == "image/png") fmt = "png";
    else if (mime == "image/jpeg" || mime == "image/jpg") fmt = "jpg";
    else if (mime == "image/webp") fmt = "webp";
    else if (mime == "audio/wav" || mime == "audio/x-wav") fmt = "wav";
    else if (mime == "audio/mp3" || mime == "audio/mpeg") fmt = "mp3";
    else if (mime == "audio/flac" || mime == "audio/x-flac") fmt = "flac";
    
    out_format = fmt;
    out_data = decode_base64(payload);
    return !out_data.empty();
}

std::vector<uint8_t> decode_base64(const std::string_view& base64_str) {
    std::vector<uint8_t> out;
    out.reserve(base64_str.size() * 3 / 4);

    int val = 0;
    int valb = -8;
    for (char c : base64_str) {
        if (c == '=') break;
        
        int d = -1;
        if (c >= 'A' && c <= 'Z') d = c - 'A';
        else if (c >= 'a' && c <= 'z') d = c - 'a' + 26;
        else if (c >= '0' && c <= '9') d = c - '0' + 52;
        else if (c == '+') d = 62;
        else if (c == '/') d = 63;
        
        if (d != -1) {
            val = (val << 6) | d;
            valb += 6;
            if (valb >= 0) {
                out.push_back(static_cast<uint8_t>((val >> valb) & 0xFF));
                valb -= 8;
            }
        }
    }
    return out;
}

} // namespace utils
} // namespace aila
