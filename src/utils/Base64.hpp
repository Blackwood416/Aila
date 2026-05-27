#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <string_view>

namespace aila {
namespace utils {

// 判断给定的字符串是否是 data: 格式的 Data URI
bool is_data_uri(const std::string& str);

// 解析 Data URI，例如 data:image/png;base64,iVBORw0KGg...
// 提取出对应的媒体后缀格式（如 "png"），并解码 base64 得到二进制字节流
bool parse_data_uri(const std::string& uri, std::string& out_format, std::vector<uint8_t>& out_data);

// 基本的 Base64 字符串解码为字节流函数（自动过滤换行等空白符，支持 padding '=' 兼容）
std::vector<uint8_t> decode_base64(const std::string_view& base64_str);

} // namespace utils
} // namespace aila
