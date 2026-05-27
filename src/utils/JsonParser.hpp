#pragma once
#include "engine/Types.hpp"
#include <string>
#include <vector>

namespace aila {
namespace utils {

bool parse_messages_json(const std::string& messages_json,
                         std::vector<Message>& out_messages,
                         GenerationConfig& out_config,
                         std::string* error_message = nullptr);

} // namespace utils
} // namespace aila
