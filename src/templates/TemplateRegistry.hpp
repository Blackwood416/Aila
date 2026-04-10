#pragma once

#include "engine/Types.hpp"
#include "../utils/Tokenizer.hpp"
#include <string>
#include <vector>

namespace aila {
namespace templating {

class TemplateRegistry {
public:
    bool render(ModelFamily family,
                const Tokenizer& tokenizer,
                const std::vector<Message>& messages,
                bool vision_enabled,
                bool add_generation_prompt,
                std::vector<int>& out_ids,
                std::string* error_message = nullptr) const;
};

} // namespace templating
} // namespace aila

