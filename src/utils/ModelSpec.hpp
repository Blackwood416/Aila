#pragma once

#include "engine/Types.hpp"
#include <string>

namespace aila {
namespace modelspec {

// Parse model config.json into a unified ModelSpec.
// Returns true on success, false on parse/IO failure.
bool load_from_dir(const std::string& model_dir,
                   ModelSpec& spec,
                   std::string* error_message = nullptr);

} // namespace modelspec
} // namespace aila

