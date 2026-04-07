#pragma once

#include "engine/Types.hpp"
#include <string>

namespace aila {
namespace modelcfg {

// Load Qwen-style config.json from model directory into Qwen3Config.
// Returns true on success, false if missing/invalid.
bool load_qwen3_config_from_dir(const std::string& model_dir,
                                Qwen3Config& cfg,
                                std::string* error_message = nullptr);

} // namespace modelcfg
} // namespace aila

