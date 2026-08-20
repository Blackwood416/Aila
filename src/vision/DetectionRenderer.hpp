#pragma once

#include "engine/Types.hpp"

#include <string>
#include <vector>

namespace aila::vision {

bool save_detection_png(const std::string& input_path, const std::string& output_path,
                        const std::vector<Detection>& detections,
                        std::string* error_message = nullptr);

} // namespace aila::vision
