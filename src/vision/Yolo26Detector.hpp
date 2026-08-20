#pragma once

#include "engine/Types.hpp"
#include "core/Context.hpp"
#include "utils/SafeTensors.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace aila::vision {

class Yolo26Detector {
public:
    Yolo26Detector();
    ~Yolo26Detector();
    Yolo26Detector(const Yolo26Detector&) = delete;
    Yolo26Detector& operator=(const Yolo26Detector&) = delete;

    bool init(Context& ctx, ModelWeights& weights, const Yolo26Config& config,
              std::string* error_message = nullptr);

    bool detect_file(const std::string& path, const DetectionConfig& config,
                     std::vector<Detection>& detections,
                     std::string* error_message = nullptr);
    bool detect_encoded(const uint8_t* data, size_t size, const DetectionConfig& config,
                        std::vector<Detection>& detections,
                        std::string* error_message = nullptr);
    bool detect_pixels(const ImageView& image, const DetectionConfig& config,
                       std::vector<Detection>& detections,
                       std::string* error_message = nullptr);

    int last_image_width() const;
    int last_image_height() const;
    double last_preprocess_ms() const;
    double last_device_wall_ms() const;
    double last_postprocess_ms() const;
    const Yolo26Config& config() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace aila::vision
