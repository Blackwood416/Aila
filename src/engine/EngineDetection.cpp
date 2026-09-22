#include "engine/Engine.hpp"

std::vector<Detection> InferenceEngine::detect_file(
            const std::string& path,
            const DetectionConfig& detection_config) {
        clear_error();
        std::vector<Detection> detections;
        if (!detector_) {
            set_error(EngineErrorCode::ModelCapability,
                      "Object detection requires a YOLO26 model");
            return detections;
        }
        std::string error;
        try {
            if (detector_->detect_file(path, detection_config, detections, &error)) return detections;
            set_error(EngineErrorCode::InvalidArgument, error);
            detections.clear();
        } catch (const std::exception& exception) {
            set_error(EngineErrorCode::RuntimeError, exception.what());
            detections.clear();
        }
        return detections;
    }

std::vector<Detection> InferenceEngine::detect_encoded(
            const uint8_t* data, size_t size,
            const DetectionConfig& detection_config) {
        clear_error();
        std::vector<Detection> detections;
        if (!detector_) {
            set_error(EngineErrorCode::ModelCapability,
                      "Object detection requires a YOLO26 model");
            return detections;
        }
        std::string error;
        try {
            if (detector_->detect_encoded(data, size, detection_config, detections, &error)) return detections;
            set_error(EngineErrorCode::InvalidArgument, error);
            detections.clear();
        } catch (const std::exception& exception) {
            set_error(EngineErrorCode::RuntimeError, exception.what());
            detections.clear();
        }
        return detections;
    }

std::vector<Detection> InferenceEngine::detect_pixels(
            const ImageView& image,
            const DetectionConfig& detection_config) {
        clear_error();
        std::vector<Detection> detections;
        if (!detector_) {
            set_error(EngineErrorCode::ModelCapability,
                      "Object detection requires a YOLO26 model");
            return detections;
        }
        std::string error;
        try {
            if (detector_->detect_pixels(image, detection_config, detections, &error)) return detections;
            set_error(EngineErrorCode::InvalidArgument, error);
            detections.clear();
        } catch (const std::exception& exception) {
            set_error(EngineErrorCode::RuntimeError, exception.what());
            detections.clear();
        }
        return detections;
    }

int InferenceEngine::last_detect_image_width() const {
        return detector_ ? detector_->last_image_width() : 0;
    }

int InferenceEngine::last_detect_image_height() const {
        return detector_ ? detector_->last_image_height() : 0;
    }

double InferenceEngine::last_detect_preprocess_ms() const {
        return detector_ ? detector_->last_preprocess_ms() : 0.0;
    }

double InferenceEngine::last_detect_device_wall_ms() const {
        return detector_ ? detector_->last_device_wall_ms() : 0.0;
    }

double InferenceEngine::last_detect_postprocess_ms() const {
        return detector_ ? detector_->last_postprocess_ms() : 0.0;
    }

size_t InferenceEngine::last_detect_peak_device_bytes() const {
        return detector_ && ctx_ ? ctx_->peak_allocated_bytes() : 0;
    }
