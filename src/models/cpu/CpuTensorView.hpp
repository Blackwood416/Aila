#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

enum class CpuDataType {
    F16,
    BF16,
    F32,
    F64,
    U8,
    S8,
    Unknown
};

struct CpuTensorView {
    std::string name;
    CpuDataType dtype = CpuDataType::Unknown;
    std::vector<int64_t> shape;
    const uint8_t* data = nullptr;
    size_t bytes = 0;

    const float* f32_data() const {
        return reinterpret_cast<const float*>(data);
    }

    const uint8_t* u8_data() const {
        return data;
    }
};
