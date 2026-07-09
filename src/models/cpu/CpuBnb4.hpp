#pragma once

#include "CpuSafetensorsStore.hpp"

#include <cstdint>
#include <string>
#include <vector>

struct CpuBnb4QuantState {
    std::string quant_type;
    std::string dtype;
    std::vector<int64_t> shape;
    int blocksize = 0;
    bool nested = false;
    int nested_blocksize = 0;
    std::string nested_dtype;
    float nested_offset = 0.0f;
};

struct CpuBnb4WeightRef {
    std::string name;
    const CpuTensorView* packed_weight = nullptr;
    const CpuTensorView* absmax = nullptr;
    const CpuTensorView* quant_map = nullptr;
    const CpuTensorView* nested_absmax = nullptr;
    const CpuTensorView* nested_quant_map = nullptr;
    const CpuTensorView* packed_quant_state = nullptr;
    CpuBnb4QuantState quant_state{};
    std::vector<float> decoded_absmax;
    std::vector<float> dense_weight;

    bool valid() const;
    int64_t out_features() const;
    int64_t in_features() const;
    int64_t logical_numel() const;
    int64_t packed_num_bytes() const;
};

bool parse_cpu_bnb4_quant_state_json(const std::string& json_text,
                                     CpuBnb4QuantState& out,
                                     std::string* error);

bool load_cpu_bnb4_weight_ref(const CpuSafetensorsStore& store,
                              const std::string& name,
                              CpuBnb4WeightRef& out,
                              std::string* error);

void cpu_bnb4_matvec(const CpuBnb4WeightRef& weight,
                     const float* input,
                     float* output);
