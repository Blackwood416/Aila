#pragma once

#include "CpuSafetensorsStore.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

enum class CpuBnb4CacheMode {
    Fp16,
    PackedNf4,
};

CpuBnb4CacheMode parse_cpu_bnb4_cache_mode(std::string_view value);

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
    CpuBnb4CacheMode cache_mode = CpuBnb4CacheMode::Fp16;
    std::vector<float> decoded_absmax;
    std::vector<uint16_t> dense_weight_f16;
    std::vector<uint8_t> packed_nf4_codes;
    std::vector<float> packed_nf4_absmax;

    bool valid() const;
    int64_t out_features() const;
    int64_t in_features() const;
    int64_t logical_numel() const;
    int64_t packed_num_bytes() const;
    size_t cache_bytes() const;
};

bool parse_cpu_bnb4_quant_state_json(const std::string& json_text,
                                     CpuBnb4QuantState& out,
                                     std::string* error);

bool load_cpu_bnb4_weight_ref(const CpuSafetensorsStore& store,
                              const std::string& name,
                              CpuBnb4WeightRef& out,
                              std::string* error,
                              CpuBnb4CacheMode cache_mode = CpuBnb4CacheMode::Fp16);

uint16_t cpu_float_to_f16(float value);
float cpu_f16_to_float(uint16_t value);
void cpu_f16_to_f32(const uint16_t* input, float* output, int64_t count);
float cpu_f16_dot_f32(const uint16_t* weights, const float* input, int64_t count);
void cpu_nf4_matvec_scalar(const CpuBnb4WeightRef& weight,
                           const float* quant_map,
                           const float* input,
                           float* output);
void cpu_nf4_matvec(const CpuBnb4WeightRef& weight,
                    const float* quant_map,
                    const float* input,
                    float* output);

void cpu_q35_parallel_rows(int64_t rows,
                           int64_t cols,
                           int64_t min_rows_per_thread,
                           const std::function<void(int64_t, int64_t)>& fn);

void cpu_bnb4_matvec(const CpuBnb4WeightRef& weight,
                     const float* input,
                     float* output);
