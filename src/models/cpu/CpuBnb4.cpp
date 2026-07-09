#include "models/cpu/CpuBnb4.hpp"

#include "simdjson.h"
#include "utils/EnvUtils.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace {

void set_error(std::string* error, const std::string& message) {
    if (error) {
        *error = message;
    }
}

int64_t tensor_numel(const CpuTensorView* tensor) {
    if (!tensor) {
        return 0;
    }
    int64_t result = 1;
    for (int64_t dim : tensor->shape) {
        result *= dim;
    }
    return result;
}

int64_t ceil_div_i64(int64_t value, int64_t divisor) {
    return (value + divisor - 1) / divisor;
}

bool read_json_int64(simdjson::dom::element root, const char* key, int64_t& out) {
    simdjson::dom::element elem;
    if (root.at_key(key).get(elem) != simdjson::SUCCESS) {
        return false;
    }
    return elem.get_int64().get(out) == simdjson::SUCCESS;
}

bool read_json_float(simdjson::dom::element root, const char* key, float& out) {
    simdjson::dom::element elem;
    if (root.at_key(key).get(elem) != simdjson::SUCCESS) {
        return false;
    }

    double value = 0.0;
    if (elem.get_double().get(value) == simdjson::SUCCESS) {
        out = static_cast<float>(value);
        return true;
    }

    int64_t int_value = 0;
    if (elem.get_int64().get(int_value) == simdjson::SUCCESS) {
        out = static_cast<float>(int_value);
        return true;
    }
    return false;
}

std::string read_json_string(simdjson::dom::element root, const char* key) {
    simdjson::dom::element elem;
    if (root.at_key(key).get(elem) != simdjson::SUCCESS) {
        return "";
    }
    std::string_view value;
    if (elem.get_string().get(value) != simdjson::SUCCESS) {
        return "";
    }
    return std::string(value);
}

float tensor_f32_at(const CpuTensorView& tensor, int64_t index) {
    return tensor.f32_data()[static_cast<size_t>(index)];
}

uint8_t tensor_u8_at(const CpuTensorView& tensor, int64_t index) {
    return tensor.u8_data()[static_cast<size_t>(index)];
}

float effective_absmax_for_block(const CpuBnb4WeightRef& weight, int64_t block) {
    if (!weight.decoded_absmax.empty()) {
        return weight.decoded_absmax[static_cast<size_t>(block)];
    }
    if (!weight.quant_state.nested) {
        return tensor_f32_at(*weight.absmax, block);
    }

    const int64_t nested_block = block / weight.quant_state.nested_blocksize;
    const uint8_t qv = tensor_u8_at(*weight.absmax, block);
    return tensor_f32_at(*weight.nested_quant_map, qv) *
               tensor_f32_at(*weight.nested_absmax, nested_block) +
           weight.quant_state.nested_offset;
}

template <typename Fn>
void parallel_rows(int64_t rows,
                   int64_t cols,
                   int64_t min_rows_per_thread,
                   Fn&& fn) {
    const int env_threads = aila::env::read_int_raw("AILA_CPU_Q35_THREADS", 2);
    const unsigned hw_threads = std::max(
        1u,
        static_cast<unsigned>(env_threads > 0
            ? env_threads
            : static_cast<int>(std::thread::hardware_concurrency())));
    const int64_t desired_threads =
        (rows >= min_rows_per_thread * 2 && cols >= 256)
            ? std::min<int64_t>(static_cast<int64_t>(hw_threads),
                                std::max<int64_t>(1, rows / min_rows_per_thread))
            : 1;
    if (desired_threads <= 1) {
        fn(0, rows);
        return;
    }

    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(desired_threads - 1));
    const int64_t rows_per_thread = (rows + desired_threads - 1) / desired_threads;
    int64_t row_begin = 0;
    for (int64_t t = 1; t < desired_threads; ++t) {
        const int64_t row_end = std::min(rows, row_begin + rows_per_thread);
        workers.emplace_back(fn, row_begin, row_end);
        row_begin = row_end;
    }
    fn(row_begin, rows);
    for (std::thread& worker : workers) {
        worker.join();
    }
}

void dequantize_dense_weight(CpuBnb4WeightRef& weight) {
    const int64_t rows = weight.out_features();
    const int64_t cols = weight.in_features();
    const int64_t total = weight.logical_numel();
    const uint8_t* packed = weight.packed_weight->u8_data();
    const float* quant_map = weight.quant_map->f32_data();
    const int blocksize = weight.quant_state.blocksize;

    weight.dense_weight.assign(static_cast<size_t>(total), 0.0f);
    parallel_rows(rows, cols, 64, [&](int64_t row_begin, int64_t row_end) {
        for (int64_t row = row_begin; row < row_end; ++row) {
            float* dst = weight.dense_weight.data() + static_cast<size_t>(row) * cols;
            for (int64_t col = 0; col < cols; ++col) {
                const int64_t flat = row * cols + col;
                const uint8_t byte = packed[static_cast<size_t>(flat / 2)];
                const uint8_t code =
                    (flat % 2 == 0) ? static_cast<uint8_t>((byte >> 4) & 0x0f)
                                    : static_cast<uint8_t>(byte & 0x0f);
                dst[col] = quant_map[static_cast<size_t>(code)] *
                           effective_absmax_for_block(weight, flat / blocksize);
            }
        }
    });
}

}  // namespace

bool CpuBnb4WeightRef::valid() const {
    return packed_weight != nullptr && absmax != nullptr && quant_map != nullptr &&
           quant_state.shape.size() == 2;
}

int64_t CpuBnb4WeightRef::out_features() const {
    return quant_state.shape.size() > 0 ? quant_state.shape[0] : 0;
}

int64_t CpuBnb4WeightRef::in_features() const {
    return quant_state.shape.size() > 1 ? quant_state.shape[1] : 0;
}

int64_t CpuBnb4WeightRef::logical_numel() const {
    return out_features() * in_features();
}

int64_t CpuBnb4WeightRef::packed_num_bytes() const {
    return tensor_numel(packed_weight);
}

bool parse_cpu_bnb4_quant_state_json(const std::string& json_text,
                                     CpuBnb4QuantState& out,
                                     std::string* error) {
    out = {};
    if (error) {
        error->clear();
    }

    try {
        simdjson::dom::parser parser;
        simdjson::dom::element root = parser.parse(json_text);

        out.quant_type = read_json_string(root, "quant_type");
        out.dtype = read_json_string(root, "dtype");

        int64_t blocksize = 0;
        if (!read_json_int64(root, "blocksize", blocksize)) {
            set_error(error, "bitsandbytes quant_state is missing blocksize");
            return false;
        }
        out.blocksize = static_cast<int>(blocksize);

        simdjson::dom::element shape_elem;
        if (root.at_key("shape").get(shape_elem) != simdjson::SUCCESS) {
            set_error(error, "bitsandbytes quant_state is missing shape");
            return false;
        }

        out.shape.clear();
        for (auto value : shape_elem.get_array()) {
            int64_t dim = 0;
            if (value.get_int64().get(dim) != simdjson::SUCCESS) {
                set_error(error, "bitsandbytes quant_state shape contains a non-integer value");
                return false;
            }
            out.shape.push_back(dim);
        }

        int64_t nested_blocksize = 0;
        if (read_json_int64(root, "nested_blocksize", nested_blocksize)) {
            out.nested = true;
            out.nested_blocksize = static_cast<int>(nested_blocksize);
        }

        out.nested_dtype = read_json_string(root, "nested_dtype");
        if (!out.nested_dtype.empty()) {
            out.nested = true;
        }

        float nested_offset = 0.0f;
        if (read_json_float(root, "nested_offset", nested_offset)) {
            out.nested = true;
            out.nested_offset = nested_offset;
        }

        return true;
    } catch (const std::exception& e) {
        set_error(error, std::string("parse bitsandbytes quant_state failed: ") + e.what());
        return false;
    }
}

bool load_cpu_bnb4_weight_ref(const CpuSafetensorsStore& store,
                              const std::string& name,
                              CpuBnb4WeightRef& out,
                              std::string* error) {
    out = {};
    out.name = name;
    if (error) {
        error->clear();
    }

    const std::string absmax_name = name + ".absmax";
    const std::string quant_map_name = name + ".quant_map";
    const std::string nested_absmax_name = name + ".nested_absmax";
    const std::string nested_quant_map_name = name + ".nested_quant_map";
    const std::string packed_quant_state_name = name + ".quant_state.bitsandbytes__nf4";

    if (!store.has(name)) {
        set_error(error, "Missing bitsandbytes packed weight tensor: " + name);
        return false;
    }
    if (!store.has(absmax_name) || !store.has(quant_map_name) ||
        !store.has(packed_quant_state_name)) {
        set_error(error, "Missing required bitsandbytes side tensors for weight: " + name);
        return false;
    }

    out.packed_weight = &store.get(name);
    out.absmax = &store.get(absmax_name);
    out.quant_map = &store.get(quant_map_name);
    out.packed_quant_state = &store.get(packed_quant_state_name);
    if (store.has(nested_absmax_name)) {
        out.nested_absmax = &store.get(nested_absmax_name);
    }
    if (store.has(nested_quant_map_name)) {
        out.nested_quant_map = &store.get(nested_quant_map_name);
    }

    if (out.packed_weight->dtype != CpuDataType::U8) {
        set_error(error, "bitsandbytes packed weight must be uint8: " + name);
        return false;
    }
    if (out.quant_map->dtype != CpuDataType::F32) {
        set_error(error, "bitsandbytes quant_map must be float32: " + quant_map_name);
        return false;
    }
    if (out.packed_quant_state->dtype != CpuDataType::U8) {
        set_error(error,
                  "bitsandbytes packed quant_state must be uint8: " +
                      packed_quant_state_name);
        return false;
    }
    if (out.absmax->dtype != CpuDataType::U8 && out.absmax->dtype != CpuDataType::F32) {
        set_error(error, "bitsandbytes absmax must be uint8 or float32: " + absmax_name);
        return false;
    }

    const std::string state_json(
        reinterpret_cast<const char*>(out.packed_quant_state->u8_data()),
        out.packed_quant_state->bytes);
    if (!parse_cpu_bnb4_quant_state_json(state_json, out.quant_state, error)) {
        return false;
    }

    if (!out.valid()) {
        set_error(error, "Invalid bitsandbytes weight view: " + name);
        return false;
    }
    if (out.quant_state.quant_type != "nf4") {
        set_error(error, "Only NF4 bitsandbytes weights are supported: " + name);
        return false;
    }
    if (out.quant_state.blocksize <= 0) {
        set_error(error, "bitsandbytes blocksize must be positive: " + name);
        return false;
    }

    const int64_t logical_numel = out.logical_numel();
    if (logical_numel <= 0) {
        set_error(error, "bitsandbytes logical weight shape is invalid: " + name);
        return false;
    }
    if (out.packed_num_bytes() * 2 != logical_numel) {
        set_error(error,
                  "bitsandbytes packed weight size does not match logical shape: " + name);
        return false;
    }

    const int64_t block_count =
        ceil_div_i64(logical_numel, out.quant_state.blocksize);
    if (tensor_numel(out.absmax) != block_count) {
        set_error(error, "bitsandbytes absmax length does not match block count: " + name);
        return false;
    }
    if (tensor_numel(out.quant_map) < 16) {
        set_error(error, "bitsandbytes quant_map must contain at least 16 entries: " + name);
        return false;
    }

    if (out.quant_state.nested) {
        if (out.nested_absmax == nullptr || out.nested_quant_map == nullptr) {
            set_error(error, "bitsandbytes nested quantization tensors are missing: " + name);
            return false;
        }
        if (out.nested_absmax->dtype != CpuDataType::F32 ||
            out.nested_quant_map->dtype != CpuDataType::F32) {
            set_error(error, "bitsandbytes nested tensors must be float32: " + name);
            return false;
        }
        if (out.quant_state.nested_blocksize <= 0) {
            set_error(error, "bitsandbytes nested blocksize must be positive: " + name);
            return false;
        }

        const int64_t nested_block_count =
            ceil_div_i64(block_count, out.quant_state.nested_blocksize);
        if (tensor_numel(out.nested_absmax) != nested_block_count) {
            set_error(error,
                      "bitsandbytes nested_absmax length does not match nested block count: " +
                          name);
            return false;
        }
    } else if (out.absmax->dtype != CpuDataType::F32) {
        set_error(error, "Non-nested bitsandbytes absmax must be float32: " + name);
        return false;
    }

    out.decoded_absmax.assign(static_cast<size_t>(block_count), 0.0f);
    for (int64_t block = 0; block < block_count; ++block) {
        out.decoded_absmax[static_cast<size_t>(block)] =
            effective_absmax_for_block(out, block);
    }
    dequantize_dense_weight(out);

    return true;
}

void cpu_bnb4_matvec(const CpuBnb4WeightRef& weight,
                     const float* input,
                     float* output) {
    if (!weight.valid()) {
        throw std::runtime_error("Invalid CPU bitsandbytes 4-bit weight reference");
    }
    if (weight.quant_state.blocksize <= 0) {
        throw std::runtime_error("CPU bitsandbytes blocksize must be positive");
    }

    const int64_t rows = weight.out_features();
    const int64_t cols = weight.in_features();
    const int64_t total = weight.logical_numel();
    const uint8_t* packed = weight.packed_weight->u8_data();
    const float* quant_map = weight.quant_map->f32_data();
    const int blocksize = weight.quant_state.blocksize;

    if (!weight.dense_weight.empty()) {
        parallel_rows(rows, cols, 64, [&](int64_t row_begin, int64_t row_end) {
            for (int64_t row = row_begin; row < row_end; ++row) {
                const float* w = weight.dense_weight.data() + static_cast<size_t>(row) * cols;
                float sum = 0.0f;
                for (int64_t col = 0; col < cols; ++col) {
                    sum += input[static_cast<size_t>(col)] * w[col];
                }
                output[static_cast<size_t>(row)] = sum;
            }
        });
        return;
    }

    auto compute_rows = [&](int64_t row_begin, int64_t row_end) {
        for (int64_t row = row_begin; row < row_end; ++row) {
            float sum = 0.0f;
            for (int64_t col = 0; col < cols; ++col) {
                const int64_t flat = row * cols + col;
                if (flat >= total) {
                    break;
                }
                const uint8_t byte = packed[static_cast<size_t>(flat / 2)];
                const uint8_t code =
                    (flat % 2 == 0) ? static_cast<uint8_t>((byte >> 4) & 0x0f)
                                    : static_cast<uint8_t>(byte & 0x0f);
                const float scale =
                    effective_absmax_for_block(weight, flat / blocksize);
                sum += input[static_cast<size_t>(col)] *
                       quant_map[static_cast<size_t>(code)] * scale;
            }
            output[static_cast<size_t>(row)] = sum;
        }
    };

    parallel_rows(rows, cols, 64, compute_rows);
}
