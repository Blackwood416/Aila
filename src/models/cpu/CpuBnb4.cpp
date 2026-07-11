#include "models/cpu/CpuBnb4.hpp"

#include "simdjson.h"
#include "utils/EnvUtils.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <functional>
#include <immintrin.h>
#include <intrin.h>
#include <mutex>
#include <stdexcept>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {

#if defined(__clang__) && (defined(_M_X64) || defined(__x86_64__))
#define AILA_TARGET_F16C __attribute__((target("avx2,f16c,fma")))
#else
#define AILA_TARGET_F16C
#endif

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

bool has_f16c() {
#if defined(_M_X64) || defined(__x86_64__)
    static const bool supported = []() {
        int regs[4] = {};
        __cpuid(regs, 1);
        const bool osxsave = (regs[2] & (1 << 27)) != 0;
        const bool avx = (regs[2] & (1 << 28)) != 0;
        const bool f16c = (regs[2] & (1 << 29)) != 0;
        if (!osxsave || !avx || !f16c || (_xgetbv(0) & 0x6) != 0x6) {
            return false;
        }
        __cpuidex(regs, 7, 0);
        return (regs[1] & (1 << 5)) != 0;
    }();
    return supported;
#else
    return false;
#endif
}

AILA_TARGET_F16C void f16_to_f32_f16c(
    const uint16_t* input, float* output, int64_t count) {
    int64_t i = 0;
    for (; i + 8 <= count; i += 8) {
        const __m128i packed =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + i));
        _mm256_storeu_ps(output + i, _mm256_cvtph_ps(packed));
    }
    for (; i < count; ++i) {
        output[i] = cpu_f16_to_float(input[i]);
    }
}

AILA_TARGET_F16C float f16_dot_f32_f16c(
    const uint16_t* weights, const float* input, int64_t count) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 16 <= count; i += 16) {
        const __m128i packed0 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(weights + i));
        const __m128i packed1 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(weights + i + 8));
        acc0 = _mm256_fmadd_ps(
            _mm256_cvtph_ps(packed0), _mm256_loadu_ps(input + i), acc0);
        acc1 = _mm256_fmadd_ps(
            _mm256_cvtph_ps(packed1), _mm256_loadu_ps(input + i + 8), acc1);
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    const __m128 lo = _mm256_castps256_ps128(acc0);
    const __m128 hi = _mm256_extractf128_ps(acc0, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    float result = _mm_cvtss_f32(sum);
    for (; i < count; ++i) {
        result += cpu_f16_to_float(weights[i]) * input[i];
    }
    return result;
}

AILA_TARGET_F16C float nf4_row_dot_avx2(const CpuBnb4WeightRef& weight,
                                        const float* quant_map,
                                        const float* input,
                                        int64_t row) {
    const int64_t cols = weight.in_features();
    const int64_t row_offset = row * cols;
    const int blocksize = weight.quant_state.blocksize;
    if (blocksize != 64 || (row_offset % blocksize) != 0) {
        float output = 0.0f;
        CpuBnb4WeightRef one_row = weight;
        one_row.quant_state.shape = {1, cols};
        one_row.packed_nf4_codes.assign(
            weight.packed_nf4_codes.begin() + row_offset / 2,
            weight.packed_nf4_codes.begin() + (row_offset + cols + 1) / 2);
        one_row.packed_nf4_absmax.assign(
            weight.packed_nf4_absmax.begin() + row_offset / blocksize,
            weight.packed_nf4_absmax.begin() +
                (row_offset + cols + blocksize - 1) / blocksize);
        cpu_nf4_matvec_scalar(one_row, quant_map, input, &output);
        return output;
    }

    const __m256 table_lo = _mm256_loadu_ps(quant_map);
    const __m256 table_hi = _mm256_loadu_ps(quant_map + 8);
    const bool use_i8_lut = weight.cache_mode == CpuBnb4CacheMode::PackedNf4I8;
    const __m128i table_i8 = _mm_loadu_si128(
        reinterpret_cast<const __m128i*>(weight.packed_nf4_lut_i8.data()));
    __m256 acc = _mm256_setzero_ps();
    for (int64_t block_col = 0; block_col < cols; block_col += blocksize) {
        const int64_t count = std::min<int64_t>(blocksize, cols - block_col);
        const float block_absmax = weight.packed_nf4_absmax[
            static_cast<size_t>((row_offset + block_col) / blocksize)];
        const __m256 scale = _mm256_set1_ps(block_absmax);
        const __m256 i8_scale =
            _mm256_set1_ps(block_absmax * weight.packed_nf4_lut_scale);
        const __m256 scaled_table_lo = _mm256_mul_ps(table_lo, scale);
        const __m256 scaled_table_hi = _mm256_mul_ps(table_hi, scale);
        int64_t lane_col = 0;
        for (; lane_col + 32 <= count; lane_col += 32) {
            const int64_t flat = row_offset + block_col + lane_col;
            const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                weight.packed_nf4_codes.data() + static_cast<size_t>(flat / 2)));
            const __m128i nibble_mask = _mm_set1_epi8(0x0f);
            const __m128i low = _mm_and_si128(bytes, nibble_mask);
            const __m128i high = _mm_and_si128(_mm_srli_epi16(bytes, 4), nibble_mask);
            const __m128i codes0 = _mm_unpacklo_epi8(low, high);
            const __m128i codes1 = _mm_unpackhi_epi8(low, high);
            const __m128i groups[4] = {
                codes0, _mm_srli_si128(codes0, 8),
                codes1, _mm_srli_si128(codes1, 8)};
            for (int group = 0; group < 4; ++group) {
                if (use_i8_lut) {
                    const __m128i quantized = _mm_shuffle_epi8(table_i8, groups[group]);
                    const __m256 values = _mm256_mul_ps(
                        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(quantized)), i8_scale);
                    acc = _mm256_fmadd_ps(
                        values,
                        _mm256_loadu_ps(input + block_col + lane_col + group * 8),
                        acc);
                    continue;
                }
                const __m256i indices = _mm256_cvtepu8_epi32(groups[group]);
                const __m256i low_indices =
                    _mm256_and_si256(indices, _mm256_set1_epi32(7));
                const __m256 low_values =
                    _mm256_permutevar8x32_ps(scaled_table_lo, low_indices);
                const __m256 high_values =
                    _mm256_permutevar8x32_ps(scaled_table_hi, low_indices);
                const __m256 high_mask = _mm256_castsi256_ps(
                    _mm256_cmpgt_epi32(indices, _mm256_set1_epi32(7)));
                const __m256 values =
                    _mm256_blendv_ps(low_values, high_values, high_mask);
                acc = _mm256_fmadd_ps(
                    values,
                    _mm256_loadu_ps(input + block_col + lane_col + group * 8),
                    acc);
            }
        }
        for (; lane_col < count; ++lane_col) {
            const int64_t flat = row_offset + block_col + lane_col;
            const uint8_t packed =
                weight.packed_nf4_codes[static_cast<size_t>(flat / 2)];
            const uint8_t code = (flat & 1)
                                     ? static_cast<uint8_t>((packed >> 4) & 0x0f)
                                     : static_cast<uint8_t>(packed & 0x0f);
            const float value = quant_map[static_cast<size_t>(code)] *
                                weight.packed_nf4_absmax[
                                    static_cast<size_t>(flat / blocksize)];
            acc = _mm256_add_ps(acc, _mm256_setr_ps(
                value * input[block_col + lane_col], 0, 0, 0, 0, 0, 0, 0));
        }
    }

    const __m128 lo = _mm256_castps256_ps128(acc);
    const __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum4 = _mm_add_ps(lo, hi);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    return _mm_cvtss_f32(sum4);
}

void lower_worker_priority() {
#ifdef _WIN32
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#endif
}

class PersistentRowPool {
public:
    explicit PersistentRowPool(int total_threads) {
        const int worker_count = std::max(0, total_threads - 1);
        workers_.reserve(static_cast<size_t>(worker_count));
        for (int worker_index = 0; worker_index < worker_count; ++worker_index) {
            workers_.emplace_back([this, worker_index]() { worker_loop(worker_index); });
        }
    }

    ~PersistentRowPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            ++generation_;
        }
        task_cv_.notify_all();
        for (std::thread& worker : workers_) {
            worker.join();
        }
    }

    PersistentRowPool(const PersistentRowPool&) = delete;
    PersistentRowPool& operator=(const PersistentRowPool&) = delete;

    void run(int64_t rows,
             int desired_threads,
             const std::function<void(int64_t, int64_t)>& fn) {
        if (desired_threads <= 1 || workers_.empty()) {
            fn(0, rows);
            return;
        }

        desired_threads = std::min<int>(
            desired_threads, static_cast<int>(workers_.size()) + 1);
        const int desired_workers = desired_threads - 1;
        const int64_t rows_per_thread =
            (rows + desired_threads - 1) / desired_threads;

        {
            std::unique_lock<std::mutex> lock(mutex_);
            idle_cv_.wait(lock, [this]() { return !in_flight_; });
            in_flight_ = true;
            task_ = fn;
            task_rows_ = rows;
            task_rows_per_thread_ = rows_per_thread;
            task_worker_count_ = desired_workers;
            active_workers_ = desired_workers;
            worker_exception_ = nullptr;
            ++generation_;
        }
        task_cv_.notify_all();

        std::exception_ptr caller_exception;
        const int64_t caller_begin =
            static_cast<int64_t>(desired_workers) * rows_per_thread;
        try {
            fn(caller_begin, rows);
        } catch (...) {
            caller_exception = std::current_exception();
        }

        std::exception_ptr worker_exception;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            done_cv_.wait(lock, [this]() { return active_workers_ == 0; });
            worker_exception = worker_exception_;
            task_ = {};
            in_flight_ = false;
        }
        idle_cv_.notify_one();

        if (caller_exception) {
            std::rethrow_exception(caller_exception);
        }
        if (worker_exception) {
            std::rethrow_exception(worker_exception);
        }
    }

private:
    void worker_loop(int worker_index) {
        lower_worker_priority();
        uint64_t observed_generation = 0;
        while (true) {
            std::function<void(int64_t, int64_t)> task;
            int64_t row_begin = 0;
            int64_t row_end = 0;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                task_cv_.wait(lock, [this, observed_generation]() {
                    return stopping_ || generation_ != observed_generation;
                });
                if (stopping_) {
                    return;
                }
                observed_generation = generation_;
                if (worker_index >= task_worker_count_) {
                    continue;
                }
                task = task_;
                row_begin = static_cast<int64_t>(worker_index) * task_rows_per_thread_;
                row_end = std::min(task_rows_, row_begin + task_rows_per_thread_);
            }

            try {
                task(row_begin, row_end);
            } catch (...) {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!worker_exception_) {
                    worker_exception_ = std::current_exception();
                }
            }

            {
                std::lock_guard<std::mutex> lock(mutex_);
                --active_workers_;
                if (active_workers_ == 0) {
                    done_cv_.notify_one();
                }
            }
        }
    }

    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable task_cv_;
    std::condition_variable done_cv_;
    std::condition_variable idle_cv_;
    std::function<void(int64_t, int64_t)> task_;
    int64_t task_rows_ = 0;
    int64_t task_rows_per_thread_ = 0;
    int task_worker_count_ = 0;
    int active_workers_ = 0;
    uint64_t generation_ = 0;
    bool in_flight_ = false;
    bool stopping_ = false;
    std::exception_ptr worker_exception_;
};

int configured_cpu_threads() {
    const int env_threads = aila::env::read_int_raw("AILA_CPU_Q35_THREADS", 4);
    return static_cast<int>(std::max(
        1u,
        static_cast<unsigned>(env_threads > 0
            ? env_threads
            : static_cast<int>(std::thread::hardware_concurrency()))));
}

PersistentRowPool& persistent_row_pool() {
    static PersistentRowPool pool(configured_cpu_threads());
    return pool;
}

}  // namespace

void cpu_q35_parallel_rows(
    int64_t rows,
    int64_t cols,
    int64_t min_rows_per_thread,
    const std::function<void(int64_t, int64_t)>& fn) {
    const int configured_threads = configured_cpu_threads();
    const int64_t desired_threads =
        (rows >= min_rows_per_thread * 2 && cols >= 256)
            ? std::min<int64_t>(static_cast<int64_t>(configured_threads),
                                std::max<int64_t>(1, rows / min_rows_per_thread))
            : 1;
    if (desired_threads <= 1) {
        fn(0, rows);
        return;
    }

    persistent_row_pool().run(rows, static_cast<int>(desired_threads), fn);
}

namespace {

void build_dense_weight_cache(CpuBnb4WeightRef& weight) {
    const int64_t rows = weight.out_features();
    const int64_t cols = weight.in_features();
    const int64_t total = weight.logical_numel();
    const uint8_t* packed = weight.packed_weight->u8_data();
    const float* quant_map = weight.quant_map->f32_data();
    const int blocksize = weight.quant_state.blocksize;

    if (weight.cache_mode == CpuBnb4CacheMode::PackedNf4 ||
        weight.cache_mode == CpuBnb4CacheMode::PackedNf4I8) {
        weight.packed_nf4_codes.assign(static_cast<size_t>((total + 1) / 2), 0);
        weight.packed_nf4_absmax = weight.decoded_absmax;
        if (weight.cache_mode == CpuBnb4CacheMode::PackedNf4I8) {
            float max_abs = 0.0f;
            for (int code = 0; code < 16; ++code) {
                max_abs = std::max(max_abs, std::abs(quant_map[code]));
            }
            weight.packed_nf4_lut_scale = max_abs / 127.0f;
            for (int code = 0; code < 16; ++code) {
                weight.packed_nf4_lut_i8[static_cast<size_t>(code)] =
                    static_cast<int8_t>(std::lrint(
                        quant_map[code] / weight.packed_nf4_lut_scale));
            }
        }
        for (int64_t flat = 0; flat < total; flat += 2) {
            const uint8_t source = packed[static_cast<size_t>(flat / 2)];
            const uint8_t first = static_cast<uint8_t>((source >> 4) & 0x0f);
            const uint8_t second = static_cast<uint8_t>(source & 0x0f);
            weight.packed_nf4_codes[static_cast<size_t>(flat / 2)] =
                static_cast<uint8_t>(first | (second << 4));
        }
        weight.decoded_absmax.clear();
        weight.decoded_absmax.shrink_to_fit();
        return;
    }

    weight.dense_weight_f16.assign(static_cast<size_t>(total), 0);
    cpu_q35_parallel_rows(rows, cols, 64, [&](int64_t row_begin, int64_t row_end) {
        for (int64_t row = row_begin; row < row_end; ++row) {
            uint16_t* dst =
                weight.dense_weight_f16.data() + static_cast<size_t>(row) * cols;
            for (int64_t col = 0; col < cols; ++col) {
                const int64_t flat = row * cols + col;
                const uint8_t byte = packed[static_cast<size_t>(flat / 2)];
                const uint8_t code =
                    (flat % 2 == 0) ? static_cast<uint8_t>((byte >> 4) & 0x0f)
                                    : static_cast<uint8_t>(byte & 0x0f);
                dst[col] = cpu_float_to_f16(
                    quant_map[static_cast<size_t>(code)] *
                    effective_absmax_for_block(weight, flat / blocksize));
            }
        }
    });
}

}  // namespace

CpuBnb4CacheMode parse_cpu_bnb4_cache_mode(std::string_view value) {
    std::string normalized(value);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (normalized == "packed_nf4_i8") {
        return CpuBnb4CacheMode::PackedNf4I8;
    }
    return normalized == "packed_nf4" ? CpuBnb4CacheMode::PackedNf4
                                      : CpuBnb4CacheMode::Fp16;
}

uint16_t cpu_float_to_f16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t exponent = (bits >> 23) & 0xffu;
    const uint32_t mantissa = bits & 0x7fffffu;

    if (exponent == 0xffu) {
        if (mantissa == 0) {
            return static_cast<uint16_t>(sign | 0x7c00u);
        }
        return static_cast<uint16_t>(sign | 0x7e00u);
    }

    const int32_t half_exponent = static_cast<int32_t>(exponent) - 127 + 15;
    if (half_exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
    if (half_exponent <= 0) {
        if (half_exponent < -10) {
            return static_cast<uint16_t>(sign);
        }
        const uint32_t normalized = mantissa | 0x800000u;
        const int shift = 14 - half_exponent;
        uint32_t rounded = normalized >> shift;
        const uint32_t remainder = normalized & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (rounded & 1u))) {
            ++rounded;
        }
        return static_cast<uint16_t>(sign | rounded);
    }

    uint32_t rounded_mantissa = mantissa >> 13;
    const uint32_t remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u ||
        (remainder == 0x1000u && (rounded_mantissa & 1u))) {
        ++rounded_mantissa;
        if (rounded_mantissa == 0x400u) {
            rounded_mantissa = 0;
            const uint32_t bumped_exponent = static_cast<uint32_t>(half_exponent + 1);
            if (bumped_exponent >= 31) {
                return static_cast<uint16_t>(sign | 0x7c00u);
            }
            return static_cast<uint16_t>(sign | (bumped_exponent << 10));
        }
    }
    return static_cast<uint16_t>(
        sign | (static_cast<uint32_t>(half_exponent) << 10) | rounded_mantissa);
}

float cpu_f16_to_float(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
    int32_t exponent = (value >> 10) & 0x1fu;
    uint32_t mantissa = value & 0x03ffu;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 1;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x03ffu;
            bits = sign |
                   (static_cast<uint32_t>(exponent + 127 - 15) << 23) |
                   (mantissa << 13);
        }
    } else if (exponent == 0x1fu) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign |
               (static_cast<uint32_t>(exponent + 127 - 15) << 23) |
               (mantissa << 13);
    }
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

void cpu_f16_to_f32(const uint16_t* input, float* output, int64_t count) {
    if (has_f16c()) {
        f16_to_f32_f16c(input, output, count);
        return;
    }
    for (int64_t i = 0; i < count; ++i) {
        output[i] = cpu_f16_to_float(input[i]);
    }
}

float cpu_f16_dot_f32(const uint16_t* weights, const float* input, int64_t count) {
    if (has_f16c()) {
        return f16_dot_f32_f16c(weights, input, count);
    }
    float result = 0.0f;
    for (int64_t i = 0; i < count; ++i) {
        result += cpu_f16_to_float(weights[i]) * input[i];
    }
    return result;
}

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

size_t CpuBnb4WeightRef::cache_bytes() const {
    return dense_weight_f16.size() * sizeof(uint16_t) +
           packed_nf4_codes.size() * sizeof(uint8_t) +
           packed_nf4_absmax.size() * sizeof(float);
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
                              std::string* error,
                              CpuBnb4CacheMode cache_mode) {
    out = {};
    out.name = name;
    out.cache_mode = cache_mode;
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
    build_dense_weight_cache(out);

    return true;
}

void cpu_nf4_matvec_scalar(const CpuBnb4WeightRef& weight,
                           const float* quant_map,
                           const float* input,
                           float* output) {
    const int64_t rows = weight.out_features();
    const int64_t cols = weight.in_features();
    const int blocksize = weight.quant_state.blocksize;
    for (int64_t row = 0; row < rows; ++row) {
        float sum = 0.0f;
        const int64_t row_offset = row * cols;
        for (int64_t col = 0; col < cols; ++col) {
            const int64_t flat = row_offset + col;
            const uint8_t packed =
                weight.packed_nf4_codes[static_cast<size_t>(flat / 2)];
            const uint8_t code = (flat & 1)
                                     ? static_cast<uint8_t>((packed >> 4) & 0x0f)
                                     : static_cast<uint8_t>(packed & 0x0f);
            const float scale = weight.packed_nf4_absmax[
                static_cast<size_t>(flat / blocksize)];
            sum += quant_map[static_cast<size_t>(code)] * scale * input[col];
        }
        output[row] = sum;
    }
}

void cpu_nf4_matvec(const CpuBnb4WeightRef& weight,
                    const float* quant_map,
                    const float* input,
                    float* output) {
    if (!has_f16c()) {
        cpu_nf4_matvec_scalar(weight, quant_map, input, output);
        return;
    }
    const int64_t rows = weight.out_features();
    const int64_t cols = weight.in_features();
    cpu_q35_parallel_rows(rows, cols, 64, [&](int64_t row_begin, int64_t row_end) {
        for (int64_t row = row_begin; row < row_end; ++row) {
            output[row] = nf4_row_dot_avx2(weight, quant_map, input, row);
        }
    });
}

AILA_TARGET_F16C void nf4_i8_matmul_rows_avx2(
    const CpuBnb4WeightRef& weight,
    const float* input,
    int64_t batch,
    int64_t row_begin,
    int64_t row_end,
    float* output) {
    const int64_t rows = weight.out_features();
    const int64_t cols = weight.in_features();
    const __m128i table_i8 = _mm_loadu_si128(
        reinterpret_cast<const __m128i*>(weight.packed_nf4_lut_i8.data()));
    const __m128i nibble_mask = _mm_set1_epi8(0x0f);
    for (int64_t row = row_begin; row < row_end; ++row) {
        __m256 acc[4] = {
            _mm256_setzero_ps(), _mm256_setzero_ps(),
            _mm256_setzero_ps(), _mm256_setzero_ps()};
        const int64_t row_offset = row * cols;
        for (int64_t block_col = 0; block_col < cols; block_col += 64) {
            const float block_scale =
                weight.packed_nf4_absmax[static_cast<size_t>(
                    (row_offset + block_col) / 64)] *
                weight.packed_nf4_lut_scale;
            const __m256 scale = _mm256_set1_ps(block_scale);
            for (int64_t lane_col = 0; lane_col < 64; lane_col += 32) {
                const int64_t flat = row_offset + block_col + lane_col;
                const __m128i bytes = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(
                        weight.packed_nf4_codes.data() + flat / 2));
                const __m128i low = _mm_and_si128(bytes, nibble_mask);
                const __m128i high =
                    _mm_and_si128(_mm_srli_epi16(bytes, 4), nibble_mask);
                const __m128i codes0 = _mm_unpacklo_epi8(low, high);
                const __m128i codes1 = _mm_unpackhi_epi8(low, high);
                const __m128i groups[4] = {
                    codes0, _mm_srli_si128(codes0, 8),
                    codes1, _mm_srli_si128(codes1, 8)};
                for (int group = 0; group < 4; ++group) {
                    const __m128i quantized =
                        _mm_shuffle_epi8(table_i8, groups[group]);
                    const __m256 values = _mm256_mul_ps(
                        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(quantized)),
                        scale);
                    const int64_t col = block_col + lane_col + group * 8;
                    for (int64_t item = 0; item < batch; ++item) {
                        acc[item] = _mm256_fmadd_ps(
                            values,
                            _mm256_loadu_ps(input + item * cols + col),
                            acc[item]);
                    }
                }
            }
        }
        for (int64_t item = 0; item < batch; ++item) {
            const __m128 lo = _mm256_castps256_ps128(acc[item]);
            const __m128 hi = _mm256_extractf128_ps(acc[item], 1);
            __m128 sum = _mm_add_ps(lo, hi);
            sum = _mm_hadd_ps(sum, sum);
            sum = _mm_hadd_ps(sum, sum);
            output[item * rows + row] = _mm_cvtss_f32(sum);
        }
    }
}

void cpu_nf4_matmul(const CpuBnb4WeightRef& weight,
                    const float* quant_map,
                    const float* input,
                    int64_t batch,
                    float* output) {
    const int64_t rows = weight.out_features();
    const int64_t cols = weight.in_features();
    if (has_f16c() && weight.cache_mode == CpuBnb4CacheMode::PackedNf4I8 &&
        (batch == 2 || batch == 4) && weight.quant_state.blocksize == 64 &&
        (cols % 64) == 0) {
        cpu_q35_parallel_rows(rows, cols, 64, [&](int64_t begin, int64_t end) {
            nf4_i8_matmul_rows_avx2(weight, input, batch, begin, end, output);
        });
        return;
    }
    for (int64_t item = 0; item < batch; ++item) {
        cpu_nf4_matvec(weight,
                       quant_map,
                       input + item * cols,
                       output + item * rows);
    }
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

    if (weight.cache_mode == CpuBnb4CacheMode::PackedNf4 ||
        weight.cache_mode == CpuBnb4CacheMode::PackedNf4I8) {
        cpu_nf4_matvec(weight, quant_map, input, output);
        return;
    }

    if (!weight.dense_weight_f16.empty()) {
        cpu_q35_parallel_rows(rows, cols, 64, [&](int64_t row_begin, int64_t row_end) {
            for (int64_t row = row_begin; row < row_end; ++row) {
                const uint16_t* w =
                    weight.dense_weight_f16.data() + static_cast<size_t>(row) * cols;
                output[static_cast<size_t>(row)] =
                    cpu_f16_dot_f32(w, input, cols);
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

    cpu_q35_parallel_rows(rows, cols, 64, compute_rows);
}

void cpu_bnb4_matmul(const CpuBnb4WeightRef& weight,
                     const float* input,
                     int64_t batch,
                     float* output) {
    if (batch <= 0) {
        return;
    }
    const int64_t cols = weight.in_features();
    const int64_t rows = weight.out_features();
    if (weight.cache_mode == CpuBnb4CacheMode::PackedNf4 ||
        weight.cache_mode == CpuBnb4CacheMode::PackedNf4I8) {
        cpu_nf4_matmul(
            weight, weight.quant_map->f32_data(), input, batch, output);
        return;
    }
    for (int64_t item = 0; item < batch; ++item) {
        cpu_bnb4_matvec(weight,
                        input + item * cols,
                        output + item * rows);
    }
}
