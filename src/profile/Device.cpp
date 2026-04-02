#include "Device.hpp"
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

int read_env_int(const char* name, int default_value) {
#ifdef _WIN32
    char* value = nullptr;
    size_t len = 0;
    if (_dupenv_s(&value, &len, name) == 0 && value) {
        int parsed = std::atoi(value);
        free(value);
        return parsed;
    }
    if (value) free(value);
    return default_value;
#else
    const char* value = std::getenv(name);
    return value ? std::atoi(value) : default_value;
#endif
}

const char* matrix_type_name(sycl::ext::oneapi::experimental::matrix::matrix_type t) {
    using matrix_type = sycl::ext::oneapi::experimental::matrix::matrix_type;
    switch (t) {
        case matrix_type::bf16: return "bf16";
        case matrix_type::fp16: return "fp16";
        case matrix_type::tf32: return "tf32";
        case matrix_type::fp32: return "fp32";
        case matrix_type::fp64: return "fp64";
        case matrix_type::sint8: return "s8";
        case matrix_type::sint16: return "s16";
        case matrix_type::sint32: return "s32";
        case matrix_type::sint64: return "s64";
        case matrix_type::uint8: return "u8";
        case matrix_type::uint16: return "u16";
        case matrix_type::uint32: return "u32";
        case matrix_type::uint64: return "u64";
        default: return "unknown";
    }
}

} // namespace

int CheckDevice()
{
    sycl::queue q(sycl::default_selector_v);
    auto dev = q.get_device();
    std::cout << "[Context] Device: " << dev.get_info<sycl::info::device::name>() << std::endl;
        std::cout << "[Context] Global memory: "
                  << dev.get_info<sycl::info::device::global_mem_size>() / (1024*1024)
                  << " MB" << std::endl;

    if (read_env_int("AILA_PRINT_MATRIX_COMBOS", 0) != 0) {
        using sycl::ext::oneapi::experimental::info::device::matrix_combinations;
        auto combos = dev.get_info<matrix_combinations>();
        std::cout << "[Context] matrix_combinations=" << combos.size() << std::endl;

        int shown = 0;
        const int show_max = 64;
        for (const auto& c : combos) {
            std::cout << "  [JM] m=" << c.msize
                      << " n=" << c.nsize
                      << " k=" << c.ksize
                      << " A=" << matrix_type_name(c.atype)
                      << " B=" << matrix_type_name(c.btype)
                      << " C=" << matrix_type_name(c.ctype)
                      << " D=" << matrix_type_name(c.dtype)
                      << std::endl;
            shown++;
            if (shown >= show_max) {
                std::cout << "  [JM] ... truncated at " << show_max << " entries" << std::endl;
                break;
            }
        }
    }

    return 0;
}
