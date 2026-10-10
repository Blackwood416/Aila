#pragma once

#include <cstdlib>
#include <string>
#include <cstring>

// ============================================================
// Cross-platform environment variable utilities
// Consolidates duplicated read_env_* functions from multiple files.
// ============================================================

namespace aila {
namespace env {

extern int g_q35_prefill_step_override;
extern bool g_kv_quant_override;

inline bool read_flag(const char* name, bool default_value) {
    if (std::strcmp(name, "AILA_KV_QUANT") == 0 && g_kv_quant_override) {
        return true;
    }
#ifdef _WIN32
    char* value = nullptr;
    size_t len = 0;
    if (_dupenv_s(&value, &len, name) == 0 && value) {
        bool enabled = (std::atoi(value) != 0);
        free(value);
        return enabled;
    }
    if (value) free(value);
    return default_value;
#else
    const char* value = std::getenv(name);
    if (!value) return default_value;
    return (std::atoi(value) != 0);
#endif
}

inline bool read_flag_if_present(const char* name, bool* out_value) {
#ifdef _WIN32
    char* value = nullptr;
    size_t len = 0;
    if (_dupenv_s(&value, &len, name) == 0 && value) {
        if (out_value) {
            *out_value = (std::atoi(value) != 0);
        }
        free(value);
        return true;
    }
    if (value) free(value);
    return false;
#else
    const char* value = std::getenv(name);
    if (!value) return false;
    if (out_value) {
        *out_value = (std::atoi(value) != 0);
    }
    return true;
#endif
}

inline bool read_scoped_kv_quant(const char* scoped_name) {
    bool scoped_value = false;
    if (scoped_name &&
        scoped_name[0] != '\0' &&
        read_flag_if_present(scoped_name, &scoped_value)) {
        return scoped_value;
    }
    return read_flag("AILA_KV_QUANT", false);
}

inline int read_int(const char* name, int default_value) {
#ifdef _WIN32
    char* value = nullptr;
    size_t len = 0;
    if (_dupenv_s(&value, &len, name) == 0 && value) {
        int parsed = std::atoi(value);
        free(value);
        return parsed > 0 ? parsed : default_value;
    }
    if (value) free(value);
    return default_value;
#else
    const char* value = std::getenv(name);
    if (!value) return default_value;
    int parsed = std::atoi(value);
    return parsed > 0 ? parsed : default_value;
#endif
}

inline std::string read_string(const char* name, const std::string& default_value = "") {
#ifdef _WIN32
    char* value = nullptr;
    size_t len = 0;
    if (_dupenv_s(&value, &len, name) == 0 && value) {
        std::string result(value);
        free(value);
        return result;
    }
    if (value) free(value);
    return default_value;
#else
    const char* value = std::getenv(name);
    return value ? std::string(value) : default_value;
#endif
}

// AILA_KV_QUANT_FORMAT selects the 1-byte KV storage when AILA_KV_QUANT is on:
// "int8" opts into the symmetric INT8 prototype, anything else (including
// unset) keeps the legacy fp8 E4M3 cache. This deliberately does not widen
// AILA_KV_QUANT itself into an enum: its `atoi != 0` reading must stay intact.
inline bool kv_quant_format_is_int8() {
    const std::string value = read_string("AILA_KV_QUANT_FORMAT");
    return value == "int8" || value == "INT8" || value == "Int8";
}

inline float read_float(const char* name, float default_value = 0.0f) {
#ifdef _WIN32
    char* value = nullptr;
    size_t len = 0;
    if (_dupenv_s(&value, &len, name) == 0 && value) {
        float parsed = std::strtof(value, nullptr);
        free(value);
        return parsed;
    }
    if (value) free(value);
    return default_value;
#else
    const char* value = std::getenv(name);
    return value ? std::strtof(value, nullptr) : default_value;
#endif
}

// Variant returning raw int (allows 0 or negative values)
inline int read_int_raw(const char* name, int default_value) {
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

// SYCL_CACHE_PERSISTENT=1 (recommended by llama.cpp-SYCL / torch XPU guides and
// set by some launchers) crashes the bundled DPC++ runtime: sycl9 dereferences
// a null device-image pointer while hashing oneDNN interop programs at the
// first JIT build (issue #4). Aila-owned processes force it off before the
// first SYCL call. AILA_KEEP_SYCL_CACHE_PERSISTENT=1 keeps the inherited value
// (for testing runtimes that fix the bug). Returns true when the value was
// overridden. On Windows _putenv_s also updates the Win32 block via the shared
// UCRT, which is where sycl9.dll reads the variable.
inline bool disable_persistent_sycl_cache() {
    if (read_flag("AILA_KEEP_SYCL_CACHE_PERSISTENT", false)) {
        return false;
    }
    const std::string value = read_string("SYCL_CACHE_PERSISTENT");
    if (value.empty() || value == "0") {
        return false;
    }
#ifdef _WIN32
    return _putenv_s("SYCL_CACHE_PERSISTENT", "0") == 0;
#else
    return setenv("SYCL_CACHE_PERSISTENT", "0", 1) == 0;
#endif
}

// UR_L0_USE_COPY_ENGINE=1 enables the dedicated hardware Copy Engine (Blitter)
// in Level Zero / Unified Runtime for PCIe DMA operations, preventing memory
// copies from stalling the Xe Core compute engine. Must be set before the
// Level Zero driver initializes (i.e. before the first Context/device creation).
inline void enable_l0_copy_engine() {
#ifdef _WIN32
    char* existing = nullptr;
    size_t len = 0;
    if (_dupenv_s(&existing, &len, "UR_L0_USE_COPY_ENGINE") == 0 && existing) {
        free(existing);
    } else {
        _putenv_s("UR_L0_USE_COPY_ENGINE", "1");
    }
    if (_dupenv_s(&existing, &len, "SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE") == 0 && existing) {
        free(existing);
    } else {
        _putenv_s("SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE", "1");
    }
#else
    if (!std::getenv("UR_L0_USE_COPY_ENGINE")) {
        setenv("UR_L0_USE_COPY_ENGINE", "1", 1);
    }
    if (!std::getenv("SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE")) {
        setenv("SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE", "1", 1);
    }
#endif
}

} // namespace env
} // namespace aila
