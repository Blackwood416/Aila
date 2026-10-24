#pragma once
#include "utils/EnvUtils.hpp"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace aila {
// Apply before model/queue construction, while read-once dispatch flags are unset.
// Explicit environment overrides remain visible; expert arithmetic stays GPU-only.
inline void ApplyQwen38A770Preset() {
    struct Setting { const char* name; const char* value; };
    constexpr Setting settings[] = {
        {"AILA_Q4E_PREFILL_XMX", "1"},
        {"AILA_Q4E_DIRECT_PREFILL_IO", "1"},
        {"AILA_Q4E_DEMAND_MICROBATCH", "32"},
        {"AILA_Q4E_PHASE_SCRATCH", "1"},
        {"AILA_Q4E_HOT_RAM_MB", "32768"},
        {"AILA_Q4E_EXPERT_READER", "readfile"},
        // Round 2: 512 forces a 3K-token prompt through six chunks, and expert
    // staging plus the expert weight reads are paid again per chunk. Measured on
    // tools/prefill-bench.ps1 at 2995 tokens: chunk 512 -> 137 tok/s,
    // chunk 2048 -> 156 tok/s, chunk 4096 -> 195 tok/s, because a single chunk
    // lets each expert's weights be read once for all of its routed token rows.
    {"AILA_Q35_PREFILL_CHUNK", "4096"},
        {"AILA_Q4E_INLINE_CHECKPOINT", "1"},
        {"AILA_Q4E_STATIC_PINNING", "0"},
        {"AILA_Q4E_GROUPED_XMX_MIN_ROWS", "0"},
        {"AILA_Q4E_GDN_CHUNK_MODE", "0"},
        {"AILA_Q4E_SMALL_ROWS_DECODE", "0"},
        {"AILA_Q4E_SMALL_ROWS_HC", "0"}
    };
    auto set = [](const char* name, const char* value) {
#ifdef _WIN32
        const int status = _putenv_s(name, value);
#else
        const int status = setenv(name, value, 1);
#endif
        if (status != 0) throw std::runtime_error("Cannot apply performance preset environment");
    };
    for (const auto& setting : settings) {
        const auto inherited = env::read_string(setting.name, "");
        if (inherited.empty()) set(setting.name, setting.value);
        std::fprintf(stderr, "[PerformancePreset] %s=%s source=%s\n", setting.name,
            inherited.empty() ? setting.value : inherited.c_str(), inherited.empty() ? "preset" : "environment");
    }
    set("AILA_Q4E_CPU_EXPERT_DISPATCH", "0");
    set("AILA_Q4E_ACTIVE_PRESET", "qwen38-a770");
}
} // namespace aila
