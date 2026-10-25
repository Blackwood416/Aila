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
    // At 7878 tokens the same effect continues: chunk 4096 -> 214 tok/s,
    // chunk 8192 -> 239 tok/s, and short prompts (2026, 2995) measure the same
    // 171 / 194 tok/s at 8192 as at 4096 because they already fit one chunk.
    {"AILA_Q35_PREFILL_CHUNK", "8192"},
        {"AILA_Q4E_INLINE_CHECKPOINT", "1"},
        {"AILA_Q4E_LONG_INLINE_CHECKPOINT", "1"},
        {"AILA_Q4E_STATIC_PINNING", "0"},
        {"AILA_Q4E_GROUPED_XMX_MIN_ROWS", "0"},
        // Expert-XMX padded band height. Round 4 sweep at the shipped 8192 chunk,
        // one run per point, bracketed by the 256 default (20,734 / 20,975 ms):
        //   128 -> 20,586 ms   160 -> 20,496 ms   192 -> 20,221 ms
        // and at 4895 tokens 192 -> 15,113 ms against 15,592 / 15,646 for 256.
        // The optimum is not a padding-volume effect: 192 pads *more* rows than
        // 160 does, so this is the oneDNN batched-GEMM tiling preferring a 192-row
        // M, not less padded work.
        {"AILA_Q4E_PREFILL_EXPERT_XMX_MIN_M", "192"},
        // Round 3 A/B on tools/prefill-bench.ps1 at 7764 tokens: mode 0 (the old
        // preset value) 311.7 tok/s, mode 1 (FP32 chunk control) 319.7, mode 2
        // 317.7. The preset used to write 0 here, which silently overrode the
        // code default of 1 (preset beats the read_int default), so the faster
        // chunked GDN path was never the shipped configuration.
        {"AILA_Q4E_GDN_CHUNK_MODE", "1"},
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
