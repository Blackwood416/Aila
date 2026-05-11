#pragma once

#include <cstdarg>
#include <cstdio>
#include <cctype>
#include <chrono>
#include <string>
#include <mutex>

// ============================================================
// Aila Unified Logging & Profiling
// ============================================================

namespace aila {

enum class LogLevel { Verbose = -1, Debug = 0, Info = 1, Warning = 2, Error = 3 };

// C-compatible callback signature (also used by C API)
using LogCallback = void(*)(int level, const char* message, void* user_data);

// Set custom log callback. Pass nullptr to restore default (stdout/stderr).
void set_log_callback(LogCallback cb, void* user_data = nullptr);

// Set minimum log level (messages below this level are suppressed).
void set_log_level(LogLevel level);

// Get current minimum log level.
LogLevel get_log_level();

// Core logging function (internal use, prefer macros below)
void log(LogLevel level, const char* fmt, ...);
void vlog(LogLevel level, const char* fmt, va_list args);

// Convert string to LogLevel (case-insensitive, supports names and numeric values)
inline LogLevel log_level_from_string(const std::string& s) {
    std::string lower;
    lower.reserve(s.size());
    for (char c : s) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    if (lower == "verbose" || lower == "trace" || lower == "-1") return LogLevel::Verbose;
    if (lower == "debug" || lower == "0")   return LogLevel::Debug;
    if (lower == "warning" || lower == "warn" || lower == "2") return LogLevel::Warning;
    if (lower == "error" || lower == "3")   return LogLevel::Error;
    return LogLevel::Info; // "info", "1", or anything else
}

// Get human-readable name for log level
inline const char* log_level_name(LogLevel level) {
    switch (level) {
        case LogLevel::Verbose: return "verbose";
        case LogLevel::Debug:   return "debug";
        case LogLevel::Info:    return "info";
        case LogLevel::Warning: return "warning";
        case LogLevel::Error:   return "error";
    }
    return "unknown";
}

// Convenience macros
#define AILA_LOG_VERBOSE(...) ::aila::log(::aila::LogLevel::Verbose, __VA_ARGS__)
#define AILA_LOG_DEBUG(...) ::aila::log(::aila::LogLevel::Debug, __VA_ARGS__)
#define AILA_LOG_INFO(...)  ::aila::log(::aila::LogLevel::Info, __VA_ARGS__)
#define AILA_LOG_WARN(...)  ::aila::log(::aila::LogLevel::Warning, __VA_ARGS__)
#define AILA_LOG_ERROR(...) ::aila::log(::aila::LogLevel::Error, __VA_ARGS__)

// ============================================================
// Scoped Timer for performance profiling
// ============================================================
class ScopedTimer {
public:
    explicit ScopedTimer(const char* label)
        : label_(label), start_(std::chrono::high_resolution_clock::now()) {}

    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start_).count();
        AILA_LOG_INFO("[Timer] %s: %.2f ms", label_, ms);
    }

    double elapsed_ms() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }

    // Non-copyable
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    const char* label_;
    std::chrono::high_resolution_clock::time_point start_;
};

} // namespace aila
