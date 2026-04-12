#include "Profiling.hpp"
#include <cstdio>
#include <cstdarg>
#include <mutex>
#include <iostream>

namespace aila {

namespace {
    // Global state protected by mutex
    std::mutex g_log_mutex;
    LogCallback g_log_callback = nullptr;
    void* g_log_user_data = nullptr;
    LogLevel g_min_level = LogLevel::Info;

    // Default handler: route logs to stderr so CLI stdout stays usable
    // for streamed/generated model text and shell piping.
    void default_log_handler(LogLevel level, const char* message) {
        (void)level;
        std::cerr << message << std::endl;
    }
} // namespace

void set_log_callback(LogCallback cb, void* user_data) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    g_log_callback = cb;
    g_log_user_data = user_data;
}

void set_log_level(LogLevel level) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    g_min_level = level;
}

void vlog(LogLevel level, const char* fmt, va_list args) {
    LogCallback cb;
    void* ud;
    LogLevel min_level;
    {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        cb = g_log_callback;
        ud = g_log_user_data;
        min_level = g_min_level;
    }

    if (level < min_level) return;

    char buf[2048];
    vsnprintf(buf, sizeof(buf), fmt, args);

    if (cb) {
        cb(static_cast<int>(level), buf, ud);
    } else {
        default_log_handler(level, buf);
    }
}

void log(LogLevel level, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vlog(level, fmt, args);
    va_end(args);
}

} // namespace aila
