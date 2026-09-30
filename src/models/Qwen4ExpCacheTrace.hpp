#pragma once

#include <chrono>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

// Coordinator-only, opt-in JSONL diagnostics. No queue waits, USM allocations,
// or worker callbacks. The caller supplies observed readiness, never guessed
// completion times. Streaming output bounds host memory independently of run length.
class Qwen4ExpCacheTrace {
public:
    explicit Qwen4ExpCacheTrace(const std::string& path) : out_(path, std::ios::trunc) {
        if (!out_) throw std::runtime_error("Cannot open Q4E cache trace: " + path);
    }
    static int64_t NowNs() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    std::ostream& Row(const char* kind) {
        out_ << "{\"type\":\"" << kind << "\",\"ns\":" << NowNs();
        return out_;
    }
    void End() { out_ << "}\n"; }
    void Flush() {
        out_.flush();
        if (!out_) throw std::runtime_error("Q4E cache trace write failed");
    }
    static void Ints(std::ostream& out, const int* values, int n) {
        out << '[';
        for (int i = 0; i < n; ++i) { if (i) out << ','; out << values[i]; }
        out << ']';
    }
private:
    std::ofstream out_;
};
