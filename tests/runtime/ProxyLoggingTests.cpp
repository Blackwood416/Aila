#include "proxy/ProxyLogging.hpp"

#include <atomic>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace {
using namespace std::chrono_literals;

void throwing_callback(int, const char*, void* user_data) {
    static_cast<std::atomic_bool*>(user_data)->store(true, std::memory_order_release);
    throw std::runtime_error("synthetic same-module callback failure");
}

void recording_callback(int, const char*, void* user_data) {
    static_cast<std::atomic_bool*>(user_data)->store(true, std::memory_order_release);
}

void wait_until(const std::atomic_bool& value, const char* message) {
    for (int attempt = 0; attempt != 100; ++attempt) {
        if (value.load(std::memory_order_acquire)) return;
        std::this_thread::sleep_for(10ms);
    }
    throw std::runtime_error(message);
}
} // namespace

int main() {
    try {
        const auto source = aila::proxy::logging::create_source();
        std::atomic_bool threw{false};
        aila::proxy::logging::set_callback(throwing_callback, &threw);
        aila::proxy::logging::enqueue(source, 1, "throwing callback");
        wait_until(threw, "throwing callback was not invoked");

        std::atomic_bool recovered{false};
        aila::proxy::logging::set_callback(recording_callback, &recovered);
        aila::proxy::logging::enqueue(source, 1, "dispatcher survived");
        wait_until(recovered, "callback exception escaped or stopped dispatcher");

        aila::proxy::logging::deactivate(source);
        aila::proxy::logging::set_callback(nullptr, nullptr);
        std::cout << "AilaProxyLoggingTests passed\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
