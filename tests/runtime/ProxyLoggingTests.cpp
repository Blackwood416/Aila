#include <windows.h>

#include "proxy/ProxyLogging.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

namespace {
using namespace std::chrono_literals;

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void wait_until(const std::atomic_bool& value, const char* message) {
    for (int attempt = 0; attempt != 200; ++attempt) {
        if (value.load(std::memory_order_acquire)) return;
        std::this_thread::sleep_for(5ms);
    }
    throw std::runtime_error(message);
}

struct BlockingCallback {
    std::mutex mutex;
    std::condition_variable condition;
    bool entered = false;
    bool release = false;
    std::atomic_int calls{0};
};

void blocking_callback(int, const char*, void* user_data) {
    auto& state = *static_cast<BlockingCallback*>(user_data);
    state.calls.fetch_add(1, std::memory_order_relaxed);
    std::unique_lock<std::mutex> lock(state.mutex);
    state.entered = true;
    state.condition.notify_all();
    state.condition.wait(lock, [&] { return state.release; });
}

void release_callback(BlockingCallback& state) {
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.release = true;
    }
    state.condition.notify_all();
}

void wait_for_callback(BlockingCallback& state) {
    std::unique_lock<std::mutex> lock(state.mutex);
    expect(state.condition.wait_for(lock, 1s, [&] { return state.entered; }),
           "blocking callback was not entered");
}

void recording_callback(int, const char*, void* user_data) {
    static_cast<std::atomic_bool*>(user_data)->store(true, std::memory_order_release);
}

void throwing_callback(int, const char*, void* user_data) {
    static_cast<std::atomic_bool*>(user_data)->store(true, std::memory_order_release);
    throw std::runtime_error("synthetic same-module callback failure");
}

void test_default_stderr_fallback() {
    HANDLE read_handle = nullptr;
    HANDLE write_handle = nullptr;
    expect(CreatePipe(&read_handle, &write_handle, nullptr, 0) != FALSE,
           "could not create stderr capture pipe");
    const HANDLE original = GetStdHandle(STD_ERROR_HANDLE);
    expect(SetStdHandle(STD_ERROR_HANDLE, write_handle) != FALSE,
           "could not redirect stderr");

    const auto source = aila::proxy::logging::create_source();
    aila::proxy::logging::set_callback(nullptr, nullptr);
    aila::proxy::logging::enqueue(source, 1, "default fallback");
    std::string output;
    for (int attempt = 0; attempt != 200; ++attempt) {
        DWORD available = 0;
        if (PeekNamedPipe(read_handle, nullptr, 0, nullptr, &available, nullptr) != FALSE &&
            available != 0) {
            output.resize(available);
            DWORD read = 0;
            if (ReadFile(read_handle, output.data(), available, &read, nullptr) != FALSE) {
                output.resize(read);
                break;
            }
        }
        std::this_thread::sleep_for(5ms);
    }
    SetStdHandle(STD_ERROR_HANDLE, original);
    CloseHandle(write_handle);
    CloseHandle(read_handle);
    aila::proxy::logging::deactivate(source);
    expect(output == "default fallback\n", "NULL callback did not use exact stderr fallback");
}

void test_callback_replacement_waits_for_old_user_data() {
    const auto source = aila::proxy::logging::create_source();
    BlockingCallback old;
    aila::proxy::logging::set_callback(blocking_callback, &old);
    aila::proxy::logging::enqueue(source, 1, "old");
    wait_for_callback(old);

    std::atomic_bool replacement_called{false};
    std::atomic_bool replacement_returned{false};
    std::thread replace([&] {
        aila::proxy::logging::set_callback(recording_callback, &replacement_called);
        replacement_returned.store(true, std::memory_order_release);
    });
    std::this_thread::sleep_for(50ms);
    expect(!replacement_returned.load(std::memory_order_acquire),
           "callback replacement returned while old user_data was in flight");
    release_callback(old);
    replace.join();
    expect(replacement_returned.load(std::memory_order_acquire),
           "callback replacement did not complete after old callback returned");

    aila::proxy::logging::enqueue(source, 1, "new");
    wait_until(replacement_called, "replacement callback was not invoked");
    expect(old.calls.load(std::memory_order_relaxed) == 1,
           "old callback ran after replacement returned");
    aila::proxy::logging::deactivate(source);
    aila::proxy::logging::set_callback(nullptr, nullptr);
}

void test_source_deactivation_waits_for_in_flight_callback() {
    const auto source = aila::proxy::logging::create_source();
    BlockingCallback callback;
    aila::proxy::logging::set_callback(blocking_callback, &callback);
    aila::proxy::logging::enqueue(source, 1, "active");
    wait_for_callback(callback);

    std::atomic_bool deactivated{false};
    std::thread deactivate([&] {
        aila::proxy::logging::deactivate(source);
        deactivated.store(true, std::memory_order_release);
    });
    std::this_thread::sleep_for(50ms);
    expect(!deactivated.load(std::memory_order_acquire),
           "source deactivation returned while callback was in flight");
    release_callback(callback);
    deactivate.join();
    aila::proxy::logging::enqueue(source, 1, "inactive");
    std::this_thread::sleep_for(50ms);
    expect(callback.calls.load(std::memory_order_relaxed) == 1,
           "callback ran after source deactivation returned");
    aila::proxy::logging::set_callback(nullptr, nullptr);
}

struct ReentrantCallback {
    aila::proxy::logging::Source source;
    std::atomic_bool returned{false};
};

void reentrant_unregister_callback(int, const char*, void* user_data) {
    auto& state = *static_cast<ReentrantCallback*>(user_data);
    aila::proxy::logging::set_callback(nullptr, nullptr);
    aila::proxy::logging::deactivate(state.source);
    state.returned.store(true, std::memory_order_release);
}

void test_callback_can_reentrantly_unregister_and_deactivate() {
    ReentrantCallback state{aila::proxy::logging::create_source()};
    aila::proxy::logging::set_callback(reentrant_unregister_callback, &state);
    aila::proxy::logging::enqueue(state.source, 1, "reentrant");
    wait_until(state.returned, "reentrant callback unregister/deactivate deadlocked");
    aila::proxy::logging::enqueue(state.source, 1, "after");
    std::this_thread::sleep_for(50ms);
}

void test_same_module_callback_exception_is_contained() {
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
}
} // namespace

int main() {
    try {
        test_default_stderr_fallback();
        test_callback_replacement_waits_for_old_user_data();
        test_source_deactivation_waits_for_in_flight_callback();
        test_callback_can_reentrantly_unregister_and_deactivate();
        test_same_module_callback_exception_is_contained();
        std::cout << "AilaProxyLoggingTests passed\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
