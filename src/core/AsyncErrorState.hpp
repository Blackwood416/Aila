#pragma once

#include <atomic>
#include <exception>
#include <mutex>
#include <utility>

namespace aila {

// Shared with a SYCL async_handler, whose lifetime may extend to queue teardown.
// Recording never formats/logs the error or invokes application callbacks. The
// execution boundary rethrows the original exception on its owning thread.
class AsyncErrorState final {
public:
    void Record(std::exception_ptr error) noexcept {
        if (!error) return;
        std::lock_guard<std::mutex> lock(mutex);
        if (!first) {
            first = std::move(error);
            failed.store(true, std::memory_order_release);
        }
    }

    bool Failed() const noexcept { return failed.load(std::memory_order_acquire); }

    void Rethrow() const {
        if (!Failed()) return;
        std::exception_ptr error;
        {
            std::lock_guard<std::mutex> lock(mutex);
            error = first;
        }
        std::rethrow_exception(error);
    }

    template <typename Drain>
    void DrainAndRethrow(Drain&& drain) {
        // A known error must not bypass the drain of already submitted work.
        try {
            std::forward<Drain>(drain)();
        } catch (...) {
            Record(std::current_exception());
        }
        Rethrow();
    }

private:
    mutable std::mutex mutex;
    std::exception_ptr first;
    std::atomic<bool> failed{false};
};

} // namespace aila
