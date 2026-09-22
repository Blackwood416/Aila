#pragma once

#include <exception>
#include <utility>

namespace aila::device_failure {
namespace detail {

template <typename Exception>
bool Matches(const std::exception_ptr& failure) noexcept {
    try {
        std::rethrow_exception(failure);
    } catch (const Exception&) {
        return true;
    } catch (...) {
        return false;
    }
}

} // namespace detail

// Preserve the original dynamic exception across a best-effort loader/fallback
// boundary. Keep this policy independent of SYCL/oneDNN headers for CPU tests.
template <typename... DeviceExceptions, typename Work, typename Recover>
void RecoverNonDeviceExceptions(Work&& work, Recover&& recover) {
    static_assert(sizeof...(DeviceExceptions) != 0, "Device exception types are required");
    try {
        std::forward<Work>(work)();
    } catch (const std::exception& error) {
        const auto failure = std::current_exception();
        if ((detail::Matches<DeviceExceptions>(failure) || ...)) throw;
        std::forward<Recover>(recover)(error);
    }
}

} // namespace aila::device_failure
