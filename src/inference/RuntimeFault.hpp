#pragma once

#include "inference/Session.hpp"
#include "memory/AllocationBudget.hpp"

#include <algorithm>
#include <array>
#include <string_view>
#include <utility>

namespace aila::inference::detail {

// Protected by the owner's execution/queue mutex. Recording the first failure
// must not allocate: a later drain failure or host OOM must not erase its cause.
class RuntimeFaultRecord final {
public:
    static constexpr size_t MaxMessageBytes = 1024;

    bool Faulted() const noexcept { return faulted; }

    void Record(std::string_view message) noexcept {
        if (faulted) return;
        faulted = true;
        messageSize = std::min(message.size(), MaxMessageBytes);
        if (messageSize < message.size()) {
            while (messageSize &&
                   (static_cast<unsigned char>(message[messageSize]) & 0xc0) == 0x80) {
                --messageSize;
            }
        }
        if (messageSize) std::copy_n(message.data(), messageSize, messageText.data());
        messageText[messageSize] = '\0';
    }

    const char* Message() const noexcept {
        return messageSize ? messageText.data() : "Device execution domain has failed";
    }

    void CheckUsable() const {
        if (faulted) throw Error(ErrorCode::RuntimeFault, Message());
    }

    template <typename Drain>
    void DrainNoThrow(Drain&& drain) noexcept {
        try {
            std::forward<Drain>(drain)();
        } catch (const std::exception& error) {
            Record(error.what());
        } catch (...) {
            Record("Unknown device failure");
        }
    }

    template <typename Drain>
    [[noreturn]] void Fail(std::string_view message, Drain&& drain) {
        // Preserve the synchronous failure before a successful drain can
        // consume it, or a second failure can obscure the original diagnosis.
        Record(message);
        DrainNoThrow(std::forward<Drain>(drain));
        throw Error(ErrorCode::RuntimeFault, Message());
    }

private:
    bool faulted = false;
    std::array<char, MaxMessageBytes + 1> messageText{};
    size_t messageSize = 0;
};

// Device/library exception types are parameters so the same lifecycle policy
// can be fault-injected on a CPU without loading a SYCL runtime. Work is never
// published until its final drain succeeds. Ordinary host/file failures remain
// local unless draining previously submitted device work also fails.
template <typename DeviceException, typename LibraryException, typename Work, typename Drain>
void RunDeviceLifecycle(RuntimeFaultRecord& fault, Work&& work, Drain&& drain) {
    fault.CheckUsable();
    try {
        std::forward<Work>(work)();
    } catch (const DeviceException& error) {
        fault.Fail(error.what(), drain);
    } catch (const LibraryException& error) {
        fault.Fail(error.what(), drain);
    } catch (const Error& error) {
        if (error.Code() == ErrorCode::RuntimeFault) fault.Fail(error.what(), drain);
        fault.DrainNoThrow(drain);
        fault.CheckUsable();
        throw;
    } catch (const aila::memory::AllocationLimitExceeded& error) {
        fault.DrainNoThrow(drain);
        fault.CheckUsable();
        throw Error(ErrorCode::ResourceLimit, error.what());
    } catch (const std::invalid_argument& error) {
        fault.DrainNoThrow(drain);
        fault.CheckUsable();
        throw Error(ErrorCode::InvalidArgument, error.what());
    } catch (...) {
        fault.DrainNoThrow(drain);
        fault.CheckUsable();
        throw;
    }
    // A failure specifically from the queue drain is a device-domain failure,
    // regardless of its C++ exception type. Do not retry it before recording it.
    fault.DrainNoThrow(drain);
    fault.CheckUsable();
}

} // namespace aila::inference::detail
