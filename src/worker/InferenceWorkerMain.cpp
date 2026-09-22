#include "worker/InferenceWorkerMain.hpp"

#include "worker/InferenceWorker.hpp"
#include "core/ExecutionTrace.hpp"
#include "profile/Profiling.hpp"

#include <bcrypt.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <thread>
#include <utility>

namespace aila::worker {
namespace {

namespace wire = ipc::v2;
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

int64_t NowMs() noexcept {
    return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now().time_since_epoch()).count();
}

std::string NewEpoch() {
    std::array<unsigned char, 16> bytes{};
    if (BCryptGenRandom(nullptr, bytes.data(), static_cast<ULONG>(bytes.size()), BCRYPT_USE_SYSTEM_PREFERRED_RNG) < 0)
        throw std::runtime_error("Cannot generate Worker epoch");
    constexpr char hex[] = "0123456789abcdef";
    std::string result;
    result.reserve(32);
    for (const auto byte : bytes) { result.push_back(hex[byte >> 4]); result.push_back(hex[byte & 15]); }
    return result;
}

bool ReadCommand(HANDLE pipe, InferenceWorker& worker, ipc::Frame& command, std::string& error) {
    std::array<std::byte, 8> prefix{};
    if (!ipc::read_exact(pipe, prefix.data(), prefix.size(), error)) return false;
    // Obtain the selected limits AFTER reading the prefix. A reader already
    // waiting while configure completes must not keep using bootstrap bounds.
    const auto bounds = wire::frame_bounds(wire::Channel::Command, worker.Limits());
    wire::FrameSizes sizes;
    wire::ValidationError validation;
    if (!wire::inspect_prefix(prefix.data(), prefix.size(), sizes, validation, bounds)) {
        error = validation.message; return false;
    }
    std::vector<std::byte> bytes(sizes.totalBytes);
    std::memcpy(bytes.data(), prefix.data(), prefix.size());
    if (!ipc::read_exact(pipe, bytes.data() + prefix.size(), bytes.size() - prefix.size(), error)) return false;
    if (!wire::decode_frame(bytes, command, validation, bounds)) { error = validation.message; return false; }
    return true;
}

bool WriteFrame(HANDLE pipe, const ipc::Frame& frame, const wire::FrameBounds& bounds,
                std::atomic<int64_t>& started, std::string& error) {
    std::vector<std::byte> bytes;
    wire::ValidationError validation;
    if (!wire::encode_frame(frame, bytes, validation, bounds)) { error = validation.message; return false; }
    started.store(NowMs());
    const bool success = ipc::write_all(pipe, bytes.data(), bytes.size(), error);
    started.store(0);
    return success;
}

void LogCallback(int level, const char* message, void* userData) noexcept {
    if (message) static_cast<InferenceWorker*>(userData)->Log(level, message);
}

struct LogRegistration {
    explicit LogRegistration(InferenceWorker& worker) { aila::set_log_callback(LogCallback, &worker); }
    ~LogRegistration() { aila::set_log_callback(nullptr, nullptr); }
};

void CancelIo(std::thread& thread) noexcept {
    if (thread.joinable()) CancelSynchronousIo(static_cast<HANDLE>(thread.native_handle()));
}

} // namespace

int RunInferenceWorker(HANDLE commandRead, HANDLE responseWrite, HANDLE eventWrite, wire::Hello hello,
                       std::unique_ptr<InferenceModelFactory> factory) {
    hello.epoch = NewEpoch();
    hello.capabilities = wire::supported_capabilities();
    hello.hardLimits = wire::hard_limits();
    ipc::Frame handshake;
    wire::ValidationError validation;
    if (!wire::make_hello(hello, handshake, validation)) throw std::runtime_error(validation.message);
    InferenceWorker worker(hello.epoch, std::move(factory));
    LogRegistration logRegistration(worker);
    std::atomic<int64_t> responseStarted{0}, eventStarted{0};
    std::atomic<bool> readerDone{false}, responseDone{false}, eventDone{false};
    std::thread reader, responseSender, eventSender;
    try {
        responseSender = std::thread([&] {
            try {
                std::string error;
                if (!WriteFrame(responseWrite, handshake, wire::bootstrap_bounds(), responseStarted, error)) {
                    worker.TransportFailed(error);
                } else {
                    while (!worker.Closed() && !worker.Failed()) {
                        InferenceWorkerOutput output;
                        if (!worker.ReadResponse(output, 25ms)) continue;
                        if (worker.Failed()) break;
                        if (!WriteFrame(responseWrite, output.frame, wire::frame_bounds(wire::Channel::Response, worker.Limits()),
                                        responseStarted, error)) { worker.TransportFailed(error); break; }
                        if (output.written) output.written();
                    }
                }
            } catch (const std::exception& error) { worker.TransportFailed(error.what()); }
            catch (...) { worker.TransportFailed("Response writer failed"); }
            responseDone.store(true);
        });
        eventSender = std::thread([&] {
            try {
                while (!worker.Closed() && !worker.Failed()) {
                    InferenceWorkerOutput output;
                    if (!worker.ReadEvent(output, 25ms)) continue;
                    if (worker.Failed()) break;
                    std::string error;
                    if (!WriteFrame(eventWrite, output.frame, wire::frame_bounds(wire::Channel::Event, worker.Limits()),
                                    eventStarted, error)) { worker.TransportFailed(error); break; }
                    if (output.written) output.written();
                }
            } catch (const std::exception& error) { worker.TransportFailed(error.what()); }
            catch (...) { worker.TransportFailed("Event writer failed"); }
            eventDone.store(true);
        });
        reader = std::thread([&] {
            try {
                while (!worker.Closed() && !worker.Failed()) {
                    ipc::Frame command;
                    std::string error;
                    if (!ReadCommand(commandRead, worker, command, error)) {
                        if (!worker.Closed()) worker.TransportFailed(error);
                        break;
                    }
                    if (!worker.Enqueue(std::move(command), error)) {
                        if (!worker.Closed()) worker.TransportFailed(error);
                        break;
                    }
                }
            } catch (const std::exception& error) { worker.TransportFailed(error.what()); }
            catch (...) { worker.TransportFailed("Command reader failed"); }
            readerDone.store(true);
        });
    } catch (...) {
        worker.TransportFailed("Could not start Worker IO threads");
        if (!reader.joinable()) readerDone.store(true);
        if (!responseSender.joinable()) responseDone.store(true);
        if (!eventSender.joinable()) eventDone.store(true);
    }
    while (!worker.Closed() && !worker.Failed()) {
        const auto now = NowMs();
        const auto timeout = worker.Limits().connectionWriteTimeoutMs;
        const auto responseStart = responseStarted.load(), eventStart = eventStarted.load();
        if ((responseStart && now - responseStart >= timeout) || (eventStart && now - eventStart >= timeout)) {
            worker.TransportFailed("Connection pipe write timeout");
            break;
        }
        std::this_thread::sleep_for(5ms);
    }
    // Repeated cancellation closes the small race between a writer checking
    // the stop flag and entering synchronous IO. No IO handle is closed while
    // its thread still owns an in-flight operation.
    while (!readerDone.load() || !responseDone.load() || !eventDone.load()) {
        if (!readerDone.load()) CancelIo(reader);
        if (!responseDone.load()) CancelIo(responseSender);
        if (!eventDone.load()) CancelIo(eventSender);
        std::this_thread::sleep_for(2ms);
    }
    if (reader.joinable()) reader.join();
    if (responseSender.joinable()) responseSender.join();
    if (eventSender.joinable()) eventSender.join();
    const bool success = worker.Closed();
    if (!success) std::fprintf(stderr, "[Aila IPC v2] %s\n", worker.FailureMessage().c_str());
    worker.Stop();
    // Stop destroys the runtime and its trace writer. A fully written close
    // line can still fail to flush; only this surviving latch can reject it.
    const bool traceFailed = diagnostics::ExecutionTrace::ProcessFailed();
    if (traceFailed)
        std::fprintf(stderr, "[Aila IPC v2] Execution trace persistence failed; Worker shutdown is not successful\n");
    return success && !traceFailed ? 0 : 5;
}

} // namespace aila::worker
