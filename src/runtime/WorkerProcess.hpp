#pragma once

#include "ipc/IpcProtocol.hpp"

#include <windows.h>

#include <chrono>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace aila::runtime {

struct ExpectedHandshake {
    uint32_t protocol = ipc::kProtocolVersion;
    uint32_t abi = ipc::kPublicAbiVersion;
    std::string build_id;
};

using EventHandler = std::function<void(const ipc::Frame&)>;

class WorkerProcess {
public:
    WorkerProcess();
    ~WorkerProcess() noexcept;

    WorkerProcess(const WorkerProcess&) = delete;
    WorkerProcess& operator=(const WorkerProcess&) = delete;
    WorkerProcess(WorkerProcess&&) = delete;
    WorkerProcess& operator=(WorkerProcess&&) = delete;

    // Launches worker_executable with --ffi-worker and decimal handle values for
    // --command-read-handle, --response-write-handle, and --event-write-handle.
    void start(
        const std::filesystem::path& runtime_directory,
        const std::filesystem::path& worker_executable,
        const ExpectedHandshake& expected,
        EventHandler event_handler = {});

    ipc::Frame request(
        const ipc::Frame& frame,
        std::chrono::milliseconds timeout = std::chrono::seconds(30));

    bool cancel(
        uint64_t request_id,
        std::chrono::milliseconds timeout = std::chrono::seconds(2));

    void shutdown(std::chrono::milliseconds timeout = std::chrono::seconds(2));

    bool healthy() const;
    std::optional<DWORD> exit_code() const;
    std::string last_error() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace aila::runtime
