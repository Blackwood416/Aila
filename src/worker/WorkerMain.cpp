#include <windows.h>

#include "aila_api.h"
#include "ipc/IpcProtocol.hpp"
#include "ipc/Win32Pipe.hpp"
#include "utils/EnvUtils.hpp"
#include "worker/CApiWorkerEngine.hpp"
#include "worker/WorkerDispatcher.hpp"
#include "simdjson.h"

#include <cerrno>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <thread>
#include <mutex>

#ifndef AILA_BUILD_ID
#error "AilaWorker requires the deterministic AILA_BUILD_ID compile definition"
#endif

namespace {

namespace fs = std::filesystem;

class UniqueHandle {
public:
    UniqueHandle() = default;
    explicit UniqueHandle(HANDLE handle) noexcept : handle_(handle) {}

    UniqueHandle(const UniqueHandle&) = delete;
    UniqueHandle& operator=(const UniqueHandle&) = delete;

    UniqueHandle(UniqueHandle&& other) noexcept : handle_(other.release()) {}
    UniqueHandle& operator=(UniqueHandle&& other) noexcept {
        if (this != &other) {
            reset(other.release());
        }
        return *this;
    }

    ~UniqueHandle() { reset(); }

    HANDLE get() const noexcept { return handle_; }
    explicit operator bool() const noexcept {
        return handle_ != nullptr && handle_ != INVALID_HANDLE_VALUE;
    }

private:
    HANDLE release() noexcept {
        const HANDLE handle = handle_;
        handle_ = nullptr;
        return handle;
    }

    void reset(HANDLE handle = nullptr) noexcept {
        if (*this) {
            CloseHandle(handle_);
        }
        handle_ = handle;
    }

    HANDLE handle_ = nullptr;
};

struct Handles {
    UniqueHandle command_read;
    UniqueHandle response_write;
    UniqueHandle event_write;
};

std::string utf8(const std::wstring& value) {
    if (value.empty()) {
        return {};
    }
    if (value.size() > static_cast<size_t>((std::numeric_limits<int>::max)())) {
        throw std::runtime_error("UTF-16 value is too long to encode as UTF-8");
    }
    const int length = static_cast<int>(value.size());
    const int size = WideCharToMultiByte(
        CP_UTF8,
        WC_ERR_INVALID_CHARS,
        value.data(),
        length,
        nullptr,
        0,
        nullptr,
        nullptr);
    if (size <= 0) {
        throw std::runtime_error("could not encode UTF-16 as UTF-8");
    }
    std::string result(static_cast<size_t>(size), '\0');
    if (WideCharToMultiByte(
            CP_UTF8,
            WC_ERR_INVALID_CHARS,
            value.data(),
            length,
            result.data(),
            size,
            nullptr,
            nullptr) != size) {
        throw std::runtime_error("could not encode UTF-16 as UTF-8");
    }
    return result;
}

std::string json_string(std::string_view value) {
    constexpr char hex[] = "0123456789abcdef";
    std::string result = "\"";
    for (const unsigned char character : value) {
        switch (character) {
            case '"': result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\b': result += "\\b"; break;
            case '\f': result += "\\f"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default:
                if (character < 0x20) {
                    result += "\\u00";
                    result += hex[character >> 4];
                    result += hex[character & 0x0f];
                } else {
                    result += static_cast<char>(character);
                }
                break;
        }
    }
    result += '"';
    return result;
}

std::wstring module_path() {
    std::vector<wchar_t> buffer(MAX_PATH);
    for (;;) {
        const DWORD copied =
            GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (copied == 0) {
            throw std::runtime_error("GetModuleFileNameW failed");
        }
        if (copied < buffer.size()) {
            return std::wstring(buffer.data(), copied);
        }
        if (buffer.size() > static_cast<size_t>((std::numeric_limits<DWORD>::max)()) / 2) {
            throw std::runtime_error("module path exceeds Win32 limits");
        }
        buffer.resize(buffer.size() * 2);
    }
}

std::wstring current_directory() {
    const DWORD capacity = GetCurrentDirectoryW(0, nullptr);
    if (capacity == 0) {
        throw std::runtime_error("GetCurrentDirectoryW size query failed");
    }
    std::vector<wchar_t> buffer(capacity);
    const DWORD copied = GetCurrentDirectoryW(capacity, buffer.data());
    if (copied == 0 || copied >= capacity) {
        throw std::runtime_error("GetCurrentDirectoryW failed");
    }
    return std::wstring(buffer.data(), copied);
}

std::wstring environment_value(const wchar_t* name) {
    SetLastError(ERROR_SUCCESS);
    const DWORD capacity = GetEnvironmentVariableW(name, nullptr, 0);
    if (capacity == 0) {
        if (GetLastError() == ERROR_ENVVAR_NOT_FOUND) {
            return {};
        }
        throw std::runtime_error("GetEnvironmentVariableW size query failed");
    }
    std::vector<wchar_t> buffer(capacity);
    const DWORD copied = GetEnvironmentVariableW(name, buffer.data(), capacity);
    if (copied >= capacity) {
        throw std::runtime_error("GetEnvironmentVariableW failed");
    }
    return std::wstring(buffer.data(), copied);
}

uintptr_t parse_handle_value(const wchar_t* value) {
    if (value == nullptr || *value == L'\0' || *value == L'-') {
        throw std::runtime_error("missing or invalid inherited handle value");
    }
    wchar_t* end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::wcstoull(value, &end, 10);
    if (errno == ERANGE || end == value || *end != L'\0' || parsed == 0 ||
        parsed > (std::numeric_limits<uintptr_t>::max)()) {
        throw std::runtime_error("invalid inherited handle value");
    }
    return static_cast<uintptr_t>(parsed);
}

HANDLE parse_handle(const wchar_t* value) {
    HANDLE handle = reinterpret_cast<HANDLE>(parse_handle_value(value));
    if (handle == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("inherited handle value is not usable");
    }
    DWORD flags = 0;
    if (GetHandleInformation(handle, &flags) == FALSE) {
        throw std::runtime_error("inherited handle is not valid in the worker process");
    }
    return handle;
}

Handles parse_arguments(int argc, wchar_t** argv) {
    if (argc != 8 || argv == nullptr || argv[1] == nullptr ||
        std::wstring_view(argv[1]) != L"--ffi-worker") {
        throw std::runtime_error("invalid worker command line");
    }

    Handles handles;
    for (int index = 2; index < argc; index += 2) {
        if (argv[index] == nullptr || argv[index + 1] == nullptr) {
            throw std::runtime_error("worker command line contains a missing option value");
        }
        const std::wstring_view option(argv[index]);
        if (option == L"--command-read-handle") {
            if (handles.command_read) {
                throw std::runtime_error("duplicate --command-read-handle option");
            }
        } else if (option == L"--response-write-handle") {
            if (handles.response_write) {
                throw std::runtime_error("duplicate --response-write-handle option");
            }
        } else if (option == L"--event-write-handle") {
            if (handles.event_write) {
                throw std::runtime_error("duplicate --event-write-handle option");
            }
        } else {
            throw std::runtime_error("unknown worker option");
        }

        HANDLE handle = parse_handle(argv[index + 1]);
        if (handle == handles.command_read.get() ||
            handle == handles.response_write.get() ||
            handle == handles.event_write.get()) {
            throw std::runtime_error("worker protocol handles must be distinct");
        }
        if (option == L"--command-read-handle") {
            handles.command_read = UniqueHandle(handle);
        } else if (option == L"--response-write-handle") {
            handles.response_write = UniqueHandle(handle);
        } else {
            handles.event_write = UniqueHandle(handle);
        }
    }
    if (!handles.command_read || !handles.response_write || !handles.event_write) {
        throw std::runtime_error("worker did not receive all protocol handles");
    }
    return handles;
}

std::string handshake_payload() {
    const fs::path executable = fs::absolute(module_path()).lexically_normal();
    const fs::path runtime = executable.parent_path().lexically_normal();
    const fs::path cwd = fs::absolute(current_directory()).lexically_normal();
    return std::string("{") +
        "\"buildId\":" + json_string(AILA_BUILD_ID) + "," +
        "\"executable\":" + json_string(utf8(executable.wstring())) + "," +
        "\"runtimeDirectory\":" + json_string(utf8(runtime.wstring())) + "," +
        "\"currentDirectory\":" + json_string(utf8(cwd.wstring())) + "," +
        "\"path\":" + json_string(utf8(environment_value(L"PATH"))) +
        "}";
}

void send_frame(HANDLE handle, const aila::ipc::Frame& frame) {
    std::string error;
    if (!aila::ipc::write_frame(handle, frame, error)) {
        throw std::runtime_error(error);
    }
}

class EventWriter {
public:
    explicit EventWriter(HANDLE handle) : handle_(handle) {}

    void send(const aila::ipc::Frame& frame) {
        std::lock_guard<std::mutex> lock(mutex_);
        send_frame(handle_, frame);
    }

    void log(int level, const char* message) noexcept {
        if (level < 0 || level > 3 || message == nullptr) return;
        try {
            const std::string_view view(message);
            if (!simdjson::validate_utf8(view)) return;
            aila::ipc::Frame event;
            event.header.kind = "event";
            event.header.method = "log";
            event.header.payload_json =
                std::string("{\"level\":") + std::to_string(level) +
                ",\"message\":" + json_string(view) + "}";
            std::lock_guard<std::mutex> lock(mutex_);
            std::string error;
            (void)aila::ipc::write_frame(handle_, event, error);
        } catch (...) {}
    }

private:
    HANDLE handle_;
    std::mutex mutex_;
};

void worker_log_callback(int level, const char* message, void* user_data) noexcept {
    static_cast<EventWriter*>(user_data)->log(level, message);
}

class WorkerLogRegistration {
public:
    explicit WorkerLogRegistration(EventWriter& writer) {
        aila_set_log_callback(worker_log_callback, &writer);
    }
    ~WorkerLogRegistration() { aila_set_log_callback(nullptr, nullptr); }
};

bool clean_command_eof(DWORD available_before_read) {
    if (available_before_read != 0) {
        return false;
    }
    const DWORD error = GetLastError();
    return error == ERROR_BROKEN_PIPE || error == ERROR_HANDLE_EOF || error == ERROR_NO_DATA;
}

int run(const Handles& handles) {
    EventWriter event_writer(handles.event_write.get());
    WorkerLogRegistration log_registration(event_writer);
    aila::worker::WorkerDispatcher dispatcher(
        std::make_unique<aila::worker::CApiWorkerEngine>());

    aila::ipc::Frame handshake;
    handshake.header.kind = "handshake";
    handshake.header.payload_json = handshake_payload();
    send_frame(handles.response_write.get(), handshake);

    aila::ipc::Frame event;
    event.header.kind = "event";
    event.header.method = "log";
    event.header.payload_json = "{\"level\":1,\"message\":\"Aila worker ready\"}";
    event_writer.send(event);

    for (;;) {
        DWORD available = 0;
        if (PeekNamedPipe(
                handles.command_read.get(),
                nullptr,
                0,
                nullptr,
                &available,
                nullptr) == FALSE) {
            return clean_command_eof(available) ? 0 : 3;
        }

        aila::ipc::Frame command;
        std::string error;
        if (!aila::ipc::read_frame(handles.command_read.get(), command, error)) {
            if (available == 0 && error.find("end of pipe") != std::string::npos) {
                return 0;
            }
            return 3;
        }
        // Stream cancellation is one-way. A very short stream may finish just
        // before its control frame is observed; discard that stale control
        // frame instead of producing an unpaired response on the shared pipe.
        if (command.header.method == "cancel") {
            continue;
        }

        if (aila::worker::WorkerDispatcher::is_stream_method(command.header.method)) {
            std::atomic_bool cancelled = false;
            std::atomic_bool finished = false;
            uint64_t emitted_event_count = 0;
            aila::ipc::Frame stream_response;
            std::thread inference([&] {
                stream_response = dispatcher.dispatch_stream(
                    command,
                    [&](const aila::ipc::Frame& stream_event) {
                        event_writer.send(stream_event);
                        ++emitted_event_count;
                        // A successfully written event must be counted even if a cancel
                        // arrives immediately afterwards. The dispatcher checks the
                        // cancellation flag again after this emitter returns.
                        return true;
                    },
                    cancelled);
                finished.store(true, std::memory_order_release);
            });
            while (!finished.load(std::memory_order_acquire)) {
                DWORD pending = 0;
                if (PeekNamedPipe(
                        handles.command_read.get(), nullptr, 0, nullptr, &pending, nullptr) == FALSE) {
                    cancelled.store(true, std::memory_order_release);
                    break;
                }
                if (pending == 0) {
                    Sleep(1);
                    continue;
                }
                aila::ipc::Frame control;
                if (!aila::ipc::read_frame(handles.command_read.get(), control, error)) {
                    cancelled.store(true, std::memory_order_release);
                    break;
                }
                simdjson::dom::parser parser;
                simdjson::dom::element payload;
                uint64_t target = 0;
                if (control.header.kind != "request" || control.header.method != "cancel" ||
                    control.header.request_id != command.header.request_id ||
                    parser.parse(control.header.payload_json).get(payload) != simdjson::SUCCESS ||
                    payload["requestId"].get_uint64().get(target) != simdjson::SUCCESS ||
                    target != command.header.request_id) {
                    cancelled.store(true, std::memory_order_release);
                    break;
                }
                cancelled.store(true, std::memory_order_release);
            }
            inference.join();
            const uint64_t terminal_count = emitted_event_count + 1;
            if (terminal_count == 0 || terminal_count > aila::ipc::kMaxStreamEventCount) {
                throw std::runtime_error("stream emitted too many events");
            }
            if (stream_response.header.kind == "result") {
                simdjson::dom::parser parser;
                simdjson::dom::element payload;
                uint64_t declared_count = 0;
                if (parser.parse(stream_response.header.payload_json).get(payload) !=
                        simdjson::SUCCESS ||
                    payload["eventCount"].get_uint64().get(declared_count) !=
                        simdjson::SUCCESS ||
                    declared_count != terminal_count) {
                    throw std::runtime_error("stream result omitted terminal eventCount");
                }
            } else if (stream_response.header.kind == "error") {
                simdjson::dom::parser parser;
                simdjson::dom::element payload;
                int64_t code = 0;
                std::string_view message;
                if (parser.parse(stream_response.header.payload_json).get(payload) !=
                        simdjson::SUCCESS ||
                    payload["code"].get_int64().get(code) != simdjson::SUCCESS ||
                    payload["message"].get_string().get(message) != simdjson::SUCCESS ||
                    message.find('\0') != std::string_view::npos) {
                    throw std::runtime_error("stream error response schema was invalid");
                }
                stream_response.header.payload_json =
                    std::string("{\"code\":") + std::to_string(code) +
                    ",\"message\":" + json_string(message) +
                    ",\"eventCount\":" + std::to_string(terminal_count) + "}";
            } else {
                throw std::runtime_error("stream dispatcher returned an invalid response kind");
            }
            aila::ipc::Frame end = command;
            end.header.kind = "event";
            end.header.payload_json =
                std::string("{\"event\":\"end\",\"eventCount\":") +
                std::to_string(terminal_count) + "}";
            end.attachment.clear();
            event_writer.send(end);
            if (!aila::ipc::write_frame(handles.response_write.get(), stream_response, error)) {
                return 4;
            }
            continue;
        }

        bool should_shutdown = false;
        const aila::ipc::Frame response = dispatcher.dispatch(command, should_shutdown);
        if (!aila::ipc::write_frame(handles.response_write.get(), response, error)) {
            return 4;
        }
        if (should_shutdown) {
            return 0;
        }
    }
}

} // namespace

int wmain(int argc, wchar_t** argv) {
    if (aila::env::disable_persistent_sycl_cache()) {
        std::fprintf(
            stderr,
            "[Aila] SYCL_CACHE_PERSISTENT was inherited enabled; forcing 0 "
            "(known runtime crash; set AILA_KEEP_SYCL_CACHE_PERSISTENT=1 to keep)\n");
    }
    try {
        const Handles handles = parse_arguments(argc, argv);
        return run(handles);
    } catch (...) {
        return 2;
    }
}
