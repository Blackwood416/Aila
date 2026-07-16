#include <windows.h>

#include "aila_api.h"
#include "ipc/IpcProtocol.hpp"
#include "ipc/Win32Pipe.hpp"
#include "worker/WorkerDispatcher.hpp"
#include "simdjson.h"

#include <cerrno>
#include <atomic>
#include <cstdint>
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

class CApiWorkerEngine final : public aila::worker::WorkerEngineApi {
public:
    CApiWorkerEngine() : engine_(aila_engine_create()) {
        if (engine_ == nullptr) {
            throw std::runtime_error("aila_engine_create failed");
        }
    }

    ~CApiWorkerEngine() override {
        aila_engine_destroy(engine_);
        engine_ = nullptr;
    }

    int init(const std::string& model, int max_seq_len) override {
        return aila_engine_init(engine_, model.c_str(), max_seq_len);
    }

    void reset_context() override { aila_engine_reset_context(engine_); }
    int context_length() const override { return aila_engine_context_length(engine_); }
    int last_error_code() const override { return aila_last_error_code(engine_); }

    std::string last_error_message() const override {
        const char* message = aila_last_error_message(engine_);
        return message == nullptr ? std::string{} : std::string(message);
    }

    bool generate_text(
        const aila::worker::TextGenerationRequest& request,
        std::string& output) override {
        char* result = nullptr;
        switch (request.method) {
            case aila::worker::TextGenerationMethod::Generate:
                result = aila_generate(
                    engine_,
                    request.input.c_str(),
                    request.has_config ? &request.config : nullptr);
                break;
            case aila::worker::TextGenerationMethod::GenerateMessages:
                result = aila_generate_messages(
                    engine_,
                    request.input.c_str(),
                    request.has_config ? &request.config : nullptr);
                break;
            case aila::worker::TextGenerationMethod::GenerateChatJson:
                result = aila_generate_chat_json(
                    engine_,
                    request.input.c_str(),
                    request.has_config ? &request.config : nullptr);
                break;
            case aila::worker::TextGenerationMethod::GenerateChatJsonEx:
                result = aila_generate_chat_json_ex(
                    engine_,
                    request.input.c_str(),
                    request.has_v2_config ? &request.config_v2 : nullptr);
                break;
            case aila::worker::TextGenerationMethod::GenerateStream:
            case aila::worker::TextGenerationMethod::GenerateMessagesStream:
            case aila::worker::TextGenerationMethod::GenerateChatJsonStreamEx:
                return false;
        }
        if (!result) {
            output.clear();
            return false;
        }
        try {
            output.assign(result);
        } catch (...) {
            aila_free_string(result);
            throw;
        }
        aila_free_string(result);
        return true;
    }

    int generate_stream(
        const aila::worker::TextGenerationRequest& request,
        const aila::worker::TokenStreamCallback& token_callback,
        const aila::worker::StructuredStreamCallback& structured_callback) override {
        struct TokenContext {
            const aila::worker::TokenStreamCallback* callback;
        } token_context{&token_callback};
        struct StructuredContext {
            const aila::worker::StructuredStreamCallback* callback;
        } structured_context{&structured_callback};
        const auto token_adapter = [](const char* text, void* opaque) -> int {
            auto* context = static_cast<TokenContext*>(opaque);
            if (text == nullptr || !(*context->callback)(text)) {
                // The legacy in-process token API only suppresses later callbacks after
                // abort. Unwinding here also stops the underlying generation promptly;
                // dispatch_stream converts this caught C-API error to status 1 when the
                // control pipe cancellation flag is set.
                throw std::runtime_error("stream generation cancelled");
            }
            return 0;
        };
        const auto structured_adapter = [](const AilaChatStreamEvent* event, void* opaque) -> int {
            auto* context = static_cast<StructuredContext*>(opaque);
            return event != nullptr && (*context->callback)(*event) ? 0 : 1;
        };
        switch (request.method) {
            case aila::worker::TextGenerationMethod::GenerateStream:
                return aila_generate_stream(
                    engine_, request.input.c_str(),
                    request.has_config ? &request.config : nullptr,
                    token_adapter, &token_context);
            case aila::worker::TextGenerationMethod::GenerateMessagesStream:
                return aila_generate_messages_stream(
                    engine_, request.input.c_str(),
                    request.has_config ? &request.config : nullptr,
                    token_adapter, &token_context);
            case aila::worker::TextGenerationMethod::GenerateChatJsonStreamEx:
                return aila_generate_chat_json_stream_ex(
                    engine_, request.input.c_str(),
                    request.has_v2_config ? &request.config_v2 : nullptr,
                    structured_adapter, &structured_context);
            default:
                return -1;
        }
    }

private:
    AilaEngine* engine_ = nullptr;
};

bool clean_command_eof(DWORD available_before_read) {
    if (available_before_read != 0) {
        return false;
    }
    const DWORD error = GetLastError();
    return error == ERROR_BROKEN_PIPE || error == ERROR_HANDLE_EOF || error == ERROR_NO_DATA;
}

int run(const Handles& handles) {
    aila::worker::WorkerDispatcher dispatcher(std::make_unique<CApiWorkerEngine>());

    aila::ipc::Frame handshake;
    handshake.header.kind = "handshake";
    handshake.header.payload_json = handshake_payload();
    send_frame(handles.response_write.get(), handshake);

    aila::ipc::Frame event;
    event.header.kind = "event";
    event.header.method = "log";
    event.header.payload_json = "{\"level\":\"info\",\"message\":\"Aila worker ready\"}";
    send_frame(handles.event_write.get(), event);

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
            aila::ipc::Frame stream_response;
            std::thread inference([&] {
                stream_response = dispatcher.dispatch_stream(
                    command,
                    [&](const aila::ipc::Frame& stream_event) {
                        send_frame(handles.event_write.get(), stream_event);
                        return !cancelled.load(std::memory_order_acquire);
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
            aila::ipc::Frame end = command;
            end.header.kind = "event";
            end.header.payload_json = "{\"event\":\"end\"}";
            end.attachment.clear();
            send_frame(handles.event_write.get(), end);
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
    try {
        const Handles handles = parse_arguments(argc, argv);
        return run(handles);
    } catch (...) {
        return 2;
    }
}
