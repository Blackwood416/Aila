#include <windows.h>

#include "ipc/IpcProtocol.hpp"
#include "ipc/Win32Pipe.hpp"

#include "simdjson.h"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace fs = std::filesystem;

std::string utf8(const std::wstring& value) {
    if (value.empty()) {
        return {};
    }
    const int size = WideCharToMultiByte(
        CP_UTF8,
        WC_ERR_INVALID_CHARS,
        value.data(),
        static_cast<int>(value.size()),
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
            static_cast<int>(value.size()),
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

void attempt_unrelated_handle_signal() {
    const std::wstring value = environment_value(L"AILA_TEST_UNRELATED_HANDLE");
    if (value.empty() || value.front() == L'-') {
        return;
    }
    wchar_t* end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::wcstoull(value.c_str(), &end, 10);
    if (errno == ERANGE || end == value.c_str() || *end != L'\0' ||
        parsed > (std::numeric_limits<uintptr_t>::max)()) {
        return;
    }
    (void)SetEvent(reinterpret_cast<HANDLE>(static_cast<uintptr_t>(parsed)));
}

std::string inspection_payload() {
    attempt_unrelated_handle_signal();
    const fs::path executable = fs::absolute(module_path()).lexically_normal();
    const fs::path runtime = executable.parent_path().lexically_normal();
    const fs::path cwd = fs::absolute(current_directory()).lexically_normal();
    return std::string("{") +
        "\"buildId\":\"fake-worker-v1\"," +
        "\"executable\":" + json_string(utf8(executable.wstring())) + "," +
        "\"runtimeDirectory\":" + json_string(utf8(runtime.wstring())) + "," +
        "\"currentDirectory\":" + json_string(utf8(cwd.wstring())) + "," +
        "\"path\":" + json_string(utf8(environment_value(L"PATH"))) + "," +
        "\"sentinel\":" + json_string(utf8(environment_value(L"AILA_TEST_SENTINEL"))) +
        "}";
}

uintptr_t parse_handle_value(const wchar_t* value) {
    if (value == nullptr || *value == L'\0' || *value == L'-') {
        throw std::runtime_error("missing or invalid inherited handle value");
    }
    wchar_t* end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::wcstoull(value, &end, 10);
    if (errno == ERANGE || end == value || *end != L'\0' ||
        parsed > (std::numeric_limits<uintptr_t>::max)()) {
        throw std::runtime_error("invalid inherited handle value");
    }
    return static_cast<uintptr_t>(parsed);
}

struct Handles {
    HANDLE command_read = INVALID_HANDLE_VALUE;
    HANDLE response_write = INVALID_HANDLE_VALUE;
    HANDLE event_write = INVALID_HANDLE_VALUE;
};

Handles parse_arguments(int argc, wchar_t** argv) {
    if (argc != 8 || std::wstring_view(argv[1]) != L"--ffi-worker") {
        throw std::runtime_error("invalid fake worker command line");
    }

    Handles handles;
    for (int index = 2; index < argc; index += 2) {
        const std::wstring_view option(argv[index]);
        HANDLE handle = reinterpret_cast<HANDLE>(parse_handle_value(argv[index + 1]));
        if (handle == nullptr || handle == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("inherited handle value is not usable");
        }
        if (option == L"--command-read-handle") {
            handles.command_read = handle;
        } else if (option == L"--response-write-handle") {
            handles.response_write = handle;
        } else if (option == L"--event-write-handle") {
            handles.event_write = handle;
        } else {
            throw std::runtime_error("unknown fake worker option");
        }
    }
    if (handles.command_read == INVALID_HANDLE_VALUE ||
        handles.response_write == INVALID_HANDLE_VALUE ||
        handles.event_write == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("fake worker did not receive all protocol handles");
    }
    return handles;
}

void send_frame(HANDLE handle, const aila::ipc::Frame& frame) {
    std::string error;
    if (!aila::ipc::write_frame(handle, frame, error)) {
        throw std::runtime_error(error);
    }
}

void send_partial_frame_and_exit(
    HANDLE handle,
    const aila::ipc::Frame& frame,
    UINT exit_code) {
    const std::vector<std::byte> encoded = aila::ipc::encode_frame(frame);
    if (encoded.size() < 12) {
        throw std::runtime_error("encoded fake frame was unexpectedly short");
    }
    std::string error;
    if (!aila::ipc::write_all(handle, encoded.data(), 12, error)) {
        throw std::runtime_error(error);
    }
    Sleep(3000);
    ExitProcess(exit_code);
}

int exit_code_from_payload(const std::string& payload_json) {
    simdjson::dom::parser parser;
    simdjson::dom::element root;
    int64_t code = 0;
    if (parser.parse(payload_json).get(root) != simdjson::SUCCESS ||
        root["code"].get_int64().get(code) != simdjson::SUCCESS ||
        code < 0 || code > 255) {
        throw std::runtime_error("test.exit requires a code from 0 through 255");
    }
    return static_cast<int>(code);
}

int run(const Handles& handles) {
    aila::ipc::Frame handshake;
    handshake.header.kind = "handshake";
    handshake.header.payload_json = inspection_payload();
    if (environment_value(L"AILA_FAKE_WORKER_MODE") == L"partial-handshake") {
        send_partial_frame_and_exit(handles.response_write, handshake, 87);
    }
    send_frame(handles.response_write, handshake);

    aila::ipc::Frame event;
    event.header.kind = "event";
    event.header.method = "log";
    event.header.payload_json = "{\"message\":\"fake worker ready\"}";
    send_frame(handles.event_write, event);

    if (environment_value(L"AILA_FAKE_WORKER_MODE") == L"stop-reading") {
        Sleep(3000);
        ExitProcess(89);
    }

    for (;;) {
        aila::ipc::Frame command;
        std::string error;
        if (!aila::ipc::read_frame(handles.command_read, command, error)) {
            return error.find("end of pipe") != std::string::npos ? 0 : 3;
        }
        if (command.header.method == "test.exit") {
            ExitProcess(static_cast<UINT>(exit_code_from_payload(command.header.payload_json)));
        }

        aila::ipc::Frame response = command;
        response.header.kind = "result";
        if (command.header.method == "test.partial-response") {
            send_partial_frame_and_exit(handles.response_write, response, 88);
        }
        if (command.header.method == "test.inspect") {
            response.header.payload_json = inspection_payload();
            response.attachment.clear();
        }
        send_frame(handles.response_write, response);
        if (command.header.method == "shutdown") {
            return 0;
        }
    }
}

} // namespace

int wmain(int argc, wchar_t** argv) {
    try {
        return run(parse_arguments(argc, argv));
    } catch (...) {
        return 2;
    }
}
