#include <windows.h>

#include "ipc/IpcProtocol.hpp"
#include "ipc/Win32Pipe.hpp"

#include "simdjson.h"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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
    const std::wstring configured_build_id =
        environment_value(L"AILA_FAKE_WORKER_BUILD_ID");
    const std::string build_id = configured_build_id.empty()
        ? "fake-worker-v1"
        : utf8(configured_build_id);
    return std::string("{") +
        "\"buildId\":" + json_string(build_id) + "," +
        "\"executable\":" + json_string(utf8(executable.wstring())) + "," +
        "\"runtimeDirectory\":" + json_string(utf8(runtime.wstring())) + "," +
        "\"currentDirectory\":" + json_string(utf8(cwd.wstring())) + "," +
        "\"path\":" + json_string(utf8(environment_value(L"PATH"))) + "," +
        "\"sentinel\":" + json_string(utf8(environment_value(L"AILA_TEST_SENTINEL"))) +
        "}";
}

void append_lifecycle_marker(std::string_view model, int max_seq_len) {
    const std::wstring marker = environment_value(L"AILA_FAKE_WORKER_LIFECYCLE_MARKER");
    if (marker.empty()) {
        return;
    }
    HANDLE file = CreateFileW(
        marker.c_str(),
        FILE_APPEND_DATA,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        nullptr,
        OPEN_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("could not open fake worker lifecycle marker");
    }
    const std::string line = std::to_string(GetCurrentProcessId()) + "|" +
        std::string(model) + "|" + std::to_string(max_seq_len) + "\n";
    DWORD written = 0;
    const bool ok = line.size() <= (std::numeric_limits<DWORD>::max)() &&
        WriteFile(
            file,
            line.data(),
            static_cast<DWORD>(line.size()),
            &written,
            nullptr) != FALSE &&
        written == line.size();
    CloseHandle(file);
    if (!ok) {
        throw std::runtime_error("could not write fake worker lifecycle marker");
    }
}

aila::ipc::Frame lifecycle_error(
    const aila::ipc::Frame& command,
    int code,
    std::string_view message) {
    aila::ipc::Frame response = command;
    response.header.kind = "error";
    response.header.payload_json = std::string("{\"code\":") + std::to_string(code) +
        ",\"message\":" + json_string(message) + "}";
    response.attachment.clear();
    return response;
}

bool is_generation_method(std::string_view method) {
    return method == "generate" ||
        method == "generate.messages" ||
        method == "generate.chat_json" ||
        method == "generate.chat_json_ex";
}

bool is_stream_generation_method(std::string_view method) {
    return method == "generate.stream" ||
        method == "generate.messages_stream" ||
        method == "generate.chat_json_stream_ex";
}

aila::ipc::Frame stream_event(
    const aila::ipc::Frame& command,
    std::string payload,
    std::string_view attachment = {}) {
    aila::ipc::Frame event = command;
    event.header.kind = "event";
    event.header.payload_json = std::move(payload);
    event.attachment.clear();
    for (unsigned char byte : attachment) {
        event.attachment.push_back(static_cast<std::byte>(byte));
    }
    return event;
}

void attach_text(aila::ipc::Frame& frame, std::string_view text) {
    frame.attachment.clear();
    frame.attachment.reserve(text.size());
    for (const unsigned char byte : text) {
        frame.attachment.push_back(static_cast<std::byte>(byte));
    }
    frame.header.payload_json =
        std::string("{\"byteCount\":") + std::to_string(text.size()) + "}";
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

    bool initialized = false;
    int context_length = 0;
    struct AsrStreamState {
        bool active = true;
        bool expect_floats = false;
        bool malformed_text = false;
    };
    std::vector<AsrStreamState> asr_streams;
    for (;;) {
        aila::ipc::Frame command;
        std::string error;
        if (!aila::ipc::read_frame(handles.command_read, command, error)) {
            return error.find("end of pipe") != std::string::npos ? 0 : 3;
        }
        if (command.header.method == "test.exit") {
            ExitProcess(static_cast<UINT>(exit_code_from_payload(command.header.payload_json)));
        }
        if (command.header.method == "cancel" &&
            command.header.payload_json.find("requestId") != std::string::npos) {
            continue;
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
        if (command.header.method == "test.invalid-event-protocol" ||
            command.header.method == "test.invalid-event-abi") {
            aila::ipc::Frame invalid;
            invalid.header.kind = "event";
            invalid.header.method = "log";
            invalid.header.payload_json = R"({"message":"invalid transport event"})";
            if (command.header.method == "test.invalid-event-protocol") {
                ++invalid.header.protocol;
            } else {
                ++invalid.header.abi;
            }
            send_frame(handles.event_write, invalid);
            Sleep(500);
            response.header.payload_json = "{\"unexpected\":true}";
            response.attachment.clear();
        }
        if (command.header.method == "engine.init") {
            if (initialized) {
                send_frame(
                    handles.response_write,
                    lifecycle_error(command, 1, "engine is already initialized"));
                continue;
            }
            simdjson::dom::parser parser;
            simdjson::dom::element payload;
            std::string_view model;
            int64_t max_seq_len = 0;
            if (parser.parse(command.header.payload_json).get(payload) != simdjson::SUCCESS ||
                payload["model"].get_string().get(model) != simdjson::SUCCESS ||
                model.empty() || model.find('\0') != std::string_view::npos ||
                payload["maxSeqLen"].get_int64().get(max_seq_len) != simdjson::SUCCESS ||
                max_seq_len <= 0 || max_seq_len > (std::numeric_limits<int>::max)()) {
                send_frame(
                    handles.response_write,
                    lifecycle_error(command, 1, "invalid engine.init payload"));
                continue;
            }
            append_lifecycle_marker(model, static_cast<int>(max_seq_len));
            initialized = true;
            context_length = static_cast<int>(max_seq_len);
            response.header.payload_json = "{\"ok\":true}";
            response.attachment.clear();
        }
        if (command.header.method == "engine.reset") {
            context_length = 0;
            response.header.payload_json = "{\"ok\":true}";
            response.attachment.clear();
        }
        if (command.header.method == "engine.context_length") {
            response.header.payload_json =
                std::string("{\"contextLength\":") +
                std::to_string(context_length) + "}";
            response.attachment.clear();
        }
        if (command.header.method == "asr.transcribe") {
            if (!initialized) {
                send_frame(handles.response_write,
                           lifecycle_error(command, 1, "engine is not initialized"));
                continue;
            }
            size_t event_count = 0;
            send_frame(handles.event_write, stream_event(
                command, R"({"event":"token","byteCount":6})", u8"识别"));
            ++event_count;
            send_frame(handles.event_write, stream_event(
                command, R"({"event":"token","byteCount":6})", " token"));
            ++event_count;
            const uint64_t terminal_count = event_count + 1;
            send_frame(handles.event_write, stream_event(
                command, "{\"event\":\"end\",\"eventCount\":" +
                    std::to_string(terminal_count) + "}"));
            const std::string transcript =
                command.header.payload_json.find("__aila_asr_empty__") != std::string::npos
                    ? std::string{} : std::string(u8"完整转录|") + command.header.payload_json;
            const std::string language =
                command.header.payload_json.find("__aila_asr_empty_language__") != std::string::npos
                    ? std::string{} : "Chinese";
            response.header.payload_json =
                "{\"transcriptBytes\":" + std::to_string(transcript.size()) +
                ",\"languageBytes\":" + std::to_string(language.size()) +
                ",\"eventCount\":" + std::to_string(terminal_count) + "}";
            response.attachment.clear();
            for (unsigned char byte : transcript) response.attachment.push_back(static_cast<std::byte>(byte));
            for (unsigned char byte : language) response.attachment.push_back(static_cast<std::byte>(byte));
            if (command.header.payload_json.find("__aila_asr_bad_lengths__") != std::string::npos) {
                response.header.payload_json =
                    "{\"transcriptBytes\":999,\"languageBytes\":0,\"eventCount\":" +
                    std::to_string(terminal_count) + "}";
            } else if (command.header.payload_json.find("__aila_asr_bad_utf8__") != std::string::npos) {
                response.attachment = {std::byte{0xc3}, std::byte{0x28}};
                response.header.payload_json =
                    "{\"transcriptBytes\":2,\"languageBytes\":0,\"eventCount\":" +
                    std::to_string(terminal_count) + "}";
            }
            send_frame(handles.response_write, response);
            continue;
        }
        if (command.header.method == "asr.stream.create") {
            if (!initialized) {
                send_frame(handles.response_write,
                           lifecycle_error(command, 1, "engine is not initialized"));
                continue;
            }
            const uint64_t id = asr_streams.size() + 1;
            asr_streams.push_back(AsrStreamState{true,
                command.header.payload_json.find("expect-floats") != std::string::npos,
                command.header.payload_json.find("malformed-text") != std::string::npos});
            response.header.payload_json = "{\"streamId\":" + std::to_string(id) + "}";
            response.attachment.clear();
        }
        if (command.header.method == "asr.stream.feed") {
            simdjson::dom::parser parser;
            simdjson::dom::element root;
            int64_t id = 0, count = 0, element = 0, bytes = 0;
            const bool valid =
                parser.parse(command.header.payload_json).get(root) == simdjson::SUCCESS &&
                root["streamId"].get_int64().get(id) == simdjson::SUCCESS &&
                root["sampleCount"].get_int64().get(count) == simdjson::SUCCESS &&
                root["elementSize"].get_int64().get(element) == simdjson::SUCCESS &&
                root["byteCount"].get_int64().get(bytes) == simdjson::SUCCESS &&
                id > 0 && count > 0 && bytes >= 0 &&
                static_cast<uint64_t>(id) <= asr_streams.size() &&
                asr_streams[static_cast<size_t>(id - 1)].active && element == 4 &&
                bytes == count * 4 && static_cast<uint64_t>(bytes) == command.attachment.size();
            bool exact = true;
            if (valid && asr_streams[static_cast<size_t>(id - 1)].expect_floats) {
                const uint32_t expected[] = {0x3fa00000u, 0xc0200000u, 0x00000000u};
                exact = command.attachment.size() == sizeof(expected) &&
                    std::memcmp(command.attachment.data(), expected, sizeof(expected)) == 0;
            }
            if (!valid || !exact) {
                send_frame(handles.response_write,
                           lifecycle_error(command, 1, "invalid ASR float attachment"));
                continue;
            }
            response.header.payload_json = "{\"ok\":true}";
            response.attachment.clear();
        }
        if (command.header.method == "asr.stream.get_text") {
            simdjson::dom::parser parser;
            simdjson::dom::element root;
            int64_t id = 0;
            if (parser.parse(command.header.payload_json).get(root) != simdjson::SUCCESS ||
                root["streamId"].get_int64().get(id) != simdjson::SUCCESS || id <= 0 ||
                static_cast<uint64_t>(id) > asr_streams.size() ||
                !asr_streams[static_cast<size_t>(id - 1)].active) {
                send_frame(handles.response_write, lifecycle_error(command, 1, "unknown ASR stream ID"));
                continue;
            }
            const std::string stable = u8"稳定";
            const std::string partial = u8"临时";
            response.attachment.clear();
            for (unsigned char byte : stable) response.attachment.push_back(static_cast<std::byte>(byte));
            for (unsigned char byte : partial) response.attachment.push_back(static_cast<std::byte>(byte));
            response.header.payload_json =
                "{\"stableBytes\":" + std::to_string(stable.size()) +
                ",\"partialBytes\":" + std::to_string(partial.size()) + "}";
            if (asr_streams[static_cast<size_t>(id - 1)].malformed_text) {
                response.header.payload_json = "{\"stableBytes\":999,\"partialBytes\":0}";
            }
        }
        if (command.header.method == "asr.stream.destroy") {
            simdjson::dom::parser parser;
            simdjson::dom::element root;
            int64_t id = 0;
            if (parser.parse(command.header.payload_json).get(root) != simdjson::SUCCESS ||
                root["streamId"].get_int64().get(id) != simdjson::SUCCESS || id <= 0 ||
                static_cast<uint64_t>(id) > asr_streams.size() ||
                !asr_streams[static_cast<size_t>(id - 1)].active) {
                send_frame(handles.response_write, lifecycle_error(command, 1, "unknown ASR stream ID"));
                continue;
            }
            asr_streams[static_cast<size_t>(id - 1)].active = false;
            response.header.payload_json = "{\"ok\":true}";
            response.attachment.clear();
        }
        if (is_stream_generation_method(command.header.method)) {
            if (!initialized) {
                send_frame(handles.response_write,
                           lifecycle_error(command, 1, "engine is not initialized"));
                continue;
            }
            size_t event_count = 0;
            auto send_stream_event = [&](const aila::ipc::Frame& value) {
                send_frame(handles.event_write, value);
                ++event_count;
            };
            if (command.header.method == "generate.chat_json_stream_ex") {
                aila::ipc::Frame structured = stream_event(
                    command,
                    u8R"({"event":"structured","structSize":64,"type":2,"text":"工具","toolCallId":null,"toolName":"search","argumentsDelta":"","finishReason":null,"warningsJson":"[]","toolCallsJson":null})");
                if (command.header.payload_json.find("__aila_stream_bad_type__") != std::string::npos) {
                    structured.header.payload_json =
                        R"({"event":"structured","structSize":64,"type":99,"text":null,"toolCallId":null,"toolName":null,"argumentsDelta":null,"finishReason":null,"warningsJson":null,"toolCallsJson":null})";
                }
                send_stream_event(structured);
            } else {
                aila::ipc::Frame first = stream_event(
                    command, R"({"event":"token","byteCount":5})", "first");
                if (command.header.payload_json.find("__aila_stream_bad_byte_count__") != std::string::npos) {
                    first.header.payload_json = R"({"event":"token","byteCount":6})";
                } else if (command.header.payload_json.find("__aila_stream_bad_utf8__") != std::string::npos) {
                    const std::string invalid("\xc3\x28", 2);
                    first = stream_event(command, R"({"event":"token","byteCount":2})", invalid);
                } else if (command.header.payload_json.find("__aila_stream_bad_identity__") != std::string::npos) {
                    first.header.method = "wrong.stream";
                } else if (command.header.payload_json.find("__aila_stream_bad_request_id__") != std::string::npos) {
                    ++first.header.request_id;
                } else if (command.header.payload_json.find("__aila_stream_bad_protocol__") != std::string::npos) {
                    ++first.header.protocol;
                } else if (command.header.payload_json.find("__aila_stream_bad_nul__") != std::string::npos) {
                    first = stream_event(
                        command, R"({"event":"token","byteCount":3})",
                        std::string_view("a\0b", 3));
                } else if (command.header.payload_json.find("__aila_stream_bad_schema__") != std::string::npos) {
                    first.header.payload_json = R"({"event":"token"})";
                }
                send_stream_event(first);
            }

            if (command.header.payload_json.find("__aila_stream_early_exit__") != std::string::npos) {
                ExitProcess(91);
            }

            const bool response_before_end =
                command.header.payload_json.find("__aila_stream_response_before_end__") !=
                std::string::npos;
            if (response_before_end) {
                response.header.payload_json = "{\"status\":0,\"eventCount\":3}";
                response.attachment.clear();
                send_frame(handles.response_write, response);
            }

            const bool short_abort_race =
                command.header.payload_json.find("__aila_stream_short_abort__") !=
                std::string::npos;
            if (!short_abort_race) {
                const bool slow_response =
                    command.header.payload_json.find("__aila_stream_slow_response__") !=
                    std::string::npos;
                Sleep(slow_response ? 250 : 30);
            }
            bool cancelled = false;
            DWORD pending = 0;
            if (!short_abort_race &&
                PeekNamedPipe(handles.command_read, nullptr, 0, nullptr, &pending, nullptr) &&
                pending != 0) {
                aila::ipc::Frame control;
                if (aila::ipc::read_frame(handles.command_read, control, error) &&
                    control.header.method == "cancel" &&
                    control.header.request_id == command.header.request_id) {
                    cancelled = true;
                }
            }
            if (!cancelled && command.header.method != "generate.chat_json_stream_ex") {
                send_stream_event(
                    stream_event(command, R"({"event":"token","byteCount":7})", u8" 第二"));
            }
            const uint64_t terminal_count = static_cast<uint64_t>(event_count + 1);
            send_stream_event(stream_event(
                command,
                std::string("{\"event\":\"end\",\"eventCount\":") +
                    std::to_string(terminal_count) + "}"));
            if (command.header.payload_json.find("__aila_stream_duplicate_end__") !=
                std::string::npos) {
                send_stream_event(stream_event(
                    command,
                    std::string("{\"event\":\"end\",\"eventCount\":") +
                        std::to_string(event_count + 1) + "}"));
            }
            if (command.header.payload_json.find("__aila_stream_post_end_data__") !=
                    std::string::npos ||
                command.header.payload_json.find("__aila_stream_error_post_end__") !=
                    std::string::npos) {
                send_stream_event(
                    stream_event(command, R"({"event":"token","byteCount":4})", "late"));
            }
            if (response_before_end) {
                continue;
            }
            if (command.header.payload_json.find("__aila_stream_engine_error__") !=
                    std::string::npos ||
                command.header.payload_json.find("__aila_stream_error_") !=
                    std::string::npos) {
                response.header.kind = "error";
                if (command.header.payload_json.find("__aila_stream_error_missing_count__") !=
                    std::string::npos) {
                    response.header.payload_json =
                        R"({"code":5,"message":"synthetic stream failure"})";
                } else if (command.header.payload_json.find("__aila_stream_error_bad_count_type__") !=
                           std::string::npos) {
                    response.header.payload_json =
                        R"({"code":5,"message":"synthetic stream failure","eventCount":"bad"})";
                } else if (command.header.payload_json.find("__aila_stream_error_oversized_count__") !=
                           std::string::npos) {
                    response.header.payload_json =
                        R"({"code":5,"message":"synthetic stream failure","eventCount":1000001})";
                } else if (command.header.payload_json.find("__aila_stream_error_zero_count__") !=
                           std::string::npos) {
                    response.header.payload_json =
                        R"({"code":5,"message":"synthetic stream failure","eventCount":0})";
                } else {
                    response.header.payload_json =
                        std::string("{\"code\":5,\"message\":\"synthetic stream failure\",\"eventCount\":") +
                        std::to_string(event_count) + "}";
                }
                response.attachment.clear();
                send_frame(handles.response_write, response);
                continue;
            }
            response.header.payload_json =
                std::string("{\"status\":") +
                ((cancelled || command.header.payload_json.find(
                    "__aila_stream_unsolicited_abort__") != std::string::npos) ? "1" : "0") +
                ",\"eventCount\":" + std::to_string(event_count) + "}";
            response.attachment.clear();
            send_frame(handles.response_write, response);
            continue;
        }
        if (is_generation_method(command.header.method)) {
            if (!initialized) {
                send_frame(
                    handles.response_write,
                    lifecycle_error(command, 1, "engine is not initialized"));
                continue;
            }
            if (command.header.payload_json.find("__aila_error__") != std::string::npos) {
                send_frame(
                    handles.response_write,
                    lifecycle_error(command, 5, "synthetic generation failure"));
                continue;
            }
            if (command.header.payload_json.find("__aila_malformed_attachment__") !=
                std::string::npos) {
                const std::string malformed("left\0right", 10);
                attach_text(response, malformed);
            } else if (command.header.payload_json.find("__aila_invalid_utf8__") !=
                       std::string::npos) {
                const std::string invalid_utf8("\xc3\x28", 2);
                attach_text(response, invalid_utf8);
            } else if (command.header.payload_json.find("__aila_empty__") !=
                       std::string::npos) {
                attach_text(response, {});
            } else {
                const std::string generated =
                    std::string("{\"method\":") + json_string(command.header.method) +
                    ",\"request\":" + command.header.payload_json + "}";
                attach_text(response, generated);
            }
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
