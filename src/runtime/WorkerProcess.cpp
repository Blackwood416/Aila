#include "runtime/WorkerProcess.hpp"

#include "ipc/Win32Pipe.hpp"
#include "runtime/ChildEnvironment.hpp"

#include "simdjson.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace aila::runtime {

bool detail::worker_file_attributes_are_safe(DWORD attributes) noexcept {
    return attributes != INVALID_FILE_ATTRIBUTES &&
        (attributes & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT)) == 0;
}

bool detail::runtime_component_attributes_are_safe(DWORD attributes) noexcept {
    return attributes != INVALID_FILE_ATTRIBUTES &&
        (attributes & FILE_ATTRIBUTE_DIRECTORY) != 0 &&
        (attributes & FILE_ATTRIBUTE_REPARSE_POINT) == 0;
}

namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

class UniqueHandle {
public:
    UniqueHandle() = default;
    explicit UniqueHandle(HANDLE handle) noexcept : handle_(handle) {}
    ~UniqueHandle() { reset(); }

    UniqueHandle(const UniqueHandle&) = delete;
    UniqueHandle& operator=(const UniqueHandle&) = delete;

    UniqueHandle(UniqueHandle&& other) noexcept : handle_(other.release()) {}
    UniqueHandle& operator=(UniqueHandle&& other) noexcept {
        if (this != &other) {
            reset(other.release());
        }
        return *this;
    }

    HANDLE get() const noexcept { return handle_; }
    explicit operator bool() const noexcept {
        return handle_ != nullptr && handle_ != INVALID_HANDLE_VALUE;
    }

    HANDLE release() noexcept {
        HANDLE result = handle_;
        handle_ = nullptr;
        return result;
    }

    void reset(HANDLE handle = nullptr) noexcept {
        if (*this) {
            CloseHandle(handle_);
        }
        handle_ = handle;
    }

private:
    HANDLE handle_ = nullptr;
};

class AttributeList {
public:
    AttributeList() {
        SIZE_T bytes = 0;
        InitializeProcThreadAttributeList(nullptr, 1, 0, &bytes);
        if (bytes == 0) {
            throw std::runtime_error("InitializeProcThreadAttributeList size query failed");
        }
        storage_.resize(bytes);
        list_ = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(storage_.data());
        if (InitializeProcThreadAttributeList(list_, 1, 0, &bytes) == FALSE) {
            list_ = nullptr;
            throw std::runtime_error("InitializeProcThreadAttributeList failed");
        }
    }

    ~AttributeList() {
        if (list_ != nullptr) {
            DeleteProcThreadAttributeList(list_);
        }
    }

    AttributeList(const AttributeList&) = delete;
    AttributeList& operator=(const AttributeList&) = delete;

    LPPROC_THREAD_ATTRIBUTE_LIST get() const noexcept { return list_; }

private:
    std::vector<std::byte> storage_;
    LPPROC_THREAD_ATTRIBUTE_LIST list_ = nullptr;
};

std::string utf8(const std::wstring& value) {
    if (value.empty()) {
        return {};
    }
    if (value.size() > static_cast<size_t>((std::numeric_limits<int>::max)())) {
        throw std::runtime_error("path is too long to encode as UTF-8");
    }
    const int input_size = static_cast<int>(value.size());
    const int output_size = WideCharToMultiByte(
        CP_UTF8,
        WC_ERR_INVALID_CHARS,
        value.data(),
        input_size,
        nullptr,
        0,
        nullptr,
        nullptr);
    if (output_size <= 0) {
        throw std::runtime_error("could not encode path as UTF-8");
    }
    std::string result(static_cast<size_t>(output_size), '\0');
    if (WideCharToMultiByte(
            CP_UTF8,
            WC_ERR_INVALID_CHARS,
            value.data(),
            input_size,
            result.data(),
            output_size,
            nullptr,
            nullptr) != output_size) {
        throw std::runtime_error("could not encode path as UTF-8");
    }
    return result;
}

std::string win32_error(const char* operation, DWORD code = GetLastError()) {
    return std::string(operation) + " failed with Win32 error " + std::to_string(code);
}

fs::path normalized_absolute(const fs::path& path) {
    return fs::absolute(path).lexically_normal();
}

bool paths_equal_case_insensitively(const fs::path& left, const fs::path& right) {
    const std::wstring left_value = left.lexically_normal().wstring();
    const std::wstring right_value = right.lexically_normal().wstring();
    if (left_value.size() > static_cast<size_t>((std::numeric_limits<int>::max)()) ||
        right_value.size() > static_cast<size_t>((std::numeric_limits<int>::max)())) {
        return false;
    }
    return CompareStringOrdinal(
               left_value.data(),
               static_cast<int>(left_value.size()),
               right_value.data(),
               static_cast<int>(right_value.size()),
               TRUE) == CSTR_EQUAL;
}

fs::path remove_extended_path_prefix(std::wstring value) {
    constexpr std::wstring_view unc_prefix = L"\\\\?\\UNC\\";
    constexpr std::wstring_view local_prefix = L"\\\\?\\";
    if (value.compare(0, unc_prefix.size(), unc_prefix) == 0) {
        value = L"\\\\" + value.substr(unc_prefix.size());
    } else if (value.compare(0, local_prefix.size(), local_prefix) == 0) {
        value.erase(0, local_prefix.size());
    }
    return fs::path(std::move(value)).lexically_normal();
}

fs::path final_path_from_handle(HANDLE handle, const char* context) {
    std::vector<wchar_t> buffer(MAX_PATH);
    for (;;) {
        const DWORD copied = GetFinalPathNameByHandleW(
            handle,
            buffer.data(),
            static_cast<DWORD>(buffer.size()),
            FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
        if (copied == 0) {
            throw std::runtime_error(win32_error(context));
        }
        if (copied < buffer.size()) {
            return remove_extended_path_prefix(std::wstring(buffer.data(), copied));
        }
        buffer.resize(static_cast<size_t>(copied) + 1);
    }
}

DWORD attributes_from_handle(HANDLE handle, const char* context) {
    BY_HANDLE_FILE_INFORMATION information{};
    if (GetFileInformationByHandle(handle, &information) == FALSE) {
        throw std::runtime_error(win32_error(context));
    }
    return information.dwFileAttributes;
}

void validate_runtime_components(const fs::path& runtime_directory) {
    fs::path current = runtime_directory.root_path();
    auto validate = [&](const fs::path& component) {
        const DWORD attributes = GetFileAttributesW(component.c_str());
        if (!detail::runtime_component_attributes_are_safe(attributes)) {
            const std::string reason =
                attributes != INVALID_FILE_ATTRIBUTES &&
                    (attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0
                ? "runtime path contains a reparse point: '"
                : "runtime path component is not an ordinary directory: '";
            throw std::runtime_error(reason + utf8(component.wstring()) + "'");
        }
    };

    if (current.empty()) {
        throw std::runtime_error("worker runtime directory must be absolute");
    }
    validate(current);
    for (const fs::path& component : runtime_directory.relative_path()) {
        current /= component;
        validate(current);
    }
}

UniqueHandle open_validation_handle(
    const fs::path& path,
    DWORD flags,
    const char* context) {
    UniqueHandle handle(CreateFileW(
        path.c_str(),
        FILE_READ_ATTRIBUTES,
        FILE_SHARE_READ,
        nullptr,
        OPEN_EXISTING,
        flags,
        nullptr));
    if (!handle) {
        throw std::runtime_error(win32_error(context));
    }
    return handle;
}

struct ValidatedLaunchPaths {
    fs::path runtime_directory;
    fs::path worker_executable;
    UniqueHandle runtime_lock;
    UniqueHandle worker_lock;

    void release_locks() noexcept {
        worker_lock.reset();
        runtime_lock.reset();
    }
};

ValidatedLaunchPaths validate_launch_paths(
    const fs::path& runtime_directory,
    const fs::path& worker_executable) {
    const fs::path normalized_runtime = normalized_absolute(runtime_directory);
    const fs::path normalized_worker = normalized_absolute(worker_executable);
    validate_runtime_components(normalized_runtime);
    if (!paths_equal_case_insensitively(
            normalized_worker.parent_path(),
            normalized_runtime)) {
        throw std::runtime_error(
            "worker executable must reside directly in the isolated runtime directory");
    }

    ValidatedLaunchPaths validated;
    validated.runtime_lock = open_validation_handle(
        normalized_runtime,
        FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT,
        "CreateFileW(runtime validation)");
    if (!detail::runtime_component_attributes_are_safe(
            attributes_from_handle(
                validated.runtime_lock.get(),
                "GetFileInformationByHandle(runtime validation)"))) {
        throw std::runtime_error("worker runtime directory handle resolved to a reparse point");
    }

    const DWORD worker_attributes = GetFileAttributesW(normalized_worker.c_str());
    if (!detail::worker_file_attributes_are_safe(worker_attributes)) {
        const std::string reason =
            worker_attributes != INVALID_FILE_ATTRIBUTES &&
                (worker_attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0
            ? "worker executable must not be a reparse point: '"
            : "worker executable is not a contained regular file: '";
        throw std::runtime_error(reason + utf8(normalized_worker.wstring()) + "'");
    }
    validated.worker_lock = open_validation_handle(
        normalized_worker,
        FILE_FLAG_OPEN_REPARSE_POINT,
        "CreateFileW(worker validation)");
    if (!detail::worker_file_attributes_are_safe(
            attributes_from_handle(
                validated.worker_lock.get(),
                "GetFileInformationByHandle(worker validation)"))) {
        throw std::runtime_error("worker executable handle resolved to a reparse point");
    }

    validated.runtime_directory = final_path_from_handle(
        validated.runtime_lock.get(),
        "GetFinalPathNameByHandleW(runtime validation)");
    validated.worker_executable = final_path_from_handle(
        validated.worker_lock.get(),
        "GetFinalPathNameByHandleW(worker validation)");
    validate_runtime_components(validated.runtime_directory);
    if (!paths_equal_case_insensitively(
            validated.worker_executable.parent_path(),
            validated.runtime_directory)) {
        throw std::runtime_error(
            "worker executable handle did not resolve inside the validated runtime directory");
    }
    return validated;
}

fs::path process_image_path(HANDLE process) {
    std::vector<wchar_t> buffer(MAX_PATH);
    for (;;) {
        DWORD size = static_cast<DWORD>(buffer.size());
        if (QueryFullProcessImageNameW(process, 0, buffer.data(), &size) != FALSE) {
            return remove_extended_path_prefix(std::wstring(buffer.data(), size));
        }
        if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
            throw std::runtime_error(win32_error("QueryFullProcessImageNameW"));
        }
        buffer.resize(buffer.size() * 2);
    }
}

void terminate_and_wait(HANDLE process, DWORD exit_code) noexcept {
    (void)TerminateProcess(process, exit_code);
    (void)WaitForSingleObject(process, 2000);
}

std::wstring quote_argument(std::wstring_view argument) {
    std::wstring quoted = L"\"";
    size_t backslashes = 0;
    for (const wchar_t character : argument) {
        if (character == L'\\') {
            ++backslashes;
            continue;
        }
        if (character == L'"') {
            quoted.append(backslashes * 2 + 1, L'\\');
            quoted.push_back(L'"');
        } else {
            quoted.append(backslashes, L'\\');
            quoted.push_back(character);
        }
        backslashes = 0;
    }
    quoted.append(backslashes * 2, L'\\');
    quoted.push_back(L'"');
    return quoted;
}

void append_argument(std::wstring& command_line, std::wstring_view argument) {
    if (!command_line.empty()) {
        command_line.push_back(L' ');
    }
    command_line += quote_argument(argument);
}

std::wstring handle_string(HANDLE handle) {
    return std::to_wstring(reinterpret_cast<uintptr_t>(handle));
}

void make_pipe(UniqueHandle& read, UniqueHandle& write) {
    SECURITY_ATTRIBUTES security{};
    security.nLength = sizeof(security);
    security.bInheritHandle = TRUE;
    HANDLE read_handle = nullptr;
    HANDLE write_handle = nullptr;
    if (CreatePipe(&read_handle, &write_handle, &security, 0) == FALSE) {
        throw std::runtime_error(win32_error("CreatePipe"));
    }
    read.reset(read_handle);
    write.reset(write_handle);
}

void make_parent_end_non_inheritable(HANDLE handle) {
    if (SetHandleInformation(handle, HANDLE_FLAG_INHERIT, 0) == FALSE) {
        throw std::runtime_error(win32_error("SetHandleInformation"));
    }
}

DWORD bounded_wait_milliseconds(Clock::time_point deadline, DWORD maximum) {
    const Clock::time_point now = Clock::now();
    if (now >= deadline) {
        return 0;
    }
    const auto remaining =
        std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
    const uint64_t count = static_cast<uint64_t>((std::max)(remaining.count(), int64_t{1}));
    return static_cast<DWORD>((std::min)(count, static_cast<uint64_t>(maximum)));
}

struct HandshakePayload {
    std::string build_id;
    std::string executable;
    std::string runtime_directory;
    std::string current_directory;
    std::string path;
};

HandshakePayload parse_handshake_payload(const std::string& payload_json) {
    simdjson::dom::parser parser;
    simdjson::dom::element root;
    const simdjson::error_code parse_error = parser.parse(payload_json).get(root);
    if (parse_error != simdjson::SUCCESS) {
        throw std::runtime_error(
            std::string("worker handshake payload is invalid JSON: ") +
            simdjson::error_message(parse_error));
    }

    auto required_string = [&](const char* field) {
        std::string_view value;
        if (root[field].get_string().get(value) != simdjson::SUCCESS) {
            throw std::runtime_error(
                std::string("worker handshake payload requires string field '") + field + "'");
        }
        return std::string(value);
    };

    return {
        required_string("buildId"),
        required_string("executable"),
        required_string("runtimeDirectory"),
        required_string("currentDirectory"),
        required_string("path"),
    };
}

} // namespace

struct WorkerProcess::Impl {
    mutable std::mutex state_mutex;
    std::mutex operation_mutex;
    UniqueHandle command_write;
    UniqueHandle response_read;
    UniqueHandle event_read;
    UniqueHandle process;
    std::thread event_thread;
    std::mutex event_mutex;
    std::condition_variable event_condition;
    std::vector<ipc::Frame> events;
    bool stop_event_thread = false;
    std::string event_transport_error;
    ExpectedHandshake expected;
    mutable std::optional<DWORD> cached_exit_code;
    std::string error;

    void set_error(std::string message) {
        std::lock_guard<std::mutex> lock(state_mutex);
        error = std::move(message);
    }

    std::optional<DWORD> query_exit_code_unlocked() const {
        if (cached_exit_code.has_value()) {
            return cached_exit_code;
        }
        if (!process) {
            return std::nullopt;
        }
        DWORD code = 0;
        if (GetExitCodeProcess(process.get(), &code) == FALSE) {
            return std::nullopt;
        }
        if (code == STILL_ACTIVE) {
            return std::nullopt;
        }
        cached_exit_code = code;
        return code;
    }

    std::optional<DWORD> query_exit_code() const {
        std::lock_guard<std::mutex> lock(state_mutex);
        return query_exit_code_unlocked();
    }

    bool is_active() const {
        std::lock_guard<std::mutex> lock(state_mutex);
        if (!process) {
            return false;
        }
        DWORD code = 0;
        if (GetExitCodeProcess(process.get(), &code) == FALSE) {
            return false;
        }
        if (code != STILL_ACTIVE) {
            cached_exit_code = code;
            return false;
        }
        return true;
    }

    std::string exit_diagnostic(std::string prefix, DWORD wait_milliseconds = 0) {
        HANDLE process_handle = nullptr;
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            process_handle = process.get();
        }
        if (process_handle != nullptr && wait_milliseconds != 0) {
            WaitForSingleObject(process_handle, wait_milliseconds);
        }
        if (const std::optional<DWORD> code = query_exit_code()) {
            prefix += "; worker exited with code " + std::to_string(*code);
        }
        return prefix;
    }

    [[noreturn]] void fail(std::string message) {
        {
            std::lock_guard<std::mutex> lock(event_mutex);
            if (!event_transport_error.empty()) {
                message = event_transport_error + "; " + message;
            }
        }
        message = exit_diagnostic(std::move(message), 200);
        set_error(message);
        throw std::runtime_error(message);
    }

    static bool is_unrelated_log(const ipc::Frame& frame) noexcept {
        return frame.header.kind == "event" && frame.header.method == "log" &&
            frame.header.request_id == 0;
    }

    void fail_event_transport(std::string message) noexcept {
        try {
            {
                std::lock_guard<std::mutex> lock(event_mutex);
                event_transport_error = message;
                stop_event_thread = true;
            }
            set_error(message);
        } catch (...) {
        }
        terminate_active_process(ERROR_BAD_FORMAT);
        event_condition.notify_all();
    }

    void event_loop() noexcept {
        try {
            constexpr size_t kMaxQueuedEvents = 1024;
            for (;;) {
                ipc::Frame frame;
                std::string read_error;
                if (!ipc::read_frame(event_read.get(), frame, read_error)) {
                    return;
                }
                if (frame.header.protocol != expected.protocol ||
                    frame.header.abi != expected.abi) {
                    fail_event_transport(
                        "worker event frame protocol or ABI did not match the handshake");
                    return;
                }

                std::unique_lock<std::mutex> lock(event_mutex);
                if (stop_event_thread) {
                    return;
                }
                while (events.size() >= kMaxQueuedEvents) {
                    const auto log = std::find_if(
                        events.begin(), events.end(),
                        [](const ipc::Frame& queued) { return is_unrelated_log(queued); });
                    if (log != events.end()) {
                        events.erase(log);
                        break;
                    }
                    if (is_unrelated_log(frame)) {
                        lock.unlock();
                        event_condition.notify_all();
                        break;
                    }
                    event_condition.wait(lock, [&] {
                        return stop_event_thread || events.size() < kMaxQueuedEvents;
                    });
                    if (stop_event_thread) {
                        return;
                    }
                }
                if (lock.owns_lock() && events.size() < kMaxQueuedEvents) {
                    events.push_back(std::move(frame));
                    lock.unlock();
                    event_condition.notify_all();
                }
            }
        } catch (const std::exception& exception) {
            fail_event_transport(
                std::string("worker event transport failed: ") + exception.what());
        } catch (...) {
            fail_event_transport("worker event transport failed with an unknown exception");
        }
    }

    struct IoState {
        std::mutex mutex;
        std::condition_variable condition;
        bool done = false;
        bool succeeded = false;
        std::string error;
        ipc::Frame frame;
    };

    struct IoOutcome {
        bool succeeded = false;
        bool timed_out = false;
        bool process_exited = false;
        std::string error;
        ipc::Frame frame;
    };

    void terminate_active_process(DWORD code) noexcept {
        HANDLE process_handle = nullptr;
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            process_handle = process.get();
        }
        if (process_handle == nullptr) {
            return;
        }
        DWORD current_code = 0;
        if (GetExitCodeProcess(process_handle, &current_code) != FALSE &&
            current_code == STILL_ACTIVE) {
            (void)TerminateProcess(process_handle, code);
        }
    }

    template <typename Operation>
    IoOutcome run_cancellable_io(Operation operation, Clock::time_point deadline) {
        const auto state = std::make_shared<IoState>();
        std::thread io_thread([state, operation = std::move(operation)]() mutable {
            ipc::Frame frame;
            std::string error;
            bool succeeded = false;
            try {
                succeeded = operation(frame, error);
            } catch (const std::exception& exception) {
                error = std::string("pipe operation threw: ") + exception.what();
            } catch (...) {
                error = "pipe operation threw an unknown exception";
            }
            {
                std::lock_guard<std::mutex> lock(state->mutex);
                state->succeeded = succeeded;
                state->error = std::move(error);
                state->frame = std::move(frame);
                state->done = true;
            }
            state->condition.notify_all();
        });

        bool timed_out = false;
        bool process_exited = false;
        for (;;) {
            std::unique_lock<std::mutex> lock(state->mutex);
            if (state->done) {
                break;
            }
            const Clock::time_point now = Clock::now();
            if (now >= deadline) {
                timed_out = true;
                break;
            }
            const Clock::time_point wake = (std::min)(deadline, now + std::chrono::milliseconds(10));
            if (state->condition.wait_until(lock, wake, [&] { return state->done; })) {
                break;
            }
            lock.unlock();

            if (!is_active()) {
                std::unique_lock<std::mutex> completion_lock(state->mutex);
                if (!state->condition.wait_for(
                        completion_lock,
                        std::chrono::milliseconds(20),
                        [&] { return state->done; })) {
                    process_exited = true;
                }
                break;
            }
        }

        if (timed_out || process_exited) {
            (void)CancelSynchronousIo(io_thread.native_handle());
            if (timed_out) {
                terminate_active_process(ERROR_TIMEOUT);
            }
        }
        io_thread.join();

        std::lock_guard<std::mutex> lock(state->mutex);
        return {
            state->succeeded,
            timed_out,
            process_exited,
            std::move(state->error),
            std::move(state->frame),
        };
    }

    [[noreturn]] void timeout_failure(std::string context) {
        cleanup(true, ERROR_TIMEOUT);
        context += " timed out";
        if (const std::optional<DWORD> code = query_exit_code()) {
            context += "; worker exited with code " + std::to_string(*code);
        }
        set_error(context);
        throw std::runtime_error(context);
    }

    void write_request(
        const ipc::Frame& frame,
        Clock::time_point deadline,
        const char* context) {
        const HANDLE handle = command_write.get();
        IoOutcome outcome = run_cancellable_io(
            [handle, &frame](ipc::Frame&, std::string& error) {
                return ipc::write_frame(handle, frame, error);
            },
            deadline);
        if (outcome.timed_out) {
            timeout_failure(std::string(context) + " while writing");
        }
        if (outcome.process_exited) {
            fail(std::string(context) + ": worker exited during request write");
        }
        if (!outcome.succeeded) {
            fail(std::string(context) + ": " + outcome.error);
        }
    }

    ipc::Frame read_response(
        Clock::time_point deadline,
        const char* context,
        bool cleanup_on_timeout = true) {
        const HANDLE handle = response_read.get();
        IoOutcome outcome = run_cancellable_io(
            [handle](ipc::Frame& frame, std::string& error) {
                return ipc::read_frame(handle, frame, error);
            },
            deadline);
        if (outcome.timed_out) {
            if (cleanup_on_timeout) {
                timeout_failure(std::string(context) + " while reading");
            }
            const std::string message = std::string(context) + " while reading timed out";
            set_error(message);
            throw std::runtime_error(message);
        }
        if (outcome.process_exited) {
            fail(std::string(context) + ": worker exited before completing its response");
        }
        if (!outcome.succeeded) {
            fail(std::string(context) + ": " + outcome.error);
        }
        return std::move(outcome.frame);
    }

    void validate_handshake(
        const ipc::Frame& frame,
        const fs::path& runtime_directory,
        const fs::path& worker_executable,
        const fs::path& system_root) {
        if (frame.header.kind != "handshake") {
            throw std::runtime_error(
                "worker handshake kind mismatch: expected 'handshake', received '" +
                frame.header.kind + "'");
        }
        if (frame.header.request_id != 0) {
            throw std::runtime_error("worker handshake request ID must be zero");
        }
        if (frame.header.protocol != expected.protocol) {
            throw std::runtime_error(
                "worker handshake protocol mismatch: expected " +
                std::to_string(expected.protocol) + ", received " +
                std::to_string(frame.header.protocol));
        }
        if (frame.header.abi != expected.abi) {
            throw std::runtime_error(
                "worker handshake ABI mismatch: expected " + std::to_string(expected.abi) +
                ", received " + std::to_string(frame.header.abi));
        }

        const HandshakePayload payload = parse_handshake_payload(frame.header.payload_json);
        if (payload.build_id != expected.build_id) {
            throw std::runtime_error(
                "worker handshake build ID mismatch: expected '" + expected.build_id +
                "', received '" + payload.build_id + "'");
        }
        const std::string expected_executable = utf8(worker_executable.wstring());
        const std::string expected_runtime = utf8(runtime_directory.wstring());
        const std::string expected_path = utf8(
            runtime_directory.wstring() + L";" +
            (system_root / L"System32").lexically_normal().wstring() + L";" +
            system_root.wstring());
        if (payload.executable != expected_executable) {
            throw std::runtime_error(
                "worker handshake executable mismatch: expected '" + expected_executable +
                "', received '" + payload.executable + "'");
        }
        if (payload.runtime_directory != expected_runtime) {
            throw std::runtime_error(
                "worker handshake runtime directory mismatch: expected '" + expected_runtime +
                "', received '" + payload.runtime_directory + "'");
        }
        if (payload.current_directory != expected_runtime) {
            throw std::runtime_error(
                "worker handshake current directory mismatch: expected '" + expected_runtime +
                "', received '" + payload.current_directory + "'");
        }
        if (payload.path != expected_path) {
            throw std::runtime_error(
                "worker handshake PATH mismatch: expected '" + expected_path +
                "', received '" + payload.path + "'");
        }
    }

    void cleanup(bool force, DWORD forced_exit_code) noexcept {
        command_write.reset();

        {
            std::lock_guard<std::mutex> lock(event_mutex);
            stop_event_thread = true;
        }
        event_condition.notify_all();

        HANDLE process_handle = nullptr;
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            process_handle = process.get();
        }
        if (process_handle != nullptr) {
            DWORD code = 0;
            if (GetExitCodeProcess(process_handle, &code) != FALSE && code == STILL_ACTIVE && force) {
                TerminateProcess(process_handle, forced_exit_code);
            }
            WaitForSingleObject(process_handle, 2000);
            if (GetExitCodeProcess(process_handle, &code) != FALSE && code != STILL_ACTIVE) {
                std::lock_guard<std::mutex> lock(state_mutex);
                cached_exit_code = code;
            }
        }

        if (event_thread.joinable()) {
            (void)CancelSynchronousIo(event_thread.native_handle());
            event_thread.join();
        }
        response_read.reset();
        event_read.reset();
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            process.reset();
        }
    }
};

WorkerProcess::WorkerProcess() : impl_(std::make_unique<Impl>()) {}

WorkerProcess::~WorkerProcess() noexcept {
    try {
        shutdown(std::chrono::seconds(2));
    } catch (...) {
    }
}

void WorkerProcess::start(
    const fs::path& runtime_directory,
    const fs::path& worker_executable,
    const ExpectedHandshake& expected,
    std::chrono::milliseconds handshake_timeout) {
    std::lock_guard<std::mutex> operation_lock(impl_->operation_mutex);
    {
        std::lock_guard<std::mutex> state_lock(impl_->state_mutex);
        if (impl_->process) {
            impl_->error = "worker process has already been started";
            throw std::runtime_error(impl_->error);
        }
        impl_->error.clear();
        impl_->cached_exit_code.reset();
    }

    try {
        if (handshake_timeout.count() <= 0) {
            throw std::runtime_error("worker handshake timeout must be positive");
        }
        ValidatedLaunchPaths validated =
            validate_launch_paths(runtime_directory, worker_executable);
        const fs::path& normalized_runtime = validated.runtime_directory;
        const fs::path& normalized_worker = validated.worker_executable;

        const fs::path system_root = system_root_directory();
        std::vector<wchar_t> environment = build_isolated_environment(
            current_environment(),
            normalized_runtime,
            system_root);

        UniqueHandle command_read;
        UniqueHandle command_write;
        UniqueHandle response_read;
        UniqueHandle response_write;
        UniqueHandle event_read;
        UniqueHandle event_write;
        make_pipe(command_read, command_write);
        make_pipe(response_read, response_write);
        make_pipe(event_read, event_write);
        make_parent_end_non_inheritable(command_write.get());
        make_parent_end_non_inheritable(response_read.get());
        make_parent_end_non_inheritable(event_read.get());

        std::array<HANDLE, 3> inherited_handles{
            command_read.get(),
            response_write.get(),
            event_write.get(),
        };
        AttributeList attributes;
        if (UpdateProcThreadAttribute(
                attributes.get(),
                0,
                PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                inherited_handles.data(),
                sizeof(inherited_handles),
                nullptr,
                nullptr) == FALSE) {
            throw std::runtime_error(win32_error("UpdateProcThreadAttribute"));
        }

        std::wstring command_line;
        append_argument(command_line, normalized_worker.wstring());
        append_argument(command_line, L"--ffi-worker");
        append_argument(command_line, L"--command-read-handle");
        append_argument(command_line, handle_string(command_read.get()));
        append_argument(command_line, L"--response-write-handle");
        append_argument(command_line, handle_string(response_write.get()));
        append_argument(command_line, L"--event-write-handle");
        append_argument(command_line, handle_string(event_write.get()));
        std::vector<wchar_t> mutable_command(command_line.begin(), command_line.end());
        mutable_command.push_back(L'\0');

        STARTUPINFOEXW startup{};
        startup.StartupInfo.cb = sizeof(startup);
        startup.lpAttributeList = attributes.get();
        PROCESS_INFORMATION process_information{};
        const DWORD flags =
            EXTENDED_STARTUPINFO_PRESENT | CREATE_UNICODE_ENVIRONMENT | CREATE_NO_WINDOW |
            CREATE_SUSPENDED;
        if (CreateProcessW(
                normalized_worker.c_str(),
                mutable_command.data(),
                nullptr,
                nullptr,
                TRUE,
                flags,
                environment.data(),
                normalized_runtime.c_str(),
                &startup.StartupInfo,
                &process_information) == FALSE) {
            throw std::runtime_error(win32_error("CreateProcessW"));
        }

        UniqueHandle child_process(process_information.hProcess);
        UniqueHandle child_thread(process_information.hThread);
        command_read.reset();
        response_write.reset();
        event_write.reset();

        try {
            const fs::path launched_image = process_image_path(child_process.get());
            if (!paths_equal_case_insensitively(launched_image, normalized_worker)) {
                throw std::runtime_error(
                    "suspended worker image mismatch: expected '" +
                    utf8(normalized_worker.wstring()) + "', received '" +
                    utf8(launched_image.wstring()) + "'");
            }
            if (ResumeThread(child_thread.get()) == (std::numeric_limits<DWORD>::max)()) {
                throw std::runtime_error(win32_error("ResumeThread"));
            }
            validated.release_locks();
        } catch (...) {
            terminate_and_wait(child_process.get(), ERROR_BAD_EXE_FORMAT);
            throw;
        }

        impl_->command_write = std::move(command_write);
        impl_->response_read = std::move(response_read);
        impl_->event_read = std::move(event_read);
        impl_->expected = expected;
        {
            std::lock_guard<std::mutex> event_lock(impl_->event_mutex);
            impl_->events.clear();
            impl_->stop_event_thread = false;
            impl_->event_transport_error.clear();
        }
        {
            std::lock_guard<std::mutex> state_lock(impl_->state_mutex);
            impl_->process = std::move(child_process);
        }
        impl_->event_thread = std::thread([implementation = impl_.get()] {
            implementation->event_loop();
        });

        const ipc::Frame handshake = impl_->read_response(
            Clock::now() + handshake_timeout,
            "worker handshake");
        impl_->validate_handshake(
            handshake,
            normalized_runtime,
            normalized_worker,
            system_root);
    } catch (const std::exception& exception) {
        const std::string message = exception.what();
        impl_->set_error(message);
        impl_->cleanup(true, ERROR_BAD_FORMAT);
        throw std::runtime_error(message);
    }
}

ipc::Frame WorkerProcess::request(
    const ipc::Frame& frame,
    std::chrono::milliseconds timeout) {
    std::lock_guard<std::mutex> operation_lock(impl_->operation_mutex);
    if (timeout.count() <= 0) {
        impl_->fail("worker request timeout must be positive");
    }
    if (!impl_->is_active()) {
        impl_->fail("cannot send a request because the worker is not healthy");
    }

    const Clock::time_point deadline = Clock::now() + timeout;
    impl_->write_request(frame, deadline, "worker request");
    const ipc::Frame response = impl_->read_response(deadline, "worker request");
    if (response.header.protocol != impl_->expected.protocol) {
        impl_->fail("worker response protocol did not match the handshake");
    }
    if (response.header.abi != impl_->expected.abi) {
        impl_->fail("worker response ABI did not match the handshake");
    }
    if (response.header.request_id != frame.header.request_id) {
        impl_->fail(
            "worker response request ID mismatch: expected " +
            std::to_string(frame.header.request_id) + ", received " +
            std::to_string(response.header.request_id));
    }
    if (response.header.kind != "result" && response.header.kind != "error") {
        impl_->fail(
            "worker response kind must be 'result' or 'error', received '" +
            response.header.kind + "'");
    }
    return response;
}

ipc::Frame WorkerProcess::request_stream(
    const ipc::Frame& frame,
    const StreamEventCallback& callback,
    std::chrono::milliseconds timeout) {
    std::lock_guard<std::mutex> operation_lock(impl_->operation_mutex);
    if (timeout.count() <= 0) {
        impl_->fail("worker stream timeout must be positive");
    }
    if (!callback) {
        impl_->fail("worker stream callback must be present");
    }
    if (!impl_->is_active()) {
        impl_->fail("cannot stream because the worker is not healthy");
    }

    const Clock::time_point deadline = Clock::now() + timeout;
    impl_->write_request(frame, deadline, "worker stream request");

    struct ResponseState {
        std::mutex mutex;
        std::condition_variable condition;
        bool done = false;
        ipc::Frame response;
        std::exception_ptr failure;
    };
    const auto response_state = std::make_shared<ResponseState>();
    std::thread response_thread([implementation = impl_.get(), response_state, deadline] {
        ipc::Frame response;
        std::exception_ptr failure;
        try {
            response = implementation->read_response(
                deadline,
                "worker stream request",
                false);
        } catch (...) {
            failure = std::current_exception();
        }
        {
            std::lock_guard<std::mutex> lock(response_state->mutex);
            response_state->response = std::move(response);
            response_state->failure = std::move(failure);
            response_state->done = true;
        }
        response_state->condition.notify_all();
    });
    struct ResponseThreadGuard {
        std::thread& thread;
        Impl* implementation;

        ~ResponseThreadGuard() noexcept {
            if (thread.joinable()) {
                implementation->terminate_active_process(ERROR_OPERATION_ABORTED);
                thread.join();
            }
        }

        void join() {
            if (thread.joinable()) {
                thread.join();
            }
        }
    } response_thread_guard{response_thread, impl_.get()};

    bool saw_end = false;
    bool sent_cancel = false;
    uint64_t observed_event_count = 0;
    std::optional<uint64_t> expected_event_count;
    std::optional<uint64_t> terminal_event_count;
    std::exception_ptr callback_failure;
    for (;;) {
        std::vector<ipc::Frame> matching;
        {
            std::lock_guard<std::mutex> lock(impl_->event_mutex);
            auto iterator = impl_->events.begin();
            while (iterator != impl_->events.end()) {
                const bool valid_unrelated_log =
                    iterator->header.kind == "event" &&
                    iterator->header.method == "log" &&
                    iterator->header.request_id == 0 &&
                    iterator->header.protocol == impl_->expected.protocol &&
                    iterator->header.abi == impl_->expected.abi;
                if (iterator->header.request_id == frame.header.request_id ||
                    !valid_unrelated_log) {
                    matching.push_back(std::move(*iterator));
                    iterator = impl_->events.erase(iterator);
                } else {
                    ++iterator;
                }
            }
        }
        impl_->event_condition.notify_all();
        for (const ipc::Frame& event : matching) {
            ++observed_event_count;
            try {
                const StreamEventAction action = callback(event);
                if (action == StreamEventAction::End) {
                    simdjson::dom::parser parser;
                    simdjson::dom::element payload;
                    uint64_t count = 0;
                    if (parser.parse(event.header.payload_json).get(payload) !=
                            simdjson::SUCCESS ||
                        payload["eventCount"].get_uint64().get(count) != simdjson::SUCCESS ||
                        count != observed_event_count || count > ipc::kMaxStreamEventCount) {
                        throw std::runtime_error(
                            "worker stream terminal eventCount did not match observed events");
                    }
                    terminal_event_count = count;
                    saw_end = true;
                } else if (action == StreamEventAction::Cancel && !sent_cancel) {
                    if (Clock::now() >= deadline) {
                        throw std::runtime_error(
                            "worker stream timed out before cancellation could be sent");
                    }
                    ipc::Frame cancel_frame;
                    cancel_frame.header.protocol = impl_->expected.protocol;
                    cancel_frame.header.abi = impl_->expected.abi;
                    cancel_frame.header.request_id = frame.header.request_id;
                    cancel_frame.header.kind = "request";
                    cancel_frame.header.method = "cancel";
                    cancel_frame.header.payload_json =
                        std::string("{\"requestId\":") +
                        std::to_string(frame.header.request_id) + "}";
                    impl_->write_request(cancel_frame, deadline, "worker stream cancellation");
                    sent_cancel = true;
                }
            } catch (...) {
                callback_failure = std::current_exception();
                impl_->terminate_active_process(ERROR_BAD_FORMAT);
                break;
            }
        }

        bool response_done = false;
        bool response_failed = false;
        {
            std::lock_guard<std::mutex> lock(response_state->mutex);
            response_done = response_state->done;
            response_failed = response_state->failure != nullptr;
            if (response_done && !response_failed && !expected_event_count &&
                (response_state->response.header.kind == "result" ||
                 response_state->response.header.kind == "error")) {
                try {
                    simdjson::dom::parser parser;
                    simdjson::dom::element payload;
                    uint64_t count = 0;
                    if (parser.parse(response_state->response.header.payload_json).get(payload) !=
                            simdjson::SUCCESS ||
                        payload["eventCount"].get_uint64().get(count) != simdjson::SUCCESS ||
                        count == 0 || count > ipc::kMaxStreamEventCount) {
                        throw std::runtime_error(
                            "worker stream response requires a bounded positive eventCount");
                    }
                    expected_event_count = count;
                } catch (...) {
                    callback_failure = std::current_exception();
                    impl_->terminate_active_process(ERROR_BAD_FORMAT);
                }
            } else if (response_done && !response_failed && !expected_event_count) {
                callback_failure = std::make_exception_ptr(std::runtime_error(
                    "worker stream response kind was neither result nor error"));
                impl_->terminate_active_process(ERROR_BAD_FORMAT);
            }
        }
        if (expected_event_count && observed_event_count > *expected_event_count) {
            callback_failure = std::make_exception_ptr(std::runtime_error(
                "worker stream emitted more events than its final eventCount"));
            impl_->terminate_active_process(ERROR_BAD_FORMAT);
        }
        if (expected_event_count && terminal_event_count &&
            *expected_event_count != *terminal_event_count) {
            callback_failure = std::make_exception_ptr(std::runtime_error(
                "worker stream response eventCount did not match terminal event"));
            impl_->terminate_active_process(ERROR_BAD_FORMAT);
        }
        const bool complete_response = response_done && expected_event_count && saw_end &&
            observed_event_count == *expected_event_count;
        if (callback_failure || response_failed || complete_response) {
            break;
        }
        if (Clock::now() >= deadline) {
            impl_->terminate_active_process(ERROR_TIMEOUT);
            break;
        }
        std::unique_lock<std::mutex> lock(impl_->event_mutex);
        impl_->event_condition.wait_for(lock, std::chrono::milliseconds(10));
    }

    response_thread_guard.join();
    if (callback_failure) {
        std::rethrow_exception(callback_failure);
    }
    {
        std::lock_guard<std::mutex> lock(response_state->mutex);
        if (response_state->failure) {
            std::rethrow_exception(response_state->failure);
        }
        if (!saw_end) {
            impl_->fail("worker stream ended without a terminal event");
        }
    }
    const ipc::Frame response = std::move(response_state->response);
    if (response.header.protocol != impl_->expected.protocol ||
        response.header.abi != impl_->expected.abi ||
        response.header.request_id != frame.header.request_id ||
        (response.header.kind != "result" && response.header.kind != "error")) {
        impl_->fail("worker stream response identity did not match the request");
    }
    return response;
}

bool WorkerProcess::cancel(uint64_t request_id, std::chrono::milliseconds timeout) {
    ipc::Frame frame;
    frame.header.protocol = impl_->expected.protocol;
    frame.header.abi = impl_->expected.abi;
    frame.header.request_id = request_id;
    frame.header.kind = "request";
    frame.header.method = "cancel";
    return request(frame, timeout).header.kind == "result";
}

std::vector<ipc::Frame> WorkerProcess::take_events() {
    std::lock_guard<std::mutex> lock(impl_->event_mutex);
    std::vector<ipc::Frame> result;
    result.swap(impl_->events);
    impl_->event_condition.notify_all();
    return result;
}

void WorkerProcess::shutdown(std::chrono::milliseconds timeout) {
    std::lock_guard<std::mutex> operation_lock(impl_->operation_mutex);
    HANDLE process_handle = nullptr;
    {
        std::lock_guard<std::mutex> state_lock(impl_->state_mutex);
        process_handle = impl_->process.get();
    }
    if (process_handle == nullptr) {
        return;
    }

    const auto bounded_timeout = (std::max)(timeout, std::chrono::milliseconds::zero());
    const Clock::time_point deadline = Clock::now() + bounded_timeout;
    DWORD code = 0;
    if (GetExitCodeProcess(process_handle, &code) != FALSE && code == STILL_ACTIVE) {
        ipc::Frame shutdown_frame;
        shutdown_frame.header.protocol = impl_->expected.protocol;
        shutdown_frame.header.abi = impl_->expected.abi;
        shutdown_frame.header.request_id = (std::numeric_limits<uint64_t>::max)();
        shutdown_frame.header.kind = "request";
        shutdown_frame.header.method = "shutdown";
        const HANDLE command_handle = impl_->command_write.get();
        Impl::IoOutcome outcome = impl_->run_cancellable_io(
            [command_handle, &shutdown_frame](ipc::Frame&, std::string& error) {
                return ipc::write_frame(command_handle, shutdown_frame, error);
            },
            deadline);
        if (outcome.timed_out) {
            impl_->set_error("worker shutdown timed out while writing; process was terminated");
        } else if (!outcome.succeeded && !outcome.process_exited) {
            impl_->set_error("worker shutdown write failed: " + outcome.error);
        }
    }
    impl_->command_write.reset();

    DWORD wait_result = WaitForSingleObject(
        process_handle,
        bounded_wait_milliseconds(deadline, INFINITE - 1));
    if (wait_result == WAIT_TIMEOUT) {
        impl_->set_error("worker shutdown timed out; process was terminated");
        TerminateProcess(process_handle, ERROR_TIMEOUT);
        wait_result = WaitForSingleObject(process_handle, 1000);
    }
    if (wait_result == WAIT_FAILED) {
        impl_->set_error(win32_error("WaitForSingleObject during worker shutdown"));
    }
    if (GetExitCodeProcess(process_handle, &code) != FALSE && code != STILL_ACTIVE) {
        std::lock_guard<std::mutex> state_lock(impl_->state_mutex);
        impl_->cached_exit_code = code;
    }

    if (impl_->event_thread.joinable()) {
        {
            std::lock_guard<std::mutex> event_lock(impl_->event_mutex);
            impl_->stop_event_thread = true;
        }
        impl_->event_condition.notify_all();
        (void)CancelSynchronousIo(impl_->event_thread.native_handle());
        impl_->event_thread.join();
    }
    impl_->response_read.reset();
    impl_->event_read.reset();
    {
        std::lock_guard<std::mutex> state_lock(impl_->state_mutex);
        impl_->process.reset();
    }
}

bool WorkerProcess::healthy() const {
    return impl_->is_active();
}

std::optional<DWORD> WorkerProcess::exit_code() const {
    return impl_->query_exit_code();
}

std::string WorkerProcess::last_error() const {
    std::lock_guard<std::mutex> lock(impl_->state_mutex);
    return impl_->error;
}

} // namespace aila::runtime
