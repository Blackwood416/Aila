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
    EventHandler event_handler;
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
        message = exit_diagnostic(std::move(message), 200);
        set_error(message);
        throw std::runtime_error(message);
    }

    void event_loop() noexcept {
        for (;;) {
            ipc::Frame frame;
            std::string read_error;
            if (!ipc::read_frame(event_read.get(), frame, read_error)) {
                return;
            }
            if (frame.header.protocol != expected.protocol || frame.header.abi != expected.abi) {
                set_error("event frame protocol or ABI did not match the worker handshake");
                continue;
            }
            if (!event_handler) {
                continue;
            }
            try {
                event_handler(frame);
            } catch (const std::exception& exception) {
                set_error(std::string("event handler threw: ") + exception.what());
            } catch (...) {
                set_error("event handler threw an unknown exception");
            }
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

    ipc::Frame read_response(Clock::time_point deadline, const char* context) {
        const HANDLE handle = response_read.get();
        IoOutcome outcome = run_cancellable_io(
            [handle](ipc::Frame& frame, std::string& error) {
                return ipc::read_frame(handle, frame, error);
            },
            deadline);
        if (outcome.timed_out) {
            timeout_failure(std::string(context) + " while reading");
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
        event_handler = {};
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
    EventHandler event_handler,
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
        const fs::path normalized_runtime = normalized_absolute(runtime_directory);
        const fs::path normalized_worker = normalized_absolute(worker_executable);
        std::error_code filesystem_error;
        if (!fs::is_directory(normalized_runtime, filesystem_error) || filesystem_error) {
            throw std::runtime_error(
                "worker runtime directory is not an existing directory: '" +
                utf8(normalized_runtime.wstring()) + "'");
        }
        const DWORD worker_attributes = GetFileAttributesW(normalized_worker.c_str());
        if ((worker_attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0 &&
            worker_attributes != INVALID_FILE_ATTRIBUTES) {
            throw std::runtime_error(
                "worker executable must not be a reparse point: '" +
                utf8(normalized_worker.wstring()) + "'");
        }
        if (!detail::worker_file_attributes_are_safe(worker_attributes)) {
            throw std::runtime_error(
                "worker executable is not a contained regular file: '" +
                utf8(normalized_worker.wstring()) + "'");
        }
        filesystem_error.clear();
        if (!fs::is_regular_file(normalized_worker, filesystem_error) || filesystem_error) {
            throw std::runtime_error(
                "worker executable is not a regular file: '" +
                utf8(normalized_worker.wstring()) + "'");
        }
        filesystem_error.clear();
        if (!fs::equivalent(normalized_worker.parent_path(), normalized_runtime, filesystem_error) ||
            filesystem_error) {
            throw std::runtime_error(
                "worker executable must reside directly in the isolated runtime directory");
        }

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
            EXTENDED_STARTUPINFO_PRESENT | CREATE_UNICODE_ENVIRONMENT | CREATE_NO_WINDOW;
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

        impl_->command_write = std::move(command_write);
        impl_->response_read = std::move(response_read);
        impl_->event_read = std::move(event_read);
        impl_->event_handler = std::move(event_handler);
        impl_->expected = expected;
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

bool WorkerProcess::cancel(uint64_t request_id, std::chrono::milliseconds timeout) {
    ipc::Frame frame;
    frame.header.protocol = impl_->expected.protocol;
    frame.header.abi = impl_->expected.abi;
    frame.header.request_id = request_id;
    frame.header.kind = "request";
    frame.header.method = "cancel";
    return request(frame, timeout).header.kind == "result";
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
        (void)CancelSynchronousIo(impl_->event_thread.native_handle());
        impl_->event_thread.join();
    }
    impl_->response_read.reset();
    impl_->event_read.reset();
    impl_->event_handler = {};
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
