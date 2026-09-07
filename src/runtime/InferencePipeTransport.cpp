#include "runtime/InferenceClient.hpp"

#ifdef _WIN32
#include "ipc/Win32Pipe.hpp"
#include "runtime/ChildEnvironment.hpp"
#include "runtime/RuntimeDirectory.hpp"

#include <Windows.h>
#include <algorithm>
#include <array>
#include <climits>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <string_view>
#include <utility>

namespace aila::runtime {
namespace {
namespace fs = std::filesystem;
using namespace std::chrono_literals;

class Handle {
public:
    Handle() = default;
    explicit Handle(HANDLE value) : value_(value) {}
    ~Handle() { Reset(); }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    Handle(Handle&& other) noexcept : value_(other.Release()) {}
    Handle& operator=(Handle&& other) noexcept {
        if (this != &other) Reset(other.Release());
        return *this;
    }
    HANDLE Get() const noexcept { return value_; }
    explicit operator bool() const noexcept { return value_ && value_ != INVALID_HANDLE_VALUE; }
    HANDLE Release() noexcept { auto result = value_; value_ = nullptr; return result; }
    void Reset(HANDLE value = nullptr) noexcept { if (*this) CloseHandle(value_); value_ = value; }
private:
    HANDLE value_ = nullptr;
};

std::string win_error(const char* operation) {
    return std::string(operation) + " failed with Win32 error " + std::to_string(GetLastError());
}

std::string utf8(const std::wstring& value) {
    if (value.size() > static_cast<size_t>(std::numeric_limits<int>::max())) throw std::runtime_error("Runtime path is too long");
    const int length = static_cast<int>(value.size());
    const int size = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.data(), length, nullptr, 0, nullptr, nullptr);
    if (size <= 0) throw std::runtime_error("Invalid runtime path encoding");
    std::string result(static_cast<size_t>(size), '\0');
    if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.data(), length, result.data(), size, nullptr, nullptr) != size) {
        throw std::runtime_error("Invalid runtime path encoding");
    }
    return result;
}

bool same_path(const fs::path& a, const fs::path& b) {
    const auto left = a.lexically_normal().wstring();
    const auto right = b.lexically_normal().wstring();
    if (left.size() > INT_MAX || right.size() > INT_MAX) return false;
    return CompareStringOrdinal(left.data(), static_cast<int>(left.size()), right.data(), static_cast<int>(right.size()), TRUE) == CSTR_EQUAL;
}

fs::path strip_extended(std::wstring value) {
    if (value.rfind(L"\\\\?\\UNC\\", 0) == 0) value = L"\\\\" + value.substr(8);
    else if (value.rfind(L"\\\\?\\", 0) == 0) value.erase(0, 4);
    return fs::path(value).lexically_normal();
}

fs::path handle_path(HANDLE handle) {
    std::vector<wchar_t> buffer(MAX_PATH);
    for (;;) {
        const DWORD size = GetFinalPathNameByHandleW(handle, buffer.data(), static_cast<DWORD>(buffer.size()), FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
        if (!size) throw std::runtime_error(win_error("GetFinalPathNameByHandleW"));
        if (size < buffer.size()) return strip_extended(std::wstring(buffer.data(), size));
        buffer.resize(static_cast<size_t>(size) + 1);
    }
}

bool ordinary_directory(DWORD attributes) {
    return attributes != INVALID_FILE_ATTRIBUTES && (attributes & FILE_ATTRIBUTE_DIRECTORY) &&
        !(attributes & FILE_ATTRIBUTE_REPARSE_POINT);
}

bool ordinary_file(DWORD attributes) {
    return attributes != INVALID_FILE_ATTRIBUTES && !(attributes & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT));
}

DWORD handle_attributes(HANDLE handle) {
    BY_HANDLE_FILE_INFORMATION info{};
    if (!GetFileInformationByHandle(handle, &info)) throw std::runtime_error(win_error("GetFileInformationByHandle"));
    return info.dwFileAttributes;
}

void verify_components(const fs::path& path) {
    fs::path current = path.root_path();
    if (current.empty() || !ordinary_directory(GetFileAttributesW(current.c_str()))) {
        throw std::runtime_error("Runtime root must be an ordinary absolute directory");
    }
    for (const auto& component : path.relative_path()) {
        current /= component;
        if (!ordinary_directory(GetFileAttributesW(current.c_str()))) {
            throw std::runtime_error("Runtime path contains a missing directory or reparse point");
        }
    }
}

Handle validation_handle(const fs::path& path, DWORD flags) {
    Handle result(CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES, FILE_SHARE_READ, nullptr,
                              OPEN_EXISTING, flags | FILE_FLAG_OPEN_REPARSE_POINT, nullptr));
    if (!result) throw std::runtime_error(win_error("CreateFileW(runtime validation)"));
    return result;
}

struct LaunchPaths {
    fs::path runtime;
    fs::path worker;
    Handle runtimeLock;
    Handle workerLock;
};

// Keep the v1 launch checks intact without changing WorkerProcess's anonymous
// implementation or its ABI: reject reparse components, hold validation
// handles through suspended image verification, and restrict inheritance.
LaunchPaths validate_paths(const fs::path& runtime, const fs::path& executable) {
    LaunchPaths result;
    result.runtime = fs::absolute(runtime).lexically_normal();
    result.worker = fs::absolute(executable).lexically_normal();
    verify_components(result.runtime);
    if (!same_path(result.worker.parent_path(), result.runtime)) {
        throw std::runtime_error("Worker must reside directly in the isolated runtime directory");
    }
    result.runtimeLock = validation_handle(result.runtime, FILE_FLAG_BACKUP_SEMANTICS);
    if (!ordinary_directory(handle_attributes(result.runtimeLock.Get())) ||
        !ordinary_file(GetFileAttributesW(result.worker.c_str()))) {
        throw std::runtime_error("Runtime/Worker must not be a reparse point");
    }
    result.workerLock = validation_handle(result.worker, 0);
    if (!ordinary_file(handle_attributes(result.workerLock.Get()))) throw std::runtime_error("Worker is not an ordinary file");
    result.runtime = handle_path(result.runtimeLock.Get());
    result.worker = handle_path(result.workerLock.Get());
    verify_components(result.runtime);
    if (!same_path(result.worker.parent_path(), result.runtime)) throw std::runtime_error("Resolved Worker escaped its runtime directory");
    return result;
}

std::wstring quote(std::wstring_view text) {
    std::wstring result = L"\"";
    size_t slashes = 0;
    for (wchar_t c : text) {
        if (c == L'\\') { ++slashes; continue; }
        result.append(c == L'"' ? slashes * 2 + 1 : slashes, L'\\');
        result.push_back(c);
        slashes = 0;
    }
    result.append(slashes * 2, L'\\');
    result.push_back(L'"');
    return result;
}

void append(std::wstring& command, std::wstring_view argument) {
    if (!command.empty()) command.push_back(L' ');
    command += quote(argument);
}

void pipe(Handle& read, Handle& write) {
    SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    HANDLE r = nullptr, w = nullptr;
    if (!CreatePipe(&r, &w, &security, 0)) throw std::runtime_error(win_error("CreatePipe"));
    read.Reset(r);
    write.Reset(w);
}

void private_end(HANDLE handle) {
    if (!SetHandleInformation(handle, HANDLE_FLAG_INHERIT, 0)) throw std::runtime_error(win_error("SetHandleInformation"));
}

class Attributes {
public:
    Attributes() {
        SIZE_T size = 0;
        InitializeProcThreadAttributeList(nullptr, 1, 0, &size);
        if (!size) throw std::runtime_error(win_error("InitializeProcThreadAttributeList"));
        storage.resize(size);
        value = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(storage.data());
        if (!InitializeProcThreadAttributeList(value, 1, 0, &size)) {
            value = nullptr;
            throw std::runtime_error(win_error("InitializeProcThreadAttributeList"));
        }
    }
    ~Attributes() { if (value) DeleteProcThreadAttributeList(value); }
    std::vector<std::byte> storage;
    LPPROC_THREAD_ATTRIBUTE_LIST value = nullptr;
};

fs::path image_path(HANDLE process) {
    std::vector<wchar_t> buffer(MAX_PATH);
    for (;;) {
        DWORD size = static_cast<DWORD>(buffer.size());
        if (QueryFullProcessImageNameW(process, 0, buffer.data(), &size)) return strip_extended(std::wstring(buffer.data(), size));
        if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) throw std::runtime_error(win_error("QueryFullProcessImageNameW"));
        buffer.resize(buffer.size() * 2);
    }
}

class PipeTransport final : public InferenceTransport {
public:
    Handle commandWrite;
    Handle responseRead;
    Handle eventRead;
    Handle process;

    ~PipeTransport() override {
        Stop(true);
        if (process) (void)WaitForSingleObject(process.Get(), 2000);
    }

    bool ReadResponse(ipc::Frame& frame, std::string& error) override {
        const bool first = firstResponse;
        const bool result = Read(responseRead.Get(), wire::Channel::Response, first, frame, error);
        firstResponse = false;
        return result;
    }

    bool ReadEvent(ipc::Frame& frame, std::string& error) override {
        return Read(eventRead.Get(), wire::Channel::Event, false, frame, error);
    }

    bool WriteCommand(const ipc::Frame& frame, std::string& error) override {
        wire::Limits limits;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (stopped) { error = "Inference pipe is closed"; return false; }
            limits = limits_;
            HANDLE thread = nullptr;
            if (!DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &thread, 0, FALSE, DUPLICATE_SAME_ACCESS)) {
                error = win_error("DuplicateHandle(command writer)");
                return false;
            }
            writerThread.Reset(thread);
        }
        std::vector<std::byte> bytes;
        wire::ValidationError validation;
        bool result = wire::encode_frame(frame, bytes, validation, wire::frame_bounds(wire::Channel::Command, limits));
        if (!result) error = validation.message;
        else result = ipc::write_all(commandWrite.Get(), bytes.data(), bytes.size(), error);
        {
            std::lock_guard<std::mutex> lock(mutex);
            writerThread.Reset();
        }
        return result;
    }

    void SetLimits(const wire::Limits& limits) override {
        std::lock_guard<std::mutex> lock(mutex);
        limits_ = limits;
    }

    void Stop(bool force) noexcept override {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopped = true;
            if (writerThread) (void)CancelSynchronousIo(writerThread.Get());
        }
        changed.notify_all();
        if (force && process) {
            DWORD code = 0;
            if (GetExitCodeProcess(process.Get(), &code) && code == STILL_ACTIVE) {
                (void)TerminateProcess(process.Get(), ERROR_OPERATION_ABORTED);
            }
        }
    }

    bool WaitForExit(std::chrono::milliseconds timeout, std::string& error) override {
        if (!process) return true;
        const auto bounded = std::max<int64_t>(0, std::min<int64_t>(timeout.count(), INFINITE - 1));
        const DWORD wait = WaitForSingleObject(process.Get(), static_cast<DWORD>(bounded));
        if (wait != WAIT_OBJECT_0) {
            error = wait == WAIT_TIMEOUT ? "Worker process exit timed out" : win_error("WaitForSingleObject");
            return false;
        }
        DWORD code = 0;
        if (!GetExitCodeProcess(process.Get(), &code) || code != 0) {
            error = "Worker exit code " + std::to_string(code);
            return false;
        }
        return true;
    }

private:
    bool ReadAvailable(HANDLE handle, std::byte* destination, size_t count, std::string& error) {
        size_t offset = 0;
        while (offset != count) {
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (stopped) { error = "Inference pipe is closed"; return false; }
            }
            DWORD available = 0;
            if (!PeekNamedPipe(handle, nullptr, 0, nullptr, &available, nullptr)) {
                error = win_error("PeekNamedPipe");
                return false;
            }
            if (!available) {
                std::unique_lock<std::mutex> lock(mutex);
                changed.wait_for(lock, 1ms, [&] { return stopped; });
                continue;
            }
            const size_t amount = std::min(count - offset, static_cast<size_t>(available));
            // There is exactly one reader of this pipe, so this bounded read
            // never waits for bytes that PeekNamedPipe has not observed.
            if (!ipc::read_exact(handle, destination + offset, amount, error)) return false;
            offset += amount;
        }
        return true;
    }

    bool Read(HANDLE handle, wire::Channel channel, bool bootstrap, ipc::Frame& frame, std::string& error) {
        std::array<std::byte, 8> prefix{};
        if (!ReadAvailable(handle, prefix.data(), prefix.size(), error)) return false;
        wire::FrameBounds bounds;
        {
            std::lock_guard<std::mutex> lock(mutex);
            bounds = bootstrap ? wire::bootstrap_bounds() : wire::frame_bounds(channel, limits_);
        }
        wire::FrameSizes sizes;
        wire::ValidationError validation;
        if (!wire::inspect_prefix(prefix.data(), prefix.size(), sizes, validation, bounds)) {
            throw InferenceClientError(ClientErrorCode::ProtocolError, validation.message, validation.code);
        }
        std::vector<std::byte> bytes(sizes.totalBytes);
        std::copy(prefix.begin(), prefix.end(), bytes.begin());
        if (!ReadAvailable(handle, bytes.data() + prefix.size(), bytes.size() - prefix.size(), error)) return false;
        if (!wire::decode_frame(bytes, frame, validation, bounds)) {
            throw InferenceClientError(validation.code == wire::ErrorCode::UnsupportedProtocol
                    ? ClientErrorCode::RemoteError : ClientErrorCode::ProtocolError,
                validation.message, validation.code);
        }
        return true;
    }

    std::mutex mutex;
    std::condition_variable changed;
    wire::Limits limits_;
    Handle writerThread;
    bool stopped = false;
    bool firstResponse = true;
};
} // namespace

std::shared_ptr<InferenceTransport> StartInferencePipeTransport(
    const fs::path& runtimeDirectory, const fs::path& workerExecutable,
    const std::string& buildId, ExpectedInferenceIdentity& expected) {
    // RuntimeDirectory supplies the conventional Worker path when requested;
    // validate_paths additionally enforces the v1 handle/path isolation rules.
    const auto selectedWorker = workerExecutable.empty() ? require_worker_executable(runtimeDirectory) : workerExecutable;
    auto validated = validate_paths(runtimeDirectory, selectedWorker);
    const auto systemRoot = system_root_directory();
    auto environment = build_isolated_environment(current_environment(), validated.runtime, systemRoot);
    auto transport = std::make_shared<PipeTransport>();
    Handle commandRead, responseWrite, eventWrite;
    pipe(commandRead, transport->commandWrite);
    pipe(transport->responseRead, responseWrite);
    pipe(transport->eventRead, eventWrite);
    private_end(transport->commandWrite.Get());
    private_end(transport->responseRead.Get());
    private_end(transport->eventRead.Get());
    std::array<HANDLE, 3> inherited{commandRead.Get(), responseWrite.Get(), eventWrite.Get()};
    Attributes attributes;
    if (!UpdateProcThreadAttribute(attributes.value, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                                   inherited.data(), sizeof(inherited), nullptr, nullptr)) {
        throw std::runtime_error(win_error("UpdateProcThreadAttribute"));
    }
    std::wstring command;
    append(command, validated.worker.wstring());
    append(command, L"--ffi-worker-v2");
    append(command, L"--command-read-handle");
    append(command, std::to_wstring(reinterpret_cast<uintptr_t>(commandRead.Get())));
    append(command, L"--response-write-handle");
    append(command, std::to_wstring(reinterpret_cast<uintptr_t>(responseWrite.Get())));
    append(command, L"--event-write-handle");
    append(command, std::to_wstring(reinterpret_cast<uintptr_t>(eventWrite.Get())));
    std::vector<wchar_t> mutableCommand(command.begin(), command.end());
    mutableCommand.push_back(L'\0');
    STARTUPINFOEXW startup{};
    startup.StartupInfo.cb = sizeof(startup);
    startup.lpAttributeList = attributes.value;
    PROCESS_INFORMATION info{};
    const DWORD flags = EXTENDED_STARTUPINFO_PRESENT | CREATE_UNICODE_ENVIRONMENT | CREATE_NO_WINDOW | CREATE_SUSPENDED;
    if (!CreateProcessW(validated.worker.c_str(), mutableCommand.data(), nullptr, nullptr, TRUE,
                        flags, environment.data(), validated.runtime.c_str(), &startup.StartupInfo, &info)) {
        throw std::runtime_error(win_error("CreateProcessW"));
    }
    transport->process.Reset(info.hProcess);
    Handle thread(info.hThread);
    commandRead.Reset();
    responseWrite.Reset();
    eventWrite.Reset();
    if (!same_path(image_path(transport->process.Get()), validated.worker)) {
        throw std::runtime_error("Suspended v2 Worker image does not match the validated executable");
    }
    expected.buildId = buildId;
    expected.executable = utf8(validated.worker.wstring());
    expected.runtimeDirectory = expected.currentDirectory = utf8(validated.runtime.wstring());
    expected.path = utf8(validated.runtime.wstring() + L";" + (systemRoot / L"System32").lexically_normal().wstring() + L";" + systemRoot.wstring());
    if (ResumeThread(thread.Get()) == std::numeric_limits<DWORD>::max()) throw std::runtime_error(win_error("ResumeThread"));
    return transport;
}
} // namespace aila::runtime

#else
namespace aila::runtime {
std::shared_ptr<InferenceTransport> StartInferencePipeTransport(
    const std::filesystem::path&, const std::filesystem::path&,
    const std::string&, ExpectedInferenceIdentity&) {
    throw std::runtime_error("Isolated inference Worker launch requires Windows");
}
}
#endif
