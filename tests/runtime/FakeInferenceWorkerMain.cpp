#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>

#include "worker/InferenceWorkerMain.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cwchar>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#ifndef AILA_BUILD_ID
#error "The fake inference Worker must use the same AILA_BUILD_ID as its host test"
#endif

namespace {

namespace fs = std::filesystem;
namespace core = aila::inference;
namespace worker = aila::worker;
using namespace std::chrono_literals;

// Kept outside the normal output range so only an explicit test prompt can
// terminate the child. The real protocol, bridge and Scheduler remain intact.
constexpr int CrashPromptToken = 1023;
constexpr int VocabularySize = 2048;

std::string Utf8(const std::wstring& value) {
    if (value.empty()) return {};
    if (value.size() > static_cast<size_t>((std::numeric_limits<int>::max)()))
        throw std::runtime_error("Identity text exceeds Win32 limits");
    const int length = static_cast<int>(value.size());
    const int size = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.data(), length,
                                        nullptr, 0, nullptr, nullptr);
    if (size <= 0) throw std::runtime_error("Could not encode identity text as UTF-8");
    std::string result(static_cast<size_t>(size), '\0');
    if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.data(), length,
                            result.data(), size, nullptr, nullptr) != size)
        throw std::runtime_error("Could not encode identity text as UTF-8");
    return result;
}

fs::path ModulePath() {
    std::vector<wchar_t> buffer(MAX_PATH);
    for (;;) {
        const DWORD copied = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (copied == 0) throw std::runtime_error("GetModuleFileNameW failed");
        if (copied < buffer.size()) return fs::absolute(std::wstring(buffer.data(), copied)).lexically_normal();
        if (buffer.size() >= 32768) throw std::runtime_error("Fake Worker executable path is too long");
        buffer.resize(buffer.size() * 2);
    }
}

std::wstring EnvironmentValue(const wchar_t* name) {
    SetLastError(ERROR_SUCCESS);
    const DWORD size = GetEnvironmentVariableW(name, nullptr, 0);
    if (size == 0) {
        if (GetLastError() == ERROR_SUCCESS || GetLastError() == ERROR_ENVVAR_NOT_FOUND) return {};
        throw std::runtime_error("GetEnvironmentVariableW size query failed");
    }
    std::vector<wchar_t> buffer(size);
    const DWORD copied = GetEnvironmentVariableW(name, buffer.data(), size);
    if (copied >= size) throw std::runtime_error("GetEnvironmentVariableW failed");
    return std::wstring(buffer.data(), copied);
}

struct Handles {
    HANDLE command = nullptr;
    HANDLE response = nullptr;
    HANDLE event = nullptr;
    Handles() = default;
    Handles(const Handles&) = delete;
    Handles& operator=(const Handles&) = delete;
    ~Handles() {
        if (command) CloseHandle(command);
        if (response) CloseHandle(response);
        if (event) CloseHandle(event);
    }
};

void ParseArguments(int argc, wchar_t** argv, Handles& handles) {
    if (argc != 8 || !argv || !argv[1] || std::wstring_view(argv[1]) != L"--ffi-worker-v2")
        throw std::runtime_error("Expected --ffi-worker-v2 and three inherited pipe handles");
    for (int index = 2; index < argc; index += 2) {
        if (!argv[index] || !argv[index + 1]) throw std::runtime_error("Missing pipe argument");
        const std::wstring_view option(argv[index]);
        HANDLE* destination = nullptr;
        if (option == L"--command-read-handle") destination = &handles.command;
        else if (option == L"--response-write-handle") destination = &handles.response;
        else if (option == L"--event-write-handle") destination = &handles.event;
        else throw std::runtime_error("Unknown fake inference Worker argument");
        if (*destination) throw std::runtime_error("Repeated pipe argument");
        const wchar_t* text = argv[index + 1];
        if (*text < L'0' || *text > L'9') throw std::runtime_error("Invalid pipe handle");
        wchar_t* end = nullptr;
        errno = 0;
        const auto value = std::wcstoull(text, &end, 10);
        if (errno == ERANGE || end == text || *end != L'\0' || value == 0 ||
            value > (std::numeric_limits<uintptr_t>::max)())
            throw std::runtime_error("Invalid pipe handle integer");
        const HANDLE handle = reinterpret_cast<HANDLE>(static_cast<uintptr_t>(value));
        DWORD flags = 0;
        if (handle == INVALID_HANDLE_VALUE || !GetHandleInformation(handle, &flags) ||
            GetFileType(handle) != FILE_TYPE_PIPE)
            throw std::runtime_error("Inherited handle is not an open pipe");
        if (handle == handles.command || handle == handles.response || handle == handles.event)
            throw std::runtime_error("Protocol pipe handles must be distinct");
        *destination = handle;
    }
    if (!handles.command || !handles.response || !handles.event)
        throw std::runtime_error("Missing protocol pipe");
}

class FakeBackend final : public core::SessionBackend {
public:
    FakeBackend(int capacity, std::shared_ptr<void> owner) : capacity(capacity), owner(std::move(owner)) {}
    int ContextLength() const override { return length; }
    int Capacity() const override { return capacity; }
    int VocabSize() const override { return VocabularySize; }
    size_t CacheBytes() const override { return static_cast<size_t>(capacity) * 16; }
    void Reset() override { length = 0; }
    int Truncate(int value) override {
        if (value < 0 || value > length) throw std::invalid_argument("Invalid fake cache truncation");
        length = value;
        return value;
    }
    core::ExecutionResult Forward(const std::vector<int>& ids, const core::SamplingInput* sampling,
                                  bool captureLogits) override {
        if (ids.empty() || ids.size() > static_cast<size_t>(capacity - length))
            throw std::invalid_argument("Invalid fake inference step");
        // A CPU executor with no pacing can fill the core ring before Windows
        // schedules its pipe threads; retain a realistic scheduling boundary.
        std::this_thread::sleep_for(1ms);
        if (std::find(ids.begin(), ids.end(), CrashPromptToken) != ids.end()) {
            // Give the accepted response a chance to cross its independent
            // pipe. Tests also keep a previously accepted peer request alive.
            std::this_thread::sleep_for(100ms);
            (void)TerminateProcess(GetCurrentProcess(), 79);
            std::abort();
        }
        length += static_cast<int>(ids.size());
        core::ExecutionResult result;
        if (sampling) result.tokenId = (ids.back() + length + 1) % 1000;
        if (captureLogits) result.logits.assign(VocabularySize, 0.0f);
        return result;
    }
private:
    const int capacity;
    int length = 0;
    std::shared_ptr<void> owner;
};

class FakeModel final : public worker::InferenceModel, public std::enable_shared_from_this<FakeModel> {
public:
    explicit FakeModel(int capacity) : capacity(capacity) {}
    std::shared_ptr<core::SessionState> CreateSession() override {
        return std::make_shared<core::SessionState>(std::make_unique<FakeBackend>(capacity, shared_from_this()));
    }
    int VocabSize() const noexcept override { return VocabularySize; }
    int MaxSequenceLength() const noexcept override { return capacity; }
    size_t SessionReservationBytes() const noexcept override { return static_cast<size_t>(capacity) * 16; }
private:
    const int capacity;
};

class FakeFactory final : public worker::InferenceModelFactory {
public:
    std::shared_ptr<worker::InferenceModel> LoadQwen3(const std::string& directory,
                                                   worker::InferenceModelOptions options) override {
        if (directory != "fake-pipe-model" || options.maxSequenceLength <= 0)
            throw std::invalid_argument("Unknown CPU fixture model");
        return std::make_shared<FakeModel>(options.maxSequenceLength);
    }
};

} // namespace

int wmain(int argc, wchar_t** argv) {
    try {
        Handles handles;
        ParseArguments(argc, argv, handles);
        const auto executable = ModulePath();
        aila::ipc::v2::Hello identity;
        identity.buildId = AILA_BUILD_ID;
        identity.executable = Utf8(executable.wstring());
        identity.runtimeDirectory = Utf8(executable.parent_path().wstring());
        identity.currentDirectory = Utf8(fs::current_path().lexically_normal().wstring());
        identity.path = Utf8(EnvironmentValue(L"PATH"));
        return worker::RunInferenceWorker(handles.command, handles.response, handles.event,
                                           std::move(identity), std::make_unique<FakeFactory>());
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[Fake inference Worker] %s\n", error.what());
        return 3;
    }
}
