#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>

#include "worker/InferenceWorkerMain.hpp"
#include "core/ExecutionTrace.hpp"
#include "profile/Profiling.hpp"

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

// The fake worker links Request.cpp (which includes EnvUtils.hpp) but not the
// main library that defines the CLI-driven KV-quant override. Follow the other
// test binaries and provide the default locally.
namespace aila::env {
bool g_kv_quant_override = false;
}

namespace {

namespace fs = std::filesystem;
namespace core = aila::inference;
namespace worker = aila::worker;
namespace diagnostics = aila::diagnostics;
using namespace std::chrono_literals;

// Kept outside the normal output range so only an explicit test prompt can
// terminate the child. The real protocol, bridge and Scheduler remain intact.
constexpr int CrashPromptToken = 1023;
constexpr int ReadWaitPromptToken = 1022;
constexpr int VocabularySize = 2048;

bool FixtureLogsEnabled() {
    static const bool enabled = [] {
        const auto value = std::getenv("AILA_FAKE_WORKER_LOGS");
        return value && std::string_view(value) == "1";
    }();
    return enabled;
}

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
    FakeBackend(int capacity, std::shared_ptr<void> owner, bool legacyText, bool httpTools)
        : capacity(capacity), owner(std::move(owner)), legacyText(legacyText), httpTools(httpTools) {}
    int ContextLength() const override { return length; }
    int Capacity() const override { return capacity; }
    int VocabSize() const override { return VocabularySize; }
    size_t CacheBytes() const override { return static_cast<size_t>(capacity) * 16; }
    void Reset() override { length = 0; toolOutput.clear(); toolCursor = 0; }
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
        if (ids.size() > 1 && FixtureLogsEnabled()) {
            AILA_LOG_DEBUG("[FakeLog] debug");
            AILA_LOG_INFO(u8"[FakeLog] 信息 α");
            AILA_LOG_WARN("[FakeLog] warning");
            AILA_LOG_ERROR("[FakeLog] error");
        }
        // Opt-in CPU seam for an overlapping blocking Read/release. Normal
        // pipe/socket fixtures never use this input token.
        if (std::find(ids.begin(), ids.end(), ReadWaitPromptToken) != ids.end())
            std::this_thread::sleep_for(500ms);
        if (std::find(ids.begin(), ids.end(), CrashPromptToken) != ids.end()) {
            // Give the accepted response a chance to cross its independent
            // pipe. Tests also keep a previously accepted peer request alive.
            std::this_thread::sleep_for(100ms);
            (void)TerminateProcess(GetCurrentProcess(), 79);
            std::abort();
        }
        length += static_cast<int>(ids.size());
        core::ExecutionResult result;
        if (httpTools) SelectToolOutput(ids);
        if (sampling) {
            if (httpTools) result.tokenId = toolCursor < toolOutput.size() ? toolOutput[toolCursor++] : 2001;
            else result.tokenId = legacyText ? 'A' + (ids.back() + length + 1) % 26 : (ids.back() + length + 1) % 1000;
        }
        if (captureLogits) result.logits.assign(VocabularySize, 0.0f);
        return result;
    }
private:
    const int capacity;
    int length = 0;
    std::shared_ptr<void> owner;
    bool legacyText;
    bool httpTools;
    std::vector<int> toolOutput;
    size_t toolCursor = 0;

    void SelectToolOutput(const std::vector<int>& ids) {
        // This model name and these markers exist only in the HTTP CPU tool
        // fixture. Token generation still passes through the real Scheduler,
        // IPC, tokenizer and socket path. Ordinary fake models are unchanged.
        int selector = -1;
        for (const int token : ids) if (token >= 2010 && token <= 2017) selector = token;
        if (selector < 0) return;
        toolOutput.clear();
        toolCursor = 0;
        std::string text;
        switch (selector) {
        case 2010: text = u8R"(<tool_call>{"name":"weather","arguments":{"city":"北京"}}</tool_call>)"; break;
        case 2011: text = u8"<tool_call>\n<function=weather>\n<parameter=city>北京</parameter>\n</function>\n</tool_call>"; break;
        case 2012: text = "<tool_call>{\"name\":\"weather\",\"arguments\":{}}</tool_call>\n<tool_call>{\"name\":\"calc\",\"arguments\":{\"x\":3}}</tool_call>"; break;
        case 2013: text = "<tool_call>{\"name\":\"weather\",\"arguments\":[]}</tool_call>"; break;
        case 2014: text = "Tool result processed."; break;
        case 2015:
            toolOutput.push_back(1400);
            for (int index = 0; index < 24; ++index) toolOutput.push_back(1401);
            toolOutput.push_back(1402);
            return;
        case 2016: text = "<tool_call>{\"name\":\"unknown\",\"arguments\":{}}</tool_call>"; break;
        case 2017: text = "<tool_call>{\"name\":\"weather\",\"arguments\":{}"; break;
        }
        for (size_t offset = 0; offset < text.size();) {
            // Exercise whole added-token and mixed-BPE tool delimiters. The
            // legacy tokenizer hides added tokens; HTTP must restore them.
            if (selector == 2010 && text.compare(offset, 11, "<tool_call>") == 0) {
                toolOutput.push_back(2018); offset += 11;
            } else if ((selector == 2010 || selector == 2011) && text.compare(offset, 12, "</tool_call>") == 0) {
                toolOutput.push_back(2019); offset += 12;
            } else {
                const auto byte = static_cast<unsigned char>(text[offset++]);
                toolOutput.push_back(1000 + byte + (byte >= 23 ? 1 : 0));
            }
        }
    }
};

class FakeModel final : public worker::InferenceModel, public std::enable_shared_from_this<FakeModel> {
public:
    FakeModel(int capacity, std::optional<aila::ipc::v2::AllocationLimits> limits, bool legacyText = false, bool httpTools = false)
        : capacity(capacity), legacyText(legacyText), httpTools(httpTools) {
        // This CPU fixture has no USM or Hybrid host mirrors. It acknowledges
        // limits on the real protocol while keeping its synthetic cache size.
        if (limits) memory = aila::ipc::v2::ModelMemoryInfo{*limits, 0, 0};
    }
    std::shared_ptr<core::SessionState> CreateSession() override {
        return std::make_shared<core::SessionState>(std::make_unique<FakeBackend>(capacity, shared_from_this(), legacyText, httpTools));
    }
    int VocabSize() const noexcept override { return VocabularySize; }
    int MaxSequenceLength() const noexcept override { return capacity; }
    size_t SessionReservationBytes() const noexcept override { return static_cast<size_t>(capacity) * 16; }
    std::optional<aila::ipc::v2::ModelMemoryInfo> MemoryInfo() const override { return memory; }
private:
    const int capacity;
    std::optional<aila::ipc::v2::ModelMemoryInfo> memory;
    bool legacyText;
    bool httpTools;
};

class FinalCloseFlushFailureSink final : public diagnostics::ExecutionTraceSink {
public:
    void WriteAndFlush(std::string_view record) override {
        // Keep a complete record before reporting a failed flush. The JSON
        // bytes alone cannot distinguish this from a successful final close.
        written.append(record.data(), record.size());
        if (record.find("\"event\":\"trace_close\"") != std::string_view::npos)
            throw std::runtime_error("injected final trace_close flush failure");
    }
private:
    std::string written;
};

class FakeFactory final : public worker::InferenceModelFactory {
public:
    aila::ipc::v2::RuntimeMemoryInfo QueryRuntimeMemory() const override {
        using namespace aila::ipc::v2;
        if (!memoryMode) return {};
        const auto query = ++memoryQueries;
        if (memoryMode == 3) std::this_thread::sleep_for(std::chrono::milliseconds(1500));
        if (memoryMode == 2 && query >= 2) throw std::runtime_error("injected memory observation failure");
        if (memoryMode == 1 && query == 2) return {RuntimeMemoryState::Busy, std::nullopt};
        const bool loaded = !memoryModel.expired();
        // Explicit sentinels for protocol/C ABI tests; no device is allocated.
        return {RuntimeMemoryState::Ready, RuntimeMemoryUsage{
            loaded ? 1024u : 0u, 4096, loaded ? 16u : 0u, 128,
            loaded ? 1u : 0u, loaded ? 1u : 0u, memoryLimits.maxDeviceBytes, memoryLimits.maxPinnedHostBytes}};
    }
    std::shared_ptr<worker::InferenceModel> LoadQwen3(const std::string& directory,
                                                   worker::InferenceModelOptions options) override {
        return Load(directory, options, false);
    }
    std::shared_ptr<worker::InferenceModel> LoadQwen35(const std::string& directory,
                                                    worker::InferenceModelOptions options) override {
        return Load(directory, options, true);
    }
    std::shared_ptr<worker::InferenceModel> LoadQwen35Bf16(const std::string& directory,
                                                        worker::InferenceModelOptions options) override {
        return Load(directory, options, true);
    }
    std::shared_ptr<worker::InferenceModel> LoadQwen4Exp(const std::string& directory,
                                                      worker::InferenceModelOptions options) override {
        // Qwen4Exp shares the Hybrid-style CPU fixture: the pipe harness names
        // its non-Qwen3 directory "fake-pipe-hybrid-model", which this factory
        // accepts only on the hybrid branch. "fake-pipe-model" is accepted on
        // either branch, so this also covers the socket fixture.
        return Load(directory, options, true);
    }
private:
    std::shared_ptr<worker::InferenceModel> Load(const std::string& directory,
                                               worker::InferenceModelOptions options, bool hybrid) {
        // Socket tests place their CPU tokenizer in an isolated absolute
        // directory; the real GPU factory is never involved in this fixture.
        const auto basename = fs::u8path(directory).filename();
        const bool closeFailure = !hybrid && basename == "fake-pipe-trace-close-failure-model";
        const bool explicitLegacyLimits = basename == "fake-legacy-text-limits-model";
        const bool legacyText = basename == "fake-legacy-text-model" || explicitLegacyLimits;
        const bool httpTools = basename == "fake-http-tools-model";
        const int observe = basename == "fake-runtime-memory-model" ? 1 :
            basename == "fake-runtime-memory-error-model" ? 2 : basename == "fake-runtime-memory-delay-model" ? 3 : 0;
        if ((basename != "fake-pipe-model" && !(hybrid && basename == "fake-pipe-hybrid-model") && !closeFailure && !legacyText && !httpTools && !observe) ||
            options.maxSequenceLength <= 0)
            throw std::invalid_argument("Unknown CPU fixture model");
        // Independent sentinels verify that the Proxy forwards every budget
        // through the real C ABI and protocol, including both Session limits.
        // This fixture does not pretend to allocate device or pinned memory.
        if (explicitLegacyLimits && (options.maxSequenceLength != 2048 || options.maxSessions != 2 ||
            options.maxSessionCacheBytes != 65536 || !options.allocationLimits ||
            options.allocationLimits->maxDeviceBytes != 111111 ||
            options.allocationLimits->maxPinnedHostBytes != 222222 ||
            options.allocationLimits->maxSessionHostBytes != 333333))
            throw std::invalid_argument("Legacy admission options did not reach the CPU model factory intact");
        if (closeFailure && !shutdownTrace)
            shutdownTrace = std::make_unique<diagnostics::ExecutionTrace>(
                std::make_unique<FinalCloseFlushFailureSink>());
        if (FixtureLogsEnabled()) AILA_LOG_INFO("[FakeLog] model-load");
        auto model = std::make_shared<FakeModel>(options.maxSequenceLength, options.allocationLimits, legacyText, httpTools);
        memoryMode = observe;
        memoryQueries = 0;
        memoryLimits = options.allocationLimits.value_or(aila::ipc::v2::AllocationLimits{});
        memoryModel = model;
        return model;
    }

    // Factory ownership mirrors DeviceRuntime: model/session reaping does not
    // destroy this trace. Only Stop's final factory.reset() triggers the fault.
    std::unique_ptr<diagnostics::ExecutionTrace> shutdownTrace;
    int memoryMode = 0;
    mutable uint64_t memoryQueries = 0; // Accessed only by the lifecycle executor.
    aila::ipc::v2::AllocationLimits memoryLimits;
    std::weak_ptr<FakeModel> memoryModel;
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
