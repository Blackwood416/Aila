#include <windows.h>
#include <tlhelp32.h>

#include "aila_api.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;
using namespace std::chrono_literals;

[[noreturn]] void fail(std::string_view message) {
    throw std::runtime_error(std::string(message));
}

void expect(bool condition, std::string_view message) {
    if (!condition) {
        fail(message);
    }
}

std::wstring wide(std::string_view value) {
    if (value.empty()) {
        return {};
    }
    const int size = MultiByteToWideChar(
        CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), static_cast<int>(value.size()), nullptr, 0);
    if (size <= 0) {
        fail("UTF-8 to UTF-16 conversion failed");
    }
    std::wstring result(static_cast<size_t>(size), L'\0');
    if (MultiByteToWideChar(
            CP_UTF8,
            MB_ERR_INVALID_CHARS,
            value.data(),
            static_cast<int>(value.size()),
            result.data(),
            size) != size) {
        fail("UTF-8 to UTF-16 conversion failed");
    }
    return result;
}

std::string utf8(std::wstring_view value) {
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
        fail("UTF-16 to UTF-8 conversion failed");
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
        fail("UTF-16 to UTF-8 conversion failed");
    }
    return result;
}

std::optional<std::wstring> environment(const wchar_t* name) {
    SetLastError(ERROR_SUCCESS);
    DWORD capacity = GetEnvironmentVariableW(name, nullptr, 0);
    if (capacity == 0) {
        if (GetLastError() == ERROR_ENVVAR_NOT_FOUND) {
            return std::nullopt;
        }
        if (GetLastError() == ERROR_SUCCESS) {
            return std::wstring();
        }
        fail("GetEnvironmentVariableW failed");
    }
    std::vector<wchar_t> buffer(capacity);
    const DWORD copied = GetEnvironmentVariableW(name, buffer.data(), capacity);
    if (copied >= capacity) {
        fail("GetEnvironmentVariableW returned an unstable value");
    }
    return std::wstring(buffer.data(), copied);
}

class ScopedEnvironment {
public:
    ScopedEnvironment(const wchar_t* name, std::optional<std::wstring> value)
        : name_(name), original_(environment(name)) {
        set(std::move(value));
    }

    ~ScopedEnvironment() {
        set(original_);
    }

    ScopedEnvironment(const ScopedEnvironment&) = delete;
    ScopedEnvironment& operator=(const ScopedEnvironment&) = delete;

    void set(std::optional<std::wstring> value) const {
        if (!SetEnvironmentVariableW(name_.c_str(), value ? value->c_str() : nullptr)) {
            fail("SetEnvironmentVariableW failed");
        }
    }

private:
    std::wstring name_;
    std::optional<std::wstring> original_;
};

class TempDirectory {
public:
    TempDirectory() {
        const fs::path root = fs::temp_directory_path();
        for (unsigned counter = 0; counter != 1000; ++counter) {
            path_ = root /
                (L"Aila Proxy 测试-" + std::to_wstring(GetCurrentProcessId()) + L"-" +
                 std::to_wstring(counter));
            std::error_code error;
            if (fs::create_directory(path_, error)) {
                return;
            }
        }
        fail("could not create Unicode proxy test directory");
    }

    ~TempDirectory() {
        std::error_code ignored;
        fs::remove_all(path_, ignored);
    }

    const fs::path& path() const { return path_; }

private:
    fs::path path_;
};

class ScopedCurrentDirectory {
public:
    explicit ScopedCurrentDirectory(const fs::path& path) {
        DWORD capacity = GetCurrentDirectoryW(0, nullptr);
        std::vector<wchar_t> buffer(capacity);
        const DWORD copied = GetCurrentDirectoryW(capacity, buffer.data());
        expect(copied != 0 && copied < capacity, "GetCurrentDirectoryW failed");
        original_.assign(buffer.data(), copied);
        expect(SetCurrentDirectoryW(path.c_str()) != FALSE, "SetCurrentDirectoryW failed");
    }

    ~ScopedCurrentDirectory() { SetCurrentDirectoryW(original_.c_str()); }

private:
    std::wstring original_;
};

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

std::unordered_set<std::string> loaded_modules() {
    HANDLE snapshot = CreateToolhelp32Snapshot(
        TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, GetCurrentProcessId());
    expect(snapshot != INVALID_HANDLE_VALUE, "CreateToolhelp32Snapshot failed");
    std::unordered_set<std::string> result;
    MODULEENTRY32W entry{};
    entry.dwSize = sizeof(entry);
    if (Module32FirstW(snapshot, &entry)) {
        do {
            result.insert(lower(utf8(entry.szModule)));
        } while (Module32NextW(snapshot, &entry));
    }
    CloseHandle(snapshot);
    return result;
}

std::unordered_set<DWORD> child_worker_processes() {
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    expect(snapshot != INVALID_HANDLE_VALUE, "process snapshot failed");
    std::unordered_set<DWORD> result;
    PROCESSENTRY32W entry{};
    entry.dwSize = sizeof(entry);
    if (Process32FirstW(snapshot, &entry)) {
        do {
            if (entry.th32ParentProcessID == GetCurrentProcessId() &&
                lower(utf8(entry.szExeFile)) == "ailaworker.exe") {
                result.insert(entry.th32ProcessID);
            }
        } while (Process32NextW(snapshot, &entry));
    }
    CloseHandle(snapshot);
    return result;
}

bool forbidden_runtime_module(std::string_view module) {
    static constexpr std::string_view patterns[] = {
        "sycl", "dnnl", "tbb", "umf", "ur_loader", "ur_win_proxy"};
    return std::any_of(std::begin(patterns), std::end(patterns), [&](std::string_view pattern) {
        return module.find(pattern) != std::string_view::npos;
    });
}

class LoadedLibrary {
public:
    explicit LoadedLibrary(const fs::path& path) {
        module_ = LoadLibraryExW(
            path.c_str(),
            nullptr,
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!module_) {
            fail("LoadLibraryExW could not load copied AilaShared.dll; error=" +
                 std::to_string(GetLastError()));
        }
    }

    ~LoadedLibrary() {
        if (module_) {
            FreeLibrary(module_);
        }
    }

    template <typename Function>
    Function symbol(const char* name) const {
        FARPROC address = GetProcAddress(module_, name);
        if (!address) {
            fail(std::string("missing proxy export: ") + name);
        }
        return reinterpret_cast<Function>(address);
    }

private:
    HMODULE module_ = nullptr;
};

struct Api {
    using Version = const char* (*)();
    using Create = AilaEngine* (*)();
    using Init = int (*)(AilaEngine*, const char*, int);
    using Destroy = void (*)(AilaEngine*);
    using DefaultConfig = AilaGenConfig (*)();
    using DefaultConfigV2 = AilaGenConfigV2 (*)();
    using Reset = void (*)(AilaEngine*);
    using ContextLength = int (*)(AilaEngine*);
    using LastErrorCode = int (*)(AilaEngine*);
    using LastErrorMessage = const char* (*)(AilaEngine*);
    using FreeString = void (*)(char*);
    using FreeSamples = void (*)(float*);
    using FreeAlignedWords = void (*)(AilaAlignedWord*, int);
    using SetLogCallback = void (*)(AilaLogCallback, void*);
    using SetLogLevel = void (*)(int);

    explicit Api(const LoadedLibrary& library)
        : version(library.symbol<Version>("aila_version")),
          create(library.symbol<Create>("aila_engine_create")),
          init(library.symbol<Init>("aila_engine_init")),
          destroy(library.symbol<Destroy>("aila_engine_destroy")),
          default_config(library.symbol<DefaultConfig>("aila_default_gen_config")),
          default_config_v2(library.symbol<DefaultConfigV2>("aila_default_gen_config_v2")),
          reset(library.symbol<Reset>("aila_engine_reset_context")),
          context_length(library.symbol<ContextLength>("aila_engine_context_length")),
          last_error_code(library.symbol<LastErrorCode>("aila_last_error_code")),
          last_error_message(library.symbol<LastErrorMessage>("aila_last_error_message")),
          free_string(library.symbol<FreeString>("aila_free_string")),
          free_samples(library.symbol<FreeSamples>("aila_free_samples")),
          free_aligned_words(
              library.symbol<FreeAlignedWords>("aila_free_aligned_words")),
          set_log_callback(library.symbol<SetLogCallback>("aila_set_log_callback")),
          set_log_level(library.symbol<SetLogLevel>("aila_set_log_level")) {}

    Version version;
    Create create;
    Init init;
    Destroy destroy;
    DefaultConfig default_config;
    DefaultConfigV2 default_config_v2;
    Reset reset;
    ContextLength context_length;
    LastErrorCode last_error_code;
    LastErrorMessage last_error_message;
    FreeString free_string;
    FreeSamples free_samples;
    FreeAlignedWords free_aligned_words;
    SetLogCallback set_log_callback;
    SetLogLevel set_log_level;
};

class EngineHandle {
public:
    explicit EngineHandle(const Api& api) : api_(&api), engine_(api.create()) {
        expect(engine_ != nullptr, "aila_engine_create returned NULL");
    }
    ~EngineHandle() { api_->destroy(engine_); }
    EngineHandle(const EngineHandle&) = delete;
    EngineHandle& operator=(const EngineHandle&) = delete;
    AilaEngine* get() const { return engine_; }
    AilaEngine* release() { return std::exchange(engine_, nullptr); }

private:
    const Api* api_;
    AilaEngine* engine_;
};

void copy_test_file(const fs::path& source, const fs::path& destination) {
    fs::create_directories(destination.parent_path());
    std::error_code error;
    fs::copy_file(source, destination, fs::copy_options::overwrite_existing, error);
    if (error) {
        fail("copy failed from '" + utf8(source.wstring()) + "' to '" +
             utf8(destination.wstring()) + "': " + error.message());
    }
}

std::vector<std::string> read_marker(const fs::path& marker) {
    for (int attempt = 0; attempt != 100; ++attempt) {
        std::ifstream input(marker);
        std::vector<std::string> lines;
        for (std::string line; std::getline(input, line);) {
            if (!line.empty()) {
                lines.push_back(std::move(line));
            }
        }
        if (!lines.empty()) {
            return lines;
        }
        std::this_thread::sleep_for(10ms);
    }
    fail("fake worker did not write its lifecycle marker");
}

std::optional<DWORD> marker_pid(
    const std::vector<std::string>& lines,
    std::string_view model,
    int max_seq_len) {
    const std::string suffix = "|" + std::string(model) + "|" + std::to_string(max_seq_len);
    for (const std::string& line : lines) {
        if (line.size() > suffix.size() &&
            line.compare(line.size() - suffix.size(), suffix.size(), suffix) == 0) {
            return static_cast<DWORD>(std::stoul(line.substr(0, line.size() - suffix.size())));
        }
    }
    return std::nullopt;
}

void expect_error_contains(const Api& api, AilaEngine* engine, std::string_view needle) {
    expect(api.last_error_code(engine) == AILA_ERR_RUNTIME, "expected AILA_ERR_RUNTIME");
    const char* pointer = api.last_error_message(engine);
    expect(pointer != nullptr, "last error message was NULL");
    const std::string first(pointer);
    expect(!first.empty(), "last error message was empty");
    expect(lower(first).find(lower(std::string(needle))) != std::string::npos,
           "last error message was not actionable: " + first);
    expect(std::string(api.last_error_message(engine)) == first,
           "last error message changed without another engine operation");
    expect(api.last_error_message(engine) == pointer,
           "last error pointer was not stable without another engine operation");
}

void verify_defaults(const Api& api) {
    expect(std::string(api.version()) == "0.1.6", "proxy version changed");
    const AilaGenConfig config = api.default_config();
    expect(config.max_new_tokens == 512, "default max_new_tokens changed");
    expect(config.temperature == 0.6f, "default temperature changed");
    expect(config.top_k == 20, "default top_k changed");
    expect(config.top_p == 0.95f, "default top_p changed");
    expect(config.repetition_penalty == 1.0f, "default repetition_penalty changed");
    expect(config.presence_penalty == 0.0f, "default presence_penalty changed");
    expect(config.frequency_penalty == 0.0f, "default frequency_penalty changed");
    expect(config.do_sample == 1, "default do_sample changed");
    expect(config.decode_chunk_size == 12, "default decode_chunk_size changed");
    expect(config.stream_chunk_size == 4, "default stream_chunk_size changed");

    const AilaGenConfigV2 v2 = api.default_config_v2();
    expect(v2.struct_size == sizeof(AilaGenConfigV2), "V2 struct_size changed");
    expect(v2.max_new_tokens == config.max_new_tokens, "V2 max_new_tokens changed");
    expect(v2.temperature == config.temperature, "V2 temperature changed");
    expect(v2.top_k == config.top_k, "V2 top_k changed");
    expect(v2.top_p == config.top_p, "V2 top_p changed");
    expect(v2.repetition_penalty == config.repetition_penalty,
           "V2 repetition_penalty changed");
    expect(v2.presence_penalty == config.presence_penalty,
           "V2 presence_penalty changed");
    expect(v2.frequency_penalty == config.frequency_penalty,
           "V2 frequency_penalty changed");
    expect(v2.do_sample == config.do_sample, "V2 do_sample changed");
    expect(v2.decode_chunk_size == config.decode_chunk_size,
           "V2 decode_chunk_size changed");
    expect(v2.stream_chunk_size == config.stream_chunk_size,
           "V2 stream_chunk_size changed");
    expect(v2.thinking_budget_tokens == -1, "V2 thinking budget changed");
    expect(v2.sampling_seed == 42, "V2 sampling seed changed");
    expect(v2.use_fixed_seed == 0, "V2 fixed-seed default changed");
    for (int value : v2.reserved) {
        expect(value == 0, "V2 reserved field was not zero");
    }
}

void test_proxy_abi_and_lifecycle() {
    TempDirectory temp;
    const fs::path integration_root = temp.path() / L"split root";
    const fs::path runtime = integration_root / L"aila_runtime";
    const fs::path copied_proxy = integration_root / L"AilaShared.dll";
    const fs::path copied_worker = runtime / L"AilaWorker.exe";
    const fs::path marker = temp.path() / L"worker lifecycle.txt";
    const fs::path elsewhere = temp.path() / L"unrelated cwd";
    fs::create_directories(elsewhere);
    copy_test_file(fs::path(wide(AILA_PROXY_DLL_PATH)), copied_proxy);
    copy_test_file(fs::path(wide(AILA_FAKE_WORKER_PATH)), copied_worker);

    ScopedEnvironment runtime_directory(L"AILA_RUNTIME_DLL_DIR", L"aila_runtime");
    ScopedEnvironment build_id(L"AILA_FAKE_WORKER_BUILD_ID", wide(AILA_BUILD_ID));
    ScopedEnvironment lifecycle_marker(
        L"AILA_FAKE_WORKER_LIFECYCLE_MARKER", marker.wstring());
    ScopedCurrentDirectory cwd(elsewhere);

    const auto modules_before = loaded_modules();
    LoadedLibrary library(copied_proxy);
    const auto modules_after_load = loaded_modules();
    for (const std::string& module : modules_after_load) {
        if (modules_before.find(module) == modules_before.end()) {
            expect(!forbidden_runtime_module(module),
                   "proxy load imported accelerator runtime module: " + module);
        }
    }

    Api api(library);
    verify_defaults(api);
    api.destroy(nullptr);
    api.reset(nullptr);
    expect(api.context_length(nullptr) == 0, "NULL context length was not zero");
    expect(api.last_error_code(nullptr) == AILA_ERR_INVALID_ARGUMENT,
           "NULL last error code was not invalid argument");
    expect(std::string(api.last_error_message(nullptr)).empty(),
           "NULL last error message was not empty");
    api.free_string(nullptr);
    api.free_samples(nullptr);
    api.free_aligned_words(nullptr, 0);
    api.set_log_callback(nullptr, nullptr);
    api.set_log_level(0);
    api.set_log_level(3);

    constexpr std::string_view model = u8R"(C:\models\模型 "quoted")";
    const auto workers_before_create = child_worker_processes();
    EngineHandle relative(api);
    expect(child_worker_processes() == workers_before_create,
           "aila_engine_create launched a worker process");
    expect(api.init(relative.get(), model.data(), 4096) == 0,
           "relative split-layout initialization failed");
    expect(api.last_error_code(relative.get()) == AILA_OK,
           "successful init did not clear last error");
    expect(api.context_length(relative.get()) == 4096,
           "fake context length was not deterministic after init");
    auto lines = read_marker(marker);
    const auto relative_pid = marker_pid(lines, model, 4096);
    expect(relative_pid.has_value(), "fake worker did not receive UTF-8 model/max length");
    api.reset(relative.get());
    expect(api.context_length(relative.get()) == 0, "reset did not clear fake context");

    {
        runtime_directory.set(runtime.wstring());
        EngineHandle absolute(api);
        expect(api.init(absolute.get(), "absolute-model", 3072) == 0,
               "absolute runtime directory initialization failed");
    }

    copy_test_file(
        fs::path(wide(AILA_FAKE_WORKER_PATH)), integration_root / L"AilaWorker.exe");
    runtime_directory.set(std::nullopt);
    {
        EngineHandle flat(api);
        expect(api.init(flat.get(), "flat-model", 2048) == 0,
               "unset flat-layout initialization failed");
    }

    runtime_directory.set(L"missing runtime");
    {
        EngineHandle missing(api);
        expect(api.init(missing.get(), "missing-model", 1024) != 0,
               "missing runtime directory unexpectedly initialized");
        expect_error_contains(api, missing.get(), "does not exist");
    }

    const fs::path empty_runtime = integration_root / L"empty runtime";
    fs::create_directories(empty_runtime);
    runtime_directory.set(empty_runtime.wstring());
    {
        EngineHandle no_worker(api);
        expect(api.init(no_worker.get(), "no-worker-model", 1024) != 0,
               "runtime without a worker unexpectedly initialized");
        expect_error_contains(api, no_worker.get(), "AilaWorker.exe");
    }

    runtime_directory.set(runtime.wstring());
    build_id.set(L"wrong-build-id");
    {
        EngineHandle retry(api);
        expect(api.init(retry.get(), "retry-model", 1536) != 0,
               "wrong worker build ID unexpectedly initialized");
        expect_error_contains(api, retry.get(), "build ID mismatch");
        build_id.set(wide(AILA_BUILD_ID));
        expect(api.init(retry.get(), "retry-model", 1536) == 0,
               "engine could not retry after a failed init");
        expect(api.init(retry.get(), "retry-model", 1536) != 0,
               "repeated successful init was not rejected");
        expect(api.last_error_code(retry.get()) == AILA_ERR_INVALID_ARGUMENT,
               "repeated init did not report invalid argument");
    }

    build_id.set(wide(AILA_BUILD_ID));
    {
        EngineHandle first(api);
        EngineHandle second(api);
        expect(api.init(first.get(), "multi-one", 1111) == 0,
               "first independent engine init failed");
        expect(api.init(second.get(), "multi-two", 2222) == 0,
               "second independent engine init failed");
        for (int attempt = 0; attempt != 100; ++attempt) {
            lines = read_marker(marker);
            if (marker_pid(lines, "multi-one", 1111) && marker_pid(lines, "multi-two", 2222)) {
                break;
            }
            std::this_thread::sleep_for(10ms);
        }
        const auto first_pid = marker_pid(lines, "multi-one", 1111);
        const auto second_pid = marker_pid(lines, "multi-two", 2222);
        expect(first_pid.has_value() && second_pid.has_value(),
               "multi-engine worker PIDs were not observable");
        expect(*first_pid != *second_pid, "multiple engines shared one worker process");
        api.destroy(first.release());
        expect(api.context_length(second.get()) == 2222,
               "destroying one engine broke another worker");
    }

    const auto modules_after_lifecycle = loaded_modules();
    for (const std::string& module : modules_after_lifecycle) {
        if (modules_before.find(module) == modules_before.end()) {
            expect(!forbidden_runtime_module(module),
                   "proxy lifecycle loaded accelerator runtime module: " + module);
        }
    }
}

} // namespace

int main() {
    try {
        test_proxy_abi_and_lifecycle();
        std::cout << "Aila proxy ABI tests passed\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "Aila proxy ABI tests failed: " << exception.what() << '\n';
        return 1;
    }
}
