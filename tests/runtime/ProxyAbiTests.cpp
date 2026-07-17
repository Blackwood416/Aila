#include <windows.h>
#include <tlhelp32.h>

#include "aila_api.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cctype>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
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
    using Generate = char* (*)(AilaEngine*, const char*, const AilaGenConfig*);
    using GenerateEx = char* (*)(AilaEngine*, const char*, const AilaGenConfigV2*);
    using GenerateStream = int (*)(AilaEngine*, const char*, const AilaGenConfig*,
                                   AilaTokenCallback, void*);
    using GenerateStreamEx = int (*)(AilaEngine*, const char*, const AilaGenConfigV2*,
                                     AilaChatStreamCallback, void*);
    using Transcribe = char* (*)(AilaEngine*, const char*, const AilaGenConfig*, const char*,
                                 const char*, float, int, AilaTokenCallback, void*, char**);
    using AsrStreamCreate = AilaTranscribeStream* (*)(AilaEngine*, const AilaGenConfig*,
                                                       const char*, const char*);
    using AsrStreamFeed = int (*)(AilaTranscribeStream*, const float*, int);
    using AsrStreamGetText = int (*)(AilaTranscribeStream*, char**, char**);
    using AsrStreamDestroy = void (*)(AilaTranscribeStream*);
    using SynthesizeWav = int (*)(AilaEngine*, const int*, int, const float*, int,
                                  const AilaGenConfig*, float**, int*);
    using SynthesizeText = int (*)(AilaEngine*, const char*, const float*, int,
                                   const AilaGenConfig*, float**, int*);
    using SynthesizeFile = int (*)(AilaEngine*, const char*, const char*, const char*,
                                   const char*, const char*, const AilaGenConfig*, const char*);
    using DecodeMimi = int (*)(AilaEngine*, const int32_t*, int, float**, int*);
    using ExtractEmbedding = int (*)(AilaEngine*, const char*, float**, int*);
    using Align = int (*)(AilaEngine*, const float*, int, int, const char*, const char*,
                          AilaAlignedWord**, int*);
    using AlignWords = int (*)(AilaEngine*, const float*, int, int, const char* const*, int,
                               AilaAlignedWord**, int*);
    using SynthesizeStream = AilaTTSStream* (*)(AilaEngine*, const char*, const char*,
                                                const char*, const char*, const char*,
                                                const AilaGenConfig*, AilaAudioCallback, void*);
    using StreamWait = int (*)(AilaTTSStream*);
    using StreamDestroy = void (*)(AilaTTSStream*);
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
          generate(library.symbol<Generate>("aila_generate")),
          generate_messages(library.symbol<Generate>("aila_generate_messages")),
          generate_chat_json(library.symbol<Generate>("aila_generate_chat_json")),
          generate_chat_json_ex(
              library.symbol<GenerateEx>("aila_generate_chat_json_ex")),
          generate_stream(library.symbol<GenerateStream>("aila_generate_stream")),
          generate_messages_stream(
              library.symbol<GenerateStream>("aila_generate_messages_stream")),
          generate_chat_json_stream_ex(
              library.symbol<GenerateStreamEx>("aila_generate_chat_json_stream_ex")),
          transcribe(library.symbol<Transcribe>("aila_transcribe")),
          asr_stream_create(library.symbol<AsrStreamCreate>("aila_transcribe_stream_create")),
          asr_stream_feed(library.symbol<AsrStreamFeed>("aila_transcribe_stream_feed")),
          asr_stream_get_text(library.symbol<AsrStreamGetText>("aila_transcribe_stream_get_text")),
          asr_stream_destroy(library.symbol<AsrStreamDestroy>("aila_transcribe_stream_destroy")),
          synthesize_wav(library.symbol<SynthesizeWav>("aila_synthesize_wav")),
          synthesize_text(library.symbol<SynthesizeText>("aila_synthesize_text_to_wav")),
          synthesize_file(library.symbol<SynthesizeFile>("aila_synthesize")),
          decode_mimi(library.symbol<DecodeMimi>("aila_decode_mimi_vocoder")),
          extract_embedding(library.symbol<ExtractEmbedding>("aila_extract_speaker_embedding")),
          align(library.symbol<Align>("aila_align")),
          align_words(library.symbol<AlignWords>("aila_align_words")),
          synthesize_stream(library.symbol<SynthesizeStream>("aila_synthesize_stream")),
          stream_wait(library.symbol<StreamWait>("aila_stream_wait")),
          stream_destroy(library.symbol<StreamDestroy>("aila_stream_destroy")),
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
    Generate generate;
    Generate generate_messages;
    Generate generate_chat_json;
    GenerateEx generate_chat_json_ex;
    GenerateStream generate_stream;
    GenerateStream generate_messages_stream;
    GenerateStreamEx generate_chat_json_stream_ex;
    Transcribe transcribe;
    AsrStreamCreate asr_stream_create;
    AsrStreamFeed asr_stream_feed;
    AsrStreamGetText asr_stream_get_text;
    AsrStreamDestroy asr_stream_destroy;
    SynthesizeWav synthesize_wav;
    SynthesizeText synthesize_text;
    SynthesizeFile synthesize_file;
    DecodeMimi decode_mimi;
    ExtractEmbedding extract_embedding;
    Align align;
    AlignWords align_words;
    SynthesizeStream synthesize_stream;
    StreamWait stream_wait;
    StreamDestroy stream_destroy;
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

void verify_asr_proxy(const Api& api, AilaEngine* engine) {
    AilaGenConfig config = api.default_config();
    config.max_new_tokens = 321;
    config.temperature = 0.25f;
    config.top_k = 17;
    config.top_p = 0.75f;
    config.repetition_penalty = 1.125f;
    config.presence_penalty = 0.5f;
    config.frequency_penalty = -0.25f;
    config.do_sample = 0;
    config.decode_chunk_size = 9;
    config.stream_chunk_size = 3;

    struct Capture { std::thread::id caller; int calls = 0; bool wrong_thread = false; };
    Capture capture{std::this_thread::get_id()};
    const auto callback = [](const char*, void* opaque) -> int {
        auto& capture = *static_cast<Capture*>(opaque);
        capture.wrong_thread = capture.wrong_thread ||
            std::this_thread::get_id() != capture.caller;
        ++capture.calls;
        return 1; // Offline ASR intentionally ignores the callback return.
    };
    char* language = reinterpret_cast<char*>(static_cast<uintptr_t>(1));
    char* transcript = api.transcribe(
        engine, u8R"(C:\音频\说话.wav)", &config, "", nullptr, 12.5f, 7,
        callback, &capture, &language);
    expect(transcript != nullptr, "offline ASR returned NULL");
    expect(language != nullptr && transcript != language,
           "offline ASR transcript/language allocations were not independent");
    const std::string transcript_text(transcript);
    expect(transcript_text.find(u8R"("wavPath":"C:\\音频\\说话.wav")") != std::string::npos,
           "offline ASR changed Unicode path");
    expect(transcript_text.find(R"("max_new_tokens":321)") != std::string::npos,
           "offline ASR dropped legacy config");
    expect(transcript_text.find(R"("forcedLanguage":"")") != std::string::npos,
           "offline ASR changed empty forced language");
    expect(transcript_text.find(R"("systemPrompt":null)") != std::string::npos,
           "offline ASR changed nullable system prompt");
    expect(transcript_text.find(R"("segmentSec":12.5)") != std::string::npos,
           "offline ASR changed segment length");
    expect(transcript_text.find(R"("pastTextConditioning":7)") != std::string::npos,
           "offline ASR changed conditioning flag");
    expect(std::string(language) == "Chinese", "offline ASR language changed");
    expect(!capture.wrong_thread && capture.calls == 2,
           "offline ASR callbacks did not run completely on caller thread");
    api.free_string(transcript);
    api.free_string(language);

    struct EmptyCapture { int calls = 0; bool saw_empty = false; } empty_capture;
    const auto empty_callback = [](const char* token, void* opaque) -> int {
        auto& capture = *static_cast<EmptyCapture*>(opaque);
        ++capture.calls;
        capture.saw_empty = token != nullptr && *token == '\0';
        return 0;
    };
    language = reinterpret_cast<char*>(static_cast<uintptr_t>(1));
    transcript = api.transcribe(
        engine, "__aila_asr_all_empty__", nullptr, nullptr, nullptr, 0.0f, 0,
        empty_callback, &empty_capture, &language);
    expect(transcript != nullptr && *transcript == '\0' && language == nullptr &&
               empty_capture.calls == 1 && empty_capture.saw_empty,
           "zero-length offline ASR token/result was not preserved safely");
    api.free_string(transcript);

    AilaTranscribeStream* empty_stream =
        api.asr_stream_create(engine, nullptr, nullptr, "empty-text");
    expect(empty_stream != nullptr, "empty ASR stream create failed");
    char* empty_stable = reinterpret_cast<char*>(static_cast<uintptr_t>(1));
    char* empty_partial = reinterpret_cast<char*>(static_cast<uintptr_t>(1));
    expect(api.asr_stream_get_text(empty_stream, &empty_stable, &empty_partial) == AILA_OK &&
               empty_stable == nullptr && empty_partial == nullptr,
           "zero-length ASR stream text did not map to NULL outputs");
    api.asr_stream_destroy(empty_stream);

    language = reinterpret_cast<char*>(static_cast<uintptr_t>(1));
    transcript = api.transcribe(
        engine, "__aila_asr_empty_language__", nullptr, nullptr, "", 0.0f, 0,
        nullptr, nullptr, &language);
    expect(transcript != nullptr && language == nullptr,
           "empty ASR language did not map to NULL output");
    api.free_string(transcript);

    language = reinterpret_cast<char*>(static_cast<uintptr_t>(1));
    expect(api.transcribe(engine, nullptr, nullptr, nullptr, nullptr, 0.0f, 0,
                          nullptr, nullptr, &language) == nullptr && language == nullptr,
           "invalid offline ASR did not clear language_out");
    expect(api.context_length(engine) != 0,
           "local invalid ASR input reaped the healthy worker");
    const uint32_t infinity_bits = 0x7f800000u;
    float infinity = 0.0f;
    std::memcpy(&infinity, &infinity_bits, sizeof(infinity));
    expect(api.transcribe(engine, "finite.wav", nullptr, nullptr, nullptr,
                          infinity, 0,
                          nullptr, nullptr, nullptr) == nullptr,
           "offline ASR accepted a non-finite segment length");
    expect(api.context_length(engine) != 0,
           "non-finite local ASR input reaped the healthy worker");

    const uint32_t nan_bits = 0x7fc00000u;
    float nan_value = 0.0f;
    std::memcpy(&nan_value, &nan_bits, sizeof(nan_value));
    for (int field = 0; field != 5; ++field) {
        AilaGenConfig invalid = api.default_config();
        switch (field) {
            case 0: invalid.temperature = nan_value; break;
            case 1: invalid.top_p = nan_value; break;
            case 2: invalid.repetition_penalty = nan_value; break;
            case 3: invalid.presence_penalty = nan_value; break;
            case 4: invalid.frequency_penalty = nan_value; break;
        }
        expect(api.transcribe(engine, "finite-config.wav", &invalid, nullptr, nullptr,
                              0.0f, 0, nullptr, nullptr, nullptr) == nullptr,
               "offline ASR accepted a non-finite legacy config field");
        expect(api.asr_stream_create(engine, &invalid, nullptr, nullptr) == nullptr,
               "ASR stream create accepted a non-finite legacy config field");
        expect(api.context_length(engine) != 0,
               "non-finite ASR config contacted or reaped the healthy worker");
    }

    AilaTranscribeStream* stream =
        api.asr_stream_create(engine, &config, "expect-floats", u8"系统");
    expect(stream != nullptr, "ASR stream create failed");
    expect(api.asr_stream_feed(stream, nullptr, 3) == AILA_ERR_INVALID_ARGUMENT,
           "ASR stream accepted NULL samples");
    const float samples[] = {1.25f, -2.5f, 0.0f};
    expect(api.asr_stream_feed(stream, samples, 3) == AILA_OK,
           "ASR stream changed raw float attachment");
    char* stable = nullptr;
    char* partial = nullptr;
    expect(api.asr_stream_get_text(stream, &stable, &partial) == AILA_OK,
           "ASR stream get_text failed");
    expect(stable && partial && stable != partial && std::string(stable) == u8"稳定" &&
               std::string(partial) == u8"临时",
           "ASR stream stable/partial outputs changed");
    api.free_string(stable);
    api.free_string(partial);
    expect(api.asr_stream_get_text(stream, nullptr, nullptr) == AILA_OK,
           "ASR stream optional outputs were not optional");
    api.asr_stream_destroy(stream);
    api.asr_stream_destroy(nullptr);
}

void verify_asr_stream_outlives_engine(const Api& api) {
    AilaEngine* engine = api.create();
    expect(engine != nullptr && api.init(engine, "asr-owner", 640) == 0,
           "ASR owner engine init failed");
    AilaTranscribeStream* stream = api.asr_stream_create(engine, nullptr, nullptr, nullptr);
    expect(stream != nullptr, "ASR owner stream creation failed");
    api.destroy(engine);
    const float sample = 0.5f;
    expect(api.asr_stream_feed(stream, &sample, 1) == AILA_OK,
           "destroying engine invalidated its live ASR stream");
    char* stable = nullptr;
    expect(api.asr_stream_get_text(stream, &stable, nullptr) == AILA_OK && stable != nullptr,
           "ASR stream had a dangling engine owner");
    api.free_string(stable);
    api.asr_stream_destroy(stream);
}

void verify_reentrant_asr_destroy_is_deferred(const Api& api, AilaEngine* engine) {
    AilaTranscribeStream* stream = api.asr_stream_create(engine, nullptr, nullptr, nullptr);
    expect(stream != nullptr, "reentrant ASR stream create failed");
    struct Context {
        const Api* api;
        AilaTranscribeStream* stream;
        int calls = 0;
    } context{&api, stream};
    const auto callback = [](const char*, void* opaque) -> int {
        auto& context = *static_cast<Context*>(opaque);
        if (context.calls++ == 0) {
            context.api->asr_stream_destroy(context.stream);
            context.stream = nullptr;
        }
        return 0;
    };
    expect(api.generate_stream(engine, "destroy-asr-in-callback", nullptr,
                               callback, &context) == 0 && context.calls == 2,
           "reentrant ASR destroy deadlocked or failed outer generation");
    AilaTranscribeStream* confirmed =
        api.asr_stream_create(engine, nullptr, nullptr, "require-no-active");
    expect(confirmed != nullptr,
           "deferred ASR destroy was not flushed before stream completion");
    api.asr_stream_destroy(confirmed);
    expect(api.context_length(engine) != 0,
           "reentrant ASR destroy made the worker unusable");

    AilaTranscribeStream* offline_stream =
        api.asr_stream_create(engine, nullptr, nullptr, nullptr);
    expect(offline_stream != nullptr, "offline reentrant ASR stream create failed");
    Context offline_context{&api, offline_stream};
    char* transcript = api.transcribe(
        engine, "offline-reentrant.wav", nullptr, nullptr, nullptr, 0.0f, 0,
        callback, &offline_context, nullptr);
    expect(transcript != nullptr && offline_context.calls == 2,
           "reentrant destroy from offline ASR callback failed");
    api.free_string(transcript);
    AilaTranscribeStream* offline_confirmed =
        api.asr_stream_create(engine, nullptr, nullptr, "require-no-active");
    expect(offline_confirmed != nullptr,
           "offline ASR deferred destroy was not flushed before completion");
    api.asr_stream_destroy(offline_confirmed);
}

void verify_malformed_asr_reaps_worker(const Api& api) {
    {
        const auto before = child_worker_processes();
        EngineHandle engine(api);
        expect(api.init(engine.get(), "bad-asr-result", 700) == 0,
               "malformed ASR engine init failed");
        DWORD pid = 0;
        for (DWORD candidate : child_worker_processes()) {
            if (before.find(candidate) == before.end()) { pid = candidate; break; }
        }
        expect(api.transcribe(engine.get(), "__aila_asr_bad_lengths__", nullptr,
                              nullptr, nullptr, 0.0f, 0, nullptr, nullptr, nullptr) == nullptr,
               "malformed offline ASR lengths succeeded");
        for (int attempt = 0; attempt != 100; ++attempt) {
            const auto current = child_worker_processes();
            if (current.find(pid) == current.end()) break;
            std::this_thread::sleep_for(10ms);
        }
        const auto remaining = child_worker_processes();
        expect(pid != 0 && remaining.find(pid) == remaining.end(),
               "malformed offline ASR result did not reap worker");
    }
    {
        const auto before = child_worker_processes();
        EngineHandle engine(api);
        expect(api.init(engine.get(), "bad-asr-stream-text", 701) == 0,
               "malformed ASR stream engine init failed");
        DWORD pid = 0;
        for (DWORD candidate : child_worker_processes()) {
            if (before.find(candidate) == before.end()) { pid = candidate; break; }
        }
        AilaTranscribeStream* stream =
            api.asr_stream_create(engine.get(), nullptr, nullptr, "malformed-text");
        expect(stream != nullptr, "malformed ASR text stream creation failed");
        char* stable = reinterpret_cast<char*>(static_cast<uintptr_t>(1));
        char* partial = reinterpret_cast<char*>(static_cast<uintptr_t>(1));
        expect(api.asr_stream_get_text(stream, &stable, &partial) == AILA_ERR_RUNTIME &&
                   stable == nullptr && partial == nullptr,
               "malformed ASR text lengths succeeded or leaked outputs");
        for (int attempt = 0; attempt != 100; ++attempt) {
            const auto current = child_worker_processes();
            if (current.find(pid) == current.end()) break;
            std::this_thread::sleep_for(10ms);
        }
        const auto remaining = child_worker_processes();
        expect(pid != 0 && remaining.find(pid) == remaining.end(),
               "malformed ASR stream text did not reap worker");
        api.asr_stream_destroy(stream);
    }
}

void verify_asr_event_identity_and_remote_ids(const Api& api) {
    {
        const auto before = child_worker_processes();
        EngineHandle engine(api);
        expect(api.init(engine.get(), "bad-asr-event", 710) == 0,
               "ASR event identity engine init failed");
        DWORD pid = 0;
        for (DWORD candidate : child_worker_processes()) {
            if (before.find(candidate) == before.end()) { pid = candidate; break; }
        }
        expect(api.transcribe(engine.get(), "__aila_asr_bad_event_protocol__", nullptr,
                              nullptr, nullptr, 0.0f, 0, nullptr, nullptr, nullptr) == nullptr,
               "ASR event with wrong protocol reached the host");
        for (int attempt = 0; attempt != 100; ++attempt) {
            const auto current = child_worker_processes();
            if (current.find(pid) == current.end()) break;
            std::this_thread::sleep_for(10ms);
        }
        const auto remaining = child_worker_processes();
        expect(pid != 0 && remaining.find(pid) == remaining.end(),
               "wrong-protocol ASR event did not reap worker");
    }
    {
        const auto before = child_worker_processes();
        EngineHandle engine(api);
        expect(api.init(engine.get(), "duplicate-asr-id", 711) == 0,
               "duplicate ASR ID engine init failed");
        DWORD pid = 0;
        for (DWORD candidate : child_worker_processes()) {
            if (before.find(candidate) == before.end()) { pid = candidate; break; }
        }
        AilaTranscribeStream* first = api.asr_stream_create(engine.get(), nullptr, nullptr, nullptr);
        expect(first != nullptr, "first ASR ID creation failed");
        AilaTranscribeStream* duplicate =
            api.asr_stream_create(engine.get(), nullptr, nullptr, "duplicate-id");
        expect(duplicate == nullptr, "duplicate/decreasing remote ASR ID was accepted");
        for (int attempt = 0; attempt != 100; ++attempt) {
            const auto current = child_worker_processes();
            if (current.find(pid) == current.end()) break;
            std::this_thread::sleep_for(10ms);
        }
        const auto remaining = child_worker_processes();
        expect(pid != 0 && remaining.find(pid) == remaining.end(),
               "duplicate remote ASR ID did not reap worker");
        api.asr_stream_destroy(first);
    }
}

void verify_stale_asr_handle_cannot_alias_reinitialized_worker(const Api& api) {
    EngineHandle engine(api);
    expect(api.init(engine.get(), "stale-asr-one", 720) == 0,
           "stale ASR first init failed");
    AilaTranscribeStream* stale = api.asr_stream_create(engine.get(), nullptr, nullptr, nullptr);
    expect(stale != nullptr, "stale ASR handle creation failed");
    expect(api.transcribe(engine.get(), "__aila_asr_bad_lengths__", nullptr,
                          nullptr, nullptr, 0.0f, 0, nullptr, nullptr, nullptr) == nullptr,
           "forced ASR worker failure did not fail");
    expect(api.init(engine.get(), "stale-asr-two", 721) == 0,
           "ASR engine could not reinitialize after worker failure");
    AilaTranscribeStream* current = api.asr_stream_create(engine.get(), nullptr, nullptr, nullptr);
    expect(current != nullptr, "ASR stream creation after reinit failed");
    const float sample = 0.5f;
    expect(api.asr_stream_feed(stale, &sample, 1) == AILA_ERR_INVALID_ARGUMENT,
           "stale ASR handle aliased a reused remote ID");
    expect(api.context_length(engine.get()) == 721,
           "stale ASR handle contacted/reaped the reinitialized worker");
    expect(api.asr_stream_feed(current, &sample, 1) == AILA_OK,
           "current ASR handle was invalidated by stale-handle rejection");
    api.asr_stream_destroy(stale);
    char* stable = nullptr;
    expect(api.asr_stream_get_text(current, &stable, nullptr) == AILA_OK && stable != nullptr,
           "stale ASR destroy targeted the current reused remote ID");
    api.free_string(stable);
    api.asr_stream_destroy(current);
}

std::string take_string(const Api& api, char* value, std::string_view operation) {
    expect(value != nullptr, std::string(operation) + " returned NULL");
    const std::string result(value);
    api.free_string(value);
    return result;
}

void expect_contains(
    std::string_view value,
    std::string_view expected,
    std::string_view message) {
    expect(value.find(expected) != std::string_view::npos,
           std::string(message) + ": " + std::string(value));
}

struct TokenStreamCapture {
    std::thread::id caller;
    std::vector<std::string> tokens;
    bool abort_first = false;
    bool wrong_thread = false;
    const Api* reentrant_api = nullptr;
    AilaEngine* reentrant_engine = nullptr;
    int reentrant_error_code = -999;
    int reentrant_context_length = -999;
    bool did_reentrant_check = false;
};

int capture_token(const char* token, void* opaque) {
    auto& capture = *static_cast<TokenStreamCapture*>(opaque);
    capture.wrong_thread = capture.wrong_thread || std::this_thread::get_id() != capture.caller;
    if (capture.reentrant_api && capture.reentrant_engine &&
        !capture.did_reentrant_check) {
        capture.did_reentrant_check = true;
        capture.reentrant_error_code =
            capture.reentrant_api->last_error_code(capture.reentrant_engine);
        capture.reentrant_context_length =
            capture.reentrant_api->context_length(capture.reentrant_engine);
    }
    capture.tokens.emplace_back(token ? token : "<NULL>");
    return capture.abort_first && capture.tokens.size() == 1 ? 1 : 0;
}

struct StructuredCapture {
    std::thread::id caller;
    bool wrong_thread = false;
    int calls = 0;
    uint32_t struct_size = 0;
    int type = -1;
    std::optional<std::string> text;
    std::optional<std::string> tool_call_id;
    std::optional<std::string> tool_name;
    std::optional<std::string> arguments_delta;
    std::optional<std::string> warnings_json;
};

int capture_structured(const AilaChatStreamEvent* event, void* opaque) {
    auto& capture = *static_cast<StructuredCapture*>(opaque);
    capture.wrong_thread = capture.wrong_thread || std::this_thread::get_id() != capture.caller;
    ++capture.calls;
    capture.struct_size = event->struct_size;
    capture.type = event->type;
    auto copy = [](const char* value) -> std::optional<std::string> {
        return value ? std::optional<std::string>(value) : std::nullopt;
    };
    capture.text = copy(event->text);
    capture.tool_call_id = copy(event->tool_call_id);
    capture.tool_name = copy(event->tool_name);
    capture.arguments_delta = copy(event->arguments_delta);
    capture.warnings_json = copy(event->warnings_json);
    return 0;
}

void verify_streaming_generation(const Api& api, AilaEngine* engine) {
    TokenStreamCapture tokens{std::this_thread::get_id()};
    tokens.reentrant_api = &api;
    tokens.reentrant_engine = engine;
    expect(api.generate_stream(engine, u8"流式", nullptr, capture_token, &tokens) == 0,
           "token stream failed");
    expect(!tokens.wrong_thread, "token callback did not run on caller thread");
    expect(tokens.reentrant_error_code == AILA_OK,
           "reentrant last_error_code could not observe stream state");
    expect(tokens.reentrant_context_length == 0,
           "reentrant context_length did not fail promptly as busy");
    expect(tokens.tokens == std::vector<std::string>({"first", u8" 第二"}),
           "token event order or bytes changed");

    TokenStreamCapture messages{std::this_thread::get_id()};
    expect(api.generate_messages_stream(
               engine, R"([{"role":"user","content":"hi"}])", nullptr,
               capture_token, &messages) == 0,
           "messages stream failed");
    expect(messages.tokens == tokens.tokens, "messages stream event order changed");

    TokenStreamCapture aborted{std::this_thread::get_id()};
    aborted.abort_first = true;
    expect(api.generate_stream(engine, "abort", nullptr, capture_token, &aborted) == 1,
           "callback abort status changed");
    expect(aborted.tokens.size() == 1, "callback received tokens after abort");
    expect(!aborted.wrong_thread, "aborting callback ran on wrong thread");
    take_string(api, api.generate(engine, "after-stream-abort", nullptr),
                "worker use after cooperative stream abort");

    TokenStreamCapture short_abort{std::this_thread::get_id()};
    short_abort.abort_first = true;
    expect(api.generate_stream(
               engine, "__aila_stream_short_abort__", nullptr,
               capture_token, &short_abort) == 1,
           "naturally completed callback abort did not return 1");
    expect(short_abort.tokens.size() == 1,
           "short stream invoked host callback after abort");
    take_string(api, api.generate(engine, "after-short-stream-abort", nullptr),
                "worker use after short stream abort race");

    TokenStreamCapture engine_error{std::this_thread::get_id()};
    expect(api.generate_stream(
               engine, "__aila_stream_engine_error__", nullptr,
               capture_token, &engine_error) == -1,
           "stream engine error became success");
    expect(engine_error.tokens == tokens.tokens,
           "valid callbacks before stream engine error changed");
    expect(api.last_error_code(engine) == AILA_ERR_CONTEXT_OVERFLOW &&
               std::string(api.last_error_message(engine)) == "synthetic stream failure",
           "stream engine error code/message changed");
    take_string(api, api.generate(engine, "after-stream-engine-error", nullptr),
                "worker use after ordinary stream engine error");

    AilaGenConfigV2 v2 = api.default_config_v2();
    StructuredCapture structured{std::this_thread::get_id()};
    const int structured_status = api.generate_chat_json_stream_ex(
        engine, "{}", &v2, capture_structured, &structured);
    expect(structured_status == 0,
           std::string("structured stream failed: ") + api.last_error_message(engine));
    expect(structured.calls == 1 && !structured.wrong_thread,
           "structured callback count or thread changed");
    expect(structured.struct_size == sizeof(AilaChatStreamEvent) &&
               structured.type == AILA_CHAT_STREAM_TOOL_CALL_DELTA,
           "structured event ABI metadata changed");
    expect(structured.text == std::optional<std::string>(u8"工具"),
           "structured Unicode text changed");
    expect(!structured.tool_call_id.has_value(), "structured NULL became non-NULL");
    expect(structured.tool_name == std::optional<std::string>("search"),
           "structured tool name changed");
    expect(structured.arguments_delta == std::optional<std::string>(""),
           "structured empty string became NULL");
    expect(structured.warnings_json == std::optional<std::string>("[]"),
           "structured warnings JSON changed");

    expect(api.generate_stream(engine, nullptr, nullptr, capture_token, &tokens) == -1,
           "NULL stream input succeeded");
    expect(api.generate_stream(engine, "x", nullptr, nullptr, nullptr) == -1,
           "NULL stream callback succeeded");
    const char invalid_utf8[] = {static_cast<char>(0xc3), static_cast<char>(0x28), 0};
    expect(api.generate_stream(engine, invalid_utf8, nullptr, capture_token, &tokens) == -1,
           "invalid UTF-8 stream input succeeded");
    AilaGenConfigV2 zero{};
    expect(api.generate_chat_json_stream_ex(
               engine, "{}", &zero, capture_structured, &structured) == -1,
           "zero-sized streaming V2 config succeeded");
    take_string(api, api.generate(engine, "after-local-stream-errors", nullptr),
                "worker after local streaming argument errors");

    TokenStreamCapture reordered{std::this_thread::get_id()};
    expect(api.generate_stream(
               engine, "__aila_stream_response_before_end__", nullptr,
               capture_token, &reordered) == 0,
           "response-before-event stream failed");
    expect(reordered.tokens == tokens.tokens,
           "final response was accepted before preceding stream events");
}

void verify_synchronous_generation(const Api& api, AilaEngine* engine) {
    const std::string generated = take_string(
        api,
        api.generate(engine, u8"Unicode 输入 🌍", nullptr),
        "aila_generate");
    expect_contains(generated, R"("method":"generate")", "generate method missing");
    expect_contains(generated, u8R"("input":"Unicode 输入 🌍")", "Unicode input changed");
    expect_contains(generated, R"("config":null)", "NULL config semantics changed");

    AilaGenConfig config{};
    config.max_new_tokens = 17;
    config.temperature = 0.25f;
    config.top_k = 7;
    config.top_p = 0.75f;
    config.repetition_penalty = 1.125f;
    config.presence_penalty = 0.375f;
    config.frequency_penalty = 0.625f;
    config.do_sample = 0;
    config.decode_chunk_size = 3;
    config.stream_chunk_size = 5;
    const std::string messages = take_string(
        api,
        api.generate_messages(engine, R"([{"role":"user","content":"hi"}])", &config),
        "aila_generate_messages");
    expect_contains(messages, R"("method":"generate.messages")", "messages method missing");
    expect_contains(messages, R"("max_new_tokens":17)", "max_new_tokens changed");
    expect_contains(messages, R"("temperature":0.25)", "temperature changed");
    expect_contains(messages, R"("top_k":7)", "top_k changed");
    expect_contains(messages, R"("top_p":0.75)", "top_p changed");
    expect_contains(messages, R"("repetition_penalty":1.125)", "repetition penalty changed");
    expect_contains(messages, R"("presence_penalty":0.375)", "presence penalty changed");
    expect_contains(messages, R"("frequency_penalty":0.625)", "frequency penalty changed");
    expect_contains(messages, R"("do_sample":0)", "do_sample changed");
    expect_contains(messages, R"("decode_chunk_size":3)", "decode chunk changed");
    expect_contains(messages, R"("stream_chunk_size":5)", "stream chunk changed");

    const std::string chat = take_string(
        api,
        api.generate_chat_json(engine, R"({"messages":[]})", nullptr),
        "aila_generate_chat_json");
    expect_contains(chat, R"("method":"generate.chat_json")", "chat method missing");
    expect_contains(chat, R"("config":null)", "chat NULL config changed");

    AilaGenConfigV2 v2{};
    v2.struct_size = sizeof(v2);
    v2.max_new_tokens = 31;
    v2.temperature = 0.5f;
    v2.top_k = 11;
    v2.top_p = 0.875f;
    v2.repetition_penalty = 1.25f;
    v2.presence_penalty = 0.125f;
    v2.frequency_penalty = 0.25f;
    v2.do_sample = 1;
    v2.decode_chunk_size = 9;
    v2.stream_chunk_size = 6;
    v2.thinking_budget_tokens = 77;
    v2.sampling_seed = 123456789;
    v2.use_fixed_seed = 1;
    for (int& reserved : v2.reserved) {
        reserved = 0x55555555;
    }
    const std::string extended = take_string(
        api,
        api.generate_chat_json_ex(engine, R"({"messages":[]})", &v2),
        "aila_generate_chat_json_ex");
    expect_contains(extended, R"("method":"generate.chat_json_ex")", "chat ex method missing");
    expect_contains(extended, R"("thinking_budget_tokens":77)", "thinking budget changed");
    expect_contains(extended, R"("sampling_seed":123456789)", "sampling seed changed");
    expect_contains(extended, R"("use_fixed_seed":1)", "fixed seed changed");
    expect(extended.find("reserved") == std::string::npos, "reserved V2 fields crossed proxy");

    AilaGenConfigV2 zero_size{};
    expect(api.generate_chat_json_ex(engine, "zero-size", &zero_size) == nullptr,
           "zero-sized V2 config succeeded");
    expect(api.last_error_code(engine) == AILA_ERR_INVALID_ARGUMENT,
           "zero-sized V2 config returned the wrong error");
    expect_contains(
        api.last_error_message(engine),
        "struct_size",
        "zero-sized V2 error was not actionable");
    take_string(api, api.generate(engine, "after-zero-size", nullptr), "call after zero-size V2");
}

void verify_audio_proxy(const Api& api, AilaEngine* engine) {
    const int tokens[]{17, -3};
    const int tokens_copy[]{17, -3};
    const float speaker[]{0.125f, -0.75f};
    float* samples = reinterpret_cast<float*>(static_cast<uintptr_t>(1));
    int count = -1;
    expect(api.synthesize_wav(engine, tokens, 2, speaker, 2, nullptr, &samples, &count) == AILA_OK,
           "synthesize_wav failed");
    expect(count == 3 && samples && samples[0] == 0.25f && samples[1] == -0.5f && samples[2] == 1.0f,
           "synthesize_wav result bytes changed");
    expect(std::memcmp(tokens, tokens_copy, sizeof(tokens)) == 0 &&
               speaker[0] == 0.125f && speaker[1] == -0.75f,
           "synthesize_wav changed caller inputs");
    api.free_samples(samples);

    for (const auto& weird : std::vector<std::pair<const float*, int>>{
             {nullptr, 17}, {speaker, 0}, {speaker, -7}}) {
        samples = nullptr;
        count = 0;
        expect(api.synthesize_wav(
                   engine, tokens, 2, weird.first, weird.second, nullptr,
                   &samples, &count) == AILA_OK && count == 3,
               "synthesize_wav did not normalize a non-present embedding");
        api.free_samples(samples);
        samples = nullptr;
        count = 0;
        expect(api.synthesize_text(
                   engine, "__aila_expect_empty_embedding__", weird.first, weird.second,
                   nullptr, &samples, &count) == AILA_OK && count == 3,
               "synthesize_text_to_wav did not normalize a non-present embedding");
        api.free_samples(samples);
    }

    samples = reinterpret_cast<float*>(static_cast<uintptr_t>(1));
    count = -1;
    expect(api.synthesize_text(engine, u8"你好", speaker, 2, nullptr, &samples, &count) == AILA_OK &&
               count == 3 && samples != nullptr,
           "synthesize_text_to_wav failed");
    api.free_samples(samples);

    int32_t codes[32]{};
    samples = nullptr;
    count = 0;
    expect(api.decode_mimi(engine, codes, 2, &samples, &count) == AILA_OK && count == 3,
           "Mimi decode failed");
    api.free_samples(samples);

    samples = nullptr;
    count = 0;
    expect(api.extract_embedding(engine, u8R"(C:\声音\参考.wav)", &samples, &count) == AILA_OK &&
               count == 3,
           "speaker embedding extraction failed");
    api.free_samples(samples);

    expect(api.synthesize_file(engine, u8"文字", nullptr, "", nullptr, "chinese", nullptr,
                               u8R"(C:\输出\结果.wav)") == AILA_OK,
           "worker-side WAV synthesis failed");
    expect(api.synthesize_file(engine, "empty output path", nullptr, nullptr, nullptr, nullptr,
                               nullptr, "") == AILA_OK,
           "non-NULL empty WAV path was rejected locally");
    samples = nullptr;
    count = 0;
    expect(api.extract_embedding(engine, "", &samples, &count) == AILA_OK && count == 3,
           "non-NULL empty embedding path was rejected locally");
    api.free_samples(samples);

    samples = reinterpret_cast<float*>(static_cast<uintptr_t>(1));
    count = 19;
    expect(api.synthesize_wav(engine, nullptr, 2, nullptr, 0, nullptr, &samples, &count) ==
               AILA_ERR_INVALID_ARGUMENT && samples == nullptr && count == 0,
           "invalid audio arguments did not clear outputs");
    take_string(api, api.generate(engine, "after-local-audio-error", nullptr),
                "worker after local audio validation");
}

void verify_alignment_proxy(const Api& api, AilaEngine* engine) {
    const float audio[]{0.25f, -0.5f, 1.0f};
    const float original[]{0.25f, -0.5f, 1.0f};
    AilaAlignedWord* words = reinterpret_cast<AilaAlignedWord*>(static_cast<uintptr_t>(1));
    int count = -1;
    expect(api.align(engine, audio, 3, 0, u8"Unicode 原文", "", &words, &count) == AILA_OK,
           "aila_align failed");
    expect(count == 3 && words && std::string(words[0].text) == u8"相同" &&
               std::string(words[1].text) == u8"相同" && std::string(words[2].text).empty() &&
               words[0].start_ms == -7 && words[0].end_ms == 11 &&
               words[1].start_ms == 12 && words[1].end_ms == 24,
           "aligned words/timestamps changed");
    expect(words[0].text != words[1].text && words[1].text != words[2].text,
           "equal/empty aligned texts did not receive independent host allocations");
    expect(std::memcmp(audio, original, sizeof(audio)) == 0,
           "alignment changed caller audio");
    api.free_aligned_words(words, count);
    api.free_aligned_words(nullptr, 7);

    const char* input_words[]{u8"猫", nullptr, u8"dog🐕"};
    words = nullptr;
    count = 0;
    expect(api.align_words(engine, audio, 3, -1, input_words, 3, &words, &count) == AILA_OK &&
               count == 3 && words != nullptr,
           "aila_align_words failed or NULL word was not normalized to empty");
    api.free_aligned_words(words, count);

    words = reinterpret_cast<AilaAlignedWord*>(static_cast<uintptr_t>(1));
    count = 99;
    expect(api.align(engine, audio, 3, 16000, "__aila_align_empty__", "English",
                     &words, &count) == AILA_OK && words == nullptr && count == 0,
           "empty alignment result was not normalized");

    words = reinterpret_cast<AilaAlignedWord*>(static_cast<uintptr_t>(1));
    count = 19;
    expect(api.align(engine, nullptr, 3, 16000, "x", "y", &words, &count) ==
               AILA_ERR_INVALID_ARGUMENT && words == nullptr && count == 0,
           "invalid alignment arguments did not clear outputs");
    expect(api.align_words(engine, audio, 3, 16000, nullptr, 1, &words, &count) ==
               AILA_ERR_INVALID_ARGUMENT,
           "NULL pre-tokenized word array succeeded");
    take_string(api, api.generate(engine, "after-local-align-error", nullptr),
                "worker after local alignment validation");
}

void verify_malformed_alignment_results_reap_worker(const Api& api) {
    for (const char* marker : {
             "__aila_align_count_overflow__", "__aila_align_count_mismatch__",
             "__aila_align_bad_utf8__", "__aila_align_bad_nul__",
             "__aila_align_missing_text__", "__aila_align_bad_attachment__",
             "__aila_align_bad_identity__"}) {
        EngineHandle engine(api);
        expect(api.init(engine.get(), marker, 1024) == 0,
               std::string("alignment malformed init failed: ") + marker);
        const float audio[]{0.0f};
        AilaAlignedWord* words = reinterpret_cast<AilaAlignedWord*>(static_cast<uintptr_t>(1));
        int count = 9;
        expect(api.align(engine.get(), audio, 1, 16000, marker, "English", &words, &count) ==
                   AILA_ERR_RUNTIME && words == nullptr && count == 0,
               std::string("malformed alignment response succeeded: ") + marker);
        expect(api.generate(engine.get(), "after-malformed-align", nullptr) == nullptr &&
                   api.last_error_code(engine.get()) == AILA_ERR_INVALID_ARGUMENT,
               std::string("malformed alignment response did not reap worker: ") + marker);
    }
}

struct AudioCapture {
    std::mutex mutex;
    std::vector<float> samples;
    std::thread::id callback_thread;
};

void capture_audio(const float* samples, int count, void* opaque) {
    auto* capture = static_cast<AudioCapture*>(opaque);
    std::lock_guard<std::mutex> lock(capture->mutex);
    capture->callback_thread = std::this_thread::get_id();
    capture->samples.insert(capture->samples.end(), samples, samples + count);
}

void verify_tts_stream_outlives_engine(const Api& api) {
    EngineHandle engine(api);
    expect(api.init(engine.get(), "tts-stream-owner", 1024) == 0, "TTS stream engine init failed");
    AudioCapture capture;
    const std::thread::id caller = std::this_thread::get_id();
    AilaTTSStream* stream = api.synthesize_stream(
        engine.get(), u8"流式语音", nullptr, "", nullptr, "chinese", nullptr,
        capture_audio, &capture);
    expect(stream != nullptr, "TTS stream creation failed");
    api.destroy(engine.release());
    expect(api.stream_wait(stream) == AILA_OK && api.stream_wait(stream) == AILA_OK,
           "TTS stream wait was not idempotent");
    {
        std::lock_guard<std::mutex> lock(capture.mutex);
        expect(capture.samples == std::vector<float>({0.25f, -0.5f, 1.0f}),
               "TTS stream chunk order/content changed");
        expect(capture.callback_thread != caller, "TTS callback ran on caller thread");
    }
    api.stream_destroy(stream);
    api.stream_destroy(nullptr);
    expect(api.stream_wait(nullptr) == AILA_ERR_INVALID_ARGUMENT,
           "NULL TTS stream wait returned wrong status");
}

void verify_malformed_audio_results_reap_worker(const Api& api) {
    for (const char* marker : {
             "__aila_audio_bad_count__", "__aila_audio_bad_element__",
             "__aila_audio_nan__", "__aila_audio_bad_identity__"}) {
        EngineHandle engine(api);
        expect(api.init(engine.get(), marker, 1024) == 0,
               std::string("malformed audio init failed: ") + marker);
        float* values = reinterpret_cast<float*>(static_cast<uintptr_t>(1));
        int count = 19;
        expect(api.synthesize_text(engine.get(), marker, nullptr, 0, nullptr,
                                   &values, &count) == AILA_ERR_RUNTIME &&
                   values == nullptr && count == 0,
               std::string("malformed audio result succeeded: ") + marker);
        expect(api.generate(engine.get(), "after-malformed-audio", nullptr) == nullptr &&
                   api.last_error_code(engine.get()) == AILA_ERR_INVALID_ARGUMENT,
               std::string("malformed audio result did not reap worker: ") + marker);
    }
}

void verify_tts_stream_destroy_cancels_promptly(const Api& api) {
    EngineHandle engine(api);
    expect(api.init(engine.get(), "tts-stream-cancel", 1024) == 0,
           "TTS cancellation engine init failed");
    AudioCapture capture;
    AilaTTSStream* stream = api.synthesize_stream(
        engine.get(), "__aila_tts_slow__", nullptr, nullptr, nullptr, nullptr, nullptr,
        capture_audio, &capture);
    expect(stream != nullptr, "slow TTS stream creation failed");
    for (int attempt = 0; attempt != 100; ++attempt) {
        {
            std::lock_guard<std::mutex> lock(capture.mutex);
            if (!capture.samples.empty()) break;
        }
        std::this_thread::sleep_for(5ms);
    }
    const auto started = std::chrono::steady_clock::now();
    api.stream_destroy(stream);
    const auto elapsed = std::chrono::steady_clock::now() - started;
    expect(elapsed < 3s, "TTS stream destroy was not bounded");
    size_t after_destroy = 0;
    {
        std::lock_guard<std::mutex> lock(capture.mutex);
        after_destroy = capture.samples.size();
    }
    std::this_thread::sleep_for(50ms);
    {
        std::lock_guard<std::mutex> lock(capture.mutex);
        expect(capture.samples.size() == after_destroy,
               "TTS callback ran after stream destroy returned");
    }
}

struct ReentrantTtsDestroyContext {
    const Api* api = nullptr;
    std::atomic<AilaTTSStream*> published{nullptr};
    std::atomic<int> callback_count{0};
    std::atomic<int> wait_status{AILA_OK};
    std::atomic_bool destroy_returned{false};
};

void destroy_tts_stream_on_later_chunk(const float*, int, void* opaque) {
    auto* context = static_cast<ReentrantTtsDestroyContext*>(opaque);
    const int call = context->callback_count.fetch_add(1, std::memory_order_acq_rel) + 1;
    if (call < 2) return;
    AilaTTSStream* stream = context->published.load(std::memory_order_acquire);
    if (!stream || context->destroy_returned.load(std::memory_order_acquire)) return;
    context->wait_status.store(
        context->api->stream_wait(stream), std::memory_order_release);
    context->api->stream_destroy(stream);
    context->destroy_returned.store(true, std::memory_order_release);
}

void verify_tts_stream_reentrant_destroy(const Api& api) {
    EngineHandle engine(api);
    expect(api.init(engine.get(), "tts-reentrant-destroy", 1024) == 0,
           "reentrant TTS engine init failed");
    ReentrantTtsDestroyContext context;
    context.api = &api;
    AilaTTSStream* stream = api.synthesize_stream(
        engine.get(), "__aila_tts_reentrant_destroy__", nullptr, nullptr, nullptr, nullptr,
        nullptr, destroy_tts_stream_on_later_chunk, &context);
    expect(stream != nullptr, "reentrant TTS stream creation failed");
    context.published.store(stream, std::memory_order_release);

    for (int attempt = 0; attempt != 200 &&
         !context.destroy_returned.load(std::memory_order_acquire); ++attempt) {
        std::this_thread::sleep_for(5ms);
    }
    expect(context.destroy_returned.load(std::memory_order_acquire),
           "reentrant TTS destroy did not return");
    expect(context.wait_status.load(std::memory_order_acquire) == AILA_ERR_RUNTIME,
           "reentrant TTS wait attempted to join its own callback thread");
    const int callbacks_after_destroy = context.callback_count.load(std::memory_order_acquire);
    std::this_thread::sleep_for(100ms);
    expect(context.callback_count.load(std::memory_order_acquire) == callbacks_after_destroy,
           "TTS callback ran after reentrant destroy returned");

    char* generated = nullptr;
    for (int attempt = 0; attempt != 200 && !generated; ++attempt) {
        generated = api.generate(engine.get(), "after-reentrant-TTS-destroy", nullptr);
        if (!generated) std::this_thread::sleep_for(5ms);
    }
    expect(generated != nullptr,
           "engine did not recover after reentrant TTS cancellation");
    api.free_string(generated);
}

void verify_v2_prefix_does_not_read_past_struct_size(const Api& api, AilaEngine* engine) {
    SYSTEM_INFO info{};
    GetSystemInfo(&info);
    const size_t page_size = info.dwPageSize;
    auto* pages = static_cast<unsigned char*>(VirtualAlloc(
        nullptr, page_size * 2, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    expect(pages != nullptr, "VirtualAlloc for V2 guard pages failed");
    DWORD old_protection = 0;
    expect(VirtualProtect(pages + page_size, page_size, PAGE_NOACCESS, &old_protection) != FALSE,
           "VirtualProtect for V2 guard page failed");

    constexpr uint32_t prefix_size = offsetof(AilaGenConfigV2, top_k);
    static_assert(prefix_size == offsetof(AilaGenConfigV2, temperature) + sizeof(float));
    unsigned char* prefix = pages + page_size - prefix_size;
    const uint32_t declared_size = prefix_size;
    const int max_new_tokens = 44;
    const float temperature = 0.375f;
    std::memcpy(prefix + offsetof(AilaGenConfigV2, struct_size), &declared_size, sizeof(declared_size));
    std::memcpy(prefix + offsetof(AilaGenConfigV2, max_new_tokens), &max_new_tokens, sizeof(max_new_tokens));
    std::memcpy(prefix + offsetof(AilaGenConfigV2, temperature), &temperature, sizeof(temperature));

    char* raw = api.generate_chat_json_ex(
        engine,
        "guarded-prefix",
        reinterpret_cast<const AilaGenConfigV2*>(prefix));
    const std::string result = take_string(api, raw, "prefix-sized V2 config");
    expect_contains(result, R"("struct_size":12)", "prefix struct size changed");
    expect_contains(result, R"("max_new_tokens":44)", "prefix token count changed");
    expect_contains(result, R"("temperature":0.375)", "prefix temperature changed");
    expect(result.find("top_k") == std::string::npos, "proxy read beyond declared V2 prefix");
    VirtualFree(pages, 0, MEM_RELEASE);
}

void verify_generation_errors_and_allocations(const Api& api, AilaEngine* engine) {
    expect(api.generate(nullptr, "x", nullptr) == nullptr, "NULL engine generation succeeded");
    expect(api.generate(engine, nullptr, nullptr) == nullptr, "NULL input generation succeeded");
    expect(api.last_error_code(engine) == AILA_ERR_INVALID_ARGUMENT,
           "NULL input returned the wrong error code");

    const char invalid_utf8[] = {static_cast<char>(0xc3), static_cast<char>(0x28), '\0'};
    expect(api.generate(engine, invalid_utf8, nullptr) == nullptr,
           "invalid UTF-8 input succeeded");
    expect(api.last_error_code(engine) == AILA_ERR_INVALID_ARGUMENT,
           "invalid UTF-8 input returned the wrong error code");
    expect_contains(
        api.last_error_message(engine),
        "UTF-8",
        "invalid UTF-8 input error was not actionable");
    take_string(api, api.generate(engine, "after-invalid-UTF-8", nullptr),
                "valid call after invalid UTF-8 input");

    char* first = api.generate(engine, "repeat", nullptr);
    char* second = api.generate(engine, "repeat", nullptr);
    expect(first != nullptr && second != nullptr, "repeat generation returned NULL");
    expect(first != second, "repeat generation aliased returned allocations");
    expect(std::string(first) == std::string(second), "repeat generation bytes changed");
    api.free_string(first);
    api.free_string(second);

    char* empty = api.generate(engine, "__aila_empty__", nullptr);
    expect(empty != nullptr, "successful empty result returned NULL");
    expect(*empty == '\0', "successful empty result was not NUL-terminated");
    api.free_string(empty);

    expect(api.generate(engine, "__aila_error__", nullptr) == nullptr,
           "worker generation error became success");
    expect(api.last_error_code(engine) == AILA_ERR_CONTEXT_OVERFLOW,
           "worker generation error code changed");
    expect(std::string(api.last_error_message(engine)) == "synthetic generation failure",
           "worker generation error message changed");
}

void verify_uninitialized_generation(const Api& api) {
    EngineHandle engine(api);
    expect(api.generate(engine.get(), "x", nullptr) == nullptr,
           "uninitialized generation succeeded");
    expect(api.last_error_code(engine.get()) == AILA_ERR_INVALID_ARGUMENT,
           "uninitialized generation returned the wrong error");
}

void verify_malformed_generation_response_reaps_worker(const Api& api) {
    const auto before = child_worker_processes();
    EngineHandle engine(api);
    expect(api.init(engine.get(), "malformed-generation", 1024) == 0,
           "malformed-response test init failed");
    const auto started = child_worker_processes();
    DWORD worker_pid = 0;
    for (DWORD pid : started) {
        if (before.find(pid) == before.end()) {
            worker_pid = pid;
            break;
        }
    }
    expect(worker_pid != 0, "malformed-response worker PID was not observable");

    expect(api.generate(engine.get(), "__aila_malformed_attachment__", nullptr) == nullptr,
           "malformed response attachment became success");
    expect(api.last_error_code(engine.get()) == AILA_ERR_RUNTIME,
           "malformed response returned the wrong error code");
    expect_contains(
        api.last_error_message(engine.get()),
        "embedded NUL",
        "malformed response error was not actionable");
    for (int attempt = 0; attempt != 100; ++attempt) {
        const auto current = child_worker_processes();
        if (current.find(worker_pid) == current.end()) {
            break;
        }
        std::this_thread::sleep_for(10ms);
    }
    const auto remaining = child_worker_processes();
    expect(remaining.find(worker_pid) == remaining.end(), "malformed response did not reap worker");
    expect(api.generate(engine.get(), "after-malformed", nullptr) == nullptr,
           "generation succeeded after malformed worker shutdown");
    expect(api.last_error_code(engine.get()) == AILA_ERR_INVALID_ARGUMENT,
           "post-malformed generation did not report uninitialized engine");
}

void verify_invalid_utf8_generation_response_reaps_worker(const Api& api) {
    const auto before = child_worker_processes();
    EngineHandle engine(api);
    expect(api.init(engine.get(), "invalid-utf8-generation", 1024) == 0,
           "invalid-UTF-8-response test init failed");
    const auto started = child_worker_processes();
    DWORD worker_pid = 0;
    for (DWORD pid : started) {
        if (before.find(pid) == before.end()) {
            worker_pid = pid;
            break;
        }
    }
    expect(worker_pid != 0, "invalid-UTF-8-response worker PID was not observable");

    expect(api.generate(engine.get(), "__aila_invalid_utf8__", nullptr) == nullptr,
           "invalid UTF-8 response attachment became success");
    expect(api.last_error_code(engine.get()) == AILA_ERR_RUNTIME,
           "invalid UTF-8 response returned the wrong error code");
    expect_contains(
        api.last_error_message(engine.get()),
        "UTF-8",
        "invalid UTF-8 response error was not actionable");
    for (int attempt = 0; attempt != 100; ++attempt) {
        const auto current = child_worker_processes();
        if (current.find(worker_pid) == current.end()) {
            break;
        }
        std::this_thread::sleep_for(10ms);
    }
    const auto remaining = child_worker_processes();
    expect(remaining.find(worker_pid) == remaining.end(),
           "invalid UTF-8 response did not reap worker");
}

void verify_malformed_stream_events_reap_worker(const Api& api) {
    const char* token_cases[] = {
        "__aila_stream_bad_byte_count__",
        "__aila_stream_bad_utf8__",
        "__aila_stream_bad_identity__",
        "__aila_stream_bad_request_id__",
        "__aila_stream_bad_protocol__",
        "__aila_stream_bad_nul__",
        "__aila_stream_bad_schema__",
        "__aila_stream_duplicate_end__",
        "__aila_stream_post_end_data__",
        "__aila_stream_unsolicited_abort__",
        "__aila_stream_error_missing_count__",
        "__aila_stream_error_bad_count_type__",
        "__aila_stream_error_oversized_count__",
        "__aila_stream_error_zero_count__",
        "__aila_stream_error_post_end__",
        "__aila_stream_early_exit__",
    };
    for (const char* marker : token_cases) {
        const auto before = child_worker_processes();
        EngineHandle engine(api);
        expect(api.init(engine.get(), marker, 1024) == 0,
               std::string("stream malformed init failed: ") + marker);
        DWORD pid = 0;
        for (DWORD candidate : child_worker_processes()) {
            if (before.find(candidate) == before.end()) { pid = candidate; break; }
        }
        expect(pid != 0, "malformed stream worker PID was not observable");
        TokenStreamCapture capture{std::this_thread::get_id()};
        const auto started_at = std::chrono::steady_clock::now();
        expect(api.generate_stream(engine.get(), marker, nullptr, capture_token, &capture) == -1,
               std::string("malformed stream event succeeded: ") + marker);
        expect(std::chrono::steady_clock::now() - started_at < 2s,
               std::string("malformed stream failure was not prompt: ") + marker);
        expect(api.last_error_code(engine.get()) == AILA_ERR_RUNTIME,
               std::string("malformed stream event returned the wrong error: ") + marker);
        for (int attempt = 0; attempt != 100; ++attempt) {
            const auto current = child_worker_processes();
            if (current.find(pid) == current.end()) break;
            std::this_thread::sleep_for(10ms);
        }
        const auto remaining = child_worker_processes();
        expect(remaining.find(pid) == remaining.end(),
               std::string("malformed stream event did not reap worker: ") + marker);
    }

    const auto before = child_worker_processes();
    EngineHandle engine(api);
    expect(api.init(engine.get(), "bad-structured-stream", 1024) == 0,
           "bad structured stream init failed");
    DWORD pid = 0;
    for (DWORD candidate : child_worker_processes()) {
        if (before.find(candidate) == before.end()) { pid = candidate; break; }
    }
    StructuredCapture capture{std::this_thread::get_id()};
    AilaGenConfigV2 config = api.default_config_v2();
    expect(api.generate_chat_json_stream_ex(
               engine.get(), "__aila_stream_bad_type__", &config,
               capture_structured, &capture) == -1,
           "invalid structured stream type succeeded");
    for (int attempt = 0; attempt != 100; ++attempt) {
        const auto current = child_worker_processes();
        if (current.find(pid) == current.end()) break;
        std::this_thread::sleep_for(10ms);
    }
    const auto remaining = child_worker_processes();
    expect(pid != 0 && remaining.find(pid) == remaining.end(),
           "invalid structured stream type did not reap worker");
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
    verify_uninitialized_generation(api);
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
    const int relative_init = api.init(relative.get(), model.data(), 4096);
    expect(relative_init == 0,
           std::string("relative split-layout initialization failed: ") +
               api.last_error_message(relative.get()));
    expect(api.last_error_code(relative.get()) == AILA_OK,
           "successful init did not clear last error");
    expect(api.context_length(relative.get()) == 4096,
           "fake context length was not deterministic after init");
    verify_synchronous_generation(api, relative.get());
    verify_audio_proxy(api, relative.get());
    verify_alignment_proxy(api, relative.get());
    verify_streaming_generation(api, relative.get());
    verify_asr_proxy(api, relative.get());
    verify_reentrant_asr_destroy_is_deferred(api, relative.get());
    verify_v2_prefix_does_not_read_past_struct_size(api, relative.get());
    verify_generation_errors_and_allocations(api, relative.get());
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
    verify_malformed_generation_response_reaps_worker(api);
    verify_invalid_utf8_generation_response_reaps_worker(api);
    verify_malformed_stream_events_reap_worker(api);
    verify_asr_stream_outlives_engine(api);
    verify_malformed_asr_reaps_worker(api);
    verify_asr_event_identity_and_remote_ids(api);
    verify_stale_asr_handle_cannot_alias_reinitialized_worker(api);
    verify_tts_stream_outlives_engine(api);
    verify_malformed_audio_results_reap_worker(api);
    verify_malformed_alignment_results_reap_worker(api);
    verify_tts_stream_destroy_cancels_promptly(api);
    verify_tts_stream_reentrant_destroy(api);
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
