#include <windows.h>

#include "runtime/ChildEnvironment.hpp"
#include "runtime/RuntimeDirectory.hpp"
#include "runtime/WorkerProcess.hpp"

#include "simdjson.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;

#ifndef AILA_FAKE_WORKER_PATH
#error AILA_FAKE_WORKER_PATH must name the built fake worker
#endif

using namespace std::chrono_literals;

[[noreturn]] void fail(const char* test_name, const std::string& message) {
    throw std::runtime_error(std::string("FAILED: ") + test_name + ": " + message);
}

void expect(bool condition, const char* test_name, const std::string& message) {
    if (!condition) {
        fail(test_name, message);
    }
}

template <typename Function>
std::string expect_runtime_error(Function&& function, const char* test_name) {
    try {
        function();
    } catch (const std::runtime_error& exception) {
        return exception.what();
    }
    fail(test_name, "expected std::runtime_error");
}

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
        throw std::runtime_error("WideCharToMultiByte size query failed");
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
        throw std::runtime_error("WideCharToMultiByte conversion failed");
    }
    return result;
}

class TempDirectory {
public:
    TempDirectory() {
        wchar_t temp_path[MAX_PATH + 1]{};
        const DWORD length = GetTempPathW(MAX_PATH, temp_path);
        if (length == 0 || length > MAX_PATH) {
            throw std::runtime_error("GetTempPathW failed");
        }
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = fs::path(temp_path) /
            (L"AilaRuntimeIsolation-" + std::to_wstring(GetCurrentProcessId()) + L"-" +
             std::to_wstring(stamp)) /
            L"插件";
        fs::create_directories(path_);
    }

    TempDirectory(const TempDirectory&) = delete;
    TempDirectory& operator=(const TempDirectory&) = delete;

    ~TempDirectory() {
        std::error_code error;
        fs::remove_all(path_.parent_path(), error);
    }

    const fs::path& path() const { return path_; }

private:
    fs::path path_;
};

class CurrentDirectoryRestore {
public:
    CurrentDirectoryRestore() : original_(fs::current_path()) {}
    ~CurrentDirectoryRestore() {
        std::error_code error;
        fs::current_path(original_, error);
    }

private:
    fs::path original_;
};

class EnvironmentRestore {
public:
    explicit EnvironmentRestore(std::wstring name) : name_(std::move(name)) {
        SetLastError(ERROR_SUCCESS);
        const DWORD required = GetEnvironmentVariableW(name_.c_str(), nullptr, 0);
        if (required != 0) {
            std::wstring buffer(required, L'\0');
            const DWORD copied = GetEnvironmentVariableW(name_.c_str(), buffer.data(), required);
            if (copied >= required) {
                throw std::runtime_error("could not snapshot environment variable");
            }
            buffer.resize(copied);
            original_ = std::move(buffer);
        } else if (GetLastError() != ERROR_ENVVAR_NOT_FOUND) {
            original_ = std::wstring{};
        }
    }

    EnvironmentRestore(const EnvironmentRestore&) = delete;
    EnvironmentRestore& operator=(const EnvironmentRestore&) = delete;

    ~EnvironmentRestore() {
        SetEnvironmentVariableW(name_.c_str(), original_ ? original_->c_str() : nullptr);
    }

private:
    std::wstring name_;
    std::optional<std::wstring> original_;
};

fs::path stage_fake_worker(const TempDirectory& temp) {
    const fs::path source = fs::path(AILA_FAKE_WORKER_PATH);
    const fs::path destination = temp.path() / L"AilaWorker.exe";
    std::error_code error;
    fs::copy_file(source, destination, fs::copy_options::overwrite_existing, error);
    if (error) {
        throw std::runtime_error("could not stage fake worker: " + error.message());
    }
    return fs::absolute(destination).lexically_normal();
}

aila::runtime::ExpectedHandshake expected_fake_handshake(std::string build_id = "fake-worker-v1") {
    return {
        aila::ipc::kProtocolVersion,
        aila::ipc::kPublicAbiVersion,
        std::move(build_id),
    };
}

aila::ipc::Frame request_frame(
    uint64_t request_id,
    std::string method,
    std::string payload_json = "{}",
    std::vector<std::byte> attachment = {}) {
    aila::ipc::Frame frame;
    frame.header.request_id = request_id;
    frame.header.kind = "request";
    frame.header.method = std::move(method);
    frame.header.payload_json = std::move(payload_json);
    frame.attachment = std::move(attachment);
    return frame;
}

std::vector<std::byte> bytes(std::initializer_list<unsigned int> values) {
    std::vector<std::byte> result;
    result.reserve(values.size());
    for (const unsigned int value : values) {
        result.push_back(static_cast<std::byte>(value));
    }
    return result;
}

struct Inspection {
    std::string build_id;
    std::string executable;
    std::string runtime_directory;
    std::string current_directory;
    std::string path;
    std::string sentinel;
};

Inspection parse_inspection(const std::string& payload_json, const char* test_name) {
    simdjson::dom::parser parser;
    simdjson::dom::element root;
    expect(
        parser.parse(payload_json).get(root) == simdjson::SUCCESS,
        test_name,
        "worker inspection payload was not valid JSON");

    auto required_string = [&](const char* field) {
        std::string_view value;
        expect(
            root[field].get_string().get(value) == simdjson::SUCCESS,
            test_name,
            std::string("inspection payload omitted string field '") + field + "'");
        return std::string(value);
    };
    return {
        required_string("buildId"),
        required_string("executable"),
        required_string("runtimeDirectory"),
        required_string("currentDirectory"),
        required_string("path"),
        required_string("sentinel"),
    };
}

class TestHandle {
public:
    explicit TestHandle(HANDLE handle) : handle_(handle) {}
    ~TestHandle() {
        if (handle_ != nullptr && handle_ != INVALID_HANDLE_VALUE) {
            CloseHandle(handle_);
        }
    }
    TestHandle(const TestHandle&) = delete;
    TestHandle& operator=(const TestHandle&) = delete;
    HANDLE get() const { return handle_; }

private:
    HANDLE handle_;
};

DWORD process_handle_count() {
    DWORD count = 0;
    if (GetProcessHandleCount(GetCurrentProcess(), &count) == FALSE) {
        throw std::runtime_error("GetProcessHandleCount failed");
    }
    return count;
}

void create_empty_file(const fs::path& path) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("could not create test file");
    }
}

fs::path normalized_absolute(const fs::path& path) {
    return fs::absolute(path).lexically_normal();
}

fs::path long_existing_path(const fs::path& path) {
    const fs::path normalized = normalized_absolute(path);
    const DWORD required = GetLongPathNameW(normalized.c_str(), nullptr, 0);
    if (required == 0) {
        throw std::runtime_error("GetLongPathNameW size query failed");
    }
    std::vector<wchar_t> buffer(static_cast<size_t>(required) + 1);
    const DWORD copied = GetLongPathNameW(
        normalized.c_str(),
        buffer.data(),
        static_cast<DWORD>(buffer.size()));
    if (copied == 0 || copied >= buffer.size()) {
        throw std::runtime_error("GetLongPathNameW failed");
    }
    return fs::path(std::wstring(buffer.data(), copied)).lexically_normal();
}

void expect_path_error(
    const fs::path& candidate,
    const char* test_name,
    const std::string& expected_detail = {}) {
    const std::string message = expect_runtime_error(
        [&] { (void)aila::runtime::require_worker_executable(candidate); },
        test_name);
    expect(
        message.find("AILA_RUNTIME_DLL_DIR") != std::string::npos,
        test_name,
        "diagnostic did not name AILA_RUNTIME_DLL_DIR: " + message);
    expect(
        message.find(utf8(normalized_absolute(candidate).wstring())) != std::string::npos,
        test_name,
        "diagnostic did not name normalized candidate: " + message);
    if (!expected_detail.empty()) {
        expect(
            message.find(expected_detail) != std::string::npos,
            test_name,
            "diagnostic did not contain expected detail: " + message);
    }
}

void test_runtime_directory_resolution() {
    constexpr const char* name = "runtime directory resolution";
    TempDirectory temp;
    const fs::path module = temp.path() / L"AilaShared.dll";
    create_empty_file(module);
    const fs::path relative_runtime = temp.path() / L"aila_runtime";
    const fs::path absolute_runtime = temp.path() / L"absolute runtime";
    fs::create_directories(relative_runtime);
    fs::create_directories(absolute_runtime);
    create_empty_file(relative_runtime / L"AilaWorker.exe");
    create_empty_file(absolute_runtime / L"AilaWorker.exe");

    expect(
        aila::runtime::resolve_runtime_directory(module, std::nullopt) ==
            normalized_absolute(module.parent_path()),
        name,
        "unset override did not select the proxy directory");
    expect(
        aila::runtime::resolve_runtime_directory(module, L"aila_runtime") ==
            normalized_absolute(relative_runtime),
        name,
        "relative override was not based on the proxy directory");
    expect(
        aila::runtime::resolve_runtime_directory(module, absolute_runtime.wstring()) ==
            normalized_absolute(absolute_runtime),
        name,
        "absolute override changed unexpectedly");

    const fs::path different_cwd = temp.path() / L"different cwd";
    fs::create_directories(different_cwd);
    CurrentDirectoryRestore restore;
    fs::current_path(different_cwd);
    expect(
        aila::runtime::resolve_runtime_directory(module, L"aila_runtime") ==
            normalized_absolute(relative_runtime),
        name,
        "current working directory affected relative resolution");

    expect(
        aila::runtime::resolve_runtime_directory(
            module,
            std::optional<std::wstring>{L"folder\\..\\.\\aila_runtime"}) ==
            normalized_absolute(relative_runtime),
        name,
        "dotted relative override was not lexically normalized");
}

void test_worker_validation() {
    constexpr const char* name = "worker executable validation";
    TempDirectory temp;

    const fs::path valid = temp.path() / L"valid runtime";
    fs::create_directories(valid);
    create_empty_file(valid / L"AilaWorker.exe");
    expect(
        aila::runtime::require_worker_executable(valid / L".") ==
            normalized_absolute(valid / L"AilaWorker.exe"),
        name,
        "valid worker path was not returned normalized and absolute");

    expect_path_error(temp.path() / L"missing runtime", name, "directory");

    const fs::path ordinary_file = temp.path() / L"not a directory";
    create_empty_file(ordinary_file);
    expect_path_error(ordinary_file, name, "directory");

    const fs::path missing_worker = temp.path() / L"missing worker";
    fs::create_directories(missing_worker);
    expect_path_error(missing_worker, name, "AilaWorker.exe");

    const fs::path worker_is_directory = temp.path() / L"worker is directory";
    fs::create_directories(worker_is_directory / L"AilaWorker.exe");
    expect_path_error(worker_is_directory, name, "AilaWorker.exe");
}

void test_runtime_directory_override_reads_unicode() {
    constexpr const char* name = "runtime directory override environment read";
    constexpr const wchar_t* variable = L"AILA_RUNTIME_DLL_DIR";
    EnvironmentRestore restore(variable);

    const std::wstring configured = L"C:\\runtime files\\插件\\.\\oneAPI";
    expect(SetEnvironmentVariableW(variable, configured.c_str()) != FALSE, name, "set failed");
    expect(
        aila::runtime::runtime_directory_override() ==
            std::optional<std::wstring>{configured},
        name,
        "Unicode environment value was not preserved exactly");

    expect(SetEnvironmentVariableW(variable, nullptr) != FALSE, name, "unset failed");
    expect(
        !aila::runtime::runtime_directory_override().has_value(),
        name,
        "unset variable did not map to nullopt");

    expect(SetEnvironmentVariableW(variable, L"") != FALSE, name, "empty set failed");
    expect(
        !aila::runtime::runtime_directory_override().has_value(),
        name,
        "empty variable did not map to nullopt");
}

struct ParsedEnvironment {
    aila::runtime::EnvironmentMap values;
    std::vector<std::wstring> names;
};

ParsedEnvironment parse_serialized_environment(
    const std::vector<wchar_t>& block,
    const char* test_name) {
    expect(block.size() >= 2, test_name, "environment block was too small");
    expect(block[block.size() - 1] == L'\0', test_name, "block lacked final NUL");
    expect(block[block.size() - 2] == L'\0', test_name, "block lacked double-NUL termination");
    if (block.size() > 2) {
        expect(block[block.size() - 3] != L'\0', test_name, "block had excess trailing NULs");
    }

    ParsedEnvironment parsed;
    size_t offset = 0;
    while (offset + 1 < block.size() && block[offset] != L'\0') {
        const auto end = std::find(block.begin() + static_cast<std::ptrdiff_t>(offset), block.end(), L'\0');
        expect(end != block.end(), test_name, "entry lacked terminator");
        const std::wstring entry(block.begin() + static_cast<std::ptrdiff_t>(offset), end);
        const size_t equals =
            entry.front() == L'=' ? entry.find(L'=', 1) : entry.find(L'=');
        expect(equals != std::wstring::npos, test_name, "malformed environment entry");
        const std::wstring key = entry.substr(0, equals);
        const std::wstring value = entry.substr(equals + 1);
        expect(parsed.values.emplace(key, value).second, test_name, "duplicate case-insensitive key");
        parsed.names.push_back(key);
        offset += entry.size() + 1;
    }
    expect(offset == block.size() - 1, test_name, "block terminated before its final NUL");
    return parsed;
}

std::vector<wchar_t> make_environment_block(
    std::initializer_list<std::wstring> entries) {
    std::vector<wchar_t> block;
    for (const std::wstring& entry : entries) {
        block.insert(block.end(), entry.begin(), entry.end());
        block.push_back(L'\0');
    }
    block.push_back(L'\0');
    if (block.size() == 1) {
        block.push_back(L'\0');
    }
    return block;
}

void test_isolated_environment_replaces_path() {
    constexpr const char* name = "isolated child environment";
    aila::runtime::EnvironmentMap inherited{
        {L"Path", L"C:\\HostPython;C:\\HostOneAPI"},
        {L"AILA_LOG_LEVEL", L"debug"},
        {L"Other", L"retained=value"},
    };
    const fs::path runtime = L"C:\\Aila Runtime\\插件";
    const fs::path system_root = L"D:\\Windows";
    const std::wstring expected_path =
        runtime.wstring() + L";" + (system_root / L"System32").wstring() + L";" +
        system_root.wstring();

    const std::vector<wchar_t> block =
        aila::runtime::build_isolated_environment(inherited, runtime, system_root);
    const ParsedEnvironment parsed = parse_serialized_environment(block, name);

    expect(parsed.values.size() == 3, name, "variable count changed unexpectedly");
    expect(parsed.values.at(L"AILA_LOG_LEVEL") == L"debug", name, "AILA variable changed");
    expect(parsed.values.at(L"OTHER") == L"retained=value", name, "ordinary variable changed");
    expect(parsed.values.at(L"PATH") == expected_path, name, "PATH was not replaced exactly");
    expect(
        parsed.values.at(L"PATH").find(L"HostPython") == std::wstring::npos &&
            parsed.values.at(L"PATH").find(L"HostOneAPI") == std::wstring::npos,
        name,
        "host PATH entries leaked into child PATH");

    aila::runtime::CaseInsensitiveLess less;
    for (size_t index = 1; index < parsed.names.size(); ++index) {
        expect(
            less(parsed.names[index - 1], parsed.names[index]),
            name,
            "environment entries were not sorted case-insensitively");
    }
}

void test_hidden_drive_environment_entries_are_preserved() {
    constexpr const char* name = "hidden drive environment entries";
    const std::vector<wchar_t> source = make_environment_block({
        L"=D:=D:\\work",
        L"=ExitCode=00000000",
        L"AILA_LOG_LEVEL=debug",
        L"Path=C:\\HostPython",
    });
    const aila::runtime::EnvironmentMap inherited =
        aila::runtime::parse_environment_block(source);

    expect(inherited.size() == 3, name, "synthetic environment entry count changed");
    expect(inherited.at(L"=d:") == L"D:\\work", name, "hidden drive entry was not parsed");
    expect(
        inherited.find(L"=ExitCode") == inherited.end(),
        name,
        "non-drive native pseudo entry was not skipped");
    expect(inherited.at(L"aila_log_level") == L"debug", name, "ordinary entry was not parsed");

    const std::vector<wchar_t> isolated = aila::runtime::build_isolated_environment(
        inherited,
        fs::path(L"C:\\Aila Runtime"),
        fs::path(L"C:\\Windows"));
    const ParsedEnvironment parsed = parse_serialized_environment(isolated, name);
    expect(parsed.values.size() == 3, name, "hidden drive entry changed variable count");
    expect(parsed.values.at(L"=D:") == L"D:\\work", name, "hidden drive entry changed");
    expect(parsed.values.at(L"AILA_LOG_LEVEL") == L"debug", name, "AILA entry changed");
    expect(parsed.names.front() == L"=D:", name, "hidden drive entry was not sorted first");

    const std::wstring first_entry(isolated.data());
    expect(first_entry == L"=D:=D:\\work", name, "hidden drive entry was not serialized exactly");
}

void test_isolated_environment_validates_entries() {
    constexpr const char* name = "isolated child environment validation";
    const fs::path runtime = L"C:\\runtime";
    const fs::path system_root = L"C:\\Windows";

    aila::runtime::EnvironmentMap bad_equals{{L"BAD=NAME", L"value"}};
    (void)expect_runtime_error(
        [&] { (void)aila::runtime::build_isolated_environment(bad_equals, runtime, system_root); },
        name);

    aila::runtime::EnvironmentMap bad_hidden_drive{{L"=1:", L"C:\\work"}};
    (void)expect_runtime_error(
        [&] {
            (void)aila::runtime::build_isolated_environment(
                bad_hidden_drive,
                runtime,
                system_root);
        },
        name);

    aila::runtime::EnvironmentMap bad_name_nul;
    bad_name_nul.emplace(std::wstring(L"BAD\0NAME", 8), L"value");
    (void)expect_runtime_error(
        [&] { (void)aila::runtime::build_isolated_environment(bad_name_nul, runtime, system_root); },
        name);

    aila::runtime::EnvironmentMap bad_value_nul;
    bad_value_nul.emplace(L"BAD_VALUE", std::wstring(L"bad\0value", 9));
    (void)expect_runtime_error(
        [&] { (void)aila::runtime::build_isolated_environment(bad_value_nul, runtime, system_root); },
        name);

    aila::runtime::EnvironmentMap equals_in_value{{L"LEGAL", L"left=right"}};
    const ParsedEnvironment parsed = parse_serialized_environment(
        aila::runtime::build_isolated_environment(equals_in_value, runtime, system_root),
        name);
    expect(parsed.values.at(L"LEGAL") == L"left=right", name, "equals in value was not preserved");
}

void test_current_environment_and_system_root() {
    constexpr const char* name = "current environment and system root";
    constexpr const wchar_t* variable = L"AILA_RUNTIME_ISOLATION_TEST_VALUE";
    EnvironmentRestore restore(variable);
    const std::wstring value = L"Unicode 插件 value";
    expect(SetEnvironmentVariableW(variable, value.c_str()) != FALSE, name, "set failed");

    const aila::runtime::EnvironmentMap environment = aila::runtime::current_environment();
    const auto found = environment.find(variable);
    expect(found != environment.end(), name, "current environment omitted test variable");
    expect(found->second == value, name, "current environment changed Unicode value");

    const fs::path system_root = aila::runtime::system_root_directory();
    expect(system_root.is_absolute(), name, "system root was not absolute");
    expect(fs::is_directory(system_root), name, "system root was not an existing directory");
}

void test_proxy_module_path() {
    constexpr const char* name = "proxy module path";
    const fs::path module_path = aila::runtime::proxy_module_path(GetModuleHandleW(nullptr));
    expect(module_path.is_absolute(), name, "module path was not absolute");
    expect(module_path == module_path.lexically_normal(), name, "module path was not normalized");
    expect(fs::is_regular_file(module_path), name, "module path did not name the test executable");
}

void test_worker_start_isolates_process_context_and_queues_events() {
    constexpr const char* name = "worker start isolation and queued event pipe";
    EnvironmentRestore path_restore(L"PATH");
    EnvironmentRestore sentinel_restore(L"AILA_TEST_SENTINEL");
    EnvironmentRestore unrelated_handle_restore(L"AILA_TEST_UNRELATED_HANDLE");
    expect(
        SetEnvironmentVariableW(L"PATH", L"C:\\HostPython;C:\\Intel\\oneAPI") != FALSE,
        name,
        "could not install synthetic host PATH");
    expect(
        SetEnvironmentVariableW(L"AILA_TEST_SENTINEL", L"retained-插件") != FALSE,
        name,
        "could not install inherited sentinel");
    SECURITY_ATTRIBUTES inheritable_security{};
    inheritable_security.nLength = sizeof(inheritable_security);
    inheritable_security.bInheritHandle = TRUE;
    const std::wstring unrelated_event_name =
        L"Local\\AilaNoInherit-" + std::to_wstring(GetCurrentProcessId()) + L"-" +
        std::to_wstring(std::chrono::steady_clock::now().time_since_epoch().count());
    TestHandle unrelated_handle(CreateEventW(
        &inheritable_security,
        TRUE,
        FALSE,
        unrelated_event_name.c_str()));
    expect(unrelated_handle.get() != nullptr, name, "could not create unrelated test handle");
    const std::wstring unrelated_handle_value =
        std::to_wstring(reinterpret_cast<uintptr_t>(unrelated_handle.get()));
    expect(
        SetEnvironmentVariableW(
            L"AILA_TEST_UNRELATED_HANDLE",
            unrelated_handle_value.c_str()) != FALSE,
        name,
        "could not publish unrelated test handle value");

    TempDirectory temp;
    const fs::path worker = stage_fake_worker(temp);
    const fs::path runtime = normalized_absolute(temp.path());
    const fs::path expected_runtime = long_existing_path(runtime);
    const fs::path expected_worker = long_existing_path(worker);
    aila::runtime::WorkerProcess process;
    process.start(runtime, worker, expected_fake_handshake());

    expect(process.healthy(), name, "worker was not healthy after its handshake");
    const aila::ipc::Frame response =
        process.request(request_frame(10, "test.inspect"), 2s);
    const Inspection inspection = parse_inspection(response.header.payload_json, name);
    const fs::path system_root = aila::runtime::system_root_directory();
    const std::wstring expected_path =
        expected_runtime.wstring() + L";" +
        (system_root / L"System32").lexically_normal().wstring() + L";" +
        system_root.wstring();

    expect(inspection.build_id == "fake-worker-v1", name, "fake build ID changed");
    expect(
        inspection.executable == utf8(expected_worker.wstring()),
        name,
        "worker executable changed: expected '" + utf8(expected_worker.wstring()) +
            "', received '" + inspection.executable + "'");
    expect(
        inspection.runtime_directory == utf8(expected_runtime.wstring()),
        name,
        "worker runtime directory changed");
    expect(
        inspection.current_directory == utf8(expected_runtime.wstring()),
        name,
        "worker current directory was not the runtime directory");
    expect(inspection.path == utf8(expected_path), name, "isolated PATH was not exact");
    expect(
        inspection.path.find("HostPython") == std::string::npos &&
            inspection.path.find("oneAPI") == std::string::npos,
        name,
        "synthetic host toolchain PATH leaked into worker");
    expect(
        inspection.sentinel == utf8(L"retained-插件"),
        name,
        "inherited sentinel was not retained");
    expect(
        WaitForSingleObject(unrelated_handle.get(), 0) == WAIT_TIMEOUT,
        name,
        "STARTUPINFOEX handle list leaked an unrelated inheritable handle");

    process.shutdown(2s);
    const std::vector<aila::ipc::Frame> events = process.take_events();
    expect(events.size() == 1, name, "event pipe did not queue exactly one fake log event");
    expect(events.front().header.kind == "event", name, "event kind changed");
    expect(events.front().header.method == "log", name, "event method changed");
    expect(
        events.front().header.payload_json.find("fake worker ready") != std::string::npos,
        name,
        "event payload changed");
    expect(process.take_events().empty(), name, "take_events did not drain the queue");
}

void test_worker_request_framing_is_repeatable() {
    constexpr const char* name = "worker request framing";
    TempDirectory temp;
    const fs::path worker = stage_fake_worker(temp);
    aila::runtime::WorkerProcess process;
    process.start(temp.path(), worker, expected_fake_handshake());

    const aila::ipc::Frame first_request =
        request_frame(41, "test.ping", "{\"sequence\":1}", bytes({0, 1, 127, 255}));
    const aila::ipc::Frame first = process.request(first_request, 2s);
    expect(first.header.request_id == 41, name, "first response request ID changed");
    expect(first.header.kind == "result", name, "first response kind was not result");
    expect(first.header.method == "test.ping", name, "first response method changed");
    expect(first.header.payload_json == "{\"sequence\":1}", name, "first payload changed");
    expect(first.attachment == first_request.attachment, name, "first attachment changed");

    const aila::ipc::Frame second_request =
        request_frame(42, "test.ping", "{\"sequence\":2}", bytes({9, 8, 7}));
    const aila::ipc::Frame second = process.request(second_request, 2s);
    expect(second.header.request_id == 42, name, "second response request ID changed");
    expect(second.header.kind == "result", name, "second response kind was not result");
    expect(second.header.payload_json == "{\"sequence\":2}", name, "second payload changed");
    expect(second.attachment == second_request.attachment, name, "second attachment changed");
    expect(process.cancel(99, 2s), name, "cancel command was not acknowledged");
    process.shutdown(2s);
}

void test_worker_stream_preserves_unrelated_log_events() {
    constexpr const char* name = "worker stream preserves unrelated logs";
    TempDirectory temp;
    const fs::path worker = stage_fake_worker(temp);
    aila::runtime::WorkerProcess process;
    process.start(temp.path(), worker, expected_fake_handshake());
    aila::ipc::Frame initialized;
    try {
        initialized = process.request(
            request_frame(51, "engine.init", R"({"model":"m","maxSeqLen":1024})"), 2s);
    } catch (const std::exception& exception) {
        fail(name, std::string("engine init transport failed: ") + exception.what());
    }
    expect(initialized.header.kind == "result", name, "fake engine init failed");

    size_t stream_events = 0;
    const aila::ipc::Frame response = process.request_stream(
        request_frame(52, "generate.stream", R"({"input":"x","config":null})"),
        [&](const aila::ipc::Frame& event) {
            ++stream_events;
            if (event.header.payload_json.find("\"event\":\"end\"") != std::string::npos) {
                return aila::runtime::WorkerProcess::StreamEventAction::End;
            }
            return aila::runtime::WorkerProcess::StreamEventAction::Continue;
        },
        2s);
    expect(response.header.kind == "result", name, "stream request failed");
    expect(stream_events == 3, name, "stream event count changed");
    const auto unrelated = process.take_events();
    expect(unrelated.size() == 1 && unrelated.front().header.method == "log" &&
               unrelated.front().header.request_id == 0,
           name,
           "unrelated ready log was consumed by active stream");
    process.shutdown(2s);
}

void test_event_transport_rejects_invalid_protocol_and_abi() {
    constexpr const char* name = "event transport rejects invalid protocol and ABI";
    for (const char* method : {"test.invalid-event-protocol", "test.invalid-event-abi"}) {
        TempDirectory temp;
        const fs::path worker = stage_fake_worker(temp);
        aila::runtime::WorkerProcess process;
        process.start(temp.path(), worker, expected_fake_handshake());
        const auto started = std::chrono::steady_clock::now();
        const std::string error = expect_runtime_error(
            [&] { (void)process.request(request_frame(61, method), 2s); }, name);
        expect(std::chrono::steady_clock::now() - started < 1s,
               name,
               "invalid event did not fail transport promptly");
        expect(error.find("event") != std::string::npos,
               name,
               "invalid event error was not actionable: " + error);
        for (const auto& queued : process.take_events()) {
            expect(queued.header.protocol == aila::ipc::kProtocolVersion &&
                       queued.header.abi == aila::ipc::kPublicAbiVersion,
                   name,
                   "invalid event escaped through take_events");
        }
        expect(!process.healthy(), name, "invalid event left worker healthy");
        process.shutdown(500ms);
    }
}

void test_worker_shutdown_is_bounded_idempotent_and_leak_free() {
    constexpr const char* name = "worker graceful shutdown";
    const DWORD handles_before = process_handle_count();
    {
        TempDirectory temp;
        const fs::path worker = stage_fake_worker(temp);
        aila::runtime::WorkerProcess process;
        process.start(temp.path(), worker, expected_fake_handshake());
        const auto started = std::chrono::steady_clock::now();
        process.shutdown(2s);
        expect(
            std::chrono::steady_clock::now() - started < 3s,
            name,
            "explicit shutdown exceeded its bound");
        process.shutdown(2s);
        expect(process.exit_code() == std::optional<DWORD>{0}, name, "exit code was not zero");
    }
    expect(
        process_handle_count() == handles_before,
        name,
        "explicit shutdown leaked a process, thread, or pipe handle");

    const DWORD destructor_handles_before = process_handle_count();
    const auto started = std::chrono::steady_clock::now();
    {
        TempDirectory temp;
        const fs::path worker = stage_fake_worker(temp);
        aila::runtime::WorkerProcess process;
        process.start(temp.path(), worker, expected_fake_handshake());
    }
    expect(
        std::chrono::steady_clock::now() - started < 4s,
        name,
        "destructor shutdown exceeded its bound");
    expect(
        process_handle_count() == destructor_handles_before,
        name,
        "destructor leaked a process, thread, or pipe handle");
}

void test_worker_crash_fails_request_without_hanging() {
    constexpr const char* name = "worker crash request failure";
    TempDirectory temp;
    const fs::path worker = stage_fake_worker(temp);
    aila::runtime::WorkerProcess process;
    process.start(temp.path(), worker, expected_fake_handshake());

    const auto started = std::chrono::steady_clock::now();
    const std::string error = expect_runtime_error(
        [&] {
            (void)process.request(
                request_frame(77, "test.exit", "{\"code\":23}"),
                2s);
        },
        name);
    expect(
        std::chrono::steady_clock::now() - started < 3s,
        name,
        "crashed worker request exceeded its deadline");
    expect(!process.healthy(), name, "crashed worker still reported healthy");
    expect(process.exit_code() == std::optional<DWORD>{23}, name, "crash exit code changed");
    expect(error.find("23") != std::string::npos, name, "request error omitted exit code 23");
    expect(
        process.last_error().find("23") != std::string::npos,
        name,
        "last_error omitted exit code 23");
    process.shutdown(500ms);
}

void test_worker_partial_response_obeys_request_deadline() {
    constexpr const char* name = "worker partial response deadline";
    TempDirectory temp;
    const fs::path worker = stage_fake_worker(temp);
    aila::runtime::WorkerProcess process;
    process.start(temp.path(), worker, expected_fake_handshake());

    const auto started = std::chrono::steady_clock::now();
    const std::string error = expect_runtime_error(
        [&] {
            (void)process.request(
                request_frame(78, "test.partial-response"),
                200ms);
        },
        name);
    const auto elapsed = std::chrono::steady_clock::now() - started;
    expect(elapsed < 2s, name, "partial response exceeded the request deadline bound");
    expect(error.find("timed out") != std::string::npos, name, "error omitted timeout detail");
    expect(!process.healthy(), name, "timed-out partial-response worker was not reaped");
    expect(process.exit_code().has_value(), name, "reaped worker exit code was not retained");
    process.shutdown(500ms);
}

void test_worker_blocked_write_obeys_request_deadline() {
    constexpr const char* name = "worker blocked write deadline";
    EnvironmentRestore mode_restore(L"AILA_FAKE_WORKER_MODE");
    expect(
        SetEnvironmentVariableW(L"AILA_FAKE_WORKER_MODE", L"stop-reading") != FALSE,
        name,
        "could not configure fake worker mode");
    TempDirectory temp;
    const fs::path worker = stage_fake_worker(temp);
    aila::runtime::WorkerProcess process;
    process.start(temp.path(), worker, expected_fake_handshake());

    std::vector<std::byte> attachment(8 * 1024 * 1024, std::byte{0x5a});
    const auto started = std::chrono::steady_clock::now();
    const std::string error = expect_runtime_error(
        [&] {
            (void)process.request(
                request_frame(79, "test.ping", "{}", std::move(attachment)),
                200ms);
        },
        name);
    const auto elapsed = std::chrono::steady_clock::now() - started;
    expect(elapsed < 2s, name, "blocked request write exceeded the deadline bound");
    expect(error.find("timed out") != std::string::npos, name, "error omitted timeout detail");
    expect(!process.healthy(), name, "timed-out blocked-write worker was not reaped");
    expect(process.exit_code().has_value(), name, "reaped worker exit code was not retained");
    process.shutdown(500ms);
}

void test_worker_partial_handshake_obeys_start_deadline() {
    constexpr const char* name = "worker partial handshake deadline";
    EnvironmentRestore mode_restore(L"AILA_FAKE_WORKER_MODE");
    expect(
        SetEnvironmentVariableW(L"AILA_FAKE_WORKER_MODE", L"partial-handshake") != FALSE,
        name,
        "could not configure fake worker mode");
    TempDirectory temp;
    const fs::path worker = stage_fake_worker(temp);
    aila::runtime::WorkerProcess process;

    const auto started = std::chrono::steady_clock::now();
    const std::string error = expect_runtime_error(
        [&] {
            process.start(
                temp.path(),
                worker,
                expected_fake_handshake(),
                200ms);
        },
        name);
    const auto elapsed = std::chrono::steady_clock::now() - started;
    expect(elapsed < 2s, name, "partial handshake exceeded the start deadline bound");
    expect(error.find("timed out") != std::string::npos, name, "error omitted timeout detail");
    expect(!process.healthy(), name, "timed-out handshake worker was not reaped");
    expect(process.exit_code().has_value(), name, "reaped worker exit code was not retained");
    process.shutdown(500ms);
}

void test_worker_executable_reparse_points_are_rejected() {
    constexpr const char* name = "worker executable reparse containment";
    expect(
        aila::runtime::detail::worker_file_attributes_are_safe(FILE_ATTRIBUTE_ARCHIVE),
        name,
        "ordinary worker file attributes were rejected");
    expect(
        !aila::runtime::detail::worker_file_attributes_are_safe(
            FILE_ATTRIBUTE_ARCHIVE | FILE_ATTRIBUTE_REPARSE_POINT),
        name,
        "reparse-point worker attributes were accepted");
    expect(
        !aila::runtime::detail::worker_file_attributes_are_safe(FILE_ATTRIBUTE_DIRECTORY),
        name,
        "directory worker attributes were accepted");
    expect(
        !aila::runtime::detail::worker_file_attributes_are_safe(INVALID_FILE_ATTRIBUTES),
        name,
        "invalid worker attributes were accepted");
    expect(
        aila::runtime::detail::runtime_component_attributes_are_safe(
            FILE_ATTRIBUTE_DIRECTORY),
        name,
        "ordinary runtime directory attributes were rejected");
    expect(
        !aila::runtime::detail::runtime_component_attributes_are_safe(
            FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT),
        name,
        "reparse runtime component attributes were accepted");
    expect(
        !aila::runtime::detail::runtime_component_attributes_are_safe(FILE_ATTRIBUTE_ARCHIVE),
        name,
        "non-directory runtime component attributes were accepted");

    TempDirectory temp;
    const fs::path link = temp.path() / L"AilaWorker.exe";
    const fs::path target = fs::absolute(fs::path(AILA_FAKE_WORKER_PATH)).lexically_normal();
    DWORD flags = SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE;
    BOOL created = CreateSymbolicLinkW(link.c_str(), target.c_str(), flags);
    if (created == FALSE && GetLastError() == ERROR_INVALID_PARAMETER) {
        created = CreateSymbolicLinkW(link.c_str(), target.c_str(), 0);
    }
    if (created != FALSE) {
        aila::runtime::WorkerProcess process;
        const std::string error = expect_runtime_error(
            [&] { process.start(temp.path(), link, expected_fake_handshake()); },
            name);
        expect(
            error.find("reparse") != std::string::npos,
            name,
            "reparse rejection diagnostic was not actionable: " + error);
    }

    const fs::path real_runtime = temp.path() / L"real runtime";
    fs::create_directories(real_runtime);
    fs::copy_file(
        target,
        real_runtime / L"AilaWorker.exe",
        fs::copy_options::overwrite_existing);
    const fs::path runtime_link = temp.path() / L"runtime link";
    flags = SYMBOLIC_LINK_FLAG_DIRECTORY | SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE;
    created = CreateSymbolicLinkW(runtime_link.c_str(), real_runtime.c_str(), flags);
    if (created == FALSE && GetLastError() == ERROR_INVALID_PARAMETER) {
        created = CreateSymbolicLinkW(
            runtime_link.c_str(),
            real_runtime.c_str(),
            SYMBOLIC_LINK_FLAG_DIRECTORY);
    }
    if (created != FALSE) {
        aila::runtime::WorkerProcess process;
        const std::string error = expect_runtime_error(
            [&] {
                process.start(
                    runtime_link,
                    runtime_link / L"AilaWorker.exe",
                    expected_fake_handshake());
            },
            name);
        expect(
            error.find("reparse") != std::string::npos,
            name,
            "runtime ancestor reparse diagnostic was not actionable: " + error);
    }
}

void test_worker_handshake_mismatch_reaps_process() {
    constexpr const char* name = "worker handshake mismatch";
    TempDirectory temp;
    const fs::path worker = stage_fake_worker(temp);
    aila::runtime::WorkerProcess process;
    const auto started = std::chrono::steady_clock::now();
    const std::string error = expect_runtime_error(
        [&] {
            process.start(
                temp.path(),
                worker,
                expected_fake_handshake("expected-production-build"));
        },
        name);
    expect(
        std::chrono::steady_clock::now() - started < 3s,
        name,
        "handshake mismatch cleanup exceeded its bound");
    expect(!process.healthy(), name, "mismatched worker was left healthy");
    expect(
        error.find("build ID") != std::string::npos &&
            error.find("expected-production-build") != std::string::npos &&
            error.find("fake-worker-v1") != std::string::npos,
        name,
        "handshake mismatch diagnostic was not actionable: " + error);
    process.shutdown(500ms);
}

} // namespace

int main() {
    try {
        test_runtime_directory_resolution();
        test_worker_validation();
        test_runtime_directory_override_reads_unicode();
        test_isolated_environment_replaces_path();
        test_hidden_drive_environment_entries_are_preserved();
        test_isolated_environment_validates_entries();
        test_current_environment_and_system_root();
        test_proxy_module_path();
        test_worker_start_isolates_process_context_and_queues_events();
        test_worker_request_framing_is_repeatable();
        test_worker_stream_preserves_unrelated_log_events();
        test_event_transport_rejects_invalid_protocol_and_abi();
        test_worker_shutdown_is_bounded_idempotent_and_leak_free();
        test_worker_crash_fails_request_without_hanging();
        test_worker_blocked_write_obeys_request_deadline();
        test_worker_partial_response_obeys_request_deadline();
        test_worker_partial_handshake_obeys_start_deadline();
        test_worker_executable_reparse_points_are_rejected();
        test_worker_handshake_mismatch_reaps_process();
        std::cout << "AilaRuntimeIsolationTests passed\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
