#include <windows.h>

#include "runtime/ChildEnvironment.hpp"
#include "runtime/RuntimeDirectory.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;

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

void create_empty_file(const fs::path& path) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("could not create test file");
    }
}

fs::path normalized_absolute(const fs::path& path) {
    return fs::absolute(path).lexically_normal();
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
        L"AILA_LOG_LEVEL=debug",
        L"Path=C:\\HostPython",
    });
    const aila::runtime::EnvironmentMap inherited =
        aila::runtime::parse_environment_block(source);

    expect(inherited.size() == 3, name, "synthetic environment entry count changed");
    expect(inherited.at(L"=d:") == L"D:\\work", name, "hidden drive entry was not parsed");
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
        std::cout << "AilaRuntimeIsolationTests passed\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
