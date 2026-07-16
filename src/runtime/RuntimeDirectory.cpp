#include "runtime/RuntimeDirectory.hpp"

#include <limits>
#include <stdexcept>
#include <system_error>
#include <vector>

namespace aila::runtime {
namespace {

namespace fs = std::filesystem;

constexpr wchar_t kRuntimeDirectoryVariable[] = L"AILA_RUNTIME_DLL_DIR";
constexpr wchar_t kWorkerExecutable[] = L"AilaWorker.exe";

std::string utf8(const std::wstring& value) {
    if (value.empty()) {
        return {};
    }
    if (value.size() > static_cast<size_t>((std::numeric_limits<int>::max)())) {
        throw std::runtime_error("path is too long to report");
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
        throw std::runtime_error("could not encode a runtime path as UTF-8");
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
        throw std::runtime_error("could not encode a runtime path as UTF-8");
    }
    return result;
}

fs::path absolute_normalized(const fs::path& path) {
    return fs::absolute(path).lexically_normal();
}

std::runtime_error invalid_runtime_directory(
    const fs::path& runtime_directory,
    const std::string& detail) {
    return std::runtime_error(
        "AILA_RUNTIME_DLL_DIR resolved to '" + utf8(runtime_directory.wstring()) +
        "': " + detail);
}

} // namespace

fs::path proxy_module_path(HMODULE module) {
    std::vector<wchar_t> buffer(MAX_PATH);
    for (;;) {
        SetLastError(ERROR_SUCCESS);
        const DWORD copied =
            GetModuleFileNameW(module, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (copied == 0) {
            throw std::system_error(
                static_cast<int>(GetLastError()),
                std::system_category(),
                "GetModuleFileNameW failed");
        }
        if (copied < buffer.size()) {
            return absolute_normalized(fs::path(std::wstring(buffer.data(), copied)));
        }
        if (buffer.size() > (std::numeric_limits<DWORD>::max)() / 2) {
            throw std::runtime_error("proxy module path is too long");
        }
        buffer.resize(buffer.size() * 2);
    }
}

std::optional<std::wstring> runtime_directory_override() {
    DWORD capacity = GetEnvironmentVariableW(kRuntimeDirectoryVariable, nullptr, 0);
    if (capacity == 0) {
        const DWORD error = GetLastError();
        if (error == ERROR_SUCCESS || error == ERROR_ENVVAR_NOT_FOUND) {
            return std::nullopt;
        }
        throw std::system_error(
            static_cast<int>(error),
            std::system_category(),
            "GetEnvironmentVariableW(AILA_RUNTIME_DLL_DIR) failed");
    }

    for (;;) {
        std::vector<wchar_t> buffer(capacity);
        SetLastError(ERROR_SUCCESS);
        const DWORD copied = GetEnvironmentVariableW(
            kRuntimeDirectoryVariable,
            buffer.data(),
            static_cast<DWORD>(buffer.size()));
        if (copied == 0) {
            const DWORD error = GetLastError();
            if (error == ERROR_SUCCESS || error == ERROR_ENVVAR_NOT_FOUND) {
                return std::nullopt;
            }
            throw std::system_error(
                static_cast<int>(error),
                std::system_category(),
                "GetEnvironmentVariableW(AILA_RUNTIME_DLL_DIR) failed");
        }
        if (copied < buffer.size()) {
            return std::wstring(buffer.data(), copied);
        }
        capacity = copied;
    }
}

fs::path resolve_runtime_directory(
    const fs::path& proxy_module,
    const std::optional<std::wstring>& configured_value) {
    const fs::path absolute_proxy = absolute_normalized(proxy_module);
    if (!configured_value || configured_value->empty()) {
        return absolute_proxy.parent_path().lexically_normal();
    }

    const fs::path configured(*configured_value);
    if (configured.is_absolute()) {
        return absolute_normalized(configured);
    }
    return absolute_normalized(absolute_proxy.parent_path() / configured);
}

fs::path require_worker_executable(const fs::path& runtime_directory) {
    const fs::path normalized_directory = absolute_normalized(runtime_directory);
    std::error_code error;
    const fs::file_status directory_status = fs::status(normalized_directory, error);
    if (error || !fs::exists(directory_status)) {
        throw invalid_runtime_directory(normalized_directory, "runtime directory does not exist");
    }
    if (!fs::is_directory(directory_status)) {
        throw invalid_runtime_directory(normalized_directory, "candidate is not a directory");
    }

    const fs::path worker = (normalized_directory / kWorkerExecutable).lexically_normal();
    error.clear();
    const fs::file_status worker_status = fs::status(worker, error);
    if (error || !fs::is_regular_file(worker_status)) {
        throw invalid_runtime_directory(
            normalized_directory,
            "expected a regular file at '" + utf8(worker.wstring()) + "'");
    }
    return worker;
}

} // namespace aila::runtime
