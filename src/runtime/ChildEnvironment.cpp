#include "runtime/ChildEnvironment.hpp"

#include <windows.h>

#include <algorithm>
#include <climits>
#include <limits>
#include <memory>
#include <stdexcept>
#include <system_error>

namespace aila::runtime {
namespace {

namespace fs = std::filesystem;

constexpr size_t kMaximumEnvironmentBlockCharacters = 32767;

struct EnvironmentStringsDeleter {
    void operator()(wchar_t* strings) const noexcept {
        if (strings != nullptr) {
            FreeEnvironmentStringsW(strings);
        }
    }
};

using EnvironmentStrings = std::unique_ptr<wchar_t, EnvironmentStringsDeleter>;

void validate_entry(const std::wstring& name, const std::wstring& value) {
    if (name.empty()) {
        throw std::runtime_error("child environment contains an empty variable name");
    }
    if (name.find(L'=') != std::wstring::npos) {
        throw std::runtime_error("child environment variable name contains '='");
    }
    if (name.find(L'\0') != std::wstring::npos) {
        throw std::runtime_error("child environment variable name contains an embedded NUL");
    }
    if (value.find(L'\0') != std::wstring::npos) {
        throw std::runtime_error("child environment variable value contains an embedded NUL");
    }
}

fs::path absolute_normalized(const fs::path& path) {
    return fs::absolute(path).lexically_normal();
}

} // namespace

bool CaseInsensitiveLess::operator()(
    const std::wstring& left,
    const std::wstring& right) const noexcept {
    if (left.size() <= static_cast<size_t>(INT_MAX) &&
        right.size() <= static_cast<size_t>(INT_MAX)) {
        const int result = CompareStringOrdinal(
            left.data(),
            static_cast<int>(left.size()),
            right.data(),
            static_cast<int>(right.size()),
            TRUE);
        if (result != 0) {
            return result == CSTR_LESS_THAN;
        }
    }

    const size_t shared_size = (std::min)(left.size(), right.size());
    for (size_t index = 0; index < shared_size; ++index) {
        wchar_t left_character = left[index];
        wchar_t right_character = right[index];
        CharUpperBuffW(&left_character, 1);
        CharUpperBuffW(&right_character, 1);
        if (left_character != right_character) {
            return left_character < right_character;
        }
    }
    return left.size() < right.size();
}

EnvironmentMap current_environment() {
    EnvironmentStrings strings(GetEnvironmentStringsW());
    if (!strings) {
        throw std::system_error(
            static_cast<int>(GetLastError()),
            std::system_category(),
            "GetEnvironmentStringsW failed");
    }

    EnvironmentMap result;
    const wchar_t* entry = strings.get();
    while (*entry != L'\0') {
        const std::wstring value(entry);
        entry += value.size() + 1;

        if (value.empty() || value.front() == L'=') {
            continue;
        }
        const size_t equals = value.find(L'=');
        if (equals == std::wstring::npos || equals == 0) {
            continue;
        }
        result.insert_or_assign(value.substr(0, equals), value.substr(equals + 1));
    }
    return result;
}

fs::path system_root_directory() {
    std::vector<wchar_t> buffer(MAX_PATH);
    for (;;) {
        const UINT copied = GetWindowsDirectoryW(buffer.data(), static_cast<UINT>(buffer.size()));
        if (copied == 0) {
            throw std::system_error(
                static_cast<int>(GetLastError()),
                std::system_category(),
                "GetWindowsDirectoryW failed");
        }
        if (copied < buffer.size()) {
            return absolute_normalized(fs::path(std::wstring(buffer.data(), copied)));
        }
        if (copied == (std::numeric_limits<UINT>::max)()) {
            throw std::runtime_error("Windows directory path is too long");
        }
        buffer.resize(static_cast<size_t>(copied) + 1);
    }
}

std::vector<wchar_t> build_isolated_environment(
    const EnvironmentMap& inherited,
    const fs::path& runtime_directory,
    const fs::path& system_root) {
    EnvironmentMap isolated = inherited;
    for (const auto& [name, value] : isolated) {
        validate_entry(name, value);
    }

    const fs::path normalized_runtime = absolute_normalized(runtime_directory);
    const fs::path normalized_system_root = absolute_normalized(system_root);
    const std::wstring path =
        normalized_runtime.wstring() + L";" +
        (normalized_system_root / L"System32").lexically_normal().wstring() + L";" +
        normalized_system_root.wstring();
    validate_entry(L"PATH", path);
    isolated.erase(L"PATH");
    isolated.emplace(L"PATH", path);

    size_t characters = 1;
    for (const auto& [name, value] : isolated) {
        const size_t entry_size = name.size() + 1 + value.size() + 1;
        if (entry_size > kMaximumEnvironmentBlockCharacters - characters) {
            throw std::runtime_error("child environment exceeds the Windows 32767-character limit");
        }
        characters += entry_size;
    }

    std::vector<wchar_t> block;
    block.reserve(characters);
    for (const auto& [name, value] : isolated) {
        block.insert(block.end(), name.begin(), name.end());
        block.push_back(L'=');
        block.insert(block.end(), value.begin(), value.end());
        block.push_back(L'\0');
    }
    block.push_back(L'\0');
    if (block.size() == 1) {
        block.push_back(L'\0');
    }
    return block;
}

} // namespace aila::runtime
