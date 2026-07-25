#include "runtime/ChildEnvironment.hpp"

#include <windows.h>

#include <algorithm>
#include <climits>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string_view>
#include <system_error>

namespace aila::runtime {
namespace {

namespace fs = std::filesystem;

constexpr size_t kMaximumEnvironmentBlockCharacters = 32767;

constexpr wchar_t kPassthroughVariable[] = L"AILA_WORKER_ENV_PASSTHROUGH";

// GPU hosts (torch.xpu, IPEX, profilers) install these instrumentation and
// loader-override variables for their own runtime. Inherited by the worker,
// they reconfigure the private oneAPI runtime and can load foreign DLLs into
// the isolated process (XPTI_SUBSCRIBERS, UR_ENABLE_LAYERS, OCL_ICD_FILENAMES),
// so the child environment drops them unless explicitly passed through.
constexpr std::wstring_view kScrubbedRuntimePrefixes[] = {
    L"XPTI_",                  // XPTI trace framework: dispatcher/subscriber DLL loading
    L"UR_",                    // oneAPI Unified Runtime loader: layers, forced adapters
    L"ZES_",                   // Level Zero sysman instrumentation
    L"ZET_",                   // Level Zero tools/metrics instrumentation
    L"ZE_ENABLE_",             // Level Zero loader layer toggles (tracing/validation/alt drivers)
    L"OCL_ICD_",               // OpenCL ICD loader overrides
    L"__KMP_REGISTERED_LIB_",  // host-process OpenMP registration bookkeeping
};

bool starts_with_case_insensitively(const std::wstring& value, std::wstring_view prefix) {
    if (value.size() < prefix.size() ||
        prefix.size() > static_cast<size_t>(INT_MAX)) {
        return false;
    }
    return CompareStringOrdinal(
               value.data(),
               static_cast<int>(prefix.size()),
               prefix.data(),
               static_cast<int>(prefix.size()),
               TRUE) == CSTR_EQUAL;
}

std::set<std::wstring, CaseInsensitiveLess> parse_passthrough_names(
    const EnvironmentMap& inherited) {
    std::set<std::wstring, CaseInsensitiveLess> names;
    const auto configured = inherited.find(kPassthroughVariable);
    if (configured == inherited.end()) {
        return names;
    }
    const std::wstring& value = configured->second;
    size_t offset = 0;
    while (offset <= value.size()) {
        size_t separator = value.find(L';', offset);
        if (separator == std::wstring::npos) {
            separator = value.size();
        }
        size_t begin = offset;
        size_t end = separator;
        while (begin < end && (value[begin] == L' ' || value[begin] == L'\t')) {
            ++begin;
        }
        while (end > begin && (value[end - 1] == L' ' || value[end - 1] == L'\t')) {
            --end;
        }
        if (end > begin) {
            names.emplace(value.substr(begin, end - begin));
        }
        offset = separator + 1;
    }
    return names;
}

struct EnvironmentStringsDeleter {
    void operator()(wchar_t* strings) const noexcept {
        if (strings != nullptr) {
            FreeEnvironmentStringsW(strings);
        }
    }
};

using EnvironmentStrings = std::unique_ptr<wchar_t, EnvironmentStringsDeleter>;

bool is_hidden_drive_name(const std::wstring& name) {
    return name.size() == 3 && name[0] == L'=' &&
        ((name[1] >= L'A' && name[1] <= L'Z') ||
         (name[1] >= L'a' && name[1] <= L'z')) &&
        name[2] == L':';
}

void validate_entry(const std::wstring& name, const std::wstring& value) {
    if (name.empty()) {
        throw std::runtime_error("child environment contains an empty variable name");
    }
    if (name.find(L'\0') != std::wstring::npos) {
        throw std::runtime_error("child environment variable name contains an embedded NUL");
    }
    if (name.front() == L'=') {
        if (!is_hidden_drive_name(name)) {
            throw std::runtime_error(
                "hidden drive environment name must have the form =<ASCII drive letter>:");
        }
    } else if (name.find(L'=') != std::wstring::npos) {
        throw std::runtime_error("child environment variable name contains '='");
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

EnvironmentMap parse_environment_block(const std::vector<wchar_t>& block) {
    if (block.size() < 2 || block[block.size() - 1] != L'\0' ||
        block[block.size() - 2] != L'\0') {
        throw std::runtime_error("environment block is not double-NUL terminated");
    }
    EnvironmentMap result;
    size_t offset = 0;
    while (offset + 1 < block.size() && block[offset] != L'\0') {
        const auto terminator =
            std::find(block.begin() + static_cast<std::ptrdiff_t>(offset), block.end(), L'\0');
        if (terminator == block.end()) {
            throw std::runtime_error("environment block entry is not NUL terminated");
        }
        const std::wstring entry(
            block.begin() + static_cast<std::ptrdiff_t>(offset),
            terminator);
        const size_t separator =
            entry.front() == L'=' ? entry.find(L'=', 1) : entry.find(L'=');
        if (separator == std::wstring::npos) {
            throw std::runtime_error("environment block entry has no value separator");
        }
        const std::wstring name = entry.substr(0, separator);
        const std::wstring value = entry.substr(separator + 1);
        // cmd.exe can add private '='-prefixed entries that are not per-drive
        // current directories; keep omitting those legacy pseudo variables.
        if (name.front() == L'=' && !is_hidden_drive_name(name)) {
            offset += entry.size() + 1;
            continue;
        }
        validate_entry(name, value);
        result.insert_or_assign(name, value);
        offset += entry.size() + 1;
    }
    if ((block.size() != 2 || block.front() != L'\0') &&
        offset != block.size() - 1) {
        throw std::runtime_error("environment block contains an early terminator");
    }
    return result;
}

EnvironmentMap current_environment() {
    EnvironmentStrings strings(GetEnvironmentStringsW());
    if (!strings) {
        throw std::system_error(
            static_cast<int>(GetLastError()),
            std::system_category(),
            "GetEnvironmentStringsW failed");
    }

    const wchar_t* begin = strings.get();
    const wchar_t* end = begin;
    while (*end != L'\0') {
        end += std::wstring(end).size() + 1;
    }
    ++end;
    std::vector<wchar_t> block(begin, end);
    if (block.size() == 1) {
        block.push_back(L'\0');
    }
    return parse_environment_block(block);
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

bool is_scrubbed_runtime_variable(const std::wstring& name) noexcept {
    for (const std::wstring_view prefix : kScrubbedRuntimePrefixes) {
        if (starts_with_case_insensitively(name, prefix)) {
            return true;
        }
    }
    return false;
}

std::vector<wchar_t> build_isolated_environment(
    const EnvironmentMap& inherited,
    const fs::path& runtime_directory,
    const fs::path& system_root) {
    EnvironmentMap isolated = inherited;
    for (const auto& [name, value] : isolated) {
        validate_entry(name, value);
    }

    const std::set<std::wstring, CaseInsensitiveLess> passthrough =
        parse_passthrough_names(isolated);
    for (auto entry = isolated.begin(); entry != isolated.end();) {
        if (is_scrubbed_runtime_variable(entry->first) &&
            passthrough.find(entry->first) == passthrough.end()) {
            entry = isolated.erase(entry);
        } else {
            ++entry;
        }
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
