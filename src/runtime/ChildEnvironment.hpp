#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace aila::runtime {

struct CaseInsensitiveLess {
    bool operator()(const std::wstring& left, const std::wstring& right) const noexcept;
};

using EnvironmentMap = std::map<std::wstring, std::wstring, CaseInsensitiveLess>;

EnvironmentMap parse_environment_block(const std::vector<wchar_t>& block);

EnvironmentMap current_environment();

std::filesystem::path system_root_directory();

std::vector<wchar_t> build_isolated_environment(
    const EnvironmentMap& inherited,
    const std::filesystem::path& runtime_directory,
    const std::filesystem::path& system_root);

} // namespace aila::runtime
